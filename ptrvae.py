import warnings
warnings.filterwarnings("ignore")
import nltk
import scipy.io
from tqdm import tqdm
from scipy.sparse import csr_matrix
from random import shuffle
import pandas as pd
import gzip
import json
import codecs
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
import numpy as np
from tensorflow.losses import compute_weighted_loss, Reduction
from helpers import ndcg_score, mrr, hit


class Model():
    def __init__(self, sample, args):
        self.sample = sample
        self.batchsize = args["batchsize"]

    def _make_embedding(self, vocab_size, embedding_size, name, trainable=True):
        W = tf.Variable(tf.random_uniform(shape=[vocab_size, embedding_size], minval=-1, maxval=1),
                        trainable=trainable, name=name)

        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
        embedding_init = W.assign(embedding_placeholder)
        return (W, embedding_placeholder, embedding_init)

    def _extract(self,sess,  item_emb, user_emb, user_content_matrix, item_content_matrix, batchsize):
        user, i1, i2, iu, i1r, i2r = self.sample[0], self.sample[1], self.sample[2], self.sample[3], self.sample[4], self.sample[5]
        user_content_feature_vector = tf.nn.embedding_lookup(user_content_matrix, user)
        item_content_feature_vector1 = tf.nn.embedding_lookup(item_content_matrix, i1)
        item_content_feature_vector2 = tf.nn.embedding_lookup(item_content_matrix, i2)
        user = tf.nn.embedding_lookup(user_emb, user) 
        i1 = tf.nn.embedding_lookup(item_emb, i1)
        order_gt = self.sample[6]
        return user, i1, i1r, i2, i2r, user_content_feature_vector, item_content_feature_vector1, item_content_feature_vector2, order_gt  

    def _sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def _gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self._sample_gumbel(tf.shape(logits))
        return tf.nn.softmax(y / temperature, axis=-1)

    def gumbel_softmax(self, logits, temperature, hard):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, bits, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, bits, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self._gumbel_softmax_sample(logits, temperature)
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, -1, keepdims=True)), y.dtype)
        y_hard = tf.stop_gradient(y_hard - y) + y
        y = tf.cond(hard, lambda: y_hard, lambda: y)
        return y

    def make_importance_embedding(self, vocab_size, trainable=True):
        W = tf.Variable(tf.random_uniform(shape=[vocab_size], minval=0.1, maxval=1),
                        trainable=trainable, name="importance_embedding")
        return W

    def make_network(self,sess, word_emb_matrix, importance_emb_matrix, user_content_matrix, item_content_matrix, item_emb, user_emb, is_training, args, max_rating, sigma_anneal, sigma_anneal_vae, batchsize, ogt):
        ## ref code: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        def bernoulliSample(x):
            """
            Uses a tensor whose values are in [0,1] to sample a tensor with values in {0, 1},
            using the straight through estimator for the gradient.
            E.g.,:
            if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
            and the gradient will be pass-through (identity).
            """
            g = tf.get_default_graph()

            with ops.name_scope("BernoulliSample") as name:
                with g.gradient_override_map({"Ceil": "Identity", "Sub": "BernoulliSample_ST"}):

                    if args["deterministic_train"]:
                        train_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.ones(tf.shape(x)) * 0.5)
                    else:
                        train_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.random_uniform(tf.shape(x)))

                    if args["deterministic_eval"]:
                        eval_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.ones(tf.shape(x)) * 0.5)
                    else:
                        eval_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.random_uniform(tf.shape(x)))

                    mus = tf.cond(is_training, train_fn, eval_fn)

                    return tf.ceil(x - mus, name=name)

        @ops.RegisterGradient("BernoulliSample_ST")
        def bernoulliSample_ST(op, grad):
            return [grad, tf.zeros(tf.shape(op.inputs[1]))]

        ###########################################################
        def item_encoder(doc, hidden_neurons_encode, encoder_layers):
            doc_layer = tf.layers.dense(doc, hidden_neurons_encode, name="item_encode_layer0", reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
            # doc_layer = tf.nn.dropout(doc_layer, dropout_keep)

            for i in range(1, encoder_layers):
                doc_layer = tf.layers.dense(doc_layer, hidden_neurons_encode, name="item_encode_layer" + str(i),
                                            reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

            doc_layer = tf.nn.dropout(doc_layer, tf.cond(is_training, lambda: 0.8, lambda: 1.0))

            doc_layer = tf.layers.dense(doc_layer,  args["bits"], name="item_last_encode", reuse=tf.AUTO_REUSE,
                                        activation=tf.nn.sigmoid)

            bit_vector = bernoulliSample(doc_layer)

            return bit_vector, doc_layer

        def user_encoder(doc, hidden_neurons_encode, encoder_layers):
            doc_layer = tf.layers.dense(doc, hidden_neurons_encode, name="user_encode_layer0",
                                        reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
            # doc_layer = tf.nn.dropout(doc_layer, dropout_keep)

            for i in range(1, encoder_layers):
                doc_layer = tf.layers.dense(doc_layer, hidden_neurons_encode, name="user_encode_layer" + str(i),
                                            reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

            doc_layer = tf.nn.dropout(doc_layer, tf.cond(is_training, lambda: 0.8, lambda: 1.0))

            doc_layer = tf.layers.dense(doc_layer,  args["bits"], name="user_last_encode", reuse=tf.AUTO_REUSE,
                                        activation=tf.nn.sigmoid)

            bit_vector = bernoulliSample(doc_layer)

            return bit_vector, doc_layer

        def user_decoder(doc, hidden_neurons_encode, encoder_layers, decoder_dim):
            doc_layer = tf.layers.dense(doc, hidden_neurons_encode, name="user_decode_layer0",
                                        reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
            # doc_layer = tf.nn.dropout(doc_layer, dropout_keep)

            for i in range(1, encoder_layers):
                doc_layer = tf.layers.dense(doc_layer, hidden_neurons_encode, name="user_decode_layer" + str(i),
                                            reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

            doc_layer = tf.nn.dropout(doc_layer, tf.cond(is_training, lambda: 0.8, lambda: 1.0))

            doc_layer = tf.layers.dense(doc_layer,  decoder_dim, name="user_last_decode", reuse=tf.AUTO_REUSE,
                                        activation=tf.nn.sigmoid)


            return doc_layer

        def item_decoder(doc, hidden_neurons_encode, encoder_layers, decoder_dim):
            doc_layer = tf.layers.dense(doc, hidden_neurons_encode, name="item_decode_layer0",
                                        reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
            # doc_layer = tf.nn.dropout(doc_layer, dropout_keep)

            for i in range(1, encoder_layers):
                doc_layer = tf.layers.dense(doc_layer, hidden_neurons_encode, name="item_decode_layer" + str(i),
                                            reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

            doc_layer = tf.nn.dropout(doc_layer, tf.cond(is_training, lambda: 0.8, lambda: 1.0))

            doc_layer = tf.layers.dense(doc_layer,  decoder_dim, name="item_last_decode", reuse=tf.AUTO_REUSE,
                                        activation=tf.nn.sigmoid)


            return doc_layer
        user, i1, i1r, i2, i2r, user_content, i1_content, i2_content, ogt = self._extract(sess, item_emb, user_emb, user_content_matrix, item_content_matrix, batchsize)
        i1_content = tf.reshape(i1_content, [batchsize, 512])
        i2_content = tf.reshape(i2_content, [batchsize, 512])
        user_content = tf.reshape(user_content, [batchsize, 512])
        
        #i1_content = tf.cond(is_training, lambda: tf.zeros_like(i1_content), lambda: tf.zeros_like(i1_content))
        #user_content = tf.cond(is_training, lambda: tf.zeros_like(user_content), lambda: tf.zeros_like(user_content))

        i1_content_hashcode, i1_content_cont = item_encoder(i1_content, args["vae_units"], args["vae_layers"])
        e = tf.random.normal([batchsize, args["bits"]])
        i1_noisy_content_hashcode = tf.math.multiply(e, sigma_anneal_vae) + i1_content_hashcode

        i2_content_hashcode, i2_content_cont = item_encoder(i2_content, args["vae_units"], args["vae_layers"])
        e = tf.random.normal([batchsize, args["bits"]])
        i2_noisy_content_hashcode = tf.math.multiply(e, sigma_anneal_vae) + i2_content_hashcode

        user_content_hashcode, user_content_cont = user_encoder(user_content, args["vae_units"], args["vae_layers"])
        e = tf.random.normal([batchsize, args["bits"]])
        user_noisy_content_hashcode = tf.math.multiply(e, sigma_anneal_vae) + user_content_hashcode

        ptrinp = tf.stack([tf.math.multiply(user_content_hashcode, i1_content_hashcode), tf.math.multiply(user_content_hashcode, i2_content_hashcode)])
        ptrinp = tf.reshape(ptrinp, [args["batchsize"], 2, args["bits"]])

        encoder_lstmcell = tf.nn.rnn_cell.BasicLSTMCell(args["bits"], reuse=tf.AUTO_REUSE)
        decoder_lstmcell1 = tf.nn.rnn_cell.BasicLSTMCell(args["bits"], reuse=tf.AUTO_REUSE)
        decoder_lstmcell2 = tf.nn.rnn_cell.BasicLSTMCell(args["bits"], reuse=tf.AUTO_REUSE)

        (e_output, e_state) = tf.nn.dynamic_rnn(cell = encoder_lstmcell, inputs = ptrinp, dtype=tf.float32)
        (d1_output, d1_state) = tf.nn.dynamic_rnn(cell = decoder_lstmcell1, inputs = e_output, dtype=tf.float32)
        (d2_output, d2_state) = tf.nn.dynamic_rnn(cell = decoder_lstmcell2, inputs = d1_output, initial_state = d1_state,  dtype=tf.float32)
        
        W1 = tf.Variable(tf.random.normal(shape=[args["bits"], 1]), trainable=True)
        W2 = tf.Variable(tf.random.normal(shape=[args["bits"], 1]), trainable=True)

        d1h,_ = d1_state
        d2h,_ = d2_state

        d1h = tf.cast(tf.reshape(d1h, [args["batchsize"], args["bits"]]), dtype=tf.float32)
        d2h = tf.cast(tf.reshape(d2h, [args["batchsize"], args["bits"]]), dtype=tf.float32)

        item1 = tf.matmul(tf.math.multiply(user_content_hashcode, i1_content_hashcode), W2)
        item1 = tf.cast(item1, tf.float32)

        item2 = tf.matmul(tf.math.multiply(user_content_hashcode, i2_content_hashcode), W2)
        item2 = tf.cast(item2, tf.float32)
        
        i1_pred = tf.math.softmax(tf.stack([tf.matmul(d1h, W1) + item1, tf.matmul(d1h, W1) + item2], axis=-1))
        i2_pred = tf.math.softmax(tf.stack([tf.matmul(d2h, W1) + item1, tf.matmul(d2h, W1) + item2], axis=-1))
        
        ogt = tf.reshape(ogt, [args["batchsize"],2])
        ogt_positions = ogt
        ogt = tf.one_hot(ogt, depth=2)
        ogt1, ogt2 = tf.split(ogt, num_or_size_splits=2, axis=-1)
        ogt1 = tf.reshape(ogt1, [args["batchsize"],2])
        ogt2 = tf.reshape(ogt2, [args["batchsize"],2])

        '''
        l1 = tf.keras.losses.MSE(ogt1, i1_pred)
        l2 = tf.keras.losses.MSE(ogt2, i2_pred)'''
        
        
        cce = tf.keras.losses.CategoricalCrossentropy()
        l1 = cce(ogt1, i1_pred)
        l2 = cce(ogt2, i2_pred)
        
        i1_discrete = tf.argmax(i1_pred, dimension=-1, output_type = tf.int32)
        i2_discrete = tf.argmax(i2_pred, dimension=-1, output_type = tf.int32)

        i_rankings = tf.stack([i1_discrete, i2_discrete], axis=-1)
        i_rankings = tf.reshape(i_rankings, [args['batchsize'],2])

        l1_act = i_rankings
        l2_act = ogt_positions

        '''
        (e_output, e_state) = tf.nn.dynamic_rnn(cell = self.lstmcell, inputs = ptrinp, dtype=tf.float32)
        e_state = tf.reshape(e_state, [args["batchsize"], tf.size(e_state)/(args["batchsize"]*args["bits"]), args["bits"]])
        (d_output, d_state) = tf.nn.dynamic_rnn(cell = self.lstmcell, inputs = e_state, dtype=tf.float32)
        '''
        '''
        i1r_onehot = tf.one_hot(tf.cast(i1r, tf.uint8), 5)
        i2r_onehot = tf.one_hot(tf.cast(i2r, tf.uint8), 5)
        rgt = tf.cond(tf.cast(i1r>i2r, tf.bool), lambda: tf.stack([i1r_onehot, i2r_onehot]), lambda: tf.stack([i2r_onehot, i1r_onehot]))
        '''
        '''
        # original pointer network without context attention
        ogt = tf.reshape(ogt, [args["batchsize"],2])
        ogt = tf.one_hot(ogt, depth=2)
        ogt1, ogt2 = tf.split(ogt, num_or_size_splits=2, axis=1)
        ogt1 = tf.reshape(ogt1, [args["batchsize"],2])
        ogt2 = tf.reshape(ogt2, [args["batchsize"],2])
        e_output1, e_output2 = tf.unstack(e_output, axis=1)
        d_output1, d_output2 = tf.unstack(d_output, axis=1)
        W1 = tf.Variable(tf.random.uniform(shape=[args["bits"], 1], minval=0.1, maxval=1), trainable=True)
        W2 = tf.Variable(tf.random.uniform(shape=[args["bits"], 1], minval=0.1, maxval=1), trainable=True)
        W3 = tf.Variable(tf.random.uniform(shape=[args["bits"], 1], minval=0.1, maxval=1), trainable=True)
        W4 = tf.Variable(tf.random.uniform(shape=[args["bits"], 1], minval=0.1, maxval=1), trainable=True)
        i1_pred = tf.math.softmax(tf.cast((tf.matmul(tf.math.multiply(e_output1, d_output1), W1), tf.matmul(tf.math.multiply(e_output2, d_output1), W2)), tf.float32))
        i2_pred = tf.math.softmax(tf.cast((tf.matmul(tf.math.multiply(e_output1, d_output2), W3), tf.matmul(tf.math.multiply(e_output2, d_output2), W4)), tf.float32))
        i1_pred = tf.reshape(i1_pred, [args["batchsize"],2])
        i2_pred = tf.reshape(i2_pred, [args["batchsize"],2])
        cce = tf.keras.losses.CategoricalCrossentropy()
        
        l1 = cce(ogt1, i1_pred)
        l2 = cce(ogt2, i2_pred)
        
        i1_discrete = tf.argmax(i1_pred, dimension=1)
        i2_discrete = tf.argmax(i2_pred, dimension=1)

        ogt1 = tf.argmax(ogt1, dimension=1)
        ogt2 = tf.argmax(ogt2, dimension=1)

        l1_act = tf.reduce_mean(tf.math.pow((i1_discrete - ogt1), 2))
        l2_act = tf.reduce_mean(tf.math.pow((i2_discrete - ogt2), 2))
        '''
        decoder_dim = 512

        i1_decoded = item_decoder(i1_noisy_content_hashcode, args["vae_units"], args["vae_layers"], decoder_dim)
        user_decoded = user_decoder(user_noisy_content_hashcode, args["vae_units"], args["vae_layers"], decoder_dim)

        user_reconloss = tf.reduce_mean(tf.math.pow(user_decoded-user_content, 2) , axis=-1)
        i1_reconloss = tf.reduce_mean(tf.math.pow(i1_decoded-i1_content, 2) , axis=-1)

        user_content_hashcode = 2*user_content_hashcode - 1
        i1_content_hashcode = 2*i1_content_hashcode - 1
        i1_org_m_noselfmask = i1_content_hashcode
        user_m = user_content_hashcode
        nonzero_bits = args["bits"]

        def make_total_loss(i1_org, i1r, anneal):
            e0 = tf.random.normal([batchsize], stddev=1.0, name='normaldis0')
            i1 = i1_org
            i1r = i1r
            i1r_m = 2*nonzero_bits * (i1r/max_rating) - nonzero_bits
            dot_i1 = tf.reduce_sum(user_content_hashcode * i1, axis=-1)
            sqr_diff = tf.math.pow((i1r_m - dot_i1)/nonzero_bits, 2)
            loss = tf.reduce_mean(sqr_diff, axis=-1)
            
            loss_kl = tf.multiply(i1_content_cont, tf.math.log(tf.maximum(i1_content_cont / 0.5, 1e-10))) + \
                      tf.multiply(1 - i1_content_cont, tf.math.log(tf.maximum((1 - i1_content_cont) / 0.5, 1e-10)))
            loss_kl = tf.reduce_sum(tf.reduce_sum(loss_kl, 1), axis=0)

            loss_kl_user = tf.multiply(user_content_cont, tf.math.log(tf.maximum(user_content_cont / 0.5, 1e-10))) + \
                      tf.multiply(1 - user_content_cont, tf.math.log(tf.maximum((1 - user_content_cont) / 0.5, 1e-10)))
            loss_kl_user = tf.reduce_sum(tf.reduce_sum(loss_kl_user, 1), axis=0)

            total_loss = loss + args["KLweight"]*(loss_kl + loss_kl_user)
            i1_dist = -dot_i1
            return total_loss, i1_dist, loss

        total_loss, ham_dist_i1, reconloss = make_total_loss(i1_content_hashcode, i1r, sigma_anneal)

        if args["item_emb_type"] == 0:
            total_loss = total_loss
        else:
            total_loss = total_loss + args["vae_weight"] * (user_reconloss + i1_reconloss) + l1 + l2
        
        loss_vae = user_reconloss + i1_reconloss

        return total_loss, reconloss, ham_dist_i1, i1_org_m_noselfmask, user_m, loss_vae, l1_act, l2_act