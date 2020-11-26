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
        item_content_feature_vector = tf.nn.embedding_lookup(item_content_matrix, i1)
        user = tf.nn.embedding_lookup(user_emb, user) 
        i1 = tf.nn.embedding_lookup(item_emb, i1)
        return user, i1, i1r, user_content_feature_vector, item_content_feature_vector 

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

    def make_network(self,sess, word_emb_matrix, importance_emb_matrix, user_content_matrix, item_content_matrix, item_emb, user_emb, is_training, args, max_rating, sigma_anneal, sigma_anneal_vae, batchsize):
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
        user, i1, i1r, user_content, i1_content = self._extract(sess, item_emb, user_emb, user_content_matrix, item_content_matrix, batchsize)
        i1_content = tf.reshape(i1_content, [batchsize, 512])
        user_content = tf.reshape(user_content, [batchsize, 512])
        
        #i1_content = tf.cond(is_training, lambda: tf.zeros_like(i1_content), lambda: tf.zeros_like(i1_content))
        #user_content = tf.cond(is_training, lambda: tf.zeros_like(user_content), lambda: tf.zeros_like(user_content))

        i1_content_hashcode, i1_content_cont = item_encoder(i1_content, args["vae_units"], args["vae_layers"])
        e = tf.random.normal([batchsize, args["bits"]])
        i1_noisy_content_hashcode = tf.math.multiply(e, sigma_anneal_vae) + i1_content_hashcode

        user_content_hashcode, user_content_cont = user_encoder(user_content, args["vae_units"], args["vae_layers"])
        e = tf.random.normal([batchsize, args["bits"]])
        user_noisy_content_hashcode = tf.math.multiply(e, sigma_anneal_vae) + user_content_hashcode

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
            total_loss = total_loss + args["vae_weight"] * (user_reconloss + i1_reconloss)
        
        loss_vae = user_reconloss + i1_reconloss

        return total_loss, reconloss, ham_dist_i1, i1_org_m_noselfmask, user_m, loss_vae