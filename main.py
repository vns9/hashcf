import warnings
warnings.filterwarnings("ignore")
import argparse
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pickle
import glob
import os
from nn_helpers import generator
from ptrvae import Model as pModel
import numpy as np
import time
from helpers import ndcg_score, mrr, hit
from scipy.io import loadmat
import datetime
from bit_inspecter import bit_histogram
import multiprocessing as mp
from scipy.io import loadmat
from scipy.special import entr
import time
import pandas as pd
from sklearn.metrics import ndcg_score as ndcg_score1
import statistics


def fprint(output_file, text):
    with open(output_file, "a") as myfile:
        myfile.write(str(text) + "\n")

def onepass_test(sess, eval_list, l1_act, l2_act, some_handle, num_samples, eval_batchsize,
                 handle, anneal_val, anneal_val_vae, batch_placeholder, is_training, force_selfmask, args, num_items, num_users,
                 full_matrix_ratings, test_samples):

    valcounter = 0

    item_matrix = np.zeros((num_items, args["bits"]))
    user_matrix = np.zeros((num_users, args["bits"]))

    total = num_samples
    valdone = False

    all_item_ids = []
    all_user_ids = []
    l1list = []
    l2list = []
    while not valdone:
        [lossval, hamdist, userval, item_ratingval, itemsample, item_emb, user_emb, \
            lossval_recon], l1_act_val, l2_act_val = sess.run([eval_list, l1_act, l2_act], feed_dict={handle: some_handle, is_training: False,
                                                                        anneal_val: 0,
                                                                        anneal_val_vae: 0,
                                                                        batch_placeholder: min(20000, eval_batchsize)})

        temp_list = []
        for i in range(l1_act_val.shape[0]):
                t1 = [x+1 for x in l1_act_val[i]]
                t2 = [x+1 for x in l2_act_val[i]]
                print(t1)
                print(t2)
                temp_list.append(ndcg_score(t1, t2))
        l1list.append(statistics.mean(temp_list))
        l2list.append(0)
        valcounter += 1
        total -= len(userval)

        if total <= 0:
            valdone = True

        item_emb = item_emb.tolist()
        user_emb = user_emb.tolist()

        for kk in range(len(userval)):
            user = userval[kk]
            item = itemsample[kk]

            item_matrix[item] = item_emb[kk]
            user_matrix[user] = user_emb[kk]

            all_item_ids.append(item)
            all_user_ids.append(user)
    print('valcounter ', valcounter)
    #assert(total == 0)
    assert(len(set(all_user_ids)) == num_users)
    #assert(len(set(all_item_ids)) == num_items)
    
    ndcgs, mse = eval_hashing(full_matrix_ratings, test_samples[0], item_matrix, user_matrix, num_users, num_items, all_user_ids, all_item_ids, args)
    print('PTR NDCG - ',np.mean(l1list))
    return np.mean(ndcgs,0), mse, item_matrix, user_matrix

def eval_hashing(full_matrix_ratings, test_samples, item_matrix, user_matrix, num_users, num_items, user_list, item_list, args):
    inps = []
    mses = []
    gt_entropies = []
    for user in (range(num_users)):
        user_emb = user_matrix[user]
        items = test_samples[user][0]
        item_ids = items
        items = item_matrix[item_ids]
        user_emb_01 = (user_emb+1)/2
        if args["force_selfmask"]:
            ham_dists = np.array([np.sum(user_emb * ( 2*((item+1)/2 * user_emb_01) - 1)) for item in items])
        else:
            ham_dists = np.array([np.sum(user_emb * item) for item in items])
        items_gt = np.squeeze(np.array(full_matrix_ratings[user, item_ids].todense()), 0)
        pgti = items_gt / items_gt.sum()
        shannon = -np.sum(pgti*np.log2(pgti))
        gt_entropies.append(shannon)
        if shannon>6.2:
            inps.append([items_gt, args["bits"]-ham_dists])
            #print(items_gt)
        tmp_gt = 2*args["bits"] * items_gt/5.0 - args["bits"]
        mse = ((tmp_gt - ham_dists)**2) / (args["bits"]*args["bits"])#(args["batchsize"])  
        mses += mse.tolist()
    describer = pd.DataFrame(np.array(gt_entropies))
    print(describer.describe())
    #print(gt_entropies)
    ndcgs = [ndcg_score(inp[0], inp[1]) for inp in inps]
    return ndcgs, np.mean(mses)


def onepass(sess, eval_list, some_handle, num_samples, eval_batchsize, handle, anneal_val, anneal_val_vae, batch_placeholder, is_training, force_selfmask, args):
    losses_val = []
    losses_val_recon = []
    losses_val_vae = []

    losses_val_eq = []
    losses_val_uneq = []

    valcounter = 0
    val_user_items = {}
    total = num_samples
    valdone = False

    user_vectors = []
    item_vectors = []

    while not valdone:
        lossval, hamdist, userval, item_ratingval, itemsample, item_emb, user_emb, \
            lossval_recon = sess.run(eval_list, feed_dict={handle: some_handle, is_training: False,
                                                                        anneal_val: 0,
                                                                        anneal_val_vae: 0,
                                                                        batch_placeholder: min(20000, eval_batchsize)})

        losses_val.append(lossval)

        losses_val_recon.append(lossval_recon)

        valcounter += 1
        total -= len(userval)
        user_vectors += user_emb.tolist()
        item_vectors += item_emb.tolist()

        if total <= 0:
            valdone = True

        for kk in range(len(userval)):
            user = userval[kk]
            item_rating = item_ratingval[kk]
            user_item_score = -np.sum(user_emb[kk] * item_emb[kk])
            if user not in val_user_items:
                val_user_items[user] = [[], []]
            val_user_items[user][0].append(int(user_item_score))
            val_user_items[user][1].append(int(item_rating))

    t = 0

    inps = []
    for user in val_user_items:
        t += len(val_user_items[user][1])
        inps.append([val_user_items[user][1], val_user_items[user][0]])
    res = pool.starmap_async(ndcg_score, inps)
    res2 = pool.starmap_async(mrr, inps)
    mrrs = res2.get()
    ndcgs = res.get()

    if not args["realvalued"]:
        user_vectors = np.array(user_vectors).astype(int)
        item_vectors = np.array(item_vectors).astype(int)
    else:
        user_vectors = np.array(user_vectors)
        item_vectors = np.array(item_vectors)
    
    fprint(args["ofile"],"val" + " ".join([str(v) for v in [np.mean(bit_histogram(user_vectors, 1)), np.mean(bit_histogram(item_vectors, 1))]]))

    return np.mean(losses_val), np.mean(ndcgs, 0), np.mean(losses_val_recon), \
           np.mean(losses_val_uneq), np.mean(losses_val_eq), len(ndcgs), ndcgs, np.mean(losses_val_vae)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", default=1024, type=int) 
    parser.add_argument("--bits", default=32, type=int) 
    parser.add_argument("--lr", default=0.000005, type=float) 
    parser.add_argument("--vae_units", default=200, type=int) 
    parser.add_argument("--vae_layers", default=4, type=int) 
    parser.add_argument("--dataset", default="software5c", type=str) 
    parser.add_argument("--vae_weight", default=1.0, type=float) 
    parser.add_argument("--mul", default=6, type=float) 
    parser.add_argument("--anneal_val", default=1.0, type=float)
    parser.add_argument("--decay_rate", default=0.9, type=float)
    parser.add_argument("--ofile", default="../lstmcf.txt", type=str)
    parser.add_argument("--deterministic_eval", default=1, type=int)
    parser.add_argument("--deterministic_train", default=1, type=int)
    parser.add_argument("--optimize_selfmask", default=0, type=int)
    parser.add_argument("--usermask_nograd", default=0, type=int)
    parser.add_argument("--KLweight", default=1.0, type=float)
    parser.add_argument("--force_selfmask", default=0, type=int)
    parser.add_argument("--save_vectors", default=0, type=int)
    parser.add_argument("--realvalued", default=0, type=int)
    parser.add_argument("--annealing_min", default=0.0, type=float)
    parser.add_argument("--annealing_max", default=1.0, type=float)
    parser.add_argument("--annealing_decrement", default=0.000001, type=float)
    parser.add_argument("--item_emb_type", default=1, type=int)
    parser.add_argument("--loss_type", default="normal", type=str)
    parser.add_argument("--loss_alpha", default=3.0, type=float)

    args = parser.parse_args()
    args.realvalued = args.realvalued > 0.5
    args.deterministic_eval = args.deterministic_eval > 0.5
    args.usermask_nograd = args.usermask_nograd > 0.5
    args.deterministic_train = args.deterministic_train > 0.5
    args.optimize_selfmask = args.optimize_selfmask > 0.5
    args.force_selfmask = args.force_selfmask > 0.5
    args.save_vectors = args.save_vectors > 0.5

    args = vars(args)
    print("Proposed model")
    print(args)
    fprint(args["ofile"], "Proposed model")
    fprint(args["ofile"], args)

    eval_batchsize = args["batchsize"]
    basepath = args["dataset"]+"/tfrecord/"
    dicfile = basepath + "dict.pkl"
    dicfile = pickle.load(open(dicfile, "rb"))

    num_users, num_items = dicfile[0], dicfile[1]
    args["num_users"] = num_users
    args["num_items"] = num_items

    trainfiles = glob.glob(basepath + "*train_*tfrecord")
    valfiles = glob.glob(basepath + "*val_*tfrecord")
    testfiles = glob.glob(basepath + "*test_*tfrecord")
    max_rating = 5.0
    train_samples = sum(1 for _ in tf.python_io.tf_record_iterator(trainfiles[0]))
    val_samples = sum(1 for _ in tf.python_io.tf_record_iterator(valfiles[0]))
    test_samples = sum(1 for _ in tf.python_io.tf_record_iterator(testfiles[0]))

    datamatlab = loadmat(args["dataset"]+"/ratings_contentaware_full.mat")
    item_content_matrix = datamatlab["item_features"]
    user_content_matrix = datamatlab["user_features"]
    tf.reset_default_graph()
    item_content_matrix = tf.convert_to_tensor(item_content_matrix)
    user_content_matrix = tf.convert_to_tensor(user_content_matrix)
    
    with tf.Session() as sess:
        sess.run(item_content_matrix)
        sess.run(user_content_matrix)

        handle = tf.placeholder(tf.string, shape=[], name="handle_iterator")
        training_handle, train_iter, gen_iter = generator(sess, handle, args["batchsize"], trainfiles, 0)
        val_handle, val_iter, _ = generator(sess, handle, eval_batchsize, valfiles, 1)
        test_handle, test_iter, _ = generator(sess, handle, eval_batchsize, testfiles, 1)
        sample = gen_iter.get_next()
        user_sample = sample[0]        
        item_sample = sample[1]
        item_rating = sample[4]
        ogt = sample[6]
        
        is_training = tf.placeholder(tf.bool, name="is_training")
        anneal_val = tf.placeholder(tf.float32, name="anneal_val", shape=())
        anneal_val_vae = tf.placeholder(tf.float32, name="anneal_val_vae", shape=())
        batch_placeholder = tf.placeholder(tf.int32, name="batch_placeholder")

        model = pModel(sample, args)

        item_emb_matrix, item_emb_ph, item_emb_init = model._make_embedding(num_items, args["bits"], "item_embedding")
        user_emb_matrix, _, _ = model._make_embedding(num_users, args["bits"], "user_embedding1")

        word_embedding_matrix, _, _ = model._make_embedding(8000, 300, "word_embedding")
        importance_embedding_matrix = model.make_importance_embedding(8000)
        content_matrix = item_content_matrix
        loss, loss_no_anneal, scores, item_embedding, user_embedding, \
            reconloss, l1_act, l2_act = model.make_network(sess, word_embedding_matrix, importance_embedding_matrix, user_content_matrix, item_content_matrix, item_emb_matrix, user_emb_matrix,
                                           is_training, args, max_rating, anneal_val, anneal_val_vae, batch_placeholder, ogt)

        step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(args["lr"], step, 100000, args["decay_rate"], staircase=True, name="lr")
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=args["lr"], name='GradientDescent')
        optimizer = tf.train.AdamOptimizer(learning_rate=args["lr"], name="Adam")
        #optimizer = tf.train.RMSPropOptimizer(lr, name="RMS")
        #optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9, name="sgd_momentum")
        train_step = optimizer.minimize(loss, global_step=step)
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())
        sess.run(tf.tables_initializer())    
        sess.run(train_iter.initializer)

        eval_list = [loss, scores, user_sample, item_rating, item_sample, item_embedding, user_embedding, reconloss]
        counter = 0
        losses_train = []
        losses_train_no_anneal = []
        reconlosses_train = []
        times = []
        l1_act_list = []
        l2_act_list = []
        anneal = args["anneal_val"]
        anneal_vae = args["annealing_max"]


        best_val_ndcg = 0
        best_val_loss = np.inf
        patience = 5

        patience_counter = 0 #Early stopping
        running = True
        print("TRAINING")
        all_val_ndcg = []
        while running:
            start = time.time()
            lossval, loss_no_anneal_val, reconlossval, hamdist, _, l1_act_val, l2_act_val= sess.run([loss, loss_no_anneal, reconloss, scores, train_step, l1_act, l2_act], feed_dict={handle: training_handle, is_training: True,
                                                                                  anneal_val: anneal,
                                                                                  anneal_val_vae: anneal_vae,
                                                                                  batch_placeholder: args["batchsize"]})
            times.append(time.time() - start)
            losses_train.append(lossval)
            losses_train_no_anneal.append(loss_no_anneal_val)
            reconlosses_train.append(reconlossval)
            temp_list = []
            for i in range(l1_act_val.shape[0]):
                temp_list.append(np.mean(ndcg_score(l1_act_val[i], l2_act_val[i])))
            l1_act_list.append(statistics.mean(temp_list))
            l2_act_list.append(l2_act_val)
            counter += 1
            anneal_vae = max(anneal_vae-args["annealing_decrement"], args["annealing_min"])
            anneal = anneal * 0.9999
            if counter % 1000 == 0: 
                print("train", np.mean(l1_act_list), np.mean(l2_act_list), np.mean(losses_train), np.mean(losses_train_no_anneal), np.mean(reconlosses_train))#, counter * args["batchsize"] / train_samples, np.mean(times), anneal)
                fprint(args["ofile"], " ".join([str(v) for v in ["train", np.mean(losses_train), np.mean(losses_train_no_anneal), counter * args["batchsize"] / train_samples, np.mean(times), anneal]]) )
                losses_train = []
                times = []
                losses_train_no_anneal = []
                reconlosses_train = []
                l1_act_list = []
                l2_act_list = []
                '''
                sess.run(val_iter.initializer)
                losses_val, val_ndcg, losses_val_recon, losses_val_uneq, losses_val_eq, NN, allndcgs, losses_val_vae = onepass(sess, eval_list, val_handle, val_samples, eval_batchsize, handle, anneal_val, anneal_val_vae, batch_placeholder, is_training, args["force_selfmask"], args)
                print("val\t\t", val_ndcg, [losses_val, losses_val_recon, losses_val_uneq, losses_val_eq], NN, losses_val_vae)
                save_val_ndcg = val_ndcg
                fprint(args["ofile"], " ".join([str(v) for v in ["val\t\t", val_ndcg, [losses_val, losses_val_recon, losses_val_uneq, losses_val_eq], NN]]) )

                all_val_ndcg.append(best_val_ndcg)
                if val_ndcg[-1] > best_val_ndcg:
                    best_val_ndcg = val_ndcg[-1]
                    best_val_loss = losses_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                '''
                #if patience_counter == 0:
                sess.run(test_iter.initializer)
                test_ndcgs, test_mse, user_matrix_vals, item_matrix_vals = onepass_test(sess, eval_list, l1_act, l2_act, test_handle,
                                test_samples, eval_batchsize,
                                handle, anneal_val, anneal_val_vae, batch_placeholder,
                                is_training, args["force_selfmask"], args,
                                num_items, num_users,
                                datamatlab["full_matrix"], datamatlab["test_rated_total"])
                print("test\t\t\t\t",  test_ndcgs, test_mse)
                fprint(args["ofile"], " ".join([str(v) for v in ["test\t\t\t\t", test_ndcgs, test_mse]]))
                
                if patience_counter >= patience:
                    running = False
                print("patience", patience_counter, "/", patience, (datetime.datetime.now()))
                fprint(args["ofile"], " ".join([str(v) for v in ["patience", patience_counter, "/", patience, (datetime.datetime.now())]]))
if __name__ == "__main__":
    pool = mp.Pool(4)
    main()
    pool.close()
    pool.join()