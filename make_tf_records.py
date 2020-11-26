import tensorflow as tf
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from random import shuffle
from multiprocessing import Pool
import pickle

'''
We generate a tfrecord file per item pair, with randomized order of the users within each file.
'''

def make_record_sample(sample):

    key_value_pairs = {
        'user': tf.train.Feature(int64_list = tf.train.Int64List(value = [sample[0]])),
        'i1': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample[1]])),
        'i2': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample[2]])),
        'i_unrated': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample[3]])),
        'i1_rating': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample[4]])),
        'i2_rating': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample[5]]))
    }

    features = tf.train.Features(feature=key_value_pairs)
    example = tf.train.Example(features=features)
    return example


def process_tfrecord_many(path, num_files, list, fn):
    #print(path)
    make_tf_name = lambda x : path + "_" + str(i) + ".tfrecord"
    N = len(list)
    for i in tqdm(range(num_files)):
        sublist = list[int(i/num_files*N) : int((i+1)/num_files*N)]
        #print(len(sublist), make_tf_name(i))
        with tf.python_io.TFRecordWriter(make_tf_name(i)) as tfwriter:
            for elm in sublist:
                tfwriter.write(fn(elm).SerializeToString())

def process_tfrecord(path, list, fn):
    with tf.python_io.TFRecordWriter(path) as tfwriter:
        for elm in list:
            tfwriter.write(fn(elm).SerializeToString())

# we want to consider samples of [user_id, i1, i2, i1_rating, i2_rating]
def make_all_pairs(ratings, notrated, full_matrix, pairs):
    samples = []
    num_users = len(ratings)
    for user in range(num_users):
        if len(ratings[user]) == 0:
            continue
        items = ratings[user][0]
        unrated_items = items #notrated[user][0]

        num_items = len(items)
        num_unrated_items = len(unrated_items)
        for i in range(num_items):
            i1 = items[i]
            i1_rating = int(full_matrix[user, i1])

            matches_found = 0
            for j in np.random.permutation(num_items):  # range(i, num_items):#num_items):
                i2 = items[j]
                i_unrated = unrated_items[j % num_unrated_items]

                i2_rating = int(full_matrix[user, i2])

                matches_found += 1
                sample = [user, i1, i2, i_unrated, i1_rating, i2_rating]
                samples.append(sample)

                assert (i1_rating > 0 and i2_rating > 0)
                if matches_found >= pairs:
                    break

    shuffle(samples)
    return samples

# we want to consider samples of [user_id, i1, i2, i1_rating, i2_rating]
def make_all_pairs_test(full_matrix, pairs):
    samples = []
    numusers = full_matrix.shape[0]
    numitems = full_matrix.shape[1]
    user = 1
    i2 = 1
    i_unrated = 1
    i1_rating = 1
    i2_rating = 1
    for i1 in range(numitems):
        sample = [user, i1, i2, i_unrated, i1_rating, i2_rating]
        samples.append(sample)
    for user in range(numusers):
        sample = [user, i2, i2, i_unrated, i1_rating, i2_rating]
        samples.append(sample)

    print("test", numitems, len(samples))
    return samples

def single_pair_tfrecord_maker(ratings, notrated, full_matrix,
                               path,
                               k):
    #print(path, k)
    '''
    # I assume there is a bug here
    if "test" in path:
        print("specific test handling")
        rated_samples = make_all_pairs_test(full_matrix, 1)
    else:
        rated_samples = make_all_pairs(ratings, notrated, full_matrix, 1)
    '''
    rated_samples = make_all_pairs(ratings, notrated, full_matrix, 1)
    #print(path, k, len(rated_samples))
    process_tfrecord(path + "_" + str(k) + ".tfrecord", rated_samples, make_record_sample)

def parallelize(rated, unrated, full_matrix, path, k_pairs):
    pool = Pool(processes=4)
    inp = []
    for i in range(len(rated)):
        inp += [[rated[i], unrated[i], full_matrix.copy(), path[i], k] for k in range(k_pairs[i])]
    v = pool.starmap_async(single_pair_tfrecord_maker, inp)
    v.get()
    pool.close()
    pool.join()

def main():
    path = 'ratings_contentaware_full.mat'
    vals = loadmat(path)

    '''
    train_matrix = csr_matrix(vals["train_matrix"])
    nonzeros_rows, nonzeros_cols = train_matrix.nonzero()
    num_users, num_items = train_matrix.shape[0], train_matrix.shape[1]

    print(num_users, num_items, np.max(nonzeros_cols), np.min(nonzeros_cols))
    '''

    full_matrix = csr_matrix(vals["full_matrix"])

    train_rated_total = vals["train_rated_total"][0]
    train_unrated_total = vals["train_unrated_total"][0]

    val_rated_total = vals["val_rated_total"][0]
    val_unrated_total = vals["val_unrated_total"][0]

    test_rated_total = vals["test_rated_total"][0]
    test_unrated_total = vals["test_unrated_total"][0]

    dict_path = "./tfrecord/dict.pkl"
    train_path = "./tfrecord/pair_train"
    val_path = "./tfrecord/pair_val"
    test_path = "./tfrecord/pair_test"

    pickle.dump(full_matrix.shape, open(dict_path, "wb"))
    parallelize([train_rated_total, val_rated_total, test_rated_total],
                [train_unrated_total, val_unrated_total, test_unrated_total],
                full_matrix,
                [train_path, val_path, test_path],
                [1,1,1])


    print("done")

if __name__ == '__main__':
    main()