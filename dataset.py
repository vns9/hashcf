import csv  
import sys
import numpy as np
import scipy.io
from tqdm import tqdm
from scipy.sparse import csr_matrix
from random import shuffle
import pandas as pd
import gzip
import json
import codecs
import tensorflow as tf
import nltk
#nltk.download()
from nltk.tokenize import word_tokenize

gloveModel = {}

def loadGloveModel(File):
    #print("Loading Glove Model")
    f = open(File,'r')
    #gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    #print(len(gloveModel)," words loaded!")
    return gloveModel


def load_glove_model(glove_file):
    """
    :param glove_file: embeddings_path: path of glove file.
    :return: glove model
    """

    f = codecs.open(glove_file + ".txt", 'r', encoding='utf-8')
    model = {}
    for line in f[1:]:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding
    return model

def convert_to_binary(embedding_path):
    """
    Here, it takes path to embedding text file provided by glove.
    :param embedding_path: takes path of the embedding which is in text format or any format other than binary.
    :return: a binary file of the given embeddings which takes a lot less time to load.
    """
    f = codecs.open(embedding_path + ".txt", 'r', encoding='utf-8')
    wv = []
    with codecs.open(embedding_path + ".vocab", "w", encoding='utf-8') as vocab_write:
        count = 0
        for line in f:
            if count == 0:
                pass
            else:
                splitlines = line.split()
                vocab_write.write(splitlines[0].strip())
                vocab_write.write("\n")
                wv.append([float(val) for val in splitlines[1:]])
            count += 1
    np.save(embedding_path + ".npy", np.array(wv))

def load_embeddings_binary(embeddings_path):
    """
    It loads embedding provided by glove which is saved as binary file. Loading of this model is
    about  second faster than that of loading of txt glove file as model.
    :param embeddings_path: path of glove file.
    :return: glove model
    """
    with codecs.open(embeddings_path + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]
    wv = np.load(embeddings_path + '.npy')
    model = {}
    for i, w in enumerate(index2word):
        model[w] = wv[i]
    return model

def get_w2v(sentence, model):
    """
    :param sentence: inputs a single sentences whose word embedding is to be extracted.
    :param model: inputs glove model.
    :return: returns numpy array containing word embedding of all words    in input sentence.
    """
    return np.array([model.get(val, np.zeros(100)) for val in sentence.split()], dtype=np.float64)

def parse(path):
  g = gzip.open(path, 'rb')
  i = 0
  for l in g:
    yield eval(l)
    i += 1
    if i % 1000000 == 0:
        print(i / 8898041)
        #break

def parse_json(path):
    i = 0
    with open(path) as f:
        for line in f:
            yield json.loads(line)
            i += 1

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    if (d["overall"]) < 1:
        print(d)
        print()

    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = 'ratings_contentaware_full.mat'

    print(input_path, output_path)

    # model = load_glove_model("glove.6B.50d")
    # convert_to_binary("glove.6B.50d")
    # load_embeddings_binary("glove.6B.50d")
    # get_w2v("this is testing", model)
    #print(gloveModel['venkat'])



    min_number = 20
    data = []
    if "amazon.json" in input_path:
        #df = getDF(input_path).values
        #print("amazon shape", df.shape)
        for row in parse_json(input_path):
            user = row["reviewerID"]
            item = row["asin"]
            rating = int(row["overall"])
            sent = " "
            if "reviewText" in row:
                sent+=row["reviewText"]
            #print(sent)
            #sent = row["reviewText"]
            timepoint = 0
            data.append([user, item, rating, timepoint, sent])
    elif "yelp" in input_path:
        tmp_user_idx_dict = {}
        tmp_item_idx_dict = {}
        tmp_user_idx = 0
        tmp_item_idx = 0

        for row in parse_json(input_path):
            user = row["user_id"]
            item = row["business_id"]

            if user not in tmp_user_idx_dict:
                tmp_user_idx_dict[user] = tmp_user_idx
                tmp_user_idx += 1
            if item not in tmp_item_idx_dict:
                tmp_item_idx_dict[item] = tmp_item_idx
                tmp_item_idx += 1

            user = tmp_user_idx_dict[user]
            item = tmp_item_idx_dict[item]

            rating = int(row["stars"])
            timepoint = 0#int(row["date"])
            review = row["text"]
            
            data.append([user, item, rating, timepoint, review])
            #print(data[-1])
    else:
        print("unknown")
        exit(-1)

    #data = np.array(data) #, dtype=int)
    #print(data.shape)

    # changed = True
    # while changed:
    unique_users, unique_users_count = np.unique([row[0] for row in data], return_counts=True)
    unique_items, unique_items_count = np.unique([row[1] for row in data], return_counts=True)

    low_users = set(unique_users[unique_users_count < min_number].tolist())
    low_items = set(unique_items[unique_items_count < min_number].tolist())

    delete_indices = []
    user2idx = {}
    item2idx = {}

    user_idx = 0
    item_idx = 0
    seen_pairs = {}
    for i in tqdm(range(len(data))):
        user = data[i][0]
        item = data[i][1]

        ui_pair = (user, item)
        if user in low_users or item in low_items or ui_pair in seen_pairs:
            delete_indices.append(i)
        else:
            seen_pairs[ui_pair] = 1
            if user not in user2idx:
                user2idx[user] = user_idx
                user_idx += 1

            if item not in item2idx:
                item2idx[item] = item_idx
                item_idx += 1

    if len(delete_indices) > 0:
        delete_indices = set(delete_indices)
        data = [x for i,x in enumerate(data) if i not in delete_indices]
        #data = np.delete(data, delete_indices, axis=0)
    # else:
    #     changed = False

    num_users = len(user2idx)
    num_items = len(item2idx)

        #print("users and items", num_users, num_items, len(data))

    print(num_users, num_items)
    matrix = np.zeros((num_users, num_items), dtype=np.float32)
    matrix = csr_matrix(matrix)

    row_idx = []
    col_idx = []
    vals = []

    user2items = {}
    item2reviews = [[] for _ in range(len(item2idx))]
    item2reviewsdict = {}
    for i in range(len(data)):
        user = user2idx[data[i][0]]
        item = item2idx[data[i][1]]
        rating = data[i][2]
        review = data[i][4]

        item2reviews[item].append(review)
        if item not in item2reviewsdict.keys():
            item2reviewsdict[item] = []
            item2reviewsdict[item].append(review)
        else:
            item2reviewsdict[item].append(review)

        if user not in user2items:
            user2items[user] = []

        if item not in user2items[user]: # some items may be rated multiple times. just pick the first...
            user2items[user].append(item)

            row_idx.append(user)
            col_idx.append(item)
            vals.append(rating)

    lens = np.array([len(user2items[user]) for user in user2items])
    #assert(len(lens[lens < min_number]) == 0)

    #print(set(vals), max(vals), min(vals))
    vals = np.array(vals, dtype=int)
    row_idx = np.array(row_idx)
    col_idx = np.array(col_idx)

    matrix = csr_matrix((vals, (row_idx, col_idx)), shape=(num_users, num_items))

    loadGloveModel("glove.6B.50d.txt")

    for i in range(len(item2reviews)):
        item2reviews[i] = " ".join(item2reviews[i])
        
    
    for i in range(len(item2reviews)):
        if len(item2reviews) == 0:
            item2reviews[i] = []
        else:
            tokens = word_tokenize(item2reviews[i])
            words = [word for word in tokens if word.isalpha()]
            sent2vec = []
            for word in words:
                if word.lower() in gloveModel.keys():
                    sent2vec.append(gloveModel[word.lower()])
            #print(sent2vec[0].shape)
            item2reviews[i] = np.array(sent2vec)
            #print(item2reviews[i].shape)
            item2reviewsdict[i] = sent2vec

    # print(matrix.shape)
    # print(np.mean(matrix>0))

    ### tf-idf
    #from sklearn.feature_extraction.text import TfidfVectorizer
    
        
        

    

    # vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95, max_features=8000)
    # item2reviews1 = item2reviews
    # item2reviews1 = vectorizer.fit_transform(item2reviews)
    # idx2term = {vectorizer.vocabulary_[key] : key for key in vectorizer.vocabulary_ }
    # idx2term = [idx2term[i] for i in range(len(idx2term))]
    # print("#terms", len(idx2term))
    # print("#stop words", len(vectorizer.stop_words_))

    # we construct a random 50/50% split
    # In each split, we pick out all rated items as well as the same amount of unrated items
    train_split = lambda x : int(x * 0.5)
    val_split = lambda  x : int(x * 0.5)
    all_items = set([i for i in range(num_items)])

    train_rated_total = []
    train_unrated_total = []
    val_rated_total = []
    val_unrated_total = []
    test_rated_total = []
    test_unrated_total = []

    row_idx = []
    col_idx = []
    vals = []

    for u in tqdm(range(num_users)):
        user_ratings = user2items[u]
        shuffle(user_ratings)

        M = len(user_ratings)
        train_ratings = user_ratings[:train_split(len(user_ratings))]
        test_ratings = user_ratings[train_split(len(user_ratings)):]

        # sample M random unrated items from each
        user_ratings = set(user_ratings)
        remaining_items = list(all_items - user_ratings)
        shuffle(remaining_items)

        train_unrated = remaining_items[:train_split(len(remaining_items))][:M]
        test_unrated  = remaining_items[train_split(len(remaining_items)):][:M]

        val_ratings = train_ratings[val_split(len(train_ratings)):]
        val_unrated = train_unrated[val_split(len(train_unrated)):]

        # remove validation from training data
        train_ratings = train_ratings[:val_split(len(train_ratings))]
        train_unrated = train_unrated[:val_split(len(train_unrated))]

        # we extract the indices to generate a "training matrix" needed for some baselines
        for item in train_ratings:
            value = matrix[u, item]
            assert(value > 0.999)
            row_idx.append(u)
            col_idx.append(item)
            vals.append(value)

        train_rated_total.append( train_ratings )
        train_unrated_total.append( train_unrated )

        val_rated_total.append( val_ratings )
        val_unrated_total.append( val_unrated )

        test_rated_total.append( test_ratings )
        test_unrated_total.append( test_unrated )

    # make a masked matrix only containing the training ratings
    matrix_train = csr_matrix((vals, (row_idx, col_idx)), shape=(num_users, num_items))
    # print(matrix_train.shape)
    # print(np.mean(matrix_train>0))
    #item2reviews = tf.constant(item2reviews)
    #print(type(item2reviews[0][0]))
    item2reviews = np.array(item2reviews)
    zero_words = np.zeros(50, dtype=np.float)
    maxlen = max(r.shape[0] for r in item2reviews)
    for r in item2reviews:
        if r.shape[0] < maxlen:
            lent = r.shape[0]
            for _ in range(maxlen - lent):
                r = np.vstack([r, zero_words])
            #print(r.shape)
    '''
    item2reviews1 = tf.ragged.constant(item2reviews)
    item2reviews1 = item2reviews1.to_tensor()
    fin = tf.constant([1])
    ans = tf.nn.embedding_lookup(item2reviews1, fin)
    
    with tf.Session() as sess:
        sess.run(ans)
        print(ans.eval())
    '''


    scipy.io.savemat(output_path, {'full_matrix':matrix, 'train_matrix':matrix_train, 'item_features':item2reviews,
                                   'train_rated_total':train_rated_total, 'train_unrated_total':train_unrated_total,
                                   'val_rated_total': val_rated_total, 'val_unrated_total': val_unrated_total,
                                   'test_rated_total':test_rated_total, 'test_unrated_total':test_unrated_total})
    print('done')