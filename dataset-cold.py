import warnings
warnings.filterwarnings("ignore")
import os
os.environ['KMP_WARNINGS'] = 'off'
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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow_hub as hub
os.environ["TFHUB_CACHE_DIR"] = './tfhub_modules'
import nltk

from nltk.tokenize import word_tokenize
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
#embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
#embed = hub.Module("./tfhub_modules/saved_model.pb")
embedding_size = 512

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
    splitter = int(sys.argv[2])
    output_path = 'ratings_contentaware_full.mat'

    min_number = 20
    data = []
    if "home5c.json" in input_path or "amazon.json" in input_path or "instrument5c.json" in input_path or "pantry5c.json" in input_path or "music5c.json" in input_path or "office5c.json" in input_path:
        #df = getDF(input_path).values
        #print("amazon shape", df.shape)
        for row in parse_json(input_path):
            user = row["reviewerID"]
            item = row["asin"]
            rating = int(row["overall"])
            if rating == 5:
                continue
            sent = " "
            if "reviewText" in row:
                sent+=row["reviewText"]
            '''
            no_punct = ""
            for char in sent:
                if char not in punctuations:
                    no_punct = no_punct + char
            sent = no_punct
            sent = word_tokenize(sent)
            sent = [word for word in sent if not word in stopwords]
            if len(sent) > 50:
                del sent[-50:]
            sent = " ".join(sent)
            '''
            timepoint = 0
            data.append([user, item, rating, timepoint, sent])
    elif "yelp.json" in input_path:
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
            if rating == 5:
                continue
            timepoint = 0
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
    user2reviews = [[] for _ in range(len(user2idx))]
    #user2reviewsdict = {}
    #item2reviewsdict = {}
    for i in range(len(data)):
        user = user2idx[data[i][0]]
        item = item2idx[data[i][1]]
        rating = data[i][2]
        review = data[i][4]

        item2reviews[item].append(review)
        user2reviews[user].append(review)

        '''
        if user not in user2reviewsdict.keys():
            user2reviewsdict[user] = []
            user2reviewsdict[user].append(review)
        else:
            user2reviewsdict[user].append(review)

        if item not in item2reviewsdict.keys():
            item2reviewsdict[item] = []
            item2reviewsdict[item].append(review)
        else:
            item2reviewsdict[item].append(review)
        '''

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

    '''
    for i in range(len(item2reviews)):
        no_punct = ""
        item2reviews[i] = " ".join(item2reviews[i])
        for char in item2reviews[i]:
            if char not in punctuations:
                no_punct = no_punct + char
        item2reviews[i] = no_punct
        item2reviews[i] = word_tokenize(item2reviews[i])
        item2reviews[i] = [word for word in item2reviews[i] if not word in stopwords]
        if len(item2reviews[i]) > 50:
            del item2reviews[i][-50:]
        item2reviews[i] = " ".join(item2reviews[i])
  

    for i in range(len(user2reviews)):
        no_punct = ""
        user2reviews[i] = " ".join(user2reviews[i])
        for char in user2reviews[i]:
            if char not in punctuations:
                no_punct = no_punct + char
        user2reviews[i] = no_punct
        user2reviews[i] = word_tokenize(user2reviews[i])
        user2reviews[i] = [word for word in user2reviews[i] if not word in stopwords]
        if len(user2reviews[i]) > 50:
            del user2reviews[i][-50:]
        user2reviews[i] = " ".join(user2reviews[i])
    '''
    item2rev = []
    for i in range(len(item2reviews)):
        item2rev += item2reviews[i][:5]
        while len(item2rev)%5!=0:
            item2rev.append(" ")

    for i in range(len(item2reviews)):
        item2reviews[i] = " ".join(item2reviews[i])
    
    for i in range(len(user2reviews)):
        user2reviews[i] = " ".join(user2reviews[i])
    
    tf.reset_default_graph()
    graph = tf.Graph()    
    user2reviews1 = None
    item2reviews1 = None
    item2rev1 = None

    sess = tf.Session()
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
    sess.run(tf.initialize_all_variables())
    sess.run(tf.tables_initializer())

    def usembed(x, sess, embed):
        ans = embed(x)
        ans = sess.run(ans)
        return ans

    xl = [item2reviews[i:i + 4000] for i in range(0, len(item2reviews), 4000)] 
    yl = [user2reviews[i:i + 2000] for i in range(0, len(user2reviews), 2000)] 
    nyl = [item2rev[i:i + 4000] for i in range(0, len(item2rev), 4000)] 

    for y in tqdm(yl):
        y = usembed(y, sess, embed)
        if user2reviews1 is None:
            user2reviews1 = y
        else:
            user2reviews1 = np.concatenate((user2reviews1, y), axis=0) 
    
    for x in tqdm(xl):
        x = usembed(x, sess, embed)
        if item2reviews1 is None:
            item2reviews1 = x
        else:
            item2reviews1 = np.concatenate((item2reviews1, x), axis=0)     

    for y in tqdm(nyl):
        p = usembed(y, sess, embed)  
        if item2rev1 is None:
            item2rev1 = p
        else:
            item2rev1 = np.concatenate((item2rev1, p), axis=0)
        
    item2rev1 = item2rev1.reshape(num_items,5,512)

    '''
    tf.reset_default_graph()
    graph = tf.Graph()    
    # Universal sentence encoder
    user2reviews1 = None
    item2reviews1 = None
    with tf.Session(graph = graph) as sess:
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
        sess.run(tf.initialize_all_variables())
        sess.run(tf.tables_initializer())
        item2reviews1 = embed(item2reviews)
        item2reviews1 = tf.reshape(item2reviews1, (num_items, 512))
        #item2reviews1 = tf.zeros([num_items, 512], tf.float32)
        item2reviews1 = sess.run(item2reviews1)
        user2reviews1 = embed(user2reviews)
        user2reviews1 = tf.reshape(user2reviews1, (num_users, 512))
        #user2reviews1 = tf.zeros([num_users, 512], tf.float32)
        user2reviews1 = sess.run(user2reviews1)
    '''
    
    item_occurrences = [np.sum(matrix[:,i]) for i in range(matrix.shape[1])]
    order = np.argsort(item_occurrences)

    train_item_idx = order[::splitter]
    train_item_idx = train_item_idx[np.random.permutation(len(train_item_idx))]

    test_item_idx = np.array([i for i in range(num_items) if i not in train_item_idx])
    test_item_idx = test_item_idx[np.random.permutation(len(test_item_idx))]
    test_item_idx = set(test_item_idx)
    
    val_item_idx = set(train_item_idx[:int(len(train_item_idx)*0.15)])
    train_item_idx = set(train_item_idx[int(len(train_item_idx)*0.15):])

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

        M = len(user_ratings)
        train_items = []
        val_items = []
        test_items = []
        for single_item in user_ratings:
            if single_item in test_item_idx:
                test_items.append(single_item)
            elif single_item in train_item_idx:
                train_items.append(single_item)
            elif single_item in val_item_idx:
                val_items.append(single_item)
            else:
                exit(-1)

        #train_ratings = user_ratings[:train_split(len(user_ratings))]
        #test_ratings = user_ratings[train_split(len(user_ratings)):]

        # sample M random unrated items from each
        #user_ratings = set(user_ratings)
        #remaining_items = list(all_items - user_ratings)
        #shuffle(remaining_items)

        train_unrated =  1 #remaining_items[:train_split(len(remaining_items))][:M]
        test_unrated  = 1 #remaining_items[train_split(len(remaining_items)):][:M]

        #val_ratings = train_ratings[val_split(len(train_ratings)):]
        val_unrated = 1 #train_unrated[val_split(len(train_unrated)):]

        # remove validation from training data
        #train_ratings = train_ratings[:val_split(len(train_ratings))]
        train_unrated = 1 #train_unrated[:val_split(len(train_unrated))]

        # we extract the indices to generate a "training matrix" needed for some baselines
        for item in train_items:
            value = matrix[u, item]
            assert(value > 0.999)
            row_idx.append(u)
            col_idx.append(item)
            vals.append(value)

        train_rated_total.append( train_items )
        train_unrated_total.append( train_unrated )

        val_rated_total.append( val_items )
        val_unrated_total.append( val_unrated )

        test_rated_total.append( test_items )
        test_unrated_total.append( test_unrated )

    matrix_train = csr_matrix((vals, (row_idx, col_idx)), shape=(num_users, num_items))


    scipy.io.savemat(output_path, {'item2rev': np.array(item2rev1), 'full_matrix':matrix, 'train_matrix':matrix_train, 'user_features':user2reviews1, 'item_features':item2reviews1,
                                   'train_rated_total':train_rated_total, 'train_unrated_total':train_unrated_total,
                                   'val_rated_total': val_rated_total, 'val_unrated_total': val_unrated_total,
                                   'test_rated_total':test_rated_total, 'test_unrated_total':test_unrated_total})
    print('done')