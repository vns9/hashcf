import warnings
warnings.filterwarnings("ignore")
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
import os
import tensorflow_hub as hub
os.environ["TFHUB_CACHE_DIR"] = './tfhub_modules'
import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
stopwords = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz",]
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
    output_path = 'ratings_contentaware_full.mat'


    # model = load_glove_model("glove.6B.50d")
    # convert_to_binary("glove.6B.50d")
    # load_embeddings_binary("glove.6B.50d")
    # get_w2v("this is testing", model)
    #print(gloveModel['venkat'])



    min_number = 20
    data = []
    if "home5c.json" in input_path or "instrument5c.json" in input_path or "pantry5c.json" in input_path or "music5c.json" in input_path or "office5c.json" in input_path:
        #df = getDF(input_path).values
        #print("amazon shape", df.shape)
        for row in parse_json(input_path):
            user = row["reviewerID"]
            item = row["asin"]
            rating = int(row["overall"])
            sent = " "
            if "reviewText" in row:
                sent+=row["reviewText"]
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

    #loadGloveModel("glove.6B.50d.txt")

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

    for i in range(len(item2reviews)):
        item2reviews[i] = " ".join(item2reviews[i])
    
    for i in range(len(user2reviews)):
        user2reviews[i] = " ".join(user2reviews[i])

    
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
    for i in range(len(item2reviews)):
        item2reviews[i] = " ".join(item2reviews[i])
        item2reviews[i] = [0] * embedding_size
    
    for i in range(len(user2reviews)):
        user2reviews[i] = " ".join(user2reviews[i])
        user2reviews[i] = [0] * embedding_size
    '''
    
        
    '''
    time series glove embeddings
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
    '''

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
    train_split = lambda x : int(x * 0.5)  #default is 0.5
    val_split = lambda  x : int(x * 0.85)
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

    '''
    Pad zeros for unequal length sentences
    item2reviews = np.array(item2reviews)
    zero_words = np.zeros((1,50), dtype=np.float)
    maxlen = max(r.shape[0] for r in item2reviews)
    for r in item2reviews:
        if r.shape[0] < maxlen:
            lent = r.shape[0]
            for _ in range(maxlen - lent):
                r = np.vstack((r, zero_words))
    '''



    '''
    item2reviews1 = tf.ragged.constant(item2reviews)
    item2reviews1 = item2reviews1.to_tensor()
    fin = tf.constant([1])
    ans = tf.nn.embedding_lookup(item2reviews1, fin)
    
    with tf.Session() as sess:
        sess.run(ans)
        print(ans.eval())
    '''
    '''
    with open('data.txt', 'w') as f:
        for item in data:
            f.write("%s\n" % item)
    '''

    scipy.io.savemat(output_path, {'full_matrix':matrix, 'train_matrix':matrix_train, 'user_features':user2reviews1, 'item_features':item2reviews1,
                                   'train_rated_total':train_rated_total, 'train_unrated_total':train_unrated_total,
                                   'val_rated_total': val_rated_total, 'val_unrated_total': val_unrated_total,
                                   'test_rated_total':test_rated_total, 'test_unrated_total':test_unrated_total})
    print('done')