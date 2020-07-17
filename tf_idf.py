""" Create by Ken at 2020 Jun 01 """
import os
from gensim import corpora, models
from pymongo import MongoClient


def load_dict_words():
    dict_words = []
    lines = open('data/id2word.dict').readlines()
    for line in lines:
        dict_words.append(line.strip())
    return dict_words


def sentences_to_words(sentences):
    result = []
    for s in sentences:
        tokens = s.split()
        for token in tokens:
            result.append(token.lower())
    return result


def make_model():
    print('Loading dictionary...')
    dict_words = load_dict_words()
    dictionary = corpora.Dictionary([dict_words])

    print('Loading documents...')
    BoW_corpus = []
    docs = collection.find()
    for doc in docs:
        words = sentences_to_words(doc['title'])
        words += sentences_to_words(doc['summary'])
        words += sentences_to_words(doc['content'])
        BoW_corpus.append(dictionary.doc2bow(words, allow_update=False))

    tfidf = models.TfidfModel(BoW_corpus, smartirs='ntc')
    os.makedirs('models/tf_idf', exist_ok=True)
    tfidf.save('models/tf_idf/tf_idf.model')


def test_model():
    print('Loading dictionary...')
    dict_words = load_dict_words()
    dictionary = corpora.Dictionary([dict_words])

    print('Loading documents...')
    BoW_corpus = []
    docs = collection.find()
    for doc in docs:
        words = sentences_to_words(doc['title'])
        words += sentences_to_words(doc['summary'])
        words += sentences_to_words(doc['content'])
        BoW_corpus.append(dictionary.doc2bow(words, allow_update=False))

    import numpy as np
    saved_model = models.TfidfModel.load('models/tf_idf/tf_idf.model')
    for doc in BoW_corpus[:3]:
        rep = saved_model[doc]
        print([[dictionary[id_], np.around(freq, decimals=2)] for id_, freq in rep])


if __name__ == '__main__':
    mongo_client = MongoClient('localhost', 27017)
    mongo_client.server_info()
    db = mongo_client['KC_01_23']
    collection = db['tokenized']

    make_model()
    # test_model()
