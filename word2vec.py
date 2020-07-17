""" Create by Ken at 2020 Jun 01 """
import os
from gensim import models, utils
from pymongo import MongoClient
import logging
from tqdm import tqdm


def prepare_data(collection):
    print('Prepare data...')
    docs = collection.find()
    for doc in tqdm(list(docs)):
        sentences = doc['title']
        sentences += doc['summary']
        sentences += doc['content']
        for s in sentences:
            yield utils.simple_preprocess(s)


def make_model():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    mongo_client = MongoClient('localhost', 27017)
    mongo_client.server_info()
    db = mongo_client['KC_01_23']
    collection = db['tokenized']
    sentences = list(prepare_data(collection))

    model = models.Word2Vec(sentences, size=300, window=2, min_count=1, workers=10, iter=10)

    os.makedirs('models/word2vec', exist_ok=True)
    model.save('models/word2vec/word2vec.model')


def test_model():
    model = models.Word2Vec.load('models/word2vec/word2vec.model')
    word_vectors = model.wv
    print(word_vectors['các'])


if __name__ == '__main__':
    make_model()
    # test_model()
