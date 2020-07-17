""" Create by Ken at 2020 Jul 17 """
from pymongo import MongoClient

from constants import *

client = MongoClient('localhost', 27017)
db = client['KC_01_23']
tf_idf_collection = db['tf_idf_representation']
lda_collection = db['lda_representation']
word2vec_collection = db['word2vec_representation']

tf_idf_collection.ensure_index('doc_name', unique=True)
lda_collection.ensure_index('doc_name', unique=True)
word2vec_collection.ensure_index('doc_name', unique=True)


def _switch_collection(model_type):
    if model_type == MODEL_TYPE_TF_IDF:
        return tf_idf_collection

    if model_type == MODEL_TYPE_LDA:
        return lda_collection

    if model_type == MODEL_TYPE_WORD2VEC:
        return word2vec_collection

    return None


def check_doc_name(model_type, doc_name):
    collection = _switch_collection(model_type)
    record = collection.find_one({'doc_name': doc_name})
    return record is None


def insert_representation(model_type, doc_name, representation):
    collection = _switch_collection(model_type)
    if collection is None:
        return 'Cannot found collection'

    collection.insert_one({
        'doc_name': doc_name,
        'representation': representation
    })


def find_representation(model_type, doc_name):
    collection = _switch_collection(model_type)
    if collection is None:
        return f"Mode type '{model_type}' is not supported!", None

    record = collection.find_one({'doc_name': doc_name})
    if record is not None:
        rep = record['representation']
    else:
        rep = None
    return None, rep
