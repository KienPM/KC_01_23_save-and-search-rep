""" Create by Ken at 2020 Jul 15 """
import os
import sys
import argparse
import numpy as np
from pymongo import MongoClient

from gensim import corpora, models
from vncorenlp import VnCoreNLP
from underthesea import sent_tokenize

client = MongoClient('localhost', 27017)
db = client['KC_01_23']
tf_idf_collection = db['tf_idf_representation']
lda_collection = db['lda_representation']
word2vec_collection = db['word2vec_representation']

tf_idf_collection.ensure_index('ten_vb', unique=True)
lda_collection.ensure_index('ten_vb', unique=True)
word2vec_collection.ensure_index('ten_vb', unique=True)


def load_dict_words():
    dict_words = []
    lines = open('data/id2word.dict').readlines()
    for line in lines:
        dict_words.append(line.strip())
    return dict_words


def doc2words(file):
    f = open(file)
    text = f.read()
    f.close()

    doc_words = []
    for line in text.split('\n'):
        sentences = sent_tokenize(line)
        for s in sentences:
            sent_words = annotator.tokenize(s)[0]
            for w in sent_words:
                doc_words.append(w.lower())

    return doc_words


def get_doc_name(input_file):
    return os.path.splitext(os.path.basename(input_file))[0]


def check_doc_name(collection, doc_name):
    record = collection.find_one({'ten_vb': doc_name})
    return record is None


def tf_idf_doc_embedding(input_file):
    doc_name = get_doc_name(input_file)
    if not check_doc_name(tf_idf_collection, doc_name):
        print('Document name already exists')
        return

    print('TF-IDF')
    print('Loading model...')
    model = models.TfidfModel.load('models/tf_idf/tf_idf.model')
    doc_words = doc2words(input_file)
    bow = dictionary.doc2bow(doc_words, allow_update=False)
    rep = model[bow]

    print(f'Saving to db...')
    tf_idf_collection.insert_one({
        'ten_vb': doc_name,
        'bieu_dien': rep
    })
    print('Done!')


def lda_doc_embedding(input_file):
    doc_name = get_doc_name(input_file)
    if not check_doc_name(lda_collection, doc_name):
        print('Document name already exists')
        return

    print('LDA')
    print('Loading model...')
    model = models.LdaModel.load('models/lda/lda.model')
    print(f'Number of topics: {model.num_topics}')
    doc_words = doc2words(input_file)
    bow = dictionary.doc2bow(doc_words, allow_update=False)
    rep = ['0'] * model.num_topics
    lda_output = model[bow]
    for item in lda_output[0]:
        rep[item[0]] = str(item[1])

    print(f'Saving to db...')
    lda_collection.insert_one({
        'ten_vb': doc_name,
        'bieu_dien': rep
    })
    print('Done!')


def word2vec_doc_embedding(input_file):
    doc_name = get_doc_name(input_file)
    if not check_doc_name(word2vec_collection, doc_name):
        print('Document name already exists')
        return

    print('word2vec')
    print('Loading model...')
    model = models.Word2Vec.load('models/word2vec/word2vec.model')
    vector_size = model.vector_size
    print(f'Vector size: {vector_size}')
    mat = []
    doc_words = doc2words(input_file)
    word_vectors = model.wv
    for w in doc_words:
        if w in word_vectors:
            mat.append(word_vectors[w])
        else:
            mat.append([0] * vector_size)

    mat = np.array(mat)
    rep = np.sum(mat, axis=0)
    rep /= mat.shape[0]
    rep = [str(item) for item in rep]

    print(f'Saving to db...')
    word2vec_collection.insert_one({
        'ten_vb': doc_name,
        'bieu_dien': rep
    })
    print('Done!')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Document embedding')
    arg_parser.add_argument(
        '--input_file',
        type=str,
        help='Path to input file'
    )
    arg_parser.add_argument(
        '--model',
        type=str,
        default='tf-idf',
        help='Model to be used (tf-idf | lda | word2vec)'
    )
    args = arg_parser.parse_args()
    input_file = args.input_file
    model_type = args.model

    if input_file is None:
        print('You have to specify input file with (e.g. python doc_embedding.py --input_file doc.txt)')
        print('Use -h or --help for more options')
        sys.exit(0)
    if not os.path.exists(input_file):
        print('Input file does not exists')
        sys.exit(0)

    annotator = VnCoreNLP(address="http://172.27.169.164", port=9000)
    print('Loading dictionary...')
    dict_words = load_dict_words()
    dictionary = corpora.Dictionary([dict_words])

    if model_type == 'tf-idf':
        tf_idf_doc_embedding(input_file)
    elif model_type == 'lda':
        lda_doc_embedding(input_file)
    elif model_type == 'word2vec':
        word2vec_doc_embedding(input_file)
    else:
        print(f'{model_type} is not supported! (tf-idf | lda | word2vec)')
