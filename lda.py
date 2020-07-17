""" Create by Ken at 2020 Jun 01 """
import os
from gensim import corpora, models
from pymongo import MongoClient

ETA = 0.02
ALPHA = 100
NUM_TOPICS = 100


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

    lda_model = models.LdaModel(corpus=BoW_corpus,
                                id2word=dictionary,
                                num_topics=NUM_TOPICS,
                                random_state=100,
                                update_every=1,
                                chunksize=100,
                                passes=10,
                                alpha='auto',
                                eta='auto',
                                per_word_topics=True)

    os.makedirs('models/lda', exist_ok=True)
    lda_model.save('models/lda/lda.model')


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

    lda_model = models.LdaModel.load('models/lda/lda.model')
    for doc in BoW_corpus[:3]:
        rep = lda_model[doc]
        print(rep)


if __name__ == '__main__':
    mongo_client = MongoClient('localhost', 27017)
    mongo_client.server_info()
    db = mongo_client['KC_01_23']
    collection = db['tokenized']

    make_model()
    # test_model()
