from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
import numpy as np
from scipy import sparse
from time import time

from features import sentence_features
from util import log
from features.base_builder import BaseFeatureBuilder

logger = log.logger

class NGramBuilder(BaseFeatureBuilder):
    '''
    This class will turn a piece of text into a vocabulary-dependent set of features

    `data_df` is the pandas DataFrame which will be used to construct the vocabulary.
    It should have the form

                text
                "some text"
                "more text"
                ...

    The text in the `text` column is used to construct a vocabulary of ngram features for
    both characters and words.
    The vocabulary will be used to construct a vector whose length is the sum of the lengths of
    all the ngram vocabularies for that type {char, word}
    '''
    def __init__(self, data_df, word_range, char_range, n_features):
        BaseFeatureBuilder.__init__(self)
        self.word_range = word_range
        self.char_range = char_range
        self.n_features = n_features
        self.word_pres_vec, self.char_count_vec = self.build_vocabs(data_df)
        self.hash_tfidf = self.build_hash_vec() # Unused currently

    def build_hash_vec(self):
        '''
            Hashing Vectors are useful because they dont need to store a vocab
            meaning that they require a lot less memory.
            Hashing is NOT currently being used but it is implemented

            More info:
            http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
        :return:
        '''
        hasher = HashingVectorizer(n_features=self.n_features,
                                   stop_words='english', alternate_sign=False,
                                   norm=None, binary=False)
        return make_pipeline(hasher, TfidfTransformer())

    def build_vocabs(self, df_data):
        '''
            Use the dataframe to construct a `CountVectorizer` for char ngrams, and
            word-presence `CountVectorizer` for words.

        :param df_data:
        :return:
        '''
        logger.info("Building CountVectorizers")
        t0 = time()
        vocab_vect_word = CountVectorizer(ngram_range=self.word_range, binary=True)
        vocab_vect_word.fit_transform(df_data['text'])
        vocab_word = vocab_vect_word.vocabulary_
        # logger.debug("Word Vocab size: %s" % len(vocab_word))

        word_pres_vec = CountVectorizer(ngram_range=self.word_range, max_features=self.n_features,
                                        vocabulary=vocab_word)

        vocab_vect_char = CountVectorizer(ngram_range=self.char_range, analyzer='char')
        vocab_vect_char.fit_transform(df_data['text'])
        vocab_char = vocab_vect_char.vocabulary_

        char_count_vec = CountVectorizer(ngram_range=self.char_range, max_features=self.n_features,
                                              vocabulary=vocab_char, analyzer='char', binary=False)

        logger.debug("Done in %0.3fs" % (time() - t0))

        del vocab_vect_word
        del vocab_vect_char

        return word_pres_vec, char_count_vec

    def featurize(self, text):
        '''
            Turn a piece of text into the set of features
        :param text:
        :return:
        '''
        s_feats = sentence_features.sentence_features(text)
        char_counts = self.count_chars(text)
        word_presence = self.word_presence(text)

        ngram_features = np.concatenate((word_presence, char_counts))
        total_features = np.insert(word_presence, 0, list(s_feats))

        return sparse.csr_matrix(total_features)

    def featurize_all(self, samples):
        '''
            Turn all samples into features.
            Samples is expected as a pandas DataFrame
        :param samples:
        :return:
        '''
        logger.info("Building Features...")
        t0 = time()

        vf = np.vectorize(lambda s: self.featurize(s))

        s_features = vf(samples)
        logger.debug("Done in %0.3fs" % (time() - t0))

        return sparse.vstack(s_features)

    def count_chars(self, text):
        '''
            Generate the ngram count vector, which is a vector of real values indicating
            the count of a char ngram in the sample text
        :param data:
        :return:
        '''
        return self.char_count_vec.fit_transform([text]).toarray()[0]


    def word_presence(self, text):
        '''
            Generate the word_presence_vector, which is a V long vector of binary values indicating
            whether or not a word is present
        :param text_col:
        :return:
        '''
        return self.word_pres_vec.fit_transform([text]).toarray()[0]