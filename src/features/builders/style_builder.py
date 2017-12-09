from features.base_builder import BaseFeatureBuilder
from features import sentence_features as sf
from util import constants, log

import numpy as np
from scipy import sparse
from time import time

logger = log.logger

class StyleBuilder(BaseFeatureBuilder):
    '''
    This choice of features is
    inspired by http://cs229.stanford.edu/proj2012/CastroLindauer-AuthorIdentificationOnTwitter.pdf

    Features:
        Text length - words and characters
        Word shape - Freq. of Upper/Camel/lower etc. words
        Word length
        Character frequencies - 'a', 'f', 'w', etc.
        Function/Stop word frequencies - 'the', 'of', 'then', etc.
        Twitter token presence - 'RT'

        TODO:
        Punctuation by %
        ...
    '''
    def __init__(self):
        BaseFeatureBuilder.__init__(self)

    def featurize(self, text):
        '''
            Convert a string into a sparse matrix of features
        :param text:
        :return:
        '''
        n_tok = sf.num_tokens(text) # float
        total_features = np.array(n_tok)

        total_features = np.append(total_features, sf.num_chars(text)) # float
        total_features = np.append(total_features, sf.average_word_len(text)) # float
        total_features = np.append(total_features, sf.case_frequency(text, sf.Case.UPPER)) # float
        total_features = np.append(total_features, sf.case_frequency(text, sf.Case.CAPITALISED)) # float
        total_features = np.append(total_features, sf.char_count(text, constants.ALPHABET)) # list of float
        total_features = np.append(total_features, sf.word_count(text, constants.STOP_WORDS)) # list of float
        total_features = np.append(total_features, sf.word_presence(text, "RT"))

        return sparse.csr_matrix(total_features)


    def featurize_all(self, samples):
        logger.info("Building Style Features...")
        t0 = time()

        vf = np.vectorize(lambda s: self.featurize(s))

        s_features = vf(samples)
        logger.debug("Done in %0.3fs" % (time() - t0))

        return sparse.vstack(s_features)