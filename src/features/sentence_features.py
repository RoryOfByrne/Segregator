import nltk
from util import log


logger = log.logger

'''
Features:
    - Num tokens
    - Average word length
    - log of the freq. of each character n-gram, normalized wrt text length.
        Smoothing term added to avoid 0 values for n-grams with 1 freq.
    - Word n-grams (presence or absence)
'''

####################################
###     Independent Features     ###
####################################
def num_tokens(text):
    return len(nltk.word_tokenize(text))

def average_word_len(text):
    '''
    Get the sum of the word lengths and divide by the number of words
    :param text:
    :return:
    '''
    word_list = nltk.word_tokenize(text)
    num_words = len(word_list)

    return sum([len(s) for s in word_list]) / num_words

def sentence_features(text):
    num_tok = num_tokens(text)
    avg_len = average_word_len(text)

    return (num_tok, avg_len)