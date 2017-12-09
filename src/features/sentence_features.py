import collections
from enum import Enum
import nltk
from util import log
from util import tokenize


logger = log.logger

'''
Features:
    - Num tokens
    - Num chars
    - Average word length
    - Frequency of words with UPPER/lower/CamelCase
    - Histogram of word lengths 1-20
    - Frequency of characters a-z
    - Frequency of non-ASCII characters (Unicode)
    - Frequency of function/stop words
    - Existence of "RT" and "MT"
'''

class Case(Enum):
    UPPER = 1
    LOWER = 2
    CAPITALISED = 3

########################
###     Features     ###
########################

def num_tokens(text):
    '''
        The number of tokens in the text
    :param text:
    :return:
    '''
    return len(tokenize.nltk_tokenize(text))


def num_chars(text):
    '''
        The number of characters in the text
    :param text:
    :return:
    '''
    return len(text)


def average_word_len(text):
    '''
        Get the sum of the word lengths and divide by the number of words
    :param text:
    :return:
    '''

    word_list = nltk.word_tokenize(text)
    num_words = len(word_list)

    return sum([len(s) for s in word_list]) / num_words


def case_frequency(text, case):
    '''
        Given a case enum, return the number of words in the text which match that case
    :param text:
    :param case:
    :return:
    '''

    words = tokenize.nltk_tokenize(text)
    if(case is Case.UPPER):
        return len([word for word in words if word.isupper()])
    if(case is Case.LOWER):
        return len([word for word in words if word.islower()])
    if(case is case.CAPITALISED):
        return len([word for word in words if word[0].isupper()])


def word_lengths(text, length_range):
    '''
        Generates a histogram of the word lenghts defined by `range`
    :param text:
    :param range:
    :return:
    '''

    tokens = tokenize.nltk_tokenize(text)
    lengths = []

    for n in range(length_range[0], length_range[1] + 1):
        filtered = [w for w in tokens if len(w) == n]
        lengths.append((n, len(filtered)))

    return lengths


def char_count(text, chars):
    counts = []
    for c in chars:
        filtered = [l for l in text if l is c]
        counts.append(len(filtered))

    return counts


def word_count(text, target_words):
    words = tokenize.nltk_tokenize(text.lower())
    counts = [words.count(t) for t in target_words]

    return counts


def word_presence(text, word):
    if(word in text):
        return 1
    else:
        return 0


def sentence_features(text):
    num_tok = num_tokens(text)
    avg_len = average_word_len(text)

    return (num_tok, avg_len)
