import logging
import nltk

###################
###   PROJECT   ###
###################

PROJECT_NAME = "Segregator"
LOG_LEVEL = logging.DEBUG

################
###   DATA   ###
################

DELIMITER = "~"
REAL_TWEETS_DIR = "data/real"
TRAINING_FILE = "data/fake/donaldonumber9_tweets.csv"

####################
###   FEATURES   ###
####################

WORD_N = (1, 3)
CHAR_N = (4, 6)
N_FEATURES = 10000
URL_REGEX = r'https?:\/\/.*[\r\n]*'

FRONT_PUNCTUATION = [',', '.', '?', '!', ':', ';']
USELESS_PUNCTUATION = ['(', ')', '"', '\'', '^', '@', '-', '_', '*', '&', '$', '+', '=']
NON_ENGLISH_WORDS = ['rt', 'amp', 'Rt']

TWITTER_TOKENS = ['RT']

SENTENCE_ENDERS = ['.', '!', '?']

EMOTICON_REGEX= r"""
            (?:
                [:=;]
                [oO\-]?
                [D\)\]\(\]/\\OpP]
            )"""  # eyes, nose, mouth (in that order)

TWEET_REGEXES = [
    EMOTICON_REGEX,
    r'(?:@[\w_]+)',  # @ tag
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hashtag
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # url
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
STOP_WORDS = list(nltk.corpus.stopwords.words('english'))

####################
###### OUTPUT ######
####################

CLUSTER_DIR = "cluster/"
CLASSIFY_DIR = "classify/"
GRAPH_DIR = "graph/"

####################
###   PERSONAL   ###
####################

DPI = 220