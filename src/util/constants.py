import logging

###################
###   PROJECT   ###
###################

PROJECT_NAME = "Segregator"
LOG_LEVEL = logging.DEBUG

################
###   DATA   ###
################

DELIMITER = "~"

REAL_TWEETS_DIR = "/home/rory/projects/personify/data/real"
FAKE_TWEETS_FILE = "/home/rory/projects/personify/data/fake/ronald_tweets.csv"

####################
###   FEATURES   ###
####################

WORD_N = (1, 3)
CHAR_N = (4, 6)
N_FEATURES = 10000
URL_REGEX = r'https?:\/\/.*[\r\n]*'

####################
###   PERSONAL   ###
####################

DPI = 220