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