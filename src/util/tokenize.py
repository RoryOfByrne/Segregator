import re
import enchant
import nltk

from util.constants import TWEET_REGEXES, NON_ENGLISH_WORDS

token_reg = re.compile(r'(' + '|'.join(TWEET_REGEXES) + ')', re.VERBOSE | re.IGNORECASE)
d = enchant.Dict("en_UK")

def regex_filter(text, filter):
    out = re.sub(filter, '', text, flags=re.MULTILINE)
    tokens = out.split(" ")
    out_toks = [w for w in tokens if len(w) > 0]

    return ' '.join(out_toks)

def is_useful(text):
    if("thank" in text and len(text) < 30):
        return False

    out = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    tokens = out.split(" ")
    comp_toks = [w for w in tokens if len(w) > 0]
    out_toks = [w for w in comp_toks if (d.check(w) or w[0] == "#")]

    if (len(out_toks) == 0):
        return False

    return True

def filter_words(tokens):
    words = [w for w in tokens if w.lower() not in NON_ENGLISH_WORDS]

    words = [w for w in words if d.check(w)]

    return words

def nltk_tokenize(text):
    '''
    Standard NLTK tokenization
    :param text:
    :return:
    '''
    return nltk.word_tokenize(text)

def tweet_tokenize(tweet):
    '''
    Non-lowercased tokens taking into account hashtags, @tags etc.
    :param tweet:
    :return:
    '''
    toks = token_reg.findall(tweet)
    return toks