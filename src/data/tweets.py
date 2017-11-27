import os
import pandas as pd

from util import constants
from util import log

logger = log.logger

def map_labels(label):
    if(label == "realdonaldtrump"):
        return 0
    elif(label == "markhumphrys"):
        return 1
    elif(label == "barackobama"):
        return 2

def load_csv(file, delim):
    data = pd.read_csv(file, delimiter=delim).sample(100)
    data = data[['text']]

    # Files are expected in the form name_tweets.csv
    # We want to chop this down to name
    filename = file.split('/')[-1]
    label = filename[:-11]
    label = map_labels(label)
    logger.debug("Label for file %s: %s" % (filename, label))

    data['label'] = label
    data.columns = ['text', 'label']

    return data

def load_directory(path, delim):
    '''
        Creates a list of DataFrames, one for each file in the directory.
        DataFrames have the form:

                text                label
                "..."               "..."
                "..."               "..."

    :param path:
    :param delim:
    :return data:
    '''
    files = os.listdir(path)
    data = []

    for f in files:
        # TODO: Strip trailing '/' if it exists in path
        samples = load_csv("%s/%s" % (path, f), delim)

        data.append(samples)

    return data


def all_tweets(dir, delim):
    real = load_directory(dir, delim)

    return pd.concat(real)