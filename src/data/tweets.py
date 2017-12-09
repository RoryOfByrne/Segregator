import os
import pandas as pd

from util import constants
from util import log

logger = log.logger

def load_csv(file, delim, num_samples):
    data = pd.read_csv(file, delimiter=delim).sample(num_samples)
    data = data[['text']]

    # Files are expected in the form name_tweets.csv
    # We want to chop this down to name
    filename = file.split('/')[-1]
    label = filename[:-11]

    data['label'] = label
    data.columns = ['text', 'label']
    logger.info("Number of samples for %s: %d" % (label, data.shape[0]))

    return data

def load_directory(path, delim, num_samples):
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
        samples = load_csv("%s/%s" % (path, f), delim, num_samples)

        data.append(samples)

    return data


def all_tweets(dir, delim, num_samples):
    real = load_directory(dir, delim, num_samples)

    return pd.concat(real)