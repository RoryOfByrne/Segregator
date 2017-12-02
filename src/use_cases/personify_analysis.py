import argparse
import pandas as pd
import sys

from features.ngram_builder import NGramBuilder
from features import dimensionality_reduction
from model.mb_kmeans import MB_KM
from util import constants, log
from util.graphing import graph_3d
from data import tweets, preprocessing

logger = log.logger

REAL_TWEETS_DIR = "/home/rory/projects/personify/data/real"

def parse_args():
    '''
        Basic argument parser. Additions here need to be reflected in
        config.settings also.
    '''
    parser = argparse.ArgumentParser(description='''Cluster and Visualise Text Data''')
    parser.add_argument('--lsa-components',
                        help="Dimensionality for preprocessing documents with latent semantic analysis",
                        type=int,
                        default=3,
                        metavar="LSA")
    parser.add_argument('--n-features',
                        help="Maximum number of features (dimensions) to extract from text",
                        type=int,
                        default=5000,
                        metavar="N_FEATURES")
    parser.add_argument('--produce-graphs', '-g',
                        help="Specify whether or not to produce graphs",
                        default=False,
                        metavar="GRAPHS")
    parser.add_argument("-c", "--config-file",
                        help="Specify config file, default is config.json",
                        default="config.json",
                        metavar="FILE")

    return parser.parse_args()

def all_tweets(dir, delim):
    real = tweets.load_directory(dir, delim)

    return pd.concat(real)

def cluster():
    '''
        Cluster the samples together and produce graphs to show the result

    :return:
    '''
    opts = parse_args()
    logger.info("Produce graphs: %s" % opts.produce_graphs)

    total_samples = tweets.all_tweets(REAL_TWEETS_DIR, "~")

    # Filter unwanted rows and remove unwanted tokens
    total_samples = preprocessing.clean(total_samples)

    labels = total_samples[['label']]
    logger.debug("Labels shape: (%s, %s)" % labels.shape)
    real_k = len(labels['label'].unique())

    # Create features from samples
    X = total_samples.drop(['label'], axis=1)
    feature_builder = NGramBuilder(X, constants.WORD_N, constants.CHAR_N, opts.n_features)
    X = feature_builder.featurize_all(X)

    logger.debug("n_samples: %d, n_features: %d" % X.shape)

    # Reduce domensionality (Curse of Dimensionality)
    reducer = dimensionality_reduction.LSA(opts.lsa_components)
    X = reducer.reduce(X)

    logger.debug("After LSA shape: %d samples, %d features" % X.shape)

    # Create the model and fit the data
    model = MB_KM(feature_builder, real_k, 10)
    model.fit(X, labels.values.reshape((-1)))

    if(opts.produce_graphs):
        graph_3d.plot_cluster(model, X)
        graph_3d.plot_truth(X, labels.values.reshape((-1)))

    reducer.common_terms(model, model.feature_builder.word_pres_vec, real_k)

def main():
    cluster()

if __name__ == "__main__":
    # execute only if run as a script
    main()

