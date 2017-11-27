from optparse import OptionParser

import numpy as np
import pandas as pd
import sys
from features.feature_builder import FeatureBuilder
from features import dimensionality_reduction
from model.mb_kmeans import MB_KM
from util import constants, log
from util.graphing import graph_3d
from data import tweets, preprocessing

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

logger = log.logger

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int", default=3,
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

def cluster():
    '''
        Cluster the samples together and produce graphs to show the result

    :return:
    '''
    total_samples = tweets.all_tweets()

    # Filter unwanted rows and remove unwanted tokens
    total_samples = preprocessing.clean(total_samples)

    labels = total_samples[['label']]
    logger.debug("Labels shape: (%s, %s)" % labels.shape)
    real_k = len(labels['label'].unique())

    # Create features from samples
    X = total_samples.drop(['label', 'real'], axis=1)

    feature_builder = FeatureBuilder(X, constants.WORD_N, constants.CHAR_N)
    X = feature_builder.featurize_all(X)

    logger.debug("n_samples: %d, n_features: %d" % X.shape)

    # Reduce domensionality (Curse of Dimensionality)
    reducer = dimensionality_reduction.LSA(opts.n_components)
    X = reducer.reduce(X)

    logger.debug("After LSA shape: %d samples, %d features" % X.shape)

    # Create the model and fit the data
    model = MB_KM(feature_builder, real_k, 10)
    model.fit(X, labels.values.reshape((-1)))

    graph_3d.plot_cluster(model, X)
    graph_3d.plot_truth(X, labels.values.reshape((-1)))

    reducer.common_terms(model, model.feature_builder.word_pres_vec, real_k)

cluster()

