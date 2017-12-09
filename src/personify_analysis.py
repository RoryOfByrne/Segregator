import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from data import tweets, preprocessing
from features.builders.lsa_builder import LSABuilder
from features.builders.ngram_builder import NGramBuilder
from features.builders.style_builder import StyleBuilder
from model.cluster.mb_kmeans import MB_KM
from model.classify.svm import SVM
from util import constants, log
from util.graphing import graph_3d

logger = log.logger

REAL_TWEETS_DIR = "/home/rory/projects/personify/data/real"

def parse_args():
    '''
        Basic argument parser. Additions here need to be reflected in
        config.settings also.
    '''
    parser = argparse.ArgumentParser(description='''Cluster and Visualise Text Data''')
    parser.add_argument('--lsa-components',
                        help="Dimensionality for Latent Semantic Analysis (ngram model)",
                        type=int,
                        default=3,
                        metavar="LSA")
    parser.add_argument('--algorithms', '-a',
                        help="Algorithms to use: 'cluster', 'classify', or 'both'",
                        type=str,
                        default='classify',
                        metavar='ALG')
    parser.add_argument('--num-samples',
                        help='The number of samples to use from each csv file',
                        type=int,
                        default=100,
                        metavar="NUM_SAMPLES")
    parser.add_argument("--model", "-m",
                        help="The model type - 'ngram' or 'style'",
                        type=str,
                        default="style",
                        metavar="model")
    parser.add_argument('--n-features',
                        help="Maximum number of features for (ngram model)",
                        type=int,
                        default=5000,
                        metavar="N_FEATURES")
    parser.add_argument('--produce-graphs', '-g',
                        action='store_true',
                        help="Specify whether or not to produce graphs",
                        default=False)
    parser.add_argument("-c", "--config-file",
                        help="Specify config file, default is config.json",
                        default="config.json",
                        metavar="FILE")

    return parser.parse_args()


def classify(X, Y, x_new, y_new):
    '''
    X and labels are both pandas dataframes and need to be manipulated before being send into the model
    :param X:
    :param labels:
    :return:
    '''
    fb = StyleBuilder()

    X = fb.featurize_all(X.values.ravel())
    Y = Y.values.reshape((-1))
    x_new = fb.featurize_all(x_new.values.ravel())
    y_new = y_new.values.reshape((-1))

    # X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    classifier = SVM(fb)
    classifier.fit(X, Y)

    classifier.test(x_new, y_new)

def cluster(X, labels):
    '''
        Cluster the data using minibatch k-means
    :param X:
    :param labels:
    :return:
    '''
    # We know the real labels, so we can get the real number of unique labels
    real_k = len(pd.Series(labels).unique())
    logger.debug("Real k: %s" % real_k)

    # Construct features
    if(opts.model == 'ngram'):
        feature_builder = NGramBuilder(X, constants.WORD_N, constants.CHAR_N, opts.n_features)
    else:
        feature_builder = StyleBuilder()

    X = feature_builder.featurize_all(X.values.ravel())
    logger.debug("n_samples: %d, n_features: %d" % X.shape)

    # Reduce domensionality (Curse of Dimensionality)
    reducer = LSABuilder(opts.lsa_components)
    X = reducer.featurize_all(X)
    logger.debug("After LSA shape: %d samples, %d features" % X.shape)

    # Create the model and fit the data
    model = MB_KM(feature_builder, real_k, 10)
    model.fit(X, labels.reshape((-1)))

    if(opts.produce_graphs):
        graph_3d.plot_cluster(model, X, opts.num_samples, opts.model)
        graph_3d.plot_truth(X, labels.reshape((-1)), model.__str__(), opts.num_samples, opts.model)

    if(opts.model == 'ngram'):
        # This is only possible with vocabulary-based feature builders
        reducer.common_terms(model, model.feature_builder.word_pres_vec, real_k)


def main():
    '''
        Cluster the samples together and produce graphs to show the result

    :return:
    '''

    logger.info("Produce graphs: %s" % opts.produce_graphs)
    logger.info("Using model %s" % opts.model)
    logger.info("Using algorithm(s): %s" % opts.algorithms)

    real_samples = tweets.all_tweets(REAL_TWEETS_DIR, "~", opts.num_samples)
    fake_samples = tweets.load_csv(constants.TRAINING_FILE, "~", opts.num_samples)

    # Filter unwanted rows and remove unwanted tokens
    real_samples = preprocessing.clean(real_samples)
    fake_tweets = preprocessing.clean(fake_samples)

    # Map string labels to numbers (0, 1, 2, etc.)
    y_numbers = real_samples['label'].astype('category').cat.codes.values
    logger.debug("Number of coded labels: %s" % len(y_numbers))

    Y = real_samples[['label']]
    y_test = fake_tweets[['label']]
    y_test['label'] = "realdonaldtrump"

    # Create features from samples
    X = real_samples.drop(['label'], axis=1)
    x_test = fake_tweets.drop(['label'], axis=1)

    if(opts.algorithms == 'both'):
        classify(X, Y, x_test, y_test)
        cluster(X, y_numbers)
    elif(opts.algorithms == 'cluster'):
        cluster(X, y_numbers)
    elif(opts.algorithms == 'classify'):
        classify(X, Y, x_test, y_test)
    else:
        raise NameError("Wrong algorithm selected via --algorithm")

if __name__ == "__main__":
    # execute only if run as a script
    opts = parse_args()
    main()

