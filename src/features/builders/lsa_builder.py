from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from time import time

from util import log
from features .base_builder import BaseFeatureBuilder

logger = log.logger

class LSABuilder(BaseFeatureBuilder):
    '''
        This class is used to reduce the dimensionality of data.

        TODO: Move the `comon_terms()` function out of here
              because I don't think it's relevant to this class.
    '''
    def __init__(self, n_components):
        BaseFeatureBuilder.__init__(self)
        self.n_components = n_components
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        self.svd = TruncatedSVD(n_components)
        self.normalizer = Normalizer(copy=False)
        self.lsa = make_pipeline(self.svd, self.normalizer)

    def __str__(self):
        return "lsa"

    def featurize_all(self, samples):
        '''
            Reduce the dimensionality of X and return it
        :param X:
        :return:
        '''
        logger.info("Performing dimensionality reduction using LSA")
        t0 = time()
        out = self.lsa.fit_transform(samples)

        logger.debug("Done in %fs" % (time() - t0))

        explained_variance = self.svd.explained_variance_ratio_.sum()
        logger.info("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        return out

    def featutize(self, text):
        raise NotImplementedError("Use featurize_all() instead")

    def common_terms(self, model, vectorizer, true_k):
        '''
        Print some info about the most relevant words to each cluster
        :param model:
        :param vectorizer:
        :param true_k:
        :return:
        '''
        original_space_centroids = self.svd.inverse_transform(model.model.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            logger.info("Cluster %d:" % i)
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()
