from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from time import time

from util import log

logger = log.logger

class LSA():
    def __init__(self, n_components):
        self.n_components = n_components
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        self.svd = TruncatedSVD(n_components)
        self.normalizer = Normalizer(copy=False)
        self.lsa = make_pipeline(self.svd, self.normalizer)

    def reduce(self, X):
        logger.info("Performing dimensionality reduction using LSA")
        t0 = time()
        out = self.lsa.fit_transform(X)

        logger.debug("Done in %fs" % (time() - t0))

        explained_variance = self.svd.explained_variance_ratio_.sum()
        logger.info("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        return out

    def common_terms(self, model, vectorizer, true_k):
        original_space_centroids = self.svd.inverse_transform(model.model.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            logger.info("Cluster %d:" % i)
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()
