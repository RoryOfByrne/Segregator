from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from time import time

from model.base_model import BaseModel
from util import log

logger = log.logger

class KM(BaseModel):
    def __init__(self, feature_builder, true_k):
        BaseModel.__init__(self, feature_builder)
        self.le = LabelEncoder()
        self.model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

    def fit_labels(self, labels):
        logger.info("Fitting labels...")
        self.le.fit(labels)

    def fit(self, X, labels):
        logger.info("Clustering with %s" % self.model)
        t0 = time()

        self.model.fit(X)

        logger.debug("Done in %0.3fs" % (time() - t0))

        logger.info("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, self.model.labels_))
        logger.info("Completeness: %0.3f" % metrics.completeness_score(labels, self.model.labels_))
        logger.info("V-measure: %0.3f" % metrics.v_measure_score(labels, self.model.labels_))
        logger.info("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(labels, self.model.labels_))
        logger.info("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, self.model.labels_, sample_size=1000))

        print()

    def predict(self, predict_x):
        raise NotImplementedError

    def test(self, x_test, y_test):
        raise NotImplementedError
