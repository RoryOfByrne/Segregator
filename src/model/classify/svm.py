from model.base_model import BaseModel
from util import log

logger = log.logger

from sklearn import svm

class SVM(BaseModel):
    def __init__(self, feature_builder):
        BaseModel.__init__(self, feature_builder)
        self.model = svm.SVC(probability=True)

    def fit(self, X, labels):
        self.model.fit(X, labels)

    def predict(self, predict_x):
        logger.info("Predicting...")
        prediction = self.model.predict(predict_x)

        return prediction

    def test(self, x_test, y_test):
        logger.info("Testing")
        prediction = self.model.score(x_test, y_test)
        logger.info("Testing score: %s" % prediction)

    def __str__(self):
        return "svm"
