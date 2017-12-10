from util import log, dump, constants

class BaseModel():
    def __init__(self, feature_builder):
        self.feature_builder = feature_builder

    def fit(self, X, labels):
        raise NotImplementedError

    def predict(self, predict_x):
        raise NotImplementedError

    def test(self, x_test, y_test):
        raise NotImplementedError

    def save(self):
        dump.to_dump(self, constants.DUMP + self.__str__() + "-" + "test" + ".pkl")

