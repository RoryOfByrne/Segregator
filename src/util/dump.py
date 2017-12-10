from sklearn.externals import joblib
import util.log

logger = util.log.logger

def to_dump(model, file):
    logger.info("Dumping %s to %s" % (model, file))
    joblib.dump(model, file)

def from_dump(file):
    logger.info("Loading dump from %s" % file)
    return joblib.load(file)
