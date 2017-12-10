import sys
from os.path import dirname
root_path = dirname(dirname(dirname(__file__))) + "/src/"
# Add the root to the path so that import work correctly
sys.path.append(root_path)

from model.classify import svm
from features.builders import style_builder
from util.dump import to_dump, from_dump

def test_to_dump():
    clf = svm.SVM(style_builder.StyleBuilder)

    to_dump(clf, "dump/test.pkl")

    from_dump("dump/test.pkl")

    assert True