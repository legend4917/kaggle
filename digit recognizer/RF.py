# coding: utf8

from ReadAndWrite import *
from sklearn.ensemble import RandomForestClassifier


def RF(train_label, train_feature, test_feature):
    clf = RandomForestClassifier(n_estimators=1000, min_samples_leaf=5)
    clf.fit(train_feature, train_label)
    test_label = clf.predict(test_feature)
    write_result(test_label)


if __name__ == '__main__':
    train_feature, train_label = load_train_data('train.csv')
    test_feature = load_test_data()
    RF(train_label, train_feature, test_feature)