# coding: utf8
from ReadAndWrite import *
from sklearn.svm import SVC


# 使用支持向量机SVM进行训练和预测
def svm_predict(train_feature, train_label):
    clf = SVC()
    clf.fit(train_feature, train_label)
    test_feature = load_test_data()
    test_label = clf.predict(test_feature)
    return test_label


if __name__ == '__main__':
    train_feature, train_label = load_train_data('train.csv')
    test_label = svm_predict(train_feature, train_label)
    write_result(test_label)
