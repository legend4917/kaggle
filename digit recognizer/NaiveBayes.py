# coding: utf8
from ReadAndWrite import *
import numpy as np


# 统计朴素贝叶斯公式计算所需要的概率值,包括标签的先验概率以及条件概率
# train_data: 训练数据集的特征
# train_label: 训练数据集的标签
def naive_bayes_calculate(train_data, train_label):
    poss_class = []     # 标签的先验概率,这里使用贝叶斯估计
    poss_feature = []    # 条件概率,这里使用贝叶斯估计
    data_num = float(len(train_label))
    for i in range(10):     # 总共是0-9共10个标签
        label_number = train_label.count(i)     # 统计标签i出现的次数
        poss_class.append(float(label_number+1) / (data_num+10))    # 拉普拉斯平滑
        train_temp = []
        for j in range(len(train_label)):   # 将与标签i相对应的特征保存到train_temp变量中
            if i == train_label[j]:
                train_temp.append(train_data[j])
        train_temp = np.array(train_temp)
        poss_feature_temp = []
        for j in range(train_temp.shape[1]):    # 遍历所有属性(即数组train_temp的列),并统计每列0和1出现的次数,计算条件概率
            a = [0, 0]
            for k in range(label_number):
                if train_temp[k, j] == 0:
                    a[0] += 1
                else:
                    a[1] += 1
            a[0] = float(a[0]+1) / (label_number+2)
            a[1] = float(a[1]+1) / (label_number+2)
            poss_feature_temp.append(a)
        poss_feature.append(poss_feature_temp)  # 结构: 外层是标签0-9,每个标签对应的所有特征,每个特征对应的所有取值概率
    return poss_class, poss_feature


# 根据训练好的朴素贝叶斯模型计算测试数据集的标签
def naive_bayes_test(poss_class, poss_feature):
    test_feature = load_test_data()
    test_label = []
    for feature in test_feature:
        max_poss = 0
        max_class = 0
        for i in range(10):
            poss = poss_class[i]
            for j in range(len(feature)):
                poss *= poss_feature[i][j][feature[j]]  # 计算后验概率
            if max_poss < poss:     # 记录后验概率最大的标签
                max_poss = poss
                max_class = i
        test_label.append(max_class)
    return test_label


# 使用支持向量机SVM进行训练和预测
def svm_predict(train_feature, train_label):
    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(train_feature, train_label)
    test_feature = load_test_data()
    test_label = clf.predict(test_feature)
    write_result(test_label)


if __name__ == '__main__':
    train_feature, train_label = load_train_data('train.csv')
    poss_class, poss_feature = naive_bayes_calculate(train_feature, train_label)
    test_label = naive_bayes_test(poss_class, poss_feature)

    # SVM方法
    svm_predict(train_feature, train_label)

