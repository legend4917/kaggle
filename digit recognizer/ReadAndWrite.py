# coding: utf8
import csv


def load_train_data(file_path):    # 读取训练数据集
    label = []    # 训练数据集的标签
    data = []  # 训练数据集的特征
    csvfile = file(file_path, 'rb')
    csvfile.readline()
    reader = csv.reader(csvfile)
    for line in reader:
        label.append(int(line[0]))
        data.append(map(lambda x: 0 if int(x) == 0 else 1, line[1:]))
    csvfile.close()
    return data, label


def load_test_data():   # 读取测试数据集
    test_feature = []
    csvfile = file('test.csv', 'rb')
    csvfile.readline()
    reader = csv.reader(csvfile)
    for line in reader:
        test_feature.append(map(lambda x: 0 if int(x) == 0 else 1, line))
    csvfile.close()
    return test_feature


# 将结果打印到csv文件
def write_result(test_label):
    result = [['ImageId', 'Label']]
    for i in range(len(test_label)):
        result.append([i+1, test_label[i]])
    csvfile = file('result.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(result)
    csvfile.close()