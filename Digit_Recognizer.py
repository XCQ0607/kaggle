import os
import csv
import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# import os
# #当前目录为根目录
# data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# print(data_dir)
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


# 加载数据
def opencsv():
    # 使用 pandas 打开
    data = pd.read_csv('data/digit_recognizer/train.csv')
    data1 = pd.read_csv('data/digit_recognizer/test.csv')

    train_data = data.values[0:, 1:]  # 读入全部训练数据,  [行，列]
    train_label = data.values[0:, 0]  # 读取列表的第一列
    test_data = data1.values[0:, 0:]  # 测试全部测试个数据
    return train_data, train_label, test_data


# 加载数据
trainData, trainLabel, testData = opencsv()

# 数据预处理-降维 PCA主成成分分析
def dRPCA(x_train, x_test, COMPONENT_NUM):
    print('dimensionality reduction...')
    trainData = np.array(x_train)
    testData = np.array(x_test)
    '''
    使用说明：https://www.cnblogs.com/pinard/p/6243025.html
    n_components>=1
      n_components=NUM   设置占特征数量比
    0 < n_components < 1
      n_components=0.99  设置阈值总方差占比
    '''
    pca = PCA(n_components=COMPONENT_NUM, whiten=False)
    pca.fit(trainData)  # Fit the model with X
    pcaTrainData = pca.transform(trainData)  # Fit the model with X and 在X上完成降维.
    pcaTestData = pca.transform(testData)  # Fit the model with X and 在X上完成降维.

    # pca 方差大小、方差占比、特征数量
    # print("方差大小:\n", pca.explained_variance_, "方差占比:\n", pca.explained_variance_ratio_)
    print("特征数量: %s" % pca.n_components_)
    print("总方差占比: %s" % sum(pca.explained_variance_ratio_))
    return pcaTrainData, pcaTestData


# 降维处理
trainDataPCA, testDataPCA = dRPCA(trainData, testData, 0.8)

#KNN
def trainModel1(trainData, trainLabel):
    clf = KNeighborsClassifier()  # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    clf.set_params(
        algorithm='auto',
        leaf_size=30,
        metric='minkowski',
        p=2,
        weights='uniform')
    clf.fit(trainData, np.ravel(trainLabel))  # ravel Return a contiguous flattened array.
    return clf

#SVM
from sklearn.svm import SVC
def trainModel2(trainData, trainLabel):
    print('Train SVM...')
    clf = SVC(C=4, kernel='rbf')
    clf.set_params(
        gamma='scale',
        probability=True,
        tol=0.001)
    clf.fit(trainData, trainLabel)  # 训练SVM
    return clf
#随机森林
from sklearn.ensemble import RandomForestClassifier
def trainModel3(X_train, y_train):
    print('Train RF...')
    clf = RandomForestClassifier(
        n_estimators=10,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=34)
    clf.fit(X_train, y_train)  # 训练rf
    return clf

#LightGBM
import lightgbm as lgb
def trainModel4(X_train, y_train):
    print('Train LGB...')
    clf = lgb.LGBMClassifier()
    clf.fit(X_train, y_train)  # 训练lgb
    return clf

#XGBoost
from xgboost import XGBClassifier
def trainModel5(X_train, y_train):
    print('Train XGB...')
    clf = XGBClassifier()
    clf.fit(X_train, y_train)  # 训练xgb
    return clf

#CatBoost
from catboost import CatBoostClassifier
def trainModel6(X_train, y_train):
    print('Train CatBoost...')
    clf = CatBoostClassifier()
    clf.fit(X_train, y_train)  # 训练catboost
    return clf


# 模型训练
# #trainModel函数期望接收训练数据和对应的标签作为参数。在您的情况下，trainDataPCA是降维后的训练数据，而trainLabel是对应的标签。
# #trainLabel包含了训练数据的标签，这是训练分类模型（如KNN）所必需的。在监督学习中，模型需要学习如何将输入数据（特征）映射到输出标签（类别）

# clf = trainModel1(trainDataPCA, trainLabel)。
# testLabel = clf.predict(testDataPCA)

# clf = trainModel2(trainDataPCA, trainLabel)
# testLabel = clf.predict(testDataPCA)

# clf = trainModel3(trainDataPCA, trainLabel)
# testLabel = clf.predict(testDataPCA)

# clf = trainModel4(trainDataPCA, trainLabel)
# testLabel = clf.predict(testDataPCA)

clf = trainModel5(trainDataPCA, trainLabel)
testLabel = clf.predict(testDataPCA)

# clf = trainModel6(trainDataPCA, trainLabel)
# testLabel = clf.predict(testDataPCA)


# def saveResult(result, csvName):
#     with open(csvName, 'w', newline='') as myFile:
#         myWriter = csv.writer(myFile)
#         myWriter.writerow(["ImageId", "Label"]) # 写入表头
#         index = 0   # 记录行号
#         for i in result:
#             tmp = []
#             index = index+1
#             tmp.append(index)
#             # tmp.append(i)
#             tmp.append(int(i))
#             myWriter.writerow(tmp)
#
# # 结果的输出
# saveResult(testLabel, 'output/digit_submission.csv')

#或者使用pd.DataFrame()方法
df = pd.DataFrame({'ImageId': range(1, len(testLabel)+1), 'Label': testLabel})
df.to_csv('output/digit_submission.csv', index=False)