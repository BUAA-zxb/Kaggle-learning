import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from time import time
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# 训练集和测试集数据路径
train_path = 'E:/AAAdesktop/Kaggle/digit-recognizer/train.csv'
test_path = 'E:/AAAdesktop/Kaggle/digit-recognizer/test.csv'

# 导入训练集和测试集数据
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 查看训练集和测试集数据信息
# train.info()
# test.info()

# 查看训练集是否有缺失值,结果是不存在缺失值
# train.isnull().any().describe()
# test.isnull().any().describe()

# 从训练集中找出数据和标签
X_train = train.drop(labels={"label"}, axis=1)
Y_train = train["label"]

# 正则化
X_train = X_train / 255
test = test / 255
print(X_train.shape)
# 显示训练集前几个数据 如果显示就不能训练
# X_train = np.array(X_train).reshape(-1, 28, 28, 1)
# plt.imshow(X_train[1][:, :, 0], interpolation="none", cmap="Greys")
# plt.show()

# 训练集和测试集数据分开
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=1)


# # 主成分分析函数
# def get_accuracy_score(n, X_train, X_test, y_train, y_test):
#     t0 = time()
#     pca = PCA(n_components=n)
#     pca.fit(X_train)
#     x_train_pca = pca.transform(X_train)
#     x_test_pca = pca.transform(X_test)
#     # 使用支持向量机分类器
#     clf = SVC()
#     clf.fit(x_train_pca, y_train)
#     y_predict = clf.predict(x_test_pca)
#     # 计算准确度
#     accuracy = metrics.accuracy_score(y_test, y_predict, normalize=True)
#     t1 = time()
#     print('n_components:{:.2f} , accuracy:{:.4f} , time:{:.2f}s'.format(n, accuracy, t1 - t0))
#     print(metrics.f1_score(y_test, y_predict, average='macro'),
#           metrics.f1_score(y_test, y_predict, average='micro'),
#           metrics.f1_score(y_test, y_predict, average='weighted'))
#     print(metrics.precision_score(y_test, y_predict, average='macro'),
#           metrics.precision_score(y_test, y_predict, average='micro'),
#           metrics.precision_score(y_test, y_predict, average='weighted'))
#     print(metrics.recall_score(y_test, y_predict, average='macro'),
#           metrics.recall_score(y_test, y_predict, average='micro'),
#           metrics.recall_score(y_test, y_predict, average='weighted'))
#     return accuracy
#
#
# # 定义得分矩阵
# all_scores = []
# # 生成n_components的取值列表 0.5~1，分成500个数
# n_components = np.linspace(0.5, 1, num=500, endpoint=False)
# for n in n_components:
#     score = get_accuracy_score(n, x_train, x_test, y_train, y_test)
#     # 最终发现0.75是效果最好
#     all_scores.append(score)

# # 找出识别有误的数据
# pca = PCA(n_components=0.75)
# pca.fit(x_train)
# X_train_pca = pca.transform(x_train)
# X_test_pca = pca.transform(x_test)
# clf = SVC()
# clf.fit(X_train_pca, y_train)
# y_pred = clf.predict(X_test_pca)
# errors = (y_pred != y_test)
# y_pred_errors = y_pred[errors]
# y_test_errors = y_test[errors].values
# X_test_errors = x_test[errors]
# print(y_pred_errors[:5])
# print(y_test_errors[:5])
# print(X_test_errors[:5])
# X_test_errors = np.array(X_test_errors).reshape(-1, 28, 28, 1)
# n = 0
# rows = 2
# cols = 5
# for row in range(rows):
#     for col in range(cols):
#         plt.imshow(X_test_errors[n][:, :, 0], interpolation="none", cmap="Greys")
#         n += 1
#         plt.show()

# n_components为0.75时, 模型的准确率最高
# 对训练集和测试集进行PCA降低维度处理, 主成分个数为33
pca = PCA(n_components=0.75)
pca.fit(x_train)
# 打印主成分个数
# print(pca.n_components_)
# 对训练集和测试集进行主成分转换
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
test_pca = pca.transform(test)

# # 使用支持向量机预测,使用网格搜索进行调参
# # 调参结果发现C=5，kernel=rbf，效果最好
# clf_svc = GridSearchCV(estimator=SVC(), param_grid={'C': [5], 'gamma': [0.03], 'kernel': ['rbf']}, cv=5, verbose=2)
# clf_svc.fit(x_train_pca, y_train)
# # 显示使模型准确率最高的参数
# print(clf_svc.best_params_)

# 使用最好参数训练算法
clf_svc = SVC(C=5, gamma=0.03, kernel='rbf')
clf_svc.fit(x_train_pca, y_train)

# 使用现有数据预测模型
y_predict = clf_svc.predict(x_test_pca)
print('准确率：', metrics.accuracy_score(y_test, y_predict, normalize=True))

print('宏平均查准率：', metrics.precision_score(y_test, y_predict, average='macro'))
print('微平均查准率：', metrics.precision_score(y_test, y_predict, average='micro'))
print('加权平均查准率：', metrics.precision_score(y_test, y_predict, average='weighted'))

print('宏平均查全率：', metrics.recall_score(y_test, y_predict, average='macro'))
print('微平均查全率：', metrics.recall_score(y_test, y_predict, average='micro'))
print('加权平均查全率：', metrics.recall_score(y_test, y_predict, average='weighted'))

print('宏平均f1-score：', metrics.f1_score(y_test, y_predict, average='macro'))
print('微平均f1-score：', metrics.f1_score(y_test, y_predict, average='micro'))
print('加权平均f1-score：', metrics.f1_score(y_test, y_predict, average='weighted'))

# 生成最终预测结果
pred = clf_svc.predict(test_pca)
image_id = pd.Series(range(1, len(pred)+1))
result_2 = pd.DataFrame({'ImageID': image_id, 'Label': pred})
# 保存为CSV文件
result_2.to_csv('result2_svc.csv', index=False)
print('Done')
