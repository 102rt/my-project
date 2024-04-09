import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.contrib import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


'''-------------------------------------------------------------------------------------------------------------------------------'''
# 读取数据
data = pd.read_csv('traindataset_label.csv', index_col=0)

# 分割特征和标签
X = data.drop(['labels'], axis=1).values
y = data['labels'].values

# # 对数据进行全局特征标准化
# global_min = np.min(X)
# global_max = np.max(X)
# X_normalized = (X - global_min) / (global_max - global_min)
#
# # 训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)


# 数据标准化
scaler = MinMaxScaler() # StandardScaler()
X_scaled = scaler.fit_transform(X)
# 训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建SVM分类器
model = svm.SVC(kernel='rbf', C=0.1, gamma=1, max_iter=1200)  # linear:30.91 rbf\0.1\1\1200:31.07 poly:30.81

# 在训练集上训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
import time
# test_start=time.time()
y_pred = model.predict(X_test)
test_end=time.time()
# print("testing time:", test_end-test_start)
import joblib
joblib.dump(model, 'svm_normalization.joblib')
# 计算评价指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

print("SVM accuracy:", accuracy)
print("SVM precision:", precision)
print("SVM recall:", recall)
print("SVM f1:", f1)
'''-----------------------------定义绘制混淆矩阵的函数---------------------------'''

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='SVM',
                          cmap=plt.cm.GnBu):
    """
    此函数用于绘制混淆矩阵的可视化图
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontdict={'family' : 'Times New Roman', 'size' : 30})
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontproperties = 'Times New Roman', size = 20)  # rotation=45
    plt.yticks(tick_marks, classes, fontproperties = 'Times New Roman', size = 20)

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    font={'size':30}  # 混淆矩阵数字字体大小
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontdict={'family' : 'Times New Roman', 'size' : 28})

    plt.ylabel('True ',fontdict={'family' : 'Times New Roman', 'size': 30})
    plt.xlabel('Predicted',fontdict={'family' : 'Times New Roman', 'size': 30})
    plt.tight_layout()

# 定义分类标签
class_names = ['Clear','Moon' , 'Covered']
cnf_matrix = confusion_matrix( y_test, y_pred)
# 绘制混淆矩阵可视化图
plt.figure(figsize=(2.5,2.5))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='SVM')

ax=plt.gca()  # 获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2)  # 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2)  #  设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2)  #  设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2)   # 设置上部坐标轴的粗细
plt.show()


# #
# data = pd.read_csv('traindataset_label.csv', index_col=0)
#
# # 分割特征和标签
# X = data.drop(['labels'], axis=1).values
# y = data['labels'].values
#
# # 对数据进行全局特征标准化
# global_min = np.min(X)
# global_max = np.max(X)
# X_normalized = (X - global_min) / (global_max - global_min)
#
# # 训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
#
# # 设置参数候选列表
# parameters = {
#     'kernel': ['linear', 'rbf', 'poly'],  # 核函数
#     'C': [0.1, 1, 10],  # 惩罚因子
#     'gamma': [0.1, 1, 10]  # 惩罚系数
# }
#
# # 创建SVM分类器
# model = svm.SVC()
#
# # 网格搜索寻找最优参数
# grid_search = GridSearchCV(model, parameters, cv=5)
# grid_search.fit(X_train, y_train)
#
# # 获取最优参数
# best_model = grid_search.best_estimator_
#
# # 在测试集上进行预测
# y_pred = best_model.predict(X_test)
#
# # 保存模型
# import joblib
# joblib.dump(best_model, 'svm_normalization.joblib')
#
# # 计算评价指标
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average=None)
# recall = recall_score(y_test, y_pred, average=None)
# f1 = f1_score(y_test, y_pred, average=None)
#
# print("Best SVM parameters:", grid_search.best_params_)
# print("SVM accuracy:", accuracy)
# print("SVM precision:", precision)
# print("SVM recall:", recall)
# print("SVM f1:", f1)
