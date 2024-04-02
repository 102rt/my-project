import joblib as joblib
import pandas as pd
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from tqdm.contrib import itertools
from xgboost import plot_importance
import time
import xgboost as xgb
from matplotlib import pyplot as plt
from xgboost import XGBClassifier, plot_importance
from lightgbm.sklearn import LGBMClassifier
from numpy import loadtxt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# load data
import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv('traindataset_label.csv',index_col=0)

# split data into X and y
data_X = data.drop(['labels'], axis=1).values   # '30_bkg_median','30_bkg_mean','30_bkg_std','30_density','30_object_median','30_object_mean','30_object_std',
print(data_X.shape)
data_y = np.ravel(data.loc[:, ['labels']].values).astype(int)
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y,test_size=0.2,random_state=42)



'''-------------------------------------------------------pso+xgb-----------------------------------------------------------------'''
model1 = XGBClassifier(learning_rate=0.1556201,max_depth=8,reg_lambda=4.1543267,

                      objective='multi:softmax',num_class=3,n_estimators=1000)
# 训练模型
# xgb_train_start = time.time()
model1.fit(X_train, y_train)
import joblib
joblib.dump(model1, 'xgboost_model.joblib')
y_pred = model1.predict(X_test)
# # xgb_test_end=time.time()
# # xgb_test=xgb_test_end-xgb_test_start
# # print('xgb test time:', xgb_test)
#
# # 计算评价指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)  # 'weighted'
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)
print("PSO+XGBoost accuracy：",accuracy)
print("PSO+XGBoost precision：",precision)
print("PSO+XGBoost recall ：",recall )
print("PSO+XGBoost f1 ：",f1 )

"""feature importance"""
fig,ax = plt.subplots(figsize=(25,15))
# plot_importance(model,height=0.5, ax=ax,max_num_features=64)
plt.rcParams.update({'font.size':15})
plot_importance(model1,height=0.5,xlabel='F score',ylabel='Features',max_num_features=None,grid=False)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
plt.tick_params(labelsize=15)
plt.close()


'''-----------------------------confusion matrix---------------------------'''

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='PSO+XGBoost',
                          cmap=plt.cm.GnBu):

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
    font={'size':30}
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
                      title='PSO+XGBoost')

ax=plt.gca()  # 获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2)  # 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2)  #  设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2)  #  设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2)   # 设置上部坐标轴的粗细
plt.show()

print('----------------------------------------------lightgbm-----------------------------------')
gbm_start=time.time()
parameters= {
    'max_depth': 12,  # 32
    'n_estimators': 1000,
    'learning_rate': 0.02,  # 0.15 0.02 96539
    'num_leaves': 32,  # 65
    'min_child_samples': 20,  # 92
    'reg_alpha': 1,
    'reg_lambda':25 }  # 25   # 30 88.91 95.87
model2 = LGBMClassifier(objective='multiclass', random_state=42, n_jobs=-1, **parameters) # 'min_child_samples': 92,
# gbm_train_start = time.time()
model2.fit(X_train, y_train)
# gbm_test_start = time.time()
y_pred = model2.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)  # 'weighted'
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)
print("lightgbm accuracy：",accuracy)
print("lightgbm precision：",precision)
print("lightgbm recall ：",recall )
print("lightgbm f1 ：",f1 )

joblib.dump(model2, 'lightgbm_model.joblib')

'''-----------------------------confusion matrix---------------------------'''

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='LightGBM',
                          cmap=plt.cm.GnBu):
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
    font={'size':30}
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontdict={'family' : 'Times New Roman', 'size' : 28})

    plt.ylabel('True ',fontdict={'family' : 'Times New Roman', 'size': 30})
    plt.xlabel('Predicted',fontdict={'family' : 'Times New Roman', 'size': 30})
    plt.tight_layout()


class_names = ['Clear','Moon' , 'Covered']
cnf_matrix = confusion_matrix( y_test, y_pred)
plt.figure(figsize=(2.5,2.5))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='LightGBM')

ax=plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
plt.close()

'''----------------------------------------------------------------rf-----------------------------------------------'''
model3 = RandomForestClassifier( n_estimators=1000,criterion='entropy', max_depth=32) # n_estimators=1300



# rf_train_start=time.time()
model3.fit(X_train, y_train)
# rf_test_start=time.time()
y_pred = model3.predict(X_test)
# rf_test_end=time.time()
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)
print("rf accuracy：",accuracy)
print("rf precision：",precision)
print("rf recall ：",recall )
print("rf f1 ：",f1 )

joblib.dump(model3, 'rf_model.joblib')
'''-----------------------------confusion matrix---------------------------'''

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='LightGBM',
                          cmap=plt.cm.GnBu):
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
    font={'size':30}
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontdict={'family' : 'Times New Roman', 'size' : 28})

    plt.ylabel('True ',fontdict={'family' : 'Times New Roman', 'size': 30})
    plt.xlabel('Predicted',fontdict={'family' : 'Times New Roman', 'size': 30})
    plt.tight_layout()


class_names = ['Clear','Moon' , 'Covered']
cnf_matrix = confusion_matrix( y_test, y_pred)
# 绘制混淆矩阵可视化图
plt.figure(figsize=(2.5,2.5))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='RF')

ax=plt.gca()
ax.spines['bottom'].set_linewidth(2)  # 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2)  #  设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2)  #  设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2)   # 设置上部坐标轴的粗细
plt.close()


'''-----------------------------KNN---------------------------'''
model4 = KNeighborsClassifier(n_neighbors=7,algorithm='auto')


# knn_train_start = time.time()
model4.fit(X_train, y_train)
# knn_train_end=time.time()
# knn_train=knn_train_end- knn_train_start
# print('knn train time:',knn_train)
# train_times.append(knn_train)


# knn_test_start = time.time()
y_pred = model4.predict(X_test)
# knn_test_end=time.time()
# knn_test=knn_test_end- knn_test_start
# print('knn test time:',knn_test)
# test_times.append(knn_test)
# 计算评价指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)
print("KNN accuracy：",accuracy)
print("KNN precision：",precision)
print("KNN recall ：",recall )
print("KNN f1 ：",f1 )

joblib.dump(model4, 'knn_model.joblib')

'''-----------------------------confusion matrix---------------------------'''

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='LightGBM',
                          cmap=plt.cm.GnBu):

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
    font={'size':30}
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

plt.figure(figsize=(2.5,2.5))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='KNN')

ax=plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
plt.close()


'''------------------------------------------------------------svm----------------------------------------------------'''
data = pd.read_csv('traindataset_label.csv', index_col=0)

# 分割特征和标签
X = data.drop(['labels'], axis=1).values
y = data['labels'].values

# 数据标准化
scaler = MinMaxScaler() # StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model5 = svm.SVC(kernel='rbf', C=0.1, gamma=1, max_iter=1200)

model5.fit(X_train, y_train)

# test_start=time.time()
y_pred = model5.predict(X_test)
test_end=time.time()
# print("testing time:", test_end-test_start)
joblib.dump(model5, 'svm_normalization.joblib')
# 计算评价指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

print("SVM accuracy:", accuracy)
print("SVM precision:", precision)
print("SVM recall:", recall)
print("SVM f1:", f1)
'''-----------------------------confusion matrix---------------------------'''

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='SVM',
                          cmap=plt.cm.GnBu):

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
    font={'size':30}
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontdict={'family' : 'Times New Roman', 'size' : 28})

    plt.ylabel('True ',fontdict={'family' : 'Times New Roman', 'size': 30})
    plt.xlabel('Predicted',fontdict={'family' : 'Times New Roman', 'size': 30})
    plt.tight_layout()


class_names = ['Clear','Moon' , 'Covered']
cnf_matrix = confusion_matrix( y_test, y_pred)

plt.figure(figsize=(2.5,2.5))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='SVM')

ax=plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
plt.show()
