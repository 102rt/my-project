import itertools

import xgboost as xgb
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
import pandas as pd
import numpy as np
from sklearn import metrics
import random
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('traindataset_label.csv',  index_col=0)  # index_col=0 第一列为index
# data_X = data.drop(['labels'], axis=1)   # 不包含标签cloudy的数据集
data_X = data.drop(['labels'], axis=1)
# '30_bkg_median','30_bkg_mean','30_bkg_std','30_density','30_object_median','30_object_mean','30_object_std',
data_y = np.ravel(data.loc[:, ['labels']].values).astype(int)  # ravel函数的功能是将原数组拉伸成为一维数组.
# data_X_featurenames = data.drop(['labels'], axis=1).columns.values  # 特征名称，不包括cloudy
#

# split data into training and validation sample
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)

# 定义了一个PSO类，构造函数接收一个参数parameters（如 [NGEN, pop_size, var_num_min, var_num_max]），其中包括迭代代数、种群大小和变量范围等信息。
class PSO:
    def __init__(self, parameters):
        """
        particle swarm optimization
        parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
        """
        # 初始化
        self.NGEN = parameters[0]  # 迭代的代数
        self.pop_size = parameters[1]  # 种群大小
        self.var_num =len(parameters[2])  # 变量个数
        self.bound = []  # 变量的约束范围
        self.bound.append(parameters[2])  # min
        self.bound.append(parameters[3])  # max

        # 初始化每个粒子的位置、速度、最优位置和全局最优位置，并通过随机数初始化它们的值。同时，计算每个粒子的适应度（fitness）并更新全局最优位置。
        self.pop_x = np.zeros((self.pop_size, self.var_num))  # 所有粒子的位置
        self.pop_v = np.zeros((self.pop_size, self.var_num))  # 所有粒子的速度
        self.p_best = np.zeros((self.pop_size, self.var_num))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.var_num))  # 全局最优的位置

        # 初始化第0代初始全局最优解

        temp = -1
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.pop_v[i][j] = random.uniform(0, 1)
            self.p_best[i] = self.pop_x[i]  # 储存最优的个体
            fit = self.fitness(self.p_best[i])  #
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    # fitness函数计算了每个粒子的适应度，即模型在当前参数下得到的准确率。其中，ind_var为当前粒子的位置
    def fitness(self, ind_var):  # train
        X = X_train
        y = y_train
        """
        个体适应值计算
        """
        x1 = ind_var[0]  # learning_rate
        x2 = ind_var[1].astype(int)    # max_depth
        x3 = ind_var[2]  # reg_lambda
        # x4 = ind_var[3]
        # x5 = ind_var[4]
        # x6 = ind_var[5]

        if x1 == 0: x1 = 1.5
        if x2 == 2: x2 =64
        if x3 == 0: x3 = 20
        # if x4 == 0: x4 = 1
        # if x5 == 0: x5 = 1
        # if x6 == 0.1: x6 = 20  # reg_lambda

        para = {"objective": 'multi:softmax', "num_class": 3,
                'eval_metric': 'mlogloss',"tree_method": "gpu_hist", "gpu_id": 0}  # 'scale_pos_weight':{1:15.8,2:2.47,3:16},
        xgb = XGBClassifier(learning_rate =x1,max_depth=x2,reg_lambda=x3,n_estimators=1200,**para)  # max_depth=int(x2),n_estimators=1000 scale_pos_weight=x6,
        xgb.fit(X, y)  # train
        y_pre = xgb.predict(X_test)  # test
        # accuracy=accuracy_score(y_test, y_pre)
        # precision = metrics.precision_score(y_test, y_pre, average='weighted')
        # recall = metrics.recall_score(y_test, y_pre, average='weighted')
        # f1_score = metrics.f1_score(y_test, y_pre, average='weighted')
        # print("accuracy = ", metrics.accuracy_score(y_test, predictval))
        return  metrics.accuracy_score(y_test,  y_pre)   # accuracy,precision,recall,f1_score
            # metrics.accuracy_score(y_test,  y_pre)  #,metrics.precision_score(y_test,  y_pre),\
             #  metrics.recall_score(y_test,  y_pre),metrics.f1_score(y_test,  y_pre)

    def update_operator(self, pop_size):
        """
        更新算子：更新下一时刻的位置和速度
        """
        # c1、c2主要影响粒子对个体经验和群体经验的信任程度;c1代表了粒子的个体意识，而c2代表了粒子的群体意识
        c1 =1.6  # 2  # 个体学习因子 1.6
        c2 = 2  # 全局学习因子  2
        w = 0.4  # 自身权重因子，决定了粒子搜索的多样性
        for i in range(pop_size):
            # 更新速度
            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.pop_x[i])
            # 更新位置
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
            # 越界保护
            for j in range(self.var_num):
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = self.bound[0][j]
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]
            # 更新p_best和g_best
            if self.fitness(self.pop_x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.pop_x[i]
            if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]

    # 整个算法的主流程，实现多次迭代调整参数的过程，并记录每次迭代的最优适应度和最优解。同时，输出最终求得的全局最优解和其对应的适应度。
    def main(self):
        popobj = []
        self.ng_best = np.zeros((1, self.var_num))[0]
        accuracies = []  # 存储 accuracy
        for gen in range(self.NGEN):
            self.update_operator(self.pop_size)
            popobj.append(self.fitness(self.g_best))
            print('############ Generation {} ############'.format(str(gen + 1)))
            if self.fitness(self.g_best) > self.fitness(self.ng_best):
                self.ng_best = self.g_best.copy()

            print('best parameters：{}'.format(self.ng_best))
            accuracy = self.fitness(self.ng_best)  # 计算当前最佳参数的 accuracy
            a=accuracies.append(accuracy)  # 将 accuracy 存入列表
            print('best score：{}'.format(accuracy))

            # precision, recall, f1_score = self.fitness(self.g_best)[1:]
            # print('precision:', precision)
            # print('recall:', recall)
            # print('f1_score:', f1_score)

            # plt.plot(range(1, gen + 2), accuracies)  # 绘制 accuracy 曲线
            # plt.title('Accuracy over generations')
            # plt.xlabel('Generation')
            # plt.ylabel('Accuracy')
            # plt.show()

        plt.plot(range(1, self.NGEN + 1), accuracies,linewidth=2, color='#FA7F6F', marker='s')  # 绘制 accuracy 曲线
        # plt.title('Accuracy over generations')
        plt.xlabel('Generation',fontsize=30)
        plt.ylabel('Accuracy',fontsize=30)
       # plt.plot(a,linewidth=2, color='#FA7F6F', marker='s')
        ax = plt.gca()  # 获得坐标轴的句柄
        ax.spines['bottom'].set_linewidth(3)  # 设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(3)  # 设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(3)  # 设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(3)  # 设置上部坐标轴的粗细
        plt.tick_params(labelsize=20)  # 刻度值字体设置
        plt.show()
        print("---- End of Searching ----")

    '''
    def main(self):
        popobj = []
        self.ng_best = np.zeros((1, self.var_num))[0]
        # self.precision_best=np.zeros((1, self.var_num))[1]

        for gen in range(self.NGEN):
            self.update_operator(self.pop_size)
            popobj.append(self.fitness(self.g_best))
            print('############ Generation {} ############'.format(str(gen + 1)))
            if self.fitness(self.g_best) > self.fitness(self.ng_best):
                self.ng_best = self.g_best.copy()

            print('best parameters：{}'.format(self.ng_best))
            print('best score：{}'.format(self.fitness(self.ng_best)))

            # precision, recall, f1_score = self.fitness(self.g_best)[1:]
            # print('precision:', precision)
            # print('recall:', recall)
            # print('f1_score:', f1_score)
        print("---- End of Searching ----")
'''
if __name__ == '__main__':
    NGEN = 25  # PSO迭代次数 25
    popsize = 20 # 粒子群数目，即鸟的数量 20
    low = [0,2,0]
    up = [1.5,64,20]  # 参数的上下界
    parameters = [NGEN, popsize, low, up]
    pso = PSO(parameters)
    pso.main()
    x1, x2, x3= pso.ng_best[0], int(pso.ng_best[1]), pso.ng_best[2]
    print('Final parameters: ', pso.ng_best)
    para = {"objective": 'multi:softmax', "num_class": 3,'eval_metric':'mlogloss',
             "tree_method": "gpu_hist", "gpu_id": 0}  #'scale_pos_weight':{1:15.8,2:2.47,3:16}, 'eval_metric': 'mlogloss',
    xgb = XGBClassifier(learning_rate=x1, max_depth=x2, reg_lambda=x3,n_estimators=1200, **para)  # n_estimators=1000
    xgb.fit(X_train, y_train)
    # # 设置验证集
    # eval_set = [(X_train, y_train), (X_test, y_test)]
    #
    # # 训练模型
    # xgb.fit(X_train, y_train, eval_metric=[ "merror","mlogloss"], eval_set=eval_set, verbose=True)
    #
    # # 可视化迭代次数与accuracy关系图
    # results = xgb.evals_result()
    # epochs = len(results['validation_0']['merror'])
    y_pred =xgb.predict(X_test)


    # average
    accuracy = accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    f1score = metrics.f1_score(y_test, y_pred, average='weighted')
    print('Accuracy: {}'.format(accuracy))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1score))
    # each class
    precision_score_average_None = precision_score(y_test, y_pred, average=None)
    recall_score_average_None = recall_score(y_test, y_pred, average=None)
    f1_score_None=f1_score(y_test, y_pred, average=None)
    print('precision_score_average_None = ', precision_score_average_None)
    print('recall_score_average_None = ', recall_score_average_None)
    print('f1_score_average_None = ', f1_score_None)



    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='PSO+XGBoost',
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
        plt.title(title, fontdict={'family': 'Times New Roman', 'size': 30})
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontproperties='Times New Roman', size=20)  # rotation=45
        plt.yticks(tick_marks, classes, fontproperties='Times New Roman', size=20)

        fmt = '.4f' if normalize else 'd'
        thresh = cm.max() / 2.
        font = {'size': 30}  # 混淆矩阵数字字体大小
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontdict={'family': 'Times New Roman', 'size': 28})

        plt.ylabel('True ', fontdict={'family': 'Times New Roman', 'size': 30})
        plt.xlabel('Predicted', fontdict={'family': 'Times New Roman', 'size': 30})
        plt.tight_layout()


    # 定义分类标签
    class_names = ['Clear', 'Moon', 'Covered']
    cnf_matrix = confusion_matrix(y_test, y_pred)
    # 绘制混淆矩阵可视化图
    plt.figure(figsize=(2.5, 2.5))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='XGBoost')

    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2)  # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2)  # 设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2)  # 设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)  # 设置上部坐标轴的粗细
    plt.show()
