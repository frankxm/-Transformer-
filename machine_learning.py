
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


path_data=r'./haiyuan_final.csv'
path_earthquake=r'./Area1_3-10.csv'
# 读取现成的
# 块读取，chunkSize规定每次读取多少行，之后合并成一个大的dataframe
df = pd.read_csv(path_data, sep=',', engine='python', iterator=True)
loop = True
chunkSize = 100000
chunks = []
index = 0
while loop:
    try:
        print(index)
        chunk = df.get_chunk(chunkSize)
        chunks.append(chunk)
        index += 1

    except StopIteration:
        loop = False
        print("Iteration is stopped.")
print('读取应力数据开始合并')
df = pd.concat(chunks, ignore_index=True)
print('读取地震目录数据')
# 读取地震目录数据
# df_area = pd.read_csv(path_earthquake, encoding='utf_8_sig')
df_area = pd.read_csv(path_earthquake, encoding='gb2312')

df_earth = df_area['发震日期（北京时间）']
df_earth1 = pd.to_datetime(df_earth)
df_time = df_earth1.dt.strftime('%Y-%m-%d-%H:%M')
df_time_np = np.array(df_time)



# 找到地震目录在数据中的索引
rule=df['datetime'].isin(df_time_np)
ind_all=[i for i in range(len(df))]
# 用集合方式区分正负样本索引
ind_list=df[rule].index.tolist()
ind_neg = list(set(ind_all) - set(ind_list))
pos_num=len(ind_list)
# 随机取负样本
random.seed(10)
neg_num =pos_num
neg_list = random.sample(ind_neg, neg_num)
# 总的样本索引，负样本索引跟在正样本索引后
ind_list.extend(neg_list)
ind_list=np.array(ind_list)

# 构造总的标签和时间
# 地震目录上的时间点标签都为1
labels = [1] * pos_num*40
lab = [0] * neg_num*40
labels.extend(lab)

df = df.drop('datetime', axis=1)
df_data=[]
sc = StandardScaler()
def get_data(x):
    df_choosed = df.iloc[x - 10:x + 30]
    df_choosed = np.array(df_choosed)
    df_data = sc.fit_transform(df_choosed)
    return df_data

df_data = list(map(lambda x: get_data(x), ind_list))
X= np.vstack(df_data)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.1, random_state = 0)


classifier_bayes = GaussianNB()
classifier_bayes.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
classifier_logistic = LogisticRegression(random_state = 0)
classifier_logistic.fit(X_train,y_train)

from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train,y_train)

from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'linear', random_state = 0)
classifier_svm.fit(X_train,y_train)

from sklearn.svm import SVC
classifier_kernelsvm = SVC(kernel = 'rbf', random_state = 0)
classifier_kernelsvm.fit(X_train,y_train)

from sklearn.tree import DecisionTreeClassifier
classifier_tree = DecisionTreeClassifier(random_state=0)
classifier_tree.fit(X_train,y_train)

# Predicting the Test set results
y_pred_bayes = classifier_bayes.predict(X_test)
y_score = classifier_bayes.predict_proba(X_test)

y_pred_logistic = classifier_logistic.predict(X_test)
y_pred_knn = classifier_knn.predict(X_test)
y_pred_svm = classifier_svm.predict(X_test)
y_pred_kernelsvm = classifier_kernelsvm.predict(X_test)
y_pred_tree=classifier_tree.predict(X_test)
# 混淆矩阵
# 真实值是positive，模型认为是positive的数量（True Positive=TP）
# 真实值是positive，模型认为是negative的数量（False Negative=FN）：这就是统计学上的第一类错误（Type I Error）
# 真实值是negative，模型认为是positive的数量（False Positive=FP）：这就是统计学上的第二类错误（Type II Error）
# 真实值是negative，模型认为是negative的数量（True Negative=TN）
# 行标签代表真实值,列标签代表预测值,左上角表示真负TN,右下角表示真正TP
# 精确率：模型预测是1中预测对的比重 召回率：在真实的1中，模型预测对的比重
cm_bayes = confusion_matrix(y_test, y_pred_bayes)
cm_logistic=confusion_matrix(y_test,y_pred_logistic)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_svm=confusion_matrix(y_test,y_pred_svm)
cm_kernelsvm = confusion_matrix(y_test, y_pred_kernelsvm)
cm_tree = confusion_matrix(y_test, y_pred_tree)
print('朴素贝叶斯混淆矩阵为:\n',cm_bayes)
print('逻辑回归混淆矩阵为:\n',cm_logistic)
print('KNN混淆矩阵为:\n',cm_knn)
print('SVM混淆矩阵为:\n',cm_svm)
print('Kernel SVM混淆矩阵为:\n',cm_kernelsvm)
print('决策树混淆矩阵为:\n',cm_tree)
from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score

accuracy_bayes=accuracy_score(y_test,y_pred_bayes)
precision_bayes=precision_score(y_test,y_pred_bayes)
recall_bayes=recall_score(y_test,y_pred_bayes)
# F1-Score的取值范围从0到1的，1代表模型的输出最好，0代表模型的输出结果最差。
f1_bayes=f1_score(y_test,y_pred_bayes)
print('朴素贝叶斯:')
print('准确率为:\n',accuracy_bayes)
print('精确率为:\n',precision_bayes)
print('召回率为:\n',recall_bayes)
print('F1score为:\n',f1_bayes)

from sklearn.model_selection import cross_val_predict

y_scores1 = cross_val_predict(classifier_bayes, X_train, y_train, cv=5)
y_scores2 = cross_val_predict(classifier_logistic, X_train, y_train, cv=5)
y_scores3 = cross_val_predict(classifier_knn, X_train, y_train, cv=5)
y_scores4 = cross_val_predict(classifier_svm, X_train, y_train, cv=5)
y_scores5 = cross_val_predict(classifier_kernelsvm, X_train, y_train, cv=5)
from sklearn.metrics import precision_recall_curve

precisions1, recalls1, thresholds1 = precision_recall_curve(y_train, y_scores1)
precisions2, recalls2, thresholds2 = precision_recall_curve(y_train, y_scores2)
precisions3, recalls3, thresholds3 = precision_recall_curve(y_train, y_scores3)
precisions4, recalls4, thresholds4 = precision_recall_curve(y_train, y_scores4)
precisions5, recalls5, thresholds5 = precision_recall_curve(y_train, y_scores5)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()
plot_precision_recall_vs_threshold(precisions1,recalls1,thresholds1)
plot_precision_recall_vs_threshold(precisions2,recalls2,thresholds2)
plot_precision_recall_vs_threshold(precisions3,recalls3,thresholds3)
plot_precision_recall_vs_threshold(precisions4,recalls4,thresholds4)
plot_precision_recall_vs_threshold(precisions5,recalls5,thresholds5)

def plot_precision_recall(precisions, recalls):
    plt.plot(recalls[:-1],precisions[:-1],"g-")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.show()
plot_precision_recall(precisions1, recalls1)
plot_precision_recall(precisions2, recalls2)
plot_precision_recall(precisions3, recalls3)
plot_precision_recall(precisions4, recalls4)
plot_precision_recall(precisions5, recalls5)

from sklearn.metrics import roc_curve
fpr1, tpr1, thresholds11 = roc_curve(y_train, y_scores1)
fpr2, tpr2, thresholds12 = roc_curve(y_train, y_scores2)
fpr3, tpr3, thresholds13 = roc_curve(y_train, y_scores3)
fpr4, tpr4, thresholds14 = roc_curve(y_train, y_scores4)
fpr5, tpr5, thresholds15 = roc_curve(y_train, y_scores5)
def plot_roc_curve(fpr, tpr, label=None):
   plt.plot(fpr, tpr, linewidth=2, label=label)
   plt.plot([0, 1], [0, 1], 'k--')
   plt.axis([0, 1, 0, 1])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.show()
plot_roc_curve(fpr1, tpr1)
plot_roc_curve(fpr2, tpr2)
plot_roc_curve(fpr3, tpr3)
plot_roc_curve(fpr4, tpr4)
plot_roc_curve(fpr5, tpr5)

from sklearn.metrics import roc_auc_score
auc1=roc_auc_score(y_train, y_scores1)
auc2=roc_auc_score(y_train, y_scores2)
auc3=roc_auc_score(y_train, y_scores3)
auc4=roc_auc_score(y_train, y_scores4)
auc5=roc_auc_score(y_train, y_scores5)
print(auc1,auc2,auc3,auc4,auc5)

accuracy_logistic=accuracy_score(y_test,y_pred_logistic)
precision_logistic=precision_score(y_test,y_pred_logistic)
recall_logistic=recall_score(y_test,y_pred_logistic)
# F1-Score的取值范围从0到1的，1代表模型的输出最好，0代表模型的输出结果最差。
f1_logistic=f1_score(y_test,y_pred_logistic)
print('逻辑回归:')
print('准确率为:\n',accuracy_logistic)
print('精确率为:\n',precision_logistic)
print('召回率为:\n',recall_logistic)
print('F1score为:\n',f1_logistic)

accuracy_knn=accuracy_score(y_test,y_pred_knn)
precision_knn=precision_score(y_test,y_pred_knn)
recall_knn=recall_score(y_test,y_pred_knn)
# F1-Score的取值范围从0到1的，1代表模型的输出最好，0代表模型的输出结果最差。
f1_knn=f1_score(y_test,y_pred_knn)
print('knn')
print('准确率为:\n',accuracy_knn)
print('精确率为:\n',precision_knn)
print('召回率为:\n',recall_knn)
print('F1score为:\n',f1_knn)

accuracy_svm=accuracy_score(y_test,y_pred_svm)
precision_svm=precision_score(y_test,y_pred_svm)
recall_svm=recall_score(y_test,y_pred_svm)
# F1-Score的取值范围从0到1的，1代表模型的输出最好，0代表模型的输出结果最差。
f1_svm=f1_score(y_test,y_pred_svm)
print('svm')
print('准确率为:\n',accuracy_svm)
print('精确率为:\n',precision_svm)
print('召回率为:\n',recall_svm)
print('F1score为:\n',f1_svm)

accuracy_kernelsvm=accuracy_score(y_test,y_pred_kernelsvm)
precision_kernelsvm=precision_score(y_test,y_pred_kernelsvm)
recall_kernelsvm=recall_score(y_test,y_pred_kernelsvm)
# F1-Score的取值范围从0到1的，1代表模型的输出最好，0代表模型的输出结果最差。
f1_kernelsvm=f1_score(y_test,y_pred_kernelsvm)
print('kernel_svm')
print('准确率为:\n',accuracy_kernelsvm)
print('精确率为:\n',precision_kernelsvm)
print('召回率为:\n',recall_kernelsvm)
print('F1score为:\n',f1_kernelsvm)


#可视化
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Naive Bayes (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
#
#
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Naive Bayes (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()