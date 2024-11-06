
import os
# 在py中临时添加的环境变量,防止用kmeans警告
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import  time
import random
from sklearn.metrics import silhouette_score, silhouette_samples,davies_bouldin_score
from matplotlib.patches import Wedge
import math
from itertools import combinations,permutations
from scipy.spatial import ConvexHull

# 聚类例
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import pairwise_distances_argmin
# from sklearn.datasets._samples_generator import make_blobs
#
# # ######################################
# # Generate sample data
# np.random.seed(0)
#
# batch_size = 45
# centers = [[1, 1], [-1, -1], [1, -1]]
# n_clusters = len(centers)
# X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)
#
# # plot result
# fig = plt.figure(figsize=(8,3))
# fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
# colors = ['#4EACC5', '#FF9C34', '#4E9A06']
#
# # original data
# ax = fig.add_subplot(1,2,1)
# row, col = np.shape(X)
# for i in range(row):
#     ax.plot(X[i, 0], X[i, 1], '#4EACC5', marker='.')
#
# ax.set_title('Original Data')
# ax.set_xticks(())
# ax.set_yticks(())
#
# # compute clustering with K-Means
# k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
# print('k_means',k_means)
# t0 = time.time()
# k_means.fit(X)
# t_batch = time.time() - t0
# print('t_batch',t_batch)
# print('k_means.cluster_centers\n',k_means.cluster_centers_)
# k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
# print(k_means_cluster_centers)
# k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
# print(k_means_labels,len(k_means_labels))
# # K-means
# ax = fig.add_subplot(1, 2, 2)
# for k, col in zip(range(n_clusters), colors):
#     my_members = k_means_labels == k		# my_members是布尔型的数组（用于筛选同类的点，用不同颜色表示）
#     print('k',k,'my_members',my_members)
#     cluster_center = k_means_cluster_centers[k]
#     print('cluster_center',cluster_center)
#     ax.plot(X[my_members, 0], X[my_members, 1], 'w',
#             markerfacecolor=col, marker='.')	# 将同一类的点表示出来
#     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#             markeredgecolor='k', marker='o')	# 将聚类中心单独表示出来
# ax.set_title('KMeans')
# ax.set_xticks(())
# ax.set_yticks(())
# plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' % (t_batch, k_means.inertia_))
# # 聚类信息
# # km.cluster_centers_				# 各聚类中心的坐标
# # km.labels_						# 每个点的标签
# # km.inertia_						# 所有样本到其最近聚类中心的平方距离总和，可表示聚类效果好坏
# # km.n_iter_						# 运行的迭代次数
# plt.show()

# 地震区域分块
# # 设置超参数
N_CLUSTERS = 9                              # 类簇的数量
MARKERS = ['*', 'v', '+', '^', 's', 'x', 'o','<','>']      # 标记样式（绘图）
COLORS = ['pink', 'g', 'm', 'c', 'y', 'b', 'orange','k','yellow']  # 标记颜色（绘图）

DATA_PATH = './data/China_cities.csv'              # 数据集路径
# 海原 银川 临夏 高台
centers = np.array([[105.61,36.51], [105.93,38.61], [103.2,35.6],[99.86,39.4]])


df = pd.read_csv('./Data/Area1_3~10.csv',encoding = 'gb2312')
x= df[['纬度(°)','经度(°)']]
x_np = np.array(x)        # 将x转化为numpy数组
print(df)
#
# 分块之前，利用SSE选择k
# 手肘法的核心指标是SSE(sum of the squared errors，误差平方和)，SSE是所有样本的聚类误差，代表了聚类效果的好坏。
SSE = []  # 存放每次结果的误差平方和
class_dis=[]
sil_score=[]
db_score=[]
for k in range(2,50):
    estimator = KMeans(n_clusters=k,random_state=2)  # 构造聚类器
    estimator.fit(x)
    SSE.append(estimator.inertia_)
    labels = estimator.labels_      # 获取聚类标签
    print('获取聚类结果总的轮廓系数',silhouette_score(x, labels))      # 获取聚类结果总的轮廓系数
    sil_score.append(silhouette_score(x,labels))
    db_score.append(davies_bouldin_score(x, labels))
    # 计算类间距离和
    classsum = 0.0
    centers = estimator.cluster_centers_.tolist()
    for center1 in centers:
        for center2 in centers:
            classsum = classsum + abs(center1[0] - center2[0])
    class_dis.append(classsum)
    print('类间距离为',classsum)

X = range(2,50)
plt.figure(1)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X,SSE)
plt.figure(2)
plt.xlabel('k')
plt.ylabel('class_distance')
plt.plot(X,class_dis)
plt.figure(3)
plt.xlabel('k')
plt.ylabel('sil_score')
plt.plot(X,sil_score)
plt.figure(4)
plt.xlabel('k')
plt.ylabel('davies_score')
plt.plot(X,db_score)
plt.show()
#
# # 模型构建
# model = KMeans(N_CLUSTERS,random_state=2)      # 构建聚类器
# model.fit(x)                    # 训练聚类器
#
# labels = model.labels_      # 获取聚类标签
# print('聚类标签',labels,len(labels),type(labels))
#
# unique,count=np.unique(labels,return_counts=True)
# data_count=dict(zip(unique,count))
# print('各类别出现个数:',data_count)
# #
# # # 轮廓系数（Silhouette Coefficient）是一种评价聚类效果的方法，其值介于[-1,1]之间，值越趋近于1代表同簇点的内聚度和异簇点的分离度都相对较优。
# # print('获取所有样本的轮廓系数',silhouette_samples(x, labels))  # 获取所有样本的轮廓系数
# # # 将所有点的轮廓系数求平均，就能得到该聚类结果总的轮廓系数
# # print('获取聚类结果总的轮廓系数',silhouette_score(x, labels))      # 获取聚类结果总的轮廓系数
# # print('输出类簇中心如下\n',model.cluster_centers_)	# 输出类簇中心
# # # 输出各簇内元素
# # for i in range(N_CLUSTERS):
# #     print(f" CLUSTER-{i+1} ".center(60, '='))
# #     print(df[labels == i])
#
#
# # 可视化
# fig,ax=plt.subplots()
# plt.title("Earthquakes in Area1", fontsize=22)
# plt.xlabel('East Longitude', fontsize=18)
# plt.ylabel('North Latitude', fontsize=18)
#
# p_list=[]
# for i in range(N_CLUSTERS):
#     members = labels == i      # members是一个布尔型数组
#     p=ax.scatter(
#         x_np[members, 1],      # 城市经度数组
#         x_np[members, 0],      # 城市纬度数组
#         marker = MARKERS[i],   # 标记样式
#         c = COLORS[i]          # 标记颜色
#     )   # 绘制散点图
#     p_list.append(p)
#
# ax.scatter(centers[:,0],centers[:,1],c='red',s=150,alpha=0.8)
# plt.grid()
# plt.legend(handles=p_list, labels=['zone0', 'zone1','zone2','zone3','zone4','zone5','zone6','zone7','zone8'], loc='best')
#
# color=['blue','red','green','pink']
# def detection(angle,center,index):
#     ang=90-angle
#     detect_len = 2
#     detect_x = detect_len * math.cos( ang/ 180 * math.pi)
#     detect_y = detect_len * math.sin(ang/ 180 * math.pi)
#     detect_dir1 = np.array([center[0] + detect_x, center[1] + detect_y])
#     ax.scatter(detect_dir1[0], detect_dir1[1], c=color[index], s=20)
#     ang1=ang-22.5
#     ang2=ang1+45
#     semicircle = Wedge((center[0], center[1]), 2, ang1,ang2,alpha=0.3,color=color[index])
#     ax.add_patch(semicircle)
#
# def disposer(start_angle,center):
#     # 画各个方向的范围图以及方向中心点
#     for i in range(8):
#         detection(start_angle+45*i,center,i%4)
#
# disposer(111,centers[0])
# # disposer(40,centers[1])
# # disposer(92,centers[2])
# # disposer(295,centers[3])
# #设置xy轴等长，否则不圆
# plt.axis('equal')
# plt.show()
#
#
#
# # # 随机生成范围内的经纬度点
# # points=[]
# # for a in range(0,1000):
# #     lat = random.uniform(33, 41)
# #     lng = random.uniform(96, 107)
# #     point = (lng, lat)
# #     p=np.array(point)
# #     if a==0:
# #         points=p
# #         continue
# #     points=np.vstack((points, p))
# # print('随机生成的经纬点',points)
# # model.fit(points)
# # labels = model.labels_      # 获取聚类标签
# # print('随机点的聚类标签',labels,len(labels))
#
#
#
# # 检验点的区域
# indexer = np.argwhere(x_np==np.array([40.24,97.85
# ]))
# ind=indexer[:,0]
# print(ind,type(ind))
# indexe=np.argmax(np.bincount(ind))
# print('当前点标签为',labels[indexe])
# print('索引为',indexe,x_np[indexe])
#
# # 检验点的区域
# indexer = np.argwhere(x_np==np.array([33.01,96.99
# ]))
# ind=indexer[:,0]
# print(ind,type(ind))
# indexe=np.argmax(np.bincount(ind))
# print('当前点标签为',labels[indexe])
# print('索引为',indexe,x_np[indexe])
#
# # 检验点的区域
# indexer = np.argwhere(x_np==np.array([33.29,96.2
# ]))
# ind=indexer[:,0]
# print(ind,type(ind))
# indexe=np.argmax(np.bincount(ind))
# print('当前点标签为',labels[indexe])
# print('索引为',indexe,x_np[indexe])
#
# # 检验点的区域
# indexer = np.argwhere(x_np==np.array([38.27,106.2
# ]))
# ind=indexer[:,0]
# print(ind,type(ind))
# indexe=np.argmax(np.bincount(ind))
# print('当前点标签为',labels[indexe])
# print('索引为',indexe,x_np[indexe])

arr=x_np
hull = ConvexHull(arr)  ###计算外接凸图案的边界点
plt.plot(arr[:,0], arr[:,1], 'o')
# plot convex hull polygon
plt.plot(arr[hull.vertices,0], arr[hull.vertices,1], 'r ', lw=4)
# plot convex full vertices
hull1=hull.vertices.tolist()
hull1.append(hull1[0])
# plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
plt.plot(arr[hull1,0], arr[hull1,1], 'r--^',lw=2)
plt.show()