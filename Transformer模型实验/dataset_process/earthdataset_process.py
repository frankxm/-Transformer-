import os
# 在py中临时添加的环境变量,防止用kmeans警告
os.environ["OMP_NUM_THREADS"] = '1'
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.cluster import KMeans
import random
from datetime import datetime
from utils.random_seed import setup_seed



class MyDataset(Dataset):
    def __init__(self,
                 dataset: str,df,df_area):
        """
        训练数据集与测试数据集的Dataset对象
        :param path: 数据集路径
        :param dataset: 区分是获得训练集还是测试集
        """
        super(MyDataset, self).__init__()
        self.dataset = dataset  # 选择获取测试集还是训练集
        self.train_len, \
        self.test_len, \
        self.input_len, \
        self.channel_len, \
        self.output_len, \
        self.train_dataset, \
        self.train_label, \
        self.test_dataset, \
        self.test_label = self.make_data(df,df_area)

    def __getitem__(self, index):
        if self.dataset == 'train':
            return self.train_dataset[index], self.train_label[index] - 1
        elif self.dataset == 'test':
            return self.test_dataset[index], self.test_label[index] - 1

    def __len__(self):
        if self.dataset == 'train':
            return self.train_len
        elif self.dataset == 'test':
            return self.test_len

    def make_data(self, df,df_area):

        setup_seed(3047)  # 设置随机数种子
        df_earth = df_area['发震日期（北京时间）']
        df_earth1 = pd.to_datetime(df_earth)
        df_time = df_earth1.dt.strftime('%Y-%m-%d-%H:%M')
        df_time_np = np.array(df_time)

        x_area = df_area[['纬度(°)', '经度(°)']]
        N_CLUSTERS = 9
        # 模型构建
        model = KMeans(N_CLUSTERS, random_state=2)  # 构建聚类器
        model.fit(x_area)  # 训练聚类器
        labels = model.labels_  # 获取聚类标签

        # 找到地震目录在数据中的索引
        rule=df['datetime'].isin(df_time_np)
        ind_list=df[rule].index.tolist()
        sample_len=len(ind_list)


        # 区分训练集测试集
        sample_num = int(0.8 * sample_len)  # 假设取80%的数据
        sample_list = [i for i in range(sample_len)]
        train_list = random.sample(sample_list, sample_num)
        test_list = list(set(sample_list) - set(train_list))
        ind_list=np.array(ind_list)
        test_list_reverse = [sample_len - 1 - l for l in test_list]
        train_list_reverse=[sample_len - 1 - l for l in train_list]
        train_ind=ind_list[train_list_reverse]
        test_ind=ind_list[test_list_reverse]


        labels = np.array(labels)
        labels_train = labels[train_list]
        labels_test = labels[test_list]
        labels_train += 1
        labels_test += 1
        labels_train = torch.from_numpy(labels_train)
        labels_test = torch.from_numpy(labels_test)

        unique, count = np.unique(labels_train, return_counts=True)
        data_count = dict(zip(unique, count))
        print('训练标签各类别出现个数:', data_count)

        unique, count = np.unique(labels_test, return_counts=True)
        data_count = dict(zip(unique, count))
        print('测试标签各类别出现个数:', data_count)

        df_copy=df.copy()
        df = df.drop('datetime', axis=1)
        train_data=[]
        test_data=[]
        test_time_list=[]
        def get_data(x):
            test_time_list.append(df_copy['datetime'].iloc[x])
            df_choosed = df.iloc[x - 10:x + 30]
            df_choosed = np.array(df_choosed)
            # z-score标准化
            mean_choosed = df_choosed.mean()
            std_choosed = df_choosed.std()
            df_choosed = (df_choosed - mean_choosed) / std_choosed
            # min-max标准化
            # max_choosed=df_choosed.max()
            # min_choosed=df_choosed.min()
            # df_choosed = ((df_choosed - min_choosed) / (max_choosed - min_choosed))
            tensor_choosed = torch.from_numpy(df_choosed).float()
            return tensor_choosed

        if(self.dataset=='train'):
            train_data = list(map(lambda x: get_data(x), train_ind))
            train_data = torch.stack(train_data, dim=0)
            print('训练集准备完成')
        elif(self.dataset=='test'):
            test_data = list(map(lambda x: get_data(x), test_ind))
            test_data = torch.stack(test_data, dim=0)
            print(f'使用的数据所在时间为{test_time_list}\n长度为{len(test_time_list)}')
            print('测试集准备完成')




        # ### 二分类
        # # 找到地震目录在数据中的索引
        # rule=df['datetime'].isin(df_time_np)
        # ind_all=[i for i in range(len(df))]
        # # 用集合方式区分正负样本索引
        # ind_list=df[rule].index.tolist()
        # ind_neg = list(set(ind_all) - set(ind_list))
        # pos_num=len(ind_list)
        # # 随机取负样本
        # random.seed(10)
        # neg_num = pos_num
        # neg_list = random.sample(ind_neg, neg_num)
        # # 总的样本索引，负样本索引跟在正样本索引后
        # ind_list.extend(neg_list)
        #
        # # 构造总的标签和时间
        # # 地震目录上的时间点标签都为1
        # labels = [1] * pos_num
        # lab = [0] * neg_num
        # labels.extend(lab)
        #
        # # 区分训练集测试集
        # total_len = pos_num+ neg_num
        # sample_num = int(0.8 * total_len)  # 假设取80%的数据
        # sample_list = [i for i in range(total_len)]
        # train_list = random.sample(sample_list, sample_num)
        # test_list = list(set(sample_list) - set(train_list))
        # ind_list=np.array(ind_list)
        # train_ind=ind_list[train_list]
        # test_ind=ind_list[test_list]
        #
        # labels = np.array(labels)
        # labels_train = labels[train_list]
        # labels_test = labels[test_list]
        # labels_train += 1
        # labels_test += 1
        # labels_train = torch.from_numpy(labels_train)
        # labels_test = torch.from_numpy(labels_test)
        #
        # df = df.drop('datetime', axis=1)
        # train_data=[]
        # test_data=[]
        #
        # def get_data(x):
        #     df_choosed = df.iloc[x - 10:x + 30]
        #     df_choosed = np.array(df_choosed)
        #     mean_choosed = df_choosed.mean()
        #     std_choosed = df_choosed.std()
        #     df_choosed = (df_choosed - mean_choosed) / std_choosed
        #     tensor_choosed = torch.from_numpy(df_choosed).float()
        #     return tensor_choosed
        #
        # if(self.dataset=='train'):
        #     train_data = list(map(lambda x: get_data(x), train_ind))
        #     train_data = torch.stack(train_data, dim=0)
        #     print('训练集准备完成')
        # elif(self.dataset=='test'):
        #     test_data = list(map(lambda x: get_data(x), test_ind))
        #     test_data = torch.stack(test_data, dim=0)
        #     print('测试集准备完成')


        train_len=len(train_list)
        test_len=len(test_list)
        input_len=40
        channel=6
        output_len=9

        return train_len, test_len, input_len, channel, output_len, train_data, labels_train, test_data, labels_test
