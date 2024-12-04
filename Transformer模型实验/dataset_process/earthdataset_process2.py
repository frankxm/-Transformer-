import os
# 在py中临时添加的环境变量,防止用kmeans警告
os.environ["OMP_NUM_THREADS"] = '1'
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.cluster import KMeans
import random
# import 只能用绝对导入不能import .Augmentation as aug
from . import Augmentation as aug
from datetime import datetime
from ..utils.random_seed import setup_seed



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


        ### 二分类
        # 找到地震目录在数据中的索引
        rule=df['datetime'].isin(df_time_np)
        ind_all=[i for i in range(len(df))]
        # 用集合方式区分正负样本索引
        ind_list=df[rule].index.tolist()

        ind_pos=df[rule].index.tolist()
        ind_neg = list(set(ind_all) - set(ind_list))
        pos_num=len(ind_list)
        # 随机取负样本
        random.seed(10)
        # 考虑到剩余的‘非地震’索引中，可能会存在不在该地震目录中的地震，所以负样本应该更多
        # neg_num = int(pos_num*1.5)
        neg_num = int(pos_num*0.2)
        neg_list = random.sample(ind_neg, neg_num)
        # 总的样本索引，负样本索引跟在正样本索引后
        ind_list.extend(neg_list)

        # 构造总的标签和时间
        # 地震目录上的时间点标签都为1
        labels = [1] * pos_num
        lab = [0] * neg_num
        labels.extend(lab)

        # 区分训练集测试集
        total_len = pos_num+ neg_num
        sample_num = int(0.8 * total_len)  # 假设取80%的数据
        sample_list = [i for i in range(total_len)]
        train_list = random.sample(sample_list, sample_num)
        test_list = list(set(sample_list) - set(train_list))
        ind_list=np.array(ind_list)
        train_ind=ind_list[train_list]
        test_ind=ind_list[test_list]

        labels = np.array(labels)
        labels_train = labels[train_list]
        labels_test = labels[test_list]
        labels_train += 1
        labels_test += 1
        labels_train = torch.from_numpy(labels_train)
        labels_test = torch.from_numpy(labels_test)


        unique, count = np.unique(labels_train, return_counts=True)
        data_count = dict(zip(unique, count))
        print('旧训练标签各类别出现个数:', data_count)

        unique, count = np.unique(labels_test, return_counts=True)
        data_count = dict(zip(unique, count))
        print('测试标签各类别出现个数:', data_count)

        df = df.drop('datetime', axis=1)
        train_data=[]
        test_data=[]
        train_final=[]
        test_final=[]
        label_train_final=[]
        label_test_final=[]
        def get_data(x):
            df_choosed = df.iloc[x - 10:x + 30]
            df_choosed = np.array(df_choosed)
            mean_choosed = df_choosed.mean()
            std_choosed = df_choosed.std()
            df_choosed = (df_choosed - mean_choosed) / std_choosed
            tensor_choosed = torch.from_numpy(df_choosed).float()
            return tensor_choosed


        if(self.dataset=='train'):
            train_data = list(map(lambda x: get_data(x), train_ind))
            train_data = torch.stack(train_data, dim=0)
            train_aug=train_data.numpy()
            # 添噪声，数据标签不变
            d_jitter = aug.jitter(train_aug)
            # 数据序列改变，不符地震序列，标签都为0
            d_permutation=aug.permutation(train_aug)
            # 数据翻转，趋势没变，标签不变
            d_rotation=aug.rotation(train_aug)
            # 数据缩放，趋势没变，标签不变
            d_scale=aug.scaling(train_aug)
            # 时间扭曲，标签变
            d_timewarp=aug.time_warp(train_aug)
            # 幅度扭曲，标签变
            d_magnitudewarp=aug.magnitude_warp(train_aug)
            tensor_jitter=torch.from_numpy(d_jitter).float()
            tensor_permutation = torch.from_numpy(d_permutation).float()
            tensor_rotation = torch.from_numpy(d_rotation).float()
            tensor_scale = torch.from_numpy(d_scale).float()
            tensor_timewarp = torch.from_numpy(d_timewarp).float()
            tensor_magnitude = torch.from_numpy(d_magnitudewarp).float()

            train_final = torch.cat((train_data, tensor_jitter,tensor_permutation,tensor_rotation,tensor_scale,tensor_timewarp,tensor_magnitude), dim=0)
            label_aug=labels_train
            label_change=torch.from_numpy(np.array([1]*len(train_data)))
            label_train_final=torch.cat((labels_train,label_aug,label_change,label_aug,label_aug,label_change,label_change),dim=0)
            print('训练集准备完成')
        elif(self.dataset=='test'):
            test_data = list(map(lambda x: get_data(x), test_ind))
            test_data = torch.stack(test_data, dim=0)
            test_aug = test_data.numpy()
            d_jitter = aug.jitter(test_aug)
            d_permutation = aug.permutation(test_aug)
            d_rotation = aug.rotation(test_aug)
            d_scale = aug.scaling(test_aug)
            d_timewarp = aug.time_warp(test_aug)
            d_magnitudewarp = aug.magnitude_warp(test_aug)
            tensor_jitter = torch.from_numpy(d_jitter).float()
            tensor_permutation = torch.from_numpy(d_permutation).float()
            tensor_rotation = torch.from_numpy(d_rotation).float()
            tensor_scale = torch.from_numpy(d_scale).float()
            tensor_timewarp = torch.from_numpy(d_timewarp).float()
            tensor_magnitude = torch.from_numpy(d_magnitudewarp).float()

            test_final = torch.cat((test_data, tensor_jitter, tensor_permutation, tensor_rotation, tensor_scale,
                                     tensor_timewarp, tensor_magnitude), dim=0)

            label_aug = labels_test
            label_change = torch.from_numpy(np.array([1] * len(test_data)))
            label_test_final = torch.cat(
                (labels_test, label_aug, label_change, label_aug, label_aug, label_change, label_change), dim=0)

            print('测试集准备完成')

        unique, count = np.unique(label_train_final, return_counts=True)
        data_count = dict(zip(unique, count))
        print('新的训练标签各类别出现个数:', data_count)
        unique, count = np.unique(label_test_final, return_counts=True)
        data_count = dict(zip(unique, count))
        print('新的训练标签各类别出现个数:', data_count)

        train_len=len(train_list)*7
        test_len=len(test_list)*7
        input_len=40
        channel=4
        output_len=2

        return train_len, test_len, input_len, channel, output_len, train_final, label_train_final, test_final, label_test_final
