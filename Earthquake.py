import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import time
import numpy as np
from scipy.signal import find_peaks
from datetime import date
from math import radians, cos, sin, asin, sqrt
from .Transformer模型实验.dataset_process import Augmentation as aug

time1 = ['00:00', '00:01', '00:02', '00:03', '00:04', '00:05', '00:06', '00:07', '00:08', '00:09', '00:10', '00:11', '00:12', '00:13', '00:14', '00:15', '00:16', '00:17', '00:18', '00:19', '00:20', '00:21', '00:22', '00:23', '00:24', '00:25', '00:26', '00:27', '00:28', '00:29', '00:30', '00:31', '00:32', '00:33', '00:34', '00:35', '00:36', '00:37', '00:38', '00:39', '00:40', '00:41', '00:42', '00:43', '00:44', '00:45', '00:46', '00:47', '00:48', '00:49', '00:50', '00:51', '00:52', '00:53', '00:54', '00:55', '00:56', '00:57', '00:58', '00:59', '01:00', '01:01', '01:02', '01:03', '01:04', '01:05', '01:06', '01:07', '01:08', '01:09', '01:10', '01:11', '01:12', '01:13', '01:14', '01:15', '01:16', '01:17', '01:18', '01:19', '01:20', '01:21', '01:22', '01:23', '01:24', '01:25', '01:26', '01:27', '01:28', '01:29', '01:30', '01:31', '01:32', '01:33', '01:34', '01:35', '01:36', '01:37', '01:38', '01:39', '01:40', '01:41', '01:42', '01:43', '01:44', '01:45', '01:46', '01:47', '01:48', '01:49', '01:50', '01:51', '01:52', '01:53', '01:54', '01:55', '01:56', '01:57', '01:58', '01:59', '02:00', '02:01', '02:02', '02:03', '02:04', '02:05', '02:06', '02:07', '02:08', '02:09', '02:10', '02:11', '02:12', '02:13', '02:14', '02:15', '02:16', '02:17', '02:18', '02:19', '02:20', '02:21', '02:22', '02:23', '02:24', '02:25', '02:26', '02:27', '02:28', '02:29', '02:30', '02:31', '02:32', '02:33', '02:34', '02:35', '02:36', '02:37', '02:38', '02:39', '02:40', '02:41', '02:42', '02:43', '02:44', '02:45', '02:46', '02:47', '02:48', '02:49', '02:50', '02:51', '02:52', '02:53', '02:54', '02:55', '02:56', '02:57', '02:58', '02:59', '03:00', '03:01', '03:02', '03:03', '03:04', '03:05', '03:06', '03:07', '03:08', '03:09', '03:10', '03:11', '03:12', '03:13', '03:14', '03:15', '03:16', '03:17', '03:18', '03:19', '03:20', '03:21', '03:22', '03:23', '03:24', '03:25', '03:26', '03:27', '03:28', '03:29', '03:30', '03:31', '03:32', '03:33', '03:34', '03:35', '03:36', '03:37', '03:38', '03:39', '03:40', '03:41', '03:42', '03:43', '03:44', '03:45', '03:46', '03:47', '03:48', '03:49', '03:50', '03:51', '03:52', '03:53', '03:54', '03:55', '03:56', '03:57', '03:58', '03:59', '04:00', '04:01', '04:02', '04:03', '04:04', '04:05', '04:06', '04:07', '04:08', '04:09', '04:10', '04:11', '04:12', '04:13', '04:14', '04:15', '04:16', '04:17', '04:18', '04:19', '04:20', '04:21', '04:22', '04:23', '04:24', '04:25', '04:26', '04:27', '04:28', '04:29', '04:30', '04:31', '04:32', '04:33', '04:34', '04:35', '04:36', '04:37', '04:38', '04:39', '04:40', '04:41', '04:42', '04:43', '04:44', '04:45', '04:46', '04:47', '04:48', '04:49', '04:50', '04:51', '04:52', '04:53', '04:54', '04:55', '04:56', '04:57', '04:58', '04:59', '05:00', '05:01', '05:02', '05:03', '05:04', '05:05', '05:06', '05:07', '05:08', '05:09', '05:10', '05:11', '05:12', '05:13', '05:14', '05:15', '05:16', '05:17', '05:18', '05:19', '05:20', '05:21', '05:22', '05:23', '05:24', '05:25', '05:26', '05:27', '05:28', '05:29', '05:30', '05:31', '05:32', '05:33', '05:34', '05:35', '05:36', '05:37', '05:38', '05:39', '05:40', '05:41', '05:42', '05:43', '05:44', '05:45', '05:46', '05:47', '05:48', '05:49', '05:50', '05:51', '05:52', '05:53', '05:54', '05:55', '05:56', '05:57', '05:58', '05:59', '06:00', '06:01', '06:02', '06:03', '06:04', '06:05', '06:06', '06:07', '06:08', '06:09', '06:10', '06:11', '06:12', '06:13', '06:14', '06:15', '06:16', '06:17', '06:18', '06:19', '06:20', '06:21', '06:22', '06:23', '06:24', '06:25', '06:26', '06:27', '06:28', '06:29', '06:30', '06:31', '06:32', '06:33', '06:34', '06:35', '06:36', '06:37', '06:38', '06:39', '06:40', '06:41', '06:42', '06:43', '06:44', '06:45', '06:46', '06:47', '06:48', '06:49', '06:50', '06:51', '06:52', '06:53', '06:54', '06:55', '06:56', '06:57', '06:58', '06:59', '07:00', '07:01', '07:02', '07:03', '07:04', '07:05', '07:06', '07:07', '07:08', '07:09', '07:10', '07:11', '07:12', '07:13', '07:14', '07:15', '07:16', '07:17', '07:18', '07:19', '07:20', '07:21', '07:22', '07:23', '07:24', '07:25', '07:26', '07:27', '07:28', '07:29', '07:30', '07:31', '07:32', '07:33', '07:34', '07:35', '07:36', '07:37', '07:38', '07:39', '07:40', '07:41', '07:42', '07:43', '07:44', '07:45', '07:46', '07:47', '07:48', '07:49', '07:50', '07:51', '07:52', '07:53', '07:54', '07:55', '07:56', '07:57', '07:58', '07:59', '08:00', '08:01', '08:02', '08:03', '08:04', '08:05', '08:06', '08:07', '08:08', '08:09', '08:10', '08:11', '08:12', '08:13', '08:14', '08:15', '08:16', '08:17', '08:18', '08:19', '08:20', '08:21', '08:22', '08:23', '08:24', '08:25', '08:26', '08:27', '08:28', '08:29', '08:30', '08:31', '08:32', '08:33', '08:34', '08:35', '08:36', '08:37', '08:38', '08:39', '08:40', '08:41', '08:42', '08:43', '08:44', '08:45', '08:46', '08:47', '08:48', '08:49', '08:50', '08:51', '08:52', '08:53', '08:54', '08:55', '08:56', '08:57', '08:58', '08:59', '09:00', '09:01', '09:02', '09:03', '09:04', '09:05', '09:06', '09:07', '09:08', '09:09', '09:10', '09:11', '09:12', '09:13', '09:14', '09:15', '09:16', '09:17', '09:18', '09:19', '09:20', '09:21', '09:22', '09:23', '09:24', '09:25', '09:26', '09:27', '09:28', '09:29', '09:30', '09:31', '09:32', '09:33', '09:34', '09:35', '09:36', '09:37', '09:38', '09:39', '09:40', '09:41', '09:42', '09:43', '09:44', '09:45', '09:46', '09:47', '09:48', '09:49', '09:50', '09:51', '09:52', '09:53', '09:54', '09:55', '09:56', '09:57', '09:58', '09:59', '10:00', '10:01', '10:02', '10:03', '10:04', '10:05', '10:06', '10:07', '10:08', '10:09', '10:10', '10:11', '10:12', '10:13', '10:14', '10:15', '10:16', '10:17', '10:18', '10:19', '10:20', '10:21', '10:22', '10:23', '10:24', '10:25', '10:26', '10:27', '10:28', '10:29', '10:30', '10:31', '10:32', '10:33', '10:34', '10:35', '10:36', '10:37', '10:38', '10:39', '10:40', '10:41', '10:42', '10:43', '10:44', '10:45', '10:46', '10:47', '10:48', '10:49', '10:50', '10:51', '10:52', '10:53', '10:54', '10:55', '10:56', '10:57', '10:58', '10:59', '11:00', '11:01', '11:02', '11:03', '11:04', '11:05', '11:06', '11:07', '11:08', '11:09', '11:10', '11:11', '11:12', '11:13', '11:14', '11:15', '11:16', '11:17', '11:18', '11:19', '11:20', '11:21', '11:22', '11:23', '11:24', '11:25', '11:26', '11:27', '11:28', '11:29', '11:30', '11:31', '11:32', '11:33', '11:34', '11:35', '11:36', '11:37', '11:38', '11:39', '11:40', '11:41', '11:42', '11:43', '11:44', '11:45', '11:46', '11:47', '11:48', '11:49', '11:50', '11:51', '11:52', '11:53', '11:54', '11:55', '11:56', '11:57', '11:58', '11:59', '12:00', '12:01', '12:02', '12:03', '12:04', '12:05', '12:06', '12:07', '12:08', '12:09', '12:10', '12:11', '12:12', '12:13', '12:14', '12:15', '12:16', '12:17', '12:18', '12:19', '12:20', '12:21', '12:22', '12:23', '12:24', '12:25', '12:26', '12:27', '12:28', '12:29', '12:30', '12:31', '12:32', '12:33', '12:34', '12:35', '12:36', '12:37', '12:38', '12:39', '12:40', '12:41', '12:42', '12:43', '12:44', '12:45', '12:46', '12:47', '12:48', '12:49', '12:50', '12:51', '12:52', '12:53', '12:54', '12:55', '12:56', '12:57', '12:58', '12:59', '13:00', '13:01', '13:02', '13:03', '13:04', '13:05', '13:06', '13:07', '13:08', '13:09', '13:10', '13:11', '13:12', '13:13', '13:14', '13:15', '13:16', '13:17', '13:18', '13:19', '13:20', '13:21', '13:22', '13:23', '13:24', '13:25', '13:26', '13:27', '13:28', '13:29', '13:30', '13:31', '13:32', '13:33', '13:34', '13:35', '13:36', '13:37', '13:38', '13:39', '13:40', '13:41', '13:42', '13:43', '13:44', '13:45', '13:46', '13:47', '13:48', '13:49', '13:50', '13:51', '13:52', '13:53', '13:54', '13:55', '13:56', '13:57', '13:58', '13:59', '14:00', '14:01', '14:02', '14:03', '14:04', '14:05', '14:06', '14:07', '14:08', '14:09', '14:10', '14:11', '14:12', '14:13', '14:14', '14:15', '14:16', '14:17', '14:18', '14:19', '14:20', '14:21', '14:22', '14:23', '14:24', '14:25', '14:26', '14:27', '14:28', '14:29', '14:30', '14:31', '14:32', '14:33', '14:34', '14:35', '14:36', '14:37', '14:38', '14:39', '14:40', '14:41', '14:42', '14:43', '14:44', '14:45', '14:46', '14:47', '14:48', '14:49', '14:50', '14:51', '14:52', '14:53', '14:54', '14:55', '14:56', '14:57', '14:58', '14:59', '15:00', '15:01', '15:02', '15:03', '15:04', '15:05', '15:06', '15:07', '15:08', '15:09', '15:10', '15:11', '15:12', '15:13', '15:14', '15:15', '15:16', '15:17', '15:18', '15:19', '15:20', '15:21', '15:22', '15:23', '15:24', '15:25', '15:26', '15:27', '15:28', '15:29', '15:30', '15:31', '15:32', '15:33', '15:34', '15:35', '15:36', '15:37', '15:38', '15:39', '15:40', '15:41', '15:42', '15:43', '15:44', '15:45', '15:46', '15:47', '15:48', '15:49', '15:50', '15:51', '15:52', '15:53', '15:54', '15:55', '15:56', '15:57', '15:58', '15:59', '16:00', '16:01', '16:02', '16:03', '16:04', '16:05', '16:06', '16:07', '16:08', '16:09', '16:10', '16:11', '16:12', '16:13', '16:14', '16:15', '16:16', '16:17', '16:18', '16:19', '16:20', '16:21', '16:22', '16:23', '16:24', '16:25', '16:26', '16:27', '16:28', '16:29', '16:30', '16:31', '16:32', '16:33', '16:34', '16:35', '16:36', '16:37', '16:38', '16:39', '16:40', '16:41', '16:42', '16:43', '16:44', '16:45', '16:46', '16:47', '16:48', '16:49', '16:50', '16:51', '16:52', '16:53', '16:54', '16:55', '16:56', '16:57', '16:58', '16:59', '17:00', '17:01', '17:02', '17:03', '17:04', '17:05', '17:06', '17:07', '17:08', '17:09', '17:10', '17:11', '17:12', '17:13', '17:14', '17:15', '17:16', '17:17', '17:18', '17:19', '17:20', '17:21', '17:22', '17:23', '17:24', '17:25', '17:26', '17:27', '17:28', '17:29', '17:30', '17:31', '17:32', '17:33', '17:34', '17:35', '17:36', '17:37', '17:38', '17:39', '17:40', '17:41', '17:42', '17:43', '17:44', '17:45', '17:46', '17:47', '17:48', '17:49', '17:50', '17:51', '17:52', '17:53', '17:54', '17:55', '17:56', '17:57', '17:58', '17:59', '18:00', '18:01', '18:02', '18:03', '18:04', '18:05', '18:06', '18:07', '18:08', '18:09', '18:10', '18:11', '18:12', '18:13', '18:14', '18:15', '18:16', '18:17', '18:18', '18:19', '18:20', '18:21', '18:22', '18:23', '18:24', '18:25', '18:26', '18:27', '18:28', '18:29', '18:30', '18:31', '18:32', '18:33', '18:34', '18:35', '18:36', '18:37', '18:38', '18:39', '18:40', '18:41', '18:42', '18:43', '18:44', '18:45', '18:46', '18:47', '18:48', '18:49', '18:50', '18:51', '18:52', '18:53', '18:54', '18:55', '18:56', '18:57', '18:58', '18:59', '19:00', '19:01', '19:02', '19:03', '19:04', '19:05', '19:06', '19:07', '19:08', '19:09', '19:10', '19:11', '19:12', '19:13', '19:14', '19:15', '19:16', '19:17', '19:18', '19:19', '19:20', '19:21', '19:22', '19:23', '19:24', '19:25', '19:26', '19:27', '19:28', '19:29', '19:30', '19:31', '19:32', '19:33', '19:34', '19:35', '19:36', '19:37', '19:38', '19:39', '19:40', '19:41', '19:42', '19:43', '19:44', '19:45', '19:46', '19:47', '19:48', '19:49', '19:50', '19:51', '19:52', '19:53', '19:54', '19:55', '19:56', '19:57', '19:58', '19:59', '20:00', '20:01', '20:02', '20:03', '20:04', '20:05', '20:06', '20:07', '20:08', '20:09', '20:10', '20:11', '20:12', '20:13', '20:14', '20:15', '20:16', '20:17', '20:18', '20:19', '20:20', '20:21', '20:22', '20:23', '20:24', '20:25', '20:26', '20:27', '20:28', '20:29', '20:30', '20:31', '20:32', '20:33', '20:34', '20:35', '20:36', '20:37', '20:38', '20:39', '20:40', '20:41', '20:42', '20:43', '20:44', '20:45', '20:46', '20:47', '20:48', '20:49', '20:50', '20:51', '20:52', '20:53', '20:54', '20:55', '20:56', '20:57', '20:58', '20:59', '21:00', '21:01', '21:02', '21:03', '21:04', '21:05', '21:06', '21:07', '21:08', '21:09', '21:10', '21:11', '21:12', '21:13', '21:14', '21:15', '21:16', '21:17', '21:18', '21:19', '21:20', '21:21', '21:22', '21:23', '21:24', '21:25', '21:26', '21:27', '21:28', '21:29', '21:30', '21:31', '21:32', '21:33', '21:34', '21:35', '21:36', '21:37', '21:38', '21:39', '21:40', '21:41', '21:42', '21:43', '21:44', '21:45', '21:46', '21:47', '21:48', '21:49', '21:50', '21:51', '21:52', '21:53', '21:54', '21:55', '21:56', '21:57', '21:58', '21:59', '22:00', '22:01', '22:02', '22:03', '22:04', '22:05', '22:06', '22:07', '22:08', '22:09', '22:10', '22:11', '22:12', '22:13', '22:14', '22:15', '22:16', '22:17', '22:18', '22:19', '22:20', '22:21', '22:22', '22:23', '22:24', '22:25', '22:26', '22:27', '22:28', '22:29', '22:30', '22:31', '22:32', '22:33', '22:34', '22:35', '22:36', '22:37', '22:38', '22:39', '22:40', '22:41', '22:42', '22:43', '22:44', '22:45', '22:46', '22:47', '22:48', '22:49', '22:50', '22:51', '22:52', '22:53', '22:54', '22:55', '22:56', '22:57', '22:58', '22:59', '23:00', '23:01', '23:02', '23:03', '23:04', '23:05', '23:06', '23:07', '23:08', '23:09', '23:10', '23:11', '23:12', '23:13', '23:14', '23:15', '23:16', '23:17', '23:18', '23:19', '23:20', '23:21', '23:22', '23:23', '23:24', '23:25', '23:26', '23:27', '23:28', '23:29', '23:30', '23:31', '23:32', '23:33', '23:34', '23:35', '23:36', '23:37', '23:38', '23:39', '23:40', '23:41', '23:42', '23:43', '23:44', '23:45', '23:46', '23:47', '23:48', '23:49', '23:50', '23:51', '23:52', '23:53', '23:54', '23:55', '23:56', '23:57', '23:58', '23:59']

# 第一次读取转换完后面直接读现成的
def get_direction(url):

    datalist = []
    step = []
    datalist2 = []
    year = []
    with open(url, 'r') as f:
        lines = f.readlines()
        j = 0
        for line in lines:
            # 记录有几行数据，即几天
            j += 1
            year.append(line[0:4] + '-' + line[4:6] + '-' + line[6:8])
            # 取日期后面的数据并以空格分割获得储存字符的列表
            temp = line[9:].strip()
            temp_str = temp.split(' ')
            temp_str2 = list(map(float, temp_str))
            # 每次合并字符列表记录次数
            datalist.extend(temp_str)
            datalist2.append(temp_str2)
            step.append(j)
    datalist = list(map(float, datalist))
    print('共有{:.0f}天{}个数据'.format(len(datalist) / 1440, len(datalist)))
    date = []
    for y in year:
        for t in time1:
            datetime = y + '-' + t
            date.append(datetime)
    return datalist,date

def get_direction2(url):

    datalist = []
    date = []
    with open(url, 'r') as f:
        lines = f.readlines()
        for line in lines:
            date.append(line[0:4] + '-' + line[4:6] + '-' + line[6:8]+'-'+line[8:10]+':'+line[10:12])
            # 取日期后面的数据并以空格分割获得储存字符的列表
            temp = line[13:].strip()
            # 每次合并字符列表记录次数
            datalist.append(temp)

    datalist = list(map(float, datalist))
    print('共有{:.0f}天{}个数据'.format(len(datalist) / 1440, len(datalist)))

    return datalist, date

# direction5,date5=get_direction2(r"D:\钻孔数据\临夏\62053_1_2321_分.TXT")
# direction6,date6=get_direction2(r"D:\钻孔数据\临夏\62053_1_2322_分.TXT")
# direction7,date7=get_direction2(r"D:\钻孔数据\临夏\62053_1_2323_分.TXT")
# direction8,date8=get_direction2(r"D:\钻孔数据\临夏\62053_1_2324_分.TXT")
# direction=['direction_fifth','direction_sixth','direction_seventh','direction_eighth']
# # 可能不同方向数据量不同，以少的为准
# data_len=min(len(direction5),len(direction6),len(direction7),len(direction8))
# direction5=direction5[0:data_len]
# direction6=direction6[0:data_len]
# direction7=direction7[0:data_len]
# direction8=direction8[0:data_len]
# date5=date5[0:data_len]
#
# # direction1,date1=get_direction(r"D:\钻孔数据\海源台\64021_2_2321_分.TXT")
# # direction2,date2=get_direction(r"D:\钻孔数据\海源台\64021_2_2322_分.TXT")
# # direction3,date3=get_direction(r"D:\钻孔数据\海源台\64021_2_2323_分.TXT")
# # direction4,date4=get_direction(r"D:\钻孔数据\海源台\64021_2_2324_分.TXT")
# # direction=['direction_first','direction_second','direction_third','direction_fourth']
# # data_len=min(len(direction1),len(direction2),len(direction3),len(direction4))

# # df=pd.DataFrame({'direction_first':direction1, 'direction_second': direction2,'direction_third':direction3,'direction_fourth':direction4}, index=date1)
# df=pd.DataFrame({'direction_fifth':direction5, 'direction_sixth': direction6,'direction_seventh':direction7,'direction_eighth':direction8}, index=date5)
#
# print(df.head(),df.shape)
# # df.to_csv('./Data/haiyuan.csv', encoding='utf_8_sig')
# df.to_csv('./Data/linxia.csv', encoding='utf_8_sig')

# # 合并haiyuan linxia
# # 读取现成的
# # 块读取，chunkSize规定每次读取多少行，之后合并成一个大的dataframe
# df = pd.read_csv(r'./Data/haiyuan_new.csv', sep=',',engine = 'python',iterator=True)
# loop = True
# chunkSize = 100000
# chunks = []
# index=0
# while loop:
#     try:
#         print(index)
#         chunk = df.get_chunk(chunkSize)
#         chunks.append(chunk)
#         index+=1
#
#     except StopIteration:
#         loop = False
#         print("Iteration is stopped.")
# print('开始合并')
# df1 = pd.concat(chunks, ignore_index= True)
# df1.rename( columns={'Unnamed: 0':'datetime'}, inplace=True )
# print(df1)
#
# df = pd.read_csv(r'./Data/linxia_new2.csv', sep=',',engine = 'python',iterator=True)
# loop = True
# chunkSize = 100000
# chunks = []
# index=0
# while loop:
#     try:
#         print(index)
#         chunk = df.get_chunk(chunkSize)
#         chunks.append(chunk)
#         index+=1
#
#     except StopIteration:
#         loop = False
#         print("Iteration is stopped.")
# print('开始合并')
# df2 = pd.concat(chunks, ignore_index= True)
# df2.rename( columns={'Unnamed: 0':'datetime'}, inplace=True )
# print(df2)
# concat_len=min(df1.shape[0],df2.shape[0])
# df1=df1.iloc[0:concat_len]
# df2=df2.iloc[0:concat_len]
# df2 = df2.drop('datetime', axis=1)
# df_final=pd.concat([df1,df2],axis=1)
# # print(df_final)
# # print("shape of data:", df_final.shape)
# # print("缺失的数据:\n",df_final.isnull().sum())
# # print('具体缺失查看\n',df_final.isnull().any())
# # print('data_shape\n',df_final.shape)
# # print('各字段情况\n',df_final.describe())
# # print('是否有重复值\n',df_final.duplicated().value_counts())
# df_final.to_csv('./Data/haiyuan_linxia.csv', encoding='utf_8_sig',index=False)



# 读取现成的
# 块读取，chunkSize规定每次读取多少行，之后合并成一个大的dataframe
df = pd.read_csv(r'./Data/haiyuan.csv', sep=',',engine = 'python',iterator=True)
loop = True
chunkSize = 100000
chunks = []
index=0
while loop:
    try:
        print(index)
        chunk = df.get_chunk(chunkSize)
        chunks.append(chunk)
        index+=1

    except StopIteration:
        loop = False
        print("Iteration is stopped.")
print('开始合并')
df = pd.concat(chunks, ignore_index= True)
df.rename( columns={'Unnamed: 0':'datetime'}, inplace=True )

# #显示所有的列
# pd.set_option('display.max_columns', None)
# #显示所有的行
# pd.set_option('display.max_rows', None)
print(df)
print('各字段情况\n',df.describe())

######画图区域，查看应力数据情况
# # 应力数据分布直方图
# plt.figure(1,figsize=(20,10))
# plt.subplot(2,1,1)
# df['direction_first'].hist(bins=1000)
# plt.grid(False)
# plt.title("Distribution of direction_first", fontsize=15)
# plt.xlabel("direction_first")
# plt.xlim(160000,210000)
#
# # 应力数据分布直方图在不同时间的情况
# plt.subplot(2,1,2)
# night = (df["datetime"].apply(lambda x: 0 <= int(x[11:13]) <= 6)) | \
#         (df["datetime"].apply(lambda x: 19 <= int(x[11:13]) <= 23))
# df[night]["direction_first"].hist(bins=1000, label="Night time: 7pm - 6am",
#                                       alpha=0.5, color="darkblue")
# df[~night]["direction_first"].hist(bins=1000, label="Day time: 7am - 6pm",
#                                        alpha=0.5, color="gold")
# plt.grid(False)
# plt.legend()
# plt.title("Distribution of direction_first for different time of day", fontsize=15)
# plt.xlabel("direction_first")
# plt.xlim(160000,210000)
#
# # pd.set_option('display.max_columns', None)
# # pd.set_option('display.max_rows', None)
#
# plt.figure(2,figsize=(20,10))
# plt.subplot(2,1,1)
# df['direction_first'].plot()
# plt.grid()
#
# # 应力数据分布散点图
# plt.subplot(2,1,2)
# df['direction_first'].plot(style='.k')
# plt.grid()
#
# plt.figure(3,figsize=(20,10))
# plt.subplot(2,1,1)
# plt.boxplot(df['direction_first'])


# # 分箱法，按四分位数进行分箱 按照数量区分qcut，找到各箱子区间
# df_copy = df.copy()
# df_copy['cut_group'] = pd.qcut(df_copy.direction_first,4)
# # print(df_copy)
# # print(df_copy['cut_group'].value_counts())




zone_low=[198600.4,198498.5,342311.15,256841.6]


# 四分位法，根据箱线图的上下限进行异常值去除
def boxplot(col,index):
    # 计算iqr：数据四分之三分位值与四分之一分位值的差
    iqr = col.quantile(0.75) - col.quantile(0.25)
    # 根据iqr计算异常值判断阈值
    # 上界
    val_up = (col.quantile(0.75) + 1.5 * iqr)
    # 下界
    val_low = (col.quantile(0.25) - 1.5 * iqr)
    # 异常值
    outlier = col[(col < val_low) | (col > val_up)]
    # 正常值
    normal_val = col[(col > val_low) & (col < val_up)]
    normal_val_array=normal_val.values
    # print(normal_val.values,type(normal_val.values),len(normal_val_array))
    z_low=zone_low[index]
    rule=(z_low<normal_val_array)&(normal_val_array<999999.0)
    normal_val_propre=normal_val_array[rule]
    outlier = np.array(outlier)
    # print('异常值如下:\n' + str(outlier))
    # print('val_up的类型：{},val_up值为:{}'.format(type(val_up), val_up))
    # print('val_low的类型：{},val_low值为:{}'.format(type(val_low), val_low))
    mean=np.mean(normal_val_propre)
    print('方向{}均值：{} 类型为 {}'.format(index+1,mean, type(mean)))
    # 将原数据中异常值替换为均值
    def change(x):
        if x > val_up :
            return np.nan
        # # 海原台无负值，不用这个条件，高台需要
        # elif x<val_low:
        #     return np.nan
        else:
            return x
    return col.map(change)

df['direction_first'] = boxplot(df['direction_first'],0)
# df['direction_second'] = boxplot(df['direction_second'],1)
# df['direction_third'] = boxplot(df['direction_third'],2)
# df['direction_fourth'] = boxplot(df['direction_fourth'],3)

# df['direction_fifth'] = boxplot(df['direction_fifth'],0)
# df['direction_sixth'] = boxplot(df['direction_sixth'],1)
# df['direction_seventh'] = boxplot(df['direction_seventh'],2)
# df['direction_eighth'] = boxplot(df['direction_eighth'],3)
print('去除极端值后各字段情况\n',df.describe())
print("缺失的数据:\n",df.isnull().sum())


###四分位后画图
# plt.subplot(2,1,2)
# plt.boxplot(df['direction_first'])
#
# plt.figure(4,figsize=(20,10))
# plt.subplot(3,1,1)
# df['direction_first'].hist()
# plt.xlim(160000,210000)
# plt.subplot(3,1,2)
# df['direction_first'].plot(style='.k')
# plt.grid()
# plt.subplot(3,1,3)
# df['direction_first'].plot()
# plt.grid()
# plt.show()




# 滚动平均，去波动较大噪声
def smooth_part(direction,index):
    df[direction].interpolate(inplace=True)
    print('当前方向{}缺失值数量'.format(direction),df[direction].isnull().sum())
    yvals = np.array(df[direction])
    max_x,_ = find_peaks(yvals,prominence=10000)
    max_y = yvals[max_x]
    # print('突变异常峰值点max_x max_y', max_x, len(max_x), max_y, len(max_y))

    def smooth(x):
        center_x = 0
        df_copy = x[direction].copy()
        for i in range(len(max_x)):
            center_x = max_x[i]
            x_copy = x[direction].iloc[center_x - 2 * 1440:center_x + 1440].copy()
            rol = x_copy.rolling(1440).mean()
            df_copy.iloc[center_x - 1440:center_x + 1440] = rol[1440:4320]
        return df_copy
    # 平滑前后比较
    plt.figure(index+1)
    df[direction].plot()
    df[direction] = smooth(df)
    df[direction].plot()
    plt.legend(['smooth before', 'smooth after'])

    plt.plot(max_x, max_y, 'o', markersize=3)  # 极大值点

smooth_part('direction_first',0)
# smooth_part('direction_second',1)
# smooth_part('direction_third',2)
# smooth_part('direction_fourth',3)

# smooth_part('direction_fifth',0)
# smooth_part('direction_sixth',1)
# smooth_part('direction_seventh',2)
# smooth_part('direction_eighth',3)
# plt.show()
# # df.to_csv('./Data/haiyuan_final.csv', encoding='utf_8_sig',index=False)
# # df.to_csv('./Data/linxia_final.csv', encoding='utf_8_sig',index=False)
#
#
#
# def plot_year_month(direction,index):
#     # # 年情况
#     # for y in range(8):
#     #     year = '201' + str(y)
#     #     rule = df['datetime'].str.contains(year)
#     #     data_year = df[rule]
#     #     plt.figure(1+index, figsize=(20, 10))
#     #     plt.subplot(2, 1, 1)
#     #     data_year[direction].hist()
#     #     plt.subplot(2, 1, 2)
#     #     data_year[direction].plot()
#     #     plt.grid()
#
#     day_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
#                 '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
#                 '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
#     mean_list = []
#
#
#     time_list = []
#     for i in day_list:
#         for j in time1:
#             temp = i + '/' + j
#             time_list.append(temp)
#
#
#
#     # # 某年某月情况
#     # s1 = '2016-07'
#     # rule = df['datetime'].str.contains(s1)
#     # data_month = df[rule]
#     # plt.figure(5+index, figsize=(20, 10), constrained_layout=True)
#     # l = len(data_month[direction])
#     # plt.plot(time_list[0:l], data_month[direction])
#     # plt.xticks(range(0, len(time_list), 1440 * 2), rotation=30, fontsize=8)
#     # plt.grid()
#
#
#
#
#     # 某年某月末日情况
#     s1 = '2016-01-21'
#     rule = df['datetime'].str.contains(s1)
#     data_month = df[rule]
#     plt.figure(6+index, figsize=(20, 10), constrained_layout=True)
#     l = len(data_month[direction])
#     plt.plot(time1[0:l], data_month[direction])
#     plt.xticks(range(0, len(time1), 20), rotation=50, fontsize=6)
#     plt.grid()
#
#
#
#
#     # 某年某月末日某时情况
#     s11 = '2016-01-21-00:48'
#     s12='2016-01-21-01:48'
#     ind1 = df[(df['datetime'] ==s11)].index.tolist()
#     ind2= df[(df['datetime'] ==s12)].index.tolist()
#     l1=int(ind1[0])
#     l2=int(ind2[0])
#     plt.figure(7+index, figsize=(20, 10), constrained_layout=True)
#     l = l2-l1
#     # print(l1,l2)
#     t1=l1%1440
#     t2=l2%1440
#     # print(time1[t1],time1[t2])
#     # print(data_month[direction].iloc[t1:t2])
#     plt.plot(time1[t1:t2], data_month[direction].iloc[t1:t2])
#     plt.xticks(time1[t1:t2:2], rotation=50, fontsize=6)
#     plt.grid()
#
#     # 数据增强方法：
#     d=data_month[direction].iloc[t1:t2]
#     d=np.array(d)
#     d=d.reshape(1,-1,1)
#     steps = np.arange(d.shape[1])
#     plt.figure(100,figsize=(20,10))
#     plt.plot(steps,d[0])
#     plt.plot(steps,aug.jitter(d)[0])
#     plt.legend(['before', 'after'])
#
#     plt.figure(101, figsize=(20, 10))
#     plt.plot(steps, d[0])
#     plt.plot(steps, aug.scaling(d)[0])
#     plt.legend(['before', 'after'])
#
#     plt.figure(102, figsize=(20, 10))
#     plt.plot(steps, d[0])
#     plt.plot(steps, aug.permutation(d)[0])
#     plt.legend(['before', 'after'])
#
#     plt.figure(103, figsize=(20, 10))
#     plt.plot(steps, d[0])
#     plt.plot(steps, aug.magnitude_warp(d)[0])
#     plt.legend(['before', 'after'])
#     d_magnitude=aug.magnitude_warp(d)
#
#     plt.figure(104, figsize=(20, 10))
#     plt.plot(steps, d[0])
#     plt.plot(steps, aug.time_warp(d)[0])
#     plt.legend(['before', 'after'])
#     magnitude_timewarp=aug.time_warp(d_magnitude)
#
#     plt.figure(105, figsize=(20, 10))
#     plt.plot(steps, d[0])
#     plt.plot(steps, aug.rotation(d)[0])
#     plt.legend(['before', 'after'])
#     d_rotation=aug.rotation(d)
#     rotation_permuta=aug.permutation(d_rotation)
#     d_permute=aug.permutation(d)
#     permute_rot=aug.rotation(d_permute)
#
#     plt.figure(106, figsize=(20, 10))
#     plt.plot(steps, d[0])
#     plt.plot(steps, aug.window_slice(d)[0])
#     plt.legend(['before', 'after'])
#
#     plt.figure(107, figsize=(20, 10))
#     plt.plot(steps, d[0])
#     plt.plot(steps, aug.window_warp(d)[0])
#     plt.legend(['before','after'])
#
#     plt.figure(108,figsize=(20,10))
#     plt.plot(steps,d[0])
#     plt.plot(steps,rotation_permuta[0])
#     plt.legend(['before', 'rotation+permutation'])
#
#     plt.figure(109, figsize=(20, 10))
#     plt.plot(steps, d[0])
#     plt.plot(steps, magnitude_timewarp[0])
#     plt.legend(['before', 'magnitude+timewarp'])
#
#     plt.figure(110, figsize=(20, 10))
#     plt.plot(steps, d[0])
#     plt.plot(steps, permute_rot[0])
#     plt.legend(['before', 'rotation+permute'])
#     # # 每年各月情况
#     # for y in range(8):
#     #     s1 = '201' + str(y)
#     #     rule = df['datetime'].str.contains(s1)
#     #     data_year = df[rule]
#     #     for m in range(12):
#     #         month = str(m + 1)
#     #         if m >= 0 and m <= 8:
#     #             month = '0' + month
#     #         s = s1 + '-' + month
#     #         rule = data_year['datetime'].str.contains(s)
#     #         plt.figure(2 + y+index, figsize=(20, 10), constrained_layout=True)
#     #         plt.subplot(3, 4, m + 1)
#     #         data_month = data_year[rule]
#     #         l = len(data_month[direction])
#     #         plt.plot(time_list[0:l], data_month[direction])
#     #         plt.xticks(range(0, len(time_list), 1440 * 2), rotation=30, fontsize=8)
#     #         plt.grid()
#
#     #         month_mean = data_month[direction].mean()
#     #         mean_list.append(month_mean)
#     #
#     # month_list = ['2010-1', '2010-2', '2010-3', '2010-4', '2010-5', '2010-6', '2010-7', '2010-8', '2010-9', '2010-10',
#     #               '2010-11', '2010-12',
#     #               '2011-1', '2011-2', '2011-3', '2011-4', '2011-5', '2011-6', '2011-7', '2011-8', '2011-9', '2011-10',
#     #               '2011-11', '2011-12',
#     #               '2012-1', '2012-2', '2012-3', '2012-4', '2012-5', '2012-6', '2012-7', '2012-8', '2012-9', '2012-10',
#     #               '2012-11', '2012-12',
#     #               '2013-1', '2013-2', '2013-3', '2013-4', '2013-5', '2013-6', '2013-7', '2013-8', '2013-9', '2013-10',
#     #               '2013-11', '2013-12',
#     #               '2014-1', '2014-2', '2014-3', '2014-4', '2014-5', '2014-6', '2014-7', '2014-8', '2014-9', '2014-10',
#     #               '2014-11', '2014-12',
#     #               '2015-1', '2015-2', '2015-3', '2015-4', '2015-5', '2015-6', '2015-7', '2015-8', '2015-9', '2015-10',
#     #               '2015-11', '2015-12',
#     #               '2016-1', '2016-2', '2016-3', '2016-4', '2016-5', '2016-6', '2016-7', '2016-8', '2016-9', '2016-10',
#     #               '2016-11', '2016-12',
#     #               '2017-1', '2017-2', '2017-3', '2017-4', '2017-5', '2017-6', '2017-7', '2017-8', '2017-9', '2017-10',
#     #               '2017-11', '2017-12']
#     #
#     # plt.figure(10+index, figsize=(20, 10))
#     # plt.plot(month_list, mean_list, 'red')
#     # plt.xlabel('date')
#     # plt.ylabel('month_mean')
#     # plt.xticks(range(0, len(month_list), 4), rotation=45)
#
#
# # index为0 10 20 30
# # plot_year_month('direction_first',0)
# # plot_year_month('direction_second',10)
# # plot_year_month('direction_third',20)
# # plot_year_month('direction_fourth',30)
# # plt.show()
# # 根据地震时间找波动
# df_area = pd.read_csv('./Data/Area1_3~10.csv',encoding = 'gb2312')
# df_earth= df_area['发震日期（北京时间）']
# df_earth1 = pd.to_datetime(df_earth)
# df_time = df_earth1.dt.strftime('%Y-%m-%d-%H:%M')
# df_time_np=np.array(df_time)
# earth_x=np.array([])
# direction_list=['direction_first','direction_second','direction_third','direction_fourth']
# df_loc= df_area[['纬度(°)','经度(°)']]
#
#
# def find_earthquake():
#     for i in range(len(df_time_np)):
#         s = str(df_time_np[i])
#         ind = df[df.datetime == s].index.tolist()[0]
#         y = df['direction_first'].iloc[ind - 60:ind + 60].copy()
#         yvals = np.array(y)
#         max_x, _ = find_peaks(yvals, prominence=100)
#         min_x,_=find_peaks(-yvals, prominence=100)
#         if (len(max_x) or len(min_x)):
#             d = 0
#             while (d < 4):
#                 direction = direction_list[d]
#                 y_v = np.array(df[direction].iloc[ind - 60:ind + 60])
#                 m_x, _ = find_peaks(y_v, prominence=100)
#                 m_y = y_v[m_x]
#                 mi_x, _ = find_peaks(-y_v, prominence=100)
#                 mi_y=y_v[mi_x]
#
#                 print('方向{}查询的时间是:'.format(direction), s, '时间索引为', i)
#                 print('方向{}数据中该点索引为'.format(direction), ind, df[direction].iloc[ind])
#                 print('方向{}查询到该范围最大值为'.format(direction), m_x, len(m_x), m_y, len(m_y))
#                 print('方向{}查询到该范围最小值为'.format(direction), mi_x, len(mi_x), mi_y, len(mi_y))
#                 print()
#                 plt.figure(10 + d, figsize=(20, 10))
#                 t_center = s[11:]
#                 t_ind = time1.index(t_center)
#                 plt.plot(time1[t_ind - 60:t_ind + 60], df[direction].iloc[ind - 60:ind + 60])
#                 plt.xticks(range(0, 120, 2), rotation=30, fontsize=8)
#                 plt.grid()
#                 for j in range(len(m_x)):
#                     plt.plot(time1[t_ind - 60 + m_x[j]], m_y[j], 'o')
#                 for k in range(len(mi_x)):
#                     plt.plot(time1[t_ind - 60 + mi_x[k]], mi_y[k], 'o')
#                 d += 1
#             # plt.show()
#
# # find_earthquake()
#
#
#
# def check_earthquake():
#     for i in range(29,len(df_time_np)):
#         s = str(df_time_np[i])
#         ind = df[df.datetime == s].index.tolist()[0]
#         d=0
#         while (d < 4):
#             direction = direction_list[d]
#             print('方向{}查询的时间是:'.format(direction), s, '时间索引为', i)
#             print('方向{}数据中该点索引为'.format(direction), ind, df[direction].iloc[ind])
#             print()
#             plt.figure(10 + d, figsize=(20, 10))
#             t_center = s[11:]
#             t_ind = time1.index(t_center)
#
#             if t_ind-30<0 or t_ind+30>1439:
#                 x_list=[]
#                 if t_ind-30<0:
#                     x_list.extend(time1[1439+t_ind-30:1439])
#                     x_list.extend(time1[0:t_ind+30])
#                     plt.plot(x_list,df[direction].iloc[ind - 30:ind + 30])
#                 elif t_ind+30>1439:
#                     x_list.extend(time1[t_ind-30:1439])
#                     x_list.extend(time1[0:t_ind+30-1439])
#                     plt.plot(x_list, df[direction].iloc[ind - 30:ind + 30])
#             else:
#                 plt.plot(time1[t_ind - 30:t_ind + 30], df[direction].iloc[ind - 30:ind + 30])
#             plt.xticks(range(0, 60), rotation=30, fontsize=8)
#             plt.grid()
#             d += 1
#         plt.show()
# # check_earthquake()
# # plt.show()
#
#
#
#
#
#
#
# # longtitude 经度 latitude 维度
# def get_distance(data):
#
#     LaA=data['纬度(°)']
#     LoA=data['经度(°)']
#
#     # 海原 银川 临夏 高台
#     centers = np.array([[105.61, 36.51], [105.93, 38.61], [103.2, 35.6], [99.86, 39.4]])
#     LoA = radians(LoA)
#     LaA = radians(LaA)
#     LoB=radians(centers[2][0])
#     LaB=radians(centers[2][1])
#
#
#     D_Lo = LoB - LoA
#     D_La = LaB - LaA
#     P = sin(D_La / 2) ** 2 + cos(LaA) * cos(LaB) * sin(D_Lo / 2) ** 2
#
#     Q = 2 * asin(sqrt(P))
#     R_km = 6371
#     dis=Q*R_km
#
#     return dis
#
#
#
# def get_degree(data):
#
#     LaB=data['纬度(°)']
#     LoB=data['经度(°)']
#
#     # 海原 银川 临夏 高台
#     centers = np.array([[105.61, 36.51], [105.93, 38.61], [103.2, 35.6], [99.86, 39.4]])
#     LoB = radians(LoB)
#     LaB = radians(LaB)
#     LoA=radians(centers[2][0])
#     LaA=radians(centers[2][1])
#
#     dLon = LoB - LoA
#     y = sin(dLon) * cos(LaB)
#     x = cos(LaA) * sin(LaB) - sin(LaA) * cos(LaB) * cos(dLon)
#     brng = math.degrees(math.atan2(y, x))
#     brng = (brng + 360) % 360
#
#     # if (brng == 0.0) or ((brng == 360.0)):
#     #     print('正北方向')
#     # elif brng == 90.0:
#     #     print('正东方向')
#     # elif brng == 180.0:
#     #     print('正南方向')
#     # elif brng == 270.0:
#     #     print('正西方向')
#     # elif 0 < brng < 90:
#     #     print(f'北偏东{brng}')
#     # elif 90 < brng < 180:
#     #     print(f'东偏南{brng - 90}')
#     # elif 180 < brng < 270:
#     #     print(f'西偏南{270 - brng}')
#     # elif 270 < brng < 360:
#     #     print(f'北偏西{360-brng }')
#     return brng
# # 给地震目录添加相对于台站的方向特征 位置特征
# dis=df_loc.apply(get_distance,axis=1)
# dgr=df_loc.apply(get_degree,axis=1)
# print('距离和方向\n',dis,len(dis),dgr,len(dgr))
# # 地震目录加入距离方向特征
# df_area.insert(df_area.shape[1], 'distance', dis)
# df_area.insert(df_area.shape[1], 'degree', dgr)
# # df_area.to_csv('./Data/Area1.2_final.csv', encoding='gb2312',index=False)
# print('新的地震目录形式\n',df_area)
# df.insert(df.shape[1], 'distance', 0)
# df.insert(df.shape[1], 'degree', 0)
#
# # 将距离角度特征加入数据
# def add_dis_dgr(df_area):
#     dis = np.array(df_area['distance'])
#     dgr = np.array(df_area['degree'])
#     df_earth1 = pd.to_datetime(df_area['发震日期（北京时间）'])
#     time = df_earth1.strftime('%Y-%m-%d-%H:%M')
#     ind = df[df.datetime == time].index.tolist()[0]
#     df['distance'].iloc[ind - 10:ind + 30] = dis
#     df['degree'].iloc[ind - 10:ind + 30] = dgr
#
#
# df_area.apply(add_dis_dgr, axis=1)
# print('新的数据形式\n',df,df.shape)
# df.to_csv('./Data/linxia_new2.csv', encoding='utf_8_sig',index=False)

