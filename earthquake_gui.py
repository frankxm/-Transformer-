import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings("ignore")
from tkinter import *
import tkinter as tk
from tkinter import messagebox,filedialog,ttk
from datetime import datetime
import webbrowser
import time
import queue, threading
import torch
#folium-聚合散点地图
import folium
from folium import plugins
import os
# 在py中临时添加的环境变量,防止用kmeans警告
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.cluster import KMeans
import pandas as pd
from scipy.spatial import ConvexHull
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score,confusion_matrix
import torch
import numpy as np
import random
from scipy.signal import find_peaks
import math
from math import radians, cos, sin, asin, sqrt
import seaborn as sns
import matplotlib.pyplot as plt
def setup_seed(seed):
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)  # gpu
    # 每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class MyDataset(Dataset):
    def __init__(self,df,df_area):

        super(MyDataset, self).__init__()
        self.test_len, \
        self.input_len, \
        self.channel_len, \
        self.output_len, \
        self.test_dataset, \
        self.test_label,\
        self.time_list= self.make_data(df,df_area)

    def __getitem__(self, index):
        return self.test_dataset[index], self.test_label[index] - 1

    def __len__(self):
        return self.test_len

    def make_data(self, df,df_area):
        setup_seed(3047)
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
        # print('原始标签',labels)
        # 找到地震目录在数据中的索引
        rule=df['datetime'].isin(df_time_np)
        # 从时间远到近
        ind_list=df[rule].index.tolist()
        sample_len=len(ind_list)
        # 直接取
        sample_num = int(1 * sample_len)
        sample_list = [i for i in range(sample_len)]
        test_list = random.sample(sample_list, sample_num)
        self.test_list=test_list
        # print('随机取后顺序',test_list)
        test_list_reverse=[sample_num-1-l for l in test_list]
        ind_list = np.array(ind_list)
        test_ind = ind_list[test_list_reverse]


        df_copy=df.copy()
        df_cur = df.drop('datetime', axis=1)
        test_time_list = []
        def get_data(x):
            test_time_list.append(df_copy['datetime'].iloc[x])
            df_choosed = df_cur.iloc[x - 10:x + 30]
            df_choosed = np.array(df_choosed)
            # z-score标准化
            mean_choosed = df_choosed.mean()
            std_choosed = df_choosed.std()
            df_choosed = (df_choosed - mean_choosed) / std_choosed
            tensor_choosed = torch.from_numpy(df_choosed).float()
            return tensor_choosed

        data_cur = list(map(lambda x: get_data(x), test_ind))
        test_data = torch.stack(data_cur, dim=0)
        # print(f'使用的数据所在时间为{test_time_list}\n长度为{len(test_time_list)}')

        labels = np.array(labels)
        labels_test = labels[test_list]
        labels_test += 1
        labels_test = torch.from_numpy(labels_test)
        # print('数据预测前索引排序',test_ind)
        # print('对应的标签排序',labels_test)
        unique, count = np.unique(labels_test, return_counts=True)
        data_count = dict(zip(unique, count))
        # print('标签各类别出现个数:', data_count)

        test_len=len(test_data)
        input_len=40
        channel=6
        output_len=9

        return  test_len, input_len, channel, output_len,   test_data, labels_test,test_time_list



class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        # 设定宽高保持出现在屏幕中间
        self.width = 1320
        self.height = 600
        self.screenwidth = self.root.winfo_screenwidth()
        self.screenheight = self.root.winfo_screenheight()
        # 出现在正中间
        self.root.geometry('%dx%d+%d+%d' % (self.width, self.height, (self.screenwidth - self.width) / 2, (self.screenheight - self.height) / 2))
        # 屏幕不可调整大小
        self.root.resizable(False, False)
        self.root.iconbitmap("./Data/haida.ico")
        self.root.title('地震区域预测工具')

        # 定义台站数量 地震数量变量 导入时间用于随时改变
        self.station_num = tk.StringVar()
        self.earthquake_num = tk.StringVar()
        self.station_import = tk.StringVar()
        self.earthquake_import = tk.StringVar()

        # 定义地震目录相关变量
        self.lat = tk.StringVar()
        self.long = tk.StringVar()
        self.depth = tk.StringVar()
        self.level = tk.StringVar()
        self.country = tk.StringVar()
        self.tim = tk.StringVar()

        # 整体结构为上下框
        self.mainFrame = ttk.LabelFrame(self.root)
        self.mainFrame.grid(row=0, column=0)
        # 信息框下方预测按钮
        self.predictButton = ttk.Button(self.root, text='开始预测', command=lambda: self.predict_loc())
        self.predictButton.grid(column=0, row=1, ipadx=10, ipady=5)


        # 上方框
        self.headings_frame = ttk.LabelFrame(self.mainFrame)
        self.headings_frame.grid(row=0)

        self.selection_frame = ttk.LabelFrame(self.headings_frame, text="台站文件信息")
        self.selection_frame.grid(column=0, row=0, rowspan=2, sticky="NW")
        self.file_frame = ttk.LabelFrame(self.headings_frame, text="地震目录文件信息")
        self.file_frame.grid(column=1, row=0, rowspan=2, sticky="NW")

        # 台站文件框
        ttk.Label(self.selection_frame, text="最新台站数据导入时间").grid(column=0, row=0, sticky="W")
        self.fileTimeEntry = ttk.Label(self.selection_frame, width=50, textvariable=self.station_import, state="readonly")
        self.fileTimeEntry.grid(column=1, row=0, sticky="W")
        ttk.Label(self.selection_frame, text="台站数量").grid(column=0, row=1, sticky="W")
        self.fileCountEntry = ttk.Label(self.selection_frame, width=50, textvariable=self.station_num, state="readonly")
        self.fileCountEntry.grid(column=1, row=1, sticky="W")
        # 地震目录文件框
        ttk.Label(self.file_frame, text="地震目录导入时间").grid(column=0, row=0, sticky="W")
        self.fileTimeEntry = ttk.Label(self.file_frame, width=50, textvariable=self.earthquake_import, state="readonly")
        self.fileTimeEntry.grid(column=1, row=0, sticky="W")
        ttk.Label(self.file_frame, text="地震目录条数").grid(column=0, row=1, sticky="W")
        self.fileCountEntry = ttk.Label(self.file_frame, width=50, textvariable=self.earthquake_num, state="readonly")
        self.fileCountEntry.grid(column=1, row=1, sticky="W")

        # 下方框
        self.details_frame = ttk.LabelFrame(self.mainFrame)
        self.details_frame.grid(row=1)
        self.summary_frame = ttk.LabelFrame(self.details_frame, text="地震目录")
        self.summary_frame.grid(row=0, column=0, sticky="W")
        self.station_frame = ttk.LabelFrame(self.details_frame, text='台站信息')
        self.station_frame.grid(row=1, column=0, sticky="W")

        # 地震目录信息
        # 纬度
        ttk.Label(self.summary_frame, text="纬度范围:").grid(column=0, row=0, sticky="E")
        self.latEntry = ttk.Label(self.summary_frame, width=20, textvariable=self.lat)
        self.latEntry.grid(column=1, row=0, sticky="W")
        # 经度
        ttk.Label(self.summary_frame, text="经度范围:").grid(column=2, row=0, sticky="E")
        self.longEntry = ttk.Label(self.summary_frame, width=20, textvariable=self.long)
        self.longEntry.grid(column=3, row=0, sticky="W")
        # 日期
        ttk.Label(self.summary_frame, text="日期范围:").grid(column=4, row=0, sticky="E")
        self.timeEntry = ttk.Label(self.summary_frame, width=20, textvariable=self.tim)
        self.timeEntry.grid(column=5, row=0, sticky="W")
        # 震级
        ttk.Label(self.summary_frame, text="震级范围:").grid(column=6, row=0, sticky="E")
        self.shakeEntry = ttk.Label(self.summary_frame, width=10, textvariable=self.level)
        self.shakeEntry.grid(column=7, row=0, sticky="W")
        # 深度
        ttk.Label(self.summary_frame, text="深度范围:").grid(column=8, row=0, sticky="E")
        self.deepEntry = ttk.Label(self.summary_frame, width=10, textvariable=self.depth)
        self.deepEntry.grid(column=9, row=0, sticky="W")
        # 全球/国内
        ttk.Label(self.summary_frame, text="地点:").grid(column=10, row=0, sticky="E")
        self.locationEntry = ttk.Label(self.summary_frame, width=4, textvariable=self.country)
        self.locationEntry.grid(column=11, row=0, sticky="W")
        self.url_web = 'https://data.earthquake.cn/datashare/report.shtml?PAGEID=earthquake_zhengshi'
        # 更多信息
        ttk.Label(self.summary_frame, text="地震更多信息：").grid(column=0, row=1, sticky="W")
        self.moreEntry = Button(self.summary_frame, text=self.url_web, command=lambda: self.getweb(self.url_web))
        self.moreEntry.grid(column=1, row=1, columnspan=8, sticky="W")

        self.loc_station1 = tk.StringVar()
        self.loc_station2 = tk.StringVar()
        self.loc_station3 = tk.StringVar()
        self.loc_station4 = tk.StringVar()

        self.lat_num1 = tk.StringVar()
        self.lat_num2 = tk.StringVar()
        self.lat_num3 = tk.StringVar()
        self.lat_num4 = tk.StringVar()

        self.long_num1 = tk.StringVar()
        self.long_num2 = tk.StringVar()
        self.long_num3 = tk.StringVar()
        self.long_num4 = tk.StringVar()

        self.data_len1 = tk.StringVar()
        self.data_len2 = tk.StringVar()
        self.data_len3 = tk.StringVar()
        self.data_len4 = tk.StringVar()
        # 定义台站详细信息变量
        self.loc_station_list = [self.loc_station1, self.loc_station2, self.loc_station3, self.loc_station4]
        self.lat_num_list =     [self.lat_num1, self.lat_num2,  self.lat_num3,  self.lat_num4]
        self.long_num_list =    [self.long_num1,self.long_num2, self.long_num3, self.long_num4]
        self.data_len_list =    [self.data_len1,self.data_len2, self.data_len3, self.data_len4]
        # 台站1
        self.detection_frame1 = ttk.LabelFrame(self.station_frame, text=f"台站{1}")
        self.detection_frame1.grid(row=1, column=0, sticky="W")
        ttk.Label(self.detection_frame1, text='台站名称:').grid(column=0, row=4, sticky="E")
        self.locEntry1 = ttk.Label(self.detection_frame1, width=25, textvariable=self.loc_station1, state="readonly")
        self.locEntry1.grid(column=1, row=4, sticky="W")
        ttk.Label(self.detection_frame1, text='纬度:').grid(column=0, row=5, sticky="E")
        self.latEntry1 = ttk.Label(self.detection_frame1, width=25, textvariable=self.lat_num1, state="readonly")
        self.latEntry1.grid(column=1, row=5, sticky="W")
        ttk.Label(self.detection_frame1, text='经度:').grid(column=0, row=6, sticky="E")
        self.longEntry1 = ttk.Label(self.detection_frame1, width=25, textvariable=self.long_num1, state="readonly")
        self.longEntry1.grid(column=1, row=6, sticky="W")
        ttk.Label(self.detection_frame1, text='数据量:').grid(column=0, row=7, sticky="E")
        self.numEntry1 = ttk.Label(self.detection_frame1, width=35, textvariable=self.data_len1, state="readonly")
        self.numEntry1.grid(column=1, row=7, sticky="W")

        # 台站2
        self.detection_frame2 = ttk.LabelFrame(self.station_frame, text=f"台站{2}")
        self.detection_frame2.grid(row=1, column=1, sticky="W")
        ttk.Label(self.detection_frame2, text='台站名称:').grid(column=0, row=4, sticky="E")
        self.locEntry2 = ttk.Label(self.detection_frame2, width=25, textvariable=self.loc_station2, state="readonly")
        self.locEntry2.grid(column=1, row=4, sticky="W")
        ttk.Label(self.detection_frame2, text='纬度:').grid(column=0, row=5, sticky="E")
        self.latEntry2 = ttk.Label(self.detection_frame2, width=25, textvariable=self.lat_num2, state="readonly")
        self.latEntry2.grid(column=1, row=5, sticky="W")
        ttk.Label(self.detection_frame2, text='经度:').grid(column=0, row=6, sticky="E")
        self.longEntry2 = ttk.Label(self.detection_frame2, width=25, textvariable=self.long_num2, state="readonly")
        self.longEntry2.grid(column=1, row=6, sticky="W")
        ttk.Label(self.detection_frame2, text='数据量:').grid(column=0, row=7, sticky="E")
        self.numEntry2 = ttk.Label(self.detection_frame2, width=35, textvariable=self.data_len2, state="readonly")
        self.numEntry2.grid(column=1, row=7, sticky="W")

        # 台站3
        self.detection_frame3 = ttk.LabelFrame(self.station_frame, text=f"台站{3}")
        self.detection_frame3.grid(row=1, column=2, sticky="W")
        ttk.Label(self.detection_frame3, text='台站名称:').grid(column=0, row=4, sticky="E")
        self.locEntry3 = ttk.Label(self.detection_frame3, width=25, textvariable=self.loc_station3, state="readonly")
        self.locEntry3.grid(column=1, row=4, sticky="W")
        ttk.Label(self.detection_frame3, text='纬度:').grid(column=0, row=5, sticky="E")
        self.latEntry3 = ttk.Label(self.detection_frame3, width=25, textvariable=self.lat_num3, state="readonly")
        self.latEntry3.grid(column=1, row=5, sticky="W")
        ttk.Label(self.detection_frame3, text='经度:').grid(column=0, row=6, sticky="E")
        self.longEntry3 = ttk.Label(self.detection_frame3, width=25, textvariable=self.long_num3, state="readonly")
        self.longEntry3.grid(column=1, row=6, sticky="W")
        ttk.Label(self.detection_frame3, text='数据量:').grid(column=0, row=7, sticky="E")
        self.numEntry3 = ttk.Label(self.detection_frame3, width=35, textvariable=self.data_len3, state="readonly")
        self.numEntry3.grid(column=1, row=7, sticky="W")

        # 台站4
        self.detection_frame4 = ttk.LabelFrame(self.station_frame, text=f"台站{4}")
        self.detection_frame4.grid(row=1, column=3, sticky="W")
        ttk.Label(self.detection_frame4, text='台站名称:').grid(column=0, row=4, sticky="E")
        self.locEntry4 = ttk.Label(self.detection_frame4, width=25, textvariable=self.loc_station4, state="readonly")
        self.locEntry4.grid(column=1, row=4, sticky="W")
        ttk.Label(self.detection_frame4, text='纬度:').grid(column=0, row=5, sticky="E")
        self.latEntry4 = ttk.Label(self.detection_frame4, width=25, textvariable=self.lat_num4, state="readonly")
        self.latEntry4.grid(column=1, row=5, sticky="W")
        ttk.Label(self.detection_frame4, text='经度:').grid(column=0, row=6, sticky="E")
        self.longEntry4 = ttk.Label(self.detection_frame4, width=25, textvariable=self.long_num4, state="readonly")
        self.longEntry4.grid(column=1, row=6, sticky="W")
        ttk.Label(self.detection_frame4, text='数据量:').grid(column=0, row=7, sticky="E")
        self.numEntry4 = ttk.Label(self.detection_frame4, width=35, textvariable=self.data_len4, state="readonly")
        self.numEntry4.grid(column=1, row=7, sticky="W")

        # 删除台站信息按钮
        self.delStation = Button(self.station_frame, text='删除', command=lambda: self.delete())
        self.delStation.grid(column=3, row=2, rowspan=5, columnspan=5, sticky="E")

        self.mainFrame.grid_configure(padx=8, pady=4)
        for child in self.mainFrame.winfo_children():
            child.grid_configure(padx=8, pady=4)
            for grandChild in child.winfo_children():
                grandChild.grid_configure(padx=5, pady=4)
                for widget in grandChild.winfo_children():
                    widget.grid_configure(padx=4, pady=4)



        self.map_url='file:///C:/python_pycharm/Machine_learning/folium_map2.html'
        # 创建菜单栏并绑定相应事件函数
        self.main_menu = Menu(self.root)
        self.main_menu.add_command(label="导入文件", command=lambda: self.upload_file())
        self.main_menu.add_command(label="导入文本", command=lambda: self.upload_txt())
        self.main_menu.add_command(label="导入模型", command=lambda: self.load_model())
        self.main_menu.add_command(label="查看结果", command=lambda: self.get_result2(self.map_url))
        self.main_menu.add_command(label="查看用法", command=lambda: self.watch_usage())
        self.root.config(menu=self.main_menu)
        self.main_menu.add_separator()

        # 海原 银川 临夏 高台
        self.centers = [[36.51, 105.61], [38.61, 105.93], [35.6, 103.2], [39.4, 99.86]]
        self.start_ind = 0

        # 进度条
        self.progress = tk.IntVar()
        self.progress_max = 100
        self.progressbar = ttk.Progressbar(self.root, mode='determinate', orient=tk.HORIZONTAL, variable=self.progress,maximum=self.progress_max)
        self.progressbar.grid(column=0,row=2,ipadx=10,ipady=5,pady=5)
        self.progress.set(0)
        self.progressbar.grid_forget()

        self.complete_num=0
        self.df_list=[]

        self.root.mainloop()
    def start(self):
        self.thread_queue = queue.Queue()
        self.new_thread = threading.Thread(target=self.listen_for_result)
        self.new_thread.start()


    def listen_for_result(self):
        try:
            cur_path=self.current_path
            cur_index=self.data_len_index
            self.get_data(cur_path,cur_index)
        except queue.Empty:
            pass
        finally:
            if self.station_list_len==self.complete_num:
                self.progressbar.stop()
                self.progressbar.grid_forget()
                self.moreEntry.config(state=tk.NORMAL)
                self.delStation.config(state=tk.NORMAL)
                self.predictButton.config(state=tk.NORMAL)
                tk.messagebox.showinfo("完成提示", "导入台站完成")

    def load_model(self):
        model_path = filedialog.askopenfilename()
        # self.DEVICE = torch.device("cpu")
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model=torch.load(model_path, map_location=self.DEVICE)
        self.BATCH_SIZE = int(model_path[model_path.find('=')+1:model_path.rfind('.')])  # 使用的模型的batch_size
        tk.messagebox.showinfo("完成提示", "导入模型完成")

    def watch_usage(self):
        tk.messagebox.showinfo("使用", "允许上传最多4个台站的应力文件、1个地震目录文件进行地震方位预测。其中文件可上传csv和txt格式，若上传txt格式自动进行处理。")

    def get_result2(self,url):
        webbrowser.open_new(url)



    def get_data(self,cur_path,cur_index):
        print('当前索引',cur_index)
        print('当前路径',cur_path)
        df = pd.read_csv(cur_path, sep=',', engine='python', iterator=True)
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
        print('开始合并')
        df = pd.concat(chunks, ignore_index=True)
        self.df_list.append(df)
        print('最后索引',self.data_len_index)
        self.data_len_list[cur_index].set('共有{:.0f}天{}个数据'.format(df.shape[0] / 1440, df.shape[0]))
        self.complete_num+=1



    def delete(self):
        station_num = self.station_num
        if station_num.get() == '' or int(station_num.get()) == 0:
            messagebox.showwarning("删除警告", "目前没有台站文件可删除")
        else:
            cur_num = int(station_num.get())
            cur_num -= 1
            station_num.set(cur_num)
            # args_detect_list = [loc_station_list, lat_num_list, long_num_list, data_len_list]
            name_station = self.loc_station_list
            lat_num = self.lat_num_list
            long_num = self.long_num_list
            data_len = self.data_len_list

            self.start_ind -= 1
            print('当前要删除的索引位', self.start_ind)
            cur_ind = self.start_ind
            name_station[cur_ind].set('')
            lat_num[cur_ind].set('')
            long_num[cur_ind].set('')
            data_len[cur_ind].set('')
            tk.messagebox.showinfo("完成提示", "删除完成")
    def process(self):
        df=self.df
        try:
            def boxplot(col):
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

                # 将原数据中异常值替换为均值
                def change(x):
                    if x > val_up:
                        return np.nan
                    # 海原台无负值，不用这个条件，高台需要
                    elif x < val_low:
                        return np.nan
                    else:
                        return x

                return col.map(change)

            df['direction_first'] = boxplot(df['direction_first'])
            df['direction_second'] = boxplot(df['direction_second'])
            df['direction_third'] = boxplot(df['direction_third'])
            df['direction_fourth'] = boxplot(df['direction_fourth'])
            print('去除极端值后各字段情况\n', df.describe())
            print("缺失的数据:\n", df.isnull().sum())

            # 滚动平均，去波动较大噪声
            def smooth_part(direction, index):
                df[direction].interpolate(inplace=True)
                print('当前方向{}缺失值数量'.format(direction), df[direction].isnull().sum())
                yvals = np.array(df[direction])
                max_x, _ = find_peaks(yvals, prominence=10000)
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

            smooth_part('direction_first', 0)
            smooth_part('direction_second', 1)
            smooth_part('direction_third', 2)
            smooth_part('direction_fourth', 3)
            try:
                # 根据地震时间找波动
                df_area = self.df_area
                df_earth = df_area['发震日期（北京时间）']
                df_earth1 = pd.to_datetime(df_earth)
                df_time = df_earth1.dt.strftime('%Y-%m-%d-%H:%M')
                df_loc = df_area[['纬度(°)', '经度(°)']]
                center = self.centre_txt

                # longtitude 经度 latitude 维度
                def get_distance(data):
                    LaA = data['纬度(°)']
                    LoA = data['经度(°)']
                    LoA = radians(LoA)
                    LaA = radians(LaA)
                    LoB = radians(center[0])
                    LaB = radians(center[1])
                    D_Lo = LoB - LoA
                    D_La = LaB - LaA
                    P = sin(D_La / 2) ** 2 + cos(LaA) * cos(LaB) * sin(D_Lo / 2) ** 2
                    Q = 2 * asin(sqrt(P))
                    R_km = 6371
                    dis = Q * R_km
                    return dis

                def get_degree(data):
                    LaB = data['纬度(°)']
                    LoB = data['经度(°)']
                    LoB = radians(LoB)
                    LaB = radians(LaB)
                    LoA = radians(center[0])
                    LaA = radians(center[1])
                    dLon = LoB - LoA
                    y = sin(dLon) * cos(LaB)
                    x = cos(LaA) * sin(LaB) - sin(LaA) * cos(LaB) * cos(dLon)
                    brng = math.degrees(math.atan2(y, x))
                    brng = (brng + 360) % 360
                    return brng

                # 给地震目录添加相对于台站的方向特征 位置特征
                dis = df_loc.apply(get_distance, axis=1)
                dgr = df_loc.apply(get_degree, axis=1)
                print('距离和方向\n', dis, len(dis), dgr, len(dgr))
                # 地震目录加入距离方向特征
                df_area.insert(df_area.shape[1], 'distance', dis)
                df_area.insert(df_area.shape[1], 'degree', dgr)
                print('新的地震目录形式\n', df_area)
                df.insert(df.shape[1], 'distance', 0)
                df.insert(df.shape[1], 'degree', 0)

                # 将距离角度特征加入数据
                def add_dis_dgr(df_area):
                    dis = np.array(df_area['distance'])
                    dgr = np.array(df_area['degree'])
                    df_earth1 = pd.to_datetime(df_area['发震日期（北京时间）'])
                    time = df_earth1.strftime('%Y-%m-%d-%H:%M')
                    ind = df[df.datetime == time].index.tolist()[0]
                    df['distance'].iloc[ind - 10:ind + 30] = dis
                    df['degree'].iloc[ind - 10:ind + 30] = dgr

                df_area.apply(add_dis_dgr, axis=1)
                print('新的数据形式\n', df, df.shape)
                messagebox.showinfo('完成提示', '当前文本数据已经处理完成，可用于模型预测')
                self.progressbar.stop()
                self.progressbar.grid_forget()
                self.moreEntry.config(state=tk.NORMAL)
                self.delStation.config(state=tk.NORMAL)
                self.predictButton.config(state=tk.NORMAL)

            # except:
            #     messagebox.showwarning('错误提示', '当前没有地震目录，请先导入目录')
            except Exception as ex:
                print(ex)

        except Exception as ex:
            print(ex)

    def process_data(self,filepaths):
        i=0
        for f in filepaths:
            filename = f.split('/')[-1][0:f.split('/')[-1].index('.')]
            if '62003_5' in filename or '62053_1' in filename:
                def get_direction2(url):
                    datalist = []
                    date = []
                    with open(url, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            date.append(
                                line[0:4] + '-' + line[4:6] + '-' + line[6:8] + '-' + line[8:10] + ':' + line[10:12])
                            # 取日期后面的数据并以空格分割获得储存字符的列表
                            temp = line[13:].strip()
                            # 每次合并字符列表记录次数
                            datalist.append(temp)

                    datalist = list(map(float, datalist))
                    print('共有{:.0f}天{}个数据'.format(len(datalist) / 1440, len(datalist)))
                    print(f'当前文件为{f}')

                    return datalist, date

                try:
                    direction5, date5 = get_direction2(f)
                    direction6, date6 = get_direction2(f)
                    direction7, date7 = get_direction2(f)
                    direction8, date8 = get_direction2(f)
                    # 可能不同方向数据量不同，以少的为准
                    data_len = min(len(direction5), len(direction6), len(direction7), len(direction8))
                    direction5 = direction5[0:data_len]
                    direction6 = direction6[0:data_len]
                    direction7 = direction7[0:data_len]
                    direction8 = direction8[0:data_len]
                    date5 = date5[0:data_len]
                    df = pd.DataFrame(
                        {'datetime':date5,'direction_first': direction5, 'direction_second': direction6, 'direction_third': direction7,
                         'direction_fourth': direction8})

                    self.df=df


                except MemoryError:
                    messagebox.showerror("导入错误", "文件过大导入失败，请重试")
            elif '64021_2' in filename or '64002_2' in filename:
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
                    print(f'当前文件为{f}')
                    date = []
                    for y in year:
                        for t in self.time1:
                            datetime = y + '-' + t
                            date.append(datetime)
                    return datalist, date

                try:
                    direction1, date1 = get_direction(f)
                    direction2, date2 = get_direction(f)
                    direction3, date3 = get_direction(f)
                    direction4, date4 = get_direction(f)
                    data_len = min(len(direction1), len(direction2), len(direction3), len(direction4))
                    direction1 = direction1[0:data_len]
                    direction2 = direction2[0:data_len]
                    direction3 = direction3[0:data_len]
                    direction4 = direction4[0:data_len]
                    date1 = date1[0:data_len]
                    df = pd.DataFrame(
                        {'datetime':date1,'direction_first': direction1, 'direction_second': direction2, 'direction_third': direction3,
                         'direction_fourth': direction4})
                    self.df=df

                except MemoryError:
                    messagebox.showerror("导入错误", "文件过大导入失败，请重试")
            name_station = self.loc_station_list
            lat_num = self.lat_num_list
            long_num = self.long_num_list
            data_len = self.data_len_list
            print(self.start_ind, 'startind')
            cur_ind = i + self.start_ind
            if '64021_2' in filename :
                self.centre_txt=self.centers[0]
                name_station[cur_ind].set('海原')
                lat_num[cur_ind].set(self.centers[0][0])
                long_num[cur_ind].set(self.centers[0][1])
            elif '64002_2' in filename:
                self.centre_txt = self.centers[1]
                name_station[cur_ind].set('银川')
                lat_num[cur_ind].set(self.centers[1][0])
                long_num[cur_ind].set(self.centers[1][1])
            elif '62053_1' in filename:
                self.centre_txt = self.centers[2]
                name_station[cur_ind].set('临夏')
                lat_num[cur_ind].set(self.centers[2][0])
                long_num[cur_ind].set(self.centers[2][1])
            elif '62003_5' in filename:
                self.centre_txt = self.centers[3]
                name_station[cur_ind].set('高台')
                lat_num[cur_ind].set(self.centers[3][0])
                long_num[cur_ind].set(self.centers[3][1])
            data_len[cur_ind].set('共有{:.0f}天{}个数据'.format(self.df.shape[0] / 1440, self.df.shape[0]))
            self.process()
            i+=1

        self.start_ind=i


    def upload_txt(self):
        file_paths = filedialog.askopenfilenames(title="选择TXT文件", filetypes=[("TXT文件", "*.txt")])
        # 高台：62003_5_2321_分 两列 海原：64021_2_2321_分 多列 临夏：62053_1_2321_分 两列 银川：64002_2_2321_分 多列
        self.time1= ['00:00', '00:01', '00:02', '00:03', '00:04', '00:05', '00:06', '00:07', '00:08', '00:09', '00:10',
                     '00:11', '00:12', '00:13', '00:14', '00:15', '00:16', '00:17', '00:18', '00:19', '00:20', '00:21',
                     '00:22', '00:23', '00:24', '00:25', '00:26', '00:27', '00:28', '00:29', '00:30', '00:31', '00:32',
                     '00:33', '00:34', '00:35', '00:36', '00:37', '00:38', '00:39', '00:40', '00:41', '00:42', '00:43',
                     '00:44', '00:45', '00:46', '00:47', '00:48', '00:49', '00:50', '00:51', '00:52', '00:53', '00:54',
                     '00:55', '00:56', '00:57', '00:58', '00:59', '01:00', '01:01', '01:02', '01:03', '01:04', '01:05',
                     '01:06', '01:07', '01:08', '01:09', '01:10', '01:11', '01:12', '01:13', '01:14', '01:15', '01:16',
                     '01:17', '01:18', '01:19', '01:20', '01:21', '01:22', '01:23', '01:24', '01:25', '01:26', '01:27',
                     '01:28', '01:29', '01:30', '01:31', '01:32', '01:33', '01:34', '01:35', '01:36', '01:37', '01:38',
                     '01:39', '01:40', '01:41', '01:42', '01:43', '01:44', '01:45', '01:46', '01:47', '01:48', '01:49',
                     '01:50', '01:51', '01:52', '01:53', '01:54', '01:55', '01:56', '01:57', '01:58', '01:59', '02:00',
                     '02:01', '02:02', '02:03', '02:04', '02:05', '02:06', '02:07', '02:08', '02:09', '02:10', '02:11',
                     '02:12', '02:13', '02:14', '02:15', '02:16', '02:17', '02:18', '02:19', '02:20', '02:21', '02:22',
                     '02:23', '02:24', '02:25', '02:26', '02:27', '02:28', '02:29', '02:30', '02:31', '02:32', '02:33',
                     '02:34', '02:35', '02:36', '02:37', '02:38', '02:39', '02:40', '02:41', '02:42', '02:43', '02:44',
                     '02:45', '02:46', '02:47', '02:48', '02:49', '02:50', '02:51', '02:52', '02:53', '02:54', '02:55',
                     '02:56', '02:57', '02:58', '02:59', '03:00', '03:01', '03:02', '03:03', '03:04', '03:05', '03:06',
                     '03:07', '03:08', '03:09', '03:10', '03:11', '03:12', '03:13', '03:14', '03:15', '03:16', '03:17',
                     '03:18', '03:19', '03:20', '03:21', '03:22', '03:23', '03:24', '03:25', '03:26', '03:27', '03:28',
                     '03:29', '03:30', '03:31', '03:32', '03:33', '03:34', '03:35', '03:36', '03:37', '03:38', '03:39',
                     '03:40', '03:41', '03:42', '03:43', '03:44', '03:45', '03:46', '03:47', '03:48', '03:49', '03:50',
                     '03:51', '03:52', '03:53', '03:54', '03:55', '03:56', '03:57', '03:58', '03:59', '04:00', '04:01',
                     '04:02', '04:03', '04:04', '04:05', '04:06', '04:07', '04:08', '04:09', '04:10', '04:11', '04:12',
                     '04:13', '04:14', '04:15', '04:16', '04:17', '04:18', '04:19', '04:20', '04:21', '04:22', '04:23',
                     '04:24', '04:25', '04:26', '04:27', '04:28', '04:29', '04:30', '04:31', '04:32', '04:33', '04:34',
                     '04:35', '04:36', '04:37', '04:38', '04:39', '04:40', '04:41', '04:42', '04:43', '04:44', '04:45',
                     '04:46', '04:47', '04:48', '04:49', '04:50', '04:51', '04:52', '04:53', '04:54', '04:55', '04:56',
                     '04:57', '04:58', '04:59', '05:00', '05:01', '05:02', '05:03', '05:04', '05:05', '05:06', '05:07',
                     '05:08', '05:09', '05:10', '05:11', '05:12', '05:13', '05:14', '05:15', '05:16', '05:17', '05:18',
                     '05:19', '05:20', '05:21', '05:22', '05:23', '05:24', '05:25', '05:26', '05:27', '05:28', '05:29',
                     '05:30', '05:31', '05:32', '05:33', '05:34', '05:35', '05:36', '05:37', '05:38', '05:39', '05:40',
                     '05:41', '05:42', '05:43', '05:44', '05:45', '05:46', '05:47', '05:48', '05:49', '05:50', '05:51',
                     '05:52', '05:53', '05:54', '05:55', '05:56', '05:57', '05:58', '05:59', '06:00', '06:01', '06:02',
                     '06:03', '06:04', '06:05', '06:06', '06:07', '06:08', '06:09', '06:10', '06:11', '06:12', '06:13',
                     '06:14', '06:15', '06:16', '06:17', '06:18', '06:19', '06:20', '06:21', '06:22', '06:23', '06:24',
                     '06:25', '06:26', '06:27', '06:28', '06:29', '06:30', '06:31', '06:32', '06:33', '06:34', '06:35',
                     '06:36', '06:37', '06:38', '06:39', '06:40', '06:41', '06:42', '06:43', '06:44', '06:45', '06:46',
                     '06:47', '06:48', '06:49', '06:50', '06:51', '06:52', '06:53', '06:54', '06:55', '06:56', '06:57',
                     '06:58', '06:59', '07:00', '07:01', '07:02', '07:03', '07:04', '07:05', '07:06', '07:07', '07:08',
                     '07:09', '07:10', '07:11', '07:12', '07:13', '07:14', '07:15', '07:16', '07:17', '07:18', '07:19',
                     '07:20', '07:21', '07:22', '07:23', '07:24', '07:25', '07:26', '07:27', '07:28', '07:29', '07:30',
                     '07:31', '07:32', '07:33', '07:34', '07:35', '07:36', '07:37', '07:38', '07:39', '07:40', '07:41',
                     '07:42', '07:43', '07:44', '07:45', '07:46', '07:47', '07:48', '07:49', '07:50', '07:51', '07:52',
                     '07:53', '07:54', '07:55', '07:56', '07:57', '07:58', '07:59', '08:00', '08:01', '08:02', '08:03',
                     '08:04', '08:05', '08:06', '08:07', '08:08', '08:09', '08:10', '08:11', '08:12', '08:13', '08:14',
                     '08:15', '08:16', '08:17', '08:18', '08:19', '08:20', '08:21', '08:22', '08:23', '08:24', '08:25',
                     '08:26', '08:27', '08:28', '08:29', '08:30', '08:31', '08:32', '08:33', '08:34', '08:35', '08:36',
                     '08:37', '08:38', '08:39', '08:40', '08:41', '08:42', '08:43', '08:44', '08:45', '08:46', '08:47',
                     '08:48', '08:49', '08:50', '08:51', '08:52', '08:53', '08:54', '08:55', '08:56', '08:57', '08:58',
                     '08:59', '09:00', '09:01', '09:02', '09:03', '09:04', '09:05', '09:06', '09:07', '09:08', '09:09',
                     '09:10', '09:11', '09:12', '09:13', '09:14', '09:15', '09:16', '09:17', '09:18', '09:19', '09:20',
                     '09:21', '09:22', '09:23', '09:24', '09:25', '09:26', '09:27', '09:28', '09:29', '09:30', '09:31',
                     '09:32', '09:33', '09:34', '09:35', '09:36', '09:37', '09:38', '09:39', '09:40', '09:41', '09:42',
                     '09:43', '09:44', '09:45', '09:46', '09:47', '09:48', '09:49', '09:50', '09:51', '09:52', '09:53',
                     '09:54', '09:55', '09:56', '09:57', '09:58', '09:59', '10:00', '10:01', '10:02', '10:03', '10:04',
                     '10:05', '10:06', '10:07', '10:08', '10:09', '10:10', '10:11', '10:12', '10:13', '10:14', '10:15',
                     '10:16', '10:17', '10:18', '10:19', '10:20', '10:21', '10:22', '10:23', '10:24', '10:25', '10:26',
                     '10:27', '10:28', '10:29', '10:30', '10:31', '10:32', '10:33', '10:34', '10:35', '10:36', '10:37',
                     '10:38', '10:39', '10:40', '10:41', '10:42', '10:43', '10:44', '10:45', '10:46', '10:47', '10:48',
                     '10:49', '10:50', '10:51', '10:52', '10:53', '10:54', '10:55', '10:56', '10:57', '10:58', '10:59',
                     '11:00', '11:01', '11:02', '11:03', '11:04', '11:05', '11:06', '11:07', '11:08', '11:09', '11:10',
                     '11:11', '11:12', '11:13', '11:14', '11:15', '11:16', '11:17', '11:18', '11:19', '11:20', '11:21',
                     '11:22', '11:23', '11:24', '11:25', '11:26', '11:27', '11:28', '11:29', '11:30', '11:31', '11:32',
                     '11:33', '11:34', '11:35', '11:36', '11:37', '11:38', '11:39', '11:40', '11:41', '11:42', '11:43',
                     '11:44', '11:45', '11:46', '11:47', '11:48', '11:49', '11:50', '11:51', '11:52', '11:53', '11:54',
                     '11:55', '11:56', '11:57', '11:58', '11:59', '12:00', '12:01', '12:02', '12:03', '12:04', '12:05',
                     '12:06', '12:07', '12:08', '12:09', '12:10', '12:11', '12:12', '12:13', '12:14', '12:15', '12:16',
                     '12:17', '12:18', '12:19', '12:20', '12:21', '12:22', '12:23', '12:24', '12:25', '12:26', '12:27',
                     '12:28', '12:29', '12:30', '12:31', '12:32', '12:33', '12:34', '12:35', '12:36', '12:37', '12:38',
                     '12:39', '12:40', '12:41', '12:42', '12:43', '12:44', '12:45', '12:46', '12:47', '12:48', '12:49',
                     '12:50', '12:51', '12:52', '12:53', '12:54', '12:55', '12:56', '12:57', '12:58', '12:59', '13:00',
                     '13:01', '13:02', '13:03', '13:04', '13:05', '13:06', '13:07', '13:08', '13:09', '13:10', '13:11',
                     '13:12', '13:13', '13:14', '13:15', '13:16', '13:17', '13:18', '13:19', '13:20', '13:21', '13:22',
                     '13:23', '13:24', '13:25', '13:26', '13:27', '13:28', '13:29', '13:30', '13:31', '13:32', '13:33',
                     '13:34', '13:35', '13:36', '13:37', '13:38', '13:39', '13:40', '13:41', '13:42', '13:43', '13:44',
                     '13:45', '13:46', '13:47', '13:48', '13:49', '13:50', '13:51', '13:52', '13:53', '13:54', '13:55',
                     '13:56', '13:57', '13:58', '13:59', '14:00', '14:01', '14:02', '14:03', '14:04', '14:05', '14:06',
                     '14:07', '14:08', '14:09', '14:10', '14:11', '14:12', '14:13', '14:14', '14:15', '14:16', '14:17',
                     '14:18', '14:19', '14:20', '14:21', '14:22', '14:23', '14:24', '14:25', '14:26', '14:27', '14:28',
                     '14:29', '14:30', '14:31', '14:32', '14:33', '14:34', '14:35', '14:36', '14:37', '14:38', '14:39',
                     '14:40', '14:41', '14:42', '14:43', '14:44', '14:45', '14:46', '14:47', '14:48', '14:49', '14:50',
                     '14:51', '14:52', '14:53', '14:54', '14:55', '14:56', '14:57', '14:58', '14:59', '15:00', '15:01',
                     '15:02', '15:03', '15:04', '15:05', '15:06', '15:07', '15:08', '15:09', '15:10', '15:11', '15:12',
                     '15:13', '15:14', '15:15', '15:16', '15:17', '15:18', '15:19', '15:20', '15:21', '15:22', '15:23',
                     '15:24', '15:25', '15:26', '15:27', '15:28', '15:29', '15:30', '15:31', '15:32', '15:33', '15:34',
                     '15:35', '15:36', '15:37', '15:38', '15:39', '15:40', '15:41', '15:42', '15:43', '15:44', '15:45',
                     '15:46', '15:47', '15:48', '15:49', '15:50', '15:51', '15:52', '15:53', '15:54', '15:55', '15:56',
                     '15:57', '15:58', '15:59', '16:00', '16:01', '16:02', '16:03', '16:04', '16:05', '16:06', '16:07',
                     '16:08', '16:09', '16:10', '16:11', '16:12', '16:13', '16:14', '16:15', '16:16', '16:17', '16:18',
                     '16:19', '16:20', '16:21', '16:22', '16:23', '16:24', '16:25', '16:26', '16:27', '16:28', '16:29',
                     '16:30', '16:31', '16:32', '16:33', '16:34', '16:35', '16:36', '16:37', '16:38', '16:39', '16:40',
                     '16:41', '16:42', '16:43', '16:44', '16:45', '16:46', '16:47', '16:48', '16:49', '16:50', '16:51',
                     '16:52', '16:53', '16:54', '16:55', '16:56', '16:57', '16:58', '16:59', '17:00', '17:01', '17:02',
                     '17:03', '17:04', '17:05', '17:06', '17:07', '17:08', '17:09', '17:10', '17:11', '17:12', '17:13',
                     '17:14', '17:15', '17:16', '17:17', '17:18', '17:19', '17:20', '17:21', '17:22', '17:23', '17:24',
                     '17:25', '17:26', '17:27', '17:28', '17:29', '17:30', '17:31', '17:32', '17:33', '17:34', '17:35',
                     '17:36', '17:37', '17:38', '17:39', '17:40', '17:41', '17:42', '17:43', '17:44', '17:45', '17:46',
                     '17:47', '17:48', '17:49', '17:50', '17:51', '17:52', '17:53', '17:54', '17:55', '17:56', '17:57',
                     '17:58', '17:59', '18:00', '18:01', '18:02', '18:03', '18:04', '18:05', '18:06', '18:07', '18:08',
                     '18:09', '18:10', '18:11', '18:12', '18:13', '18:14', '18:15', '18:16', '18:17', '18:18', '18:19',
                     '18:20', '18:21', '18:22', '18:23', '18:24', '18:25', '18:26', '18:27', '18:28', '18:29', '18:30',
                     '18:31', '18:32', '18:33', '18:34', '18:35', '18:36', '18:37', '18:38', '18:39', '18:40', '18:41',
                     '18:42', '18:43', '18:44', '18:45', '18:46', '18:47', '18:48', '18:49', '18:50', '18:51', '18:52',
                     '18:53', '18:54', '18:55', '18:56', '18:57', '18:58', '18:59', '19:00', '19:01', '19:02', '19:03',
                     '19:04', '19:05', '19:06', '19:07', '19:08', '19:09', '19:10', '19:11', '19:12', '19:13', '19:14',
                     '19:15', '19:16', '19:17', '19:18', '19:19', '19:20', '19:21', '19:22', '19:23', '19:24', '19:25',
                     '19:26', '19:27', '19:28', '19:29', '19:30', '19:31', '19:32', '19:33', '19:34', '19:35', '19:36',
                     '19:37', '19:38', '19:39', '19:40', '19:41', '19:42', '19:43', '19:44', '19:45', '19:46', '19:47',
                     '19:48', '19:49', '19:50', '19:51', '19:52', '19:53', '19:54', '19:55', '19:56', '19:57', '19:58',
                     '19:59', '20:00', '20:01', '20:02', '20:03', '20:04', '20:05', '20:06', '20:07', '20:08', '20:09',
                     '20:10', '20:11', '20:12', '20:13', '20:14', '20:15', '20:16', '20:17', '20:18', '20:19', '20:20',
                     '20:21', '20:22', '20:23', '20:24', '20:25', '20:26', '20:27', '20:28', '20:29', '20:30', '20:31',
                     '20:32', '20:33', '20:34', '20:35', '20:36', '20:37', '20:38', '20:39', '20:40', '20:41', '20:42',
                     '20:43', '20:44', '20:45', '20:46', '20:47', '20:48', '20:49', '20:50', '20:51', '20:52', '20:53',
                     '20:54', '20:55', '20:56', '20:57', '20:58', '20:59', '21:00', '21:01', '21:02', '21:03', '21:04',
                     '21:05', '21:06', '21:07', '21:08', '21:09', '21:10', '21:11', '21:12', '21:13', '21:14', '21:15',
                     '21:16', '21:17', '21:18', '21:19', '21:20', '21:21', '21:22', '21:23', '21:24', '21:25', '21:26',
                     '21:27', '21:28', '21:29', '21:30', '21:31', '21:32', '21:33', '21:34', '21:35', '21:36', '21:37',
                     '21:38', '21:39', '21:40', '21:41', '21:42', '21:43', '21:44', '21:45', '21:46', '21:47', '21:48',
                     '21:49', '21:50', '21:51', '21:52', '21:53', '21:54', '21:55', '21:56', '21:57', '21:58', '21:59',
                     '22:00', '22:01', '22:02', '22:03', '22:04', '22:05', '22:06', '22:07', '22:08', '22:09', '22:10',
                     '22:11', '22:12', '22:13', '22:14', '22:15', '22:16', '22:17', '22:18', '22:19', '22:20', '22:21',
                     '22:22', '22:23', '22:24', '22:25', '22:26', '22:27', '22:28', '22:29', '22:30', '22:31', '22:32',
                     '22:33', '22:34', '22:35', '22:36', '22:37', '22:38', '22:39', '22:40', '22:41', '22:42', '22:43',
                     '22:44', '22:45', '22:46', '22:47', '22:48', '22:49', '22:50', '22:51', '22:52', '22:53', '22:54',
                     '22:55', '22:56', '22:57', '22:58', '22:59', '23:00', '23:01', '23:02', '23:03', '23:04', '23:05',
                     '23:06', '23:07', '23:08', '23:09', '23:10', '23:11', '23:12', '23:13', '23:14', '23:15', '23:16',
                     '23:17', '23:18', '23:19', '23:20', '23:21', '23:22', '23:23', '23:24', '23:25', '23:26', '23:27',
                     '23:28', '23:29', '23:30', '23:31', '23:32', '23:33', '23:34', '23:35', '23:36', '23:37', '23:38',
                     '23:39', '23:40', '23:41', '23:42', '23:43', '23:44', '23:45', '23:46', '23:47', '23:48', '23:49',
                     '23:50', '23:51', '23:52', '23:53', '23:54', '23:55', '23:56', '23:57', '23:58', '23:59']
        if len(file_paths):
            name_list=[]
            tk.messagebox.showinfo("导入提示", "导入文本过大，处理需要一定时间")

            self.progress.set(0)
            self.moreEntry.config(state=tk.DISABLED)
            self.delStation.config(state=tk.DISABLED)
            self.predictButton.config(state=tk.DISABLED)
            self.progressbar.grid(column=0, row=2, ipadx=10, ipady=5, pady=5)
            self.progressbar.start()

            self.new_thread = threading.Thread(target=self.process_data,args=(file_paths,))
            self.new_thread.start()


    def upload_file(self):
        file_paths = filedialog.askopenfilenames(title="选择CSV文件", filetypes=[("CSV文件", "*.csv")])

        if len(file_paths):
            station_list = []
            earth_path = ''
            name_list = []
            earth_list = []
            for f in file_paths:
                file_name = f.split('/')[-1][0:f.split('/')[-1].index('.')]
                if 'Area' not in file_name:
                    station_list.append(f)
                    name_list.append(file_name)
                else:
                    earth_list.append(f)
                    earth_path = f
            sn = self.station_num
            cur_num = -1
            if sn.get() == '':
                cur_num = 0
            else:
                cur_num = int(sn.get())
            if len(earth_list) > 1 or (len(station_list) + cur_num) > 4:
                messagebox.showwarning("导入警告", "导入文件超过规定数量，请重试")
            else:

                if earth_path != '':
                    df_area = pd.read_csv(earth_path, encoding='gb2312')
                    self.df_area=df_area
                    en = self.earthquake_num
                    en.set(df_area.shape[0])
                    earthquake_import = self.earthquake_import
                    earthquake_import.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    lat_max = df_area['纬度(°)'].max()
                    lat_min = df_area['纬度(°)'].min()
                    long_max = df_area['经度(°)'].max()
                    long_min = df_area['经度(°)'].min()
                    depth_max = df_area['震源深度(Km)'].max()
                    depth_min = df_area['震源深度(Km)'].min()
                    level_max = df_area['震级'].max()
                    level_min = df_area['震级'].min()
                    lat =self.lat
                    long = self.long
                    depth = self.depth
                    level = self.level
                    country = self.country
                    tim = self.tim
                    lat.set(f'{lat_min}-{lat_max}')
                    long.set(f'{long_min}-{long_max}')
                    depth.set(f'{depth_min}-{depth_max}')
                    level.set(f'{level_min}-{level_max}')
                    country.set('国内')
                    df_earth = df_area['发震日期（北京时间）']
                    df_earth1 = pd.to_datetime(df_earth)
                    df_time = df_earth1.dt.strftime('%Y/%m/%d')
                    tim.set(f'{df_time.min()}~{df_time.max()}')

                if len(station_list):
                    tk.messagebox.showinfo("导入提示", "导入文件过大，处理需要一定时间")

                    self.progress.set(0)
                    self.moreEntry.config(state=tk.DISABLED)
                    self.delStation.config(state=tk.DISABLED)
                    self.predictButton.config(state=tk.DISABLED)
                    self.progressbar.grid(column=0, row=2, ipadx=10, ipady=5, pady=5)
                    self.progressbar.start()

                    self.complete_num=0
                    name_station =self.loc_station_list
                    lat_num = self.lat_num_list
                    long_num = self.long_num_list
                    data_len = self.data_len_list
                    i = 0

                    print(self.start_ind, 'startind')

                    sn.set(len(station_list) + self.start_ind)
                    self.station_list_len=len(station_list)
                    for f in station_list:
                        self.current_path=f
                        cur_ind = i + self.start_ind
                        if 'haiyuan' in name_list[i]:
                            name_station[cur_ind].set('海原')
                            lat_num[cur_ind].set(self.centers[0][0])
                            long_num[cur_ind].set(self.centers[0][1])
                        elif 'yinchuan' in name_list[i]:
                            name_station[cur_ind].set('银川')
                            lat_num[cur_ind].set(self.centers[1][0])
                            long_num[cur_ind].set(self.centers[1][1])
                        elif 'linxia' in name_list[i]:
                            name_station[cur_ind].set('临夏')
                            lat_num[cur_ind].set(self.centers[2][0])
                            long_num[cur_ind].set(self.centers[2][1])
                        if 'gaotai' in name_list[i]:
                            name_station[cur_ind].set('高台')
                            lat_num[cur_ind].set(self.centers[3][0])
                            long_num[cur_ind].set(self.centers[3][1])
                        self.data_len_index=cur_ind
                        print('data_len_index为',self.data_len_index)
                        self.start()

                        i += 1
                    self.start_ind = self.start_ind + len(station_list)
                    station_import = self.station_import
                    station_import.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))



    def getweb(self,url):
        webbrowser.open_new(url)

    def predict_loc(self):
        print(f'目前{self.station_num.get()}个台站数据')
        try:
            for i in range(0, int(self.station_num.get())):
                df_cur = self.df_list[i]
                df_earth = self.df_area
                test_dataset = MyDataset(df_cur, df_earth)
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

                DATA_LEN = test_dataset.test_len
                d_input = test_dataset.input_len
                d_channel = test_dataset.channel_len
                d_output = test_dataset.output_len
                # 维度展示
                print('data structure: [lines, timesteps, features]')
                print(f'test data size: [{DATA_LEN, d_input, d_channel}]')
                print(f'Number of classes: {d_output}')
                y_true = np.array([])
                y_pred = np.array([])

                with torch.no_grad():
                    c_0 = c_1 = c_2 = c_3 = c_4 = c_5 = c_6 = c_7 = c_8 = 0
                    total = 0
                    try:
                        for x, y in test_dataloader:
                            x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                            y_pre, encoding, score_input, score_channel, gather_input, gather_channel, gate = self.model(
                                x.to(self.DEVICE), 'test')
                            _, label_index = torch.max(y_pre.data, dim=-1)
                            total += label_index.shape[0]
                            # 获取对的样本数
                            correct_labels = y[label_index == y.long()]
                            c_0 += (correct_labels == 0).sum().item()
                            c_1 += (correct_labels == 1).sum().item()
                            c_2 += (correct_labels == 2).sum().item()
                            c_3 += (correct_labels == 3).sum().item()
                            c_4 += (correct_labels == 4).sum().item()
                            c_5 += (correct_labels == 5).sum().item()
                            c_6 += (correct_labels == 6).sum().item()
                            c_7 += (correct_labels == 7).sum().item()
                            c_8 += (correct_labels == 8).sum().item()

                            y_numpy = np.array(y.data.cpu())
                            label_index_numpy = np.array(label_index.data.cpu())
                            y_true = np.append(y_true, y_numpy)
                            y_pred = np.append(y_pred, label_index_numpy)
                            correct_index = np.where(y_true == y_pred)
                            self.correct_index = correct_index
                            print(correct_index, len(correct_index))
                        unique, count = np.unique(test_dataset.test_label, return_counts=True)
                        print(
                            f'测试标签各类别出现个数: 1:{count[0]},2:{count[1]},3:{count[2]},4:{count[3]},5:{count[4]},6:{count[5]},7:{count[6]},8:{count[7]},9:{count[8]}')
                        print(f'测试标签各类别预测对的个数：1:{c_0},2:{c_1},3:{c_2},4:{c_3},5:{c_4},6:{c_5},7:{c_6},8:{c_7},9:{c_8}')
                        print(
                            f'测试标签各类别预测准确率：1:{round(c_0 / count[0] * 100, 2)},2:{round(c_1 / count[1] * 100, 2)},3:{round(c_2 / count[2] * 100, 2)},4:{round(c_3 / count[3] * 100, 2)},'
                            f'5:{round(c_4 / count[4] * 100, 2)},6:{round(c_5 / count[5] * 100, 2)},7:{round(c_6 / count[6] * 100, 2)},8:{round(c_7 / count[7] * 100, 2)},9:{round(c_8 / count[8] * 100, 2)}')
                        cm = confusion_matrix(y_true, y_pred)
                        print('混淆矩阵为:\n', cm)
                        conf_matrix = pd.DataFrame(cm,
                                                   index=['Zone0', 'Zone1', 'Zone2', 'Zone3', 'Zone4', 'Zone5', 'Zone6',
                                                          'Zone7', 'Zone8'],
                                                   columns=['Zone0', 'Zone1', 'Zone2', 'Zone3', 'Zone4', 'Zone5',
                                                            'Zone6', 'Zone7', 'Zone8'])

                        # plot size setting
                        fig, ax = plt.subplots(figsize=(20, 10))
                        sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues")
                        plt.ylabel('True label', fontsize=18)
                        plt.xlabel('Predicted label', fontsize=18)
                        plt.xticks(fontsize=15)
                        plt.yticks(fontsize=10)
                        # plt.show()
                        print('Weighted precision', precision_score(y_true, y_pred, average='weighted'))
                        print('Weighted recall', recall_score(y_true, y_pred, average='weighted'))
                        print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted'))
                        print('accuracy', accuracy_score(y_true, y_pred))
                        test_time_list = test_dataset.time_list
                        test_time_list = np.array(test_time_list)
                        time_correct = test_time_list[self.correct_index]
                        print('猜对的时间', time_correct, len(time_correct))
                        test_label = test_dataset.test_label - 1
                        print('对应的标签为', test_label[self.correct_index], len(test_label[self.correct_index]))
                        test_list_nd = np.array(test_dataset.test_list)
                        print('猜对的时间在地震目录中索引为', test_list_nd[self.correct_index])
                        self.display_index = test_list_nd[self.correct_index]
                        test_label=test_label.numpy()
                        true_0_index=np.argwhere(test_label[self.correct_index]==0)
                        true_0_index=true_0_index.squeeze()
                        display_0_index=self.display_index[true_0_index]
                        display_0_index=display_0_index.tolist()

                        true_1_index = np.argwhere(test_label[self.correct_index] == 1)
                        true_1_index = true_1_index.squeeze()
                        display_1_index = self.display_index[true_1_index]
                        display_1_index = display_1_index.tolist()

                        true_2_index = np.argwhere(test_label[self.correct_index] == 2)
                        true_2_index = true_2_index.squeeze()
                        display_2_index = self.display_index[true_2_index]
                        display_2_index = display_2_index.tolist()

                        true_3_index = np.argwhere(test_label[self.correct_index] == 3)
                        true_3_index = true_3_index.squeeze()
                        display_3_index = self.display_index[true_3_index]
                        display_3_index = display_3_index.tolist()

                        true_4_index = np.argwhere(test_label[self.correct_index] == 4)
                        true_4_index = true_4_index.squeeze()
                        display_4_index = self.display_index[true_4_index]
                        display_4_index = display_4_index.tolist()

                        true_5_index = np.argwhere(test_label[self.correct_index] == 5)
                        true_5_index = true_5_index.squeeze()
                        display_5_index = self.display_index[true_5_index]
                        display_5_index = display_5_index.tolist()

                        true_6_index = np.argwhere(test_label[self.correct_index] == 6)
                        true_6_index = true_6_index.squeeze()
                        display_6_index = self.display_index[true_6_index]
                        display_6_index = display_6_index.tolist()

                        true_7_index = np.argwhere(test_label[self.correct_index] == 7)
                        true_7_index = true_7_index.squeeze()
                        display_7_index = self.display_index[true_7_index]
                        display_7_index = display_7_index.tolist()

                        true_8_index = np.argwhere(test_label[self.correct_index] == 8)
                        true_8_index = true_8_index.squeeze()
                        display_8_index = self.display_index[true_8_index]
                        display_8_index = display_8_index.tolist()
                        self.display_index_list=[display_0_index,display_1_index,display_2_index,display_3_index,display_4_index,display_5_index,display_6_index,display_7_index,display_8_index]
                        self.get_map(i)
                    except RuntimeError:
                        messagebox.showwarning("模型警告", "台站文件特征数与模型不匹配，请重新导入")
                        break
        except AttributeError:
            messagebox.showerror('操作错误','请先导入地震目录或者模型再进行预测')



    def get_map(self,index):
        # 地震区域分块
        # # 设置超参数
        N_CLUSTERS = 9  # 类簇的数量
        MARKERS = ['*', 'v', '+', '^', 's', 'x', 'o', '<', '>']  # 标记样式（绘图）
        COLORS = ['pink', 'g', 'm', 'c', 'y', 'b', 'orange', 'k', 'yellow']  # 标记颜色（绘图）

        x = self.df_area[['纬度(°)', '经度(°)']]
        x_np = np.array(x)  # 将x转化为numpy数组

        # 模型构建
        model = KMeans(N_CLUSTERS, random_state=2)  # 构建聚类器
        model.fit(x)  # 训练聚类器
        labels = model.labels_  # 获取聚类标签
        print('聚类标签', labels, len(labels), type(labels))
        unique, count = np.unique(labels, return_counts=True)
        data_count = dict(zip(unique, count))
        print('各类别出现个数:', data_count)


        # 获取地图基底
        #baseSource 0:谷歌地图, 1: 高德地图，2:腾讯地图
        def getMapObject(baseSource=1, centerLoc=[0, 0], baseLayerTitle='baseLayer'):
            if baseSource == 0:
                m = folium.Map(location=centerLoc,
                               min_zoom=0,
                               max_zoom=19,
                               zoom_start=5,
                               control=False,
                               control_scale=True
                               )

            elif baseSource == 1:
                # 使用高德地图作为绘图的基底
                m = folium.Map(location=centerLoc,
                               zoom_start=5,
                               control_scale=True,
                               control=False,
                               tiles=None
                               )

                folium.TileLayer(
                    tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                    attr="&copy; <a href=http://ditu.amap.com/>高德地图</a>",
                    min_zoom=0,
                    max_zoom=19,
                    control=True,
                    show=True,
                    overlay=False,
                    name=baseLayerTitle
                ).add_to(m)
            else:
                # 使用腾讯地图作为绘图的基底
                m = folium.Map(location=centerLoc,
                               zoom_start=5,
                               control_scale=True,
                               control=False,
                               tiles=None
                               )

                folium.TileLayer(
                    tiles='http://rt{s}.map.gtimg.com/realtimerender?z={z}&x={x}&y={y}&type=vector&style=0',
                    attr="&copy; <a href=http://map.qq.com/>腾讯地图</a>",
                    min_zoom=0,
                    max_zoom=19,
                    control=True,
                    show=True,
                    overlay=False,
                    name=baseLayerTitle
                    ).add_to(m)
            return m

        plotmap1 = getMapObject(centerLoc=[36.47, 101.79])
        # 海原 银川 临夏 高台
        centers = [[36.51, 105.61], [38.61, 105.93], [35.6, 103.2], [39.4, 99.86]]
        folium.Marker(location=[36.51, 105.61], popup=folium.Popup("海原台 纬度36.51 经度105.61", max_width=100),
                      icon=folium.Icon(icon='cloud', color='red'), tooltip='观测点').add_to(plotmap1)
        folium.Marker(location=[38.61, 105.93], popup=folium.Popup("银川台 纬度38.61 经度105.93", max_width=100),
                      icon=folium.Icon(icon='cloud', color='red'), tooltip='观测点').add_to(plotmap1)
        folium.Marker(location=[35.6, 103.2], popup=folium.Popup("临夏台 纬度35.6 经度103.2", max_width=100),
                      icon=folium.Icon(icon='cloud', color='red'), tooltip='观测点').add_to(plotmap1)
        folium.Marker(location=[39.4, 99.86], popup=folium.Popup("高台 纬度39.4 经度99.86", max_width=100),
                      icon=folium.Icon(icon='cloud', color='red'), tooltip='观测点').add_to(plotmap1)

        # 添加经纬度
        plotmap1.add_child(folium.LatLngPopup())
        area_list=[]
        colors = ['pink', 'green', 'magenta', 'cyan', 'brown', 'blue', 'orange', 'black', 'yellow']
        colors_predict=['pink', 'green', 'lightred', 'lightblue', 'beige', 'blue', 'orange', 'black', 'lightgray']
        for i in range(0, 9):
            color = colors[i]
            # 定位不同区域的坐标
            loc_ind = np.where(labels == i)
            # 获取np.where返回值中符合条件的个数
            # print('符合标签的个数为 ',len(loc_ind[0]))
            arr = x_np[loc_ind]
            # 计算外接凸图案的边界点
            hull = ConvexHull(arr)
            # 获取原数据中外围边界点的索引
            hull1 = hull.vertices.tolist()
            hull1.append(hull1[0])
            arr = arr[hull1]
            location = [[ar[0], ar[1]] for ar in arr]
            nd_location = np.array(location)
            # min_xy = np.min(nd_location, axis=0)
            # max_xy = np.max(nd_location, axis=0)
            # area_xy = np.vstack((min_xy, max_xy))
            # area_list.append(area_xy)
            plotmap1.add_child(plugins.MarkerCluster(location))
            folium.Polygon(
                locations=location,
                popup=folium.Popup(f'区域{i}', max_width=200),
                color=color,  # 线颜色
                fill=True,  # 是否填充
                weight=3,  # 边界线宽
            ).add_to(plotmap1)



        # 创建一个预测对的组
        truegroup_0 = folium.FeatureGroup(name='true_0_Layer', control=True)
        truegroup_1 = folium.FeatureGroup(name='true_1_Layer', control=True)
        truegroup_2 = folium.FeatureGroup(name='true_2_Layer', control=True)
        truegroup_3 = folium.FeatureGroup(name='true_3_Layer', control=True)
        truegroup_4 = folium.FeatureGroup(name='true_4_Layer', control=True)
        truegroup_5 = folium.FeatureGroup(name='true_5_Layer', control=True)
        truegroup_6 = folium.FeatureGroup(name='true_6_Layer', control=True)
        truegroup_7 = folium.FeatureGroup(name='true_7_Layer', control=True)
        truegroup_8 = folium.FeatureGroup(name='true_8_Layer', control=True)
        truegroup_list=[truegroup_0,truegroup_1,truegroup_2,truegroup_3,truegroup_4,truegroup_5,truegroup_6,truegroup_7,truegroup_8]
        display_index_list=self.display_index_list
        for index in range(0,len(display_index_list)):
            display_cur_index=display_index_list[index]
            truegroup_cur=truegroup_list[index]
            if type(display_cur_index)==int:
                display_cur_index=[display_cur_index]
            for i in display_cur_index:
                folium.Marker(location=[x_np[i][0], x_np[i][1]],
                              popup=folium.Popup(f"纬度{x_np[i][0]} 经度{x_np[i][1]}", max_width=100),
                              icon=folium.Icon(color=colors_predict[index]), tooltip='预测正确点').add_to(
                    truegroup_cur)
            truegroup_cur.add_to(plotmap1)
        folium.LayerControl().add_to(plotmap1)

        plotmap1.save(f'folium_map{index+1}.html')
        flag = messagebox.askyesno(title="预测成功", message="需要查看结果吗")
        if flag:
            url='file:///C:/python_pycharm/Machine_learning/'+f'folium_map{index+1}.html'
            self.get_result2(url)


if __name__ == '__main__':
    win=MainWindow()









