import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
print('当前使用的pytorch版本：', torch.__version__)
from utils.random_seed import setup_seed
from torch.utils.data import DataLoader
from dataset_process.earthdataset_process import MyDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

setup_seed(3047)
DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 选择要跑的模型
save_model_path = r'C:\Users\86139\Downloads\kaggle_results_final\bestmodel linxia_new 93.75 batch=128.pth'
file_name = save_model_path.split('/')[-1].split(' ')[0]+''+save_model_path.split('/')[-1].split(' ')[1]


# 绘制HeatMap的命名准备工作
ACCURACY = save_model_path.split('/')[-1].split(' ')[2]  # 使用的模型的准确率
# BATCH_SIZE = int(save_model_path[save_model_path.find('=')+1:save_model_path.rfind('.')])  # 使用的模型的batch_size
BATCH_SIZE=32
heatMap_or_not = True  # 是否绘制Score矩阵的HeatMap图
gather_or_not = True  # 是否绘制单个样本的step和channel上的聚类图
gather_all_or_not = True  # 是否绘制所有样本在特征提取后的聚类图

# 加载模型
net = torch.load(save_model_path, map_location=DEVICE)

path_data=r'./data/linxia_new.csv'
path_earthquake=r'./data/Area1_3-10.csv'
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
df_area = pd.read_csv(path_earthquake, encoding='gb2312')
test_dataset = MyDataset('test',df,df_area)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

DATA_LEN = test_dataset.test_len
d_input = test_dataset.input_len
d_channel = test_dataset.channel_len
d_output = test_dataset.output_len
# 维度展示
print('data structure: [lines, timesteps, features]')
print(f'test data size: [{DATA_LEN, d_input, d_channel}]')
print(f'Number of classes: {d_output}')




correct = 0
total = 0

y_true=np.array([])
y_pred=np.array([])
with torch.no_grad():
    c_0=c_1=c_2=c_3=c_4=c_5=c_6=c_7=c_8=0
    total=0
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre, encoding, score_input, score_channel, gather_input, gather_channel, gate = net(x.to(DEVICE), 'test')
        _, label_index = torch.max(y_pre.data, dim=-1)
        total += label_index.shape[0]
        print(x,x.shape)
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
        y_true=np.append(y_true,y_numpy)
        y_pred=np.append(y_pred,label_index_numpy)


    unique, count = np.unique(test_dataset.test_label, return_counts=True)
    data_count = dict(zip(unique, count))
    print(f'测试标签各类别出现个数: 1:{count[0]},2:{count[1]},3:{count[2]},4:{count[3]},5:{count[4]},6:{count[5]},7:{count[6]},8:{count[7]},9:{count[8]}')
    correct=round((100 * correct / total), 2)
    print(f'测试标签各类别预测对的个数：1:{c_0},2:{c_1},3:{c_2},4:{c_3},5:{c_4},6:{c_5},7:{c_6},8:{c_7},9:{c_8}')
    print(f'测试标签各类别预测准确率：1:{round(c_0/count[0]*100,2)},2:{round(c_1/count[1]*100,2)},3:{round(c_2/count[2]*100,2)},4:{round(c_3/count[3]*100,2)},'
          f'5:{round(c_4/count[4]*100,2)},6:{round(c_5/count[5]*100,2)},7:{round(c_6/count[6]*100,2)},8:{round(c_7/count[7]*100,2)},9:{round(c_8/count[8]*100,2)}')
    cm = confusion_matrix(y_true, y_pred)
    print('混淆矩阵为:\n',cm)
    conf_matrix = pd.DataFrame(cm, index=['Zone0', 'Zone1', 'Zone2','Zone3','Zone4','Zone5','Zone6','Zone7','Zone8'],
                               columns=['Zone0', 'Zone1', 'Zone2','Zone3','Zone4','Zone5','Zone6','Zone7','Zone8'])

    # plot size setting
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues")
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=10)
    print('------Weighted------')
    print('Weighted precision', precision_score(y_true, y_pred, average='weighted'))
    print('Weighted recall', recall_score(y_true, y_pred, average='weighted'))
    print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted'))
    print('accuracy',accuracy_score(y_true,y_pred))
    plt.show()

