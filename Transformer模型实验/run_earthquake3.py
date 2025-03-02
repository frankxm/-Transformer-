from torch.utils.data import DataLoader
from dataset_process.earthdataset_process import MyDataset
import torch.optim as optim
from time import time
from tqdm import tqdm
import numpy as np
from module.transformer import Transformer
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization
import torch, gc
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

# 以下代码从全局设置字体为SimHei（黑体），解决显示中文问题【Windows】
# 设置font.sans-serif 或 font.family 均可
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决中文字体下坐标轴负数的负号显示问题
plt.rcParams['axes.unicode_minus'] = False


setup_seed(3047)  # 设置随机数种子
reslut_figure_path = 'result_figure'  # 结果图像保存路径

path_data=r'./data/linxia_new.csv'
path_earthquake=r'./data/Area1_3-10.csv'

test_interval = 1  # 测试间隔 单位：epoch
draw_key = 1  # 大于等于draw_key才会保存图像
file_name = path_data.split('/')[-1][0:path_data.split('/')[-1].index('.')]  # 获得文件名字

# 线性缩放规则：当 minibatch 大小乘以 k 时，将学习率乘以 k。
# 超参数设置
EPOCH = 2000
BATCH_SIZE =128
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU

print(f'use device: {DEVICE}')




d_model = 512
d_hidden = 1024
q = 8
v = 8
h = 8
N = 8
dropout = 0.2
pe = True  # # 设置的是双塔中 score=pe score=channel默认没有pe
mask = True  # 设置的是双塔中 score=input的mask score=channel默认没有mask
# 优化器选择
optimizer_name = 'Adagrad'
# optimizer_name = 'Adam'



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

train_dataset = MyDataset('train',df,df_area)
test_dataset = MyDataset('test',df,df_area)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

DATA_LEN = train_dataset.train_len  # 训练集样本数量
d_input = train_dataset.input_len  # 时间部数量
d_channel = train_dataset.channel_len  # 时间序列维度
d_output = train_dataset.output_len  # 分类类别

# 维度展示
print('data structure: [lines, timesteps, features]')
print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
print(f'mytest data size: [{train_dataset.test_len, d_input, d_channel}]')
print(f'Number of classes: {d_output}')

batch_times=DATA_LEN//BATCH_SIZE
if DATA_LEN%BATCH_SIZE!=0:
    batch_times+=1

batch_val_times=train_dataset.test_len//BATCH_SIZE
if train_dataset.test_len%BATCH_SIZE!=0:
    batch_val_times+=1
print(f'每次训练迭代batch为{batch_times},测试迭代batch为{batch_val_times}')
# 创建Transformer模型
net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                  q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)


writer = SummaryWriter(log_dir='./log')
fake_input = torch.randn(240,40,6).to(DEVICE)
writer.add_graph(model=net, input_to_model=fake_input)
writer.close()


# 创建loss函数 此处使用 交叉熵损失
loss_function = Myloss()
if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=600,gamma=0.1)

# 用于记录准确率变化
correct_on_train = []
correct_on_test = []
# 用于记录损失变化
loss_list = []
time_cost = 0

val_sum=0
val_loss=[]
loss_list2=[]

losses=[]
losses_val=[]

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []


correct_0=[]
correct_1=[]
correct_2=[]
correct_3=[]
correct_4=[]
correct_5=[]
correct_6=[]
correct_7=[]
correct_8=[]



def find_lr(init_value = 1e-12, final_value=10., beta = 0.98):
    num = len(train_dataloader)-1
    mult = (final_value / init_value) ** (1.0/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_dataloader:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data

        optimizer.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
        outputs, _, _, _, _, _, _ = net(inputs.to(DEVICE), 'train')

        loss = loss_function(outputs, labels.to(DEVICE))

        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses



# 测试函数
def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        net.eval()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        c_0 = c_1 = c_2 = c_3 = c_4 = c_5 = c_6 = c_7 = c_8 = 0
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, _, _, _, _, _, _ = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            loss_val = loss_function(y_pre, y.to(DEVICE))
            print(f'val_loss:{loss_val.item()}\t\t')
            global val_sum
            val_sum+=loss_val.item()
            correct += (label_index == y.long()).sum().item()

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

            y_numpy=np.array(y.data.cpu())
            label_index_numpy=np.array(label_index.data.cpu())
            y_true = np.append(y_true, y_numpy)
            y_pred = np.append(y_pred, label_index_numpy)


        if flag == 'test_set':
            correct_on_test.append(round((100 * correct / total), 2))
            unique, count = np.unique(test_dataset.test_label, return_counts=True)
            data_count = dict(zip(unique, count))
            print(f'测试标签各类别出现个数: 1:{count[0]},2:{count[1]},3:{count[2]},4:{count[3]},5:{count[4]},6:{count[5]},7:{count[6]},8:{count[7]},9:{count[8]}')
            print(f'测试标签各类别预测对的个数：1:{c_0},2:{c_1},3:{c_2},4:{c_3},5:{c_4},6:{c_5},7:{c_6},8:{c_7},9:{c_8}')
            correct_0.append(round(c_0/count[0]*100,2))
            correct_1.append(round(c_1/count[1]*100,2))
            correct_2.append(round(c_2/count[2]*100,2))
            correct_3.append(round(c_3/count[3]*100,2))
            correct_4.append(round(c_4/count[4]*100,2))
            correct_5.append(round(c_5/count[5]*100,2))
            correct_6.append(round(c_6/count[6]*100,2))
            correct_7.append(round(c_7/count[7]*100,2))
            correct_8.append(round(c_8/count[8]*100,2))

            # 计算多分类指标
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            print('样本整体指标：\n')
            print('准确率为:\n', accuracy * 100)
            print('精确率为:\n', precision * 100)
            print('召回率为:\n', recall * 100)
            print('F1score为:\n', f1)

            accuracy_list.append(accuracy * 100)
            precision_list.append(precision * 100)
            recall_list.append(recall * 100)
            f1_list.append(f1)

        elif flag == 'train_set':
            correct_on_train.append(round((100 * correct / total), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))

        return round((100 * correct / total), 2)


# 训练函数
def train():
    gc.collect()
    torch.cuda.empty_cache()

    net.train()
    max_accuracy = 0
    pbar = tqdm(total=EPOCH)
    begin = time()


    for index in range(EPOCH):
        loss_sum=0
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            gc.collect()
            torch.cuda.empty_cache()

            y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')

            loss = loss_function(y_pre, y.to(DEVICE))

            beta=0.98
            batch_num+=1
            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            print(f'Epoch:{index + 1}:\t\tloss:{loss.item()}\t\t')
            print("Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))

            loss_list.append(loss.item())
            loss_sum=loss_sum+loss.item()

            loss.backward()
            optimizer.step()
        loss_sum=loss_sum/batch_times
        loss_list2.append(loss_sum)
        writer.add_scalar(tag='TrainLoss', scalar_value=loss_sum, global_step=index)

        if ((index + 1) % test_interval) == 0:
            current_accuracy = test(test_dataloader)
            global val_sum
            val_sum=val_sum/(batch_val_times)

            writer.add_scalars(main_tag='Metrics_avg', tag_scalar_dict={'ValLoss': val_sum,'Currentacc_test':current_accuracy,
                                                                    }, global_step=index)
            writer.add_scalars(main_tag='labels',tag_scalar_dict={'Correct_0': correct_0[-1], 'Correct_1': correct_1[-1],'Correct_2': correct_2[-1], 'Correct_3': correct_3[-1],
            'Correct_4': correct_4[-1], 'Correct_5': correct_5[-1],'Correct_6': correct_6[-1], 'Correct_7': correct_7[-1],'Correct_8':correct_8[-1]},global_step=index)
            val_loss.append(val_sum)
            val_sum=0

            train_accuracy=test(train_dataloader, 'train_set')
            print(f'当前最大准确率\t测试集:{max(correct_on_test)}%\t 训练集:{max(correct_on_train)}%')

            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                torch.save(net,'bestmodel.pth')
                torch.save(net, r'C:\Users\86139\Downloads\GTN-master\Gated Transformer 论文IJCAI版\saved_model\{} batch={}.pth'.format(file_name,BATCH_SIZE))


        for name, param in net.named_parameters():
            # print('name:',name,type(name),len(name))
            # print('param',param,type(param),len(param))
            writer.add_histogram(tag=name + '_grad', values=param.grad, global_step=index)
            writer.add_histogram(tag=name + '_data', values=param.data, global_step=index)

        pbar.update()
        scheduler.step()

    writer.close()
    os.rename(r'C:\Users\86139\Downloads\GTN-master\Gated Transformer 论文IJCAI版\saved_model\{} batch={}.pth'.format(file_name,BATCH_SIZE),
              r'C:\Users\86139\Downloads\GTN-master\Gated Transformer 论文IJCAI版\saved_model\{} {} batch={}.pth'.format(file_name,max_accuracy, BATCH_SIZE))

    end = time()
    time_cost = round((end - begin) / 60, 2)

    count=0
    # 结果图
    result_visualization(loss_list=loss_list, correct_on_test=correct_on_test, correct_on_train=correct_on_train,
                         test_interval=test_interval,
                         d_model=d_model, q=q, v=v, h=h, N=N, dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
                         time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key, reslut_figure_path=reslut_figure_path,
                         file_name=file_name,
                         optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask,val_loss=val_loss,batch_times=batch_times,loss_list2=loss_list2,
                         batch_val_times=batch_val_times,index=count)



    fig4 = plt.figure(500, figsize=(20, 10))
    ax6 = fig4.add_subplot(221)
    ax7 = fig4.add_subplot(222)
    ax8 = fig4.add_subplot(223)
    ax9 = fig4.add_subplot(224)

    ax6.plot(accuracy_list)
    ax7.plot(precision_list)
    ax8.plot(recall_list)
    ax9.plot(f1_list)

    ax6.set_xlabel(f'epoch/{test_interval}')
    ax6.set_ylabel('accuracy')
    ax6.set_title('accuracy on Test ')

    ax7.set_xlabel(f'epoch/{test_interval}')
    ax7.set_ylabel('precision')
    ax7.set_title('precision on Test ')

    ax8.set_xlabel(f'epoch/{test_interval}')
    ax8.set_ylabel('recall')
    ax8.set_title('recall on Test ')

    ax9.set_xlabel(f'epoch/{test_interval}')
    ax9.set_ylabel('f1')
    ax9.set_title('f1_score on Test ')

    plt.tight_layout()
    plt.savefig(f'{reslut_figure_path}/ {BATCH_SIZE} {LR} indicaters.png')
    plt.show()
#### 测试阶段评价指标更细化 ######
if __name__ == '__main__':
    train()

    # logs, losses = find_lr()
    # plt.figure(1, figsize=(20, 10))
    # plt.plot(logs, losses)
    # plt.xlabel('log10(lr) lr量级')
    # plt.ylabel('loss')
    # plt.title(f'batchsize为{BATCH_SIZE}下loss随log10(lr)变化')
    # print(f'最小loss对应的lr量级{logs[losses.index(min(losses))]}')
    # print(losses)
    # print(logs)
    # plt.savefig(f'{reslut_figure_path}/{file_name} {BATCH_SIZE} 学习率-loss.png')
    # plt.show()


