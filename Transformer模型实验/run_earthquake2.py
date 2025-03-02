from torch.utils.data import DataLoader
from dataset_process.earthdataset_process import MyDataset
import torch.optim as optim
from time import time
from tqdm import tqdm
import os

from module.transformer import Transformer
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization
import torch, gc
import pandas as pd
import torch.nn as nn
import math
import matplotlib.pyplot as plt



# 测试函数
def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
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

        if flag == 'test_set':
            correct_on_test.append(round((100 * correct / total), 2))
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
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            gc.collect()
            torch.cuda.empty_cache()

            y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')

            loss = loss_function(y_pre, y.to(DEVICE))

            # for p in optimizer.param_groups:
            #     p['lr'] = lr
            print(f'Epoch:{index + 1}:\t\tloss:{loss.item()}\t\t')
            print("Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))

            loss_list.append(loss.item())
            loss_sum=loss_sum+loss.item()
            loss.backward()

            optimizer.step()
        loss_sum=loss_sum/batch_times
        loss_list2.append(loss_sum)
        if ((index + 1) % test_interval) == 0:
            current_accuracy = test(test_dataloader)
            global val_sum
            val_sum=val_sum/(batch_val_times)
            val_loss.append(val_sum)
            val_sum=0
            train_accuracy=test(train_dataloader, 'train_set')
            print(f'当前最大准确率\t测试集:{max(correct_on_test)}%\t 训练集:{max(correct_on_train)}%')

            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                torch.save(net,'bestmodel.pth')
                torch.save(net, r'C:\Users\86139\Downloads\GTN-master\Gated Transformer 论文IJCAI版\saved_model\{} batch={}.pth'.format(file_name,BATCH_SIZE))

        pbar.update()
        # scheduler.step()

    os.rename(r'C:\Users\86139\Downloads\GTN-master\Gated Transformer 论文IJCAI版\saved_model\{} batch={}.pth'.format(file_name,BATCH_SIZE),
              r'C:\Users\86139\Downloads\GTN-master\Gated Transformer 论文IJCAI版\saved_model\{} {} batch={}.pth'.format(file_name,max_accuracy, BATCH_SIZE))

    end = time()
    time_cost = round((end - begin) / 60, 2)
    time_cost_list.append(time_cost)

    # # 结果图
    # result_visualization(loss_list=loss_list, correct_on_test=correct_on_test, correct_on_train=correct_on_train,
    #                      test_interval=test_interval,
    #                      d_model=d_model, q=q, v=v, h=h, N=N, dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
    #                      time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key, reslut_figure_path=reslut_figure_path,
    #                      file_name=earth_name,
    #                      optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask, val_loss=val_loss,
    #                      batch_times=batch_times, loss_list2=loss_list2,batch_val_times=batch_val_times,index=count)

###### 探究batchsize对训练的影响 ##########
if __name__ == '__main__':
    setup_seed(3047)  # 设置随机数种子
    reslut_figure_path = 'result_figure'  # 结果图像保存路径

    path_data = r'./data/linxia_new.csv'
    path_earthquake = r'./data/Area1_3-10.csv'

    test_interval = 1  # 测试间隔 单位：epoch
    draw_key = 1  # 大于等于draw_key才会保存图像
    file_name = path_data.split('/')[-1][0:path_data.split('/')[-1].index('.')]  # 获得文件名字
    earth_name=path_earthquake.split('/')[-1][0:path_earthquake.split('/')[-1].index('c')]
    earth_name=earth_name[0:-1]
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



    # 线性缩放规则：当 minibatch 大小乘以 k 时，将学习率乘以 k。
    # 超参数设置
    EPOCH = 2000
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
    # optimizer_name = 'Adagrad'
    optimizer_name = 'Adam'

    plt.style.use('seaborn')
    fig3 = plt.figure(100, figsize=(20, 10))
    ax4 = fig3.add_subplot(211)
    ax5 = fig3.add_subplot(212)

    BATCH_SIZE_list=[128,256]
    max_test_correct_list=[]
    min_train_loss_list=[]
    time_cost_list=[]
    count=0
    for b in BATCH_SIZE_list:

        # 用于记录准确率变化
        correct_on_train = []
        correct_on_test = []
        # 用于记录损失变化
        loss_list = []
        time_cost = 0

        val_sum = 0
        val_loss = []
        loss_list2 = []
        BATCH_SIZE=b
        print(f'当前batchsize为{BATCH_SIZE}')
        train_dataset = MyDataset('train', df, df_area)
        test_dataset = MyDataset('test', df, df_area)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        DATA_LEN = train_dataset.train_len  # 训练集样本数量
        d_input = train_dataset.input_len  # 时间部数量
        d_channel = train_dataset.channel_len  # 时间序列维度
        d_output = train_dataset.output_len  # 分类类别

        # 维度展示
        print('data structure: [lines, timesteps, features]')
        print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
        print(f'mytest data size: [{train_dataset.test_len, d_input, d_channel}]')
        print(f'Number of classes: {d_output}')

        batch_times = DATA_LEN // BATCH_SIZE
        if DATA_LEN % BATCH_SIZE != 0:
            batch_times += 1

        batch_val_times = train_dataset.test_len // BATCH_SIZE
        if train_dataset.test_len % BATCH_SIZE != 0:
            batch_val_times += 1
        print(f'每次训练迭代batch为{batch_times},测试迭代batch为{batch_val_times}')
        # 创建Transformer模型
        net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                          q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)

        # 创建loss函数 此处使用 交叉熵损失
        loss_function = Myloss()
        if optimizer_name == 'Adagrad':
            optimizer = optim.Adagrad(net.parameters(), lr=LR)
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=LR)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        train()
        max_test_correct_list.append(max(correct_on_test))
        min_train_loss_list.append(min(loss_list))




        ax4.plot(loss_list2)
        ax4.set_xlabel('epoch')
        ax4.set_ylabel('train_loss')
        ax4.set_title(f'mean_loss in train')
        plt.legend(['Batchsize:128', 'Batchsize:256'])

        ax5.plot(correct_on_test)
        ax5.set_xlabel(f'epoch/{test_interval}')
        ax5.set_ylabel('correct')
        ax5.set_title('CORRECT on Test')
        plt.legend([ 'Batchsize:128', 'Batchsize:256'])
        plt.tight_layout()

        count+=1




    plt.savefig(f'{reslut_figure_path}/{earth_name}  {optimizer_name} epoch={EPOCH}  lr={LR} loss_accuracy.png')
    plt.show()
    for i in range(len(BATCH_SIZE_list)):
        print(f'batchsize为{BATCH_SIZE_list[i]}:')
        print(f' 最大测试正确率为{max_test_correct_list[i]} ',end='')
        print(f' 最小loss：{min_train_loss_list[i]} ',end='')
        print(f' 共耗时{time_cost_list[i]}分钟 ')