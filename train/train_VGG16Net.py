import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transformers
import torch.optim as optim
import os
import json
from tqdm import tqdm
import sys
sys.path.append('E:/VScode/python3/rubbish')
from model.VGG16Net import VGG16Net

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

# 设置画图的字体样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 训练主函数
def main():
     # 选择使用CPU还是GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
    print('using {} train model'.format(device))

    # 对数据集图片进行预处理
    data_transformer = {
        'train': transformers.Compose([
            # 将输入图片大小改成224*224*3
            transformers.RandomResizedCrop(224),
            # 将训练数据集里面的图片进行随机翻转和打乱操作
            transformers.RandomHorizontalFlip(),
            # 把训练数据集图片变成tensor格式
            transformers.ToTensor(),
            # 训练集图片每个通道的均值和方差设置为0.5
            transformers.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) #均值 方差
        ]),
        'val': transformers.Compose([
            # 测试数据集图片大小变成224*224*3
            transformers.Resize((224,224)),
             # 将测试数据集转成tensor格式
            transformers.ToTensor(),
            # 测试集图片每个通道的均值和方差设置为0.5
            transformers.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    }

    # 检查数据集的位置是否存在
    data_path = 'Data/rubbishdata'
    assert os.path.exists(data_path),"data path is not exist"

    # 批大小和工作数量数
    batchsize = 32
    nw = 8
    # 训练数据集定位和读取以及分批处理
    train_data = datasets.ImageFolder(root=os.path.join(data_path,"train"),transform=data_transformer['train'])
    train_data_loader = torch.utils.data.DataLoader(train_data,batch_size=batchsize,shuffle=True,num_workers=nw)
    # 测试数据集定位和读取以及分批处理
    val_data = datasets.ImageFolder(root=os.path.join(data_path,'val'),transform=data_transformer['val'])
    val_data_loader = torch.utils.data.DataLoader(val_data,batch_size=batchsize,shuffle=True,num_workers=nw)
     # 将测试数据集放入迭代器中
    val_iter = iter(val_data_loader)
    val_images,val_labels = next(val_iter)
    # 展示训练数据集和测试数据集分批输入后的大小
    print('using %5d train_data  %5d val_data'%(len(train_data),len(val_data)))
     # 以字典方式展示4种垃圾对应的编号
    class_index = train_data.class_to_idx
    print(class_index)
    index_class = dict((value,key) for key,value in class_index.items())
    print(index_class)

    index_class_json = json.dumps(index_class,indent=4)  #将python 字典格式转为json格式
    # 将垃圾种类和对应的编号写入json文件中
    json_file = './index_class.json'
    with open(json_file,'w') as jf:
        jf.write(index_class_json)

    # 将VGG16Net模型实例化
    model_VGG16Net = VGG16Net().to(device)
    # 损失函数使用交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()
    # 优化器选择Adam优化器，学习率为0.01
    optimizer = optim.Adam(model_VGG16Net.parameters(),lr=0.0001)
    # 定义训练轮数
    epochs = 100
    # 基础准确率为0
    best_acc = 0.0
    # 训练结束的模型参数结果保存位置
    save_path = 'pth/VGG16Net_0.01.pth'
    # 定义画图对应的VGG16Net网络的准确率和损失
    VGG16Net_loss = []
    VGG16Net_acc = []
    
    
    # 训练epochs轮
    for epoch in range(epochs):
        # 定义每轮初始损失为0
        running_loss = 0.0
        # 进度条显示
        train_bar = tqdm(train_data_loader)
        # 将每批训练数据放入迭代器中
        for step1,data in enumerate(train_bar):
            # 将梯度下降初始化为0
            optimizer.zero_grad()
            # 将读入的训练数据集中对应的图片和标签读取出来
            images,lables = data
            # 
            data_num = lables.size()[0]
            # 正向传播得出输出
            output = model_VGG16Net(images.to(device))
            # 按行最大输出对应的索引值
            output_class  = torch.max(output,dim=1)[1]   #见例子代码 dim=1 行最大
            # 计算预测的准确率
            output_acc = torch.eq(output_class,(lables.to(device))).sum().item() #见例子代码
            # 通过交叉熵损失函数计算每次前向传播的损失值
            loss = loss_function(output,lables.to(device))
             # 反向传播参数更新
            loss.backward()
            # 梯度下降
            optimizer.step()
            # 将每批的损失加到损失中得到总损失
            running_loss += loss
            # 打印输出进度条、轮数、准确率、损失
            train_bar.desc = "[epoch%d] train_loss:%.3f train_acc:%.3f"%(epoch,running_loss/(step1+1),output_acc/data_num)

        # 测试集循环
        with torch.no_grad():
             # 通过网络预测出结果
            val_output = model_VGG16Net(val_images.to(device))
            # 计算此参数用测试集图片的对应的损失值
            val_loss = loss_function(val_output,val_labels.to(device))
            # 按行输出预测得到的最大值对应的索引值并保存到val_output_class中
            val_output_class = torch.max(val_output,dim=1)[1]
            # 测试得出的准确率
            val_acc = torch.eq(val_output_class,val_labels.to(device)).sum().item()
            # 打印输出得出的准确率和损失
            print("[epoch%d] val_loss:%.3f  val_acc:%.3f"%(epoch,val_loss,val_acc/val_labels.size()[0]))
            # 将对应的损失放入损失的列表中
            VGG16Net_loss.append(val_loss)
            # 将tensor类型数据转成float类型并加入准确率对应的列表中
            VGG16Net_acc.append(val_acc/val_labels.size()[0])

        # 如果准确率大于基准的准确率0，就保存此训练参数文件
        if (val_acc/val_labels.size()[0]) > best_acc:
            best_acc = val_acc/val_labels.size()[0]
            torch.save(model_VGG16Net.state_dict(),save_path)
    # 打印输出训练结束
    print('train_finished')

    # 定义和写入训练轮数的列表
    epoch_ls = list(range(1, epochs + 1))
    VGG16Net_acc = torch.tensor(VGG16Net_acc, device='cpu')
    VGG16Net_loss = torch.tensor(VGG16Net_loss, device='cpu')
    # 定义画图的画布大小
    fig, ax = plt.subplots(figsize = (7, 5))
    # 在画布中画出VGG16Net损失曲线
    VGG16Net_loss_plt, = ax.plot(epoch_ls, 
                             VGG16Net_loss, 
                             color = 'red', 
                             linewidth = 1.5, 
                             linestyle = ":", 
                             markersize = 4,
                             marker = 'x',
                             label = 'VGG16Net损失')
    # 在画布中画出VGG16Net的准确率曲线
    VGG16Net_acc_plt, = ax.plot(epoch_ls,
                            VGG16Net_acc,
                            color = 'orange',
                            linewidth = 1.5,
                            linestyle = ":",
                            markersize = 4,
                            marker = 'x',
                            label = 'VGG16Net准确率')
    # 
    ax.legend(frameon = False)
    # x,y轴对应的标签和字体设置
    ax.set_xlabel('训练轮数epoch', fontsize=12,fontfamily = 'SimHei',fontstyle='italic')
    ax.set_ylabel('损失值loss/准确率acc', fontsize=12,fontstyle='oblique')
    # 设置图片标题
    ax.set_title('VGG16Net', fontsize = 18)
    # 展示画出的图
    plt.show()
    print('show_finished')
    
    # 训练数据保存到data_train.xlsx中
    df = pd.read_excel('./data_train.xlsx')
    df['VGG16Net_loss_0.001'], df['VGG16Net_acc_0.001'] = VGG16Net_loss, VGG16Net_acc
    df.to_excel('./data_train.xlsx', index=None)
    print('write_finished')



if __name__ == '__main__':
    main()
    
    
    

