# 导包
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取训练数据
df = pd.read_excel('./data_train.xlsx')

# 轮数（x轴）
epoch_ls = df['epoch']

# 三种模型对于的训练数据提取出来
LeNet_001_loss, LeNet_01_loss = df['LeNet_loss_0.001'], df['LeNet_loss_0.01']
LeNet_001_acc, LeNet_01_acc = df['LeNet_acc_0.001'], df['LeNet_acc_0.01']

AlexNet_Adam_loss, AlexNet_SGD_loss = df['AlexNet_loss_Adam_0.001'], df['AlexNet_loss_SGD_0.001']
AlexNet_Adam_acc, AlexNet_SGD_acc = df['AlexNet_acc_Adam_0.001'], df['AlexNet_acc_SGD_0.001']

VGG16Net_001_loss, VGG16Net_01_loss = df['VGG16Net_loss_0.001'], df['VGG16Net_loss_0.01']
VGG16Net_001_acc, VGG16Net_01_acc = df['VGG16Net_acc_0.001'], df['VGG16Net_acc_0.01']


# 画出LeNet网络不同学习率情况下的准确率与损失率对比图
def draw_LeNet_loss_acc():

    # 设置画布大小
    fig, ax_LeNet_acc = plt.subplots(figsize = (7, 5))
    fig, ax_LeNet_loss = plt.subplots(figsize = (7, 5))

    # lr=0.001时的acc折线图画出来
    LeNet_acc_vs_001, = ax_LeNet_acc.plot(epoch_ls,
                                LeNet_001_acc,
                                color = 'red',
                                linewidth = 1.5,
                                linestyle = ":",
                                markersize = 4,
                                marker = 'x',
                                label = 'LeNet_lr=0.001_acc')
    
    # lr=0.01时的acc折线图画出来
    LeNet_acc_vs_01, = ax_LeNet_acc.plot(epoch_ls,
                                LeNet_01_acc,
                                color = 'black',
                                linewidth = 1.5,
                                linestyle = ":",
                                markersize = 4,
                                marker = 'x',
                                label = 'LeNet_lr=0.001_acc')
    
    # lr=0.001时的loss折线图画出来
    LeNet_loss_vs_001, = ax_LeNet_loss.plot(epoch_ls,
                                LeNet_001_loss,
                                color = 'red',
                                linewidth = 1.5,
                                linestyle = ":",
                                markersize = 4,
                                marker = 'x',
                                label = 'LeNet_lr=0.001_loss')
    
    # lr=0.01时的loss折线图画出来
    LeNet_loss_vs_01, = ax_LeNet_loss.plot(epoch_ls,
                                LeNet_01_loss,
                                color = 'black',
                                linewidth = 1.5,
                                linestyle = ":",
                                markersize = 4,
                                marker = 'x',
                                label = 'LeNet_lr=0.01_loss')
    

    ax_LeNet_acc.legend(frameon = False)
    ax_LeNet_loss.legend(frameon = False)

    # 设置坐标轴备注
    ax_LeNet_acc.set_xlabel('训练轮数epoch', 
                            fontsize=12,
                            fontfamily = 'SimHei',
                            fontstyle='italic')
    
    ax_LeNet_acc.set_ylabel('准确率acc', 
                            fontsize=12,
                            fontstyle='oblique')
    
    ax_LeNet_loss.set_xlabel('训练轮数epoch', 
                            fontsize=12,
                            fontfamily = 'SimHei',
                            fontstyle='italic')
    
    ax_LeNet_loss.set_ylabel('损失率loss', 
                            fontsize=12,
                            fontstyle='oblique')
    
    # 展示图片
    plt.show()


# 画出AlexNet网络不同优化器情况下的准确率与损失率对比图
def draw_AlexNet_loss_acc():
    # 设置画布大小
    fig, ax_AlexNet_acc = plt.subplots(figsize = (7, 5))
    fig, ax_ALexNet_loss = plt.subplots(figsize = (7, 5))

    # optim=Adam时的acc折线图画出来
    AlexNet_acc_vs_Adam, = ax_AlexNet_acc.plot(epoch_ls,
                                AlexNet_Adam_acc,
                                color = 'red',
                                linewidth = 1.5,
                                linestyle = ":",
                                markersize = 4,
                                marker = 'x',
                                label = 'AlexNet_Adam_acc')
    
    # optim=SGD时的acc折线图画出来
    AlexNet_acc_vs_SGD, = ax_AlexNet_acc.plot(epoch_ls,
                                AlexNet_SGD_acc,
                                color = 'black',
                                linewidth = 1.5,
                                linestyle = ":",
                                markersize = 4,
                                marker = 'x',
                                label = 'LeNet_SGD_acc')
    
    # optim=Adam时的loss折线图画出来
    LeNet_loss_vs_Adam, = ax_ALexNet_loss.plot(epoch_ls,
                                AlexNet_Adam_loss,
                                color = 'red',
                                linewidth = 1.5,
                                linestyle = ":",
                                markersize = 4,
                                marker = 'x',
                                label = 'LeNet_Adam_loss')
    
    # optim=SGD时的loss折线图画出来
    LeNet_loss_vs_SGD, = ax_ALexNet_loss.plot(epoch_ls,
                                AlexNet_SGD_loss,
                                color = 'black',
                                linewidth = 1.5,
                                linestyle = ":",
                                markersize = 4,
                                marker = 'x',
                                label = 'LeNet_SGD_loss')
    
    # 
    ax_AlexNet_acc.legend(frameon = False)
    ax_ALexNet_loss.legend(frameon = False)

    # 设置坐标轴备注
    ax_AlexNet_acc.set_xlabel('训练轮数epoch', 
                            fontsize=12,
                            fontfamily = 'SimHei',
                            fontstyle='italic')
    
    ax_AlexNet_acc.set_ylabel('准确率acc', 
                            fontsize=12,
                            fontstyle='oblique')
    
    ax_ALexNet_loss.set_xlabel('训练轮数epoch', 
                            fontsize=12,
                            fontfamily = 'SimHei',
                            fontstyle='italic')
    
    ax_ALexNet_loss.set_ylabel('损失率loss', 
                            fontsize=12,
                            fontstyle='oblique')
    
    # 展示图片
    plt.show()


def draw_VGG16Net_loss_acc():

    # 设置画布大小
    fig, ax_VGG16Net_acc = plt.subplots(figsize = (7, 5))
    fig, ax_VGG16Net_loss = plt.subplots(figsize = (7, 5))

    # lr=0.001时的acc折线图画出来
    VGG16Net_acc_vs_001, = ax_VGG16Net_acc.plot(epoch_ls,
                                VGG16Net_001_acc,
                                color = 'red',
                                linewidth = 1.5,
                                linestyle = ":",
                                markersize = 4,
                                marker = 'x',
                                label = 'VGG16Net_lr=0.001_acc')

    # lr=0.01时的acc折线图画出来
    VGG16Net_acc_vs_01, = ax_VGG16Net_acc.plot(epoch_ls,
                                VGG16Net_01_acc,
                                color = 'black',
                                linewidth = 1.5,
                                linestyle = ":",
                                markersize = 4,
                                marker = 'x',
                                label = 'VGG16Net_lr=0.01_acc')

    # lr=0.001时的loss折线图画出来
    LeNet_loss_vs_001, = ax_VGG16Net_loss.plot(epoch_ls,
                                VGG16Net_001_loss,
                                color = 'red',
                                linewidth = 1.5,
                                linestyle = ":",
                                markersize = 4,
                                marker = 'x',
                                label = 'VGG16Net_lr=0.001_loss')

    # lr=0.01时的loss折线图画出来
    LeNet_loss_vs_01, = ax_VGG16Net_loss.plot(epoch_ls,
                                VGG16Net_01_loss,
                                color = 'black',
                                linewidth = 1.5,
                                linestyle = ":",
                                markersize = 4,
                                marker = 'x',
                                label = 'VGG16Net_lr=0.01_loss')
    
    ax_VGG16Net_acc.legend(frameon = False)
    ax_VGG16Net_loss.legend(frameon = False)

    # 设置坐标轴备注
    ax_VGG16Net_acc.set_xlabel('训练轮数epoch', 
                            fontsize=12,
                            fontfamily = 'SimHei',
                            fontstyle='italic')
    
    ax_VGG16Net_acc.set_ylabel('准确率acc', 
                            fontsize=12,
                            fontstyle='oblique')
    
    ax_VGG16Net_loss.set_xlabel('训练轮数epoch', 
                            fontsize=12,
                            fontfamily = 'SimHei',
                            fontstyle='italic')
    
    ax_VGG16Net_loss.set_ylabel('损失率loss', 
                            fontsize=12,
                            fontstyle='oblique')
    
    # 展示图片
    plt.show()


# 画出三种模型对比的acc和loss
def LeNet_AlexNet_VGG16Net_acc_loss():
    # 设置画布大小
    fig, ax_LeNet_AlexNet_VGG16Net_acc = plt.subplots(figsize = (7, 5))
    fig, ax_LeNet_AlexNet_VGG16Net_loss = plt.subplots(figsize = (7, 5))

    # 三种模型在同一学习率条件下对应acc的折线图
    ax_LeNet_AlexNet_VGG16Net_acc_, = ax_LeNet_AlexNet_VGG16Net_acc.plot(epoch_ls,
                                       LeNet_001_acc,
                                       color = 'red',
                                       linewidth = 1.5,
                                       linestyle = ":",
                                       markersize = 4,
                                       marker = 'x',
                                       label = 'LeNet_acc')
    
    ax_LeNet_AlexNet_VGG16Net_acc_, = ax_LeNet_AlexNet_VGG16Net_acc.plot(epoch_ls,
                                       AlexNet_Adam_acc,
                                       color = 'blue',
                                       linewidth = 1.5,
                                       linestyle = ":",
                                       markersize = 4,
                                       marker = 'x',
                                       label = 'AlexNet_acc')
    
    ax_LeNet_AlexNet_VGG16Net_acc_, = ax_LeNet_AlexNet_VGG16Net_acc.plot(epoch_ls,
                                       VGG16Net_001_acc,
                                       color = 'black',
                                       linewidth = 1.5,
                                       linestyle = ":",
                                       markersize = 4,
                                       marker = 'x',
                                       label = 'VGG16Net_acc')
    
    # 三种模型在同一学习率条件下对应loss的折线图
    ax_LeNet_AlexNet_VGG16Net_loss_, = ax_LeNet_AlexNet_VGG16Net_loss.plot(epoch_ls,
                                       LeNet_001_loss,
                                       color = 'red',
                                       linewidth = 1.5,
                                       linestyle = ":",
                                       markersize = 4,
                                       marker = 'x',
                                       label = 'LeNet_loss')
    
    ax_LeNet_AlexNet_VGG16Net_loss_, = ax_LeNet_AlexNet_VGG16Net_loss.plot(epoch_ls,
                                       AlexNet_Adam_loss,
                                       color = 'blue',
                                       linewidth = 1.5,
                                       linestyle = ":",
                                       markersize = 4,
                                       marker = 'x',
                                       label = 'AlexNet_loss')
    
    ax_LeNet_AlexNet_VGG16Net_loss_, = ax_LeNet_AlexNet_VGG16Net_loss.plot(epoch_ls,
                                       VGG16Net_001_loss,
                                       color = 'black',
                                       linewidth = 1.5,
                                       linestyle = ":",
                                       markersize = 4,
                                       marker = 'x',
                                       label = 'VGG16Net_loss')
    
    ax_LeNet_AlexNet_VGG16Net_acc.legend(frameon = False)
    ax_LeNet_AlexNet_VGG16Net_loss.legend(frameon = False)

    # 坐标轴标注
    ax_LeNet_AlexNet_VGG16Net_acc.set_xlabel('训练轮数epoch', 
                            fontsize=12,
                            fontfamily = 'SimHei',
                            fontstyle='italic')
    
    ax_LeNet_AlexNet_VGG16Net_acc.set_ylabel('准确率acc', 
                            fontsize=12,
                            fontstyle='oblique')
    
    ax_LeNet_AlexNet_VGG16Net_loss.set_xlabel('训练轮数epoch', 
                            fontsize=12,
                            fontfamily = 'SimHei',
                            fontstyle='italic')
    
    ax_LeNet_AlexNet_VGG16Net_loss.set_ylabel('损失率loss', 
                            fontsize=12,
                            fontstyle='oblique')
    
    # 展示图片
    plt.show()


if __name__ == '__main__':
    # 选择画图的种类
    
    print('Please select the type of drawing(LeNet,AlexNet,VGG16Net,All):')
    input1 = input('')
    if input1 == 'LeNet':
        draw_LeNet_loss_acc()
        
    elif input1 == 'AlexNet':
        draw_AlexNet_loss_acc()
        
    elif input1 == 'VGG16Net':
        draw_VGG16Net_loss_acc()
        
    elif input1 == 'All':
        draw_AlexNet_loss_acc()
        draw_AlexNet_loss_acc()
        draw_VGG16Net_loss_acc()