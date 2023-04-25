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

# 画出单模型acc和loss曲线图
def draw_single_model_acc_loss():
    '''
    # 设置画布大小
    fig, ax1 = plt.subplots(figsize = (7, 5))
    # 在画布中画出LeNet损失曲线
    LeNet_001loss_plt, = ax1.plot(epoch_ls, 
                             LeNet_001_loss, 
                             color = 'red', 
                             linewidth = 1.5, 
                             linestyle = ":", 
                             markersize = 4,
                             marker = 'x',
                             label = 'LeNet损失')
    # 在画布中画出LeNet的准确率曲线
    LeNet_001acc_plt, = ax1.plot(epoch_ls,
                            LeNet_001_acc,
                            color = 'orange',
                            linewidth = 1.5,
                            linestyle = ":",
                            markersize = 4,
                            marker = 'x',
                            label = 'LeNet准确率')
    # 
    ax1.legend(frameon = False)
    # x,y轴对应的标签和字体设置
    ax1.set_xlabel('训练轮数epoch', fontsize=12,fontfamily = 'SimHei',fontstyle='italic')
    ax1.set_ylabel('损失值loss/准确率acc', fontsize=12,fontstyle='oblique')
    # 设置图片标题
    ax1.set_title('LeNet', fontsize = 18)
    # 展示画出的图
    plt.show()

    # 设置画布大小
    fig, ax2 = plt.subplots(figsize = (7, 5))
    # 在画布中画出LeNet损失曲线
    LeNet_01loss_plt, = ax2.plot(epoch_ls, 
                             LeNet_01_loss, 
                             color = 'red', 
                             linewidth = 1.5, 
                             linestyle = ":", 
                             markersize = 4,
                             marker = 'x',
                             label = 'LeNet损失')
    # 在画布中画出LeNet的准确率曲线
    LeNet_01acc_plt, = ax2.plot(epoch_ls,
                            LeNet_01_acc,
                            color = 'orange',
                            linewidth = 1.5,
                            linestyle = ":",
                            markersize = 4,
                            marker = 'x',
                            label = 'LeNet准确率')
    # 
    ax2.legend(frameon = False)
    # x,y轴对应的标签和字体设置
    ax2.set_xlabel('训练轮数epoch', fontsize=12,fontfamily = 'SimHei',fontstyle='italic')
    ax2.set_ylabel('损失值loss/准确率acc', fontsize=12,fontstyle='oblique')
    # 设置图片标题
    ax2.set_title('LeNet', fontsize = 18)
    # 展示画出的图
    plt.show()
    '''

    # 设置画布大小
    fig, ax3 = plt.subplots(figsize = (7, 5))
    # 在画布中画出LeNet损失曲线
    ANet_001loss_plt, = ax3.plot(epoch_ls, 
                             AlexNet_Adam_loss, 
                             color = 'red', 
                             linewidth = 1.5, 
                             linestyle = ":", 
                             markersize = 4,
                             marker = 'x',
                             label = 'AlexNet损失')
    # 在画布中画出LeNet的准确率曲线
    LeNet_001acc_plt, = ax3.plot(epoch_ls,
                            AlexNet_Adam_acc,
                            color = 'orange',
                            linewidth = 1.5,
                            linestyle = ":",
                            markersize = 4,
                            marker = 'x',
                            label = 'AlexNet准确率')
    # 
    ax3.legend(frameon = False)
    # x,y轴对应的标签和字体设置
    ax3.set_xlabel('训练轮数epoch', fontsize=12,fontfamily = 'SimHei',fontstyle='italic')
    ax3.set_ylabel('损失值loss/准确率acc', fontsize=12,fontstyle='oblique')
    # 设置图片标题
    ax3.set_title('AlexNet-Adam', fontsize = 18)
    # 展示画出的图
    plt.show()

    # 设置画布大小
    fig, ax4 = plt.subplots(figsize = (7, 5))
    # 在画布中画出LeNet损失曲线
    ANet_01loss_plt, = ax4.plot(epoch_ls, 
                             AlexNet_SGD_loss, 
                             color = 'red', 
                             linewidth = 1.5, 
                             linestyle = ":", 
                             markersize = 4,
                             marker = 'x',
                             label = 'AlexNet损失')
    # 在画布中画出LeNet的准确率曲线
    LeNet_01acc_plt, = ax4.plot(epoch_ls,
                            AlexNet_SGD_acc,
                            color = 'orange',
                            linewidth = 1.5,
                            linestyle = ":",
                            markersize = 4,
                            marker = 'x',
                            label = 'AlexNet准确率')
    # 
    ax4.legend(frameon = False)
    # x,y轴对应的标签和字体设置
    ax4.set_xlabel('训练轮数epoch', fontsize=12,fontfamily = 'SimHei',fontstyle='italic')
    ax4.set_ylabel('损失值loss/准确率acc', fontsize=12,fontstyle='oblique')
    # 设置图片标题
    ax4.set_title('AlexNet-SGD', fontsize = 18)
    # 展示画出的图
    plt.show()

    # 设置画布大小
    fig, ax5 = plt.subplots(figsize = (7, 5))
    # 在画布中画出LeNet损失曲线
    VNet_001loss_plt, = ax5.plot(epoch_ls, 
                             VGG16Net_001_loss, 
                             color = 'red', 
                             linewidth = 1.5, 
                             linestyle = ":", 
                             markersize = 4,
                             marker = 'x',
                             label = 'VGGNet损失')
    # 在画布中画出LeNet的准确率曲线
    LeNet_001acc_plt, = ax5.plot(epoch_ls,
                            VGG16Net_001_acc,
                            color = 'orange',
                            linewidth = 1.5,
                            linestyle = ":",
                            markersize = 4,
                            marker = 'x',
                            label = 'VGGNet准确率')
    # 
    ax5.legend(frameon = False)
    # x,y轴对应的标签和字体设置
    ax5.set_xlabel('训练轮数epoch', fontsize=12,fontfamily = 'SimHei',fontstyle='italic')
    ax5.set_ylabel('损失值loss/准确率acc', fontsize=12,fontstyle='oblique')
    # 设置图片标题
    ax5.set_title('VGGNet-0.001', fontsize = 18)
    # 展示画出的图
    plt.show()

    # 设置画布大小
    fig, ax6 = plt.subplots(figsize = (7, 5))
    # 在画布中画出LeNet损失曲线
    VNet_01loss_plt, = ax6.plot(epoch_ls, 
                             VGG16Net_01_loss, 
                             color = 'red', 
                             linewidth = 1.5, 
                             linestyle = ":", 
                             markersize = 4,
                             marker = 'x',
                             label = 'VGGNet损失')
    # 在画布中画出LeNet的准确率曲线
    VNet_01acc_plt, = ax6.plot(epoch_ls,
                            VGG16Net_01_acc,
                            color = 'orange',
                            linewidth = 1.5,
                            linestyle = ":",
                            markersize = 4,
                            marker = 'x',
                            label = 'VGGNet准确率')
    # 
    ax6.legend(frameon = False)
    # x,y轴对应的标签和字体设置
    ax6.set_xlabel('训练轮数epoch', fontsize=12,fontfamily = 'SimHei',fontstyle='italic')
    ax6.set_ylabel('损失值loss/准确率acc', fontsize=12,fontstyle='oblique')
    # 设置图片标题
    ax6.set_title('VGGNet-0.01', fontsize = 18)
    # 展示画出的图
    plt.show()




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
                                label = 'LeNet_lr=0.01_acc')
    
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
                                label = 'AlexNet_SGD_acc')
    
    # optim=Adam时的loss折线图画出来
    LeNet_loss_vs_Adam, = ax_ALexNet_loss.plot(epoch_ls,
                                AlexNet_Adam_loss,
                                color = 'red',
                                linewidth = 1.5,
                                linestyle = ":",
                                markersize = 4,
                                marker = 'x',
                                label = 'ALexNet_Adam_loss')
    
    # optim=SGD时的loss折线图画出来
    LeNet_loss_vs_SGD, = ax_ALexNet_loss.plot(epoch_ls,
                                AlexNet_SGD_loss,
                                color = 'black',
                                linewidth = 1.5,
                                linestyle = ":",
                                markersize = 4,
                                marker = 'x',
                                label = 'AlexNet_SGD_loss')
    
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
        # draw_LeNet_loss_acc()
        # draw_AlexNet_loss_acc()
        # draw_VGG16Net_loss_acc()
        # LeNet_AlexNet_VGG16Net_acc_loss()
        draw_single_model_acc_loss()

