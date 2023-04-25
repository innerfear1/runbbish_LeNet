import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def Draw_a_bar_chart():
    # 生成数据
    x = np.arange(1, 5)
    y = [3051,7226,1487,1035]
    # 生成图形
    plt.bar(x, y, width=0.5, color='blue', tick_label=['厨余垃圾', '可回收垃圾', '其他垃圾', '有害垃圾'])
    # 添加标题
    plt.title('垃圾分类数据集')
    # 添加横轴标签
    plt.xlabel('垃圾类型')
    # 添加纵轴标签
    plt.ylabel('数量')
    # 每列柱子上方添加数值标签
    for a, b in zip(x, y):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
    
    # 显示图形
    plt.show()


if __name__ == '__main__':
    Draw_a_bar_chart()