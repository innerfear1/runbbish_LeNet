import torch
import torch.nn as nn
import torchvision.transforms as transformers
import json
from model.VGG16Net import VGG16Net


from PIL import Image, ImageTk # 导入图像处理函数库
import tkinter as tk
from tkinter import filedialog   #导入文件对话框函数库



# 创建窗口 设定大小并命名
window = tk.Tk()
window.title('图像显示界面')
window.geometry('600x500')
global img_png           # 定义全局变量 图像的
var = tk.StringVar()    # 这时文字变量储存器
pre = tk.StringVar()    # 定义预测的变量存储器



# 创建打开图像和显示图像函数
def Open_Img():
    global img_png
    global file_path
    OpenFile = tk.Tk() #创建新窗口
    OpenFile.withdraw()
    file_path = filedialog.askopenfilename()
    Img = Image.open(file_path)
    img_png = ImageTk.PhotoImage(Img)
    var.set('已打开')
    label_Img = tk.Label(window, image=img_png)
    label_Img.pack()
    var.set('已显示')




def predict():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using {} train model'.format(device))


    transformer = transformers.Compose([
        transformers.Resize((224,224)),
        transformers.ToTensor(),
        transformers.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    global img_png

    img = Image.open(file_path)
    img = transformer(img) #[3,224,224]
    img = torch.unsqueeze(img,dim=0)#[1,3,224,224]

    json_file = './index_class.json'
    index_class_json = open(json_file,'r')
    index_class_dict = json.load(index_class_json)
    print(index_class_dict)


    model = VGG16Net().to(device)
    weight_path = 'pth/VGG16Net_0.001.pth'
    model.load_state_dict(torch.load(weight_path))

    with torch.no_grad():
        output = model(img)
        output_class = torch.max(output,dim=1)[1]
        print(index_class_dict[str(output_class.item())])

    pre.set('{}'.format(index_class_dict[str(output_class.item())]))




# 创建文本窗口，显示当前操作状态
Label_Show = tk.Label(window,
    textvariable=var,   # 使用 textvariable 替换 text, 因为这个可以变化
    bg='white', font=('Arial', 12), width=15, height=2)
Label_Show.pack()

# 创建打开图像按钮
btn_Open = tk.Button(window,
    text='打开图像',      # 显示在按钮上的文字
    width=15, height=2,
    command=Open_Img)     # 点击按钮式执行的命令
btn_Open.pack()    # 按钮位置

# 创建识别图像按钮
btn_Show = tk.Button(window,
    text='识别图像',      # 显示在按钮上的文字
    width=15, height=2,
    command=predict)     # 点击按钮式执行的命令
btn_Show.pack()    # 按钮位置

# 创建结果界面
Predict_Show = tk.Label(window,
                        textvariable=pre,
                        bg='white',
                        font=('Arial', 12),
                        width=20,
                        height=2)
Predict_Show.pack()

# 运行整体窗口
window.mainloop()
