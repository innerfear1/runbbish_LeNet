import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import  summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        # （224，224，3）    （5，5，6）
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        # （）     （5，5，16）
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        # （53*53*16）
        self.fc1 = nn.Linear(16*53*53,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,4)


    def forward(self,x):
        x = x.cuda()
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x,start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

