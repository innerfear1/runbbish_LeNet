import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import  summary

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # input为224*224*3 96个卷积核11*11*3 
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=96,
                               kernel_size=11,
                               stride=4,
                               padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        # input为27*27*96 256个卷积核5*5*96
        self.conv2 = nn.Conv2d(in_channels=96,
                               out_channels=256,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        # input为13*13*256 384个卷积核3*3*256
        self.conv3 = nn.Conv2d(in_channels=256,
                               out_channels=384,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        # input为13*13*384 384个卷积核3*3*384
        self.conv4 = nn.Conv2d(in_channels=384,
                               out_channels=384,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        # input为13*13*384 256个卷积核3*3*384
        self.conv5 = nn.Conv2d(in_channels=384,
                               out_channels=256,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3,stride=2)
        # input为6*6*256 
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc2_drop = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, 4)


    def forward(self,x):
        x = x.cuda()
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = F.relu(self.conv5(x))
        x = self.pool5(x)

        x = torch.flatten(x,start_dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)

        return x