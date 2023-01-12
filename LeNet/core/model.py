import torch
from torch import nn, optim
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# 原始LeNet-5模型
class LeNet5_Origin(nn.Module):
    def __init__(self):
        super(LeNet5_Origin, self).__init__()
        self.Convolution1 = nn.Sequential(
            # [bs,1,28,28] -> [bs,6,24,24]
            nn.Conv2d(1, 6, 5, 1, 0),
            nn.Sigmoid(),
            # [bs,6,28,28] -> [bs,6,12,12]
            nn.MaxPool2d(2, 2)
        )
        self.Convolution2 = nn.Sequential(
            # [bs,6,12,12] -> [bs,16,8,8]
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.Sigmoid(),
            # [bs,16,8,8] -> [bs,16,4,4]
            nn.MaxPool2d(2, 2)
        )
        self.Full_Connected = nn.Sequential(
            # [bs,256,1,1] -> [bs,120,1,1]
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            # [bs,120,1,1] -> [bs,84,1,1]
            nn.Linear(120, 84),
            nn.Sigmoid(),
            # [bs,84,1,1] -> [bs,10,1,1]
            nn.Linear(84, 10),  # 10个手写数字对应10个输出
        )

    def forward(self, x):
        # [bs,1,28,28] -> [bs,6,12,12]
        x = self.Convolution1(x)
        # [bs,6,12,12] -> [bs,16,4,4]
        x = self.Convolution2(x)
        # [bs,16,4,4] -> [bs,256,1,1]
        # x = x.view(x.size()[0], -1)  # 变形
        x = x.view(-1, 16*4*4)
        # [bs,256,1,1] -> [bs,10,1,1]
        x = self.Full_Connected(x)
        return x


# 改进LeNet-5模型
class LeNet5_Better(nn.Module):
    def __init__(self):
        super(LeNet5_Better, self).__init__()
        self.Convolution1 = nn.Sequential(
            # [bs,1,28,28] -> [bs,10,24,24]
            nn.Conv2d(1, 10, 5, 1, 0),
            nn.ReLU(),
            # [bs,10,24,24] -> [bs,10,12,12]
            nn.MaxPool2d(2, 2)
        )
        self.Convolution2 = nn.Sequential(
            # [bs,10,12,12] -> [bs,20,8,8]
            nn.Conv2d(10, 20, 5, 1, 0),
            nn.ReLU(),
            # [bs,20,8,8] -> [bs,20,4,4]
            nn.MaxPool2d(2, 2)
        )
        self.Full_Connected = nn.Sequential(
            # [bs,320,1,1] -> [bs,10,1,1]
            nn.Linear(20*4*4, 10)
        )

    def forward(self, x):
        # [bs,1,28,28] -> [bs,10,12,12]
        x = self.Convolution1(x)
        # [bs,10,12,12] -> [bs,20,4,4]
        x = self.Convolution2(x)
        # [bs,20,4,4] -> [bs,320,1,1]
        # x = x.view(x.size()[0], -1)  # 变形
        x = x.view(-1, 20*4*4)
        # [bs,320,1,1] -> [bs,10,1,1]
        x = self.Full_Connected(x)
        return x
