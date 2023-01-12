import torch
from torch import nn, optim
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(AlexNet, self).__init__()
        # 提取特征层
        self.Features = nn.Sequential(
            # C1: [bs,3,224,224] -> [bs,48,55,55]
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            # S2: [bs,48,55,55] -> [bs,48,27,27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # C3: [bs,48,27,27] -> [bs,128,27,27]
            nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            # S4: [bs,128,27,27] -> [bs,128,13,13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # C5: [bs,128,13,13] -> [bs,192,13,13]
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # C6: [bs,192,13,13] -> [bs,192,13,13]
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # C7: [bs,192,13,13] -> [bs,128,13,13]
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # S8: [bs,128,13,13] -> [bs,128,6,6]
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 分类层
        self.Classifier = nn.Sequential(
            # FC9: [bs,4608,1,1] -> [bs,2048,1,1]
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6, 2048),
            nn.ReLU(),
            # FC10: [bs,2048,1,1] -> [bs,2048,1,1]
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            # FC11: [bs,2048,1,1] -> [bs,num_classes,1,1]
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self.init_bias()
    
    def init_bias(self):
        for layer in self.Features:
            if isinstance(layer, nn.Conv2d):  # 判断是否为卷积层
                nn.init.normal_(layer.weight, mean=0, std=0.01)  # std=0.01的零均值高斯分布
                nn.init.constant_(layer.bias, 0)
        # 原始论文中,对第2,4,5层卷积层初始化常数1
        nn.init.constant_(self.Features[4].bias, 1)
        nn.init.constant_(self.Features[10].bias, 1)
        nn.init.constant_(self.Features[12].bias, 1)

    def forward(self, x):
        # [bs,3,224,224] -> [bs,128,6,6]
        x = self.Features(x)
        # [bs,128,6,6] -> [bs,4608,1,1]
        x = x.view(x.size()[0], -1)
        # [bs,4608,1,1] -> [bs,num_classes,1,1]
        x = self.Classifier(x)
        return x