import torch
from torch import nn, optim
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class vgg(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=False):
        super(vgg, self).__init__()
        # 提取特征层
        self.Features = features
        # 分类层
        self.Classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*1*1, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self.init_bias()
    # 初始化权重
    def init_bias(self):
        # 便利每一个子模块
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # 若遍历到卷积层
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)  # 初始化卷积核权重
                if m.bias is not None:  # 若采用偏置，将偏置初始化为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    # 前向传播
    def forward(self, x):
        # N x 3 x 32 x 32
        x = self.Features(x)
        # N x 512 x 1 x 1
        x = torch.flatten(x, start_dim=1)
        # N x 512*1*1
        x = self.Classifier(x)
        return x


def VGG(model_name, **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]
    model = vgg(make_features(cfg), **kwargs)
    return model


def make_features(cfg: list):
    layers, in_channels = [], 3
    for out_channels in cfg:
        if out_channels == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
            layers += [nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)  # 非关键字传入