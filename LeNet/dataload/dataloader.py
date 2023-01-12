import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from core.config import *
import os

#预处理,自定义的train_transformer
transforms=transforms.Compose([
    transforms.ToTensor(),  #转为tensor, 范围改为0~1
    transforms.Normalize((0.1307,), (0.3081,))  #归一化
])


# 加载数据
def load_data(data_path, batch_size=BATCH_SIZE):
    # 返回transformer后的图片和对应标签(类别)
    train_data = MNIST(root=data_path, train=True, download=False, transform=transforms)
    test_data = MNIST(root=data_path, train=False, download=False, transform=transforms)
    
    # 在转为tensor前,为PIL文件,可显示图片.即删除transform=transforms后才可显示
    
    # print(test_data[4])
    # test_data[4][0].show()  #展示图片
    
    # 数据加载
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    # print(len(train_loader))  # 60000 / 64 = 937.5 -> 938
    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = load_data(data_path)
