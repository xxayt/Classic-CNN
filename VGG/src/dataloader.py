# http://www.cs.toronto.edu/~kriz/cifar.html
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from src.config import *
import os

data_path = "D:/2Codefield/VS_code/python/Learn_Base/CNN/DATA/CIFAR10"
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#预处理,自定义的train_transformer
transform=transforms.Compose([
    transforms.RandomResizedCrop(32),  # 随机裁剪为不同大小，默认0.08~1.0，期望输出大小32x32
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像，默认概率0.5
    transforms.ToTensor(),  # 转为tensor, 范围改为0~1
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# 加载数据
def load_data(data_path, batch_size=BATCH_SIZE):
    # CIFAR10数据集大小为32x32
    train_data = CIFAR10(root=data_path, train=True, download=False, transform=transform)
    test_data = CIFAR10(root=data_path, train=False, download=False, transform=transform)
    
    # 在转为tensor前,为PIL文件,可显示图片.即删除transform=transforms后才可显示
    # print(test_data[10])
    # test_data[10][0].show()  #展示图片

    # 数据加载
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    num_iter = len(train_loader)  # 50000 / 64 = 781.3 -> 782
    return train_loader, test_loader, num_iter


if __name__ == '__main__':
    train_loader, test_loader = load_data(data_path)
