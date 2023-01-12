import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from core.config import *
import os

data_path = "D:/2Codefield/VS_code/python/Learn_Base/CNN/DATA/Flower_data"
CLASSES = ('daisy雏菊', 'dandelion蒲公英', 'roses玫瑰', 'sunflower向日葵', 'tulips郁金香')

#预处理,自定义的train_transformer
transforms={
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  #转为tensor, 范围改为0~1
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #归一化
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

# 加载数据
def load_data(data_path, batch_size=BATCH_SIZE):
    # 返回transformer后的图片和对应标签(类别)
    train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=transforms["train"])
    test_data = datasets.ImageFolder(root=data_path + "/test", transform=transforms["test"])
    '''
    print("using {} images for training, {} images for testing.".format(len(train_data), len(test_data)))
    print(train_data.class_to_idx)
    # 在转为tensor前,为PIL文件,可显示图片.即删除transform=transforms后才可显示
    print(test_data[4])
    # test_data[4][0].show()  #展示图片
    '''
    # 数据加载
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)
    num_iter = len(train_loader)  # 3306 / 64 = 51.6 -> 52
    return train_loader, test_loader, num_iter


if __name__ == '__main__':
    train_loader, test_loader = load_data(data_path)
