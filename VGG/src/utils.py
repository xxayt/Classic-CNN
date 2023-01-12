# 自定义工具包
import torch
import numpy as np
import os
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import host_subplot
from src.config import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# 计算并储存更新loss和acc
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n=1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def Paint_Train(iter, loss, acc, title):
    host = host_subplot(111)
    plt.subplots_adjust(right=0.8) # ajust the right boundary of the plot window
    par1 = host.twinx()
    # set labels
    host.set_xlabel("iterations")
    host.set_ylabel("log loss")
    par1.set_ylabel("train accuracy")
    
    # plot curves
    p1 = host.plot(iter, loss, label="Loss", color = 'r', linewidth= 0.5)
    p2 = par1.plot(iter, acc, label="Train Accuracy", color = 'b', linewidth= 0.5)
    
    host.legend(loc=5)
    
    # 设置坐标轴
    host.axis["left"].label.set_color('red')
    par1.axis["right"].label.set_color('blue')
    # 设置标题
    plt.title(title)
    plt.savefig(model_path + "/" + title + ".png")  #保存图片
    plt.show()

def write2file(iter, loss, acc, title):
    f=open(model_path + "/" + title + ".txt", "w")
    for i in range(len(iter)):
        f.write(str(iter[i]) + ' ')
        f.write(str(loss[i]) + ' ')
        f.write(str(acc[i]) + '\n')
    f.close()

# Paint_Train(list(range(1, 6)), [0.12, 0.2, 0.3, 0.4, 0.5], [0.12, 0.1, 0.2, 0.6, 0.57], title="train")
# write2file(list(range(1, 6)), [0.12, 0.2, 0.3, 0.4, 0.5], [0.12, 0.1, 0.2, 0.6, 0.57], title="train")