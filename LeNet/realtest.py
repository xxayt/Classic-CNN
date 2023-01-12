import torch
import os
from core.model import *
from dataload.dataloader import *
from core.config import *
from PIL import Image
import numpy as np

pic_path = "D:/2Codefield/VS_code/python/Learn_Base/CNN/DATA/num.png"

model = LeNet5_Origin()
# 加载保存模型的参数
if os.path.exists(model_path + "/model_origin.pkl"):
    state_dict = torch.load(model_path + "/model_origin.pkl")
    model.load_state_dict(state_dict)

# model = LeNet5_Better()
# # 加载保存模型的参数
# if os.path.exists(model_path + "/model_better.pkl"):
#     state_dict = torch.load(model_path + "/model_better.pkl")
#     model.load_state_dict(state_dict)

if __name__ == '__main__':
    

    # 手写测试
    img = Image.open(pic_path).convert("L")
    img.show()
    img = transforms(img)  # 利用上面transform进行预处理
    # print(img)
    
    img.view(-1, 784)
    y_predict = model(img)
    a, result = torch.max(y_predict.data, dim=1)
    print(y_predict)
    print(a)
    print("the result is:", result.item())