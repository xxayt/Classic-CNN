import torch
import os
from core.model import *
from core.dataloader import *
from core.config import *
from PIL import Image
import numpy as np

pic_path = "D:/2Codefield/VS_code/python/Learn_Base/CNN/DATA/Flower_data"

model = AlexNet(num_classes=5)
# 加载保存模型的参数
if os.path.exists(model_path + "/model90.pkl"):
    state_dict = torch.load(model_path + "/model90.pkl")
    model.load_state_dict(state_dict)

if __name__ == '__main__':
    # 手写测试
    img = Image.open(pic_path + "/向日葵.png")
    # img.show()
    img = transforms["test"](img)  # 利用上面transform进行预处理
    img = torch.unsqueeze(img, dim=0)   # 扩展第一维为batch_size
    # print(img)
    
    img.view(-1, 50176)
    y_predict = model(img)
    a, result = torch.max(y_predict.data, dim=1)
    print(y_predict)
    print(a)
    print("the result is:", CLASSES[result.item()])