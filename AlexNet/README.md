[toc]



# 深度卷积神经网络AlexNet讲解

## 1. 文件夹介绍

```python
PS D:\2Codefield\VS_code\python\Learn_Base\CNN\AlexNet> tree /F
卷 Data 的文件夹 PATH 列表
卷序列号为 C2BD-263D
D:.
│  AlexNet Structure.png  # AlexNet原始结构图
│  main.py  # 主文件，训练时只需要运行此文件即可（在修改各保存路径后）
│  README.md
│  realtest.py  # 测试文件，预测网络下载的图片
│  
├─AlexNetModel  # 保存训练完模型参数和优化器参数
│      AlexNet.png
│      AlexNet.txt
│      AlexNet_print.txt
│      model90.pkl
│      optimizer90.pkl
│
├─core
│  │  config.py  # 设定超参数及特定保存路径
│  │  model.py  # 模型主题框架，包含两个模型，可分别调用
│  │  utils.py  # 自定义工具包，主要包括保存和画图函数
│  │
│  └─__pycache__
│          config.cpython-38.pyc
│          model.cpython-38.pyc
│          utils.cpython-38.pyc
│
├─dataload
│  │  dataloader.py  # 调用，预处理及加载Flower数据
│  │
│  └─__pycache__
│          dataloader.cpython-38.pyc
│
└─__pycache__
        main.cpython-38.pyc
```



## 2. 讲解AlexNet

-   **网络要点**：作为深度网络的分水岭，AlexNet是根据Alex Krizhevsky和其导师Hinton设计的，并在2012年的ImageNet竞赛中获得冠军，因此本代码力争还原其文章[《ImageNet Classification with Deep Convolutional Neural Networks》](https://readpaper.com/paper/2163605009)中设计的网络框架，其中主要包含了五大Tricks，其中ReLU和Dropout均为后面数年的研究提供了相当大的启发。

    ![LeNet原始结构](https://github.com/xxayt/Classic-CNN/blob/main/AlexNet/AlexNet%20Structure.png)

    1.  选取非线性非饱和的**ReLU**函数代替Sigmoid，Tanh函数等作为激活函数
    2.  采用局部响应归一化（**LRN**）形成某种形式的横向抑制，从而提高网络的泛化能力。但此部分在后来发现可通过调参弥补，用处不大
    3.  由于硬件技术的限制，采用**双GPU**并行训练，只在第三层卷积 时交换数据
    4.  **重叠最大池化**，采用池化步长小于核尺寸的方法，避免过拟合
    5.  使用**Dropout**随机互留一部分神经元，避免过拟合

-   这篇文章介绍的网络框架，主要包含了五层卷积，三层池化，三层全连接这11层。在 `model.py` 文件中的 `AlexNet` 网络层结构还根据原模型，加入了LRN结构和初始化bias的操作。

    ```python
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
    ```

-   我在五分类花朵数据集[Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)上进行的测试，简要下载方式可以参考[这里](https://github.com/xxayt/deep-learning-for-image-processing/tree/master/data_set)，测试数据准确率达到了75%，相比于其他基础模型，效果较好。


    ![原始框架效果](https://github.com/xxayt/Classic-CNN/blob/main/AlexNet/AlexNetModel/AlexNet.png)
    
    ```python
    Train Epoch=90, TrainLoss=0.64415, TrainAcc=0.75923
    Train Epoch=90, TestAcc=0.75000
    ```
    
    在我通过网络随意找的图片花朵数据上，几乎可以准确预测
    
    ```python
    tensor([[ -2.5933,  -8.0850, -19.1720,  29.6008,  -7.5790]],
           grad_fn=<AddmmBackward0>)
    tensor([29.6008])
    the result is: sunflower向日葵
    ```
    
    
