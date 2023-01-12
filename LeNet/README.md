

# LeNet讲解

## 1. 文件夹介绍

```python
PS D:\2Codefield\VS_code\python\Learn_Base> tree /F
卷 Data 的文件夹 PATH 列表
卷序列号为 C2BD-263D
D:.
│  LeNet Structure.png  # LeNet原始结构图
│  main.py  # 主文件，训练时只需要运行此文件即可（在修改各保存路径后）
│  README.md
│  realtest.py  # 测试文件，预测画板画出的真实手写数字
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
│  │  dataloader.py  # 调用，预处理及加载MNIST数据
│  │
│  └─__pycache__
│          dataloader.cpython-38.pyc
│
└─LeNetModel  # 保存训练完模型参数和优化器参数
	better.png
        better.txt  # LeNet5_Better框架训练得到的loss和acc
        model_better.pkl
        model_origin.pkl
        optimizer_better.pkl
        optimizer_origin.pkl
        origin.png
        origin.txt  # LeNet5_Origin框架训练得到的loss和acc
```

## 2. 讲解LeNet

-   **原始LeNet5**：根据Yann Lecun在1998年发布的论文[《Gradient-based learning applied to document recognition》](https://readpaper.com/paper/2112796928)中的框架

    ![LeNet原始结构](https://github.com/xxayt/Classic-CNN/blob/main/LeNet/LeNetOrigin%20Structure.png)

    -   这篇文章介绍的网络框架，主要包含了 $C1,S2,C3,S4,C5,F6,OUT$ 这7层，这篇[文章](https://blog.csdn.net/hgnuxc_1993/article/details/115566799)比较清楚的讲解了框架。因此 `model.py` 文件中的 `LeNet5_Origin` 网络层结构就是按照原始文章搭建的。

        ```python
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
                    # [bs,16,4,4] -> [bs,120,1,1]
                    nn.Linear(16*4*4, 120),
                    nn.Sigmoid(),
                    # [bs,120,1,1] -> [bs,84,1,1]
                    nn.Linear(120, 84),
                    nn.Sigmoid(),
                    # [bs,84,1,1] -> [bs,10,1,1]
                    nn.Linear(84, 10),  # 10个手写数字对应10个输出
                )
        
            def forward(self, x):
                x = self.Convolution1(x)
                x = self.Convolution2(x)
                # x = x.view(x.size()[0], -1)  # 变形
                x = x.view(-1, 16*4*4)
                x = self.Full_Connected(x)
                return x
        ```

    -   我在经典数字集MNIST上进行训练，发现效果不是很好：

        ![原始框架效果](https://github.com/xxayt/Classic-CNN/blob/main/LeNet/LeNetModel/origin.png)

        ```python
        Train Epoch=10, TrainLoss=0.05923, TrainAcc=0.98195
        Train Epoch=10, TestAcc=0.98230
        ```

        虽然在四五个epoch后测试集准确率也达到了98% ，但是在真实画图手写的测试中，几乎无法预测正确。

        ```python
        tensor([[ 2.9016, -3.2674, -1.9055, -3.5084,  1.1511,  2.7620,  4.3779, -3.6428, 2.5368, -1.1161]], grad_fn=<AddmmBackward0>)
        tensor([4.3779])
        the result is: 6  # 实际图片是5，预测成了6
        ```

        我认为原因有以下几点：

        -   网络框架原因：
            1.  卷积层输出通道较少(特征提取不足)
            2.  sigmoid激活函数非线性时效果不好
            3.  三层FC层导致参数过多(模型太复杂导致冗余信息过多)
        -   数据及内部原因：
            1.  MNIST数据集中训练集和测试集较为相近，使得测试acc较高，且西方人手写数字和我们差异较大，使得我画图手写测试的效果不好
            2.  运用pytorch框架，虽然网络层框架与原文一致，但是卷积的选取完全不同（例：原文C3层中将6通道变为16通道的函数对照表与 `nn.Conv2d(6, 16, 5, 1, 0)` 一定不同）

-   **改进LeNet**：根据上述网络框架原因，我将卷积层的输出通道数增加，激活函数改为 `ReLU()`，并只使用一层FC。

    ```python
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
                # [bs,20,4,4] -> [bs,10,1,1]
                nn.Linear(20*4*4, 10)
            )
    
        def forward(self, x):
            x = self.Convolution1(x)
            x = self.Convolution2(x)
            # x = x.view(x.size()[0], -1)  # 变形
            x = x.view(-1, 20*4*4)
            x = self.Full_Connected(x)
            return x
    ```

    -   效果明显提升：

        ![改进框架效果](https://github.com/xxayt/Classic-CNN/blob/main/LeNet/LeNetModel/better.png)

        ```python
        Train Epoch=10, TrainLoss=0.05140, TrainAcc=0.98588
        Train Epoch=10, TestAcc=0.98730
        ```

        在第一个epoch上就可以达到97% 的准确率，并且在我的手写数据集上几乎可以完全预测准确

        ```python
        tensor([[  3.7950, -10.5007,  -2.7558,   5.1051,  -3.3835,  19.5448,  -1.4199, -18.6940,   8.4183,   3.5093]], grad_fn=<AddmmBackward0>)     
        tensor([19.5448])
        the result is: 5
        ```

        
