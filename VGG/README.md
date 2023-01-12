
[toc]

# VGG讲解

|     参数     |               我的代码               |
| :----------: | :----------------------------------: |
|   模型名称   |                 VGG                  |
|     资源     |              RTX 2080Ti              |
|    数据集    |    CIFAR10（原论文：ILSVRC-2012）    |
| 输入图片大小 |     32x32x3（原论文：224x224x3）     |
|   训练参数   | epoch=50, batch_size = 64, lr=0.0002 |
|    优化器    |                 Adam                 |
|   损失函数   |      CrossEntropyLoss交叉熵损失      |

-   **模型框架**：

<img src="D:\2Codefield\VS_code\python\Learn_Base\CNN\VGG\VGG Structure.png" alt="VGG Structure" style="zoom:50%;" />

-   **主要特点**：较大的感受野可用过较小的过滤器堆叠来实现（利用 $2$ 个 $3\times3$ 代替一个 $5\times5$ ，利用 $3$ 个 $3\times3$ 代替一个 $7\times7$）
    -   **好处**：
        1.  增加过滤器，可引入更多激活函数，使输出数据更加离散化，更容易被分类
        2.  减少参数量，若一个 $7\times7$ 过滤器含有 $49C^2$ 个参数量($C$为数据深度)，则三个 $3\times3$ 过滤器只有 $27C^2$ 个参数，减少了 $45\%$ 
-   **其他贡献**：

# 我的效果

>    CIFAR10数据集作为一个十分类数据集，随机预测约为10%的准确率

|     模型     | vgg11  | vgg13  | vgg16  |
| :----------: | :----: | :----: | :----: |
| 测试集准确率 | 0.7595 | 0.7826 | 0.7734 |

-   **VGG11**：发现在epoch43左右，测试集准确率已到达最高，后期训练集准确率增高，大概率出现了过拟合

    ```python
    Train Epoch=50, TrainLoss=0.54582, TrainAcc=0.81460
    Train Epoch=50, TestAcc=0.75950
    Best now is epoch=50, test acc=0.7595
    ```

    <img src="D:\2Codefield\VS_code\python\Learn_Base\CNN\VGG\VGGModel\vgg11.png" alt="vgg11" style="zoom:50%;" />

-   **VGG13**：效果最好，大概率是因为次数经济过于简单

    ```python
    Train Epoch=49, TrainLoss=0.52326, TrainAcc=0.82326
    Train Epoch=49, TestAcc=0.78260
    Best now is epoch=49, test acc=0.7826
    ```

    <img src="D:\2Codefield\VS_code\python\Learn_Base\CNN\VGG\VGGModel\vgg13.png" alt="vgg13" style="zoom:50%;" />

-   **VGG16**：

    ```python
    Train Epoch=48, TrainLoss=0.57503, TrainAcc=0.80748
    Train Epoch=48, TestAcc=0.77340
    Best now is epoch=48, test acc=0.7734
    ```

    <img src="D:\2Codefield\VS_code\python\Learn_Base\CNN\VGG\VGGModel\vgg16.png" alt="vgg16" style="zoom:100%;" />

-   **VGG19**：不知为何无法训练，准确率一直在20%徘徊
