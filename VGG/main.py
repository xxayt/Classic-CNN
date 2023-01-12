import torch
from torch import nn, optim, load
from src.model import *
from src.dataloader import *
from src.utils import *
from src.config import *
import os
from tqdm import tqdm
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

is_cuda = True if torch.cuda.is_available() else False

# 载入数据
train_loader, test_loader, num_iter = load_data(data_path)
# 可选模型['vgg11', 'vesgg13', 'vgg16', 'vgg19']
tit = "vgg16"
model = VGG(model_name=tit, num_classes=10, init_weights=True)
if is_cuda:
    model.cuda()

# 损失函数
loss_function = nn.CrossEntropyLoss()  # 交叉熵损失，相当于softmax+log+nlloss
# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.0002)
# 记录loss和acc
train_loss, train_acc, test_acc = AverageMeter(), AverageMeter(), AverageMeter()
max_test_acc = 0.0
# 画图
paint_iter = list(range(1, num_iter * TOTAL_EPOCH + 1))
paint_train_loss, paint_train_acc, paint_test_acc = [], [], []


# 训练集
def train(epoch):
    train_loss.reset()
    train_acc.reset()
    # train_bar = tqdm(train_loader)
    # for iter, data in enumerate(train_bar):
    for iter, data in enumerate(train_loader):
        input, labels = data  # input为输入数据, label为标签
        if is_cuda:
            input = input.cuda()
            labels = labels.cuda()
        batch_size = labels.size(0)
        # 更新参数
        optimizer.zero_grad()  # 梯度清零
        y_predict = model(input)  # 跑模型
        loss = loss_function(y_predict, labels)  # 计算损失
        loss.backward()  # 反向传播求导
        optimizer.step()  # 参数更新
        _, result = torch.max(y_predict.data, dim=1)
        # 更新训练loss和acc
        nowloss = loss.item()
        nowacc = (result==labels).sum().item() / batch_size
        # 画图数据
        paint_train_loss.append(nowloss)
        paint_train_acc.append(nowacc)
        train_loss.update(nowloss, batch_size)
        train_acc.update(nowacc, batch_size)
        # print(iter)
        if iter % 100 == 99:
            print('[%d:%d] loss is %f, acc is %f'%(epoch+1, iter+1, nowloss, nowacc))
    print('Train Epoch={}, TrainLoss={loss.avg:.5f}, TrainAcc={acc.avg:.5f}'.format(epoch + 1, loss=train_loss, acc=train_acc))

# 测试集
def test():
    test_acc.reset()
    with torch.no_grad():  # 测试不用计算梯度
        for data in test_loader:
            input, labels = data
            if is_cuda:
                input = input.cuda()
                labels = labels.cuda()
            batch_size = labels.size(0)
            y_predict = model(input)  # y_predict输出10个预测取值,其中最大的即为预测的数
            _, result = torch.max(y_predict.data, dim=1)  # 返回一个元组，第一个为最大值，第二个为最大值的下标
            # 更新测试acc
            nowacc = (result==labels).sum().item() / batch_size
            test_acc.update(nowacc, batch_size)
    print('Train Epoch={}, TestAcc={acc.avg:.5f}'.format(epoch+1, acc=test_acc))

# 保存模型
def save_model():
    global max_test_acc
    if max_test_acc <= test_acc.avg:
        max_test_acc = test_acc.avg
        torch.save(model.state_dict(), model_path + "/"+tit+"_model.pkl")
        torch.save(optimizer.state_dict(), model_path + "/"+tit+"_optimizer.pkl")

if __name__ == '__main__':
    start = time.time()
    for epoch in range(TOTAL_EPOCH):
        train(epoch)
        test()
        save_model()
    end = time.time()
    print('Sum time is %f'%(end-start))
    
    write2file(paint_iter, paint_train_loss, paint_train_acc, title=tit)
    Paint_Train(paint_iter, paint_train_loss, paint_train_acc, title=tit)