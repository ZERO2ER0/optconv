import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from PIL import Image

# 超参数设置
EPOCH = 10   #遍历数据集次数
BATCH_SIZE = 64      #批处理尺寸(batch_size)
LR = 0.001        #学习率
 
# 定义数据预处理方式
transform = transforms.ToTensor()
 
# 定义训练数据集
trainset = tv.datasets.MNIST(root='/Users/lichen/Downloads/DataSets/',
                             train=True,
                             download= False,
                             transform=transform)

# 定义训练批处理数据
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
    )
 
# 定义测试数据集
testset = tv.datasets.MNIST(root='/Users/lichen/Downloads/DataSets/',
                            train=False,
                            download=False,
                            transform=transform)
 
# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
    )


# 定义损失函数loss function 和优化方式（采用SGD）
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = fft_conv2d().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

for epoch in range(EPOCH):
        sum_loss = 0.0
        # 数据读取
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
 
            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        # 每跑完一次epoch测试一下准确率
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, (100 * correct / total)))

