import os
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision as tv
from torchvision import datasets, models, transforms
import numpy as np
from PIL import Image
import model
import pdb
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


# 超参数设置
EPOCH = 100   #遍历数据集次数
BATCH_SIZE = 128      #批处理尺寸(batch_size)
LR = 0.001        #学习率
 


DATASET = 'mnist'



train_data_npy = np.load('/home/lichen/media/Datasets/quickdraw/quickdraw16_train.npy')
test_data_npy = np.load('/home/lichen/media/Datasets/quickdraw/quickdraw16_test.npy')
idcs = np.random.randint(0, np.shape(train_data)[0])
def get_feed(train, batch_size=50):
    if train:
        idcs = np.random.randint(0, np.shape(train_data)[0], batch_size)
        x = train_data[idcs, :]
        y = np.zeros((batch_size, classes))
        y[np.arange(batch_size), idcs//8000] = 1
        
    else:
        x = test_data
        y = np.zeros((np.shape(test_data)[0], classes))
        y[np.arange(np.shape(test_data)[0]), np.arange(np.shape(test_data)[0])//100] = 1                
    
    return x, y

x_test, y_test = get_feed(train=False)
# pdb.set_trace()

# 定义损失函数loss function 和优化方式（采用SGD）
# 定义是否使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
net = model.Multi_fft_conv2d().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99)
for epoch in range(EPOCH):
    # pdb.set_trace()
    sum_loss = 0.0
    # 数据读取
    for i, data in enumerate(trainloader):
        inputs, labels = data
        # pdb.set_trace()
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

