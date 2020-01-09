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
BATCH_SIZE = 100    #批处理尺寸(batch_size)
LR = 0.001        #学习率
 


DATASET = 'quickdraw16'
# pdb.set_trace()
if DATASET == 'mnist':
    # pdb.set_trace()
    # 定义数据预处理方式
    transform = transforms.ToTensor()

    # 定义训练数据集
    trainset = tv.datasets.MNIST(root='/home/lichen/media/DataSets/',
                                train=True,
                                download= False,
                                transform=transform)

    # 定义训练批处理数据
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            )
    
    # 定义测试数据集
    testset = tv.datasets.MNIST(root='/home/lichen/media/DataSets/',
                                train=False,
                                download=False,
                                transform=transform)
    
    # 定义测试批处理数据
    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                            )
    
elif DATASET == 'ucm':
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Scale(256),
            transforms.Resize((28,28)),
		    # transforms.RandomSizedCrop(224),
		    #transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
		    transforms.ToTensor(),
		    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
        'val': transforms.Compose([
		    # transforms.Scale(256),
            transforms.Resize((28,28)),
		    # transforms.CenterCrop(30),
            transforms.Grayscale(num_output_channels=1),
		    transforms.ToTensor(),
		    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ]),
	}
    data_dir = '/home/lichen/media/DataSets/UCM/Images'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    trainloader = dataloader['train']
    testloader = dataloader['val']
elif DATASET == 'quickdraw16':
    train_data_npy = np.load('/home/lichen/media/DataSets/quickdraw/quickdraw16_train.npy')
    test_data_npy = np.load('/home/lichen/media/DataSets/quickdraw/quickdraw16_test.npy')

    train_data = train_data_npy.reshape((np.shape(train_data_npy)[0],28,28))
    train_label = np.zeros((np.shape(train_data_npy)[0]))
    for i in range(np.shape(train_label)[0]):
        train_label[i] = i//8000

    test_data = test_data_npy.reshape((np.shape(test_data_npy)[0],28,28))
    test_label = np.zeros((np.shape(test_data_npy)[0]))
    for i in range(np.shape(test_label)[0]):
        test_label[i] = i//100

    train_data = torch.from_numpy(train_data)
    train_data=train_data.float()
    train_label = torch.from_numpy(train_label)
    train_label = train_label.long()
    test_data = torch.from_numpy(test_data)
    train_data=train_data.float()
    test_label = torch.from_numpy(test_label)
    test_label = test_label.long()
    trainloader = torch.utils.data.TensorDataset(train_data,train_label)
    testloader = torch.utils.data.TensorDataset(test_data,test_label)
    trainloader = torch.utils.data.DataLoader(trainloader, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testloader, batch_size=BATCH_SIZE, shuffle=True)


# pdb.set_trace()

# 定义损失函数loss function 和优化方式（采用SGD）
# 定义是否使用GPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
net = model.Multi_fft_conv2d().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
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

