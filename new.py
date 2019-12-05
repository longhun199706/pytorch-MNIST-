import torch
import torch.nn as nn
import torch.nn.functional as F
# torchvision 包收录了若干重要的公开数据集、网络模型和计算机视觉中的常用图像变换
import torchvision
import torchvision.transforms as transforms
import cv2

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Conv2d(128, 1024, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 10, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x

    def retrieve_features(self, x):
        # 该函数专门用于提取卷积神经网络的特征图的功能，返回feature_map1, feature_map2为前两层卷积层的特征图
        feature_map1 = F.relu(self.conv1(x))  # 完成第一层卷积
        x = self.pool(feature_map1)  # 完成第一层pooling
        print('type(feature_map1)=', feature_map1)
        feature_map2 = F.relu(self.conv2(x))  # 第二层卷积，两层特征图都存储到了feature_map1, feature_map2中
        return (feature_map1, feature_map2)
#计算预测正确率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    path = 'C:/Users/BOOM/OneDrive/作业/模式识别/python神经网络'
    path2 = 'C:/Users/BOOM/OneDrive/作业/模式识别/python神经网络/resent.pth'


    # transform=transforms.ToTensor():将图像转化为Tensor，在加载数据的时候，就可以对图像做预处理
    train_dataset = torchvision.datasets.MNIST(root=path, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root=path, train=False, transform=transforms.ToTensor(), download=True)
    # 训练数据集的加载器，自动将数据分割成batch，顺序随机打乱
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=50, shuffle=True)


    #接下来把测试数据中的前5000个样本作为验证集，后5000个样本作为测试集
    indices = range(len(test_dataset))
    indices_val = indices[:5000-1]
    indices_test = indices[5000:]
    # 通过下标对验证集和测试集进行采样
    sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
    sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)
    # 根据采样器来定义加载器，然后加载数据
    validation_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=50, sampler=sampler_val)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=50, sampler=sampler_test)


    #预览一个批次
    images, labels = next(iter(train_loader))
    img = torchvision.utils.make_grid(images)#把图片排列成网格形状。
    print(img.shape)
    img = img.numpy().transpose(1, 2, 0)#具体的变换高，宽，通道数
    print(img.shape)
    #std = [0.5, 0.5, 0.5]
    #mean = [0.5, 0.5, 0.5]
    #img = img * std + mean#颜色反转
    print(labels)
    # print([labels[i] for i in range(64)])
    # 由于matplotlab中的展示图片无法显示，所以现在使用OpenCV中显示图片
    # plt.imshow(img)
    cv2.imshow('win', img)
    key_pressed = cv2.waitKey(0)


    def accuracy(predictions, labels):
        # torch.max的输出：out (tuple, optional维度) – the result tuple of two output tensors (max, max_indices)
        pred = torch.max(predictions.data, 1)[1]  # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
        right_num = pred.eq(labels.data.view_as(pred)).sum()  # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
        return right_num, len(labels)  # 返回正确的数量和这一次一共比较了多少元素


    image_size = 28
    num_classes = 10
    num_epochs = 6
    batch_size = 50

    net = ConvNet()
    print(net)

    criterion = nn.CrossEntropyLoss()  # Loss函数的定义，交叉熵
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-08)  # 定义优化器，普通的随机梯度下降算法momentum=0.9

    record = []  # 记录准确率等数值的list
    weights = []  # 每若干步就记录一次卷积核

    for epoch in range(num_epochs):
        train_accuracy = []  # 记录训练数据集准确率的容器

        # 一次迭代一个batch的 data 和 target
        for batch_id, (data, target) in enumerate(train_loader):
            net.train()  # 给网络模型做标记，标志说模型正在训练集上训练，这种区分主要是为了打开关闭net的training标志，从而决定是否运行dropout

            output = net(data)  # forward
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracies = accuracy(output, target)
            train_accuracy.append(accuracies)

            if batch_id % 100 == 0:  # 每间隔100个batch执行一次打印等操作
                net.eval()  # 给网络模型做标记，将模型转换为测试模式。
                val_accuracy = []  # 记录校验数据集准确率的容器

                for (data, target) in validation_loader:  # 计算校验集上面的准确度
                    output = net(data)  # 完成一次前馈计算过程，得到目前训练得到的模型net在校验数据集上的表现
                    accuracies = accuracy(output, target)  # 计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
                    val_accuracy.append(accuracies)

                # 分别计算在已经计算过的训练集，以及全部校验集上模型的分类准确率

                # train_r为一个二元组，分别记录目前已经经历过的所有训练集中分类正确的数量和该集合中总的样本数，
                train_r = (sum([tup[0] for tup in train_accuracy]), sum([tup[1] for tup in train_accuracy]))
                # val_r为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
                val_r = (sum([tup[0] for tup in val_accuracy]), sum([tup[1] for tup in val_accuracy]))
    #            torch.save(net, path2)
                # 打印准确率等数值，其中正确率为本训练周期Epoch开始后到目前batch的正确率的平均值
                print('Epoch [{}/{}] [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t训练正确率: {:.2f}%\t校验正确率: {:.2f}%'.format(
                    epoch + 1, num_epochs, batch_id * batch_size, len(train_loader.dataset),
                    100. * batch_id / len(train_loader), loss.item(),
                    100. * train_r[0] / train_r[1],
                    100. * val_r[0] / val_r[1]))
    torch.save(net, path2)
    PATH = 'D:/resent.pth'
    torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)

    #torch.save(net.state_dict(), path2)


    # 在测试集上进行测试
    net.eval() #标志模型当前为测试阶段
    vals = [] #记录准确率所用列表

    with torch.no_grad():
        for data, target in test_loader:
            output = net(data)
            val = accuracy(output, target)
            #print(val[0].data)
            vals.append(val)

    #计算准确率
    rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
    right_rate = 1.0 * rights[0].data.numpy() / rights[1]

    print("accuracy:", right_rate)
