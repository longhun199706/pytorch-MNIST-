#coding = utf-8
#MNIST黑底白字
#50 1 28 28;;3 1 25 25

from new import ConvNet
import torch
import torch.nn as nn
import torch.nn.functional
import torchvision
import torchvision.transforms as transforms
import cv2
import os.path
import glob
from PIL import Image
import numpy as np


PATH = 'D:/resent.pth'
path2 = 'C:/Users/BOOM/OneDrive/作业/模式识别/python神经网络/resent.pth'
model = torch.load(path2) #要将类的定义添加到加载模型的这个py文件中
model.eval()#不启用batchnormalize和dropout


#A = torch.empty(1, 1, 28, 28)

n = 1
for img in glob.glob(r'test\1\*.bmp'):

    img = cv2.imread(img)#不使用绝对路径，不使用中文路径
    print(img.shape)#这里的图片25*25*3
    size = (int(28), int(28))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA) #改大小

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_info = img.shape
    image_height = img_info[0]
    image_weight = img_info[1]
    dst=np.zeros((image_height,image_weight,1),np.uint8)
    for i in range(image_height):
        for j in range(image_weight):
            grayPixel=gray[i][j]
            dst[i][j]=255-grayPixel
    #cv2.imshow('gary',dst)
    #cv2.waitKey(0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#单通道，图片28*28
    img = torchvision.transforms.functional.to_tensor(img)#图片1*28*28
    img = torch.unsqueeze(img, 0)#图片1*1*28*28
    #img = img/255
    print(img)
    print(img.shape)

#    b = A.data.numpy().shape[0]
    if n == 1:
        A = img
    else:
        A = torch.cat((A, img), 0)
    n = n+1

output = model(A)
print(output)
pred = torch.max(output.data, 1)[1].data.numpy()# 返回每一行中最大值的那个元素的索引，然后将tenor变为numpy输出
print('预测的数字为{}'.format(pred))

