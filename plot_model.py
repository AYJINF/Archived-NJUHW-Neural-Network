import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

import os
import sys

import torch.onnx
from torch.autograd import Variable
import onnx
from onnx import shape_inference

# 自定义Dataset类
from torch.utils.tensorboard import SummaryWriter


class MyEmotionDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 将pixels转化为1*48*48的tensor
        pixels = np.array([int(i) for i in self.data.iloc[idx]['pixels'].split()]).astype(np.uint8)

        # 直方图均衡化
        pixels = cv2.equalizeHist(pixels)
        pixels = np.array(pixels.reshape(48, 48) / 255).astype(np.float32)

        pixels = pixels.reshape(48, 48, 1)

        # 将emotion转化为tensor
        emotion = torch.tensor(int(self.data.iloc[idx]['emotion']), dtype=torch.long)

        # 数据增强等操作
        if self.transform:
            pixels = self.transform(pixels)

        return pixels, emotion


# 用nn.Sequential构建block
def block(input_channels, output_channels, pool=False, kernel_size=5):
    layer = [
        nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layer.append(nn.MaxPool2d(2))
    return nn.Sequential(*layer)


# 构建我的CNN神经网络模型
class MyEmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            block(1, 24, pool=True),  # 1*48*48 -> 24*22*22
            block(24, 48),  # 24*22*22 -> 48*18*18
            block(48, 64, pool=True),  # 48*18*18 -> 64*7*7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 7)
        )

    def forward(self, x):
        return self.main(x)


# 用nn.Sequential构建res_block(保证前后尺寸不变)
def res_block(input_channels, output_channels):
    layer = [
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layer)


# 构建ResNet作为尝试
class MyResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = block(1, 24, pool=True)  # 1*48*48 -> 24*22*22
        self.res1 = nn.Sequential(res_block(24, 24), res_block(24, 24))  # 24*22*22
        self.drop1 = nn.Dropout(0.4)

        self.conv2 = block(24, 96, pool=True)  # 24*22*22 -> 96*9*9
        self.res2 = nn.Sequential(res_block(96, 96), res_block(96, 96))  # 96*9*9
        self.drop2 = nn.Dropout(0.6)

        self.conv3 = block(96, 192)  # 96*9*9 -> 192*5*5
        self.res3 = nn.Sequential(res_block(192, 192), res_block(192, 192))  # 192*5*5
        self.drop3 = nn.Dropout(0.6)

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(192 * 5 * 5, 7))

    def forward(self, x):
        out = self.conv1(x)
        out = self.res1(out) + out
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.drop3(out)

        return self.classifier(out)


model1 = MyEmotionCNN()
model2 = MyResNet()
input = torch.rand(10, 1, 48, 48)
model = torch.load('epoch71.pkl', map_location=torch.device('cpu'))
# writer = SummaryWriter("./visual_logs")
# writer.add_graph(model1, input)
# writer.add_graph(model2, input)
# writer.close()
torch.onnx.export(model, input, 'model.onnx', input_names=['input'], output_names=["output"], opset_version=11)
# onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model)), model)
