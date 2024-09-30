# %%
import torch
import pandas as pd
import numpy as np
import cv2

# 自定义Dataset类
import torch.utils.data as data
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset


# 训练模型时的Dataset
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


# 便于测试的自定义Dataset
class MyDateset(data.Dataset):
    def __init__(self, x_data):
        super(MyDateset, self).__init__()
        self.x_data = x_data

    def __getitem__(self, item):
        face_tensor = self.x_data[item]
        return face_tensor

    def __len__(self):
        return self.x_data.shape[0]


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


train = pd.read_csv('train.csv')

# 提取pixels与emotion
train_pixels = []
for pixel in train['pixels'].values:
    pix = np.array([int(i) for i in pixel.split()]).astype(np.uint8)
    train_pixels.append(pix)
train_x = np.array(train_pixels, dtype='float32')
train_y = train['emotion'].values

# 设备设置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
img = torch.from_numpy(train_x[0].reshape(1, 1, 48, 48)).to(device)

model = torch.load('epoch71.pkl', map_location=torch.device('cpu'))
model.eval()

# model_weights = []
# conv_layers = []
#
#
# def get_conv_layers(model):
#     for layer in model.children():
#         if isinstance(layer, nn.Conv2d):
#             model_weights.append(layer.weight)
#             conv_layers.append(layer)
#         elif isinstance(layer, nn.Sequential):
#             get_conv_layers(layer)
#         else:
#             for child in layer.children():
#                 get_conv_layers(child)
#
#
# # 获取所有卷积层和其权重
# get_conv_layers(model)
#
# # %%
# outputs = []
# names = []
# for layer in conv_layers:
#     img = layer(img)
#     outputs.append(img)
#     names.append(str(layer))
# print(len(outputs))
# # 打印特征图的形状
# for feature_map in outputs:
#     print(feature_map.shape)
#
# # %%
# processed = []
# for feature_map in outputs:
#     feature_map = feature_map.squeeze(0)
#     gray_scale = torch.sum(feature_map, 0)
#     gray_scale = gray_scale / feature_map.shape[0]
#     processed.append(gray_scale.data.cpu().numpy())
# for fm in processed:
#     print(fm.shape)
#
# # %%
# import matplotlib.pyplot as plt
#
# fig = plt.figure(figsize=(8, 8))
# for i in range(len(processed)):
#     a = fig.add_subplot(5, 4, i + 1)
#     imgplot = plt.imshow(processed[i])
#     a.axis("off")
#     a.set_title(names[i].split('(')[0], fontsize=8)
# fig.suptitle('The Feature Map')
# plt.show()

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

model_weights = []
conv_layers = []


def get_conv_layers(model):
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            model_weights.append(layer.weight)
            conv_layers.append(layer)
        elif isinstance(layer, nn.Sequential):
            get_conv_layers(layer)
        else:
            for child in layer.children():
                get_conv_layers(child)


# 获取所有卷积层和其权重
get_conv_layers(model)

# 模型设备设置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 假设有多张图片，放在一个列表中
image_list = [torch.from_numpy(train_x[i].reshape(1, 1, 48, 48)).to(device) for i in range(len(train_x))]

j = 1

for img in image_list:
    outputs = []
    names = []

    for layer in conv_layers:
        img = layer(img)
        outputs.append(img)
        names.append(str(layer))

    # 打印特征图的形状
    for feature_map in outputs:
        print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    for fm in processed:
        print(fm.shape)

    # 可视化特征图
    fig = plt.figure(figsize=(8, 8))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i + 1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=8)
    fig.suptitle('The Feature Map')
    plt.savefig('Feature Map'+str(j)+'.png')
    j += 1

