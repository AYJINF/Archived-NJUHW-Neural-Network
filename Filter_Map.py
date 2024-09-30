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


# 创建 MyResNet 模型的实例
model = MyResNet()

# 载入预训练模型
model = torch.load('epoch71.pkl', map_location=torch.device('cpu'))

# 将模型设置为评估模式
model.eval()

# 读取 train.csv 数据
train_data = pd.read_csv('train.csv')

# 提取七张图像的数据和标签
selected_emotions = [0, 1, 2, 3, 4, 5, 6]
selected_images = []

for emotion in selected_emotions:
    # 选择与情感对应的图像
    selected_image = train_data[train_data['emotion'] == emotion]['pixels'].values[0]
    selected_image = np.array([int(i) for i in selected_image.split()]).astype(np.uint8)
    selected_image = selected_image.reshape(1, 1, 48, 48) / 255.0  # 归一化
    selected_images.append(torch.from_numpy(selected_image).float())

# 选择观察的卷积层
selected_conv_layer = model.conv1  # 选择第一个卷积层


# 定义梯度上升步骤
def gradient_ascent(img, model, selected_layer, iterations=50, lr=0.1):
    img.requires_grad_()

    for _ in range(iterations):
        model.zero_grad()

        # 前向传播
        output = model(img)
        selected_output = selected_layer(img)

        # 定义一个损失函数，目标是最大化选定层的输出
        loss_fn = nn.CrossEntropyLoss()

        # 根据任务定义标签
        label = torch.tensor([emotion], dtype=torch.long)

        loss = loss_fn(output, label)

        # 反向传播，计算梯度
        loss.backward()

        # 更新输入图像
        img.data = img.data + lr * img.grad.data

    return img


# 使用梯度上升方法观察特定层的哪些图像最容易激活
for img, emotion in zip(selected_images, selected_emotions):
    result_img = gradient_ascent(img, model, selected_conv_layer)

    # 显示结果图像和情感标签
    plt.imshow(result_img.squeeze().detach().numpy(), cmap='gray')
    plt.title(f'Emotion: {emotion}')
    plt.axis('off')
    plt.show()
