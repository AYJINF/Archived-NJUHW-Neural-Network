# %%
import torch
import pandas as pd
import numpy as np
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt

# 自定义Dataset类
import torch.utils.data as data
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

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(192*5*5, 7))

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

# 提取pixels
train_pixels = []
for pixel in train['pixels'].values:
    pix = np.array([int(i) for i in pixel.split()]).astype(np.uint8)
    train_pixels.append(pix)
train_x = np.array(train_pixels, dtype='float32')

# 设备设置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x = torch.from_numpy(train_x.reshape(train_x.shape[0], 1, 48, 48)).to(device)


model = torch.load('epoch71.pkl', map_location=torch.device('cpu'))
model.eval()


for_sal_x = MyDateset(x)
batch_size = 1
sal_dataset = data.DataLoader(for_sal_x, batch_size)
for param in model.parameters():
    param.requires_grad = False

i = 1
fig = plt.figure(figsize=(12, 18))
fig.suptitle('Image & Saliency Map')

for vir_x in sal_dataset:
    vir_x.requires_grad = True
    output = model.forward(vir_x)
    score, preds = torch.max(output, 1)

    # 逆梯度计算
    score.backward()
    slc, _ = torch.max(torch.abs(vir_x.grad[0]), dim=0)
    slc = np.array(((slc - slc.min()) / (slc.max() - slc.min())).cpu().numpy() * 255, dtype=np.int8)
    img = np.array(vir_x.detach().cpu().numpy().reshape(48, 48) * 255, dtype=np.int8)

    plt.subplot(8, 4, 2 * i - 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplot(8, 4, 2 * i)
    plt.imshow(slc, cmap='hot')
    plt.axis('off')
    plt.tight_layout()
    i += 1

    if i == 17:  # 绘制前16张图片的Saliency Map
        break

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
plt.show()
