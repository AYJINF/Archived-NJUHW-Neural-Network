import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# 自定义Dataset类
class MyTestEmotionDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = np.array([int(i) for i in self.data.iloc[idx]['pixels'].split()]).astype(np.uint8)
        pixels = pixels.reshape(48, 48, 1)

        if self.transform:
            pixels = self.transform(pixels)

        return pixels


# 用nn.Sequential构建block
def block(input_channels, output_channels, pool=False):
    layer = [
        nn.Conv2d(input_channels, output_channels, kernel_size=5),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layer.append(nn.MaxPool2d(2))
    return nn.Sequential(*layer)


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


# 加载模型
model = torch.load('epoch71.pkl', map_location=torch.device('cpu'))

# 读取测试数据
test_raw = pd.read_csv('test.csv')
train_raw = pd.read_csv('train.csv')

# 定义数据转换方式
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        # TODO
    ])

# 创建测试集Dataset实例
test_dataset = MyTestEmotionDataset(test_raw, transform)
train_dataset = MyTestEmotionDataset(train_raw, transform)

# 创建DataLoader
batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# 预测并保存结果
predictions = []

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.eval()
with torch.no_grad():
    for img in train_loader:
        img = img.to(device)
        output = model(img)
        prediction = torch.argmax(output, axis=1).cpu().numpy()
        predictions.extend(prediction)

# 将预测结果保存到 DataFrame 中
test_raw['emotion'] = predictions

# 保存结果到 CSV 文件
test_raw.to_csv('train_predictions.csv', index=False)
