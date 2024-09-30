import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

# from google.colab import drive
# drive.mount('/content/drive')


# 自定义Dataset类
class MyEmotionDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 将pixels转化为1*48*48的tensor
        pixels = np.array([int(i) for i in self.data.iloc[idx]['pixels'].split()]).astype(np.uint8)
        pixels = pixels.reshape(48, 48, 1)

        # 将emotion转化为tensor
        emotion = torch.tensor(int(self.data.iloc[idx]['emotion']), dtype=torch.long)

        # 数据预处理
        if self.transform:
            pixels = self.transform(pixels)

        return pixels, emotion


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


# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 准确率计算
def cal_accuracy(model, val_dataloader):
    acc = 0
    num = 0
    for img, label in val_dataloader:
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        prediction = torch.argmax(output, axis=1)
        acc += np.sum((prediction == label).cpu().numpy()).item()
        num += img.shape[0]
    return acc / num


# 训练设置
def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    train_acc_list = []
    val_acc_list = []
    loss_list = []
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

    model = MyResNet()
    model.to(device)

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wt_decay)

    # 设置损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 学习率调度
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # 开始训练
    for epoch in range(epochs):
        model.train()
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss_list.append(loss)
            loss.backward()
            optimizer.step()
        model.eval()
        scheduler.step()

        # 计算训练集和验证集上的准确率
        train_acc = cal_accuracy(model, train_dataloader)
        val_acc = cal_accuracy(model, val_dataloader)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        print(
            'epoch {}: train accuracy = {}, test accuracy = {}, loss = {}'.format(epoch + 1, train_acc, val_acc, loss))

    return train_acc_list, val_acc_list, loss_list, model


if __name__ == '__main__':

    # 读取train数据
    train_raw = pd.read_csv('train.csv')

    # 划分训练集和验证集
    train_data, val_data = train_test_split(train_raw, test_size=0.2, random_state=1)

    # 定义数据转换方式
    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # TODO
    ])

    # 创建Dataset实例
    train_dataset = MyEmotionDataset(train_data, transform)
    val_dataset = MyEmotionDataset(val_data, transform)
    # print(train_dataset)

    # 创建DataLoader
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_acc_list, test_acc_list, loss_list, model = train(train_dataset, val_dataset,
                                                            batch_size=256, epochs=30,  # TODO
                                                            learning_rate=1e-3, wt_decay=1e-5)


    torch.save(model, "model1.pkl")

    # test_raw = pd.read_csv('test.csv')
    # test_dataset = MyEmotionDataset(test_raw, transform)

