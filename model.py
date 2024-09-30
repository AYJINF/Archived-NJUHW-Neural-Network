import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn as nn
import torch
from torch import optim
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import pandas as pd
from sklearn.decomposition import PCA

# 一些超参
random.seed(1024)
valid_rate = 0.2
image_size = 32
epochs = 450
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-3
lamb = 0.2

# 设备适配
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义数据预处理方式
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

valid_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 损失函数定义
class_loss_function = nn.CrossEntropyLoss()  # 交叉熵
domain_loss_function = nn.BCEWithLogitsLoss()  # 二分类


# 对训练集和测试集的数据进行整理
def data_organizing(train_root, test_root, valid_rate):
    # 训练集类别
    class_list = os.listdir(train_root)
    path_label_list = []
    for cls in class_list:
        cls_root_path = os.path.join(train_root, cls)
        # path_label = [[图像路径, 类别cls],...]
        path_label = [[os.path.join(cls_root_path, path), cls] for path in os.listdir(cls_root_path)]
        path_label_list.extend(path_label)
    # 打乱顺序
    random.shuffle(path_label_list)

    all_len = len(path_label_list)  # 所有train样本数
    valid_len = int(all_len * valid_rate)  # 验证集样本数
    train_data = path_label_list[valid_len:]  # 训练集
    valid_data = path_label_list[:valid_len]  # 验证集

    with open("train.txt", 'w') as f:
        for path_label in tqdm(train_data):
            f.write(path_label[0] + "," + path_label[1] + "\n")
    with open("valid.txt", 'w') as f:
        for path_label in tqdm(valid_data):
            f.write(path_label[0] + "," + path_label[1] + "\n")

    # 整理测试集txt, 标签空出
    test_list = os.listdir(test_root)
    with open("test.txt", 'w') as f:
        for path in tqdm(test_list):
            test_path = os.path.join(test_root, path)
            f.write(test_path + "\n")


# 定义数据集
class MyDataset(data.Dataset):
    # file_name为存储路径与类别的txt文件, mode = 'train', 'valid' or 'test'
    def __init__(self, file_name, mode, transform=None):
        self.path_list = []
        self.label_list = []
        with open(file_name, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(',')
                img_pth = '/'.join(line[0].split('//'))
                self.path_list.append(img_pth)
                if mode != 'test':
                    self.label_list.append(int(line[1]))

        self.transform = transform
        self.mode = mode

    def __getitem__(self, idx):
        # read img（H,W,C）
        img = Image.open(self.path_list[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # 注意：test数据集和valid/train的返回值不同！
        if self.mode != 'test':
            label = np.array(self.label_list[idx], dtype='int64')
            return img, label
        else:
            return [img]

    def __len__(self):
        return len(self.path_list)


# 用于卷积的block
def cnn_block(input_channels, output_channels, kernel_size=3, stride=1, padding=1, inplace=True, pool_size=2, drop=False):
    layer = [
        nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=inplace),
        nn.MaxPool2d(pool_size)
    ]
    if drop:
        layer.append(nn.Dropout(0.5))
    return nn.Sequential(*layer)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.cnn1 = cnn_block(1, 128)  # (1, 32, 32) -> (128, 16, 16)
        self.cnn2 = cnn_block(128, 256, drop=True)  # (256, 8, 8)
        self.cnn3 = cnn_block(256, 512, drop=True)  # (512, 4, 4)
        self.cnn4 = cnn_block(512, 1024, drop=True)  # (1024, 2, 2)
        self.cnn5 = cnn_block(1024, 512)  # (512, 1, 1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.cnn2(out)
        out = self.cnn3(out)
        out = self.cnn4(out)
        return self.cnn5(out)


class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 9),
        )

    def forward(self, x):
        return self.layer0(x)


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.layer(x)


# 数据准备
train_root = 'train_data'
test_root = 'test_data'
data_organizing(train_root, test_root, valid_rate=valid_rate)

train_d = MyDataset(file_name='train.txt', mode='train', transform=train_transform)
valid_d = MyDataset(file_name='valid.txt', mode='valid', transform=valid_transform)
test_d = MyDataset(file_name='test.txt', mode='test', transform=test_transform)

train_dataloader = data.DataLoader(dataset=train_d, batch_size=batch_size)
valid_dataloader = data.DataLoader(dataset=valid_d, batch_size=batch_size)
target_dataloader = data.DataLoader(dataset=test_d, batch_size=batch_size)
test_dataloader = data.DataLoader(dataset=test_d, batch_size=batch_size)

# 网络实例
ExtractorNet = FeatureExtractor()  # 特征提取
PredictorNet = LabelPredictor()  # 标签预测
ClassifierNet = DomainClassifier()  # 领域对抗
ExtractorNet.to(device)
PredictorNet.to(device)
ClassifierNet.to(device)

# 优化器
optimizer_Ext = optim.Adam(params=ExtractorNet.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer_Pre = optim.Adam(params=PredictorNet.parameters(), lr=learning_rate)
optimizer_Cla = optim.Adam(params=ClassifierNet.parameters(), lr=learning_rate)


# 训练主函数(lamb为领域对抗的权重)
def train_epoch_with_domain(train_dataloader, target_dataloader, lamb):  # lamb控制域判别器loss
    # 训练模式
    ExtractorNet.train()
    PredictorNet.train()

    # 初始化损失(前者为领域对抗(ClassifierNet)的loss, total_loss = 分类与领域对抗综合)
    classifier_loss, total_loss = 0.0, 0.0
    # 初始化分类准确率与样本数
    acc_num, total_num = 0.0, 0.0

    for i, ((train_data, train_label), (target_data,)) in enumerate(zip(train_dataloader, target_dataloader)):
        # 将train_data和target_data放入ClassifierNet(领域对抗)
        train_data = train_data.to(device)
        target_data = target_data.to(device)
        train_label = train_label.to(device)

        # 设置领域标签
        mixed_data = torch.concat([train_data, target_data], axis=0)
        domain_label = torch.zeros([train_data.shape[0] + target_data.shape[0], 1]).to(device)
        domain_label[:train_data.shape[0]] = 1  # 源域设置

        # Step 1: 训练领域分类器(Domain Classifier)
        feature = ExtractorNet(mixed_data)
        domain_logits = ClassifierNet(feature.detach())
        loss = domain_loss_function(domain_logits, domain_label)
        loss.backward()
        optimizer_Cla.step()
        classifier_loss += loss.cpu().detach().numpy().item()

        # Step 2: 训练标签预测器（Label Predictor）和特征提取器（Feature Extractor）：
        label_logits = PredictorNet(feature[:train_data.shape[0]])
        domain_logits = ClassifierNet(feature)
        loss = class_loss_function(label_logits, train_label) - lamb * domain_loss_function(domain_logits, domain_label)
        loss.backward()
        optimizer_Ext.step()
        optimizer_Pre.step()
        total_loss += loss.cpu().detach().numpy().item()

        # 梯度清零避免累积
        optimizer_Cla.zero_grad()
        optimizer_Ext.zero_grad()
        optimizer_Pre.zero_grad()

        acc_num += np.sum((torch.argmax(label_logits, axis=1) == train_label).cpu().numpy()).item()
        total_num += train_data.shape[0]

    # 返回该次的损失和准确率
    return classifier_loss / (i + 1), total_loss / (i + 1), acc_num / total_num


# 没有领域对抗的训练
def train_epoch_without_domain(train_dataloader):
    ExtractorNet.train()
    PredictorNet.train()

    total_acc, total_num = 0.0, 0.0

    for (train_data, train_label) in train_dataloader:
        train_data = train_data.to(device)
        train_label = train_label.to(device) # TODO
        feature = ExtractorNet(train_data)
        class_logits = PredictorNet(feature)
        loss = class_loss_function(class_logits, train_label)
        loss.backward()

        optimizer_Ext.step()
        optimizer_Pre.step()
        optimizer_Ext.zero_grad()
        optimizer_Pre.zero_grad()

        total_acc += np.sum((torch.argmax(class_logits, axis=1) == train_label).cpu().numpy())
        total_num += train_data.shape[0]
        return total_acc / total_num


# 验证集准确率评估
def evaluate(valid_dataloader):
    # 评估模式
    ExtractorNet.eval()
    PredictorNet.eval()

    total_acc, total_num = 0.0, 0.0

    for data in valid_dataloader:
        x_data = data[0].to(device)
        y_data = data[1].to(device)
        features = ExtractorNet(x_data)
        predicts = PredictorNet(features)
        total_acc += np.sum((torch.argmax(predicts, axis=1) == y_data).cpu().numpy()).item()
        total_num += x_data.shape[0]
    return total_acc / total_num


# 可视化损失和准确率
def visual_loss_acc(epochs, train_total_loss_list, train_acc_list, valid_acc_list, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(range(epochs), train_total_loss_list, label='train_loss')
    plt.plot(range(epochs), train_acc_list, label='train_acc')
    plt.plot(range(epochs), valid_acc_list, label='valid_acc')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)


# 训练无领域对抗的模型
for epoch in range(epochs):
    train_acc = train_epoch_without_domain(train_dataloader)
    valid_acc = evaluate(valid_dataloader)
    print('without_domain: epoch {:>3d},train acc:{:6.4f}, valid acc {:6.4f}'.format(epoch, train_acc, valid_acc))
torch.save(ExtractorNet, 'extractor_model_without_domain.pth')
torch.save(PredictorNet, 'predictor_model_without_domain.pth')

# 训练有领域对抗的模型
train_Cla_loss_list, train_total_loss_list = [], []
train_acc_list, valid_acc_list = [], []
for epoch in range(epochs):
    train_Cla_loss, train_total_loss, train_acc = train_epoch_with_domain(train_dataloader, target_dataloader, lamb=lamb)
    valid_acc = evaluate(valid_dataloader)
    print('with_domain: epoch {:>3d}: train Cla loss: {:6.4f}, train total loss: {:6.4f}, train acc {:6.4f}, '
          'valid acc {:6.4f}'.format(epoch + 1, train_Cla_loss, train_total_loss, train_acc, valid_acc))
    train_Cla_loss_list.append(train_Cla_loss)
    train_total_loss_list.append(train_total_loss)
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)
torch.save(ExtractorNet, 'extractor_model.pth')
torch.save(PredictorNet, 'predictor_model.pth')
torch.save(ClassifierNet, 'classifier_model.pth')

# 绘制领域对抗的损失和准确率
visual_loss_acc(epochs, train_total_loss_list, train_acc_list, valid_acc_list, 'loss_and_acc.png')

# 加载领域对抗模型
ExtModel = torch.load('extractor_model.pth')
PreModel = torch.load('predictor_model.pth')
ExtModel.eval()
PreModel.eval()

# 预测测试集
result = []
for (test_data,) in tqdm(test_dataloader):
    test_data = test_data.to(device)
    output = PreModel(ExtModel(test_data))
    predicts = torch.argmax(output, axis=1).cpu().detach().numpy()
    result.append(predicts)

# 输出预测结果
result = np.concatenate(result)
dataframe = pd.DataFrame({'ID': np.arange(0, len(result)), 'label': result})
dataframe.to_csv('res.csv', index=False)

# # 无领域对抗的model
# ExtModel_noDomain = torch.load('extractor_model_without_domain.pth')
# PreModel_noDomain = torch.load('predictor_model_without_domain.pth')
# ExtModel_noDomain.eval()
# PreModel_noDomain.eval()
#
# # 用于无领域的dataset
# train_dataloader_noDomain = data.DataLoader(dataset=train_d, batch_size=1)
# test_dataloader_noDomain = data.DataLoader(dataset=test_d, batch_size=1)
#
# # 提取无领域的feature
# train_feature_list = []
# test_feature_list = []
#
#
# pca = PCA(n_components=2)
#
# for i, (train_data, train_label) in enumerate(train_dataloader_noDomain):
#     train_data = train_data.to(device)
#     output = PreModel_noDomain(ExtModel_noDomain(train_data))
#     predicts2 = output.cpu().detach().numpy().reshape((9))
#     train_feature_list.append(predicts2)
#     if i == 4499:
#         break
# train_feature_list = pca.fit_transform(train_feature_list)
#
# # %%
# for i, (test_data,) in enumerate(test_dataloader_noDomain):
#     test_data = test_data.to(device)
#     output = PreModel_noDomain(ExtModel_noDomain(test_data))
#     preds = output.cpu().detach().numpy().reshape((9))
#     test_feature_list.append(preds)
#     if i == 4499:
#         break
# test_feature_list = pca.fit_transform(test_feature_list)
#
# # %%
# plt.figure()
# plt.scatter(train_feature_list[:, 0], train_feature_list[:, 1], s=2, c='red', label='source')
# plt.scatter(test_feature_list[:, 0], test_feature_list[:, 1], s=2, c='blue', label='target')
# plt.legend()
# plt.show()

