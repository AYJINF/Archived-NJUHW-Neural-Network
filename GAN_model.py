import os
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import tqdm
from PIL import Image

# 参数设置
img_size = 96
noise_size = 100
gen_feature = 64
dis_feature = 64
batch_size = 256
sum_epochs = 500
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 定义训练图像的数据集
class MyDataset(data.DataLoader):
    def __init__(self, path_list, transform):
        self.path_list = path_list
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.path_list[idx])
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.path_list)


# 用于Generator的block, 核心部分为反卷积
def gen_block(input_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False):
    # ConvTranspose2d 每边计算公式: output=(input-1)*stride + kernel_size -2*padding + output_padding
    layer = [
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(True)
    ]
    return nn.Sequential(*layer)


# 用于Discriminator的block, 核心部分为卷积
def dis_block(input_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False):
    # Conv2d 每边计算公式: output=(input-kernel+2*padding)/stride +1
    layer = [
        nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.2, inplace=True)
    ]
    return nn.Sequential(*layer)


# Generator网络结构
class MyGenerator(nn.Module):
    def __init__(self):  # TODO: 网络层的参数还可以调节
        super(MyGenerator, self).__init__()  # 100*1*1
        self.block1 = gen_block(noise_size, gen_feature * 8, kernel_size=4, stride=1, padding=0)  # 64*8*4*4
        self.block2 = gen_block(gen_feature * 8, gen_feature * 4)  # 64*4*8*8
        self.block3 = gen_block(gen_feature * 4, gen_feature * 2)  # 64*2*16*16
        self.block4 = gen_block(gen_feature * 2, gen_feature * 1)  # 64*1*32*32
        self.layer5 = nn.ConvTranspose2d(gen_feature * 1, 3, kernel_size=5, stride=3, padding=1, bias=False)  # 3*96*96
        self.tanh = nn.Tanh()

    def forward(self, x):  # 100*1*1
        out = self.block1(x)  # 64*8*4*4
        out = self.block2(out)  # 64*4*8*8
        out = self.block3(out)  # 64*2*16*16
        out = self.block4(out)  # 64*32*32
        out = self.layer5(out)  # 3*96*96
        return self.tanh(out)  # 限制像素值


# Discriminator网络结构，主体结构为cnn
class MyDiscriminator(nn.Module):
    def __init__(self):
        super(MyDiscriminator, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, dis_feature, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 3*96*96 -> 64*32*32
        self.cnn2 = dis_block(dis_feature, dis_feature * 2)  # 64*2*16*16
        self.cnn3 = dis_block(dis_feature * 2, dis_feature * 4)  # 64*4*8*8
        self.cnn4 = dis_block(dis_feature * 4, dis_feature * 8)  # 64*8*4*4
        self.cnn5 = nn.Sequential(
            nn.Conv2d(dis_feature * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )  # 64*8*4*4 -> 1

    def forward(self, x):
        out = self.cnn1(x)
        out = self.cnn2(out)
        out = self.cnn3(out)
        out = self.cnn4(out)
        return self.cnn5(out)


# 数据准备
dir_path = 'faces'  # TODO: 注意修改路径
path_list = [os.path.join(dir_path, img_path) for img_path in os.listdir(dir_path)]
path_list = path_list[:51200]  # TODO: 可以修改来让模型训练更快

# 模型建立
Generator = MyGenerator()
Discriminator = MyDiscriminator()
Generator.to(device)
Discriminator.to(device)

# 优化器
gen_optimizer = optim.Adam(Generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
dis_optimizer = optim.Adam(Discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
# 损失函数用二分类交叉熵
criterion = torch.nn.BCELoss()


# 训练过程
def train(dataset, batch_size, noise_size, gen_epoch, dis_epoch):
    sum_loss_D = 0
    sum_loss_G = 0

    train_dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size)

    true_labels = torch.ones(batch_size)
    true_labels = true_labels.view(-1, 1, 1, 1)
    true_labels = true_labels.to(device)
    fake_labels = torch.zeros(batch_size)
    fake_labels = fake_labels.view(-1, 1, 1, 1)
    fake_labels = fake_labels.to(device)

    for i, img in tqdm.tqdm(enumerate(train_dataloader)):  # 256 * 3 * 96 * 96
        img = img.to(device)

        # 训练判别器 Discriminator
        if (i + 1) % dis_epoch == 0:
            dis_optimizer.zero_grad()  # 消除上一步训练Generator时的反向传播

            # true图片loss与反向传播
            output = Discriminator(img)
            dis_true_loss = criterion(output, true_labels)
            dis_true_loss.backward()

            # fake图片loss与反向传播
            dis_noise = torch.randn(batch_size, noise_size, 1, 1)
            dis_noise = dis_noise.to(device)
            dis_fake_img = Generator(dis_noise).detach()  # 固定Generator的参数
            dis_fake_output = Discriminator(dis_fake_img)
            dis_fake_loss = criterion(dis_fake_output, fake_labels)
            dis_fake_loss.backward()

            # 优化器
            dis_optimizer.step()

            # 总损失
            sum_loss_D += (dis_true_loss + dis_fake_loss).item()

        # 训练生成器 Generator
        if (i + 1) % gen_epoch == 0:
            gen_optimizer.zero_grad()

            gen_noise = torch.randn(batch_size, noise_size, 1, 1)
            gen_noise = gen_noise.to(device)
            gen_fake_img = Generator(gen_noise)  # 此时需要得到dis的结果, 不detach, 对dis的影响会被训练前的zero_grad()消除
            gen_fake_output = Discriminator(gen_fake_img)
            generator_loss = criterion(gen_fake_output, true_labels)
            generator_loss.backward()

            gen_optimizer.step()

            sum_loss_G += generator_loss.item()

    return sum_loss_D / sum_epochs, sum_loss_G / sum_epochs


# 展示 Generator 生成效果
def show(num, batch_size, noise_size):
    fix_noises = torch.randn(batch_size, noise_size, 1, 1)
    fix_noises = fix_noises.to(device)

    fix_fake_imgs = Generator(fix_noises)
    fix_fake_imgs = fix_fake_imgs.data.cpu()[:16] * 0.5 + 0.5

    fig = plt.figure()

    i = 1
    for image in fix_fake_imgs:
        fig.add_subplot(4, 4, eval('%d' % i))
        plt.axis('off')
        plt.imshow(image.permute(1, 2, 0))  # reshape (96,96,3)
        i += 1
    plt.subplots_adjust(left=None, right=None, bottom=None, top=None, wspace=0.05, hspace=0.05)
    plt.suptitle('epoch {}'.format(num), fontsize=15)
    plt.savefig('epoch {}'.format(num) + '.png')


train_dataset = MyDataset(path_list, transform)
plot_epoch = [49, 99, 149, 199, 249, 299, 349, 399, 449, 499]
lossD_list = []
lossG_list = []
for i in range(sum_epochs):
    lossD, lossG = train(train_dataset, batch_size, noise_size, gen_epoch=1, dis_epoch=1)
    lossD_list.append(lossD)
    lossG_list.append(lossG)
    print('epoch {}: loss of D {} , loss of G {}'.format(i, lossD, lossG))
    if i in plot_epoch:
        show(i + 1, batch_size, noise_size)

torch.save(Discriminator, 'Discriminator.pkl')
torch.save(Generator, 'Generator.pkl')

# %%
plt.figure(figsize=(6, 4))
plt.plot(range(sum_epochs), lossD_list, label='lossD')
plt.plot(range(sum_epochs), lossG_list, label='lossG')
plt.grid(True)
plt.legend()
plt.savefig('G_lossD_vs_lossG.png')
