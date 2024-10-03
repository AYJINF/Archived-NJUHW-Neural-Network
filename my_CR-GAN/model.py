import torch.nn as nn
import torch.nn.parallel
import torch
import pdb

dd = pdb.set_trace

view_size = 2  # TODO: 这里只关注两个视角
feature_size = 128 - view_size


# 先卷积再平均池化
class conv_mean_pool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_mean_pool, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.pooling = nn.AvgPool2d(2)

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.pooling(out)
        return out


# 先平均池化再卷积
class mean_pool_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(mean_pool_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.pooling = nn.AvgPool2d(2)

    def forward(self, x):
        out = x
        out = self.pooling(out)
        out = self.conv(out)
        return out


# 先上采样再卷积
class upsample_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample_conv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        out = x
        out = self.upsample(out)
        out = self.conv(out)
        return out


# 用于下采样的残差块（避免梯度消失和梯度爆炸，尺寸缩小2倍）
class residual_block_down(nn.Module): # for discriminator, no batchnorm
    def __init__(self, in_channels, out_channels):
        super(residual_block_down, self).__init__()
        self.conv_shortcut = mean_pool_conv(in_channels, out_channels)  # 先平均池化再卷积（下采样）
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = conv_mean_pool(out_channels, out_channels)  # 先卷积再平均池化
        self.ReLU = nn.ReLU()

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        out = x
        out = self.ReLU(out)
        out = self.conv1(out)
        out = self.ReLU(out)
        out = self.conv2(out)

        return shortcut + out


# 用于上采样的残差块（对特征进行重建和提取，尺寸增大2倍）
class residual_block_up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(residual_block_up, self).__init__()
        self.conv_shortcut = upsample_conv(in_channels, out_channels)  # 先上采样再卷积
        self.conv1 = upsample_conv(in_channels, out_channels)  # 先上采样再卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        out = x
        out = self.bn1(out)
        out = self.ReLU(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.ReLU(out)
        out = self.conv2(out)

        return shortcut + out


# 判断图片的视角，提取特征
class Gen_x_view_feature(nn.Module):
    def __init__(self):
        super(Gen_x_view_feature, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1)  # 3*128*128 --> 64*128*128
        self.resBlock0 = residual_block_down(64, 64)  # 64*128*128 --> 64*64*64
        self.resBlock1 = residual_block_down(64, 128)  # 64*64*64 --> 128*32*32
        self.resBlock2 = residual_block_down(128, 256)  # 128*32*32 --> 256*16*16
        self.resBlock3 = residual_block_down(256, 512)  # 256*16*16 --> 512*8*8
        self.resBlock4 = residual_block_down(512, 512)  # 512*8*8 --> 512*4*4
        self.fc_view = nn.Linear(512*4*4, view_size)
        self.fc_feature = nn.Linear(512*4*4, feature_size)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.resBlock0(out)
        out = self.resBlock1(out)
        out = self.resBlock2(out)
        out = self.resBlock3(out)
        out = self.resBlock4(out)

        out = out.view(-1, 512*4*4)
        view = self.fc_view(out)
        view = self.softmax(view)  # 视角分类
        feature = self.fc_feature(out)  # 特征向量

        return view, feature


# 根据视角与特征生成图片
class Gen_view_feature_x(nn.Module):
    def __init__(self):
        super(Gen_view_feature_x, self).__init__()
        # in_channels, out_channels, kernel_size, stride, padding
        # H_out = (H_in-1)*stride[0] - 2*padding[0] + kernel_size[0] + output_padding[0]
        # W_out = (W_in-1)*stride[1] - 2*padding[1] + kernel_size[1] + output_padding[1]
        self.fc = nn.Linear(view_size + feature_size, 4*4*512)
        self.resBlock1 = residual_block_up(512, 512)  # 4*4-->8*8
        self.resBlock2 = residual_block_up(512, 256)  # 8*8-->16*16
        self.resBlock3 = residual_block_up(256, 128)  # 16*16-->32*32
        self.resBlock4 = residual_block_up(128, 64)  # 32*32-->64*64
        self.resBlock5 = residual_block_up(64, 64)  # 64*64*64-->64*128*128
        self.bn = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()
        self.conv = nn.Conv2d(64, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, view, feature):
        x = torch.cat((view,feature), 1)
        out = self.fc(x) # out: 512*4*4
        out = out.view(-1, 512, 4, 4) # (-1, 512, 4, 4)
        out = self.resBlock1(out)
        out = self.resBlock2(out)
        out = self.resBlock3(out)
        out = self.resBlock4(out)
        out = self.resBlock5(out)
        out = self.bn(out)
        out = self.ReLU(out)
        out = self.conv(out)  # 64*128*128 --> 3*128*128
        out = self.tanh(out)

        return out


class Dis_x_view_score(nn.Module):
    def __init__(self):
        super(Dis_x_view_score, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1)
        self.resBlock0 = residual_block_down(64, 64)  # 64*128*128 --> 64*64*64
        self.resBlock1 = residual_block_down(64, 128)  # 64*64*64 --> 128*32*32
        self.resBlock2 = residual_block_down(128, 256)  # 128*32*32 --> 256*16*16
        self.resBlock3 = residual_block_down(256, 512)  # 256*16*16 --> 512*8*8
        self.resBlock4 = residual_block_down(512, 512)  # 512*8*8 --> 512*4*4
        self.fc_view = nn.Linear(512*4*4, view_size)
        self.fc_csore = nn.Linear(512*4*4, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv(x)
        x = self.resBlock0(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)
        x = x.view(-1, 512*4*4)
        view = self.fc_view(x)
        view = self.softmax(view)
        score = self.fc_csore(x)

        return view, score
