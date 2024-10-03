# -*- coding: utf-8 -*-

import time
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
import dataloader
from torch.autograd import Variable
from model import Gen_x_view_feature, Gen_view_feature_x, Dis_x_view_score
from itertools import *
import pdb
import matplotlib.pyplot as plt

dd = pdb.set_trace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_list", type=str, default="./list.txt")  # 图片路径+视角列表
    parser.add_argument("-ns", "--nsnapshot", type=int, default=10)  # 间隔几个epoch保存一次模型与结果
    parser.add_argument("-b", "--batch_size", type=int, default=16)  # 一个batch大小
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)  # 学习率
    parser.add_argument("-m" , "--momentum", type=float, default=0.5)  # 第一个动量参数 0.5
    parser.add_argument("-m2", "--momentum2", type=float, default=0.9)  # 第二个动量参数 0.999
    parser.add_argument('--outf', default='./img_out', help='folder to output images and model checkpoints')  # 生成图片保存路径
    parser.add_argument('--modelf', default='./model_out', help='folder to output images and model checkpoints')  # 模型保存路径
    parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)  # 是否使用cuda加速
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')  # 进程数量 TODO:?
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')  # 训练轮数


    # 网络权重初始化
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('LayerNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)


    args = parser.parse_args()
    print(args)

    # 创建输出文件夹
    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    # 用cuda运行
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # 初始化模型与网络权重
    G_xvz = Gen_x_view_feature()
    G_vzx = Gen_view_feature_x()
    D_xvs = Dis_x_view_score()
    G_xvz.apply(weights_init)
    G_vzx.apply(weights_init)
    D_xvs.apply(weights_init)


    train_list = args.data_list
    train_loader = torch.utils.data.DataLoader(  # TODO: transfrom可以改进
        dataloader.ImageList( train_list, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)


    # L1损失函数
    def L1_loss(x, y):
        return torch.mean(torch.sum(torch.abs(x-y), 1))

    v_siz = 2
    z_siz = 128-v_siz
    x1 = torch.FloatTensor(args.batch_size, 3, 128, 128)
    x2 = torch.FloatTensor(args.batch_size, 3, 128, 128)
    v1 = torch.FloatTensor(args.batch_size, v_siz)
    v2 = torch.FloatTensor(args.batch_size, v_siz)
    z = torch.FloatTensor(args.batch_size, z_siz)

    if args.cuda:
        G_xvz = torch.nn.DataParallel(G_xvz).cuda()
        G_vzx = torch.nn.DataParallel(G_vzx).cuda()
        D_xvs = torch.nn.DataParallel(D_xvs).cuda()
        x1 = x1.cuda()
        x2 = x2.cuda()
        v1 = v1.cuda()
        v2 = v2.cuda()
        z = z.cuda()

    x1 = Variable(x1)
    x2 = Variable(x2)
    v1 = Variable(v1)
    v2 = Variable(v2)
    z = Variable(z)

    # 加载预训练模型及其参数
    def load_model(net, path, name):
        state_dict = torch.load('%s/%s' % (path,name))
        own_state = net.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print('not load weights %s' % name)
                continue
            own_state[name].copy_(param)
            print('load weights %s' % name)

    #load_model(G_xvz, args.modelf, 'netG_xvz_epoch_24_699.pth')
    #load_model(G_vzx, args.modelf, 'netG_vzx_epoch_24_699.pth')
    #load_model(D_xvs, args.modelf, 'netD_xvs_epoch_24_699.pth')

    lr = args.learning_rate
    ourBetas = [args.momentum, args.momentum2]
    batch_size = args.batch_size
    snapshot = args.nsnapshot
    start_time = time.time()

    G_xvz_optimizer = optim.Adam(G_xvz.parameters(), lr = lr, betas=ourBetas)
    G_vzx_optimizer = optim.Adam(G_vzx.parameters(), lr = lr, betas=ourBetas)
    D_xvs_optimizer = optim.Adam(D_xvs.parameters(), lr = lr, betas=ourBetas)

    cudnn.benchmark = True

    crossEntropyLoss = nn.CrossEntropyLoss().cuda()

    D_xvs_loss_list = []
    D_x_loss_list = []
    G_xvz_loss_list = []
    G_vzx_loss_list = []

    for epoch in range(args.epochs):

        D_xvs_loss = 0
        D_x_loss = 0
        G_xvz_loss = 0
        G_vzx_loss = 0

        for i, (view1, view2, data1, data2) in enumerate(train_loader):
            # our framework:
            # path 1: (v, z)-->G_vzx-->x_bar--> D_xvs( (v,x_bar), (v,x) )
            # 用判别器D_xvs保证G_vzx可以将任意输入拟合到真实图片空间与视角
            # path 2: x-->G_xvz-->(v_bar, z_bar)-->G_vzx-->x_bar_bar--> D_xvs( (v,x_bar_bar), (v,x) ) + L1_loss(x_bar_bar, x)
            # 保证G_xvz可以正确识别视角与特征后，使得G_vzx生成的图片真实、视角正确，即 G_xvz is the reverse of G_vzx
            eps = random.uniform(0, 1)
            tmp = random.uniform(0, 1)
            reconstruct_fake = False
            if tmp < 0.5:
                reconstruct_fake = True

            D_xvs.zero_grad()
            G_xvz.zero_grad()
            G_vzx.zero_grad()

            img1 = data1
            img2 = data2

            # get x-->real image v--> view and z-->random vector
            x1.data.resize_(img1.size()).copy_(img1)
            x2.data.resize_(img2.size()).copy_(img2)
            v1.data.zero_()
            v2.data.zero_()

            # 将视角信息编码为one-hot向量
            for d in range(view1.size(0)):
                v1.data[d][view1[d]] = 1
            for d in range(view2.size(0)):
                v2.data[d][view2[d]] = 1

            z.data.uniform_(-1, 1) # 此处z为随机向量
            
            # 将v1与v2转换为目标向量
            targetNP = v1.cpu().data.numpy()
            idxs = np.where(targetNP>0)[1]
            tmp = torch.LongTensor(idxs)
            vv1 = Variable(tmp).cuda() # v1 target
            targetNP = v2.cpu().data.numpy()
            idxs = np.where(targetNP>0)[1]
            tmp = torch.LongTensor(idxs)
            vv2 = Variable(tmp).cuda() # v2 target
            
            ## path 1: (v, z)-->G_vzx-->x_bar--> D_xvs( (v,x_bar), (v,x_real) )
            # path 1, update D_xvs 训练判别器
            x_bar = G_vzx(v1, z)  # 用随机向量生成x_bar

            x_hat = eps*x1.data + (1-eps)*x_bar.data  # 将x_bar与img1做插值
            x_hat = Variable(x_hat, requires_grad=True)
            D_x_hat_v, D_x_hat_s = D_xvs(x_hat)

            # 获取D_x_hat_s对x_hat的梯度
            grads = autograd.grad(outputs = D_x_hat_s,
                                inputs = x_hat,
                                grad_outputs = torch.ones(D_x_hat_s.size()).cuda(),
                                retain_graph = True,
                                create_graph = True,
                                only_inputs = True)[0]
            
            # 计算grads的L2范数
            grad_norm = grads.pow(2).sum().sqrt()
            # 梯度Lipschitz惩罚项
            gp_loss = torch.mean((grad_norm - 1) ** 2)  # gradient with v1

            x_bar_loss_v, x_bar_loss_s = D_xvs(x_bar.detach())  # x_bar的得分，冻结G_vzx参数
            x_bar_loss_s = x_bar_loss_s.mean()

            x_loss_v, x_loss_s = D_xvs(x1)  # img1的得分
            x_loss_s = x_loss_s.mean()

            v_loss_x = crossEntropyLoss(x_loss_v, vv1)  # 判别器的视角预测损失
            
            # 判别器的目标是能识别img1的视角，区分真实图片与G_vzx生成的图片
            d_xvs_loss = x_bar_loss_s - x_loss_s + 10. * gp_loss + v_loss_x
            d_xvs_loss.backward()
            D_xvs_optimizer.step()
            D_xvs_loss += d_xvs_loss.item()

            # path 1, update G_vzx 训练生成器G_vzx
            D_xvs.zero_grad()
            G_xvz.zero_grad()
            G_vzx.zero_grad()

            x_bar_loss_v, x_bar_loss_s = D_xvs(x_bar) # x_bar的得分，现在不冻结G_vzx
            x_bar_loss_s = x_bar_loss_s.mean()

            v_loss_x_bar = crossEntropyLoss(x_bar_loss_v, vv1) # x_bar与真实图片的视角损失

            # 生成器G_vzx总损失
            g_vzx_loss = -x_bar_loss_s + v_loss_x_bar
            g_vzx_loss.backward()
            G_vzx_optimizer.step()
            G_vzx_loss += g_vzx_loss.item()

            ## path 2: x-->G_xvz-->(v_bar, z_bar)-->G_vzx-->x_bar_bar--> D_xvs( (v,x_bar_bar), (v,x) ) + L1_loss(x_bar_bar, x)
            # path 2, update D_x 训练判别器
            D_xvs.zero_grad()
            G_xvz.zero_grad()
            G_vzx.zero_grad()

            if reconstruct_fake is True:
                v_bar, z_bar = G_xvz(x_bar.detach())  # 随机图片的特征与视角
                x_bar_bar = G_vzx(v1, z_bar)
                x_hat = eps*x1.data + (1-eps)*x_bar_bar.data # x1 与 x_bar_bar(v1) 的插值
            else:
                v_bar, z_bar = G_xvz(x1) # img1的特征与视角 
                x_bar_bar = G_vzx(v2, z_bar) # 另一视角的img1（即img2_bar）
                x_hat = eps*x2.data + (1-eps)*x_bar_bar.data # x2 与 x_bar_bar(v2) 的差值
            
            x_hat = Variable(x_hat, requires_grad=True)
            D_x_hat_v, D_x_hat_s = D_xvs(x_hat)

            # 计算梯度惩罚项
            grads = autograd.grad(outputs = D_x_hat_s,
                                inputs = x_hat,
                                grad_outputs = torch.ones(D_x_hat_s.size()).cuda(),
                                retain_graph = True,
                                create_graph = True,
                                only_inputs = True)[0]
            grad_norm = grads.pow(2).sum().sqrt()
            gp_loss = torch.mean((grad_norm - 1) ** 2)
            
            x_loss_v, x_loss_s = D_xvs(x2)
            x_loss_s = x_loss_s.mean()
            x_bar_bar_loss_v, x_bar_bar_loss_s = D_xvs(x_bar_bar.detach()) # x_bar_bar 的得分
            x_bar_bar_loss_s = x_bar_bar_loss_s.mean()
            
            v_loss_x = crossEntropyLoss(x_loss_v, vv2) # 判别器对img2的视角损失

            # 判别器的目标是能识别另一视角，区分G_xvz+G_vzx生成的图片（可能是G_vzx从随机图片生成的v1视角A/从img1生成的v2视角B）与真实图片
            # A与B都是对G_xvz特征提取能力的考验，B还需考验G_vzx根据v1的feature生成v2的能力
            d_x_loss = x_bar_bar_loss_s - x_loss_s + 10. * gp_loss + v_loss_x
            d_x_loss.backward()
            D_xvs_optimizer.step()
            D_x_loss += d_x_loss.item()

            # 2st path, update G_xvz
            x_bar_bar_loss_v, x_bar_bar_loss_s = D_xvs(x_bar_bar) # x_bar_bar score
            x_bar_bar_loss_s = x_bar_bar_loss_s.mean()
            
            if reconstruct_fake is True:
                x_l1_loss = L1_loss(x_bar_bar, x_bar.detach())
                v_loss_x_bar_bar = crossEntropyLoss(x_bar_bar_loss_v, vv1) # ACGAN loss of x_bar_bar(v1)
            else:
                x_l1_loss = L1_loss(x_bar_bar, x2) # L1 loss between x_bar_bar and x2
                v_loss_x_bar_bar = crossEntropyLoss(x_bar_bar_loss_v, vv2) # ACGAN loss of x_bar_bar(v2)
            
            v_loss_x = crossEntropyLoss(v_bar, vv1)

            # 对G_xvz，目标是最大化判别器得分，最小化x_bar_bar与原图的图像差距和视角差距，最小化v_bar与vv1的差距（保证视角识别正确）
            g_xvz_loss = -x_bar_bar_loss_s + 4*x_l1_loss + v_loss_x_bar_bar + 0.01*v_loss_x 
            g_xvz_loss.backward()
            G_xvz_loss += g_xvz_loss.item()


            # 若未使用img2，相当于同时训练更新一次G_vzx
            if reconstruct_fake is False:
                G_vzx_optimizer.step()
            
            G_xvz_optimizer.step()

        D_xvs_loss_list.append(D_xvs_loss)
        D_x_loss_list.append(D_x_loss)
        G_xvz_loss_list.append(G_xvz_loss)
        G_vzx_loss_list.append(G_vzx_loss)

        print("Epoch: %2d, time: %4.4f, "
            "loss_D_vx: %.4f, loss_D_x: %.4f, loss_G_xvz: %.4f, loss_G_vzx: %.4f"
            % (epoch, time.time() - start_time,
                D_xvs_loss, D_x_loss, G_xvz_loss, G_vzx_loss))
        if epoch % snapshot == snapshot-1:
            vutils.save_image(x_bar.data,
                            '%s/epoch_%03d_x_bar.png' % (args.outf, epoch),normalize=True)
            vutils.save_image(x_bar_bar.data,
                            '%s/epoch_%03d_x_bar_bar.png' % (args.outf, epoch),normalize=True)
            vutils.save_image(x1.data,
                    '%s/epoch_%03d_x1.png' % (args.outf, epoch),normalize=True)
            vutils.save_image(x2.data,
                    '%s/epoch_%03d_x2.png' % (args.outf, epoch),normalize=True)

            torch.save(G_xvz.state_dict(), '%s/epoch_%d_netG_xvz.pth' % (args.modelf, epoch))
            torch.save(G_vzx.state_dict(), '%s/epoch_%d_netG_vzx.pth' % (args.modelf, epoch))
            torch.save(D_xvs.state_dict(), '%s/epoch_%d_netD_xvs.pth' % (args.modelf, epoch))
    

    # 绘制损失变化
    plt.figure(figsize=(6, 4))
    plt.plot(range(args.epochs), D_xvs_loss_list, label='lossD_xvs')
    plt.plot(range(args.epochs), D_x_loss_list, label='lossD_x')
    plt.plot(range(args.epochs), G_xvz_loss_list, label='lossG_xvz')
    plt.plot(range(args.epochs), G_vzx_loss_list, label='lossD_vzx')
    plt.grid(True)
    plt.legend()
    plt.savefig('%s/G_lossD_vs_lossG.png' % (args.outf))

