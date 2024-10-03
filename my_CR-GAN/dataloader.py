# Xi Peng, Feb 2017
# Yu Tian, Apr 2017
import os, sys
import numpy as np
from PIL import Image
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pdb

dd = pdb.set_trace

views = ['051', '110'] # 051: frontal; 110: profile


# 读取图片并转换成 128*128 大小
def read_img(img_path):
    # my img_path: ../data/001/frontal/001_01_01_051_00_crop_128.png
    # img_path: ../data/192/192_01_02_140_07_crop_128.png
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128,128), Image.ANTIALIAS)
    return img


# 返回与img_path类型相同而视角随机替换的图片
def get_database_img(img_path):
    # my img_path: ../data/001/frontal/001_01_01_051_00_crop_128.png
    # img_path: ../data/192/192_01_02_140_07_crop_128.png

    token = img_path.split('/')
    name = token[-1]
    view_str = token[-2]
        
    token = name.split('_')
    ID = token[0]
    status = token[2]
    bright = token[4]

    if view_str == 'frontal':
        view = '110'
        view_str = 'profile'
        view2 = 1
    else:
        view = '051'
        view_str = 'frontal'
        view2 = 0

    # TODO: 随机换掉了文件名中的视角部分？为啥子哦    要改路径！！！
    img2_path = '../data/' + ID + '/' + view_str + '/'+ ID + '_01_' + status + '_' + view + '_' + bright + '_crop_128.png'

    # 读取替换后的图片图片并转换成 128*128 大小
    img2 = read_img(img2_path)
    img2 = img2.resize((128,128), Image.ANTIALIAS)
    return view2, img2


class ImageList(data.Dataset):
    def __init__( self, list_file, transform=None, is_train=True, 
                  img_shape=[128,128] ):
        img_ptah_list = [line.rstrip('\n') for line in open(list_file)]
        print('total %d images' % len(img_ptah_list))

        self.img_ptah_list = img_ptah_list
        self.transform = transform
        self.is_train = is_train
        self.img_shape = img_shape
        self.transform_img = transforms.Compose([self.transform])

    # 返回原图片与另一个视角的图片，及其相应视角
    def __getitem__(self, index):
        # img_name: ../data/192/192_01_02_140_07_crop_128.png
        img1_path = self.img_ptah_list[index]
        token = img1_path.split(' ')
        
        img1_fpath = token[0]  # 图片路径
        view1 = int(token[1])  # 图片视角
        
        # 读取图片并转换成 128*128 大小
        img1 = read_img( img1_fpath )

        # 随机替换视角
        view2, img2 = get_database_img(img1_fpath)

        if self.transform_img is not None:
            img1 = self.transform_img(img1) # [0,1], c x h x w
            img2 = self.transform_img(img2)

        return view1, view2, img1, img2

    def __len__(self):
        return len(self.img_ptah_list)
