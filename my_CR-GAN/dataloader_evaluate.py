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

def read_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128,128), Image.ANTIALIAS)
    return img

def get_database_img(img_path):
    tmp = random.randint(0, 1)
    view2 = tmp

    view = views[tmp]

    token = img_path.split('/')
    name = token[-1]
    view_str = token[-2]
        
    token = name.split('_')
    ID = token[0]
    status = token[2]
    bright = token[4]

    if view_str == 'frontal':
        view = '051'
    else:
        view = '110'

    img2_path = '../data/' + ID + '/' + view_str + '/'+ ID + '_01_' + status + '_' + view + '_' + bright + '_crop_128.png'

    # 读取替换后的图片图片并转换成 128*128 大小
    img2 = read_img(img2_path)
    img2 = img2.resize((128,128), Image.ANTIALIAS)
    return view2, img2


class ImageList(data.Dataset):
    def __init__( self, list_file, transform=None, is_train=True, 
                  img_shape=[128,128] ):
        img_list = [line.rstrip('\n') for line in open(list_file)]
        print('total %d images' % len(img_list))

        self.img_list = img_list
        self.transform = transform
        self.is_train = is_train
        self.img_shape = img_shape
        self.transform_img = transforms.Compose([self.transform])

    def __getitem__(self, index):
        # img_name: /home/yt219/data/multi_PIE_crop_128/192/192_01_02_140_07_crop_128.png
        img1_path = self.img_list[index]
        
        img1 = read_img( img1_path )

        if self.transform_img is not None:
            img1 = self.transform_img(img1) # [0,1], c x h x w

        return img1

    def __len__(self):
        return len(self.img_list)
