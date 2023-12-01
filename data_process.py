# -*- codeing = utf-8 -*-
# @Time : 2021/10/18 15:44
# @Author : 黄子涵
# @File : data_process.py
# @Software: PyCharm
"""
胶囊网络的数据处理阶段
原始数据:
First Set   Normal 89
            OSCC 439
Second Set  Normal 201
            OSCC 495
先将两个数据集合并:
Normal  290
OSCC    934
all     1224
现在对数据集中的少量样本进行数据增强:
1.旋转
2.裁剪
3.填充
4.亮度
使得normal样本达934列，并将其中的20%用作验证数据集，最后输出数据:
./Dataset/Oral Cancer3/train
Normal 747
OSCC 747
./Dataset/Oral Cancer3/test
Normal 187
OSCC 187
"""

import os
import glob
import random
from torchvision import transforms
import math
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
root_path = './Dataset/BreaKHis_Total_dataset'
target_path = './Dataset/BreaKHis_Total_dataset_processed'

# OSCC数据不需要做数据增强，随机按照比例划分选联合验证数据

def data_process(root_path, split_percentage, target_path, type):
    data_path = os.path.join(root_path, type)
    data_path_list = glob.glob(data_path + '/*.png')

    data_num = len(data_path_list)
    # 验证集按照比例向上取整
    test_data_index = random.sample(range(len(data_path_list)), int(math.ceil(data_num * split_percentage)))
    # 遍历一遍将下标在验证集内的图片复制到test文件夹，其余的放到train文件夹
    test_path = os.path.join(target_path, 'test', type)
    train_path = os.path.join(target_path, 'train', type)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    for i in range(data_num):
        if i in test_data_index:
            shutil.copy(data_path_list[i], test_path)
        else:
            shutil.copy(data_path_list[i], train_path)
        print(type + '数据集划分进度:{:.2f}%'.format((i+1)/data_num * 100))
    print(type + '类数据已经处理完成')
    return 0

def augment(root_path, target_numm, type='Normal'):
    data_path = os.path.join(root_path, type)
    data_path_list = glob.glob(data_path + '/*.png')

    data_num = len(data_path_list)

    # 随机抽取需要数据增强的图片路径下标
    aug_data_index = np.random.choice(range(data_num), target_numm - data_num)
    for i, index in enumerate(aug_data_index):
        img_path = data_path_list[index]
        img = Image.open(img_path)
        # 随机确定数据增强的参
        centercrop = random.sample([0.1, 0.2], 1)[0]
        # pad = random.sample(range(40), 1)[0]
        # Rotaion_degree = random.sample([90, 180, 270], 1)[0]
        # img = transforms.functional.rotate(img, Rotaion_degree, expand=True)
        transform = transforms.Compose([transforms.CenterCrop((img.size[1] - int(centercrop * img.size[1]), img.size[0] - int(centercrop * img.size[0]))),
                                       # transforms.Pad(pad, (0, 0, 0)),
                                       # transforms.RandomRotation(degrees=degree, expand=True),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomVerticalFlip(p=0.5)
                                       # transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2), saturation=(0.5, 2))
                                    ])
        img = transform(img)
        point, file_path, file_ending = img_path.split('.')
        save_path = '.' + file_path + f'_aug{i}' + '.' + file_ending
        img.save(save_path)
        print('数据增强已完成:{:.2f}%'.format((i+1)/len(aug_data_index) * 100))
    print('数据增强完毕')
    return 0


# augment(root_path, 934, type='Normal')

# 将原本的数据按比例划分到指定文件的train与test中
data_process(root_path, 0.2, target_path, 'benign')
data_process(root_path, 0.2, target_path, 'malignant')

# 对目标文件中的Normal文件进行扩充，值至与OSCC的数量一致
train_root_path = os.path.join(target_path, 'train')
test_root_path = os.path.join(target_path, 'test')

train_target_num = len(glob.glob(os.path.join(train_root_path, 'malignant') + '/*.png'))
test_target_num = len(glob.glob(os.path.join(test_root_path, 'malignant') + '/*.png'))

augment(train_root_path, train_target_num, type='benign')
augment(test_root_path, test_target_num, type='benign')