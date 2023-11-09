"""
Author:      郑辉
StudentID:   23320231154460
Date:        2023-11-05
File:        data_split.py
Description: 计算机视觉课程作业二：利用Fine-tuning完成图像分类
"""
import os
import shutil
import random

random.seed(0)
# Scene categories 数据集类别
categories = {
    '00': 'Bedroom',
    '01': 'Suburb',
    '02': 'Industrial',
    '03': 'Kidchen',
    '04': 'LivingRoom',
    '05': 'Coast',
    '06': 'Forest',
    '07': 'Highway',
    '08': 'InsideCity',
    '09': 'Mountain',
    '10': 'Opencountry',
    '11': 'Street',
    '12': 'TallBuilding',
    '13': 'Office',
    '14': 'Store'
}

# 数据集路径
data_dir = './15-Scene'
to_dataset = './dataset'

if not os.path.exists(to_dataset):
    os.makedirs(to_dataset)

# 划分训练集和测试集
for id, cate in categories.items():
    # 训练集
    train_dir = os.path.join(to_dataset, 'train', cate)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    # 测试集
    test_dir = os.path.join(to_dataset, 'test', cate)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # 划分训练集和测试集
    img_dir = os.path.join(data_dir, id)
    img_list = random.sample(os.listdir(img_dir), len(os.listdir(img_dir)))
    train_list = img_list[:int(len(img_list)*0.8)]
    test_list = img_list[int(len(img_list)*0.8):]

    # 复制图片到指定目录
    for img in train_list:
        img_path = os.path.join(img_dir, img)
        dst_path = os.path.join(train_dir, img)
        shutil.copy(img_path, dst_path)

    for img in test_list:
        img_path = os.path.join(img_dir, img)
        dst_path = os.path.join(test_dir, img)
        shutil.copy(img_path, dst_path)

print('Done!')
