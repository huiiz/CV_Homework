"""
Author:      郑辉
StudentID:   23320231154460
Date:        2023-11-08
File:        res50_model.py
Description: 计算机视觉课程作业二：利用Fine-tuning完成图像分类
"""
import torch
import torchvision

from torchvision import models
from torch import nn
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(3407)


def resnet50_model():
    # 加载预训练的ResNet50模型，并修改最后一层全连接层
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 15)
    model.to(device)
    return model
