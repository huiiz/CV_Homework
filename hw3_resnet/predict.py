"""
Author:      郑辉
StudentID:   23320231154460
Date:        2023-11-08
File:        predict.py
Description: 计算机视觉课程作业二：利用Fine-tuning完成图像分类
"""
import os
import random
import torch
from torchvision import transforms
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from res50_model import resnet50_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

cates = sorted(list(categories.values()))


def main():
    model = resnet50_model()
    model.load_state_dict(torch.load(
        './resnet50_finetuning_best.pth', map_location=device))

    # 设置训练和测试的数据集和数据加载器
    transform = transforms.Compose([
        transforms.Resize(256),  # 缩放图片，保持长宽比不变，最短边为256像素
        transforms.CenterCrop(224),  # 从图片中间裁剪224x224的图片
        transforms.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])  # 标准化至[-1, 1]，规定均值和标准差
    ])

    # imgs = []
    # 每个分类选择一张图进行预测，并绘制在matplotlib, 一共15张图，分为3行5列
    i = 1
    for cate in cates:
        img_path = os.path.join('./dataset/test', cate)
        img = random.sample(os.listdir(img_path), 1)
        img = os.path.join(img_path, img[0])

        # 读取图片
        img0 = Image.open(img)
        img = img0.convert('RGB')
        # 对图片进行预处理
        img = transform(img)

        with torch.no_grad():
            # 将图片放入模型进行预测
            model.eval()
            img = img.to(device)
            img = torch.unsqueeze(img, dim=0)
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)

            # 将图片放入matplotlib
            ax = plt.subplot(3, 5, i)
            ax.axis('off')
            ax.set_title(f'predicted: {list(cates)[predicted]}\n real: {cate}')
            plt.imshow(img0, cmap='gray')
            print(f'predicted: {list(cates)[predicted]}, real: {cate}')
            i += 1
    plt.show()


if __name__ == '__main__':
    main()
