"""
Author:      郑辉
StudentID:   23320231154460
Date:        2023-11-08
File:        train.py
Description: 计算机视觉课程作业二：利用Fine-tuning完成图像分类
"""

import torch
from torch import nn
from torchvision import transforms
import torchvision
from res50_model import resnet50_model
from torch.utils.tensorboard import SummaryWriter


epochs = 20
batch_zise = 64
num_workers = 2
learning_rate = 0.0001
weight_decay = 0.0001


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(3407)
writer = SummaryWriter('logs')

best_acc = 0.0

def train(model, trainloader, testloader, criterion, optimizer, epoch):
    running_loss = 0.0
    model.train()
    model.zero_grad()
    for i, data in enumerate(trainloader, 0):
        # 获取输入
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播，后向传播，优化
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if i % 20 == 19:
            print(
                f'Epoch:{epoch+1}, {i+1} / {len(trainloader)} loss: {(running_loss/20):.3f}')
            running_loss = 0.0
        writer.add_scalar('training loss', loss.item(),
                          epoch * len(trainloader) + i)

    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for data in testloader:

            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(
            f'Accuracy of the network on the test images:{100 * correct / total:.3f}%')
        writer.add_scalar('test accuracy', 100 * correct / total, epoch)
    
    global best_acc
    if 100 * correct / total > best_acc:
        best_acc = 100 * correct / total
        PATH = './resnet50_finetuning_best.pth'
        torch.save(model.state_dict(), PATH)
    # 保存模型
    PATH = './resnet50_finetuning_last.pth'
    torch.save(model.state_dict(), PATH)


def main():
    model = resnet50_model()
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 设置训练和测试的数据集和数据加载器
    transform = transforms.Compose([
        transforms.Resize(256),  # 缩放图片，保持长宽比不变，最短边为256像素
        transforms.CenterCrop(224),  # 从图片中间裁剪224x224的图片
        transforms.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])  # 标准化至[-1, 1]，规定均值和标准差
    ])

    # Scene categories 数据集
    trainset = torchvision.datasets.ImageFolder(
        root='./dataset/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_zise, shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.ImageFolder(
        root='./dataset/test', transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_zise, shuffle=False, num_workers=num_workers)

    # 训练模型
    for epoch in range(epochs):  # loop over the dataset multiple times
        train(model, trainloader, testloader, criterion, optimizer, epoch)

    print('Finished Training')



if __name__ == '__main__':
    main()
