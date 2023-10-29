"""
Author:      郑辉
StudentID:   23320231154460
Date:        2023-10-27
File:        fc.py
Description: 计算机视觉课程作业二：MNIST手写数字识别，多层全连接神经网络
"""
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary


class FCNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(1)


    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.log_softmax(self.fc3(out))
        return out


def train(model, train_loader, optimizer, epoch_count):
    for i, (image, label) in enumerate(train_loader):
        images = image.reshape(-1, input_size)
        outputs = model(images)
        loss = F.cross_entropy(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(
                f"Train Epoch: {epoch_count} [{i * len(image)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


def test(model, test_loader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
            images = image.reshape(-1, input_size)
            outputs = model(images)
            test_loss += F.cross_entropy(outputs, label, reduction='sum').item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

if __name__ == '__main__':
    input_size = 28 * 28
    num_classes = 10
    num_epochs = 10
    batch_size = 128
    learning_rate = 0.001
    save_model = True

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = FCNet(input_size, num_classes)
    summary(model, (1, 28 * 28))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch_count in range(num_epochs):
        train(model, train_loader, optimizer, epoch_count + 1)
        test(model, test_loader)

    if save_model:
        torch.save(model.state_dict(), 'mnist_fc.pt')