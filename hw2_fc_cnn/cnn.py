"""
Author:      郑辉
StudentID:   23320231154460
Date:        2023-10-27
File:        cnn.py
Description: 计算机视觉课程作业二：MNIST手写数字识别，卷积神经网络
"""
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(1)
        self.log_softmax = nn.LogSoftmax(1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.max_pool(out)
        out = self.dropout1(out)
        out = self.flatten(out)
        out = self.relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.log_softmax(self.fc2(out))
        return out



def train(model, train_loader, optimizer, epoch_count):
    model.train()
    for i, (image, label) in enumerate(train_loader):
        outputs = model(image)
        loss = F.cross_entropy(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(
                f"Train Epoch: {epoch_count} [{i * len(image)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
            outputs = model(image)
            test_loss += F.cross_entropy(outputs, label, reduction='sum').item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

if __name__ == '__main__':
    num_epochs = 10
    batch_size = 256
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

    model = CnnNet()
    summary(model, (1, 28, 28))
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch_count in range(num_epochs):
        train(model, train_loader, optimizer, epoch_count + 1)
        test(model, test_loader)

    if save_model:
        torch.save(model.state_dict(), 'mnist_cnn.pt')