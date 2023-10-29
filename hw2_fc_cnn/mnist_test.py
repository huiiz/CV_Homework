from cnn import CnnNet
from fc import FCNet
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import torch

from matplotlib import pyplot as plt


cnn_model = CnnNet()
cnn_model.load_state_dict(torch.load('mnist_cnn.pt'))
cnn_model.eval()

input_size, num_classes = 28 * 28, 10
fc_model = FCNet(input_size, num_classes)
fc_model.load_state_dict(torch.load('mnist_fc.pt'))
fc_model.eval()

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])



# 从MNIST测试集中各挑选4张图像，测试两个模型，使用matplotlib绘制图像
def plot_images_labels_prediction(images, labels, predictions, index, title, num=4):
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    plt.title(title,fontsize=30,loc='center')

    for i in range(0, num):
        ax = plt.subplot(2, 2, i + 1)
        ax.imshow(images[index].squeeze(), cmap='gray')
        title = f"label={labels[index]}"
        if len(predictions) > 0:
            title += f",predict={predictions[index]}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        index += 1
    plt.show()


def test_mnist_cnn():
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=10,
        shuffle=True
    )

    images, labels = next(iter(test_loader))
    # plot_images_labels_prediction(images.numpy(), labels.numpy(), [], 0, 4)

    outputs = cnn_model(images)
    _, predictions = torch.max(outputs, 1)
    print(f'Predicted: ', ' '.join(f'{predictions[j].item()}' for j in range(4)))
    plot_images_labels_prediction(images.numpy(), labels.numpy(), predictions.numpy(), 0, 'CNN model Test',4)


def test_mnist_fc():
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=4,
        shuffle=True
    )

    images, labels = next(iter(test_loader))
    # plot_images_labels_prediction(images.numpy(), labels.numpy(), [], 0, 4)

    images1 = images.reshape(-1, input_size)
    outputs = fc_model(images1)
    _, predictions = torch.max(outputs, 1)
    print(f'Predicted: ', ' '.join(f'{predictions[j].item()}' for j in range(4)))
    plot_images_labels_prediction(images.numpy(), labels.numpy(), predictions.numpy(), 0, 'FC model Test', 4)


if __name__ == '__main__':
    test_mnist_cnn()
    test_mnist_fc()