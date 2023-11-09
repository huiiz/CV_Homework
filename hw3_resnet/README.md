# 计算机视觉第三次作业

## 要求

- 利用 Fine-tuning 完成图像分类
- ResNet-50 模型
- 数据集：Scene categories （15 类）
- 评估预测结果、并绘制混淆矩阵
  https://figshare.com/articles/dataset/15-Scene_Image_Dataset/7007177
  https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

## 程序介绍

- 数据集划分 data_split.py
  将数据分为训练集与测试集

- 模型 res50_model.py
  使用预训练的 resnet

- 训练 train.py
  训练网络

- 测试 predict.py
  测试网络
