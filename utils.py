# -*-coding:utf-8-*-
# author lyl
import paddle
from paddle.vision import transforms
from paddle.io import DataLoader
from matplotlib import pyplot as plt
import numpy as np


def mnist_loader(batch_size=64, num_workers=1):
    train_dataset = paddle.vision.datasets.MNIST(
                                                 mode='train',
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor()
                                                 ]))

    test_dataset = paddle.vision.datasets.MNIST(
                                                 mode='test',
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor()
                                                 ]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    return train_loader, test_loader, classes
