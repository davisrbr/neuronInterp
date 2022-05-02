import torch.nn as nn
import torch.nn.functional as F

# Based on https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/
# https://github.com/davidcpage/cifar10-fast/blob/master/experiments.ipynb
# "Basic Net"


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), x.size(1))


def mCNN_k(c=64, num_classes=10):  # no Batch Norm
    return nn.Sequential(
        # Prep
        nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        # Layer 1
        nn.Conv2d(c, c * 2, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 2
        nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 3
        nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c * 8, num_classes, bias=False),
    )


def mCNN_bn_k(c=64, num_classes=10):
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
        # Layer 1
        nn.Conv2d(c, c * 2, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c * 2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 2
        nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c * 4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 3
        nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c * 8),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Layer 4
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c * 8, num_classes, bias=True),
    )


def mCNN(c=64, num_classes=10):
    return mCNN_bn_k(c, num_classes)


def sCNN_k(c=64, num_classes=10):
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c),
        nn.ReLU(),
        # Layer 1
        nn.Conv2d(c, c * 2, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c * 2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Layer 2
        nn.Conv2d(c * 2, c * 2, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c * 2),
        nn.ReLU(),
        nn.MaxPool2d(4),
        # Layer 3
        nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c * 4),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Layer 4
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c * 4, num_classes, bias=True),
    )
