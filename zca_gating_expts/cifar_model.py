import torch
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


class ConvSigmoid(nn.Module):
    """Modifies an nn.Module from W to MW,
    where W is the original op and M is an orthonormal matrix.
    Projection implemented via a 1x1 convolution."""

    def __init__(
        self, in_channels, out_channels, kernel_size, bias=False, padding=1, mode=0
    ):
        super(ConvSigmoid, self).__init__()
        self.conv_weight = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ).cuda()

        self.conv_gating = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ).cuda()

        self.conv_gating.weight.data = self.conv_weight.weight.data.clone()

        self.w_init = self.conv_weight.weight.data.clone()

        if mode == 1:
            for param in self.conv_weight.parameters():
                param.requires_grad = False
        if mode == 2:
            for param in self.conv_gating.parameters():
                param.requires_grad = False

    def forward(self, x, x_init=None):
        x = torch.sigmoid(self.conv_gating(x)) * self.conv_weight(x)
        return x


class ConvSigmoidNorm(nn.Module):
    """Modifies an nn.Module from W to MW,
    where W is the original op and M is an orthonormal matrix.
    Projection implemented via a 1x1 convolution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=False,
        padding=1,
        stride=1,
        mode=0,
    ):
        super(ConvSigmoidNorm, self).__init__()
        self.conv_weight = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ).cuda()

        self.conv_gating = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ).cuda()

        self.conv_gating.weight.data = self.conv_weight.weight.data.clone()

        self.w_init = self.conv_weight.weight.data.clone()

        if mode == 1:
            for param in self.conv_weight.parameters():
                param.requires_grad = False
        if mode == 2:
            for param in self.conv_gating.parameters():
                param.requires_grad = False

    def forward(self, x, norm_layer, x_init=None):
        x = torch.sigmoid(self.conv_gating(x)) * norm_layer(self.conv_weight(x))
        return x


class ConvSigmoidNormResidual(nn.Module):
    """Modifies an nn.Module from W to MW,
    where W is the original op and M is an orthonormal matrix.
    Projection implemented via a 1x1 convolution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=False,
        padding=1,
        stride=1,
        mode=0,
    ):
        super(ConvSigmoidNormResidual, self).__init__()
        self.conv_weight = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ).cuda()

        self.conv_gating = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ).cuda()

        self.conv_gating.weight.data = self.conv_weight.weight.data.clone()

        self.w_init = self.conv_weight.weight.data.clone()

        if mode == 1:
            for param in self.conv_weight.parameters():
                param.requires_grad = False
        if mode == 2:
            for param in self.conv_gating.parameters():
                param.requires_grad = False

    def forward(self, x, norm_layer, identity, downsample, x_init=None):
        if downsample is not None:
            identity = downsample(identity)
        x = norm_layer(self.conv_weight(x))
        x += identity
        x = torch.sigmoid(self.conv_gating(x)) * x
        return x


class ConvTwoStream(nn.Module):
    """Modifies an nn.Module from W to MW,
    where W is the original op and M is an orthonormal matrix.
    Projection implemented via a 1x1 convolution."""

    def __init__(
        self, in_channels, out_channels, kernel_size, bias=False, padding=1, mode=0
    ):
        super(ConvTwoStream, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ).cuda()

        self.conv_init = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ).cuda()

        if mode != 2:
            self.conv_init.weight.data = self.conv.weight.data.clone()

        for param in self.conv_init.parameters():
            param.requires_grad = False

        if mode == 3:
            for param in self.conv.parameters():
                param.requires_grad = False

        self.mode = mode

    def forward(self, x, x_init=None):
        if x_init is None:
            x_init = x.clone()

        if self.mode == 0:
            x_init = F.relu(self.conv_init(x_init))
            x = F.relu(self.conv(x))

        if self.mode == 1 or self.mode == 2 or self.mode == 3:
            x_init = self.conv_init(x_init)
            x = self.conv(x) * (x_init > 0).detach()
            x_init = F.relu(x_init)

        return x, x_init


class ConvTwoStreamNorm(nn.Module):
    """Modifies an nn.Module from W to MW,
    where W is the original op and M is an orthonormal matrix.
    Projection implemented via a 1x1 convolution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False,
        padding=1,
        mode=0,
    ):
        super(ConvTwoStreamNorm, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ).cuda()

        self.conv_init = nn.Conv2d(
            in_channels,
            out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ).cuda()

        if mode != 2:
            self.conv_init.weight.data = self.conv.weight.data.clone()

        for param in self.conv_init.parameters():
            param.requires_grad = False

        if mode == 3:
            for param in self.conv.parameters():
                param.requires_grad = False

        self.mode = mode

    def forward(self, x, norm_layer, x_init=None):
        if x_init is None:
            x_init = x.clone()

        if self.mode == 0:
            x_init = F.relu(self.conv_init(x_init))
            x = F.relu(norm_layer(self.conv(x)))

        if self.mode == 1 or self.mode == 2 or self.mode == 3:
            x_init = self.conv_init(x_init)
            x = norm_layer(self.conv(x)) * (x_init > 0).detach()
            x_init = F.relu(x_init)

        return x, x_init


class ConvTwoStreamResidual(nn.Module):
    """Modifies an nn.Module from W to MW,
    where W is the original op and M is an orthonormal matrix.
    Projection implemented via a 1x1 convolution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False,
        padding=1,
        mode=0,
    ):
        super(ConvTwoStreamResidual, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ).cuda()

        self.conv_init = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ).cuda()

        if mode != 2:
            self.conv_init.weight.data = self.conv.weight.data.clone()

        for param in self.conv_init.parameters():
            param.requires_grad = False

        if mode == 3:
            for param in self.conv.parameters():
                param.requires_grad = False

        self.mode = mode

    def forward(self, x, norm_layer, identity, downsample, x_init=None):
        if downsample is not None:
            identity = downsample(identity)

        if x_init is None:
            x_init = x.clone()

        if self.mode == 0:
            x_init = F.relu(self.conv_init(x_init))
            x = self.conv(x)
            x += identity
            x = F.relu(x)

        if self.mode == 1 or self.mode == 2 or self.mode == 3:
            x_init = self.conv_init(x_init)
            x = norm_layer(self.conv(x))

            x += identity
            x = x * (x_init > 0).detach()
            x_init = F.relu(x_init)

        return x, x_init


def mCNN_k_g(c=64, num_classes=10, mode=0):  # no Batch Norm
    return nn.Sequential(
        # Prep
        ConvSigmoid(3, c, kernel_size=3, padding=1, bias=True, mode=mode),
        # Layer 1
        ConvSigmoid(c, c * 2, kernel_size=3, padding=1, bias=False, mode=mode),
        nn.MaxPool2d(2),
        # Layer 2
        ConvSigmoid(c * 2, c * 4, kernel_size=3, padding=1, bias=False, mode=mode),
        nn.MaxPool2d(2),
        # Layer 3
        ConvSigmoid(c * 4, c * 8, kernel_size=3, padding=1, bias=False, mode=mode),
        nn.MaxPool2d(2),
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c * 8, num_classes, bias=False),
    )


class mCNN_fixed(nn.Module):
    """Modifies an nn.Module from W to MW,
    where W is the original op and M is an orthonormal matrix.
    Projection implemented via a 1x1 convolution."""

    def __init__(self, c=64, num_classes=10, mode=0):
        super(mCNN_fixed, self).__init__()

        self.mode = mode

        self.conv1 = ConvTwoStream(
            3, c, kernel_size=3, padding=1, bias=True, mode=mode
        ).cuda()

        self.conv2 = ConvTwoStream(
            c, c * 2, kernel_size=3, padding=1, bias=False, mode=mode
        ).cuda()

        self.mp_2 = nn.MaxPool2d(2)
        self.mp_4 = nn.MaxPool2d(4)

        self.conv3 = ConvTwoStream(
            c * 2, c * 4, kernel_size=3, padding=1, bias=False, mode=mode
        ).cuda()

        self.conv4 = ConvTwoStream(
            c * 4, c * 8, kernel_size=3, padding=1, bias=False, mode=mode
        ).cuda()

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(c * 8, num_classes, bias=False)

    def forward(self, x):
        x, x_init = self.conv1(x)
        x, x_init = self.conv2(x, x_init)
        x = self.mp_2(x)
        x_init = self.mp_2(x_init)
        x, x_init = self.conv3(x, x_init)
        x = self.mp_2(x)
        x_init = self.mp_2(x_init)
        x, x_init = self.conv4(x, x_init)
        x = self.mp_2(x)
        # x_init = self.mp_2(x_init)
        x = self.mp_4(x)  # x_init = self.mp_4(x_init)
        # x_init = self.flatten(x_init)
        x = self.flatten(x)

        # x_init = self.linear(x_init)
        x = self.linear(x)

        return x


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
