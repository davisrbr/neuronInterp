import torch
import torch.nn as nn
import sys
# sys.path.append("/people/brow843/neuronInterp")
sys.path.append("/people/brow843/neuronInterp/gatingExpts")
import math
from cifar_model_oldbn import (
    ConvTwoStream,
    ConvTwoStreamResidual,
    ConvTwoStreamNorm,
    ConvTwoStreamResidualFixBN,
    ConvTwoStreamNormFixBN,
    ConvSigmoidNorm,
    ConvSigmoidNormResidual,
    ConvTwoStreamLearned,
    ConvTwoStreamResidualLearned,
)
from torchinfo import summary


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockFixedBN(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        mode=0,
        gate_width_ratio=10,
    ):
        super(BasicBlockFixedBN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if mode == 4:
            self.conv1_two_stream = ConvTwoStreamLearned(
                inplanes, planes, kernel_size=3, stride=stride, mode=mode, gate_width_ratio=gate_width_ratio
            )
            self.conv2_two_stream = ConvTwoStreamResidualLearned(
                planes, planes, kernel_size=3, mode=mode, stride=1, gate_width_ratio=gate_width_ratio
            )
            self.bn1_init = norm_layer(math.ceil(planes / gate_width_ratio), affine=False)
            self.bn2_init = norm_layer(math.ceil(planes / gate_width_ratio), affine=False)

        else:
            self.conv1_two_stream = ConvTwoStreamNormFixBN(
                inplanes, planes, kernel_size=3, stride=stride, mode=mode
            )
            self.conv2_two_stream = ConvTwoStreamResidualFixBN(
                planes, planes, kernel_size=3, mode=mode, stride=1
            )
            self.bn1_init = norm_layer(planes, affine=False)
            self.bn2_init = norm_layer(planes, affine=False)

        self.bn1_conv = norm_layer(planes)

        self.bn2_conv = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, xs):
        x, x_init = xs
        identity = x
        out, x_init = self.conv1_two_stream(x, self.bn1_conv, self.bn1_init, x_init)
        out, x_init = self.conv2_two_stream(
            out, self.bn2_conv, self.bn2_init, identity, downsample=self.downsample, x_init=x_init
        )

        return out, x_init


class BasicBlockSigmoidGating(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        mode=0,
    ):
        super(BasicBlockSigmoidGating, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1_sigmoid = ConvSigmoidNorm(
            inplanes, planes, kernel_size=3, stride=stride, mode=mode
        )

        self.bn1_conv = norm_layer(planes)
        # self.bn1_init = norm_layer(planes, affine=False)

        self.bn2_conv = norm_layer(planes)
        # self.bn2_init = norm_layer(planes, affine=False)

        self.downsample = downsample
        self.stride = stride
        self.conv2_two_stream = ConvSigmoidNormResidual(
            planes, planes, kernel_size=3, mode=mode, stride=1
        )

    # def forward(self, x, x_init):
    def forward(self, x):
        identity = x
        # out = self.conv1_sigmoid(x, self.bn1_conv, self.bn1_init)
        out = self.conv1_sigmoid(x, self.bn1_conv)
        # out = self.conv2_two_stream(
        #     out, self.bn2_conv, self.bn2_init, identity, downsample=self.downsample
        # )
        out = self.conv2_two_stream(
            out, self.bn2_conv, identity, downsample=self.downsample
        )

        return out


class ResNetFixedBN(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=100,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        mode=0,
        gate_width_ratio=10,
    ):
        super(ResNetFixedBN, self).__init__()

        self.mode = mode
        self.gate_width_ratio = gate_width_ratio

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        if mode == 4:
            self.conv1_two_stream = ConvTwoStreamLearned(
                3, self.inplanes, kernel_size=3, stride=1,
                padding=1, bias=False, mode=mode, gate_width_ratio=self.gate_width_ratio,
                initial_layer=True
            )
            self.bn1_init = norm_layer(math.ceil(self.inplanes / self.gate_width_ratio), affine=False)
        else:
            self.conv1_two_stream = ConvTwoStreamNormFixBN(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False, mode=mode
            )
            self.bn1_init = norm_layer(self.inplanes, affine=False)


        self.bn1_conv = norm_layer(self.inplanes)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(
            block,
            32,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            64,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                try:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                except AttributeError:
                    # BN with affine=False
                    pass

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckFixed):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                self.mode,
                self.gate_width_ratio
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    mode=self.mode,
                    gate_width_ratio = self.gate_width_ratio
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x, x_init = self.conv1_two_stream(x, self.bn1_conv, self.bn1_init)
        # x = self.conv1(x)
        # x = self.bn1(x)

        x, x_init = self.layer1((x, x_init))
        x, x_init = self.layer2((x, x_init))
        x, x_init = self.layer3((x, x_init))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNetSigmoid(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=100,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        mode=0,
    ):
        super(ResNetSigmoid, self).__init__()

        self.mode = mode

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1_sigmoid_norm = ConvSigmoidNorm(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False, mode=mode
        )
        self.bn1_conv = norm_layer(self.inplanes)
        # self.bn1_init = norm_layer(self.inplanes, affine=False)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(
            block,
            32,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            64,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckFixed):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                self.mode,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    mode=self.mode,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # x = self.conv1_sigmoid_norm(x, self.bn1_conv, self.bn1_init)
        x = self.conv1_sigmoid_norm(x, self.bn1_conv)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet_sigmoid(arch, block, layers, mode, **kwargs):
    model = ResNetSigmoid(block, layers, mode=mode, **kwargs)
    return model


def _resnet_fixed(arch, block, layers, mode, **kwargs):
    model = ResNetFixedBN(block, layers, mode=mode, **kwargs)
    return model


def resnet20_fixed(mode=0, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        mode (int): experiment mode
    """
    return _resnet_fixed("resnet20", BasicBlockFixedBN, [3, 3, 3], mode, **kwargs)


def resnet32_fixed(mode=0, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        mode (int): experiment mode
    """
    return _resnet_fixed("resnet32", BasicBlockFixedBN, [5, 5, 5], mode, **kwargs)


def resnet20_sigmoid(mode=0, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        mode (int): experiment mode
    """
    return _resnet_sigmoid(
        "resnet20", BasicBlockSigmoidGating, [3, 3, 3], mode, **kwargs
    )


def resnet32_sigmoid(mode=0, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        mode (int): experiment mode
    """
    return _resnet_sigmoid(
        "resnet32", BasicBlockSigmoidGating, [5, 5, 5], mode, **kwargs
    )


if __name__ == "__main__":
    # test defining a model and passing in a tensor
    test_tens = torch.randn(1, 3, 32, 32, device="cuda")

    # model = resnet20_fixed().to("cuda")
    # model(test_tens)
    # model = resnet20_sigmoid().to("cuda")
    # model(test_tens)
    # model = resnet32_fixed().to("cuda")
    # model(test_tens)
    # model = resnet32_sigmoid().to("cuda")
    # model(test_tens)

    # model = resnet20_fixed(mode=1).to("cuda")
    # model(test_tens)
    # model = resnet32_fixed(mode=1).to("cuda")
    # model(test_tens)

    # model = resnet20_fixed(mode=2).to("cuda")
    # model(test_tens)
    # model = resnet20_sigmoid(mode=2).to("cuda")
    # model(test_tens)
    # model = resnet32_fixed(mode=2).to("cuda")
    # model(test_tens)
    # model = resnet32_sigmoid(mode=2).to("cuda")
    # model(test_tens)

    # model = resnet20_fixed(mode=3).to("cuda")
    # model(test_tens)
    # model = resnet32_fixed(mode=3).to("cuda")
    # model(test_tens)

    model = resnet20_fixed(mode=4, gate_width_ratio=10).to("cuda")
    model(test_tens)
    print(summary(model))
    model = resnet32_fixed(mode=4, gate_width_ratio=10).to("cuda")
    model(test_tens)
