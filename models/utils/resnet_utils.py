from torch import nn
import numpy as np


def get_kwargs(arch: str, compress_factor: float = 16, decoder: bool = False):
    planes = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]

    if arch == "resnet18":
        kwargs = {"layers": [2, 2, 2, 2], "block": BasicBlock}
    elif arch == "resnet34":
        kwargs = {"layers": [3, 4, 6, 3], "block": BasicBlock}
    elif arch == "resnet50":
        kwargs = {"layers": [3, 4, 6, 3], "block": Bottleneck}
    elif arch == "resnet101":
        kwargs = {"layers": [3, 4, 23, 3], "block": Bottleneck}
    elif arch == "resnet152":
        kwargs = {"layers": [3, 8, 36, 3], "block": Bottleneck}
    elif arch == "resnext50_32x4d":
        kwargs = {"layers": [3, 4, 6, 3], "block": Bottleneck, "groups": 32, "width_per_group": 4}
    elif arch == "resnext101_32x8d":
        kwargs = {"layers": [3, 4, 23, 3], "block": Bottleneck, "groups": 32, "width_per_group": 8}
    elif arch == "wide_resnet50_2":
        kwargs = {"layers": [3, 4, 6, 3], "block": Bottleneck, "width_per_group": 64 * 2}
    elif arch == "wide_resnet101_2":
        kwargs = {"layers": [3, 4, 23, 3], "block": Bottleneck, "width_per_group": 64 * 2}
    else:
        raise ValueError(f"Unknown architecture {arch}")

    compress_list = np.cumprod(strides) * 4
    truncate_idx = compress_list.searchsorted(compress_factor, side="right")
    kwargs["layers"] = kwargs["layers"][:truncate_idx]
    kwargs["planes"] = planes[:truncate_idx]
    kwargs["strides"] = strides[:truncate_idx]

    if decoder:
        kwargs["layers"] = kwargs["layers"][::-1]
        kwargs["planes"] = kwargs["planes"][::-1]
        kwargs["strides"] = kwargs["strides"][::-1]
        kwargs["block"] = BasicBlockTranspose if kwargs["block"] == BasicBlock else BottleNeckTranspose

    return kwargs


def conv1x1(in_planes, out_planes, stride=1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def convtranspose1x1(in_planes, out_planes, stride=1) -> nn.ConvTranspose2d:
    """1x1 convolution"""
    kernel_size = 3 if stride == 2 else 1
    padding = int((kernel_size - 1) / 2)
    output_padding = stride - 1
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        bias=False,
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def convtranspose3x3(in_planes, out_planes, stride=1, groups=1, dilation=1) -> nn.ConvTranspose2d:
    """3x3 convolution with padding"""
    output_padding = stride - 1
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        output_padding=output_padding,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockTranspose(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1, base_width=64):
        super(BasicBlockTranspose, self).__init__()

        self.conv1 = convtranspose3x3(planes, inplanes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = convtranspose3x3(inplanes, inplanes)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleNeckTranspose(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1, base_width=64):
        super(BottleNeckTranspose, self).__init__()

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(planes * self.expansion, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = convtranspose3x3(width, width, stride, groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, inplanes)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out
