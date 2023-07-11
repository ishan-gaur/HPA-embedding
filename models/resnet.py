from collections import OrderedDict
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models

from .utils.effnet_utils import Conv2dNormActivation, ConvTranspose2dNormActivation
from .utils.resnet_utils import conv1x1, get_kwargs, convtranspose1x1


class ResNetEncoder(nn.Module):
    def __init__(
        self, model_name, in_channels=3, out_channels=1024, compress_factor=16
    ):
        super(ResNetEncoder, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1

        kwargs = get_kwargs(model_name, compress_factor)
        block = kwargs["block"]
        planes = kwargs["planes"]
        strides = kwargs["strides"]
        layers = kwargs["layers"]
        self.groups = kwargs["groups"] if "groups" in kwargs else 1
        self.base_width = (
            kwargs["width_per_group"] if "width_per_group" in kwargs else 64
        )

        modules: List[nn.Module] = []

        for i in range(2):
            modules.append(
                Conv2dNormActivation(
                    in_channels if i == 0 else self.inplanes,
                    self.inplanes,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    norm_layer=self._norm_layer,
                    activation_layer=nn.ReLU,
                )
            )

        for i, (stride, layer) in enumerate(zip(strides, layers)):
            modules.append(
                nn.Sequential(
                    *self._make_layer(
                        block,
                        planes[i],
                        layer,
                        stride=stride,
                    )
                )
            )

        modules.append(
            nn.Conv2d(
                self.inplanes,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        self.features = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width
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
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResNetDecoder(nn.Module):
    def __init__(self, model_name, in_channels=1024, out_channels=3, expand_factor=16):
        super(ResNetDecoder, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1

        kwargs = get_kwargs(model_name, expand_factor, decoder=True)
        block = kwargs["block"]
        planes = kwargs["planes"]
        strides = kwargs["strides"]
        layers = kwargs["layers"]
        self.groups = kwargs["groups"] if "groups" in kwargs else 1
        self.base_width = (
            kwargs["width_per_group"] if "width_per_group" in kwargs else 64
        )

        modules: List[nn.Module] = []
        input_planes = [planes[i] * block.expansion for i in range(len(planes))]
        self.inplanes = input_planes[0]

        modules.append(
            nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1)
        )
        modules.append(
            ConvTranspose2dNormActivation(
                self.inplanes,
                self.inplanes,
                kernel_size=3,
                stride=1,
                norm_layer=self._norm_layer,
                activation_layer=nn.ReLU,
            )
        )

        for i, (stride, layer) in enumerate(zip(strides, layers)):
            modules.append(
                self._make_layer(
                    block,
                    planes[i],
                    layer,
                    next_inplanes=planes[i + 1] * block.expansion
                    if i + 1 < len(planes)
                    else planes[i],
                    stride=stride,
                )
            )

        for i in range(2):
            modules.append(
                ConvTranspose2dNormActivation(
                    self.inplanes if i == 0 else planes[-1],
                    planes[-1],
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    norm_layer=self._norm_layer,
                    activation_layer=nn.ReLU,
                )
            )

        modules.append(
            ConvTranspose2dNormActivation(
                planes[-1],
                planes[-1],
                kernel_size=3,
                stride=1,
                norm_layer=self._norm_layer,
                activation_layer=nn.ReLU,
            )
        )

        modules.append(
            nn.Conv2d(planes[-1], out_channels, kernel_size=3, stride=1, padding=1)
        )
        modules.append(nn.Tanh())

        self.features = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, next_inplanes, stride=1):
        norm_layer = self._norm_layer

        layers = []
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                )
            )

        upsample = None
        if stride != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                convtranspose1x1(self.inplanes, next_inplanes, stride),
                norm_layer(next_inplanes),
            )

        layers.append(
            block(next_inplanes, planes, stride, upsample, self.groups, self.base_width)
        )
        self.inplanes = next_inplanes

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
