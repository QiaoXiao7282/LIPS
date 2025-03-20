import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from fling.utils.registry_utils import MODEL_REGISTRY
import torch.nn.functional as F

class FedRodHead(nn.Module):

    def __init__(self, input_dim, class_number):
        super(FedRodHead, self).__init__()
        self.fedrod_g_head = nn.Linear(input_dim, class_number)
        self.fedrod_p_head = nn.Linear(input_dim, class_number)

    def forward(self, x):
        g = self.fedrod_g_head(x)
        p = self.fedrod_p_head(x) + g.detach()
        return g, p

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    # def __init__(self, block, num_blocks, num_classes):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            fedrod_head: bool = False,
            features: List[int] = [64, 128, 256, 512],
            linear_hidden_dims: List[int] = [],
            input_channel: int = 3,
            class_number: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  ##nn.AvgPool2d，MaxPool2d

        self.layers = [self._make_layer(block, 64, layers[0], stride=1)]
        for num in range(1, len(layers)):
            self.layers.append(self._make_layer(block, features[num], layers[num], stride=2))

        self.layers = nn.Sequential(*self.layers)
        # self.classifier = nn.Linear(features[len(layers) - 1]*block.expansion, class_number, bias=False)

        if not fedrod_head:
            self.classifier = nn.Linear(features[len(layers) - 1] * block.expansion, class_number)
        else:
            self.fc = FedRodHead(input_dim=features[len(layers) - 1] * block.expansion, class_number=class_number)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, mode: str = 'compute-logit') -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layers(out)
        # out = F.avg_pool2d(out, 4)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(out.size(0), -1)
        y = self.classifier(out)

        if mode == 'compute-logit':
            return y
        elif mode == 'compute-feature-logit':
            return out, y
        else:
            return y

    def forward(self, x: Tensor, mode: str = 'compute-logit') -> Tensor:
        return self._forward_impl(x, mode)


@MODEL_REGISTRY.register('ResNet6')
def ResNet6(**kwargs: Any) -> ResNet:  # 6 = 2 + 2 * (1 + 1)
    return ResNet(BasicBlock, [1, 1], **kwargs)

@MODEL_REGISTRY.register('ResNet4')
def ResNet4(**kwargs: Any) -> ResNet:  # 4 = 2 + 2 * (1)
    return ResNet(BasicBlock, [1], **kwargs)

@MODEL_REGISTRY.register('ResNet8')
def ResNet8(**kwargs: Any) -> ResNet:  # 8 = 2 + 2 * (1 + 1 + 1)
    return ResNet(BasicBlock, [1, 1, 1], **kwargs)

@MODEL_REGISTRY.register('ResNet10')
def ResNet10(**kwargs: Any):
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)

@MODEL_REGISTRY.register('ResNet18')
def ResNet18(**kwargs: Any):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

@MODEL_REGISTRY.register('ResNet34')
def ResNet34(**kwargs: Any):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
