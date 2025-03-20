"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from fling.utils.registry_utils import MODEL_REGISTRY
from typing import Type, Any, Callable, Union, List, Optional


VGG_CONFIGS = {
    'A' : [64,     'M', 128,      'M', 256,          'M', 512,         'M', 512,         'M'],
    "B" : [64,     "M", 128,      "M", 256, 256,     "M", 512, 512,    "M", 512, 512,    "M"],
    'C' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, config, batch_norm, input_channel=3, class_number=100):
        super().__init__()
        self.features = self.make_layers(config, batch_norm=batch_norm)

        # self.classifier = nn.Sequential(
        #     nn.Linear(512, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_class)
        # )

        self.classifier = nn.Sequential(
            # nn.Linear(512, 512),  # 512 * 7 * 7 in the original VGG
            # nn.ReLU(True),
            # nn.BatchNorm1d(512),  # instead of dropout
            nn.Linear(512, class_number),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

    def make_layers(self, cfg, batch_norm=False):
        layers = []

        input_ch = 3
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [nn.Conv2d(input_ch, l, kernel_size=3, padding=1)]

            if batch_norm:
                layers += [nn.BatchNorm2d(l)]

            layers += [nn.ReLU(inplace=True)]
            input_ch = l

        return nn.Sequential(*layers)

@MODEL_REGISTRY.register('vgg11_bn')
def vgg11_bn(**kwargs: Any) -> VGG:
    return VGG(VGG_CONFIGS['A'], batch_norm=True, **kwargs)

@MODEL_REGISTRY.register('vgg13_bn')
def vgg13_bn(**kwargs: Any) -> VGG:
    return VGG(VGG_CONFIGS['B'], batch_norm=True, **kwargs)

@MODEL_REGISTRY.register('vgg16_bn')
def vgg16_bn(**kwargs: Any) -> VGG:
    return VGG(VGG_CONFIGS['D'], batch_norm=True, **kwargs)

@MODEL_REGISTRY.register('vgg19_bn')
def vgg19_bn(**kwargs: Any) -> VGG:
    return VGG(VGG_CONFIGS['E'], batch_norm=True, **kwargs)