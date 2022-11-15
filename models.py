"""
Code for constructing a ResNet model, used for 12-lead ECG data
author = @Sergi-Andreu

Here only ResNet models are defined, adapted to 1-d signals.

No exhaustive search on the best models for this task has been done.

I have decided to use this sort of models because they seem to be useful for 1d-signal data.
I possess no domain knowledge on ECG data, nor I have done an exhaustive literature study on the field.

Better models may exist. However, I just wanted to implement ones that "do the job".

This script could be extended to contain more models and do a better hyperparameter / model search.

Some works mentioning ResNet models for this sort of signal data:
- "ECG Heartbeat Classification Based on ResNet and Bi-LSTM" <https://iopscience.iop.org/article/10.1088/1755-1315/428/1/012014>
- For ResNet18 and Resnet34 specific models: "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`


This code has been created following the parameters on https://arxiv.org/pdf/1512.03385.pdf , and using multiple repositories
doing this same task (ResNet18 / ResNet34 for 1d data) for reference. 

"""

# Import required libraries
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.nn.functional as F
from torch.distributions.normal import Normal

# Define the convolutional layer used (for simplifying the code)
def conv15x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=stride, padding=7)

# Define the basic block used for the model
class BasicBlock(nn.Module):
    
    expansion = 1
    # The ResNet block has this expansion parameter to act on the number of feature-maps through the conv layer
    # here set to 1 (as in other works)

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm1d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv15x1(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv15x1(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
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

class ResNet(nn.Module):

    def __init__(self, block, layers, num_outputs=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=nn.BatchNorm1d):

        super(ResNet, self).__init__()
        
        self.num_outputs = num_outputs # Number of outputs. 
        # In this specific work it would be the number of labels wanted to be predicted (5)
        self._norm_layer = norm_layer

        self.inplanes = 32

        self.conv1 = nn.Conv1d(12, self.inplanes, kernel_size=15, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(256, num_outputs))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
#             print("Got to downsample place")
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        op = self._forward_impl(x)
        return op


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=False, progress=False,
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [3, 4, 6, 3], pretrained=False, progress=False,
                   **kwargs)
