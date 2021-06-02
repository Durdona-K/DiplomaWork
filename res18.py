# Adjacency Resnet 18 - Unrolled implementation
import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Conv2d, MaxPool2d, Sigmoid, ReLU
from torch.autograd import Variable
import torchvision

import torch.nn.functional as F
import torch.nn as nn
from layers import ALinear, AConv2d

import numpy as np

class AResNet18(nn.Module):
    def __init__(self, num_classes=5,channels=3,tasks=20,keep_prob=1.0):
        super(AResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = AConv2d(channels, 64, 3, 1, 1, datasets=tasks)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(64) for j in range(tasks)])

        self.conv1a = AConv2d(64, 64, kernel_size=3, stride=1, padding=1, datasets=tasks)   # layer1
        self.bn1a = nn.ModuleList([nn.BatchNorm2d(64) for j in range(tasks)])
        self.conv1b = AConv2d(64, 64, kernel_size=3, stride=1, padding=1, datasets=tasks)
        self.bn1b = nn.ModuleList([nn.BatchNorm2d(64) for j in range(tasks)])
        self.conv1c = AConv2d(64, 64, kernel_size=3, stride=1, padding=1, datasets=tasks)
        self.bn1c = nn.ModuleList([nn.BatchNorm2d(64) for j in range(tasks)])
        self.conv1d = AConv2d(64, 64, kernel_size=3, stride=1, padding=1, datasets=tasks)
        self.bn1d = nn.ModuleList([nn.BatchNorm2d(64) for j in range(tasks)])

        self.conv2a = AConv2d(64, 128, kernel_size=3, stride=2, padding=1, datasets=tasks)   # layer2
        self.bn2a = nn.ModuleList([nn.BatchNorm2d(128) for j in range(tasks)])
        self.conv2b = AConv2d(128, 128, kernel_size=3, stride=1, padding=1, datasets=tasks)
        self.bn2b = nn.ModuleList([nn.BatchNorm2d(128) for j in range(tasks)])
        
        self.shortcut_conv2 = AConv2d(64, 128, kernel_size=1, stride=2, datasets=tasks) # shortcut
        self.shortcut_bn2 = nn.ModuleList([nn.BatchNorm2d(128) for j in range(tasks)])

        self.conv2c = AConv2d(128, 128, kernel_size=3, stride=1, padding=1, datasets=tasks)
        self.bn2c = nn.ModuleList([nn.BatchNorm2d(128) for j in range(tasks)])
        self.conv2d = AConv2d(128, 128, kernel_size=3, stride=1, padding=1, datasets=tasks)
        self.bn2d = nn.ModuleList([nn.BatchNorm2d(128) for j in range(tasks)])

        self.conv3a = AConv2d(128, 256, kernel_size=3, stride=2, padding=1, datasets=tasks)   # layer3
        self.bn3a = nn.ModuleList([nn.BatchNorm2d(256) for j in range(tasks)])
        self.conv3b = AConv2d(256, 256, kernel_size=3, stride=1, padding=1, datasets=tasks)
        self.bn3b = nn.ModuleList([nn.BatchNorm2d(256) for j in range(tasks)])
        
        self.shortcut_conv3 = AConv2d(128, 256, kernel_size=1, stride=2, datasets=tasks) # shortcut
        self.shortcut_bn3 = nn.ModuleList([nn.BatchNorm2d(256) for j in range(tasks)])

        self.conv3c = AConv2d(256, 256, kernel_size=3, stride=1, padding=1, datasets=tasks)
        self.bn3c = nn.ModuleList([nn.BatchNorm2d(256) for j in range(tasks)])
        self.conv3d = AConv2d(256, 256, kernel_size=3, stride=1, padding=1, datasets=tasks)
        self.bn3d = nn.ModuleList([nn.BatchNorm2d(256) for j in range(tasks)])

        self.conv4a = AConv2d(256, 512, kernel_size=3, stride=2, padding=1, datasets=tasks)   # layer4
        self.bn4a = nn.ModuleList([nn.BatchNorm2d(512) for j in range(tasks)])
        self.conv4b = AConv2d(512, 512, kernel_size=3, stride=1, padding=1, datasets=tasks)
        self.bn4b = nn.ModuleList([nn.BatchNorm2d(512) for j in range(tasks)])
        
        self.shortcut_conv4 = AConv2d(256, 512, kernel_size=1, stride=2, datasets=tasks) # shortcut
        self.shortcut_bn4 = nn.ModuleList([nn.BatchNorm2d(512) for j in range(tasks)])

        self.conv4c = AConv2d(512, 512, kernel_size=3, stride=1, padding=1, datasets=tasks)
        self.bn4c = nn.ModuleList([nn.BatchNorm2d(512) for j in range(tasks)])
        self.conv4d = AConv2d(512, 512, kernel_size=3, stride=1, padding=1, datasets=tasks)
        self.bn4d = nn.ModuleList([nn.BatchNorm2d(512) for j in range(tasks)])

        self.linear = ALinear(512*1, num_classes, datasets=tasks)
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, AConv2d):
                m.weight.data.normal_(0, 1e-2)
                if m.bias is not None:
                    m.bias.data.normal_(0.5, 1e-2)
            elif isinstance(m, ALinear):
                m.weight.data.normal_(0, 2.0 * 1e-1)
                if m.bias is not None:
                    m.bias.data.normal_(0.5, 1e-2)

    def forward(self, x, task = 0, round_ = False):
        out = F.relu(self.bn1[task](self.conv1(x, dataset=task, round_=round_)))

        out = F.relu(self.bn1a[task](self.conv1a(out, dataset=task, round_=round_))) # blocklayer1
        out = F.relu(self.bn1b[task](self.conv1b(out, dataset=task, round_=round_)))
        out = F.relu(self.bn1c[task](self.conv1c(out, dataset=task, round_=round_)))
        out = F.relu(self.bn1d[task](self.conv1d(out, dataset=task, round_=round_)))
        # print(out.shape)

        out_1 = F.relu(self.bn2a[task](self.conv2a(out, dataset=task, round_=round_))) # blocklayer2
        out_1 = self.bn2b[task](self.conv2b(out_1, dataset=task, round_=round_))
        st = self.shortcut_bn2[task](self.shortcut_conv2(out, dataset=task, round_=round_))
        # print(out_1.shape, st.shape)
        out = F.relu(out_1 + st)
        out = F.relu(self.bn2c[task](self.conv2c(out, dataset=task, round_=round_)))
        out = F.relu(self.bn2d[task](self.conv2d(out, dataset=task, round_=round_)))
        # print(out.shape)

        out_1 = F.relu(self.bn3a[task](self.conv3a(out, dataset=task, round_=round_))) # blocklayer3
        out_1 = self.bn3b[task](self.conv3b(out_1, dataset=task, round_=round_))
        out = F.relu(out_1 + self.shortcut_bn3[task](self.shortcut_conv3(out, dataset=task, round_=round_)))
        out = F.relu(self.bn3c[task](self.conv3c(out, dataset=task, round_=round_)))
        out = F.relu(self.bn3d[task](self.conv3d(out, dataset=task, round_=round_)))
        # print(out.shape)

        out_1 = F.relu(self.bn4a[task](self.conv4a(out, dataset=task, round_=round_))) # blocklayer4
        out_1 = self.bn4b[task](self.conv4b(out_1, dataset=task, round_=round_))
        out = F.relu(out_1 + self.shortcut_bn4[task](self.shortcut_conv4(out, dataset=task, round_=round_)))
        out = F.relu(self.bn4c[task](self.conv4c(out, dataset=task, round_=round_)))
        out = F.relu(self.bn4d[task](self.conv4d(out, dataset=task, round_=round_)))
        # print(out.shape)

        out = F.avg_pool2d(out, 4)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.linear(out, dataset=task, round_=round_)
        # print(out.shape)
        # exit(0)
        return out



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
    expansion = 4

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
    def __init__(self, block, num_blocks, num_classes=10,channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print("Init convs: ", out.shape)
        out = self.layer1(out)
        # print("post layer1: ", out.shape)
        out = self.layer2(out)
        # print("post layer2: ", out.shape)
        out = self.layer3(out)
        # print("post layer3: ", out.shape)
        out = self.layer4(out)
        # print("post layer4: ", out.shape)
        out = F.avg_pool2d(out, 4)
        # print("Post pooling: ", out.shape)
        out = out.view(out.size(0), -1)
        # print("post reshape: ", out.shape)
        out = self.linear(out)
        # print("post linear: ", out.shape)
        # exit()
        return out


def ResNet18(num_classes,channels):
    return ResNet(BasicBlock, [2,2,2,2],num_classes,channels)

