import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict

def get_model(name, n_classes, n_time):
    if name == "resnet":
        return ResNet(n_classes, n_time)
    elif name == "resnet_256":
        return ResNet(n_classes, n_time, downpool_layers=[False, False, False, False])
    elif name == "resnet_512":
        return ResNet(n_classes, n_time, downpool_layers=[True, False, False, False])
    elif name == "resnet_1024":
        return ResNet(n_classes, n_time, downpool_layers=[True, True, False, False])
    elif name == "resnet_2048":
        return ResNet(n_classes, n_time, downpool_layers=[True, True, True, False])
    elif name == "resnet_4096":
        return ResNet(n_classes, n_time, downpool_layers=[True, True, True, True])
    elif name == "resnet_8192":
        return ResNet(n_classes, n_time, downpool_layers=[True, True, True, True])
    elif name == "resnet_16384":
        return ResNet(n_classes, n_time, downpool_layers=[True, True, True, True])
    elif name == "resnet_big":
        return ResNet(n_classes, n_time, n_layer1=64, n_layer2=128, n_layer3=256)
    else:
        raise ValueError("model with name {} not defined ... ")

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1, downpool=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.downpool = downpool
        

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if self.downpool:
            out = self.maxpool(out)
        out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        
                

        return out


class ResNet(nn.Module):

    def __init__(self, n_classes, n_time, block=BasicBlock, keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=5, n_layer1 = 32, n_layer2 = 32, n_layer3 = 32, n_layer4 = 32, downpool_layers=[True, True, True, True]):
        self.inplanes = 1
        super(ResNet, self).__init__()
        # settings
        pooling_size = (4,2)
        embedding_dim = 128

        self.layer1 = self._make_layer(block, n_layer1, stride=2, drop_rate=drop_rate, downpool=downpool_layers[0])
        self.layer2 = self._make_layer(block, n_layer2, stride=2, drop_rate=drop_rate, downpool=downpool_layers[1])
        self.layer3 = self._make_layer(block, n_layer3, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size, downpool=downpool_layers[2])
        self.layer4 = self._make_layer(block, n_layer4, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size, downpool=downpool_layers[3])

        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.pool = nn.AdaptiveAvgPool2d(pooling_size)

        self.fc1 = nn.Linear(np.prod(pooling_size)*n_layer4, embedding_dim)

        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(embedding_dim, n_classes*n_time)

        self.n_classes = n_classes
        self.n_time = n_time

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1, downpool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, downpool=downpool))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.pool(x)

        # flatten
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x_rep = F.relu(x)
        x = self.dropout(x_rep)
        x = self.fc2(x)

        y_pred = x.view((-1, self.n_classes, self.n_time))

        return y_pred, x_rep
