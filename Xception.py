import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
        
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, repetitions, strides=1):
        super(Block, self).__init__()

        self.blocks = nn.Sequential()

        for i in range(repetitions):
            if i == 0:
                self.blocks.add_module(f"Rep{i}", BlockUnit(in_channels, out_channels, strides))
            else:
                self.blocks.add_module(f"Rep{i}", BlockUnit(out_channels, out_channels))

    def forward(self, input):
        return self.blocks(input)

class BlockUnit(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        super(BlockUnit, self).__init__()
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False)
        self.residual_bn = nn.BatchNorm2d(out_channels)

        self.sepConv1 = SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.sepConv2 = SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.maxpool = nn.MaxPool2d(3, strides, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        residual = self.residual_conv(input)
        residual = self.residual_bn(residual)

        x = self.sepConv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.sepConv2(x)
        x = self.bn2(x)

        x = self.maxpool(x)
        
        return self.relu(x + residual)


class MiniXception(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(MiniXception, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 8, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(8)

        self.blocks = nn.Sequential(
            Block(8, 16, 2, 2),
            Block(16, 32, 2, 2),
            Block(32, 64, 2, 2),
            Block(64, 128, 2, 2),
        )
        
        self.last_conv = nn.Conv2d(128, num_classes, kernel_size=3, padding=1, bias=False)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.blocks(x)

        x = self.last_conv(x)
        x = self.global_avg_pool(x).reshape(x.shape[0], -1)
        output = self.softmax(x)

        return output


class BigXception(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(BigXception, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.blocks = nn.Sequential(
            Block(64, 128, 2, 2),
            Block(128, 256, 2, 2),
            Block(256, 728, 2, 2),
            Block(728, 728, 11),
            Block(728, 1024, 2, 2),
        )

        self.conv3 = SeparableConv2d(1024, 1536, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(2048)

        self.last_conv = nn.Conv2d(2048, num_classes, kernel_size=3, padding=1, bias=False)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.blocks(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.last_conv(x)
        x = self.global_avg_pool(x).reshape(x.shape[0], -1)
        output = self.softmax(x)

        return output