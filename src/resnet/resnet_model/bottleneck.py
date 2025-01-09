import torch.nn as nn
from ..constants import EPSILON, MOMENTUM

class Bottleneck(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size = 3, stride = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels, eps=EPSILON, momentum=MOMENTUM, track_running_stats=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=1, stride = stride, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=in_channels, eps=EPSILON, momentum=MOMENTUM, track_running_stats=True)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels, eps=EPSILON, momentum=MOMENTUM, track_running_stats=True)
        self.relu = nn.ReLU()
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self,x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        
        return x