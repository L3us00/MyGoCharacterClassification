import torch.nn as nn
from ..constants import EPSILON, MOMENTUM

class Bottleneck(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size = 3, stride = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels, eps=EPSILON, momentum=MOMENTUM, track_running_stats=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride = stride, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=EPSILON, momentum=MOMENTUM, track_running_stats=True)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x