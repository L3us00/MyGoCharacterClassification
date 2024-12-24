import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, MeanSquaredError

EPSILON = 0.1
MOMENTUM = 0.1
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
    
class ResNet(nn.Module):
    def __init__(self, input_channel=3, learning_rate=0.01):
        super(ResNet, self).__init__()
        self.conv_block_0 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,
                      out_channels=64,
                      kernel_size=7,
                      padding=3,
                      stride=2,
                      bias=False),
            nn.BatchNorm2d(num_features=64,eps=EPSILON,momentum=MOMENTUM,track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,padding=1,stride=2)
            )
        self.bottleneck_1 = Bottleneck(64,64)
        self.bottleneck_2 = Bottleneck(64,64)
        self.bottleneck_3 = Bottleneck(64,64, stride=2)
        self.residual_3 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=1, stride=2, bias=False)
        self.res_bn_3 = nn.BatchNorm2d(num_features=64,eps=EPSILON,momentum=MOMENTUM,track_running_stats=True)
        self.bottleneck_4 = Bottleneck(64,128)
        self.residual_4 = nn.Conv2d(in_channels=64,out_channels=128, kernel_size=1, bias=False)
        self.res_bn_4 = nn.BatchNorm2d(num_features=128,eps=EPSILON,momentum=MOMENTUM,track_running_stats=True)
        self.bottleneck_5 = Bottleneck(128,128)
        self.bottleneck_6 = Bottleneck(128,128)
        self.bottleneck_7 = Bottleneck(128,128,stride=2)
        self.residual_7 = nn.Conv2d(in_channels=128,out_channels=128, kernel_size=1, stride=2, bias=False)
        self.res_bn_7 = nn.BatchNorm2d(num_features=128,eps=EPSILON,momentum=MOMENTUM,track_running_stats=True)
        self.bottleneck_8 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=256,
                           eps=EPSILON,
                           momentum=MOMENTUM,
                           track_running_stats=True),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=256,
                           eps=EPSILON,
                           momentum=MOMENTUM,
                           track_running_stats=True),
            nn.ReLU()
        )
        self.residual_8 = nn.Conv2d(in_channels=128,out_channels=256, kernel_size=1, bias=False)
        self.res_bn_8 = nn.BatchNorm2d(num_features=256,eps=EPSILON,momentum=MOMENTUM,track_running_stats=True)
        self.bottleneck_9 = Bottleneck(256,256)
        self.bottleneck_10 = Bottleneck(256,256)
        self.bottleneck_11 = Bottleneck(256,256)
        self.bottleneck_12 = Bottleneck(256,256)
        self.bottleneck_13 = Bottleneck(256,256,stride=2)
        self.residual_13 = nn.Conv2d(in_channels=256,out_channels=256, kernel_size=1, stride=2, bias=False)
        self.res_bn_13 = nn.BatchNorm2d(num_features=256,eps=EPSILON,momentum=MOMENTUM,track_running_stats=True)
        self.bottleneck_14 = Bottleneck(256,512)
        self.residual_14 = nn.Conv2d(in_channels=256,out_channels=512, kernel_size=1, bias=False)
        self.res_bn_14 = nn.BatchNorm2d(num_features=512,eps=EPSILON,momentum=MOMENTUM,track_running_stats=True)
        self.bottleneck_15 = Bottleneck(512,512)
        self.bottleneck_16 = Bottleneck(512,512, stride=2)
        self.residual_16 = nn.Conv2d(in_channels=512,out_channels=512, kernel_size=1, stride=2, bias=False)
        self.global_avg_layer = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Sequential(
            nn.Linear(in_features=512,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=1)
        )
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self,x):
        x = self.conv_block_0(x)
        temp = self.bottleneck_1(x)
        x = x + temp

        temp = self.bottleneck_2(x)
        x = x + temp

        temp = self.bottleneck_3(x)
        x = self.residual_3(x)
        x = x + temp
        x = self.res_bn_3(x)

        temp = self.bottleneck_4(x)
        x = self.residual_4(x)
        x = x + temp
        temp = self.res_bn_4(x)

        temp = self.bottleneck_5(x)
        x = x + temp

        temp = self.bottleneck_6(x)
        x = x + temp

        temp = self.bottleneck_7(x)
        x = self.residual_7(x)
        x = x + temp
        x = self.res_bn_7(x)

        temp = self.bottleneck_8(x)
        x = self.residual_8(x)
        x = x + temp
        x = self.res_bn_8(x)

        temp = self.bottleneck_9(x)
        x = x + temp

        temp = self.bottleneck_10(x)
        x = x + temp

        temp = self.bottleneck_11(x)
        x = x + temp

        temp = self.bottleneck_12(x)
        x = x + temp

        temp = self.bottleneck_13(x)
        x = self.residual_13(x)
        x = x + temp
        x = self.res_bn_13(x)

        temp = self.bottleneck_14(x)
        x = self.residual_14(x)
        x = x + temp
        x = self.res_bn_14(x)

        temp = self.bottleneck_15(x)
        x = x + temp

        temp = self.bottleneck_16(x)
        x = self.residual_16(x)
        x = x + temp

        x = self.global_avg_layer(x).squeeze()
        x = self.fc(x)
        
        return x
