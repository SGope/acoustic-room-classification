import torch.nn
import torch.nn.functional as F
from torch.nn import init

class AudioClassifier(torch.nn.Module):
    def __init__(self):
        """Initialising model"""
        super().__init__()
        conv_layers = []

        #First layer
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3))
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(8)
        #init.kaiming_normal_(self.conv1.weight, a=0.1)
        #self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        #Second layer
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3))
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm2d(16)
        #init.kaiming_normal_(self.conv2.weight, a=0.1)
        #self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        #Third layer
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm2d(32)
        #init.kaiming_normal_(self.conv3.weight, a=0.1)
        #self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        #Final layer
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.relu4 = torch.nn.ReLU()
        self.bn4 = torch.nn.BatchNorm2d(64)
        #init.kaiming_normal_(self.conv4.weight, a=0.1)
        #self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        #Linear classifier
        self.ap = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = torch.nn.Linear(in_features=64, out_features=7)
        #Wrapping convolutional blocks
        self.conv = torch.nn.Sequential(*conv_layers)

    def forward(self, x):
        """Forward pass computations"""
        x = self.conv(x)

        #Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)

        return x