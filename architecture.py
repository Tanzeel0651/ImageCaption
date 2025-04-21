import torch
import torch.nn as nn
import torch.optim as optim


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet_18(nn.Module):
    def __init__(self, image_channels):
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None

        if stride != 1 or in_channels != out_channels:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
                Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
                Block(out_channels, out_channels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)

        return x

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels)
        )


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # lstm Layers
        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=128)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64)

        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(64, vocab_size)

    def forward(self, enriched_input):
        x, (hn, cn) = self.lstm1(enriched_input)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.linear(self.dropout(x))
        return x




