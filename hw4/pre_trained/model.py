import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision

import torch.nn.functional as F


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet18(num_class):
    model = torchvision.models.resnet.ResNet(
        torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(model_zoo.load_url(model_urls["resnet18"], model_dir="./"))
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(model.fc.in_features, num_class)
    return model


class ResNet18(nn.Module):
    def __init__(self, num_class):
        super(ResNet18, self).__init__()
        pre_trained_model = resnet18(num_class)
        self.layer = pre_trained_model

    def forward(self, x):
        out = self.layer(x)
        return out

class Block(nn.Module):
    expansion = 1  # expansion = last_block_channel/first_block_channel

    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        # because at the end we need to add residual to the orginal tensor
        # need to guarantee they have the same shape
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=(3, 3), padding=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.block1(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, basic_block, num_blocks, num_class, input_size=32):
        super(ResNet, self).__init__()
        self.input_size = input_size

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5)
            )
        self.layer1 = self._make_layer(basic_block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(basic_block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(basic_block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(basic_block, 256, num_blocks[3], stride=2)

        self.pool_layer = nn.MaxPool2d(kernel_size=(3, 3))
        self.fc = nn.Linear(256*basic_block.expansion, num_class)

    def _make_layer(self, block, num_filter, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.input_size, num_filter, stride))
            self.input_size = num_filter * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool_layer(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    model = ResNet18(100)
    a = torch.rand((5,3,32,32))
    out = model(a)
    print(out.size())


