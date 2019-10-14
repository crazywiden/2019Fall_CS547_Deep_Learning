import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # because at the end we need to add residual to the original tensor
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
    def __init__(self, basic_block, num_blocks, num_class):
        super(ResNet, self).__init__()
        self.input_size = 64

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.layer1 = self._make_layer(basic_block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(basic_block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(basic_block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(basic_block, 512, num_blocks[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # not sure whether this works for dimension
        self.fc = nn.Linear(512 * basic_block.expansion, num_class)

    def _make_layer(self, block, num_filter, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
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
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    a = torch.rand((1, 3, 64, 64))
    model = ResNet(Block, [3, 4, 6, 3], 200)
    out = model(a)
    print(out.size())
