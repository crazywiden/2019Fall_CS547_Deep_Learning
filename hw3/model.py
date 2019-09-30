import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

class CNN2(nn.Module):
    def __init__(self, num_class):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_class)
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x


class CNN(nn.Module):
    """this archtecture is very bad"""
    def __init__(self, num_class=10):
        super(CNN, self).__init__()
        # input image size should be 32 by 32
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=2), # image size goes to 33 by 33
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2), # image size goes to 34 by 34
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # image size goes to 17 by 17
            nn.Dropout(0.1),

            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2), # image size goes to 18 by 18
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2), # image size goes to 16 by 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # image size goes to 8 by 8
            nn.Dropout(0.1)
            )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2), # image size goes to 8 by 8
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # image size goes to 6 by 6
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # image size goes to 4 by 4
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # image size goes to 2 by 2
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout(0.1)
            )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64*4*4, out_features=500, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=500, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=num_class, bias=True)
            )


    def forward(self, X):
        # single image in X should be 32 by 32
        out = self.cnn_layer1(X)
        out = self.cnn_layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out