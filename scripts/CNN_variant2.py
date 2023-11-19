# cnn_variant2.py

import torch
import torch.nn as nn

class CNNVariant2(nn.Module):
    def __init__(self):
        super(CNNVariant2, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            # Experiment with different kernel sizes
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),  # Larger kernel
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            # Experiment with different kernel sizes
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, padding=1),  # Smaller kernel
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )
        self.fc_input_size = 64 * 12 * 12
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.fc_input_size, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)  # Assuming 10 classes, update accordingly
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
