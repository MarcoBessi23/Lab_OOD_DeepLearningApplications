import torch.nn.functional as F
from torch import nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import os






class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size= 3, dilation=1, padding = 1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size= 3, dilation=1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size= 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size= 5, dilation=1, padding = 1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size= 5, dilation=1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(3200,64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x