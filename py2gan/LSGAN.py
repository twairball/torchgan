
import torch
import os

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class generator(nn.Module):
    def __init__(self, z_dim=62, image_shape=(3, 64, 64)):
        super(generator, self).__init__()

        self.image_shape = image_shape
        c, h, w = self.image_shape

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (h // 4) * (w // 4)),
            nn.BatchNorm1d(128 * (h // 4) * (w // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, c, 4, 2, 1),
            nn.Sigmoid(),
        )
        initialize_weights(self)

    def forward(self, input):
        c, h, w = self.image_shape

        x = self.fc(input)
        x = x.view(-1, 128, (h // 4), (w // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):

    def __init__(self, image_shape=(3, 64, 64), y_dim=1):
        super(discriminator, self).__init__()

        self.image_shape = image_shape
        c, h, w = self.image_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (h // 4) * (w // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, y_dim),
            nn.Sigmoid(),
        )
        initialize_weights(self)

    def forward(self, input):
        c, h, w = self.image_shape

        x = self.conv(input)
        x = x.view(-1, 128 * (h // 4) * (w // 4))
        x = self.fc(x)

        return x

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.zero()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
