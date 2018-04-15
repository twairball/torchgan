
import torch
import os

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LSGAN(object):
    def __init__(self, opt):
        """
        Least-Squares GAN
        (ref: ____)

        opts:
            z_dim: int, generator input noise dimension. Default=62
            image_shape: tuple of int with shape (3,1). Default=(3,64,64)  
            batch_size: int. Default=32
            y_dim: int, discriminator output dimension. Default=1
            g_lr: float, generator learning rate. Default=0.0002
            d_lr: float, discriminator learning rate. Default=0.0002
            beta1: float, Adam beta1 param. Default=0.5
            beta2: float, Adam beta2 param. Default=0.999
            

        """
        self.opt = opt
        self.D = discriminator(opt.y_dim, opt.image_shape)
        self.G = generator(opt.z_dim, opt.image_shape)

        if opt.cuda:
            self.D.cuda()
            self.G.cuda()

        self.optimizerD = optim.Adam(self.D.parameters(), lr=opt.g_lr, betas=(opt.beta1, opt.beta2))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.d_lr, betas=(opt.beta1, opt.beta2))
    
    def sample_noise(self):
        z = Variable(torch.randn(self.opt.batch_size, self.opt.z_dim))
        if self.opt.cuda:
            z = z.cuda()
        return z

    def train_step(self, train_iter, d_steps=3):
        """
        Train for a single step
        
        Args:
            train_iter: iterator that returns (images, labels) of type Tensor.
            d_steps: num of steps to train Discriminator before training Generator. Default=3
        
        Returns:
            D_loss: Discriminator loss, Tensor.
            G_loss: Generator loss, Tensor. 
        """
        # discriminator
        for _ in range(d_steps):
            z = self.sample_noise()

            X, Y = train_iter.next() # return tensors
            X = Variable(X)

            # Dicriminator
            G_sample = self.G(z)
            D_real = self.D(X)
            D_fake = self.D(G_sample)

            # least squares loss
            D_loss = 0.5 * (torch.mean((D_real - 1)**2) + torch.mean(D_fake**2))

            # update
            D_loss.backward()
            self.optimizerD.step()
            self.D.zero_grad()
            self.G.zero_grad()

        # generator
        z = self.sample_noise()

        G_sample = self.G(z)
        D_fake = self.D(G_sample)

        # least squares loss
        G_loss = 0.5 * torch.mean((D_fake - 1)**2)

        # update 
        G_loss.backward()
        self.optimizerG.step()
        self.D.zero_grad()
        self.G.zero_grad()

        return D_loss, G_loss
        
class generator(nn.Module):
    def __init__(self, z_dim=62, image_shape=(3, 64, 64)):
        super(generator, self).__init__()

        self.image_shape = image_shape
        c, h, w = self.image_shape

        # project input noise vector
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (h // 4) * (w // 4)),
            nn.BatchNorm1d(128 * (h // 4) * (w // 4)),
            nn.ReLU(),
        )

        # deconv to image 
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, c, 4, 2, 1),
            nn.Sigmoid(),
        )
        initialize_weights(self)
    
    def forward(self, noise):
        c, h, w = self.image_shape

        x = self.fc(noise)
        x = x.view(-1, 128, (h // 4), (w // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):

    def __init__(self, y_dim=1, image_shape=(3, 64, 64)):
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

    def forward(self, x):
        c, h, w = self.image_shape

        f = self.conv(x)
        f = f.view(-1, 128 * (h // 4) * (w // 4))
        f = self.fc(f)

        return f

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
