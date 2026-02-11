import torch
import torch.nn as nn
from models.attention import SelfAttention

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            SelfAttention(256),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
