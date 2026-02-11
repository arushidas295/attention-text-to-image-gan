import torch
import torch.nn as nn
from models.attention import SelfAttention

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

            SelfAttention(128),

            nn.Conv2d(128, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
