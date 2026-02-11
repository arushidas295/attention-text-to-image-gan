import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        query = self.query_conv(x).view(B, -1, W * H)
        key = self.key_conv(x).view(B, -1, W * H)
        energy = torch.bmm(query.permute(0, 2, 1), key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value_conv(x).view(B, -1, W * H)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        return self.gamma * out + x

