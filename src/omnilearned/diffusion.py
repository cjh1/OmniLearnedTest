import torch
import torch.nn as nn
from typing import Dict, Tuple
import torch.nn.functional as F
import numpy as np

class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return x.unsqueeze(-1)*y.to(x.dtype)

@torch.compile
def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20., shift = 1.):
    b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
    a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - b
    return -2. * torch.log(torch.tan(a * t + b)*shift)

#@torch.compile
def get_logsnr_alpha_sigma(time,shift=1.):
    logsnr = logsnr_schedule_cosine(time,shift=shift)[:,None,None]
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return logsnr, alpha, sigma


def perturb(x,mask,time):
    eps = torch.randn_like(x)  # eps ~ N(0, 1)
    logsnr, alpha, sigma = get_logsnr_alpha_sigma(time)
    z = alpha * x + eps * sigma
    v = alpha*eps - sigma*x
    return z*mask, v*mask
