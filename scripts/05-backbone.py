# %%
from datetime import datetime
import subprocess
import sys
import os
import time
from dataclasses import dataclass, field
from typing import Literal
import logging

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from einops import rearrange
from torchvision.transforms import v2
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from jaxtyping import Float
from tabulate import tabulate

# %% [markdown]
"""
We continue adding the changes from David Page's *How to Train Your ResNet* series to the [DAWNNet](04-dawnnet.py) implementation.

In [part 3](https://web.archive.org/web/20231207234442/https://myrtle.ai/learn/how-to-train-your-resnet-3-regularisation/), Page:
- Realizes that moving *all* weights to fp16 triggers a slow code path for batchnorms. Moving them back to fp32 fixes this.
- Adds 8x8 cutout (zeroing out random 8x8 patches) to the data augmentation pipeline.
- Increases the batch size to 768.
- Changes the learning rate schedule. The new schedule peaks roughly 25% of the way through training, then linearly decays to 0 until the end of training. With these changes, he can reduce the training duration to 30 epochs.

In [part 4](https://web.archive.org/web/20231108123408/https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/), Page completely updates the architecture!
- He starts by reducing the ResNet down to its shortest path - just the residual stream, trained for 20 epochs.
- This (of course) does not work well, bringing the accuracy down to 55.9%. He's able to bring it back up to 91.1% by:
    - Removing repeated batchnorm-ReLU pairs.
    - Replacing the 1x1 downsampling (`stride=2`) convolutions with `stride=1`, 3x3 convolutions followed by 2x2 max pooling layers.
    - Replacing the concatenated max/average pooling layer with max pooling. To compensate for the reduced input for the final linear layer, he doubles the output size of the final convolution.
    - Using unit initialization for the BatchNorm scale weights (gamma). The PyTorch 0.4 default was random uniform initialization over [0, 1].
    - Scaling the final classifier layer by 0.125.
- He then applies brute force architecture search, finding that adding residual blocks (consisting of two 3x3 convolutions -> batchnorm -> ReLU sequences with identity shortcuts) after the pooling in the first and third layers performs well. With this architecture he achieves 94.08% accuracy in 79s!

Note that we switch to epoch-based scheduling to better match Page's results.
"""

# %% 1. Global Constants

# assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

BASE_DIR = f"{os.path.dirname(__file__)}/.."
DATA_DIR = f"{BASE_DIR}/data"
DEVICE = "cuda"
DTYPE = torch.float16

LOGGING_COLUMNS = ['step', 'time', 'interval_time', 'lr', 'train_loss', 'train_acc1',
                   'train_acc5', 'test_loss', 'test_acc1', 'test_acc5']
HEADER_FMT = "|{:^6s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|"
ROW_FMT = "|{:>6d}|{:>10,.3f}|{:>10,.3f}|{:>10,.3e}|{:>10,.3f}|{:>10.3%}|{:>10.3%}|{:>10,.3f}|{:>10.3%}|{:>10.3%}|"

# %% 2. Configuration and Hyperparameters

@dataclass
class Config:
    
    # --- Model architecture --- #
    n_classes: int = 100
    input_conv_filters: int = 64
    "c_out for the convolution applied to the input."
    group_residual: list[bool] = field(default_factory=lambda: [True, False, True])
    "Whether each group has a residual block."
    fc_scale: float = 0.125
    "We scale the final classifier layer by this amount."
    
    # --- Training --- #
    train_steps: int = 14_000
    "Number of mini-batch iterations we train for."
    eval_every: int = 2_000
    "Set to 0 to disable evaluation."
    save_every: int = 14_000
    "Set to 0 to disable checkpointing. Must be a multiple of eval_every."
    batch_size: int = 512
    momentum: float = 0.9
    "SGD momentum (not BatchNorm momentum)."
    weight_decay: float = 5e-4 * batch_size # this is very aggressive
    lrs: list[float] = field(default_factory=lambda: [0, 0.44, 0.005, 0])
    "We linearly interpolate between these learning rates with np.interp(step, milestones, lrs)."
    milestones: list[int] = field(default_factory=lambda: [0, 6_000, 12_000, 14_001])
    "The number of steps at which we reach each learning rate. Last step is train_steps + 1 to avoid a final step with lr=0.0."
    
    # --- Data Augmentation --- #
    flip: bool = True
    "Random horizontal flipping."
    pad_mode: Literal['reflect', 'constant'] = 'reflect'
    crop_padding: int = 4
    "Set to 0 to disable padding and random cropping."
    
    # --- Setup and Flags --- #
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    float32_matmul_precision: Literal['highest', 'high', 'medium'] = 'medium'
    seed: int = 20250802
    "Set to 0 to disable seeding."
    n_workers: int = 12

    def __post_init__(self):
        assert self.save_every % self.eval_every == 0, "save_every must be a multiple of eval_every"
            
# %% 3. Model

class BackboneBlock(nn.Module):
    
    def __init__(self, c_in: int, c_out: int, residual: bool = False):
        super().__init__()
        self.residual = residual

        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if residual:
            self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(c_out)
            
            self.conv3 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(c_out)
    
    def forward(self, x: Float[Tensor, "batch c_in h_in w_in"]):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.pool(out)
        
        if self.residual:
            res = F.relu(self.bn2(self.conv2(out)), inplace=True)
            res = F.relu(self.bn3(self.conv3(res)), inplace=True)
            out = out + res
        
        return out

class BackboneResnet(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.conv = nn.Conv2d(3, cfg.input_conv_filters, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cfg.input_conv_filters)
        
        n_groups = len(cfg.group_residual)
        c_ins = [
            cfg.input_conv_filters * 2 ** i
            for i in range(n_groups)
        ]

        self.layers = nn.Sequential(*[
            BackboneBlock(c_in, c_out, res) for c_in, c_out, res in 
            zip(c_ins, c_ins[1:], cfg.group_residual)
        ])
        
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(c_ins[-1], cfg.n_classes)
        with torch.no_grad():
            self.fc.weight.mul_(cfg.fc_scale)
    
    def forward(self, x: Float[Tensor, "batch channel height width"]):
        out = F.relu(self.bn(self.conv(x)), inplace=True)
        out = self.layers(out)
        out = self.pool(out).flatten(start_dim=1)
        out = self.fc(out)
        return out

# %%