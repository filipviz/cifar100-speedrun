# %%
from dataclasses import dataclass, field
import os
import argparse

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from jaxtyping import Float
from torchvision.transforms import v2
from torchvision.datasets import CIFAR100
from einops import einsum, rearrange

# %% [markdown]
"""
A bottleneck ResNet for CIFAR-100 based on [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385). Also see the notes in [`01-resnet.py`](scripts/01-resnet.py).

### ImageNet Implementation (Sec 3.4)

- Resize with shorter side $\in [256, 480]$ for scale augmentation then take a 224x224 crop.
- Random horizontal flipping, *per-pixel* mean subtracted, standard color augmentation as used in [AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html).
- SGD with `momentum=0.9, weight_decay=1e-4`. 0.1 learning rate, divided by 10 when error plateaus. Trained for 60e4 iterations with a mini-batch size of 256. [He initialization](https://arxiv.org/abs/1502.01852), trained from scratch.
- No dropout.
- Fully-convolutional form w/ 10-crop testing, with scores averaged across scales (shorter side $\in \{224,256,384,480,640\}$).
- See table 1 for architectural details at different scales.
- Section 4.1 introduces more efficient bottleneck architectures. Rather than
  using two consecutive 3x3 convolutions per block with a fixed number of
  channels, they reduce the number of channels with a 1x1 convolution, apply a
  3x3 convolution, then scale back up with another 1x1 convolution. These are
  used in tandem with parameter-free residual connections for the
  50/101/152-layer ResNets.

### Color Augmentation

From [*ImageNet Classification with Deep Convolutional Neural Networks*](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) sec. 4.1

- Perform PCA on the RGB channels of the training set.
- Add multiples of the principal components with magnitudes proportional to the corresponding eigenvalues times a random variable $\sim N(0, 0.1)$.

### Bottleneck Implementation (sec. 4.1)

- Each residual function is a stack of 3 layers: a 1x1 convolution which reduces the number of channels by some factor, a 3x3 convolution, then another 1x1 convolution which restores the original number of channels.
- They use parameter-free residual connections, with 1x1 convolutions for increasing dimensions (option B). If they used projections everywhere the model size and time complexity would be doubled!
- The caption below table 1 states that downsampling is performed by the first 1x1 convolution with stride=2.

## Our Implementation

CIFAR-100 has the same 32x32 format as CIFAR-10, so we can't use their stride/kernel size approach.
- Rather than applying 7x7 convolution and 3x3 max pooling to the input, we might apply a single 3x3 convolution -> BatchNorm as they did for CIFAR-10.
- We can use 3 rather than 4 groups.
"""
# %% 1. Constants 

BASE_DIR = f"{os.path.dirname(__file__)}/.."
DATA_DIR = f"{BASE_DIR}/data"

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

# %%

@dataclass
class Config:
    # --- Model architecture --- #
    n_classes: int = 100
    block_filters: list[int] = field(default_factory=lambda: [64, 256, 512, 1024])
    block_bottlenecks: list[int] = field(default_factory=lambda: [64, 128, 256])
    block_n_layers: list[int] = field(default_factory=lambda: [3, 4, 6])
    block_strides: list[int] = field(default_factory=lambda: [1, 2, 2])

    # --- Training --- #
    train_batch_size: int = 128
    
# %%

class BottleneckBlock(nn.Module):
    """
    A bottleneck ResNet block as described in sec. 4.1 and illustrated in figure 5.
    """

    def __init__(self, c_in: int, c_out:int, c_bottleneck: int, stride: int = 1):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.c_bottleneck = c_bottleneck
        self.stride = stride
        
        self.conv1 = nn.Conv2d(c_in, c_bottleneck, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(c_bottleneck)

        self.conv2 = nn.Conv2d(c_bottleneck, c_bottleneck, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_bottleneck)
        
        self.conv3 = nn.Conv2d(c_bottleneck, c_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c_out)
        
        self.downsample = c_out != c_in or stride != 1
        self.shortcut = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(c_out),
        ) if self.downsample else nn.Identity()
        
    def forward(
        self,
        x: Float[Tensor, "batch channel height width"]
    ) -> Float[Tensor, "batch channel height width"]:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out)) + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out
    
    def extra_repr(self) -> str:
        return f"in_channels={self.c_in}, out_channels={self.c_out}, bottleneck_channels={self.c_bottleneck}, stride={self.stride}"
    
class BottleneckResnet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        
        # Input layer
        self.conv1 = nn.Conv2d(3, cfg.block_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg.block_filters[0])
        
        # Config-defined number of stages
        self.blocks = nn.Sequential(*[
            self._make_block(c_in, c_out, c_bottleneck, stride, n_layers)
            for c_in, c_out, c_bottleneck, stride, n_layers in
            zip(cfg.block_filters, cfg.block_filters[1:], cfg.block_bottlenecks, cfg.block_strides, cfg.block_n_layers)
        ])
        
        # Output layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cfg.block_filters[-1], cfg.n_classes)
        
        self.apply(self._init_weights)
        
    def _make_block(self, c_in: int, c_out: int, c_bottleneck: int, stride: int, n_layers: int) -> nn.Sequential:
        layers = [BottleneckBlock(c_in, c_out, c_bottleneck, stride)]
        for _ in range(1, n_layers):
            layers.append(BottleneckBlock(c_out, c_out, c_bottleneck))
        return nn.Sequential(*layers)
    
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.blocks(out)
        out = self.pool(out).flatten(start_dim=1)
        return self.fc(out)
# %% Data Augmentation

# Unusually, He et al 2015 normalizes with the per-pixel mean and without std.
CIFAR100_MEAN_PATH = f"{DATA_DIR}/cifar_100_mean.npy"
if not os.path.exists(CIFAR100_MEAN_PATH):
    np_mean = CIFAR100(root=DATA_DIR, train=True, download=True).data.mean(axis=0)
    # Convert to PyTorch-friendly format: float32 scaled to [0, 1] in (C, H, W) format.
    np_mean = np_mean.astype(np.float32) / 255.0
    np_mean = rearrange(np_mean, "h w c -> c h w")
    np.save(CIFAR100_MEAN_PATH, np_mean)
else:
    np_mean = np.load(CIFAR100_MEAN_PATH)

CIFAR100_MEAN = torch.from_numpy(np_mean)

# They also apply ImageNet-style PCA color augmentation. I'm not certain this offers an advantage over v2.ColorJitter.
# TODO: We should try ablating this.
CIFAR100_EIG_PATH = f"{DATA_DIR}/cifar_100_eig.npz"
if not os.path.exists(CIFAR100_EIG_PATH):
    np_cifar = CIFAR100(root=DATA_DIR, train=True, download=True).data
    # Convert to PyTorch-friendly format: float32 scaled to [0, 1].
    np_cifar = rearrange(np_cifar, "b h w c -> (b h w) c").astype(np.float32) / 255.0
    covariance_matrix = np.cov(np_cifar, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    np.savez(CIFAR100_EIG_PATH, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
else:
    npz = np.load(CIFAR100_EIG_PATH)
    eigenvalues, eigenvectors = npz["eigenvalues"], npz["eigenvectors"]

EIGENVALUES = torch.from_numpy(eigenvalues).to(torch.float32)
EIGENVECTORS = torch.from_numpy(eigenvectors).to(torch.float32)

@torch.compile(mode='max-autotune')
def pca_aug_and_norm(x: Float[Tensor, "channel height width"]) -> Float[Tensor, "channel height width"]:
    """Apply PCA color augmentation and subtract the mean."""
    c, h, w = x.shape
    coeffs = torch.empty(c, device=x.device).normal_(std=0.1)
    color_aug = einsum(EIGENVECTORS, EIGENVALUES, coeffs, "c1 c2, c2, c2 -> c1")
    return x - CIFAR100_MEAN + color_aug.view(c, 1, 1) 

train_transform = v2.Compose([
    v2.ToImage(),
    # TODO: Unclear if this will work for CIFAR-100 (32x32).
    # Try replacing with RandomCrop or removing.
    v2.RandomResizedCrop(size=(32, 32), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(pca_aug_and_norm),
])
test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(lambda x: x - CIFAR100_MEAN),
])
