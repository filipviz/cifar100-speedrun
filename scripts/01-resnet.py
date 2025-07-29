
# %%
import platform
import sys
import os
import time
import uuid
import subprocess
from dataclasses import dataclass, field, fields, MISSING
from typing import Literal, get_origin, get_args
import logging
import argparse

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
import numpy as np
from einops import einsum, rearrange
from torchvision.transforms import v2
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import wandb
from jaxtyping import Float
from tabulate import tabulate

# %% [markdown]
"""
A research script for training ResNets based on [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385). Our goal is to achieve 90% top-5 test accuracy on CIFAR-100 as quickly as possible on a single H100 SXM.

## Residual Blocks (Sec 3.2)

- They use $\mathbf y = \mathcal F(\mathbf x, \{W_i\}) + \mathbf x$. The
  dimensions of $\mathbf x$ and $\mathcal F$ must be equal. If this is not the
  case, we can perform a linear projection $W_s$ by the shortcut connections to
  match the dimension.
- A non-linearity (ReLU) is applied after the addition.
- $\mathcal F$ is flexible, and generally has two or three layers.

## Network Architecture (Sec 3.3)

- Convolutions use 3x3 filters with `padding=1`.
- `c_out = c_in` when feature map size stays constant. When feature map size is
  halved with `stride=2`, `c_out = 2 * c_in`.
- Ends with global average pooling, a fully-connected layer, and softmax to
  produce logits.
- For shortcut connections where `c_in ≠ c_out`, either (a) 2x2 pool/subsample the input tensor and pad with extra zeros or (b) use a 1x1 convolution with `stride=2` for the shortcut connection.

## Bottleneck Implementation for ImageNet (Sec 3.4)

- SGD with `momentum=0.9, weight_decay=1e-4`. 0.1 learning rate, divided by 10 when error plateaus. Trained for 60e4 iterations with a mini-batch size of 256. [He initialization](https://arxiv.org/abs/1502.01852), trained from scratch. No dropout.
- See table 1 for architectural details at different scales.
- Section 4.1 introduces more efficient bottleneck architectures. Rather than using two consecutive 3x3 convolutions per block with a fixed number of channels, they reduce the number of channels (generally by a factor of 4) with a 1x1 convolution, apply a 3x3 convolution, then scale back up with another 1x1 convolution. These are used in tandem with parameter-free residual connections for the 50/101/152-layer ResNets.
    - They use parameter-free residual connections, with 1x1 convolutions for increasing dimensions (option B). If they used projections everywhere the model size and time complexity would be doubled!
    - The caption below table 1 states that downsampling is performed by the first 1x1 convolution with stride=2.
- Train-time augmentation: they resize with shorter side $\in [256, 480]$ for scale augmentation then take a 224x224 crop. They also apply random horizontal flipping, *per-pixel* mean subtracted, and PCA color augmentation as used in [AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html).
    - Described in sec. 4.1 of the AlexNet paper.
    - They perform PCA on the RGB channels of the training set.
    - Add multiples of the principal components with magnitudes proportional to the corresponding eigenvalues times a random variable $\sim N(0, 0.1)$.
- Test-time augmentation: they apply fully-convolutional form w/ 10-crop testing, with scores averaged across scales (shorter side $\in \{224,256,384,480,640\}$).

## Non-Bottleneck Implementation for CIFAR-10 (Sec 4.2)

- A 3x3 convolution is applied to the input. Then a stack of 6n 3x3 convolution
  layers with feature maps of sizes $\in \{32,16,8\}$ is applied, with
  `stride=2` for subsampling layers. They compare $n \in \{3,5,7,9\}$.
- They use downsampling with zero-padding for the shortcut connections (option A from sec. 3.3).
- Trained for 64k iterations with a mini-batch size of 128. 0.1 learning rate,
  divided by 10 after 32k and 48k iterations.
- Train-time augmentation: 4 pixels padded on each side, with a 32x32 crop sampled from the image or its horizontal flip. *Per-pixel* mean subtracted.
- No test-time augmentation.
- They also explore a deeper ResNet with `n=18`. To assist convergence they use
  `lr=0.01` to warm up for 400 iterations. An `n=200` network also converges,
  but overfits and has worse test accuracy.
- Otherwise mirrors the ImageNet implementation.

## Our Implementation

CIFAR-100 has the same 32x32 format as CIFAR-10, so we can largely apply the non-bottleneck approach without modification. We can't apply their stride/kernel size approach for ImageNet (as used in the bottleneck architectures). Rather than applying 7x7 convolution and 3x3 max pooling to the input, we might apply a single 3x3 convolution -> BatchNorm as they did for CIFAR-10. We can also use 3 rather than 4 groups.
"""

# %% 1. Global Constants

BASE_DIR = f"{os.path.dirname(__file__)}/.."
DATA_DIR = f"{BASE_DIR}/data"

LOGGING_COLUMNS = ['step', 'time', 'lr', 'train_loss', 'train_acc1',
                   'train_acc5', 'test_loss', 'test_acc1', 'test_acc5']
HEADER_FMT = "|{:^6s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|"
ROW_FMT = "|{:>6d}|{:>10.3f}|{:>10.3e}|{:>10.3f}|{:>10.3f}|{:>10.3f}|{:>10.3f}|{:>10.3f}|{:>10.3f}|"

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

# %% 2. Configuration and Hyperparameters

@dataclass
class Config:
    
    # --- Model architecture --- #
    n_classes: int = 100
    bottleneck: bool = True
    input_conv_filters: int = 32
    "c_out for the convolution applied to the input."
    group_filters: list[int] = field(default_factory=lambda: [128, 256, 512])
    "Number of filters for the layers in each group."
    group_bottlenecks: list[int] = field(default_factory=lambda: [32, 64, 128])
    "Bottleneck filters for the layers in each group. Ignored if bottleneck=False."
    group_n_blocks: list[int] = field(default_factory=lambda: [3, 4, 6])
    "Number of blocks in each group."
    group_strides: list[int] = field(default_factory=lambda: [1, 2, 2])
    "Strides for the first convolution in each group. A stride of 2 halves the feature map size."
    shortcut: Literal["A", "B"] = "B"
    "Shortcut connection type for downscaling blocks. Option A is parameter-free and uses 2x2 subsampling on the input, then pads with zeros. Option B uses a 1x1 convolution with stride=2."
    
    # --- Training --- #
    train_steps: int = 64_000
    "Number of mini-batch iterations we train for."
    eval_every: int = 4_000
    "Set to 0 to disable evaluation."
    save_every: int = 16_000
    "Set to 0 to disable checkpointing. Must be a multiple of eval_every."
    batch_size: int = 256
    initial_lr: float = 0.1
    momentum: float = 0.9
    "SGD momentum (not BatchNorm momentum)."
    weight_decay: float = 1e-4
    milestones: list[int] = field(default_factory=lambda: [32_000, 48_000])
    "MultiStepLR milestones."
    gamma: float = 0.1
    "MultiStepLR multiplier."
    
    # --- Setup and Flags --- #
    count_flops: bool = True
    warmup_iters: int = 3
    "Set to 0 to disable warmup."
    float32_matmul_precision: Literal['highest', 'high', 'medium'] = 'high'
    autocast_dtype: Literal['bf16', 'fp32'] = 'bf16'
    # No fp16 support yet (would require amp scaling).
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    allow_tf32: bool = True
    seed: int = 20250723
    "Set to 0 to disable seeding."
    device: Literal['cuda', 'mps', 'cpu'] = DEVICE
    memory_format: Literal['preserve_format', 'channels_last'] = 'preserve_format' if DEVICE == 'mps' else 'channels_last'
    "Memory format for our model and data. channels_last improves convolution locality but causes crashes on mps."
    
    # --- Logging --- #
    use_wandb: bool = False
    wandb_project: str | None = None
    "Leave as None to use a default project name."
    wandb_name: str | None = None
    "Leave as None to use a default name."
    
    def __post_init__(self):
        assert (
            len(self.group_filters)
            == len(self.group_n_blocks)
            == len(self.group_strides)
        ), "All group_* lists must be the same length"

        if self.bottleneck:
            assert len(self.group_bottlenecks) == len(self.group_filters), "All group_* lists must be the same length"
        
        # block layers + 1 input layer + 1 output layer.
        layers_per_block = 3 if self.bottleneck else 2
        self.layers = layers_per_block * sum(self.group_n_blocks) + 2
            
        if self.use_wandb:
            if self.wandb_project is None:
                self.wandb_project = f"cifar100-speedrun"
            if self.wandb_name is None:
                self.wandb_name = f"resnet{self.layers}{'-bottleneck' if self.bottleneck else ''}"
            
        assert self.save_every % self.eval_every == 0, "save_every must be a multiple of eval_every"

# %% 3. Model

class BasicBlock(nn.Module):
    """
    A ResNet building block as described in sec. 3.2. Two Conv2d -> BatchNorm2d
    -> ReLU sequences with a parameter-free residual connection (option A from
    sec. 4.1). In downsampling blocks, we use stride=2 and c_out = 2 * c_in.

    Args:
        c_in: Number of input channels.
        downsample: If true, use a stride of 2 (halves the feature map size)
        and double the number of output channels.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        stride: int = 1,
        shortcut: Literal["A", "B"] = "B",
    ) -> None:
        super().__init__()
        self.c_in, self.c_out, self.stride = c_in, c_out, stride

        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)

        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        
        self.downsample = c_out != c_in or stride != 1
        if not self.downsample:
            self.shortcut = nn.Identity()
        elif shortcut == "A":
            # Option A from sec. 3.3: spatially downsample then pad with zeros along the channel dimension for a parameter-free shortcut.
            pad_chs = c_out - c_in
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=1, stride=stride),
                nn.ZeroPad3d((0, 0, 0, 0, 0, pad_chs)),
            )
        else:
            # Option B from sec. 3.3: use a 1x1 convolution with stride=2.
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(
        self,
        x: Float[Tensor, "batch c_in h_in w_in"],
    ) -> Float[Tensor, "batch c_out h_out w_out"]:
        """
        If this is a downsampling block, c_out = 2*c_in and h_out/w_out are
        half of h_in/w_in respectively. Otherwise, they all match.
        """
        out = self.bn1(self.conv1(x))
        out = F.relu(out, inplace=True)
        out = self.bn2(self.conv2(out)) + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out

    def extra_repr(self) -> str:
        return f"in_channels={self.c_in}, out_channels={self.c_out}, stride={self.stride}"

class BottleneckBlock(nn.Module):
    """
    A bottleneck ResNet block as described in sec. 4.1 and illustrated in figure 5.
    """

    def __init__(
        self,
        c_in: int,
        c_out:int,
        c_bottleneck: int,
        stride: int = 1,
        shortcut: Literal["A", "B"] = "B",
    ) -> None:
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
        if not self.downsample:
            self.shortcut = nn.Identity()
        elif shortcut == "A":
            # Option A from sec. 3.3: spatially downsample then pad with zeros along the channel dimension for a parameter-free shortcut.
            pad_chs = c_out - c_in
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=1, stride=stride),
                nn.ZeroPad3d((0, 0, 0, 0, 0, pad_chs)),
            )
        else:
            # Option B from sec. 3.3: use a 1x1 convolution with stride=2.
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        
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

class ResNet(nn.Module):
    """A non-bottleneck ResNet as described in sec. 4.2"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.bottleneck = cfg.bottleneck
        self.shortcut = cfg.shortcut
        self.layers = cfg.layers

        # Input layer
        self.conv1 = nn.Conv2d(3, cfg.input_conv_filters, 3, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg.input_conv_filters)
        
        # Lists to define groups of blocks.
        c_ins = [cfg.input_conv_filters] + cfg.group_filters
        c_outs = cfg.group_filters
        c_bottlenecks  = cfg.group_bottlenecks if cfg.bottleneck else [None] * len(cfg.group_filters)

        self.groups = nn.Sequential(*[
            self._make_group(c_in, c_out, stride, n_blocks, c_bottleneck)
            for c_in, c_out, stride, n_blocks, c_bottleneck in
            zip(c_ins, c_outs, cfg.group_strides, cfg.group_n_blocks, c_bottlenecks)
        ])
        
        # Output layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cfg.group_filters[-1], cfg.n_classes)
        
        self.apply(self._init_weights)
    
    def _make_group(
        self,
        c_in: int,
        c_out: int,
        stride: int,
        n_blocks: int,
        c_bottleneck: int | None = None,
    ) -> nn.Sequential:
            if self.bottleneck:
                group = [BottleneckBlock(c_in, c_out, c_bottleneck, stride, self.shortcut)]
                for _ in range(1, n_blocks):
                    group.append(BottleneckBlock(c_out, c_out, c_bottleneck, 1, self.shortcut))
            else:
                group = [BasicBlock(c_in, c_out, stride, self.shortcut)]
                for _ in range(1, n_blocks):
                    group.append(BasicBlock(c_out, c_out, 1, self.shortcut))

            return nn.Sequential(*group)
    
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
        
    def forward(
        self,
        x: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, "batch n_classes"]:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.groups(out)
        out = self.pool(out).flatten(start_dim=1)
        return self.fc(out)
    
    def extra_repr(self) -> str:
        return f"resnet{self.layers}-{'bottleneck' if self.bottleneck else ''}, shortcut={self.shortcut}"

# %% 4. Data Augmentation

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

# TODO: We're not applying PCA augmentation here. We should run ablations over different augmentation pipelines.
train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(lambda x: x - CIFAR100_MEAN),
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(lambda x: x - CIFAR100_MEAN),
])

# %% [markdown]
"""
The paper's ImageNet train-time augmentation pipeline would roughly look like this.
TODO: Try out different augmentation pipelines, and especially compare AlexNet-style PCA color augmentation to v2.ColorJitter and mean-pixel normalization against v2.Normalize.

```python
CIFAR100_EIG_PATH = f"{DATA_DIR}/cifar_100_eig.npz"
if not os.path.exists(CIFAR100_EIG_PATH):
    np_cifar = CIFAR100(root=DATA_DIR, train=True, download=True).data
    # Convert to PyTorch-friendly format: float32 scaled to [0, 1].
    np_cifar = rearrange(np_cifar, "b h w c -> (b h w) c").astype(np.float32) / 255.0
    covariance_matrix = np.cov(np_cifar, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    assert eigenvalues.dtype == eigenvectors.dtype == np.float64, \
        "Eigenvalues and eigenvectors should not have imaginary components (covariance matrix is symmetric)"
    np.savez(CIFAR100_EIG_PATH, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
else:
    npz = np.load(CIFAR100_EIG_PATH)
    eigenvalues, eigenvectors = npz["eigenvalues"], npz["eigenvectors"]

EIGENVALUES = torch.from_numpy(eigenvalues).to(torch.float32)
EIGENVECTORS = torch.from_numpy(eigenvectors).to(torch.float32)

@torch.compile(mode='max-autotune')
def pca_aug_and_norm(x: Float[Tensor, "channel height width"]) -> Float[Tensor, "channel height width"]:
    ""Apply PCA color augmentation and subtract the mean.""
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
```
"""

# %% 5. Trainer

class ResNetTrainer():
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.memory_format = torch.preserve_format if cfg.memory_format == 'preserve_format' else torch.channels_last
        self.autocast_dtype = torch.bfloat16 if cfg.autocast_dtype == 'bf16' else torch.float32
        
        # TODO: Experiment with different numbers of workers.
        if self.device.type == 'cuda':
            pin = True
            n_workers = os.cpu_count()
            assert n_workers is not None, "Could not determine number of CPUs with os.cpu_count()"
        else:
            pin = False
            n_workers = 0
        
        if cfg.seed > 0:
            generator = torch.Generator().manual_seed(cfg.seed)           
        else:
            generator = None

        self.train_loader = DataLoader(
            CIFAR100(root=DATA_DIR, train=True, transform=train_transform, download=True),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=n_workers,
            persistent_workers=n_workers > 0,
            drop_last=False,
            pin_memory=pin,
            generator=generator,
        )
        self.test_loader = DataLoader(
            CIFAR100(root=DATA_DIR, train=False, transform=test_transform, download=True),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=n_workers,
            drop_last=False,
            pin_memory=pin,
        )

        self.model = ResNet(cfg).to(self.device, memory_format=self.memory_format)
        if cfg.count_flops:
            self.count_flops() # Profile before compilation.

        if self.device.type != 'mps':
            self.model = torch.compile(self.model, mode="max-autotune")
            self.warmup(cfg.warmup_iters)
        
        if self.cfg.use_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_name,
                config=vars(self.cfg),
            )
            wandb.watch(self.model, log="all")
        
        self.opt = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg.initial_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.opt,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )

        if self.cfg.save_every > 0:
            os.makedirs("checkpoints", exist_ok=True)
        
    def count_flops(self):
        """
        Logs model parameters and forward/backward FLOPs. Must be called after
        self.model and self.train_loader are initialized but before compilation.
        """

        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        bs = self.cfg.batch_size
        batch = torch.rand((bs, 3, 32, 32), device=self.device).to(memory_format=self.memory_format)
        labels = torch.randint(0, self.cfg.n_classes, (bs,), device=self.device)
        
        flop_counter = FlopCounterMode(display=False)
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype), flop_counter:
            self.model(batch)

        shape_str = f"With batch shape {tuple(batch.shape)}"
        logging.info(f"{shape_str}, the forward pass incurs {flop_counter.get_total_flops()} FLOPs.")
        logging.info(flop_counter.get_table())

        temp_opt = torch.optim.SGD(self.model.parameters(), lr=0.0)
        with flop_counter:
            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                out = self.model(batch)
                loss = F.cross_entropy(out, labels)
            loss.backward()
            temp_opt.step()
            temp_opt.zero_grad(set_to_none=True)

        logging.info(f"{shape_str}, a full training step incurs {flop_counter.get_total_flops()} FLOPs.")
        logging.info(flop_counter.get_table())

        # TODO: Does this even do anything?
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()

    def warmup(self, iters: int):
        """Warm up our compiled model."""
        if iters == 0:
            return

        temp_opt = torch.optim.SGD(self.model.parameters(), lr=0.0)
        for _ in range(iters):
            bs = self.cfg.batch_size
            batch = torch.rand((bs, 3, 32, 32), device=self.device).to(memory_format=self.memory_format)
            labels = torch.randint(0, self.cfg.n_classes, (bs,), device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                out = self.model(batch)
                loss = F.cross_entropy(out, labels)
            loss.backward()
            temp_opt.zero_grad(set_to_none=True)

    def train(self) -> dict:
        self.model.train()

        loader_iter = iter(self.train_loader)
        synchronize = (
            torch.cuda.synchronize if self.device.type == 'cuda'
            else torch.mps.synchronize if self.device.type == 'mps'
            else torch.cpu.synchronize # No-op
        )
        pbar = tqdm(range(1, self.cfg.train_steps+1), desc="Training")
        training_time = 0.0
        
        # Start the clock.
        synchronize()
        t0 = time.perf_counter()

        for step in pbar:
            # ---- Training ---- #
            try:
                batch, labels = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                batch, labels = next(loader_iter)
                
            batch = batch.to(self.device, non_blocking=True, memory_format=self.memory_format)
            labels = labels.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                pred = self.model(batch)
                loss = F.cross_entropy(pred, labels)
            
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            # We call scheduler.step() after evaluation.
            # If we stepped here, we'd misreport the learning rate.
            
            # ---- Evaluation ---- #
            last_step = step == self.cfg.train_steps
            if last_step or self.cfg.eval_every > 0 and step % self.cfg.eval_every == 0:
                synchronize()
                training_time += time.perf_counter() - t0
                
                if self.cfg.save_every > 0 and (
                    last_step or step % self.cfg.save_every == 0
                ):
                    torch.save({
                        'model': self.model.state_dict(),
                        'optimizer': self.opt.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'step': step,
                    }, f"checkpoints/resnet{self.cfg.layers}-{step}.pt")
                
                # Clone to avoid CUDA graph issues.
                pred = pred.detach().clone()
                
                self.model.eval()
                test_metrics = self.evaluate()
                self.model.train()
                
                # Our trainining metrics are only estimates (computed on a single batch).
                metrics = {
                    "step": step, 
                    "time": training_time,
                    "lr": self.scheduler.get_last_lr()[0],
                    "train_loss": loss.item(),
                    "train_acc1": (pred.argmax(dim=1) == labels).float().mean().item() * 100,
                    "train_acc5": (pred.topk(5)[1] == labels.view(-1, 1)).any(dim=1).float().mean().item() * 100,
                    **test_metrics,
                }

                logging.info(ROW_FMT.format(*[metrics[col] for col in LOGGING_COLUMNS]))
                if self.cfg.use_wandb:
                    metrics.pop('step')
                    wandb.log(metrics, step)
                
                pbar.set_postfix(train_loss=metrics['train_loss'], test_loss=metrics['test_loss'])
                
                # Start the clock again.
                synchronize()
                t0 = time.perf_counter()

            self.scheduler.step()
            
        synchronize()
        training_time += time.perf_counter() - t0
        logging.info(f"Total training time: {training_time:.2f}s")
        wandb.finish()
    
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        items = len(self.test_loader.dataset)
        cum_loss = torch.tensor(0.0, device=self.device)
        n_correct_top1 = torch.tensor(0.0, device=self.device)
        n_correct_top5 = torch.tensor(0.0, device=self.device)

        pbar = tqdm(self.test_loader, desc="Evaluating", position=1, leave=False)
        for batch, labels in pbar:
            batch = batch.to(self.device, non_blocking=True, memory_format=self.memory_format)
            labels = labels.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                pred = self.model(batch)
                loss = F.cross_entropy(pred, labels, reduction="sum")

            cum_loss += loss
            n_correct_top1 += (pred.argmax(dim=1) == labels).sum()
            n_correct_top5 += (pred.topk(5)[1] == labels.view(-1, 1)).sum()
            
        return {
            "test_loss": cum_loss.item() / items,
            "test_acc1": n_correct_top1.item() / items * 100,
            "test_acc5": n_correct_top5.item() / items * 100,
        }

# %%

def parse_cfg() -> Config:
    """Parse command-line arguments into a Config."""

    def _default(f):
        """Return the run-time default for a dataclass field, honouring default_factory."""
        if f.default is not MISSING:
            return f.default
        if f.default_factory is not MISSING:
            return f.default_factory()
        return None

    parser = argparse.ArgumentParser(description="Train a non-bottleneck ResNet on CIFAR-100.")
    for f in fields(Config):
        origin = get_origin(f.type)
        if f.type in (int, float, str):
            parser.add_argument(f"--{f.name}", type=f.type, default=_default(f))
        elif f.type is bool:
            parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=_default(f))
        elif origin is list:
            elem_type = get_args(f.type)[0] if get_args(f.type) else str
            parser.add_argument(f"--{f.name}", type=elem_type, nargs="+", default=_default(f))
        elif origin is Literal:
            parser.add_argument(f"--{f.name}", type=str, default=_default(f), choices=get_args(f.type))
        else: # Fallback: treat as string.
            parser.add_argument(f"--{f.name}", type=str, default=_default(f))

    return Config(**vars(parser.parse_args()))

if __name__ == "__main__":
    cfg = parse_cfg()

    # Flags
    torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    if cfg.seed > 0:
        torch.manual_seed(cfg.seed) # seeds all devices

    # Set up logging
    run_id = uuid.uuid4()
    os.makedirs(f"{BASE_DIR}/logs", exist_ok=True)
    logging.basicConfig(filename=f"{BASE_DIR}/logs/{run_id}.txt", format="%(message)s", level=logging.INFO)
    
    logging.info(" ".join(sys.argv))
    logging.info(tabulate(vars(cfg).items(), headers=["Config Field", "Value"]))
    logging.info(f"Running Python {sys.version}")
    logging.info(f"Running PyTorch {torch.version.__version__}")
    if cfg.device == 'cuda':
        logging.info(f"Using CUDA {torch.version.cuda} and cuDNN {torch.backends.cudnn.version()}")
    if cfg.device == 'mps':
        release, _, machine = platform.mac_ver()
        logging.info(f"Using mps on MacOS {release} for {machine}.")
        
    # Train our model
    trainer = ResNetTrainer(cfg)
    logging.info(HEADER_FMT.format(*LOGGING_COLUMNS))
    logging.info(HEADER_FMT.format(*['---' for _ in LOGGING_COLUMNS]))
    trainer.train()

    if cfg.device == 'cuda':
        smi = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.info(smi.stdout)
        logging.info(f"Max memory allocated: {torch.cuda.max_memory_allocated() // 1024**2} MiB")
        logging.info(f"Max memory reserved: {torch.cuda.max_memory_reserved() // 1024**2} MiB")
    
    # Write this source code to our logs.
    with open(sys.argv[0]) as f:
        logging.info(f.read())
