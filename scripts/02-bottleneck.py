from datetime import datetime
import random
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
from einops import einsum, rearrange
from torchvision.transforms import v2
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from jaxtyping import Float
from tabulate import tabulate

# %% [markdown]
"""
A script which naively implements bottleneck ResNets based on [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) for CIFAR-100. See the [non-bottleneck implementation](01-resnet.py) for additional context.

## Bottleneck Implementation for ImageNet (Sec 4.1)

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

Otherwise, the implementation is the same as the non-bottleneck implementation.

## Our Implementation

As with our non-bottleneck implementation, we're ignoring *many* straightforward optimizations in order to faithfully implement the paper. We allow tf32 convolutions and matmuls which allow us to leverage the H100's tensor cores with only a marginal loss in accuracy. Additionally:

1. Since CIFAR-100 has the same 32x32 format as CIFAR-10, we can't apply the 7x7 convolution  and 3x3 max pooling they apply to the input. We'll use the size-preserving convolution they apply to CIFAR-10.
2. We're not applying the test-time augmentation the paper uses for ImageNet.

"""
# %% 1. Global Constants

assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

BASE_DIR = f"{os.path.dirname(__file__)}/.."
DATA_DIR = f"{BASE_DIR}/data"

LOGGING_COLUMNS = ['step', 'time', 'iter_time', 'lr', 'train_loss', 'train_acc1',
                   'train_acc5', 'test_loss', 'test_acc1', 'test_acc5']
HEADER_FMT = "|{:^6s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|"
ROW_FMT = "|{:>6d}|{:>10,.3f}|{:>10,.3f}|{:>10,.3e}|{:>10,.3f}|{:>10.3%}|{:>10.3%}|{:>10,.3f}|{:>10.3%}|{:>10.3%}|"

# %% 2. Configuration and Hyperparameters

@dataclass
class Config:
    
    # --- Model architecture --- #
    n_classes: int = 100
    bottleneck_factor: int = 4
    input_conv_filters: int = 16
    "c_out for the convolution applied to the input."
    group_chs: list[int] = field(default_factory=lambda: [64, 128, 256])
    "The number of channels in each group."
    group_strides: list[int] = field(default_factory=lambda: [1, 2, 2])
    "The stride of the first block in each group."
    group_n_blocks: list[int] = field(default_factory=lambda: [9, 9, 9])
    "Number of blocks in each group."
    
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
    gamma: float = 0.1
    "ReduceLROnPlateau multiplier."
    patience: int = 3
    "If our validation loss plateaus for this many eval cycles, we scale the learning rate by gamma."
    
    # --- Setup and Flags --- #
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    float32_matmul_precision: Literal['highest', 'high', 'medium'] = 'high'
    seed: int = 20250804
    "Set to 0 to disable seeding."
    n_workers: int = 12
    
    def __post_init__(self):
        assert len(self.group_chs) == len(self.group_strides) == len(self.group_n_blocks)
        assert self.save_every % self.eval_every == 0, "save_every must be a multiple of eval_every"

        # block layers + 1 input layer + 1 output layer.
        self.layers = 3 * sum(self.group_n_blocks) + 2
            
# %% 3. Model

class BottleneckBlock(nn.Module):
    """
    A bottleneck ResNet block as described in sec. 4.1 and illustrated in figure 5. A 1x1 convolution is used to reduce the number of channels by a factor of 4, a 3x3 convolution is used to extract features, and a 1x1 convolution is used to increase the number of channels. Each convolution is followed by a BatchNorm and ReLU. Downsampling is performed by the first 1x1 convolution with stride=2.
    """

    def __init__(
        self,
        c_in: int,
        c_out:int,
        c_bottleneck: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(c_in, c_bottleneck, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(c_bottleneck)

        self.conv2 = nn.Conv2d(c_bottleneck, c_bottleneck, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_bottleneck)
        
        self.conv3 = nn.Conv2d(c_bottleneck, c_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c_out)
        
        self.downsample = c_out != c_in
        # Option B from sec. 3.3: use a 1x1 convolution with stride=2.
        self.shortcut = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(c_out),
        ) if self.downsample else nn.Identity()
        
    def forward(
        self,
        x: Float[Tensor, "batch c_in h_in w_in"]
    ) -> Float[Tensor, "batch c_out h_out w_out"]:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out)) + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out

class BottleneckResNet(nn.Module):
    """A bottleneck ResNet as described in sec. 4.1"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.bottleneck_factor = cfg.bottleneck_factor

        # Input layer (mirrors CIFAR-10 implementation)
        self.conv1 = nn.Conv2d(3, cfg.input_conv_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg.input_conv_filters)
        
        c_ins = [cfg.input_conv_filters] + cfg.group_chs
        self.groups = nn.Sequential(*[
            self._make_group(c_in, c_out, stride, n_blocks)
            for c_in, c_out, stride, n_blocks in
            zip(c_ins, cfg.group_chs, cfg.group_strides, cfg.group_n_blocks)
        ])
        
        # Output layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cfg.group_chs[-1], cfg.n_classes)
        
        self.apply(self._init_weights)
    
    def _make_group(
        self,
        c_in: int,
        c_out: int,
        stride: int,
        n_blocks: int,
    ) -> nn.Sequential:
        c_bottleneck = c_out // self.bottleneck_factor
        group = [BottleneckBlock(c_in, c_out, c_bottleneck, stride)]
        for _ in range(1, n_blocks):
            group.append(BottleneckBlock(c_out, c_out, c_bottleneck, 1))
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

# %% 4. Dataset and Augmentation

# Unusually, He et al 2015 normalizes with the per-pixel mean and without std.
CIFAR100_MEAN_PATH = f"{DATA_DIR}/cifar_100_mean.npy"
if not os.path.exists(CIFAR100_MEAN_PATH):
    np_mean = CIFAR100(root=DATA_DIR, train=True, download=True).data.mean(axis=0)
    # Convert to PyTorch-friendly format: float32 scaled to [0, 1] in (C, H, W) format.
    np_mean = rearrange(np_mean, "h w c -> c h w")
    np_mean = np_mean.astype(np.float32) / 255.0
    np.save(CIFAR100_MEAN_PATH, np_mean)
else:
    np_mean = np.load(CIFAR100_MEAN_PATH)

CIFAR100_MEAN = torch.from_numpy(np_mean)

# They also use PCA color augmentation as used in AlexNet.
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
    """Apply PCA color augmentation and subtract the mean."""
    c, h, w = x.shape
    coeffs = torch.empty(c, device=x.device).normal_(std=0.1)
    color_aug = einsum(EIGENVECTORS, EIGENVALUES, coeffs, "c1 c2, c2, c2 -> c1")
    return x - CIFAR100_MEAN + color_aug.view(c, 1, 1) 

train_transform = v2.Compose([
    v2.ToImage(),
    # TODO: RandomResizedCrop may be too aggressive for CIFAR-100 (32x32).
    # Try replacing with RandomCrop.
    v2.RandomResizedCrop(scale=(0.5, 1.0), size=(32, 32), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(pca_aug_and_norm),
])

@torch.compile(mode='max-autotune')
def sub_mean(x: Float[Tensor, "channel height width"]) -> Float[Tensor, "channel height width"]:
    """Subtract the per-pixel mean."""
    return x - CIFAR100_MEAN

# TODO: We're not applying the test-time augmentation the paper uses for ImageNet.
# Fully-convolutional form w/ 10-crop testing, with scores averaged across scales (shorter side $\in \{224,256,384,480,640\}$).
test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(sub_mean),
])

# %% 5. Trainer

class BottleneckTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda")
        
        generator = torch.Generator().manual_seed(cfg.seed) if cfg.seed > 0 else None
        
        self.train_loader = DataLoader(
            CIFAR100(root=DATA_DIR, train=True, transform=train_transform, download=True),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.n_workers,
            persistent_workers=cfg.n_workers > 0,
            drop_last=False,
            pin_memory=True,
            generator=generator,
        )
        self.test_loader = DataLoader(
            CIFAR100(root=DATA_DIR, train=False, transform=test_transform, download=True),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.n_workers,
            persistent_workers=cfg.n_workers > 0, # Can disable if hitting memory limits.
            drop_last=False,
            pin_memory=True,
        )

        self.model = BottleneckResNet(cfg).to(self.device, memory_format=torch.channels_last)

        self.opt = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg.initial_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            fused=True,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode="max",
            factor=cfg.gamma,
            patience=cfg.patience,
        )

        if self.cfg.save_every > 0:
            os.makedirs("checkpoints", exist_ok=True)
    
    def train(self):
        self.model.train()

        loader_iter = iter(self.train_loader)
        pbar = tqdm(range(1, self.cfg.train_steps+1), desc="Training")
        training_time = 0.0
        
        # Start the clock.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        for step in pbar:
            # ---- Training ---- #
            
            # 1. Load our batch and labels.
            try:
                batch, labels = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                batch, labels = next(loader_iter)
                
            batch = batch.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(self.device, non_blocking=True)

            # 2. Forward pass.
            pred = self.model(batch)
            loss = F.cross_entropy(pred, labels)

            # 3. Backward pass.
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            # ---- Evaluation ---- #
            last_step = step == self.cfg.train_steps
            if last_step or self.cfg.eval_every > 0 and step % self.cfg.eval_every == 0:
                torch.cuda.synchronize()
                iter_time = time.perf_counter() - t0
                training_time += iter_time
                
                # Save a checkpoint.
                if self.cfg.save_every > 0 and (
                    last_step or step % self.cfg.save_every == 0
                ):
                    torch.save({
                        'model': self.model.state_dict(),
                        'optimizer': self.opt.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'step': step,
                    }, f"checkpoints/run-{self.cfg.run_id}-step-{step}.pt")
                
                self.model.eval()
                test_metrics = self.evaluate()
                self.model.train()
                
                # Our trainining metrics are only estimates (computed on a single batch).
                metrics = {
                    "step": step, 
                    "time": training_time,
                    "iter_time": iter_time,
                    "lr": self.scheduler.get_last_lr()[0],
                    "train_loss": loss.item(),
                    "train_acc1": (pred.argmax(dim=1) == labels).float().mean().item(),
                    "train_acc5": (pred.topk(5)[1] == labels.view(-1, 1)).any(dim=1).float().mean().item(),
                    **test_metrics,
                }
                logging.info(ROW_FMT.format(*[metrics[col] for col in LOGGING_COLUMNS]))
                pbar.set_postfix(train_loss=metrics['train_loss'], test_loss=metrics['test_loss'])
                
                # Start the clock again.
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                # Each time we evaluate, check if our accuracy has plateaued.
                # Note that this is cheating! This run is more of an educational exercise, but using validation results to adjust the learning rate means we should include validation in our time budget.
                self.scheduler.step(metrics['test_acc5'])
            
        torch.cuda.synchronize()
        training_time += time.perf_counter() - t0
        logging.info(f"Total training time: {training_time:,.2f}s")

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        items = len(self.test_loader.dataset)
        cum_loss = torch.tensor(0.0, device="cuda")
        n_correct_top1 = torch.tensor(0.0, device="cuda")
        n_correct_top5 = torch.tensor(0.0, device="cuda")

        pbar = tqdm(self.test_loader, desc="Evaluating", position=1, leave=False)
        for batch, labels in pbar:
            batch = batch.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(self.device, non_blocking=True)
            pred = self.model(batch)
            loss = F.cross_entropy(pred, labels, reduction="sum")

            cum_loss += loss
            n_correct_top1 += (pred.argmax(dim=1) == labels).sum()
            n_correct_top5 += (pred.topk(5)[1] == labels.view(-1, 1)).sum()
            
        return {
            "test_loss": cum_loss.item() / items,
            "test_acc1": n_correct_top1.item() / items,
            "test_acc5": n_correct_top5.item() / items,
        }
    
# %%

if __name__ == "__main__":
    cfg = Config()

    # Flags
    torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    if cfg.seed > 0:
        torch.manual_seed(cfg.seed) # seeds all devices

    # Set up logging
    run_id = datetime.now().isoformat(timespec="seconds")
    cfg.run_id = run_id
    os.makedirs(f"{BASE_DIR}/logs", exist_ok=True)
    logging.basicConfig(filename=f"{BASE_DIR}/logs/{run_id}.txt", format="%(message)s", level=logging.INFO)
    
    logging.info(" ".join(sys.argv))
    logging.info(f"{run_id=}")
    logging.info(f"Running Python {sys.version} and PyTorch {torch.version.__version__}")
    logging.info(f"Running CUDA {torch.version.cuda} and cuDNN {torch.backends.cudnn.version()}")
    logging.info(torch.cuda.get_device_name())
    logging.info(tabulate(vars(cfg).items(), headers=["Config Field", "Value"]))
    
    # Train our model
    trainer = BottleneckTrainer(cfg)
    logging.info(HEADER_FMT.format(*LOGGING_COLUMNS))
    logging.info(HEADER_FMT.format(*['---' for _ in LOGGING_COLUMNS]))
    trainer.train()
    
    smi = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    logging.info(smi.stdout)
    logging.info(f"Max memory allocated: {torch.cuda.max_memory_allocated() // 1024**2:,} MiB")
    logging.info(f"Max memory reserved: {torch.cuda.max_memory_reserved() // 1024**2:,} MiB")
    
    # Write this source code to our logs.
    with open(sys.argv[0]) as f:
        logging.info(f.read())
