
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
A script which naively implements non-bottleneck ResNets based on [*Deep
Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) for
CIFAR-100.

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
- For shortcut connections where `c_in ≠ c_out`, either (a) 2x2 pool/subsample
  the input tensor and pad with extra zeros or (b) use a 1x1 convolution with
  `stride=2` for the shortcut connection.

## Non-Bottleneck Implementation for CIFAR-10 (Sec 4.2)

- A 3x3 convolution is applied to the input. Then a stack of 6n 3x3 convolution
  layers with feature maps of sizes $\in \{32,16,8\}$ is applied, with
  `stride=2` for subsampling layers. They compare $n \in \{3,5,7,9\}$.
- They use downsampling with zero-padding for the shortcut connections (option
  A from sec. 3.3).
- Trained for 64k iterations with a mini-batch size of 128. 0.1 learning rate,
  divided by 10 after 32k and 48k iterations.
- Train-time augmentation: 4 pixels padded on each side, with a 32x32 crop
  sampled from the image or its horizontal flip. *Per-pixel* mean subtracted.
- No test-time augmentation.
- They also explore a deeper ResNet with `n=18`. To assist convergence they use
  `lr=0.01` to warm up for 400 iterations. An `n=200` network also converges,
  but overfits and has worse test accuracy.
- Otherwise mirrors the ImageNet implementation.

## Our Implementation

We're ignoring *many* straightforward optimizations in order to faithfully implement the paper. We do deviate slightly by allowing tf32 convolutions and matmuls, which allow us to take advantage of the H100's tensor cores with only a marginal loss in accuracy.
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
    input_conv_filters: int = 16
    "c_out for the convolution applied to the input."
    group_c_ins: list[int] = field(default_factory=lambda: [16, 16, 32])
    "c_in for the first convolution in each group."
    group_downsample: list[bool] = field(default_factory=lambda: [False, True, True])
    "Whether the first block in each group downsamples the feature map size and doubles the number of channels."
    group_n_blocks: list[int] = field(default_factory=lambda: [9, 9, 9])
    "Number of blocks in each group."
    
    # --- Training --- #
    train_steps: int = 64_000
    "Number of mini-batch iterations we train for."
    eval_every: int = 4_000
    "Set to 0 to disable evaluation."
    save_every: int = 16_000
    "Set to 0 to disable checkpointing. Must be a multiple of eval_every."
    batch_size: int = 128
    initial_lr: float = 0.1
    momentum: float = 0.9
    "SGD momentum (not BatchNorm momentum)."
    weight_decay: float = 1e-4
    milestones: list[int] = field(default_factory=lambda: [32_000, 48_000])
    "MultiStepLR milestones."
    gamma: float = 0.1
    "MultiStepLR multiplier."
    
    # --- Setup and Flags --- #
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    float32_matmul_precision: Literal['highest', 'high', 'medium'] = 'high'
    seed: int = 20250802
    "Set to 0 to disable seeding."
    n_workers: int = 12
    
    def __post_init__(self):
        assert len(self.group_c_ins) == len(self.group_n_blocks) == len(self.group_downsample)
        assert self.save_every % self.eval_every == 0, "save_every must be a multiple of eval_every"

        # block layers + 1 input layer + 1 output layer.
        self.layers = 2 * sum(self.group_n_blocks) + 2
            

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
        downsample: bool,
    ) -> None:
        super().__init__()

        self.c_out = c_out = 2 * c_in if downsample else c_in
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)

        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        
        if downsample:
            # Parameter-free shortcut (Option A from sec. 3.3).
            pad_chs = c_out - c_in
            self.shortcut = nn.Sequential(
                # Spatially downsample (we're not actually doing any pooling).
                nn.MaxPool2d(kernel_size=1, stride=stride),
                # Pad with zeros along the channel dimension.
                nn.ZeroPad3d((0, 0, 0, 0, 0, pad_chs)),
            )
        else:
            self.shortcut = nn.Identity()

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

class ResNet(nn.Module):
    """A non-bottleneck ResNet as described in sec. 4.2."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()

        # Input layer
        self.conv1 = nn.Conv2d(3, cfg.input_conv_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg.input_conv_filters)

        self.groups = nn.Sequential(*[
            self._make_group(c_in, downsample, n_blocks)
            for c_in, downsample, n_blocks in
            zip(cfg.group_c_ins, cfg.group_downsample, cfg.group_n_blocks)
        ])
        
        # Output layer
        if cfg.group_downsample[-1]:
            c_out = cfg.group_c_ins[-1] * 2
        else:
            c_out = cfg.group_c_ins[-1]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_out, cfg.n_classes)
        
        self.apply(self._init_weights)
    
    def _make_group(
        self,
        c_in: int,
        downsample: bool,
        n_blocks: int,
    ) -> nn.Sequential:
        group = [BasicBlock(c_in, downsample)]
        for _ in range(1, n_blocks):
            group.append(BasicBlock(group[-1].c_out, False))

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

@torch.compile(mode='max-autotune')
def sub_mean(x: Float[Tensor, "channel height width"]) -> Float[Tensor, "channel height width"]:
    """Subtract the per-pixel mean."""
    return x - CIFAR100_MEAN

train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(sub_mean),
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(sub_mean),
])

# %% 5. Trainer

class ResNetTrainer:
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

        self.model = ResNet(cfg).to(self.device, memory_format=torch.channels_last)

        self.opt = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg.initial_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            fused=True,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.opt,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
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

            # We call scheduler.step() after evaluation.
            # If we stepped here, we'd misreport the learning rate.
            
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
                    "train_acc1": (pred.argmax(dim=1) == labels).float().mean().item() * 100,
                    "train_acc5": (pred.topk(5)[1] == labels.view(-1, 1)).any(dim=1).float().mean().item() * 100,
                    **test_metrics,
                }
                logging.info(ROW_FMT.format(*[metrics[col] for col in LOGGING_COLUMNS]))
                pbar.set_postfix(train_loss=metrics['train_loss'], test_loss=metrics['test_loss'])
                
                # Start the clock again.
                torch.cuda.synchronize()
                t0 = time.perf_counter()

            # 4. Update our learning rate.
            self.scheduler.step()

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
    trainer = ResNetTrainer(cfg)
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
