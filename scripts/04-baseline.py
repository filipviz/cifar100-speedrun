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
import wandb

# %% [markdown]
"""
In this script, we implement the baseline ResNet described in [*How to Train Your ResNet 1: Baseline*](https://web.archive.org/web/20221206112356/https://myrtle.ai/how-to-train-your-resnet-1-baseline/).

Since we're starting to make more opinionated and idiosyncratic changes, we'll add wandb logging to more easily compare results across runs.

It's based on Ben Johnson's DAWNBench submission, which:
- Is an 18-layer pre-activation ResNet.
- Uses 64 -> 128 -> 256 -> 512 channels. Much shallower and wider than the original ResNet.
- Uses 1x1 convolutions for downsampling shortcuts.
- Uses four groups with 2 blocks each. The first is not a downsampling group, and the rest are (meaning we apply 4x4 mean pooling at the end).
- Uses typical mean/std normalization rather than per-pixel mean alone (as in He et al 2015).
- Uses a somewhat odd learning rate schedule with a long linear warmup, long linear decay, and a few small jumps.
- Uses "mixed-precision" training (it's all fp16).

Page makes the following changes:
- Removes the batchnorm + ReLU from the input stem, since they're redundant with the batchnorm + ReLU in the first residual block.
- Removes some of the jumps in the learning rate schedule.
- Preprocesses the dataset in advance. Rather than applying padding, normalization, and random horizontal flipping each time they load a batch, he pre-applies these. This leaves only random cropping and flipping.
- He removes dataworkers to avoid the overhead associated with launching them. Keeping everything in the main thread saves time!

Our implementation departs from theirs in a few ways:
1. We update our learning rate each batch rather than each epoch.
2. Oddly, they apply the first batchnorm + ReLU to both the residual stream and the convolution path (which hurts performance). It doesn't make a huge difference for a network this shallow, but I won't do this.
3. We take a more aggressive approach to data loading by loading the entire dataset into GPU memory (in uint8) and applying pre-processing on-device during the first call to `__iter__`.
4. We're using bfloat16 rather than fp16 (TODO - compare).

TODO:
- Combine random number calls into bulk calls up front (how did he do this?).

[article](https://web.archive.org/web/20221206112356/https://myrtle.ai/how-to-train-your-resnet-1-baseline/)
[notebook](https://github.com/davidcpage/cifar10-fast/blob/master/experiments.ipynb)
"""

# %% 1. Global Constants

assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

BASE_DIR = f"{os.path.dirname(__file__)}/.."
DATA_DIR = f"{BASE_DIR}/data"
DEVICE = "cuda"
DTYPE = torch.bfloat16

LOGGING_COLUMNS = ['step', 'time', 'iter_time', 'lr', 'train_loss', 'train_acc1',
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
    group_c_ins: list[int] = field(default_factory=lambda: [64, 64, 128, 256])
    "c_in for the first convolution in each group."
    group_downsample: list[bool] = field(default_factory=lambda: [False, True, True, True])
    "Whether the first block in each group downsamples the feature map size and doubles the number of channels."
    group_n_blocks: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    "Number of blocks in each group."
    
    # --- Training --- #
    train_steps: int = 13_999
    "Number of mini-batch iterations we train for."
    eval_every: int = 4_000
    "Set to 0 to disable evaluation."
    save_every: int = 16_000
    "Set to 0 to disable checkpointing. Must be a multiple of eval_every."
    batch_size: int = 128
    momentum: float = 0.9
    "SGD momentum (not BatchNorm momentum)."
    weight_decay: float = 1e-4
    lrs: list[float] = field(default_factory=lambda: [0, 0.1, 0.005, 0])
    "We linearly interpolate between these learning rates with np.interp(step, milestones, lrs)."
    milestones: list[int] = field(default_factory=lambda: [0, 6_000, 12_000, 14_000])
    "The number of steps at which we reach each learning rate."
    
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

    # --- wandb --- #
    wandb_project: str | None = None
    wandb_name: str | None = None
    use_wandb: bool = False
    
    def __post_init__(self):
        assert len(self.group_c_ins) == len(self.group_n_blocks) == len(self.group_downsample)
        assert self.save_every % self.eval_every == 0, "save_every must be a multiple of eval_every"

        if self.use_wandb:
            assert self.wandb_project and self.wandb_name, "must set wandb_project and wandb_name if use_wandb is True"

        # block layers + 1 input layer + 1 output layer.
        self.layers = 2 * sum(self.group_n_blocks) + 2
            

# %% 3. Model

class BaselineBlock(nn.Module):
    """
    A baseline ResNet building block as described in Page's first article.

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

        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)

        # 1x1 convolution shortcut in downsampling blocks.
        self.shortcut = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(c_out),
        ) if downsample else nn.Identity()

    def forward(
        self,
        x: Float[Tensor, "batch c_in h_in w_in"],
    ) -> Float[Tensor, "batch c_out h_out w_out"]:
        """
        If this is a downsampling block, c_out = 2*c_in and h_out/w_out are
        half of h_in/w_in respectively. Otherwise, they all match.
        """
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return out

class BaselineResnet(nn.Module):
    """A baseline ResNet as described in Page's first article."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()

        # Input layer (only a convolution)
        self.conv1 = nn.Conv2d(3, cfg.input_conv_filters, 3, padding=1, bias=False)

        self.groups = nn.Sequential(*[
            self._make_group(c_in, downsample, n_blocks)
            for c_in, downsample, n_blocks in
            zip(cfg.group_c_ins, cfg.group_downsample, cfg.group_n_blocks)
        ])
        
        # Output layer
        c_out = self.groups[-1][-1].c_out
        self.bn2 = nn.BatchNorm2d(c_out)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_out, cfg.n_classes)
        
        self.apply(self._init_weights)
    
    def _make_group(
        self,
        c_in: int,
        downsample: bool,
        n_blocks: int,
    ) -> nn.Sequential:
        group = [BaselineBlock(c_in, downsample)]
        for _ in range(1, n_blocks):
            group.append(BaselineBlock(group[-1].c_out, False))

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
        out = self.conv1(x)
        out = self.groups(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.pool(out).flatten(start_dim=1)
        return self.fc(out)

# %% 4. Dataset and Augmentation

CIFAR100_MEAN = torch.tensor((0.5078125, 0.486328125, 0.44140625), dtype=DTYPE, device=DEVICE)
CIFAR100_STD = torch.tensor((0.267578125, 0.255859375, 0.275390625), dtype=DTYPE, device=DEVICE)

def batch_crop(images: Float[Tensor, "b c h_in w_in"], crop_size: int = 32) -> Float[Tensor, "b c h_out w_out"]:
    """View-based batch cropping."""
    b, c, h, w = images.shape
    r = (h - crop_size) // 2
 
    # Create strided views of all possible crops.
    b_s, c_s, h_s, w_s = images.stride()
    crops_shape = (b, c, 2*r+1, 2*r+1, crop_size, crop_size)
    crops_stride = (b_s, c_s, h_s, w_s, h_s, w_s)
    crops = torch.as_strided(
        images[:, :, :h-crop_size+1, :w-crop_size+1],
        size=crops_shape, stride=crops_stride
    )
    
    # Select the appropriate crop for each image.
    batch_idx = torch.arange(b, device=images.device)
    shifts = torch.randint(0, 2*r+1, size=(2, b), device=images.device)
    return crops[batch_idx, :, shifts[0], shifts[1]]

def batch_flip_lr(images: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
    """Apply random horizontal flipping to each image in the batch"""
    flip_mask = (torch.rand(len(images), device=images.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, images.flip(-1), images)

class Cifar100Loader:
    """
    Infinite data loader for CIFAR-100. Loads entire dataset into GPU memory, performs transformations on GPU, and doesn't require re-initialization each epoch.
    """
    def __init__(self, cfg: Config, train: bool = True, device: str = DEVICE):
        self.cfg = cfg
        self.device, self.train = device, train
        self.batch_size = cfg.batch_size

        # Load or download data
        cifar_path = os.path.join(DATA_DIR, 'train.pt' if train else 'test.pt')
        if not os.path.exists(cifar_path):
            np_cifar = CIFAR100(root=DATA_DIR, train=train, download=True)
            images = torch.tensor(np_cifar.data)
            labels = torch.tensor(np_cifar.targets)
            torch.save({'images': images, 'labels': labels, 'classes': np_cifar.classes}, cifar_path)

        # Transfer as uint8 then convert on GPU. This is faster than loading pre-processed bf16 data.
        data = torch.load(cifar_path, map_location=device)
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        
        # Convert to bf16, normalize, and rearrange on GPU
        self.images = self.images.to(DTYPE) / 255.0
        self.images = (self.images - CIFAR100_MEAN) / CIFAR100_STD
        self.images = rearrange(self.images, "b h w c -> b c h w").to(memory_format=torch.channels_last)
        
        self.n_images = len(self.images)
        
    def __iter__(self):
        indices = torch.randperm(self.n_images, device=self.device)
        pos = 0
        
        while True:
            # If we need to wrap around, we combine remaining indices with indices from the new epoch.
            if pos + self.batch_size > self.n_images:
                remaining = indices[pos:]
                indices = torch.randperm(self.n_images, device=self.device)
                needed = self.batch_size - len(remaining)
                batch_indices = torch.cat([remaining, indices[:needed]])
                pos = needed
            else:
                # Otherwise, we take the next batch of indices from the current epoch.
                batch_indices = indices[pos:pos + self.batch_size]
                pos += self.batch_size
            
            images = self.images[batch_indices]
            labels = self.labels[batch_indices]
            
            if self.train:
                if self.cfg.crop_padding > 0:
                    images = F.pad(images, (self.cfg.crop_padding,) * 4, mode=self.cfg.pad_mode)
                    images = batch_crop(images, crop_size=32)
                if self.cfg.flip:
                    images = batch_flip_lr(images)
            
            yield images, labels


# %% 5. Trainer

class BaselineTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda")
        
        self.train_loader = Cifar100Loader(cfg, train=True, device=self.device)
        self.test_loader = Cifar100Loader(cfg, train=False, device=self.device)

        self.model = BaselineResnet(cfg).to(
            device=self.device,
            dtype=DTYPE, 
            memory_format=torch.channels_last
        )

        self.opt = torch.optim.SGD(
            self.model.parameters(),
            lr=1.0, # LambdaLR sets lr=initial_lr * lambda(step)
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            fused=True,
        )
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt,
            lambda step: np.interp(step, cfg.milestones, cfg.lrs).item(),
        )

        if self.cfg.save_every > 0:
            os.makedirs("checkpoints", exist_ok=True)
        
        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.wandb_name,
            config=vars(self.cfg),
            config_exclude_keys=["wandb_project", "wandb_name", "use_wandb"],
            save_code=True,
        )
        # wandb.watch(self.model, log="all", log_freq=cfg.eval_every)

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
            batch, labels = next(loader_iter)
            assert batch.is_contiguous(memory_format=torch.channels_last)
            batch = batch.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 2. Forward pass.
            pred = self.model(batch)
            loss = F.cross_entropy(pred, labels)

            # 3. Update our learning rate.
            # We do this before the optimizer step to avoid a first step with lr=0.0
            self.scheduler.step(step)

            # 4. Backward pass.
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
                del metrics['step']
                wandb.log(metrics, step=step)
                
                # Start the clock again.
                torch.cuda.synchronize()
                t0 = time.perf_counter()

        torch.cuda.synchronize()
        training_time += time.perf_counter() - t0
        logging.info(f"Total training time: {training_time:,.2f}s")
        wandb.finish()
    
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        assert not self.model.training, "Model must be in eval mode"
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
    trainer = BaselineTrainer(cfg)
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
