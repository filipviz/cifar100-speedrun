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
- Uses mixed-precision training.

Oddly, they apply the first batchnorm + ReLU to both the residual stream and the convolution path (which performs worse). It doesn't make a huge difference for a network this shallow, but I won't do this.

Page makes the following changes:
- Removes the batchnorm + ReLU from the input stem, since they're redundant with the batchnorm + ReLU in the first residual block.
- Removes some of the jumps in the learning rate schedule.
- Preprocesses the dataset in advance. Rather than applying padding, normalization, and random horizontal flipping each time they load a batch, he pre-applies these. This leaves only random cropping and flipping.
- He removes dataworkers to avoid the overhead associated with launching them. Keeping everything in the main thread saves time!

TODO:
- Replicate learning rate schedule from notebook.
- Preprocess the dataset.
- Mixed-precision training.
- Remove dataloader workers (everything in main thread).
- Combine random number calls into bulk calls up front (how did he do this?).

[article](https://web.archive.org/web/20221206112356/https://myrtle.ai/how-to-train-your-resnet-1-baseline/)
[notebook](https://github.com/davidcpage/cifar10-fast/blob/master/experiments.ipynb)
"""

# %% 1. Global Constants

# assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

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
    input_conv_filters: int = 64
    "c_out for the convolution applied to the input."
    group_c_ins: list[int] = field(default_factory=lambda: [64, 64, 128, 256])
    "c_in for the first convolution in each group."
    group_downsample: list[bool] = field(default_factory=lambda: [False, True, True, True])
    "Whether the first block in each group downsamples the feature map size and doubles the number of channels."
    group_n_blocks: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
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
    warmup_steps: int = 400
    "Number of steps for the warmup learning rate."
    warmup_lr: float = 0.01
    "Warmup learning rate."
    
    # --- Setup and Flags --- #
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    float32_matmul_precision: Literal['highest', 'high', 'medium'] = 'high'
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
    A pre-activation ResNet building block as described in He et al 2016 sec. 4.1.

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
    """A pre-activation ResNet as described in He et al 2016."""

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

CIFAR100_MEAN = (0.5070757270, 0.4865502417, 0.4409192204)
CIFAR100_STD = (0.2673342824, 0.2564384639, 0.2761504650)

train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(CIFAR100_MEAN, CIFAR100_STD, inplace=True),
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(CIFAR100_MEAN, CIFAR100_STD, inplace=True),
])

# %% 5. Trainer

class PreActTrainer:
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
            drop_last=True,
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

        self.model = BaselineResnet(cfg).to(self.device, memory_format=torch.channels_last)

        self.opt = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg.initial_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            fused=True,
        )
        
        warmup = torch.optim.lr_scheduler.ConstantLR(
            self.opt,
            factor=cfg.warmup_lr / cfg.initial_lr,
            total_iters=cfg.warmup_steps,
        )
        multistep = torch.optim.lr_scheduler.MultiStepLR(
            self.opt,
            milestones=[m - cfg.warmup_steps for m in cfg.milestones],
            gamma=cfg.gamma,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.opt,
            schedulers=[warmup, multistep],
            milestones=[cfg.warmup_steps],
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

            # 4. Update our learning rate.
            self.scheduler.step()
            
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
    trainer = PreActTrainer(cfg)
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
