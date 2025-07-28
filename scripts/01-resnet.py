# %%
import platform
import sys
import os
import uuid
import subprocess
from dataclasses import dataclass, field, fields, MISSING
from typing import Literal, get_origin, get_args
import logging
import argparse

from tabulate import tabulate
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
import numpy as np
from einops import rearrange
from torchvision.transforms import v2
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import wandb
from jaxtyping import Float

torch.backends.cudnn.benchmark = True
# %% [markdown]
"""
A ResNet for CIFAR-100 based on [*Deep Residual Learning for Image
Recognition*](https://arxiv.org/abs/1512.03385).

### Residual Block (Sec 3.2)

- They use $\mathbf y = \mathcal F(\mathbf x, \{W_i\}) + \mathbf x$. The
  dimensions of $\mathbf x$ and $\mathcal F$ must be equal. If this is not the
  case, we can perform a linear projection $W_s$ by the shortcut connections to
  match the dimension.
- A non-linearity (ReLU) is applied after the addition.
- $\mathcal F$ is flexible, and generally has two or three layers.

### Network Architecture (Sec 3.3)

- Convolutions with 3x3 filters. 
- `c_out = c_in` when feature map size stays constant. When feature map size is
  halved with `stride=2`, `c_out = 2 * c_in`.
- Ends with global average pooling, a fully-connected layer, and softmax to
  produce logits.
- For shortcut connections where `c_in ≠ c_out`, either (a) pad with extra
  zeros and 2x2 pool/subsample the input tensor or (b) use a 1x1 convolution
  with `stride=2` for the shortcut connection.

### CIFAR-10 Implementation (Sec 4.2)

- **Per-pixel** mean subtracted.
- A 3x3 convolution is applied to the input. Then a stack of 6n 3x3 convolution
  layers with feature maps of sizes $\in \{32,16,8\}$ is applied, with
  `stride=2` for subsampling layers. They compare $n \in \{3,5,7,9\}$.
- Trained for 64k iterations with a mini-batch size of 128. 0.1 learning rate,
  divided by 10 after 32k and 48k iterations.
- 4 pixels padded on each side, with a 32x32 crop sampled from the image or its
  horizontal flip.
- They use zero-padding for the shortcut connections (option A).
- Otherwise mirrors the ImageNet implementation.
- No test-time augmentation.
- They also explore a deeper ResNet with `n=18`. To assist convergence they use
  `lr=0.01` to warm up for 400 iterations. An `n=200` network also converges,
  but overfits and has worse test accuracy.

## Our Implementation

CIFAR-100 has the same 32x32 format as CIFAR-10, so we can borrow the
corresponding approach to augmentation. We'll start by using the CIFAR-10
architecture with n=9 (i.e. 56 layers). We're prioritizing faithfulness to the
original implementation, so we're leaving lots of speed on the table (e.g.
mixed precision). We should implement GPU-accelerated data loading at some
point to keep the baseline fair.
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
    num_classes: int = 100
    blocks_per_group: int = 9
    "This is n from sec. 4.2. We have 2n layers in each of our 3 groups."
    block_filters: list[int] = field(default_factory=lambda: [16, 32, 64])
    "Number of filters for each group. block_filters[0] is c_out for our input convolution."
    
    # --- Training --- #
    train_steps: int = 64_000
    "Number of mini-batch iterations we train for."
    eval_every: int = 1_000
    "Set to 0 to disable evaluation."
    save_every: int = 10_000
    "Set to 0 to disable checkpointing. Must be a multiple of eval_every."
    train_batch_size: int = 128
    eval_batch_size: int = 512
    initial_lr: float = 0.1
    momentum: float = 0.9
    "SGD momentum (not BatchNorm momentum)."
    weight_decay: float = 1e-4
    milestones: list[int] = field(default_factory=lambda: [32_000, 48_000])
    "MultiStepLR milestones."
    gamma: float = 0.1
    "MultiStepLR multiplier."
    device: Literal['cuda', 'mps', 'cpu'] = DEVICE
    memory_format: Literal['preserve_format', 'channels_last'] = 'preserve_format' if DEVICE == 'mps' else 'channels_last'
    "Memory format for our model and data. channels_last improves convolution locality but causes crashes on mps."
    seed: int = 20250723
    "Set to 0 to disable seeding."
    
    # --- Logging --- #
    use_wandb: bool = False
    wandb_project: str | None = 'cifar100-speedrun'
    wandb_name: str | None = 'resnet{layers}'
    
    def __post_init__(self):
        # 3 groups * 2n layers per group + 1 input layer + 1 output layer.
        self.layers = 3 * self.blocks_per_group * 2 + 2
        if self.use_wandb:
            assert self.wandb_project is not None, "wandb_project is required"
            assert self.wandb_name is not None, "wandb_name is required"
            self.wandb_name = self.wandb_name.format(layers=self.layers)
            
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

    def __init__(self, c_in: int, downsample: bool = False) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out = c_in * 2 if downsample else c_in
        self.stride = stride = 2 if downsample else 1
        self.downsample = downsample

        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        
        if downsample:
            pad_chs = c_out - c_in
            # Option A from sec. 4.1: spatially downsample then pad with zeros
            # for a parameter-free shortcut.
            self.shortcut = nn.Sequential(
                # Downsample spatially (we're not actually doing any pooling here)
                nn.MaxPool2d(kernel_size=1, stride=2),
                # Pad with zeros along the channel dimension.
                # Final arg in tuple is padding_back: applied to the channel dimension.
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


    def extra_repr(self) -> str:
        return f"downsample={self.downsample}, in_channels={self.c_in}, out_channels={self.c_out}, stride={self.stride}"

class ResNet(nn.Module):
    """A non-bottleneck ResNet as described in sec. 4.2"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()

        # Input layer
        self.conv1 = nn.Conv2d(3, cfg.block_filters[0], 3, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(cfg.block_filters[0])
        
        # 3 groups, each with 2n layers. c_out is c_in/2 for downsampling groups.
        n = cfg.blocks_per_group
        self.group1 = self._make_group(c_in=cfg.block_filters[0], blocks=n, downsample=False)
        self.group2 = self._make_group(c_in=cfg.block_filters[0], blocks=n, downsample=True)
        self.group3 = self._make_group(c_in=cfg.block_filters[1], blocks=n, downsample=True)

        # Output layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cfg.block_filters[-1], cfg.num_classes)

        self.apply(self._init_weights)
    
    def _make_group(self, c_in: int, blocks: int, downsample: bool) -> nn.Sequential:
        group = [BasicBlock(c_in, downsample=downsample)]
        for _ in range(1, blocks):
            group.append(BasicBlock(group[-1].c_out, downsample=False))
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
    ) -> Float[Tensor, "batch num_classes"]:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = self.pool(out).flatten(start_dim=1)
        return self.fc(out)

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

# %% 5. Trainer

class ResNetTrainer():
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.memory_format = torch.preserve_format if cfg.memory_format == 'preserve_format' else torch.channels_last
        
        if self.device.type == 'cuda':
            pin = True
            workers = os.cpu_count()
            assert workers is not None, "Could not determine number of CPUs with os.cpu_count()"
        else:
            pin = False
            workers = 0
        
        if cfg.seed > 0:
            generator = torch.Generator().manual_seed(cfg.seed)           
        else:
            generator = None

        self.train_loader = DataLoader(
            CIFAR100(root=DATA_DIR, train=True, transform=train_transform, download=True),
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=workers,
            persistent_workers=workers > 0,
            drop_last=False,
            pin_memory=pin,
            generator=generator,
        )
        self.test_loader = DataLoader(
            CIFAR100(root=DATA_DIR, train=False, transform=test_transform, download=True),
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=workers,
            drop_last=False,
            pin_memory=pin,
        )

        self.model = ResNet(cfg).to(self.device, memory_format=self.memory_format)
        self.profile_model() # Profile before compilation.

        if self.device.type == 'cuda':
            self.model = torch.compile(self.model, mode="max-autotune")
            self.warmup()
        
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
        
    def profile_model(self):
        """
        Logs model parameters and forward/backward FLOPs. Must be called after
        self.model and self.train_loader are initialized but before compilation.
        """

        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        bs = self.cfg.train_batch_size
        batch = torch.rand((bs, 3, 32, 32), device=self.device).to(memory_format=self.memory_format)
        labels = torch.randint(0, self.cfg.num_classes, (bs,), device=self.device)
        
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            self.model(batch)

        shape_str = f"With batch shape {tuple(batch.shape)}"
        logging.info(f"{shape_str}, the forward pass incurs {flop_counter.get_total_flops()} FLOPs.")
        logging.info(flop_counter.get_table())

        temp_opt = torch.optim.SGD(self.model.parameters(), lr=0.0)
        with flop_counter:
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

    def warmup(self):
        """Warm up our compiled model."""
        for _ in range(3):
            bs = self.cfg.train_batch_size
            batch = torch.rand((bs, 3, 32, 32), device=self.device).to(memory_format=self.memory_format)
            labels = torch.randint(0, self.cfg.num_classes, (bs,), device=self.device)
            out = self.model(batch)
            loss = F.cross_entropy(out, labels)
            loss.backward()

    def train(self) -> dict:
        self.model.train()

        loader_iter = iter(self.train_loader)
        starter = torch.Event(device=self.device, enable_timing=True)
        ender = torch.Event(device=self.device, enable_timing=True)
        # TODO: Does elapsed_time synchronize? If so, we can remove this.
        # TODO: Should we be using time.perf_counter() instead of Events?
        synchronize = (
            torch.cuda.synchronize if self.device.type == 'cuda'
            else torch.mps.synchronize if self.device.type == 'mps'
            else torch.cpu.synchronize # No-op
        )

        pbar = tqdm(range(1, self.cfg.train_steps+1), desc="Training")
        total_time_seconds = 0.0
        starter.record()

        for step in pbar:
            # ---- Training ---- #
            try:
                batch, labels = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                batch, labels = next(loader_iter)
                
            # Note that we don't use autocast here to match the original implementation.
            batch = batch.to(self.device, non_blocking=True, memory_format=self.memory_format)
            labels = labels.to(self.device, non_blocking=True)
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
                ender.record()
                synchronize()
                total_time_seconds += 1e-3 * starter.elapsed_time(ender)
                
                if self.cfg.save_every > 0 and (
                    last_step or step % self.cfg.save_every == 0
                ):
                    torch.save({
                        'model': self.model.state_dict(),
                        'optimizer': self.opt.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'step': step,
                    }, f"checkpoints/resnet{self.cfg.layers}-{step}.pt")
                
                self.model.eval()
                test_metrics = self.evaluate()
                self.model.train()
                
                # Our trainining metrics are only estimates (computed on a single batch).
                metrics = {
                    "step": step, 
                    "time": total_time_seconds,
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
                starter.record()

            self.scheduler.step()
            
        ender.record()
        synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)
        logging.info(f"Total time: {total_time_seconds:.2f}s")
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
            pred = self.model(batch)

            cum_loss += F.cross_entropy(pred, labels, reduction="sum")
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
        else: # Fallback: treat as string.
            parser.add_argument(f"--{f.name}", type=str, default=_default(f))

    return Config(**vars(parser.parse_args()))

if __name__ == "__main__":
    run_id = uuid.uuid4()
    os.makedirs(f"{BASE_DIR}/logs", exist_ok=True)
    logging.basicConfig(filename=f"{BASE_DIR}/logs/{run_id}.txt", format="%(message)s", level=logging.INFO)
    
    cfg = parse_cfg()
    if cfg.seed > 0:
        torch.cuda.manual_seed(cfg.seed)
        torch.mps.manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    logging.info(tabulate(vars(cfg).items(), headers=["Config Field", "Value"]))
    logging.info(f"Running Python {sys.version}")
    logging.info(f"Running PyTorch {torch.version.__version__}")
    if cfg.device == 'cuda':
        logging.info(f"Using CUDA {torch.version.cuda} and cuDNN {torch.backends.cudnn.version()}")
    if cfg.device == 'mps':
        release, _, machine = platform.mac_ver()
        logging.info(f"Using mps on MacOS {release} for {machine}.")
        
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
