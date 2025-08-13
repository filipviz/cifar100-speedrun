# %%
from datetime import datetime
import subprocess
import sys
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Literal
import logging

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from jaxtyping import Float
from tabulate import tabulate
import wandb

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

This is the first time that one of Page's networks fails to generalize to CIFAR-100! Some brief testing suggests the learning rate is far too high.

In our implementation, we:
- Add wandb and set up a hyperparameter sweep.
- Continue using step-based scheduling (rather than epoch-based scheduling).
- Use mean reduction for our loss. As a result, we don't scale down our learning rate.
- Rather than linear warmup and decay, we use a shorter linear warmup and cosine decay.
- We add autocast and grad scaling to improve numerical stability.

Since the network is smaller now, we'll use wandb to sweep over hyperparameters.
"""

# %% 1. Global Constants

BASE_DIR = f"{os.path.dirname(__file__)}/.."
DATA_DIR = f"{BASE_DIR}/data"
DEVICE = "cuda"

LOGGING_COLUMNS = ['step', 'time', 'interval', 'lr', 'train_loss', 'train_acc1',
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
    train_steps: int = 1_500
    "Number of mini-batch iterations we train for."
    eval_every: int = 100
    "Set to 0 to disable evaluation."
    save_every: int = 0
    "Set to 0 to disable checkpointing. Must be a multiple of eval_every."
    batch_size: int = 768
    momentum: float = 0.9
    "SGD momentum (not BatchNorm momentum)."
    nesterov: bool = False
    "Whether to use Nesterov momentum (SGD only)."
    beta_2: float = 0.999
    "AdamW beta_2."
    weight_decay: float = 0.01
    "Weight decay."
    max_lr: float = 3e-2
    "Our maximum learning rate."
    warmup_steps: int = 300
    "The number of steps over which we linearly increase the learning rate from 0 to max_lr."
    optimizer: Literal['sgd', 'adamw'] = 'sgd'
    "The optimizer to use."
    
    # --- Data Augmentation --- #
    flip: bool = True
    "Random horizontal flipping."
    pad_mode: Literal['reflect', 'constant'] = 'reflect'
    crop_padding: int = 4
    "Set to 0 to disable padding and random cropping."
    cutout_size: int = 8
    "Set to 0 to disable cutout."
    
    # --- Setup and Flags --- #
    dtype: Literal['fp16', 'bf16', 'fp32'] = 'fp16'
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    float32_matmul_precision: Literal['highest', 'high', 'medium'] = 'medium'
    seed: int = 20250812
    "Set to 0 to disable seeding."
    
    # --- Wandb --- #
    use_wandb: bool = True
    wandb_project: str = "backbone-resnet"
    wandb_note: str = ""
    sweep_count: int = 50

    def __post_init__(self):
        assert 0 <= self.warmup_steps <= self.train_steps, "warmup_steps must be between 0 and train_steps"
        if self.eval_every > 0:
            assert self.save_every % self.eval_every == 0, "save_every must be a multiple of eval_every"
        if self.use_wandb:
            assert self.wandb_project, "Must specify a wandb_project"
            
# %% 3. Model

class BackboneBlock(nn.Module):
    """
    A block in David Page's backbone resnet architecture. When residual is False, this is just a convolution -> batchnorm -> ReLU -> max pooling sequence.

    Args:
        c_in: The number of input channels.
        c_out: The number of output channels.
        residual: Whether to add two serial 3x3 convolution -> batchnorm -> ReLU sequences with an identity shortcut.
    """
    
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
    
    def forward(
        self, x: Float[Tensor, "batch c_in h_in w_in"]
    ) -> Float[Tensor, "batch c_out h_out w_out"]:
        """h_out and w_out are one half of h_in and w_in, respectively."""
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
            for i in range(n_groups+1)
        ]

        self.layers = nn.Sequential(*[
            BackboneBlock(c_in, c_out, res) for c_in, c_out, res in 
            zip(c_ins, c_ins[1:], cfg.group_residual)
        ])
        
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(c_ins[-1], cfg.n_classes)
        with torch.no_grad():
            self.fc.weight.mul_(cfg.fc_scale)
    
    def forward(self, x: Float[Tensor, "batch channel height width"]) -> Float[Tensor, "batch n_classes"]:
        out = F.relu(self.bn(self.conv(x)), inplace=True)
        out = self.layers(out)
        out = self.pool(out).flatten(start_dim=1)
        out = self.fc(out)
        return out

# %% 4. Dataset and Augmentation

CIFAR100_MEAN = torch.tensor((0.507076, 0.486550, 0.440919), device=DEVICE)
CIFAR100_STD = torch.tensor((0.267334, 0.256438, 0.276150), device=DEVICE)

def batch_crop(images: Float[Tensor, "b c h_in w_in"], crop_size: int = 32) -> Float[Tensor, "b c h_out w_out"]:
    """Strided view-based (in-place) batch cropping."""
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
    shift_h = torch.randint(0, 2*r+1, size=(b,), device=images.device)
    shift_w = torch.randint(0, 2*r+1, size=(b,), device=images.device)
    return crops[batch_idx, :, shift_h, shift_w]

def batch_flip_lr(images: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
    """Apply random horizontal flipping to each image in the batch"""
    flip_mask = torch.rand(len(images), device=images.device) < 0.5
    images[flip_mask] = images[flip_mask].flip(-1)
    return images

def batch_cutout(images: Float[Tensor, "b c h w"], size: int) -> Float[Tensor, "b c h w"]:
    """In-place vectorized cutout using advanced indexing."""
    if size <= 0:
        return images
    b, c, h, w = images.shape
    dev = images.device
    lo = size // 2
    hi = size - lo

    h_center = torch.randint(0, h, (b, 1, 1), device=dev)
    w_center = torch.randint(0, w, (b, 1, 1), device=dev)
    dh = torch.arange(-lo, hi, device=dev).view(1, size, 1)
    dw = torch.arange(-lo, hi, device=dev).view(1, 1, size)
    h_idx = (h_center + dh).clamp(0, h - 1)
    w_idx = (w_center + dw).clamp(0, w - 1)
    b_idx = torch.arange(b, device=dev).view(b, 1, 1)

    images[b_idx, :, h_idx, w_idx] = 0
    return images

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

        # Transfer as uint8 then convert on GPU. This is faster than loading pre-processed fp16 data.
        data = torch.load(cifar_path, map_location=device)
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        
        # Convert to floats, normalize, and rearrange on GPU.
        # I tried doing this in fp16 but it didn't make a difference.
        self.images = self.images.to(torch.float32) / 255.0
        self.images = (self.images - CIFAR100_MEAN) / CIFAR100_STD
        if self.train and cfg.crop_padding > 0:
            self.images = F.pad(self.images, (0, 0) + (cfg.crop_padding,) * 4, mode=cfg.pad_mode)
        self.images = rearrange(self.images, "b h w c -> b c h w").to(memory_format=torch.channels_last)
        
        self.n_images = len(self.images)
    
    def __len__(self):
        # Needed for tqdm to work when self.train=False.
        # math.ceil(self.n_images / self.batch_size)
        return (self.n_images + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        indices = torch.randperm(self.n_images, device=self.device)
        position = 0
        loop = True

        while loop:
            last_batch = position + self.batch_size > self.n_images
            # If we need to wrap around, we combine remaining indices with indices from the new epoch.
            if last_batch and self.train:
                remaining = indices[position:]
                indices = torch.randperm(self.n_images, device=self.device)
                needed = self.batch_size - len(remaining)
                batch_indices = torch.cat([remaining, indices[:needed]])
                position = needed
            else:
                # Otherwise, we take the next batch of indices from the current epoch.
                batch_indices = indices[position:position + self.batch_size]
                position += self.batch_size
                if last_batch and not self.train:
                    loop = False
            
            images = self.images[batch_indices]
            labels = self.labels[batch_indices]
            
            if self.train:
                if self.cfg.crop_padding > 0:
                    images = batch_crop(images, crop_size=32)
                if self.cfg.flip:
                    images = batch_flip_lr(images)
                if self.cfg.cutout_size > 0:
                    images = batch_cutout(images, size=self.cfg.cutout_size)

            yield images, labels
# %% 5. Trainer

class BackboneTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda")
        self.dtype = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            'fp32': torch.float32,
        }[self.cfg.dtype]
        
        self.train_loader = Cifar100Loader(cfg, train=True, device=self.device)
        self.test_loader = Cifar100Loader(cfg, train=False, device=self.device)

        self.model = BackboneResnet(cfg).to(
            device=self.device,
            memory_format=torch.channels_last
        )

        if cfg.optimizer == 'sgd':
            self.opt = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.max_lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                nesterov=cfg.nesterov,
                fused=True,
            )
        elif cfg.optimizer == 'adamw':
            self.opt = torch.optim.AdamW(
                self.model.parameters(),
                betas=(cfg.momentum, cfg.beta_2),
                lr=self.cfg.max_lr,
                weight_decay=cfg.weight_decay,
                fused=True,
            )
        else:
            raise ValueError(f"Invalid optimizer: {cfg.optimizer}")
        
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.opt,
            start_factor=1e-5,
            end_factor=1.0,
            total_iters=self.cfg.warmup_steps,
        )
        cosine_decay = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=self.cfg.train_steps - self.cfg.warmup_steps,
            eta_min=0.0,
        )

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.opt,
            schedulers=[warmup, cosine_decay],
            milestones=[self.cfg.warmup_steps],
        )
        
        if self.cfg.save_every > 0:
            os.makedirs(f"{BASE_DIR}/checkpoints", exist_ok=True)

    def train(self):
        self.model.train()

        loader_iter = iter(self.train_loader)
        pbar = tqdm(range(1, self.cfg.train_steps+1), desc="Training")
        training_time = 0.0
        
        # Start the clock.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        scaler = torch.GradScaler(enabled=(self.dtype == torch.float16))
        for step in pbar:
            # ---- Training ---- #

            # 1. Load our batch and labels.
            batch, labels = next(loader_iter)

            # 2. Forward pass.
            with torch.autocast(self.device.type, dtype=self.dtype):
                pred = self.model(batch)
                loss = F.cross_entropy(pred, labels)

            # 3. Backward pass.
            self.opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(self.opt)
            scaler.update()

            # 4. Update our learning rate.
            self.scheduler.step()
            
            if self.cfg.use_wandb:
                wandb.log({
                    'train_loss': loss.item(),
                    'lr': self.scheduler.get_last_lr()[0],
                }, step)

            # ---- Evaluation ---- #
            last_step = step == self.cfg.train_steps
            if last_step or self.cfg.eval_every > 0 and step % self.cfg.eval_every == 0:
                torch.cuda.synchronize()
                interval_time = time.perf_counter() - t0
                training_time += interval_time
                
                # Save a checkpoint.
                if self.cfg.save_every > 0 and (
                    last_step or step % self.cfg.save_every == 0
                ):
                    torch.save({
                        'model': self.model.state_dict(),
                        'optimizer': self.opt.state_dict(),
                        'step': step,
                    }, f"{BASE_DIR}/checkpoints/run-{self.cfg.run_id}-step-{step}.pt")
                
                self.model.eval()
                test_metrics = self.evaluate()
                self.model.train()
                
                # Our trainining metrics are only estimates (computed on a single batch).
                metrics = {
                    "step": step, 
                    "time": training_time,
                    "interval": interval_time,
                    "lr": self.opt.param_groups[0]['lr'],
                    "train_loss": loss.item(),
                    "train_acc1": (pred.argmax(dim=1) == labels).float().mean().item(),
                    "train_acc5": (pred.topk(5)[1] == labels.view(-1, 1)).any(dim=1).float().mean().item(),
                    **test_metrics,
                }
                logging.info(ROW_FMT.format(*[metrics[col] for col in LOGGING_COLUMNS]))
                pbar.set_postfix(train_loss=metrics['train_loss'], test_loss=metrics['test_loss'])

                if self.cfg.use_wandb:
                    del metrics['step']
                    wandb.log(metrics, step)
                
                # Start the clock again.
                torch.cuda.synchronize()
                t0 = time.perf_counter()

        torch.cuda.synchronize()
        training_time += time.perf_counter() - t0
        logging.info(f"Total training time: {training_time:,.2f}s")
    
    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        assert not self.model.training, "Model must be in eval mode"
        items = self.test_loader.n_images
        cum_loss = torch.tensor(0.0, device=self.device)
        n_correct_top1 = torch.tensor(0.0, device=self.device)
        n_correct_top5 = torch.tensor(0.0, device=self.device)

        pbar = tqdm(self.test_loader, desc="Evaluating", position=1, leave=False)
        for batch, labels in pbar:
            with torch.autocast(self.device.type, dtype=self.dtype):
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
    assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

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

    if cfg.use_wandb:
        def train_fn():
            name = datetime.now().isoformat(timespec="seconds")
            parsed_cfg = wandb.helper.parse_config(
                asdict(cfg), 
                exclude=('use_wandb', 'wandb_project', 'wandb_note', 'sweep_count')
            )

            with wandb.init(
                project=cfg.wandb_project,
                dir=f"{BASE_DIR}/wandb",
                config=parsed_cfg,
                name=name,
                notes=cfg.wandb_note,
                save_code=True,
            ) as run:
                logging.info(f"New run {name} with config {run.config}")
                run_cfg = Config(**run.config)
                logging.info(HEADER_FMT.format(*LOGGING_COLUMNS))
                logging.info(HEADER_FMT.format(*['---' for _ in LOGGING_COLUMNS]))

                trainer = BackboneTrainer(run_cfg)
                run.watch(trainer.model, log="all", log_freq=cfg.eval_every)
                trainer.train()

        sweep_config = {
            "method": "random",
            "metric": {"goal": "maximize", "name": "test_acc5"},
            "parameters": {
                "dtype": {"values": ["fp16", "bf16"]},
                "fc_scale": {"min": 0.03, "max": 1.0, "distribution": "log_uniform_values"},

                "optimizer": {"values": ["sgd", "adamw"]},
                "max_lr": {"min": 1e-4, "max": 0.5, "distribution": "log_uniform_values"},
                "warmup_steps": {"min": 0, "max": 600, "distribution": "int_uniform"},

                "weight_decay": {"min": 1e-6, "max": 0.2, "distribution": "uniform"},
                "momentum": {"min": 0.8, "max": 0.99, "distribution": "uniform"},
                "beta_2": {"min": 0.93, "max": 0.9999, "distribution": "uniform"},
                "nesterov": {"values": [True, False]},
            },
            "early_terminate": {
                "type": "hyperband",
                "max_iter": ...,
                "eta": ...,
                "s": ...,
            },
        }
        sweep_id = wandb.sweep(sweep_config, project=cfg.wandb_project)
        wandb.agent(sweep_id, train_fn, project=cfg.wandb_project, count=cfg.sweep_count)

    else:
        logging.info(HEADER_FMT.format(*LOGGING_COLUMNS))
        logging.info(HEADER_FMT.format(*['---' for _ in LOGGING_COLUMNS]))
        trainer = BackboneTrainer(cfg)
        trainer.train()
    
    try:
        smi = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.info(smi.stdout)
    except Exception as e:
        logging.info(f"Error running nvidia-smi: {e}")

    logging.info(f"Max memory allocated: {torch.cuda.max_memory_allocated() // 1024**2:,} MiB")
    logging.info(f"Max memory reserved: {torch.cuda.max_memory_reserved() // 1024**2:,} MiB")
    
    # Write this source code to our logs.
    with open(sys.argv[0]) as f:
        logging.info(f.read())
