from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import Literal

import torch

@dataclass
class SharedCfg:
    device: Literal['cuda', 'mps', 'cpu'] = 'cuda'
    "MPS and CPU are only available for testing. Training requires CUDA."
    dtype: Literal['fp16', 'bf16', 'fp32'] = 'fp16'
    base_dir: str = os.getcwd()
    run_id: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # --- torch.compile --- #
    compile_enabled: bool = False
    compile_mode: Literal['default', 'reduce-overhead', 'max-autotune'] = 'default'

@dataclass
class TrainerCfg:
    # --- Training --- #
    train_steps: int = 1_500
    "Number of mini-batch iterations we train for."
    model_warmup_steps: int = 0
    "Number of (untimed) iterations to warm up for. Use when enabling torch.compile."
    eval_every: int = 100
    "Set to 0 to disable evaluation."
    checkpoint_every: int = 0
    "Set to 0 to disable checkpointing. Must be a multiple of eval_every."

    label_smoothing: float = 0.0
    "Label smoothing epsilon. Set to 0 to disable."

    ema_update_every: int = 0
    "We update the EMA model every n steps. Set to 0 to disable."
    ema_decay: float = 0.99

    # --- Wandb --- #
    use_wandb: bool = False
    wandb_project: str = "cifar10"
    wandb_note: str = ""
    sweep_count: int = 0
    "Number of sweeps to run. Set to 0 to disable sweeps."

    def __post_init__(self):
        if self.eval_every > 0:
            assert self.checkpoint_every % self.eval_every == 0, "checkpoint_every must be a multiple of eval_every"
        if self.use_wandb:
            assert self.wandb_project, "Must specify a wandb_project"


@dataclass
class GPULoaderCfg:
    batch_size: int = 768

    # --- Data Augmentation --- #
    normalize: bool = True
    flip: bool = True
    "Random horizontal flipping."
    pad_mode: Literal['reflect', 'constant'] = 'reflect'
    crop_padding: int = 4
    "Set to 0 to disable padding and random cropping."
    cutout_size: int = 8
    "Set to 0 to disable cutout."

@dataclass
class TorchLoaderCfg:
    batch_size: int = 128
    n_workers: int = 12

    # --- Data Augmentation --- #
    normalize_he: bool = True
    "Normalize with the per-pixel mean and without std, as in He et al 2015."
    normalize_torch: bool = False
    "Normalize with the per-channel mean and std, using torchvision's v2.Normalize."
    flip: bool = True
    "Random horizontal flipping."
    pad_mode: Literal['reflect', 'constant'] = 'constant'
    crop_padding: int = 4
    "Set to 0 to disable padding and random cropping."
    cutout_size: int = 0
    "Set to 0 to disable cutout."

    def __post_init__(self):
        assert not (self.normalize_he and self.normalize_torch), "Cannot normalize with both methods"



@dataclass
class ExperimentCfg:
    shared: SharedCfg = field(default_factory=SharedCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)
    loader: GPULoaderCfg = field(default_factory=GPULoaderCfg)

    # --- Setup and Flags --- #
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    float32_matmul_precision: Literal['highest', 'high', 'medium'] = 'medium'
    seed: int = 20250825
    "Set to 0 to disable seeding."

    def __post_init__(self):
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.set_float32_matmul_precision(self.float32_matmul_precision)
        if self.seed > 0:
            torch.manual_seed(self.seed) # seeds all devices
