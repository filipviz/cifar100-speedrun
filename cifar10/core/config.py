from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import Literal

import torch
from .gpuloader import GPULoaderCfg
from .torchloader import TorchLoaderCfg
from .trainer import TrainerCfg

@dataclass
class SharedCfg:
    device: Literal['cuda', 'mps', 'cpu'] = 'cuda'
    "MPS and CPU are only available for testing. Training requires CUDA."
    dtype: Literal['fp16', 'bf16', 'fp32'] = 'fp16'
    base_dir: str = os.getcwd()
    run_id: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

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
