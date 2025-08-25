from dataclasses import dataclass, field
from .gpuloader import GPULoaderCfg
from .torchloader import TorchLoaderCfg
from .trainer import TrainerCfg

@dataclass
class ExperimentCfg:
    ...