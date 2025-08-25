from dataclasses import dataclass, field

@dataclass
class GPULoaderCfg:
    ...

class GPULoader:
    """GPU-accelerated data loader for CIFAR-10."""
    def __init__(self, cfg: GPULoaderCfg):
        pass
    
    def __len__(self):
        pass
    
    def __iter__(self):
        pass
    
# Data augmentation methods and utils below.