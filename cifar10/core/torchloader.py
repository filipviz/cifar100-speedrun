from dataclasses import dataclass, field

@dataclass
class TorchLoaderCfg:
    ...

class TorchLoader:
    """Simple PyTorch DataLoader wrapper for CIFAR-10."""
    def __init__(self, cfg: TorchLoaderCfg):
        pass
    
    def __len__(self):
        pass
    
    def __iter__(self):
        pass
    
# Data augmentation methods and utils below.