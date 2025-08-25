import torch
from dataclasses import dataclass, field

@dataclass
class TrainerCfg:
    ...

class Trainer:
    def __init__(self, cfg: TrainerCfg):
        pass
    
    def train(self):
        pass
    
    @torch.inference_mode()
    def evaluate(self):
        pass