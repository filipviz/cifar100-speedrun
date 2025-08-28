import torch
import torch.nn.functional as F
from torch import nn

class GhostBatchNorm(nn.BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        num_splits: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        weight: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        if momentum is None:
            raise ValueError("GhostBatchNorm does not support CMA via momentum=None")

        super().__init__(num_features, eps, momentum, affine, True, device, dtype)

        self.num_splits = num_splits
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

        self.register_buffer('running_mean', torch.zeros(num_features*self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features*self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(
                self.running_mean.view(self.num_splits, self.num_features), dim=0
            ).repeat(self.num_splits)
            self.running_var = torch.mean(
                self.running_var.view(self.num_splits, self.num_features), dim=0
            ).repeat(self.num_splits)
        return super().train(mode)
        
    def forward(self, input):
        if self.training or not self.track_running_stats:
            N, C, H, W = input.shape
            assert N % self.num_splits == 0, f"batch size {N} not divisible by num_splits {self.num_splits}"
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W) 
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features], 
                self.weight, self.bias, False, self.momentum, self.eps)