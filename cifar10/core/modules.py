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
        self.num_splits, self.num_features = num_splits, num_features

        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

        self.register_buffer('running_mean', torch.zeros(num_features*self.num_splits, device=device, dtype=dtype))
        self.register_buffer('running_var', torch.ones(num_features*self.num_splits, device=device, dtype=dtype))

    def train(self, mode=True):
        # Lazily collate stats when we need to use them
        if (self.training is True) and (mode is False):
            S, F = self.num_splits, self.num_features
            self.running_mean = self.running_mean.view(S, F).mean(dim=0).repeat(S)
            self.running_var = self.running_var.view(S, F).mean(dim=0).repeat(S)
        return super().train(mode)

    def forward(self, input):
        if self.training or not self.track_running_stats:
            N, C, H, W = input.shape
            stride = input.stride()
            assert N % self.num_splits == 0, f"batch size {N} not divisible by num_splits {self.num_splits}"
            return F.batch_norm(
                input.reshape(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).reshape(N, C, H, W).to(memory_format=torch.channels_last)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)