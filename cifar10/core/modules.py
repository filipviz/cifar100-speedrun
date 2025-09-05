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
        requires_weight: bool = True,
        requires_bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        if momentum is None:
            raise ValueError("GhostBatchNorm does not support CMA via momentum=None")

        super().__init__(num_features, eps, momentum, affine, True, device, dtype)
        self.num_splits = num_splits

        self.register_buffer('running_mean', torch.zeros((self.num_splits, self.num_features), device=device, dtype=dtype))
        self.register_buffer('running_var',  torch.ones ((self.num_splits, self.num_features), device=device, dtype=dtype))

    def train(self, mode=True):
        # Lazily collate stats when we need to use them
        if (self.training is True) and (mode is False):
            with torch.no_grad():
                self.running_mean.copy_(self.running_mean.mean(0, keepdim=True).expand_as(self.running_mean))
                self.running_var.copy_(self.running_var .mean(0, keepdim=True).expand_as(self.running_var))
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return F.batch_norm(x, self.running_mean[0], self.running_var[0],
                self.weight, self.bias, False, self.momentum, self.eps)

        N, C, _, _ = x.shape
        assert C == self.num_features, f"channels {C} must match num_features {self.num_features}"
        assert N % self.num_splits == 0, f"batch size {N} not divisible by num_splits {self.num_splits}"

        # chunk and cat are both view-based, so we preserve the channels_last layout.
        # This ends up being faster than vectorized approaches.
        outs = [F.batch_norm(c, self.running_mean[i], self.running_var[i],
            self.weight, self.bias, True, self.momentum, self.eps
        ) for i, c in enumerate(x.chunk(self.num_splits))]
        return torch.cat(outs)