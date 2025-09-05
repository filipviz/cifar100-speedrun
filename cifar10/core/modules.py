import torch
import torch.nn.functional as F
from torch import nn

class GhostBatchNorm(nn.BatchNorm2d):
    """Ghost BatchNorm using a naive chunked approach. Batch size must be divisible by num_splits (set drop_last=True)."""
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

        if affine:
            self.weight.requires_grad = requires_weight
            self.bias.requires_grad = requires_bias

        # nn.BatchNorm2d uses float32 for running_mean and running_var.
        self.register_buffer('running_mean', torch.zeros((self.num_splits, self.num_features), device=device, dtype=torch.float32))
        self.register_buffer('running_var',  torch.ones ((self.num_splits, self.num_features), device=device, dtype=torch.float32))

    def train(self, mode=True):
        # Lazily collate stats when we need to use them
        if (self.training is True) and (mode is False):
            with torch.no_grad():
                # correction=0 matches BatchNorm's convention
                var_of_means, mean_of_means = torch.var_mean(self.running_mean, dim=0, keepdim=True, correction=0)
                mean_of_vars = self.running_var.mean(0, keepdim=True)

                self.running_mean.copy_(mean_of_means.expand_as(self.running_mean))
                # Law of total variance: Var(x) = E[Var(x)] + Var(E[x])
                self.running_var.copy_((mean_of_vars + var_of_means).expand_as(self.running_var))
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return F.batch_norm(x, self.running_mean[0], self.running_var[0],
                self.weight, self.bias, False, 0.0, self.eps)

        N, C, _, _ = x.shape
        assert C == self.num_features, f"channels {C} must match num_features {self.num_features}"
        assert N % self.num_splits == 0, f"batch size {N} not divisible by num_splits {self.num_splits}"

        # chunk is view-based, and we preserve the channels_last layout.
        # This ends up being faster than vectorized approaches for num_features < 512.
        outs = [F.batch_norm(c, self.running_mean[i], self.running_var[i],
            self.weight, self.bias, True, self.momentum, self.eps
        ) for i, c in enumerate(x.chunk(self.num_splits))]
        return torch.cat(outs)