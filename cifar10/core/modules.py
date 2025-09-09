import torch
import torch.nn.functional as F
from torch import nn, Tensor
from jaxtyping import Float

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


class PCAWhiteningBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, images: Float[Tensor, "b c h w"]):
        super().__init__()
        patch_size = 3
        c_mid = c_in * patch_size * patch_size

        self.whitener = nn.Conv2d(c_in, c_mid,
            kernel_size=patch_size, padding=1, bias=False)
        self.pointwise = nn.Conv2d(c_mid, c_out, kernel_size=1, bias=False)
        self.gbn = GhostBatchNorm(c_out, num_splits=16, requires_weight=False)
        self.celu = nn.CELU(alpha=0.3)

        with torch.no_grad():
            self.whitener.weight.copy_(
                self._pca_weights(images[:5_000, :, 4:-4, 4:-4], patch_size)
            )
            self.whitener.weight.requires_grad_(False)

    @staticmethod
    @torch.inference_mode()
    def _pca_weights(
        images: Float[Tensor, "b c h w"],
        size: int = 3,
        eps: float = 1e-2
    ) -> Float[Tensor, "d c size size"]:
        """Produce the PCA-whitening convolution weights based on the input images."""
        c = images.size(1)
        h = w = size
        d = c * h * w

        # $X \in \mathbb{R}^{d \times N}$
        patches = images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, d).T
        # $\Sigma = E[X X^T] \in \mathbb{R}^{d \times d}$
        sigma = torch.cov(patches)
        # $Q \Lambda Q^T = \Sigma$
        evals, evecs = torch.linalg.eigh(sigma)
        # $W_\text{pca} = \Lambda^{-1/2} Q^T$
        W = (evals + eps).rsqrt().diag() @ evecs.T

        return W.reshape(d, c, h, w)

    def forward(self, x: Float[Tensor, "b c_in h w"]) -> Float[Tensor, "b c_out h w"]:
        out = self.whitener(x)
        out = self.pointwise(out)
        out = self.gbn(out)
        return self.celu(out)
