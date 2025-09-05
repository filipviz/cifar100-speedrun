# Benchmarking different ghost batchnorm implementations.
# These implementations aren't equivalent and have several issues (failing to account for the total law of variance, for example).
# Used to build performance intuitions around the forward pass - not perfectly rigorous.
# %%
import torch
import torch.nn.functional as F
from torch import nn

from bench_utils import benchmark

# %%

class GBN_Vectorized(nn.BatchNorm2d):
    """Uses reshape such that the forward pass requires a single call to F.batch_norm."""
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
        self.num_splits, self.num_features = num_splits, num_features

        self.weight.requires_grad = requires_weight
        self.bias.requires_grad = requires_bias

        self.register_buffer('running_mean', torch.zeros(num_features*self.num_splits, device=device, dtype=dtype))
        self.register_buffer('running_var', torch.ones(num_features*self.num_splits, device=device, dtype=dtype))

    def train(self, mode=True):
        # Lazily collate stats when we need to use them
        # Note that this doesn't account for the total law of variance.
        if (self.training is True) and (mode is False):
            S, F = self.num_splits, self.num_features
            self.running_mean = self.running_mean.view(S, F).mean(dim=0).repeat(S)
            self.running_var = self.running_var.view(S, F).mean(dim=0).repeat(S)
        return super().train(mode)

    def forward(self, input):
        if self.training or not self.track_running_stats:
            N, C, H, W = input.shape
            assert N % self.num_splits == 0, f"batch size {N} not divisible by num_splits {self.num_splits}"
            return F.batch_norm(
                input.reshape(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).reshape(N, C, H, W).to(memory_format=torch.channels_last)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)

class GBN_Chunked(nn.BatchNorm2d):
    """Uses a naive for loop over the chunked input and concatenates the results."""
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

class BN_Page(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, requires_weight=True, requires_bias=True, device=None, dtype=None):
        super().__init__(num_features, eps=eps, momentum=momentum, device=device, dtype=dtype)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = requires_weight
        self.bias.requires_grad = requires_bias

class GBN_Page(BN_Page):
    """Page's implementation doesn't support channels_last, so we've slightly modified it."""
    def __init__(self, num_features, num_splits, device=None, dtype=None, **kw):
        super().__init__(num_features, device=device, dtype=dtype, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features*self.num_splits, device=device, dtype=dtype))
        self.register_buffer('running_var', torch.ones(num_features*self.num_splits, device=device, dtype=dtype))

    def train(self, mode=True):
        # Note that Page's implementation doesn't account for the total law of variance here!
        if (self.training is True) and (mode is False): #lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.reshape(-1, C*self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).reshape(N, C, H, W).to(memory_format=torch.channels_last)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)

class GBN_vmap(nn.BatchNorm2d):
    """Uses vmap to vectorize the F.batch_norm call over the input."""
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

        self.register_buffer('running_mean', torch.zeros((self.num_splits, num_features), device=device, dtype=dtype))
        self.register_buffer('running_var', torch.ones((self.num_splits, num_features), device=device, dtype=dtype))
        
        def _bn_per_split(x: torch.Tensor, rm: torch.Tensor, rv: torch.Tensor) -> torch.Tensor:
            return F.batch_norm(x, rm, rv, self.weight, self.bias, True, self.momentum, self.eps)
        self._bn_per_split = torch.vmap(_bn_per_split)

    def train(self, mode=True):
        # Lazily collate stats when we need to use them
        if (self.training is True) and (mode is False):
            with torch.no_grad():
                self.running_mean.copy_(self.running_mean.mean(dim=0, keepdim=True).expand_as(self.running_mean))
                self.running_var.copy_(self.running_var.mean(dim=0, keepdim=True).expand_as(self.running_var))
        return super().train(mode)

    def forward(self, input):
        if not self.training:
            return F.batch_norm(
                input, self.running_mean[0], self.running_var[0],
                self.weight, self.bias, False, 0.0, self.eps)

        B = input.size(0)
        assert B % self.num_splits == 0, f"batch size {B} not divisible by num_splits {self.num_splits}"
        
        return self._bn_per_split(
            input.unflatten(0, (self.num_splits, -1)),
            self.running_mean,
            self.running_var,
        ).flatten(0, 1).to(memory_format=torch.channels_last)

# %%

if __name__ == "__main__":
    impls = [GBN_Vectorized, GBN_Chunked, GBN_Page, GBN_vmap]

    device = 'cuda'
    batch_size = 512
    ghost_batch_size = 32
    features = 512
    num_splits = batch_size // ghost_batch_size
    fmt = torch.channels_last
    dtype = torch.float16

    x = torch.randn((batch_size, features, 32, 32)).to(device, dtype, memory_format=fmt)

    # Train-time
    train_variants = []
    bn_train = nn.BatchNorm2d(features, device=device, dtype=dtype).train()
    train_variants.append(("BatchNorm2d.train", bn_train))

    for impl in impls:
        gbn = impl(num_features=features, num_splits=num_splits, device=device, dtype=dtype).train()
        train_variants.append((f"{impl.__name__}.train", gbn))

    def make_inputs_train():
        return (x.clone(memory_format=fmt),)

    print("=== train ===")
    benchmark(
        variants=train_variants,
        make_inputs=make_inputs_train,
        iters=100,
        warmup=20,
    )

    # Eval-time
    eval_variants = []
    bn_eval = nn.BatchNorm2d(features, device=device, dtype=dtype).eval()
    eval_variants.append(("BatchNorm2d.eval", bn_eval))

    for impl in impls:
        gbn = impl(num_features=features, num_splits=num_splits, device=device, dtype=dtype).eval()
        eval_variants.append((f"{impl.__name__}.eval", gbn))

    def make_inputs_eval():
        return (x.clone(memory_format=fmt),)

    print("=== eval ===")
    benchmark(
        variants=eval_variants,
        make_inputs=make_inputs_eval,
        iters=100,
        warmup=20,
    )

    # Basic correctness checks for ghost BN variants
    bn = nn.BatchNorm2d(features, device=device, dtype=dtype)
    y_bn = bn(x)
    for impl in impls:
        gbn = impl(num_features=features, num_splits=num_splits, device=device, dtype=dtype)
        y_gbn = gbn(x)
        assert x.is_contiguous(memory_format=fmt), f"{x.is_contiguous(memory_format=fmt)=}"
        assert y_gbn.is_contiguous(memory_format=fmt), f"{impl.__name__} is not {fmt}"
        assert x.shape == y_gbn.shape, f"{x.shape=}, {y_gbn.shape=}"

        gbn_single = impl(num_features=features, num_splits=1, device=device, dtype=dtype)
        y_gbn_single = gbn_single(x)
        torch.testing.assert_close(y_bn, y_gbn_single)
