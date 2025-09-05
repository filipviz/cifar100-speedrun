# %%
import torch
import torch.nn.functional as F
from torch import nn

from bench_utils import benchmark

# %%

class GBN_Vectorized(nn.BatchNorm2d):
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

class GBN_Chunked(nn.Module):
    def __init__(self, num_features: int, num_splits: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True,
                 track_running_stats: bool = True, device=None, dtype=None):
        super().__init__()
        if momentum is None:
            raise ValueError("GhostBatchNorm does not support momentum=None")

        self.num_features = int(num_features)
        self.num_splits = int(num_splits)
        self.eps, self.momentum = eps, momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(self.num_features, device=device, dtype=dtype))
            self.bias   = nn.Parameter(torch.zeros(self.num_features, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias',   None)

        # Per-split running stats (shape C*S)
        self.register_buffer('running_mean', torch.zeros(self.num_features * self.num_splits, device=device, dtype=dtype))
        self.register_buffer('running_var',  torch.ones (self.num_features * self.num_splits, device=device, dtype=dtype))

    def train(self, mode=True):
        # On train->eval, lazily average per-split stats so eval uses global stats
        if (self.training is True) and (mode is False) and self.track_running_stats:
            with torch.no_grad():
                m = self.running_mean.view(self.num_splits, self.num_features).mean(0)
                v = self.running_var .view(self.num_splits, self.num_features).mean(0)
                self.running_mean = m.repeat(self.num_splits)
                self.running_var  = v.repeat(self.num_splits)
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        S = self.num_splits
        assert C == self.num_features
        assert N % S == 0, f"batch size {N} not divisible by num_splits {S}"

        if self.training or not self.track_running_stats:
            chunks = x.chunk(S, dim=0)  # views; no copies for NCHW or NHWC
            outs = []
            for i, c in enumerate(chunks):
                rm = self.running_mean[i*C:(i+1)*C]
                rv = self.running_var [i*C:(i+1)*C]
                y = F.batch_norm(
                    c, rm, rv,
                    self.weight if self.affine else None,
                    self.bias   if self.affine else None,
                    True, self.momentum, self.eps
                )
                outs.append(y)
            return torch.cat(outs, dim=0).to(memory_format=torch.channels_last)
        else:
            # eval path uses the first C (already averaged in train(False))
            y = F.batch_norm(
                x, self.running_mean[:C], self.running_var[:C],
                self.weight if self.affine else None,
                self.bias   if self.affine else None,
                False, self.momentum, self.eps
            )
            return y.to(memory_format=torch.channels_last)

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
        self.gb_momentum = 1 - (1 - momentum) ** num_splits

        if affine:
            self.weight.requires_grad = requires_weight
            self.bias.requires_grad = requires_bias
        
        def _bn_per_split(x: torch.Tensor) -> torch.Tensor:
            return F.batch_norm(x, None, None, self.weight, self.bias, True, 0.0, self.eps)
        self._bn_per_split = torch.vmap(_bn_per_split)

    def forward(self, input):
        if not self.training:
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, 0.0, self.eps)

        B = input.size(0)
        assert B % self.num_splits == 0, f"batch size {B} not divisible by num_splits {self.num_splits}"
        with torch.no_grad():
            # The updates aren't exactly equivalent to standard batchnorm.
            v, m = torch.var_mean(input, dim=(0, -1, -2), correction=0)
            self.running_var.lerp_(v, self.gb_momentum)
            self.running_mean.lerp_(m, self.gb_momentum)
        
        return self._bn_per_split(
            input.unflatten(0, (self.num_splits, -1))
        ).flatten(0, 1).to(memory_format=torch.channels_last)

# %%

if __name__ == "__main__":
    impls = [GBN_Vectorized, GBN_Chunked, GBN_Page, GBN_vmap]

    device = 'cuda'
    batch_size = 512
    ghost_batch_size = 32
    features = 128
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
        iters=1_000,
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
        iters=1_000,
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
