# %% [markdown]
"""
Benchmarking different cutout methods. On a 3060:
- The naive implementation averages ~4.78 s.
- The mask-based implementation cuts this down to 3.90 ms.
- Our final indexing-based implementation brings us to 790.73 us.
A 6045x speedup!
"""
# %%
import torch
from torch import Tensor
from jaxtyping import Float
from torchvision.datasets import CIFAR10
from einops import rearrange

from bench_utils import benchmark

# %%

def batch_cutout_naive(images: Float[Tensor, "b c h w"], size: int) -> Float[Tensor, "b c h w"]:
    """Naive loop-based implementation of cutout."""
    if size <= 0:
        return images

    batch_size, _, h, w = images.shape
    cutout_y = torch.randint(0, h, (batch_size,), device=images.device)
    cutout_x = torch.randint(0, w, (batch_size,), device=images.device)

    y1 = torch.clamp(cutout_y - size // 2, 0, h)
    y2 = torch.clamp(cutout_y + size // 2, 0, h)
    x1 = torch.clamp(cutout_x - size // 2, 0, w)
    x2 = torch.clamp(cutout_x + size // 2, 0, w)

    images = images.clone()
    for i in range(batch_size):
        images[i, :, y1[i]:y2[i], x1[i]:x2[i]] = 0

    return images

def batch_cutout_mask(images: Float[Tensor, "b c h w"], size: int) -> Float[Tensor, "b c h w"]:
    """Mask-based cutout implementation."""
    if size <= 0:
        return images
    b, c, h, w = images.shape
    dev = images.device

    lo = size // 2
    hi = size - lo
    center_h = torch.randint(0, h, (b, 1, 1, 1), device=dev)
    center_w = torch.randint(0, w, (b, 1, 1, 1), device=dev)

    min_h = torch.clamp_min(center_h - lo, 0)
    min_w = torch.clamp_min(center_w - lo, 0)

    max_h = torch.clamp_max(center_h + hi, h - 1)
    max_w = torch.clamp_max(center_w + hi, w - 1)

    hs = torch.arange(h, device=dev).view(1, 1, h, 1)
    ws = torch.arange(w, device=dev).view(1, 1, 1, w)

    mask = (hs >= min_h) & (hs < max_h) & (ws >= min_w) & (ws < max_w)
    images.masked_fill_(mask, 0)

    return images

def batch_cutout_sparse(images: Float[Tensor, "b c h w"], size: int) -> Float[Tensor, "b c h w"]:
    """Vectorized cutout using advanced indexing."""
    if size <= 0:
        return images
    b, c, h, w = images.shape
    dev = images.device

    lo = size // 2
    hi = size - lo

    h_center = torch.randint(0, h, (b, 1, 1), device=dev)
    w_center = torch.randint(0, w, (b, 1, 1), device=dev)

    dh = torch.arange(-lo, hi, device=dev).view(1, size, 1)
    dw = torch.arange(-lo, hi, device=dev).view(1, 1, size)

    h_idx = (h_center + dh).clamp(0, h - 1)
    w_idx = (w_center + dw).clamp(0, w - 1)
    b_idx = torch.arange(b, device=dev).view(b, 1, 1)

    images[b_idx, :, h_idx, w_idx] = 0
    return images

# %%

if __name__ == '__main__':
    device, dtype = 'cuda', torch.float16

    cifar = CIFAR10(root='data', train=True, download=True)
    imgs = torch.tensor(cifar.data).to(device)
    imgs = rearrange(imgs, 'b h w c -> b c h w').to(dtype, memory_format=torch.channels_last) / 255.0

    variants = [
        # batch_cutout_naive is extremely slow.
        # ("batch_cutout_naive", batch_cutout_naive),
        ("batch_cutout_mask", batch_cutout_mask),
        ("batch_cutout_sparse", batch_cutout_sparse),
    ]

    def make_inputs():
        return (imgs.clone(), 8)

    benchmark(
        variants=variants,
        make_inputs=make_inputs,
        iters=100,
        warmup=10,
    )
