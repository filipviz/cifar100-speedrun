# Benchmark different cutout methods.
# %%
import torch
from torch.utils import benchmark
from torch import Tensor
from jaxtyping import Float
from torchvision.datasets import CIFAR100
from einops import rearrange

assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

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
    center_h, center_w = torch.randint(0, w, (2, b, 1, 1, 1), device=dev)

    min_h = torch.clamp_min(center_h - lo, 0)
    min_w = torch.clamp_min(center_w - lo, 0)

    # We don't need to clamp max values - PyTorch automatically handles this.
    max_h = center_h + hi
    max_w = center_w + hi
    
    hs = torch.arange(h, device=dev).view(1, 1, h, 1)
    ws = torch.arange(w, device=dev).view(1, 1, 1, w)
    
    mask = (hs >= min_h) & (hs < max_h) & (ws >= min_w) & (ws < max_w)
    images.masked_fill_(mask, 0)

    return images

def batch_cutout_sparse(images: Float[Tensor, "b c h w"], size: int) -> Float[Tensor, "b c h w"]:
    if size <= 0:
        return images
    b, c, h, w = images.shape
    dev = images.device
    
    lo = size // 2
    hi = size - lo
    center_h, center_w = torch.randint(0, w, (2, b, 1, 1, 1), device=dev)
    
    hs = center_h + torch.arange(-lo, hi, device=dev).view(1, size, 1)
    ws = center_w + torch.arange(-lo, hi, device=dev).view(1, 1, size)
    
    b_idx = torch.arange(b, device=dev).view(b, 1, 1, 1)
    c_idx = torch.arange(c, device=dev).view(1, c, 1, 1)
    
    images[b_idx, c_idx, hs, ws] = 0
    return images

# %% 

device, dtype = 'cuda', torch.float16

cifar = CIFAR100(root='../data', train=True, download=True)
imgs = torch.tensor(cifar.data).to(device)
imgs = rearrange(imgs, 'b h w c -> b c h w') #.to(dtype, memory_format=torch.channels_last) / 255.0

for impl in [batch_cutout_naive, batch_cutout_mask, batch_cutout_sparse]:
    t = benchmark.Timer(
        stmt='impl(images, 8)',
        setup='images = imgs.clone()',
        globals={'impl': impl, 'imgs': imgs},
    )
    print(f"{impl.__name__}: {t.timeit(10)}")