# Benchmark different horizontal flip implementations. All in 2-4 ms range. No major speedups to be had.
# %%
import torch
from torch import Tensor
from jaxtyping import Float
from torchvision.datasets import CIFAR10
from einops import rearrange

from bench_utils import benchmark

# %%

def batch_flip_naive(images: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
    flip_mask = (torch.rand(len(images), device=images.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, images.flip(-1), images)

def batch_flip_bool(images: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
    b = images.size(0)
    flip_mask = torch.rand(b, device=images.device) < 0.5
    images[flip_mask] = images[flip_mask].flip(-1)
    return images

def batch_flip_gather(images: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
    b, c, h, w = images.shape
    flip_mask = torch.rand(b, device=images.device) < 0.5
    ws = torch.arange(w, device=images.device)
    ws_flipped = ws.flip(0)

    indices = torch.where(
        flip_mask.view(-1, 1),
        ws_flipped,
        ws
    )

    # Expand indices for all batch items
    indices = indices.view(b, 1, 1, w).expand(-1, c, h, -1)

    return images.gather(-1, indices)

# %%

if __name__ == '__main__':
    assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."
    device, dtype = 'cuda', torch.float16

    cifar = CIFAR10(root='data', train=True, download=True)
    imgs = torch.tensor(cifar.data).to(device)
    imgs = rearrange(imgs, 'b h w c -> b c h w').to(dtype, memory_format=torch.channels_last) / 255.0

    variants = [
        ("batch_flip_naive", batch_flip_naive),
        ("batch_flip_bool", batch_flip_bool),
        ("batch_flip_gather", batch_flip_gather),
    ]

    def make_inputs():
        return (imgs.clone(),)

    benchmark(
        variants=variants,
        make_inputs=make_inputs,
        iters=100,
        warmup=10,
    )
