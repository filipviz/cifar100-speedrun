# Benchmark different cropping methods.
# %%
import torch
from torch import Tensor
from jaxtyping import Float
from torchvision.datasets import CIFAR10
from einops import rearrange
import torch.nn.functional as F

from bench_utils import benchmark

def batch_crop_vec(images: Float[Tensor, "b c h_in w_in"], crop_size: int = 32) -> Float[Tensor, "b c h_out w_out"]:
    """Vectorized cropping using index grids."""
    n, c, h, w = images.shape
    r = (w - crop_size) // 2
    shifts = torch.randint(-r, r+1, size=(n, 2), device=images.device)

    # Convert shifts to absolute positions
    y = r + shifts[:, 0]
    x = r + shifts[:, 1]

    # Create index grids
    i = torch.arange(n, device=images.device).view(-1, 1, 1, 1)
    j = torch.arange(c, device=images.device).view(1, -1, 1, 1)
    yy = torch.arange(crop_size, device=images.device).view(1, 1, -1, 1)
    xx = torch.arange(crop_size, device=images.device).view(1, 1, 1, -1)
    y = y.view(-1, 1, 1, 1)
    x = x.view(-1, 1, 1, 1)

    return images[i, j, y + yy, x + xx]

def batch_crop_keller(images: Float[Tensor, "b c h_in w_in"], crop_size: int = 32) -> Float[Tensor, "b c h_out w_out"]:
    """
    Keller's method with conditional strategies.
    Uses grouping for small ranges, two-pass for medium ranges.
    Note that this has a bug! It does not preserve channels_last format.
    TODO: Submit a PR to Keller's repo to fix this.
    """
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

def batch_crop_fast(images: Float[Tensor, "b c h_in w_in"], crop_size: int = 32) -> Float[Tensor, "b c h_out w_out"]:
    """Strided view-based batch cropping."""
    b, c, h, w = images.shape
    r = (h - crop_size) // 2

    # Create strided views of all possible crops.
    b_s, c_s, h_s, w_s = images.stride()
    crops_shape = (b, c, 2*r+1, 2*r+1, crop_size, crop_size)
    crops_stride = (b_s, c_s, h_s, w_s, h_s, w_s)
    crops = torch.as_strided(
        images[:, :, :h-crop_size+1, :w-crop_size+1],
        size=crops_shape, stride=crops_stride
    )

    # Select the appropriate crop for each image.
    batch_idx = torch.arange(b, device=images.device)
    shift_h = torch.randint(0, 2*r+1, size=(b,), device=images.device)
    shift_w = torch.randint(0, 2*r+1, size=(b,), device=images.device)
    return crops[batch_idx, :, shift_h, shift_w]

# %%

if __name__ == '__main__':
    assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

    cifar = CIFAR10(root='data', train=True, download=True)
    imgs = torch.tensor(cifar.data).to('cuda')
    imgs = rearrange(imgs, 'b h w c -> b c h w').to(torch.float16, memory_format=torch.channels_last) / 255.0

    # Methods assume images are already padded.
    img_pad = F.pad(imgs, (4,) * 4, mode='constant', value=0)

    variants = [
        ("batch_crop_vec", batch_crop_vec),
        ("batch_crop_keller", batch_crop_keller),
        ("batch_crop_fast", batch_crop_fast),
    ]

    def make_inputs():
        return (img_pad.clone(), 32)

    benchmark(
        variants=variants,
        make_inputs=make_inputs,
        iters=10,
        warmup=5,
    )
