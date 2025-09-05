import os
import torch
from torch import Tensor
from jaxtyping import Float
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
from einops import rearrange

from .config import GPULoaderCfg, SharedCfg


class GPULoader:
    """
    GPU-accelerated data loader for CIFAR-10. Loads entire dataset into GPU memory, performs transformations on GPU, and doesn't require re-initialization each epoch. Ignores the shared_cfg.dtype.
    """
    def __init__(self, train: bool, cfg: GPULoaderCfg, shared_cfg: SharedCfg):
        self.cfg = cfg
        self.batch_size, self.train = cfg.batch_size, train

        self.device = shared_cfg.device
        data_dir = os.path.join(shared_cfg.base_dir, 'data')

        self.cifar10_mean = torch.tensor((0.4914, 0.4822, 0.4465), device=self.device, dtype=torch.float32)
        self.cifar10_std = torch.tensor((0.2470, 0.2435, 0.2616), device=self.device, dtype=torch.float32)

        # Load or download data
        cifar_path = os.path.join(data_dir, 'train.pt' if self.train else 'test.pt')
        if not os.path.exists(cifar_path):
            np_cifar = CIFAR10(root=data_dir, train=train, download=True)
            images = torch.tensor(np_cifar.data)
            labels = torch.tensor(np_cifar.targets)
            torch.save({'images': images, 'labels': labels, 'classes': np_cifar.classes}, cifar_path)

        # Transfer as uint8 then convert on GPU. This is faster than loading pre-processed fp16 data.
        data = torch.load(cifar_path, map_location=self.device)
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']

        # Convert to floats, normalize, and rearrange on GPU.
        # I tried doing this in fp16 but it didn't make a difference.
        self.images = self.images.to(torch.float32) / 255.0
        if cfg.normalize:
            self.images = (self.images - self.cifar10_mean) / self.cifar10_std
        if train and cfg.crop_padding > 0:
            self.images = F.pad(self.images, (0, 0) + (cfg.crop_padding,) * 4, mode=cfg.pad_mode)
        self.images = rearrange(self.images, "b h w c -> b c h w").to(memory_format=torch.channels_last)

        self.n_images = len(self.images)
        assert self.batch_size <= 2 * self.n_images, "To support this batch size you have to update __iter__"

        if train:
            disable = not shared_cfg.compile_enabled
            @torch.compile(mode=shared_cfg.compile_mode, fullgraph=True, disable=disable)
            def augment(images: Float[Tensor, "b c h_in w_in"]) -> Float[Tensor, "b c h_out w_out"]:
                if cfg.crop_padding > 0:
                    images = batch_crop(images, crop_size=32)
                if cfg.flip:
                    images = batch_flip_lr(images)
                if cfg.cutout_size > 0:
                    images = batch_cutout(images, size=cfg.cutout_size)
                return images

            self.augment = augment

    def __len__(self):
        # Needed for tqdm to work when self.train=False.
        # math.ceil(self.n_images / self.batch_size)
        assert not self.train, "__len__ is not defined for a GPULoader with train=True"
        return (self.n_images + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        position = 0
        if not self.train:
            while position < self.n_images:
                next_position = position + self.batch_size
                images = self.images[position:next_position]
                labels = self.labels[position:next_position]
                position = next_position
                yield images, labels
            return

        indices = torch.randperm(self.n_images, device=self.device)
        while True:
            last_batch = position + self.batch_size > self.n_images
            # If we need to wrap around, we combine remaining indices with indices from the new epoch.
            if last_batch:
                remaining = indices[position:]
                indices = torch.randperm(self.n_images, device=self.device)
                needed = self.batch_size - len(remaining)
                batch_indices = torch.cat([remaining, indices[:needed]])
                position = needed
            else:
                # Otherwise, we take the next batch of indices from the current epoch.
                batch_indices = indices[position:position + self.batch_size]
                position += self.batch_size

            images = self.images[batch_indices]
            labels = self.labels[batch_indices]

            images = self.augment(images)
            yield images, labels


def batch_crop(images: Float[Tensor, "b c h_in w_in"], crop_size: int = 32) -> Float[Tensor, "b c h_out w_out"]:
    """Strided view-based (in-place) batch cropping."""
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

def batch_flip_lr(images: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
    """Apply random horizontal flipping to each image in the batch"""
    flip_mask = torch.rand(len(images), device=images.device) < 0.5
    images[flip_mask] = images[flip_mask].flip(-1)
    return images

def batch_cutout(images: Float[Tensor, "b c h w"], size: int) -> Float[Tensor, "b c h w"]:
    """In-place vectorized cutout using advanced indexing."""
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
