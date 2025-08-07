# Prototyping and experimenting with our data loader.
# %%
import os
from typing import Literal
import torch
from torch import Tensor
from jaxtyping import Float
from torchvision.datasets import CIFAR100
from einops import rearrange
from dataclasses import dataclass
import torch.nn.functional as F

assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

torch.backends.cudnn.benchmark = True

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = f"{BASE_DIR}/data"
DEVICE = 'cuda'
CIFAR100_MEAN = torch.tensor((0.5078125, 0.486328125, 0.44140625), dtype=torch.bfloat16, device=DEVICE)
CIFAR100_STD = torch.tensor((0.267578125, 0.255859375, 0.275390625), dtype=torch.bfloat16, device=DEVICE)

# %%

@dataclass
class Config:
    """Configuration for data augmentation"""
    flip: bool = True
    pad_mode: Literal['reflect', 'constant'] = 'reflect'
    crop_padding: int = 4
    "Set to 0 to disable padding and random cropping."
    cutout_size: int = 0
    "Set to 0 to disable cutout."

def batch_crop(images: Float[Tensor, "b c h_in w_in"], crop_size: int = 32) -> Float[Tensor, "b c h_out w_out"]:
    """View-based batch cropping."""
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
    shifts = torch.randint(0, 2*r+1, size=(2, b), device=images.device)
    return crops[batch_idx, :, shifts[0], shifts[1]]

def batch_flip_lr(images: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
    """Apply random horizontal flipping to each image in the batch"""
    flip_mask = (torch.rand(len(images), device=images.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, images.flip(-1), images)

def batch_cutout(images: Float[Tensor, "b c h w"], size: int) -> Float[Tensor, "b c h w"]:
    """
    Apply cutout augmentation to batch of images
    TODO: If we actually use this we should make it more efficient.
    """
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

class Cifar100Loader:
    """
    Infinite data loader for CIFAR-100. Loads entire dataset into GPU memory, performs transformations on GPU, and doesn't require re-initialization each epoch.
    """
    def __init__(
        self,
        path: str,
        batch_size: int,
        train: bool = True,
        cfg: Config = None,
        device: str = 'cuda',
    ):
        self.device = device
        self.batch_size = batch_size
        self.train = train
        self.cfg = cfg if cfg is not None else Config()

        # Load or download data
        cifar_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(cifar_path):
            np_cifar = CIFAR100(root=path, train=train, download=True)
            images = torch.tensor(np_cifar.data)
            labels = torch.tensor(np_cifar.targets)
            torch.save({'images': images, 'labels': labels, 'classes': np_cifar.classes}, cifar_path)

        # Transfer as uint8 then convert on GPU. This is faster than loading pre-processed bf16 data.
        data = torch.load(cifar_path, map_location=device)
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        
        # Convert to bf16, normalize, and rearrange on GPU
        self.images = self.images.bfloat16() / 255.0
        self.images = (self.images - CIFAR100_MEAN) / CIFAR100_STD
        self.images = rearrange(self.images, "b h w c -> b c h w").to(memory_format=torch.channels_last)
        
        self.n_images = len(self.images)
        
    def __iter__(self):
        indices = torch.randperm(self.n_images, device=self.device)
        pos = 0
        
        while True:
            # If we need to wrap around, we combine remaining indices with indices from the new epoch.
            if pos + self.batch_size > self.n_images:
                remaining = indices[pos:]
                indices = torch.randperm(self.n_images, device=self.device)
                needed = self.batch_size - len(remaining)
                batch_indices = torch.cat([remaining, indices[:needed]])
                pos = needed
            else:
                # Otherwise, we take the next batch of indices from the current epoch.
                batch_indices = indices[pos:pos + self.batch_size]
                pos += self.batch_size
            
            images = self.images[batch_indices]
            labels = self.labels[batch_indices]
            
            if self.train:
                if self.cfg.crop_padding > 0:
                    images = F.pad(images, (self.cfg.crop_padding,) * 4, mode=self.cfg.pad_mode)
                    images = batch_crop(images, crop_size=32)
                if self.cfg.flip:
                    images = batch_flip_lr(images)
                if self.cfg.cutout_size > 0:
                    images = batch_cutout(images, self.cfg.cutout_size)
            
            yield images, labels
# %%
