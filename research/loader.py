# Prototyping and experimenting with our data loader.
# %%
import os
from typing import Literal
import torch
from torch import Tensor
from jaxtyping import Float
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2
from einops import rearrange
from dataclasses import dataclass
import torch.nn.functional as F

assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

torch.backends.cudnn.benchmark = True

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = f"{BASE_DIR}/data"
DEVICE = 'cuda'
DTYPE = torch.float16
CIFAR100_MEAN = torch.tensor((0.5078125, 0.486328125, 0.44140625), dtype=DTYPE, device=DEVICE)
CIFAR100_STD = torch.tensor((0.267578125, 0.255859375, 0.275390625), dtype=DTYPE, device=DEVICE)

# %%

@dataclass
class Config:
    """Configuration for data augmentation"""
    batch_size: int = 128
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
    def __init__(self, cfg: Config, train: bool = True, device: str = DEVICE):
        self.cfg = cfg
        self.device, self.train = device, train
        self.batch_size = cfg.batch_size

        # Load or download data
        cifar_path = os.path.join(DATA_DIR, 'train.pt' if train else 'test.pt')
        if not os.path.exists(cifar_path):
            np_cifar = CIFAR100(root=DATA_DIR, train=train, download=True)
            images = torch.tensor(np_cifar.data)
            labels = torch.tensor(np_cifar.targets)
            torch.save({'images': images, 'labels': labels, 'classes': np_cifar.classes}, cifar_path)

        # Transfer as uint8 then convert on GPU. This is faster than loading pre-processed bf16 data.
        data = torch.load(cifar_path, map_location=device)
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        
        # Convert to bf16, normalize, and rearrange on GPU
        self.images = self.images.to(DTYPE) / 255.0
        self.images = (self.images - CIFAR100_MEAN) / CIFAR100_STD
        if self.train and cfg.crop_padding > 0:
            self.images = F.pad(self.images, (0, 0) + (cfg.crop_padding,) * 4, mode=cfg.pad_mode)
        self.images = rearrange(self.images, "b h w c -> b c h w").to(memory_format=torch.channels_last)
        
        self.n_images = len(self.images)
    
    def __len__(self):
        # math.ceil(self.n_images / self.batch_size)
        return (self.n_images + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        indices = torch.randperm(self.n_images, device=self.device)
        position = 0
        loop = True

        while loop:
            last_batch = position + self.batch_size > self.n_images
            # If we need to wrap around, we combine remaining indices with indices from the new epoch.
            if last_batch and self.train:
                remaining = indices[position:]
                indices = torch.randperm(self.n_images, device=self.device)
                needed = self.batch_size - len(remaining)
                batch_indices = torch.cat([remaining, indices[:needed]])
                position = needed
            else:
                # Otherwise, we take the next batch of indices from the current epoch.
                batch_indices = indices[position:position + self.batch_size]
                position += self.batch_size
                if last_batch and not self.train:
                    loop = False
            
            images = self.images[batch_indices]
            labels = self.labels[batch_indices]
            
            if self.train:
                if self.cfg.crop_padding > 0:
                    images = batch_crop(images, crop_size=32)
                if self.cfg.flip:
                    images = batch_flip_lr(images)

            yield images, labels

# %% 

train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(DTYPE, scale=True),
    v2.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(DTYPE, scale=True),
    v2.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

# %%


if __name__ == "__main__":
    from torch.profiler import profile, record_function

    cfg = Config()

    with profile(with_stack=True, profile_memory=True) as prof:
        with record_function("init_torch_train_loader"):
            torch_train_loader = DataLoader(
                CIFAR100(root=DATA_DIR, train=True, download=True, transform=test_transform),
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=12,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
            )
        with record_function("init_torch_test_loader"):
            torch_test_loader = DataLoader(
                CIFAR100(root=DATA_DIR, train=False, download=True, transform=test_transform),
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=12,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
            )
        with record_function("init_cifar_train_loader"):
            cifar_train_loader = Cifar100Loader(cfg, train=True, device=DEVICE)
        with record_function("init_cifar_test_loader"):
            cifar_test_loader = Cifar100Loader(cfg, train=False, device=DEVICE)
        
        with record_function("torch_train_iter"):
            torch_train_iter = iter(torch_train_loader)
        with record_function("torch_test_iter"):
            torch_test_iter = iter(torch_test_loader)
        with record_function("cifar_train_iter"):
            cifar_train_iter = iter(cifar_train_loader)
        with record_function("cifar_test_iter"):
            cifar_test_iter = iter(cifar_test_loader)
        
        with record_function("torch_train_next"):
            for _ in range(10):
                img, label = next(torch_train_iter)
                img, label = img.to(DEVICE), label.to(DEVICE)
        with record_function("torch_test_next"):
            for _ in range(10):
                img, label = next(torch_test_iter)
                img, label = img.to(DEVICE), label.to(DEVICE)
        with record_function("cifar_train_next"):
            for _ in range(10):
                img, label = next(cifar_train_iter)
        with record_function("cifar_test_next"):
            for _ in range(10):
                img, label = next(cifar_test_iter)

    prof.export_chrome_trace("loaders.json")
    print(prof.key_averages().table(sort_by="cuda_time_total"))
