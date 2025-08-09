# %% [markdown]
"""
Prototyping and experimenting with our GPU-accelerated data loader. On a Runpod H100 SXM instance, we see significant speedups - especially at larger batch sizes!

At batch size 128, we already see a 1.21x overall training speedup over the PyTorch dataloader. At batch size 2048, we see a 3.76x speedup! Note that we're using a smaller ResNet18 model here.

=== Torch train (bs=128, iters=3000) ===
Time to initialize: 0.847s
Time for iter(): 0.106s
Total training time: 49.322s
Average time per step: 0.016s
Total time: 50.275s
Max memory allocated: 160 MiB
Max memory reserved: 172 MiB

=== Cifar train (bs=128, iters=3000) ===
Time to initialize: 0.152s
Time for iter(): 0.000s
Total training time: 40.872s
Average time per step: 0.014s
Total time: 41.024s
Max memory allocated: 1,277 MiB
Max memory reserved: 1,618 MiB

=== Torch train (bs=2048, iters=3000) ===
Time to initialize: 1.347s
Time for iter(): 0.225s
Total training time: 171.822s
Average time per step: 0.057s
Total time: 173.394s
Max memory allocated: 592 MiB
Max memory reserved: 676 MiB

=== Cifar train (bs=2048, iters=3000) ===
Time to initialize: 0.137s
Time for iter(): 0.000s
Total training time: 45.719s
Average time per step: 0.015s
Total time: 45.856s
Max memory allocated: 1,277 MiB
Max memory reserved: 1,602 MiB
"""
# %%
import contextlib
from datetime import datetime
import gc
import logging
import os
import sys
import time
from typing import Literal
from tabulate import tabulate
import torch
from torch import Tensor
from jaxtyping import Float
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2
from torchvision.models import resnet18
from einops import rearrange
from dataclasses import dataclass
import torch.nn.functional as F
from tqdm import trange
# %%

assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

torch.backends.cudnn.benchmark = True

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = f"{BASE_DIR}/data"
DEVICE = 'cuda'
DTYPE = torch.float16
CIFAR100_MEAN = torch.tensor((0.5078125, 0.486328125, 0.44140625), dtype=DTYPE, device=DEVICE)
CIFAR100_STD = torch.tensor((0.267578125, 0.255859375, 0.275390625), dtype=DTYPE, device=DEVICE)
CPU_MEAN, CPU_STD = CIFAR100_MEAN.cpu(), CIFAR100_STD.cpu()
TOTAL_ITERS = 3_000

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

        # Transfer as uint8 then convert on GPU. This is faster than loading pre-processed fp16 data.
        data = torch.load(cifar_path, map_location=device)
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        
        # Convert to fp16, normalize, and rearrange on GPU
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
    v2.RandomCrop(32, padding=4, padding_mode="reflect"),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(DTYPE, scale=True),
    v2.Normalize(CPU_MEAN, CPU_STD),
])

"""
test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(DTYPE, scale=True),
    v2.Normalize(CPU_MEAN, CPU_STD),
])
"""

# %%

if __name__ == "__main__":
    if not os.path.exists(f"{BASE_DIR}/logs"):
        os.makedirs(f"{BASE_DIR}/logs", exist_ok=True)
        
    log_id = datetime.now().isoformat(timespec="seconds")
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{BASE_DIR}/logs/{log_id}-loader.txt"),
        ],
    )

    logging.info(" ".join(sys.argv))
    logging.info(f"{log_id=}")
    logging.info(f"Running Python {sys.version} and PyTorch {torch.version.__version__}")
    logging.info(f"Running CUDA {torch.version.cuda} and cuDNN {torch.backends.cudnn.version()}")
    logging.info(torch.cuda.get_device_name())
    logging.info(tabulate(vars(Config()).items(), headers=["Config Field", "Value"]))

    def model_and_opt():
        model = resnet18(num_classes=100).to(
            device=DEVICE,
            dtype=DTYPE,
            memory_format=torch.channels_last,
            non_blocking=True,
        )
        opt = torch.optim.SGD(
            model.parameters(),
            lr=1e-3,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True,
            fused=True,
        )
        return model, opt
    
    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    def log_times(title, t0, t1, t2, t3):
        strs = [
            f"=== {title} ===",
            f"Time to initialize: {t1 - t0:,.3f}s",
            f"Time for iter(): {t2 - t1:,.3f}s",
            f"Total training time: {t3 - t2:,.3f}s",
            f"Average time per step: {(t3 - t2) / TOTAL_ITERS:,.3f}s",
            f"Total time: {t3 - t0:,.3f}s",
            f"Max memory allocated: {torch.cuda.max_memory_allocated() // 1024**2:,} MiB",
            f"Max memory reserved: {torch.cuda.max_memory_reserved() // 1024**2:,} MiB",
        ]
        logging.info("\n".join(strs))

    
    for batch_size in [128, 512, 768, 2048]:

        with contextlib.suppress(NameError):
            del model, opt, cifar_train_loader, cifar_train_iter, batch, labels, out, loss
        cleanup()

        model, opt = model_and_opt()
        title = f"Torch train (bs={batch_size}, iters={TOTAL_ITERS})"
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch_train_loader = DataLoader(
            CIFAR100(root=DATA_DIR, train=True, download=True, transform=train_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        torch_train_iter = iter(torch_train_loader)

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        for _ in trange(TOTAL_ITERS, desc=title):
            try:
                batch, labels = next(torch_train_iter)
            except StopIteration:
                torch_train_iter = iter(torch_train_loader)
                batch, labels = next(torch_train_iter)
                
            batch = batch.to(device=DEVICE, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device=DEVICE, non_blocking=True)

            out = model(batch)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        assert batch.is_contiguous(memory_format=torch.channels_last)
        log_times(title, t0, t1, t2, t3)
        
        del model, opt, torch_train_loader, torch_train_iter, batch, labels, out, loss
        cleanup()

        model, opt = model_and_opt()
        cfg = Config(batch_size=batch_size)
        title = f"Cifar train (bs={batch_size}, iters={TOTAL_ITERS})"
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cifar_train_loader = Cifar100Loader(cfg, train=True, device=DEVICE)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        cifar_train_iter = iter(cifar_train_loader)

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        for _ in trange(TOTAL_ITERS, desc=title):
            batch, labels = next(cifar_train_iter)
            out = model(batch)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t3 = time.perf_counter()
        assert batch.is_contiguous(memory_format=torch.channels_last)
        log_times(title, t0, t1, t2, t3)
    
    with torch.profiler.profile(
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.profiler.record_function("iter"):
            new_iter = iter(cifar_train_loader)
        with torch.profiler.record_function("next"):
            for _ in range(10):
                batch, labels = next(new_iter)
    
    prof.export_chrome_trace(f"{BASE_DIR}/logs/loader-trace.json")