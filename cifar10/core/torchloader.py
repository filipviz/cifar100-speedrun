import os
from functools import cached_property

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
import numpy as np
from einops import rearrange

from .config import SharedCfg, TorchLoaderCfg

class TorchLoader:
    """Simple PyTorch DataLoader wrapper for CIFAR-10."""
    def __init__(self, train: bool, cfg: TorchLoaderCfg, shared_cfg: SharedCfg):

        self.cfg = cfg
        self.train, self.device = train, shared_cfg.device
        self.data_dir = os.path.join(shared_cfg.base_dir, 'data')
        self.batch_size = cfg.batch_size

        # --- Transformations --- #
        crop = train and cfg.crop_padding > 0
        flip = train and cfg.flip
        cutout = train and cfg.cutout_size > 0
        cutout_scale = cfg.cutout_size ** 2 / 32 ** 2

        ops = [
            v2.ToImage(),
            v2.RandomCrop(32, padding=cfg.crop_padding, padding_mode=cfg.pad_mode) if crop else None,
            v2.RandomHorizontalFlip() if flip else None,
            v2.RandomErasing(
                p=1.0,
                scale=(cutout_scale, cutout_scale),
                ratio=(1, 1)
            ) if cutout else None,
            v2.ToDtype(torch.float32, scale=True),
        ]

        if cfg.normalize_he:
            # Pre-compute the per-pixel mean
            _ = self._cifar10_mean
            ops.append(v2.Lambda(lambda x: x - self._cifar10_mean))

        elif cfg.normalize_torch:
            cifar10_mean = (0.4914, 0.4822, 0.4465)
            cifar10_std = (0.2470, 0.2435, 0.2616)
            ops.append(v2.Normalize(cifar10_mean, cifar10_std))

        transform = v2.Compose([op for op in ops if op is not None])

        # --- Data --- #
        self.dataset = CIFAR10(root=self.data_dir, train=train, download=True, transform=transform)
        self.n_images = len(self.dataset)
        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=train,
            num_workers=cfg.n_workers,
            persistent_workers=cfg.n_workers > 0,
            drop_last=train,
            pin_memory=True,
        )

    def __len__(self):
        # Needed for tqdm to work when self.train=False.
        assert not self.train, "__len__ is not defined for a TorchLoader with train=True"
        return len(self.loader)

    def __iter__(self):
        while True:
            for (batch, labels) in self.loader:
                batch = batch.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(self.device, non_blocking=True)
                yield batch, labels

            if not self.train:
                break

    @cached_property
    def _cifar10_mean(self):
        mean_path = f"{self.data_dir}/cifar_10_mean.npy"
        if not os.path.exists(mean_path):
            np_mean = CIFAR10(root=self.data_dir, train=True, download=True).data.mean(axis=0)
            # Convert to PyTorch-friendly format: float32 scaled to [0, 1] in (C, H, W) format.
            np_mean = rearrange(np_mean, "h w c -> c h w")
            np_mean = np_mean.astype(np.float32) / 255.0
            np.save(mean_path, np_mean)
        else:
            np_mean = np.load(mean_path)

        return torch.from_numpy(np_mean)
