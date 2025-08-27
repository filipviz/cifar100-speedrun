# %%
from dataclasses import dataclass, field

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from jaxtyping import Float

from core.config import ExperimentCfg, SharedCfg, TorchLoaderCfg, TrainerCfg
from core.torchloader import TorchLoader
from core.trainer import Trainer, finish_logging, setup_logging

# %% [markdown]
"""
A script which naively applies the bottleneck ResNet architecture from [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) to CIFAR-10. In the paper, this architecture is used for ImageNet.

## Bottleneck Implementation for ImageNet (Sec 4.1)

- Trained for 600,000 steps with a batch size of 256.
- They use [He initialization](https://arxiv.org/abs/1502.01852).
- Optimized using SGD with momentum=0.9 and weight_decay=1e-4. The learning rate starts at 0.1 and is divided by 10 when the training error plateaus.
- See table 1 for architectural details at different scales.
- Section 4.1 introduces the bottleneck architecture. Rather than using two consecutive 3x3 convolutions per block with a fixed number of channels, they reduce the number of channels (generally by a factor of 4) with a 1x1 convolution, apply a 3x3 convolution, then scale back up with another 1x1 convolution. These blocks are combined with parameter-free residual connections for the 50/101/152-layer ResNets.
    - Most blocks use identity residual connections, but downscaling blocks apply a 1x1 convolution to increase the number of channels (option B). If they used these projections in each block, the model size and time complexity would be doubled!
    - The caption below table 1 explains that downsampling is performed by the first 1x1 convolution with stride=2.
- Train-time augmentation: they resize with shorter side $\in [256, 480]$ for scale augmentation then take a 224x224 crop. They also apply random horizontal flipping, subtract the *per-pixel* mean, and apply PCA color augmentation (as used in [AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)).
    - PCA color augmentation is described in sec. 4.1 of the AlexNet paper. They perform PCA on the RGB channels of the training set, then add multiples of the principal components with magnitudes proportional to the corresponding eigenvalues times a random variable $\sim N(0, 0.1)$ to each sample.
- Test-time augmentation: they apply fully-convolutional form w/ 10-crop testing, with scores averaged across different scales (shorter side $\in \{224,256,384,480,640\}$).

Otherwise, the implementation is the same as the [non-bottleneck implementation](01-resnet.py).

## Our Implementation

As with our non-bottleneck implementation, we're ignoring many straightforward optimizations in order to faithfully implement the paper.

1. Since CIFAR-10 has the same 32x32 format as CIFAR-10, we can't apply the 7x7 convolution  and 3x3 max pooling they apply to the input. We'll use the size-preserving convolution they apply in the non-bottleneck implementation.
2. We're not applying the ImageNet test-time augmentation, nor are we applying scale augmentation during training.
3. We're using the same training and learning rate schedule as the non-bottleneck implementation. We train for 64k steps, starting with a learning rate of 0.1 and dividing by 10 after 32k and 48k steps.
"""

# %% 1. Model Hyperparameters

@dataclass
class BottleneckCfg:
    n_classes: int = 10
    bottleneck_factor: int = 4
    input_conv_filters: int = 16
    "c_out for the convolution applied to the input."
    group_chs: list[int] = field(default_factory=lambda: [64, 128, 256])
    "The number of output channels for each group."
    group_strides: list[int] = field(default_factory=lambda: [1, 2, 2])
    "Stride of the first block in each group."
    group_n_blocks: list[int] = field(default_factory=lambda: [9, 9, 9])
    "Number of blocks in each group."


# %% 2. Model

class BottleneckBlock(nn.Module):
    """
    Bottleneck block: 1x1 (reduce) -> 3x3 -> 1x1 (expand) with BN/ReLU after
    each conv, and a projection shortcut on downsampling.
    """

    def __init__(self, c_in: int, c_out: int, c_bottleneck: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(c_in, c_bottleneck, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(c_bottleneck)

        self.conv2 = nn.Conv2d(c_bottleneck, c_bottleneck, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_bottleneck)

        self.conv3 = nn.Conv2d(c_bottleneck, c_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c_out)

        downsample = c_out != c_in or stride != 1
        self.shortcut = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(c_out),
        ) if downsample else nn.Identity()

    def forward(
        self,
        x: Float[Tensor, "batch c_in h_in w_in"],
    ) -> Float[Tensor, "batch c_out h_out w_out"]:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out)) + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class BottleneckResNet(nn.Module):
    def __init__(self, cfg: BottleneckCfg) -> None:
        super().__init__()
        self.bottleneck_factor = cfg.bottleneck_factor

        # Input stem (size-preserving conv for CIFAR-10)
        self.conv1 = nn.Conv2d(3, cfg.input_conv_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg.input_conv_filters)

        c_ins = [cfg.input_conv_filters, *cfg.group_chs]
        self.groups = nn.Sequential(*[
            self._make_group(c_in, c_out, stride, n_blocks)
            for c_in, c_out, stride, n_blocks in
            zip(c_ins, cfg.group_chs, cfg.group_strides, cfg.group_n_blocks)
        ])

        # Output layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cfg.group_chs[-1], cfg.n_classes)

        self.apply(self._init_weights)

    def _make_group(self, c_in: int, c_out: int, stride: int, n_blocks: int) -> nn.Sequential:
        c_bottleneck, mod = divmod(c_out, self.bottleneck_factor)
        assert mod == 0, "c_out must be divisible by bottleneck_factor"

        group = [BottleneckBlock(c_in, c_out, c_bottleneck, stride)]
        for _ in range(1, n_blocks):
            group.append(BottleneckBlock(c_out, c_out, c_bottleneck, 1))
        return nn.Sequential(*group)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, "batch n_classes"]:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.groups(out)
        out = self.pool(out).flatten(start_dim=1)
        return self.fc(out)


# %% 3. Training and Evaluation

if __name__ == "__main__":
    assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

    cfg = ExperimentCfg(
        shared=SharedCfg(
            dtype='fp32',
        ),
        trainer=TrainerCfg(
            train_steps=64_000,
            eval_every=4_000,
        ),
        allow_tf32=False,
        float32_matmul_precision='highest',
    )

    train_loader = TorchLoader(True, cfg.loader, cfg.shared)
    test_loader = TorchLoader(False, cfg.loader, cfg.shared)

    def make_optimizer(model: nn.Module) -> optim.Optimizer:
        return optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4,
        )

    def make_scheduler(optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[32_000, 48_000], gamma=0.1
        )

    model_cfg = BottleneckCfg()
    model = BottleneckResNet(model_cfg)

    setup_logging(cfg, model_cfg)
    trainer = Trainer(cfg.trainer, cfg.shared, train_loader, test_loader, make_optimizer, make_scheduler, model)
    trainer.train()
    finish_logging()
