from dataclasses import dataclass, field

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from jaxtyping import Float

from cifar10.core.config import ExperimentCfg, SharedCfg, TorchLoaderCfg, TrainerCfg
from cifar10.core.torchloader import TorchLoader
from cifar10.core.trainer import Trainer, finish_logging, setup_logging

# %% [markdown]
"""
A script which naively implements non-bottleneck ResNets based on [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) for CIFAR-10.

## Residual Blocks (Sec 3.2)

- They use $\mathbf y = \mathcal F(\mathbf x, \{W_i\}) + \mathbf x$. The dimensions of $\mathbf x$ and $\mathcal F$ must be equal. If this is not the case, we can perform a linear projection $W_s$ by the shortcut connections to match the dimension.
- A non-linearity (ReLU) is applied after the addition.
- $\mathcal F$ is flexible, and generally has two or three layers.

## Network Architecture (Sec 3.3)

- Convolutions use 3x3 filters with `padding=1`.
- `c_out = c_in` when feature map size stays constant. When feature map size is halved with `stride=2`, `c_out = 2 * c_in`.
- Ends with global average pooling, a fully-connected layer, and softmax to produce logits.
- For shortcut connections where `c_in ≠ c_out`, either (a) 2x2 pool/subsample the input tensor and pad with extra zeros or (b) use a 1x1 convolution with `stride=2` for the shortcut connection.

## Non-Bottleneck Implementation for CIFAR-10 (Sec 4.2)

- A 3x3 convolution is applied to the input. Then a stack of 6n 3x3 convolution layers with feature maps of sizes $\in \{32,16,8\}$ is applied, with `stride=2` for subsampling layers. They compare $n \in \{3,5,7,9\}$.
- They use downsampling with zero-padding for the shortcut connections (option A from sec. 3.3).
- Trained for 64k iterations with a mini-batch size of 128. 0.1 learning rate, divided by 10 after 32k and 48k iterations.
- Train-time augmentation: 4 pixels padded on each side, with a 32x32 crop sampled from the image or its horizontal flip. *Per-pixel* mean subtracted.
- No test-time augmentation.
- They also explore a deeper ResNet with `n=18`. To assist convergence they use `lr=0.01` to warm up for 400 iterations. An `n=200` network also converges, but overfits and has worse test accuracy.
- Otherwise mirrors the ImageNet implementation.

## Our Implementation

We're ignoring a number of straightforward optimizations in order to faithfully implement the paper. In particular, we're using full fp32 precision everywhere.
"""

# %% 1. Model Hyperparameters

@dataclass
class ResNetCfg:
    n_classes: int = 10
    input_conv_filters: int = 16
    "c_out for the convolution applied to the input."
    group_c_ins: list[int] = field(default_factory=lambda: [16, 16, 32])
    "c_in for the first convolution in each group."
    group_downsample: list[bool] = field(default_factory=lambda: [False, True, True])
    "Whether the first block in each group downsamples and doubles channels."
    group_n_blocks: list[int] = field(default_factory=lambda: [9, 9, 9])
    "Number of blocks in each group."

    def __post_init__(self):
        # block layers + 1 input layer + 1 output layer.
        self.layers = 2 * sum(self.group_n_blocks) + 2


# %% 2. Model

class BasicBlock(nn.Module):
    """
    A ResNet building block as described in sec. 3.2 (two Conv-BN-ReLU stacks) with a parameter-free shortcut (Option A) for downsampling blocks.
    """

    def __init__(self, c_in: int, downsample: bool) -> None:
        super().__init__()

        self.c_out = c_out = 2 * c_in if downsample else c_in
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)

        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)

        if downsample:
            # Parameter-free shortcut (Option A from sec. 3.3).
            pad_chs = c_out - c_in
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=1, stride=stride),
                nn.ZeroPad3d((0, 0, 0, 0, 0, pad_chs)),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(
        self,
        x: Float[Tensor, "batch c_in h_in w_in"],
    ) -> Float[Tensor, "batch c_out h_out w_out"]:
        out = self.bn1(self.conv1(x))
        out = F.relu(out, inplace=True)
        out = self.bn2(self.conv2(out)) + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    """A non-bottleneck ResNet as described in sec. 4.2."""

    def __init__(self, cfg: ResNetCfg) -> None:
        super().__init__()

        # Input stem
        self.conv1 = nn.Conv2d(3, cfg.input_conv_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg.input_conv_filters)

        self.groups = nn.Sequential(*[
            self._make_group(c_in, downsample, n_blocks)
            for c_in, downsample, n_blocks in
            zip(cfg.group_c_ins, cfg.group_downsample, cfg.group_n_blocks)
        ])

        # Output layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        c_out = self.groups[-1][-1].c_out
        self.fc = nn.Linear(c_out, cfg.n_classes)

        self.apply(self._init_weights)

    def _make_group(self, c_in: int, downsample: bool, n_blocks: int) -> nn.Sequential:
        group = [BasicBlock(c_in, downsample)]
        for _ in range(1, n_blocks):
            group.append(BasicBlock(group[-1].c_out, False))
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
        loader=TorchLoaderCfg(
            batch_size=128,
            n_workers=12,
            cutout_size=0,
            pad_mode='constant',
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

    model_cfg = ResNetCfg()
    model = ResNet(model_cfg)

    setup_logging(cfg, model_cfg)
    trainer = Trainer(cfg.trainer, cfg.shared, train_loader, test_loader, make_optimizer, make_scheduler, model)
    trainer.train()
    finish_logging()
