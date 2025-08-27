# %%
from dataclasses import dataclass, field

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from jaxtyping import Float

from cifar10.core.config import ExperimentCfg, TorchLoaderCfg, TrainerCfg
from cifar10.core.torchloader import TorchLoader
from cifar10.core.trainer import Trainer, finish_logging, setup_logging

# %% [markdown]
"""
In this script, we implement the pre-activation ResNet architecture as described in [*Identity Mappings in Deep Residual Networks*](https://arxiv.org/abs/1603.05027). Rather than convolution -> batchnorm -> ReLU, the paper uses batchnorm -> ReLU -> convolution.

The key details are in the appendix:
- They largely follow the [non-bottleneck architecture](./01-resnet.py) from [He et al 2015](https://arxiv.org/abs/1512.03385).
- They only use translation and horizontal flipping augmentation for training.
- They warm up with a learning rate of 0.01. After 400 steps, they increase the learning rate to 0.1. They divide the learning rate by 10 after 32k steps and divide it again after 48k steps.
- They use a batch size of 128, a weight decay of 1e-4, and set their SGD momentum to 0.9.
- They (perhaps redundantly) apply BN + ReLU in the input convolution stem (before splitting into two paths). They apply an extra BN + ReLU after the elementwise addition in the last block (before the pooling + fully-connected layer).

Readers may also be interested in [Kaiming He's implementation](https://github.com/KaimingHe/resnet-1k-layers).
"""

# %% 1. Model Hyperparameters

@dataclass
class PreActConfig:
    n_classes: int = 10
    input_conv_filters: int = 16
    "c_out for the convolution applied to the input."
    group_c_ins: list[int] = field(default_factory=lambda: [16, 16, 32])
    "c_in for the first convolution in each group."
    group_downsample: list[bool] = field(default_factory=lambda: [False, True, True])
    "Whether the first block in each group downsamples the feature map size and doubles the number of channels."
    group_n_blocks: list[int] = field(default_factory=lambda: [9, 9, 9])
    "Number of blocks in each group."


# %% 2. Model

class PreActBlock(nn.Module):
    """
    A pre-activation ResNet building block as described in He et al 2016 sec. 4.1.

    Args:
        c_in: Number of input channels.
        downsample: If true, use a stride of 2 (halves the feature map size)
        and double the number of output channels.
    """

    def __init__(
        self,
        c_in: int,
        downsample: bool,
    ) -> None:
        super().__init__()

        self.c_out = c_out = 2 * c_in if downsample else c_in
        stride = 2 if downsample else 1

        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)

        if downsample:
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
        """
        If this is a downsampling block, c_out = 2*c_in and h_out/w_out are
        half of h_in/w_in respectively. Otherwise, they all match.
        """
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return out

class PreActResNet(nn.Module):
    """A pre-activation ResNet as described in He et al 2016."""

    def __init__(self, cfg: PreActConfig) -> None:
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
        c_out = self.groups[-1][-1].c_out
        self.bn2 = nn.BatchNorm2d(c_out)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_out, cfg.n_classes)

        self.apply(self._init_weights)

    def _make_group(
        self,
        c_in: int,
        downsample: bool,
        n_blocks: int,
    ) -> nn.Sequential:
        group = [PreActBlock(c_in, downsample)]
        for _ in range(1, n_blocks):
            group.append(PreActBlock(group[-1].c_out, False))

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
        out = F.relu(self.bn2(out), inplace=True)
        out = self.pool(out).flatten(start_dim=1)
        return self.fc(out)


# %% 3. Training and Evaluation

if __name__ == "__main__":
    assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

    cfg = ExperimentCfg(
        trainer=TrainerCfg(
            train_steps=64_000,
            eval_every=4_000,
        ),
        loader=TorchLoaderCfg(
            batch_size=128,
            n_workers=12,
            cutout_size=0,
        ),
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
        warmup_steps = 400
        warmup = optim.lr_scheduler.ConstantLR(
            optimizer, factor=0.1, total_iters=warmup_steps
        )
        multistep = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[32_000 - warmup_steps, 48_000 - warmup_steps],
            gamma=0.1,
        )

        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, multistep], milestones=[warmup_steps]
        )

    model_cfg = PreActConfig()
    model = PreActResNet(model_cfg)

    setup_logging(cfg, model_cfg)
    trainer = Trainer(cfg.trainer, cfg.shared, train_loader, test_loader, make_optimizer, make_scheduler, model)
    trainer.train()
    finish_logging()
