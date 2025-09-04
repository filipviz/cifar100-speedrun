# %%
from dataclasses import dataclass, field

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
import numpy as np
from jaxtyping import Float

from core.config import ExperimentCfg, GPULoaderCfg, TrainerCfg
from core.gpuloader import GPULoader
from core.trainer import Trainer, finish_logging, setup_logging

# %% [markdown]
"""
In this script, we implement the DAWNNet ResNet with the changes described in [*How to Train Your ResNet 1: Baseline*](https://web.archive.org/web/20221206112356/https://myrtle.ai/how-to-train-your-resnet-1-baseline/) and [*How to Train Your ResNet 2: Mini-batches*](https://web.archive.org/web/20231207232347/https://myrtle.ai/learn/how-to-train-your-resnet-2-mini-batches/).

It's based on Ben Johnson's DAWNBench submission, which:
- Is an 18-layer pre-activation ResNet.
- Uses 64 -> 128 -> 256 -> 256 channels. Much shallower and wider than the original ResNet.
- Uses 1x1 convolutions for downsampling shortcuts.
- Uses four groups with 2 blocks each. Groups 2 and 3 are downsampling groups, leaving an 8x8 map before global mean pooling at the end.
- Uses typical mean/std normalization rather than per-pixel mean alone (as in He et al 2015).
- Uses a somewhat odd learning rate schedule with a long linear warmup, long linear decay, and a few small jumps.
- Uses half-precision (fp16) training.
- Pads with mode='reflect'.
- Rather than simply average-pooling before the final linear layer, they apply both average-pooling and max-pooling, concatenate the results, and feed the result to the linear layer.

Page makes the following changes in his first article:
- Removes the batchnorm + ReLU from the input stem, since they're redundant with the batchnorm + ReLU in the first residual block.
- Removes some of the jumps in the learning rate schedule.
- Preprocesses the dataset in advance. Rather than applying padding, normalization, and random horizontal flipping each time they load a batch, he pre-applies these. This leaves only random cropping and flipping.
- He removes dataworkers to avoid the overhead associated with launching them. Keeping everything in the main thread saves time!
- He uses reduction="sum" in his loss function, and scales the weight decay by the batch size to account for this.
- He uses SGD with (PyTorch-style) Nesterov momentum.
- He combines the random number calls into bulk calls up front.

In the second article, he further:
- Slightly increases the learning rate.
- Increases the batch size to 512, under very principled motivations. I highly recommend the article!

Also see Page's [implementation notebook](https://github.com/davidcpage/cifar10-fast/blob/master/experiments.ipynb).

Our implementation departs from his in a few ways:
1. We use step-based (rather than epoch-based) scheduling.
2. Since our trainer uses mean loss reduction, we don't divide the learning rate by the batch size or multiply the weight decay by the batch size.
3. Oddly, he applies the first batchnorm + ReLU to both the residual stream and the convolution path (which hurts performance). It doesn't make a huge difference for a network this shallow, but I won't do this.
4. We take a more aggressive approach to data loading by loading the entire dataset into GPU memory (in uint8) and applying pre-processing on-device during the first call to `__iter__`. Tracing suggests we're only spending 4ms per epoch generating randperm indices and 29µs generating random numbers per batch. This comes out to about 0.24 seconds across the entire training run, so I don't think it's worth optimizing further yet.
"""

# %% 1. Model Hyperparameters

@dataclass
class DAWNCfg:
    n_classes: int = 10
    input_conv_filters: int = 64
    "c_out for the convolution applied to the input."
    group_c_ins: list[int] = field(default_factory=lambda: [64, 64, 128, 256])
    "c_in for the first convolution in each group."
    group_downsample: list[bool] = field(default_factory=lambda: [False, True, True, False])
    "Whether the first block in each group downsamples the feature map size and doubles the number of channels."
    group_n_blocks: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    "Number of blocks in each group."


# %% 2. Model

class DAWNBlock(nn.Module):
    """
    A DAWNNet ResNet building block. Does not apply batchnorm and ReLU to the residual stream.

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

        # 1x1 convolution shortcut in downsampling blocks.
        if downsample:
            self.shortcut = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False)

    def forward(
        self,
        x: Float[Tensor, "batch c_in h_in w_in"],
    ) -> Float[Tensor, "batch c_out h_out w_out"]:
        """
        If this is a downsampling block, c_out = 2*c_in and h_out/w_out are
        half of h_in/w_in respectively. Otherwise, they all match.
        """
        out = F.relu(self.bn1(x), inplace=True)
        # He et al 2016, appendix: "when preactivation is used, these projection shortcuts are also with pre-activation."
        residual = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        return out + residual

class DAWNNet(nn.Module):
    """
    A DAWNNet ResNet. Similar to a pre-activation ResNet, but shallower and wider. Uses concatenated average and max pooling before the final linear layer.
    """

    def __init__(self, cfg: DAWNCfg) -> None:
        super().__init__()

        # Input stem (only a convolution)
        self.conv1 = nn.Conv2d(3, cfg.input_conv_filters, 3, padding=1, bias=False)

        self.groups = nn.Sequential(*[
            self._make_group(c_in, downsample, n_blocks)
            for c_in, downsample, n_blocks in
            zip(cfg.group_c_ins, cfg.group_downsample, cfg.group_n_blocks)
        ])

        # Output layer
        c_out = self.groups[-1][-1].c_out
        self.bn2 = nn.BatchNorm2d(c_out)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2*c_out, cfg.n_classes, bias=False)

        self.apply(self._init_weights)

    def _make_group(
        self,
        c_in: int,
        downsample: bool,
        n_blocks: int,
    ) -> nn.Sequential:
        group = [DAWNBlock(c_in, downsample)]
        for _ in range(1, n_blocks):
            group.append(DAWNBlock(group[-1].c_out, False))
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
        out = self.conv1(x)
        out = self.groups(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = torch.cat([
            self.avg_pool(out).flatten(start_dim=1),
            self.max_pool(out).flatten(start_dim=1),
        ], dim=1)
        return self.fc(out)

# %% 3. Training and Evaluation

if __name__ == "__main__":
    assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

    batch_size = 512
    steps_per_epoch = (50_000 + batch_size - 1) // batch_size
    train_steps = steps_per_epoch * 35

    cfg = ExperimentCfg(
        trainer=TrainerCfg(
            train_steps=train_steps,
            eval_every=steps_per_epoch,
        ),
        loader=GPULoaderCfg(
            batch_size=512,
            cutout_size=0,
        ),
    )

    train_loader = GPULoader(True, cfg.loader, cfg.shared)
    test_loader = GPULoader(False, cfg.loader, cfg.shared)

    def make_optimizer(model: nn.Module) -> optim.Optimizer:
        return optim.SGD(
            model.parameters(),
            lr=torch.tensor(1.0),
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
            fused=True,
        )

    def make_scheduler(optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
        lrs = [0, 0.44, 0.005, 0]
        epoch_milestones = [0, 15, 30, 35]
        milestones = [steps_per_epoch * epoch for epoch in epoch_milestones]

        def lr_fn(step: int) -> float:
            return np.interp(step, milestones, lrs).item()

        return optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    model_cfg = DAWNCfg()
    model = DAWNNet(model_cfg)

    setup_logging(cfg, model_cfg)
    trainer = Trainer(cfg.trainer, cfg.shared, train_loader, test_loader, make_optimizer, make_scheduler, model)
    trainer.train()
    finish_logging()
