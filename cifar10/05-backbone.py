# %%
from dataclasses import dataclass, field

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from jaxtyping import Float

from core.config import ExperimentCfg
from core.gpuloader import GPULoader, GPULoaderCfg
from core.trainer import Trainer, TrainerCfg, finish_logging, setup_logging

# %% [markdown]
"""
We continue adding the changes from David Page's *How to Train Your ResNet* series to the [DAWNNet](04-dawnnet.py) implementation.

In [part 3](https://web.archive.org/web/20231207234442/https://myrtle.ai/learn/how-to-train-your-resnet-3-regularisation/), Page:
- Realizes that moving *all* weights to fp16 triggers a slow code path for batchnorms. Moving them back to fp32 fixes this.
- Adds 8x8 cutout (zeroing out random 8x8 patches) to the data augmentation pipeline.
- Increases the batch size to 768.
- Changes the learning rate schedule. The new schedule peaks roughly 25% of the way through training, then linearly decays to 0 until the end of training. With these changes, he can reduce the training duration to 30 epochs.

In [part 4](https://web.archive.org/web/20231108123408/https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/), Page completely updates the architecture!
- He starts by reducing the ResNet down to its shortest path - just the residual stream, trained for 20 epochs.
- This (of course) does not work well, bringing the accuracy down to 55.9%. He's able to bring it back up to 91.1% by:
    - Removing repeated batchnorm-ReLU pairs.
    - Replacing the 1x1 downsampling (`stride=2`) convolutions with `stride=1`, 3x3 convolutions followed by 2x2 max pooling layers.
    - Replacing the concatenated max/average pooling layer with max pooling. To compensate for the reduced input for the final linear layer, he doubles the output size of the final convolution.
    - Using unit initialization for the BatchNorm scale weights (gamma). The PyTorch 0.4 default was random uniform initialization over [0, 1].
    - Scaling the final classifier layer by 0.125.
- He then applies brute force architecture search, finding that adding residual blocks (consisting of two 3x3 convolutions -> batchnorm -> ReLU sequences with identity shortcuts) after the pooling in the first and third layers performs well. With this architecture he achieves 94.08% accuracy in 79s (24 epochs)!

In our implementation:
- We're using autocast and grad scaling.
- We continue using step-based scheduling (rather than epoch-based scheduling).
- We use mean reduction for our loss. As a result, we don't divide our learning rate by the batch size.

TODO:
- Benchmark without autocast and grad scaling.
- Use Page's weight decay.
"""

# %% 1. Model Hyperparameters

@dataclass
class BackboneCfg:
    n_classes: int = 10
    input_conv_filters: int = 64
    "c_out for the convolution applied to the input."
    group_residual: list[bool] = field(default_factory=lambda: [True, False, True])
    "Whether each group has a residual block."
    fc_scale: float = 0.125
    "We scale the activations by this amount before the softmax."

# %% 2. Model

class BackboneBlock(nn.Module):
    """
    A block in David Page's backbone resnet architecture. When residual is False, this is just a convolution -> batchnorm -> ReLU -> max pooling sequence.

    Args:
        c_in: The number of input channels.
        c_out: The number of output channels.
        residual: Whether to add two serial 3x3 convolution -> batchnorm -> ReLU sequences with an identity shortcut.
    """

    def __init__(self, c_in: int, c_out: int, residual: bool = False):
        super().__init__()
        self.residual = residual

        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if residual:
            self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(c_out)

            self.conv3 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(c_out)

    def forward(
        self, x: Float[Tensor, "batch c_in h_in w_in"]
    ) -> Float[Tensor, "batch c_out h_out w_out"]:
        """h_out and w_out are one half of h_in and w_in, respectively."""
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.pool(out)

        if self.residual:
            res = F.relu(self.bn2(self.conv2(out)), inplace=True)
            res = F.relu(self.bn3(self.conv3(res)), inplace=True)
            out = out + res

        return out

class BackboneResnet(nn.Module):

    def __init__(self, cfg: BackboneCfg):
        super().__init__()
        self.conv = nn.Conv2d(3, cfg.input_conv_filters, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cfg.input_conv_filters)

        n_groups = len(cfg.group_residual)
        c_ins = [
            cfg.input_conv_filters * 2 ** i
            for i in range(n_groups+1)
        ]

        self.layers = nn.Sequential(*[
            BackboneBlock(c_in, c_out, res) for c_in, c_out, res in
            zip(c_ins, c_ins[1:], cfg.group_residual)
        ])

        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(c_ins[-1], cfg.n_classes, bias=False)
        self.fc_scale = cfg.fc_scale

    def forward(self, x: Float[Tensor, "batch channel height width"]) -> Float[Tensor, "batch n_classes"]:
        out = F.relu(self.bn(self.conv(x)), inplace=True)
        out = self.layers(out)
        out = self.pool(out).flatten(start_dim=1)
        out = self.fc(out) * self.fc_scale
        return out

# %% 3. Training and Evaluation

if __name__ == "__main__":
    assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

    batch_size = 512
    steps_per_epoch = (50_000 + batch_size - 1) // batch_size
    warmup_steps = steps_per_epoch * 5
    train_steps = steps_per_epoch * 24

    cfg = ExperimentCfg(
        trainer=TrainerCfg(
            train_steps=train_steps,
            eval_every=steps_per_epoch,
        ),
        loader=GPULoaderCfg(
            batch_size=batch_size,
        ),
    )

    train_loader = GPULoader(True, cfg.loader, cfg.shared)
    test_loader = GPULoader(False, cfg.loader, cfg.shared)

    def make_optimizer(model: nn.Module) -> optim.Optimizer:
        return optim.SGD(
            model.parameters(),
            lr=0.4,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )

    def make_scheduler(optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
        )
        decay = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=train_steps - warmup_steps
        )

        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
        )

    model_cfg = BackboneCfg()
    model = BackboneResnet(model_cfg)

    setup_logging(cfg, model_cfg)
    trainer = Trainer(cfg.trainer, cfg.shared, train_loader, test_loader, make_optimizer, make_scheduler, model)
    trainer.train()
    finish_logging()
