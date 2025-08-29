# %%
from dataclasses import dataclass, field

from jaxtyping import Float
import torch
from torch import nn, optim, Tensor

from core.modules import GhostBatchNorm
from core.gpuloader import GPULoader
from core.trainer import Trainer, finish_logging, setup_logging
from core.config import ExperimentCfg, GPULoaderCfg, TrainerCfg

# %% [markdown]
"""
We implement the final changes from [Page's final post](https://web.archive.org/web/20231118151801/https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/).

- Page both pre-processes and augments data on the GPU. To avoid the overhead associated with launching several kernels for each image, he applies the same augmentation to randomly selected groups of images. He caps this at 200 randomly selected groups per epoch.
- Mixed-precision training adds a second to training time, so he keeps it disabled (training in fp16 alone). TODO: benchmark this.
- He adds label smoothing with $\epsilon=0.2$ to the loss function. As he adjusts the number of epochs, he adjusts the number of warmup steps as well (warmup = total steps / 5).
- He replaces the ReLU activation functions with CELU (alpha=0.075). This boosts accuracy, which allows him to reduce the number of epochs and thus training time.
- He adds [ghost batchnorms](https://arxiv.org/abs/1705.08741). Rather than computing batchnorm statistics for the full batch, separate statistics are computed for each group of 32 images.
- He increases the learning rate by 50%.
- He freezes batchnorm scales at 1, rescales the CELU $\alpha$ by a factor of 4, then scales the learning rate for batchnorm biases up by a factor of 16 (and reducing the weight decay for batchnorm biases by a factor of 16).
- He applies 3x3 patch-based PCA whitening (via a convolution with frozen weights) to the input images, followed by a learnable 1x1 convolution. TODO: try ZCA
- Afterwards, he increases the learning rate by 50% *again* and reduces cutout from 8x8 to 5x5 to compensate for the extra regularization that the high lr brings.
- He takes the exponential moving average over the weights every 5 batches with a momentum of 0.99, seemingly across the entire run. TODO: restrict to last n epochs.
- He adds horizontal flipping test-time augmentation, then removes cutout. This allows him to reduce training time to 10 epochs!
"""

# %% 1. Model Hyperparameters

@dataclass
class MyrtleCfg:
    n_classes: int = 10
    input_conv_filters: int = 64
    "c_out for the convolution applied to the input."
    group_residual: list[bool] = field(default_factory=lambda: [True, False, True])
    "Whether each group has a residual block."
    fc_scale: float = 0.125
    "We scale the activations by this amount before the softmax."
    celu_alpha: float = 0.075
    n_splits: int = 16
    "The number of ghost batches to use for GhostBatchNorm. Should be batch_size // ghost_batch_size."

# %% 2. Model

class MyrtleBlock(nn.Module):
    """
    A block in David Page's final ResNet architecture.

    Args:
        c_in: The number of input channels.
        c_out: The number of output channels.
        residual: Whether to add two serial 3x3 convolution -> batchnorm -> CELU sequences with an identity shortcut.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        residual: bool = False,
        celu_alpha: float = 0.075,
        n_splits: int = 32,
    ):
        super().__init__()
        self.residual = residual

        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gbn1 = GhostBatchNorm(c_out, n_splits=n_splits)
        self.celu = nn.CELU(alpha=celu_alpha)

        if residual:
            self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
            self.gbn2 = GhostBatchNorm(c_out, n_splits=n_splits)

            self.conv3 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
            self.gbn3 = GhostBatchNorm(c_out, n_splits=n_splits)

    def forward(
        self, x: Float[Tensor, "batch c_in h_in w_in"]
    ) -> Float[Tensor, "batch c_out h_out w_out"]:
        """h_out and w_out are one half of h_in and w_in, respectively."""
        out = self.pool(self.conv1(x))
        out = self.celu(self.gbn1(out))

        if self.residual:
            res = self.celu(self.gbn2(self.conv2(out)))
            res = self.celu(self.gbn3(self.conv3(res)))
            out = out + res

        return out

class MyrtleResnet(torch.nn.Module):
    def __init__(self, cfg: MyrtleCfg):
        super().__init__()
        self.conv = nn.Conv2d(3, cfg.input_conv_filters, 3, padding=1, bias=False)
        self.gbn = GhostBatchNorm(cfg.input_conv_filters, n_splits=cfg.n_splits)
        self.celu = nn.CELU(alpha=cfg.celu_alpha)

        n_groups = len(cfg.group_residual)
        c_ins = [
            cfg.input_conv_filters * 2 ** i
            for i in range(n_groups+1)
        ]

        self.layers = nn.Sequential(*[
            MyrtleBlock(c_in, c_out, res, cfg.celu_alpha, cfg.n_splits)
            for c_in, c_out, res in zip(c_ins, c_ins[1:], cfg.group_residual)
        ])

        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(c_ins[-1], cfg.n_classes, bias=False)
        self.fc_scale = cfg.fc_scale

    def forward(self, x: Float[Tensor, "batch channel height width"]) -> Float[Tensor, "batch n_classes"]:
        out = self.celu(self.gbn(self.conv(x)))
        out = self.layers(out)
        out = self.pool(out).flatten(start_dim=1)
        out = self.fc(out) * self.fc_scale
        return out

# %% 3. Training and Evaluation

if __name__ == "__main__":
    assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU."

    batch_size = 512
    steps_per_epoch = (50_000 + batch_size - 1) // batch_size
    train_steps = steps_per_epoch * 20
    warmup_steps = train_steps // 5

    cfg = ExperimentCfg(
        trainer=TrainerCfg(
            train_steps=train_steps,
            eval_every=steps_per_epoch,
            label_smoothing=0.2,
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

    ghost_batch_size = 32
    model_cfg = MyrtleCfg(
        n_splits=batch_size // ghost_batch_size
    )
    model = MyrtleResnet(model_cfg)

    setup_logging(cfg, model_cfg)
    trainer = Trainer(cfg.trainer, cfg.shared, train_loader, test_loader, make_optimizer, make_scheduler, model)
    trainer.train()
    finish_logging()
