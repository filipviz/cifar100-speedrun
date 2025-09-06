import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from collections.abc import Iterable, Callable

from tabulate import tabulate
import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor
from tqdm import tqdm, trange
import wandb

from .config import ExperimentCfg, SharedCfg, TrainerCfg


class Trainer:
    def __init__(
        self,
        cfg: TrainerCfg,
        shared_cfg: SharedCfg,
        train_loader: Iterable[tuple[Tensor, Tensor]],
        test_loader: Iterable[tuple[Tensor, Tensor]],
        make_optimizer: Callable[[nn.Module], optim.Optimizer],
        make_scheduler: Callable[[optim.Optimizer], optim.lr_scheduler.LRScheduler],
        model: nn.Module,
    ):
        logging_is_configured = logging.getLogger().hasHandlers()
        assert logging_is_configured, "Logging must be configured before a Trainer is instantiated. Call setup_logging."

        self.cfg, self.step = cfg, 1
        self.run_id, self.device = shared_cfg.run_id, shared_cfg.device
        self.dtype = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            'fp32': torch.float32,
        }[shared_cfg.dtype]

        self.autocast_enabled = self.dtype != torch.float32
        self.train_loader, self.test_loader = train_loader, test_loader
        self.model = model.to(
            device=self.device,
            memory_format=torch.channels_last
        )

        self.opt = make_optimizer(self.model)
        self.scheduler = make_scheduler(self.opt)

        if shared_cfg.compile_enabled:
            self.model.compile(mode=shared_cfg.compile_mode, fullgraph=True)
            self.opt.step = torch.compile(self.opt.step, mode=shared_cfg.compile_mode)
            self.scheduler.step = torch.compile(self.scheduler.step, mode=shared_cfg.compile_mode)

        need_checkpoint_dir = self.cfg.checkpoint_every > 0 or self.cfg.model_warmup_steps > 0
        if need_checkpoint_dir:
            self.checkpoint_dir = os.path.join(shared_cfg.base_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        if cfg.model_warmup_steps > 0:
            logging.info(f"Warming up for {cfg.model_warmup_steps} steps")
            checkpoint_path = self.save_checkpoint(step=self.step)
            self.train(warmup=True)
            self.load_checkpoint(checkpoint_path)
            os.remove(checkpoint_path)
    
    def train(self, warmup: bool = False):
        logging.info(HEADER_FMT.format(*LOGGING_COLUMNS))
        logging.info(HEADER_FMT.format(*['---' for _ in LOGGING_COLUMNS]))
        if warmup:
            steps, desc = self.cfg.model_warmup_steps + 1, "warmup"
        else:
            steps, desc = self.cfg.train_steps + 1, "training"

        self.model.train()
        loader_iter = iter(self.train_loader)
        pbar = trange(self.step, steps, desc=desc, initial=self.step, total=steps)
        training_time = 0.0

        # Start the clock.
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        scaler = torch.GradScaler(enabled=(self.dtype == torch.float16))
        for step in pbar:
            # ---- Training ---- #

            # 1. Load our batch and labels.
            batch, labels = next(loader_iter)

            # 2. Forward pass.
            with torch.autocast(self.device, dtype=self.dtype, enabled=self.autocast_enabled):
                pred = self.model(batch)
                loss = F.cross_entropy(pred, labels, label_smoothing=self.cfg.label_smoothing)

            # 3. Backward pass.
            self.opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(self.opt)
            scaler.update()

            # 4. Update our learning rate.
            self.scheduler.step()

            if self.cfg.use_wandb and not warmup:
                wandb.log({
                    'train_loss': loss.item(),
                    'lr': self.scheduler.get_last_lr()[0],
                }, step)

            # ---- Evaluation ---- #
            last_step = step + 1 == steps
            eval_step = self.cfg.eval_every > 0 and step % self.cfg.eval_every == 0
            if last_step or eval_step:
                torch.cuda.synchronize()
                interval_time = time.perf_counter() - t0
                training_time += interval_time

                checkpoint_enabled = self.cfg.checkpoint_every > 0 and not warmup
                checkpoint_step = checkpoint_enabled and (last_step or step % self.cfg.checkpoint_every == 0)
                if checkpoint_step:
                    self.save_checkpoint(step)

                with torch.no_grad():
                    # Clone to avoid accessing tensor output of CUDAGraphs.
                    pred = pred.clone()

                    self.model.eval()
                    test_metrics = self.evaluate()
                    self.model.train()

                    # Our trainining metrics are only estimates (computed on a single batch).
                    metrics = {
                        "step": step,
                        "time": training_time,
                        "interval": interval_time,
                        "lr": self.scheduler.get_last_lr()[0],
                        "train_loss": loss.item(),
                        "train_acc1": (pred.argmax(dim=1) == labels).float().mean().item(),
                        "train_acc5": (pred.topk(5)[1] == labels.view(-1, 1)).any(dim=1).float().mean().item(),
                        **test_metrics,
                    }

                logging.info(ROW_FMT.format(*[metrics[col] for col in LOGGING_COLUMNS]))
                pbar.set_postfix(train_loss=metrics['train_loss'], test_loss=metrics['test_loss'])

                if self.cfg.use_wandb and not warmup:
                    del metrics['step']
                    wandb.log(metrics, step)

                # Start the clock again.
                torch.cuda.synchronize()
                t0 = time.perf_counter()

        torch.cuda.synchronize()
        training_time += time.perf_counter() - t0
        logging.info(f"Total {desc} time: {training_time:,.2f}s")

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        assert not self.model.training, "Model must be in eval mode"
        items = self.test_loader.n_images
        cum_loss = torch.tensor(0.0, device=self.device)
        n_correct_top1 = torch.tensor(0.0, device=self.device)
        n_correct_top5 = torch.tensor(0.0, device=self.device)

        pbar = tqdm(self.test_loader, desc="evaluating", position=1, leave=False)
        for batch, labels in pbar:
            with torch.autocast(self.device, dtype=self.dtype, enabled=self.autocast_enabled):
                pred = self.model(batch)
                loss = F.cross_entropy(pred, labels, reduction="sum")

            cum_loss += loss
            n_correct_top1 += (pred.argmax(dim=1) == labels).sum()
            n_correct_top5 += (pred.topk(5)[1] == labels.view(-1, 1)).sum()

        return {
            "test_loss": cum_loss.item() / items,
            "test_acc1": n_correct_top1.item() / items,
            "test_acc5": n_correct_top5.item() / items,
        }

    def save_checkpoint(self, step: int) -> str:
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.run_id}-step-{step}.pt")
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': step,
        }, checkpoint_path)
        logging.info(f"Wrote step {step} checkpoint to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        data = torch.load(checkpoint_path)
        self.model.load_state_dict(data['model'])
        self.opt.load_state_dict(data['optimizer'])
        self.scheduler.load_state_dict(data['scheduler'])
        self.step = data['step']
        logging.info(f"Loaded step {self.step} checkpoint from {checkpoint_path}")


# --- Logging Utilities --- #
# Call setup_logging before initializing a Trainer, and finish_logging after training is complete.

LOGGING_COLUMNS = ['step', 'time', 'interval', 'lr', 'train_loss', 'train_acc1',
                   'train_acc5', 'test_loss', 'test_acc1', 'test_acc5']
HEADER_FMT = "|{:^6s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|"
ROW_FMT = "|{:>6d}|{:>10,.3f}|{:>10,.3f}|{:>10,.3e}|{:>10,.3f}|{:>10.3%}|{:>10.3%}|{:>10,.3f}|{:>10.3%}|{:>10.3%}|"

def setup_logging(cfg: ExperimentCfg, model_cfg: dataclass):
    """Call before initializing a Trainer."""

    log_dir = os.path.join(cfg.shared.base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{cfg.shared.run_id}.log")
    logging.basicConfig(filename=log_file, format="%(message)s", level=logging.INFO)

    logging.info(" ".join(sys.argv))
    logging.info(f"Run ID: {cfg.shared.run_id}")
    logging.info(f"Running Python {sys.version} and PyTorch {torch.__version__}")
    logging.info(f"Running CUDA {torch.version.cuda} and cuDNN {torch.backends.cudnn.version()}")
    logging.info(torch.cuda.get_device_name())
    logging.info(tabulate(vars(cfg).items(), headers=["Config Field", "Value"]))
    logging.info(f"Config class: {model_cfg.__class__.__name__}")
    logging.info(tabulate(vars(model_cfg).items(), headers=["Config Field", "Value"]))

def finish_logging():
    """Call after training is complete."""
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        logging.info(smi.stdout)
        logging.info(smi.stderr)
    except subprocess.CalledProcessError as e:
        logging.info(f"Error running nvidia-smi: {e}")

    logging.info(f"Max memory allocated: {torch.cuda.max_memory_allocated() // 1024**2:,} MiB")
    logging.info(f"Max memory reserved: {torch.cuda.max_memory_reserved() // 1024**2:,} MiB")

    # Write entry point source to our logs.
    with open(sys.argv[0]) as f:
        logging.info(f.read())
