import time
import statistics
from collections.abc import Callable

import torch

device = 'cuda'

if device == 'cuda':
    assert torch.cuda.is_available()
    sync = torch.cuda.synchronize
elif device == 'mps':
    assert torch.mps.is_available()
    sync = torch.mps.synchronize
else:
    assert device == 'cpu'
    sync = torch.cpu.synchronize


def _fmt_seconds(s: float) -> str:
    if s < 1e-6:
        return f"{s * 1e9:.2f} ns"
    if s < 1e-3:
        return f"{s * 1e6:.2f} us"
    if s < 1:
        return f"{s * 1e3:.2f} ms"
    return f"{s:.3f} s"


def _measure(fn: Callable[..., object], args: tuple[object, ...], warmup: int, iters: int) -> tuple[float, float]:
    """Returns (median, mean) time in seconds."""
    for _ in range(warmup):
        fn(*args)
    sync()

    samples: list[float] = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn(*args)
        sync()
        samples.append(time.perf_counter() - t0)

    return statistics.median(samples), (sum(samples) / len(samples))


Variant = tuple[str, Callable[..., object]]

def benchmark(
    variants: list[Variant],
    make_inputs: Callable[[], tuple[object, ...]],
    *,
    iters: int = 100,
    warmup: int = 10,
) -> None:
    """Time eager and compiled variants with proper device synchronization."""

    for name, fn in variants:
        # Fresh inputs for each variant
        args = make_inputs()

        # Eager
        med, mean = _measure(fn, args, warmup, iters)
        print(f"{name} [eager]: median={_fmt_seconds(med)} mean={_fmt_seconds(mean)}")

        # Compiled
        compiled = torch.compile(fn, mode="max-autotune")
        med_c, mean_c = _measure(compiled, args, warmup, iters)
        print(f"{name} [compiled]: median={_fmt_seconds(med_c)} mean={_fmt_seconds(mean_c)}")
