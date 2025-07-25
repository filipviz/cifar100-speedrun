Speedrunning the [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) benchmark (in progress).

## Motivation

In this context, "speedrunning" means training a neural network to a certain accuracy on a benchmark as rapidly as possible within fixed hardware constraints.

Keller Jordan's motivations for [speedrunning CIFAR-10](https://github.com/KellerJordan/cifar10-airbench) were:

1. CIFAR-10 facilitates [thousands of research projects](https://paperswithcode.com/dataset/cifar-10) per year. Fast, stable training baselines serve to accelerate this research.
2. Perhaps more importantly, such baselines act as a [telescope](https://twitter.com/kellerjordan0/status/1786330520366010646) to find new phenomena within neural network training. Among other techniques, Jordan's CIFAR-10 research begat the [Muon optimizer](https://kellerjordan.github.io/posts/muon/), which has since been applied [at scale](https://github.com/MoonshotAI/Kimi-K2) with apparent success.

Applying this approach to new benchmarks with modern hardware could be similarly fruitful. CIFAR-100 seems like a reasonable next step.

## Usage

I recommend using [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/filipviz/cifar100-speedrun
cd cifar100-speedrun
uv sync
uv run scripts/XX-some_script.py
```

## Results

> Train a neural network to 90% top-5 accuracy on CIFAR-100 using an NVIDIA H100.

| Implementation | Source | Top-5/1 accuracy | Time | PFLOPs | Steps/Epochs |
| -------------- | ------ | ---------------- | ---- | ------ | ------------ |

## Baselines

- [ ] [ResNet20, He et al 2015](https://arxiv.org/abs/1512.03385)
- [ ] [`davidcpage/cifar10-fast`](https://github.com/davidcpage/cifar10-fast)
- [ ] [`99991/cifar10-fast-simple`](https://github.com/99991/cifar10-fast-simple)
- [ ] [`tysam-code/hlb-CIFAR10`](https://github.com/tysam-code/hlb-CIFAR10)
- [ ] [Jordan, 2024](https://arxiv.org/abs/2404.00498)
- [ ] [`KellerJordan/cifar10-airbench`](https://github.com/KellerJordan/cifar10-airbench)
