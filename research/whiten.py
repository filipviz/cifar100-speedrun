import torch
from torch import Tensor, nn
from torchvision.datasets import CIFAR10
from jaxtyping import Float
from einops import einsum, rearrange
import matplotlib.pyplot as plt

@torch.inference_mode()
def _pca_weights(
    images: Float[Tensor, "b c h w"],
    size: int = 3,
    eps: float = 1e-2
) -> tuple[Float[Tensor, "d c size size"], Float[Tensor, "c"]]: # noqa: F821
    """
    Produce the PCA-whitening convolution weights based on the input images.
    Note that this is a true whitening transform.
    But if a whitening convolution is followed by batchnorm, we don't need the bias.
    """
    c = images.size(1)
    h = w = size
    d = c * h * w

    # $X \in \mathbb{R}^{d \times N}$
    patches = images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, d).T

    # $\Sigma = E[X X^T] \in \mathbb{R}^{d \times d}$
    sigma = torch.cov(patches)

    N = patches.size(1)
    mu = patches.mean(1, keepdim=True)
    X = (patches - mu) / torch.sqrt(N - 1)
    sigma_2 = X @ X.T
    torch.testing.assert_close(sigma, sigma_2)

    # $Q \Lambda Q^T = \Sigma$
    evals, evecs = torch.linalg.eigh(sigma)
    # $W_\text{pca} = \Lambda^{-1/2} Q^T$
    W = (evals + eps).rsqrt().diag() @ evecs.T

    W2 = einsum(evecs, (evals + eps).rsqrt(),
        'chw d, d -> d chw',)
    torch.testing.assert_close(W, W2)

    weight = W.reshape(d, c, h, w)
    bias = -(W @ mu.squeeze())

    return weight, bias

if __name__ == '__main__':
    images = torch.tensor(CIFAR10(root=".", train=True, download=True).data).permute(0, 3, 1, 2).float() / 255.0
    conv = nn.Conv2d(3, 27, 3, padding=0, bias=True)
    with torch.no_grad():
        weight, bias = _pca_weights(images[:5000])
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
        conv.weight.requires_grad_(False)
        conv.bias.requires_grad_(False)

    baseline_patches = images.unfold(2, 3, 1).unfold(3, 3, 1).transpose(1, 3).reshape(-1, 27).T

    whitened_images = conv(images)
    whitened_patches = rearrange(whitened_images, "b d h w -> d (b h w)")

    baseline_mu = baseline_patches.mean(1)
    whitened_mu = whitened_patches.mean(1)

    # Our whitened data should have per-channel means equal to zero.
    print(f"{baseline_mu.mean()=}, {baseline_mu.max()=}")
    print(f"{whitened_mu.mean()=}, {whitened_mu.max()=}")

    # Our whitened data's covariance matrix should have off-diagonal values of zero.
    baseline_cov = torch.cov(baseline_patches)
    whitened_cov = torch.cov(whitened_patches)
    baseline_cov_off_diagonal = (baseline_cov - baseline_cov.diag().diag())
    whitened_cov_off_diagonal = (whitened_cov - whitened_cov.diag().diag())
    print(f"{baseline_cov_off_diagonal.sum()=}, {whitened_cov_off_diagonal.sum()=}")

    # Our whitened data's principal components should align with the basis vectors.
    baseline_evals, baseline_evecs = torch.linalg.eigh(baseline_cov)
    whitened_evals, whitened_evecs = torch.linalg.eigh(whitened_cov)

    fig, ax = plt.subplots(4, 2, figsize=(12, 24))
    im = ax[0, 0].imshow(baseline_cov, interpolation='none')
    ax[0, 0].set_title(f"Baseline cov max={baseline_cov.max().item():.4f}")
    fig.colorbar(im, ax=ax[0, 0])
    im = ax[0, 1].imshow(whitened_cov, interpolation='none')
    ax[0, 1].set_title(f"Whitened cov max={whitened_cov.max().item():.4f}")
    fig.colorbar(im, ax=ax[0, 1])
    im = ax[1, 0].imshow(baseline_cov_off_diagonal, interpolation='none')
    ax[1, 0].set_title(f"Baseline cov off max={baseline_cov_off_diagonal.max().item():.4f}")
    fig.colorbar(im, ax=ax[1, 0])
    im = ax[1, 1].imshow(whitened_cov_off_diagonal, interpolation='none')
    ax[1, 1].set_title(f"Whitened cov off max={whitened_cov_off_diagonal.max().item():.4f}")
    fig.colorbar(im, ax=ax[1, 1])
    im = ax[2, 0].imshow(baseline_evecs, interpolation='none')
    ax[2, 0].set_title(f"Baseline eigenvectors max={baseline_evecs.max().item():.4f}")
    fig.colorbar(im, ax=ax[2, 0])
    im = ax[2, 1].imshow(whitened_evecs, interpolation='none')
    ax[2, 1].set_title(f"Whitened eigenvectors max={whitened_evecs.max().item():.4f}")
    ax[3, 0].stem(baseline_evals)
    ax[3, 0].set_title("Baseline eigenvalues")
    ax[3, 1].stem(whitened_evals)
    ax[3, 1].set_title("Whitened eigenvalues")
    fig.colorbar(im, ax=ax[2, 1])
    fig.savefig("cov.png")
