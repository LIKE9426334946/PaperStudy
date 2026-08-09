"""MNIST input pipeline matching the paper's preprocessing."""

from .mnist import (
    PaperEpochSampler,
    PaperMNIST,
    build_dataloaders,
    build_test_loader,
)
from .transforms import (
    DeterministicAffineDistortion,
    paper_normalize,
    prepare_mnist_tensor,
)

__all__ = [
    "PaperEpochSampler",
    "PaperMNIST",
    "build_dataloaders",
    "build_test_loader",
    "DeterministicAffineDistortion",
    "paper_normalize",
    "prepare_mnist_tensor",
]
