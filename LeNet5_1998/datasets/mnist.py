"""MNIST dataset, virtual distortion set and fixed paper-order sampler."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import MNIST
from torchvision.transforms.functional import pil_to_tensor

from .transforms import DeterministicAffineDistortion, prepare_mnist_tensor


class PaperMNIST(Dataset[tuple[torch.Tensor, int]]):
    """MNIST with paper normalization and optional virtual distortions."""

    def __init__(
        self,
        root: str | Path,
        train: bool,
        download: bool,
        augmentation: dict[str, Any] | None = None,
        seed: int = 1998,
    ) -> None:
        self.base = MNIST(root=str(root), train=train, download=download)
        self.train = train
        augmentation = augmentation or {}
        self.augmentation_enabled = bool(
            train and augmentation.get("enabled", False)
        )
        self.variants_per_image = (
            int(augmentation.get("variants_per_image", 1))
            if self.augmentation_enabled
            else 1
        )
        if self.variants_per_image < 1:
            raise ValueError("variants_per_image must be at least one.")
        self.distortion = DeterministicAffineDistortion(
            translate_pixels=float(augmentation.get("translate_pixels", 0.0)),
            scale_fraction=float(augmentation.get("scale_fraction", 0.0)),
            squeeze_fraction=float(augmentation.get("squeeze_fraction", 0.0)),
            shear_degrees=float(augmentation.get("shear_degrees", 0.0)),
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.base) * self.variants_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        base_index = index % len(self.base)
        variant = index // len(self.base)
        image, target = self.base[base_index]
        tensor = pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)
        if self.augmentation_enabled and variant > 0:
            tensor = self.distortion(tensor, sample_key=index)
        return prepare_mnist_tensor(tensor), int(target)


class PaperEpochSampler(Sampler[int]):
    """Return one fixed shuffled chunk of a virtual dataset per epoch.

    With 600,000 virtual samples and 60,000 samples per epoch, epochs 0-9
    cover every virtual sample exactly once; epochs 10-19 repeat that order.
    This mirrors the paper's 20 x 60,000 presentations over a 600,000-item
    augmented set.
    """

    def __init__(
        self,
        dataset_size: int,
        samples_per_epoch: int,
        seed: int,
    ) -> None:
        if samples_per_epoch <= 0 or samples_per_epoch > dataset_size:
            raise ValueError("samples_per_epoch must be in [1, dataset_size].")
        if dataset_size % samples_per_epoch != 0:
            raise ValueError(
                "dataset_size must be divisible by samples_per_epoch."
            )
        self.dataset_size = dataset_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        generator = torch.Generator().manual_seed(seed)
        self._permutation = torch.randperm(
            dataset_size,
            generator=generator,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        chunks = self.dataset_size // self.samples_per_epoch
        chunk_index = self.epoch % chunks
        start = chunk_index * self.samples_per_epoch
        end = start + self.samples_per_epoch
        return iter(self._permutation[start:end].tolist())

    def __len__(self) -> int:
        return self.samples_per_epoch


def _loader_kwargs(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    num_workers = int(config.get("num_workers", 0))
    return {
        "num_workers": num_workers,
        "pin_memory": bool(config.get("pin_memory", False))
        and device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }


def build_dataloaders(
    data_config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[
    DataLoader[tuple[torch.Tensor, int]],
    DataLoader[tuple[torch.Tensor, int]],
    PaperEpochSampler,
]:
    """Build paper-ordered training and ordinary MNIST test loaders."""

    train_dataset = PaperMNIST(
        root=data_config["root"],
        train=True,
        download=bool(data_config.get("download", True)),
        augmentation=data_config.get("augmentation"),
        seed=seed,
    )
    test_dataset = PaperMNIST(
        root=data_config["root"],
        train=False,
        download=bool(data_config.get("download", True)),
        seed=seed,
    )
    sampler = PaperEpochSampler(
        dataset_size=len(train_dataset),
        samples_per_epoch=int(data_config["samples_per_epoch"]),
        seed=seed,
    )
    kwargs = _loader_kwargs(data_config, device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_config["batch_size"]),
        sampler=sampler,
        **kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(128, int(data_config["batch_size"])),
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader, sampler


def build_test_loader(
    data_config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> DataLoader[tuple[torch.Tensor, int]]:
    """Build only the test loader for checkpoint evaluation."""

    dataset = PaperMNIST(
        root=data_config["root"],
        train=False,
        download=bool(data_config.get("download", True)),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=max(128, int(data_config["batch_size"])),
        shuffle=False,
        **_loader_kwargs(data_config, device),
    )
