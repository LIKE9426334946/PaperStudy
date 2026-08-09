"""Checks for deterministic sampling, preprocessing and SDLM updates."""

import torch

from datasets.mnist import PaperEpochSampler
from datasets.transforms import (
    DeterministicAffineDistortion,
    paper_normalize,
    prepare_mnist_tensor,
)
from optim import StochasticDiagonalLevenbergMarquardt


def test_paper_normalization_and_padding() -> None:
    image = torch.zeros(1, 28, 28)
    image[:, 10:12, 10:12] = 1.0
    result = prepare_mnist_tensor(image)
    assert result.shape == (1, 32, 32)
    assert torch.isclose(result.min(), torch.tensor(-0.1))
    assert torch.isclose(result.max(), torch.tensor(1.175))
    assert torch.equal(paper_normalize(torch.tensor([0.0])), torch.tensor([-0.1]))


def test_distortion_is_deterministic_per_virtual_sample() -> None:
    transform = DeterministicAffineDistortion(
        translate_pixels=2.0,
        scale_fraction=0.15,
        squeeze_fraction=0.15,
        shear_degrees=15.0,
        seed=1998,
    )
    image = torch.zeros(1, 28, 28)
    image[:, 8:20, 12:16] = 1.0
    first = transform(image, sample_key=123)
    second = transform(image, sample_key=123)
    assert torch.equal(first, second)
    assert first.shape == image.shape
    assert 0.0 <= first.min() <= first.max() <= 1.0


def test_epoch_sampler_covers_virtual_set_before_repeating() -> None:
    sampler = PaperEpochSampler(20, samples_per_epoch=5, seed=7)
    chunks = []
    for epoch in range(4):
        sampler.set_epoch(epoch)
        chunks.extend(list(sampler))
    assert sorted(chunks) == list(range(20))
    sampler.set_epoch(4)
    repeated = list(sampler)
    sampler.set_epoch(0)
    assert repeated == list(sampler)


def test_sdlm_update_matches_equation_21() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = StochasticDiagonalLevenbergMarquardt(
        [("weight", parameter)],
        global_lr=0.5,
        mu=0.25,
    )
    optimizer.set_curvature({"weight": torch.tensor([0.75, 1.75])})
    parameter.grad = torch.tensor([2.0, 4.0])
    optimizer.step()
    expected = torch.tensor([0.0, 1.0])
    assert torch.allclose(parameter, expected)
