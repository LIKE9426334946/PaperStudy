"""Checks for the paper's RBF prototypes and loss equations."""

import torch

from models.rbf import (
    EuclideanRBF,
    MAPDiscriminativeLoss,
    RBFMSELoss,
    build_digit_codes,
)


def test_digit_codes_are_fixed_84_dimensional_binary_vectors() -> None:
    codes = build_digit_codes()
    assert codes.shape == (10, 84)
    assert set(codes.unique().tolist()) == {-1.0, 1.0}


def test_matching_rbf_center_has_zero_penalty() -> None:
    layer = EuclideanRBF()
    penalties = layer(build_digit_codes()[[3]])
    assert penalties.shape == (1, 10)
    assert penalties.argmin(dim=1).item() == 3
    assert penalties[0, 3].item() == 0.0


def test_rbf_mse_selects_only_target_penalty() -> None:
    penalties = torch.tensor([[4.0, 2.0, 9.0], [3.0, 8.0, 1.0]])
    targets = torch.tensor([1, 2])
    assert RBFMSELoss()(penalties, targets).item() == 1.5


def test_map_loss_is_nonnegative_and_differentiable() -> None:
    penalties = torch.tensor(
        [[1.0, 3.0, 4.0], [5.0, 2.0, 7.0]],
        requires_grad=True,
    )
    loss = MAPDiscriminativeLoss(rubbish_class_j=10.0)(
        penalties,
        torch.tensor([0, 1]),
    )
    loss.backward()
    assert loss.item() >= 0.0
    assert penalties.grad is not None
    assert torch.isfinite(penalties.grad).all()
