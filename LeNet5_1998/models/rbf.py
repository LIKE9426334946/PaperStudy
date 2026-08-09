"""Euclidean RBF output layer and losses from Eqs. (7)-(9)."""

from __future__ import annotations

import torch
from torch import nn


DIGIT_BITMAPS: tuple[tuple[str, ...], ...] = (
    (
        ".#####.", "##...##", "##...##", "##...##",
        "##...##", "##...##", "##...##", "##...##",
        "##...##", "##...##", "##...##", ".#####.",
    ),
    (
        "...##..", "..###..", ".####..", "...##..",
        "...##..", "...##..", "...##..", "...##..",
        "...##..", "...##..", "...##..", ".######",
    ),
    (
        ".#####.", "##...##", ".....##", ".....##",
        "....##.", "...##..", "..##...", ".##....",
        "##.....", "##.....", "##.....", "#######",
    ),
    (
        ".#####.", "##...##", ".....##", ".....##",
        "....##.", "..####.", ".....##", ".....##",
        ".....##", ".....##", "##...##", ".#####.",
    ),
    (
        "....##.", "...###.", "..####.", ".##.##.",
        "##..##.", "##..##.", "##..##.", "#######",
        "....##.", "....##.", "....##.", "....##.",
    ),
    (
        "#######", "##.....", "##.....", "##.....",
        "######.", ".....##", ".....##", ".....##",
        ".....##", ".....##", "##...##", ".#####.",
    ),
    (
        "..####.", ".##....", "##.....", "##.....",
        "######.", "##...##", "##...##", "##...##",
        "##...##", "##...##", "##...##", ".#####.",
    ),
    (
        "#######", ".....##", ".....##", "....##.",
        "....##.", "...##..", "...##..", "..##...",
        "..##...", ".##....", ".##....", ".##....",
    ),
    (
        ".#####.", "##...##", "##...##", "##...##",
        "##...##", ".#####.", "##...##", "##...##",
        "##...##", "##...##", "##...##", ".#####.",
    ),
    (
        ".#####.", "##...##", "##...##", "##...##",
        "##...##", "##...##", ".######", ".....##",
        ".....##", ".....##", "....##.", ".####..",
    ),
)


def build_digit_codes() -> torch.Tensor:
    """Return the ten fixed 84-D (+1/-1) digit prototype vectors.

    The paper only publishes these hand-designed codes as a raster figure.
    The explicit templates here are a reproducible transcription of the
    described 7x12 stylized digit construction.
    """

    codes = []
    for bitmap in DIGIT_BITMAPS:
        if len(bitmap) != 12 or any(len(row) != 7 for row in bitmap):
            raise ValueError("Each RBF bitmap must have shape 12x7.")
        values = [
            1.0 if pixel == "#" else -1.0
            for row in bitmap
            for pixel in row
        ]
        codes.append(values)
    return torch.tensor(codes, dtype=torch.float32)


class EuclideanRBF(nn.Module):
    """Compute squared Euclidean penalties to the class prototypes."""

    def __init__(self, trainable: bool = False) -> None:
        super().__init__()
        codes = build_digit_codes()
        if trainable:
            self.centers = nn.Parameter(codes)
        else:
            self.register_buffer("centers", codes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        differences = inputs.unsqueeze(1) - self.centers.unsqueeze(0)
        return differences.square().sum(dim=-1)


class RBFMSELoss(nn.Module):
    """Eq. (8): mean penalty of the RBF belonging to the target class."""

    def forward(
        self,
        penalties: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        selected = penalties.gather(1, targets.view(-1, 1))
        return selected.mean()


class MAPDiscriminativeLoss(nn.Module):
    """Discriminative MAP criterion from Eq. (9)."""

    def __init__(self, rubbish_class_j: float = 10.0) -> None:
        super().__init__()
        if rubbish_class_j <= 0:
            raise ValueError("rubbish_class_j must be positive.")
        self.rubbish_class_j = rubbish_class_j

    def forward(
        self,
        penalties: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        correct = penalties.gather(1, targets.view(-1, 1)).squeeze(1)
        rubbish = penalties.new_full(
            (penalties.shape[0], 1),
            -self.rubbish_class_j,
        )
        log_partition = torch.logsumexp(
            torch.cat((rubbish, -penalties), dim=1),
            dim=1,
        )
        return (correct + log_partition).mean()
