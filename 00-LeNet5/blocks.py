"""Core building blocks for the paper-faithful LeNet-5 reference model."""

from __future__ import annotations

from typing import Literal, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# Table I in the paper: for every C3 output feature map, list the S2 maps
# connected to it. There are 60 input-output map connections in total.
C3_CONNECTIONS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    (3, 4, 5),
    (0, 4, 5),
    (0, 1, 5),
    (0, 1, 2, 3),
    (1, 2, 3, 4),
    (2, 3, 4, 5),
    (0, 3, 4, 5),
    (0, 1, 4, 5),
    (0, 1, 2, 5),
    (0, 1, 3, 4),
    (1, 2, 4, 5),
    (0, 2, 3, 5),
    (0, 1, 2, 3, 4, 5),
)


def scaled_tanh(x: Tensor) -> Tensor:
    """The paper's sigmoid-like squashing function: 1.7159*tanh(2x/3)."""

    return 1.7159 * torch.tanh((2.0 / 3.0) * x)


class ScaledTanh(nn.Module):
    """Module wrapper around :func:`scaled_tanh`."""

    def forward(self, x: Tensor) -> Tensor:
        return scaled_tanh(x)


class TrainableSubsampling2d(nn.Module):
    """LeNet-5's S2/S4 operation, including its trainable parameters.

    Each non-overlapping 2x2 neighborhood is summed. For every feature map,
    the sum is multiplied by one learned coefficient and shifted by one
    learned bias. The activation is kept outside this module so that the
    model's forward method shows the paper's data flow explicitly.
    """

    def __init__(self, channels: int, kernel_size: int = 2) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")

        self.channels = channels
        self.kernel_size = kernel_size
        self.scale = nn.Parameter(torch.empty(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.empty(1, channels, 1, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The paper specifies that these values are trainable, but not a
        # reusable PyTorch initialization rule. Starting at an exact average
        # is a documented engineering choice.
        nn.init.constant_(self.scale, 1.0 / (self.kernel_size**2))
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"expected [B, {self.channels}, H, W], got {tuple(x.shape)}"
            )

        # avg_pool2d computes mean(x); multiplying by the window area recovers
        # the sum explicitly described by the paper.
        neighborhood_sum = F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
        ) * (self.kernel_size**2)
        return neighborhood_sum * self.scale + self.bias


class SparseC3Convolution(nn.Module):
    """C3 convolution with the exact partial connectivity from Table I.

    A dense ``Conv2d(6, 16, 5)`` would allocate unused weights. Instead, one
    small convolution is constructed per output map and receives only the
    connected input maps. This makes the parameter count exactly 1,516.
    """

    def __init__(
        self,
        connections: Sequence[Sequence[int]] = C3_CONNECTIONS,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        normalized = tuple(tuple(group) for group in connections)
        if len(normalized) != 16:
            raise ValueError("C3 must define exactly 16 output feature maps")
        if any(not group for group in normalized):
            raise ValueError("every C3 output map needs at least one input map")
        if any(index < 0 or index > 5 for group in normalized for index in group):
            raise ValueError("C3 input-map indices must be in [0, 5]")

        self.connections = normalized
        self.convolutions = nn.ModuleList(
            nn.Conv2d(len(group), 1, kernel_size=kernel_size, stride=1)
            for group in self.connections
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"expected [B, 6, H, W], got {tuple(x.shape)}")

        output_maps = []
        for input_indices, convolution in zip(
            self.connections,
            self.convolutions,
            strict=True,
        ):
            selected_maps = x[:, input_indices, :, :]
            output_map = convolution(selected_maps)  # [B, 1, H-4, W-4]
            output_maps.append(output_map)

        return torch.cat(output_maps, dim=1)  # [B, 16, H-4, W-4]


# The paper fixes each RBF center to a +/-1 bitmap of its class on a 7x12
# grid. The original figure is graphical rather than a machine-readable table.
# These ten readable digit rasters are therefore an explicit reconstruction;
# callers may pass paper-extracted centers to EuclideanRBF instead.
_DIGIT_RASTERS: tuple[tuple[str, ...], ...] = (
    (
        ".......",
        "..###..",
        ".##.##.",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        ".##.##.",
        "..###..",
        ".......",
    ),
    (
        ".......",
        "...##..",
        "..###..",
        ".####..",
        "...##..",
        "...##..",
        "...##..",
        "...##..",
        "...##..",
        "...##..",
        ".######",
        ".......",
    ),
    (
        ".......",
        ".#####.",
        "##...##",
        ".....##",
        "....##.",
        "...##..",
        "..##...",
        ".##....",
        "##.....",
        "##.....",
        "#######",
        ".......",
    ),
    (
        ".......",
        ".#####.",
        "##...##",
        ".....##",
        ".....##",
        "..####.",
        ".....##",
        ".....##",
        ".....##",
        "##...##",
        ".#####.",
        ".......",
    ),
    (
        ".......",
        "....##.",
        "...###.",
        "..####.",
        ".##.##.",
        "##..##.",
        "##..##.",
        "#######",
        "....##.",
        "....##.",
        "....##.",
        ".......",
    ),
    (
        ".......",
        "#######",
        "##.....",
        "##.....",
        "##.....",
        "######.",
        ".....##",
        ".....##",
        ".....##",
        "##...##",
        ".#####.",
        ".......",
    ),
    (
        ".......",
        "..####.",
        ".##....",
        "##.....",
        "##.....",
        "######.",
        "##...##",
        "##...##",
        "##...##",
        ".##.##.",
        "..###..",
        ".......",
    ),
    (
        ".......",
        "#######",
        ".....##",
        "....##.",
        "....##.",
        "...##..",
        "...##..",
        "..##...",
        "..##...",
        ".##....",
        ".##....",
        ".......",
    ),
    (
        ".......",
        "..###..",
        ".##.##.",
        "##...##",
        ".##.##.",
        "..###..",
        ".##.##.",
        "##...##",
        "##...##",
        ".##.##.",
        "..###..",
        ".......",
    ),
    (
        ".......",
        "..###..",
        ".##.##.",
        "##...##",
        "##...##",
        ".######",
        ".....##",
        ".....##",
        ".....##",
        "....##.",
        ".####..",
        ".......",
    ),
)


def make_digit_rbf_centers() -> Tensor:
    """Return reconstructed fixed digit templates with shape ``[10, 84]``."""

    bitmaps = []
    for digit, rows in enumerate(_DIGIT_RASTERS):
        if len(rows) != 12 or any(len(row) != 7 for row in rows):
            raise RuntimeError(f"digit {digit} is not a 12-row by 7-column bitmap")
        bitmap = torch.tensor(
            [[character == "#" for character in row] for row in rows],
            dtype=torch.bool,
        )
        center = torch.where(bitmap, 1.0, -1.0).flatten()
        bitmaps.append(center)
    return torch.stack(bitmaps)


class EuclideanRBF(nn.Module):
    """Fixed-center Euclidean RBF output units from the original LeNet-5."""

    def __init__(self, centers: Tensor | None = None) -> None:
        super().__init__()
        if centers is None:
            centers = make_digit_rbf_centers()
        centers = torch.as_tensor(centers, dtype=torch.float32)
        if centers.ndim != 2 or centers.shape[1] != 84:
            raise ValueError("centers must have shape [num_classes, 84]")

        # Fixed templates are model state, but they are not trainable weights.
        self.register_buffer("centers", centers.clone())

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != 84:
            raise ValueError(f"expected [B, 84], got {tuple(x.shape)}")

        # x[:, None, :]       : [B, 1, 84]
        # centers[None, :, :] : [1, K, 84]
        difference = x[:, None, :] - self.centers[None, :, :]
        return difference.square().sum(dim=-1)  # [B, K], lower is better


class LeNet5MAPLoss(nn.Module):
    """Equation (9), the MAP criterion used with LeNet-5's RBF energies.

    ``junk_energy`` is the paper's positive constant ``J``. It remains an
    explicit constructor argument because its chosen value is a training
    setting rather than part of the network architecture.
    """

    def __init__(
        self,
        junk_energy: float,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        if junk_energy <= 0:
            raise ValueError("junk_energy must be positive")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.junk_energy = float(junk_energy)
        self.reduction = reduction

    def forward(self, energies: Tensor, targets: Tensor) -> Tensor:
        if energies.ndim != 2:
            raise ValueError("energies must have shape [B, num_classes]")
        if targets.ndim != 1 or targets.shape[0] != energies.shape[0]:
            raise ValueError("targets must have shape [B]")

        targets = targets.to(device=energies.device, dtype=torch.long)
        target_energy = energies.gather(1, targets[:, None]).squeeze(1)

        # log(exp(-J) + sum_i exp(-y_i)), computed stably with logsumexp.
        junk_logit = energies.new_full((energies.shape[0], 1), -self.junk_energy)
        log_normalizer = torch.logsumexp(
            torch.cat((junk_logit, -energies), dim=1),
            dim=1,
        )
        per_sample_loss = target_energy + log_normalizer

        if self.reduction == "none":
            return per_sample_loss
        if self.reduction == "sum":
            return per_sample_loss.sum()
        return per_sample_loss.mean()
