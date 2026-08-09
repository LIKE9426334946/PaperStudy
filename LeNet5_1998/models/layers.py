"""Paper-faithful layers used by LeNet-5."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as functional


class ScaledTanh(nn.Module):
    """Scaled tanh from Eq. (6): A * tanh(S * x)."""

    def __init__(
        self,
        amplitude: float = 1.7159,
        slope: float = 2.0 / 3.0,
    ) -> None:
        super().__init__()
        self.amplitude = amplitude
        self.slope = slope

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.amplitude * torch.tanh(self.slope * inputs)


class TrainableSubsampling(nn.Module):
    """Non-overlapping 2x2 sum pooling with one scale/bias per map.

    This implements the S2/S4 operation described in Section II-B rather
    than replacing it with modern max pooling.
    """

    def __init__(
        self,
        channels: int,
        activation: nn.Module,
        kernel_size: int = 2,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.scale = nn.Parameter(torch.empty(channels))
        self.bias = nn.Parameter(torch.empty(channels))
        self.activation = activation

    def reset_parameters(self) -> None:
        bound = 2.4 / float(self.kernel_size**2)
        nn.init.uniform_(self.scale, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = functional.avg_pool2d(
            inputs,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
        )
        pooled = pooled * float(self.kernel_size**2)
        outputs = pooled * self.scale.view(1, -1, 1, 1)
        outputs = outputs + self.bias.view(1, -1, 1, 1)
        return self.activation(outputs)


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


class SparseC3Convolution(nn.Module):
    """LeNet-5 C3 convolution with the sparse mapping from Table 1."""

    def __init__(
        self,
        connections: Sequence[Sequence[int]] = C3_CONNECTIONS,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.connections = tuple(tuple(item) for item in connections)
        self.convolutions = nn.ModuleList(
            nn.Conv2d(len(indices), 1, kernel_size=5)
            for indices in self.connections
        )
        self.activation = (
            activation if activation is not None else ScaledTanh()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feature_maps = []
        for indices, convolution in zip(
            self.connections,
            self.convolutions,
            strict=True,
        ):
            selected = inputs[:, indices, :, :]
            feature_maps.append(convolution(selected))
        return self.activation(torch.cat(feature_maps, dim=1))


def initialize_paper_uniform(module: nn.Module) -> None:
    """Initialize weights uniformly in [-2.4/fan_in, 2.4/fan_in]."""

    if isinstance(module, (nn.Conv2d, nn.Linear)):
        fan_in = module.weight[0].numel()
        bound = 2.4 / float(fan_in)
        nn.init.uniform_(module.weight, -bound, bound)
        if module.bias is not None:
            nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, TrainableSubsampling):
        module.reset_parameters()
