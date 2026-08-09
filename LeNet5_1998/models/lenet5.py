"""Faithful PyTorch implementation of the 1998 LeNet-5 architecture."""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from .layers import (
    ScaledTanh,
    SparseC3Convolution,
    TrainableSubsampling,
    initialize_paper_uniform,
)
from .rbf import EuclideanRBF


class LeNet5(nn.Module):
    """LeNet-5 with sparse C3, trainable subsampling, F6 and RBF output."""

    def __init__(
        self,
        activation_amplitude: float = 1.7159,
        activation_slope: float = 2.0 / 3.0,
        trainable_rbf: bool = False,
    ) -> None:
        super().__init__()
        activation = partial(
            ScaledTanh,
            amplitude=activation_amplitude,
            slope=activation_slope,
        )
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.c1_activation = activation()
        self.s2 = TrainableSubsampling(6, activation())
        self.c3 = SparseC3Convolution(activation=activation())
        self.s4 = TrainableSubsampling(16, activation())
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.c5_activation = activation()
        self.f6 = nn.Linear(120, 84)
        self.f6_activation = activation()
        self.output = EuclideanRBF(trainable=trainable_rbf)
        self.apply(initialize_paper_uniform)

    def forward_features(
        self,
        inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features: dict[str, torch.Tensor] = {}
        features["c1"] = self.c1_activation(self.c1(inputs))
        features["s2"] = self.s2(features["c1"])
        features["c3"] = self.c3(features["s2"])
        features["s4"] = self.s4(features["c3"])
        features["c5"] = self.c5_activation(self.c5(features["s4"]))
        flattened = torch.flatten(features["c5"], start_dim=1)
        features["f6"] = self.f6_activation(self.f6(flattened))
        features["penalties"] = self.output(features["f6"])
        return features

    def forward(
        self,
        inputs: torch.Tensor,
        output: str = "penalties",
    ) -> torch.Tensor:
        features = self.forward_features(inputs)
        if output not in features:
            raise ValueError(f"Unknown output '{output}'.")
        return features[output]

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self(inputs).argmin(dim=1)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count model parameters that receive gradient updates."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
