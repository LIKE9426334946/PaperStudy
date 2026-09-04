"""A paper-oriented PyTorch reference implementation of LeNet-5."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

try:
    from .blocks import (
        EuclideanRBF,
        ScaledTanh,
        SparseC3Convolution,
        TrainableSubsampling2d,
    )
except ImportError:  # Allows reading/running model.py directly from this folder.
    from blocks import (  # type: ignore[no-redef]
        EuclideanRBF,
        ScaledTanh,
        SparseC3Convolution,
        TrainableSubsampling2d,
    )


class LeNet5(nn.Module):
    """LeNet-5 as described by LeCun, Bottou, Bengio, and Haffner (1998).

    The returned values are RBF energies, not probabilities or ordinary
    classification logits. Therefore, the predicted class is ``argmin``.
    """

    def __init__(self, rbf_centers: Tensor | None = None) -> None:
        super().__init__()

        # C1: one 5x5 kernel bank creates six 28x28 feature maps.
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.c1_activation = ScaledTanh()

        # S2: 2x2 trainable subsampling creates six 14x14 feature maps.
        self.s2 = TrainableSubsampling2d(channels=6, kernel_size=2)
        self.s2_activation = ScaledTanh()

        # C3: Table-I sparse connections create sixteen 10x10 feature maps.
        self.c3 = SparseC3Convolution(kernel_size=5)
        self.c3_activation = ScaledTanh()

        # S4: 2x2 trainable subsampling creates sixteen 5x5 feature maps.
        self.s4 = TrainableSubsampling2d(channels=16, kernel_size=2)
        self.s4_activation = ScaledTanh()

        # C5: its 5x5 receptive field covers all of S4, yielding 120x1x1.
        self.c5 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.c5_activation = ScaledTanh()

        # F6: the 84 outputs correspond to a 7x12 character bitmap.
        self.f6 = nn.Linear(120, 84)
        self.f6_activation = ScaledTanh()

        # Output: fixed-template squared Euclidean distances.
        self.output = EuclideanRBF(centers=rbf_centers)

    def forward(
        self,
        x: Tensor,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 32, 32):
            raise ValueError(
                "LeNet-5 expects centered 32x32 grayscale images with shape "
                f"[B, 1, 32, 32], got {tuple(x.shape)}"
            )

        intermediates: dict[str, Tensor] = {}

        # C1 convolution and squashing
        c1 = self.c1(x)                     # [B, 6, 28, 28]
        c1 = self.c1_activation(c1)         # [B, 6, 28, 28]
        intermediates["C1"] = c1

        # S2 trainable subsampling and squashing
        s2 = self.s2(c1)                    # [B, 6, 14, 14]
        s2 = self.s2_activation(s2)         # [B, 6, 14, 14]
        intermediates["S2"] = s2

        # C3 sparse convolution and squashing
        c3 = self.c3(s2)                    # [B, 16, 10, 10]
        c3 = self.c3_activation(c3)         # [B, 16, 10, 10]
        intermediates["C3"] = c3

        # S4 trainable subsampling and squashing
        s4 = self.s4(c3)                    # [B, 16, 5, 5]
        s4 = self.s4_activation(s4)         # [B, 16, 5, 5]
        intermediates["S4"] = s4

        # C5 convolution over the full S4 spatial extent
        c5 = self.c5(s4)                    # [B, 120, 1, 1]
        c5 = self.c5_activation(c5)         # [B, 120, 1, 1]
        intermediates["C5"] = c5

        # F6 fully connected transformation
        flattened = c5.flatten(start_dim=1) # [B, 120]
        f6 = self.f6(flattened)             # [B, 84]
        f6 = self.f6_activation(f6)         # [B, 84]
        intermediates["F6"] = f6

        # Euclidean RBF output: one energy per digit class
        energies = self.output(f6)          # [B, 10]

        if return_intermediates:
            return energies, intermediates
        return energies

    def predict(self, x: Tensor) -> Tensor:
        """Return the class whose fixed RBF template has minimum energy."""

        energies = self.forward(x)
        if not isinstance(energies, Tensor):  # Defensive; false for this call.
            raise RuntimeError("unexpected intermediate-output tuple")
        return energies.argmin(dim=1)

    def extra_repr(self) -> str:
        trainable = sum(parameter.numel() for parameter in self.parameters())
        return f"trainable_parameters={trainable}, output=RBF energies"


def lenet5(rbf_centers: Tensor | None = None, **_: Any) -> LeNet5:
    """Construct the paper-faithful model.

    The unused keyword capture is intentionally small: it makes clear that
    optimizer, data, and training configuration do not belong to the model.
    """

    return LeNet5(rbf_centers=rbf_centers)
