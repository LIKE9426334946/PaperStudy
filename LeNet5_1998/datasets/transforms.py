"""Image transforms used for the MNIST reproduction."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as functional


def paper_normalize(image: torch.Tensor) -> torch.Tensor:
    """Map background 0 to -0.1 and foreground 1 to 1.175."""

    return image * 1.275 - 0.1


def prepare_mnist_tensor(image: torch.Tensor) -> torch.Tensor:
    """Pad a 28x28 MNIST tensor to 32x32 and apply paper normalization."""

    if image.ndim != 3 or image.shape != (1, 28, 28):
        raise ValueError(
            "Expected a tensor with shape (1, 28, 28), "
            f"received {tuple(image.shape)}."
        )
    padded = functional.pad(image, (2, 2, 2, 2), value=0.0)
    return paper_normalize(padded)


class DeterministicAffineDistortion:
    """Apply a deterministic affine distortion for one virtual sample.

    The paper names translation, scale, squeeze and horizontal shear but
    does not publish their sampling ranges. The ranges are therefore explicit
    configuration values rather than hidden assumptions.
    """

    def __init__(
        self,
        translate_pixels: float,
        scale_fraction: float,
        squeeze_fraction: float,
        shear_degrees: float,
        seed: int,
    ) -> None:
        self.translate_pixels = translate_pixels
        self.scale_fraction = scale_fraction
        self.squeeze_fraction = squeeze_fraction
        self.shear_degrees = shear_degrees
        self.seed = seed

    @staticmethod
    def _uniform(
        generator: torch.Generator,
        lower: float,
        upper: float,
    ) -> float:
        value = torch.rand((), generator=generator).item()
        return lower + (upper - lower) * value

    def __call__(self, image: torch.Tensor, sample_key: int) -> torch.Tensor:
        if image.shape != (1, 28, 28):
            raise ValueError("Affine distortion expects a (1, 28, 28) tensor.")

        generator = torch.Generator().manual_seed(self.seed + sample_key)
        translation_x = self._uniform(
            generator,
            -self.translate_pixels,
            self.translate_pixels,
        )
        translation_y = self._uniform(
            generator,
            -self.translate_pixels,
            self.translate_pixels,
        )
        scale = self._uniform(
            generator,
            1.0 - self.scale_fraction,
            1.0 + self.scale_fraction,
        )
        squeeze = self._uniform(
            generator,
            1.0 - self.squeeze_fraction,
            1.0 + self.squeeze_fraction,
        )
        shear_angle = self._uniform(
            generator,
            -self.shear_degrees,
            self.shear_degrees,
        )

        scale_x = max(scale * squeeze, 1e-4)
        scale_y = max(scale / squeeze, 1e-4)
        shear = math.tan(math.radians(shear_angle))
        translate_x = 2.0 * translation_x / image.shape[-1]
        translate_y = 2.0 * translation_y / image.shape[-2]

        theta = image.new_tensor(
            [
                [
                    1.0 / scale_x,
                    -shear / scale_x,
                    -translate_x,
                ],
                [0.0, 1.0 / scale_y, -translate_y],
            ]
        ).unsqueeze(0)
        batch = image.unsqueeze(0)
        grid = functional.affine_grid(
            theta,
            size=batch.shape,
            align_corners=False,
        )
        result = functional.grid_sample(
            batch,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return result.squeeze(0).clamp_(0.0, 1.0)
