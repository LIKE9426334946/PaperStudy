"""Stochastic diagonal Levenberg-Marquardt optimizer from Appendix C."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.func import functional_call, jacrev, vmap
from torch.optim import Optimizer


class StochasticDiagonalLevenbergMarquardt(Optimizer):
    """Apply Eq. (18) with the per-parameter step size from Eq. (21)."""

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        global_lr: float,
        mu: float = 0.02,
    ) -> None:
        items = [
            (name, parameter)
            for name, parameter in named_parameters
            if parameter.requires_grad
        ]
        if not items:
            raise ValueError("Optimizer received no trainable parameters.")
        if global_lr <= 0 or mu <= 0:
            raise ValueError("global_lr and mu must be positive.")
        self._names = {id(parameter): name for name, parameter in items}
        super().__init__(
            [parameter for _, parameter in items],
            defaults={"global_lr": global_lr, "mu": mu},
        )
        for group in self.param_groups:
            for parameter in group["params"]:
                self.state[parameter]["curvature"] = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )

    def set_global_lr(self, value: float) -> None:
        if value <= 0:
            raise ValueError("global learning rate must be positive.")
        for group in self.param_groups:
            group["global_lr"] = value

    @torch.no_grad()
    def set_curvature(self, curvature: dict[str, torch.Tensor]) -> None:
        expected = set(self._names.values())
        if set(curvature) != expected:
            missing = sorted(expected - set(curvature))
            extra = sorted(set(curvature) - expected)
            raise ValueError(
                f"Curvature keys do not match parameters; "
                f"missing={missing}, extra={extra}."
            )
        for group in self.param_groups:
            for parameter in group["params"]:
                name = self._names[id(parameter)]
                estimate = curvature[name]
                if estimate.shape != parameter.shape:
                    raise ValueError(f"Curvature shape mismatch for {name}.")
                self.state[parameter]["curvature"].copy_(
                    estimate.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            global_lr = float(group["global_lr"])
            mu = float(group["mu"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                curvature = self.state[parameter]["curvature"]
                denominator = curvature.add(mu)
                parameter.addcdiv_(
                    parameter.grad,
                    denominator,
                    value=-global_lr,
                )
        return loss


def _trainable_parameter_dict(model: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _empty_curvature(
    parameters: dict[str, nn.Parameter],
) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(parameter)
        for name, parameter in parameters.items()
    }


def _exact_batch_curvature(
    model: nn.Module,
    parameters: dict[str, nn.Parameter],
    buffers: dict[str, torch.Tensor],
    images: torch.Tensor,
) -> dict[str, torch.Tensor]:
    def single_f6(
        current_parameters: dict[str, nn.Parameter],
        current_buffers: dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> torch.Tensor:
        return functional_call(
            model,
            (current_parameters, current_buffers),
            (image.unsqueeze(0),),
            {"output": "f6"},
        ).squeeze(0)

    jacobian_function = jacrev(single_f6, argnums=0)
    jacobians = vmap(
        jacobian_function,
        in_dims=(None, None, 0),
        randomness="different",
    )(parameters, buffers, images)
    return {
        name: 2.0 * jacobian.square().sum(dim=(0, 1))
        for name, jacobian in jacobians.items()
    }


def _hutchinson_batch_curvature(
    model: nn.Module,
    parameters: dict[str, nn.Parameter],
    images: torch.Tensor,
    probes: int,
) -> dict[str, torch.Tensor]:
    estimates = _empty_curvature(parameters)
    parameter_values = tuple(parameters.values())
    for image in images:
        f6 = model(image.unsqueeze(0), output="f6").squeeze(0)
        for _ in range(probes):
            signs = torch.empty_like(f6).bernoulli_(0.5).mul_(2).sub_(1)
            gradients = torch.autograd.grad(
                outputs=(f6 * signs).sum(),
                inputs=parameter_values,
                retain_graph=True,
                allow_unused=False,
            )
            for (name, _), gradient in zip(
                parameters.items(),
                gradients,
                strict=True,
            ):
                estimates[name].add_(gradient.square(), alpha=2.0 / probes)
    return estimates


def estimate_rbf_ggn_diagonal(
    model: nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    num_samples: int,
    method: str = "exact",
    probes: int = 1,
) -> dict[str, torch.Tensor]:
    """Estimate the nonnegative diagonal Gauss-Newton curvature.

    For Eq. (8), the Hessian with respect to F6 is exactly 2I. Therefore
    the parameter-space generalized Gauss-Newton diagonal is
    2 * diag(J_F6^T J_F6). The exact method computes this Jacobian directly;
    the Hutchinson method is a faster unbiased diagonal estimator.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if method not in {"exact", "hutchinson"}:
        raise ValueError("method must be 'exact' or 'hutchinson'.")
    if probes <= 0:
        raise ValueError("probes must be positive.")

    was_training = model.training
    model.eval()
    parameters = _trainable_parameter_dict(model)
    buffers = dict(model.named_buffers())
    totals = _empty_curvature(parameters)
    processed = 0

    for images, _ in data_loader:
        if processed >= num_samples:
            break
        remaining = num_samples - processed
        images = images[:remaining].to(device)
        if method == "exact":
            batch_estimate = _exact_batch_curvature(
                model,
                parameters,
                buffers,
                images,
            )
        else:
            batch_estimate = _hutchinson_batch_curvature(
                model,
                parameters,
                images,
                probes,
            )
        batch_count = images.shape[0]
        for name in totals:
            totals[name].add_(batch_estimate[name])
        processed += batch_count

    model.train(was_training)
    if processed == 0:
        raise RuntimeError("Curvature loader produced no samples.")
    return {
        name: value.div(float(processed)).detach()
        for name, value in totals.items()
    }
