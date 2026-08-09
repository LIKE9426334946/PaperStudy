"""Train LeNet-5 on MNIST with paper or fast experiment settings."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader, Subset

from datasets import build_dataloaders
from models import (
    LeNet5,
    MAPDiscriminativeLoss,
    RBFMSELoss,
    count_trainable_parameters,
)
from optim import (
    StochasticDiagonalLevenbergMarquardt,
    estimate_rbf_ggn_diagonal,
)
from utils import (
    choose_device,
    evaluate,
    learning_rate_for_epoch,
    load_config,
    resolve_project_paths,
    set_reproducible,
)
from utils.io import append_jsonl, load_checkpoint, save_checkpoint, save_json


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/paper.yaml",
        help="YAML path, relative to this paper directory.",
    )
    parser.add_argument("--resume", help="Checkpoint path to resume.")
    parser.add_argument("--device", help="Override the configured device.")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a synthetic forward/backward/optimizer check without MNIST.",
    )
    return parser.parse_args()


def build_model(config: dict[str, Any]) -> LeNet5:
    return LeNet5(
        activation_amplitude=float(config["activation_amplitude"]),
        activation_slope=float(config["activation_slope"]),
        trainable_rbf=bool(config.get("trainable_rbf", False)),
    )


def build_criterion(config: dict[str, Any]) -> nn.Module:
    if config["name"] == "rbf_mse":
        return RBFMSELoss()
    if config["name"] == "map":
        return MAPDiscriminativeLoss(
            rubbish_class_j=float(config["rubbish_class_j"])
        )
    raise ValueError(f"Unsupported loss: {config['name']}.")


def build_optimizer(
    model: LeNet5,
    config: dict[str, Any],
    initial_lr: float,
) -> Optimizer:
    if config["optimizer"] == "sdlm":
        return StochasticDiagonalLevenbergMarquardt(
            model.named_parameters(),
            global_lr=initial_lr,
            mu=float(config["mu"]),
        )
    if config["optimizer"] == "sgd":
        return SGD(
            model.parameters(),
            lr=initial_lr,
            momentum=float(config.get("momentum", 0.0)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    raise ValueError(f"Unsupported optimizer: {config['optimizer']}.")


def train_one_epoch(
    model: LeNet5,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int,
    gradient_clip_norm: float | None,
    max_batches: int | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch_index, (images, targets) in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        penalties = model(images)
        loss = criterion(penalties, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at batch {batch_index}.")
        loss.backward()
        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        if log_interval > 0 and (batch_index + 1) % log_interval == 0:
            print(
                f"epoch={epoch + 1} batch={batch_index + 1} "
                f"loss={total_loss / total_examples:.6f}"
            )
    if total_examples == 0:
        raise RuntimeError("Training loader produced no samples.")
    return total_loss / total_examples


def make_curvature_loader(
    train_loader: DataLoader,
    num_samples: int,
    batch_size: int,
) -> DataLoader:
    sample_count = min(num_samples, len(train_loader.dataset))
    subset = Subset(train_loader.dataset, range(sample_count))
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


def run_smoke_test(device: torch.device) -> None:
    """Exercise every model stage and one SDLM update using synthetic data."""

    set_reproducible(1998)
    model = LeNet5().to(device)
    if count_trainable_parameters(model) != 60000:
        raise AssertionError("LeNet-5 must contain exactly 60,000 parameters.")
    images = torch.rand(2, 1, 32, 32, device=device) * 1.275 - 0.1
    targets = torch.tensor([1, 7], device=device)
    optimizer = StochasticDiagonalLevenbergMarquardt(
        model.named_parameters(),
        global_lr=0.0005,
        mu=0.02,
    )
    curvature = estimate_rbf_ggn_diagonal(
        model,
        [(images[:1], targets[:1])],
        device=device,
        num_samples=1,
        method="hutchinson",
        probes=1,
    )
    optimizer.set_curvature(curvature)
    penalties = model(images)
    loss = RBFMSELoss()(penalties, targets)
    loss.backward()
    optimizer.step()
    if penalties.shape != (2, 10) or not torch.isfinite(loss):
        raise AssertionError("Smoke test produced invalid model output.")
    print(
        "Smoke test passed: "
        f"parameters=60000, output={tuple(penalties.shape)}, "
        f"loss={loss.item():.6f}"
    )


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        run_smoke_test(choose_device(args.device or "auto"))
        return

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = resolve_project_paths(load_config(config_path), PROJECT_ROOT)
    seed = int(config["experiment"]["seed"])
    set_reproducible(seed)
    device = choose_device(args.device or config.get("device", "auto"))
    print(f"device={device}")

    train_loader, test_loader, sampler = build_dataloaders(
        config["data"],
        seed,
        device,
    )
    model = build_model(config["model"]).to(device)
    parameter_count = count_trainable_parameters(model)
    if not config["model"].get("trainable_rbf", False) and parameter_count != 60000:
        raise RuntimeError(
            f"Expected 60,000 trainable parameters, got {parameter_count}."
        )
    criterion = build_criterion(config["loss"])
    training_config = config["training"]
    schedule = training_config["learning_rate_schedule"]
    optimizer = build_optimizer(
        model,
        training_config,
        learning_rate_for_epoch(schedule, 0),
    )

    start_epoch = 0
    best_error_rate = float("inf")
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer,
            map_location=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_error_rate = float(checkpoint["best_error_rate"])

    output_dir = Path(config["experiment"]["output_dir"])
    checkpoint_dir = Path(config["experiment"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)

    for epoch in range(start_epoch, int(training_config["epochs"])):
        epoch_start = time.perf_counter()
        sampler.set_epoch(epoch)
        learning_rate = learning_rate_for_epoch(schedule, epoch)
        if isinstance(
            optimizer,
            StochasticDiagonalLevenbergMarquardt,
        ):
            if config["loss"]["name"] != "rbf_mse":
                raise ValueError(
                    "SDLM curvature implementation currently requires rbf_mse."
                )
            optimizer.set_global_lr(learning_rate)
            curvature_loader = make_curvature_loader(
                train_loader,
                num_samples=int(training_config["curvature_samples"]),
                batch_size=int(training_config["curvature_batch_size"]),
            )
            curvature = estimate_rbf_ggn_diagonal(
                model,
                curvature_loader,
                device=device,
                num_samples=int(training_config["curvature_samples"]),
                method=str(training_config["curvature_method"]),
                probes=int(training_config.get("curvature_probes", 1)),
            )
            optimizer.set_curvature(curvature)
        else:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

        training_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            log_interval=int(training_config["log_interval"]),
            gradient_clip_norm=training_config.get("gradient_clip_norm"),
            max_batches=args.max_train_batches,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            max_batches=args.max_test_batches,
        )
        elapsed = time.perf_counter() - epoch_start
        record = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "training_loss": training_loss,
            "test_loss": test_metrics.loss,
            "test_accuracy": test_metrics.accuracy,
            "test_error_rate": test_metrics.error_rate,
            "elapsed_seconds": elapsed,
        }
        append_jsonl(output_dir / "metrics.jsonl", record)
        print(
            f"epoch={epoch + 1} train_loss={training_loss:.6f} "
            f"test_error={100.0 * test_metrics.error_rate:.3f}% "
            f"seconds={elapsed:.1f}"
        )

        is_best = test_metrics.error_rate < best_error_rate
        best_error_rate = min(best_error_rate, test_metrics.error_rate)
        save_checkpoint(
            checkpoint_dir / "latest.pth",
            model,
            optimizer,
            epoch,
            best_error_rate,
            config,
        )
        if is_best:
            save_checkpoint(
                checkpoint_dir / "best.pth",
                model,
                optimizer,
                epoch,
                best_error_rate,
                config,
            )


if __name__ == "__main__":
    main()
