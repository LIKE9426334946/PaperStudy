"""Evaluate a trained LeNet-5 checkpoint on the MNIST test set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from datasets import build_test_loader
from train import PROJECT_ROOT, build_criterion, build_model
from utils import choose_device, evaluate, resolve_project_paths, set_reproducible
from utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/evaluation")
    parser.add_argument("--max-test-batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    config = resolve_project_paths(checkpoint["config"], PROJECT_ROOT)
    seed = int(config["experiment"]["seed"])
    set_reproducible(seed)
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = build_criterion(config["loss"])
    test_loader = build_test_loader(config["data"], seed, device)
    metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        max_batches=args.max_test_batches,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "test_metrics.json", metrics.as_dict())
    with (output_dir / "confusion_matrix.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["target/prediction", *range(10)])
        for label, row in enumerate(metrics.confusion_matrix):
            writer.writerow([label, *row])

    print(
        f"examples={metrics.examples} "
        f"accuracy={100.0 * metrics.accuracy:.3f}% "
        f"error_rate={100.0 * metrics.error_rate:.3f}%"
    )


if __name__ == "__main__":
    main()
