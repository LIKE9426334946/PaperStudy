"""Classification metrics and evaluation loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ClassificationMetrics:
    """Loss, accuracy, error rate and confusion matrix."""

    loss: float
    accuracy: float
    error_rate: float
    examples: int
    confusion_matrix: list[list[int]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "error_rate": self.error_rate,
            "examples": self.examples,
            "confusion_matrix": self.confusion_matrix,
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: Any,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> ClassificationMetrics:
    """Evaluate RBF predictions, where the smallest penalty wins."""

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_examples = 0
    correct = 0
    confusion = torch.zeros((10, 10), dtype=torch.int64)

    for batch_index, (images, targets) in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        targets = targets.to(device)
        penalties = model(images)
        loss = criterion(penalties, targets)
        predictions = penalties.argmin(dim=1)
        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        correct += predictions.eq(targets).sum().item()
        flat_indices = (targets * 10 + predictions).to("cpu")
        confusion += torch.bincount(flat_indices, minlength=100).reshape(10, 10)

    model.train(was_training)
    if total_examples == 0:
        raise RuntimeError("Evaluation loader produced no samples.")
    accuracy = correct / total_examples
    return ClassificationMetrics(
        loss=total_loss / total_examples,
        accuracy=accuracy,
        error_rate=1.0 - accuracy,
        examples=total_examples,
        confusion_matrix=confusion.tolist(),
    )
