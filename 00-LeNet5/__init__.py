"""LeNet-5 paper study package."""

from .blocks import LeNet5MAPLoss
from .model import LeNet5, lenet5

__all__ = ["LeNet5", "LeNet5MAPLoss", "lenet5"]
