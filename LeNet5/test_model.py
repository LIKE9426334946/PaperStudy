"""Small structural checks for the LeNet-5 reference implementation."""

import unittest

import torch

from .blocks import C3_CONNECTIONS, LeNet5MAPLoss
from .model import LeNet5


class LeNet5ReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = LeNet5()

    def test_forward_shapes(self) -> None:
        x = torch.randn(2, 1, 32, 32)
        energies, features = self.model(x, return_intermediates=True)

        self.assertEqual(tuple(features["C1"].shape), (2, 6, 28, 28))
        self.assertEqual(tuple(features["S2"].shape), (2, 6, 14, 14))
        self.assertEqual(tuple(features["C3"].shape), (2, 16, 10, 10))
        self.assertEqual(tuple(features["S4"].shape), (2, 16, 5, 5))
        self.assertEqual(tuple(features["C5"].shape), (2, 120, 1, 1))
        self.assertEqual(tuple(features["F6"].shape), (2, 84))
        self.assertEqual(tuple(energies.shape), (2, 10))

    def test_original_trainable_parameter_count(self) -> None:
        trainable = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertEqual(trainable, 60_000)

    def test_c3_has_sixty_map_connections(self) -> None:
        self.assertEqual(sum(map(len, C3_CONNECTIONS)), 60)
        c3_parameters = sum(
            parameter.numel() for parameter in self.model.c3.parameters()
        )
        self.assertEqual(c3_parameters, 1_516)

    def test_map_loss_backpropagates(self) -> None:
        x = torch.randn(2, 1, 32, 32)
        targets = torch.tensor([2, 7])
        energies = self.model(x)
        loss = LeNet5MAPLoss(junk_energy=40.0)(energies, targets)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(self.model.c1.weight.grad)


if __name__ == "__main__":
    unittest.main()
