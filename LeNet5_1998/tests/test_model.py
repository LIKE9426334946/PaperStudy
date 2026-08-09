"""Architecture-level checks against the dimensions in Section II-B."""

import torch

from models import LeNet5, count_trainable_parameters
from models.layers import C3_CONNECTIONS, ScaledTanh, TrainableSubsampling


def test_paper_parameter_count_and_shapes() -> None:
    model = LeNet5()
    features = model.forward_features(torch.zeros(2, 1, 32, 32))
    assert count_trainable_parameters(model) == 60000
    assert features["c1"].shape == (2, 6, 28, 28)
    assert features["s2"].shape == (2, 6, 14, 14)
    assert features["c3"].shape == (2, 16, 10, 10)
    assert features["s4"].shape == (2, 16, 5, 5)
    assert features["c5"].shape == (2, 120, 1, 1)
    assert features["f6"].shape == (2, 84)
    assert features["penalties"].shape == (2, 10)


def test_c3_connection_table_has_1516_parameters() -> None:
    model = LeNet5()
    assert len(C3_CONNECTIONS) == 16
    assert sum(
        parameter.numel() for parameter in model.c3.parameters()
    ) == 1516


def test_scaled_tanh_has_paper_asymptotes() -> None:
    activation = ScaledTanh()
    result = activation(torch.tensor([-100.0, 0.0, 100.0]))
    expected = torch.tensor([-1.7159, 0.0, 1.7159])
    assert torch.allclose(result, expected, atol=1e-4)


def test_subsampling_uses_a_four_pixel_sum() -> None:
    layer = TrainableSubsampling(1, torch.nn.Identity())
    with torch.no_grad():
        layer.scale.fill_(1.0)
        layer.bias.zero_()
    inputs = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    expected = torch.tensor([[[[10.0, 18.0], [42.0, 50.0]]]])
    assert torch.equal(layer(inputs), expected)
