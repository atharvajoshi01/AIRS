"""Test DGM layer and network shapes."""

import torch

from dgmlib.models.dgm import DGMLayer, DGMNet
from dgmlib.models.mlp_baseline import MLPBaseline


def test_dgm_layer_forward():
    """Test DGM layer forward pass."""
    layer = DGMLayer(input_dim=3, output_dim=20)

    x = torch.randn(32, 3)
    S = torch.randn(32, 20)

    S_new = layer(x, S)

    assert S_new.shape == (32, 20)


def test_dgm_net_forward():
    """Test DGM network forward pass."""
    model = DGMNet(
        input_dim=2,
        hidden_dim=30,
        output_dim=1,
        num_layers=3,
    )

    x = torch.randn(64, 2)
    out = model(x)

    assert out.shape == (64, 1)


def test_dgm_net_parameter_count():
    """Test parameter counting."""
    model = DGMNet(
        input_dim=2,
        hidden_dim=20,
        num_layers=2,
    )

    n_params = model.count_parameters()
    assert n_params > 0
    assert isinstance(n_params, int)


def test_mlp_baseline_forward():
    """Test MLP baseline forward pass."""
    model = MLPBaseline(
        input_dim=2,
        hidden_dim=30,
        num_layers=3,
    )

    x = torch.randn(64, 2)
    out = model(x)

    assert out.shape == (64, 1)


def test_model_gradient_flow():
    """Test that gradients flow through model."""
    model = DGMNet(input_dim=2, hidden_dim=20, num_layers=2)

    x = torch.randn(10, 2)
    out = model(x)
    loss = out.pow(2).mean()

    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_spectral_norm():
    """Test spectral normalization option."""
    model = DGMNet(
        input_dim=2,
        hidden_dim=20,
        num_layers=2,
        use_spectral_norm=True,
    )

    # Check that spectral norm is applied
    assert hasattr(model.input_layer, "weight_orig")
    assert hasattr(model.output_layer, "weight_orig")
