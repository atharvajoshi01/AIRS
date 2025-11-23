"""Test Greeks computation and consistency."""

import numpy as np
import pytest
import torch

from dgmlib.pde.black_scholes import BlackScholesPDE
from dgmlib.training.metrics import compute_greeks
from dgmlib.utils.numerics import black_scholes_analytical


def test_call_delta_bounds():
    """Test that call delta is in [0, 1]."""
    pde = BlackScholesPDE(r=0.05, sigma=0.2, K=100.0, T=1.0, option_type="call")

    S_values = np.linspace(50, 150, 20)

    for S in S_values:
        _, delta, _ = black_scholes_analytical(S, 100.0, 0.05, 0.2, 1.0, "call")
        assert 0.0 <= delta <= 1.0, f"Delta out of bounds at S={S}: {delta}"


def test_put_delta_bounds():
    """Test that put delta is in [-1, 0]."""
    pde = BlackScholesPDE(r=0.05, sigma=0.2, K=100.0, T=1.0, option_type="put")

    S_values = np.linspace(50, 150, 20)

    for S in S_values:
        _, delta, _ = black_scholes_analytical(S, 100.0, 0.05, 0.2, 1.0, "put")
        assert -1.0 <= delta <= 0.0, f"Put delta out of bounds at S={S}: {delta}"


def test_gamma_positive():
    """Test that gamma is always positive."""
    S_values = np.linspace(50, 150, 20)

    for S in S_values:
        _, _, gamma = black_scholes_analytical(S, 100.0, 0.05, 0.2, 1.0, "call")
        assert gamma >= 0.0, f"Gamma negative at S={S}: {gamma}"


def test_gamma_call_put_equal():
    """Test that gamma is the same for calls and puts."""
    S_values = [80.0, 100.0, 120.0]

    for S in S_values:
        _, _, gamma_call = black_scholes_analytical(S, 100.0, 0.05, 0.2, 1.0, "call")
        _, _, gamma_put = black_scholes_analytical(S, 100.0, 0.05, 0.2, 1.0, "put")

        assert abs(gamma_call - gamma_put) < 1e-6


def test_compute_greeks_shape():
    """Test Greeks computation from model."""
    from dgmlib.models.dgm import DGMNet

    model = DGMNet(input_dim=2, hidden_dim=20, num_layers=2)

    points = torch.tensor([
        [0.0, 90.0],
        [0.0, 100.0],
        [0.0, 110.0],
    ])

    greeks = compute_greeks(model, points)

    assert "delta" in greeks
    assert "gamma" in greeks
    assert greeks["delta"].shape == (3, 1)
    assert greeks["gamma"].shape == (3, 1)
