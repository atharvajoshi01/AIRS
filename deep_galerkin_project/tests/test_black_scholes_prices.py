"""Test Black-Scholes pricing against known values."""

import numpy as np
import pytest
import torch

from dgmlib.pde.black_scholes import BlackScholesPDE
from dgmlib.utils.numerics import black_scholes_analytical


@pytest.mark.parametrize(
    "S,K,r,sigma,T,option_type,expected_price",
    [
        # ATM call
        (100, 100, 0.05, 0.2, 1.0, "call", 10.45),
        # ITM call
        (110, 100, 0.05, 0.2, 1.0, "call", 15.15),
        # OTM call
        (90, 100, 0.05, 0.2, 1.0, "call", 6.04),
        # ATM put
        (100, 100, 0.05, 0.2, 1.0, "put", 5.57),
    ],
)
def test_bs_analytical_known_values(S, K, r, sigma, T, option_type, expected_price):
    """Test analytical BS formula against known values."""
    price, _, _ = black_scholes_analytical(S, K, r, sigma, T, option_type)

    # Allow 1% tolerance
    assert abs(price - expected_price) < expected_price * 0.01


def test_bs_pde_analytical_consistency():
    """Test that PDE analytical method matches numerics function."""
    pde = BlackScholesPDE(r=0.05, sigma=0.2, K=100.0, T=1.0, option_type="call")

    points = torch.tensor([
        [0.0, 90.0],
        [0.0, 100.0],
        [0.0, 110.0],
    ])

    pde_prices = pde.analytical_solution(points).numpy()

    for i, point in enumerate(points):
        S = point[1].item()
        price_numeric, _, _ = black_scholes_analytical(
            S, 100.0, 0.05, 0.2, 1.0, "call"
        )

        assert abs(pde_prices[i] - price_numeric) < 1e-5


def test_bs_call_put_parity():
    """Test put-call parity: C - P = S - K*exp(-r*T)."""
    S = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    call_price, _, _ = black_scholes_analytical(S, K, r, sigma, T, "call")
    put_price, _, _ = black_scholes_analytical(S, K, r, sigma, T, "put")

    parity_lhs = call_price - put_price
    parity_rhs = S - K * np.exp(-r * T)

    assert abs(parity_lhs - parity_rhs) < 1e-6


def test_bs_boundary_conditions():
    """Test boundary conditions for calls and puts."""
    # Call at S=0 should be 0
    pde_call = BlackScholesPDE(r=0.05, sigma=0.2, K=100.0, T=1.0, option_type="call")
    price_at_zero = pde_call.analytical_solution(torch.tensor([[0.0, 0.0]])).item()
    assert abs(price_at_zero) < 1e-6

    # Put at S=0 should be K*exp(-r*T)
    pde_put = BlackScholesPDE(r=0.05, sigma=0.2, K=100.0, T=1.0, option_type="put")
    price_at_zero = pde_put.analytical_solution(torch.tensor([[0.0, 0.0]])).item()
    expected = 100.0 * np.exp(-0.05 * 1.0)
    assert abs(price_at_zero - expected) < 1e-4
