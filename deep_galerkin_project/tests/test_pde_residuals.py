"""Test PDE residual computations."""

import torch

from dgmlib.pde.black_scholes import BlackScholesPDE


def test_bs_pde_residual_shape(bs_pde, simple_dgm_model):
    """Test that PDE residual has correct shape."""
    points = torch.randn(100, 2)
    residual = bs_pde.pde_residual(simple_dgm_model, points)

    assert residual.shape == (100,)


def test_bs_pde_residual_gradient_flow(bs_pde, simple_dgm_model):
    """Test that gradients flow through residual computation."""
    points = torch.randn(10, 2)
    residual = bs_pde.pde_residual(simple_dgm_model, points)
    loss = residual.pow(2).mean()

    loss.backward()

    # Check that model parameters have gradients
    for param in simple_dgm_model.parameters():
        assert param.grad is not None


def test_bs_terminal_condition_correctness(bs_pde):
    """Test terminal condition matches payoff."""
    # At t=T, value should equal payoff
    T = bs_pde.params["T"]
    K = bs_pde.params["K"]

    S_values = torch.tensor([80.0, 100.0, 120.0])
    t_values = torch.full_like(S_values, T)

    points = torch.stack([t_values, S_values], dim=1)

    # Create a simple model that outputs the payoff
    class PayoffModel(torch.nn.Module):
        def forward(self, x):
            S = x[:, 1:2]
            return torch.maximum(S - K, torch.zeros_like(S))

    model = PayoffModel()

    ic_residual = bs_pde.initial_condition(model, points)

    # Residual should be near zero
    assert torch.allclose(ic_residual, torch.zeros_like(ic_residual), atol=1e-6)


def test_bs_analytical_vs_pde_solution(bs_pde):
    """Test that analytical solution satisfies PDE (approximately)."""
    # The analytical solution should have very small residual
    points = torch.tensor([
        [0.0, 100.0],
        [0.5, 100.0],
        [0.9, 100.0],
    ])

    # Create model that returns analytical solution
    class AnalyticalModel(torch.nn.Module):
        def __init__(self, pde):
            super().__init__()
            self.pde = pde

        def forward(self, x):
            return self.pde.analytical_solution(x).unsqueeze(1)

    model = AnalyticalModel(bs_pde)

    # Due to numerical differentiation, residual won't be exactly zero
    # but should be small
    residual = bs_pde.pde_residual(model, points)

    # Allow some tolerance due to numerical differentiation
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-2)
