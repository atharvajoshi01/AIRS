"""Pytest configuration and fixtures."""

import pytest
import torch

from dgmlib.models.dgm import DGMNet
from dgmlib.pde.black_scholes import BlackScholesPDE
from dgmlib.utils.seeds import set_seed


@pytest.fixture(scope="session")
def device():
    """Device fixture."""
    return "cpu"


@pytest.fixture(scope="session")
def seed():
    """Seed fixture."""
    return 42


@pytest.fixture(scope="function", autouse=True)
def reset_seed(seed):
    """Reset seed before each test."""
    set_seed(seed)


@pytest.fixture
def simple_dgm_model():
    """Simple DGM model for testing."""
    return DGMNet(
        input_dim=2,
        hidden_dim=20,
        output_dim=1,
        num_layers=2,
    )


@pytest.fixture
def bs_pde():
    """Black-Scholes PDE fixture."""
    return BlackScholesPDE(
        r=0.05,
        sigma=0.2,
        K=100.0,
        T=1.0,
        option_type="call",
    )


@pytest.fixture
def sample_points():
    """Sample evaluation points."""
    t = torch.linspace(0, 1, 10).unsqueeze(1)
    S = torch.linspace(50, 150, 10).unsqueeze(1)
    return torch.cat([t, S], dim=1)
