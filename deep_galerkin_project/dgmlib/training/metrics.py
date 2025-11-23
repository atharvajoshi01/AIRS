"""Metrics for evaluating PDE solver performance."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from dgmlib.pde.base_pde import BasePDE
from dgmlib.utils.autodiff import compute_gradient


def compute_metrics(
    model: nn.Module,
    pde: BasePDE,
    test_points: torch.Tensor,
    analytical_values: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        model: Trained model.
        pde: PDE object.
        test_points: Test points for evaluation.
        analytical_values: Optional analytical solution values.

    Returns:
        Dictionary of metrics.
    """
    metrics = {}

    model.eval()

    # PDE residual (requires gradients)
    pde_residual = pde.pde_residual(model, test_points)
    with torch.no_grad():
        metrics["residual_l2"] = torch.sqrt(torch.mean(pde_residual**2)).item()
        metrics["residual_max"] = torch.max(torch.abs(pde_residual)).item()

    # Price comparison (if analytical available)
    if analytical_values is not None:
        with torch.no_grad():
            pred_values = model(test_points).squeeze()
            mae = torch.mean(torch.abs(pred_values - analytical_values)).item()
            rmse = torch.sqrt(torch.mean((pred_values - analytical_values)**2)).item()
            mape = torch.mean(
                torch.abs((pred_values - analytical_values) / (analytical_values + 1e-8))
            ).item() * 100

            metrics["mae"] = mae
            metrics["rmse"] = rmse
            metrics["mape"] = mape

    return metrics


def compute_greeks(
    model: nn.Module,
    points: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute option Greeks via automatic differentiation.

    Args:
        model: Trained model.
        points: Evaluation points of shape (N, d) where last dim is S.

    Returns:
        Dictionary with "delta" and "gamma".
    """
    model.eval()

    # Assume last dimension is S (stock price)
    S = points[:, -1:].clone().detach().requires_grad_(True)
    other_vars = points[:, :-1]

    inputs = torch.cat([other_vars, S], dim=1)
    V = model(inputs)

    # Delta: ∂V/∂S
    delta = compute_gradient(V, S)

    # Gamma: ∂²V/∂S²
    delta_sum = delta.sum()
    gamma = torch.autograd.grad(
        delta_sum,
        S,
        create_graph=False,
    )[0]

    return {
        "delta": delta.detach(),
        "gamma": gamma.detach(),
    }
