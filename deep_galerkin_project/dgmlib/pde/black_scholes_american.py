"""
American option pricing via obstacle/penalty method.

Formulation: min{L[V], V - φ(S)} = 0
where φ(S) is the payoff function (exercise value).

We use a smooth penalty approximation for stability.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from dgmlib.pde.black_scholes import BlackScholesPDE
from dgmlib.utils.autodiff import compute_gradient, compute_hessian


class BlackScholesAmericanPDE(BlackScholesPDE):
    """American option pricing with obstacle penalty method."""

    def __init__(
        self,
        r: float,
        sigma: float,
        K: float,
        T: float,
        option_type: str = "put",
        S_min: float = 0.0,
        S_max: float = 200.0,
        penalty_lambda: float = 100.0,
    ) -> None:
        """
        Initialize American option PDE.

        Args:
            r: Risk-free rate.
            sigma: Volatility.
            K: Strike price.
            T: Time to maturity.
            option_type: "call" or "put" (typically put for American).
            S_min: Minimum stock price.
            S_max: Maximum stock price.
            penalty_lambda: Penalty coefficient for obstacle constraint.
        """
        super().__init__(r, sigma, K, T, option_type, S_min, S_max)
        self.penalty_lambda = penalty_lambda

    def pde_residual(
        self,
        model: nn.Module,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        American option PDE residual with obstacle penalty.

        We enforce: min{L[V], V - φ(S)} = 0
        Penalty form: L[V] + λ * max(φ(S) - V, 0) = 0

        Args:
            points: Tensor of shape (N, 2) with columns [t, S].

        Returns:
            Residual of shape (N,).
        """
        t = points[:, 0:1]
        S = points[:, 1:2]

        t.requires_grad_(True)
        S.requires_grad_(True)

        V = model(torch.cat([t, S], dim=1))

        # Compute PDE operator L[V]
        V_t = compute_gradient(V, t)
        V_S = compute_gradient(V, S)
        V_SS = compute_hessian(V, S)

        sigma = self.params["sigma"]
        r = self.params["r"]

        L_V = (
            V_t
            + 0.5 * sigma**2 * S**2 * V_SS
            + r * S * V_S
            - r * V
        )

        # Payoff (obstacle)
        phi = self.payoff(S)

        # Penalty term: max(φ - V, 0)
        obstacle_penalty = torch.maximum(phi - V, torch.zeros_like(V))

        # Combined residual
        residual = L_V + self.penalty_lambda * obstacle_penalty

        return residual.squeeze()

    def analytical_solution(
        self,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        No closed-form solution for American options.
        Returns None (will use finite difference for comparison).
        """
        return None
