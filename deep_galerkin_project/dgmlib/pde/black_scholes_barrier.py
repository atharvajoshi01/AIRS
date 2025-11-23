"""
Barrier option pricing under Black-Scholes.

Supports up-and-out, down-and-out, up-and-in, down-and-in variants.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from dgmlib.pde.black_scholes import BlackScholesPDE


class BlackScholesBarrierPDE(BlackScholesPDE):
    """Barrier option pricing."""

    def __init__(
        self,
        r: float,
        sigma: float,
        K: float,
        T: float,
        barrier: float,
        barrier_type: str = "up-and-out",
        option_type: str = "call",
        S_min: float = 0.0,
        S_max: float = 200.0,
    ) -> None:
        """
        Initialize barrier option PDE.

        Args:
            r: Risk-free rate.
            sigma: Volatility.
            K: Strike price.
            T: Time to maturity.
            barrier: Barrier level.
            barrier_type: One of "up-and-out", "down-and-out", "up-and-in", "down-and-in".
            option_type: "call" or "put".
            S_min: Minimum stock price.
            S_max: Maximum stock price.
        """
        super().__init__(r, sigma, K, T, option_type, S_min, S_max)

        if barrier_type not in ["up-and-out", "down-and-out", "up-and-in", "down-and-in"]:
            raise ValueError(f"Invalid barrier_type: {barrier_type}")

        self.barrier = barrier
        self.barrier_type = barrier_type

        # Adjust domain based on barrier
        if "up" in barrier_type:
            self.domain["S"] = (S_min, min(S_max, barrier))
        else:  # down
            self.domain["S"] = (max(S_min, barrier), S_max)

    def boundary_condition(
        self,
        model: nn.Module,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce boundary conditions including barrier constraint.

        For "out" options: V = 0 at barrier.
        For "in" options: V = vanilla option value at barrier (approximation).

        Args:
            points: Boundary points of shape (N, 2).

        Returns:
            BC residual of shape (N,).
        """
        t = points[:, 0:1]
        S = points[:, 1:2]

        V_pred = model(torch.cat([t, S], dim=1))

        # Standard BC from parent class
        residual = super().boundary_condition(model, points)

        # Barrier boundary condition
        if "up" in self.barrier_type:
            # At upper barrier
            mask_barrier = (S >= self.barrier * 0.99)
        else:
            # At lower barrier
            mask_barrier = (S <= self.barrier * 1.01)

        if "out" in self.barrier_type:
            # Knock-out: V = 0 at barrier
            V_barrier = torch.zeros_like(V_pred)
        else:
            # Knock-in: approximate with vanilla payoff
            V_barrier = self.payoff(S)

        residual = torch.where(
            mask_barrier.squeeze(),
            (V_pred - V_barrier).squeeze(),
            residual
        )

        return residual

    def analytical_solution(
        self,
        points: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Analytical solution for some barrier options exists but is complex.
        Return None for now (use Monte Carlo for benchmarking).
        """
        # TODO: Implement analytical formulas for standard barrier options
        return None
