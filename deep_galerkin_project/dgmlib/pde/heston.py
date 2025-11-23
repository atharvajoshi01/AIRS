"""
Heston stochastic volatility model PDE.

State: (S, v, t) where v is variance.
PDE: ∂V/∂t + 0.5*v*S²*∂²V/∂S² + ρ*σ*v*S*∂²V/∂S∂v + 0.5*σ²*v*∂²V/∂v²
     + r*S*∂V/∂S + κ(θ - v)*∂V/∂v - r*V = 0
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from dgmlib.pde.base_pde import BasePDE
from dgmlib.utils.autodiff import compute_gradient, compute_hessian


class HestonPDE(BasePDE):
    """European option under Heston stochastic volatility."""

    def __init__(
        self,
        r: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        K: float,
        T: float,
        option_type: str = "call",
        S_min: float = 0.0,
        S_max: float = 200.0,
        v_min: float = 0.01,
        v_max: float = 1.0,
    ) -> None:
        """
        Initialize Heston PDE.

        Args:
            r: Risk-free rate.
            kappa: Mean reversion speed.
            theta: Long-term variance.
            sigma_v: Volatility of variance (vol of vol).
            rho: Correlation between asset and variance.
            K: Strike price.
            T: Time to maturity.
            option_type: "call" or "put".
            S_min: Min stock price.
            S_max: Max stock price.
            v_min: Min variance.
            v_max: Max variance.
        """
        domain = {
            "S": (S_min, S_max),
            "v": (v_min, v_max),
            "t": (0.0, T),
        }
        params = {
            "r": r,
            "kappa": kappa,
            "theta": theta,
            "sigma_v": sigma_v,
            "rho": rho,
            "K": K,
            "T": T,
        }
        super().__init__(domain, params)

        if option_type not in ["call", "put"]:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
        self.option_type = option_type

    def _validate_domain(self) -> None:
        """Validate domain variables."""
        required = {"S", "v", "t"}
        if not required.issubset(self.domain.keys()):
            raise ValueError(f"Domain must contain {required}")

    def _validate_params(self) -> None:
        """Validate parameters."""
        required = {"r", "kappa", "theta", "sigma_v", "rho", "K", "T"}
        if not required.issubset(self.params.keys()):
            raise ValueError(f"Params must contain {required}")

    def pde_residual(
        self,
        model: nn.Module,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Heston PDE residual.

        Args:
            points: Tensor of shape (N, 3) with columns [t, S, v].

        Returns:
            Residual of shape (N,).
        """
        t = points[:, 0:1]
        S = points[:, 1:2]
        v = points[:, 2:3]

        t.requires_grad_(True)
        S.requires_grad_(True)
        v.requires_grad_(True)

        inputs = torch.cat([t, S, v], dim=1)
        V = model(inputs)

        # First derivatives
        V_t = compute_gradient(V, t)
        V_S = compute_gradient(V, S)
        V_v = compute_gradient(V, v)

        # Second derivatives
        V_SS = compute_hessian(V, S)
        V_vv = compute_hessian(V, v)

        # Mixed derivative ∂²V/∂S∂v
        V_S_flat = V_S.sum()  # Need scalar for second derivative
        V_Sv = torch.autograd.grad(
            V_S_flat,
            v,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Heston PDE
        r = self.params["r"]
        kappa = self.params["kappa"]
        theta = self.params["theta"]
        sigma_v = self.params["sigma_v"]
        rho = self.params["rho"]

        residual = (
            V_t
            + 0.5 * v * S**2 * V_SS
            + rho * sigma_v * v * S * V_Sv
            + 0.5 * sigma_v**2 * v * V_vv
            + r * S * V_S
            + kappa * (theta - v) * V_v
            - r * V
        )

        return residual.squeeze()

    def boundary_condition(
        self,
        model: nn.Module,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """Heston boundary conditions (simplified)."""
        # TODO: Implement proper BC for Heston
        return torch.zeros(points.shape[0], device=points.device)

    def initial_condition(
        self,
        model: nn.Module,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """Terminal payoff."""
        S = points[:, 1:2]
        V_pred = model(points)

        K = self.params["K"]

        if self.option_type == "call":
            V_terminal = torch.maximum(S - K, torch.zeros_like(S))
        else:
            V_terminal = torch.maximum(K - S, torch.zeros_like(S))

        return (V_pred - V_terminal).squeeze()

    def analytical_solution(
        self,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """No simple closed-form; use Monte Carlo for benchmarking."""
        return None

    def payoff(self, S: torch.Tensor) -> torch.Tensor:
        """Option payoff."""
        K = self.params["K"]
        if self.option_type == "call":
            return torch.maximum(S - K, torch.zeros_like(S))
        else:
            return torch.maximum(K - S, torch.zeros_like(S))
