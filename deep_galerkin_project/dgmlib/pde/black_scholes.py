"""
Black-Scholes PDE for European options.

PDE: ∂V/∂t + 0.5 * σ² * S² * ∂²V/∂S² + r * S * ∂V/∂S - r * V = 0
Terminal condition: V(T, S) = payoff(S)
Boundary conditions:
    - Call: V(t, 0) = 0, V(t, S_max) ≈ S - K*exp(-r*(T-t))
    - Put: V(t, 0) = K*exp(-r*(T-t)), V(t, S_max) = 0
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from scipy.stats import norm

from dgmlib.pde.base_pde import BasePDE
from dgmlib.utils.autodiff import compute_gradient, compute_hessian


class BlackScholesPDE(BasePDE):
    """European option pricing under Black-Scholes model."""

    def __init__(
        self,
        r: float,
        sigma: float,
        K: float,
        T: float,
        option_type: str = "call",
        S_min: float = 0.0,
        S_max: float = 200.0,
    ) -> None:
        """
        Initialize Black-Scholes PDE.

        Args:
            r: Risk-free interest rate.
            sigma: Volatility.
            K: Strike price.
            T: Time to maturity.
            option_type: "call" or "put".
            S_min: Minimum stock price in domain.
            S_max: Maximum stock price in domain.
        """
        # IMPORTANT: domain order must match input order [t, S] for model
        domain = {"t": (0.0, T), "S": (S_min, S_max)}
        params = {"r": r, "sigma": sigma, "K": K, "T": T}
        super().__init__(domain, params)

        if option_type not in ["call", "put"]:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
        self.option_type = option_type

    def _validate_domain(self) -> None:
        """Validate domain variables."""
        required = {"S", "t"}
        if not required.issubset(self.domain.keys()):
            raise ValueError(f"Domain must contain {required}, got {self.domain.keys()}")

    def _validate_params(self) -> None:
        """Validate PDE parameters."""
        required = {"r", "sigma", "K", "T"}
        if not required.issubset(self.params.keys()):
            raise ValueError(f"Params must contain {required}, got {self.params.keys()}")

        if self.params["sigma"] <= 0:
            raise ValueError("sigma must be positive")
        if self.params["K"] <= 0:
            raise ValueError("K must be positive")
        if self.params["T"] <= 0:
            raise ValueError("T must be positive")

    def pde_residual(
        self,
        model: nn.Module,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Black-Scholes PDE residual.

        Args:
            points: Tensor of shape (N, 2) with columns [t, S].

        Returns:
            Residual L[V] of shape (N,).
        """
        t = points[:, 0:1]
        S = points[:, 1:2]

        # Require gradients
        t.requires_grad_(True)
        S.requires_grad_(True)

        # Forward pass
        V = model(torch.cat([t, S], dim=1))

        # Compute derivatives
        V_t = compute_gradient(V, t)
        V_S = compute_gradient(V, S)
        V_SS = compute_hessian(V, S)

        # Black-Scholes PDE: V_t + 0.5 * σ² * S² * V_SS + r * S * V_S - r * V = 0
        sigma = self.params["sigma"]
        r = self.params["r"]

        residual = (
            V_t
            + 0.5 * sigma**2 * S**2 * V_SS
            + r * S * V_S
            - r * V
        )

        return residual.squeeze()

    def boundary_condition(
        self,
        model: nn.Module,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce boundary conditions at S_min and S_max.

        Args:
            points: Boundary points of shape (N, 2).

        Returns:
            BC residual of shape (N,).
        """
        t = points[:, 0:1]
        S = points[:, 1:2]

        V_pred = model(torch.cat([t, S], dim=1))

        T = self.params["T"]
        K = self.params["K"]
        r = self.params["r"]
        tau = T - t  # Time to maturity

        # At S = 0
        S_min = self.domain["S"][0]
        mask_lower = (S.abs() < 1e-3)

        if self.option_type == "call":
            # Call at S=0 should be 0
            V_lower = torch.zeros_like(V_pred)
        else:
            # Put at S=0 should be K * exp(-r * tau)
            V_lower = K * torch.exp(-r * tau)

        # At S = S_max (large S)
        S_max = self.domain["S"][1]
        mask_upper = (S >= S_max * 0.99)

        if self.option_type == "call":
            # Call at large S: V ≈ S - K * exp(-r * tau)
            V_upper = S - K * torch.exp(-r * tau)
        else:
            # Put at large S: V ≈ 0
            V_upper = torch.zeros_like(V_pred)

        # Combine
        residual = torch.zeros_like(V_pred.squeeze())
        residual = torch.where(
            mask_lower.squeeze(),
            (V_pred - V_lower).squeeze(),
            residual
        )
        residual = torch.where(
            mask_upper.squeeze(),
            (V_pred - V_upper).squeeze(),
            residual
        )

        return residual

    def initial_condition(
        self,
        model: nn.Module,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Terminal condition: V(T, S) = payoff(S).

        Args:
            points: Terminal points of shape (N, 2) with t = T.

        Returns:
            IC residual of shape (N,).
        """
        S = points[:, 1:2]
        V_pred = model(points)

        K = self.params["K"]

        if self.option_type == "call":
            V_terminal = torch.maximum(S - K, torch.zeros_like(S))
        else:
            V_terminal = torch.maximum(K - S, torch.zeros_like(S))

        residual = V_pred - V_terminal
        return residual.squeeze()

    def analytical_solution(
        self,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Black-Scholes analytical formula for European options.

        Args:
            points: Evaluation points of shape (N, 2) with columns [t, S].

        Returns:
            Analytical option values of shape (N,).
        """
        t = points[:, 0]
        S = points[:, 1]

        T = self.params["T"]
        K = self.params["K"]
        r = self.params["r"]
        sigma = self.params["sigma"]

        tau = T - t  # Time to maturity

        # Avoid division by zero at maturity
        tau = torch.clamp(tau, min=1e-10)

        d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))
        d2 = d1 - sigma * torch.sqrt(tau)

        # Use scipy for standard normal CDF (more accurate)
        d1_np = d1.detach().cpu().numpy()
        d2_np = d2.detach().cpu().numpy()

        N_d1 = torch.tensor(norm.cdf(d1_np), dtype=torch.float32, device=points.device)
        N_d2 = torch.tensor(norm.cdf(d2_np), dtype=torch.float32, device=points.device)

        if self.option_type == "call":
            value = S * N_d1 - K * torch.exp(-r * tau) * N_d2
        else:
            N_neg_d1 = torch.tensor(norm.cdf(-d1_np), dtype=torch.float32, device=points.device)
            N_neg_d2 = torch.tensor(norm.cdf(-d2_np), dtype=torch.float32, device=points.device)
            value = K * torch.exp(-r * tau) * N_neg_d2 - S * N_neg_d1

        return value

    def payoff(self, S: torch.Tensor) -> torch.Tensor:
        """Compute option payoff."""
        K = self.params["K"]
        if self.option_type == "call":
            return torch.maximum(S - K, torch.zeros_like(S))
        else:
            return torch.maximum(K - S, torch.zeros_like(S))
