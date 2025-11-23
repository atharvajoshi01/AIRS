"""
Base PDE interface for physics-informed neural network training.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn as nn


class BasePDE(ABC):
    """
    Abstract base class for partial differential equations.

    Defines the interface for PDE residual computation, boundary conditions,
    and initial/terminal conditions used in Deep Galerkin Method training.
    """

    def __init__(
        self,
        domain: Dict[str, Tuple[float, float]],
        params: Dict[str, float],
    ) -> None:
        """
        Initialize PDE with domain and parameters.

        Args:
            domain: Dictionary mapping variable names to (min, max) tuples.
                   e.g., {"S": (0.0, 200.0), "t": (0.0, 1.0)}
            params: Dictionary of PDE parameters.
                   e.g., {"r": 0.05, "sigma": 0.2, "K": 100.0}
        """
        self.domain = domain
        self.params = params
        self._validate_domain()
        self._validate_params()

    @abstractmethod
    def _validate_domain(self) -> None:
        """Validate that required domain variables are present."""
        pass

    @abstractmethod
    def _validate_params(self) -> None:
        """Validate that required PDE parameters are present."""
        pass

    @abstractmethod
    def pde_residual(
        self,
        model: nn.Module,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PDE residual L[u] = 0 at given points.

        Args:
            model: Neural network approximating the solution.
            points: Tensor of shape (N, d) where d is spatial + temporal dims.

        Returns:
            Residual tensor of shape (N,).
        """
        pass

    @abstractmethod
    def boundary_condition(
        self,
        model: nn.Module,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute boundary condition residual.

        Args:
            model: Neural network approximating the solution.
            points: Boundary points tensor of shape (N, d).

        Returns:
            BC residual tensor of shape (N,).
        """
        pass

    @abstractmethod
    def initial_condition(
        self,
        model: nn.Module,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute initial/terminal condition residual.

        Args:
            model: Neural network approximating the solution.
            points: IC points tensor of shape (N, d).

        Returns:
            IC residual tensor of shape (N,).
        """
        pass

    @abstractmethod
    def analytical_solution(
        self,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Analytical solution if available (for benchmarking).

        Args:
            points: Evaluation points of shape (N, d).

        Returns:
            Exact solution values of shape (N,).
            Returns None if no analytical solution exists.
        """
        pass

    def get_domain_bounds(self, var_name: str) -> Tuple[float, float]:
        """Get domain bounds for a variable."""
        return self.domain[var_name]

    def get_param(self, param_name: str) -> float:
        """Get a PDE parameter value."""
        return self.params[param_name]
