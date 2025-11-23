"""PDE residual loss computation."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from dgmlib.pde.base_pde import BasePDE


class PDELoss(nn.Module):
    """
    Combined loss for Deep Galerkin Method training.

    Loss = λ_pde * L_pde + λ_bc * L_bc + λ_ic * L_ic + λ_data * L_data
    """

    def __init__(
        self,
        pde: BasePDE,
        lambda_pde: float = 1.0,
        lambda_bc: float = 1.0,
        lambda_ic: float = 10.0,
        lambda_data: float = 0.0,
    ) -> None:
        """
        Initialize PDE loss.

        Args:
            pde: PDE object defining residuals and conditions.
            lambda_pde: Weight for PDE residual loss.
            lambda_bc: Weight for boundary condition loss.
            lambda_ic: Weight for initial/terminal condition loss.
            lambda_data: Weight for data matching loss (if available).
        """
        super().__init__()
        self.pde = pde
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.lambda_data = lambda_data

    def forward(
        self,
        model: nn.Module,
        interior_points: torch.Tensor,
        boundary_points: torch.Tensor,
        initial_points: torch.Tensor,
        data_points: torch.Tensor = None,
        data_values: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Args:
            model: Neural network model.
            interior_points: Interior collocation points for PDE residual.
            boundary_points: Boundary points for BC.
            initial_points: Initial/terminal condition points.
            data_points: Optional data points for supervised loss.
            data_values: Optional data values.

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses.
        """
        loss_dict = {}

        # PDE residual loss
        if interior_points is not None and len(interior_points) > 0:
            pde_residual = self.pde.pde_residual(model, interior_points)
            loss_pde = torch.mean(pde_residual**2)
            loss_dict["pde"] = loss_pde.item()
        else:
            loss_pde = torch.tensor(0.0, device=model.parameters().__next__().device)
            loss_dict["pde"] = 0.0

        # Boundary condition loss
        if boundary_points is not None and len(boundary_points) > 0:
            bc_residual = self.pde.boundary_condition(model, boundary_points)
            loss_bc = torch.mean(bc_residual**2)
            loss_dict["bc"] = loss_bc.item()
        else:
            loss_bc = torch.tensor(0.0, device=model.parameters().__next__().device)
            loss_dict["bc"] = 0.0

        # Initial/terminal condition loss
        if initial_points is not None and len(initial_points) > 0:
            ic_residual = self.pde.initial_condition(model, initial_points)
            loss_ic = torch.mean(ic_residual**2)
            loss_dict["ic"] = loss_ic.item()
        else:
            loss_ic = torch.tensor(0.0, device=model.parameters().__next__().device)
            loss_dict["ic"] = 0.0

        # Data matching loss (optional)
        if data_points is not None and data_values is not None:
            pred_values = model(data_points)
            loss_data = torch.mean((pred_values.squeeze() - data_values)**2)
            loss_dict["data"] = loss_data.item()
        else:
            loss_data = torch.tensor(0.0, device=model.parameters().__next__().device)
            loss_dict["data"] = 0.0

        # Combined loss
        total_loss = (
            self.lambda_pde * loss_pde
            + self.lambda_bc * loss_bc
            + self.lambda_ic * loss_ic
            + self.lambda_data * loss_data
        )

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict
