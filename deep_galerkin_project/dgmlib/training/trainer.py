"""Deep Galerkin Method trainer."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import Progress
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from dgmlib.loss.residuals import PDELoss
from dgmlib.pde.base_pde import BasePDE
from dgmlib.training.callbacks import EarlyStopping, LRSchedulerCallback, ModelCheckpoint
from dgmlib.training.metrics import compute_metrics

logger = logging.getLogger(__name__)
console = Console()


class DGMTrainer:
    """Trainer for Deep Galerkin Method."""

    def __init__(
        self,
        model: nn.Module,
        pde: BasePDE,
        loss_fn: PDELoss,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        use_amp: bool = False,
        gradient_clip: Optional[float] = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Neural network model.
            pde: PDE definition.
            loss_fn: Loss function.
            optimizer: Optimizer.
            device: Device to train on.
            use_amp: Use automatic mixed precision.
            gradient_clip: Gradient clipping value.
            checkpoint_dir: Directory for checkpoints.
            log_dir: Directory for logs.
        """
        self.model = model.to(device)
        self.pde = pde
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip

        self.scaler = GradScaler() if use_amp else None
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "loss": [],
            "pde_loss": [],
            "bc_loss": [],
            "ic_loss": [],
        }

    def train_step(
        self,
        interior_points: torch.Tensor,
        boundary_points: torch.Tensor,
        initial_points: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            interior_points: Interior collocation points.
            boundary_points: Boundary points.
            initial_points: Initial condition points.

        Returns:
            Dictionary of losses.
        """
        self.model.train()
        self.optimizer.zero_grad()

        if self.use_amp:
            with autocast():
                loss, loss_dict = self.loss_fn(
                    self.model,
                    interior_points,
                    boundary_points,
                    initial_points,
                )

            self.scaler.scale(loss).backward()

            if self.gradient_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, loss_dict = self.loss_fn(
                self.model,
                interior_points,
                boundary_points,
                initial_points,
            )

            loss.backward()

            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.optimizer.step()

        return loss_dict

    def train(
        self,
        sampler,
        n_interior: int,
        n_boundary: int,
        n_initial: int,
        epochs: int,
        val_points: Optional[torch.Tensor] = None,
        val_values: Optional[torch.Tensor] = None,
        callbacks: Optional[List] = None,
        log_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            sampler: Sampler for generating collocation points.
            n_interior: Number of interior points per epoch.
            n_boundary: Number of boundary points per epoch.
            n_initial: Number of initial condition points per epoch.
            epochs: Number of training epochs.
            val_points: Optional validation points.
            val_values: Optional validation values.
            callbacks: Optional list of callbacks.
            log_interval: Logging frequency.

        Returns:
            Training history.
        """
        console.print(f"[bold blue]Starting training for {epochs} epochs...[/bold blue]")
        console.print(f"Model parameters: {self.model.count_parameters():,}")

        callbacks = callbacks or []

        with Progress() as progress:
            task = progress.add_task("[cyan]Training...", total=epochs)

            for epoch in range(epochs):
                # Sample collocation points
                interior_points = sampler.sample(n_interior, device=self.device)

                # Sample boundary points
                S_min, S_max = self.pde.get_domain_bounds("S")
                t_min, t_max = self.pde.get_domain_bounds("t")

                # Boundary at S_min and S_max
                boundary_points = self._sample_boundary_points(
                    sampler, n_boundary, S_min, S_max, t_min, t_max
                )

                # Initial/terminal condition points (at t = T)
                initial_points = sampler.sample_boundary(
                    n_initial,
                    "t",
                    t_max,
                    device=self.device,
                )

                # Training step
                loss_dict = self.train_step(
                    interior_points,
                    boundary_points,
                    initial_points,
                )

                # Record history
                self.history["loss"].append(loss_dict["total"])
                self.history["pde_loss"].append(loss_dict["pde"])
                self.history["bc_loss"].append(loss_dict["bc"])
                self.history["ic_loss"].append(loss_dict["ic"])

                # Validation
                if val_points is not None:
                    val_metrics = compute_metrics(
                        self.model,
                        self.pde,
                        val_points,
                        val_values,
                    )
                    val_metrics["val_loss"] = loss_dict["total"]
                else:
                    val_metrics = {}

                # Logging
                if (epoch + 1) % log_interval == 0:
                    log_str = f"Epoch {epoch + 1}/{epochs} - Loss: {loss_dict['total']:.6f}"
                    if val_metrics:
                        log_str += f" - Val MAE: {val_metrics.get('mae', 0):.6f}"
                    console.print(log_str)

                # TensorBoard
                self.writer.add_scalar("Loss/total", loss_dict["total"], epoch)
                self.writer.add_scalar("Loss/pde", loss_dict["pde"], epoch)
                self.writer.add_scalar("Loss/bc", loss_dict["bc"], epoch)
                self.writer.add_scalar("Loss/ic", loss_dict["ic"], epoch)

                for key, value in val_metrics.items():
                    self.writer.add_scalar(f"Validation/{key}", value, epoch)

                # Callbacks
                for callback in callbacks:
                    if isinstance(callback, (EarlyStopping,)):
                        if callback(loss_dict["total"]):
                            console.print("[yellow]Early stopping triggered[/yellow]")
                            break
                    elif isinstance(callback, ModelCheckpoint):
                        callback(self.model, self.optimizer, epoch, val_metrics)
                    elif isinstance(callback, LRSchedulerCallback):
                        callback(val_metrics if val_metrics else loss_dict)

                progress.update(task, advance=1)

        console.print("[bold green]Training complete![/bold green]")
        return self.history

    def _sample_boundary_points(
        self,
        sampler,
        n_boundary: int,
        S_min: float,
        S_max: float,
        t_min: float,
        t_max: float,
    ) -> torch.Tensor:
        """Sample points on spatial boundaries."""
        n_per_boundary = n_boundary // 2

        # Lower boundary (S = S_min)
        lower_points = sampler.sample_boundary(
            n_per_boundary,
            "S",
            S_min,
            device=self.device,
        )

        # Upper boundary (S = S_max)
        upper_points = sampler.sample_boundary(
            n_per_boundary,
            "S",
            S_max,
            device=self.device,
        )

        return torch.cat([lower_points, upper_points], dim=0)

    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            **kwargs,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint
