"""Training callbacks for checkpointing, early stopping, and learning rate scheduling."""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-6,
        mode: str = "min",
    ) -> None:
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: "min" or "max" for metric.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def __call__(self, metric_value: float) -> bool:
        """
        Check if training should stop.

        Args:
            metric_value: Current metric value.

        Returns:
            True if should stop.
        """
        if self.best_value is None:
            self.best_value = metric_value
            return False

        if self.mode == "min":
            improved = metric_value < (self.best_value - self.min_delta)
        else:
            improved = metric_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    """Save model checkpoints."""

    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
    ) -> None:
        """
        Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints.
            monitor: Metric to monitor.
            mode: "min" or "max".
            save_best_only: Only save when metric improves.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value: Optional[float] = None

    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """
        Save checkpoint if criteria met.

        Args:
            model: Model to save.
            optimizer: Optimizer state.
            epoch: Current epoch.
            metrics: Dictionary of metrics.
        """
        metric_value = metrics.get(self.monitor, None)

        if metric_value is None:
            return

        should_save = False

        if not self.save_best_only:
            should_save = True
        else:
            if self.best_value is None:
                should_save = True
                self.best_value = metric_value
            else:
                if self.mode == "min" and metric_value < self.best_value:
                    should_save = True
                    self.best_value = metric_value
                elif self.mode == "max" and metric_value > self.best_value:
                    should_save = True
                    self.best_value = metric_value

        if should_save:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            }

            checkpoint_path = self.checkpoint_dir / f"best_model.pt"
            torch.save(checkpoint, checkpoint_path)


class LRSchedulerCallback:
    """Learning rate scheduler callback."""

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        monitor: Optional[str] = None,
    ) -> None:
        """
        Initialize LR scheduler callback.

        Args:
            scheduler: PyTorch LR scheduler.
            monitor: Optional metric to monitor (for ReduceLROnPlateau).
        """
        self.scheduler = scheduler
        self.monitor = monitor

    def __call__(self, metrics: Dict[str, float]) -> None:
        """
        Step the scheduler.

        Args:
            metrics: Dictionary of metrics.
        """
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if self.monitor is not None:
                metric_value = metrics.get(self.monitor)
                if metric_value is not None:
                    self.scheduler.step(metric_value)
        else:
            self.scheduler.step()
