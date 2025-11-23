"""Training utilities for Deep Galerkin Method."""

from dgmlib.training.trainer import DGMTrainer
from dgmlib.training.metrics import compute_metrics

__all__ = ["DGMTrainer", "compute_metrics"]
