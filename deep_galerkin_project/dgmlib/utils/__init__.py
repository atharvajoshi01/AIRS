"""Utility modules for Deep Galerkin Method."""

from dgmlib.utils.autodiff import compute_gradient, compute_hessian
from dgmlib.utils.seeds import set_seed

__all__ = ["compute_gradient", "compute_hessian", "set_seed"]
