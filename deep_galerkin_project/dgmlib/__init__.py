"""
Deep Galerkin Model Library for PDE-based Option Pricing

A production-grade implementation of Deep Galerkin Methods (physics-informed neural networks)
for solving partial differential equations in quantitative finance.
"""

__version__ = "0.1.0"
__author__ = "Quant ML Engineering Team"

from dgmlib.models.dgm import DGMNet
from dgmlib.pde.black_scholes import BlackScholesPDE

__all__ = ["DGMNet", "BlackScholesPDE", "__version__"]
