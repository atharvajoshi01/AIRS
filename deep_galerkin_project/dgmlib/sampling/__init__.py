"""Sampling strategies for collocation points."""

from dgmlib.sampling.latin_hypercube import LatinHypercubeSampler
from dgmlib.sampling.sobol import SobolSampler

__all__ = ["LatinHypercubeSampler", "SobolSampler"]
