"""Sobol quasi-random sequence sampling."""

from typing import Dict, Tuple

import torch
from scipy.stats import qmc


class SobolSampler:
    """
    Sobol sequence sampler for low-discrepancy quasi-random points.

    Sobol sequences provide better coverage of the domain than random sampling.
    """

    def __init__(
        self,
        domain: Dict[str, Tuple[float, float]],
        seed: int = 42,
    ) -> None:
        """
        Initialize Sobol sampler.

        Args:
            domain: Dictionary mapping variable names to (min, max) tuples.
            seed: Random seed for reproducibility.
        """
        self.domain = domain
        self.dim = len(domain)
        self.var_names = list(domain.keys())
        self.seed = seed

        # Create Sobol engine
        self.engine = qmc.Sobol(d=self.dim, scramble=True, seed=seed)

    def sample(self, n_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Generate Sobol samples.

        Args:
            n_samples: Number of samples to generate.
            device: Device to place samples on.

        Returns:
            Tensor of shape (n_samples, dim) with samples in [0, 1]^d,
            scaled to domain bounds.
        """
        # Generate Sobol samples in [0, 1]^d
        # Round up to next power of 2 for Sobol
        m = 2 ** ((n_samples - 1).bit_length())
        samples_unit = self.engine.random(m)[:n_samples]

        # Scale to domain
        samples_scaled = torch.zeros(n_samples, self.dim, dtype=torch.float32)

        for i, var_name in enumerate(self.var_names):
            low, high = self.domain[var_name]
            samples_scaled[:, i] = torch.tensor(
                qmc.scale(samples_unit[:, i:i+1], [low], [high]).squeeze(),
                dtype=torch.float32
            )

        return samples_scaled.to(device)

    def sample_boundary(
        self,
        n_samples: int,
        boundary_var: str,
        boundary_value: float,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Sample points on a boundary.

        Args:
            n_samples: Number of boundary samples.
            boundary_var: Variable to fix at boundary.
            boundary_value: Value to fix boundary variable at.
            device: Device for samples.

        Returns:
            Tensor of shape (n_samples, dim) with boundary_var fixed.
        """
        # Sample other variables
        other_vars = [v for v in self.var_names if v != boundary_var]
        other_domain = {v: self.domain[v] for v in other_vars}

        # Create temporary sampler for other dimensions
        temp_sampler = SobolSampler(other_domain, seed=self.seed)
        other_samples = temp_sampler.sample(n_samples, device=device)

        # Build full sample with boundary fixed
        samples = torch.zeros(n_samples, self.dim, dtype=torch.float32, device=device)

        boundary_idx = self.var_names.index(boundary_var)
        other_idx = [i for i in range(self.dim) if i != boundary_idx]

        samples[:, boundary_idx] = boundary_value
        samples[:, other_idx] = other_samples

        return samples
