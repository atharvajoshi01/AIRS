"""Latin Hypercube Sampling for collocation points."""

from typing import Dict, Tuple

import torch
from scipy.stats import qmc


class LatinHypercubeSampler:
    """
    Latin Hypercube Sampling (LHS) for generating well-distributed points.

    LHS ensures better stratification than pure random sampling.
    """

    def __init__(
        self,
        domain: Dict[str, Tuple[float, float]],
        seed: int = 42,
    ) -> None:
        """
        Initialize LHS sampler.

        Args:
            domain: Dictionary mapping variable names to (min, max) tuples.
            seed: Random seed.
        """
        self.domain = domain
        self.dim = len(domain)
        self.var_names = list(domain.keys())
        self.seed = seed

    def sample(self, n_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Generate Latin Hypercube samples.

        Args:
            n_samples: Number of samples.
            device: Device to place samples on.

        Returns:
            Tensor of shape (n_samples, dim).
        """
        # Create LHS engine
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.seed)
        samples_unit = sampler.random(n=n_samples)

        # Scale to domain
        samples_scaled = torch.zeros(n_samples, self.dim, dtype=torch.float32)

        for i, var_name in enumerate(self.var_names):
            low, high = self.domain[var_name]
            samples_scaled[:, i] = torch.tensor(
                qmc.scale(samples_unit[:, i:i+1], [low], [high]).squeeze(),
                dtype=torch.float32
            )

        return samples_scaled.to(device)
