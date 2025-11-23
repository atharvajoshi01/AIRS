"""Curriculum learning strategies for progressive refinement."""

from typing import Dict, Tuple

import torch


class CurriculumSampler:
    """
    Curriculum learning sampler that progressively refines sampling.

    Starts with coarse sampling and progressively adds finer samples
    in regions with high residual.
    """

    def __init__(
        self,
        domain: Dict[str, Tuple[float, float]],
        base_sampler,
        refinement_schedule: str = "linear",
    ) -> None:
        """
        Initialize curriculum sampler.

        Args:
            domain: Domain bounds.
            base_sampler: Base sampler (e.g., SobolSampler).
            refinement_schedule: Schedule for refinement ("linear", "exponential").
        """
        self.domain = domain
        self.base_sampler = base_sampler
        self.refinement_schedule = refinement_schedule
        self.current_epoch = 0

    def sample(
        self,
        n_samples: int,
        residuals: torch.Tensor = None,
        points: torch.Tensor = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Sample with curriculum strategy.

        Args:
            n_samples: Number of samples.
            residuals: Optional residuals from previous iteration.
            points: Optional points corresponding to residuals.
            device: Device.

        Returns:
            Sampled points.
        """
        # Base samples
        base_samples = self.base_sampler.sample(n_samples, device=device)

        # If we have residuals, add importance sampling
        if residuals is not None and points is not None:
            n_importance = max(1, n_samples // 4)  # 25% importance samples

            # Find high-residual regions
            k = min(len(residuals), n_importance * 2)
            _, top_indices = torch.topk(residuals.abs(), k)

            # Add noise around high-residual points
            high_res_points = points[top_indices[:n_importance]]
            noise_scale = 0.1 * self._get_refinement_factor()

            noise = torch.randn_like(high_res_points) * noise_scale
            importance_samples = high_res_points + noise

            # Combine
            combined = torch.cat([base_samples[:-n_importance], importance_samples], dim=0)
            return combined

        return base_samples

    def _get_refinement_factor(self) -> float:
        """Get current refinement factor based on schedule."""
        if self.refinement_schedule == "linear":
            return max(0.01, 1.0 - 0.01 * self.current_epoch)
        elif self.refinement_schedule == "exponential":
            return max(0.01, 0.9 ** self.current_epoch)
        else:
            return 0.1

    def step(self) -> None:
        """Advance curriculum epoch."""
        self.current_epoch += 1
