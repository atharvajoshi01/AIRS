#!/usr/bin/env python3
"""
Evaluation script for trained DGM models.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/bs_european/best_model.pt --config dgmlib/configs/bs_european.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from dgmlib.pde.black_scholes import BlackScholesPDE
from dgmlib.sampling.sobol import SobolSampler
from dgmlib.training.metrics import compute_greeks, compute_metrics
from dgmlib.utils.config import load_config
from dgmlib.utils.plots import (
    plot_greeks,
    plot_option_surface,
    plot_price_comparison,
)

from scripts.train import create_model, create_pde


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate DGM Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for evaluation",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded configuration: {config.experiment_name}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PDE
    pde = create_pde(config)

    # Load model
    model = create_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()
    print("Model loaded successfully")

    # Create evaluation grid
    S_min, S_max = pde.get_domain_bounds("S")
    t_min, t_max = pde.get_domain_bounds("t")

    S_grid = np.linspace(S_min + 1, S_max - 1, config.evaluation.S_grid_size)
    t_grid = np.linspace(t_min, t_max, config.evaluation.t_grid_size)

    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing="ij")

    # Evaluate DGM
    print("Evaluating DGM model...")
    V_dgm = np.zeros_like(S_mesh)

    with torch.no_grad():
        for i, S_val in enumerate(S_grid):
            for j, t_val in enumerate(t_grid):
                point = torch.tensor(
                    [[t_val, S_val]],
                    dtype=torch.float32,
                    device=args.device,
                )
                V_dgm[i, j] = model(point).item()

    # Compare with analytical if available
    if hasattr(pde, "analytical_solution") and pde.analytical_solution is not None:
        print("Computing analytical solution...")
        V_analytical = np.zeros_like(S_mesh)

        for i, S_val in enumerate(S_grid):
            for j, t_val in enumerate(t_grid):
                point = torch.tensor([[t_val, S_val]], dtype=torch.float32)
                V_analytical[i, j] = pde.analytical_solution(point).item()

        # Compute error
        abs_error = np.abs(V_dgm - V_analytical)
        rel_error = abs_error / (np.abs(V_analytical) + 1e-8)

        print(f"\nAbsolute Error Statistics:")
        print(f"  Mean: {np.mean(abs_error):.6f}")
        print(f"  Max:  {np.max(abs_error):.6f}")
        print(f"  RMSE: {np.sqrt(np.mean(abs_error**2)):.6f}")

        print(f"\nRelative Error Statistics:")
        print(f"  Mean: {np.mean(rel_error)*100:.2f}%")
        print(f"  Max:  {np.max(rel_error)*100:.2f}%")

        # Plot comparison at t=0
        t0_idx = 0
        plot_price_comparison(
            S_grid,
            {
                "DGM": V_dgm[:, t0_idx],
                "Analytical": V_analytical[:, t0_idx],
            },
            title=f"{config.experiment_name} - Price Comparison at t=0",
            save_path=str(output_dir / "price_comparison.png"),
        )

    # Plot option surface
    plot_option_surface(
        S_grid,
        t_grid,
        V_dgm,
        title=f"{config.experiment_name} - Option Value Surface",
        save_path=str(output_dir / "option_surface.png"),
    )

    # Compute and plot Greeks
    if config.evaluation.compute_greeks:
        print("\nComputing Greeks...")
        t0_points = torch.tensor(
            [[0.0, S] for S in S_grid],
            dtype=torch.float32,
            device=args.device,
        )

        greeks = compute_greeks(model, t0_points)
        delta = greeks["delta"].cpu().numpy().squeeze()
        gamma = greeks["gamma"].cpu().numpy().squeeze()

        print(f"Delta range: [{delta.min():.4f}, {delta.max():.4f}]")
        print(f"Gamma range: [{gamma.min():.6f}, {gamma.max():.6f}]")

        plot_greeks(
            S_grid,
            delta,
            gamma,
            title=f"{config.experiment_name} - Greeks at t=0",
            save_path=str(output_dir / "greeks.png"),
        )

    # Compute metrics on test set
    print("\nComputing test set metrics...")
    sampler = SobolSampler(pde.domain, seed=999)
    test_points = sampler.sample(1000, device=args.device)

    test_values = None
    if hasattr(pde, "analytical_solution") and pde.analytical_solution is not None:
        test_values = pde.analytical_solution(test_points)

    metrics = compute_metrics(model, pde, test_points, test_values)

    print(f"\nTest Set Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    # Save metrics
    import json

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
