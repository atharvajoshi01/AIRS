#!/usr/bin/env python3
"""
CLI for quick option pricing using different methods.

Usage:
    python scripts/price_cli.py --S 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 --type call --method bs
"""

import argparse

import torch

from dgmlib.pde.black_scholes import BlackScholesPDE
from dgmlib.utils.numerics import (
    black_scholes_analytical,
    finite_difference_european,
    monte_carlo_european,
)


def main():
    """Main pricing function."""
    parser = argparse.ArgumentParser(description="Option Pricing CLI")

    # Option parameters
    parser.add_argument("--S", type=float, required=True, help="Current stock price")
    parser.add_argument("--K", type=float, required=True, help="Strike price")
    parser.add_argument("--r", type=float, required=True, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, required=True, help="Volatility")
    parser.add_argument("--T", type=float, required=True, help="Time to maturity")
    parser.add_argument(
        "--type",
        type=str,
        choices=["call", "put"],
        default="call",
        help="Option type",
    )

    # Pricing method
    parser.add_argument(
        "--method",
        type=str,
        choices=["bs", "mc", "fd", "dgm"],
        default="bs",
        help="Pricing method",
    )

    # Method-specific options
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/bs_european/best_model.pt",
        help="Path to DGM checkpoint (for --method dgm)",
    )
    parser.add_argument(
        "--n-paths",
        type=int,
        default=100000,
        help="Number of MC paths",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Option Pricing")
    print(f"{'='*60}")
    print(f"Stock Price (S):    {args.S:.2f}")
    print(f"Strike (K):         {args.K:.2f}")
    print(f"Risk-free rate (r): {args.r:.4f}")
    print(f"Volatility (σ):     {args.sigma:.4f}")
    print(f"Maturity (T):       {args.T:.4f} years")
    print(f"Type:               {args.type}")
    print(f"Method:             {args.method.upper()}")
    print(f"{'='*60}\n")

    if args.method == "bs":
        # Analytical Black-Scholes
        price, delta, gamma = black_scholes_analytical(
            args.S, args.K, args.r, args.sigma, args.T, args.type
        )
        print(f"Price:  {price:.6f}")
        print(f"Delta:  {delta:.6f}")
        print(f"Gamma:  {gamma:.6f}")

    elif args.method == "mc":
        # Monte Carlo
        price, std_err = monte_carlo_european(
            args.S,
            args.K,
            args.r,
            args.sigma,
            args.T,
            args.type,
            n_paths=args.n_paths,
        )
        print(f"Price:     {price:.6f}")
        print(f"Std Error: {std_err:.6f}")
        print(f"95% CI:    [{price - 1.96*std_err:.6f}, {price + 1.96*std_err:.6f}]")

    elif args.method == "fd":
        # Finite Difference
        S_grid, t_grid, V_grid = finite_difference_european(
            args.K,
            args.r,
            args.sigma,
            args.T,
            args.type,
        )

        # Find closest grid point to S
        S_idx = (abs(S_grid - args.S)).argmin()
        price = V_grid[S_idx, 0]  # At t=0

        print(f"Price:  {price:.6f}")
        print(f"(Interpolated from FD grid)")

    elif args.method == "dgm":
        # Deep Galerkin Method
        from dgmlib.models.dgm import DGMNet
        from dgmlib.training.metrics import compute_greeks
        from pathlib import Path

        # Check if checkpoint exists
        if not Path(args.checkpoint).exists():
            print(f"Error: Checkpoint not found at {args.checkpoint}")
            print("Train a model first: python scripts/train.py --config dgmlib/configs/bs_european.yaml")
            return

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        # Create model with dummy bounds (will be loaded from checkpoint)
        dummy_bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
        model = DGMNet(input_dim=2, hidden_dim=50, num_layers=3, input_bounds=dummy_bounds)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Predict price
        inputs = torch.tensor([[0.0, args.S]], dtype=torch.float32)
        with torch.no_grad():
            price = model(inputs).item()

        # Compute Greeks
        greeks = compute_greeks(model, inputs)
        delta = greeks["delta"].item()
        gamma = greeks["gamma"].item()

        print(f"Price:  {price:.6f}")
        print(f"Delta:  {delta:.6f}")
        print(f"Gamma:  {gamma:.6f}")

    print()


if __name__ == "__main__":
    main()
