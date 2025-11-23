#!/usr/bin/env python3
"""
Comprehensive model validation against test data.

Validates DGM model performance on:
- Standard scenarios
- Stress tests
- Edge cases
- Greeks accuracy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from dgmlib.models.dgm import DGMNet
from dgmlib.training.metrics import compute_greeks

console = Console()


def load_model(checkpoint_path: str) -> DGMNet:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get model config from checkpoint or use default
    if "config" in checkpoint:
        config = checkpoint["config"]
        # Need dummy bounds for buffers to exist
        dummy_bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
        model = DGMNet(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            input_bounds=dummy_bounds,
        )
    else:
        # Default architecture with dummy bounds
        dummy_bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
        model = DGMNet(input_dim=2, hidden_dim=50, num_layers=3, input_bounds=dummy_bounds)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def validate_scenarios(
    model: DGMNet,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate model on scenario test cases."""
    predictions = []

    for _, row in df.iterrows():
        # Model input: [t=0, S]
        inputs = torch.tensor([[0.0, row["S"]]], dtype=torch.float32, requires_grad=True)

        with torch.no_grad():
            dgm_price = model(inputs).item()

        # Greeks (requires gradients)
        greeks = compute_greeks(model, inputs)
        dgm_delta = greeks["delta"].item()
        dgm_gamma = greeks["gamma"].item()

        predictions.append({
            "dgm_price": dgm_price,
            "dgm_delta": dgm_delta,
            "dgm_gamma": dgm_gamma,
            "price_error": abs(dgm_price - row["bs_price"]),
            "price_error_pct": abs(dgm_price - row["bs_price"]) / row["bs_price"] * 100,
            "delta_error": abs(dgm_delta - row["bs_delta"]),
            "gamma_error": abs(dgm_gamma - row["bs_gamma"]),
        })

    df_pred = pd.concat([df, pd.DataFrame(predictions)], axis=1)
    return df_pred


def print_validation_report(df: pd.DataFrame):
    """Print comprehensive validation report."""
    console.print("\n[bold cyan]═" * 30 + " VALIDATION REPORT " + "═" * 30)

    # Overall statistics
    table = Table(title="Overall Performance Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_column("Threshold", justify="right", style="yellow")
    table.add_column("Status", justify="center")

    metrics = [
        ("Mean Price Error ($)", df["price_error"].mean(), 0.01),
        ("Max Price Error ($)", df["price_error"].max(), 0.05),
        ("Mean Price Error (%)", df["price_error_pct"].mean(), 1.0),
        ("Max Price Error (%)", df["price_error_pct"].max(), 5.0),
        ("Mean Delta Error", df["delta_error"].mean(), 0.01),
        ("Max Delta Error", df["delta_error"].max(), 0.05),
        ("Mean Gamma Error", df["gamma_error"].mean(), 0.001),
        ("Max Gamma Error", df["gamma_error"].max(), 0.01),
    ]

    for metric_name, value, threshold in metrics:
        status = "✓ PASS" if value < threshold else "✗ FAIL"
        status_color = "green" if value < threshold else "red"
        table.add_row(
            metric_name,
            f"{value:.6f}",
            f"< {threshold}",
            f"[{status_color}]{status}[/{status_color}]",
        )

    console.print(table)

    # Performance by scenario type
    console.print("\n[bold cyan]Performance by Scenario Type")
    type_table = Table(show_header=True)
    type_table.add_column("Scenario Type", style="cyan")
    type_table.add_column("Count", justify="right")
    type_table.add_column("Mean Price Error", justify="right")
    type_table.add_column("Max Price Error", justify="right")

    for stype in df["scenario_type"].unique():
        df_type = df[df["scenario_type"] == stype]
        type_table.add_row(
            stype,
            str(len(df_type)),
            f"{df_type['price_error'].mean():.6f}",
            f"{df_type['price_error'].max():.6f}",
        )

    console.print(type_table)

    # Worst cases
    console.print("\n[bold red]Top 5 Worst Cases")
    worst = df.nlargest(5, "price_error")[
        ["scenario_name", "S", "K", "sigma", "T", "bs_price", "dgm_price", "price_error"]
    ]
    console.print(worst.to_string())

    # Best cases
    console.print("\n[bold green]Top 5 Best Cases")
    best = df.nsmallest(5, "price_error")[
        ["scenario_name", "S", "K", "sigma", "T", "bs_price", "dgm_price", "price_error"]
    ]
    console.print(best.to_string())


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate DGM model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_results",
        help="Output directory for results",
    )
    args = parser.parse_args()

    console.print("[bold blue]Loading model...")
    model = load_model(args.checkpoint)
    console.print(f"✓ Model loaded: {model.count_parameters():,} parameters")

    # Load test data
    console.print("\n[bold blue]Loading test data...")
    test_dir = Path(args.test_data)
    df_scenarios = pd.read_csv(test_dir / "scenario_ground_truth.csv")
    console.print(f"✓ Loaded {len(df_scenarios)} test scenarios")

    # Run validation
    console.print("\n[bold blue]Running validation...")
    df_results = validate_scenarios(model, df_scenarios)

    # Print report
    print_validation_report(df_results)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_dir / "validation_results.csv", index=False)

    # Save summary
    summary = {
        "total_scenarios": len(df_results),
        "mean_price_error": float(df_results["price_error"].mean()),
        "max_price_error": float(df_results["price_error"].max()),
        "mean_price_error_pct": float(df_results["price_error_pct"].mean()),
        "mean_delta_error": float(df_results["delta_error"].mean()),
        "mean_gamma_error": float(df_results["gamma_error"].mean()),
        "pass_threshold": float(df_results["price_error"].mean()) < 0.01,
    }

    with open(output_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n✓ Results saved to {output_dir}/")

    # Return exit code based on validation
    if summary["pass_threshold"]:
        console.print("\n[bold green]✓ VALIDATION PASSED[/bold green]")
        return 0
    else:
        console.print("\n[bold red]✗ VALIDATION FAILED[/bold red]")
        return 1


if __name__ == "__main__":
    exit(main())
