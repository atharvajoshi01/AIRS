#!/usr/bin/env python3
"""
Generate synthetic test data for validating the DGM pricing model.

This script creates realistic option pricing scenarios for testing:
- Various market conditions (bull, bear, sideways)
- Different volatility regimes
- Multiple maturities and strikes
- Edge cases and stress scenarios
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from dgmlib.utils.numerics import black_scholes_analytical, monte_carlo_european


def generate_market_scenarios() -> List[Dict]:
    """Generate diverse market scenarios for testing."""
    scenarios = []

    # 1. Standard ATM scenarios
    for vol in [0.1, 0.2, 0.3, 0.4]:
        for T in [0.25, 0.5, 1.0, 2.0]:
            scenarios.append({
                "name": f"ATM_vol{int(vol*100)}_T{T}",
                "S": 100.0,
                "K": 100.0,
                "r": 0.05,
                "sigma": vol,
                "T": T,
                "option_type": "call",
                "scenario_type": "standard",
            })

    # 2. ITM scenarios
    for moneyness in [0.8, 0.9]:
        scenarios.append({
            "name": f"ITM_K{int(moneyness*100)}",
            "S": 100.0,
            "K": moneyness * 100.0,
            "r": 0.05,
            "sigma": 0.2,
            "T": 1.0,
            "option_type": "call",
            "scenario_type": "itm",
        })

    # 3. OTM scenarios
    for moneyness in [1.1, 1.2]:
        scenarios.append({
            "name": f"OTM_K{int(moneyness*100)}",
            "S": 100.0,
            "K": moneyness * 100.0,
            "r": 0.05,
            "sigma": 0.2,
            "T": 1.0,
            "option_type": "call",
            "scenario_type": "otm",
        })

    # 4. Put options
    for option_type in ["put"]:
        scenarios.append({
            "name": f"ATM_{option_type}",
            "S": 100.0,
            "K": 100.0,
            "r": 0.05,
            "sigma": 0.2,
            "T": 1.0,
            "option_type": option_type,
            "scenario_type": "standard",
        })

    # 5. Extreme scenarios
    scenarios.extend([
        {
            "name": "deep_ITM",
            "S": 100.0,
            "K": 50.0,
            "r": 0.05,
            "sigma": 0.2,
            "T": 1.0,
            "option_type": "call",
            "scenario_type": "extreme",
        },
        {
            "name": "deep_OTM",
            "S": 100.0,
            "K": 150.0,
            "r": 0.05,
            "sigma": 0.2,
            "T": 1.0,
            "option_type": "call",
            "scenario_type": "extreme",
        },
        {
            "name": "high_vol",
            "S": 100.0,
            "K": 100.0,
            "r": 0.05,
            "sigma": 0.8,
            "T": 1.0,
            "option_type": "call",
            "scenario_type": "stress",
        },
        {
            "name": "low_vol",
            "S": 100.0,
            "K": 100.0,
            "r": 0.05,
            "sigma": 0.05,
            "T": 1.0,
            "option_type": "call",
            "scenario_type": "stress",
        },
        {
            "name": "near_maturity",
            "S": 100.0,
            "K": 100.0,
            "r": 0.05,
            "sigma": 0.2,
            "T": 0.01,
            "option_type": "call",
            "scenario_type": "stress",
        },
    ])

    return scenarios


def compute_ground_truth(scenarios: List[Dict]) -> pd.DataFrame:
    """Compute ground truth prices using analytical and MC methods."""
    results = []

    for scenario in scenarios:
        # Analytical Black-Scholes
        bs_price, bs_delta, bs_gamma = black_scholes_analytical(
            scenario["S"],
            scenario["K"],
            scenario["r"],
            scenario["sigma"],
            scenario["T"],
            scenario["option_type"],
        )

        # Monte Carlo (for validation)
        mc_price, mc_std = monte_carlo_european(
            scenario["S"],
            scenario["K"],
            scenario["r"],
            scenario["sigma"],
            scenario["T"],
            scenario["option_type"],
            n_paths=100000,
        )

        results.append({
            "scenario_name": scenario["name"],
            "S": scenario["S"],
            "K": scenario["K"],
            "r": scenario["r"],
            "sigma": scenario["sigma"],
            "T": scenario["T"],
            "option_type": scenario["option_type"],
            "scenario_type": scenario["scenario_type"],
            "bs_price": bs_price,
            "bs_delta": bs_delta,
            "bs_gamma": bs_gamma,
            "mc_price": mc_price,
            "mc_std": mc_std,
            "intrinsic_value": max(scenario["S"] - scenario["K"], 0)
            if scenario["option_type"] == "call"
            else max(scenario["K"] - scenario["S"], 0),
        })

    return pd.DataFrame(results)


def generate_stress_test_grid() -> pd.DataFrame:
    """Generate a comprehensive grid for stress testing."""
    S_range = np.linspace(50, 150, 21)
    K_range = np.linspace(70, 130, 13)
    vol_range = [0.1, 0.2, 0.3, 0.5]
    T_range = [0.1, 0.5, 1.0, 2.0]

    grid_data = []

    for S in S_range:
        for K in K_range:
            for vol in vol_range:
                for T in T_range:
                    price, delta, gamma = black_scholes_analytical(
                        S, K, 0.05, vol, T, "call"
                    )

                    grid_data.append({
                        "S": S,
                        "K": K,
                        "r": 0.05,
                        "sigma": vol,
                        "T": T,
                        "bs_price": price,
                        "bs_delta": delta,
                        "bs_gamma": gamma,
                    })

    return pd.DataFrame(grid_data)


def main():
    """Generate all test datasets."""
    parser = argparse.ArgumentParser(description="Generate test data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_data",
        help="Output directory for test data",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating market scenarios...")
    scenarios = generate_market_scenarios()

    # Save scenarios
    with open(output_dir / "scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"✓ Saved {len(scenarios)} scenarios to scenarios.json")

    # Compute ground truth
    print("\nComputing ground truth prices...")
    df_scenarios = compute_ground_truth(scenarios)
    df_scenarios.to_csv(output_dir / "scenario_ground_truth.csv", index=False)
    print(f"✓ Saved ground truth to scenario_ground_truth.csv")

    # Generate stress test grid
    print("\nGenerating stress test grid...")
    df_grid = generate_stress_test_grid()
    df_grid.to_csv(output_dir / "stress_test_grid.csv", index=False)
    print(f"✓ Saved {len(df_grid)} test points to stress_test_grid.csv")

    # Summary statistics
    print("\n" + "=" * 60)
    print("TEST DATA SUMMARY")
    print("=" * 60)
    print(f"\nTotal scenarios: {len(scenarios)}")
    print(f"Scenario types:")
    for stype in df_scenarios["scenario_type"].unique():
        count = len(df_scenarios[df_scenarios["scenario_type"] == stype])
        print(f"  - {stype}: {count}")

    print(f"\nStress test grid: {len(df_grid)} points")
    print(f"  - S range: [{df_grid['S'].min():.1f}, {df_grid['S'].max():.1f}]")
    print(f"  - K range: [{df_grid['K'].min():.1f}, {df_grid['K'].max():.1f}]")
    print(f"  - Vol range: {df_grid['sigma'].unique()}")
    print(f"  - T range: {df_grid['T'].unique()}")

    print(f"\n✓ All test data saved to {output_dir}/")
    print("\nUsage:")
    print(f"  python scripts/validate_model.py --test-data {output_dir}")


if __name__ == "__main__":
    main()
