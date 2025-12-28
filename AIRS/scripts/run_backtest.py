#!/usr/bin/env python3
"""
Run AIRS portfolio backtesting simulation.

This script backtests the de-risking strategy using trained models
and historical market data.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --report  # Generate detailed report
    python scripts/run_backtest.py --model ./data/models/ensemble_model.pkl
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from airs.config import get_settings
from airs.data import DataAggregator
from airs.features import FeaturePipeline
from airs.targets import DrawdownCalculator, LabelGenerator
from airs.models import StackingEnsemble
from airs.backtest import BacktestEngine, BacktestConfig, PerformanceMetrics, StressPeriodAnalyzer
from airs.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


# Default portfolio weights
DEFAULT_WEIGHTS = {
    "SPY": 0.40,
    "VEU": 0.20,
    "AGG": 0.25,
    "DJP": 0.10,
    "VNQ": 0.05,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AIRS portfolio backtesting simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to market data. Default: ./data/processed/market_data.parquet",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model. Default: ./data/models/ensemble_model.pkl",
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Backtest start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Backtest end date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--initial-value",
        type=float,
        default=100000.0,
        help="Initial portfolio value. Default: 100000",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Alert probability threshold. Default: 0.5",
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed performance report",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results. Default: ./data/backtest",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def load_data(data_path: Path) -> pd.DataFrame:
    """Load market data from file."""
    if data_path.suffix == ".parquet":
        return pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        return pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")


def extract_prices(data: pd.DataFrame) -> pd.DataFrame:
    """Extract asset prices from data."""
    prices = pd.DataFrame(index=data.index)

    # Map common column patterns to assets
    asset_patterns = {
        "SPY": ["SPY", "prices_SPY", "prices_close_SPY"],
        "VEU": ["VEU", "prices_VEU", "prices_close_VEU"],
        "AGG": ["AGG", "prices_AGG", "prices_close_AGG"],
        "DJP": ["DJP", "prices_DJP", "prices_close_DJP"],
        "VNQ": ["VNQ", "prices_VNQ", "prices_close_VNQ"],
    }

    for asset, patterns in asset_patterns.items():
        for pattern in patterns:
            if pattern in data.columns:
                prices[asset] = data[pattern]
                break

    # If we couldn't find direct columns, try to infer
    if prices.empty:
        logger.warning("Could not find standard price columns, attempting inference...")
        for col in data.columns:
            for asset in DEFAULT_WEIGHTS.keys():
                if asset in col and "close" in col.lower():
                    prices[asset] = data[col]
                    break

    # Validate we have all assets
    missing = set(DEFAULT_WEIGHTS.keys()) - set(prices.columns)
    if missing:
        logger.warning(f"Missing price data for: {missing}")

    return prices


def generate_signals(
    model,
    data: pd.DataFrame,
    pipeline: FeaturePipeline,
) -> pd.Series:
    """Generate model predictions as signals."""
    # Generate features
    features = pipeline.generate_features(data, add_prefix=True)
    features = pipeline.clean_features(
        features,
        max_missing_pct=0.3,
        drop_constant=True,
        fill_method="ffill",
        fill_limit=5,
    )

    # Get model predictions
    proba = model.predict_proba(features)

    # Handle probability format
    if len(proba.shape) > 1:
        proba = proba[:, 1]

    signals = pd.Series(proba, index=features.index, name="signal")

    return signals


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    initial_value: float,
    threshold: float,
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """Run the backtesting simulation."""
    # Create config using imported BacktestConfig
    config = BacktestConfig(
        initial_value=initial_value,
        target_weights=DEFAULT_WEIGHTS,
        alert_threshold=threshold,
    )

    # Initialize engine
    engine = BacktestEngine(config)

    # Run backtest
    results = engine.run(prices, signals, start_date, end_date)

    # Run benchmark
    benchmark = engine.run_benchmark(prices, start_date, end_date)

    # Compare
    comparison = engine.compare_to_benchmark(prices, signals)

    return {
        "strategy": results,
        "benchmark": benchmark,
        "comparison": comparison,
        "engine": engine,
    }


def generate_report(
    results: dict,
    prices: pd.DataFrame,
    signals: pd.Series,
    output_dir: Path,
) -> str:
    """Generate detailed performance report."""
    strategy = results["strategy"]
    benchmark = results["benchmark"]
    comparison = results["comparison"]

    report_lines = [
        "=" * 70,
        "AIRS BACKTEST PERFORMANCE REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "PORTFOLIO CONFIGURATION",
        "-" * 40,
    ]

    for asset, weight in DEFAULT_WEIGHTS.items():
        report_lines.append(f"  {asset}: {weight*100:.0f}%")

    report_lines.extend([
        "",
        "STRATEGY PERFORMANCE",
        "-" * 40,
        f"  Total Return:      {strategy.get('total_return', 0)*100:>10.2f}%",
        f"  Annualized Return: {strategy.get('annualized_return', 0)*100:>10.2f}%",
        f"  Volatility:        {strategy.get('volatility', 0)*100:>10.2f}%",
        f"  Max Drawdown:      {strategy.get('max_drawdown', 0)*100:>10.2f}%",
        f"  Sharpe Ratio:      {strategy.get('sharpe_ratio', 0):>10.2f}",
        f"  Sortino Ratio:     {strategy.get('sortino_ratio', 0):>10.2f}",
        f"  Calmar Ratio:      {strategy.get('calmar_ratio', 0):>10.2f}",
        "",
        "BENCHMARK (BUY & HOLD) PERFORMANCE",
        "-" * 40,
        f"  Total Return:      {benchmark.get('total_return', 0)*100:>10.2f}%",
        f"  Annualized Return: {benchmark.get('annualized_return', 0)*100:>10.2f}%",
        f"  Volatility:        {benchmark.get('volatility', 0)*100:>10.2f}%",
        f"  Max Drawdown:      {benchmark.get('max_drawdown', 0)*100:>10.2f}%",
        f"  Sharpe Ratio:      {benchmark.get('sharpe_ratio', 0):>10.2f}",
        "",
        "STRATEGY VS BENCHMARK",
        "-" * 40,
        f"  Excess Return:     {comparison.get('excess_return', 0)*100:>10.2f}%",
        f"  Excess Sharpe:     {comparison.get('excess_sharpe', 0):>10.2f}",
        f"  DD Improvement:    {comparison.get('drawdown_improvement', 0)*100:>10.2f}%",
        "",
        "TRADING ACTIVITY",
        "-" * 40,
        f"  Number of Alerts:  {strategy.get('n_alerts', 0):>10d}",
        f"  Number of Trades:  {strategy.get('n_trades', 0):>10d}",
        "",
        "=" * 70,
    ])

    report = "\n".join(report_lines)

    # Save report
    report_path = output_dir / "backtest_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    return report


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    logger.info("=" * 60)
    logger.info("AIRS Portfolio Backtest")
    logger.info("=" * 60)

    # Get settings
    settings = get_settings()

    # Determine paths
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = settings.data_dir / "processed" / "market_data.parquet"

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = settings.data_dir / "models" / "ensemble_model.pkl"

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = settings.data_dir / "backtest"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate paths
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Run 'make fetch-data' first")
        return 1

    try:
        # Load data
        logger.info(f"Loading data from {data_path}...")
        data = load_data(data_path)
        logger.info(f"Loaded data: {data.shape}")

        # Extract prices
        logger.info("Extracting asset prices...")
        prices = extract_prices(data)
        logger.info(f"Price data: {prices.shape}")

        # Load or train model
        if model_path.exists():
            logger.info(f"Loading model from {model_path}...")
            model = StackingEnsemble()
            model.load(model_path)
        else:
            logger.warning(f"Model not found at {model_path}")
            logger.info("Training new model...")

            # Quick train for backtest
            pipeline = FeaturePipeline(lookback_window=252)
            features = pipeline.generate_features(data, add_prefix=True)
            features = pipeline.clean_features(features)

            # Get labels
            label_gen = LabelGenerator(threshold=-0.05, horizon=15)
            spy_prices = prices["SPY"] if "SPY" in prices.columns else prices.iloc[:, 0]
            labels = label_gen.generate_binary_labels(spy_prices)

            # Align
            common_idx = features.index.intersection(labels.index)
            features = features.loc[common_idx]
            labels = labels.loc[common_idx]

            # Train
            model = StackingEnsemble()
            model.fit(features, labels)

            # Save
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")

        # Generate signals
        logger.info("Generating prediction signals...")
        pipeline = FeaturePipeline(lookback_window=252)
        signals = generate_signals(model, data, pipeline)
        logger.info(f"Generated {len(signals)} signals")
        logger.info(f"Signal stats: mean={signals.mean():.3f}, std={signals.std():.3f}")

        # Run backtest
        logger.info("Running backtest simulation...")
        results = run_backtest(
            prices=prices,
            signals=signals,
            initial_value=args.initial_value,
            threshold=args.threshold,
            start_date=args.start,
            end_date=args.end,
        )

        # Print summary
        strategy = results["strategy"]
        benchmark = results["benchmark"]

        logger.info("")
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Strategy Total Return:  {strategy.get('total_return', 0)*100:.2f}%")
        logger.info(f"Benchmark Total Return: {benchmark.get('total_return', 0)*100:.2f}%")
        logger.info(f"Strategy Sharpe:        {strategy.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Strategy Max DD:        {strategy.get('max_drawdown', 0)*100:.2f}%")
        logger.info(f"Benchmark Max DD:       {benchmark.get('max_drawdown', 0)*100:.2f}%")

        # Generate detailed report if requested
        if args.report:
            logger.info("Generating detailed report...")
            report = generate_report(results, prices, signals, output_dir)
            print("\n" + report)

            # Stress period analysis
            logger.info("Analyzing stress periods...")
            analyzer = StressPeriodAnalyzer()

            # Get portfolio values if available
            if hasattr(results.get("engine"), "portfolio_values"):
                stress_report = analyzer.generate_report(
                    analyzer.analyze_all_periods(
                        results["engine"].portfolio_values,
                        benchmark,
                        signals,
                    )
                )
                stress_path = output_dir / "stress_analysis.txt"
                with open(stress_path, "w") as f:
                    f.write(stress_report)
                logger.info(f"Stress analysis saved to: {stress_path}")

        # Save results
        results_path = output_dir / "backtest_results.csv"
        pd.DataFrame([strategy]).to_csv(results_path)
        logger.info(f"Results saved to: {results_path}")

        logger.info("")
        logger.info("Backtest completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
