#!/usr/bin/env python3
"""
AIRS Command Line Interface.

This module provides the CLI entry point for the AIRS application.
It uses Typer for command handling and Rich for formatted output.

Usage:
    airs --help
    airs fetch --all
    airs train --model ensemble
    airs backtest --report
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Initialize Typer app
app = typer.Typer(
    name="airs",
    help="AI-Driven Early-Warning System for Portfolio Drawdown Risk",
    add_completion=False,
)

console = Console()


@app.callback()
def main():
    """
    AIRS: AI-Driven Early-Warning System for Portfolio Drawdown Risk.

    A production-ready ML system for detecting portfolio stress and
    generating de-risking recommendations.
    """
    pass


@app.command()
def fetch(
    all_data: bool = typer.Option(False, "--all", help="Fetch all historical data"),
    incremental: bool = typer.Option(False, "--incremental", help="Fetch only new data"),
    start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD)"),
    skip_macro: bool = typer.Option(False, "--skip-macro", help="Skip macro indicators"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    """
    Fetch market and economic data.

    Downloads data from Yahoo Finance, FRED, and Alpha Vantage.
    """
    from airs.config import get_settings
    from airs.data import DataAggregator
    from airs.utils.logging import setup_logging, get_logger

    setup_logging(level=log_level)
    logger = get_logger(__name__)
    settings = get_settings()

    console.print(Panel.fit("[bold blue]AIRS Data Fetch[/bold blue]"))

    # Validate API key
    if not settings.fred_api_key:
        console.print("[red]Error: FRED_API_KEY not set[/red]")
        console.print("Get your free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        raise typer.Exit(1)

    # Determine date range
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    if start is None:
        if incremental:
            start_dt = datetime.now() - timedelta(days=30)
        else:
            start_dt = datetime.now() - timedelta(days=365 * 10)
        start = start_dt.strftime("%Y-%m-%d")

    console.print(f"Date range: {start} to {end}")

    try:
        with console.status("[bold green]Fetching data..."):
            aggregator = DataAggregator(
                fred_api_key=settings.fred_api_key,
                alpha_vantage_api_key=settings.alpha_vantage_api_key or None,
            )

            df = aggregator.prepare_features_dataset(
                start_date=start,
                end_date=end,
                forward_fill=True,
                drop_na_threshold=0.5,
            )

            output_path = aggregator.save_dataset(df, name="market_data", format="parquet")

        # Show summary
        table = Table(title="Data Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Observations", str(len(df)))
        table.add_row("Features", str(len(df.columns)))
        table.add_row("Date Range", f"{df.index.min().date()} to {df.index.max().date()}")
        table.add_row("Output Path", str(output_path))

        console.print(table)
        console.print("[green]Data fetch completed successfully![/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def train(
    data: Optional[Path] = typer.Option(None, help="Path to market data"),
    model: str = typer.Option("ensemble", help="Model type: xgboost, lightgbm, ensemble"),
    output: Optional[Path] = typer.Option(None, help="Output directory"),
    track: bool = typer.Option(False, "--track", help="Enable MLflow tracking"),
    threshold: float = typer.Option(-0.05, help="Drawdown threshold"),
    horizon: int = typer.Option(15, help="Prediction horizon (days)"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    """
    Train drawdown prediction models.

    Trains the specified model on historical data.
    """
    import subprocess
    import sys

    # Build command
    cmd = [sys.executable, "scripts/train_model.py", "--model", model]

    if data:
        cmd.extend(["--data", str(data)])
    if output:
        cmd.extend(["--output", str(output)])
    if track:
        cmd.append("--track")

    cmd.extend(["--threshold", str(threshold)])
    cmd.extend(["--horizon", str(horizon)])
    cmd.extend(["--log-level", log_level])

    console.print(Panel.fit("[bold blue]AIRS Model Training[/bold blue]"))
    console.print(f"Model type: {model}")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    raise typer.Exit(result.returncode)


@app.command()
def backtest(
    data: Optional[Path] = typer.Option(None, help="Path to market data"),
    model_path: Optional[Path] = typer.Option(None, "--model", help="Path to trained model"),
    start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD)"),
    initial_value: float = typer.Option(100000.0, help="Initial portfolio value"),
    threshold: float = typer.Option(0.5, help="Alert probability threshold"),
    report: bool = typer.Option(False, "--report", help="Generate detailed report"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    """
    Run portfolio backtesting.

    Simulates the de-risking strategy on historical data.
    """
    import subprocess
    import sys

    # Build command
    cmd = [sys.executable, "scripts/run_backtest.py"]

    if data:
        cmd.extend(["--data", str(data)])
    if model_path:
        cmd.extend(["--model", str(model_path)])
    if start:
        cmd.extend(["--start", start])
    if end:
        cmd.extend(["--end", end])
    if report:
        cmd.append("--report")

    cmd.extend(["--initial-value", str(initial_value)])
    cmd.extend(["--threshold", str(threshold)])
    cmd.extend(["--log-level", log_level])

    console.print(Panel.fit("[bold blue]AIRS Portfolio Backtest[/bold blue]"))

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    raise typer.Exit(result.returncode)


@app.command()
def api(
    host: str = typer.Option("0.0.0.0", help="API host"),
    port: int = typer.Option(8000, help="API port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """
    Start the AIRS REST API server.
    """
    import subprocess
    import sys

    console.print(Panel.fit("[bold blue]AIRS API Server[/bold blue]"))
    console.print(f"Starting server at http://{host}:{port}")
    console.print(f"API docs: http://{host}:{port}/docs")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "airs.api.main:app",
        "--host", host,
        "--port", str(port),
    ]

    if reload:
        cmd.append("--reload")

    result = subprocess.run(cmd)
    raise typer.Exit(result.returncode)


@app.command()
def status():
    """
    Show AIRS system status and configuration.
    """
    from airs.config import get_settings

    settings = get_settings()

    console.print(Panel.fit("[bold blue]AIRS System Status[/bold blue]"))

    # Configuration table
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Environment", settings.airs_env)
    table.add_row("Log Level", settings.log_level)
    table.add_row("Data Directory", str(settings.data_dir))
    table.add_row("FRED API Key", "Set" if settings.fred_api_key else "[red]Not Set[/red]")
    table.add_row("Alpha Vantage Key", "Set" if settings.alpha_vantage_api_key else "Not Set")
    table.add_row("Database URL", settings.postgres_host + ":" + str(settings.postgres_port))
    table.add_row("MLflow URI", settings.mlflow_tracking_uri)

    console.print(table)

    # Check data files
    data_table = Table(title="Data Files")
    data_table.add_column("File", style="cyan")
    data_table.add_column("Status", style="green")

    data_files = [
        settings.data_dir / "processed" / "market_data.parquet",
        settings.data_dir / "models" / "ensemble_model.pkl",
    ]

    for f in data_files:
        status = "[green]Exists[/green]" if f.exists() else "[red]Missing[/red]"
        data_table.add_row(str(f.name), status)

    console.print(data_table)


@app.command()
def version():
    """
    Show AIRS version information.
    """
    console.print("[bold]AIRS[/bold] - AI-Driven Early-Warning System for Portfolio Drawdown Risk")
    console.print("Version: 0.1.0")
    console.print("Python: 3.10+")


if __name__ == "__main__":
    app()
