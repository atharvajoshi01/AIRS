"""
Unit tests for backtesting framework.
"""

import numpy as np
import pandas as pd
import pytest


class TestPortfolio:
    """Tests for portfolio management."""

    def test_portfolio_initialization(self, backtest_config: dict):
        """Test portfolio initialization."""
        from airs.backtest.portfolio import Portfolio

        portfolio = Portfolio(
            initial_value=backtest_config["initial_value"],
            target_weights=backtest_config["target_weights"],
        )

        assert portfolio.total_value == backtest_config["initial_value"]
        assert portfolio.cash == backtest_config["initial_value"]
        assert len(portfolio.positions) == 0

    def test_portfolio_initialize_with_prices(
        self,
        backtest_config: dict,
        sample_prices: pd.DataFrame,
    ):
        """Test portfolio initialization with prices."""
        from airs.backtest.portfolio import Portfolio

        portfolio = Portfolio(
            initial_value=backtest_config["initial_value"],
            target_weights=backtest_config["target_weights"],
        )

        portfolio.initialize(sample_prices.iloc[0])

        # Should have positions
        assert len(portfolio.positions) > 0

        # Weights should approximately match targets
        for symbol, target in backtest_config["target_weights"].items():
            if symbol in portfolio.current_weights:
                actual = portfolio.current_weights[symbol]
                assert abs(actual - target) < 0.01

    def test_portfolio_update_values(
        self,
        backtest_config: dict,
        sample_prices: pd.DataFrame,
    ):
        """Test portfolio value updates."""
        from airs.backtest.portfolio import Portfolio

        portfolio = Portfolio(
            initial_value=backtest_config["initial_value"],
            target_weights=backtest_config["target_weights"],
        )

        portfolio.initialize(sample_prices.iloc[0])
        initial_value = portfolio.total_value

        # Update with new prices
        portfolio.update_values(sample_prices.iloc[10])

        # Value should have changed
        assert portfolio.total_value != initial_value

    def test_portfolio_rebalance(
        self,
        backtest_config: dict,
        sample_prices: pd.DataFrame,
    ):
        """Test portfolio rebalancing."""
        from airs.backtest.portfolio import Portfolio

        portfolio = Portfolio(
            initial_value=backtest_config["initial_value"],
            target_weights=backtest_config["target_weights"],
        )

        portfolio.initialize(sample_prices.iloc[0])

        # Modify weights
        new_weights = {k: v * 0.8 for k, v in backtest_config["target_weights"].items()}
        new_weights["CASH"] = 0.20

        trades = portfolio.rebalance_to_weights(new_weights, sample_prices.iloc[10])

        # Should have executed some trades
        assert len(trades) > 0


class TestTransactionCosts:
    """Tests for transaction cost modeling."""

    def test_basic_cost_calculation(self):
        """Test basic transaction cost calculation."""
        from airs.backtest.costs import TransactionCostModel

        model = TransactionCostModel(
            trading_cost_bps=10.0,
            slippage_bps=5.0,
        )

        trade_value = 10_000
        cost = model.calculate_cost(trade_value)

        # 15 bps = 0.15%
        expected_cost = trade_value * 0.0015
        assert abs(cost - expected_cost) < 0.01

    def test_round_trip_cost(self):
        """Test round-trip cost estimation."""
        from airs.backtest.costs import TransactionCostModel

        model = TransactionCostModel(
            trading_cost_bps=10.0,
            slippage_bps=5.0,
        )

        position_value = 10_000
        round_trip = model.estimate_round_trip_cost(position_value)

        # Should be 2x one-way cost
        one_way = model.calculate_cost(position_value)
        assert abs(round_trip - 2 * one_way) < 0.01

    def test_slippage_calculation(self):
        """Test slippage calculation."""
        from airs.backtest.costs import TransactionCostModel

        model = TransactionCostModel(slippage_bps=10.0)

        price = 100.0
        shares = 100

        # Buy should increase price
        buy_price = model.calculate_slippage(price, shares, "buy")
        assert buy_price > price

        # Sell should decrease price
        sell_price = model.calculate_slippage(price, shares, "sell")
        assert sell_price < price


class TestPerformanceMetrics:
    """Tests for performance metric calculation."""

    def test_total_return(self, sample_prices: pd.DataFrame):
        """Test total return calculation."""
        from airs.backtest.metrics import PerformanceMetrics

        calculator = PerformanceMetrics()

        # Use SPY as portfolio
        portfolio_values = sample_prices["SPY"]
        metrics = calculator.calculate_all(portfolio_values)

        assert "total_return" in metrics
        # Return should be reasonable
        assert -0.9 < metrics["total_return"] < 10.0

    def test_sharpe_ratio(self, sample_prices: pd.DataFrame):
        """Test Sharpe ratio calculation."""
        from airs.backtest.metrics import PerformanceMetrics

        calculator = PerformanceMetrics()
        portfolio_values = sample_prices["SPY"]
        metrics = calculator.calculate_all(portfolio_values)

        assert "sharpe_ratio" in metrics
        # Sharpe should be reasonable
        assert -5 < metrics["sharpe_ratio"] < 5

    def test_max_drawdown(self, sample_prices: pd.DataFrame):
        """Test max drawdown calculation."""
        from airs.backtest.metrics import PerformanceMetrics

        calculator = PerformanceMetrics()
        portfolio_values = sample_prices["SPY"]
        metrics = calculator.calculate_all(portfolio_values)

        assert "max_drawdown" in metrics
        # Drawdown should be negative
        assert metrics["max_drawdown"] <= 0
        # And reasonable
        assert metrics["max_drawdown"] > -1.0

    def test_volatility(self, sample_prices: pd.DataFrame):
        """Test volatility calculation."""
        from airs.backtest.metrics import PerformanceMetrics

        calculator = PerformanceMetrics()
        portfolio_values = sample_prices["SPY"]
        metrics = calculator.calculate_all(portfolio_values)

        assert "volatility" in metrics
        # Volatility should be positive
        assert metrics["volatility"] > 0
        # And reasonable for equity
        assert metrics["volatility"] < 1.0


class TestBacktestEngine:
    """Tests for the backtest engine."""

    def test_backtest_execution(
        self,
        backtest_config: dict,
        sample_prices: pd.DataFrame,
        sample_signals: pd.Series,
    ):
        """Test basic backtest execution."""
        from airs.backtest.engine import BacktestEngine, BacktestConfig

        config = BacktestConfig(**backtest_config)
        engine = BacktestEngine(config)

        results = engine.run(sample_prices, sample_signals)

        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "n_trades" in results

    def test_derisking_logic(
        self,
        backtest_config: dict,
        sample_prices: pd.DataFrame,
    ):
        """Test that de-risking is triggered correctly."""
        from airs.backtest.engine import BacktestEngine, BacktestConfig

        config = BacktestConfig(**backtest_config)
        engine = BacktestEngine(config)

        # Create signals with clear high-risk period
        signals = pd.Series(0.2, index=sample_prices.index)
        signals.iloc[100:110] = 0.9  # High risk period

        results = engine.run(sample_prices, signals)

        # Should have triggered at least one de-risking event
        assert results["n_alerts"] > 0

    def test_benchmark_comparison(
        self,
        backtest_config: dict,
        sample_prices: pd.DataFrame,
        sample_signals: pd.Series,
    ):
        """Test strategy vs benchmark comparison."""
        from airs.backtest.engine import BacktestEngine, BacktestConfig

        config = BacktestConfig(**backtest_config)
        engine = BacktestEngine(config)

        results = engine.compare_to_benchmark(sample_prices, sample_signals)

        assert "strategy" in results
        assert "benchmark" in results
        assert "excess_return" in results


class TestStressPeriodAnalysis:
    """Tests for stress period analysis."""

    def test_analyze_period(
        self,
        sample_prices: pd.DataFrame,
        sample_signals: pd.Series,
    ):
        """Test single stress period analysis."""
        from airs.backtest.analysis import StressPeriodAnalyzer, StressPeriod

        analyzer = StressPeriodAnalyzer()

        # Create a mock stress period
        period = StressPeriod(
            name="Test Period",
            start_date="2021-01-01",
            end_date="2021-03-31",
            description="Test",
            severity="moderate",
        )

        strategy_values = sample_prices["SPY"]
        benchmark_values = sample_prices["AGG"]

        result = analyzer.analyze_period(
            strategy_values,
            benchmark_values,
            period,
            sample_signals,
        )

        if result.get("available"):
            assert "strategy" in result
            assert "benchmark" in result
            assert "outperformance" in result
