"""
Portfolio management for backtesting.

Tracks positions, values, and handles rebalancing.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from airs.backtest.costs import TransactionCostModel
from airs.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents a position in a single asset."""

    symbol: str
    shares: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis of position."""
        return self.shares * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.market_value - self.cost_basis

    @property
    def unrealized_return(self) -> float:
        """Unrealized return percentage."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis

    def update_price(self, price: float) -> None:
        """Update current price."""
        self.current_price = price

    def add_shares(self, shares: float, price: float) -> None:
        """Add shares to position."""
        total_cost = self.shares * self.avg_cost + shares * price
        self.shares += shares
        if self.shares > 0:
            self.avg_cost = total_cost / self.shares
        self.current_price = price

    def remove_shares(self, shares: float, price: float) -> float:
        """
        Remove shares from position.

        Returns realized P&L.
        """
        shares = min(shares, self.shares)
        realized_pnl = shares * (price - self.avg_cost)
        self.shares -= shares
        self.current_price = price
        return realized_pnl


class Portfolio:
    """
    Portfolio manager for backtesting.

    Tracks positions, cash, and handles rebalancing.
    """

    def __init__(
        self,
        initial_value: float = 100_000.0,
        target_weights: dict[str, float] | None = None,
    ):
        """
        Initialize portfolio.

        Args:
            initial_value: Starting portfolio value
            target_weights: Target allocation weights
        """
        self.initial_value = initial_value
        self.target_weights = target_weights or {}

        self.cash = initial_value
        self.positions: dict[str, Position] = {}

        # Trading history
        self.trade_history: list[dict] = []
        self.value_history: list[dict] = []

    @property
    def total_value(self) -> float:
        """Total portfolio value (positions + cash)."""
        position_value = sum(p.market_value for p in self.positions.values())
        return position_value + self.cash

    @property
    def position_values(self) -> dict[str, float]:
        """Market value of each position."""
        return {symbol: pos.market_value for symbol, pos in self.positions.items()}

    @property
    def current_weights(self) -> dict[str, float]:
        """Current weight of each position."""
        total = self.total_value
        if total == 0:
            return {}

        weights = {symbol: pos.market_value / total for symbol, pos in self.positions.items()}
        weights["CASH"] = self.cash / total
        return weights

    def initialize(self, prices: pd.Series) -> None:
        """
        Initialize portfolio with target allocation.

        Args:
            prices: Current asset prices
        """
        for symbol, weight in self.target_weights.items():
            if symbol == "CASH":
                continue

            if symbol in prices.index:
                price = prices[symbol]
                target_value = self.initial_value * weight
                shares = target_value / price

                self.positions[symbol] = Position(
                    symbol=symbol,
                    shares=shares,
                    avg_cost=price,
                    current_price=price,
                )

                self.cash -= target_value

        logger.info(
            f"Initialized portfolio: {len(self.positions)} positions, "
            f"${self.cash:,.2f} cash"
        )

    def update_values(self, prices: pd.Series) -> None:
        """
        Update position values with current prices.

        Args:
            prices: Current asset prices
        """
        for symbol, position in self.positions.items():
            if symbol in prices.index:
                position.update_price(prices[symbol])

    def rebalance_to_weights(
        self,
        target_weights: dict[str, float],
        prices: pd.Series,
        cost_model: TransactionCostModel | None = None,
    ) -> list[dict]:
        """
        Rebalance portfolio to target weights.

        Args:
            target_weights: Target allocation weights
            prices: Current prices
            cost_model: Transaction cost model

        Returns:
            List of executed trades
        """
        trades = []
        total_value = self.total_value

        for symbol, target_weight in target_weights.items():
            if symbol == "CASH":
                continue

            target_value = total_value * target_weight
            current_value = self.position_values.get(symbol, 0)
            trade_value = target_value - current_value

            if abs(trade_value) < 100:  # Skip small trades
                continue

            if symbol not in prices.index:
                continue

            price = prices[symbol]

            # Apply transaction costs
            if cost_model:
                trade_cost = cost_model.calculate_cost(abs(trade_value))
            else:
                trade_cost = 0

            # Execute trade
            shares_delta = trade_value / price

            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol=symbol)

            if shares_delta > 0:
                # Buy
                actual_cost = trade_value + trade_cost
                self.positions[symbol].add_shares(shares_delta, price)
                self.cash -= actual_cost
            else:
                # Sell
                realized_pnl = self.positions[symbol].remove_shares(-shares_delta, price)
                actual_proceeds = -trade_value - trade_cost
                self.cash += actual_proceeds

            trades.append({
                "symbol": symbol,
                "shares": shares_delta,
                "price": price,
                "value": trade_value,
                "cost": trade_cost,
                "side": "buy" if shares_delta > 0 else "sell",
            })

        return trades

    def get_weight_deviation(self) -> dict[str, float]:
        """
        Calculate deviation from target weights.

        Returns:
            Dictionary of weight deviations
        """
        current = self.current_weights
        deviations = {}

        for symbol, target in self.target_weights.items():
            current_weight = current.get(symbol, 0)
            deviations[symbol] = current_weight - target

        return deviations

    def needs_rebalance(self, threshold: float = 0.05) -> bool:
        """
        Check if portfolio needs rebalancing.

        Args:
            threshold: Minimum deviation to trigger rebalance

        Returns:
            True if rebalance needed
        """
        deviations = self.get_weight_deviation()
        max_deviation = max(abs(d) for d in deviations.values())
        return max_deviation > threshold

    def get_summary(self) -> dict[str, Any]:
        """Get portfolio summary."""
        return {
            "total_value": self.total_value,
            "cash": self.cash,
            "positions": {
                symbol: {
                    "shares": pos.shares,
                    "price": pos.current_price,
                    "value": pos.market_value,
                    "weight": pos.market_value / self.total_value if self.total_value > 0 else 0,
                    "pnl": pos.unrealized_pnl,
                }
                for symbol, pos in self.positions.items()
            },
            "weights": self.current_weights,
            "pnl": self.total_value - self.initial_value,
            "return": (self.total_value / self.initial_value - 1) * 100,
        }
