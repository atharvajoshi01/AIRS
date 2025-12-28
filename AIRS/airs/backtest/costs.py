"""
Transaction cost modeling for backtesting.

Includes trading costs, slippage, and market impact.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from airs.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TransactionCostModel:
    """
    Transaction cost model for realistic backtesting.

    Includes:
    - Fixed trading costs (commissions, fees)
    - Proportional costs (bid-ask spread, slippage)
    - Market impact (for large orders)
    """

    # Costs in basis points
    trading_cost_bps: float = 10.0  # 10 bps = 0.1%
    slippage_bps: float = 5.0  # 5 bps = 0.05%

    # Market impact parameters
    market_impact_enabled: bool = False
    market_impact_coefficient: float = 0.1
    average_daily_volume: float = 1_000_000  # Default ADV

    # Fixed costs
    min_commission: float = 0.0  # Minimum per trade

    def calculate_cost(
        self,
        trade_value: float,
        adv: float | None = None,
    ) -> float:
        """
        Calculate total transaction cost.

        Args:
            trade_value: Absolute value of trade
            adv: Average daily volume (for market impact)

        Returns:
            Total cost in dollars
        """
        # Proportional costs
        prop_cost = trade_value * (self.trading_cost_bps + self.slippage_bps) / 10000

        # Market impact
        if self.market_impact_enabled:
            adv = adv or self.average_daily_volume
            participation_rate = trade_value / adv
            impact_cost = (
                trade_value
                * self.market_impact_coefficient
                * np.sqrt(participation_rate)
            )
            prop_cost += impact_cost

        # Apply minimum
        return max(prop_cost, self.min_commission)

    def calculate_slippage(
        self,
        price: float,
        shares: float,
        side: Literal["buy", "sell"],
    ) -> float:
        """
        Calculate execution price with slippage.

        Args:
            price: Quote price
            shares: Number of shares
            side: Buy or sell

        Returns:
            Execution price after slippage
        """
        slippage_pct = self.slippage_bps / 10000

        if side == "buy":
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)

    def estimate_round_trip_cost(self, position_value: float) -> float:
        """
        Estimate round-trip cost (buy + sell).

        Args:
            position_value: Position value

        Returns:
            Estimated round-trip cost
        """
        return 2 * self.calculate_cost(position_value)

    def get_cost_summary(self) -> dict:
        """Get cost model summary."""
        return {
            "trading_cost_bps": self.trading_cost_bps,
            "slippage_bps": self.slippage_bps,
            "total_one_way_bps": self.trading_cost_bps + self.slippage_bps,
            "round_trip_bps": 2 * (self.trading_cost_bps + self.slippage_bps),
            "market_impact_enabled": self.market_impact_enabled,
        }


class RealisticCostModel(TransactionCostModel):
    """
    More realistic cost model with asset-specific costs.
    """

    # Asset-specific spreads (approximate bid-ask spreads in bps)
    ASSET_SPREADS = {
        "SPY": 1,  # Very liquid
        "AGG": 3,
        "VEU": 5,
        "VNQ": 5,
        "DJP": 10,
        "GLD": 3,
        "TLT": 3,
        "HYG": 10,
    }

    def calculate_cost(
        self,
        trade_value: float,
        symbol: str | None = None,
        adv: float | None = None,
    ) -> float:
        """
        Calculate cost with asset-specific spreads.

        Args:
            trade_value: Trade value
            symbol: Asset symbol
            adv: Average daily volume

        Returns:
            Total cost
        """
        # Get asset-specific spread
        if symbol and symbol in self.ASSET_SPREADS:
            spread_bps = self.ASSET_SPREADS[symbol]
        else:
            spread_bps = self.slippage_bps

        # Calculate cost
        prop_cost = trade_value * (self.trading_cost_bps + spread_bps) / 10000

        # Market impact for large trades
        if self.market_impact_enabled and adv:
            participation = trade_value / adv
            if participation > 0.01:  # More than 1% of ADV
                impact = trade_value * 0.1 * np.sqrt(participation)
                prop_cost += impact

        return max(prop_cost, self.min_commission)
