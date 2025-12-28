"""
Recommendation engine for portfolio de-risking.

Maps model predictions to actionable portfolio recommendations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from airs.utils.logging import get_logger

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(Enum):
    """Types of recommended actions."""

    HOLD = "hold"
    REDUCE_EQUITY = "reduce_equity"
    INCREASE_BONDS = "increase_bonds"
    INCREASE_CASH = "increase_cash"
    ADD_HEDGES = "add_hedges"
    FULL_DERISK = "full_derisk"


@dataclass
class AssetRecommendation:
    """Recommendation for a single asset."""

    symbol: str
    current_weight: float
    target_weight: float
    action: str  # "buy", "sell", "hold"
    urgency: str  # "immediate", "gradual", "optional"
    rationale: str


@dataclass
class Recommendation:
    """Complete portfolio recommendation."""

    timestamp: pd.Timestamp
    alert_level: AlertLevel
    probability: float
    confidence: float
    headline: str
    summary: str
    asset_recommendations: list[AssetRecommendation]
    key_drivers: list[dict[str, Any]]
    historical_context: str
    suggested_timeline: str
    estimated_turnover: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "alert_level": self.alert_level.value,
            "probability": self.probability,
            "confidence": self.confidence,
            "headline": self.headline,
            "summary": self.summary,
            "asset_recommendations": [
                {
                    "symbol": r.symbol,
                    "current_weight": r.current_weight,
                    "target_weight": r.target_weight,
                    "action": r.action,
                    "urgency": r.urgency,
                    "rationale": r.rationale,
                }
                for r in self.asset_recommendations
            ],
            "key_drivers": self.key_drivers,
            "historical_context": self.historical_context,
            "suggested_timeline": self.suggested_timeline,
            "estimated_turnover": self.estimated_turnover,
            "metadata": self.metadata,
        }


class RecommendationEngine:
    """
    Generate portfolio recommendations based on model predictions.

    Maps probability scores to actionable recommendations with
    constraints on turnover and position limits.
    """

    # Default asset allocation
    DEFAULT_WEIGHTS = {
        "SPY": 0.40,
        "VEU": 0.20,
        "AGG": 0.25,
        "DJP": 0.10,
        "VNQ": 0.05,
    }

    # Asset characteristics for recommendations
    ASSET_CHARACTERISTICS = {
        "SPY": {"type": "equity", "risk": "high", "liquidity": "high"},
        "VEU": {"type": "equity", "risk": "high", "liquidity": "medium"},
        "AGG": {"type": "bond", "risk": "low", "liquidity": "high"},
        "DJP": {"type": "commodity", "risk": "medium", "liquidity": "medium"},
        "VNQ": {"type": "reit", "risk": "high", "liquidity": "medium"},
        "GLD": {"type": "commodity", "risk": "medium", "liquidity": "high"},
        "TLT": {"type": "bond", "risk": "medium", "liquidity": "high"},
        "CASH": {"type": "cash", "risk": "none", "liquidity": "high"},
    }

    # Alert thresholds
    ALERT_THRESHOLDS = {
        AlertLevel.NONE: (0.0, 0.3),
        AlertLevel.LOW: (0.3, 0.5),
        AlertLevel.MODERATE: (0.5, 0.7),
        AlertLevel.HIGH: (0.7, 0.85),
        AlertLevel.CRITICAL: (0.85, 1.0),
    }

    def __init__(
        self,
        target_weights: dict[str, float] | None = None,
        max_daily_turnover: float = 0.20,
        min_position_size: float = 0.02,
        max_cash_allocation: float = 0.50,
    ):
        """
        Initialize recommendation engine.

        Args:
            target_weights: Normal target allocation
            max_daily_turnover: Maximum portfolio turnover per day
            min_position_size: Minimum position size
            max_cash_allocation: Maximum cash allocation
        """
        self.target_weights = target_weights or self.DEFAULT_WEIGHTS
        self.max_daily_turnover = max_daily_turnover
        self.min_position_size = min_position_size
        self.max_cash_allocation = max_cash_allocation

    def generate_recommendation(
        self,
        probability: float,
        current_weights: dict[str, float],
        feature_contributions: dict[str, float] | None = None,
        model_confidence: float | None = None,
        timestamp: pd.Timestamp | None = None,
    ) -> Recommendation:
        """
        Generate portfolio recommendation based on model output.

        Args:
            probability: Drawdown probability (0-1)
            current_weights: Current portfolio weights
            feature_contributions: Feature importance/SHAP values
            model_confidence: Model confidence score
            timestamp: Recommendation timestamp

        Returns:
            Complete recommendation object
        """
        timestamp = timestamp or pd.Timestamp.now()
        confidence = model_confidence or self._estimate_confidence(probability)

        # Determine alert level
        alert_level = self._get_alert_level(probability)

        # Calculate target weights based on alert level
        target_weights = self._calculate_target_weights(
            alert_level, probability, current_weights
        )

        # Generate asset-level recommendations
        asset_recommendations = self._generate_asset_recommendations(
            current_weights, target_weights, alert_level
        )

        # Calculate turnover
        turnover = self._calculate_turnover(current_weights, target_weights)

        # Generate key drivers
        key_drivers = self._format_key_drivers(feature_contributions)

        # Generate narrative elements
        headline = self._generate_headline(alert_level, probability)
        summary = self._generate_summary(
            alert_level, probability, key_drivers, turnover
        )
        historical_context = self._generate_historical_context(
            probability, feature_contributions
        )
        timeline = self._generate_timeline(alert_level, turnover)

        return Recommendation(
            timestamp=timestamp,
            alert_level=alert_level,
            probability=probability,
            confidence=confidence,
            headline=headline,
            summary=summary,
            asset_recommendations=asset_recommendations,
            key_drivers=key_drivers,
            historical_context=historical_context,
            suggested_timeline=timeline,
            estimated_turnover=turnover,
            metadata={
                "target_weights": target_weights,
                "model_probability": probability,
            },
        )

    def _get_alert_level(self, probability: float) -> AlertLevel:
        """Map probability to alert level."""
        for level, (low, high) in self.ALERT_THRESHOLDS.items():
            if low <= probability < high:
                return level
        return AlertLevel.CRITICAL if probability >= 0.85 else AlertLevel.NONE

    def _calculate_target_weights(
        self,
        alert_level: AlertLevel,
        probability: float,
        current_weights: dict[str, float],
    ) -> dict[str, float]:
        """Calculate target weights based on alert level."""
        if alert_level == AlertLevel.NONE:
            return self.target_weights.copy()

        # De-risking intensity based on probability
        intensity = self._calculate_derisk_intensity(probability)

        target_weights = {}
        equity_reduction = 0.0

        for asset, normal_weight in self.target_weights.items():
            characteristics = self.ASSET_CHARACTERISTICS.get(asset, {})
            asset_type = characteristics.get("type", "other")

            if asset_type == "equity":
                # Reduce equity exposure
                reduction = normal_weight * intensity * 0.6
                target_weights[asset] = max(
                    normal_weight - reduction, self.min_position_size
                )
                equity_reduction += reduction

            elif asset_type == "reit":
                # REITs also reduced (correlated with equity)
                reduction = normal_weight * intensity * 0.5
                target_weights[asset] = max(
                    normal_weight - reduction, self.min_position_size
                )
                equity_reduction += reduction

            elif asset_type == "bond":
                # Increase high-quality bonds
                increase = normal_weight * intensity * 0.3
                target_weights[asset] = min(normal_weight + increase, 0.40)

            elif asset_type == "commodity":
                # Reduce commodities slightly
                reduction = normal_weight * intensity * 0.3
                target_weights[asset] = max(
                    normal_weight - reduction, self.min_position_size
                )
                equity_reduction += reduction

            else:
                target_weights[asset] = normal_weight

        # Allocate freed capital to cash
        cash_allocation = equity_reduction * 0.7  # 70% to cash
        target_weights["CASH"] = min(
            current_weights.get("CASH", 0) + cash_allocation,
            self.max_cash_allocation,
        )

        # Normalize weights
        total = sum(target_weights.values())
        if total > 0:
            target_weights = {k: v / total for k, v in target_weights.items()}

        return target_weights

    def _calculate_derisk_intensity(self, probability: float) -> float:
        """Calculate de-risking intensity (0-1) based on probability."""
        if probability < 0.3:
            return 0.0
        elif probability < 0.5:
            return 0.2
        elif probability < 0.7:
            return 0.4
        elif probability < 0.85:
            return 0.6
        else:
            return 0.8

    def _generate_asset_recommendations(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        alert_level: AlertLevel,
    ) -> list[AssetRecommendation]:
        """Generate recommendations for each asset."""
        recommendations = []

        all_assets = set(current_weights.keys()) | set(target_weights.keys())

        for asset in all_assets:
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            delta = target - current

            # Determine action
            if abs(delta) < 0.01:
                action = "hold"
            elif delta > 0:
                action = "buy"
            else:
                action = "sell"

            # Determine urgency
            if alert_level in [AlertLevel.CRITICAL, AlertLevel.HIGH]:
                urgency = "immediate"
            elif alert_level == AlertLevel.MODERATE:
                urgency = "gradual"
            else:
                urgency = "optional"

            # Generate rationale
            rationale = self._generate_asset_rationale(
                asset, current, target, alert_level
            )

            recommendations.append(
                AssetRecommendation(
                    symbol=asset,
                    current_weight=current,
                    target_weight=target,
                    action=action,
                    urgency=urgency,
                    rationale=rationale,
                )
            )

        # Sort by magnitude of change
        recommendations.sort(
            key=lambda r: abs(r.target_weight - r.current_weight), reverse=True
        )

        return recommendations

    def _generate_asset_rationale(
        self,
        asset: str,
        current: float,
        target: float,
        alert_level: AlertLevel,
    ) -> str:
        """Generate rationale for asset recommendation."""
        characteristics = self.ASSET_CHARACTERISTICS.get(asset, {})
        asset_type = characteristics.get("type", "other")
        delta = target - current

        if abs(delta) < 0.01:
            return "Maintain current allocation"

        if asset_type == "equity":
            if delta < 0:
                return f"Reduce equity exposure by {abs(delta)*100:.1f}% to limit drawdown risk"
            else:
                return f"Increase equity allocation as risk conditions improve"

        elif asset_type == "bond":
            if delta > 0:
                return f"Increase bond allocation for defensive positioning"
            else:
                return f"Reduce bonds to normalize allocation"

        elif asset_type == "cash":
            if delta > 0:
                return f"Raise cash to {target*100:.1f}% for risk protection"
            else:
                return f"Deploy cash as market conditions stabilize"

        elif asset_type == "commodity":
            if delta < 0:
                return f"Reduce commodity exposure during risk-off period"
            else:
                return f"Add commodities for diversification"

        elif asset_type == "reit":
            if delta < 0:
                return f"Reduce REIT exposure (correlated with equity risk)"
            else:
                return f"Restore REIT allocation"

        return f"Adjust allocation based on risk assessment"

    def _calculate_turnover(
        self, current: dict[str, float], target: dict[str, float]
    ) -> float:
        """Calculate portfolio turnover."""
        all_assets = set(current.keys()) | set(target.keys())
        turnover = sum(
            abs(target.get(a, 0) - current.get(a, 0)) for a in all_assets
        )
        return turnover / 2  # One-way turnover

    def _estimate_confidence(self, probability: float) -> float:
        """Estimate confidence based on probability extremity."""
        # More confident at extreme probabilities
        if probability < 0.2 or probability > 0.8:
            return 0.8
        elif probability < 0.3 or probability > 0.7:
            return 0.7
        else:
            return 0.6

    def _format_key_drivers(
        self, feature_contributions: dict[str, float] | None
    ) -> list[dict[str, Any]]:
        """Format feature contributions as key drivers."""
        if not feature_contributions:
            return []

        # Sort by absolute contribution
        sorted_features = sorted(
            feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        drivers = []
        for feature, contribution in sorted_features[:5]:
            drivers.append(
                {
                    "feature": feature,
                    "contribution": contribution,
                    "direction": "increasing risk" if contribution > 0 else "decreasing risk",
                    "magnitude": "high" if abs(contribution) > 0.1 else "moderate",
                }
            )

        return drivers

    def _generate_headline(self, alert_level: AlertLevel, probability: float) -> str:
        """Generate headline for recommendation."""
        headlines = {
            AlertLevel.NONE: "Market conditions stable - maintain allocation",
            AlertLevel.LOW: "Minor stress signals detected - monitor closely",
            AlertLevel.MODERATE: "Elevated risk indicators - consider reducing exposure",
            AlertLevel.HIGH: "Significant drawdown risk - de-risk recommended",
            AlertLevel.CRITICAL: "ALERT: High probability of drawdown - immediate action advised",
        }
        return headlines.get(alert_level, "Market assessment in progress")

    def _generate_summary(
        self,
        alert_level: AlertLevel,
        probability: float,
        key_drivers: list[dict],
        turnover: float,
    ) -> str:
        """Generate summary paragraph."""
        prob_pct = probability * 100

        if alert_level == AlertLevel.NONE:
            return (
                f"Our models indicate a {prob_pct:.0f}% probability of significant drawdown "
                f"over the next 10-20 days. Current market conditions are favorable and "
                f"no portfolio adjustments are recommended at this time."
            )

        driver_text = ""
        if key_drivers:
            top_drivers = [d["feature"].replace("_", " ") for d in key_drivers[:3]]
            driver_text = f" Key risk factors include: {', '.join(top_drivers)}."

        if alert_level == AlertLevel.LOW:
            return (
                f"Early warning indicators show a {prob_pct:.0f}% drawdown probability. "
                f"While not yet actionable, we recommend heightened monitoring.{driver_text}"
            )

        elif alert_level == AlertLevel.MODERATE:
            return (
                f"Risk models indicate a {prob_pct:.0f}% probability of a 5%+ portfolio "
                f"drawdown. We recommend gradually reducing equity exposure over the next "
                f"3-5 trading days.{driver_text} Estimated portfolio turnover: {turnover*100:.1f}%."
            )

        elif alert_level == AlertLevel.HIGH:
            return (
                f"WARNING: Elevated {prob_pct:.0f}% probability of significant drawdown. "
                f"Recommend reducing equity exposure and raising cash allocation over the "
                f"next 1-2 trading days.{driver_text} Estimated turnover: {turnover*100:.1f}%."
            )

        else:  # CRITICAL
            return (
                f"CRITICAL ALERT: {prob_pct:.0f}% probability of major drawdown. "
                f"Immediate de-risking strongly recommended. Reduce equity to minimum levels "
                f"and maximize defensive positions.{driver_text} "
                f"Estimated turnover: {turnover*100:.1f}%."
            )

    def _generate_historical_context(
        self,
        probability: float,
        feature_contributions: dict[str, float] | None,
    ) -> str:
        """Generate historical context for current conditions."""
        if probability < 0.5:
            return (
                "Current market conditions are within normal historical ranges. "
                "Similar configurations have typically been followed by stable or "
                "positive market performance."
            )

        elif probability < 0.7:
            return (
                "Current indicators resemble pre-correction periods such as late 2018 "
                "and early 2022. Historical analysis suggests elevated but not extreme risk."
            )

        else:
            return (
                "Current stress levels are comparable to periods preceding significant "
                "market dislocations (2020 COVID crash, 2008 crisis onset). Historical "
                "precedent suggests high probability of near-term volatility."
            )

    def _generate_timeline(self, alert_level: AlertLevel, turnover: float) -> str:
        """Generate suggested action timeline."""
        if alert_level == AlertLevel.NONE:
            return "No action required. Next assessment in 24 hours."

        elif alert_level == AlertLevel.LOW:
            return "Monitor daily. Prepare contingency de-risking plan."

        elif alert_level == AlertLevel.MODERATE:
            return (
                f"Execute {turnover*100:.1f}% turnover over 3-5 trading days. "
                f"Daily rebalancing of ~{turnover*100/4:.1f}% recommended."
            )

        elif alert_level == AlertLevel.HIGH:
            return (
                f"Execute {turnover*100:.1f}% turnover over 1-2 trading days. "
                f"Priority on reducing highest-risk positions first."
            )

        else:  # CRITICAL
            return (
                f"Immediate execution recommended. Target completion within 24 hours "
                f"using limit orders to minimize market impact."
            )

    def apply_constraints(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Apply portfolio constraints to target weights.

        Args:
            target_weights: Unconstrained target weights
            current_weights: Current portfolio weights

        Returns:
            Constrained target weights
        """
        constrained = target_weights.copy()

        # Apply maximum turnover constraint
        turnover = self._calculate_turnover(current_weights, constrained)

        if turnover > self.max_daily_turnover:
            # Scale back changes to meet turnover limit
            scale = self.max_daily_turnover / turnover

            for asset in constrained:
                current = current_weights.get(asset, 0)
                target = constrained[asset]
                delta = target - current
                constrained[asset] = current + delta * scale

        # Ensure minimum position sizes
        for asset in constrained:
            if 0 < constrained[asset] < self.min_position_size:
                constrained[asset] = 0

        # Ensure cash doesn't exceed maximum
        if constrained.get("CASH", 0) > self.max_cash_allocation:
            excess = constrained["CASH"] - self.max_cash_allocation
            constrained["CASH"] = self.max_cash_allocation

            # Redistribute excess to bonds
            if "AGG" in constrained:
                constrained["AGG"] += excess

        # Normalize
        total = sum(constrained.values())
        if total > 0:
            constrained = {k: v / total for k, v in constrained.items()}

        return constrained

    def get_action_plan(
        self, recommendation: Recommendation
    ) -> list[dict[str, Any]]:
        """
        Convert recommendation to executable action plan.

        Args:
            recommendation: Portfolio recommendation

        Returns:
            List of actions to execute
        """
        actions = []

        # Sort recommendations: sells first, then buys
        sells = [
            r for r in recommendation.asset_recommendations if r.action == "sell"
        ]
        buys = [
            r for r in recommendation.asset_recommendations if r.action == "buy"
        ]

        # Add sell orders first (to free up cash)
        for rec in sells:
            if abs(rec.target_weight - rec.current_weight) >= 0.01:
                actions.append(
                    {
                        "order": len(actions) + 1,
                        "action": "sell",
                        "symbol": rec.symbol,
                        "from_weight": rec.current_weight,
                        "to_weight": rec.target_weight,
                        "change_pct": (rec.target_weight - rec.current_weight) * 100,
                        "urgency": rec.urgency,
                        "rationale": rec.rationale,
                    }
                )

        # Then add buy orders
        for rec in buys:
            if abs(rec.target_weight - rec.current_weight) >= 0.01:
                actions.append(
                    {
                        "order": len(actions) + 1,
                        "action": "buy",
                        "symbol": rec.symbol,
                        "from_weight": rec.current_weight,
                        "to_weight": rec.target_weight,
                        "change_pct": (rec.target_weight - rec.current_weight) * 100,
                        "urgency": rec.urgency,
                        "rationale": rec.rationale,
                    }
                )

        return actions
