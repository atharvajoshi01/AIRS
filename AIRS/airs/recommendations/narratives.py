"""
Natural language narrative generation for AIRS.

Converts model outputs and recommendations into human-readable narratives.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from airs.recommendations.engine import AlertLevel, Recommendation
from airs.recommendations.explainability import FeatureAttribution
from airs.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MarketContext:
    """Current market context for narrative generation."""

    vix_level: float | None = None
    yield_curve_slope: float | None = None
    credit_spread: float | None = None
    recent_drawdown: float | None = None
    days_since_high: int | None = None


class NarrativeGenerator:
    """
    Generate natural language narratives for recommendations.

    Creates human-readable summaries, explanations, and action plans.
    """

    # Templates for different alert levels
    HEADLINE_TEMPLATES = {
        AlertLevel.NONE: [
            "Market conditions remain stable",
            "No significant stress signals detected",
            "Risk indicators within normal range",
        ],
        AlertLevel.LOW: [
            "Early warning signals warrant attention",
            "Minor risk indicators elevated",
            "Monitoring recommended as conditions shift",
        ],
        AlertLevel.MODERATE: [
            "Risk indicators suggest caution",
            "Multiple stress signals active",
            "Consider defensive positioning",
        ],
        AlertLevel.HIGH: [
            "Significant drawdown risk identified",
            "Multiple warning flags triggered",
            "Defensive action recommended",
        ],
        AlertLevel.CRITICAL: [
            "ALERT: High probability of market stress",
            "CRITICAL: Immediate risk mitigation advised",
            "WARNING: Multiple crisis indicators active",
        ],
    }

    # Feature category mappings
    FEATURE_CATEGORIES = {
        "rates": ["yield", "curve", "rate", "treasury", "duration"],
        "credit": ["spread", "hy", "ig", "credit", "default"],
        "volatility": ["vix", "vol", "variance", "skew"],
        "macro": ["lei", "pmi", "claims", "sentiment", "recession"],
        "technical": ["momentum", "trend", "correlation", "regime"],
    }

    def __init__(self):
        """Initialize narrative generator."""
        self.template_idx = 0

    def generate_full_narrative(
        self,
        recommendation: Recommendation,
        attributions: list[FeatureAttribution] | None = None,
        market_context: MarketContext | None = None,
    ) -> dict[str, str]:
        """
        Generate complete narrative package.

        Args:
            recommendation: Portfolio recommendation
            attributions: Feature attributions from explainer
            market_context: Current market context

        Returns:
            Dictionary with narrative components
        """
        return {
            "headline": self.generate_headline(recommendation.alert_level),
            "executive_summary": self.generate_executive_summary(
                recommendation, attributions
            ),
            "risk_assessment": self.generate_risk_assessment(
                recommendation, attributions
            ),
            "key_drivers": self.generate_driver_narrative(attributions),
            "action_plan": self.generate_action_narrative(recommendation),
            "historical_context": self.generate_historical_narrative(
                recommendation, market_context
            ),
            "timeline": self.generate_timeline_narrative(recommendation),
            "caveats": self.generate_caveats(recommendation),
        }

    def generate_headline(self, alert_level: AlertLevel) -> str:
        """Generate headline for alert level."""
        templates = self.HEADLINE_TEMPLATES.get(alert_level, ["Market update"])
        # Rotate through templates
        self.template_idx = (self.template_idx + 1) % len(templates)
        return templates[self.template_idx]

    def generate_executive_summary(
        self,
        recommendation: Recommendation,
        attributions: list[FeatureAttribution] | None = None,
    ) -> str:
        """Generate executive summary paragraph."""
        prob = recommendation.probability * 100
        alert = recommendation.alert_level.value

        # Opening statement
        if recommendation.alert_level == AlertLevel.NONE:
            opening = (
                f"Our multi-factor risk model currently indicates a {prob:.0f}% "
                f"probability of significant portfolio drawdown over the next 10-20 days. "
                f"This is within normal historical ranges and suggests stable market conditions."
            )
        elif recommendation.alert_level in [AlertLevel.LOW, AlertLevel.MODERATE]:
            opening = (
                f"Our risk assessment system has flagged {alert} risk conditions with a "
                f"{prob:.0f}% probability of drawdown. While not yet critical, these "
                f"conditions warrant attention and potentially gradual defensive positioning."
            )
        else:
            opening = (
                f"IMPORTANT: Our models indicate {alert} risk conditions with a {prob:.0f}% "
                f"probability of significant market stress. We recommend reviewing portfolio "
                f"exposure and considering defensive adjustments."
            )

        # Add key driver summary
        if attributions and len(attributions) > 0:
            top_contributors = [a for a in attributions[:3] if a.contribution > 0]
            if top_contributors:
                driver_names = [
                    a.feature_name.replace("_", " ") for a in top_contributors
                ]
                opening += f" Key drivers include: {', '.join(driver_names)}."

        # Add recommendation summary
        if recommendation.estimated_turnover > 0.01:
            opening += (
                f" Recommended portfolio adjustment involves approximately "
                f"{recommendation.estimated_turnover*100:.1f}% turnover."
            )

        return opening

    def generate_risk_assessment(
        self,
        recommendation: Recommendation,
        attributions: list[FeatureAttribution] | None = None,
    ) -> str:
        """Generate detailed risk assessment."""
        sections = []

        # Overall risk level
        prob = recommendation.probability * 100
        sections.append(f"**Overall Risk Level: {recommendation.alert_level.value.upper()}**")
        sections.append(f"- Drawdown probability: {prob:.1f}%")
        sections.append(f"- Model confidence: {recommendation.confidence*100:.0f}%")

        # Category breakdown
        if attributions:
            category_risks = self._categorize_risks(attributions)

            if category_risks:
                sections.append("\n**Risk by Category:**")
                for category, risk_level in category_risks.items():
                    sections.append(f"- {category.title()}: {risk_level}")

        # Add specific concerns
        concerns = self._identify_concerns(attributions)
        if concerns:
            sections.append("\n**Specific Concerns:**")
            for concern in concerns:
                sections.append(f"- {concern}")

        return "\n".join(sections)

    def _categorize_risks(
        self, attributions: list[FeatureAttribution]
    ) -> dict[str, str]:
        """Categorize risk contributions by feature type."""
        category_scores = {cat: 0.0 for cat in self.FEATURE_CATEGORIES}

        for attr in attributions:
            feature_lower = attr.feature_name.lower()
            for category, keywords in self.FEATURE_CATEGORIES.items():
                if any(kw in feature_lower for kw in keywords):
                    category_scores[category] += attr.contribution
                    break

        # Convert to risk levels
        category_risks = {}
        for category, score in category_scores.items():
            if score > 0.15:
                category_risks[category] = "HIGH"
            elif score > 0.05:
                category_risks[category] = "ELEVATED"
            elif score > 0:
                category_risks[category] = "MODERATE"
            else:
                category_risks[category] = "LOW"

        return category_risks

    def _identify_concerns(
        self, attributions: list[FeatureAttribution] | None
    ) -> list[str]:
        """Identify specific concerns from feature attributions."""
        if not attributions:
            return []

        concerns = []

        for attr in attributions[:5]:
            if attr.contribution <= 0:
                continue

            name = attr.feature_name.lower()

            if "vix" in name and attr.value > 25:
                concerns.append(
                    f"Elevated volatility (VIX at {attr.value:.1f}) signals market stress"
                )
            elif "curve" in name and "inversion" in name and attr.value > 0:
                concerns.append("Yield curve inversion historically precedes recessions")
            elif "spread" in name and "hy" in name and attr.value > 450:
                concerns.append(
                    f"High-yield spreads widening ({attr.value:.0f}bps) indicates credit stress"
                )
            elif "regime" in name and "high_vol" in name and attr.value > 0.6:
                concerns.append("High probability of volatile regime persisting")
            elif "recession" in name and attr.value > 0.3:
                concerns.append(
                    f"Elevated recession probability ({attr.value*100:.0f}%)"
                )

        return concerns

    def generate_driver_narrative(
        self, attributions: list[FeatureAttribution] | None
    ) -> str:
        """Generate narrative explaining key drivers."""
        if not attributions:
            return "Detailed driver analysis unavailable."

        # Separate positive and negative contributors
        risk_increasing = [a for a in attributions if a.contribution > 0.01]
        risk_decreasing = [a for a in attributions if a.contribution < -0.01]

        sections = []

        # Risk increasing factors
        if risk_increasing:
            sections.append("**Factors Increasing Risk:**")
            for attr in risk_increasing[:4]:
                desc = self._get_feature_description(attr)
                sections.append(f"- {desc}")

        # Risk decreasing factors
        if risk_decreasing:
            sections.append("\n**Factors Moderating Risk:**")
            for attr in risk_decreasing[:3]:
                desc = self._get_feature_description(attr, positive=False)
                sections.append(f"- {desc}")

        return "\n".join(sections)

    def _get_feature_description(
        self, attr: FeatureAttribution, positive: bool = True
    ) -> str:
        """Generate description for a feature attribution."""
        name = attr.feature_name.replace("_", " ").title()
        value = attr.value
        contribution = abs(attr.contribution) * 100

        if positive:
            return f"{name} at {value:.2f} (contributing +{contribution:.1f}% to risk)"
        else:
            return f"{name} at {value:.2f} (reducing risk by {contribution:.1f}%)"

    def generate_action_narrative(self, recommendation: Recommendation) -> str:
        """Generate action plan narrative."""
        if recommendation.alert_level == AlertLevel.NONE:
            return (
                "**Recommended Action: MAINTAIN**\n\n"
                "No portfolio adjustments recommended at this time. Continue monitoring "
                "daily risk signals and maintain current strategic allocation."
            )

        sections = ["**Recommended Actions:**\n"]

        # Group recommendations by action type
        sells = []
        buys = []
        holds = []

        for rec in recommendation.asset_recommendations:
            if rec.action == "sell":
                sells.append(rec)
            elif rec.action == "buy":
                buys.append(rec)
            else:
                holds.append(rec)

        # Sell recommendations
        if sells:
            sections.append("*Reduce Exposure:*")
            for rec in sells:
                change = (rec.current_weight - rec.target_weight) * 100
                sections.append(
                    f"- **{rec.symbol}**: Reduce from {rec.current_weight*100:.1f}% to "
                    f"{rec.target_weight*100:.1f}% (-{change:.1f}pp)"
                )
                sections.append(f"  Rationale: {rec.rationale}")
            sections.append("")

        # Buy recommendations
        if buys:
            sections.append("*Increase Allocation:*")
            for rec in buys:
                change = (rec.target_weight - rec.current_weight) * 100
                sections.append(
                    f"- **{rec.symbol}**: Increase from {rec.current_weight*100:.1f}% to "
                    f"{rec.target_weight*100:.1f}% (+{change:.1f}pp)"
                )
                sections.append(f"  Rationale: {rec.rationale}")
            sections.append("")

        # Summary
        sections.append(
            f"*Total Estimated Turnover: {recommendation.estimated_turnover*100:.1f}%*"
        )

        return "\n".join(sections)

    def generate_historical_narrative(
        self,
        recommendation: Recommendation,
        market_context: MarketContext | None = None,
    ) -> str:
        """Generate historical context narrative."""
        prob = recommendation.probability

        if prob < 0.3:
            base_narrative = (
                "Current market conditions are consistent with typical low-volatility "
                "periods observed historically. Similar configurations over the past "
                "decade have generally been followed by stable or positive returns."
            )
        elif prob < 0.5:
            base_narrative = (
                "The current risk profile resembles conditions seen during minor "
                "corrections such as Q4 2018 or early 2016. While elevated from baseline, "
                "these periods typically resulted in limited drawdowns of 5-10%."
            )
        elif prob < 0.7:
            base_narrative = (
                "Risk indicators are reaching levels comparable to pre-correction periods "
                "such as late 2018 (before the Q4 selloff) and early 2020 (before COVID). "
                "Historical precedent suggests meaningful but not extreme drawdown risk."
            )
        else:
            base_narrative = (
                "Current stress levels are approaching those seen before significant "
                "market dislocations. The combination of factors is reminiscent of "
                "conditions preceding the 2020 COVID crash and 2008 financial crisis. "
                "Defensive positioning is strongly recommended."
            )

        # Add specific context if available
        if market_context:
            additions = []

            if market_context.vix_level and market_context.vix_level > 25:
                additions.append(
                    f"VIX at {market_context.vix_level:.1f} indicates elevated fear levels."
                )

            if market_context.yield_curve_slope and market_context.yield_curve_slope < 0:
                additions.append(
                    "The inverted yield curve has historically preceded recessions "
                    "by 6-18 months."
                )

            if market_context.days_since_high and market_context.days_since_high > 20:
                additions.append(
                    f"Market has been below recent highs for {market_context.days_since_high} days."
                )

            if additions:
                base_narrative += "\n\n" + " ".join(additions)

        return base_narrative

    def generate_timeline_narrative(self, recommendation: Recommendation) -> str:
        """Generate execution timeline narrative."""
        if recommendation.alert_level == AlertLevel.NONE:
            return "No immediate action required. Next scheduled review: 24 hours."

        turnover = recommendation.estimated_turnover

        if recommendation.alert_level == AlertLevel.LOW:
            return (
                "**Timeline: 1-2 Weeks**\n\n"
                "Gradual positioning adjustment recommended. No immediate trades required. "
                "Continue monitoring and prepare contingency plans for potential escalation."
            )

        elif recommendation.alert_level == AlertLevel.MODERATE:
            daily_turnover = turnover / 4
            return (
                "**Timeline: 3-5 Trading Days**\n\n"
                f"Execute rebalancing gradually to minimize market impact:\n"
                f"- Day 1-2: Begin reducing highest-risk positions\n"
                f"- Day 3-4: Continue rebalancing (~{daily_turnover*100:.1f}%/day)\n"
                f"- Day 5: Complete defensive positioning\n\n"
                f"Use limit orders where possible to minimize execution costs."
            )

        elif recommendation.alert_level == AlertLevel.HIGH:
            return (
                "**Timeline: 1-2 Trading Days**\n\n"
                "Expedited execution recommended:\n"
                "- Day 1: Reduce equity exposure, raise cash\n"
                "- Day 2: Fine-tune defensive positioning\n\n"
                "Prioritize reducing highest-beta positions first. "
                "Consider market orders if liquidity permits."
            )

        else:  # CRITICAL
            return (
                "**Timeline: IMMEDIATE (Within 24 Hours)**\n\n"
                "Priority execution required:\n"
                "- First 2 hours: Exit highest-risk positions\n"
                "- Next 4 hours: Complete major rebalancing\n"
                "- End of day: Verify defensive positioning achieved\n\n"
                "Accept reasonable market impact to ensure timely execution. "
                "Consider using ETF baskets for efficient de-risking."
            )

    def generate_caveats(self, recommendation: Recommendation) -> str:
        """Generate appropriate caveats and disclaimers."""
        caveats = [
            "**Important Considerations:**",
            "",
            "- Model predictions are probabilistic estimates, not certainties",
            "- Past performance of the model does not guarantee future results",
            "- Transaction costs and market impact may affect execution",
            "- Individual circumstances may warrant different actions",
        ]

        if recommendation.confidence < 0.7:
            caveats.append(
                "- Model confidence is below typical levels; consider waiting for confirmation"
            )

        if recommendation.estimated_turnover > 0.15:
            caveats.append(
                "- High turnover may result in significant transaction costs"
            )

        caveats.extend([
            "",
            "*This analysis is for informational purposes only and does not constitute "
            "investment advice. Consult with a qualified financial advisor before making "
            "investment decisions.*"
        ])

        return "\n".join(caveats)

    def generate_email_report(
        self,
        recommendation: Recommendation,
        attributions: list[FeatureAttribution] | None = None,
    ) -> str:
        """
        Generate formatted email report.

        Args:
            recommendation: Portfolio recommendation
            attributions: Feature attributions

        Returns:
            Formatted email text
        """
        narrative = self.generate_full_narrative(recommendation, attributions)

        report = f"""
================================================================================
AIRS PORTFOLIO RISK REPORT
{datetime.now().strftime('%Y-%m-%d %H:%M')} UTC
================================================================================

{narrative['headline'].upper()}

{'-'*80}
EXECUTIVE SUMMARY
{'-'*80}

{narrative['executive_summary']}

{'-'*80}
RISK ASSESSMENT
{'-'*80}

{narrative['risk_assessment']}

{'-'*80}
KEY RISK DRIVERS
{'-'*80}

{narrative['key_drivers']}

{'-'*80}
RECOMMENDED ACTIONS
{'-'*80}

{narrative['action_plan']}

{'-'*80}
EXECUTION TIMELINE
{'-'*80}

{narrative['timeline']}

{'-'*80}
HISTORICAL CONTEXT
{'-'*80}

{narrative['historical_context']}

{'-'*80}

{narrative['caveats']}

================================================================================
End of Report
================================================================================
"""
        return report

    def generate_slack_message(
        self,
        recommendation: Recommendation,
    ) -> dict[str, Any]:
        """
        Generate Slack-formatted message.

        Args:
            recommendation: Portfolio recommendation

        Returns:
            Slack message payload
        """
        # Emoji based on alert level
        emojis = {
            AlertLevel.NONE: ":white_check_mark:",
            AlertLevel.LOW: ":warning:",
            AlertLevel.MODERATE: ":large_orange_diamond:",
            AlertLevel.HIGH: ":red_circle:",
            AlertLevel.CRITICAL: ":rotating_light:",
        }

        emoji = emojis.get(recommendation.alert_level, ":chart_with_downwards_trend:")
        prob = recommendation.probability * 100

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} AIRS Risk Alert: {recommendation.alert_level.value.upper()}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Drawdown Probability:* {prob:.0f}%\n*Confidence:* {recommendation.confidence*100:.0f}%",
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Summary:*\n{recommendation.summary}",
                },
            },
        ]

        # Add top recommendations
        if recommendation.asset_recommendations:
            action_text = "*Key Actions:*\n"
            for rec in recommendation.asset_recommendations[:3]:
                if rec.action != "hold":
                    action_text += f"• {rec.action.upper()} {rec.symbol}: {rec.current_weight*100:.0f}% → {rec.target_weight*100:.0f}%\n"

            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": action_text},
            })

        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Generated: {recommendation.timestamp.strftime('%Y-%m-%d %H:%M')} UTC | Turnover: {recommendation.estimated_turnover*100:.1f}%",
                }
            ],
        })

        return {"blocks": blocks}
