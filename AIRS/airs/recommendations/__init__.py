"""
Recommendation engine for AIRS.

Provides actionable portfolio recommendations based on model predictions.
"""

from airs.recommendations.engine import RecommendationEngine, Recommendation
from airs.recommendations.explainability import ModelExplainer, FeatureAttribution
from airs.recommendations.narratives import NarrativeGenerator

__all__ = [
    "RecommendationEngine",
    "Recommendation",
    "ModelExplainer",
    "FeatureAttribution",
    "NarrativeGenerator",
]
