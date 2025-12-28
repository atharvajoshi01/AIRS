"""Feature engineering module for AIRS."""

from airs.features.base import FeatureGenerator
from airs.features.rates import RateFeatures
from airs.features.credit import CreditFeatures
from airs.features.volatility import VolatilityFeatures
from airs.features.macro import MacroFeatures
from airs.features.cross_asset import CrossAssetFeatures
from airs.features.regime import RegimeDetector
from airs.features.composite import CompositeFeatures
from airs.features.pipeline import FeaturePipeline

__all__ = [
    "FeatureGenerator",
    "RateFeatures",
    "CreditFeatures",
    "VolatilityFeatures",
    "MacroFeatures",
    "CrossAssetFeatures",
    "RegimeDetector",
    "CompositeFeatures",
    "FeaturePipeline",
]
