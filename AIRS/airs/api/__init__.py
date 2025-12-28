"""
REST API for AIRS.

FastAPI-based API for accessing alerts, recommendations, and features.
"""

from airs.api.main import create_app

__all__ = ["create_app"]
