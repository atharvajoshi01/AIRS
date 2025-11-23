"""Pricing service for model inference."""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from dgmlib.models.dgm import DGMNet
from dgmlib.training.metrics import compute_greeks
from dgmlib.utils.numerics import black_scholes_analytical


class PricingService:
    """Service for option pricing using trained DGM models."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        """
        Initialize pricing service.

        Args:
            checkpoint_path: Path to trained model checkpoint.
            device: Device for inference.
        """
        self.device = device
        self.model: Optional[nn.Module] = None

        if checkpoint_path is not None:
            self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract model config from checkpoint if available
        config = checkpoint.get("config", None)

        if config is not None:
            # Create model from config
            from scripts.train import create_model
            self.model = create_model(config)
        else:
            # Assume default architecture
            self.model = DGMNet(input_dim=2, hidden_dim=50, num_layers=3)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def price(
        self,
        S: float,
        t: float = 0.0,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Price an option using the loaded model.

        Args:
            S: Stock price.
            t: Current time (default 0).
            **kwargs: Additional state variables (e.g., v for Heston).

        Returns:
            Dictionary with price and optionally Greeks.
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        # Build input tensor
        inputs = torch.tensor([[t, S]], dtype=torch.float32, device=self.device)

        # Add additional state variables if provided
        for key in sorted(kwargs.keys()):
            val = torch.tensor([[kwargs[key]]], dtype=torch.float32, device=self.device)
            inputs = torch.cat([inputs, val], dim=1)

        # Inference
        with torch.no_grad():
            price = self.model(inputs).item()

        result = {"price": price}

        # Compute Greeks if 2D problem
        if inputs.shape[1] == 2:
            try:
                greeks = compute_greeks(self.model, inputs)
                result["delta"] = greeks["delta"].item()
                result["gamma"] = greeks["gamma"].item()
            except Exception:
                pass  # Greeks computation may fail for some models

        return result

    def price_analytical(
        self,
        S: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        option_type: str = "call",
    ) -> Dict[str, float]:
        """
        Price using analytical Black-Scholes formula.

        Args:
            S: Stock price.
            K: Strike.
            r: Risk-free rate.
            sigma: Volatility.
            T: Time to maturity.
            option_type: "call" or "put".

        Returns:
            Dictionary with price and Greeks.
        """
        price, delta, gamma = black_scholes_analytical(
            S, K, r, sigma, T, option_type
        )

        return {
            "price": float(price),
            "delta": float(delta),
            "gamma": float(gamma),
        }
