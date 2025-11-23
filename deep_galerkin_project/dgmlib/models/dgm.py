"""
Deep Galerkin Method (DGM) neural network architecture.

Reference: Sirignano & Spiliopoulos (2018)
"DGM: A deep learning algorithm for solving partial differential equations"
"""

from typing import Optional

import torch
import torch.nn as nn


class DGMLayer(nn.Module):
    """
    Single DGM layer with gating mechanism.

    The DGM layer uses gates to control information flow:
    Z = σ(U_z·x + W_z·S + b_z)  # Update gate
    G = σ(U_g·x + W_g·S + b_g)  # Forget gate
    R = σ(U_r·x + W_r·S + b_r)  # Relevance gate
    H = φ(U_h·x + W_h·(S⊙R) + b_h)  # Candidate
    S_new = (1-G)⊙H + Z⊙S  # New state
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "tanh",
    ) -> None:
        """
        Initialize DGM layer.

        Args:
            input_dim: Dimension of input x.
            output_dim: Dimension of output/state S.
            activation: Activation function ("tanh" or "gelu").
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Gates: Z (update), G (forget), R (relevance)
        self.Uz = nn.Linear(input_dim, output_dim)
        self.Wz = nn.Linear(output_dim, output_dim, bias=False)

        self.Ug = nn.Linear(input_dim, output_dim)
        self.Wg = nn.Linear(output_dim, output_dim, bias=False)

        self.Ur = nn.Linear(input_dim, output_dim)
        self.Wr = nn.Linear(output_dim, output_dim, bias=False)

        # Candidate
        self.Uh = nn.Linear(input_dim, output_dim)
        self.Wh = nn.Linear(output_dim, output_dim, bias=False)

        # Activation
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DGM layer.

        Args:
            x: Input tensor of shape (batch, input_dim).
            S: State tensor of shape (batch, output_dim).

        Returns:
            Updated state of shape (batch, output_dim).
        """
        # Gates
        Z = torch.sigmoid(self.Uz(x) + self.Wz(S))  # Update gate
        G = torch.sigmoid(self.Ug(x) + self.Wg(S))  # Forget gate
        R = torch.sigmoid(self.Ur(x) + self.Wr(S))  # Relevance gate

        # Candidate
        H = self.activation(self.Uh(x) + self.Wh(S * R))

        # New state
        S_new = (1 - G) * H + Z * S

        return S_new


class DGMNet(nn.Module):
    """
    Deep Galerkin Method network with multiple DGM layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        num_layers: int = 3,
        activation: str = "tanh",
        use_spectral_norm: bool = False,
        dropout: float = 0.0,
        input_bounds: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize DGM network.

        Args:
            input_dim: Input dimension (e.g., 2 for [t, S]).
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (typically 1 for option value).
            num_layers: Number of DGM layers.
            activation: Activation function.
            use_spectral_norm: Apply spectral normalization.
            dropout: Dropout probability.
            input_bounds: Tensor of shape (input_dim, 2) with [min, max] for each input.
                         If None, no normalization is applied.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input normalization bounds
        if input_bounds is not None:
            self.register_buffer('input_lb', input_bounds[:, 0])
            self.register_buffer('input_ub', input_bounds[:, 1])
            self.use_normalization = True
        else:
            self.use_normalization = False

        # Initial layer: map input to hidden dimension
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # DGM layers
        self.dgm_layers = nn.ModuleList([
            DGMLayer(input_dim, hidden_dim, activation)
            for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Activation
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Spectral normalization
        if use_spectral_norm:
            self.input_layer = nn.utils.spectral_norm(self.input_layer)
            self.output_layer = nn.utils.spectral_norm(self.output_layer)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Output tensor of shape (batch, output_dim).
        """
        # Normalize inputs to [-1, 1]
        if self.use_normalization:
            x_norm = 2 * (x - self.input_lb) / (self.input_ub - self.input_lb) - 1
        else:
            x_norm = x

        # Initial transformation
        S = self.activation(self.input_layer(x_norm))

        # DGM layers
        for dgm_layer in self.dgm_layers:
            S = dgm_layer(x_norm, S)
            if self.dropout is not None:
                S = self.dropout(S)

        # Output
        out = self.output_layer(S)

        return out

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
