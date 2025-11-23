"""Plotting utilities for visualization."""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_option_surface(
    S_grid: np.ndarray,
    t_grid: np.ndarray,
    V_grid: np.ndarray,
    title: str = "Option Value Surface",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot option value surface V(S, t).

    Args:
        S_grid: Stock price grid of shape (n_S,).
        t_grid: Time grid of shape (n_t,).
        V_grid: Value grid of shape (n_S, n_t).
        title: Plot title.
        save_path: Optional path to save figure.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')

    surf = ax.plot_surface(
        S_mesh,
        t_mesh,
        V_grid,
        cmap='viridis',
        alpha=0.8,
        edgecolor='none',
    )

    ax.set_xlabel('Stock Price (S)')
    ax.set_ylabel('Time (t)')
    ax.set_zlabel('Option Value (V)')
    ax.set_title(title)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_price_comparison(
    S_values: np.ndarray,
    prices_dict: dict,
    title: str = "Price Comparison",
    save_path: Optional[str] = None,
) -> None:
    """
    Compare prices from different methods.

    Args:
        S_values: Stock prices.
        prices_dict: Dictionary mapping method name to prices.
        title: Plot title.
        save_path: Optional save path.
    """
    plt.figure(figsize=(10, 6))

    for method_name, prices in prices_dict.items():
        plt.plot(S_values, prices, label=method_name, linewidth=2)

    plt.xlabel('Stock Price (S)', fontsize=12)
    plt.ylabel('Option Value (V)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_greeks(
    S_values: np.ndarray,
    delta: np.ndarray,
    gamma: np.ndarray,
    title: str = "Greeks",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot option Greeks.

    Args:
        S_values: Stock prices.
        delta: Delta values.
        gamma: Gamma values.
        title: Plot title.
        save_path: Optional save path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Delta
    ax1.plot(S_values, delta, 'b-', linewidth=2)
    ax1.set_xlabel('Stock Price (S)', fontsize=12)
    ax1.set_ylabel('Delta (∂V/∂S)', fontsize=12)
    ax1.set_title('Delta', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Gamma
    ax2.plot(S_values, gamma, 'r-', linewidth=2)
    ax2.set_xlabel('Stock Price (S)', fontsize=12)
    ax2.set_ylabel('Gamma (∂²V/∂S²)', fontsize=12)
    ax2.set_title('Gamma', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    losses: List[float],
    metrics: dict,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training loss and metrics.

    Args:
        losses: List of training losses.
        metrics: Dictionary of metrics over epochs.
        save_path: Optional save path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Metrics
    for metric_name, values in metrics.items():
        axes[1].plot(values, label=metric_name, linewidth=2)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Metric Value', fontsize=12)
    axes[1].set_title('Validation Metrics', fontsize=14)
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_residual_distribution(
    residuals: np.ndarray,
    title: str = "PDE Residual Distribution",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot histogram of PDE residuals.

    Args:
        residuals: Array of residual values.
        title: Plot title.
        save_path: Optional save path.
    """
    plt.figure(figsize=(10, 6))

    plt.hist(residuals, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')

    plt.xlabel('Residual Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
