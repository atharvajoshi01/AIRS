#!/usr/bin/env python3
"""
Generate 3D pricing surface visualizations.

Creates beautiful plots showing how option prices vary across
stock price and time dimensions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from dgmlib.models.dgm import DGMNet
from dgmlib.utils.numerics import black_scholes_analytical


def create_pricing_surface(checkpoint_path: str, output_dir: str = "plots"):
    """Create 3D pricing surface visualization."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    dummy_bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    model = DGMNet(input_dim=2, hidden_dim=50, num_layers=3, input_bounds=dummy_bounds)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create grid
    t_values = np.linspace(0, 1, 50)
    S_values = np.linspace(60, 140, 50)
    T, S = np.meshgrid(t_values, S_values)
    
    # Compute DGM prices
    dgm_prices = np.zeros_like(T)
    bs_prices = np.zeros_like(T)
    
    for i in range(len(S_values)):
        for j in range(len(t_values)):
            t, S_val = T[i, j], S[i, j]
            inputs = torch.tensor([[t, S_val]], dtype=torch.float32)
            with torch.no_grad():
                dgm_prices[i, j] = model(inputs).item()
            
            bs_price, _, _ = black_scholes_analytical(
                S_val, 100.0, 0.05, 0.2, 1.0-t, 'call'
            )
            bs_prices[i, j] = bs_price
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 6))
    
    # DGM Surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(T, S, dgm_prices, cmap='viridis', alpha=0.9)
    ax1.set_xlabel('Time (t)', fontsize=12)
    ax1.set_ylabel('Stock Price (S)', fontsize=12)
    ax1.set_zlabel('Option Price', fontsize=12)
    ax1.set_title('DGM Neural Network Pricing Surface', fontsize=14, fontweight='bold')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # Black-Scholes Surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(T, S, bs_prices, cmap='plasma', alpha=0.9)
    ax2.set_xlabel('Time (t)', fontsize=12)
    ax2.set_ylabel('Stock Price (S)', fontsize=12)
    ax2.set_zlabel('Option Price', fontsize=12)
    ax2.set_title('Analytical Black-Scholes Surface', fontsize=14, fontweight='bold')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pricing_surfaces.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved pricing surfaces to {output_dir}/pricing_surfaces.png")
    
    # Error heatmap
    fig2, ax = plt.subplots(figsize=(10, 8))
    error = np.abs(dgm_prices - bs_prices)
    im = ax.contourf(T, S, error, levels=20, cmap='RdYlGn_r')
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Stock Price (S)', fontsize=12)
    ax.set_title('Absolute Pricing Error: |DGM - Black-Scholes|', 
                 fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Error ($)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved error heatmap to {output_dir}/error_heatmap.png")
    
    print(f"\nStatistics:")
    print(f"  Mean Absolute Error: ${error.mean():.4f}")
    print(f"  Max Absolute Error:  ${error.max():.4f}")
    print(f"  RMSE:                ${np.sqrt((error**2).mean()):.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/bs_european/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='plots',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    create_pricing_surface(args.checkpoint, args.output)
    print("\n✅ Visualization complete!")
