"""
Streamlit demo for interactive option pricing.

Usage:
    streamlit run ui/app.py
"""

import warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')

import numpy as np
import streamlit as st
import torch

from dgmlib.inference.service import PricingService
from dgmlib.utils.numerics import (
    black_scholes_analytical,
    finite_difference_european,
    monte_carlo_european,
)

st.set_page_config(
    page_title="Deep Galerkin Option Pricing",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Deep Galerkin Option Pricing")
st.markdown("Interactive demo for option pricing using physics-informed neural networks")

# Sidebar for parameters
st.sidebar.header("Option Parameters")

S = st.sidebar.slider("Stock Price (S)", 50.0, 200.0, 100.0, 1.0)
K = st.sidebar.slider("Strike Price (K)", 50.0, 200.0, 100.0, 1.0)
r = st.sidebar.slider("Risk-free Rate (r)", 0.0, 0.15, 0.05, 0.01)
sigma = st.sidebar.slider("Volatility (σ)", 0.05, 0.8, 0.2, 0.05)
T = st.sidebar.slider("Time to Maturity (T)", 0.1, 3.0, 1.0, 0.1)
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

st.sidebar.header("Pricing Methods")
use_bs = st.sidebar.checkbox("Black-Scholes (Analytical)", value=True)
use_mc = st.sidebar.checkbox("Monte Carlo", value=True)
use_fd = st.sidebar.checkbox("Finite Difference", value=False)
use_dgm = st.sidebar.checkbox("Deep Galerkin Method", value=False)

if use_dgm:
    checkpoint_path = st.sidebar.text_input(
        "DGM Checkpoint Path",
        "checkpoints/bs_european/best_model.pt",
    )

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Pricing Results")

    results = {}

    # Black-Scholes
    if use_bs:
        with st.spinner("Computing Black-Scholes..."):
            price, delta, gamma = black_scholes_analytical(
                S, K, r, sigma, T, option_type
            )
            results["Black-Scholes"] = {
                "Price": f"${price:.4f}",
                "Delta": f"{delta:.4f}",
                "Gamma": f"{gamma:.6f}",
            }

    # Monte Carlo
    if use_mc:
        with st.spinner("Running Monte Carlo simulation..."):
            price, std_err = monte_carlo_european(
                S, K, r, sigma, T, option_type, n_paths=50000
            )
            results["Monte Carlo"] = {
                "Price": f"${price:.4f}",
                "Std Error": f"${std_err:.4f}",
                "95% CI": f"[${price - 1.96*std_err:.4f}, ${price + 1.96*std_err:.4f}]",
            }

    # Finite Difference
    if use_fd:
        with st.spinner("Solving via Finite Difference..."):
            S_grid, t_grid, V_grid = finite_difference_european(
                K, r, sigma, T, option_type, n_space=100, n_time=50
            )
            S_idx = (abs(S_grid - S)).argmin()
            price_fd = V_grid[S_idx, 0]
            results["Finite Difference"] = {
                "Price": f"${price_fd:.4f}",
            }

    # Deep Galerkin Method
    if use_dgm:
        try:
            with st.spinner("Loading DGM model..."):
                service = PricingService(checkpoint_path)
                result = service.price(S=S, t=0.0)
                results["DGM"] = {
                    "Price": f"${result['price']:.4f}",
                }
                if "delta" in result:
                    results["DGM"]["Delta"] = f"{result['delta']:.4f}"
                if "gamma" in result:
                    results["DGM"]["Gamma"] = f"{result['gamma']:.6f}"
        except Exception as e:
            st.error(f"Error loading DGM model: {e}")

    # Display results
    for method, values in results.items():
        st.subheader(method)
        for key, val in values.items():
            st.metric(key, val)
        st.divider()

with col2:
    st.header("Price Surface")

    # Generate price surface
    S_range = np.linspace(max(50, S - 50), min(200, S + 50), 50)
    t_range = np.linspace(0, T, 30)

    S_mesh, t_mesh = np.meshgrid(S_range, t_range)
    V_surface = np.zeros_like(S_mesh)

    # Compute surface using Black-Scholes
    for i in range(len(t_range)):
        for j in range(len(S_range)):
            tau = T - t_range[i]
            if tau > 0:
                price, _, _ = black_scholes_analytical(
                    S_range[j], K, r, sigma, tau, option_type
                )
                V_surface[i, j] = price

    # Plot surface
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        S_mesh,
        t_mesh,
        V_surface,
        cmap="viridis",
        alpha=0.8,
        edgecolor="none",
    )

    ax.set_xlabel("Stock Price (S)")
    ax.set_ylabel("Time (t)")
    ax.set_zlabel("Option Value (V)")
    ax.set_title(f"{option_type.capitalize()} Option Value Surface")

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    st.pyplot(fig)

# Greeks visualization
st.header("Greeks Visualization")

S_range_greeks = np.linspace(max(50, K - 50), min(200, K + 50), 100)
delta_values = []
gamma_values = []

for S_val in S_range_greeks:
    _, delta, gamma = black_scholes_analytical(S_val, K, r, sigma, T, option_type)
    delta_values.append(delta)
    gamma_values.append(gamma)

fig_greeks, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(S_range_greeks, delta_values, "b-", linewidth=2)
ax1.axvline(x=S, color="red", linestyle="--", label="Current S")
ax1.axvline(x=K, color="green", linestyle="--", label="Strike K")
ax1.set_xlabel("Stock Price (S)")
ax1.set_ylabel("Delta")
ax1.set_title("Delta (∂V/∂S)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(S_range_greeks, gamma_values, "r-", linewidth=2)
ax2.axvline(x=S, color="red", linestyle="--", label="Current S")
ax2.axvline(x=K, color="green", linestyle="--", label="Strike K")
ax2.set_xlabel("Stock Price (S)")
ax2.set_ylabel("Gamma")
ax2.set_title("Gamma (∂²V/∂S²)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig_greeks)

# Footer
st.markdown("---")
st.markdown(
    """
    **About**: This application demonstrates option pricing using various methods including
    the Deep Galerkin Method (physics-informed neural networks for solving PDEs).

    **Methods**:
    - **Black-Scholes**: Analytical closed-form solution
    - **Monte Carlo**: Stochastic simulation with variance reduction
    - **Finite Difference**: Numerical PDE solver (Crank-Nicolson)
    - **Deep Galerkin Method**: Neural network-based PDE solver
    """
)
