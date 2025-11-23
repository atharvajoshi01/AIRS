"""
Numerical methods for option pricing (baselines for comparison).

Includes:
- Monte Carlo simulation
- Finite Difference methods (Crank-Nicolson)
- Analytical Black-Scholes formula
"""

from typing import Tuple

import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import norm


def black_scholes_analytical(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
) -> Tuple[float, float, float]:
    """
    Analytical Black-Scholes formula for European options.

    Args:
        S: Current stock price.
        K: Strike price.
        r: Risk-free rate.
        sigma: Volatility.
        T: Time to maturity.
        option_type: "call" or "put".

    Returns:
        Tuple of (price, delta, gamma).
    """
    if T <= 0:
        # At maturity
        if option_type == "call":
            return max(S - K, 0.0), float(S > K), 0.0
        else:
            return max(K - S, 0.0), float(S < K), 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)

    # Gamma is same for call and put
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    return price, delta, gamma


def monte_carlo_european(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 100000,
    antithetic: bool = True,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Monte Carlo pricing for European options with variance reduction.

    Args:
        S0: Initial stock price.
        K: Strike price.
        r: Risk-free rate.
        sigma: Volatility.
        T: Time to maturity.
        option_type: "call" or "put".
        n_paths: Number of simulation paths.
        antithetic: Use antithetic variates for variance reduction.
        seed: Random seed.

    Returns:
        Tuple of (price, std_error).
    """
    np.random.seed(seed)

    if antithetic:
        n_paths = n_paths // 2

    # Generate random normal samples
    Z = np.random.randn(n_paths)

    # Simulate terminal stock prices
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Payoffs
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    # Antithetic variates
    if antithetic:
        ST_anti = S0 * np.exp((r - 0.5 * sigma**2) * T - sigma * np.sqrt(T) * Z)
        if option_type == "call":
            payoffs_anti = np.maximum(ST_anti - K, 0)
        else:
            payoffs_anti = np.maximum(K - ST_anti, 0)

        payoffs = (payoffs + payoffs_anti) / 2

    # Discount to present value
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(len(payoffs))

    return price, std_error


def finite_difference_european(
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    S_max: float = 200.0,
    n_space: int = 200,
    n_time: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finite difference (Crank-Nicolson) for European options.

    Args:
        K: Strike price.
        r: Risk-free rate.
        sigma: Volatility.
        T: Time to maturity.
        option_type: "call" or "put".
        S_max: Maximum stock price in grid.
        n_space: Number of spatial grid points.
        n_time: Number of time steps.

    Returns:
        Tuple of (S_grid, t_grid, V_grid) where V_grid[i, j] = V(S_i, t_j).
    """
    # Grid setup
    dS = S_max / n_space
    dt = T / n_time
    S = np.linspace(0, S_max, n_space + 1)
    t = np.linspace(0, T, n_time + 1)

    # Initialize solution
    V = np.zeros((n_space + 1, n_time + 1))

    # Terminal condition
    if option_type == "call":
        V[:, -1] = np.maximum(S - K, 0)
    else:
        V[:, -1] = np.maximum(K - S, 0)

    # Boundary conditions
    if option_type == "call":
        V[0, :] = 0  # S = 0
        V[-1, :] = S_max - K * np.exp(-r * (T - t))  # S = S_max
    else:
        V[0, :] = K * np.exp(-r * (T - t))  # S = 0
        V[-1, :] = 0  # S = S_max

    # Crank-Nicolson scheme
    # Coefficients
    alpha = 0.25 * dt * (sigma**2 * (np.arange(n_space + 1)**2) - r * np.arange(n_space + 1))
    beta = -0.5 * dt * (sigma**2 * (np.arange(n_space + 1)**2) + r)
    gamma = 0.25 * dt * (sigma**2 * (np.arange(n_space + 1)**2) + r * np.arange(n_space + 1))

    # Tridiagonal matrices
    M1 = np.zeros((n_space - 1, n_space - 1))
    M2 = np.zeros((n_space - 1, n_space - 1))

    for i in range(n_space - 1):
        idx = i + 1  # Adjust for interior points

        # M1: (I + A*dt/2)
        M1[i, i] = 1 - beta[idx]
        if i > 0:
            M1[i, i - 1] = -alpha[idx]
        if i < n_space - 2:
            M1[i, i + 1] = -gamma[idx]

        # M2: (I - A*dt/2)
        M2[i, i] = 1 + beta[idx]
        if i > 0:
            M2[i, i - 1] = alpha[idx]
        if i < n_space - 2:
            M2[i, i + 1] = gamma[idx]

    # Time stepping (backward)
    for j in range(n_time - 1, -1, -1):
        # RHS
        b = M2 @ V[1:-1, j + 1]

        # Boundary adjustments
        b[0] += alpha[1] * (V[0, j] + V[0, j + 1])
        b[-1] += gamma[-2] * (V[-1, j] + V[-1, j + 1])

        # Solve
        V[1:-1, j] = np.linalg.solve(M1, b)

    return S, t, V


def finite_difference_american_put(
    K: float,
    r: float,
    sigma: float,
    T: float,
    S_max: float = 200.0,
    n_space: int = 200,
    n_time: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finite difference for American put with projected SOR.

    Args:
        K: Strike price.
        r: Risk-free rate.
        sigma: Volatility.
        T: Time to maturity.
        S_max: Maximum stock price.
        n_space: Spatial grid points.
        n_time: Time steps.

    Returns:
        Tuple of (S_grid, t_grid, V_grid).
    """
    # Grid
    dS = S_max / n_space
    dt = T / n_time
    S = np.linspace(0, S_max, n_space + 1)
    t = np.linspace(0, T, n_time + 1)

    # Payoff
    payoff = np.maximum(K - S, 0)

    # Initialize
    V = np.zeros((n_space + 1, n_time + 1))
    V[:, -1] = payoff

    # Boundary conditions
    V[0, :] = K * np.exp(-r * (T - t))
    V[-1, :] = 0

    # Implicit scheme with projection
    for j in range(n_time - 1, -1, -1):
        V_old = V[:, j + 1].copy()

        # Simple implicit Euler with SOR iteration
        for _ in range(20):  # SOR iterations
            V_new = V_old.copy()

            for i in range(1, n_space):
                # Implicit scheme coefficient
                a = 0.5 * sigma**2 * i**2
                b = r * i
                c = r

                alpha_i = a * dt / dS**2 - b * dt / (2 * dS)
                beta_i = -2 * a * dt / dS**2 - c * dt - 1
                gamma_i = a * dt / dS**2 + b * dt / (2 * dS)

                # Update
                V_new[i] = (
                    -alpha_i * V_new[i - 1]
                    - gamma_i * V_old[i + 1]
                    - V_old[i]
                ) / beta_i

                # Projection onto constraint V >= payoff
                V_new[i] = max(V_new[i], payoff[i])

            V_old = V_new

        V[:, j] = V_old

    return S, t, V
