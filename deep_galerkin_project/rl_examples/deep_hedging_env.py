"""
Deep Hedging environment using Gymnasium.

Implements a continuous hedging problem where an agent learns to hedge
an option position by dynamically trading the underlying asset.
"""

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DeepHedgingEnv(gym.Env):
    """
    Deep Hedging Environment.

    State: [t, S, position, PnL]
    Action: Delta hedge adjustment (continuous)
    Reward: -transaction_cost - risk_penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        S0: float = 100.0,
        K: float = 100.0,
        r: float = 0.05,
        sigma: float = 0.2,
        T: float = 1.0,
        dt: float = 1.0 / 252,  # Daily rehedging
        transaction_cost: float = 0.001,
        risk_aversion: float = 0.5,
    ):
        """
        Initialize Deep Hedging environment.

        Args:
            S0: Initial stock price.
            K: Option strike.
            r: Risk-free rate.
            sigma: Volatility.
            T: Time to maturity.
            dt: Time step (e.g., 1/252 for daily).
            transaction_cost: Proportional transaction cost.
            risk_aversion: Risk aversion parameter.
        """
        super().__init__()

        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.transaction_cost = transaction_cost
        self.risk_aversion = risk_aversion

        self.n_steps = int(T / dt)

        # State: [normalized_t, normalized_S, position, normalized_PnL]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -5.0, -10.0]),
            high=np.array([1.0, 5.0, 5.0, 10.0]),
            dtype=np.float32,
        )

        # Action: delta hedge adjustment
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment."""
        super().reset(seed=seed)

        self.current_step = 0
        self.S = self.S0
        self.position = 0.0  # Current hedge position
        self.PnL = 0.0
        self.done = False

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        t_norm = self.current_step * self.dt / self.T
        S_norm = self.S / self.S0
        position_norm = self.position
        PnL_norm = self.PnL / self.S0

        return np.array([t_norm, S_norm, position_norm, PnL_norm], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Hedge adjustment (delta change).

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Action is hedge adjustment
        new_position = float(action[0])

        # Transaction cost
        trade_size = abs(new_position - self.position)
        cost = self.transaction_cost * trade_size * self.S

        # Update position
        old_position = self.position
        self.position = new_position

        # Simulate stock price change (geometric Brownian motion)
        dW = np.random.randn() * np.sqrt(self.dt)
        dS = self.r * self.S * self.dt + self.sigma * self.S * dW
        old_S = self.S
        self.S += dS

        # Hedging PnL (mark-to-market)
        hedge_pnl = old_position * (self.S - old_S)

        # Update cumulative PnL
        self.PnL += hedge_pnl - cost

        # Risk penalty (variance of PnL)
        risk_penalty = self.risk_aversion * (self.PnL ** 2)

        # Reward: negative of (cost + risk)
        reward = -cost - risk_penalty

        # Advance time
        self.current_step += 1

        # Check termination
        terminated = self.current_step >= self.n_steps

        if terminated:
            # Final payoff
            option_payoff = max(self.S - self.K, 0)

            # Final PnL includes option payoff (we're short the option)
            final_pnl = self.PnL - option_payoff

            # Final reward based on total PnL
            reward = -abs(final_pnl)  # Minimize absolute deviation from zero

        obs = self._get_observation()
        info = {
            "S": self.S,
            "position": self.position,
            "PnL": self.PnL,
            "cost": cost,
        }

        return obs, reward, terminated, False, info

    def render(self):
        """Render environment (optional)."""
        t = self.current_step * self.dt
        print(f"t={t:.3f}, S={self.S:.2f}, pos={self.position:.3f}, PnL={self.PnL:.2f}")
