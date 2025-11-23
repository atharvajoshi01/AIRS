#!/usr/bin/env python3
"""
Train a PPO agent for deep hedging.

This is a minimal example demonstrating how RL can be applied
to optimal control problems in finance.

Requires: gymnasium, stable-baselines3

Usage:
    python rl_examples/train_ppo.py
"""

import argparse

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("Error: stable-baselines3 not installed.")
    print("Install with: pip install stable-baselines3")
    exit(1)

from rl_examples.deep_hedging_env import DeepHedgingEnv


def make_env():
    """Create environment."""
    return DeepHedgingEnv(
        S0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.2,
        T=1.0,
        dt=1.0 / 252,
        transaction_cost=0.001,
        risk_aversion=0.5,
    )


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train Deep Hedging Agent")
    parser.add_argument("--steps", type=int, default=100000, help="Training steps")
    parser.add_argument("--save-path", type=str, default="models/deep_hedging_ppo")
    args = parser.parse_args()

    # Create environment
    env = make_env()

    # Check environment
    print("Checking environment...")
    check_env(env, warn=True)

    # Vectorized environment
    vec_env = DummyVecEnv([make_env])

    # Create PPO agent
    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/deep_hedging/",
    )

    # Train
    print(f"Training for {args.steps} steps...")
    model.learn(total_timesteps=args.steps, progress_bar=True)

    # Save
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")

    # Test trained agent
    print("\nTesting trained agent...")
    env = make_env()
    obs, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()

    print(f"\nTest episode reward: {total_reward:.2f}")
    print(f"Final PnL: {info['PnL']:.2f}")


if __name__ == "__main__":
    main()
