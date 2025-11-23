#!/usr/bin/env python3
"""
Training script for Deep Galerkin Method.

Usage:
    python scripts/train.py --config dgmlib/configs/bs_european.yaml
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig

from dgmlib.loss.residuals import PDELoss
from dgmlib.models.dgm import DGMNet
from dgmlib.models.mlp_baseline import MLPBaseline
from dgmlib.pde.base_pde import BasePDE
from dgmlib.pde.black_scholes import BlackScholesPDE
from dgmlib.pde.black_scholes_american import BlackScholesAmericanPDE
from dgmlib.pde.black_scholes_barrier import BlackScholesBarrierPDE
from dgmlib.pde.heston import HestonPDE
from dgmlib.sampling.latin_hypercube import LatinHypercubeSampler
from dgmlib.sampling.sobol import SobolSampler
from dgmlib.training.callbacks import EarlyStopping, LRSchedulerCallback, ModelCheckpoint
from dgmlib.training.trainer import DGMTrainer
from dgmlib.utils.config import load_config
from dgmlib.utils.seeds import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_pde(config: DictConfig):
    """Create PDE object from config."""
    pde_type = config.pde.type

    if pde_type == "black_scholes":
        return BlackScholesPDE(
            r=config.pde.r,
            sigma=config.pde.sigma,
            K=config.pde.K,
            T=config.pde.T,
            option_type=config.pde.option_type,
            S_min=config.pde.S_min,
            S_max=config.pde.S_max,
        )
    elif pde_type == "black_scholes_american":
        return BlackScholesAmericanPDE(
            r=config.pde.r,
            sigma=config.pde.sigma,
            K=config.pde.K,
            T=config.pde.T,
            option_type=config.pde.option_type,
            S_min=config.pde.S_min,
            S_max=config.pde.S_max,
            penalty_lambda=config.pde.penalty_lambda,
        )
    elif pde_type == "black_scholes_barrier":
        return BlackScholesBarrierPDE(
            r=config.pde.r,
            sigma=config.pde.sigma,
            K=config.pde.K,
            T=config.pde.T,
            barrier=config.pde.barrier,
            barrier_type=config.pde.barrier_type,
            option_type=config.pde.option_type,
            S_min=config.pde.S_min,
            S_max=config.pde.S_max,
        )
    elif pde_type == "heston":
        return HestonPDE(
            r=config.pde.r,
            kappa=config.pde.kappa,
            theta=config.pde.theta,
            sigma_v=config.pde.sigma_v,
            rho=config.pde.rho,
            K=config.pde.K,
            T=config.pde.T,
            option_type=config.pde.option_type,
            S_min=config.pde.S_min,
            S_max=config.pde.S_max,
            v_min=config.pde.v_min,
            v_max=config.pde.v_max,
        )
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")


def create_model(config: DictConfig, pde: BasePDE) -> nn.Module:
    """Create model from config."""
    model_type = config.model.type

    if model_type == "dgm":
        # Create input bounds from PDE domain
        # For Black-Scholes: inputs are [t, S], so domain keys in order ['t', 'S']
        if 't' in pde.domain and 'S' in pde.domain:
            domain_keys = ['t', 'S']
        elif 't' in pde.domain and 'S' in pde.domain and 'v' in pde.domain:
            domain_keys = ['t', 'S', 'v']  # For Heston
        else:
            domain_keys = sorted(pde.domain.keys())

        input_bounds = torch.tensor(
            [pde.domain[key] for key in domain_keys],
            dtype=torch.float32
        )  # Shape: (input_dim, 2)

        return DGMNet(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.output_dim,
            num_layers=config.model.num_layers,
            activation=config.model.activation,
            use_spectral_norm=config.model.use_spectral_norm,
            dropout=config.model.dropout,
            input_bounds=input_bounds,
        )
    elif model_type == "mlp":
        return MLPBaseline(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.output_dim,
            num_layers=config.model.num_layers,
            activation=config.model.activation,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_sampler(config: DictConfig, pde):
    """Create sampler from config."""
    sampling_method = config.sampling.method

    if sampling_method == "sobol":
        return SobolSampler(pde.domain, seed=config.seed)
    elif sampling_method == "latin_hypercube":
        return LatinHypercubeSampler(pde.domain, seed=config.seed)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")


def create_optimizer(model: nn.Module, config: DictConfig) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    optimizer_type = config.training.optimizer.lower()

    if optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(config.training.learning_rate),
            weight_decay=float(config.training.weight_decay),
        )
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(config.training.learning_rate),
            weight_decay=float(config.training.weight_decay),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(optimizer, config: DictConfig):
    """Create learning rate scheduler."""
    if not config.training.use_scheduler:
        return None

    scheduler_type = config.training.scheduler_type

    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            eta_min=1e-6,
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.epochs // 3,
            gamma=0.1,
        )
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=20,
        )
    else:
        return None


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Deep Galerkin Model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Experiment: {config.experiment_name}")

    # Set seed
    set_seed(config.seed)

    # Create PDE
    pde = create_pde(config)
    logger.info(f"Created PDE: {config.pde.type}")

    # Create model
    model = create_model(config, pde)
    logger.info(f"Created model: {config.model.type}")
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Create loss
    loss_fn = PDELoss(
        pde,
        lambda_pde=config.training.lambda_pde,
        lambda_bc=config.training.lambda_bc,
        lambda_ic=config.training.lambda_ic,
    )

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Create trainer
    trainer = DGMTrainer(
        model=model,
        pde=pde,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=args.device,
        use_amp=config.training.use_amp,
        gradient_clip=config.training.gradient_clip,
        checkpoint_dir=config.logging.checkpoint_dir,
        log_dir=config.logging.log_dir,
    )

    # Create sampler
    sampler = create_sampler(config, pde)
    logger.info(f"Created sampler: {config.sampling.method}")

    # Create validation points
    val_sampler = SobolSampler(pde.domain, seed=config.seed + 1)
    val_points = val_sampler.sample(config.validation.n_points, device=args.device)

    # Get analytical values if available
    val_values = None
    if config.validation.compare_analytical:
        val_values = pde.analytical_solution(val_points)

    # Create callbacks
    callbacks = []

    if config.training.early_stopping:
        early_stop = EarlyStopping(
            patience=int(config.training.patience),
            min_delta=float(config.training.min_delta),
        )
        callbacks.append(early_stop)

    if config.logging.save_checkpoints:
        checkpoint = ModelCheckpoint(
            checkpoint_dir=config.logging.checkpoint_dir,
            monitor="val_loss",
            mode="min",
        )
        callbacks.append(checkpoint)

    scheduler = create_scheduler(optimizer, config)
    if scheduler is not None:
        lr_callback = LRSchedulerCallback(scheduler)
        callbacks.append(lr_callback)

    # Train
    logger.info("Starting training...")
    history = trainer.train(
        sampler=sampler,
        n_interior=config.training.n_interior,
        n_boundary=config.training.n_boundary,
        n_initial=config.training.n_initial,
        epochs=config.training.epochs,
        val_points=val_points,
        val_values=val_values,
        callbacks=callbacks,
        log_interval=config.logging.log_interval,
    )

    # Save final checkpoint
    final_path = Path(config.logging.checkpoint_dir) / "final_model.pt"
    trainer.save_checkpoint(str(final_path), config=config)
    logger.info(f"Training complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
