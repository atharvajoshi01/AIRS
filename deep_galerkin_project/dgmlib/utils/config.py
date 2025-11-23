"""Configuration management utilities."""

from pathlib import Path
from typing import Any, Dict

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        OmegaConf configuration object.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return OmegaConf.create(config_dict)


def save_config(config: DictConfig, save_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: OmegaConf configuration object.
        save_path: Path to save YAML file.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(OmegaConf.to_container(config), f, default_flow_style=False)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations with override taking precedence.

    Args:
        base_config: Base configuration.
        override_config: Override configuration.

    Returns:
        Merged configuration.
    """
    return OmegaConf.merge(base_config, override_config)
