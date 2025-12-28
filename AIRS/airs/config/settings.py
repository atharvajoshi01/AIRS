"""
Application settings using Pydantic Settings.

Loads configuration from environment variables and .env files.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    airs_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # API Keys
    fred_api_key: str = Field(default="", description="FRED API key")
    alpha_vantage_api_key: str = Field(default="", description="Alpha Vantage API key")

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "airs"
    postgres_user: str = "airs"
    postgres_password: str = "airs_dev_password"
    database_url: PostgresDsn | None = None

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "airs-drawdown-prediction"

    # Data storage
    data_dir: Path = Path("./data")

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Model parameters
    default_lookback_days: int = 252  # 1 trading year
    default_prediction_horizon: int = 15  # days ahead to predict
    default_drawdown_threshold: float = 0.05  # 5% drawdown threshold

    @field_validator("data_dir", mode="before")
    @classmethod
    def parse_data_dir(cls, v: str | Path) -> Path:
        """Convert string to Path."""
        return Path(v)

    @property
    def database_dsn(self) -> str:
        """Construct database URL from components or use provided URL."""
        if self.database_url:
            return str(self.database_url)
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.airs_env == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.airs_env == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
