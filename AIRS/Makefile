# AIRS Makefile - Development and deployment commands

.PHONY: help install install-dev setup clean lint format test test-cov \
        docker-build docker-up docker-down docker-logs \
        db-migrate db-upgrade fetch-data train backtest api

# Default target
help:
	@echo "AIRS Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make setup          Full development setup"
	@echo ""
	@echo "Development:"
	@echo "  make lint           Run linters (ruff, mypy)"
	@echo "  make format         Format code (black, ruff)"
	@echo "  make test           Run tests"
	@echo "  make test-cov       Run tests with coverage"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo "  make docker-logs    View service logs"
	@echo ""
	@echo "Database:"
	@echo "  make db-migrate     Create new migration"
	@echo "  make db-upgrade     Apply migrations"
	@echo ""
	@echo "Pipeline:"
	@echo "  make fetch-data     Fetch latest market data"
	@echo "  make train          Train models"
	@echo "  make backtest       Run backtest"
	@echo "  make api            Start API server locally"

# =============================================================================
# Setup Commands
# =============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[all]"
	pre-commit install

setup: install-dev
	@echo "Creating data directories..."
	mkdir -p data/raw data/processed data/features data/models
	@echo "Copying environment template..."
	cp -n .env.example .env 2>/dev/null || true
	@echo ""
	@echo "Setup complete! Please edit .env with your API keys."

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check airs/ tests/
	mypy airs/

format:
	black airs/ tests/
	ruff check --fix airs/ tests/

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=airs --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# =============================================================================
# Docker Commands
# =============================================================================

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d postgres redis mlflow api

docker-up-all:
	docker-compose --profile airflow up -d

docker-down:
	docker-compose --profile airflow down

docker-logs:
	docker-compose logs -f

docker-logs-api:
	docker-compose logs -f api

docker-shell:
	docker-compose exec api /bin/bash

docker-clean:
	docker-compose down -v --remove-orphans
	docker system prune -f

# =============================================================================
# Database Commands
# =============================================================================

db-migrate:
	@read -p "Migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-reset:
	@echo "WARNING: This will delete all data!"
	@read -p "Are you sure? [y/N] " confirm; \
	if [ "$$confirm" = "y" ]; then \
		alembic downgrade base; \
		alembic upgrade head; \
	fi

# =============================================================================
# Pipeline Commands
# =============================================================================

fetch-data:
	python scripts/fetch_data.py --all

fetch-data-incremental:
	python scripts/fetch_data.py --incremental

train:
	python scripts/train_model.py

train-mlflow:
	python scripts/train_model.py --track

backtest:
	python scripts/run_backtest.py

backtest-report:
	python scripts/run_backtest.py --report

# =============================================================================
# API Commands
# =============================================================================

api:
	uvicorn airs.api.main:app --reload --host 0.0.0.0 --port 8000

api-prod:
	gunicorn airs.api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# =============================================================================
# Jupyter Notebooks
# =============================================================================

notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# =============================================================================
# MLflow
# =============================================================================

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# =============================================================================
# Airflow Commands
# =============================================================================

airflow-init:
	airflow db init
	airflow users create \
		--username admin \
		--password admin \
		--firstname Admin \
		--lastname User \
		--role Admin \
		--email admin@example.com

airflow-webserver:
	airflow webserver --port 8080

airflow-scheduler:
	airflow scheduler

# =============================================================================
# Documentation
# =============================================================================

docs:
	@echo "Documentation is in the docs/ directory"
	@echo "Key files:"
	@echo "  - docs/DATA.md"
	@echo "  - docs/FEATURES.md"
	@echo "  - docs/MODELS.md"
	@echo "  - docs/BACKTEST.md"
	@echo "  - docs/API.md"
	@echo "  - docs/DEPLOYMENT.md"
