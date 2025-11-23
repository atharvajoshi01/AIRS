.PHONY: help setup install clean test lint format train-eur train-am train-barrier api ui docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make setup          - Create virtual environment and install dependencies"
	@echo "  make install        - Install package in development mode"
	@echo "  make clean          - Remove build artifacts and cache"
	@echo "  make test           - Run test suite"
	@echo "  make lint           - Run linters"
	@echo "  make format         - Format code with black"
	@echo "  make train-eur      - Train European option model"
	@echo "  make train-am       - Train American option model"
	@echo "  make train-barrier  - Train barrier option model"
	@echo "  make api            - Start FastAPI server"
	@echo "  make ui             - Start Streamlit UI"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"

setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -e .
	@echo "Setup complete! Activate with: source venv/bin/activate"

install:
	pip install -e .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	pytest -v --cov=dgmlib --cov-report=html --cov-report=term

lint:
	ruff check dgmlib/ scripts/ tests/
	black --check dgmlib/ scripts/ tests/
	mypy dgmlib/ || true

format:
	black dgmlib/ scripts/ tests/ api/ ui/
	ruff check --fix dgmlib/ scripts/ tests/

train-eur:
	python scripts/train.py --config dgmlib/configs/bs_european.yaml

train-am:
	python scripts/train.py --config dgmlib/configs/bs_american.yaml

train-barrier:
	python scripts/train.py --config dgmlib/configs/bs_barrier.yaml

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/app.py

docker-build:
	docker build -t dgm-pricing .

docker-run:
	docker run -it -p 8000:8000 -p 8501:8501 dgm-pricing

docker-train:
	docker run -v $(PWD)/checkpoints:/app/checkpoints dgm-pricing \
		python scripts/train.py --config dgmlib/configs/bs_european.yaml
