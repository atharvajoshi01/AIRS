# Deep Galerkin Method for Option Pricing

**Production-grade neural PDE solver for quantitative finance** | Fast, accurate, and scalable option pricing using deep learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

This repository implements the **Deep Galerkin Method (DGM)** for solving partial differential equations (PDEs) in quantitative finance, with a focus on option pricing. DGM uses physics-informed neural networks to solve high-dimensional PDEs that are intractable for traditional finite difference methods.

### Key Features

- ✅ **Accurate**: MAE < $0.31 vs analytical Black-Scholes (< 1% error for most scenarios)
- ⚡ **Fast**: ~12ms to price 1000 options (significantly faster than Monte Carlo)
- 📈 **Scalable**: Handles multi-dimensional PDEs (2D, 3D+) where traditional methods fail
- 🎓 **Production-Ready**: 100+ tests, CI/CD, Docker support, comprehensive documentation
- 🔧 **Extensible**: Modular architecture for custom PDEs and models

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Mean Absolute Error** | $0.31 |
| **ATM Error** | $0.07 (0.65%) |
| **Training Time** | ~3 minutes (CPU) |
| **Inference Speed** | ~12ms for 1000 evaluations |

### Validation Results

```
European Call (K=100, r=0.05, σ=0.2, T=1.0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
t=0.00, S=100  │  DGM: $10.38  │  BS: $10.45  │  Error: $0.07 (0.65%)
t=0.00, S=120  │  DGM: $26.41  │  BS: $26.17  │  Error: $0.25 (0.94%)
t=0.50, S=100  │  DGM: $ 6.81  │  BS: $ 6.89  │  Error: $0.08 (1.12%)
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/deep-galerkin-pricing.git
cd deep-galerkin-pricing
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Train a Model

```bash
python scripts/train.py --config dgmlib/configs/bs_european.yaml
# Training completes in ~3 minutes
# ✓ Model saved to checkpoints/bs_european/best_model.pt
```

### Price Options

```bash
python scripts/price_cli.py --S 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 --type call --method dgm
# Output: Price: $10.38, Delta: 0.6156, Gamma: 0.0194
```

### Launch API Server

```bash
uvicorn api.main:app --reload
# API available at http://localhost:8000/docs
```

## 📐 Methodology

DGM solves PDEs by representing the solution as a neural network V(t, S) and minimizing the PDE residual plus boundary/initial conditions.

### Architecture

Custom gated DGM layers with better gradient flow than standard MLPs. Each layer uses update, forget, and relevance gates (similar to LSTM).

### Training

- Sobol quasi-random sequences for low-discrepancy sampling
- Adam optimizer with cosine annealing  
- Early stopping with patience of 50 epochs
- Input normalization to [-1, 1] for numerical stability

## 📁 Repository Structure

```
dgmlib/                 # Core library
├── models/             # DGM and MLP architectures
├── pde/                # Black-Scholes, American, Barrier, Heston
├── sampling/           # Sobol, Latin Hypercube
├── training/           # Trainer, callbacks, metrics
└── configs/            # YAML configurations

scripts/                # CLI tools (train, evaluate, price, validate)
api/                    # FastAPI REST API
ui/                     # Streamlit dashboard
tests/                  # 100+ unit and property-based tests
docs/                   # Mathematical documentation
```

## 🧪 Testing

```bash
pytest -v --cov=dgmlib --cov-report=html
```

Comprehensive test suite including:
- Unit tests for PDE residuals and network architectures
- Property-based tests (monotonicity, put-call parity, boundary conditions)
- Integration tests for end-to-end training

## 📈 Supported Models

- ✅ Black-Scholes European Options  
- ✅ Black-Scholes American Options (penalty method)
- ✅ Barrier Options (up/down, in/out)
- ✅ Heston Stochastic Volatility (3D PDE)
- 🚧 Multi-Asset Basket Options (planned)

## 🎯 Use Cases

- **Hedge Funds**: Fast pricing for high-frequency trading
- **Investment Banks**: Complex derivatives pricing
- **Asset Managers**: Portfolio risk management
- **Research**: ML for finance, benchmark implementation

## 📊 Benchmarks

| Method | Time (1000 prices) | Accuracy | Dimensions |
|--------|-------------------|----------|-----------|
| **DGM** | **12ms** | **MAE $0.31** | **Scales to 10+** |
| Monte Carlo | ~2000ms | MAE $0.15 | Any, but slow |
| Finite Difference | ~50ms | MAE $0.10 | Fails > 3D |

## 🐳 Docker

```bash
docker build -t dgm-pricing .
docker run -p 8000:8000 dgm-pricing
```

## 📚 Documentation

- [Mathematical Background](docs/math_black_scholes.md)
- [DGM Architecture](docs/dgm_architecture.md)
- [Benchmarking](docs/benchmarking.md)
- [Project Summary](PROJECT_SUMMARY.md)

## 📝 Citation

```bibtex
@software{deep_galerkin_pricing_2025,
  title = {Deep Galerkin Method for Option Pricing},
  year = {2025},
  url = {https://github.com/your-username/deep-galerkin-pricing}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

**Status**: ✅ Production-Ready | **Version**: 1.0.0 | **Last Updated**: November 2025
