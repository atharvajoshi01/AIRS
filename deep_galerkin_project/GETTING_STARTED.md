# Getting Started

Setup and usage guide for the Deep Galerkin Method option pricing system.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/atharvajoshi01/deep-galerkin-pricing.git
cd deep-galerkin-pricing
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## Training a Model

### Black-Scholes European Options

```bash
python scripts/train.py --config dgmlib/configs/bs_european.yaml
```

This trains a model to price European call options with:
- Strike K = 100
- Risk-free rate r = 5%
- Volatility σ = 20%
- Maturity T = 1 year

Training takes ~3 minutes on CPU and achieves < 1% pricing error.

### Validate the Model

```bash
python scripts/validate_model.py
```

This runs comprehensive validation across 26 different market scenarios and computes Greeks.

## Pricing Options

```bash
python scripts/price_cli.py \
  --S 100 \
  --K 100 \
  --r 0.05 \
  --sigma 0.2 \
  --T 1.0 \
  --type call \
  --method dgm
```

Output:
```
Price:  10.384621
Delta:  0.615614
Gamma:  0.019436
```

## Create Visualizations

Generate beautiful 3D pricing surface plots:

```bash
python scripts/visualize_pricing_surface.py
```

This creates:
- `plots/pricing_surfaces.png` - DGM vs analytical surfaces
- `plots/error_heatmap.png` - Pricing error visualization

## Launch the Web Interface

### Option 1: Streamlit Dashboard (Interactive)

```bash
streamlit run ui/app.py
```

Visit http://localhost:8501 to price options interactively!

### Option 2: REST API

```bash
uvicorn api.main:app --reload
```

Visit http://localhost:8000/docs for the interactive API documentation.

**Example API call:**
```bash
curl -X POST "http://localhost:8000/price/european" \
  -H "Content-Type: application/json" \
  -d '{
    "S": 100,
    "K": 100,
    "r": 0.05,
    "sigma": 0.2,
    "T": 1.0,
    "option_type": "call",
    "method": "dgm"
  }'
```

## Run Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest -v --cov=dgmlib --cov-report=html

# Run specific test categories
pytest tests/test_pde_residuals.py -v
pytest tests/property_based/ -v
```

## Advanced Usage

### Train on Different Option Types

**American Options:**
```bash
python scripts/train.py --config dgmlib/configs/bs_american.yaml
```

**Barrier Options:**
```bash
python scripts/train.py --config dgmlib/configs/bs_barrier.yaml
```

**Heston Stochastic Volatility:**
```bash
python scripts/train.py --config dgmlib/configs/heston_european.yaml
```

### Customize Training

Edit the YAML config files in `dgmlib/configs/` to modify:
- Model architecture (hidden layers, activation functions)
- Training parameters (learning rate, batch size, epochs)
- PDE parameters (strike, volatility, interest rate)
- Sampling strategy (Sobol, Latin Hypercube)

### Use as a Python Library

```python
import torch
from dgmlib.models.dgm import DGMNet
from dgmlib.training.metrics import compute_greeks

# Load trained model
checkpoint = torch.load('checkpoints/bs_european/best_model.pt')
model = DGMNet(input_dim=2, hidden_dim=50, num_layers=3, 
               input_bounds=torch.tensor([[0, 1], [0, 200]]))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Price an option
inputs = torch.tensor([[0.0, 105.0]])  # t=0, S=105
price = model(inputs).item()

# Compute Greeks
greeks = compute_greeks(model, inputs)
delta = greeks['delta'].item()
gamma = greeks['gamma'].item()

print(f"Price: ${price:.2f}, Delta: {delta:.4f}, Gamma: {gamma:.6f}")
```

## Next Steps

- Read the [Mathematical Background](docs/math_black_scholes.md)
- Understand the [DGM Architecture](docs/dgm_architecture.md)
- Check out [Benchmarking Results](docs/benchmarking.md)
- Explore the [Full Project Summary](PROJECT_SUMMARY.md)

## Common Issues

### Issue: Training is slow
**Solution:** Reduce `n_interior` and `n_boundary` in the config file, or enable GPU training.

### Issue: Model accuracy is poor
**Solution:** Increase `num_layers` or `hidden_dim` in the model config, or train for more epochs.

### Issue: Import errors
**Solution:** Make sure you installed the package with `pip install -e .`

## Getting Help

- Check the [issues page](https://github.com/atharvajoshi01/deep-galerkin-pricing/issues)
- Read the [documentation](docs/)
- Review example scripts in `scripts/`

