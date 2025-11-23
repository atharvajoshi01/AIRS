# Contributing to Deep Galerkin Option Pricing

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/your-username/deep-galerkin-pricing.git
cd deep-galerkin-pricing
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### Code Style

We use:
- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **MyPy** for type checking

Format your code before committing:

```bash
make format
make lint
```

### Type Hints

All functions should have type hints:

```python
def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 100,
) -> Dict[str, List[float]]:
    """Train the model."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def compute_greeks(
    model: nn.Module,
    points: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute option Greeks via automatic differentiation.

    Args:
        model: Trained neural network model.
        points: Evaluation points of shape (N, d).

    Returns:
        Dictionary containing "delta" and "gamma" tensors.

    Example:
        >>> model = DGMNet(input_dim=2, hidden_dim=50, num_layers=3)
        >>> points = torch.tensor([[0.0, 100.0], [0.0, 110.0]])
        >>> greeks = compute_greeks(model, points)
        >>> print(greeks["delta"])
    """
    ...
```

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names: `test_bs_pde_residual_shape`

Example:

```python
def test_dgm_forward_pass_shape():
    """Test that DGM forward pass has correct output shape."""
    model = DGMNet(input_dim=2, hidden_dim=20, num_layers=2)
    x = torch.randn(10, 2)
    output = model(x)
    assert output.shape == (10, 1)
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_dgm_layer_shapes.py

# With coverage
pytest --cov=dgmlib --cov-report=html
```

### Property-Based Tests

Use Hypothesis for property-based testing:

```python
from hypothesis import given, settings
from hypothesis import strategies as st

@given(
    S=st.floats(min_value=50.0, max_value=150.0),
    K=st.floats(min_value=80.0, max_value=120.0),
)
def test_call_price_increases_with_stock_price(S, K):
    """Test that call price increases with stock price."""
    ...
```

## Pull Request Process

### 1. Ensure Quality

Before submitting:

```bash
# Format code
make format

# Run linters
make lint

# Run tests
make test
```

### 2. Update Documentation

- Add docstrings to new functions
- Update README if adding major features
- Add examples if applicable

### 3. Commit Messages

Use clear, descriptive commit messages:

```
Add support for Heston stochastic volatility model

- Implement HestonPDE class
- Add 3D DGM network support
- Include configuration file for Heston
- Add tests for Heston PDE residuals
```

### 4. Submit PR

1. Push your branch to GitHub
2. Open a Pull Request against `main`
3. Fill out the PR template
4. Wait for review

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
Describe testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented if necessary)
```

## Adding New Features

### Adding a New PDE

1. Create file in `dgmlib/pde/`:

```python
# dgmlib/pde/my_new_pde.py
from dgmlib.pde.base_pde import BasePDE

class MyNewPDE(BasePDE):
    def __init__(self, ...):
        ...

    def pde_residual(self, model, points):
        ...
```

2. Add tests in `tests/test_my_new_pde.py`

3. Add configuration in `dgmlib/configs/my_new_pde.yaml`

4. Update documentation

### Adding a New Model Architecture

1. Create file in `dgmlib/models/`
2. Inherit from `nn.Module`
3. Implement `forward()` method
4. Add `count_parameters()` method
5. Add tests
6. Update documentation

## Code Review Guidelines

Reviewers will check for:

- **Correctness**: Does the code do what it's supposed to?
- **Tests**: Are there adequate tests with good coverage?
- **Documentation**: Are functions documented?
- **Style**: Does code follow project style?
- **Performance**: Are there obvious performance issues?
- **Breaking Changes**: Are breaking changes documented?

## Bug Reports

When reporting bugs, include:

1. **Environment**: Python version, OS, package versions
2. **Reproduction**: Minimal code to reproduce the issue
3. **Expected vs Actual**: What you expected vs what happened
4. **Error Messages**: Full error traceback

## Feature Requests

When requesting features:

1. **Use Case**: Describe the problem you're trying to solve
2. **Proposed Solution**: How would you like it to work?
3. **Alternatives**: Any alternative solutions considered?

## Questions

For questions:

- Check existing issues/discussions
- Ask in GitHub Discussions
- For sensitive topics, email maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes (for significant contributions)
- Credited in relevant documentation

Thank you for contributing!
