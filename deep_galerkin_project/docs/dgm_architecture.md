# Deep Galerkin Method Architecture

## Overview

The Deep Galerkin Method (DGM) introduces a specialized neural network architecture designed for solving PDEs. Unlike standard feedforward networks, DGM uses a gated architecture that maintains better information flow across layers.

## DGM Layer

### Mathematical Formulation

Given:
- Input: **x** ∈ ℝ^d (e.g., [t, S])
- Previous state: **S** ∈ ℝ^n

The DGM layer computes:

```
Z = σ(U_z·x + W_z·S + b_z)        # Update gate
G = σ(U_g·x + W_g·S + b_g)        # Forget gate
R = σ(U_r·x + W_r·S + b_r)        # Relevance gate
H = φ(U_h·x + W_h·(S ⊙ R) + b_h)  # Candidate state
S_new = (1 - G) ⊙ H + Z ⊙ S       # New state
```

where:
- σ(·) is the sigmoid function
- φ(·) is the activation (tanh or gelu)
- ⊙ denotes element-wise multiplication
- U, W are weight matrices
- b are bias vectors

### Intuition

The gates serve specific purposes:

1. **Update gate (Z)**: Controls how much of the previous state to keep
2. **Forget gate (G)**: Decides what information to discard
3. **Relevance gate (R)**: Filters which parts of the state are relevant for the candidate
4. **Candidate (H)**: Proposes new information based on input and gated state

This is similar to LSTM/GRU but adapted for feed-forward processing of spatial coordinates.

## Full DGM Network

### Architecture

```
Input (x) ∈ ℝ^d
    ↓
Initial Layer: S₀ = φ(W₀·x + b₀) ∈ ℝ^n
    ↓
DGM Layer 1: S₁ = DGM(x, S₀)
    ↓
DGM Layer 2: S₂ = DGM(x, S₁)
    ↓
    ...
    ↓
DGM Layer L: S_L = DGM(x, S_{L-1})
    ↓
Output Layer: y = W_out·S_L + b_out ∈ ℝ
```

### Implementation

```python
class DGMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()

        # Initial transformation
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # DGM layers
        self.dgm_layers = nn.ModuleList([
            DGMLayer(input_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Output
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Initial state
        S = torch.tanh(self.input_layer(x))

        # Pass through DGM layers
        for dgm_layer in self.dgm_layers:
            S = dgm_layer(x, S)

        # Output
        return self.output_layer(S)
```

## Advantages Over Standard MLPs

### 1. Better Gradient Flow

The gated architecture provides skip connections that help gradients flow backwards, mitigating the vanishing gradient problem.

### 2. Information Persistence

The update and forget gates allow the network to maintain information about the input **x** across many layers.

### 3. Flexibility

By conditioning each layer on the original input **x**, the network can learn different transformations at different depths while maintaining awareness of the coordinate location.

### 4. Empirical Performance

Studies show DGM networks:
- Converge faster than standard MLPs
- Achieve lower PDE residuals
- Better approximate derivatives (important for Greeks)
- More stable training for high-dimensional problems

## Hyperparameter Guidelines

### Hidden Dimension

- **Small problems** (1D-2D): 20-50 neurons
- **Medium problems** (2D-3D): 50-100 neurons
- **Large problems** (>3D): 100-200 neurons

### Number of Layers

- **Smooth PDEs**: 3-4 layers
- **Complex geometry**: 4-6 layers
- **High-dimensional**: 5-8 layers

### Activation Functions

- **tanh**: Default choice, smooth derivatives
- **gelu**: Alternative, sometimes better for deep networks
- **sigmoid**: Avoid for hidden layers (saturation issues)

### Regularization

```python
DGMNet(
    input_dim=2,
    hidden_dim=50,
    num_layers=3,
    activation="tanh",
    use_spectral_norm=True,   # Stabilizes training
    dropout=0.1,               # Prevents overfitting
)
```

## Comparison: DGM vs MLP

### Memory Complexity

**DGM Layer**: O(d·n + n²) parameters per layer
**MLP Layer**: O(n²) parameters per layer

For typical d << n, they're comparable.

### Computational Cost

Both have similar forward pass complexity: O(n²) per layer.

However, DGM requires computing gates, adding ~4x operations per layer.

### Training Time

In practice:
- DGM: Slightly slower per epoch (~1.2-1.5x)
- DGM: Fewer epochs needed (often ~0.5-0.7x)
- Overall: Similar total training time

### Accuracy

For PDEs:
- DGM typically achieves 2-10x lower residuals
- Better approximation of derivatives
- More robust to hyperparameter choices

## Tips for Training

### 1. Initialization

Xavier/Glorot initialization works well:

```python
nn.init.xavier_normal_(layer.weight)
nn.init.zeros_(layer.bias)
```

### 2. Learning Rate

- Start with 1e-3
- Use cosine annealing or reduce on plateau
- Warmup can help for deep networks

### 3. Loss Weighting

```python
loss = (
    1.0 * pde_residual_loss +      # PDE enforcement
    1.0 * boundary_loss +           # Boundary conditions
    10.0 * terminal_loss            # Terminal condition (higher weight)
)
```

### 4. Collocation Points

- Start with ~1000-2000 interior points
- Include ~100-200 boundary points per boundary
- Increase if residuals plateau

### 5. Validation

Monitor:
- PDE residual L2 norm on held-out points
- Boundary condition violations
- Comparison with analytical solution (if available)

## References

1. Sirignano, J., & Spiliopoulos, K. (2018). *DGM: A deep learning algorithm for solving partial differential equations.* Journal of Computational Physics, 375, 1339-1364.

2. Beck, C., Becker, S., Grohs, P., Jaafari, N., & Jentzen, A. (2021). *Solving the Kolmogorov PDE by means of deep learning.* Journal of Scientific Computing, 88(3), 1-28.
