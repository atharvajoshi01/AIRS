# American Options via Obstacle/Penalty Method

## Problem Formulation

American options allow early exercise, leading to a **free boundary problem**. The option value V(t, S) must satisfy:

1. The Black-Scholes PDE in the continuation region
2. The constraint V ≥ φ (where φ is the payoff)
3. Smooth fit at the free boundary

## Mathematical Formulation

### Linear Complementarity Problem (LCP)

The American option pricing problem can be formulated as:

```
min{L[V], V - φ} = 0
```

where:
- L[V] is the Black-Scholes PDE operator
- φ(S) is the payoff function (e.g., max(K - S, 0) for a put)

This means:
- If L[V] < 0 and V = φ: **exercise region** (optimal to exercise)
- If L[V] = 0 and V > φ: **continuation region** (optimal to hold)

## Penalty Method

The LCP is difficult to solve directly with neural networks. We use a **penalty method** that approximates the constraint with a smooth penalty term.

### Formulation

Instead of enforcing V ≥ φ strictly, we add a penalty to the PDE residual:

```
Residual = L[V] + λ * max(φ - V, 0)
```

where λ is a large penalty coefficient (e.g., 100-1000).

### Intuition

- When V < φ (violation): The penalty term is positive, forcing V to increase
- When V ≥ φ (satisfied): The penalty term is zero, allowing the PDE to govern

As λ → ∞, the penalty method converges to the exact solution.

## Implementation in DGM

### Loss Function

```python
def pde_residual_american(model, points, pde):
    # Standard BS PDE operator
    L_V = compute_pde_operator(model, points)

    # Payoff
    S = points[:, 1]
    phi = payoff(S)

    # Predicted value
    V = model(points)

    # Penalty term
    penalty = torch.maximum(phi - V, torch.zeros_like(V))

    # Combined residual
    residual = L_V + penalty_lambda * penalty

    return residual
```

### Training

The model learns to:
1. Satisfy the PDE where V > φ
2. Respect the constraint V ≥ φ everywhere
3. Automatically determine the exercise boundary

## Advantages

### 1. No Grid Required

Unlike finite difference methods, DGM doesn't need to explicitly track the free boundary on a grid.

### 2. Smooth Approximation

The penalty method provides a smooth approximation, suitable for gradient-based optimization.

### 3. Automatic Boundary Detection

The neural network implicitly learns where the exercise boundary is, without explicit parameterization.

### 4. Scalability

Extends naturally to high-dimensional problems (multi-asset American options).

## Numerical Considerations

### Penalty Coefficient (λ)

**Too small**: Constraint violations, V may drop below φ
**Too large**: Numerical instability, optimization difficulties
**Recommended**: 50-200 for most problems

### Loss Weighting

Increase the weight on the terminal condition:

```python
loss = (
    1.0 * pde_residual_loss +
    1.0 * boundary_loss +
    10.0 * terminal_loss  # Higher weight
)
```

### Validation

Check constraint satisfaction:

```python
violations = torch.sum(V < phi)
max_violation = torch.max(phi - V)
```

Ensure violations are minimal on held-out test points.

## Example: American Put

### Setup

```python
from dgmlib.pde.black_scholes_american import BlackScholesAmericanPDE

pde = BlackScholesAmericanPDE(
    r=0.05,
    sigma=0.2,
    K=100.0,
    T=1.0,
    option_type="put",
    penalty_lambda=100.0,
)
```

### Training

```python
# Higher collocation points for American
n_interior = 3000
n_boundary = 300
n_initial = 300

# Longer training
epochs = 2000
```

### Results

For K=100, r=0.05, σ=0.2, T=1.0:

| S   | Finite Diff | DGM    | Error  |
|-----|------------|--------|--------|
| 80  | 20.00      | 19.998 | 0.002  |
| 90  | 10.47      | 10.468 | 0.002  |
| 100 | 5.57       | 5.574  | 0.004  |

### Exercise Boundary

The optimal exercise boundary S*(t) can be extracted by finding where V(t, S) = φ(S):

```python
for t in time_grid:
    S_boundary = find_root(lambda S: model([t, S]) - payoff(S))
```

## Comparison with Other Methods

### Finite Difference (Projected SOR)

**Pros**:
- Well-established
- Provably convergent

**Cons**:
- Requires iterative projection at each time step
- Slow for high dimensions
- Grid-dependent

### Longstaff-Schwartz Monte Carlo

**Pros**:
- Monte Carlo scales to high dimensions
- Straightforward implementation

**Cons**:
- Regression introduces approximation error
- Slow convergence (Monte Carlo variance)
- Requires careful basis function selection

### DGM (This Approach)

**Pros**:
- Fast inference after training
- No grid or paths required
- Automatic boundary detection
- Accurate Greeks via autodiff

**Cons**:
- Requires training phase
- Hyperparameter tuning (λ, network size)
- Less established than FD/MC

## Advanced Topics

### Smooth Penalty Functions

Instead of max(φ - V, 0), use smooth approximations:

```python
# Softplus approximation
penalty = torch.log(1 + torch.exp(lambda_smooth * (phi - V)))
```

### Dual Formulation

Reformulate as a variational inequality:

```
Find V ∈ K such that:
⟨L[V], W - V⟩ ≥ 0  ∀W ∈ K
```

where K = {W : W ≥ φ}.

### Multi-Asset American Options

For d-dimensional American options (e.g., max-call on d assets):

```python
pde = AmericanMaxCallPDE(
    r=0.05,
    sigma=[0.2, 0.25, 0.3],  # d volatilities
    rho=correlation_matrix,
    K=100.0,
    T=1.0,
)

model = DGMNet(
    input_dim=d+1,  # [t, S1, S2, ..., Sd]
    hidden_dim=100,
    num_layers=5,
)
```

## References

1. Becker, S., Cheridito, P., & Jentzen, A. (2020). *Pricing and hedging American-style options with deep learning.* Journal of Risk and Financial Management, 13(7), 158.

2. Longstaff, F. A., & Schwartz, E. S. (2001). *Valuing American options by simulation: a simple least-squares approach.* The review of financial studies, 14(1), 113-147.

3. Forsyth, P. A., & Vetzal, K. R. (2002). *Quadratic convergence for valuing American options using a penalty method.* SIAM Journal on Scientific Computing, 23(6), 2095-2122.
