# Black-Scholes PDE Derivation

## The Black-Scholes Model

### Assumptions

1. The stock price follows a geometric Brownian motion:
   ```
   dS = μS dt + σS dW
   ```
   where μ is the drift, σ is the volatility, and W is a Wiener process.

2. No transaction costs or taxes
3. Continuous trading is possible
4. The risk-free rate r is constant
5. Short selling is allowed
6. No dividends

### Derivation via Itô's Lemma

Consider a derivative V(S, t) whose value depends on the stock price S and time t.

By Itô's lemma:

```
dV = (∂V/∂t + μS∂V/∂S + ½σ²S²∂²V/∂S²) dt + σS∂V/∂S dW
```

Construct a portfolio Π consisting of:
- One derivative (long)
- Δ shares of stock (short)

```
Π = V - ΔS
```

The change in portfolio value is:

```
dΠ = dV - Δ dS
```

Substituting:

```
dΠ = (∂V/∂t + μS∂V/∂S + ½σ²S²∂²V/∂S²) dt + σS∂V/∂S dW - Δ(μS dt + σS dW)
```

Choose Δ = ∂V/∂S (delta hedging) to eliminate the stochastic term:

```
dΠ = (∂V/∂t + ½σ²S²∂²V/∂S²) dt
```

Since the portfolio is now risk-free, it must earn the risk-free rate:

```
dΠ = rΠ dt = r(V - S∂V/∂S) dt
```

Equating the two expressions:

```
∂V/∂t + ½σ²S²∂²V/∂S² = r(V - S∂V/∂S)
```

Rearranging gives the **Black-Scholes PDE**:

```
∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
```

## Boundary and Initial Conditions

### European Call Option

**Terminal condition** (at maturity T):
```
V(T, S) = max(S - K, 0)
```

**Boundary conditions**:
- At S = 0: V(t, 0) = 0 (worthless)
- As S → ∞: V(t, S) ≈ S - Ke^(-r(T-t)) (intrinsic value)

### European Put Option

**Terminal condition**:
```
V(T, S) = max(K - S, 0)
```

**Boundary conditions**:
- At S = 0: V(t, 0) = Ke^(-r(T-t)) (present value of strike)
- As S → ∞: V(t, S) → 0

## Analytical Solution

The Black-Scholes formula for a European call is:

```
C(S, t) = SN(d₁) - Ke^(-r(T-t))N(d₂)
```

where:

```
d₁ = [ln(S/K) + (r + σ²/2)(T-t)] / (σ√(T-t))
d₂ = d₁ - σ√(T-t)
```

and N(·) is the cumulative standard normal distribution.

For a European put:

```
P(S, t) = Ke^(-r(T-t))N(-d₂) - SN(-d₁)
```

## Greeks

### Delta (∂V/∂S)

**Call**: Δ = N(d₁)

**Put**: Δ = -N(-d₁) = N(d₁) - 1

**Properties**:
- 0 ≤ Δ_call ≤ 1
- -1 ≤ Δ_put ≤ 0

### Gamma (∂²V/∂S²)

For both calls and puts:

```
Γ = N'(d₁) / (Sσ√(T-t))
```

where N'(x) = (1/√(2π))e^(-x²/2)

**Properties**:
- Γ ≥ 0 always
- Γ is maximum near at-the-money (S ≈ K)

### Vega (∂V/∂σ)

```
ν = S√(T-t)N'(d₁)
```

### Theta (∂V/∂t)

**Call**:
```
Θ = -(Sσ N'(d₁))/(2√(T-t)) - rKe^(-r(T-t))N(d₂)
```

### Rho (∂V/∂r)

**Call**: ρ = K(T-t)e^(-r(T-t))N(d₂)

## Put-Call Parity

For European options with the same strike K and maturity T:

```
C - P = S - Ke^(-r(T-t))
```

This no-arbitrage relation is fundamental and must hold for any valid pricing method.

## Deep Galerkin Formulation

In the DGM approach, we:

1. **Approximate** the solution: V(t, S) ≈ NN(t, S; θ)

2. **Compute derivatives** via automatic differentiation:
   ```
   V_t = ∂NN/∂t
   V_S = ∂NN/∂S
   V_SS = ∂²NN/∂S²
   ```

3. **Define PDE residual**:
   ```
   R(t, S) = V_t + ½σ²S²V_SS + rSV_S - rV
   ```

4. **Minimize** the loss:
   ```
   L = E[(R(t, S))²] + λ_bc·L_bc + λ_ic·L_ic
   ```

   where:
   - L_bc enforces boundary conditions
   - L_ic enforces the terminal payoff

This transforms the PDE-solving problem into an optimization problem solvable via stochastic gradient descent.
