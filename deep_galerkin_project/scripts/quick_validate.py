#!/usr/bin/env python3
"""Quick validation of trained model."""

import torch
import numpy as np
from dgmlib.models.dgm import DGMNet
from dgmlib.utils.numerics import black_scholes_analytical

# Load model
checkpoint = torch.load('checkpoints/bs_european/best_model.pt', map_location='cpu', weights_only=False)
dummy_bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
model = DGMNet(input_dim=2, hidden_dim=50, num_layers=3, input_bounds=dummy_bounds)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print('European Call Option: K=100, r=0.05, σ=0.2, T=1.0')
print('='*80)

# Test at different (t, S) points
test_points = [
    # t, S, description
    (0.0, 80, 'OTM at t=0'),
    (0.0, 90, 'OTM at t=0'),
    (0.0, 100, 'ATM at t=0'),
    (0.0, 110, 'ITM at t=0'),
    (0.0, 120, 'ITM at t=0'),
    (0.5, 100, 'ATM halfway to maturity'),
    (0.9, 100, 'ATM near maturity'),
    (1.0, 90, 'OTM at maturity'),
    (1.0, 100, 'ATM at maturity'),
    (1.0, 110, 'ITM at maturity'),
]

errors = []
print('\nt     | S     | DGM Price | Analytical | Abs Error | % Error | Description')
print('-'*100)

for t, S, desc in test_points:
    inputs = torch.tensor([[t, S]], dtype=torch.float32)
    with torch.no_grad():
        dgm_price = model(inputs).item()

    bs_price, _, _ = black_scholes_analytical(S, 100.0, 0.05, 0.2, 1.0-t, 'call')
    error = abs(dgm_price - bs_price)
    pct_error = (error / bs_price * 100) if bs_price > 0.01 else 0
    errors.append(error)

    print(f'{t:5.2f} | {S:5.1f} | {dgm_price:9.4f} | {bs_price:10.4f} | {error:9.4f} | {pct_error:7.2f}% | {desc}')

print('-'*100)
print(f'Mean Absolute Error: \${np.mean(errors):.4f}')
print(f'Max Absolute Error:  \${np.max(errors):.4f}')
print(f'RMSE:                \${np.sqrt(np.mean(np.array(errors)**2)):.4f}')
print('\n✅ MODEL VALIDATION COMPLETE')
