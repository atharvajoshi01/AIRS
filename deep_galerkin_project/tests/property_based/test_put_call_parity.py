"""Property-based tests for put-call parity."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from dgmlib.utils.numerics import black_scholes_analytical


@given(
    S=st.floats(min_value=50.0, max_value=200.0),
    K=st.floats(min_value=50.0, max_value=200.0),
    r=st.floats(min_value=0.0, max_value=0.15),
    sigma=st.floats(min_value=0.05, max_value=0.8),
    T=st.floats(min_value=0.01, max_value=3.0),
)
@settings(max_examples=100, deadline=None)
def test_put_call_parity_holds(S, K, r, sigma, T):
    """
    Test put-call parity: C - P = S - K*exp(-r*T).

    This fundamental no-arbitrage relation must hold for European options.
    """
    call_price, _, _ = black_scholes_analytical(S, K, r, sigma, T, "call")
    put_price, _, _ = black_scholes_analytical(S, K, r, sigma, T, "put")

    lhs = call_price - put_price
    rhs = S - K * np.exp(-r * T)

    # Allow small numerical tolerance
    assert np.isclose(lhs, rhs, rtol=1e-5, atol=1e-6)
