"""Property-based tests for boundary behavior."""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from dgmlib.utils.numerics import black_scholes_analytical


@given(
    K=st.floats(min_value=50.0, max_value=200.0),
    r=st.floats(min_value=0.0, max_value=0.15),
    sigma=st.floats(min_value=0.1, max_value=0.8),
    T=st.floats(min_value=0.1, max_value=3.0),
)
@settings(max_examples=50, deadline=None)
def test_call_zero_at_zero_spot(K, r, sigma, T):
    """Test that call value at S=0 is zero."""
    # Small positive S to avoid numerical issues
    S = 1e-6
    price, _, _ = black_scholes_analytical(S, K, r, sigma, T, "call")

    assert price < 0.01  # Essentially zero


@given(
    K=st.floats(min_value=50.0, max_value=200.0),
    r=st.floats(min_value=0.0, max_value=0.15),
    sigma=st.floats(min_value=0.1, max_value=0.8),
    T=st.floats(min_value=0.1, max_value=3.0),
)
@settings(max_examples=50, deadline=None)
def test_put_equals_discounted_strike_at_zero_spot(K, r, sigma, T):
    """Test that put value at S=0 equals K*exp(-r*T)."""
    S = 1e-6
    price, _, _ = black_scholes_analytical(S, K, r, sigma, T, "put")

    expected = K * np.exp(-r * T)

    assert np.isclose(price, expected, rtol=0.01)


@given(
    S=st.floats(min_value=50.0, max_value=200.0),
    K=st.floats(min_value=50.0, max_value=200.0),
    r=st.floats(min_value=0.0, max_value=0.15),
    sigma=st.floats(min_value=0.1, max_value=0.8),
)
@settings(max_examples=50, deadline=None)
def test_option_equals_intrinsic_at_maturity(S, K, r, sigma):
    """Test that option value equals intrinsic value at maturity."""
    T = 1e-6  # Near maturity

    call_price, _, _ = black_scholes_analytical(S, K, r, sigma, T, "call")
    call_intrinsic = max(S - K, 0)

    put_price, _, _ = black_scholes_analytical(S, K, r, sigma, T, "put")
    put_intrinsic = max(K - S, 0)

    assert np.isclose(call_price, call_intrinsic, rtol=0.01, atol=0.01)
    assert np.isclose(put_price, put_intrinsic, rtol=0.01, atol=0.01)
