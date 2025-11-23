"""Property-based tests for strike monotonicity."""

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np

from dgmlib.utils.numerics import black_scholes_analytical


@given(
    S=st.floats(min_value=50.0, max_value=150.0),
    K1=st.floats(min_value=70.0, max_value=100.0),
    K2=st.floats(min_value=100.0, max_value=130.0),
    r=st.floats(min_value=0.01, max_value=0.1),
    sigma=st.floats(min_value=0.1, max_value=0.5),
    T=st.floats(min_value=0.1, max_value=2.0),
)
@settings(max_examples=50, deadline=None)
def test_call_decreases_with_strike(S, K1, K2, r, sigma, T):
    """Test that call price decreases as strike increases."""
    price1, _, _ = black_scholes_analytical(S, K1, r, sigma, T, "call")
    price2, _, _ = black_scholes_analytical(S, K2, r, sigma, T, "call")

    # Call should decrease with higher strike
    assert price1 >= price2 or np.isclose(price1, price2, atol=1e-6)


@given(
    S=st.floats(min_value=50.0, max_value=150.0),
    K1=st.floats(min_value=70.0, max_value=100.0),
    K2=st.floats(min_value=100.0, max_value=130.0),
    r=st.floats(min_value=0.01, max_value=0.1),
    sigma=st.floats(min_value=0.1, max_value=0.5),
    T=st.floats(min_value=0.1, max_value=2.0),
)
@settings(max_examples=50, deadline=None)
def test_put_increases_with_strike(S, K1, K2, r, sigma, T):
    """Test that put price increases as strike increases."""
    price1, _, _ = black_scholes_analytical(S, K1, r, sigma, T, "put")
    price2, _, _ = black_scholes_analytical(S, K2, r, sigma, T, "put")

    # Put should increase with higher strike
    assert price2 >= price1 or np.isclose(price1, price2, atol=1e-6)
