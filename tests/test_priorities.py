import numpy as np
import pytest

from explainit.priorities.linear import basic_linear
from explainit.priorities.nonlinear import basic_linear_step, exponential

PRIORITY_FUNCTIONS = [basic_linear, basic_linear_step, exponential]


@pytest.mark.parametrize("fn", PRIORITY_FUNCTIONS)
def test_increasing_is_zero_below_and_one_above_the_ramp(fn):
    np.testing.assert_allclose(fn([-1.0, 0.2, 0.8, 5.0], 0.2, 0.8), [0.0, 0.0, 1.0, 1.0])


@pytest.mark.parametrize("fn", PRIORITY_FUNCTIONS)
def test_decreasing_is_one_below_and_zero_above_the_ramp(fn):
    np.testing.assert_allclose(
        fn([-1.0, 0.2, 0.8, 5.0], 0.2, 0.8, increasing=False), [1.0, 1.0, 0.0, 0.0]
    )


@pytest.mark.parametrize("fn", PRIORITY_FUNCTIONS)
@pytest.mark.parametrize("increasing", [True, False])
def test_output_is_bounded_and_monotonic(fn, increasing):
    x = np.linspace(-1.0, 2.0, 301)
    y = fn(x, 0.0, 1.0, increasing=increasing)
    assert y.shape == x.shape
    assert np.all((y >= 0.0) & (y <= 1.0))
    steps = np.diff(y)
    assert np.all(steps >= -1e-12) if increasing else np.all(steps <= 1e-12)


@pytest.mark.parametrize("fn", PRIORITY_FUNCTIONS)
def test_increasing_and_decreasing_are_complementary(fn):
    x = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(fn(x, 0.1, 0.9) + fn(x, 0.1, 0.9, increasing=False), 1.0)


@pytest.mark.parametrize("fn", PRIORITY_FUNCTIONS)
def test_scalar_input_returns_zero_dimensional_array(fn):
    result = fn(0.5, 0.0, 1.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == ()


@pytest.mark.parametrize("fn", PRIORITY_FUNCTIONS)
def test_degenerate_range_returns_constant(fn):
    x = np.array([0.0, 0.5, 1.0])
    np.testing.assert_array_equal(fn(x, 0.5, 0.5, increasing=True), 0.0)
    np.testing.assert_array_equal(fn(x, 0.5, 0.5, increasing=False), 1.0)


def test_linear_midpoint_is_one_half():
    assert float(basic_linear(0.5, 0.0, 1.0)) == pytest.approx(0.5)


def test_exponential_is_convex_and_steepness_controls_curvature():
    mild = float(exponential(0.5, 0.0, 1.0, a=1))
    steep = float(exponential(0.5, 0.0, 1.0, a=10))
    assert 0.0 < steep < mild < 0.5
