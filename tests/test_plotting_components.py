import numpy as np
import pytest

from explainit.plotting._common import (
    PriorityCurve,
    bar_width,
    infer_target_type,
    is_actionable_numeric,
    level_positions,
    normalize_target_values,
    normalize_vectors,
    numerical_path,
    pair_label,
    values_at,
)
from explainit.plotting.style import bar_color, pair_color
from explainit.priorities.linear import basic_linear


def test_normalize_vectors():
    assert normalize_vectors(None) is None
    assert normalize_vectors(3.0) is None
    one, two = normalize_vectors([1, 2]), normalize_vectors([[1, 2], [3, 4]])
    assert one is not None
    assert two is not None
    assert [v.tolist() for v in one] == [[1.0, 2.0]]
    assert [v.tolist() for v in two] == [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError, match="ndim=3"):
        normalize_vectors(np.zeros((1, 1, 1)))


def test_normalize_target_values():
    assert normalize_target_values(None, 2) is None
    assert normalize_target_values(0.5, 3) == [0.5, 0.5, 0.5]
    assert normalize_target_values([1, None, "x"], 4) == [1.0, None, None, None]


def test_values_at_skips_missing_and_non_finite_values():
    vectors = [np.array([1.0, np.nan]), np.array([2.0])]
    assert values_at(vectors, 3, 0) == [1.0, 2.0, None]
    assert values_at(vectors, 2, 1) == [None, None]
    assert values_at(None, 2, 0) == [None, None]


def test_labels_colors_and_widths():
    assert pair_label("Sample", 0, 1) == "Sample"
    assert pair_label("Sample", 1, 3) == "Sample #2"
    assert pair_color(0) == pair_color(8)
    assert bar_color(1) == bar_color(7)
    assert [bar_width(n) for n in (1, 2, 4, 6, 9)] == [0.1, 0.4, 0.5, 0.6, 0.8]


@pytest.mark.parametrize(
    ("constraint", "expected"),
    [
        ({"min": 0.0, "max": 1.0, "function": abs}, True),
        ({"min": 0.5, "max": 0.5, "function": abs}, False),
        ({"min": 0.0, "max": 1.0, "function": None}, False),
        ({"min": None, "max": 1.0, "function": abs}, False),
        (None, False),
    ],
)
def test_is_actionable_numeric(constraint, expected):
    assert is_actionable_numeric(constraint) is expected


def test_infer_target_type():
    assert infer_target_type(np.array([0, 1, 1, 0])) == "categorical"
    assert infer_target_type(np.linspace(0, 1, 50)) == "continuous"
    assert infer_target_type(np.array([np.nan])) == "unknown"


def test_priority_curve():
    curve = PriorityCurve.from_constraint(
        {"min": 0.0, "max": 1.0, "function": lambda x: basic_linear(x, 0.0, 1.0)}, points=11
    )
    assert curve.at(0.25) == pytest.approx(0.25)
    assert curve.at(1.5) is None
    assert curve.at(None) is None
    assert not curve.is_constant


def test_level_positions():
    x = np.linspace(0.0, 1.0, 101)
    ramp = np.clip((x - 0.2) / 0.6, 0.0, 1.0)  # 0 until 0.2, 1 from 0.8
    assert level_positions(x, ramp, 0.5) == pytest.approx([0.5])
    assert level_positions(x, ramp, 0.0) == pytest.approx([0.2])
    assert level_positions(x, ramp, 1.0) == pytest.approx([0.8])
    assert level_positions(x[:1], ramp[:1], 0.5) == []


def test_numerical_path(tmp_path):
    assert numerical_path(tmp_path, "p", 3, "loan amount").name == "p_3_loan_amount.png"
    assert numerical_path(tmp_path, "p", 3, None).name == "p_3_feature_3.png"
