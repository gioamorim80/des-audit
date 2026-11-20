"""Decisions a test needs."""

import pytest

from des_audit.power import effect_difference, mean_difference


def test_effect_difference_worked_example():
    """Rate 0.17, a quarter high, half treated, 2 points: about 59,100 decisions."""
    assert effect_difference(0.02, 0.17, 0.25).total == pytest.approx(59_065, abs=5)


def test_mean_difference_example():
    """σ 20 and a difference of 1: 2 × 400 × 2.8016² = 6,279 per arm."""
    r = mean_difference(1.0, 20.0, arms=3, decisions_per_month=5_000)
    assert r.per_arm == 6_279 and r.total == 3 * 6_279
    assert r.months == pytest.approx(3 * 6_279 / 5_000)


def test_halving_the_difference_quadruples_the_decisions():
    """n scales as 1/Δ²."""
    assert effect_difference(0.01, 0.17, 0.25).total == pytest.approx(
        4 * effect_difference(0.02, 0.17, 0.25).total, rel=1e-3
    )


def test_a_balanced_cut_needs_the_fewest_decisions():
    """For a given difference, half the decisions high is the cheapest split."""
    assert effect_difference(0.02, 0.17, 0.5).total < effect_difference(0.02, 0.17, 0.25).total


@pytest.mark.parametrize(
    "call",
    [
        lambda: mean_difference(0, 1),
        lambda: mean_difference(1, 1, arms=1),
        lambda: effect_difference(0.02, 1.2, 0.5),
        lambda: effect_difference(0.02, 0.2, 0.5, alpha=0),
    ],
)
def test_bad_inputs_are_refused(call):
    """Out-of-range inputs raise."""
    with pytest.raises(ValueError):
        call()
