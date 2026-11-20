"""Bins, weights, checks and the bootstrap."""

import numpy as np
import pytest

from des_audit.estimators import (
    bin_edges,
    bin_index,
    bootstrap,
    counterfactual_warning,
    merge_bins,
    resample_rows,
    ridge,
    snipw_by_bin,
)


def test_two_decimal_scores_land_in_their_own_bin():
    """At 10 bins, 0.3, 0.6 and 0.7 open bins 3, 6 and 7; 0 and 1 go to the first and last."""
    edges = bin_edges(10)
    assert bin_index(np.array([0.0, 0.29, 0.3, 0.6, 0.7, 1.0]), edges).tolist() == [0, 2, 3, 6, 7, 9]


def test_the_row_floor_applies_to_each_arm():
    """A bin with 500 control rows and one treated row is merged; 40 and 40 is enough."""
    edges = bin_edges(2)
    assert merge_bins(np.array([[500, 500], [1, 500]]), edges, 30).tolist() == [0.0, 1.0]
    assert merge_bins(np.array([[40, 40], [40, 40]]), edges, 30).tolist() == [0.0, 0.5, 1.0]


def test_constant_propensity_gives_the_plain_mean():
    """Self-normalized IPW at a constant propensity is the mean of the arm's outcomes."""
    y = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
    out = snipw_by_bin(np.zeros(5, dtype=int), np.ones(5, dtype=bool), y, np.full(5, 2.0), 1)
    assert out[0] == pytest.approx(0.6)


def test_cluster_resampling_takes_whole_clusters():
    """Every cluster drawn comes with all its rows."""
    clusters = np.array([0, 0, 0, 1, 2, 2])
    order = np.argsort(clusters, kind="stable")
    sizes = np.bincount(clusters)
    rows = resample_rows(np.random.default_rng(3), 6, clusters, order, np.cumsum(sizes) - sizes, sizes)
    counts = np.bincount(clusters[rows], minlength=3)
    assert all(c % s == 0 for c, s in zip(counts, sizes, strict=True))


def test_bootstrap_reports_resamples_that_could_not_compute():
    """A statistic that needs row 0 cannot be computed in about (1 − 1/n)^n of resamples."""
    boot = bootstrap(20, lambda s: np.array([1.0 if 0 in s else np.nan]), 4000, 0)
    assert boot.unestimable[0] == pytest.approx((1 - 1 / 20) ** 20, abs=0.03)


def test_ridge_recovers_a_line():
    """With no penalty ridge is least squares."""
    x = np.arange(10.0).reshape(-1, 1)
    model = ridge(x, 3 + 2 * x[:, 0], lam=0.0)
    assert model.predict(np.array([[20.0]]))[0] == pytest.approx(43.0)


def test_counterfactual_check_in_two_arm_logs():
    """A row logged with propensity 1 had no chance of the other arm; with three arms the other
    arms' chances are not in the log and the check says nothing."""
    action = np.array(["no_offer", "free_shipping", "no_offer"])
    assert "had none" in counterfactual_warning(action, np.array([1.0, 0.5, 0.5]), ("free_shipping", "no_offer"))
    assert counterfactual_warning(action, np.array([0.5, 0.5, 0.5]), ("free_shipping", "no_offer")) is None
    assert counterfactual_warning(action, np.array([1.0, 0.5, 0.5]), ("a", "b", "c")) is None
