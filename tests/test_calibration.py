"""The offer's effect by bin of a judgment, on logs with a known truth."""

import numpy as np
import pandas as pd
import pytest

from des_audit.calibration import calibration
from des_audit.log import validate

from logs import two_arm_log


@pytest.fixture(scope="module")
def result():
    """Calibration of the shipping judgment on a 40,000-decision log: effect 0.02 below 0.5, 0.06 above."""
    return calibration(validate(two_arm_log()), "j_shipping", resamples=500)


def test_per_bin_effects_cover_the_truth(result):
    """Both bins' intervals contain their true effect, and the difference's contains 0.04."""
    low, high = result.bins
    assert low.purchase_effect.lo < 0.02 < low.purchase_effect.hi
    assert high.purchase_effect.lo < 0.06 < high.purchase_effect.hi
    assert result.contrast.purchase.lo < 0.04 < result.contrast.purchase.hi


def test_the_difference_is_last_bin_minus_first(result):
    """The contrast's point is the high bin's effect minus the low bin's."""
    low, high = result.bins
    assert result.contrast.purchase.point == pytest.approx(high.purchase_effect.point - low.purchase_effect.point)


def test_constant_propensity_is_the_difference_in_means():
    """At a constant propensity each bin's effect is the plain difference in purchase rates."""
    frame = two_arm_log(5_000, seed=2)
    r = calibration(validate(frame), "j_shipping", resamples=50)
    high = frame[frame["j_shipping"] >= 0.5]
    plain = (
        high.loc[high["action"] == "free_shipping", "outcome"].mean()
        - high.loc[high["action"] == "no_offer", "outcome"].mean()
    )
    assert r.bins[1].purchase_effect.point == pytest.approx(plain)


def test_standard_error_is_the_textbook_one_at_a_constant_propensity(result):
    """With a constant propensity the difference's variance is Σ p(1 − p)/n over the four cells."""
    frame = two_arm_log()
    cells = frame.groupby([frame["j_shipping"] >= 0.5, "action"])["outcome"]
    variance = (cells.mean() * (1 - cells.mean()) / cells.count()).sum()
    assert result.contrast.se == pytest.approx(np.sqrt(variance))
    assert result.contrast.detectable == pytest.approx(2.8016 * np.sqrt(variance), rel=1e-4)


def test_a_bin_with_one_treated_row_is_merged_not_reported():
    """40 high-score chats with one treated: the high bin is merged into the low one."""
    frame = two_arm_log(2_000, seed=3)
    frame.loc[frame["j_shipping"] >= 0.5, "action"] = "no_offer"
    frame.loc[frame["j_shipping"] >= 0.5, "propensity"] = 0.5
    high = frame.index[frame["j_shipping"] >= 0.5][:40]
    frame = pd.concat([frame[frame["j_shipping"] < 0.5], frame.loc[high]])
    frame.loc[high[0], "action"] = "free_shipping"
    r = calibration(validate(frame), "j_shipping", resamples=50)
    assert len(r.bins) == 1
    assert any("merged" in w for w in r.warnings)


def test_a_positivity_violation_is_flagged():
    """A fifth of the shoppers forced to the control with propensity 1: the check says so."""
    frame = two_arm_log(10_000, seed=4)
    forced = np.random.default_rng(5).random(len(frame)) < 0.2
    frame.loc[forced, "action"] = "no_offer"
    frame.loc[forced, "propensity"] = 1.0
    r = calibration(validate(frame), "j_shipping", resamples=50)
    assert any("had none" in w for w in r.warnings)
    assert any("weighted count" in w for w in r.warnings)


def test_no_warning_on_a_clean_randomized_log(result):
    """A 50/50 coin flip with nothing missing raises no warning."""
    assert result.warnings == []


def test_missing_judgments_are_dropped_and_a_gap_between_arms_is_flagged():
    """Rows missing the judgment are dropped; missing it in one arm far more is a warning."""
    frame = two_arm_log(4_000, seed=6)
    treated = frame.index[frame["action"] == "free_shipping"][:200]
    frame.loc[treated, "j_shipping"] = np.nan
    r = calibration(validate(frame), "j_shipping", resamples=50)
    assert r.n == 4_000 - 200
    assert r.missing_by_arm["no_offer"] == 0.0
    assert any("missing more often" in w for w in r.warnings)
    assert "dropped for a missing judgment: 9.9% of free_shipping" in r.table()


def test_clusters_widen_the_interval_when_rows_repeat():
    """Each row copied three times into one cluster: resampling clusters gives a wider interval."""
    frame = two_arm_log(3_000, seed=7)
    copies = pd.concat([frame.assign(shopper=frame["decision_id"])] * 3, ignore_index=True)
    copies["decision_id"] = np.arange(len(copies))
    log = validate(copies)
    plain = calibration(log, "j_shipping", resamples=300)
    clustered = calibration(log, "j_shipping", resamples=300, cluster="shopper")
    width = lambda r: r.contrast.purchase.hi - r.contrast.purchase.lo  # noqa: E731
    assert width(clustered) > 1.4 * width(plain)
    assert clustered.contrast.se > 1.6 * plain.contrast.se
