"""Checking the log."""

import numpy as np
import pytest

from des_audit.log import LogError, validate

from logs import two_arm_log


def test_a_good_log_passes():
    """A complete log gives its arms, control, context and judgment columns."""
    log = validate(two_arm_log(100))
    assert log.arms == ("free_shipping", "no_offer")
    assert log.ctx_columns == ("ctx_pages",) and log.judgment_columns == ("j_shipping", "j_noise")


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda f: f.drop(columns="propensity"), "missing required"),
        (lambda f: f.assign(propensity=0.0), "propensity"),
        (lambda f: f.assign(propensity=1.5), "propensity"),
        (lambda f: f.assign(outcome=2), "outcome"),
        (lambda f: f.assign(j_shipping=1.2), "must lie in"),
        (lambda f: f.assign(ctx_pages=np.nan), "missing values"),
        (lambda f: f.assign(decision_id=0), "repeats"),
        (lambda f: f.assign(action="no_offer"), "single arm"),
        (lambda f: f.iloc[:0], "empty"),
    ],
)
def test_bad_logs_are_refused_with_the_reason(change, message):
    """Each problem is refused and named."""
    with pytest.raises(LogError, match=message):
        validate(change(two_arm_log(100)))


def test_control_must_have_rows():
    """A control arm that never appears is refused."""
    with pytest.raises(LogError, match="control arm"):
        validate(two_arm_log(100), control="nothing")


def test_missing_judgments_are_allowed_and_counted():
    """A judgment may be missing on some rows; the share is reported per arm."""
    frame = two_arm_log(100)
    frame.loc[frame["action"] == "no_offer", "j_shipping"] = np.nan
    log = validate(frame)
    assert not log.complete(["j_shipping"])[frame["action"] == "no_offer"].any()
    assert log.missing_by_arm(["j_shipping"]) == {"free_shipping": 0.0, "no_offer": 1.0}
    assert log.complete(["j_noise"]).all()


def test_clusters():
    """A cluster column becomes codes; a missing one is refused."""
    frame = two_arm_log(6).assign(shopper=["a", "b", "a", "c", "b", "a"])
    log = validate(frame)
    assert log.clusters("shopper").tolist() == [0, 1, 0, 2, 1, 0]
    assert log.clusters(None) is None
    with pytest.raises(LogError, match="cluster column"):
        log.clusters("nope")


def test_an_empty_unused_judgment_is_accepted():
    """A judgment column with no values refuses nothing until a command uses it."""
    from des_audit.calibration import calibration

    log = validate(two_arm_log(200).assign(j_noise=np.nan))
    assert calibration(log, "j_shipping", resamples=20).n == 200
    with pytest.raises(LogError, match="no no_offer or no free_shipping"):
        calibration(log, "j_noise", resamples=20)
