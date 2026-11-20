"""Off-policy value of deciding with the judgments."""

import numpy as np
import pytest

from des_audit.log import validate
from des_audit.value import split, value

from logs import routing_log


@pytest.fixture(scope="module")
def result():
    """Value of each judgment and of both together, on a log where j_signal routes the offer."""
    return value(validate(routing_log()), ["j_signal", "j_noise"], joint=True, resamples=300)


def test_the_judgment_that_routes_the_offer_is_worth_something(result):
    """Deciding with j_signal beats the context alone, by both estimators."""
    signal = next(r for r in result.rows if r.name == "ctx + j_signal")
    assert signal.dr_diff.lo > 0 and signal.ipw_diff.lo > 0


def test_a_noise_judgment_is_not(result):
    """Deciding with j_noise cannot be told from the context alone."""
    noise = next(r for r in result.rows if r.name == "ctx + j_noise")
    assert noise.dr_diff.lo < 0 < noise.dr_diff.hi


def test_the_joint_policy_is_last(result):
    """With joint the last row uses every judgment."""
    assert result.rows[-1].name == "ctx + all 2 judgments"
    assert result.rows[-1].dr_diff.lo > 0


def test_split_keeps_clusters_whole():
    """No cluster is on both sides of the split."""
    clusters = np.random.default_rng(0).integers(0, 50, 1000)
    fit, held = split(1000, 0.5, 0, clusters)
    assert not set(clusters[fit]) & set(clusters[held])
    assert len(fit) + len(held) == 1000


def test_rows_missing_a_judgment_in_use_are_dropped():
    """A missing j_noise drops the row only when j_noise is used."""
    frame = routing_log(4_000)
    frame.loc[:99, "j_noise"] = np.nan
    log = validate(frame)
    assert value(log, ["j_signal"], resamples=20).dropped == 0
    assert value(log, ["j_noise"], resamples=20).dropped == 100


def test_nothing_asked_is_refused():
    """value needs judgments or joint."""
    with pytest.raises(ValueError, match="needs judgments"):
        value(validate(routing_log(1_000)), [])
