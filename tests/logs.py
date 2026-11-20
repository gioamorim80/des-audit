"""Small decision logs with a known truth, for the tests."""

import numpy as np
import pandas as pd


def two_arm_log(
    n: int = 40_000, seed: int = 0, low: float = 0.02, high: float = 0.06, base: float = 0.15, propensity: float = 0.5
) -> pd.DataFrame:
    """Return a randomized log: free shipping with a constant ``propensity``, a judgment uniform on
    [0, 1], and an offer effect of ``low`` below 0.5 and ``high`` at or above it."""
    rng = np.random.default_rng(seed)
    j = np.round(rng.random(n), 2)
    treated = rng.random(n) < propensity
    p = base + np.where(treated, np.where(j >= 0.5, high, low), 0.0)
    bought = rng.random(n) < p
    return pd.DataFrame(
        {
            "decision_id": np.arange(n),
            "action": np.where(treated, "free_shipping", "no_offer"),
            "propensity": np.where(treated, propensity, 1 - propensity),
            "outcome": bought.astype(int),
            "revenue": 50.0 * bought,
            "cost": 8.0 * (bought & treated),
            "ctx_pages": rng.integers(0, 10, n),
            "j_shipping": j,
            "j_noise": np.round(rng.random(n), 2),
        }
    )


def routing_log(n: int = 20_000, seed: int = 1) -> pd.DataFrame:
    """Return a three-arm log, each arm with probability 1/3, where ``j_signal`` decides which offer
    pays: above 0.5 10% off adds 30 points of purchase probability, below it free shipping does.
    ``j_noise`` means nothing and the context says nothing about which offer pays."""
    rng = np.random.default_rng(seed)
    j = rng.random(n)
    arm = rng.integers(0, 3, n)
    p = 0.1 + np.where(arm == 1, 0.3 * (j >= 0.5), 0.0) + np.where(arm == 2, 0.3 * (j < 0.5), 0.0)
    bought = rng.random(n) < p
    return pd.DataFrame(
        {
            "decision_id": np.arange(n),
            "shopper_id": rng.integers(0, n // 2, n),
            "action": np.array(["no_offer", "discount_10", "free_shipping"])[arm],
            "propensity": np.full(n, 1 / 3),
            "outcome": bought.astype(int),
            "revenue": 50.0 * bought,
            "cost": np.where(arm == 1, 5.0, np.where(arm == 2, 8.0, 0.0)) * bought,
            "ctx_pages": rng.integers(0, 10, n),
            "j_signal": j,
            "j_noise": rng.random(n),
        }
    )
