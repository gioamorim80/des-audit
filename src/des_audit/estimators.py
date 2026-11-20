"""Shared estimators: score bins, self-normalized IPW by bin, propensity checks, the percentile
bootstrap (over decisions or over clusters of decisions) and ridge regression."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

MIN_ARM_ROWS_PER_BIN = 30
FLOOR_PROPENSITY = 0.01
BALANCE_TOLERANCE = 0.10
# A bin's interval is shown only if the statistic could be computed in nearly every resample;
# otherwise the resamples that could compute it are a biased subset.
MAX_UNESTIMABLE_SHARE = 0.01


@dataclass(frozen=True)
class Interval:
    """A point estimate with an interval. ``None`` means not estimable or not shown."""

    point: float | None
    lo: float | None
    hi: float | None

    def text(self, digits: int = 3) -> str:
        """Return ``point [lo, hi]``, the point alone, or ``n/a``."""
        if self.point is None:
            return "n/a"
        if self.lo is None or self.hi is None:
            return f"{self.point:.{digits}f}"
        return f"{self.point:.{digits}f} [{self.lo:.{digits}f}, {self.hi:.{digits}f}]"

    def as_dict(self) -> dict[str, float | None]:
        """Return the three numbers for JSON."""
        return {"point": self.point, "lo": self.lo, "hi": self.hi}


def bin_edges(n_bins: int) -> np.ndarray:
    """Return ``n_bins + 1`` equal-width edges on [0, 1].

    Edges are k / n_bins, so a score of exactly 0.3 at 10 bins lands in bin 3, not bin 2 (an
    evenly spaced grid computed by repeated addition puts the edge at 0.30000000000000004).

    Raises:
        ValueError: If ``n_bins`` is below 1.
    """
    if n_bins < 1:
        raise ValueError(f"bins must be at least 1, got {n_bins}")
    return np.arange(n_bins + 1) / n_bins


def bin_index(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return each score's bin, bins ``[lo, hi)``; a score of exactly 1.0 lands in the last bin."""
    idx = np.searchsorted(edges, scores, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def merge_bins(counts: np.ndarray, edges: np.ndarray, min_rows: int) -> np.ndarray:
    """Merge adjacent bins left to right until every arm has at least ``min_rows`` rows in each.

    ``counts`` is (arms, bins): the rows of each compared arm per bin. The floor applies to each
    arm separately, so a bin with 500 control rows and one treated row is merged rather than given
    an interval that only reflects the control arm. A remainder at the right that stays below the
    floor joins the previous bin. Returns the merged edges.
    """
    groups: list[tuple[int, int]] = []
    start, total = 0, np.zeros(counts.shape[0])
    for b in range(counts.shape[1]):
        total = total + counts[:, b]
        if total.min() >= min_rows:
            groups.append((start, b))
            start, total = b + 1, np.zeros(counts.shape[0])
    if start < counts.shape[1]:
        if groups:
            first, _ = groups.pop()
            groups.append((first, counts.shape[1] - 1))
        else:
            groups.append((0, counts.shape[1] - 1))
    return np.array([edges[0]] + [edges[last + 1] for _, last in groups])


def snipw_by_bin(
    bins: np.ndarray, in_arm: np.ndarray, values: np.ndarray, weight: np.ndarray, n_bins: int
) -> np.ndarray:
    """Return the self-normalized IPW mean of ``values`` per bin over the rows of one arm,
    Σ w·v ÷ Σ w, NaN where the arm has no row in the bin. With a constant propensity it is the
    plain mean."""
    num = np.bincount(bins[in_arm], weights=(weight * values)[in_arm], minlength=n_bins)
    den = np.bincount(bins[in_arm], weights=weight[in_arm], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def balance_by_bin(bins: np.ndarray, in_arm: np.ndarray, weight: np.ndarray, n_bins: int) -> np.ndarray:
    """Return Σ (1/π) over an arm's rows in each bin ÷ all rows in the bin.

    With positivity it is about 1 for every arm: each row stands for 1/π rows like it. Well below
    1 means some shoppers in the bin could not get that arm (π of 0 for them), so the arm's rate
    is estimated on a different population from the other arm's.
    """
    total = np.bincount(bins, minlength=n_bins)
    weighted = np.bincount(bins[in_arm], weights=weight[in_arm], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, weighted / total, np.nan)


def floor_warning(propensity: np.ndarray) -> str | None:
    """Return a warning when a logged propensity is below 0.01 (a weight above 100), else None."""
    p_min = float(propensity.min())
    if p_min < FLOOR_PROPENSITY:
        return (
            f"smallest logged propensity {p_min:.4g} (weight {1 / p_min:.4g}): a few decisions carry the IPW "
            "estimates and the intervals understate it"
        )
    return None


def counterfactual_warning(action: np.ndarray, propensity: np.ndarray, arms: tuple[str, ...]) -> str | None:
    """In a two-arm log, return a warning when some rows had less than 0.01 chance of the other
    arm, else None. A row logged with π = 1 had no chance of the other arm at all, and the two
    arms' rates are then estimated on different shoppers. With more arms the other arms'
    probabilities are not in the log; ``balance_by_bin`` is the check there."""
    if len(arms) != 2:
        return None
    other = 1.0 - propensity
    low = other < FLOOR_PROPENSITY
    if not low.any():
        return None
    return (
        f"{int(low.sum())} decisions had less than {FLOOR_PROPENSITY} chance of the other arm "
        f"({int((other == 0).sum())} had none): the arms are compared on different shoppers there"
    )


@dataclass(frozen=True)
class Bootstrap:
    """Percentile-bootstrap results per statistic.

    Attributes:
        lo: Lower bound, NaN where no resample could compute the statistic.
        hi: Upper bound.
        se: Standard deviation of the resampled statistic.
        unestimable: Share of resamples in which the statistic could not be computed.
        resamples: Number of resamples.
        seed: Seed of the resampling.
        clustered: Whether clusters, not single decisions, were resampled.
    """

    lo: np.ndarray
    hi: np.ndarray
    se: np.ndarray
    unestimable: np.ndarray
    resamples: int
    seed: int
    clustered: bool


def resample_rows(
    rng: np.random.Generator, n: int, clusters: np.ndarray | None, order=None, starts=None, sizes=None
) -> np.ndarray:
    """Return one bootstrap sample of row indices: n rows drawn with replacement, or, with
    clusters, as many clusters as there are drawn with replacement and all their rows taken."""
    if clusters is None:
        return rng.integers(0, n, size=n)
    picked = rng.integers(0, len(sizes), size=len(sizes))
    count = sizes[picked]
    offset = np.repeat(np.cumsum(count) - count, count)
    return order[np.repeat(starts[picked], count) + np.arange(int(count.sum())) - offset]


def bootstrap(
    n: int,
    statistic: Callable[[np.ndarray], np.ndarray],
    resamples: int,
    seed: int,
    alpha: float = 0.05,
    clusters: np.ndarray | None = None,
) -> Bootstrap:
    """Resample the decisions (or clusters of them) and return percentile bounds per statistic.

    ``statistic`` takes an index array and returns a float array, NaN where it cannot be
    computed on that resample.

    Raises:
        ValueError: If ``resamples`` is below 2 or ``alpha`` is not in (0, 1).
    """
    if resamples < 2:
        raise ValueError(f"resamples must be at least 2, got {resamples}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    rng = np.random.default_rng(seed)
    order = starts = sizes = None
    if clusters is not None:
        order = np.argsort(clusters, kind="stable")
        sizes = np.bincount(clusters)
        starts = np.cumsum(sizes) - sizes
    draws = np.stack([statistic(resample_rows(rng, n, clusters, order, starts, sizes)) for _ in range(resamples)])
    finite = np.isfinite(draws)
    some = finite.any(axis=0)
    lo, hi, se = (np.full(draws.shape[1], np.nan) for _ in range(3))
    lo[some] = np.nanpercentile(draws[:, some], 100 * alpha / 2, axis=0)
    hi[some] = np.nanpercentile(draws[:, some], 100 * (1 - alpha / 2), axis=0)
    se[some] = np.nanstd(draws[:, some], axis=0, ddof=1)
    return Bootstrap(lo, hi, se, 1.0 - finite.mean(axis=0), resamples, seed, clusters is not None)


def interval(point: float, boot: Bootstrap, k: int) -> Interval:
    """Return statistic ``k``'s interval; NaN becomes None, and the bounds are withheld when the
    statistic could not be computed in more than 1% of resamples."""
    if np.isnan(point):
        return Interval(None, None, None)
    if boot.unestimable[k] > MAX_UNESTIMABLE_SHARE or np.isnan(boot.lo[k]):
        return Interval(float(point), None, None)
    return Interval(float(point), float(boot.lo[k]), float(boot.hi[k]))


@dataclass(frozen=True)
class Ridge:
    """A fitted ridge regression on standardized features.

    Attributes:
        mean: Per-feature mean on the fit rows.
        scale: Per-feature standard deviation on the fit rows (1 where constant).
        coef: Coefficients on the standardized features.
        intercept: Intercept, not penalized.
    """

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predictions for rows ``x`` of shape (n, d)."""
        return ((np.asarray(x, dtype=float) - self.mean) / self.scale) @ self.coef + self.intercept


def ridge(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> Ridge:
    """Fit y ≈ intercept + z·coef with an L2 penalty ``lam`` on coef, z the standardized x.

    Raises:
        ValueError: If x is not 2-d, the lengths differ, there are no rows, or ``lam`` < 0.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError(f"ridge needs a 2-d x with one row per y; got {x.shape} and {y.shape}")
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = (x - mean) / scale
    # Centering y and z takes the intercept out of the penalized system.
    coef = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - y.mean())) if z.shape[1] else np.zeros(0)
    return Ridge(mean, scale, coef, float(y.mean()))
