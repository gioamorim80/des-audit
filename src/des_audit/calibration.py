"""``calibration``: the effect of an offer by bin of a judgment's score, and the difference in
effect between the highest and the lowest bin.

In each bin the purchase rate and the profit per decision under the control arm and under the
treatment arm are estimated by self-normalized inverse-propensity weighting (each row counts
1/propensity; with a constant propensity this is the plain difference in means). The effect in a
bin is treatment minus control: a group average treatment effect, E[Y(1) − Y(0) | score in bin].
Intervals are percentile-bootstrap intervals from one joint resampling of decisions (or of
clusters), so the per-bin effects and their difference come from the same draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from des_audit._table import format_table
from des_audit.estimators import (
    BALANCE_TOLERANCE,
    MIN_ARM_ROWS_PER_BIN,
    Interval,
    balance_by_bin,
    bin_edges,
    bin_index,
    bootstrap,
    counterfactual_warning,
    floor_warning,
    interval,
    merge_bins,
    snipw_by_bin,
)
from des_audit.log import Log, LogError

MISSING_GAP = 0.02


@dataclass(frozen=True)
class BinRow:
    """One (possibly merged) score bin.

    Attributes:
        lo: Lower edge; the bin is ``[lo, hi)`` except the last, which is closed.
        hi: Upper edge.
        n_control: Control decisions in the bin.
        n_treatment: Treatment decisions in the bin.
        control_rate: Purchase rate under the control arm.
        purchase_effect: Treatment purchase rate minus control purchase rate.
        profit_effect: Treatment minus control in revenue − cost per decision.
        balance: Per arm, Σ 1/propensity over the arm's rows ÷ the bin's rows (about 1 when every
            shopper in the bin could get either arm).
        flag: Why a number is missing or withheld, or None.
    """

    lo: float
    hi: float
    n_control: int
    n_treatment: int
    control_rate: Interval
    purchase_effect: Interval
    profit_effect: Interval
    balance: dict[str, float]
    flag: str | None

    def as_dict(self) -> dict:
        """Return the row for JSON."""
        return {
            "lo": self.lo,
            "hi": self.hi,
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
            "control_rate": self.control_rate.as_dict(),
            "purchase_effect": self.purchase_effect.as_dict(),
            "profit_effect": self.profit_effect.as_dict(),
            "balance": dict(self.balance),
            "flag": self.flag,
        }


@dataclass(frozen=True)
class Contrast:
    """The last bin's effect minus the first bin's, with what this log could detect.

    Attributes:
        purchase: Difference in purchase effect, with its interval.
        profit: Difference in profit effect, with its interval.
        se: Standard error of the purchase difference by the delta method (cluster-robust with
            clusters); it carries no resampling noise.
        detectable: The purchase difference this log detects with probability ``power`` in a
            two-sided test at level alpha: (z(1 − alpha/2) + z(power)) × se.
        decisions_needed: Decisions in the two arms for that power if the true difference were the
            one observed, n × (detectable / observed)²; None when the observed difference is 0.
        power: The power used.
    """

    purchase: Interval
    profit: Interval
    se: float | None
    detectable: float | None
    decisions_needed: int | None
    power: float

    def as_dict(self) -> dict:
        """Return the contrast for JSON."""
        return {
            "purchase": self.purchase.as_dict(),
            "profit": self.profit.as_dict(),
            "se": self.se,
            "detectable": self.detectable,
            "decisions_needed": self.decisions_needed,
            "power": self.power,
        }


@dataclass(frozen=True)
class CalibrationResult:
    """The effect of one treatment arm against the control by bin of one judgment.

    Attributes:
        judgment: The judgment column.
        control: Control arm.
        treatment: Treatment arm.
        n: Decisions of the two arms used (rows missing the judgment are dropped).
        bins: One row per (merged) bin, left to right.
        contrast: Last bin minus first bin; None with a single bin.
        alpha: Intervals are at level 1 − alpha.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.
        cluster: The cluster column resampled, or None for single decisions.
        missing_by_arm: Share of each arm's rows missing the judgment.
        warnings: Checks that failed, possibly none.
    """

    judgment: str
    control: str
    treatment: str
    n: int
    bins: list[BinRow]
    contrast: Contrast | None
    alpha: float
    resamples: int
    seed: int
    cluster: str | None
    missing_by_arm: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    def table(self) -> str:
        """Return the readable output."""
        level = f"{100 * (1 - self.alpha):g}%"
        unit = f"clusters of {self.cluster}" if self.cluster else "decisions"
        head = (
            f"effect of {self.treatment} against {self.control} by {self.judgment}: {self.n:,} decisions, "
            f"self-normalized IPW, {level} percentile-bootstrap intervals from {self.resamples:,} resamples "
            f"of {unit} (seed {self.seed})"
        )
        rows = [
            (
                f"[{b.lo:.2f}, {b.hi:.2f}{']' if i == len(self.bins) - 1 else ')'}",
                f"{b.n_control:,}",
                f"{b.n_treatment:,}",
                b.control_rate.text(),
                b.purchase_effect.text(),
                b.profit_effect.text(2),
                b.flag or "",
            )
            for i, b in enumerate(self.bins)
        ]
        dropped = {a: share for a, share in self.missing_by_arm.items() if share > 0}
        if dropped:
            head += "; dropped for a missing judgment: " + ", ".join(
                f"{100 * share:.1f}% of {a}" for a, share in dropped.items()
            )
        lines = [
            head,
            "",
            format_table(
                ("score", "n control", "n treatment", "control rate", "purchase effect", "profit effect", "flag"), rows
            ),
        ]
        c = self.contrast
        if c is not None:
            lines += [
                "",
                f"last bin minus first: purchase effect {c.purchase.text()}, profit effect {c.profit.text(2)}",
            ]
            if c.detectable is not None:
                text = (
                    f"This log detects a difference in purchase effect of {c.detectable:.3f} with "
                    f"{100 * c.power:g}% power (two-sided, alpha {self.alpha:g})"
                )
                if c.decisions_needed is not None and c.purchase.point is not None:
                    text += (
                        f"; a true difference of {abs(c.purchase.point):.3f} would need about "
                        f"{c.decisions_needed:,} decisions in these two arms"
                    )
                lines.append(text + ".")
        return "\n".join(lines + [f"warning: {w}" for w in self.warnings])

    def as_dict(self) -> dict:
        """Return the result for JSON."""
        return {
            "command": "calibration",
            "judgment": self.judgment,
            "control": self.control,
            "treatment": self.treatment,
            "n": self.n,
            "alpha": self.alpha,
            "resamples": self.resamples,
            "seed": self.seed,
            "cluster": self.cluster,
            "missing_by_arm": dict(self.missing_by_arm),
            "bins": [b.as_dict() for b in self.bins],
            "contrast": None if self.contrast is None else self.contrast.as_dict(),
            "warnings": list(self.warnings),
        }


def _contrast_se(
    idx: np.ndarray,
    is_c: np.ndarray,
    is_t: np.ndarray,
    outcome: np.ndarray,
    weight: np.ndarray,
    k: int,
    clusters: np.ndarray | None,
) -> float | None:
    """Return the delta-method standard error of (last bin's purchase effect − first bin's).

    Each cell's self-normalized mean Σ w·y ÷ Σ w moves with row i by ψ_i = w_i (y_i − mean) ÷ Σ w
    for the rows in the cell; the contrast's ψ adds the four cells' with signs, and its variance is
    Σ ψ_i², or Σ over clusters of (Σ ψ_i within the cluster)². With a constant propensity a cell
    contributes p(1 − p)/n, the textbook variance of a proportion.
    """
    psi = np.zeros(len(idx))
    for b, sign_b in ((k - 1, 1.0), (0, -1.0)):
        for arm, sign_a in ((is_t, 1.0), (is_c, -1.0)):
            cell = arm & (idx == b)
            total = weight[cell].sum()
            if total <= 0:
                return None
            mean = (weight[cell] * outcome[cell]).sum() / total
            psi[cell] += sign_b * sign_a * weight[cell] * (outcome[cell] - mean) / total
    if clusters is not None:
        psi = np.bincount(clusters, weights=psi)
    return float(np.sqrt((psi**2).sum()))


def calibration(
    log: Log,
    judgment: str,
    treatment: str | None = None,
    bins: int = 2,
    resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    min_rows: int = MIN_ARM_ROWS_PER_BIN,
    cluster: str | None = None,
    power: float = 0.8,
) -> CalibrationResult:
    """Estimate the treatment's effect against the control by bin of a judgment's score.

    Rows of other arms are not used. Rows missing the judgment are dropped, and the share missing
    is reported per arm. Adjacent bins are merged until each holds ``min_rows`` rows of each of
    the two arms (0 turns merging off).

    Raises:
        LogError: If the judgment or the treatment is not in the log.
        ValueError: If ``bins`` < 1, ``resamples`` < 2, or ``alpha`` or ``power`` is not in (0, 1).
    """
    if not 0 < power < 1:
        raise ValueError(f"power must lie in (0, 1), got {power}")
    scores = log.judgment(judgment)
    treatment = log.treatment_or_default(treatment)
    edges = bin_edges(bins)
    warnings: list[str] = []

    present = ~np.isnan(scores)
    action = log.frame["action"].to_numpy()
    pair = (action == log.control) | (action == treatment)
    missing = {a: float((~present[action == a]).mean()) for a in (log.control, treatment)}
    if max(missing.values()) - min(missing.values()) > MISSING_GAP:
        warnings.append(
            f"the judgment is missing more often in one arm ({missing}): if it was read after the "
            "offer, the offer may have changed it"
        )
    used = present & pair
    if not (used & (action == log.control)).any() or not (used & (action == treatment)).any():
        raise LogError(f"no {log.control} or no {treatment} decision has a value of {judgment}")
    s = scores[used]
    is_c, is_t = action[used] == log.control, action[used] == treatment
    propensity = log.frame["propensity"].to_numpy(dtype=float)[used]
    weight = 1.0 / propensity
    outcome = log.frame["outcome"].to_numpy(dtype=float)[used]
    profit = (log.frame["revenue"].to_numpy(dtype=float) - log.frame["cost"].to_numpy(dtype=float))[used]
    clusters = log.clusters(cluster)
    clusters = None if clusters is None else np.unique(clusters[used], return_inverse=True)[1]

    if min_rows > 0:
        idx0 = bin_index(s, edges)
        counts = np.stack([np.bincount(idx0[m], minlength=bins) for m in (is_c, is_t)])
        merged = merge_bins(counts, edges, min_rows)
        if len(merged) < len(edges):
            warnings.append(
                f"{bins} bins merged into {len(merged) - 1} so each holds at least {min_rows} rows of each arm"
            )
        edges = merged
    k = len(edges) - 1
    idx = bin_index(s, edges)

    def statistic(sample: np.ndarray) -> np.ndarray:
        b, c, t, w = idx[sample], is_c[sample], is_t[sample], weight[sample]
        rate_c, rate_t = snipw_by_bin(b, c, outcome[sample], w, k), snipw_by_bin(b, t, outcome[sample], w, k)
        cash_c, cash_t = snipw_by_bin(b, c, profit[sample], w, k), snipw_by_bin(b, t, profit[sample], w, k)
        purchase, cash = rate_t - rate_c, cash_t - cash_c
        return np.concatenate([rate_c, purchase, cash, [purchase[-1] - purchase[0], cash[-1] - cash[0]]])

    point = statistic(np.arange(len(s)))
    boot = bootstrap(len(s), statistic, resamples, seed, alpha, clusters)

    # Balance uses every complete row in the bin, whatever its arm, as the population.
    all_rows = present
    idx_all = bin_index(scores[all_rows], edges)
    w_all = 1.0 / log.frame["propensity"].to_numpy(dtype=float)[all_rows]
    balance = {a: balance_by_bin(idx_all, action[all_rows] == a, w_all, k) for a in (log.control, treatment)}

    rows = []
    for b in range(k):
        n_c, n_t = int((is_c & (idx == b)).sum()), int((is_t & (idx == b)).sum())
        flag = None
        if n_c == 0 or n_t == 0:
            flag = "no control rows" if n_c == 0 else "no treatment rows"
        elif boot.unestimable[k + b] > 0.01:
            flag = f"interval withheld: not estimable in {100 * boot.unestimable[k + b]:.0f}% of resamples"
        rows.append(
            BinRow(
                float(edges[b]),
                float(edges[b + 1]),
                n_c,
                n_t,
                interval(point[b], boot, b),
                interval(point[k + b], boot, k + b),
                interval(point[2 * k + b], boot, 2 * k + b),
                {a: float(v[b]) for a, v in balance.items()},
                flag,
            )
        )
        off = [a for a, v in balance.items() if abs(v[b] - 1) > BALANCE_TOLERANCE]
        if off:
            warnings.append(
                f"bin [{edges[b]:.2f}, {edges[b + 1]:.2f}]: weighted count of {off} is "
                f"{', '.join(f'{balance[a][b]:.2f}' for a in off)} of the bin's rows, not about 1; "
                "some shoppers there could not get that arm"
            )

    contrast = None
    if k >= 2:
        diff = point[3 * k]
        se = _contrast_se(idx, is_c, is_t, outcome, weight, k, clusters)
        z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
        detectable = None if se is None else z * se
        needed = None
        if detectable is not None and np.isfinite(diff) and diff != 0:
            needed = int(round(len(s) * (detectable / diff) ** 2))
        contrast = Contrast(
            interval(diff, boot, 3 * k), interval(point[3 * k + 1], boot, 3 * k + 1), se, detectable, needed, power
        )
    else:
        warnings.append("a single bin: there is no difference between bins to estimate")

    for warning in (
        floor_warning(propensity),
        counterfactual_warning(action, log.frame["propensity"].to_numpy(dtype=float), log.arms),
    ):
        if warning:
            warnings.append(warning)
    return CalibrationResult(
        judgment, log.control, treatment, len(s), rows, contrast, alpha, resamples, seed, cluster, missing, warnings
    )
