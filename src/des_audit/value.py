"""``value``: what deciding with the judgments is worth, in revenue − cost per decision, estimated
off-policy on decisions the policies were not fit on.

The decisions are split at random (by cluster when one is given). On the fit part a greedy policy
is fit per feature set: one ridge regression of revenue − cost per arm, the policy picking the arm
with the highest prediction. The feature sets are the context alone, the context plus one
judgment at a time and, with ``joint``, the context plus every judgment. On the evaluation part
each policy is valued by self-normalized IPW and by the doubly robust estimator (Dudík, Langford
and Li 2011), and each is compared with the context-only policy by a paired bootstrap of the
difference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from des_audit._table import format_table
from des_audit.estimators import Interval, bootstrap, counterfactual_warning, floor_warning, interval, ridge
from des_audit.log import Log, LogError

BASE_NAME = "ctx only"
MIN_ARM_ROWS = 10
MIN_MATCHED = 30


@dataclass(frozen=True)
class PolicyValue:
    """One greedy policy's value on the evaluation part.

    Attributes:
        name: ``ctx only``, ``ctx + <judgment>`` or ``ctx + all <n> judgments``.
        matched: Evaluation decisions where the logged action equals the policy's choice.
        ipw: Self-normalized IPW value per decision, or None when no decision matched.
        dr: Doubly robust value per decision.
        ipw_diff: IPW difference against the context-only policy; None for that policy.
        dr_diff: DR difference against the context-only policy; None for that policy.
        arm_shares: Share of evaluation decisions on which the policy picks each arm.
    """

    name: str
    matched: int
    ipw: float | None
    dr: float
    ipw_diff: Interval | None
    dr_diff: Interval | None
    arm_shares: dict[str, float]

    def as_dict(self) -> dict:
        """Return the row for JSON."""
        return {
            "name": self.name,
            "matched": self.matched,
            "ipw": self.ipw,
            "dr": self.dr,
            "ipw_diff": None if self.ipw_diff is None else self.ipw_diff.as_dict(),
            "dr_diff": None if self.dr_diff is None else self.dr_diff.as_dict(),
            "arm_shares": dict(self.arm_shares),
        }


@dataclass(frozen=True)
class ValueResult:
    """The ``value`` command's result.

    Attributes:
        arms: Arm names, sorted.
        n_fit: Decisions the policies were fit on.
        n_eval: Decisions they were valued on.
        dropped: Decisions dropped for a missing judgment.
        base: The context-only policy.
        rows: The other policies, the joint one last.
        alpha: Intervals are at level 1 − alpha.
        resamples: Bootstrap resamples.
        seed: Seed of the split and the bootstrap.
        cluster: The cluster column, or None.
        warnings: Checks that failed, possibly none.
    """

    arms: tuple[str, ...]
    n_fit: int
    n_eval: int
    dropped: int
    base: PolicyValue
    rows: list[PolicyValue]
    alpha: float
    resamples: int
    seed: int
    cluster: str | None
    warnings: list[str]

    def table(self) -> str:
        """Return the readable output."""
        unit = f"clusters of {self.cluster}" if self.cluster else "decisions"
        head = (
            f"value of deciding with the judgments: greedy ridge policies over {len(self.arms)} arms, fit on "
            f"{self.n_fit:,} decisions and valued on the other {self.n_eval:,}; differences against the "
            f"context-only policy in revenue − cost per decision, {100 * (1 - self.alpha):g}% paired bootstrap "
            f"intervals from {self.resamples:,} resamples of {unit} (seed {self.seed})"
        )
        rows = [
            (
                r.name,
                f"{r.matched:,}",
                "n/a" if r.ipw_diff is None else r.ipw_diff.text(2),
                "n/a" if r.dr_diff is None else r.dr_diff.text(2),
            )
            for r in self.rows
        ]
        base_ipw = "n/a" if self.base.ipw is None else f"{self.base.ipw:.2f}"
        lines = [
            head,
            "",
            format_table(("policy", "matched", "IPW difference", "DR difference"), rows),
            "",
            f"The context-only policy is worth {base_ipw} (IPW) and {self.base.dr:.2f} (DR) per decision.",
        ]
        if self.dropped:
            lines.append(f"{self.dropped:,} decisions missing a judgment in use were dropped.")
        return "\n".join(lines + [f"warning: {w}" for w in self.warnings])

    def as_dict(self) -> dict:
        """Return the result for JSON."""
        return {
            "command": "value",
            "arms": list(self.arms),
            "n_fit": self.n_fit,
            "n_eval": self.n_eval,
            "dropped": self.dropped,
            "alpha": self.alpha,
            "resamples": self.resamples,
            "seed": self.seed,
            "cluster": self.cluster,
            "base": self.base.as_dict(),
            "rows": [r.as_dict() for r in self.rows],
            "warnings": list(self.warnings),
        }


def split(n: int, holdout: float, seed: int, clusters: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (fit rows, evaluation rows): a random share ``holdout`` of the decisions, or of the
    clusters, is held out. Clusters are kept whole so a shopper is never on both sides.

    Raises:
        ValueError: If a side would be empty.
    """
    rng = np.random.default_rng(seed)
    if clusters is None:
        perm = rng.permutation(n)
        n_eval = int(round(holdout * n))
        fit, held = perm[n_eval:], perm[:n_eval]
    else:
        groups = rng.permutation(int(clusters.max()) + 1)
        held_groups = groups[: int(round(holdout * len(groups)))]
        held_mask = np.isin(clusters, held_groups)
        fit, held = np.flatnonzero(~held_mask), np.flatnonzero(held_mask)
    if len(fit) == 0 or len(held) == 0:
        raise ValueError(f"a holdout of {holdout} on {n} decisions leaves an empty side")
    return fit, held


def value(
    log: Log,
    judgments: Sequence[str] | None = None,
    joint: bool = False,
    holdout: float = 0.5,
    resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    cluster: str | None = None,
    lam: float = 1.0,
    min_arm_rows: int = MIN_ARM_ROWS,
) -> ValueResult:
    """Value the context-only policy, the context plus each judgment and, with ``joint``, the
    context plus every judgment, on held-out decisions.

    Raises:
        LogError: If a judgment is not in the log or an arm has fewer than ``min_arm_rows`` rows
            on the fit side.
        ValueError: If nothing is asked for, ``holdout`` is not in (0, 1) or a side is empty.
    """
    if not 0 < holdout < 1:
        raise ValueError(f"holdout must lie strictly between 0 and 1, got {holdout}")
    judgments = list(judgments or [])
    if not judgments and not joint:
        raise ValueError("value needs judgments, joint, or both")
    for j in judgments:
        log.judgment(j)  # raises LogError when absent
    in_use = list(log.judgment_columns) if joint else judgments
    complete = log.complete(in_use)
    dropped = int((~complete).sum())
    if dropped:
        log = log.subset(complete)
    frame = log.frame
    ctx = frame[list(log.ctx_columns)].to_numpy(dtype=float).reshape(log.n, -1)
    sets = [(BASE_NAME, ctx)] + [(f"ctx + {j}", np.hstack([ctx, frame[[j]].to_numpy(dtype=float)])) for j in judgments]
    every = np.hstack([ctx, frame[in_use].to_numpy(dtype=float)])
    if joint:
        sets.append((f"ctx + all {len(in_use)} judgments", every))

    clusters = log.clusters(cluster)
    fit, held = split(log.n, holdout, seed, clusters)
    arm_of = {a: i for i, a in enumerate(log.arms)}
    arm = np.array([arm_of[a] for a in frame["action"]])
    reward = frame["revenue"].to_numpy(dtype=float) - frame["cost"].to_numpy(dtype=float)
    propensity = frame["propensity"].to_numpy(dtype=float)
    counts = np.bincount(arm[fit], minlength=len(log.arms))
    for a, c in zip(log.arms, counts, strict=True):
        if c < min_arm_rows:
            raise LogError(f"arm {a!r} has {int(c)} rows on the fit side; a per-arm model needs {min_arm_rows}")

    def predictions(x: np.ndarray) -> np.ndarray:
        """Fit one ridge per arm on the fit side and predict every arm on the held-out side."""
        return np.column_stack(
            [
                ridge(x[fit][arm[fit] == a], reward[fit][arm[fit] == a], lam).predict(x[held])
                for a in range(len(log.arms))
            ]
        )

    # One outcome model for every policy's DR estimate, so DR differences compare policies, not models.
    q = predictions(every)
    logged, w, r = arm[held], 1.0 / propensity[held], reward[held]
    rows = np.arange(len(held))
    choices = np.stack([np.argmax(predictions(x), axis=1) for _, x in sets])
    match = (choices == logged).astype(float)
    ipw_num, ipw_den = match * w * r, match * w
    dr_terms = q[rows[None, :], choices] + match * w * (r - q[rows, logged])

    def estimates(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return each policy's (IPW, DR) value on the held-out decisions ``sample`` indexes."""
        den = ipw_den[:, sample].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ipw = np.where(den > 0, ipw_num[:, sample].sum(axis=1) / den, np.nan)
        return ipw, dr_terms[:, sample].mean(axis=1)

    def statistic(sample: np.ndarray) -> np.ndarray:
        ipw, dr = estimates(sample)
        return np.concatenate([ipw[1:] - ipw[0], dr[1:] - dr[0]])

    ipw, dr = estimates(rows)
    point = statistic(rows)
    held_clusters = None if clusters is None else np.unique(clusters[held], return_inverse=True)[1]
    boot = bootstrap(len(held), statistic, resamples, seed, alpha, held_clusters)
    m = len(sets) - 1

    def policy(k: int) -> PolicyValue:
        shares = np.bincount(choices[k], minlength=len(log.arms)) / len(held)
        return PolicyValue(
            sets[k][0],
            int(match[k].sum()),
            None if np.isnan(ipw[k]) else float(ipw[k]),
            float(dr[k]),
            None if k == 0 else interval(point[k - 1], boot, k - 1),
            None if k == 0 else interval(point[m + k - 1], boot, m + k - 1),
            {a: float(s) for a, s in zip(log.arms, shares, strict=True)},
        )

    base, others = policy(0), [policy(k) for k in range(1, len(sets))]
    warnings = [
        x
        for x in (floor_warning(propensity), counterfactual_warning(frame["action"].to_numpy(), propensity, log.arms))
        if x
    ]
    for p in (base, *others):
        if p.matched < MIN_MATCHED:
            warnings.append(
                f"{p.name}: only {p.matched} held-out decisions match the policy; its IPW value rests on those"
            )
        if p.dr_diff is not None and p.dr_diff.hi is not None and p.dr_diff.hi < 0:
            warnings.append(
                f"{p.name} is worth less than the context-only policy: its columns fit noise at this "
                "size, which costs a little; it is not a finding of value"
            )
    return ValueResult(log.arms, len(fit), len(held), dropped, base, others, alpha, resamples, seed, cluster, warnings)
