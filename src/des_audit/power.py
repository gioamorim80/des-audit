"""``power``: how many decisions a test needs before it is run.

Two questions, one formula. With an estimate whose standard error is c / √n, a two-sided test at
level alpha detects a true difference Δ with probability ``power`` when

    n = c² (z(1 − alpha/2) + z(power))² / Δ².

For a difference in mean reward between two arms that each get n decisions, c² = 2σ². For a
difference in an offer's effect on purchase between shoppers whose judgment is high and low, with
a share h of decisions in the high bin, a share q of each bin treated and purchase rate p,
c² = p(1 − p) / (q(1 − q)) × (1/h + 1/(1 − h)), and n counts every decision of the test.

Worked example: purchase rate 0.17, a quarter of chats in the high bin, half treated, a
difference of 2 points. c² = 0.1411 / 0.25 × (4 + 1.333) = 3.01, (1.960 + 0.842)² = 7.85, so
n = 3.01 × 7.85 / 0.0004 = 59,100 chats.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True)
class PowerResult:
    """Decisions needed to detect a difference.

    Attributes:
        question: ``"mean difference"`` or ``"effect difference"``.
        delta: The difference to detect.
        alpha: Two-sided level.
        power: Probability of detecting it.
        per_arm: Decisions per arm (mean difference), or None.
        total: Decisions in the whole test.
        months: Months at ``decisions_per_month``, or None without it.
        inputs: The inputs that set the noise.
    """

    question: str
    delta: float
    alpha: float
    power: float
    per_arm: int | None
    total: int
    months: float | None
    inputs: dict[str, float]

    def table(self) -> str:
        """Return the readable output."""
        z = NormalDist().inv_cdf(1 - self.alpha / 2) + NormalDist().inv_cdf(self.power)
        if self.question == "mean difference":
            lines = [
                f"decisions per arm to detect a difference of {self.delta:g} in mean reward "
                f"(σ = {self.inputs['sigma']:g}): {self.per_arm:,}",
                f"decisions in total over {int(self.inputs['arms'])} arms: {self.total:,}",
            ]
        else:
            lines = [
                f"decisions to detect a difference of {self.delta:g} in the offer's effect on purchase between "
                f"high and low judgments (purchase rate {self.inputs['rate']:g}, {self.inputs['high_share']:g} of "
                f"decisions high, {self.inputs['treated_share']:g} treated): {self.total:,}"
            ]
        if self.months is not None:
            lines.append(f"months at {int(self.inputs['decisions_per_month']):,} decisions a month: {self.months:.1f}")
        lines.append(
            f"two-sided test at alpha {self.alpha:g}, power {self.power:g} (z sum {z:.3f}); halving the "
            "difference to detect quadruples the decisions"
        )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """Return the result for JSON."""
        return {
            "command": "power",
            "question": self.question,
            "delta": self.delta,
            "alpha": self.alpha,
            "power": self.power,
            "per_arm": self.per_arm,
            "total": self.total,
            "months": self.months,
            "inputs": dict(self.inputs),
            "warnings": [],
        }


def _z(alpha: float, power: float) -> float:
    """Return z(1 − alpha/2) + z(power).

    Raises:
        ValueError: If alpha or power is not in (0, 1).
    """
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError(f"alpha and power must lie in (0, 1), got {alpha} and {power}")
    return NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)


def _months(total: int, per_month: float | None) -> float | None:
    """Return the months the test takes at a monthly volume, or None without one."""
    if per_month is None:
        return None
    if per_month <= 0:
        raise ValueError(f"decisions per month must be positive, got {per_month}")
    return total / per_month


def mean_difference(
    delta: float,
    sigma: float,
    arms: int = 2,
    alpha: float = 0.05,
    power: float = 0.8,
    decisions_per_month: float | None = None,
) -> PowerResult:
    """Return the decisions needed to detect a difference ``delta`` in mean reward between two
    arms when a decision's reward has standard deviation ``sigma``: 2σ²(z sum)²/Δ² per arm.

    Raises:
        ValueError: If delta or sigma is not positive, arms < 2, or alpha or power is not in (0, 1).
    """
    if delta <= 0 or sigma <= 0:
        raise ValueError(f"delta and sigma must be positive, got {delta} and {sigma}")
    if arms < 2:
        raise ValueError(f"arms must be at least 2, got {arms}")
    per_arm = int(round(2 * sigma**2 * _z(alpha, power) ** 2 / delta**2))
    total = per_arm * arms
    return PowerResult(
        "mean difference",
        delta,
        alpha,
        power,
        per_arm,
        total,
        _months(total, decisions_per_month),
        {"sigma": sigma, "arms": arms, **({"decisions_per_month": decisions_per_month} if decisions_per_month else {})},
    )


def effect_difference(
    delta: float,
    rate: float,
    high_share: float,
    treated_share: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
    decisions_per_month: float | None = None,
) -> PowerResult:
    """Return the decisions needed to detect a difference ``delta`` in an offer's effect on
    purchase between the high and the low bin of a judgment.

    Raises:
        ValueError: If delta is not positive, or rate, high_share, treated_share, alpha or power
            is not in (0, 1).
    """
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    for name, x in (("rate", rate), ("high_share", high_share), ("treated_share", treated_share)):
        if not 0 < x < 1:
            raise ValueError(f"{name} must lie in (0, 1), got {x}")
    c2 = rate * (1 - rate) / (treated_share * (1 - treated_share)) * (1 / high_share + 1 / (1 - high_share))
    total = int(round(c2 * _z(alpha, power) ** 2 / delta**2))
    return PowerResult(
        "effect difference",
        delta,
        alpha,
        power,
        None,
        total,
        _months(total, decisions_per_month),
        {
            "rate": rate,
            "high_share": high_share,
            "treated_share": treated_share,
            **({"decisions_per_month": decisions_per_month} if decisions_per_month else {}),
        },
    )
