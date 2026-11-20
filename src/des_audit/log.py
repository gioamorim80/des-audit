"""The decision log: one CSV or Parquet file in the documented schema, checked before any estimate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("decision_id", "action", "propensity", "outcome", "revenue", "cost")
CTX_PREFIX = "ctx_"
JUDGMENT_PREFIX = "j_"


class LogError(ValueError):
    """The log cannot be used as given; the message names the problem."""


@dataclass(frozen=True)
class Log:
    """A checked decision log.

    Attributes:
        frame: The rows, with every column of the file.
        ctx_columns: Names of the context columns, in file order.
        judgment_columns: Names of the judgment columns, in file order. A judgment may be missing
            (NaN) on some rows; a command drops the rows missing a judgment it uses.
        arms: Distinct action names, sorted.
        control: The control arm's name.
    """

    frame: pd.DataFrame
    ctx_columns: tuple[str, ...]
    judgment_columns: tuple[str, ...]
    arms: tuple[str, ...]
    control: str

    @property
    def n(self) -> int:
        """Number of decisions."""
        return len(self.frame)

    def judgment(self, name: str) -> np.ndarray:
        """Return one judgment column as floats, NaN where it is missing.

        Raises:
            LogError: If the column is not a judgment column of the log.
        """
        if name not in self.judgment_columns:
            raise LogError(f"judgment column {name!r} is not in the log; it has {list(self.judgment_columns)}")
        return self.frame[name].to_numpy(dtype=float)

    def complete(self, judgments: Sequence[str]) -> np.ndarray:
        """Return a mask of the rows where every named judgment is present."""
        mask = np.ones(self.n, dtype=bool)
        for name in judgments:
            mask &= ~np.isnan(self.judgment(name))
        return mask

    def missing_by_arm(self, judgments: Sequence[str]) -> dict[str, float]:
        """Return, per arm, the share of its rows missing any of the named judgments."""
        missing = ~self.complete(judgments)
        action = self.frame["action"].to_numpy()
        return {a: float(missing[action == a].mean()) for a in self.arms}

    def subset(self, rows: np.ndarray) -> Log:
        """Return the log restricted to a boolean mask of rows.

        Raises:
            LogError: If the rows left hold no control decision or a single arm.
        """
        frame = self.frame[rows].reset_index(drop=True)
        arms = tuple(sorted(str(a) for a in frame["action"].unique()))
        if self.control not in arms or len(arms) < 2:
            raise LogError(f"after dropping rows with a missing judgment the log has arms {list(arms)}")
        return Log(frame, self.ctx_columns, self.judgment_columns, arms, self.control)

    def clusters(self, column: str | None) -> np.ndarray | None:
        """Return the cluster of each row as integer codes 0..G-1, or None without a column.

        Raises:
            LogError: If the column is not in the log or has missing values.
        """
        if column is None:
            return None
        if column not in self.frame.columns:
            raise LogError(f"cluster column {column!r} is not in the log")
        if self.frame[column].isna().any():
            raise LogError(f"cluster column {column!r} has missing values")
        codes, _ = pd.factorize(self.frame[column])
        return codes.astype(np.int64)

    def treatment_or_default(self, treatment: str | None) -> str:
        """Return ``treatment`` after checking it is an arm, or the most logged non-control arm.

        Raises:
            LogError: If ``treatment`` is not an arm or is the control.
        """
        if treatment is None:
            counts = self.frame["action"].value_counts()
            return [str(a) for a in counts.index if a != self.control][0]
        if treatment not in self.arms:
            raise LogError(f"treatment arm {treatment!r} is not in the log; arms are {list(self.arms)}")
        if treatment == self.control:
            raise LogError(f"treatment arm {treatment!r} is the control arm")
        return treatment


def _read(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet file.

    Raises:
        LogError: If the suffix is not supported or Parquet support is not installed.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError as e:
            raise LogError(f"{path} is empty") from e
    if suffix == ".parquet":
        try:
            import pyarrow  # noqa: F401  (checked here so the error names the extra to install)
        except ImportError as e:
            raise LogError("reading Parquet needs pyarrow: pip install 'des-audit[parquet]'") from e
        return pd.read_parquet(path)
    raise LogError(f"{path} must be a .csv or .parquet file")


def validate(frame: pd.DataFrame, control: str = "no_offer") -> Log:
    """Check a frame already in memory and return the ``Log``.

    Raises:
        LogError: On any schema or value problem; the message names it.
    """
    if len(frame) == 0:
        raise LogError("the log is empty")
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise LogError(f"missing required columns: {missing}")
    ctx = tuple(str(c) for c in frame.columns if str(c).startswith(CTX_PREFIX))
    judgments = tuple(str(c) for c in frame.columns if str(c).startswith(JUDGMENT_PREFIX))
    if not judgments:
        raise LogError("the log has no judgment column (prefix j_)")
    nan_columns = [c for c in (*REQUIRED, *ctx) if frame[c].isna().any()]
    if nan_columns:
        raise LogError(f"missing values in columns {nan_columns}; only judgment columns may have them")
    for c in ("propensity", "outcome", "revenue", "cost", *ctx, *judgments):
        if not pd.api.types.is_numeric_dtype(frame[c]):
            raise LogError(f"column {c!r} must be numeric")
    propensity = frame["propensity"].to_numpy(dtype=float)
    bad = (propensity <= 0) | (propensity > 1)
    if bad.any():
        raise LogError(
            f"propensity must lie in (0, 1]; {int(bad.sum())} values do not, for example {propensity[bad][0]:g}"
        )
    outcome = frame["outcome"].to_numpy(dtype=float)
    if not np.isin(outcome, (0.0, 1.0)).all():
        raise LogError("outcome must be 0 or 1")
    for c in judgments:
        values = frame[c].to_numpy(dtype=float)
        present = values[~np.isnan(values)]
        if (present < 0).any() or (present > 1).any():
            raise LogError(f"judgment column {c!r} must lie in [0, 1]")
    if frame["decision_id"].duplicated().any():
        raise LogError(f"decision_id repeats ({int(frame['decision_id'].duplicated().sum())} duplicate rows)")
    actions = frame["action"].astype(str).to_numpy()
    arms = tuple(sorted(str(a) for a in np.unique(actions)))
    if len(arms) < 2:
        raise LogError(f"the log has a single arm {arms!r}; an audit needs at least two")
    if control not in arms:
        raise LogError(f"control arm {control!r} has no rows; arms are {list(arms)}")
    clean = frame.copy()
    clean["action"] = actions
    clean["outcome"] = outcome
    clean["propensity"] = propensity
    return Log(clean.reset_index(drop=True), ctx, judgments, arms, control)


def load_log(path: str | Path, control: str = "no_offer") -> Log:
    """Read and check a log file.

    Raises:
        LogError: If the file cannot be read or fails the checks.
    """
    path = Path(path)
    if not path.exists():
        raise LogError(f"{path} does not exist")
    return validate(_read(path), control)
