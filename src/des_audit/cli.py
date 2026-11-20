"""The ``des-audit`` command. Exit codes: 0 done, 1 the log or a request was refused (the reason
is on stderr), 2 usage error."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from des_audit.calibration import calibration
from des_audit.log import LogError, load_log
from des_audit.power import effect_difference, mean_difference
from des_audit.value import value


def _positive_int(text: str) -> int:
    """argparse type: an integer above zero."""
    x = int(text)
    if x <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {x}")
    return x


def _unit(text: str) -> float:
    """argparse type: a number strictly between 0 and 1."""
    x = float(text)
    if not 0 < x < 1:
        raise argparse.ArgumentTypeError(f"must lie strictly between 0 and 1, got {x}")
    return x


def _positive(text: str) -> float:
    """argparse type: a number above zero."""
    x = float(text)
    if x <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {x}")
    return x


def _names(text: str) -> list[str]:
    """argparse type: comma-separated column names."""
    names = [t.strip() for t in text.split(",") if t.strip()]
    if not names:
        raise argparse.ArgumentTypeError("needs at least one name")
    return names


def _log_args(parser: argparse.ArgumentParser) -> None:
    """Add the arguments every log-reading command shares."""
    parser.add_argument("--log", required=True, help="CSV or Parquet file in the des-audit schema")
    parser.add_argument("--control", default="no_offer", help="the control arm (default no_offer)")
    parser.add_argument("--resamples", type=_positive_int, default=2000, help="bootstrap resamples (default 2000)")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap and split seed (default 0)")
    parser.add_argument("--alpha", type=_unit, default=0.05, help="intervals are at level 1 − alpha (default 0.05)")
    parser.add_argument("--cluster", help="resample this column's groups, for example shopper_id, not single rows")
    parser.add_argument("--json", action="store_true", help="print one JSON object instead of the table")


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="des-audit", description="Audit judgment columns in a randomized decision log."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibration", help="the offer's effect by bin of a judgment's score")
    _log_args(cal)
    cal.add_argument("--judgment", required=True, help="judgment column, for example j_shipping_cost_objection")
    cal.add_argument("--treatment", help="treatment arm; default the most logged non-control arm")
    cal.add_argument("--bins", type=_positive_int, default=2, help="equal-width score bins (default 2: cut at 0.5)")
    cal.add_argument("--power", type=_unit, default=0.8, help="power for the detectable difference (default 0.8)")

    val = sub.add_parser("value", help="what deciding with the judgments is worth, off-policy")
    _log_args(val)
    val.add_argument("--judgments", type=_names, help="comma-separated judgment columns, valued one at a time")
    val.add_argument("--all", action="store_true", dest="joint", help="also value every judgment column together")
    val.add_argument("--holdout", type=_unit, default=0.5, help="share of decisions held out (default 0.5)")

    pw = sub.add_parser("power", help="decisions a test needs")
    pw.add_argument("--delta", type=_positive, required=True, help="the difference to detect")
    noise = pw.add_mutually_exclusive_group(required=True)
    noise.add_argument("--sigma", type=_positive, help="reward noise per decision: a difference in mean reward")
    noise.add_argument("--rate", type=_unit, help="purchase rate: a difference in effect between high and low bins")
    pw.add_argument("--arms", type=_positive_int, default=2, help="arms sharing the traffic, with --sigma (default 2)")
    pw.add_argument("--high-share", type=_unit, default=0.5, help="share of decisions in the high bin, with --rate")
    pw.add_argument("--treated-share", type=_unit, default=0.5, help="share treated, with --rate (default 0.5)")
    pw.add_argument("--alpha", type=_unit, default=0.05, help="two-sided level (default 0.05)")
    pw.add_argument("--power", type=_unit, default=0.8, help="power (default 0.8)")
    pw.add_argument("--decisions-per-month", type=_positive, help="your volume, to print the months")
    pw.add_argument("--json", action="store_true", help="print one JSON object instead")
    return parser


def _emit(result, as_json: bool) -> None:
    """Print the JSON object, or the table with its warnings on stderr."""
    if as_json:
        print(json.dumps(result.as_dict(), indent=2))
        return
    lines = result.table().splitlines()
    print("\n".join(ln for ln in lines if not ln.startswith("warning: ")))
    for ln in lines:
        if ln.startswith("warning: "):
            print(ln, file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command and return the exit code."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "value" and not (args.judgments or args.joint):
            parser.error("value needs --judgments, --all, or both")
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2
    try:
        if args.command == "power":
            if args.sigma is not None:
                result = mean_difference(
                    args.delta, args.sigma, args.arms, args.alpha, args.power, args.decisions_per_month
                )
            else:
                result = effect_difference(
                    args.delta,
                    args.rate,
                    args.high_share,
                    args.treated_share,
                    args.alpha,
                    args.power,
                    args.decisions_per_month,
                )
        else:
            log = load_log(args.log, args.control)
            if args.command == "calibration":
                result = calibration(
                    log,
                    args.judgment,
                    args.treatment,
                    args.bins,
                    args.resamples,
                    args.seed,
                    args.alpha,
                    cluster=args.cluster,
                    power=args.power,
                )
            else:
                result = value(
                    log, args.judgments, args.joint, args.holdout, args.resamples, args.seed, args.alpha, args.cluster
                )
        _emit(result, args.json)
        return 0
    except (LogError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
