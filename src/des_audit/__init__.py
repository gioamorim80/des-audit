"""des-audit: does a judgment model's reading of a chat tell you whom an offer moves?

Run on a store's own randomized decision log: ``calibration`` gives the offer's effect by bin of a
judgment's score and the difference between bins, ``value`` what deciding with the judgments is
worth off-policy, and ``power`` how many decisions a test needs.
"""

from des_audit.calibration import CalibrationResult, calibration
from des_audit.log import Log, LogError, load_log, validate
from des_audit.power import PowerResult, effect_difference, mean_difference
from des_audit.value import ValueResult, value

__all__ = [
    "CalibrationResult",
    "Log",
    "LogError",
    "PowerResult",
    "ValueResult",
    "calibration",
    "effect_difference",
    "load_log",
    "mean_difference",
    "validate",
    "value",
]
