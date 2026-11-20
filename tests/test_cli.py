"""The command line."""

import json

from des_audit.cli import main

from logs import two_arm_log


def test_calibration_prints_json(tmp_path, capsys):
    """A good log gives exit code 0 and one JSON object."""
    path = tmp_path / "log.csv"
    two_arm_log(2_000).to_csv(path, index=False)
    assert main(["calibration", "--log", str(path), "--judgment", "j_shipping", "--resamples", "50", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["command"] == "calibration" and len(out["bins"]) == 2


def test_a_refused_log_exits_1(tmp_path, capsys):
    """A missing file is refused with the reason on stderr."""
    assert main(["calibration", "--log", str(tmp_path / "none.csv"), "--judgment", "j_x"]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_usage_errors_exit_2(tmp_path):
    """value with nothing to value, or power with no noise given, is a usage error."""
    path = tmp_path / "log.csv"
    two_arm_log(100).to_csv(path, index=False)
    assert main(["value", "--log", str(path)]) == 2
    assert main(["power", "--delta", "1"]) == 2


def test_power_prints(capsys):
    """power needs no log."""
    assert main(["power", "--delta", "0.02", "--rate", "0.17", "--high-share", "0.25"]) == 0
    assert "59,065" in capsys.readouterr().out
