"""The driver CLI (typed-decision-models U5.6): `init`, `resolve` and `status` over
`decisionmodels.campaign` (plan4 §3.5, plan6 §3.2 "Subcommands"). The behaviour is
`tests/test_campaign.py`'s; these pin the command line: arguments, exit codes and output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from decisionmodels import driver
from decisionmodels.campaign import Campaign

COMPONENT = Path(__file__).resolve().parents[1]


def fixture(tmp_path: Path) -> Path:
    path = tmp_path / "items.jsonl"
    item = {"condition": "the user has stated a budget", "path": "turns", "understanding": {},
            "turns": [{"role": "user", "displayName": "Ana", "text": "Orçamento de R$ 100"}]}
    path.write_text(json.dumps(item, ensure_ascii=False) + "\n", "utf-8")
    return path


def test_init_records_the_durations_given(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = tmp_path / "camp"
    code = driver.main(["init", "--point", "guard-judge", "--phase", "smoke", "--root", str(root),
                        "--fixture", str(fixture(tmp_path)), "--batch-timeout", "30m",
                        "--launch-fail-bound", "8h"])
    assert code == 0 and "initialised guard-judge smoke" in capsys.readouterr().out
    init = Campaign(root).events()[0]
    assert (init["batchTimeout"], init["launchFailBound"]) == (1800, 28800)


def test_a_refusal_exits_1_and_writes_nothing(tmp_path: Path,
                                              capsys: pytest.CaptureFixture) -> None:
    root = tmp_path / "camp"
    code = driver.main(["init", "--point", "guard-judge", "--phase", "smoke", "--root",
                        str(root)])
    assert code == 1 and "--fixture" in capsys.readouterr().err
    assert not root.exists()


def test_a_bad_duration_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        driver.main(["init", "--point", "guard-judge", "--phase", "pilot", "--root",
                     str(tmp_path / "c"), "--launch-fail-bound", "0h"])
    assert raised.value.code == 2


def test_resolve_actions_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        driver.main(["resolve", "--root", str(tmp_path), "--decision", "x", "--skip-batch",
                     "--repin-instructions"])
    assert raised.value.code == 2


def test_resolve_on_a_campaign_that_is_not_halted_is_refused(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = tmp_path / "camp"
    driver.main(["init", "--point", "guard-judge", "--phase", "pilot", "--root", str(root)])
    before = (root / "store" / "ledger.jsonl").read_bytes()
    assert driver.main(["resolve", "--root", str(root), "--decision", "x"]) == 1
    assert "not halted" in capsys.readouterr().err
    assert (root / "store" / "ledger.jsonl").read_bytes() == before


def test_status_prints_the_campaign(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = tmp_path / "camp"
    driver.main(["init", "--point", "guard-judge", "--phase", "smoke", "--root", str(root),
                 "--fixture", str(fixture(tmp_path))])
    capsys.readouterr()
    (root / "store" / "pause").touch()
    assert driver.main(["status", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "point guard-judge, phase smoke" in out and "1 pending" in out and "PAUSED" in out


def test_status_without_a_campaign_is_refused(tmp_path: Path) -> None:
    assert driver.main(["status", "--root", str(tmp_path)]) == 1


def test_next_and_ingest_are_not_subcommands(tmp_path: Path) -> None:
    """plan6 §3.2: they are library entry points of `driver run`, listed for no session."""
    for command in ("next", "ingest"):
        with pytest.raises(SystemExit):
            driver.main([command, "--root", str(tmp_path)])


def test_the_module_runs_as_a_script(tmp_path: Path) -> None:
    done = subprocess.run([sys.executable, "-m", "decisionmodels.driver", "status", "--root",
                           str(tmp_path)], cwd=COMPONENT, capture_output=True, text=True)
    assert done.returncode == 1 and done.stderr.startswith("refused: no campaign")
