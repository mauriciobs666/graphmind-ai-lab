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


# --- `start`: a full campaign's round 1 ----------------------------------------------------------


def _full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Campaign:
    from test_topup import full_campaign
    return full_campaign(tmp_path, monkeypatch, "guard-judge")


def test_start_draws_round_one_and_queues_its_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture) -> None:
    from decisionmodels import seeds
    camp = _full(tmp_path, monkeypatch)
    assert driver.main(["start", "--root", str(camp.root)]) == 0
    count = seeds.round_one_count("guard-judge")
    state = Campaign(camp.root).state()
    assert [r["genRound"] for r in state.rounds] == [1] and len(state.rounds[0]["seeds"]) == count
    pending = [b for b in state.batches.values() if b.kind == "generate" and b.state == "pending"]
    assert sum(len(b.entries) for b in pending) == count
    assert capsys.readouterr().out.strip() == (
        f"round 1: {count} seeds drawn, {len(pending)} generation runs queued")


def test_start_then_next_serves_the_first_run(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    camp = _full(tmp_path, monkeypatch)
    assert driver.main(["start", "--root", str(camp.root)]) == 0
    served = Campaign(camp.root, claude_dir=camp.claude_dir).next()
    assert served.kind == "generate"


@pytest.mark.parametrize("phase", ["smoke", "pilot"])
def test_start_refuses_a_phase_that_is_not_full(tmp_path: Path, phase: str,
                                                capsys: pytest.CaptureFixture) -> None:
    root = tmp_path / "camp"
    args = ["init", "--point", "guard-judge", "--phase", phase, "--root", str(root)]
    if phase == "smoke":
        args += ["--fixture", str(fixture(tmp_path))]
    assert driver.main(args) == 0
    ledger = Campaign(root).ledger_path
    before = ledger.read_bytes()
    capsys.readouterr()
    assert driver.main(["start", "--root", str(root)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("refused: ") and "full campaign" in err and "Traceback" not in err
    assert ledger.read_bytes() == before


def test_start_refuses_without_an_install_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture) -> None:
    camp = _full(tmp_path, monkeypatch)
    ledger = camp.ledger_path
    lines = [line for line in ledger.read_text("utf-8").splitlines()
             if json.loads(line)["event"] != "expectations"]
    ledger.write_text("\n".join(lines) + "\n", "utf-8")
    before = ledger.read_bytes()
    assert driver.main(["start", "--root", str(camp.root)]) == 1
    assert "driver install" in capsys.readouterr().err
    assert ledger.read_bytes() == before


def test_start_a_second_time_refuses_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture) -> None:
    camp = _full(tmp_path, monkeypatch)
    assert driver.main(["start", "--root", str(camp.root)]) == 0
    capsys.readouterr()
    before = camp.ledger_path.read_bytes()
    assert driver.main(["start", "--root", str(camp.root)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("refused: round 1 is already drawn") and "Traceback" not in err
    assert camp.ledger_path.read_bytes() == before


def test_start_retried_after_a_crash_between_draw_and_queue_defers_to_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    camp = _full(tmp_path, monkeypatch)
    monkeypatch.setattr(Campaign, "_queue_runs", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        driver.main(["start", "--root", str(camp.root)])
    monkeypatch.undo()
    state = camp.state()
    assert state.rounds and not state.batches
    capsys.readouterr()
    assert driver.main(["start", "--root", str(camp.root)]) == 1
    assert "not queued yet" in capsys.readouterr().err
    served = Campaign(camp.root, claude_dir=camp.claude_dir).next()      # `run`'s first step
    assert served.kind == "generate" and camp.state().batches


def test_start_refuses_while_a_runner_holds_the_lock(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture) -> None:
    camp = _full(tmp_path, monkeypatch)
    before = camp.ledger_path.read_bytes()
    with camp.locked():
        assert driver.main(["start", "--root", str(camp.root)]) == 1
    assert "locked" in capsys.readouterr().err and camp.ledger_path.read_bytes() == before
