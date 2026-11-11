"""s28_run.py `--extract`: the projection script a run used is selectable and its sha256 is recorded (P1-34 step 1).

No Joern/Docker: `s27.run` (the subprocess boundary) is replaced by a recording stub that writes the content-check log the
script reads; the work tree is a tmp dir. Mutants run on COPIES via the S28_RUN_SRC env var (see test_s28_metrics.py).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

CPG = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("S28_RUN_SRC") or CPG / "spikes" / "frontend-fitness")
sys.path.insert(0, str(SRC))
import s28_run as R  # noqa: E402

OLD = CPG / "spikes" / "vertical-slice" / "extract.sc"
NEW = CPG / "extractors" / "joern" / "extract.sc"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def harness(tmp_path, monkeypatch):
    base = tmp_path / "joern" / "python"
    (base / "stage").mkdir(parents=True)
    for cfg in ("off", "on"):
        (base / cfg).mkdir()
    (base / "s27-run.json").write_text(json.dumps({"runs": {f"{c}.parse": {"cpgFile": f"/x/{c}.cpg"} for c in ("off", "on")}}))
    calls = []

    def fake_run(argv, log, label):
        calls.append((label, list(argv)))
        log.write_text("CONTENT=ok\n" if label.endswith("content-check") else "")
        return {"stage": label, "rc": 0}

    monkeypatch.setattr(R, "WORK", tmp_path / "joern")
    monkeypatch.setattr(R.s27, "run", fake_run)
    monkeypatch.setattr(R.s27, "newest", lambda *a, **k: None)

    def go(*args):
        calls.clear()
        monkeypatch.setattr(sys, "argv", ["s28_run.py", "python", *args])
        assert R.main() == 0
        rec = json.loads((base / "s28-run.json").read_text())["runs"]
        return rec, calls

    return go


def script_of(calls, cfg):
    argv = dict(calls)[f"{cfg}.extract"]
    return argv[argv.index("--script-file") + 1]


def test_default_uses_old_projection_script(harness):
    rec, calls = harness()
    for cfg in ("off", "on"):
        assert rec[f"{cfg}.extract"]["extractSha256"] == sha(OLD)
        assert Path(script_of(calls, cfg)).resolve() == OLD.resolve()
    assert sha(OLD) != sha(NEW)


@pytest.mark.parametrize("arg", ["cpg/extractors/joern/extract.sc", str(NEW)])
def test_extract_arg_uses_p1_09_script(harness, arg):
    rec, calls = harness("--extract", arg)
    for cfg in ("off", "on"):
        assert rec[f"{cfg}.extract"]["extractSha256"] == sha(NEW)
        assert Path(script_of(calls, cfg)).resolve() == NEW.resolve()
        argv = dict(calls)[f"{cfg}.extract"]
        assert "/scripts/extract.sc" in argv


def test_guard_records_guard_script_hash(harness):
    rec, _ = harness("--guard")
    assert rec["off.extract-guard"]["extractSha256"] == sha(R.GUARD)
