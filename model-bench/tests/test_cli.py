"""The three commands S1 ships: `compare`, `index rebuild`, `models --tested` (plan §3.6a).

`attest`, `validate` and `run` are S2's and must not exist yet — asserted, so the stage boundary is
a test rather than a promise. Exit codes are §3.6a's closed set.
"""

from __future__ import annotations

import json

import pytest
from conftest import classification_aggregates, item, model_fields, run

from modelbench.cli import main
from modelbench.results import store

PACK = "guard-judge-understanding"

MANIFEST = {
    "packId": PACK,
    "packVersion": "1.0.0",
    "role": "guard-judge",
    "schemaVersion": 1,
    "sampling": {
        "scripts": 40,
        "replicatesPerScript": 1,
        "seed": 20260902,
        "pairingKey": ["itemId"],
        "analysisUnit": "itemId",
    },
    "metrics": {
        "verdictMetrics": ["falseAdvanceRate"],
        "headlineMetric": "falseAdvanceRate",
    },
}


@pytest.fixture()
def workspace(tmp_path):
    pack_dir = tmp_path / "packs" / PACK
    pack_dir.mkdir(parents=True)
    (pack_dir / "pack.json").write_text(json.dumps(MANIFEST))
    return tmp_path


def _store_arm(root, name: str, correct: int, total: int = 40) -> None:
    items = [
        item(f"g{i:02d}", correct=i < correct, metric="falseAdvanceRate") for i in range(total)
    ]
    store(
        run(
            name,
            items=items,
            aggregates=classification_aggregates(correct, total),
            fingerprint_fields=model_fields(modelKey=name, packId=PACK),
        ),
        root,
    )


def test_compare_writes_a_report_and_exits_zero(workspace, capsys) -> None:
    _store_arm(workspace, "cand", 40)
    _store_arm(workspace, "incumbent", 34)
    code = main(["compare", "--pack", PACK, "--root", str(workspace)])
    out = capsys.readouterr().out
    assert code == 0
    assert "falseAdvanceRate" in out
    reports = list((workspace / "reports").glob("*.md"))
    assert len(reports) == 1
    assert reports[0].name.startswith(f"{PACK}-")
    assert reports[0].name.endswith("-01.md")


def test_a_same_day_rerun_does_not_overwrite_the_earlier_comparison(workspace) -> None:
    """Plan §3.5 — 'the one behaviour a tool built around durable history must not have'."""
    _store_arm(workspace, "cand", 40)
    _store_arm(workspace, "incumbent", 34)
    main(["compare", "--pack", PACK, "--root", str(workspace)])
    main(["compare", "--pack", PACK, "--root", str(workspace)])
    names = sorted(p.name for p in (workspace / "reports").glob("*.md"))
    assert len(names) == 2
    assert names[0].endswith("-01.md") and names[1].endswith("-02.md")


def test_compare_exits_zero_even_when_every_record_is_invalid(workspace, capsys) -> None:
    """§3.6a — 'that is a report, not an operational failure'."""
    _store_arm(workspace, "cand", 40)
    path = workspace / "results" / "runs" / "cand.json"
    raw = json.loads(path.read_text())
    raw["fingerprint"]["kvCacheSetting"] = ""
    path.write_text(json.dumps(raw))

    code = main(["compare", "--pack", PACK, "--root", str(workspace)])
    assert code == 0
    assert "INVALID RESULTS EXCLUDED" in capsys.readouterr().out


def test_negative_control_reports_not_distinguishable(workspace, capsys) -> None:
    """S1 done-condition 9 — a **smoke check**, and it says so in its own docstring.

    With two copies of one stored record `b = c = 0` by construction, so this **cannot fail**: it
    proves the mode is wired, not that the harness is sound. The real negative control is two
    *independent* runs of the same model and is §5 test 19a, an acceptance step (`-ml` §9).
    """
    _store_arm(workspace, "cand", 34)
    code = main(["compare", "--pack", PACK, "--negative-control", "--root", str(workspace)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Not distinguishable at this sample size." in out
    assert "b=0, c=0" in out


def test_compare_with_an_out_path(workspace, tmp_path) -> None:
    _store_arm(workspace, "cand", 40)
    _store_arm(workspace, "incumbent", 34)
    target = tmp_path / "custom.md"
    assert main(["compare", "--pack", PACK, "--out", str(target), "--root", str(workspace)]) == 0
    assert "falseAdvanceRate" in target.read_text()


def test_compare_selects_the_named_models(workspace, capsys) -> None:
    _store_arm(workspace, "cand", 40)
    _store_arm(workspace, "incumbent", 34)
    _store_arm(workspace, "third", 20)
    code = main(
        ["compare", "--pack", PACK, "--models", "third,incumbent", "--root", str(workspace)]
    )
    assert code == 0
    assert "third" in capsys.readouterr().out


def test_an_unknown_pack_exits_four(workspace, capsys) -> None:
    """§3.6a's closed exit set: 4 = invalid pack."""
    assert main(["compare", "--pack", "nope", "--root", str(workspace)]) == 4
    assert "nope" in capsys.readouterr().err


def test_a_manifest_without_a_headline_key_exits_four(workspace, capsys) -> None:
    manifest = json.loads((workspace / "packs" / PACK / "pack.json").read_text())
    del manifest["metrics"]["headlineMetric"]
    (workspace / "packs" / PACK / "pack.json").write_text(json.dumps(manifest))
    _store_arm(workspace, "cand", 40)
    assert main(["compare", "--pack", PACK, "--root", str(workspace)]) == 4


def test_bad_arguments_exit_two(capsys) -> None:
    assert main(["compare"]) == 2
    assert main(["nosuchcommand"]) == 2


def test_index_rebuild(workspace, capsys) -> None:
    _store_arm(workspace, "cand", 40)
    assert main(["index", "rebuild", "--root", str(workspace)]) == 0
    text = (workspace / "results" / "index.csv").read_text()
    assert "cand" in text
    assert "runId,date,role" in text
    assert str(workspace / "results" / "index.csv") in capsys.readouterr().out


def test_models_tested_lists_stored_models(workspace, capsys) -> None:
    _store_arm(workspace, "cand", 40)
    assert main(["models", "--tested", "--root", str(workspace)]) == 0
    assert "cand" in capsys.readouterr().out


def test_models_tested_filters_by_pack(workspace, capsys) -> None:
    _store_arm(workspace, "cand", 40)
    assert main(["models", "--tested", "--pack", "other", "--root", str(workspace)]) == 0
    assert "cand" not in capsys.readouterr().out


def test_s2_commands_are_not_shipped_yet(capsys) -> None:
    """Plan §3.6a assigns `attest`, `validate` and `run` to S2. The boundary is a test."""
    for command in ("attest", "validate", "run"):
        assert main([command]) == 2


def test_module_entrypoint_exists() -> None:
    import modelbench.__main__ as entry

    assert hasattr(entry, "main")
