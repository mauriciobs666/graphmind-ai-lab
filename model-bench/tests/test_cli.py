"""The three commands S1 ships: `compare`, `index rebuild`, `models --tested` (plan §3.6a).

`attest`, `validate` and `run` are S2's and must not exist yet — asserted, so the stage boundary is
a test rather than a promise. Exit codes are §3.6a's closed set.
"""

from __future__ import annotations

import json

import pytest
from conftest import classification_aggregates, item, model_fields, run

from modelbench.cli import main
from modelbench.results import ItemResult, store

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


def _store_arm(root, name: str, correct: int, total: int = 40, session: str = "s1") -> None:
    items = [
        item(f"g{i:02d}", correct=i < correct, metric="falseAdvanceRate") for i in range(total)
    ]
    store(
        run(
            name,
            items=items,
            aggregates=classification_aggregates(correct, total),
            fingerprint_fields=model_fields(modelKey=name, packId=PACK),
            session_id=session,
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


def test_an_incomplete_item_record_does_not_take_the_comparison_down_with_it(
    workspace, capsys
) -> None:
    """Review P4-5 — one bad item aborted `compare` outside §3.6a's closed exit-code set.

    `results.py` raises `IncompleteItemRecord` for an item that declares a metric scoreable and
    records no count; `report.py` is its only caller, and `_cmd_compare` catches only
    `PackConfigError`. Verified end-to-end before the fix: one such item in **one** of three
    otherwise-valid stored records aborted with an uncaught traceback, **exit 1** — not one of
    `0/2/3/4/5` — **no report written at all**, and the two good arms lost with it.

    The comparison must survive: two valid arms still compare, the third is excluded and named in
    the block AC-2 already owns, the exit code stays `0` (the tool ran and reported), and the
    report is on disk.
    """
    _store_arm(workspace, "cand", correct=40)
    _store_arm(workspace, "incumbent", correct=34)
    broken = [item(f"g{i:02d}", correct=True, metric="falseAdvanceRate") for i in range(39)]
    broken.append(
        ItemResult(itemId="g39", pairingKey=("g39",), outcome="pass",
                   scoreable={"falseAdvanceRate": True}, counts={}, latencyMs=1300.0, detail={})
    )
    store(
        run("halfscored", items=broken, aggregates=classification_aggregates(40, 40),
            fingerprint_fields=model_fields(modelKey="halfscored", packId=PACK)),
        workspace,
    )

    code = main(["compare", "--root", str(workspace), "--pack", PACK])
    out = capsys.readouterr().out

    assert code == 0
    assert "**INVALID RESULTS EXCLUDED** (AC-2)" in out
    assert "`halfscored`" in out and "items[g39].counts.falseAdvanceRate" in out
    assert "is better than" in out              # the two valid arms still compared
    written = list((workspace / "reports").glob("*.md"))
    assert len(written) == 1 and "halfscored" in written[0].read_text()


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
    """Review M-6 — the shipped assertion was only `"third" in out`, which holds whether or not the
    filter runs, so deleting `--models` entirely was green. The arm that must *not* be there is
    what makes this a test; it is matched on its Arms-table row, because the word "candidate"
    appears in every report's best-case caveat."""
    _store_arm(workspace, "cand", 40)
    _store_arm(workspace, "incumbent", 34)
    _store_arm(workspace, "third", 20)
    code = main(
        ["compare", "--pack", PACK, "--models", "third,incumbent", "--root", str(workspace)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "| third |" in out and "| incumbent |" in out
    assert "| cand |" not in out


def test_compare_exits_two_naming_a_models_key_with_no_stored_run(workspace, capsys) -> None:
    """Review M-6 — `[by_key[m] for m in wanted if m in by_key]` dropped an unmatched key in
    silence, so `--models cand,incumbnet` rendered a **one-arm** report that then asserted a
    deterministic-arm reason that is untrue. A typo in a model key is a usage error (§3.6a's
    exit 2), not a comparison."""
    _store_arm(workspace, "cand", 40)
    _store_arm(workspace, "incumbent", 34)
    code = main(
        ["compare", "--pack", PACK, "--models", "cand,incumbnet", "--root", str(workspace)]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "incumbnet" in err
    assert not list((workspace / "reports").glob("*.md"))


def test_a_single_selected_arm_is_reported_as_such(workspace, capsys) -> None:
    """Review M-6's rendered half: one arm is not two deterministic arms."""
    _store_arm(workspace, "cand", 40)
    code = main(["compare", "--pack", PACK, "--models", "cand", "--root", str(workspace)])
    out = capsys.readouterr().out
    assert code == 0
    assert "fewer than two arms" in out
    assert "two deterministic arms" not in out


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


def test_the_negative_control_report_says_on_its_face_that_it_is_a_smoke_check(
    workspace, capsys
) -> None:
    """Review P3-4 (major) — the mode wrote a durable report indistinguishable from a real
    comparison, filed beside it under a filename differing only in its sequence number.

    `grep -ic negative` on the produced report returned **0**: what a reader got was an ordinary
    `b=0, c=0` "not distinguishable" verdict with both arms bearing the same label. `-ml` §9 and
    plan §3.9(5) are explicit that the real negative control is two *independent* runs and that two
    copies **cannot fail** — so a stored artifact reading as a validated null is the one output
    this tool's value claim cannot afford.
    """
    _store_arm(workspace, "cand", 34)
    assert main(["compare", "--pack", PACK, "--negative-control", "--root", str(workspace)]) == 0
    out = capsys.readouterr().out
    report = next((workspace / "reports").glob("*.md")).read_text(encoding="utf-8")

    assert "NEGATIVE CONTROL (WIRING SMOKE CHECK)" in report
    assert "b = c = 0 by construction" in report
    assert "cannot fail" in report
    # It is on stdout too — a reader who never opens the file still sees it.
    assert "NEGATIVE CONTROL (WIRING SMOKE CHECK)" in out


def test_a_negative_control_with_no_stored_runs_does_not_claim_it_cannot_fail(
    workspace, capsys
) -> None:
    """Review P4-1 — a durable report asserting something untrue of itself, P3-4's own failure mode
    re-entered through the case P3-4's fix did not cover.

    With no stored runs for the pack, `_select_arms`' `if negative_control and candidates` is false
    and it returns `[]`, but the banner was emitted before `_comparison_pair` was ever consulted.
    Verified end-to-end: **exit 0**, a report written to `reports/<packId>-<date>-01.md`, opening

        both arms are the *same stored record*, so `b = c = 0 by construction` and this comparison
        **cannot fail**

    and stating ten lines below, in the same document, *"None: fewer than two arms were selected,
    so there is nothing to compare."* No record was duplicated and no wiring was exercised, so the
    banner's subject does not exist.

    **The banner's substance is otherwise sound** and is not what changes here: with two copies of
    one record `b = c = 0`, so `distinguishable` is unreachable on either path. The defect is the
    zero-arm case emitting it at all.
    """
    code = main(["compare", "--root", str(workspace), "--pack", PACK, "--negative-control"])
    out = capsys.readouterr().out

    assert code == 0
    assert "cannot fail" not in out
    assert "both arms are the *same stored record*" not in out
    # ...and the durable artifact still says the mode was asked for and did not run
    assert "**NEGATIVE CONTROL REQUESTED, NOT RUN**" in out
    assert "fewer than two arms were selected" in out
    written = list((workspace / "reports").glob("*.md"))
    assert len(written) == 1 and "NOT RUN" in written[0].read_text()


def test_a_negative_control_with_a_stored_run_still_carries_the_full_banner(
    workspace, capsys
) -> None:
    """The other side of P4-1's gate: where the mode *did* duplicate a record, the banner that
    tells a reader this comparison cannot fail is mandatory (review P3-4)."""
    _store_arm(workspace, "cand", correct=40)
    assert main(["compare", "--root", str(workspace), "--pack", PACK, "--negative-control"]) == 0
    out = capsys.readouterr().out
    assert "**NEGATIVE CONTROL (WIRING SMOKE CHECK)**" in out
    assert "cannot fail" in out
    assert "NOT RUN" not in out


def test_the_negative_control_duplicates_the_first_arm_in_the_requested_order(
    workspace, capsys
) -> None:
    """P4-13 — *which* record the mode duplicates was undocumented and untested.

    `candidates[-1]` survived the suite, so the choice was neither pinned nor stated anywhere. It
    is the **first arm in the order the operator asked for** — the same order every other part of
    `compare` uses, so `--models X,Y` puts X in both arms exactly as it puts X in arm A of an
    ordinary comparison. Pinning it is what makes the mode's output predictable enough to be a
    smoke check at all.
    """
    _store_arm(workspace, "aaa", correct=40)
    _store_arm(workspace, "zzz", correct=20)

    assert main(["compare", "--root", str(workspace), "--pack", PACK, "--negative-control",
                 "--models", "zzz,aaa"]) == 0
    out = capsys.readouterr().out
    assert "| zzz | falseAdvanceRate | 20/40 |" in out
    assert "aaa" not in out.split("## Verdicts")[0]
    # ...and the two arms carry no P4-10 disambiguating suffix: they are one record, not two arms
    # that happen to share a model key, and the banner above already says so.
    assert "(run zzz)" not in out and "(session " not in out


def test_an_ordinary_comparison_carries_no_negative_control_banner(workspace, capsys) -> None:
    """The negative of P3-4: a banner that appears on every report says nothing."""
    _store_arm(workspace, "cand", 40)
    _store_arm(workspace, "incumbent", 34)
    assert main(["compare", "--pack", PACK, "--root", str(workspace)]) == 0
    assert "NEGATIVE CONTROL" not in capsys.readouterr().out


def test_compare_session_restricts_the_arm_set_to_that_session(workspace, capsys) -> None:
    """Review P3-8 — `--session` was entirely untested: deleting its filter left 314 passed, and
    `grep -n session tests/test_cli.py` returned nothing. It is one of `compare`'s four options and
    the one FR-16's same-session pairing rests on; Pass 1's m-4 closed the same gap for `--role`.

    Asserted as *which arms reached the report*, from both sides: the named session's two arms are
    there and the other session's is not.
    """
    _store_arm(workspace, "cand", 40, session="s1")
    _store_arm(workspace, "incumbent", 34, session="s1")
    _store_arm(workspace, "outlier", 20, session="s2")

    assert main(["compare", "--pack", PACK, "--session", "s1", "--root", str(workspace)]) == 0
    out = capsys.readouterr().out
    assert "| cand | falseAdvanceRate |" in out
    assert "| incumbent | falseAdvanceRate |" in out
    assert "outlier" not in out
    assert "paired, same session" in out


def test_compare_session_naming_a_session_with_one_arm_reports_too_few_arms(
    workspace, capsys
) -> None:
    """The filter's other direction, and the one an unfiltered `--session` cannot fake: a session
    holding a single run has nothing to compare, and the report says exactly that (review M-6)."""
    _store_arm(workspace, "cand", 40, session="s1")
    _store_arm(workspace, "incumbent", 34, session="s1")
    _store_arm(workspace, "outlier", 20, session="s2")

    assert main(["compare", "--pack", PACK, "--session", "s2", "--root", str(workspace)]) == 0
    out = capsys.readouterr().out
    assert "fewer than two arms were selected" in out
    assert "is better than" not in out


def test_the_report_filename_is_the_manifests_pack_id_not_the_directory_name(
    workspace, tmp_path
) -> None:
    """Review P3-14 — `_cmd_compare` called `_report_path(root, args.pack)` while the parameter is
    named `pack_id` and the docstring promises `reports/<pack-id>-<date>-<n>.md`. `args.pack` is
    the pack **directory**; the two coincide by the §3.3 `packs/<pack-id>/` convention and nothing
    enforces it. This is the other half of Pass 1's m-6, which fixed the `load_history` call and
    left the filename on the directory name.
    """
    other_dir = workspace / "packs" / "a-directory-with-another-name"
    other_dir.mkdir(parents=True)
    (other_dir / "pack.json").write_text(json.dumps(MANIFEST))
    _store_arm(workspace, "cand", 40)
    _store_arm(workspace, "incumbent", 34)

    assert main(
        ["compare", "--pack", "a-directory-with-another-name", "--root", str(workspace)]
    ) == 0
    names = [p.name for p in (workspace / "reports").glob("*.md")]
    assert len(names) == 1
    assert names[0].startswith(f"{PACK}-")
    assert not names[0].startswith("a-directory-with-another-name")

    # Review P4-9 — the **sibling** call, `load_history(root, packId=pack.packId)`, is the other
    # half of the same distinction and was the half left unpinned: re-pointing it at `args.pack`
    # filtered out every stored record (each declares `packId` = the manifest's), leaving a
    # zero-arm report that still passed every assertion above, since the filename half is
    # independent of it. The arms have to be shown to have loaded, not just the file named.
    body = (workspace / "reports" / names[0]).read_text()
    assert "| cand | falseAdvanceRate | 40/40 |" in body
    assert "| incumbent | falseAdvanceRate | 34/40 |" in body
    assert "fewer than two arms were selected" not in body
