"""S1's three commands (`compare`, `index rebuild`, `models --tested`) plus S2's `attest`,
`validate`, and `run` (plan §3.6a) — the runner-spec's §7 Step 2, the last S2 unit. Exit codes are
§3.6a's closed set, amended by the dispatch-failure note's §4(d) artifacts-before-exit-4 clause.
"""

from __future__ import annotations

import json

import pytest
from conftest import classification_aggregates, item, model_fields, run

from modelbench import cli, hostinfo
from modelbench.cli import main
from modelbench.lmstudio import LMStudioCallTimeout, LoadResult, ModelInfo
from modelbench.results import ItemResult, ItemTiming, store
from modelbench.runner import DispatchFailureDisclosure

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
        ItemResult(
            itemId="g39", pairingKey=("g39",), outcome="pass",
            scoreable={"falseAdvanceRate": True}, counts={},
            timing=ItemTiming(wallClockMs=1300.0, calls=(), withheldFor=None), detail={},
        )
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


def test_validate_and_run_are_now_recognized_commands(capsys) -> None:
    """Runner-spec §7 Step 2: `validate` and `run` graduate from "a later S2 unit's, deliberately
    still absent" to real commands, closing S2. Read from `--help`'s command list rather than a
    bare `main(["validate"])`/`main(["run"])` exit code: **both** an unrecognized command and a
    recognized one missing its own required `--pack` exit `2`, so the exit code alone cannot tell
    "not shipped" from "shipped but called wrong" apart — only the parser's own subcommand list
    can (this is also `test_bad_arguments_exit_two`'s `main(["nosuchcommand"])` case, still exit 2,
    still meaning "unrecognized")."""
    main(["--help"])
    out = capsys.readouterr().out
    assert "{compare,index,models,attest,validate,run}" in out
    assert main(["validate"]) == 2  # recognized, but --pack is required — still exit 2, usage
    assert main(["run"]) == 2  # recognized, but --pack/--model are required — still exit 2


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


# --- attest (plan §3.6a) -------------------------------------------------------------------------
#
# `_cmd_attest` constructs its own `LMStudio(args.api_base_url)`, which would otherwise open a real
# socket; every test below monkeypatches `cli.LMStudio` with `_FakeLMStudio`, the same
# injected-network seam `tests/test_lmstudio.py` uses at the `LMStudio(opener=...)` layer, one
# level up.

ATTESTED_SET_FLAGS = [
    "--set", "lmStudioAppVersion=0.3.31",
    "--set", "kvCacheSetting=f16",
    "--set", "hostRamGb=16",
    "--set", "otherResidentWorkloads=docker: falkordb-dev, windows desktop session",
]


class _FakeLMStudio:
    """Stands in for `modelbench.lmstudio.LMStudio` — records the `base_url` it was built with
    and returns a canned `probe()` outcome, never opening a socket."""

    last_base_url: str | None = None

    def __init__(self, base_url: str, probe_result: str = "api-v0") -> None:
        self.base_url = base_url
        self._probe_result = probe_result
        type(self).last_base_url = base_url

    def probe(self) -> str:
        return self._probe_result


def _patch_lmstudio(monkeypatch, probe_result: str = "api-v0") -> None:
    monkeypatch.setattr(
        cli, "LMStudio", lambda base_url: _FakeLMStudio(base_url, probe_result)
    )


def test_attest_writes_a_host_json_matching_the_schema_and_exits_zero(
    workspace, capsys, monkeypatch
) -> None:
    _patch_lmstudio(monkeypatch)
    code = main(
        ["attest", "--root", str(workspace), "--api-base-url", "http://localhost:1234"]
        + ATTESTED_SET_FLAGS
    )
    out = capsys.readouterr().out
    assert code == 0
    path = workspace / "host.json"
    assert f"wrote {path}" in out
    assert hostinfo.validate_host_info(json.loads(path.read_text())) == []


def test_attest_writes_exactly_the_four_attested_fields_and_residency_source_only(
    workspace, monkeypatch
) -> None:
    """Plan §3.4.4's schema, checked field by field — including that `observedAtAttestation`
    carries `residencySource` and **omits** `runtimeName`/`runtimeVersion` (plan-gate P4-6:
    neither probed endpoint exposes a `runtime`, and `attest` has no model to call one against)."""
    _patch_lmstudio(monkeypatch)
    main(["attest", "--root", str(workspace)] + ATTESTED_SET_FLAGS)
    data = json.loads((workspace / "host.json").read_text())

    assert data["schemaVersion"] == 1
    assert data["apiBaseUrl"] == "http://localhost:1234"  # the CLI's own default
    assert data["attested"] == {
        "lmStudioAppVersion": "0.3.31",
        "kvCacheSetting": "f16",
        "hostRamGb": 16,
        "otherResidentWorkloads": ["docker: falkordb-dev", "windows desktop session"],
    }
    assert data["observedAtAttestation"] == {"residencySource": "lmstudio-api-v0"}
    assert "runtimeName" not in data["observedAtAttestation"]
    assert "runtimeVersion" not in data["observedAtAttestation"]


def test_attest_uses_the_given_api_base_url(workspace, monkeypatch) -> None:
    _patch_lmstudio(monkeypatch)
    main(
        ["attest", "--root", str(workspace), "--api-base-url", "http://10.0.0.5:1234"]
        + ATTESTED_SET_FLAGS
    )
    assert _FakeLMStudio.last_base_url == "http://10.0.0.5:1234"
    data = json.loads((workspace / "host.json").read_text())
    assert data["apiBaseUrl"] == "http://10.0.0.5:1234"


def test_attest_exits_three_when_lm_studio_is_unreachable(workspace, capsys, monkeypatch) -> None:
    """§3.4.4a's first distinguishing message — no `host.json` is written."""
    _patch_lmstudio(monkeypatch, probe_result="unreachable")
    code = main(
        ["attest", "--root", str(workspace), "--api-base-url", "http://localhost:1234"]
        + ATTESTED_SET_FLAGS
    )
    err = capsys.readouterr().err
    assert code == 3
    assert "not reachable at http://localhost:1234" in err
    assert not (workspace / "host.json").exists()


def test_attest_exits_three_when_only_v1_answers(workspace, capsys, monkeypatch) -> None:
    """§3.4.4a's second distinguishing message — the OpenAI-compatible surface answered but not
    LM Studio's native catalog, a *different* message from plain unreachability."""
    _patch_lmstudio(monkeypatch, probe_result="v1-only")
    code = main(["attest", "--root", str(workspace)] + ATTESTED_SET_FLAGS)
    err = capsys.readouterr().err
    assert code == 3
    assert "not LM Studio's native /api/v0 catalog" in err
    assert not (workspace / "host.json").exists()


def test_attest_exits_two_when_the_api_base_url_is_empty(workspace, capsys, monkeypatch) -> None:
    """P13-5 (review Pass 13) — `_cmd_attest` previously caught only `hostinfo.AttestProbeFailed`.
    `hostinfo.attest`'s own defensive re-validation raises `HostInfoError` on an empty
    `--api-base-url` (M.4/B's exact repro), which escaped as an uncaught traceback, exit `1` —
    outside §3.6a's closed set."""
    _patch_lmstudio(monkeypatch)
    code = main(
        ["attest", "--root", str(workspace), "--api-base-url", ""] + ATTESTED_SET_FLAGS
    )
    err = capsys.readouterr().err
    assert code == 2
    assert err.strip()
    assert not (workspace / "host.json").exists()


def test_attest_exits_two_when_stdin_is_exhausted_while_prompting(
    workspace, capsys, monkeypatch
) -> None:
    """P13-6 (review Pass 13) — a partly-specified non-interactive invocation with no piped stdin
    previously raised an uncaught `EOFError` (M.4/C). `--set` is §3.6a's own non-interactive
    route, so this is normal usage, not abuse; the fix names every field still unset."""
    _patch_lmstudio(monkeypatch)

    def _no_more_input(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _no_more_input)
    code = main(
        [
            "attest", "--root", str(workspace),
            "--set", "lmStudioAppVersion=0.3.31",
            "--set", "kvCacheSetting=f16",
        ]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "hostRamGb" in err
    assert "otherResidentWorkloads" in err
    assert not (workspace / "host.json").exists()


def test_attest_prompts_interactively_for_fields_not_given_via_set(
    workspace, monkeypatch
) -> None:
    """§3.6a: "Prompts for the four operator-attested fields" is the default; `--set` is the
    non-interactive override. Only `hostRamGb` is left unset here."""
    _patch_lmstudio(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt="": "32")
    code = main(
        [
            "attest", "--root", str(workspace),
            "--set", "lmStudioAppVersion=0.3.31",
            "--set", "kvCacheSetting=f16",
            "--set", "otherResidentWorkloads=",
        ]
    )
    assert code == 0
    data = json.loads((workspace / "host.json").read_text())
    assert data["attested"]["hostRamGb"] == 32


def test_attest_exits_two_on_an_unrecognized_set_key(workspace, capsys, monkeypatch) -> None:
    _patch_lmstudio(monkeypatch)
    code = main(["attest", "--root", str(workspace), "--set", "notAField=x"])
    err = capsys.readouterr().err
    assert code == 2
    assert "notAField" in err
    assert not (workspace / "host.json").exists()


def test_attest_exits_two_on_a_malformed_set_pair(workspace, capsys, monkeypatch) -> None:
    _patch_lmstudio(monkeypatch)
    code = main(["attest", "--root", str(workspace), "--set", "lmStudioAppVersion"])
    err = capsys.readouterr().err
    assert code == 2
    assert "key=value" in err


def test_attest_exits_two_on_a_non_integer_host_ram_gb(workspace, capsys, monkeypatch) -> None:
    _patch_lmstudio(monkeypatch)
    code = main(
        [
            "attest", "--root", str(workspace),
            "--set", "lmStudioAppVersion=0.3.31",
            "--set", "kvCacheSetting=f16",
            "--set", "hostRamGb=sixteen",
            "--set", "otherResidentWorkloads=",
        ]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "hostRamGb" in err
    assert not (workspace / "host.json").exists()


def test_attest_other_resident_workloads_empty_string_is_an_empty_list(
    workspace, monkeypatch
) -> None:
    """A clean box has no other resident workloads — `[]`, not `[""]` (the same absent-versus-
    empty discipline the fingerprint applies one file over)."""
    _patch_lmstudio(monkeypatch)
    main(
        [
            "attest", "--root", str(workspace),
            "--set", "lmStudioAppVersion=0.3.31",
            "--set", "kvCacheSetting=f16",
            "--set", "hostRamGb=16",
            "--set", "otherResidentWorkloads=",
        ]
    )
    data = json.loads((workspace / "host.json").read_text())
    assert data["attested"]["otherResidentWorkloads"] == []


# --- validate (runner-spec §6.1) ------------------------------------------------------------------
#
# Structural only — no LM Studio connection, no model catalog. `load_pack`/`validate_pack`'s own
# axes are `packs.py`'s tested surface (`tests/test_packs.py`); this only confirms `_cmd_validate`
# wires them and maps their outcomes onto §3.6a's exit codes.

VALIDATE_MANIFEST = {
    "packId": "validate-fixture",
    "packVersion": "1.0.0",
    "role": "guard-judge",
    "schemaVersion": 1,
    "environment": {"requires": ["lmstudio-chat"]},
    "sampling": {"seed": 20260902, "pairingKey": ["itemId"], "analysisUnit": "itemId"},
    "metrics": {"verdictMetrics": ["falseAdvanceRate"], "headlineMetric": "falseAdvanceRate"},
}


@pytest.fixture()
def valid_pack_dir(tmp_path):
    pack_dir = tmp_path / "a-pack"
    pack_dir.mkdir()
    (pack_dir / "pack.json").write_text(json.dumps(VALIDATE_MANIFEST))
    return pack_dir


def test_validate_exits_zero_on_a_structurally_valid_pack(valid_pack_dir, capsys) -> None:
    code = main(["validate", "--pack", str(valid_pack_dir)])
    out = capsys.readouterr().out
    assert code == 0
    assert "validate-fixture 1.0.0 (guard-judge): valid" in out


def test_validate_exits_four_on_a_pack_load_error(tmp_path, capsys) -> None:
    missing = tmp_path / "nope"
    code = main(["validate", "--pack", str(missing)])
    err = capsys.readouterr().err
    assert code == 4
    assert "nope" in err


def test_validate_exits_four_when_validate_pack_reports_problems(valid_pack_dir, capsys) -> None:
    manifest = json.loads((valid_pack_dir / "pack.json").read_text())
    del manifest["environment"]
    (valid_pack_dir / "pack.json").write_text(json.dumps(manifest))
    code = main(["validate", "--pack", str(valid_pack_dir)])
    err = capsys.readouterr().err
    assert code == 4
    assert "environment.requires" in err


def test_validate_strict_is_a_deliberate_deferral_not_a_silent_no_op(valid_pack_dir) -> None:
    """Runner-spec §9: `--strict`'s semantics are never stated anywhere in the plan, and the spec
    names a `NotImplementedError` + citing comment as the sanctioned way to leave that open rather
    than accept the flag as a no-op. `main()` catches only `SystemExit`, so this propagates."""
    with pytest.raises(NotImplementedError, match="strict"):
        main(["validate", "--pack", str(valid_pack_dir), "--strict"])


def test_validate_without_strict_still_runs_normally(valid_pack_dir, capsys) -> None:
    """The negative of the above: omitting `--strict` entirely must not somehow trip the deferral
    (`args.strict` defaults `False` via `action="store_true"`)."""
    code = main(["validate", "--pack", str(valid_pack_dir)])
    assert code == 0


# --- run (runner-spec §6.2) -----------------------------------------------------------------------
#
# `run`'s own refusal points before any item/conversation is ever driven (LM Studio unreachable,
# the callSurface/catalog-type cross-check, the tool-calling eligibility gate, a stale attestation)
# are exercised through the real `run_pack` against a minimal on-disk pack and a stub `LMStudio`
# (`cli.LMStudio` monkeypatched, mirroring `_patch_lmstudio` above) — `run_pack`'s own capture-order
# discrimination is already `tests/test_runner.py`'s (offline, no live LM Studio needed here
# either). The success path and the dispatch-failure store-before-exit-4 ordering are exercised by
# faking `cli.run_pack` directly, since by that point it is `_cmd_run`'s own sequencing being
# tested, not `run_pack`'s.

ITEM_PACK = "guard-judge-run-fixture"
ITEM_PACK_MANIFEST = {
    "packId": ITEM_PACK,
    "packVersion": "1.0.0",
    "role": "guard-judge",
    "schemaVersion": 1,
    "environment": {"requires": ["lmstudio-chat"]},
    "prompt": {"historyReplay": "none"},
    "sampling": {"seed": 20260902, "pairingKey": ["itemId"], "analysisUnit": "itemId"},
    "metrics": {"verdictMetrics": ["falseAdvanceRate"], "headlineMetric": "falseAdvanceRate"},
}

TOOL_CALLER_PACK = "tool-caller-run-fixture"
TOOL_CALLER_PACK_MANIFEST = {
    "packId": TOOL_CALLER_PACK,
    "packVersion": "1.0.0",
    "role": "tool-caller",
    "schemaVersion": 1,
    "environment": {"requires": ["lmstudio-chat"]},
    "sampling": {"seed": 20260909, "pairingKey": ["scriptId"], "analysisUnit": "scriptId"},
    "metrics": {"verdictMetrics": ["cleanThroughTurnH"], "headlineMetric": "cleanThroughTurnH"},
}


@pytest.fixture()
def run_workspace(tmp_path):
    for pack_id, manifest in (
        (ITEM_PACK, ITEM_PACK_MANIFEST),
        (TOOL_CALLER_PACK, TOOL_CALLER_PACK_MANIFEST),
    ):
        pack_dir = tmp_path / "packs" / pack_id
        pack_dir.mkdir(parents=True)
        (pack_dir / "pack.json").write_text(json.dumps(manifest))
    return tmp_path


def _model_info(*, model_type: str = "llm", capabilities: tuple[str, ...] | None = ("tool_use",)):
    return ModelInfo(
        id="qwen/qwen3-4b-2507",
        object="model",
        type=model_type,
        publisher="qwen",
        arch="qwen3",
        compatibility_type="gguf",
        quantization="Q4_K_M",
        state="loaded",
        max_context_length=262144,
        capabilities=capabilities,
        loaded_context_length=8192,
    )


class _StubLMStudioForRun:
    """A minimal `LMStudio`-shaped stub for `run`-level CLI tests — only what `run_pack`'s
    capture-order refusal paths touch before any item/conversation is ever driven (`probe`,
    `catalog`, `residency`, `warm_up`). `run`'s actual item/tool-caller driving is `runner.py`'s
    own already-tested surface (`tests/test_runner.py`); re-testing it here is out of this step's
    scope (runner-spec §7 Step 2)."""

    def __init__(
        self,
        base_url: str,
        *,
        probe_result: str = "api-v0",
        catalog_result: list[ModelInfo] | None = None,
        warm_up_error: Exception | None = None,
    ) -> None:
        self.base_url = base_url
        self._probe_result = probe_result
        self._catalog_result = list(catalog_result or [])
        self._warm_up_error = warm_up_error

    def probe(self) -> str:
        return self._probe_result

    def catalog(self) -> list[ModelInfo]:
        return list(self._catalog_result)

    def residency(self):
        return []

    def warm_up(self, model_key, *, call_surface, system_prompt, was_resident_before, timeout_s):
        if self._warm_up_error is not None:
            raise self._warm_up_error
        return LoadResult(
            wallClockMs=500.0,
            wasResidentBefore=was_resident_before,
            runtime={"name": "llama.cpp", "version": "1.52.0"} if call_surface == "chat" else None,
            stats=None,
        )


def _patch_run_lmstudio(monkeypatch, **kwargs) -> None:
    monkeypatch.setattr(cli, "LMStudio", lambda base_url: _StubLMStudioForRun(base_url, **kwargs))


def _write_host_json(root, *, runtime_name="llama.cpp", runtime_version="1.52.0") -> None:
    hostinfo.write_host_info(
        root,
        {
            "schemaVersion": 1,
            "apiBaseUrl": "http://localhost:1234",
            "attested": {
                "lmStudioAppVersion": "0.3.31",
                "kvCacheSetting": "f16",
                "hostRamGb": 16,
                "otherResidentWorkloads": [],
            },
            "attestedAt": "2026-09-10T00:00:00Z",
            "observedAtAttestation": {
                "residencySource": "lmstudio-api-v0",
                "runtimeName": runtime_name,
                "runtimeVersion": runtime_version,
                "runtimeObservedAt": "2026-09-10T00:00:00Z",
            },
        },
    )


def test_run_exits_four_on_a_pack_load_error(run_workspace, capsys) -> None:
    code = main(
        ["run", "--root", str(run_workspace), "--pack", "no-such-pack", "--model", "m"]
    )
    err = capsys.readouterr().err
    assert code == 4
    assert "no-such-pack" in err


def test_run_exits_four_when_validate_pack_reports_problems(run_workspace, capsys) -> None:
    manifest_path = run_workspace / "packs" / ITEM_PACK / "pack.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["environment"]
    manifest_path.write_text(json.dumps(manifest))
    code = main(["run", "--root", str(run_workspace), "--pack", ITEM_PACK, "--model", "m"])
    err = capsys.readouterr().err
    assert code == 4
    assert "environment.requires" in err


def test_run_exits_five_when_host_json_is_absent(run_workspace, capsys) -> None:
    """§3.6a's `5`: `host.json` absent. Reached before `run_pack` even opens a connection —
    `_cmd_run` reads `host.json` itself to get `apiBaseUrl` before constructing `LMStudio`."""
    code = main(["run", "--root", str(run_workspace), "--pack", ITEM_PACK, "--model", "m"])
    err = capsys.readouterr().err
    assert code == 5
    assert "host.json" in err
    assert not (run_workspace / "results").exists()  # refused before anything was ever written


def test_run_exits_five_when_host_json_fails_schema_validation(run_workspace, capsys) -> None:
    (run_workspace / "host.json").write_text(json.dumps({"schemaVersion": 1}))
    code = main(["run", "--root", str(run_workspace), "--pack", ITEM_PACK, "--model", "m"])
    err = capsys.readouterr().err
    assert code == 5
    assert "fails schema validation" in err


def test_run_exits_three_when_lm_studio_is_unreachable(run_workspace, capsys, monkeypatch) -> None:
    """§3.6a's `3`, the first of `probe()`'s two distinct negative outcomes."""
    _write_host_json(run_workspace)
    _patch_run_lmstudio(monkeypatch, probe_result="unreachable")
    code = main(
        ["run", "--root", str(run_workspace), "--pack", ITEM_PACK,
         "--model", "qwen/qwen3-4b-2507"]
    )
    err = capsys.readouterr().err
    assert code == 3
    assert "not reachable" in err


def test_run_exits_three_when_lm_studio_is_v1_only(run_workspace, capsys, monkeypatch) -> None:
    """§3.6a's `3`, the second of `probe()`'s two distinct negative outcomes — a different message
    from plain unreachability, sharing the code (§3.4.4a)."""
    _write_host_json(run_workspace)
    _patch_run_lmstudio(monkeypatch, probe_result="v1-only")
    code = main(
        ["run", "--root", str(run_workspace), "--pack", ITEM_PACK,
         "--model", "qwen/qwen3-4b-2507"]
    )
    err = capsys.readouterr().err
    assert code == 3
    assert "/api/v0" in err


def test_run_exits_four_on_call_surface_versus_catalog_type_contradiction(
    run_workspace, capsys, monkeypatch
) -> None:
    """§3.4.4a: the pack declares `callSurface=chat` but the catalog says `type=embeddings`."""
    _write_host_json(run_workspace)
    embeddings_model = _model_info(model_type="embeddings", capabilities=None)
    _patch_run_lmstudio(monkeypatch, catalog_result=[embeddings_model])
    code = main(
        ["run", "--root", str(run_workspace), "--pack", ITEM_PACK,
         "--model", "qwen/qwen3-4b-2507"]
    )
    err = capsys.readouterr().err
    assert code == 4
    assert "callSurface" in err


def test_run_exits_four_on_the_tool_calling_eligibility_gate(
    run_workspace, capsys, monkeypatch
) -> None:
    """§3.6: a `tool-caller` pack against a catalog model that is `llm`-typed but lacks
    `tool_use` — a DIFFERENT exit-4 trigger from the callSurface/type contradiction above, and the
    gate that only runs at all for a `tool-caller` role."""
    _write_host_json(run_workspace)
    ineligible_model = _model_info(capabilities=())
    _patch_run_lmstudio(monkeypatch, catalog_result=[ineligible_model])
    code = main(
        ["run", "--root", str(run_workspace), "--pack", TOOL_CALLER_PACK,
         "--model", "qwen/qwen3-4b-2507"]
    )
    err = capsys.readouterr().err
    assert code == 4
    assert "not eligible" in err


def test_run_exits_three_when_warm_up_times_out(run_workspace, capsys, monkeypatch) -> None:
    _write_host_json(run_workspace)
    _patch_run_lmstudio(
        monkeypatch, catalog_result=[_model_info()], warm_up_error=LMStudioCallTimeout("timed out")
    )
    code = main(
        ["run", "--root", str(run_workspace), "--pack", ITEM_PACK,
         "--model", "qwen/qwen3-4b-2507"]
    )
    err = capsys.readouterr().err
    assert code == 3
    assert "first-call-timeout" in err


def test_run_exits_five_when_attestation_trip_wire_is_stale(
    run_workspace, capsys, monkeypatch
) -> None:
    """§3.6a's `5`, the second trigger: `host.json` attested `runtimeVersion=1.51.0`, the warm-up
    observes `1.52.0` — a mismatch on the `"compared"` outcome (step 6)."""
    _write_host_json(run_workspace, runtime_name="llama.cpp", runtime_version="1.51.0")
    _patch_run_lmstudio(monkeypatch, catalog_result=[_model_info()])
    code = main(
        ["run", "--root", str(run_workspace), "--pack", ITEM_PACK,
         "--model", "qwen/qwen3-4b-2507"]
    )
    err = capsys.readouterr().err
    assert code == 5
    assert "re-check the app version" in err


def test_run_exits_zero_and_stores_the_record_on_a_full_success(
    run_workspace, monkeypatch, capsys
) -> None:
    """`_cmd_run`'s own sequencing on the clean path — `run_pack`'s actual driving is faked here,
    since this step's job is the wiring around it, not `run_pack` itself."""
    fake_result = run("cli-run-happy", items=[item("g00", correct=True)])
    monkeypatch.setattr(cli, "run_pack", lambda pack, cfg, *, lmstudio, root: (fake_result, ()))
    _write_host_json(run_workspace)

    code = main(
        ["run", "--root", str(run_workspace), "--pack", ITEM_PACK, "--model", "m"]
    )
    out = capsys.readouterr().out
    stored_path = run_workspace / "results" / "runs" / "cli-run-happy.json"

    assert code == 0
    assert f"stored: {stored_path}" in out
    assert stored_path.exists()
    assert "PACK DISPATCH FAILURES" not in out


def test_run_stores_the_record_before_returning_exit_four_on_dispatch_failure_disclosures(
    run_workspace, monkeypatch, capsys
) -> None:
    """Dispatch-failure note §4(d): "the record is written, then run exits 4" — a data-quality
    finding discovered only once a partial record exists, never an aborted run. `run_pack`'s own
    production of a disclosure is already `test_runner.py`'s (E2); this is `_cmd_run`'s own
    ordering obligation, isolated by faking `run_pack`'s return so only the CLI's own sequencing
    is under test.
    """
    fake_result = run("cli-dispatch-failure", items=[item("s0", correct=False)])
    disclosure = DispatchFailureDisclosure(
        scriptId="dispatch-fails", turn=3, tool="lookup_product_fact",
        reason="RuntimeError: lookup_product_fact is broken",
    )
    monkeypatch.setattr(
        cli, "run_pack", lambda pack, cfg, *, lmstudio, root: (fake_result, (disclosure,))
    )
    _write_host_json(run_workspace)

    code = main(
        ["run", "--root", str(run_workspace), "--pack", TOOL_CALLER_PACK, "--model", "m"]
    )
    out = capsys.readouterr().out
    stored_path = run_workspace / "results" / "runs" / "cli-dispatch-failure.json"

    assert code == 4
    assert "PACK DISPATCH FAILURES" in out
    assert "dispatch-fails" in out and "lookup_product_fact" in out
    # the load-bearing ordering claim: the record is on disk, not skipped because the run "failed"
    assert stored_path.exists()


# ==================================================================================================
# S3 spec §3.3/§7.2 — `_cmd_run`'s deterministic-arm hook: a real, on-disk `embedder` pack whose
# `"scorer": "retrieval"` resolves to the real `modelbench.scoring.retrieval` module (so
# `deterministic_arm()` runs for real, over a tiny fixture corpus/query set — offline throughout,
# BM25 makes no live call). `run_pack` itself is faked (this file's own S2 convention), since this
# step's job is `_cmd_run`'s sequencing around it, not `run_pack`'s own driving.
# ==================================================================================================

EMBEDDER_PACK = "embedder-run-fixture"
EMBEDDER_PACK_MANIFEST = {
    "packId": EMBEDDER_PACK,
    "packVersion": "1.0.0",
    "role": "embedder",
    "schemaVersion": 1,
    "scorer": "retrieval",
    "environment": {"requires": ["lmstudio-embeddings"]},
    "data": {
        "items": "queries.jsonl",
        "corpus": "corpus.jsonl",
        "corpusEmbeddings": "corpus.embeddings.json",
    },
    "embedding": {
        "queryPrefix": "",
        "documentPrefix": "",
        "bm25": {"k1": 1.2, "b": 0.75, "stopwordsFile": "bm25_stopwords.txt"},
    },
    "sampling": {"seed": 1, "pairingKey": ["itemId"], "analysisUnit": "itemId"},
    "metrics": {"verdictMetrics": ["mrr"], "headlineMetric": "mrr", "recallAtK": {"ks": [1]}},
}


def _write_embedder_pack_fixture(root) -> None:
    pack_dir = root / "packs" / EMBEDDER_PACK
    pack_dir.mkdir(parents=True)
    (pack_dir / "pack.json").write_text(json.dumps(EMBEDDER_PACK_MANIFEST))
    corpus_rows = [
        {"docId": "d1", "text": "apple banana"},
        {"docId": "d2", "text": "car train"},
    ]
    (pack_dir / "corpus.jsonl").write_text(
        "\n".join(json.dumps(r) for r in corpus_rows) + "\n", encoding="utf-8"
    )
    query_row = {"itemId": "q1", "query": "apple banana", "relevantDocIds": ["d1"]}
    (pack_dir / "queries.jsonl").write_text(json.dumps(query_row) + "\n", encoding="utf-8")
    (pack_dir / "bm25_stopwords.txt").write_text("the\na\n", encoding="utf-8")


def test_cmd_run_stores_the_deterministic_arm_under_the_same_session_id(
    run_workspace, monkeypatch, capsys
) -> None:
    _write_embedder_pack_fixture(run_workspace)
    fake_result = run(
        "cli-run-embedder",
        role="embedder",
        arm_kind="model",
        call_surface="embeddings",
        items=[],
        session_id="sess-xyz",
    )
    monkeypatch.setattr(cli, "run_pack", lambda pack, cfg, *, lmstudio, root: (fake_result, ()))
    _write_host_json(run_workspace)

    code = main(
        [
            "run", "--root", str(run_workspace), "--pack", EMBEDDER_PACK, "--model", "m",
            "--session", "sess-xyz",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "stored (deterministic arm):" in out

    # Asserted via store()'s own file output — the runs directory — not the in-memory RunResult:
    # two distinct stored records, one `model`, one `deterministic`, sharing the given sessionId.
    stored_files = sorted((run_workspace / "results" / "runs").glob("*.json"))
    assert len(stored_files) == 2
    bodies = [json.loads(p.read_text()) for p in stored_files]
    assert {b["armKind"] for b in bodies} == {"model", "deterministic"}
    assert {b["sessionId"] for b in bodies} == {"sess-xyz"}


def test_cmd_run_skips_the_deterministic_arm_hook_for_tool_caller(
    run_workspace, monkeypatch, capsys
) -> None:
    """`tool-caller` resolves its scorer through `_load_conversation_scorer`, a different
    function this hook does not touch — never a `deterministic_arm` no-op path reached at all."""
    fake_result = run(
        "cli-run-toolcaller-only", role="tool-caller", items=[item("s0", correct=True)]
    )
    monkeypatch.setattr(cli, "run_pack", lambda pack, cfg, *, lmstudio, root: (fake_result, ()))
    _write_host_json(run_workspace)

    code = main(["run", "--root", str(run_workspace), "--pack", TOOL_CALLER_PACK, "--model", "m"])
    out = capsys.readouterr().out

    assert code == 0
    assert "stored (deterministic arm):" not in out
    stored_files = sorted((run_workspace / "results" / "runs").glob("*.json"))
    assert len(stored_files) == 1
