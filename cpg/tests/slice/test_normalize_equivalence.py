"""P1-05 behaviour (2) — the ported normalizer reproduces the vertical slice's stored bundles (**5a: Python half**).

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-05 (2); the rule is `docs/plans/connected-code-graph-p1.md` §8 ("Equivalence before improvement") and D7; D9 is the skip rule below. The oracle is the slice's
S14a stored bundle for `falkor-chat/server`, `cpg/.ccg-work/vertical-slice/s14a/<unitId>.bundle.jsonl` (sha256 pinned
below), with the raw facts and staged tree beside it. These tests carry the `slice` marker and skip when an artifact is
absent: **a skip is unrun, never green.** The run summary must read `0 skipped`.

**5b adds the whole bundle**, both units (`falkor-chat/server` and `salesperson`): the bundle the production code writes through
`ccg.atrest.write_bundle` is compared with the stored one under three definitions (strict bytes with the slice's own
`extract.sc` as the hashed script; bytes with the three header fields that are inputs or stamps exempt; the multiset of
records), see `test_whole_bundle_*`.

The first dispatch covers what the Python frontend decides: every `symbol` record's identity, kind, name, parent, span
and fingerprint (all but `complexity` and `lastChanged*`, which need conds and blame; 5b) and every `import` record on every field. Expected values are read from the stored bundle, never from the code under test. The whole-bundle byte
comparison (and the JS unit, `salesperson`) is the second dispatch.
"""
from __future__ import annotations

import atexit
import collections
import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

import overlay_5d
from ccg import atrest, bundle, ids
from ccg.units import Unit
from extractors.joern import run
from extractors.joern.normalize import assemble_bundle, core, python

CPG = Path(__file__).resolve().parent.parent.parent
S14A = CPG / ".ccg-work" / "vertical-slice" / "s14a"
REPO_ID = "graphmind-ai-lab"
FCS_ROOT = "falkor-chat/server"
FCS_BUNDLE = S14A / "0fb899742d4d3687495c30357ca16253.bundle.jsonl"
#: sha256 of the stored bundle recorded 2026-11-10 (read from the file in this run).
FCS_BUNDLE_SHA = "a0cae513fb6795605a5818942942869fbb1c575de7cdbd4a264ff82833ba1c86"
#: counts recorded in the stored bundle's header (`counts`).
FCS_COUNTS = {"symbol": 4686, "import": 926, "file": 107}
#: stored `import` records that carry a `symbolId` (counted in the stored bundle in this run).
FCS_IMPORTS_WITH_SYMBOL = 87


def need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"slice artifact absent: {path} (UNRUN, not green)")
    return path


def stored(kind: str) -> list[dict]:
    out = []
    for line in need(FCS_BUNDLE).read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        if o.get("type") == kind:
            out.append(o)
    return out


def facts_of():
    cpg = need(S14A / "fcs" / "cpg")
    found = sorted(cpg.glob("*.facts.jsonl"))
    assert len(found) == 1
    return core.load_facts(found[0])


@pytest.fixture(scope="module")
def built():
    facts = facts_of()
    B = core.Builder("python", FCS_ROOT, REPO_ID, facts, core.Source(need(S14A / "fcs" / "stage")))
    scopes, _, _ = python.build_scopes(B)
    python.apply_py_spans(B, scopes)
    for s in scopes.values():
        s.sid = ids.symbol_id("python", REPO_ID, s.qname)
    return B, scopes


@pytest.mark.slice
def test_the_stored_bundle_is_the_artifact_recorded_with_its_sha256_and_counts():
    assert hashlib.sha256(need(FCS_BUNDLE).read_bytes()).hexdigest() == FCS_BUNDLE_SHA
    header = json.loads(FCS_BUNDLE.read_text(encoding="utf-8").splitlines()[0])
    assert {k: header["counts"][k] for k in FCS_COUNTS} == FCS_COUNTS


@pytest.mark.slice
def test_every_stored_symbol_is_rebuilt_field_for_field(built):
    B, scopes = built
    want = {r["id"]: r for r in stored("symbol")}
    assert len(want) == FCS_COUNTS["symbol"]
    got = {}
    for s in scopes.values():
        rec = {"id": s.sid, "qname": s.qname, "path": B.P(s.file), "fileId": ids.file_id(REPO_ID, B.P(s.file)),
               "kind": s.kind, "name": core.sym_name(s), "nameNorm": core.name_norm(core.sym_name(s)),
               "lineStart": core.span_start(B, s), "lineEnd": max(s.lineEnd, core.span_start(B, s)),
               "fingerprint": core.fingerprint(B, s)}
        if s.parent is not None and s.kind != "module":
            rec["parentId"] = s.parent.sid
        got[s.sid] = rec
    assert set(got) == set(want)
    keys = ("qname", "path", "fileId", "kind", "name", "nameNorm", "lineStart", "lineEnd", "fingerprint", "parentId")
    bad = [(i, k) for i, r in got.items() for k in keys if r.get(k) != want[i].get(k)]
    assert bad == []


@pytest.mark.slice
def test_every_stored_import_is_rebuilt_field_for_field(built):
    B, scopes = built
    files = sorted(f for f in core.lang_files(B.facts, "python") if B.src.exists(f))
    got = []
    for f in files:
        for k, r in enumerate(python.py_imports(B, f)):
            fid = ids.file_id(REPO_ID, B.P(f))
            rec = {"id": ids.import_id(fid, k), "fileId": fid, "line": r["line"], "ordinal": k, "module": r["module"],
                   "names": r["names"], "dynamic": r["dynamic"]}
            if r.get("resolvedPath"):
                rec["resolvedPath"] = r["resolvedPath"]
            sid = core.import_symbol_id(B, scopes, f, r["line"])
            if sid:
                rec["symbolId"] = sid
            got.append(rec)
    want = [{k: v for k, v in r.items() if k != "type"} for r in stored("import")]
    assert sum("symbolId" in r for r in want) == FCS_IMPORTS_WITH_SYMBOL
    assert len(want) == FCS_COUNTS["import"]
    key = lambda r: r["id"]
    assert sorted(got, key=key) == sorted(want, key=key)


@pytest.mark.slice
def test_no_symbol_of_the_stored_unit_needed_the_orphan_parent_fallback(built):
    assert built[0].report["pythonParentFallbacks"] == 0


# -- 5b: the whole bundle, both units -------------------------------------------------------------------------------
REPO = CPG.parent
SLICE_SCRIPT = CPG / "spikes" / "vertical-slice" / "extract.sc"
SP_ROOT = "salesperson"
SP_BUNDLE = S14A / "c2341bca0404a6a3eadab1d72ba5f4db.bundle.jsonl"
SP_BUNDLE_SHA = "f2ef6f5add18691d94cecfcb05a742e03d3e06f00cf3fedea462d79886e13e65"
UNIT_TAG_OF_ROOT = {FCS_ROOT: "fcs", SP_ROOT: "sp"}
UNITS = {"fcs": (FCS_ROOT, "python", FCS_BUNDLE, FCS_BUNDLE_SHA), "sp": (SP_ROOT, "javascript", SP_BUNDLE, SP_BUNDLE_SHA)}
#: header fields that are an input or a stamp, not content: spec3 §8.5 (`builtAt`, `sandboxAuditIds`). The hash of the script file is
#: exempt in definition 2 only: the shipped `extract.sc` is no longer the slice's but for its header comment (P1-09 changed the projection).
EXEMPT = ("builtAt", "sandboxAuditIds")


_OVERLAID: dict[str, Path] = {}


def overlaid(tag: str) -> Path:
    """The oracle's facts file of ``tag`` with the 5d overlay (`tests/overlay_5d.py`): the hand-listed `rt` values and export-block
    `const` facts added, nothing else. The oracle itself is never rewritten (D9); the copy lives in a temporary directory."""
    if tag not in _OVERLAID:
        d = Path(tempfile.mkdtemp(prefix="ccg-5d-overlay-"))
        atexit.register(shutil.rmtree, d, ignore_errors=True)
        src = sorted(need(S14A / tag / "cpg").glob("*.facts.jsonl"))[0]
        _OVERLAID[tag] = overlay_5d.overlaid_facts(src, tag, d / f"{tag}.facts.jsonl")
    return _OVERLAID[tag]


def raw_of(tag: str, commit: str, audit_ids) -> run.RawFacts:
    root, lang = UNITS[tag][:2]
    cpg = need(S14A / tag / "cpg")
    facts = [overlaid(tag)]
    astgen = sorted(cpg.glob("*.astgen"))
    unit = Unit(ids.unit_id(REPO_ID, root, lang), REPO_ID, root, lang)
    return run.RawFacts(unit, commit, facts[0], tuple(audit_ids), need(S14A / tag / "stage"), cpg,
                        astgen[0] if astgen else None, cpg / "none", 0)


def stored_lines(tag: str) -> list[bytes]:
    return need(UNITS[tag][2]).read_bytes().split(b"\n")[:-1]


# -- the delta list: how a later step changes the output without ever editing a stored bundle ------------------------------
#: ORDERED list of ``(name, normalizerVersion, function)``. The expected bundle is the stored one with every delta applied in
#: order (P1-05 behaviour (2)). **To register a delta, append one entry:** a ``@delta("name", "<version>")`` function
#: ``fn(header: dict, records: list[dict]) -> int`` that edits both in place (remove or change records, add header fields,
#: lower the ``counts`` it affected) and returns how many records it touched. The framework itself sets the header's
#: ``tools.normalizerVersion`` to the entry's literal (a delta PINS its version; the step that adds it bumps
#: ``assemble_bundle.NORMALIZER_VERSION`` to the same value, and the last entry must equal that constant), and fails if the
#: ``counts`` no longer equal a recount of the records (a delta that forgets its counts). Landing order: 5c ("0.2"), P1-06
#: ("0.3"), P1-08 ("0.4"), P1-41 ("0.5"). A stored record that no delta explains makes the byte comparison below fail.
DELTAS: list = []


def delta(name: str, version: str):
    def register(fn):
        DELTAS.append((name, version, fn))
        return fn
    return register


@delta("drop-lambda-decorators", "0.2")
def _drop_lambda_decorators(header, records):
    """5c (b): a call that takes a lambda is no decorator (spec3 §5.12); the normalizer stops writing the record."""
    lambdas = {r["id"] for r in records if r["type"] == "symbol" and r["kind"] == "lambda"}
    kept = [r for r in records if not (r["type"] == "decorator" and r["targetId"] in lambdas)]
    removed = len(records) - len(kept)
    records[:] = kept
    header["counts"]["decorator"] -= removed
    return removed


# P1-06 (F-9): every stored `cond.code` the masker changes, written out BY HAND from the stored bundles and reviewed by eye
# (id -> expected code). Not derived from `mask_cond_code`: a cond the masker changes that is not listed makes the byte
# comparison fail. 41 + 2 = 43. A literal whose body holds a name-rule word ("key", "signal", "token", "conn", "session")
# becomes "…"; "assistant", "Message", "wait" and the other words stay.
P1_06_MASKED = {
    "fcs": {
        "09004242003bb47b8cb7a17714fa9631": 'not s["…"] == "assistant"',
        "3f9cf0e8a06e5089a1c65f40d7ce6839": 'not s["…"] == "assistant"',
        "7e519c77240d9560f73db4cd8bb9f3e8": 'not s["…"] == "assistant"',
        "a3000c028e3570fa74a8e81dd0c7abac": 'not s["…"] == "assistant"',
        "a5bf2cd16cfc4449e3e7dda95ed4f0ab": 'not s["…"] == "assistant"',
        "ba0c3ec577670b2f399e94b23a5dfb11": 'not s["…"] == "assistant"',
        "bb99aed46b63ed9bd25548c84bac4f96": 'not s["…"] == "assistant"',
        "c5d15c493fc30e85a55def0a8b955884": 'not s["…"] == "assistant"',
        "b2df3d62d332b4a218e8dc9c7ffe336f": 'not s["…"] == "ended"',
        "be3a9360dd119875f839105ae026e32e": 's["…"] == "onboarding"',
        "bef21a10b7e3dfc320c4a075ecf9b9b7": 'not s.get("…") == step_key',
        "bac0d8a256395f15e1a6d4ca115fe045": 'step["…"] is not None and step["…"] not in seen_steps',
        "d49917162bbacc02db16af61503bc66f": 'step["…"] is not None',
        "122c1abca29f3bf2d7f351318ff76783": '"…" in str(exc)',
        "d2c4c3fae6a0d08d0ba9c41b2b92d37e": '"…" in str(exc)',
        "d43bbad43d1cd8baa1dd857a105d8240": '"…" in str(exc)',
        "d5594bf47a1914b9c2f405218f22e9f8": '"…" in str(exc)',
        "def365b32a429a4c57e39c653e6d8fcf": '"…" in str(exc)',
        "e9bd70272e4ae40238be539655b7864e": '"…" in str(exc)',
        "1afb3da0fab6d8f417dd883b0c44d8f4": 'r["…"] == "Message"',
        "4850141e5b0912636b741c2de735bb70": 's["…"] == "Message"',
        "4733ad4479982d4aed3ba5a09b2368c0": 'r["…"] == "Chunk"',
        "41fec8db6f49f6cc8cd3c34de0c86881": 'not p["…"] == "Chunk"',
        "32208fe36346fbdbcacab0deb71205e3": 'item["…"] == 0',
        "71f0e16ed558681909e12b3425476ccc": 'not r.get("…", "Message") == "Message"',
        "806f9abedf3afc57a3ebed31859754ab": 'not s["…"] == "m1"',
        "9c7c6e38f0692ba2e781ee001be507aa": 'not s["…"] == "ch1"',
        "5eb6e9835bdc4f71de6834b9649d15e9": 'step.get("type") == "wait" and config.get("…")',
        "725214c0e4eb302a77c5e21d3fb3dd38": 'raw.get("…")',
        "749d77590ea68f7c53aea6012654f624": 'd["path"] in ("…", "…")',
        "82e16adf8bd8440d556fc5a784af3490": '"…" in warnings[1]',
        "8dd66855f862e5ea34207916efe0b583": 'meta = graph.ro_query(Repository._READ_META_CYPHER.format(label = label), {"…":key, "version":version})',
        "92988d794bcfff1e140852ea0febcc4a": '"…" in no_start',
        # raw code is 283 characters: masking first leaves room for 25 more characters before the cut at 200
        "a2a06f121ea91e3415f2020163c6b19a": 'not d["path"] in ("…", "…") or tmp44 = d["path"] d["path"].startswith("steps[") and not tmp45 = d["path"] d["path"].endswith((".type",".config")) or tmp46 = d["path"] d["path"].startswith("transitions',
        "aaf1e9b66d15e9df8ac0f2033d4f3502": 'tmp25 = {} tmp25["…"] = shop.presenter_login(body.key) tmp25',
        # raw code is 218 characters: 43 shrink to one, so the whole of it fits
        "acff4327f7f52bce9c5edc90ecbb889c": 'receipt = tmp117 = {} tmp117["status"] = "error" tmp117["error"] = "each batch item must be a dict with a string \'text\' " "…" tmp117["errorType"] = "MalformedItemError" tmp117',
        "b195b96abb852662424bc5be20414d28": 'two["…"] == ["a", "b"]',
        "d9dd47dbd9ec49dfc966ecedfd969ef2": 'structure and "…" in structure',
        "f0dcf3a45eb5e0d088d1bddc3c46dce0": 'run is None or run.get("status") != "waiting" or run.get("…") != step_key',
        "f23240429d99245cd993446cf218d57f": 'not step_key is None or s["…"] == step_key',
        "f8700da26e183023eb82d0832ff41eb4": '"…" not in one',
    },
    "sp": {
        "9c15884fb9273f59f1be7cf9f6a0eacb": "action?.kind === 'fieldError' && action.field === \"…\"",
        "bc2ae5073c05f00783febd768fd780aa": 'route === "…"',
    },
}



@delta("p1-06-mask-cond-code", "0.3")
def _mask_cond_code(header, records):
    """P1-06: a string literal in a condition that fails the token screen or the name rule is written as `"…"`; only the
    conds of `P1_06_MASKED` change, each to its hand-written text. No record is added or removed, so no count moves."""
    want = {i: code for unit in P1_06_MASKED.values() for i, code in unit.items()}
    touched = 0
    for r in records:
        if r["type"] == "cond" and r["id"] in want:
            assert r["code"] != want[r["id"]]
            r["code"] = want[r["id"]]
            touched += 1
    return touched


# P1-08: `header.skippedFiles` of the stored `salesperson` bundle, written BY HAND from its stage: of the 74 JS/TS sources staged
# under `salesperson/`, 41 are stored `file` records and these 33 are not: 28 match the closed `test` pattern (`*.test.*`,
# `*.spec.*`, a segment `e2e`) and 5 are the `src/i18n/` production files the frontend drops (`jssrc2cpg` 4.0.579 has no
# override, `extractors/joern/README.md`). Byte order. `falkor-chat/server` is Python: none.
P1_08_SKIPPED_SP = [
    {"path": "salesperson/src/api/dispatch.test.ts", "reason": "test"},
    {"path": "salesperson/src/api/hooks.test.tsx", "reason": "test"},
    {"path": "salesperson/src/components/message/Composer.test.tsx", "reason": "test"},
    {"path": "salesperson/src/components/message/DeadTurnNotice.test.tsx", "reason": "test"},
    {"path": "salesperson/src/components/message/MessageBubble.test.tsx", "reason": "test"},
    {"path": "salesperson/src/components/message/Transcript.test.tsx", "reason": "test"},
    {"path": "salesperson/src/components/message/TurnIndicator.test.tsx", "reason": "test"},
    {"path": "salesperson/src/components/message/composerNotice.test.ts", "reason": "test"},
    {"path": "salesperson/src/components/message/optimistic.test.ts", "reason": "test"},
    {"path": "salesperson/src/components/message/welcome.test.ts", "reason": "test"},
    {"path": "salesperson/src/components/sheets/BottomSheet.test.tsx", "reason": "test"},
    {"path": "salesperson/src/components/sheets/ResetControl.test.tsx", "reason": "test"},
    {"path": "salesperson/src/i18n/LanguageChooser.test.tsx", "reason": "test"},
    {"path": "salesperson/src/i18n/LanguageChooser.tsx", "reason": "frontend-ignore"},
    {"path": "salesperson/src/i18n/Provider.tsx", "reason": "frontend-ignore"},
    {"path": "salesperson/src/i18n/config.ts", "reason": "frontend-ignore"},
    {"path": "salesperson/src/i18n/format.test.ts", "reason": "test"},
    {"path": "salesperson/src/i18n/format.ts", "reason": "frontend-ignore"},
    {"path": "salesperson/src/i18n/locales.test.ts", "reason": "test"},
    {"path": "salesperson/src/i18n/useLocale.ts", "reason": "frontend-ignore"},
    {"path": "salesperson/src/layout/Header.test.tsx", "reason": "test"},
    {"path": "salesperson/src/layout/Shell.test.tsx", "reason": "test"},
    {"path": "salesperson/src/session/storage.test.ts", "reason": "test"},
    {"path": "salesperson/src/views/CartPanel.test.tsx", "reason": "test"},
    {"path": "salesperson/src/views/CatalogPanel.test.tsx", "reason": "test"},
    {"path": "salesperson/src/views/ChatView.test.tsx", "reason": "test"},
    {"path": "salesperson/src/views/OrderPanel.test.tsx", "reason": "test"},
    {"path": "salesperson/src/views/PresenterKeyScreen.test.tsx", "reason": "test"},
    {"path": "salesperson/src/views/PresenterResetAllControl.test.tsx", "reason": "test"},
    {"path": "salesperson/src/views/PresenterRoster.test.tsx", "reason": "test"},
    {"path": "salesperson/src/views/ProfilePanel.test.tsx", "reason": "test"},
    {"path": "salesperson/tests/e2e/mobile-shell.spec.ts", "reason": "test"},
    {"path": "salesperson/tests/e2e/presenter.spec.ts", "reason": "test"},
]


@delta("p1-08-skipped-files", "0.4")
def _skipped_files(header, records):
    """P1-08: a JS unit's header gains `skippedFiles`; header-only, the stored bundle already lacks these files' records."""
    if header["unit"]["rootPath"] != "salesperson":
        return 0
    assert not {e["path"] for e in P1_08_SKIPPED_SP} & {r["path"] for r in records if r["type"] == "file"}
    header["skippedFiles"] = [dict(e) for e in P1_08_SKIPPED_SP]
    return len(P1_08_SKIPPED_SP)


# P1-41 (c2): an imported name resolves through its import, not to a same-named "constant of this file" (Joern lists the import-bound
# name among the importing file's module-level assignments, so the stored `ref` named the importing file, a constant with no
# record). The valueSets that change, id -> expected `values`, are listed BY HAND in `tests/data/p1_41_import_precedence.json`
# (derived once from a real run, each checked by eye: only an element naming the importing file changes, to the import's target
# `ref` or to an `importRef`; no record is added or removed, so no count moves). A valueSet that changes and is not listed fails.
P1_41_IMPORT_PRECEDENCE = json.loads((CPG / "tests" / "data" / "p1_41_import_precedence.json").read_text(encoding="utf-8"))
P1_41_COUNTS = {"fcs": 76, "sp": 3}


@delta("p1-41-import-before-constant", "0.5")
def _import_before_constant(header, records):
    want = P1_41_IMPORT_PRECEDENCE[UNIT_TAG_OF_ROOT[header["unit"]["rootPath"]]]
    touched = 0
    for r in records:
        if r["type"] == "valueSet" and r["id"] in want:
            assert r["values"] != want[r["id"]]
            r["values"] = want[r["id"]]
            touched += 1
    assert touched == len(want) == P1_41_COUNTS[UNIT_TAG_OF_ROOT[header["unit"]["rootPath"]]]
    return touched


# P1-05 5d: a name-linked call is not `confirmed` (assemble's type rule and `self` rule, NORMALIZER_VERSION "0.6"). The stored bundles hold
# no `rt`, so the expected edit is written here from independent inputs and `assemble` is fed the stored facts plus the hand-listed `rt`
# table (`tests/overlay_5d.py`), never the other way round. Per listed call site (`p1_05_5d_extras.json`): the type-backed targets
# (`kept`, derived from the Joern fullNames) survive; with none, the site keeps exactly the one class method among its stored targets
# when the `self` rule applies (the `ast` reference over the oracle's stage tree below, plus the hand-listed `selfByJoernNarrowing`),
# else it is `unresolved`/`possible`/`name-match` with no targets. `salesperson` has no such site (`jssrc2cpg` demoted nothing there)
# but gains the constant records of its export-block constants (`constRecords`), `counts.constant` raised by their number.
PINNED_ORACLE = "defca60ff16534aa12a74e35856de9be98bfc710"           # the commit the oracle's stage trees were built at


def reference_self_sites() -> collections.Counter:
    """The independent reference for the `self` rule, read from the oracle's stage tree (the sources at `PINNED_ORACLE`), never the
    working tree: a source edited since the oracle was built must not move it."""
    return overlay_5d.ast_self_sites(need(S14A / "fcs" / "stage"), FCS_ROOT)


def expected_site_edits(records: list[dict], tag: str) -> dict[str, list[str] | None]:
    """call-site id -> the expected final target symbol ids ([] = demoted), for every listed site, from the stored ``records``."""
    syms = {r["id"]: r for r in records if r["type"] == "symbol"}
    calls = {r["id"]: r for r in records if r["type"] == "callsite"}
    strict = collections.Counter(reference_self_sites()) if tag == "fcs" else collections.Counter()
    known = set(overlay_5d.extras()["units"][tag].get("selfByJoernNarrowing", []))
    out = {}
    for s in sorted(overlay_5d.sites(tag), key=lambda s: s["site"]):
        rec = calls[s["site"]]
        stored = [t["symbolId"] for t in rec["targets"]]
        key = (syms[rec["symbolId"]]["path"], rec["line"], rec["calleeName"])
        is_self = s["site"] in known
        if not is_self and strict[key] > 0:
            strict[key] -= 1
            is_self = True
        if s["kept"]:
            out[s["site"]] = list(s["kept"])
        elif is_self:
            cm = [t for t in stored if syms[t]["kind"] == "method" and syms[syms[t]["parentId"]]["kind"] == "type"]
            assert len(cm) == 1, (s["site"], cm)
            out[s["site"]] = cm
        else:
            out[s["site"]] = []
    return out


@delta("p1-05-5d-name-linked", "0.6")
def _name_linked(header, records):
    tag = UNIT_TAG_OF_ROOT[header["unit"]["rootPath"]]
    touched = 0
    calls = {r["id"]: r for r in records if r["type"] == "callsite"}
    for site, final in expected_site_edits(records, tag).items():
        rec = calls[site]
        if [t["symbolId"] for t in rec["targets"]] == final:
            continue                                   # a `self`-rule site whose one stored target is the class method: unchanged
        if final:
            rec["targets"] = [t for t in rec["targets"] if t["symbolId"] in final]
        else:
            rec.update(resolution="unresolved", targets=[], status="possible", reason="name-match")
        touched += 1
    new = overlay_5d.extras()["units"][tag]["constRecords"]
    if new:
        assert not {r["id"] for r in new} & {r["id"] for r in records}
        records.extend(copy.deepcopy(new))
        records.sort(key=bundle._sort_key)
        header["counts"]["constant"] += len(new)
        touched += len(new)
    return touched


def apply_deltas(lines: list[bytes], deltas=None) -> tuple[list[bytes], dict[str, int]]:
    """``lines`` (a stored bundle, header first) through ``deltas`` (default: ``DELTAS``); returns the expected canonical
    lines and, per delta name, how many records it touched."""
    header, *records = [json.loads(x) for x in lines]
    touched = {}
    for name, version, fn in (DELTAS if deltas is None else deltas):
        touched[name] = fn(header, records)
        header["tools"]["normalizerVersion"] = version
        recount = collections.Counter(r["type"] for r in records)
        assert header["counts"] == {t: recount.get(t, 0) for t in header["counts"]}, f"delta {name!r} left the counts wrong"
    return [bundle.canonical_line(r).rstrip(b"\n") for r in [header, *records]], touched


def expected_lines(tag: str) -> list[bytes]:
    return apply_deltas(stored_lines(tag))[0]


def expected_bytes(tag: str) -> bytes:
    return b"".join(x + b"\n" for x in expected_lines(tag))


@pytest.fixture(scope="module", params=["fcs", "sp"])
def written(request, tmp_path_factory):
    """(tag, written bytes, written path) of the unit's bundle, built by `build_bundle` through `ccg.atrest`, with the stored
    header's own `builtAt`, audit ids and the slice's script, so that nothing in it is exempt."""
    tag = request.param
    root, lang, bundle_file, _ = UNITS[tag]
    need(bundle_file)
    header = json.loads(stored_lines(tag)[0])
    work = tmp_path_factory.mktemp("work")
    os.chmod(work, 0o700)
    raw = raw_of(tag, header["builtCommit"], header["sandboxAuditIds"])
    path = assemble_bundle.build_bundle(raw.unit, raw, REPO, header["builtCommit"], system="slice", root=work,
                                        built_at=header["builtAt"], extract_script=SLICE_SCRIPT)
    return tag, path.read_bytes(), path


@pytest.mark.slice
@pytest.mark.parametrize("tag", ["fcs", "sp"])
def test_the_stored_bundles_are_the_artifacts_recorded_with_their_sha256(tag):
    assert hashlib.sha256(need(UNITS[tag][2]).read_bytes()).hexdigest() == UNITS[tag][3]


@pytest.mark.slice
def test_whole_bundle_is_byte_identical_to_the_stored_one(written):
    """Definition 1 (strict): every byte, the header included; the inputs the stamps come from are the stored header's."""
    tag, data, _ = written
    assert data == expected_bytes(tag)


@pytest.mark.slice
def test_whole_bundle_is_identical_with_the_ported_extract_script_but_for_its_hash(written, tmp_path):
    """Definition 2: with the production `extract.sc` the only difference is `tools.extractScriptHash`, and it is that file's sha256."""
    tag, _, _ = written
    header = json.loads(stored_lines(tag)[0])
    raw = raw_of(tag, header["builtCommit"], header["sandboxAuditIds"])
    os.chmod(tmp_path, 0o700)
    script = tmp_path / "extract.sc"                          # a snapshot: the build hashes this copy, so an edit of the shipped
    script.write_bytes(run.EXTRACT_SC.read_bytes())           # script while the test runs cannot split the hash from what was built
    path = assemble_bundle.build_bundle(raw.unit, raw, REPO, header["builtCommit"], system="slice", root=tmp_path,
                                        built_at=header["builtAt"], extract_script=script)
    got, want = path.read_bytes().split(b"\n")[:-1], expected_lines(tag)
    gh, wh = json.loads(got[0]), json.loads(want[0])
    assert gh["tools"].pop("extractScriptHash") == hashlib.sha256(script.read_bytes()).hexdigest()
    assert wh["tools"].pop("extractScriptHash") == hashlib.sha256(SLICE_SCRIPT.read_bytes()).hexdigest()
    assert gh == wh
    assert got[1:] == want[1:]


@pytest.mark.slice
def test_whole_bundle_records_are_the_same_multiset_per_type(written):
    """Definition 3 (weakest): per record type, the same multiset of canonical lines, and the same header counts."""
    tag, data, _ = written
    def by_type(lines):
        out = collections.defaultdict(collections.Counter)
        for ln in lines[1:]:
            out[json.loads(ln)["type"]][ln] += 1
        return out
    got, want = by_type(data.split(b"\n")[:-1]), by_type(expected_lines(tag))
    assert {t: sum(c.values()) for t, c in got.items()} == {t: sum(c.values()) for t, c in want.items()}
    assert got == want
    assert json.loads(data.split(b"\n")[0])["counts"] == json.loads(expected_lines(tag)[0])["counts"]


@pytest.mark.slice
def test_the_written_bundle_is_private_0600_in_a_0700_directory(written):
    _, _, path = written
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert path.parent.name == "slice" and path.parent.parent.name == "bundles"


@pytest.mark.slice
def test_no_delta_at_all_reproduces_the_stored_bundles_bytes_through_the_same_serializer():
    for tag in UNITS:
        assert apply_deltas(stored_lines(tag), [])[0] == stored_lines(tag)


@pytest.mark.slice
def test_the_lambda_decorator_delta_removes_exactly_3_records_all_of_falkor_chat_server_and_none_of_salesperson():
    fcs, fcs_touched = apply_deltas(stored_lines("fcs"))
    sp, sp_touched = apply_deltas(stored_lines("sp"))
    assert (fcs_touched["drop-lambda-decorators"], sp_touched["drop-lambda-decorators"]) == (3, 0)
    assert (fcs_touched["p1-06-mask-cond-code"], sp_touched["p1-06-mask-cond-code"]) == (41, 2)
    assert (fcs_touched["p1-08-skipped-files"], sp_touched["p1-08-skipped-files"]) == (0, 33)
    assert (fcs_touched["p1-05-5d-name-linked"], sp_touched["p1-05-5d-name-linked"]) == (1216, 3)
    assert "skippedFiles" not in json.loads(fcs[0]) and len(json.loads(sp[0])["skippedFiles"]) == 33
    assert json.loads(fcs[0])["counts"]["decorator"] == 210 == json.loads(stored_lines("fcs")[0])["counts"]["decorator"] - 3
    assert sum(json.loads(x)["type"] == "decorator" for x in fcs[1:]) == 210
    assert json.loads(sp[0])["counts"]["decorator"] == 0


@pytest.mark.slice
def test_each_delta_pins_its_version_and_the_last_one_is_the_normalizers_current_version():
    versions = [v for _, v, _ in DELTAS]
    assert len(set(versions)) == len(versions) and len({n for n, _, _ in DELTAS}) == len(DELTAS)
    assert versions[-1] == assemble_bundle.NORMALIZER_VERSION
    for tag in UNITS:
        assert json.loads(stored_lines(tag)[0])["tools"]["normalizerVersion"] == "0.1"
        assert json.loads(expected_lines(tag)[0])["tools"]["normalizerVersion"] == assemble_bundle.NORMALIZER_VERSION


@pytest.mark.slice
def test_a_delta_that_leaves_the_counts_wrong_is_refused_by_the_mechanism():
    lazy = [("lazy", "9.9", lambda header, records: records.pop() and 1)]
    with pytest.raises(AssertionError, match="counts wrong"):
        apply_deltas(stored_lines("sp"), lazy)


@pytest.mark.slice
def test_ownedregions_changes_nothing_on_the_slice_bundles(written):
    """5c (c): no P1 frontend sets the slot, so neither the stored nor the written bundle carries the field."""
    tag, data, _ = written
    for raw in (*stored_lines(tag), *data.split(b"\n")[:-1]):
        assert b"ownedRegions" not in raw


# -- 5d: the name-link rule against independent references --------------------------------------------------------------------
@pytest.fixture(scope="module")
def fcs_built(tmp_path_factory):
    """(written records by id, the assemble report) of `falkor-chat/server` built from the overlaid stored facts."""
    header = json.loads(stored_lines("fcs")[0])
    work = tmp_path_factory.mktemp("work5d")
    os.chmod(work, 0o700)
    raw = raw_of("fcs", header["builtCommit"], header["sandboxAuditIds"])
    report: dict = {}
    path = assemble_bundle.build_bundle(raw.unit, raw, REPO, header["builtCommit"], system="slice", root=work,
                                        built_at=header["builtAt"], extract_script=SLICE_SCRIPT, report=report)
    recs = [json.loads(x) for x in path.read_bytes().split(b"\n")[:-1]][1:]
    return recs, report


@pytest.mark.slice
def test_the_reference_reads_the_oracles_stage_tree_pinned_at_the_oracle_commit_not_the_working_tree():
    """Done-condition 1: the `ast` reference's sources are the stage the oracle was built from, and that stage is byte-identical to the
    git blobs at the oracle commit (so an edit of the working tree since cannot move the reference)."""
    stage = need(S14A / "fcs" / "stage")
    if subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", PINNED_ORACLE + "^{commit}"], capture_output=True).returncode != 0:
        pytest.skip(f"the oracle commit {PINNED_ORACLE} is not in this clone (UNRUN, not green)")
    files = sorted(stage.rglob("*.py"))
    assert len(files) > 50
    for f in files:
        blob = subprocess.run(["git", "-C", str(REPO), "show", f"{PINNED_ORACLE}:{FCS_ROOT}/{f.relative_to(stage).as_posix()}"],
                              capture_output=True, check=True).stdout
        assert f.read_bytes() == blob, f
    seen = {}
    real = overlay_5d.ast_self_sites
    overlay_5d.ast_self_sites = lambda st, root, **kw: seen.setdefault("stage", Path(st)) and real(st, root, **kw)
    try:
        reference_self_sites()
    finally:
        overlay_5d.ast_self_sites = real
    assert seen["stage"] == stage and seen["stage"] != REPO / FCS_ROOT          # the oracle's stage, not the working tree


@pytest.mark.slice
def test_the_self_rule_confirms_the_ast_references_sites_and_lists_every_further_one(fcs_built):
    """Done-condition 2. The strict `ast` reference (a `self.m()` in a method of the class that defines `m`, no other class of the unit
    defining `m`) must be a subset of what `assemble` confirms by the `self` rule; every further confirmed site is LISTED, and the
    list is the hand-written `selfByJoernNarrowing` (Joern's own type recovery had already narrowed the site to the enclosing class's
    method, so the in-set held one class method although another class names it). Nothing is excluded silently: a disagreement in
    either direction fails with its ids."""
    recs, report = fcs_built
    syms = {r["id"]: r for r in recs if r["type"] == "symbol"}
    calls = {r["id"]: r for r in recs if r["type"] == "callsite"}
    confirmed = sorted(s["site"] for s in overlay_5d.sites("fcs") if not s["kept"] and calls[s["site"]]["status"] == "confirmed")
    key = lambda cid: (syms[calls[cid]["symbolId"]]["path"], calls[cid]["line"], calls[cid]["calleeName"])
    strict, loose = reference_self_sites(), overlay_5d.ast_self_sites(S14A / "fcs" / "stage", FCS_ROOT, strict=False)
    left = collections.Counter(strict)
    further = []
    for cid in confirmed:
        k = key(cid)
        if left[k] > 0:
            left[k] -= 1
        else:
            further.append(cid)
    assert +left == collections.Counter(), f"the reference lists self sites that assemble did not confirm: {sorted(+left)}"
    assert further == sorted(overlay_5d.extras()["units"]["fcs"]["selfByJoernNarrowing"]), f"unlisted disagreement: {further}"
    assert not (collections.Counter(map(key, confirmed)) - loose), "a confirmed site lies outside even the loose reference"
    assert report["nameLinkCounts"]["selfResolved"] == len(confirmed)
    assert (len(confirmed), sum(strict.values()), len(further)) == (264, 147, 117)


@pytest.mark.slice
def test_every_listed_site_has_exactly_the_expected_nameLinked_entry_and_no_other_site_has_one(fcs_built):
    recs, report = fcs_built
    stored = {r["id"]: r for r in map(json.loads, stored_lines("fcs")[1:]) if r["type"] == "callsite"}
    want = {}
    for site, final in expected_site_edits([json.loads(x) for x in stored_lines("fcs")[1:]], "fcs").items():
        dropped = sorted(t["symbolId"] for t in stored[site]["targets"] if t["symbolId"] not in final)
        if dropped:
            want[site] = dropped
    got = {cid: sorted(t["symbolId"] for t in ts) for cid, ts in report["nameLinked"].items()}
    assert got == want
    assert report["nameLinkCounts"]["nameLinkedDemoted"] == len(want) == 1216


@pytest.mark.slice
def test_without_the_overlay_the_stored_facts_demote_nothing(tmp_path):
    """A facts file without `rt` is never demoted (an old facts file leaves every site as it was): the stored bundle, byte for byte."""
    header = json.loads(stored_lines("fcs")[0])
    cpg = need(S14A / "fcs" / "cpg")
    raw = run.RawFacts(Unit(ids.unit_id(REPO_ID, FCS_ROOT, "python"), REPO_ID, FCS_ROOT, "python"), header["builtCommit"],
                       sorted(cpg.glob("*.facts.jsonl"))[0], tuple(header["sandboxAuditIds"]), need(S14A / "fcs" / "stage"), cpg,
                       None, cpg / "none", 0)
    os.chmod(tmp_path, 0o700)
    report: dict = {}
    path = assemble_bundle.build_bundle(raw.unit, raw, REPO, header["builtCommit"], system="slice", root=tmp_path,
                                        built_at=header["builtAt"], extract_script=SLICE_SCRIPT, report=report)
    got = path.read_bytes().split(b"\n")[:-1]
    deltas_before_5d = [d for d in DELTAS if d[1] != "0.6"]
    want = apply_deltas(stored_lines("fcs"), deltas_before_5d)[0]
    gh, wh = json.loads(got[0]), json.loads(want[0])
    gh["tools"].pop("normalizerVersion"), wh["tools"].pop("normalizerVersion")
    assert gh == wh and got[1:] == want[1:]
    assert report["nameLinked"] == {} and report["nameLinkCounts"] == {"nameLinkedDemoted": 0, "selfResolved": 0}


# -- 5b live tier: two real Joern runs of one stage ------------------------------------------------------------------
PINNED = "defca60ff16534aa12a74e35856de9be98bfc710"


def _without_exempt(data: bytes) -> list[bytes]:
    lines = data.split(b"\n")[:-1]
    header = json.loads(lines[0])
    for k in EXEMPT:
        header.pop(k)
    return [json.dumps(header, sort_keys=True).encode()] + lines[1:]


#: P1-09 (projection fix, `extract.sc`) adds def->use records the slice's stored `salesperson` bundle never had: 112 `data` and 5
#: `controlledBy`. They are listed BY HAND, in bundle order, as canonical lines in `tests/data/p1_09_sp_projection_extras.jsonl`
#: (derived once from a real live run; the allow-list pins every field of each record, not just the id). The live bundle must be the
#: stored one plus exactly these, so any other record changing, going missing, or appearing makes the live test fail.
P1_09_SP_EXTRAS = CPG / "tests" / "data" / "p1_09_sp_projection_extras.jsonl"
P1_09_SP_EXTRA_COUNTS = {"data": 112, "controlledBy": 5}
#: the golden hash of the shipped `extract.sc` this allow-list was derived under: a further script change must be decided HERE.
EXTRACT_SC_GOLDEN = CPG / "tests" / "golden" / "projection" / "extract.sc.sha256"


def assert_is_stored_plus_p1_09_extras(data: bytes, extras_file: Path = P1_09_SP_EXTRAS, counts: dict | None = None,
                                       expected: bytes | None = None) -> None:
    """``data`` (a live `salesperson` bundle, header `sandboxAuditIds` exempt) equals the expected stored bundle plus exactly the
    allow-listed records (in bundle order), and its header equals the expected one with the `counts` raised by ``counts``."""
    counts = P1_09_SP_EXTRA_COUNTS if counts is None else counts
    extras = extras_file.read_bytes().split(b"\n")[:-1]
    assert len(set(extras)) == len(extras)
    assert collections.Counter(json.loads(x)["type"] for x in extras) == counts
    got = _without_exempt(data)
    want = _without_exempt(expected_bytes("sp") if expected is None else expected)
    listed = set(extras)
    assert [x for x in got[1:] if x in listed] == [x for x in extras]          # every allow-listed record, once, in order
    assert [x for x in got[1:] if x not in listed] == want[1:]                 # and nothing else differs
    want_header = json.loads(want[0])
    for kind, n in counts.items():
        want_header["counts"][kind] += n
    assert json.loads(got[0]) == want_header


@pytest.mark.slice
def test_extract_sc_is_the_one_the_p1_09_allow_list_was_derived_under():
    assert hashlib.sha256(run.EXTRACT_SC.read_bytes()).hexdigest() == EXTRACT_SC_GOLDEN.read_text(encoding="utf-8").strip()


@pytest.mark.slice
@pytest.mark.live
def test_two_real_joern_runs_of_one_unit_give_one_bundle_and_it_is_the_stored_one(tmp_path, monkeypatch):
    """The raw `d`/`cb` facts differ in file order between Joern runs (they agree as multisets); the bundle must not. The unit is
    `salesperson` (JS, the smaller one): two full extractions through the runner at the slice's pinned commit, each normalized
    with the stored header's `builtAt` and the slice's script, then compared with each other and with the stored bundle plus the
    hand-listed P1-09 projection records (`P1_09_SP_EXTRAS`), header `sandboxAuditIds` (new audit ids every run) exempt."""
    assert hashlib.sha256(run.EXTRACT_SC.read_bytes()).hexdigest() == EXTRACT_SC_GOLDEN.read_text(encoding="utf-8").strip(), \
        "extract.sc changed: re-derive P1_09_SP_EXTRAS (and decide the stored-bundle delta) before this test means anything"
    monkeypatch.setenv("CCG_AUDIT_DIR", str(tmp_path / "audit"))     # never the real cpg/.audit/sandbox.jsonl
    stored_header = json.loads(stored_lines("sp")[0])
    unit = Unit(ids.unit_id(REPO_ID, SP_ROOT, "javascript"), REPO_ID, SP_ROOT, "javascript")
    area = tmp_path / ".ccg-work"
    area.mkdir(mode=0o700)
    raws = [run.extract_unit(REPO, unit, [unit], PINNED, area / f"run{i}", area=area) for i in (1, 2)]
    facts = [core.load_facts(r.facts) for r in raws]
    for kind in ("d", "cb"):                                          # same multiset, whatever the order
        key = lambda f: json.dumps(f, sort_keys=True)
        assert sorted(map(key, facts[0][kind])) == sorted(map(key, facts[1][kind]))
    out = []
    for i, raw in enumerate(raws, start=1):
        path = assemble_bundle.build_bundle(unit, raw, REPO, PINNED, system="live", root=area, build=f"run{i}",
                                            built_at=stored_header["builtAt"], extract_script=SLICE_SCRIPT)
        out.append(path.read_bytes())
    assert raws[0].sandbox_audit_ids != raws[1].sandbox_audit_ids          # every run has its own audit ids ...
    assert _without_exempt(out[0]) == _without_exempt(out[1])             # ... and they are all that tells the bundles apart
    for data in out:
        assert_is_stored_plus_p1_09_extras(data)


@pytest.mark.slice
def test_the_stored_files_are_the_staged_sources_less_the_hand_listed_33():
    """Independent of `skipped_files`: staged JS/TS sources (non-empty, not symlinks) minus the hand-written list = the stored `file` set."""
    stage = need(S14A / "sp" / "stage")
    exts = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")
    staged = set()
    for d, _, names in os.walk(stage):
        for n in names:
            p = Path(d, n)
            if n.endswith(exts) and not p.is_symlink() and p.stat().st_size > 0:
                staged.add("salesperson/" + p.relative_to(stage).as_posix())
    listed = {e["path"] for e in P1_08_SKIPPED_SP}
    stored_files = {json.loads(x)["path"] for x in stored_lines("sp") if json.loads(x).get("type") == "file"}
    assert (len(staged), len(listed), len(stored_files)) == (74, 33, 41)
    assert listed <= staged and staged - listed == stored_files
    assert sorted(listed, key=lambda s: s.encode()) == [e["path"] for e in P1_08_SKIPPED_SP]
    assert sum(e["reason"] == "test" for e in P1_08_SKIPPED_SP) == 28
