#!/usr/bin/env python3
"""check_content_loss_selftest.py — mutation test for `check_content_loss.py`.

Not a pytest suite, for the same reason `flag_split_candidates_selftest.py` isn't one (see that
file's own docstring: no pytest package anywhere under `claude/`, this is a single standalone
script, not a library). A plain assertion script, this repo's own house style for a mechanical
check.

WHAT THIS CHECKS
-----------------
Builds a clean baseline directly from `fixtures/content_loss_test_kb.md` (claim texts are sliced
straight out of the real fixture body, byte-for-byte, so the baseline is correct by construction,
not by re-typing) covering the three real split shapes: a single-claim heading, a bulleted heading
(marker stripped), and a paragraph-boundary two-claim split. Confirms the checker reports all
three clean. Then, deliberately, four real defect cases — the exact four named in the migration
brief:

  (a) drop one bullet's claim entirely -> its content must surface as an UNACCOUNTED gap.
  (b) truncate a claim's text mid-sentence -> the chopped-off remainder must surface as a gap.
  (c) substitute an ASCII lookalike for a unicode character (the real `statistical-method-
      techniques.md` fidelity slip: κ/≤/α) -> the claim must be reported NOT_FOUND, not silently
      accepted as "close enough".
  (d) duplicate a bullet's claim text into a second claim under the same heading -> both claims
      locating to the identical source span must surface as a DUPLICATE overlap.

A checker that stays clean on all four mutants proves nothing (per this team's standing
mutation-testing practice) -- each one is asserted to actually flip the checker's verdict, and the
specific finding kind is asserted too, not just "not clean".

ADDED PER `analyst`'S REVIEW (`claude/docs/reviews/kb-content-loss-checker.md`, commit `a0dfeb4`
was reviewed and came back "needs changes"; this revision addresses all three majors plus the
cheap minor/nit)
------------------------------------------------------------------------------------------------
  - **Major 1 (non-adjacent nested overlap)**: the original adjacent-pairs-only overlap check
    missed a wide span nesting two non-adjacent narrower ones (the review's own Appendix A
    construction). `check_nonadjacent_nested_overlap` reproduces it and asserts BOTH overlapping
    pairs are reported by document id, not just that `report.overlaps` is non-empty (the review's
    own callout: the old mutation-test style would have passed even with the bug present).
  - **Major 2 (whitespace normalization had zero coverage)**: `check_rewrapped_claim_still_locates`
    is the positive case (same words, re-wrapped at a different column -> still `found`, clean);
    `check_rewrapped_and_word_dropped_still_not_found` is the companion negative case (re-wrapped
    AND missing one word -> must still be `not_found`, proving the match stays exact rather than
    becoming accidentally permissive).
  - **Major 3 (CLI/manifest layer untested)**: `check_missing_document_status` covers the
    `missing_document` outcome at the `check_partition` level (no CLI plumbing needed, per the
    review's own "at minimum"); `check_cli_primary_manifest_shape`,
    `check_cli_manifest_flat_key_shape`, `check_cli_documents_manual_mode`, and
    `check_cli_missing_document_via_dump` drive `check_content_loss.main()` end-to-end with `argv`
    against small synthetic manifest/source/dump files under a temp directory (no FalkorDB
    needed); `check_cli_live_missing_venv_errors_cleanly` smoke-tests the `--live` "binary
    missing" error path by pointing `CYPHER_MCP_VENV_PYTHON` at a path that doesn't exist.
  - **Minor (empty claim text)**: `check_empty_text_status` confirms an empty/whitespace-only
    claim text gets its own `empty_text` outcome rather than a free "found" at a zero-length span.
  - **Nit (mutually exclusive args)**: `check_documents_and_flat_key_are_mutually_exclusive`
    confirms argparse itself refuses `--documents` and `--manifest-flat-key` together.

Run: `python3 claude/cobb/scripts/check_content_loss_selftest.py`
Exit 0 and "ALL PASS" on success; exit 1 and the failing assertion's diagnosis otherwise.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_content_loss as ccl  # noqa: E402
import flag_split_candidates as fsc  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "content_loss_test_kb.md"

SINGLE_HEADING = "A single-claim heading migrates as one verbatim claim"
BULLET_HEADING = "Three independent facts about connection pooling"
PARAGRAPH_HEADING = "A worked incident, told in two parts"

BULLET_SPLIT_RE = re.compile(r"(?m)^- ")


def load_sections() -> dict[str, str]:
    return dict(fsc.parse_sections(FIXTURE.read_text(encoding="utf-8")))


def bullets_of(body: str) -> list[str]:
    """Split a bulleted section body into its bullets' content, marker stripped -- the same
    transformation a real bullet-per-claim migration performs, so these are guaranteed
    byte-exact substrings of `body` once the leading `- ` is gone."""
    parts = BULLET_SPLIT_RE.split(body)
    return [p.rstrip("\n") for p in parts[1:]]  # parts[0] is the pre-first-bullet material


def build_baseline() -> tuple[dict[str, str], dict[str, list[ccl.Claim]], dict[str, dict]]:
    """Returns (sections, claims_by_heading, store) for a fully clean, correct migration."""
    sections = load_sections()
    store: dict[str, dict] = {}
    claims_by_heading: dict[str, list[ccl.Claim]] = {}

    # Single-claim heading: the whole body, verbatim.
    single_text = sections[SINGLE_HEADING].strip()
    store["single-1"] = {"title": SINGLE_HEADING, "text": single_text}
    claims_by_heading[SINGLE_HEADING] = [ccl.Claim(document_id="single-1", title=SINGLE_HEADING)]

    # Bulleted heading: one claim per bullet, marker stripped.
    bullets = bullets_of(sections[BULLET_HEADING])
    assert len(bullets) == 3, f"fixture drifted -- expected 3 bullets, got {len(bullets)}"
    bullet_claims = []
    for i, bullet_text in enumerate(bullets, start=1):
        doc_id = f"bullet-{i}"
        store[doc_id] = {"title": f"{BULLET_HEADING} — bullet {i}", "text": bullet_text}
        bullet_claims.append(ccl.Claim(document_id=doc_id, title=store[doc_id]["title"]))
    claims_by_heading[BULLET_HEADING] = bullet_claims

    # Paragraph-boundary heading: two claims, each one paragraph.
    body = sections[PARAGRAPH_HEADING]
    para_a, para_b = body.strip().split("\n\n", 1)
    store["para-a"] = {"title": f"{PARAGRAPH_HEADING} — part 1", "text": para_a.strip()}
    store["para-b"] = {"title": f"{PARAGRAPH_HEADING} — part 2", "text": para_b.strip()}
    claims_by_heading[PARAGRAPH_HEADING] = [
        ccl.Claim(document_id="para-a", title=store["para-a"]["title"]),
        ccl.Claim(document_id="para-b", title=store["para-b"]["title"]),
    ]

    return sections, claims_by_heading, store


def check_baseline_clean() -> None:
    sections, claims_by_heading, store = build_baseline()
    fetch = store.get
    for heading, claims in claims_by_heading.items():
        report = ccl.check_partition(sections[heading], claims, fetch)
        assert report.clean, (
            f"baseline (uncorrupted) heading {heading!r} was NOT reported clean:\n"
            f"{ccl.render_report(heading, report)}"
        )


def check_mutation_dropped_bullet() -> None:
    """(a) Drop one bullet's claim entirely -- its content must surface as an UNACCOUNTED gap."""
    sections, claims_by_heading, store = build_baseline()
    fetch = store.get
    body = sections[BULLET_HEADING]
    claims = claims_by_heading[BULLET_HEADING]
    dropped = claims[1]  # the middle bullet
    remaining = [c for c in claims if c.document_id != dropped.document_id]

    report = ccl.check_partition(body, remaining, fetch)
    assert not report.clean, "dropping a bullet's claim should NOT report clean"
    assert report.gaps, f"expected an UNACCOUNTED gap for the dropped bullet, got none: {report}"
    dropped_text = store[dropped.document_id]["text"]
    # The gap must actually correspond to (be a substring relationship with) the dropped bullet's
    # own content, not some unrelated whitespace artifact.
    gap_texts = [g[2] for g in report.gaps]
    assert any(dropped_text[:30] in g or g.strip() in dropped_text for g in gap_texts), (
        f"gap(s) found but none match the dropped bullet's content.\n"
        f"dropped: {dropped_text[:60]!r}\ngaps: {[g[:60] for g in gap_texts]}"
    )
    # The two remaining claims must still be found cleanly -- the mutation should isolate the
    # dropped bullet, not collaterally break its siblings.
    for outcome in report.outcomes:
        assert outcome.status == "found", f"un-dropped claim {outcome.claim.document_id} broke too"


def check_mutation_truncated_claim() -> None:
    """(b) Truncate a claim's text mid-sentence -- the chopped-off remainder must surface as a
    gap, and the truncated claim itself must still be locatable (it's a real prefix)."""
    sections, claims_by_heading, store = build_baseline()
    body = sections[BULLET_HEADING]
    claims = claims_by_heading[BULLET_HEADING]
    target = claims[0]
    full_text = store[target.document_id]["text"]
    cut_point = full_text.index("validates a connection's liveness")  # mid-sentence, deliberate
    truncated_text = full_text[:cut_point].rstrip()
    assert truncated_text != full_text, "fixture drifted -- truncation produced no change"

    mutated_store = dict(store)
    mutated_store[target.document_id] = {**store[target.document_id], "text": truncated_text}
    fetch = mutated_store.get

    report = ccl.check_partition(body, claims, fetch)
    assert not report.clean, "truncating a claim mid-sentence should NOT report clean"
    assert report.gaps, f"expected a gap for the truncated remainder, got none: {report}"
    remainder = full_text[cut_point:]
    gap_texts = [g[2] for g in report.gaps]
    assert any(remainder[:25] in g for g in gap_texts), (
        f"no gap matches the truncated-off remainder.\nremainder: {remainder[:60]!r}\n"
        f"gaps: {[g[:60] for g in gap_texts]}"
    )
    # The truncated claim is still found (it's a genuine prefix of the real text) -- the
    # truncation shows up as a DOWNSTREAM gap, not as a NOT_FOUND on the claim itself.
    truncated_outcome = next(o for o in report.outcomes if o.claim.document_id == target.document_id)
    assert truncated_outcome.status == "found", (
        "a truncated-but-still-a-real-prefix claim should still locate; the loss should show up "
        "as the downstream gap, which is asserted separately above"
    )


def check_mutation_unicode_substitution() -> None:
    """(c) Substitute an ASCII lookalike for a unicode character (the real `statistical-method-
    techniques.md` fidelity slip: κ/≤/α) -- must be reported NOT_FOUND, never silently accepted."""
    sections, claims_by_heading, store = build_baseline()
    claim = claims_by_heading[SINGLE_HEADING][0]
    original_text = store[claim.document_id]["text"]
    assert "κ" in original_text and "≤" in original_text and "α" in original_text, (
        "fixture drifted -- expected the κ≤α unicode markers in the single-claim heading"
    )
    corrupted_text = original_text.replace("κ", "k").replace("≤", "<=").replace("α", "a")
    assert corrupted_text != original_text

    mutated_store = dict(store)
    mutated_store[claim.document_id] = {**store[claim.document_id], "text": corrupted_text}
    fetch = mutated_store.get

    report = ccl.check_partition(sections[SINGLE_HEADING], [claim], fetch)
    assert not report.clean, "an ASCII-substituted unicode character must NOT report clean"
    outcome = report.outcomes[0]
    assert outcome.status == "not_found", (
        f"expected the corrupted claim to be reported not_found, got {outcome.status!r} -- a "
        "checker that locates a fuzzy near-match here would silently accept the fidelity slip "
        "this mutation reproduces"
    )
    assert outcome.diagnostic, "a not_found claim should carry a diff-anchored diagnostic"


def check_mutation_duplicated_bullet() -> None:
    """(d) Duplicate a bullet's claim text into a second claim under the same heading -- both
    locating to the identical source span must surface as a DUPLICATE overlap."""
    sections, claims_by_heading, store = build_baseline()
    body = sections[BULLET_HEADING]
    claims = list(claims_by_heading[BULLET_HEADING])
    original = claims[0]
    duplicate_id = "bullet-1-duplicate"
    mutated_store = dict(store)
    mutated_store[duplicate_id] = {
        "title": "duplicate of bullet 1",
        "text": store[original.document_id]["text"],
    }
    claims.append(ccl.Claim(document_id=duplicate_id, title="duplicate of bullet 1"))
    fetch = mutated_store.get

    report = ccl.check_partition(body, claims, fetch)
    assert not report.clean, "a duplicated claim should NOT report clean"
    assert report.overlaps, f"expected a DUPLICATE overlap, got none: {report}"
    involved_ids = {o.claim.document_id for pair in report.overlaps for o in pair}
    assert original.document_id in involved_ids and duplicate_id in involved_ids, (
        f"the overlap reported doesn't name the two duplicated claims: {involved_ids}"
    )
    # Every claim (including the duplicate) still individually locates -- the defect is the
    # overlap, not a lookup failure.
    for outcome in report.outcomes:
        assert outcome.status == "found", f"claim {outcome.claim.document_id} unexpectedly broke"


# --------------------------------------------------------------------------------------------
# Major 1 (analyst review): non-adjacent nested overlap
# --------------------------------------------------------------------------------------------


def check_nonadjacent_nested_overlap() -> None:
    """The review's own Appendix A construction: a wide span nesting two narrower, NON-ADJACENT
    (in sort order) ones. The old adjacent-pairs-only comparison reported only (wide, narrow1) and
    never the real (wide, narrow2) overlap. Asserts the full overlapping-pair SET, not just that
    `report.overlaps` is non-empty -- the review's own callout that the old assertion style would
    have passed even with the bug present."""
    body = "A" * 10 + "B" * 20 + "C" * 10 + "D" * 20 + "E" * 40  # len 100
    store = {
        "wide": {"title": "wide", "text": body[0:60]},  # span (0, 60)
        "narrow1": {"title": "narrow1", "text": body[10:30]},  # span (10, 30), nested in wide
        "narrow2": {"title": "narrow2", "text": body[40:60]},  # span (40, 60), nested, NOT
        # adjacent to narrow1 once sorted by start (wide, narrow1, narrow2) -- narrow1 sits
        # between wide and narrow2 in sort order, which is exactly what an adjacent-only
        # comparison misses.
    }
    claims = [ccl.Claim(document_id=k, title=v["title"]) for k, v in store.items()]
    report = ccl.check_partition(body, claims, store.get)

    assert not report.clean, "a nested duplicate should NOT report clean"
    pairs = {frozenset((o1.claim.document_id, o2.claim.document_id)) for o1, o2 in report.overlaps}
    expected = {frozenset(("wide", "narrow1")), frozenset(("wide", "narrow2"))}
    assert pairs == expected, (
        f"expected overlap pairs {expected}, got {pairs} -- the wide<->narrow2 overlap is the "
        "one an adjacent-only comparison misses"
    )
    # narrow1 and narrow2 genuinely don't overlap each other -- confirm that non-pair isn't
    # spuriously reported either (a sanity check on the fix, not just the miss it fixes).
    assert frozenset(("narrow1", "narrow2")) not in pairs


# --------------------------------------------------------------------------------------------
# Major 2 (analyst review): whitespace normalization -- positive and negative coverage
# --------------------------------------------------------------------------------------------


def check_rewrapped_claim_still_locates() -> None:
    """Positive case: a claim built from a real source bullet but re-wrapped at a DIFFERENT
    column than the source's own hard-wrap (same words, different newline placement) must still
    locate cleanly. This is the real defect shape `build_normalized` exists to fix
    (`coordination-techniques.md`, live data) -- without it, this case regresses to NOT_FOUND."""
    sections, claims_by_heading, store = build_baseline()
    body = sections[BULLET_HEADING]
    claims = list(claims_by_heading[BULLET_HEADING])
    target = claims[0]
    original_text = store[target.document_id]["text"]

    one_line = " ".join(original_text.split())
    rewrapped = textwrap.fill(one_line, width=25)  # the fixture wraps at ~95 cols -- a very
    # different column, so this can't pass by accidentally matching the source's own line breaks
    assert rewrapped != original_text, "fixture drifted -- rewrap produced no visible change"
    assert " ".join(rewrapped.split()) == one_line, "rewrap changed the words, not just layout"

    mutated_store = dict(store)
    mutated_store[target.document_id] = {**store[target.document_id], "text": rewrapped}
    report = ccl.check_partition(body, claims, mutated_store.get)

    assert report.clean, (
        f"a claim re-wrapped at a different column (same words) should still locate cleanly:\n"
        f"{ccl.render_report(BULLET_HEADING, report)}"
    )


def check_rewrapped_and_word_dropped_still_not_found() -> None:
    """Negative companion: the SAME re-wrap, but with one word also dropped. Must still report
    NOT_FOUND -- proving whitespace normalization stays exact on real content and doesn't become
    accidentally permissive just because it tolerates layout differences."""
    sections, claims_by_heading, store = build_baseline()
    body = sections[BULLET_HEADING]
    claims = list(claims_by_heading[BULLET_HEADING])
    target = claims[0]
    original_text = store[target.document_id]["text"]

    words = original_text.split()
    dropped_word = words.pop(len(words) // 2)  # remove one word from the middle
    one_line_minus_word = " ".join(words)
    rewrapped_minus_word = textwrap.fill(one_line_minus_word, width=25)
    assert dropped_word not in rewrapped_minus_word.split()

    mutated_store = dict(store)
    mutated_store[target.document_id] = {**store[target.document_id], "text": rewrapped_minus_word}
    report = ccl.check_partition(body, claims, mutated_store.get)

    assert not report.clean, "re-wrapped text with a real word dropped must NOT report clean"
    outcome = next(o for o in report.outcomes if o.claim.document_id == target.document_id)
    assert outcome.status == "not_found", (
        f"expected not_found for the re-wrapped-and-shortened claim, got {outcome.status!r} -- "
        "whitespace normalization must not be permissive enough to paper over a real dropped word"
    )


# --------------------------------------------------------------------------------------------
# Minor (analyst review): empty/whitespace-only claim text
# --------------------------------------------------------------------------------------------


def check_empty_text_status() -> None:
    """An empty/whitespace-only claim `text` must get its own `empty_text` outcome, not a free
    "found" at a bogus zero-length span (the old behavior: `"".find("")` returns 0
    unconditionally)."""
    body = "Some real content here that should be fully claimed by one document."
    report = ccl.check_partition(
        body,
        [ccl.Claim(document_id="empty-claim")],
        lambda k: {"title": "oops", "text": "   \n  "},
    )
    outcome = report.outcomes[0]
    assert outcome.status == "empty_text", f"expected empty_text, got {outcome.status!r}"
    assert not report.clean
    assert outcome.span is None


# --------------------------------------------------------------------------------------------
# Major 3 (analyst review): the CLI/manifest layer
# --------------------------------------------------------------------------------------------

CLI_SOURCE_MD = """## Heading One

Some short body text for heading one, used only by this synthetic CLI self-test.

## Heading Two

Different short body text for heading two, also synthetic, also self-test-only.
"""


def _cli_source_bodies() -> dict[str, str]:
    return dict(fsc.parse_sections(CLI_SOURCE_MD))


def check_missing_document_status() -> None:
    """The `missing_document` outcome, at the `check_partition` level -- no CLI plumbing needed
    (the review's own "at minimum" ask). A fetch that returns None for a real id must be reported
    as its own status, not silently dropped or conflated with `not_found`."""
    body = "Body text that exists in the source but has no matching claim fetched."
    report = ccl.check_partition(
        body, [ccl.Claim(document_id="ghost-id")], lambda k: None
    )
    outcome = report.outcomes[0]
    assert outcome.status == "missing_document", f"expected missing_document, got {outcome.status!r}"
    assert not report.clean
    # The real body content, unclaimed by anything, must still surface as a gap -- a
    # missing_document claim doesn't get to silently cover its own span.
    assert report.gaps, "the unclaimed body should show up as a gap"


def check_cli_primary_manifest_shape() -> None:
    """Drives `check_content_loss.main()` end-to-end (argv, temp files, no FalkorDB) against the
    PRIMARY manifest shape (`files[path]["headings"]`) via `--dump`. Covers `main`,
    `load_manifest`, `claims_from_manifest_file_entry`, and `make_dump_fetcher` together."""
    bodies = _cli_source_bodies()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "source.md"
        source.write_text(CLI_SOURCE_MD, encoding="utf-8")
        source_key = str(source)

        manifest = {
            "files": {
                source_key: {
                    "headings": [
                        {
                            "heading": "Heading One",
                            "claims": [{"title": "Heading One", "documentId": "cli-1"}],
                        },
                        {
                            "heading": "Heading Two",
                            "claims": [{"title": "Heading Two", "documentId": "cli-2"}],
                        },
                    ]
                }
            }
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        dump = {
            "cli-1": {"title": "Heading One", "text": bodies["Heading One"].strip()},
            "cli-2": {"title": "Heading Two", "text": bodies["Heading Two"].strip()},
        }
        dump_path = tmp_path / "dump.json"
        dump_path.write_text(json.dumps(dump), encoding="utf-8")

        rc = ccl.main([str(source), "--manifest", str(manifest_path), "--dump", str(dump_path)])
        assert rc == 0, f"expected exit 0 for a clean primary-shape run, got {rc}"


def check_cli_manifest_flat_key_shape() -> None:
    """Same synthetic file, but through the flat-key fallback -- covers
    `claims_from_manifest_flat_key` and the whole-file `combined_body` path end-to-end."""
    bodies = _cli_source_bodies()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "source.md"
        source.write_text(CLI_SOURCE_MD, encoding="utf-8")

        manifest = {"_cli_flat_test": {"documentIds": ["cli-1", "cli-2"]}}
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        dump = {
            "cli-1": {"title": "Heading One", "text": bodies["Heading One"].strip()},
            "cli-2": {"title": "Heading Two", "text": bodies["Heading Two"].strip()},
        }
        dump_path = tmp_path / "dump.json"
        dump_path.write_text(json.dumps(dump), encoding="utf-8")

        rc = ccl.main(
            [
                str(source),
                "--manifest",
                str(manifest_path),
                "--manifest-flat-key",
                "_cli_flat_test",
                "--dump",
                str(dump_path),
            ]
        )
        assert rc == 0, f"expected exit 0 for a clean flat-key run, got {rc}"


def check_cli_documents_manual_mode() -> None:
    """`--documents` manual mode never touches the manifest at all -- confirm it still works
    end-to-end through `main()` with no `--manifest`/`--manifest-flat-key` given."""
    bodies = _cli_source_bodies()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "source.md"
        source.write_text(CLI_SOURCE_MD, encoding="utf-8")

        dump = {
            "cli-1": {"title": "Heading One", "text": bodies["Heading One"].strip()},
            "cli-2": {"title": "Heading Two", "text": bodies["Heading Two"].strip()},
        }
        dump_path = tmp_path / "dump.json"
        dump_path.write_text(json.dumps(dump), encoding="utf-8")

        rc = ccl.main(
            [str(source), "--documents", "cli-1", "cli-2", "--dump", str(dump_path)]
        )
        assert rc == 0, f"expected exit 0 for a clean --documents run, got {rc}"


def check_cli_missing_document_via_dump() -> None:
    """End-to-end `missing_document` through the real CLI path: a dump missing one referenced
    id must make `main()` exit 1, not silently succeed."""
    bodies = _cli_source_bodies()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "source.md"
        source.write_text(CLI_SOURCE_MD, encoding="utf-8")

        dump = {"cli-1": {"title": "Heading One", "text": bodies["Heading One"].strip()}}
        # "cli-2" deliberately absent from the dump.
        dump_path = tmp_path / "dump.json"
        dump_path.write_text(json.dumps(dump), encoding="utf-8")

        rc = ccl.main(
            [str(source), "--documents", "cli-1", "cli-2", "--dump", str(dump_path)]
        )
        assert rc == 1, f"expected exit 1 when a referenced document is missing from the dump, got {rc}"


def check_cli_live_missing_venv_errors_cleanly() -> None:
    """`--live` smoke test for the "binary missing" error path (no FalkorDB needed): point
    `CYPHER_MCP_VENV_PYTHON` at a path that doesn't exist and confirm `main()` fails loud with an
    actionable message, rather than crashing on some unrelated exception."""
    original_venv_python = ccl.CYPHER_MCP_VENV_PYTHON
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "source.md"
        source.write_text(CLI_SOURCE_MD, encoding="utf-8")
        ccl.CYPHER_MCP_VENV_PYTHON = tmp_path / "does-not-exist" / "python3"
        try:
            try:
                ccl.main([str(source), "--documents", "cli-1", "--live"])
            except SystemExit as exc:
                message = str(exc)
            else:
                raise AssertionError("expected SystemExit when the venv python doesn't exist")
        finally:
            ccl.CYPHER_MCP_VENV_PYTHON = original_venv_python
    assert "setup.sh" in message or "does-not-exist" in message, (
        f"expected an actionable error naming the missing venv/setup step, got: {message!r}"
    )


# --------------------------------------------------------------------------------------------
# Nit (analyst review): --documents and --manifest-flat-key are mutually exclusive
# --------------------------------------------------------------------------------------------


def check_documents_and_flat_key_are_mutually_exclusive() -> None:
    """argparse itself must refuse `--documents` and `--manifest-flat-key` together, rather than
    one silently winning with no warning."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "source.md"
        source.write_text(CLI_SOURCE_MD, encoding="utf-8")
        try:
            ccl.main(
                [
                    str(source),
                    "--documents",
                    "cli-1",
                    "--manifest-flat-key",
                    "whatever",
                    "--live",
                ]
            )
        except SystemExit as exc:
            assert exc.code == 2, f"expected argparse's usage-error exit code 2, got {exc.code}"
        else:
            raise AssertionError(
                "expected argparse to reject --documents + --manifest-flat-key together"
            )


CHECKS = [
    ("baseline (uncorrupted fixture) reports clean on all three heading shapes", check_baseline_clean),
    ("mutation (a): dropped bullet -> UNACCOUNTED gap", check_mutation_dropped_bullet),
    ("mutation (b): truncated claim -> UNACCOUNTED gap for the remainder", check_mutation_truncated_claim),
    ("mutation (c): ASCII-for-unicode substitution -> NOT_FOUND", check_mutation_unicode_substitution),
    ("mutation (d): duplicated bullet across two claims -> DUPLICATE overlap", check_mutation_duplicated_bullet),
    ("[review Major 1] non-adjacent nested overlap -> both pairs reported", check_nonadjacent_nested_overlap),
    ("[review Major 2a] re-wrapped claim (same words) -> still locates cleanly", check_rewrapped_claim_still_locates),
    ("[review Major 2b] re-wrapped AND word dropped -> still NOT_FOUND", check_rewrapped_and_word_dropped_still_not_found),
    ("[review minor] empty/whitespace-only claim text -> empty_text status", check_empty_text_status),
    ("[review Major 3] missing_document status (check_partition-level)", check_missing_document_status),
    ("[review Major 3] CLI: primary manifest shape end-to-end", check_cli_primary_manifest_shape),
    ("[review Major 3] CLI: manifest-flat-key shape end-to-end", check_cli_manifest_flat_key_shape),
    ("[review Major 3] CLI: --documents manual mode end-to-end", check_cli_documents_manual_mode),
    ("[review Major 3] CLI: missing_document via --dump -> exit 1", check_cli_missing_document_via_dump),
    ("[review Major 3] CLI: --live with missing venv fails loud", check_cli_live_missing_venv_errors_cleanly),
    ("[review nit] --documents + --manifest-flat-key rejected by argparse", check_documents_and_flat_key_are_mutually_exclusive),
]


def main() -> int:
    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAIL: {name}\n      {exc}")
            failures += 1
        else:
            print(f"PASS: {name}")
    if failures:
        print(f"\n{failures}/{len(CHECKS)} checks FAILED")
        return 1
    print(f"\nALL PASS ({len(CHECKS)}/{len(CHECKS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
