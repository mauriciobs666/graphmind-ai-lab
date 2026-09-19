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

Run: `python3 claude/cobb/scripts/check_content_loss_selftest.py`
Exit 0 and "ALL PASS" on success; exit 1 and the failing assertion's diagnosis otherwise.
"""

from __future__ import annotations

import re
import sys
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


CHECKS = [
    ("baseline (uncorrupted fixture) reports clean on all three heading shapes", check_baseline_clean),
    ("mutation (a): dropped bullet -> UNACCOUNTED gap", check_mutation_dropped_bullet),
    ("mutation (b): truncated claim -> UNACCOUNTED gap for the remainder", check_mutation_truncated_claim),
    ("mutation (c): ASCII-for-unicode substitution -> NOT_FOUND", check_mutation_unicode_substitution),
    ("mutation (d): duplicated bullet across two claims -> DUPLICATE overlap", check_mutation_duplicated_bullet),
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
