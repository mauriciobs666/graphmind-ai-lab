#!/usr/bin/env python3
"""flag_split_candidates_selftest.py — mutation test for `flag_split_candidates.py`.

Not a pytest suite. `claude/` has no pytest package anywhere under it (no `pyproject.toml`,
no `tests/` collection root — see `claude/cobb/TESTING.md`, which reserves pytest for "future
library/utility code" living in its own component directory with that scaffolding, e.g.
`excel_extractor/`), and this is a single standalone script with zero runtime dependencies, not a
library. A plain assertion script run directly, in this repo's own house style for a mechanical
check (compare `falkor-chat/scripts/verify_workflows.sh`, `verify_catalog.sh`: exit 0/1, printed
diagnosis, no test framework), is the better fit than inventing a first pytest package under
`claude/` for one file.

WHAT THIS CHECKS
-----------------
1. **The known-answer fixture** (`fixtures/mutation_test_kb.md`) — one section written to be
   clearly single-claim (one technique, one worked example, one attribution) and one written to
   be clearly multi-claim (three independently-discovered, independently-fixed, independently-
   attributed failure modes under one heading). Asserts the tool tells them apart: the first
   section unflagged, the second flagged.
2. **Mutation of the multi-claim section** — strip its three bold sub-header markers and its
   three "Origin:" lines down to a single flowing paragraph with one attribution, keeping the word
   count essentially unchanged. Asserts the mutated section drops BELOW the flag threshold,
   proving the structural/attribution signals — not raw length — are what drove the original flag
   (a tool that flagged on word count alone would still flag this mutant, which would mean the
   structural signal was never actually load-bearing).
3. **Mutation of the single-claim section** — pad it with bold sub-headers and extra "Origin:"
   lines that don't correspond to any real second claim (an adversarial false-multi-claim mutant).
   Asserts this now DOES flag — proving the tool is reacting to the signals it claims to detect,
   not to some other property of the fixture text, and demonstrating the tool's own honestly-named
   limitation (module docstring, LIMITATIONS): it counts markers, it does not verify they
   correspond to genuinely separate claims — that verification is cobb's, not this script's.

Run: `python3 claude/cobb/scripts/flag_split_candidates_selftest.py`
Exit 0 and "ALL PASS" on success; exit 1 and the failing assertion's diagnosis otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flag_split_candidates as fsc  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mutation_test_kb.md"

SINGLE_CLAIM_HEADING = "Retrying a flaky health probe with linear backoff"
MULTI_CLAIM_HEADING = (
    "Three separable failure modes when caching a per-request auth token, "
    "each independently attributed"
)


def _find(stats: list[fsc.SectionStats], heading: str) -> fsc.SectionStats:
    for s in stats:
        if s.heading == heading:
            return s
    raise AssertionError(f"fixture section not found: {heading!r}")


def check_fixture_baseline() -> None:
    stats = fsc.analyze_file(FIXTURE)
    single = _find(stats, SINGLE_CLAIM_HEADING)
    multi = _find(stats, MULTI_CLAIM_HEADING)

    assert not single.flagged(fsc.DEFAULT_WORD_THRESHOLD, fsc.DEFAULT_STRUCT_THRESHOLD), (
        f"single-claim fixture section was flagged (words={single.words}, "
        f"structural={single.structural_count}, attribution={single.attribution_count}) "
        "-- the tool should not flag a short, single-attribution, single-technique section"
    )
    assert multi.flagged(fsc.DEFAULT_WORD_THRESHOLD, fsc.DEFAULT_STRUCT_THRESHOLD), (
        f"multi-claim fixture section was NOT flagged (words={multi.words}, "
        f"structural={multi.structural_count}, attribution={multi.attribution_count}) "
        "-- three independently-attributed sub-claims under one heading must trip the trigger"
    )
    # Sanity on the signals themselves, not just the final boolean -- catches a heuristic that
    # flags/unflags for the wrong reason.
    assert multi.structural_count >= 2, (
        f"expected >=2 structural signals (bold sub-headers/enum cues) in the multi-claim "
        f"section, got {multi.structural_count}"
    )
    assert multi.attribution_count >= 2, (
        f"expected >=2 attribution markers in the multi-claim section, got "
        f"{multi.attribution_count}"
    )


def check_mutation_strip_structure_from_multi_claim() -> None:
    """Collapse the multi-claim section's 3 bold sub-headers + 3 Origins into one flowing
    paragraph, same rough length. If the tool still flags this, the flag was never actually
    driven by "multiple sub-claims" -- it was driven by raw length, which is exactly the failure
    mode the structural signal exists to avoid (module docstring, trigger (b))."""
    text = FIXTURE.read_text(encoding="utf-8")
    heading_marker = f"## {MULTI_CLAIM_HEADING}"
    start = text.index(heading_marker)
    next_heading = text.find("\n## ", start + len(heading_marker))
    end = next_heading if next_heading != -1 else len(text)

    mutated_body = (
        "Caching a resolved auth token across requests inside one process avoiding "
        "re-validating it on every call needs care around expiry, tenant scoping under a "
        "process-global cache, and simultaneous revalidation under load, all of which were "
        "observed together during one extended investigation into a single reported "
        "slowdown, worked through end to end without a clean separation between the "
        "individual causes because the underlying token-caching code changed shape several "
        "times over the course of that investigation and the write-up here reflects the "
        "investigation as it actually happened rather than a tidy retrospective decomposition "
        "of it into distinct named failure modes, which is why what follows reads as one "
        "long account rather than several short ones even though more than one thing changed "
        "over its course.\n\n"
        + ("filler prose to hold the word count roughly steady with the original section. " * 30)
        + "\n"
    )
    mutated_section = f"{heading_marker}\n\n{mutated_body}\n"
    mutated_text = text[:start] + mutated_section + text[end:]

    tmp = FIXTURE.parent / "_mutant_stripped_structure.md"
    tmp.write_text(mutated_text, encoding="utf-8")
    try:
        stats = fsc.analyze_file(tmp)
        mutant = _find(stats, MULTI_CLAIM_HEADING)
        assert mutant.structural_count < 2, (
            f"mutant should have <2 structural signals after stripping bold sub-headers, "
            f"got {mutant.structural_count}"
        )
        assert not mutant.flagged(fsc.DEFAULT_WORD_THRESHOLD, fsc.DEFAULT_STRUCT_THRESHOLD), (
            f"mutant (structure stripped, length kept roughly steady at {mutant.words} words) "
            "was still flagged -- the original flag was driven by length alone, not by the "
            "structural/attribution signals this tool claims to detect"
        )
    finally:
        tmp.unlink(missing_ok=True)

    # Keep the mutation honest: confirm the un-mutated fixture still carries the signal the
    # mutation is supposed to remove (a regression here would make the mutation trivially pass).
    original_stats = fsc.analyze_file(FIXTURE)
    original_multi = _find(original_stats, MULTI_CLAIM_HEADING)
    assert original_multi.structural_count >= 2, "fixture regressed before mutation was applied"


def check_mutation_pad_false_structure_into_single_claim() -> None:
    """Adversarial direction: inject bold sub-headers and extra "Origin:" lines into the
    single-claim section without adding any real second claim. Confirms the tool reacts to the
    signals it names, not to some other property of the fixture -- and demonstrates, honestly,
    that the tool cannot distinguish real sub-claims from padding shaped like them (module
    docstring, LIMITATIONS)."""
    text = FIXTURE.read_text(encoding="utf-8")
    heading_marker = f"## {SINGLE_CLAIM_HEADING}"
    start = text.index(heading_marker)
    next_heading = text.find("\n## ", start + len(heading_marker))
    end = next_heading if next_heading != -1 else len(text)

    padded_addition = (
        "\n\n**A cosmetic restatement of the same point, not a second claim.** This is the "
        "identical backoff technique described above, repeated with different wording, to see "
        "whether the tool can tell a real second sub-claim from a reworded restatement of the "
        "first one -- by construction here, it cannot.\n\n"
        "Origin: added by the self-test mutant, not a real second incident.\n\n"
        "**A second cosmetic restatement, still not a second claim.** Same technique again, "
        "worded a third way, to give the padding two bold markers rather than one.\n\n"
        "Origin: also added by the self-test mutant, not a real third incident.\n"
    )
    mutated_text = text[:end] + padded_addition + text[end:]

    tmp = FIXTURE.parent / "_mutant_padded_structure.md"
    tmp.write_text(mutated_text, encoding="utf-8")
    try:
        stats = fsc.analyze_file(tmp)
        mutant = _find(stats, SINGLE_CLAIM_HEADING)
        assert mutant.structural_count >= 2, (
            f"padding should raise structural_count to >=2, got {mutant.structural_count}"
        )
        assert mutant.flagged(fsc.DEFAULT_WORD_THRESHOLD, fsc.DEFAULT_STRUCT_THRESHOLD), (
            "padded single-claim section (now with 2 bold markers + 2 Origins) was not "
            "flagged -- the tool is not reacting to the signals it claims to count"
        )
    finally:
        tmp.unlink(missing_ok=True)


CHECKS = [
    ("fixture baseline (single unflagged, multi flagged)", check_fixture_baseline),
    ("mutation: strip structure from multi-claim section -> unflags", check_mutation_strip_structure_from_multi_claim),
    ("mutation: pad false structure into single-claim section -> flags", check_mutation_pad_false_structure_into_single_claim),
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
