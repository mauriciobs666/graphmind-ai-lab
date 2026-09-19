#!/usr/bin/env python3
"""flag_split_candidates.py — split-candidate flagging tool for the agent knowledge-base
migration (K-030, `claude/docs/plans/agent-knowledge-base-strategy.md` §6, Track 2 Stage 6).

WHAT THIS IS (read this before trusting its output)
-----------------------------------------------------
This is a **candidate-flagging tool, not a splitter**. It never decides a split boundary and
never writes anything. For every `## `-delimited section in every input file, it computes three
mechanical signals and applies a heuristic trigger; a "yes" means "cobb, look at this one during
the migration's curator judgment pass" — nothing stronger. An unflagged section migrates as a
single claim unchanged; a flagged one may still turn out to be one claim that is merely long or
example-heavy — the tool cannot tell the difference, only cobb's editorial read can.

Implements the plan's Stage 6 step 1 literally: "per file, per `## ` section: word count, count of
bolded sub-headers/enumerated checks, count of 'Origin:' mentions." The trigger is the ML method
note's own wording (`agent-knowledge-base-strategy-ml.md`, Recommendation 2): "a heading becomes N
documents when it contains N independently-attributed sub-claims — operationally, N distinct
bolded sub-headers/enumerated checks each carrying their own 'Origin:'/worked-example, or a heading
materially longer than ~500 words spanning more than one verifiable claim."

THE THREE SIGNALS, AND WHY THEY'RE DETECTED THIS WAY
------------------------------------------------------
1. **Word count** — `len(body.split())` over the section body (everything between one `## `
   heading and the next, or EOF), a plain mechanical count, code fences included.

2. **Sub-header/enumerated-check count** — this repo's existing dense KBs
   (`claude/analyst/review-techniques.md`, `claude/teco/coordination-techniques.md`) mark a
   distinct sub-point one of three ways, and all three are counted as one signal:
     - a bullet or numbered list item opening with a bold run, e.g.
       `1. **Keyword-set completeness.**` or `- **Deletion, not modification, is the mutation.**`
     - a paragraph whose own first token is a bold run used as an inline pseudo-header, e.g.
       `**Same blindness, AST flavour.** A guard that walks one function for...`
     - an explicit small-number-plus-noun cue in running prose, e.g. "**Two checks** for a
       multi-shape...", "Three shapes it takes:", "two axes" — `claude/teco/coordination-
       techniques.md` bundles its multi-claim sections this way (flowing prose, no markdown
       structure at all) rather than with bold/list markers, so a tool that only looked for
       markdown bold/list syntax would be blind to that file's entire convention. This cue regex
       is deliberately narrow (a small number word/digit followed by one of a fixed set of nouns:
       shapes, checks, flavo(u)rs, cases, claims, kinds, axes, questions, reasons, ways) — it will
       under-count prose enumerations phrased some fourth way, which is an accepted, named
       limitation (see LIMITATIONS below), not a silent one.

3. **Attribution-marker count** — literal `Origin:` mentions, per the plan's exact wording, PLUS a
   second pattern, `Verified <YYYY-MM-DD>` / `verified <YYYY-MM-DD>` (case-insensitive), because
   `review-techniques.md` itself does not use `Origin:` uniformly — several of its most clearly
   multi-claim sections (e.g. "A guard derived from the artifact it guards...") attribute each
   sub-claim with "Verified 2026-09-08 on a synthetic pair..." instead. The task brief's own
   parenthetical ("count of 'Origin:' (or similar attribution-marker) mentions") licenses this;
   the two counts are reported separately in the report and summed only for the trigger decision.

THE TRIGGER, EXACTLY
---------------------
A section is flagged when EITHER:
  (a) `structural_count >= STRUCT_THRESHOLD` (default 2) — two or more distinct sub-header/
      enumerated-check/cue signals is read as "this heading names more than one discrete point,"
      independent of length. This is what catches `review-techniques.md`'s "Two checks for a
      multi-shape authorization/security-gate function" (322 words — under the 500-word floor,
      but structurally two enumerated, independently-actionable checks) and
      `coordination-techniques.md`'s "Mutation-test the green-on-arrival tests" (379 words, no
      bold markers at all, but an explicit "Two shapes..." cue plus a bold lead-in) — both are
      exactly the shape Recommendation 2 describes, and both would be missed by a word-count-only
      rule.
  (b) `word_count > WORD_THRESHOLD` (default 500) AND (`structural_count >= 1` OR
      `attribution_count >= 1`) — a heading past the ML note's own ~500-word figure, with at least
      one visible sign of internal structure or more than one attributed claim. A long section
      with NEITHER (pure prose, no bold/enumeration/cue, no repeated attribution) is read as one
      long worked example, not evidence of bundling, and is deliberately left unflagged even past
      500 words — the plan's own framing is "spanning more than one verifiable claim," not merely
      "long."

Both thresholds are CLI-overridable (`--word-threshold`, `--struct-threshold`) since they are
explicitly provisional in the source material, not physical constants.

KNOWN CALIBRATION, AGAINST THE REAL FILES
-------------------------------------------
Run against `claude/analyst/review-techniques.md` (the plan's own "densest KB" example): flags
25/54 sections (~46%), including both sections the ML note names by name as multi-claim bundles
("A guard derived from the artifact it guards is blind along the derivation axis...", 1287 words,
15 structural signals, 4 attributions; "Two checks for a multi-shape authorization/security-gate
function", 322 words, 3 structural signals) — see `docs/plans/agent-knowledge-base-strategy-ml.md`
Findings section for the citation. A 46% flag rate looks high for a "candidate" filter, but this
file is independently described, in both the plan and the ML note, as the one KB that most
violates the "one heading = one technique" assumption ("headings bundling 5+ independently-
verified sub-claims... running well past 1,500 words") — a broad flag set here is the tool
correctly reflecting a genuinely dense file, not over-triggering; cobb's judgment pass is exactly
the step that turns this candidate list into real split boundaries.

Run against `claude/teco/coordination-techniques.md` (flagged in the plan's §3 Track 0 note as
having "a handful of sections... still bundle 2-3 separable claims"): flags 2/37 sections
("Mutation-test the green-on-arrival tests", "Verifying which files are actually yours before an
integration commit: the three-way diff check") — both independently read, on inspection, as
bundling multiple separable claims (the first: backup-verification-by-chaining, argument-level
mutation, and the invented-name/uniform-stub blind spots; the second: three distinct diagnostic
readings of a three-way diff). This file's much lower flag rate versus `review-techniques.md` is
real, not a tuning artifact: its prevailing convention is flowing prose with almost no markdown
bold/list structure and (per a direct count) exactly one literal "Origin:" mention in the whole
file — so the tool has far less structural signal to work with there, and a handful of hits is the
expected, not an inflated, result.

LIMITATIONS (named, not hidden)
---------------------------------
- The enumeration-cue regex is a fixed, small vocabulary; a section that bundles multiple claims
  in prose using none of its cue words, and no bold/list markers, and stays under the word
  threshold, will not be flagged. This is a real, accepted recall gap, not a false "clean."
- "Structural signal" and "attribution marker" are counted per SECTION, not paired per sub-claim —
  the tool cannot confirm that a given bold sub-header and a given "Origin:" mention actually
  belong to the same sub-claim, only that both appear somewhere in the section. This is why the
  report shows both raw counts: a human skim of a flagged section is still required.
- This tool has no opinion on WHERE inside a flagged section a split boundary belongs — that is
  explicitly cobb's call (plan §6 step 2), never this tool's.

USAGE
-----
    python3 flag_split_candidates.py FILE [FILE ...]
    python3 flag_split_candidates.py claude/*/*.md
    python3 flag_split_candidates.py --only-flagged claude/analyst/review-techniques.md
    python3 flag_split_candidates.py --format plain claude/teco/coordination-techniques.md

File arguments are plain paths — shell globbing (`claude/*/*.md`) does the filtering; this script
never globs on its own, so the caller (cobb) always controls exactly which confirmed KB files are
in scope, per the migration brief's own requirement.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)

# A line opening with an optional bullet/number marker, then a bold run: covers
# "- **...**", "* **...**", "1. **...**", and a bare "**...**" lead-in paragraph.
SUBHEADER_RE = re.compile(r"^(?:[-*]\s+|\d+\.\s+)?\*\*", re.MULTILINE)

# A small-number-word-or-digit followed (within a short span) by one of a fixed
# set of enumeration nouns -- the prose-only equivalent of a bold/list sub-header,
# needed for a file like coordination-techniques.md that has almost no markdown
# structure at all. See the module docstring's signal-2 discussion.
ENUM_CUE_RE = re.compile(
    r"\b(two|three|four|five|2|3|4|5)\b[^.\n]{0,40}\b"
    r"(shapes?|checks?|flavou?rs?|cases?|claims?|kinds?|axes|questions?|reasons?|ways?)\b",
    re.IGNORECASE,
)

ORIGIN_RE = re.compile(r"\bOrigin:")
VERIFIED_RE = re.compile(r"\b[Vv]erified\b[^.\n]{0,40}\d{4}-\d{2}-\d{2}")

DEFAULT_WORD_THRESHOLD = 500
DEFAULT_STRUCT_THRESHOLD = 2


@dataclass
class SectionStats:
    file: Path
    heading: str
    words: int
    subheaders: int
    enum_cues: int
    origin_mentions: int
    verified_mentions: int

    @property
    def structural_count(self) -> int:
        return self.subheaders + self.enum_cues

    @property
    def attribution_count(self) -> int:
        return self.origin_mentions + self.verified_mentions

    def flagged(self, word_threshold: int, struct_threshold: int) -> bool:
        if self.structural_count >= struct_threshold:
            return True
        if self.words > word_threshold and (
            self.structural_count >= 1 or self.attribution_count >= 1
        ):
            return True
        return False


def parse_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading, body) for every top-level `## ` section in `text`.

    Content before the first `## ` heading (the H1 title, the `Status:` header block) is not a
    section and is never analyzed -- it carries no distillable claim, per this repo's own doc-
    header convention.
    """
    matches = list(HEADING_RE.finditer(text))
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((heading, text[start:end]))
    return sections


def analyze_file(path: Path) -> list[SectionStats]:
    text = path.read_text(encoding="utf-8")
    stats = []
    for heading, body in parse_sections(text):
        stats.append(
            SectionStats(
                file=path,
                heading=heading,
                words=len(body.split()),
                subheaders=len(SUBHEADER_RE.findall(body)),
                enum_cues=len(ENUM_CUE_RE.findall(body)),
                origin_mentions=len(ORIGIN_RE.findall(body)),
                verified_mentions=len(VERIFIED_RE.findall(body)),
            )
        )
    return stats


def render_markdown(rows: list[SectionStats], word_threshold: int, struct_threshold: int) -> str:
    lines = [
        "| File | Heading | Words | Sub-headers/enum | Origin: | Verified <date> | Flagged |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for s in rows:
        flagged = "yes" if s.flagged(word_threshold, struct_threshold) else "no"
        heading = s.heading.replace("|", "\\|")
        lines.append(
            f"| `{s.file}` | {heading} | {s.words} | {s.structural_count} | "
            f"{s.origin_mentions} | {s.verified_mentions} | {flagged} |"
        )
    return "\n".join(lines)


def render_plain(rows: list[SectionStats], word_threshold: int, struct_threshold: int) -> str:
    lines = []
    for s in rows:
        flagged = "FLAG" if s.flagged(word_threshold, struct_threshold) else "    "
        lines.append(
            f"{flagged}  words={s.words:5d}  subheaders/enum={s.structural_count:2d}  "
            f"origin={s.origin_mentions}  verified={s.verified_mentions}  "
            f"[{s.file}] {s.heading}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Flag ## sections past the agent-KB migration's split-candidate trigger "
            "(K-030 Stage 6). Surfaces candidates for cobb's curator judgment pass; "
            "never splits or decides a boundary. Run with --help for the full heuristic."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="+", type=Path, help="Markdown KB files to analyze.")
    parser.add_argument(
        "--word-threshold",
        type=int,
        default=DEFAULT_WORD_THRESHOLD,
        help=f"Word-count trigger for the length-based rule (default: {DEFAULT_WORD_THRESHOLD}).",
    )
    parser.add_argument(
        "--struct-threshold",
        type=int,
        default=DEFAULT_STRUCT_THRESHOLD,
        help=(
            "Minimum sub-header/enumerated-check count that flags a section on its own, "
            f"regardless of length (default: {DEFAULT_STRUCT_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--only-flagged",
        action="store_true",
        help="Print only the flagged sections, not the full per-section report.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "plain"],
        default="markdown",
        help="Output format (default: markdown table).",
    )
    args = parser.parse_args(argv)

    all_stats: list[SectionStats] = []
    for path in args.files:
        if not path.is_file():
            print(f"error: not a file: {path}", file=sys.stderr)
            return 2
        all_stats.extend(analyze_file(path))

    rows = all_stats
    if args.only_flagged:
        rows = [s for s in rows if s.flagged(args.word_threshold, args.struct_threshold)]

    if args.format == "markdown":
        print(render_markdown(rows, args.word_threshold, args.struct_threshold))
    else:
        print(render_plain(rows, args.word_threshold, args.struct_threshold))

    total = len(all_stats)
    flagged = sum(
        1 for s in all_stats if s.flagged(args.word_threshold, args.struct_threshold)
    )
    print(f"\n{flagged}/{total} sections flagged as split candidates.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
