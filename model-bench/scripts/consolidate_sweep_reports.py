#!/usr/bin/env python3
"""scripts/consolidate_sweep_reports.py — Unit C of the small-model catalog sweep's report
feature (`docs/plans/small-model-catalog-sweep.md` §3.5, §4 Unit C; requirements FR-9/FR-10).

**Not a `modelbench` runtime code path** and **not a `modelbench` computation of any kind.** It
reads the five already-rendered `rank` report files (produced by the not-yet-built `rank` CLI
command, Unit B) as opaque text and stitches them into one consolidated document: a mechanically
filled index table plus narrative placeholders. It performs **zero arithmetic** — every value it
writes into the index table is copied verbatim (as a string, never parsed to a number) from one
report's own `<!-- rank-report: ... -->` marker line. This is the letter and the purpose of the
"never compute a cross-pack number" invariant FR-9/AC-5 require: no cell, column, or sentence in
the output may combine a score from more than one pack.

**Marker contract (plan §3.5, fixed by review — `docs/reviews/small-model-catalog-sweep.md` §2.2,
Pass 2 disposition 2):** `rank_report` (Unit A) emits one `<!-- rank-report: ... -->` HTML comment
per rendered ranked table, immediately after that table:

    <!-- rank-report: pack=<packId> metric=<metricName> top=<modelKey> value=<v> ci=[<lo>,<hi>] -->

Four packs' report files carry exactly one marker each; the **guard-judge** report carries
**two** (one per `verdictMetrics` member, since that pack has no single headline metric) — so a
five-file sweep contributes **six** markers total, not five. This script is written for a variable
marker count per file from the start: it parses **every** marker line found across the files
named on `--reports`, keyed by `(pack, metric)`, never assumed 1:1 with files.

**Explicit input paths only, never glob-discovered** (plan §3.5) — a one-time, human-triggered
assembly step, and explicit inputs cannot silently pick up a stale or wrong-session report.

Output is a skeleton, not a finished document: the sample-size honesty preamble (AC-6) and each
role's "insights and recommendations" narrative are left as `<!-- TODO -->` placeholders for
whoever executes the coordination ledger's U7 by hand — a judgment call no template can discharge
faithfully (plan §3.5, option 2 "chosen").
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

#: This sweep's five packs, mapped to the FR-21 role each belongs to
#: (`docs/plans/small-model-catalog-sweep.md` §2.3's own table). A marker line names only its
#: `pack`, never its role, so this script needs its own small lookup to group the narrative
#: section's `### <role>` headings — the marker format itself is Unit A's contract, unchanged by
#: this addition. Deliberately a closed, explicit mapping rather than a live `pack.json` read
#: (`role`, nested under top-level `role` today): this script never opens a pack directory, only
#: the already-rendered report text handed to it on `--reports` (plan §3.5's own "pure text
#: assembly" framing) — reading `packs/<id>/pack.json` for one field would be a second way to
#: learn something this five-entry constant already states, and this component's own convention
#: (AGENTS.md) is to name a closed set explicitly rather than re-derive it from a file this script
#: has no other reason to touch.
_ROLE_BY_PACK_ID: Mapping[str, str] = {
    "embedder-graphrag-retrieval": "embedder",
    "guard-judge-understanding": "guard-judge",
    "nlq-structured-query": "nlq-generator",
    "tool-caller-shop-assistant": "tool-caller",
    "chat-responder-grounded-answers": "chat-responder",
}

#: `<!-- rank-report: pack=<packId> metric=<metricName> top=<modelKey> value=<v>
#: ci=[<lo>,<hi>] -->` (plan §3.5). Every captured field is kept as a raw string — never `float()`
#: or otherwise parsed — so this script has no representation in which a value could be combined,
#: rounded, or reformatted; it can only ever be copied.
_MARKER_RE = re.compile(
    r"<!--\s*rank-report:\s*"
    r"pack=(?P<pack>\S+)\s+"
    r"metric=(?P<metric>\S+)\s+"
    r"top=(?P<top>\S+)\s+"
    r"value=(?P<value>\S+)\s+"
    r"ci=\[\s*(?P<ci_low>[^,\]]+?)\s*,\s*(?P<ci_high>[^\]]+?)\s*\]\s*"
    r"-->"
)


class ConsolidateSweepReportsError(RuntimeError):
    """A script-level failure: no marker found anywhere in the given `--reports` files, or a
    marker names a `pack` this script has no role mapping for. Raised, never silently worked
    around — an empty or partial consolidated document is worse than a loud refusal."""


@dataclass(frozen=True)
class RankMarker:
    """One `<!-- rank-report: ... -->` line, fields kept verbatim as strings (never parsed to a
    number — see `_MARKER_RE`'s own docstring note)."""

    pack: str
    metric: str
    top: str
    value: str
    ci_low: str
    ci_high: str
    source: Path


def parse_markers(text: str, *, source: Path) -> list[RankMarker]:
    """Every `<!-- rank-report: ... -->` line in `text`, in the order they appear — a report
    file's own top-to-bottom order, which for the guard-judge report is the order its two ranked
    tables were rendered in (plan §3.5)."""
    return [
        RankMarker(
            pack=match["pack"],
            metric=match["metric"],
            top=match["top"],
            value=match["value"],
            ci_low=match["ci_low"],
            ci_high=match["ci_high"],
            source=source,
        )
        for match in _MARKER_RE.finditer(text)
    ]


def load_markers(report_paths: Sequence[Path]) -> list[RankMarker]:
    """Reads each of `report_paths` in the given order and concatenates their markers — file
    order first, then in-file order (never re-sorted): the assembled index table's row order is a
    direct, traceable function of the operator's own `--reports` argument order."""
    markers: list[RankMarker] = []
    for report_path in report_paths:
        text = report_path.read_text(encoding="utf-8")
        markers.extend(parse_markers(text, source=report_path))
    return markers


def _role_for_pack(pack_id: str) -> str:
    try:
        return _ROLE_BY_PACK_ID[pack_id]
    except KeyError:
        known = ", ".join(sorted(_ROLE_BY_PACK_ID))
        raise ConsolidateSweepReportsError(
            f"marker names pack {pack_id!r}, which this script has no role mapping for; "
            f"known packs: {known}"
        ) from None


def _distinct_roles_in_order(markers: Sequence[RankMarker]) -> list[str]:
    """The roles actually represented across `markers`, each named once, in first-appearance
    order — never the full five-role set regardless of input: a `--reports` call given fewer than
    five files must not manufacture placeholder headings for roles nobody supplied a report for
    (and guard-judge's two markers, same pack, must not double its own heading)."""
    seen: set[str] = set()
    roles: list[str] = []
    for marker in markers:
        role = _role_for_pack(marker.pack)
        if role not in seen:
            seen.add(role)
            roles.append(role)
    return roles


def _index_table(markers: Sequence[RankMarker], *, out_dir: Path) -> str:
    lines = [
        "| Pack | Metric | Top model | Value | 95% CI | Report |",
        "|---|---|---|---|---|---|",
    ]
    for marker in markers:
        relative_report = os.path.relpath(marker.source.resolve(), start=out_dir.resolve())
        lines.append(
            f"| `{marker.pack}` | `{marker.metric}` | `{marker.top}` | {marker.value} | "
            f"[{marker.ci_low}, {marker.ci_high}] | [{marker.source.name}]({relative_report}) |"
        )
    return "\n".join(lines)


_PREAMBLE_TODO = (
    "<!-- TODO: state plainly what this sweep's sample sizes can and cannot prove — capable of "
    "catching a large collapse, not fine-grained ranking among closely-matched 3-4B models "
    "(AC-6) -->"
)

_NARRATIVE_TODO = "<!-- TODO: narrative, grounded only in this role's own within-pack report -->"


def _narrative_section(markers: Sequence[RankMarker]) -> str:
    lines = ["## Insights and recommendations", ""]
    for role in _distinct_roles_in_order(markers):
        lines.append(f"### {role}")
        lines.append("")
        lines.append(_NARRATIVE_TODO)
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def build_consolidated_document(markers: Sequence[RankMarker], *, out_path: Path) -> str:
    """Assembles the full consolidated document (plan §3.5): a preamble TODO, the mechanical index
    table (one row per marker, zero arithmetic), and one narrative TODO per distinct role. Raises
    `ConsolidateSweepReportsError` on an empty `markers` list — a consolidated document with no
    rows at all means every `--reports` file was either wrong or carried no marker, and writing an
    empty skeleton anyway would look like a legitimate, if odd, result rather than the input
    mistake it actually is."""
    if not markers:
        raise ConsolidateSweepReportsError(
            "no `<!-- rank-report: ... -->` marker found in any of the given --reports files"
        )
    sections = [
        "# Small-Model Catalog Sweep — Consolidated Report",
        "",
        _PREAMBLE_TODO,
        "",
        "## Index",
        "",
        _index_table(markers, out_dir=out_path.parent),
        "",
        _narrative_section(markers),
    ]
    return "\n".join(sections).rstrip("\n") + "\n"


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="consolidate_sweep_reports.py",
        description=(
            "Mechanical text-extraction assembler: stitches the small-model catalog sweep's "
            "already-rendered `rank` reports into one consolidated index + narrative-placeholder "
            "document (FR-9/FR-10). Zero arithmetic — every index-table value is copied verbatim "
            "from one report's own `<!-- rank-report: ... -->` marker line, never combined with "
            "another pack's."
        ),
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        type=Path,
        help="explicit paths to the rendered `rank` report files, one per pack (the guard-judge "
        "report contributes two index rows, from its own two marker lines) — never glob-expanded",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="path to write the consolidated document to"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    for report_path in args.reports:
        if not report_path.is_file():
            print(
                f"consolidate_sweep_reports.py: no such report file: {report_path}",
                file=sys.stderr,
            )
            return 2

    markers = load_markers(args.reports)

    try:
        document = build_consolidated_document(markers, out_path=args.out)
    except ConsolidateSweepReportsError as exc:
        print(f"consolidate_sweep_reports.py: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
