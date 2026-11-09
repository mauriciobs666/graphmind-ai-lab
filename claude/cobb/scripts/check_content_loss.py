#!/usr/bin/env python3
"""check_content_loss.py — content-loss/fidelity checker for the agent knowledge-base migration
(K-030, `claude/docs/plans/agent-knowledge-base-strategy.md` §6, Track 2 Stage 6).

WHAT THIS IS
------------
Given a source KB markdown file and the set of `ws:agent-team` documents that claim to have
migrated it (per `kb-claim-manifest.json`, or an explicit id list), verifies **no content was
lost or altered**: every substantive sentence/bullet from the source heading(s) appears,
byte-for-byte-equivalent modulo the split, in exactly one of the resulting claim documents — no
dropped bullets, no truncation, no duplicated-then-diverged text, no silently-altered characters
(the exact fidelity slip `cobb` caught in `statistical-method-techniques.md`: a claim first typed
with ASCII substitutes for Greek/math unicode). It never decides a split boundary and never writes
anything — a read-only verifier, the mirror image of `flag_split_candidates.py`.

THE METHOD: locate, don't diff-by-word
---------------------------------------
Inspection of real migrated documents (`get_document` against several already-clean claims) shows
`text` is either the section body verbatim in full (a single-claim heading) or an exact,
contiguous substring of it (a paragraph-/subsection-boundary split, e.g.
`guard-testing-techniques.md`'s heading 1 -> 4 claims) — internal line wraps, bold markers, code
spans and all, character-for-character. The one exception is a bulleted heading split one claim
per bullet (`falkordb-reference.md`, `frontend-quirks.md`, `ops-quirks.md`'s style): there, the
claim text is the bullet's content with its leading `- `/`* `/`N. ` list marker stripped — the
marker itself is legitimately consumed by the split, not lost content.

So the check is: for each claim, **locate** its `text` as an exact substring of the source body
(trying the bare text first, then the text prefixed by each list-marker shape) rather than
tokenizing and comparing bags of words. This is strictly byte-exact by construction (Python
substring search is exact-character), and it turns each of the four required defect shapes into a
distinct, legible signal without any fuzzy scoring:

- **dropped bullet/sentence** -> its span is never claimed by anyone -> shows up as leftover
  non-whitespace content in a **gap** between located spans.
- **truncated claim** -> the truncated text is trivially still a substring (a prefix of the real
  one) and is found -> but the chopped-off remainder is not claimed by anything else -> the same
  **gap** signal, immediately downstream of the truncated claim's own span.
- **substituted/altered character(s)** (the ASCII-for-unicode slip) -> even a single differing
  code point makes the claim text NOT an exact substring anywhere -> **not found**; a
  `difflib`-anchored diagnostic then shows exactly which characters differ, using the source's
  best-matching region as the reference.
- **duplicated bullet into two claims** -> both claims' text independently locate to the exact same
  source span -> reported as an **overlap** between two different documentIds.

A gap or an overlap or a not-found claim is a genuine finding, always; a clean run reports zero of
all three. There is no threshold to tune and nothing here is a heuristic judgment call — unlike
`flag_split_candidates.py`, this tool is not surfacing candidates for a human to review, it is
answering a yes/no question about a partition that has already been decided.

MANIFEST SCHEMA — a real fork, resolved by degrading gracefully rather than guessing
---------------------------------------------------------------------------------------
`kb-claim-manifest.json`, read directly rather than assumed, carries **two materially different
shapes** as of this writing (2026-09-18):

1. **The primary shape** — `manifest["files"][<repo-relative path>]["headings"]`, a list of
   `{"heading", "claims": [{"title", "documentId", ...}]}` — a full per-heading breakdown. This
   covers all 8 of the files reported clean in the migration ledger and is what `--manifest`
   mode (the default) reads.
2. **Four irregular top-level keys** (`_teco_coordination_techniques`,
   `_tdd_engineer_guard_testing_techniques`, `_data_scientist_lm_studio_model_notes`,
   `_graph_dba_falkordb_quirks_IN_PROGRESS`) carry only a **flat** `documentIds` list (or, for the
   last one, two separately-named partial lists) with no heading breakdown at all — the
   heading-to-claim mapping lives only in that key's free-text `_note`, not in structured data.
   The manifest is a live, evolving document (`cobb`'s own in-flight migration ledger, not part of
   this tool's commit) — the exact key names/counts above are a snapshot, already observed to
   have moved on since (`analyst` review, Nit 2: `_graph_dba_falkordb_quirks_IN_PROGRESS` has since
   been renamed and restructured). Neither fallback below depends on these specifics staying
   accurate; `--manifest-flat-key` fails loud on a key that no longer has a flat `documentIds` (as
   it always did), and `--documents` never reads the manifest at all.

Rather than parse that prose to reconstruct heading boundaries (a guess this tool refuses to make
silently), this tool offers two schema-agnostic fallbacks that never need per-heading structure:
`--manifest-flat-key KEY` (reads a `documentIds` array directly, for the three keys that have one)
and `--documents ID [ID ...]` (a fully manual id list, for the one that doesn't, or for any ad hoc
re-check). Both run the check at **whole-file granularity**: every claim in the list against the
concatenation of every (or, with `--headings`, a named subset of) the source file's section bodies.
This is a strictly *more* conservative check than the per-heading mode, not a weaker one — it still
catches every defect shape above (a drop/truncation/alteration/duplication is still a drop/
truncation/alteration/duplication when the search space is the whole file instead of one heading's
body) — it just can't tell you *which* heading a finding belongs to, only where in the file.

USAGE
-----
    # Primary shape, live fetch (talks to FalkorDB directly, no MCP session needed):
    python3 check_content_loss.py claude/graph-dba/falkordb-reference.md --live

    # Primary shape, offline (fetch texts yourself, e.g. from inside an agent session via
    # get_document, dump them, then check -- see --dump's format below):
    python3 check_content_loss.py claude/graph-dba/falkordb-reference.md --dump /tmp/docs.json

    # Irregular flat-key entry, whole file:
    python3 check_content_loss.py claude/teco/coordination-techniques.md \\
        --manifest-flat-key _teco_coordination_techniques --live

    # Fully manual re-check (e.g. re-verifying guard-testing-techniques.md's 19 stuck documents --
    # the exact case named in this tool's own usage note):
    python3 check_content_loss.py claude/tdd-engineer/guard-testing-techniques.md --live \\
        --documents 957f7131fc2a48dfb1740345b3ea55e1 42951b59bbf34aef8459e0f31f6cea6d ...

    # Partial migration, restricted to the headings actually done so far:
    python3 check_content_loss.py claude/graph-dba/falkordb-quirks.md --live \\
        --headings "Indexing, constraints & DDL" "Concurrency & atomicity" \\
        --documents 7ac2792f... 36d7a77c...

`--live` connects directly to FalkorDB via `cypher-mcp/.venv`'s `falkordb` client (same
`FALKORDB_HOST`/`FALKORDB_PORT` convention `cypher-mcp` itself uses, default `127.0.0.1:6379`) as
a subprocess -- no running MCP session required, but `cypher-mcp/setup.sh` must have been run once.
`--dump PATH` reads a JSON file `{documentId: {"title": ..., "text": ...}}` instead -- the offline/
testable path, and the natural choice from inside a session that already has `get_document`
(fetch, write the dict to a file, run this script). Exactly one of `--live`/`--dump` is required.

    # python-web-quirks copy (docs/plans/python-web-quirks-kb.md D3/D4; manifest
    # pwq-kb-manifest.json, {pin, migratedOn, units:[{id,title,headings,documentId,sha256}],
    # waivedSpans}). Stored texts carry a header up to a `<!-- body -->` line, then one or more
    # `## ` parts (merged units span non-adjacent sections). Checks: each part is located in the
    # source section of the same heading; the copy's `<!-- preamble -->` block is a substring of
    # the source preamble and every other non-blank span of it equals a waivedSpans entry; the
    # manifest's heading set equals the source's (an extra source heading fails by name).
    python3 check_content_loss.py skills/python-web-quirks/SKILL.md --stored-format pwq \\
        --copy claude/knowledge/python-web-quirks.md --live --ws pwq-test
    # AC-5: the same check with the copy as the source side (--source-format copy).
    # --ws NAME picks the workspace graph `ws:NAME` for any --live run (default agent-team).

Exit 0 and a clean report on a fully-accounted, non-overlapping partition; exit 1 and an itemized
finding list otherwise (dropped/truncated content, altered claim text, duplicated claim,
unresolvable heading/documentId) -- same exit-code contract as `falkor-chat/scripts/
verify_workflows.sh`/`verify_catalog.sh` and this script's own sibling `flag_split_candidates.py`.

LIMITATIONS (named, not hidden)
---------------------------------
- `locate_claim` uses `str.find`, which returns the FIRST occurrence. If a claim's exact text
  (bare or marker-prefixed) genuinely occurs more than once in the source body, a wrong occurrence
  could be picked, misreporting a real gap/overlap elsewhere. Not observed in this corpus's actual
  files (distilled technique prose does not repeat itself verbatim); named rather than silently
  assumed away.
- Whole-file/flat mode cannot attribute a finding to a specific heading — only per-heading mode
  (the primary manifest shape) can. See "MANIFEST SCHEMA" above.
- This tool has no opinion on whether a split boundary was the *right* one editorially — only
  whether the content on both sides of it is completely and exactly accounted for.
- **False positive: a claim pair sharing one duplicated context/label sentence.** This corpus's
  documented, deliberate split convention for two claims that share one context sentence/citation
  which "doesn't split by trap" (e.g. a shared Origin, duplicated in both siblings) gives each
  claim the shared text plus its own distinct portion — so claim B's content sits *between*
  claim A's two halves in the source, and neither sibling's stored text is one contiguous
  substring of it. `locate_claim` requires contiguity, so both siblings read as NOT_FOUND (and
  the interleaved span between them as an UNACCOUNTED gap) even though the pair's *union*, once
  the duplicated shared prefix is stripped, exactly reconstructs the source with nothing missing,
  added, or altered. Not a tool bug to fix — a legitimate split shape this contiguous-substring
  method cannot represent. Worked example, byte-for-byte verified: `claude/docs/reviews/
  agent-knowledge-base-strategy4-stage6.md` (Finding 1) — concrete instances at
  `review-techniques.md`'s h13/h14, h21/h22, h39/h40 documentId pairs and
  `qa-testing-techniques.md`'s model-bench-attest heading.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flag_split_candidates as fsc  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "kb-claim-manifest.json"
DEFAULT_PWQ_MANIFEST = Path(__file__).resolve().parent / "pwq-kb-manifest.json"
DEFAULT_WS = "agent-team"
PREAMBLE_HEADING = "(preamble)"
BODY_DELIMITER = "<!-- body -->"
CYPHER_MCP_VENV_PYTHON = REPO_ROOT / "cypher-mcp" / ".venv" / "bin" / "python3"

MARKER_RE = re.compile(r"^\s*(?:[-*]\s+|\d{1,3}\.\s+)$")

# Runs against `ws:<--ws>` (default `agent-team`) over `cypher-mcp/.venv`'s `falkordb` client, mirroring
# `cypher-mcp/server.py`'s own connection convention exactly (same env vars, same default host/
# port, a plain read query). stdin carries a JSON object {"ws": <workspace name>, "ids":
# [documentId, ...]}; stdout carries a JSON
# object mapping each id to {"title","text","status"} or null if not found. No f-string/format
# templating of host/port into the snippet -- they're read from the environment inside it, so
# there is nothing here to shell-inject.
LIVE_FETCH_SNIPPET = r'''
import json, os, sys
from falkordb import FalkorDB

host = os.environ.get("FALKORDB_HOST", "127.0.0.1")
port = int(os.environ.get("FALKORDB_PORT", "6379"))
db = FalkorDB(host=host, port=port)
request = json.loads(sys.stdin.read())
graph = db.select_graph("ws:" + request["ws"])
ids = request["ids"]
out = {}
for doc_id in ids:
    res = graph.ro_query(
        "MATCH (d:Document {documentId: $id}) RETURN d.title AS title, d.text AS text, "
        "d.status AS status",
        params={"id": doc_id},
    )
    if res.result_set:
        title, text, status = res.result_set[0]
        out[doc_id] = {"title": title, "text": text, "status": status}
    else:
        out[doc_id] = None
sys.stdout.write(json.dumps(out))
'''


@dataclass
class Claim:
    document_id: str
    title: str = ""


@dataclass
class ClaimOutcome:
    claim: Claim
    status: str  # "found" | "not_found" | "missing_document"
    span: tuple[int, int] | None = None
    diagnostic: str = ""


@dataclass
class Report:
    outcomes: list[ClaimOutcome] = field(default_factory=list)
    overlaps: list[tuple[ClaimOutcome, ClaimOutcome]] = field(default_factory=list)
    gaps: list[tuple[int, int, str]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        bad = any(o.status != "found" for o in self.outcomes)
        return not bad and not self.overlaps and not self.gaps


# --------------------------------------------------------------------------------------------
# Locating a claim's text inside a source body
# --------------------------------------------------------------------------------------------


def build_normalized(body: str) -> tuple[str, list[int]]:
    """Collapse every run of whitespace (including a newline placed at a different word than the
    source's own hard-wrap) into a single space, and return `(normalized, index_map)` where
    `index_map[k]` is `normalized[k]`'s offset in the ORIGINAL `body` (and `index_map[len(...)]`
    is `len(body)`, a sentinel for "end of match"). Needed because a claim's text is not always a
    literal, line-break-for-line-break substring of its source: real data (`coordination-
    techniques.md`) shows a claim independently re-wrapped at a different column than the
    source's own hard-wrap, with every word identical -- a pure formatting difference, not lost
    or altered content, and a naive exact-substring search flags it as NOT_FOUND regardless.
    Normalizing whitespace this way keeps the match exact on every non-whitespace character
    (still catches a real dropped word, truncation, or altered character) while no longer being
    sensitive to *where* a soft line-wrap happens to fall."""
    out_chars: list[str] = []
    index_map: list[int] = []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c.isspace():
            j = i
            while j < n and body[j].isspace():
                j += 1
            out_chars.append(" ")
            index_map.append(i)
            i = j
        else:
            out_chars.append(c)
            index_map.append(i)
            i += 1
    index_map.append(n)
    return "".join(out_chars), index_map


def locate_claim(body: str, text: str) -> tuple[int, int] | None:
    """Find `text` inside `body` under whitespace-normalized exact matching (see
    `build_normalized`), then map the match back to `body`'s own original character offsets.
    `text` is found this way whether or not a bullet-per-claim split stripped a leading
    `- `/`* `/`N. ` list marker ahead of it, since the marker-free text is trivially still a
    substring of the marker-prefixed original. What differs is the reported span: if the text
    immediately follows (nothing but the marker itself) the start of its own original line, the
    marker is absorbed into the span too -- its removal is expected migration behavior, not lost
    content, so it counts as accounted-for rather than showing up as a bogus gap."""
    norm_body, index_map = build_normalized(body)
    norm_text, _ = build_normalized(text.strip())
    norm_idx = norm_body.find(norm_text)
    if norm_idx == -1:
        return None
    idx = index_map[norm_idx]
    end = index_map[norm_idx + len(norm_text)]
    line_start = body.rfind("\n", 0, idx) + 1
    prefix = body[line_start:idx]
    if MARKER_RE.match(prefix):
        return line_start, end
    return idx, end


def diagnose_not_found(body: str, text: str) -> str:
    """`text` isn't a match anywhere, even whitespace-normalized -- a genuine character-level
    difference. Anchor on the longest common run between the two normalized forms, then render a
    diff of that region so a single altered character (the ASCII-for-unicode slip) is visible,
    not just "not found"."""
    norm_body, _ = build_normalized(body)
    norm_text, _ = build_normalized(text.strip())
    matcher = difflib.SequenceMatcher(None, norm_body, norm_text, autojunk=False)
    match = matcher.find_longest_match(0, len(norm_body), 0, len(norm_text))
    if match.size < 20:
        return (
            "no significant overlap with the source body at all -- likely the wrong "
            "heading/file, a fabricated claim, or content copied from elsewhere entirely"
        )
    region_start = max(0, match.a - match.b)
    region_end = min(len(norm_body), region_start + len(norm_text) + 40)
    region = norm_body[region_start:region_end]
    # Word-level diff, not line-level: both sides are whitespace-normalized (no real newlines
    # left to split on), and a word-level diff isolates exactly which token(s) differ -- the
    # useful signal for an ASCII-for-unicode slip or a dropped/changed word.
    diff = list(difflib.ndiff(region.split(" "), norm_text.split(" ")))
    changed = [line for line in diff if line.startswith(("+ ", "- "))]
    return "\n".join(changed) if changed else "(near-exact but no word-level diff rendered -- inspect manually)"


# --------------------------------------------------------------------------------------------
# The core check
# --------------------------------------------------------------------------------------------


def check_partition(body: str, claims: list[Claim], fetch) -> Report:
    report = Report()
    spans: list[tuple[int, int, ClaimOutcome]] = []

    for claim in claims:
        doc = fetch(claim.document_id)
        if doc is None:
            report.outcomes.append(
                ClaimOutcome(
                    claim=claim,
                    status="missing_document",
                    diagnostic="get_document/live fetch returned nothing for this id",
                )
            )
            continue
        text = doc["text"]
        if not text.strip():
            # `locate_claim` would normalize this to "" and `str.find("")` returns 0
            # unconditionally, silently reporting a zero-length span as "found" -- an
            # ingestion-bug shape (a document with genuinely empty/whitespace-only stored text)
            # deserves its own flagged outcome, not a free pass into the clean bucket.
            report.outcomes.append(
                ClaimOutcome(
                    claim=claim,
                    status="empty_text",
                    diagnostic="claim text is empty or whitespace-only",
                )
            )
            continue
        loc = locate_claim(body, text)
        if loc is None:
            outcome = ClaimOutcome(
                claim=claim, status="not_found", diagnostic=diagnose_not_found(body, text)
            )
            report.outcomes.append(outcome)
            continue
        outcome = ClaimOutcome(claim=claim, status="found", span=loc)
        report.outcomes.append(outcome)
        spans.append((loc[0], loc[1], outcome))

    # Proper interval-overlap check, not just adjacent-in-sort-order pairs: a wide span can
    # nest two narrower, non-adjacent ones (wide=(0,60), narrow1=(10,30), narrow2=(40,60)) --
    # comparing only consecutive pairs after sorting misses the wide<->narrow2 overlap entirely
    # (analyst review, kb-content-loss-checker.md, Major finding 1 / Appendix A). For each span
    # (sorted by start), compare against every later span until one starts at or after the
    # current span's end -- correct and still cheap, since every later span's start is
    # non-decreasing once sorted, so the break point is valid.
    spans.sort(key=lambda s: s[0])
    for i in range(len(spans)):
        start_i, end_i, outcome_i = spans[i]
        for j in range(i + 1, len(spans)):
            start_j, _, outcome_j = spans[j]
            if start_j >= end_i:
                break
            report.overlaps.append((outcome_i, outcome_j))

    covered = bytearray(len(body))
    for start, end, _ in spans:
        for i in range(start, end):
            covered[i] = 1

    i, n = 0, len(body)
    while i < n:
        if covered[i]:
            i += 1
            continue
        j = i
        while j < n and not covered[j]:
            j += 1
        gap_text = body[i:j]
        if gap_text.strip():
            report.gaps.append((i, j, gap_text))
        i = j

    return report


# --------------------------------------------------------------------------------------------
# Source-file section handling (reuses flag_split_candidates.py's own `## `-section parser)
# --------------------------------------------------------------------------------------------


def load_sections(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    return dict(fsc.parse_sections(text))


def combined_body(sections: dict[str, str], headings: list[str] | None) -> str:
    if headings is None:
        chosen = list(sections.values())
    else:
        chosen = []
        for h in headings:
            if h not in sections:
                raise SystemExit(
                    f"error: heading not found in source file: {h!r}\n"
                    f"available headings: {list(sections.keys())}"
                )
            chosen.append(sections[h])
    return "\n\n".join(chosen)


# --------------------------------------------------------------------------------------------
# Manifest handling
# --------------------------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def claims_from_manifest_file_entry(manifest: dict, source_path: str) -> dict[str, list[Claim]]:
    """Primary shape: manifest["files"][source_path]["headings"] -> {heading: [Claim, ...]}."""
    files = manifest.get("files", {})
    entry = files.get(source_path)
    if entry is None:
        raise SystemExit(
            f"error: {source_path!r} not found under manifest[\"files\"] -- "
            f"known files: {list(files.keys())}\n"
            "(if this file's manifest entry is one of the irregular flat-key ones, use "
            "--manifest-flat-key or --documents instead -- see this script's module docstring)"
        )
    result: dict[str, list[Claim]] = {}
    for heading_entry in entry.get("headings", []):
        heading = heading_entry["heading"]
        claims = [
            Claim(document_id=c["documentId"], title=c.get("title", ""))
            for c in heading_entry.get("claims", [])
        ]
        result[heading] = claims
    return result


def claims_from_manifest_flat_key(manifest: dict, key: str) -> list[Claim]:
    entry = manifest.get(key)
    if entry is None:
        raise SystemExit(f"error: manifest has no top-level key {key!r}")
    ids = entry.get("documentIds")
    if ids is None:
        raise SystemExit(
            f"error: manifest[{key!r}] has no flat \"documentIds\" list (it uses some other, "
            "irregular shape -- inspect it and use --documents to supply the id list manually "
            "instead of guessing at its structure)"
        )
    return [Claim(document_id=i) for i in ids]


# --------------------------------------------------------------------------------------------
# Fetching document text: --dump (offline) or --live (direct FalkorDB)
# --------------------------------------------------------------------------------------------


def make_dump_fetcher(dump_path: Path):
    data = json.loads(dump_path.read_text(encoding="utf-8"))

    def fetch(document_id: str):
        return data.get(document_id)

    return fetch


def make_live_fetcher(all_ids: list[str], ws: str = DEFAULT_WS):
    if not CYPHER_MCP_VENV_PYTHON.is_file():
        raise SystemExit(
            f"error: --live needs {CYPHER_MCP_VENV_PYTHON} to exist -- run `cypher-mcp/setup.sh` "
            "once first (see cypher-mcp/README.md), or use --dump with a pre-fetched JSON file "
            "instead."
        )
    proc = subprocess.run(
        [str(CYPHER_MCP_VENV_PYTHON), "-c", LIVE_FETCH_SNIPPET],
        input=json.dumps({"ws": ws, "ids": sorted(set(all_ids))}),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise SystemExit(f"error: live fetch failed:\n{proc.stderr}")
    data = json.loads(proc.stdout)

    def fetch(document_id: str):
        return data.get(document_id)

    return fetch


# --------------------------------------------------------------------------------------------
# `--stored-format pwq` (docs/plans/python-web-quirks-kb.md D3/D4): the python-web-quirks copy
# --------------------------------------------------------------------------------------------

PREAMBLE_FENCE = "<!-- preamble -->"
NOTE_FENCE = "<!-- preamble-note -->"
UNIT_FENCE_RE = re.compile(r"^<!-- unit (\S+) -->$")


def strip_pwq_header(text: str) -> str:
    """Return the stored text after the first line equal to `<!-- body -->` (D3). Raises
    ValueError if no such line exists -- a header without its delimiter is a defect."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == BODY_DELIMITER:
            return "\n".join(lines[i + 1 :])
    raise ValueError(f"no {BODY_DELIMITER!r} delimiter line")


def split_parts(body: str) -> list[tuple[str, str]]:
    """Split a stripped body at its `## ` lines into (heading, text) parts; the heading line is
    excluded from the text, exactly as `parse_sections` does for the source."""
    return fsc.parse_sections(body)


def strip_front_matter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 3)
        if end != -1:
            return text[end + 5 :]
    return text


def source_preamble(text: str) -> str:
    """Source text between the front matter and the first `## ` heading."""
    body = strip_front_matter(text)
    m = fsc.HEADING_RE.search(body)
    return body[: m.start()] if m else body


def parse_copy(copy_text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (preamble block text, [(unit id, unit text)]) per the D3 preamble grammar. The
    unit text is what was ingested (everything after the fence line). ValueError when the
    `<!-- preamble -->` fence is absent."""
    block: str | None = None
    units: list[tuple[str, str]] = []
    current: list[str] | None = None
    unit_id = ""
    kind = ""

    def close() -> None:
        nonlocal block
        if current is None:
            return
        text = "\n".join(current).strip("\n")
        if kind == "preamble":
            block = text
        elif kind == "unit":
            units.append((unit_id, text + "\n"))

    for line in copy_text.split("\n"):
        m = UNIT_FENCE_RE.match(line)
        if line == PREAMBLE_FENCE or line == NOTE_FENCE or m:
            close()
            current = []
            kind = "preamble" if line == PREAMBLE_FENCE else "note" if line == NOTE_FENCE else "unit"
            unit_id = m.group(1) if m else ""
        elif current is not None:
            current.append(line)
    close()
    if block is None:
        raise ValueError(f"the copy has no {PREAMBLE_FENCE!r} fence")
    return block, units


def _issue(report: Report, ident: str, title: str, status: str, diagnostic: str) -> None:
    report.outcomes.append(
        ClaimOutcome(claim=Claim(document_id=ident, title=title), status=status, diagnostic=diagnostic)
    )


def check_pwq(
    source_text: str,
    copy_text: str,
    manifest: dict,
    fetch,
    source_format: str = "source",
) -> list[tuple[str, Report]]:
    """Run the D4 checks; one (label, Report) per scope: the preamble, the heading-set
    completeness, each source heading, and any unit that could not be read. `fetch(documentId)`
    returns {"text": ...} or None."""
    block, copy_units = parse_copy(copy_text)
    if source_format == "copy":
        sections = {}
        for _, unit_text in copy_units:
            for heading, text in split_parts(strip_pwq_header(unit_text)):
                sections[heading] = text
        pre_source = block
    else:
        sections = dict(fsc.parse_sections(source_text))
        pre_source = source_preamble(source_text)
    reports: list[tuple[str, Report]] = []

    # 3. preamble: the copy block is a substring; the rest must equal a waived span exactly
    waived = {w.strip() for w in manifest.get("waivedSpans", [])}
    pre = check_partition(
        pre_source, [Claim(document_id="(copy preamble block)", title=PREAMBLE_HEADING)],
        lambda _id: {"text": block},
    )
    pre.gaps = [g for g in pre.gaps if g[2].strip() not in waived]
    reports.append((PREAMBLE_HEADING, pre))

    # 4. completeness: manifest heading set == source heading set (+ the preamble, checked above)
    listed: set[str] = set()
    for unit in manifest.get("units", []):
        listed.update(unit.get("headings", []))
    complete = Report()
    for heading in sections:
        if heading not in listed:
            _issue(complete, "(manifest)", heading, "unlisted_heading",
                   "source heading is in no manifest unit")
    for heading in sorted(listed - set(sections)):
        _issue(complete, "(manifest)", heading, "unknown_heading",
               "manifest lists a heading the source does not have")
    reports.append(("heading set", complete))

    # 1+2. each unit: strip the header, split into parts, one claim per part
    stored: dict[str, str] = {}
    claims_by_heading: dict[str, list[Claim]] = {h: [] for h in sections}
    unit_problems = Report()
    for unit in manifest.get("units", []):
        doc_id, title = unit["documentId"], unit.get("title", unit.get("id", ""))
        doc = fetch(doc_id)
        if doc is None:
            unit_problems.outcomes.append(ClaimOutcome(
                claim=Claim(document_id=doc_id, title=title), status="missing_document",
                diagnostic="get_document/live fetch returned nothing for this id"))
            continue
        try:
            parts = split_parts(strip_pwq_header(doc["text"]))
        except ValueError as exc:
            _issue(unit_problems, doc_id, title, "bad_header", str(exc))
            continue
        if {h for h, _ in parts} != set(unit.get("headings", [])):
            _issue(unit_problems, doc_id, title, "heading_mismatch",
                   f"stored parts {[h for h, _ in parts]} != manifest headings {unit.get('headings')}")
        for heading, text in parts:
            part_id = f"{doc_id}#{heading}"
            if heading not in sections:
                _issue(unit_problems, part_id, heading, "not_found",
                       "part heading is not a source heading")
                continue
            stored[part_id] = text
            claims_by_heading[heading].append(Claim(document_id=part_id, title=heading))
    if unit_problems.outcomes:
        reports.append(("units", unit_problems))
    for heading, body in sections.items():
        reports.append((heading, check_partition(body, claims_by_heading[heading],
                                                 lambda cid: {"text": stored[cid]})))
    return reports


def _main_pwq(args) -> int:
    if args.copy is None:
        print("error: --stored-format pwq needs --copy <path>", file=sys.stderr)
        return 2
    manifest_path = args.manifest or DEFAULT_PWQ_MANIFEST
    try:
        manifest = load_manifest(manifest_path)
        copy_text = args.copy.read_text(encoding="utf-8")
        source_text = args.source.read_text(encoding="utf-8")
        ids = [u["documentId"] for u in manifest.get("units", [])]
        fetch = make_dump_fetcher(args.dump) if args.dump else make_live_fetcher(ids, args.ws)
        reports = check_pwq(source_text, copy_text, manifest, fetch, args.source_format)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    dirty = 0
    for label, report in reports:
        print(render_report(label, report))
        dirty += 0 if report.clean else 1
    print(f"\n{len(reports)} scope(s) checked, {dirty} with findings.", file=sys.stderr)
    return 1 if dirty else 0


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------


def render_report(label: str, report: Report) -> str:
    lines = [f"== {label} =="]
    if report.clean:
        lines.append(f"clean: {len(report.outcomes)} claim(s), fully accounted, no overlaps.")
        return "\n".join(lines)
    for outcome in report.outcomes:
        if outcome.status == "found":
            continue
        title = outcome.claim.title or "(no title)"
        lines.append(f"[{outcome.status.upper()}] {outcome.claim.document_id} — {title}")
        if outcome.diagnostic:
            for diag_line in outcome.diagnostic.splitlines():
                lines.append(f"    {diag_line}")
    for prev, cur in report.overlaps:
        lines.append(
            f"[DUPLICATE] {prev.claim.document_id} ({prev.claim.title!r}) and "
            f"{cur.claim.document_id} ({cur.claim.title!r}) both claim the same source span "
            f"{prev.span} / {cur.span}"
        )
    for start, end, gap_text in report.gaps:
        preview = gap_text.strip().replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "…"
        lines.append(f"[UNACCOUNTED] source[{start}:{end}] not claimed by anything: {preview!r}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify no content was lost/altered/duplicated migrating a KB markdown file's "
            "section(s) into ws:agent-team claim documents (K-030 Stage 6). Read-only; never "
            "writes or decides a split boundary. Run with --help for the full method."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source", type=Path, help="Source KB markdown file.")
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Manifest path (default: kb-claim-manifest.json; with --stored-format pwq: "
        "pwq-kb-manifest.json).",
    )
    parser.add_argument(
        "--stored-format", choices=["default", "pwq"], default="default",
        help="`pwq`: stored texts are python-web-quirks units (header up to `<!-- body -->`, one "
        "or more `## ` parts); checks parts per section, the preamble (needs --copy) and that "
        "the manifest's heading set equals the source's.",
    )
    parser.add_argument(
        "--copy", type=Path,
        help="pwq mode: the copy of record; its `<!-- preamble -->` block is checked against "
        "the source preamble (manifest waivedSpans cover the rest).",
    )
    parser.add_argument(
        "--source-format", choices=["source", "copy"], default="source",
        help="pwq mode: `copy` builds the source side from --copy's unit bodies (AC-5) instead "
        "of the source file.",
    )
    parser.add_argument(
        "--ws", default=DEFAULT_WS,
        help="--live: the workspace to read, graph key `ws:<name>` (default: agent-team).",
    )
    # Mutually exclusive: if both were accepted, one would silently win with no warning (analyst
    # review, Nit 1) -- argparse itself is the cheapest place to refuse the ambiguous combination.
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument(
        "--manifest-flat-key",
        help="Read manifest[KEY]['documentIds'] as a flat whole-file claim list, instead of the "
        "primary per-heading manifest['files'][...] shape.",
    )
    scope_group.add_argument(
        "--documents",
        nargs="+",
        metavar="ID",
        help="Bypass the manifest entirely: check this explicit documentId list against the "
        "whole source file (or --headings subset).",
    )
    parser.add_argument(
        "--headings",
        nargs="+",
        metavar="HEADING",
        help="Restrict a --manifest-flat-key or --documents run to these exact `## ` heading(s) "
        "only, instead of the whole file (e.g. for a partially-migrated file).",
    )
    fetch_group = parser.add_mutually_exclusive_group(required=True)
    fetch_group.add_argument(
        "--dump", type=Path, help="JSON file {documentId: {title, text}}, pre-fetched offline."
    )
    fetch_group.add_argument(
        "--live", action="store_true", help="Fetch documents directly from FalkorDB."
    )
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"error: not a file: {args.source}", file=sys.stderr)
        return 2

    if args.stored_format == "pwq":
        return _main_pwq(args)
    if args.copy is not None or args.source_format != "source":
        print("error: --copy/--source-format apply only with --stored-format pwq", file=sys.stderr)
        return 2
    args.manifest = args.manifest or DEFAULT_MANIFEST

    sections = load_sections(args.source)
    if not sections:
        print(f"error: no `## ` sections found in {args.source}", file=sys.stderr)
        return 2

    manifest = None
    if args.documents is None:
        manifest = load_manifest(args.manifest)

    # Resolve which repo-relative path key the manifest uses (it stores paths like
    # "claude/graph-dba/falkordb-reference.md" -- relative to the repo root).
    try:
        source_key = str(args.source.resolve().relative_to(REPO_ROOT))
    except ValueError:
        source_key = str(args.source)

    reports: list[tuple[str, Report]] = []
    all_ids: list[str] = []

    if args.documents is not None:
        if args.headings is None:
            print(
                f"note: no --headings given -- checking against ALL {len(sections)} heading(s) "
                f"in {source_key}. If this id list covers only some of them (a partial "
                "migration), every unmigrated heading's whole body will show up as an "
                "UNACCOUNTED gap -- expected noise, not necessarily a real defect.",
                file=sys.stderr,
            )
        claims = [Claim(document_id=i) for i in args.documents]
        body = combined_body(sections, args.headings)
        all_ids.extend(c.document_id for c in claims)
        pending = [("whole file (manual --documents list)", body, claims)]
    elif args.manifest_flat_key is not None:
        if args.headings is None:
            print(
                f"note: no --headings given -- checking against ALL {len(sections)} heading(s) "
                f"in {source_key}. If manifest[{args.manifest_flat_key!r}]'s id list covers only "
                "some of them (a partial migration), every unmigrated heading's whole body will "
                "show up as an UNACCOUNTED gap -- expected noise, not necessarily a real defect.",
                file=sys.stderr,
            )
        claims = claims_from_manifest_flat_key(manifest, args.manifest_flat_key)
        body = combined_body(sections, args.headings)
        all_ids.extend(c.document_id for c in claims)
        pending = [(f"whole file (manifest key {args.manifest_flat_key!r})", body, claims)]
    else:
        per_heading = claims_from_manifest_file_entry(manifest, source_key)
        headings = args.headings if args.headings else list(per_heading.keys())
        pending = []
        for heading in headings:
            if heading not in per_heading:
                print(f"error: no manifest claims recorded for heading: {heading!r}", file=sys.stderr)
                return 2
            if heading not in sections:
                print(f"error: heading in manifest but not in source file: {heading!r}", file=sys.stderr)
                return 2
            claims = per_heading[heading]
            all_ids.extend(c.document_id for c in claims)
            pending.append((heading, sections[heading], claims))

    if args.dump:
        fetch = make_dump_fetcher(args.dump)
    else:
        fetch = make_live_fetcher(all_ids, args.ws)

    any_dirty = False
    for label, body, claims in pending:
        report = check_partition(body, claims, fetch)
        reports.append((label, report))
        print(render_report(label, report))
        if not report.clean:
            any_dirty = True

    total_claims = sum(len(r.outcomes) for _, r in reports)
    dirty_sections = sum(1 for _, r in reports if not r.clean)
    print(
        f"\n{len(reports)} section(s)/scope(s) checked, {total_claims} claim(s) total, "
        f"{dirty_sections} with findings.",
        file=sys.stderr,
    )
    return 1 if any_dirty else 0


if __name__ == "__main__":
    raise SystemExit(main())
