#!/usr/bin/env python3
"""The `cypher` MCP server — one tool, `query`, over a named FalkorDB graph.

Design and rationale: ``docs/plans/cpg-query-access.md`` §4.4 (the frozen tool
contract) and ``docs/plans/generic-cypher-mcp.md`` (the write path below). The
short version:

* **One tool, two required parameters** (``graph``, ``cypher``) **plus one
  optional** (``agent``) — FR-2/FR-4 of the original contract, widened by
  ``generic-cypher-mcp.md`` FR-1. No ``limit``, no ``params``, no ``mode``;
  every other knob is an environment variable.
* **Read-only via ``GRAPH.RO_QUERY`` — still the whole read path, unchanged.**
  That is what bounds the blast radius on an instance that also holds non-CPG
  graphs: writes are rejected server-side, *and* a typo'd graph name cannot
  materialise an empty key (plain ``GRAPH.QUERY`` would create one). Safe by
  construction, not by validation.
* **A write is a distinct, narrower path, gated by ``authorize_write()``.**
  ``GRAPH.RO_QUERY`` rejecting a statement (or FalkorDB reporting "empty key"
  for a plausibly-first write, e.g. the one-time kaizen-graph migration) is
  what routes a call into enforcement — never the mere presence of the
  ``agent`` argument. Six write shapes are ever authorized: the legacy
  author-write (an agent creating its own ``:KaizenEntry`` via a literal
  ``author:`` match), the current producer-write (any agent, its own
  ``MERGE (:Agent {agentId:...}) + CREATE (...)-[:PRODUCED]->(...:KaizenEntry
  {...})``), a curator MENTIONS-write, two curator edge-resolve shapes
  (``PRODUCED``/``MENTIONS``), and the curator full-node clear by ``entryId``.
  See ``docs/plans/generic-cypher-mcp.md`` §3.1/§3.2 (the original two shapes)
  and ``docs/plans/kaizen-agent-ontology.md`` §3.1/§3.1a (the four added for
  the ``:Agent``/``PRODUCED``/``MENTIONS`` ontology, plus the cross-clause
  smuggling closure) for the full design and rationale.
* **``EXPLAIN`` is honoured, ``PROFILE`` is actively refused.** Raw FalkorDB
  *ignores* both prefixes and executes the query, so a passive drop would return
  results to a caller that asked for a profile — a wrong answer rather than an
  error. ``GRAPH.PROFILE`` additionally executes writes, which is why it is not
  offered at all. The ``EXPLAIN`` path is preceded by a ``GRAPH.LIST`` check
  because ``GRAPH.EXPLAIN`` is not a read-only command and would otherwise
  materialise a key for a mistyped graph name.
* **Display-only truncation.** The full result set is materialised before
  formatting; the caps below shape the *rendering*, so the reported row count is
  exact below FalkorDB's ``RESULTSET_SIZE`` (default 10000), at or above which
  it is itself a cap.

Operational constraints this module obeys, all load-bearing for a stdio server:
importing it must not connect; nothing is ever written to stdout (the transport
owns it — diagnostics go to stderr); and the tool never raises, because a stdio
server that dies mid-session is not reconnected by the harness.
"""

from __future__ import annotations

import os
import re
import sys
import time

from falkordb import FalkorDB
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

# --------------------------------------------------------------------------
# Configuration (environment only — FR-2 forbids extra tool parameters)
# --------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back on anything odd.

    A bad value must not take the server down at import time: an unparseable or
    non-positive override is reported on stderr and the default is used.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        _log(f"{name}={raw!r} is not an integer; using {default}")
        return default
    if value <= 0:
        _log(f"{name}={value} must be positive; using {default}")
        return default
    return value


def _log(message: str) -> None:
    """Diagnostics go to stderr only — stdout carries the MCP protocol."""
    print(f"[cypher-mcp] {message}", file=sys.stderr, flush=True)


FALKORDB_HOST = os.environ.get("FALKORDB_HOST", "127.0.0.1")
FALKORDB_PORT = _env_int("FALKORDB_PORT", 6379)

MAX_ROWS = _env_int("CYPHER_MCP_MAX_ROWS", 200)
MAX_CELL = _env_int("CYPHER_MCP_MAX_CELL", 300)
MAX_CHARS = _env_int("CYPHER_MCP_MAX_CHARS", 30000)
TIMEOUT_MS = _env_int("CYPHER_MCP_TIMEOUT_MS", 30000)

#: Agents allowed to run the curator-clear write shape (FR-8's "curator"
#: capability — clearing an entry it did not author, e.g. `cobb`'s
#: distillation workflow). Comma-separated; defaults to `cobb`, the only
#: curator this pilot names (`docs/plans/generic-cypher-mcp.md` §3.2).
CURATOR_AGENTS = frozenset(
    a.strip() for a in os.environ.get("CYPHER_MCP_CURATOR_AGENTS", "cobb").split(",") if a.strip()
)

#: Declared to the harness as ``_meta["anthropic/maxResultSizeChars"]``. Claude
#: Code otherwise estimates a *token* budget and, above it, persists the result
#: to disk and replaces it with a file reference — which would swallow the
#: truncation notice on exactly the runs that are truncated. Pinning it in the
#: same unit as our own cap (with 2x headroom for the stats/header/notice lines)
#: keeps the two from ever disagreeing. Hard ceiling: 500 000 chars.
MAX_RESULT_SIZE_CHARS = min(2 * MAX_CHARS, 500_000)

# --------------------------------------------------------------------------
# Fixed strings (frozen by the plan — tests pin them)
# --------------------------------------------------------------------------

TOOL_DESCRIPTION = (
    "Run OpenCypher against a named FalkorDB graph — not limited to cpg_* graphs; any graph "
    "key on this instance is reachable (e.g. cpg_<component>, or kaizen_team). `graph` "
    "is the graph key (caller-supplied); `cypher` is the query text, sent verbatim — "
    "multi-line welcome, no shell quoting. Reads run unrestricted via GRAPH.RO_QUERY. A "
    "write additionally requires the optional `agent` parameter (your agent slug) and is "
    "authorized only in 6 shapes: a producer-write (MERGE (:Agent {agentId:...}) + CREATE "
    "(...)-[:PRODUCED]->(...:KaizenEntry {...}) for one's own agentId), the legacy author-write "
    "(a CREATE map literal with a matching `author:` value), a curator MENTIONS-write, two "
    "curator edge-resolve shapes (PRODUCED/MENTIONS), or a curator full-node clear by `entryId` "
    "— every other write is rejected. Prefix "
    "EXPLAIN for a query plan; PROFILE is not supported (it executes the query). FalkorDB "
    "is OpenCypher: no APOC, no GDS. CPG schema: "
    "skills/joern-cpg/references/cpg-model.md."
)

SERVER_INSTRUCTIONS = (
    "The `cypher` server exposes a single tool, `query`: OpenCypher against a named FalkorDB "
    "graph — not limited to cpg_* graphs; typically a Joern Code Property Graph loaded as "
    "`cpg_<component>`, but any graph key on this instance is reachable, e.g. "
    "`kaizen_team` (the team's kaizen working memory: :Agent nodes, PRODUCED/MENTIONS edges). "
    "Use it to answer call-graph, data-flow, impact-analysis and test-gap questions about a "
    "codebase without reading files, or to read/write the team's kaizen entries. Graph names are "
    "always supplied by the caller; a query against an unknown graph answers with the list of "
    "loaded graphs. Reads need no `agent` and are unrestricted. A write additionally requires "
    "`agent` (the caller's agent slug) and is authorized only in 6 shapes: any agent's own "
    "producer-write (MERGE (:Agent {agentId:...}) + CREATE (...)-[:PRODUCED]->(...:KaizenEntry "
    "{...})), the legacy author-write (a CREATE map literal with a matching `author:` value), or "
    "a recognized curator agent (`CYPHER_MCP_CURATOR_AGENTS`, default `cobb`) running a MENTIONS-"
    "write, a PRODUCED/MENTIONS edge-resolve, or a full-node clear by `entryId` (`MATCH "
    "(e:KaizenEntry {entryId:'...'}) DETACH DELETE e`) — every other write shape, and any "
    "attribution mismatch, is rejected."
)

PROFILE_REFUSAL = (
    "PROFILE is not available through this tool: GRAPH.PROFILE executes the query, "
    "including writes. Use EXPLAIN for the plan. For measured profiling use the fallback: "
    "redis-cli -p 6379 GRAPH.PROFILE <graph> '<cypher>' --no-raw."
)

SCHEMA_POINTER = (
    "FalkorDB is OpenCypher (no APOC/GDS); CPG property keys are UPPER_CASE — see "
    "skills/joern-cpg/references/cpg-model.md."
)

# --------------------------------------------------------------------------
# Directive sniffing (plan §4.4 D5a) — classification only, never a rewrite
# --------------------------------------------------------------------------

#: ``str.lstrip()`` semantics, spelled out so the scan below stays explicit.
_WHITESPACE = " \t\r\n\f\v"

#: ``\b`` matters: ``EXPLAIN_ME`` / ``PROFILER`` / ``PROFILEDATA`` are ordinary
#: identifiers and must classify as plain queries, while ``EXPLAIN(``,
#: ``EXPLAIN\n`` and ``explain\tMATCH`` are directives.
_DIRECTIVE_RE = re.compile(r"(EXPLAIN|PROFILE)\b", re.IGNORECASE)


def _scan_leading_trivia(cypher: str) -> int:
    """Return the index of the first character that is not leading trivia.

    Trivia is whitespace, ``//``-to-end-of-line comments and ``/* … */`` blocks,
    in any order and any number. An **unterminated** ``/*`` stops the scan where
    it starts: the statement is malformed, so it classifies as a plain query and
    FalkorDB gets to return its own error verbatim.

    The scan stops at the first real character, so comment markers *inside* a
    string literal later in the query are never seen — no lexer is needed.
    """
    i = 0
    n = len(cypher)
    while i < n:
        j = i
        while j < n and cypher[j] in _WHITESPACE:
            j += 1
        if cypher.startswith("//", j):
            newline = cypher.find("\n", j + 2)
            j = n if newline == -1 else newline + 1
        elif cypher.startswith("/*", j):
            # Cypher block comments do not nest — match the first `*/`.
            end = cypher.find("*/", j + 2)
            if end == -1:
                return j
            j = end + 2
        if j == i:  # nothing consumed this round: we are at real text
            return i
        i = j
    return i


def split_directive(cypher: str) -> tuple[str, str]:
    """Classify a statement as ``"query"``, ``"explain"`` or ``"profile"``.

    Returns ``(kind, cypher_to_send)``:

    * ``"query"`` — the caller's string **byte-for-byte**. The normalisation
      above is used for classification only; FR-3 promises verbatim
      transmission, and stripping comments would corrupt e.g.
      ``CONTAINS '// x'``.
    * ``"explain"`` — the text after the keyword, left-stripped. The consumed
      leading trivia is dropped; it cannot change a plan.
    * ``"profile"`` — the empty string. Nothing is ever sent.
    """
    start = _scan_leading_trivia(cypher)
    match = _DIRECTIVE_RE.match(cypher, start)
    if match is None:
        return "query", cypher
    if match.group(1).upper() == "PROFILE":
        return "profile", ""
    return "explain", cypher[match.end():].lstrip()


# --------------------------------------------------------------------------
# Write authorization (FR-8) — `docs/plans/generic-cypher-mcp.md` §3.2 (the
# original 2 shapes) and `docs/plans/kaizen-agent-ontology.md` §3.1/§3.1a (4
# more, for the :Agent/PRODUCED/MENTIONS ontology).
#
# A static, pre-execution, string-literal-aware text scan — deliberately not a
# real Cypher parser (FR-8's own stated trust bar: "well-behaved callers can't
# do this by accident," not hardened against a malicious one). Six recognized
# write shapes: the legacy author-write (an agent creating its own
# `:KaizenEntry` via a literal `author:` claim inside a
# `CREATE (...:KaizenEntry {...})` map body), the current producer-write (any
# agent, its own `MERGE (:Agent {agentId:...}) + CREATE (...)-[:PRODUCED]->
# (...:KaizenEntry {...})`), a curator MENTIONS-write, two curator edge-resolve
# shapes (PRODUCED/MENTIONS), and a recognized curator agent clearing an entry
# whole by `entryId` (`MATCH (...) DETACH DELETE`).
# --------------------------------------------------------------------------

#: Lightweight pre-classification only, used solely on the "empty key" branch
#: of `run_query()` to decide whether `agent` being supplied plausibly signals
#: write intent — never itself an authorization decision.
_WRITE_KEYWORD_RE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE)\b", re.IGNORECASE)


def _looks_like_write(cypher: str) -> bool:
    """Lightweight pre-classification only — never an authorization decision. A
    read whose free-text predicate happens to quote a write keyword (e.g. `WHERE
    e.context CONTAINS 'DETACH DELETE'`) can false-positive here; the only
    consequence is routing into `authorize_write()`, which is itself
    string-literal- and CREATE-span-aware and would reject it as
    "not a recognized write shape" rather than the more accurate "graph not
    found" — a less-precise message, never a wrong authorization. This branch
    only ever fires pre-migration or on a graph-name typo, since a populated
    `kaizen_team` never returns "empty key" again after the one-time import.
    """
    return bool(_WRITE_KEYWORD_RE.search(cypher))


# The map-KEY form only — never `.author = ...` (SET). Restricting extraction to a
# CREATE's own map literal (below) already excludes SET syntactically, since SET
# never appears inside a CREATE's `{...}` body — this pattern doesn't even need to
# recognize the SET spelling to make that true.
_AUTHOR_LITERAL_RE = re.compile(r"""\bauthor\s*:\s*['"]([^'"]*)['"]""", re.IGNORECASE)

# The ONE recognized curator-clear skeleton (`graph-dba`'s plan §3), whitespace-
# collapsed before matching. Deliberately narrow to :KaizenEntry — see the risks
# section of `docs/plans/generic-cypher-mcp.md` §9 for the revisit trigger.
_CURATOR_CLEAR_RE = re.compile(
    r"^MATCH \([a-zA-Z_]\w*:KaizenEntry \{entryId\s*:\s*['\"][^'\"]+['\"]\}\) "
    r"DETACH DELETE [a-zA-Z_]\w*;?$",
    re.IGNORECASE,
)


def _string_literal_spans(text: str) -> list[tuple[int, int]]:
    """[start, end) index ranges of every quoted string literal in `text` (single
    or double quoted, backslash-escape aware)."""
    spans, i, n, in_string, start = [], 0, len(text), None, None
    while i < n:
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == in_string:
                spans.append((start, i + 1))
                in_string = None
        elif ch in ("'", '"'):
            in_string, start = ch, i
        i += 1
    return spans


def _scan_matched_brace(text: str, open_index: int) -> int | None:
    """Given ``text[open_index] == "{"``, return the index just past the
    matching ``"}"`` (string-literal aware — a free-text field's own `{`/`}`
    or quoted decoy text can't desync it), or ``None`` if unterminated.

    Factored out of `_kaizen_entry_create_map_spans()` (plan §3.1 step 1) as a
    shared primitive — the brace-depth-counting logic is identical, only the
    caller's anchor pattern differs (a `CREATE (...:KaizenEntry {` clause vs.
    `_producer_write_agent_id`'s optional `{sessionId: ...}` map and its
    `KaizenEntry` map)."""
    depth, i, in_string, n = 1, open_index + 1, None, len(text)
    while i < n and depth > 0:
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == in_string:
                in_string = None
        elif ch in ("'", '"'):
            in_string = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return i if depth == 0 else None


def _kaizen_entry_create_map_spans(cypher: str) -> list[str]:
    """Body text of every map literal `{...}` immediately following a
    `CREATE (<var>:KaizenEntry ...)` clause. Brace-matched via
    `_scan_matched_brace()`, so a free-text field containing a literal `{`/`}`
    can't desync the match. A `SET`, `MATCH`, or `MERGE` clause simply produces no
    spans here — there is no separate "exclude SET" rule to get wrong.

    The `CREATE`-keyword *location* step is itself string-literal-aware (Pass-2
    review, M1-residual): a whole-text `_string_literal_spans` pass is computed
    first, and any `CREATE` match whose start falls inside one of those spans is
    skipped — otherwise a free-text field that happens to quote a *complete*
    `CREATE (...:KaizenEntry {...})`-shaped example (not just a bare `author:`
    fragment) would be misread as a second, independent top-level clause."""
    outer_spans = _string_literal_spans(cypher)
    spans = []
    for cm in re.finditer(r"\bCREATE\b", cypher, re.IGNORECASE):
        if any(s <= cm.start() < e for s, e in outer_spans):
            continue
        tail = cypher[cm.end():]
        m = re.match(r"\s*\(\s*[a-zA-Z_]\w*\s*:\s*KaizenEntry\s*\{", tail)
        if not m:
            continue
        body_start = cm.end() + m.end()
        end = _scan_matched_brace(cypher, body_start - 1)
        if end is None:
            continue
        spans.append(cypher[body_start:end - 1])
    return spans


def _author_claims(cypher: str) -> list[str]:
    """Every literal `author: '<value>'` found strictly inside a CREATE's own
    KaizenEntry map body, and NOT nested inside a sibling property's own string
    value (a decoy `author:`-shaped substring inside `evidence`/`context` sits
    inside THAT property's string-literal span and is excluded)."""
    claims = []
    for span in _kaizen_entry_create_map_spans(cypher):
        literal_ranges = _string_literal_spans(span)
        for m in _AUTHOR_LITERAL_RE.finditer(span):
            if not any(s <= m.start() < e for s, e in literal_ranges):
                claims.append(m.group(1))
    return claims


# --------------------------------------------------------------------------
# M8 kaizen-agent-ontology (docs/plans/kaizen-agent-ontology.md §3.1) — 4 new
# write shapes: a non-curator producer-write, and 3 curator shapes (MENTIONS-
# write, producer-edge-resolve, mention-edge-resolve).
# --------------------------------------------------------------------------

#: §3.1 step 2a — the `MERGE (<var>:Agent {agentId:'<value>'})` clause a
#: producer-write must open with, anchored at the very start of the string.
_PRODUCER_WRITE_MERGE_RE = re.compile(
    r"\A\s*MERGE\s*\(\s*([a-zA-Z_]\w*)\s*:\s*Agent\s*\{\s*agentId\s*:\s*(['\"])([^'\"]+)\2\s*\}\s*\)\s*",
    re.IGNORECASE,
)

#: §3.1 step 2d — the tail after the optional `{sessionId:...}` map: the
#: `PRODUCED` relationship pattern's close, then the `KaizenEntry` node and
#: the opening `{` of its own map.
_PRODUCER_WRITE_TAIL_RE = re.compile(
    r"\s*\]\s*->\s*\(\s*[a-zA-Z_]\w*\s*:\s*KaizenEntry\s*\{", re.IGNORECASE
)

#: §3.1 step 2e — nothing may follow the `KaizenEntry` map's closing `}` but
#: the `CREATE` clause's own node-pattern close, an optional semicolon, and
#: end of string. This is what makes the shape "exactly one `MERGE` clause
#: followed by exactly one `CREATE` clause, nothing else" — not merely
#: "contains."
_PRODUCER_WRITE_TRAILER_RE = re.compile(r"\s*\)\s*;?\s*\Z")


def _producer_write_shape_match(cypher: str) -> tuple[str, int] | None:
    """Match the producer-write skeleton up through a valid `KaizenEntry` map
    — `MERGE (<var>:Agent {agentId:...})` immediately followed by
    `CREATE (<var>)-[:PRODUCED {...}]->(<var2>:KaizenEntry {...})`, with the
    map's own braces balanced — WITHOUT checking what (if anything) follows
    the map's closing `}` (that is `_PRODUCER_WRITE_TRAILER_RE`'s job, done
    by the caller). Returns `(agent_id, map_end)` — `map_end` is the index
    just past the `KaizenEntry` map's closing `}` — or `None` if any
    component through the map itself failed to match.

    Factored out of `_producer_write_agent_id()` so that a near-miss (shape
    matches everywhere except the trailer) can be told apart from any other
    kind of non-match, for a more specific rejection message (`authorize_write`,
    FR-8) — this helper is pure shape recognition and asserts nothing about
    authorization on its own."""
    merge_match = _PRODUCER_WRITE_MERGE_RE.match(cypher)
    if not merge_match:
        return None
    var, agent_id = merge_match.group(1), merge_match.group(3)

    create_re = re.compile(
        r"CREATE\s*\(\s*" + re.escape(var) + r"\s*\)\s*-\s*\[\s*:\s*PRODUCED\s*",
        re.IGNORECASE,
    )
    create_match = create_re.match(cypher, merge_match.end())
    if not create_match:
        return None
    pos = create_match.end()

    # Optional `{sessionId: ...}` property map on the `PRODUCED` relationship
    # (step 2c) — brace-depth-matched, string-literal aware, same as the
    # KaizenEntry map below. Anything other than `{` here (e.g. `]`) means
    # the map was omitted, exactly per the template's own "or omit this key".
    if pos < len(cypher) and cypher[pos] == "{":
        end = _scan_matched_brace(cypher, pos)
        if end is None:
            return None
        pos = end

    tail_match = _PRODUCER_WRITE_TAIL_RE.match(cypher, pos)
    if not tail_match:
        return None
    map_open = tail_match.end() - 1  # index of the KaizenEntry map's own '{'
    map_end = _scan_matched_brace(cypher, map_open)
    if map_end is None:
        return None

    return agent_id, map_end


def _producer_write_agent_id(cypher: str) -> str | None:
    """Recognize exactly one `MERGE (<var>:Agent {agentId:...})` clause
    immediately followed by exactly one
    `CREATE (<var>)-[:PRODUCED {...}]->(<var2>:KaizenEntry {...})` clause,
    nothing else in the statement (plan §3.1 step 2). Returns the claimed
    `agentId`, or `None` if the text isn't shaped exactly like this skeleton
    — never raises, never partially authorizes."""
    match = _producer_write_shape_match(cypher)
    if match is None:
        return None
    agent_id, map_end = match
    if not _PRODUCER_WRITE_TRAILER_RE.match(cypher, map_end):
        return None
    return agent_id


#: Longest trailing-text excerpt shown in the near-miss trailer message
#: (`_producer_write_trailer_message`) — long enough to show the offending
#: clause, short enough to stay a one-line pointer rather than an echo of the
#: whole statement.
_PRODUCER_WRITE_TRAILER_EXCERPT_MAX = 80


def _producer_write_trailer_message(cypher: str, map_end: int) -> str:
    """The specific near-miss rejection for a statement that matches the
    producer-write skeleton through a valid `KaizenEntry` map, but continues
    afterward (a trailing `RETURN`, or any other clause) — the one thing
    `_PRODUCER_WRITE_TRAILER_RE` requires not happen (plan §3.1 step 2e,
    deliberately strict, not to be relaxed). Names the trap instead of
    falling through to the fully generic FR-8 message."""
    trailing = cypher[map_end:].strip()
    if len(trailing) > _PRODUCER_WRITE_TRAILER_EXCERPT_MAX:
        trailing = trailing[:_PRODUCER_WRITE_TRAILER_EXCERPT_MAX] + "…"
    return (
        "Rejected: this looks like a producer-write, but the statement continues "
        f"after the KaizenEntry map's closing brace (found: '{trailing}'). The "
        "producer-write shape must end there — no trailing RETURN or other clause. "
        "Issue a separate follow-up read for the entryId instead."
    )


#: The 3 new curator-gated shapes (plan §3.1) — narrow fixed skeletons,
#: matched against a whitespace-collapsed copy of the statement, same style
#: as `_CURATOR_CLEAR_RE`. The backreferences (`\1`/`\2`) enforce that the
#: second clause's variables are the *same* ones the first clause bound —
#: stricter than `_CURATOR_CLEAR_RE`, which doesn't need to be (single clause,
#: single variable, nothing to cross-reference).
_MENTIONS_WRITE_RE = re.compile(
    r"^MATCH \(([a-zA-Z_]\w*):KaizenEntry \{entryId\s*:\s*['\"][^'\"]+['\"]\}\) "
    r"MERGE \(([a-zA-Z_]\w*):Agent \{agentId\s*:\s*['\"][^'\"]+['\"]\}\) "
    r"MERGE \(\1\)-\[:MENTIONS\]->\(\2\);?$",
    re.IGNORECASE,
)
_PRODUCER_EDGE_RESOLVE_RE = re.compile(
    r"^MATCH \(:Agent\)-\[([a-zA-Z_]\w*):PRODUCED\]->\([a-zA-Z_]\w*:KaizenEntry "
    r"\{entryId\s*:\s*['\"][^'\"]+['\"]\}\) DELETE \1;?$",
    re.IGNORECASE,
)
_MENTION_EDGE_RESOLVE_RE = re.compile(
    r"^MATCH \([a-zA-Z_]\w*:KaizenEntry \{entryId\s*:\s*['\"][^'\"]+['\"]\}\)-\[([a-zA-Z_]\w*):MENTIONS\]"
    r"->\(:Agent \{agentId\s*:\s*['\"][^'\"]+['\"]\}\) DELETE \1;?$",
    re.IGNORECASE,
)

# Same keyword set as `_WRITE_KEYWORD_RE` (line 239), minus `CREATE` — every
# write keyword a *second*, unrelated clause could open with, since an
# accepted shape-1 statement's own `CREATE` is exactly what this check must
# not itself trip on.
_FOREIGN_TRIGGER_RE = re.compile(r"\b(?:MERGE|DELETE|SET|REMOVE)\b", re.IGNORECASE)


def _has_foreign_trigger_outside_strings(cypher: str) -> bool:
    """True if `cypher` contains a bare (not inside a string literal) `MERGE`,
    `DELETE`, `SET`, or `REMOVE` keyword anywhere in the statement — i.e. a
    second, different recognized shape's trigger (or an arbitrary-tampering
    `SET`/`REMOVE`) chained alongside an accepted author-write clause (Finding
    1, Pass 1 + Pass 2, docs/reviews/kaizen-agent-ontology.md)."""
    literal_ranges = _string_literal_spans(cypher)
    for m in _FOREIGN_TRIGGER_RE.finditer(cypher):
        if not any(s <= m.start() < e for s, e in literal_ranges):
            return True
    return False


def authorize_write(cypher: str, agent: str | None) -> str | None:
    """Return a curated rejection message, or None if the write is authorized."""
    if not agent:
        return (
            "Write detected but no `agent` parameter supplied. Declare the caller's "
            "identity: mcp__cypher__query(graph, cypher, agent='<your-agent-slug>')."
        )
    claims = _author_claims(cypher)
    if claims:
        mismatched = [c for c in claims if c != agent]
        if mismatched:
            return (
                f"Rejected: this write attributes an entry to author '{mismatched[0]}', "
                f"but the call declared agent='{agent}'. One agent's write cannot be "
                "accepted as another's (FR-8)."
            )
        if _has_foreign_trigger_outside_strings(cypher):
            return (
                "Rejected: this statement combines a valid author-write with another "
                "recognized shape's trigger (a bare MERGE, DELETE, SET, or REMOVE elsewhere "
                "in the same statement) — chaining an unrelated clause onto a self-attributed "
                "CREATE is not authorized, regardless of the author-write's own validity."
            )
        return None   # every author: literal found inside a CREATE:KaizenEntry body
                       # matches the declared agent, and nothing else is chained on — allowed

    producer_agent_id = _producer_write_agent_id(cypher)
    if producer_agent_id is not None:
        if producer_agent_id != agent:
            return (
                f"Rejected: this write's MERGE (:Agent {{agentId: '{producer_agent_id}'}}) "
                f"claims a different agent than the call declared (agent='{agent}'). One "
                "agent's write cannot be accepted as another's (FR-8)."
            )
        return None   # producer-write, any agent may write its own

    # Not a full producer-write — but if the skeleton matched everywhere
    # through a valid KaizenEntry map and only the trailer failed (i.e. the
    # statement continues after the map's closing brace), that specific near-
    # miss gets a specific message instead of falling through to the fully
    # generic FR-8 rejection below. Message-only: the authorization outcome
    # (rejected) is unchanged from before this check existed.
    producer_shape = _producer_write_shape_match(cypher)
    if producer_shape is not None:
        _, map_end = producer_shape
        if not _PRODUCER_WRITE_TRAILER_RE.match(cypher, map_end):
            return _producer_write_trailer_message(cypher, map_end)

    normalized = " ".join(cypher.split())
    if _CURATOR_CLEAR_RE.match(normalized):
        if agent in CURATOR_AGENTS:
            return None   # the one recognized curator-clear shape, by a curator agent
        return (
            f"Rejected: this is the curator-clear shape (MATCH ... DETACH DELETE by "
            f"entryId), but agent='{agent}' is not a recognized curator "
            f"({sorted(CURATOR_AGENTS)}). Only a curator may clear an entry it did not "
            "author."
        )

    if (
        _MENTIONS_WRITE_RE.match(normalized)
        or _PRODUCER_EDGE_RESOLVE_RE.match(normalized)
        or _MENTION_EDGE_RESOLVE_RE.match(normalized)
    ):
        if agent in CURATOR_AGENTS:
            return None
        return (
            f"Rejected: this is a curator-only write shape (MENTIONS tagging or edge "
            f"resolution), but agent='{agent}' is not a recognized curator "
            f"({sorted(CURATOR_AGENTS)})."
        )

    return (
        "Rejected: this write is neither an author-write (no literal `author: "
        f"'{agent}'` found inside a CREATE (...:KaizenEntry {{...}}) clause), a "
        "producer-write (no matching MERGE (:Agent {agentId:...}) + CREATE (...)-"
        "[:PRODUCED]->(...:KaizenEntry {...}) found), nor a recognized curator shape "
        "(MENTIONS-write, edge-resolve, or the full-node clear). This tool only "
        "authorizes those shapes (FR-8)."
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


class _ReprAsIs:
    """Wraps pre-rendered text so ``repr()`` inserts it verbatim, unquoted.

    Used by ``_normalize_for_repr`` to make a nested ``bool`` render as
    ``true``/``false`` even though it sits inside a ``dict``/``list`` whose
    own ``repr()`` will call ``repr()`` on each element in turn.
    """

    __slots__ = ("_text",)

    def __init__(self, text: str) -> None:
        self._text = text

    def __repr__(self) -> str:
        return self._text


def _normalize_for_repr(value: object) -> object:
    """Recursively prepare a value so ``repr()`` renders it consistently.

    Applied before ``repr()`` to any ``dict``/``list``/``tuple`` cell (and,
    recursively, to every value nested inside one) so that a ``bool`` or a
    map renders the same way at *any* depth, not just at the top level:
    booleans lowercase (``true``/``false``), and maps as plain ``dict``
    literals regardless of the concrete mapping type the FalkorDB client
    handed back (e.g. its ``OrderedDict``, which normally leaks its class
    name into ``repr()``, at any nesting depth — ``RETURN properties(m)``
    style results routinely nest one map's values inside another).
    """
    if isinstance(value, bool):
        return _ReprAsIs("true" if value else "false")
    if isinstance(value, dict):
        return {key: _normalize_for_repr(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_normalize_for_repr(item) for item in value)
    return value


def render_cell(value: object, max_chars: int) -> str:
    """Render one result cell as a single line of at most ~``max_chars`` chars.

    ``None`` becomes ``null`` (distinguishable from an empty string), booleans
    render lowercase (``true``/``false``, matching Cypher/JSON convention
    rather than Python's), strings pass through, nodes/edges use their
    FalkorDB ``__str__``, lists and maps use ``repr`` after recursively
    normalizing every nested value via ``_normalize_for_repr`` — so a map is
    rendered as a plain ``dict`` literal and a boolean lowercase at *any*
    nesting depth, not just when it is the cell's own top-level value (a
    client-library subclass, e.g. ``falkordb``'s ``OrderedDict``, therefore
    never leaks its class name into the rendering, however deeply nested).
    Newlines and tabs are escaped so that one row is always one line; pipes
    are deliberately *not* escaped (return a single column when a value must
    be copied out exactly).
    """
    if value is None:
        text = "null"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list, tuple)):
        text = repr(_normalize_for_repr(value))
    else:
        text = str(value)

    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    if len(text) > max_chars:
        dropped = len(text) - max_chars
        text = f"{text[:max_chars]}…(+{dropped} chars)"
    return text


def _column_names(header: list) -> list[str]:
    """Column names from a ``QueryResult.header`` (``[[type, name], …]``).

    Tolerates a plain list of names, which is what hand-built fixtures use.
    """
    names: list[str] = []
    for column in header or []:
        if isinstance(column, (list, tuple)) and len(column) >= 2:
            names.append(str(column[1]))
        else:
            names.append(str(column))
    return names


def truncation_notice(shown: int, total: int, reason: str) -> str:
    """The one honest truncation sentence, emitted first *and* last, verbatim.

    The "unordered" clause is not padding: the first N rows of an unordered
    result set are arbitrary, and an agent reading "showing 200 of 79581" may
    otherwise draw a conclusion from a non-deterministic sample.
    """
    return (
        f"… truncated: showing {shown} of {total} rows ({reason}) — results are unordered "
        "unless the query has ORDER BY; narrow with LIMIT, a projection, or an aggregate."
    )


def format_result(
    graph: str,
    header: list,
    rows: list,
    elapsed_ms: float,
    max_rows: int = MAX_ROWS,
    max_cell: int = MAX_CELL,
    max_chars: int = MAX_CHARS,
) -> str:
    """Render a result set as plain text (JSON roughly doubles the token cost).

    Layout, with the notice lines present only when something was actually cut::

        … truncated: showing 200 of 79581 rows (row cap) — …
        graph=cpg_falkorchat · rows=79581 · 812.4ms
        caller | file | line
        …rows…
        … truncated: showing 200 of 79581 rows (row cap) — …

    ``rows=`` is always the **true** total. When the char cap binds, whole rows
    are dropped from the tail (never a partial row) and the notice names that
    cap instead.
    """
    total = len(rows)
    stats = f"graph={graph} · rows={total} · {elapsed_ms:.1f}ms"

    names = _column_names(header)
    if not names:
        # A statement with no RETURN (not expected under RO) has no table.
        return stats

    head = " | ".join(names)
    if total == 0:
        return f"{stats}\n{head}\n(no rows)"

    kept = rows[:max_rows]
    body = [" | ".join(render_cell(cell, max_cell) for cell in row) for row in kept]
    notice = truncation_notice(len(kept), total, "row cap") if total > len(kept) else None

    def payload(lines: list[str], note: str | None) -> str:
        parts = [stats, head, *lines]
        if note is not None:
            parts = [note, *parts, note]
        return "\n".join(parts)

    rendered = payload(body, notice)
    if len(rendered) > max_chars:
        # Char cap binds: drop whole rows from the tail until the payload —
        # including *both* copies of the notice — fits.
        reason = f"char cap {max_chars}"
        keep = len(body)
        while keep > 0:
            note = truncation_notice(keep, total, reason)
            if len(payload(body[:keep], note)) <= max_chars:
                break
            keep -= 1
        body = body[:keep]
        notice = truncation_notice(keep, total, reason)
        rendered = payload(body, notice)
    return rendered


def format_plan(graph: str, plan: object) -> str:
    """Render an ``EXPLAIN`` result. Nothing was executed to produce it."""
    return f"graph={graph} · EXPLAIN (plan only — nothing was executed)\n{plan}"


#: ``QueryResult`` mutation-counter properties, verified against the pinned
#: client's own source (``falkordb`` 1.6.2, ``falkordb/query_result.py``) —
#: each one an ``int`` parsed from FalkorDB's stats line, 0 when nothing of
#: that kind happened. Confirmed real, not guessed: every name below is a
#: ``@property`` on ``falkordb.query_result.QueryResult`` in the installed
#: package.
_WRITE_STAT_ATTRS = (
    "labels_added",
    "labels_removed",
    "nodes_created",
    "nodes_deleted",
    "properties_set",
    "properties_removed",
    "relationships_created",
    "relationships_deleted",
    "indices_created",
    "indices_deleted",
)


def format_write_result(graph: str, result: object, elapsed_ms: float) -> str:
    """Render a write's outcome: which mutation counters moved, and how long it took.

    Only non-zero counters are shown, comma-separated, so a curator-clear
    (``nodes_deleted=1``) doesn't print nine zeros next to it. If the result
    object doesn't expose any of the expected counters (an unexpected client
    shape), falls back to a minimal "write ok" line with no stats rather than
    guessing — this repo's standing "don't assert unverified behavior"
    convention.
    """
    parts = [
        f"{attr}={value}"
        for attr in _WRITE_STAT_ATTRS
        if (value := getattr(result, attr, 0))
    ]
    stats = ", ".join(parts) if parts else "no counters reported"
    return f"graph={graph} · write ok ({stats}) · {elapsed_ms:.1f}ms"


# --------------------------------------------------------------------------
# Errors — every failure is a curated, actionable message, never a traceback
# --------------------------------------------------------------------------


def _is_missing_graph(exc: Exception) -> bool:
    return isinstance(exc, ResponseError) and "empty key" in str(exc).lower()


def graph_not_found_message(graph: str, graphs: list[str] | None) -> str:
    loaded = ", ".join(graphs) if graphs else "(could not read GRAPH.LIST)"
    return (
        f"Graph '{graph}' does not exist. Loaded graphs: {loaded}. If no CPG is loaded, "
        "building and loading one is the graph-dba agent's job (joern-cpg pipeline) — this tool "
        "only queries."
    )


def explain_error(
    exc: Exception,
    graph: str,
    host: str,
    port: int,
    graphs: list[str] | None = None,
) -> str:
    """Map an exception to the curated message for its condition (plan §4.4)."""
    if isinstance(exc, ResponseError):
        message = str(exc)
        lowered = message.lower()
        if "empty key" in lowered:
            return graph_not_found_message(graph, graphs)
        if "ro_query" in lowered:
            return (
                "This tool is read-only (GRAPH.RO_QUERY). Loading/writing a CPG goes through "
                "the joern-cpg pipeline (graph-dba), or redis-cli for ad-hoc writes."
            )
        if "timeout" in lowered or "timed out" in lowered:
            return (
                f"Query exceeded {TIMEOUT_MS} ms. Bound variable-length traversals (*1..N), "
                "add LIMIT, or prefix EXPLAIN to inspect the plan first."
            )
        # Syntax/semantic error: FalkorDB's own message carries line, column and
        # context — it is better than anything we could write. Pass it through.
        return f"{message}\n{SCHEMA_POINTER}"

    if isinstance(exc, (RedisConnectionError, RedisTimeoutError, ConnectionError, TimeoutError)):
        return (
            f"FalkorDB unreachable at {host}:{port}. Start it "
            "(falkor-chat/scripts/start_falkordb.sh, or docker start falkordb-dev) and retry."
        )

    return f"Unexpected error: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Connection — one lazily created client, reused (redis-py pools and reconnects)
# --------------------------------------------------------------------------

_client: FalkorDB | None = None


def get_client() -> FalkorDB:
    """Return the shared client, creating it on first use.

    Import must not connect: a server whose module-level code talks to FalkorDB
    would fail to start whenever the database is down, and a stdio server that
    fails to start is simply absent from the session.
    """
    global _client
    if _client is None:
        _client = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
    return _client


def _list_graphs(client: FalkorDB) -> list[str] | None:
    try:
        return list(client.list_graphs())
    except Exception:  # noqa: BLE001 — best-effort enrichment of an error message
        return None


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------

mcp = FastMCP(name="cypher", instructions=SERVER_INSTRUCTIONS)


@mcp.tool(
    name="query",
    description=TOOL_DESCRIPTION,
    annotations=ToolAnnotations(readOnlyHint=True),
    # `-> str` alone is NOT enough on mcp 1.28.x: FastMCP would synthesise an
    # `outputSchema {result: string}` and then return the payload twice (text
    # content *and* structuredContent), doubling every capped result.
    structured_output=False,
    meta={"anthropic/maxResultSizeChars": MAX_RESULT_SIZE_CHARS},
)
def query(graph: str, cypher: str, agent: str | None = None) -> str:
    """Run Cypher against `graph` and return a plain-text table (or write summary).

    Reads run unrestricted, exactly as before, and need no `agent`. A write is
    only ever authorized in one of 6 shapes: any agent's own producer-write,
    the legacy author-write, or a recognized curator's MENTIONS-write,
    edge-resolve, or full-node clear — see `docs/plans/generic-cypher-mcp.md`
    §3.2 and `docs/plans/kaizen-agent-ontology.md` §3.1/§3.1a.
    """
    return run_query(graph, cypher, agent)


def run_query(graph: str, cypher: str, agent: str | None = None) -> str:
    """The tool body, importable and callable without the MCP plumbing.

    Never raises: every failure path returns a curated message instead, because
    a crashed stdio server is not reconnected mid-session.
    """
    try:
        kind, to_send = split_directive(cypher)

        # Refused before any server call: FalkorDB would silently ignore the
        # prefix and return results, which is a wrong answer, not an error.
        # PROFILE can never be used to dodge write authorization below.
        if kind == "profile":
            return PROFILE_REFUSAL

        client = get_client()
        target = client.select_graph(graph)

        if kind == "explain":
            # GRAPH.EXPLAIN is not a read-only command, so a mistyped name would
            # materialise an empty key. Check first; the list doubles as the
            # not-found message's content. A failing GRAPH.LIST propagates to the
            # curated handler rather than letting an unverified explain() through.
            # Plan preview only, regardless of write intent — nothing executes.
            graphs = list(client.list_graphs())
            if graph not in graphs:
                return graph_not_found_message(graph, graphs)
            return format_plan(graph, target.explain(to_send))

        # kind == "query"
        try:
            started = time.perf_counter()
            result = target.ro_query(to_send, timeout=TIMEOUT_MS)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return format_result(graph, result.header, result.result_set, elapsed_ms)
        except ResponseError as exc:
            lowered = str(exc).lower()
            if "ro_query" in lowered:
                # A write, against a graph that EXISTS — RO_QUERY itself proved
                # it (parsed the statement, rejected it only for being a
                # write). No further classification needed; go straight to
                # enforcement.
                rejection = authorize_write(to_send, agent)
                if rejection is not None:
                    return rejection
                started = time.perf_counter()
                result = target.query(to_send, timeout=TIMEOUT_MS)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                return format_write_result(graph, result, elapsed_ms)
            if "empty key" in lowered:
                # The graph doesn't exist (yet). "empty key" fires for ANY
                # query — read or write — against a missing/mistyped name, so
                # `agent` being set is NOT on its own proof of write intent (a
                # caller — plausibly graph-dba itself, which "owns" this
                # graph — may pass `agent` out of habit on a read). Classify
                # from the TEXT, not from whether `agent` happened to be
                # supplied:
                if agent is not None and _looks_like_write(to_send):
                    # The only caller this branch exists for in practice — the
                    # one-time import, materializing kaizen_team for the
                    # first time. A false positive here still only routes into
                    # enforcement, never to an unconditional execute.
                    rejection = authorize_write(to_send, agent)
                    if rejection is not None:
                        return rejection
                    started = time.perf_counter()
                    result = target.query(to_send, timeout=TIMEOUT_MS)  # materializes the graph
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    return format_write_result(graph, result, elapsed_ms)
                # Not recognizably a write (or no agent): today's exact
                # behavior, unchanged — a plain read against a missing/
                # mistyped graph gets the real answer either way.
                graphs = list(client.list_graphs())
                return graph_not_found_message(graph, graphs)
            # everything else (syntax errors, timeouts, ...): re-raise so the
            # outer handler's curated classification runs exactly as before.
            raise
    except Exception as exc:  # noqa: BLE001 — the curated path is the only exit
        graphs = _list_graphs(get_client()) if _is_missing_graph(exc) else None
        return explain_error(exc, graph, FALKORDB_HOST, FALKORDB_PORT, graphs)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
