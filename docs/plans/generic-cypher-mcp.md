# Generic Cypher MCP — tool mechanism, enforcement, and rollout plan

> **Status:** archived · **Owner:** `architect` · **Tracks:** — (M5 proposed) · **Version:** 1.2 —
> revised after `analyst`'s U3 plan-gate (Pass 1 and Pass 2: needs changes); see §10 for the dated
> revision notes.

Design for the MCP-tool-mechanism slice of
[`../requirements/generic-cypher-mcp.md`](../requirements/generic-cypher-mcp.md) (FR-1, FR-4,
FR-8, FR-9, FR-10, FR-11; AC-1, AC-3, AC-4, AC-5, AC-6, AC-7, AC-8), per
[`generic-cypher-mcp-coordination.md`](./generic-cypher-mcp-coordination.md) unit U2. Builds on,
and does not redesign, [`generic-cypher-mcp-graph.md`](./generic-cypher-mcp-graph.md) (`graph-dba`,
U1) — the `:KaizenEntry` schema, `author` as a plain property, curator-clear = `DETACH DELETE`,
the append-before-delete ordering, and the `kaizen_graph_dba` graph key are all taken as given and
cited by path below, not re-derived.

**CPG:** considered, not relevant — this is a design task over `cpg/mcp/server.py` (~500 lines,
read in full) and a handful of markdown docs; no Joern CPG is loaded for the `cpg` component
itself (loaded CPGs in this repo cover `falkor-chat`/`salesperson` application code, not this
repo's own MCP tooling), and the investigation was direct file reads of a small, already
well-documented surface — a call-graph tool would add nothing here.

---

## 1. Goal & scope

Turn `mcp__cpg__query(graph, cypher)` — today read-only, `cpg_*`-oriented by convention only —
into the write-capable, attribution-aware, graph-agnostic tool FR-1 asks for, and pilot it end to
end on `graph-dba`'s kaizen working memory (FR-2/FR-3/FR-6, owned by U1's schema). In scope:

- The tool mechanism itself: same server, same tool name, one new optional parameter.
- FR-8's author/curator enforcement, concretely — what gets checked, on what text, against what.
- FR-4's frozen-inbox signal on `claude/graph-dba/kaizen/inbox.md`.
- Container/build implications of the code change.
- The two required doc edits: `docs/requirements/cpg-query-access.md`'s supersession pointer
  (AC-8) and `docs/BACKLOG.md`'s M5 section.
- An implementation step table sized for direct `teco` dispatch.

Out of scope (per the requirements doc's own Out of scope, unchanged here): falkor-chat
integration, documents-as-graph-data, extending past `graph-dba`'s inbox, a general
cross-agent-working-memory feature, `BACKLOG.md`-as-graph, the stakeholder's own read/write access,
guaranteed semantic search, hardened/cryptographic auth.

---

## 2. Context & findings

- **`cpg/mcp/server.py`** (read in full) is a ~500-line FastMCP stdio server, one tool
  (`mcp__cpg__query`), two required params (`graph`, `cypher`). The read path already runs
  `GRAPH.RO_QUERY` against *whatever graph key it's given* — nothing in `run_query()` special-cases
  `cpg_*` names. `cpg/mcp/README.md` says the same explicitly. **The read side needs zero code
  change for graph-agnosticism — it already is.** The gap FR-1 actually opens is the write side.
- **`GRAPH.RO_QUERY` rejects a write query only after parsing it** — confirmed live by `graph-dba`
  itself (`claude/graph-dba/kaizen/inbox.md`, 2026-08-16 entry): a syntactically valid write against
  an *existing* graph fails with a message containing `"...to be executed only on read-only
  queries"`; a query against a *non-existent* graph fails with `"...empty key"` instead — before
  parsing even gets that far. `server.py`'s `explain_error()` already keys off the first of these
  (`"ro_query" in lowered`) to produce its own curated read-only-refusal message, and `_is_missing_graph()`
  already keys off the second (`"empty key" in lowered`). Both substrings are reused, not
  reinvented, below.
- **`docs/requirements/cpg-query-access.md`'s header is `Status: archived`.** Root `AGENTS.md`'s
  doc-lifecycle convention: *"A header pointer is metadata, not an amendment — it is the one edit
  permitted on an archived document."* This matters for §6 below: the FR-1 text says to mirror how
  `joern-cpg-pipeline.md` FR-9 was superseded, but that document was (and is) `Status: active` when
  its body was rewritten in place — `cpg-query-access.md` is not, so the mirrored pattern cannot be
  applied literally. §6 works out the correct, convention-compliant substitute.
- **Neither `graph-dba` nor `cobb` carries a restrictive `tools:` allowlist** — both run with the
  harness default ("all tools"), unlike `analyst`/`architect`/`qa-engineer`, which needed
  `mcp__cpg__query` added to an explicit allowlist in M3 (C-304) before the tool became visible to
  them. So **no agent-wiring step is needed here** — both consumers this delivery actually needs
  (`graph-dba` as author, `cobb` as curator) already see `mcp__cpg__query` today.
- **`falkordb-py` is pinned `1.6.x`** (`cpg/mcp/requirements.txt`). The exact `QueryResult`
  mutation-counter attribute names (`nodes_created`, `properties_set`, etc.) are **not verified** in
  this investigation — flagged in §5.5 and §9 for the implementer to confirm against the pinned
  client before writing `format_write_result()`.
- **`GRAPH.CONSTRAINT CREATE` is not Cypher** — it is a bare Redis module command, unreachable
  through `GRAPH.QUERY`/`GRAPH.RO_QUERY` at all. This bounds what the generic tool can ever be asked
  to do for schema DDL (§4.4).

---

## 3. Design & rationale

### 3.1 FR-1 — tool mechanism: extend `cpg/mcp/` in place, one tool, one new optional parameter

**Decision: `cpg/mcp/server.py` is widened in place. No new sibling server, no second tool.** The
tool stays `mcp__cpg__query`, gains one new **optional** third parameter:

```
mcp__cpg__query(graph: str, cypher: str, agent: str | None = None) -> str
```

**Why extend in place, not a new server.** FR-1's own wording anchors on continuity — *"the same
two-parameter shape as today's `mcp__cpg__query(graph, cypher)`"* — which only makes sense if the
callable stays `mcp__cpg__query`. A sibling server would mean a second `.mcp.json` entry, a second
container image, a second set of allowlist edits on every consuming agent, and two places a future
caller has to remember to check — for a component that (per §2) is *already* graph-agnostic on the
90% case (reads). The container/build machinery (`build.sh`'s content-hash tag, `docker-run.sh`'s
self-healing rebuild-on-miss, the in-container test gate) is a real, working investment;
duplicating it for a server that differs from `cpg/mcp/` only in a write branch would be pure
churn with no payoff. The one real cost of extending in place — the server is still literally named
`cpg` while it now also holds `kaizen_graph_dba` — is real but small for a *one-agent pilot*; see
§9 for why a rename is deliberately not done now.

**Why one tool, not a second `write` tool.** Two considered shapes:

| Shape | Tool count | Read-path blast radius |
|---|---|---|
| **A — widen `query`** (chosen) | 1 | Zero: `agent` defaults to `None`, unused unless a write is detected |
| B — add `mcp__cpg__write(graph, cypher, agent)` | 2 | Zero, but doubles the allowlist/harness surface for every future consumer |

`docs/BACKLOG.md`'s M3 section records the standing reversal trigger against a multi-tool FalkorDB
MCP surface: `@falkordb/mcpserver`'s 7 tools including an unfiltered `delete_graph` were rejected,
with *"an upstream server that can be filtered down to one read-only tool"* as the recorded
condition for reconsidering. This delivery **does** reopen the "one tool" purity — write access is
new — but Shape A keeps the tool count at exactly one, still narrow, still attributed, and every
new capability gated by the enforcement in §3.2. That is not the same shape as the rejected
alternative (unrestricted, unfiltered, un-attributed multi-tool write access); it is a single,
still-filtered surface that happens to now accept one more optional parameter. Shape B was rejected
because it multiplies exactly the harness-config/allowlist surface area Shape A avoids, for zero
behavioral gain — the enforcement logic in §3.2 has to exist either way, and putting it behind a
second tool name buys nothing.

**Why `agent` is optional, not required.** FR-6 requires reads to stay ungated — *any* agent reads
`kaizen_graph_dba` today, with no identity declared. Making `agent` required on every call would
force every existing and future *read* caller to supply a value it doesn't need, which is exactly
the "same shape" FR-1 asks to preserve for the common case. `agent` is required only when the tool
detects a write attempt (§3.2) — enforced in code, not by the parameter's own optionality.

**Detecting "this call is a write" — reusing `graph-dba`'s own live-verified technique.**
`run_query()`'s existing `"query"`-kind branch already calls `target.ro_query(to_send, ...)`
first. Today, any failure there falls straight to `explain_error()`. The new logic inserts one
branch **before** that fallthrough:

```
kind, to_send = split_directive(cypher)          # unchanged: query | explain | profile
if kind == "profile":  return PROFILE_REFUSAL      # unchanged — refused before any server call,
                                                     #   so PROFILE can never be used to dodge §3.2
if kind == "explain":  ... unchanged (plan preview only, read or write text alike) ...

# kind == "query" — the new branch:
try:
    result = target.ro_query(to_send, timeout=TIMEOUT_MS)
    return format_result(...)                       # unchanged: it really was a read
except ResponseError as exc:
    lowered = str(exc).lower()
    if "ro_query" in lowered:
        # A write, against a graph that EXISTS — RO_QUERY itself proved it (parsed the
        # statement, rejected it only for being a write). No further classification
        # needed; go straight to enforcement.
        rejection = authorize_write(to_send, agent)
        if rejection is not None:
            return rejection
        result = target.query(to_send, timeout=TIMEOUT_MS)
        return format_write_result(graph, result, elapsed_ms)
    if "empty key" in lowered:
        # The graph doesn't exist (yet). "empty key" fires for ANY query — read or
        # write — against a missing/mistyped name, so `agent` being set is NOT on its
        # own proof of write intent (Pass-1 review, M3: a caller — plausibly graph-dba
        # itself, which "owns" this graph — may pass `agent` out of habit on a read).
        # Classify from the TEXT, not from whether `agent` happened to be supplied:
        if agent is not None and _looks_like_write(to_send):
            # The only caller this branch exists for in practice — the one-time
            # import, §3.4, creating kaizen_graph_dba for the first time. A false
            # positive here still only routes into enforcement, never to an
            # unconditional execute.
            rejection = authorize_write(to_send, agent)
            if rejection is not None:
                return rejection
            result = target.query(to_send, timeout=TIMEOUT_MS)   # materializes the graph
            return format_write_result(graph, result, elapsed_ms)
        # Not recognizably a write (or no agent): today's exact behavior, unchanged —
        # a plain read against a missing/mistyped graph gets the real answer either way.
        graphs = list(client.list_graphs())
        return graph_not_found_message(graph, graphs)
    # everything else (syntax errors, timeouts, ...): falls through to explain_error()
    # exactly as today. UNCHANGED.
```

```python
_WRITE_KEYWORD_RE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE)\b", re.IGNORECASE)


def _looks_like_write(cypher: str) -> bool:
    """Lightweight pre-classification only — never an authorization decision. A
    read whose free-text predicate happens to quote a write keyword (e.g. `WHERE
    e.context CONTAINS 'DETACH DELETE'`) can false-positive here; the only
    consequence is routing into `authorize_write()`, which is itself
    string-literal- and CREATE-span-aware (§3.2) and would reject it as
    "not a recognized write shape" rather than the more accurate "graph not
    found" — a less-precise message, never a wrong authorization. This branch
    only ever fires pre-migration or on a graph-name typo, since a populated
    `kaizen_graph_dba` never returns "empty key" again after the one-time import.
    """
    return bool(_WRITE_KEYWORD_RE.search(cypher))
```

**Why the probe is always safe.** `GRAPH.RO_QUERY` refuses a write **server-side, before
executing** it (this is the whole reason the original tool is "safe by construction" per
`cpg/mcp/README.md`) — so the probe call itself can never perform a write, regardless of which
branch it ends up in. Nothing above changes that guarantee; it only decides, after a proven-safe
probe, whether to run the same text again through the write-capable `target.query()`.

**Why this doesn't regress FR-3/FR-4 of the *original* `cpg-query-access.md` tool contract.** A
genuinely broken write (bad syntax) still gets FalkorDB's own syntax error via the unchanged
`explain_error()` fallthrough — `graph-dba`'s own inbox note confirms `RO_QUERY` parses *before*
classifying, so a malformed write never even reaches the "read-only" message, and therefore never
reaches `authorize_write()` either. Enforcement only ever runs on syntactically valid write text.

### 3.2 FR-8 — author/curator enforcement

**Two, and only two, recognized write shapes.** Pass-1 review found two real gaps in the original
design of this check: **M1** — a whole-query regex scan misreads an `author:`-shaped substring
embedded in free-text `evidence`/`context` as a real claim (over-rejects a legitimate write); **M2**
— the original "author" shape also authorized a `SET`-based reassignment of an *existing* entry's
`author`, wider than FR-8's "creates new entries attributed to itself only" (a latent
over-authorization that activates the moment a second author exists, per FR-10). Both are closed by
the same redesign: **only the body of a `CREATE (...:KaizenEntry {...})` map literal is ever
inspected for an `author:` claim, and only the part of that body that sits outside any nested
string literal.** Still no Cypher parser/AST — a small, structurally-scoped, string-literal-aware
scan, matching the trust level FR-8 itself states ("well-behaved callers can't do this by
accident," not hardened against a malicious one).

```python
CURATOR_AGENTS = frozenset(
    a.strip() for a in os.environ.get("CPG_MCP_CURATOR_AGENTS", "cobb").split(",") if a.strip()
)

# The map-KEY form only — never `.author = ...` (SET). Restricting extraction to a
# CREATE's own map literal (below) already excludes SET syntactically, since SET
# never appears inside a CREATE's `{...}` body — this pattern doesn't even need to
# recognize the SET spelling to make that true.
_AUTHOR_LITERAL_RE = re.compile(r"""\bauthor\s*:\s*['"]([^'"]*)['"]""", re.IGNORECASE)

# The ONE recognized curator-clear skeleton (graph-dba's plan §3), whitespace-collapsed
# before matching. Deliberately narrow to :KaizenEntry — see §9 revisit trigger.
_CURATOR_CLEAR_RE = re.compile(
    r"^MATCH \([a-zA-Z_]\w*:KaizenEntry \{entryId: ['\"][^'\"]+['\"]\}\) "
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


def _kaizen_entry_create_map_spans(cypher: str) -> list[str]:
    """Body text of every map literal `{...}` immediately following a
    `CREATE (<var>:KaizenEntry ...)` clause. Brace-matched using the same
    string-literal scan as above, so a free-text field containing a literal `{`/`}`
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
        depth, i, in_string = 1, body_start, None
        while i < len(cypher) and depth > 0:
            ch = cypher[i]
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
        spans.append(cypher[body_start:i - 1])
    return spans


def _author_claims(cypher: str) -> list[str]:
    """Every literal `author: '<value>'` found strictly inside a CREATE's own
    KaizenEntry map body, and NOT nested inside a sibling property's own string
    value (M1's fix: a decoy `author:`-shaped substring inside `evidence`/`context`
    sits inside THAT property's string-literal span and is excluded)."""
    claims = []
    for span in _kaizen_entry_create_map_spans(cypher):
        literal_ranges = _string_literal_spans(span)
        for m in _AUTHOR_LITERAL_RE.finditer(span):
            if not any(s <= m.start() < e for s, e in literal_ranges):
                claims.append(m.group(1))
    return claims


def authorize_write(cypher: str, agent: str | None) -> str | None:
    """Return a curated rejection message, or None if the write is authorized."""
    if not agent:
        return (
            "Write detected but no `agent` parameter supplied. Declare the caller's "
            "identity: mcp__cpg__query(graph, cypher, agent='<your-agent-slug>')."
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
        return None   # every author: literal found inside a CREATE:KaizenEntry body
                       # matches the declared agent — allowed
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
    return (
        "Rejected: this write is neither an author-write (no literal `author: "
        f"'{agent}'` found inside a CREATE (...:KaizenEntry {{...}}) clause) nor the "
        "recognized curator-clear shape. This tool only authorizes those two write "
        "shapes (FR-8)."
    )
```

**Why a plain-property equality check, not a relationship traversal.** `graph-dba`'s U1 note (§2)
already made `author` a plain string property specifically so this check would be a "plain
equality predicate, no traversal or extra lookup needed" — this design takes that foundation as
given and builds exactly that predicate, nothing more.

**M2's fix, explicitly.** `_author_claims()` only ever looks inside spans returned by
`_kaizen_entry_create_map_spans()`, which only ever finds `CREATE (...:KaizenEntry {...})` bodies.
A `SET e.author = 'graph-dba'` clause — anywhere in the query, against any `MATCH`-selected node —
produces **zero** spans to search, so it can never register as an "author claim," regardless of
whether the value matches `agent`. It therefore always falls to the final `return` (neither shape
recognized) and is rejected outright — including the exact reproduction the review gave,
`MATCH (e:KaizenEntry {entryId:'<not-mine>'}) SET e.author = 'graph-dba'`, even with a perfectly
matching `agent='graph-dba'` (see §8.1 test 15). Chosen over the review's second offered fix
(require a `WHERE e.author = '<agent>'` ownership guard on `SET`) because FR-8's own wording is
"creates new entries," not "creates or corrects" — there is no stated requirement for
self-correcting `SET`, and removing the path entirely is strictly simpler and safer than adding a
second guard whose own correctness would need equal scrutiny. If a genuine need for author
self-correction surfaces later, it is a new, separately-designed third write shape — not folded
into "author" by relaxing this restriction.

**M1's fix, explicitly.** `_author_claims()` rejects any `author:`-shaped match whose start
position falls inside `_string_literal_spans(span)` — i.e., inside another property's own quoted
value. A migrated `evidence` field that verbatim quotes `author: 'x'` as part of documenting this
very schema (plausible, since this pilot is `graph-dba` dogfooding its own new property — the real
`inbox.md` entries already quote Cypher/config/output verbatim) sits inside `evidence`'s own
string-literal span and is excluded; the real `author: 'graph-dba'` key sits at the map's top
level, outside every string span, and is found (see §8.1 test 14).

**M1-residual, explicitly (Pass-2 review).** The fix above closes a *bare* `author:`-shaped
fragment sitting in free text, but `_kaizen_entry_create_map_spans()`'s own `CREATE`-keyword
*location* step originally scanned the raw, full query text for candidate `CREATE` keywords
without itself being string-literal-aware — only the body-extraction step *after* a match was
found tracked string literals. So a free-text field whose content happens to quote a **complete**
`CREATE (<var>:KaizenEntry {...})`-shaped example — exactly the kind of thing a `graph-dba` kaizen
entry documenting this very migration/schema would contain, since every `evidence` example in this
very document is a full `CREATE (...:KaizenEntry {...})` snippet — registered as a second,
independent top-level clause. Two verified consequences: (1) **over-rejection** — a legitimate
write with a correct top-level `author: 'graph-dba'` gets rejected because the decoy clause's own
embedded `author: 'evil'` also registers as a claim and mismatches; (2) **under-enforcement** — if
the real top-level `CREATE` clause omits its own `author:` property while a free-text field embeds
a decoy `CREATE (...:KaizenEntry {author: '<declared-agent>'})`-shaped substring, the decoy's claim
alone satisfies `authorize_write()` and the write is wrongly allowed, creating a real node with no
genuine `author` property. **Fixed** by computing `_string_literal_spans` once over the *whole*
`cypher` text before locating `CREATE` candidates (shown in the code block above), and skipping any
`CREATE` match that falls inside one of those spans — so an embedded, free-text-quoted `CREATE`
clause is never treated as a real top-level one, regardless of how complete it looks. (1) fails
safe on its own (no write executes, just a confusing message) and (2) requires the caller to omit
the literal from the real top-level map — a deviation from every recipe this plan hands to
`graph-dba`/`cobb` (§3.2, §3.4, §3.5 all embed the literal directly), so it sits closer to the
already-accepted aliasing-evasion trade-off than to an accidental trap — but both share one root
cause and this one fix closes both (see §8.1 test 16).

**Why curator-clear is narrowed to one exact skeleton, not "cobb can write anything."** The brief
is explicit that curator status must not become a blanket write grant. `_CURATOR_CLEAR_RE` matches
**only** `MATCH (x:KaizenEntry {entryId: '<literal>'}) DETACH DELETE x` — the one operation
`graph-dba`'s plan §3 defines for curator-clear. A `cobb`-attributed call with any other write shape
(a `SET`, a bulk `DELETE`, a different label) falls through to the final `return` and is rejected
exactly as it would be for any other agent. This is what makes curator a **role-scoped capability**,
not a standing grant.

**AC-6, concretely satisfied.** *"a call claims to be `graph-dba` and attempts to create an entry
attributed to `cobb`"* → the Cypher's `author:` literal (inside its `CREATE (...:KaizenEntry {...})`
body) is `'cobb'`, the declared `agent` is `'graph-dba'` → `claims == ['cobb']`,
`mismatched == ['cobb']` → rejected, **before** `target.query()` is ever called (no partial write,
nothing to roll back). The reverse direction is symmetric.

**Known, accepted limitation (unchanged by the M1/M2 fixes — see §9).** `_author_claims()` still
only recognizes a *literal* quoted value. `WITH 'graph-dba' AS a CREATE (k:KaizenEntry {author: a})`
— an aliased/computed value — still evades detection entirely, since no quoted literal ever appears
next to `author:`. This is accepted because FR-8 states the bar explicitly: *"enforced at the
'well-behaved callers can't do this by accident' level, not hardened against a malicious caller."*
Every real recipe this design hands to `graph-dba`/`cobb` (§3.4, §3.5) writes the literal directly;
nothing in this pilot has a reason to alias it.

### 3.3 FR-4 — frozen-inbox signal

Prepend a blockquote note directly under `claude/graph-dba/kaizen/inbox.md`'s existing H1, above
its current intro paragraph — nothing else in the file changes (AC-3: content preserved, only the
new note added):

```markdown
> **FROZEN — 2026-08-17.** This file is a historical snapshot only. Its contents (as of this date)
> were imported once into the `kaizen_graph_dba` FalkorDB graph
> (`docs/plans/generic-cypher-mcp.md`); `graph-dba` no longer appends here. New raw learnings are
> written directly into the graph and are immediately queryable by any agent:
> `mcp__cpg__query(graph='kaizen_graph_dba', cypher='MATCH (e:KaizenEntry) RETURN e.date, e.fact,
> e.evidence, e.context, e.suggestedHome, e.author ORDER BY e.date')`. Content below is preserved
> for historical reference and will not change.
```

A header note was chosen over the alternatives considered — renaming the file (breaks every
existing inbound reference, including this very document's own citations, for no reader benefit),
or moving it to an `archive/` tree (the doc-lifecycle convention in root `AGENTS.md` explicitly
retired that pattern in favor of in-place `Status:` markers) — because it is the smallest,
most-visible-on-open signal that satisfies AC-3 without disturbing anything else. Date the note
with the actual migration date at implementation time, not the value shown here.

### 3.4 FR-2/FR-3 — the one-time import runs *through* the new write path, not around it

`graph-dba`'s plan §4 explicitly left the migration script unwritten, flagging it as "an
`architect`-sequenced unit." **Decision: no separate committed script — the migration is
`graph-dba` itself dogfooding the generic tool**, using the very "empty key + agent" branch built in
§3.1:

1. `graph-dba` reads `claude/graph-dba/kaizen/inbox.md` (all six current entries), generates
   `entryId` (`uuid4()`, six times) and one shared `createdAt` (import-run ISO-8601 timestamp), and
   builds a single Cypher text: `UNWIND [<six per-row maps, each carrying entryId/date/fact/
   evidence/context/suggestedHome/createdAt — NOT author>] AS e CREATE (k:KaizenEntry {entryId:
   e.entryId, date: e.date, fact: e.fact, evidence: e.evidence, context: e.context, suggestedHome:
   e.suggestedHome, author: 'graph-dba', createdAt: e.createdAt})`. The `author` value is set
   **once, as a literal, in the `CREATE` clause itself** — not per-row inside the `UNWIND` list —
   because every row in this one-time batch shares the same author by construction; the six per-row
   maps never need their own `author` key. `graph-dba` calls `mcp__cpg__query(graph=
   'kaizen_graph_dba', cypher=<that text>, agent='graph-dba')`. `kaizen_graph_dba` does not exist
   yet → `ro_query` fails "empty key" → `_looks_like_write()` sees the `CREATE` keyword → true →
   `authorize_write()` runs `_author_claims()`, which finds exactly **one** literal
   `author: 'graph-dba'` inside the `CREATE`'s own `KaizenEntry` map body, matching `agent` →
   **allowed** → `target.query()` runs for real, materializing the graph as a side effect of this
   one legitimate write. (`_author_claims()` returning a *list*, not a single value, still matters
   in general — a batch with several distinct `CREATE` clauses, each setting its own literal
   `author:`, is enforced the same way, all must match `agent` — this migration only ever needs
   one shared literal, since the whole batch shares one author.)
2. Index + uniqueness constraint (`graph-dba`'s plan §5) are **not** sent through the generic tool.
   `CREATE INDEX FOR (e:KaizenEntry) ON (e.entryId)` is valid Cypher but is schema DDL, not an
   entry-authoring write — outside the two shapes §3.2 recognizes on purpose (§3.2's rejection
   branch would otherwise catch it, correctly, since it has no `author:` literal and isn't the
   curator-clear shape). `GRAPH.CONSTRAINT CREATE kaizen_graph_dba UNIQUE LABEL KaizenEntry
   PROPERTIES 1 entryId` **cannot** go through the tool at all — it is a bare Redis module command,
   not Cypher, unreachable via `GRAPH.QUERY`/`GRAPH.RO_QUERY`. Both run the way `graph-dba`
   provisions indexes on any graph today: directly, via its own already-available `redis-cli`/Bash
   access, respecting the index-before-constraint ordering (`falkordb-quirks.md`).
3. `graph-dba` prepends the §3.3 frozen note to `inbox.md` in the same working session.

### 3.5 FR-9 — the curator-clear tool-call sequence, with the ordering constraint preserved

`graph-dba`'s plan §3 names a **non-negotiable** ordering: `history.md` append confirmed durable
*before* the graph delete. The generic tool has no transaction spanning a markdown file edit and a
FalkorDB write — they are, and remain, two independent tool calls. This design does not (cannot)
enforce the order mechanically; it is spelled out as a mandatory sequence in `cobb`'s distillation
procedure (§5, `skills/agent-maintenance/SKILL.md` §5 edit):

1. Read the raw entry: `mcp__cpg__query(graph='kaizen_graph_dba', cypher="MATCH (e:KaizenEntry
   {entryId: '<id>'}) RETURN e.date, e.fact, e.evidence, e.context, e.suggestedHome, e.author")` —
   a plain read, `agent` omitted.
2. Verify the entry (unchanged step of today's distillation workflow).
3. `Edit` `claude/graph-dba/kaizen/history.md`, appending the promotion in the existing format.
   **Confirm the edit tool call succeeded** before step 4 — do not proceed on an error.
4. Only then: `mcp__cpg__query(graph='kaizen_graph_dba', cypher="MATCH (e:KaizenEntry {entryId:
   '<id>'}) DETACH DELETE e", agent='cobb')`. This is exactly `_CURATOR_CLEAR_RE`'s one recognized
   shape, `agent='cobb'` is in `CURATOR_AGENTS` → allowed.

Fail-safe direction, restated from `graph-dba`'s note: a crash between steps 3 and 4 leaves the
entry duplicated (still in the graph, now also in `history.md`) — a harmless no-op on `cobb`'s next
pass. A crash **before** step 3 (or a reordering that deletes first) risks the one failure mode this
design cannot tolerate — permanent loss from both places. This is why §7's doc-edit step states the
order as mandatory, not advisory.

**Resolving the review's open question: does `graph-dba`'s own prompt also need to record this
ordering?** Decision: **no** — document it solely in `skills/agent-maintenance/SKILL.md` §5
(`cobb`'s side, step 4b in §7). `graph-dba` never runs the delete half of this sequence (only
`cobb` does, in its curator role), and it only ever runs the append half indirectly by way of
`cobb`'s workflow — `graph-dba`'s own operative prompt has no action item that depends on knowing
this ordering exists. Duplicating an ordering constraint into a prompt that has no step governed by
it would be exactly the kind of "enumerated summary fact copied into an always-loaded doc" the
`agent-maintenance` skill itself warns against (§2, "don't create enumerated summary facts... they
duplicate ground truth, cost tokens every session, and rot") — the ordering's one true owner is the
agent that actually performs both halves of the sequence.

### 3.6 FR-10 — future human-authored entry: confirmed, no redesign needed

`agent` is an unconstrained string; `authorize_write()` never inspects *who* the string names,
only whether it appears as the `author:` literal (or is in `CURATOR_AGENTS`). A future human caller
supplies `agent='<some human identifier>'` and the identical equality/regex logic governs it —
zero code change. (If the human/agent distinction itself ever needs to be *queried*, `graph-dba`'s
plan §2 already named the additive fix: an `authorKind` property — still no redesign of this
enforcement layer.)

---

## 4. Container/build implications

- **`cpg/mcp/server.py`** — the only source file that changes. New: `CURATOR_AGENTS`,
  `_AUTHOR_LITERAL_RE`, `_CURATOR_CLEAR_RE`, `_WRITE_KEYWORD_RE`, `_looks_like_write()`,
  `_string_literal_spans()`, `_kaizen_entry_create_map_spans()`, `_author_claims()`,
  `authorize_write()`, `format_write_result()`; the `query` tool's signature gains
  `agent: str | None = None`; `run_query()` gains the branch in §3.1. `TOOL_DESCRIPTION`/
  `SERVER_INSTRUCTIONS` are rewritten to state: not limited to `cpg_*`
  graphs, the new `agent` parameter, and the two recognized write shapes in one sentence each — keep
  `SERVER_INSTRUCTIONS` under the ~2000-char pin `test_server_instructions_are_present_and_bounded`
  (C-318) already asserts; extend that test's bound check after editing, don't just trust it.
- **No new dependency.** `re` is stdlib, already imported. `requirements*.txt` unchanged.
- **`Dockerfile`, `.dockerignore`, `build.sh`, `docker-run.sh`, `image-tag.sh` — no structural
  change.** The image tag is a content hash over (among others) `server.py` and everything under
  `tests/`; editing `server.py` and adding a test module changes the hash automatically. The next
  `docker-run.sh` launch self-heals (miss → build) exactly as designed — no script edits needed.
  Implementer's done-condition: run `cpg/mcp/build.sh` once by hand after the code change (don't
  rely solely on the self-heal path for the first verification), then the in-container test gate
  (`cpg/mcp/README.md` § "The in-container test gate").
- **`.mcp.json`** — **unchanged.** Same server name, same tool name, same launch command. This is
  the concrete payoff of §3.1's "extend in place, widen one tool" decision: zero harness-config
  churn.
- **No agent `tools:` allowlist edit needed** — per §2's finding, `graph-dba` and `cobb` already run
  with unrestricted tool access.
- **`cpg/mcp/README.md`** — update "The tool" section's parameter table (add `agent`), add a
  "Writing through this tool" subsection describing the two shapes and the `CPG_MCP_CURATOR_AGENTS`
  env var, matching the file's existing register (measured facts, explicit "unverified" flags where
  applicable per §5.5's `falkordb-py` caveat).

---

## 5. `docs/requirements/cpg-query-access.md` — the AC-8 edit (header pointer only)

Per §2's finding (the document is `Status: archived`), the edit is **header-metadata only** — the
body's "Non-CPG graphs / general agent access to FalkorDB" out-of-scope bullet is **not** touched,
matching the doc-lifecycle rule that a header pointer is "the one edit permitted on an archived
document." This is a deliberate divergence from FR-1's literal "mirrors how FR-6 superseded
`joern-cpg-pipeline.md` FR-9" instruction, because that precedent document was (and remains)
`Status: active` — the mirrored in-place-body-rewrite pattern does not apply to an archived one.

**Old header (lines 1–4):**
```markdown
# CPG query access — Feature Requirements
> **Status:** archived · **Owner:** `tico` · **Tracks:** C-301…C-307 (M3) ·
> **Delivered ✅** — AC-1…AC-4 met and accepted (M3, 2026-07-25); follow-ups tracked in
> [`../BACKLOG.md`](../BACKLOG.md) · **Last updated:** 2026-07-25
```

**New header:**
```markdown
# CPG query access — Feature Requirements
> **Status:** archived · **Owner:** `tico` · **Tracks:** C-301…C-307 (M3) ·
> **Delivered ✅** — AC-1…AC-4 met and accepted (M3, 2026-07-25); follow-ups tracked in
> [`../BACKLOG.md`](../BACKLOG.md) · **Last updated:** 2026-07-25
>
> **Note:** the "Non-CPG graphs / general agent access to FalkorDB" and "Authentication, per-user
> grants, and read-only enforcement" lines below are widened by
> [`generic-cypher-mcp.md`](./generic-cypher-mcp.md) FR-1 — read that document for the current
> scope; this archived document's body is left exactly as originally written.
```

**Plain prose, not the `Supersedes:`/`Superseded by:` field pair (revised — Pass-1 review, m1).**
The original draft of this edit used `**Superseded by:**`. The doc-lifecycle convention introduces
that field pair specifically for "a second document of the same kind and topic" — an ordinal
successor within one slug family (`x.md` → `x2.md`) — and this pointer instead spans two different
topic slugs (`cpg-query-access` → `generic-cypher-mcp`) in two different `docs/` kinds
(`requirements/` on both sides, but not the same family). The review flagged this as a plausible
but unconfirmed stretch of a mechanism built for same-slug evolution, not a clear defect. Rather
than adjudicate whether cross-topic reuse of that specific field is intended, this revision drops
it in favor of a plain bolded `**Note:**` line carrying the identical pointer without borrowing a
field name reserved for a different relationship. Still header-only, still the one edit an archived
document permits, still satisfies AC-8 ("no reader finds the two documents disagreeing") the way
the doc-lifecycle convention intends an archived document to be read: header first. Owner: `coder`,
bundled with the `server.py` change (mirrors the M3 precedent — C-305, "Requirements
reconciliation," owner `coder`).

---

## 6. `docs/BACKLOG.md` — M5 proposal

Add after the M4 section, mirroring the M3/M4 milestone-map row and item format exactly:

**Milestone-map row:**
```markdown
| **M5 — Generic Cypher MCP** | `mcp__cpg__query` gains write capability (an optional `agent`
param, two enforced write shapes) and is piloted end to end on `graph-dba`'s kaizen working
memory: the graph replaces `inbox.md` as the raw-capture layer, `history.md` is unchanged, `cobb`'s
distillation workflow runs against the graph. AC-1…AC-8 acceptance-tested. | **C-501 → C-506** |
```

**Section body** (`## M5 — Generic Cypher MCP`), items numbered to match §7's step table below —
`C-501` (server write path, step 1), `C-502` (requirements pointer, step 2), `C-503` (migration +
inbox freeze, step 3), `C-504` (repo-wide catalog/backlog docs, step 4a), `C-505` (both agents'
operative-prompt + distillation-workflow docs, step 4b — split out from `C-504` per Pass-1 review's
B1, which found the original single docs step omitted the two files that actually drive agent
behavior), `C-506` (acceptance pass, step 5) — each one line, same style as M3/M4 entries, status
`🔵 proposed` until each step closes. Owner: whoever closes out docs in step 4a/4b (§7) — mirrors
the M3/M4 precedent where the catalog-sync step (`C-307`/`C-407`, owner `cobb`) also wrote the
milestone's own backlog section.

---

## 7. Implementation step table

Six steps, each ≤3 files, sized for individual or small-adjacent-cluster `teco` dispatch (steps 1
and 2 share an owner and can dispatch together; 4a/4b likewise). **Revised from five to six steps**
per Pass-1 review's blocker (B1): the original step 4 listed three files
(`claude/AGENTS.md`, `skills/agent-maintenance/SKILL.md`, `docs/BACKLOG.md`) but omitted the two
files that are each an agent's own *operative* prompt — `claude/graph-dba/graph-dba.md:76`
currently instructs `graph-dba` to append learnings to `kaizen/inbox.md` (directly contradicts FR-2
the moment this ships) and `claude/cobb/cobb.md:71` claims *"every agent (you included) appends...
to its `kaizen/inbox.md`"* (false for `graph-dba` post-delivery, and it's `cobb`'s own operative
description of how to run distillation, FR-9) — plus `claude/README.md:22-32`, which describes the
convention generically with no `graph-dba` carve-out. Folding all six files into one step would
exceed the ≤3-file sizing guideline, so step 4 is split into two small, adjacent-dispatchable units
by *kind* of doc: 4a (repo-wide catalog/backlog prose) and 4b (the two agents' own operative
prompts + the distillation-procedure skill).

| # | Owner | Files | Depends on | Done-condition |
|---|---|---|---|---|
| **1** | `coder` | `cpg/mcp/server.py`, `cpg/mcp/tests/test_server.py` (or new `test_write.py`), `cpg/mcp/README.md` | — | §8's test list green offline (`cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q`); `cpg/mcp/build.sh` run once by hand; in-container gate green (`cpg/mcp/README.md` § "The in-container test gate"); `SERVER_INSTRUCTIONS` length re-verified ≤2000 chars |
| **2** | `coder` | `docs/requirements/cpg-query-access.md` | — | §5's header edit applied verbatim; `git diff` shows **only** the header block changed |
| **3** | `graph-dba` | `claude/graph-dba/kaizen/inbox.md` | 1 (needs the live write path) | §3.3's frozen note prepended, rest of file byte-identical below it; live query confirms `MATCH (e:KaizenEntry) RETURN count(e)` = 6 on `kaizen_graph_dba`; index + uniqueness constraint present (`GRAPH.CONSTRAINT` verified via `redis-cli`) |
| **4a** | `cobb` | `claude/AGENTS.md`, `claude/README.md`, `docs/BACKLOG.md` (§6 above) | 3 | Repo-wide catalog/convention prose describes `graph-dba`'s actual post-migration behavior; no remaining unconditional claim that *every* agent appends to `inbox.md`; `docs/BACKLOG.md` M5 section + milestone-map row added |
| **4b** | `cobb` | `claude/graph-dba/graph-dba.md`, `claude/cobb/cobb.md`, `skills/agent-maintenance/SKILL.md` (§5) | 3 | `graph-dba.md`'s "Learning capture" section no longer directs new learnings to `kaizen/inbox.md` — it directs them to the graph (FR-2), with `falkordb-quirks.md`/other-topic routing otherwise unchanged; `cobb.md`'s distillation-duties bullet no longer makes the blanket "every agent... appends... to `kaizen/inbox.md`" claim — it states `graph-dba`'s raw capture is graph-based, others' is still file-based; `agent-maintenance` §5 states the graph-read-then-distill sequence for `graph-dba` specifically, including §3.5's append-before-delete ordering (resolved in §3.5 to live **only** here, not duplicated into `graph-dba.md`) |
| **5** | `qa-engineer` | — (execution only; produces `docs/test-plans/generic-cypher-mcp.md` + `docs/test-reports/generic-cypher-mcp-report.md`) | 1, 2, 3, 4a, 4b | AC-1…AC-8 each exercised live (§8) |

**Close-out done-condition for 4a+4b jointly (replaces a fixed file-list as the actual FR-11/AC-7
gate, per Pass-1 review's suggested fix — a search catches a doc this pass didn't think to name):**
run `grep -rln 'kaizen/inbox\.md\|append.*inbox' claude/ skills/agent-maintenance/SKILL.md` **before**
4a/4b (to capture the starting hit set) and again **after**. Every hit in the "after" pass must be
either (a) one of the five files this table already names as updated, confirmed edited, or (b)
confirmed **not** `graph-dba`-specific — i.e. still-correct generic convention prose for every other
agent, left as-is on purpose. **Expect on the order of 30 hits on the current tree, not a short
list** (Pass-2 review actually ran the sweep: 35 hits total) — most are legitimately non-
`graph-dba`-specific noise (every other agent's own `kaizen/inbox.md` convention, hook scripts,
other agents' `kaizen/history.md` entries that happen to contain "append" near "inbox") and need no
edit; only the five files this table already names require a change. This is a real triage cost,
not a rubber-stamp — size the dispatch for it, don't expect a five-file scan.

**Sequencing:**
```
1 ─┬─▶ 3 ─┬─▶ 4a ─┐
2 ─┘      └─▶ 4b ─┼─▶ 5 ⇒ M5 done
```
2 has no dependency on 1/3/4; 4a and 4b both depend on 3 (not on each other) so the docs describe
`graph-dba`'s actual, now-migrated behavior rather than an aspirational one; 4a/4b can dispatch as a
small adjacent cluster (same owner, same dependency).

Before any of these dispatch, this plan and `generic-cypher-mcp-graph.md` go through the combined
`analyst` plan-gate (U3, already queued in the coordination doc) — not drawn as a step here, per
that document's own framing.

---

## 8. Test strategy

### 8.1 `cpg/mcp` unit tests (offline, step 1) — the concrete list for `coder`

Extends the existing `FakeGraph`-based suite (`test_server.py`), which already fakes `ro_query`/
`explain`/errors — add a `query()` method to the fake that records calls and returns a
`FakeResult`-like object with mutation-counter attributes, so "was a real write attempted" is
directly assertable per case.

1. `test_read_path_unchanged_without_agent` — a plain `MATCH...RETURN`, no `agent` → unchanged
   behavior, regression pin.
2. `test_write_without_agent_is_rejected` — write Cypher, `agent` omitted, graph exists (fake
   `ro_query` raises the "read-only" `ResponseError`) → curated "no agent supplied" message; assert
   `FakeGraph.query` was **never called**.
3. `test_author_write_with_matching_agent_succeeds` — `CREATE (k:KaizenEntry {..., author:
   'graph-dba', ...})`, `agent='graph-dba'` → `query()` called with the verbatim text;
   write-summary rendered.
4. `test_author_write_with_mismatched_agent_is_rejected` — `author: 'cobb'` literal inside the
   `CREATE`'s map body, `agent='graph-dba'` → rejected; assert `query()` never called (AC-6, no
   partial write).
5. `test_curator_clear_shape_with_cobb_succeeds` — the exact `MATCH (e:KaizenEntry {entryId:
   '...'}) DETACH DELETE e` skeleton, `agent='cobb'` → allowed.
6. `test_curator_clear_shape_with_non_curator_is_rejected` — same skeleton, `agent='graph-dba'` →
   rejected (curator-only, not "any known agent").
7. `test_unrecognized_write_shape_is_rejected` — e.g. `MATCH (n) DETACH DELETE n` (no `author:`
   literal, not the curator skeleton) → rejected regardless of `agent`.
8. `test_migration_shaped_batch_with_single_shared_author_literal_succeeds` — the real §3.4 shape:
   `UNWIND [<six maps, no per-row author>] AS e CREATE (k:KaizenEntry {..., author: 'graph-dba',
   ...})`, `agent='graph-dba'` → one claim found, matches → allowed (proves the corrected migration
   Cypher, post-M1/M2 fix, still authorizes cleanly).
9. `test_write_to_nonexistent_graph_without_agent_is_graph_not_found` — "empty key" error, no
   `agent` → today's exact `graph_not_found_message()`, unchanged (regression).
10. `test_write_to_nonexistent_graph_with_agent_and_matching_author_creates_it` — "empty key",
    `agent='graph-dba'`, the §3.4-shaped `CREATE` with a matching `author:` literal → `query()`
    called (the migration path).
11. `test_write_to_nonexistent_graph_with_agent_but_non_write_text_is_graph_not_found` — **new,
    Pass-1 review M3.** "empty key", `agent='graph-dba'` (supplied out of habit), Cypher is a plain
    `MATCH (e:KaizenEntry) RETURN e.fact` (no write keyword) → `_looks_like_write()` is `False` →
    routed straight to `graph_not_found_message()`, `authorize_write()` never called. Pins the exact
    gap the review found: `agent` being set is no longer sufficient on its own to enter enforcement.
12. `test_profile_still_refused_regardless_of_agent` — `PROFILE` prefix, any `agent` → unchanged
    refusal, no `ro_query`/`query` call at all (proves `PROFILE` can't be used to dodge §3.2).
13. `test_explain_still_works_on_write_cypher` — `EXPLAIN` + a write statement → plan returned via
    the unchanged `explain()` path, regardless of `agent`; nothing executed.
14. `test_author_write_succeeds_despite_decoy_author_substring_in_evidence` — **new, Pass-1 review
    M1 / minor m2.** `CREATE (k:KaizenEntry {..., evidence: "the pattern author: 'someone-else'
    shows up in the log line", ..., author: 'graph-dba', ...})`, `agent='graph-dba'` → the decoy
    substring sits inside `evidence`'s own string-literal span and is excluded by
    `_string_literal_spans`; only the real top-level `author: 'graph-dba'` is found → allowed. Pins
    the exact false-rejection the review reproduced against real `inbox.md`-shaped content.
15. `test_set_based_author_reassignment_is_always_rejected` — **new, Pass-1 review M2 / minor m2.**
    `MATCH (e:KaizenEntry {entryId:'<not-mine>'}) SET e.author = 'graph-dba'`, `agent='graph-dba'`
    (a **matching** value) → still rejected, because `_kaizen_entry_create_map_spans()` finds no
    `CREATE` clause at all, so `_author_claims()` returns empty and the query falls to the final
    "neither shape recognized" branch. Proves the `SET` path is categorically unreachable, not just
    unmatched by coincidence — reproduces the review's exact finding as a regression pin.
16. `test_nested_create_decoy_in_free_text_is_excluded` — **new, Pass-2 review, M1-residual.** Two
    sub-cases, both copied verbatim from the review's own verified reproductions:
    - **Over-rejection:** `CREATE (real:KaizenEntry {fact: 'x', evidence: "example: CREATE
      (k:KaizenEntry {author: 'evil'})", context: 'c', suggestedHome: 'unsure', author:
      'graph-dba', createdAt: 't'})`, `agent='graph-dba'` → `_author_claims()` returns
      `['graph-dba']` only (the embedded `CREATE` inside `evidence`'s string literal is excluded by
      the whole-text `_string_literal_spans` pre-filter) → **allowed**. Before the fix this
      wrongly returned `['graph-dba', 'evil']` and rejected.
    - **Under-enforcement:** `CREATE (real:KaizenEntry {fact: 'x', evidence: "example: CREATE
      (k:KaizenEntry {author: 'graph-dba'})", context: 'c', suggestedHome: 'unsure', createdAt:
      't'})` (note: the real top-level map has **no** `author:` property at all), `agent=
      'graph-dba'` → `_author_claims()` returns `[]` (the only `author:`-shaped text is inside the
      excluded embedded clause) → falls to the curator-clear check (no match) → **rejected**
      ("neither shape recognized"). Before the fix this wrongly returned `None` (authorized) via
      the decoy's claim alone.

### 8.2 Per-acceptance-criterion strategy (step 5, `qa-engineer`)

| AC | How it's checked | Altitude |
|---|---|---|
| AC-1 | Live `mcp__cpg__query` call from a non-`graph-dba` agent context against `kaizen_graph_dba`, no `agent` param | Live, one call |
| AC-2 | `MATCH (e:KaizenEntry) RETURN count(e)` = 6 post-migration; spot-check 1–2 entries' fields against the pre-migration `inbox.md` text | Live |
| AC-3 | `git diff` on `inbox.md` shows only the prepended note (§3.3); read the rendered file, confirm the note reads as unambiguous "frozen" | Static + live-file read |
| AC-4 | `graph-dba` writes one real (or realistic test) new learning via `mcp__cpg__query(..., agent='graph-dba')`, then a second, independent read confirms it's queryable — no second copy anywhere | Live, exercises §3.1's "graph exists" write branch fresh (not the migration's "empty key" branch) |
| AC-5 | `cobb` runs the real §3.5 four-step sequence end to end on one live raw entry — append to `history.md`, confirm, then curator-clear. **Not a toy**: dispatch `cobb` for a real distillation pass, not a scripted stand-in | Live, full workflow — this is the criterion the brief flags as needing a real acceptance pass, not a unit test |
| AC-6 | Live: one call claiming `agent='graph-dba'` with an `author: 'cobb'` literal (and the reverse) → both rejected | Live, mirrors unit tests 3–4 but against the real server |
| AC-7 | Read `claude/AGENTS.md` + `skills/agent-maintenance/SKILL.md` §5 after step 4, confirm no sentence still describes `graph-dba`'s raw capture as appending to `inbox.md` | Static read |
| AC-8 | `grep -m1 -H 'Status:\|Note:' docs/requirements/cpg-query-access.md` shows the new pointer (revised — §5's edit uses plain `**Note:**` prose, not a `Superseded by:` field, per Pass-1 review m1); no contradicting claim found elsewhere | Static, one command |

Unit tests 1–16 (§8.1) are the automatable half; AC-1/AC-6 have unit-test mirrors but still get one
live confirmation each because the enforcement logic's real value is in the live FalkorDB
round-trip (the "empty key" vs. "read-only" distinction is a live FalkorDB behavior, `graph-dba`'s
inbox note itself only trusted it after a live check). AC-5 is the one criterion that cannot be
satisfied by anything short of a real `cobb` dispatch — flagged explicitly per the brief's own
instruction.

---

## 9. Risks & open questions

- **Ordering (§3.5) is procedural, not mechanical.** The append-before-delete constraint cannot be
  enforced by the MCP server (two independent tool calls, no shared transaction) — it lives entirely
  in `cobb`'s documented procedure (`skills/agent-maintenance/SKILL.md` §5, step 4b). A `cobb`
  dispatch that skips step 3 or reorders 3/4 would violate it with nothing in the tool stopping it.
  Accepted per `graph-dba`'s own framing ("enforcing the order is implementation, not schema") —
  flagged here so the plan-gate reviewer sees it as a known, named risk rather than a silent gap.
  §3.5 also resolves, with stated rationale, the review's open question of whether `graph-dba`'s own
  prompt needs to record this ordering too: **no**, `graph-dba` never runs the delete half, so only
  `cobb`'s side documents it.
- **Regex-based enforcement is bypassable by a creative caller via variable aliasing**
  (`WITH 'graph-dba' AS a CREATE (k:KaizenEntry {author: a})` — §3.2's known, accepted limitation,
  unchanged by the M1/M2 fixes). Accepted, matches FR-8's explicitly stated trust bar. **The two
  *unintended* gaps Pass-1 review found in the same area — an over-*rejection* risk from scanning
  free-text fields (M1) and an over-*authorization* risk from not distinguishing `CREATE` from `SET`
  (M2) — are fixed in §3.2**, not accepted; only the aliasing evasion (which under-rejects, and was
  always a stated, deliberate trade-off, not an oversight) remains open. If this pattern ever needs
  to survive an adversarial caller, the enforcement needs a real Cypher parser, which is a
  materially different (and heavier) design — not undertaken here.
- **`falkordb-py` 1.6.x mutation-counter attribute names are unverified in this plan.** The
  implementer (step 1) must confirm the real attribute names on `QueryResult` (or the equivalent
  the pinned client exposes) before writing `format_write_result()`; if none exist, fall back to a
  minimal "write ok" line with no stats, per this agent's standing "don't assert unverified
  behavior" convention (mirrored from `graph-dba`'s own kaizen notes).
- **Naming tension, deliberately not resolved now.** The server/tool stays named `cpg`/`query`
  while now also holding `kaizen_graph_dba` — a future reader unfamiliar with this document's
  history could reasonably ask why. Not renamed here because (a) FR-1's own wording assumes
  `mcp__cpg__query` stays the callable name, and (b) a rename's blast radius (every consuming
  agent's allowlist, `.mcp.json`, every doc citing the tool) is disproportionate to a one-agent
  pilot. **Revisit if** this pattern extends past `graph-dba` to a second agent's working memory —
  the same trigger `graph-dba`'s U1 note names for promoting `author` from a property to a node.
- **`_CURATOR_CLEAR_RE` and `_kaizen_entry_create_map_spans` are both hardcoded to `:KaizenEntry`.**
  Correct and narrow for this pilot; extending either shape to a second agent's inbox later needs
  the label parameterized (or the regex generalized) — flagged, not built, per the Out of scope line
  "moving any other agent's kaizen inbox to the graph."
- **`_looks_like_write()` (§3.1, M3's fix) is a lightweight, non-string-literal-aware keyword scan
  by design** — it only ever affects which curated *message* comes back on the narrow "empty key +
  agent supplied" path (never an authorization outcome, since `authorize_write()` itself is fully
  string-literal- and CREATE-span-aware), and it only matters pre-migration or on a graph-name typo.
  A read whose free-text predicate quotes a write keyword (`WHERE e.context CONTAINS 'DETACH
  DELETE'` — genuinely plausible against this exact corpus, see `claude/graph-dba/kaizen/inbox.md`'s
  2026-08-16 `DETACH DELETE` entries) could false-positive into `authorize_write()` and get "not a
  recognized write shape" instead of "graph not found" — a less-precise message, never a wrong
  authorization, and only reachable before `kaizen_graph_dba` exists for good. Accepted as
  sufficient for a pre-classification step, not gold-plated into a second string-literal-aware pass.
- **Resolved by this revision (previously an open question posed to the `analyst` plan-gate):** is a
  static regex check an acceptable substitute for FR-8's enforcement logic? Pass-1 review answered
  directly — **not as originally scoped** (M1/M2/M3 were concrete, real gaps, now fixed above), but
  the *shape* (a static, pre-execution, structurally-scoped text check, no parser) remains
  right-sized for FR-8's stated trust bar once those three are closed; a full Cypher parser is not
  warranted by anything in the requirements doc. This plan adopts that verdict rather than
  re-litigating it.

---

## 10. Revision note (Pass 2 — addresses `analyst`'s U3 Pass-1 plan-gate)

2026-08-17 — `docs/reviews/generic-cypher-mcp.md` (verdict: needs changes) found one blocker and
three majors, all addressed in place in this revision (`Version: 1.1`); U1 (`graph-dba`'s schema
note) had zero findings and is unchanged.

- **B1 (blocker) — fixed.** §7's step 4 is split into 4a/4b, adding the two previously-missing
  operative-prompt files (`claude/graph-dba/graph-dba.md`, `claude/cobb/cobb.md`) plus
  `claude/README.md`, and the close-out condition is now a `grep`-based search over `claude/` and
  `skills/agent-maintenance/SKILL.md` rather than a fixed enumeration, so a doc this pass didn't
  think to name doesn't repeat the gap. §6's BACKLOG.md proposal renumbered `C-501…C-506`
  accordingly.
- **M1 (major) — fixed.** §3.2's `_author_claims()` is redesigned to scan only the body of a
  `CREATE (...:KaizenEntry {...})` map literal, and only the part of that body outside any nested
  string literal (`_string_literal_spans`) — a decoy `author:`-shaped substring inside `evidence`/
  `context` free text can no longer be misread as a claim. New test 14 (§8.1) pins this.
- **M2 (major) — fixed.** The same redesign closes M2 as a side effect: `_kaizen_entry_create_map_spans()`
  only ever finds `CREATE` clauses, so a `SET <var>.author = '<agent>'` reassignment of an
  *existing*, not-necessarily-owned node produces zero spans to search and is always rejected,
  regardless of whether the value matches `agent` — chosen over the review's alternative
  (`WHERE`-ownership guard on `SET`) as simpler and safer, since FR-8 only asks for entry
  *creation*. New test 15 (§8.1) pins this as a regression, including the review's exact
  reproduction case.
- **M3 (major) — fixed.** §3.1's "empty key" branch now classifies write-vs-read from the Cypher
  text itself (`_looks_like_write()`, a lightweight keyword scan) before ever calling
  `authorize_write()` — `agent` being supplied is no longer, by itself, treated as proof of write
  intent. A plain read against a missing/mistyped graph name now correctly falls through to
  `graph_not_found_message()` regardless of whether `agent` was set. New test 11 (§8.1) pins this.
- **m1 (minor) — addressed.** §5's header edit now uses plain `**Note:**` prose instead of the
  `Supersedes:`/`Superseded by:` field pair, avoiding the cross-topic-slug reuse of a field the
  doc-lifecycle convention introduces for same-slug ordinal successors.
- **m2 (minor) — addressed.** §8.1 gained tests 14 and 15 (above), covering exactly the two failure
  modes the review named.
- **Open question — resolved.** §3.5 states the decision (with rationale) that the append-before-
  delete ordering is documented solely in `agent-maintenance` §5 (`cobb`'s side), not duplicated
  into `graph-dba.md`, because `graph-dba` never runs the delete half of the sequence.
- **§3.4 (migration Cypher shape)** was additionally revised, not because the review flagged it
  directly, but because M1/M2's fix changes what shape of Cypher the migration must send to stay
  authorizable: the `author: 'graph-dba'` literal now lives once in the `CREATE` clause itself
  (not per-row inside the `UNWIND` list, which no longer carries an `author` key at all) — see §3.4
  and test 8 (§8.1).

## 11. Revision note (Pass 3 — addresses `analyst`'s U3 Pass-2 re-gate)

2026-08-17 — `docs/reviews/generic-cypher-mcp.md` `## Pass 2 — 2026-08-17` (verdict: needs changes)
verified Version 1.1 by directly executing the plan's own Python (`_string_literal_spans`,
`_kaizen_entry_create_map_spans`, `_author_claims`, `authorize_write`, `_looks_like_write`), not
just re-reading it. B1, M2, M3, m1, m2 were all confirmed **closed** by that execution. One Major
remained open (M1-residual) plus one new, non-blocking minor (m3). Both addressed in place in this
revision (`Version: 1.2`).

- **M1-residual (major) — fixed.** `_kaizen_entry_create_map_spans()`'s `CREATE`-keyword location
  step was only body-extraction-aware, not itself string-literal-aware, so a free-text field
  quoting a *complete* `CREATE (...:KaizenEntry {...})`-shaped example (not just a bare `author:`
  fragment) registered as a spurious second top-level clause — causing both a verified
  over-rejection case and a more serious verified under-enforcement (bypass) case. Fixed with the
  reviewer's own supplied, execution-verified one-method patch: compute `_string_literal_spans`
  once over the whole `cypher` text before locating `CREATE` candidates, and skip any match that
  falls inside one of those spans (§3.2's code block and its new "M1-residual, explicitly"
  paragraph). New test 16 (§8.1) reproduces both adversarial cases verbatim from the review, both
  now landing on the correct outcome (allowed / rejected respectively).
- **m3 (minor) — addressed.** §7's 4a/4b close-out done-condition now states the sweep's real
  volume explicitly ("expect on the order of 30 hits on the current tree... only the five named
  files require a change") instead of implying a short list, per the review's actual `grep -rln`
  run (35 hits).
- Nothing else changed — Pass 2 confirmed every other Version 1.1 finding (B1, M2, M3, m1, m2, the
  §3.4 migration-shape revision, the §3.5 open-question resolution) holds as-is.
