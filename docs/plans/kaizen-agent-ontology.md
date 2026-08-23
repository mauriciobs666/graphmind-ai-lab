# Kaizen agent/learning-note ontology — Implementation Plan (M8)

> **Status:** archived · **Owner:** `architect` · **Tracks:** — (M8) · **Version:** 3

## Revision note — 2026-08-22 (Version 3)

Revised in place per `analyst`'s **Pass 2** re-review of Version 2
(`docs/reviews/kaizen-agent-ontology.md`, `## Pass 2` section, verdict **needs changes**, one
Blocker; findings 2-4 from Pass 1 re-confirmed correctly resolved, no other gap found across
shapes 1-6). The Blocker: §3.1a's closure was **under-scoped** — `_FOREIGN_TRIGGER_RE` screened
only for a bare `MERGE`/`DELETE`, not the full write-keyword set this same file already enumerates
elsewhere (`_WRITE_KEYWORD_RE`: `CREATE|MERGE|SET|DELETE|REMOVE`). A third attack (**Attack C**,
the review's own reproduction) chains a self-attributed decoy `CREATE` with a `MATCH ... SET
victim.author = '<other-agent>', victim.fact = 'tampered'` clause against a node the caller doesn't
own — still authorized under Version 2's design, and its `victim.author = '<other-agent>'`
sub-case is precisely the SET-based author-reassignment attack tests 15/15b were written to keep
closed (those two only pin a *standalone* `SET` with no `CREATE` anywhere in the statement, so they
never exercised this chained case). Fixed, mechanically, no design rework:

1. `_FOREIGN_TRIGGER_RE` widened to `r"\b(?:MERGE|DELETE|SET|REMOVE)\b"` — see the revised §3.1a.
2. Two more pinned regression tests added to §5 alongside items 18/19: item 22 (Attack C's
   `SET`-chained variant, including the `victim.author = '<other-agent>'` sub-case explicitly) and
   item 23 (a `REMOVE`-chained variant).
3. The mutation-testing directive extended to also confirm dropping `SET|REMOVE` from the widened
   regex is caught by at least one of items 22/23.
4. The "closed" language in §3.1a and the Version-2 revision note is softened to state the closure
   is scoped to the full write-keyword set (`MERGE|DELETE|SET|REMOVE`), not just two of its four
   members — the earlier unqualified "closed" was premature.

Nothing else changes: the surrounding string-literal-exclusion logic, the `if claims:` gating, and
the confirmation that a genuine producer-write is never subjected to this check at all (it returns
before `claims` is consulted) all hold unmodified, independently re-verified by Pass 2.

## Revision note — 2026-08-22 (Version 2)

Revised in place per `analyst`'s plan-gate review (`docs/reviews/kaizen-agent-ontology.md`,
verdict **approve with suggestions**), all four findings addressed:

1. **Finding 1 (Major) — closed, per the stakeholder's direction, not accepted-as-risk.** The
   review traced a real gap: `authorize_write()`'s shape-1 (author-write) path returns "authorized"
   the instant `_author_claims()` finds a self-matching claim *anywhere* in the text, without ever
   checking whether the rest of the same statement hides an unrelated second clause — a
   curator-only `DETACH DELETE`, or (this plan's own new, higher-stakes case) a
   `MERGE (:Agent {agentId:'<other-agent>'})`-based provenance forgery. §3.1 now adds one new,
   narrow check — `_has_foreign_trigger_outside_strings()` — closing both of the review's traced
   attacks. See the revised §3.1 for the exact mechanism and §5 for the two new pinned regression
   tests reproducing the review's own attack text verbatim. **(Version 3 note: this check's initial
   keyword set was itself under-scoped — see the Version 3 revision note above; the mechanism this
   note describes is otherwise unchanged.)**
2. **Finding 2 (Major) — the MENTIONS-before-count ordering invariant is now stated explicitly**
   in §3.3 and S4's done-condition, not left implicit in the step numbering.
3. **Finding 3 (Minor) — the step table's S3/S4 "Depends on" cells no longer overstate a hard
   gate**; reworded to match the softer, cost-weighed recommendation the surrounding prose already
   argued for.
4. **A `qa-engineer` step is added** (S6), per the review's recommendation and the stakeholder's
   acceptance of it — a live dry-run pass confirming finding 1's closure holds against the
   deployed container, plus one real distillation dry-run, complementing (not duplicating) S1's
   own scripted live acceptance sub-step. §4's step table and dependency graph updated accordingly.

Everything below is the revised document; nothing from Version 1 is preserved by silent omission —
where a section changed, it changed in place.

---

Implements [`../requirements/kaizen-agent-ontology.md`](../requirements/kaizen-agent-ontology.md)
(FR-1…FR-8, AC-1…AC-7, all read in full — relationship names/directions `PRODUCED`/`MENTIONS`
and the `author` property's outright removal are **locked**, not relitigated here) against the
schema/write-shape design in
[`kaizen-agent-ontology-graph.md`](./kaizen-agent-ontology-graph.md) (`graph-dba`, read in full —
its §1 crux, §2 producer write, §3 MENTIONS write, §4 deletion, §5 query patterns, §6
indexing/DDL are design decisions this plan sequences and assigns, not re-derives). Coordinated at
[`kaizen-agent-ontology-coordination.md`](./kaizen-agent-ontology-coordination.md) (`teco`).

**CPG:** considered, not relevant — this is a code-level task (`cypher-mcp/server.py`'s
`authorize_write()`), but `GRAPH.LIST` against the live FalkorDB instance (queried directly for
this plan, 2026-08-22) shows only `cpg_salesperson` and `cpg_falkorchat` loaded; `cypher-mcp` and
`claude/` carry no CPG, so the `cpg-analysis` skill has nothing to query here.

---

## 1. Goal & scope

Replace the plain `author` string property on `:KaizenEntry` nodes (in the shared `kaizen_team`
FalkorDB graph, all 13 team agents writing to it) with real `:Agent` identity nodes and two
locked, directional relationships — `(:Agent)-[:PRODUCED {sessionId}]->(:KaizenEntry)` and
`(:KaizenEntry)-[:MENTIONS]->(:Agent)` — for every entry created after this feature ships. This
requires, in the same delivery: growing `cypher-mcp/server.py`'s `authorize_write()` from 2 to 6
recognized write shapes (the load-bearing security gate this whole feature routes through),
retargeting all 13 agents' "Learning capture" prompt sections to the new producer-write shape,
rewriting `cobb`'s distillation procedure (`skills/agent-maintenance/SKILL.md` §5) for the new
read/route/delete mechanics, and updating the docs that describe the mechanism
(`cypher-mcp/README.md`, `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`).

**Out of scope** (per the requirements doc, not revisited here): retrofitting the 20 historical
`:KaizenEntry` nodes currently in `kaizen_team` (confirmed live, 2026-08-22 — zero `:Agent` nodes
exist yet) with the new edges; any change to `cobb`'s distillation cadence or `history.md`'s
output format; anything starting before M7 (already landed and archived — FR-7/AC-6 satisfied).

## 2. Context & findings

### 2.1 The substrate (M7, already shipped)

One shared `kaizen_team` graph, `author`-partitioned, holds `:KaizenEntry` nodes with
`entryId, date, fact, evidence, context, suggestedHome, author, createdAt` (plus `sessionId` for
entries created after M7). Live-confirmed 2026-08-22: `kaizen_team` has 20 `:KaizenEntry` nodes,
labels = `[KaizenEntry]` only — no `:Agent` label exists yet, so the new index/constraint (§4,
S0) is issued against a genuinely empty label, no idempotency question.

### 2.2 The schema this plan targets (graph-dba, full detail in the graph note)

```
(:Agent {agentId})
(:Agent)-[:PRODUCED {sessionId}]->(:KaizenEntry)     // FR-2 — locked name/direction
(:KaizenEntry)-[:MENTIONS]->(:Agent)                 // FR-3 — locked name/direction, 0..N
```

`agentId` is identity-only (the requirements doc's one open question, resolved by its own
default) — the same slug already used as `author`'s value, the MCP tool's `agent` parameter, and
`CYPHER_MCP_CURATOR_AGENTS`'s members. `author` is dropped outright for new entries (FR-2, no
coexisting field); `sessionId` moves from the node onto the `PRODUCED` edge (FR-8) for new
entries only — M7-window entries (after M7, before M8) keep it on the node, unmigrated (AC-7).

### 2.3 `cypher-mcp/server.py` today (read directly, `cypher-mcp/server.py:224-382`)

`authorize_write()` recognizes exactly **two** write shapes, both keyed on an `author:` literal or
a fixed `entryId`-scoped skeleton:

1. **Author-write** — `_author_claims()` (line 334) finds every literal `author: '<value>'`
   strictly inside a `CREATE (<var>:KaizenEntry {...})` map body, located by
   `_kaizen_entry_create_map_spans()` (line 291): a string-literal-aware scan that anchors on
   `\bCREATE\b` immediately followed by `\(\s*[a-zA-Z_]\w*\s*:\s*KaizenEntry\s*\{` — i.e. the
   `KaizenEntry`-labeled node must be the pattern element **immediately after `CREATE`**, reached
   via no relationship — then brace-depth-matches the map body (string-literal aware, so a
   free-text field's own `{`/`}` or quoted decoy text can't desync it, lines 313-330). A claim
   matching the declared `agent` exactly authorizes; a mismatch rejects.
2. **Curator-clear** — the one whitelisted skeleton `MATCH (var:KaizenEntry {entryId: '...'})
   DETACH DELETE var` (`_CURATOR_CLEAR_RE`, line 265), matched against a **whitespace-collapsed**
   copy of the statement (`" ".join(cypher.split())`), gated to `CURATOR_AGENTS` (default `cobb`).

If neither matches, rejection is unconditional (line 376-381). **FR-2 drops `author` outright, so
every FR-2-conformant write finds zero claims and falls straight through to rejection** — this is
the crux graph-dba's note documents in full (its §1) and this plan does not re-derive: the
mechanism's only non-curator "allow" path is the thing FR-2 removes, independent of `MERGE` vs.
`CREATE` choice, and independent of whether `author` is kept as a redundant property (the
producer-write shape puts `KaizenEntry` **second** in the pattern, reached via `[:PRODUCED]` from
the `Agent` node — not the pattern element immediately after `CREATE` — so even a
hypothetical FR-2-noncompliant design would still fail today's scanner on that structural ground
alone).

### 2.4 `cypher-mcp/tests/test_server.py` (read directly, 954 lines)

Section 8 ("write authorization", lines 562-815) is the existing coverage this plan's new tests
extend, not replace: 16 parametrized/individual cases (read-without-agent, write-without-agent,
matching/mismatched author, curator-clear allowed/denied, unrecognized shape, the real §3.4
migration batch shape, empty-key-branch classification, decoys in free text on both the
over-rejection and under-enforcement side, `SET`-based reassignment always rejected). A `live`
section (lines 817-954) runs real Cypher against a throwaway scratch graph
(`_cypher_mcp_selftest_<uuid4>`, created and `graph.delete()`d by a module-scoped fixture) —
deliberately never touching `cpg_*`, `ws:*`, `reference`, or (implicitly, since it predates this
feature) `kaizen_team` itself.

### 2.5 `cypher-mcp/README.md` — "Writing through this tool" (lines 129-…)

Documents the two shapes with copy-pasteable examples, and explicitly states schema DDL
(`CREATE INDEX`, `GRAPH.CONSTRAINT CREATE`) is rejected unconditionally, live-verified during M7's
own `S0` — the same constraint this plan's DDL step (§4, S0) must fall back to `redis-cli` for.

### 2.6 The 13 agent prompts (grepped directly, not assumed)

`grep -rl "KaizenEntry" claude/` finds exactly 13 files with a "Learning capture" section:
`analyst`, `architect`, `cobb`, `coder`, `data-scientist`, `devops`, `frontend-engineer`,
`graph-dba`, `qa-engineer`, `security-expert`, `tdd-engineer`, `teco`, `tico`. Each one's fenced
Cypher block is **structurally identical** — only the `author: '<agent-slug>'` value (already
per-file) differs:

```cypher
CREATE (k:KaizenEntry {
  entryId: '<uuid4>', date: '<YYYY-MM-DD>', fact: '<the fact, one line>',
  evidence: '<what was run/read/observed>', context: '<the task where it surfaced, one line>',
  suggestedHome: 'prompt | knowledge base | project docs | unsure',
  author: '<agent-slug>', createdAt: '<ISO-8601 write time>',
  sessionId: '<value of $CLAUDE_CODE_SESSION_ID, or omit this key entirely if unavailable>'
})
```
called as `mcp__cypher__query(graph='kaizen_team', cypher=<that text>, agent='<agent-slug>')`.

The **surrounding prose is not uniform** — verified by reading all 13, not assumed: `cobb`'s
paragraph has an extra clause about self-promoting in the same run; `devops` has an extra
parenthetical ("this graph resolves in every project"); `qa-engineer` has one ("defects belong in
the test report, not here"); `graph-dba`'s opening sentence differs (FalkorDB-quirk routing);
`security-expert`'s closing sentence lacks the "This replaces the earlier `kaizen/inbox.md`..."
clause the other 12 carry (it postdates the inbox removal). **The "called as ..." line is
identical across all 13 and does not change under this feature** — the `agent` parameter still
names the caller; only the fenced Cypher block changes. This narrows the per-file edit to exactly
the fenced block, which *is* uniform in shape (§4, S3) — the one place a blanket transform is
actually safe here, not the whole section.

### 2.7 `cobb`'s distillation procedure (`skills/agent-maintenance/SKILL.md` §5, read in full)

The full read→verify→route→log-and-clear procedure lives here, not in `cobb.md` itself. Step 1's
read query is today a plain `MATCH (e:KaizenEntry) RETURN ... ORDER BY e.date` (optionally scoped
by `{author: '<agent>'}`). Step 4's clear is always the one curator-clear shape, unconditionally —
there is no partial-edge concept yet because there are no edges yet. Both need rewriting for M8
(§4, S4).

### 2.8 Doc surfaces describing the mechanism in prose

`claude/README.md`'s "Kaizen" section (lines 21-58) carries the copy-pasteable one-query recipe
(`MATCH (e:KaizenEntry) RETURN e.author, ...`) and describes attribution as "its own `author`
value." `claude/AGENTS.md`'s opening paragraph and root `AGENTS.md`'s `claude/` bullet both
describe the graph as "`author`-partitioned" with no `:Agent`/relationship concept. All three need
updating to describe the new ontology (§4, S5) — none of this is new-file work, all three already
exist and already carry the paragraph being amended.

### 2.9 `docs/BACKLOG.md` / `docs/HISTORY.md` convention

Root `docs/BACKLOG.md` already carries cross-cutting milestone rows in this exact shape for M2, M3,
M5, M6, **M7** (item-ID prefix `C-`, hundreds digit = milestone number — M7 used `C-701…C-721`).
M8 will use `C-801…C-8xx`. Per precedent (M7's row was compiled and enriched across that
milestone's whole lifecycle — initial stub, then gate outcomes, then a 2026-08-21 close-out
addendum — not written in one shot by one implementing unit), this plan treats the `BACKLOG.md`
row and `HISTORY.md`'s eventual closing entry as `teco`'s standing documentation-curator duty
(§4), not a discrete implementation step with its own file-list entry.

## 3. Design & rationale

### 3.1 `authorize_write()` — from 2 to 6 recognized shapes

Graph-dba's §1 established *that* 6 shapes are needed and *what* each one authorizes; this section
is the concrete parsing design (graph-dba's own explicit non-goal, left to this plan). The design
keeps every existing shape and its exact rejection/success wording untouched, and adds four new,
narrow, fixed-skeleton recognizers — same complexity class as today's one curator-clear regex,
not a general Cypher parser (FR-8's stated trust bar: well-behaved callers, not a malicious one).

**Shape 3 (new, non-curator) — producer-write.** Any agent, its own `agentId`, FR-2/FR-8:

```cypher
MERGE (a:Agent {agentId: '<agent-slug>'})
CREATE (a)-[:PRODUCED {
  sessionId: '<value of $CLAUDE_CODE_SESSION_ID, or omit this key entirely if unavailable>'
}]->(k:KaizenEntry {
  entryId: '<uuid4>', date: '<YYYY-MM-DD>', fact: '<the fact, one line>',
  evidence: '<what was run/read/observed>', context: '<the task where it surfaced, one line>',
  suggestedHome: 'prompt | knowledge base | project docs | unsure',
  createdAt: '<ISO-8601 write time>'
})
```

This is the hard case: the `KaizenEntry` map's free-text fields (`fact`/`evidence`/`context`) can
legitimately contain literal `{`/`}` characters (exactly the concern the existing
`_kaizen_entry_create_map_spans()` already solves for the old shape), so a `[^}]*`-style regex is
unsafe. **Recognition algorithm** (operates on the *original*, non-whitespace-collapsed `cypher`
text — like the existing author-claim path, not like curator-clear's normalized-string path,
because whitespace inside a free-text field must not be silently altered before the brace-depth
scan runs):

1. Factor the existing depth-counting loop in `_kaizen_entry_create_map_spans()`
   (`cypher-mcp/server.py:313-330`) into a shared helper,
   `_scan_matched_brace(text: str, open_index: int) -> int | None` — given `text[open_index] ==
   "{"`, returns the index just past the matching `"}"` (string-literal aware, same logic,
   verbatim) or `None` if unterminated. Re-point `_kaizen_entry_create_map_spans()` at this helper
   — a pure refactor, behavior-preserving; run the full existing suite before adding anything new
   to confirm zero behavior change from the refactor alone.
2. New function `_producer_write_agent_id(cypher: str) -> str | None`:
   a. Match `\A\s*MERGE\s*\(\s*([a-zA-Z_]\w*)\s*:\s*Agent\s*\{\s*agentId\s*:\s*(['"])([^'"]+)\2\s*\}\s*\)\s*`
      at the very start of the string (case-insensitive on keywords). No match → `None`. Capture
      the bound variable name (`var`) and the claimed `agentId`.
   b. Immediately after, require `CREATE\s*\(\s*<var>\s*\)\s*-\s*\[\s*:\s*PRODUCED\s*` (with `<var>`
      substituted via `re.escape`, enforcing the `CREATE` clause references the *same* variable the
      `MERGE` just bound). No match → `None`.
   c. If what follows (after skipping whitespace) is `{`, brace-depth-match it via
      `_scan_matched_brace()` (the optional `{sessionId: '...'}` property map) and advance past it;
      if it is anything else (e.g. `]`), skip this step — the map is optional (the template's own
      "or omit this key entirely if unavailable").
   d. Require `\s*\]\s*->\s*\(\s*[a-zA-Z_]\w*\s*:\s*KaizenEntry\s*\{` next. No match → `None`.
      Brace-depth-match the `KaizenEntry` map body via `_scan_matched_brace()`.
   e. Require **exactly** `\s*\)\s*;?\s*\Z` after that map's closing `}` — the `CREATE` clause's
      node-pattern close, an optional semicolon, then **end of string**. Anything else trailing
      (a stray extra clause) → `None`. This is what makes the shape "exactly one `MERGE` clause
      followed by exactly one `CREATE` clause, nothing else" (graph-dba §1), not merely "contains."
   f. Return the captured `agentId`.
3. In `authorize_write()`, after the existing `claims = _author_claims(cypher)` check finds
   nothing (which it always will for an FR-2-conformant write, by construction — §2.3), call
   `_producer_write_agent_id(cypher)`. If it returns a value: mismatch against the declared `agent`
   → reject (wording mirrors the existing author-mismatch message); match → authorize (`return
   None`), **not** curator-gated — any agent may run this for its own `agentId`.

**Shapes 4-6 (new, curator-gated)** — no free text, no brace-matching needed; matched the same way
as today's curator-clear, against `" ".join(cypher.split())`:

```python
_MENTIONS_WRITE_RE = re.compile(
    r"^MATCH \(([a-zA-Z_]\w*):KaizenEntry \{entryId: ['\"][^'\"]+['\"]\}\) "
    r"MERGE \(([a-zA-Z_]\w*):Agent \{agentId: ['\"][^'\"]+['\"]\}\) "
    r"MERGE \(\1\)-\[:MENTIONS\]->\(\2\);?$",
    re.IGNORECASE,
)
_PRODUCER_EDGE_RESOLVE_RE = re.compile(
    r"^MATCH \(:Agent\)-\[([a-zA-Z_]\w*):PRODUCED\]->\([a-zA-Z_]\w*:KaizenEntry "
    r"\{entryId: ['\"][^'\"]+['\"]\}\) DELETE \1;?$",
    re.IGNORECASE,
)
_MENTION_EDGE_RESOLVE_RE = re.compile(
    r"^MATCH \([a-zA-Z_]\w*:KaizenEntry \{entryId: ['\"][^'\"]+['\"]\}\)-\[([a-zA-Z_]\w*):MENTIONS\]"
    r"->\(:Agent \{agentId: ['\"][^'\"]+['\"]\}\) DELETE \1;?$",
    re.IGNORECASE,
)
```

Each verified by hand against graph-dba's exact recipes (§3, §4.2 of the graph note) collapsed
through `" ".join(text.split())` — they match verbatim. Note the backreferences (`\1`/`\2`)
enforce the `DELETE`/second-`MERGE` clause references the *same* variable the `MATCH`/first-`MERGE`
bound — a stricter check than today's existing `_CURATOR_CLEAR_RE`, which does **not**
backreference its `MATCH`/`DETACH DELETE` variables (a pre-existing looseness, left as-is per
"don't touch shapes 1-2"; the new shapes don't need to repeat it).

**`authorize_write()`'s new control flow** (shapes 1-2 keep their exact existing code/wording;
only the additions are new):

```python
def authorize_write(cypher, agent):
    if not agent:
        return ...  # unchanged

    claims = _author_claims(cypher)          # unchanged
    if claims:
        ...                                   # unchanged

    producer_agent_id = _producer_write_agent_id(cypher)   # NEW
    if producer_agent_id is not None:
        if producer_agent_id != agent:
            return (
                f"Rejected: this write's MERGE (:Agent {{agentId: '{producer_agent_id}'}}) "
                f"claims a different agent than the call declared (agent='{agent}'). One "
                "agent's write cannot be accepted as another's (FR-8)."
            )
        return None   # producer-write, any agent may write its own

    normalized = " ".join(cypher.split())

    if _CURATOR_CLEAR_RE.match(normalized):      # unchanged
        ...

    if (_MENTIONS_WRITE_RE.match(normalized)      # NEW
            or _PRODUCER_EDGE_RESOLVE_RE.match(normalized)
            or _MENTION_EDGE_RESOLVE_RE.match(normalized)):
        if agent in CURATOR_AGENTS:
            return None
        return (
            f"Rejected: this is a curator-only write shape (MENTIONS tagging or edge "
            f"resolution), but agent='{agent}' is not a recognized curator "
            f"({sorted(CURATOR_AGENTS)})."
        )

    return (
        "Rejected: this write matches none of the recognized shapes — an author-write, a "
        "producer-write (MERGE (:Agent {agentId:...}) + CREATE (...)-[:PRODUCED]->(...:"
        "KaizenEntry {...})), a curator MENTIONS-write, a curator edge-resolve (PRODUCED or "
        "MENTIONS), or the curator full-node clear. This tool only authorizes those shapes (FR-8)."
    )
```

`TOOL_DESCRIPTION`, `SERVER_INSTRUCTIONS`, and the module's own top-of-file docstring
(`cypher-mcp/server.py:1-42`, which currently says "Only two write shapes are ever authorized")
all need the same update — a short enumeration of all 6, not the full regex detail.

**Why keep shapes 1-2 recognized at all, if no current agent will ever emit an author-write again
after S3 lands?** Because nothing in FR-2 forbids a historical-shaped write from existing, the
brief's own test-strategy explicitly asks to "confirm this doesn't regress the two existing write
shapes," and — the number graph-dba's note states independently — 2 old + 4 new = 6, which only
adds up if shapes 1-2 are additions-to, not replacements-of. Removing them would also strand the
existing regression suite's 16 cases. Decision: **shapes 1-2 are untouched, not deprecated** —
except for the one closure below, which changes shape 1's *acceptance condition*, not its claim
extraction.

### 3.1a Closing the cross-clause smuggling gap (analyst review, Finding 1)

**The gap, traced by the review, not hypothetical.** `_kaizen_entry_create_map_spans()` scans the
*entire* cypher text for every `CREATE (<var>:KaizenEntry {...})` occurrence, not just the first,
and `authorize_write()`'s shape-1 branch returns "authorized" the moment every claim it finds
matches the declared `agent` — **without ever checking whether the rest of the same statement
contains an unrelated second clause.** Because ordinary multi-clause Cypher (one `GRAPH.QUERY`
call, several top-level clauses) is exactly what the accepted §3.4 migration-batch shape already
relies on, a caller can chain a harmless, self-attributed decoy `CREATE (...:KaizenEntry
{..., author:'<self>', ...})` in front of *anything else* and have the whole statement authorize on
the decoy's strength alone. Three concrete instances (the review's own reproductions — A and B from
Pass 1, C added in Pass 2 once A/B's fix exposed the same gap was wider than `MERGE`/`DELETE`
alone — all preserved verbatim as new tests in §5):

```cypher
-- Attack A (pre-existing gap, not introduced by this plan): smuggles an unauthorized delete
CREATE (junk:KaizenEntry {entryId:'z1', date:'2026-08-22', fact:'f', evidence:'e', context:'c',
  suggestedHome:'unsure', author:'analyst', createdAt:'t'})
MATCH (victim:KaizenEntry {entryId:'not-mine'}) DETACH DELETE victim
```
```cypher
-- Attack B (this plan's own new, higher-stakes case): forges another agent's provenance
CREATE (junk:KaizenEntry {entryId:'z1', date:'2026-08-22', fact:'f', evidence:'e', context:'c',
  suggestedHome:'unsure', author:'analyst', createdAt:'t'})
MERGE (a:Agent {agentId: 'cobb'})
CREATE (a)-[:PRODUCED {sessionId:'s'}]->(k:KaizenEntry {entryId:'forged', date:'2026-08-22',
  fact:'fabricated, attributed to cobb', evidence:'e', context:'c', suggestedHome:'unsure',
  createdAt:'t'})
```
```cypher
-- Attack C (Pass 2, analyst): smuggles arbitrary property tampering, including author reassignment
CREATE (junk:KaizenEntry {entryId:'z1', date:'2026-08-22', fact:'f', evidence:'e', context:'c',
  suggestedHome:'unsure', author:'analyst', createdAt:'t'})
MATCH (victim:KaizenEntry {entryId:'not-mine'})
SET victim.author = 'nobody', victim.fact = 'tampered'
```
All three, called with `agent='analyst'` (not a curator, not `'cobb'`, not the victim's own agent):
`_author_claims()` finds exactly one claim (`'analyst'`, from the first `CREATE`) matching the
declared agent, so an unfixed `authorize_write()` returns `None` before ever evaluating anything
about the second clause. **Attack C's `victim.author = '<other-agent>'` sub-case is, specifically,
the exact SET-based author-reassignment scenario tests 15/15b (`cypher-mcp/tests/test_server.py`)
exist to keep closed** — those two tests only pin a *standalone* `SET` with no `CREATE` clause
anywhere in the same statement (so `_author_claims()` finds nothing and the statement never
reaches the "already accepted, check for chaining" branch at all); neither one exercises this
chained case, which is why Pass 1's closure (scoped to `MERGE`/`DELETE` only) left it open.

**Resolution chosen — close it, not merely pin/accept it (stakeholder direction).** Generalize
shape 1's acceptance condition with one new, narrow check, run only in the branch that would
otherwise accept (claims found, all matching `agent` — mismatches are unaffected, already
rejected):

```python
# Same keyword set as `_WRITE_KEYWORD_RE` (line 239), minus `CREATE` — every write
# keyword a *second*, unrelated clause could open with, since an accepted shape-1
# statement's own `CREATE` is exactly what this check must not itself trip on.
_FOREIGN_TRIGGER_RE = re.compile(r"\b(?:MERGE|DELETE|SET|REMOVE)\b", re.IGNORECASE)

def _has_foreign_trigger_outside_strings(cypher: str) -> bool:
    """True if `cypher` contains a bare (not inside a string literal) `MERGE`,
    `DELETE`, `SET`, or `REMOVE` keyword anywhere in the statement — i.e. a
    second, different recognized shape's trigger (or an arbitrary-tampering
    `SET`/`REMOVE`) chained alongside an accepted author-write clause (Finding 1,
    Pass 1 + Pass 2, docs/reviews/kaizen-agent-ontology.md)."""
    literal_ranges = _string_literal_spans(cypher)
    for m in _FOREIGN_TRIGGER_RE.finditer(cypher):
        if not any(s <= m.start() < e for s, e in literal_ranges):
            return True
    return False
```

```python
claims = _author_claims(cypher)
if claims:
    mismatched = [c for c in claims if c != agent]
    if mismatched:
        return ...   # unchanged
    if _has_foreign_trigger_outside_strings(cypher):        # NEW
        return (
            "Rejected: this statement combines a valid author-write with another "
            "recognized shape's trigger (a bare MERGE, DELETE, SET, or REMOVE elsewhere "
            "in the same statement) — chaining an unrelated clause onto a self-attributed "
            "CREATE is not authorized, regardless of the author-write's own validity."
        )
    return None
```

**Why a bare keyword scan, not clause-span bookkeeping.** An earlier design considered computing
the full `[start, end)` span of every accepted `CREATE (...:KaizenEntry {...})` clause and scanning
only the *excised* leftover for other-shape triggers — rejected as unnecessary complexity: neither
of the two accepted shape-1 statements (a plain author-write, or the §3.4 migration-batch's
`UNWIND [...] AS e CREATE (k:KaizenEntry {...})`) ever legitimately contains the bare word `MERGE`,
`DELETE`, `SET`, or `REMOVE` anywhere in its own structure outside a string literal — property keys
are `entryId`, `date`, `fact`, `evidence`, `context`, `suggestedHome`, `author`, `createdAt`,
`sessionId`, none of which contain any of the four words, and `\b...\b` word-boundaries mean a decoy
value like `'DELETED'`, `'DELETE_marker'`, or (the closest near-miss) `'suggestedHome'` cannot
false-positive-match `SET` — there is no token boundary between `suggestedHome`'s own internal
characters for `\bSET\b` to land on. So a **whole-statement** bare-keyword scan over the *full*
write-keyword set (minus `CREATE`, which every legitimate shape-1 statement legitimately contains)
is both sufficient and simpler than tracking which spans are "already accounted for," and including
`SET`/`REMOVE` is exactly as safe as including `MERGE`/`DELETE` already was — same reasoning,
applied to the two keywords Pass 1's narrower scan had left out. **Confirmed by hand against the
migration-batch fixture** (`_MIGRATION_CYPHER` in `test_server.py`): its `UNWIND` list literal and
`CREATE` clause together contain none of the four bare words, so this closure does **not** regress
it — the review flagged this as something to confirm, not assume, and this plan states the
confirmation here rather than leaving it for S1 to discover.

**Scope of the fix — shape 1 only, not shapes 2-6.** Shapes 2 (curator-clear) and the three new
curator shapes (4-6, §3.1 above) are already anchored `^...$` against the fully
whitespace-collapsed statement — by construction, *nothing* may follow or precede their one
recognized skeleton, so they were never vulnerable to this class of chaining. Shape 3
(producer-write) is likewise immune by its own step-2e "nothing else follows" end-anchor (§3.1) —
a decoy chained *after* a producer-write is already rejected today because 2e requires end-of-string
right after the `KaizenEntry` map's closing `)`. The one place a decoy chained *before* the
recognized clause could hide is shape 1's own claim-scan, precisely because
`_kaizen_entry_create_map_spans()`'s whole point is to find its target clause *anywhere* in the
text (to support the migration-batch shape) — this is the one narrow spot needing this one narrow
fix.

**Residual scope, stated explicitly (not silently inherited).** This closes all three traced
attacks (A, B, and C) and any other statement shaped like "one self-attributed `CREATE` plus a bare
`MERGE`, `DELETE`, `SET`, or `REMOVE` elsewhere" — the full write-keyword set this file elsewhere
enumerates (`_WRITE_KEYWORD_RE`), minus the one keyword (`CREATE`) every legitimate shape-1
statement itself legitimately contains. It does not attempt to catch every conceivable
chained-clause shape (e.g. a second, syntactically-inert clause that opens with none of those four
words) — nothing in FR-1…FR-8 asks for one, and FR-8's own trust bar ("well-behaved callers can't
do this by accident, not hardened against a malicious one") still applies to whatever residual is
left; two independent review passes traced this design and found no further gap of this shape.
If a future write shape legitimately needs a bare `MERGE`/`DELETE`/`SET`/`REMOVE` alongside an
author-write claim in one statement, that is a new recognized shape to design explicitly, not a
reason to loosen this check.

### 3.2 Why not fold the producer-write recognizer into `_kaizen_entry_create_map_spans()`?

Considered and rejected: that function's whole contract is "find `author:` literals inside a
`CREATE (<var>:KaizenEntry {...})` clause," and the producer-write shape has no such clause (the
`KaizenEntry` node is the *second* pattern element, reached via `[:PRODUCED]`, exactly the
structural difference §2.3 already establishes as the second independent reason old-shape
recognition can't be stretched to cover it). Bolting a second anchor pattern onto an
already-subtle function would raise its cyclomatic complexity for a case it was never designed to
express, versus a small new sibling function that reuses only the one genuinely shared primitive
(`_scan_matched_brace`). The trade-off: one more top-level function vs. a harder-to-read existing
one — favors the new sibling.

### 3.3 `cobb`'s distillation rewrite (`skills/agent-maintenance/SKILL.md` §5)

Three concrete changes to the existing 4-step procedure, all mechanical once graph-dba's §4/§5
recipes are in hand:

- **Step 1's read** gains the traversal-based recipe (graph-dba §5) run *alongside* the existing
  `author`-filtered read (§7's no-retrofit consequence: historical entries have no edges to
  traverse, so they are silently absent from the new query, not an error) — both queries needed
  side by side for as long as any pre-M8 entry remains uncleared, per graph-dba's own §7 note.
- **Step 3's routing** gains a new branch: if the entry is really about a different agent than its
  producer, run the MENTIONS-write (graph-dba §3) tagging it — done by `cobb`, during
  distillation, per FR-4 (not by the producing agent).
- **Step 4's log-and-clear** replaces its single always-full-`DETACH DELETE` action with
  graph-dba's read-then-decide sequence: run §4.1's count query first, then either §4.2's one
  partial-edge delete (the producer's own pass always resolves `PRODUCED` regardless of remaining
  `MENTIONS` edges, per FR-6; a mentioned-agent's pass resolves only that one `MENTIONS` edge) or
  §4.3's unchanged full `DETACH DELETE` once nothing else remains. The append-before-clear
  ordering (history.md write confirmed durable *before* the graph mutation) is unchanged and still
  applies to every disposition, partial or full.

**Explicit ordering invariant (analyst review, Finding 2).** Within one distillation pass on one
entry: **the MENTIONS-tag (if any) for a given entry must be committed before that entry's §4.1
count-and-decide read runs in the same pass — the count must reflect any edge just added this
pass.** This is not merely the order the SKILL.md's steps already happen to run in (1→2→3→4) — it
is load-bearing: if the count ran before a same-pass MENTIONS tag landed, `otherRemaining` could
read `0` when it should read `>0`, and §4.3's full `DETACH DELETE` would fire before the just-added
`MENTIONS` edge was ever attached, silently discarding the very cross-agent link FR-3/FR-4 exist to
create — with no error, since a `DETACH DELETE` on an already-deleted-and-recreated timeline has
nothing to complain about. S4's rewrite of `SKILL.md` §5 must state this sentence explicitly, not
rely on a reader inferring it from step numbering — a future edit that reorders or parallelizes
per-entry work for speed would otherwise have nothing flagging the regression.

### 3.4 Doc updates — what changes where

- **`cypher-mcp/README.md`** "Writing through this tool": rewrite to describe all 6 shapes with
  worked examples (mirrors the existing two-shape presentation), state the mutation/back-reference
  distinction from §3.1 above is an implementation detail not worth exposing to a reader, and keep
  the existing "schema DDL is rejected unconditionally, no carve-out" paragraph (still true,
  restated for `Agent.agentId`'s index/constraint too).
- **`claude/README.md`**'s Kaizen section: the one-query recipe (lines 38-42) becomes the
  traversal-based union of "produced by or mentions agent X" (graph-dba §5) generalized to "every
  entry, any agent" (drop the `{agentId: ...}` filter from both halves), kept alongside the old
  plain `MATCH (e:KaizenEntry) ... {author: '<agent>'}` recipe for historical entries, with a
  one-line note on why both are needed (§7's no-retrofit consequence). Line 34's "attributed to
  itself... with its own `author` value" becomes "via a `PRODUCED` edge from its own `:Agent`
  node."
- **`claude/AGENTS.md`**'s opening paragraph and **root `AGENTS.md`**'s `claude/` bullet: both
  currently say "`author`-partitioned" with no further structure — add one clause describing the
  `:Agent`/`PRODUCED`/`MENTIONS` shape for entries created after M8, historical entries unchanged.
- **`docs/BACKLOG.md`** / **`docs/HISTORY.md`**: `teco`'s standing curator duty (§2.9) — a new M8
  row (`C-801` onward) added as units land, a closing `HISTORY.md` entry once the milestone is
  fully delivered. Not a step in the table below.

## 4. Step-by-step implementation

| Step | Owner | Files | Depends on | Done-condition |
|---|---|---|---|---|
| **S0** | `graph-dba` | none (live DDL only, via `redis-cli GRAPH.QUERY` against `kaizen_team`) | — | `CREATE INDEX FOR (a:Agent) ON (a.agentId)` then `GRAPH.CONSTRAINT CREATE kaizen_team UNIQUE NODE Agent PROPERTIES 1 agentId` (index-before-constraint; `NODE` keyword, not `LABEL`, per `falkordb-quirks.md`). Poll `CALL db.constraints()` until `status = OPERATIONAL`. Confirm before running: `CALL db.labels()` still shows no `Agent` (true as of 2026-08-22 — 20 `:KaizenEntry`, zero `:Agent`). **Not run through `mcp__cypher__query`** — schema DDL is unconditionally rejected there (live-confirmed during M7's `S0`). Hard predecessor of S1's live acceptance sub-step, and of S3/S4 going live (real agents relying on the new shape). |
| **S1** | `tdd-engineer` | `cypher-mcp/server.py`, `cypher-mcp/tests/test_server.py`, `cypher-mcp/README.md` | none for the code+offline-tests (can start immediately, parallel with S0) | §3.1's design implemented test-first: `_scan_matched_brace()` factored out (behavior-preserving refactor, full existing suite still green before anything new is added), `_producer_write_agent_id()`, the 3 new curator regexes, **§3.1a's `_has_foreign_trigger_outside_strings()` closure wired into shape 1's accept branch**, `authorize_write()`'s new control flow, `TOOL_DESCRIPTION`/`SERVER_INSTRUCTIONS`/module docstring/README all updated. New tests per §5 below (including the four Finding-1 regression cases, §5 items 18-19 and
22-23), all green; mutation-test at least one boundary in each of the new logic paths, §3.1a's closure included (§5); full existing 16-case suite unmodified and still green (regression). A small closing sub-step — **depends on S0 having landed** — runs one real producer-write + MENTIONS-write + both deletion shapes against the actual `kaizen_team` graph with a disposable `agentId`/`entryId`, verifying end to end and leaving zero residue (§5's "live functional check"); this is separate from the automated `pytest -m live` suite (§5), which runs against its own scratch graph and needs no dependency on S0, and separate from S6's later, `qa-engineer`-owned live re-confirmation of the Finding-1 closure specifically. |
| **S2** | `analyst` | (review only) | S1 | Independent diff-scoped code review (`docs/reviews/kaizen-agent-ontology.md`) — producer ≠ reviewer, this team's standing gate. Must explicitly confirm the mutation-testing step was actually performed, not merely asserted, and that the regression suite (existing 16 cases) is unmodified. |
| **S3** | `cobb` | all 13 `claude/<agent>/*.md` | `S2 approved` (recommended — drafting can start earlier; see §4 rationale) | Every agent's fenced Cypher block (§2.6) retargeted from the old `CREATE (k:KaizenEntry {..., author: '<agent>', ...})` shape to §3.1's producer-write shape (drop `author`; wrap in `MERGE (a:Agent {agentId:'<agent>'}) CREATE (a)-[:PRODUCED {sessionId:...}]->(k:KaizenEntry {...})`) — one substitution, applied 13 times, only the `agentId` literal (already per-file) varying. The "called as ..." line and all surrounding prose (verified non-uniform, §2.6) stay untouched. |
| **S4** | `cobb` | `skills/agent-maintenance/SKILL.md` §5 | `S2 approved` (recommended — drafting can start earlier; see §4 rationale) | §3.3's three changes: new read query (traversal + retained historical fallback), new MENTIONS-write routing branch, read-then-decide deletion logic (§4.1 count → §4.2 partial or §4.3 full) — **including the explicit MENTIONS-before-count ordering invariant** (§3.3, Finding 2) stated in the rewritten procedure text itself, not left implicit. |
| **S5** | `cobb` | `cypher-mcp/README.md` is already S1's file — this row is `claude/README.md`, `claude/AGENTS.md`, root `AGENTS.md` | S3, S4 (describes what actually shipped) | §3.4's doc updates for the three team-catalog docs. |
| **S6** | `qa-engineer` | (no tracked files — a live/manual pass; findings, if any, land as a written report at `docs/test-reports/kaizen-agent-ontology.md`) | S1 (deployed, container rebuilt), S3, S4 | Live dry-run pass against the deployed `cypher-mcp` container and the real `kaizen_team` graph, added per the analyst review's recommendation (Finding 1's "confirm dynamically that a static read cannot") and the stakeholder's acceptance of it: (a) re-run Attacks A, B, and C (§3.1a, including Attack C's `victim.author = '<other-agent>'` sub-case) against the live, rebuilt container, confirming §3.1a's closure — over the full widened `MERGE|DELETE|SET|REMOVE` keyword set — actually holds end to end, not just in the offline suite; (b) one real dry-run of `cobb`'s updated distillation procedure (tag a disposable entry with MENTIONS → run the count-and-decide read → partial-or-full delete) against disposable entries in the real graph, cleaning up fully afterward. Complements, not duplicates, S1's own scripted live acceptance sub-step (which covers the 4 new shapes' happy paths, not the adversarial chaining case or a real distillation dry-run). |
| *(ongoing, not a step)* | `teco` | `docs/BACKLOG.md`, `docs/HISTORY.md` | tracks S0-S6 | Standing documentation-curator duty (§2.9) — new M8 row/items as units land, closing `HISTORY.md` entry at milestone close. |

**Dependency graph, stated explicitly (not just file-conflict analysis):**

- **S0 and S1 are mutually independent and should be dispatched in parallel** — no file overlap,
  and S1's own offline test suite needs no live `kaizen_team` state. S1's automated `pytest -m
  live` tests run against a **fresh scratch graph** with its own freshly-provisioned DDL (mirrors
  the existing `live_graph` fixture pattern), so they too have no dependency on S0.
- **S1's one small manual/live acceptance sub-step** (against the *real* `kaizen_team` graph) is
  the one place S0 is a true predecessor of S1's own completion — sequence it after S0 lands, before
  declaring S1 done and ready for S2.
- **S2 depends on S1** (reviews its diff).
- **S3 and S4 are both `cobb`-owned but touch disjoint files** (13 agent prompts vs. one skill
  section) — dispatchable as one bundled `cobb` unit or two parallel ones, teco's call; either way
  **do not let any agent's retargeted prompt go live (i.e., actually be relied on for a real write)
  until S2 has approved and that code is deployed** (`cypher-mcp/build.sh` rebuild + container
  restart) **and S0's DDL is live** — drafting the text itself has no such dependency, but this
  plan recommends sequencing S3/S4 *after* S2's approval anyway, not just before go-live: a
  regex/parsing change is exactly the kind of thing that can need a small shape tweak during
  review (this codebase's own history shows two rounds of "Pass-1"/"Pass-2" review fixes on the
  *existing* author-write scanner), and drafting 13+1 files against a pre-review shape risks
  redoing them if S2 asks for a change. The cost of waiting one review cycle is small; the cost of
  13 files going stale is not.
- **S5 depends on S3+S4** (documents the shipped shape, not a drafted one) but has no file overlap
  with either — could be drafted in parallel if teco prefers speed over that small staleness risk,
  same trade-off as above.
- **S6 depends on S1 being deployed (not merely S2-approved) and on S3+S4 having landed** — it is a
  black-box pass against the *running* container and the *real* `kaizen_team` graph, so it needs
  the actual rebuilt image (`cypher-mcp/build.sh` + restart) live, and the distillation dry-run half
  needs `cobb`'s updated procedure (S4) to actually exercise. S6 has no file-conflict with anything
  else in this table (it produces a report, not a source/prompt edit) and is the last step before
  `teco`'s closing `HISTORY.md` entry.

## 5. Test strategy

**Unit (offline, `cypher-mcp/tests/test_server.py`, new cases added to the existing "8 — write
authorization" section, none of the existing 16 modified):**

1. Refactor regression: after factoring `_scan_matched_brace()` out, the full existing suite is
   green with **zero** test changes — proves the refactor is behavior-preserving before any new
   logic is added.
2. Producer-write, matching `agent` → authorized; write summary rendered (mirrors test 3's shape).
3. Producer-write, `agent` mismatched against the `MERGE`'s `agentId` → rejected, no partial write
   (mirrors test 4).
4. Producer-write with the optional `{sessionId: ...}` property map present → authorized.
5. Producer-write with the `sessionId` map omitted entirely (the template's "or omit this key"
   case) → still authorized.
6. Producer-write whose `KaizenEntry` map's free-text `fact`/`evidence` contains literal `{`/`}`
   characters → still authorized (mirrors tests 14/16's decoy-robustness intent, now for the new
   anchor).
7. Producer-write with a trailing extra clause after the `CREATE`'s closing `)` (e.g. an appended
   `SET`) → rejected — pins the "nothing else follows" structural check (§3.1 step 2e).
8. Producer-write with the `CREATE`'s referenced variable not matching the `MERGE`'s bound variable
   → rejected — pins the var-binding check (§3.1 step 2b).
9. Producer-write with `CREATE`/`MERGE` in reversed order → rejected (not the one recognized
   skeleton).
10. MENTIONS-write, `agent='cobb'` → authorized; mirrors test 5's shape but for the new regex.
11. MENTIONS-write, non-curator `agent` → rejected (mirrors test 6).
12. MENTIONS-write with the second `MERGE`'s referenced variables not matching the first `MATCH`/
    `MERGE`'s bound variables → rejected — pins the backreference check.
13. Producer-edge-resolve, `agent='cobb'` → authorized (`relationships_deleted=1` in the fake
    write-stats, mirroring test 5's counter-assertion style).
14. Producer-edge-resolve, non-curator → rejected.
15. Mention-edge-resolve, `agent='cobb'` → authorized.
16. Mention-edge-resolve, non-curator → rejected.
17. All 4 new shapes still correctly rejected without `agent` supplied (mirrors test 2's coverage,
    extended).
18. **Finding-1 closure, Attack A** (analyst review, `docs/reviews/kaizen-agent-ontology.md`,
    reproduced verbatim): the exact self-attributed-decoy-`CREATE`-plus-`DETACH DELETE` compound
    statement (§3.1a), called with `agent='analyst'` → rejected in full; no partial write (assert
    the fake client's `query()` — the real-write path — is never called at all, mirroring the
    existing `_writes_never_ran` helper).
19. **Finding-1 closure, Attack B** (same source, reproduced verbatim): the exact
    self-attributed-decoy-`CREATE`-plus-forged-`MERGE (:Agent {agentId:'cobb'})`-`PRODUCED`
    compound statement (§3.1a), called with `agent='analyst'` → rejected in full; no partial write.
20. A **legitimate** single-clause author-write and the **legitimate** migration-batch shape
    (`_MIGRATION_CYPHER`, already in the suite) both **still succeed** after `_has_foreign_trigger_
    outside_strings()` is added — pins §3.1a's own claim that the closure does not regress either
    accepted shape-1 case (this should already be covered by not modifying tests 3/8/10, but add
    one explicit assertion that a bare-word scan of `_MIGRATION_CYPHER`'s exact fixture text finds
    neither `MERGE` nor `DELETE`, as a documentation-grade regression pin independent of the
    end-to-end test).
21. A **decoy that itself quotes the words "MERGE" or "DELETE" inside a free-text field**
    (`evidence`/`context`) of an otherwise-legitimate single-clause author-write → still authorized
    (string-literal exclusion in `_has_foreign_trigger_outside_strings()` correctly ignores it) —
    extends the existing decoy-robustness style (tests 14/16) to the new check specifically.
22. **Finding-1 closure, Attack C, `SET`-chained variant** (analyst review Pass 2, reproduced
    verbatim, §3.1a): the self-attributed-decoy-`CREATE`-plus-`MATCH ... SET` compound statement,
    called with `agent='analyst'` → rejected in full; no partial write. **Explicitly include the
    `victim.author = '<other-agent>'` sub-case** (not just `victim.fact = 'tampered'`) — this is
    the sub-case that reopens tests 15/15b's exact concern in chained form, so it must be asserted
    on its own, not merely implied by the more general `SET` rejection.
23. **Finding-1 closure, `REMOVE`-chained variant** (symmetric to 22, per the review's own note that
    "a `REMOVE victim.author` variant is symmetric and equally unguarded"): the same decoy-`CREATE`
    chained with a `MATCH (victim:KaizenEntry {...}) REMOVE victim.author` clause, `agent='analyst'`
    → rejected in full; no partial write.

**Mutation-testing directive (explicit ask from the brief, not optional):** after the new tests are
green, deliberately break one regex boundary in the new logic — e.g. drop the `\1` backreference
from `_MENTIONS_WRITE_RE` (so a mismatched-variable MENTIONS-write would wrongly authorize), loosen
`_producer_write_agent_id`'s step-2e trailing-content check to accept anything, remove the new
`_has_foreign_trigger_outside_strings()` call from shape 1's accept branch entirely, **or narrow
`_FOREIGN_TRIGGER_RE` back down to `r"\b(?:MERGE|DELETE)\b"` (dropping `SET|REMOVE`)** — and confirm
at least one of tests 7/8/9/12/18/19/22/23 above actually fails for each mutation attempted (the
`SET|REMOVE`-narrowing mutation specifically must be caught by 22 and/or 23, not by 18/19, which
only exercise `MERGE`/`DELETE`). Revert immediately after confirming each one. Do not trust a green
run alone as proof the boundary checks are load-bearing; this mirrors how the *existing* shape's
tests (14, 15, 15b, 16 — see their own docstrings citing "Pass-1 review M1/M2," "Pass-2 review
M1-residual") were hardened through exactly this kind of adversarial re-check, and is now itself
the fourth round of that same discipline applied to this file (the third — this plan's own Finding
1 close — having itself needed a follow-up round once Pass 2 found it under-scoped).

**Live, automated (`pytest -m live`, own scratch graph, mirrors the existing `live_graph`
fixture):** extend the module-scoped fixture (or add a sibling one) to also provision
`Agent.agentId`'s index+constraint on the scratch graph directly via the raw client (never through
`mcp__cypher__query` — same DDL restriction), then run: a real producer-write, a real
MENTIONS-write, a real producer-edge-resolve, a real mention-edge-resolve, and confirm the two
*existing* live-suite shapes (author-write via the old shape, curator-clear) still pass unchanged.

**Live, manual, one-time (against the real `kaizen_team` graph, after S0 lands — not a permanent
pytest case):** deliberately *not* automated as a permanent `live` test, because `kaizen_team` is
shared team working memory with 20 real, un-distilled entries (confirmed live 2026-08-22) — an
automated test that creates/deletes `Agent`/`KaizenEntry` nodes there on every CI run would risk
interfering with that population, unlike the existing `live` suite's own dedicated scratch graph.
Instead: `tdd-engineer`, as the closing sub-step of S1 (§4), manually runs one producer-write with a
clearly-disposable `agentId` (e.g. `_cypher_mcp_selftest_<uuid4>`, not a real team member and not
in `CURATOR_AGENTS`), reads it back via the traversal query, runs one MENTIONS-write against it,
then fully resolves both edges (producer-edge-resolve, mention-edge-resolve) and confirms the node
is gone — leaving zero residue in the real graph. Also confirm, against the real graph, that a
mismatched-`agent` producer-write is still rejected. **This sub-step is deliberately narrower than
S6** (§4): it exercises the 4 new shapes' happy paths and one mismatch case against the real graph,
not the adversarial cross-clause chaining of §3.1a (Attacks A/B) or a real end-to-end distillation
dry-run — those two are S6's own, later, `qa-engineer`-owned job, run against the deployed
container after S3/S4 land.

**Edge cases the above deliberately covers:** the optional `sessionId` map (present/absent), the
"nothing else follows" structural boundary, variable-binding consistency (both new families),
free-text braces inside the new anchor, non-curator rejection on all 3 new curator shapes, and
non-regression of both pre-existing shapes.

## 6. Risks & open questions

- **Regex/parsing correctness is the highest-risk part of this delivery**, by this plan's own
  design choice to keep the mechanism a text scanner rather than a real parser (matching FR-8's
  stated trust bar and graph-dba's recommendation) — the existing code's own history (two rounds
  of post-hoc "Pass-1"/"Pass-2" review fixes on the *simpler* author-write shape) is direct
  evidence this class of change is easy to get subtly wrong, and this plan's own gate process just
  produced two further rounds of exactly that on this plan's own new logic: `analyst`'s Pass 1
  Finding 1 (the `MERGE`/`DELETE` chaining gap) and Pass 2's follow-up (the same gap was itself
  under-scoped, missing `SET`/`REMOVE`) — both now closed in §3.1a. Mitigation: §5's
  mutation-testing directive, S2's independent review gate (now spanning two passes), and S6's live
  black-box re-confirmation are all non-negotiable steps in this plan, not optional polish — the
  pattern across all four rounds (two on the shipped code, two on this plan's own design) is that a
  static read alone, however careful, has repeatedly needed a second pass to catch a chaining-shaped
  gap a concrete adversarial reproduction then confirmed — so the extra gate earns its cost, and a
  third review pass should not be assumed unnecessary just because two have now run clean on this
  exact class of issue.
- **Residual scope of §3.1a's closure, stated explicitly, not silently inherited** — it closes
  every statement shaped like "one self-attributed `CREATE` plus a bare `MERGE`, `DELETE`, `SET`,
  or `REMOVE` elsewhere in the same text" (the full write-keyword set minus `CREATE` itself); it is
  not a general guarantee that no other chaining shape could ever exist against shapes 1-2's design
  (e.g. a decoy that contains none of those four words yet still smuggles something — no such shape
  is reachable under the current 6-shape allowlist and none was found across two independent review
  passes, but the point is narrow-fix, not proof of completeness). Revisit if a 7th write shape is
  ever added that shares shape 1's whole-text-scan style, or if a new Cypher write keyword outside
  this set is ever introduced.
- **A `qa-engineer` black-box acceptance pass is now part of this plan (S6)** — resolved from an
  open question in the prior version: the analyst review recommended it specifically because of
  Finding 1 (a static review can trace an adversarial scenario but not confirm dynamically that the
  deployed, rebuilt container actually behaves as designed), and the stakeholder accepted that
  recommendation. Scoped narrower than M5's full acceptance pass, per this plan's own scoping
  argument that M8 builds on an already-accepted foundation.
- **`_producer_write_agent_id`'s "exactly one MERGE + exactly one CREATE, nothing else"
  requirement (§3.1 step 2e) is intentionally strict** — a well-behaved caller following the
  recipe verbatim (as all 13 retargeted prompts will) always produces exactly this shape, but any
  future variation (e.g. an agent that wants to batch-create several entries in one call, echoing
  the old §3.4 migration-batch shape for author-write) would need a new recognized shape, not a
  loosening of this one. Flagging as a known future-extension seam, not a defect.
- **Docs drift between S3/S4 (text) and S1 (code) if dispatched out of the recommended order** —
  §4's dependency-graph section states the recommendation (wait for S2) and the trade-off
  explicitly; this is a scheduling risk `teco` can choose to accept, not a correctness one.
- **No migration/rollback concern for existing data** — this feature adds new node/relationship
  types and one new authorization path; it does not touch or move any existing `:KaizenEntry`
  node or property, and `authorize_write()`'s existing two shapes are additions-to, not
  replacements-of. Rolling back is: revert `cypher-mcp/server.py`/tests/README (S1), revert the 13
  prompts (S3) and the skill doc (S4), leave S0's DDL in place (harmless — an unused index/
  constraint on a still-empty label costs nothing and doesn't need to be un-provisioned).
