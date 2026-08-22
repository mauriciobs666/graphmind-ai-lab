# Review — `docs/manuals/graph-ontology.md`

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (—)

## Scope & verdict

Static factual/architectural review of `docs/manuals/graph-ontology.md` (authored by `tico`),
covering all four graph families it documents: Code Property Graphs (`cpg_<component>`),
`kaizen_team`, `reference`, and `ws:<workspaceId>`. Reviewed against:

- `skills/joern-cpg/references/cpg-model.md` and live queries against `cpg_falkorchat` /
  `cpg_salesperson` (`CALL db.labels()`, `CALL db.relationshipTypes()`, `properties(n)` sampling
  across most listed node labels, plus direct counts for the CALL-node-vs-edge and sparse-caller
  claims).
- `docs/requirements/kaizen-agent-ontology.md` and live queries against `kaizen_team` and
  `kaizen_analyst`.
- `falkor-chat/docs/DESIGN.md` (§3, §5–§9), `falkor-chat/docs/QUERIES.md` (§6, §10), and live
  queries against `reference` and `ws:acme`, plus a source-code check of
  `falkor-chat/server/falkorchat/{repository,embedding}.py` and `scripts/bootstrap_schema.sh` for
  the GraphRAG-corpus "implemented write paths" claim specifically.
- All 6 Mermaid diagrams, rendered standalone with `@mermaid-js/mermaid-cli` (all 6 produced valid
  SVG output — syntax confirmed, not just eyeballed).
- Every "Try it" Cypher snippet spot-checked: the ones tested (`reference` entry-step query,
  CPG `post_message` lookup) both ran and returned sensible rows.

This did not review the manual's behavioral walkthroughs against a running app — that's
`qa-engineer`'s lane, not mine, per the brief.

**Verdict: approve with suggestions.** The manual is unusually well-grounded — node/relationship
label counts, casing conventions, directions, and almost every documented gotcha matched the live
graphs and source docs exactly, including several places where the manual gets subtle details
right that would be easy to get wrong (CALL node-vs-edge, EMITTED direction, the `kaizen_analyst`
leftover, the exact set of 10 currently-loaded graph keys). One finding (below) is a real factual
overstatement worth fixing before this is treated as fully authoritative, plus two minor
clarity nits.

**CPG:** considered, not relevant — this is a documentation review of a manual describing graph
schemas (including CPG schemas), not a code change to a CPG-producing or CPG-consuming component;
the two live CPGs (`cpg_falkorchat`, `cpg_salesperson`) were queried directly via `mcp__cypher__query`
as the grounding source for §1, which is the tool this review needed, not a loaded Joern CPG of
some other codebase under review.

## Findings

### Major

**M1 — §4b overstates the GraphRAG corpus as having "implemented write paths"; the codebase says
otherwise for `Chunk`/`Document`/`HAS_CHUNK`/`DERIVED_FROM`/`ABOUT`.**

`docs/manuals/graph-ontology.md:376-381` states:

> "the labels/edges above are real (implemented write paths, indexed, live-verified) but in every
> workspace currently loaded, `Document`/`Chunk`/`Entity` sit at zero nodes."

I verified the "indexed" half is true (`scripts/bootstrap_schema.sh:97-101,184-191,239-240`
creates indexes/constraints/vector-index for `Document`, `Chunk`, `Entity` unconditionally), and
"currently zero nodes" is true (confirmed live on `ws:acme`: `Document`/`Chunk`/`Entity`/
`WorkspaceConfig` all `count(n) = 0`).

But "implemented write paths" is not true for `Document`, `Chunk`, or the edges `HAS_CHUNK` /
`DERIVED_FROM` / `ABOUT`. A repo-wide grep of `falkor-chat/server/falkorchat/*.py` (excluding
`.venv`) for `Chunk`, `Document`, `HAS_CHUNK`, `DERIVED_FROM` turns up **zero** write-path
references — the only hits at all are a comment in `config.py` and, decisively,
`server/falkorchat/embedding.py:135-137`'s own docstring:

> "`Chunk` is never written by any code path in this codebase today (`-graph.md` §3.3/§3.4, plan
> §4.5) and is deliberately never consulted here"

The only one of the four GraphRAG relationship types with *any* code presence is `MENTIONS`
(`repository.py:779`, an `OPTIONAL MATCH` in `hybrid_search`'s read path) — and even that is
explicitly documented as dormant/no-op (`repository.py:761-763`, `DESIGN.md:793` "dormant Entity
no-op"), which the manual correctly captures elsewhere ("Entity extraction is a documented no-op
today"). Live confirmation: `CALL db.relationshipTypes()` on `ws:acme` returns **20** types, and
none of `HAS_CHUNK`, `DERIVED_FROM`, `ABOUT`, `MENTIONS` is among them (FalkorDB only registers a
relationship type once at least one edge of that type has ever been created — unlike node labels,
which register at index-creation time regardless of data, which is why `Document`/`Chunk`/`Entity`
*do* show up in `db.labels()` at zero nodes while their edges don't show up in
`db.relationshipTypes()` at all). A reader who runs the manual's own suggested diagnostic
(`CALL db.relationshipTypes()`, from the Overview's "three queries") on `ws:acme` will see this
asymmetry directly and it will read as a contradiction of what §4b told them to expect.

**Why it matters:** this manual's stated audience is a developer about to write Cypher against
these graphs. "Implemented write paths… live-verified" reads as "the ingestion pipeline exists,
it's just never been exercised on this data" — when the reality is closer to "the schema is
provisioned but no code in this repo populates `Document`/`Chunk`/`HAS_CHUNK`/`DERIVED_FROM`/
`ABOUT` at all." A reader relying on the manual to scope a GraphRAG-corpus feature would
under-estimate the work by an entire ingestion/chunking layer.

**Suggested fix:** narrow the claim to what's actually true — the *schema* (indexes/constraints/
vector index) is provisioned and live-verified; only `MENTIONS` has any code-path presence at all
(a dormant read-side no-op); `Document`/`Chunk`/`HAS_CHUNK`/`DERIVED_FROM`/`ABOUT` have no write
path in the codebase today, full stop. Something like: *"The labels/edges above have schema
provisioned (indexed, live-verified) but, except for `MENTIONS`'s dormant read-side no-op, no
write path exists in this codebase yet — building a document-ingestion pipeline that populates
`Document`/`Chunk`/`HAS_CHUNK`/`DERIVED_FROM`/`ABOUT` is still greenfield work."*

### Minor

**m1 — §4c's `WorkspaceConfig` line reads as "this exists" without the caveat given two
paragraphs later.** Line 422-424: *"Also present, singleton config: `WorkspaceConfig` — one
`{workspaceConfigId: 'default'}` node per workspace…"* — "Also present" implies live data. Live
`ws:acme` has zero `WorkspaceConfig` nodes (confirmed: `MATCH (n:WorkspaceConfig) RETURN
properties(n), count(n)` → no rows). The gotchas section (line 441-444) *does* correctly list
`WorkspaceConfig` among the labels that can have zero live nodes — but a reader who stops at the
dedicated `WorkspaceConfig` callout, without continuing to the gotchas at the end of §4, gets the
opposite impression. Suggested fix: change "Also present" to something like "Schema also defines
a singleton config label" or add "(zero nodes in every workspace currently loaded, same caveat as
below)" inline, so the caveat isn't dependent on the reader reaching a later paragraph.

**m2 — Overview's naming-convention table has a dangling sentence.** Line 56: *"Relationship types
| `SCREAMING_SNAKE` (`AST`, `REACHING_DEF`) | `UPPER_SNAKE` (`POSTED_BY`, `HAS_STEP`) — looks
similar but..."* — the "but..." trails off with no completion in that cell or the next. Worth
noting: `SCREAMING_SNAKE` and `UPPER_SNAKE` are, in fact, the *same* casing convention (both
families use identical ALL_CAPS_WITH_UNDERSCORES for relationship types) — the real divergence is
one row down, in property-key casing (`UPPER_CASE` vs `camelCase`). As written, "looks similar
but..." primes the reader to expect a difference in relationship-type casing specifically, which
doesn't exist; the actual point (the property-key row) is disconnected from the sentence that
promises it. Suggested fix: either finish the sentence ("...it's the *property keys*, below, where
they actually diverge") or drop the "but..." and let the property-keys row make its own point.

## What's solid

- Every node-label and relationship-type count checked out **exactly** against live data: 21
  labels / 19 relationship types on `cpg_falkorchat` (manual lists 21/19, itemized), `cpg_salesperson`
  correctly identified as the same 21 minus `CpgBuildInfo` (confirmed: 20 live labels, no
  `CpgBuildInfo`), `reference`'s 3 labels / 3 relationship types matched exactly, `ws:acme`'s 15
  labels matched exactly (6 chat + 3 GraphRAG + 5 workflow-run + `WorkspaceConfig`), and the
  10-graph "loaded right now" list matched the live server's own reported graph list verbatim
  (including the `kaizen_analyst` and absent-`identity` claims).
- The CPG gotchas (CALL node-vs-edge, sparse caller resolution, `METHOD_FULL_NAME`
  unreliability, `FILENAME` reliability) are not just consistent with
  `skills/joern-cpg/references/cpg-model.md` — they're close paraphrases of its "Consumer-query
  facts" section, and I independently re-confirmed the two sharpest claims live: zero
  `(:METHOD)-[:CALL]->(:METHOD)` edges exist, and `(:CALL)-[:CALL]->(:METHOD)` covers only 2,853
  of 45,745 `CONTAINS`-reachable call sites (~6%) — "sparse" is accurate, not exaggerated.
- §2's kaizen framing is exactly right on the one hard requirement the brief flagged: it correctly
  presents `:Agent`/`PRODUCED`/`MENTIONS` as designed-not-built, cites
  `docs/requirements/kaizen-agent-ontology.md` whose header status is verbatim "Ready for design,"
  and never writes a sentence that could be read as "this exists today." `kaizen_team`'s zero
  relationship types and `kaizen_analyst`'s fully-empty state were both reproduced live. The
  `KaizenEntry` property list matches all 15 live-sampled entries' key sets exactly (`entryId`,
  `date`, `fact`, `evidence`, `context`, `suggestedHome`, `author`, `createdAt`, optional
  `sessionId`).
- Every relationship *direction* I spot-checked against live data matched the manual exactly:
  `EMITTED` answer→seed (confirmed `assistant`-role source, `user`-role target on live edges),
  `MENTIONS_MEMBER` Message→participant (confirmed `Agent`/`User` targets), `CALL` site→callee,
  `IS_CALL_FOR_IMPORT` call-site→import. The "four distinct edge types, don't conflate" claim
  (`EMITTED`/`PRODUCED`/`MENTIONS`/`MENTIONS_MEMBER`) matches `DESIGN.md`'s own explicit D2
  decision register and QUERIES.md §10 almost verbatim.
- The casing-convention claim (CPG `UPPER_CASE` vs falkor-chat `camelCase`) holds up under every
  live sample pulled during this review (`m.NAME`/`m.FULL_NAME`/`m.CODE` vs `msgId`/`createdAt`/
  `embedding`).
- All 6 Mermaid diagrams are syntactically valid (rendered clean with `mermaid-cli`, no errors),
  and each one's edges/labels match its accompanying prose table — no diagram invents a label or
  relationship the table doesn't also list, and no diagram omits a directional detail the table
  asserts.
- The manual stays close to the right altitude for its stated audience throughout — it doesn't
  wander into request-handling internals, algorithm-level implementation, or code layout. The one
  borderline case (the Overview's "who writes it / who reads it" columns naming specific agents)
  is arguably in-scope: it tells a Cypher-writing agent who else might be touching the same graph
  concurrently, which is closer to "know before you query" than internal architecture — not flagged
  as a finding.

## Open questions

- None that block approval. M1 is the one item I'd want folded in before treating §4b as fully
  reliable for someone scoping GraphRAG-corpus work; m1/m2 are low-stakes wording tightenings.
