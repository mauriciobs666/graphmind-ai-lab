# Agent knowledge-base strategy — implementation plan

> **Status:** active — Stage 0 closed; Track 1 (FR-8/FR-9, raw-capture migration) and Track 2
> (FR-2–FR-7, distilled-knowledge ingestion) below are the current design, ready for
> implementation dispatch · **Owner:** `architect` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)
> · **Reviews:** `claude/docs/reviews/agent-knowledge-base-strategy.md` (Pass 2, approve with
> suggestions — see the closing section for the one reasoning-only correction it required)

**Revision note (2026-09-17).** The 2026-09-10 block
(`falkor-chat/docs/requirements/document-ingestion2.md`) is lifted — that feature shipped and
archived (`falkor-chat/docs/plans/document-ingestion2-coordination.md`, closed 2026-09-13, commit
`8906878`). The requirements doc
(`claude/docs/requirements/agent-knowledge-base-strategy.md`, decision log, 2026-09-17 entries)
re-ran the Option A/Option B substrate choice on the merits — not an automatic flip — and settled
it as **Option B (falkor-chat ingestion), chosen anyway**, accepting two of §1's three original
objections as costs to resolve within this feature's own design rather than grounds to fall back
to Option A (see the revised §1 below). The requirements doc also **expanded scope**: raw
kaizen-capture migration into a new, dedicated falkor-chat workspace (FR-8/FR-9) is now this
effort's first deliverable, sequenced *before* the distilled-knowledge-base ingestion work this
plan originally covered alone (FR-2–FR-7) — both tracks converge on the same new workspace.
**Everything below except §3's Stage 0 subsection is a substantive revision** of the pre-2026-09-17
draft — §1, §2, and §4 onward no longer describe the `kaizen_team`-substrate (Option A) design;
they describe Option B's design against the actual shipped falkor-chat capability, verified by
reading `falkor-chat/server/falkorchat/{services,repository,mcp,background,chunking,config}.py`
directly (the CPG is stale for this purpose — see the CPG line below).

**Document-convention call (`AGENTS.md` collision rule 5).** This document has one small,
independently-scoped part that has been executed against and gated: §3 Stage 0 (K-030's four
interim flat-file KBs), shipped as commit `6d834f0`, reviewed
(`claude/docs/reviews/agent-knowledge-base-strategy-impl.md`, approve with suggestions),
coordination closed (`claude/docs/plans/agent-knowledge-base-strategy-coordination.md`, archived).
Everything else — §1 onward, the actual bulk of the document — was never authorized to build: the
2026-09-10 block notice said so explicitly ("§2 onward stays... on hold, not re-cast... there is no
falkor-chat update/delete capability to design against yet"), and the closing "Ready to implement"
section named only Stage 0 as actionable. Rule 5's test is "has the earlier document been approved,
gated, or executed against?" — read at the level the rule is actually protecting (a design a
reviewer/stakeholder has signed off on and work has started against), the substrate-design sections
answer **no**: nobody has built against them, reviewed them as shipped, or gated a delivery on
their content — the only content anyone has acted on is Stage 0, preserved untouched below. Forking
to `agent-knowledge-base-strategy2.md` here would split one still-single design thread (the
substrate work) across two files for no reader's benefit, and would orphan §1–§8's extensive,
still-relevant investigation (the source findings, the migration-effort analysis) behind a
"superseded" flag it doesn't deserve — none of it was wrong; most of it (the ML note's four
recommendations, the claim-level-granularity finding, the file-counterpart resolution) carries
forward unchanged into this revision. **Revising in place**, per rule 5's "no → revise it in place"
branch, with a dated revision note (this one) rather than a stacked narrative, per this repo's own
"an open item is rewritten, not appended to" discipline. Stage 0's own subsection is carried forward
byte-identical in substance (only its position in the stage table changes, from a lone row to the
header of "Track 0 — already closed").

Implements `claude/docs/requirements/agent-knowledge-base-strategy.md` (Status: Ready for design,
decision log through 2026-09-17) FR-1 through FR-9. Covers: which substrate Option B's design lands
on for both tracks, the two concrete pieces of new capability this needs (one small falkor-chat
extension, one workspace bootstrap), the content model for both raw capture and distilled knowledge
inside falkor-chat's own `Document`/`Chunk` shape, the retrieval convention (falkor-chat's own
shipped `search_documents`, no new MCP server), sequencing (FR-9: Track 1 first), and every Open
question the requirements doc left for `architect`/`graph-dba` (structural shape, write-access
mechanism — Open questions #3/#4 in the requirements doc's current numbering).

**CPG:** considered, not relevant — `cpg_falkorchat` exists but is stale for this task (built
2026-09-12, source commit `ca25a20`, predating `document-ingestion2`'s Stage F close, commit
`8906878`, by several stages): it does not see that feature's update/delete/list methods, the
`INGESTED_BY` actor-resolution shape, or any of the other current behavior this revision depends on
(per this coordination's own brief). Every falkor-chat-side finding below (chunking behavior, the
single-hardcoded-actor seam, `INGESTED_BY`'s `User`/`Agent` coalesce, the extraction-scheduling
call sites) is from a direct, current read of
`falkor-chat/server/falkorchat/{services,repository,mcp,background,chunking,config}.py` and
`falkor-chat/docs/{DESIGN,SERVER}.md`, not the CPG. `cypher-mcp` (no longer touched by this
revision — see §1) still has no loaded CPG, unchanged from before.

**ML method note:** `data-scientist`'s `claude/docs/plans/agent-knowledge-base-strategy-ml.md` is
**not revised by this document** — its four recommendations are stated as **substrate-agnostic by
design** in the note's own closing section, and nothing in this revision's findings contradicts
that claim. What changes is *how* each recommendation is realized under Option B, not the
recommendation itself — mapped out where each is used below (§4.4) and summarized once, for
`data-scientist`, in the closing section's item (a). Quoted once here, in the note's own words, and
cited (not re-argued) at each point below: **keep Qwen3-Embedding-0.6B with the asymmetric
query-instruction prefix on the query side only; chunk at one distinct claim per node, not one
Markdown heading per node; retrieve with top-K=5 and a golden-set-calibrated score floor, with a
`familyId` sibling link as the one earned traversal; gate AC-2 on a ~40-pair, independently-authored
golden set scored as recall@5 with a Wilson-interval CI.**

---

## 1. The substrate — Option B, chosen with two named costs accepted, not resolved to zero

The requirements doc's decision log has settled this, not reopened it for this revision: **Option
B (falkor-chat ingestion)**, not `kaizen_team` (Option A). This plan is written against that
choice; §1 here documents what it concretely commits to and how this design carries the two costs
the stakeholder is knowingly accepting, rather than re-litigating the choice itself.

### Recap: the original three objections, and which are actually closed

The pre-2026-09-17 draft of this plan rejected Option B on three concrete grounds. The requirements
doc's decision log (2026-09-17 entries) checked each one directly against `document-ingestion2`'s
actual shipped scope (its own FR-1 through FR-8) before the stakeholder re-chose:

1. **No update/delete/list — closed.** `document-ingestion2` shipped exactly this:
   `delete_document`/`list_documents`/`get_document_history`/the `SUPERSEDES` versioning chain, all
   live-verified (AC-1 through AC-8, `falkor-chat/docs/plans/document-ingestion2-coordination.md`).
   Confirmed directly against the current source (`repository.py`/`services.py`/`mcp.py`) rather
   than taken on the coordination ledger's word alone, since the CPG is too stale to check it.
2. **Chunking-granularity mismatch — NOT closed.** `document-ingestion2`'s FR-1 through FR-8 are
   entirely about update/delete/versioning/audit; it never touched chunking. Confirmed directly:
   `services.ingest_document` still calls `chunking.split_into_chunks(text)` with no
   size/strategy override exposed to any caller (`services.py:1180`), and `chunking.py`'s boundary
   rule (paragraph → sentence → hard-cut, no heading/claim awareness) is unchanged. This cost is
   **real and unresolved by falkor-chat's own code** — this plan's design (§2) mitigates it at the
   calling-convention level instead (see below), because there is no server-side knob to mitigate
   it with.
3. **Tenancy/side-effect mismatch — half closed.** The "no natural workspace" half is closed by
   FR-8 itself: this feature now creates one (`ws:agent-team`, §4.2/§4.3). The "automatic LLM
   entity/relationship extraction" half is **not closed**. Confirmed directly, precisely stated
   (corrected 2026-09-17 per `analyst`'s review, minor finding 1 — the conclusion is unchanged,
   the mechanism description was slightly overstated): `mcp.ingest_document` gates the
   embed/extract scheduling call behind `if _embed_worker is not None or _ingestion_pipeline is
   not None:` (`mcp.py:309`) — conditioned on deployment-level wiring, not literally unconditional
   at the tool level. But `app._build_default_app` (`app.py:595-657`) always constructs and wires
   `ingestion_pipeline` whenever `FALKORCHAT_ENABLE_AGENT` is on, which `start_server.sh` defaults
   to `1` — and the dedicated `ws:agent-team` process this plan now requires (§4.3) must run with
   `FALKORCHAT_ENABLE_AGENT=1` anyway, since `search_documents` needs the embedder wired. So under
   the actual deployment this plan runs against, extraction fires on every `ingest_document` call
   as a practical matter — there is still no parameter anywhere on the MCP/REST/Services surface to
   suppress it for one call while keeping embedding on.

### What "Option B anyway" concretely costs, and what this plan does about it

The stakeholder's own reasoning (decision log, 2026-09-17): the one-substrate principle governs
even with problems 2 and 3 only partly resolved, and the remaining costs are to be **accepted or
resolved within this feature's own design**, not a reason to fall back to Option A. This plan takes
both paths, one per cost:

- **Chunking (problem 2) — mitigated, not resolved, by a calling-convention decision (§2/§4.2):**
  ingest **one `Document` per atomic claim/entry**, never one `Document` per file. Since falkor-chat
  has no chunking-strategy knob to change, this plan moves the granularity decision to the
  *caller* (`cobb`, at write time) instead of the chunker — the generic splitter then only ever
  operates within one already-atomic unit's own text, so its worst case is splitting one claim
  across two chunks of the same `documentId` (a mechanical artifact, recoverable via
  `get_document`), never the cross-claim topic dilution the original objection was about. Not a
  complete fix (an unusually long worked example can still land in a second chunk); named as an
  accepted residual in §8, not claimed as solved outright.
- **Extraction noise (problem 3) — accepted as inert for this feature's own retrieval path.**
  `services.search_documents` is confirmed ANN-only over `Chunk.embedding`
  (`repository.search_chunks`, no `Entity`/`RELATES_TO` traversal at all) — so the extra `Entity`
  nodes/edges this pipeline produces from kaizen/KB prose sit in the graph unused by anything this
  feature queries. It costs RAM and adds noise a future feature might trip over, but it does not
  degrade this feature's own retrieval quality or correctness. Accepted as-is, named in §8, not
  engineered around.

### Two costs neither the requirements doc nor the original §1 named — surfaced here, closed in §4.1 and §4.3

Reading `config.py`'s auth/tenancy seam directly (`falkor-chat/docs/SERVER.md` §1.3) surfaced two
genuinely new costs the requirements doc's Open Questions #3/#4 were flagging space for but that
no earlier draft of this plan had concretely checked — **`get_context()` pins *both halves* of
`CallContext` process-wide, not only `actor`**: `CallContext(ws=WS_ID, actor=USER_ID)`, both
env-fixed at import; "MCP ignores any client-supplied `from`."

- **The `actor` half** — every `ingest_document` call from *any* agent would, unmitigated, write
  `INGESTED_BY` against the same single configured actor, silently discarding the real,
  today-load-bearing per-producer attribution `kaizen_team`'s `(:Agent)-[:PRODUCED]->(:KaizenEntry)`
  edges carry (`claude/AGENTS.md` — `cobb`'s distillation review, per-agent knowledge routing, and
  FR-8's own "team-wide, not only K-030's four" framing all depend on knowing *which* agent wrote
  what). Not a hypothetical: `repository.create_document`'s `INGESTED_BY` write already resolves
  its target against **either** `(u:User {userId: $ingestedBy})` **or** `(a:Agent {agentId:
  $ingestedBy})` — the schema already anticipates agent-typed ingestors — but the *value* passed in
  is always the one pinned `ctx.actor` string, never a per-call identity. §4.1 closes this with a
  small, additive extension, following a precedent already set in this same codebase (the
  storefront feature's own second, per-caller auth path bolted alongside the process-constant seam,
  `SERVER.md` §1.3, rather than a full auth rework).
- **The `ws` half — found independently by `analyst`'s first review of this plan (blocker
  finding), not by this original drafting pass, and closed in §4.3.** `ctx.ws` is pinned by the
  *exact same mechanism* as `ctx.actor`, confirmed across every front door (MCP, REST, and the one
  precedent for varying `CallContext` per caller, `Storefront.context_for`, which varies only
  `actor`, `ws` stays `self._ws`, the whole `Storefront` instance's fixed workspace). So "wire every
  agent to falkor-chat's MCP server" does **not**, by itself, get any agent's writes into
  `ws:agent-team` — every call through *whichever* falkor-chat server process an agent's `.mcp.json`
  entry happens to point at lands in **that process's** pinned workspace, not `ws:agent-team`,
  regardless of §4.2's bootstrap having created the `ws:agent-team` graph correctly. This corrects
  an inaccurate framing the earlier draft of this section carried ("every Claude Code agent in this
  repo would share the *same* falkor-chat MCP endpoint," used to motivate the `actor`-half finding
  above): that sentence is only true **once** a dedicated `ws:agent-team` endpoint exists (§4.3) —
  today it conflates "falkor-chat's machinery" (shared correctly, by design) with "falkor-chat's
  *running instance*" (not shared, and must not be, once §4.3 is designed correctly). "Reuse the
  actual machinery literally" (§4.4) therefore costs a second standing server process, not zero
  extra infrastructure — named explicitly rather than left implicit.

---

## 2. Content model — `Document`/`Chunk` inside `ws:agent-team`, resolves Open Question #3

Both tracks reuse falkor-chat's existing `Document`/`Chunk`/`INGESTED_BY` shape as-is — **no new
label, no new edge type**, beyond §4.1's small attribution extension. One `ingest_document` call
per unit of content, landing in one shared workspace, `ws:agent-team` (§4.2):

- **Track 1 (raw capture, FR-8).** One call per today's `:KaizenEntry`-shaped note. `title` = the
  entry's `fact` (one line — mirrors kaizen_team's own `fact` field, and keeps
  `Document.title`'s existing fulltext index useful when skimming `list_documents`). `text` = a
  fixed, labeled rendering:
  ```
  Fact: <fact>
  Evidence: <evidence>
  Context: <context>
  Suggested home: <suggestedHome>
  ```
  — one paragraph per labeled field, chosen deliberately to line up with the chunker's own
  paragraph-boundary rule, so a short entry stays one chunk and a longer one splits along field
  boundaries rather than mid-sentence. `source_format="text"`, `produced_by=<writing agent's own
  agentId>` (§4.1).
- **Track 2 (distilled knowledge, FR-2).** One call per **claim** — per the ML note's
  Recommendation 2 (cited, not re-argued: the common case is one call per existing `##`-headed
  technique; a heading bundling several independently-attributed sub-claims becomes several calls,
  one per sub-claim, each self-contained down to its own worked example). `title` = the claim's
  short name. `text` = the full distilled prose including its own worked/counter-example.
  `produced_by='cobb'` (the standing curator).

**Why one call per claim/entry closes the chunking-granularity cost (§1) without any falkor-chat
change:** it moves the granularity decision to the caller instead of the chunker (§1's mitigation,
restated at the content-model level it actually lands on).

**`familyId` sibling linkage (ML note Recommendation 3) is degraded, not carried forward as a graph
edge.** The Option A draft gave siblings split from one heading a shared `familyId` property and a
Cypher-side sibling pull; `Document` has no such property, and `search_documents` does no traversal
at all (confirmed, §4.4). Mitigation: a shared **title-prefix convention**
(`"<family-slug> — <claim-title>"`), manually recoverable via `list_documents`/`get_document` but
with no automatic "pull my siblings" step in retrieval. Named as an accepted loss in §8 — each
claim-document is self-contained by construction (the ML note's own actionability floor), so the
loss is convenience, not correctness.

**No new falkor-chat index beyond §4.1's.** `Document`/`Chunk`/`INGESTED_BY`/`SUPERSEDES`/
`HAS_CHUNK`, the vector index on `Chunk.embedding`, and the `Document.title` fulltext index all
already exist (`document-ingestion2`, shipped). `ws:agent-team`, as a fresh workspace, still needs
its own `bootstrap_schema.sh` run (every per-workspace index/constraint is graph-scoped, not
global, per `falkor-chat/AGENTS.md`'s "one graph per workspace") — a mechanical bootstrap step
(§4.2), not a new schema design.

---

## 3. Sequencing — Track 0 (closed) → Track 1 (FR-8/FR-9) → Track 2 (FR-2–FR-7)

FR-9 fixes the order across tracks: raw-capture migration (FR-8) completes before distilled-
knowledge ingestion (FR-2 through FR-7) begins. Track 0 (K-030's interim relief) is independent of
both and already closed.

### Track 0 — closed, unchanged, not reopened

| Stage | What | Owner | Status |
|---|---|---|---|
| **0** | K-030's four interim flat-file KBs (`teco`, `architect`, `data-scientist`, `tdd-engineer`), authored per the existing pattern | `cobb` | **Done** — commit `6d834f0`, `analyst`-reviewed (approve with suggestions), coordination closed (`claude/docs/plans/agent-knowledge-base-strategy-coordination.md`) |

**FR-7 authoring constraint, historical record (already satisfied):** the four interim files were
authored with the same convention the existing six use — one `##`-heading-delimited technique per
section, one claim per section where possible. `analyst`'s Stage 0 review confirmed zero content
loss; a handful of sections in `teco/coordination-techniques.md` still bundle 2–3 separable claims,
deferred to Stage 6 below (FR-7's "where possible" bar, not a defect).

### Track 1 — raw-capture migration (FR-8/FR-9), first

| Stage | What | Owner | Blocks on |
|---|---|---|---|
| **1** | Small, additive falkor-chat extension: `produced_by` on `ingest_document`/`ingest_documents` (§4.1) | `graph-dba` (design note) → `coder` (implement) | falkor-chat design-note review |
| **2** | `ws:agent-team` workspace bootstrap (`bootstrap_schema.sh agent-team`) + `Agent`-roster seed script (§4.2) | `graph-dba`/`devops` | Stage 1 |
| **3** | Stand up a **dedicated** falkor-chat server process pinned `FALKORCHAT_WS_ID=agent-team` (§4.3, resolves `analyst`'s blocker finding — this is real design/ops work, not a wiring formality), then wire every agent's `.mcp.json` at that process's URL | `devops` (bring-up/lifecycle), `graph-dba` (naming/config review) | Stage 1 (must run a build including the `produced_by` patch) and Stage 2 (workspace should be bootstrapped before real use) |
| **4** | Raw-capture write-convention rollout: `claude/AGENTS.md`'s "Learnings capture" section repointed from `cypher-mcp`'s producer-write shape to `ingest_document` (§2); piloted with 1–2 agents before team-wide cutover | `cobb` | Stages 1–3 |
| **5** | Curator review/clear hook: `agent-maintenance` SKILL.md §5 gains the falkor-chat read (`list_documents`/`get_document`) + clear (`delete_document`) step, replacing `kaizen_team`'s `DETACH DELETE`-by-`entryId` for entries produced after Stage 4's cutover (§4.5) — named hook point, redesign of §5 stays out of scope per the requirements doc | `cobb` | Stage 4 |

`kaizen_team` is **not** decommissioned by any of these stages — it keeps running, and any entry
already there stays queryable there; Stage 4's cutover is prospective (new writes route to
`ws:agent-team` going forward), not a bulk migration of existing `kaizen_team` content, which
FR-8/AC-6 do not ask for (see §8 on this reading).

### Track 2 — distilled-knowledge ingestion (FR-2–FR-7), after Track 1

| Stage | What | Owner | Blocks on |
|---|---|---|---|
| **6** | Migrate the five/six existing KBs (FR-6/AC-4) + Stage 0's four interim ones (FR-7/AC-5) — `cobb`-led curator judgment pass, one `ingest_document` call per resulting claim (§6) | `cobb`, tooled by `graph-dba`/`coder` | Track 1 complete |
| **7** | Retrieval convention: publish `skills/agent-kb-retrieval/SKILL.md` with the query-instruction-prefix template + calibrated score floor (§4.4), point every consuming agent's prompt at it with one line, and add the `audit-team.sh` drift check (needs `cobb`'s confirmation first, §closing item b) | `cobb` (skill authoring + audit-team.sh), `architect`/`data-scientist` consulted | Stage 6 (needs real embedded content to exercise against) |
| **8** | Golden-set evaluation: pilot run → floor calibration → full ~40-pair AC-2 gate (§7, ML note Recommendation 4, unchanged) | `data-scientist`/`qa-engineer` | Stage 7 |
| **9** | `cobb`'s `agent-maintenance` SKILL.md §5 gains the ongoing distillation-ingestion step (delete-old + ingest-new, §4.5) | `cobb` | Stage 8 |

---

## 4. Design & rationale — two small, additive pieces, plus reusing what's already shipped

Under Option B, `search_documents`/`ingest_document`/`delete_document`/`list_documents`/
`get_document_history` already do everything Option A's draft had to build from scratch
(`kaizen-rag-mcp`, two `cypher-mcp` write shapes, an out-of-band embedding backfill). What Option B
still needs, confirmed by direct source reading, is exactly two small, additive things — neither a
new component, both extensions of already-shipped falkor-chat surface.

### 4.1 Attribution — a small, additive extension to `ingest_document`/`ingest_documents` (resolves Open Question #4)

**The gap, precisely** (§1): `get_context()` pins `ctx.actor` to one process-constant value for
every caller; `repository.create_document`'s `INGESTED_BY` write already resolves its target
against **either** `(u:User {userId: $ingestedBy})` **or** `(a:Agent {agentId: $ingestedBy})` — the
schema already anticipates agent-typed ingestors — but the value passed in is always that one
pinned string, never a per-call identity. Unmitigated, every agent's write in `ws:agent-team` would
attribute to the same single configured actor, discarding the per-producer attribution `kaizen_team`
carries today.

**Design:** add an optional parameter, `produced_by: str | None = None`, to
`Services.ingest_document`/`ingest_documents` and their MCP/REST surface. When given,
`repository.create_document`/`create_document_with_auto_supersede`'s ingestor-resolution clause
matches it against `(a:Agent {agentId: $producedBy})` **only** (not the `User`/`Agent` coalesce
used for `ctx.actor`) and **fails loudly** — a new `AgentNotFoundError`, mirroring
`DocumentNotFoundError`'s posture — rather than silently falling back to `ctx.actor`, if no such
`Agent` node exists yet. Silent misattribution is worse than a loud, fixable error, consistent with
this codebase's own "never silently overwrite/misattribute" convention (`document-ingestion2`'s
FR-1/FR-6 provenance-first framing). When `produced_by` is omitted, behavior is **unchanged** for
every existing caller (chat, storefront, any other workspace) — purely additive, no breaking change
to `ingest_document`'s existing contract. **Precedent for this shape of change already exists in
this codebase**: the storefront feature bolted a second, per-caller auth path
(`Storefront.context_for(pid)`) alongside the same process-constant `get_context()` seam, rather
than reworking the seam itself (`falkor-chat/docs/SERVER.md` §1.3) — this extension is smaller
still (one caller-declared string, no token/session verification, the same trust level
`cypher-mcp`'s own unverified `agent` parameter already assumes for every Claude Code agent in this
repo).

**Prerequisite this creates:** the target `Agent` node must exist in `ws:agent-team` before that
agent's first write — a one-time seed step (§4.2), not per-write.

**Scope vs. this plan's remit:** this is new server-side capability in `falkor-chat/server`, a
component this plan does not own. `graph-dba` should turn this interface spec into a short design
note (the exact Cypher predicate change, whether `ingest_documents`'s batch item schema also
carries a per-item `producedBy`, and any index implication) before `coder` implements it — the same
interface-level/implementation-detail split the pre-revision draft of this plan already used for
`cypher-mcp`'s two write shapes. Named as item (b) in the closing section.

### 4.2 Workspace bootstrap — `ws:agent-team`

Resolves FR-8's workspace-naming delegation: literal id **`agent-team`** (graph key
`ws:agent-team`) — short, matches the `ws:{workspaceId}` convention exactly (`ws:acme`/`ws:demo`/
`ws:docprobe` precedent), and reads directly as the stakeholder's own stated direction ("agent
team") without inventing a longer slug. `graph-dba`/`devops` run `bootstrap_schema.sh agent-team`
(`EMBEDDING_DIM=1024` per this repo's standing convention) once, then a new seed script,
`seed_agent_team.sh` — mirrors `seed_demo.sh`'s existing "register the demo `Agent`" step exactly,
a direct precedent rather than a new pattern: `MERGE (:Agent {agentId})` for every agent in
`claude/AGENTS.md`'s roster, idempotent, re-run whenever the roster changes (flagged for whoever
next updates `claude/AGENTS.md`'s agent-catalog maintenance rule to also touch this script — likely
`cobb`, once Track 1 ships).

### 4.3 A dedicated `ws:agent-team` falkor-chat server process — resolves `analyst`'s blocker finding

**This was originally scoped (pre-review) as a trivial `.mcp.json` wiring task — it is not.**
`analyst`'s review of this plan's first draft (`claude/docs/reviews/agent-knowledge-base-strategy.md`,
blocker finding) found that `ctx.ws` is pinned exactly the same way `ctx.actor` is (§1) — a
process-wide constant, not a per-call value, confirmed across MCP, REST, and the one existing
per-caller precedent (`Storefront.context_for`, which varies only `actor`). So reaching
`ws:agent-team` is not a matter of pointing every agent's `.mcp.json` at "falkor-chat's MCP
server," because there is no single such server in the sense that framing implies — whichever
falkor-chat process answers that endpoint is pinned, at its own startup, to one workspace, and
every call through it lands there regardless of which graph this feature's own bootstrap (§4.2)
created.

**Decision: stand up a dedicated falkor-chat server process pinned `FALKORCHAT_WS_ID=agent-team`,
`FALKORCHAT_ENABLE_AGENT=1`** (the second flag because `search_documents` needs the embedder
wired, `app._build_default_app`) — following `start_demo.sh`'s exact, already-working precedent for
the storefront's own separate `demo` workspace (`falkor-chat/AGENTS.md`'s `Key scripts` table:
"**Pins `FALKORCHAT_WS_ID=demo`**, never `config.py`'s `"acme"` default"). Every agent's `.mcp.json`
entry for this feature points at **that specific process's** URL, not at whatever endpoint an
existing chat/demo deployment happens to expose.

**Weighed against the alternative and rejected: extending `get_context()`'s resolution to vary
`ws` per-caller, the way `Storefront.context_for` already varies `actor`.** This is the more
"elegant"-looking fix at first glance (one process, many workspaces, resolved per call) — and,
corrected 2026-09-17 per `analyst`'s Pass 2 re-gate finding (this section previously misdescribed
the mechanics; the decision below is unchanged, only the reasoning is), it is **not** actually a
large change on pure mechanics: `ModelGateway`, `EmbeddingWorker`, and `IngestionPipeline` —
everything `ingest_document`/`search_documents` actually run through — already take `ws` as a
**per-call argument**, not a construction-time one. `ModelGateway.__init__`'s own docstring states
the intent directly ("resolving per call, not at construction, is what lets [the] workspace
override apply with no signature changes," `modelconfig.py:602-607`); `.llm()`/`.embedder()` take
`ws: str | None = None` on every call (`modelconfig.py:729-793`, already serving an unrelated
per-workspace model-override feature); `EmbeddingWorker.embed_chunk(self, ws: str, ...)`
(`embedding.py:230`) caches only `(ws, label)`-keyed index dimensions specifically so one instance
safely serves more than one workspace (`embedding.py:120-127`); `IngestionPipeline.extract_chunk`
likewise takes `ws` per call (`ingestion.py:100-123`); every `Repository` method resolves its graph
handle fresh per call (`db.workspace_graph`, `db.py:79-85`, no caching). So an additive
`target_ws: str | None = None` parameter on `ingest_document`/`search_documents`, mirroring §4.1's
`produced_by` shape exactly, would **not** require touching any of those three components'
internals.

**The real reason to still reject it — the one this design actually rests on:**
`get_context()` is the single auth/tenancy seam `falkor-chat/docs/SERVER.md` §1.1 locks as the one
point where "real auth replaces it without touching services/repo" — its entire value is that no
caller can pick which workspace it writes into. A caller-supplied `target_ws` would reopen exactly
that seam: any MCP/REST caller could then write into *any* `ws:{id}` graph, not only the one it's
"supposed" to use, with no authorization check guarding which workspace a given caller may name —
a cross-tenant write capability this codebase has never had and that seam is deliberately built to
foreclose. This is a materially stronger reason than a re-architecting cost that (per the source
above) doesn't actually exist. A dedicated process avoids the risk entirely, by deployment topology
rather than an unauthenticated parameter, at the cost of the extra infrastructure this section
already scopes: the dedicated-process route gets the identical outcome (an agent's write reliably
lands in `ws:agent-team`) with **zero falkor-chat code change**, reusing a mechanism the codebase
already runs in production for exactly this shape of need — and without reopening a seam this
codebase has deliberately kept closed.

**A third option — target `ws:agent-team` by feeding raw kaizen/KB writes into whatever workspace
falkor-chat's chat/demo deployment already happens to be pinned to** — is rejected outright: it
directly contradicts FR-8's explicit requirement for a *new, dedicated* workspace and the
stakeholder's own stated naming intent, not a trade-off worth weighing.

**What this concretely adds to Track 1 Stage 3, `devops`-owned, sized as real design work, not a
formality:** its own port (distinct from any chat/demo deployment's); a bring-up script mirroring
`start_demo.sh`'s shape (proposed name: `start_agent_team.sh`, `graph-dba`/`devops`'s call on the
exact name); a health check; a lifecycle decision (always-on, since this is a knowledge base agents
consult on demand throughout a session, not a scheduled batch job — recommended, not mandated,
`devops`'s call to finalize); and every consuming agent's `.mcp.json` entry pointing at this
process's URL specifically. This process must run a falkor-chat build that includes §4.1's
`produced_by` patch for FR-8/AC-6 to actually deliver correct attribution — Stage 3 depends on
Stage 1 landing in the codebase this process runs, not only on Stage 2's DDL/seed existing in the
graph it serves. **Must use the same, default model-config overlay as the rest of the deployment**
(no `ws:agent-team`-specific override in `FALKORCHAT_MODEL_CONFIG`'s workspace-override section,
`ModelGateway.resolve`'s `ws=`-scoped precedence rung) — correct and implicit by default (an
unconfigured workspace simply falls through to the shared default), named explicitly here
(`analyst`'s Pass 2 finding) so `devops` doesn't accidentally point this process at a different
embedding model when standing it up, which would silently invalidate every calibration §4.4/§7
depend on.

### 4.4 Retrieval — falkor-chat's own `search_documents`, no new MCP server

This is where "reuse the actual machinery" is realized **literally** — unlike the pre-revision
draft's `kaizen-rag-mcp` (dropped entirely, not needed): every agent calls falkor-chat's
already-shipped `search_documents(query, limit)` MCP tool directly. Read-only, `GRAPH.RO_QUERY`-
backed (`repository.search_chunks`'s own docstring), zero new component, zero new container.

- **Asymmetric query prefix (ML note Recommendation 1/Response 1, unchanged in substance).**
  `search_documents` embeds whatever string it's given verbatim — confirmed:
  `embedder.embed(query)` in `services.py`, no server-side template or preprocessing. So the prefix
  moves from server code (the Option A draft's design) to a **documented calling convention**: the
  querying agent constructs
  `f"Instruct: Given a coding agent's description of its current situation, retrieve the "`
  `f"distilled technique or rule that applies to it.\nQuery: {situation}"` itself before calling
  `search_documents`. Stored `Document`/`Chunk` text stays unprefixed (the write side never
  changes) — matches Qwen3-Embedding's documented asymmetric convention exactly, realized
  client-side instead of server-side.

  **Where this lives, concretely (resolves `analyst`'s major finding, and the ML note's own
  attached condition — "one shared, cited artifact, never duplicated").** `data-scientist`'s
  Response 1 proposed a `skills/`-style package holding the template/floor as "a small tested
  constant"; `analyst`'s review checked that against this repo's actual conventions and found the
  gap: every existing `skills/` package is markdown-only (no Python), and `claude/`'s own pytest
  standard (`cobb/TESTING.md`) is reserved for library/utility code, not agent-facing prose — there
  is no precedent here for a Python-importable, pytest-covered string constant, and (as the ML
  note's own next paragraph already concedes) nothing at runtime would import it anyway, since
  this lab's agents build MCP tool-call arguments from their own prompt-driven reasoning with no
  shared wrapper code forcing a call through it. Resolved as: **the canonical artifact is a new
  markdown skill, `skills/agent-kb-retrieval/SKILL.md`**, holding the exact prefix string and the
  current calibrated floor (with the run/date that produced it, §4.4 score-floor bullet below) as
  **fenced literals**, matching every other `skills/` package's actual shape — not a Python
  constant. Every consuming agent's prompt cites it by one line, per this repo's own "point to the
  source, don't duplicate" convention, exactly like an existing on-demand knowledge-base pointer.
  **"Tested" is realized as a structural drift check, not a pytest unit test against a runtime
  import** — a small addition to `cobb`'s `scripts/audit-team.sh` (or a standalone script in the
  same family) that greps the skill file for the exact expected fenced string and fails loudly on
  drift, the same kind of mechanical, non-pytest markdown verification `teco`'s own `Status:`-flip
  hook already performs elsewhere in this repo. This closes the **definition**-drift risk the plan's
  §7 test always meant to catch; it does **not** close **compliance** drift (one agent's live call
  silently omitting the prefix) — that gap is real, was correctly named by `data-scientist`'s
  Response 1, and is accepted, not solved, with Stage 8's recurring golden-set run as the only
  aggregate backstop (§7, §8). **Needs `cobb`'s confirmation before Stage 7 dispatch** — both that
  a new `skills/` package is the right home versus an existing agent's own knowledge-base file, and
  that `audit-team.sh` is the right place for the drift check versus a smaller standalone script —
  named in the closing section, item (b).
- **Top-K = 5** — `search_documents(query=..., limit=5)`, direct, per Recommendation 3.
- **Score floor** — `search_documents` has no floor parameter (confirmed: it returns whatever
  `limit` ranks, sorted by `score` ascending, no threshold). The floor is applied **client-side**,
  by the same calling convention, against the returned `score` — same golden-set calibration
  process as before (§7), different code applying the resulting number.
- **`familyId` sibling pull** — not available through this tool (§2); accepted loss.
- **Full-claim recovery** — a hit reports `documentId`; a caller wanting the whole claim (if its
  text happened to split across 2 chunks, §1) follows up with `get_document(documentId)`.

### 4.5 The file counterpart — settled by the requirements doc itself, not re-derived here

FR-2 already answers Open Question #2 directly: "the flat file stays the authoritative,
git-versioned artifact `cobb` distills into... ingestion is what makes its content searchable, not
a replacement for it." This **reverses** the pre-revision draft's design (there, the graph was
primary and the file was a generated export) — under Option B, `cobb`'s authoring surface is
**unchanged**: keep hand-editing the Markdown files exactly as today; the new step is *ingesting*
the (re)distilled content, not writing Cypher directly.

**A real operational wrinkle this reversal surfaces, absent from the Option A design:**
`document-ingestion2`'s update-detection machinery is built for a caller who does *not* know in
advance whether a resubmission is an edit — the opposite of `cobb`'s situation, who always knows
precisely which claim it is revising. Relying on the auto/suggested-tier detection
(`create_document_with_auto_supersede`'s exact-hash auto tier, or the background shingled-Jaccard
suggested tier) would mean an edited claim's re-ingest usually lands as a **pending** `SUPERSEDES`
suggestion (only a byte-identical-modulo-whitespace resubmission auto-supersedes) — leaving old and
new both searchable until someone calls `confirm_document_update`, exactly the staleness defect
`document-ingestion2` exists to prevent, reintroduced by omission if that confirm step is ever
forgotten. **This plan's design avoids that path entirely:** `cobb` tracks each claim's current
`documentId` (§6) and, on revision, calls `delete_document(oldId)` **then**
`ingest_document(newText)` explicitly — deterministic, immediate, no pending-suggestion window —
mirroring the pre-revision draft's own "retract-and-replace is native" reasoning, just realized
through the shipped `delete_document`/`ingest_document` pair instead of a custom Cypher
`DETACH DELETE`+`CREATE`. The auto/suggested-tier machinery still exists as a safety net for an
accidental duplicate resubmission; it is not the primary mechanism this design relies on.

---

## 5. `cobb`'s hook points (named, not redesigned)

Per the requirements doc, redesigning `agent-maintenance` SKILL.md §5 step-by-step stays out of
scope here. This plan names the exact points that future edit touches, so it is a small,
well-specified follow-up rather than a rediscovery — split across the two tracks:

**Track 1 (Stage 5) — the producer/curator write path:**
- Every agent's own raw-capture write (today: `cypher-mcp`'s producer-write shape against
  `kaizen_team`) is repointed to an `ingest_document(text=..., title=..., produced_by=<own
  agentId>)` call against `ws:agent-team` (§2/§4.1), once Track 1 Stages 1–4 ship.
- `cobb`'s existing curator review step gains a second read source: `list_documents`/`get_document`
  against `ws:agent-team` alongside `kaizen_team`'s raw `:KaizenEntry` reads, for as long as both
  exist in parallel (§3).
- `cobb`'s existing clear-after-distillation step gains a second target: `delete_document(documentId)`
  (real hard delete, `document-ingestion2` FR-4) for an entry that came from `ws:agent-team`,
  alongside the unchanged `DETACH DELETE`-by-`entryId` shape for one that came from `kaizen_team`.

**Track 2 (Stage 9) — the distillation-ingestion path:**
- `cobb`'s distillation pass gains a new step after editing a KB `.md` file: for each claim
  touched, `delete_document(oldId)` (if a prior version exists) then `ingest_document(newText,
  title=..., produced_by='cobb')` against `ws:agent-team` (§4.5) — the file edit stays the
  authoring act; ingestion is a follow-on sync step, not a replacement for it.
- Tracking each claim's current `documentId` (so the delete-then-recreate step knows what to
  delete) is **left to Stage 9's implementer**: a small local manifest, or a `list_documents`+
  title-match lookup at the corpus's expected small scale (~100–250 entries, infrequent cadence) —
  either is workable; not decided here (§closing item c).
- **`MENTIONS`-equivalent tagging (a second agent) has no direct falkor-chat analogue** — flagged
  as a genuinely open item, not resolved: automatic entity extraction (§1) might incidentally
  surface an agent-name mention as a noisy `Entity`, but that is not a substitute for `cobb`'s
  deliberate curator tag. Since `kaizen_team` stays operational in parallel (§3), an entry that
  truly needs this can still be written there in the interim.

---

## 6. Migration (FR-6/AC-4, FR-7/AC-5) — Track 2 Stage 6

**A `cobb`-led curator judgment pass, not a mechanical script** — unchanged in substance from the
pre-revision draft, per the ML note's Recommendation 2 (cited, not re-argued: the requirements
doc's "one heading = one technique" assumption is false for the actual files —
`review-techniques.md` alone has headings bundling 5+ independently-attributed sub-claims past
1,500 words). Only the write mechanism at the end changes (an `ingest_document` call, not a Cypher
write):

1. **`graph-dba`/`coder` build a candidate-flagging tool** (not a splitter) — per file, per `##`
   section: word count, count of bolded sub-headers/enumerated checks, count of "Origin:" mentions.
   Flags a heading as a split candidate past the ML note's stated trigger (~500 words spanning more
   than one verifiable claim, or multiple independently-attributed sub-claims) — mechanical,
   scriptable, surfacing candidates, not deciding them.
2. **`cobb` reviews every flagged heading and decides the actual split boundary** — the same
   editorial judgment it already exercises during ordinary distillation (§5). An unflagged heading
   migrates as a single claim unchanged.
3. **`cobb` issues one `ingest_document` call per resulting claim** (§2), `produced_by='cobb'`,
   `title` carrying the shared family-slug prefix (§2) for any claim split from the same original
   heading. Verify each multi-line `text` write byte-exact via a `get_document` re-read, mirroring
   `cypher-mcp/README.md`'s documented "verify a multi-line write byte-exact, never by eyeballing a
   truncated response" gotcha — the same trap applies to any multi-line text write in this lab.
4. Both existing embedding and (accepted, §1) extraction fire automatically per §4.4's shipped
   pipeline — no separate backfill step is needed under Option B (a genuine simplification versus
   the Option A draft's dedicated backfill script, §4).

**Content-loss check (AC-4/AC-5's own bar), unchanged in method:** every word of a source section's
body must be accounted for by exactly one resulting claim-document's `text` (a `cobb`-verified
partition, not merely a count match) — spot-checked via a scripted diff of "source section text"
against "concatenation of its family's claim texts," not a manual read of the whole corpus.

**Effort, sized rather than assumed away** (the ML note's own flagged risk, unchanged): this is
real, additional migration work for the pre-existing six KBs relative to a naive "mechanical split"
assumption — but because Track 0's four interim KBs go through the identical process (the Stage 0
FR-7 constraint, §3), the *relative* cost FR-7 gates stays even; only the *absolute* cost of Stage 6
as a whole is real, worth sizing before dispatch, not a violation of FR-7's own bar.

Whether the source `.md` files stay the permanent authoritative copy indefinitely (settled, §4.5 —
yes, per FR-2's own text) means this migration never produces a "which copy wins" ambiguity the
Option A draft's generated-export design had to resolve — one less open question under Option B.

---

## 7. Test strategy

- **The `produced_by` extension (§4.1)** — owned by `graph-dba`/`coder` once the design note lands
  (item (b), closing section); required coverage named here so it isn't rediscovered: an existing
  `Agent` succeeds and `INGESTED_BY` resolves to it (not `ctx.actor`); a missing `Agent` raises
  `AgentNotFoundError` and creates nothing (no partial `Document`); omitting `produced_by` is
  byte-for-byte unchanged behavior for an existing caller (a regression test against the current
  chat/storefront ingestion path, not just a new-path test).
- **Attribution correctness, end-to-end** — once Track 1 Stage 4 is live, a spot check that
  `list_documents`'s `ingestedByKind`/`ingestedById` for a sample of `ws:agent-team` entries
  actually varies by producing agent, not uniformly the single configured actor — the concrete,
  observable proof that §4.1's fix landed, not just that its unit tests pass.
- **Chunking-mitigation spot check (§1/§2)** — for a realistic sample of claim lengths (a mix of
  short kaizen entries and long distilled claims with worked examples), confirm most stay one
  chunk and any that split stay within one `documentId`, both chunks retrievable together via
  `get_document` — not a hard gate, a sanity check that the mitigation is doing what §1 claims.
- **AC-2 (semantic match under reworded queries)** — the golden-set design is the ML note's
  Recommendation 4, quoted once here: a **~40-pair, independently-authored** (never `cobb` — a
  curator's own paraphrase would echo the stored entry's vocabulary and understate the real gap
  AC-2 exists to close) golden set, **stratified** across (a) every KB weighted by entry count, (b)
  code/regex/shell-heavy entries vs. prose-only, (c) 4–6 explicit **negative** queries with no
  matching entry, to confirm the (now client-side, §4.4) score floor rejects rather than
  force-feeds, and (d) a handful of near-duplicate stress pairs. Scored as **recall@5 primary, MRR
  secondary**, reported with a **Wilson-interval CI**, never a bare percentage. **Sequencing:** a
  ~15–20-pair pilot on one dense agent's KB once Track 2 Stage 6 has produced a first real slice
  (this is what produces the first calibrated score floor, §4.4); the full ~40-pair set becomes the
  AC-2 regression gate once Stage 6 migration covers both Track 0's four interim KBs and the
  pre-existing five/six.
- **AC-4/AC-5 (migration, no content loss)** — the scripted partition-diff check in §6, run once
  per migrated file.
- **Full pipeline smoke** — one hand-run raw-capture entry through `ingest_document` →
  `search_documents`, before Track 1 Stage 4's team-wide cutover; one hand-run distillation claim
  through delete-then-`ingest_document` → `search_documents`, before Track 2 Stage 6's bulk
  migration — each catches a convention mistake on one entry rather than discovering it mid-rollout.
- **The query-prefix template/score floor (§4.4)** gets a structural drift check — a small addition
  to `scripts/audit-team.sh` (or a standalone sibling script) asserting the exact fenced string in
  `skills/agent-kb-retrieval/SKILL.md`, not a pytest unit test (no precedent for a
  pytest-covered, runtime-imported constant in `skills/`/`claude/`, per `analyst`'s major finding
  — §4.4). This catches **definition** drift (e.g. a future edit dropping the `Instruct:`/`Query:`
  framing) only — it cannot catch one agent's live call silently omitting the prefix
  (**compliance** drift), a gap `data-scientist`'s ML note names explicitly and this plan accepts,
  with Stage 8's recurring golden-set run as the only aggregate backstop.

**Two questions the ML note explicitly leaves open, carried forward rather than resolved here**
(Recommendation 1's own "what would change my mind," and that note's own risks section): whether
Qwen3-Embedding-0.6B's code-retrieval sub-score holds up on this corpus's code/regex-dense entries
— answered only once the stratified golden-set run produces a number — and the exact score-floor/
top-K values, provisional until that same run.

---

## 8. Risks & open questions

- **Chunking-granularity mitigation is real but partial (§1/§2).** One-claim-per-document sidesteps
  cross-claim dilution but not an unusually long worked example landing in a second chunk of the
  same document — a residual, not a full fix. Watch for it in the Stage 8 golden-set run (a
  near-duplicate/long-entry stress case would surface it).
- **Extraction noise is an accepted, not eliminated, cost (§1).** Every `ingest_document` call
  fires unconditional LLM-based entity/relationship extraction into `ws:agent-team`; confirmed
  inert for this feature's own retrieval (§4.4's `search_documents` is `Chunk`-ANN-only, no
  traversal) but it is real RAM/complexity a future feature reaching into this workspace could trip
  over — named for whoever next builds against `ws:agent-team`, not mitigated here.
- **The `produced_by` extension (§4.1) is a real prerequisite, not a formality.** Until it ships,
  Track 1's team-wide, per-agent-attributed migration (FR-8's own framing, AC-6) cannot actually
  deliver differentiated attribution — every write would land under the same single configured
  actor. Sequencing this as Track 1's first stage (§3) is deliberate, not incidental.
- **`familyId` sibling-pull is a real, accepted quality loss versus the Option A draft (§2/§4.4)** —
  mitigated only by a title-prefix naming convention, not automatic retrieval. Each claim-document
  is self-contained by construction for the primary (single-rule) retrieval case, so this is
  probably a convenience loss, not a correctness one — but it is an empirical question, not
  architect's or `data-scientist`'s to assert: the Stage 8 golden set's **dedicated multi-facet/
  sibling-claim stratum** (ML note Recommendation 4(e), scored separately by set-recall, never
  blended into headline recall@5) is what actually settles it. **Correction:** an earlier version of
  this bullet pointed at the golden set's *near-duplicate* stress pairs as the check — wrong; those
  test discrimination (false-positive avoidance), a different construct from this stratum's
  completeness question, per the ML note's own explicit correction.
- **The dedicated `ws:agent-team` server process (§4.3) is real, previously-unscoped infrastructure,
  not a wiring formality** — a second always-on falkor-chat deployment `devops` must design
  (port, bring-up, health check, lifecycle), found only by `analyst`'s independent review, not by
  this plan's original drafting pass. Loop `devops` in alongside Track 1 Stage 1's `graph-dba`
  design note, ahead of dispatch, rather than discovering this mid-implementation (this plan's own
  answer to the review's first open question).
- **`.mcp.json` per-agent wiring is still a real, easy-to-forget one-time task once §4.3's process
  exists** — the same per-directory trust-approval quirk `cypher-mcp/README.md` already documents,
  now against a URL specific to this new process, not an existing chat/demo endpoint.
- **`delete_document`'s audit attribution is not covered by §4.1's `produced_by` fix
  (`analyst`'s minor finding 2).** `repository.delete_document` still stamps `deletedBy` from the
  single pinned `ctx.actor`, never the calling agent — every `delete_document` call this plan names
  (§5's curator clear-after-distillation step, Track 2 Stage 9's delete-then-recreate revision)
  is `cobb`'s own, so this doesn't undermine FR-8's actual driver (differentiating *producers*), but
  `get_document_deletion`'s `deletedById` will read as "the pinned config actor, always" for anyone
  who later expects real per-agent attribution there — named explicitly rather than left to be
  discovered as a surprise. Not designed around here; extending `produced_by`'s pattern to
  `delete_document` if it's ever needed is a small, same-shape follow-up, not a redesign.
- **This plan reads FR-8/AC-6 as a prospective cutover, not a bulk backfill of existing
  `kaizen_team` content** — "every agent... can write and read its raw capture through a
  falkor-chat-workspace destination" is read as "new writes route there going forward," with
  existing `kaizen_team` entries staying queryable in place, not copied. If the stakeholder actually
  intends a one-time bulk migration of `kaizen_team`'s current backlog into `ws:agent-team` too,
  that is additional, unscoped work this plan does not currently cover — flagged explicitly since
  either reading is buildable from where this plan stands (a bulk migration would be a strict
  addition, not a rework), not something worth blocking dispatch on.
- **Migration effort for Track 2 Stage 6 is real, not assumed away** — a `cobb`-led curator pass,
  not a mechanical script, because the "one heading = one technique" premise the requirements doc
  originally carried is false against the real files (§6). Sizing/scheduling it (especially for
  `review-techniques.md`, the densest KB) deserves a deliberate look before dispatch.
- **Code-retrieval quality on this corpus is genuinely unverified** (ML note Recommendation 1) —
  stays open until the stratified golden-set run (§7) produces a number; the documented fallback is
  a cheap, re-embed-only upgrade (`qwen3-embedding:4b`, same 1024-dim family), not a redesign.
- **Score floor/top-K are provisional** (§4.4, §7) until the golden-set pilot run — ship the
  retrieval convention with the floor disabled and calibrate from the pilot, exactly as the
  pre-revision draft already specified for its own retrieval server.
- **The claim→`documentId` tracking mechanism for Stage 9's delete-then-recreate step (§5)** is
  named but not resolved — genuinely Stage 9's implementer's call.

---

## Ready to implement

Plan at `claude/docs/plans/agent-knowledge-base-strategy.md`, with its ML method note at
`claude/docs/plans/agent-knowledge-base-strategy-ml.md` (Version 2 — already revised by
`data-scientist` to answer this plan's original two asks; not revised again by this pass, see item
(a) below for why).

**Revision note (2026-09-17, third pass): responds to `analyst`'s Pass 2 re-gate**
(`claude/docs/reviews/agent-knowledge-base-strategy.md`, `## Pass 2`, verdict: **approve with
suggestions**). All four Pass 1 findings confirmed genuinely fixed. Pass 2's interaction check
(does the dedicated-process design conflict with anything else this plan or the ML note assumes)
found one new, narrow issue: §4.3's stated reason for *rejecting* the per-call-`ws` alternative was
factually wrong — `ModelGateway`/`EmbeddingWorker`/`IngestionPipeline` already take `ws` as a
per-call argument (verified independently against `modelconfig.py`/`embedding.py`/`ingestion.py`/
`db.py` before editing, not taken on the review's word), so "re-architecting those components"
overstated the cost. **The decision (a dedicated process) doesn't change** — §4.3 now gives the
actually-correct, stronger reason: a per-call `target_ws` parameter would reopen `get_context()`'s
locked auth/tenancy seam to unauthorized cross-tenant writes, which a dedicated process avoids by
deployment topology instead. Also folded in: §4.3/§3 Stage 3 now states explicitly that the
dedicated process must use the same, default model-config overlay as the rest of the deployment
(implicit and correct today, cheap to make explicit so `devops` doesn't accidentally diverge it).
Nothing else in this document changed — Pass 2 confirmed the rest of the fixes and the
plan/ML-note interaction hold as designed.

**Revision note (2026-09-17, second pass): responds to `analyst`'s first independent review**
(`claude/docs/reviews/agent-knowledge-base-strategy.md`, verdict: needs changes). Blocker resolved:
§1 and §4.3 now name and close the `ws`-half of the same process-pinning finding §1 already had for
`actor` — reaching `ws:agent-team` needs a **dedicated falkor-chat server process**
(`FALKORCHAT_WS_ID=agent-team`), not a `.mcp.json` edit against an existing endpoint; the
alternative (extending `get_context()` to vary `ws` per-call) was weighed and rejected as a much
larger, disproportionate falkor-chat-core change (§4.3). Major resolved: the query-prefix/score-floor
artifact is now concretely placed — `skills/agent-kb-retrieval/SKILL.md` (fenced literals, matching
every other `skills/` package's actual markdown-only shape) plus a structural drift check on
`scripts/audit-team.sh`, not a pytest-covered Python constant with no precedent or runtime import
path (§4.4). Both minors folded in: §1's extraction-scheduling description now matches
`mcp.ingest_document`'s actual conditional gate; §8 names `delete_document`'s audit attribution as
not covered by §4.1's fix, low-stakes given every delete this plan calls for is `cobb`'s own.

**Dispatchable now: Track 1 (§3), Stages 1–5, in order — with Stage 3 correctly sized as real
`devops`/`graph-dba` design work, not the wiring formality it was scoped as before this review.**
Stage 1 (the small `produced_by` extension to falkor-chat's `ingest_document`/`ingest_documents`,
§4.1) and Stage 3 (the dedicated `ws:agent-team` process, §4.3) are both real prerequisites with
no dependency on each other's *design* (only Stage 3's *deployment* needs Stage 1's code merged) —
loop both `graph-dba` (Stage 1's design note) and `devops` (Stage 3's process design) in now, ahead
of dispatch, per this plan's own answer to the review's first open question. Track 2 (Stages 6–9)
is sequenced after Track 1 completes, per FR-9 — do not dispatch it early.

**(a) ML method note status — no further revision needed from this review round.** The blocker
(workspace-pinning/process design) is outside the ML note's remit entirely — corpus, embedding,
chunking, retrieval-parameter, and evaluation design are all unaffected by which server process
serves `ws:agent-team`. The major finding's resolution (§4.4 — the artifact is a `skills/`-style
markdown package with a structural drift check, not a pytest constant) **makes concrete, rather
than contradicts, `data-scientist`'s own Version 2 recommendation** — that note already proposed
"a small `skills/`-style package" for the template/floor (Response 1, quoted verbatim in §4.4
above); it did not itself resolve what "tested" means given no pytest precedent for agent-facing
prose (`analyst`'s major finding named that gap directly), and this pass closes exactly that gap —
so it does not need a v3. `data-scientist`'s Version 2 already answered this plan's original two
asks in full (client-side prefix/floor realization confirmed sound with the "one canonical
artifact" condition this pass resolves; the `familyId`-loss question converted into a testable
golden-set stratum rather than asserted either way) — both responses are cited by section
throughout §1/§2/§4.4/§7/§8 above, not re-argued here.

**(b) Design notes/confirmations needed before implementation starts:** `graph-dba` should turn
§4.1's `produced_by` interface spec into a short falkor-chat-side design note (exact Cypher
predicate change, whether `ingest_documents`'s batch schema also carries a per-item `producedBy`,
any index implication) before `coder` implements it. `devops` (with `graph-dba` reviewing naming/
config) owns §4.3's dedicated-process design — port, bring-up script, health check, lifecycle — a
genuinely new scope item this review surfaced, not previously sized. `graph-dba`/`devops` also own
§4.2's workspace-bootstrap mechanics (`bootstrap_schema.sh agent-team` + `seed_agent_team.sh`).
`cobb` should confirm, before Track 2 Stage 7 dispatch: that a new `skills/agent-kb-retrieval/`
package is the right home for the query-prefix/floor artifact (versus folding it into an existing
agent's knowledge base), and that `scripts/audit-team.sh` is the right place for its drift check
(versus a standalone script) — §4.4's design is this plan's recommendation, not a decision only
`cobb` can make final.

**(c) Left open for a later stage's own implementer, not resolved here** (mirrors this plan's own
prior practice of naming rather than guessing these): how `cobb` tracks each claim's current
`documentId` for the delete-then-recreate revision step, a manifest vs. a `list_documents` scan
(Track 2 Stage 9, §5); the pilot rollout order for Track 1 Stage 4's team-wide cutover (which 1–2
agents go first); the exact lifecycle posture (always-on vs. something else) for §4.3's dedicated
process, `devops`'s call to finalize; and, per the requirements doc's own Open Questions, the
`MENTIONS`-equivalent tagging gap (§5) and `kaizen_team`'s eventual retirement timing — both
explicitly out of scope for this revision, unchanged from the requirements doc.

**This plan's own view on next steps:** ready for a second, narrower review pass focused on the
four findings' resolutions (§1, §4.3, §4.4, §8) rather than a full re-review — the rest of the
document is unchanged from what `analyst`'s first pass already found sound (content model,
migration approach, cross-document consistency with the ML note). Recommend `analyst` confirm the
blocker/major fixes land as designed before Track 1 Stage 1/3 dispatch; the two minors don't
warrant gating re-review on their own.
