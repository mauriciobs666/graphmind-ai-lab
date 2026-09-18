# Agent knowledge-base strategy — design review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

## Scope & verdict

First independent review of the substrate design for the agent knowledge-base strategy — until
now only Stage 0 (the interim flat-file KB relief) had been reviewed
(`claude/docs/reviews/agent-knowledge-base-strategy-impl.md`, diff-scoped implementation review).
Reviewed together, both independently and cross-checked against each other:

- `claude/docs/plans/agent-knowledge-base-strategy.md` (Status: active, `architect`, revised in
  place 2026-09-17 against the requirements doc's Option B decision)
- `claude/docs/plans/agent-knowledge-base-strategy-ml.md` (Status: active, Version 2,
  `data-scientist`, revised in place 2026-09-17 answering `architect`'s two explicit asks)

against `claude/docs/requirements/agent-knowledge-base-strategy.md` (decision log through
2026-09-17) and, for every falkor-chat-side claim in either document, the real, current source:
`falkor-chat/server/falkorchat/{services,repository,mcp,background,chunking,config,app,
storefront}.py` and `falkor-chat/AGENTS.md`/`falkor-chat/docs/SERVER.md`. Out of scope: §3's
Stage 0 subsection (closed, not reopened), and any judgment of whether Option B was the right
stakeholder call (settled upstream, not this review's to relitigate).

**Verdict (Pass 1): needs changes.** One blocker: the plan's wiring stage (§4.3) treats reaching
`ws:agent-team` as a trivial `.mcp.json` edit, but falkor-chat's workspace resolution is a
single, process-wide constant — the design as written would silently route every agent's write
into whatever workspace the reachable falkor-chat server process happens to be pinned to, not
`ws:agent-team`, unless a dedicated new server process is stood up and designed for. Everything
else — the attribution fix, the content model, the migration approach, the ML methodology
(including its two explicit answers to `architect`) — is sound, accurately grounded in the real
source, and internally consistent between the two documents.

**Superseded by `## Pass 2` below — read that section for the current verdict.**

**CPG:** considered, not relevant — `cpg_falkorchat` exists but is stale for this task (built
2026-09-12, source `ca25a20`, predating `document-ingestion2`'s Stage F close, commit `8906878`,
per both documents' own CPG notes and the coordination brief). Every falkor-chat-side claim below
was verified by direct, current source reads, not the CPG.

## Findings

### Blocker — falkor-chat's workspace (`ws`) is a process-wide pin, exactly like `actor`; §4.3 treats reaching `ws:agent-team` as a trivial wiring task, but it isn't

`config.get_context()` (`falkor-chat/server/falkorchat/config.py:276-284`) returns
`CallContext(ws=WS_ID, actor=USER_ID)`, both module-level constants read once from env
(`config.py:16-17`). The plan's §1 finding about `ctx.actor` being pinned is accurate and
well-evidenced (confirmed independently below) — but `ctx.ws` is pinned by the *exact same
mechanism*, and neither document names this. Confirmed across every front door:

- MCP: `mcp.py:41`, `_get_context: Callable[[], CallContext] = config.get_context` — a
  module-global, swappable only at `configure()`-time (test injection), never per-call.
- REST: `api.py`, every route depends on `Depends(get_context)` (`config.get_context` re-exported)
  — no per-request workspace parameter anywhere in the router.
- The one precedent for varying `CallContext` per caller, `Storefront.context_for`
  (`storefront.py:678-685`), varies only `actor` (`participant_id`) — `ws` stays `self._ws`, the
  same single workspace the whole `Storefront` instance was built with.

And `falkor-chat/AGENTS.md`'s own `Key scripts` table independently confirms the operational
consequence already lives in this exact codebase: `start_demo.sh` "**Pins
`FALKORCHAT_WS_ID=demo`**, never `config.py`'s `"acme"` default" — a different target workspace
already requires a **separate running server process** with its own env, not a parameter on a
shared one. `app._build_default_app` (`app.py:580-657`) confirms why: `ModelGateway`,
`EmbeddingWorker`, `IngestionPipeline` — everything `search_documents`/`ingest_document` need —
are constructed once per process at startup, gated only on `FALKORCHAT_ENABLE_AGENT` (default `1`,
`start_server.sh:93`), with no workspace-scoping of any kind.

So Track 1 Stage 3 as the plan currently states it — "the URL/port, any transport-specific
`.mcp.json` shape difference... and the 'one-time trust approval' step... sized small and
concrete, not a design question" (§4.3) — is not achievable by wiring agents to whichever
falkor-chat MCP endpoint already exists (the one serving chat, pinned to `acme`/whatever `WS_ID`
that deployment runs with, or the storefront's `demo`). Every `ingest_document`/`search_documents`
call through that endpoint would land in **that** endpoint's pinned workspace, not
`ws:agent-team`, regardless of §4.2's bootstrap having created the `ws:agent-team` graph
correctly. This isn't hypothetical drift — it's how `get_context()` is documented to behave today
("M1 resolves every call to one hardcoded tenant," `config.py:277-278`), and it would silently
defeat AC-6/AC-7 (every agent "can write and read its raw capture through a falkor-chat-workspace
destination") without any test in §7 catching it, since nothing there checks *which graph* a
write landed in.

**Fix, scoped, not a whole-design rework:** name a new stage (or fold into Stage 2/3) for standing
up a **dedicated falkor-chat server process** pinned to `FALKORCHAT_WS_ID=agent-team` (and
`FALKORCHAT_ENABLE_AGENT=1`, since `search_documents` needs the embedder wired) — following the
exact precedent `start_demo.sh` already sets for the storefront's `demo` workspace. This has real,
previously-unstated operational shape `devops` needs to design: its own port, its own
bring-up/health-check/lifecycle (who starts/monitors it, whether it's always-on), and every
agent's `.mcp.json` entry must point at *that* instance's URL specifically, not "falkor-chat's MCP
server" as if there is only one. Also revise §1's framing — "every Claude Code agent in this repo
would share the *same* falkor-chat MCP endpoint" (used to motivate the attribution-gap finding)
is only true *once* this dedicated endpoint exists; today it conflates "falkor-chat's machinery"
(shared, correctly) with "falkor-chat's *running instance*" (not shared, once this is designed
correctly) — worth being explicit about, since it's part of why "reuse the actual machinery
literally" (§4.4) costs a second standing process, not zero extra infrastructure.

### Major — data-scientist's "single canonical artifact, tested constant" (item c) isn't concrete enough to dispatch Stage 7 against as-is

The ML note's Answer 1 recommends landing the query-prefix template and score floor as "a small
`skills/`-style package... each as a fenced literal / a small tested constant... asserting the
exact string for a given `situation`." Checked directly: every existing `skills/` package is
markdown-only (`find skills -iname "*.py"` → only `joern-cpg/scripts/cpg-to-falkordb.py`, an
operational script, not a tested library), and `claude/`'s own testing convention
(`claude/cobb/TESTING.md`, "The two-altitude standard") reserves pytest for **library/utility
code** (its exemplar is `excel_extractor/`) — agent-facing prose gets the eval/bless harness
instead, which asserts substrings in a model's *response*, not a stored string's definition.
Neither `skills/` nor `claude/` has any precedent for a Python-importable, pytest-covered constant
today. This isn't just a missing scaffold: the ML note's own next paragraph ("A gap the plan's §7
test does not close") already concedes **no shared wrapper code forces any call through this
artifact** — this lab's agents build MCP tool-call arguments from their own prompt-driven
reasoning, reading whatever prose the skill file states. So a "tested Python constant" would be
checked by a test nothing at runtime ever imports or executes — the artifact an agent actually
*reads* (markdown prose) and the artifact a test actually *checks* (a Python string) would be two
copies of the same fact with no mechanism keeping them in sync, which is precisely the
duplication risk "single canonical artifact, never duplicated" was meant to close. Recommend this
become its own explicit decision before Stage 7 dispatch — where the prose lives, what "tested"
concretely means given no pytest precedent in `claude/`/`skills/`, and how (if at all) the two
stay provably identical — rather than treating item (c) as resolved by this note.

### Minor — plan §1's "unconditionally call `_schedule_chunk_processing`... no toggle" slightly overstates the mechanism, though its conclusion holds

`mcp.ingest_document` gates the call behind `if _embed_worker is not None or _ingestion_pipeline
is not None:` (`mcp.py:309`) — conditioned on deployment-level wiring, not literally
unconditional. Confirmed the practical conclusion is still right: `app._build_default_app`
(`app.py:595-657`) always constructs and wires `ingestion_pipeline` whenever
`FALKORCHAT_ENABLE_AGENT` is on, which `start_server.sh` defaults to `1` — so under the deployment
this plan will actually run against, extraction does fire unconditionally per call, just not
because the MCP tool itself has no gate. Wording nit only; doesn't change §1's "accepted, not
eliminated" verdict on the extraction-noise cost.

### Minor — the `produced_by` fix (§4.1) doesn't extend to `delete_document`'s audit attribution

`repository.delete_document` (`repository.py:1288-1326`) stores `deletedBy: $deletedBy` from
`services.delete_document`'s `deleted_by=ctx.actor` (`services.py:1362-1365`) — read back via
`get_document_deletion` (`repository.py:1328-1348`). §4.1's `produced_by` extension patches only
`ingest_document`/`ingest_documents`'s `INGESTED_BY` edge, so every `delete_document` call
(§4.5's curator clear-after-distillation step, and the delete half of Track 2 Stage 9's
delete-then-recreate revision flow) still stamps the audit trail with the single pinned
`ctx.actor`, never the calling agent. Low-stakes as designed — every `delete_document` call this
plan's own hook points name is `cobb`'s (§5), not per-producer-differentiated, so this doesn't
undermine FR-8's actual driver (differentiating *producers*) — but worth naming explicitly in §8's
risk list rather than leaving `get_document_deletion`'s `deletedById` silently meaning "the
pinned config actor, always" for anyone who later reads it expecting real attribution.

## What's solid

- **The requirements doc's §1/decision log is honest about what's actually resolved.** Verified
  directly: `document-ingestion2` shipped exactly update/delete/versioning
  (`delete_document`/`list_documents`/`get_document_history` all present and reachable,
  `services.py:1358-1401`) — the doc correctly states only one of three original objections
  closed, naming the other two as accepted costs rather than folding them into "Option B is now
  fully clean."
- **The attribution-gap finding (§1) and its `produced_by` fix (§4.1) are accurately grounded.**
  Independently confirmed `get_context()`'s single-actor pin and `create_document`'s existing
  dual `User`/`Agent` OPTIONAL-MATCH-and-coalesce resolution (`repository.py:1061-1062,
  1803-1804`) exactly as described, plus the precedent cited for "fail loudly" (`UnknownActorError`
  already exists, `services.py:214`) and for the additive-second-context-path shape
  (`Storefront.context_for`, confirmed above).
- **`search_documents`'s behavior (ML note Answer 1) checks out a third time, byte-for-byte.**
  `services.py:1309-1354` matches the cited line range exactly: verbatim `embedder.embed(query)`,
  no template, no floor, `k = limit * SEARCH_DOCUMENTS_OVERFETCH`, `repository.search_chunks`
  confirmed ANN-only with zero `Entity`/traversal expansion (`repository.py:1232-1266`).
- **Answer 2's multi-facet golden-set stratum is a sound way to close the `familyId`-loss
  question empirically** — bounded (4-6 pairs within the existing ~40-pair budget), a genuinely
  different metric (set-recall) from the existing near-duplicate stratum, a stated decision rule,
  and a concrete, no-schema-change fallback (an extra `list_documents` title-prefix call) named
  but not built speculatively. This converts an architectural guess into a falsifiable test rather
  than deferring the risk unmeasured.
- **Cross-document consistency is clean.** Everywhere the plan and ML note touch the same point —
  content model (one claim per `Document`), title-prefix sibling mitigation, client-side
  prefix/floor, top-K=5, owning-agent scoping via `produced_by` — they agree, cite each other
  correctly, and neither overstates what the other settled.
- **Document-convention compliance is sound.** Both documents' revise-in-place calls under rule 5
  are correctly reasoned (Stage 0 is the only executed/gated part; the substrate sections were
  never approved against, confirmed by this being their first review); the header blocks (`Status:`
  token-first, `Version: 2` on the ML note) are correctly formed; the paragraph-length dated
  revision notes match this repo's own established precedent for plan-kind documents
  (`falkor-chat/docs/plans/graphrag-eval.md`'s v2-v4 revision notes are the same shape), not a
  violation of the "one dated line" text read literally.
- **Test strategy (§7) is proportionate and correctly sequenced**, including honestly naming what
  it *can't* catch (compliance drift on the query-prefix template) rather than overclaiming
  coverage.

## Open questions

- Should `devops` be looped in now, ahead of Track 1 dispatch, to scope the dedicated
  `ws:agent-team` server-process design the Blocker above requires — sized alongside Stage 2/3
  rather than discovered mid-implementation?
- For the Major finding on the query-prefix/floor artifact: is there an existing or planned
  lightweight test mechanism in this lab (outside `claude/`/`skills/`'s current conventions) that
  the Blocker's dedicated `ws:agent-team` service surface might reasonably also carry, which would
  give "a small tested constant" an actual home — worth `architect`/`data-scientist` deciding
  together rather than either asserting alone.

## Pass 2 (2026-09-17) — narrow re-gate of `architect`'s in-place revision

Scope: `architect` revised `claude/docs/plans/agent-knowledge-base-strategy.md` in place (no
successor) to address all four Pass 1 findings. This pass verifies each fix against the actual
current text and the real falkor-chat source (not the prose), and specifically checks the fixes
**against each other** for interaction defects, per the coordinator's brief. Stage 0 and the ML
note (unchanged, still v2) stayed out of scope except where the interaction check required
tracing into the ML note.

**Verdict: approve with suggestions.** All four Pass 1 findings are genuinely fixed. One new
finding, surfaced by the interaction check the coordinator specifically asked for: the plan's own
justification for *rejecting* the per-call `ws` alternative (§4.3) rests on a factual claim about
`ModelGateway`/`EmbeddingWorker`/`IngestionPipeline` that direct source-reading shows is wrong.
The **decision** the plan reaches (a dedicated server process) is still correct — arguably more
strongly justified than the plan itself realizes, on grounds it doesn't currently name — so this
does not block Track 1 Stage 1–3 dispatch, but the reasoning text should not ship uncorrected.

### Disposition of the four Pass 1 findings

- **Blocker (workspace pinning, §1/§4.3) — fixed, as a buildable decision.** §4.3 now mandates a
  dedicated falkor-chat server process pinned `FALKORCHAT_WS_ID=agent-team`
  (`FALKORCHAT_ENABLE_AGENT=1`), following `start_demo.sh`'s exact precedent; §1, §3 Stage 3, §8,
  and the closing section all reference this consistently (re-checked each site, no stray
  reference to the old "just wire `.mcp.json`" framing remains). This correctly closes AC-6/AC-7's
  routing gap. **But see the new Major finding below** — the stated reason for rejecting the
  per-call alternative is inaccurate, even though the decision itself holds.
- **Major (prefix/floor artifact, §4.4) — fixed, concretely.** §4.4 now names
  `skills/agent-kb-retrieval/SKILL.md` (fenced literals, matching every other `skills/` package's
  actual markdown-only shape) plus a structural drift check on `scripts/audit-team.sh`, explicitly
  *not* a pytest-covered constant — correctly resolving the runtime-import mismatch this review's
  Pass 1 raised. Confirmed present: the closing section's item (b) explicitly names `cobb`'s
  confirmation (home + drift-check location) as needed before Stage 7 dispatch, not silently
  assumed.
- **Minor 1 (extraction-scheduling wording, §1) — fixed.** §1 point 3 now states the
  `if _embed_worker is not None or _ingestion_pipeline is not None:` gate precisely
  (`mcp.py:309`) and separately explains why it fires unconditionally *in practice* for this
  deployment — re-verified against the same source line, matches.
- **Minor 2 (`delete_document` attribution gap, §8) — fixed.** §8 gained the bullet naming
  `deletedBy` as uncovered by §4.1's fix, low-stakes because every `delete_document` call this
  plan names is `cobb`'s own — matches Pass 1's finding and its own stated low-stakes framing.

### Self-reported fixes — spot-checked

- **§8's `familyId` risk-bullet correction** — verified as a genuine, correct fix, not just a
  claim. Pass 1's plan text had this bullet pointing at "the Stage 8 golden set's near-duplicate
  stress pairs" as the check for sibling-pull loss; the ML note's Response 2 (Version 2, already
  in place at Pass 1) explicitly says near-duplicate pairs test discrimination, not completeness,
  and that the two must be scored and reported separately. The Pass 1 plan text and the
  already-shipped ML note therefore genuinely disagreed on this one point — **a real
  plan/ML-note inconsistency Pass 1 missed** (this review's own Pass 1 "cross-document consistency
  is clean" claim was too broad by this one bullet). The current text's "Correction:" paragraph
  fixes it accurately, now citing the right stratum (the dedicated multi-facet/set-recall one).
- **Closing-section mis-cited quote/document** — no independent before/after diff was available
  (Pass 1 didn't cite this specific line as a finding), so this can't be falsified against a prior
  state. Spot-checked the current closing section's citations for accuracy instead: item (a)'s
  quote, "a small `skills/`-style package," matches the ML note's Response 1 text verbatim, and
  every document/section cross-reference checked (the review's own blocker citation, `§4.3`,
  `§8`) resolves correctly. No residual inaccuracy found in the current text.

### New finding — Major: §4.3's rejection of the per-call `ws` alternative rests on an inaccurate claim about `ModelGateway`/`EmbeddingWorker`/`IngestionPipeline`

§4.3 argues extending `get_context()` to vary `ws` per-call (mirroring how `produced_by` varies
`actor`) is rejected because "`ModelGateway`, `EmbeddingWorker`, and `IngestionPipeline`... are
constructed once per process at startup... with no workspace-scoping of any kind... varying `ws`
too would mean re-architecting those three components to be workspace-parametrized per call."
**The "constructed once at startup" half is true; the "no workspace-scoping of any kind" half is
false, and the false half is what the "re-architecting" conclusion depends on.** Checked directly:

- `ModelGateway.__init__`'s own docstring states the design intent outright: "resolving per call
  (not at construction) is what lets [the] workspace override apply with no signature changes"
  (`modelconfig.py:602-607`). `.llm()`/`.embedder()`/`.resolve()` all take `ws: str | None = None`
  as a call-time argument (`modelconfig.py:729-793`) and already resolve a per-workspace override
  through `GraphWorkspaceOverrides` on every call — this machinery exists today for an unrelated
  reason (FR-16/FR-17's per-workspace model override), but it means `ModelGateway` is *already*
  workspace-parametrized per call, not fixed at construction.
- `EmbeddingWorker.embed_chunk(self, ws: str, ...)` takes `ws` as a call argument
  (`embedding.py:230`), and its only cached state is `self._index_dim_cache: dict[tuple[str,
  str], int]` — keyed by `(ws, label)` specifically so one instance can safely serve more than one
  workspace (`embedding.py:120-127`, the comment names this explicitly).
- `IngestionPipeline.extract_chunk(self, ws: str, ...)` likewise takes `ws` per call
  (`ingestion.py:100-101`) and calls `self._models.llm("extraction", ws=ws)` — per-call, not
  baked in at construction (`ingestion.py:123`).
- `Repository`'s own layer is the same shape throughout — every method (`create_document`,
  `search_chunks`, `delete_document`, etc.) takes `ws` as its first argument and resolves the
  graph handle fresh per call (`db.workspace_graph(db, ws)`, `db.py:79-85`, no caching).

So the *only* thing actually pinning `ws` process-wide is `config.get_context()`'s one-line
hardcoded return (`config.py:276-284`) — exactly parallel to how `ctx.actor` is pinned, which is
exactly why §4.1's `produced_by` fix (an additive, optional per-call parameter, no core-pipeline
change) works for `actor`. An analogous additive parameter for `ws` (e.g. an optional
`target_ws: str | None = None` on `ingest_document`/`search_documents`, defaulting to `ctx.ws`
when omitted) would **not** require touching `ModelGateway`/`EmbeddingWorker`/`IngestionPipeline`
internals at all — they already accept `ws` as an argument. The "categorically larger change...
new development against falkor-chat's core ingestion/embedding architecture" framing does not
hold up against the source.

**This does not change the recommended decision — it changes why it's right.** There is a real,
better reason to still prefer the dedicated-process route that the plan doesn't currently name:
`get_context()` is the one auth/tenancy seam `falkor-chat/docs/SERVER.md`/`DESIGN.md` §1.1 locks
as the single point where "real auth replaces it without touching services/repo" — its entire
value is that no caller can pick which workspace it writes into. Adding a caller-supplied
`target_ws` parameter would reopen exactly that seam: any MCP/REST caller could then write into
*any* `ws:{id}` graph, not just the one it's "supposed" to use, with no authorization check
guarding which workspace a given caller may name — a cross-tenant write capability this codebase
has never had and that `get_context()`'s docstring frames as deliberately closed off. A dedicated
process sidesteps that risk entirely (isolation by deployment topology, not by an
unauthenticated parameter), at the cost of the extra infrastructure §4.3 already scopes. That's a
materially stronger justification than "the pipeline would need re-architecting" — recommend
`architect` swap the two before dispatch: keep the decision, replace the reasoning.

### Interaction check (the coordinator's specific ask) — confirmed, `architect`'s "infrastructure, not method" claim holds

Traced whether standing up a second, dedicated falkor-chat process changes anything the ML note
(still v2, unrevised) assumed. It does not, on every axis checked:

- **§4.1 `produced_by`** is a `Services`-level parameter, orthogonal to which process runs it;
  §4.3 already names the one real dependency correctly (the dedicated process must run a build
  that includes the `produced_by` patch, Stage 3 blocks on Stage 1) — nothing for the ML note to
  revise here.
- **§4.4 retrieval** — the embedding model resolution (`ModelGateway.embedder("embedding",
  ws=ctx.ws)`) reads the *same* default `config/models.json`/`FALKORCHAT_MODEL_CONFIG` overlay
  any falkor-chat deployment reads unless a workspace-specific override is set — so a second
  process, absent a divergent model-config override, resolves to the identical
  Qwen3-Embedding-0.6B model the ML note's method was designed against. The prefix template,
  top-K, and score-floor calibration are all applied client-side against whatever `score` a
  `search_documents` call returns (`services.py:1309-1354`, re-confirmed unchanged) — none of that
  math depends on which process answered the call, only on which model embedded the text, which
  is unaffected by the process split under the default (no-override) configuration this plan
  assumes throughout.
- **Stage 8's golden-set calibration run** naturally exercises the dedicated process (it's the
  only one that can reach `ws:agent-team`'s content at all), so no separate coordination step is
  needed there either.

One minor gap worth a one-line addition, not a blocker: §4.3/§3 Stage 3 doesn't explicitly state
that the dedicated process must use the **same, default** model-config overlay as the rest of the
deployment (no workspace-specific override pointing it at a different embedding model) — implicit
today, and correct by default, but cheap to make explicit so `devops` doesn't accidentally
diverge it when standing the process up.
