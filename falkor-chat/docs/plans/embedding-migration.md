# Embedding model migration & index rebuild — Implementation Plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — (M6+)

## 1. Goal & scope

Build a **reusable, on-demand mechanism** that migrates one workspace's stored `Message`/`Chunk`
embeddings to a new embedding model — recomputing every vector, rebuilding the workspace's vector
index at the new (possibly different) dimension, and keeping the workspace's explicit
per-workspace embedding-model override in lockstep with the data — with safe interrupt/resume, so
an urgent model swap (the immediate trigger: Qwen3-Embedding-0.6B is overrunning LM Studio's memory
budget) never corrupts or silently orphans a workspace's retrieval. This plan also designs the
*safety net* around every global-default change: an explicit per-workspace override recorded for
every existing workspace before the default can move (FR-1), and automatically at birth for every
new one (FR-2), using a seam that (per §2.3) already exists in the codebase.

Full functional requirements: `falkor-chat/docs/requirements/embedding-migration.md` (FR-1..FR-10,
read there — not repeated here in full). In scope: FR-1 through FR-10 as written there, at the
orchestration level for FR-3/FR-4/FR-10 (graph-level Cypher/DDL mechanics are `graph-dba`'s
companion note, §4). **Out of scope** (per the requirements doc, unchanged here): zero-downtime/
dual-serving migration, choosing the destination model, choosing which production workspace(s) to
migrate, and embedding retention/eviction.

**CPG:** considered, not relevant — `cpg_falkorchat` (built `07c252d`, `HEAD` one unrelated commit
ahead) is available for structural navigation, but every finding below was verified by reading the
live source (`modelconfig.py`, `repository.py`, `embedding.py`, the backfill scripts,
`conftest.py`) directly; the CPG added nothing a direct read didn't already settle faster for a
module this size.

## 2. Context & findings

### 2.1 The failure mode this closes

DESIGN.md §7.1 and SERVER.md §1.8 (FR-19) both document the same live-verified fact: FalkorDB
accepts a `SET n.embedding = vecf32([...])` write at any length — a vector whose length doesn't
match the vector index's declared dimension is **silently accepted, then invisible to
`db.idx.vector.queryNodes`** (drops out of ANN, no error anywhere in the chain). `EmbeddingWorker`
already guards the *live write path* against this (`embedding.py:147-214`, FR-19): before calling
the embedder at all, it compares the resolved model's declared `dim` against
`Repository.read_index_dimension(ws, label=...)` and raises `EmbeddingDimensionError` on mismatch,
no HTTP call made. That guard is the fail-safe an **unmigrated** workspace relies on after the
global default moves (acceptance criteria, last bullet) — this plan's mechanism must migrate a
workspace end-to-end without ever routing around that guard for ordinary traffic, and must restore
its cached state correctly once migration completes (§2.7).

### 2.2 Workspace creation has no code hook — it's an ops runbook

There is no `create_workspace()` call anywhere in `server/falkorchat/app.py` or `services.py`. A
workspace comes into existence purely operationally: an operator runs
`./scripts/bootstrap_schema.sh <wsId>` (DDL only — indexes, constraints, vector index at
`EMBEDDING_DIM`), then whichever `seed_*.sh` scripts apply. **`bootstrap_schema.sh` is
deliberately DDL-only** — `falkor-chat/AGENTS.md`'s own Key Scripts row is explicit: "DDL only, no
`MERGE`/`CREATE (n)`/`DELETE`" — which is exactly what makes bootstrapping a throwaway probe
workspace safe for `reference`'s data. **This constrains FR-2's design**: "built into workspace
bootstrap itself" cannot mean *inside* `bootstrap_schema.sh` without breaking that documented
DDL-only invariant (and the safety properties other tooling relies on it for). §3.2 designs the
non-invasive alternative.

### 2.3 The per-workspace override seam already exists — FR-1/FR-2/FR-5 need no new mechanism

Read in full: `server/falkorchat/modelconfig.py`, `repository.py:3231-3316`. K-042 Landing 2
already shipped exactly the "per-workspace embedding-model override, hard cap over the global
default" mechanism FR-1/FR-2/FR-5 ask for — this plan does **not** need to invent one:

- **Storage:** `WorkspaceConfig {workspaceConfigId:'default'}`, one MERGE-backed singleton per
  `ws:{id}` graph, property `embeddingModelOverride` (plus `agentModelOverride`/
  `guardModelOverride`/`responderModelOverride` — three sibling kinds this feature must not touch).
- **Write:** `Repository.write_model_overrides(ws, *, agent=, guard=, embedding=, responder=, at=,
  by=)` (`repository.py:3237`) — a single `MERGE ... SET` of all four properties at once.
  **Landmine, confirmed by reading the method's own docstring and Cypher:** passing `None` for a
  kind **clears** that override (`SET c.xModelOverride = $x` with `$x = None` nulls the property) —
  it is *not* "leave unchanged." Any caller that wants to touch only `embedding` **must first
  `read_model_overrides(ws)` and pass the other three kinds' current values through unchanged**,
  or it silently wipes any existing agent/guard/responder override for that workspace. This method
  has no production caller today (only `test_repository.py`) — this plan is its first real caller.
- **Read:** `Repository.read_model_overrides(ws)` (`repository.py:3287`) returns
  `{agentModel, guardModel, embeddingModel, responderModel}`; zero-row (never written) and
  one-row-all-`NULL` (written, this kind never set) both read back as `None` — "no override," one
  code path, not an error.
- **Resolution precedence:** `ModelGateway.resolve(kind, *, requested=, ws=, overrides=)`
  (`modelconfig.py:729-755`): workspace override → `requested` → per-kind default, **first-match-
  wins, workspace is a hard cap even over an explicit `requested=`**. Critically for §3.3's
  migration step: **the hard cap only triggers when `ws=` (or a pre-fetched `overrides=`) is
  passed** (`_workspace_override_ref`, `modelconfig.py:708-727`) — calling
  `gateway.embedder("embedding", requested=<targetRef>)` with **no `ws=`** never reads the
  workspace override at all, so `requested` wins outright. This is exactly the escape hatch the
  migration script needs (§3.3): the workspace's override still names the *old* model until FR-5's
  final step, so re-embedding must explicitly bypass the hard cap, not fight it.
- **Crosswalk:** the module's `kind="embedding"` maps 1:1 to property `embeddingModelOverride`
  (`_KIND_TO_OVERRIDE_KEY["embedding"] = "embeddingModel"`, `modelconfig.py:102-107`) — no
  crosswalk surprise for this kind (unlike `agent`/`step`, which are swapped).

**Conclusion:** FR-1 (snapshot), FR-2 (pin-at-birth) and FR-5 (override moves with the data) are
all just calls to `write_model_overrides`/`read_model_overrides` with the read-before-write
discipline above — no new `WorkspaceConfig` property, no new resolution logic.

### 2.4 The backfill-script convention — and where this feature must diverge from it

`scripts/backfill_thread_ids.sh` / `backfill_document_current.sh`: both are pure `bash` +
`redis-cli GRAPH.QUERY`, one or two idempotent `WHERE <property> IS NULL` Cypher statements,
`<workspaceId>...` positional args, a `PING` reachability preflight, and "re-running reports 0"
idempotency. This convention **cannot cover FR-3** as-is: re-embedding requires an HTTP call per
row to an embedding model behind `ModelGateway`/LM Studio — not expressible as a bare
`GRAPH.QUERY`. The codebase already has the right precedent for *this* shape of script, though:
`scripts/seed_eval_corpus.py` (wrapped by `scripts/seed_eval_corpus.sh`) is a Python script, run via
`server/.venv/bin/python`, that imports `falkorchat.modelconfig.ModelGateway`,
`falkorchat.repository.Repository`, and `falkorchat.db` directly, resolves the real configured
model via `ModelGateway.from_env()`, and is idempotent (`_has_embedding` skips already-embedded
messages). Its `.sh` wrapper does exactly the reachability/venv/config-file preflight the backfill
scripts do, then `exec`s the Python script with `$@`. **This plan's script follows
`seed_eval_corpus.{py,sh}`'s shape, not the bare-Cypher backfill shape** — same `<workspaceId>`-arg
and idempotent-resume *conventions* FR-10 asks for, adapted to a script that must call out to a
model. Also confirmed absent, per the requirements doc: no existing script re-embeds stored data or
rebuilds a vector index outside test-only helpers.

### 2.5 The only existing vector-index-rebuild precedent

`server/tests/conftest.py::rebuild_vector_indexes` (`ws`, `dim`): two `DROP VECTOR INDEX FOR
(n:{label}) ON (n.embedding)` + `CREATE VECTOR INDEX FOR (n:{label}) ON (n.embedding) OPTIONS
{dimension:{dim}, similarityFunction:'cosine'}` statements, for `Message` and `Chunk`. Cheap, no
subprocess, no full schema/constraint rebuild. This is FR-4's mechanism, generalized from a
hardcoded `ws:test`/dim-4 test fixture to an arbitrary `(ws, dim)` pair — the exact interface
handed to `graph-dba` in §4.1, **now confirmed and generalized by `graph-dba`'s companion note**
(`docs/plans/embedding-migration-graph.md`, landed since this plan's first draft — §4 below quotes
its answers directly rather than leaving them as an open dependency).

### 2.6 This deployment's tenancy: one server process serves exactly one workspace

`server/falkorchat/config.py:276-284`, `get_context()`'s own docstring: **"M1 resolves every call
to one hardcoded tenant"** — `CallContext(ws=WS_ID, actor=USER_ID)`, `WS_ID` read once from
`FALKORCHAT_WS_ID` at process start (`config.py:16`), and "both front doors (REST and MCP)
attribute calls through here." There is no code path in this build by which a running
`falkorchat.app` process serves a request against any workspace other than its own fixed `WS_ID` —
confirmed by grepping every `WS_ID` reference in `server/falkorchat/*.py`: `get_context()` is the
only place it's read for live request attribution, and every background loop (the responder, the
executor's `wait`/`human` sweep, the ingestion pipeline) is constructed from that same one
process-wide context. **Consequence for §3.4's step 0:** in this deployment, a workspace is
"served" by exactly one specific process (whichever one was started with that workspace's
`FALKORCHAT_WS_ID` — `start_server.sh` defaults to `acme`, `start_demo.sh` pins `demo`,
`start_agent_team.sh` pins `agent-team`, each its own separate process/port) — stopping *that one*
process provably cannot affect any other workspace's own separately-run process, because no process
ever reaches across workspaces. This is a narrower, more precise fact than "stop the whole shared
server, scope to other workspaces unclear" — there is no shared server in this architecture, only
possibly-several independently-run single-workspace ones. `ws:eval` (§3.5) has no such process
running against it in normal operation at all (§2.8) — the practical case this plan actually
migrates. This is an M1-era fact, not a permanent one: real multi-tenant request routing (K-016) is
still open, and whoever builds it must revisit this section.

### 2.7 A process-lifetime cache the migration must account for

`EmbeddingWorker._index_dim_cache: dict[tuple[ws, label], int]` (`embedding.py:120-145`) caches
`read_index_dimension`'s answer **for the process lifetime**, explicitly because "the dimension
provably cannot change in place... only drop+recreate, an out-of-band admin action, changes it" —
this feature is precisely that out-of-band admin action. **If the server process serving this
workspace is not restarted after the index rebuild, its live `EmbeddingWorker` keeps comparing new
writes against the stale pre-migration dimension** and will raise `EmbeddingDimensionError` on
every legitimate post-migration write — the exact opposite of what the migration bought. §3.4 makes
a process restart a required, explicit step, not an incidental side effect of "the outage."

### 2.8 The FR-6 validation hook is a static, corpus-file comparison — not a live graph read

`model-bench/packs/embedder-graphrag-retrieval/pack.json`: `data.items`/`data.corpus` are files
(`queries.jsonl`, `corpus.jsonl`) copied **one-way** from `falkor-chat/server/tests/eval/
golden_retrieval.jsonl` and `scripts/seed_eval_corpus.py`'s corpus constant
(`PROVENANCE.md`) by `model-bench/scripts/refresh_golden.py`. The pack's `environment.requires` is
`["lmstudio-embeddings"]` — it computes embeddings live against LM Studio for whichever model is
named on the CLI, entirely independent of any FalkorDB graph. The already-run report
(`model-bench/reports/embedder-graphrag-retrieval-20260918-01.md`) is the existing baseline
(Qwen3-Embedding-0.6B vs. BM25); the CLI shape (`model-bench/README.md`) is
`./run.sh run --pack embedder-graphrag-retrieval --model <key>` (one model × one pack, against live
LM Studio) then `./run.sh compare --pack embedder-graphrag-retrieval --models <a>,<b>` (renders a
markdown report to `reports/` + stdout). **Consequence for sequencing (§3.5):** because the pack
embeds the *same corpus text* the live `ws:eval` graph is seeded from, and embeddings are
deterministic given `(model, text, params)`, a model-bench comparison run with the candidate model
is numerically equivalent evidence to embedding that same text through the live graph — the pack
does not need `ws:eval`'s live graph to already be migrated, and this plan's migration script does
not need to invoke model-bench programmatically. FR-6's gate is procedural, not a code dependency.

### 2.9 Message/Chunk write surface relevant to the re-embed step

`Repository.set_embedding(ws, *, msg_id, embedding, expected_dim=None)` and
`.set_chunk_embedding(ws, *, chunk_id, embedding, expected_dim=None)` (`repository.py:773`,
`:1202`) both validate `len(embedding) == expected_dim` **client-side only** — neither checks the
live vector index's dimension (that check is `EmbeddingWorker`'s FR-19 guard, one layer up, never
touched by these methods directly). Message text is `Message.text`/`Message.msgId`; Chunk text is
`Chunk.text`/`Chunk.chunkId` (`docs/QUERIES.md` §14.1). `OpenAICompatibleEmbedder.embed(text)`
(`embedding.py:71`) embeds one string per HTTP call — there is no batch-embed API in this codebase
today, so "batch size" in FR-10's resume design (§3.3) means *rows per commit/progress checkpoint*,
not a batched embedding HTTP call.

## 3. Design & rationale

### 3.1 Shape: one Python migration tool, two commands, thin `.sh` wrappers

**Decision:** a new Python module, `scripts/embedding_migration.py`, importing `falkorchat`
directly (same posture as `seed_eval_corpus.py`), exposing two subcommands via a single
`argparse` entry point:

- `pin` — FR-1/FR-2's snapshot/pin-at-birth operation (one workspace, idempotent, no model call).
- `migrate` — FR-3/FR-4/FR-5/FR-8/FR-10's re-embed + index-rebuild + verify operation (one
  workspace, one target model ref, idempotent-resume).

Two thin wrapper scripts, `scripts/pin_workspace_embedding_model.sh <wsId> [...]` and
`scripts/migrate_embeddings.sh <wsId> <targetModelRef> [options]`, mirror
`seed_eval_corpus.sh`'s exact preflight (FalkorDB `PING`, venv exists, `FALKORCHAT_OPENCODE_CONFIG`
file exists) then `exec` the Python module with `$@` — matching this repo's `<workspaceId>...`
positional-arg and env-var (`FALKORDB_HOST`/`FALKORDB_PORT`) conventions from every other
`scripts/*.sh`.

**Rejected alternative:** extend the bare-Cypher `backfill_*.sh` shape. Rejected because FR-3
requires an HTTP call per row through `ModelGateway` — not expressible as `redis-cli GRAPH.QUERY`
— and reimplementing `{env:}`/`{file:}` substitution, `/v1` normalization, or role/fallback-chain
resolution in bash to read `config/models.json` correctly would create a second, silently-drifting
copy of `modelconfig.py`'s logic. `seed_eval_corpus.py` already proves the Python-script-importing-
`falkorchat` shape is accepted practice in this `scripts/` directory for exactly this reason.

**Rejected alternative:** a REST admin endpoint (`POST /workspaces/{id}/migrate-embeddings`) on the
running server. Rejected for this feature: FR-7 already accepts a full outage, so there is no
"keep the server up while migrating" requirement an HTTP endpoint would buy; a long-running,
resumable, operator-invoked batch job is a better fit for an offline script than for a request/
response HTTP handler (no natural place to report multi-minute progress, no auth/tenancy story
yet — K-016 is still open). Reversible if a future need for programmatic/remote triggering
emerges — nothing here forecloses adding a route that shells out to this same script later.

### 3.2 FR-1/FR-2: one idempotent "pin" operation — but *which* enforcement mechanism is a real fork, not a rounding error

**The `pin` operation itself is settled and needed either way:** "if `read_model_overrides(ws)
['embeddingModel']` is already set, no-op; if unset, write it to the current global default
(`Overlay.default_for('embedding')`, read via `ModelGateway.from_env()`, never hardcoded) —
preserving the other three override kinds exactly as read back (§2.3's landmine)." FR-1 runs it as
a one-time sweep across every existing workspace, once, before the global default is ever changed.
This part of §5 (steps 1 and 3) is unaffected by everything below and can be built now.

**What is *not* settled — revised after review:** the requirements doc's decision log states,
dated and marked "Confirmed exactly as read back": *"Rule (2) is to be built into workspace
bootstrap itself (**not left as a manual convention**)."* This plan's first draft designed FR-2 as
a second script an operator must remember to run after `bootstrap_schema.sh`, documented in
`AGENTS.md` — which the review correctly identified as precisely the manual convention that
decision-log line rejects, not a design that satisfies it. Below are the two real options, at equal
concreteness, with **§5 step 2 explicitly gated on picking one** — this plan does not get to settle
FR-2's enforcement mechanism by building the documentation-only version and hoping it reads as
close enough.

**Option A — documentation-only convention** (this plan's original design): `./scripts/bootstrap_schema.sh <wsId> && ./scripts/pin_workspace_embedding_model.sh <wsId>`,
documented as the recipe for a new *real* workspace in `AGENTS.md`'s Key Scripts table (mirroring
how `seed_demo.sh` already documents "run after `bootstrap_schema.sh`"). Zero code touched outside
the new `pin` script itself; zero regression risk to any existing script. Its cost is exactly what
the review named: an operator can still run `bootstrap_schema.sh` alone and skip the pin step —
this is the manual convention the decision log rejects, verbatim.

**Option B — `create_workspace.sh` replaces `bootstrap_schema.sh` as the canonical entry point for a real workspace.**
A new wrapper, `./scripts/create_workspace.sh <wsId> [<wsId> ...]` (forwarding `EMBEDDING_DIM` and
any other `bootstrap_schema.sh` args unchanged): `bootstrap_schema.sh "$@" && ` then, for each given
`wsId`, the `pin` operation — in-process (`embedding_migration.pin(ws)`) if the caller is already
Python, or via `pin_workspace_embedding_model.sh` if bash-to-bash. This makes FR-2 a structural
guarantee for every call site that adopts it, not a step to remember. Concretely, at the same level
of detail as Option A:

- **Call sites needing a logic change** (every place that invokes `bootstrap_schema.sh` directly
  for a *real*, live-served workspace): `scripts/start_server.sh` (step 3, `ws:acme` by default),
  `scripts/start_demo.sh:166` (`ws:demo`), `scripts/start_agent_team.sh:205` (`ws:agent-team`) — each
  swaps its direct `bootstrap_schema.sh "$wsId"` call for `create_workspace.sh "$wsId"`.
  `scripts/seed_eval_corpus.py`'s `_BOOTSTRAP`-driven reseed (§2.4 — already the recipe used
  whenever `ws:eval`'s embedding model/dim changes) can call `embedding_migration.pin(EVAL_WS)`
  directly in-process instead of shelling out, since it already imports `falkorchat` and already
  resolves `ModelGateway.from_env()` for its own embed pass.
- **Call site deliberately left unchanged:** `scripts/test_queries.sh:103`'s `ws:test` bootstrap.
  §3.2's own carve-out already holds regardless of which option wins — a throwaway/offline-suite
  workspace has no need for FR-1/FR-2, and the offline suite must not gain a new hard dependency on
  `FALKORCHAT_OPENCODE_CONFIG` parsing successfully (today it has none).
  `scripts/seed_nlq_eval_corpus.sh`'s documented direct `bootstrap_schema.sh nlq-eval` call is the
  same ops-decision-not-design-decision already flagged in §7 for `ws:demo`/`ws:agent-team`'s FR-1
  sweep membership — not resolved further here.
- **Documentation-only updates, no logic change** (six files): `seed_agent_team.sh`,
  `seed_catalog.sh`, `seed_workflows.sh`, `seed_demo.sh`, `seed_salesperson.sh` each state
  "run after `bootstrap_schema.sh`" as a precondition comment (they never call it themselves) —
  each gets one clause added, "(or `create_workspace.sh`, which also pins the embedding-model
  override, FR-2)". `AGENTS.md`'s own `bootstrap_schema.sh` Key Scripts row gets the same clause.
- **The one concrete regression risk found, and its mitigation:** `start_server.sh` explicitly
  documents `FALKORCHAT_ENABLE_AGENT=0` as a supported mode — "serve the UI/REST without the AI
  loop" — in which `FALKORCHAT_OPENCODE_CONFIG` is not required at all today (nothing downstream
  reads it). If `create_workspace.sh`'s pin step unconditionally called `ModelGateway.from_env()`
  and aborted on a parse failure, a `FALKORCHAT_ENABLE_AGENT=0` run with no valid opencode.json —
  legal and documented today — would newly hard-fail at the bootstrap step. **Mitigation:** `pin`
  (both as a library function and as `create_workspace.sh`'s embedded call) treats a
  `ModelConfigError` from `ModelGateway.from_env()` as **best-effort, not fatal** — print a WARNING
  ("could not pin the embedding-model override for `<wsId>`: `<reason>` — this workspace is
  uncovered by FR-1/FR-2's safety net until `pin_workspace_embedding_model.sh <wsId>` is re-run once
  a valid model config is available") and let workspace creation proceed. This preserves today's
  documented no-config no-AI mode exactly, while giving every call site that *does* have a valid
  config (the overwhelmingly common case — `start_server.sh`'s own default already points at
  `$HOME/.config/opencode/opencode.json`, and `start_agent_team.sh` hard-requires `ENABLE_AGENT=1`
  already) FR-2's guarantee automatically, no operator action required. `start_demo.sh` and
  `start_agent_team.sh` carry no equivalent no-config escape hatch (`start_agent_team.sh:135-137`
  hard-errors if `ENABLE_AGENT` isn't `1`) — no incremental regression risk at those two call sites.

**Recommendation: Option B, with the best-effort mitigation above.** The decision log's line is
dated, explicit, and already confirmed with the stakeholder — "not left as a manual convention" is
not close-enough-in-spirit to Option A's "left as a documented manual convention, enforced by
nobody." Option B's blast radius is now bounded and named (three call sites' logic, six comments,
one identified regression risk with a concrete fix) rather than the open-ended "larger, repo-wide
change" this plan's first draft used to justify deferring it. This is not, in the end, a genuine
toss-up this plan can't resolve — but the choice is the stakeholder's confirmed requirement to make,
not this plan's to default past, so **§5 step 2 is gated on it explicitly** rather than proceeding
as written.

### 3.3 FR-3/FR-10: re-embed with an explicit, hard-cap-bypassing target model and a migration-owned marker property

**The hard-cap trap (why `ws=` must not be passed during re-embed):** the workspace's
`embeddingModelOverride` still names the *old* model until FR-5's final step (§3.1's ordering
deliberately does FR-5 last). If the re-embed step resolved the embedder via
`gateway.embedder("embedding", ws=ws, requested=targetRef)`, §2.3's hard-cap precedence would make
the *old* model win over the explicit target — the migration would silently re-embed every row
with the model it's trying to move away from. The fix, derived directly from reading
`_workspace_override_ref` (`modelconfig.py:708-727`): call
`gateway.embedder("embedding", requested=targetRef)` with **no `ws=` and no `overrides=`** — this
skips the workspace-override read entirely, so `requested` (the explicit target) wins outright.
This is the single most important correctness detail in this plan; get it wrong and the migration
looks like it succeeded (writes complete, no errors) while silently doing nothing.

**Precondition:** `config/models.json`'s `models.<targetRef>.dim` must be declared before running
`migrate` (the same convention `lmstudio/text-embedding-qwen3-embedding-0.6b` already uses — a
one-line config edit, DESIGN §12's existing swap path). The script resolves
`gateway.resolve("embedding", requested=targetRef).primary.dim` and refuses to start if it comes
back `None` — falling back to `config.EMBEDDING_DIM` (as `EmbeddingWorker` does for the *current*
model) would silently use the *old* workspace's dimension for a *new* model, defeating the entire
point of a dimension-agnostic mechanism.

**Deliberate, by-construction bypass of the FR-19 guard (review minor, worth stating explicitly):**
the re-embed write path (§4 item 2) never routes through `EmbeddingWorker._resolve_and_embed`
(embedding.py:147-214) — it cannot. That guard compares the *resolved model's* declared dimension
against the *live index's current* dimension and raises before any HTTP call on a mismatch; by
construction, for every row this migration touches, the old vector index is still at the *old*
dimension while the target model resolves to the *new* one (§3.4 rebuilds the index only after
every row is done) — routing through the guard would make it reject every single row. This is not
an oversight or a hole in FR-19's coverage: the guard's job is to protect the *ordinary* write path
(new messages/chunks arriving through normal traffic) from ever writing a mismatched vector, and
this plan's own §3.4 step 0 (below) is what keeps ordinary traffic away from the workspace for the
duration — the migration path and the guarded hot path are mutually exclusive by design, never
simultaneously active against the same workspace.

**Idempotent resume marker — new property, migration-script-owned only:** FR-8(a)'s count check
("every intended row now carries a new-model embedding, with none missed") needs a durable signal
of *which model* produced a stored vector — dimension alone can't disambiguate two different
models that happen to share a dimension (the shortlist already has two: `granite-embedding-278m-
multilingual` and `multilingual-e5-base`, both 768-dim). **Decision:** add `Message.embeddingModel`
/ `Chunk.embeddingModel` (a `"<provider>/<model-id>"` ref string, same shape as the override), set
by this migration script's own write query alongside `embedding` — never by
`EmbeddingWorker`/`repository.set_embedding`/`set_chunk_embedding` (the ordinary hot write path is
**not** touched by this plan). Resume/verify query: `WHERE coalesce(n.embeddingModel, '') <>
$targetRef` — true for a row never touched by any migration (property absent), a row migrated to a
*different* target in an earlier run, and a row from an interrupted run of *this* migration alike;
re-running always converges. A message/chunk that somehow never got embedded at all (`embedding
IS NULL`) is also caught by this same clause and gets embedded as a side effect — a reasonable,
mildly self-healing behavior worth calling out rather than leaving implicit, not a design goal in
itself.

**Rejected alternative:** stamp `embeddingModel` on the ordinary hot path too (`EmbeddingWorker`,
`set_embedding`/`set_chunk_embedding`), so every row is always self-describing. Rejected on a
blast-radius trade-off: it would change the signature/behavior of methods every existing message/
chunk write already depends on, for a benefit (avoiding redundant re-embedding of already-correct
rows on some *future* migration) that is purely a performance nicety — migrations are rare,
deliberate, operator-invoked events, not a hot path. The cost of the rejected alternative (touching
widely-exercised production code) outweighs the one-time cost of the chosen alternative (a future
migration harmlessly re-embeds rows written by ordinary traffic since the last migration, because
they carry no marker at all). Reversible either way; not a fork requiring escalation.

**Batching — confirmed by `graph-dba`'s companion note, keyset not `SKIP`:** read up to
`--batch-size` (default 50) unmigrated rows per label via `WHERE id > $lastId AND
coalesce(embeddingModel,'') <> $target ORDER BY id ASC LIMIT $batchSize` (§4 item 1) —
`GRAPH.PROFILE`-verified on a 5000-row scratch label to cost `O(rows)` total across a full
migration, against a `SKIP`-based page's `O(rows²/batchSize)` (every page re-scans and re-sorts the
entire label). `$lastId` restarts at `''` on every fresh `migrate` invocation (including a
post-crash resume) rather than persisting a cross-process cursor — cheap (one extra scan-only pass
over already-migrated rows) and correct regardless, since the `coalesce(...) <> $target` clause,
not `$lastId`, is the actual correctness gate. Embed each row individually (`embedder.embed(text)`
— no batch-embed API exists, §2.9), write each individually (§4 item 2), print progress per batch
(row counts, running total) — matching the backfill scripts' progress-reporting style. Sequential,
not concurrent: correctness and simplicity first; concurrent HTTP calls to LM Studio are a valid
future optimization, explicitly out of scope for this first landing (§7).

**Vanished row mid-migration (review minor, judged rather than left as an open interface question):**
§4 item 2's write query reports a no-op via `properties_set == 0` when the target `msgId`/`chunkId`
no longer matches anything (`graph-dba`'s confirmed detection mechanism). **Decision: skip and log,
never abort the batch or retry.** Neither `Message` nor `Chunk` has a documented delete path in this
codebase today (both are effectively append-only — DESIGN.md never describes a deletion query for
either label), so this is an extremely low-probability event in practice, and FR-10's own posture
("only rows not yet migrated are (re-)processed... never a requirement to start over from zero")
already implies the migration should be resilient to exactly this kind of single-row surprise rather
than treating it as fatal — the same graceful posture the existing backfill scripts take toward
their own idempotent no-ops (`dupMsg` is reported, never raised). A skipped row (never marked with `embeddingModel`) would still show up as "unmigrated" in §4 item
3's count check if it turns out to have still existed at read time and only vanished by write time
— which is exactly correct: it genuinely wasn't migrated. If it was a true deletion, the count check
never sees it again either way (the read-unmigrated-batch query no longer returns a row that no
longer exists), so the two behave consistently without the script needing to distinguish them.

### 3.4 FR-4/FR-7: rebuild the index only after every row is already correct — and stop traffic before step 1, not just after

**Decision on rebuild ordering (unchanged, now live-verified rather than inferred):** re-embed
**all** rows for both labels to completion *before* touching the vector index at all; drop +
recreate the index (§2.5's `rebuild_vector_indexes` pattern, generalized to `(ws, dim)`) only once
the FR-8(a) count check reports zero unmigrated rows for that label. This plan's first draft
inferred this was safe from §2.1's documented engine behavior (a wrong-dimension write silently
drops out of ANN, never an error); `graph-dba`'s companion note has since **live-verified it
directly** — a throwaway probe graph was rewritten one row at a time from dim 4 to dim 8 while the
dim-4 index stayed attached throughout, and every single intermediate step produced a correct,
narrowing ANN result set, no error, no corruption, no internal consistency check tripped
(`embedding-migration-graph.md`, item 6). **§3.4's ordering needs no amendment.**

**Fix for review Blocker 2 — nothing previously created the outage FR-7 assumes.** The first draft
designed the *post*-migration restart (step 5 below) in detail but had no step preceding re-embed
that actually stops traffic from reaching the target workspace. That gap is real and specific: a
live write landing between step 2's count check (0 unmigrated) and step 3's index rebuild is not
merely "unmigrated" — it reproduces §2.1's exact silent-ANN-drop failure mode this whole feature
exists to prevent (the ordinary write path embeds at the *old* model into an index that, by the
time the write lands, has already been rebuilt at the *new* dimension — no error, and the row is
unmarked, so nothing schedules its repair). A write landing *before* the count check is at least
caught (§4 item 3 would report it as unmigrated, aborting before the index touches); only this one
window is unguarded. Fixed with an explicit **step 0**, using §2.6's now-confirmed fact that in this
deployment a workspace is served by exactly one specific process:

0. **Stop traffic.** Stop the one `falkorchat.app` process whose `FALKORCHAT_WS_ID` equals the
   target workspace (§2.6) — the only process that could originate a live write against it in this
   architecture. `migrate`'s CLI requires an explicit `--i-have-stopped-traffic` confirmation flag
   (or an interactive y/n prompt when run without it) before step 1 begins — this cannot be
   verified programmatically (the script has no way to confirm some other process isn't running),
   so it is a named, confirmed precondition rather than a silently-assumed one, symmetric with step
   5's explicit restart instruction below. For `ws:eval` specifically (§3.5, §2.6) no such process
   exists in normal operation, so this step is a no-op confirmation, not an actual outage.
1. Re-embed all `Message` rows to completion (§3.3), then all `Chunk` rows to completion.
2. FR-8(a) count check for both labels (§4 item 3) — zero unmigrated, or abort before touching the
   index.
3. Drop + recreate `Message.embedding` and `Chunk.embedding` vector indexes at the new dimension
   (§4 item 5 — `graph-dba`-confirmed exact DDL). **Guard the drop:** call
   `Repository.read_index_dimension(ws, label=label) is not None` before `DROP VECTOR INDEX` and
   skip the drop when it's already `None` — `graph-dba` live-verified that `DROP VECTOR INDEX`
   hard-errors (`"no such index"`) when no vector index exists at all, which a naive unconditional
   drop-then-create would hit on any resume from a crash that landed exactly between a prior run's
   drop and create. This is the one FR-10 resumability hazard `graph-dba`'s verification surfaced
   that this plan's first draft hadn't accounted for.
4. FR-5: update `WorkspaceConfig.embeddingModelOverride` to `targetRef` (read-then-write, §2.3).
5. **Restart the server process** serving this workspace (§2.7 — required, not incidental: clears
   `EmbeddingWorker._index_dim_cache`'s stale pre-migration dimension for this `(ws, label)`, and
   lets ordinary post-migration writes resolve through the now-updated override with no hard-cap
   surprise).
6. FR-8(b): retrieval sanity check, then resume traffic (start the process stopped in step 0).

### 3.5 FR-6: the model-bench pack is the validation gate, run before this script ever touches a live graph

Per §2.8's finding (the pack is a static-corpus comparison, not a live-graph read), this plan's
migration script does **not** invoke model-bench programmatically — FR-6 is a **procedural** gate
an operator satisfies manually, in this order, before running `migrate` against any workspace:

1. `./run.sh run --pack embedder-graphrag-retrieval --model <candidate-key>` (against live LM
   Studio with the candidate model loaded) — `model-bench`, already proven against the current
   Qwen3-Embedding-0.6B baseline (`reports/embedder-graphrag-retrieval-20260918-01.md`).
2. `./run.sh compare --pack embedder-graphrag-retrieval --models <baseline-key>,<candidate-key>` —
   renders the paired comparison; review the verdict (not "silently assumed to pass," per the
   acceptance criteria).
3. Only once that comparison is reviewed and accepted: run `migrate_embeddings.sh eval
   <candidateRef>` against the live `ws:eval` graph (the confirmed first migration target,
   decision log) — this exercises FR-3/4/5/8/10 against real (if disposable) graph data, distinct
   from step 1's static-fixture run, and is what actually proves the *mechanism*, not just the
   model.
4. FR-8(b) for `ws:eval` specifically: because `ws:eval` is a data corpus for the eval harness, not
   a live-chat-served workspace (`seed_eval_corpus.py`'s own docstring — no `AgentResponder` loop
   runs against it in normal operation), the "real questions run through the assistant" check is
   satisfied by re-running a handful of `docs/QUERIES.md` §6's hybrid-retrieval reads directly
   against the now-migrated live graph and confirming the returned chunks/messages are still the
   expected ones for those golden queries — cheaper and more direct than standing up a live
   responder loop against a workspace that was never meant to serve one.
5. Only after 1-4: consider migrating a real/production workspace (deliberately deferred, decision
   log — not designed further here).

## 4. Handoff to `graph-dba` — answered, companion note landed

`falkor-chat/docs/plans/embedding-migration-graph.md` answers every item below, each live-verified
against disposable scratch graphs on the pinned build (not merely transcribed DDL) — quoted here
once, at the exact shape `scripts/embedding_migration.py` implements, so this plan stays
self-contained; full verification detail/rationale is cited, not repeated.

1. **Read-unmigrated-batch, confirmed keyset, not `SKIP`.** One per label:
   ```cypher
   CYPHER lastId=$lastId target=$targetRef b=$batchSize
   MATCH (n:Message)
   WHERE n.msgId > $lastId AND coalesce(n.embeddingModel, '') <> $target
   RETURN n.msgId AS id, n.text AS text ORDER BY n.msgId ASC LIMIT $b
   ```
   (swap `Chunk`/`chunkId`). `GRAPH.PROFILE`-verified on a 5000-row scratch label: the keyset form
   is index-anchored on the existing `msgId`/`chunkId` range index and costs `O(rows)` total across
   a migration; `SKIP` re-scans and re-sorts the whole label on every page, `O(rows²/batchSize)` —
   invisible at `ws:eval`'s scale, real at `docs/test-reports/capacity-report.md`'s larger ones.
   `$lastId` restarts at `''` on every fresh invocation, including a post-crash resume (§3.3).
   **Also surfaced, not this plan's concern to design around further:** `<`/`<=` against an
   *indexed* string property silently returns the whole label rather than filtering on this build —
   a genuine engine bug, filed to `claude/graph-dba/falkordb-quirks.md`, irrelevant here only
   because this query uses `>` exclusively, which is verified safe — worth remembering if a future
   revision ever wants a descending cursor.
2. **Write-one-embedding.** One per label:
   ```cypher
   MATCH (n:Message {msgId: $id})
   SET n.embedding = vecf32($embedding), n.embeddingModel = $targetRef
   ```
   (swap `Chunk {chunkId: $id}`). No-op (vanished row) detection: `properties_set == 0` on the
   query result — a real match always sets exactly two properties. Client-side dimension validation
   happens in the calling script before this query runs, same division of responsibility as
   `Repository.set_embedding`/`set_chunk_embedding` (§2.9) — this plan does **not** route through
   those two methods (they don't set `embeddingModel` and validate against the wrong dimension by
   default); two new, small repository methods or inline queries carry this shape instead.
3. **Count-unmigrated, confirmed two-statement shape, verified clean on an empty graph.** One per
   label: `MATCH (n:Message) WHERE coalesce(n.embeddingModel,'') <> $target RETURN count(n) AS
   unmigrated` plus `MATCH (n:Message) RETURN count(n) AS total` (FalkorDB has no single-query
   shape for both without a `CASE`-based conditional aggregate that buys nothing here). Report
   `migrated = total - unmigrated`, matching the acceptance criteria's exact wording.
4. **`embeddingModel` — no index, confirmed; no `bootstrap_schema.sh` change, confirmed.** A plain
   scalar property needs no DDL on this build at all — written on first `SET`, read correctly
   whether present/absent/`null`, no prior `CREATE INDEX` required. Same shape as
   `Message.threadId` (DESIGN §7.1) for the same reason: read only by this script's own occasional
   scans, never a hot-path filter.
5. **Vector-index rebuild, confirmed safe over a uniformly-migrated graph — plus one real
   resumability hazard.** The pair (label/dimension f-string-interpolated — DDL identifiers and
   `OPTIONS` keys aren't parameterizable on this build, same as `conftest.py`):
   ```cypher
   DROP VECTOR INDEX FOR (n:{label}) ON (n.embedding)
   ```
   ```cypher
   CREATE VECTOR INDEX FOR (n:{label}) ON (n.embedding)
   OPTIONS {dimension: {newDim}, similarityFunction: 'cosine'}
   ```
   Live-verified clean over a graph whose nodes already all carry uniformly-`newDim` vectors (the
   state §3.4 guarantees before this step runs); no other index (the `msgId` range index, the
   `Message.text` full-text index) is touched by dropping/recreating this one. **The hazard:**
   `DROP VECTOR INDEX` succeeds on a label with zero vectors, but **hard-errors** (`"no such
   index"`) when no vector index exists on that label/property *at all* — meaning a crash between a
   prior run's `DROP` and `CREATE` makes a naive unconditional retry of "drop, then create" fail
   loudly on resume. §3.4 step 3 folds in the recommended guard (`read_index_dimension(...) is not
   None`, an existing, already-tested method) before calling `DROP`.
6. **Ordering safety — SAFE as designed, live-verified, no amendment.** §3.4 quotes the result
   directly; full verification method (a 5-row, dim-4→dim-8 incremental rewrite under a live
   dim-4 index, watched via ANN query result counts at every step) is in the companion note, not
   repeated here.

## 5. Step-by-step implementation

Sequenced so the tree stays buildable and each step is independently reviewable. `graph-dba`'s
companion note has landed (§4) — nothing here is blocked on it any longer. **Step 2 is the one step
gated on a decision this plan cannot make alone** (§3.2's Option A/B fork) — everything else is
unblocked regardless of which option the stakeholder picks.

1. **`scripts/embedding_migration.py` — `pin` subcommand (FR-1/FR-2's mechanism), no open
   dependency.**
   - `argparse` entry with subcommands `pin` and `migrate` (this step: `pin` only).
   - `pin(ws: str) -> dict`: `gateway = ModelGateway.from_env()`; `current =
     repo.read_model_overrides(ws)`; if `current["embeddingModel"]` is already set, return it
     unchanged (idempotent no-op, printed as such). Otherwise `default_ref =
     gateway.resolve("embedding").primary.ref` (the current global default, no `ws=`/`requested=`)
     and `repo.write_model_overrides(ws, agent=current["agentModel"], guard=current["guardModel"],
     embedding=default_ref, responder=current["responderModel"], at=<now, ms>, by="embedding_migration.pin")`
     — the read-before-write discipline from §2.3, non-negotiable. Catches `ModelConfigError` from
     `ModelGateway.from_env()` and degrades to a printed WARNING rather than raising (§3.2's Option
     B mitigation) — this makes `pin()` safe to call unconditionally from *either* option, so this
     step's code is identical regardless of §3.2's outcome.
   - `scripts/pin_workspace_embedding_model.sh <wsId> [<wsId> ...]` — thin wrapper, `seed_eval_corpus.sh`'s
     preflight shape (PING, venv check, `FALKORCHAT_OPENCODE_CONFIG` file check), loops over every
     given `wsId` the same way `backfill_thread_ids.sh` does, `exec`s the Python module per id (or
     passes the whole list — implementer's call, either is a small, reviewable detail).
   - Done when: run against a workspace with no override → override gets set to the current
     `defaults.embedding`; run again → no-op, unchanged; run against a workspace that already has
     an `agentModelOverride` set → that value survives untouched after a `pin` call; run with no
     valid `FALKORCHAT_OPENCODE_CONFIG` → prints the WARNING, does not raise.
2. **FR-2's enforcement mechanism — gated on the stakeholder's pick between §3.2's Option A and
   Option B (this plan recommends B).** Do not build both; do not default to A by proceeding as if
   documentation were sufficient. Once picked:
   - **If A:** update `falkor-chat/AGENTS.md`'s Key Scripts table — add the
     `pin_workspace_embedding_model.sh` row, and one sentence on the `bootstrap_schema.sh` row
     naming it as the required next step for a new *real* workspace.
   - **If B (recommended):** add `scripts/create_workspace.sh <wsId> [<wsId> ...]` (§3.2 — wraps
     `bootstrap_schema.sh` + the `pin` operation per given id); update the three named call sites
     (`start_server.sh` step 3, `start_demo.sh:166`, `start_agent_team.sh:205`) to call it instead
     of `bootstrap_schema.sh` directly; update `seed_eval_corpus.py`'s reseed path to call
     `embedding_migration.pin(EVAL_WS)` in-process; leave `test_queries.sh:103`'s `ws:test`
     bootstrap untouched (§3.2's carve-out); add the "(or `create_workspace.sh`...)" clause to the
     six seed scripts' precondition comments and to `AGENTS.md`'s `bootstrap_schema.sh` row.
   - Done when (either option): a fresh `FALKORCHAT_WS_ID=smoketest ./scripts/start_server.sh` (or
     the equivalent for whichever option won) leaves `ws:smoketest` with an explicit
     `embeddingModelOverride` set, with zero extra operator action beyond the command already
     documented for standing up a new workspace today.
3. **FR-1's one-time sweep** — not new code: run `pin_workspace_embedding_model.sh` against every
   existing real workspace (today: `acme`, plus `eval`/`demo`/`agent-team` if they should carry an
   explicit pin too — an operator/ops decision, not a design one, §7) once, before the global
   default in `config/models.json` is ever edited. Record this as a runbook step (§3.2), not a
   script of its own.
4. **`scripts/embedding_migration.py` — `migrate` subcommand (FR-3/FR-4/FR-5/FR-8/FR-10), Cypher
   per §4's confirmed shapes.**
   - `migrate(ws: str, target_ref: str, *, batch_size: int = 50, traffic_stopped: bool = False) ->
     MigrationReport`:
     a. **§3.4 step 0:** if `traffic_stopped` is not `True` (CLI: `--i-have-stopped-traffic`, or an
        interactive y/n prompt naming §2.6's precise process-to-stop when the flag is absent and
        the CLI is attached to a terminal), abort before any graph write — this is a confirmed,
        named precondition, not a silent assumption (§3.4).
     b. Preflight: `resolution = gateway.resolve("embedding", requested=target_ref)` (no `ws=` —
        §3.3's hard-cap bypass); abort loudly if `resolution.primary.dim is None` (§3.3's
        precondition — `models.<target_ref>.dim` must be declared in `config/models.json` first).
     c. For each label in `("Message", "Chunk")`: loop §4 item 1's keyset read (batch of
        `{id, text}` where `id > lastId`) → embed each row via `gateway.embedder("embedding",
        requested=target_ref)` (built once, reused across rows) → §4 item 2's write, skip-and-log
        on a `properties_set == 0` no-op (§3.3) → advance `lastId` to the batch's last id → print
        batch progress → repeat until a batch returns empty rows.
     d. §4 item 3's count check for both labels; abort before touching the index if either label
        reports a nonzero unmigrated count (a defect in step c, not a state to build an index over).
     e. §4 item 5's drop + recreate, **guarded**: skip `DROP VECTOR INDEX` when
        `Repository.read_index_dimension(ws, label=label) is None` (§3.4 step 3 — the crash-between-
        drop-and-create resume hazard `graph-dba` found), then `CREATE VECTOR INDEX ... OPTIONS
        {dimension: resolution.primary.dim, ...}` for both labels.
     f. FR-5: `current = repo.read_model_overrides(ws)`; `repo.write_model_overrides(ws,
        agent=current["agentModel"], guard=current["guardModel"], embedding=target_ref,
        responder=current["responderModel"], at=<now>, by="embedding_migration.migrate")` — same
        read-before-write discipline as `pin`.
     g. Print an explicit, impossible-to-miss instruction: **"restart the falkor-chat server
        process serving this workspace now, before resuming traffic"** (§2.7/§3.4 step 5) — the
        script cannot restart a process it isn't running as, so this is an operator action the
        tool must surface, not silently assume.
     h. Return a `MigrationReport` (workspace, target ref, per-label migrated/total/skipped counts,
        elapsed time) — both printed and returned, so `scripts/migrate_embeddings.sh` can echo a
        clean final summary and a test can assert on the returned structure directly.
   - `scripts/migrate_embeddings.sh <wsId> <targetModelRef> [--batch-size N] [--i-have-stopped-traffic]`
     — same wrapper shape as step 1's `pin` wrapper.
   - Done when: against a freshly-seeded `ws:eval` at the current 1024-dim model, running `migrate`
     to a declared-768-dim candidate ref (with `--i-have-stopped-traffic`) rewrites every
     `Message`/`Chunk` embedding, rebuilds both vector indexes at 768, updates the override, and a
     re-run of `migrate` with the same target reports zero rows processed (idempotent, FR-10) with
     the count check already at 0 unmigrated; running without the flag (and not attached to a
     terminal to answer the prompt) aborts before any write.
5. **Interrupt/resume test scaffold** — since FR-10 is an acceptance criterion, not just a nice-to-
   have: a way to kill the `migrate` loop mid-batch in a test (e.g., inject a fake embedder that
   raises after N calls) and confirm a second `migrate` call completes the remaining rows without
   re-processing already-migrated ones and without re-running the index rebuild until the count
   check passes; and a way to simulate a crash between the index `DROP` and `CREATE` (write the
   `DROP` directly in the test, then call `migrate` again) and confirm the guard (step 4e) prevents
   the hard error `graph-dba` found.

## 6. Test strategy

Follows `server/`'s existing split: offline `pytest` (network-free, stub embedder injected) for
every unit of logic below, `pytest -m live` (or a manual `ws:eval` run) for the one thing that
genuinely needs LM Studio. Ordered as behaviors to drive red→green if handed to `tdd-engineer`;
each maps to an acceptance-criteria bullet in the requirements doc where one exists.

**`pin` (FR-1/FR-2):**
1. Workspace with no `WorkspaceConfig` at all → `pin` writes `embeddingModelOverride` equal to the
   current `defaults.embedding`; `agentModelOverride`/etc. remain absent (not nulled from nothing).
2. Workspace with an existing `agentModelOverride` set (no `embeddingModelOverride`) → `pin` sets
   `embeddingModelOverride` and leaves `agentModelOverride` byte-identical (the §2.3 landmine,
   pinned directly — this is the one test most likely to catch a regression here).
3. Workspace with `embeddingModelOverride` already set → `pin` is a no-op; return value matches
   the pre-call read; a second `write_model_overrides` call is never issued (assert via a spy repo
   or mock, not just the end state, so a future refactor can't silently reintroduce a wasted write).
4. Acceptance-criteria bullets 1-3 (existing workspace unaffected by a later default change; new
   workspace pins the default in effect at its own birth) — an integration-level test: `pin` two
   workspaces with different `defaults.embedding` values (monkeypatch the overlay between calls,
   same pattern `conftest.py`'s `_model_config_env` fixture already uses) and confirm each keeps
   its own value after the global default changes again.

**`migrate` (FR-3/FR-4/FR-5/FR-8/FR-10), stub embedder (deterministic vectors, no network), always
called with `traffic_stopped=True` unless the test is specifically about step 0 (test 6):**
5. A workspace with N `Message` and M `Chunk` rows at the old dimension, all embedded → `migrate`
   to a new target/dimension rewrites all N+M rows, both `embedding` (new length) and
   `embeddingModel` (== target ref); vector index dimension after the run reads back as the new
   dimension (`read_index_dimension`).
5a. **The self-healing path (review minor, previously untested by name):** same fixture as test 5,
    but one `Message` and one `Chunk` row carry `embedding IS NULL` (never embedded at all) going
    in — `migrate` sweeps both up via the same `coalesce(embeddingModel,'') <> $target` clause
    (§3.3) and they end up embedded and marked `embeddingModel == target_ref` exactly like every
    other row, with no special-casing anywhere in the implementation.
6. **The step-0 traffic-stop precondition (§3.4, review Blocker 2):** calling `migrate` with
   `traffic_stopped=False` (or the CLI with no `--i-have-stopped-traffic` and no attached terminal
   to prompt) aborts before any read/write against the graph — assert zero `GRAPH.QUERY` calls were
   made, not just that the function raised.
7. **The hard-cap-bypass regression test (§3.3), the single most important test in this plan:**
   workspace already has `embeddingModelOverride` set to the *old* model when `migrate` is called
   with a *different* target — assert the embedder actually used for every row is built from
   `target_ref`, not the workspace override (e.g., a spy on `ModelGateway.embedder`'s call args, or
   two distinguishable stub embedders wired to old/new refs and asserting only the target's stub
   was ever invoked). This is exactly the silent-wrong-model failure mode §3.3 identified — see §7
   for why it is no longer ranked as the *sole* highest-severity risk in this plan.
8. Idempotent resume (FR-10, acceptance criteria "if interrupted partway through, re-running only
   processes unmigrated rows"): inject a fake embedder that raises after K calls; first `migrate`
   call processes K rows then raises (count check/index/override untouched — steps d/e/f in §5
   step 4 never reached); a second `migrate` call with a non-raising embedder completes the
   remaining rows and produces the same end state as an uninterrupted run; the index rebuild
   happens exactly once, only after the second call's count check passes.
9. **The DROP-guard resume hazard (`graph-dba`'s finding, §4 item 5/§3.4 step 3):** simulate a crash
   landing exactly between the index `DROP` and `CREATE` (drop the index directly in the test
   fixture, leaving the graph in that exact intermediate state), then call `migrate` again — assert
   it does **not** attempt an unconditional `DROP VECTOR INDEX` (which `graph-dba` confirmed hard-
   errors with no index present) and instead proceeds straight to `CREATE VECTOR INDEX` at the
   target dimension.
10. FR-8(a) count check reports `migrated == total, 0 unmigrated` on success; reports the actual
    nonzero unmigrated count (not just "not done") when called mid-migration.
11. Edge case: a workspace with zero `Message`/`Chunk` nodes — `migrate` completes cleanly (count
    check 0/0), index rebuild still runs (guarded per test 9's logic — a fresh/empty workspace's
    index already exists from `bootstrap_schema.sh`, so this is the ordinary guarded-drop path, not
    the missing-index one), no crash.
12. Edge case: `target_ref`'s `dim` not declared in the overlay — `migrate` aborts before any HTTP
    call or graph write, naming the missing `models.<ref>.dim` config key (mirrors
    `EmbeddingDimensionError`'s existing "no wasted inference" posture).
13. Edge case: a row's write reports `properties_set == 0` (vanished between read and write) —
    `migrate` logs and continues (§3.3's "skip and log" decision), does not abort the batch, and the
    row is correctly still absent from `migrated` in the final count report (it was never marked).
14. FR-5: after a successful `migrate`, `read_model_overrides(ws)["embeddingModel"] == target_ref`
    and the other three kinds are unchanged from their pre-migration values (same landmine as
    test 2, on the write side this time).

**FR-2's enforcement mechanism (§3.2/§5 step 2) — whichever option the stakeholder picks:**
15. `pin`'s own tests (1-4) are unaffected by the Option A/B choice — its internal logic doesn't
    change either way. If Option B wins: a test that `create_workspace.sh`'s (or the equivalent
    in-process call for `seed_eval_corpus.py`) end-to-end run leaves the target workspace with an
    explicit `embeddingModelOverride`, and a second test that a `ModelConfigError` from
    `ModelGateway.from_env()` degrades to a printed warning rather than aborting workspace creation
    (§3.2's regression mitigation for `FALKORCHAT_ENABLE_AGENT=0`).

**Live/manual (FR-6, FR-8b — not unit-testable, run once per real migration):**
16. `./run.sh run` + `compare` (model-bench) with the candidate model, reviewed, before `migrate`
    is ever invoked against `ws:eval` (§3.5) — a documented runbook step, not a pytest case.
17. Post-migration `ws:eval` retrieval sanity check: re-run a handful of `docs/QUERIES.md` §6
    hybrid-retrieval reads against the live graph and confirm the returned chunks/messages still
    match the golden queries' expected answers (§3.5 step 4) — manual review, the same posture
    `pytest -m live` already uses for anything needing a real LM Studio round trip.
18. Full-outage manual check: restart the server process after a `migrate` run (§3.4 step 5) and
    confirm a subsequent ordinary write to the migrated workspace succeeds (no stale
    `EmbeddingDimensionError` from the process-lifetime cache, §2.7) — this is the test that would
    have caught the cache bug if it had gone unnoticed. For `ws:eval` specifically this is a
    formality (§2.6 — no live-serving process exists for it); worth actually exercising once against
    a throwaway workspace that *does* have one (e.g. a disposable `ws:smoketest` stood up with
    `start_server.sh`) before the first real migration of a live-served workspace ever runs.

## 7. Risks & open questions

**Risks — two comparably severe, both now named and mitigated (revised per review):**
- **The hard-cap bypass** (§3.3, test 7): getting it backwards produces no error, no crash, and a
  report that looks like success — only a retrieval-quality regression (or, if old/new models share
  a dimension, nothing observable at all until someone notices answers have quietly gotten worse).
  Mitigated by an explicit, named design decision (§3.3) and the single most emphasized test case
  (§6 test 7), but this is exactly the kind of defect a code review must specifically look for, not
  something the test suite alone should be trusted to catch on first landing.
- **The pre-migration traffic race** (§3.4 Blocker-2 fix, test 6): this plan's first draft designed
  the *post*-migration restart in detail but had no step actually creating the outage FR-7 assumes
  — a live write landing between the count check and the index rebuild reproduces the *identical*
  silent-ANN-drop failure mode as the hard-cap bypass, for FR-9's general-reuse case in particular.
  Comparably severe to the item above, and was **entirely unmitigated and unnamed** in this plan's
  first draft — not a smaller risk than the hard-cap bypass, a different one. Mitigated now by §3.4
  step 0's explicit, confirmed precondition (`--i-have-stopped-traffic` / interactive prompt) and
  §2.6's precise scoping of what "stop traffic" means in this deployment's actual architecture — but
  it remains, like the item above, an operator-discipline risk a technical flag can prompt but not
  fully enforce (nothing stops an operator from passing the flag without actually having stopped the
  process).
- **The process-restart step is easy to forget** (§2.7, §3.4 step 5): nothing *forces* the operator
  to restart the server after `migrate` completes; a forgotten restart produces
  `EmbeddingDimensionError` on the next ordinary write, which is at least loud (not silent
  corruption) but is a confusing failure mode if the connection to "I ran a migration ten minutes
  ago" isn't obvious to whoever's on call. §5 step 4g makes the script print the instruction
  explicitly; whether that's enough (vs., e.g., the script attempting to signal the running process
  itself) is a judgment call for review, not resolved further here.
- **Batch size and re-embed duration are unmeasured.** This plan doesn't know how many
  `Message`/`Chunk` rows a real production workspace carries, or LM Studio's per-call embedding
  latency for the new candidate model — a workspace with tens of thousands of rows, embedded
  sequentially one HTTP call at a time, could make FR-7's "brief" outage (now that §3.4 step 0
  actually creates it) considerably less brief than the phrase implies, for the entire duration of
  which the workspace is down. Not a blocker for `ws:eval` (small, disposable, already sized by
  `seed_eval_corpus.py`'s fixed 12-thread corpus, and never live-served per §2.6 anyway) but worth
  measuring before any production workspace is migrated (deliberately deferred already, per the
  decision log) — flagged here so whoever picks up that later work knows to measure first.

**Open questions (for the stakeholder, not blocking this plan's handoff):**
- **§3.2's Option A vs. Option B is a genuine stakeholder decision, not a design gap this plan left
  open by omission.** Both are now designed to the same level of concreteness (call sites, comment
  updates, the one identified regression risk and its mitigation for Option B). This plan's own
  recommendation is **Option B** (a `create_workspace.sh` replacing `bootstrap_schema.sh` as the
  real-workspace entry point, with the best-effort-pin mitigation for `FALKORCHAT_ENABLE_AGENT=0`)
  because the decision log's "not left as a manual convention" line is dated, specific, and already
  confirmed — but §5 step 2 is explicitly gated on the stakeholder's actual pick, not this plan's
  recommendation defaulting into place unconfirmed.
- Should `ws:demo`/`ws:agent-team` receive an FR-1 pin alongside `ws:acme` and `ws:eval` in the
  one-time sweep, or is either out of scope until actually chosen as a migration target? Not
  resolved here — an ops decision at sweep time (§5 step 3), not a design one.
- **The multi-tenant future of §3.4 step 0.** §2.6's "stop the one process" answer is precise and
  correct for this M1 architecture (one process, one workspace, verified via `config.get_context()`)
  but will need revisiting the moment real per-request tenancy (K-016) lands and one process can
  legitimately serve several workspaces at once — flagged for whoever picks up that work, not
  resolved further here (this plan's review agreed this is fine to defer, since `ws:eval`'s own
  no-live-traffic status makes it moot for the one migration this plan actually schedules).

---

**Ready to implement:** `falkor-chat/docs/plans/embedding-migration.md` designs the full mechanism
— `scripts/embedding_migration.py`'s `pin`/`migrate` subcommands (§5), the `graph-dba` Cypher/DDL
half now confirmed and quoted (§4), and an 18-case test list (§6) with the pre-migration-traffic
precondition (test 6) and the hard-cap-bypass regression (test 7) both called out as top-priority,
comparably severe cases (§7). **Nothing is blocked on `graph-dba` any longer** — the companion note
landed and item 6's ordering question came back safe as designed, no amendment needed. The **one**
remaining gate before §5 step 2 specifically is the stakeholder's pick between §3.2's Option A and
Option B (this plan recommends B) — every other step (§5 steps 1, 3, 4, 5) is unblocked and can
proceed now, in parallel with that one decision.
