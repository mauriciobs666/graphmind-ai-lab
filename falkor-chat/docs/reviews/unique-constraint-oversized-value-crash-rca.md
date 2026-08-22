# `unique-constraint-oversized-value-crash` — RCA

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** K-049

## 1. Symptom & impact

During the K-028 (workflow timers) implementation, `coder`'s first attempt at a ctx-merge
length-bound test used a deliberately oversized (~8KB) `Step.key` to trigger the check. Publishing
that workflow def **crashed the shared dev `falkordb-dev` container outright** — connection
dropped mid-request, the `--rm` container vanished from `docker ps -a` entirely, reproduced twice
across independent restarts. `coder` worked around it by using an oversized `ctx` value instead
(opaque, unindexed — safe), so K-028 itself never trips this. The underlying engine behavior was
real and unresolved — this RCA closes that gap.

**Confirmed root cause:** FalkorDB v4.18.11 (module `41811`) segfaults the whole `redis-server`
process — not just the query, not just the graph, the entire instance — when a `CREATE`/`MERGE`
writes a value **longer than exactly 4096 bytes** into a property backed by a `GRAPH.CONSTRAINT
CREATE ... UNIQUE` constraint. This is a genuine FalkorDB engine defect (a missing bounds/NULL
check), not a K-028 code defect, not resource exhaustion, and — per today's actual write paths —
**not currently reachable through falkor-chat's REST API** (see §5). It remains a real hazard for
any in-process caller that bypasses the REST pydantic boundary (tests, scripts, a future MCP tool,
`materialize_snapshot`'s already-known-unguarded path).

## 2. Reproduction (isolated container, never `falkordb-dev`)

All testing ran against a disposable container, explicitly separate from the shared
`falkordb-dev` instance used by `kaizen_team`/`cpg_falkorchat`/live workspaces:

```bash
docker run -d --name falkordb-repro-k049 -p 16379:6379 falkordb/falkordb:v4.18.11   # NO --rm
```

`falkordb-dev` was verified running/untouched before, during, and after every test in this RCA.

Schema mirrored falkor-chat's real `Step` DDL exactly (`scripts/bootstrap_schema.sh`):

```cypher
CREATE INDEX FOR (n:Step) ON (n.key)
CREATE INDEX FOR (n:Step) ON (n.stepUid)
GRAPH.CONSTRAINT CREATE <graph> UNIQUE NODE Step PROPERTIES 1 stepUid
```

**Exact production shape** (`repository.py`'s `_PUBLISH_CYPHER`) reproduced verbatim via
`falkordb-py` 1.6.1 — a `MERGE` on a **computed string concatenation** as the match key, inside an
`UNWIND`, targeting the constrained property:

```cypher
MERGE (st:Step {stepUid: $key + ':' + $version + ':' + s.key})
  ON CREATE SET st.key = s.key, st.type = s.type, st.config = s.config
```

With `s.key` = `"x" * 8000` (matching the reported ~8KB size, `MAX_CONFIG_LEN`): **connection
closed by server after 3.5s; container exited, code 139 (SIGSEGV).** Reproduced on the first try,
matching the original report exactly.

## 3. Captured crash evidence (no `--rm` this time)

```
=== REDIS BUG REPORT START: Cut & paste starting from here ===
1:M 22 Aug 2026 00:59:46.053 # Redis 8.6.3 crashed by signal: 11, si_code: 1
1:M 22 Aug 2026 00:59:46.053 # Accessing address: (nil)
1:M 22 Aug 2026 00:59:46.053 # Crashed running the instruction at: 0x7fe7495de609

------ STACK TRACE ------
EIP:
/var/lib/falkordb/bin/falkordb.so(EnforceUniqueEntity+0x1e9) [0x7fe7495de609]
...
29 thread-pool-thr *
/var/lib/falkordb/bin/falkordb.so(EnforceUniqueEntity+0x1e9) [0x7fe7495de609]
/var/lib/falkordb/bin/falkordb.so(Schema_EnforceConstraints+0x68) [0x7fe74963f8c8]
/var/lib/falkordb/bin/falkordb.so(CommitNewEntities+0x532) [0x7fe749602222]
/var/lib/falkordb/bin/falkordb.so(+0x25285e) [0x7fe7495fa85e]
... ExecutionPlan_Execute ... _query ...
```

`docker inspect`: `exitcode=139` (128+SIGSEGV), `OOMKilled=false`. **Signal 11, si_code 1
(SEGV_MAPERR), faulting address `(nil)`** — a null-pointer dereference, not an OOM abort, not a
resource-exhaustion crash. The call chain is unambiguous: `CommitNewEntities` (the write-commit
path) → `Schema_EnforceConstraints` → `EnforceUniqueEntity` — i.e. the crash fires specifically
while the engine is **checking the `UNIQUE` constraint**, not while parsing, indexing, or storing
the property generally. Full captured log: available in this session's scratchpad
(`crash_8000_full.log`), not checked into the repo (throwaway diagnostic artifact, not a durable
doc).

## 4. Bounding the failure mode

Four questions from the backlog item's scope sketch, each answered by an isolated test (fresh
graph key per test so one crash never contaminates the next; container `docker start`s back up
non-destructively after each crash — confirmed alive via `PING` before proceeding):

| Question | Test | Result |
|---|---|---|
| Specific to `Step.key`/`stepUid`, or any constrained property? | Same shape on a throwaway `:U {uid}` label/property, `UNIQUE NODE U PROPERTIES 1 uid` | **Any UNIQUE-constrained property** — not `Step`-specific. |
| Is it the index, or the constraint? | Same 8000-byte value on an **indexed-only, no-constraint** property, then swept to 1,000,000 bytes | **Never crashes without a constraint** — RANGE index alone is safe at any size tested (up to 1MB). |
| Is it any oversized value, or a threshold? | Binary search 100 → 8000 bytes on the constrained property | **Exact boundary: 4096 bytes safe, 4097 bytes crashes.** Reproduced consistently at 4097/4100/4104/4112/4128/4200/4500/5000/6000/7000/8000 — all crash; 100/1000/4000/4096 — all safe. |
| Composite constraints (`PROPERTIES 2 key version`)? | One column oversized + one small (`key`×4097, `version`×100) vs. both under threshold (`key`×3000 + `version`×3000, combined 6000) | **Threshold is per-property, not combined-encoding.** 3000+3000 (each <4096) is safe; 4097+100 (one column over) crashes. |
| Does the write-clause shape matter (`CREATE` vs `MERGE` vs computed-concat-in-`UNWIND`)? | Bare `CREATE`, bare `MERGE`, and the exact `_PUBLISH_CYPHER` shape | **Identical outcome in all three** — the crash is in constraint enforcement at commit time, after the value is already computed; how it got there doesn't matter. |

**Mechanism, as far as black-box testing can establish it:** the exact 4096 = 2^12 boundary is a
strong signal of a fixed-size (likely stack) buffer — probably a `char buf[4096]`-shaped encoding
buffer used when hashing/comparing the constrained property's value for the uniqueness check.
When the value exceeds that buffer, something in the encoding path returns (or leaves) a NULL
pointer that `EnforceUniqueEntity` then dereferences unchecked. This is consistent with every
observation above: it only fires on the **constraint-enforcement code path** (not on indexing or
plain storage), it fires per-property (each column gets its own encode/compare call), and it is
completely insensitive to how the oversized value was constructed (literal, concatenation,
`UNWIND`-batched).

**Not a resource-exhaustion crash:** `OOMKilled=false`; the offending value is ~4-8KB, trivially
small; the fault address is a literal null pointer, not a large/wild address. This is a missing
bounds/NULL check in the C constraint-enforcement code, not a "went too big for available RAM"
story — no amount of RAM headroom on `falkordb-dev` would have prevented it.

## 5. Is this actually reachable through falkor-chat's app today?

**No — not through the REST front door.** `server/falkorchat/schemas.py` already bounds every
field that feeds a `UNIQUE`-constrained property reachable from `publish_workflow_def`:
`PublishWorkflowDefIn.key`/`.version` and `WorkflowStepIn.key`/`WorkflowTransitionIn.from_/to/on`
all carry `Field(max_length=MAX_KEY_LEN)` with `MAX_KEY_LEN = 200` — comfortably under the 4096-byte
cliff even at a pessimistic 4-bytes-per-character UTF-8 blowup (200 × 4 = 800 ≪ 4096).
`publish_workflow_def` is REST-only (`api.py:181-184`) — it is not exposed as an MCP tool, so
`PublishWorkflowDefIn`'s pydantic validation is the *only* gate a real client request passes
through, and today that gate already closes this specific vector.

**What actually happened:** `coder`'s test called `Services.publish_workflow_def(...)` **directly**
(the standard `wf_repo`-fixture pattern in `test_workflow_timers.py`), which — like every other
in-process caller — bypasses `schemas.py` entirely. `services.py` has **no runtime re-check** of
key/version/step-key length; `_validate_def_spec` (`services.py:961`) validates `kind`, step
types, step-key uniqueness, the single-start-step invariant, and transition-endpoint resolvability
— but never a length bound. This is precisely the gap `services.py`'s own docstring already
diagnoses for a different field: *"bounds pydantic structurally CANNOT enforce"* (the comment
introducing `MAX_CONFIG_LEN`'s import, `services.py:47-63`) — the team already solved this exact
problem once for the ctx-merge/sweep-limit case by duplicating the guard at the service layer, and
has not yet done the same for `key`/`version`/step keys.

**Second unguarded path, already flagged (K-030 re-gate r-1):** `materialize_def` →
`materialize_snapshot` (`repository.py:1397`, `:1669`) reuses the identical `_PUBLISH_CYPHER` shape
against a workspace graph and performs **no spec validation at all** — not the length gap, not
K-030's zero-transition gap. Any def that reaches `reference` with an oversized key (bypassing
`publish_workflow_def`'s REST bound some other way — a future non-REST publish path, a direct
repository call, a corrupted/hand-edited seed) would crash the engine again on materialize, with
no independent guard to catch it there either.

## 6. Proposed fix layer (design only — no application code changed here)

**1. Service-layer defense-in-depth guard, extending `_validate_def_spec`.** Add a length check
   inside `_validate_def_spec` (`services.py:961`) for every `step["key"]` and every transition's
   `from`/`to`/`on`, raising the existing `WorkflowDefSpecError` — same failure shape K-024's U4b
   transition-less-spec check already uses ("prevention, not a nicer exception"). Also add a
   check on `key`/`version` themselves at the top of `publish_workflow_def`
   (`services.py:1241-1244`, before `_validate_def_spec` or folded into it via two extra
   parameters) — today those two arrive with **zero service-layer validation**, pydantic-only.
   Reuse `MAX_KEY_LEN` (already imported nowhere in `services.py` — would need adding alongside
   the existing `MAX_CONFIG_LEN` import from `.schemas`) as the bound; no new number needed, since
   200 is already far inside the safe margin below the measured 4096-byte cliff.

**2. Close `materialize_snapshot`'s independent gap in the same change.** `materialize_def` /
   `materialize_snapshot` should call the same extended `_validate_def_spec` (or a shared helper)
   before writing into the workspace graph — this is the same integration point K-030 already
   proposes for its own (different) zero-transition gap; one guard call added to that path closes
   both K-030's re-gate finding and this RCA's finding together, worth sequencing as one unit
   rather than two separate patches touching the same function.

**3. Document the engine cliff next to the constant, not just in this RCA.** Add a one-line
   comment at `MAX_KEY_LEN`'s definition (`schemas.py:51`) recording *why* 200 was chosen relative
   to a hard engine limit (4096 bytes, confirmed crash) — so a future contributor doesn't casually
   raise `MAX_KEY_LEN` toward "closer to 4096" without knowing the risk is a crash, not a
   truncation.

**4. Fold the standing rule into `falkor-chat/AGENTS.md`'s schema conventions.** Add: *"A value
   written into a `UNIQUE`-constrained property must never be unbounded — FalkorDB v4.18.11
   segfaults the whole instance (not just the write) above 4096 bytes on any single constrained
   property, with no engine-side guard. Every constrained property fed by caller-supplied content
   needs an app-side length bound enforced at the service layer, not only at the REST/pydantic
   boundary — a non-REST caller (test, script, future MCP tool) bypasses pydantic entirely."* This
   is the durable, non-obvious fact this RCA exists to hand forward; it belongs in the schema
   conventions, not buried in a backlog comment.

**5. Scope check for other constrained fields.** Every other `UNIQUE`-constrained property in the
   schema (`userId`, `agentId`, `channelId`, `threadId`, `msgId`, `documentId`, `chunkId`,
   `entityId`, `runId`, `stepRunId`, `traceId`, `cursorId`, `workspaceConfigId`) is server-minted
   (hex UUIDs, per `MAX_ID_LEN`'s own comment, `schemas.py:28-30`) — fixed short length, never
   caller-supplied free text, so **not** at risk the way `key`/`version`/step-keys are (those are
   the only caller-supplied semantic strings feeding a constrained property today). No action
   needed there; noted so a future auditor doesn't have to re-derive this.

**Not proposed:** lowering or otherwise re-deriving `MAX_KEY_LEN`'s value — 200 already has ample
margin under the 4096-byte cliff (even accounting for UTF-8 multi-byte expansion). The defect is a
**coverage gap** (pydantic-only enforcement, no service-layer mirror), not a wrong threshold.

## 7. Upstream report (K-007 OQ6 precedent)

This is a genuine FalkorDB engine defect independent of any app design choice — worth reporting to
`github.com/FalkorDB/FalkorDB` regardless of the app-side fix above, since any FalkorDB user with a
`UNIQUE` constraint on an unbounded-length property is exposed to a full-instance crash. Recommended
report contents (mirrors the `K-007 OQ6` "confirmed engine anomaly" precedent in
`falkor-chat/docs/BACKLOG.md`):

- **Engine:** `falkordb/falkordb:v4.18.11`, module version `41811`, Redis 8.6.3.
- **Defect:** `CREATE`/`MERGE` writing a value >4096 bytes into a property backed by a
  `GRAPH.CONSTRAINT CREATE ... UNIQUE` constraint crashes the entire `redis-server` process
  (SIGSEGV, signal 11, si_code 1 `SEGV_MAPERR`, faulting address `(nil)`) — a null-pointer
  dereference inside `EnforceUniqueEntity`, called from `Schema_EnforceConstraints` ←
  `CommitNewEntities` (the commit/write path).
- **Exact boundary:** 4096 bytes safe, 4097 bytes crashes — reproducible, deterministic, not
  timing- or load-dependent.
- **Scope:** constraint-enforcement path only (a RANGE index with no constraint is safe at least
  to 1MB tested; an unindexed property is safe at least to 8000 bytes tested); per-property for a
  composite `PROPERTIES N ...` constraint, not the combined encoded key; independent of write
  shape (`CREATE`, `MERGE`, `MERGE` on a computed string-concatenation expression inside `UNWIND`
  all reproduce identically).
- **Repro:** the exact Cypher and container recipe in §2 above.

## 8. Test strategy for the eventual fix

Once §6's guard lands: a regression test at the service layer (not against a live engine — the
whole point is the guard must reject **before** any Cypher reaches FalkorDB) asserting
`WorkflowDefSpecError` for an oversized `key`/`version`/step-key/transition-endpoint, mirroring
`test_workflow_timers.py`'s existing pattern of testing the `MAX_CONFIG_LEN` ctx-merge bound
end-to-end without ever attempting the oversized write against a real graph. No live-engine crash
test is warranted or safe to keep in the suite — the isolated-container repro in this RCA is the
one-time proof; it should not become a standing CI step against any shared instance.

## Pass 1 — 2026-08-21 (`analyst` gate)

**Scope.** Independent verification of this RCA's own evidence and conclusions — not a re-derivation
from scratch. Checked: (1) the crash mechanism/boundary/stack trace against a fresh, disposable
FalkorDB v4.18.11 container (never `falkordb-dev`); (2) the production-unreachability claim against
the live `schemas.py`/`services.py`/`api.py`/`repository.py` source; (3) every file:line citation in
the doc against the real source; (4) the proposed fix design's soundness/completeness. Did not
independently re-run the composite-constraint or write-clause-shape-independence sub-tests (§4 rows
4–5) — graded the doc's own evidence quality for those instead (see below).

`CPG: considered, not relevant — cpg_falkorchat exists but the crash mechanism under review lives
inside the closed-source FalkorDB engine binary, which no CPG in this lab models; the app-side
reachability questions (§5) were small and concrete enough to verify directly against the named
files/lines faster and more reliably than a call-graph traversal would.`

**Verdict: approve with suggestions.** No blockers. The core engine-defect claim and the
"not-reachable-in-production" claim both hold up under independent re-derivation — see below. Two
minor findings, neither undermines the RCA's conclusions or blocks the proposed fix from proceeding.

### Independent reproduction — CONFIRMED

Ran a disposable, isolated `falkordb/falkordb:v4.18.11` container (`analyst-repro-k049`, port
16380, never touching `falkordb-dev`/`kaizen_team`/`cpg_falkorchat`/any `ws:*` graph — verified
`falkordb-dev` up and answering `PING` before, during and after). Mirrored the doc's schema
recipe (`CREATE INDEX FOR (n:U) ON (n.uid)` then `GRAPH.CONSTRAINT CREATE repro UNIQUE NODE U
PROPERTIES 1 uid`, polled `CALL db.constraints()` to confirm `OPERATIONAL`).

- **4096-byte value on the constrained property: safe** (`Properties set: 1`, no error) — matches
  the doc exactly.
- **4097-byte value on the same property: crashed the whole container**, `docker ps -a` showing
  `Exited (139)`. The captured crash log is **byte-for-byte the same fault** the RCA reports:
  `Redis 8.6.3 crashed by signal: 11, si_code: 1`, `Accessing address: (nil)`, and the full stack —
  `EnforceUniqueEntity+0x1e9` ← `Schema_EnforceConstraints+0x68` ← `CommitNewEntities+0x532` ←
  `ExecutionPlan_Execute+0x4e` — reproduced independently, in a separate container, on a separate
  run. This is as strong a confirmation as this claim can get short of reading the FalkorDB source.
- **Index-vs-constraint distinction: confirmed.** An 8000-byte value on an indexed-but-unconstrained
  property (`CREATE INDEX FOR (n:V) ON (n.vid)`, no `GRAPH.CONSTRAINT`) wrote cleanly with no crash
  — the doc's central "constraint-specific, not index-specific" claim holds on direct test, not just
  on the doc's own say-so.
- **Not independently re-run:** the composite-constraint row and the write-clause-shape row (bare
  `CREATE` vs `MERGE` vs computed-concat-in-`UNWIND`) in §4's table. Grading the doc's own evidence
  for these instead: the methodology is sound (isolated fresh graph key per test, explicit
  before/after `PING` liveness checks, binary search with the boundary bracketed on both sides), and
  the claimed results are mechanistically consistent with the confirmed stack trace (a per-property
  encode/compare call inside constraint enforcement, blind to how the value was constructed
  upstream) — I'd trust these two rows on the doc's evidence alone.

### Production-unreachability claim — CONFIRMED

- `schemas.py:51` `MAX_KEY_LEN = 200` bounds exactly the fields claimed: `WorkflowStepIn.key`
  (`:70`), `WorkflowTransitionIn.from_/to/on` (`:79-81`), `PublishWorkflowDefIn.key/version`
  (`:87-88`) — read directly, matches.
- `services.py:961` `_validate_def_spec` — read the full body: it checks `kind`, step `type`,
  step-key uniqueness, single-start-step, transition-endpoint resolvability, `waitsForHuman`,
  `requiredTools` shape, guard structure, and the K-024 U4b "at least one transition" rule. **No
  length check anywhere.** Confirms the claimed gap exactly.
- `services.py:1241` `publish_workflow_def` — confirmed `key`/`version` arrive with zero
  service-layer validation before `_validate_def_spec`, as claimed.
- `server/tests/test_workflow_timers.py` + `conftest.py:101`'s `wf_repo` fixture — confirmed this
  builds a real `Repository(conn)` against the live connection and `Services(repo, ...)` directly;
  the test suite calls `services.publish_workflow_def(...)` with no pydantic model in between,
  exactly the bypass the RCA describes as what actually happened.
- `api.py:181-184` `publish_workflow_def` route — confirmed REST-only; grepped the whole server tree
  for an MCP tool wrapping it — none exists. Claim holds.
- **K-030 re-gate finding r-1** (`docs/BACKLOG.md:735-739`) independently confirms, in its own
  words, written before this RCA existed: *"`services.materialize_def` → `repository.materialize_snapshot`
  reuses the same query shape and performs no spec validation"* — this RCA's §5 "second unguarded
  path" claim is not new speculation, it's citing an already-documented, independently-derived finding
  that matches exactly.
- Server-minted IDs (`userId`, `agentId`, etc.): confirmed `uuid.uuid4().hex` in both
  `services.py:180` and `executor.py:96` — fixed 32-hex-char length, not caller-supplied. Claim 5's
  "no action needed there" holds.
- Confirmed FalkorDB has **zero `RELATIONSHIP`-type constraints anywhere in this schema**
  (`grep -n RELATIONSHIP scripts/bootstrap_schema.sh` → no matches) — so `TRANSITION.on`/`.guard`
  never feed a constrained match key at all. This slightly over-broadens §6 item 1's phrasing
  ("length check ... for every transition's from/to/on") — `on` isn't at risk of *this* crash class
  the way `from`/`to` are (they compose `Step.stepUid`'s MERGE key; `on` doesn't). Bounding it anyway
  is harmless and already mirrors the existing pydantic bound, so this is a precision nit for the
  implementer, not a defect in the design.

### Findings

**Minor — two stale line citations in §5.** "`materialize_def` → `materialize_snapshot`
(`repository.py:1397`, `:1669`)" points at the wrong code: `:1397` lands inside
`record_step_result`'s parameter dict (unrelated write), and `:1669` lands inside
`sweep_due_workflow_runs`'s docstring (also unrelated) — verified by reading both lines directly.
The correct locations, verified against the live source: `materialize_def` is `services.py:1308`
(not `repository.py` at all — it's a `Services` method that *calls* `repository.materialize_snapshot`);
`materialize_snapshot` itself is `repository.py:1864`; the shared `_PUBLISH_CYPHER` constant is
`repository.py:1037` (not `:992`, which is inside `list_thread_participants`). This appears to be
copied verbatim from `falkor-chat/AGENTS.md`'s own "Probing shared graph state" section, which
carries the identical stale numbers (`repository.py:1669` / `:992`) — likely both went stale after
the same later edit shifted line numbers in `repository.py`, and neither doc was re-diffed against
current line numbers before citing them. The underlying claim is still correct (independently
verified above at the right locations) — this is a citation-accuracy defect, not a substantive one,
but it's exactly the kind of thing a reader chasing the citation would trip on. Suggested fix: correct
the two line numbers in this RCA's §5, and separately flag `falkor-chat/AGENTS.md`'s matching stale
pair for its own owner (`tico`, per the doc-flip table) to fix in a follow-up — out of scope for this
RCA to fix itself since `AGENTS.md` isn't its artifact.

**Minor — `materialize_def`'s REST route doesn't declare the same `Path(max_length=...)` bound as
its sibling route.** `api.py:235-239` (`POST /workflow-defs/{key}/versions/{version}/materialize`)
declares `key: str, version: str` with no length bound, unlike the immediately-preceding
`get_workflow_def_structure` route (`api.py:222-226`) which declares
`Path(..., min_length=1, max_length=MAX_KEY_LEN)` on both. Traced whether this actually opens a new
crash vector: it doesn't — `read_def_subgraph(key=key, version=version)` (an exact-match `MATCH`,
not a write) must find an existing `WorkflowDef` node whose stored key/version equals the path
param byte-for-byte before `materialize_snapshot` ever runs, so an oversized path param can only
reach the write if a `WorkflowDef` with that exact oversized key already exists in `reference` —
which, per this RCA's own reasoning, could only have arrived via the same non-REST bypass already
in scope. So this doesn't change the RCA's threat model, but it's a real hygiene inconsistency the
§6 fix pass should close in the same change (add the matching `Path()` bound), since defense-in-depth
consistency is this RCA's own stated theme.

### What's solid

The reproduction methodology is rigorous and disciplined: fresh graph key per test so no crash
contaminates the next, explicit liveness checks (`PING`) before/after each crash, `--rm` dropped
specifically to capture logs, and the exact production Cypher shape reproduced verbatim rather than
a simplified stand-in. The mechanism hypothesis (fixed ~4096-byte encode/compare buffer, NULL
returned/left on overflow, dereferenced unchecked) is not overclaimed — the doc is explicit that
this is "as far as black-box testing can establish it," not a source-level claim. The fix design
correctly identifies the coverage gap (pydantic-only enforcement, no service-layer mirror) rather
than mis-diagnosing it as a wrong threshold, and correctly sequences the `materialize_snapshot` gap
with K-030's own pre-existing, independently-filed finding on the same function rather than treating
it as a fresh discovery. The upstream-report section (§7) is complete enough to file as-is.

### Open questions

None — nothing here needs a stakeholder decision. The two findings above are both self-contained,
low-stakes fixes the next person touching this RCA (or its §6 fix pass) can apply without further
input.
