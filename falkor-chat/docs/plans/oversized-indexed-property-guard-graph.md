# Oversized indexed-property guard — graph findings

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** K-049 (M-current)

## 0. Read this first — this work already exists

Before doing any repro I found that **U1's entire scope was already completed and analyst-approved
five days before this coordination was opened**: `falkor-chat/docs/reviews/unique-constraint-oversized-value-crash-rca.md`,
committed 2026-08-21 (`daf3ff0`), `Owner: graph-dba`, `Status: active`, carries a full root-cause
(§1–§5), a finalized fix design (§6), an upstream-report recommendation (§7), a test strategy (§8),
and an independent `analyst` Pass 1 gate (2026-08-21, **verdict: approve with suggestions, no
blockers**) that re-reproduced the crash in its own isolated container and independently verified
every claim. The coordination doc's framing ("root cause was never established") is stale — it
matches the state of K-049 in `BACKLOG.md` before the RCA existed, not the state of the repo today.

**What actually never happened is the harden phase.** `grep -n MAX_KEY_LEN server/falkorchat/services.py`
returns nothing — the RCA's §6 guard was designed and approved but never implemented, and U2
(`tdd-engineer`) was never dispatched against it. That is the real remaining gap this coordination
should close.

**Recommendation to `teco`:** don't re-run U1 as a fresh investigation. Treat this document as a
confirmation + finalization pass over the existing RCA (below), and dispatch U2 directly against
§3's guard design here (which supersedes the RCA's §6 with corrected line citations and the two
`analyst` Pass-1 findings folded in). Reconcile `docs/plans/oversized-indexed-property-guard-coordination.md`
and `BACKLOG.md`'s K-049 text accordingly — both currently read as if no root-cause work has
happened yet.

I did not stop and ask before proceeding: producing the requested content this way satisfies every
item in the task's deliverable list, costs less than a blind re-investigation, and is reversible
(cheap to redo differently if a fuller independent re-derivation is actually wanted) — not a
scope/irreversibility fork worth a mid-run pause. Flagging it here and in my final report instead.

## 1. Fresh confirmation (today, isolated container, third independent run)

I did not simply trust the existing RCA — I re-ran the core repro myself, disposably, before relying
on it for a design deliverable that gates real code changes.

```bash
docker run -d --name falkordb-repro-k049 -p 16399:6379 falkordb/falkordb:v4.18.11   # NO --rm
```

`falkordb-dev` verified `Up`/`PONG` before, during, and after (never touched, never stopped).
Schema mirrored `scripts/bootstrap_schema.sh`'s real DDL for the `reference` graph exactly:

```cypher
CREATE INDEX FOR (n:WorkflowDef) ON (n.key)
CREATE INDEX FOR (n:WorkflowDef) ON (n.version)
CREATE INDEX FOR (n:Step) ON (n.key)
CREATE INDEX FOR (n:Step) ON (n.stepUid)
GRAPH.CONSTRAINT CREATE reference UNIQUE NODE WorkflowDef PROPERTIES 2 key version
GRAPH.CONSTRAINT CREATE reference UNIQUE NODE Step PROPERTIES 1 stepUid
```

(waited for `CALL db.constraints()` → `OPERATIONAL` on both before writing, per the project's
own index-before-constraint / async-constraint-lifecycle facts.)

Reproduced the exact production write shape — `repository.py`'s `_PUBLISH_CYPHER` (verified current
location: `repository.py:1721`, see §2 citation corrections below) — via `falkordb-py` 1.6.1 (the
pinned client), `Step.key` = `"y" * 8000`:

```cypher
MERGE (st:Step {stepUid: $key + ':' + $version + ':' + s.key})
  ON CREATE SET st.key = s.key, st.type = s.type, st.config = s.config
```

**Result:** `ConnectionError: Connection closed by server.` (client-side), container `Exited (139)`
(128+SIGSEGV). Captured `docker logs` before removing the container (no `--rm`):

```
=== REDIS BUG REPORT START: Cut & paste starting from here ===
1:M 26 Aug 2026 22:02:16.422 # Redis 8.6.3 crashed by signal: 11, si_code: 1
1:M 26 Aug 2026 22:02:16.422 # Accessing address: (nil)
1:M 26 Aug 2026 22:02:16.422 # Crashed running the instruction at: 0x7efeaffde609

------ STACK TRACE ------
EIP:
/var/lib/falkordb/bin/falkordb.so(EnforceUniqueEntity+0x1e9) [0x7efeaffde609]
...
22 thread-pool-thr *
/var/lib/falkordb/bin/falkordb.so(EnforceUniqueEntity+0x1e9) [0x7efeaffde609]
/var/lib/falkordb/bin/falkordb.so(Schema_EnforceConstraints+0x68) [0x7efeb003f8c8]
/var/lib/falkordb/bin/falkordb.so(CommitNewEntities+0x532) [0x7efeb0002222]
.../falkordb.so(+0x25285e) ... /falkordb.so(ExecutionPlan_Execute+0x4e) [0x7efeaffea0be]
/var/lib/falkordb/bin/falkordb.so(_query+0x4d5) [0x7efeaffda485]
```

**Byte-for-byte the same fault signature** the RCA and its `analyst` gate both already captured on
2026-08-21/22: same offsets (`EnforceUniqueEntity+0x1e9`, `Schema_EnforceConstraints+0x68`,
`CommitNewEntities+0x532`), same signal (11), same `si_code` (1, `SEGV_MAPERR`), same faulting
address (`nil`). Three independent runs (RCA author, `analyst`'s gate, this one), three separate
containers, identical result — as strong a confirmation as black-box testing gets.

I also re-ran the composite-key boundary check via the real `stepUid = key + ':' + version + ':' +
s.key` concatenation (not a bare property, unlike the RCA's simplified `:U {uid}` test) — with
`s.key` = 4096 bytes, the *composite* `stepUid` totals 4096 + len("timers-oversized:1:") = 4116
bytes, and it **still crashed**, consistent with the RCA's already-established "boundary is on the
final constrained value, insensitive to how it was constructed" finding (§4 row 5 there). I did not
re-run the full binary search (100→8000 bytes) or the pure composite-vs-single-column breakdown —
the RCA's own bisection (100/1000/4000/4096 safe; 4097/4100/4104/4112/4128/4200/4500/5000/6000/7000/8000
crash, each independently spot-checked by `analyst` at the two boundary points) is thorough and
already independently verified twice; a third full re-bisection would not change the answer and
isn't worth the container-crash cost.

**Conclusion: the RCA's root-cause and boundary findings stand, now with a third independent
confirmation.** Nothing here contradicts or refines the existing RCA.

## 2. Root cause (from the RCA, restated for this deliverable's own completeness)

FalkorDB v4.18.11 (module `41811`, Redis 8.6.3) segfaults the **entire `redis-server` process** —
not the query, not the graph, the whole shared instance — when a `CREATE`/`MERGE` commits a value
longer than **exactly 4096 bytes** into a property backed by a `GRAPH.CONSTRAINT CREATE ... UNIQUE`
constraint. Signal 11 (`SEGV_MAPERR`), faulting address `(nil)` — a null-pointer dereference, not an
OOM abort (`OOMKilled: false`; the offending value is a few KB, trivial against available RAM). Call
chain: `CommitNewEntities` (write-commit path) → `Schema_EnforceConstraints` →
`EnforceUniqueEntity` — the crash fires specifically while the engine checks the `UNIQUE`
constraint, not while parsing, indexing, or storing the property. The 4096 = 2^12 boundary strongly
suggests a fixed-size (likely stack) encode/compare buffer that overflows silently and leaves/returns
a NULL the caller dereferences unchecked — this is as far as black-box testing can establish the
mechanism; nothing here claims to have read the FalkorDB source.

**This general engine fact belongs in, and already lives in, `claude/graph-dba/falkordb-quirks.md`**
(added 2026-08-21/22 alongside the RCA — see lines 129–152 there). No new quirks-file entry is
needed from this session; the existing one already carries the exact boundary, the
constraint-vs-index distinction, the composite-constraint behavior, and the write-clause-shape
independence, and is dated/evidenced correctly.

## 3. Failure-mode boundary (from the RCA — independently re-verified in part, see §1)

| Question | Answer |
|---|---|
| Specific to `Step.key`/`stepUid`? | No — **any `UNIQUE`-constrained property**, any label. |
| Index or constraint? | **Constraint-specific.** A RANGE index with no `UNIQUE` constraint is safe at least to 1MB tested. |
| Threshold or any oversized value? | **Exact boundary: 4096 bytes safe, 4097 bytes crashes.** Deterministic, not timing/load-dependent. |
| Composite constraints (`PROPERTIES 2 a b`)? | **Per-property, not combined-encoding.** Each column is checked against the same 4096-byte limit independently. |
| Write-clause shape (`CREATE`/`MERGE`/computed-concat-in-`UNWIND`)? | **No effect** — same crash regardless; the fault is at commit-time constraint enforcement, after the value is already computed. |

## 4. Is this reachable through falkor-chat's app today?

**No, not through REST** — confirmed by re-reading the current source (line numbers below verified
against the live files, correcting two stale citations the RCA and `AGENTS.md` both carry):

- `schemas.py:87` `MAX_KEY_LEN = 200` bounds `WorkflowStepIn.key` (`:108`),
  `WorkflowTransitionIn.from_/to/on` (`:116-118`), `PublishWorkflowDefIn.key/version` (`:123-124`)
  — all comfortably under the 4096-byte cliff even at a pessimistic 4-bytes-per-char UTF-8 blowup
  (200 × 4 = 800 ≪ 4096).
- `publish_workflow_def` is REST-only (`api.py:315-318`) — no MCP tool wraps it.

**Reachable everywhere else that bypasses pydantic** — confirmed unchanged since the RCA:

- `services.py:1294` `_validate_def_spec` — read the current full body (§1.4 of this session):
  checks `kind`, step `type`, step-key uniqueness, single-start-step, transition-endpoint
  resolvability, `waitsForHuman`, `requiredTools` shape, `cmp`-guard structure, and the "at least
  one transition" rule. **Still no length check anywhere.** `MAX_KEY_LEN` is imported nowhere in
  `services.py` — confirmed via `grep`.
- `services.py:1574` `publish_workflow_def` — `key`/`version` still arrive with zero service-layer
  validation before `_validate_def_spec` runs.
- `server/tests/test_workflow_timers.py` and every other test using the `wf_repo` fixture calls
  `Services.publish_workflow_def(...)` directly — exactly the bypass that produced the original
  crash report.
- **Second unguarded path, corrected citations:** `services.py:1641` `materialize_def` reads an
  already-published def's subgraph from `reference` (`self._repo.read_def_subgraph`, whose
  `sub["steps"]`/`sub["transitions"]` carry the same `key`/`from`/`to` strings) and passes it
  straight to `self._repo.materialize_snapshot` (`repository.py:2561`, using the same
  `_PUBLISH_CYPHER` constant at `repository.py:1721`) with **no re-validation at all** — not the
  length gap, not K-030's zero-transition gap. (The RCA's own citations here —
  `repository.py:1397`/`:1669` — are stale; so is `falkor-chat/AGENTS.md`'s matching pair at line
  120, both flagged already by `analyst`'s Pass 1 and still uncorrected. `tico` owns fixing
  `AGENTS.md`'s copy per the doc-flip table; this document carries the corrected numbers for its
  own purposes.) Any def that reaches `reference` with an oversized key by some means other than
  `publish_workflow_def` (a hand-edited seed, a future non-REST publish path, direct repository
  access) crashes the engine again on materialize, with nothing to catch it there.
- **Newly confirmed this session, not in the original RCA:** `api.py:369-373`
  (`POST /workflow-defs/{key}/versions/{version}/materialize`) still declares
  `key: str, version: str` with **no `Path(max_length=...)` bound**, unlike its sibling route
  `get_workflow_def_structure` (`api.py:362-363`, which does declare
  `Path(..., min_length=1, max_length=MAX_KEY_LEN)`). `analyst`'s Pass 1 flagged this as a minor
  hygiene gap on 2026-08-21 and it is still unfixed. It does not open a *new* crash vector on its
  own (materialize only writes an oversized value if a `WorkflowDef` with that exact key already
  exists in `reference`, which requires the same non-REST bypass above), but closing it is one line
  and keeps the two sibling routes consistent — include it in U2's diff.

## 5. Guard design for `tdd-engineer` (U2) — finalized, supersedes RCA §6 with citations corrected

Do not lower or re-derive `MAX_KEY_LEN` — 200 already has ample margin under the 4096-byte cliff.
The defect is a **coverage gap** (pydantic bounds the REST front door only; nothing mirrors it at
the service layer, which every non-REST caller — tests, scripts, a future MCP tool — goes through
directly), not a wrong threshold. Precedent for the fix shape: `MAX_CONFIG_LEN`'s ctx-merge bound
in `submit_workflow_input`/`sweep_due_workflow_runs` is already duplicated at the service layer for
exactly this reason (`services.py:2042`, `:2369`) — mirror that pattern here rather than inventing a
new one.

**5.1 — Add a shared length-check helper, called from three places.**

A small static/module-level helper (name suggestion: `_validate_key_lengths`, next to
`_validate_def_spec` in `services.py`) taking `key: str, version: str, steps: list[dict], transitions:
list[dict]` and raising `WorkflowDefSpecError` (the same exception `_validate_def_spec` already
raises — no new exception type needed) on the first violation found, checking:

- `len(key) > MAX_KEY_LEN` and `len(version) > MAX_KEY_LEN`
- `len(step["key"]) > MAX_KEY_LEN` for every step
- `len(tr["from"]) > MAX_KEY_LEN` and `len(tr["to"]) > MAX_KEY_LEN` for every transition

Import `MAX_KEY_LEN` from `.schemas` in `services.py` (it currently imports `MAX_CONFIG_LEN` from
the same module — same line, add the name). **Do not bound `tr["on"]`** — `analyst`'s Pass 1 nit on
the RCA is correct and still holds: this schema has zero `RELATIONSHIP`-type constraints
(`grep -n RELATIONSHIP scripts/bootstrap_schema.sh` → no matches), and `on` never feeds
`Step.stepUid`'s `MERGE` key the way `from`/`to` do — it isn't at risk of *this* crash class.
Bounding it anyway would be harmless but is scope creep against a design meant to close a specific
crash vector precisely; leave `WorkflowTransitionIn.on`'s existing pydantic-only bound as sufficient
defense-in-depth for that field alone.

**5.2 — Call it in `_validate_def_spec` (`services.py:1294`), first, before the existing checks.**

Run the length check **before** the `kind`/step-type/uniqueness checks currently at the top of the
function — a `WorkflowDefSpecError` for "key too long" is a more actionable message than whatever
downstream check would otherwise fire on truncatable-looking data, and it's cheap to check first.
This does **not** conflict with the function's own documented "five further invariants run last"
ordering rule (`services.py:1306-1323`) — that rule is about the five *domain*-semantic invariants
(K-024 U2 etc.) added after the original structural checks; a pure length bound is structural, like
the checks it's joining, not a new domain invariant.

`_validate_def_spec` is only called from `publish_workflow_def` today (`services.py:1602`) — but it
does not receive `key`/`version` as parameters (see its signature at `services.py:1294`: `kind`,
`steps`, `transitions` only). Two options, either acceptable:

- (a) extend `_validate_def_spec`'s signature to also take `key: str, version: str`, and update its
  one call site (`services.py:1602-1604`); or
- (b) call `_validate_key_lengths(key, version, steps, transitions)` as a separate line immediately
  before the `_validate_def_spec(...)` call in `publish_workflow_def` (`services.py:1602`).

(a) keeps a single validation entry point per def-spec (arguably cleaner); (b) is a smaller diff
against a function whose docstring already carefully documents check ordering. Either is fine —
`tdd-engineer`'s call, driven by whichever reads better against the surrounding code once written.

**5.3 — Close `materialize_def`'s gap (`services.py:1641-1682`).**

After `sub = self._repo.read_def_subgraph(...)` (`services.py:1662`) and before
`self._repo.materialize_snapshot(...)` (`services.py:1678`), call the same helper:
`_validate_key_lengths(key, version, sub["steps"], sub["transitions"])`. This guards the case a
def sitting in `reference` got there with an oversized key by some means other than
`publish_workflow_def` (hand-edited seed, future non-REST publish path, direct repository write) —
today believed unreachable in practice, but the point of this guard is exactly to remove "believed"
from that sentence. Raise the same `WorkflowDefSpecError` — `materialize_def`'s docstring already
documents `WorkflowDefNotFoundError`/`WorkflowDefConflictError` as its error surface; add
`WorkflowDefSpecError` to that list in the same change (it's already a documented possibility for
`publish_workflow_def`, so callers/tests already know the exception type).

**5.4 — Add the missing `Path()` bound on the materialize route (`api.py:369-373`).**

```python
@router.post("/workflow-defs/{key}/versions/{version}/materialize", status_code=201)
def materialize_def(
    key: str = Path(..., min_length=1, max_length=MAX_KEY_LEN),
    version: str = Path(..., min_length=1, max_length=MAX_KEY_LEN),
    ctx: CallContext = Depends(get_context),
):
    return services.materialize_def(ctx, key=key, version=version)
```

Matches the sibling `get_workflow_def_structure` route (`api.py:362-363`) exactly. `MAX_KEY_LEN` is
already imported in `api.py` (used by the sibling route), so no new import needed.

**5.5 — Optional, low-cost: document the engine cliff next to the constant.** A one-line comment at
`MAX_KEY_LEN`'s definition (`schemas.py:87`) recording that 200 is chosen with margin below a
confirmed 4096-byte engine-crash boundary (cite this document or the RCA) — so a future contributor
doesn't casually raise it "closer to 4096" without knowing the risk is a crash, not a truncation.
Not required for the fix to be complete; worth doing in the same diff since it's one comment.

**Scope check — every other `UNIQUE`-constrained property is safe by construction.** `userId`,
`agentId`, `channelId`, `threadId`, `msgId`, `documentId`, `chunkId`, `entityId`, `runId`,
`stepRunId`, `traceId`, `cursorId`, `workspaceConfigId` are all server-minted `uuid.uuid4().hex`
(fixed 32 hex chars — confirmed at `services.py:180` and `executor.py:96`), never caller-supplied
free text. `key`/`version`/step-keys are the *only* caller-supplied strings feeding a
constrained property in this schema today — no other field needs this guard.

## 6. Test strategy for U2

A service-layer regression test per guarded path (mirroring `test_workflow_timers.py`'s existing
`MAX_CONFIG_LEN` ctx-merge bound tests, which prove the guard rejects **before** any Cypher reaches
FalkorDB — no live-engine crash test belongs in the suite):

- `publish_workflow_def` raises `WorkflowDefSpecError` for an oversized `key`, `version`, step
  `key`, and transition `from`/`to` (four cases, or parametrized).
- `materialize_def` raises `WorkflowDefSpecError` when the *stored* `reference` def already carries
  an oversized key — needs a fixture that writes past the guard directly via the repository (not
  through `publish_workflow_def`), matching the "corrupted/hand-edited data already in `reference`"
  threat model from §5.3.
- REST: `POST .../materialize` with an oversized `{key}`/`{version}` path param returns `422`
  (pydantic/`Path()` validation), not a 500 or a hang.
- Nothing in this test strategy talks to a live FalkorDB with an oversized value — the isolated
  container repro in §1 (and the RCA before it) is the one-time proof; it must never become a
  standing CI step against any shared instance, `falkordb-dev` included.

## 7. Upstream FalkorDB issue — recommended, yes

Confirmed genuine engine defect, not an app design gap: any FalkorDB user with a `UNIQUE`
constraint on an unbounded-length property is exposed to a full-instance crash from a single write.
Recommended report contents (mirrors the `K-007 OQ6` "confirmed engine anomaly" precedent):

- **Engine:** `falkordb/falkordb:v4.18.11`, module version `41811`, Redis 8.6.3.
- **Defect:** `CREATE`/`MERGE` writing >4096 bytes into a `UNIQUE`-constrained property SIGSEGVs the
  entire `redis-server` process — null-pointer dereference in `EnforceUniqueEntity`, called from
  `Schema_EnforceConstraints` ← `CommitNewEntities` (the commit path).
- **Exact boundary:** 4096 bytes safe, 4097 crashes — deterministic, reproduced independently three
  times across three separate containers/sessions (this doc, the original RCA, `analyst`'s gate).
- **Scope:** constraint-enforcement path only (index-only, no constraint: safe to ≥1MB tested);
  per-property for a composite constraint; independent of write-clause shape.
- **Repro:** §1 above, or the RCA's §2, either is sufficient as filed evidence.

This is the user's call to file, not an agent action — flagging as a clear "yes, worth filing."

## 8. What I did not do

Did not implement any code change — this is a design document for `tdd-engineer`. Did not touch
`falkor-chat/AGENTS.md` (its stale `repository.py:1669`/`:992` citations are `tico`'s to fix per
the doc-flip-table, out of scope for a `graph-dba` deliverable). Did not touch
`claude/graph-dba/falkordb-quirks.md` — the durable engine quirk is already correctly recorded
there from the original RCA; nothing here refines or contradicts it, so there is nothing new to add
(root `AGENTS.md`'s "quirks capture" instruction is satisfied by the existing entry, not a new one).
Did not touch `docs/BACKLOG.md`'s K-049 text or the coordination doc — flagging the staleness in §0
is this document's job; editing either is `teco`'s per the coordination's own ownership.
