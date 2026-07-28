# K-031 — Def/snapshot **structure** read surface (make the create-only split-brain detectable)

> **Status:** archived · **Owner:** `architect` · **Tracks:** K-031 (M3 follow-ups) ·
> v2 2026-07-24; re-gated and delivered
>
> **Version:** **v2 — 2026-07-24** (revision pass after the `analyst` U2-G1 gate returned
> *needs changes*: 2 blocker · 3 major · 6 minor · 4 nit). v1 = 2026-07-24, design complete.
> Every finding's disposition is recorded in **§11**.
> Planning-only artifact — no code, DDL or doc was changed by writing it.
> **Review:** `docs/reviews/workflow-def-structure-read.md`.
> **Closes:** **K-031** (`docs/BACKLOG.md`), filed out of the K-025 QA pass as **DEF-1** +
> feedback item 5 (`docs/archive/test-reports/m3-workflow-engine-report.md` **§5 (DEF-1)** and
> **§7 items 1 and 5**).
> **Coordination:** `docs/plans/m3-followups-coordination.md` (U2 → U2-G1 → U3 → U3-G).
> **Explicitly NOT in scope:** changing publish semantics (create-only is a decision), **the
> additive-`MERGE` finding and every doc correction it implies (→ K-034, §0.2)**, converging the
> seed def sources (**K-029**), allowing zero-transition defs (**K-030**), any K-027 work, any MCP
> surface, repairing live def/snapshot divergence (**stakeholder decision, §6 R-1**).
> **Baselines:** `./scripts/test_queries.sh` **256/256** (must stay *exactly* 256 — §1.3);
> `server` pytest — **re-derive the entry count at U3 start**, do not trust a number written here
> (§1.3).

---

## 0. Headline findings (read these first)

1. **The "no new Cypher" assumption HOLDS — with one qualification. CONFIRMED at the gate.** Every
   byte of Cypher this plan needs already exists and is already verified:
   `repository._READ_META_CYPHER` (`:961`) and `_READ_TRANSITIONS_CYPHER` (`:968`) are
   `{label}`-templated and are already formatted with **both** `WorkflowDef` and
   `WorkflowDefSnapshot` by the single shared `_read_subgraph` (`:976`). No new query string, no
   modified query string, no DDL, no index, no property. **⇒ no `graph-dba` gate, no
   `test_queries.sh` delta.** The qualification is finding 2, which asks the implementer to read
   *more rows of the same query* than `_read_subgraph` does today — Python, not Cypher.
2. **A create-only re-publish is not purely a no-op — it is also *additive*. CONFIRMED at the gate,
   and it is now `K-034`, not K-031.** `_PUBLISH_CYPHER` (`repository.py:937-956`) is
   `MERGE … ON CREATE SET` on *properties*, but its `MERGE` **patterns** create structure: a new
   step key mints a `Step` + `HAS_STEP`; a changed `to`/`on`/`order` mints a **parallel
   `TRANSITION`**; a changed start step mints a **second `START`**. Only `guard`, `d.name`,
   `d.kind` and `st.type`/`st.config` are genuinely create-only. `materialize_snapshot` (`:1483`)
   shares the constant verbatim, so the snapshot side is additive too.
   **Scope (binding stakeholder decision, coordination log 2026-07-24):** the finding **leaves
   K-031 entirely** and is filed as its own backlog item, **K-034**, carrying the analyst's
   evidence — including the two live consequences this plan did *not* name (nondeterministic
   branch selection in `executor._select_transition`, and `repository.start_run`'s duplicate
   `CREATE` against the `UNIQUE NODE WorkflowRun … runId` constraint, `bootstrap_schema.sh:180`)
   and the ten shipped assertions it falsifies. **Do not restate that evidence here** — cite
   `docs/reviews/workflow-def-structure-read.md` §B-2. K-031 stays *"make the current semantics
   observable"*: the structure read is the **detection** mechanism for the state K-034 describes,
   and it must **cross-reference** K-034 rather than absorb, test, or document-correct it (§8).
3. **A def with two `START` edges makes today's structure read *silently ambiguous*.**
   `_READ_META_CYPHER` returns `start.key` as a non-aggregated grouping key next to
   `collect(DISTINCT …)`. QUERIES.md §11.2 states the one-row collapse holds *because* "`start.key`
   is constant across the fan-out" — with two `START` edges that premise fails and the query is
   expected to yield **one row per start key**, of which `_read_subgraph` takes `result_set[0]`
   arbitrarily. The observability endpoint must not inherit that blind spot; §3.1/U1 handles it by
   reading all rows. **This is an engine-semantics question and is verified live before U1 is
   written** — §7 **V-1**, now snapshot-side, isolated, torn down, and carrying a
   stop-and-escalate rule.
4. **The `maxSteps` off-by-one fix lands *inside* the SHA-locked `_drive_loop`** (`executor.py:410`
   OUTCOME A and `:427` OUTCOME C, both `rec["stepCount"] > max_steps`). **Stakeholder decision:
   DOCUMENT-ONLY in this slice** — `executor.py` is not touched, the lock `71055f756280` stays
   intact, `tests/test_executor.py:158` keeps its assertion. The real fix is filed as a
   **self-standing proposed K-033** (§5).

---

## 1. Goal & scope

### 1.1 Goal

Give an operator a **black-box, read-only** way to answer three questions that today require raw
Cypher:

| Question | New answer |
|---|---|
| *Is what I think is published actually published?* | `GET /workflow-defs/{key}/versions/{version}` — full structure: `startKey`, steps (`key`/`type`/`config`), transitions (`from`/`to`/`on`/`order`/`guard`) |
| *Is the workspace running the same thing?* | `GET /workspaces/{ws}/snapshots/{key}/versions/{version}` — the same shape, from `ws:{id}` |
| *Have `reference` and `ws:{id}` gone stale independently?* | `GET /workspaces/{ws}/snapshots/{key}/versions/{version}/diff` — one call, `inSync` + an enumerated difference list |

Plus `scripts/verify_workflows.sh <wsId>` turning the documented re-seed discipline into a
one-command, exit-coded check (QA feedback item 5).

### 1.2 Out of scope (do not drift into these)

- **Publish semantics.** No new validation, no `ON MATCH SET`, no repair path, no "fix the def"
  affordance. This item makes current semantics **observable**, nothing else. If the read exposes
  something ugly, the deliverable is a **report + a backlog entry**, never a fix.
- **The additive-`MERGE` finding (§0.2) — it is `K-034`.** No test of additive re-publish, no
  correction of the "immutable / no-op" prose in `AGENTS.md`, `DESIGN.md`, `QUERIES.md` §11 or the
  three docstrings. K-031 cites K-034; K-034 corrects them.
- **Repairing live def/snapshot divergence** — binding stakeholder decision (OQ-3, now §6 R-1).
- **K-029** (converge seed def sources), **K-030** (zero-transition defs), **K-027** (parse
  robustness — U1 of this same run, a different file set).
- **MCP.** `mcp.py` exposes no workflow tools at all today; the audience for a structure read is an
  operator/QA human, not an agent. Adding MCP tools here would be new surface with no consumer.
- **Retrofitting `response_model=` onto existing routes** (§3.4 explains why the new routes get one
  and the old ones are left alone).
- **`web/`** — no UI for this.

### 1.3 Suite expectations

- **`server` pytest — measure the entry baseline, do not assume one.** The 533 figure in the
  coordination log is the **pre-U1** baseline; U1 (K-027 slice A) is adding tests to the same tree
  in parallel. At U3 start, run the **non-mutating** `cd server && .venv/bin/python -m pytest
  --collect-only -q` and record the number; report the result as **`entry → entry + N`**
  (N ≈ 20–26 new tests, enumerated in §7). For reference, `--collect-only` reported **552 collected
  / 1 deselected** on 2026-07-24 with U1's churn in the tree — that number will have moved again.
  No existing test should need editing; if one does, that is a finding to report, not to absorb.
- `./scripts/test_queries.sh`: **256/256, unchanged.** This one *is* invariant and is the plan's
  best tripwire — a delta here means new Cypher landed. Stop and escalate (§6 R-4).

---

## 2. Context & findings (what the codebase already gives us)

| Fact | Evidence |
|---|---|
| The def subgraph read exists and is verified | `repository.py:1025 read_def_subgraph` → `_read_subgraph` (`:976`) → `_READ_META_CYPHER` (`:961`) / `_READ_TRANSITIONS_CYPHER` (`:968`); QUERIES.md §11.2 |
| The snapshot subgraph read exists too, same helper, different label | `repository.py:1498 get_snapshot`; QUERIES.md §11.5 |
| A **service** wrapper for the snapshot structure already exists — it is just not routed | `services.py:687 get_snapshot`; used by `executor._drive_loop`, `services.submit_workflow_input`, `scripts/seed_workflows.sh` |
| There is **no** service wrapper for the def subgraph; `read_def_subgraph` is called only from `materialize_def` | `services.py:663` |
| The REST surface stops at metadata | `api.py:228 GET /workflow-defs/{key}` → `get_def` → `{key, version, name, kind}` (`repository.py:1035`); `api.py:304 GET /workspaces/{ws}/snapshots` → `list_snapshots` |
| Steps/transitions come back **unordered** by design (F6 — no `length(path)` on this build) | QUERIES.md §11.2 note; `repository.py:958-960` comment |
| `config`/`guard` are always **strings** by the time they are stored (`None` → `""`) | `services._serialize_opaque` (`:146`) |
| A def publish has **no graph seam** — `publish_def` writes to a hardcoded global `reference` | `repository.py:1011` → `_reference()` (`:132-134`) → `db.reference_graph` (`db.py:87-94`), `select_graph("reference")`, no parameter/env/config override. **This is why V-1 runs snapshot-side (§7).** |
| `materialize_snapshot` formats *the same* `_PUBLISH_CYPHER` constant with `label="WorkflowDefSnapshot"` against a workspace graph | `repository.py:1470-1490` |
| Publish is create-only on **properties**, additive on **structure** | `_PUBLISH_CYPHER` (`repository.py:937-956`) — finding §0.2; **owned by K-034** |
| The publish/materialize receipt's `stepCount`/`transitionCount` count the **submitted spec**, not the stored subgraph | `_PUBLISH_CYPHER`: `WITH d, count(st) AS stepCount` sits immediately after `UNWIND $steps` (one row per unwound input element) — see §3.2 |
| Tenancy seam unchanged | `config.get_context`; `api.py:304` comment — the path `{ws}` is descriptive, `get_context` resolves the real workspace |
| No route in the codebase declares `response_model` | grep: zero hits in `server/falkorchat/*.py` |
| `api.py` imports **only** `MAX_ID_LEN` from `.schemas` — `MAX_KEY_LEN` (`schemas.py:42`) needs adding | `api.py:17-26` |
| Allowed workflow kinds are exactly `{conversation, process}` | `services.py:51 WORKFLOW_KINDS`, enforced at `:531` |
| A workspace bootstrap creates `Step.stepUid` index + constraint and the `WorkflowDefSnapshot` indexes (V-1 depends on this) | `scripts/bootstrap_schema.sh:112-120`, `:176-189` |
| "Per-endpoint response schemas" is standing QA feedback, raised three times | `docs/BACKLOG.md:793`; test reports `m1-chat-mcp-report.md:89`, `m1-hardening-regression-report.md:109`, `k007-m2-groundwork-report.md:140` |
| Error handlers already map `WorkflowDefNotFoundError → 404` | `app.py:86` |
| The budget check and its off-by-one | `executor.py:410` / `:427` (`rec["stepCount"] > max_steps`), pinned by `tests/test_executor.py:142-158`; the park path (OUTCOME B) is deliberately **not** budget-checked (`executor.py:415-421` comment) |

**The seam this change lands on is therefore purely adapter-layer:** two new repository methods that
*reuse existing query constants*, two service normalizers, one service comparator, three routes,
four response models. Nothing below `repository.py`'s query strings moves.

---

## 3. Design & rationale

### 3.1 Endpoint shape — a dedicated resource path, **not** `?expand=`

The backlog named three candidates. Decision and reasons:

| Candidate | Verdict |
|---|---|
| `GET /workflow-defs/{key}?expand=steps` (expand-on-demand) | **Rejected.** It makes *one* route return two different response shapes — the precise failure mode of the standing "field subsets vary" feedback (BACKLOG.md:793). It also collides with that route's *other* job: `{key}` with no version resolves **latest**, so `?expand=steps` on a version-less request would silently structure-read a version the caller never named — the worst possible affordance in a tool whose purpose is verifying *which* version is live. |
| `GET /workflow-defs/{key}/versions/{version}` (**chosen**) | The version-qualified def **is** a distinct resource, and the path already exists in the surface as the materialize parent (`POST /workflow-defs/{key}/versions/{version}/materialize`, `api.py:239`). One path, one shape, one declarable schema. Being forced to name the version is a feature here, not friction. |
| `GET /workflow-defs/{key}/versions/{version}/subgraph` (a `/subgraph` sibling) | **Rejected.** "Subgraph" is storage vocabulary leaking into the HTTP contract, and it implies a *second* representation of the same resource — inviting exactly the shape-drift the first option was rejected for. The version-qualified resource has no other representation to be confused with. |
| A dedicated diff endpoint | **Accepted *in addition***, not instead — see §3.3. |

**No `latest` alias.** An operator investigating a version mismatch must state the version; they get
it from the existing `GET /workflow-defs/{key}` (metadata + latest resolution) or
`GET /workflow-defs` (list). Documented in DESIGN §14.4.

**Snapshot mirror:** `GET /workspaces/{ws}/snapshots/{key}/versions/{version}`, extending the
existing `GET /workspaces/{ws}/snapshots` list route and keeping `{ws}` descriptive exactly as that
route documents (`api.py:310-312`). The response shape is **identical** to the def route's except
for the `source` field — deliberate, so a human or `jq` can diff the two bodies directly if they
prefer not to use the server-side diff.

**Multi-`START` handling (finding 3).** The new repository readers consume **all** rows of
`_READ_META_CYPHER`, not `result_set[0]`. The response carries `startKey` (the first, for the common
single-row case) **and** `startKeys: [...]` whenever more than one distinct start key comes back;
`startKeys` is omitted when there is exactly one. Rationale: the state K-034 describes can *produce*
a two-`START` def, and a read surface that hides it would be worse than no read surface. Cost:
~10 lines of Python, zero Cypher. **V-1 (§7) settles the row behaviour before U1 is written**, and
the design is defined for both outcomes.

### 3.2 Response contract (the "don't make the schema problem worse" clause)

Both structure routes return **exactly** this shape, declared as a Pydantic model and wired with
`response_model=`:

```jsonc
// GET /workflow-defs/{key}/versions/{version}
// GET /workspaces/{ws}/snapshots/{key}/versions/{version}
{
  "source": "reference",            // "reference" | "workspace"
  "key": "access-request",
  "version": "v1",
  "name": "Access request",
  "kind": "process",
  "startKey": "submit",
  "startKeys": ["submit", "intake"],   // OMITTED unless > 1 (see §3.1)
  "stepCount": 6,
  "transitionCount": 6,
  "steps": [                        // sorted by key (§3.5)
    {"key": "approval", "type": "human", "config": "{\"waitsForHuman\":true}"}
  ],
  "transitions": [                  // sorted by (from, order, to, on) (§3.5)
    {"from": "route", "to": "approval", "on": "needs_approval",
     "order": 0, "guard": "{\"kind\":\"cmp\",...}"}
  ]
}
```

- `config`/`guard` are returned **verbatim as opaque strings** (AGENTS.md rule 8) — never parsed,
  never re-serialized, never pretty-printed. Byte fidelity is the whole point: a diff that
  round-trips JSON would hide a whitespace-only divergence.
- **`stepCount`/`transitionCount` count what is *stored*, and that is not the same thing as the
  publish receipt** (gate M-1). They are derived app-side (`len(...)`) from the structure the graph
  returned. The identically-named fields on the publish/materialize response come from
  `_PUBLISH_CYPHER`'s `count(st)`/`count(rel)` **immediately after the `UNWIND`s**, i.e. they count
  the *submitted* spec (`len($steps)`/`len($transitions)`), never the stored subgraph. The names are
  kept because they are the right names — but the docs must state the relationship plainly:

  > The publish/materialize receipt counts what was **submitted**; the structure read counts what is
  > **stored**. **A divergence between the two is a signal, not an endpoint bug** — see K-034.

  This sentence goes in DESIGN's §14.4 paragraph (§8) as well as here. Getting it backwards is
  exactly wrong in the one case that matters: after an additive re-publish the receipt says `6` and
  the structure read says `7`.
- Field naming is **camelCase**, matching every other response body (`stepCount`, `atStepKey`,
  `defKey`). The repository's `start_key` is renamed at the service boundary. **Do not change
  `services.get_snapshot`'s existing `start_key` key** — `executor._drive_loop` (SHA-locked) and
  `services.submit_workflow_input` read it. New methods, new names (§3.6).
- **404** when the root node is absent (`WorkflowDefNotFoundError` for the def route; a plain
  `HTTPException(404)` for the snapshot route, mirroring `get_workflow_def`'s style at `api.py:236`).

### 3.3 The diff — **server-side**, under `/workspaces/…`, and why

**Decision (was OQ-2, now closed as an architect call — gate n-1 — and separately confirmed by the
stakeholder):** server-side, at
`GET /workspaces/{ws}/snapshots/{key}/versions/{version}/diff`. The comparison is inherently
workspace-scoped (the snapshot side is the workspace's), and `{ws}` already exists as a descriptive
segment on `list_snapshots` (`api.py:304-312`). The alternative —
`/workflow-defs/{key}/versions/{version}/diff` — would hang a workspace-dependent answer off a
global resource path; rejected on that ground alone.

Client-side (two reads + compare) was the cheaper option and is rejected on three concrete grounds:

1. **Equality here is not JSON equality.** Steps and transitions are unordered at the source (F6);
   a naive client comparison of two arrays reports false divergences on ordering alone. Canonical
   ordering + identity-keyed pairing is *real logic* (§3.5), and duplicating it in every client,
   script and QA pass is how the next split-brain gets missed.
2. **Identity pairing is non-obvious.** A transition's identity is the 4-tuple
   `(from, to, on, order)` — because that is what `_PUBLISH_CYPHER`'s `MERGE` pattern makes unique —
   and its only comparable payload is `guard`. A client author would plausibly key on `(from, to)`
   and mis-report an added parallel edge as a modified one. The server knows the write path's
   identity rules; clients should not have to.
3. **The backlog and the QA feedback both asked for "in one call"** (BACKLOG K-031 scope sketch;
   report §7 item 1). Two reads plus client logic is not that, and `verify_workflows.sh` would have
   to re-implement the comparator in bash/Python anyway.

Response contract:

```jsonc
// GET /workspaces/{ws}/snapshots/{key}/versions/{version}/diff
{
  "key": "access-request",
  "version": "v1",
  "defPresent": true,
  "snapshotPresent": true,
  "inSync": false,
  "differences": [
    {"path": "meta.name",                       "def": "Access request", "snapshot": "Access req"},
    {"path": "steps[escalate]",                 "def": "present",        "snapshot": "absent"},
    {"path": "steps[approval].config",          "def": "{\"waitsFo…",    "snapshot": "{\"waitsFo…"},
    {"path": "transitions[route->deny@rejected#1].guard",
                                                "def": "{\"kind\"…",     "snapshot": null}
  ],
  "differenceCount": 4
}
```

Decisions inside the diff:

- **`inSync` is the one-glance answer**; `differences` is the evidence. `inSync == (differenceCount
  == 0 and defPresent and snapshotPresent)`.
- **One side missing is a 200, not a 404.** `defPresent: false, snapshotPresent: true` is *the*
  documented trap after a `pytest` or `test_queries.sh` run wipes `reference` while `ws:{id}`
  survives — it is a first-class reportable state, and returning an error for it would push the
  operator straight back to raw Cypher. **Both** sides missing → **404** (`WorkflowDefNotFoundError`,
  already mapped at `app.py:86`); `differences` is `[]` in the one-sided case and the two presence
  flags carry the whole story.
- **Difference values are previews, not payloads** — `config`/`guard` values are truncated to
  `MAX_DIFF_PREVIEW = 200` chars with a `…` suffix. This is the diff endpoint's size bound (§3.7):
  the response is O(differences), never O(def). An operator who needs the full value reads the two
  structure endpoints.
- `path` grammar is stable and documented: `meta.<field>` · `steps[<key>]` (presence) ·
  `steps[<key>].<type|config>` · `transitions[<from>-><to>@<on>#<order>]` (presence) ·
  `transitions[…].guard`. Presence differences use the literal strings `"present"`/`"absent"`.
- **`startKey` is compared as part of `meta`**, and a multi-`START` def surfaces as
  `meta.startKeys` with the two lists.
- **Direction is named, not implied.** `def` = `reference`, `snapshot` = `ws:{id}`. The docs must
  say plainly: **the snapshot is what the executor drives** (`executor._drive_loop` → `get_snapshot`),
  so `snapshot` is the operational truth and `def` is the intended truth.
- **The diff is version-qualified — it answers "same version, different content", not "wrong
  version"** (gate m-1). It cannot detect *"`reference` now has `v2`, the workspace only ever
  materialized `v1`"*, which is the shape a `key`/`version` bump produces (the documented way to
  land a def edit). One sentence in the DESIGN §14.4 paragraph **and** in the route's own comment:
  *"to detect a stale **version**, compare `GET /workflow-defs` against `GET /workspaces/{ws}/snapshots`
  first; this route compares content within one named version."* `verify_workflows.sh` covers the
  version case for the two seeded defs because it reads the expected version from
  `config`/`proof_defs` (§3.8).

### 3.4 Response models: new routes only

The three new routes declare `response_model=` (`WorkflowDefStructureOut`, `WorkflowDiffOut`,
`WorkflowDiffEntry`). Existing routes are **not** retrofitted.

Rationale: the standing feedback is real, but retrofitting it is a cross-cutting change with
behavioural risk (FastAPI's `response_model` *filters* undeclared fields — a wrong model silently
drops a field the web client reads), and it is not what K-031 was filed for. Declaring schemas on
the new surface satisfies "don't make it worse" and gives the eventual retrofit a worked precedent.
**Note for the backlog:** this leaves the repo with a mixed convention; record it on the parking-lot
item at BACKLOG.md:793 rather than pretending it is closed.

### 3.5 Canonical form (where normalization lives)

In `services.py`, **not** `repository.py`. The repository stays a 1:1 mirror of QUERIES.md
(AGENTS.md layering); ordering is app-side by explicit design (F6 says "the app reconstructs order").

- Steps sorted by `key`.
- Transitions sorted by `(from, order, to, on)`.
- Comparison of `config`/`guard`: **exact string equality**, no JSON normalization (rule 8 — the
  bytes in the graph are the contract; `_serialize_opaque` already guarantees a stable key order for
  dict-shaped input, and a byte difference from any other source is a real difference).
- Step identity: `key`. Transition identity: `(from, to, on, order)`.

Deterministic ordering is also what makes the structure endpoints diffable by hand (`curl … | jq`)
and the contract tests stable.

### 3.6 Layering & naming (locked conventions honoured)

```
api.py                      services.py                          repository.py                  Cypher
GET /workflow-defs/{k}/     get_workflow_def_structure(ctx,…) →   read_def_structure(…)      →  _READ_META_CYPHER      (QUERIES §11.2, EXISTING)
    versions/{v}                                                                                _READ_TRANSITIONS_CYPHER
GET /workspaces/{ws}/       get_snapshot_structure(ctx,…)    →    read_snapshot_structure(…) →  same, label=WorkflowDefSnapshot (§11.5, EXISTING)
    snapshots/{k}/versions/{v}
…/diff                      diff_def_snapshot(ctx,…)         →    both of the above
```

`read_def_subgraph`, `get_snapshot`, `_read_subgraph` and `_PUBLISH_CYPHER` are **left byte-identical**
— they are on the materialize and executor paths. The ~12 lines of duplication between
`_read_subgraph` and the new `_read_structure` are deliberate and must carry a comment saying so.

**Documented shape divergence (gate m-6).** `_read_structure` returns `start_keys: list[str]` where
QUERIES.md §11.2's documented contract is a scalar `startKey`. That is a (small, deliberate) stretch
of "`repository.py` is a 1:1 mirror of QUERIES.md" (DESIGN §14.2). It is mitigated on both sides:
the §11.2 **note** (§8) documents it from the query side, and `_read_structure`'s docstring must
cite that note **by name** and say *why the shape differs* — not only why the helper duplicates
`_read_subgraph`.

### 3.7 Bounding the response (AGENTS.md rule 6 / K-031 risk line)

**RAM call-out, stated explicitly as the rule requires: this change adds no node type, no label, no
property, no index and no vector dimension. Per-workspace RAM is unchanged. Zero graph-RAM cost.**
The only size question is the HTTP response, bounded as follows:

| Surface | Bound | Mechanism |
|---|---|---|
| Structure reads | Whole-object, **unpaginated** | Inherited from the publish boundary: `schemas.py` caps a REST-published def at `MAX_STEPS = 200`, `MAX_TRANSITIONS = 500`, `MAX_CONFIG_LEN = 8000` ⇒ ≈ 5.6 MB absolute worst case; the two real defs are ~2–4 KB. Matches the precedent set by the other §12 RO reads (`GET /workflow-runs/{id}/step-runs`, `/trace`), which take **no** `limit` either and are bounded upstream by `MAX_RUN_STEPS = 50`. |
| Diff | O(number of differences), values ≤ 200 chars | `MAX_DIFF_PREVIEW` truncation (§3.3) |

**Why no `?limit=`:** a truncated subgraph is actively harmful for the use case — an operator who
gets 50 of 60 steps back concludes "10 steps are missing from the snapshot". Partial structure is a
worse answer than a big one. **Honest residual:** service-layer publishers (`seed_workflows.sh`,
`proof_defs.py`, tests) bypass Pydantic, so the caps above are not universal; documented as an
accepted limitation, *not* fixed here (adding a publish-time size cap would change publish
semantics — out of scope).

### 3.8 `scripts/verify_workflows.sh <wsId>` — **recommended, include it**

Include it. It is the QA-requested (report §7 item 5) closing move that converts the trap from
documented to checkable, and it is ~60 lines once the comparator exists.

- **Drive the service layer via a Python one-shot**, exactly mirroring `scripts/seed_workflows.sh`'s
  structure (`server/.venv/bin/python - <<'PY'`), **not** HTTP. Reason: it must work with no uvicorn
  running — the moment it is most needed is right after a `pytest`/`test_queries.sh` run, before the
  server is back up.
- **Strictly read-only.** No publish, no materialize, no fallback "let me just re-seed that for
  you". If a def is missing it prints the `seed_workflows.sh` command and exits non-zero.
- Checks, per def in the same `DEFS` set `seed_workflows.sh` uses (`triage@v1` from
  `config.TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION`; `access-request@v1` from
  `proof_defs.ACCESS_REQUEST_DEF`):
  1. def present in `reference` **at the expected version** (this is the version-staleness check the
     diff endpoint structurally cannot do — §3.3, gate m-1),
  2. snapshot present in `ws:<id>` at that version,
  3. `inSync` per the comparator, printing each difference otherwise,
  4. `startKeys` singular (the finding-3 tripwire).
- Exit **0** = all green; **1** = anything missing or divergent. One table on stdout.
- `AGENTS.md`'s script table gains a row; the `test_queries.sh` and `seed_workflows.sh` rows gain a
  pointer to it.

### 3.9 The `maxSteps` off-by-one — see §5 (it is a decision, not a step)

---

## 4. Step-by-step implementation

Six units, each independently green. **U1→U2→U3** is the spine; **U4** depends on U2; **U5** on U4;
**U6** closes. Nothing here touches `executor.py`, `_PUBLISH_CYPHER`, `bootstrap_schema.sh`,
`QUERIES.md` query bodies, or `web/`.

**Before U1: run V-1 (§7).** It is a live write in a throwaway graph; its outcome decides whether
the multi-row path is live code or a dormant tripwire, and any surprising outcome is a
stop-and-escalate.

### U1 — repository: structure readers (no new Cypher)

**File:** `server/falkorchat/repository.py` (add after `_read_subgraph`, `:997`).

1. Add `_read_structure(graph, *, label, key, version) -> dict | None`:
   - runs `_READ_META_CYPHER.format(label=label)` and `_READ_TRANSITIONS_CYPHER.format(label=label)`
     — **the existing constants, unmodified**;
   - `None` when the meta result set is empty;
   - iterates **all** meta rows (not `result_set[0]`), collecting distinct non-null `startKey`s into
     `start_keys` and unioning the `steps` lists (de-duplicated on `key`, preserving first-seen
     values); `name`/`kind` come from row 0 (they are identical across rows by construction — they
     are properties of the single root node);
   - returns `{name, kind, start_keys: list[str], steps: list[dict], transitions: list[dict]}`.
   - Carries a docstring/comment covering three things: *why* it duplicates `_read_subgraph` (that
     one feeds materialize + the SHA-locked executor and must not move), *why* it reads every row
     (finding 3 / QUERIES §11.2's "start.key is constant" premise, with a pointer to **K-034** for
     how a two-`START` def arises), and *why the returned shape differs* from §11.2's documented
     scalar `startKey` — citing the §11.2 note by name (§3.6, gate m-6).
2. `read_def_structure(*, key, version)` → `_read_structure(self._reference(), label="WorkflowDef", …)`.
3. `read_snapshot_structure(ws, *, key, version)` → `_read_structure(self._graph(ws),
   label="WorkflowDefSnapshot", …)`.

**Done when:** both return the full structure for a published/materialized def; `None` for an absent
one; `read_def_subgraph`/`get_snapshot` are unchanged (assert by diff review).

### U2 — services: normalization + the two structure reads

**File:** `server/falkorchat/services.py` (add after `get_snapshot`, `:697`).

1. Module-level `_canonical_structure(raw: dict, *, source: str, key: str, version: str) -> dict`:
   sorts (§3.5), renames `start_key(s)` → `startKey`/`startKeys` (omit `startKeys` when
   `len(start_keys) <= 1`), adds `source`/`key`/`version`/`stepCount`/`transitionCount`.
2. `get_workflow_def_structure(ctx, *, key, version) -> dict` — raises
   `WorkflowDefNotFoundError` when the repo returns `None` (message names the key@version and points
   at `POST /workflow-defs`), consistent with `materialize_def` (`:665`). Global read, no `ctx.ws`.
3. `get_snapshot_structure(ctx, *, key, version) -> dict | None` — returns `None` when absent (the
   route 404s), matching `get_workflow_def`'s passthrough style. Uses `ctx.ws`.

**Done when:** service-level tests (offline, `FakeRepo`) pin ordering, camelCase renaming, counts,
`startKeys` omission/inclusion, and the `ctx.ws` seam.

### U3 — REST routes + response models

**Files:** `server/falkorchat/schemas.py`, `server/falkorchat/api.py`.

1. `schemas.py`: `WorkflowStepOut`, `WorkflowTransitionOut` (note `from` is a Python keyword —
   follow the existing `from_` + `alias="from"` pattern at `:58`, and serialize `by_alias=True`),
   `WorkflowDefStructureOut`, `WorkflowDiffEntry`, `WorkflowDiffOut`. Add `MAX_DIFF_PREVIEW = 200`.
2. `api.py`, in the existing §11 block (after `get_workflow_def`, `:237`):
   - **Add `MAX_KEY_LEN` to the `from .schemas import (…)` block** (`api.py:17-26` currently imports
     only `MAX_ID_LEN`; `MAX_KEY_LEN` is `schemas.py:42`) — gate m-5.
   - `GET /workflow-defs/{key}/versions/{version}` → `services.get_workflow_def_structure`,
     `response_model=WorkflowDefStructureOut`. Path params bounded with
     `Path(..., min_length=1, max_length=MAX_KEY_LEN)` — matching the `MAX_ID_LEN` bounding style
     already used on the run routes (`:270`).
   - `GET /workspaces/{ws}/snapshots/{key}/versions/{version}` → `get_snapshot_structure`;
     `None` → `HTTPException(404, "workflow snapshot not found")`. Place it beside the existing
     `list_snapshots` route (`:304`) and repeat that route's `{ws}`-is-descriptive comment.
3. Confirm route-ordering: the static `web/` mount is registered **last** in `app.py` and these are
   `router` routes — no ordering change needed. (State it; do not modify `app.py`.)

**Done when:** API tests cover 200-with-full-structure, 404 both routes, and that a
`config`/`guard` string comes back byte-identical to what was published.

### U4 — the diff

**Files:** `server/falkorchat/services.py`, `api.py`, `schemas.py`.

1. `services._diff_structures(def_s, snap_s) -> list[dict]` — pure function over two canonical
   structures, implementing §3.3/§3.5 (meta fields, step presence + `type`/`config`, transition
   presence keyed on the 4-tuple + `guard`), values preview-truncated.
2. `services.diff_def_snapshot(ctx, *, key, version) -> dict` — reads both (def globally, snapshot
   from `ctx.ws`), raises `WorkflowDefNotFoundError` when **both** are absent, otherwise returns the
   §3.3 envelope.
3. `api.py`: `GET /workspaces/{ws}/snapshots/{key}/versions/{version}/diff`,
   `response_model=WorkflowDiffOut`. Its route comment carries the version-qualification caveat
   (§3.3, gate m-1).

**Done when:** the divergence fixture (§7) shows every difference class; identical def/snapshot ⇒
`inSync: true, differences: []`; one-sided ⇒ 200 with the presence flags; both absent ⇒ 404.

### U5 — `scripts/verify_workflows.sh`

**File:** `scripts/verify_workflows.sh` (new, `chmod +x`), modelled line-for-line on
`scripts/seed_workflows.sh`'s venv/one-shot preamble. Behaviour per §3.8. Read-only; no writes of
any kind.

**Done when:** it exits 0 on a freshly seeded workspace, exits 1 with a readable table after a
`pytest` run has wiped `reference`, and never writes.

### U6 — documentation (mandatory, part of done)

See §8 for the full list, including the `maxSteps` decision text and the K-034 cross-references.

---

## 5. The `maxSteps` off-by-one — decision and costing

**Stakeholder decision (binding, coordination log 2026-07-24): DOCUMENT-ONLY.** `executor.py:410`
and `:427` are **not** changed; the `_drive_loop` SHA lock (`71055f756280`) stays intact;
`tests/test_executor.py:158` keeps its current assertion.

**The defect (confirmed by reading, and *pinned by a passing test*):** `_drive_loop` records a step,
then checks `rec["stepCount"] > max_steps` (`executor.py:410` on OUTCOME A, `:427` on OUTCOME C).
With `maxSteps = 2`: step 1 → `1 > 2` false; step 2 → `2 > 2` false; **step 3 runs**, `3 > 2` →
fail. So `maxSteps` means *"at least N, then one more"*. `tests/test_executor.py:158` asserts
exactly this (`assert len(trail) == 4  # maxSteps=3 → the 4th advance trips the guard`), and
QUERIES.md §12.5 documents the comparison as `stepCount > maxSteps`.

**Why not fix it here** (the costing that produced the decision):

| Cost | Detail |
|---|---|
| Lock break + re-lock ceremony | Recompute the SHA, then re-quote it in `falkor-chat/AGENTS.md:256`, `docs/BACKLOG.md:25` + `:241`, `docs/HISTORY.md:144` + `:291` |
| **Frozen archive documents cannot be rewritten** | `docs/archive/plans/m3-process-flow.md` (×4), `docs/archive/reviews/m3-process-flow.md` (×5), `docs/archive/plans/m3-executor-coordination.md` (×3), `m3-process-flow-coordination.md` — these are *historical records* asserting the SHA was unchanged **throughout K-024**. Any re-lock has to be expressed as "as of K-0xx the lock is `<new>`; archived records quote the pre-K-0xx value", i.e. the lock stops being a single grep-able constant |
| Test edits | `tests/test_executor.py:142-158` (pinned count 4 → 3, and its explanatory comment) — plus a sweep of `test_process_flow.py`'s step accounting and the `access-request@v1` `maxSteps: 24` headroom |
| Behavioural blast radius | Every run's effective budget shrinks by one. Harmless for the two proof defs (8/6/6 steps against 24) but it is a live semantic change shipped inside an **observability** slice |
| Risk profile | Changes this slice from "read-only, zero-behaviour-change, no gate" to "touches the engine's locked hot loop" — which would also re-open the question of whether K-031 still needs an executor-competent reviewer |

**The documentation text to land** (at the six sites listed in §8):

> `maxSteps` is a **runaway tripwire checked *after* each recorded step**, not a hard cap: a run
> executes at most **`maxSteps + 1`** steps before failing with `"step budget exceeded"`. The check
> runs only on the two driving outcomes — a guard fired (OUTCOME A, `executor.py:410`) and a
> legitimate self-loop (OUTCOME C, `:427`). It is **deliberately not applied on the park path**
> (OUTCOME B — a parked run cannot self-drive; see the comment at `executor.py:415-421`) **or on the
> terminal path**. Treat it as a safety bound, not an SLA or a cost budget.

The "+1" alone would misread as the whole story; the outcome-by-outcome sentence is what makes it
actionable (gate answer 5).

**And file the fix, self-standing** (gate M-2): a new proposed backlog item —
***K-033 — make `maxSteps` an exact cap (`>` → `>=` in `_drive_loop`)*** — filed as a **complete item
on its own terms**, not as a rider conditional on another item. Bundling is a **preference**, not a
precondition, and its premise is **unverified**:

> Prefer to land K-033 alongside the next item that breaks the `_drive_loop` lock — plausibly
> **K-027 item 2** (the terminal-node-must-post engine contract) — so one lock break buys one
> re-lock ceremony and two fixes. **Unverified:** that K-027 item 2 *must* touch `_drive_loop` is an
> assumption, not an established fact — the lock covers `_drive_loop` only, and `_execute_step`,
> `_select_transition`, `_trace_step` and `resume` sit outside it (`AGENTS.md:256-257`), so a
> terminal-post guarantee might be implementable at one of those seams. K-027 is 🔵 proposed and
> unscheduled (`BACKLOG.md:315`). If no such item arrives, **K-033 breaks the lock on its own** —
> the ceremony is the cost either way.

Without that framing, K-033 orphans indefinitely while the honest `maxSteps + 1` prose sits in six
documents — the cheap half of a permanent divergence.

---

## 6. Risks

| # | Risk | Mitigation |
|---|---|---|
| **R-1** | **The read exposes ugly truth on the two live defs** — `triage@v1` in `reference` diverging from `ws:acme`'s snapshot, or duplicate `TRANSITION`/`START` edges from a past re-seed. Tempting to "just fix it". | **BINDING STAKEHOLDER DECISION (OQ-3, closed): repairing live def/snapshot divergence is OUT OF SCOPE. The implementer reports and files it — never repairs it.** No delete-and-republish, no version bump, no "helpful" re-seed of a divergent def. Deleting a snapshot breaks live `WorkflowRun`s that point at it via `OF_DEF`/`AT_STEP` — a destructive shared-state operation needing its own item and its own approval. Deliverable if it fires: the endpoint's verbatim output recorded in `docs/HISTORY.md`, plus a new backlog item describing the divergence (cross-referencing **K-034** if the shape is the additive one). |
| **R-2** | Finding 3's multi-row hypothesis is wrong on this build (the engine collapses differently). | **V-1 (§7) verifies it before U1 is written**, snapshot-side and torn down. The design works either way: if only one row ever returns, `startKeys` is simply never emitted and the code path is a cheap tripwire. **Any outcome other than "N rows for N distinct start keys" is a stop-and-escalate**, not an improvised design change (§7 V-1). |
| **R-3** | `response_model` filtering silently drops a field. | Only new routes are modelled; contract tests assert exact key sets on both structure routes and the diff. |
| **R-4** | Scope creep into new Cypher (e.g. "while I'm here, let me index X" or "let me count START edges properly in Cypher"). | Hard stop: if the implementer believes new/changed Cypher is required, **stop and escalate to teco** — it re-opens the owner chain (graph-dba gate + `test_queries.sh` assertions + a QUERIES.md §11.7). The suite staying at exactly **256/256** is the tripwire. |
| **R-5** | `verify_workflows.sh` acquires a "helpful" re-seed fallback and becomes a write path. | Stated as a hard constraint in §3.8 and in the script's own header comment; the docs must say it is read-only. |
| **R-6** | The def read is a **global** (`reference`) read exposed on a workspace-scoped API with no auth (M2.5). | Pre-existing and unchanged — `GET /workflow-defs` already exposes the global list. No new exposure class; note it, don't solve it (K-016). |
| **R-7** | Response size on a pathological service-published def (no Pydantic caps). | Accepted and documented (§3.7). Real defs are single-digit KB. |
| **R-8** | Doc drift: several documents describe the create-only trap, and K-034 will be correcting a neighbouring set of claims in parallel. | §8 is the list of docs **K-031 must touch**; it is deliberately **not** claimed to be exhaustive of everything the additive finding invalidates — that set belongs to K-034. Done-condition, verified at integration: `grep -rn -i "immutab\|no-op" docs/ server/falkorchat/ falkor-chat/AGENTS.md` and confirm every hit is either (a) already correct, (b) touched by this plan's §8, or (c) explicitly on K-034's list. Anything in none of the three is a report to teco. |
| **R-9** | K-034 has not been filed yet when U6 writes its cross-references. | The coordinator files K-034 (with the analyst's evidence) before/alongside U3. If the item is not in `docs/BACKLOG.md` when U6 runs, the implementer writes the cross-reference as **"K-034 (filing in flight)"** and **reports it** — it does not file K-034 itself, and it does not silently drop the reference. |

**RAM (AGENTS.md rule 6), stated explicitly:** **no impact.** No new node type, label, property, index
or vector dimension; both structure reads are `GRAPH.RO_QUERY` against existing indexed anchors
(`Node By Index Scan | (d:WorkflowDef)` / `(snap:WorkflowDefSnapshot)`, per QUERIES.md §11.2/§11.5's
verified plans) — no re-PROFILE needed because the query text is unchanged.

---

## 7. Test strategy

The backlog's three tests are the **floor**; this is the full list. All offline (FalkorDB only —
the repository/API tests use the existing `wf_repo`/`wf_client` fixtures at `tests/test_api.py:400`,
which wipe `ws:test` **and** `reference`, `conftest.py:86-93`). No `live` marker anywhere in this
slice.

### V-1 — live verification to run *before* U1 (~10 minutes). **This is a WRITE**, in a throwaway workspace graph, torn down.

**Why the snapshot side, not the def side (gate B-1).** `publish_def` (`repository.py:1011`) writes
to `self._reference()` → `db.reference_graph` (`db.py:87-94`) = `select_graph("reference")` — a
hardcoded literal with **no** parameter, env var or config override. *There is no such thing as a
per-workspace def publish*, so "an isolated throwaway workspace" cannot exist for a `WorkflowDef`.
`materialize_snapshot` (`repository.py:1470-1490`) formats **the same `_PUBLISH_CYPHER` constant**
with `label="WorkflowDefSnapshot"` against `self._graph(ws)`, and `_READ_META_CYPHER` is the same
constant formatted with the same label. **The query text is identical, so the answer transfers to
the `WorkflowDef` side unchanged — do not re-litigate this.**

Procedure:

1. `./scripts/bootstrap_schema.sh k031probe` — the `Step.stepUid` index + constraint and the
   `WorkflowDefSnapshot.key`/`.version` indexes must exist (`bootstrap_schema.sh:112-120`,
   `:176-189`; AGENTS.md: every `MERGE` is constraint-backed). **Note:** the script always re-runs
   `bootstrap_reference` (`:239`) — that is **idempotent DDL only, no data**, and is V-1's *only*
   interaction with `reference`. Nothing is published into `reference` at any point.
2. A Python one-shot on `server/.venv/bin/python`, building the repository directly
   (`Repository(db.connect())`) — deliberately below `services`, since service-level validation is
   not what is under test:
   - call 1 — `repo.materialize_snapshot("k031probe", key="probe", version="v1", name="probe",
     kind="process", start_key="a", steps=[{"key":"a","type":"decision","config":""},
     {"key":"b","type":"decision","config":""}],
     transitions=[{"from":"a","to":"b","on":"go","order":0,"guard":""}])`
   - call 2 — identical **except `start_key="b"`**.
   - ⚠️ **At least one transition is mandatory.** `_PUBLISH_CYPHER` ends in `UNWIND $transitions`;
     an empty list collapses the row stream and `materialize_snapshot`'s `res.result_set[0]` raises
     `IndexError` (the K-030 / K-024 O-6 shape). The single transition above is the minimum.
3. Read the **raw** result set:
   `db.workspace_graph(conn, "k031probe").ro_query(Repository._READ_META_CYPHER.format(label="WorkflowDefSnapshot"),
   {"key": "probe", "version": "v1"}).result_set` — print `len(result_set)` and each row's
   `startKey`. Also print the edge count:
   `MATCH (d:WorkflowDefSnapshot {key:'probe', version:'v1'})-[r:START]->() RETURN count(r)`.
   **Expected: 2 `START` edges and 2 meta rows with start keys `{a, b}`.**
4. Teardown: `redis-cli -h 127.0.0.1 -p 6379 GRAPH.DELETE ws:k031probe` (the same teardown
   `scripts/test_queries.sh:1059` uses), then confirm the graph is gone.

**Isolation contract:** V-1 must never write to `reference`, `ws:acme` or `ws:test`.

**Escalation rule (gate m-4) — this is an engine-semantics question, not a design detail.** Record
the outcome, do not adapt to it: **any** result other than *"N meta rows for N distinct start keys"*
— one row carrying an arbitrary `startKey`, an error, N rows for one `START` edge, or a `START`
count other than 2 — is a **stop-and-escalate to teco / `graph-dba`**, not an adjustment the
implementer absorbs. Same shape as R-4's tripwire. Record whatever V-1 returns in the U3 delivery
notes and, if it is a durable engine fact, append it to the `graph-dba` learnings inbox for
`claude/graph-dba/falkordb-quirks.md`.

### Repository (`tests/test_repository.py`)

1. `read_def_structure` returns exact steps + transitions + `start_keys` for a published def.
2. `read_snapshot_structure` mirrors it after materialize.
3. Both return `None` for an absent key/version.
4. `read_def_subgraph`/`get_snapshot` behaviour unchanged (existing tests must pass untouched — this
   is the regression fence around the materialize/executor path).

### Services (`tests/test_services.py`, `FakeRepo`)

5. Canonical ordering: steps sorted by `key`, transitions by `(from, order, to, on)` — feed the fake
   a deliberately shuffled structure.
6. `startKeys` **omitted** for one start key, **present** for two.
7. `ctx.ws` seam: the snapshot read uses `ctx.ws`, the def read passes no `ws` at all.
8. `WorkflowDefNotFoundError` on an absent def structure.
9. `_diff_structures` unit table: identical ⇒ `[]`; each difference class in isolation (meta field,
   step-only-in-def, step-only-in-snapshot, step `type`, step `config`, transition presence,
   transition `guard`, `startKeys`); preview truncation at `MAX_DIFF_PREVIEW`.

### API contract (`tests/test_api.py`)

10. **The backlog's test 1 —** publish `DEF_BODY`, `GET /workflow-defs/onboarding/versions/1`, and
    assert the response equals the published spec **exactly**: `startKey`, every step's
    `key`/`type`/`config` (byte-identical opaque strings), every transition's
    `from`/`to`/`on`/`order`/`guard`, plus `stepCount`/`transitionCount`. Assert the **exact key
    set** of the body (the anti-drift assertion the standing feedback asks for).
11. **The backlog's test 2 (create-only *on properties*, pinned not hidden) —** publish; re-publish
    the same `(key, version)` with a changed `name`, a changed `kind` and a changed step `config`;
    assert the re-publish returns **201**; assert the structure read comes back **unchanged**.
    ⚠️ The substituted `kind` **must be a member of `WORKFLOW_KINDS`** (`services.py:51` —
    `{"conversation", "process"}`, enforced at `:531`), otherwise the re-publish is a **400** and the
    test pins the wrong thing: `DEF_BODY`'s `kind` is `process`, so use `conversation` (gate n-4).
    A comment must state that this pins a *decision*, not a bug, that K-031 deliberately does not
    change it, and that **the structural (additive) half of re-publish is K-034's, not tested here.**
12. Snapshot structure route: 200 after materialize with the same shape and `source: "workspace"`;
    404 before materialize.
13. Def structure route 404 for an unknown key and for a known key with an unknown version.
14. `response_model` fidelity: `from` is serialized as `from` (not `from_`).

> **Removed in v2 (stakeholder decision):** the former test 12 — "re-publish with an added step and
> an added transition, assert the structure read shows them" — travels to **K-034** with the finding.
> K-031 must not pin publish-structure semantics it does not own.

### Divergence fixture (`tests/test_api.py` or a new `tests/test_workflow_structure.py`)

15. **The backlog's test 3 —** produce a genuine divergence **the way the documented live trap
    produces one**, without depending on K-034's additive semantics: publish `A@1` + materialize into
    `ws:test`; **wipe `reference`** in-test (`db.reference_graph(conn).query("MATCH (n) DETACH DELETE
    n")`, the same statement `conftest.py:93` uses — factor a tiny `_wipe_reference` helper);
    re-publish `A@1` with an **edited `name`**, an **added step** and an **edited `guard`** — a fresh
    create into an empty `reference`, so every edit lands. Assert the diff reports `inSync: false`
    with `meta.name`, `steps[<new>]` `"present"`/`"absent"`, and the `transitions[…].guard`
    difference. This is exactly the post-`pytest`/post-`test_queries.sh` re-seed shape AGENTS.md
    warns about.
    *(v2 note: the v1 alternative "publish a second, edited def under a different key and
    materialize it under the first key's identity" is **deleted** — `services.materialize_def`
    (`services.py:653-671`) reads and writes the **same** `key@version`, so there is no seam for it
    short of calling `repo.materialize_snapshot` with mismatched arguments. Gate m-2.)*
16. `defPresent: false, snapshotPresent: true` (materialize, then wipe `reference`) ⇒ **200**,
    `inSync: false`, `differences: []`. This is the post-`pytest` trap, asserted.
17. Both absent ⇒ **404**.
18. Identical def and snapshot ⇒ `inSync: true, differenceCount: 0`.

### Script

19. Manual acceptance (recorded in the delivery notes, not automated): on a seeded `ws:acme`,
    `./scripts/verify_workflows.sh acme` exits **0**; after `cd server && .venv/bin/python -m pytest -q`
    (which wipes `reference`) it exits **1** naming both defs; after `./scripts/seed_workflows.sh acme`
    it exits **0** again. **Leave the environment seeded.** If this pass reveals a *real* live
    divergence on `triage@v1`/`access-request@v1`: **report and file — do not repair** (R-1).

### Suite gates

- `server` pytest: **`entry → entry + N` passed / 1 deselected**, where `entry` is measured at U3
  start with `pytest --collect-only -q` (§1.3). Enumerate both numbers in HISTORY.md. Do **not**
  treat the historical 533 as the expected entry count.
- `./scripts/test_queries.sh` **256/256 — unchanged** (the no-new-Cypher tripwire).
- `ruff check .` in `server/` clean.
- **Re-seed after both suites** (`./scripts/seed_workflows.sh acme`) before finishing — and then
  prove it with the new script.

---

## 8. Documentation update list (mandatory — part of done)

**Scope note (gate B-2 + stakeholder decision):** this list covers what **K-031 adds**. The ten
shipped assertions that the additive-`MERGE` finding falsifies (QUERIES.md §11 preamble / §11.1 /
§11.4 footnotes, `DESIGN.md:544`/`:102`/`:144-149`, the `publish_def` / `materialize_snapshot` /
`services.materialize_def` docstrings, the `AGENTS.md` `seed_workflows.sh` row, and K-029's premise
line) belong to **K-034** and are **not** corrected here. K-031 cross-references K-034; it does not
pre-empt it. See R-8 for the grep-based done-condition and R-9 for the "K-034 not yet filed" case.

| Doc | Change |
|---|---|
| `docs/DESIGN.md` §14.4 | **Follow the section's existing convention** (gate m-3): §14.4's table deliberately excludes the §11/§12 routes — *"they are read/publish paths and are described at their own sections"* (`DESIGN.md:806-809`). So **do not** add the three routes to the table. Instead: extend that parenthetical to name them (`GET /workflow-defs/{key}/versions/{version}`, `GET /workspaces/{ws}/snapshots/{key}/versions/{version}`, `…/diff` → QUERIES.md §11.2 / §11.5), and add one short paragraph beneath it carrying the four operator-facing facts: the structure reads are **whole-object and unpaginated** (bounded by the publish-time caps); the diff is bounded by preview truncation; **the snapshot is what the executor drives**; the diff is **version-qualified** — to detect a stale *version*, compare `GET /workflow-defs` against `GET /workspaces/{ws}/snapshots` first (gate m-1). Include the receipt-vs-stored sentence from §3.2 verbatim, cross-referencing K-034. |
| `docs/DESIGN.md` §6 (~`:350`, `:390`) | The `maxSteps` text from §5, including the "not checked on the park or terminal paths" clause. |
| `docs/QUERIES.md` §11.2 | A **note only — no query change**: state what **V-1 actually observed** about the one-row collapse when a root has more than one `START` edge (the §11.2 footnote's "`start.key` is constant across the fan-out" premise), that the K-031 structure reader consumes all rows for exactly that reason, and that **how such a state arises is K-034's**. Also record the deliberate `start_keys` shape divergence between `_read_structure` and §11.2's scalar `startKey` (gate m-6). Mirror one line into §11.5. |
| `docs/QUERIES.md` §12.5 note (`:1141`) + `$maxSteps` comments (`:1034`, `:1254`) | The `maxSteps + 1` clarification (§5 wording). |
| `falkor-chat/AGENTS.md` — the `seed_workflows.sh` script-table row | **Add the detection half only** — *"…and here is how you now detect it: `GET /workspaces/{ws}/snapshots/{key}/versions/{version}/diff`, or `./scripts/verify_workflows.sh <wsId>` for both defs at once."* **Do not** rewrite the row's create-only/immutability description — that correction is **K-034's** (§1.2). Add the same pointer to the `test_queries.sh` row (which warns about the `reference` wipe). |
| `falkor-chat/AGENTS.md` — key-scripts table | New row for `scripts/verify_workflows.sh` (read-only, exit 0/1, works without the server running). |
| `falkor-chat/AGENTS.md` — executor invariants block | The `maxSteps + 1` clarification, next to the `_drive_loop` SHA-lock bullet. |
| `docs/HISTORY.md` | Dated entry: the three endpoints, the diff, the script, the **measured** entry→exit pytest counts, `test_queries.sh` unchanged at 256/256, the documented `maxSteps` disposition + K-033 filing, **V-1's verbatim outcome**, and **whatever the diff of the two live defs actually reported** (R-1 — if the live defs are out of sync, that goes on the record *as a report*, with the filed item's number). |
| `docs/BACKLOG.md` | K-031 → ✅ delivered, stating the `maxSteps` disposition (documented; fix filed as K-033) and **cross-referencing K-034** as the owner of the additive-re-publish finding this read surface *detects*; **add the proposed self-standing K-033 item** per §5 (bundling stated as an explicit *preference* with the unverified premise flagged); append one line to the parking-lot response-schema entry (`:793`) noting the new routes declare `response_model` and the rest do not. **Do not file K-034 here** — the coordinator owns that filing (R-9). |
| `README.md` | Only if it enumerates endpoints — it does **not** (grep: one `GET /health` mention). No change expected; confirm rather than assume. |
| `docs/plans/m3-followups-coordination.md` | U2/U3 rows updated by the coordinator, not the implementer. |

---

## 9. Open questions

**All three of v1's open questions are closed** by binding stakeholder decisions recorded in
`docs/plans/m3-followups-coordination.md` (2026-07-24). They are listed here as *decisions*, not
questions — do not reopen them:

- **OQ-1 · `maxSteps` off-by-one → DOCUMENT-ONLY.** `executor.py:410`/`:427` unchanged, SHA lock
  intact, `test_executor.py:158` unchanged. Document at the six sites (§8) and file **K-033**
  self-standing (§5).
- **OQ-2 · Diff path → `/workspaces/{ws}/snapshots/{key}/versions/{version}/diff`.** Folded into
  §3.3 as an architect decision with the rejected alternative recorded (gate n-1).
- **OQ-3 · Repairing live def/snapshot divergence → OUT OF SCOPE.** Report and file, never repair.
  Promoted from an aside to an explicit instruction at **R-1** and §7 test 19.

**One residual, owned by the coordinator, not blocking:** K-034's filing must land before or
alongside U3 so U6's cross-references resolve (R-9).

---

## 10. Ready to implement

**Plan:** `falkor-chat/docs/plans/workflow-def-structure-read.md` (this file, **v2**).
**Owner chain:** `architect` (v2 done) → `analyst` re-gate (U2-G1) → `coder` (U3) → `analyst` (U3-G).
**No `graph-dba` gate** — confirmed at the gate: zero new or modified Cypher (§0.1, §2).

Six units: **U1** repository structure readers reusing the existing query constants · **U2** service
canonicalization + two structure reads · **U3** two REST routes with declared response models ·
**U4** the server-side diff · **U5** `scripts/verify_workflows.sh` · **U6** docs (§8) including the
`maxSteps` text and the K-034 cross-references.

Non-negotiables for the implementer:

1. **Run V-1 (§7) first** — snapshot-side, `ws:k031probe`, torn down with `GRAPH.DELETE`. Any
   surprising outcome is a **stop-and-escalate**, not a design adjustment.
2. **Measure the pytest entry baseline** at U3 start (`pytest --collect-only -q`); report
   `entry → entry + N`. Keep `test_queries.sh` at exactly **256/256** — a delta means new Cypher
   slipped in: stop and escalate.
3. **Do not touch** `executor.py`, `_PUBLISH_CYPHER`, `_read_subgraph`, `read_def_subgraph`,
   `services.get_snapshot`, or any published def's content.
4. **Do not repair** a live divergence if one is found — report it, file it (R-1).
5. **Do not absorb K-034** — no additive-re-publish test, no immutability-prose corrections (§1.2,
   §8).

---

## 11. Review dispositions (gate U2-G1, `docs/reviews/workflow-def-structure-read.md`)

Verdict: *needs changes* — 2 blocker · 3 major · 6 minor · 4 nit. **All 15 adopted**; two are
adopted in a *re-scoped* form set by binding stakeholder decisions, and one offered two options of
which one was chosen. Nothing was rejected on merits.

| # | Finding | Disposition | Where |
|---|---|---|---|
| **B-1** | V-1 unexecutable — a def publish has no workspace seam | **Adopted** — V-1 rewritten snapshot-side (`ws:k031probe`, same `_PUBLISH_CYPHER`/`_READ_META_CYPHER` constants, `GRAPH.DELETE` teardown), with the query-text-identity transfer argument stated so it is not re-litigated, an explicit isolation contract, and the `≥1 transition` trap called out | §7 V-1, §2 (two new evidence rows), §4 preamble |
| **B-2** | Additive-`MERGE` finding confirmed, filed nowhere | **Adopted, re-scoped per stakeholder decision** — the finding leaves K-031 entirely and becomes **K-034** (filed by the coordinator with the analyst's evidence). K-031 keeps the *detection* mechanism and cross-references K-034; the additive-re-publish test is removed from §7 and the `AGENTS.md` create-only correction removed from §8. The review's suggested improvements 1–3 (file it, correct K-029's premise, correct the ten sites) are **K-034's deliverables, not deferred** | §0.2, §1.2, §7 (test 12 removed + note), §8 scope note, R-9 |
| **B-2 / R-8** | "§7's list is exhaustive" claim | **Adopted** — the exhaustiveness claim is dropped and replaced with a grep-verified done-condition (`grep -rn -i "immutab\|no-op" …`) plus an explicit three-way classification of every hit | R-8 |
| **M-1** | Publish receipt counts the *submitted* spec, not the stored def | **Adopted** — §3.2's comparability rationale replaced with the truth and the operator-facing sentence ("receipt = submitted, structure read = stored; a divergence is a signal, not a bug"); the same sentence is mandated in the DESIGN §14.4 paragraph; a supporting evidence row added to §2 | §3.2, §8, §2 |
| **M-2** | K-033's bundling premise unverified, K-027 unscheduled | **Adopted** — K-033 is now filed **self-standing**, with bundling stated as a *preference* and the "K-027 item 2 must break the lock anyway" premise explicitly marked **unverified** (with the `AGENTS.md:256-257` seams that could avoid it, and `BACKLOG.md:315`'s 🔵 proposed status) | §5, §8 (BACKLOG row) |
| **M-3** | `533 → 533 + N` gate already stale (working tree collects 551–552) | **Adopted** — the gate is now "measure `entry` at U3 start with the non-mutating `pytest --collect-only -q`, report `entry → entry + N`"; 533 is labelled the pre-U1 baseline; the 2026-07-24 measurement (552 collected / 1 deselected) is quoted as a moving reference only. `test_queries.sh` **256/256** left exactly as-is | §1.3, §7 suite gates, front matter |
| **m-1** | Diff is version-qualified — cannot detect a stale *version* | **Adopted** — stated as a named limitation in §3.3 with the "compare `GET /workflow-defs` against `GET /workspaces/{ws}/snapshots` first" remedy, mandated in the DESIGN §14.4 paragraph and in the route's own comment, and `verify_workflows.sh`'s check 1 upgraded to an explicit *expected-version* check | §3.3, §3.8, §4 U4, §8 |
| **m-2** | Test 16's first alternative unreachable through the service layer | **Adopted, and improved** — the unreachable alternative is deleted (with the `services.py:653-671` reason recorded), and the surviving fixture is re-specified to **not depend on K-034's additive semantics at all**: wipe `reference` in-test, then re-publish `A@1` edited into the empty graph. That is both reachable and the documented live trap shape | §7 test 15 |
| **m-3** | DESIGN §14.4 excludes §11 routes by convention — adding three leaves it half-populated | **Adopted, option B chosen** — follow the existing convention: the three routes are **not** added to the table; the existing parenthetical (`DESIGN.md:806-809`) is extended to name them and point at QUERIES.md §11.2/§11.5, with the operator paragraph beneath it. Rejected the alternative (enumerate all §11/§12 routes in the table) as a larger doc change than K-031 warrants | §8 (DESIGN §14.4 row) |
| **m-4** | V-1 is an engine-semantics question handed to a `coder` with no escalation rule | **Adopted** — an explicit stop-and-escalate rule added, enumerating the surprising outcomes (one row with an arbitrary `startKey`, an error, N rows for one `START` edge, a `START` count ≠ 2), in R-4's tripwire shape, plus the graph-dba inbox routing | §7 V-1, R-2, §10 |
| **m-5** | `api.py` does not import `MAX_KEY_LEN` | **Adopted** — the import addition is an explicit U3 step, with `api.py:17-26` / `schemas.py:42` cited; an evidence row added to §2 | §4 U3.2, §2 |
| **m-6** | `_read_structure`'s `start_keys` shape is not in QUERIES.md §11.2 | **Adopted** — the docstring must now cite the new §11.2 note **by name** and explain *why the shape differs*, not only why the helper duplicates `_read_subgraph`; §8's §11.2 row records the divergence from the query side | §3.6, §4 U1.1, §8 |
| **n-1** | OQ-2 is not a stakeholder call | **Adopted** — folded into §3.3 as an architect decision with the rejected alternative recorded. (The stakeholder has since confirmed the same placement independently; §9 records it as closed either way) | §3.3, §9 |
| **n-2** | §7 calls V-1 "read-only-ish" | **Adopted** — V-1 is now labelled a **write**, in a throwaway graph, torn down; that framing is what makes the snapshot-side reformulation the obvious one | §7 V-1 heading, §4 preamble |
| **n-3** | QA report cited as "§5 / §7.1 / §7.5" | **Adopted** — corrected to **§5 (DEF-1)** and **§7 items 1 and 5** everywhere it appears | Front matter, §3.3, §3.8 |
| **n-4** | Test 11 re-publishes with a changed `kind` | **Adopted** — the substituted kind must be a member of `WORKFLOW_KINDS` (`services.py:51`, `{conversation, process}`, enforced at `:531`); since `DEF_BODY.kind == "process"`, the test uses `conversation`, otherwise the re-publish 400s and the test pins the wrong thing | §7 test 11, §2 |
| **Gate answer 5 (b)** | The `maxSteps + 1` wording should say the budget is not checked on the park/terminal paths | **Adopted** — the documentation text now names OUTCOME A (`:410`) and OUTCOME C (`:427`) as the only checked paths and states the park (OUTCOME B, `:415-421` comment) and terminal paths are deliberately unchecked | §5 |
| **Review §5 Q1** | K-034 as a new item vs. folded into K-029 | **Closed by the stakeholder**: a **new item, K-034**, cross-referenced from K-029 — matching the reviewer's own recommendation | §0.2, §1.2 |
| **Review §5 Q3** | R-1 still unknown (live def/snapshot state not inspected) | **Not resolved here, and deliberately so** — V-1 does not touch the live graphs, and the live state is first observed by U5/§7 test 19. OQ-3 governs what happens then: **report and file, never repair** | R-1, §7 test 19 |
