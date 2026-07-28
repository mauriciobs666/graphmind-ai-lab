# Web API Coverage — Implementation Plan

> **Status:** active · **Owner:** `architect` · **Tracks:** K-036 (new — recommend `teco` add to
> `BACKLOG.md`, milestone M3.5) · relates to K-018 (real-time push, explicitly NOT pulled forward)
> and K-031 (def/snapshot structure reads, delivered 2026-07-24, which FR-10 builds on)
>
> **Version:** **v3 — 2026-07-28** (revision pass after the `analyst` Pass 2 re-review returned
> *needs changes*: one new blocker, B2 — the v2 fix for B1 reworded §3.3's new AC-5 note and
> §5.2's AC-5 bullet correctly, but left the old, just-rejected "collapsed/unchanged-by-default"
> reading standing in three other spots that contradicted it; swept in this revision). v2 =
> 2026-07-28 (revision pass after the `analyst` Pass 1 gate returned *needs changes*: 1 blocker ·
> 2 major · 3 minor · 1 nit). v1 = 2026-07-28 (initial design). Every finding's disposition is
> recorded in **§8**. **Review:** `docs/reviews/web-api-coverage.md`.

Requirements: `falkor-chat/docs/requirements/web-api-coverage.md` (FR-1..FR-14, AC-1..AC-6,
committed scope = FR-1..FR-10/AC-1..AC-6; FR-11..FR-14 explicitly not committed).

---

## 1. Goal & scope

Make the M3 agent/workflow story (and the rest of the demo path) drivable entirely from
`falkor-chat/web/` — defs, an in-thread run cue, a run detail panel (status/steps/trace/failure),
a structured-input form for parked steps, a thread participants list, and a workspace
ready-to-demo check — without enlarging the chat page's default visual footprint (FR-9/AC-5).

**In scope:** FR-1..FR-10 / AC-1..AC-6 (the committed demo path), plus the minimum new
server-side read paths FR-8 and FR-2 require.

**Out of scope (per the requirements doc):** FR-11..FR-14 (publish/materialize UI, deep snapshot
browsing, explicit "start a run" UI, health/get-message UI — nice-to-have, later iteration);
end-user UX polish; MCP parity; chat layout/theming/framework migration; auth; in-browser def
authoring. Also out of scope for *this* plan: pulling K-018 (Pub/Sub → WS/SSE) forward — FR-4 is
met by polling (§3.2 below).

---

## 2. Context & findings

### 2.1 What already exists (no server change needed)

Cross-checking FR-1..FR-10 against the current REST surface (`server/falkorchat/api.py`,
mirrored in `docs/DESIGN.md` §14.4) shows **most of the committed path is pure UI wiring**:

| FR | Needs | Already exists? |
|---|---|---|
| FR-1 (defs list + shape) | `GET /workflow-defs`, `GET /workflow-defs/{key}/versions/{version}` | ✅ yes |
| FR-3 (run status/steps/trace) | `GET /workflow-runs/{id}`, `.../step-runs`, `.../trace` | ✅ yes |
| FR-5 (waiting + prompt) | `GET /workflow-runs/{id}` (`atStepKey`) + the snapshot's step `config.prompt` (`GET /workspaces/{ws}/snapshots/{key}/versions/{version}`) | ✅ yes — client-side join of two existing reads |
| FR-6 (structured input form) | `config.fields` (top-level keys) / `config.expects` (allowed values) on the parked step, from the same snapshot read; submit via `POST /workflow-runs/{id}/input` | ✅ yes — see `proof_defs.ACCESS_REQUEST_DEF`'s `human`/`wait` steps for the exact shape (`server/falkorchat/proof_defs.py:73-107`) |
| FR-7 (failure + reason) | `GET /workflow-runs/{id}`: `status: "failed"`, and `ctx` (opaque JSON string) carries `{"error": "..."}` stamped by `executor._fail_with_note` (`executor.py:804-815`) on every fail path (budget exhaustion, drive fault) | ✅ yes — client `JSON.parse`s `ctx.error` |

**FR-2 and FR-8 are the genuine gaps** (see 2.2/2.3). **FR-10** needs a small new aggregation
endpoint that composes *existing* service methods (see 2.4) — no new Cypher.

### 2.2 FR-2 — the inline run cue needs a new read path

`POST /threads/{tid}/messages` (`api.py:143-168`) schedules the workflow trigger
(`_safe_run_workflow` → `WorkflowTrigger.maybe_trigger`, `trigger.py`) on **`BackgroundTasks`** —
off the request path. The HTTP response to the post carries no run information. There is also no
existing query/method that lists a thread's workflow runs: `find_waiting_run_for_thread`
(QUERIES.md §12.9) only ever returns a *currently parked* run, by design (denormalized
`waitingThreadId`, deliberately not a `TRIGGERED_BY` traversal — see the §12.9 "Decision" note).
The browser has no way to discover "a run started in this thread" without a new poll-friendly
read.

`WorkflowRun` carries `-[:TRIGGERED_BY]->(:Message)` for chat-triggered runs (QUERIES.md §12,
model block), and `Message.threadId` is a denormalized (unindexed) property (§4 v2 notes). The
natural new read is: anchor on `WorkflowRun` (small, workspace-scoped cardinality — far smaller
than `Message`), traverse `TRIGGERED_BY` outward, filter on `Message.threadId`. This is exactly
the alternative §12.9 *considered and rejected for the single "find the parked run" case*
(because the denorm was simpler there) — but for "every run this thread has ever had", the
denorm doesn't apply (it only reflects live park state), so the `TRIGGERED_BY` traversal is not
an alternative here, it's the only correct shape.

### 2.3 FR-8 — thread participants: no existing modeling fits, but no new modeling is needed either

Per the requirements doc's own note, `resolve_member_kinds` only validates caller-supplied ids —
there is no "who is here" read at all. Checking the schema (`AGENTS.md` conventions,
`docs/QUERIES.md` §2/§3): membership (`MEMBER_OF`) is modeled at **Channel** granularity only —
`(:User|:Agent)-[:MEMBER_OF]->(:Channel)` — there is no `Thread`-level membership edge, and
`seed_demo.sh`'s own comment confirms the intent: *"MEMBER_OF is seeded for roster/scoping."*
`Thread` has no `channelId` property; the only path back to a channel is the reverse traversal
`(:Channel)-[:HAS_THREAD]->(:Thread)`.

A `List channel members` query already exists, verified, and covered by `test_queries.sh`
(QUERIES.md §2 "List channel members"; live-tested at `scripts/test_queries.sh:429-432`) —
it returns both `User` and `Agent` members with `coalesce(userId,agentId)` + `labels()`, which is
exactly FR-8's "humans and agents, visually distinguishable" shape. **Decision: "a thread's
participants" = its parent channel's roster.**

**Correction (v2):** v1 of this section justified the decision partly on "mentions resolve
against members, and membership is scoped at the channel" — that claim is **false** and has been
removed. Checked directly: `resolve_member_kinds` (`server/falkorchat/repository.py:904-920`) does
a plain `User`/`Agent` id lookup with **no `MEMBER_OF` traversal or filter of any kind**, and its
only caller, `_validate_and_derive_role` (`server/falkorchat/services.py:438-462`), accepts any id
that resolves to *any* known `User`/`Agent` in the workspace — not scoped to the thread's channel
or any channel at all. `scripts/seed_demo.sh:23` says so in its own comment: *"mention validation
... looks up `agentId`, not channel membership; `MEMBER_OF` is seeded for roster/scoping."* A
repo-wide grep confirms `MEMBER_OF` is read nowhere except the (currently uncalled) "List channel
members" query. **There is currently no server-side scoping of who can be `@mentioned` at all** —
today, any workspace member can be mentioned from any thread, regardless of channel roster.

Rationale, corrected: (a) the channel roster is a **reasonable proxy for "who's around,"** matching
the user story's framing loosely, but it is *not* a technically-derived "who I can mention" set —
it's a UI/visibility design choice, not an enforcement mirror; (b) it needs zero new
schema/edges/indexes, just one new 2-hop composition of two already-verified query shapes; (c) the
alternative (derive participants from `POSTED_BY`/`MENTIONS_MEMBER` activity in the thread) would
require walking the thread's message chain on every read, is heavier, and answers a different
question ("who has talked here") than the story asks ("who could I address"). Legs (b) and (c)
alone justify the same decision; leg (a) no longer claims a technical grounding it doesn't have.

**Known, accepted gap:** because mention validation has no membership check, the participants list
this feature ships will be **narrower** than "everyone I could actually `@mention`" — a user in the
same workspace but a different channel can still be successfully mentioned in this thread, yet will
not appear in its participants list. This is acceptable for a visibility/demo feature (it is not a
security or access-control boundary — nothing here claims to be one), but it should not be read as
"the set of mentionable ids," and a future requirement that wants that guarantee needs real
`MEMBER_OF`-scoped mention validation, which is new work, not something this plan does silently. If
a future requirement genuinely wants thread-level (not channel-level) rosters, that is new schema
work and its own decision — not silently invented here.

**This is a new query composition, not new graph modeling** (no new label, edge type, index, or
constraint) — see §3.4 for the call on whether it needs a `graph-dba` design note.

### 2.4 FR-10 — ready-to-demo reuses `verify_workflows.sh`'s exact check, server-side

`scripts/verify_workflows.sh` already implements "is what I think is published actually
published, and does the workspace agree" by calling `services.diff_def_snapshot` (+ the
structure reads, for the `startKeys` multi-`START` tripwire) over a **hardcoded pair**:
`[(config.TRIGGER_DEF_KEY, config.TRIGGER_DEF_VERSION), (proof_defs.ACCESS_REQUEST_DEF["key"],
proof_defs.ACCESS_REQUEST_DEF["version"])]` (`scripts/verify_workflows.sh:80-83`). There is no
REST endpoint exposing this aggregate — only the script, run over the service layer directly. FR-10
needs the *same* answer reachable over HTTP. This is pure service-layer composition of methods
that already exist (`diff_def_snapshot`, `get_workflow_def_structure`, `get_snapshot_structure`)
— **no new Cypher**.

### 2.5 Existing UI patterns to reuse (`web/index.html` + `web/app.js`)

- **Polling idiom**: `state.pollTimer` / `startPolling()` / `stopPolling()`, `POLL_MS = 3000`,
  already drives the open thread's message catch-up (`app.js:14-18, 168-192`). Same shape reused
  for run-progress polling (§3.2).
- **Collapsed-overlay idiom**: `#results` (search results) is `position:absolute`,
  `display:none` by default, toggled on demand (`index.html:59-63, 115-118`; `app.js:196-223`).
  Same shape reused for the defs viewer and the run detail panel (§3.3).
- **Kind-badge idiom**: `.msg.assistant` + `.badge`/`--agent` CSS already distinguishes
  agent-authored messages with an "AI" pill (`index.html:47-54`). Same tokens reused for
  participant/run-cue agent-vs-human badges — no new visual language.
- **Toast idiom**: `showError`/`showNotice` (`app.js:236-246`) for non-blocking feedback (e.g. a
  rejected structured-input submit).
- **Error contract**: non-2xx responses carry `{"error": "<ClassName>", "detail": "..."}`
  (`app.js`'s `api()` helper already reads `body.error`/`body.detail`) — the new endpoints follow
  the same envelope for free (they raise the same `ServiceError` subclasses the app-level handlers
  in `app.py` already map).

---

## 3. Design & rationale

### 3.1 New server-side read paths

**(a) `GET /threads/{thread_id}/workflow-runs?limit=`** — FR-2.

New Cypher (QUERIES.md, new §12.14 `find_runs_for_thread`):

```cypher
// $threadId, $limit
MATCH (r:WorkflowRun)-[:TRIGGERED_BY]->(m:Message)
WHERE m.threadId = $threadId
RETURN r.runId AS runId, r.status AS status, r.defKey AS defKey,
       r.defVersion AS defVersion, r.startedAt AS startedAt, r.endedAt AS endedAt
ORDER BY r.startedAt DESC
LIMIT $limit
```

Anchors on `Node By Label Scan | (r:WorkflowRun)` — there is no property predicate on `WorkflowRun`
itself to index-anchor on, but the run count is workspace-scoped and orders of magnitude smaller
than `Message` count (this is the same cardinality argument §12.9 already accepted for rejecting
the `TRIGGERED_BY` alternative there — here it just runs the other direction because the *need*
is different: history, not "the one waiting run"). **Verify this empirically with
`GRAPH.PROFILE` before landing** (§3.4/§5 U1) — if the label scan turns out to matter at realistic
demo scale, the fallback is a `WorkflowRun.startedAt` range index (cheap, one more index).

**Caveat (v2):** this query has **no `WHERE` predicate on `startedAt`** — only
`ORDER BY r.startedAt DESC`. Neither `docs/QUERIES.md` nor `claude/graph-dba/falkordb-quirks.md`
currently documents whether FalkorDB's planner can use a range index to serve an `ORDER BY` with
no accompanying range filter (the documented, verified pattern elsewhere is a range index avoiding
a label scan *because there's a `WHERE`*, e.g. quirks KB lines ~140-149 — not this shape). So the
"cheap, one more index, no query-shape change" framing above is a hypothesis, not a known-good
escape hatch — U1 must `GRAPH.PROFILE` **this exact query, with the index in place**, and confirm
the plan actually changes (no more `Node By Label Scan`) before relying on it. If the index turns
out not to be picked up for an `ORDER BY`-only shape, U1 needs to change the query itself (e.g. add
a supporting predicate such as `WHERE r.startedAt >= 0`, or reconsider the anchor) rather than ship
an index that does nothing.

- `repository.find_runs_for_thread(ws, *, thread_id, limit=10) -> list[dict]`
- `services.list_workflow_runs_for_thread(ctx, *, thread_id, limit=10) -> list[dict]` — validates
  `thread_exists` first, raising `ThreadNotFoundError` exactly like `_validate_and_derive_role`
  does (`services.py:447-448`) — reuse that idiom, don't invent a new one.
- Route: `GET /threads/{thread_id}/workflow-runs?limit=` (1-50, default 10) → 200 (list, possibly
  empty), 404 if the thread doesn't exist. No `response_model` (matches the surface's convention
  — only the three K-031 structure/diff routes declare one, `api.py:250-252`).

**(b) `GET /threads/{thread_id}/participants`** — FR-8.

New Cypher (QUERIES.md, new §2 subsection "List thread participants"):

```cypher
// $threadId — a thread's participants = its parent channel's roster (§2.3 decision)
MATCH (c:Channel)-[:HAS_THREAD]->(t:Thread {threadId: $threadId})
MATCH (u)-[:MEMBER_OF]->(c)
RETURN coalesce(u.userId, u.agentId) AS memberId,
       u.displayName                 AS displayName,
       labels(u)                     AS type
ORDER BY u.displayName
```

Anchors on `Node By Index Scan | (t:Thread)` (`Thread.threadId`), one backward hop to `c`, one
forward hop per member via `MEMBER_OF` (bounded by channel roster size — small). Mirrors the
already-verified §2 "List channel members" query with an added leading hop; **verify with
`GRAPH.PROFILE`** that the two-hop composition still avoids a label scan (§5 U1).

- `repository.list_thread_participants(ws, *, thread_id) -> list[dict]`
- `services.list_thread_participants(ctx, *, thread_id) -> list[dict]` — same `thread_exists` /
  `ThreadNotFoundError` guard as (a).
- Route: `GET /threads/{thread_id}/participants` → 200 `[{"memberId", "displayName", "kind"}]`
  (`kind` = `"User"` or `"Agent"`, derived from `labels(u)[0]` — same normalization
  `resolve_member_kinds` already does, QUERIES.md §2), 404 if the thread doesn't exist.

**(c) `GET /workspaces/{ws}/readiness`** — FR-10.

No new Cypher. New `services.check_demo_readiness(ctx) -> dict`:

```python
DEMO_EXPECTED_DEFS: tuple[tuple[str, str], ...] = (
    (config.TRIGGER_DEF_KEY, config.TRIGGER_DEF_VERSION),
    (proof_defs.ACCESS_REQUEST_DEF["key"], proof_defs.ACCESS_REQUEST_DEF["version"]),
)
```

For each `(key, version)`: call `diff_def_snapshot` (catching `WorkflowDefNotFoundError` exactly
like `verify_workflows.sh`'s `read()` helper does, substituting the same `ABSENT` shape), then
(for parity with the script's Finding-3 tripwire) call `get_workflow_def_structure` /
`get_snapshot_structure` and flag `"startKeys" in structure` as a problem. Return:

```json
{
  "ready": false,
  "defs": [
    {"key": "triage", "version": "v1", "defPresent": true, "snapshotPresent": true,
     "inSync": true, "problems": []},
    {"key": "access-request", "version": "v1", "defPresent": true, "snapshotPresent": false,
     "problems": ["not materialized into ws:acme"]}
  ]
}
```

`ready` = every def's `defPresent AND snapshotPresent AND inSync AND problems == []`. `problems`
carries human-readable strings (reuse the exact phrasing `verify_workflows.sh` already prints,
e.g. `"{label}: not materialized into ws:{ws}"`, `"{label}: reference def and ws:{ws} snapshot
diverge ({n} differences)"`) so AC-6's "names the offending definition" is satisfied verbatim and
the page and the script never disagree on what "ready" means.

- Route: `GET /workspaces/{ws}/readiness` → 200 (always 200; readiness is a *report*, never a
  404/error condition — mirrors `list_snapshots`' `ws` path convention, tenancy from
  `get_context`).
- **Recommended cleanup (should-do, not blocking):** have `verify_workflows.sh`'s Python one-shot
  import `DEMO_EXPECTED_DEFS` (or call `services.check_demo_readiness` directly) instead of
  declaring its own `DEFS` list — the two lists would otherwise be a second, silent copy of the
  exact kind of drift this whole feature exists to catch. Low risk (the script already imports
  `services`/`config`/`proof_defs`), small diff, worth doing in the same unit that adds
  `check_demo_readiness`.

### 3.2 FR-4 freshness — polling, not K-018

**Mechanism: plain polling, reusing the existing `POLL_MS = 3000` cadence** (`app.js:17`), same
interval already driving message catch-up. Worst-case latency from a state change landing in the
graph to it being visible = one poll interval + one round trip ≈ 3.0–3.3s, comfortably under the
5s bar (AC-2) with margin, using infrastructure that already exists and is already load-bearing
in this codebase.

**Explicitly not K-018** (Pub/Sub → WebSocket/SSE): the requirements doc's own context note says
this is a freshness requirement, not a transport choice, and the 5s bar does not need push. Pull
K-018 forward only if a future requirement tightens the bar below what polling can meet with
acceptable request volume — not needed here, and reopening it is out of this plan's scope.

**What polls, and when (ties into FR-9):**
- *Thread-runs* (endpoint 3.1a) piggybacks on the **existing** per-thread poll loop
  (`startPolling`/`pollMessages`, `app.js:168-192`) — one more lightweight fetch alongside the
  message catch-up, only while a thread is open. Drives the inline cue (FR-2).
- *Run detail* (status/step-runs/trace) polls on the **same 3000ms cadence**, but only while the
  run detail panel is open — its own `pollTimer`, started when the panel opens, stopped when it
  closes (mirrors `stopPolling()` exactly). A user who never opens the panel causes zero extra
  traffic beyond the one thread-runs check.
- *Readiness* (3.1c) is **not** on the 5s bar (FR-4 scopes to run progress, not readiness) —
  checked once on page load plus a manual "recheck" affordance. No poll loop needed.

### 3.3 Web UI structure (FR-1/2/3/5/6/7/8/9/10, AC-1..AC-6)

Every new surface's *content* is additive and collapsed-by-default; three surfaces' *trigger*
affordances are themselves always visible — small, minimal, and matched in weight to existing
header elements (see the AC-5 reading just below). The 3-column grid
(`main { grid-template-columns: 220px 240px 1fr }`) is untouched.

**AC-5 reading, decided here (v2 — resolves review finding B1):** AC-5's actual text is "a chat
page whose default layout is no more crowded than today's," not "zero new pixels." Three of the
five surfaces below (§3.3.1 defs button, §3.3.4 participants toggle, §3.3.5 readiness badge) are
**trigger affordances that are themselves always visible** — a small header button, a small toggle
next to `#thread-heading`, a small badge next to the `tenant` span — each sized and styled to match
an existing header element (the defs button explicitly matches the "Search" form's visual weight).
None of them render or fetch any *content* until clicked/expanded; only the trigger pixel itself is
always present. This satisfies AC-5 as written (no more crowded — three small header-level
affordances, not a busier page) without pretending the page is byte-for-byte unchanged, which it
is not. The fourth and fifth surfaces (§3.3.2 inline run cue, §3.3.3 run detail panel) are
genuinely zero-footprint until a run exists / the cue is clicked. If a stricter bar (literally zero
new always-visible chrome) turns out to be the actual product intent, that is a scope change to
this design, to be raised as a new decision if it comes up later — it is not adopted here, and
§5.2's AC-5 verification instruction (below) is written against the reading actually shipped by
this section.

1. **Defs viewer (FR-1)** — a small header button ("Workflow defs"), same visual weight as the
   existing "Search" form, opens an overlay panel styled like `#results` (`position:absolute`,
   hidden by default). Lists `GET /workflow-defs` (key/version/name); clicking a row fetches
   `GET /workflow-defs/{key}/versions/{version}` and renders steps + transitions as a plain
   list/table (no graph-drawing library — text is enough for "explain what the agent is about to
   run," and stays inside the minimalist mandate). No polling (defs are static once published).

2. **Inline run cue (FR-2)** — a small pill/line that appears **only when the thread-runs poll
   returns a non-empty list**, placed just above the composer (zero footprint when absent — the
   AC-5 case). Text: `"{defKey} {defVersion} — {status}"` + a "View" link. When more than one run
   exists for the thread, the cue shows the most relevant one (non-terminal status —
   `running`/`waiting` — takes priority over `done`/`failed`; ties broken by most recent
   `startedAt`); clicking "View" opens the run detail panel for that run (a "history" affordance
   for older runs is a nice-to-have, not required by any AC — note as an open idea, not a task).

3. **Run detail panel (FR-3/FR-4/FR-5/FR-6/FR-7)** — overlay panel (same `#results`-style shape),
   opened from the inline cue. Renders (via `GET /workflow-runs/{id}` + `.../step-runs` +
   `.../trace`, polled per §3.2 while open):
   - status + started/ended timestamps;
   - the step-run list (key, status, start/end) from `read_workflow_step_runs`;
   - a "show trace" toggle that lazily fetches `.../trace` only when opened (avoid an always-on
     trace fetch — most runs aren't debug instances and return `[]` anyway, but the fetch itself
     is skippable weight);
   - **when `status === "waiting"`**: the parked step's `config.prompt` (looked up from the
     snapshot read, keyed by `atStepKey`) rendered as the "waiting for" text (FR-5), plus a form
     built from `config.fields` (one input per top-level key; if `config.expects[field]` is a
     list, render a `<select>` of those values instead of free text) that `POST`s to
     `/workflow-runs/{id}/input` on submit (FR-6). A rejected submit (400
     `WorkflowInputRejectedError`) surfaces via the existing `showError` toast — nothing new to
     build there. **On a successful submit, immediately re-poll the run** (mirrors the existing
     post-message idiom, `web/app.js:298`'s `await pollMessages()` right after `postMessage()`)
     instead of waiting for the next scheduled tick — free, reuses an established pattern rather
     than leaving this one flow a step behind the message flow, and gives AC-2's 5s bar more
     margin than the worst-case poll-interval math in §3.2 already allows for.
   - **when `status === "failed"`**: `JSON.parse(run.ctx).error` rendered as the failure reason
     (FR-7) — `ctx` is already returned verbatim by `GET /workflow-runs/{id}`.

4. **Thread participants (FR-8/AC-4)** — a small "Participants (n)" toggle next to
   `#thread-heading`, collapsed by default; expanding fetches
   `GET /threads/{tid}/participants` and renders a compact chip row (`displayName` + the existing
   `.badge`/`--agent` styling for `kind === "Agent"`, plain text for `"User"`). No polling — a
   roster is stable enough to refresh on open/thread-switch only (AC-4 says "when the user looks
   at it," not "in real time"; FR-4's freshness bar doesn't cover this list).

5. **Ready-to-demo banner (FR-10/AC-6)** — a small badge near the header `tenant` span (e.g. a
   colored dot + "Ready to demo" / "Not ready"), fetched once on load from
   `GET /workspaces/{ws}/readiness`. Clicking it (only when not-ready, or always — implementer's
   call, low-stakes) expands a short list of `problems` strings per offending def, plus a manual
   recheck button. No poll loop (§3.2).

### 3.4 Graph-dba handoff — call made explicitly

**No `docs/plans/web-api-coverage-graph.md` design note is needed.** Neither new query (§3.1a/b)
introduces a new label, edge type, index, or constraint, or reopens any row in DESIGN.md §1's
decision register — both are compositions of already-verified query shapes (§12.9's traversal
alternative; §2's "List channel members," extended by one hop). There is no new *decision* to
record. What **is** needed, and belongs to `graph-dba` by the repo's established division of
labor (DESIGN §14.6 step 0 — the `list_channels` query gap was graph-dba's prerequisite before
the repository method could be built; `graph-dba` "owns" `QUERIES.md`/`test_queries.sh` changes):
**author + `GRAPH.PROFILE`-verify the two new queries and land them in `QUERIES.md` +
`scripts/test_queries.sh`** before the repository methods that depend on them are written. This
is unit **U1** below — a normal build unit, not a separate design deliverable.

---

## 4. Build sequence

Units are ordered by dependency; units in the same wave have no dependency on each other and can
run in parallel.

### Wave 1 — no dependencies, start immediately

- **U1 (graph-dba).** Author + `GRAPH.PROFILE`-verify the two new queries (§3.1a `find_runs_for_thread`,
  §3.1b thread-participants) in `docs/QUERIES.md` (new §12.14 and a new §2 subsection
  respectively) and add corresponding cases to `scripts/test_queries.sh` (raising the 256/256
  baseline). Confirm both avoid an unbounded `Node By Label Scan` at realistic demo scale (a
  handful of runs / channel members); if the `WorkflowRun` label scan in §3.1a is a real profile
  concern, add a `WorkflowRun.startedAt` range index **and `GRAPH.PROFILE` this exact query again
  with the index in place** to confirm the plan actually changes (per §3.1a's v2 caveat — an
  `ORDER BY`-only query with no `WHERE` on the indexed property is not guaranteed to pick up a
  range index the way a `WHERE`-filtered query does; do not assume it does without checking). If
  the index doesn't change the plan, change the query itself (e.g. a supporting
  `WHERE r.startedAt >= 0` predicate) instead of shipping a no-op index. Note the RAM delta if an
  index is added (repo rule 6). **Done:** both queries in `QUERIES.md`, `test_queries.sh` green
  with the new cases, RAM/PROFILE findings noted inline per the query (matching the file's existing
  annotation style), and — if an index was added — a PROFILE result showing it is actually used by
  §3.1a's query, not just present.

- **U2 (coder or tdd-engineer — backend).** `services.check_demo_readiness` (§3.1c) +
  `GET /workspaces/{ws}/readiness` route. Reuses `diff_def_snapshot` +
  `get_workflow_def_structure`/`get_snapshot_structure` — no repository/Cypher change. Include
  the `verify_workflows.sh` dedup (import `DEMO_EXPECTED_DEFS` instead of the script's own inline
  `DEFS` list). **Done:** `server/tests/test_services.py` covers all-present/all-sync,
  missing-def, missing-snapshot, and diverging cases (fake repo, mirroring the script's `ABSENT`
  fixture); `server/tests/test_api.py` covers the route's 200 shape; `verify_workflows.sh` still
  passes unchanged against a live server. No dependency on U1.

- **U3 (frontend-engineer).** Defs viewer (§3.3.1, FR-1) — wired entirely against the
  already-existing `GET /workflow-defs` + `GET /workflow-defs/{key}/versions/{version}`. **Done:**
  header button opens/closes the overlay; list renders; selecting a def renders its steps +
  transitions; manual check against a running server with `triage`/`access-request` seeded; the
  header button itself is always present on page load (matched in visual weight to the existing
  "Search" form, per §3.3's AC-5 reading) — no overlay *content* renders or fetches until the
  button is clicked.

### Wave 2 — depends on U1

- **U4 (coder or tdd-engineer — backend).** `repository.find_runs_for_thread` +
  `services.list_workflow_runs_for_thread` (`ThreadNotFoundError` guard, §3.1a) +
  `GET /threads/{thread_id}/workflow-runs` route. **Done:** `test_repository.py` (integration,
  `ws:test`), `test_services.py` (unit, fake repo), `test_api.py` (contract: empty list for a
  thread with no runs, populated + ordered newest-first, 404 for an unknown thread) all green.
  Depends on U1 (needs the verified query in place).

- **U5 (coder or tdd-engineer — backend).** `repository.list_thread_participants` +
  `services.list_thread_participants` (`ThreadNotFoundError` guard, §3.1b) +
  `GET /threads/{thread_id}/participants` route. **Done:** same three test layers as U4 (a thread
  in a channel with a user + an agent member returns both, correctly `kind`-labeled; unknown
  thread → 404). Depends on U1.

### Wave 3 — depends on Wave 2 (and U2/U3)

- **U6 (frontend-engineer).** Inline run cue + run detail panel shell: status, step-run list,
  trace toggle (§3.3.2/3.3.3 partial — FR-2/FR-3/FR-4, AC-1's "watch the run appear and
  progress" half). Wires `GET /threads/{tid}/workflow-runs` (the cue) and
  `GET /workflow-runs/{id}` + `.../step-runs` + `.../trace` (the panel), with the panel's own
  poll/stop lifecycle per §3.2. **Done:** cue appears only when a run exists for the open thread
  and updates within one poll tick of a status change (manually verified: mention the demo agent,
  watch the cue appear, watch status move `running → waiting → done`/`failed`); panel opens/closes
  without touching the default layout; no console errors when no run exists; the cue's "most
  relevant run" tie-break (§3.3.2) is extracted as a dependency-free pure function and covered by a
  handful of plain-assertion unit tests runnable via bare `node`, per §5.2's v2 requirement (review
  finding m3) — not optional for this one function. Depends on U4.

- **U7 (frontend-engineer).** Thread participants toggle (§3.3.4, FR-8/AC-4). Depends on U5.
  **Done:** toggle collapsed by default; expanding shows both member kinds distinguishably
  (reusing the existing agent badge token); collapses again on thread switch; the toggle itself is
  always present next to `#thread-heading` (small, per §3.3's AC-5 reading) — no participant
  *content* renders or fetches until expanded.

- **U8 (frontend-engineer).** Ready-to-demo banner (§3.3.5, FR-10/AC-6). Depends on U2. **Done:**
  banner reads "ready"/"not ready" correctly against a synced workspace and against one with a
  deliberately un-materialized or hand-edited-out-of-sync def (reuse the same fixtures
  `verify_workflows.sh`/K-031's structure routes were tested against); not-ready state names the
  offending def.

### Wave 4 — depends on U6

- **U9 (frontend-engineer).** FR-5/FR-6/FR-7 inside the run detail panel: waiting-step prompt +
  structured-input form + failure display (§3.3.3 remainder). **Done:** against a run parked on
  `access-request@v1`'s `submit`/`approval` steps (see §7 risk #1 — now resolved — on how such a
  run is reached in the demo environment), the form renders exactly the step's declared `fields`
  (select where `expects` constrains values), submits successfully, **immediately re-polls the run
  on success** (§3.3.3), and the panel shows the state flip essentially immediately rather than
  waiting a full poll tick (AC-2); a rejected submit toasts an error and does not close the form; a
  failed run (e.g. force one via a malformed guard fixture, or step-budget exhaustion in a test
  def) shows a readable reason (AC-3).

### Wave 5 — after everything lands

- **U10 (qa-engineer).** Black-box acceptance pass against AC-1..AC-6 end to end, driving the
  actual running app (server + seeded `ws:acme`), per §5.2 below.

```
Wave 1:  U1(graph-dba)  U2(backend)  U3(frontend, defs viewer)      ← all independent
Wave 2:  U4(backend, thread-runs)     U5(backend, participants)      ← both need U1
Wave 3:  U6(frontend, cue+panel)  U7(frontend, participants)  U8(frontend, readiness)
              ↑ needs U4              ↑ needs U5                ↑ needs U2
Wave 4:  U9(frontend, waiting/form/failure)                          ← needs U6
Wave 5:  U10(qa-engineer, black-box AC pass)                         ← needs everything
```

---

## 5. Test strategy

### 5.1 Server (`server/`)

Existing gates stay authoritative: `.venv/bin/python -m pytest -q` (needs FalkorDB up,
network-free) and `-m live` (needs LM Studio too, for the `triage` agent steps — not needed for
the new endpoints, which touch no LLM path). `./scripts/test_queries.sh` must stay green with the
two new U1 cases added.

Per new unit: repository-layer integration tests against the isolated `ws:test` graph (mirrors
`test_repository.py`'s existing per-method style); service-layer unit tests with a fake repo
(mirrors `test_services.py`); API-layer `TestClient` contract tests (status codes, body shape,
404 on an unknown thread) mirroring `test_api.py`. This is the same three-layer pattern DESIGN
§14.6 already prescribes for every prior unit — no new pattern to invent.

Edge cases that matter:
- thread-runs: a thread with zero runs (empty list, not 404); a thread with multiple runs
  (newest-first order); unknown thread (404).
- participants: a channel with only a human, only an agent, and both; unknown thread (404); a
  thread whose channel has no members (empty list — this is possible via the demo seed script's
  timing, not an error).
- readiness: all-present-and-synced; one def missing entirely; one def present but its snapshot
  missing; present-both-but-diverging (reuse or adapt the K-031 diff test fixtures); the
  `startKeys` tripwire path (a second `START` edge) if a fixture for it already exists from K-031
  — otherwise this edge is lower priority (K-034's territory, not this feature's).

### 5.2 Web UI

There is **no existing JS test harness**, and this plan recommends **not building one for this
feature**: the front end is deliberately thin (every validation/business rule lives server-side
and already has pytest coverage — the browser only renders responses and calls REST/JSON). The
highest-value verification is **qa-engineer's black-box pass** driving the actual running app
against the seeded demo workspace, directly exercising AC-1..AC-6 as written (they are already
phrased as Given/When/Then browser scenarios).

**Exception (v2 — resolves review finding m3), required not optional:** the inline run cue's
"most relevant run" tie-break (§3.3.2 — non-terminal status beats terminal, ties broken by most
recent `startedAt`) is the **one piece of genuine branching logic** in this whole plan that isn't
pure rendering, and U10 is a single manual black-box pass unlikely to hand-construct a multi-run,
mixed-status thread to exercise it. `frontend-engineer` must extract this rule into a small,
dependency-free pure function (input: a list of run summaries; output: the one to show) and cover
it with a handful of plain assertions runnable with a bare `node path/to/test.js` — no framework,
no DOM, no new dependency in `web/`. This is U6's responsibility (§4) and part of U6's done
condition, not a nice-to-have left to `frontend-engineer`'s discretion. Any *other* pocket of
client-side logic `frontend-engineer` finds worth isolating (e.g. "derive form fields from a
step's `config`") is still a reasonable, low-cost addition at their discretion — not required.

**U10 session shape (v2 — resolves review finding M2):** because `TRIGGER_DEF_KEY`/
`TRIGGER_DEF_VERSION` is a single process-wide env var (§7 risk #1, now resolved), AC-1 (needs the
plain-chat-reply resume path, `triage`-only) and AC-2 (needs the structured-input-form resume path,
`access-request`-only) **cannot both be verified in one continuous browser session.** U10 is
therefore **two passes with a server restart in between**, not a flat AC-1..AC-6 checklist:

- **Pass A — server started with default config (`FALKORCHAT_TRIGGER_DEF_KEY=triage`,
  `_VERSION=v1`, i.e. no override needed).** Verify:
  - AC-1 end to end (defs visible → `@mention` → cue appears → panel opens → progresses →
    parks → resumes via plain reply → reaches terminal state) with no reload/curl/terminal;
  - AC-3 (failure + reason readable) — force a failure under `triage` (e.g. step-budget
    exhaustion) if convenient; otherwise defer AC-3 to Pass B, whichever def makes forcing a
    failure easier;
  - AC-4 (participants list, both kinds distinguishable);
  - AC-5 (default layout no more crowded than today's) — load the page fresh and confirm (a) the
    **only** always-visible additions are the header "Workflow defs" button, the
    "Participants (n)" toggle, and the readiness badge — each sized/styled to match an existing
    header element, per the §3.3 v2 reading of AC-5 — and (b) **none of them expand, render, or
    fetch content** until clicked/expanded; the inline run cue and the run detail panel are absent
    entirely when no run exists / isn't open. This is not "zero new elements" (that's not what
    AC-5 says, and it's not what §3.3 ships) — it's "no busier," verified against the three
    specific affordances above;
  - AC-6 (ready/not-ready + names the offending def) against both a synced and a deliberately
    desynced workspace.
- **Restart the server** with `FALKORCHAT_TRIGGER_DEF_KEY=access-request`,
  `FALKORCHAT_TRIGGER_DEF_VERSION=v1` set.
- **Pass B — server restarted pointing at `access-request`/`v1`.** Verify:
  - AC-2 timing (`@mention` starts a run → it parks on a *structured*-input step → fill the form →
    submit → visible state change within 5s) — a stopwatch check against the 3000ms poll (plus the
    immediate re-poll on submit, §3.3.3) is sufficient, no special instrumentation needed;
  - AC-3, if not already covered in Pass A.

Record in the test report which pass covered AC-3, and confirm the server was restarted back to
the default (`triage`/`v1`) config afterward, since that default is what every other environment
(pytest, other manual checks) assumes.

---

## 6. FR/AC → unit traceability

| Requirement | Satisfied by |
|---|---|
| FR-1 (defs list + shape) | U3 |
| FR-2 (inline run cue) | U1, U4, U6 |
| FR-3 (run detail: status/steps/trace) | U6 |
| FR-4 (≤5s freshness, no reload) | §3.2 polling design, exercised by U6/U9, verified by U10 |
| FR-5 (waiting + prompt) | U9 |
| FR-6 (structured input form) | U9 |
| FR-7 (failure + reason) | U9 |
| FR-8 (thread participants) | U1, U5, U7 |
| FR-9 (minimalist default) | Design constraint on U3/U6/U7/U8/U9 (§3.3), verified by U10 |
| FR-10 (ready-to-demo) | U2, U8 |
| AC-1 (full M3 story, no terminal) | U3 + U6 + U9 (chat-triggered `triage`), verified by U10 Pass A |
| AC-2 (structured input, ≤5s) | U9, verified by U10 Pass B (chat-triggered `access-request`) |
| AC-3 (failure readable) | U9, verified by U10 (Pass A or B — see §5.2) |
| AC-4 (participants, kinds distinguishable) | U5 + U7, verified by U10 Pass A |
| AC-5 (default layout no more crowded than today's — §3.3 v2 reading) | All frontend units (design constraint), verified by U10 Pass A |
| AC-6 (ready/not-ready + names offender) | U2 + U8, verified by U10 Pass A |

---

## 7. Risks & open questions

1. **Resolved (v2) — how the demo path reaches a run parked on a *structured*-input step (needed
   for FR-6/AC-2).** The only chat-triggered def in the seeded environment is `triage@v1`
   (`kind: 'conversation'`, `type: 'agent'` steps, `waitsForHuman: true` but **no**
   `config.fields`/`expects` — it parks for a *plain chat reply*, by design; see
   `scripts/seed_workflows.sh:143-223`). The def whose steps declare `fields`/`expects`
   (`access-request@v1`, `proof_defs.py`) is only reachable today via `POST /workflow-runs`
   (untriggered start) — and building a UI affordance for that is exactly **FR-13, explicitly not
   committed**. `config.TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION` (`config.py:83-84`) is a single
   env-var knob read once at process startup (`trigger.py:41-49` constructs `WorkflowTrigger` with
   one fixed `def_key`/`def_version` for the whole process) — so the two defs cannot both be
   chat-triggerable within one running server.

   **Stakeholder decision (relayed via `teco`): go with this plan's original workaround (option
   1).** The demo operator temporarily points `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` at
   `access-request`/`v1` to exercise FR-6/AC-2 through the *same* chat-mention → cue → panel path
   built for `triage`, then restarts the server pointing back at `triage`/`v1` (the default) to
   exercise AC-1's plain-reply story — or the reverse order. **No FR-13 work, no dual-trigger
   config change, no "both chat-triggerable" scoping work.** This is an operational choice for the
   demo/QA session, not a product feature, and it closes this risk — it does not need revisiting at
   U9. The knock-on effect on the U10 test session shape is now spelled out explicitly in §5.2
   (finding M2 in the review this revision addresses).
2. **`WorkflowRun` label-scan cost (§3.1a).** Acceptable at demo scale; flagged for `GRAPH.PROFILE`
   verification in U1, with an indexed fallback identified if it isn't — **but the fallback's
   effectiveness for this exact `ORDER BY`-only query shape (no `WHERE`) is itself unverified
   against FalkorDB's planner** (v2 addition, review finding m1) and must be confirmed by PROFILE,
   not assumed; U1's done condition (§4) now requires that confirmation explicitly. Low risk, cheap
   mitigation either way, not blocking.
3. **BACKLOG.md entry.** This plan assigns K-036 by inspection (next free id after K-035) but does
   not edit `BACKLOG.md` (outside this agent's write scope) — `teco` should add the entry as part
   of standing documentation curation when this unit sequence is picked up.
4. **`verify_workflows.sh` dedup (§3.1c, U2) touches a script outside `server/`.** Low risk
   (mechanical), but flagged explicitly since it's a "should," not a hard requirement — if time
   pressure makes it not worth the diff, leaving the script's inline `DEFS` list as-is does not
   block this feature, it just re-introduces the exact duplication pattern this feature's design
   tries to avoid.
5. **No visual workflow diagram.** FR-1's "view a chosen def's shape" is designed as a plain
   text/table rendering of steps + transitions (§3.3.1), not a graph drawing — consistent with the
   minimalist mandate (FR-9) and avoids pulling in a rendering library. Flag if the stakeholder's
   expectation for "explain what the agent is about to run" is closer to a visual flow diagram —
   that would be a scope/tooling decision, not a small addition.

---

## 8. Review dispositions (gate, `docs/reviews/web-api-coverage.md`)

Verdict: *needs changes* — 1 blocker · 2 major · 3 minor · 1 nit. **All 7 adopted.** Nothing was
rejected on merits; the blocker was resolved by the architect (a five-minute design-document fix,
per the review's own framing) and the majors by a stakeholder decision (M2) and a factual
correction (M1).

| # | Finding | Disposition | Where |
|---|---|---|---|
| **B1** | §3.3's always-visible new chrome (defs button, participants toggle, readiness badge) contradicts §5.2's AC-5 instruction ("no new visual element ... until explicitly opened") | **Adopted, option (a) chosen** — §3.3 gained an explicit "AC-5 reading, decided here" note: the three trigger affordances stay always-visible-but-minimal (matched in weight to existing header elements), only their *content* is gated on interaction; AC-5's actual text ("no more crowded," not "zero new elements") supports this reading. §5.2's AC-5 verification instruction rewritten to check exactly those three affordances and that none render content unopened, instead of a "no new visual element" bar the design was never going to pass | §3.3 (new note before §3.3.1), §5.2 AC-5 bullet |
| **M1** | §2.3's rationale claims mention resolution checks channel membership — false per `repository.py:904-920`/`services.py:438-462`/`seed_demo.sh:23` | **Adopted** — the false claim removed and replaced with the verified fact (no `MEMBER_OF` check anywhere in mention validation, confirmed by the same three citations the review used); the channel-roster decision itself kept, now justified only on its two sound legs (zero new schema; heavier alternative); a new "Known, accepted gap" paragraph states the participants list is narrower than the truly-mentionable set | §2.3 |
| **M2** | Risk #1's `TRIGGER_DEF_KEY` single-process-var constraint isn't threaded through to §5.2's U10 session shape | **Adopted, with the stakeholder's resolution of risk #1 folded in** — risk #1 marked resolved (stakeholder/`teco`: original workaround, option 1 — temporary env-var point-and-restart, no FR-13 work); §5.2 rewritten from a flat AC-1..AC-6 list into an explicit two-pass structure (Pass A: default `triage` config, covers AC-1/AC-4/AC-5/AC-6 + optionally AC-3; restart; Pass B: `access-request` config, covers AC-2 + AC-3 if not already covered), with an explicit reminder to restart back to the default config afterward | §7 risk #1, §5.2 |
| **m1** | The `WorkflowRun.startedAt` index fallback's effect on an `ORDER BY`-only query (no `WHERE`) is unverified against the FalkorDB planner | **Adopted** — §3.1a and U1's done condition both gained an explicit caveat: PROFILE the exact query with the index in place and confirm the plan actually changes before relying on it; if it doesn't, change the query (e.g. add a supporting `WHERE`) rather than ship a no-op index. Risk #2 cross-references the same caveat | §3.1a, §4 U1, §7 risk #2 |
| **m2** | Structured-input submit doesn't reuse the "poll immediately after write" idiom (`app.js:298`) | **Adopted** — §3.3.3's FR-6 bullet and U9's done condition both now call for an immediate re-poll of the run on a successful submit, mirroring `pollMessages()` after `postMessage()` | §3.3.3, §4 U9 |
| **m3** | The FR-2 cue's "most relevant run" tie-break is real branching logic with only optional test coverage | **Adopted, promoted to required** — §5.2 now requires (not recommends) extracting the tie-break into a dependency-free pure function covered by plain `node`-runnable assertions; U6's done condition updated to match | §5.2, §4 U6 |
| **n1** | U9 cites "§6 risk" for where the demo-path-reachability risk lives; risks are actually in §7 | **Adopted** — corrected to "§7 risk #1 (now resolved)" | §4 U9 |

**Pass 2 — 2026-07-28.** New blocker found; the seven findings above stayed confirmed-resolved (see the review's own "Verification of Pass 1's other dispositions").

| # | Finding | Disposition | Where |
|---|---|---|---|
| **B2** | v2's B1 fix correctly reworded §3.3's new AC-5 note and §5.2's AC-5 bullet, but left the old, just-rejected "collapsed/unchanged-by-default" reading standing verbatim in three other spots: §3.3's own opening sentence, and U3's and U7's build-unit done-conditions in §4 (U7 additionally miscited AC-5 as requiring the stricter, rejected bar) | **Adopted (v3)** — §3.3's opening sentence reworded to distinguish content-collapsed-by-default from the three always-visible trigger affordances; U3's done-condition reworded to state the defs header button is always present and only its overlay *content* is gated on click; U7's done-condition reworded to state the participants toggle is always present and only its *content* is gated on expand, with the AC-5 mis-citation removed | §3.3 opening sentence, §4 U3, §4 U7 |
