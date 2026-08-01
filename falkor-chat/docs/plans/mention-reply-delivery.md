# `mention-reply-delivery` — CI blind-spot follow-up (K-039 item 3)

> **Status:** archived · **Owner:** `architect` · **Tracks:** K-039 (M3.5)
>
> **Version:** **v2 — 2026-07-31** (revision pass after the `analyst` plan-gate review returned
> *approve with suggestions*: 4 minor · 1 nit, no blockers — the core §3 decision was confirmed
> sound and unchanged). v1 = 2026-07-31, design complete. Every finding's disposition is recorded
> in **§7**. **Review:** `docs/reviews/mention-reply-delivery.md`.

## 1. Goal & scope

Close **K-039 item 3** (`docs/BACKLOG.md`, K-039 entry, scope item 3, ~line 1167): the RCA's
contributing factor 2 (`docs/reviews/mention-reply-delivery-rca.md` §4) found that `pytest -m live`'s
AC-4 answer-post assertion — the only test that would have caught "workflow completes `done` but
posts nothing" — is excluded from the default `pytest -q` loop (`server/pyproject.toml`,
`addopts = -m "not live"`), so a green baseline gave false confidence about the exact path that was
broken. §5 item 2 offers two non-exclusive directions to close that gap; this plan investigates both
and decides.

**In scope:** deciding and designing the CI-blind-spot fix (or combination), a small readiness-signal
addition if that's the chosen (or partial) direction, and the one-time verification task the shared
K-039 done-condition still owes (confirming `pytest -m live`'s AC-4 assertion now passes after item
1's fix, and correcting the stale "known-RED" note it left behind).

**Out of scope:** the full K-027 item 2 "terminal-node-must-post" engine contract (separately owned,
`architect`-designed when it is picked up); any unrelated K-036 follow-up; re-opening the `pytest -q`
"network-free by default" design decision itself (see §3 — reopening it is exactly what this plan
declines to do, and why).

---

## 2. Context & findings

### 2.1 What item 1 already shipped (2026-07-31, `docs/HISTORY.md`)

`executor.py`'s `_run_agent_node` now synthesizes an implicit `post_message` call when a node's
granted tools include `post_message`, the loop ends via the non-tool-call branch with non-empty
`result.text`, and nothing has posted yet this loop. This is verified by **five new, deterministic,
network-free tests already running in the default `pytest -q` baseline** (642 → 647 passed):

- `server/tests/test_executor_agent.py`:
  `test_plain_text_with_granted_post_message_is_posted_as_implicit_fallback` (a **fake LLM stub**
  returning plain text — the exact shape the RCA live-reproduced), plus a rejected-mention-recovery
  case and two negative guards.
- `server/tests/test_executor_produced.py`:
  `test_implicit_post_when_tool_not_called_still_creates_produced_edge_live` — full integrated path
  (real `Services`/`ToolRegistry`, live `ws:test` graph), asserting a real `Message` + `StepRun
  -[:PRODUCED]-> Message` edge, mirroring the RCA's live repro exactly.

**This matters directly for item 3's design decision** (§3): the specific regression class the RCA
flagged — a `type:'agent'` node ending on plain text instead of dispatching its granted
`post_message` tool, silently discarding the reply — already has a **deterministic, offline,
default-loop regression test**. It does not depend on a real LLM's mood, a real LM Studio instance,
or network reachability. If this exact defect reappears (e.g. a future refactor of
`_run_agent_node` reintroduces the silent-discard branch), `pytest -q` — no flags, no live
dependency — goes red today, in the default loop that's actually run day to day. That is the literal
ask in RCA §4 contributing factor 2 ("nothing in the normal CI-equivalent loop signals this is
currently broken") — already met, for the specific mechanism that broke.

### 2.2 The `pytest -m live` test itself — why it is not a fast, narrow signal

`server/tests/test_workflow_live.py`, `test_triage_flow_runs_end_to_end_against_live_llm` (the only
test carrying `pytestmark = pytest.mark.live`, i.e. the whole module is gated):

- Requires **both** FalkorDB and a real LM Studio (chat + embedding model) reachable
  (`live_dim`/`live_ws` fixtures, `pytest.skip` with a reason otherwise — the precedent the RCA
  points at for a "reachability-gated" test).
- Drives a **multi-round conversation** against a live, fuzzy LLM-judged guard (up to
  `MAX_CLARIFY_ROUNDS = 4`, `intake → research → answer`), asserting AC-1 through AC-4 together, not
  just "did the answer node post something." It probes the live embedding dimension, bootstraps a
  throwaway `ws:live`, publishes the real triage def via `seed_workflows.sh`, and seeds a 5-message
  embedded corpus — this is an end-to-end scenario test, not a unit-sized regression check.
  Non-deterministic round count and judge behavior are explicitly called out in the module's own
  docstring as expected, not a bug.
- **This is deliberate, not an oversight.** `server/pyproject.toml`'s own `addopts` comment records
  *why* `-m "not live"` exists, in these words: *"a reachability-skip alone would not [keep the
  standard run fast] — it would run them"* whenever LM Studio happens to be up. That is exactly the
  scenario this plan's brief describes for option (a): promoting a reachability-gated version into
  the default loop. The project already deliberately rejected that shape for this reason, in the
  same file the option would need to change.

### 2.3 Existing readiness surface (K-036) to extend for option (b)

- `Services.check_demo_readiness` (`server/falkorchat/services.py:1001`), routed at
  `GET /workspaces/{ws}/readiness` (`server/falkorchat/api.py:~413`). Purely **structural** today:
  presence/sync/multi-start checks per `DEMO_EXPECTED_DEFS` pair (`services.py:357`); always `200`,
  `ready` is a report, never an error.
- Web banner: `web/app.js` `loadReadiness()`/`renderReadinessBadge()`/`renderReadinessPanel()`
  (`app.js:606-632`), fetched once on load (`web/app.js:607` — "not on the 5s freshness bar... a
  manual Recheck covers it"), `#readiness-badge`/`#readiness-panel`/`#readiness-content` in
  `web/index.html` (badge styling `.readiness--ready`/`.readiness--not-ready` at `index.html:95-99`).
- `WorkflowRun` schema/indexes (`scripts/bootstrap_schema.sh:122-159`): indexes on `runId`,
  `status`, `startedAt`; **no index on `defKey`**. Comment at `bootstrap_schema.sh:148-156` notes
  `WorkflowRun` cardinality is tiny per workspace, and the `startedAt` index exists to anchor an
  index scan even though the predicate it's paired with (`>= 0`) is "functionally a no-op" — the
  established precedent for a residual-filtered, index-anchored read is
  `Repository.find_runs_for_thread` (`repository.py:1511-1544`, doc comment 1514-1526, the cited
  clause at `WHERE r.startedAt >= 0 AND m.threadId = $threadId`, ~1529-1531).
  **Nuance (added at the plan-gate review, §7 finding 2):** that idiom is load-bearing there
  specifically because its *other* filter (`m.threadId = …`) sits on a **different** pattern
  variable (`m`, pulling the anchor off `r`). §12.15's filters (`defKey`/`defVersion`/`status`)
  are all on `r` itself, and `WorkflowRun.status` is independently indexed too
  (`bootstrap_schema.sh:145-146`) — a materially different planner situation, so "copies the
  established idiom" only partially transfers. §4 Step 1 now asks `graph-dba` to record *which*
  index the query actually lands on, not just confirm "an index scan."
- `StepRun -[:PRODUCED]-> Message` (D2) is the existing provenance edge item 1's fix already
  populates (`QUERIES.md` §12.6, `repository.link_step_emission`).

### 2.4 The RCA's own preferred phrasing for (b)

RCA §5 item 2 explicitly suggests *"last N triage runs: N posted a reply / N did not"* — a **count**,
not a time window. This resolves an otherwise-open design question (time window vs. fixed sample)
in the RCA's own words; §3.3 below follows it.

### 2.5 What's still owed regardless of (a)/(b): a stale doc note

`docs/BACKLOG.md` K-027, "Addendum from the K-025 QA pass" (~line 431-433): *"Also recorded:
`pytest -m live` is RED deterministically (2/2) on the AC-4 answer-post assertion — a known, filed
limitation (D12-B), not an unknown regression."* Item 1's fix targets exactly this failure mode, so
this note is now very likely **stale** — but nobody has re-run `pytest -m live` since the fix landed
(`docs/HISTORY.md`'s 2026-07-31 entry explicitly says so: *"the 1 deselected is the known
`@pytest.mark.live` characterization test, unaffected... not in scope here"*). The shared K-039
done-condition (`docs/BACKLOG.md` ~line 1173-1178) says this assertion "should also flip from its
documented deterministic-RED to green, or its known-limitation note should be corrected" — this is
still an open, one-time action independent of which of (a)/(b) gets built.

---

## 3. Design & rationale

### 3.1 Decision: decline (a) as literally framed; do not promote any `@pytest.mark.live` test into the default loop

**Rejected**, for a reason grounded in the repo's own recorded design intent (§2.2), not just cost:
promoting a reachability-gated live test into `pytest -q`'s default `addopts` would reintroduce
exactly the failure mode `-m "not live"` was written to prevent — a plain `pytest -q` starting to
make real network/LLM calls, silently, whenever a developer happens to have LM Studio open (which,
per the RCA's own session, is a normal and even common state during demo prep — the environment
where this exact bug was found). It would also import `test_workflow_live.py`'s scenario-level
non-determinism (a fuzzy-judged, up-to-4-round conversation) into what is supposed to be the fast,
deterministic default loop, for a defect class that offline testing already covers deterministically
(§2.1). Weighed against the brief's own framing: (a)'s upside — "a direct, deterministic regression
signal for this exact defect class" — is **already delivered**, just not via a *live* test; its
downside (flake/cost profile, LM-Studio-in-CI availability) is real and avoidable. There is no
residual gap left that a live default-loop test would close and the offline tests don't.

*(If a narrower live smoke test — one real LLM call per node, no multi-round guard — were built
instead of promoting the existing scenario test, it would still make `pytest -q` non-network-free
whenever LM Studio is reachable, which is the property `addopts` was written to guarantee regardless
of reachability. The same objection applies to any live-gated addition to the default `addopts`, not
just to promoting the existing test verbatim.)*

### 3.2 Decision: build (b) — a "recent triage post-success" signal on the existing readiness route

Cheap, always-on, and — unlike any pytest run — exercises **real production data**, so it is the
only one of the two directions that would catch a *future* regression class this repo hasn't seen
yet (e.g. a model swap that reintroduces silent-discard behavior in a different shape, or a partial
regression the offline fake-stub tests don't happen to construct). It is explicitly a **lagging**
indicator (only informative after runs have happened, per the brief) and gates nothing in CI — that
trade-off is accepted, because CI-equivalent gating for *this* defect class is already covered by
§2.1's offline tests. `check_demo_readiness` already exists as "the pre-demo check surface" (RCA §5
item 2's own suggestion), so this is additive to an established seam, not a new one.

### 3.3 Shape of the signal

- **Sample:** last `N` (constant `POST_SUCCESS_SAMPLE_SIZE = 20`, mirroring the
  `RAG_QUERY_TIMEOUT_MS`-style plain module constant already used in `services.py:99` — not an env
  var; nobody has asked to tune this per-deployment) **terminal** (`status IN ['done', 'failed']`)
  runs of `config.TRIGGER_DEF_KEY`@`config.TRIGGER_DEF_VERSION` — the `@mention`-triggered def, by
  `startedAt DESC`. `waiting`/`running` runs are excluded from the sample (they haven't reached a
  verdict yet); a `failed` run counts as "did not post" alongside a `done`-but-silent one, since both
  are equally bad for a demo.
- **Scope is deliberately narrow to the mention-triggered def, not every `DEMO_EXPECTED_DEFS`
  pair.** `access-request@v1` is a different flow shape (business-process, not necessarily
  "answer via chat"); folding it in would conflate two different "must-communicate" contracts and
  is exactly the kind of scope creep the brief asks to avoid.
- **Metric:** `postedCount` = how many of those runs have at least one
  `StepRun -[:PRODUCED]-> Message` edge; `sampleSize` = how many runs were sampled; `rate =
  postedCount / sampleSize` (or `null` if `sampleSize == 0` — a fresh workspace with no triage runs
  yet is "no data," not "0% healthy").
- **Deliberately informational, not folded into the existing boolean `ready`.** `ready` today is a
  purely structural, deterministic signal (def published/materialized/in sync/no multi-start) with
  a clear, actionable "problems" remediation per offender. Mixing in a model-behavior-driven,
  inherently noisier metric would make `ready` flip on LLM mood rather than configuration state,
  undermining the one guarantee that route currently makes reliably. The new field
  (`postSuccess.status`: `"ok"` / `"degraded"` / `"no-data"`) is surfaced **alongside** `ready`, not
  inside its computation. Flagged as an open question in §6 in case product ownership disagrees.

### 3.4 Alternatives considered and rejected

- **Promote the whole `test_workflow_live.py` test as-is, unconditionally, into `addopts`** —
  rejected outright per §3.1; breaks the documented "network-free by default" property for every
  contributor, not just this test.
- **A brand-new, narrower live smoke test, reachability-gated into the default loop** — rejected;
  same objection as above (§3.1 parenthetical) — reachability-gating alone does not preserve
  network-free-when-LM-Studio-is-down *and* fast-and-deterministic-when-it's-up simultaneously,
  which is the actual property the repo has committed to.
- **Time-windowed sample (e.g. "last 24h") instead of last-N** — rejected in favor of the RCA's own
  explicit "last N" wording (§2.4); a time window also degrades awkwardly for a bursty demo
  workspace (zero runs in the last hour reads identically to "healthy" and to "untested").
- **Fold `postSuccess` into `ready`** — rejected, see §3.3.

---

## 4. Step-by-step implementation

Sequencing follows the repo's established graph-dba-gate → coder pattern (e.g. K-008): the query is
authored/verified/PROFILEd first, then the service/API/web layers consume it. This is an **additive
read composition with a fully-specified I/O contract up front** (this plan pins the exact query,
fields, and edge cases) — there is no ambiguous behavior to discover through red/green iteration, so
step 2 is sized for `coder`, not `tdd-engineer`; `tdd-engineer` is the right call for a bug-fix or a
contract that needs discovering, which this isn't.

### Step 1 — `graph-dba`: author, verify, and PROFILE the new read

- Add **§12.15** to `docs/QUERIES.md` (following the existing §12.x numbering), proposed form (to be
  verified/tuned against the live instance, not treated as final):

  ```cypher
  MATCH (r:WorkflowRun)
  WHERE r.startedAt >= 0
    AND r.defKey = $defKey AND r.defVersion = $defVersion
    AND r.status IN ['done', 'failed']
  WITH r ORDER BY r.startedAt DESC LIMIT $limit
  OPTIONAL MATCH (r)-[:HAS_STEP_RUN]->(:StepRun)-[:PRODUCED]->(m:Message)
  WITH r, count(m) AS producedCount
  RETURN count(r) AS sampleSize,
         sum(CASE WHEN producedCount > 0 THEN 1 ELSE 0 END) AS postedCount
  ```

  This starts from the `r.startedAt >= 0` index-anchor idiom used in
  `find_runs_for_thread` (`repository.py:1511-1544`, WHERE clause ~1529-1531) rather than inventing
  a new anchor — but that idiom's rationale doesn't fully transfer (§2.3's added nuance: there, the
  anchor is load-bearing because the *other* filter sits on a different pattern variable; here,
  `defKey`/`defVersion`/`status` are all on `r`, and `WorkflowRun.status` is independently indexed).
  So confirm via `GRAPH.PROFILE`, against both `ws:test` and (read-only) `ws:acme`, **which index the
  query actually lands on** (`startedAt` vs. `status`) and record that explicitly in the §12.15
  writeup — not just "an index scan, not a label scan" (AGENTS.md rule 3). **Edge case, corrected at
  the plan-gate review (§7 finding 1) — live-verified against the pinned FalkorDB build
  (`v4.18.11`) through the project's actual `falkordb-py` client:** `sum(CASE WHEN … THEN 1 ELSE 0
  END)` never returns `NULL`/`None`, in either the zero-row or non-empty case — it returns a Python
  **`float`** (`0.0`/`1.0`/…). `count(r)` (`sampleSize`) stays a clean `int`. Record this observed
  Python-side result-type pair (`sampleSize: int`, `postedCount: float`) directly in the §12.15
  QUERIES.md writeup, so step 2 doesn't have to rediscover it by inspecting `res.result_set` by
  hand.
- **RAM/index implication (AGENTS rule 6): none.** No new index, no new label, no new property. The
  query reuses the existing `WorkflowRun.startedAt` index; `WorkflowRun` cardinality is tiny per
  workspace by the bootstrap script's own comment, so no new index is needed for `defKey`/`status`
  even as a residual filter.
- Add a `./scripts/test_queries.sh` assertion (small fixture: a couple of `done` runs, one with a
  `PRODUCED` edge and one without, plus a `failed` run with none), pushing the suite past its current
  **276/276** baseline (`docs/QUERIES.md` header) — enumerate the new count in the same header line,
  per existing convention.
- **Done:** QUERIES.md §12.15 verified live with a PROFILE excerpt; `test_queries.sh` green at the
  new count.

### Step 2 — `coder`: repository method + service/API wiring

- `Repository.read_recent_post_success(ws, *, def_key, def_version, limit) -> dict[str, int]`
  (`server/falkorchat/repository.py`, near `find_runs_for_thread` §12.14, ~line 1419) — 1:1 the
  query from step 1. **Corrected at the plan-gate review (§7 finding 1):** the real gotcha is not a
  `None`/`NULL` to coalesce — `sum()` over this `CASE` expression never returns `NULL`, it returns a
  `float` in both the empty and non-empty case (verified live, §4 Step 1). Cast `postedCount` to
  `int` before building the dict (e.g. `int(posted_count)`), so `sampleSize`/`postedCount` are both
  clean `int`s in the returned dict — left uncast, the JSON response would carry `"postedCount":
  1.0` instead of `1`, and step 3's banner text would literally render `"1.0/2 replied"`.
- `services.py`: a module constant `POST_SUCCESS_SAMPLE_SIZE = 20` near `RAG_QUERY_TIMEOUT_MS`
  (`services.py:99`). Extend `check_demo_readiness` (`services.py:1001`) to also call
  `self.repo.read_recent_post_success(ctx.ws, def_key=config.TRIGGER_DEF_KEY,
  def_version=config.TRIGGER_DEF_VERSION, limit=POST_SUCCESS_SAMPLE_SIZE)` and add a `postSuccess`
  key to the returned dict:

  ```python
  {
      "defKey": config.TRIGGER_DEF_KEY,
      "defVersion": config.TRIGGER_DEF_VERSION,
      "sampleSize": sample_size,
      "postedCount": posted_count,
      "rate": (posted_count / sample_size) if sample_size else None,
      "status": (
          "no-data" if sample_size == 0
          else "ok" if posted_count == sample_size
          else "degraded"
      ),
  }
  ```
  Backward-compatible: an added key on an existing, `response_model`-less route
  (`api.py:check_demo_readiness` declares no `response_model` — confirmed, no `schemas.py` change
  needed).
- No `api.py` change needed beyond what's already wired (`GET /workspaces/{ws}/readiness` already
  calls `services.check_demo_readiness(ctx)` and returns it verbatim).
- **Tests to add** (coder writes these as part of "done," following the repo's existing test
  layering for this exact feature):
  - `server/tests/test_repository.py`: near the existing `find_runs_for_thread`/`link_step_emission`
    tests (~line 1330-1470) — seed 2-3 `WorkflowRun`s via the existing `_start`/`_start_at` +
    `complete_run`/`fail_run` + `link_step_emission` helpers already used there; assert
    `sampleSize`/`postedCount` for: all posted, some posted, none posted, zero runs (fresh
    workspace — assert `postedCount == 0`, not an exception), `limit` truncation, and that a
    `waiting`/`running` run is excluded from the sample.
  - `server/tests/test_services.py`: extend the `FakeRepo` (near `runs_by_thread`,
    ~`test_services.py:73`) with a scriptable `post_success_result` (mirroring the
    `start_run_result = _UNSET` override pattern already in that file), and add cases alongside the
    existing `test_check_demo_readiness_*` tests (~line 1274) asserting `postSuccess.status` for
    `"ok"`/`"degraded"`/`"no-data"`, and that `postSuccess.rate` is exactly `postedCount/sampleSize`
    (or `None`).
  - `server/tests/test_api.py`: extend the existing `test_readiness_route_*` tests (~line 763) to
    assert the response body now also carries `postSuccess` with the expected shape. **Flagged at
    the plan-gate review (§7 finding 3) — this isn't just an addition, it's a required fix:**
    `test_api.py:757` defines `_READINESS_KEYS = {"ready", "defs"}` and
    `test_readiness_route_not_ready_when_nothing_seeded` (`test_api.py:768`) asserts an **exact**
    `set(body) == _READINESS_KEYS` equality — landing `postSuccess` breaks that assertion
    immediately, independent of any new assertion added. Widen `_READINESS_KEYS` to `{"ready",
    "defs", "postSuccess"}` as part of this step, not as a follow-up red-test discovery.
- **Done:** `pytest -q` stays green with the new tests included (deterministic, no live dependency);
  suite count enumerated in the same style as prior entries.

### Step 3 — `frontend-engineer`: surface the signal in the K-036 readiness banner

- `web/app.js` `renderReadinessPanel` (`app.js:622-627`): render `report.postSuccess` as an
  additional line under the existing `problems`/"all in sync" content — e.g. `"Recent triage
  post-success: {postedCount}/{sampleSize} replied"`, or a "no runs yet" message when `status ===
  "no-data"`. Visually distinct (not necessarily alarming-red) when `status === "degraded"` — a new,
  small CSS class in `web/index.html`'s existing `<style>` block (alongside `.readiness--ready`/
  `.readiness--not-ready`, `index.html:95-99`), reusing the existing warning-toned palette rather
  than inventing a new one.
- **Explicit, deliberate non-change:** `renderReadinessBadge`/`#readiness-badge`'s
  ready/not-ready color stays driven by `report.ready` alone (§3.3) — do not fold `postSuccess` into
  the badge's color logic.
- **Done:** manual check against a running server (existing convention for this UI per `HISTORY.md`
  K-036 U8/U9 entries — "no automated JS test harness... front end is thin pass-through UI").

### Step 4 — `qa-engineer`: acceptance pass + the outstanding live-verification action

Two independent pieces, both owed before this can close:

1. **Acceptance of the new banner/route** (per the brief: "qa-engineer for acceptance if the web
   banner changes" — it does). Drive the running app/`GET /workspaces/acme/readiness` directly;
   verify the three `postSuccess.status` states render distinctly. **Citation corrected at the
   plan-gate review (§7 finding 4):** the RCA's own live-repro run (`runId
   00d95a27ac2a4dc8b74a86ed117b5c95`) no longer exists — it was deleted by the same-day `ws:acme`
   cleanup entry in `docs/HISTORY.md` (2026-07-31, listed after the immediate-mitigation entry).
   What's actually still live in `ws:acme` (three `triage@v1` `WorkflowRun`s total): one `done`
   with a `PRODUCED` message (`6dea1ba3c5d543cebf5f5a578ad07073`, the RCA's separately-noted
   corroborating run, left untouched by the cleanup), one `done` with zero `PRODUCED` messages, and
   one still `waiting` (correctly excluded by this signal's own terminal-status filter) — a real,
   live 1/2 "degraded" case, ready to exercise as-is. There is still no live "ok"-only example, so a
   synthetic/throwaway workspace is still needed for that state (and for "no-data").
2. **The outstanding shared K-039 done-condition action** (§2.5, independent of steps 1-3): run
   `.venv/bin/python -m pytest -m live -s` once, with LM Studio reachable, and confirm
   `test_triage_flow_runs_end_to_end_against_live_llm`'s AC-4 assertion now passes (or record why it
   still doesn't, if the live model's guard/tool-call behavior has an unrelated issue). Then correct
   `docs/BACKLOG.md` K-027's "Addendum from the K-025 QA pass" note (~line 431-433) — either flip the
   "RED deterministically (2/2)" claim to reflect the fix, or narrow it precisely if the test still
   fails for a *different*, now-isolated reason.
- **Done:** a short qa note/finding for each of the two pieces (does not need a full new
  `docs/test-reports/*` document unless a defect is found — this is a small acceptance check, not a
  fresh test-plan cycle; if a defect is found, follow the normal test-report path).

### Step 5 — doc close-out

- `docs/HISTORY.md`: one entry for this change (query + service/API + web + the live-verification
  outcome), dated at delivery.
- `docs/BACKLOG.md` K-039: mark item 3 delivered (same style as item 1's ✅ annotation), and update
  the entry's own status line/summary now that only the full K-027 item 2 contract remains open.
  `docs/BACKLOG.md` K-027's stale note is corrected in step 4.
- Plan/review `Status:` flips per root `AGENTS.md`'s per-kind table once implemented and re-gated:
  this plan (`docs/plans/mention-reply-delivery.md`) flips by `architect`; the coordination doc
  (`docs/plans/mention-reply-delivery-coordination.md`) flips by `teco`.

---

## 5. Test strategy

All new behavior is deterministic and offline (no `@pytest.mark.live` marker anywhere in this
plan's own deliverable — consistent with §3.1's decision):

1. **Repository layer** (`test_repository.py`, live `ws:test` integration, existing convention):
   `read_recent_post_success` returns correct `sampleSize`/`postedCount` for — all-posted,
   some-posted, none-posted, zero-runs, `limit` truncation, and exclusion of non-terminal
   (`waiting`/`running`) runs from the sample.
2. **Service layer** (`test_services.py`, `FakeRepo`): `check_demo_readiness`'s `postSuccess` field
   composition — `"ok"`/`"degraded"`/`"no-data"` status derivation, `rate` arithmetic (including the
   `None`-when-zero case), and that it does not affect the existing `ready` boolean's test cases
   (regression: the five existing `test_check_demo_readiness_*` tests must stay green unmodified in
   their assertions about `ready`/`defs`).
3. **API layer** (`test_api.py`, `wf_client`): the route's JSON body carries `postSuccess` alongside
   the existing `ready`/`defs` keys — **including widening `_READINESS_KEYS`** (`test_api.py:757`)
   to `{"ready", "defs", "postSuccess"}`, or `test_readiness_route_not_ready_when_nothing_seeded`
   (`test_api.py:768`)'s exact key-set assertion fails the moment `postSuccess` lands (§7 finding 3).
4. **Query suite** (`test_queries.sh`): the new §12.15 fixture (mixed posted/unposted/failed runs),
   asserted against a live FalkorDB instance, `GRAPH.PROFILE`-checked to confirm which index
   (`startedAt` vs. `status`) the plan actually lands on, not just that it's an index scan.
5. **Web** (manual, per existing convention for this thin UI layer): the three `postSuccess.status`
   states render as expected in the readiness panel; the badge's ready/not-ready color is unaffected.
6. **The one-off live-verification action** (not a permanent test, a one-time execution + doc
   correction): `pytest -m live` re-run once LM Studio is reachable, confirming AC-4 now passes.

Edge cases already called out inline above: zero-sample "no-data" (must not divide by zero or read as
0% healthy), `sum()` over this `CASE` expression returning a Python `float` in both the empty and
non-empty case — never `NULL`/`None` — so `postedCount` needs an explicit `int()` cast (§4 Steps 1-2,
corrected at the plan-gate review, §7 finding 1), non-terminal runs excluded from the sample, and
`access-request@v1` explicitly out of this signal's scope.

---

## 6. Risks & open questions

- **Open question (product-scope, not mine to decide silently):** §3.3 keeps `postSuccess` purely
  informational and never flips the existing `ready` boolean. If whoever owns the demo-readiness
  product surface wants a persistently "degraded" post-success rate to actually block "ready to
  demo," that's a deliberate product decision this plan declines to make unilaterally — flagged here
  rather than guessed. The design as specified is easy to extend that way later (the `status` field
  already carries the signal `ready`'s computation would need) if that call is made.
- **Risk — the live-verification action (step 4.2) could reveal AC-4 still fails, for a different
  reason.** Item 1's fix targets the two failure shapes the RCA measured; if the live model exhibits
  a third shape (e.g. a genuinely malformed tool-call the parser still can't recover), the "known-RED"
  note in K-027 shouldn't be blindly flipped to green — the step already accounts for this ("or
  record why it still doesn't").
- **Risk — none identified for the query/schema side.** No new index, no new node/edge type, no
  RAM change (§4 Step 1); `WorkflowRun` cardinality is small per the bootstrap script's own comment.
- **Not a risk, a deliberate non-goal:** this plan does not attempt to make `pytest -q` catch a live
  LLM behaving badly in general — that is what `pytest -m live` (run manually/on-demand) and, longer
  term, K-027's judge-calibration items are for. Item 3's job was narrowly "close the false-confidence
  gap for the specific defect K-039 found," which §2.1 shows is already closed for the *mechanism*;
  §3.2's signal closes it for *production drift* going forward.

---

## 7. Review dispositions (plan-gate review, `docs/reviews/mention-reply-delivery.md`)

Verdict: **approve with suggestions** — 4 minor · 1 nit, no blockers. **All 5 adopted.** The core
§3 decision (decline promoting any `@pytest.mark.live` test into the default loop; build the
readiness-route post-success signal instead), the RAM/index claim, the scope discipline, and the
unit sequencing/ownership were all independently verified by the reviewer (live queries against the
pinned FalkorDB build, a live `ws:acme` check, and a fresh `pytest -q` run) and confirmed sound —
**nothing here changes §3's design or decision**, only citations and implementation-level guidance.

| # | Finding | Disposition | Where |
|---|---|---|---|
| **Minor 1** | `sum()`-over-zero-rows edge case stated as `NULL`/`None`; actually a `float` (`0.0`), never `NULL`, in both empty and non-empty cases — the real gotcha is int/float type, not a coalesce | **Adopted** — §4 Step 1's edge-case note rewritten to describe the live-verified `float` behavior and instruct `graph-dba` to record the Python-side result types in QUERIES.md §12.15; §4 Step 2 changed from "coalesce `None` to `0`" to "cast `postedCount` to `int`" | §4 Step 1, §4 Step 2, §5 closing edge-cases line |
| **Minor 2** | Plan cites a nonexistent method, `Repository.read_thread_workflow_runs`, in two places — the real method at that location is `find_runs_for_thread` | **Adopted** — both citations (§2.3, §4 Step 1) corrected to `find_runs_for_thread`; also folded in the reviewer's planner-situation nuance (the `startedAt >= 0` idiom is load-bearing there because its other filter sits on pattern variable `m`, not `r` — a materially different situation from §12.15's all-on-`r` filters) and instructed `graph-dba` to record *which* index (`startedAt` vs. `status`) the query actually lands on | §2.3, §4 Step 1 |
| **Minor 3** | `test_api.py:757`'s `_READINESS_KEYS = {"ready", "defs"}` exact-key-set assertion breaks the moment `postSuccess` lands, and the plan didn't say so | **Adopted** — §4 Step 2's test list now names the exact assertion and instructs widening it to `{"ready", "defs", "postSuccess"}`; also called out in §5's API-layer test-strategy line | §4 Step 2, §5 |
| **Minor 4** | §4 Step 4 cites `runId 00d95a27ac2a4dc8b74a86ed117b5c95` in `ws:acme` as still-live QA evidence; it was deleted by the same-day cleanup entry in `docs/HISTORY.md` | **Adopted** — citation replaced with the still-live corroborating run (`6dea1ba3c5d543cebf5f5a578ad07073`) and the reviewer's live-queried current state of `ws:acme` (3 `triage@v1` runs: one posted, one didn't, one still `waiting`) — a real live "degraded" case exists; the existing hedge that a throwaway workspace is still needed for the "ok"/"no-data" states is kept | §4 Step 4 |
| **Nit** | `postedCount`'s `float` type is worth a one-line callout in QUERIES.md §12.15 itself (authored in step 1), not only in the service-layer guidance (step 2) | **Adopted** — folded into the same §4 Step 1 edit as Minor 1 (the QUERIES.md writeup instruction now explicitly names both result types) | §4 Step 1 |
