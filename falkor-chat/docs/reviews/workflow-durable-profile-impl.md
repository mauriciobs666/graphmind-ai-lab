# Durable user-profile data for workflows — Implementation Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-054 (M6) · **Extends:** `docs/reviews/workflow-durable-profile.md`

## Scope & verdict

Reviewed the full combined diff of `663093d` (cluster 1: `repository.py`/`services.py` + their
tests) and `36096d0` (cluster 2: `tools.py`, `proof_defs.py` v3 bump, `seed_salesperson.sh`/
`verify_salesperson.sh`/`AGENTS.md`, `docs/QUERIES.md` §17, `test_queries.sh`, `test_tools.py`,
`test_salesperson_scaffold.py`) against `docs/plans/workflow-durable-profile.md` (architect),
`docs/plans/workflow-durable-profile-graph.md` `Version: 2` (graph-dba), and
`docs/requirements/workflow-durable-profile.md` (FR-1..FR-4, AC-1..AC-3). Static review only, per
the coordinator's operational note (no destructive suite run — a `qa-engineer` live pass is
concurrent against `ws:acme`). Read-only sanity checks (`verify_salesperson.sh acme`, a direct
`GRAPH.RO_QUERY` against `reference`) were run and are cited below as evidence, not as a substitute
for `qa-engineer`'s acceptance pass.

**Verdict: approve.**

**CPG:** not applicable — `cpg_falkorchat` is stale and, independent of that, this is new-code
review over four already-identified files (`repository.py`/`services.py`/`tools.py`/`proof_defs.py`)
read directly, per the coordinator's brief.

## Findings

### nit — commit `663093d`'s message overcounts cluster 1's new tests by one

**Evidence:** the commit message claims "12 new tests (repository + service level)." Counting
`^+def test_` across the diff of `test_repository.py` + `test_services.py`
(`git show 663093d -- server/tests/test_repository.py server/tests/test_services.py | grep -c
'^+def test_'`) gives **11**: six in `test_repository.py`
(`test_get_profile_none_when_no_customer_node_yet`,
`test_upsert_profile_full_write_creates_customer_and_sets_both_fields`,
`test_upsert_profile_on_existing_cart_only_customer_adds_profile_fields`,
`test_upsert_profile_partial_update_omitted_name_leaves_name_unchanged_ac2`,
`test_upsert_profile_partial_update_omitted_address_leaves_address_unchanged`,
`test_profile_persists_across_repository_instances_ac1`) and five in `test_services.py`
(`test_get_profile_returns_both_fields_none_when_no_customer_yet`,
`test_save_profile_full_write_then_get_profile_round_trips`,
`test_profile_persists_across_a_fresh_thread_same_actor_and_ws_ac1`,
`test_save_profile_partial_update_omitted_name_leaves_name_unchanged_ac2`,
`test_save_profile_partial_update_omitted_address_leaves_address_unchanged`).

**Why it matters:** purely a commit-message accuracy slip, not a code defect — the test coverage
itself is exactly what the message describes qualitatively (both AC-2 directions, at both layers).
Low stakes, but the review checklist calls for verifying counted claims rather than taking them on
faith, and this one is off by one.

**Suggested improvement:** none needed for the code; if the commit is ever amended for another
reason, correct "12" to "11." Not a gate.

## What's solid

- **Cypher fidelity is exact.** `repository.py`'s `upsert_profile` (new lines added at end of
  `repository.py`, §17 comment block) transcribes graph note §3 `Version: 2` verbatim: `MERGE` +
  `ON CREATE SET c.createdAt`, then `SET c.name = coalesce($name, c.name), c.deliveryAddress =
  coalesce($deliveryAddress, c.deliveryAddress), c.profileUpdatedAt = $now` — the exact
  `coalesce()`-per-field guard that closed the prior design-gate BLOCKER (v1's unconditional `SET`
  would have nulled an omitted field). `get_profile`'s Cypher matches graph note §4 exactly, and its
  `None`-on-zero-rows vs. both-fields-`None`-on-existing-`Customer` distinction is preserved, not
  collapsed. `docs/QUERIES.md` §17.1/§17.2 and `test_queries.sh`'s new §17 block transcribe the same
  query text a third time (shell/live-graph level) and match byte-for-byte.
- **The partial-update case is genuinely closed at three altitudes, not just asserted.** Unlike a
  service-level fake alone (which could mask a Cypher regression), `test_repository.py`'s six new
  tests run against the live `ws:test` graph fixture (`conftest.py`'s `repo`/`conn` fixtures wipe
  node data but keep schema, over a real FalkorDB connection — confirmed by reading
  `server/tests/conftest.py:85-97`) and directly exercise both partial-update directions
  (`test_upsert_profile_partial_update_omitted_name_leaves_name_unchanged_ac2` and its address-side
  sibling), including a `count(Customer) = 1` assertion ruling out a duplicate node from `MERGE`.
  `test_queries.sh`'s new §17 section repeats the same two-direction check a third time directly
  against the live shell/`redis-cli` path. `test_services.py`/`test_tools.py` add the same two
  directions again at the fake-service and stub-tool layers. This closes the exact gap the
  coordination brief flagged (a fake alone would not catch a Cypher-level regression) — confirmed by
  reading the tests, not by taking the implementer's kaizen note at its word.
- **`get_profile`'s always-populated shape and `save_profile`'s `None`-passthrough are both
  correct.** `services.get_profile` returns `{"name": None, "deliveryAddress": None}` on a missing
  profile, never a `{"found": false}` abstention (plan §3.2, matched). `tools.py`'s
  `SaveProfileTool.run` passes `arguments.get("name")`/`arguments.get("deliveryAddress")` straight
  through as `None` when the model omits an argument — never coerced to `""` or dropped — which is
  exactly what the `coalesce()`-guarded write downstream needs to treat as "leave unchanged."
  `test_tools.py`'s `test_save_profile_omitted_field_is_passed_through_as_none_not_dropped` pins
  this at the tool boundary specifically.
- **`config.model` survives the `v3` bump, confirmed against the actual diff and against the live
  graph, not just the commit message.** `proof_defs.py`'s diff shows `"model":
  "lmstudio/mistralai/ministral-3-3b"` still present, unchanged, inside the same `v3` `config` dict
  that gained `get_profile`/`save_profile` in `tools` and the extended `systemPrompt` — topology
  (2 steps: `assistant`/`agent`, `ended`/`decision`; 1 transition) is otherwise byte-identical to
  v2/v2.1. Independently confirmed live: `./scripts/verify_salesperson.sh acme` reports
  `salesperson@v3` in sync with the expected 2-step/1-transition topology, and a direct
  `GRAPH.RO_QUERY reference "MATCH (d:WorkflowDef {key:'salesperson', version:'v3'})-[:HAS_STEP]->
  (s:Step {key:'assistant'}) RETURN s.config"` shows `"model":"lmstudio/mistralai/ministral-3-3b"`
  in the actually-published node, not merely in source. `test_salesperson_scaffold.py`'s
  `test_salesperson_def_pins_ministral_model_and_version_bump` and the mutation-tested
  `test_v1_and_v2_coexist_in_the_same_workspace_without_conflict` (tool-count delta correctly bumped
  from `+5` to `+7` to account for the two new profile tools) both pin this.
- **Fit with codebase conventions is strong throughout.** No `ensure_customer` call needed before
  `upsert_profile` (correctly reasoned in both `repository.py`'s and `services.py`'s new comments:
  the `MERGE` creates the anchor itself, unlike `add_to_cart`), tool registration follows the
  existing "always register, fence by `config.tools` per node" posture verbatim, and
  `docs/QUERIES.md` §17 / `AGENTS.md`'s `seed_salesperson.sh` row / `test_queries.sh` were all
  updated in the same two commits — no doc drift left behind.
- **Coordination-doc self-verification is real, not just asserted.** `teco`'s entries for U23/U24 in
  `docs/plans/workflow-salesperson-demo-coordination.md` describe independently re-running the
  offline suite, independently re-running `test_queries.sh`, and independently mutation-testing both
  the `coalesce()` guard and the `config.model` carry-forward on copied-aside files — this matches
  what a fresh read of the diff and a live topology check corroborate.

## Open questions

None. The one nit above does not gate anything; `qa-engineer`'s concurrent live acceptance pass
(against `ws:acme`/a fresh throwaway workspace) is the remaining gate before this milestone item can
be considered fully closed, and is out of this review's scope by the coordinator's own operational
note.
