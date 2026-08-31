# Salesperson tool-orchestration reliability — K-057 + K-058 — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-057, K-058 (post-M6, not a milestone gate)

Coordinating the investigation and (where warranted) fix of the two live, unresolved
`salesperson@v4` tool-orchestration defects on `mistralai/ministral-3-3b` surfaced during M6's
close-out: **K-057** (intermittent self-contradictory/incomplete answer on a compound
category+price filter question, `docs/BACKLOG.md` K-057) and **K-058** (silent duplicate
write-tool re-fire on a follow-up instruction, `docs/BACKLOG.md` K-058). Both are in the same root
class — this model's tool-orchestration reliability under this scaffold — but are mechanistically
distinct defects with different evidence bases and different next steps; they are coordinated
together because they share a live-testing environment and a natural stopping point (a combined
regression pass once both are addressed), not because they're expected to share one fix.

**Prior art (read before dispatching or picking this up cold):**
- `docs/reviews/salesperson-tool-reliability-ml.md` — the `data-scientist` diagnostic note for the
  whole model-reliability thread. §1-§7 diagnosed and resolved K-056 (qwen3-4b skip-and-fabricate,
  now `HISTORY.md`-recorded as resolved-by-model-swap). §8 is the controlled alternative-model eval
  that picked `ministral-3-3b` and, as a byproduct, first surfaced K-058's duplicate-instruction
  defect (§8.4, n=10, 30%). §9 is the scoped K-058 follow-up eval that narrowed the estimate (n=32,
  6 conditions, 4.2% pooled) and proposed but did not implement a candidate fix (§9.4: a
  dispatch-time check that a write tool's resolved target appears in the current turn's own text).
  §10 is an unrelated model-landscape survey, not needed for this thread.
- `docs/reviews/salesperson-tool-reliability-impl.md` — the `analyst` diff review of K-056's two
  falsified/reverted scaffold-level mitigation attempts. Relevant here only as precedent for what
  NOT to do (a naive dedup-by-signature for K-058 is already ruled out in §9.4 for the same
  reason one of these attempts failed) — not a live blocker for this thread.
- `docs/test-reports/workflow-nl-query-generation2-report.md` (DEF-01) — the live QA finding behind
  K-057; K-057's own BACKLOG entry has the full reasoning and candidate angles.
- `docs/BACKLOG.md` K-057/K-058 — both items carry their own "Owner"/"Test strategy" fields; treat
  those as the brief, not a re-derivation.

**Sequencing.** Two independent tracks, dispatched in parallel (different files/code paths; both
use the same already-loaded `ministral-3-3b`, so no LM-Studio JIT-reload thrash risk from running
together — the documented hazard is specifically *concurrent requests to different models*):
- **Track A (K-057):** `data-scientist` diagnoses first — root cause is not yet conclusively
  determined (orchestration-layer suspected, not confirmed). Its output determines whether Track A
  needs a follow-up implementation unit (e.g. a `systemPrompt` steering change) or closes as
  "diagnosed, accepted low-rate risk, no fix warranted."
- **Track B (K-058):** the candidate fix is already fully specified (§9.4) — goes straight to
  `tdd-engineer` for TDD implementation + mutation testing, no separate diagnosis unit needed.

Both tracks close with `analyst` review of any code diff and a combined live regression pass
(`qa-engineer` or a `data-scientist` eval, sized to the unit) before this coordination closes.

**Operational note (2026-08-30):** U3 (`proof_defs.py`, live) and U4 (`analyst` reviewing U2,
which ran a destructive default `pytest` restore cycle) were dispatched in parallel on the
assumption that disjoint files meant no collision risk — wrong: both touch the same shared
`falkordb-dev` `reference` graph via `seed_salesperson.sh`'s hardcoded version default, and U4's
offline-suite verification briefly republished a mislabeled `salesperson@v4` containing U3's
uncommitted `v5` content. U4 caught, root-caused, and fixed it (full account in
`docs/reviews/salesperson-tool-reliability-impl2.md`'s Appendix); U3 was flagged mid-run to
re-verify before finalizing. No further unit in this coordination is dispatched in parallel with
another unit that runs a destructive offline-suite/reseed cycle against shared `reference` —
verified sequentially from here on, even across disjoint files.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 (Track A) | `data-scientist` (fresh) | `a6686c8ef000c2ec1` | delivered | K-057 root-cause diagnosis + rate estimate → `docs/reviews/salesperson-tool-reliability-ml.md` §11 | self → accepted (backlog premise falsified; dominant defect is a 50% boundary-rounding error, not orchestration; recommends `systemPrompt` wording fix, follow-up unit needed) | 201k tok / 72 tools |
| U2 (Track B) | `tdd-engineer` (fresh) | `a057dc6683287232c` | delivered | K-058 fix: dispatch-time text-presence guard on write-mutating tool calls, TDD + mutation-tested + live regression check (`executor.py`, `test_executor_agent.py`) | `analyst` → pending (U4) | 227k tok / 114 tools |
| U3 (Track A follow-up) | `coder` | `a6bc71a78fc5750b3` | delivered | K-057 fix: `filter_products` inclusive-bound wording + `systemPrompt` non-revision guidance, `v4`→`v5` (`proof_defs.py`, `tools.py`, +`nlq-40`); 2nd iteration tried, reverted, filed as K-060 | `analyst` → pending (U5) | 302k tok / 75 tools |
| U4 (gate, U2) | `analyst` (fresh) | `a1e425dd10e023f9f` | delivered | Diff review of U2's `executor.py` K-058 guard → `docs/reviews/salesperson-tool-reliability-impl2.md` | self → **approve** (2 MINOR, 1 residual-risk note — none blocking; filed K-059 for the `place_order` gap) | 179k tok / 67 tools |
| U5 (gate, U3) | `analyst` (fresh) | `a04a9e3bd8fd3c142` | delivered | Diff review of U3's `proof_defs.py`/`tools.py` K-057 wording fix → `docs/reviews/salesperson-tool-reliability-impl3.md` | self → **approve** (1 MINOR — untested `minPrice` symmetric guidance, folded into K-060's test strategy) | 107k tok / 36 tools |
| U6 (close) | `qa-engineer` (fresh) | `a5ea21105cf3c30a6` | in-flight | Combined live regression pass over both K-057 (`v5`) + K-058 (write guard) together, alongside the rest of the M6 tool surface → `docs/test-reports/salesperson-tool-reliability-regression.md` | self → — | — |
