# Salesperson tool-orchestration reliability — round 4 (K-060) — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-060 (post-M6, not a milestone gate)

Successor to `docs/plans/salesperson-tool-reliability3-coordination.md` (archived 2026-08-31,
K-059 closed) — ordinal-bumped per root `AGENTS.md`'s collision rule 5 (same kind/topic/role,
`-coordination`, round 3 already executed against). Round 2 (K-061) stays `active`, untouched —
its own live-regression confirmation is a separate, still-open item, not part of this round.

Picking up **K-060** (`docs/BACKLOG.md`), the last open follow-up from this investigation thread
besides K-062 (still filed as opportunistic-pickup-only, not worth its own round). K-060 is the
oldest open item in the thread (disclosed 2026-08-30 during K-057's own fix verification) and the
only one with **two already-failed wording attempts** behind it — per the thread's own repeated
lesson (K-057's reverted second iteration, K-060's own two reverted attempts), a fix shape guessed
ahead of a proper root-cause keeps not holding, so this round is diagnosis-only by design, same as
every prior round's first unit.

**Prior art (read before dispatching or picking this up cold):**
- `docs/BACKLOG.md` K-060 — the filed item: why it exists (found during K-057 fix verification,
  `filter_products` called with no `category` silently drops a genuine match from the synthesized
  reply), the two failed wording attempts and their own measured rates, the `Owner` note
  (`data-scientist`, check whether the tool's own return payload is a more reliable lever than
  wording), and the test-strategy note (isolate no-`category` calls, larger n, plus fold in the
  `analyst`-flagged `minPrice`/"more than $X" regression gap from K-057's own review).
- `docs/reviews/salesperson-tool-reliability-ml.md` §11 — the K-057 diagnosis that *disclosed*
  K-060 as a side finding; same harness precedent this round should reuse (§11.2: real
  `ModelGateway.from_env()`, not `StaticModelGateway` — load-bearing, not stylistic, since `v4`'s
  `query_graph_data` resolves its own internal call through the shared `step`-role default,
  independent of the `assistant` step's own model pin; a `LoggingToolRegistry` ground-truth wrapper
  since the executor's own trace payloads truncate tool results at 200 chars).
- `docs/reviews/salesperson-tool-reliability-ml.md` §9.4/§8.4/§4.2 — this thread's own standing
  discipline: never ship or reattempt an unverified mitigation; root-cause and rate-estimate first.
- `server/falkorchat/tools.py:437-500` (`FilterProductsTool` — schema, the K-052
  `DEFAULT_FILTER_LIMIT=20` demo-scale cap, abstention shape) and `server/falkorchat/
  repository.py:2724-2769` (`repository.filter_products` — confirms no missing-capability angle,
  same as §11.1 already ruled out for the sibling K-057 defect).
- `docs/reviews/salesperson-tool-reliability-impl3.md` — the `analyst` finding to fold in: the
  shipped K-057 fix's `minPrice` inclusive-bound guidance was added by analogy but never itself
  live-regression-tested in the "more than $X" direction.

**Scope discipline (carried from rounds 1-3):** diagnosis first, larger n, before any fix
attempt. K-060 specifically already has **two** failed wording-only attempts behind it (BACKLOG's
own note) — this round's unit is diagnosis + rate estimate + mechanism finding, explicitly **not**
a third wording guess. Whether a fix is warranted at all, and if so whether it's a payload
restructuring (the `Owner` note's own suggested alternative lever) or something else, is a decision
for *after* this unit lands, not before.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `data-scientist` | `a501e8d44383850f2` | delivered | `docs/reviews/salesperson-tool-reliability-ml.md` §14 | — → — | 193.6k tok / 64 tools |

## Notes

- Single-unit start, same shape as every prior round's own first unit — `teco` verifies the
  diagnosis directly (re-checkable stats/ground-truth, no code change) rather than dispatching a
  separate `analyst` gate for a diagnosis-only deliverable. A fix unit, if warranted, follows
  normal implementer + `analyst` review gating, dispatched as its own follow-up unit once this
  lands and is reviewed.
- No parallel dispatch risk (single unit; round 2's coordination has no unit currently in flight).
- **U1 delivered and independently re-verified by `teco` 2026-08-31.** All 6 distinct Wilson 95%
  CIs recomputed from scratch and matched exactly (29/30, 1/30, 1/10, 5/50, 15/15×2, 11/11). Every
  condition's stated ground truth (A/B/C/D's item lists, prices, categories) independently
  re-queried against the live `reference` catalog and matched exactly, including category counts.
  Code citations verified: `tools.py:467-474`'s `minPrice`/`maxPrice` inclusive-bound wording;
  `proof_defs.py`'s `SALESPERSON_DEF["transitions"]` confirmed `kind: "cmp"` only (never `"fuzzy"`),
  supporting the `guard_judge=None` harness simplification; `config/models.json` confirmed to pin
  `temperature: 0` only for `qwen/qwen3-4b-2507`, with no entry at all for
  `mistralai/ministral-3-3b` (the pinned `assistant`-step model) — the delegate's own explanation
  for the two samples' non-identical point estimates. `ws:ds-k060` confirmed torn down (absent from
  the live graph list, probed directly); `reference`/`ws:acme` confirmed present and undisturbed,
  no stray graph key this time. Diff confirmed scoped to exactly this section (265 insertions, one
  file, `docs/BACKLOG.md` untouched by the delegate as instructed).
  **Verdict: diagnosis accepted as delivered.** Findings folded into `docs/BACKLOG.md` K-060
  directly by `teco` (rewritten in place, still 🟡 in-progress — not resolved, so **not** removed
  from `BACKLOG.md`; the pooled rate, the self-correction finding, the payload-lever assessment and
  its cheaper §14.6 follow-up design, and the now-closed `minPrice` regression gap were all folded
  in). No fix is warranted on current evidence (§14.7) — a third wording guess is explicitly
  rejected, and a live payload A/B is explicitly assessed as underpowered through the full
  conversation harness. This round's own sole unit (diagnosis) is complete; no fix unit follows
  from it. Two secondary findings outside this pass's own scored question were surfaced and
  intentionally left as notes, not new backlog items, per the delegate's own §14.7 point 6: a
  tool-selection phrasing sensitivity (`filter_products` vs `query_graph_data`) and a narrower
  reframing of §11.3's `query_graph_data` weakness (compound-predicate-specific, not price-only) —
  both worth a mention to whoever next touches `nlq_golden_set.jsonl` or tool-description wording,
  not standalone follow-ups on this evidence alone.
