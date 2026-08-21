# K-027 item 4 — golden-set expansion (D11): coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-027 (item 4)

**2026-08-21: coordination complete, K-027 epic closed.** Both gates passed (plan re-gate: approve;
implementation re-gate: approve). `tdd-engineer`'s delivery independently re-verified by `teco` at
every step (before/after test counts, mutation checks, doc claims, live-run numbers). Archiving this
log now that its work is done — see the ledger below for the full unit history.

## Goal

Close the last open scope item of K-027 (`docs/BACKLOG.md`). The method-design work is done:
`docs/plans/golden-set-expansion-ml.md` (`data-scientist`, `active`, not yet gated/finalized)
covers sizing, stratification, and sourcing for expanding `server/tests/eval/golden_guards.jsonl`.
Its §10 named two open decisions. Both are now **resolved by the user**:

1. **Boundary-tier independent second-labeler requirement: DROPPED.** No second labeler is
   available. The `boundary` tier will NOT get independent-labeler validation — it stays
   single-labeler/LLM-drafted like `clear_advance`/`clear_suspend` (§5(c) option (a), applied
   uniformly across all three strata). This is an explicit descope from the backlog item's stated
   intent (independence) — must be recorded as such, not silently.
2. **`clear_suspend` stratum size: n=40`** (the plan's own §3.1 lead recommendation — zero-tolerance
   FAR screen, clears the ≤10% Wilson bound with margin at zero observed failures). Not n=53.

## Units

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U1 | `data-scientist` | `aecdcb840b1e96deb` | delivered | `docs/plans/golden-set-expansion-ml.md` (finalized, Version 2) | `analyst` → — |
| U2 | `analyst` | `a991a173f2a925ffa` | gated | `docs/reviews/golden-set-expansion.md` (plan gate) | `analyst` → **needs changes** (2 major, 2 minor) |
| U1b | `data-scientist` | `aecdcb840b1e96deb` | delivered | `docs/plans/golden-set-expansion-ml.md` Version 3 — fixed all 4 findings (§7 step 5 contradiction, `r1_probe` misapplication on `cs-10`/`cs-13`, line-cite off-by-one, Wilson n=53/x=0 stale figure) | `analyst` → — |
| U2b | `analyst` | `aabb2d9bd0842008b` | accepted | `docs/reviews/golden-set-expansion.md` — dated `## Pass 2` section, diff-scoped v2→v3 re-gate | `analyst` → **approve** (independently re-verified by teco: header verdict line + full Pass 2 section read directly, all 4 findings confirmed fixed against real files) |
| U3 | `tdd-engineer` | `a2571412651ff5e6d` | delivered | fixture (26→85 rows, 30/40/15 composition), 5 literal-constant edits, new `test_guard_set_integrity.py`, live calibration run + report, BACKLOG/HISTORY updates | `analyst` → — |
| U4 | `analyst` | `ad2cd062b3d26a7de` | accepted | `docs/reviews/golden-set-expansion.md` — dated `## Pass 3` (implementation re-gate) | `analyst` → **approve** (independently re-verified by teco: header verdict + full Pass 3 section read directly, all 7 checks confirmed, 2 minor non-blocking follow-ups noted, no blockers) |

**2026-08-21: both gates passed. K-027 epic closed.** Pass 3's two minor, explicitly non-blocking
follow-ups (tighten `BACKLOG.md`'s G1 clause to spell out "cases" and add an "at the gate boundary"
caveat; extend the new offline integrity test to also check the `r1_probe`-restricted-to-
`clear_advance` constraint) are recorded here as backlog-worthy polish, not dispatched as further
units — Pass 3's own verdict states plainly neither blocks the epic-closure claim, and re-litigating
cosmetic text/coverage-completeness beyond a gate's own "approve, with suggestions" call would be
ceremony past the point of value. Proceeding to: (a) archive the three process documents this
closure retires (per root `AGENTS.md`'s "freeze at milestone close" convention — each document's own
owner performs its flip, per the by-kind routing table), (b) integration commit.

**2026-08-21 teco independent verification of U3 (before dispatching U4):** re-ran
`pytest tests/eval/test_guard_calibration.py tests/eval/test_golden_set_integrity.py
tests/eval/test_guard_set_integrity.py` myself → **499 passed** (matches claim). Re-derived fixture
composition from the file directly (not the plan's claim): 85 unique ids, `clear_advance` 30
(18/12), `clear_suspend` 40 (24/16), `boundary` 15 (9/6) — exact match. Read the diffs on
`test_guard_calibration_live.py`/`BACKLOG.md`/`HISTORY.md` directly — all five literal-constant
edits and the doc claims check out verbatim. Ran my own independent mutation check (flipped
`bd-05`'s `expected` to `true`, confirmed `test_boundary_rows_are_always_expected_false[bd-05]`
fails with the expected message, restored via diff-verified backup — clean). `verify_workflows.sh
acme` confirms `reference`/`ws:acme` back in sync after the implementer's full-suite run. **Note for
the record (not a blocker):** the live calibration report's G1 false-advance rate landed at exactly
10.0% (12/120 calls) — right at the ≤10% gate boundary, not comfortably under it. Verdict "wire" per
the harness's own non-blocking report logic; flagging for user visibility, not overriding.

**2026-08-21: plan gate closed, approved unconditionally.** Baseline offline suite (pre-implementation,
this session, `server/` venv): `pytest tests/eval/test_guard_calibration.py tests/eval/test_golden_set_integrity.py`
→ **141 passed**. `golden_guards.jsonl` currently 26 rows. Proceeding to U3 (`tdd-engineer`).

**2026-08-21 resume note (fresh `teco` session):** the stalled prior session's ledger was corrected
here — U1b was actually delivered (Version 3 is on disk, uncommitted, and independently re-verified
this session to address all 4 of U2's findings) but the stale ledger still showed it "in-flight".
Re-verification method: read both `docs/plans/golden-set-expansion-ml.md` (V3, in full) and
`docs/reviews/golden-set-expansion.md` directly and cross-checked each of the 4 findings against
V3's text — not taken on the data-scientist's own "fixed" claim. Proceeding to U2b (analyst
diff-scoped re-gate) next, per this repo's revise-in-place convention for an `active` review
document (root `AGENTS.md`: "a review gets a dated `## Pass N` section" rather than a new
`-impl.md` file, since the existing review was never gated/superseded/archived — it's still
`active` and this is a normal re-gate cycle within the same review).

Strict sequential pipeline (each unit depends on the prior). No parallelism available.

## Notes

- No graph/DDL surface — this is a file-based eval fixture + pytest harness item. `graph-dba` not
  needed (confirmed at U1/U3 dispatch, reconfirmed this session).
- K-027 epic closure: items 1, 2, 3, 5 are already ✅ delivered. If U3+U4 land clean, item 4 closes
  the epic's last open bullet — U3's brief includes flipping the K-027 header in `docs/BACKLOG.md`
  accordingly, verified by `teco` before acceptance.
