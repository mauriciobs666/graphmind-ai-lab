# `mention-reply-delivery` CI blind-spot (K-039 item 3) — coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-039 (M3.5)

## Goal

K-039 item 1 (immediate `post_message` fallback mitigation) shipped 2026-07-31. Item 3, the
**CI blind-spot follow-up**, is still open: `docs/reviews/mention-reply-delivery-rca.md` §5 item 2
notes that `pytest -m live`'s AC-4 answer-post assertion is the only test that would have caught
this class of failure, and it's excluded from the default `pytest -q` loop
(`server/pyproject.toml` `addopts = -m "not live"`) — so a green default suite gives false
confidence that the demo's `@mention` path actually posts replies. RCA §5 item 2 offers two
non-exclusive directions: (a) promote a reachability-gated subset of the live AC-4 assertion into
the default loop, or (b) surface a "last N triage runs: posted / did not" signal on the existing
`GET /workspaces/{ws}/readiness` route / web ready-to-demo banner (K-036).

Full context: `docs/reviews/mention-reply-delivery-rca.md` (§5 item 2 + causal chain contributing
factor 2), `docs/BACKLOG.md` K-039 (item 3, ~line 1167), `docs/HISTORY.md` 2026-07-31 "K-039
immediate mitigation" entry ("Left alone, by design" section explicitly defers this).

## Units

1. **architect** — design plan, `docs/plans/mention-reply-delivery.md`. — **done.** Recommendation:
   decline promoting a live test into the default loop (item 1's fix is already covered by
   deterministic offline tests); instead add a "last-20 triage runs post-success" signal to
   `check_demo_readiness`/the K-036 readiness banner (informational, not folded into `ready`), plus
   a one-time re-run of `pytest -m live` to confirm AC-4 now passes and correct the stale
   known-RED note at `docs/BACKLOG.md` K-027. Flags one open product-scope question (§6): should a
   degraded post-success rate ever flip `ready`? — left informational-only by design, noted for the
   final report rather than decided unilaterally.
2. **analyst** — plan gate review, `docs/reviews/mention-reply-delivery.md`. — **done.**
   Verdict: approve with suggestions (4 minor · 1 nit, no blockers). Architect folded all 5
   findings into the plan in place (now v2) — see plan §7 "Review dispositions".
3. **implementer(s)**, run sequentially — **all done**:
   - `graph-dba`: QUERIES.md §12.15 authored + PROFILEd (compound index scan on `startedAt`+
     `status`, new planner fact promoted to `claude/graph-dba/falkordb-quirks.md`);
     `test_queries.sh` 276/276 → 282/282. **Side note:** running the suite wiped the shared
     `reference` graph (documented pre-existing hazard) — `teco` restored it via
     `./scripts/seed_workflows.sh acme` (idempotent, create-only) and re-verified
     `verify_workflows.sh acme` → in sync.
   - `coder`: `Repository.read_recent_post_success` + `POST_SUCCESS_SAMPLE_SIZE` +
     `check_demo_readiness`'s new `postSuccess` field. `pytest -q` 647 → 658 passed (11 new
     tests), mutation-tested (3 deliberate breaks, all caught). Widened `test_api.py`'s
     `_READINESS_KEYS` per the analyst's plan-gate finding.
   - `frontend-engineer`: `renderPostSuccess` in `web/app.js` + `.post-success`/
     `.post-success--degraded` CSS in `web/index.html`. Verified live against `ws:acme`'s real
     degraded case + synthetic ok/no-data via a `vm`-sandboxed run of the real `app.js`. Badge
     color confirmed still driven by `ready` alone.
4. **analyst** — diff-scoped re-gate on the implemented change (`docs/reviews/mention-reply-delivery.md`
   "Pass 2"). — **done. Verdict: approve, no findings.** `pytest -q` 658/658 (via `-m "not live"`
   default), `test_queries.sh` 282/282, independently reproduced all 3 mutation-test claims,
   confirmed scope discipline. Re-wiped/re-seeded `reference` again (own `test_queries.sh` run) and
   confirmed `ws:acme` back in sync before finishing.
5. **qa-engineer** — plan step 4: acceptance pass on the new banner/route + the one-time
   `pytest -m live` re-run and BACKLOG K-027 note correction. — **done. No defects.** Live-verified
   `postSuccess: degraded` on `ws:acme` (screenshot + DOM check via headless Chrome CDP), confirmed
   badge color unaffected by `postSuccess`. Re-ran `pytest -m live` once (LM Studio reachable) —
   AC-4 now **passes** (`1 passed, 658 deselected`), confirming item 1's fix resolved the exact
   failure mode. Corrected the stale K-027 "known-RED" note in `docs/BACKLOG.md`.
6. **doc close-out** — **done.** `docs/BACKLOG.md` K-039 marked "✅ delivered — items 1 & 3 ✅ both
   delivered 2026-07-31" (one factual correction made by `teco`: the drafted `HISTORY.md` entry
   invented a nonexistent "rate ≥ 70%" threshold for `postSuccess.status` — corrected to the actual
   `services.py` logic, `"ok"` iff every sampled run posted). `docs/HISTORY.md` entry added.
   `Status:` flips complete: plan → `architect` (archived), review → `analyst` (archived), this
   coordination doc → `teco` (archived).

## Outcome

K-039 is now fully delivered (items 1 + 3; item 2's full K-027 engine contract is separately
tracked, never in K-039's own scope). The shared K-039 done-condition — `pytest -m live`'s AC-4
assertion flipping from documented RED to green — is met. One product-scope question was
deliberately left open rather than decided unilaterally: architect's plan §6 asks whether a
persistently `"degraded"` post-success rate should someday flip the readiness route's `ready`
boolean; today it stays purely informational. Flagging this for the user/product owner, not
re-opening it here.

## Notes

- Standing process lessons applied: two analyst gates (plan + diff, both independently verified
  live rather than trusting the plan/implementers' claims), mutation-test implementers' own new
  tests (verified by both the implementer and, independently, the diff-gate reviewer), serialize
  any unit touching the same file (units ran strictly sequentially: graph-dba → coder →
  frontend-engineer), never mutate the user's git tree.
- Operational side-effect encountered twice: `./scripts/test_queries.sh` wipes the shared
  `reference` graph as documented (`falkor-chat/AGENTS.md`) — both `graph-dba`'s and the
  diff-gate `analyst`'s runs triggered this; both times `./scripts/seed_workflows.sh acme` restored
  it and `./scripts/verify_workflows.sh acme` confirmed in-sync before moving on. No lasting
  impact on `ws:acme`.
- Scope fence: `falkor-chat/` only — held throughout (confirmed by both review gates).
