# GraphRAG Eval Harness — Test Plan (K-026)

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-026 (M2.5-quality)

## Scope & objective

Acceptance/QA pass on the K-026 GraphRAG retrieval + generation evaluation harness itself
(`server/tests/eval/`) — verifying the harness's own behavior end-to-end, not GraphRAG's retrieval
quality as a product feature. The harness has already been through four `analyst` plan-gate passes,
two `analyst` code/content gates (Unit 2b, Unit 3), one `analyst` corpus content review, one
`analyst` golden-set content review, and two `data-scientist` methodology sign-offs (D1
self-preference-bias framing, D6 baseline sign-off) — all closed **Approve** or **Approve with
suggestions**, zero open blockers. This is deliberately a **light-ceremony** pass: the goal is to
execute what static review couldn't (real suite runs, a fresh live run, a fresh generated report)
and confirm the backlog's done-condition holds against actual behavior, not to re-litigate design
decisions already gated.

## References

- `docs/BACKLOG.md` — K-026 entry (scope, done-condition).
- `docs/plans/graphrag-eval-ml.md` v2 — method note (design authority, metrics/thresholds).
- `docs/plans/graphrag-eval.md` v4 — implementation plan (D1–D7, Units 1–3).
- `docs/reviews/graphrag-eval.md` — all `analyst` gates (plan Pass 1–4, corpus review, golden-set
  review, Unit 2b review + re-gate, Unit 3 review).
- `docs/reviews/graphrag-eval-ml.md` — `data-scientist` D1 + D6 sign-offs.
- `docs/plans/graphrag-eval-coordination.md` — build ledger.
- `server/tests/eval/*` — code under test.

## Risk assessment

Given the depth of prior static review, residual risk is concentrated in things static review
*cannot* see:
1. **Does the default suite actually run green, right now, on this checkout** (not just "as of the
   last recorded count")? Medium likelihood of drift, low-medium impact.
2. **Does the live suite (judge layer) actually work against a real LM Studio** — the only prior
   live run was from an earlier session (2026-08-15's report); nobody has re-run it live this
   session. Medium impact if broken (silent judge failure would undermine the harness's stated
   purpose).
3. **Does `generate_report.py` actually produce a correct fresh report**, including the mandatory
   same-model caveat, when run today (not read from the old artifact)? High impact if wrong — this
   is the harness's one human-facing output.
4. **Does the backlog done-condition literally hold** (baseline recorded, judge-human agreement
   reported, self-retrieval guard enforced, both suites green, re-runnable)?
5. Already-accepted, non-blocking findings (root-conftest write-mode-query pattern,
   `generate_report.py`'s missing unit-test coverage, same-model judge limitation, small-N
   calibration) are **not** re-tested for their own sake — only touched incidentally if execution
   passes through them.

**Explicitly out of scope:** re-deriving the ML methodology (data-scientist's remit, already
signed off); re-reviewing code style/structure (analyst's remit, already gated); evaluating whether
GraphRAG retrieval quality is *good* as a product matter (that's what the harness measures, not
what this pass judges); re-seeding `ws:eval` (destructive/mutating risk to a persistent shared
workspace, not needed to prove re-runnability — two deterministic pytest runs against the existing
corpus is sufficient evidence per D2).

## Test items

| ID | Title | Preconditions | Steps | Expected result | Priority | Type |
|---|---|---|---|---|---|---|
| TP-001 | Repo-wide default suite is green | FalkorDB up | `cd server && .venv/bin/python -m pytest -q` | All tests pass, live-marked tests deselected, no errors | High | Functional |
| TP-002 | Golden-set integrity suite is genuinely network/DB-free | none | `pytest tests/eval/test_golden_set_integrity.py -q -s` before touching `ws:eval`; inspect for any DB call | Passes with zero FalkorDB/network calls | High | Functional |
| TP-003 | Retrieval eval reproduces the committed baseline | `ws:eval` seeded (already true) | `pytest tests/eval/test_retrieval_eval.py tests/eval/test_metrics.py -q -s`, capture printed recall@10/recall@5/MRR | Matches `retrieval_baseline.json` (recall@10=0.9737, recall@5=0.8947, MRR=0.6259, n=38) exactly; regression gate passes | High | Regression/Contract |
| TP-004 | Regression-gate branching (`check_regression`) behaves correctly | none | Re-inspect `test_metrics.py`'s fabricated-dict tests; run them | recall@10 regression fires, MRR-within-tolerance passes, MRR-beyond-tolerance fires, equal-to-baseline passes, both-regress reports 2 reasons | Medium | Functional (confirms Unit 2b M-1 fix still holds) |
| TP-005 | Live judge suite runs against real LM Studio | LM Studio reachable at `localhost:1234` | `pytest -m live -s` (full live marker, includes `test_judge_live.py` and any other live tests) | Completes without error; `judge_calibration.json` regenerated with fresh timestamp; `ws:eval` message count unchanged before/after (D2 read-only invariant) | High | E2E/Live |
| TP-006 | Fresh report generation | TP-003/TP-005 have run (fresh `retrieval_baseline.json`-consistent + fresh `judge_calibration.json`) | `cd server && .venv/bin/python tests/eval/generate_report.py` | Exits 0; writes `docs/test-reports/graphrag-eval-<today>.md`; report renders without exception | High | E2E |
| TP-007 | Mandatory same-model caveat present verbatim in the fresh report | TP-006 done, `sameModelAsAgentUnderTest: true` | Read the freshly generated report | Caveat block present, adjacent to judge numbers (not footnote), distinguishes calibration vs. generation sub-pass, includes "does not license trusting" sentence and sign-off placeholder | High | Contract |
| TP-008 | Self-retrieval guard is mechanically enforced | none | Inspect `test_golden_set_integrity.py`'s substring-guard test result within TP-002's run; confirm it actually executes (not vacuous) | Guard test present and passing against all 38 pairs' `target_text` | Medium | Functional |
| TP-009 | Backlog done-condition cross-check | TP-001–TP-008 done | Read `docs/BACKLOG.md` K-026 done-condition line by line against actual artifacts/results | Every clause verifiably holds: baseline recorded ✓/✗, harness re-runnable ✓/✗, judge–human agreement reported ✓/✗, no-verbatim-self-retrieval asserted ✓/✗, both suites green ✓/✗ | High | Acceptance |
| TP-010 | Harness re-runnability (non-destructive) | TP-001, TP-003 already run once | Re-run `pytest tests/eval -q` a second time | Identical results (deterministic), no side effects, no `ws:eval` mutation | Medium | Regression |
| TP-011 | Exploratory: anything the static gates couldn't see | all above done | Skim for behavior-level surprises during actual execution (timing, stray warnings, unexpected skips, environment-only failures) not visible from reading code alone | Any new, previously-undocumented defect reported with evidence | Medium | Exploratory |

## Environment & data setup

- FalkorDB: `falkordb-dev` container, already running (confirmed via `docker ps`).
- LM Studio: reachable at `localhost:1234`, serving `qwen/qwen3-4b-2507` (agent-under-test and,
  per D1, judge) plus several other models — confirmed via `curl localhost:1234/v1/models`.
- `ws:eval`: already seeded (121 messages / 12 threads, per `corpus_provenance.json`) — **not**
  re-seeded by this pass (D2: persistent workspace, mutating it is out of scope and unnecessary).
- No destructive operations planned. `pytest -m live` performs read-only `hybrid_search` calls
  against `ws:eval` (per D2/Unit 3's own asserted invariant) and writes only to
  `server/tests/eval/judge_calibration.json` (a harness artifact, not shared graph state) and a new
  dated file under `docs/test-reports/`.

## Entry/exit criteria

**Entry:** FalkorDB reachable; prior review gates all closed (confirmed above — no open blockers).

**Exit:** all High-priority items pass or have a clearly reported defect/blocker; Medium items
executed and reported; verdict rendered as pass / pass-with-issues / fail against K-026's literal
done-condition.

## Out of scope

- Re-running/re-seeding `ws:eval` from scratch.
- Re-reviewing ML methodology or code style (already gated by `data-scientist`/`analyst`).
- Fixing any defect found (report only, per role).
- Re-litigating the already-accepted findings enumerated in the task brief (root-conftest
  write-mode-query pattern, `generate_report.py`'s own missing unit tests, D1's same-model
  limitation, small-N calibration caveat).
