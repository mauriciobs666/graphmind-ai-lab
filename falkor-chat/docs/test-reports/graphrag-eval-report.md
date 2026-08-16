# GraphRAG Eval Harness — Test Report (K-026)

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-026 (M2.5-quality)

## Summary

Acceptance/QA pass on the K-026 GraphRAG retrieval + generation evaluation harness
(`server/tests/eval/`), executed 2026-08-16 against `main` (working tree clean aside from
unrelated, explicitly out-of-scope in-flight changes under `claude/*`/`docs/plans/cpg-agent-adoption*`
— untouched by this pass). Test plan: `docs/test-plans/graphrag-eval.md`.

The harness had already been through four `analyst` plan-gate passes, an `analyst` corpus content
review, an `analyst` golden-set content review, two `analyst` code/content gates (Unit 2b, Unit 3,
including a fix-and-regate cycle), and two `data-scientist` methodology sign-offs (D1 same-model
framing, D6 baseline) — all closed Approve or Approve with suggestions. This pass adds what static
review could not: real execution — the full default suite, a fresh live run against a real LM
Studio, and a freshly generated report — verified against actual, observed output.

**Overall verdict: PASS.** All eleven test-plan items pass. K-026's backlog done-condition holds
against real execution, with real numbers, not paraphrase. No new defects found. One already-known,
already-accepted non-blocking finding was incidentally re-touched (root `conftest.py`'s write-mode
probe, noted below per the task brief, not re-flagged as new) and confirmed not to be exercised
destructively during this run.

## Results table

| ID | Title | Result | Evidence |
|---|---|---|---|
| TP-001 | Repo-wide default suite green | **PASS** | `pytest -q` → `1034 passed, 2 deselected in 10.20s` |
| TP-002 | Golden-set integrity suite genuinely network/DB-free | **PASS** | `pytest tests/eval/test_golden_set_integrity.py -q -s` → `117 passed in 0.38s` (sub-second, no FalkorDB call in the file) |
| TP-003 | Retrieval eval reproduces committed baseline | **PASS** | `pytest tests/eval/test_retrieval_eval.py tests/eval/test_metrics.py -q -s` → printed `recall@10=0.9737 recall@5=0.8947 mrr=0.6259` (n=38); `retrieval_baseline.json` = `{"recall_at_10": 0.9736842105263158, "recall_at_5": 0.8947368421052632, "mrr": 0.6258771929824561, "n": 38}` — exact match |
| TP-004 | `check_regression` branching correct | **PASS** | `pytest tests/eval/test_metrics.py -k regression -v` → `6 passed` (recall@10-regress fires, MRR-within-tolerance passes, MRR-beyond-tolerance fires, equal-to-baseline passes, both-regress reports 2 reasons — all present) |
| TP-005 | Live judge suite runs against real LM Studio | **PASS** | `pytest -m live -s -v` → `2 passed, 1034 deselected in 175.92s (0:02:55)`; `judge_calibration.json` regenerated (`generatedAt: 2026-08-16T15:08:44Z`, `sameModelAsAgentUnderTest: true`); `ws:eval` message count `121` before **and** after (both the test's own assertion and my independent `redis-cli GRAPH.QUERY "ws:eval" "MATCH (m:Message) RETURN count(m)"` → `121`) |
| TP-006 | Fresh report generation | **PASS** | `python tests/eval/generate_report.py` → exit 0, `wrote /home/mauricio/prg/graphmind-ai-lab/falkor-chat/docs/test-reports/graphrag-eval-2026-08-16.md` |
| TP-007 | Mandatory same-model caveat present verbatim in fresh report | **PASS** | `docs/test-reports/graphrag-eval-2026-08-16.md:38-42` — caveat block present, adjacent to the judge numbers (not a footnote), distinguishes calibration vs. generation sub-pass, includes "A passing calibration number does not license trusting these — they are two different validity claims" and the `data-scientist` sign-off placeholder, verbatim |
| TP-008 | Self-retrieval guard mechanically enforced | **PASS** | Report line 51: `**PASS** — no golden query is a verbatim substring of...`; underlying test included and passing in TP-002's 117 |
| TP-009 | Backlog done-condition cross-check | **PASS** | See breakdown below |
| TP-010 | Harness re-runnability (non-destructive) | **PASS** | `pytest tests/eval -q` (second run, after TP-005's live run) → `164 passed, 1 deselected in 0.57s`, identical shape; `git status` confirms no `ws:eval`/corpus files touched |
| TP-011 | Exploratory: behavior static review couldn't see | **PASS (no new defect)** | See "Exploratory findings" below |

### TP-009 breakdown — K-026 done-condition, literal text from `docs/BACKLOG.md`

> "baseline recall@10/recall@5/MRR recorded; harness re-runnable; judge–human agreement reported;
> golden set asserts no verbatim self-retrieval; both suites green."

| Clause | Status | Evidence |
|---|---|---|
| baseline recall@10/recall@5/MRR recorded | ✓ | `retrieval_baseline.json`: recall@10=97.4%, recall@5=89.5%, MRR=0.6259, n=38; `data-scientist`-signed-off (`docs/reviews/graphrag-eval-ml.md`) |
| harness re-runnable | ✓ | TP-003/TP-010: default suite, `tests/eval` alone, and the live suite all reproduced byte-identical/consistent numbers on a fresh, independent run this session |
| judge–human agreement reported | ✓ | Fresh live run: faithfulness agreement 90.0% (9/10), relevance agreement 70.0% (7/10) — identical per-item pattern to the 2026-08-15 run (same jc-08/jc-02/jc-04/jc-09 disagreements), reported in `judge_calibration.json` and the fresh generated report |
| golden set asserts no verbatim self-retrieval | ✓ | `test_golden_set_integrity.py`'s substring guard runs and passes against all 38 pairs' `target_text` (TP-002/TP-008) |
| both suites green | ✓ | default: 1034 passed/2 deselected; live: 2 passed/1034 deselected (0 failed in either) |

**All five clauses verifiably hold against real execution.**

## Defects

**None found.** No new defects surfaced during this pass.

## Exploratory findings (TP-011)

Executed rather than merely read, three things the static review gates flagged as *manually
verified but automated-test-free* (`docs/reviews/graphrag-eval.md`'s Unit 3 M-1) were independently
re-exercised, read-only, via a throwaway interpreter session (module-level path constants
monkeypatched in-process; no file on disk touched, confirmed via `git status` afterward):

1. `judge_calibration.json` absent → `_render_judge_section` renders the "Not run" marker. **Confirmed correct.**
2. `retrieval_baseline.json` absent → `build_report()` raises `ReportError` with a clear message. **Confirmed correct.**
3. `_self_retrieval_guard_failures` on a fabricated leaking row (`query="hello world"`,
   `target_text="say hello world now"`) → correctly flagged `["fake-leak"]`. **Confirmed correct.**

This corroborates (does not contradict) the already-accepted, already-known finding that
`generate_report.py` has no dedicated automated test file for these branches — all four behave
correctly today, as the analyst's manual review already found; this pass adds a second,
independent confirmation via actual execution rather than code-reading alone. Not a new defect —
consistent with the task brief's "already-accepted" list.

Two additional observations, informational only, not defects:

- **The live run's generation sub-pass is again a perfect 20/20 on both faithfulness and relevance**
  (identical to the 2026-08-15 run). This is the exact pattern the same-model self-preference-bias
  caveat exists to warn a reader against over-trusting (already flagged by the Unit 3 `analyst`
  review as "worth the `data-scientist` sign-off's attention," not re-flagged here as new).
- **The judge's per-item calibration verdicts are identical across the two live runs one day apart**
  (same jc-08 faithfulness disagreement, same jc-02/04/09 relevance disagreements) — the harness is
  behaviorally deterministic in this environment for the calibration sub-pass, which is reassuring
  for reproducibility but not itself evidence about judge *quality*.
- **Root `server/tests/conftest.py`'s `_falkordb_reachable()`** still has the identical write-mode
  `GRAPH.QUERY` pattern already fixed in the eval subtree's own `conftest.py` (Unit 2b's B-1 fix).
  I read this code as part of verifying TP-001's green baseline but did not exercise the specific
  vulnerable path (it only bites on a not-yet-existing `ws:test`, and `ws:test` already existed
  throughout this session). Per the task brief, this is already logged and not re-flagged as new —
  noted here only because I touched the code path during verification, per the instruction to note
  it "if you happen to touch/exercise this path."

## Coverage & gaps

**Covered by this pass:** full default suite execution; the eval-only subtree in isolation (twice,
confirming determinism); the live judge+generation suite end-to-end against real LM Studio; a fresh
report generation and its mandatory caveat text; all five literal clauses of the backlog
done-condition; three of `generate_report.py`'s branches that had no automated test (executed
directly, read-only).

**Not covered, deliberately (per the test plan's scope):**
- Re-seeding `ws:eval` from scratch (`./scripts/seed_eval_corpus.sh`) — not exercised this pass;
  D2 makes `ws:eval` a persistent workspace and re-seeding it was assessed as unnecessary,
  destructive-risk-bearing work to prove re-runnability (the existing corpus, run twice
  deterministically, is sufficient evidence).
- Re-deriving ML methodology validity (corpus representativeness, judge validity as a *product*
  question) — `data-scientist`'s remit, already signed off twice.
- Re-reviewing code style/structure — `analyst`'s remit, already gated four times over.
- Any of the already-accepted findings enumerated in the task brief (root-conftest pattern,
  `generate_report.py`'s missing dedicated test file, the same-model judge limitation itself, the
  small-N calibration caveat) — confirmed still present/still true where touched, not re-litigated
  as new.

**Residual risk, inherited and unchanged by this pass (not new):** corpus representativeness
(method note's own #1-ranked risk, `analyst`-reviewed not human-verified); the generation sub-pass's
self-preference-bias exposure (D1, accepted); the ~10-example calibration set's small-N statistical
weakness (D4, accepted, and the caveat text is confirmed present and correct); the `data-scientist`
sign-off on Unit 3's same-model numbers is still recorded as pending in the coordination ledger and
the fresh report's placeholder correctly still reads `[pending / not yet reviewed]`.

## Feedback & recommendations

1. **No blocking recommendation.** The harness works correctly end-to-end, on real execution, with
   real numbers — this pass found nothing that should hold K-026's delivery.
2. **Minor, non-blocking observation (new, informational):** the harness-generated report's own
   header block (`docs/test-reports/graphrag-eval-<date>.md`) hardcodes `**Owner:** \`qa-engineer\``
   in `generate_report.py`'s `build_report()` (`server/tests/eval/generate_report.py:289`), even
   though the file is produced by the harness code (Unit 3, `tdd-engineer`), not authored by
   `qa-engineer` at generation time. Per root `AGENTS.md`'s doc-lifecycle convention, `Owner:` is
   "the producing agent" — a literal reading would make this file's own header slightly
   inaccurate about its producer, though functionally harmless (this generated file's `Status:`
   flip is owned by `qa-engineer` regardless, per the `test-reports/*` row in `AGENTS.md`'s
   closed-flip table, so the current value happens to be the one that matters for lifecycle
   purposes). Not raised as a defect — purely a documentation-convention nit for whoever next
   touches `generate_report.py`, if ever.
3. **Reiterating, not discovering, the coordination ledger's own recommendation:** the `data-scientist`
   sign-off on Unit 3's same-model judge numbers is still open. This QA pass does not gate on it
   (per D1, Unit 3 is explicitly non-gating/descriptive), but it should land before anyone treats
   the 90%/70% agreement numbers or the 20/20 generation scores as more than directional.
4. **Reiterating the method note's own standing ask:** a real human spot-check of the ~10-example
   `golden_judge_calibration.jsonl` set (recommended by the `analyst`'s M-5, accepted-as-flagged by
   the coordinator, not yet done as far as this pass can tell) would strengthen the "judge–human
   agreement" claim's literal meaning. Not a blocker for this pass's PASS verdict — the
   done-condition asks that the number be *reported*, which it is, correctly, with its caveats
   intact.

## Deliverables

- Test plan: `/home/mauricio/prg/graphmind-ai-lab/falkor-chat/docs/test-plans/graphrag-eval.md`
- Test report (this document): `/home/mauricio/prg/graphmind-ai-lab/falkor-chat/docs/test-reports/graphrag-eval-report.md`
- Fresh harness output artifact produced during this pass (not a QA deliverable, but evidence):
  `/home/mauricio/prg/graphmind-ai-lab/falkor-chat/docs/test-reports/graphrag-eval-2026-08-16.md`
- Side effect of TP-005's live run (expected, a harness artifact, not shared graph state):
  `/home/mauricio/prg/graphmind-ai-lab/falkor-chat/server/tests/eval/judge_calibration.json`
  regenerated with a fresh timestamp and numbers; `ws:eval` itself was not mutated (verified).
