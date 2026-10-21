# Salesperson `systemPrompt` language salience — Mitigation D for K-065/DEF-6 — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-065 (gates the first live, audience-facing demo, not a milestone)

## Why this exists

`docs/BACKLOG.md` K-065: an `en`-configured storefront participant occasionally gets a fully-formed
Spanish reply under concurrency, root-caused (application code bypassed entirely) to LM Studio's own
serving layer — `docs/plans/salesperson-ui-ml.md` (repo root; this specific plan/review/test-plan
family for the salesperson-ui feature lives at the repo root per `AGENTS.md`'s `salesperson/`
Structure bullet, not under `falkor-chat/docs/`). That plan's **Recommendation** section names a
priority-ordered mitigation path — **D → C → B**, explicitly **not A** (full serialization) — and
requires a live re-test against QA's own TP-007 protocol before calling any increment closed.

**User-authorized scope for this coordination: mitigation (D) only** — strengthen
`SALESPERSON_DEF`'s `systemPrompt` language salience via a cheap `v8` version bump. C and B are
*not* in scope here; if D's live re-test doesn't clear the bar, that is a decision point back to the
stakeholder, not an automatic escalation to C.

**Read before picking up any unit below:** `docs/plans/salesperson-ui-ml.md` (repo root) §"Mitigation
options assessed" (item D) and §"Recommendation" in full — do not work from this summary alone.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `tdd-engineer` | `aa84d8a313ab78f7d` | accepted | `falkor-chat/server/falkorchat/proof_defs.py` (+ sweep) | `analyst` (U2) → approve with suggestions | 144k tok / 59 tools |
| U2 | `analyst` | `af2954b05a0768e1e` | accepted | `falkor-chat/docs/reviews/salesperson-language-salience.md` | `analyst` → approve with suggestions | 105k tok / 28 tools |
| U3 | `qa-engineer` | `adcdf1e1b0ba2c48e` | accepted | `falkor-chat/docs/test-reports/salesperson-language-salience-report.md` | `teco` (direct verification) → clean result, see below | 361.5k tok / 519 tools |
| U4 | `qa-engineer` | `a0411acf57dbe0b05` | gated | `salesperson/scripts/language_salience_probe.py` (committed harness + smoke run n=2) | `analyst` (U5) → approve with suggestions | 164k tok / 49 tools |
| U5 | `analyst` | `a954f171484317f46` | accepted | `falkor-chat/docs/reviews/salesperson-language-salience.md` (`## Pass 2`, harness review) | — → — | — |
| U6 | `qa-engineer` | `a0411acf57dbe0b05` (U4 resumed) | in-flight | `falkor-chat/docs/test-reports/salesperson-language-salience2-report.md` (dedicated-instance rerun, K-065 + K-066) | `teco` (direct verification) → — | — |
| U7 | `tdd-engineer` | — | queued | harness fixes per review Pass 2 (F1–F7, fake-storefront tests for `run_participant`) + v3 re-score of U6's stored JSONL | `analyst` → — | — |

**Mid-run relay to U3 (2026-09-18):** user surfaced LM Studio server-log lines timestamped
22:54:53–22:54:55 during/near U3's window — a terminated tool-call generation with a Spanish
partial output for `mistralai/ministral-3-3b`, a "Channel Error", and a failed model load for
`qwen/qwen3-4b-2507` ("Engine protocol startup was aborted"). Relayed to U3 by `SendMessage`
(not acted on directly — U3 holds the trial timestamps needed to tell whether this falls inside
its window, and to classify it correctly: an aborted tool-call producing no user-facing reply is
not the same failure shape as TP-007's completed-wrong-language metric). Asked U3 to check, classify
if relevant, and report it as a distinct engine-stability observation rather than silently folding
it into the language-adherence count — not asked to redo completed trials over this alone.

**U2 verdict: approve with suggestions, no blockers.** Verified independently by `teco`: the
three traps hold (spot-checked the diff directly), the mutation-test claim is internally
consistent with the pinned phrases in the new test, the doc-sweep re-check matches my own earlier
sweep. **Major finding actioned, not just noted**: three stale "not yet mitigated" claims
(`BACKLOG.md`, `AGENTS.md`, `salesperson/README.md`) rewritten to current true state — see
`falkor-chat/docs/HISTORY.md`'s 2026-09-18 K-065 entry for the full record. Both U1's implementation
and U2's review are now **committed** (`999141f`, `c420ab6`, `fe3cfe3` — implementation, review,
doc-accuracy catch-up), per this repo's code-implementation-chain convention (commit each verified
unit immediately, don't hold for the whole chain) and the K-056 precedent (implement/ship, then
live-verify, revert-if-negative as the fallback).

**Open question from U2, deliberately not actioned as a gate**: whether a `data-scientist`
methodology sanity-check on system-role prompt salience is worth doing before spending U3's live
budget. Judged not worth inserting as a blocking unit — U3's live re-test *is* the cheap,
fast-iterating validation loop this mitigation was designed around (local LLM, not per-token
API cost), and the empirical answer from U3 is what actually resolves the plan's own stated risk
either way. Noted for the user in case they'd rather have it first.

**U1 verified by teco, not just self-reported.** Independently re-ran: targeted test file (8/8
green), full offline suite (`2845 passed, 14 deselected` — matches the report's claimed count
exactly), read the full diff on `proof_defs.py`/`test_salesperson_scaffold.py`/all three scripts/
`falkor-chat/AGENTS.md`/`falkor-chat/README.md` line-by-line against the brief, and ran an
unfiltered repo-wide `grep` for stale `v7` mentions to check the "left alone, and rightly so" list
in the report. That sweep found one gap the subagent's own sweep missed:
**`salesperson/README.md:132`** (`FALKORCHAT_TRIGGER_DEF_VERSION=v7` in a manual bring-up example,
same code block as line 126 which *was* caught) — a genuinely trivial single-file, single-string
fix, made directly by `teco` rather than re-dispatching. All other `v7` hits confirmed correctly
left alone (dated `HISTORY.md` entries, `archived`/point-in-time planning docs narrating what was
true when written, and two arbitrary monkeypatch fixture values in `test_app.py`/
`test_storefront.py` unrelated to the shipped constant).

U1 → U2 → U3 are strictly sequential (each depends on the prior unit's output).

**U3 result: n=25 trials, zero wrong-language occurrences for `en`/`pt-BR`/`es`, every trial, every
reply** — turn-level 61/61 · 46/46 · 33/33, trial-level 0/23 (Wilson 95% CI [0.0%, 14.3%]), down
from the ~20% baseline point estimate. **Independently re-verified by `teco`, not accepted on the
subagent's word**: every Wilson interval in the report (turn-level ×3, trial-level ×3, both
baseline comparisons) recomputed by hand from raw x/n and matched to the reported figure; the
per-trial table's 75 numerator cells and 75 dead-turn cells cross-summed against the headline
61/46/33 replied and 235-no-reply/14-dead-but-replied/249-total-dead figures — all reconcile
exactly; `SALESPERSON_DEF.config.model` confirmed via direct grep to be
`lmstudio/mistralai/ministral-3-3b` only, matching the report's model-attribution claim; `fe3cfe3`
confirmed an ancestor of current `HEAD` (`56e2e02`) with zero commits touching `proof_defs.py` in
between; all three cited commit shas (`999141f`, `c420ab6`, `fe3cfe3`) and the DEF-3 fix
(`4cebd96`) confirmed to exist; `git status` confirmed no repo file touched by this unit besides
its own report; the kaizen entry it claims to have logged confirmed present in `kaizen_team`
verbatim. No defects found in any reported figure.

**Result is reported as a measured number only — not a sufficiency verdict.** Whether 0/23
(CI ceiling 14.3%) clears the bar for the first live, audience-facing demo is a stakeholder
risk-tolerance call, explicitly out of scope for this coordination per its own charter (see "Why
this exists" above) — relayed to the user, not decided here.

**A second, separate finding surfaced during the live pass and is not part of Mitigation D's own
result**: 62.7% of turns (235/375) got no reply at all, from LM Studio's own serving layer erroring
under this pass's 3-way concurrent load against the pinned model — confirmed contended by a second,
unrelated live process sharing the same LM Studio instance throughout the run, so not yet a clean
baseline. Filed as a new backlog item, **K-066** (`falkor-chat/docs/BACKLOG.md`), rather than folded
into K-065 — distinct failure mechanism (no reply vs. wrong-language reply), distinct owner/next
step (a dedicated-instance rerun before the rate can be trusted), and the report itself recommends
weighing it separately.

**Mid-run relay outcome**: the user-surfaced LM Studio log lines (22:54:53-55) were traced by U3 to
trial 12/pt-BR/turn 3 — an aborted tool-call whose never-delivered draft argument was itself
Spanish. Correctly excluded from the adherence tally (no reply reached the user) and reported as a
narrow, single-instance, not-separately-actionable sub-finding in the test report and in K-066.

**Coordination status: all three units delivered, verified, and (U1/U2) committed. U3's report and
this coordination doc are the last two pieces to commit** — holding both for this same integration
pass since they're the final artifacts of an otherwise-closed unit chain. K-065 itself stays open
in `BACKLOG.md`, marked in-progress pending the stakeholder's sufficiency call — not closed by this
coordination, per its own charter.

## Phase 2 — dedicated-instance rerun (2026-10-20)

**Stakeholder decision 2026-10-20:** Mitigation D's 0/23 (CI ceiling 14.3%) is not yet accepted as
sufficient; run one rerun against an LM Studio instance with no other consumer, at a larger n, so it
both tightens K-065's bound and gives K-066 a trustworthy dead-turn baseline. K-065 closes on that
result if it stays clean; C/B remain unauthorized.

The U3 harness (`language_salience_probe.py`) lived in a dead session's scratchpad and is lost —
U4 rebuilds it as a committed script (the U3 report's own Feedback item 4), U5 gates it statically
before the live budget is spent, U6 runs it. U4 → U5 → U6 strictly sequential. U6 is dispatched only
after the stakeholder confirms the LM Studio instance is dedicated for the run window.

**U4 verified by `teco`** (2026-10-20): 21/21 offline tests re-run green; smoke JSONL recounted
independently (per-language replied / no-reply / dead / dead-but-replied all match U4's table); own
argument mutant (`is_wrong_language` fed a constant `"en"` instead of the participant's language)
killed by the suite, file restored byte-identical. Not yet committed (held until the U5 gate). U5
runs in parallel with U6 to use the stakeholder's LM Studio window — safe for classifier defects,
since every raw reply is stored and re-scorable, not for turn-driving defects, which `teco` checked
by reading the post/poll/read loop. **U4 finding carried into U6:** the demo has two LLM consumers —
`ministral-3-3b` (assistant step) and `qwen/qwen3-4b-2507` (`query_graph_data` → `llm("step")`,
`falkor-chat/server/falkorchat/tools.py`) — so the 2026-09-18 report's attribution of the `qwen`
failures to the :8200 process was at least partly wrong. U6 preloads both and splits dead turns by
failing model.

**U5 verdict: approve with suggestions, no re-run needed** (`falkor-chat/docs/reviews/salesperson-language-salience.md`
Pass 2). `teco` reproduced F1 directly (`"Agregado: USB-C Hub 7-in-1 por $39.99"` → `en` for an `en`
participant) and confirmed the harness md5 unchanged. Relayed to U6 mid-run, 2026-10-20: the headline
wrong-language count is U6's **hand-read of every replied turn**; the v2 classifier figure is reported
alongside for comparability only; plus the unknown-trial count and the F4 trigger counts. Harness fixes
are U7, after U6, never during it.
