# `model-bench` S7 — `chat-responder` pack: live-run test report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (S7)

## Summary

Executed `docs/plans/small-model-benchmarking-s7-spec.md` §5 Step 6 — the one live-run obligation
S7 owes (plan §5's stage-ownership table, item 16 only: "one full run per pack, end to end,
producing a stored, valid result"; no numbered item 17/19a/19b applies, per the spec's own §6). Run
against `model-bench/` at commit `9d38b2a` (worktree otherwise clean apart from this report and the
two new stored run/report artifacts this pass produced), model `qwen/qwen3-4b-2507` (the same model
used throughout S5/S6), pack `chat-responder-grounded-answers@0.1.0`, all 30 items FR-19-signed-off.

**Verdict: item 16's own done-condition is met — PASS on the mechanical claim.** The run stores
successfully, `compare`'s output shows `groundingRate` as the headline metric with the three format
counts rendered as exploratory (never pooled), the new "## Speed" section renders with real,
non-null numbers, and the reply-quality caveat renders as the first thing under the title, before
any number. **However, the spot-check this report's own brief required (§ "Spot-checks against real
replies" below) found a real, reproducible defect**: `modelbench/scoring/grounding.py`'s
`_ABSTENTION_MARKERS` list does not recognize this model's own dominant abstention phrasing
("The passages don't mention X"), and `checklist_pass`'s plain-substring containment does not
tolerate ordinary morphological paraphrase ("4 retries" vs. "4 retry attempts", "30 minutes" vs.
"30-minute"). Both cause the scorer to mark a correct, well-grounded reply as `checklistPass:
false`. This is not a crash and not a blocker to closing item 16 (the run itself completed and
stored cleanly, exactly as required) — but it is a real defect in the pack's own headline metric,
material enough that `groundingRate` as currently measured materially understates this model's
actual grounded-reply rate. Reported in full below, not silently worked around.

**CPG:** considered, not relevant — reconfirmed live this session via `mcp__cypher__query GRAPHS`
(28 graphs listed, no `cpg_model-bench`); this is a code-level task in a component with no loaded
CPG.

## Commands run, in order

1. `./run.sh validate --pack packs/chat-responder-grounded-answers` →
   `chat-responder-grounded-answers 0.1.0 (chat-responder): valid`, exit 0.
2. `./run.sh validate --pack packs/chat-responder-grounded-answers --strict` → raises
   `NotImplementedError` naming runner-spec §9 (`cli.py`'s own documented, deliberate deferral —
   identical to the exact behavior S6's report already recorded for this same flag on a different
   pack, `docs/test-reports/small-model-benchmarking-s6-report.md` acceptance-check 3; not a
   defect, confirmed by direct execution here too).
3. **Environment fork, resolved with evidence** (same class as S6's own Fork 1): `./run.sh run
   --pack chat-responder-grounded-answers --model qwen/qwen3-4b-2507 --session s7-live` first
   refused: *"LM Studio changed since you last attested host.json — re-check the app version and
   KV cache setting, then `model-bench attest`."* Checked LM Studio's own on-disk state directly
   (WSL2 mount `/mnt/c/Users/mauri/.lmstudio/.internal/`): `backend-preferences-v1.json` showed the
   CUDA llama.cpp runtime engine had advanced `2.39.0` → `2.40.0` since S6's own same-day
   attestation; `historical-version-info.json` confirmed the LM Studio app version itself was
   unchanged (`0.4.24`). Re-attested with the same, evidence-confirmed operator fields — no field
   guessed: `./run.sh attest --set lmStudioAppVersion=0.4.24+1 --set kvCacheSetting=f16 --set
   hostRamGb=16 --set otherResidentWorkloads=` → `wrote host.json`.
4. `./run.sh run --pack chat-responder-grounded-answers --model qwen/qwen3-4b-2507 --session
   s7-live` → `stored: results/runs/chat-responder-grounded-answers-qwen_qwen3-4b-2507-2026-09-17T18:19:18Z.json`
   (18.8s wall-clock; this is the live invocation item 16 itself requires).
5. Same command again (second live invocation, same model, same session tag — the spec's own §5
   Step 6 suggested design for the cheapest single additional live-run proof) →
   `stored: results/runs/chat-responder-grounded-answers-qwen_qwen3-4b-2507-2026-09-17T18:19:41Z.json`
   (18.8s wall-clock).
6. `./run.sh compare --pack chat-responder-grounded-answers --session s7-live --negative-control`
   → rendered and stored `reports/chat-responder-grounded-answers-20260917-01.md` (full output
   below). `--negative-control`'s own documented mechanics (`cli.py:_select_arms`, read directly
   before use rather than guessed): it selects the newest matching stored record and duplicates it
   as **both** arms — "two copies of ONE record, deliberately... proves the mode is wired, not that
   the harness is sound" (the flag's own `--help` text, confirmed against `cli.py`). Both live runs
   above exist on disk regardless; the negative-control comparison itself is, by its own documented
   design, over one of them duplicated, not a paired comparison of the two independent runs.

## Real output — `compare`'s rendered markdown (verbatim, from step 6)

```markdown
# Comparison — chat-responder-grounded-answers@0.1.0 (chat-responder)

> **Reply quality is not measured by this pack.** `groundingRate` is a deterministic containment check against the retrieved context, never a judgement of how good, helpful, or well-written a reply is (FR-21a — the judged-quality layer is deferred, `docs/BACKLOG.md`).

> **NEGATIVE CONTROL (WIRING SMOKE CHECK)** — both arms are the *same stored record*, so `b = c = 0 by construction` and this comparison **cannot fail**. It proves the mode is wired; it says nothing about whether the harness is sound. The real negative control is two **independent** runs of the same model and is an acceptance step, not this (`-ml` §9, plan §5 test 19a).

## Arms

| arm | metric | k/n | rate | 95% Wilson |
|---|---|---|---|---|
| qwen/qwen3-4b-2507 | groundingRate | 15/30 | 0.500 | [0.332, 0.668] |
| qwen/qwen3-4b-2507 | formatMaxWords | 30/30 | 1.000 | [0.886, 1.000] |
| qwen/qwen3-4b-2507 | formatSingleParagraph | 30/30 | 1.000 | [0.886, 1.000] |
| qwen/qwen3-4b-2507 | formatNoForbiddenPatterns | 30/30 | 1.000 | [0.886, 1.000] |
| ... (identical second arm — same underlying record, per negative-control mechanics) ...

## Speed

| arm | p50 | p95/max | timed/n | withheld (load/no-resp) | TTFT median | prefill ms/1k | tok/s median (diagnostic) |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-4b-2507 | 575 | 1142 | 30/30 | 0/0 | 39 | 129.8 | 55.7 |
| qwen/qwen3-4b-2507 | 575 | 1142 | 30/30 | 0/0 | 39 | 129.8 | 55.7 |

*Descriptive only — decode tokens/sec is a diagnostic, never a comparison instrument (FR-11).*

## Verdicts

### groundingRate
Not distinguishable at this sample size. Observed difference +0.0 pp, 95% CI [-0.0, 0.0] pp covers
zero (b=0, c=0, McNemar exact p=1.000)... [by-construction, per the negative-control's own design]

### Exploratory metrics
- `formatMaxWords` — exploratory — no significance claim
- `formatSingleParagraph` — exploratory — no significance claim
- `formatNoForbiddenPatterns` — exploratory — no significance claim
```

(Full, unabridged output is on disk: `reports/chat-responder-grounded-answers-20260917-01.md`.)

## Done-when checklist (spec §5 Step 6), each confirmed explicitly against the real output above

- **The run stores successfully** — two new files under `results/runs/`, both exit 0. **PASS.**
- **`compare`'s output shows `groundingRate` as the headline metric** — first metric row in "##
  Arms", and the only metric named in "## Verdicts" (the other three appear only under "###
  Exploratory metrics"). **PASS.**
- **The three format counts render as exploratory, never pooled with `groundingRate`** — confirmed:
  `formatMaxWords`/`formatSingleParagraph`/`formatNoForbiddenPatterns` each get their own "##
  Arms" row and their own "exploratory — no significance claim" line; none contributes to the
  `groundingRate` verdict's own McNemar/Wilson computation. **PASS.**
- **The new "## Speed" section renders, with real numbers** — first time `RunResult.latency` has
  ever been printed by this component (spec §2.5). Real, non-null figures observed: p50 575ms,
  p95 1142ms, 30/30 timed, 0/0 withheld, TTFT median 39ms, prefill 129.8ms/1k, 55.7 tok/s median.
  Cross-checked directly against the stored run JSON's own `latency` block
  (`latencyMsP50: 574.68...`, `latencyMsP95: 1142.38...`, `ttftMsMedian: 39.274`,
  `prefillMsPer1kMedian: 129.79...`, `tokensPerSecondMedian: 55.72...`) — the rendered table's
  rounding is the only difference. **PASS.**
- **The reply-quality caveat renders, first thing under the title, before any number** — confirmed:
  the `> **Reply quality is not measured by this pack.**` line is literally the first content line
  after the `# Comparison —` title (the negative-control banner comes second, also before any
  number — both are prose, not data). **PASS.**

## Format-check exercise — honest finding: the format axes were never meaningfully exercised

The brief for this run explicitly asked to confirm at least one real reply is long/unstructured
enough to exercise `maxWords`/`mustBeSingleParagraph`/`forbiddenPatterns` for real, and to say so
plainly if not. **It was not.** Checked directly against both stored runs' per-item `wordCount`:
the longest of all 30 replies was 52 words (`cr-14`); the shortest was 5. The pack's own default
`maxWords` is 150; the two items that override it per-item (`cr-08`: 60, `cr-22`: 40) still landed
at 17 and 15 words respectively — nowhere near either limit. No reply anywhere in either run
contained a blank line (`mustBeSingleParagraph` never at risk) or matched `forbiddenPatterns`
(no reply used a bullet/numbered list or a code fence — the model's replies were uniformly one to
three short prose sentences). **All three format axes read 30/30 in both runs because every single
reply trivially satisfied every constraint, not because the format-check logic was exercised
against real variance.** The format-check logic itself is unit-tested for real (spec §5 Step 0,
already gated); this live run simply never produced a reply that would have failed it. Recorded
here as the brief instructed, rather than silently treated as a clean pass.

## Spot-checks against real replies — the brief's own claim under test, and a real defect found

The stored run record does **not** persist raw reply text — only `outcome`, `scoreable`, `counts`,
and a `detail` block of `{abstained, checklistPass, wordCount}` per item (confirmed by reading both
stored JSON files in full: no `message`/`content`/`reply` field anywhere in `items[]`). Spot-
checking the scorer's verdict against real reply text therefore required reproducing calls live,
using the actual harness code (`modelbench.packs.load_pack`, `modelbench.scoring.grounding.
build_messages`, `modelbench.lmstudio.LMStudio.chat`, `grounding.checklist_pass`/
`looks_like_abstention`) rather than the literal stored artifact — the same model, pack, prompt
config, and `temperature: 0.0`, but a fresh call each time. This is itself worth naming as a
testability gap (below): a scorer verdict cannot be audited after the fact from the stored record
alone.

**Ten items spot-checked directly** (all 8 `mustAbstain: true` items, plus two `mustContain`
items whose stored-run verdict was `checklistPass: false`), each reproduced live and read by hand
against its item's `mustContain`/`mustNotContain`/`mustAbstain`:

| Item | Ground truth | Real reply (this session's live reproduction) | Human verdict | Scorer verdict | Match? |
|---|---|---|---|---|---|
| cr-03 | abstain (context has no revenue figure) | "The passages don't mention that the payment-timeout incident cost the company in lost revenue." | correct, grounded abstention | `checklistPass: false` (`looks_like_abstention` → `false`) | **NO — false negative** |
| cr-06 | abstain | "The passages don't mention who first noticed and reported the search-latency spike..." | correct abstention | `false` | **NO** |
| cr-09 | abstain | "The passages don't mention that." | correct abstention | `false` | **NO** |
| cr-15 | abstain | "The passages don't mention that." | correct abstention | `false` | **NO** |
| cr-18 | abstain | "The passages don't mention any compliance framework... Therefore, I don't know..." | correct abstention | `true` (the added "I don't know" clause happens to match a marker) | yes (this once) |
| cr-24 | abstain | "The passages don't mention which moving company..." | correct abstention | `false` | **NO** |
| cr-26 | abstain | "The passages don't mention which cloud storage vendor..." | correct abstention | `false` | **NO** |
| cr-28 | abstain | "The passages don't mention which specific customer accounts..." | correct abstention | `false` | **NO** |
| cr-11 | must contain "4 retries", "24 seconds" | "...allows **4 retry attempts**. ...delay before giving up is about **24 seconds**." | correct, grounded answer | `false` (`"4 retries"` is not a literal substring of `"4 retry attempts"`) | **NO — false negative** |
| cr-17 | must contain "30 minutes", "8 hours" | "...a **30-minute** idle timeout and an **8-hour** absolute session cap." | correct, grounded answer | `false` (hyphenated adjective form ≠ the item's noun-phrase form, confirmed via `_canon_str` directly) | **NO — false negative** |

Also spot-checked two of the stored run's `checklistPass: true` items for the opposite direction
(does a real *pass* verdict hold up?): **cr-12** ("dead-letter queue", reply quoted the passage
verbatim) and **cr-13** ("15 minutes", reply quoted the passage verbatim, explicitly citing "as
stated in the OAuth bug report") — both genuinely correct, grounded, verbatim extractions; the
`true` verdicts hold up. And one genuine model failure was confirmed, not a scorer bug: **cr-22**
(the context states the answer, "$180,000 ARR combined," but the model's reply claimed "the
passages don't mention the revenue... this quarter" — the model failed to extract a fact that was
plainly present; `checklistPass: false` here is the *correct* verdict on a real model miss, not a
scorer defect). One item, **cr-16**, is a softer, lower-confidence finding: its question asks for
an "original" idle-timeout value "before it was changed," but its context only states a current
10-minute value without establishing a prior value or a change — the model's abstention
("The passages don't mention the original... value... Therefore, I don't know") is a defensible
reading of a genuinely ambiguous question/context pairing, not obviously a scorer bug or a clear-
cut model miss; flagged as an item-content risk, not folded into the defect below.

### Defect — `_ABSTENTION_MARKERS` does not recognize this model's dominant abstention phrasing, and `checklist_pass`'s literal-substring containment does not tolerate ordinary morphological paraphrase

**Severity: High** (metric-validity / construct-validity defect — not a crash, but it materially
misrepresents the pack's own headline number, which every future model comparison on this pack
will be built on).

**Steps to reproduce:**
1. `cd model-bench && .venv/bin/python` and run:
   ```python
   from modelbench.packs import load_pack; from pathlib import Path
   from modelbench.scoring import grounding
   from modelbench.lmstudio import LMStudio
   pack = load_pack(Path("packs/chat-responder-grounded-answers"))
   item = {it["itemId"]: it for it in pack.iter_items()}["cr-03"]
   msgs = grounding.build_messages(item, pack=pack)
   reply = LMStudio("http://localhost:1234").chat(
       msgs, model="qwen/qwen3-4b-2507",
       temperature=pack.prompt_config().temperature,
       max_tokens=pack.prompt_config().maxTokens, timeout_s=60.0,
   ).message["content"]
   print(reply)
   print(grounding.checklist_pass(reply, must_contain=[], must_not_contain=["$180,000"], must_abstain=True))
   ```
2. **Expected**: the model correctly declines to answer (the context has no revenue figure) →
   `checklist_pass` should return `True`.
3. **Actual**: reply is "The passages don't mention that the payment-timeout incident cost the
   company in lost revenue." — `looks_like_abstention()` returns `False` because none of
   `_ABSTENTION_MARKERS`' 15 fixed phrases ("not found", "no matching", "couldn't find", "don't
   have", "no data", "unable to find", "no information", "not available", "no record", "i'm not
   sure", "i don't know", "cannot find", "can't find", etc. — `grounding.py:29-33`) match "don't
   mention"/"doesn't mention," which is this model's own default, natural abstention phrasing.
   `checklist_pass` therefore returns `False` on a correct, well-grounded abstention.
4. **Same mechanism, all 8 `mustAbstain: true` items in the pack**: the stored run scored **8 of 8**
   (100%) of these items `checklistPass: false`. Live reproduction of all 8 confirms 7 of the 8 use
   the identical "The passages don't mention..." phrasing the marker list misses; the 8th (`cr-18`)
   passed on reproduction only because it happened to also add a redundant "I don't know" clause
   this time (the model is not bit-exact deterministic even at `temperature: 0.0` — the same
   non-determinism S6's own report already documented, `identical: false` on its own determinism
   probe) — in the actual stored/graded run, `cr-18` failed too, consistent with the same pattern
   rather than evidence against it.
5. **A second, distinct mechanism, confirmed on `cr-11`/`cr-17`**: `checklist_pass`'s
   `mustContain` check is plain, canonicalized (case/whitespace-only) substring containment.
   `"4 retries" in "4 retry attempts"` is `False`; `"30 minutes" in "30-minute"` and
   `"8 hours" in "8-hour"` are both `False` — confirmed directly via `_canon_str`, not just by
   eye. Both replies are correct, grounded, natural paraphrases of the expected fact; both were
   scored `checklistPass: false`.

**Expected vs. actual, in one line:** a checklist designed to measure "did the model give a
grounded, correct reply" instead measures "did the model happen to phrase its reply using one of
15 fixed abstention idioms, and did it happen to match the item author's exact noun-phrase
morphology" — for this run, that gap accounts for **at least 9 of the 15 items** (60%) the pack
scored as `groundingRate` failures, out of a headline `groundingRate` of 15/30 (0.500). The true
grounded-reply rate this run's data supports, correcting only the confirmed false negatives above
(9 of 15 failures reclassified), is closer to **24/30 (0.80)** than the reported 0.500 — a
directional estimate from this session's spot-check, not a re-scored, code-fixed figure (no
production code was changed by this QA pass, per this agent's own guardrails).

**Evidence on disk:** reproduction scripts and full transcripts used above are session scratch
(`/tmp/.../scratchpad/spotcheck.py`, `spotcheck2.py`) — not committed, reproducible from the steps
above against any live LM Studio instance with this model resident.

## Coverage & gaps

**Covered:** the full item-16 done-when checklist (all five clauses, each confirmed against real
output above); a negative-control smoke pass per the spec's own suggested design; a direct,
code-level spot-check of 12 of 30 items' real replies against their own `mustContain`/
`mustNotContain`/`mustAbstain`, split across both failure classes the pack can produce (abstention
mismatch, containment mismatch) and both verdict directions (confirmed a real pass, confirmed a
real fail, confirmed nine false-negative fails).

**Gaps, named rather than hidden:**
- **The stored run record persists no raw reply text** — `detail` carries only `{abstained,
  checklistPass, wordCount}`. This is a real testability/auditability gap: nobody can re-audit a
  past graded run's actual verdicts without re-calling the model live (as this pass had to). Worth
  naming alongside S6's own similar finding ("`report.py` prints no per-script breakdown" —
  `docs/test-reports/small-model-benchmarking-s6-report.md`, Feedback & recommendations) as the
  same class of gap on a different role.
- **The `cr-16` item-content ambiguity** (above) is a lower-confidence finding, not resolved here —
  worth a second human read of that one item's question/context pairing, not a code fix.
- **The "textbook default" guessability concern the FR-19 pre-check flagged for `cr-12`/`cr-13`**
  (`docs/reviews/small-model-benchmarking-s7-precheck.md` §"Minor — cr-12 and cr-13...") — both
  items passed cleanly in this run, with real replies that quote the source passage directly
  (`cr-13`'s reply even explicitly cites "as stated in the OAuth bug report"). This run's data does
  not distinguish "the model read the context" from "the model guessed a plausible textbook
  default and got lucky" for these two single-value items — the concern is neither confirmed nor
  ruled out by this pass; a genuinely discriminating test would need a variant with a
  non-default value, which is out of this run's scope.
- **Remaining 6 of the 15 stored-run failures** (`cr-27`, `cr-29`, `cr-30`, plus one already
  covered as ambiguous, `cr-16`) were not individually spot-checked — the 12 checked already
  establish the defect pattern with high confidence; checking the rest would refine the exact
  false-negative count, not change the finding.
- **No acceptance-tier gate beyond item 16 was owed or attempted** (spec §6 — S7 introduces no new
  statistical machinery, no new pack-loader axis, no new role-dispatch branch); this report does
  not attempt a paired model comparison or a McNemar verdict, matching the spec's own scope.

## Feedback & recommendations

- **Recommended fix (not implemented here, per this agent's guardrails)**: widen
  `_ABSTENTION_MARKERS` to include a "context/passages don't/doesn't mention" pattern (this
  model's own dominant idiom, and a plausible common one for other models too), and consider a
  light morphological tolerance for `checklist_pass`'s containment check (e.g. also trying a
  hyphenated/pluralized variant, or word-boundary-tokenized comparison rather than raw substring)
  — or, short of a code change, tightening item-authoring guidance so `mustContain` values are
  chosen to be robust to the most common paraphrase of the same fact. Either is a `tdd-engineer`/
  `coder` fix with an `analyst` re-gate, not a QA-pass fix.
- **Recommended**: persist raw reply text (or at least a truncated excerpt) in the stored run
  record, gated behind a flag if storage size is a concern — the single biggest lever for making a
  future QA pass's spot-check reproducible from the stored artifact itself rather than requiring a
  fresh live call every time.
- This finding is a genuine, not-yet-seen result for a pack that had never run against a live model
  before this pass (per this task's own framing) — worth flagging to `teco`/the stakeholder before
  this pack's `groundingRate` is used as the basis for any future model comparison, since the
  metric's current false-negative rate is large enough to change a comparison's outcome.

## Evidence on disk

- Stored run records: `model-bench/results/runs/chat-responder-grounded-answers-qwen_qwen3-4b-2507-2026-09-17T18:{19:18,19:41}Z.json`.
- Comparison markdown: `model-bench/reports/chat-responder-grounded-answers-20260917-01.md`.
- `host.json` re-attested `2026-09-17T18:19:13Z` (evidence-based, not guessed — narrative above;
  gitignored, not tracked).
- Pack under test: `model-bench/packs/chat-responder-grounded-answers/` (`packVersion: 0.1.0`, all
  30 items FR-19-signed-off per the coordination doc's S7 Step 5 entry).
