# Review — K-071 D-3: broadened LM Studio unload marker in the transient-retry classifier

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-071 (follow-up of K-067)

## Scope & verdict

**Reviewed:** the uncommitted diff against `71826e36` in `falkor-chat/server/falkorchat/retry.py`
and `falkor-chat/server/tests/test_retry.py`. The change does three things:

- it replaces the D-2 marker `"model unloaded"` with `"unloaded"`;
- it adds the marker `"predict request failed: fetch failed"`;
- it adds four transient rows and one permanent row (`No models loaded`).

**Baselines:** plan `falkor-chat/docs/plans/extract-transient-retry.md` (§3.1, §3.2.1, §3.5, §6
R-1/R-2), the D-2 precedent `21ff9f9b` gated in Pass 7 of
`falkor-chat/docs/reviews/extract-transient-retry.md`, the coordination doc
`falkor-chat/docs/plans/extract-transient-retry2-coordination.md`, and the re-drive evidence in
teco's scratchpad (`k071/d3-error-bodies.txt`, `k071/server-log-redrive1.txt`).

**Not reviewed:** the 422 on one `POST /documents` and the `background fuse failed` tracebacks
(D-1 / K-068) in the same log, plus any docs work. I did not touch the `:8200` server or
LM Studio.

**Verdict: approve with suggestions.** There are no blockers or majors. One minor is a test gap
that a mutant demonstrates; two nits concern comment wording. There is also one out-of-scope
observation.

CPG: considered, not relevant — a two-string allow-list change; verified by executing the classifier, the two suites and mutants directly.

### Evidence gathered (executed)

- **Suites.** `tests/test_retry.py` and `tests/test_transport.py` give **96 passed**.
- **Observed bodies.** I passed every distinct body in `d3-error-bodies.txt` to `is_transient` as
  `ProviderCallError(kind="http", status=…, detail=…)`. All 8 shapes return `True`. That covers
  the 6 unload wordings, `terminated`, the fetch-failed body, and the 500 HTML page (transient by
  status).
- **Mutants.** I ran these on a scratch copy and `cmp` confirmed the live file is untouched
  afterwards. The full table is in Appendix A.
  - **Killed:** `"model unloaded"` (the D-2 revert) → 3 failed; `"model is unloaded"` → 2;
    `"loaded"` → 2; deleting the fetch marker → 1.
  - **Survived:** `"unload"`, `"fetch failed"`, and the phrase list
    `("unloaded by", "is unloaded", "was unloaded")`. The last one is F-1.
- **Callers of the classifier.** `retry.call_with_retry` has exactly two production callers:
  - `ingestion.py:138`, the `extraction` kind;
  - `embedding.py:259`, chunk embed only.

  `embed_message` and the agent/step/guard calls are never retried. The protocol is
  OpenAI-compatible only (`modelconfig._infer_protocol` rejects anthropic at load time). Shipped
  defaults point both kinds at LM Studio (`config/models.json`). The live
  `~/.config/opencode/opencode.json` declares one provider, which is LM Studio.

## Findings

### F-1 (minor): two of the six observed unload wordings are not pinned by any row

- **Evidence.** `{"error":"Model unloaded."}` (5 chat hits) and `{"error":"Model unloaded.."}`
  (1 embed hit) are in `d3-error-bodies.txt`, but they have no row in `_TRANSIENT`.
- **Why it is unpinned.** The marker covers them today only because the substring also hits
  the D-2 and D-3 rows.
- **Executed proof.** I replaced `"unloaded"` with the phrase list
  `"unloaded by", "is unloaded", "was unloaded"`. That is exactly the shape the exact-phrase
  alternative would take. Result: **96 passed**, yet `_has_marker('{"error":"Model unloaded."}')`
  returned `False`.
- **Why it matters.** A later "tighten the marker" change could silently reintroduce the D-3
  failure for the second-most-common chat body in this re-drive.
- **Fix.** Make the observed-body axis the probe. Add one `_TRANSIENT` row per **distinct
  verbatim body captured live**, which is all six unload wordings in `d3-error-bodies.txt`.
  Note in the comment that a new capture adds its row. A single "covers all" row is not enough.
  Owner: `tdd-engineer`. It costs two rows and kills the mutant above.

### F-2 (nit): the fetch-failed comment states an inference as fact

- **Evidence.** `retry.py:59`: "Engine's internal fetch to its model worker failed mid-swap".
- **What was actually observed.** Only the body and the fact that it arrived inside the swap
  burst. Chunk `ab08bc8b…` had its embed fail on `Model was unloaded while the request was still
  in queue..` (log :840/:886). Its extract then got the fetch-failed body (:1756/:1802), and a
  `terminated` body follows at :1850.
- **What is unconfirmed.** That this is LM Studio's internal IPC, and that it happened
  "mid-swap", are both readings of the text and the timing.
- **Fix.** Append `(inferred, one observation)`, the same honesty tag the `"channel error"`
  line carries.

### F-3 (nit): the unload comment says "any wording"

- **Evidence.** `retry.py:55`.
- **Fix.** It matches any wording *containing "unloaded"*. Say that, so the next reader does not
  assume "unload"/"unloading" are covered.

### O-1 (observation, outside D-3, pre-existing by design): jobs of an already-failed document still make attempt 1

- **What the log shows.** The 18 `owning document no longer processing; abandoning retry` lines
  are each jobs that called the provider, failed transiently, and then found the document
  already failed at the pre-sleep gate read.
- **Why it happens.** `_safe_*` has no status check (plan §2.6, line 190). The gate is read only
  *between* attempts (§3.5). So every remaining job of a failed document still spends one
  provider call. Under JIT swapping, those calls add swaps.
- **Not a defect.** It is the specified design, and D-3 does not change it.
- **Possible follow-up, for `architect`.** Read the gate once before attempt 1. It is a natural
  companion to K-070, which cuts swaps per document.
- **Caveat.** This is **inferred** from reading the code. I cannot count from the log how many of
  these calls began *after* the failure.

## Answers to the brief's questions

**Q1: is `"unloaded"` too broad?** No. The false-positive risk is real but bounded and already
accepted, and it is worth paying.

- **Exposure is narrow.**
  - Only two call sites reach the classifier (Evidence).
  - Only `kind in {"http","body_error"}` consults markers, and only for a non-408/429/5xx status.
  - `detail` is the provider's body, capped at 500 chars (`transport.py:63,107`).
  - No OpenAI 4xx text I probed contains "unloaded" (Appendix B). Neither does any LM Studio
    permanent body known to the plan or review:
    - `No models loaded…`
    - `Model not found`
    - `model is not loaded`
    - `Failed to unload model`
    - `Invalid model identifier…`
- **The worst plausible case.** An LM Studio JIT-off configuration error worded with "unloaded".
  That is hypothetical: I could not confirm LM Studio's JIT-off wording, and the web search
  found nothing. Its cost would be the plan's **R-2** class: one ladder of ~40–60 s, 4 calls,
  then the *same* `failed` outcome.
  - The liveness gate caps it to one budget per document. On REST that is the first job; on MCP
    it is wall time (§6 R-2, "Accepted").
  - Any of those jobs then hits the gate.
- **Precedent.** A bare common word is already accepted as a marker: `"terminated"`. That word is
  likelier than "unloaded" to appear in an echoed input, and that risk has not materialised.
- **The alternative has failed twice.** Two observation rounds produced six wordings:
  - "unloaded by user or API request."
  - "unloaded."
  - "unloaded.."
  - "is unloaded."
  - "is unloaded.."
  - "was unloaded while … in queue.."

  An exact-phrase list needs at least three phrases and is still one wording away from the next
  D-n. The cost asymmetry also favours the substring. A missed transient fails a whole document
  permanently (11 of 13 here). A wrong retry costs one bounded ladder. Pass 7's "by user"
  trade-off (an operator eject is retried) is unchanged.

**Q2: is the fetch-failed marker justified on one observation?** Yes.

- §3.2.1 pre-authorizes a marker for "a 4xx whose body names a load, crash, or serving failure".
  An engine's predict request failing on a failed fetch is a serving-layer transport failure.
  - It is the HTTP-400-wrapped analogue of `kind="connection"`, which is transient
    unconditionally.
  - It shares the `Engine protocol …` family with the existing `"engine protocol startup was
    aborted"`.
- The log context puts it inside the swap burst (F-2's evidence).
- The marker is a specific phrase rather than bare `"fetch failed"`, which is the right
  conservative choice for n = 1.
- The broadening mutant `"fetch failed"` survives. That is acceptable: I know of no permanent
  shape to pin against, and inventing one would be imagination, not evidence.
- An unobserved embed-side variant (`… embed request failed: fetch failed`, if it exists) stays
  under R-1. Add it when it is captured.

**Q3: are coverage and comments adequate?** Mostly.

- **Coverage.**
  - The D-3 rows use verbatim live bodies.
  - The permanent `No models loaded` http row correctly pins the `"loaded"` over-broadening (the
    mutant is killed).
  - A `body_error` row per new marker is unnecessary for the reason Pass 7 Q3 gives: one
    `_has_marker` serves both branches, and both branches are already pinned.
  - Gap: F-1.
- **Comments.**
  - Test comments are accurate. The `# chat` / `# embeddings` tags match the evidence file, and
    "none matched the D-2 marker" is true for the four chosen rows.
  - `retry.py` comments: F-2 and F-3.
  - The permanent-row comment's rationale ("configuration, not a swap") is sound.
- **Module docstring.** It needs no change: it already describes markers generically.

**Q4: is "abandoning retry" after a sibling's fast failure consistent with the K-067 design?**
Yes, and there is no new defect.

- **The traced sequence.**
  1. A job of a document that was already `failed` by a sibling's fast (pre-D-3, unclassified)
     unload body still runs attempt 1, by design (O-1).
  2. It gets a body that *was* already transient: a 500, `terminated`, or `Model unloaded.`.
  3. The pre-sleep gate reads `False`.
  4. It logs the §3.5 INFO line with `label`/`kind`/`status`, re-raises the same exception, and
     `_safe_*` logs it.
  5. `report_document_job_done` changes nothing: first-terminal-wins (K-051).
- **What I checked.**
  - Every abandon line's following traceback that I could attribute carries a body that was
    transient on the pre-D-3 server: 9× HTML 500, 3× `terminated`, 3× `Model unloaded.`,
    1× `Model unloaded..` (16 of 18; two adjacent abandon pairs share a traceback block).
  - The one fetch-failed body (chunk `ab08bc8b…`, :1756) has **no** abandon line, as expected:
    pre-D-3 it was non-transient, so it re-raised with no log.
  - Post-D-3 the same event would log an abandon INFO instead, with the identical document
    outcome.
- **What it means.** The abandon lines are a *consequence* of D-3: siblings failed fast on
  unmatched unload bodies and failed the document. They are not a separate defect.
- **What D-3 does not cure.** Whether retries converge under 13 overlapping document chains
  thrashing one-model-at-a-time LM Studio is not something D-3 can guarantee. If the next
  re-drive still ends `failed` with transient kinds in the give-up lines, the plan's §3.2.1
  closure reasoning applies: the mechanism fired, and the remedy is K-070 (fewer swaps), not more
  markers.

## What's solid

- **Minimal and correct.** The change is a two-string allow-list edit inside the one module that
  owns classification. Plan §3.1's facts/policy split is preserved.
- **Verbatim, attributed test bodies.** Every new row uses a live-captured body, dated and
  attributed to its source.
- **A sound negative row.** The permanent `No models loaded` row targets the exact
  over-broadening a reader would worry about.
- **Within pre-authorization.** The authorization reading matches A1 and Pass 7 (R-1 + §3.2.1),
  so no plan revision is needed.
- **Claims reproduce.** The implementer's mutant claims reproduce (Appendix A).

## Open questions

- **F-1 timing.** Take F-1 before commit, or record it as a nit-level follow-up? I recommend
  before commit, since it is two rows.
- **D-3 docs.** D-3 needs its own `falkor-chat/docs/HISTORY.md` entry. The 2026-10-16 entry
  names `"model unloaded"`, which is now superseded in code. Per the coordination doc's
  documentation-impact list, that is teco's.

## Appendix

### A. Mutant runs

These ran on a scratch copy of `falkorchat/` + `tests/` with `PYTHONPATH` precedence, over
`test_retry.py` + `test_transport.py`.

| Mutant (replacing `"unloaded"` unless noted) | Result |
|---|---|
| none (baseline) | 96 passed |
| `"model unloaded"` (D-2 revert) | 3 failed |
| `"model is unloaded"` | 2 failed |
| `"loaded"` | 2 failed |
| delete fetch marker | 1 failed |
| fetch marker → `"fetch failed"` | 96 passed (broadening; accepted, Q2) |
| `"unload"` | 96 passed (broadening; no permanent "unload…" body reaches these call sites) |
| `"unloaded by"`, `"is unloaded"`, `"was unloaded"` | **96 passed, but `Model unloaded.` → `False`** (F-1) |

### B. `_has_marker` probe on plausible permanent bodies (all `False`)

- `No models loaded. Please load a model in the developer page or use the \`lms load\` command.`
- `Model not found`
- `model is not loaded`
- `Failed to unload model`
- `Invalid model identifier "foo". …`
- `Model reloaded`
- OpenAI `The model \`gpt-x\` does not exist…`
- OpenAI `context_length_exceeded`

A hypothetical `Unloaded model cannot be used. JIT loading is disabled.` returns `True`. That is
the R-2-class case discussed in Q1; it was invented for the probe, not observed.
