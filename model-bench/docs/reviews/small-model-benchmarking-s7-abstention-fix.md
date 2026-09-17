# S7 `chat-responder` abstention-detection fix — independent code gate (U174)

> **Status:** active · **Owner:** `analyst` · **Tracks:** U174, U175 (S7)

## Scope & verdict

Reviewed the uncommitted diff on `modelbench/scoring/grounding.py`,
`tests/test_scoring_grounding.py`, and `docs/BACKLOG.md` — `tdd-engineer`'s fix (U173) for the S7
live-run defect (`docs/test-reports/small-model-benchmarking-s7-report.md`, "Defect —
`_ABSTENTION_MARKERS` does not recognize..."): `looks_like_abstention` gained a second detection
path, `_MENTION_ABSTENTION_RE` (`\b(?:don't|doesn't) mention\b`), gated by
`_CONTRASTIVE_CONTINUATION_RE` (`\b(?:but|however)\b`) searched only *after* the idiom match. Did
not re-derive `teco`'s own independent re-verification of U173 (suite reproduction, two mutation
tests, one fresh live call) — took that as given per the brief — and instead ran my own
verification: read the full diff, re-ran the full suite and `ruff`, and probed the regex's actual
boundary behavior against edge cases the implementer's own fixtures don't cover, executing every
claim below rather than inspecting it.

**Verdict: approve with suggestions.** The fix correctly resolves the reported defect (verified:
suite 1693 passed/3 deselected, `ruff` clean, original 15-marker path structurally unreachable
except via early-return before the new code runs — no regression). No blocker. Three Major findings
below describe foreseeable recurrences of the *same defect class* this fix targets — narrower in
scope than the original, not yet observed in live data, and outside what the `data-scientist`
consult explicitly scoped this fix to cover — so they do not block landing, but should not be
treated as closed either.

**CPG: not applicable — no `cpg_model-bench` graph exists on this FalkorDB instance (confirmed
repeatedly this coordination); regex/heuristic logic in a Python scoring module has no CPG
call-graph or data-flow angle a graph would add here regardless.**

## Findings

### Major — contrastive-marker list is too narrow; semantically equivalent connectives reintroduce the exact false-negative-on-a-real-answer defect this fix targets

`_CONTRASTIVE_CONTINUATION_RE` only recognizes "but"/"however". Verified live against the running
code (`.venv/bin/python`, not just read):

```
"The passages don't mention this directly, although based on the numbers given, the answer is 42,000."  -> True (should be False)
"The passages don't mention this specific figure, though the context lets us compute it..."              -> True (should be False)
"The passages don't mention this directly, yet the numbers given add up to 42,000."                      -> True (should be False)
```

All three are hedge-then-answer replies structurally identical to the fix's own three covered
adversarial fixtures, differing only in the connective word. Each is misclassified `abstained: True`
on a reply that actually answers — exactly the shape of defect this fix exists to close (a correct,
grounded answer scored as a `groundingRate` failure), just triggered by a different, equally common
English connective. Suggest widening the alternation to
`\b(?:but|however|although|though|yet|even though)\b`, backed by a test per new connective (see the
test-coverage finding below for why a case list alone under-covers this).

### Major — the "after the idiom only" search direction is unpinned by any test, and a plausible answer-then-hedge reply defeats it

The guard calls `_CONTRASTIVE_CONTINUATION_RE.search(canon, match.end())` — only positions *after*
the idiom. I mutated this to an order-insensitive `search(canon)` (whole string) and re-ran all 8
new tests: **all 8 pass identically** under the mutant, so no test in the diff actually pins the
"later in the reply" design choice the docstring (`grounding.py:42-44`) calls out as intentional.

Probing the direction this leaves open, an answer-then-hedge reply is misclassified:
```
"The answer is 42,000, but the passages don't mention the exact breakdown." -> True (should be False)
```
The model states the answer first, then hedges about a missing detail — "but" precedes the idiom,
so the current guard never sees it and the reply is wrongly scored as abstention. Given the
observed corpus (all 7 confirmed real abstentions lead with "The passages don't mention...", so
`teco`/`tdd-engineer`'s empirical grounding is sound for what's been seen), this specific ordering
hasn't occurred live yet — but it's a materially different, equally plausible LLM reply shape, and
nothing in the fix's design note limits itself to "hedge-first" replies. Suggest either widening the
search to the whole string (matching the order-insensitive framing the docstring already uses
informally — "not followed later in the reply", which invites but doesn't require "later than the
match") or, if hedge-first is intentionally the only covered shape, saying so explicitly in the
comment and adding a test that pins it (a red test today, since the mutant above proves none does).

### Minor — a genuine abstention with an unrelated post-idiom "but"/"however" is wrongly excluded (opposite-direction false negative)

Verified live:
```
"The passages don't mention this specific detail, but neither do they contain any related information, so I cannot determine the answer." -> False (should be True)
"The passages don't mention the figure, however I searched carefully and found nothing else relevant either."                              -> False (should be True)
```
Both are genuine abstentions that use "but"/"however" for reasoning unconnected to "the reply goes
on to actually answer" — the exact false-negative class the original defect exhibited, reintroduced
by the new guard itself. Lower severity than the two above because it requires the model to add
extra reasoning clauses after the idiom, a narrower shape than the plain hedge-then-answer pattern
the fix was built against; still worth a documented limitation or a test pinning the
known-acceptable trade-off, since right now nothing records that this trade-off was made
deliberately versus overlooked.

### Minor — only contracted "don't"/"doesn't" is recognized; "does not"/"do not mention" is not

Verified: `looks_like_abstention("The passages does not mention that fact.")` and the `do not`
variant both return `False`. Plausible alternate phrasing from the same or a different model,
outside this fix's empirically-scoped 7-reply evidence base. Cheap to close now (extend the
alternation to `\b(?:don't|doesn't|do not|does not) mention\b`) or defer explicitly — either is
fine, but it should be a decision, not a gap nobody noticed.

### Minor — test coverage pins known cases, not the regex's actual boundary

Beyond the order-insensitivity gap above: the 8 new tests are all either the report's own quoted
real replies or constructed hedge-then-answer variants using the two implemented connectives. None
probes the boundary the fix's own docstring claims to have chosen deliberately (search direction,
connective set). Per this program's own established idiom elsewhere (the S7 report's own careful
adversarial-fixture discipline), a boundary this consequential — it gates a "High"-severity
metric-validity defect class — warrants tests for the *closed* alternative (order-insensitive
search) and the *rejected* connectives (`although`/`though`), not just the shipped behavior.

## What's solid

The core fix is correct and well-evidenced: `_MENTION_ABSTENTION_RE` is checked only after the
fixed 15-marker list fails (`grounding.py:54-56`), so the original markers' behavior is provably
unreachable-to-change — confirmed by reading the diff (pure addition, no edits to the `any(...)`
branch or to `checklist_pass`) and by the full suite staying green. The regex itself
(`\b(?:don't|doesn't) mention\b`) is correctly scoped with word boundaries — no partial-word or
cross-token false matches found in testing. The three original hedge-then-answer adversarial tests
are genuine, real-shaped fixtures, not synthetic straw cases. `docs/BACKLOG.md`'s new item
accurately transcribes the cr-11/cr-17 evidence verbatim-consistent with the report (`"4 retries" in
"4 retry attempts"` is `False`, `"30 minutes" in "30-minute"`/`"8 hours" in "8-hour"` both `False`)
— checked directly against `docs/test-reports/small-model-benchmarking-s7-report.md:233-238`, no
drift. Suite (1693 passed, 3 deselected) and `ruff check .` both reproduced clean, matching the
claimed baseline exactly.

## Open questions

Whether to fold the Major findings' fixes (widen the connective set; decide the search-direction
question explicitly) into this same unit before it lands, or land as-is and track them as a fast
follow-up — the underlying defect class is the same one just fixed, so leaving it open has a real
(if currently unobserved) cost to the pack's headline metric on a future run with a different reply
shape. That call belongs to `teco`/the stakeholder, not to this gate.

## Pass 2 — 2026-09-17 (U175 re-gate)

**Verdict: approve with suggestions**, unchanged. Findings 1 and 4 closed. The findings-2/3
trade-off resolution is sound and its own documented limitation is accepted as-is — no cleaner
heuristic exists without real semantic understanding inside this component's zero-runtime-
dependency, regex-only design constraint (FR-23), and both directions are honestly pinned by
their own tests. But the *mechanism* chosen to implement the trade-off (`_sentence_span`) has one
new, directly-verified Major defect that silently reopens finding 2 through a different trigger.

- **Finding 1 (narrow connectives) — fixed, confirmed.** Re-ran all three original repro strings
  (`although`/`though`/`yet` hedge-then-answer) directly against `07cccb1`: all three now correctly
  return `False`.
- **Finding 4 (contracted-only idiom) — fixed, confirmed.** `"does not mention"`/`"do not mention"`
  both now correctly return `True`.
- **Findings 2/3 (search-direction trade-off) — accepted resolution, confirmed sound.** Re-ran the
  original finding-2 case (`"The answer is 42,000, but the passages don't mention the exact
  breakdown."` → `False`) and both finding-3 cases (`False`-classified-should-be-`True` in pass 1,
  now `True`) directly: all match. The digit-proxy's own documented residual limitation
  (`"...but there were only 2 documents retrieved, so I cannot determine the answer."` →
  misclassified `False`) was reproduced exactly as the code's own docstring and dedicated test
  describe — an honest, load-bearing trade-off, not a hidden gap.

### Major (new) — `_sentence_span` conflates a decimal number's internal period with a sentence boundary, silently reopening finding 2

Root-caused by tracing `_sentence_span` against a constructed case, then confirmed live:

```
"The average value is 42, but according to page 4.5 the passages don't mention the source." -> True (should be False)
```

This is exactly finding 2's shape (answer-first, hedge-after, connective before the idiom) — it
passes in every finding-2 regression test *without* a decimal number between the connective and
the idiom, and fails the moment one is added. Mechanism: `_sentence_span`'s
`canon.rfind(ch, 0, start)` over `.!?` treats the period inside `"4.5"` as a sentence boundary, so
`sentence_start` lands just after that decimal point — *after* `"but"`. The truncated sentence
handed to `_CONTRASTIVE_CONTINUATION_RE.search` therefore never contains `"but"`, `connective` is
`None`, and `looks_like_abstention` falls through to `return True`. Suite/`ruff` don't catch this —
it's a fresh case, not a regression of any committed test.

This is squarely in-scope, not a contrived corner case: this pack's own domain (dollar amounts,
percentages, page/section references) is decimal-heavy, and a reply citing a decimal figure
between a hedge and the "don't mention" idiom is a plausible real shape from a small model
narrating its reasoning. Suggested remedy: `_sentence_span`'s boundary regex should not treat a
period as a boundary when it sits between two digits (a `\d\.\d` guard, e.g.
`re.compile(r"(?<!\d)[.!?](?!\d)")` in place of the bare-character `rfind`/`find` scan), with a
regression test using this exact reply shape (or a close paraphrase) added to
`TestLooksLikeAbstentionContrastiveScope`. Not a blocker on its own — findings 1-4 as originally
filed are genuinely closed, and this is a new, narrower recurrence of the same defect class rather
than a failure to address what was asked — but it should be tracked as a fast follow-up rather than
treated as done, for the same reason findings 1-4 were: it's the identical construct-validity risk
this whole line of fixes exists to close.
