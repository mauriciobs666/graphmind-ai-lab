# `salesperson` tool-call reliability (K-056) — implementation review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-056 (M6)

## Scope & verdict

Diff-scoped review of the uncommitted working-tree diff (`git diff`, not yet committed) that
implements U37's (`tdd-engineer`) K-056 fix pass: `server/falkorchat/executor.py` (`StepResult.
toolsUsed`, `_looks_fact_bearing`/`_note_possible_fabrication`, breadcrumb tagging in
`_assemble_messages`, `_link_emissions` threading), `server/falkorchat/repository.py`
(`link_step_emission` now `SET`s `Message.toolsUsed`; `read_thread` returns it),
`server/falkorchat/services.py` (pass-through), the three touched test files, and the doc updates
(`docs/QUERIES.md` §4/§12.6, `docs/HISTORY.md`, `docs/BACKLOG.md` K-056 rewrite). Baseline: the
prior committed state (`git diff` against `HEAD`) plus
`docs/reviews/salesperson-tool-reliability-ml.md` (the design note this pass implements) and
`docs/plans/workflow-salesperson-demo-coordination.md`'s U36/U37 notes (context on what was already
decided and why). Not in scope: whether to dispatch K-053 (explicitly out of scope this session),
and any live re-reproduction of D-1 itself (already live-verified 2/2 by U37; not re-run here).

**Verdict: approve with suggestions.** No blocker — the code is correct, the tests genuinely pin
the new behavior (not just execute it), the Cypher change is sound, and the doc updates
(`QUERIES.md`, `HISTORY.md`, `BACKLOG.md`) accurately and honestly describe what shipped, what
worked, and what didn't. One MAJOR finding on the specific question the brief asked me to weigh in
on: the breadcrumb-imitation risk is not an inert leftover, it is an active increase in the
existing defect's severity, and I'd recommend reverting just that one code path rather than
shipping it as-is — see Finding 1.

**CPG:** not applicable — `cpg_falkorchat` is known stale (confirmed by the coordination doc,
`da10d57`/`f30c378` since build) and this is a diff-scoped correctness/test/doc review of new code,
not a structural-impact-analysis question; no CPG query would answer anything asked here.

## Findings

### MAJOR 1 — the breadcrumb-imitation risk is a real severity increase on an already-open defect, not a neutral leftover; recommend reverting the tagging specifically

**Evidence:** U37 live-verified 2/2 that (a) the breadcrumb does not reduce fabrication
(`docs/HISTORY.md` 2026-08-28 entry, `docs/BACKLOG.md` K-056) and (b) on every fabricating turn in
both passes, the model's own posted reply text imitated the breadcrumb's surface format
(`"... [verified via <tool>]"`) **without calling the tool**. I traced the mechanism in
`executor.py:1025-1030` (the tagging is appended only from `Message.toolsUsed`, so a fabricated
reply's *own* stored message still carries an empty `toolsUsed` — the fake tag exists only in the
message's free-text `content`, not as a structured claim). That has a consequence beyond "does
nothing": once such a fabricated reply is posted, it becomes part of the replayed history for the
*next* turn (`_read_thread_context`) with the fake `[verified via ...]` string sitting in its raw
text — the model's own future in-context precedent now contains a self-authored example of
"answer confidently and claim verification" with no code-side tag distinguishing it from a real
one. This is the same instruction-vs-in-context-precedent mechanism §4.1 of the ml note diagnoses
for the underlying bug, now reinforcing a *more deceptive* pattern, not neutral filler.

**Why it matters:** before this change, a fabricated D-1 answer was wrong but plain — a customer
reading it had no code-shaped textual cue telling them it was checked. After this change, the same
fabricated answer can carry an explicit, specific-sounding verification claim
(`"[verified via lookup_product_fact]"`) that a customer (or a stakeholder watching a live demo)
is more likely to trust precisely because it looks like machine-generated audit output. That is a
strictly worse failure mode for the exact defect this pass was trying to mitigate, live-confirmed
2/2, not a hypothetical. `docs/BACKLOG.md`'s rewritten K-056 entry and the `HISTORY.md` entry both
describe this finding accurately and don't hide it — my disagreement is with the *shipping*
decision on this one code path, not with how it's documented.

**Suggested improvement:** revert just the tagging line in `_assemble_messages`
(`executor.py`, the `if tools_used: content += f" [verified via ...]"` block) while keeping
everything else — `StepResult.toolsUsed`, `_link_emissions`'s threading, `repository.
link_step_emission`'s `SET`, `read_thread`'s surfacing, and the observability signal
(`_note_possible_fabrication`) all have independent value and carry none of this risk (they are
pure audit/logging, never fed back into a prompt the model reads). That is a small, low-risk
diff on top of what's already here — the plumbing this pass built stays fully intact, only the one
code path that turned out to have a live-confirmed negative side effect goes away. If the team
still wants a replayed-history breadcrumb eventually, it should go through the controlled eval
the ml note's own §5 scope item (2) originally called for (testing the tag's *format* for
imitability, not just its presence) before it's re-shipped — this pass shipped it without that
eval on an explicit user instruction to skip straight to a live-verify-and-stop, which is a
legitimate call for *investigating* the mechanism quickly, but the negative result it surfaced
is exactly the kind of finding that eval step exists to catch before shipping, not after.

### MINOR 1 — `_looks_fact_bearing`'s bare two-decimal branch flags non-price numbers

**Evidence:** `executor.py`'s `_FACT_BEARING_RE = re.compile(r"\$\s?\d|\b\d+\.\d{2}\b")`. I ran it
directly: `"version 3.14 release"` and `"that is 100.00 dollars"` both match, but so does any
bare `\d+\.\d{2}` token with no currency context at all (a version string, a coordinate, a
percentage written with two decimals). The doc comment states the design intent correctly
("deliberately narrow — false negatives over false positives"), but the bare-decimal alternative
is looser than the currency-prefixed one and can fire on non-price facts.

**Why it matters:** low — this only affects an advisory, log-only signal (`_note_possible_
fabrication` never raises or blocks), so the cost is occasional noisy `WARNING` log lines, not a
functional defect. Still worth tightening once this signal is used for more than ad hoc log
review, since K-053/K-054/K-055 will add more numeric-looking legitimate content (quantities,
order totals) that could trip the bare-decimal branch even when correctly grounded.

**Suggested improvement:** either require the two-decimal token to be adjacent to a currency
symbol or a price-shaped context, or accept the current heuristic as intentionally coarse and
note in the docstring that it will also fire on non-price two-decimal numbers (a smaller,
cheaper fix than tightening the regex, if the noise is judged acceptable).

## What's solid

- `StepResult.toolsUsed` → `_link_emissions` → `services.link_step_emission` →
  `repository.link_step_emission` (`SET m.toolsUsed = $toolsUsed`) → `read_thread`
  (`coalesce(m.toolsUsed, [])`) → `_assemble_messages` is a clean, fully-parameterized,
  correctly-threaded pipeline. I verified it live: `test_repository.py`'s new tests round-trip a
  list property against the real FalkorDB build, and `test_executor_produced.py`'s new
  integration test drives a real node through executor→services→repository and reads
  `Message.toolsUsed` back off the graph — this isn't mocked at the boundary that matters.
- `_note_possible_fabrication` genuinely mirrors `_note_must_post_violation`'s structure and
  posture (unconditional `_log.warning`, trace-append-on-debug-only, staticmethod, advisory/
  never-raises) — the implementer's claimed precedent match holds up on inspection.
- Tests pin behavior, not just paths: e.g. `test_assemble_messages_never_tags_a_user_turn_even_
  if_toolsused_is_present` and `test_fabrication_warning_also_fires_on_max_iterations_exhaustion`
  each target a specific, easy-to-regress edge case rather than a happy path alone.
- I ran the full offline suite live (`1829 passed, 4 deselected`) and the three touched test
  files in isolation (`246 passed`) against the real `falkordb-dev` instance — both match the
  implementer's own claimed counts exactly.
- No new index/constraint is needed for `Message.toolsUsed` — it's never used as a filter
  predicate, only read back wholesale per message; the doc updates correctly don't propose one.
- `docs/QUERIES.md`, `docs/HISTORY.md`, and `docs/BACKLOG.md` (K-056 rewritten in place, per the
  no-dated-`Update:`-stacking convention) are unusually candid about the negative result — no
  spin, the fabrication-imitation risk is stated plainly in both `HISTORY.md` and `BACKLOG.md`.

## Open questions

- Is there an appetite to take the one-line revert in Finding 1 now (before K-053 dispatch, since
  K-053/K-054/K-055 will multiply the number of live conversations this risk can surface in), or
  is the team comfortable carrying it as a known, documented risk under the still-open K-056 item
  until the controlled eval (ml note §5 item 2) happens? Both are defensible; this needs the
  team's call, not mine to make unilaterally.

## Appendix — regex probe (Finding MINOR 1)

```
>>> import re; R = re.compile(r"\$\s?\d|\b\d+\.\d{2}\b")
True  It costs $29.99
True  29.99 is the price
False call me at 4.55pm
True  version 3.14 release        # false positive
False the year is 2026
True  It is $ 30
False room 4.02b
True  that is 100.00 dollars       # false positive (no currency token)
False no numbers here
```
