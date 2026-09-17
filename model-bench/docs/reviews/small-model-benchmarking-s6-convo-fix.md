# S6 `convo.py` fix — the double `role:"system"` message defect (U160)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** U160, S6 (M6)

## Scope & verdict

Independent code review of the uncommitted diff in `model-bench/modelbench/convo.py` +
`model-bench/tests/test_convo.py` (`git diff` against `HEAD`, both files still unstaged at review
time). This is `tdd-engineer`'s fix for a live defect: `assemble()` used to unconditionally emit
**two** separate `role:"system"` messages when both a system prompt and tool schemas were
configured; `mistralai/ministral-3-3b`'s own chat template rejects a second system-role message
outright (confirmed live, HTTP 400). The fix replaces `_system_message` + `_tool_schema_message`
with `_tool_schema_text` (pure text) + `_prologue_system_message` (collapses to **at most one**
`role:"system"` message), wired into `assemble()`, with 5 new tests and 2 pre-existing tests
corrected. Baseline: working tree at review time; no other files in this diff.

**Verdict: approve with suggestions.** The fix is correct on every axis I checked (truth table,
no orphaned callers, docstrings, suite+lint), and I independently mutation-tested a third
boundary (the `turn_index == 0` condition) that neither the delegate nor the coordinator had
already covered — it reddened as expected. One documentation-debt finding (minor) is worth a
follow-up: the design plan's own §4 S2 order clause still describes the pre-fix shape.

CPG: considered, not relevant — `model-bench` carries no requirement for a Joern CPG
(`skills/cpg-analysis/SKILL.md` lists no `cpg_model-bench` graph; `mcp__cypher__query(graph="GRAPHS")`
confirms none is loaded), and this is a small, self-contained diff in one module a static read
covers directly — a graph traversal would add nothing a `grep` for the two old symbol names
didn't already answer.

## Findings

### Minor — the design plan's §4 S2 order clause still states the pre-fix, two-message shape

`docs/plans/small-model-benchmarking.md:5132-5133` (v1.31, still `active`) reads: *"in this order:
system prompt if any · the tool-schema text block on turn 0, or every turn when
`representToolSchemasEachTurn` · the replayed history · ..."* — silent on the new invariant that
these two pieces now merge into **one** `role:"system"` message. `convo.py`'s own docstrings (the
module docstring's `representToolSchemasEachTurn` bullet is unchanged and still fine, but
`assemble()`'s "Order:" clause and `_prologue_system_message`'s own docstring) were updated
correctly by this fix; the plan that those docstrings cite by section number (§3.3, §3.8.4, §4 S2)
was not. This repo's own convention for this exact document is unusually strict about plan/code
consistency (a per-rule pin table at §7, `docs/plans/small-model-benchmarking.md` history entries
for every prior revision), so a reader who trusts the plan over the code — which is the document's
whole premise — would rebuild the two-message shape that just broke a live run. Suggested fix: a
dated plan revision note (the plan's own convention: bump to v1.32, one line) updating §4 S2's
order clause to state the merge explicitly, e.g. "...one merged `role:"system"` message, the
system prompt text (if any) then the tool-schema text block when it applies, concatenated rather
than sent as two messages...". This is a documentation-only follow-up, not a code change; routes
to `architect` (the plan's owner) rather than back to `tdd-engineer`.

## What's solid

- **Truth table (item 1 of the brief), enumerated by hand:** the merge condition is exactly the
  old two-branch condition's union. `cfg.systemPrompt` truthy gates the prompt part in both old and
  new code, unconditionally on `turn_index`; `cfg.toolSchemas and (turn_index == 0 or
  cfg.representToolSchemasEachTurn)` gates the schema part in both, byte-identical as a boolean
  expression (`convo.py:392`, unchanged from the old `assemble` inline check). The only change is
  *how many messages* the two parts produce when both are true (one, joined by `"\n\n"`, vs the
  old two) — verified against all 4 presence combinations at `turn_index == 0`
  (`test_assemble_merges_system_prompt_and_tool_schemas_into_one_system_message`,
  `test_assemble_system_prompt_alone_is_unaffected_by_the_merge`,
  `test_assemble_tool_schemas_alone_is_unaffected_by_the_merge`,
  `test_assemble_no_system_prompt_omits_the_system_message`) and against the
  `representToolSchemasEachTurn`/`turn_index` interaction at later turns via the two
  **pre-existing** tests (`test_assemble_represent_tool_schemas_each_turn_{false,true}_...`, both
  still green, unmodified by this diff) plus the new
  `test_assemble_merges_system_prompt_and_tool_schemas_on_a_later_turn_too`.
- **No orphaned callers (item 2):** `grep -rn "_system_message\|_tool_schema_message" model-bench`
  finds zero references outside the deleted definitions themselves — both were genuinely private
  to this one call site. Two other message-builders exist elsewhere in the codebase
  (`modelbench/runner.py:334`'s `_item_chat_messages`, `modelbench/lmstudio.py:651`'s `warm_up`)
  but neither ever appends more than one `role:"system"` message, so neither carries the same
  latent defect and neither is in scope for this fix.
- **Independent mutation test (item 3):** I corrupted the `turn_index == 0` boundary myself —
  dropped the `or cfg.representToolSchemasEachTurn` branch (a different mutation from both the
  delegate's revert-to-two-messages and the coordinator's join-order swap) — reran the suite,
  confirmed exactly one test reddened
  (`test_assemble_represent_tool_schemas_each_turn_true_keeps_schemas_every_turn`, with a clear
  assertion failure showing the schema text missing from turn 1's message), restored via `cp` from
  a pre-mutation copy, confirmed `diff -q` byte-identical, and reran the full suite green (1625
  passed, 3 deselected).
- **Docstrings (item 4):** `_prologue_system_message`'s own docstring and `assemble()`'s "Order:"
  clause both accurately state the new single-message merge and its motivation (the live HTTP 400
  against `mistralai/ministral-3-3b`); neither over- nor under-claims against what the code does.
- **Full suite + lint (item 5), run myself:** `.venv/bin/python -m pytest -q` → **1625 passed, 3
  deselected** (the `live` marker, per `pyproject.toml`'s default `-m "not live"`). `.venv/bin/ruff
  check .` → **all checks passed**.

## Open questions

None — the fix is sound and complete for the defect it targets; the one finding above is a
documentation follow-up, not a gate on this diff.
