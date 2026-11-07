# Review: team-learnings follow-ups (uncommitted cobb edits)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-063

## Scope & verdict

Scope: `git diff` (uncommitted, vs HEAD) of `skills/python-web-quirks/SKILL.md`, `skills/README.md`, `skills/agent-standards/claude-code.md`. Root `REVIEW.md` applied (its own diff, the "did not reproduce" line, is not under review).

**Verdict: needs changes** — one major (README row lost a column); everything else is clean.

CPG: considered, not relevant — the change is Markdown/skill prose only; no code-level component.

## Findings

### F1 (major, verified) — README catalog row for `python-web-quirks` has 3 cells, not 4
`skills/README.md` catalog: every other row has four cells (skill | what it is | when to load | owner/origin; counted with `awk -F'|'`: five rows give 4, `python-web-quirks` gives 3). The new row folds the "when to load" content into the description cell ("For `coder`, …") and drops the column. Rendered, the owner cell shifts left under the "when to load" header. Brief check (2) said other columns stay intact; they do not.
Fix: restore a fourth cell, e.g. `| Writing/reviewing Python where a behavior contradicts a quick docs read (asyncio/background tasks, FastAPI/Starlette routing and error handling, urllib/redis clients, pytest/import-timing, TestClient) — for coder, tdd-engineer, architect, analyst, qa-engineer. |` before the `cobb` cell. Other rows: unchanged (the diff touches exactly one row, verified).

### F2 (minor, verified) — one trigger class not carried into the new description
The old description named a bare `pytest.raises` with no `match=` passing on an unrelated same-type raise. The body keeps it (`match=` x2) but the new description has no trigger for it, so a "my raises-test passes for the wrong reason" situation won't match. Add e.g. "`pytest.raises` without `match=`" to the Triggers list (description is 914 chars; ~100 spare).

### F3 (nit) — the two edits of the hook point read slightly in tension
The enclosing bullet is titled "a hook allow is **not** a full override"; new point 3 documents that under `-p`+`acceptEdits` the probe's hook allow was the whole decision. Not a contradiction (point 3 explicitly says the counterfactual was not run), but a reader skimming the title may take point 3 as refuting it. Optional: open point 3 with "Under `-p`:" in the sentence that reports the probe write.

## Checks run

1. **python-web-quirks description.** `yaml.safe_load` parses both versions; keys `name, description, allowed-tools`; new description 914 chars (≤1024; note the old one was 6,492, already over the limit the README states). Body unchanged: `git diff` has a single hunk `@@ -4,70 +4,11 @@`, all within the frontmatter. Grepped the body for ~40 distinctive tokens from the old description (create_task, total_tokens, proxy_headers, exclude_unset, 404.html, app.routes, include_router, responses=, add_exception_handler, min_length, IncompleteRead, exc.read, isfinite, vars(, DontReadFromInput, config.option.keyword, /v1, `__pycache__`, PYTHONDONTWRITEBYTECODE, match=, raise_server_exceptions, ServerErrorMiddleware, app.state, `_global_shutdown_lock`, "can't start new thread", -q -q, PYTHONPYCACHEPREFIX, isError, --lf, --deselect, daemon…): every one has ≥1 hit. Nothing lost from the body. Symptom-class coverage of the new triggers: all old classes map to a trigger except F2.
2. **README.** Only the `python-web-quirks` row changed (see F1). Skill and origin cells intact.
3. **claude-code.md.** Old bullet fully re-homed as point 3 of the "PreToolUse allow is not a full override" bullet: 6,708 / 109 / 107, sessions `b0bbcc3d` and `973830b1`, 2.1.316, v2.1.295+, "a run killed in between loses the attachment", "inferred, not shown", "counterfactual not run", the hooks-docs "not re-read" caveat, and "settings deny/ask rules still apply after an allow" (now correctly cross-referenced to point 1, which is the settings-rule layer) — all present; "scan of the same files" correctly rewritten to "this repo's session files" since the antecedent bullet was not adjacent. Grep for the old title across the repo (excluding `.git`): only the two frozen `docs/reviews/team-learnings-distillation6*.md` remain (history, left alone); no live dangling reference.
4. **Size discipline.** No `*AGENTS.md` file in the diff (`git diff --stat -- '*AGENTS.md'` empty); no always-loaded line changed. (`claude-code.md` lines 278-280 exceed 700 chars but are untouched and the file is not always-loaded.)

## What's solid
Description rewrite is a genuine improvement (valid length, trigger-oriented, body untouched); the claude-code.md merge is lossless and removes a partly-overlapping duplicate.

## Open questions
None.
