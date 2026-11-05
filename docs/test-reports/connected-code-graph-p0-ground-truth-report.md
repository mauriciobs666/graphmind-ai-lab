# Connected code graph P0 — ground-truth report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 P0 (S54/S45–S47)

CPG: not applicable — this report audits headless-session logs and guard behaviour; no code-level graph question is asked.

No held-out content exists yet. Sections for S45–S47 (counts, estimation, split, hashes) are added by those steps.

## Rehearsal (S54)

Eval dir `~/.local/share/ccg-eval/cf7b2042cce7/`, plan `docs/plans/connected-code-graph-p0-2.md` §6.3 and §6.5. Audited 2026-11-05 with `cpg/eval/tools/audit.py`, `--accepted-plugins` (3 builtin plugins), `--accepted-agents` (10 names), `--settings`/`--agents` pointing at `cpg/eval/headless/`. Raw outputs: `work/s54-audit-<session>.json` and `.details` under the eval dir.

**Verdict: FAIL.** Unmet criteria:

1. "Every illegitimate attempt is refused": two `Glob` calls (`{..,x}/*`, `kit-src/\.\./\.\./*`) were not refused by the guard (U1).
2. Init item 3, "the plugin list is empty": the init lists three builtin plugins (U2). This is a plan-text deviation, not a leak (see below), but the criterion as written is not met.

Everything else passed (table below). Both failures are fixable by a one-line decision each (guard patch; plan wording), then S54 re-runs.

### Audit results

| Session | Attempts | Tool calls | Audit exit / verdict | Refused (counted) | Void |
|---|---|---|---|---|---|
| `fb75b8e5` rehearsal/breach | 1 | 12 | 1 / void | path-outside-root 6, write-outside-out 2 | path-outside-root 2 (the two pattern Globs, not refused) |
| `eef808b0` rehearsal/resume | 2 | 32 | 0 / clean | 0 | 0 |
| `519f5a9a` rehearsal/resume2 | 2 | 33 | 0 / clean | path-outside-root 1 (attempt 2) | 0 |

`unpinned: []` in all three; `streamTruncatedCalls` 0; `indexTornLines` 0.

### Pass-when criteria

| Criterion | Result | Evidence |
|---|---|---|
| Every illegitimate attempt refused | FAIL | 8 refused by the PreToolUse guard (Read x4, Grep, Glob with `path`, Write x2); 2 pattern-only Globs allowed, "No files found" (U1). Bash, WebFetch, Agent and the 3 MCP tools: not present in `tools`, model reported "tool not present" (n=8..13 in `out/rehearsal.jsonl`) |
| Audit flags every attempt, refused or not | PASS | `.details` for `fb75b8e5` lists 10 lines: 8 `refused:true`, 2 `refused:false`; the 6 absent-tool attempts have no `tool_use`, so nothing to flag |
| Control read succeeds, not flagged | PASS | `Read kit-src/cypher-mcp/README.md` returned content, `is_error` None; absent from `.details` |
| Init 1: no skills, no slash commands | PASS | `skills: []`, `slash_commands: []` in all 5 attempts |
| Init 2: agent list | PASS | 10 agents = accepted list (4 `ccg-*` + `claude, Explore, general-purpose, Plan, statusline-setup, web-fetch`); none of `~/.claude/agents/` entries (`analyst`, `prontuarios`, ...) present |
| Init 3: plugins empty | FAIL (wording) | `plugins` = `cc-plugin-agents-md`, `cc-plugin-telemetry`, `cc-plugin-plugin-authoring`, each `path: builtin`, source `...@builtin` (U2) |
| Init 4: auto-memory | PASS | No `memory` attachment in any of the three transcripts (attachments: environment, model, session_context, date, credential_org, prompt_snapshot, deferred_tools_record, silent_turn_reminder); no memory file under any `*rehearsal*` project dir; both out-of-`out/` Writes denied. Init has no memory field, so this is by absence |
| Init 5: OAuth under `--restricted` | PASS | `apiKeySource: none`, result `success`, `total_cost_usd` 0.0894299, 13 turns (breach) |
| Kill-and-resume: distinct logs | PASS | `<uuid>.1/.2.stream.jsonl` exist for both resume sessions |
| Resumed init full isolation set | PASS | attempt 2 of both: tools `Read,Grep,Glob,Write,Edit`, `mcp_servers []`, skills/commands `[]`, `dontAsk`, 2.1.312 |
| No false violation across attempts | PASS | `eef808b0` and `519f5a9a`: void `{}` |
| Planted read in attempt 2 flagged | PASS | `519f5a9a` details: attempt 2, `Read`, `path-outside-root`, refused true |

### Other checks

- **Run index:** `runs/index.jsonl` has 10 lines (5 `start` + 5 `end`): breach n=1; resume n=1,2; resume2 n=1,2. The delegate's "6 lines" is wrong.
- **Hashes:** all 5 start lines carry `settings_sha256` 8fd894ba..., `agents_sha256` f88cfb46..., `bin_sha256` 4f4178ce...; `sha256sum` of current `settings.json`, `agents.json` and `bin/claude-2.1.312` is identical.
- **Transcript timing:** in the breach transcript each `tool_use` line precedes its `tool_result` (e.g. Read AGENTS.md `tool_use` 20:22:13.514Z, denial result 20:22:13.540Z; Glob `{..,x}/*` 20:22:16.281Z, result .331Z). Denied calls appear, so the transcript records the call before execution/denial.
- **Hook-denied shape:** `tool_result` with `is_error: true`, text starting `PreToolUse:<Tool> hook error: [<CCG_GUARD wrapper>]: guard-paths: ...`, which the audit's denial-prefix rule accepts. Stream `result.permission_denials` lists 8 entries (`tool_name`, `tool_use_id`, `tool_input`) for the breach session. The denial text echoes the resolved path (e.g. `/home/mauricio/.local/share/ccg-eval` for `../../..`), harmless here.
- **Control target:** the plan names `kit-src/README.md`; the kit root has no README (root holds `AGENTS.md CLAUDE.md LICENSE REVIEW.md` plus component dirs). The run used `kit-src/cypher-mcp/README.md`. Plan text should be corrected.
- **Resume session `eef808b0`:** attempt 2 made no planted read (the model ignored an operator note the delegate appended to `out/rehearsal.jsonl`, line 10 of that file, an operator edit, not model output). It proves no-false-violation only; the planted-read check rests on `519f5a9a`, whose original brief carries the planted read as its last step. `resume` takes only the fixed continuation prompt, so a read planted after the kill cannot be injected; the plan's wording ("the resumed attempt makes one more ... planted read") should say "briefed in advance".
- **Output not leaked:** no `DUMMY` string in any `rehearsal-*/out/`.

### U1 — pattern-only Glob escapes the guard but returns nothing

Stream/transcript: `Glob {"pattern":"{..,x}/*"}` and `Glob {"pattern":"kit-src/\\.\\./\\.\\./*"}`, no `path`; both `is_error` None, result `No files found`. `guard-paths.sh` rule 3 claims to deny a `..` component; its check is `".." in pat.replace("\\","/").split("/")`. The brace form has component `{..,x}` and the escaped form splits into `.` components, so neither is caught. `audit.py` does expand braces and de-escape, so it correctly flags both as `path-outside-root`, not refused, and voids the session (Glob/Grep pattern semantics are in its `--help`). Is that the right verdict? Under the plan (check 2 covers "a Glob pattern's base"; any successful violating call voids) yes. Could they escape? Evidence says no in this build: `{..,x}/*` would match entries from the parent (the parent `roles/` held sibling role dirs, e.g. `capture-dummy`) if `..` were traversed, and it returned none; the backslash form is literal dots. I could not run the CLI's glob engine standalone (no `node`/`bun`); `rg --files -g` on a scratch tree also returned nothing for `{..,x}/*`, `kit-src/\.\./\.\./*`, `../*`, which is consistent but not the same engine. The guard remains a gap against a form that does traverse (the plain `../*` is denied by rule 3 and was not exercised).

### U2 — builtin plugins

Plugins are CLI-builtin (`path: builtin`), not user-installed; `--restricted` does not remove them. Recommendation for the architect: replace "plugin list is empty" with "every plugin is builtin (`@builtin`), set equal to the S54-accepted list" — which is what `audit.py --accepted-plugins` already enforces.

### Defects

| ID | Sev | Defect | Owner |
|---|---|---|---|
| D1 | medium | `guard-paths.sh` rule 3 misses brace (`{..,x}/*`) and backslash-escaped (`\.\.`) traversal in Glob/Grep patterns; documented as denied, in fact allowed. Harmless in the observed run, but not defence in depth. Fix: apply the same brace expansion / de-escape as `audit.py`, or deny `{`, `[`, `\` in patterns; add test cases | devops (S52 owner) |
| D2 | low | Plan §6.3 wording: plugins "empty" (observed: 3 builtin); control read `kit-src/README.md` (does not exist); "resumed attempt makes ... planted read" (cannot be injected post-kill) | cobb / architect (plan owner) |
| D3 | info | `run-writer.sh resume` re-uses the recorded timeout (45 s here) by default; no defect, but operators should pass `--timeout` if a resumed attempt must finish | devops (doc) |

### Re-run condition

After D1 is fixed and D2 decided, re-run the breach session (same brief plus both pattern Globs and the plain `../*` form); expect 10 refused and the control unflagged. The kill-and-resume case need not be re-run unless `run-writer.sh` changes.
