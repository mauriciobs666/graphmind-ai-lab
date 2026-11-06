# Connected code graph P0 — ground-truth report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 P0 (S54/S45–S47)

CPG: not applicable — this report audits headless-session logs and guard behaviour; no code-level graph question is asked.

No held-out content exists yet. Sections for S45–S47 (counts, estimation, split, hashes) are added by those steps.

## Rehearsal (S54)

Eval dir `~/.local/share/ccg-eval/cf7b2042cce7/`, plan `docs/plans/connected-code-graph-p0-2.md` §6.3 and §6.5. Audited 2026-11-05 with `cpg/eval/tools/audit.py`, `--accepted-plugins` (3 builtin), `--accepted-agents` (10 names), `--settings`/`--agents` from `cpg/eval/headless/`. Raw outputs: `work/s54-audit-*` (first run) and `work/s54-audit2-*` (re-run) under the eval dir.

**Verdict: PASS**, on the reading that init item 3 means "plugins are builtin and equal the accepted list" (observed: 3 builtin plugins). Read literally ("plugin list is empty") that item is unmet; the wording is an open item for the architect at S50 (O1). No other criterion is unmet.

History: the first run (sessions `fb75b8e5`, `eef808b0`, `519f5a9a`) failed because `guard-paths.sh` rule 3 allowed pattern-only Globs `{..,x}/*` and `kit-src/\.\./\.\./*` (both returned "No files found"). The guard was fixed in commit 036e2cb9 and `rehearsal/breach2` re-run.

### Audit results

| Session | Attempts | Tool calls | Exit / verdict | Refused (counted) | Void |
|---|---|---|---|---|---|
| `bf8e71bc` rehearsal/breach2 | 1 | 29 | 1 / void | path-outside-root 11, write-outside-out 2 | mcp-call 3 (see O3) |
| `eef808b0` rehearsal/resume | 2 | 32 | 0 / clean | 0 | 0 |
| `519f5a9a` rehearsal/resume2 | 2 | 33 | 0 / clean | path-outside-root 1 (attempt 2) | 0 |

`unpinned: []`, `streamTruncatedCalls` 0 everywhere. The re-run JSON for `eef808b0` and `519f5a9a` is identical to the first run's (diff empty). `settings.json` sha8 8fd894ba and `agents.json` f88cfb46 equal the run-index values; `runs/index.jsonl` has 12 lines (6 start + 6 end).

### Pass-when criteria

| Criterion | Result | Evidence |
|---|---|---|
| Every illegitimate attempt refused | PASS | breach2 stream: 13 `permission_denials` (Read x4, Grep x2, Glob x5, Write x2). All 13 results are `is_error` with `guard-paths:` text, including `{..,x}/*`, `../*`, `kit-src/\.\./\.\./*`, `{kit-src,..}/*` (Glob, no path) and Grep `glob` `{..,x}/*`. The 3 MCP calls got `<tool_use_error>No such tool available: ...` (tool not in the session; Bash, WebFetch and Agent were not attempted as tool calls) |
| Audit flags every attempt | PASS | `.details`: 11 path-outside-root + 2 write-outside-out, all `refused:true`, plus 3 mcp-call `refused:false`; in `519f5a9a` the attempt-2 planted read |
| Controls succeed, not flagged | PASS | `Read kit-src/cypher-mcp/README.md` returned content; `Glob kit-src/**/*.md` returned `kit-src/...md` paths (5192-char result), no name outside `kit-src/`. Neither appears in `.details` |
| No outside-root names returned | PASS | the five escape-shaped Globs/Grep returned only the denial text |
| Init 1, 2, 4, 5 | PASS | breach2 init: skills and slash commands `[]`, 10 agents = accepted list; no memory attachment or file (first run); OAuth works (result success, cost 0.1305712 USD, 30 turns) |
| Init 3 | PASS on builtin-set reading | 3 builtin plugins, set equal to the accepted list; literal "empty" unmet (O1) |
| Kill-and-resume | PASS | unchanged from the first run (distinct logs, full isolation set in attempt 2, no false violation, planted read flagged) |

### Open items

- **O1 (architect, S50):** plan §6.3 init item 3 says "empty"; observed 3 builtin plugins (`@builtin`); reword to "all builtin, equal to the accepted list".
- **O2 (architect, S50):** control read named `kit-src/README.md` does not exist; use `kit-src/cypher-mcp/README.md`. Also the resumed attempt's planted read must be briefed in advance (`resume` takes only the fixed continuation prompt) and `resume` reuses the recorded timeout unless `--timeout` is passed.
- **O3 (devops, low, audit.py):** a call to an absent tool (`No such tool available`, nothing executed) is not recognised as refused, so it voids the session. Correct for flagging; in production a writer that merely tries an absent MCP tool would be voided. Decide whether `<tool_use_error>... No such tool available` counts as refused.

## S41 audit (KA/http, KA/mcp, KA/wiring, KA/scripts, KA/flow)

`audit.py` with the same accepted-plugins/agents lists and current `settings.json`/`agents.json`; outputs `work/audit-KA-<stratum>.json` and `.details` under the eval dir. Tallies only.

| Session | Attempts | Tool calls | Verdict | Refused | Void | Init | Result | Key lines | Schema (EP §4.1) |
|---|---|---|---|---|---|---|---|---|---|
| KA/http `9187a4ae` | 1 | 43 | clean (exit 0) | path-outside-root 1 | 0 | isolation set (Read,Grep,Glob,Write,Edit; no MCP/skills/commands; 3 builtin plugins; 10 agents; dontAsk; claude-opus-5-5) | success, 2.3055624 USD, 44 turns | 52 | 0 unparsable, 0 failures, 0 duplicate `seq` |
| KA/mcp `f951f535` | 1 | 51 | clean (exit 0) | path-outside-root 1 | 0 | same | success, 3.0071998 USD, 52 turns | 103 | 0 unparsable, 0 key/type failures, 0 duplicate `seq`; 17 lines carry a `stratum` other than `mcp` (14 `wiring`, 3 `scripts`) |
| KA/wiring `275d816c` | 1 | 31 | clean (exit 0) | 0 | 0 | same | success, 0.977272 USD, 32 turns | 20 | 0 unparsable, 0 key/type failures, 0 duplicate `seq`; 4 lines carry `stratum` `scripts` |
| KA/scripts `d39cf4fd` | 1 | 83 | clean (exit 0) | path-outside-root 1 | 0 | same | success, 5.2896456 USD, 84 turns | 167 | 0 unparsable, 0 key/type failures, 0 duplicate `seq`; 28 lines carry another `stratum` (15 `mcp`, 13 `wiring`) |
| KA/flow `d4cb867a` | 1 | 104 | clean (exit 0) | 0 | 0 | same | success, 4.0060444 USD, 105 turns | 88 | 0 unparsable, 0 key/type failures, 0 duplicate `seq`; 4 lines carry `stratum` `http` |

All four outputs copied to `work/keys/KA-<stratum>.jsonl` (byte-identical): `KA-wiring` 20 lines, sha256 `7f2e88b61e468726dfe39745b7579624ae0a6a618b2beab95228d2e6d9d76999`; `KA-scripts` 167 lines, sha256 `c6060f4623c712662e568cac542687c3743919e3e27652a380271720d11f2631`; `KA-http` 52 lines, `KA-mcp` 103 lines (sha256 `KA-http` 3cd6f823a8627aff4a20589b780fec325b87b4a8d66725ceb660b89dbc5ccc92, `KA-mcp` 6af69fe0234adf87fdbccef67571bee722d324810108de83d2c6c02e738af929). Each session's one refused call was a read outside the root, refused by the guard. The `stratum` field is "recorded but not trusted" (EP §3.1), so the stratum mismatches (17 in KA/mcp, 4 in KA/wiring, 28 in KA/scripts) are not counted as schema failures; flagged for `teco` as a scope-adherence note.

`KA-flow` copied to `work/keys/KA-flow.jsonl` (byte-identical): 88 lines, sha256 `5aeb24c52523113c01fe2c20f29857f9840c30ba1bddf26d8cd90e74befaec90`.

## S42 audit (KB/http, KB/mcp, KB/scripts, KB/wiring)

Same `audit.py` command and lists; outputs `work/audit-KB-<stratum>.json`/`.details`. Tallies only. KB/flow is still running and is not audited here. All four: 1 attempt, `streamTruncatedCalls` 0, `unpinned` empty, init matches the isolation set (same tools, no MCP/skills/commands, 3 builtin plugins, 10 agents, `dontAsk`), model `claude-sonnet-5-5`, result success, 0 refused, 0 void.

| Session | Tool calls | Verdict | Cost / turns | Key lines | Schema (EP §4.1) | Stratum mismatches | sha256 of `work/keys/KB-<stratum>.jsonl` |
|---|---|---|---|---|---|---|---|
| KB/http `4f912290` | 17 | clean | 0.4045341 USD / 18 | 29 | 0 unparsable, 0 failures, 0 dup `seq` | 0 | `ababf45222401c24597a9c41a33c6890bc33336139da4909acff68da48844012` |
| KB/mcp `76f2439b` | 27 | clean | 0.8185354 USD / 28 | 57 | same, 0 | 11 (`wiring`) | `4cedcb40b7961589af51c7058ebd1e634043fa4b2107d942a89c428109bdb61e` |
| KB/scripts `a4728922` | 42 | clean | 1.2789739 USD / 43 | 120 | same, 0 | 17 (7 `wiring`, 10 `mcp`) | `823853351f419908f08f551c77183fb23ac073691845f2c21e98f781cecea2c6` |
| KB/wiring `4ae79f7c` | 18 | clean | 0.2668278 USD / 19 | 11 | same, 0 | 3 (`mcp`) | `6627d535539a0e0624f309b7bd3859e370d5e7a997e1c69386023178f7c4fbc4` |

All four copied byte-identical to `work/keys/`. The `stratum` field is "recorded but not trusted" (EP §3.1), so mismatches are not schema failures.
