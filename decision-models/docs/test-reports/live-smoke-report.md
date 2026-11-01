# U5.7 live smoke report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** U5.7 (plan6 §4)

Run 2026-11-01 against commit f14aeefa, Claude Code 2.1.316. Throwaway roots under the session scratchpad (`.../scratchpad/smoke1`, `smoke2`); real `~/.claude` read by the CLI for auth only.
CPG: not applicable — a live-smoke of an agent campaign driver, run through its CLI, not a code-level analysis.

## Setup
- `smoke-fixture` wrote 50 items; `init --phase smoke --fixture`; `install` rc=0, guard command printed: `python3 "$CLAUDE_PROJECT_DIR/.claude/guard_hook.py"`.

## Batch 1 — 50-item at-cap `pt_br_item_at_cap`, `maxTurns: 14` (`driver run --root smoke1`)

Command: `driver run --root .../smoke1` (human-started). Output: `launched session 9c6b5850-...` / `stopped: done (nothing is pending)`, rc=0, ~36 s wall (ledger `issued` 23:15:19Z, `outcome` 23:15:54Z).

- **Ingest, no void:** ledger `outcome: ingested`, `voidClass: null`, `details: []`; request file 321,963 bytes; `status` = "1 ingested". PASS.
- **Foreground shape (first spawn):** `.meta.json` = `{"agentType":"dm-labeler","requestShape":"foreground","requestNonInteractive":true,"finished":true,"spawnDepth":1}`; driver `toolUseResult.status` = `"completed"`. PASS.
- **Largest page tokens:** NOT directly observable as the plan specifies. The labeler issued the 9 page Reads **in one parallel assistant message**, so there is no usage delta per Read. Observed: that one message's follow-up request has `cache_creation_input_tokens` 164,763 (vs 437 before) for all 9 results together; no `is_error` on any Read (Read's 25,000-token cap not hit). Tool-result sizes (JSON-dumped chars): 36,771 / 38,004 / 37,991 x6 / 18,995. Proportional estimate for the largest page = 164,763 x 38,004 / 321,716 = about 19,460 tokens (estimate, not a measurement; the aggregate delta also includes ~2k output tokens of that turn, so the true figure is lower). Upper bound by sum = 164,763 for the 9 pages. Verdict: **PASS on estimate (<= 22,500), but the plan's "usage deltas around each Read" method cannot measure a parallel-Read batch** (finding F-1).
- **Turn counts, clean 50-item batch:** subagent = 4 assistant API messages (Read header; 9 parallel Reads; Write; `done`) with `maxTurns: 14`; driver = 2 assistant messages; run JSON `num_turns: 2`. 11 tool uses (10 Read, 1 Write), 11 guard `hook_success` entries + 1 on the driver's `Agent` call (12 lines in `store/hook-log.jsonl`, all `allow`).
- **`SubagentHandback`:** no `SubagentHandback` tool call in either transcript; the subagent ended `end_turn` with text `done`; subagent tools used = {Read, Write}. The driver's `Agent` tool_result is wrapped in a harness "[Subagent hand-back]" text frame, which is not a tool call. NOT PRESENT, so "is it hooked" is unobservable (unchanged: U).
- **`hook_success.command` vs `guardCommand`:** driver and subagent `hook_success` both carry `command` = `python3 "$CLAUDE_PROJECT_DIR/.claude/guard_hook.py"` byte-equal to the ledger's `guardCommand`. PASS. Unexpanded in the record.
- **`$CLAUDE_PROJECT_DIR` expands in the guard command:** yes. Guard ran (`exitCode 0`, stdout a JSON `permissionDecision: allow`, reason "the pending spawn", `durationMs` 26); no `hookMissing`. PASS. No fallback needed.
- **`timeout: 30` accepted:** hook ran normally with `timeout: 30` in `settings.json`; no warning/error in `.err` (empty). PASS.
- **Extra `Agent` input keys:** driver's `Agent` input keys = `description`, `prompt`, `subagent_type` only. None.
- **Cost/tokens observable:** run JSON `total_cost_usd` 0.4965734, `modelUsage` {claude-sonnet-5-5: inputTokens 12, outputTokens 4204, cacheRead 175,854, cacheCreation 172,936}; `costBasis: "list"`; ledger run record tokens: input 8, cacheWrite 169,880, cacheRead 173,305, output 3,923. Subagent self-report: `subagent_tokens: 169885`, 11 tool uses, `duration_ms: 30788`. Driver `permission_denials: []`, `terminal_reason: completed`.
- **Injected context:** no `instructions`, ledger `instructions: []`; audit raised no void (injected-context rows implicitly passed; I did not separately re-derive them).

## Batch 2 — `maxTurns: 2` labeler copy (`install --agents-dir .../scratchpad/agents2`, root `smoke2`)

Smoke-only copy of `agents/` with `dm-labeler.md` `maxTurns: 14` -> `2` (diff: that one line). `driver run --root smoke2` printed three `launched session ...` lines then `stopped: done (nothing is pending)`, rc=0. One `run` made **three** spawns of the same batch: attempts 1-3 each `outcome: void`, `reason: maxTurns`, `voidClass: run-attributable`, `details: ["the run ended without a final reply"]`; then `status` = "1 dropped", `dropped: {"retryCapExceeded": {"items": 50, "seeds": 0}}`.

Cut-short shape (attempt 1, session ad70d7d5):
- Subagent: 2 assistant API messages (turn 1: one `Read` of the header; turn 2: nine parallel page `Read`s), every Read guard-allowed and answered (no `is_error`), then the transcript **ends on a user `tool_result` entry** (page offset 50, 20,703 chars). No final assistant entry, no `Write`, no `SubagentHandback`, no stop marker in the subagent transcript.
- Driver's `Agent` `tool_result` text: `NOTE: this agent stopped at its 2-turn limit before finishing. It was still calling tools and had produced no report.` plus `<usage>subagent_tokens: 4836 / tool_uses: 11 / duration_ms: 13563</usage>`; `toolUseResult.status` = `"completed"`, `canContinueAgent: false`, `toolsUsed: ["Read"]`, `.meta.json` `finished: true`, `requestShape: "foreground"`. The driver then ended `end_turn` with `done` (run JSON `terminal_reason: completed`, `num_turns: 2`).
- Terminal entry the audit keys on: the subagent's last entry = user `tool_result` (no assistant reply) together with the driver-side `NOTE: ... stopped at its 2-turn limit` text. Audit class matched: void `maxTurns` (not `spawnShapeUnknown`). PASS for the provisional row's behavior; U5.5 should swap its fixture for this shape.
- Cost: three driver runs `total_cost_usd` 0.0315, 0.0327, 0.0327 (sum 0.0969); 35 guard lines in `hook-log.jsonl`, all from this campaign.

## Items summary

| Item | Result |
|---|---|
| install on throwaway root | PASS (rc 0) |
| Batch 1: 50-item at-cap batch ingests, no void | PASS |
| Largest page tokens <= 22,500 | PASS on estimate only (~19,460, proportional; not directly measurable, parallel Reads; F-1) |
| First spawn foreground shape | PASS (`completed` / `foreground`) |
| `SubagentHandback` appears / hooked | Not present in either batch; hooked: unobservable |
| `hook_success.command` == `guardCommand` | PASS, verbatim, unexpanded |
| `$CLAUDE_PROJECT_DIR` expands in guard command | PASS, guard ran; no `hookMissing` |
| Extra `Agent` input keys | None (`description`, `prompt`, `subagent_type`) |
| Clean 50-item turn count | Subagent 4 turns, driver 2 (`num_turns`); one run each |
| `timeout: 30` accepted | PASS (hook ran, empty `.err`) |
| Batch 2 `maxTurns: 2` cut-short shape + terminal entry | Recorded (above) |
| Cost/tokens | Batch 1 $0.4966 list-basis; batch 2 $0.0969 over 3 attempts |

## Findings
- **F-1 (minor, spec/testability):** the plan's page-token method (usage delta around each Read) does not hold for a labeler that issues the page Reads in parallel, which both batches did. Only the aggregate (164,763 cache-write tokens for 9 pages) is observable; per-page figure must be estimated from result size or the method replaced (e.g. measure with a one-page-per-turn probe, or assert only that no Read errored at the 25,000-token cap, as observed). No Read hit the cap.
- **F-2 (info):** with `maxTurns: 2` one `driver run` auto-retried the batch twice more (3 spawns) before dropping it as `retryCapExceeded`; nothing to fix, but the smoke's spend is the retry count, not one batch.
- **F-3 (info):** with `maxTurns: 14` the labeler used 4 turns because it parallelizes; a serial reader would need ~12 (header + 10 pages + Write + reply), so `P_max + 4` margin should be judged against the serial case, not 4.
- No defects found. `SubagentHandback` guard allow-branch remains unexercised (U).

Not run: no repeat batches; the repo suite was not re-run (no source touched). Unrelated working-tree changes (`docs/plans/connected-code-graph*.md`) are not mine.
