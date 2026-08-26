# QA testing techniques — this lab's environment

> **On-demand knowledge base for `qa-engineer`.** Environment/tooling techniques discovered
> driving real black-box QA passes in this lab, not test-plan content — those live in the
> component's own `docs/test-plans/`. Consult when a pass needs one of these mechanics.
>
> Origin: distilled 2026-08-11 from the `qa-engineer` agent's learnings inbox via
> `agent-maintenance` skill §5.

## This WSL2 box has no native browser-automation stack — Windows Chrome + raw CDP over the mirrored-network localhost is the fallback for visual/interactive checks

`playwright`/`selenium` aren't installed for the WSL Python, and there's no Linux-side Node
(`node`/`npx` resolve only to the Windows install). Working path: launch Windows Chrome headless
with remote debugging (`"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless
--disable-gpu --remote-debugging-port=<port> --user-data-dir="C:\Windows\Temp\<profile>" <url>`),
enumerate targets from WSL with `curl http://localhost:<port>/json` (mirrored networking makes the
port visible cross-side), then drive it with a small script run via the **Windows** `node.exe` (not
WSL Node, which doesn't exist) that opens the page's `webSocketDebuggerUrl` and sends
`Runtime.evaluate`/`Page.captureScreenshot` CDP commands. Output must be written to a native Windows
path (`C:\Windows\Temp\...`) — a WSL-side path like `/tmp/...` passed to the Windows `node.exe`'s
`fs.writeFileSync` does **not** resolve — then `cp /mnt/c/Windows/Temp/<file> <wsl-path>` to pull it
back. This is what lets a genuinely interactive check (click a toggle, screenshot the revealed
panel) succeed, where `chrome.exe --headless --screenshot=...` alone only captures the initial page.

## `tmux` — not `expect`, not raw stdin piping — reliably drives a genuinely interactive TUI for black-box QA

Piping input (`printf '/cmd\n' | some-tui ...`) doesn't reach a TUI that needs real TTY/raw-mode
behavior — output shows only the startup banner, no command response. `tmux new-session -d ...` +
`tmux send-keys ...` + `tmux capture-pane -p` works cleanly for both slash commands and literal
`@`-containing free text, and reliably captures rendered pane content as evidence. `expect` was
unavailable in this environment; `tmux`/`script` were present.

## A CLI's "doctor"/health-check subcommand is not guaranteed read-only — verify before running one reflexively as a pure status probe

`kiro-cli doctor`, run expecting a pure `Auth ✔`-style status check, auto-remediated shell
integration by **appending** a sourcing block to `~/.bashrc` and `~/.profile` (idempotent, guarded
by an `if [[ -f ... ]]` line, but still an unannounced environment mutation outside any repo). A
command's prior framing as a status check doesn't guarantee side-effect-freedom — check what a
"doctor"/"check"/"diagnose" subcommand actually does (read its `--help`, or its source if
available) before running it reflexively during environment probing.

## A clean static/diff gate on an agent's prompt wording is necessary, never sufficient, when the acceptance criteria are prompt-compliance claims about the agent's own output

Twice-verified prompt wording (plan gate + diff gate, both `analyst`, both zero findings) still
failed to manifest correctly across all 3 live subagent dispatches sampled during the
`cpg-agent-adoption` M4 U6 acceptance pass — and in three *different* ways at once: `coder`
reasoned correctly but emitted loose prose ("**CPG freshness note:**") instead of the mandated
literal `CPG: <shape> — <clause>` line (a `grep "CPG:"` spot-check false-negatives despite correct
underlying behavior); `architect` used the CPG but explicitly declined the paired "mandatory, not
optional" freshness check, substituting its own weaker cross-check reasoning; `tdd-engineer`,
dispatched against a component with no loaded CPG, produced zero mention of "CPG" in any form —
indistinguishable from the check never having run (`docs/test-reports/cpg-agent-adoption-report.md`
DEF-1/DEF-2/DEF-3). When a feature's acceptance criteria are claims about what an agent's own
output must contain or do (a required line, a mandated sub-step), treat a clean static gate on the
wording as necessary but never sufficient — budget live-dispatch sampling as its own required test
layer, and expect independent, differently-shaped failure modes per dispatch rather than one shared
root cause: a passing dispatch does not predict the next agent's compliance, even with
near-identical prompt wording.

A related trap on the *re*-pass: when re-dispatching narrowly to "confirm these N named defects are
closed," still read each re-dispatch's **full** output against the whole convention, not just the
specific clause each defect targeted. The M4 U9 re-pass confirmed DEF-1/DEF-2/DEF-3 closed cleanly
— but DEF-3's fix (emit *some* `CPG:` line even when the check was skipped) exposed a **new**,
previously-masked defect: the emitted line picked the wrong one of the convention's three defined
shapes (`CPG: not applicable —...` where the correct shape for "component has code but no loaded
CPG" is `CPG: considered, not relevant —...`). Pass 1 never surfaced this because Pass 1's failure
mode (zero `CPG:` line at all) left nothing to shape-check. Fixing one defect can newly expose a
decision point the original failure had been hiding entirely — a "confirm closure" re-pass is a
fresh opportunity to find a new defect, not a narrower check than the original
(`docs/test-reports/cpg-agent-adoption-report.md` Pass 2, DEF-4).

## A live `pytest -m live` suite with many sequential local-LLM calls should be launched expecting >120s wall-clock

`falkor-chat`'s K-026 eval harness (`server/tests/eval/test_judge_live.py`) issues ~50 sequential
`.complete()` calls (generation + judge-of-generation + calibration-judge) against a local model via
LM Studio; real wall-clock was `175.92s`, past Bash's default 120s foreground timeout — the call
auto-moved to background rather than failing. A live-marked suite doing many sequential local-LLM
round trips should be launched with `run_in_background` proactively, or the runner should be ready
to hand off to `Monitor`/an until-loop, rather than assuming a single foreground `Bash` call will
return in time.

## Verifying an MCP server/tool rename or reconfig from inside the same session that made the edit, before any restart: use the `claude mcp` CLI surface, not the session's own tool binding

A session's own interactive MCP tool call (e.g. `mcp__<name>__query`) is a stdio connection
established at session start — it keeps resolving through the *old* binding even after
`.mcp.json`/`.claude/settings.json` are edited underneath it (confirmed structurally stale: it kept
answering after the on-disk launch script it would need to re-spawn from had been relocated
entirely). `claude mcp list` and `claude mcp get <name>`, run via `Bash`, are by contrast **fresh
CLI processes** independent of that binding — they read the live `.mcp.json` from scratch and
reflect the current config immediately, no restart required
(`docs/test-reports/cpg-mcp-rename-report.md` TP-003/TP-006). Don't default to reporting such a
check as session-blocked-pending-restart; try the `claude mcp` CLI surface first. A raw
protocol-level probe (piping JSON-RPC at the server's launch script directly, bypassing the Claude
Code MCP client) is the complementary technique for verifying the *server's own* live behavior
(tool list, self-description text, actual query results) under the same no-restart constraint.

## A first-attempt `mcp__cypher__query` write blocked by Claude Code's Auto-Mode classifier is a harness false start, not a defect signal by itself — retry once

During a live acceptance pass a well-formed, in-shape author-write `CREATE` (correct `KaizenEntry`
shape, `author` matching the declared `agent`) was denied with "Blocked by classifier" before the
call ever reached FalkorDB; the byte-identical call succeeded immediately on retry, with no change
to the query or the environment. A single blocked write attempt during a live pass is therefore not
by itself evidence of a real authorization defect in the tool under test — retry once before
concluding the write path is broken (`docs/test-plans/generic-cypher-mcp2.md` TP-004/AC-4).

## Proving a *manual* sweep resumed a run (not the automatic periodic one): fire the manual call immediately after the due moment, and read its own response for `runId` in `resumed`

When a feature runs both a manual on-demand endpoint and an automatic in-process periodic task
over the same idempotent sweep logic (falkor-chat K-028's `POST /workflow-runs/due` vs. the
`asyncio` periodic sweep task), a naive black-box attempt to exercise the *manual* path — sleep
past the due time, then call the endpoint — usually loses the race: the automatic tick fires first
and the manual call reports `checked:0, resumed:[]`, having correctly found nothing left to do. A
run's own final state afterward is ambiguous about which sweep actually resumed it. To reliably
attribute the resume to the manual call, fire it immediately after the due moment elapses — right
after your own sleep, before the next automatic tick — and read `runId` out of *that call's own*
`resumed` list, not just the run's eventual state (`docs/test-reports/…` K-028 TP-004: a
loosely-timed first attempt showed `checked:0`; a tightly-timed retry, sleep 3.2s then immediate
manual sweep, captured `resumed:[{runId,...}]` directly in the manual response).

## Reading a long `mcp__cypher__query` text field exactly: page it one column at a time, never several long chunks in one row

A multi-column chunking query (`RETURN substring(field, 0, 260) AS p0, substring(field, 260, 260)
AS p1, ...`) can render corrupted/duplicated text in the chat transcript even when the underlying
data is provably fine (confirmed via independent `size()`/`substring()` checks on the raw field) —
see `cypher-mcp/README.md`'s "Result format and truncation" section for the full writeup and the
safe single-column recipe. Relevant to any acceptance pass that needs to read back a long stored
value (a `KaizenEntry` field, a long property) verbatim rather than just spot-check it.
