# K-030 Track 2 Stage 7 — retrieval-convention skill review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (Stage 7)

**CPG:** considered, not relevant — no `cpg_claude` graph is loaded (`GRAPHS` shows
`cpg_falkorchat` plus the workspace/`kaizen_team` graphs only), and this dispatch's deliverable
(a markdown skill, a Bash grep check, and one-line prompt edits) is not code a CPG would add
leverage over — a direct read of every touched file plus live execution of the checks was the
stronger method here.

## Scope & verdict

Diff-scoped review of K-030 Track 2 Stage 7 (`claude/docs/plans/agent-knowledge-base-strategy.md`
§3/§4.4/§7, closing item (b)) as delivered in commits `b907893`/`9b58ccf`: the new
`skills/agent-kb-retrieval/SKILL.md`, `claude/scripts/audit-team.sh` check 11, and the one-line
pointer added to 9 agents' prompts. Verified by execution — ran `audit-team.sh` myself,
mutation-tested check 11 myself (corrupt → FAIL, byte-identical restore → PASS, `diff` confirmed),
queried `ws:agent-team` directly for the title-convention and attribution claims, and traced each
of the 9 consuming agents' frontmatter `tools:` field against what the new pointer asks them to
call. Did not re-derive the Stage 6 corpus tally or re-review Stage 6's own findings — that's
`agent-knowledge-base-strategy4-stage6.md`'s scope, already closed.

**Verdict: needs changes.** One blocker: the dispatch added a working-convention pointer to 9
agents' prompts but only gave the underlying MCP tools to 5 of them — the other 4 (including
Stage 8's own `data-scientist` owner) cannot actually call `search_documents`/`get_document` yet,
which is silent-inert exactly the way this team's own `agent-standards` skill warns an MCP feature
goes wrong. **This should block Stage 8**, or at minimum be fixed before Stage 8's `data-scientist`
unit (U5) starts, since U5 needs to call `search_documents` itself to design and pilot the
golden-set evaluation. Everything else checked (prefix byte-exactness, score-floor discipline,
drift-check correctness, pointer placement/wording, Step 0 reasoning, the title-convention claim)
holds and needs no rework.

## Findings

### Blocker — the retrieval pointer is silently inert for 4 of the 9 consuming agents: their `tools:` allowlist never gained the new MCP read tools

**Evidence.** `skills/agent-kb-retrieval/SKILL.md` names its `allowed-tools:` as
`mcp__falkor-chat-agent-team__search_documents, mcp__falkor-chat-agent-team__get_document`, and
its own description lists 9 consumers: `teco`, `architect`, `tdd-engineer`, `frontend-engineer`,
`qa-engineer`, `analyst`, `data-scientist`, `graph-dba`, `devops`. Checking each of the 9 agents'
frontmatter `tools:` field directly:

- **No explicit `tools:` (inherits everything — fine):** `tdd-engineer`, `frontend-engineer`,
  `qa-engineer`, `graph-dba`, `devops`.
- **Explicit `tools:` allowlist, missing both new MCP tools:** `teco` (`tools: Read, Bash, Agent,
  SendMessage, AskUserQuestion, Write, Edit, mcp__cypher__query,
  mcp__falkor-chat-agent-team__ingest_document` — has the *write* tool from the Track 1 pilot, but
  not the two read tools), `architect`, `analyst`, `data-scientist` (all three: `Read, Grep, Glob,
  Bash, Write, Edit, WebFetch, WebSearch, Agent, mcp__cypher__query` — neither read tool present).

This team's own knowledge base states the exact mechanism directly, verified 2026-07-25
(`skills/agent-standards/claude-code.md:1023-1026`): *"A subagent's `tools:` is an allowlist, and
MCP tools are subject to it. An agent that declares `tools:` sees **no** MCP tool that isn't named
there — the single easiest way to ship an MCP feature that is silently inert for some agents...
A skill's `allowed-tools` pre-approves the listed tools for the turn that invokes the skill; it
does not gate them, and it does not grant a tool the session doesn't have."* `claude/README.md`
already documents this exact pattern for `cpg-analysis` — e.g. architect's/analyst's catalog rows
both say "hence the `mcp__cypher__query` entry in this agent's `tools:` allowlist, without which
the tool would be invisible to it" — so the team has hit this failure mode before and knows the
fix; it just wasn't applied to this dispatch's 4 restricted-allowlist agents.

**Why it matters.** `architect`, `analyst`, `data-scientist`, and `teco` are all named consumers
with a fresh pointer line in their own prompt telling them to call `search_documents`/
`get_document` — and none of the four can. This isn't hypothetical for Stage 8: the coordination
ledger names `data-scientist` as U5's owner ("Stage 8 golden-set design + pilot calibration"),
and the parent plan's §7/§8 test strategy requires that unit to actually run
`search_documents` calls against a stratified query set to produce the pilot's first calibrated
score floor. As delivered, `data-scientist` would hit this wall the moment it tries.

**Suggested fix.** Add `mcp__falkor-chat-agent-team__search_documents,
mcp__falkor-chat-agent-team__get_document` to the four restricted `tools:` lines (`teco.md`,
`architect.md`, `analyst.md`, `data-scientist.md`) — a one-line frontmatter edit per file, the same
shape `teco.md` already carries for `ingest_document`. This is small enough for `teco` to fix
directly (or route to `cobb`, who owns these agent-definition files) rather than a new dispatch
unit. No test suite exercises Claude Code's own tool-visibility behavior (it can't — this is a
harness runtime property, not something a script or CPG query observes), so the closing check here
is a live one: after the frontmatter edit, have `data-scientist` (or `architect`/`analyst`) report
its own available tools and confirm `search_documents`/`get_document` are now listed, the same
live-probe method `agent-standards` itself used to verify this failure mode in the first place
(`claude-code.md:280`, a comparable "ask the delegate what tools it has" check).

## What's solid

- **Prefix template is byte-exact.** The plan's §4.4 two-literal quote
  (`f"Instruct: Given a coding agent's description of its current situation, retrieve the "` +
  `f"distilled technique or rule that applies to it.\nQuery: {situation}"`) concatenates to exactly
  the same string as `SKILL.md`'s single fenced literal and `audit-team.sh`'s grep pattern
  (`kb_prefix=...`) — compared directly, character by character, not by eye.
- **Score floor is genuinely provisional, not a dressed-up guess.** `SKILL.md` states "No score
  floor is applied client-side at this stage," instructs callers to use whatever `search_documents`
  returns as-is, and explicitly flags itself as stale if read after Stage 8 lands a real number —
  matches the plan's §4.4/§8 "ship with the floor disabled" instruction exactly, no invented number
  anywhere.
- **Check 11 mutation-tested independently and confirmed correct**, not taken on `cobb`'s word:
  corrupting the fenced string → FAIL with the expected message; restoring from a backup → `diff`
  byte-identical, check → PASS again. Also independently ran the full `audit-team.sh` and got
  exactly the same 5 pre-existing FAILs `cobb` reported (all in check 7, personal-info leaks in
  `docs/reviews/bypass-permissions-subagent-gap.md`, `model-bench/packs/tool-caller-shop-assistant/
  conversations.jsonl`, `opencode/agents/tank/opencode.json`, `opencode/docs/test-reports/
  devops-opencode-headless-report.md` — none touched by this dispatch).
- **All 9 pointer lines present exactly once**, placed consistently right after each agent's
  existing on-demand-KB section, wording near-identical across all 9. `graph-dba`'s markdown-link
  variant (`[skills/agent-kb-retrieval/SKILL.md](../../skills/agent-kb-retrieval/SKILL.md)`)
  resolves correctly (`claude/graph-dba/` → `../../` → repo root → the real file) and matches that
  same file's own pre-existing link style (line 44 already links `joern-cpg` the same way) — not a
  broken path, just a file-local convention difference from the other 8's plain backticks.
  `devops.md`'s previously-flagged mid-sentence insertion reads clean on inspection.
- **The "family-slug — claim-title" convention holds against live data** — queried `ws:agent-team`
  directly (`MATCH (d:Document) WHERE d.title CONTAINS ' — ' RETURN d.title`) and found it exactly
  as described, e.g. "Cypher on FalkorDB — write idiomatic Cypher within the supported surface, and
  verify before relying on it."
- **Attribution is real, not uniform** — `MATCH (a)-[:INGESTED_BY]-(d:Document) RETURN a.agentId,
  count(d)` returns `cobb: 333, teco: 1`, confirming `produced_by` differentiation actually landed
  in the graph, not just in the design.
- **Step 0 reasoning is sound, not just asserted.** New `skills/` package: correctly argued as the
  only shape that avoids making 8 of 9 consumers point sideways at a peer's own KB file, and
  correctly checked against the real precedent (every existing `skills/` package is markdown-only).
  `audit-team.sh` for the drift check: correctly argued on "gets run every time the team is
  certified" versus a standalone script nobody remembers to invoke — a real, checkable trade-off,
  not hand-waved.

## Open questions

**Item 7 (claude/README.md inline mention) — my answer: correctly out of scope, as delivered, with
one caveat tied to the blocker above.** The `cpg-analysis`/`python-web-quirks` inline mentions in
`claude/README.md` earn their place because they carry *differentiating, conditional* information
per agent — whether a CPG exists for the task, whether the codebase is Python-web-shaped, and
(materially) whether the agent's own `tools:` allowlist needed a new entry to see the MCP tool at
all ("hence the `mcp__cypher__query` entry..."). `agent-kb-retrieval` applies identically and
unconditionally to all 9 consumers — there is no per-agent variance to report, so a mention on each
of the 9 rows would be nine copies of the same clause with zero differentiation value, which is
exactly the over-enumeration `claude/AGENTS.md`'s own K-032 backlog item is trying to reduce, not
add more of. **The caveat:** once the blocker above is fixed, 4 of those 9 rows (`teco`,
`architect`, `analyst`, `data-scientist`) *will* have gained a new `tools:` entry to make an MCP
tool visible — at that point they meet the exact "hence the X entry..." precedent `cpg-analysis`
already sets, and a one-clause addition to those 4 rows (not all 9) would be consistent with
existing practice. Not blocking on its own; worth a follow-up note once the blocker is fixed.

Everything else asked in the brief (items 1-6, 8) is resolved above under Findings/What's solid —
no further open questions.
