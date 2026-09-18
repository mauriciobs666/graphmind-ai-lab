# Agent-team write-convention pilot (K-030 Track 1 Stage 4) — diff review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

**Scope.** `git diff -- claude/AGENTS.md claude/README.md claude/cobb/cobb.md claude/teco/teco.md claude/cobb/kaizen/history.md claude/cobb/kaizen/plan.md`, all uncommitted in the working tree at review time. This is `cobb`'s Stage 4 unit of `claude/docs/plans/agent-knowledge-base-strategy.md` (Track 1): repointing the raw-capture "Learning capture" write convention from `kaizen_team` (`mcp__cypher__query`) to `ingest_document` against the new `ws:agent-team` falkor-chat workspace, piloted with `cobb`/`teco` only. Not in scope: the `ingest_document`/`produced_by` server-side implementation itself (Stage 1, already reviewed/approved — `falkor-chat/docs/reviews/agent-team-ingestion-produced-by.md`), and whether team-wide cutover should happen now (a coordination-level call already made).

**CPG:** considered, not relevant — the diff itself is prose-only (`claude/` agent prompts and docs), and the one place it touches real server behavior (`ingest_document`/`delete_document`'s call shape) was verified by reading `falkor-chat/server/falkorchat/{mcp,services}.py` directly, since `cpg_falkorchat` (confirmed loaded, `GRAPHS`) predates the `produced_by` extension this pilot depends on — the parent plan's own CPG note makes the same call for the same reason.

**Verdict: approve with suggestions.** The operative change — the new `ingest_document` calling convention in `cobb.md`/`teco.md`, `AGENTS.md`'s self-description as a summary (not the mechanism), and `teco.md`'s `tools:` addition — is accurate, complete, and independently verifiable against the real `falkor-chat` MCP surface. One convention violation in `claude/cobb/kaizen/plan.md` (a stale claim left standing beside new, contradicting text) should be fixed before this is treated as closed, but it does not undermine the delivered pilot itself.

## Findings

### Major — `plan.md`'s K-030 item is appended to, not rewritten, leaving a now-false claim standing next to the text that falsifies it

The diff adds a new bullet, "**Track 1 (Stages 1-4) since delivered, 2026-09-18**," reporting that Stages 1-4 (including this pilot) are dispatched and delivered. It leaves two older passages in the same item completely unchanged, both now false:

- The item's own `Status:` line (`claude/cobb/kaizen/plan.md:394-397`): "...the substrate design for the rest of the plan...is no longer blocked — a full design pass completed this session (below) — **but no implementation has been dispatched yet**."
- A bullet a few lines below it (`plan.md:432-438`): "**Not resolved by this: no implementation has been dispatched — this was design work only.** Concretely still open before Track 1 Stage 1 can start: a `graph-dba` design note...and the `devops` work to stand up the dedicated `ws:agent-team` falkor-chat process."

Both of these are now contradicted by the new bullet 7-40 lines later in the same item, which reports Stage 1's `produced_by` extension merged, Stage 2's workspace bootstrapped, and Stage 3's dedicated process live — i.e. exactly the two "still open" items this older bullet names. A reader scanning the item top-to-bottom (which is how a `BACKLOG`/plan-style item is meant to be read whole, per root `AGENTS.md`'s living-document convention) hits the false claim first and the true one second, with nothing telling them which one is current.

This is precisely the anti-pattern root `AGENTS.md` names explicitly: "**An open item is rewritten, not appended to**... What an update supersedes is owed nothing: a stale guess at what remained was never acted on...verify that, then drop it." The item's `Notes:` line at the very bottom (`plan.md:455-458`) *was* correctly updated to a one-dated-line form ("Track 1 Stages 1-4 delivered and logged here 2026-09-18") — so the convention was followed in one place in this same diff and not in the two others.

**Suggested fix:** rewrite the `Status:` line to state the current truth (Track 1 Stages 1-4 delivered, Stage 5 + team-wide cutover + Track 2 still open) instead of "no implementation has been dispatched yet," and either delete the now-superseded "Not resolved by this" bullet outright (its two named blockers are both resolved) or rewrite it to name only what's still actually open before Track 2 Stage 7 (the `skills/agent-kb-retrieval/SKILL.md` placement confirmation, which — checked against `plan.md:435-438` — is still genuinely open and worth keeping).

### Minor — `history.md`'s Stage 4 entry undercounts `audit-team.sh`'s pre-existing FAIL count

`claude/cobb/kaizen/history.md`'s new entry states: "`claude/scripts/audit-team.sh` re-run after all edits: **2** pre-existing FAILs (personal-identifier leaks) traced to unrelated `model-bench/` data-pack files." A direct re-run (`bash claude/scripts/audit-team.sh`) shows **5** FAIL lines (git user.name, git user.email, hostname, username, home-path leaks), all in files unrelated to this diff (`model-bench/`, `opencode/agents/tank/opencode.json`, `docs/plans/salesperson-ui.md`, `docs/reviews/cpg-provenance-stamp.md`) — matching the brief's own independently-verified "all 5 FAIL categories are pre-existing." The substantive conclusion (pre-existing, unrelated to the 6 touched files) is correct and independently confirmed twice over (`teco`'s brief, my own re-run); only the reported count is off.

**Suggested fix:** correct "2 pre-existing FAILs" to "5" in the `history.md` entry — a one-word fix, not worth a re-review.

## What's solid

- The new `ingest_document` calling convention in `cobb.md`/`teco.md` matches the real MCP tool signature exactly: `mcp.py:287-289` (`text`, `title`, `source_format`, `source_label`, `produced_by`, all keyword-capable, none positional-only) and `services.py:1154-1232`. The `AgentNotFoundError` claim (`history.md`'s "raised `AgentNotFoundError`...per `services.py:1225`'s `raise AgentNotFoundError(produced_by)`") checks out verbatim at that line.
- `AGENTS.md`'s rewrite correctly demotes itself to a *summary* of the real mechanism ("each agent's own `<name>.md` carries the actual operative...write instruction; this bullet is the summary...not the mechanism itself") — an accurate description of the scope correction `cobb` made mid-run, and consistent with `README.md`'s own pre-existing framing ("a 'Learning capture' closing protocol in every prompt").
- The "11 of 13 agents untouched" framing is arithmetically and factually correct: `grep -rl '^## Learning capture' claude/*/*.md` returns exactly 13 files, and only `cobb.md`/`teco.md` were rewritten this diff.
- `teco.md`'s `tools:` frontmatter gained exactly one entry, `mcp__falkor-chat-agent-team__ingest_document` — confirmed the only frontmatter change (`permissionMode`/`hooks` byte-identical in the diff) — and it is both necessary (the new prompt text calls that tool) and sufficient (`teco`'s new text never calls `list_documents`/`get_document`/`delete_document`, unlike `cobb`'s, which needs no allowlist entry since `cobb.md` declares none and inherits every tool — confirmed by reading its frontmatter).
- `AGENTS.md`'s "Distillation" bullet clause ("`ws:agent-team`'s equivalent read/clear step...is Stage 5's own, not yet wired into the skill") matches the parent plan's Stage 5 row and §5's Track 1 bullet almost verbatim (`agent-knowledge-base-strategy.md:264`, `:507-516`) — not merely plausible-sounding, actually the same scope.
- `history.md`'s new entry is a single dated, non-stacked entry (correct for a lookup-kind document); the scope-correction narrative (discovering the mechanism lives in each agent's own file, not `AGENTS.md` alone) is accurately described and matches what the diff actually contains.
- `.mcp.json`'s `falkor-chat-agent-team` entry (`streamable-http`, port 8200) is in place, giving the `mcp__falkor-chat-agent-team__ingest_document` tool name in `teco.md`'s frontmatter a real backing server to resolve against.

## Open questions

None — the one substantive finding (plan.md staleness) has a concrete, low-risk fix named above; no decision needed from the caller.
