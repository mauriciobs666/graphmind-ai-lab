# The Claude Agent Team manual: behavioral check report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** —

## Summary

Lightweight black-box check of `claude/docs/manuals/agent-team.md` (tico's rewrite, uncommitted
working-tree version, md5 `d85d48d3…`, on HEAD `df69acc8`, run 2026-10-20). Scope, set by the
caller: (1) every command or path the manual tells a reader to run or open, and (2) whether every
` ```mermaid ` block parses. The caller waived a separate test plan, so the check IDs below
(`TR-NN`) stand in for plan items. No `claude --agent …` sessions were launched, and no
service, graph or repo file was changed apart from this report.

**Verdict: pass with one low-severity defect** (TR-04: the paper-trail `find` one-liner
over-matches across slugs). All 10 Mermaid blocks parse with the real renderer.

CPG: not applicable — the target is an end-user manual (prose, shell one-liners, Mermaid). There is no code under test.

## Results

| ID | Check | Result | Evidence |
|---|---|---|---|
| TR-01 | `grep -m1 -H 'Status:' docs/plans/*.md` run inside `falkor-chat/` (manual §"Document status") | pass | rc=0. Printed 65 `path:> **Status:** <token> …` lines for 65 files in `docs/plans/`, so every plan was reported. |
| TR-02 | `find . -path '*docs*' -name 'x*'` with a real slug, `server-logging-config`, inside `falkor-chat/` (§"One name per feature") | pass | Returned exactly `./docs/plans/server-logging-config-coordination.md` and `./docs/reviews/server-logging-config.md`. These are the only two files in `falkor-chat/docs` that mention the slug (`grep -rl`). 0.03 s. Also correct when run from the repo root. |
| TR-03 | Same `find`: does it sweep `.venv`/`node_modules` noise? | pass, with a caveat | `falkor-chat/server/.venv` exists, but only 4 of its paths contain `docs` (e.g. `…/fastapi/openapi/docs.py`), so a real feature slug picks up no noise. Inside a component that has `node_modules`, a short prefix does pick up noise: `salesperson/` has 150 matching paths under `node_modules/*/docs/` (e.g. `./node_modules/react-router/docs/how-to/resource-routes.md` for `-name 'r*'`). This is low risk because `salesperson/` keeps its docs in `falkor-chat/` and the manual says to run the command in a component's own tree. |
| TR-04 | Same `find`: does it show *only* the feature's paper trail? | **fail (low)** | `-name 'x*'` is a bare prefix, so it also matches different topics whose slug starts with `x`. Inside `falkor-chat/`, `-name 'agent-team*'` returns `./docs/plans/agent-team-ingestion-graph.md` and `./docs/reviews/agent-team-ingestion-produced-by.md`. Neither belongs to an `agent-team` feature. See D-1. |
| TR-05 | `claude/scripts/audit-team.sh` exists and runs read-only (§"How the team gets better") | pass | The script exists and is executable (`-rwxr-xr-x`). Its header says "Read-only", and a grep for `rm`/`mv`/`sed -i`/`tee`/`git add`/`commit`/`docker`/`redis` and for output redirections found no writes. Run with `git status --porcelain` captured before and after: the two outputs were identical. Exit code was 1 (`RESULT: FAIL`). The cause is 5 failures in check 7, the personal-info leak check, triggered by files outside the manual, such as `claude/docs/plans/bypass-permissions-subagent-gap.md` and `opencode/agents/tank/opencode.json`. The manual itself contains none of the leaked identifiers (`grep` count 0). This is not a defect in the manual. It is noted under Feedback. |
| TR-06 | Cited paths exist | pass | 13 agent folders. `claude/<name>/kaizen/plan.md` and `history.md` both exist 13/13, which matches "thirteen agents". `claude/docs/requirements/` exists (7 docs). `claude/` has no `docs/BACKLOG.md` (matches "no backlog of its own"). Root `docs/BACKLOG.md` and `docs/HISTORY.md` exist, as do `falkor-chat/docs/BACKLOG.md`, `falkor-chat/docs/HISTORY.md` and `falkor-chat/docs/plans/*-coordination.md`. |
| TR-07 | MCP server name in the FAQ (`/mcp` → reconnect `falkor-chat-agent-team`) | pass | The repo-root `.mcp.json` defines `"falkor-chat-agent-team"`. The `/mcp` reconnect itself was not exercised (it is interactive and out of scope). |
| TR-08 | All 10 Mermaid blocks parse (manual lines 27, 74, 112, 146, 165, 206, 241, 270, 393, 405) | pass | Each block was extracted and rendered with `npx -y @mermaid-js/mermaid-cli@11.4.2` (Windows `npx` via `cmd.exe`, with the input on a `\\wsl.localhost` UNC path). All 10 exited with rc=0 and produced a 13–38 KB SVG. None contains `Syntax error in text`. Negative control: a deliberately broken flowchart exited with rc=1 and `Error: Parse error on line 2`, so the rc=0 results are meaningful. Risky constructs (`<br/>`, `<i>`, `<b>` in labels, `·`/`→`/`⏸`/em-dash characters, the `state "Ready for design" as ReadyForDesign` alias, the `[("…")]` cylinder, dotted labelled edges) all rendered. |
| TR-09 | Mermaid label cosmetics | observation | In the unit-state diagram (line 206) the state renders literally as `in_flight`, with an underscore, because the state ID is the label. The ledger table above it spells it `in-flight`. Readable, but inconsistent. See Feedback. |

## Defects

**D-1: the paper-trail `find` one-liner over-matches into other topics** (severity: low; TR-04)

- *Steps:* `cd falkor-chat && find . -path '*docs*' -name 'agent-team*'`
- *Expected (manual line 158):* "shows you a feature's entire paper trail", meaning only that slug's family.
- *Actual:* it also lists `agent-team-ingestion-graph.md` and `agent-team-ingestion-produced-by.md`,
  which belong to a different topic. Any slug that is a prefix of another slug gets the same mix. In a
  component with `node_modules`, a short slug also matches package docs (TR-03).
- *Suggested fix:* either soften the claim ("lists every document whose name starts with the
  slug") or give an exact family match. This version was verified to return only the family (two
  files for `server-logging-config`, none for `agent-team`):
  `find docs -regextype posix-extended -regex '.*/x(-(coordination|ml|graph|rca|impl|report))?\.md'`.
  A middle ground is `find docs -name 'x.md' -o -name 'x-*.md'`. It searches `docs/` only, so there is
  no `.venv`/`node_modules` noise, but it still over-matches `x-<longer-slug>`.

## Coverage and gaps

- Covered: every runnable command and every cited repo path in the manual, plus parsing of all
  Mermaid blocks with the real renderer.
- Not covered, by design: the walkthroughs that need live agent sessions (`claude --agent tico|teco`)
  and the approval-prompt table (hook behaviour). The manual's factual and architectural claims
  are `analyst`'s half and were not judged here. That includes who commits what, the ledger
  thresholds, the status-flip ownership and the hook fences. Visual layout was not inspected
  beyond confirming that each block rendered.

## Feedback and recommendations

1. Apply D-1's fix (tico).
2. TR-09: optionally give the state a label (`state "in flight" as in_flight`, as line 272
   already does for `Ready for design`) so the diagram matches the ledger's `in-flight`.
3. Not the manual's concern: `audit-team.sh` currently exits with FAIL on the repo-wide
   personal-info leak check (check 7, 5 identifier classes, pre-existing files). A reader who
   follows the manual's mention and runs it will see `RESULT: FAIL`. Route this to `cobb` for
   cleanup.
4. Environment note: no Linux `node` is installed. Mermaid validation works through the Windows
   `node`/`npx` (v24.14.1) with UNC paths, using an already-cached puppeteer Chrome.
