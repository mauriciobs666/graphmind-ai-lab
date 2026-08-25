# Kaizen — Improvement Plan: cobb

> Forward-looking backlog for the `cobb` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-08-21

## Active

| ID | Added | Priority | Status | Summary |
|-------|------------|----------|--------|---------|
| K-001 | 2026-05-31 | high | 🔵 | Periodically re-verify the documented standards against live official docs (these ecosystems change fast). |
| K-002 | 2026-05-31 | medium | 🔵 | Add a worked "port an agent across tools" reference example (Claude subagent ↔ OpenCode agent ↔ Kiro steering) — now skill material, candidate for the `agent-maintenance` bundle. |
| K-003 | 2026-05-31 | low | 🔵 | Track additional agentic tools as they mature (e.g. Codex CLI, Cursor, Gemini CLI) where they share the open AGENTS.md / Agent Skills standards. |
| K-005 | 2026-06-07 | high | 🔵 | Automate doc-drift detection: a scheduled routine that re-fetches the canonical docs, diffs vs. stored snapshots, and files a kaizen item on change. |
| K-008 | 2026-06-07 | low | 🔵 | Dog-food the frontmatter cobb teaches: evaluate adding `memory: project` for a persistent cross-session drift/verified-date store (distinct from kaizen). |
| K-014 | 2026-07-25 | medium | 🔵 | Skills outside cobb's two have **no kaizen home**: `skills/` carries no `kaizen/` dirs and `skills/README.md` routes only `agent-maintenance`/`agent-standards` changes to cobb's history — so edits to `cpg-analysis`, `joern-cpg`, etc. land with no per-artifact log. Three skills now cover three different "owner" shapes — cobb's own machinery (`agent-maintenance`/`agent-standards`), `graph-dba`-driven (`cpg-analysis`, `joern-cpg`), and nobody's (`python-web-quirks`) — and **all three landed in an owning agent's history** by by-example precedent. That is converging evidence that "owner-agent's history" already *is* the convention; the work is writing it into `skills/README.md` + `claude/AGENTS.md` where a new skill author would find it, not deciding it. |
| K-009 | 2026-06-20 | medium | 🔵 | Add a CI/script guard that every component `AGENTS.md` has a sibling `CLAUDE.md` = `@AGENTS.md` stub (so Claude Code never silently misses context — it reads `CLAUDE.md`, not `AGENTS.md`). Fold into the K-005 drift job. *(Sibling shipped 2026-07-09: `claude/scripts/audit-team.sh` covers the agent-collection invariants — the `@AGENTS.md`-stub check could join it.)* |
| K-015 | 2026-07-31 | medium | 🔵 | `analyst/kaizen/inbox.md` has a substantial backlog of already-verified, "suggested home: prompt" entries never distilled (stub-package HEAD-vs-working-tree import, review-safe pytest subset, isolatable snapshot side, byte-identity AST hash, line-number-invariance re-gate, exclude_unset nested-model gotcha, scratch-copy-reverse-patch). Run a full §5 pass: verify each still holds, promote the prompt-worthy ones into `analyst.md` (or a knowledge base for the FastAPI/FalkorDB/MCP-version-sensitive ones), log in `analyst/kaizen/history.md`, clear the inbox. |
| K-017 | 2026-08-20 | low | 🔵 | Item 4 of the "Broader team-verbosity reduction" diagnosis (surfaced 2026-08-19; items 1-3 delivered, see `history.md`): prune hedge-stacking once a rule has structural backup (a hook, a routing table) instead of three defensive clauses. No specific instances identified yet — start with a scan across the agent prompts for hedge-stacked clauses backed by real harness enforcement (a `PreToolUse` hook, a routing table) and trim each to one clean statement. |
| K-019 | 2026-08-21 | **high — filed upstream** | 🔵 | **Systemic, now confirmed matcher-agnostic too. `PreToolUse` "ask" hooks do not reliably pause execution in this session under Auto Mode, on either `Bash` or `Write`/`Edit`, regardless of hook source or execution context.** Four independent, isolated live tests, 2026-08-21, Claude Code 2.1.238, all under Auto Mode: (1) `graph-dba`'s own frontmatter `Bash` hook, Task-dispatched with `subagent_type` explicitly correct — didn't fire. (2) The identical guard mirrored as a session-wide `.claude/settings.local.json` `Bash` hook, run from `cobb`'s own **main session** — didn't fire. (3) Same test repeated after the user explicitly reloaded hook config via `/hooks` (visibly listed as registered, `[Local] Bash — 1 hook`) — still didn't fire. (4) **`cobb`'s own frontmatter `Write`/`Edit` hook** (`guard-cobb-topic-writes.sh`) — a `Write` to a path plainly outside cobb's allowlist (`docs/_hook_test_k019_scratch.md`) went through with zero interruption; re-fed the exact real payload to the script directly afterward and confirmed it correctly returns `ask` for that path. **Every test used a real, disposable payload (scratch graph or scratch file, immediately cleaned up) and independently pipe-test-confirmed correct hook logic** — ruling out `subagent_type` omission, stale config, hook-not-registered, and matcher-specific quirks as explanations. **Working hypothesis:** Auto Mode's classifier layer silently resolves/overrides a correctly-emitted `ask` decision before a human ever sees it, across both tool matchers tested. **Filed upstream 2026-08-21** via `/feedback` (user-submitted, confirmed "Feedback / bug report submitted") with the 3-test Bash repro; the 4th (Write/Edit) test landed after filing, not yet included in a follow-up report. **Practical consequence, effective immediately: every "harness-enforced" Guardrails claim across every guarded agent in this team — all three destructive-ops guards, all eight doc-write allow-list guards, the one broad-write deny-list guard — is currently unverified, and actively disconfirmed on the two mechanisms tested, under Auto Mode, in every execution context tried.** Not yet tested: the Write/Edit + Task-dispatched-subagent combination specifically (all 4 tests covered 3 of the 4 matcher×context cells) — very likely shares the gap given the pattern, not confirmed. **Next steps:** (1) monitor for an Anthropic response to the filed report; (2) treat this as the standing state of the team's enforcement model — Auto Mode being off is the only known workaround, untested/not decided; (3) fill the last untested cell (Write/Edit, subagent-dispatched) if a clean answer is ever needed before Anthropic responds. |

### K-001 — Re-verify standards against live docs
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** Frontmatter fields, directory paths, and inclusion modes for Claude Code, Kiro, and OpenCode shift between releases. Stale specifics would make Cobb produce broken artifacts.
- **Proposed change:** On a cadence (or whenever a user reports a mismatch), fetch kiro.dev/docs, opencode.ai/docs, code.claude.com/docs, platform.claude.com/docs and reconcile the "Standards you know cold" section. Log diffs in history.md.
- **Notes:** Baseline verified 2026-05-31 at creation. Subagent context-loading + frontmatter re-verified 2026-06-07 against code.claude.com/docs/en/sub-agents. **Kiro + OpenCode re-verified 2026-06-07** (kiro.dev/docs/steering, opencode.ai/docs/agents+rules) during the K-007 skill extraction — caught real OpenCode drift (`mode: all` default, new `disable`/`color`/`top_p`/`steps` fields, granular permission keys, AGENTS.md precedence). All current specifics now live in the `agent-standards` skill with per-file `Verified:` stamps. **2026-06-20:** Claude Code **subagents** re-verified against code.claude.com/docs/en/sub-agents (tool inheritance + withheld-tools list, expanded frontmatter, discovery/scopes, agent teams + background agents) — claude-code.md stamp bumped to 2026-06-20. Claude Code **Skills/Memory/Hooks/MCP/SDK** still on the 2026-05-31 baseline — next refresh target. **Kiro** docs re-read 2026-06-20 (agents/subagents, steering, specs, hooks). NB: the old "does `inclusion: always` steering reach a subagent" dispute is **not** resolved by a doc re-read — docs affirm it, but it's field-disputed and needs a runtime test on the install. Specs/Hooks confirmed *not* reaching subagents. **OpenCode** re-verified 2026-06-20 (agents/permissions/rules — caught the `tools`→`permission` deprecation; subagent nesting documented; flagged the parent-context divergence vs Claude). **All three tools' agent/subagent surfaces now current as of 2026-06-20.** **2026-07-25:** Claude Code **MCP** fully re-verified against `code.claude.com/docs/en/mcp` and rewritten from a 3-line stub into a full section (scopes/approval + untrusted-workspace gate, `.mcp.json` shape, `${VAR}`-only expansion + the `CLAUDE_PROJECT_DIR`-in-the-server's-env trap, tool naming, the four timeout layers, **tool search on by default** + `alwaysLoad` + server `instructions`, **output limits and the persist-to-disk/file-reference behaviour**, stdio non-reconnection, and how MCP meets subagent `tools:` allowlists); **OpenCode** MCP + Skills sections added/re-verified against `opencode.ai/docs/{mcp-servers,skills}`. Remaining stale: Claude Code Skills/Memory/Hooks/SDK (2026-05-31); Kiro MCP not re-checked since 2026-06-20.

### K-002 — Worked cross-tool porting example
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** Porting between tools is a common request; a canonical example would make answers faster and more consistent.
- **Proposed change:** Add a reference walkthrough mapping a Claude Code subagent's frontmatter/body to an OpenCode markdown agent and to Kiro steering, noting what each tool drops or renames.
- **Notes:** Could live as a skill rather than bloat the agent prompt.

### K-003 — Broaden tool coverage
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** The open AGENTS.md and Agent Skills standards are adopted by more tools than the core three.
- **Proposed change:** Add concise coverage of Codex CLI, Cursor, Gemini CLI, Copilot where they intersect the open standards, clearly flagged as secondary.
- **Notes:** Keep the big-three depth primary; don't dilute.

### K-005 — Automate doc-drift detection
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** K-001 (manual re-verify) relies on someone remembering. A frozen prompt silently rots between checks. Determinism beats hope: a harness-run job that diffs the official docs and surfaces changes is the real safeguard.
- **Proposed change:** A scheduled agent (Claude Code `/schedule` cron routine, or local cron) that, per tool, fetches the canonical pages (code.claude.com/docs, opencode.ai/docs, kiro.dev/docs, platform.claude.com/docs), diffs the relevant sections against a stored `sources/` snapshot (last-verified date + section excerpt/hash), and on change appends an item to this plan + pings the user. Keeps perishable specifics out of the always-on prompt and re-checked on a cadence.
- **Notes:** Surfaced 2026-06-07 — user asked "how do we ensure the info won't drift?" Pairs with the new "Drift-resistance" principle (timestamp + verify volatile facts). Offered to build it; awaiting go-ahead.

### K-008 — Dog-food the frontmatter cobb teaches
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** Cobb runs on `name`/`description`/`model` only, yet teaches a rich field set (`memory`, `disallowedTools`, `permissionMode`, `skills`, `isolation`, `effort`). A `memory: project` store (auto-injected `MEMORY.md`) would give cobb persistent cross-session knowledge of drift findings / verified-dates / gotchas — distinct from kaizen (human change-log, not auto-injected into the prompt).
- **Proposed change:** Evaluate adding `memory: project`. Leave the `agent-maintenance` skill on-demand (do NOT pin via `skills:` — pinning defeats leanness; the on-demand choice is deliberate).
- **Notes:** Surfaced 2026-06-07 self-review.

## Parking lot / ideas
- **Extend the independent-review-gate practice to `cobb.md` self-edits specifically (surfaced
  2026-08-20, `Q2`'s D-1 finding + its own fix).** `Q2`'s closing acceptance pass
  (`docs/test-reports/generic-cypher-mcp2-report.md`) found `cobb.md` had shipped a self-edit
  (the M7 `C-cobb` "Learning capture" retarget) that only touched one of two affected sections,
  leaving lines 65/71 stale and self-contradictory — undetected because no independent reviewer
  ever read that diff (the self-edit carve-out, §3.7 of `docs/plans/generic-cypher-mcp2.md`, makes
  `cobb` both author and sole editor of that one file). The fix for D-1 (this same 2026-08-20,
  `history.md` above) is **itself** another unreviewed `cobb.md` self-edit — the exact shape most
  likely to repeat the miss. Cheap mitigation the report proposes: whenever a self-edit unit
  closes, route a one-line "did every section I was supposed to touch actually change?" grep-diff
  check to a second agent. Not actionable unilaterally (no reviewer in a direct, non-`teco`
  dispatch) — raise with `teco` next time a `cobb.md` touch is coordinated, or self-apply the
  grep-diff check as a matter of discipline even without a formal second reviewer.
- **`kaizen/inbox.md` headers are enforced-frozen in practice, not just by written convention
  (surfaced 2026-08-20, M7 `C-<agent>` header-retarget attempt).** `docs/plans/generic-cypher-mcp2.md`
  §4.2/P3-M3 reasoned that a header note's *prescriptive* clause (the copy-pasteable
  `mcp__cypher__query(graph='kaizen_<agent>', ...)` pointer) was safely editable because each
  header's own immutability promise is scoped to "Content below," not the header itself — a
  textually sound argument, gated through 3 plan-review passes. Live execution disagreed: the
  permission system denied 3 of 4 attempted edits outright ("this is frozen"), and the stakeholder,
  relayed through `teco`, then directed dropping the header-retarget half entirely and reverting the
  one edit that had already landed (`teco`'s). **Don't plan future work that treats the "Content
  below" scoping argument as actionable without re-confirming live first** — a textual carve-out in
  a doc is not the same thing as a carve-out the actual permission gate (or the stakeholder) will
  honor at execution time. If a future delivery genuinely needs a frozen `inbox.md` header touched,
  raise it as its own small, explicitly-flagged ask rather than folding it into a larger unit's
  done-condition.
- **Redirected from `teco`'s 2026-08-12 inbox entry (distilled 2026-08-15):** a directly-invoked
  (non-`teco`-coordinated) large sweep I ran — the 39-file full-team kaizen-inbox distillation,
  gated "needs changes" by `analyst` — had no coordination ledger, and the session that ran it hit
  a mid-run credit exhaustion before the fix pass was dispatched. Recovery only worked because the
  review (`docs/reviews/kaizen-inbox-distillation2.md`) was self-sufficient: explicit baseline
  commit + explicit file scope, `analyst`'s standing review-header practice. **Not filed as a
  prompt change** — one data point, no repeat, and the safety net that saved it is already
  standard practice, not a gap. Parking here as a reminder: if I (`cobb`) ever run another large,
  review-gated, directly-invoked (not `teco`-routed) sweep, keep that same discipline (explicit
  baseline + scope in whatever review/report anchors the work) rather than assuming it'll survive
  on vibes — and if this pattern repeats, it graduates from parking lot to an actual guardrail.
- From the 2026-08-09 self-review during the C-308/C-312/C-319 skill-review pass
  (`docs/reviews/cpg-followups-skills-impl.md`): the C-319 promotion compressed two independently
  true, parallel "cwd-independent" facts (`.mcp.json` discovery walk-up; `${CLAUDE_PROJECT_DIR}`
  expansion) into one causal clause ("stays uniform via ...") that the source evidence never
  established. General lesson for future inbox distillations (§5): when promoting a fact that
  echoes or sits next to another fact in the doc, keep them stated as separate claims unless the
  *mechanism link* between them was itself verified — don't add connective "via"/"because" prose
  as free editorial polish. Consider adding a line to §5's procedure about this specific failure
  mode (causal-compression during promotion) if it recurs.
- From the 2026-08-09 independent review of `analyst`'s inbox-distillation pass
  (`docs/reviews/analyst-inbox-distillation.md`): consider a sub-list format for `analyst.md`'s
  "Evidence over vibes" Guardrails bullet (now 5 sub-rules in one run-on sentence after four clause
  extensions) to restore scannability without adding to the bullet count. Also: `claude-code.md`'s
  top-of-file `Verified:` stamp block could gain a one-line pointer to the new "Bash tool
  environment" section (observed-not-doc-sourced, so it doesn't fit the existing dated-doc-citation
  pattern, but a reader skimming only the header wouldn't know the section exists).

- **All 13 agents' `permissionMode: acceptEdits` frontmatter is confirmed dead configuration for
  controlling their own top-level starting mode (surfaced 2026-08-24,
  `claude/docs/plans/permission-default-mode.md`).** The documented session-start decision order has
  no frontmatter step; `~/.claude/settings.json`'s explicit `"defaultMode": "auto"` is what actually
  decides it, every time. Decide: remove the misleading line team-wide (13-file edit, its own
  blast-radius question — a stakeholder call, not unilateral), or leave it as declared intent for a
  future dispatch-time inheritance case (parent in `default`/`plan`/`dontAsk`, currently never
  arising since every parent starts in `auto`). Not urgent — the field being inert doesn't currently
  cause a behavior regression, just misleading configuration.
- **The graph-shaped Learning-capture template is the floor now (noted 2026-08-20).** The
  2026-08-19 verbosity pass had cut the near-identical "Learning capture" section (~1,500 w across
  13 files) by pointing each agent's paragraph at its own `kaizen/inbox.md` header. That option no
  longer exists: capture is graph-shaped, and the inline `mcp__cypher__query` template is verbosier
  than either prior state — a deliberate trade of prompt leanness for mechanism consistency.
  Recorded so the next verbosity pass over this section doesn't rediscover the file-pointer fix and
  find it unavailable. The one unexecuted slice of that pass is **K-017** (Active table).
- From the 2026-08-12 corrective pass fixing `analyst`'s gate on the 2026-08-11 distillation
  (`docs/reviews/kaizen-distillation-2026-08.md`): a raw inbox entry can leak the maintainer's
  home path/username into a tracked file the moment an agent *appends* it — before any
  distillation ever runs, since `kaizen/inbox.md` is itself a tracked file and an entry's
  `**Evidence:**` line often quotes a live shell command's literal output (`ls -la ~/.claude/
  agents` prints the real symlink target). Caught here only because `audit-team.sh` check 7 was
  re-run incidentally, not because anything in the distillation-review workflow prompts for it.
  Worth deciding whether check 7 (or a lighter version of it) should run as part of *any* agent's
  closing protocol when it appends an inbox entry that quotes command output, not just at
  distillation time — candidate for `agent-maintenance` §5 or the "Learning capture" boilerplate
  itself.
- **A `fork` subagent can drift into narrating the parent's own already-completed work instead of
  its assigned directive (observed 2026-08-21, team certification's §7 lint fold-in).** One of
  three forks, each given an explicit, narrow directive (§7 lint on 4 named files), came back
  reporting a status summary of *my own* preceding work instead of any finding about its assigned
  files — plausible mechanism: a fork inherits the full parent transcript, and this one launched
  right after a transcript stretch dense with the parent's own narrated fixes, which may have
  pulled its generation toward continuing that narration. Handled by treating the result as
  unverified and doing the bounded piece directly rather than re-forking blind. One data point —
  not promoted to a `kaizen_team` entry yet (see `claude/cobb/kaizen/history.md`, 2026-08-21
  certificate entry, for the full reasoning). If this recurs, it's worth: (a) a `kaizen_team`
  entry, and (b) a prompting mitigation — e.g. explicitly telling a fork mid-directive "ignore
  what the parent session already did; your only output is findings on these N files."
- Maintain a small catalog of agents/skills Cobb has authored, cross-linking their kaizen files.
- The §7 prompt-lint is judgment-only by design; if a *deterministic* pre-check for a single artifact ever proves cheap (frontmatter valid, description non-empty, no personal identifiers), consider a small script assist — but keep the seven semantic dimensions in the skill, not a grep. *(Noted 2026-07-16 during the §7 build; the composition load-set enumerator the design floated was skipped as not-cheap-enough.)*
