# Kaizen — Change History: teco

> Dated log of actual changes to the `teco` agent. Most recent first.

## 2026-07-29 — Manuals join the routing table, handoff contracts, doc scan, and review-gate defaults
- **What:** Four small additions reflecting tico's new Mode 2/3 (didactic explanation + user-manual maintenance, same day): (1) a new routing-table row — live explanations stay pause→user (tico isn't a delegation target), but a self-contained manual write/update is delegable to tico like any other subagent deliverable; (2) the tico handoff-contract line now names `docs/manuals/<slug>.md` alongside the requirements doc; (3) the documentation-impact scan bullet now lists user manuals (flag, don't write — `tico` owns them); (4) the "Work ships independently reviewed" guardrail gained a manuals entry: split by claim — `qa-engineer` verifies walkthroughs against the running app, `analyst` checks architectural/factual claims and clarity. The manuals-delegable routing row also notes the review gate still applies when teco routes a manual update this way.
- **Why:** user ruling following the 2026-07-29 team certification, which flagged manuals as the one doc kind with no independent-review gate; user chose the qa-engineer/analyst split (behavioral vs. everything else) and "mandatory in teco + offered in tico's first-order sessions" for how forced the gate should be.
- **Plan items:** none.

## 2026-07-29 — PII leak fixed in this file (found by the team certification pass)
- **What:** The K-009 entry below (added earlier the same day) had embedded the literal
  flattened `~/.claude/projects/...` transcript-directory path, which leaks the OS username —
  genericized to `<flattened-repo-path>`. Working-tree fix only; the leak reached one shared
  commit (`e7ec4a3`) before being caught — not rewritten, per the repo's don't-rewrite-shared-history
  norm.
- **Why:** Surfaced by `claude/scripts/audit-team.sh` check 7 during the 2026-07-29 team
  certification (see cobb's kaizen history for the full pass).
- **Plan items:** none.

## 2026-07-29 — Learnings inbox distilled: 3 entries → 1 promoted to teco.md, 1 to agent-standards, 1 discarded as duplicate
- **What:** Processed all three pending entries in `kaizen/inbox.md` (agent-maintenance skill §5):
  1. **2026-07-25 — "`.mcp.json` server materializes only at session start; subagents inherit MCP tools from the parent session"** — genuinely new, not previously captured anywhere in the repo (checked `AGENTS.md`, `cpg/mcp/README.md`, `skills/agent-standards/claude-code.md`). **Promoted** to `skills/agent-standards/claude-code.md` § MCP → Lifecycle (a harness-level fact, not teco-specific — belongs in the on-demand reference cobb maintains, not an always-loaded prompt), with a `Verified: 2026-07-25` stamp and the cpg-query-access delivery as evidence.
  2. **2026-07-25 — "verifying 'no new audit failures' needs a diff against the last commit, not a re-read of the gate's verdict"** — checked `skills/agent-maintenance/SKILL.md` §4 and found it **already promoted**, word-for-word disposition, same origin date (2026-07-25) and same task (`docs/plans/cpg-query-access.md` rework). **Discarded** as a duplicate of an already-landed promotion — nothing to do.
  3. **2026-07-27 — "a brief that fences off `claude/` silently disables the delegate's own learnings inbox"** — teco's own coordination mistake, still live risk (any future brief that excludes `claude/` for collision-avoidance repeats it). **Promoted** into teco's own prompt (step 3, appended to the model-routing sentence): carve out the delegate's `kaizen/inbox.md` explicitly, or have the learning come back in the report, whenever a brief fences off a subtree containing it.
- **Why:** user asked to process the inbox after the K-006/008/009/010/011 backlog pass. Each entry got the full §5 treatment (verify still true / not already documented, route to exactly one destination, log, clear) rather than a blanket append.
- **Plan items:** none (inbox distillation, not a plan item).

## 2026-07-29 — K-006, K-008, K-009, K-010, K-011 ✅: all five open plan items closed
- **What:** Worked the full active backlog in one pass.
  - **K-008 (verified, then adopted):** Live-tested whether the `Agent` tool's per-call `model` override reaches a call made *from inside* a subagent — spawned a `general-purpose` agent that itself called `Agent(model:"haiku", run_in_background:true, ...)`; grepped the resulting nested transcript (`agent-<id>.jsonl`) and found `"model":"claude-haiku-4-5-20251001"`, confirming the override is honored one level down. Added a sentence to step 3: pass `model: "haiku"` on cost-insensitive units (routine doc touch-ups, small-diff re-reviews, suite runs); anything with design/code-quality stakes stays on the inherited model.
  - **K-009 (audited, then dropped):** Grepped all 5 of teco's own session transcripts (`~/.claude/projects/<flattened-repo-path>-claude-teco/*.jsonl`) for direct `WebFetch`/`WebSearch` tool_use. Found exactly one hit: 2026-07-24, during the K-002 agent-teams evaluation, teco fetched `code.claude.com/docs/en/agent-teams` and `/agent-view` directly instead of delegating — research that its own routing table already assigns to `cobb` ("Agent/subagent/skill/prompt/hook engineering"). One mis-routed use in the whole history doesn't justify the grant; dropped `WebFetch`, `WebSearch` from `tools:`.
  - **K-006 (decided):** The independent-review guardrail's defaults line named `analyst` for "plans and code" without saying whether that covered `graph-dba` design notes or `cobb` agent/skill deliverables. Made both explicit in the same clause rather than adding a new row: `plans and code (including graph-dba design notes and cobb's agent/skill artifacts) → analyst`.
  - **K-010 (trimmed):** Description's trailing clause ("Does not design solutions and routes non-trivial implementation to a specialist; may fix a genuinely trivial single-file no-brainer directly instead of delegating it" — 170 chars) shortened to "Delegates non-trivial implementation; may fix a trivial single-file no-brainer itself." (88 chars) — the routing table's row 1 already carries the full tie-breaker prose, so the description only needs the routing signal.
  - **K-011 (pruned):** Removed the three single-use Bash allow-rules (exact escaped Cypher literals from the K-001 probe run) from `.claude/settings.local.json`, leaving only the `test_queries.sh` entry.
- **Why:** user asked to work the teco kaizen backlog. K-008 and K-009 were verification-gated/evidence-gated rather than pure opinion calls, so both were resolved empirically (live nested-call test; transcript grep) instead of by inference.
- **Plan items:** K-006 ✅, K-008 ✅, K-009 ✅, K-010 ✅, K-011 ✅ — all moved here; active table now empty.

## 2026-07-29 — Credit/interface analysis backlogged as K-008..K-011 (review only, no source change)
- **What:** A user-requested analysis of teco's interfaces and credit consumption produced four new plan items, filed (not implemented): **K-008** (high) — route cost-sensitive delegations to a cheaper model via the `Agent` tool's own per-call `model` param, distinct from the per-agent frontmatter pin the team already rejected; needs live verification it reaches nested calls before adopting. **K-009** (medium) — audit whether teco itself ever uses its `WebFetch`/`WebSearch` grants (vs. delegating research), drop if unused. **K-010** (low) — scheduled token-cost recompression: description regrew 568→694 chars (+22%) since 2026-07-25, body regrew 9,866→12,656 (+28%) since 2026-07-11, both from legitimate feature additions rather than waste. **K-011** (low, hygiene) — prune three single-use Bash allow-rules left in `.claude/settings.local.json` from the one-off K-001 probe run.
- **Why:** the same analysis session that produced K-007 (above) surfaced these as lower-priority or verification-gated items not to act on immediately; recorded per the agent-maintenance skill's "record new ideas even on a review-only pass" rule rather than left informal in chat.
- **Plan items:** K-008, K-009, K-010, K-011 opened (all 🔵 proposed).

## 2026-07-29 — K-007 ✅: `SendMessage` continuation replaces cold respawn in the defect→fix→re-run loop
- **What:** Three touch points, no catalog change needed (internal execution mechanism, not a routing/deliverable-contract change). (1) `tools:` gained **`SendMessage`** — it was absent, so K-007 as previously worded would have been unshippable even after a step-4 rewrite. (2) Step 3 gained one clause: note the name/id each `Agent` call returns for any unit carrying a review gate or likely to need a follow-up round, since that identifier is what a later `SendMessage` addresses. (3) Step 4's two re-brief paths — the review "needs changes"/qa-defects loop, and the K-004 deficient-result path (errored/out-of-turns/off-brief/empty) — now both `SendMessage` the original delegate by that identifier (resumes from its own transcript, no re-explaining context) instead of a fresh `Agent` call; cold respawn is reserved for when the identifier **no longer resolves** (no name/id was ever returned, or a newer agent has since taken the same name) — the actual boundary condition per `SendMessage`'s own tool description, not the "errored/out-of-turns" split first drafted (see below).
- **Why:** the session's own analysis (prompted by a user request to find credit/interface optimizations) identified this as the highest-value unshipped lever: the defect→fix→re-run loop was re-explaining full context to a cold `Agent` spawn every retry cycle. `SendMessage`'s live tool description (fetched via `ToolSearch` this session, not a cached doc page) resolved K-007's open verification question — it explicitly states a send "resumes it from its transcript" for a named agent, "even after an agent completes," matching the `Agent` tool's own description ("use SendMessage with the agent's ID or name ... resumes it with full context"). This is the current harness's own self-description, stronger evidence than the two doc pages (`agent-teams`, `agent-view`) the original K-007 note flagged as describing two different mechanisms — still worth confirming empirically on the first real re-brief cycle, but no longer blocking.
- **Self-caught fix during drafting:** the first draft of the step-4 rewrite said "fall back to a fresh `Agent` call... when the original agent errored out entirely or exhausted its turn budget" — directly contradicting the same sentence's "deficient" category, which lists "errored, ran out of turns" as `SendMessage`-retry triggers. Caught by a §7-style self-check before this entry was written; corrected to the identifier-resolution boundary instead (see above).
- **Verified no regression:** `claude/scripts/audit-team.sh` re-run clean on teco (no teco-related FAIL; the 4 pre-existing FAILs are root `AGENTS.md` missing `coder`/`devops`/`frontend-engineer`/`tdd-engineer` — unrelated drift, out of scope here, reported not chased).
- **Plan items:** K-007 ✅ done (moved to plan.md's done-notes block).

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-27 — Milestone-close freeze becomes a coordination duty; coordination docs open with the header block (step 2 of `docs/plans/doc-reference-convention.md`)
- **What:** Two body edits, no frontmatter and no hook change. (1) *How you work* step 2 gains one line — *"Open the document with the header block from root `AGENTS.md`."* — the canonical sentence, byte-identical across the six producing prompts. (2) *Documentation curation* gains a third bullet: at milestone close, list every document the close freezes and make flipping each one's header to `Status: archived` a **done-condition of the closing unit, routed to that document's owner** (root `AGENTS.md` carries the per-kind routing table); nothing moves; `teco` coordinates and performs only the flip the table assigns it — its own `docs/plans/<slug>-coordination.md`.
- **Why:** `docs/plans/doc-reference-convention.md` v1.4, blocker **B5** and M2. Under D4 a frozen document no longer moves to `archive/` — it gets `Status: archived` in place — which turns "archiving" from a file operation nobody had to schedule into a **flip somebody must be told to perform**, at a moment (`milestone close`) only the coordinator sees. Without this bullet the lifecycle signal the whole convention rests on would simply never be set. Routing rather than performing is forced by the guard topology, not by ceremony: `teco`'s `PreToolUse` allowlist reaches `docs/plans/*` only, so a flip it performed on a review, requirements doc, test plan or test report would raise an interactive human approval prompt **per file** — `falkor-chat/docs/reviews/` alone holds four active documents. The routing table is pointed at, not copied, for the same reason the header block is (v1.4 M20): root `AGENTS.md` is already in every agent's context via the root `CLAUDE.md` `@AGENTS.md` import, so the hop costs nothing while a second copy would drift. `claude/README.md` row 7 re-checked — it already describes `teco` as documentation curator who makes doc updates part of every unit's done-condition, which is exactly what this bullet instantiates; no catalog edit needed.
- **Plan items:** none. (K-006/K-007 untouched.)

## 2026-07-25 — Trivial single-file no-brainer fixes: teco may make them directly instead of delegating
- **What:** Relaxed the "coordinates, never implements" invariant one notch: teco may now make a genuinely trivial, single-file, no-design-needed fix (a typo, an obvious one-liner, a config value, a rename) directly instead of spinning up a specialist for it. Four touch points, no hook-allowlist change: (1) frontmatter `description`'s closing line now reads "does not design solutions and routes non-trivial implementation to a specialist; may fix a genuinely trivial single-file no-brainer directly instead of delegating it" (was an unqualified "Does NOT design or write code itself"); (2) opening persona paragraph states the exception inline; (3) Routing table gained a leading row (trivial single-file no-brainer → teco directly, tie-breaker: multiple files/design judgment/security-data-model-test-critical → delegate instead); (4) Guardrails' coordination bullet and ceremony bullet updated to match. The `PreToolUse` hook (`guard-coordination-doc-writes.sh`) is **unchanged in behavior** — its allowed globs still only cover `docs/plans/` and the kaizen inbox, so a trivial fix still hits the "ask" escalation and needs a one-time human approval; only the escalation *message* was reworded (no longer "deny by default", now "approve if this is genuinely that kind of trivial fix"). This keeps a human check on every non-coordination-doc write teco makes, trivial or not — it just stops teco from having to pretend the option doesn't exist.
- **Why:** User request: too much delegation overhead going to `coder` for small no-brainer changes. Discussed the trade-off first (this reopens ground settled by the 2026-07-08 architect/teco K-003 hook-enforcement work) and the user chose the narrowest of three options offered — prompt-level permission for trivial edits only, hook left as the safety net — over widening the hook's allowlist or just trimming routing ceremony elsewhere.
- **Plan items:** none (out-of-band user request). Worth revisiting if the "ask" escalation for trivial fixes turns out to fire often enough to reintroduce the friction this was meant to remove — that would be the signal to reconsider widening the hook allowlist (the second, rejected option).

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 661 → 568 chars (-14%): tightened phrasing, dropped restated detail. `teco` has no boundary pairs in `claude/scripts/audit-team.sh`; full audit re-verified green regardless. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — K-002 ✅: agent-teams evaluation closed — reject team-lead reframe; SendMessage sub-case spun to K-007
- **What:** Read `code.claude.com/docs/en/agent-teams` and `/en/agent-view` (the concrete step K-002 asked for) and closed with disposition: **reject** reframing teco as an agent-teams lead. Agent teams are experimental (opt-in `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`), built for teammates that talk directly to each other on independent, discussion-benefiting work (parallel review lenses, competing-hypothesis debugging, cross-layer ownership) — the docs are explicit that "for sequential tasks... or work with many dependencies, a single session or subagents are more effective." Teco's actual loop (decompose → sequence on dependencies → delegate → independently-reviewed gate) is exactly that latter shape; teams would add token overhead for no matching benefit. The 2026-07-12 sub-case (defect→fix→re-run re-spawning cold agents) turned out not to be an agent-teams question at all — it's answered by `SendMessage` continuation of the original delegate (confirmed available for `Agent`-tool subagents per the harness's own tool description), independent of the experimental teams flag. Spun off as **K-007**.
- **Why:** User asked to follow through on K-002's own proposed next step (read the docs, assess fit) rather than leave the plan item open indefinitely.
- **Plan items:** K-002 ✅ done (moved here); opened K-007 (adopt SendMessage continuation in step 4's defect→fix→re-run loop).

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change across the team (`coder`, `tdd-engineer`, `frontend-engineer`, `architect`, `qa-engineer`, `analyst`, `devops`, `graph-dba`, `joern`). File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this is safe: `PreToolUse` hooks fire *before* any permission-mode check, and a hook's `"ask"` decision still forces the prompt even under `acceptEdits`/`bypassPermissions`. `teco`'s `guard-coordination-doc-writes.sh` hook (escalates to ask on any Write/Edit outside the allowed coordination-doc paths) keeps working exactly as before; only writes it would already let through silently stop re-prompting every session.
- **Plan items:** none.

## 2026-07-16 — Applied K-004 + K-005 to teco.md (from the §7 lint)
- **What:** Two surgical prompt additions, user-approved from the same-day §7 lint. **K-004** — Step 4 ("Integrate & verify") gained a *deficient-result* path: when a delegate errors, runs out of turns, or returns something off-brief/empty (explicitly distinct from a *blocker* that changes direction and a review *verdict*), re-brief the same owner once with the gap made explicit, and pause to the user if it recurs or the unit is mis-scoped — "rather than re-spawning blindly". **K-005** — the Documentation-curation "Scan at decomposition" list now names `docs/HISTORY.md` (which takes an entry for every delivered change) and `docs/BACKLOG.md` "where the module uses the convention", closing the gap between teco's curator role and the module-documentation convention in root `AGENTS.md`. No frontmatter/description/catalog change — role unchanged, so the catalog entries still describe teco correctly. K-006 left proposed (not approved).
- **Why:** User approved acting on the two higher-value lint findings; both were surgical additions at teco's existing altitude, not a rewrite.
- **Plan items:** K-004 ✅, K-005 ✅ (moved to the done-notes block in plan.md); K-006 stays open.

## 2026-07-16 — §7 prompt-quality lint (review-only, no prompt change)
- **What:** cobb ran the new `agent-maintenance` §7 single-artifact prompt-lint against `teco.md` across all six dimensions, resolving teco's full load-set (root + `claude/` `CLAUDE.md`→`@AGENTS.md` chain, the injected specialist `description`s, the coordination-doc write guard) for the composition check. **Persona:** clean. **Contradiction / ambiguity / cognitive load:** clean bar minor nits (parked). **Coverage + composition:** three findings filed — K-004 (no deficient/failed-delegate-result path), K-005 (doc-curation scope omits the module `docs/HISTORY.md`/`BACKLOG.md` conventions from `AGENTS.md` — highest-value, surfaced only by the composition load-set resolution), K-006 (no independent reviewer assigned for agent-engineering deliverables). No blocker; no source change.
- **Why:** Smoke test of the §7 procedure cobb authored the same day; teco is a mature, certified prompt so a clean-ish result was expected and validated that §7 surfaces real gaps without manufacturing findings.
- **Plan items:** opened K-004, K-005, K-006; minors parked.

## 2026-07-12 — K-003 ✅: review-gate invariant proven on the first fully-gated run — kept, no prompt change
- **What:** Closed K-003 with disposition **(a) keep the invariant** — "work ships
  independently reviewed; when you trim ceremony, the review gate is the last thing to go."
  falkor-chat **K-022 Landing 1** (U1–U10, committed `3921f87`) ran as the team's first fully-gated
  coordinated delegation with the analyst post-implementation review as a non-negotiable
  done-condition, and the cost datapoint the plan asked for is now recorded in
  `falkor-chat/docs/plans/m3-executor-coordination.md` ("Cost datapoint" table). **No prompt
  change** — the datapoint vindicates the existing guardrail rather than forcing the (b) rewrite
  to risk-signal-gated review.
- **Evidence / reasoning from the datapoint:**
  1. **The gate is cheap.** Analyst review = ~149k tokens / 25 tool uses / ~7 min — ~12% of the
     ~1.20M-token, ~4h gated run, a thin slice on top of the six implementation delegations.
  2. **The gate paid.** On a diff the implementers considered done it returned
     approve-with-suggestions with **1 major (M-1, the drive try/except) + 3 minor + 3 nit** —
     exactly the class of defect the K-020/21 "review left to the user" skip would have shipped
     unseen.
  3. **The headline ~12× vs. the K-001 baseline is a units artifact, not the gate's cost** — 10
     units + independent gate vs. an ungated 2-unit slice (~100k / 23 / ~45 min). Per-unit the run
     is comparable; the review is the cheap part.
  4. Therefore the concern that opened K-003 — "an invariant that never fires is hopeful prose" —
     is resolved: it fired, cheaply, and caught real signal. Keeping review-by-default is the
     right risk posture; the low marginal cost means the default stands even at n=1.
- **Honest caveat (recorded, not blocking):** this is **one** gated run. It proves the gate can pay
  its way and is affordable, not that every gate will catch a major. The cost is low enough that
  "keep the default, skip only with stated justification for genuinely trivial units" remains
  correct pending more datapoints — re-examine if a run of gates comes back all-nits at real cost.
- **Why:** User asked to close the K-003 thread now that K-022 is committed. The experiment ran end
  to end (gate enforced + datapoint captured); the disposition is the last step the plan item
  named ((a) keep / (b) rewrite).
- **Plan items:** **K-003 ✅ done** (moved here). No change to `teco.md`, `README.md`, or the
  context catalogs — behavior/routing unchanged; this is a decision to *keep* the current prompt.
  Counterparts still open on their own agents: `analyst` K-001 (its code-review shakedown — the
  same run validated it; closeable on analyst's side) and `qa-engineer` K-003 (defect→fix→re-run
  loop — **unexercised**, the review returned 0 blockers so no needs-changes loop fired).

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol + guard allowlist + integration check
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt; the coordination-doc write guard's allowlist gained exactly teco's own inbox path. Step 4 (Integrate & verify) additionally gained the learnings-ride-the-handoff check: when a specialist's result reports a durable environment discovery, confirm it was filed in that agent's inbox (a one-line check, not a gate).
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture during runs, curated promotion by cobb. Teco is the collection point on orchestrated work — the integration check catches learnings a delegate forgot to file. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — graph-dba added to the handoff contracts (certification fix)
- **What:** The "Handoff contracts" list gained the `graph-dba` entry: implementer-bound design work (data model, schema/DDL, ingestion/migration) arrives as a design note at `<component>/docs/plans/<slug>-graph.md`; quick consults and tuning diagnoses stay inline. Matches the same-day addition of the convention to graph-dba's own prompt (its kaizen K-004).
- **Why:** Team-coherence certification (2026-07-11): graph-dba was the only design-producing specialist whose deliverable teco had to paraphrase into the next brief — the exact lossy handoff the "by path, never paraphrased" rule exists to prevent.
- **Plan items:** none (graph-dba K-004 on the producer side).

## 2026-07-11 — Prompt body compressed (token-cost pass, part 2)
- **What:** Body compressed in place, 15,023 → 9,866 chars (−34%): the routing table's per-agent capability prose was cut down to pure routing judgment (tie-breakers, boundaries, pipeline defaults), explicitly leaning on the injected frontmatter descriptions teco already receives at spawn through its `Agent` tool; "How you work", documentation curation, pause rules, and guardrails were tightened without dropping any rule or contract. All 11 specialist names remain in the file (audit check 4 green, full audit pass); frontmatter (description, tools, hook) unchanged. No on-demand reference file — teco uses its whole body every run, so offloading would just add a mandatory Read.
- **Why:** teco.md was the team's second-heaviest prompt and loads on every teco spawn; the injected description catalog already carries each specialist's capabilities, so restating them in the body was pure duplication (~1,450 tokens saved per spawn).
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 1286 to 659 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-11 — Guard hook refactored to a thin wrapper over a shared core
- **What:** `guard-coordination-doc-writes.sh` was reduced from a ~60-line standalone script to a thin wrapper that `exec`s the new shared core `claude/scripts/guard-doc-writes.sh` with two parameters — this agent's allowed-path globs (`docs/plans/*|*/docs/plans/*`) and its escalation-message template (`__PATH__` placeholder for the offending path). The core carries the shared machinery unchanged: jq→python3 path extraction, fail-open on unparseable input, `/tmp/*` always allowed, `permissionDecision: "ask"` JSON emit. The wrapper resolves the core via `readlink -f "$0"`, so it works when invoked through the `~/.claude/agents/<name>` deployment symlink; the frontmatter hook command is unchanged. Verified: `bash -n`, allowed/denied/scratchpad/fail-open cases through the symlink path, the no-jq python3 fallback, and `claude/scripts/audit-team.sh` all pass.
- **Why:** a repo redundancy audit (2026-07-11) found the five doc-scoped guards (analyst, architect, data-scientist, teco, tico) byte-identical except one `case` glob and one message string — ~250 duplicated lines that had to be patched five times per fix. One parameterized core removes the drift risk. (`devops/hooks/guard-destructive-ops.sh` stays standalone — it matches Bash command patterns, not write paths.)
- **Plan items:** none.

## 2026-07-10 — Independent review made the default mindset
- **What:** independent review is now a standing principle, not an optional gate. Four touch points: (1) a new guardrail — "**Work ships independently reviewed**": no deliverable is accepted on its producer's word alone, teco's own integration check is fit/completeness (not a substitute for review), and every significant deliverable defaults to a reviewer who didn't produce it (plans/code → `analyst`, ML methodology → `data-scientist`, behavior/acceptance → `qa-engineer`); skipping a gate is the justified exception for genuinely trivial, low-risk units, stated explicitly in the report. (2) Step 2 now assigns each unit its **review gate** alongside owner/inputs/done-condition. (3) The typical-feature paragraph flips `analyst` from "slotted in where the stakes warrant it" to the **default review gate**, and the match-ceremony-to-task rule gains "when you trim ceremony, the review gate is the last thing to go, not the first." (4) The frontmatter `description` advertises the default.
- **Why:** User request: teco should "always have in his mindset the need for the work to be independently reviewed." The previous phrasing made review an exception teco had to argue itself into; the risk posture the user wants is the inverse — review by default, skip only with justification.
- **Plan items:** none.

## 2026-07-10 — Standing documentation-curator duty
- **What:** teco is now the team's **documentation curator**, keeping project docs always in sync with delivered work. Four touch points: (1) a new "Documentation curation" section with the standing rules — documentation-impact scan at decomposition (READMEs, `AGENTS.md`/`CLAUDE.md`, design/reference docs, catalogs, recorded in the coordination doc), affected docs named in the unit's brief with same-change updates part of the deliverable (the unit's owner writes them; agent/skill docs → `cobb`), verification by actually reading the flagged docs at integration (stale docs = incomplete unit → re-brief), and pre-existing drift reported as a follow-up rather than silently chased; (2) step 2 runs the scan as part of the breakdown; (3) step 4 makes documentation part of done; (4) the frontmatter `description` advertises the curator duty. teco still never writes these docs itself — `Write`/`Edit` stays hook-scoped to the coordination doc.
- **Why:** User request: teco should "keep track of the docs updates, being the curator for an always updated documentation." Curation (track → brief → verify) fits teco's coordinator identity and existing hook scope; the writing routes to the owner of each change.
- **Plan items:** none.

## 2026-07-10 — Hook command made machine-independent (`$HOME` symlink path)
- **What:** the frontmatter `PreToolUse` hook command was rewired from the absolute repo path (`/home/<user>/prg/graphmind-ai-lab/claude/teco/hooks/guard-coordination-doc-writes.sh`) to `$HOME/.claude/agents/teco/hooks/guard-coordination-doc-writes.sh`, which resolves through the user-scope deployment symlink (`~/.claude/agents/teco` → the repo folder). Shell-form hook commands (no `args`) run via `sh -c`, so `$HOME` expands — verified 2026-07-10 against `code.claude.com/docs/en/hooks`. Resolution through the symlink confirmed (`test -x` passes).
- **Why:** the committed agent source leaked the user's personal home path into the repo; the symlink path is identical on any machine that follows the deployment convention (`~/.claude/agents/<name>` → `claude/<name>`), keeping the hook enforceable without machine-specific paths. (`${CLAUDE_PROJECT_DIR}` was rejected: the agents are user-scoped and must guard in any project, where the project dir isn't this repo.)
- **Plan items:** none.

## 2026-07-09 — Roster: added data-scientist (AI/ML/DS advisory specialist)
- **What:** the routing table gained a `data-scientist` row (AI/ML/data-science **method** questions — model/embedding selection, retrieval strategy, RAG/GraphRAG evaluation design, quality metrics, experiment/A-B design, statistical validity — plus methodology reviews and model/retrieval-underperformance diagnosis; boundary notes: advisory-only — implementation of its recommendations routes to the implementers with its note as the brief, general correctness review stays with `analyst`, in-graph vector mechanics/Cypher with `graph-dba`); the handoff-contracts list gained its two deliverables (method note `docs/plans/<slug>-ml.md`, methodology review `docs/reviews/<slug>-ml.md`, hook-enforced advisory-only writes); the frontmatter parenthetical now includes it.
- **Why:** an AI/ML/data-science specialist joined the team; the orchestrator's roster must enumerate every delegate with its current contract (the drift class the 2026-07-09 interface review exists to catch).
- **Plan items:** none.

## 2026-07-09 — Roster: added frontend-engineer (UI-depth implementer)
- **What:** the routing table gained a `frontend-engineer` row (UI-heavy front-end work — components, styling, accessibility, client-side state, front-end performance, Streamlit screens — with the boundary note that back-end/non-UI code stays with `coder`/`tdd-engineer` and incidental template touches don't need the specialist); the frontmatter parenthetical and the typical-feature pipeline now include it among the implementers.
- **Why:** a front-end specialist joined the team; the orchestrator's roster must enumerate every delegate (the drift class the 2026-07-09 interface review existed to catch).
- **Plan items:** none.

## 2026-07-09 — Roster restructured into an explicit routing table + handoff contracts
- **What:** "The team you coordinate" reformatted from prose bullets into two artifacts: a **routing table** (task shape → owner → tie-breaker/boundary, one row per routable signal, including the "requirements vague → pause, recommend tico" row and the two built-ins) and a **handoff contracts** list (per-agent document paths and by-path handoff rules for tico/architect/analyst/qa-engineer). Content is unchanged — same roster, same routing rules, same contracts — only made scannable and self-checkable; the typical-feature pipeline paragraph kept as-is. Catalogs (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`) describe routing behavior, not prompt format — verified accurate, no edits needed.
- **Why:** User asked how teco decides routing and for a "clear configuration". Routing is LLM judgment over prompt text; the clearest configuration of that judgment is an explicit decision table teco self-checks before each delegation (the parking-lot "routing cheat-sheet" idea, now fully addressed — including the coder-vs-tdd tie-breakers on both implementer rows).
- **Plan items:** parking-lot "routing cheat-sheet / decision tree" ✅ resolved.

## 2026-07-09 — Roster: analyst gained RCA routing
- **What:** analyst's roster entry (and the frontmatter parenthetical) now also routes **cause-unknown defects/failures** to it for a root cause analysis at `<component>/docs/reviews/<slug>-rca.md`, whose suggested fix then briefs the implementer (typically `tdd-engineer`, reproduction test first) by path.
- **Why:** analyst extended with an RCA mode the same day (user request); the orchestrator's roster must describe each specialist's current contract.
- **Plan items:** none.

## 2026-07-09 — Roster: added analyst (plan & code review gate)
- **What:** Added `analyst` to the frontmatter specialist list and the roster, slotted it into the typical-feature pipeline as an optional review gate (after architect on high-blast-radius plans and/or after the implementer before QA), and extended step 4's defect loop to cover a "needs changes" review verdict (re-brief the owner with the review path, then re-review). The roster entry encodes the handoff contract: review doc at `<component>/docs/reviews/<slug>.md`, handed off by path, review-only on code (hook-enforced).
- **Why:** New team member created 2026-07-09 — the orchestrator's roster must be updated in the same change as the agent (agent-maintenance §2 step 3; the qa-engineer/devops roster-drift lesson).
- **Plan items:** none.

## 2026-07-09 — tico reframed: first-order agent, not a delegation target
- **What:** Removed tico from the frontmatter routing list; its roster entry now marks it **not a delegation target** — tico runs as the user's own main-session agent (`claude --agent tico`) and teco **consumes** its requirements doc (`<component>/docs/requirements/<slug>.md`) by path, treating vague/uncaptured requirements as a pause point that recommends a tico interview. Pipeline reads **tico (user-run) → architect → implementers → qa**.
- **Why:** User ruling, same day as the roster addition below: tico is a first-order conversational agent, not a subagent — the interview must be a live conversation, which delegation can't provide.
- **Plan items:** none.

## 2026-07-09 — Roster: added tico (product-owner interviewer, upstream of architect)
- **What:** Added `tico` to the frontmatter specialist list and the roster, and prefixed the typical-feature pipeline with it (**tico → architect → implementers → qa**, skipped when requirements are already clear). The roster entry encodes the round-trip contract: tico's question batches are a pause point — relay to the user verbatim, re-delegate with the answers + the doc path (`<component>/docs/requirements/<slug>.md`); the finished doc hands to the architect by path.
- **Why:** New team member created 2026-07-09 — the orchestrator's roster must be updated in the same change as the agent (agent-maintenance §2 step 3; the qa-engineer/devops roster-drift lesson).
- **Plan items:** none.

## 2026-07-09 — Roster: implementer routing de-personalized (efficiency rule)
- **What:** Replaced the coder/tdd-engineer routing guidance in the roster. Dropped the *"(This user prefers TDD — lean toward `tdd-engineer` for implementation unless told otherwise)"* note; both bullets now carry a task-shape rule — route by **efficiency, not ceremony**: detailed architect plan ready to execute → `coder`; bug fix (repro test first), safety-net refactor, test-focused work, or clear-contract feature → `tdd-engineer`.
- **Why:** User ruling: personal-preference notes don't belong in agent prompts — their standing preferences are quality and efficiency, expressed as objective routing rules. Part of the same-day coder/tdd-engineer boundary fix (coder K-001 ✅).
- **Plan items:** none (out-of-band).

## 2026-07-09 — K-001 ✅: live nested-delegation validation run (falkor-chat M3 slice 1)
- **What:** Ran teco end-to-end on a real assignment — kick off falkor-chat **M3 — Workflow
  engine**, decompose the milestone, deliver slice 1 (K-020 def model + K-021 snapshot
  materialization). Launch brief + observation checklist: `k001-run-brief.md` (executed verbatim).
  Scored against the checklist from the run transcript + independent re-verification:
  1. **Depth — PASS.** teco (opus) spawned architect → graph-dba → tdd-engineer (one `Agent` call
     each, sequenced on their upstream artifacts); all three nested runs completed with no
     depth-related degradation observed.
  2. **Path-based handoff — PASS.** All three delegate briefs carried the plan-doc path
     (`docs/plans/m3-workflow-engine.md`); the plan was never paraphrased wholesale into a brief
     (briefs ~6.7–7.7 KB, self-contained context + path).
  3. **Brief fidelity — PASS.** Every brief included the "this brief is your entire context"
     framing and the blockers-back-as-deliverable reminder. No observed information loss; the
     one plan gap (no `start_key` param on `publish_workflow_def`) was an *architect plan*
     omission, resolved sensibly by the implementer and surfaced by teco as a follow-up —
     exactly the intended behavior.
  4. **Hook enforcement — PASS (unexercised).** teco's own Write/Edit calls (1 Write + 5 Edits)
     all targeted its coordination doc (`m3-workflow-engine-coordination.md`); the
     guard-coordination-doc-writes hook never needed to fire.
  5. **Decision points — PASS.** The §13 guard-expression-language question was correctly
     assessed as *not forced* by slice 1 (opaque strings, evaluated at run time) and deferred to
     K-022's architect pass with an explicit return-to-user; `ws:acme`/`reference` kept
     additive-only; zero scope creep (executor/linkage/proof flows untouched).
  6. **Integration & honesty — PASS.** teco re-ran both suites itself and reported truthfully;
     independently re-verified afterwards: `test_queries.sh` **193/193**, pytest **196** — both
     matching teco's claims. Nothing committed (correct; review left to the user).
- **Why:** K-001 was the open proof that an orchestrator subagent works in practice — depth,
  context-passing fidelity, and result quality were validated on a real deliverable, not a toy.
- **Prompt changes:** **none needed** — the run surfaced no prompt weakness. Deliverables landed
  in falkor-chat (see `falkor-chat/docs/HISTORY.md` 2026-07-09). Run cost datapoint: ~100k
  subagent tokens / 23 tool uses / ~45 min for a 2-item slice with 3 nested specialists.
- **Plan items:** K-001 ✅ done (moved here). Same-run evidence closed **architect K-002**
  (plan executed cold by an isolated implementer) and updated **coder K-002** (contract proven
  via tdd-engineer; coder-specific run still open). K-002 (agent teams) remains the sole active item.

## 2026-07-09 — Interface review: roster completed (qa-engineer, devops) + guard hook + brief/verify upgrades
- **What:** Thorough review of teco and its interfaces produced five prompt changes and one new artifact:
  1. **Roster completed** — `qa-engineer` (with its `docs/test-plans/` / `docs/test-reports/` artifact conventions) and `devops` (environment blockers routed there instead of bounced to the user) added to the roster, the frontmatter `description`, and the typical-feature pipeline (now `architect → implementer → qa-engineer`, `devops` unblocking env issues). Both agents postdate teco's creation (qa-engineer 2026-07-01, devops ~2026-07) and had never been folded in.
  2. **Brief template generalized** (step 3) — path-based handoff is now the rule for *every* document deliverable (architect plan named as the canonical case, qa plan/report as the other standing instance); briefs must remind delegates they can't ask mid-run (blockers/questions come back as the deliverable).
  3. **Parallel-delegation mechanics** (step 3) — independent delegations go out as parallel `Agent` calls in one turn; dependent ones sequence on their upstream artifact.
  4. **Verify step clarified** (step 4) — running the project's suites/scripts is in-bounds read-only verification; acceptance-level verification routes to `qa-engineer`, with the defect→fix→re-run loop (re-brief implementer with the report path, re-run failed items — qa-engineer kaizen K-003's teco side).
  5. **Guard hook (harness enforcement parity with architect)** — new `teco/hooks/guard-coordination-doc-writes.sh` wired in frontmatter (matcher `Write|Edit`): any target outside `docs/plans/` (or `/tmp`) escalates to the human (`permissionDecision: "ask"`); same fail-open jq→python3 contract as the architect/devops hooks. Unit-driven: allowed path passes silently, violating path emits the ask JSON.
  - **Counterpart fixes in the same change:** `tdd-engineer` gained the plan-doc-path handoff line (mirroring coder) + subagent-awareness ("return the question/blocker as your result"); `qa-engineer` gained the same subagent-awareness in its scope step and environment guardrail. Catalogs synced: `claude/AGENTS.md`, `claude/README.md` (teco row + hook-gotcha list), root `AGENTS.md` teco cell.
- **Why:** Review found teco's core design sound but stale at the edges: two specialists were invisible to it (it literally could not route QA or infra work), its doc-scoping guardrail was prompt-only while the identical architect contract is hook-enforced, and the delegation protocol's key rules (path handoff, no-mid-run-questions) existed only as special cases instead of general brief requirements.
- **Plan items:** parking-lot "routing cheat-sheet" idea partially addressed (complete roster + routing signals per entry); K-001 (live nested-delegation run) and K-002 (agent teams) remain open.

## 2026-07-08 — Path-based architect handoff + coordination-doc convention (K-003 ✅)
- **What:** Two prompt changes, synced with the architect's same-day overhaul: (1) step 3 no longer says to pass the architect's plan **verbatim** — the architect now writes its plan to `<component>/docs/plans/<slug>.md` and teco hands the implementer the **path** with an instruction to read the file itself, never a paraphrase; the roster's architect line states the convention. (2) K-003 resolved: teco's coordination/work-breakdown doc gets a fixed convention too — `<component>/docs/plans/<slug>-coordination.md`, co-located with the architect's plan (baked into step 2). Catalog entries updated (`claude/AGENTS.md`).
- **Why:** Design review of the architect found the verbatim copy-through was the weakest link in the teco pipeline: a long plan returned as a subagent message and re-pasted into a brief risks truncation/paraphrase, and leaves no durable artifact. A file handed off by path is lossless, cheap to brief, and reviewable after the fact. The coordination-doc convention rode along since it was the same decision (architect K-001 fixed the location).
- **Plan items:** K-003 ✅ done (moved here); K-001 note updated — the live nested-delegation validation is still pending but no longer needs to stress brief fidelity for the plan itself.

## 2026-07-05 — Added `Edit` (scoped to the coordination doc)
- **What:** Added `Edit` to teco's frontmatter tools (`Read, Grep, Glob, Bash, Agent, Write, Edit, WebFetch, WebSearch`). Updated the guardrail to `Write`/`Edit` = **coordination/work-breakdown document only** (Write to create, Edit to revise in place as steps complete) — still **never** source/tests/config. Also tightened "How you work" step 2 to mention editing the doc in place. Mirrored the wording in `claude/AGENTS.md`.
- **Why:** User asked to give teco the `Edit` tool. With `Write` only, teco could create a coordination doc but had to overwrite it wholesale to update it; `Edit` lets it surgically revise the doc across a long-running orchestration (mark steps done, append findings). Scoped deliberately to the coordination doc — parallels `architect`, which carries `Write`+`Edit` guardrailed to its plan doc — so teco's "coordinate, don't implement" identity is preserved.
- **Plan items:** none (out-of-band user request); relevant to K-003 (coordination-doc convention).

## 2026-06-20 — Created
- **What:** Created the `teco` subagent (`teco/teco.md`, `model: opus`). Technical coordinator / tech lead: decomposes a multi-step goal into a sequenced work breakdown and **delegates each unit to the right specialist** (architect, coder, tdd-engineer, graph-dba, cobb; Explore/Plan built-ins) via the `Agent` tool, then integrates and verifies. **Hybrid mode:** delegates execution itself by default but pauses and returns to the user at genuine decision points / blockers / ambiguity. Tools: `Read, Grep, Glob, Bash, Agent, Write, WebFetch, WebSearch` — **no `Edit`/`NotebookEdit`** (it coordinates, doesn't implement); `Write` is for the coordination doc only; `Bash` read-only by guardrail.
- **Why:** User asked for a third agent on top of the architect→coder pair — "teco the technical coordinator" — to orchestrate the specialist roster.
- **Plan items:** seeded K-001..K-003.

## Decisions & verification recorded at creation
- **Subagents CAN delegate to subagents — verified 2026-06-20** against `code.claude.com/docs/en/sub-agents`. The doc enumerates the tools withheld from subagents (`AskUserQuestion`, `EnterPlanMode`, `ExitPlanMode`, `ScheduleWakeup`, `WaitForMcpServers`); the `Agent`/Task tool is **not** withheld, so an orchestrator subagent is viable. (Older lore said subagents couldn't spawn subagents — that constraint no longer holds per the live doc. Claude Code now also has first-class *agent teams* and *background agents*.)
- **Key limitation baked into the prompt:** `AskUserQuestion` is unavailable to subagents, so teco **cannot ask interactively** — the hybrid design has it *return* to the user with the decision instead of guessing. teco also doesn't see the parent conversation, and delegated agents don't see teco's or each other's context → the prompt mandates **self-contained briefs** (pass the architect's plan verbatim to the implementer, etc.).
- **No `name`-conflict / collection consistency:** dropped any "senior" framing to match the 2026-06-20 harmonized collection. Defaults implementation routing toward `tdd-engineer` given the user's documented TDD preference.
