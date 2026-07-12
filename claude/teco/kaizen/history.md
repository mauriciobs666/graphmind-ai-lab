# Kaizen — Change History: teco

> Dated log of actual changes to the `teco` agent. Most recent first.

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
