# Kaizen — Change History: tdd-engineer

> Dated log of actual changes to the `tdd-engineer` agent. Most recent first.

## 2026-07-11 — Inbound RCA handoff + qa-engineer boundary (certification fixes)
- **What:** (1) Workflow step 1 now names the second doc-path input the team routes here: an `analyst` RCA at `<component>/docs/reviews/<slug>-rca.md` — its reproduction evidence is the first RED, its suggested fix the target. (2) The `description` now routes acceptance/black-box QA passes to `qa-engineer`, making the altitude boundary symmetric at the routing-contract level (qa's side already named tdd-engineer); `tdd-engineer:qa-engineer` added to `audit-team.sh` `BOUNDARY_PAIRS` so the symmetry is scripted.
- **Why:** Team-coherence certification (2026-07-11, handoff symmetry): analyst and teco both route RCA docs to this agent, but its own prompt only named architect plans as doc-path input; and the qa↔tdd boundary was one-sided.
- **Plan items:** closes the 2026-07-11 parking-lot handoff-symmetry item (same-day).

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 606 to 446 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-09 — Efficiency-based routing boundary with `coder` (description narrowed + made symmetric)
- **What:** Rewrote the `description`'s trigger. Was "use proactively whenever the user asks to implement a feature, fix a bug, refactor…" — which shadowed `coder`'s trigger on any feature task; now scoped to where test-first is the **efficient path** (bug fix with reproduction test first, safety-net refactor, adding/improving tests, feature with a clear up-front behavior contract) and points back to `coder` for executing an already-detailed plan/spec (the pointer was previously one-directional, coder→tdd only). Catalogs synced: teco roster (personal-preference note removed), `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`.
- **Why:** User ruling on the coder/tdd-engineer overlap review: route between the implementers by efficiency, not by an assumed blanket TDD preference; personal-preference framing removed from agent prompts (the user's standing preferences are quality and efficiency). Closes coder K-001 from this side.
- **Plan items:** none active (out-of-band; counterpart of coder K-001 ✅).

## 2026-07-09 — Plan-doc-path handoff + subagent-awareness (teco interface review)
- **What:** Two workflow additions, made during the teco interface review: (1) step 1 now states that an `architect` plan arrives as a **document path** (`<component>/docs/plans/<slug>.md`) — read the file itself as the source of truth; its test-strategy section is the red→green sequence. Mirrors the line `coder` has carried since the 2026-07-08 path-based-handoff change. (2) Step 2 and the red-suite + environment-blocker branches of step 3 gained subagent-awareness: when running delegated (e.g. by teco), "ask one sharp question" / "ask whether to fix first" / "ask before installing" becomes "return the question/blocker as your result" — subagents can't ask mid-run. Catalog entry (`claude/AGENTS.md`) updated.
- **Why:** teco routes implementation to this agent *preferentially* (user's TDD preference), yet the handoff contract was documented only on `coder`; and the "ask" phrasing silently assumed an interactive session the agent doesn't get under delegation.
- **Plan items:** none (out-of-band, driven by teco's 2026-07-09 review).

## 2026-06-20 — Dropped "Senior" from description (collection harmonization)
- **What:** Frontmatter `description` "Senior software engineer who implements…" → "Software engineer who implements…". Catalog row in `claude/README.md` "Senior engineer who implements…" → "Software engineer who implements…".
- **Why:** Collection-wide harmonization. The new `architect`/`coder` agents dropped "senior" entirely over the overconfidence concern; this brings tdd-engineer in line. **Supersedes the 2026-06-05 decision** that deliberately *kept* "Senior" as a role/altitude signal — the collection now omits it everywhere, relying on concrete process + guardrails for altitude/calibration instead.
- **Plan items:** —

## 2026-06-05 — Dropped tenure-boast framing
- **What:** Removed "with decades of experience" from the `description` and "with decades of hands-on experience" from the opening body line (now "You are a software engineer who works across many languages, paradigms, and frameworks."). Kept "Senior" in the description as a role/altitude signal.
- **Why:** User feedback — the "decades of experience" framing reads as cocky and doesn't change behavior. Applied collection-wide (also graph-dba, dra-claudia).
- **Plan items:** —

## 2026-06-05 — Implemented K-005 (third cold-start branch: tests can't run in this env)
- **What:** Edited `tdd-engineer.md` workflow step 3. Added a third sub-branch alongside "greenfield" and "suite already red on arrival": **"Framework exists but the suite can't run here."** It instructs the agent to recognize an environmental block (deps not installed, missing runtime/toolchain, required build/service step) as *not* a code RED, avoid misattributing it or thrashing on setup, report the blocker plainly, propose the bootstrap step (`npm install`, `uv sync`, build, etc.), and ask before installing/changing the environment — establishing a runnable baseline before the first RED.
- **Why:** Closes K-005. The original two branches silently assumed an existing framework was runnable; in practice a present-but-unexecutable suite is common and was previously unhandled, risking false REDs.
- **Plan items:** K-005 ✅ (closed, removed from active backlog). Active backlog now empty; K-003 deferred remains the only standing decision.

## 2026-06-05 — K-003 deferred (keep tools unconstrained)
- **What:** Decision only, no file change to the agent. Marked K-003 ⚪ deferred — `tdd-engineer` keeps no `tools` key and continues to inherit all tools.
- **Why:** User chose to keep the agent flexible for now (able to spawn subagents and fetch docs mid-task) rather than restrict to a focused TDD set. Recorded so it isn't re-proposed; revisit only if broad tool access causes surprise in practice.
- **Plan items:** K-003 ⚪ deferred.

## 2026-06-05 — Implemented K-004 (discoverability: catalog + context files)
- **What:** Collection-level docs for the three Claude agents (no agent-prompt change). Created `claude/README.md` (human catalog: one row per agent with what/when/model + links to each source file and `kaizen/` folder; kaizen index; conventions). Created `claude/CLAUDE.md` (agent-context: concise per-agent pointers to source + kaizen, maintenance rules, "don't paste full prompts"). Extended repo-root `AGENTS.md` to register the `claude/` collection — added it to Structure, Component docs (pointing at `claude/README.md` + `claude/CLAUDE.md`), and "Working in this repo".
- **Why:** Closes K-004 — `tdd-engineer` (and cobb, dra-claudia) were invisible to both humans browsing and other agents; no catalog or context entry existed and root `AGENTS.md` covered only OpenCode/salesperson. Satisfies cobb's dual-audience documentation rule.
- **Plan items:** K-004 ✅ (closed, removed from active backlog). Shared deliverable also benefits cobb and dra-claudia.

## 2026-06-05 — Review #2 (no behavior change)
- **What:** Re-reviewed `tdd-engineer.md` at the user's request ("just review/advise"); no prompt edit. Re-verified K-004's discoverability claim against the repo: confirmed no `claude/README.md`, and root `AGENTS.md` documents only the OpenCode/salesperson components — the only `CLAUDE.md` lives under `opencode/agents/severino`, so the three Claude agents (cobb, dra-claudia, tdd-engineer) have no catalog and no context entry. Added **K-005** (workflow step 3 misses the "framework exists but tests can't run in this env — deps/toolchain not installed" case; risk of misreading an environmental failure as a code RED). Added a parking-lot idea on advanced test techniques (table-driven/property-based/mutation testing) as optional, low-priority enrichment.
- **Why:** User chose the review-only path; kaizen rules say record new findings in plan.md even without implementing. Verified rather than assumed the still-open items remain accurate.
- **Plan items:** added K-005; re-verified/annotated K-004; K-003 unchanged.

## 2026-06-05 — Implemented K-001 (test altitude) and K-002 (cold-start/red-baseline)
- **What:** Edited `tdd-engineer.md`. **K-001:** reconciled the "unit test" framing with the agent's broader stated scope — dropped "unit" from the `description` ("failing test", "add/improve tests"); reworded the RED step to "smallest possible test — a unit test by default; reach for an integration or contract test only when that's the genuine seam"; added a new **Right altitude of test** principle (smallest honest test, write real integration/contract tests at seams a unit can't reach, prefer many fast units + a thin higher-level layer); softened the isolation principle to "Fast, isolated, deterministic **by default**" with a note that integration/contract tests are deliberately slower/broader but still deterministic. **K-002:** rewrote workflow step 3 from "Find the test command" to "Establish a green baseline" with two explicit branches — greenfield (set up the minimal runner first as its own announced step, confirm a real baseline) and suite-already-red-on-arrival (stop, report failures, ask fix-first vs. proceed; never build on red or misattribute a pre-existing failure).
- **Why:** User asked to implement backlog items K-001 and K-002. K-001 removed a wording tension that could push the agent to force-fit unit tests where a higher-level test is the honest seam; K-002 closed the cold-start and red-baseline gaps the original step 3 silently assumed away.
- **Plan items:** K-001 ✅, K-002 ✅ (both closed, removed from the active backlog).

## 2026-06-05 — Bootstrapped kaizen + first review (no behavior change)
- **What:** Created `tdd-engineer/kaizen/plan.md` and `history.md` (the agent predated the kaizen convention and had neither). Conducted a review of `tdd-engineer.md` without editing the prompt. Seeded the backlog with K-001 (unit-vs-broader-scope tension), K-002 (cold-start / red-baseline handling), K-003 (tool-permissions decision), K-004 (catalog/discoverability gap), plus parking-lot ideas (no-auto-commit note, coverage-as-guide, flaky-test handling, opus-vs-sonnet cost).
- **Why:** User asked to work on the agent and chose "just review, advise" + "bootstrap kaizen files." Review found the prompt fundamentally solid (clean frontmatter, tight red/green/refactor loop, strong guardrails and anti-hallucination stance); the actionable findings are scope-framing, cold-start coverage, and housekeeping/discoverability — captured as backlog rather than applied.
- **Plan items:** seeded K-001, K-002, K-003, K-004.

## 2026-05-29 — Agent created (retroactively logged)
- **What:** Initial authoring of the `tdd-engineer` agent — a senior engineer that implements features and fixes strictly via Test-Driven Development (red → green → refactor). Frontmatter `name: tdd-engineer`, `model: opus`, routing-oriented `description` with proactive-use triggers (implement a feature, fix a bug, refactor with a safety net, add/improve tests). Body covers the TDD loop, principles, invocation workflow, communication style, and guardrails.
- **Why:** User wanted a dedicated test-first engineer agent (memory: "Prefers TDD"). Logged here retroactively since the agent predates the kaizen convention; date approximated from the source file's mtime.
- **Plan items:** —
