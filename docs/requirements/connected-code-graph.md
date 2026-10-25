# Connected code graph — Feature Requirements

> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (not yet filed) · **Last updated:** 2026-10-25

## Intent

The stakeholder wants the team's code graphs (CPGs) to be **connected across boundaries**, so that a
question like "if I change this server API, which storefront screens are affected?" can be answered
end to end, not stopped where one language or component hands off to another.

The underlying goal is a **robust system that can be easily queried** over that connected view. The
trigger is the stakeholder's workplace, where cross-component breakages "happen all the time". The
system must work on **any codebase**, the workplace's included, with this monorepo as the first
proving ground.

The connected view must serve **five use cases**, and the stakeholder wants to make sure the team
actually benefits from each one:

1. **Impact mapping**: list everything a change affects.
2. **Root-cause analysis**: from a symptom or error message, build a tree of hypotheses to be
   evaluated further.
3. **Code review**: for a change under review, find its impacted callers and the like.
4. **Test coverage map**: what code tests reach, and what they don't.
5. **Security**: security analysis over the connected view.

## Problem & current state

- Only one CPG is loaded today: `cpg_falkorchat` (Python, `falkor-chat/server`, built 2026-10-15).
- CPGs are built one per component and, in practice, one per language. The current frontend
  (`salesperson/`, React/TypeScript) has no CPG. The earlier Python Streamlit frontend had one
  (`cpg_salesperson`), but that app was retired and its graph is no longer loaded. Frontend coverage
  lapsed through that history, not through any recorded decision.
- Even with a CPG per language, the graphs **do not link to each other**: when one component calls
  another over HTTP (or another protocol), there is no call edge between them, so impact or data-flow
  questions stop at that boundary.
- Today's single-graph `cpg-analysis` skill already carries recipes for four of the five use cases
  (impact, RCA, code review, test gap; `skills/cpg-analysis/references/`), and `security-expert` uses
  it for data-flow analysis. All of them stop at the edge of one graph, and the stakeholder's concern
  is that the team is not getting the full benefit of any of them.
- Context for the architect (not a requirement): the installed Joern ships a JS/TS frontend
  (`jssrc2cpg`), but every `cpg-analysis` recipe and schema note is verified only on `pysrc2cpg`-built
  graphs (`skills/cpg-analysis/SKILL.md`, `skills/joern-cpg/references/cpg-model.md`).

## Guiding principles

- **Prefer FalkorDB.** FalkorDB (and Cypher) is the default store and query language, consistent
  with the rest of this lab. The architect may choose otherwise, with a stated reason.
- **Prefer Joern for the per-language code map.** Joern CPGs, the team's current toolchain, are the
  default way to map the code inside each program. The architect may use something else, or
  something alongside, with a stated reason. Context (not a requirement): the installed Joern ships
  frontends for all seven FR-1 languages, but builds one program in one language at a time, so
  cross-program links (FR-2, FR-22), code→data links (FR-3), config (FR-5), runtime coverage
  (FR-14), and dependency/secrets findings (FR-15) need more than Joern alone. Team recipes are
  verified only on Python CPGs.
- **Extend by addition.** New languages, protocols, data stores and in-house conventions are added,
  never designed in. Every use case benefits from each addition automatically.
- **Correctness over speed.** A slower, complete, correct answer beats a fast, partial one. No hard
  response-time limits.
- **Foundation first.** Delivery may be phased, but the shared foundation (the connected view and
  how it is queried) and these principles are settled first, so the chosen architecture suits all
  five use cases rather than only the first phase.

## User stories

- **Impact:** as a developer about to change code, I want to list everything the change affects,
  across components, languages, data, wiring and tests, so that breakages elsewhere are caught
  before they ship.
- **RCA:** as someone facing a symptom or an error message, I want a tree of plausible causes
  traced through the connected code, so that I know what to evaluate first.
- **Review:** as a reviewer, I want a change's impacted callers (and other consequences) across
  boundaries, so that I review what the change really touches, not just the diff.
- **Test map:** as a developer or QA, I want to see which code is reached by tests and which isn't,
  across boundaries, so that test gaps are visible.
- **Security:** as a security reviewer, I want security analysis that follows code across
  boundaries, so that risks spanning several components are not missed.
- As `teco` coordinating a change, I want impact answers available to the specialists I dispatch at
  the points in the development flow where they apply, so that impact is checked as part of the
  normal flow rather than on request.
- As someone adopting this on a different codebase, I want to point it at that codebase without
  hand-describing how its parts connect, so that it works beyond this monorepo.

## Functional requirements

- FR-0 — Works on any codebase, not only this monorepo: nothing about this repo's layout, languages,
  or components is assumed. This monorepo is the first proving ground.
- FR-1 — Every non-deprecated component's code is represented. Languages that must be supported:
  Python, TypeScript/JavaScript, Java, Kotlin, C#/.NET, Go, C/C++.
- FR-2 — Wherever one component depends on another, the two sides are linked so a question can be
  followed across that boundary. Kinds that must be covered: REST/HTTP, gRPC/RPC, messaging/events
  (publisher ↔ consumer of the same queue/topic), shared libraries (library ↔ every consumer), and
  this repo's MCP tool calls. **Tests are linked too:** each test is linked to the code it
  exercises, across boundaries.
- FR-3 — Code is linked to the stored data it reads and writes: SQL databases (tables/columns,
  migrations included), graph databases such as FalkorDB (labels/relationships), and NoSQL stores
  and caches.
- FR-4 — Agent and skill scripts/hooks under `claude/` and `skills/` are covered and linked to the
  components they call.
- FR-5 — Config and wiring files that decide which component talks to which at runtime (`.mcp.json`,
  Compose files, start scripts) are covered and reflected in the links.
- FR-6 — All five use cases are used inside the `teco`-maintained development flow wherever they
  apply, not only on direct request.
- FR-7 — The connected view is refreshed periodically ("every so often"); it need not track every
  commit. A refresh can be triggered by a schedule, on demand (a person, agent, or `teco`), by a
  staleness threshold (the code has moved on too far since the last refresh), or by a CI event. A
  full rebuild on each refresh is acceptable. How stale the view is must be knowable when it is used.
- FR-7a — If one repository or component fails to build during a refresh, the rest still refreshes.
  The failed part keeps its last good version, and every answer touching it says that part is older
  and why.
- FR-8 — When an impact cannot be proven (e.g. a runtime-built URL, a dynamic import), it is still
  listed, marked as *possible* with the reason for doubt. Missing a real impact is worse than a
  false alarm.
- FR-9 — Whether analysis runs locally or uses hosted services (including LLMs) is
  **configurable**. Both must be supported. Each codebase has a default setting, and a single run
  may override it, except that a codebase set to local-only can never be overridden into using
  hosted services. When local-only applies, no code leaves the machine.
- FR-10 — For each use case, a person can ask in plain language and get a grouped answer
  (by component; confirmed vs. possible; the reason for each link). The answer can optionally be saved
  as a written report.

**Querying:**

- FR-16 — The connected view can be queried by four kinds of client: agents (through an MCP tool),
  people in plain language, automation/CI without a person in the loop, and power users writing raw
  queries for questions no built-in answer covers.
- FR-17 — A question can start from any of: a code location (symbol name, or file + line); a change
  (diff, commit, branch, or PR); a contract point (API route/URL, gRPC method, queue/topic, table/
  column or graph label, config key); or failure evidence (error message, stack trace, log excerpt,
  test name).
- FR-18 — Asking never requires knowing which component, language, or underlying graph the code
  lives in, nor the internal structure of the stored view. Only power users' raw queries (FR-16)
  touch those internals.
- FR-19 — The same question against the same snapshot of the code returns the **same facts**
  (the listed impacts, causes, gaps, findings) for every client. Plain-language wording around them
  may vary. One exception: RCA's hypothesis set must be identical, but its ranking order may shift
  slightly between runs.
- FR-20 — Every reported item carries its **full chain of evidence**: each link from the starting
  point to the item, with file and line at every step, so anyone can verify it.
- FR-21 — No fixed response-time budget: time may grow with the code's size and the depth of the
  analysis. Correctness takes priority over speed (Guiding principles).
- FR-22 — A system may consist of **many repositories** (e.g. a front end, several services and a
  shared library, each in its own repo). Links and answers cross repository boundaries exactly as they
  cross component boundaries.
- FR-23 — Questions run against the latest refreshed snapshot, **or the latest snapshot with a
  proposed change (diff/PR) overlaid**, so un-merged work can be checked for impact and reviewed.
- FR-24 — Answers respect the asker's repository access: an answer includes only repositories the
  asker may read. An impact in a repository they cannot read is reported only as "affected, no
  access", with no code shown.
- FR-25 — Raw queries (FR-16) are read-only: they can never change the stored view.

**Trust in links:**

- FR-28 — A designated person (a named owner or reviewer of the repository concerned) can confirm a
  *possible* link (FR-8) or dismiss it as a false alarm, with a reason. The verdict and who gave it
  are shown in later answers, and survive refreshes (including full rebuilds).
- FR-29 — A dismissed link is left out of normal answers but stays listable on request ("show
  dismissed"), with who dismissed it, when, and why. A dismissal **expires when the code at either
  end of the link changes**: the link returns as *possible*, noting the earlier dismissal, for a
  fresh verdict.
- FR-30 — Recording verdicts is separate from raw queries, which stay read-only (FR-25).

**Extensibility:**

- FR-26 — Each language, protocol, and data store is an addable unit. Adding one never requires
  changing the core or the units already supported, and every use case (FR-11 to FR-15) and every
  query client (FR-16) works on it automatically, with no per-use-case changes.
- FR-27 — A team can describe its own in-house conventions (e.g. a homegrown RPC library or event
  bus) declaratively, without changing the system's code, and the matching links appear. This
  supplements FR-0: standard, widely used libraries and frameworks are recognized with no
  description at all.

**Use cases:**

- FR-11 — **Impact mapping:** given a code element or a change, list everything it affects.
- FR-12 — **Root-cause analysis:** accepts any of an error message, a stack trace, a log excerpt
  (possibly spanning several services), or a plain-language symptom. Produces a hypothesis tree of
  candidate causes, each traced to the code involved and **ranked by likelihood with its evidence**.
  Evaluating the ranked hypotheses is left to a person or an agent.
- FR-13 — **Code review:** given a change under review, list its impacted callers and other affected
  code across boundaries.
- FR-14 — **Test coverage map:** show which code is reached by tests and which is not, from static
  analysis (following calls and links) **merged with runtime coverage data from test runs where
  available**. Places where the two disagree are shown.
- FR-15 — **Security:** findings that follow code across boundaries, covering: untrusted input
  reaching a dangerous operation (SQL, shell, file, eval) in another component; paths to sensitive
  operations or data that skip an authorization check; secrets in code/config or flowing into logs,
  responses, or other services; known-vulnerable dependencies, stating whether the vulnerable code
  is actually reachable.

## Out of scope

- `deprecated/`: retired components are not mapped or linked.
- Tracking every commit: periodic refresh is enough (FR-7).
- Querying past snapshots or comparing two snapshots (FR-23 covers latest + a proposed change only).
- Fixing what it finds: the system reports impacts, causes, gaps and risks. It does not change code
  or open tickets.

## Acceptance criteria

_Proposed. The stakeholder will review this list after the expert reviews and architecture proposal
(see Open questions). AC-1 to AC-6 exercise impact mapping (FR-11)._

- AC-1 (HTTP) — Given this monorepo, when asked what changing a `falkor-chat` server endpoint used by
  the storefront affects, then the `salesperson/` code that calls it is listed.
- AC-2 (MCP + wiring) — When asked about changing a `falkor-chat` MCP tool, then its callers (e.g.
  `kiro/`, `mcp-monitor/` configuration) and the wiring files that connect them are listed.
- AC-3 (code→data) — When asked about renaming a FalkorDB label, then every function that reads or
  writes it, in any component, is listed.
- AC-4 (scripts) — When asked about changing a `cypher-mcp` tool, then the agent/skill scripts and
  hooks under `claude/`/`skills/` that call it are listed.
- AC-5 (never miss) — Given a seeded list of known cross-boundary impacts in the proving repo, every
  one appears in the answer, as confirmed or possible.
- AC-6 (doubt flagged) — A link the system cannot prove is shown as *possible* with its reason.
- AC-7 (any codebase) — Given a second, unfamiliar multi-service codebase using at least two of the
  required languages and gRPC, messaging, and a SQL database, the system builds its connected view
  without anyone hand-describing how the parts connect, and AC-5 holds there too.
- AC-8 (configurable) — A local-only codebase sends no code off the machine, even when a run
  requests hosted services. A codebase defaulting to hosted can be run locally by a per-run override.
- AC-9 (staleness) — Every answer states when the view was last refreshed.
- AC-9a (refresh triggers) — A refresh starts from each of the four triggers in FR-7.
- AC-9b (partial failure) — Given a refresh where one repository fails to build, the others are
  updated, and an answer crossing the failed repository states it is from the previous refresh and
  why.
- AC-10 (in the flow) — In a `teco`-coordinated change, the specialists' briefs/reviews at the
  applicable stages carry the impact answer for the code being changed.
- AC-11 (answer forms) — For each use case, a plain-language question returns a grouped answer,
  and the same answer can be saved as a markdown report.
- AC-12 (RCA) — Given a bug seeded in the proving repo whose cause lies in a different component
  from where it surfaces, then starting from each of the four input forms the true cause appears in
  the hypothesis tree, each hypothesis cites the code involved, and the ranking shows its evidence.
- AC-13 (review) — Given a change to a server endpoint's contract, the review answer lists its
  callers in other components and the tests that exercise it.
- AC-14 (test map) — The map identifies production code reached by no test, including code only
  reachable across a boundary. When runtime coverage data is supplied, it is merged in and
  static/runtime disagreements are listed.
- AC-15 (security) — Given seeded examples of each of the four finding kinds in FR-15, at least one
  spanning two components, each is reported. A vulnerable dependency's report states whether the
  vulnerable code is reachable.
- AC-16 (clients) — The same impact question, asked by an agent through MCP, by a person in plain
  language, by a CI script, and as a raw query, returns the same facts (FR-16, FR-19).
- AC-17 (starting points) — A question starting from each of the four groups in FR-17 (e.g. a file +
  line, a PR, a queue name, a stack trace) is answered without the asker naming a component,
  language, or graph (FR-18).
- AC-18 (evidence) — Every item in every answer shows its full chain of links, with file and line at
  each step (FR-20).
- AC-19 (multi-repo + overlay) — Given a system split across at least three repositories, an impact
  question about an un-merged change in one repository lists affected code in the others (FR-22,
  FR-23).
- AC-20 (access) — An asker without read access to one of those repositories sees its impacts only
  as "affected, no access", with no code shown (FR-24).
- AC-21 (read-only) — A raw query that tries to change the stored view is rejected and the view is
  unchanged (FR-25).
- AC-21a (verdicts) — A designated person dismisses a possible link with a reason; later answers
  omit it, "show dismissed" lists it with who, when and why, and it is still dismissed after a full
  refresh. After the code at one end changes and the view is refreshed, the link is back as
  possible, noting the earlier dismissal. A person who is not designated for that repository cannot record a verdict (FR-28, FR-29).
- AC-22 (extension) — Adding support for one more language or protocol touches no existing unit
  and no use case, and all five use cases then answer across it (FR-26).
- AC-23 (conventions) — Given a codebase using an in-house messaging library, a declarative
  description of it, with no system code changed, makes publisher↔consumer links appear (FR-27).

## Open questions

1. Expert review pending: `architect` (architecture proposal), `graph-dba`, `data-scientist`, and
   `analyst` review the requirements as agreed so far
   (`docs/plans/connected-code-graph-coordination.md`). Their findings may reopen requirements.
2. Acceptance criteria: the stakeholder reviews AC-1 to AC-23 after the expert reviews.

## Decision log

- 2026-10-22 — Which outcome does fuller CPG coverage serve: frontend-internal impact checks, a
  cross-boundary view, or completeness? → **Cross-boundary**: "I want to connect everything together."
- 2026-10-22 — Prior-decision check: no requirement, plan, backlog item, or kaizen history records a
  decision for or against cross-language/cross-component CPG linking. This is a fresh request.
- 2026-10-22 — Which boundaries must be connected (offered: storefront→server HTTP, MCP clients→MCP
  servers, code→graph data, every component)? → **Every component**: every component in the monorepo
  gets a map, joined wherever one calls another. Edge cases still open: `deprecated/`, non-component
  code, and code→data links.
- 2026-10-22 — Borderline areas (offered: code→graph data, agent & skill scripts, `deprecated/`, config
  & wiring files)? → **In:** code→graph data, agent & skill scripts (`claude/`, `skills/`), config &
  wiring files (`.mcp.json`, Compose, start scripts). **Out:** `deprecated/`.
- 2026-10-22 — Trigger / who needs it? → Cross-component breakages "happen all the time where I
  work"; the goal is "a robust system that can be easily queried to enlist all impacts". The
  workplace mention may widen the target beyond this repo; logged as Open question 1, not assumed.
- 2026-10-22 — Target codebases? → **Any codebase**; this monorepo is the first proving ground.
- 2026-10-22 — Who asks? → Incorporated **where applicable into the development flow maintained by
  `teco`**.
- 2026-10-22 — How current? → **Every so often is fine**; per-commit tracking not required. FR-7's
  "staleness knowable" clause is carried over from the existing CPG freshness practice. Confirm at
  readback.
- 2026-10-22 — Uncertain impacts? → **Over-list, flag doubt** (FR-8).
- 2026-10-22 — Confidentiality? → **Local only; a local LLM is allowed**, no hosted services (FR-9).
- 2026-10-22 — Workplace languages? → **Java/Kotlin, C#/.NET, Go, plus Python and TS/JS** (FR-1).
- 2026-10-22 — Workplace connection kinds? → **REST/HTTP, gRPC/RPC, messaging/events, shared
  libraries** (FR-2, plus this repo's MCP).
- 2026-10-22 — Data stores for code→data links? → **SQL, graph (FalkorDB etc.), NoSQL/caches** (FR-3).
- 2026-10-22 — Answer form? → **Both**: conversational answer, optionally saved as a report (FR-10).
- 2026-10-25 — Readback revision by stakeholder: the connected view must serve **five use cases**:
  impact mapping, root-cause analysis (symptom/error → hypothesis tree for further evaluation), code
  review (impacted callers etc.), test coverage map, security (FR-11 to FR-15).
- 2026-10-25 — FR-1 gains **C/C++**.
- 2026-10-25 — FR-9 reversed: previously local-only with a local LLM allowed (2026-10-22). Now
  **local or hosted, configurable**. Out-of-scope "hosted services" line dropped; AC-8 rewritten.
- 2026-10-25 — FR-2 gains **links to tests**: each test is linked to the code it exercises.
- 2026-10-25 — RCA inputs? → **All four**: error message, stack trace, log excerpt, described
  symptom. Evaluation? → **Auto-ranked first** (likelihood + evidence), then a person or agent
  evaluates (FR-12).
- 2026-10-25 — Security finding kinds? → **All four**: cross-service taint, missing auth checks,
  secrets exposure, vulnerable dependencies with reachability (FR-15).
- 2026-10-25 — Coverage evidence? → **Static + runtime**, merged, disagreements shown (FR-14).
- 2026-10-25 — FR-9 configuration level? → **Per-codebase default plus per-run override**; a
  local-only codebase can never be overridden into hosted.
- 2026-10-25 — Readback response: phasing is acceptable, **but a strong foundation and principles
  come first** so the architecture suits every use case (Guiding principles). Stakeholder opened a new
  thread on how the view is queried. Readback not yet confirmed.
- 2026-10-25 — Who queries directly? → **All four**: agents via MCP, people in plain language,
  automation/CI, power users with raw queries (FR-16).
- 2026-10-25 — Is FalkorDB + Cypher fixed? → **Prefer FalkorDB**: the default, but the architect may
  deviate with a stated reason (Guiding principles).
- 2026-10-25 — Question starting points? → **All four groups**: code locations, changes, contract
  points, failure evidence (FR-17).
- 2026-10-25 — Must askers know component/language/graph/schema? → **No, never**; only raw queries
  touch internals (FR-18).
- 2026-10-25 — Same question, same snapshot? → **Same facts, wording may vary** (FR-19).
- 2026-10-25 — Evidence per item? → **Always, full chain** with file/line per step (FR-20).
- 2026-10-25 — Speed? → **Depends on use case** (FR-21). Budgets still to be set.
- 2026-10-25 — Response-time budgets (offered three sets)? → **None of them**: "no hard limit,
  depend on actual code and analysis depth, should prioritize correctness" (FR-21, Guiding
  principles).
- 2026-10-25 — Is RCA ranking order a fixed fact? → **Hypothesis set fixed, order may shift** (FR-19).
- 2026-10-25 — Span several repositories? → **Yes, many repos** (FR-22).
- 2026-10-25 — Queryable versions? → **Latest + a proposed change overlaid** (FR-23); past snapshots
  and comparisons out of scope.
- 2026-10-25 — Respect asker's access? → **Respect repo access**; hidden repos reported only as
  "affected, no access" (FR-24).
- 2026-10-25 — Raw queries? → **Read-only** (FR-25).
- 2026-10-25 — Extensibility? → **Plug-in, and no re-teaching**: every use case and client works on a
  new unit automatically (FR-26, Guiding principles).
- 2026-10-25 — In-house conventions? → **Yes, declaratively**, no system code changes (FR-27).
- 2026-10-25 — Refresh triggers? → **All four**: schedule, on demand, staleness threshold, CI event
  (FR-7).
- 2026-10-25 — One part fails to build? → **Keep last good, mark it** (FR-7a).
- 2026-10-25 — Full or incremental? → **Full rebuild fine** (FR-7).
- 2026-10-25 — Verdicts on possible links? → **Confirm or dismiss** (FR-28).
- 2026-10-25 — Dismissed links later? → **Hidden, recoverable** via "show dismissed" (FR-29).
  The expire-on-change option was not chosen; flagged as an open question.
- 2026-10-25 — Who records verdicts? → **Designated people** per repository (FR-28).
- 2026-10-25 — Dismissal expiry? → **Yes, expires when code at either end changes**; the link returns
  as possible, noting the earlier dismissal (FR-29). Open question closed.
- 2026-10-25 — Is the HOW Joern CPG? → Requirements had not named it. Stakeholder chose **Prefer
  Joern**: the default per-language code map, deviation allowed with a stated reason (Guiding
  principles).
- 2026-10-25 — Readback: **"no fixing" confirmed** (Out of scope). Not yet Ready for design. Before
  reviewing the acceptance criteria, the stakeholder wants `architect`, `analyst`, `graph-dba`, and
  `data-scientist` to review what was agreed and suggest an implementation architecture. `tico` runs
  this as a docs-only coordination (`docs/plans/connected-code-graph-coordination.md`).
