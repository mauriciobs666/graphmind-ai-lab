# Connected code graph — Feature Requirements

> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (not yet filed) · **Last updated:** 2026-10-26

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
  something alongside, with a stated reason. Stakeholder note: Joern's control-flow and
  data-dependency relationships should be used to track impacts for all use cases, not only its
  call graph. Context (not a requirement): the installed Joern ships
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
- **Delivery preferences (stakeholder decisions, not design).** The system grows the existing
  `cpg/` component rather than starting a new one. Plain-language questions (FR-10) are part of the
  first delivery phase. Blind answer keys for evaluation are written by an isolated agent with
  read-only code access and no view of the system's rules or output, spot-checked by the
  stakeholder.

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
  or components is assumed. This monorepo is the first proving ground. Listing which repositories
  form a system (plus exclusions) is acceptable input; describing how the parts talk to each other
  is not.
- FR-1 — Every non-deprecated component's code is represented. Languages that must be supported:
  Python, TypeScript/JavaScript, Java, Kotlin, C#/.NET, Go, C/C++, plus shell scripts and
  structured config (JSON, YAML, TOML, Compose files).
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
  Compose files, start scripts) are covered and reflected in the links. An agent prompt that names
  a tool, and an example/template config, count as *possible* callers with that reason; code
  snippets inside documentation do not count.
- FR-6 — All five use cases are used inside the `teco`-maintained development flow wherever they
  apply, not only on direct request.
- FR-7 — The connected view is refreshed periodically ("every so often"); it need not track every
  commit. A refresh can be triggered by a schedule, on demand (a person, agent, or `teco`), by a
  staleness threshold (the code has moved on too far since the last refresh), or by a CI event. A
  full rebuild on each refresh is acceptable. How stale the view is must be knowable when it is used.
- FR-7a — If one repository or component fails to build during a refresh, the rest still refreshes.
  The failed part keeps its last good version, and every answer touching it says that part is older
  and why. A part that has **never** built is reported as "not mapped", and anything crossing into
  it is shown as *possible* with that reason.
- FR-8 — When an impact cannot be proven (e.g. a runtime-built URL, a dynamic import), it is still
  listed, marked as *possible* with the reason for doubt. Missing a real impact is worse than a
  false alarm.
- FR-9 — Whether analysis runs locally or uses hosted services (including LLMs) is **configurable**.
  Both must be supported. Each codebase has a default setting, and a single run may override it,
  except that a codebase set to local-only can never be overridden into using hosted services. When
  local-only applies, **nothing derived from the repositories** leaves the machine — source, names,
  paths, error messages, stack traces, dependency lists. "The machine" is an explicit, configured
  allow-list of local endpoints (e.g. LM Studio on the Windows host under WSL); anything not on it
  counts as hosted. Building never downloads dependencies (it uses caches prepared in advance, or
  accepts less precise analysis without them). Local-only is **sticky**: any answer that touches a
  local-only repository is handled entirely locally. A weaker local model may need more rephrasing
  of questions, but never changes the facts returned.
- FR-10 — For each use case, a person can ask in plain language and get a grouped answer
  (by component; confirmed vs. possible; the reason for each link). The answer can optionally be saved
  as a written report.

**Querying:**

- FR-16 — The connected view can be queried by four kinds of client: agents (through an MCP tool),
  people in plain language, automation/CI without a person in the loop, and power users writing raw
  queries for questions no built-in answer covers.
- FR-17 — A question can start from any of: a code location (symbol name, or file + line); a change
  (diff, commit range, or branch — a PR from a hosting service is an optional add-on); a contract
  point (API route/URL, gRPC method, queue/topic, table/ column or graph label, config key); or
  failure evidence (error message, stack trace, log excerpt, test name).
- FR-18 — Asking never requires knowing which component, language, or underlying graph the code
  lives in, nor the internal structure of the stored view. Only power users' raw queries (FR-16)
  touch those internals.
- FR-19 — The same question against the same snapshot — the same code **and** the same set of
  recorded confirm/dismiss decisions (FR-28) — returns the **same facts** (the listed impacts,
  causes, gaps, findings) for every client. Plain-language wording around them may vary. This
  includes RCA's ranking: the same failure evidence yields the same hypotheses **in the same order**
  (the ranking is a stated, repeatable score). Every answer publishes its **recipe** — the exact
  queries it ran plus the name and version of the algorithm that combined them — and for a raw
  query, "same facts" means re-running that recipe on the same snapshot reproduces them. "Same
  question" means the question **as the system understood it**: every answer opens with that
  interpretation, and an ambiguous question is answered with the possible readings rather than a
  silent guess. Every answer is stamped with the code snapshot and decision state it reflects.
- FR-20 — Every reported item carries its **full chain of evidence**: each link from the starting
  point to the item, with file and line at every step, so anyone can verify it.
- FR-21 — No fixed response-time budget: time may grow with the code's size and the depth of the
  analysis. Correctness takes priority over speed (Guiding principles).
- FR-22 — A system may consist of **many repositories** (e.g. a front end, several services and a
  shared library, each in its own repo). Links and answers cross repository boundaries exactly as they
  cross component boundaries.
- FR-23 — Questions run against the latest refreshed snapshot, **or the latest snapshot with a
  proposed change (diff/PR) overlaid**, so un-merged work can be checked for impact and reviewed.
  The number of concurrent change previews is capped (configurable, default about 4; the oldest
  idle one is retired first).
- FR-24 — Answers respect the asker's repository access: an answer includes only repositories the
  asker may read. An impact in a repository they cannot read is reported only as "affected, no
  access", with no code shown. When an agent or CI job asks, the access that applies is that of the
  person who launched it, or of a configured service identity with its own grants. What is protected
  is **content** — code, file contents, internal names; the asker may learn that something there is
  affected and how much. Raw queries (FR-16) are available only to people who can read every
  repository in the system.
- FR-25 — Raw queries (FR-16) are read-only: they can never change the stored view. This is
  enforced by the database for the raw-query credential, not only by the tool in front of it.

**Trust in links:**

- FR-28 — A designated person (a named owner or reviewer of the repository concerned, listed in a
  CODEOWNERS-style file; identified by the OS user when local — honest users only — and by the
  company identity provider when hosted) can confirm a *possible* link (FR-8) or dismiss it as a
  false alarm, with a reason. The verdict and who gave it are shown in later answers, take effect
  **immediately**, and survive refreshes (including full rebuilds). For a link between two
  repositories, an owner of either side may give the verdict, and it is shown to both sides.
- FR-29 — A dismissed link is left out of normal answers but stays listable on request ("show
  dismissed"), with who dismissed it, when, and why. A verdict — dismissal **or confirmation** —
  **expires when the code at either end of the link changes** ("change" means the enclosing
  function or config entry): the link returns as *possible*, noting the earlier verdict, for a
  fresh one. When code is renamed, moved, or its signature changes, the system cannot prove it is
  the same code: the link returns as *possible* without that note, and the old verdict stays
  listable.
- FR-30 — Recording verdicts is separate from raw queries, which stay read-only (FR-25).

**Extensibility:**

- FR-26 — Each language, protocol, and data store is an addable unit. Adding one never requires
  changing the core or the units already supported, and every use case (FR-11 to FR-15) and every
  query client (FR-16) works on it automatically, with no per-use-case changes. This holds within
  the shared vocabulary of concepts (calls, endpoints, topics, tables, …); a genuinely new concept
  is a deliberate, versioned core change.
- FR-27 — A team can describe its own in-house conventions (e.g. a homegrown RPC library or event
  bus) declaratively, without changing the system's code, and the matching links appear. This
  supplements FR-0: standard, widely used libraries and frameworks are recognized with no
  description at all.

**Use cases:**

- FR-11 — **Impact mapping:** given a code element or a change, list everything it affects.
  - Impact follows **calls, data flow and control dependencies** — e.g. a changed value flowing
    into other code, or a changed condition guarding other code — not only who calls whom.
  - Each item is marked **direct or indirect** and shows its **distance** (number of links) from
    the starting point.
  - The list is **grouped** by repository and component, then ordered by distance, with confirmed
    items before possible ones.
  - The **reverse question** is also answered: "what does this code depend on?", with the same
    marking and grouping.
- FR-12 — **Root-cause analysis:** accepts any of an error message, a stack trace, a log excerpt
  (possibly spanning several services), or a plain-language symptom. Produces a hypothesis tree of
  candidate causes, each traced to the code involved and **ranked by a stated score, with its
  evidence** (the factors behind each rank are shown; no claimed probabilities).
  Evaluating the ranked hypotheses is left to a person or an agent.
  - Several pieces of evidence can be given **at once** (e.g. a stack trace, a log excerpt and a
    description), and they reinforce each other.
  - Candidate causes include **config values, code↔data schema mismatches and wiring errors**
    (wrong port, wrong topic), not only code — each traced to its file or entry.
  - Causes are traced **backwards through data flow and control dependencies** — where a wrong
    value came from, which condition chose a wrong branch — not only up the call stack.
  - **Recent changes** are evidence: code changed within a window set per codebase (its own setting,
    separate from the secrets-history window) is marked, with when it changed.
  - The tree always includes an explicit **"cause may lie outside the mapped code"** branch
    (infrastructure, environment, third parties) with the evidence pointing there, rather than
    forcing a code hypothesis.
  - The asker can mark hypotheses **ruled out or confirmed** and get a re-ranked, expanded tree.
    These marks apply to that investigation only; nothing persists.
  - When the evidence matches nothing in the mapped code, the answer says so, shows what was tried,
    and suggests what further evidence would help.
- FR-13 — **Code review:** given a change under review (normally un-merged, FR-23), list across
  boundaries:
  - its impacted callers and other affected code;
  - **broken contracts of every kind** — a renamed or removed endpoint, message topic, table/column
    or graph label, MCP tool, or shared-library signature — with every consumer of the old one;
  - **test impact** — the tests that exercise the changed code, and impacted code no test reaches;
  - **security impact** — security findings (FR-15) the change introduces or touches;
  - **config and wiring** — config, start-script and Compose entries that reference what changed.

  The answer focuses on what the change **introduces or makes worse**; issues already present in
  the affected code before the change are available on request ("show existing").
- FR-14 — **Test coverage map:** show which code is reached by tests and which is not, from static
  analysis (following calls and links) **merged with runtime coverage data from test runs where
  available**. Places where the two disagree are shown. Runtime data must be **per test** (which
  test ran which code), not only suite-wide totals. For a project that cannot supply per-test data,
  only static coverage is shown, marked "runtime data unavailable (needs per-test coverage)".
  - "Reached" is judged at **function level**.
  - Each test is marked by **kind**: unit, integration, or end-to-end.
  - A direct question answers **"which tests does this change reach?"**, so the right tests can be
    run for a change outside a full review.
  - Untested code is **ranked by risk** — how many callers reach it, whether it is
    security-relevant, whether it changed recently — with the reason shown.
  - Runtime coverage collected on a different commit from the snapshot is **flagged as stale**,
    never silently merged.
- FR-15 — **Security:** findings that follow code across boundaries, covering: untrusted input
  reaching a dangerous operation (SQL, shell, file, eval) in another component; paths to sensitive
  operations or data that skip an authorization check; secrets in code/config or flowing into logs,
  responses, or other services; known-vulnerable dependencies, stating whether the vulnerable code
  is actually reachable — at package level always, and at function level where the advisory names
  the vulnerable functions; and **risky configuration** in config, Compose and start-script files
  (e.g. a database port exposed publicly, debug mode on, a privileged container). Secrets are also
  sought in the repository's **git history**, over a window configurable per codebase (last N
  months or since a date) — a secret committed and later removed is reported, with advice to rotate
  it.
  - Every finding carries a **severity** on a four-level scale (critical, high, medium, low) with
    its reason: vulnerable dependencies map from their advisory score, other kinds follow stated
    rules.
  - A designated person can confirm a finding or dismiss it as a false alarm, under the same rules
    as links (FR-28, FR-29): shown with who/when/why, listable when dismissed, expiring when the
    code at the finding changes.
  - The system **never stores or shows a secret's value**: a secret is identified by its location
    and a fingerprint only.

## Out of scope

- `deprecated/`: retired components are not mapped or linked.
- Tracking every commit: periodic refresh is enough (FR-7).
- Querying past snapshots or comparing two snapshots (FR-23 covers latest + a proposed change only).
  The exceptions are the secrets scan of git history (FR-15) and RCA's recent-changes window
  (FR-12).
- Fixing what it finds: the system reports impacts, causes, gaps and risks. It does not change code
  or open tickets.

## Acceptance criteria

_All criteria were reviewed with the stakeholder on 2026-10-26 (decisions S1–S38, then the
remaining criteria in three groups). Targets marked provisional are ratified when the phase that
builds them has measured reality. AC-1 to AC-6 exercise impact mapping (FR-11)._

- AC-1 (HTTP) — Given this monorepo, when asked what changing a `falkor-chat` server endpoint used by
  the storefront affects, then the `salesperson/` code that calls it is listed.
- AC-2 (MCP + wiring) — When asked about changing a `falkor-chat` MCP tool, then its callers (e.g.
  `kiro/`, `mcp-monitor/` configuration) and the wiring files that connect them are listed.
- AC-3 (code→data) — When asked about renaming a FalkorDB label, then every function that reads or
  writes it, in any component, is listed.
- AC-4 (scripts) — When asked about changing a `cypher-mcp` tool, then the agent/skill scripts and
  hooks under `claude/`/`skills/` that call it are listed.
- AC-5 (never miss) — Given a **frozen, blind** list of at least 75 known cross-boundary impacts in
  the proving repo (written before the system sees it, never used for development), every one
  appears in the answer **individually**, with its evidence, as confirmed or possible. At least 95%
  of items shown as *confirmed* are real, and wrong *possible* items stay at or under 4 per real
  impact (provisional; ratified or adjusted once the first phase has measured the real ratio).
- AC-6 (doubt flagged) — Given planted links of named unprovable kinds (e.g. an address built at
  runtime, a topic read from config), each is shown as *possible* with its reason, and none is
  shown as confirmed.
- AC-7 (any codebase) — Given a second, unfamiliar multi-service codebase using at least two of the
  required languages and gRPC, messaging, and a SQL database, the system builds its connected view
  without anyone hand-describing how the parts connect, and AC-5 holds there too. The candidate is
  the OpenTelemetry demo; the system's configuration is frozen before first contact and the
  first-contact result is reported as-is. A third codebase, unseen by anyone building the system,
  is held back for a later final check.
- AC-8 (configurable) — A local-only codebase sends nothing derived from its repositories to any
  endpoint outside the local allow-list, even when a run requests hosted services, and an answer
  touching it alongside a hosted repository is handled entirely locally. A codebase defaulting to
  hosted can be run locally by a per-run override.
- AC-9 (staleness) — Every answer states when the view was last refreshed and which decision state
  (recorded verdicts) it reflects (FR-7, FR-19).
- AC-9a (refresh triggers) — A refresh starts from each of the four triggers in FR-7.
- AC-9b (partial failure) — Given a refresh where one repository fails to build, the others are
  updated, and an answer crossing the failed repository states it is from the previous refresh and
  why. Given a repository that has never built, it is reported as "not mapped" and anything crossing
  into it is shown as possible with that reason (FR-7a).
- AC-10 (in the flow) — In a `teco`-coordinated change, the specialists' briefs/reviews at the
  applicable stages carry the impact answer for the code being changed.
- AC-11 (answer forms) — For each use case, a plain-language question returns a grouped answer,
  and the same answer can be saved as a markdown report.
- AC-12 (RCA) — Given a bug seeded in the proving repo whose cause lies in a different component
  from where it surfaces, then starting from each of the four input forms the true cause appears in
  the hypothesis tree, each hypothesis cites the code involved, and the ranking shows its evidence.
  Measured over at least 40 such planted bugs: the true cause is in the top 5 for at least 70% of
  them from an error message, stack trace or log, and for at least 50% from a plain-language
  symptom, and the ranking beats a "closest to the error first" baseline (targets provisional,
  confirmed in the phase that builds RCA). Additionally: a planted cause in a config value, in a
  code↔data schema mismatch, and in wiring each appears as a hypothesis traced to its file or entry;
  a planted cause reachable only by following a value backwards through data flow is found; a
  planted recent change in the window is marked as recently changed; every tree has an "outside the
  mapped code" branch; giving a stack trace and a log together ranks the true cause at least as high
  as either alone; marking the top hypothesis ruled out yields a re-ranked tree without it; and an
  error message matching nothing returns an explicit "no match" with what was tried and what to
  provide next (FR-12).
- AC-13 (review) — Given one planted, un-merged change of each contract kind (an endpoint, a
  message topic, a table/column or graph label, an MCP tool, a shared-library signature), each
  review answer lists the consumers of the old contract in other components, the tests that
  exercise the changed code and any impacted code no test reaches, any security finding the change
  introduces, and the config/wiring entries that reference it. A planted pre-existing issue in the
  affected code is left out of the answer and appears under "show existing" (FR-13).
- AC-14 (test map) — The map identifies production code reached by no test, including code only
  reachable across a boundary. When per-test runtime coverage data is supplied, it is merged in and
  static/runtime disagreements are listed; otherwise the runtime half is marked unavailable. Every
  test is marked unit, integration, or end-to-end. Asking "which tests does this change reach?" for
  a planted change lists exactly the tests that exercise the changed functions. Untested functions
  are ranked by risk with the reason shown. Runtime coverage collected on a different commit is
  shown as stale, not merged (FR-14).
- AC-15 (security) — Given at least 10 real and 10 look-alike-safe seeded cases of each of the five
  finding kinds in FR-15, at least one spanning two components, the real ones are reported and the
  false-alarm rate on look-alikes is reported. A vulnerable dependency's report states whether the
  vulnerable code is reachable, and a reachable one is **never** reported as unreachable. Every
  finding shows a severity with its reason. A secret committed and later removed within the
  configured history window is reported, and no secret value appears anywhere in any answer or
  stored view. A dismissed false-positive finding is left out of later answers, is listable, and
  returns after the code at the finding changes.
- AC-16 (clients) — The same impact question, asked by an agent through MCP, by a person in plain
  language, by a CI script, and by re-running its published recipe as a raw query, returns the same
  facts (FR-16, FR-19).
- AC-17 (starting points) — A question starting from each of the four groups in FR-17 (e.g. a file +
  line, a branch, a queue name, a stack trace) is answered without the asker naming a component,
  language, or graph (FR-18).
- AC-18 (evidence) — Every item in every answer shows its full chain of links, with file and line at
  each step (FR-20).
- AC-19 (multi-repo + overlay) — Given a system split across at least three repositories, an impact
  question about an un-merged change in one repository lists affected code in the others (FR-22,
  FR-23).
- AC-20 (access) — An asker without read access to one of those repositories sees its impacts only
  as "affected, no access", with no code shown. The same holds when the question comes from an agent
  or CI job launched by that asker, and that asker is refused raw queries (FR-24, FR-16).
- AC-21 (read-only) — A raw query that tries to change the stored view is rejected by the database
  for the raw-query credential, and the view is unchanged (FR-25).
- AC-21a (verdicts) — A designated person dismisses a possible link with a reason; later answers
  omit it immediately, "show dismissed" lists it with who, when and why, and it is still dismissed
  after a full refresh. After the code at one end changes and the view is refreshed, the link is
  back as possible, noting the earlier dismissal. A confirmation expires the same way. After code at
  one end is renamed, moved, or has its signature changed, the link is back as possible without the
  earlier-verdict note, and the old verdict is still listable. On a link between two repositories,
  an owner of either side can record the verdict, and both sides see it. A person who is not
  designated for that repository cannot record a verdict (FR-28, FR-29).
- AC-22 (extension) — Adding support for one more language or protocol, using only concepts already
  in the shared vocabulary, touches no existing unit and no use case, and all five use cases then
  answer across it (FR-26).
- AC-23 (conventions) — Given a codebase using an in-house messaging library, a declarative
  description of it, with no system code changed, makes publisher↔consumer links appear (FR-27).
- AC-24 (understanding) — Given about 100 realistic plain-language questions, in both English and
  Portuguese, at least 92% are understood correctly (the interpretation shown at the top of the
  answer is right), and every ambiguous one is answered with its possible readings rather than a
  guess (FR-18, FR-19; provisional target).
- AC-25 (impact shape) — Given a planted change whose effect travels through a data flow (a changed
  return value used two calls away) and through a control dependency (a changed condition guarding
  other code), both effects are listed. Every item is marked direct or indirect with its distance,
  grouped by repository and component, then by distance, confirmed before possible. Asking the
  reverse question, "what does this depend on?", for the same function lists its dependencies the
  same way (FR-11).

## Open questions

None.

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
- 2026-10-26 — Expert review delivered (plan v2, `-graph` and `-ml` notes, `analyst` review:
  approve with changes). Its 38 deduplicated stakeholder decisions (review §5, S1–S38) are being
  taken theme by theme, as the AC review.
- 2026-10-26 — S7: raw queries read-only enforced by the database credential itself? → **Yes**
  (FR-25, AC-21). `graph-dba` showed the existing query tool is not reliably read-only.
- 2026-10-26 — S1: what makes a raw query "agree" with an answer? → **Re-running the answer's
  published recipe (queries + algorithm name/version) reproduces the facts** (FR-19, AC-16). A raw
  query written from scratch may legitimately differ for multi-step answers.
- 2026-10-26 — S2: what counts as "the same question"? → **The question as the system understood
  it, shown at the top of every answer**; ambiguity answered with options (FR-19). How often wording
  is understood correctly is measured separately (S15).
- 2026-10-26 — S3: may RCA order shift between runs? → **No, the order is fixed too.** Supersedes
  the 2026-10-25 "set fixed, order may shift" answer, made when model-assisted ranking was expected;
  the ranking is now a deterministic score. Reopen only if an LLM re-ranker is deliberately adopted
  (FR-19).
- 2026-10-26 — S4: do confirm/dismiss decisions count as part of "the same snapshot"? → **Yes:
  same code and same decision state**, stamped on every answer (FR-19). Decisions can then take
  effect immediately without breaking consistency (see S23).
- 2026-10-26 — S6: what does repository access protect? → **Content only** (code, contents,
  internal names); existence and count of affected items may be shown (FR-24).
- 2026-10-26 — S5: who may run raw queries? → **Only people who can read every repository in the
  system**; others get the same facts through normal answers (FR-24, FR-16). No per-access-group
  filtered views.
- 2026-10-26 — S8: how is "never miss" judged? → **Frozen blind list (≥75), each impact listed
  individually with evidence; ≥95% of confirmed items real; false-alarm cap 4:1, provisional until
  phase 1 measures it** (AC-5).
- 2026-10-26 — S9: RCA quality bar? → **≥40 planted bugs; top-5 ≥70% (≥50% from a plain
  symptom); must beat "closest to the error first"; provisional** (AC-12).
- 2026-10-26 — S10: "ranked by likelihood"? → **"Ranked by a stated score, with its evidence"**
  (FR-12); calibrated probabilities would need hundreds of labelled incidents.
- 2026-10-26 — S11: testing doubt? → **Planted unprovable links of named kinds; all possible with
  reason, none confirmed** (AC-6).
- 2026-10-26 — S12: testing security? → **≥10 real + ≥10 look-alike-safe per kind; a reachable
  vulnerable dependency is never called unreachable** (AC-15).
- 2026-10-26 — S13: reachability depth? → **Package level always; function level only where the
  advisory names the functions** (FR-15).
- 2026-10-26 — S14: the "any codebase" test? → **OpenTelemetry demo, config frozen before first
  contact, first result reported as-is; a third codebase held back** (AC-7).
- 2026-10-26 — S15: test plain-language understanding? → **Yes, new AC-24: ~100 questions, ≥92%
  understood, ambiguity answered with options; provisional**.
- 2026-10-26 — S16: does a weaker local model change facts? → **No; it may need more rephrasing
  only** (FR-9).
- 2026-10-26 — S18: may local-only builds download dependencies? → **No; pre-provisioned caches or
  less precise analysis** (FR-9). Joern's Java/Kotlin/C#/Go readers can download, and Kotlin's
  sends import names to an online service (`analyst` review B4).
- 2026-10-26 — S19: mixed local-only/hosted systems? → **Local-only is sticky per answer** (FR-9 ×
  FR-22).
- 2026-10-26 — S17: what must not leave in local-only mode, and what is "the machine"? → **Strict:
  nothing derived from the repositories (source, names, paths, messages, stack traces, dependency
  lists); "the machine" is an explicit allow-list of local endpoints** (FR-9, AC-8).
- 2026-10-26 — S20: shell/config and prompts? → **Shell scripts and structured config added to FR-1;
  a prompt naming a tool is a possible caller** (FR-1, FR-5).
- 2026-10-26 — S21: example configs and doc snippets? → **Example/template configs are possible
  callers ("example config"); documentation snippets don't count** (FR-5).
- 2026-10-26 — S22: do confirmations expire? → **Yes, same rule as dismissals; "change" = the
  enclosing function or config entry** (FR-29).
- 2026-10-26 — S23: when does a verdict take effect? → **Immediately, surviving refreshes** (FR-28).
- 2026-10-26 — S24: rename/move/signature change? → **Link returns as possible without the
  "earlier verdict" note; old verdict stays listable** (FR-29).
- 2026-10-26 — S25: who are the designated people? → **CODEOWNERS-style file; OS user locally
  (honest users only), company identity provider when hosted** (FR-28).
- 2026-10-26 — S26: whose access applies to an agent or CI job? → **The launching person's, or a
  configured service identity's own grants** (FR-24).
- 2026-10-26 — S27: who may give a verdict on a cross-repository link? → **An owner of either side;
  shown to both** (FR-28).
- 2026-10-26 — S28: a part that has never built? → **"Not mapped"; anything crossing into it is
  possible** (FR-7a).
- 2026-10-26 — S29: runtime coverage granularity? → **Per-test detail required**, not only
  suite-wide totals (FR-14). Chosen over the recommendation (totals only).
- 2026-10-26 — S30: is a list of a system's repositories "hand-describing"? → **No, it is acceptable
  input**; describing how parts talk is not (FR-0, AC-7).
- 2026-10-26 — S29 follow-up: project without per-test data? → **Static coverage only, runtime
  half marked unavailable** (FR-14).
- 2026-10-26 — S31: what is a "PR"? → **Core takes diff, commit range or branch; PR is an optional
  hosting-service add-on** (FR-17).
- 2026-10-26 — S32: "never changes the core"? → **Within the shared vocabulary; a genuinely new
  concept is a versioned core change** (FR-26).
- 2026-10-26 — S33: concurrent change previews? → **Capped, configurable, default ~4, oldest idle
  retired first** (FR-23).
- 2026-10-26 — S34: where does it live? → **Grow `cpg/`** (chosen over the recommendation of a new
  `codegraph/` component).
- 2026-10-26 — S35: plain-language questions in phase 1? → **Yes** (chosen over the recommendation
  to defer; early people-facing value accepted with an early LLM dependency).
- 2026-10-26 — S36: who writes blind answer keys? → **Isolated read-only agent, spot-checked by the
  stakeholder**.
- 2026-10-26 — S37: question languages? → **English and Portuguese** (AC-24).
- 2026-10-26 — S38: supply counts-only workplace statistics? → **No.** Workplace-scale readability
  of the "could be any endpoint" fallback is checked only on the OpenTelemetry demo and this repo;
  the risk is accepted.
- 2026-10-26 — Consistency edits following S17/S23/S29/S31: AC-8, AC-14, AC-17 and AC-21a reworded
  to match; no new decisions.
- 2026-10-26 — AC review, AC-1 to AC-4: **accepted as written**. AC-2's "listed" includes a caller
  shown as possible (e.g. `mcp-monitor`'s example config, per S21).
- 2026-10-26 — AC review, AC-9 to AC-13: **accepted, with two additions**: AC-9 also checks the
  decision-state stamp (S4); AC-9b also checks the never-built "not mapped" case (S28).
- 2026-10-26 — AC review, AC-16 to AC-23: **accepted, with three additions**: AC-20 covers agent/CI
  askers and raw-query refusal (S5, S26); AC-21a covers confirmation expiry, rename/signature change,
  and cross-repository verdicts (S22, S24, S27); AC-22 is bounded by the shared vocabulary (S32).
  AC review complete.
- 2026-10-26 — What does a code-review answer include beyond impacted callers (the earlier
  "etc.")? → **All four: broken contracts of every kind, test impact, security impact, config and
  wiring affected** (FR-13).
- 2026-10-26 — Review answer: pre-existing issues? → **New or worsened only; pre-existing on request**
  (FR-13).
- 2026-10-26 — AC-13 scope? → **One planted un-merged change per contract kind**, checking all four
  review elements and the "show existing" split (AC-13).
- 2026-10-26 — Security gaps found on re-check → **all added**: dismiss false-positive findings
  (same rules as links), severity with reason on every finding, a fifth kind for risky
  configuration, secret values never stored or shown, and secrets in git history in scope (an
  exception to latest-snapshot-only) (FR-15, Out of scope).
- 2026-10-26 — Severity scale? → **Four levels with stated rules** (dependencies map from their
  advisory score). History window? → **Configurable per codebase** (FR-15). AC-15 widened to the five
  kinds and the new rules.
- 2026-10-26 — Impact gaps on re-check → **all added**: direct/indirect with distance, grouping by
  repository/component then distance (confirmed first), and the reverse "what does this depend
  on?" question. Impact follows calls, data flow and control dependencies; stakeholder notes that
  Joern's control-flow and data-dependency relationships should serve all use cases (FR-11,
  guiding principles).
- 2026-10-26 — Test-map gaps on re-check → **added**: "which tests does this change reach?", test
  kind (unit/integration/end-to-end), risk-ranked untested code, stale runtime data flagged.
  Granularity → **function level** (FR-14).
- 2026-10-26 — AC-14 extended (test kind, tests-for-a-change, risk ranking, stale data) and new
  AC-25 (impact shape: data-flow and control-dependency effects, direct/indirect, grouping, reverse
  question) → **accepted**.
- 2026-10-26 — RCA gaps on re-check → **all seven added**: config/schema/wiring causes, backward
  tracing through data flow and control dependencies, recent changes as evidence (own per-codebase
  window), an explicit "outside the mapped code" branch, combined evidence, ruled-out/confirmed
  feedback with re-ranking (this investigation only, nothing persists), and an explicit "no match"
  answer (FR-12, AC-12, Out of scope).
