# Agent reading cost — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-11-01 · **Reviews:** `docs/reviews/agent-reading-cost.md`

## Intent

Cut what each agent run spends on reading before it can do its work, by making more use of the
knowledge the team is already building: the `ws:agent-team` knowledge base (GraphRAG in
falkor-chat) and the code property graphs being connected under C-1001.

Both kinds of reading are in scope, linked: an agent should be able to go from a step in a plan to
the code it changes, and from code back to the decisions behind it, without reading either whole.
Success is measured twice: tokens per run go down, and review quality does not.

## Problem & current state

Evidence from the typed-decision-models family (measured 2026-11-01 at commit `1784168b`, `tico`
in conversation; costs from the Cost column of `docs/plans/typed-decision-models-coordination.md`):

- About 45 units so far, each typically 150–330k tokens, almost each with its own `analyst` gate.
- Before reading any code, a build unit is pointed at about 41k words (≈55k tokens): the plan and
  its four extending successors (25.3k), the method note (9.6k), and two `AGENTS.md` files (5.9k).
- Successors that extend an earlier plan are never folded back, so a brief cites several of them
  for one unit (e.g. "v1.3 §4 U1d; plan2 §4.1; plan3 §4.1").
- Review documents that collect one pass per build unit keep growing (28.8k words, 22 passes).
- The coordination ledger has grown to 10.1k words and `teco` re-reads it whole at each dispatch.
- The whole family is about 130k words, as big as the connected-code-graph family that triggered
  `docs/requirements/document-size-discipline.md`.
- Ledgers count tokens inconsistently: some rows give one run, some a cumulative total across
  resumes, some none (e.g. connected-code-graph U26, U28–U31); a gate's own cost is recorded
  only sometimes, and never for connected-code-graph U20–U32.

What already exists:

- `docs/requirements/document-size-discipline.md` stops a document growing across revisions. It
  left token-cost targets out of scope on purpose, and does not bound how many documents a reader
  must combine.
- `ws:agent-team` holds agents' raw learnings, and agents that keep an on-demand reference file
  can search the distilled knowledge semantically
  (`claude/docs/requirements/agent-knowledge-base-strategy.md`, which scopes the knowledge base to
  all agent/team knowledge). This feature extends that corpus to the team documents FR-1 names.
- The connected code graph (`docs/requirements/connected-code-graph.md`) answers questions about
  code: impact, root cause, review, test coverage, security. Linking code to project documents is
  not in its scope.
- `falkor-chat/docs/BACKLOG.md` K-102: 11 `ws:agent-team` documents sit in `failed` with
  unembedded chunks, invisible to semantic search (context for FR-6).

## User stories

- As a **build or review agent**, I want to look up just the part of a plan or spec my unit needs,
  including detail moved out of `AGENTS.md`, and follow it to the code it changes, so that I spend
  my tokens on the work rather than on reading.
- As a **reviewer**, I want every claim an agent drew from a lookup to cite its source section or
  code, so that I can check it without re-reading everything.
- As the **stakeholder**, I want the saving measured against real finished work, with review
  quality checked alongside, so that I know it is real and that nothing got worse.

## Functional requirements

- FR-1 — An agent can retrieve a specific part of a requirements document, plan (including method
  and graph notes), live spec, or any document that receives information moved out of an
  `AGENTS.md` file (FR-4), without reading the whole document. `AGENTS.md` files themselves are
  not retrieval targets: they are loaded in full automatically. History moved out of an
  `AGENTS.md` goes to `HISTORY.md`, which stays outside retrieval (Out of scope).
- FR-2 — An agent can go from a plan step to the code it changes, and from code to the
  requirements and plan decisions behind it.
- FR-3 — Every claim an agent makes from a retrieved part cites the document section or code
  location it came from.
- FR-4 — Each `AGENTS.md` keeps only a core that answers the context-file convention's question
  (what is true now, and what will bite) plus where to look, and is measurably smaller than at the
  start of delivery. Information outside that core moves to a document better suited to it, which
  is retrievable (FR-1).
- FR-5 — Tokens per run are counted the same way for every run compared.
- FR-6 — A retrieved document part or code location is never silently out of date: it matches the
  current file, or it says it is stale and names the file to read instead.
- FR-7 — For any plan family started after delivery (a plan with its `-ml` and `-graph` notes), a
  reader can get the current design as one statement, without combining an original plan with the
  successors that extend it. Plan text that was approved, gated or executed against stays
  recoverable, and every existing citation of it still resolves.
- FR-8 (supersession) — Root `AGENTS.md` collision rule 5, as it applies to plan families started
  after delivery, no longer leaves the current design spread across an original and its extending
  successors; every convention, agent prompt and other file that states or relies on the rule
  agrees with the change.
- FR-13 — Review quality does not drop when retrieval replaces whole reading: gates catch the same
  defects, no more defects escape them, and mutation checks are caught at the same rate.
- FR-14 — Units run with retrieval meet the reduction target set under AC-1.

## Out of scope

- Reviews, coordination ledgers, `HISTORY.md` and `BACKLOG.md` as knowledge-base content (agents
  keep reading them as today); moving `BACKLOG.md` to the graph stays the separate future item of
  `claude/docs/requirements/agent-knowledge-base-strategy.md`.
- Changing the connected code graph's scope (C-1001): this feature uses it as a source.
- Chat-graph summaries for agents, humans and workflows: they stay in
  `falkor-chat/docs/requirements/summary-nodes.md`.
- Bringing existing plan families to one current design (FR-7).
- Kiro steering files (`.kiro/steering/`): FR-4 covers `AGENTS.md` files only.

## Acceptance criteria

- AC-1 (baseline) — Given the next 10–15 connected-code-graph units run before retrieval ships,
  each counted per FR-5, then their tokens per run form the before figure, and a reduction target
  is set from it, per role, as an absolute figure or a ratio to that role's before median.
- AC-2 (same defects caught, FR-13) — Given each U20–U32 unit with recorded gate findings, when
  its gate is re-run with retrieval on the deliverable version the original gate reviewed, with
  retrieval serving the documents as of that gate's commit, then it
  reports every blocker and major finding on record for that unit, confirmed by a reader other
  than the gate.
- AC-3 (no escapes, FR-13) — Over the first 20 units run with retrieval, the share of units with
  at least one defect raised after their gate was accepted (a later fix round, QA run, live run or
  later unit's review) is no higher than the upper 95% bound of the same share over the last 20 or
  more accepted units before retrieval.
- AC-4 (mutation checks, FR-13) — Given a unit run with retrieval that ships tests, when the
  coordinator (not the code's author) applies its own mutants, then every mutant fails at least one
  test, the same rule applied before retrieval.
- AC-5 (citations) — Given the output of each gate re-run under AC-2, then each claim drawn from a
  retrieved part names its document section or code location (FR-3).
- AC-6 (never stale) — Given a document edited, or code moved, after it was last made retrievable,
  when an agent retrieves that part or code location, then it gets the current text or a stale
  notice naming the file (FR-6).
- AC-7 (doc consistency) — After delivery, a sweep of the whole repo finds no file that tells a
  plan author to leave a new family's current design split across an original and its extending
  successors, each hit having been assigned an owner and fixed (FR-8); a plan family started after
  delivery gives one current design, and every citation of earlier gated plan text still resolves
  (FR-7).
- AC-11 (`AGENTS.md` core) — Given each `AGENTS.md` outside `deprecated/` that holds more than
  the core at the start of delivery, then it is smaller than at the start of delivery, every
  item moved out is listed with its new home, each moved item is retrievable there, the
  `AGENTS.md` names where to look, and no information is lost in the move (FR-4).
- AC-12 (part retrieval) — Given a question answered by one section of a requirements document,
  plan or live spec, when an agent asks it, then it gets that section with its source named,
  without the rest of the document (FR-1).
- AC-13 (plan↔code) — Given a plan step of a U20–U32 connected-code-graph unit, when an agent
  follows it, with documents and code as of that unit's commit, then it reaches the code it
  changed; and from that code, it reaches the plan step and requirement behind it (FR-2).
- AC-14 (one count) — Every new token figure is produced by one counting rule, stated once,
  covering resumed runs and gates alike (FR-5).
- AC-16 (target met) — Given the first 10 or more units run with retrieval after the target is
  set, each counted per FR-5, then their median tokens per run, per role, meets the target (FR-14).

## Open questions

(none)

## Decision log

- 2026-11-01 — Which reading should agents stop doing in full? → Both design documents and code,
  linked so an agent can go from a plan step to the code it changes and back. (Intent, FRs TBD)
- 2026-11-01 — How do we know it worked? → Both a measured token reduction and a check that review
  quality has not dropped. (ACs TBD)
- 2026-11-01 — How big a token reduction? → No number yet: measure a baseline first, set the
  target from the first measurements. (AC TBD)
- 2026-11-01 — What shows review quality held? → All four: gates still catch known defects on a
  replayed past unit; no rise in defects escaping gates; mutation checks still caught at the same
  rate; every claim drawn from a lookup cites its document section or code. (ACs TBD)
- 2026-11-01 — Where is it proven first? → connected-code-graph (C-1001), whose build units
  exercise the code side of the link. (Scope, baseline AC TBD)
- 2026-11-01 — Which team documents should the knowledge base hold? → Requirements and plans
  (incl. method and graph notes), live specs, and `AGENTS.md` context files; not reviews, not
  ledgers or `HISTORY.md`. (Scope, Out of scope)
- 2026-11-01 — What happens to the automatic `AGENTS.md` load? → Shrink it to a short core (what
  bites, where to look); the rest is looked up on demand. (FR-4)
- 2026-11-01 — How is the connected-code-graph baseline taken? → Replay already-finished units
  (U20–U32) with lookups and compare with their recorded cost and findings. (AC-1, AC-2)
- 2026-11-01 — How fresh must a retrieved part be? → Never stale: it reflects the current file, or
  says it is stale and points at the file to read. (FR-6, AC-6)
- 2026-11-01 — Where does the document↔code link live? → In this feature, which uses the
  connected code graph as a source; C-1001 keeps its scope. (FR-2, Out of scope)
- 2026-11-01 — Is the plan-successor chain in scope? → Yes: a reader always gets one current
  design. Supersedes, for plans, root `AGENTS.md` collision rule 5's "successor that extends,
  earlier one stays intact" outcome. (FR-7, FR-8, AC-7)
- 2026-11-01 — How does `falkor-chat/docs/requirements/summary-nodes.md` relate? → Fold it into
  this feature. (Scope; extent TBD)
- 2026-11-01 — How much of summary-nodes folds in? → All of it, chat-graph summaries for humans
  and workflows included; summary-nodes is superseded, its settled 2026-07-12 decisions carried
  as FR-9–FR-12, its out-of-scope lines and AC-8–AC-10. (Intent, FR-9–FR-12, AC-8–AC-10)
- 2026-11-01 — Is `AGENTS.md` itself looked up? → No: it stays auto-loaded; information outside
  its core moves to more suitable documents, and those are indexed. Overturns the `AGENTS.md` part
  of "Which team documents should the knowledge base hold?" and refines "What happens to the
  automatic `AGENTS.md` load?". (FR-1, FR-4)
- 2026-11-01 — Which way of creating a summary comes first? → Left to backlog ordering. (FR-11)
- 2026-11-01 — Does the readback capture the need? → Yes, with the FR-1/FR-4 change. (all)
- 2026-11-01 — Consult: `analyst` on readiness for design → needs changes, B-1, M-1–M-7, m-1–m-5,
  n-1 (`docs/reviews/agent-reading-cost.md` Pass 1). (AC-5, AC-6, AC-12–AC-14, FR-1, FR-6, Out of
  scope)
- 2026-11-01 — Do agent reading cost and chat-graph summaries stay one feature? → No, split back
  out: summaries stay in `falkor-chat/docs/requirements/summary-nodes.md`, which is not
  superseded. Overturns "How much of summary-nodes folds in?" and "How does summary-nodes
  relate?". (Intent, Out of scope; FR-9–FR-12, AC-8–AC-10, AC-15 retired)
- 2026-11-01 — What is the baseline, given incomplete recorded costs? → Recorded costs, only for
  units whose record covers the whole run. Refines "How is the connected-code-graph baseline
  taken?". (AC-1, AC-14)
- 2026-11-01 — Does one current design apply to existing plan families? → New work only; gated
  text stays recoverable and existing citations still resolve. (FR-7, FR-8, AC-7, Out of scope)
- 2026-11-01 — How is "short" measured for `AGENTS.md`? → Measurably smaller than at the start of
  delivery, every moved item listed with its new home; no fixed cap. (FR-4, AC-11)
- 2026-11-01 — Consult: `data-scientist` on testable quality and target ACs → no recorded cost
  covers a gate, so AC-1 as written selects no unit; AC-2 underpowered without a no-retrieval gate
  rerun; AC-3 a gross-regression guardrail only; AC-4 reworded; AC-16 needs a before figure counted
  per FR-5. (AC-1–AC-4, AC-16; AC-4 reworded)
- 2026-11-01 — What is the baseline, now that no recorded cost covers a gate? → Count the next
  10–15 units per FR-5 before retrieval ships. Overturns "What is the baseline, given incomplete
  recorded costs?". (AC-1, AC-16)
- 2026-11-01 — Add a no-retrieval gate rerun for AC-2? → No: recorded findings only, blocker and
  major, accepting it catches a large drop only. (AC-2)
- 2026-11-01 — Accept the escape check as a gross-regression guardrail? → Yes. (AC-3)
- 2026-11-01 — Do `-ml`/`-graph` notes count as part of a plan family? → Yes. Are Kiro steering
  files in FR-4? → No. (FR-7, Out of scope)
- 2026-11-01 — Consult: `analyst` Pass 2 → approve with suggestions; N2-1–N2-6 folded
  (`docs/reviews/agent-reading-cost.md` Pass 2). (AC-2, AC-3, AC-4, AC-11, AC-13, FR-7)
- 2026-11-01 — Does the `AGENTS.md` core also keep "what is true now"? → Yes. Refines "What
  happens to the automatic `AGENTS.md` load?". (FR-4)
- 2026-11-01 — Do `deprecated/` and already-minimal `AGENTS.md` files have to shrink? → No. (AC-11)
- 2026-11-01 — Which documents do gate re-runs and plan↔code checks use? → As of that unit's
  commit. (AC-2, AC-13)
- 2026-11-01 — Ready for design? → Yes, on the final readback. (Status)
