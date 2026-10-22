# In-flight work, backlogs and reference lookup in the graph — Feature Requirements

> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-10-21

## Intent

The team's FalkorDB graph should hold the work that is **alive right now**: in-flight
coordination (the per-feature ledgers `teco`/`tico` keep today as Markdown tables) and the
indexed reference knowledge agents look up mid-task — and, reconsidered the same day, the open
backlog items that precede that work. In the stakeholder's words: *"ongoing tasks and reference
indexation can greatly benefit from it."* History stays out: the graph lives in RAM and history
only grows. (Backlogs were first excluded on the same grounds, then brought back in once measured:
~0.3 MB of text across every backlog, kept bounded by the forward-only rule.)

Two beneficiaries, both with a need today (settled 2026-10-20):

- **The stakeholder** — one view across every component of what's in flight, what's waiting on
  them, and what's blocked, instead of opening ledger files one by one.
- **The agents** — find a unit's status, agent id, deliverable or open question by query rather
  than reading a whole Markdown table, so resuming work is faster and cheaper.

## Problem & current state

**Coordination ledgers** — `<component>/docs/plans/<slug>-coordination.md`, one Markdown table
per multi-unit job (76 in the repo; median 6 rows, largest 86). They are git-versioned and
crash-durable, which is what lets a coordination resume after a session dies. But:

- Progress is readable only by opening each file; there's no view across components ("what's
  waiting on me?"). Cells routinely exceed 600 characters.
- An agent resuming work must read and parse a whole table to find one row's status or agent id.
- Nothing about ledgers exists in the graph today. The `cypher` MCP tool authorizes only six
  learning-note write shapes, and falkor-chat's `ingest_document` chunks, embeds and
  LLM-extracts every write — not a fit for exact row updates.

**Reference knowledge** — agents' on-demand knowledge files are already mirrored into
`ws:agent-team` and searchable (K-030; `skills/agent-kb-retrieval/SKILL.md`): 366 documents, all
`ready`; measured recall@5 0.875 (95% CI 0.72–0.95, n=32). The flat files stay authoritative and
the graph is a hand-synced mirror, and it has already drifted: a claim deleted on 2026-10-14 is
still served, a claim added that day is missing, and four knowledge files aren't in the sync
manifest at all (filed as `claude/cobb/kaizen/plan.md` K-058).

**Durability context** — the local FalkorDB persists by RDB snapshot only (`appendonly no`;
`save 3600 1 300 100 60 10000`), so a crash can lose up to ~60 minutes of writes.

## User stories

- **US-1.** As the stakeholder, I want to see everything in flight across all components in one
  place — including what's waiting on me and what's blocked — so that I don't have to open each
  coordination file to know where work stands.
- **US-2.** As a coordinating agent (`teco`/`tico`) resuming a job, I want to look up a unit's
  status, agent id and deliverable directly, so that I don't spend context reading and parsing an
  entire ledger table.

- **US-3.** As the stakeholder, I want to ask what's planned for a component or an agent and get
  its open backlog items, so that I can see what comes next without opening backlog files.
- **US-4 (the stakeholder's stated genuine use).** As the stakeholder, I want the backlog in the
  graph so that the execution of the backlog can be **planned optimally** — *"in order to be able
  to optimally plan the execution of the backlog, the backlog must be in the graph"*
  (2026-10-20). The plan takes the form of a **milestone proposal**: open items grouped into the
  next milestone(s), each with its internal order.

## Functional requirements

- **FR-1 (scope boundary — backlog in, history out).** Open backlog items join the graph view;
  history logs (`HISTORY.md`, agent `kaizen/history.md`) do **not** — they grow without bound and
  the graph lives in RAM. This **keeps** root `AGENTS.md`'s direction that "Backlogs are headed
  for the graph" (the stakeholder's own 2026-08-25 ruling) and adds that history is not.
  "Backlogs" means **every** component `BACKLOG.md` (including root `docs/BACKLOG.md`, `cpg/`'s)
  **and** every agent's `claude/<name>/kaizen/plan.md` — every item each holds, with its status
  marker.
- **FR-2 (source of truth).** For every coordination ledger, the Markdown file
  (`<component>/docs/plans/<slug>-coordination.md`) remains the authoritative record. The graph
  holds a queryable mirror of it; where the two disagree, the file wins.
- **FR-3 (no loss on crash).** A crash of the graph, the machine or a session must lose **no**
  ledger state. Because the file is authoritative (FR-2), the graph mirror must be fully
  rebuildable from the ledger files alone — no ledger fact may exist only in the graph.
- **FR-4 (freshness — same moment).** A ledger update counts as complete only when both the file
  and the graph reflect it. The graph view is never knowingly behind the file.
- **FR-5 (staleness is visible).** Every answer drawn from the graph states when the ledger(s) it
  covers were last synced, and flags any row whose graph value disagrees with the file — so a
  reader (the stakeholder or an agent) knows when to trust the file instead. Drift may never be
  silent.

- **FR-6 (what "ongoing" contains).** The in-flight view covers, across every component:
  (a) each open coordination's units with owner, status, agent id and deliverable;
  (b) **questions waiting on the stakeholder** — paused units' open questions, with the date each
  was asked; (c) **open review verdicts** — gates that returned "needs changes" and aren't yet
  resolved; (d) **requirements not yet picked up** — requirements documents at `Interviewing` or
  `Ready for design` that no coordination has started on. Items (c) and (d) partly live outside
  ledger files (review documents, requirements headers); FR-2/FR-3/FR-5 apply to those source
  files equally — the files stay authoritative, the graph mirrors them.
- **FR-7 (first way in).** In the first delivery, the stakeholder reaches the view by **asking an
  agent** in a Claude Code session ("what's in flight?", "what's waiting on me?") and getting the
  answer in conversation. A terminal command, a falkor-chat web page, and asking inside a
  falkor-chat conversation are all wanted, in later deliveries.
- **FR-8 (agent lookup).** An agent can retrieve a single unit's status, agent id, deliverable
  and open question (if paused) by query, without reading the ledger file (US-2).

- **FR-9 (leaving the view).** A coordination stays in the in-flight view until its document's
  `Status:` flips to `archived` (milestone close) — not merely when its last unit is accepted. A
  requirements document likewise leaves once picked up by a coordination or archived.
- **FR-10 (reference knowledge freshness).** The existing reference-knowledge mirror in
  `ws:agent-team` (agents' on-demand knowledge files, K-030) is held to the same standard as
  ledgers: the files stay authoritative (FR-2), an edit counts as complete only once the mirror
  reflects it (FR-4), and search answers carry sync time and flag drift (FR-5). Search quality
  itself is not changed by this feature.
- **FR-11 (backlog source of truth — file now, graph later).** In this delivery, backlog files
  stay authoritative and are mirrored under the same rules as ledgers: FR-2 (file wins), FR-3
  (rebuildable from files), FR-4 (same-moment), FR-5 (visible drift). Making the graph the
  backlog's home — the 2026-08-25 ruling's "not even a file" — is a recorded later step, not
  designed here.
- **FR-12 ("what's planned" is its own question).** Backlog items do **not** appear in the
  in-flight view (FR-6). They answer a separate question — "what's planned / proposed for
  component X (or for agent Y)?" — reached the same way as FR-7 (ask an agent, first delivery).

- **FR-13 (milestone proposal).** On request, the system proposes how open backlog items group
  into the next milestone(s) and in what order within each, judged by three criteria, all
  required: (a) **dependencies respected** — no item before the items it needs, and work that
  unblocks the most other items earlier; (b) **no overlapping items side by side** — two items
  *overlap* when their recorded touched areas (FR-14) share a **file**, a **graph key** (a named
  FalkorDB graph such as `ws:test`, whose live data both would read or rebuild), or a **fact**
  (one item changes something the other's result describes or depends on — a name, a
  signature, a cited location). Overlapping items are **always** sequenced, never proposed to run
  in parallel, with the overlap named; (c) **priority/value first** — higher-priority items land
  sooner. Each placement states which criterion drove it. *(Context for design: `teco`'s own
  dispatch-time overlap checks are unaffected — the proposal plans from recorded predictions,
  dispatch still checks the actual work.)*
  A proposal covers **one component (or one agent's plan) at a time**, matching how milestones
  are tracked per backlog today. A dependency on **another component's** item is respected — the
  dependent item waits until that item is delivered — and the proposal names the external
  blocker and the backlog it sits in.
- **FR-14 (planning facts are recorded on the item).** Every backlog item carries, as recorded
  facts: its **dependencies** (the items it needs first), its **priority**, and the **areas it
  will touch** (files, graph keys, or facts it changes — the collision basis). The proposal uses
  only recorded facts, never guesses. This requires one common item shape across every backlog
  (today: headed sections, grouped lists, tables and bullets; ~5 explicit dependency mentions in
  total; priority recorded mainly in agent plans; touched areas nowhere). Existing items gain
  these facts **as each is next touched** — no one-time backfill. A proposal plans only items
  whose facts are recorded, and lists every other open item as **"not plannable yet"**, with
  what's missing.
- **FR-15 (who asks, who approves).** Either the stakeholder (by asking an agent) or `teco` may
  request a proposal. A proposal **never takes effect** — never changes milestones or order in
  any backlog — without the stakeholder's explicit approval.

## Out of scope

- History logs in the graph (FR-1).
- Making the graph the backlog's authoritative home (no file) — a recorded later step (FR-11).
- The data model, write mechanism and tooling for ledgers in the graph — design work for
  `architect`/`graph-dba` once this document is Ready for design.
- Analytics over past work (token cost per unit/agent, review-loop length, pause frequency) —
  not a need today (2026-10-20).
- Linking coordination units to the reference knowledge they used or produced — not a need
  today (2026-10-20).
- Improving reference-search quality (recall, score floor) — 2026-10-20.
- Fixing the existing reference-mirror drift itself — already filed as cobb K-058; this document
  may set the *requirement* that the mirror stays fresh, not the fix.

## Acceptance criteria

- **AC-1 (doc consistency for FR-1).** Given FR-1 keeps the backlog-to-graph direction and rules
  history out, when this feature is delivered, then root `AGENTS.md` states both (backlogs in the
  graph, history stays a file) and no active document says history is headed for the graph.

- **AC-2 (crash safety, FR-3).** Given the graph is emptied or lost entirely, when the mirror is
  rebuilt from the ledger files alone, then every ledger fact the graph held before is restored —
  nothing is missing.
- **AC-3 (same-moment freshness, FR-4).** Given a coordinator reports a ledger update as done,
  when anyone queries that unit immediately afterwards, then the graph returns the updated value,
  identical to the file.
- **AC-4 (visible drift, FR-5).** Given a ledger file is changed outside the normal update path
  (e.g. a hand edit), when the graph is queried for that ledger, then the answer shows the
  last-sync time and flags the rows that now disagree with the file.

- **AC-5 (the view, FR-6/FR-7).** Given open coordinations in at least two components, a paused
  unit, an unresolved "needs changes" verdict and a requirements doc at `Interviewing`, when the
  stakeholder asks an agent "what's in flight / waiting on me?", then the answer lists all four
  kinds, each with its component and source file path, plus the sync time (FR-5).
- **AC-6 (agent lookup, FR-8).** Given a ledger with a paused unit, when an agent asks for that
  unit, then it gets status, agent id, deliverable and the open question with its date — without
  opening the ledger file.

- **AC-7 (leaving the view, FR-9).** Given a coordination whose units are all `accepted` but whose
  `Status:` is still `active`, when the view is queried, then it still appears; once its `Status:`
  is `archived`, it no longer does.
- **AC-8 (reference freshness, FR-10).** Given a knowledge-file edit (e.g. a section deleted or
  added outside a cobb distillation pass), when an agent searches immediately after the edit is
  reported done, then the deleted claim is not returned and the added one is retrievable; and a
  deliberately unsynced edit is flagged in the answer, not served silently.
- **AC-9 (what's planned, FR-1/FR-11/FR-12).** Given open items in at least one component
  `BACKLOG.md`, in root `docs/BACKLOG.md` and in one agent's `kaizen/plan.md`, when the
  stakeholder asks an agent "what's planned for <component/agent>?", then the answer lists those
  items with ID, status marker and source file, plus sync time — and none of them appears in the
  in-flight view's answer (AC-5).
- **AC-10 (backlog mirror freshness, FR-11).** Given a backlog item is added, rewritten or removed
  in its file, when the planned view is queried immediately after, then it reflects the change;
  a hand edit that bypassed sync is flagged, not served silently.
- **AC-11 (milestone proposal criteria, FR-13).** Given a component backlog where item B depends
  on A; C and D touch the same file; G and H both rebuild the same graph key; I renames something
  J's deliverable cites; and E has higher priority than F (all with recorded facts), when a
  proposal is requested, then A is placed before B; each of C/D, G/H and I/J is sequenced, never
  in parallel, with its overlap named; E lands no later than F; and each placement states the
  criterion that drove it.
- **AC-12 (not plannable yet, FR-14).** Given an open item missing any of dependencies, priority
  or touched areas, when a proposal is requested, then that item is not placed and is listed as
  "not plannable yet" with the missing facts named.
- **AC-13 (cross-component blocker, FR-13).** Given an item that depends on an undelivered item in
  another component's backlog, when its component's proposal is produced, then the item is held
  back and the proposal names the external blocker and its backlog.
- **AC-14 (approval gate, FR-15).** Given a proposal requested by the stakeholder or by `teco`,
  when nobody approves it, then no backlog file's milestones or order change; after the
  stakeholder's explicit approval, they do.

## Open questions

None — all settled at the 2026-10-20 readback (see Decision log).

## Decision log

- 2026-10-20 — Is the graph a home for backlogs and history? → first answer: **No** — the graph
  lives in RAM; ongoing tasks and reference indexation are where it pays off. Drafted as FR-1
  reversing root `AGENTS.md`'s "Backlogs are headed for the graph". **Revised later the same day**
  (entry below).
- 2026-10-20 — Context gathered before the interview (tico, read-only): `ws:agent-team` state,
  KB-mirror drift, FalkorDB persistence settings, ledger sizes — see Problem & current state.
  Drift filed separately as cobb K-058 at the stakeholder's request.
- 2026-10-20 — Who benefits? → **The stakeholder** (cross-component in-flight view) and **the
  agents** (resume & find a unit by query). Analytics over past work and work↔knowledge linking
  were offered and **not** chosen — recorded as out of scope for now.
- 2026-10-20 — Which copy of a ledger is the truth? → **The Markdown file**; the graph is a
  mirror (FR-2). Chosen over graph-authoritative-with-export and graph-only.
- 2026-10-20 — How much in-flight state may a crash lose? → **Nothing** (FR-3). Combined with
  FR-2 this means the mirror must be rebuildable from files; it does **not** by itself require
  changing FalkorDB's RDB-only persistence. Freshness becomes the critical requirement (FR-4),
  since the same file-plus-mirror pattern is what drifted for reference knowledge (K-058).
- 2026-10-20 — How fresh must the mirror be? → **Same moment**: an update is done only when file
  and graph both reflect it (FR-4). Chosen over ~1-minute lag and on-demand refresh.
- 2026-10-20 — What does a reader experience when the mirror is stale? → **Told, with sync
  time** — answers carry last-sync time and flag disagreeing rows (FR-5). Chosen over silent
  self-healing and self-heal-plus-warn.
- 2026-10-20 — What does "ongoing" contain? → **Units plus all three extras**: waiting questions,
  open "needs changes" verdicts, requirements not yet picked up (FR-6). (A first answer selected
  "units only" alongside the extras; re-asked, clarified.)
- 2026-10-20 — How to reach the view? → All four channels wanted (ask an agent, terminal
  command, falkor-chat page, falkor-chat chat); **first delivery: ask an agent** only (FR-7).
- 2026-10-20 — What does this feature require of reference knowledge? → **Same freshness
  rules** as ledgers (FR-10). Chosen over "fine once K-058 lands" and "freshness plus better
  search"; search quality stays out of scope.
- 2026-10-20 — When does an item leave the view? → **When its document is archived** (FR-9).
  Chosen over "when nothing is open" and "done, kept a few days".
- 2026-10-20 — FR-1 reconsidered at readback. Provenance traced: the `AGENTS.md` sentence is the
  stakeholder's own 2026-08-25 ruling (commit `1d8aed75`; `claude/cobb/kaizen/history.md`
  2026-08-25 G2 pass 2), given as the reason for the forward-only backlog rule; the 2026-09-10
  decision in `claude/docs/requirements/agent-knowledge-base-strategy.md` deferred it without
  dropping it. Measured: all backlogs ~0.3 MB of text (bounded), all histories ~3.3 MB (unbounded),
  live ledgers ~0.16 MB, FalkorDB using 376 MB. → **Backlog in, history out** (FR-1, AC-1 revised).
  Chosen over "both out, direction kept as future item" and "both out, direction dropped".
- 2026-10-20 — Which backlogs? → **Components and agents** (every `BACKLOG.md` incl. root/`cpg`,
  plus every agent `kaizen/plan.md`). Truth? → **File now, graph later** (FR-11). Where shown? →
  **A separate "what's planned" question**, not the in-flight view (FR-12).
- 2026-10-20 — Stakeholder named the genuine use for backlog-in-graph: **optimally planning the
  backlog's execution** (US-4). Opens a new interview thread.
- 2026-10-20 — What should the plan give? → a **milestone proposal** (chosen over a full ordered
  schedule, a next-pick answer, or both). Optimize for? → **dependencies respected, fewest
  collisions, priority/value first**; lowest cost/effort **not** chosen (FR-13).
- 2026-10-20 — Where do planning facts come from? → **Recorded on each item** (FR-14), chosen over
  inferred-then-confirmed and recorded-plus-inferred; accepted that this needs a common item
  format. Scope? → **One component (or agent) at a time** (FR-13).
- 2026-10-20 — Who asks/approves? → **Either the stakeholder or `teco` asks; only the stakeholder
  approves** (FR-15). Existing items? → **Facts added as each item is next touched**; unplanned
  ones listed "not plannable yet" (FR-14). Cross-component dependency? → **Respected and shown**
  as an external blocker (FR-13).
- 2026-10-20 — How strict is the overlap rule? → **Always order them**: overlapping items (shared
  file, graph key or fact) are never proposed in parallel (FR-13(b), "overlap" defined in place).
  Chosen over "order by default, parallel as a flagged accepted risk" and "flag only".
- 2026-10-20 — Closing readback confirmed by the stakeholder (incl. FR-1 revision, US-4/FR-13–FR-15
  planning thread, strict overlap rule). → Status **Ready for design**; handed to `architect`.
- 2026-10-21 — Architect's plan questions (`claude/docs/plans/coordination-graph.md` §6), answered
  by the stakeholder: **Q1** a one-time *layout* reformat of every backlog into the common item
  format is accepted — FR-14's "no one-time backfill" bans filling in planning facts, not
  reformatting. **Q1b** "Parking lot / ideas" sections in agent kaizen plans **are** items the
  planner sees (they get ids and the item format in the same reformat; chosen over the
  recommended exclusion). **Q2** if step S0 shows the post-edit hook doesn't fire in every agent
  context: accept instant sync where it fires and FR-5 staleness flagging elsewhere — no watcher,
  no prompt-level sync. **Q3** milestones cap at 6 items by default, adjustable per proposal.
  **Q4** `docs/requirements/cross-initiative-coordination.md` stays open and gets re-scoped by
  `tico` against this feature once the plan is approved. **Q5** the stakeholder applies the
  `.claude/settings.json` blocks (S0 probe, S8 hook) by hand. **Q6** keep RDB-only snapshots;
  the post-crash window is flagged stale and repaired by `sync --all`.
- 2026-10-21 — Review-driven questions (`claude/docs/reviews/coordination-graph.md` Pass 1, open
  questions 1–2): while the sync cannot run (FalkorDB or LM Studio down), **warn on every edit**
  made during the failure — stricter than the analyst's proposed "once per file version" — but
  retries are backed off and never make a command hang (B1 stands). An **approved `Milestone`
  is fixed**: later proposals place only unassigned items around approved ones and flag any
  approved placement a new dependency makes impossible.
- 2026-10-21 — Plan v2 §6 Q7 (multi-chunk KB edits stay "pending" until falkor-chat's
  embed-after-extract ordering is fixed): **schedule `falkor-chat:K-070` alongside**, in parallel
  with the KB-mirror step, as a prerequisite of AC-8's final acceptance; K-070 stays
  falkor-chat's own item and is not folded into this plan.
- 2026-10-22 — Review Pass 2 open question (`claude/docs/reviews/coordination-graph.md`): a
  high-priority new item **may** be proposed into a gap before an approved later milestone, marked
  as inserted ahead of it, and approved or rejected like any placement — approved milestones
  themselves never move. The one-time reformat **converts** an existing milestone table's
  open rows into `Milestone` fields on their items (falkor-chat: M2.5 → K-016, K-017, K-018), so
  the planner treats them as already placed; the table may remain as the human-readable milestone
  map.
