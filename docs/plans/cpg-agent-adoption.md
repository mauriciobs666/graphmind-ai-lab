# CPG agent adoption — primary design

> **Status:** archived · **Owner:** `cobb` · **Tracks:** cpg-agent-adoption (M4) · **Extended by:** `docs/plans/cpg-agent-adoption2.md` (FR-5/FR-6 freshness-check ownership moved to `teco`; AC-2 unaffected)

Design for the `cobb`-owned slice of
[`../requirements/cpg-agent-adoption.md`](../requirements/cpg-agent-adoption.md) (Status: Ready
for design) — **FR-1, FR-2, FR-3, FR-4, FR-6's surfacing/wording half, FR-9**, and **AC-1, AC-2,
AC-5, AC-6**. Sibling design: [`cpg-agent-adoption-graph.md`](./cpg-agent-adoption-graph.md)
(`graph-dba`, delivered — FR-5, FR-6's mechanical half, FR-7, FR-8, the `:CpgBuildInfo` freshness
marker and its `references/freshness.md` read recipe). Coordination:
[`cpg-agent-adoption-coordination.md`](./cpg-agent-adoption-coordination.md) (unit U2). This
document does not redesign the MCP read path or the freshness mechanics — it cites both and
builds the roster/discovery/evidence layer on top.

**This is design only.** No agent prompt, `SKILL.md`, `AGENTS.md`, or `docs/BACKLOG.md` edit is
made here. §6 hands the implementer (a later, separately-dispatched `cobb` unit, U4b per the
coordination ledger) a concrete, ordered task list; §7 proposes the `BACKLOG.md` M4 shape as a
proposal, not a write.

---

## 1. The agent roster (FR-1, AC-1)

The requirements doc deliberately left this open — "not a fixed list… any agent doing code-level
work is the target" — and named the judgment test itself: does the agent do code-level reasoning
over an actual source tree the CPG could cover (impact analysis, RCA, review, test-gap), or is
its work orthogonal (coordination, requirements interviews, agent/skill markdown authoring rather
than source code)?

**Grounding fact that shapes every call below:** today exactly two CPGs are loaded —
`cpg_falkorchat` (falkor-chat's Python server) and `cpg_salesperson` (salesperson's Python
Streamlit app) — both Python, both application source trees. An agent whose own work routinely
touches Python source in one of those two trees has an immediately relevant CPG *today*; an agent
whose work is infra-as-code, prose, or a different language has none *yet*, which is a coverage
fact (FR-7/FR-4 already price this in as an acceptable no-op), not a reason to exclude it from
the roster on principle. The roster decision below is about the *shape of the agent's work*, not
today's two-graph snapshot — a roster call that would flip the moment a third CPG existed would
be the wrong kind of call to make.

| Agent | Call | Reasoning |
|---|---|---|
| `analyst` | **In (already wired)** | Static review + RCA over real source trees — exactly the impact/RCA/taint/test-gap shape the four existing recipes cover. No change to its consumer status; only its discovery wording changes (§2). |
| `architect` | **In (already wired)** | Plan authoring investigates the codebase first — impact analysis is squarely architect's "what would this change touch" question. No change to consumer status; discovery wording changes. |
| `qa-engineer` | **In (already wired)** | Test-gap analysis (prod code no test structurally reaches) is a named recipe with `qa-engineer` as its consumer. No change to consumer status; discovery wording changes. |
| `graph-dba` | **In (already wired, producer)** | Builds/loads the CPG and now stamps its freshness marker; not a new *consumer* addition under FR-1, but worth noting it can and should use its own `references/freshness.md` recipe when asked to judge whether an existing graph needs a rebuild. No roster change. |
| **`coder`** | **New — in** | Implements approved plans end-to-end against real source trees (its "Orient" step reads the plan and the code it touches). Before changing a function, "what calls this / what would break" (impact analysis) is exactly the question `coder` should be asking instead of grepping by hand. Both live CPGs are Python codebases `coder` already works in. |
| **`tdd-engineer`** | **New — in** | Same source-tree-touching shape as `coder`, TDD-flavored: reproduction-test-first work benefits directly from RCA/impact recipes ("what's the actual call path to the symptom," "what else exercises this function"), and test-gap analysis is a natural companion to "what should I be testing." |
| **`frontend-engineer`** | **New — in** | This lab's frontend work today *is* `salesperson/chatbot.py` — a Streamlit (Python) UI, which `cpg_salesperson` already covers. `frontend-engineer`'s own prompt names that file as the concrete UI surface it owns. When it changes a handler or a shared UI helper, "who else calls this" is the same impact-analysis question `coder` asks. Coverage is Python-only today (no CPG for a future JS/TS frontend), which is a normal discovery-miss case (FR-4), not a reason to leave the agent unwired — the wiring should exist before the coverage gap closes, not after. |
| **`devops`** | **Out** | Its source tree is Dockerfiles, Compose, CI, shell scripts, and dependency manifests — none of it is Joern-parseable (Joern targets application source: Python/JS/TS/etc., not infra-as-code or Bash). A CPG discovery check in devops's typical task would be a *structural* miss every time, not a judgment call to make per-task — there is nothing for it to discover, ever, given the CPG's actual domain. Wiring it would be decorative. |
| **`cobb`** (this agent) | **Out** | This is the requirements doc's own example of orthogonal work — "agent/skill markdown authoring rather than source code." `cobb` reasons about prompts and frontmatter, not call graphs. |
| **`data-scientist`** | **Out** | Advisory-only; never edits or executes code. Its questions are about method validity (embedding choice, chunking strategy, eval design, statistical soundness) — not "what calls this function" or "does a test reach this branch." Where it does read pipeline code to judge a method, that's methodology judgment, not structural code reasoning, and the general-correctness half already routes to `analyst`, which *is* CPG-wired. Wiring `data-scientist` to `cpg-analysis` would duplicate `analyst`'s coverage without adding a question the CPG actually answers for this discipline. |
| **`teco`** | **Out** | Coordinates and routes; doesn't do code-level reasoning itself (its one carve-out — a trivial single-file no-brainer — doesn't rise to "impact analysis, RCA, review, or test-gap"). Also the requirements doc's own orthogonal-work example. |
| **`tico`** | **Out** | Product-altitude only — WHAT/WHY, interviews, manuals. Never reads code as a reasoning target; explicitly "does not design the solution and does not write code." |

**Resulting wired roster (post-M4): `analyst`, `architect`, `qa-engineer`, `coder`, `tdd-engineer`,
`frontend-engineer`** as consumers, **`graph-dba`** as producer. Six consumers, up from three —
the widening FR-1 asked for, without touching `devops`/`cobb`/`data-scientist`/`teco`/`tico`,
each excluded for a stated, code-shape reason rather than by omission.

---

## 2. The discovery step (FR-2, FR-4, AC-1, AC-5)

### 2.1 Where it lives — judgment call, with reasoning

Two real candidates were weighed, per the brief:

**Candidate A — a shared line in root `AGENTS.md`'s "Working in this repo" section.** Rejected
as the *primary* mechanism. Custom subagents in this harness auto-load the `CLAUDE.md`/`AGENTS.md`
hierarchy regardless of their own `tools:`/description wiring, so a root-level bullet would reach
every agent's session, including the five just ruled **out** in §1 — `devops`, `cobb`,
`data-scientist`, `teco`, `tico` would all carry the instruction whether or not it ever applies to
their work. That's contrary to this repo's own lean-context discipline (skills exist specifically
so on-demand capability doesn't bloat every session) and it blurs a roster decision this document
was explicitly asked to make into an unbounded "any agent, self-select" instruction. The
stakeholder's decision-log analogy — *"like reading `AGENTS.md` today… without needing to be
reminded"* — reads best as a description of **behavioral character** (unprompted, standard, not
requiring a reminder), not a literal instruction to add this to the `AGENTS.md` file itself.

**Candidate B — each wired agent's own prompt (description + body), matching the existing
`cpg-analysis`/`python-web-quirks` wiring pattern.** **Chosen.** This repo already has a working,
proven mechanism for exactly this shape of capability: `analyst`/`architect`/`qa-engineer`'s
`description` frontmatter already names `cpg-analysis` and `python-web-quirks`, and `teco`'s own
routing table treats each specialist's injected `description` as "the capability catalog — don't
re-derive it." Extending that same field for the three new consumers, and *strengthening* its
wording for all six from conditional ("when a CPG is loaded…") to default ("checks whether a
relevant CPG exists as part of its normal orientation…"), is the smallest change that (a) reaches
exactly the roster decided in §1, (b) feeds `teco`'s routing awareness the same way the existing
mentions do, and (c) puts the actual "how to check" step in the body prompt's existing
orientation section — the same place these agents already read `AGENTS.md`, `CLAUDE.md`, and
project docs — which is the literal analogue the stakeholder's own phrasing pointed at.

**A skill's own `description`/body is not sufficient on its own**, confirming the brief's
hypothesis: a skill activates once reached for — an agent has to already think "let me check
`cpg-analysis`" for it to load. That's exactly the reminder-driven behavior FR-2 rules out. The
skill stays the destination (§2.2 below adds one clarification to it), not the trigger.

### 2.2 The mechanic — how an agent checks cheaply

Documented once, in `skills/cpg-analysis/SKILL.md` §1 ("Finding the graph name"), which already
states there is deliberately no `list_graphs` tool and names two existing fallbacks
(`GRAPH.LIST` via `redis-cli`, or reading the not-found error's graph listing). The implementer
adds one clarifying paragraph there — not a new mechanism, a documented **first guess** to try
before falling back to those two:

> Both live graphs today follow the pattern **`cpg_<component>`**, component-directory name with
> hyphens stripped (`falkor-chat` → `cpg_falkorchat`, `salesperson` → `cpg_salesperson`). Guess
> that name first and send a cheap query against it (e.g. `MATCH (n) RETURN count(n)` or the
> freshness recipe itself, which doubles as an existence probe — see
> [`references/freshness.md`](references/freshness.md)). A **hit** means the CPG exists **and**
> hands you its freshness in the same call (§2.3 below). A **miss** returns the tool's
> not-found error, which lists every graph actually loaded — check that list for a differently
> named match before concluding there is none, then stop; that is the whole discovery cost.

This is **one `mcp__cpg__query` call** (or, off Claude Code, one `redis-cli GRAPH.QUERY`/
`GRAPH.LIST`) — no shell round-trip ceremony under the MCP path, no process spun up beyond the
already-deferred tool load the harness does automatically on first use. That satisfies AC-5 ("no
material delay or noise") concretely: a miss costs exactly one fast, read-only query and produces
nothing worth narrating beyond the one-line note in §4.

Each wired agent's prompt does **not** restate this mechanic — it points at the SKILL.md section
by reference, consistent with the skill's own "recipes cite it, don't restate it" discipline.

### 2.3 Bundling the freshness check into the same default step (item 3 of the brief)

`graph-dba`'s recipe (`cpg-agent-adoption-graph.md` §2, proposed `references/freshness.md`)
already carries complete model-facing surfacing language — the "Surfacing the suggestion (FR-6)"
paragraph is sufficient as written and is **not** rewritten here (that would be redesigning
graph-dba's slice, out of bounds per the brief). What this document adds is the **integration
point**: the discovery-step wording in every wired agent's prompt says, in substance —

> When a relevant CPG is found and you use it for the task, also run the freshness check
> (`references/freshness.md`) as part of that same step — not a separate, optional pass. Note
> what it tells you (built-at age, whether the source has moved since) in your deliverable
> (§4), and if it looks stale, surface graph-dba's suggested-refresh language rather than
> silently trusting or rebuilding it yourself.

This makes "discover → use → check freshness → note it" one motion instead of three separately
memorable steps, which is what item 3 asked for. It does not change a single word of
`references/freshness.md` itself.

### 2.4 Exactly which files/sections change (implementation-facing, not executed here)

| Agent | `description` frontmatter | Body prompt section | New deliverable convention (§4) |
|---|---|---|---|
| `analyst` | `claude/analyst/analyst.md` L3 — reword "With a loaded Joern CPG, uses…" → default-check framing; add `coder`/`tdd-engineer`/`frontend-engineer` are *not* named here (this is analyst's own description, unaffected by their addition) | "How you work" step 2 ("Read the real thing," L45) | Review skeleton item 1 ("Scope & verdict," L61) |
| `architect` | `claude/architect/architect.md` L3 | "How you work" step 2 ("Investigate the codebase first," L36) | Plan skeleton item 2 ("Context & findings," L25) |
| `qa-engineer` | `claude/qa-engineer/qa-engineer.md` L3 | Phase 1 REASON, "Read the sources of truth" bullet (L28) | Test report "Summary" (L50) |
| `coder` | `claude/coder/coder.md` L3 — add the capability clause | Step 1 "Orient" (L18) | Step 5 "Verify and report" (L22) |
| `tdd-engineer` | `claude/tdd-engineer/tdd-engineer.md` L3 — add the capability clause | Step 1 "Understand first" (L35) | Step 5 "Verify honestly" (L42) |
| `frontend-engineer` | `claude/frontend-engineer/frontend-engineer.md` L3 — add the capability clause | "Orient first" numbered list — new item 4, after the existing three (L11-19) | Step 4 "Verify in the running UI" (L57) |

Line numbers are as of this design pass (2026-08-16) and are a locator for the implementer, not a
guarantee — re-find the anchor text if the file has moved since.

---

## 3. Spot-check evidence trail (AC-2)

**Convention: a one-line "CPG consideration" note, in whatever the agent already hands back** —
its written deliverable when it writes one (review, plan, test report), or its final inline
summary when it doesn't. Three allowed shapes, always one of:

- `CPG: used <graph> — <one clause on what for>` (e.g. `CPG: used cpg_falkorchat — impact
  analysis on Services.post_message before proposing the signature change`)
- `CPG: considered, not relevant — <one clause of reasoning>` (e.g. `CPG: considered, not
  relevant — this change is in opencode/, which has no loaded CPG`)
- `CPG: not applicable — <one clause>` for tasks with no code-level component at all (rare for
  the six wired agents, but keeps the convention total rather than silently absent on an
  edge case)

This is deliberately **not** a new section, checklist, or ceremony — it is one line appended to a
part of the deliverable that already exists for every wired agent (analyst's verdict line,
architect's findings section, qa-engineer's report summary, coder/tdd-engineer/frontend-engineer's
final "what changed" summary), which is what keeps it inside FR-4's "no material delay or noise"
budget. A stakeholder spot-checking a transcript greps for `CPG:` and gets a direct answer either
way, per AC-2's own wording ("evidence the agent considered/used the CPG, or an explicit, reasoned
not-relevant-here") — silence is what this convention rules out, not brevity.

---

## 4. Reconciliation with M2/M3 (FR-9, AC-6)

This plan **extends** — it does not override, narrow, or silently diverge from — the
consumer-scope boundary [`m2-cpg-analysis-skill.md`](./m2-cpg-analysis-skill.md) (M2) and
[`../requirements/cpg-query-access.md`](../requirements/cpg-query-access.md) (M3) drew. Concretely:

- **M2's shape is untouched by this document's own slice.** One `cpg-analysis` skill, lean
  `SKILL.md` core plus four bundled `references/*.md` recipes (impact-analysis, rca, code-review,
  test-gap), citing `skills/joern-cpg/references/cpg-model.md` as the single schema source. This
  plan (`cobb`'s slice, U2) adds zero new recipes and zero new skills — the same four recipes
  simply gain three more callers. *(The sibling plan, `cpg-agent-adoption-graph.md` §2, does add
  a fifth reference file, `references/freshness.md` — additive to the skill's file surface, but
  orthogonal to the consumer-scope question AC-6 governs: it doesn't touch who can query, only
  what one more recipe answers once queried. Mentioned here so a reader of this section alone
  isn't misled into thinking the combined M4 feature leaves the skill's file count untouched —
  only this document's own slice does.)*
- **M3's read path is untouched (FR-8, confirmed independently by `cpg-agent-adoption-graph.md`
  §4).** `mcp__cpg__query(graph, cypher)` keeps its two parameters, its `GRAPH.RO_QUERY`
  execution, its truncation/error behavior, and its documented `redis-cli` fallback. Nothing in
  this document changes `cpg/mcp/server.py`, `.mcp.json`, or the tool's contract.
- **What actually widens is the *consumer list* and the *default-ness* of discovery** — three
  agents (`coder`, `tdd-engineer`, `frontend-engineer`) join `analyst`/`architect`/`qa-engineer`
  as named consumers (§1), and the existing three's own wiring moves from conditional
  ("with a loaded CPG…") to default-orientation framing (§2). Both are additive: nothing the
  original three consumers could do before this plan becomes unavailable, and no access path
  changes for them.
- **The two prior documents remain historically accurate as written.** `m2-cpg-analysis-skill.md`
  and `cpg-query-access.md` describe the state as of their own delivery (2026-07-19 and
  2026-07-25 respectively) — a three-consumer roster and a single-tool read path. This document
  does not retroactively rewrite either; it records, here and in the M4 backlog section (§7),
  that M4 is a widening that happened *after* and *on top of* both, not a correction to them.

This satisfies AC-6's literal requirement — a downstream plan that "states explicitly that it
extends — not silently overrides — the consumer-scope boundary" — in this document's own words,
as required.

---

## 5. What this plan deliberately does not touch

Per the brief's "What NOT to do," restated as a checklist so the implementer (and any reviewer)
can verify scope discipline directly against this document:

- No change to `mcp__cpg__query`'s shape, parameters, execution mode, or the `redis-cli` fallback.
- No change to the freshness mechanics — the `:CpgBuildInfo` node, its properties, the
  `pipeline.sh` stamping step, or `references/freshness.md`'s content. Cited by path only.
- No automatic/proactive CPG build-out, no auto-rebuild trigger, no usage-tracking dashboard.
  Nothing in §2–§4 suggests, schedules, or implies any of the three.
- No agent prompt, `SKILL.md`, `AGENTS.md`, or `docs/BACKLOG.md` file is edited by this document.
  §6 is a task list for a later unit; §7 is a backlog-section proposal, not a write.

---

## 6. Implementation task list (for the later, separately-dispatched `cobb` unit — U4b)

Sequenced. U4a (`graph-dba`, freshness-marker implementation) lands first per the coordination
ledger — step 1 below touches the same file (`skills/cpg-analysis/SKILL.md`) that U4a's
`references/freshness.md` + nav-table row land in, so U4b should start **after** U4a is merged to
avoid two agents editing the same file concurrently, and should re-read the file fresh before
editing rather than assuming this design pass's line numbers still hold.

1. **`skills/cpg-analysis/SKILL.md` — broaden + document the discovery mechanic.**
   - Frontmatter `description`: widen the named consumers from "analyst, architect, or
     qa-engineer" to include `coder`, `tdd-engineer`, `frontend-engineer` (stay within the
     1024-char budget — trim other clauses if needed rather than let it overflow).
   - §1 "Finding the graph name": add the `cpg_<component>` naming-convention paragraph from §2.2
     above, as the first-guess discovery mechanic every wired agent's prompt will point to.
   - §4 "Navigation" table, **Consumer** column: update per recipe to reflect the post-M4 roster
     — the impact-analysis row gains `coder`, `tdd-engineer`, `frontend-engineer` (the recipe
     §1 of this document ties all three to); leave the rca/code-review/test-gap rows as-is
     (`analyst`/`qa-engineer`) unless a stated reason widens them too — none is stated here, so
     none change. Otherwise the skill's own internal consumer listing reads stale ("analyst,
     architect" only) the day after six agents are wired, which is the exact "one doc says three,
     another says six" drift this feature exists to close.
   - Done-check: re-read the file after U4a's freshness-recipe edit lands, confirm no
     nav-table/content collision, confirm the description still parses under the frontmatter
     schema (`skills/agent-standards/claude-code.md`'s skill-frontmatter rules) and stays under
     the character budget.

2. **`claude/analyst/analyst.md`, `claude/architect/architect.md`, `claude/qa-engineer/qa-engineer.md`
   — reword existing wiring from conditional to default.** Per §2.4's table: one description-line
   edit and one body-orientation-step edit each, plus the one-line evidence-trail convention (§3)
   added to each agent's deliverable skeleton. Run `cobb`'s own §7 prompt-quality lint (per
   `agent-maintenance`) on each changed description.

3. **`claude/coder/coder.md`, `claude/tdd-engineer/tdd-engineer.md` — add as new consumers.** Per
   §2.4: description clause, orientation-step sentence (§2.2/§2.3 content by reference), and the
   §3 evidence-trail line in each agent's final report step. No `tools:` frontmatter change needed
   — both agents omit `tools:` today and therefore already inherit `mcp__cpg__query` (verified
   against `skills/agent-standards/claude-code.md`, "Agents that omit `tools:` inherit
   everything").

4. **`claude/frontend-engineer/frontend-engineer.md` — add as a new consumer.** Same shape as
   step 3; the new "Orient first" list item should explicitly name `cpg_salesperson` /
   `chatbot.py` as the concrete case this agent will hit today, so the instruction reads as
   grounded rather than hypothetical. No `tools:` change needed (also omits `tools:`).

5. **Catalog + doc sync (one change, per this repo's own precedent at M2's C-208/M3's C-304).**
   - `claude/README.md` — update the six affected agents' capability-line entries: `architect`
     (row ~9), `coder` (row ~10), `tdd-engineer` (row ~13), `frontend-engineer` (row ~14),
     `qa-engineer` (row ~15), `analyst` (row ~16). Re-find by agent name, not line number, since
     earlier edits in this list shift later rows.
   - `skills/README.md` — update the `cpg-analysis` row's consumer list (currently "analyst,
     architect, or qa-engineer") to name all six.
   - Root `AGENTS.md` — update the `skills/` bullet's `cpg-analysis` description if it still reads
     as three-consumer-scoped after step 1's `SKILL.md` change (check current wording before
     editing; do not introduce a claim that then needs a second sync pass).
   - `docs/BACKLOG.md` — new `## M4 — CPG agent adoption` section + milestone-map row, per §7's
     proposal below. `docs/HISTORY.md` — dated delivery entry once M4 lands.

6. **Kaizen bookkeeping** for every agent file actually edited in steps 2–4 (per
   `agent-maintenance`): a dated `kaizen/history.md` entry per touched agent stating what changed
   and why (widened CPG consumer status / defaulted discovery wording).

7. **Gate.** Per the coordination ledger, U4b's diff goes to `analyst` for the U5 re-gate (a
   diff-scoped review distinct from this document's own U3 plan-gate), then `qa-engineer` for the
   U6 acceptance pass against AC-1…AC-6.

---

## 7. Proposed `docs/BACKLOG.md` M4 section (proposal only — not written to `BACKLOG.md`)

Mirrors the M2/M3 section shape and the `C-` numbering convention (hundreds digit = milestone, so
M4 = `C-4xx`).

```markdown
## M4 — CPG agent adoption

Widens which agents discover and use a loaded CPG (roster: `coder`, `tdd-engineer`,
`frontend-engineer` added to `analyst`/`architect`/`qa-engineer`), makes discovery a default
orientation step instead of a conditional one, and lets a consulting agent judge/flag graph
staleness via graph-dba's new `:CpgBuildInfo` freshness marker. Requirements:
[`requirements/cpg-agent-adoption.md`](./requirements/cpg-agent-adoption.md) (FR-1…FR-9 /
AC-1…AC-6) · plans: [`plans/cpg-agent-adoption-graph.md`](./plans/cpg-agent-adoption-graph.md)
(freshness mechanics, graph-dba) + [`plans/cpg-agent-adoption.md`](./plans/cpg-agent-adoption.md)
(roster/discovery/evidence-trail, cobb) · coordination:
[`plans/cpg-agent-adoption-coordination.md`](./plans/cpg-agent-adoption-coordination.md).

**Extends, does not override, M2/M3.** See `plans/cpg-agent-adoption.md` §4 — the MCP read path
and the skill's four recipes are unchanged; only the consumer list and the default-ness of
discovery widen.

### Items

- **C-401 — Freshness marker mechanics.** `:CpgBuildInfo` singleton node (BUILT_AT/SOURCE_PATH/
  SOURCE_COMMIT/SOURCE_DIRTY), stamped at the end of `pipeline.sh`'s `--load` branch after
  verification passes; new `skills/cpg-analysis/references/freshness.md` recipe; one nav-table
  row in `SKILL.md`. FR-5/FR-6 (mechanical)/FR-7/FR-8. Owner: `graph-dba` (unit U4a).
- **C-402 — `cpg-analysis` SKILL.md: broaden + discovery mechanic.** Frontmatter `description`
  widened to six consumers; `cpg_<component>` naming-convention paragraph (the one-query,
  no-noise-on-a-miss discovery mechanic) added to §1; Navigation-table Consumer column (§4)
  updated per recipe (m-2 fix, see §6 step 1). FR-1 (skill-side), **FR-4 / AC-5** (the discovery
  mechanic this item documents is what makes a miss cost one query and nothing else). Owner:
  `cobb`.
- **C-403 — Wire `analyst`/`architect`/`qa-engineer`: default-orientation reword.** Description +
  orientation-step + evidence-trail-line edits on the three already-wired agents, including the
  freshness-check bundling from §2.3 (run the freshness recipe as part of the same default step,
  note it, surface a refresh suggestion if stale). FR-2, FR-6 (surfacing integration) / AC-1,
  **AC-3, AC-4** (the freshness-signal-in-deliverable + stale-surfacing behavior lands here, on
  the three agents that consult the CPG for actual analysis today). Owner: `cobb`.
- **C-404 — Wire `coder`/`tdd-engineer` as new consumers.** Description clause + orientation
  sentence (including the §2.3 freshness-check bundling) + evidence-trail line. FR-1, FR-2, FR-3
  / AC-1, **AC-3, AC-4** (these two agents also bundle the freshness check into their orientation
  step per §2.3, not just the original three). Owner: `cobb`.
- **C-405 — Wire `frontend-engineer` as a new consumer.** Same shape as C-404, grounded in
  `cpg_salesperson`/`chatbot.py`. FR-1, FR-2, FR-3 / AC-1, **AC-3, AC-4**. Owner: `cobb`.
- **C-406 — Evidence-trail convention.** The `CPG: used | considered, not relevant | not
  applicable` one-line convention landed in all six wired agents' deliverable skeletons, including
  the freshness signal the deliverable reports when the CPG was actually consulted (§2.3/§3).
  **AC-2**, and reinforces **AC-3** (the freshness signal surfaces in the deliverable, not just
  inside the agent's reasoning). Owner: `cobb`.
- **C-407 — Catalog & doc sync.** `claude/README.md` (six rows), `skills/README.md`
  (`cpg-analysis` row), root `AGENTS.md` (skills bullet, if stale after C-402), this backlog →
  `HISTORY.md`. Per repo convention, lands in the same change as C-402…C-406. Owner: `cobb`.

### Requirements coverage

FR-1…FR-8 and AC-1…AC-5 each carry an explicit tag on at least one of C-401…C-406 above (see each
item's FR/AC line — C-401 for FR-5/FR-7/FR-8 and FR-6's mechanical half, C-402 for FR-1/FR-4/AC-5,
C-403 for FR-2/FR-6's surfacing half/AC-1/AC-3/AC-4, C-404/C-405 for FR-1/FR-2/FR-3/AC-1/AC-3/AC-4,
C-406 for AC-2/AC-3). **FR-9 and AC-6 are the one deliberate exception**, carried by no backlog
item because they aren't implementation work — AC-6's "states explicitly that it extends" is a
property of `plans/cpg-agent-adoption.md` §4 itself (already written, already satisfied), not a
task C-407 or any other item performs. `C-407` (catalog/doc sync) is untagged for the same reason
each catalog-sync item was in M2/M3's own backlog sections — it is process bookkeeping, not a
requirement-bearing deliverable.
```

Milestone-map row to append to the existing table:

```markdown
| **M4 — CPG agent adoption** 🔵 | Six agents (`analyst`/`architect`/`qa-engineer`/`coder`/
`tdd-engineer`/`frontend-engineer`) default-orient on CPG discovery, freshness is knowable via
`:CpgBuildInfo`, and a spot-checked transcript shows `CPG:` evidence either way — extends, does
not override, M2/M3 | **C-401 → C-407** |
```

---

## 8. Assumptions made without a stakeholder round (background run — stated per the brief)

- **Root `AGENTS.md` is not the discovery step's home** (§2.1) — a judgment call, reasoned
  explicitly rather than left implicit, because the requirements doc's own decision log did not
  settle the *mechanism*, only the *default character* of discovery.
- **The `cpg_<component>` naming convention is treated as durable enough to document as the first
  discovery guess**, even though it is an observed pattern from two data points, not a stated
  contract anywhere else in the repo. If a future CPG breaks the pattern, the fallback (error-
  message graph enumeration / `GRAPH.LIST`) still finds it — the guess is an optimization, not a
  requirement, so this is a low-risk assumption.
- **`coder`/`tdd-engineer`/`frontend-engineer` need no `tools:` frontmatter change** — verified
  against `skills/agent-standards/claude-code.md`'s "omit `tools:` to inherit everything" rule
  and cross-checked against `qa-engineer`, which is already CPG-wired today under the same
  no-`tools:`-field shape. Flagged explicitly in case a future harness version changes that
  inheritance default — re-verify against the skill's `Verified:` stamp before the implementation
  unit relies on it.
