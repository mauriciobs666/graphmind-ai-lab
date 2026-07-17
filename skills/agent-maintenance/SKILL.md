---
name: agent-maintenance
description: Procedures for maintaining agent/skill artifacts — kaizen plan & history upkeep, dual-audience documentation (human README catalog + agent-context files), file-location conventions, the audit/reconcile method for already-drifted context docs, the team-coherence certification pass (inter-agent rosters, handoff contracts, hook enforcement parity), the learnings-inbox distillation procedure (verify → route → log → clear each agent's kaizen/inbox.md), and the single-artifact prompt-quality lint (§7 — contradiction, ambiguity, persona, cognitive-load, coverage, composition-conflict review of one prompt/skill/steering doc). Use whenever creating, editing, renaming, removing, or reviewing a Claude Code / OpenCode / Kiro agent, subagent, skill, steering doc, or memory file — or when asked to certify/audit an agent team, lint a single prompt's quality, or process its learnings inboxes.
allowed-tools: Read, Write, Edit, Glob, Grep, Bash
---

# Agent maintenance

The bookkeeping that keeps an agent collection healthy: every agent/skill you
touch carries a living **kaizen** plan + history (agents additionally a
learnings inbox, §5), and stays **documented for two audiences** (humans and
other agents). This skill holds the procedures and
templates so the resident agent prompt stays lean — load it when you do any
maintenance work, follow it, and mention at the end which kaizen/doc files you
touched.

## When this applies

- **Creating** an agent/skill → seed kaizen files, add catalog entries.
- **Editing** an agent/skill → advance kaizen, update its catalog entry.
- **Renaming / removing** → update or delete entries everywhere.
- **Reviewing** (no source change) → still record new improvement ideas in `plan.md`.
- **Reconciling** an already-drifted context doc → run the audit pass (§3).
- **Certifying team coherence** (rosters, handoff contracts, enforcement parity across the collection) → run the certification pass (§4).
- **Distilling learnings inboxes** (on request, and folded into every certification pass) → run the distillation procedure (§5).
- **Linting a single artifact's prompt quality** (on authoring/review, and folded into §4) → run the prompt-quality lint (§7).

---

## 1. Kaizen — improvement plan & history

Every agent/skill you create or touch carries a forward-looking `plan.md` and a
dated `history.md`. Keep them as part of the work, not as an afterthought.

### Where the files live

Locate the artifact's **development directory** — the folder its source lives in:

- **Has its own folder** (a skill's `<dir>/<name>/SKILL.md`, or an agent in its
  own subdirectory like `~/.claude/agents/<name>/<name>.md`) → the development
  directory is that folder.
- **A lone file sharing a directory** with sibling artifacts (e.g. flat OpenCode
  `.opencode/agents/<name>.md`, or Kiro `.kiro/steering/`) → the development
  directory is that shared directory.

Place the kaizen files as:

- **Own folder:** `<folder>/kaizen/plan.md` and `<folder>/kaizen/history.md`
  (no extra nesting — the folder is already artifact-specific).
- **Shared directory:** `<dir>/kaizen/<name>/plan.md` and `.../history.md`
  (namespace by `<name>` so siblings don't collide).

Example (per-agent folders): `~/.claude/agents/cobb/kaizen/plan.md` and `.../history.md`.

### Procedure

1. **Creating:** create both files. Seed `history.md` with a dated "created"
   entry and `plan.md` with improvements you already foresee. In collections
   that run the learning-capture loop (§5 — graphmind-ai-lab's `claude/` does),
   also seed an empty `inbox.md` from the §5 template.
2. **Modifying:** before editing, check `plan.md` for relevant items; after
   editing, append a dated `history.md` entry (*what* changed and *why*), and
   update the status of any plan items you advanced — move completed ones out of
   the active table into `history.md`.
3. **Reviewing (no change):** record new ideas in `plan.md` even if not implemented now.
4. **Always** read existing kaizen files first — don't duplicate items, and
   respect prior decisions (including things explicitly rejected/deferred).

Use `Read`/`Glob` to check for existing files, `Write` to create, `Edit` to
update. Keep entries concise.

### `plan.md` template

```markdown
# Kaizen — Improvement Plan: {name}

> Forward-looking backlog for the `{name}` {agent|skill}.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: YYYY-MM-DD

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | YYYY-MM-DD | high/med/low | 🔵 | … |

### K-001 — {title}
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** why this matters
- **Proposed change:** what to do concretely
- **Notes:** open questions, links

## Parking lot / ideas
- {half-formed ideas not yet prioritized}
```

### `history.md` template

```markdown
# Kaizen — Change History: {name}

> Dated log of actual changes to the `{name}` {agent|skill}. Most recent first.

## YYYY-MM-DD — {short title}
- **What:** what changed
- **Why:** motivation / trigger
- **Plan items:** K-00X (if this closed or advanced a planned item)
```

---

## 2. Documentation — keep both audiences informed

Every agent/skill you create, edit, rename, or remove must stay documented for
**two distinct audiences**, as part of the same change — never leave docs
trailing the source.

### Audience 1 — Humans → `README.md`

A human-facing catalog `README.md` at the **root of the agents collection** (the
directory holding the agent folders/files), or the repo root if there is one.
One entry per agent/skill, kept in sync.

Each entry: the **name**, a one-line **what it does**, **when to use it**, and
links to its **source file** and its **`kaizen/` folder**. On edits update the
entry; on removal delete it. Don't advertise per-agent deployment choices
(model, tool lists) in the catalog — frontmatter is their single source of
truth and the catalog copy just drifts.

### Audience 2 — Agents → the project's context convention(s)

So *other* agents in the project know this agent exists, also record it in
whatever agent-context convention the project uses. **Detect what's present and
update each that's in use** (don't blindly create all of them):

| Ecosystem | File / location | Notes |
|-----------|-----------------|-------|
| Claude Code | `CLAUDE.md` (nearest in tree, or `~/.claude/CLAUDE.md`) | Claude-specific project rules |
| Open / cross-tool · OpenCode | `AGENTS.md` (project root, or `~/.config/opencode/AGENTS.md`) | The portable standard |
| Kiro | `.kiro/steering/*.md` | e.g. an `agents.md` steering doc with `inclusion: always`, or a note in `structure.md` |

If **none** exists, create the one matching the active tool — default to
`CLAUDE.md` inside a `.claude/` tree, `AGENTS.md` otherwise. Keep these entries
**concise**: name, purpose, pointers to source + kaizen files — do **not** paste
the whole system prompt; point to it. Keep them in sync on edit/rename/remove.

**Don't duplicate the same catalog into two files.** When a project would carry
identical content in both `CLAUDE.md` and `AGENTS.md`, keep one source of truth
and have the other import it: a `CLAUDE.md` of just `@AGENTS.md` pulls the
catalog in (Claude Code `@`-import; tool-specific, not part of the portable
standard). Put the content in `AGENTS.md` (broadest reach) and point `CLAUDE.md`
at it.

### In-scope vs. cross-scope (which duty fires when)

Two different obligations with two different correct mechanisms:

- **In-scope (per-edit):** you edit an agent → update *its* kaizen + *its*
  catalog entry. Fires every time, in the same change. Resident duty.
- **Cross-scope (reconcile):** keeping the repo-root catalog reflecting *all*
  components/agents is **not** a per-edit push — a session scoped to one
  component may never see the parent catalog. Treat it as an on-demand
  **reconcile pass** (§3), run when asked to "sync the docs" or when you notice
  drift, not bolted onto every edit.

### No personal information in committed artifacts (rule)

Committed artifacts — frontmatter hook commands, scripts, configs, docs, kaizen
logs, anything git tracks — must never contain the maintainer's **personal
identifiers**: home path (`/home/<user>/…`), OS username, real name, email, or
hostname. A home path additionally breaks on every other machine. In prose
(kaizen entries, origin notes), genericize: `/home/<user>/…`, "the maintainer".
When a tracked file must reference a deployed script, anchor it to an
expansion-safe location:

- **User-scope agents** (symlinked into `~/.claude/agents/`):
  `$HOME/.claude/agents/<name>/hooks/<script>.sh` — resolves through the
  deployment symlink on any machine. Shell-form hook commands (no `args`) run
  via `sh -c`, so `$HOME` expands (verified 2026-07-10 against
  `code.claude.com/docs/en/hooks`).
- **Project-scoped hooks** (settings.json in a repo): `${CLAUDE_PROJECT_DIR}` —
  but it does **not** fit user-scoped agents, which must guard in projects
  where the project dir isn't the agents' repo.

The certification script (§4, check 7) greps **every tracked file in the
repo** for the runtime-derived identifiers (`$HOME`, `id -un`, git
`user.name`/`user.email`, `hostname`) and fails on any hit — the patterns are
never hardcoded in the script (that would itself be the leak), so the check
protects whoever runs it.

> Origin: 2026-07-10 — six agents' frontmatter hook commands had been committed
> with the maintainer's absolute `/home/<user>/prg/…` path; the user then asked
> for the guardrail to cover all personal information, not just the home dir.

### Order of operations when you create or edit an artifact

1. Write/edit the agent or skill source.
2. Update its `kaizen/{plan,history}.md` (§1; agents also carry `inbox.md`, §5).
3. **If you added, renamed, or removed an agent:** update every prompt that
   **enumerates the team** in the same change — an orchestrator's roster (e.g.
   teco's "The team you coordinate"). Other agents' prompts are consumers of
   the roster too; catalogs alone won't catch this drift class (origin:
   2026-07-09, teco's roster silently missed two specialists created after
   it). Better still: don't create enumerated summary facts ("all N agents use
   model X") in always-loaded context docs at all — they duplicate frontmatter
   ground truth, cost tokens every session, and rot; delete them when found.
4. Update `README.md` (humans) and the relevant context file(s) (agents).
5. Mention at the end which docs you touched.

---

## 3. Audit & reconcile a drifted context doc

A standalone pass to bring an *already-drifted* `AGENTS.md` / `CLAUDE.md` /
steering doc back in line with repo reality — distinct from the "sync on my own
edits" duty above. Use when a doc has silently fallen behind (missing whole
components or agents that exist on disk).

1. **Enumerate ground truth.** `git ls-files` for the real file tree; read each
   component's `README` / `CLAUDE.md` / `SKILL.md` headers and each agent's
   frontmatter to learn what actually exists.
2. **Diff against the doc's claims.** List what the context doc currently
   advertises; compare to ground truth.
3. **Reconcile.** Add missing entries, fix changed ones, remove entries for
   things no longer present. Preserve the doc's existing structure and altitude.
4. **Apply the DRY import rule** (§2) if you find the same catalog duplicated
   across `CLAUDE.md` and `AGENTS.md`.

> Origin: surfaced reconciling graphmind-ai-lab's root `AGENTS.md`, which had
> silently lost the entire `falkor-chat/` component, the `graph-dba` agent, and
> the `severino` OpenCode agent.

---

## 4. Team coherence certification (inter-agent audit)

The doc audit (§3) checks *catalogs vs. disk*. This pass checks the
**interfaces between agents** — the drift class catalogs can't see. Run it
when a specialist is added/renamed/removed, when an orchestrator or a handoff
contract changes, or on demand ("certify the team").

> Origin: the 2026-07-09 teco review — `qa-engineer` and `devops` had existed
> for days with perfect catalog entries while teco's roster still enumerated a
> five-agent team, and several delegates carried "ask one sharp question"
> phrasing that assumes an interactive session subagents never get.

**Deterministic half — run the script first:**
`claude/scripts/audit-team.sh` (read-only, exit 1 on any FAIL) verifies the
greppable invariants: every agent folder has its `<name>.md` + kaizen pair, is
symlinked into `~/.claude/agents/`, its frontmatter hook commands exist and
are executable (after mirroring the shell-form `$HOME`/`~` expansion), every
agent is named in the orchestrator's prompt and in all three catalogs, each
declared boundary pair (`BOUNDARY_PAIRS` in the script — coder↔tdd-engineer,
analyst↔qa-engineer, graph-dba↔devops) names its partner in its frontmatter
`description` (the routing contract every router sees), and no tracked file
anywhere in the repo contains the maintainer's personal identifiers —
home path, username, git name/email, hostname, derived at runtime (the
personal-info rule, §2). Fix any FAIL before judging the rest.

**Judgment half — checklist (what the script can't see):**

1. **Roster accuracy** — the orchestrator's roster describes each specialist's
   *current* contract (deliverable paths, conventions, guardrails), not just
   its name.
2. **Handoff symmetry** — every producer/consumer convention is stated on
   *both* sides (e.g. the architect's plan-doc path on architect **and** every
   implementer that consumes it; the qa-engineer's plan/report paths on
   qa-engineer **and** the orchestrator).
3. **Subagent-awareness** — every delegate-able agent handles the
   can't-ask-mid-run constraint: questions, blockers, and approval requests
   return as the deliverable; no "ask" phrasing that assumes an interactive
   session.
4. **Enforcement parity** — every prompt guardrail that claims harness
   enforcement has a live hook, and every wired hook is described in the
   prompt it guards (no silent machinery, no hopeful prose posing as
   enforcement).
5. **Boundary reciprocity** — when agent A says "defer X to B", B actually
   claims X (and vice versa for "B routes X to me"). The name-level half is
   scripted (check 6 greps each declared pair's `description` for the
   partner); whether the claimed scopes actually complement each other stays
   judgment — and when a new specialist borders an existing one, add the pair
   to `BOUNDARY_PAIRS` in the script.

**Fold in the single-artifact lint (§7):** run the prompt-quality lint over
every artifact changed since the last certification — the semantic,
intra-artifact defects (contradiction, ambiguity, persona, cognitive load,
coverage, composition conflict) that the five inter-agent checks above can't
see. Mirrors the §5 inbox-distillation fold-in; roll its findings into the
certificate.

**Certificate:** log a dated entry in the maintainer's kaizen history (cobb's,
in graphmind-ai-lab) recording scope, script result, findings (including any
§7 lint findings), and fixes — so "when was the team last certified?" is
answerable from the log.

---

## 5. Learnings inboxes — capture & distillation

The self-improvement loop for a stateless agent team: **capture is cheap and
unreviewed; promotion is curated.** Each agent carries an append-only
**learnings inbox** at `<agent>/kaizen/inbox.md` (sibling of plan/history — in
graphmind-ai-lab, enforced by `audit-team.sh` check 1). During runs the agent
appends dated, evidence-backed observations of **durable, non-obvious
environment facts in its discipline** — tool quirks, undocumented behaviors,
conventions that live only in the code. Agents never promote their own
entries; the inbox is the only kaizen file an agent writes itself (the
doc-scoped write guards allowlist exactly that path). The maintainer (cobb)
distills — on request, and folded into every certification pass (§4):

1. **Read every inbox** (`claude/*/kaizen/inbox.md`).
2. **Verify each entry** — is it still true? Re-check cheaply against the live
   system or docs; environment facts rot on upgrades. Unverifiable ≠ discard —
   date-stamp the doubt and keep or drop by value.
3. **Route each surviving entry to exactly one destination:**
   - **The agent's always-loaded prompt** — only if it changes behavior or
     routing in most sessions. Highest bar: every session pays tokens for it.
   - **An on-demand knowledge base** — the `graph-dba/falkordb-quirks.md`
     pattern: dated, live-verified entries in a `<agent>/<topic>.md` referenced
     from the prompt but loaded on demand. Create one when an agent's domain
     facts accumulate; don't inline them into the prompt.
   - **Project docs** (`AGENTS.md`, the component's `docs/`) — facts about *the
     project* belong where every agent sees them, never hoarded in one agent's
     private files where they drift out of sync.
   - **Discard** — stale, task-specific, or already documented.
4. **Log & clear.** Every promotion gets a dated entry in the agent's
   `history.md` (what, why, where it went); processed entries are then removed
   from the inbox — the history entry is the durable record. Promotion into a
   prompt or catalog is a normal agent edit: full §1/§2 bookkeeping applies.

**Inbox template** (seed on creation; keep the header, entries append below):

```markdown
# Kaizen — Learnings Inbox: {name}

> Append-only capture of durable, non-obvious environment facts the `{name}` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

*(empty — no unprocessed learnings)*
```

> Origin: 2026-07-12 — the user asked how the agents could self-improve from
> what they learn exploring their areas; the answer generalized graph-dba's
> quirks-file pattern into a team-wide capture→distill loop.

---

## 6. Testing standards (reference)

For *how* to test the agents/skills you maintain — the two-altitude standard
(pytest for deterministic code; the eval/bless harness for agent behavior) and
the reusable agent-eval-harness pattern — see **`claude/cobb/TESTING.md`** in the
graphmind-ai-lab repo. Keep it in sync when the harness pattern evolves.

---

## 7. Prompt quality review (single-artifact lint)

The passes above are *structural* (catalogs vs. disk, §3) and *inter-agent*
(the interfaces between agents, §4). This one is *semantic* and
*intra-artifact*: a judgment lint of a **single** prompt — an agent/subagent
system prompt, a `SKILL.md` body, a steering doc, a memory file — for the
defects an LLM reviewer can see but a grep can't. Run it when authoring or
reviewing any one artifact, and — folded into §4 — over every artifact changed
since the last certification.

It is LLM judgment, not a script: read the artifact, then work the six
dimensions below. For each, emit findings as **`finding — severity
(blocker / major / minor) — suggested rewrite`** (a concrete rewrite where the
fix is a wording change; a pointer where it's structural). "Clean" on a
dimension is a valid result — say so rather than inventing a finding.
Severity for a prompt: **blocker** = would cause wrong behavior in most
sessions; **major** = a real gap that bites in some sessions; **minor** =
polish. When one issue spans dimensions, **report it once under the most
informative dimension** (note the others in a clause) rather than filing it
several times.

1. **Contradiction** — logical, behavioral, or format conflicts *within* the
   one prompt. Probes: does any instruction countermand another (an "always X"
   and an "except-when-X" placed far apart)? Do two rules demand incompatible
   outputs (terse vs. exhaustive; JSON-only vs. prose)? Does a tool/permission
   grant collide with a "never do Y" rule?
2. **Semantic ambiguity** — instructions open to more than one reading. Probes:
   pronouns/scope with no clear referent ("update it"); undefined thresholds
   ("large files", "when appropriate") with no operational test; sequencing
   left implicit where order matters. Rewrite each into one operational reading.
3. **Persona consistency** — one coherent voice and altitude. Probes:
   conflicting traits (terse *and* thorough; deferential *and* decisive) with
   no resolving rule; tone drift across sections; role labels that overclaim
   (the 2026-07 "senior"-removal class). Rewrite toward a single stance.
4. **Cognitive load** — is the prompt followable in one pass? Probes: count the
   hard rules and competing priorities (when many items are top-priority, none
   is); nested/conditional depth (if-A-unless-B-except-C); always-loaded bulk
   that belongs in a progressively-disclosed skill (the Lean-context principle,
   here for instruction-following load, not just tokens). Suggest splitting,
   ranking, or demoting to on-demand.
5. **Semantic coverage** — gaps in the intents the artifact claims to handle.
   Probes: for each stated responsibility, is the *success* path **and** the
   failure/empty/blocked path specified? Are the triggers for each named
   handoff defined? For a subagent, is the can't-ask-mid-run path covered
   (question returns as the deliverable)? Flag the missing path; don't invent
   scope the artifact never claimed.
6. **Composition conflict** — the artifact never runs alone. **Resolve its full
   load-set first** — the `CLAUDE.md`/`AGENTS.md` chain that auto-loads, every
   `@`-import, wired skills, and steering with a reaching `inclusion` mode
   (what reaches a subagent differs per tool — verify via the `agent-standards`
   skill) — then re-run the contradiction / persona / coverage probes over the
   *combined* context. Probes: does the artifact restate (and now contradict) a
   rule its memory file already sets? Does a loaded skill's persona fight the
   prompt's? Do two loaded sources give conflicting defaults? Report each
   conflict as a pair (artifact rule ↔ load-set rule) with which should win.

**Output:** a per-dimension list of findings (or "clean"), each with severity
and a rewrite/pointer. On a review-only pass, record the notable ones in the
artifact's `plan.md` (§1). This is a lint, not a gate — it surfaces; the author
(or the user) decides.

> Origin: 2026-07-16 — the user asked whether cobb's machinery covered six
> LLM-judgment prompt dimensions (contradiction, ambiguity, persona, cognitive
> load, coverage, composition); it covered only structural (§3) and inter-agent
> (§4) drift. Promoted from cobb's dormant "self-review checklist" parking-lot
> idea (re-flagged 2026-06-07).
