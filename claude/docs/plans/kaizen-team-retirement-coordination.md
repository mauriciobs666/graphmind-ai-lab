# `kaizen_team` retirement — deletion + doc sweep

> **Status:** archived · **Owner:** `teco` · **Tracks:** — (—)

User-directed follow-on to `claude/docs/plans/kaizen-distillation3-coordination.md` (that pass
drained `kaizen_team` to 0 `KaizenEntry` nodes and closed with the deletion decision explicitly
left to the user). User confirmed explicitly in this session: **"please go ahead and delete
it."** This coordination covers the actual destructive deletion (approval-gated, now granted) plus
the documentation sweep it invalidates — deleting the graph makes several currently-live docs
describe a thing that no longer exists, which is part of "done" per the standing documentation
curator duty, not scope creep.

**Scope boundary, stated up front:** only docs that assert `kaizen_team` as a **currently live/
existing** fact are in scope. The large number of frozen `docs/plans/`, `docs/reviews/`,
`docs/requirements/`, `docs/test-reports/`, `docs/test-plans/`, and `docs/HISTORY.md` mentions
found by a repo-wide grep (~100+ files) are correctly left untouched — they are historical record
of work done while `kaizen_team` existed and are not process/routing bugs (root `AGENTS.md`:
"A document that freezes does not move"; "History is not context"). Also deliberately left
untouched: `claude/teco/coordination-techniques.md`'s one `kaizen_team` mention (a properly-cited
historical worked example inside a coordination technique, not a live-fact assertion) and
`cypher-mcp/README.md`'s example graph references (the curator/producer write-shape *mechanism*
that server documents is generic, not `kaizen_team`-specific — flagged as a follow-up, not a unit,
since a since-deleted example graph name in a syntax example doesn't mislead anyone into acting on
a false current fact the way an agent's own prompt would).

**Docs found asserting `kaizen_team` as currently live** (repo-wide `grep -rln kaizen_team
--include=*.md .`, filtered to non-frozen/always-loaded/current-reference docs):
- Root `AGENTS.md` — `claude/` structure bullet ("every agent's raw capture writes into one
  shared `kaizen_team` FalkorDB graph").
- `claude/AGENTS.md` — the write-cutover note still describes `kaizen_team` as an existing (if
  no-longer-written-to) graph, not a deleted one.
- `claude/README.md` — detailed dual-store description, several passages describing `kaizen_team`
  as "fully live in parallel."
- `claude/cobb/cobb.md` — the maintainer's own "Learnings distillation" duty description, and the
  `ws:agent-team`-clear note that says `kaizen_team`'s older shape "stays available, unchanged."
- `claude/teco/teco.md` — **teco's own always-loaded prompt**, "Learnings ride the handoff" line,
  checks a delegate wrote into `kaizen_team` — wrong destination now. `teco` cannot self-edit; this
  routes through `cobb` like any other agent prompt.
- Every other agent's `.md` prompt (`claude/{analyst,architect,coder,data-scientist,devops,
  frontend-engineer,graph-dba,qa-engineer,security-expert,tdd-engineer,tico}/*.md`) — all carry
  the identical boilerplate sentence in their "Learning capture" section: "`kaizen_team`'s older
  shape (...) stays available, unchanged, for any entry already there." Mechanical, uniform fix
  across all of them.
- `skills/agent-maintenance/SKILL.md` §5 — the distillation procedure itself is built around
  reading `kaizen_team` as one of three live sources; needs the `kaizen_team`-reading steps
  retired, leaving `ws:agent-team` as the sole live read source, while preserving the section's
  existing historical/consolidation narrative as history (cited, not restated).
- `claude/docs/manuals/team-knowledge-base.md` (tico-owned) — has a whole FAQ entry built on
  "why does `kaizen_team` (the old graph) still exist?" — now literally false.

## Units

- **U1 (`devops`)** — the destructive deletion itself + verification. Must complete, and be
  independently verified by `teco`, before U2/U3 write "is deleted" as a stated fact.
- **U2 (`cobb`)** — the full `claude/`-side + root-`AGENTS.md` sweep (all agent prompts, `cobb.md`,
  `teco.md`, `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`, `skills/
  agent-maintenance/SKILL.md` §5). One coherent editorial pass — cobb's own domain.
- **U3 (`tico`)** — `claude/docs/manuals/team-knowledge-base.md`'s FAQ rewrite. Disjoint files
  from U2 — runs in parallel with U2, both sequenced after U1.

**No independent review gate**: `teco` independently re-verifies every file U2/U3 touch (diffs,
no stale references left via a fresh grep, no duplicate headings where relevant) before
acceptance, same discipline as the just-closed distillation pass — narrow, mechanical, high-file-
count but low-design-risk edits, not a design or implementation deliverable.

**Commit posture:** U1 is a live infra action, nothing to commit for it beyond the ledger record.
U2+U3 are pure `.md` edits (docs-only chain) — held uncommitted until both are accepted, then
committed together by explicit path in one commit, per `claude/AGENTS.md`'s docs-only batching
convention.

## Ledger

| Unit | Agent | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | devops | `a1e38ec570eadccbd` | accepted | `GRAPH.DELETE kaizen_team` executed against shared `falkordb-dev`; teco-reverified via `mcp__cypher__query(graph='GRAPHS')` — 8 graphs remain, exactly the pre-delete set minus `kaizen_team` | none (see above) → — | 60k tok, 9 tools |
| U2 | cobb | `a31c4d526914b2821` | accepted | 17 files: root `AGENTS.md`, `claude/AGENTS.md`, `claude/README.md`, `claude/cobb/cobb.md`, `claude/teco/teco.md`, 11 agent prompts (uniform boilerplate removal), `skills/agent-maintenance/SKILL.md` §5 rewrite — teco-reverified diffs + dup-heading scan + final grep, no orphaned live-fact mentions remain | none (see above) → — | 224.9k tok, 85 tools |
| U3 | tico | `a38d57a119ec74d5d` | accepted | `claude/docs/manuals/team-knowledge-base.md` (overview bullet + FAQ rewrite, both kaizen_team mentions correctly reframed as history) | none (see above) → — | 72.5k tok, 10 tools |

## Follow-ups

- **`cypher-mcp/README.md`'s example graph references** — the curator/producer write-shape
  mechanism it documents is generic infrastructure, not `kaizen_team`-specific; the examples just
  happen to use the now-deleted graph's name for illustration. Not misleading enough to warrant a
  unit in this pass (no agent would act on a false current fact from it), but worth a note next
  time that README is touched for another reason.
- **`claude/docs/reviews/security-expert.md:139`** (`Status: active`) suggests "a candidate for a
  future `kaizen_team` entry" as one option for an open cross-cutting finding — stale now (the
  graph is gone) but harmless: not a load-bearing routing instruction, trivially substitutable
  with `ws:agent-team` if that finding is ever actually acted on. Found during `teco`'s U2
  verification sweep; not worth a dedicated unit for one word in an unrelated active review.

## Close

Both units accepted, independently re-verified by `teco` (diffs, duplicate-heading scans, a final
fresh repo-wide grep for orphaned `kaizen_team` live-fact claims, and a Status-header check on
every review/plan doc the grep still turned up) before acceptance. `kaizen_team` is deleted
(U1, independently re-confirmed) and no currently-live `claude/`-side doc, root `AGENTS.md`, or
the team-knowledge-base manual describes it as existing any more — every remaining mention across
the repo is either this pass's own correctly-framed historical record, or a pre-existing frozen
document left untouched per this coordination's stated scope boundary. Docs-only chain: U2 (17
files) + U3 (1 file) + this document (19 total) are committed together, by explicit path, in one
commit per `claude/AGENTS.md`'s batching convention.
