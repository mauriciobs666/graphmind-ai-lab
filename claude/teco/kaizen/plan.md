# Kaizen — Improvement Plan: teco

> Forward-looking backlog for the `teco` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-29

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-006 | 2026-07-16 | low | 🔵 | The review-default list assigns no independent reviewer for **agent-engineering (cobb) deliverables**; graph-dba design notes are only implicitly covered by "plans → analyst". |
| K-008 | 2026-07-29 | high | 🔵 | Use the `Agent` tool's own `model` parameter for per-delegation cost routing (cheap model for cost-insensitive units), instead of a persistent per-agent pin — verify it reaches nested calls before adopting. |
| K-009 | 2026-07-29 | medium | 🔵 | Audit whether teco itself ever uses its `WebFetch`/`WebSearch` grants (vs. delegating research to specialists) — drop if unused to trim per-spawn tool-schema overhead. |
| K-010 | 2026-07-29 | low | 🔵 | Next token-cost pass: description regrew 568→694 chars (+22%) and body 9,866→12,656 chars (+28%) since the 2026-07-11/24 compression passes — re-trim the description's trivial-fix clause (redundant with routing-table row 1) and recheck the body. |
| K-011 | 2026-07-29 | low | 🔵 | `.claude/settings.local.json` carries three single-use Bash allow-rules (exact escaped Cypher literals from the one-off K-001 probe run) that will never match again — prune. |

> **K-007 — SendMessage continuation instead of cold respawn — ✅ done 2026-07-29** (moved to
> history.md). Step 3 now has teco note each delegate's returned name/id when a follow-up round
> is likely; step 4's two re-brief paths (review "needs changes"/defects, and a deficient result)
> both `SendMessage` the original delegate by that identifier instead of a fresh `Agent` call,
> falling back to cold respawn only when the identifier no longer resolves. `tools:` gained
> `SendMessage` (was missing — K-007 was otherwise unshippable).
>
> **K-004 — deficient/failed-delegate-result path — ✅ done 2026-07-16** (moved to history.md).
> Step 4 now handles a deficient result (errored / out of turns / off-brief / empty): re-brief the
> same owner once with the gap explicit, pause to the user if it recurs or the unit is mis-scoped —
> distinguished from a *blocker* and a review *verdict*.
>
> **K-005 — doc-curation scan includes HISTORY.md / BACKLOG.md — ✅ done 2026-07-16** (moved to
> history.md). The documentation-impact scan now lists `docs/HISTORY.md` (entry per delivered change)
> and `docs/BACKLOG.md` where the module uses the convention.
>
> **K-002 — agent teams / background agents evaluation — ✅ done 2026-07-24** (moved to
> history.md). Disposition: **reject** reframing teco as an agent-teams lead — its loop is
> sequential/dependency-gated, exactly the shape the agent-teams docs say a single
> session/subagents handles better than teams. The 2026-07-12 sub-case (defect→fix→re-run
> respawning cold) isn't an agent-teams question — it's answered by `SendMessage` continuation
> of the original delegate. Spun off as **K-007**.
>
> **K-001 — validate nested delegation end-to-end — ✅ done 2026-07-09** (moved to history.md).
> Live run: falkor-chat M3 slice 1 through teco → architect → graph-dba → tdd-engineer, all
> checklist items passed. Launch brief + observation checklist preserved at
> [`k001-run-brief.md`](./k001-run-brief.md).
>
> **K-003 — review-gate invariant: prove it or renegotiate it — ✅ done 2026-07-12** (moved to
> history.md). Disposition **(a) keep the invariant** — the first fully-gated run (falkor-chat
> K-022 Landing 1) enforced the analyst gate and captured the cost datapoint: the gate is cheap
> (~7% of wall time, ~12% of tokens) and caught a major + minors on a "done" diff. No prompt
> change. Counterparts still open: `analyst` K-001, `qa-engineer` K-003 (unexercised — 0 blockers).

### K-006 — Review-default list has no reviewer for agent-engineering deliverables
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** The "work ships independently reviewed" invariant names defaults for plans/code (`analyst`), ML methodology (`data-scientist`), and behavior/acceptance (`qa-engineer`). A **cobb** agent/skill deliverable has no assigned independent reviewer, and a **graph-dba** design note is only implicitly a "plan → analyst". So the invariant ("every significant deliverable checked by someone other than its producer") has a coverage hole for the team's own agent-engineering work.
- **Proposed change:** Decide and state the reviewer for agent/skill deliverables (analyst on the prompt-as-artifact? a second cobb pass? explicitly out-of-gate for trivial agent edits) and confirm graph-dba design notes route to analyst review. Low priority — agent edits are infrequent and cobb self-lints via the §7 pass.
- **Notes:** Surfaced by cobb's §7 prompt-lint (semantic-coverage dimension), 2026-07-16.

### K-008 — Route cost-sensitive delegations to a cheaper model via the `Agent` tool's per-call `model` param
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** teco already carries the `Agent` tool, whose own schema accepts a per-call `model` override (`sonnet`/`opus`/`haiku`/`fable`). This is a materially different lever than the per-agent frontmatter `model:` pin the team explicitly rejected on 2026-07-27 ("model choice belongs at the session level, not duplicated across 13 frontmatter files") — that decision was about a *persistent*, duplicated config; a *per-delegation* override made by teco's own judgment at dispatch time doesn't reopen it, since it's reversible and scoped to one call. On a run like the K-022 example (~1.2M tokens across 10 units), routing cost-insensitive units (doc-only touch-ups, a re-review of a tiny diff, routine suite runs) to a cheaper model per-call could cut real spend without touching any specialist's identity.
- **Proposed change:** First verify the `model` override actually reaches an `Agent` call made *from inside* a subagent (teco is itself a subagent one level down from the top session) — confirm on one real run before relying on it. If confirmed, add a line to step 3 naming which unit shapes are candidates for a cheaper override (and which are not — anything needing design judgment or code-quality stakes stays at the inherited model).
- **Notes:** Surfaced by the 2026-07-29 credit/interface analysis session. Distinguish carefully from the rejected per-agent pin — the write-up for this item should make the "per-call, not persistent" distinction explicit so it doesn't get read as reopening that decision.

### K-009 — Tool-grant audit: does teco itself ever use `WebFetch`/`WebSearch`?
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** teco's own reads are local-repo docs (`AGENTS.md`, READMEs, `HISTORY.md`/`BACKLOG.md`); live web lookups are consistently a specialist's job (`cobb`, `architect`, `data-scientist`). No kaizen/history entry evidences teco itself calling `WebFetch`/`WebSearch` directly. Every granted tool's schema is injected into context on every spawn, so an unused grant is pure per-spawn overhead.
- **Proposed change:** Check recent teco transcripts/session logs for actual `WebFetch`/`WebSearch` invocations by teco itself (not by its delegates). If confirmed unused over a reasonable sample, drop both from `tools:` — delegates keep their own grants regardless, so no behavior loss.
- **Notes:** Surfaced by the 2026-07-29 credit/interface analysis session.

### K-010 — Scheduled token-cost recompression pass (description + body regrew after feature additions)
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** The description was compressed twice (1286→659→568 chars) but has regrown to 694 (+22%) since the 2026-07-25 trivial-fix feature landed; the body was compressed 15,023→9,866 (−34%) on 2026-07-11 but has regrown to 12,656 (+28%) since, from five legitimate feature additions. Not waste on its own — real functionality — but the description is paid on every session and every subagent spawn carrying the `Agent` tool (most of the roster), so it's the highest-multiplier text in the file.
- **Proposed change:** Next time teco.md is touched, trim the description's trailing clause ("does not design solutions and routes non-trivial implementation to a specialist; may fix a genuinely trivial single-file no-brainer directly instead of delegating it") down to the routing *signal* only — the routing table's row 1 already carries the tie-breaker prose in full. Re-check the body for the same kind of restatement once it's grown further; still 16% below the pre-compression peak, so not urgent.
- **Notes:** Surfaced by the 2026-07-29 credit/interface analysis session.

### K-011 — Prune single-use Bash allow-rules from `settings.local.json`
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** `claude/teco/.claude/settings.local.json` carries three highly specific, single-use Bash allow-rules — exact escaped Cypher literals from the one-off K-001 probe run (`k001-run-brief.md`). Claude Code's Bash permission matching is effectively exact-string for these, so they'll never match a future command; pure clutter, not a credit cost.
- **Proposed change:** Remove the three stale entries next time this file is touched.
- **Notes:** Surfaced by the 2026-07-29 credit/interface analysis session. Hygiene only, no functional impact.

## Parking lot / ideas
- **Watch the milestone-close freeze in a real close (noted 2026-07-27).** The new curation bullet is prompt-level only — nothing enforces that the `Status: archived` flips actually land, and the owners performing them (`architect`, `analyst`, `tico`, `qa-engineer`, `data-scientist`, `graph-dba`) have no matching instruction in their own prompts yet; they learn it from the brief. If a close ships with documents left `active`, the fix is either a line in each owner's prompt or the optional checker (step 7 of `docs/plans/doc-reference-convention.md`, which today gates nothing) — decide from evidence, not now.
- ~~A routing cheat-sheet / decision tree teco self-checks before delegating (which specialist for which signal), to reduce mis-routing between `coder` and `tdd-engineer`.~~ *(✅ Resolved 2026-07-09: the roster is now an explicit routing table — task shape → owner → tie-breaker — with the coder-vs-tdd efficiency rule on both implementer rows, plus a separate handoff-contracts list. See history.md.)*
- Guard against over-orchestration: a heuristic for "this is a single-specialist job, skip the breakdown."
- Minor §7-lint nits (2026-07-16, low value — noted not filed): (a) the Guardrail "`Write`/`Edit` is for the **coordination doc only**" is stricter than the enforcement it describes (the hook escalates only writes *outside* `docs/plans/`, permitting any file there) — prose and backstop are intentionally different scopes but read as if aligned. (b) The implementer-routing efficiency rule is stated three times (description, routing table, How-you-work) — deliberate reinforcement, some redundancy. (c) The Handoff-contracts list restates specialist doc paths that also live in each specialist's injected `description`, mild tension with teco's own "don't re-derive [descriptions]" line — but this is the §4 handoff-symmetry pattern (state on both sides), so it's a feature with a drift cost, not a defect.
