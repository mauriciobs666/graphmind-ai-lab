# Doc-reference-convention — `manuals/` + collision-rule-5 gloss + coordination-doc authorship — coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-005 (`claude/tico/kaizen/plan.md`)

## Goal

Close the paper trail between root `AGENTS.md` (already extended directly, live-enforced) and the
formal, heavily-reviewed spec at `docs/plans/doc-reference-convention.md` (Owner: `architect`,
Status: active, Tracks: C-322), per `tico`'s kaizen item K-005
(`claude/tico/kaizen/plan.md`:43-48). Three additive items, batched as one architect pass — none
reopens an existing ruling (D1/D4/D6 etc. stand):

1. **(a)** Add a `manuals/` mention to the convention doc: §9.2's `<kind>` list (currently
   `plans · reviews · requirements · test-plans · test-reports`, line ~1165) and §9.5 rule 2's
   family chain (line ~1298-1302), matching wording root `AGENTS.md` already ships: *"optionally
   `manuals/x.md` (only when the manual documents that exact feature end-to-end; a manual with
   broader scope is its own topic slug, not a family member)"*. Plus a dated changelog entry
   (this document uses versioned changelog amendments — v1.1 through v1.4 — under its own H1; the
   next one is v1.5).
2. **(b)** A gloss on §9.5 rule 5's selector (*"Has the earlier document been approved, gated, or
   executed against?"*, line ~1326-1334): a document reaching its approval **gate** alone (e.g.
   `Status: Ready for design`) does not by itself force a successor if nothing downstream has
   actually **executed** against the specific content being revised. Source: a 2026-08-20 kaizen
   distillation finding (`claude/tico/kaizen/history.md`, 2026-08-21 entry) — `docs/requirements/
   generic-cypher-mcp2.md`'s 2026-08-20 Decision-log entry and `docs/plans/generic-cypher-mcp2.md`'s
   `T1` precedent paragraph both establish this reading in practice; the rule's literal disjunctive
   text doesn't yet say so.
3. **(c)** A note that `plans/<slug>-coordination.md` (§9.4's closed role-set table, line
   ~1272-1283, `-coordination` row currently lists producer = `teco` only) may now be
   authored/extended by either `tico` (docs-only chains) or `teco` (any chain touching code), with
   the milestone-close `archived` flip staying `teco`'s regardless of authorship. Root `AGENTS.md`
   was deliberately left textually unchanged for this (the existing `teco` row already covers it
   without an edit) — the formal spec doesn't yet explain *why* a `tico`-authored coordination doc
   is still `teco`'s to flip.

All three are the same underlying job per K-005's own note: *"an architect pass adding clarifying
content to the same owned document — batch them rather than splitting into further K-items."*

## Units

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `a82a66991518c04b4` | accepted | `docs/plans/doc-reference-convention.md` (v1.5.1) · `docs/reviews/doc-reference-convention.md` Part IV (§21-30) | `analyst` (`af1ce8bafad36ea77`) → APPROVE (recheck §30, all 3 findings fixed correctly) | 132.4k+161.5k tok, 38+25 tools (architect) · 132.6k+146.3k tok, 47+12 tools (analyst) |
| U2 | `tico` | `af561bb521b420750` | accepted | `claude/tico/kaizen/{plan,history}.md` K-005 closed, commit `50afb1c` | none → — | 60.9k tok, 10 tools |

**Both units delivered and committed (`80ab320`, `50afb1c`). Coordination closed.**

U2 depends on U1's gated, committed result (it cites the landed change). Below the doc-only
ceremony floor for most tasks, but this crosses the coordination-doc threshold because U1 carries
a review gate (this document's own established practice — every substantive version bump, v1.1
through v1.4, was analyst-reviewed).

## Notes

- U1 is a **revise-in-place** per this very document's own collision rule 5 "No" branch (design
  precedent: v1.1-v1.4 are all in-place version bumps, not successor documents) — bump `Version:`
  in the header changelog block, add a dated v1.5 entry, no new filename, no new inbound
  references.
- Gate is scoped as a **light spot-check** (same shape as v1.4's "patch pass answering the analyst
  spot-check", not a full re-review): confirm no existing ruling reopened, the `manuals/` wording
  matches what root `AGENTS.md` already ships (so it isn't inventing new policy), and the two new
  glosses ((b) and (c)) are sound, evidence-backed, and don't contradict anything else in the
  20+ section document.
