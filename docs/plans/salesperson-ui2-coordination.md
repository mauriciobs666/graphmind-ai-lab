# The one salesperson UI — Coordination (continued)

> **Status:** active · **Owner:** `teco` · **Tracks:** — (M<n> TBD) · **Extends:** `docs/plans/salesperson-ui-coordination.md`

## Why this document exists

`docs/plans/salesperson-ui-coordination.md` carries the full history of this feature from the
first architecture-plan draft through the close of **S9** (all five sub-units, S9a-e, accepted and
committed) — over a dozen plan-gate passes, a dozen-plus implementation units, several with 2-4
review-gate rounds of their own, and ~110 individually-titled retrospective sections recording
specific incidents, corrections, and lessons along the way. That document **stays intact and
authoritative for everything through S9** — nothing here restates it; every fact from that phase is
cited by path, not re-narrated.

This successor exists because that predecessor crossed the point where continuing to grow it
imposed a real, rising cost on every future resume (the "reconcile the ledger before acting" step
means reading the whole thing), while a clean phase boundary exists exactly at S9's close: the
backend turn-mechanics work is done, and what remains — presenter-route relocation, the demo
bring-up script, the entire frontend, QA, and docs closeout — is a distinct, forward-looking phase
with no open units carried over. Per this repo's own doc-collision convention (`AGENTS.md`, rule 5:
once an earlier document has been "approved, gated, or executed against," a successor may be
written with an ordinal on the slug), this is licensed housekeeping, not a scope change. The
predecessor's `Status:` stays `active` — it isn't being retired, only extended — and gains an
`Extended by:` pointer to this file.

## Goal (unchanged — see the predecessor for the full statement)

Deliver the business-facing salesperson UI specified in `docs/requirements/salesperson-ui.md`
(FR-1…FR-11, AC-1…AC-11), replacing the retired standalone `salesperson/` Streamlit app.

**Definition of done:** AC-1…AC-11 verified; the old `salesperson/` app retired; documentation
(root `AGENTS.md`, component READMEs, `HISTORY.md`, a `tico` user manual) reflects the delivered
surface.

## State inherited from the predecessor (read the cited sections, don't take this list as a
## substitute for them)

- **S9 fully closed** — all five sub-units (S9a-e) accepted and committed; S9e (the last) landed at
  `dcec3f2`, review approved at Pass 28 (`docs/reviews/salesperson-ui-impl.md`). See the
  predecessor's `## Ledger` for the full S0-S9e history and its `RESUME HERE` section for the
  closing summary.
- **Both stakeholder decision points from 2026-09-09 are closed** — see the predecessor's
  `RESUME HERE` section, paragraph 2, for the disposition (nothing further routes through either).
- **The plan (`docs/plans/salesperson-ui.md`) is at v1.33, plan-gate lane closed** — no further
  Pass N re-gates on the plan document itself are expected; implementation units are gated
  individually against it.
- **`cpg_falkorchat` freshness**: check before relying on it for any S10/S11 unit — it was
  intentionally left stale through the whole S9 chain (per the predecessor's dispatch notes) to
  avoid tearing against live S9 units. That reason no longer applies now that S9 is closed; a
  rebuild is worth reconsidering before S10 dispatches if a unit leans on structural analysis.
- **Shared-tree hazard, worth carrying forward as a standing caution, not a premise**: a concurrent,
  unrelated coordination (`document-ingestion2`) was found editing the same `falkor-chat/server/`
  files S9e also touched, causing a real, recurring commit-hygiene collision (see the predecessor's
  S9e ledger row and Pass 27/28). No reason to expect this specific collision to recur on S10/S11's
  files, but the general pattern — check `git log`/`git status` for unexpected drift on shared files
  before trusting the working tree — is now a demonstrated risk in this repo, not a hypothetical.
- **Open, unanswered stakeholder question, carried forward**: whether to keep driving straight
  through S9f/S10/S11 and then stop to explicitly scope the frontend push (S12a-d/S13/S14) as its
  own decision point, or some other pacing. Not yet answered as of this document's opening.

## RESUME HERE — state as of 2026-09-11, opening this successor immediately after S9's close

**Read this section first.** Reconcile it against `git log` and `git status` before acting — if
they disagree, they win. Nothing is in flight as of this writing.

**Next units, in the order the predecessor's plan implies:**
- **S9f** — trivial per the predecessor's own note (effectively already answered by an earlier
  measurement); rides the next `qa-engineer` dispatch on this surface rather than its own unit.
- **S10** — presenter surface: relocate the three already-existing `presenter/*` routes from
  `storefront_api.py` onto `Storefront` itself, plus the login rate-limiter and reset-everyone's
  quiesce. Mostly a move, not new design.
- **S11** — demo bring-up script (`falkor-chat/scripts/start_demo.sh`).
- **S12a-d, S13, S14** — the entire frontend UI (zero code exists yet), routes to
  `frontend-engineer`. S12a is the blocking foundation (shared entry files) everything else in the
  UI depends on. **This is the point where the stakeholder's pacing question (above) should be
  resolved before diving in**, given its size relative to everything so far.
- **S15** — test suites & AC evidence (load harness, live-LLM run, mobile Playwright pass, versioned
  test plan/report) → `qa-engineer`.
- **S16** — docs close-out across root `AGENTS.md`, root `docs/HISTORY.md`,
  `falkor-chat/README.md`/`AGENTS.md`/`docs/SERVER.md`.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
