# Tico proactive specialist collaboration — Feature Requirements
> **Status:** archived · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-08-29

## Intent
Today, when Tico's own work (a requirements interview, an explanation, a manual) runs into
something outside its own knowledge or verification ability, it can only *offer* to route a
finished artifact to a specialist, or *offer* a Mode-3 verification pass — the stakeholder must
accept before anything happens, and the exchange is a single fire-and-forget round: Tico hands
over a path, gets back a finished artifact, and folds it in. The stakeholder wants Tico to be a
more actively engaged member of the team: able to pull in the right colleague (architect, analyst,
graph-dba, data-scientist, and others) on its own initiative when a review is needed, and to hold
a real back-and-forth with that colleague — follow-up questions, clarifications — rather than one
shot in, one shot out. The stakeholder also wants to be able to ask Tico a technical question that
goes deeper than Tico can answer alone (e.g. "why is the ontology modeled this way") and have Tico
bring in the right specialist to get a grounded answer, for as long as that takes, while staying
inside the boundaries of Tico's own work (requirements capture, explanation, manuals) — never
turning into Tico directing code implementation.

Two distinct flavors of engagement are in scope, not one:
1. **Review-shaped** — Tico's own in-progress or finished work (a requirements doc, a manual, a
   claim in an explanation) gets checked by the right specialist before Tico relies on it or hands
   it off.
2. **Fast-tracked direct Q&A** — a quick technical question, not tied to any artifact under review,
   fired off to a colleague mid-conversation and answered back directly (e.g. "does the ontology
   actually support that relationship today?"), so Tico doesn't have to guess, stall, or hand the
   stakeholder off to a separate conversation just to get one fact.

## Problem & current state
- Tico's current `Agent` use is scoped to four things, all **offered, never initiated**: a wide
  read-only `Explore` sweep, an *offered* Mode-3 verification pass (`qa-engineer`/`analyst`) on a
  new or substantially-rewritten manual, an *offered* live demo via `devops`, and *offering* to
  route an already-finished artifact onward (`architect`/`analyst`/`qa-engineer`). Each is a single
  round: brief out, deliverable back, no continuation.
- Tico has no `SendMessage` tool and no mechanism to resume a specialist it already spawned — every
  new question to a colleague today would mean a fresh, cold `Agent` spawn with no memory of the
  prior exchange, if Tico even had standing to initiate one.
- **This reverses part of a previously recorded decision.** On 2026-07-30, a stakeholder proposal
  to widen Tico's authority toward orchestrator-like behavior was evaluated by `cobb`
  (`claude/cobb/kaizen/history.md`, 2026-07-30 entry, "Declined 'give tico commit authority over
  its summoned team'") and **declined**, on grounds including: root `AGENTS.md`, `claude/AGENTS.md`
  and `claude/README.md` all frame Tico as never orchestrating a team, `teco`'s own routing table
  states "`tico` is not a delegation target" (a different claim — about `teco` not delegating *to*
  Tico — but part of the same "single orchestrator" framing), and `claude/scripts/audit-team.sh`
  hardcodes a single `ORCHESTRATOR="teco"`. That review was about **commit authority** specifically
  and was evaluated on its own terms; this request is narrower in one way (no commit-authority
  change is being asked for) and broader in another (proactive initiation + multi-turn follow-up,
  not just artifact hand-off) — but it lands squarely inside the same "should Tico behave like a
  second orchestrator" question the 2026-07-30 review closed against. Any design built from this
  document must explicitly reconcile with that decision (supersede the relevant parts of it,
  narrowly, rather than leave the two decisions silently contradicting each other) — see FR-9 and
  the acceptance criteria below.
- Tico's current guardrails are explicit that "Routing is not coordinating": sequencing several
  units of work, gating them, or chaining one delegate's output into another's brief is `teco`'s
  job. The stakeholder's request must be scoped to preserve that line, not erase it.

## User stories
- As a stakeholder, I want Tico to bring in the right specialist on its own initiative when it
  judges a review is genuinely needed, so that I don't have to remember to ask for a check every
  time one would help.
- As a stakeholder, I want Tico to be able to have a real back-and-forth with the specialist it
  calls in — asking follow-up questions, getting clarifications — instead of a single round-trip,
  so that the consultation actually resolves the question instead of coming back half-answered.
- As a stakeholder, I want to ask Tico a technical question that's beyond what it can verify alone
  (e.g. graph/ontology design rationale) and have it loop in the right specialist to get me a
  grounded answer, so I don't have to go start a separate conversation with that specialist myself.
- As a stakeholder, I want Tico to be able to fast-track a quick technical question to a colleague
  and get the answer back directly, mid-conversation, without that question needing to be wrapped
  up as a formal "review" of some artifact first — so a one-fact question doesn't carry the
  overhead of a full review round.
- As a stakeholder, I want this new engagement to stay inside Tico's own lane — requirements,
  explanations, manuals — and never extend into Tico directing code implementation or sequencing
  multi-unit delivery work, so that `teco`'s coordination role isn't duplicated or confused.

## Functional requirements
- **FR-1 — Proactive initiation.** Tico may initiate a specialist consultation on its own
  judgment, not only offer one, when its own work (a requirements interview, an explanation, a
  manual) surfaces a review-shaped need or a fact-finding gap it cannot close itself.
- **FR-2 — Roster.** The colleagues reachable under this capability are: `architect`, `analyst`,
  `graph-dba`, `data-scientist`, `qa-engineer`, `security-expert`, `devops`. `coder`,
  `tdd-engineer`, and `frontend-engineer` are excluded — code implementation is never directed by
  Tico.
- **FR-3 — Announce, then proceed.** Before making a proactive call, Tico states in one line who
  it's calling and why, then proceeds without waiting for explicit stakeholder approval; the
  stakeholder can stop or redirect it in the moment. This applies uniformly, including to the
  manual-verification case that was previously an offer (see FR-7).
- **FR-4 — Two interaction shapes.** Both are supported:
  - **Review-shaped**: a colleague checks a specific piece of Tico's in-progress or finished work
    (a doc section, a manual claim, a requirements assumption) and may return a written finding or
    annotate a deliverable per its own existing conventions.
  - **Fast-tracked direct Q&A**: a single, quick technical question is sent to a colleague and
    answered back directly, inline in the conversation, with no artifact produced and no formal
    review framing.
- **FR-5 — Multi-turn follow-up.** Either interaction shape may extend into a follow-up exchange
  with the *same* specialist (continuing that consult, not a fresh cold spawn) when the initial
  answer needs clarification.
- **FR-6 — Stopping rule.** A multi-turn consult ends when: the original question is resolved, the
  specialist reports it cannot progress further, or roughly 3-4 rounds pass without resolution —
  whichever happens first. Tico then reports the outcome (resolved or not) to the stakeholder in
  plain language.
- **FR-7 — Manual-verification supersession.** The existing Mode-3 behavior — `qa-engineer`
  verifying a manual's walkthroughs, `analyst` verifying its factual/architectural claims, for a
  new or substantially-rewritten manual — stops requiring stakeholder acceptance beforehand and
  becomes a proactive, announced call under FR-3.
- **FR-8 — Traceability.** Every consult Tico initiates under this feature, fast-track or
  review-shaped, however brief, is recorded in the visible decision-log trail of the document
  Tico is working in (or, when a Mode-2 explanation has no open document, stated to the
  stakeholder as part of the answer) — naming who was asked, why, and what came back.
- **FR-9 — Boundary with `teco`.** This capability is limited to single-topic consultation
  (one review, one question, one follow-up exchange) — it never becomes multi-unit work
  breakdown, gating, or chaining one delegate's output into another's brief. That remains
  `teco`'s coordination-ledger job. Any implementation of this feature must explicitly record
  which part of the 2026-07-30 `cobb` design-review decision it supersedes (proactive
  initiation and multi-turn technical/review consultation) and which parts stand unchanged
  (Tico's commit-authority scope), so the two decisions don't silently contradict each other.

## Out of scope
- Tico directing or sequencing code implementation work (that stays `teco` → `coder`/
  `tdd-engineer`/`frontend-engineer`).
- Tico taking on multi-unit work breakdown, gating, or chaining one delegate's output into
  another's brief — that is `teco`'s coordination-ledger job, not this feature.
- Any change to Tico's commit authority (unaffected by this request; the 2026-07-30 review's
  commit-authority conclusion stands as-is unless a future request reopens it specifically).
- Consulting `coder`, `tdd-engineer`, or `frontend-engineer` under this capability, even for a
  quick question — implementers are reachable only through `teco`.
- Any standing/durable coordination ledger analogous to `teco`'s — a consult under this feature is
  single-topic and its record is the decision log entry (FR-8), not a resumable multi-unit ledger.

## Acceptance criteria
1. **Given** Tico finishes writing or substantially rewriting a manual, **when** the write-up is
   done, **then** Tico announces (one line: who, why) and proceeds to call `qa-engineer`
   (behavioral claims) and/or `analyst` (factual/architectural claims) without first waiting for
   stakeholder acceptance, and the call is recorded in the decision log.
2. **Given** a stakeholder asks Tico a technical question Tico cannot verify from docs/code it has
   already read (e.g. an ontology design question), **when** Tico recognizes the gap, **then**
   Tico announces a fast-track question to the appropriate specialist (e.g. `graph-dba`) and
   returns the answer inline in the conversation, without wrapping it as a formal review.
3. **Given** a specialist's first answer in a consult leaves the original question unresolved,
   **when** Tico determines clarification is needed, **then** Tico continues the same specialist
   exchange (not a fresh cold spawn) for further rounds, stopping once the question is resolved,
   the specialist says it cannot progress further, or ~3-4 rounds pass without resolution —
   reporting the outcome to the stakeholder in all three cases.
4. **Given** any consult Tico initiates under this feature, **when** it starts, **then** a
   decision-log entry (or, absent an open document, an explicit statement to the stakeholder)
   names who was asked and why, and is updated with the outcome once the consult concludes.
5. **Given** this feature ships, **when** `tico.md`'s own guardrail language (currently: "offer,
   never dispatch unasked"; "Routing is not coordinating") is read afterward, **then** it has been
   revised to state the new proactive-consultation behavior precisely, without contradicting the
   still-standing "no multi-unit sequencing, no commit-authority change, no implementer roster"
   boundaries from this document's Out of scope.
6. **Given** this feature ships, **when** `claude/cobb/kaizen/history.md`'s 2026-07-30 entry is
   read afterward, **then** the implementing change's own record (its own history/plan entry, or
   a pointer from this document) states explicitly which part of that decision is now superseded
   and which parts remain standing — no silent contradiction between the two records.
7. **Given** Tico is asked to consult `coder`, `tdd-engineer`, or `frontend-engineer` directly, or
   to sequence/gate multiple units of delegated work, **when** that request is made under this
   feature's authority, **then** Tico declines and points to `teco` instead.

## Open questions
None outstanding — all resolved in the decision log below and folded into the FRs and acceptance
criteria above.

## Decision log
- 2026-08-29 — Surfaced that this request partially reverses the 2026-07-30 `cobb` design-review
  decision declining orchestrator-like behavior for Tico. Stakeholder request stands; documented
  as an explicit, narrow supersession rather than a fresh, unrelated FR (see Problem & current
  state and FR-6).
- 2026-08-29 — Autonomy level: **announce, then proceed** — Tico states who it's calling and why
  in one line, then makes the call without waiting for explicit approval first; the stakeholder
  can stop it in the moment.
- 2026-08-29 — Roster: **full non-implementer roster** — architect, analyst, graph-dba,
  data-scientist, qa-engineer, security-expert, devops are all in scope depending on topic;
  coder/tdd-engineer/frontend-engineer remain excluded (code-implementation stays `teco`'s job).
- 2026-08-29 — Traceability: **log every consult**, even a quick fact-check, in the working
  document's decision log — who was asked, why, what came back.
- 2026-08-29 — Clarified scope is **not limited to routing/reviewing finished artifacts**: a
  second, distinct flavor is in scope — a fast-tracked direct technical question, fired to a
  colleague mid-conversation via `SendMessage` and answered back directly, with no artifact or
  formal review round involved. Folded into Intent/User stories as flavor 2.
- 2026-08-29 — The existing Mode-3 manual-verification **offer** (qa-engineer/analyst check on a
  new or substantially-rewritten manual) is superseded by this feature's "announce, then proceed"
  behavior — it stops being an offer the stakeholder must accept and becomes a proactive call
  Tico makes and announces, same as every other review trigger here.
- 2026-08-29 — Stopping rule: **resolution-based with a safety valve** — a multi-turn consult runs
  until the original question is actually answered (or the colleague says it can't go further),
  but if it passes roughly 3-4 rounds without resolving, Tico stops and surfaces that to the
  stakeholder instead of continuing silently.
- 2026-08-29 — Confirmed the review-trigger list (manual verification, a requirements-interview
  fact-check, an explanation accuracy check) is complete as stated — no missing case.
- 2026-08-29 — Drafted FR-1 through FR-9, the full Out of scope list, and acceptance criteria 1-7
  from the decisions above; all open questions resolved and folded in.
- 2026-08-29 — Stakeholder confirmed the full readback. **Status → Ready for design.**
- 2026-08-29 — `cobb` designed and implemented the feature end-to-end (commit `fc0fb5b`:
  `tico.md`, hook cores, `claude/AGENTS.md`, `claude/README.md`, kaizen), with the required
  2026-07-30-decision reconciliation logged in `claude/cobb/kaizen/history.md`, and
  `audit-team.sh` verified clean. Fully delivered in the same session it was requested — nothing
  left for this document to track. Stakeholder requested archival. **Status → archived.** The
  shipped `claude/tico/tico.md` is now the source of truth for this behavior; live end-to-end
  exercise of the new capability remains tracked as open kaizen items (K-010/K-011/K-012 in
  `claude/tico/kaizen/plan.md`), not by this document.
