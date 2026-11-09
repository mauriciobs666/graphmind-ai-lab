# python-web-quirks on the knowledge base — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-11-09

## Intent

The stakeholder's goal (2026-11-09): "Refactor the python-web-quirks skill completely so it relies
on a knowledge base provided by falkor-chat (GraphRAG) to retrieve context-relevant info as
needed."

Today an agent that loads `python-web-quirks` pays for the whole skill, but needs one gotcha at a
time. The wanted end state: an agent describing the situation it is in gets back the one gotcha
that applies, and nothing else is read.

## Problem & current state

Observed 2026-11-09:

- `skills/python-web-quirks/SKILL.md` is 76,291 bytes with 42 `##` sections, each a live-verified
  gotcha, plus a preamble outside those sections (H1, scope paragraph, a verified-version caveat,
  and a "this is a cache, not the source of truth" note with its lineage). The skill body is
  loaded whole when the skill activates.
- Only 16 of the 42 sections carry a date; 32 carry a version-like token.
- Some sections refer to one another by position ("the section above", "the two sections above").
- Skill activation is the only route to the content: no consumer agent prompt in the repo names
  the skill (`grep -rn python-web-quirks claude/*/*.md` finds only `claude/README.md`).
  `tico` and `security-expert` have `ingest_document` but not `search_documents`/`get_document`.
- A retrieval convention for the team knowledge base exists (`skills/agent-kb-retrieval/SKILL.md`,
  `ws:agent-team`); this feature follows it.
- Known platform hazards: K-100 (long text embedded by its head only) and K-102 (documents with
  unembedded chunks are invisible to semantic search), `falkor-chat/docs/BACKLOG.md`.
- Raw captures in `ws:agent-team` mention the skill; one verified (`38cd49cddb004207a5c7abd10181a4eb`)
  restates section 34 and states a gap the skill does not cover.
- Overlap: `docs/requirements/agent-reading-cost.md` (FR-1, FR-3, FR-6) covers retrieval of team
  documents and code, not skills; cited, not restated.
  `claude/docs/requirements/agent-knowledge-base-strategy.md` scopes the KB to all agent/team
  knowledge and left open a git-tracked counterpart for distilled content (decision log
  2026-09-10); FR-8 answers that for this content.

Method for FR-12 (golden sets, targets, floor, chunking): `docs/plans/python-web-quirks-kb-ml.md`.

## User stories

- As a coding or review agent, I want to describe my situation and get the one applicable
  gotcha, so that I do not read ~76 KB to use a few paragraphs.
- As the stakeholder, I want the content to stay correct and findable as it grows, so that the
  skill does not repeat the size problem.
- As the team maintainer, I want to add, correct or retire a gotcha through one defined path, so
  that the knowledge base and its source never disagree.
- As an agent working while the knowledge base is unreachable, I want to know it, so that "no
  result" is never mistaken for "no gotcha applies".

## Functional requirements

- FR-1 — Retrieval by situation: given a situation description, the knowledge base returns the
  gotcha that applies, without the agent reading unrelated gotchas.
- FR-2 — No loss: every gotcha, and the preamble's content, is retrievable in full after
  migration, with verification evidence and version pins intact. The stored text of a gotcha is a
  delimited metadata header (title, versions, date or "unverified, migrated <date>") plus the
  gotcha body verbatim; the verbatim comparison covers the body only. Verified mechanically (the
  team's content-loss check is the precedent). The preamble is migrated or explicitly
  dispositioned, and its "this is a cache" statement is corrected to match FR-8.
- FR-3 — A hit is a complete gotcha and says which gotcha it is (title) whichever part of it
  matched. Sections that cross-refer to one another are one retrievable unit, verbatim; the list
  of such units is derived from the cross-references at migration start.
- FR-4 — Every migrated gotcha is searchable in every part: none may be invisible to semantic
  search (K-102) or reachable only through its beginning (K-100); reported per gotcha.
- FR-5 — A hit (through the stored header, FR-2) shows the gotcha's title, its library versions where the section names them, and
  its verification date where the section states one; otherwise the hit reads "unverified,
  migrated <date>". No date or version is invented (staleness rule:
  `docs/requirements/agent-reading-cost.md` FR-6).
- FR-6 — The `python-web-quirks` skill no longer exists after migration: no skill directory,
  no catalog row. Retrieval is agent-initiated.
- FR-7 — Retrieval follows the team's existing retrieval convention; this feature does not fork
  it. Any threshold or top-K change this content needs is made there, for all users. That
  convention names this content class (Python/web/test gotchas) so an agent knows to look there,
  without changing its guarded query-prefix literal (the one `claude/scripts/audit-team.sh`
  checks).
- FR-8 — A version-controlled copy of every gotcha exists outside the knowledge base: the source
  of record, at a fixed location that is not an auto-loaded skill body. The knowledge base can be
  rebuilt from it, and a gotcha can be reviewed in a diff.
- FR-9 — When the retrieval call fails (server unreachable, call erroring), the agent reports
  that explicitly and reads the needed gotcha from the version-controlled copy, one section at a
  time, never the whole file. The guidance for this is file-backed, not in the knowledge base,
  and reachable within one file read from the pointer in FR-14 (the plan picks the home; the
  shared retrieval convention file is the natural one). An empty result is never the only signal.
- FR-10 — One defined path adds, edits and retires a gotcha; the result reaches both the
  version-controlled copy and the knowledge base (raw capture, then `cobb`), and `cobb` can still
  write to the copy after migration (its write guard allows it). A divergence between the two is
  detectable. `cobb` derives from the workspace the list of raw captures that mention the skill,
  confirms coverage, and considers the gap statement in the verified capture for promotion; the
  stakeholder deletes the captures.
- FR-11 — An agent can see all gotcha titles in one bounded read without reading bodies, in
  normal and degraded (FR-9) modes. The degraded listing has a source that stays in step with
  the copy (covered by the FR-10 divergence check).
- FR-12 — Delivery is measured against the golden sets and targets approved by the stakeholder
  and set out in `docs/plans/python-web-quirks-kb-ml.md` §1 and §4: sealed-set positive recall,
  coverage of every gotcha in the derived unit list (a hit on a merged unit counts for each
  section in it), and negative rejection split off-domain/near-domain, at the stated figures.
  The dev set may be tuned against; the sealed set is scored once. Once the unit list exists, the
  method note's set sizes and chunk counts are re-derived (plan-time obligation).
- FR-13 — Every consuming agent can call the retrieval tools (`search_documents`,
  `get_document`) directly. The consuming agents are the skill's audience clause (`coder`,
  `tdd-engineer`, `architect`, `analyst`, `qa-engineer`) plus `frontend-engineer`, `tico` and
  `security-expert`, as the stakeholder named. Agents with no `tools:` line (all tools) need no change; every consuming
  agent whose `tools:` line enumerates tools and lacks them gets the grant. The edits are
  `cobb`'s, through the normal agent-update path.
- FR-14 — Each consuming agent carries a discovery pointer: it tells the agent that the knowledge
  base holds Python/web gotchas and to use the team retrieval convention for the call. Where an
  agent already carries the retrieval-convention line (checked 2026-11-09: all but `coder`,
  `tico`, `security-expert`, `cobb`), the pointer extends that line rather than adding a second.
  Wording is the plan's; the edit is `cobb`'s.
- FR-15 — Removing the skill leaves nothing dangling: every guard and citation of the skill file
  is updated or retired (the AC-9 sweep; the cobb write guard's `skills/python-web-quirks/*`
  entry is retired in the same change as the copy's new location, AC-12). `cpg/eval/kit-manifest.json`
  is not edited: it hashes a `git archive` of the pinned eval commit, which still holds the skill.

## Out of scope

- Converting any other skill (this skill only; not a pilot).
- Changing the wording of any gotcha beyond what migration formatting needs; re-verifying gotchas
  against current library versions.
- Changing the embedding model, chunking design or `search_documents` itself; K-100 and K-102
  are hazards gated by FR-4, not scoped in.
- Token-cost targets for whole agent runs (`docs/requirements/agent-reading-cost.md` AC-1).
- Any agent-definition change beyond the tool grants (FR-13) and the one-line pointer (FR-14).

## Acceptance criteria

- AC-1 (FR-2) — Given the whole of `skills/python-web-quirks/SKILL.md` at a pinned commit (the
  baseline; the section count is derived from it, not fixed), when the content-loss check runs
  against the knowledge base, then every section's body (verbatim) and the preamble's content is accounted for, and
  an added extra section would fail it. Stored metadata headers (FR-2) are outside the comparison.
- AC-2 (FR-4, FR-3) — Given the migrated gotchas, when each is queried by its title, by a
  situation, and by a phrase from its last paragraph, then each query returns that gotcha, the
  hit names its title, and zero chunks are unembedded. The cross-reference units are listed and
  each is retrieved as one unit.
- AC-3 (FR-1, FR-12) — Given the sealed set, scored once through the retrieval convention, then
  the FR-12 targets hold, counted over the derived unit list. A negative passes when none of the gotcha documents is returned under
  the score floor (other team documents are ignored).
- AC-4 (FR-6) — Given the migration done, then `skills/python-web-quirks/` does not exist and no
  skill catalog lists it.
- AC-5 (FR-8) — Given the knowledge base emptied of this content in a test workspace, when it is
  rebuilt from the version-controlled copy, then the content-loss check (body comparison) run against that copy
  (not the gutted skill), with its manifest regenerated, and AC-2 pass again.
- AC-6 (FR-9) — Given the server unreachable, and separately a retrieval call that errors, when
  an agent looks up a gotcha, then it reports the failure and, following the guidance reachable in
  one file read from its pointer, retrieves one section from the version-controlled copy
  without loading the whole file.
- AC-7 (FR-10) — Given a gotcha added, one edited and one retired through the defined path, then
  each is correct in both the copy and the knowledge base (AC-2 for the added and edited ones;
  the retired one is no longer returned). Given a planted divergence between the two, the
  divergence check fails and names the gotcha.
- AC-8 (FR-5, FR-11) — Given a hit for a dated section, the hit shows its date; for an undated
  one it shows "unverified, migrated <date>"; versions are shown where the section names them.
  Given the title listing, all titles are visible in one bounded read, in normal and degraded
  mode; the plan states the exact call and the bound. A 43rd title added to the copy appears in
  the degraded listing or the divergence check fails.
- AC-9 (consistency, FR-15) — Every file in the repo citing `skills/python-web-quirks` is listed
  (unfiltered search) and each is updated (re-pointed to the retrieval route or the copy, or the
  mention removed) or has a stated no-change ruling. The kit manifest gets a no-change ruling
  plus a check that re-exporting the kit at the pinned eval commit still reproduces it. At least:
  `skills/README.md`, root `AGENTS.md`, `claude/README.md`, `falkor-chat/docs/SERVER.md:638`,
  `skills/agent-standards/claude-code.md:165`, `docs/requirements/cpg-agent-adoption.md`,
  `falkor-chat/server/falkorchat/app.py`, `falkor-chat/server/falkorchat/storefront_api.py`,
  `falkor-chat/docs/plans/workflow-timers.md`, `cpg/eval/kit-manifest.json` (no change, see FR-15) and `claude/cobb/hooks/guard-cobb-topic-writes.sh` (lists
  `skills/python-web-quirks/*`; entry retired, and the copy's final location allowed).
  Frozen reviews and kaizen files need a one-line ruling.
- AC-10 (FR-13) — Given the eight consuming agents, then each can call both tools: listed in its
  `tools:` line, or tools unrestricted. Checked 2026-11-09: `analyst`, `architect` list both;
  `tico`, `security-expert` do not (the grants expected); `coder`, `tdd-engineer`,
  `frontend-engineer`, `qa-engineer` have no `tools:` line.
- AC-11 (FR-10) — Given the raw captures mentioning the skill, then `cobb`'s derived coverage
  list exists, the verified capture's gap statement has a promotion ruling, and the stakeholder's
  deletion is recorded.
- AC-12 (FR-10, FR-15) — Given the migration done, when `cobb` writes a gotcha to the
  version-controlled copy, then its write guard allows it, and the guard no longer lists
  `skills/python-web-quirks/*`.
- AC-13 (FR-14) — Given a fixed literal for the pointer, a search across the eight consuming
  agents finds it in each (an existing-line case for `frontend-engineer`, as for the others that
  carry the retrieval line); removing it from one (a mutant) turns the check red.
- AC-14 (FR-7) — Given the retrieval convention file, it names the Python/web/test gotcha content
  class, and `claude/scripts/audit-team.sh` still passes (prefix literal unchanged).

## Open questions

None.

## Decision log

2026-11-09 — Stakeholder goal received: refactor `python-web-quirks` to rely on a falkor-chat
GraphRAG knowledge base for context-relevant retrieval → recorded as Intent; scope decisions
open (FR-1–FR-12).
2026-11-09 — Scope: all 42 gotchas or a subset? → All 42, one pass (FR-2, AC-1).
2026-11-09 — Is this the pilot for other skills? → No, this skill only (Out of scope).
2026-11-09 — Behaviour when falkor-chat is down? → Read the git-tracked copy by section, never the whole file, plus an explicit "KB unavailable" signal (FR-9, AC-6).
2026-11-09 — Git-tracked source of record required? → Yes; the KB is a rebuildable index (FR-8, AC-5).
2026-11-09 — Which workspace? → `ws:agent-team` (FR-7).
2026-11-09 — Who owns authoring after migration? → Raw capture, then `cobb` (FR-10, AC-7).
2026-11-09 — Retrieval-quality target? → No number now; `data-scientist` proposes golden set and minimum top-5 recall, stakeholder approves; resolved by the 2026-11-09 entry below (FR-12, AC-3).
2026-11-09 — Is losing whole-skill browsing acceptable? → Yes, given a cheap title listing (FR-11, AC-8).
2026-11-09 — How are agents prompted to consult the KB? → Skill trigger description only, no consumer prompt edits (FR-6, Out of scope).
2026-11-09 — Keep the skill? → Yes, a thin stub (FR-6, AC-4).
2026-11-09 — Wait for K-100/K-102? → No; gate on every chunk searchable and check chunk behaviour on the largest sections first (FR-4, AC-2).
2026-11-09 — 16 of 42 sections carry a date; how should a hit for an undated one read? → "unverified, migrated <date>", versions shown where the section names them, no content change (FR-5, AC-8).
2026-11-09 — `tico` and `security-expert` lack `search_documents`/`get_document`: grant them? → Yes, both; edit is `cobb`'s via the agent-update path; partially overturns the 2026-11-09 "no consumer prompt edits" entry: tool grants only, no prompt text (FR-13, AC-10, Out of scope).
2026-11-09 — Sections that cross-refer: reword or co-locate? → Co-locate as one retrievable unit, content verbatim; unit list derived from the cross-references (FR-3, AC-2).
2026-11-09 — Golden set and recall target? (supersedes the "Retrieval-quality target?" entry) → Approved as proposed in `docs/plans/python-web-quirks-kb-ml.md`: sealed-set recall at least 0.90 top-5 (76/84), all 42 hit, at least 27/30 negatives rejected (15/15 off-domain, 12/15 near-domain); dev set tunable, sealed set scored once (FR-12, AC-3).
2026-11-09 — How is a negative scored? → Passes when none of the gotcha documents is returned under the floor; other team documents are ignored (AC-3).
2026-11-09 — The four raw captures mentioning the skill? → `cobb` confirms coverage and lists them; the stakeholder deletes them (FR-10, AC-11).
2026-11-09 — Cross-referencing units change the method note's 42-document counts: how handled? → Counts re-derived once the unit list exists; a hit on a merged unit counts for each section in it (FR-12, AC-3).
2026-11-09 — Consuming agents (coder, tdd-engineer, etc.) searching the KB directly, and the skill deleted? → Yes: no stub skill; every consuming agent has the retrieval tools and a one-line discovery pointer, edits by `cobb`; supersedes the "How are agents prompted to consult the KB?" entry (description only, no prompt edits), the "Keep the skill?" entry (thin stub) and the tico/security-expert-only scoping of the grant (FR-6, FR-13, FR-14, FR-15, AC-4, AC-10, AC-13).
2026-11-09 — Where do degraded-mode guidance and the version-controlled copy live now? → Left to the plan: guidance reachable from the pointer, copy at a fixed location outside any auto-loaded skill, pins and guard follow it (FR-8, FR-9, FR-15, AC-9).
2026-11-09 — Which agents are the consuming agents? → The skill's audience clause (`coder`, `tdd-engineer`, `architect`, `analyst`, `qa-engineer`) plus `tico` and `security-expert` as the stakeholder named; `frontend-engineer` was in the 2026-11-09 consuming-agents entry's list but not in the audience clause, so it is not required (it already has all tools and the retrieval line); `data-scientist`, `devops`, `graph-dba`, `teco` are outside the clause and already carry the retrieval line (FR-13, FR-14, AC-10, AC-13).
2026-11-09 — Should the cpg kit manifest be edited when the skill is deleted? → No: it hashes a `git archive` of pinned eval commit `cf7b2042`, which still holds the skill; no-change ruling plus a re-export check (FR-15, AC-9).
2026-11-09 — Add `frontend-engineer` to the consuming agents? (amends the "Which agents are the consuming agents?" entry) → Yes; the set is eight (FR-13, FR-14, AC-10, AC-13).
2026-11-09 — Are the requirements ready for design? → Stakeholder confirms; analyst Pass 3 approve with suggestions, growth-stops check included (Status).
