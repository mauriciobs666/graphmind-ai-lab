# Kaizen-team distillation — U7 (`data-scientist`) gate review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (no backlog id; `docs/plans/kaizen-team-distillation-coordination.md`, unit U7, the last unit)

## Scope & verdict

Reviewed the uncommitted working-tree diff produced by `cobb`'s standing kaizen-distillation pass
(`skills/agent-maintenance/SKILL.md` §5) over the shared `kaizen_team` FalkorDB graph's
`data-scientist`-produced raw `:KaizenEntry` node, unit U7 (the last unit) of
`docs/plans/kaizen-team-distillation-coordination.md`. One file changed:
`claude/data-scientist/kaizen/history.md` (new dated 2026-09-17 entry, no other file touched).
Every factual and duplication claim in the new entry was independently traced to current source —
not taken on `cobb`'s report — and the graph was independently re-queried, both for
`data-scientist` alone and, since this closes the whole 7-unit coordination, across all seven
targeted agents.

**Verdict: approve.** No blocker, no major, no minor. One informational observation below (not a
defect) on the whole-graph sweep the brief asked for.

**CPG:** not applicable — this is a documentation-only kaizen-distillation task (one dated prose
entry in a Markdown history file), with no code-level component to load a CPG for.

## Findings

None that require action. Everything checked held up:

1. **The underlying fact — verified true, independently re-derived.**
   `falkor-chat/server/falkorchat/proof_defs.py:363-367` — `SALESPERSON_DEF` `v7`'s `systemPrompt`
   carries exactly one language instruction: *"Reply in the language named by `language` in the
   CONTEXT block; if no language is named there, reply in English."* No other per-language text
   appears anywhere else in the prompt. `falkor-chat/server/falkorchat/executor.py:1276-1277`
   (`_append_turn(messages, "user", f"CONTEXT:\n{context}")`) confirms the per-run `language`
   value reaches the model only as a trailing `CONTEXT:` JSON block appended after the whole
   thread history — a single JSON key at the tail, never restated in `systemPrompt` text. Both
   citations check out exactly as the entry states.
2. **The duplication claim — verified genuine, not a thin rationalization for discarding real
   content.** Read `docs/plans/salesperson-ui-ml.md` in full (Mitigations + Recommendation).
   Mitigation D (lines 181-193) states the identical observation — *"The language cue is
   currently a single JSON key ... at the tail of a long, mostly-prompt-identical-across-
   participants message — a weak, easy-to-miss signal"* — with strictly more precision than the
   raw kaizen entry (concurrency framing, a named concrete fix: interpolate the language directly
   into `systemPrompt` text via a `v8` bump) and is ranked **#1** in the Recommendation's
   priority-ordered mitigation list (lines 208-212). The doc's header confirms `Owner: `
   `data-scientist``, `Status: active`, and `git log` confirms it was authored 2026-09-16 (commit
   `6ddf88b`) — the same day as the raw entry's own `date` field, matching "captured while
   mid-writing a still-open plan doc" exactly. `falkor-chat/AGENTS.md`'s K-065 note ("Before
   running a live, audience-facing storefront demo, check `docs/BACKLOG.md`'s `K-065`... confirmed
   reproducing at LM Studio's own serving layer, application code bypassed
   (`docs/plans/salesperson-ui-ml.md`), not yet mitigated") independently confirms that document,
   not a fresh project-docs write, is already the designated home. Promoting the raw entry
   separately would indeed have produced a thinner, less actionable duplicate.
3. **The `history.md` entry itself — correctly placed, accurate, and honest about the reasoning.**
   The new 2026-09-17 section sits immediately above the existing 2026-09-16 (`tdd-engineer`
   mid-pass) entry with no duplication of it. It records the discard as a discard (not silently
   dropped), states the entry's ID, verification method, the specific duplication finding, the
   "no `MENTIONS` needed" call and why, and the graph-clear arithmetic — matching every other
   unit's history-entry convention in this same file (compare the 2026-09-13, 2026-09-10 U56, and
   2026-09-09 U31 sections, all of which record a discard with equivalent rigor).
4. **`data-scientist`'s graph state — independently re-confirmed clean.** Re-ran both reads
   against `kaizen_team`: `MATCH (a:Agent {agentId:'data-scientist'})-[:PRODUCED]->(k:KaizenEntry)
   RETURN count(k)` → `0`; legacy `MATCH (k:KaizenEntry {author:'data-scientist'}) RETURN count(k)`
   → `0`. Also confirmed the specific node itself is gone (`MATCH (k:KaizenEntry
   {entryId:'a1e6b3f4-2c7d-4e3a-9b5f-6d8c1a2f9e01'}) RETURN k` → no rows), i.e. a real
   `DETACH DELETE`, not merely an edge removal.
5. **Whole-coordination sweep (see Open questions below for the one thing worth flagging).**
   `MATCH (a:Agent)-[:PRODUCED]->(k:KaizenEntry) RETURN a.agentId, count(k)` currently returns:
   `analyst`=8, `architect`=1, `cobb`=1, `tico`=1 — and, notably, `frontend-engineer`,
   `tdd-engineer`, `qa-engineer`, `coder` and `data-scientist` are **absent from the result
   entirely** (i.e. genuinely zero). `cobb` and `tico` were never in this coordination's
   7-agent target list (the snapshot table in `docs/plans/kaizen-team-distillation-coordination.md`
   names only `analyst`, `frontend-engineer`, `architect`, `tdd-engineer`, `qa-engineer`, `coder`,
   `data-scientist`), so their counts are out of scope and expected. `analyst` and `architect` are
   in the target list, and both currently show entries again after being swept to 0 by U1
   (`analyst`, committed `1896a1f`, 2026-09-16 20:32:35 -03) and U3 (`architect`, committed
   `b6e1454`, 2026-09-16 21:51:27 -03) respectively. This is **not evidence of a missed sweep**:
   every `analyst` entry is dated 2026-09-16 or 2026-09-17 and every `architect` entry is dated
   2026-09-17 — i.e. all postdate each unit's own clearing commit, consistent with the standing
   capture loop producing fresh entries from the six subsequent gate reviews `analyst` itself ran
   for U2-U7 (and this very U7 review) plus whatever `architect` work happened after U3. The
   coordination's snapshot was explicitly "at open (2026-09-16)" — it targeted that fixed backlog,
   not a standing "stays at zero forever" guarantee, and the capture loop is designed to keep
   filling as agents keep working.

## What's solid

- Both citations in the new history entry (`proof_defs.py:363-367`, `executor.py:1277`) are exact
  — no drift between what the entry claims and what the code currently does.
- The duplication call is not a rubber-stamp: `docs/plans/salesperson-ui-ml.md` genuinely subsumes
  the raw entry with more precision and an actionable, prioritized fix, and the K-065 note
  independently corroborates that this document — not a new project-docs write — is the intended
  home.
- The entry correctly declines a `MENTIONS` tag with a stated reason (the finding is squarely
  `data-scientist`'s own domain), rather than defaulting to tagging out of caution.
- This is the seventh and final unit of a 7-unit coordination, and every one of the seven target
  agents' original snapshot backlog is now independently confirmed cleared to zero.

## Open questions

None that block acceptance. The `analyst`/`architect` re-population noted in finding 5 is worth
`teco` being aware of only as context for closing out the coordination doc (e.g. when writing the
milestone-close summary, it's accurate to say "the snapshot backlog was fully cleared," not "the
graph is now and will stay empty for these agents") — no action is needed on it as part of this
gate.
