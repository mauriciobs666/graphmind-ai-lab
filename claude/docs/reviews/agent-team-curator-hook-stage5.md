# Agent-team curator hook, Stage 5 — diff-scoped review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

## Scope & verdict

Reviewed the uncommitted working-tree diff of `skills/agent-maintenance/SKILL.md`,
`claude/AGENTS.md`, `claude/cobb/kaizen/history.md`, and `claude/cobb/kaizen/plan.md` — `cobb`'s
Stage 5 unit for K-030 Track 1 (`claude/docs/plans/agent-knowledge-base-strategy.md` §3 row 5,
§5's named hook point), extending `agent-maintenance` SKILL.md §5's curator review/clear steps
with a third read source (`ws:agent-team` via `list_documents`/`get_document`) and a third
disposal shape (`delete_document`), alongside the unchanged `kaizen_team` shapes. Baseline: the
parent plan §3/§5, the Stage 4→5 coordination ledger
(`claude/docs/plans/agent-knowledge-base-strategy3-coordination.md`), the full (not just diffed)
text of SKILL.md §5, and the actual falkor-chat MCP server / `cypher-mcp` authorizer source. Not
in scope: whether Stage 4's pilot should widen team-wide, or `kaizen_team`'s retirement timing —
both are named, explicitly open in the diff itself.

**Verdict: approve with suggestions.** Every factual/API claim I checked against source held up
exactly (see Findings below); the diff stays inside its named scope (no §5 step-renumbering, the
`MENTIONS`-equivalent gap is correctly left open rather than quietly resolved); the disposal-shape
framing is defensible on the actual `Document` model. One minor drift and one nit, both cosmetic.

**CPG: not applicable — this is a pure documentation change (a skill's procedure text, a context
file, two kaizen logs); no application source is touched, and no CPG is loaded for `claude/` or
`skills/` in the first place (`GRAPHS` on this instance: `cpg_falkorchat`, `kaizen_team`,
`reference`, four `ws:*` graphs — none for this component).**

## Findings

### Minor — `plan.md`'s trailing "Notes:" line still says "Stages 1-4," contradicting the same entry's own Status line and Track 1 bullet three lines away

`claude/cobb/kaizen/plan.md:394` (Status) and `:435` (Track 1 bullet) were both correctly bumped
by this diff to "Track 1 Stages 1-5" / "all five stages." The K-030 entry's own `**Notes:**` line,
four lines later, was not touched and still reads (`plan.md:455`): "Track 1 Stages 1-4 delivered
and logged here 2026-09-18 (detail in `history.md`)." Confirmed by grep — `Stages 1-4` appears
only at line 455, `Stages 1-5`/`all five stages` at 27/394/435. This is exactly the failure mode
root `AGENTS.md`'s "an open item is rewritten... reads as one present-tense statement" rule
targets: a reader who lands on the Notes line alone would believe Stage 5 hadn't shipped, four
lines below a bullet saying it had. Low stakes (the prominent Status/table-row text is right), but
real — a future revisit of this same K-030 entry inherits the contradiction if not caught here.
**Fix:** bump `plan.md:455`'s "Stages 1-4" to "Stages 1-5" in the same pass that lands this diff.

### Nit — `plan.md`'s quoted `"today: cobb/teco only"` isn't a verbatim quote from the SKILL.md text it cites

`plan.md:445-446` presents `"today: `cobb`/`teco` only"` in quotation marks as if citing SKILL.md
literally. The actual SKILL.md text (`skills/agent-maintenance/SKILL.md:405-409`) says "That's the
case today: ... only `cobb`/`teco` currently write to `ws:agent-team`" — accurate in substance,
but the quoted string doesn't appear verbatim anywhere in the SKILL.md diff. Not misleading about
the fact itself (confirmed true — Stage 4 was piloted with `cobb`/`teco` only, per the
coordination ledger's U10 row), just a sloppy citation. **Fix:** drop the quote marks, or
paraphrase without them.

## What's solid

- **Every checked API/grounding claim held exactly.** Verified directly against
  `falkor-chat/server/falkorchat/mcp.py`/`services.py`/`repository.py` and `cypher-mcp/server.py`:
  `list_documents(current_only: bool = True, limit: int = 50)` signature match; `get_document`'s
  docstring and thin-passthrough implementation confirm no cell-level truncation (unlike
  `cypher-mcp`'s `MAX_CELL`, confirmed `=300` at `cypher-mcp/server.py:100`); `delete_document` is
  a real hard delete (`document-ingestion2` FR-4, confirmed verbatim in
  `falkor-chat/docs/requirements/document-ingestion2.md:76`) with **no** curator/role check
  anywhere in `mcp.py`/`services.py` — a plain, unauthenticated-beyond-`ctx` MCP call, exactly as
  claimed; `ingestedById`/`ingestedByKind` field names match `repository.py:1370-1371` exactly;
  `produced_by` resolving only against a real `Agent` node and raising `AgentNotFoundError`
  (never a silent actor fallback) is confirmed at `services.py:1225`.
- **The "four curator-gated shapes" count is correct in the landed text.** `cypher-mcp/server.py`
  defines exactly four curator-gated regexes (`_CURATOR_CLEAR_RE`, `_MENTIONS_WRITE_RE`,
  `_PRODUCER_EDGE_RESOLVE_RE`, `_MENTION_EDGE_RESOLVE_RE`); the diff's closing paragraph
  (`SKILL.md:710-712`) names exactly four items — legacy clear, `PRODUCED`-resolve,
  `MENTIONS`-resolve, full-node clear — matching the four documented clearing contexts in §5 step
  4/5 (even though legacy-clear and current-shape full-node-clear share one underlying regex,
  `_CURATOR_CLEAR_RE`, the text names them as the two distinct *contexts* it actually describes
  just above, not as five or three).
- **Scope discipline held.** §5's step numbering (1-4) and step-4's internal sub-numbering (1-5)
  are unchanged; the new material is inserted as sub-bullets/sub-paragraphs within the existing
  structure. The `MENTIONS`-equivalent tagging gap is correctly left open, citing the parent plan's
  own §5 Track 2 Stage 9 bullet (`agent-knowledge-base-strategy.md:527-531`), which does say
  "flagged as a genuinely open item, not resolved" — an accurate, non-deciding citation.
- **The disposal-shape framing is defensible, not just asserted.** `services.delete_document`
  (`repository.py`/`services.py`) confirms falkor-chat's `Document` model carries no
  producer/mentions-edge analogue for `cobb` to resolve before deleting — so "one unconditional
  call, closer to the legacy shape than the current-shape read-then-decide one" is an accurate
  characterization of the actual code, not a convenient assertion.
- **History/plan hygiene otherwise clean.** New dated `history.md` entry, not a stacked `Update:`
  clause on Stage 4's; the K-030 backlog-table row (`plan.md:27`) was correctly compacted to point
  at the narrative entry rather than duplicating it; no duplicate `##`/`###` headings introduced in
  either file (`grep '^## ' | sort | uniq -d` clean on both).
- **`claude/AGENTS.md`'s rewritten Distillation bullet matches what SKILL.md actually implements** —
  "reads/clears both sources in parallel" is a fair one-line summary of SKILL.md's "needed
  alongside these two for as long as any agent's raw capture is still landing there," and the
  rewrite correctly drops the old bullet's now-inaccurate "clears it with a curator-scoped `DETACH
  DELETE`" as the sole clearing mechanism.

## Open questions

None — the diff's own named follow-ups (team-wide cutover, `kaizen_team` retirement timing, the
`MENTIONS`-equivalent gap) are already correctly flagged as open in the artifact itself and are
explicitly out of this review's scope per the brief.
