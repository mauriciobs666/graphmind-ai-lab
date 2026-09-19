# Review: K-030 follow-up U1 — team-wide write-convention cutover + Stage 9 orphan-doc fix

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (M8), `agent-knowledge-base-strategy5-coordination.md` U1

## Scope & verdict

Reviewed commit `209017c` (15 files) against the brief in
`claude/docs/plans/agent-knowledge-base-strategy5-coordination.md` U1: (1) the 3 agent files
`teco` hadn't fully read (`architect.md`, `data-scientist.md`, `security-expert.md`), checked for
pattern fidelity and for `security-expert.md` not picking up anything Stage 7 deliberately left
out; (2) the `skills/agent-maintenance/SKILL.md` orphan-doc fix, judged on logical correctness,
not just its citations; (3) a diff-scoped accuracy/scope-creep check on the prose and `cobb`'s own
`kaizen/history.md`/`plan.md` bookkeeping; (4) a CPG relevance check. Did not re-review the 8
agent files `teco` already read in full, or re-litigate Stages 6–9's own review trail.

**Verdict: approve with suggestions.** No blocker. Two Major findings on the SKILL.md fix's
logical soundness — both real, evidenced gaps, neither an active incident today, both cheap to
close with a follow-up SKILL.md edit rather than a revert of this commit.

**CPG:** considered, not relevant — this dispatch touches only agent-prompt/skill Markdown, no
application source; the only loaded graph relevant to this repo's code is `cpg_falkorchat`
(confirmed via `GRAPHS` — no `cpg_claude` exists), consistent with every other unit across this
whole K-030 effort.

## Findings

### Major — the orphan-doc sweep's `list_documents` call returns the *oldest* N documents, not the most recent, and has no way to ask for the newest

`skills/agent-maintenance/SKILL.md`'s new "Found none" branch relies on
`list_documents(current_only=True, limit=<comfortably above the current corpus size, e.g. 1000>)`
failing to find a title match as proof the orphan genuinely doesn't exist. But the underlying
query is hardcoded oldest-first with no ordering override:
`falkor-chat/server/falkorchat/repository.py:1353` ("oldest first") /`:1372`/`:1386`
(`ORDER BY d.createdAt` ascending, ascending in both the `current_only` and full branches), and
the MCP tool's own docstring confirms it (`falkor-chat/server/falkorchat/mcp.py:408`, "oldest
first") with no sort/order parameter exposed at all (`mcp.py:404-412`). The orphan document the
sweep exists to find was, by construction, just created — it's one of the *newest* documents in
the corpus. Once the live `ws:agent-team` corpus exceeds whatever `limit` a distiller picks, the
sweep silently returns the oldest N and never reaches the target, producing a false "found none"
— which takes the "case (a)" branch and calls `ingest_document` again, creating exactly the
second, orphaned, untracked document the fix exists to prevent. No error surfaces; the failure is
indistinguishable from a genuine "old doc gone" result.

Not an active incident today — I queried `ws:agent-team` directly and it holds **334**
`currentVersion:true` documents (`MATCH (d:Document {currentVersion: true}) RETURN count(d)`),
comfortably under the doc's suggested `limit=1000` — but the corpus grows monotonically (13
agents' raw captures plus every distilled claim land here with no pruning), and nothing in the
fixed text re-verifies the count before trusting a "found none." The margin will erode silently.

**Suggested fix (route to `cobb`, its own file):** SKILL.md's sweep step should not treat
`limit=1000` as a fire-and-forget constant. Add an explicit corpus-count check before the sweep
(`mcp__cypher__query(graph='ws:agent-team', cypher="MATCH (d:Document {currentVersion:true})
RETURN count(d)")` — a plain read, no MCP `list_documents` tool needed for the count) and require
`limit` to exceed that count before trusting a "found none" result; if it doesn't, the sweep is
inconclusive and the recovery step should stop/escalate rather than silently re-ingesting.

### Major — a false-positive title match on the sweep can lead to `delete_document`ing an unrelated claim's live document

The "Found one" branch (SKILL.md, new text) adopts a `list_documents` title match, then "fall[s]
through to the byte-exact check above." If the adopted document's text does *not* match the
claim's current `.md` text — which is exactly what happens on a title collision with a different
claim's document — the existing "non-`None` result that doesn't match" branch fires: "the
interruption landed before `delete_document` ever ran — the old document genuinely still exists —
so re-run the whole … sequence … its own `delete_document(documentId)` call will succeed as
written, since the id it's deleting is still live." In the collision case that id is a *different*
claim's live, correctly-tracked document — the fixed text would delete it, corrupting two claims'
tracking in one recovery pass (the colliding claim loses its real document; this claim's manifest
now points at a deleted id).

Currently very low-probability: I checked `claude/cobb/scripts/kb-claim-manifest.json` and, of 90
tracked claims, there are **0 duplicate titles** — titles are full-sentence claim headings (e.g.
"Verify a hook gate by what it pattern-matches, not by its stated intent"), not generic labels, so
an accidental match is unlikely in practice. But the fix's own logic doesn't rely on or state that
invariant, and the failure mode if it's ever wrong is destructive (a live document deleted), not
merely a missed match.

**Suggested fix (route to `cobb`):** the byte-mismatch branch currently means one thing
unconditionally ("stale pre-edit document, safe to delete"). After a sweep-adopted match
specifically, a byte mismatch means something different and should not fall into the same
`delete_document` branch — treat it as a sign the title match was wrong and stop for the
distiller to resolve by hand, rather than auto-deleting.

## What's solid

- All 3 previously-unchecked agent files (`architect.md`, `data-scientist.md`,
  `security-expert.md`) match the established `teco.md`/`cobb.md` pattern exactly: framing clause
  preserved, correct `produced_by=<own name>` literal, identical trailing note. `security-expert.md`
  correctly did **not** pick up the `agent-kb-retrieval` pointer (`grep` confirms no
  `search_documents`/`ws:agent-team`-searchable mention anywhere in the file besides the Learning
  capture section itself) — consistent with it never being a named Stage 7 consumer.
- The 5 `tools:` allowlist edits are exactly as specified: `analyst`/`architect`/`data-scientist`
  keep their existing `search_documents`/`get_document` pair and gain `ingest_document`;
  `security-expert`/`tico` gain only `ingest_document`. The other 6 agents' files have zero
  `tools:` line touched — confirmed by diffing all six, ruling out the failure mode where adding
  an unrelated explicit `tools:` line would have silently narrowed an inherit-everything agent.
- `delete_document` raising `DocumentNotFoundError` on an already-gone id is real
  (`falkor-chat/server/falkorchat/services.py:1418`, matching
  `falkor-chat/docs/plans/document-ingestion2.md:356/437/472/475`) — the fixed text correctly
  removes the implied second `delete_document` call.
- `claude/AGENTS.md`'s summary bullet and `SKILL.md` §5's stale-pilot-note rewrite both accurately
  describe what landed — no overclaiming; "team-wide since 2026-09-19" is dated correctly and
  matches the commit date.
- `claude/cobb/kaizen/history.md`'s new entry and `plan.md`'s K-030 row closure are accurate and
  proportionate: the K-052 tool-visibility finding is filed as a `proposed` parking-lot item, not
  silently applied to any prompt — correctly scoped as investigation, not a prompt change.
  `audit-team.sh` re-run independently (not just trusted): same 5 pre-existing FAIL lines
  (username/home-path leaks in files this unit never touched), no new category, check 11
  (`agent-kb-retrieval` prefix template) passes.
- No scope creep: exactly the 15 files the brief named, nothing extra landed in a prompt or skill
  file. The K-052 investigation was explicitly requested by the dispatching brief's "close the
  loop" instruction (per `cobb`'s own history entry) and resulted only in a filed kaizen item, not
  an unrequested change.
- On `teco`'s own K-052 refinement (nested-subagent-of-subagent vs. top-level probe): I have no
  independent evidence to add here — it's a single data point either way and correctly filed as
  `proposed`, not promoted. No objection to how it's currently recorded.

## Open questions

None blocking. Both Major findings are independent of each other and of the rest of the dispatch
— they can be fixed in a small, disjoint follow-up to `skills/agent-maintenance/SKILL.md` without
touching any of the 11 agent files or `claude/AGENTS.md`.

## Pass 4 (2026-09-19) — re-check of commit `b7e39a1`

**Verdict: needs changes.** The stale-"found-none" fix is clean. The collision-unsafe-delete fix
closes the same-pass deletion but leaves a write ordering gap that reopens the identical hazard
one distillation pass later — a new finding, not a re-argument of the original one.

**Finding 1 (stale "found none") — fixed.** `skills/agent-maintenance/SKILL.md:837-847` inserts
the corpus-count guard before either "Found" branch is reachable, and the escalate path
(`limit`-exceeded) performs no manifest write at all — confirmed by reading lines 837-890 in full:
nothing between "stop here and escalate" and the next bullet touches `documentId`/`verified`, so a
retried pass safely re-enters the same check from the same `"pending"` state. Clean.

**Finding 2 (collision-unsafe delete) — not fully fixed; new gap.** The "Found one" branch
(`SKILL.md:865-867`) still writes `documentId := <found id>` into the manifest *before* the
byte-exact check runs and *unconditionally* — the fix (lines 877-880) only gates the
`delete_document` call on that check's outcome, not this earlier write. On a genuine title
collision the manifest is therefore left pointing at a different claim's live document, with
`verified: "pending"`, even after escalating. The **next** distillation pass that revisits this
`"pending"` entry hits the upstream "pending-at-start-of-pass" check
(`SKILL.md:803-821`, byte-identical before and after this fix, untouched by it): `get_document
(documentId)` now returns non-`None` (the adopted id is real and live) and mismatches this claim's
own `.md` text (it's someone else's content) — and that check treats *any* non-`None` mismatch
unconditionally as "the old document genuinely still exists … re-run the sequence … its own
`delete_document(documentId)` call will succeed as written, since the id it's deleting is still
live." That deletes the other claim's live document — the exact hazard Finding 2 was meant to
close, reintroduced one pass later instead of prevented. Evidence: read `SKILL.md:799-890` in full
(both the unchanged 803-821 block and the new 837-890 block) and traced the two-pass sequence by
hand against both branches — not just the single pass the fix's own text addresses.

**Suggested fix (route to `cobb`, same file):** gate the manifest overwrite on the same
confirmation that gates the delete — write `documentId := found_id` only *after*
`get_document(found_id)` confirms a byte-exact match; on mismatch, leave the manifest entry
exactly as it was at the top of this recovery attempt (still the pre-recovery id, already
confirmed `None`) so a future pass re-enters the *ambiguous-`None`* branch — which now safely
re-runs the corpus-count-gated sweep — rather than the upstream unconditional-delete branch. This
is a two-pass defect class (a write that outruns its own verification, surfaced only on a *later*
pass through a *different*, unrelated branch): before accepting the next revision, trace what a
**second, later pass** does with whatever partial manifest state each escalate/abort branch
leaves behind, for every branch in this bullet that writes to the manifest ahead of its own
verification completing — not only what the same pass does with it.
