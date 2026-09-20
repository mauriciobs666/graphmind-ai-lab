# graphmind-ai-lab

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M6+)

# Review: `falkor-chat/docs/manuals/embedding-migration.md`

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M6+)

## Scope & verdict

Static factual/architectural review of `falkor-chat/docs/manuals/embedding-migration.md` against
its real sources: `falkor-chat/scripts/embedding_migration.py`, `scripts/create_workspace.sh`,
`scripts/pin_workspace_embedding_model.sh`, `scripts/migrate_embeddings.sh`,
`server/falkorchat/repository.py` (`write_model_overrides`/`read_model_overrides`/
`read_index_dimension`), `docs/DESIGN.md` §1.3, `scripts/bootstrap_schema.sh`, and
`docs/plans/embedding-migration-coordination.md`'s ledger — plus one live query against the
running FalkorDB instance to check a claim about current graph state. The behavioral/walkthrough
half (does `pin`/`migrate` actually work end-to-end) is out of scope — that's `qa-engineer`'s.

**Verdict: needs changes** (one blocker).

**CPG:** not applicable — this is a documentation-accuracy review with no code change under
review; the manual's factual claims were checked directly against script/repository source rather
than via `cpg_falkorchat` (a manual-vs-source cross-check is line/property matching, not a
call-graph or data-flow question the CPG would add value on).

## Findings

### BLOCKER — Walkthrough 1's "every existing workspace was already pinned" claim is false, verified live

**Manual text** (lines 90-92): *"Every **existing** workspace at the time this capability shipped
was pinned once, up front, to whatever model it was already effectively using — so no workspace
was ever silently left unpinned by the rollout itself."*

This is stated as a completed, past-tense fact. It is not true:

- `docs/plans/embedding-migration.md:588-592` (§5 step 3, "FR-1's one-time sweep") describes this
  as a **not-yet-run runbook step**: *"run `pin_workspace_embedding_model.sh` against every
  existing real workspace ... once, before the global default ... is ever edited."*
- The same plan's open questions (`:783-785`) leave even the *scope* of that sweep unresolved:
  *"Should `ws:demo`/`ws:agent-team` receive an FR-1 pin alongside `ws:acme` and `ws:eval` ... or
  is either out of scope ... ? Not resolved here."*
- Live-verified against the running instance (`mcp__cypher__query`, `MATCH (c:WorkspaceConfig)
  RETURN c`): `ws:acme`, `ws:demo`, `ws:eval`, and `ws:agent-team` **each return zero rows** — none
  of them has a `WorkspaceConfig` node at all, so none carries an `embeddingModelOverride`. The
  only graphs where a `WorkspaceConfig` node with `embeddingModelOverride` exists are throwaway
  `qa-engineer` probe graphs (`ws:qa-pin-test-*`), confirming `pin()` has been exercised only in
  testing, never against a real workspace.
- `scripts/seed_eval_corpus.py:605-608` only calls `embedding_migration.pin(EVAL_WS, ...)` inside
  the conditional *rebootstrap* branch (triggered by a dimension mismatch) — not unconditionally —
  so `ws:eval` isn't guaranteed a pin either, consistent with the zero-row result above.

**Why it matters:** this is the manual's central safety claim for the shipped half of the
feature — "no workspace was ever silently left unpinned." An operator reading it would reasonably
conclude every real workspace is already protected against a `config/models.json` default change,
when in fact **all of them are currently exposed**: editing the global default today would
silently change embedding behavior for `acme`/`demo`/`eval`/`agent-team` on their next write, with
no override to stop it.

**Suggested fix:** rewrite the bullet to state the sweep's actual status — an open, unexecuted
runbook step (cite `docs/plans/embedding-migration.md` §5 step 3 and its open question) — and
either drop the "no workspace was ever left unpinned" framing entirely or replace it with an
operator action item ("run `pin_workspace_embedding_model.sh acme demo eval agent-team` before
touching the global default"). Re-verify against live graph state (as done here) before restoring
any past-tense claim of completion.

### MAJOR — "a fully-migrated workspace re-run does nothing" understates real side effects

**Manual text** (Walkthrough 2, line 136-138): *"It resumes, it never restarts from zero. ...
A fully-migrated workspace re-run does nothing and reports zero rows processed."*

Read against `scripts/embedding_migration.py:336-350` (`migrate()` steps e/f), a re-run against an
already-fully-migrated workspace does re-embed **zero rows**, but it still unconditionally:

- calls `_rebuild_vector_index()` for both `Message` and `Chunk` — which, per its own guard
  (`:207-208`), still issues `DROP VECTOR INDEX` (since one exists) followed by `CREATE VECTOR
  INDEX`, i.e. a real index drop-and-rebuild with no data change;
- calls `write_model_overrides()` again (step f), rewriting `modelOverrideUpdatedAt`/`By` even
  though `embeddingModelOverride`'s value doesn't change;
- prints the mandatory "RESTART the server process now" line again (step g).

`server/tests/test_embedding_migration.py::test_migrate_rerun_after_success_is_idempotent` (lines
679-696) only asserts `embedder2.calls == 0` and the migrated/total counts — it never asserts the
index isn't dropped/recreated on the no-op re-run, so the test suite doesn't contradict this
reading, it just doesn't cover it.

**Why it matters:** "does nothing" invites an operator to treat a verification re-run as
harmless/side-effect-free, when it in fact drops and rebuilds both vector indexes (a brief window
where that label's ANN index doesn't exist) and re-demands the mandatory restart. Framed as "does
nothing," an operator might skip the restart on a re-run "since nothing happened" — which is
exactly the failure mode the rest of the walkthrough is careful to warn against elsewhere.

**Suggested fix:** rephrase to something like: "A fully-migrated workspace re-run re-embeds zero
rows, but it still rebuilds both vector indexes (drop+recreate, briefly leaving that label
unsearchable) and still requires the same post-run restart — it is idempotent in effect, not a
no-op in execution."

### MINOR — "shipped, reviewed, and in production use" overstates `pin`'s current real-world footprint

**Manual text** (line 32-33): *"**Shipped, reviewed, and in production use** (every workspace
created via the canonical entry point is pinned automatically at birth)."*

`pin` is genuinely shipped and reviewed (U6, `docs/plans/embedding-migration-coordination.md`,
`analyst`-approved). But "in production use" is optimistic given the BLOCKER above: no real
workspace has actually gone through `create_workspace.sh` yet (all of today's real workspaces
predate the feature), and the parenthetical's mechanism is the only thing actually exercised live.

**Suggested fix:** either drop "in production use" or qualify it precisely: "wired into every
real-workspace creation path (`start_server.sh`, `start_demo.sh`, `start_agent_team.sh`); not yet
run retroactively against any pre-existing workspace — see the FAQ/callout above."

## What's solid

- The `migrate()` step-by-step Mermaid flowchart (Walkthrough 2) matches the code's actual order
  exactly: traffic-stop precondition before any repo/gateway construction (`:285-291`), dimension
  check before re-embedding (`:300-310`), Message-then-Chunk re-embed to completion (`:312-318`,
  `_MIGRATION_LABELS` ordering at `:48`), count-check-before-rebuild abort semantics (`:320-334`),
  guarded rebuild (`:336-338`), override move (`:340-350`), restart print (`:352-357`).
- The "Graph data structures" section's property names are exactly right: `WorkspaceConfig`'s four
  override properties (`repository.py:3315-3330`, `:3347-3353`) and the read-before-write
  discipline that protects the other three kinds; `Message`/`Chunk`'s `embedding`/`embeddingModel`
  pair (`embedding_migration.py:184-188`); `pin` never touching either property (confirmed — `pin`
  only calls `write_model_overrides`, never `_graph(ws).query` against `Message`/`Chunk`).
  `EmbeddingDimensionError` (`repository.py:23`) is real and matches its described role.
  Resumability via `coalesce(embeddingModel, '') <> target_ref` (`:168`) matches the manual's
  description exactly, including the skip-and-log (not fail) treatment of a vanished row
  (`:236-242`) and the count-check abort wording.
- `Configuration & integration`'s `EMBEDDING_DIM` row (env var, `bootstrap_schema.sh`, default
  `1536`) is correct against `scripts/bootstrap_schema.sh:399` (`local dim="${EMBEDDING_DIM:-1536}"`)
  and `docs/DESIGN.md` §1.3 — no repeat of the placement bug the brief flagged from a prior manual.
  `--batch-size` (default 50) and `--i-have-stopped-traffic` match
  `embedding_migration.py:398-407` exactly; `FALKORCHAT_OPENCODE_CONFIG`'s pin-degrades/
  migrate-is-fatal split matches `pin_workspace_embedding_model.sh` vs. `migrate_embeddings.sh`'s
  preflight checks verbatim.
- The manual's framing of `migrate`'s review-gate status (code-complete, extensively tested, not
  yet signed off) matches `docs/plans/embedding-migration-coordination.md`'s U7/U8 rows exactly
  ("delivered" / gate "in-flight," joint with `analyst`) — neither overstated nor understated.
- CLI usage examples for all three scripts (`create_workspace.sh`, `pin_workspace_embedding_model.sh`,
  `migrate_embeddings.sh`) match their actual argument parsing and flags.

## Open questions

- The BLOCKER above is really two things tangled together: a stale/aspirational manual claim, and
  an actually-unprotected set of real workspaces. Fixing the manual's wording closes the doc
  defect; whether `tico`/`teco` also wants to actually run the sweep (or file it as a tracked
  backlog item) is a product decision outside this review's remit — flagging it since the manual
  currently hides that it's still open.
- The Mermaid diagrams were checked for structural/semantic correctness (node relationships,
  property names, step order) against source, not rendered through a Mermaid engine — no rendering
  tool was available in this session. No syntax issues were spotted on inspection, but a
  render-through check (e.g. via `qa-engineer` viewing the built manual page) would be the way to
  fully close item 4 of the brief.
