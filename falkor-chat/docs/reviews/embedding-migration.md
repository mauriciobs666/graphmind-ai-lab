# Embedding model migration & index rebuild — Plan Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M6+)

## Scope & verdict

Reviewed: `falkor-chat/docs/plans/embedding-migration.md` (orchestration-level implementation
plan), against `falkor-chat/docs/requirements/embedding-migration.md` (FR-1..FR-10, acceptance
criteria, decision log, Status: Ready for design). Verified by reading the live source directly —
`server/falkorchat/modelconfig.py`, `repository.py` (§3231-3316 and the `set_embedding`/
`set_chunk_embedding`/`read_index_dimension` trio), `embedding.py`, `server/tests/conftest.py`,
`model-bench/packs/embedder-graphrag-retrieval/pack.json`, `config/models.json`,
`config.py` — not the plan's paraphrase of them. The companion Cypher/DDL note
(`embedding-migration-graph.md`) was intentionally not reviewed, per the brief.

**Verdict: needs changes.** Two blockers below; the rest of the plan (its reading of the
`ModelGateway`/`Repository` seams, the hard-cap-bypass analysis, the test strategy) is
well-grounded and, findings aside, implementable as designed.

**CPG:** considered, not relevant — `cpg_falkorchat` is available and the plan itself already
disclaims using it for the same reason this review reaches: every claim here was cheaper to settle
by reading the cited source directly (a handful of specific files/line ranges) than by a
call-graph traversal over a component this size.

## Findings

### Blocker 1 — FR-2's "documentation-only" design conflicts with an already-confirmed decision-log line, and the plan doesn't gate on resolving it

`docs/requirements/embedding-migration.md`'s decision log states, dated and marked "Confirmed
exactly as read back": *"Rule (2) is to be built into workspace bootstrap itself (**not left as a
manual convention**) — captured as FR-1/FR-2."* The plan's §3.2 design for FR-2 is precisely a
manual convention: a second script (`pin_workspace_embedding_model.sh`) an operator must
separately remember to run after `bootstrap_schema.sh`, enforced only by an `AGENTS.md` sentence.
The plan's own words concede this ("a documentation-enforced convention, not a technical
guarantee... an operator can still run `bootstrap_schema.sh` alone and skip the pin step") but
frames the conflict as an open wording question for the stakeholder (§7) rather than a design that
contradicts a specific, already-confirmed requirement line — and, critically, does **not** gate
§5 step 2 (documenting the recipe in `AGENTS.md`, effectively declaring FR-2 satisfied) on getting
that confirmation first. If implementation proceeds as sequenced, FR-2 ships as the very "manual
convention" the stakeholder explicitly rejected, without ever surfacing that specific tension as a
decision to be made rather than a recommendation to be read past.

This doesn't block everything: the `pin` subcommand and the FR-1 one-time sweep (§5 steps 1, 3)
are needed regardless of which enforcement mechanism wins, so that code can be built now. What
must not proceed un-gated is treating the `AGENTS.md`-documented recipe as FR-2's closing answer.
**Suggested fix:** split §5 step 2 out from steps 1/3, and make it explicitly conditional on a
stakeholder response to §7's open question — or have the plan present the rejected
`create_workspace.sh`-replaces-`bootstrap_schema.sh` alternative with enough concrete scope (which
call sites in `start_server.sh`/every `seed_*.sh` would need updating) that the stakeholder is
choosing between two designed options, not rubber-stamping the one already written.

### Blocker 2 — No design or runbook step actually creates FR-7's "outage"; the plan only handles restarting *after*

The plan is explicit that `EmbeddingWorker._index_dim_cache` requires a **post**-migration server
restart (§2.6, §3.4 step 5, test 14) and calls this out prominently. But nothing in §3.4's sequence
or §5's implementation steps instructs an operator to stop the server (or otherwise ensure no
traffic reaches the target workspace) **before** step 1 begins. Confirmed no such mechanism exists
to pause traffic to a single workspace without a broader change (`server/falkorchat/services.py`
has no maintenance/read-only-mode switch, and one server process typically serves every
`ws:{id}` per `falkor-chat/AGENTS.md`'s topology) — so in practice, this "outage" is either the
whole server going down (unaddressed operational scope: which workspaces else share that process)
or nothing is stopped at all, and the target workspace keeps serving live writes through the
`EmbeddingWorker` hot path throughout re-embed + index rebuild.

That matters because a live write landing in the specific window **between §3.4 step 2's count
check (0 unmigrated) and step 3's index drop+recreate** is not merely unmigrated — it produces
exactly the silent, ANN-invisible vector DESIGN §7.1 documents and this whole feature exists to
prevent: the ordinary write path embeds at the *old* model/dimension (workspace override not yet
updated until step 4) into an index that, by the time that write lands, has already been rebuilt
at the *new* dimension. No error anywhere in that chain (§2.1's documented behavior); the row is
also unmarked (`embeddingModel` absent, since the hot path never sets it), so nothing schedules its
repair unless someone happens to re-run `migrate` later. A write landing *before* the count check
is at least caught (aborts before touching the index, per §5 step c) — only this specific gap is
unguarded.

For the one migration this plan actually schedules (`ws:eval`, explicitly "not a live-chat-served
workspace," §3.5 step 4), the practical risk is near zero. But FR-9 requires this to be a general,
reusable capability for *any* workspace, and the plan does not confine this gap to a known,
accepted risk for that broader case — it isn't named as a risk at all in §7, unlike the (arguably
lower-probability, definitely better-mitigated) hard-cap bypass. **Suggested fix:** add an explicit
runbook precondition to §3.4 (a "step 0," mirroring how step 5's restart is called out explicitly)
naming what "ensure no traffic reaches this workspace" concretely means for this codebase's
single-process topology, and revisit whether §7's risk ranking still holds once this is priced in.

### Major — the plan's own risk ranking should be revisited given Blocker 2

§7 calls the hard-cap bypass "the single highest-severity correctness risk" in the plan. It's a
strong claim for that specific bug (named, tested — test 6 — and mitigated by an explicit design
decision), but Blocker 2's race is unflagged, untested, and reproduces the identical
silent-corruption failure mode for the plan's own stated general-reuse case (FR-9). The two aren't
mutually exclusive to fix, but the plan should not present the hard-cap bypass as the *sole*
highest-severity item when a comparably severe, currently *unmitigated* risk exists alongside it.

### Minor — the migration write path's deliberate bypass of the FR-19 guard is never stated as intentional

§4 item 2 designs a raw `SET n.embedding = ..., n.embeddingModel = ...` write, distinct from
`Repository.set_embedding`/`set_chunk_embedding`, and never routes through
`EmbeddingWorker._resolve_and_embed`'s FR-19 pre-flight guard (confirmed at `embedding.py:147-214`:
it compares the *resolved model's* dim against the *live index's* dim and raises before any HTTP
call on mismatch). This is not an oversight — it has to be this way, since by construction the old
index is still at the old dimension while the target model is at the new one, so the guard would
reject every single re-embed row. But the plan never says this explicitly; an implementer who
knows the guard exists (§2.1 goes out of its way to describe it) may reasonably wonder why the
migration path doesn't call it. **Suggested fix:** one sentence in §3.3 or §4 item 2 stating the
guard is deliberately not on this path and why.

### Minor — undefined behavior when the write-query detects a vanished row mid-migration

§4 item 2 asks for a way to "detect a no-op (id not found)... for FR-10's caller to notice a row
that vanished mid-migration," but neither §4 nor §5 step 5b says what the `migrate` loop actually
does with that signal (skip and continue, log and continue, abort the batch, retry). Low
probability given messages/chunks are effectively append-only, but the interface asks for the
signal without saying what to do with it. **Suggested fix:** one line in §5 step 5b's bullet
committing to "skip and log," the simplest option consistent with FR-10's idempotent-resume
posture.

### Minor — test list has no case for the "self-healing" unembedded-row path

§3.3 calls out, as a deliberate side effect, that a row with `embedding IS NULL` gets swept up and
embedded by the same `coalesce(n.embeddingModel, '') <> $targetRef` clause. §6's 14-case list has
no test exercising this specific path (all of tests 5/6/7 assume every row starts "all embedded").
**Suggested fix:** extend test 5 (or add a dedicated case) with a row that has no `embedding` at
all in the pre-migration fixture, asserting it ends up embedded and marked like every other row —
this is the one behavior in the plan documented as intentional but currently unverified by name.

## What's solid

- **Grounding is strong and specific.** Every cited file/line range I checked
  (`modelconfig.py:708-727`, `:729-755`, `:102-107`; `repository.py:773`, `:1202`, `:3237`,
  `:3287`; `embedding.py:120-214`; `conftest.py::rebuild_vector_indexes`;
  `model-bench/packs/embedder-graphrag-retrieval/pack.json`) matched the plan's description of it
  exactly, including the subtle kind↔property crosswalk (`agent`↔`responderModel`,
  `step`↔`agentModel`, confirmed swapped; `embedding`↔`embeddingModel`, confirmed 1:1).
- **The hard-cap-bypass analysis (§3.3) is correct and well-tested.** Traced independently:
  `gateway.embedder("embedding", requested=target_ref)` with no `ws=`/`overrides=` does skip
  `_workspace_override_ref` entirely, so `requested` wins outright — exactly as claimed, and test 6
  targets it directly.
- **The read-before-write discipline for `write_model_overrides`** is correctly named as a
  landmine and correctly followed at both call sites (§5 steps 1 and 5e) — the `agentModel`/
  `guardModel`/`responderModel` values are read back and passed through unchanged in both.
- **The idempotent-resume marker design** (a migration-owned `embeddingModel` property, never
  touched by the ordinary hot path) is a sound, low-blast-radius choice, and the rejected
  alternative (stamping it on every hot-path write) is argued convincingly.
- **The `graph-dba` handoff (§4)** is precise: six items, each stating exactly what's needed and
  what's still open, with item 6 correctly flagged as the one requiring live verification rather
  than transcription.

## Open questions

- Blocker 1's resolution is a genuine stakeholder call (documentation-only vs. a harder technical
  guarantee) — not something this review can settle. Recommend routing it back through `tico`/the
  stakeholder before `AGENTS.md` is updated to declare FR-2 satisfied.
- Blocker 2's fix is mechanical (name the precondition explicitly) but the multi-tenant question
  it surfaces — what "stop traffic to one workspace" means when one process serves several — may
  be worth a short, separate design note if a production migration is ever scheduled; not
  necessary to resolve before `ws:eval`'s own migration, given that workspace's already-established
  no-live-traffic status.
