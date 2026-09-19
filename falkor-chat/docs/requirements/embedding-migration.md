# Embedding model migration & index rebuild — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — (M6+) · **Last updated:** 2026-09-19

## Intent
The current embedding model, running in LM Studio, is consuming more host RAM/VRAM than the box
can sustain — this is a **resource-pressure problem with the model-serving process itself**, not
(as far as diagnosed so far) FalkorDB's per-vector storage RAM. The stakeholder needs to move to a
genuinely smaller embedding model, urgently, and needs the existing data migrated to match —
without silently corrupting retrieval in the process.

## Problem & current state
Today, changing which embedding model `falkor-chat` uses is a one-line config edit
(`config/models.json`'s `defaults.embedding`) plus a restart — see
`falkor-chat/docs/DESIGN.md` §1.3/§12. But nothing re-computes the vectors already stored on
existing `Message.embedding`/`Chunk.embedding` properties, and the vector index's dimension is
fixed at workspace-bootstrap time via DDL, independent of the model config. A model swap to a
different output dimension leaves old vectors silently unsearchable (wrong-dim writes are
accepted but drop out of ANN, per DESIGN §7.1); even a same-dimension swap leaves old and new
vectors from different semantic spaces mixed in the same index. There is no existing script or
mechanism to re-embed stored data or to rebuild the vector index at a new dimension — confirmed
absent by inspection (only test-only helpers exist, e.g. `server/tests/conftest.py`'s
`rebuild_vector_indexes`, and unrelated one-off backfill scripts like
`scripts/backfill_thread_ids.sh` follow the same pattern but don't touch embeddings).

The trigger is a **specific, urgent swap**: the current model (Qwen3-Embedding-0.6B, DESIGN §1.3)
is overrunning the host's LM Studio memory budget and must be replaced with a smaller model soon.

## User stories
- As the operator running `falkor-chat`, I want to change the global default embedding model
  without silently changing behavior for workspaces that already exist, so that a memory-driven
  model swap can't corrupt or block unrelated workspaces.
- As the operator, I want every new workspace to pin its own embedding-model choice the moment
  it's created, so that a future global-default change never silently shifts an existing
  workspace's behavior underneath it.
- As the operator, I want to validate a candidate model's retrieval quality against a safe,
  disposable dataset before touching real data, so I don't have to migrate production twice if
  the new model underperforms.
- As the operator, I want to re-embed a workspace's existing messages/chunks and rebuild its
  vector index to match a new model in one coordinated operation, so old and new vectors are
  never left mixed or mismatched in a way that silently breaks retrieval.
- As the operator, I want to confirm a migration actually worked, so I'm not left guessing
  whether retrieval quietly broke.

## Functional requirements
- **FR-1 (existing-workspace safety net):** Before the global default embedding model changes,
  every existing workspace must have an explicit per-workspace embedding-model override recorded,
  equal to whatever model it is currently and effectively using — so the global change cannot
  silently alter an existing workspace's behavior.
- **FR-2 (pin-at-birth rule):** Workspace creation/bootstrap must automatically write an explicit
  per-workspace embedding-model override equal to the global default in effect at that moment.
  Every workspace's model choice is fixed at birth; it never implicitly tracks later global-default
  changes.
- **FR-3 (re-embed):** The system must provide a way to recompute every existing `Message` and
  `Chunk` embedding in a target workspace using a newly chosen model.
- **FR-4 (index rebuild):** The system must provide a way to rebuild a target workspace's vector
  index at the new model's output dimension, coordinated with FR-3's re-embedding so there is no
  point where stored vectors and the index's expected dimension disagree in a way that silently
  loses data (builds on the existing `EmbeddingDimensionError` guard, SERVER.md §1.8 FR-19, which
  must keep functioning throughout).
- **FR-5 (override moves with the data):** Completing a workspace's migration must update that
  workspace's explicit embedding-model override to the new model, so its recorded "current actual"
  stays in lockstep with its data.
- **FR-6 (validate before production):** The migration capability must be exercised against
  `ws:eval` — checked against its existing golden-set + baseline recall/MRR mechanism — before
  being run against any workspace carrying real/production traffic.
- **FR-7 (outage tolerance):** A brief full outage/pause of retrieval and assistant answering is
  acceptable for the duration of a workspace's migration. The capability is not required to keep
  serving old vectors while new ones are being written, or to guarantee zero downtime.
- **FR-8 (verification):** After a migration completes, it must be possible to confirm success via
  (a) a count check that every intended `Message`/`Chunk` row now carries a new-model embedding,
  with none missed, and (b) a retrieval sanity check — real questions run through the assistant —
  confirming answers still make sense.
- **FR-9 (reusable, on-demand capability):** The migration mechanism (FR-3/FR-4) must be a
  general, repeatable capability parameterized by workspace and target model — invokable against
  any named workspace whenever needed — not a one-off script built only for `ws:eval` or for this
  one urgent swap. Having this readily available for the *next* model change is as important to
  the stakeholder as resolving the current one.
- **FR-10 (idempotent resume):** If a workspace's migration is interrupted partway through,
  re-running it must safely resume — only rows not yet migrated are (re-)processed, matching this
  project's existing backfill-script convention (`scripts/backfill_thread_ids.sh`,
  `scripts/backfill_document_current.sh`) — never a requirement to start over from zero.

## Out of scope
- Zero-downtime / dual-serving migration (old and new vectors both live and comparable
  mid-migration) — FR-7 accepts a brief outage instead.
- Choosing the destination model itself — that's `data-scientist`/`model-bench` territory
  (in progress alongside this interview, see decision log); this document scopes the migration
  *mechanism*, not the model decision.
- Retention/eviction of old embeddings (DESIGN.md §13's open retention question) — unrelated to
  this feature.

## Acceptance criteria
- Given an existing workspace with an implicit (unset) per-workspace embedding-model override,
  when FR-1's snapshot step runs, then that workspace has an explicit override recorded matching
  its pre-change effective model, and its embedding behavior is unchanged.
- Given the global default embedding model is then changed, when any existing (already-snapshotted)
  workspace embeds a new message, then it continues to use its own pinned model, not the new global
  default.
- Given a brand-new workspace is created after the global default has changed, when it embeds its
  first message, then it uses the global default in effect at its creation time, recorded as its
  own explicit override.
- Given `ws:eval` has been migrated to a candidate model, when its existing recall/MRR check is
  re-run, then the result is compared against the pre-migration baseline and reported (not silently
  assumed to pass) before any production workspace is touched.
- Given a target workspace's migration completes, when the verification step runs, then it reports
  a count of migrated vs. total `Message`/`Chunk` rows with zero unmigrated rows, and a retrieval
  sanity check has been performed and reviewed.
- Given a workspace has NOT yet been migrated, when a message is posted to it after the global
  default has changed, then the embedding-dimension guard (FR-19) raises loudly rather than
  writing a mismatched vector — the workspace fails safe, visibly, until migrated.

## Open questions
None blocking this document. Two items remain deliberately open, tracked in the decision log, and
neither gates design of the migration mechanism itself (FR-9 makes it work regardless of either):
which candidate model ultimately wins the `model-bench` validation run against `ws:eval`, and
which real/production workspace(s) get migrated once that validation lands — both by the
stakeholder's own choice to decide later rather than now.

## Decision log
- 2026-09-19 — What's driving this now? → Specific swap planned (not general readiness).
- 2026-09-19 — Where is the memory pressure showing up? → The model-serving process (LM Studio),
  not FalkorDB's per-vector RAM. Same box as documented in DESIGN.md (RTX 4050, 6GB VRAM budget
  shared with the 4B LLM, LM Studio on Windows/WSL2) — environment unchanged.
- 2026-09-19 — Replacement model picked yet? → No. Target footprint: "as small as reasonably
  possible," no harder ceiling given. Consulted `data-scientist` (in progress) for a shortlist of
  smaller candidate models (dimension, size, EN/PT-BR quality trade-off) and a recommendation on
  whether to route the choice through `model-bench`'s existing `embedder-graphrag-retrieval` pack
  before committing — outcome to be folded back in once it returns.
- 2026-09-19 — `data-scientist` consult returned. Shortlist (all smaller than current
  Qwen3-Embedding-0.6B, GGUF/LM-Studio-compatible, EN+PT-BR capable): **granite-embedding-278m-
  multilingual** (IBM, 768-dim, ~2x smaller, explicit PT-BR support, lmstudio-community-packaged
  — recommended first pick), multilingual-e5-base (768-dim, ~2x smaller), multilingual-e5-small
  (384-dim, ~5x smaller, larger expected quality drop — fallback if 2x isn't enough headroom).
  MRL dimension truncation (mentioned in DESIGN §1.3 as a future path) is a dead end for this
  problem — it shrinks the stored vector, not the serving process's resident memory. **Every
  candidate changes the vector dimension away from 1024**, so the migration capability must be
  built dimension-agnostic, not special-cased to one target size. Recommendation: validate the
  lead candidate through `model-bench`'s existing `embedder-graphrag-retrieval` pack (already has
  a comparable Qwen3-Embedding-0.6B-vs-BM25 baseline) before committing to a production migration
  — near-zero setup cost, and cheaper than migrating twice if quality regresses.
- 2026-09-19 — Validate before migrating? → Yes (recommended path adopted).
- 2026-09-19 — Stakeholder proposed an additional candidate, nomic-embed-text-v1.5 (GGUF).
  `data-scientist` checked it and ruled it **out**: it is English-primary/English-only (confirmed
  live, not the same model as the genuinely multilingual `nomic-embed-text-v2-moe`), which fails
  the EN+PT-BR requirement outright regardless of its footprint/packaging (both of which were
  otherwise competitive — 137M params, best-in-class native GGUF packaging, strongest Matryoshka
  support of the shortlist, still 768-dim not 1024). Recommended lead candidate is unchanged:
  granite-embedding-278m-multilingual, fallback multilingual-e5-small. `nomic-embed-text-v2-moe`
  was noted as the honest substitute if the stakeholder wants to keep the Nomic family in play,
  but is unvetted against the other criteria — open only if pursued.
- 2026-09-19 — Stakeholder set the safety rule for the global default vs. per-workspace override:
  (1) one-time snapshot — every existing workspace gets an explicit override recorded, matching
  its actual current model, before the global default is ever changed; (2) standing rule — every
  newly created workspace automatically gets an explicit override written at creation time,
  capturing the global default as of that moment, so no workspace ever implicitly floats with
  future global-default changes. Confirmed exactly as read back. Rule (2) is to be built into
  workspace bootstrap itself (not left as a manual convention) — captured as FR-1/FR-2.
- 2026-09-19 — `ws:eval` confirmed as the first (validation) migration target, consistent with the
  earlier workspace-coverage discussion and `data-scientist`'s validate-before-migrating
  recommendation. Which real/production workspace(s) need migrating afterward is deliberately
  deferred — stakeholder's priority is that the migration capability itself be easily available
  on demand (captured as FR-9) rather than committing to a production target now.
- 2026-09-19 — Chunk.embedding in scope? → Yes, both `Message` and `Chunk` are in scope (not
  `Message`-only), so document-ingestion data (M5/K-050) doesn't need a second migration pass
  later.
- 2026-09-19 — Resume behavior on interruption? → Idempotent resume (re-running only processes
  unmigrated rows), matching the project's existing backfill-script convention — never a
  requirement to start over from zero.
- 2026-09-19 — Stakeholder confirmed the full readback (intent, FR-1…FR-10, out of scope,
  deliberately-deferred open items). Status flipped to Ready for design.
