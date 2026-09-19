# Agent knowledge-base strategy — DEF-1 held-out embedding trial coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

User-authorized follow-up to `agent-knowledge-base-strategy5-coordination.md` (archived): execute
Recommendation 1 of `data-scientist`'s DEF-1 diagnosis
(`claude/docs/plans/agent-knowledge-base-strategy-ml.md` §"DEF-1 diagnosis", "Recommendation —
what's actually worth trying" item 1) — a bounded, held-out trial of `qwen3-embedding:4b` against
the corpus-neighborhood-density hypothesis. **Only item 1 is in scope here** — items 2 (hybrid
lexical+semantic fusion, routed to `graph-dba`/`architect`, not requested) and 3 (the cheap
`cobb` X1/T1 curation edit, not requested) are separate, undispatched follow-ups, not part of this
coordination.

**Exact recommendation, quoted from `ml.md` (read there for full context, not restated further
here):** re-embed the ~150-200-claim `b-prose` neighborhood plus a `b-code` control sample under
`qwen3-embedding:4b`, re-derive the prefix/floor convention for the new model, and re-run the 4
originally-missed queries (C1, G2, P1, X1) plus 5-10 already-hitting controls. **Acceptance
criterion, verbatim:** "if at least 3 of these 4 queries' expected documents land in top-5, the
model-capacity hypothesis is supported and a fuller corpus re-embed is justified; if fewer
(especially if C1/G2 still miss even at `limit=20` under the 4B model), the hypothesis is
falsified and effort should move to item 2."

## Pre-dispatch check (teco, 2026-09-19)

`curl http://localhost:1234/v1/models` confirmed `qwen3-embedding:4b` (or any 4B-parameter
embedding variant) is **not currently loaded/available** in LM Studio — only
`text-embedding-qwen3-embedding-0.6b`, `text-embedding-granite-embedding-278m-multilingual`, and
`text-embedding-nomic-embed-text-v1.5` are present. **Also found, by chance, while checking**: a
concurrent session has an active, unrelated coordination in flight —
`falkor-chat/docs/plans/embedding-migration.md` (uncommitted, `architect`-owned, not touched by
this coordination) — whose own stated trigger is "Qwen3-Embedding-0.6B is overrunning LM Studio's
memory budget." That's a live memory-pressure signal on the **same shared LM Studio instance**
this trial would also load a model into (a 4B embedding model is materially larger than the 0.6B
already causing strain) — real enough to brief `devops` on explicitly, not something to discover
mid-trial. This coordination does not touch any file from that concurrent effort.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 (provision `qwen3-embedding:4b` in LM Studio, judge memory feasibility) | `devops` | `a7909d6eb3fb54000` | in-flight | User resolved the memory blocker directly (manually unloaded the model on the Windows host) — teco-reverified: GPU now 5848 MiB free of 6141 (was 122 MiB). Resumed to finish provisioning: still needs the correct LM Studio Hub identifier for a 4B Qwen3 embedding model (unresolved as of round 2) | n/a — environment readiness | 82.6k + 90.5k tok / 17+22 tools (2 rounds) |
| U2 (design + run the held-out trial, judge against the acceptance criterion) | `data-scientist` | — | queued (blocked on U1) | — | — → — (advisory, teco-reverified directly, same precedent as U2/U3/U4 in `-strategy5-coordination.md`) | — |

U1 → U2 is a hard sequential dependency (U2 needs the model actually loaded and responding before
it can embed anything) — not dispatched in parallel.

## Session pause (2026-09-19)

Paused here at the user's request, mid-U1. `devops` (agentId `a7909d6eb3fb54000`) is still
running its own background work (resolving the correct LM Studio Hub identifier for a 4B Qwen3
embedding model, then provisioning it) — it had not reported a final result when the pause was
requested. **On resume:** check whether that agentId is still reachable/has a result via
`SendMessage`/a completion notification before re-dispatching anything; if the id no longer
resolves (a new session), state-recovery is cheap here since U1 makes no repo file changes — just
re-check `curl http://localhost:1234/v1/models` for what's now loaded/available and restart from
wherever that leaves off, rather than re-running the whole identifier hunt blind. U2 has not been
dispatched. Nothing in this coordination has been committed elsewhere; this file itself is being
committed now, uncommitted-normally per the docs-only-chain batching convention, at the user's
explicit request to persist state before the pause — not a signal that the chain reached its
terminal state.

## Notes

- **Scope boundary for U2**: the trial must not write to or otherwise mutate production
  `ws:agent-team` — it's a held-out, throwaway comparison, not a migration. Compute/compare
  locally (e.g. a scratch script calling the LM Studio embeddings endpoint directly + cosine
  similarity, or an isolated scratch FalkorDB graph key if a vector index genuinely helps) rather
  than ingesting anything into the live corpus.
- **Deliverable convention**: results land as a new dated section in
  `claude/docs/plans/agent-knowledge-base-strategy-ml.md` (next version bump), per the file's own
  existing convention for addenda — not a new document, and not restated into this coordination
  doc beyond a one-line pointer once delivered.
