# Agent knowledge-base strategy — DEF-1 held-out embedding trial coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

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
| U1 (provision `qwen3-embedding:4b` in LM Studio, judge memory feasibility) | `devops` | `a7909d6eb3fb54000` | accepted | Resolved via HF URL (`Qwen/Qwen3-Embedding-4B-GGUF`, Q4_K_M, 2.50 GB) since the Ollama-style tag and bare Hub-path guesses never resolved (same issue affected the 0.6B sibling — imported via direct URL, not the curated catalog). Loaded, API id `text-embedding-qwen3-embedding-4b` — teco-reverified independently: GPU free (360 MiB) matches report, `/v1/models` lists it, and a fresh `/v1/embeddings` call (different input than devops used) returned a correct 2560-dim non-degenerate vector. **Flagged for U2**: only ~357-360 MiB VRAM free with this model resident — no headroom for a second concurrent model load during the trial | n/a — environment readiness | 82.6k + 90.5k + 111.9k tok / 17+22+42 tools (3 rounds) |
| U2 (design + run the held-out trial, judge against the acceptance criterion) | `data-scientist` | `ab536d41ef3d3ef04` | accepted | `claude/docs/plans/agent-knowledge-base-strategy-ml.md` (Version 6→7, new dated section), commit pending — teco-reverified: independently re-embedded the C1 query + its expected document myself (fresh Python script, same model/prefix), computed cosine distance = 0.44374… vs. reported 0.4437 — exact match; confirmed the reported `mcp__cypher__query` truncation finding directly (same document showed a "(+525 chars)" truncation marker); confirmed no `ingest_document` call touched the KB corpus (only the standing raw-capture kaizen write, expected of every agent, separate from the corpus). **Verdict: hypothesis falsified** — 2/4 target queries (P1, X1) land in top-5, below the 3/4 bar; C1 and G2 still miss. Recommends moving to item 2 (hybrid lexical+semantic fusion, `graph-dba`/`architect`) — not yet authorized/dispatched | — → — (advisory, teco-reverified directly, same precedent as U2/U3/U4 in `-strategy5-coordination.md`) | 161.6k tok / 51 tools / 597s |

U1 → U2 is a hard sequential dependency (U2 needs the model actually loaded and responding before
it can embed anything) — not dispatched in parallel.

## Closing summary (2026-09-19)

Both units accepted. Recommendation 1 (the `qwen3-embedding:4b` held-out trial) executed exactly
as scoped, with real evidence: **the model-capacity hypothesis is falsified** — only 2 of the 4
originally-missed queries (P1, X1) land in top-5 under the larger model, below the 3-of-4
acceptance bar; C1 and G2 still miss. Model capacity is a real, partial lever (it measurably
helped P1/X1) but does not close the gap alone, especially for G2's pattern (many loosely-related
competitors rather than one identifiable near-duplicate). Full results, method, and honest
limitations (held-out slice not full corpus, no cross-session stability check, floor only
illustrated not re-derived, production cost/latency untested) are in `agent-knowledge-base-
strategy-ml.md`'s new dated section — not restated here.

**Next step, per the diagnosis's own ranking, not yet authorized:** item 2, hybrid lexical+
semantic score fusion, routed to `graph-dba`/`architect` — this trial's own result (P1/G2 sharing
a lexical-overlap-without-embedding-win pattern) is itself additional evidence for that direction.
Not dispatched; a decision for the user.

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
