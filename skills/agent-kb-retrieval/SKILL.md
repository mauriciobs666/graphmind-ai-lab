---
name: agent-kb-retrieval
description: >-
  The retrieval calling convention for querying the team's distilled-knowledge base —
  every KB claim migrated into falkor-chat's ws:agent-team workspace (K-030 Track 2) —
  via the already-shipped search_documents/get_document MCP tools. Holds, as exact fenced
  literals, the asymmetric query-instruction prefix template a querying agent must build
  itself before calling search_documents (the stored Document/Chunk text is never
  prefixed), the fixed top-K, and the current score-floor value (0.42 cosine distance,
  provisionally calibrated 2026-09-19 by Stage 8 Phase 1's golden-set pilot; Phase 2's
  full run still owed). Use whenever a coding agent
  needs to look up a distilled technique/rule/pattern from the team knowledge base by
  describing its current situation, instead of re-deriving the convention from memory or
  restating it in a prompt. Consuming agents (teco, architect, tdd-engineer,
  frontend-engineer, qa-engineer, analyst, data-scientist, graph-dba, devops) cite this
  skill by one line; they do not duplicate its content.
allowed-tools: mcp__falkor-chat-agent-team__search_documents, mcp__falkor-chat-agent-team__get_document
---

# agent-kb-retrieval — the team knowledge-base retrieval convention

Read this before calling `search_documents` against `ws:agent-team` for anything shaped
like "does the team already have a distilled technique/rule for this situation." It is the
**query-side** convention only — the write side (`ingest_document`, chunking, attribution)
is `claude/docs/plans/agent-knowledge-base-strategy.md` §2/§4.1/§4.5 and stays unchanged by
this file.

## The calling convention, step by step

1. **Build the prefixed query string yourself** — substitute your own situation description
   for `{situation}` in the exact template below. This is a client-side, calling-convention
   string: nothing in `falkor-chat` applies a prefix for you, and nothing strips one you
   forget to add.
2. **Call** `search_documents(query=<the prefixed string>, limit=5)` against `ws:agent-team`.
3. **Full-claim recovery.** A hit reports a `documentId`, not necessarily the whole claim's
   text — a claim can (rarely) split across two chunks of the same document (a mitigated,
   not eliminated, chunking residual — see the plan's §1/§8). If you need the complete text,
   follow up with `get_document(documentId)` rather than trusting one chunk's excerpt.
4. **Apply the score floor yourself** — reject any returned hit with `score > 0.42`. See
   below for the value's derivation and caveats. Do not invent a different number.

## The query-instruction prefix (exact, fenced — do not paraphrase or reformat)

```
f"Instruct: Given a coding agent's description of its current situation, retrieve the distilled technique or rule that applies to it.\nQuery: {situation}"
```

This is the asymmetric instruction-prefix convention Qwen3-Embedding-0.6B (this lab's
standing embedding model, `data-scientist`'s ML note) expects on the **query** side only.
`{situation}` is a placeholder you substitute with your own one- or two-sentence
description of what you're trying to find — not literal text to send as-is. The stored
`Document`/`Chunk` text in `ws:agent-team` is **never** prefixed; only the query is. This
exact string is what `claude/scripts/audit-team.sh`'s drift check greps for — an edit here
that changes it (even a rewrap, a smart-quote substitution, or dropping the `\n`) will fail
that check the next time it runs, which is the intended trip-wire for **definition** drift.
It does **not** and cannot catch **compliance** drift — one agent's live call silently
omitting the prefix — which stays an accepted gap, backstopped only by Stage 8's recurring
golden-set run (`claude/docs/plans/agent-knowledge-base-strategy.md` §7/§8).

## Top-K

**Fixed at 5** — always call `search_documents(query=..., limit=5)`. Not user-tunable per
call; a different value here is itself a drift signal, not a legitimate per-situation
choice.

## Score floor — calibrated, apply it

**Reject any hit with `score > 0.42`** (cosine distance; lower = more similar). Calibrated
2026-09-19 by `data-scientist` (Stage 8 Phase 1 pilot, `claude/docs/plans/
agent-knowledge-base-strategy-ml.md`, "Stage 8 Phase 1" section) against 16 of a 45-pair
golden set, run with the exact prefix template above, against the post-Stage-6-migration
corpus (327/332 claims `ready`). Derivation: the worst-scoring true positive that was
actually found in the pilot scored 0.409; the closest false match returned for a genuine
negative query (no matching entry in the corpus at all) scored 0.446; 0.42 sits in that
0.037 gap. At this floor, none of the pilot's 14 found true-positive documents were wrongly
dropped, and all 4 pilot negative queries were correctly rejected.

**This is a first, provisional floor, not a finished calibration — treat it as directional,
not exact.** The margin above is thin, and the pilot's negative queries are the same ones
used to derive it (no held-out validation yet). Stage 8 Phase 2 (the full ~40-45-pair
regression gate, `qa-engineer`, not yet dispatched as of this writing) runs the 29
not-yet-executed rows of the same golden-set design — genuine out-of-sample evidence for
this exact number — and should confirm or re-derive it from the pooled result rather than
patching 0.42 in isolation. If you're reading this well after Stage 8 Phase 2 has landed and
this section still cites only the Phase 1 pilot, treat it as stale and flag it to `cobb`.

## `familyId` sibling linkage — not available through this tool

A claim split from one original heading during migration shares a title-prefix convention
(`"<family-slug> — <claim-title>"`) but `search_documents` does no graph traversal to pull
siblings automatically (plan §2/§4.4). If a retrieved claim reads as partial, a manual
`list_documents` scan for the same title prefix is the only recovery path today — accepted
as a convenience loss, not engineered around.

## What this skill does not cover

- **Writing** into `ws:agent-team` (`ingest_document`, `produced_by`, chunking granularity,
  delete-then-recreate on revision) — `claude/docs/plans/agent-knowledge-base-strategy.md`
  §2/§4.1/§4.5, and each agent's own "Learning capture" prompt section.
- **Raw kaizen capture** (Track 1) vs. **distilled KB content** (Track 2) — both currently
  live in the same `ws:agent-team` workspace and are both reachable through the same
  `search_documents` call; this skill's convention applies to querying either kind, the
  plan's content-model section (§2) covers what distinguishes them on the write side.
