---
name: agent-kb-retrieval
description: >-
  The retrieval calling convention for querying the team's distilled-knowledge base —
  every KB claim migrated into falkor-chat's ws:agent-team workspace (K-030 Track 2) —
  via the already-shipped search_documents/get_document MCP tools. Holds, as exact fenced
  literals, the asymmetric query-instruction prefix template a querying agent must build
  itself before calling search_documents (the stored Document/Chunk text is never
  prefixed), the fixed top-K, and the current score-floor value (0.43 cosine distance,
  re-derived 2026-09-19 by Stage 8 Phase 2's full-set gate over the pooled 45-pair golden
  set). Use whenever a coding agent
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
4. **Apply the score floor yourself** — reject any returned hit with `score > 0.43`. See
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

**Reject any hit with `score > 0.43`** (cosine distance; lower = more similar). Re-derived
2026-09-19 by `qa-engineer` (Stage 8 Phase 2, `claude/docs/test-reports/
agent-knowledge-base-strategy-ac2-report.md`), executing the 29 design-only rows of the
same 45-pair golden set `data-scientist` designed and partially piloted in Stage 8 Phase 1
(`claude/docs/plans/agent-knowledge-base-strategy-ml.md`, "Stage 8 Phase 1" section), then
pooling with Phase 1's 16 executed rows for the full 45-pair sample.

**Why 0.42 didn't survive out-of-sample validation.** Phase 1's provisional floor (0.42) was
fit on only 4 negative queries and 14 found positives, margin 0.037 — explicitly flagged
there as thin and non-held-out. Phase 2 found a genuine true positive scoring *above* it: the
second sibling of stratum-(e) family h40 (`5b1b477ff67e4e3b81c57f14899bbe48`, query R6)
scored **0.4201** — over the 0.42 floor by 0.0001, which would wrongly reject a real answer
that the retrieval step actually found. Per this file's own standing instruction ("if a Phase
2 positive scores above it... re-derive"), the floor is re-derived rather than patched.

**Derivation, pooled over all 45 pairs (14 Phase 1 + 27 Phase 2 found true-positive
documents; 4 Phase 1 + 2 Phase 2 negative queries):** the worst-scoring true positive found
anywhere in the pooled sample is 0.4201 (the h40 sibling above); the closest false match
returned for a genuine negative query is still 0.446 (N4, Phase 1, React Server Components →
a React-18-batching claim — unchanged by Phase 2's 2 additional negatives, N5/N6, whose
closest scores were 0.614 and 0.463, both further away). Gap: **0.0259**. **0.43** sits in
that gap, ~0.0099 above the worst surviving true positive and ~0.016 below the closest
negative false match. At this floor: **0 of the 41 pooled found true-positive documents are
wrongly dropped**, and **all 6 pooled negative queries (N1-N6) are correctly rejected**.

**This is now a full-set calibration, not a provisional one — re-derive only on a material
change** (a corpus/prefix change per the standing discipline above, or a future regression run
that again finds a positive scoring above 0.43 or a negative scoring below it). Full
per-row score data, the crowding-out assessment, and the prose-vs-code retrieval-quality
finding are in the Stage 8 Phase 2 report cited above — this section states only the number
and its derivation, per this skill's own "point to the source, don't duplicate" convention.

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
