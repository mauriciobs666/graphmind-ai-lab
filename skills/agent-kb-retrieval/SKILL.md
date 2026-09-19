---
name: agent-kb-retrieval
description: >-
  The retrieval calling convention for querying the team's distilled-knowledge base —
  every KB claim migrated into falkor-chat's ws:agent-team workspace (K-030 Track 2) —
  via the already-shipped search_documents/get_document MCP tools. Holds, as exact fenced
  literals, the asymmetric query-instruction prefix template a querying agent must build
  itself before calling search_documents (the stored Document/Chunk text is never
  prefixed), the fixed top-K, and the current score-floor value (0.43 cosine distance,
  interim per Stage 8 Phase 2's full-set gate — one borderline document's score is confirmed
  unstable across sessions and its floor-safety is not fully certified, escalated to
  data-scientist). Use whenever a coding agent
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

## Score floor — interim value, one known unresolved risk

**Reject any hit with `score > 0.43`** (cosine distance; lower = more similar). This is an
**interim** value, not a finished calibration — see below before treating it as settled.
Landed 2026-09-19 by `qa-engineer` (Stage 8 Phase 2, `claude/docs/test-reports/
agent-knowledge-base-strategy-ac2-report.md`, revised in place 2026-09-19 per `analyst`'s
review, `claude/docs/reviews/agent-knowledge-base-strategy4-stage8-phase2.md`), executing the
29 design-only rows of the 45-pair golden set `data-scientist` designed and partially piloted
in Stage 8 Phase 1 (`claude/docs/plans/agent-knowledge-base-strategy-ml.md`, "Stage 8 Phase
1"), pooled with Phase 1's 16 executed rows.

**0.43 is confirmed safe for every pooled found true-positive document except one, and for
all 6 pooled negative queries.** The worst of the other 40 found true positives is 0.4140
(C4); the closest false match on a genuine negative is 0.446 (N4, stable across three
independent measurements). **One document's score is confirmed unstable across sessions**:
the second sibling of stratum-(e) family h40 (`5b1b477ff67e4e3b81c57f14899bbe48`, query R6)
scored **0.4201** in three separate live calls in one session, and a stable-but-different
**0.4405** in `analyst`'s independent review session (two calls, byte-identical to each
other) — both readings internally reproducible, but 0.0204 apart, which is not ordinary
per-call jitter. **No fixed two-decimal floor can be shown safe against this specific
document across both observed readings**: 0.43 admits the 0.4201 reading but rejects the
0.4405 reading; a floor high enough to admit 0.4405 (≥0.4405) would leave only 0.0055 margin
to the closest negative (0.446) — no working safety margin at all.

**This is a genuine, unresolved methodology question, not a number this file should silently
patch.** Escalated to `data-scientist`: is this instability isolated to this one
query/document pair, or does it indicate broader backend non-determinism worth
characterizing systematically — and is a fixed-point floor even the right mechanism given
confirmed sub-0.02 instability this close to the decision boundary? Until that consult
lands, **0.43 stays the operative floor** (correct for every other measured case), but do
not read "the floor is calibrated" as meaning R6/h40's second sibling is guaranteed
admitted — it may be wrongly rejected depending on which backend state answers a given call,
a known, accepted, named residual risk. Full derivation, the instability investigation, and
the pooled score data are in the Stage 8 Phase 2 report cited above; this section states only
the operative number and the caveat, per this skill's own "point to the source, don't
duplicate" convention.

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
