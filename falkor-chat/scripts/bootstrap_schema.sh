#!/usr/bin/env bash
# bootstrap_schema.sh — create indexes and constraints for falkor-chat graphs.
#
# Usage:
#   ./scripts/bootstrap_schema.sh                      # reference graph only
#   ./scripts/bootstrap_schema.sh acme globex          # reference + ws:acme + ws:globex
#
# Idempotency: re-running is safe. FalkorDB returns "already indexed" /
# "Constraint already exists" for duplicates but does NOT abort — redis-cli
# exits 0 on Redis-level errors, so the script continues cleanly.
#
# Ordering rule (live-verified): GRAPH.CONSTRAINT CREATE requires an existing
# range index on the same property. Indexes are always created before constraints.
#
# Env vars:
#   FALKORDB_HOST  (default: 127.0.0.1)
#   FALKORDB_PORT  (default: 6379)

set -euo pipefail

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-6379}"

gquery() {
  # gquery <graph> <cypher>
  redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$1" "$2"
}

gconstraint() {
  # gconstraint <graph> UNIQUE NODE <Label> PROPERTIES <n> <prop...>
  redis-cli -h "$HOST" -p "$PORT" GRAPH.CONSTRAINT CREATE "$@"
}

# ─────────────────────────────────────────────────────────────
# reference graph
# ─────────────────────────────────────────────────────────────
bootstrap_reference() {
  local g="reference"
  echo ""
  echo "── reference graph ──────────────────────────────────────"

  echo "[index] WorkflowDef.key"
  gquery "$g" "CREATE INDEX FOR (n:WorkflowDef) ON (n.key)"

  echo "[index] WorkflowDef.version"
  gquery "$g" "CREATE INDEX FOR (n:WorkflowDef) ON (n.version)"

  echo "[index] Entity.entityId"
  gquery "$g" "CREATE INDEX FOR (n:Entity) ON (n.entityId)"

  # Step.key: display/traversal anchor only — never a global identity (a step key
  # is unique only *within* a def), so it carries NO uniqueness constraint (§7.2).
  echo "[index] Step.key"
  gquery "$g" "CREATE INDEX FOR (n:Step) ON (n.key)"

  # Step.stepUid: synthetic MERGE-backing identity "{defKey}:{version}:{stepKey}"
  # (globally unique within the graph) — index first so its UNIQUE constraint can
  # attach (K-020). This is what lets publish/materialize MERGE steps idempotently.
  echo "[index] Step.stepUid"
  gquery "$g" "CREATE INDEX FOR (n:Step) ON (n.stepUid)"

  # K-052 M6: Product catalog (`docs/plans/workflow-catalog-lookup.md` §3.1) — a
  # global, seed-script-write-only reference catalog, looked up by the salesperson
  # demo's tools. productId is the standard {label}Id identity anchor. Index first
  # so the UNIQUE constraint below can attach (§7.1 rule 1).
  echo "[index] Product.productId"
  gquery "$g" "CREATE INDEX FOR (n:Product) ON (n.productId)"

  # nameNormalized backs lookup_product's exact-name `=` comparison (FR-1/AC-1/
  # AC-4) — same tier-mechanics precedent as Entity.nameNormalized above. Range
  # index, no constraint: two products could in principle share a normalized name
  # in a different category.
  echo "[index] Product.nameNormalized"
  gquery "$g" "CREATE INDEX FOR (n:Product) ON (n.nameNormalized)"

  # categoryNormalized backs filter_products' category equality predicate
  # case-insensitively (K-049-adjacent live finding, 2026-08-28: a live LLM
  # call lowercased a category argument and the exact-case `category` match
  # silently returned zero rows). Same normalized-property + exact-match
  # precedent as nameNormalized above, not a runtime toLower() — see
  # QUERIES.md §15.2's sargability note for why. No constraint, filtering
  # only. The raw `category` property carries no index of its own (same
  # convention as `name`/`nameNormalized`: only the normalized property is
  # indexed; the raw property is returned, never filtered on).
  echo "[index] Product.categoryNormalized"
  gquery "$g" "CREATE INDEX FOR (n:Product) ON (n.categoryNormalized)"

  # Hot-filter index backing filter_products' price-range shape (FR-3) — no
  # constraint, filtering only. This is the query's actual scan anchor.
  echo "[index] Product.price"
  gquery "$g" "CREATE INDEX FOR (n:Product) ON (n.price)"

  echo "[constraint] WorkflowDef unique {key, version}"
  gconstraint "$g" UNIQUE NODE WorkflowDef PROPERTIES 2 key version

  echo "[constraint] Entity unique {entityId}"
  gconstraint "$g" UNIQUE NODE Entity PROPERTIES 1 entityId

  echo "[constraint] Step unique {stepUid}"
  gconstraint "$g" UNIQUE NODE Step PROPERTIES 1 stepUid

  echo "[constraint] Product unique {productId}"
  gconstraint "$g" UNIQUE NODE Product PROPERTIES 1 productId
}

# ─────────────────────────────────────────────────────────────
# workspace graph  ws:{workspaceId}
# ─────────────────────────────────────────────────────────────
bootstrap_workspace() {
  local wid="$1"
  local g="ws:${wid}"
  echo ""
  echo "── workspace graph: ${g} ────────────────────────────────"

  # ── identity anchors (must exist before constraints) ────────
  echo "[index] User.userId"
  gquery "$g" "CREATE INDEX FOR (n:User) ON (n.userId)"

  echo "[index] Agent.agentId"
  gquery "$g" "CREATE INDEX FOR (n:Agent) ON (n.agentId)"

  echo "[index] Channel.channelId"
  gquery "$g" "CREATE INDEX FOR (n:Channel) ON (n.channelId)"

  echo "[index] Thread.threadId"
  gquery "$g" "CREATE INDEX FOR (n:Thread) ON (n.threadId)"

  echo "[index] Message.msgId"
  gquery "$g" "CREATE INDEX FOR (n:Message) ON (n.msgId)"

  echo "[index] Document.documentId"
  gquery "$g" "CREATE INDEX FOR (n:Document) ON (n.documentId)"

  echo "[index] Chunk.chunkId"
  gquery "$g" "CREATE INDEX FOR (n:Chunk) ON (n.chunkId)"

  echo "[index] Entity.entityId"
  gquery "$g" "CREATE INDEX FOR (n:Entity) ON (n.entityId)"

  # K-050 M5 Stage 3: the FR-8 exact-tier fusion lookup (a future stage) needs a
  # real `=` comparison, decoupled from RediSearch tokenizer/stemmer behavior —
  # a plain RANGE index, no uniqueness constraint (distinct real entities can
  # share a normalized name+type before fusion runs). Written starting this
  # stage (`repository.create_entity`) even though fusion is the first reader,
  # so no backfill migration is needed later (`document-ingestion-graph.md` §2.3).
  echo "[index] Entity.nameNormalized"
  gquery "$g" "CREATE INDEX FOR (n:Entity) ON (n.nameNormalized)"

  # K-050 M5 Stage 4 (FR-6/7/8/9/10): the SAME_AS fusion edge — relationship-
  # scoped index-before-constraint, same ordering rule as every node identity
  # above (`document-ingestion-graph.md` §1.1/§1.5). Every SAME_AS-anchored
  # query matches its endpoints UNLABELED ((a)/(b), never (a:Entity)) — a bare
  # label on either endpoint forces a full Node By Label Scan even though the
  # relationship-property scan alone is fully selective (graph note §1.4).
  echo "[index] SAME_AS.matchId"
  gquery "$g" "CREATE INDEX FOR ()-[r:SAME_AS]-() ON (r.matchId)"

  echo "[index] SAME_AS.status"
  gquery "$g" "CREATE INDEX FOR ()-[r:SAME_AS]-() ON (r.status)"

  # Materialized snapshot steps land in the workspace graph too (K-021), so the
  # same Step identity DDL as the reference graph applies here.
  # Step.key: display/traversal anchor only, no constraint (§7.1).
  echo "[index] Step.key"
  gquery "$g" "CREATE INDEX FOR (n:Step) ON (n.key)"

  # Step.stepUid: synthetic MERGE-backing identity — index first, constraint below.
  echo "[index] Step.stepUid"
  gquery "$g" "CREATE INDEX FOR (n:Step) ON (n.stepUid)"

  echo "[index] WorkflowDefSnapshot.key"
  gquery "$g" "CREATE INDEX FOR (n:WorkflowDefSnapshot) ON (n.key)"

  echo "[index] WorkflowDefSnapshot.version"
  gquery "$g" "CREATE INDEX FOR (n:WorkflowDefSnapshot) ON (n.version)"

  echo "[index] WorkflowRun.runId"
  gquery "$g" "CREATE INDEX FOR (n:WorkflowRun) ON (n.runId)"

  echo "[index] StepRun.stepRunId"
  gquery "$g" "CREATE INDEX FOR (n:StepRun) ON (n.stepRunId)"

  # TraceEvent: debug-only per-run trace records (M3 executor, K-022). Index first
  # so the UNIQUE constraint can attach (index-before-constraint). Written ONLY for
  # debug runs (WorkflowRun.trace = true); non-debug runs create zero TraceEvent
  # nodes — this DDL is inert RAM until a debug run writes to it.
  echo "[index] TraceEvent.traceId"
  gquery "$g" "CREATE INDEX FOR (n:TraceEvent) ON (n.traceId)"

  echo "[index] ReadCursor.cursorId"
  gquery "$g" "CREATE INDEX FOR (n:ReadCursor) ON (n.cursorId)"

  # ── hot-filter indexes (no constraint needed) ────────────────
  echo "[index] Thread.updatedAt"
  gquery "$g" "CREATE INDEX FOR (n:Thread) ON (n.updatedAt)"

  echo "[index] Message.createdAt"
  gquery "$g" "CREATE INDEX FOR (n:Message) ON (n.createdAt)"

  echo "[index] WorkflowRun.status"
  gquery "$g" "CREATE INDEX FOR (n:WorkflowRun) ON (n.status)"

  # WorkflowRun.startedAt (K-036 / U1): NOT sufficient on its own to anchor
  # find_runs_for_thread's plan (a WHERE-filtered ORDER BY-only property does not
  # pull the label-scan anchor by itself — live-verified, QUERIES.md §12.14) but
  # becomes a real Node By Index Scan once paired with that query's supporting
  # `r.startedAt >= 0` predicate. WorkflowRun cardinality is tiny per workspace
  # (same argument §12.9 already accepted for the status index) — RAM cost is
  # negligible.
  echo "[index] WorkflowRun.startedAt"
  gquery "$g" "CREATE INDEX FOR (n:WorkflowRun) ON (n.startedAt)"

  echo "[index] StepRun.status"
  gquery "$g" "CREATE INDEX FOR (n:StepRun) ON (n.status)"

  # WorkspaceConfig: workspace-level configuration singleton (K-042 Landing 2,
  # FR-16/FR-17 — the model-override hard cap). One row per workspace
  # (workspaceConfigId: 'default'); index first so the UNIQUE constraint below can
  # attach. `docs/plans/llm-provider-config-graph.md` §2.2/§4.
  echo "[index] WorkspaceConfig.workspaceConfigId"
  gquery "$g" "CREATE INDEX FOR (n:WorkspaceConfig) ON (n.workspaceConfigId)"

  # ── cart / order anchors (K-053 M6, workflow-cart-and-totals-graph.md §7) ────
  echo "[index] Customer.customerId"
  gquery "$g" "CREATE INDEX FOR (n:Customer) ON (n.customerId)"

  echo "[index] Cart.customerId"
  gquery "$g" "CREATE INDEX FOR (n:Cart) ON (n.customerId)"

  echo "[index] CartItem.customerId"
  gquery "$g" "CREATE INDEX FOR (n:CartItem) ON (n.customerId)"

  echo "[index] CartItem.productId"
  gquery "$g" "CREATE INDEX FOR (n:CartItem) ON (n.productId)"

  echo "[index] Order.orderId"
  gquery "$g" "CREATE INDEX FOR (n:Order) ON (n.orderId)"

  # ── uniqueness constraints ───────────────────────────────────
  echo "[constraint] User unique {userId}"
  gconstraint "$g" UNIQUE NODE User PROPERTIES 1 userId

  echo "[constraint] Agent unique {agentId}"
  gconstraint "$g" UNIQUE NODE Agent PROPERTIES 1 agentId

  echo "[constraint] Channel unique {channelId}"
  gconstraint "$g" UNIQUE NODE Channel PROPERTIES 1 channelId

  echo "[constraint] Thread unique {threadId}"
  gconstraint "$g" UNIQUE NODE Thread PROPERTIES 1 threadId

  echo "[constraint] Message unique {msgId}"
  gconstraint "$g" UNIQUE NODE Message PROPERTIES 1 msgId

  echo "[constraint] Document unique {documentId}"
  gconstraint "$g" UNIQUE NODE Document PROPERTIES 1 documentId

  echo "[constraint] Chunk unique {chunkId}"
  gconstraint "$g" UNIQUE NODE Chunk PROPERTIES 1 chunkId

  echo "[constraint] Entity unique {entityId}"
  gconstraint "$g" UNIQUE NODE Entity PROPERTIES 1 entityId

  echo "[constraint] SAME_AS unique {matchId}"
  gconstraint "$g" UNIQUE RELATIONSHIP SAME_AS PROPERTIES 1 matchId

  echo "[constraint] Step unique {stepUid}"
  gconstraint "$g" UNIQUE NODE Step PROPERTIES 1 stepUid

  echo "[constraint] WorkflowRun unique {runId}"
  gconstraint "$g" UNIQUE NODE WorkflowRun PROPERTIES 1 runId

  echo "[constraint] StepRun unique {stepRunId}"
  gconstraint "$g" UNIQUE NODE StepRun PROPERTIES 1 stepRunId

  echo "[constraint] TraceEvent unique {traceId}"
  gconstraint "$g" UNIQUE NODE TraceEvent PROPERTIES 1 traceId

  echo "[constraint] WorkflowDefSnapshot unique {key, version}"
  gconstraint "$g" UNIQUE NODE WorkflowDefSnapshot PROPERTIES 2 key version

  echo "[constraint] ReadCursor unique {cursorId}"
  gconstraint "$g" UNIQUE NODE ReadCursor PROPERTIES 1 cursorId

  echo "[constraint] WorkspaceConfig unique {workspaceConfigId}"
  gconstraint "$g" UNIQUE NODE WorkspaceConfig PROPERTIES 1 workspaceConfigId

  echo "[constraint] Customer unique {customerId}"
  gconstraint "$g" UNIQUE NODE Customer PROPERTIES 1 customerId

  echo "[constraint] Cart unique {customerId}"
  gconstraint "$g" UNIQUE NODE Cart PROPERTIES 1 customerId

  echo "[constraint] CartItem unique {customerId, productId}"
  gconstraint "$g" UNIQUE NODE CartItem PROPERTIES 2 customerId productId

  echo "[constraint] Order unique {orderId}"
  gconstraint "$g" UNIQUE NODE Order PROPERTIES 1 orderId

  # ── full-text index ─────────────────────────────────────────
  echo "[fulltext] Message.text"
  gquery "$g" "CALL db.idx.fulltext.createNodeIndex('Message', 'text')"

  # K-050 M5 Stage 3: the FR-9 suggested-tier fusion lookup (a future stage) —
  # inert until then, same "bootstrapped, not yet queried" posture the original
  # Chunk scaffolding demonstrated (`document-ingestion-graph.md` §2.3/§5).
  echo "[fulltext] Entity.name"
  gquery "$g" "CALL db.idx.fulltext.createNodeIndex('Entity', 'name')"

  # ── vector indexes ───────────────────────────────────────────
  # Dimension must match the embedding model and is FIXED at index creation —
  # it cannot be altered in place, so choose it per model BEFORE creating the
  # workspace. Default kept at 1536 (text-embedding-ada-002; still the DESIGN §13
  # open question) so the default is model-neutral.
  #
  # M2 GraphRAG (K-008): the locked M2 stack is Qwen3-Embedding-0.6B, cosine,
  # EMBEDDING_DIM=1024 (DESIGN §1.3). Any workspace that will run §6 hybrid
  # retrieval MUST be bootstrapped at 1024 — the dimension is enforced at
  # query time ("Vector dimension mismatch, expected N but got M") and cannot
  # be changed afterward without dropping/recreating the index. Quirk to know:
  # a wrong-dimension vecf32 write is silently accepted at SET (no error) but
  # the node then falls out of ANN results — validate embedding length client-
  # side. RAM at 1024: ~12.4 KB/message ≈ 1.25 GB per 100k-message workspace
  # (DESIGN §11; GRAPH.MEMORY USAGE under-reports vector memory).
  #   EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh <ws>
  local dim="${EMBEDDING_DIM:-1536}"

  echo "[vector] Message.embedding (dim=${dim}, cosine)"
  gquery "$g" "CREATE VECTOR INDEX FOR (n:Message) ON (n.embedding) OPTIONS {dimension:${dim}, similarityFunction:'cosine'}"

  echo "[vector] Chunk.embedding (dim=${dim}, cosine)"
  gquery "$g" "CREATE VECTOR INDEX FOR (n:Chunk) ON (n.embedding) OPTIONS {dimension:${dim}, similarityFunction:'cosine'}"

  echo "done: ${g}"
}

# ─────────────────────────────────────────────────────────────
# verify connectivity
# ─────────────────────────────────────────────────────────────
echo "Checking FalkorDB at ${HOST}:${PORT}..."
redis-cli -h "$HOST" -p "$PORT" PING | grep -q PONG || {
  echo "ERROR: cannot reach FalkorDB at ${HOST}:${PORT}" >&2
  exit 1
}
echo "OK"

# ─────────────────────────────────────────────────────────────
# run
# ─────────────────────────────────────────────────────────────
bootstrap_reference

for wid in "$@"; do
  bootstrap_workspace "$wid"
done

echo ""
echo "Bootstrap complete."
echo "Verify with:"
echo "  redis-cli -h ${HOST} -p ${PORT} GRAPH.QUERY reference \"CALL db.indexes()\""
for wid in "$@"; do
  echo "  redis-cli -h ${HOST} -p ${PORT} GRAPH.QUERY ws:${wid} \"CALL db.constraints()\""
done
