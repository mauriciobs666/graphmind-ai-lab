// ddl_v1.cypher -- the index and constraint set of the connected-code-graph store, model 1.1.
// Spec: docs/plans/connected-code-graph-p0-graph.md section 1.6 (D10, S-1) and 1.3 (SkippedFile), plus
// docs/plans/connected-code-graph-p1-graph.md section 4 (ConfigEntry).
// This file IS the DDL; cpg/tests/test_ddl_live.py checks it against both tables.
//
// Engine: falkordb v4.18.11. Two statement kinds, one per ';', comments are '//' lines:
//   CREATE INDEX FOR (n:<Label>) ON (n.<prop>);          Cypher; run with GRAPH.QUERY
//   GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE <Label> PROPERTIES 1 <prop>;
//                                                        a Redis command, NOT Cypher; <key> is the
//                                                        graph key the statement is run against
//
// ORDER AND APPLY PROTOCOL (FalkorDB: a UNIQUE constraint needs an existing exact-match index on
// the same label+property, so index-before-constraint is mandatory, and the loader's edge
// MATCHes need the indexes before any data):
//   1. Apply PART 1 (indexes) to the EMPTY build key, in file order, BEFORE the first node.
//      The first CREATE INDEX creates the graph key.
//   2. Load nodes, then edges (UNWIND batches; an explicit query timeout = TIMEOUT_MAX on every
//      query whenever TIMEOUT_MAX is nonzero, D16).
//   3. Apply PART 2 (constraints) AFTER the load, in file order. Creation is ASYNCHRONOUS:
//      the command returns at once and the constraint is PENDING; poll CALL db.constraints()
//      until every one is OPERATIONAL (bounded wait). Any FAILED fails the build.
//   4. Never apply PART 2 before PART 1 has fully run; never put constraints on the empty key
//      "to save a step" (the backfill-free path is the point of loading first).
// Deferred to the phase that loads the label/edge (section 1.6; Pack and Snapshot stay unindexed, p1-graph 4):
// SchemaEntry/Mark/Finding/LogTemplate id indexes, READS/WRITES/DISMISSED linkId relationship
// indexes, full-text index. SkippedFile has indexes only, no constraint (section 1.3).

// ---- PART 1: indexes (20), empty key, before the load -------------------------------------
CREATE INDEX FOR (n:Repo) ON (n.repoId);
CREATE INDEX FOR (n:Unit) ON (n.unitId);
CREATE INDEX FOR (n:File) ON (n.fileId);
CREATE INDEX FOR (n:File) ON (n.path);
CREATE INDEX FOR (n:SkippedFile) ON (n.path);
CREATE INDEX FOR (n:SkippedFile) ON (n.repoId);
CREATE INDEX FOR (n:Symbol) ON (n.symbolId);
CREATE INDEX FOR (n:Symbol) ON (n.nameNorm);
CREATE INDEX FOR (n:Symbol) ON (n.qname);
CREATE INDEX FOR (n:CallSite) ON (n.callSiteId);
CREATE INDEX FOR (n:Param) ON (n.pointId);
CREATE INDEX FOR (n:Ret) ON (n.pointId);
CREATE INDEX FOR (n:Cond) ON (n.pointId);
CREATE INDEX FOR (n:Name) ON (n.nameKey);
CREATE INDEX FOR (n:ContractPoint) ON (n.cpId);
CREATE INDEX FOR (n:ContractPoint) ON (n.keyNorm);
CREATE INDEX FOR (n:Link) ON (n.linkId);
CREATE INDEX FOR (n:Dependency) ON (n.depId);
CREATE INDEX FOR (n:ConfigEntry) ON (n.entryId);
CREATE INDEX FOR (n:ConfigEntry) ON (n.path);

// ---- PART 2: UNIQUE constraints (12), after the load, polled to OPERATIONAL ---------------
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE Unit PROPERTIES 1 unitId;
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE File PROPERTIES 1 fileId;
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE Symbol PROPERTIES 1 symbolId;
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE CallSite PROPERTIES 1 callSiteId;
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE Param PROPERTIES 1 pointId;
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE Ret PROPERTIES 1 pointId;
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE Cond PROPERTIES 1 pointId;
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE Name PROPERTIES 1 nameKey;
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE ContractPoint PROPERTIES 1 cpId;
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE Link PROPERTIES 1 linkId;
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE Dependency PROPERTIES 1 depId;
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE ConfigEntry PROPERTIES 1 entryId;
