# Kaizen agent/learning-note ontology — Test Report (M8, S6)

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** — (M8)

## Summary

Executed `docs/test-plans/kaizen-agent-ontology.md` (13 items, TP-001…TP-013) against the real,
shared `kaizen_team` FalkorDB graph, on 2026-08-22, targeting the freshly rebuilt `cypher-mcp`
image `cypher-mcp:cb712173ab57` (content-hash confirmed current against the repo's checked-in
`cypher-mcp/server.py` — recomputed via `cypher-mcp/image-tag.sh` myself, not merely assumed).

**Overall verdict: PASS.** All three adversarial attacks the `analyst` review traced (A, B, and
C's `SET`/`REMOVE` sub-variants) are correctly rejected by the deployed, rebuilt container. One
full producer-write → MENTIONS-tag → count-and-decide → partial-resolve ×2 → full-node-clear
distillation dry-run completed successfully with disposable identities, and the graph was left
clean (zero residual `:KaizenEntry` nodes from this run; the graph's `:KaizenEntry` count returned
to its exact pre-run baseline of 27). No regression found in the shipped closure. Two findings are
recorded below — one operational/testability finding significant enough to flag prominently
(the MCP connection I inherited from this run's parent session was silently bound to a **stale,
pre-M8 image**, not the rebuilt one the task briefed me to expect "automatically"), and one
informational note about the review's literal attack-text grammar. Neither is a regression in the
M8 code itself; both are documented in full below because they materially affected how this pass
had to be conducted.

**CPG:** considered, not relevant — this is a live black-box QA pass against a running MCP
container and FalkorDB graph (execution, not static code reading); `GRAPH.LIST` carries no CPG for
`cypher-mcp`/`claude`, consistent with the plan's and review's own `CPG:` lines for this same
delivery.

## Results table

| ID | Result | Evidence |
|---|---|---|
| TP-001 | PASS | Producer-write (`agent='_qa_selftest_producer_4e24af1e'`) → `write ok (labels_added=2, nodes_created=2, properties_set=9, relationships_created=1)`. |
| TP-002 (Attack A) | PASS (see Finding 2) | Literal review text → FalkorDB parse error, not an authorization decision (Finding 2). Grammar-valid equivalent (`WITH 1 AS _dummy` bridge) → `Rejected: this statement combines a valid author-write with another recognized shape's trigger (a bare MERGE, DELETE, SET, or REMOVE elsewhere in the same statement)...`. Victim entry confirmed unmodified (TP-006). |
| TP-003 (Attack B) | PASS | Literal review-shaped text (valid grammar as-is — no `MATCH` follows an updating clause) → same rejection message, trigger `MERGE`. No forged `:Agent {agentId:'_qa_selftest_forged_4e24af1e'}` node created (confirmed absent). |
| TP-004 (Attack C, `SET`) | PASS (see Finding 2) | Same grammar issue/fix as TP-002; grammar-valid variant → same rejection message, trigger `SET`. Victim's `author`/`fact` confirmed unchanged. |
| TP-005 (Attack C, `REMOVE`) | PASS (see Finding 2) | Same pattern; grammar-valid variant → same rejection message, trigger `REMOVE`. |
| TP-006 | PASS | Re-read after all four attacks: `qa-selftest-victim-4e24af1e` — `fact='disposable QA victim entry'` (unchanged), `author=null` (never set); zero `qa-selftest-decoy-4e24af1e-*` nodes exist; zero `_qa_selftest_forged_4e24af1e` `:Agent` node exists. |
| TP-007 | PASS | `write ok (labels_added=2, nodes_created=2, properties_set=9, relationships_created=1)` for `qa-selftest-e2e-4e24af1e`. |
| TP-008 | PASS | `agent='cobb'` MENTIONS-write → `write ok (labels_added=1, nodes_created=1, properties_set=1, relationships_created=1)`. |
| TP-009 | PASS | Count-and-decide read → `producedEdges=1, mentionEdges=1`. |
| TP-010 | PASS | `DELETE p` (`agent='cobb'`) → `write ok (relationships_deleted=1)`; re-read → `producedEdges=0, mentionEdges=1`, node still present. |
| TP-011 | PASS, with a self-caught execution note (Finding 3) | Recount before this pass showed `otherRemaining = 0 + 1 - 1 = 0` — the design's own last-edge branch (`docs/plans/kaizen-agent-ontology-graph.md` §4.3, mirrored in `skills/agent-maintenance/SKILL.md` §5 step 5) calls for the full `DETACH DELETE`, not the partial `DELETE m`. I initially ran the partial `DELETE m` (my own slip, not a system defect), which correctly executed and left a transiently orphaned zero-edge node; caught by the next read, then corrected by running the full clear, which removed it cleanly. Final state confirms the design's documented branch is correct and works as specified. |
| TP-012 | PASS | `_qa_selftest_producer_4e24af1e`, `_qa_selftest_producer2_4e24af1e`, `_qa_selftest_mentioned_4e24af1e` all still present as `:Agent` nodes after all node deletions — expected, not a leak (`:Agent` nodes are never deleted by design). `_qa_selftest_attacker_4e24af1e`/`_qa_selftest_forged_4e24af1e` never became `:Agent` nodes at all, since every write that would have created them was rejected. |
| TP-013 | PASS | Final `MATCH (n) RETURN labels(n), count(n)`: `['Agent'] | 4` (1 pre-existing real entry's producer, `architect`, + the 3 disposable ones from TP-012 — intentionally left, not cleaned), `['KaizenEntry'] | 27` — exactly the pre-run baseline (confirmed live before TP-001). Zero `entryId STARTS WITH 'qa-selftest'` rows remain. |

## Defects / findings

### Finding 1 (Medium, operational/testability — not a code regression in the M8 deliverable) — a long-running session's MCP connection silently keeps serving a stale, pre-rebuild `cypher-mcp` image

The task brief stated: "your own MCP connection (via `mcp__cypher__query` in this session) will
resolve to this new image automatically." **This was false for this run.** My first live call
(a plain, correctly-shaped producer-write) came back:

```
Rejected: this write is neither an author-write (no literal `author: '...'` found inside a CREATE
(...:KaizenEntry {...}) clause) nor the recognized curator-clear shape. This tool only authorizes
those two write shapes (FR-8).
```

This is the **pre-M8** rejection wording verbatim — the shipped `cypher-mcp/server.py` (confirmed
by direct `grep`) no longer contains this string anywhere; its current final-fallback message
enumerates all 6 shapes and reads "...nor a recognized curator shape...". Tracing it: this
`qa-engineer` subagent inherited its MCP connections from its parent Claude Code process (a
long-running `teco` session, PID 152005, started 2026-08-21 — confirmed via `ps -ef`/`pstree`),
whose `cypher-mcp/docker-run.sh`-spawned container (`docker run ... cypher-mcp:aa088de045e2`,
image built 2026-08-20, confirmed pre-M8 by `grep`) has been running continuously since before
today's rebuild and was never restarted. `docker ps` at the time of this pass showed **three**
separate `cypher-mcp` containers simultaneously live, on two different images — the just-rebuilt
`cb712173ab57` (12 minutes old) and two much older `aa088de045e2` instances (10h and 21h old) —
each tied to a different long-running Claude Code process's own stdio pipe. **A container rebuild
does not retroactively affect a process that already has its own container running**; nothing in
the tool's responses (no version/image-hash field, nothing in `TOOL_DESCRIPTION`) would let a
caller detect this without independently checking `docker ps`/`docker inspect`, as I did.

**Why this matters beyond my own run:** any other agent whose session predates a `cypher-mcp`
rebuild is silently writing against the old `authorize_write()` logic for the rest of that
session's lifetime — for this specific rebuild, that means such a session could not use the new
producer-write/MENTIONS/edge-resolve shapes at all (all would reject with the old two-shape
message), and would still carry the **pre-M8, not-yet-hardened** version of the cross-clause
chaining gap (Attack A's `CREATE`+`DETACH DELETE` shape) that this exact delivery closes in the
current image. This did not block my testing (I re-verified the actual current image directly,
below), but it is a live, real risk right now, independent of this task, for as long as those two
older containers keep running.

**Workaround used (documented, not silent):** the `cypher-mcp` project already carries the `mcp`
Python SDK in its own host venv (`cypher-mcp/.venv`). I wrote a small standalone MCP client
(`mcp.client.stdio` + `ClientSession`, calling the tool named `query` with the exact same
`graph`/`cypher`/`agent` parameters `mcp__cypher__query` exposes) that spawns its own
`docker run ... cypher-mcp:cb712173ab57` directly, bypassing my inherited stale connection
entirely. First read through this fresh connection (`MATCH (n) RETURN labels(n), count(n)`)
returned `['Agent'] | 1`, `['KaizenEntry'] | 27` — one real `:Agent {agentId:'architect'}` node
with a genuine `PRODUCED` edge to a real kaizen entry about FalkorDB's lack of an APOC-style
node-merge procedure, confirming the new producer-write shape is already in real use by at least
one other, unrelated live agent session against the correct image. All TP-001 through TP-013
results reported above were captured through this direct connection, not my inherited stale one.

**Recommendation:** (a) `cypher-mcp/README.md` should state explicitly that a rebuild only takes
effect for *new* MCP connections — an already-running Claude Code session (or subagent inheriting
one) keeps talking to whatever container it started with until that session/connection is
restarted; "resolves automatically" is not accurate and should not be assumed by anyone planning
a live-verification step around a rebuild, as this plan's S6 briefing did. (b) Consider exposing
a short image/version marker in the tool's responses (or a dedicated no-op diagnostic query) so a
caller can positively confirm which build it is actually talking to, rather than needing to shell
out to `docker ps`/`docker inspect` as this report did. (c) Worth a deliberate decision on whether
the two long-lived stale containers found here should be restarted now, given the residual
Attack-A-class exposure they carry independent of this task.

### Finding 2 (Low, informational — methodology, not a functional defect) — the review's literal Attack A/C reproduction text is not valid openCypher grammar on this FalkorDB build

`docs/reviews/kaizen-agent-ontology.md`'s Attack A and Attack C reproductions are shaped
`CREATE (...) MATCH (...) DETACH DELETE ...` / `CREATE (...) MATCH (...) SET/REMOVE ...` — an
updating clause (`CREATE`) directly followed by a reading clause (`MATCH`) with no intervening
`WITH`. Run verbatim against the live container, both come back not as an `authorize_write()`
rejection but as a genuine FalkorDB parser error: `"A WITH clause is required to introduce MATCH
after an updating clause."` — i.e. these exact statements never reach the authorization logic at
all on this engine; they are rejected at the Cypher-grammar level first. Attack B (`CREATE` →
`MERGE` → `CREATE`, no `MATCH` in the mix) has no such issue and reached `authorize_write()`
directly as written.

This does **not** indicate a gap: I re-ran grammar-valid equivalents (inserting a bridging
`WITH 1 AS _dummy` between the decoy `CREATE` and the following `MATCH`, changing nothing
semantically) for all three A/C variants, and all three reached `authorize_write()` and were
correctly rejected with the expected chaining message (see TP-002/004/005 above) — the closure
holds for the actual attack shape, grammar aside. It also does not undermine the offline pytest
suite's items 18/19/22/23: those call `authorize_write()` directly against a fake client, in-process,
with no real FalkorDB parser in the loop, so grammar validity there is irrelevant to what they
test. The finding is narrowly: **if these exact literal strings are ever reused as a *live*
(`pytest -m live` or manual) fixture against a real FalkorDB instance, they will need the same
`WITH`-bridging adjustment to actually exercise anything** — worth a one-line note wherever they
might get copy-pasted into a live context in the future.

## Coverage & gaps

Covered: the full closure (Attacks A, B, C-`SET`, C-`REMOVE`) against the actual current image;
one complete producer-write → MENTIONS-tag → count-and-decide → partial-resolve (`PRODUCED`) →
partial-resolve (`MENTIONS`) → full-node-clear cycle; confirmation that `:Agent` nodes persist by
design after their entries are gone.

Not covered here, by design (per the test plan's stated scope, complementing rather than
duplicating other gates): the offline pytest suite (`tdd-engineer`'s S1, already run); the 13
retargeted agent prompts' prose or `SKILL.md`'s prose itself (`cobb`'s S3/S4, doc-only,
spot-checked only to the extent §5's actual mutation logic was read to diagnose Finding 1's TP-011
note); performance/load; any chaining shape beyond the four keywords `_WRITE_KEYWORD_RE` already
enumerates (out of scope per the plan's own §6 residual-scope statement, re-affirmed by three
`analyst` review passes).

**Residual risk**, stated explicitly: Finding 1's two stale containers, for as long as they keep
running, are a live (if narrow) exposure to the pre-M8 chaining gap for whichever sessions remain
bound to them — this is an operational fact about the current moment, not something this report's
test items were scoped to remediate.

## Feedback & recommendations

- Finding 1's three recommendations above (README caveat on rebuild semantics, a
  version/image-hash surfaced to callers, and a decision on the two stale live containers).
- Finding 2's note: flag the `WITH`-bridging requirement wherever Attack A/C's literal text might
  get reused as a live fixture in the future.
- TP-011's self-caught note: `skills/agent-maintenance/SKILL.md` §5's read-then-decide branch is
  documented correctly, but is easy to get wrong under manual/procedural discipline alone (I did,
  briefly, mid-run) — worth considering, as a low-priority follow-up, whether `cobb`'s tooling
  could compute `otherRemaining` and select the branch programmatically rather than relying purely
  on a human/agent reading the count and choosing correctly every time.

## Verdict

**PASS.** S6's live, black-box confirmation holds: the deployed, rebuilt `cypher-mcp` image
(`cypher-mcp:cb712173ab57`) correctly rejects all three adversarial cross-clause attacks the
`analyst` review traced across its three passes, and one full real distillation dry-run against
disposable identities completed correctly end to end, with the graph left exactly at its pre-run
baseline. Two findings are recorded for awareness/follow-up (Finding 1, Medium — MCP connection
staleness; Finding 2, Low — informational grammar note); neither is a regression in the M8
deliverable itself.
