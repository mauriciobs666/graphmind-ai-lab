# Kaizen — Learnings Inbox: joern

> Append-only capture of durable, non-obvious environment facts the `joern` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

## 2026-07-18 — Joern distribution is NOT installed on this box despite the pinned-path assumption
- **Evidence:** `find / -maxdepth 6 -name joern-parse` and searches of `$HOME`, `/opt`, `/usr/local`, snap/apt, SDKMAN all empty; `$HOME/joern/joern-cli/` does not exist. `java -version` → 21.0.11 (present). This contradicts M1's HISTORY (`docs/HISTORY.md`, 2026-07-17 "live-load verified against Joern v4.0.579") — the distribution was present during M1 and is gone now. Disk is tight: `df` shows 7.1G free / 93% used on `/`.
- **Context:** M2 substrate task — asked to build+load a CPG; fully blocked at stage 1 (build) with no Joern binary. Per boundary, provisioning Joern is devops's, so escalated rather than reinstalling.
- **Suggested home:** project docs (a note in the CPG component docs that Joern install is not persistent/guaranteed and must be verified before a run) + possibly prompt (reinforce the pre-flight `joern-env.sh` check as a hard gate).

## 2026-07-18 — FalkorDB start script pulls & runs falkordb/falkordb:v4.18.11 cleanly; module ver 41811
- **Evidence:** `falkor-chat/scripts/start_falkordb.sh -d` pulled `v4.18.11` (image cached only `:edge` before), container `falkordb-dev`, ports 6379 + web 3000, volume `falkordb-data`. `redis-cli -p 6379 module list` → `graph ver 41811 ... TIMEOUT 1000 RESULTSET_SIZE 10000`, plus `vectorset ver 1`. `redis_version:8.6.3`.
- **Context:** Bringing up the DB half of the M2 CPG substrate while the Joern build was blocked.
- **Suggested home:** knowledge base (confirms the skill's FalkorDB prereq command + pinned module version for the CPG→FalkorDB path).

## 2026-07-19 — The skill loader (cpg-to-falkordb.py) breaks on real repos: batched statement exceeds redis-cli argv limit
- **Evidence:** `pipeline.sh ... --load` on falkor-chat/server (79,581 nodes / 522,182 edges) crashed at stage 4 with `OSError: [Errno 7] Argument list too long: 'redis-cli'`. The loader passes each UNWIND-batched statement as a single `redis-cli GRAPH.QUERY <graph> "<query>"` argv; with `--batch 500` and large `CODE` properties a node batch reached 175–215 KB, over Linux `MAX_ARG_STRLEN` (128 KiB per single argv). Statement-length check: `awk '{print length}'` on load.cypher showed max 215,219 bytes; statements 2–4 all >128 KiB. Note pipeline.sh reported exit 0 despite the crash (set -euo pipefail did not propagate the python failure through the final echo).
- **Fix that worked:** feed each statement to redis-cli via STDIN with `-x` (`printf '%s' "$stmt" | redis-cli -p 6379 -x GRAPH.QUERY <graph>`), which bypasses argv limits entirely. Loaded all 1224 statements, 0 failures, counts matched export exactly. Durable options for the skill: (a) switch the loader to `-x`/stdin, or (b) lower `--batch` (~25) — but stdin is the robust fix since a single huge CODE node could still blow a tiny batch.
- **Suggested home:** project docs (skill fix — cpg-to-falkordb.py) + knowledge base

## 2026-07-19 — pysrc2cpg call-graph (:CALL METHOD->METHOD) is sparse/unreliable; use CONTAINS->CALL node + METHOD_FULL_NAME instead
- **Evidence:** `MATCH (m:METHOD {NAME:'post_message'})-[:CALL]->(callee)` returned nothing; the METHOD->METHOD call graph is largely absent. Call sites are recoverable via `(:METHOD)-[:CONTAINS]->(:CALL)` where the CALL node carries `CODE` and a `METHOD_FULL_NAME` best-effort resolution. That resolution is inconsistent for the SAME callee: prod call sites resolved `services.post_message(...)` to `Services.post_message` (short) while test call sites resolved to the full `falkorchat/services.py:<module>.Services.post_message` or to a phantom external `Services.__init__.<returnValue>.post_message`. pysrc2cpg also emits an IS_EXTERNAL=true stub METHOD per unresolved attribute call (236 external methods total). Any reachability recipe over a Python CPG must traverse CONTAINS->CALL and match on NAME/CODE, not rely on :CALL edges.
- **Suggested home:** knowledge base (CPGQL/pysrc2cpg gotchas) + references/cpg-model.md

## 2026-07-19 — Framework-invoked entrypoints (FastAPI routes, MCP tools) are not statically linked to their tests; "test-gap" needs transitive reachability, not direct-call counts
- **Evidence:** In falkor-chat/server, test_api.py drives routes over FastAPI TestClient (HTTP), so NO static CALL edge exists from any test_* to `api.py:<module>.build_router.*` route handlers — every route handler and mcp tool reads as prod-only in the CPG. Meanwhile private helpers (e.g. services._serialize_opaque) show 0 DIRECT test callers yet ARE transitively exercised because tests call their public encloser directly (publish_workflow_def, called by test_services.py). So direct-caller = 0 does NOT mean untested. A valid test-gap recipe must compute transitive reachability from two seed sets and diff them.
- **Suggested home:** project docs (cpg-analysis recipe design, for graph-dba) + knowledge base
