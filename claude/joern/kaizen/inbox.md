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

## 2026-07-17 — On Joern v4.0.579, parse Python with `--language pythonsrc`, not `python`
- **Evidence:** `joern-parse --language python <dir>` fails: `java.lang.AssertionError: assertion failed: CPG generator does not exist at: /home/mauricio/joern/joern-cli/py2cpg.sh` (legacy `PyCpgGenerator` shells out to a `py2cpg.sh` not shipped in v4). `joern-parse --list-languages` shows both `python` and `pythonsrc`; only `pythonsrc` (the `pysrc2cpg` frontend, present under `joern-cli/frontends/pysrc2cpg`) works. Re-run with `JOERN_LANGUAGE=pythonsrc` built + exported cleanly.
- **Context:** Building CPGs for the repo's two Python apps (falkor-chat/server/falkorchat, salesperson).
- **Suggested home:** project docs (skill SKILL.md — its `JOERN_LANGUAGE=python forces a frontend` hint is misleading for Python; should say `pythonsrc`) + knowledge base

## 2026-07-17 — skill `pipeline.sh` reports EXIT=0 even when the parse frontend fails
- **Evidence:** Both `pipeline.sh … --load` runs with the broken `--language python` printed the Java stacktrace but the background task completed "exit code 0"; the real failure was only visible in the captured log. Error not propagated from the build stage.
- **Context:** Same CPG build run; the misleading exit code masked the frontend failure until the log was read.
- **Suggested home:** project docs (skill scripts — propagate stage failure)

## 2026-07-17 — skill loader's per-statement `redis-cli` load fails two ways at scale; a single persistent RESP connection fixes both
- **Evidence:** `cpg-to-falkordb.py --load` spawns one `redis-cli` process per batched statement. At `--batch 500`: `OSError: [Errno 7] Argument list too long: 'redis-cli'` (Linux `MAX_ARG_STRLEN` = 128KB per single argv string; a 500-node UNWIND CREATE with inlined CODE exceeds it). At `--batch 50`: intermittent `Error: Connection reset by peer` after ~2000 statements (thousands of short-lived TCP connections — a connection storm; server logs showed no crash/OOM, clean BGSAVE). A ~60-line custom loader streaming the same `load.cypher` (one statement/line, longest ~20KB) over ONE socket via RESP loaded 4319/4319 and 2691/2691 statements, 0 failures; FalkorDB counts matched the transformer totals exactly (29447/185517, 17549/116005).
- **Context:** Loading the two Python-app CPGs into FalkorDB (cpg_falkorchat, cpg_salesperson).
- **Suggested home:** project docs (skill — make the `--load` path use a persistent connection / pipe the artifact instead of per-statement `redis-cli`) + knowledge base

## 2026-07-17 — the three items above are now FIXED in-skill (`skills/joern-cpg/`); cobb, reconcile rather than re-file
- **Evidence:** (1) `cpg-to-falkordb.py --load` now streams over ONE persistent socket via a built-in RESP loader (`load_statements()` + `_resp_encode`/`_resp_read_reply`), replacing the per-statement `redis-cli` spawn — verified: single fresh load of the 4319-stmt falkorchat artifact → 4319/4319 ok, graph exactly 29447 nodes / 185517 edges / 29447 distinct ids. (2) `pipeline.sh` gained `--language` (passthrough to `JOERN_LANGUAGE`), `--reset` (guard-gated `GRAPH.DELETE` before `--load`), a post-transform guard that FAILS if the export produced no `nodes_*_data.csv` (catches the exit-0-on-frontend-failure case), and a post-load node/edge verify. (3) SKILL.md corrected: the `JOERN_LANGUAGE=python` hint now says Python→`pythonsrc`, plus new Gotchas entries for the frontend token and the persistent-connection load. Personal one-off `~/cpg-work/rebuild.sh` deleted (superseded by the generic `pipeline.sh`); `~/cpg-work/load_cypher.py` kept only as a standalone `.cypher` replayer.
- **Gotcha surfaced while verifying:** counting FalkorDB results with `redis-cli GRAPH.QUERY … --no-raw | grep -oE '[0-9]+' | tail -1` is WRONG — it grabs digits from the `Query internal execution time: 0.08 milliseconds` stat line, producing phantom huge counts (saw a real 29447 read back as 273336/365549). Correct extraction: `awk '/^[0-9]+$/{last=$0} END{print last}'` (the standalone integer result row). Applies to any scripted count check, not just this skill.
- **Context:** Hardening the `joern-cpg` skill into a generic, robust rebuild path at the user's request.
- **Suggested home:** knowledge base (the count-extraction gotcha is reusable across all FalkorDB scripting) + note the three fixes as resolved

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

## 2026-07-25 — CPG `FILENAME` is relative to the **parse root**, so the parse root silently decides whether every `STARTS WITH 'tests/'` recipe filter works
- **Evidence:** the previous `cpg_falkorchat` was built with `falkor-chat/server/falkorchat` as the parse root and its METHOD `FILENAME`s are bare basenames (`services.py`, `api.py`) — every `cpg-analysis` filter of the form `m.FILENAME STARTS WITH 'tests/'` / `'falkorchat/'` matched nothing (inert, not erroring). Rebuilding from a staged root containing exactly `{falkorchat, tests}` yielded `falkorchat/services.py`, `tests/test_api.py`, … (42 distinct FILENAMEs, verified in the load.cypher artifact before loading and again in FalkorDB). Joern emits paths relative to the directory handed to `joern-parse`, and `cpg-to-falkordb.py` never rewrites `FILENAME`.
- **Consequence worth stating:** a "correct" CPG can be *silently useless* to the analysis recipes purely because of the parse root; and the defect is invisible from node/edge counts. Check `MATCH (m:METHOD) RETURN DISTINCT m.FILENAME LIMIT 10` right after any load.
- **Context:** S8 of `docs/plans/cpg-query-access.md` — rebuilding `cpg_falkorchat` from `{falkorchat,tests}`.
- **Suggested home:** project docs (`skills/joern-cpg/SKILL.md` gotcha + `references/cpg-model.md` on `FILENAME`) + knowledge base

## 2026-07-25 — `build-cpg.sh`/`pipeline.sh` have no exclusion mechanism: scoping a parse means staging a copy
- **Evidence:** `pipeline.sh` passes `<source>` to `build-cpg.sh` → `joern-parse "$SRC"` verbatim; there is no `--exclude`/ignore option anywhere in the skill's scripts. `falkor-chat/server/` holds 41 first-party `.py` files next to `.venv/` (1,808 `.py`), `.pytest_cache/`, `.ruff_cache/`, `falkorchat.egg-info/` — parsing it directly would drown the CPG in dependency code. Working procedure: `cp -r` the wanted subtrees into a scratch root, `find … -name __pycache__ -prune -exec rm -rf {} +`, parse that. This is also the only lever for the `FILENAME`-prefix issue above, so the two are one decision.
- **Context:** same S8 run.
- **Suggested home:** project docs (`skills/joern-cpg/SKILL.md` — a "scoping the parse" note next to the pipeline usage)

## 2026-07-25 — `cpg-to-falkordb.py --load` re-transforms the export; it has no "replay this .cypher" mode
- **Evidence:** the parse-first ordering (pipeline without `--load` → explicit `GRAPH.DELETE` → `cpg-to-falkordb.py <export-dir> --load`) re-reads all 94,818 export CSVs and rewrites `load.cypher` before streaming it (log: `110048 nodes, 734929 edges -> 1711 Cypher statements written … loaded into 'cpg_falkorchat' (1711/1711 ok, 0 failed)`). Cost is seconds, not a re-parse, so the ordering is cheap — but the artifact is regenerated, and there is no flag that consumes an existing `.cypher`. A standalone replayer has to stream it over one socket by hand (per-statement `redis-cli` still hits the 128 KB argv limit).
- **Context:** S8 used the parse-first variant so a failed parse could not leave the graph deleted.
- **Suggested home:** project docs (`skills/joern-cpg/SKILL.md` — note the `--load` path re-transforms) | knowledge base

## 2026-07-25 — sizing data point: 41 Python files → 110k nodes / 735k edges / 433 MB neo4jcsv export
- **Evidence:** staged `{falkorchat, tests}` (41 `.py`, ~1 MB of source) parsed with `pythonsrc` produced `cpg.bin` 4.5 MB, a neo4jcsv export of **94,818 files / 433 MB** (nested per method), a **50 MB** `load.cypher` of 1,711 UNWIND-batched statements, and a FalkorDB graph of **110,048 nodes / 734,929 edges** (distinct `id` = 110,048, i.e. dedup lossless). Roughly **2,700 nodes and 18,000 edges per source file** for Python with default overlays. Whole run (parse → export → transform → load) completed inside a single 10-minute foreground tool call on a 5 GB-RAM WSL2 box.
- **Why it matters:** the export's *inode* count (94.8k files) is the surprising cost, not its bytes; and a repo 10× this size projects to ~1.1 M nodes / 7.3 M edges in FalkorDB RAM — a `graph-dba` sizing conversation, not a footnote.
- **Suggested home:** knowledge base (CPG scale rules of thumb)
