# Connected code graph P1 close-out CO-8: QUERY_MEM_CAPACITY security review

> **Status:** active · **Owner:** `security-expert` · **Tracks:** CO-8 (P1 close-out) · **Reviews:** commit `94382dc8` (`cpg/infra/ccg-falkordb.sh`, `cpg/infra/tests/test-ccg-falkordb.sh`, `cpg/README.md`, `cpg/infra/README.md`, `docs/plans/connected-code-graph-p1-raw-query-memory-graph.md`)

**Lens:** secrets/infra-hardening plus attack-surface judgement (advisory; `devops` decides and changes). No CPG applies (shell, config and docs; the engine query set was read directly).
**Verdict: approve with suggestions.** No blocker. The change is honest about its main limit, but one sentence in the README can still be read as protection, and the "restart" story has one gap worth closing. I implemented nothing.

## Probe (own throwaway only)

One throwaway, `ccg-falkordb-tw-16520` (script `--throwaway up`, so cap 256 MiB, 2g container, image `v4.18.11`), with a synthetic 100,000-node `:C` graph of 8 properties each. It was removed afterwards and its conf file with it. `ccg-falkordb`, `-tw-16390`, `-tw-16510` and `falkordb-dev` were not touched. Approval basis: the brief's pre-approval for the documented shape against my own throwaway.

| Observation | Result |
|---|---|
| `GRAPH.EXPLAIN` of the cross product (`MATCH (a:C),(b:C) WITH a,b LIMIT ...`) | plan contains `Cartesian Product` over two `Node By Label Scan` |
| `GRAPH.EXPLAIN` of `MATCH (a:C {line:1})` | no `Cartesian Product` |
| `GRAPH.EXPLAIN` of `UNWIND range(..) AS x UNWIND range(..) AS y RETURN count(*)` | `Unwind` x2, no `Cartesian Product` |
| Cross-product aggregate, one query, per-request timeout 1000 / 2000 / 4000 ms (cap 256 MiB) | timed out at 1.2 / 2.4 / 4.8 s; container peak about 290 / 555 / 963 MiB (about 225 MiB/s growth); memory released afterwards; not OOM-killed |
| Same shape, 3 concurrent queries, timeout 3000 ms each | **`OOMKilled=true`, container exited** |

So the cap at 256 MiB does not touch this shape (confirmed, matches devops). The wall-clock bound does work for a single query, but only because the kill needs about 9 s of allocation at this rate; the rate is host-specific and the bound is defeated by concurrency.

## Findings

### Major

**M1. The README sentence that reads "capped" can be taken as protection.** `cpg/README.md` (changed bullet): "`raw_query` memory is capped per query only partly ... a cross product feeding an aggregate is **not** bounded by it". The limit is stated, but the lead phrase ("is capped ... partly") and the README's own mention of the cap invite the inference "the instance is protected, a careless query is refused". Reality (confirmed): the one shape that matters for a careless or hostile author, an unfiltered two-pattern `MATCH`, is exactly the uncapped one, and the OOM kill is reachable by an ordinary role. Concurrent queries are not bounded either (probe above: three 3 s queries kill it; the plan note already says "not several concurrent ones" but the README bullet omits that). Suggested rewording: "`raw_query` can still kill the instance. `QUERY_MEM_CAPACITY` 256 MiB only refuses some single large-`collect`/sort shapes; a cross product feeding an aggregate, or several concurrent queries, are not bounded and run it into the 2g container cap (OOM kill drops graphs, ACL users, runtime config)". Put "can still kill the instance" first. The plan doc's "Does not bound" section is accurate and needs no change; the `cpg/infra/README.md` table row says "per query" without the caveat and links to the plan, which is acceptable.

**M2. Nothing in code or docs tells the reader that the drift check does not protect against a kill, only against a silent restart.** `status` drift and the `up` refusal (`module_config_ok` -> `die ... down --force, then up`) are correct and well tested (S-cases for 0, other value, unreadable; `U1c`). But after the OOM kill the container is `exited` (observed above; no restart policy), and `up` on an exited own container removes and recreates it, which re-applies the cap: so a kill cannot silently leave a cap-less running instance. That is good. The gap is the other restart path: **if `devops` adds a restart policy** (my control C2 below, or a Docker/WSL auto-restart), the container returns with `requirepass` (from the mounted conf, so it is not default-open) but with **no ACL users and the module config reset (cap off, TIMEOUT_MAX 10000 default)**, and nothing runs `up`/`create_users`. Clients get auth failures (loud, not insecure), but any later `GRAPH.CONFIG`-dependent assumption, and the cap, are silently off until someone runs `status`. Recommendation: if a restart policy is adopted, move the module config into the config file or a `--loadmodule` arg path (`FALKORDB_ARGS` is deliberately not passed today) so a restart keeps it, or leave restart policy off and keep "a kill is loud and manual". Treat as a design note for `devops`, not a defect of this diff.

### Minor

**m1. The restart default-open question: answered, no exposure.** Every start path writes `requirepass <admin pw>` via the mounted 600 conf (`write_conf`), so `default` is never passwordless, including after a recreate before `create_users` (the wait is on an authenticated PING; `apply_module_config` runs on that connection). The window before `create_users` is "no role users exist", which is fail-closed (no one but the admin password holder can connect), not default-open. Confirmed by reading `write_conf`/`run_container`; I did not kill the shared instance to observe it. One consequence: `create_users` rotates all four secrets, so a surviving MCP/CLI client holds a stale credential and fails with an auth error until it re-reads the at-rest credentials. Loud, fine.

**m2. `QUERY_MEM_CAPACITY` is module-wide, so it also applies to the build (`GRAPH.QUERY`) and verdict roles.** The measurements show the build under it with 2x headroom on `name_in` at the proving system's size, and the plan says the figure must be re-measured for a larger system. The failure mode for an engine read is a loud `StoreError`, but for an indexed repo whose content inflates `name_in` (many same-named symbols and call sites resolving by name to one chunk) a hostile repo author can make a given ask fail by content alone. This is bounded (one anchor's answers fail, no instance damage) and theoretical; I did not construct it. Worth one line in the plan doc as a known coupling between indexed content and engine availability.

**m3. Engine queries are not crashable by a crafted anchor in the way the question feared (confirmed by reading).** `cpg/ccg/engine/reader.py` queries are fixed text with `$ids`/`$path`/`$qname`/`$name`/`$key` parameters, chunked at `CHUNK = 1000`, depth bound 2 (`DEFAULT_DEPTH_BOUND`); an anchor never alters query text, so no anchor makes the engine emit a cross product or a larger query. Anchor influence is limited to which rows match (a very common `nameNorm` selects many symbols; the heaviest read, `name_in`, is the one under 2x headroom of the cap). Not exploitable to an OOM kill as far as I traced; the cap actually helps here, since `name_in` is an accounted-operator shape.

**m4. Lowering `TIMEOUT_MAX` is not free.** The engine pins `QUERY_TIMEOUT_MS = 10_000` and the engine role cannot read `TIMEOUT_MAX` (`cpg/ccg/engine/config.py` lines 14-16), so a server `TIMEOUT_MAX` below 10 s makes every engine ask fail with `RawRefused` unless that constant moves in lockstep, and `status` accepts any nonzero `TIMEOUT_MAX`, so the drift check would not flag a mismatch. See C3.

### Nit

n1. `QUERY_MEM_CAPACITY_UP` comment says "bounds ONE query, not several concurrent ones"; add "and not every operator" so the in-code note matches the plan.

## Attack surface (judgement)

- **Who can reach `raw_query`:** any process that can read the system's at-rest `raw` credential (`acl.read_credentials`, through the `ccg` MCP server's `raw_query` tool or the CLI `raw_query`) plus 127.0.0.1:6390; or anyone with the admin password (`~/.config/ccg/falkordb-admin.pass`, mode 600, same Unix user). Any agent that is granted the `ccg` MCP server has the tool, and `.mcp.json` wiring makes that every Claude Code session in this repo. The server forwards caller-supplied `cypher` and `timeoutMs` (capped by `TIMEOUT_MAX`). No input filtering exists on the text; the raw role is read-only by ACL (`+graph.ro_query +graph.explain +info`, keys `~ccg_<s>`, `~ccg_<s>__ov_*`), which does not limit memory.
- **Hostile or merely careless?** Both. No skill is needed: the 8-second kill is the plain two-pattern `MATCH` plus `collect`, or equally an accidental `MATCH (a),(b)` aggregate. A prompt-injected agent can issue it as one tool call. Confirmed live (probe above) that three concurrent 3 s queries also kill it, so even a timeout lowered to a few seconds does not remove the vector.
- **Cost of the kill:** the container exits (no restart policy), the graphs, ACL users and runtime config are gone; recovery is `up`, `create_users` (rotates secrets), `refresh` (a rebuild of about 30 minutes at the proving system's size per the plan: 1,908 to 2,014 s). Availability and time cost, no confidentiality or integrity consequence (see m1). The OOM stays inside the 2g container (`--memory-swap` equal), so it does not reach `falkordb-dev` or the host, which the 2g cap and no-swap choice were made for.
- **Diff widens nothing:** no new published port (still `127.0.0.1:6390`), no env or credential change, no new file or permission (the conf stays 600, no volume), no egress. The only added Redis write is `GRAPH.CONFIG SET QUERY_MEM_CAPACITY` on an instance the run just started, behind the ownership label gate; throwaways inherit it (confirmed: the probe instance reported 268435456 on start).

## Recommended cheap controls (not implemented)

Ranked by value for effort:

- **C1. Plan-based refusal in the client, before running.** `raw_query` already holds a role with `+graph.explain`. Call `GRAPH.EXPLAIN` first and refuse when the plan has `Cartesian Product` (and, if wanted, two or more `Unwind` operators). This is robust to query-text spelling (`MATCH (a:C),(b:C)`, two `MATCH` clauses, aliases), unlike a text filter, which is bypassable by whitespace, comments, reordering or an `UNWIND` product. Limits: it is a client-side guard, so a direct redis connection with the credential bypasses it; and it blocks legitimate cross products and multi-`UNWIND` queries (add an escape such as requiring a `LIMIT` or a smaller `timeoutMs`). Treat a text refusal as a tripwire only.
- **C2. Concurrency and timeout, together.** Observed rate about 225 MiB/s per query on this host, so a lone query needs 9 s to reach 2 GiB; three queries need 3 s. Lowering `TIMEOUT_MAX`/client `timeoutMs` to about 4 s keeps a single cross product near 1 GiB, and `THREAD_COUNT`/`MAX_QUEUED_QUERIES` (default 8 / 25) bound concurrency (a `THREAD_COUNT` of 2 caps concurrent read queries at 2, so about 2 x 4 s x 225 = 1.8 GiB, still marginal). That is arithmetic from one host, not a guarantee, and `THREAD_COUNT` is a module load argument (restart, not `GRAPH.CONFIG SET`), which collides with M2's reset-on-restart. Cost: lockstep change of `config.QUERY_TIMEOUT_MS` (m4) and re-measure of engine/build query times. A lower container cap (for example 1g) does not help; the shape reaches any cap, and the build/engine need the headroom (`maxmemory` 1536mb). A restart policy only shortens the outage and has the M2 problem.
- **C3. Make drift check the engine/server timeout pair.** `status` could assert `TIMEOUT_MAX >= config.QUERY_TIMEOUT_MS` so a lowered value cannot silently break the engine (m4).
- **C4. Config knobs verified in `v4.18.11` (`GRAPH.CONFIG GET *`):** `TIMEOUT_MAX`, `MAX_QUEUED_QUERIES` (25), `RESULTSET_SIZE` (10,000), `QUERY_MEM_CAPACITY`, `THREAD_COUNT`/`OMP_THREAD_COUNT` (8), `VKEY_MAX_ENTITY_COUNT`, `DELTA_MAX_PENDING_CHANGES`. None bounds an intermediate result: `RESULTSET_SIZE` trims the reply only (and the aggregate here returns one row), `MAX_QUEUED_QUERIES` bounds the queue only. There is no knob that closes the cross-product aggregate; that is the engine's accounting gap devops measured. A bug report upstream with the 8-property reproducer is cheap and could help.
- **P4 per-user limits would add:** a per-role memory/rate bound (ACL has no memory dimension today), per-role concurrency, and an audit of who issued the killing query (today none is logged; the kill is attributable only by timing). Until then C1 is the only control that stops a careless query before it runs.

## What is solid

- The doc set is candid: the plan note's "Does not bound" section states the OOM with exact measurements and no overclaim, and the commit message says "NOT a hard limit".
- `up` ordering: the cap is applied right after a container this run started; a running instance with a lost cap makes `up` fail, not "fix it silently" (`module_config_ok` -> `die`), and `status` adds drift with unit cases for off, wrong and unreadable values. A cap lost to a restart cannot be silently ignored.
- No widening of binding, credentials, permissions or egress; ownership-label gating unchanged.
- Measured headroom on the engine and build paths, with the method stated and the "re-measure at a larger size" caveat.

## Open questions

1. (`devops`) Is a restart policy wanted? If so, M2 needs the module config to survive restarts first.
2. (stakeholder, scope) The accepted P1 limit leaves a prompt-injected agent a one-call way to take the store down for about 30 minutes. Is C1 (plan check before running) worth a P1 close-out item, or does it wait for P4?
3. (`devops`) Should `status` also assert `TIMEOUT_MAX >= engine QUERY_TIMEOUT_MS` (C3)?
