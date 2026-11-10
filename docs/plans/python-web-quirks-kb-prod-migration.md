# python-web-quirks KB: promote the 4b embedder to production `ws:agent-team`

> **Status:** active · **Owner:** `architect` · **Tracks:** — (W7 of `docs/plans/python-web-quirks-kb-path-ml.md` §6) · **Reviews:** `docs/reviews/python-web-quirks-kb-prod-migration.md`

Plan for the stakeholder's approval. Nothing on `:8200` changes until the approvals in §8 are given. Figures were observed read-only unless marked "estimate" or "not observed".

## 1. Goal, scope, recommendation, what the evidence supports

**Goal.** Move production `ws:agent-team` (served by `:8200`) from `text-embedding-qwen3-embedding-0.6b` (1024-dim, skill floor 0.43) to `text-embedding-qwen3-embedding-4b` (2560-dim, floor 0.42), the configuration that passed sealed Set B3 on staging.

**Out of scope.** Any workspace other than `ws:agent-team`; the server default embedder; chunking; deleting `ws:pwq-test` / stopping `:8210`; re-scoring B, B2 or B3 (spent).

**Recommendation: Option A (in-place per-workspace migration), behind a read-only pre-gate (§4 step 0) that can stop it before any outage.** Not promoting (C1) is the fallback.

**What the stakeholder is approving, stated narrowly (M4).** B3 passed R1 on staging at 115/126 against a bar of 114 (one query); raw recall without the floor was 123/126, the floor removed 8 (`docs/test-reports/python-web-quirks-kb-report.md` W5). B3 was never scored on the 0.6b, so there is **no paired 0.6b/4b result on one sealed set** (the 0.6b's 110/126 is B2, a different set). The 4b was the best of many Set A looks (`docs/plans/python-web-quirks-kb-path-ml.md` §7), and the floor 0.42 sits 0.002 under the closest staging near-negative (0.422). The post-migration checks (Set A, saturated and not blind; the 45 pairs, n=32 applicable, Wilson 95% CI [0.719, 0.950] at the 0.6b) support **"no gross regression and no loss of a pair that was found before"**, not "no regression in production". A claim about production quality needs a new sealed set on the post-migration corpus (**B4**): optional follow-up unit with its own authoring/scoring and its own approval (§8), schedulable after cutover because rollback R1 stays available. It is not a go gate: the change is reversible and 465 of staging's 518 documents are shared with production.

## 2. Findings that shape the design

- **One server process serves exactly one workspace.** `/proc/1268/environ`: `FALKORCHAT_WS_ID=agent-team`, `FALKORCHAT_EMBEDDING_DIM=1024`, `AGENT_TEAM_PORT=8200`, `FALKORCHAT_OPENCODE_CONFIG=~/.config/opencode/opencode.local.json`, no `FALKORCHAT_MODEL_CONFIG`; `config.py:16`; `docs/plans/embedding-migration.md` §2.6. Only two falkorchat uvicorns run (pid 1268 `:8200`, pid 3487158 `:8210`). Stopping `:8200` takes down only the agent-team KB front door; `ws:demo`/`salesperson`/`acme`/`eval`/`nlq-eval` are separate processes or none (the embedder note's §3(a) "every workspace on that process" is wrong here).
- **Dimension is per model and per workspace.** Index dimension is DDL on each `ws:{id}`; the checked dimension is the overlay's `models.<ref>.dim`, else `FALKORCHAT_EMBEDDING_DIM` (`SERVER.md` §1.8; `embedding.py:182-197`). `ws:agent-team` Chunk index is 1024; `ws:pwq-test` is 2560 in the same FalkorDB. `models.pwq-staging.json` differs from `config/models.json` by exactly the 4b `dim: 2560` entry. Pins: `ws:agent-team` and `ws:demo` observed 0.6b; `acme`/`eval`/`nlq-eval` follow the default (embedder note §3); `k069probe` pinned.
- **`ws:agent-team` today:** 505 `Document` (14 `failed`, 0 `processing`; the brief's 17/1 is stale, re-read at execution), 884 `Chunk`, 871 embedded, 0 `Message`. The 13 unembedded chunks are in 12 of the 14 failed documents (`pendingJobs` 0 on all). `Chunk.embeddingModel` absent. `pwq_kb.py gate` against `:8200`: `gate: clean`.
- **Migration mechanics** (`falkor-chat/scripts/migrate_embeddings.sh`, `DESIGN.md` §7.1): server stopped; re-embeds Message then Chunk rows whose `embeddingModel` differs from the target (including the null-embedding ones); count-check, then drop/recreate both vector indexes, override set last; restart required. Resumable. Staging: 902 chunks in 75.0 s (the embedder note's earlier 114 s is a different, uncontrolled run; only 75.0 s is used here).
- **The start lock and the autostart hook (M1, verified).** `claude/teco/hooks/ensure-environment.sh` (runs at any teco session start and teco's step-1 probe) relaunches `scripts/start_agent_team.sh` detached, under `flock -n "$SERVER_LOCK"` with `SERVER_LOCK=$STATE_DIR/agent-team-8200.lock` = `~/.local/state/graphmind-ai-lab/agent-team-8200.lock`, whenever `:8200/health` refuses the connection; it `env -i`-launches with no `EMBEDDING_DIM`, so the script's default 1024 applies (`start_agent_team.sh:137`), and the script bootstraps at that dimension. The lock is held for the **running server's lifetime**: `/proc/1268/fd/3` is that lock file, parent pid 1261 is `flock -n <lock> env -i ... start_agent_team.sh`. So the lock frees when `:8200` is stopped, and from that instant the hook may start a 1024-dim server against a graph mid-migration, and again after cutover if the default stays 1024. The hook treats a held lock as "already being started" and launches nothing.
- **MCP does not reconnect by itself** (M3; `claude/teco/coordination-techniques.md`: the `falkor-chat-agent-team` MCP gives up after its ~7 s initial-connect window; the user runs `/mcp` -> `falkor-chat-agent-team` -> Reconnect). Captures attempted during the outage fail and are not retried.
- **`GRAPH.COPY` carries vector indexes and embeddings** (`docs/plans/python-web-quirks-kb-graph.md` §1, "Verdict: yes"); it can fail transiently (`could not fork`); retry rule there: check `GRAPH.LIST`, retry once, never query the destination key between attempts.
- **Host.** `free -m` (WSL) total 5930 MB, free 686, available 3291 (a later read by the reviewer: 562 free, 3323 available); FalkorDB 478.34 MB; RDB only (`aof_enabled:0`). The 4b loads in LM Studio, footprint not observed.
- **Staging is not production:** 4 production documents staging lacks, 17 deleted S7 captures staging holds (excluded at B3), captures written since. Hence step 6 re-measures on production.

CPG: considered, not relevant — operational migration plan; the code facts used (migration script, start script, hook) are small shell/Python files read directly.

## 3. Options

| | Description | Verdict |
|---|---|---|
| **A** | Stop `:8200`; `migrate_embeddings.sh agent-team <4b>`; restart with 2560 declared; other workspaces untouched via pins. | **Recommended** |
| A2 | Change the server default embedder to the 4b. | Rejected: `acme`/`eval`/`nlq-eval` follow the default and would embed 2560-dim into 1024-dim indexes (`EmbeddingDimensionError`); pinned `agent-team` would not move. |
| B | Second instance on another port for the 4b workspace; repoint agents. | Rejected: in this deployment every workspace is already its own process, so B = A plus a new port; repoint `.mcp.json` (`:14`), `pwq_kb.py` `TARGETS` (`:558`), the autostart hook, `start_agent_team.sh` and docs naming `:8200`; two KB copies drift unless cut over at one instant, which is A's outage. |
| B' | Migrate a `GRAPH.COPY` on a side port, swap by `RENAME`. | Rejected: agents write to the live graph continuously, so a delta merge or write freeze is needed. |
| C1 | Do not promote. | Valid fallback; production keeps the AC-3 miss (B 73/84, B2 110/126 at the 0.6b). |
| C2 | Repoint agents to `:8210` (`ws:pwq-test`). | Rejected: stale copy, carries deleted captures, a test workspace becomes the team store. |

**Deciding trade-off:** A changes one workspace's state with a bounded single-door outage; B changes consumer configuration across all agents and keeps two stores.

## 4. Runbook (Option A)

Owners: `devops` (processes, LM Studio, lock, backup export), `graph-dba` (migration, copy, indexes), `tdd-engineer` (fixture/runner unit), `data-scientist` (decision rule), `qa-engineer` (gates), `coder` (config/script edits), `cobb` (skill), `teco` (coordination, record). Approvals in §8 precede step 1.

**Step 0a: unit `W7a`, the 45-pair fixture and runner (owner `tdd-engineer`; M2).** The 45 pairs exist only as a markdown table (`claude/docs/plans/agent-knowledge-base-strategy-ml.md` "Stage 8 Phase 1"; results `claude/docs/test-reports/agent-knowledge-base-strategy-ac2-report.md`); no runnable set exists. Deliver a committed JSON fixture (query situation text, expected `documentId`, answer class, negatives) plus a small runner beside `claude/cobb/scripts/pwq_kb.py` (document-level top-5, the skill's literal prefix, `limit=5`, scores kept, floors 0.43 and 0.42 reported, closest-negative and R6-document scores) with selftest and mutation tests per repo convention; `analyst` reviews before step 0b. The 48 expected ids in that table exist in both `ws:agent-team` and `ws:pwq-test` (48 found, 48 `ready`, checked in this revision), so no pair fails to map.

**Step 0b: the pre-registered paired rule (owner `data-scientist`; unit `W7b`, note `docs/plans/python-web-quirks-kb-prod-migration-ml.md`).** Written and frozen **before the first run**: a paired table (pairs found at 0.6b/0.43 and lost at 4b/0.42, and the reverse), the numeric stop threshold, how R6-type floor-unstable rows and the 6 negatives count. This plan fixes no number: at n=32 one pair is about 3 points and the CI is about +-0.12, so "no regression" is not a rule.

**Step 0c: pre-gate (read-only, no outage; `qa-engineer`).** Run the fixture on `:8200` (same-day 0.6b baseline; the corpus changed since 2026-09-19) and on `:8210` (4b), plus the §7 non-pwq smoke list, and apply the W7b rule. A stop here ends the plan; any row between 0.42 and 0.43 goes to the stakeholder. Results into a `docs/test-reports/` addendum.

**Step 1: pre-checks (`devops`, `graph-dba`).**
1. No storefront/demo running; no `ingest_document` batch in flight; **no teco session start and no `ensure-environment.sh` run during steps 3-5** (it is the relauncher, §2); announce the slot (open Claude sessions cannot be messaged; §8).
2. LM Studio reachable, 4b present, one probe embedding returns 2560 values. Acceptance: probe under 5 s warm; a cold load must finish inside `models.json` `timeouts.embedding` (30 s) or `devops` pre-loads the 4b before step 3 (the same bound is re-checked after cutover in step 6).
3. Before-record: `pwq_kb.py gate`, document counts by status, chunk counts. **Record the 14 failed `documentId`s and the 13 unembedded `chunkId`s** into the ledger/report for `falkor-chat/docs/BACKLOG.md` K-102 (the migration erases that reproduction state).
4. `RDB`: `redis-cli BGSAVE`, wait for `rdb_last_save_time` to change (the other backups are in step 3).
5. Verify the lock semantics: `flock -n -E 75 ~/.local/state/graphmind-ai-lab/agent-team-8200.lock true` must return 75 while pid 1268 runs (the hook's own `lock_held` test); after step 3's lock is taken, `ensure-environment.sh </dev/null` must print "already being started" and launch nothing (`devops` runs it once, from a non-teco shell).
6. Brief the coordinator (`teco`) that the lock path is the relaunch guard and that the `start_agent_team.sh` default becomes 2560 (step 2).

**Step 2: repo edits (`coder`).** (a) `falkor-chat/config/models.json`: add `"lmstudio/text-embedding-qwen3-embedding-4b": {"dim": 2560}` (default embedder stays the 0.6b; the staging overlay is left alone). (b) `falkor-chat/scripts/start_agent_team.sh`: **default `EMBEDDING_DIM` becomes 2560** (line 137) and the header notes the 4b dim is declared in `models.json`. This is part of cutover, not an option: the hook's `env -i` launch carries no `EMBEDDING_DIM`, so the default is the only protection against a relaunch bootstrapping a 1024 index (the migration, `embedding.py` and `bootstrap_schema.sh` "already indexed" behaviour are secondary: what FalkorDB does with a differing-dimension `CREATE VECTOR INDEX` is not observed). Rollback R1 reverts it to 1024. (c) Run `falkor-chat/server/tests/test_ensure_demo_models_script.py` and `falkor-chat/server/tests/test_opencode_config_resolution.py` (they read `models.json` for `defaults.embedding` only; inferred, not run) and a `ModelGateway` resolution of the 4b ref from the server venv. The running `:8200` ignores both edits until restart.

**Step 3: outage begins; take the relaunch guard and the backups (`devops`, `graph-dba`).** Confirm pid 1268's environ shows `agent-team`. Stop it (`kill 1268`; the parent `flock` exits and frees the lock) and **within the same shell command** take the lock with `flock -w 10 ~/.local/state/graphmind-ai-lab/agent-team-8200.lock <wrapper>` (the wrapper of step 4), so the hook reports "already being started" for the rest of the window; the seconds-wide gap between the kill and the lock is covered by step 1.1. Then, with the API down so nothing is lost between copy and migration (m1): `GRAPH.LIST` (destination absent), `GRAPH.COPY ws:agent-team ws:agent-team-pre4b` (retry rule §2; verify node/edge counts equal and Chunk vector index `OPERATIONAL` at 1024 on the copy; never serve it), and the logical export (every `Document` id, title, status, verbatim text via read-only Cypher into a file outside the repo; location recorded). Free RAM check first (`free -m`, FalkorDB `used_memory`).

**Step 4-5: migrate and restart under one lock (`graph-dba` + `devops`).** One detached wrapper holds the lock through migration and exec's the server, logging to `~/.local/state/graphmind-ai-lab/agent-team-8200.log`:
`setsid nohup flock -w 10 <lock> bash -c '(cd falkor-chat && FALKORCHAT_OPENCODE_CONFIG=$HOME/.config/opencode/opencode.local.json ./scripts/migrate_embeddings.sh agent-team lmstudio/text-embedding-qwen3-embedding-4b --i-have-stopped-traffic); rc=$?; [ $rc -eq 0 ] || exec sleep infinity; exec env -i <hook's kept env> FALKORCHAT_WS_ID=agent-team FALKORDB_HOST=127.0.0.1 FALKORDB_PORT=6379 AGENT_TEAM_PORT=8200 falkor-chat/scripts/start_agent_team.sh' >>log 2>&1 &`
(the `env -i` set mirrors the hook, so the supervised state equals what the hook would produce; on a migrate failure the wrapper keeps the lock until `devops` kills it, so nothing relaunches over a half-migrated graph; `exec` keeps the lock on the server for its lifetime, as today). Expect `Chunk: migrated 884/884`, `Message: migrated 0/0`. `graph-dba` verifies read-only: both vector indexes `dimension: 2560`; 0 chunks without `embedding`; all `embeddingModel` = the 4b ref; override = the 4b ref; counts equal step 1.3; `ws:demo` override still the 0.6b. Fallback if the script's bootstrap step fails on the existing index: the direct `uvicorn falkorchat.app:app --port 8200 --log-config config/logging.json` command from `server/` with `FALKORCHAT_EMBEDDING_DIM=2560`, as W1 did for `:8210`. **Estimated outage: under 10 minutes** (migration 75.0 s on staging). Outage ends at `/health` 200.

**Step 6: verification, time-boxed (`qa-engineer`, read-only).** Agents use the 4b at the old 0.43 floor from restart until step 7; the box is **30 minutes from `/health` 200 to the floor edit**, and 6.1-6.2 gate step 7, because step 0c already measured the 45 pairs on the 4b. A failure inside the box means R1 and a second outage.
1. `pwq_kb.py gate --ws agent-team --base-url http://localhost:8200 --manifest claude/cobb/scripts/pwq-kb-manifest.json --copy claude/knowledge/python-web-quirks.md` -> `gate: clean`.
2. `pwq_kb.py eval --set A --floor 0.42 --sweep` same target flags, plus the perturbed forms W3 used: compare with the W3 4b table (raw 42/41/38/41, floor-applied 42/37/26/36, near 0/7, off 0/7, within 1 per form); search latency per query under the 30 s embedding timeout. No `--exclude` (the 17 S7 captures are deleted in production). No sealed set is run.
3. (after step 7, may run past the box) the 45-pair fixture on `:8200`, W7b rule; record the R6 document.
4. The §7 non-pwq smoke.
5. The next organic capture: `qa-engineer` checks within a day that it has `embedded == chunks` and `embeddingModel` = the 4b ref (no disposable test write to production).

**Step 7: skill floor (`cobb`, after 6.1-6.2 are green).** `skills/agent-kb-retrieval/SKILL.md`: 0.43 -> 0.42 at lines 9, 40, 94, 96, 103, 106, 121, 132 (the number occurs nowhere else under `claude/*/*.md`, `skills/*`; agent prompts cite the skill by path; `pwq_kb_selftest.py` holds 0.43 as test literals, not a runtime floor); replace the 0.6b model naming; restate the derivation (rule 1.03 x largest full-form Set A positive, `docs/plans/python-web-quirks-kb-embedder-ml.md` §4) and that R6-type instability is re-measured on the 4b by 6.3, not inherited. Open sessions that cached the skill keep 0.43 until they restart (accepted, listed as irreversible-in-effect in §5).

**Step 8: record (`teco`).** Report addendum (before/after counts, backup locations, the K-102 id lists), coordination ledger; the live-spec sentences that name the 0.6b for `ws:agent-team` (`SERVER.md` §1.8, `DESIGN.md` §1.3) go through the root `AGENTS.md` close (delegated writer, `analyst` check).

## 5. Rollback and what is irreversible

- **R1 (default; keeps post-cutover writes).** Under the same lock-first discipline (stop; take the lock): `migrate_embeddings.sh agent-team lmstudio/text-embedding-qwen3-embedding-0.6b --i-have-stopped-traffic` (`dim: 1024` already declared), restart with `EMBEDDING_DIM=1024` (revert the step 2(b) default in the same change, otherwise the hook relaunches at 2560), `cobb` reverts the skill to 0.43/0.6b, gate + Set A at 0.43. The embedder note measured about 18 s for 902 chunks.
- **R2 (only if R1 fails).** Lock, stop, `RENAME ws:agent-team ws:agent-team-failed`, `RENAMENX ws:agent-team-pre4b ws:agent-team` (the copy carries its 1024 indexes and 0.6b override), restart at 1024. The copy was taken after the stop (step 3), so only post-cutover writes are lost. RDB restore (step 1.4) stops every graph: last resort.
- **Triggers:** 6.1 not clean; Set A outside tolerance; W7b rule failed; smoke regression the stakeholder rejects; dimension errors unresolved in 15 minutes; embedding timeouts under the 30 s bound.
- **Retention:** keep `ws:agent-team-pre4b` until K-102 is closed or its id list is in the ledger, and not less than one week after step 6 is green; deletion is the stakeholder's call.
- **Irreversible:** (1) **learning captures attempted during the outage are lost** (agents do not retry; accept by approval 2); (2) the 13 chunks of K-102 documents become embedded and stay so after R1; (3) original 0.6b vectors are overwritten (R1 regenerates approximately, R2 exactly but loses newer writes); (4) `Chunk.embeddingModel` is populated; (5) spent seals cannot be re-run; (6) already-open agent sessions stay without the KB tools until each user runs `/mcp` -> Reconnect (not data loss, but a per-session manual action) and keep 0.43 until restart. Reversible: skill text, `models.json` entry, script default, override, indexes (R1/R2), the process.

## 6. Other consumers, other agents' retrieval, ingestion afterwards

- **Consumers of `:8200`:** every Claude agent through `.mcp.json` `falkor-chat-agent-team` (the skill's twelve consumers plus `cobb`, which writes raw learnings) — down during the outage, and each open session needs the `/mcp` Reconnect after restart (`claude/teco/coordination-techniques.md`); `pwq_kb.py` (`TARGETS`); the autostart hook. `ws:demo`/`salesperson`: separate processes, unaffected, no step touches them. `mcp-monitor`: no `:8200` reference in its tracked docs/config (its local `config.toml` is not in the tree). `tank` shares the LM Studio model copy only. Before the next demo `devops` checks that a resident 4b does not trip `ensure_demo_models.sh` (behaviour with a resident foreign embedder not observed).
- **Raw learnings and cobb.** Raw captures and distilled claims share `ws:agent-team` and one `search_documents`. After migration all are embedded and queried by the 4b; scores change scale (Set A best positive 0.408 vs 0.417, closest near-negative 0.422 vs 0.380). The only floor owner is the skill. Quality of the 4b on non-pwq raw text is measured only by the 45 pairs and the smoke list.
- **Failed documents.** Status is not touched; the 12 documents with unembedded chunks become searchable (search has no status filter). A document `processing` at execution time must reach a terminal state first (`devops` confirms no ingest is in flight).
- **New documents.** The pipeline resolves the embedder through the workspace override (hard cap): 4b at 2560, dimension guard at first write after restart; entity extraction (`qwen/qwen3-4b-2507`) is independent. The 4b embedding and the extraction model share one GPU slot; K-102's candidate cause is eviction, not measured with the 4b: `qa-engineer` counts `failed` documents created in the week after cutover against prior weeks.

## 7. Non-pwq smoke list (owner `qa-engineer`)

Ten situation queries from the fixture's non-pwq rows plus four raw-learning situations (FalkorDB, asyncio, agent hooks, WSL/LM Studio; chosen without reading any sealed data), skill-prefixed, `limit=5`, floors 0.42 and 0.43, run on `:8200` before migration (step 0c), on `:8210`, and on `:8200` after (step 6.4): record top-1 title, score, count under floor. Condition: no known answer leaves the top five or crosses above 0.42; otherwise the scores go to the stakeholder, never tuned.

## 8. The stakeholder's decision set

1. **Promote (A) or not (C1)**, knowing §1's disclosure: one-query B3 margin, raw 123/126, no paired 0.6b/4b sealed result, post-migration checks support "no gross regression" only.
2. **The outage and its slot** (about 10 minutes, `:8200` only; no demo, no heavy ingest): agents lose KB access, learning captures in the window are lost, and every open session must run `/mcp` -> Reconnect afterwards.
3. **Floor 0.43 -> 0.42 skill-wide**, conditional on step 0c and 6.1-6.3; rows between 0.42 and 0.43 come back to you.
4. **K-102 side effect:** 12 failed documents become searchable.
5. **Repo edits:** `falkor-chat/config/models.json` (one entry), `start_agent_team.sh` default `EMBEDDING_DIM` 2560 (permanent), `skills/agent-kb-retrieval/SKILL.md`; and the two new units (`W7a` fixture/runner, `W7b` rule note).
6. **Backups:** `ws:agent-team-pre4b` and the logical export, retained until K-102 closes (at least one week).
7. **The irreversible list in §5.**
8. **Optional follow-up, separate approval:** post-cutover sealed set B4 on production (the only instrument for a production-quality claim).

## 9. Risks and open questions

- Floor margin is thin on both sides; embedding-session instability applies to the 4b.
- Production's competitor set differs slightly from staging's (4 documents, later captures); step 6 re-measures.
- Not observed, each with a stated fallback: what FalkorDB does with a differing-dimension `CREATE VECTOR INDEX` over an existing index; 4b footprint in LM Studio; `ensure_demo_models.sh` with a resident 4b.
- The seconds-wide gap between stopping pid 1268 and taking the lock is closed only by step 1.1 (no teco start); `devops` may prefer to stop `ensure-environment.sh`'s trigger by briefly disabling the hook if the stakeholder allows it.
- After promotion `ws:pwq-test`/`:8210` remain as a test double; retiring them is the stakeholder's call.
