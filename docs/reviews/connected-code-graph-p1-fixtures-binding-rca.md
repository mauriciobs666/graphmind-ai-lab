# RCA: P1 fixture build, bindings do not bind (defect D3)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

CPG: considered, not relevant — the subject is the ccg linker/pack/extractor code and the fixture inputs, read and executed directly; `cpg_*` Joern graphs do not model them.

Scope: defect D3 of `docs/test-reports/connected-code-graph-p1-5d-50-report.md` (D3 section, Section E rows LNK-16/18/19/21/22/45, AC6-EX-2). Baseline: working tree at HEAD `86c98ef4`, fixtures materialized from `cpg/fixtures/system1`. Root and `cpg/` `REVIEW.md` read; none changes an RCA.

## 1. Symptom and impact

On the fixture build `BOUND_BY` 0, `binding-confirmed` 0, `LAUNCHES` 1, `loopback_port_residual` 1 (QA report). Rows red: LNK-12/14/18/21/22/45 (expected `binding-confirmed`), LNK-16 (no link), LNK-19 (links to both `reindex` providers, NC-5 violated), AC6-EX-2 (reason unverified).

## 2. Reproduction and evidence

**Ran (offline, no store, no FalkorDB started, `ccg-falkordb:6390` / `falkordb-dev` / existing throwaways untouched):**
1. `cpg/.venv/bin/python cpg/fixtures/materialize.py --out <scratch>/fx`, then `build_units` (real `build_unit`, Joern, 2m19s) over `fixtures/system1/system.yaml` at the fixture heads, with a scratch `keys_root` (script `<scratch>/build.py`).
2. The refresh `link` stage verbatim: `run_packs(bundles, DEFAULT_PACKS)`, `Binder(..., units=units)`, `link(...)` (script `<scratch>/link.py`). Result equals QA's offline figures: `links 37` (http 27, mcp_tool 10), by reason `key-match 12, unknown-key 12, partial-key 5, prompt-mention 4, shell-dynamic 4`, `loopback_port_residual 1`, `bind_key_mismatch []`.
3. Hop-by-hop probes of the `Binder` internals, and three in-memory mutations of the bundle records (`<scratch>/mut.py`: `none`, `dockerfile`, `script`). Scratch dir: `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad`.

**Read, not run:** the proving store's 114 `BOUND_BY` (QA figure; I did not query 6390), `LAUNCHES` count (derived from `ccg/assemble/rows.py:769-779`: one edge per `launchUnit` fact; I observed one `launchUnit`).

## 3. Causal chain

`BOUND_BY` entries exist only on a `Link` whose consumer site got a `Binding` from `Binder.bind` (`ccg/linker/link.py` `_evidence`). Of the 20 pack consumers plus 3 prompt mentions in the fixture build, **one** gets a `Binding`, and its link is dropped. Per consumer class:

| # | Consumer | Observed | First missing precondition |
|---|---|---|---|
| a | 17 http consumers (fx-web TS) | `_own_urls`/`_env_urls`/`_pack_urls` all empty | No bindable URL: bases come from `import.meta.env` / constants (`bind.py` module docstring: leaves no trace), or lifted keys are relative. Binder takes no input here. Nothing to fix in the Binder. |
| b | Python consumers (LNK-08/11/14/15/37, TW-2) | not present | D1 (no typed `requests` verb), outside D3. |
| c | `JobStatus.tsx` `fetch("http://localhost:8000/job/status")` (LNK-16) | **bound**: loopback :8000 matches only compose `api` port (`compose ports {8000: api, 8300: job}`) so unit = svc; `loopback_port_residual 1` | Provider `GET /job/status` is in unit job; `link.py` `_candidates` skips providers outside `binding.units` ("a key that matches a provider outside them makes no link", `Bindings` docstring). The second :8000 listener is `fx-job/scripts/dev.sh` (`exec uvicorn ... --port 8000`), present only as a shell `callsite`; no extractor turns it into a `launchFact` (`extractors/shell/extract.py:9`, `extractors/config/launch.py` table). So no `port-shared`. |
| d | Mention `mcp__fx-svc-exec__list_orders` (LNK-45) | `bind` returns `None` | `_launch_chains`: server entry found, `_script_of` resolves `scripts/run-mcp.sh`, then `_commands_at[(fx-svc, scripts/run-mcp.sh)]` = 0 (shell scripts carry no `launchFact`) and `_commands_at[(fx-svc, scripts/Dockerfile)]` = 0 (no such file). Chain dies at hop 2. |
| e | Shell `refresh.sh` `mcp-call reindex` (LNK-19) | `bind` returns `None`; both providers linked `confirmed`/`key-match` | No URL source for a shell consumer: `_own_urls`/`_env_urls` are `http`-only, `_pack_urls` reads `streamablehttp_client` callsites and `.mcp.json` URLs, `export MCP_URL=...` (a shell `constant`) is read by nothing. Independently, no pack `port` fact for :8200 exists (see f). |
| f | Pack launch facts | only `launchUnit` for `fx-job/Dockerfile` (`job.trigger_server:app`) and its `port 8300` | `fx-svc/deploy/Dockerfile` CMD arg `fx_svc.api.app:create_app` is looked up as file suffix `fx_svc/api/app.py` (`packrt/launch.py` `_launched_unit`); the corpus path is `api/app.py` (unit root = repo root; the `fx_svc` prefix exists only as `pyproject.toml` `package-dir = {"fx_svc" = "."}`). No unique hit, no `launchUnit`, no :8000 pack port, no LAUNCHES. `python -m fx_svc.mcp.server` (run-mcp.sh) and `--port 8200` (`.mcp.json` args) are not `module:attr`/uvicorn forms. |

**Why the proving store differs (read, plus a bundle scan):** its stored bundles carry four command launch facts: `.mcp.json` `cypher` -> `cypher-mcp/docker-run.sh`, `cypher-mcp/Dockerfile` (`python server.py`, sits beside the script), `falkor-chat/Dockerfile` (`falkorchat.app:app`, matches `falkor-chat/server/falkorchat/app.py` by suffix), `.mcp.json` `ccg` (redacted). The repository layout satisfies both hops the Binder supports (Dockerfile beside the script; module path equal to a path suffix). The fixtures were authored against a richer chain than the Binder implements.

**Hypotheses ruled out:** key epoch (`bind_key_mismatch []`, `bind_parts_usable` ok); Compose binding and cross-repo build context (probed: `http://api:8000` -> svc, `http://job:8300` -> job, `../../fx-job` resolves); `.mcp.json` parsing (the server entries and the `bash -c exec "$CLAUDE_PROJECT_DIR/..."` form reach `_script_of` correctly); Joern (every unit's build returned ok in my run).

## 4. Root cause (confirmed by execution)

D3 is not one defect but a **contract gap between the fixture rows and what P1 implements**, which the fixture author closed by derivation, not by running ("the manifest row is derived by reading `bind.py`... not run", `docs/test-plans/connected-code-graph-fixtures.md` LNK-45 note). Three independent contract mismatches:

1. **Script hop.** Fixture rows LNK-21/45 expect `run-mcp.sh` (`exec python -m fx_svc.mcp.server`, "`targetArg`") to resolve to unit svc. `docs/plans/connected-code-graph-p1-linking.md` P1-17 (6) says the opposite: no extractor emits a shell script's launch fact; the P1 chain is server `command` entry, then the `Dockerfile` `CMD` beside the script. The fixture has no Dockerfile beside `scripts/run-mcp.sh`, and `targetArg` is `module-attr` only.
2. **Module path vs package-dir.** The fx-svc Dockerfile uses the installed package name `fx_svc.*`; the launch resolver matches repo-relative path suffixes. Same for `-m fx_svc.mcp.server`.
3. **LNK-45 status.** Even with a complete chain, a prompt mention stays `possible`/`prompt-mention` with `BOUND_BY`: plan P1-17 line 519 ("a prompt mention is `possible`/`prompt-mention`"), `link.py` `_causes` (`c["status"] != "confirmed"`), pinned by `tests/test_linker_bind.py:192,260`. The expected `confirmed`/`binding-confirmed` is a derivation error; the QA-observed `possible`/`prompt-mention` is correct for that column.

Mutation evidence (in-memory, bundles only): (`dockerfile`) rewriting the svc Dockerfile arg to `api.app:create_app` gives a second `launchUnit` and `port 8000`, and changes no link; (`script`) adding a `scripts/Dockerfile` command fact with `mcp.server:app` makes the `list_orders` link carry 1 `BOUND_BY` entry and stay `possible`/`prompt-mention` (point 3); neither touches LNK-19 or LNK-16.

**Trigger:** P1-41's first run on a real fixture build. **Contributing factors:** rows derived by reading, not running; no test builds the fixture bundles through `Binder` (binding tests use synthetic records, `tests/linker_bind_helpers.py`); `test_fixtures.py` checks manifest consistency, not the build.

## 5. Is LNK-19's NC-5 violation the same cause?

**Separate cause, same family.** LNK-19's binder-visible input is `export MCP_URL=http://localhost:8200/mcp`; no code path binds a shell consumer by URL (`grep MCP_URL cpg/ccg cpg/packs cpg/tests/test_linker_bind.py`: nothing; `bind.py` has no shell/`constant` reader). Fixing launch facts (#1/#2) does not change it, and mutation `script` confirmed LNK-19 still links both providers. Its current output (`confirmed`/`key-match` to every `reindex` provider) is what the `Bindings` docstring prescribes for an unbound site. NC-5's "LNK-19 has exactly one provider" cannot be met without a new shell URL source, which no plan step assigns (P1-46 states "Neither the pack format nor the shell extractor changes").

## 6. Suggested fixes (findings to judge, each needs a ruling)

| ID | Sev | Finding and suggestion | Owner |
|---|---|---|---|
| R1 | major | LNK-45 expected status contradicts the plan and the pinned linker tests. Change the manifest `p1Form` and the fixtures test-plan row to `possible`/`prompt-mention` with a `BOUND_BY` chain (the chain assertion is the real P1 test). | `architect` rules; `tdd-engineer` edits manifest + `test_fixtures.py` |
| R2 | major | LNK-21/22/45 chain: pick one. (a) conform the fixture: add `fx-svc/scripts/Dockerfile` with a `module:attr` CMD that resolves to unit svc (adds a file, so UNIT_FLOORS/family counts move); (b) extend P1 (shell `exec` launch facts, `-m` `targetArg`), contradicting P1-17 (6); (c) move LNK-21/45 to P2. I favour (a) or (c); (b) is scope growth. Mutation `script` shows (a)'s mechanics work. | `architect` |
| R3 | major | LNK-16 `port-shared` needs a second :8000 launch fact from `fx-job/scripts/dev.sh` (callsite only today), unavailable at P1 by the same contract. Re-scope to P2, or give job a Dockerfile/compose :8000 listener (changes LNK-15/16 elsewhere: check). | `architect` |
| R4 | major | LNK-19/NC-5: re-scope (LNK-19 expected unbound `key-match` to both providers at P1; NC-5 holds for LNK-12 only, itself blocked by D2), or add a shell URL source (P2-sized, no plan step). | `architect` |
| R5 | minor | fx-svc Dockerfile CMD `fx_svc.api.app:create_app` can never resolve under suffix matching. Rewrite to a unit-relative module (consistent with the test plan's "imports are unit-relative" rule, line 121), or accept as a P2 package-dir item. Verified mutation: gives `launchUnit` + :8000 port, no link change. | `architect` |
| R6 | major | Pin (below). | `tdd-engineer` |

**The test that pins it (offline, no Joern, no store):** build the config and shell units of `fx-svc`/`fx-job` only (extractors/config over `.mcp.json`, `deploy/compose.yaml`, both Dockerfiles; shell for `scripts/*.sh`), `run_packs`+`Binder`+`link` over those bundles plus the symbol/file records, and assert the ruled expectation row by row: the `launchUnit` set; for each of `fx-svc`, `fx-svc-exec`, the `_launch_chains` result (unit or none) against the manifest row; LNK-45 `(status, reason)` and its `BOUND_BY` ids; LNK-19's provider count. **Mutants that must turn it red:** delete `scripts/Dockerfile` (or the chosen resolver); change a CMD module path; flip LNK-45's expectation to `confirmed`. Expected values come from the manifest rows, never from `Binder` output. Do not place the assertion in the same change that fixes the fixture (it must be authored red first, per the ruling).

## 7. What is solid

The Binder's Compose, loopback-port, service-host and cross-repo-context binding all work on the fixtures (probed); key-epoch handling; QA's counts reproduce exactly offline, so the QA measurement is trustworthy.

## 8. Open questions

1. Which of R2 (a)/(b)/(c) and R3/R4 re-scopes does the architect choose? They decide whether P1-41's 89-row count stays.
2. Not probed: LNK-14's config-file-bound `base_url` (`fx-job/config/job.yaml`) has no Binder input either (only env vars and `.mcp.json` URLs bind); blocked first by D1, so undetermined whether it is a fourth mismatch.
3. AC6-EX-2's `example-config` reason was not reproduced (its consumer is a TS `import.meta.env` base; case (a) above says it cannot bind).
