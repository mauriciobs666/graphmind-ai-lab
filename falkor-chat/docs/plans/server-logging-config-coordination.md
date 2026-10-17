# Server logging config — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-069

Goal: close `falkor-chat/docs/BACKLOG.md` **K-069**. The app installs no log config, so under uvicorn's default config every `falkorchat.*` INFO line is dropped, and WARNINGs reach stderr bare, with no timestamp (`docs/test-reports/extract-transient-retry-report.md` §6 F-1).
**Done:**
- the shipped start scripts produce `falkorchat.*` INFO lines with timestamps;
- uvicorn's own access/error lines still appear;
- a caller's `UVICORN_ARGS` override doesn't silently lose the config, or the scripts document that it does;
- proven by starting a throwaway server through a script (never by touching the shared `:8200`);
- `analyst` passes the diff;
- HISTORY has an entry and K-069 is removed from BACKLOG.

Baseline: `abe1cd1f`. CPG: not consulted (the change is in shell scripts and config).

## Documentation impact

- `falkor-chat/docs/HISTORY.md`: one entry.
- `falkor-chat/docs/BACKLOG.md`: remove K-069, and check any cross-references to it.
- `falkor-chat/docs/SERVER.md`: wherever it describes the server process, start scripts or logs.
- `falkor-chat/AGENTS.md`: Key scripts table, only if what a script does changes.
- The start scripts' own header comments.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| L1 | `devops` | `a34001a6dea1ca167` | accepted — round 2 (+`logformat.py`, `test_logging_config.py`, SERVER.md §1.10; 11 mutants killed; teco re-ran test 1 passed, teco mutant falkorchat→`default` handler killed) (round 2 brief: G1 M1/M2/m3–n8; teco decided m3 = intended noise cut, F-1 erratum goes in HISTORY) (teco: in-process check OK; teco mutant `falkorchat.propagate=true` doubles lines and survives `check.sh`'s ≥1 rule, so it goes to G1) | `config/logging.json` + 3 start scripts + `Dockerfile` + README, SERVER.md §1.7, HISTORY, BACKLOG; mutants A–C killed; probe graph `ws:k069probe` left | `analyst` (G1) → Pass 1 needs changes; Pass 2 **approve w/ suggestions** (n9 HISTORY clause applied by teco) | 141k+192k tok / 47+29 tools |
| G1 | `analyst` | `a2216ebecee464258` | accepted (Pass 2 approve w/ suggestions, re-verified `--reload` live on `:8294`; Pass 1 **needs changes**: K-069 premise false, FastMCP `basicConfig` already printed INFO bare; teco verified 18 bare `abandoning retry` lines in `:8200` log) | `falkor-chat/docs/reviews/server-logging-config.md` | — | 117k+146k tok / 39+15 tools |

## Closure (2026-10-17)

K-069 delivered; record in `falkor-chat/docs/HISTORY.md` 2026-10-17, which also carries the F-1/B.5 erratum: FastMCP's import-time `basicConfig` meant INFO lines were printed bare, not dropped. Left for the stakeholder:
- restart `:8200` to pick up the config;
- the probe graph `ws:k069probe` (cleanup needs approval).
