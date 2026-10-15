# teco environment auto-bring-up at agent start — coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** cobb K-054 (—)

Goal and open questions: `claude/cobb/kaizen/plan.md` § K-054 (baseline commit `5c82170f`).
Done: teco checks and brings up FalkorDB + the `ws:agent-team` server (:8200) automatically at
start, via a hook if the harness supports it, live-verified, docs/kaizen updated, gated.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `cobb` | `a975961e6b799842a` | accepted (superseded by U6's final revision) | `claude/teco/hooks/ensure-environment.sh` + 8 doc/kaizen files (cobb history 2026-10-14) | `analyst` → — | 255k tok / 95 tools (cumulative) — next cobb work dispatch fresh |
| U2 | `analyst` | `a7b04f38ba0b7899e` | accepted (Pass 2: approve w/ suggestions; 5-point done-condition closes gate, no Pass 3) | `claude/docs/reviews/teco-env-autostart.md` — approve w/ suggestions (M1, M2, m1–m4, n1–n3) | — | 243k tok / 65 tools (cumulative) |
| U3 | `graph-dba` | `aad7a8a03b5bb17b8` | accepted | `cpg_falkorchat` rebuilt @ `5c82170f`, tree `9e41122e` = HEAD, dirty false; artifacts refreshed; `docs/HISTORY.md` entry committed (see `git log -- docs/HISTORY.md`); `joern-cpg` skill lacks a refresh-artifacts rule → cobb follow-up | teco verify → ok (marker re-queried, check 0 equal, METHOD 6140 / 1534 / 3885 re-counted) | 100k tok / 25 tools |
| U4 | `cobb` | `a975961e6b799842a` (folded into U1's revision) | accepted (in U1 rev 2, final in U6) | hook gains CPG freshness check-and-report (user chose report, not auto-rebuild, 2026-10-14); same file as U1 → after U2 verdict, resume U1's agent | `analyst` (re-gate) → — | — |
| U6 | `cobb` (fresh) | `a67e580bd86210fa3` | accepted | final hook revision vs review Pass 2 done-condition + `claude/teco/hooks/test-ensure-environment.sh` + host-stall fact (`claude/devops/ops-quirks.md`) + `joern-cpg` refresh-artifacts rule | teco checks the 5 points → ok (DC1 grep 0 / timeout 30; DC2 `--mutants` re-run: shipped green, 3 mutants red; DC3–5 cells in the shipped run; own mutants WS_ID value + pinned FALKORDB_PORT each caught, 5 failures) | 193k tok / 46 tools |
| U7 | `coder` (haiku) | `a0d4b5ae1e4fa1ae8` | accepted | port-first FalkorDB check in `falkor-chat/scripts/start_server.sh` + `start_demo.sh` + HISTORY entry — commit `baa781e6` | teco verify → ok (diff read, harness blocks = tree, own port-arg mutant caught) | 94k tok / 15 tools |
| U5 | `coder` (haiku) | `acc62998ccdec26f7` | accepted | `falkor-chat/scripts/start_agent_team.sh` step 1 port-first + HISTORY entry — commit `e7a020cc`; siblings `start_server.sh`/`start_demo.sh` share the pattern → fixed in U7 | teco verify → ok (diff read, mutation shown, probe inputs checked) | 80k tok / 11 tools |

Closed 2026-10-14: the review gate is closed by its Pass 2 done-condition (no Pass 3). Open follow-ups: cobb K-055; `start_agent_team.sh` has not had a real cold start under the `env -i` allowlist (devops, see U6 report); CPG parse root excludes `falkor-chat/scripts/` (user's call).
