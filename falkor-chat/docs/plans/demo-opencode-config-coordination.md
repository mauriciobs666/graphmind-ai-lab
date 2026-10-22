# Demo bring-up finds a reachable LM Studio config by itself; ensure-step leftovers — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-083

## Goal

Close `falkor-chat/docs/BACKLOG.md` K-083. The backlog item is the spec, so there is no architect
plan. One `devops` unit, then an `analyst` code gate.

**Decision (teco, 2026-10-22, on the stakeholder's "K-083 please" after recommending the
fallback):** the bring-up entry points that default `FALKORCHAT_OPENCODE_CONFIG` adopt
`start_agent_team.sh`'s resolution order: an explicit env var, then `opencode.local.json`, then the
shared `opencode.json` with a warning. The scope is `start_demo.sh`, `start_server.sh`, and
standalone `ensure_demo_models.sh`. The one-off maintenance scripts (`seed_eval_corpus.sh`,
`migrate_embeddings.sh`, `pin_workspace_embedding_model.sh`) are out of scope: they deliberately
name the shared file.

## Definition of done

- `./scripts/start_demo.sh` with no env var reaches a reachable LM Studio on this box.
- The three scripts resolve the config the same way, from one shared definition rather than three
  copies.
- Offline tests cover the resolution order and are mutation-tested.
- Leftovers fixed: settle env parse, settle-wait test, `HTTPException` traceback, and the
  `start_demo.sh` exit-3 test (or a stated reason it isn't feasible).
- Docs updated.
- `analyst` gate passed.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 fallback + leftovers + tests + docs | `devops` | `a1c73e8098bb43c19` | accepted (`32d9b842`) — 60 green (teco re-run); 14/14 mutants red; teco mutant (helper local path renamed) → 7 red, restored (`cmp`); live ensure no-op with INFO line, exit 0 | `falkor-chat/scripts/shlib/opencode_config.sh` (not `lib/`: root `.gitignore`), 5 scripts, 2 test files, `falkor-chat/AGENTS.md`, `falkor-chat/README.md`, `salesperson/README.md` | `analyst` (`ad3a705fcb4173bb7`, 115k tok / 35 tools) → pass 1 approve w/ suggestions (m1 `RemoteDisconnected` loses hint, m2 resolved value not exported before `create_workspace.sh`, m3 stale docs, nits) → fixes by `devops` (m1 clause reorder, m2 helper exports, n1–n3, `seed_eval_corpus.sh` comment; 69 green teco re-run; 6/6 new mutants red; teco mutant local→shared assignment → 10 red, restored `cmp`; 187k tok) → pass 2 **approve** · `falkor-chat/docs/reviews/demo-opencode-config.md` | 156k tok / 36 tools |
| U2 manual line (`llm-provider-config.md`, review m3) | `tico` | `a5705f178d5396fc8` | accepted (`32d9b842`) — Walkthrough 1 step 2 rewritten (+7/−4) | `falkor-chat/docs/manuals/llm-provider-config.md` | teco fact-check: the four scripts it names are exactly those sourcing `resolve_opencode_config` → holds | 53k tok / 4 tools |

## Documentation-impact scan

- `falkor-chat/AGENTS.md`: the scripts table and the demo paragraph's local-config sentence.
- `salesperson/README.md`: "Running the demo", which currently says to export `opencode.local.json`.
- The header comments in `start_demo.sh`, `start_server.sh` and `ensure_demo_models.sh`.
- `falkor-chat/docs/SERVER.md`, if it documents the default.
- At close (teco): `BACKLOG.md` K-083 and `HISTORY.md`.

## Close (2026-10-22)

Definition of done met: the four bring-up scripts resolve the config through one helper, and the
value is exported to child scripts. The leftovers are fixed and tested (69 green, mutants red). The
`start_demo.sh` exit-3 branch is tested. Docs are updated, including the manual. `analyst` approved
after 2 passes. Code commit `32d9b842`. K-083 is removed from `falkor-chat/docs/BACKLOG.md`; the
record is in `falkor-chat/docs/HISTORY.md` 2026-10-22.

Follow-ups:
- K-084: the one-off scripts' default (stakeholder call) and the custom-identifier re-probe.
- Not filed: `claude/data-scientist/lm-studio-model-notes.md` still describes the old default. It
  is under `claude/**`, another session's area.
