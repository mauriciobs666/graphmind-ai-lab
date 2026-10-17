# Fusion fuzzy-query escaping — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-068

Goal: close `falkor-chat/docs/BACKLOG.md` **K-068**. `fusion._fuzzy_query` sends entity-name tokens to RediSearch unescaped, so `_safe_fuse` drops the suggested-tier `SAME_AS` lookup for any name containing query syntax. The K-071 re-drive logged 157 `background fuse failed` lines in the `:8200` log, across many trigger shapes besides `-<digit>`.
**Done:**
- every entity name that failed live builds a query RediSearch accepts;
- plain names keep matching;
- pure-punctuation names yield no query;
- the fix is mutation-checked;
- `analyst` passes the diff;
- HISTORY has an entry and K-068 is removed from BACKLOG.

Baseline: `3d91f750`. Precedent: `repository._escape_fuzzy_token` (HISTORY 2026-09-13, `find_update_shortlist`). The cause is already known (K-068 body + precedent), so there is no RCA unit. Evidence: `~/.local/state/graphmind-ai-lab/agent-team-8200.log` (`background fuse failed (entityId=…)` lines; names are readable from `ws:agent-team`).
CPG `cpg_falkorchat` is stale (5 server commits since the build). The fix is two small modules, so it leans on the CPG only lightly; I'm proceeding without a rebuild.

## Documentation impact

- `falkor-chat/docs/HISTORY.md`: one entry.
- `falkor-chat/docs/BACKLOG.md`: remove K-068, and check K-064's cross-reference to it.
- Docstrings: `fusion._fuzzy_query`, plus `repository._escape_fuzzy_token`, whose text says fusion is left untouched.
- `falkor-chat/docs/QUERIES.md` / `scripts/test_queries.sh` §14.x: only if they show the fusion query shape. The implementer checks.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| F1 | `tdd-engineer` | `a101aa308120caf60` | accepted (pass 2: ASCII-punct/space/tab split, controls+`\` deleted; K-075 filed; teco re-ran 141 passed/1 xfailed, replay 157/157, 7480 names 0 errors; teco mutant (no tab) killed 2) | `server/falkorchat/fusion.py` (split on non-word runs, drop `_`-only tokens) + `server/tests/test_fusion.py` + docstrings, `QUERIES.md` §14.6, HISTORY, BACKLOG; M1–M4 killed; teco re-ran 45 passed, teco mutant (split on whitespace/hyphen only) killed 25, replay re-run 157/157 clean, 7450 names 0 errors | `analyst` (G1) → Pass 1 needs changes; Pass 2 **approve w/ suggestions** (P2-1/P2-2 → K-076, HISTORY wording nit applied by teco) | 150k+209k tok / 47+28 tools |
| G1 | `analyst` | `a43457618998d668a` | accepted | `falkor-chat/docs/reviews/fusion-fuzzy-escape.md` → **needs changes** (F-1..F-5); Pass 2 → **approve w/ suggestions**; implementer's F-1 regex overrule confirmed by probe | — | 101k+138k tok / 25+18 tools |

## Closure (2026-10-17)

K-068 delivered; record in `falkor-chat/docs/HISTORY.md` 2026-10-17. Follow-ups filed: K-075 (title-path lone `_` crash), K-076 (symbol-only junk suggestions, `\`-escaped punctuation).
