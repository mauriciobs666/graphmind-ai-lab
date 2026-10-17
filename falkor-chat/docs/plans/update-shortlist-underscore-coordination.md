# Update-shortlist lone-underscore title crash — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-075

Goal: close `falkor-chat/docs/BACKLOG.md` **K-075**: a title token that is only `_` builds `%_%` in `find_update_shortlist`'s title branch, a RediSearch syntax error that also discards the band candidates.
**Done:**
- an underscore-only title token never reaches the query;
- the band candidates survive such a title;
- the fix is mutation-checked;
- `analyst` passes the diff;
- HISTORY has an entry and K-075 is removed from BACKLOG.

Baseline: `4f978ddf`. Scope boundary and test strategy: the K-075 body. Precedent: `fusion._fuzzy_query` (K-068). CPG `cpg_falkorchat` is stale (6 server commits since the build). This is a one-function change that leans on it only lightly, so no rebuild.

## Documentation impact

- `falkor-chat/docs/HISTORY.md`: one entry.
- `falkor-chat/docs/BACKLOG.md`: remove K-075, and check cross-references to it (K-064, K-076).
- Docstrings: `repository._escape_fuzzy_token` and `find_update_shortlist`; also `fusion._fuzzy_query` if it describes the title path's behaviour.
- `falkor-chat/docs/QUERIES.md`: only if it describes title-token handling. The implementer checks.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| F1 | `tdd-engineer` | `a296d0119bb4b37bb` | accepted (`_escape_fuzzy_token` returns `""` for underscore-only tokens; mutants a–d killed; teco re-ran 36 passed, teco mutant (keep 3+ underscores) killed 1) | `server/falkorchat/repository.py` + `server/tests/test_repository.py` + `QUERIES.md` §14.6, HISTORY, BACKLOG | `analyst` (G1) → **approve w/ suggestions** (F-1/F-2 doc wording applied by teco; docstring nit left) | 114k tok / 40 tools |
| G1 | `analyst` | `a52da3a0f56fa4bf5` | accepted | `falkor-chat/docs/reviews/update-shortlist-underscore.md` → approve w/ suggestions; premises (a)–(d) confirmed by execution | — | 87k tok / 22 tools |

## Closure (2026-10-17)

K-075 delivered; record in `falkor-chat/docs/HISTORY.md` 2026-10-17. No follow-ups filed; the title path's implicit AND stays with K-064.
