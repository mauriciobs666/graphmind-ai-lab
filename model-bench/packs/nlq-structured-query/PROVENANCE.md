# Provenance — nlq-structured-query

> **Pack version:** 1.1.0 · **Generated:** 2026-09-12T02:08:25Z

Data files below are copied one-way from `falkor-chat/` by `model-bench/scripts/refresh_golden.py` (plan §3.1 point 3, D1: "copy the data, clean-build the code"). Re-running the importer requires bumping `pack.json`'s `packVersion` by hand first — a content-hash-changing edit under an unchanged version number is refused.

| Origin | Destination | Source git SHA | Source SHA-256 | Copied at |
|---|---|---|---|---|
| `falkor-chat/server/tests/eval/nlq_golden_set.jsonl` | `items.jsonl` | `d58125ad3164c4f589dcc135ee53c89daa3b2d93` | `f198290dab95ce8fd2f95204b6d24b195846e078d4c08fd0a959e81d7b02e0da` | 2026-09-12T01:12:50Z |
| `falkor-chat/scripts/seed_catalog.sh` | `tables.json#catalog` | `14891c94f9d4cefeec4adbd873697dc5b11b1fb7` | `58d80fcc93cd20db49cc66e28bcf090d92771e0916285b4356124bbbe18045b7` | 2026-09-12T01:12:50Z |
| `falkor-chat/server/falkorchat/querygen.py` | `schema.json` | `3d014975aca6244ab78c589caadedd39db5a6392` | `6276c28a69c5d8ed469efb3730478d6769fd9d0a4f5f2d6d71b9f911e90297b6` | 2026-09-12T01:12:50Z |
| `ws:nlq-eval (live FalkorDB snapshot)` | `tables.json#knowledge_base` | `0dcba6361cd841f3b394ce1030c0f84302a11693` | `7f7857233eb91f9ab5d16ff85bbd8dc96fda37294276450cdd399620c1f670f3` | 2026-09-12T02:08:25Z |

**Note on the `ws:nlq-eval` row's "Source git SHA" (U134, S4 Step 4, §2.6):** unlike the three
file-copy origins above, a live Cypher read against `ws:nlq-eval` has no single tracked source
commit — there is no file in the monorepo whose content the row's SHA-256 is a hash of. The value
recorded is this monorepo's own HEAD commit at snapshot time
(`0dcba6361cd841f3b394ce1030c0f84302a11693`), used as the closest honest proxy for "when", not a
claim that a specific `falkor-chat/` file was copied — the row's own "live FalkorDB snapshot"
origin label already says so. The snapshot itself (62 `Entity` / 12 `Document` / 12 `Chunk` rows,
taken via `mcp__cypher__query` against `ws:nlq-eval`) is captured verbatim in this unit's delivery
report.
