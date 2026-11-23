# Held-out strata counts for the narrowed AC-5 first gate (counts only)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (CCG P1)

Counts-only custodian query. No ids, paths, qnames, keys or item content were printed or recorded. Held-out artifacts were read, not modified.

CPG: considered, not relevant — a counts query over held-out gold files, no code-level analysis.

## Counts (held-out `ac5-<stratum>.jsonl`, as of 2026-11-23)

| Stratum | Items (line recount) | Items (manifest `items`) | Counting items | Clusters containing a stratum item |
|---|---|---|---|---|
| http | 26 | 26 | 26 | 15 |
| mcp | 46 | 46 | 46 | 4 |
| scripts | 79 | 79 | 79 | 36 |
| wiring | 8 | 8 | 8 | 5 |
| flow | 99 | 99 | 0 | 12 |

http+mcp together: 72 items, 72 counting items, 19 distinct clusters, smallest stratum count 26 (http).

Against the plan floors (8 per stratum, 75 pooled counting items): both strata are at or above 8 (26 and 46). http+mcp pooled is 72 items, 3 short of the 75 pooled floor. The pooled floor was tested on counting items H, and all 72 http+mcp items are counting. Whether the 75 floor applies to the narrowed pair is a decision for the caller and is not made here.

## Method agreement

- Method A, manifest's own recorded counts (`cpg/eval/heldout-manifest.json`, `files[].items`), against Method B, a line recount of the held-out files: agree for all five strata.
- Counting-item counts and cluster counts come from a third, in-memory pass. It re-ran `split_manifest.split()` over `work/s45/match-scoped.json` with the recorded eval commit, then mapped held http/mcp itemIds to clusters. Nothing was written. That reconstruction yields http 26, mcp 46, wiring 8, flow 99 (equal to the held lists) but scripts 80 (the held list has 79). Every held item is contained in the reconstructed held set, and the one scripts difference does not touch http/mcp. This is the reason the cluster figures are marked as reconstructed. `s45/split-report.json` records `heldClustersPerStratum` by primary stratum (http 15, mcp 3, scripts 36, wiring 4, flow 12). That counts a cluster once, under its dominant stratum. The table counts clusters containing at least one item of the stratum, so mcp is 4 there and wiring is 5. The 19 distinct clusters for http+mcp use the second definition.

## Verify

`split_manifest.py verify --manifest cpg/eval/heldout-manifest.json --heldout-dir ~/.local/share/ccg-eval/cf7b2042cce7/heldout` printed `checked: 77`, `mismatches: 0` and exited 0. `runs/index.jsonl` (size 19031, mtime unchanged) and the held-out directory file listing/mtimes were identical before and after the run. Nothing was re-recorded.

## Open question for the caller

http+mcp reaches the per-stratum floor of 8 but has 72 counting items against the pooled 75. Is the pooled floor kept at 75 for the narrowed gate, restated, or is the shortfall accepted? Not decided here.
