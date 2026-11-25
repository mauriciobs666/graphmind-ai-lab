# ccg P1 AC-5 gate: contacted clusters moved to dev (counts only)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1)

CPG: not applicable — a procedural custodian operation on eval data lists, no code-level analysis.

## Summary

Custodian operation under the stakeholder's 2026-11-25 authorisation (renew and follow the record's rule) and `teco`'s route decision, after the freeze-check at R=`f4ef2239` reported 7 hits (`docs/test-reports/connected-code-graph-p1-freeze-check-report.md`; prereg §4 item 6). **Result: done, exit 0 on every command.** The 7 hit hashes map to 35 held-out items: 34 in ac5 lists (4 anchors, 3 clusters) and 1 in `ac24`. The 3 clusters moved to dev as a pure move (`split_manifest.py regate-list --missed`, EP §12.1, no fresh items). No held-out content, id or path was printed; no engine, build, `gate.py run` or second freeze-check was run. Nothing is committed.

## Route and why

- `contact` + `split` with an enlarged C was rejected: a dry run (in memory) pulled 26 new items from dev/reserve into held-out and shifted every gated count, a fresh draw. It also reopens C, which EP §8.4 closes at S47.
- `regate-list --missed` with the unchanged contact file (6 anchors) and the gate-1 manifest: it first re-derives gate 1 from `work/s45/match-corrected.json` and refuses unless the lists hash to the manifest (`gate1_problems`); that held (the `match-scoped.json` input does not reproduce scripts, 80 vs 79).
- `contact.jsonl` is unchanged (`contactAnchors` stays 6, raw artifact hashes equal).

## Commands and exit status

| Command | Exit | Counts |
|---|---|---|
| `regate-list` (match-corrected, context, contact, eval commit, manifest, missed = 4 anchors, new held-out dir `heldout2/`, dev dir `cpg/eval/dev`) | 0 | missed-clusters 3, kept 222, fresh 0, discarded 0, served-anchor 0, moved-out mcp 16 / scripts 15 / wiring 5 / http 0 / flow 0 |
| `manifest` over `heldout2/` (same protocol string and 70 raw artifacts) | 0 | files 7, raw 70 |
| `verify` of the new manifest against `heldout2/` | 0 | checked 77, mismatches 0 |
| `cmp` dev `ac5-http`, `ac5-flow`, `ac24`, `ac24-r2` against the backup | 0 | byte-identical |
| `check-report` (this report; heldout2, dev lists, contact) | 0 | see below |

`gate1_problems` is not applicable to the new state by design (it compares against the old manifest and old lists). Equivalent checks, in memory: new held set (257 items incl. reserve) is a strict subset of the old (293); the 36 moved items are all in the new dev lists; held-out and dev item sets are disjoint (0 common).

## Counts

| Stratum | Moved to dev | Held-out items before → after | Held-out counting before → after | Dev items before → after |
|---|---|---|---|---|
| http | 0 | 26 → 26 | 26 → 26 | 18 → 18 |
| mcp | 16 | 46 → 30 | 46 → 30 | 20 → 36 |
| scripts | 15 | 79 → 64 | 79 → 64 | 49 → 64 |
| wiring | 5 | 8 → 3 | 8 → 3 | 11 → 16 |
| flow | 0 | 99 → 99 | 0 → 0 | 28 → 28 |

Reserve unchanged (35 counting items, 23 anchors). Held-out ac5 anchors per file: http 15, mcp 4→2, scripts 36→35, wiring 7→3, flow 12.

- **Clusters contacted:** 3 (4 anchors). Clusters per hit file: ac5 mcp, scripts, wiring items only; none in reserve. The three clusters are by primary stratum one each of mcp, scripts, wiring.
- **Cluster count c over the three gated strata:** no verb prints it. Derived in memory (not a verb output): held clusters holding at least one gated-stratum item 52, previously 55.
- **Floors (gated strata only, wiring and flow withheld):** pooled counting 26+30+64 = **120** against 75, met; per gated stratum 26, 30, 64, each at least 8, met. The verb's `minimums-met: false` reflects wiring at 3, which is withheld and exempt (stakeholder decision, close-out plan §Q-wiring).
- **25% cap:** denominator (counting scripts items before exclusion) is **64**; threshold 16, so 17 or more withheld refuses the spend (prereg text says 20 of 79). The re-gate fresh floor reports `floor-met: false` with 0 fresh items; that is the re-gate E12 check and does not apply to this first-gate list.
- **Held-out ac24:** 1 question hit. Left in held-out per EP §10 (contact does not apply to AC-24; AC-24 is not in this gate; the retry uses a fresh dev set at CO-11).

## Repo paths changed (git diff --stat)

- `cpg/eval/dev/ac5-mcp.jsonl` (+16 lines), `cpg/eval/dev/ac5-scripts.jsonl` (+15), `cpg/eval/dev/ac5-wiring.jsonl` (+5): additions only.
- `cpg/eval/heldout-manifest.json` (30 lines changed: three files' sha256/bytes/items/countingItems/anchors).
- New: this report. (`claude/docs/reviews/coordination-graph-guard.md` and the two `resume_teco_*.sh` were already modified/untracked before this unit.)

## Custody changes (outside the repo)

- New `heldout2/` (ac5 x5, reserve, `regate-flags.jsonl`, plus `ac24.jsonl` and `exposure.jsonl` copied unchanged). Old `heldout/` untouched (the backup).
- `backup-pre-contact-move/` in the custody dir: the previous repo manifest and previous `cpg/eval/dev/*`. Nothing deleted.

## What the gate needs (no harness edit)

The held-out list is a CLI argument: `gate_run.py run --lists <comma-separated files> --custodian <dir>` takes explicit paths under the custodian area; point it at the `heldout2/ac5-{http,mcp,scripts}.jsonl` files. No constant or default in `gate.py`, `gate_run.py` or `split_manifest.py` names `heldout/` (`gate.py` `verify_heldout` takes the dir as a parameter). `preflight` reads only the manifest's `protocol` field (equal). Not checked here: the S47 probe map the harness requires under the custodian area (`s47-probe-map.json`; `custodian-p1/` does not list it), and the prereg record's counts and dev-gold hash, which this move changes: the record, rehearsal and ledger line need redoing per prereg §4 item 6.

## Residual

The prereg's stated numbers (151 pooled, scripts 79, dev 18/20/49) are now stale (120, 64; dev 18/36/64). Rehearsal redo, a renewed once-permission for the next freeze-check, and the ledger line belong to the caller.
