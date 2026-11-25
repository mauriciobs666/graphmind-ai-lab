# Connected code graph P1: freeze-check at R3 (`ac5-gate-1-r3`)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1 close-out)

Custodian run of the one authorised `split_manifest.py freeze-check` at R3 = commit `31f60bf9` (2026-11-25). Counts and 12-hex hashes only. Authority: prereg `docs/plans/connected-code-graph-p1-closeout-prereg.md` §4 item 6; template `docs/test-reports/connected-code-graph-p1-freeze-check-report.md`.

CPG: not applicable — a custodian run of one eval-tool verb over committed text, no code-level analysis.

## Verdict

**Not clean: exit 1, 1 hit.** The verb printed one probe hash found in one committed document. Per the authorisation, the S47 probe map was **not generated** and no second freeze-check was run. Stopped and reported.

## Preconditions (a)

- HEAD `2d101714a05975665ef3dcc5200a94e7283cdaaa`; `git merge-base --is-ancestor 31f60bf9 HEAD` true.
- `git diff --name-only 31f60bf9 HEAD -- cpg` lists nothing outside `cpg/eval/` or `*.md` (the filter output was empty).
- `git status --short cpg` empty (clean).

## Command (run from the repo root, once)

```
python3 cpg/eval/tools/split_manifest.py freeze-check --record 31f60bf9 --repo . \
  --heldout $H/ac5-flow.jsonl $H/ac5-http.jsonl $H/ac5-mcp.jsonl $H/ac5-reserve.jsonl $H/ac5-scripts.jsonl $H/ac5-wiring.jsonl $H/ac24.jsonl \
  --dev cpg/eval/dev/ac24-r2.jsonl cpg/eval/dev/ac24.jsonl cpg/eval/dev/ac5-flow.jsonl cpg/eval/dev/ac5-http.jsonl cpg/eval/dev/ac5-mcp.jsonl cpg/eval/dev/ac5-scripts.jsonl cpg/eval/dev/ac5-wiring.jsonl \
  --contact $C/work/contact.jsonl \
  --out $S/freeze-out.json --probe-map $S/probe-map.json --positive-control
```

`C` = `~/.local/share/ccg-eval/cf7b2042cce7`; `H` = `$C/heldout2` (the retired `heldout/` not used); `S` = `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad/co9-t3/custodian` (outside the repo). The held-out list is the one `check_heldout_list` requires: six ac5 files plus reserve, plus `ac24.jsonl` because it exists. `--s47` took its default. `--positive-control` is part of the same verb and the same process. The run exceeded the 120 s foreground limit and finished in the background; it was started once and not repeated.

## Output (verbatim, exit status 1, stderr empty)

```
diff-paths: 1942
excluded: {"custodian-output":10,"non-p1-commits":1499}
deleted: 2
checked: 431
failed: 0
probes: 1585
hits: 1
hit: docs/plans/connected-code-graph-bundle-spec3.md c4e35747a8e0
control: ok
```

## Reading

- Exit 1 means hits and no failures: 1 hit (1 distinct probe hash) in 1 file, 0 failed files, positive control ok. Probes 1585 (held-out and contact material at R3; 1759 at R).
- The hit is `docs/plans/connected-code-graph-bundle-spec3.md`, hash `c4e35747a8e0`. The same path and hash appeared in the earlier run at `f4ef2239` (it was one of its 7 hits); the other 6 hits of that run are gone, consistent with the contact move. The hash-to-itemId map sits in the out-of-repo `probe-map.json`, unread. Whether the hit is a builder contact with a held-out item, and which cluster it belongs to, needs that map and belongs to the custodian or the stakeholder; the prereg says a contact hit moves a cluster to dev and changes c, the floors, the withheld counts and the dev gold hash, so the rehearsal is redone and the permission renewed.
- "Clean" in the spend checklist means no probe literal in the path set. It is not met.

## S47 probe map (step c)

**Not generated.** The condition was exit 0; the exit was 1. `s47-probe-map.json` remains absent from the custodian root. Note for later: the verb's `--probe-map` output is the hash-to-itemId reverse map, not the S47 unit map (`{unit anchorId: probe cpId}` from `split_manifest.split(records, units, contact_ids, eval_commit)["probe"]`, as `gate.read_probe_map` reads it), so it cannot stand in for it. Generating it needs the match/context/contact inputs and the eval commit, an in-process read of held-out-derived data.

## Ledger values (c, total w, fresh scripts supply, cluster-id list sha256)

**Not produced.** The verb prints none of them, and computing them needs a custodian in-memory read of held-out data beyond the authorised verb. With the freeze-check not clean they are moot until the contact is resolved and the freeze-check repeated under a renewed permission.

## Not done, by authorisation

No second freeze-check, no other held-out verb, no read of held-out content, custodian files or the `--out` / `--probe-map` outputs, and no itemId, path literal or probe literal printed beyond the verb's own hit line. Nothing committed. `git status --short cpg` is unchanged by this run; the only repo change is this report.

## Required next step

`teco` and the stakeholder decide on the one hit: custodian resolution via the (unread) probe map to name the cluster, then per prereg §4 item 6 the contact move, a redone rehearsal and a new once-permission for the next freeze-check. The out-of-repo `freeze-out.json` and `probe-map.json` in `$S` are retained for that custodian step.

## Custodian lookup of hit `c4e35747a8e0` (2026-11-25)

Authorised by the user in-session; names and counts only. A script read the out-of-repo `probe-map.json` entry for the hash and tested membership of its item id against the seven lists under `heldout2/` (no verb of `split_manifest.py` run, no custodian file modified).

- Hash `c4e35747a8e0` maps to **1** item.
- That item belongs to the **`ac24`** held-out list only. It is in none of `ac5-http`, `ac5-mcp`, `ac5-scripts`, `ac5-flow`, `ac5-wiring`, `ac5-reserve` (0 items in each).
- Distinct ac5 clusters: **0** (the item is not on an ac5 list; the cluster question does not arise).
