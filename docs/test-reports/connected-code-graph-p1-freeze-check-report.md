# ccg P1 AC-5 gate: custodian freeze-check at R

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1)

CPG: not applicable — a procedural custodian run of the gate's own pre-spend verb, no code-level analysis.

## Summary

One run of `split_manifest.py freeze-check` at the record commit R = `f4ef2239`, under the stakeholder's 2026-11-25 once-permission (counts and 12-hex hashes only). **Verdict: NOT CLEAN. Exit 1, 7 hits.** Per the prereg §4 item 6 a hit after the rehearsal moves a cluster to dev and changes c, the floors, the withheld counts and the dev gold hash. The rehearsal is then redone and the once-permission renewed. No held-out file was opened or printed by the operator, and no other held-out verb was run. The `--out` and `--probe-map` files sit outside the repo and were not read.

## Preconditions

- `git rev-parse HEAD` = `bb0cf2421dbff9dcebc6b969ebd4fe8cf4d9190f`, a descendant of R (`merge-base --is-ancestor` ok).
- `git diff --name-only f4ef2239 HEAD -- cpg` listing nothing outside `cpg/eval/` or `*.md`: empty.
- `git status --short cpg`: clean. (Repo-wide: `claude/docs/reviews/coordination-graph-guard.md` modified, two untracked `resume_teco_*.sh`, all outside `cpg/`.)

## Command (run from the repo root, once)

```
python3 cpg/eval/tools/split_manifest.py freeze-check --record f4ef2239 --repo . \
  --heldout $C/heldout/ac5-flow.jsonl $C/heldout/ac5-http.jsonl $C/heldout/ac5-mcp.jsonl $C/heldout/ac5-reserve.jsonl $C/heldout/ac5-scripts.jsonl $C/heldout/ac5-wiring.jsonl $C/heldout/ac24.jsonl \
  --dev cpg/eval/dev/ac24-r2.jsonl cpg/eval/dev/ac24.jsonl cpg/eval/dev/ac5-flow.jsonl cpg/eval/dev/ac5-http.jsonl cpg/eval/dev/ac5-mcp.jsonl cpg/eval/dev/ac5-scripts.jsonl cpg/eval/dev/ac5-wiring.jsonl \
  --contact $C/work/contact.jsonl \
  --out <scratchpad>/freeze-out.json --probe-map <scratchpad>/probe-map.json --positive-control
```

`C` = `~/.local/share/ccg-eval/cf7b2042cce7`; `<scratchpad>` = `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad` (outside the repo). The held-out list is the one `check_heldout_list` requires (six ac5 files plus reserve, and `ac24.jsonl` because it exists). `--s47` took its default `511b14cf`. `--positive-control` was added by the operator, not named in the brief: it is part of the same verb, same process and the same single held-out read. It only adds the line `control: ok`. The rehearsal reports contain no earlier freeze-check invocation to copy.

## Output (verbatim, exit status 1, stderr empty)

```
diff-paths: 1931
excluded: {"custodian-output":7,"non-p1-commits":1491}
deleted: 2
checked: 431
failed: 0
probes: 1759
hits: 7
hit: claude/graph-dba/falkordb-quirks.md 92234570806a
hit: cpg/tests/golden/system-units.json 23df7dada3be
hit: cpg/tests/golden/system-units.json e30d9bb150bb
hit: cpg/tests/golden/system-units.json a985f86444fb
hit: cpg/tests/golden/system-units.json 44442ad79e02
hit: cpg/tests/test_eval_tools.py 7d8b97fbf4b9
hit: docs/plans/connected-code-graph-bundle-spec3.md c4e35747a8e0
control: ok
```

## Reading

- **Not clean:** 7 hits (7 distinct probe hashes) in 4 files, positive control ok, 0 failed files. Exit 1 means hits and no failures.
- A hit is a held-out probe (itemId-derived 12-hex hash) found as text in a file committed at R within the P1 path set, per the verb's `check_text` against held-out, dev and contact material. The hash-to-itemId map is in the out-of-repo probe map, unread.
- **Cluster count c: not printed.** The verb prints no c and no cluster information. Item six's remaining needs from this run are the exit status (1), `hits: 7`, `probes: 1759` and the path/hash lines above. Which cluster each hash belongs to needs the probe map, a held-out-derived read outside this authorisation.
- The 4 files: `claude/graph-dba/falkordb-quirks.md`, `cpg/tests/golden/system-units.json` (4 hashes), `cpg/tests/test_eval_tools.py`, `docs/plans/connected-code-graph-bundle-spec3.md`. The system-units golden and the test file may be tied to dev or fixture material rather than builder knowledge of held-out items. That judgment needs the probe map and belongs to the custodian.

## Required next step (not taken here)

Stop. The custodian and `teco` decide per prereg §4 item 6: treat each hit's cluster as moved to dev, recompute c, floors, withheld counts and the dev gold hash, redo the rehearsal, and obtain a renewed once-permission from the stakeholder before any further freeze-check. This run spent the current permission. Nothing was committed.
