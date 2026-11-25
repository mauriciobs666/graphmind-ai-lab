# Connected code graph P1 — AC-5 held-out gate run 1 at R3 (`ac5-gate-1-r3`): refused twice, list not spent

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1 close-out, CO-6b)

CPG: not applicable — a black-box execution of the committed AC-5 gate harness; no code-level analysis.

## Verdict

**Refused twice, list not spent.** Both attempts ended at exit 2 with the same harness refusal after a full build; no marker was taken either time; no hits were scored, so there are no per-stratum counts, floor figures, hit@100 or burden counts. Root cause: `docs/reviews/connected-code-graph-p1-gate-snapshot-rca.md` (a per-build random HMAC key makes the snapshot id differ by construction, so a dev probe from another build can never match).

Refusal text (identical both times): "REFUSED: the dev-gold probe vouches for another snapshot than the one this run built: rehearse again on this build. the list is not spent (no marker was taken), but the built graph stays in this store and the store refuses every later fresh run: for the next run provision a new empty store (create its ACL users again) and rebuild (a fresh build takes 16-21 minutes)."

## Common setup and command

Committed-tree clone, detached checkout; private config/credential dirs mode 700; throwaway store; 4 ACL users. Command: `gate_run.py run --gate-id ac5-gate-1-r3 --lists <heldout2>/ac5-http.jsonl,<heldout2>/ac5-mcp.jsonl,<heldout2>/ac5-scripts.jsonl --custodian ~/.local/share/ccg-eval/cf7b2042cce7 --repo <clone> --system graphmind-ai-lab --host 127.0.0.1 --port <port> --dev-probe <t3 dev-probe.json>` (probe map default; `--allow-after-adaptation` not used; `--lists` is comma-separated, an earlier space-separated invocation was rejected by argparse at once, exit 2, 0 s, no store contact). Probe snapshot id: `a965482a84f90cecff7b66a484642e38` (t3 rehearsal). Preconditions both times: HEAD descends from `31f60bf9`; `git diff --name-only 31f60bf9 HEAD -- cpg` outside `cpg/eval/` and `*.md` empty; `git status --short cpg` clean; forbidden ports 6379, 6390, 16390, 16400 and the running services untouched. Scratch: `.../scratchpad/gate-run-1/` and `.../gate-run-2/`.

## Attempt 1

Clone `9fec76e9`, store port 16630. Exit 2 after 901 s. Refused (snapshot mismatch). No marker. Store torn down.

## Attempt 2

Clone `337910eb` (tree `277a2cb4...`, equal to that commit's tree; the commit the t3 rehearsal used), store port 16640, fresh private dirs. Started 17:23. Exit 2 after 916 s, same refusal. No marker. Store `ccg-falkordb-tw-16640` torn down after the RCA finished reading it; remaining containers: `ccg-falkordb`, `falkordb-dev`, `ccg-falkordb-tw-16390`, `hopeful_brattain`, all untouched.

## Results

| Item | Attempt 1 | Attempt 2 |
|---|---|---|
| Exit status | 2 (REFUSED) | 2 (REFUSED) |
| Wall time | 901 s | 916 s |
| Marker / run id | none | none |
| Hits, counting items, withheld-excluded per stratum; pooled vs floor; hit@100; burden | not produced | not produced |
| List spent | no | no |

## State

Custodian root still holds, from the refused runs, the `gate-run-5f1dd650...lock` file and `gate-work-`, `gate-build-`, `gate-audit-`, `gate-keys-5f1dd650...` entries (same hash suffix both times); not removed by me. No process held the lock at the start of attempt 2. No commits; no further gate runs.
