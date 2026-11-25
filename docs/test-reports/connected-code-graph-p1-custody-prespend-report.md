# ccg P1 AC-5 gate: custody pre-spend housekeeping at R2 (counts and booleans only)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1)

CPG: not applicable — file-level custody housekeeping, no code-level analysis.

## Summary

Run at record R2 = `67ea5fd4`, file-level only (no file content opened, no freeze-check, no `gate.py run`, no engine, no build). **Stopped at item 2: the S47 probe map is absent from the place the harness looks.** Items 3 and 4 were not run (stop rule). Nothing was created, deleted or opened.

## Results

| Item | Result |
|---|---|
| 1 Retire old custody dir | DONE. `heldout/` renamed to `heldout-RETIRED-pre-contact-move/` (same custodian root `~/.local/share/ccg-eval/cf7b2042cce7/`). 8 files before, 8 after, same names, sizes and mtimes; none opened. `heldout/` no longer exists; `heldout2/` untouched (9 files). |
| 2 S47 probe map present | **ABSENT** (see below). |
| 3 Negative-control preflight at `ac5-gate-1` | NOT RUN (stop rule). Code read: the entry point is `cpg/eval/tools/gate.py verify --gate-id <id> [--repo] [--record-commit]` (`gate.preflight`; exit 0 only if label is first-contact, else 1). It takes no `--lists` argument and by its code opens no held-out content (git state, prereg line, manifest pins, harness hashes), so it is pure and can be run in a follow-up; it cannot be given the `heldout2/` lists. |
| 4 sha256 comparison with R2 pins | NOT RUN (stop rule). |

## Item 1: custody dir listing (names, sizes in bytes)

Before (`heldout/`): ac24.jsonl 20500; ac5-flow.jsonl 194134; ac5-http.jsonl 73588; ac5-mcp.jsonl 106216; ac5-reserve.jsonl 75960; ac5-scripts.jsonl 158076; ac5-wiring.jsonl 26202; exposure.jsonl 0.

After (`heldout-RETIRED-pre-contact-move/`): identical 8 names and sizes. Custodian root now holds: backup-pre-contact-move, briefs, custodian-p1, heldout-RETIRED-pre-contact-move, heldout2, kit-src, oracle, roles, runs, stakeholder-s36, work (no `heldout`).

`heldout2/` (unchanged): ac24 20500; ac5-flow 194134; ac5-http 73588; ac5-mcp 80164; ac5-reserve 75960; ac5-scripts 114434; ac5-wiring 7871; exposure 0; regate-flags 13320.

## Item 2: S47 probe map

- Expected by the code: `gate_run.py` `PROBE_MAP_NAME = "s47-probe-map.json"`, default `<--custodian>/s47-probe-map.json` (`gate_run._run_gate`), overridable with `run --probe-map <path>`; the path must lie under `~/.local/share/ccg-eval/` (`G.within_custodian`, `G.read_probe_map`). Absent means `run` refuses before the build ("the S47 probe map is missing under the custodian root").
- Found: `~/.local/share/ccg-eval/cf7b2042cce7/s47-probe-map.json` absent; `heldout2/s47-probe-map.json` absent; `custodian-p1/s47-probe-map.json` absent.
- Related but not the same name: `~/.local/share/ccg-eval/cf7b2042cce7/work/probe-map.json`, 221 bytes, mtime 2026-11-22 10:28 (mode 0600), older than the contact move. Whether it is the S47 `{unit anchorId: probe cpId}` map was not checked (content not opened); a custodian decision whether to copy it or pass it via `--probe-map`.
- Which verb creates it: none. No CLI verb writes it; it is the return value `split_manifest.split(...)["probe"]` (library call), written custodian-side by a custodian script (`gate.py` `read_probe_map` docstring). The `freeze-check --probe-map` option writes a different file (hash to itemId). `regate-list` (used for the contact move) does not write it.
- Open point for `teco`/custodian: the map is keyed by S47 unit anchors; the contact move only removed items, but whether the existing map still covers every unit anchor of `heldout2/` is a custodian check.

## Not run, to resume

Items 3 and 4 are independent of item 2 and read-only; re-dispatch them or run them directly once the map question is settled. Run item 3 as `gate.py verify --gate-id ac5-gate-1` (expected to refuse: exit 1).
