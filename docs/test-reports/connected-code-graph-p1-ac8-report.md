# Connected code graph P1: AC-8 part 1 run report

> **Status:** active · **Owner:** `devops` · **Tracks:** P1-40 (P1)

Run date 2026-11-23, host WSL2, `strace` 6.8 present (`/usr/bin/strace`, nothing installed). The host-address probes (test_ac8.sh cases 3-5) and the `.invalid` DNS lookups were run under the stakeholder's authorisation of 2026-11-23, nothing beyond that file's probes.

## 1. Verdict

**P1-40 is NOT run. AC-8 part 1 stays not closed.** The shipped harness cannot produce the run P1-40 requires (`selfTest == []`); this is a scope question returned to the coordinator (section 4). The harness's own live test, `cpg/sandbox/tests/test_ac8.sh`, was run in full: 72 passed, 0 failed, 0 not run, exit 0.

## 2. Step 1: `cpg/sandbox/tests/test_ac8.sh` in full

| Test | Result observed |
|---|---|
| 1 detector control scenario | all checks ok (8 checks; 43 `conn` records, one labelled probe record, residue 0, rerun with same id refused) |
| 2 netprobe and throwaway network | all checks ok (no default route v4/v6, DNS cut, a reachable target is reported FAIL by the probe, nonce routing) |
| 3 main run (stub model, stand-in store; runs P-a host-address probe) | all ok; exit 3 `incomplete`, `selfTest` names the four stand-ins; zero `conn` records |
| 4 component that connects to the detector | all ok; exit 1, row 8.1 fails naming the `conn` |
| 5 no strace (`CCG_AC8_NO_STRACE=1`) | all ok; row 8.16 unrun, exit 3 |
| 6 run dies / TERM / refused by P-e | all ok; exit 70 / 143 / 1, cleanup verified |

Total: **72 passed, 0 failed, 0 not run.** No probe result other than unreachable was reported (test 3's P-a passed, zero `conn` records). `test_ac8_lib` was not run (no suites run by me); no flake observed in this file.

Cleanup verified: the test reported every non-`ccg-ac8-*` container unchanged, and afterwards `docker ps -a` and `docker network ls` show no `ccg-ac8-*` item. `falkordb-dev`, `ccg-falkordb`, `ccg-falkordb-tw-16390` are up and were not touched.

## 3. Step 2: the P1-40 row table (§8.2)

Rows below are the stand-in run of test 3 (what that test asserts), **not** P1-40. Test 3 is a self-test run (`stub-resolver`, `stub-model`, `store-standin`, `witness-parse`) and by the spec's verdict rule cannot be a pass.

| # | §8.2 status | Observed in test 3 (self-test run) | P1-40 result |
|---|---|---|---|
| 8.1 | computed | pass | unrun |
| 8.2 | vacuous | vacuous | vacuous (nothing can fail it under option C) |
| 8.3 | computed (negative); positive half vacuous | pass | unrun (positive half vacuous) |
| 8.4 | vacuous | vacuous | vacuous |
| 8.5 | computed; proxied half vacuous | pass | unrun (proxied half vacuous) |
| 8.6 | computed | pass | unrun |
| 8.7 | computed | pass | unrun |
| 8.8 | computed | pass | unrun |
| 8.9 | computed | pass | unrun |
| 8.10 | computed | pass | unrun |
| 8.11 | computed | pass | unrun |
| 8.12 | vacuous | vacuous | vacuous |
| 8.13 | vacuous; residue computed | pass | unrun (residue); rest vacuous |
| 8.14 | computed | pass | unrun |
| 8.15 | (i) computed; (ii), (iii) vacuous | pass | unrun (i); (ii), (iii) vacuous |
| 8.16 | computed with strace | pass (strace present); also unrun in test 5 as designed | unrun |
| close pairing | added row | pass | unrun |

"Pass" in the third column is the test's own stand-in evidence that the harness computes the rows, not a result for the real components, store, resolver or model. Counts for the P1-40 column: 0 computed, 4 vacuous (8.2, 8.4, 8.12 whole; 8.13 mostly), 13 unrun.

## 4. Why P1-40 was not run

`docs/plans/connected-code-graph-egress-spec3.md` §8.1 "P1-31 and P1-40" lists what P1-40 adds: (a) a harness-created real store with `REPLICAOF`/`SLAVEOF`, `MIGRATE`, `MODULE` disabled, (b) a mode probing the real P1-25/P1-26 containers in place, (c) a probe container with a fixed address, classified by source IP, (d) a real run with `selfTest == []`. `cpg/sandbox/ac8/run.sh` implements none of these: `--store-standin` is mandatory (line 249, `die` without it), `--store-container`/`--store-port` are refused (line 395), and the `selfTest` list always includes `store-standin` (line 266). Building them is implementation work, not a run, and the acceptance (exit 0, `selfTest == []`) cannot be reached by any invocation of the existing harness. Using `--workload` to drive real components would run them unconfined on the host, outside the measured scenario, and would still carry `store-standin`.

Decision for the coordinator:
- **A (recommended):** dispatch a build unit (`devops`, with `security-expert` advisory) for (a)-(d) in `run.sh`/`lib.sh`, then re-dispatch P1-40. Needs the real P1-25/P1-26 container images and a real-store image; not checked here.
- **B:** record AC-8 part 1 as "harness live-tested, P1-40 blocked on missing harness options" and carry it.
- **C:** accept an `incomplete` stand-in run as the P1-40 evidence; contradicts the spec's acceptance, not recommended.

## 5. Notes

- The earlier "no, later" decline is replaced by the 2026-11-23 authorisation; the host-probe part of AC-8 is therefore no longer withheld, only the real-component run is blocked. Outcome (2026-11-24): the real-component run is carried to P2 by stakeholder decision (the component image and an in-scenario local-model path do not exist); AC-8 part 1 is reported not closed at P1 (harness live-tested only); the real-component run is carried to the P1 close-out unit. The user confirmed on 2026-11-24 that the host probes were authorised.
- Row 8.16's `strace` route: the gates doc says `devops` installs it; it was already present, nothing installed.
