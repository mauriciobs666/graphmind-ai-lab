# model-bench

A standalone, human-started harness that measures **one local model at a time** against **one named
task pack**, stores the result with a full environment fingerprint, and compares it against
previously stored results **for the same role** — with confidence intervals and a visible flag
whenever a comparison is not apples-to-apples.

The point is not a score. The point is that a result taken today still lines up against a model
tested months ago, on this hardware and this data, instead of against a published leaderboard number
for a task nobody here runs.

**The live spec** — what the harness does, command by command, and every file format — is
`docs/DESIGN.md` (read whole) and `docs/FORMATS.md` (read by lookup). This README is how to run it.
End-user walkthroughs are in `docs/manuals/small-model-benchmarking.md`; what shipped and when is
`docs/HISTORY.md`; open work is `docs/BACKLOG.md`. The statistical method is the note
`docs/plans/small-model-benchmarking-ml.md` (repo root).

## Three things this tool deliberately is not

- **No CI hook, no scheduler, no timer.** `model-bench` only ever runs because a person typed a
  command. Nothing here is wired into any other component's test suite.
- **No pass/fail gate.** There is no threshold of its own; a decision-point report states the
  requirement's bar and never acts on it. The only non-zero exits are operational
  (`docs/DESIGN.md` §2.2).
- **No leaderboard and no cross-role aggregate number.** Comparison is always within one pack, and
  that is enforced by the shape of the code, not by a convention anyone has to remember.

It is also **standalone**: no runtime code path reads anything outside `model-bench/`. Golden data
that originates elsewhere is copied in, versioned here, and carries its provenance
(`docs/DESIGN.md` §1.3).

## Quick start

Run everything with `model-bench/` as the working directory — the repo has no root pytest
configuration, so from the repo root pytest ignores this component's `testpaths` and walks into
other components' suites.

```bash
./setup.sh                        # create .venv, install the package + dev extra
.venv/bin/python -m pytest -q     # the suite (network-free); `-m live` opts into real LM Studio
.venv/bin/ruff check .            # lint
./run.sh --help                   # the CLI
```

## Commands

Semantics, refusals and exit codes are `docs/DESIGN.md` §2.

```bash
./run.sh attest [--api-base-url <url>] [--set key=value ...]   # write host.json (once, and after an LM Studio update)
./run.sh validate --pack <path>                 # structural pack check, no LM Studio needed
./run.sh run --pack <id> (--model <key> | --candidate <id>) [--session <id>]
    [--warmup <n>] [--first-call-timeout <s>] [--request-timeout <s>]
./run.sh compare --pack <id> [--models a,b] [--session <id>] [--negative-control] [--out <path>]
./run.sh rank --pack <id> [--session <id>] [--reference <key>] [--footprints <path.json>]
./run.sh models --tested [--pack <id>] [--role <role>]
./run.sh index rebuild                           # regenerate results/index.csv
```

A decision-point pack (`packs/*-synthetic/`) is run with `--candidate` and compared with
`--session` and `--models`; see `docs/DESIGN.md` §7 and §9.4. `run-sweep.sh` and `scripts/` are
operator tooling (`docs/DESIGN.md` §2.10).

## Adding a pack

A pack is a directory under `packs/<pack-id>/` declared by `pack.json`; its format per role is
`docs/FORMATS.md` §1, and the rules `validate` enforces are `docs/DESIGN.md` §5.2.

1. Create `packs/<new-id>/` and write `pack.json` declaring the role, the scorer name and the
   sampling and metrics contract.
2. Copy in golden data with real, verified provenance and a `PROVENANCE.md`; never fabricate it.
3. Implement (or reuse) a scorer module under `modelbench/scoring/` for the role
   (`docs/DESIGN.md` §3.7, §6).
4. Run `./run.sh validate --pack packs/<new-id>`, then drive it live with `./run.sh run`.

Any change to a pack's bytes — a golden item, a prompt, a tool schema — is a new content hash, so
old and new results are distinguishable and never silently paired.

## Stored results

A run is stored as `results/runs/<runId>.json`; `results/index.csv` is derived. The record shape
and the schema-version rule are `docs/FORMATS.md` §2 to §4.
