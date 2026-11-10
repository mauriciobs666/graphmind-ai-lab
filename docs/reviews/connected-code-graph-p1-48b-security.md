# Security review: P1-48b runner indexer entries (C-1001 P1)

> **Status:** active · **Owner:** `security-expert` · **Tracks:** C-1001 (P1)

## 1. Scope & verdict

Lens: code/app security (advisory, no exploitation). Artifact: uncommitted `cpg/sandbox/offline-run.sh`, `forbidden-flags.txt`, `tests/*`, `README.md`, against `docs/plans/connected-code-graph-egress-spec3.md` §7.4, §7.5, §7.6, §10.1, R-E9, R-E10. `cpg/ccg/broker/*` not reviewed; the `analyst` correctness pass is not duplicated. Earlier advisories (`-spec3-indexer-security.md`, `-spec3-security.md`) were taken as baseline.

Method: static read of the runner, offline probes by sourcing the runner's own functions (`sbx_indexer_grammar`, `sbx_audit_elements`) on 36 argv cases, and throwaway `--network none` containers on `ccg-toolchain-indexer:73b9fb76bd2a`. Nothing ran against `falkordb-dev` or any service; no tree-mutating git commands.

**Verdict: approve with suggestions.** No blocker. One major (runner trusts the stager for symlink containment of `--cwd`/`--environment`), the rest minor/nit.

CPG: not applicable — the artifact is a bash runner script, which Joern does not model and no `cpg_*` graph exists for; the traces were done by running the functions.

## 2. Findings

### M1 (major, confirmed): `--cwd`/`--environment` containment is textual only; an in-tree symlink reaches the whole image, unbounded

`sbx_idx_path_ok` (`offline-run.sh:251-262`) checks the string shape only. `--cwd /` and `--cwd /src/..` are refused (confirmed), but `--cwd /src/lnk` is accepted when `lnk -> /` exists in the staged tree. Reproduced: a tree with `lnk -> /` and `rel -> ../../..` mounted ro at `/src`; in the container `cd /src/lnk && pwd -P` printed `/` and listed the image root; `/src/rel` also resolves to `/`. This is the R-E10 hazard (whole-image `scip-python` run, no time or memory limit) by a path the grammar cannot see. `--environment /src/lnk/etc/...` likewise reads arbitrary image files (image content only; low impact).

The only control is P1-15's stager rejecting outward symlinks (§7.4), which does not exist in the reviewed scope and is not tested here (no symlink test in `tests/test_sandbox.sh`). Runner-level defence in depth is cheap and fail-closed, because the host has `$src` resolved already (`offline-run.sh:853`).

Suggested improvement: after the grammar passes, for each `--cwd`/`--environment` value `v`, take `rel=${v#/src}`, `r=$(realpath -e -- "$src$rel")`, require `$r == "$src" || $r == "$src"/*` (and `-d` for `--cwd`), else refuse 64 `forbidden-flag`. Host resolution of an absolute symlink gives the host path, so it is refused (conservative; right for a stager that already forbids outward links). Add a test: `lnk -> /`, `lnk -> ../..`, `lnk -> /src/sub` each refused, a plain subdirectory accepted. Remaining TOCTOU (tree mutated after the check) is the stager/operator's staged-tree ownership, not the runner's.

### m1 (minor, confirmed): label check is a wrong-image guard, not an integrity control, and checks a mutable name

`sbx_try_container` (`:405-408`) reads the label via `docker image inspect "$SBX_IMAGE"` and launches `"$SBX_IMAGE"` by name (`:484`), while `SBX_IMAGE_ID` was captured earlier and only recorded. A retag between inspect and `docker run` swaps the image after the check; and any image can carry the label, since `CCG_TOOLCHAIN_IMAGE` is caller-chosen. The fixed path `/opt/indexer/bin/<name>`, never `PATH`, is right (`env -i` execs the absolute path; the image `PATH` has only root-owned dirs, no `/src` or `/tmp`; verified from the image Env). So the label stops a mistake, not an adversary; the spec and README should say so, since the caller already has the host's Docker socket.

Suggested improvement: launch with `"$SBX_IMAGE_ID"` (the sha256 id) instead of the name, so the checked image is the run image. State in README §7.4 that the label is a guard against a wrong image only.

### m2 (minor): path components have no character class

A component may contain a newline, glob characters, spaces, control bytes: `--cwd $'/src/a\n'` and `--cwd '/src/*'` pass the grammar (probe). Docker argv is not shell-expanded, so this is not an injection; it is noise in the audit record and a looser grammar than needed (`/src/--infer-tsconfig` also false-positives against the forbidden list via the substring rule, harmless).

Suggested improvement: require each component to match `^[A-Za-z0-9._@+-]+$` (not `.`/`..`), in the same loop; @AUDIT_ID@ is substituted before the grammar runs, so it still passes.

### m3 (minor, confirmed): a crashed indexer leaves a partial artifact in `/out`

The no-`.git`, no-`--project-version` crash exited 1 but left a 39-byte `o.scip` in the output directory. The runner returns the tool's rc correctly; the risk is a consumer that globs `<auditId>.scip`. Suggested improvement: P1-15 must treat rc != 0 as "no artifact" and delete or ignore the file; note it in §7.8.

### n1 (nit): the four `scip-typescript` forbidden entries close nothing the grammar does not already close

`--pnpm-workspaces`, `--yarn-workspaces`, `--yarn-berry-workspaces`, `--infer-tsconfig` are all refused as unknown elements by the closed grammar (probe: `--infer-tsconfig` gave 2 hits). They are sweep and drift-alarm entries, and as such correct. Real paths: the workspace flags spawn `pnpm`/`yarn`; the image has `corepack` in `/opt/indexer/node/bin` but no `pnpm`/`yarn` shim on `PATH`, so they would fail (and could not download, `--network none`). `--infer-tsconfig` writes into the ro `/src`. Nothing network-shaped is missing from either tool's `index --help` (checked live: scip-typescript lists `--cwd`, `--output`, the workspace flags, `--infer-tsconfig`, `--no-global-caches`, `--max-file-byte-size`, progress flags; scip-python lists `--project-name/-version/-namespace`, `--cwd`, `--target-only`, `--output`, `--quiet`, `--environment`, `--dev`). The `[projects...]` positional of scip-typescript is refused (probe), which is correct and worth a test.

## 3. Argv grammar: verdict on the probes

Closed in all 36 cases: `--flag=value` (hit), repeated `--cwd` (hit), `@resp` (hit), abbreviations `--cw`/`--env` (hit), `--target-only`, `--quiet`, `-J-Xmx1g`, `--` and a second `index` (hit), uppercase `INDEX` (hit), a value-position swallowing (`--cwd --environment /src`, `--project-name --cwd`: hit), empty and `-`-leading project names (hit), `/src//a`, `/src/a/`, `/src/./a`, `/src/../x`, `/srcx` (hit); `/src` and `/src/` accepted as one root, as specified. Caller `--output`/`-o`/`--output=` is refused before the indexer branch (`:884-887`) and the runner appends its own. Gap: only the symlink route (M1).

## 4. PYTHONSAFEPATH

Confirmed effective on the image's Python 3.12.3 (the variable needs 3.11+): with a hostile `json.py` in the cwd, `python3 -c 'import json'` printed `SHADOW EXEC` without the variable and loaded `/usr/lib/python3.12/json/__init__.py` with it. The pass-list adds it only for indexer entries (`:319`). If the image Python is ever below 3.11 the variable is silently inert; suggest a build smoke that asserts `python3 -c 'import sys; assert sys.flags.safe_path'` with the variable set. Related: hostile `.git/config` (`core.fsmonitor`, `alias.rev-parse`, `hooksPath`) did not execute anything under `git rev-parse HEAD` in the image (probe, nothing was created).

## 5. `controlled` / namespace / `auto`

Correct as built: indexer with `controlled` is refused 64 `unknown-frontend` before the generic `mode-not-implemented` (`:846`); `namespace` yields a recorded 69 `backend-unavailable` after the argv checks, and `auto` resolves to container only (`:825-829`), so there is no fall-through to a host backend for an indexer. 64 outranks 69 as specified. No finding.

## 6. Ruling: `--project-version` in the grammar, or `.git` in the staged tree

**Admit `--project-version`; do not stage a `.git`.** Confirmed live: without either, `scip-python` 0.6.8 crashed (rc 1); with `--project-version 1.0.0` it wrote the index (rc 0) offline.

Reasons: (1) it is one more audited, closed, machine-checked element, whereas a `.git` is an unaudited repo-derived directory (config, packed objects, a `gitdir:` pointer) inside the tree that `git` then executes against; the hostile-config probe shows `rev-parse HEAD` runs nothing today, but that is a property of one git version and one subcommand, and R-E9 already lists tool children as unaudited. (2) A valid `.git` needs a real commit object, so the stager would have to run `git init`/`commit` (hooks, config, identity) or hand-craft objects. (3) A target's own `.git` carries history and remotes, which must not enter a staged tree.

Required form: `--project-version <v>`, `scip-python` only, at most once, two separate elements, with `<v>` full-matching `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$` (the `--project-name` pattern; it admits a 40-hex commit id and a 64-hex sha256, no leading `-`, no whitespace, no `=`, so it cannot be read as an option). The stager should supply a deterministic value (the pinned commit, or a digest of the staged tree), never a timestamp, so the index is reproducible. Add the same tests as for `--project-name` (empty, `-x`, 65 chars, repeated, `--project-version=1`, on `scip-typescript` it is a hit), and update `README.md` and spec §7.4 and the `usage` list. Keep it optional, not mandatory, so a tree that does carry a revision still works.

## 7. What is solid

Closed grammar with no prefix or `=` forms; runner-owned `--output`; absolute indexer path with `env -i`; image `PATH` free of writable dirs; fail-closed label and backend handling; `PYTHONSAFEPATH` scoped to indexers; 64-over-69 precedence; mutants cover the added branches.

## 8. Open questions

- `devops`/`architect`: is a runner-side wall-clock cap for indexers (a `timeout` wrapper around the docker client with the existing signal cleanup) worth taking now, given R-E10 only has P1-15's timeout and M1 shows how an unbounded run can still be reached? Left to the measured-need rule; M1 removes the known route.
- `architect`: confirm P1-15's stager rejects outward symlinks (and covers `--environment`'s file) as §7.4 says; M1 assumes it but the runner cannot see it.

## Pass 2 (fix round; spec `docs/plans/connected-code-graph-egress-spec4.md` §1)

**Verdict: approve with suggestions.** No blocker; the major is fixed. Re-run on the tree: 23 walk cases through `sbx_idx_walk`, three end-to-end runner invocations (throwaway `--network none` container, scratch audit dir), one FIFO container probe.

- **M1: fixed.** Refused: `lnk -> /`, `up -> ../..`, `abs1 -> /src/sub`, an absolute link carrying the host path, `lnk2 -> deep/abs` (absolute link reached through a relative one), dangling, missing, a link chain escaping via `..`, a link loop, `sub/../../..`, and `/src/lnk/etc`. Accepted: `/src`, `/src/`, plain subdirectory, internal relative links (`rel_ok`, `chain -> sub/back -> ../real`, `self`). End to end: `--cwd /src/lnk` with `--project-version` missing is `forbidden-flag` (hits `--cwd /src/lnk`), so containment outranks `usage` per spec4. Walk logic is sound: `cur` only ever holds a non-link path, so `..` pops a real parent and relative targets resolve against the right directory; hops are capped at 40. Residual, reasoned not exploited:
  - TOCTOU: a link swapped after the check, before or during the run, defeats it, since `/src` is a live bind of a host directory. Spec4 assigns this to the stager/operator; acceptable only if the staged tree is runner-uid-private and not writable by another process. Say so in the README.
  - Hardlinks: not directories, so no walk escape; file content only (`env.json`), the stager's reject-list covers them.
  - Submounts under the source: the non-recursive bind shows the container an empty directory where the host walk sees content; the container sees less, so safe.
- **m-new (minor, confirmed): `--environment` is accepted when the target is not a regular file.** A FIFO `/src/fifo` passes the walk (`-e`) and `cat` of it blocks until killed (probe: rc 124 on a 5 s guard). That is an unbounded run (R-E10), reachable only if the stager misses a FIFO. Fix: require `-f` for `--environment` in `sbx_idx_contained`.
- **m1: fixed.** `SBX_IMAGE` (the name) is used once, to inspect the ID (`offline-run.sh:441`); env read, label check and launch all use `SBX_IMAGE_ID` (`:442`, `:454`, `:532`). Live: the launched record's `launcherArgv` contains the `sha256:` ID and not the name. README should still call the label a wrong-image guard only (spec4 does).
- **m2: fixed.** Components must match `^[A-Za-z0-9._@+-]+$` and not be `.`/`..` (`:260`); my earlier newline and `*` cases are now refused by the grammar.
- **m3: fixed in spec** (§7.8 "failure means no artifact"); the obligation is P1-15's, not testable here.
- **n1: unchanged by design** (redundant entries, sweep alarms); no action.
- **`--project-version`: as ruled.** Mandatory for `scip-python` only, once, same pattern as `--project-name`, as two elements. Live legit run with it: rc 0, non-empty `.scip`, no `.git` staged. Mandatory rather than optional is fine, because no `.git` is ever staged.
- **Precedence (spec4):** forbidden (incl. containment, and empty argv as `(missing index)`) outranks a missing mandatory element; confirmed live for both.

**R-E10, still reachable? Yes, by content and by stager gaps, not by `--cwd`.** The `--cwd /` route is closed. Remaining: the FIFO above; a pathological or very large staged tree (deep or huge files, regex-heavy sources); in-tree relative link cycles the stager admits (pyright/tsc loop handling is the tool's, unmeasured); plus the TOCTOU swap. **A runner-side wall-clock cap is now worth taking** (answers spec4 §6 Q2): the closed grammar bounds argv, not work, and P1-15's timeout is the only bound, enforced outside the runner. Cheap shape: a runner constant (not an environment variable; the runner reads none beyond two), default about 30 min for indexer entries, enforced by a watchdog that on expiry runs the existing `sbx_on_signal` path (KILL the client, `sbx_remove_own`), then writes the record with exit 124 and `refused:false`. Note `timeout` alone is not enough: a TERM to `docker run` is proxied and a hung node may ignore it, so removal by label must follow.

Open: confirm the staged tree is private to the runner's uid (TOCTOU), and the owner's call on the cap.
