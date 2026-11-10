# Connected code graph egress spec 3: indexer entries, security advisory review

> **Status:** active · **Owner:** `security-expert` · **Tracks:** C-1001 (P1-48, P1-15)

## Scope & verdict

- **Lens:** code/app security review of one spec change (static; no exploitation, no network, no
  FalkorDB, no tree writes). Advisory: `architect` owns the spec, `devops` the runner and image.
- **Reviewed:** `docs/plans/connected-code-graph-egress-spec3.md` v1.3 — §7.4 indexer entries, §7.6 forbidden
  names, §7.8 `.scip` output, §10.1 tests (21)–(25), §11 R-E9 — against the built runner
  `cpg/sandbox/offline-run.sh` (855 lines; still the ten-name set, line 51), `cpg/infra/toolchain/Dockerfile`
  (`indexer` stage), `cpg/infra/toolchain/README.md`, `cpg/infra/README.md` "Indexer image", and the **bundled
  code of the two indexers inside the local image** `ccg-toolchain-indexer:73b9fb76bd2a` (read with
  `docker run --network none --read-only … grep`; no source mounted, nothing executed from `/src`).
  Provisioning intent is read from `docs/plans/connected-code-graph-p1-extraction.md` (P1-15) and
  `-p1-ml.md` (Provisioning). The runner has no indexer entry yet, so "the runner enforces X" is a design
  claim, not tested code.
- **CPG: not applicable — a design spec plus third-party bundled JS inside an image; no `cpg_*` graph covers it, so the data flow was traced by hand in the bundles.**
- **Growth-stops:** the v1.3 additions are new scope (two entries), not rework. The R-E9 paragraph restates
  README/§7.4 content once; no finding.
- **Verdict: needs changes** (spec text only; no blockers). The argv grammar does close the *argv-level*
  escape the analyst found (no `npm`/`pip`/`uv` is reachable by argv). But the closed set is not equivalent to
  "no caller-controlled code runs": scip-python executes `python3` with the caller's staged tree as its working
  directory (M1), and the host-side staging is outside every control (M2). Both are contained by the A1–A7
  boundary, not by the grammar, and the spec currently attributes the safety to the grammar.

Confidence: **traced** = followed through the bundled code or spec rules; **theoretical** = reasoned, not
run; nothing here was reproduced.

## Per-axis result

| Axis | Spec's stated control | Verdict | Residual |
|---|---|---|---|
| npm/pip/uv by argv | first element must be `index`; closed grammar | **holds** (design; untested until P1-48 lands) | none by argv |
| npm/pip spawned by the indexer | none (R-E9: "bounded by no-route network") | **holds, and R-E9 is accurate for these versions** — traced: scip-typescript 0.6.0 `child_process` is used only by `pnpm ls` / `yarn` workspace listing (`--pnpm-workspaces`, `--yarn-workspaces`, `--yarn-berry-workspaces`: all outside the grammar); scip-python 0.6.8 runs `pip list` only when `--environment` is absent (`default(e,t,i){if(i){…readFileSync(i)…}`) | `--environment` mandatory is what keeps pip away; a bundle bump needs a re-trace (see m2) |
| npm/pip lifecycle scripts | none in the sandbox (no installer runs) | **holds in the sandbox; unsound at staging** | host-side provisioning runs code with network (M2) |
| node startup files | pass-list closed; `NODE_OPTIONS` etc. absent | **holds** — Node 24 reads no cwd config without an opt-in flag; scip-typescript has no `require`/`import()` of project paths (traced: no dynamic require in `dist/src`) | none found |
| python startup files / cwd | none | **unsound** — see M1 | staged `/src/json.py` (any shadowable stdlib module the probe imports) runs |
| `--environment` outside `/src` | `<p>` is `/src` or below, no `..` | **holds for host paths** (host paths are not mounted, A5); **lexical only** inside the container | symlink below `/src` to `/`, `/out`, `/tmp`, `/opt` is readable by the indexer; not host data. `/srcfoo` prefix and `/src/` forms untested (m1) |
| `--environment` resolution under staged `/src` (R-E9's open item) | unverified | **answerable from code now**: the value is passed to `readFileSync` as given, and the spec's grammar makes it absolute, so `process.chdir(--cwd)` does not affect it. `env.json` `files` are used only to map files to packages, never opened (traced from `default(e,t,i)`) | only the file's content format on a venv built at a different host path is unmeasured |
| dependency staging writing outside the source dir | "staged by the caller" | **untested, no control in the runner** (M2) | host-side |
| output path tricks | runner appends `--output /out/<auditId>.scip`; caller `--output` refused | **holds** — scip-python and scip-typescript each have one `--output`; grammar full-match refuses `-o`, `--output=`, abbreviations (commander 9.5 / 15 do no prefix matching, and any non-grammar element is refused anyway). §7.5 (4) text still says "for a parse frontend" (m2) | none |
| argv grammar bypass | full-match, each flag at most once | **mostly holds**; `--project-name` value may begin with `-` (m1), order and `--cwd` optionality unspecified | commander 9.5 takes a required option's next element as its value even when it starts with `-`; a name `--output` is consumed, not interpreted — low |
| symlinks in staged deps | none stated | **host-safe, indexer-visible** (see `--environment` row) | in-sandbox reads of the image and `/out` only |
| forbidden env names (§7.6) | documentary; pass-list is closed | **holds** — test 7 plus A6 are the real control; the list adds nothing for safety | list is incomplete as a reader's aid (nit) |
| §7.8 `.scip` output | auditId-stemmed file in `/out` | **holds** | the file is untrusted input to the P1-15 parser (m5) |

## Findings

### Major

**M1. scip-python runs `python3 -c` with the caller-staged tree as its working directory, so staged
files execute in the sandbox; the spec's R-E9 and the closed-argv framing omit this. (traced; theoretical
until run.)**
- Evidence: `scip-python.js` (bundle, image path `/opt/indexer/npm/scip-python/node_modules/@sourcegraph/scip-python/dist/`):
  `const i=e.cwd … process.chdir(i)` before indexing; `pyright-internal.js`:
  `_executePythonInterpreter` → `execFileSync("python3", ["-c", "<…>; import sys, json; json.dump(dict(path=orig_sys_path, prefix=sys.prefix), sys.stdout)"])`
  with the child inheriting the process cwd. `python3 -c` prepends the cwd (`''`) to `sys.path`, so a staged
  `/src/json.py` is imported instead of the stdlib module and runs. The probe runs on every scip-python
  invocation, `--environment` or not. The spec mandates `--cwd /src` (P1-15's argv), so the cwd is the staged
  tree. ScipPyrightConfig also reads `scip-pyrightconfig.json`, `pyrightconfig.json` and `pyproject.toml`
  from the staged tree, so the tree also steers the interpreter probe's settings.
- Why it matters: §7.4 rests on "script mode bounds a script's arguments, not what the script executes, so
  an indexer is never launched through it", and §7.3 says a script never comes from `/src`. For
  scip-python the staged source *is* a script source. The capability equals the joern-script one the analyst
  raised: arbitrary exec of any image binary (`/tmp` is exec-capable, `npm`/`pip` present), write to `/out`.
  Containment is only A1–A7. A benign repository with a top-level `json.py` triggers it unintentionally.
  Today's corpus is the lab's own pinned repos, so the exposure is accidental, not adversarial.
- Suggested improvement: (a) add `PYTHONSAFEPATH=1` to the pass-list (Python 3.12.3 in the image; the setting
  is honoured for `-c` and inherited by the probe child; it is safety-positive, like `GOTOOLCHAIN=local`),
  and remove it from nothing in the forbidden list; (b) add a test: a staged `/src/json.py` that writes a
  marker into `/out` — **with** `PYTHONSAFEPATH=1` the marker is absent, and with the variable removed it is
  present (that is the mutant that makes the check fail); (c) state in R-E9 that staged content can execute
  inside an indexer run, that the boundary (A1–A7, the launcher clause) is what limits it, and that indexer
  runs are for trusted/pinned sources only. Do not claim the grammar closes code execution.

**M2. Dependency staging is outside every control and runs with network and the target's own code. (traced
from the plan text; theoretical as to exploitation.)**
- Evidence: §7.4 "staged by the caller into the staged source directory before the call". P1-15 / `-p1-ml.md`
  Provisioning: `npm ci --ignore-scripts` (good) and, for Python, `uv pip install -e ".[dev]"` then
  generating `env.json` from the venv.
- Why it matters: an editable install of the target runs its PEP 517 build backend / `setup.py` on the host,
  and dependency sdists may build code, with a network, as the user. That is where npm/pip lifecycle hooks
  actually execute, and it is a larger escape than anything inside the sandbox. The venv's `bin/python` is a
  host symlink, dangling in the sandbox (harmless in itself, but a symptom that the staged tree is a host
  artifact copied across a trust line).
- Suggested improvement: record in §7.4 (or the P1-15 step) that provisioning is a host step with network
  and is trusted-repo-only; for Python, install the dependencies only (`--no-deps` for the target itself,
  `--only-binary :all:` or `--no-build` where the corpus allows), never the target's build backend; run `npm
  ci --ignore-scripts` as planned; and before the call have the stager reject, in the staged tree, symlinks
  that point outside it, hardlinks and device/FIFO files (a FIFO or `/dev/zero` named as `env.json` would hang
  the run; see m3). That check belongs to P1-15's stager, not the runner.

### Minor

**m1. Grammar gaps and untested cases (§7.4, test 22).** (a) `--project-name` accepts a leading `-` or
`--`; require `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`. (b) State whether the optional flags may come in any order;
test both. (c) Make `--cwd` mandatory (without it the indexer runs in the image's `/work`, producing an empty
index that looks like success) — or say what it does. (d) "`/src` or a path below it" needs a test for
`/srcfoo`, `/src/`, `/src//a`, `/src/./a` and `--cwd=/src` / `--project-name=x` forms (all expected
`forbidden-flag`); test (22) covers only `--cwd /src/../x` and one repeat. Add the repeat of each flag and
abbreviation `--proj`, `--env` as cases.

**m2. Text and drift-guard inconsistencies.** (a) §7.5 (4) still reads "for a parse frontend, a caller-supplied
`-o`/`--output` is refused"; §7.4 cites it for the indexers. Extend the sentence. (b) The classification regex
(`fetch|download|url|…|jar4`) matches none of the exec-capable flags actually found (`--pnpm-workspaces`,
`--yarn-workspaces`, `--infer-tsconfig`). So test (24) cannot detect a new spawn-capable flag; it is a
network-flag drift guard only. Say so, and note that the grammar whitelist is the control; consider adding
`workspaces|install|infer` to the indexer sweep. (c) R-E9's "scip-typescript runs `tsc` in process" and "scip-python has
no reason to call pip" are true of 0.6.0/0.6.8 by trace above; the pins are lockfile-fixed, so say "at these
versions" and require a re-trace when `truth-npm/` changes.

**m3. No CPU/memory/wall-clock limit on the container** (`--pids-limit 4096` only; `grep -- '--memory\|--cpus\|timeout'`
finds none in the launch). Node/TypeScript indexing of a large tree, or a staged FIFO, can exhaust the 6 GB
host. Pre-existing for Joern; heavier here. Add `--memory`/`--cpus` and a runner timeout, or open an item.

**m4. The image is not validated.** The runner takes any `CCG_TOOLCHAIN_IMAGE`. An indexer entry run on the
wrong image fails with ENOENT (safe); the reverse — the indexer image under `--frontend joern` — works and
gives a script access to npm/pip (no network, so no new capability). Suggest the runner check the image label
`ccg-toolchain-indexer=1` for indexer entries and its absence otherwise, and record the image digest in the
`sandbox` record.

**m5. `.scip` output is untrusted input.** The file comes from a process that may have run staged code (M1).
P1-15's `scip.py` parser should treat it as hostile protobuf (size cap, no path from it used for a write).

### Nit

- §7.6's new names are documentary (the pass-list is closed). If kept as a reader's aid, also name
  `PYTHONSTARTUP`, `PYTHONUSERBASE`, `PYTHONSAFEPATH` (as *required*), `UV_*`, `NODE_PRESERVE_SYMLINKS`,
  `LD_PRELOAD`, `BASH_ENV`; or say the list is illustrative.
- `cpg/infra/toolchain/README.md` and `cpg/infra/README.md` still say the runner has no indexer entry; update
  at P1-48 close.

## What's solid

- Absolute fixed executable paths, no `PATH` lookup; offline only; container only; `controlled` refused.
- `--environment` mandatory for scip-python is the right control and matches the code (pip only without it).
- The runner owning `--output` and the refusal of caller `-o`/`--output`; full-match grammar over a
  substring audit; at-most-once flags.
- Node/Python do not read config from the cwd without opt-in, and scip-typescript has no dynamic require of
  project code, so the only code-exec path found is the Python probe.
- Dependencies arrive through the existing read-only, `noexec` `/src` mount; no new mount or runner option.
- Host paths are not mounted, so in-tree symlinks cannot reach host data.

## Open questions

1. (`architect`) Accept `PYTHONSAFEPATH=1` on the shared pass-list, or scope it to the indexer entries only?
   It changes test 7/A6's expected set.
2. (`architect`/`devops`) Is provisioning for P1-15 restricted to the lab's pinned corpora? If yes, say so
   in §7.4; if arbitrary repositories are ever indexed, M1 and M2 become blockers.
3. (`devops`) Container resource limits (m3): in scope for P1-48 or a new item?

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, "Always check": "An interpreter-based tool run on a caller-staged tree is checked for
what the interpreter itself imports or loads from its working directory (cwd on `sys.path`, config files);
a closed argv does not bound that."
