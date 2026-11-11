# Connected code graph git hardening — security review

> **Status:** active · **Owner:** `security-expert` · **Tracks:** —

Lens: code/app security (advisory). Subject: the uncommitted git-hardening change in `cpg/` (`ccg/gitsafe.py`, `tests/test_gitsafe.py`, the git lines of `ccg/units.py`, `extractors/joern/normalize/blame.py`, `extractors/joern/run.py`, one line in `ccg/assemble/snapshot.py`). Baseline: threat "hostile repository under analysis must not make any git call execute code or touch anything outside intended places". git 2.43.0.

**Verdict: needs changes** — two confirmed blockers (code execution from repository config that the helper does not neutralise).

CPG: considered, not relevant — the subject is a subprocess/argv/environment wrapper around git, not an application data-flow path; every claim was proved by running the helper entry points against hostile tmp repos, which the graph cannot show.

Method: hostile repos under `/tmp/claude-1000/.../scratchpad/sec-gitsafe/` (marker-creating scripts for `core.fsmonitor`, `diff.<drv>.textconv`, `diff.external`, `filter.*`, `core.pager`, `core.hooksPath`, `gpg.program`, `include.path`, `includeIf`, a promisor remote). Entry points run with `cpg/.venv/bin/python -I`: `_git_files`, `blame.blame`/`blame.git`, `run._resolve_commit`/`_tree_entries`, `gitsafe.command()`+`environment()` for the `cat-file --batch` Popen. Each result is a bare-git control vs the helper. Nothing in the real repo was touched. All findings below are **confirmed** by running unless marked theoretical.

## Findings

### B1 (blocker, confirmed) — `log.showSignature` + `gpg.program` executes repo-chosen code through `blame.git("log", ...)`

`extractors/joern/normalize/blame.py:111` runs `git log -1 --format=%ct %H <commit> -- <path>`. A hostile `.git/config` with `log.showSignature=true` and `gpg.program=<script>`, plus any commit that touches the path and carries a `gpgsig` header (forgeable: the object is never verified before git tries to), makes git exec the program. `HARDENING_CONFIG` (`gitsafe.py:34`) names neither key. Probe: bare git fired the marker; the helper (`blame.git(...)`) fired it too; `-c log.showSignature=false`, `--no-show-signature` and `-c gpg.program=false` each left it clean.

Fix: add `log.showSignature=false`, `gpg.program=false`, `gpg.openpgp.program=false`, `gpg.x509.program=false`, `gpg.ssh.program=false` to `HARDENING_CONFIG`, and `--no-show-signature` at `blame.py:111` (belt and braces, as `--no-textconv` is for blame). Add a behavioural test with a forged `gpgsig` commit (recipe: `git hash-object -t commit -w --literal` of a commit object carrying a `gpgsig` header; executable `gpg.program` script that touches a marker; mutant = remove the `-c`).

### B2 (blocker, confirmed) — repo-set `protocol.<name>.allow` defeats `-c protocol.allow=never`; a promisor remote runs `ext::` code on any missing-object read

`protocol.allow=never` is only the default for protocols with no `protocol.<name>.allow`; the repository's own `protocol.ext.allow=always` wins. With `extensions.partialClone=origin`, `remote.origin.promisor=true`, `remote.origin.url=ext::<script>` and a missing blob, any object read triggers a lazy fetch: the `cat-file --batch` Popen in `run.py:180` (the blob oids come from the tree, so a hostile repo simply omits them) ran the script — marker created via `gitsafe.command(...)` + `environment()`; bare git did the same. `ls-tree`, `log`, `blame` reaching a missing tree/commit/blob are the same route. The same route also reaches `ssh` (repo `core.sshCommand` is overridden by `-c`, but `core.gitProxy`-style and URL-scheme choice remain repo-picked) and `file://`/`git://` network reads. (`uploadpack.packObjectsHook` and `core.alternateRefsCommand` are server-side and ignored from repo config; not separately probed — theoretical.)

Fix (both verified clean on 2.43.0): set `GIT_ALLOW_PROTOCOL=none` in `environment()` (git documents it as overriding every repo `protocol.*` setting; probe: `fatal: transport 'ext' not allowed`) **and** `GIT_NO_LAZY_FETCH=1` (probe: `lazy fetching disabled`, no fetch at all; honoured on 2.43.0 here, so a later version is not a precondition). The no-network guarantee should be env-level, not a `-c`. Test: promisor repo with one missing blob, `protocol.ext.allow=always`, marker script; run `cat-file --batch` for the missing oid; mutant = drop both env vars. This is also the behavioural proof for `core.sshCommand`/`protocol` (an `ssh://` promisor URL with a marker `core.sshCommand` and `protocol.ssh.allow=always`).

### M1 (major, confirmed) — the helper does not protect diff readers; `--no-textconv`/`--no-ext-diff` are load-bearing and caller-only

Mutation probes: `blame` without `--no-textconv` fires the repo's textconv; `diff` fires `diff.external`; `diff --no-ext-diff` fires textconv; `show` and `log -p` fire textconv; `--no-ext-diff --no-textconv` is clean. The in-tree `.gitattributes` (`diff=evil`) is not neutralised by `core.attributesFile=/dev/null`, which only drops the user-level file. Current callers are fine (`blame` has `--no-textconv`; `ls-files`, `ls-tree`, `rev-parse`, `cat-file --batch`, `log -1 --format` do not read diffs), and the docstring says so. But `gitsafe.git` is "the one way", and a future `git diff`/`show`/`log -p` through it is silently code execution. Fix: have `git()` refuse (or prepend the two flags to) `diff|show|log -p|blame|cat-file --textconv|--filters` unless the caller passes both flags, or expose `git_diff_reader()`. Pin: a test that calls each of those subcommands through `gitsafe.git` with the hostile repo and asserts a clean marker.

### M2 (major, confirmed) — `.git` file / `gitdir:` redirect and alternates read from anywhere

A checkout whose `.git` is a file `gitdir: <abs path>` is accepted: `_git_files(g)` listed the files of a different repository (`$S/r`) that the hostile tree named. Likewise `objects/info/alternates`. Read-only, no code execution (config in the redirected dir is neutralised the same as local config), but a hostile repo can make the build read history/objects of any git dir on the host (a private repo) into the graph/bundle: an information-disclosure and integrity issue. Fix: at the unit/stage entry, compare `git rev-parse --absolute-git-dir` to `<repo>/.git` (or `--git-dir` pinned) and reject a `.git` that is not a directory, and reject `objects/info/alternates`.

### M3 (major, confirmed) — `info/grafts` and `.git/shallow` are honoured under `--no-replace-objects` on 2.43

`--no-replace-objects` stops `refs/replace` only. A grafts file truncated a 2-commit history to 1 under both default and `--no-replace-objects` (git prints a deprecation hint, still honours it); `.git/shallow` likewise. Data-integrity, not execution: blame/last-touch results are attributable to a forged history. Shallow is already refused by `blame.py:54`; grafts are not. Fix: reject a repo with `info/grafts` or `shallow` in `--git-path`, or add `GIT_GRAFT_FILE=/dev/null` (and test the mutant).

### m1 (minor, confirmed) — `cat-file --batch` stream is uncapped and unbounded in memory/disk

`run.py:179–215` reads `int(head[2])` bytes of a blob into memory with no cap and spools every blob into a tar before extraction, and there is no timeout. A hostile multi-GB blob is a memory/disk DoS (availability only; stderr is `DEVNULL`, tar extraction uses `filter="data"`, so no path escape). Fix: refuse `head[2]` above a per-blob cap and a total cap before `read`; bound the loop with a deadline. Severity is high if units run unattended.

### m2 (minor, confirmed) — NUL in an arg or path raises `ValueError`, not a `SubprocessError`

`gitsafe.git(..., paths=["a\0b"])` raised `ValueError`; `_git_files`/`_resolve_commit` catch `(OSError, SubprocessError)` only. Inputs are manifest-supplied, so low; but fail-closed value-free handling is lost. Fix: validate (`"\0" in arg`) and raise `GitError`.

### m3 (minor, by inspection) — `preexec_fn` and uncapped stderr

`preexec_fn=limit` (a single `setrlimit` call) is documented unsafe with threads; no threads in these modules today (grep: none). Prefer a `prlimit`-style wrapper (`sh -c 'ulimit -f ...'` is worse) only if threading is ever added. `stderr=PIPE` is unbounded, but git's stderr volume is small in practice. The rlimit is inherited by every git child (no impact observed). Do not rely on SIGXFSZ: CPython restores it in the child (`restore_signals`), and either the signal or `EFBIG` kills git; the post-hoc size check (`gitsafe.py:95`) is what raises, and the probe confirmed `GitOutputTooLarge`.

### n1 (nit) — `blame.git` error text and `StageError` embed argv

`HistoryError`/`StageError` messages include `repo`, `commit`, root and (via `CalledProcessError` `str`) the full argv; they come from the manifest, not from repo-controlled branch/path names, and stderr is not echoed. Acceptable; the ownership case below reports no repo string beyond the argv.

## Answers to the brief

- **Config neutralisation.** Verified clean through the helper for: `core.fsmonitor` (also via `include.path` and `includeIf`: `-c` beats the include; `git status` clean too), `core.hooksPath` (no hook is reachable by the used commands), `core.pager` (stdout is a file/pipe, never a tty; not behaviourally provable except with a pty), `core.attributesFile`, `filter.*` (not reached by index-only/tree-only commands; `cat-file --filters` is a diff-reader class call, see M1), `diff.external` and textconv for the real callers (blame with `--no-textconv`; mutation proof above), `core.worktree` (ignored by `ls-files` index read; no effect), `core.bare`, `extensions.*` (only `partialClone` mattered: B2). **Not** covered: gpg (B1), repo-set `protocol.<name>.allow`/promisor (B2), grafts/shallow (M3), `.git` redirect/alternates (M2).
- **`GIT_CONFIG_GLOBAL=/dev/null` and includes.** The helper does not stop repo-local includes (they are processed); the `-c` overrides do for the keys they name. So the protection is per-key, and B1/B2 are exactly keys that are not named. That per-key shape is the structural weakness: prefer env-level kill switches (`GIT_ALLOW_PROTOCOL`, `GIT_NO_LAZY_FETCH`) and per-command flags over enumerating config.
- **`safe.directory`.** Left unset: a repo owned by another uid fails with git's "dubious ownership" (exit 128), i.e. `CalledProcessError` -> `_git_files` returns `None` (not listable) and `_resolve_commit` raises `StageError`: fail-closed. Not probed live (needs a second uid; theoretical for the exact message). Side effect: a user's own `safe.directory` in the real global config is also ignored (`GIT_CONFIG_GLOBAL=/dev/null`), so legitimate other-owner repos also fail: intended.
- **Env hygiene.** All `GIT_*` dropped, then `GIT_CONFIG_GLOBAL/SYSTEM/NOSYSTEM` re-added, `HOME`/`XDG_CONFIG_HOME` -> one empty `mkdtemp` (0700, `atexit` cleanup). Inherited: `PATH`, `LD_PRELOAD`, `TMPDIR`, `LANG`, `PAGER`/`EDITOR`, `SSH_*`, proxy vars. `PATH` is needed to find `git` and is caller-controlled, not repo-controlled: acceptable; `PAGER`/`EDITOR` are unreachable with a non-tty stdout; network variables become moot after B2. A cwd-relative `PATH` entry would be the caller's problem.
- **Temp file, cap, timeout.** `tempfile.TemporaryFile` is `O_TMPFILE`/unlinked, mode 0600, no name to race; cleanup automatic. Cap and timeout work (probes: `GitOutputTooLarge`, `subprocess.run` kills on timeout). `--literal-pathspecs`, `--` before paths (`paths=`) and `--end-of-options` for `rev-parse` are in place; a leading-`-` commit was rejected by git, not parsed as an option (probe: `CalledProcessError`).
- **Behavioural proof for hooks/pager/protocol/sshCommand.** Protocol and sshCommand: feasible (B2 recipe, cheap, needs only a marker script). Hooks: no read-only command used here reaches a hook, so the wiring assertion is the honest pin; a behavioural test would need a mutating command. Pager: needs a pty; low value since stdout is never a tty.

## What is solid

`environment()` strips all `GIT_*` (verified: no `GIT_CONFIG_COUNT` injection route); `core.fsmonitor=false` and the blame `--no-textconv` are mutation-proven; the cap/timeout/temp-file mechanics are sound; `--end-of-options`/`--`/`--literal-pathspecs` usage is correct; tarfile extraction uses `filter="data"`.

## Open questions

- Should the helper be a fail-closed allowlist of subcommands (`ls-files`, `ls-tree`, `rev-parse`, `cat-file --batch`, `log -1 --format`, `blame`), given M1?
- Is the build ever run unattended on untrusted third-party repos (m1 severity), or only on repos the user chose?
- Not probed: the ownership failure with a second uid (needs root), `uploadpack`/`transfer.fsckObjects` (no transport remains reachable once B2 is fixed), and `url.<base>.insteadOf` (network only).

### Proposed REVIEW.md lines

Target: `cpg/REVIEW.md` (or root `REVIEW.md`) under "Always check": *A hardened wrapper that lists the config keys it forces off is checked against the git version's executable-config surface (gpg.*, log.showSignature, protocol.<name>.allow, promisor remotes, diff/blame textconv) with one marker-creating hostile repo per key; a `-c` list is a bound, not proof.* (Source: B1, B2.)

## Pass 2

**Verdict: approve with suggestions** (no blocker remains). Re-run live against the same hostile repos, helper entry points as in pass 1 (`ccg.gitsafe.git`, `stage_unit`).

Dispositions: **B1 fixed** (forged `gpgsig` commit through `log -1 --format` and `show`: clean). **B2 fixed** (promisor `ext::` + `protocol.ext.allow=always`: `cat-file` fails, `ls-tree` ok, marker clean). **M1 fixed for the helper's own flags** (`status`, `config`, `-c` as a subcommand refused; `cat-file --textconv/--filters` refused; but see N1). **M2 fixed** (`.git` file redirect and `objects/info/alternates`: `UnsafeRepository`, value-free: only a path relative to the git dir). **M3 fixed for grafts** (refused; shallow is `blame.require_full_history`'s, accepted). **m1 fixed**: the per-blob cap (1000-byte mutant), the stage cap (1500-byte mutant) and the kill timer (shim `git` that hangs on `--batch`, `STAGE_TIMEOUT=2`: `StageError` after 2.0 s) all fired live. **m2 fixed** (NUL -> `GitError`, by reading `_checked`; not re-run).

New findings:

- **N1 (minor, confirmed) — caller-supplied args can undo the helper's flags.** `_checked` puts `--no-ext-diff --no-textconv` *before* the caller's args, so a later `--ext-diff`/`--textconv` wins: `diff --ext-diff` fired `diff.external`, `diff --textconv` and `show --ext-diff --textconv` fired the repo's drivers. `diff --output=<f>`, `-O<f>` and `format-patch -o <dir>` wrote/read a caller-named path (`outfile`, `fpout` created). `--git-dir=`/`-c` in args do not smuggle: `-c` as a subcommand is refused, `ls-files --git-dir=` is a git error. Inputs here are validated hex, helper-made shas and paths after `--`, so no repository-controlled value reaches these positions today: caller-discipline, not exploitable. Fix: append the two flags *after* the caller's args (last wins) and refuse `--ext-diff`, `--textconv`, `--output*`, `-O*`, `-o`/`--output-directory` in `_checked`, with a mutant test per refusal.
- **N2 (minor, confirmed) — `stage_unit`'s `cat-file --batch` Popen bypasses `_checked`/`check_repository`.** It uses `command()`+`environment()` directly; covered only because `_resolve_commit` and `_tree_entries` ran through `git()` just before. Fine with N3's acceptance; cleaner to call `check_repository(repo)` immediately before the Popen.
- **N3 (minor, confirmed by reading, race theoretical) — a timeout during the final blob read is not reported.** `expire()` kills `cat-file`; the `timed_out` check runs only after `readline()`. If it fires during `proc.stdout.read(size)` of the last blob, the data is short, the loop ends, and a truncated final blob is staged as success. Fix: re-check `timed_out.is_set()` after the loop (and after each `read`).

Probes with no finding: `rev-list --exclude-hidden`, `for-each-ref --shell --format`, `merge-base`, `format-patch --stdout`, `blame --contents/-S <attacker path>` and `--ignore-revs-file=<attacker path>`: no marker (no repo-config exec path; the file arguments are read-only and caller-named). Walk-up: `check_repository` on `wk/sub/deep` saw the parent's `info/grafts` and refused, i.e. it sees the repository git would use. It walks all ancestors without git's filesystem-boundary stop, so it may refuse more than git would use (fail-closed, acceptable).

TOCTOU: a repository changed between `check_repository` and the call (a graft/alternates added in the window) is accepted as residual. It needs a concurrent writer on the analysed tree, and the effect is integrity or read-only (forged history, objects read from elsewhere), not execution, since code-exec neutralisation does not depend on the check. Acceptable if the analysed repository is quiescent during a build; state that assumption.

Residual hooks/pager: agree wiring-pinned only. No allowlisted subcommand runs a hook (`format-patch` ran no hook or sendemail config in the probe) and stdout is never a tty, so a behavioural test would need a mutating command or a pty and proves nothing the allowlist does not already. Not probed: the ownership failure with a second uid (needs root), `uploadpack`/`transfer.fsckObjects` (unreachable: no transport).
