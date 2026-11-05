# cpg/eval/headless — the headless-writer launcher

`run-writer.sh` launches the C-1001 P0 ground-truth writer sessions (S53, unit P0-44), following
`docs/plans/connected-code-graph-p0-2.md` §2 D20, §3 S53, §6.2–§6.3 and §13 K-P0-12. Verbs:
`prepare-bin`, `prepare`, `start`, `resume`, `status`. All are built and tested offline against a
stub binary; no real writer session has been run (S55/S54 do that).

## `prepare-bin` — take the pinned Claude CLI build

```bash
cpg/eval/headless/run-writer.sh prepare-bin
```

This command copies `~/.local/share/claude/versions/2.1.312` to
`~/.local/share/ccg-eval/bin/claude-2.1.312`. The updater moves `~/.local/bin/claude` and prunes
old builds, so the launcher runs its own copy. After `prepare-bin`, nothing reads the versions
directory again. The command only reads `~/.local/bin/claude` and `~/.local/share/claude/`, and
writes only under `~/.local/share/ccg-eval/bin/`.

**Where the SHA-256 is recorded** (in `~/.local/share/ccg-eval/bin/`):

| File | Content |
|---|---|
| `claude-<version>` | the copy, mode `0555` |
| `claude-<version>.sha256` | `sha256sum -c` format: `cd ~/.local/share/ccg-eval/bin && sha256sum -c claude-2.1.312.sha256` |
| `PIN` | `key=value` lines: `version`, `sha256`, `binary`, `source`, `requested`, `fallback`, `prepared_at`. Mode `0444`, written last, so a `PIN` means the copy is complete. **This is the record later verbs read.** `start`/`resume` refuse a binary whose hash differs from `PIN`'s `sha256` and take the binary path from `PIN`. |

**Behaviour:**
- **Copy:** the build is copied to a temp name in `bin/`, hashed against the source, and renamed
  into place. A copy whose hash doesn't match is discarded (exit 5).
- **Idempotent:** if `PIN` exists and its binary still hashes to the recorded `sha256`, the
  command does nothing and exits 0. This holds even after the updater has pruned the source.
- **Refuse, never overwrite:** a binary whose hash doesn't match `PIN`, a `PIN` without its
  binary, or a `claude-*` file without a `PIN` all exit 3 and change nothing. To re-pin on
  purpose (K-P0-12's deliberate step), move `bin/` aside by hand, re-run, then re-run S55 and S54.
- **Fallback (K-P0-12):** if 2.1.312 is gone, the command takes the **current** build (the target
  of `~/.local/bin/claude`) and stores it under its real name, `claude-<that version>`. It sets
  `fallback=1` in `PIN`, writes `bin/FALLBACK-REREAD-6.3`, and prints a WARNING banner on that run
  and every re-run. §6.3's flags must then be re-read against that build's `--help` before S55.
  §6.3's `BIN=…/claude-2.1.312` changes with it, which is one more reason later verbs read `PIN`.

Exit codes: 0 done / already prepared · 2 usage · 3 existing state refused · 4 no source build ·
5 copy failed verification.

Overrides, for tests: `CCG_EVAL_HOME`, `CLAUDE_VERSIONS_DIR`, `CLAUDE_LAUNCHER`, `CCG_PIN_VERSION`.

### Pinned build on this host (taken 2026-10-30)

| | |
|---|---|
| version | `2.1.312 (Claude Code)`, from `env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 timeout 10 ~/.local/share/ccg-eval/bin/claude-2.1.312 --version`, exit 0 |
| sha256 | `4f4178ce41d83e23341ebfbaecdbcf8ea6bf7baea0690025abe49eeec3a5d01d` |
| fallback | no; 2.1.312 was on disk, while `~/.local/bin/claude` pointed at 2.1.314 |

## `prepare`, `start`, `resume`, `status` — run one writer role

`<E>` is the eval commit (first 12 characters are used); `<role>` is `KA KB QW AN1 AN2 rehearsal
capture`; `<stratum>` is lowercase (`flow`, `dummy`, ...). All state lives under
`~/.local/share/ccg-eval/<E>/` (§6.2); `CCG_EVAL_HOME` overrides the root.

```bash
run-writer.sh prepare <E> <role> <stratum> --brief BRIEF.md [--kit-src DIR]   # default DIR = <E>/kit-src
run-writer.sh start   [--dry-run] [--restart] [--model M] [--timeout S] [--session-id U] <E> <role> <stratum>
run-writer.sh resume  [--dry-run] [--assume-ended] [--timeout S] [--session-id U]       <E> <role> <stratum>
run-writer.sh status  [<E> [<role> <stratum>]]
```

- **`prepare`** lays out `roles/<role>-<stratum>/{kit-src,BRIEF.md,out}` plus `runs/` and `work/`.
  `kit-src` and `BRIEF.md` are read-only copies. Idempotent: identical inputs are a no-op;
  different inputs are refused (exit 3), never overwritten.
- **`start`** and **`resume`** build the command with the single `build_argv` function (§6.3's
  argv, `env -i`, `timeout --kill-after=60`, `--tools Read,Grep,Glob,Write,Edit`,
  `--disable-slash-commands`, `CCG_GUARD=<this dir>/guard-paths.sh`). `--dry-run` prints it one
  token per line and writes nothing; start's and resume's dry output differ only in
  `--session-id <u>` vs `--resume <u>`, the prompt (`BRIEF.md` vs the fixed continuation prompt) and
  the stream-log name. The binary is read from `bin/PIN`, never from the versions directory.
- **Per-role defaults** (§6.1, §6.3): KA `ccg-key-writer-provider` opus 4 h; KB
  `ccg-key-writer-consumer` sonnet 4 h; QW `ccg-question-writer` opus 2 h; AN1 `ccg-annotator`
  sonnet 1 h; AN2 `ccg-annotator` opus 1 h; `rehearsal`/`capture` `ccg-annotator` sonnet 30 min.
  The `rehearsal`/`capture` agent and the 30 min timeout are this script's choice (the plan names
  neither); `--model` and `--timeout` override.
- **Attempts:** every attempt writes a new `runs/<uuid>.<n>.stream.jsonl` (noclobber; `n` is one past
  the highest of the index and the files on disk, so an orphan log is skipped, never reused).
  `runs/index.jsonl` gets a `start` line (role, stratum, session, n, agent, model, timeout,
  `cli_version`, `bin_sha256`, `settings_sha256`, `agents_sha256`, stream name) before the run and
  an `end` line (exit code; 124 = timeout) after it. Cost is not recorded: it is in the stream's
  result event.
- **`start` refuses** a role that already has a session (`--restart` allowed only when `out/` is
  empty; the launcher never deletes anything), a non-empty `out/`, an existing stream log for the
  session, a pinned binary that no longer hashes to `PIN`, or a missing guard / `/usr/bin/python3`.
- **`resume` refuses** (exit 3): no recorded session, an unknown or non-current `--session-id`, a
  changed CLI version, binary hash, `settings.json` hash or `agents.json` hash against any earlier
  attempt, or a previous attempt with no `end` line (unless `--assume-ended`). It reuses the
  recorded agent, model and timeout. Auditing the partial session first (§6.3) is the operator's step.
- **Locking and signals:** a real `start`/`resume` takes an atomic lock (`runs/.lock-<role>-<stratum>`,
  `mkdir` plus the launcher's pid) before reading the index and holds it until it exits, so two
  parallel launches on one role give exactly one `start` line and one exit 3. A lock whose pid is dead is
  taken over; `--dry-run` takes none. The launch runs with stdin from `/dev/null`; SIGTERM/SIGINT to the
  launcher is forwarded to `timeout` (which stops claude) and the `end` line is still written. If the
  launcher is killed with SIGKILL, the writer may survive: **find and kill the orphan before using
  `--assume-ended`**. Accepted limits: takeover of a stale lock is not atomic (two launchers racing on
  one dead pid could both run; the window is tiny), and a stale lock whose pid was reused by another process
  refuses with exit 3 (remove the lock directory by hand). A torn last index line is terminated before the
  next record is appended.
- **Size and links:** a brief over 100,000 bytes is refused (exit 2) by `prepare` and by `start`
  before any session id, stream log or index line exists. `prepare` refuses (exit 3) a kit that
  contains a symlink; flatten it first.
- **`status`** is read-only: with no argument it lists eval dirs; with `<E>` (and role/stratum) it
  prints the binary check and one line per attempt, with `DRIFT:` when a config or the binary no
  longer matches the recorded hashes.

Exit codes for these verbs: 0 done / no-op · 2 usage · 3 existing state or drift refused · 4 a
required input is missing (PIN, brief, kit, role root).

## Tests

```bash
cpg/eval/headless/tests/test_prepare_bin.sh   # fake versions dir under a temp HOME; never reads the real ~
cpg/eval/headless/tests/mutants.sh            # sed-mutated copies of run-writer.sh; each must turn the suite red
cpg/eval/headless/tests/test_launcher.sh      # prepare/start/resume/status against a stub binary in a temp HOME
cpg/eval/headless/tests/mutants_launcher.sh   # exact-string mutants of run-writer.sh against test_launcher.sh
bash -n cpg/eval/headless/run-writer.sh       # shellcheck is not installed on this host (§7, P3-11)
```
