# P1-15 indexer adoption: security review (hostile `.scip`, stager, runner)

> **Status:** active · **Owner:** `security-expert` · **Tracks:** C-1001 P1-15

Lens 1 (code/app security), advisory. Baseline: uncommitted `cpg/extractors/indexers/{scip,run}.py`,
`cpg/extractors/joern/normalize/merge_scip.py`, read against `docs/plans/connected-code-graph-p1-extraction.md` P1-15,
`-egress-spec3.md` §7.4, `-egress-spec4.md` §1, and the runner `cpg/sandbox/offline-run.sh`. Threat: the `.scip` is
hostile (written by a process that may have run staged repo code); the staged tree is untrusted repo plus operator
dependencies. All probes ran with `cpg/.venv/bin/python -I`, cwd `cpg/`, `ulimit -v 1500000`, `timeout -s KILL 45`, on crafted
files and stub runners under the scratchpad (`.../scratchpad/sec-p1-15/`). No container, service, or tree file was touched.

**Verdict: needs changes** (two blockers, both on the hostile-index parser: the size cap does not bound its CPU or memory).

CPG: considered, not relevant — the files are new and uncommitted, so no loaded `cpg_*` graph contains them; traced by reading and live probes.

## Blockers

### B1. Quadratic CPU in reference de-duplication: a ~0.5 MB index hangs the build (confirmed)
`scip.py:320-322`: `slot = refs.setdefault(...); if t not in slot: slot.append(t)`. A hostile index puts N occurrences with N
distinct symbols at one `(path, line, col)`; each `in` scans the whole list of dataclass comparisons. Measured:
N=10 000 (269 KB file) 10.7 s; N=20 000 (549 KB) 40.7 s; N=40 000 (1.1 MB) not finished in 45 s (KILLed). Extrapolating
to the occurrence cap (2 000 000, about 55 MB, under the 128 MB size cap) is on the order of 10^2 hours. The two caps the
docstring relies on do not bound this; there is no wall-clock bound on the parse.
**Fix:** de-duplicate with a per-position `dict`/`set` keyed on the symbol id `k` (already interned), O(1) per occurrence;
additionally cap targets per position (e.g. 64) and count the excess in `rejected_occurrences`. Add a test with 50 000 distinct
symbols at one position that must finish in a bounded time (this fails on the current code, passes on the fix).

### B2. Per-document span lists are built before the occurrence cap is checked: memory is ~100x file size (confirmed)
`scip.py:265-271`: every `occurrence` (field 2) and `SymbolInformation` (field 3) span is appended to `occs`/`doc_syms`
for the whole document first; `max_occurrences` is only tested while iterating them (`:284-286`), and `doc_syms` has no cap at
all. A document of N empty submessages (2 bytes each) costs about 36 bytes of file per ~130 bytes of tuple/int overhead. Measured:
a 40 MB file of 20 M empty occurrences, and likewise 20 M empty symbol-infos, both ended in `MemoryError` at the 1.5 GB limit
(7 s); at the 128 MB cap this is about 4.5 GB or more. Also `parse_scip` holds the chunk list and the joined bytes at once (2x file).
**Fix:** count and cap while collecting (raise `ScipError` when `len(occs) + occ_count > max_occurrences`, and add
`max_symbol_infos`), or stream: iterate `_fields` lazily instead of materialising span lists; read into one `bytearray` with
`readinto` instead of `b"".join`. Test: a 40 MB all-empty-occurrence file must raise `ScipError` with bounded RSS.

## Majors

### M1. Runner output is buffered unbounded in the parent; a hostile indexer can exhaust memory (partly confirmed)
`run.py:237-242` uses `PIPE` plus `communicate()`. The runner does not redirect the container's stdout/stderr
(`offline-run.sh:753`, `"${SBX_LAUNCHER[@]}" < /dev/null &`), so a staged-code indexer that prints for up to 1800 s is
accumulated in the Python process. Confirmed on the Python side with a stub runner that `exec yes`: `MemoryError` at 1457 MB in
under 20 s. That the real container's output reaches that pipe is by code reading (docker run is attached), not run live.
The same pattern exists in `extractors/joern/run.py:_run`, which this module copied.
**Fix:** redirect runner stdout/stderr to files in `<work>/indexer/` opened by the parent (`stdout=fh`), with a size cap
enforced by `ulimit -f` or a bounded reader thread; log only a tail. Test: stub runner emitting 2 GB must not grow RSS.

### M2. After a successful TERM the process group is never swept (confirmed with a stub)
`run.py:243-252`: on outer-bound expiry only the runner gets TERM; `_killpg` runs only if `communicate(kill_grace)` also times out. A runner
that exits on TERM while a group member survives and holds no pipe (the stub's detached `sleep`; in reality the watchdog
timer subshell, which redirects to `/dev/null`, or a docker client) outlives the call. Measured: `RunnerTimeout` raised, survivor
pid still alive afterwards. The real runner's timer self-exits within 30 s and its trap removes the container, so the
practical residue is small, but nothing guarantees it. **Fix:** always `_killpg(p.pid)` after the TERM/communicate sequence (ESRCH
is already swallowed; the pgid cannot be recycled while a member lives). Test: the stub above.

### M3. Parent death or Ctrl-C orphans the runner (theoretical, from code)
`start_new_session=True` detaches the runner from the terminal's foreground group, so a SIGINT to the calling process
(or an exception such as B1/M1's `MemoryError`, or `KeyboardInterrupt`) leaves the runner and its container alive until the
runner's own 1800 s watchdog. **Fix:** wrap `communicate` in `try/except BaseException`: send TERM, wait the grace, `_killpg`, re-raise.

## Minors

- **m1. Order of validation in `run_indexer`.** `env_json` gets only its last component checked in `_missing` (`:210`);
  `_sha256(stage / env_json)` (`:330`) and `stage / lockfile` (`:331`) run before `tool_argv` applies the full `/src/` grammar (`:334`).
  A value like `../../x` is hashed (read, no disclosure) before `ValueError`. Operator-supplied, so low; validate with
  `_path_ok(f"/src/{env_json}")` in `_missing` and validate `lockfile` the same way.
- **m2. Forged resolutions inside the corpus (design, theoretical).** A hostile index decides `confirmed`/`resolved-call`
  (`merge_scip.py:104-106`) and which `defs` file a symbol belongs to (`scip.py:282,308`, first wins). It cannot reach outside
  `known`, so the damage is mislabelled in-corpus edges, but "confirmed" then rests on an unauthenticated writer. Worth
  one sentence in the plan's adoption criteria (the M1/M2 measures on truth data bound the benign case only).
- **m3. Control characters in document paths and `audit_id`.** `_path_ok` accepts newlines/control chars; `audit_id` is
  `scip.name[:-5]` of a file name the hostile container chose (`_new_scip`, `run.py:358`). Used as keys only today, but any
  later log line or path join that trusts them inherits it; reject `[\x00-\x1f\x7f]` and match `audit_id` against
  `^[0-9a-f-]+$`-style runner format.
- **m4. No count or byte bound on the provisioned tree copy** (`_overlay`); an oversized `node_modules` fills `area`.
  Operator-trusted, so only a note; a file-count/byte ceiling in `validate_tree` is cheap.
- **m5. Validate-then-copy window on the provisioned root** (`:192-201`): a FIFO planted after `validate_tree(root)` would hang
  `copy2` before the final `validate_tree(stage)`. Requires a concurrent local writer; theoretical.

## What is solid (probed)

- **`.scip` open path** (`scip.py:332-352`): `O_NOFOLLOW|O_NONBLOCK`, `fstat` on the same fd, read bounded to `max_bytes+1`, so no
  TOCTOU between cap and read. Live: FIFO, socket, directory, symlink-to-file, `/dev/zero`, `/dev/null`, missing file all raise
  `ScipError` instantly. A symlink in a *parent* directory is followed, acceptable because `out_dir` is host-created.
- **Wire decoder**: varint capped at 10 bytes, length fields checked against the remaining bytes (a claimed gigabyte costs
  nothing), only fixed nesting depth (no recursion), wire types 3/4 refused, ranges `>= 2^31` rejected. `parse_symbol` and
  `_SIMPLE` are linear (no backtracking). Document paths are dictionary keys only; no write-capable call in the module.
- **Stager** (`validate_tree`): 17 adversarial trees probed live. Rejected: absolute links, `..` escapes, link-to-parent, chains
  through an inside link, dir-link plus `..` lexical/real divergence, sibling-prefix (`in-evil`), hardlinks (including to an
  outside file), FIFO, socket. Accepted correctly: link to `.`, dangling inside, inside loops. Validation runs on the provisioned
  root before the copy, and on the whole stage after the overlay. Top-level collisions with the repo's own `node_modules`/`venv`
  are refused. `stage_unit` uses tarfile's `data` filter on top, and the runner has a second host-side symlink walk (`sbx_idx_walk`).
- **argv**: list-form `Popen` (no shell); `--project-name`/`--project-version` match `NAME_RE` (leading alnum, so no option
  smuggling), `env_json` goes through the §7.4 component grammar, the version is a resolved commit sha, and the runner consumes
  `--codebase`/`--source` values positionally (`offline-run.sh:935-941`) so a value starting with `-` is not parsed as a flag.
  `--frontend joern` is refused by `runner_argv`.
- **Environment** to the runner is only `PATH`, `CCG_TOOLCHAIN_IMAGE` (regex-checked) and, if set, `CCG_AUDIT_DIR`. No `HOME`,
  tokens, or proxy variables are forwarded.
- A failed or timed-out run never reaches `parse_scip`, and a symlinked `*.scip` planted by the container is refused by `O_NOFOLLOW`.

## Open questions

1. Should the parse have a wall-clock bound (SIGALRM or a subprocess) in addition to the structural caps, as defence in depth
   beyond B1/B2?
2. Is the Joern `_run` (same unbounded-pipe and TERM-only pattern as M1/M2) in scope for the same fix by this step's coder, or a
   separate item for the P1-48 owner?
3. For m2, does `data-scientist` want the adoption criterion to state that SCIP is an unauthenticated writer?

Exploitation (FR-10): none attempted against any service. Findings above come from crafted local files and stub runner scripts only.

## Pass 2

Re-probed against the coder's fixes with the same crafted files (`ulimit -v 1500000`, `timeout -s KILL 45`). **Verdict: approve with suggestions** (no blocker remains; one major and two minors are new).

### Dispositions

- **B1 fixed.** `sp20000`/`sp40000`/`sp200000` (the files that took 40 s and hung) now parse in 0.1/0.19/0.96 s at 15-46 MB, one position holding 64 targets, the rest counted in `rejected_occurrences`.
- **B2 fixed.** The 40 MB empty-occurrence and empty-symbol-info files, and a 3 M-entry variant, raise `ScipError` at the cap in about 0.6 s at 18-51 MB (was `MemoryError`). A 3 000-input mutate/truncate fuzz raised no non-`ScipError` exception. `readv` growth check reads sound.
- **Cap-respecting worst case confirmed.** 1 M occurrences with distinct symbols at distinct positions (30.9 MB file): 10.2 s, 658 MB peak, completes under the 1.5 GB limit (coder: 10.8 s, 723 MB). Bounded, acceptable for P1 as a batch step.
- **M1 fixed.** `yes` stub, 6 s outer bound: 27 MB peak, `RunnerTimeout`. A stub that closes stdout and writes 3 MB of random bytes to stderr: rc 3, stdout 0 bytes, stderr capped at about 2 KB for `keep=1024`, no deadlock, binary safe.
- **M2 fixed.** The TERM-exiting stub's detached survivor is now dead after `RunnerTimeout` (was alive).
- **M3 fixed on reading** (`_run` `except BaseException: _end(...)`, `run.py:373-375`); not exercised with a live SIGINT. The memory-error variant of the trigger is covered by the M1 probe.
- **m1 fixed.** `check_provisioned_names` runs in `run_indexer` before `_missing`/hashing; `../x` raises `ValueError`. `fullmatch` change: `"abc\n"` is refused by `NAME_RE`, `SAFE_NAME_RE` and `_path_ok`; the old `$`-newline slack is closed.
- **m3 fixed.** Control characters, DEL in document paths are rejected (`\n`, `\x1f`, `\x7f` probed); the `.scip` name must fullmatch the runner uuid.
- **m4 fixed.** `validate_tree(max_entries=10)` and `(max_bytes=100)` on a 30-file/300-byte tree reject; uncapped passes.
- **m2, m5, wall-clock parse bound, Joern `_run` pipe: acceptable for P1** (reasons below).

### New findings

**N1 (major, confirmed). `_run` can hang indefinitely when the runner exits normally but a descendant keeps a pipe open.**
`run.py:376-380`: the `finally` does `r.join(timeout=5)` then `p.stdout.close()`. The reader threads are still blocked in `read1`
(`run.py:322`), and `BufferedReader.close()` waits on the buffer lock the reader holds. Stub: runner exits 0 after starting a `sleep 120`
that inherits stdout/stderr; the call did not return in 45 s (KILLed); `faulthandler` shows the main thread at `run.py:379`, both readers at `:322`.
Normal path only: `_end` (with the group sweep) runs on exceptions and timeouts, not on a clean exit. The real runner's timer subshell
redirects to `/dev/null` and the Docker client exits first, so this needs a stray descendant, but the call has no bound when it happens.
**Fix:** on the normal path also `_killpg(p.pid)` before joining, and do not `close()` while a reader is alive: join with a timeout, and
if a reader is still alive leave the file objects to the daemon threads (or close the underlying fd with `os.close(p.stdout.fileno())`).
Test: the leaky stub must return within the join timeout.

**N2 (minor, confirmed). The `~/.config/ccg` refusal in `load_provisioned` is lexical only and bypassable.** With `$HOME` pointed at a scratch
home (the real one untouched): `CCG_PROVISION_DIR=<x>/../home/.config/ccg` returns a `Provisioned` rooted in the keys directory (`..`
is not normalised, so `keys in root.parents` is false); and a symlinked system component (`prov/ccglink -> ~/.config/ccg`, unit `unitk`) also
returns it, because only the last component is checked with `is_symlink`. Direct base == keys dir and base under it are refused, relative base refused,
`system`/`unit_id` traversal (`..`, `.`, `a/b`, trailing newline) refused, a symlinked unit dir returns `None`, a symlinked `provision.json` is refused.
Impact: the settings come from the operator's environment, not from a repo, so this is a misconfiguration guard failing, but if the chosen
directory held `node_modules` its content (keys) would be copied into the staged tree the indexer container reads. **Fix:** also compare
`os.path.realpath(root)` against `os.path.realpath(keys)` (and reject any `..` component in `CCG_PROVISION_DIR`), without creating anything;
keep the lexical check too. Test: the two inputs above.

**N3 (nit).** `provision.json` is itself copied into `/src` by `_overlay` (harmless, and reserved by name against a repo collision); mention it in the docstring or exclude it.

### The coder's undone items

- **Joern `_run` unbounded pipe (`extractors/joern/run.py`, P1-04's file): acceptable for P1 with a backlog item.** It is the same memory-only hazard as M1, but its tools (`joern-parse`, scripts) parse text and do not execute repo code, and the runner is time-bounded by the caller. File it against P1-04/P1-48 so it is not lost; copy `_Capped`/`_end` from this module.
- **m2 (hostile index can pick which in-corpus target is `confirmed`): acceptable.** It cannot leave `known`, the M1/M2 truth measures bound the benign case, and the merge's `possible` marking on disagreement means a forged claim only displaces, never drops, Joern's candidates. Record it as a known limit in the adoption note.
- **m5 (validate-then-copy window): acceptable.** The post-overlay `validate_tree(stage)` re-checks the result and the window needs a concurrent local writer on an operator-owned directory.
- **No wall-clock bound on the parse: acceptable for P1.** The caps now bound the work: worst case measured at 10.2 s, 658 MB, and the quadratic path is gone. A `SIGALRM`/subprocess guard would be cheap defence in depth, not required.

## Pass 3

- **N1 fixed.** The leaky stub (runner exits 0, a same-group `sleep 120` holds the pipes) now returns in 0.3 s and the descendant is dead (was a hang past 45 s). A `setsid` descendant (escapes the group) leaves the reader threads alive after the 5 s join, as designed: 26 concurrent runs held about 25 threads / 28 fds while the descendants lived and returned to the baseline 1 / 4 once they exited, so nothing leaks across runs (a descendant that never exits would pin 2 threads and 2 fds per run, a hazard only a host-side stray process can cause, acceptable).
- **N2 fixed.** Refused (scratch `$HOME`): `..` in the base, a base that is a symlink to the keys dir, a symlinked system component, a symlinked `$HOME` with a real or a symlinked base; the normal case still loads.
- **N3 fixed.** `_overlay` stages only `env.json` and `node_modules` from a directory that also holds `provision.json`.
- **Pid reuse (theoretical, acceptable).** `_killpg(p.pid)` runs after the leader is reaped, so it could hit an unrelated group only if the pid is recycled and made a group leader within microseconds; to close it fully, send the group KILL before `wait()` or use `waitid(WNOWAIT)`.
- **Verdict: approve.** Only the three items above were re-probed; no new finding.
