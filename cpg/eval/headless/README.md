# cpg/eval/headless — the headless-writer launcher

`run-writer.sh` launches the C-1001 P0 ground-truth writer sessions (S53, unit P0-44), following
`docs/plans/connected-code-graph-p0-2.md` §2 D20, §3 S53, §6.2–§6.3 and §13 K-P0-12. Only
**`prepare-bin`** exists so far. `prepare`, `start`, `resume` and `status` follow S52; until then
they exit 2.

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
| `PIN` | `key=value` lines: `version`, `sha256`, `binary`, `source`, `requested`, `fallback`, `prepared_at`. Mode `0444`, written last, so a `PIN` means the copy is complete. **This is the record later verbs read.** `start`/`resume` must refuse a binary whose hash differs from `PIN`'s `sha256` (§9.1) and must take the binary path from `PIN`, never from the versions directory. |

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

## Tests

```bash
cpg/eval/headless/tests/test_prepare_bin.sh   # fake versions dir under a temp HOME; never reads the real ~
cpg/eval/headless/tests/mutants.sh            # sed-mutated copies of run-writer.sh; each must turn the suite red
bash -n cpg/eval/headless/run-writer.sh       # shellcheck is not installed on this host (§7, P3-11)
```
