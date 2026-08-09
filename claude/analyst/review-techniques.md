# Analyst review techniques — on-demand

> Specialized verification techniques, loaded on demand from `analyst.md` (not part of the
> always-loaded prompt body) — mirrors the `graph-dba/falkordb-quirks.md` pattern of keeping a
> resident prompt lean and pushing a growing, occasionally-needed fact/technique list into its
> own file. Distilled from the `analyst` learnings inbox via `agent-maintenance` skill §5; see
> `claude/analyst/kaizen/history.md` (2026-08-09) for the promotion record.

## Byte-identity confirmation via an AST line-range hash

When a plan or gate requires proving an artifact ("this function is locked, prove it's
byte-for-byte unchanged") wasn't touched — even when its line offset moved (e.g. code was
inserted above it) — a `git diff` line-range read can mislead. Instead:

1. `ast.parse` the file at each revision, walk for the target `FunctionDef`/`ClassDef` by name.
2. Hash `src.splitlines()[node.lineno-1:node.end_lineno]` extracted from `git show <rev>:<path>`
   at each revision under comparison.
3. If the hashes match, the function is byte-identical regardless of where it now sits in the
   file.
4. If they differ, a second pass comparing `ast.dump()` with docstring `Expr` nodes stripped
   distinguishes a docstring-only edit from a real behavioral change.

Origin: proved `executor._drive_loop` identical (same hash, same byte length) across three
revisions despite its line offset moving by 14 lines, which a naive `git diff` line-range check
would have mis-read as "moved, can't tell."

## Verifying an uncommitted diff without mutating the working tree

Two techniques for gathering *executed* evidence about a change that is still uncommitted (no
`git stash`/worktree available, or blocked by the harness's Bash safety classifier — see
`skills/agent-standards/claude-code.md` § Bash tool environment for that layer):

**(a) Load a `HEAD` version of a module alongside the working-tree one, via a stub package.**
`pip install -e '.[dev]'`-style editable installs register a `MetaPathFinder` that is consulted
before `sys.path`, so copying the tree to a scratch dir and prepending it to `PYTHONPATH` still
imports the *working-tree* version — it does not shadow it. What works instead, with zero
working-tree writes:
```python
# 1. Extract the HEAD version of the file to scratch, unmodified:
#    git show HEAD:path/to/module.py > $SCRATCH/module_head.py
# 2. Build a stub package in-process to satisfy the module's own relative imports
#    (only needed if the module does `from . import something`):
import types, sys, importlib.util
pkg = types.ModuleType("hp"); pkg.__path__ = []; sys.modules["hp"] = pkg
sys.modules["hp.config"] = <a ModuleType exposing whatever `config` symbols the module needs>
spec = importlib.util.spec_from_file_location("hp.module_head", f"{SCRATCH}/module_head.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
# 3. `old.some_function` and the working-tree import of the same function are now both
#    live in one process — diff their behavior directly on a table of inputs.
```
Works cleanly when the module's only intra-package import is a small, enumerable set (`from .
import config`); a module with a wider relative-import fan-out needs more stubs. Use this to
independently confirm a claim like "N tests were red before the fix" without trusting the
implementer's narration, and to catch an undisclosed behavior change versus `HEAD` in the same
pass.

**(b) `cp -r` a scratch copy of the working tree, then reverse-apply the diff there with `patch
-p<N> -R`.** Stronger isolation than even `git stash --keep-index`: the tracked working tree is
never touched at all.
```bash
cp -r <component>/server "$SCRATCH/server-check"
git diff -- <file> > "$SCRATCH/x.diff"
(cd "$SCRATCH/server-check" && patch -p<N> -R < "$SCRATCH/x.diff")   # N = path depth from the copy root
# symlink .venv (and any repo-root-relative script dir fixtures shell out to) into the
# scratch copy so its pytest runs for real, including live-integration tests.
```
Confirm zero residue on the real tree with `git status` before/after. This substitution is
sound **specifically because it doesn't touch the tracked tree at all** — a strictly stronger
property than a blocked `git stash`, not a lower-visibility route to the same effect. It is
**not** a general "when something blocks you, find another way to the same effect" license: the
justification is the zero-touch property, not that a workaround exists — the block was doing its
job; this substitute earns the exception on its own zero-touch merits. Confirm your own
substitute is actually zero-touch before reusing this, not just that it wasn't caught.

## Re-gating a fix pass by line-number invariance

When a fix pass lands on an *uncommitted* tree (so both the reviewed change and the fix pass are
uncommitted, `git diff` shows their union, and there is no pre-fix baseline — no stash, no
`.orig`), and the implementer claims "all N prior findings/pins survive unchanged": check that
every `file:line` your round-1 review quoted still lands on the same construct in the new
version.

- If a quoted line number is **unmoved**, the code above it in the file is unedited (an edit
  above would have shifted everything below).
- If a quoted line number **moved**, confirm the new blocks are **pure insertions** at the old
  position (i.e., everything below your quoted line shifted by a consistent offset, and nothing
  *above* your earliest quoted line moved at all) — that proves the fix pass only added code,
  it didn't touch what you already reviewed.

Combined with reading the bodies at their new locations, this is sufficient; line-number
invariance alone or body-reading alone is not. This is also why it matters to **quote `file:line`
for every finding and every pin you certify** in round 1 — that's what makes a later re-gate
provable without a diff.
