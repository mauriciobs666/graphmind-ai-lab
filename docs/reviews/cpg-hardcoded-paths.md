# Review: cpg hardcoded-path removal (uncommitted)

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

**Scope:** uncommitted diff vs `eca54ba29338882b91cfc2baea63baf7662e0373` (52 files: `cpg/spikes/frontend-fitness/{adjudication/tools,labeller-scripts,labels/superseded}`, `cpg/tests/test_eval_tools.py`). Root `REVIEW.md` read; no `cpg/REVIEW.md`.
**CPG:** considered, not relevant — path-literal substitution in one-off scripts; text diff and execution settle it.
**Verdict: approve with suggestions** (no blockers).

## Evidence run
- Old literals taken via `git show <sha>:<path>` for all 51 script files (every language dir: cpp, csharp, go, java, javascript, kotlin, python, labeller-scripts, labels/superseded). The `parents[N]` depth was checked against each directory layout: N=5 for `tools/<lang>/` and `labels/superseded/<lang>/`, N=4 for `labeller-scripts/<lang>/`, and 2 for `tests/`. All resolve to the `cpg` dir (or repo root for the test), so the new path equals the old one modulo the home prefix. `git grep` for `$HOME` or `/tmp/claude` in the touched files now returns nothing.
- `python3 -m py_compile` on all 52 files: clean. I imported or ran no script, because they write into data dirs.
- `.venv/bin/python -m pytest tests/test_eval_tools.py tests/test_contact.py`: 453 passed, 1 failed (see 4).

## Findings
**Minor 1 — scratch-path scripts change what they resolve to.** Six scripts (cpp/build_queue, csharp/canonicalize, java/build_queue, python/assemble, kotlin/w.py, and the csharp `NORM` path) pointed at the session scratchpad `/tmp/claude-1000/.../13504717-.../scratchpad/...`. They now use `$CCG_SCRATCH` or `tempfile.gettempdir()`, i.e. `/tmp`.
- The old session scratchpad still exists, but it has no `s57_*` dirs and no `S.pkl`, so the old paths were already broken: writes to `s57_java` or `s57_cpp` fail on a missing dir, and `S.pkl` fails to load.
- The new default is also broken in the same way, because `/tmp/s57_*` and `/tmp/S.pkl` are absent. It also changes the semantics: `python/assemble.py` has `S=_SCRATCH+'/'`, so a bare `/tmp/` could pick up unrelated files. These scripts need `CCG_SCRATCH` set and populated.
- Suggestion: document `CCG_SCRATCH` in the spikes README, or fail fast when the dir is missing.

**Minor 2 — style.** The change adds 40+ lines of `import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[N])` and `import os, pathlib as _pl, tempfile as _tf; ...`. These are semicolon one-liners placed mid-file after other statements, plus redundant `(…+"…")` parens, e.g. `Path((_CPG+"/x"))`. The result is functional but ugly, and it is repeated per file where the same file already imports `Path`. Harmless in spike or superseded scripts.

**Nit — test hygiene.** `REPO_AGENTS` is now `parents[2]/AGENTS.md`. That is correct, and `test_eval_tools.py` passes.

## Check 3 — cwd mutation
`CAP_ROOT = _INIT["cwd"]` comes from the captured stream, so it is a fixed data value, not derived from Home. `str(Path.home())` replaces `$HOME` only as the *mutated* value. `ROLES_DIR` is `CAP_ROOT.rsplit("/",1)[0]`, so the mutated value is distinct from CAP_ROOT unless CAP_ROOT's cwd equals `$HOME`. I did not print the actual CAP_ROOT value. The test passed here (HOME is `$HOME`). Inference only: in a checkout where the captured cwd equals the runner's home, `cwd` would not mutate and `init-cwd` would go undetected. Low risk; pin to a literal such as `"/nonexistent-home"` for determinism (optional).

## Check 4 — test_contact
Failure message: `derivedFrom ebe66607… is not an ancestor of HEAD` (`make_contact.py:345`). `make_contact.check` reads only git blobs of `derivedFrom`, never the working tree, and the touched files are not in contact.json's inputs, which are falkor-chat and salesperson paths. This change neither causes nor masks it; it is a git-history precondition, failing independently of this change (confirmed by the message; I did not check it out at HEAD).

## Solid
The `parents[N]` depths are correct for every directory family, no literals remain, py_compile is clean, and the suite is otherwise green (453 passed).
