# K-083 demo bring-up opencode-config fallback and ensure-step leftovers — Code review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-083

## Scope & verdict

**Reviewed:** the uncommitted working tree against HEAD `45244f1e`. That covers
`scripts/shlib/opencode_config.sh` (new) and `test_opencode_config_resolution.py` (new), plus the
diffs to `start_demo.sh`, `start_server.sh`, `start_agent_team.sh`, `ensure_demo_models.sh`,
`ensure_demo_models.py`, `test_ensure_demo_models_script.py`, `falkor-chat/AGENTS.md`,
`falkor-chat/README.md` and `salesperson/README.md`. The baseline is `BACKLOG.md` §K-083 and the
definition of done in `docs/plans/demo-opencode-config-coordination.md`. I also read the
`claude/teco/hooks/ensure-environment.sh` caller and the out-of-scope maintenance scripts.

**Verdict: approve with suggestions.** There are no blockers and no majors. All four DoD items hold:

- a fallback is defined once and sourced by all four scripts;
- a no-env `start_demo.sh` resolves to `http://localhost:1234`, which answers `200`, while the
  shared file's `192.168.0.69:1234` times out (curl exit 28);
- all four leftovers are fixed and tested;
- the docs are updated.

The three minor findings are worth fixing before close. Two are cheap.

CPG: considered, not relevant — the change is in bash entry points plus one standalone Python
script outside the server package. I verified it by running scripts and tests directly, not by
call-graph analysis.

**Evidence run:**

- **Targeted tests:** `pytest -q tests/test_ensure_demo_models_script.py
  tests/test_opencode_config_resolution.py` gave **60 passed**.
- **Helper from another cwd:** worked.
- **Helper through a symlink:** failed, at HEAD as well (see below).
- **Helper with `HOME` unset:** failed.
- **`git check-ignore`:** checked on both `lib/` and `shlib/`.
- **`read_models` against raw-socket servers:** checked for accept-and-close, a truncated body and a
  garbage status line, with the current code and with a scratch copy that has the `except` clauses
  reordered.
- **`ModelGateway.from_env()` under both opencode files:** checked.
- **LM Studio probe:** read-only `curl` only. No `lms` command was run.

## Findings

### m1 (minor) — A connection closed before any response now gets "cut off or malformed" instead of "unreachable", which drops the config-fix hint

**Where:** `scripts/ensure_demo_models.py` `read_models`. The new `except http.client.HTTPException`
clause sits before `except (urllib.error.URLError, OSError)`.

**Why it happens:** `http.client.RemoteDisconnected` subclasses both `ConnectionResetError` (an
`OSError`) and `BadStatusLine` (an `HTTPException`). `urllib`'s `do_open` does not wrap
`h.getresponse()` in `URLError`, so the exception reaches `read_models` raw. I checked the MRO and
the source.

**Verified:** a socket that accepts, reads the request and closes now yields `Refusal 1 | the answer
from … was cut off or malformed (RemoteDisconnected(...))`. At HEAD it went to the `OSError` clause:
"LM Studio API unreachable … The URL is the lmstudio baseURL in FALKORCHAT_OPENCODE_CONFIG (…) Fix:
… opencode.local.json".

**Why it matters:** the exit code is the same (1). But an accept-then-drop connection, for example
a port relay with nothing behind it, is the "wrong URL" symptom that K-083 is about. That message
now omits which config file was used and how to fix it. The new clause's comment ("The server
answered, then broke the HTTP exchange") is also false for this case, because nothing was answered.

**Fix:** move the `(URLError, OSError)` clause above the `HTTPException` clause. I verified this on
a scratch copy:

- `RemoteDisconnected` gives "unreachable";
- `IncompleteRead` and `BadStatusLine` still give "cut off or malformed", exit 1, with no bypass
  hint.

**Test that would catch a regression:** add a rig case where the server accepts and closes, and
assert `"unreachable" in out.stderr` and `returncode == 1`. The existing truncation test keeps the
other half pinned.

### m2 (minor) — The bring-up's own child scripts don't see the resolved value

**Where:** the helper sets the variable but does not export it (`opencode_config.sh:9`, by design).
Each caller exports it late:

- `start_demo.sh:364`, after `create_workspace.sh` runs in step 4;
- `start_server.sh:250`, after `create_workspace.sh` (window line 62);
- `start_agent_team.sh:255`, after `create_workspace.sh`.

**What happens:** `create_workspace.sh` runs `pin_workspace_embedding_model.sh`, which applies its
own `:-$HOME/.config/opencode/opencode.json`. One bring-up run therefore reads two different files:

- the ensure step and uvicorn get `opencode.local.json`;
- the pin step gets the shared `opencode.json`.

**Impact today (verified):** none. `ModelGateway.from_env()` gives the same embedding ref,
`lmstudio/text-embedding-qwen3-embedding-0.6b`, from both files, and `pin()` makes no network call.

**Why it matters anyway:** the gap is old. The pattern existed at HEAD, and `start_agent_team.sh` has
always had it. But this change rewrites `salesperson/README.md` to drop the
`FALKORCHAT_OPENCODE_CONFIG=… ./scripts/start_demo.sh` env prefix, and that prefix used to export
the value to every child. On a box that has only `opencode.local.json`, pin now prints a WARNING
and leaves `ws:demo` unpinned. The same happens on a box where the two files' embedding refs
diverge, except that pin then writes the wrong ref.

**Fix:** add `export FALKORCHAT_OPENCODE_CONFIG` right after `resolve_opencode_config` in the three
callers. Alternatively, make the helper export the variable and update its comment. The later
`export` lines then become redundant but harmless.

**Assertion that would catch it:** in `Tree`, the `create_workspace.sh` stub already records the
env it receives. Assert `tree.config_seen_by("create_workspace.sh") == str(local)` in
`test_entry_point_local_present` for `start_server.sh` and `start_agent_team.sh`. For
`start_demo.sh`, the ensure stub has to exit 0 so the run reaches step 4.

### m3 (minor) — Two live docs still state the old default; fix `seed_eval_corpus.sh`'s comment now

1. **End-user manual.** `falkor-chat/docs/manuals/llm-provider-config.md:75` says
   "`./scripts/start_server.sh` sets a convenience default of `$HOME/.config/opencode/opencode.json`".
   That is now false. It was not in the coordination doc's documentation-impact scan. Route it to
   `tico`, who owns manuals. Suggested wording: "…uses `$HOME/.config/opencode/opencode.local.json`
   if present, else `$HOME/.config/opencode/opencode.json`, and says which on stderr".
2. **`seed_eval_corpus.sh:25`.** The comment reads "dev script supplies the convenience default
   instead, same as start_server.sh". It is now wrong, and it is a one-line comment edit, so fix it
   in this change rather than deferring it. Leaving it would put a false cross-reference next to
   the exact difference the coordination doc calls deliberate. Suggested wording: "…supplies the
   shared file as its default, deliberately — unlike start_server.sh, which falls back through
   `shlib/opencode_config.sh`". The behaviour stays out of scope, as decided. See open question 1.
3. **Nit.** `claude/data-scientist/lm-studio-model-notes.md:87` says "defaulting to
   `$HOME/.config/opencode/opencode.json`". That is now only half true. It is not a falkor-chat doc,
   so mention it to `cobb`/`data-scientist` if you want it corrected. It does not block this change.

### n1 (nit) — `HOME` unset plus an explicit env var now aborts

**Where:** `opencode_config.sh:20-21` assigns `local local_override="$HOME/…"` before testing
whether the variable is set. Under the callers' `set -u` this aborts ("HOME: unbound variable") even
when `FALKORCHAT_OPENCODE_CONFIG` was passed explicitly. At HEAD, `${FALKORCHAT_OPENCODE_CONFIG:-$HOME/…}`
never expanded `HOME` when the variable was set. Both were verified by running them.

**Why it's only a nit:** the teco hook keeps `HOME` (`KEEP_ENV`), and `start_agent_team.sh` already
behaved this way.

**Fix:** move the two `local` assignments into the `else` branches. Or leave it.

### n2 (nit) — The settle-knob docs understate the validation

`ensure_demo_models.sh` says "a non-numeric value is a refusal". The `.py` docstring says
"non-numeric, negative or non-finite". The code also refuses a zero interval
(`allow_zero=False`), and the tests cover that case. One shared phrase would fix both, for example
"a value that isn't a finite number of seconds (> 0 for the interval) is a refusal".

### n3 (nit, pre-existing and adjacent) — The README's manual uvicorn snippet hardcodes the shared file

`salesperson/README.md:143` sets `FALKORCHAT_OPENCODE_CONFIG="$HOME/.config/opencode/opencode.json"`.
The rewritten paragraph a few lines above says that file is stale on the usual dev box. Consider
`opencode.local.json` there, or a comment.

## Answers to the brief's specific checks

1. **Sourcing robustness.** All checks below were run.
   - **Another cwd:** relative-path invocation from another cwd works for `ensure_demo_models.sh`.
     No script `cd`s before it sources the helper.
   - **Symlink:** invocation through a symlink fails with `…/shlib/opencode_config.sh: No such
     file`. The HEAD version fails through a symlink too ("venv missing"), because `REPO_DIR` also
     derives from `dirname "$0"`. So this is not a regression, and no symlinks to these scripts
     exist.
   - **The teco hook:** it invokes `start_agent_team.sh` by absolute path (`$FC_SCRIPTS`, built
     from `readlink -f "$0"`) with cwd `falkor-chat/` and `env -i` that keeps `HOME`. That works.
   - **`set -e`/`set -u`:** the helper's tests sit in `if` conditions, which are exempt from
     `set -e`. `${VAR:-}` is safe under `set -u`. The only interaction is n1.
2. **Ordering.**
   - In every caller, `usage`/`--help` text is static and runs before resolution, with no use of
     the variable.
   - Resolution precedes the first use in every caller: the "Model config:" banners at
     `start_demo.sh:349`, `start_server.sh:231` and `start_agent_team.sh:243`, and the ensure call.
   - `start_demo.sh:243` passes the resolved value explicitly to the ensure step. Because it is
     non-empty there, the step's own resolve stays silent, so there is no duplicate INFO line.
   - `start_demo.sh:364` exports the value to uvicorn.
   - The one gap is the children before the export (m2).
3. **The exit-3 test.**
   - `Tree.__init__` copies `_SCRIPTS / script` and the real helper with `shutil.copy2` at test
     time. There is no checked-in fixture, so it always tests the in-place bytes.
   - It asserts that rc 3 maps to the bypass message, that rc 1 maps to the no-bypass message, and
     that no seed or uvicorn runs. Either way the run exits 1.
   - If `start_demo.sh` later grows a dependency that the tree doesn't stub, the test fails loudly
     rather than passing stale.
   - The `.py` exit-3 cases are covered separately, and `ensure_demo_models.sh` `exec`s the
     Python, so the chain is covered end to end in two halves.
4. **Docs.**
   - The changed sentences in `falkor-chat/AGENTS.md` (row 87 and the demo paragraph),
     `falkor-chat/README.md` and `salesperson/README.md` match the code. I verified that on this box
     the local file is the reachable one.
   - Row 87 is under the 700-character bar. Row 85's 980 characters predate this change.
   - Remaining staleness is covered in m3. Fix `seed_eval_corpus.sh` in this change.
5. **Same-revision interactions.**
   - `settle_config()` is the first statement of `ensure()`. A knob refusal therefore precedes any
     read or load, and the test asserts `rig.calls() == []`. It cannot fire after a load.
   - An `HTTPException` during the post-load settle re-read is exit 1 after a successful load. The
     models stay loaded and a re-run is a no-op. That is acceptable and consistent with "never
     unloads".
   - The new before-load test pins the wait to the post-load path. A wait moved before the load
     would settle and exit 0, but the test asserts exit 1 and no "waiting for". The strengthened
     post-load test catches removal of the wait.
   - Pre-existing and not introduced here: a post-load re-read that hits a 404/405 would exit 3 and
     offer `DEMO_ENSURE_MODELS=0` after a load. This is only theoretical, so I am not filing it as a
     finding.

## What's solid

- **One definition.** `start_agent_team.sh`'s messages are byte-identical: the helper's
  `echo` strings match HEAD's inline block once its two variables are substituted, and nothing
  else in the script used those removed variables.
- **The `shlib/` rationale is verified.** `git check-ignore -v` reports that `scripts/lib/…` is
  ignored by `.gitignore:17 lib/`, and that `shlib/` is not ignored.
- **The resolution tests drive the real scripts.** They run in a stubbed tree copy with a tmp
  `HOME`, so they never read the real `~/.config/opencode`, and they include the "explicit var
  wins over present local" case. The parametrised "no inline default" assertion guards against a
  second copy creeping back.
- **The settle-knob refusal is well reasoned.** It refuses rather than defaulting, which matters
  because NaN would make the wait unbounded. Its tests cover non-numeric, NaN, negative, infinite
  and zero-interval values, and that zero settle is accepted.
- **The HTTPException fix keeps exit 1.** It does not use exit 3, so a broken answer is never
  taken as evidence for the bypass.

## Open questions

1. `seed_eval_corpus.sh` and `migrate_embeddings.sh` make real embedder calls through the shared
   file's baseURL, which is unreachable on this box. "They deliberately name the shared file" holds
   for `pin` (no network), but for these two it means they fail here unless the variable is
   exported. That is the stakeholder's call, not a defect in this change. If it is to stay, the
   comment from m3 should say so.

## Pass 2

2026-10-22. I re-checked the working tree after `devops`'s fixes and `tico`'s manual edit. The
targeted suite gives **69 passed** (I re-ran it).

**Verdict: approve.** The fixes introduced no new blocker, major or defect.

**Dispositions:**

- **m1 — fixed.** The clauses are reordered, with a comment explaining the order. I re-ran the
  raw-socket harness against the in-place file:
  - accept-and-close now gives "LM Studio API unreachable" with the config hint, exit 1;
  - `IncompleteRead` and `BadStatusLine` still give "cut off or malformed", exit 1.

  My recommended fix held up. It is pinned by
  `test_connection_closed_before_any_answer_is_unreachable_with_the_config_hint`.
- **m2 — fixed, inside the helper rather than in the callers.** Executed: a child shell sees the
  resolved value in all three branches (explicit, local, shared). The new
  `test_create_workspace_sees_the_resolved_config` covers all three callers.
- **m3 (`seed_eval_corpus.sh`) — fixed.** The comment now states the deliberate difference and
  the stale-baseURL caveat. Item 3 is out of scope per the brief.
- **m3 (manual, `llm-provider-config.md` Walkthrough 1 step 2) — fixed and fact-checked.** It
  makes three claims, all verified:
  - The fallback applies when the variable is unset, and the order is: `opencode.local.json` if it
    exists, otherwise the shared file with a warning.
  - The scripts "say which on startup", and that holds in both fallback branches (INFO or
    WARNING); with the variable set there is silence, which is outside the sentence's "if you
    leave it unset".
  - The list of four scripts matches the four that source the helper.

  The wording is suitable for its readers.
- **n1 — fixed.** Executed: with `HOME` unset and the variable set explicitly, `--print-set`
  succeeds. With `HOME` unset and the variable unset, it still aborts on `HOME`, the same as HEAD.
  That is correct, since there is no path to build.
- **n2 — fixed.** The `.sh` header and the `.py` docstring now both say "finite, > 0 for the
  interval".
- **n3 — fixed.** The `salesperson/README.md` uvicorn snippet now uses `opencode.local.json`.

**Cross-checks:**

- **n1 against m2.** The early-return branch exports before it returns, so the explicit case is
  exported too. I executed a child shell to confirm.
- **The callers' later `export` lines.** They cannot re-export a different value: no caller
  assigns `FALKORCHAT_OPENCODE_CONFIG` after `resolve_opencode_config`. The only assignment is
  `start_demo.sh:243`, an env-prefix that passes the same value through.
- **The earlier export reaches more children.** Exporting earlier now also reaches `seed_demo.sh`,
  `seed_workflows.sh`, `start_falkordb.sh` and the seed/verify scripts. None of them reads the
  variable; I grepped `scripts/`.
