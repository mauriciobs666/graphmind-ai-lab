# Typed Decision Models, Plan 4 — Implementation Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## Pass 1 — U5.1 scaffold (2026-10-31)

### Scope & verdict

**Reviewed:** the untracked `decision-models/` tree (`pyproject.toml`, `.gitignore`, `README.md`,
`AGENTS.md`, `setup.sh`, `decisionmodels/__init__.py`, `tests/test_fr8_isolation.py`,
`tests/test_package.py`, `docs/HISTORY.md`) and the two added root `AGENTS.md` lines. **Baseline:**
`docs/plans/typed-decision-models4.md` v1.2 §3.1 and §4.1 (U5.1 row), v1.3
`docs/plans/typed-decision-models.md` §3.4 and §3.2.2, `docs/plans/typed-decision-models2.md`
§3.5, `docs/plans/typed-decision-models3.md` §2 (decision-models cannot import model-bench), and
FR-8/AC-7 in `docs/requirements/typed-decision-models.md`. Later units (U5.2 onwards) are out of
scope except where U5.1's text makes claims about them.

**Verdict: approve with suggestions.** The done-condition holds. I ran the suite from
`decision-models/`: `.venv/bin/pytest -q` gives 20 passed, `ruff check .` is clean and `bash -n setup.sh`
passes. Planting `import redis` in `decisionmodels/__init__.py` reddens
`test_decisionmodels_reaches_only_stdlib_and_itself`. No blocker and no major. The findings are about
pinning the check (F1), stating its limits accurately (F2, F3), and one design boundary that is the
architect's call (F4).

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded (`GRAPHS`: only `cpg_falkorchat`); the component is new and untracked, with two source files read in full.

**Method.** I used a fresh `PYTHONPYCACHEPREFIX` and `-p no:cacheprovider` for every run. Each
mutation was made against a copy kept aside in the scratchpad, then restored and checked equal with
`cmp`. No git command changed the tree. The final suite run after all mutations gave 20 passed.

### Findings

**F1 — minor — three forbidden fragments are unpinned; deleting any one leaves the suite green.**
*Owner: `coder`.* Verified by mutation: removing `"falkorchat"` (M1), `"falkor_chat"` (M2) or
`"falkordb"` (M3) from `FORBIDDEN_FRAGMENTS` (`tests/test_fr8_isolation.py:35-42`) gives 20 passed
each time. Every planted case that mentions these names is an *import*, which the allowlist refuses,
and the case's `expected` substring (`'falkordb'`) matches the allowlist message too. Removing
`"falkor-chat"`, `"redis"` or `6379` does redden (M4–M6). This matters because, without M3, a string
reach such as `subprocess.run(["docker", "exec", "falkordb-dev", ...])` (E3b, red today) would pass.
**Fix:** add one planted case per fragment that refuses it as a *string literal*. Examples:
`"falkordb-dev"` in a `subprocess` argv, `Path("falkor_chat")`, and `"falkorchat.app"` as a string.
Assert the `string literal reaches '<fragment>'` message, not just the bare name. Then re-run M1–M3
and confirm each goes red.

**F2 — minor — the test's docstring overclaims the dynamic-import coverage, and the README says
"enforces".** *Owner: `coder`.* `tests/test_fr8_isolation.py:9-10` says `__import__`/`import_module`
calls count as imports, and a non-literal name is refused. Three evasions of the dynamic-import
detection stayed green, verified by planting each in `decisionmodels/__init__.py`: the aliased form
`from importlib import import_module as im; im("modelbench")` (E5), `getattr(importlib,
"import_module")("modelbench")` (E6), and `vars(builtins)["__import__"]("modelbench")` (E6b). The
full table of 16 planted evasions, 8 green, is in the Appendix. The docstring's closing caveat (run-time assembly is not seen) is honest,
but neither `README.md:21-23` ("enforces that") nor the `AGENTS.md` invariant states the residual.
**Judgement:** the residual is acceptable for a static tripwire. FR-8's realistic failure is an
implementer reaching a real path or database by habit, which almost always uses a literal. Two more
layers stand behind it: AC-7 is checked on items (`origin: "synthetic"` plus the generator hash,
v1.3 §3.4), and U5.7's `guard_hook.py` confines the labelling session's reads. **Fix:** (a) make
the docstring exact: literal-named `__import__`/`import_module` calls by bare name or attribute, with
aliases, `getattr`, `exec`, `runpy`, bytes and assembled strings or ports not seen. (b) Change
README "enforces" to "is a static tripwire for". (c) Add one clause to the `AGENTS.md` FR-8 bullet
citing the docstring for what the check cannot see. Optional, cheap hardening, which I prototyped
on a scratch copy: scan `bytes` constants (decode as latin-1), and add the `asname` of an
`ImportFrom` of `import_module`/`__import__` to the call names. With those, E5 and E7 go red and the
14 planted cases plus the negative control stay green. Each needs its own planted case.

**F3 — minor — `AGENTS.md` misstates the boundary: the shingle script is not "the one sanctioned
reader of model-bench data".** *Owner: `coder`.* `decision-models/AGENTS.md:22-23`. v1.3 §3.2.2 gives
decision-models a guarded equality test. That test byte-compares its contract copies against
`model-bench/tests/fixtures/decision-contract/`, so it reads model-bench data (U5.2). Plan2 §3.5's
rule is narrower: only the script reads the **side-check items**. This was verified against the plan
text. As written, the invariant will be false the moment U5.2 lands, and a context file is where a
false invariant hurts most. **Fix:** change "the one sanctioned reader of model-bench data" to
"the one sanctioned reader of the side-check items
(`model-bench/packs/guard-judge-understanding/`)".

**F4 — minor — `scripts/` and `agents/*.py` are entirely outside FR-8's check.** *Owner:
`architect`, via `teco`.* The implementer is conformant: plan2 §3.5 says "the FR-8 test still scans
`decisionmodels/` only". Excluding the script from the *side-check fragment* is necessary, because
it must name `guard-judge-understanding`. But excluding whole directories also removes the import
allowlist and the falkor-chat, FalkorDB and Redis fragments from every future script and from
`agents/guard_hook.py`, which runs inside the labelling session. A second script that reads a real
falkor-chat thread into `data/` would then be guarded only by the `AGENTS.md` prose. **Suggestion
(a design decision, not prototyped):** scan `scripts/` and `agents/` with the same checker plus a
per-file exemption map, `{"scripts/import_side_check_shingles.py": {"guard-judge-understanding"}}`.
Two planted cases go with it: that script naming `falkor-chat` reddens, and a *different* script
naming the side-check slug reddens. The natural unit is U5.10, when the script lands. This is not a
U5.1 blocker.

**F5 — nit — the bare `redis` substring false-positives on ordinary words.** *Owner: `coder`.*
Verified: planting `("predisposition", "redistribuir")` reddens the real-package test (E15). U5.3's
pt-BR seed axes and rubrics are plausible sources of such words. The `AGENTS.md` rule "never widen
either to make a unit green" would then turn a non-reach into a plan-change escalation. **Fix:**
match `(?<![a-z])redis(?![a-z])` on the lower-cased literal. I checked it: it hits `redis://`,
`redis-cli`, `redis_url`, `REDIS_HOST` and `falkordb-redis`, and misses `predisposition` and
`redistribuir`. The trade-off is that it misses `myredis`. Add `"predisposition"` to the negative
control.

**F6 — nit — the root `AGENTS.md` bullet's "(FR-8)" has no referent at root.** *Owner: `coder`.*
In an always-loaded root file, "FR-8" does not say which requirements document is meant. Write
"(synthetic inputs only)", or drop it, because the Component-docs row already cites
`docs/requirements/typed-decision-models.md`. Otherwise both lines are one line each (101 and 171
characters), and the row matches model-bench's shape.

**Judgement calls (asked by the caller):**
1. *Scan `decisionmodels/` only:* **accept.** This is conformant to plan2 §3.5. The wider boundary
   is F4's architect question.
2. *Docstrings exempt:* **accept.** Module docs must be able to describe FR-8. The exemption is
   pinned: removing it reddens the negative control (M7). Its only evasion is reading `__doc__` back
   as data (E8), which is contrived.
3. *A forbidden list wider than v1.3's literal:* **accept.** v1.3 §3.4 already names `falkordb` and
   `redis`. The side-check slug implements plan2 §3.5's "must never read the 85 items", and refusing
   `modelbench` implements plan3 §2. Allowing the text "model-bench" is right: decision-models
   legitimately names model-bench's contract fixtures. The list's pinning gaps and its one false
   positive are F1 and F5.
4. *`pythonpath = ["."]` and reading the version from `pyproject.toml`:* **accept.** Both checks
   can fail. Bumping `__version__` to `0.1.1` reddens `test_version_matches_pyproject`, and adding
   `dependencies = ["requests"]` reddens `test_no_runtime_dependencies`; both are verified. Reading
   `pyproject.toml` also avoids editable-install metadata, which goes stale until reinstall
   (inferred, not run).
5. *No `BACKLOG.md`, `labeler.py` or `live` marker:* **accept.** Root `AGENTS.md` says modules
   adopt `BACKLOG.md` "when first needed". `labeler.py` is void under plan4 §1 and §3.1. No live
   test exists yet, so a marker would select nothing. There is one carry-forward for `teco`'s
   ledger: the first unit that adds a network or LM Studio test must add model-bench's
   `-m "not live"` `addopts`. `AGENTS.md:33` already promises a network-free default suite.

**(d) `docs/HISTORY.md`.** The entry is accurate. I counted 14 planted cases plus a negative control,
and 3 + 14 + 1 + 2 = 20 tests, which matches the 20 passed I ran. The pins are model-bench's: the
`pytest`/`ruff` ranges and the ruff select are identical in `model-bench/pyproject.toml`. "`setup.sh`
(model-bench's, renamed)" is true, verified by a name-normalized diff: only two comment lines differ.
The format follows model-bench's `**What:**`/`**Verified:**` style. I could not re-observe the
implementer's six `__init__.py` mutants as runs, but I reproduced the `import redis` one (C1). The
entry needs no change for this pass. If F1, F2, F3 or F5 is fixed in U5.1, add one line to the same
entry rather than a new one.

### What's solid

- The AST walk covers imports inside functions, `from` imports, relative imports and literal
  dynamic imports. A non-literal `import_module(name)` is refused, which is pinned (M8). Allowing
  relative imports is pinned too (M9).
- `test_the_scan_sees_the_real_package` prevents a vacuous green from a renamed package.
- The planted cases run in a scratch package under `tmp_path`, so the checker is tested without
  touching the real tree. The negative control exercises both the docstring exemption and allowed
  stdlib network modules.
- `README.md` and `AGENTS.md` state the present tense with no history: 268 and 263 words, no line
  over 700 characters. Work status is limited to one "Scaffold only" line in each, which the next
  unit rewrites.
- `.venv`, `*.egg-info`, `.pytest_cache` and `.ruff_cache` are all ignored. `git status
  --untracked-files=all` lists only the nine authored files.

### Open questions

- F4: should FR-8's static check extend to `scripts/` and `agents/` with per-file exemptions, in
  U5.10? This is the architect's call, because it amends plan2 §3.5's "scans `decisionmodels/` only".

### Appendix — evasion and checker mutants

Evasions were planted by appending to `decisionmodels/__init__.py`. I ran
`test_decisionmodels_reaches_only_stdlib_and_itself`, then restored the file and confirmed it with
`cmp`. **Green** means the evasion is unseen.

| ID | Planted | Result |
|---|---|---|
| C1 | `import redis` (positive control) | red |
| E1 | `"red" + "is"` | green |
| E2 | `os.path.join(os.environ.get("REPO", ".."), "falkor" + "-chat", "server")` | green |
| E3 | `subprocess.run([..., "redis" "-cli"])` (implicit concatenation, joined at parse time) | red |
| E3b | `subprocess.run(["docker", "exec", "falkordb-dev", "true"])` | red |
| E4 | `socket.create_connection(("localhost", "6379"))` (string port) | green |
| E4b | `socket.create_connection(("localhost", 6000 + 379))` | green |
| E4c | `socket.create_connection(("localhost", 6380))` | green |
| E5 | `from importlib import import_module as im; im("modelbench")` | green |
| E6 | `getattr(importlib, "import_module")("modelbench")` | green |
| E6b | `vars(builtins)["__import__"]("modelbench")` | green |
| E7 | `b"...redis"` bytes literal | green |
| E8 | a function docstring `"redis://localhost:6379"` read back via `__doc__` | green |
| E9 | `runpy.run_path("../model-bench/modelbench/packs.py")` | green |
| E10 | `importlib.util.spec_from_file_location("mb", "../model-bench/...")` | green |
| E11 | `sys.path.insert(0, "../falkor-chat/server")` | red |
| E12 | `Path("..").glob("model-bench/packs/guard-judge-*/items.jsonl")` | green |
| E13 | `sys.path` + `import_module("modelbench")` (literal) | red |
| E14 | `exec("import modelbench")` | green |
| E15 | `("predisposition", "redistribuir")` (false-positive probe) | red |

Checker mutants were applied to `tests/test_fr8_isolation.py`. I ran the full suite, then restored
the file and confirmed it with `cmp`.

| ID | Mutation | Result |
|---|---|---|
| M1 | drop `"falkorchat"` fragment | 20 passed (unpinned, F1) |
| M2 | drop `"falkor_chat"` fragment | 20 passed (unpinned, F1) |
| M3 | drop `"falkordb"` fragment | 20 passed (unpinned, F1) |
| M4 | drop `"falkor-chat"` fragment | 1 failed |
| M5 | drop `"redis"` fragment | 2 failed |
| M6 | `FORBIDDEN_INTS = frozenset()` | 1 failed |
| M7 | no docstring exemption | 2 failed |
| M8 | non-literal dynamic import not reported | 1 failed |
| M9 | absolute `from` imports skipped | 2 failed |
| M10 | treat `bool` as int | 20 passed (equivalent mutant: `True != 6379`) |
