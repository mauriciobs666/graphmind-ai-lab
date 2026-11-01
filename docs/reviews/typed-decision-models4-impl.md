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

## Pass 2 — U5.2 guard contract copies (2026-10-31)

### Scope & verdict

**Reviewed:** the uncommitted work on top of `9ccc2518`, all in `decision-models/`:
- `decisionmodels/render.py` and `decisionmodels/hashing.py`;
- the byte-copies `tests/fixtures/decision-contract/render-guard.jsonl` (14 rows) and
  `input-hash.jsonl` (4 rows);
- `tests/test_decision_contract.py`;
- the edits to `AGENTS.md`, `README.md` and `docs/HISTORY.md`.

**Against:**
- plan4 §4.1's U5.2 row (done when the copies are `cmp`-equal and every row passes);
- v1.3 §3.2, §3.2.1, §3.2.2 and §3.5;
- plan4 §3.2, the request writer that consumes `render.py`.

**Truth sources, read in full for the functions concerned:**
- falkor-chat's real `app._render_judge_user` (`falkor-chat/server/falkorchat/app.py:695-730`)
  and `guards._recent_turns` (`guards.py:538-570`);
- model-bench's transcriptions `_normalize_turns` and `_render_judge_user`
  (`model-bench/modelbench/scoring/classification.py:29-82`);
- model-bench's `input_hash` (`model-bench/modelbench/decisions.py:54-56`, `:168-183`).

**Verdict: approve with suggestions.** The done-condition holds, and I verified each part:
- Both copies are `cmp`-equal to model-bench's.
- `.venv/bin/pytest -q` gives 53 passed, and `ruff check .` is clean.
- A three-way differential agrees on every generated input. It ran over 12,000 generated
  cases plus 720 boundary cases, with no mismatch for normalization, rendering or the composed item
  rendering (Appendix B).

There are no blockers and no majors.

`render.py`'s md5 is `2ce93c54…`. That file is behaviorally sound on the contract's domain: it
matches falkor-chat's real function on every differential case and every vector row. So no
behavior-changing residue from the implementer's failed restore remains. A behavior-neutral residue
would be invisible to this check, but the file reads clean line by line.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; the two modules (135 lines) and every truth-source function were read in full.

**Method.** I used a fresh `PYTHONPYCACHEPREFIX` and `-p no:cacheprovider`. Each mutation was made
against a copy kept aside, then restored and checked equal with `cmp`. At the end I compared the
sha256 of every file under `decisionmodels/` and `tests/` against a baseline taken before the first
mutation: all 8 were unchanged. No git command changed the tree. The final suite run gave 53 passed.

### Findings

**P2-F1 — minor — `render_guard_item`'s turn order is unpinned. Reversing it leaves 53 green.**
*Owner: `coder`.* Verified with mutant R18: `normalize_turns(...)[::-1]` in `render.py:88` gives 53
passed. The vector rows exercise `render_guard_input` over already-normalized `recentTurns`. The
two `render_guard_item` tests use one turn and zero turns. But `render_guard_item` is the function
plan4 §3.2's request writer calls to produce the text the labeller votes on. An order or wiring
slip there would silently change the labelled inputs while every vector stays green.
**Fix:** add one `render_guard_item` test with at least 8 raw rows. Include an invalid row, a speaker
fallback, and a newest turn long enough to force the cap to drop a turn. Write the expected string
by hand from v1.3 §3.2's rule, oldest first. Then confirm R18 goes red.

**P2-F2 — minor — the `HISTORY.md` "Verified" line is not accurate as written.** *Owner: `coder`.*
`docs/HISTORY.md` says "every file was restored `cmp`-equal". By the implementer's own report to
`teco`, one run left two mutants in `render.py`, and they were removed afterwards. The mutant table
is also not reproducible as stated: it lists "the cap check's separator arithmetic", but this
implementation has no separator arithmetic, because it searches directly (judgement call 3).
`HISTORY.md` is the record, so a false line there misleads. **Fix:** replace the mutant sentence with
one present-tense line. It should say that the mutants were re-run after the failed restore was
recovered (md5 `2ce93c54…`), and cite this pass for the independent re-sample. Do not
narrate the incident.

**P2-F3 — minor — turn normalization has no cross-component pin. This answers judgement call 1.**
*Owner: `architect`, via `teco`. The vector would be built by a model-bench `coder` unit.* The rule
is shared by falkor-chat (`guards._recent_turns`, the truth), model-bench (`_normalize_turns`) and
now decision-models. v1.3 §3.2.1 also makes it the join-in renderer's rule. But v1.3 §3.2.2's
vector families do not include it. In this component, the four hand-written tests do pin the rule
as written: my mutants R9, R10, R11, R13, R14 and R15 all go red, and so do the coordinator's two.
Today all three copies agree: 0 mismatches in 12,000 differential cases. What a hand test cannot
catch is drift *at the truth source*, for example falkor-chat changing `RECENT_TURNS_N`. Through
plan4 §3.2, such drift would silently make the labeller see different evidence from what
falkor-chat's judge sees at run time. **Suggestion:** add a model-bench-owned
`normalize-turns.jsonl`, with expected outputs produced by running falkor-chat's real
`_recent_turns` (as `render-guard.jsonl` is produced). Byte-copy it here, and fold it into U5.2b
or U4, which needs the rule anyway for join-in. Until then the hand tests are adequate for a
contract that matches today. This is not a U5.2 blocker.

**P2-F4 — nit — `AGENTS.md` grew 282 → 362 words, partly by restating `HISTORY.md`.** *Owner:
`coder`.* "Current state" was rewritten rather than appended to, which is right. But it opens with
"Landed: …" and then says what landed is in `HISTORY.md`. It will grow again with every U5.x unit.
The new "Contract copies are byte-copies" invariant is genuine new scope and should stay. Its
clause "pins row counts to the file's non-blank lines" is test detail that belongs in the test's
docstring. **Fix:** make "Current state" one sentence naming the modules present
(`render.py`, `hashing.py`, guard only), cite `HISTORY.md` for the rest, and drop the row-count
clause.

**Judgement calls:**
1. *Normalization in `render.py`, pinned only by hand tests:* the pin is adequate for U5.2, but a
   vector should pin it later. See P2-F3; ownership is the architect's call, with the vector
   produced in model-bench.
2. *`INPUT_FIELDS_BY_ROLE` is `guard-judge` only:* **accept.** It equals model-bench's mapping today
   (`dict(...) ==` checked), and model-bench itself has only that row until U4. An unknown role
   raises `ValueError`, which is tested. `input_hash` matches model-bench's on 20,000 random nested
   items, with 0 mismatches. Mutants H1–H7 all go red.
3. *A direct newest-k search instead of the separator arithmetic:* **accept.** No divergence was
   found:
   - It matches falkor-chat's real function and model-bench's port on 12,000 random cases, about
     3,400 of which render within 100 characters of the cap.
   - It also matches on 720 boundary cases, and at the exact edge: a single turn whose message is
     exactly 6,000 characters is kept, and one character more falls back to the base.
   - The differential has power: the positive controls `MAX_CHARS` 5999 and 6001 each produce 453
     mismatches in 1,500 cases.

   Its cost is quadratic in turn count, which falkor-chat removed in its K-027 n-2 change. That is
   irrelevant behind the 6-turn normalization.
4. *README Status rewritten:* **accept.** It was replaced in place and is in the present tense. It
   tracks work status, so `teco` lists it at milestone close (root `AGENTS.md`, compaction bullet).

**FR-8:** this holds, and it is verified. `render.py` imports only `json`, `collections.abc` and
`typing`; `hashing.py` imports only `hashlib`, `json`, `collections.abc`, `types` and `typing`.
`tests/test_fr8_isolation.py` gives 24 passed with both new modules in its scan. "falkor-chat"
appears only in module docstrings, which are exempt. The model-bench path (`_MODEL_BENCH`) lives
only in `tests/test_decision_contract.py`.

**Growth-stops:** root `AGENTS.md`'s growth-stops bullet covers requirements, plans and reviews, not
these files. I checked them against their own rules instead:
- `AGENTS.md` follows the context-file rule (rewrite, do not append); see P2-F4.
- `HISTORY.md` is read by lookup, so appending one entry is correct; see P2-F2 for accuracy.

### What's solid

- The guarded equality test reads both files as bytes. The row-count test counts from raw bytes
  rather than through the loader. A one-byte fixture change goes red (F1).
- Most of my re-sample went red (Appendix A): 24 of 26 mutants. That includes every
  canonicalisation step of the hash and every block, separator and cap edge of the renderer.
  Of the 2 survivors, R18 is P2-F1's gap and R12 is equivalent on this domain.
- R12 deserves a note: dropping `default=str` survives because JSON-loaded items carry only
  JSON-native values. It would matter only for an in-memory `understanding` holding, say, a
  `datetime`, and no current caller passes one.
- A non-Mapping turn row raises `AttributeError` here, exactly as in model-bench, whereas
  falkor-chat skips it. That is consistent with the evaluation side and loud. U5.4's schema check
  is the right place to refuse such an item.

### Appendix A — re-sampled mutants (Pass 2)

Each mutant was a single replacement, followed by the full suite. Mutants whose target text also
appears in a docstring were re-applied to the code occurrence; the first attempt hit the docstring
and was discarded.

| ID | Mutation | Result |
|---|---|---|
| R1 | `MAX_CHARS` 6000→5999 | 4 failed |
| R2 | `<= MAX_CHARS` → `<` | 2 failed |
| R3 | state block always emitted | 11 failed |
| R4 | `sort_keys=False` (state) | 3 failed |
| R5 | `indent=1` (state) | 5 failed |
| R6 | keep the oldest turns instead of the newest | 2 failed |
| R7 | no hard truncation of `base` | 2 failed |
| R8 | turns joined by `\n\n` | 4 failed |
| R9 | default speaker `Member` | 2 failed |
| R10 | slice before filtering | 1 failed |
| R11 | no `authorId` fallback | 1 failed |
| R12 | no `default=str` | **53 passed** (equivalent on JSON-loaded items) |
| R13 | empty text kept | 1 failed |
| R14 | default role `assistant` | 1 failed |
| R15 | `render_guard_item` skips normalization | 1 failed |
| R16 | header gains a trailing space | 9 failed |
| R17 | state with `ensure_ascii=False` | 1 failed |
| R18 | `render_guard_item` reverses turn order | **53 passed** (P2-F1) |
| H1 | `ensure_ascii=True` | 1 failed |
| H2 | spaced separators | 4 failed |
| H3 | `path` dropped from the fields | 4 failed |
| H4 | absent field written as `null` | 1 failed |
| H5 | `sort_keys=False` | 4 failed |
| H6 | UTF-16 encoding | 4 failed |
| H7 | `itemId` added to the fields | 4 failed |
| F1 | one byte in the `render-guard.jsonl` copy | 2 failed |

### Appendix B — differential

The differential script ran with `python -I`, against `decisionmodels.render`, model-bench's
`classification`, and falkor-chat's `_render_judge_user`/`_recent_turns`. The falkor-chat functions
were extracted from source by AST, so no falkor-chat import ran. Scratch script:
`/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/p2/diff/differential.py`.

**Generated inputs:**
- turn counts of 0–40;
- texts of 0–3,000 characters, with lengths of 399, 400 and 401 over-sampled, using non-ASCII,
  newline, quote and emoji characters;
- invalid `text` values: `None`, the empty string and an int;
- empty, `None` and `0` speaker fallbacks;
- conditions and `understanding` sized from 5,000 to 6,200 characters.

**Runs:** seeds 1, 2 and 3, 4,000 cases each, plus 240 boundary cases per seed. Each run reported
`{"norm": 0, "render": 0, "item": 0}` mismatches.

**Positive controls**, patching `decisionmodels.render` in process with seed 1 and 1,500 cases:

| Patch | norm | render | item |
|---|---|---|---|
| `MAX_CHARS=6001` | 0 | 453 | 333 |
| `MAX_CHARS=5999` | 0 | 453 | 333 |
| `RECENT_TURNS_KEPT=7` | 297 | 0 | 174 |
| `TURN_TEXT_MAX_CHARS=401` | 854 | 0 | 551 |

**`input_hash` against model-bench's:** 20,000 random nested items, 0 mismatches.

## Pass 3 — U5.3 points, rubrics, seeds (2026-11-01)

### Scope & verdict

**Reviewed:** the uncommitted U5.3 work against `b8555aca`. It is new `decisionmodels/points.py` and
`seeds.py`, `points/**` (three `point.json`, `rubric.md` and `generator.md` files, plus
`points/common/`), and `tests/test_points.py` and `test_seeds.py`. The edits to `AGENTS.md`,
`README.md` and `docs/HISTORY.md` are reviewed too. The baseline is plan4 §4.1's U5.3 row, plus
§3.2, §3.4, §3.6 and §3.7, and note §Q1.1–§Q1.3. Rubric wording, seed design and leakage method
belong to `data-scientist`'s parallel review in `docs/reviews/typed-decision-models-ml.md`; this
pass covers code, contracts, tests and docs only.

**Verdict: approve with suggestions.** There is no blocker. I checked these by running them:
- The live tree gives 161 passed, and `ruff check .` is clean.
- `AGENTS.md` is 484 words.
- The coordinator's `_restricted` mutant reddens 4 tests (C0).

All mutation ran in a private copy under the scratchpad (`u53-gate/copy/`), with `model-bench/`
symlinked in. I confirmed that the copy's `decisionmodels` is the one imported. Every mutant was
restored and checked equal with `cmp`, and a final sha256 over the copy showed it unchanged. The live
tree was never written: a sha256 over its 29 `decision-models/` files is unchanged at the end. No git
command changed the tree. One major finding goes to the plan, not to U5.3's code: P3-F1.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; both modules (425 lines) and both test files were read in full.

### Findings

**P3-F1 — major — a later draw can silently reuse a `seedId` for a different seed, and no unit is
assigned the guard.** *Owner: `architect`, for the plan4 §5 U5.6 and U5.9 rows, via `teco`. The
build goes to `coder`.* `_seed_id` hashes (role, RNG seed, restriction, position), not `count`
(`seeds.py:373-376`). I ran it: a top-up `draw_seeds(role, 37, 42, design_stratum=None)` after a
round-1 draw with RNG seed 42 shares **all 37 ids** with round 1, and all 37 have different axes. This
held for every point. Message type and join-in have no design stratum, so their top-ups are always
unrestricted. Guard draws restricted to a stratum shared 0 ids. The docstring makes distinct RNG
seeds "the caller's (U5.8, U5.9)", but plan4 §5's U5.6 and U5.9 rows test no such thing. The cost of
a reused id is silent: splits are assigned "by `seedId`" with one item per seed (plan4 §3.6, plan2
§3.5). **Fix:** add a U5.6 row in which the ledger refuses a draw whose RNG seed it already holds for
the role, and refuses any `seedId` already in the ledger. Name the planted case: a second draw with a
reused RNG seed is refused. Do not fold `count` into the hash instead, because an identical repeat
draw would still collide.

**P3-F2 — minor — a restricted draw cannot be reproduced from its own record, and no test notices.**
*Owner: `coder`.* Mutant X1 makes `SeedDraw.to_record` write `"designStratum": None`, and the suite
stays at 161 passed. The record round-trip test (`test_seeds.py:68-74`) covers only unrestricted
draws. The restricted test (`:77-79`) re-calls `draw_seeds` with a literal stratum and never goes
through the record. A top-up draw is exactly the restricted case. **Fix:** route the restricted
reproduction through `json.loads(json.dumps(first.to_record()))`, as the unrestricted test does. Then
confirm X1 goes red.

**P3-F3 — minor — the within-draw collision guard can never fail in the suite.** *Owner: `coder`.*
Mutant X3 deletes the `RuntimeError` check at `seeds.py:421-422`, and the suite gives 161 passed.
Root `REVIEW.md` counts a check that cannot fail as a finding. **Fix:** monkeypatch
`seeds._seed_id` to a constant and assert `RuntimeError` on a 2-seed draw.

**P3-F4 — minor — the guarantee that "seeds reproduce from the recorded RNG seed" holds only within
one Python version.** *Owner: `coder`.* `seeds.py` uses `Random.shuffle`, and it also uses
`Random.random`. Python guarantees across versions only the seeder and `random()`'s sequence
(Python docs, `random` "Notes on Reproducibility", checked). `shuffle` uses `_randbelow`, which is not
covered. `SeedDraw.to_record` records no interpreter version. **Fix:** record
`pythonVersion` (`major.minor`) in the draw record. Qualify the docstring and the `AGENTS.md` line to
say reproduction holds on the same Python minor version. The recorded seeds remain the source of
truth. `PINNED_DIGESTS` would redden on an interpreter that shuffles differently. That is the right
signal, and the `AGENTS.md` line should say so.

**P3-F5 — minor — the side-check shingle test turns into a skip when the pack moves.** *Owner:
`coder`.* It skips on `_SIDE_CHECK.is_file()` (`test_points.py:374`). The skip itself is real and
visible: with `model-bench/` removed, the `-ra` summary shows `SKIPPED [1] tests/test_points.py:375`
next to the two contract skips. But a renamed `guard-judge-understanding` pack, with `model-bench/`
still present, would also skip, and the note §Q1.4 gate would quietly stop running. The contract
test applies v1.3 §3.2.2's rule instead: skip only when `model-bench/` is absent. **Fix:** skip on
`_MODEL_BENCH.is_dir()`, and fail when model-bench is present but the items file is missing.

**P3-F6 — minor — the switch-over from the provisional field list is only half mechanised.**
*Owner: `coder` (U5.2b).* `test_the_input_fields_match_the_leakage_key` uses
`INPUT_FIELDS_BY_ROLE.get(role, PLANNED…)`. When U5.2b adds rows, they are compared automatically.
Mutant X13 adds a matching join-in row and stays green; X14 adds it in a different order and goes
red. But if U5.2b adds no row, nothing reddens, and the provisional list stays in force.
`test_guard_is_the_one_role_hashing_defines_so_far` does not assert that the others are absent, so
it cannot fail when they land. **Fix:** replace it with a
`pytest.mark.xfail(strict=True)` check, per non-guard role, that the role is in
`INPUT_FIELDS_BY_ROLE`. When U5.2b lands the rows, the strict XPASS reddens and forces the
provisional fallback to be deleted. Compare as sets, so that tuple order between `point.json` and
model-bench's row is not churn.

**P3-F7 — nit — the plan does not say how §3.7's `promptSha256` maps to the composed line-2
text.** *Owner: `architect`, via `teco`. `coder` applies it in U5.8 and U5.10.* §3.2's request line 2
is `{"rubric"}` or `{"instructions"}`. `Point.rubric` and `Point.generator_instructions` each
compose the body with the `points/common/` answer section. That matches §3.2 and §3.4's spawn
template ("Follow the instructions on line 2, write the answer where the header says"). The answer
sections match §3.2's answer format: line 1 `bodySha256`, `{"key","answer"}` or `{"key","item"}`
lines, and `.k.jsonl` parts. But §3.7 hashes `rubric` and `requestTemplate.{generator,labeler}`
separately, and U5.9's tranche relabels "under the new rubric hash". Whether `rubric` covers
`rubric_body` or the composed text is unstated. **Fix:** state it in plan4 §3.7. One option is
`rubric` = `rubric_body`, `requestTemplate.labeler` = `label-answer.md`, and
`requestTemplate.generator` = the composed generator text. `Point` already exposes both parts
separately.

**Questions asked, settled by execution:**
1. *Is the `points/` boundary sound?* **Yes.** `points/` holds no code, so the AST allowlist has
   nothing to check. The text scan covers every file under `points/`, `point.json` included, and
   `test_the_points_data_is_all_scanned` pins the file set. The positive controls go red:
   - a side-check sentence planted in `rubric.md` (Q1a, Q1b);
   - "falkor-chat" in a `point.json` axis text (Q1c);
   - `redis-cli` in `points/common/` (Q1d).

   The skip is real and visible, not a silent pass; its scope is P3-F5.
2. *Does line 2 match plan4 §3.2 and §3.4?* **Yes**, for the composed text; the open hashing
   question is P3-F7.
3. *Are `seedId`s unique and collision-safe?* Within a draw they are unique and guarded, but the
   guard is untested (P3-F3). Across draws with distinct RNG seeds, ids differed in every run
   (`test_different_rng_seeds_give_different_draws`). With a reused RNG seed, all ids collide
   (P3-F1).

   `ID_PATTERN` fullmatches every minted and defined id (`cond_00`, `dom_07`, `cue_01`,
   `pur_01`, `sd_` plus 12 digits), which is the form note §Q1.2 "Hits" names for U5.8. Two
   cautions for U5.8's screen, which are not U5.3 defects. First, the pattern is lowercase-only and
   unanchored, so `DOM_07` or `dom-07` would pass. Second, `Seed.axes` sends `conditionId`,
   `domainId`, `cueId` and `purposeId` to the generator, which is where a leaked id would come from.
   Whether the generator needs the ids at all is a method question, raised for `data-scientist`.
4. *Is the provisional field list's switch-over mechanised?* Only half; see P3-F6.
5. *Are the digest pins documented?* **Yes.** `test_seeds.py:101-103` labels them regression pins,
   and `AGENTS.md` says that a design or algorithm change re-pins them deliberately. The interpreter
   caveat is P3-F4.
6. *What did my own data-plumbing mutants show?* Of 15 (Appendix C), 11 went red. Of the four
   survivors, X1 is P3-F2 and X3 is P3-F3. X9 lets a `bool` through an enum: it is equivalent today,
   because no enum holds ints. X5 drops the blank line between the rubric and the answer section:
   it is cosmetic, so it is not reported.

**Docs.**
- `HISTORY.md`: the entry's counts reproduce (161 passed; 54 → 161).
- `AGENTS.md` grew from 319 to 484 words. The growth is new scope: two invariants about `points/`
  and seeds, which are present-tense rules a reader needs. "Current state" was rewritten, not
  appended to, and no line is over 700 characters. One clause belongs in the test docstrings rather
  than the context file: "Expected answer spaces, fields and counts in the tests are written from
  the plans… never read back." This is a nit, with no finding filed.
- The README Status was replaced in place, consistent with Pass 2's judgement call 4.

### Prior findings (dispositions)

Rechecked at `b8555aca` and in the tree:
- P1-F1: fixed. String-literal planted cases exist for `falkordb-dev`, `falkor_chat`, `falkorchat`
  (`test_fr8_isolation.py:195-206`).
- P1-F2: fixed. The README and `AGENTS.md` say "static tripwire", and the bytes scan and alias
  tracking are in place (`:87-88`, `:133-134`).
- P1-F3: fixed. The `AGENTS.md` FR-8 bullet names the side-check items.
- P1-F4: not fixed, carried. The coordination ledger carries it to the U5.10 brief.
- P1-F5: fixed. `redis` is letter-bounded, and there is a "predisposition" control (`:41-42`,
  `:248`).
- P1-F6: fixed. The root bullet cites `docs/requirements/typed-decision-models.md`.
- P2-F1: fixed. `test_render_guard_item_keeps_turn_order_through_normalization_and_the_cap`
  (`test_decision_contract.py:112`).
- P2-F2: fixed. The `HISTORY.md` U5.2 entry records the recovered restore and cites Pass 2.
- P2-F3: not fixed, carried. The ledger carries it to the U4 brief (`normalize-turns.jsonl`).
- P2-F4: fixed. "Current state" is one sentence at `b8555aca`.

**Growth-stops check on this document:** Pass 3 adds words only for new scope, the new unit U5.3.
Prior findings take one line each above, and Passes 1–2 are unchanged: their pre-Pass-3 text is
byte-identical to `HEAD` (`cmp`).

### What's solid

- Every expected count in `test_seeds.py` is derived by hand in the test, with the arithmetic shown
  (for example, 880 → 147 clear / 73 near-miss).
- `test_draw_seeds_has_no_intent_parameter` pins plan4 §3.6's rule that the sampler names no
  label.
- The schema checker refuses unexpected nested fields (X7 goes red), and `maxItems` is enforced
  (X10 goes red).
- Seed bindings catch an item that strays from its seed: a wrong turn count goes red (X6).
- Reproduction is checked under three `PYTHONHASHSEED` values in subprocesses.
- Dropping the id field breaks 6 tests (X15).

### Appendix C — Pass 3 mutants

These ran in the private copy, with the full suite after each, and were restored and checked equal
with `cmp`.

| ID | Mutation | Result |
|---|---|---|
| C0 | the coordinator's: `_restricted(...)` bypassed | 4 failed |
| X1 | `SeedDraw.to_record` writes `designStratum: None` | **161 passed** (P3-F2) |
| X2 | `_seed_id` ignores the restriction | 3 failed (via the digest pins) |
| X3 | within-draw collision check removed | **161 passed** (P3-F3) |
| X4 | `_seed_id` ignores the role | 3 failed |
| X5 | no separator between the rubric and the answer section | **161 passed** (cosmetic) |
| X6 | `lengthAxis` binding ignored | 1 failed |
| X7 | nested unexpected fields allowed | 1 failed |
| X8 | `allowEmpty` admits any object | 3 failed |
| X9 | `bool` allowed in an enum | **161 passed** (equivalent today) |
| X10 | `maxItems` ignored | 1 failed |
| X11 | conditional pool assigned in reverse | 1 failed |
| X12 | no final shuffle | 3 failed |
| X13 | hashing gains a matching join-in row | 161 passed (expected; P3-F6) |
| X14 | hashing gains a reordered join-in row | 1 failed |
| X15 | `<axis>Id` not set | 6 failed |
| Q1a | side-check request sentence planted in the guard rubric | 1 failed |
| Q1b | side-check `known` entry planted in the guard rubric | 1 failed |
| Q1c | "falkor-chat" in a join-in `point.json` axis text | 2 failed |
| Q1d | `redis-cli` in `points/common/generate-answer.md` | 1 failed |
| — | `model-bench/` symlink removed | 158 passed, 3 skipped (all three listed by `-ra`) |

## Pass 4 — U5.3 fix round (2026-11-01)

**Verdict: approve with suggestions.** The suite reproduces the reported state: 330 passed, 2
xfailed, 1 failed, and `ruff check .` is clean. The one failure is the known `input-hash.jsonl`
equality test. The live copy is a strict byte prefix of model-bench's: 4 of 10 lines, and the 6
added lines are join-in and message-type rows.

Every mutant ran in a private copy (`scratchpad/u53-gate2/copy/`, with `model-bench/` symlinked
in), with that one test deselected. Each was restored and checked equal with `cmp`, and a final
sha256 over the copy matched. A sha256 over the 29 live `decision-models/` files was unchanged at
the end. No git command changed the tree. Method content is `data-scientist`'s Pass 2.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; `seeds.py`, `points.py` and the changed tests were read in full.

### Pass 3 findings

- **P3-F1:** fixed in the plan. Plan5 §3.2 rules that the ledger refuses a held RNG seed or
  `seedId`, and the U5.6 and U5.9 rows carry planted cases. `seeds.py` stays pure, as proposed.
- **P3-F2:** fixed. `test_a_restricted_draw_reproduces_from_its_own_record` round-trips through
  JSON for both strata. It also asserts the conditional axes' `None` pattern: `missingForm` on
  `understanding`, `turnCount` on `turns`. Mutant X1 (record drops `designStratum`) now reddens 2
  tests (it was 0 in Pass 3).
- **P3-F3:** fixed. A monkeypatched constant `_seed_id` raises `RuntimeError`, and deleting the
  check reddens 1 test (it was 0). The test still means something under the new allocation: a
  2-seed draw still reaches the guard.
- **P3-F4:** fixed. The record carries `pythonVersion`, and `_digest` excludes it. Writing `"3.99"`
  reddens only `test_a_draw_records_its_python_version`; all three pins stay green, so the exclusion
  works.
- **P3-F5:** fixed. With `model-bench/` present but the pack missing, the test fails
  (`test_points.py:512: AssertionError`). With `model-bench/` absent, it skips and the `-ra`
  summary lists the skip. `test_a_moved_side_check_pack_fails_rather_than_skips` catches the skip
  exception correctly.
- **P3-F6:** fixed. A mismatched join-in row reddens 2 tests (an `AssertionError` that the
  `raises=KeyError` xfail does not absorb, plus the tuple test). A matching row in a different
  order also reddens 2: the strict XPASS and the order-sensitive tuple test, which plan5 §3.4
  deletes at U5.2b. Neither stays xfailed.
- **P3-F7:** fixed in the plan. Plan5 §3.3 hashes `rubric` from `rubric_body` and
  `requestTemplate.labeler` from `label_answer_section`.

### New findings

**P4-F1 — minor — a wrong-kind item reports a `missingForm` error, so plan5 §3.5 would count it as
`formMismatch` too.** *Owner: `coder` for the code; `architect`, via `teco`, for the precedence
rule (plan5 v1.3).*

What I ran: a schema-valid `turns` item (`understanding: {}`, `path: "turns"`) against real
`understanding` seeds. It yields **four** errors, not two. Three are on `evidenceKind`: `path`,
`understanding` empty, and `turns` present. The fourth is on `missingForm`:
"understanding.missing: expected a list".

Why it matters:
- The fourth error is always redundant. `missing` is a required field, so a missing list passes
  the schema only when `understanding` is `{}`, which is exactly the case the `presentWhen`
  binding already reports.
- Under plan5 §3.5, one wrong-kind item would carry both `bindingMismatch` and `formMismatch`. It
  would inflate `dropped.formMismatch` and that form's 5% halt with items that never attempted the
  form.
- The path is unpinned: mutant P4 (return `[]` there) gives 330 passed.

**Fix:**
- `_emptiness_errors` returns `[]` when the field is not a non-empty mapping, because the
  `presentWhen` binding owns that case.
- Add the four-error case to `test_points.py`, asserting that only `evidenceKind` errors remain.
- Plan5 §3.5 states the per-item precedence: any non-form binding error makes the item a
  `bindingMismatch`, never also a `formMismatch`.

**P4-F2 — minor — the expected-mean tolerance is about 10 standard errors, so a 5% start bias
passes.** *Owner: `coder`.* `_tolerance(size) = 0.1 + 0.02·size`. Over 1,500 fixed RNG seeds, the
standard error for a ⅓ share at size 1 is about 0.012, so the size-1 tolerance of 0.12 is about 10
of them. With the digest pins excluded:
- largest remainder reddens 59 tests;
- a fixed half start reddens 58;
- a start scaled by 0.9 reddens 3;
- a start scaled by 0.95 or 0.97 reddens **0**. At 0.95 the bias is about 0.035 per seed on a ⅓
  share, roughly 3 standard errors.

Plan5's U5.3 row is met: largest remainder reddens. But the tests cannot see a small bias. The
seeds are fixed, so a tight bound is not flaky.

**Fix:** add an exact test of `allocate` alone. Stub `rng.random()` with the grid
`(k + 0.5)/N` for `k < N`, where `N = 12·W`, and assert with `Fraction` that the mean share
equals the quota. I prototyped it over 9 weight vectors × totals 0–29:
- the real `allocate` gives 0 failures;
- the 0.97 mutant gives 165 failures.

After that, the draw-level mean tests check only composition, and their tolerance can drop to
about 4 standard errors. One limit: a bias of about 0.1% (×0.999) slips between the grid points.

**P4-F3 — minor — a third check that can never fail: the conditional-case overlap guard.** *Owner:
`coder`.* Mutant P2 deletes the `ValueError` at `seeds.py:196-197`, and the suite gives 330 passed.
This is the P3-F3 class again: a guard with no test that can trip it. **Fix:** plant two
overlapping cases in a scratch design and assert the `ValueError`.

**P4-F4 — minor — the `HISTORY.md` entry says message type's pin is unchanged, but all three
pins moved.** *Owner: `coder`.* Compared with the Pass 3 tree (`u53-gate/copy`):
- message type moved from `12995ec1…` to `299beb51…`, although its `design` is byte-equal;
- guard moved from `50e5ff2e…` to `341d8815…`;
- join-in moved from `679bce0f…` to `26109599…`.

Systematic rounding changes every allocation, so the move is expected. The other figures do
reproduce: 330 passed / 2 xfailed / 1 failed, and `AGENTS.md` at 528 words. **Fix:** "all three
re-pinned; message type's design is unchanged, but the allocation change moved its pin".

**P4-F5 — minor — the `input-hash.jsonl` red should be fixed by re-scheduling, not by tolerating
appends.** *Owner: `teco`, for sequencing; `architect` only if plan4 §4.3's order needs amending.*
The copy must stay strictly equal. Prefix tolerance would also let an appended *guard* row, which
is a new pinned behaviour, lag silently.

Re-copying alone does not fix it either. The six new rows are join-in and message-type rows, and
`hashing.INPUT_FIELDS_BY_ROLE` has no rows for those roles. `test_input_hash_matches_input_hash_jsonl`
would therefore raise `ValueError` on them.

The real cost is timing. U5.2b sits after U5.10, so the decision-models suite would carry a
deselected red through U5.4–U5.10, and every gate would run blind on that test.

**Fix:** split U5.2b. A hashing half (rows, re-copy, retiring the P3-F6 xfails) runs right after
U4.1. A renderer half follows the U4 unit that lands the renderers. Until then, gate commands
deselect by explicit node id, and the ledger records it. The `AGENTS.md` instruction "A red
equality test means re-copy … then fix the code" should say the re-copy happens in the unit that
adds the matching rows.

**P4-F6 — nit — `allocate` mixes a float start with integer edges.** *Owner: `coder`.* An integer
start, `j = int(rng.random() * W)`, with `ceil` done in integers gives identical shares. I checked
this on 20,000 random (weights, total, seed) cases with 0 differences. It also removes a
float-rounding edge that has a negligible but non-zero chance of shifting a unit at an interval
boundary. Consider it; the RNG stream and the pins are unchanged.

**Checked and sound:**
- **Conditional axes.** A mutant matching `when` on its first key only reddens 48 tests, so
  plan5's "never a value outside their row" is pinned.
- **Your mutant and the delegate's are consistent with mine.** Your `turnCount` weight mutant
  turned 9 red, and the delegate's largest-remainder mutant turned 32 red against my 59 (my
  implementation differs).
- **My own data-plumbing mutants:**
  - P3 (version scope) is pinned;
  - P4 (emptiness non-list path) survived, see P4-F1;
  - P2 (overlap guard) survived, see P4-F3.
- **`AGENTS.md` (484 → 528 words).** Growth-stops formally covers requirements, plans and
  reviews; judged by the context-file rule instead, this passes:
  - the seeds bullet was replaced in place;
  - the never-swap-in-largest-remainder rule is a durable invariant;
  - the new `hashing.py` xfail bullet is a present-tense trap, which U5.2b deletes along with the
    xfails.
- **`HISTORY.md`.** Rewriting the not-yet-gated U5.3 entry in place is correct. Its only wrong
  figure is P4-F4.

**Growth-stops check on this document:** Pass 4 records one line per Pass 3 finding. Its new text
covers six new findings only. Passes 1–3 are byte-identical to their pre-Pass-4 text (`cmp`).

### Appendix D — Pass 4 mutants

These ran in the private copy, with the full suite after each and the `input-hash.jsonl` equality
test deselected. Baseline: 330 passed, 2 xfailed.

| ID | Mutation | Result |
|---|---|---|
| A1 | largest remainder in `allocate` | 59 failed |
| A2 | start ×0.9 | 4 failed (3 with the pins deselected) |
| A3 | start ×0.8 | 24 failed |
| A4 | start fixed at W/2 | 58 failed |
| A5 | start `u^1.25` | 8 failed (7 with the pins deselected) |
| — | start ×0.95 / ×0.97, pins deselected | **0 failed** (P4-F2) |
| X1 | draw record drops `designStratum` | 2 failed |
| X3 | collision check removed | 1 failed |
| P1 | conditional `when` matches its first key only | 48 failed |
| P2 | case-overlap check removed | **330 passed** (P4-F3) |
| P3 | `pythonVersion` written as `"3.99"` | 1 failed (version test only; pins green) |
| P4 | emptiness binding ignores a non-list `missing` | **330 passed** (P4-F1) |
| P5 | `hashing` gains a mismatched join-in row | 2 failed, 1 xfailed |
| P6 | `hashing` gains a matching, reordered join-in row | 2 failed, 1 xfailed |

## Pass 5 — U5.4 exchange and ingester (2026-11-01)

### Scope & verdict

**Reviewed:** the uncommitted U5.4 work in `decision-models/` against `d7c7465a`: new
`decisionmodels/exchange.py`, `ingest.py`, `tests/test_exchange.py`, `tests/test_ingest.py`, and
the `AGENTS.md`, `README.md` and `docs/HISTORY.md` edits. Baseline: plan4 §3.2, §3.3's answer
rows, idempotence and second-answer refusal, the §4.1 and §5 U5.4 rows; plan5 §3.4 and its U5.4
row; note §Q1.2 "Blind".

**Verdict: needs changes.** One major code defect (P5-F1) needs a fix and a test first. Everything
else is minor. What I ran:
- The live tree, with the `input-hash.jsonl` node deselected: 458 passed, 1 deselected, 4 xfailed.
  `ruff check .` is clean, `tests/test_fr8_isolation.py` gives 24 passed, and `AGENTS.md` is 598
  words (527 at HEAD).
- Mutants ran in a private copy (`scratchpad/p5u54/repo/`, with `model-bench/` symlinked in). I
  confirmed that the copy's `decisionmodels` is the one imported. Each mutant was restored and
  checked with `cmp` (Appendix E).
- The coordinator's mutant reproduces: `mint_key(minted)` stays green.
- A sha256 over the live component's files matched the baseline at the end. No git command changed
  the tree.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; both modules (734 lines) and both test files were read in full.

### Findings

**P5-F1 — major — an answer line with a non-string key crashes ingest instead of voiding.**
*Owner: `tdd-engineer`. Verified by running it.* `_check_entries` evaluates `key not in known` on
a raw JSON value (`ingest.py:160`).
- `{"key": ["a"], "answer": true}` or `{"key": {"a": 1}, …}` raises `TypeError: unhashable type`,
  in both label and generation answers.
- Plan4 §3.3 makes malformed answers run-attributable. A crash instead leaves the batch `issued`.
  `next` then reconciles by ingesting it again (plan4 §3.5) and crashes again, so the campaign
  stalls on one model-written line.
- Related: a line with a repeated name, `{"key": k, "answer": false, "answer": true}`, ingests
  silently as `true` (last wins). The labeler gave two answers, so the line is not the closed shape.

**Fix:**
- After `ingest.py:159`, a non-`str` key is `MALFORMED`. I ran this: my two parametrized cases
  pass and the full suite stays green.
- Parse answer lines with an `object_pairs_hook` that refuses a duplicate name. The hook raises a
  plain `ValueError`, not `JSONDecodeError`, so the `except` must widen to catch it (I checked
  this).
- Add both shapes to `test_a_malformed_line_is_void`.

**P5-F2 — minor — key uniqueness is untested; the test named for it cannot fail.**
*Owner: `tdd-engineer`. Verified by running it.*
- Your mutant M0 (`mint_key(minted)`) stays green, and so does mine, M1
  (`mint_key(set(held_keys))`, which drops the within-request check). That is a real gap against
  plan4 §3.2's "unique across the campaign and redrawn on collision".
- `test_keys_are_unique_within_a_request_and_avoid_held_keys` uses 5 random held keys in a 32¹⁰
  space, so it cannot fail.
- Production impact is low: a collision needs both a regression and a draw at about 10⁻¹⁵ per pair.
  But a test whose name claims the contract also hides it from the next reviewer.

**Fix:** two builder-level tests that patch `exchange.secrets.choice` to a scripted stream, as
`test_a_key_collision_is_redrawn` already does:
- the stream `aaaaaaaaaa`, then `bbbbbbbbbb`, with held `{"aaaaaaaaaa"}`, yields
  `list(built.keys) == ["bbbbbbbbbb"]`;
- the stream `aaaaaaaaaa` twice, then `bbbbbbbbbb`, for 2 items with nothing held, yields
  `["aaaaaaaaaa", "bbbbbbbbbb"]`.

The stream also needs 16 more characters for the opaque name. I ran both tests: green on the live
code, and M0 and M1 each redden one. The existing test can then be dropped or renamed.

**P5-F3 — minor — two answer-normalization behaviors are unpinned.** *Owner: `tdd-engineer`.
Verified by running it.*
- M3 hashes the parts raw, with no newline appended: 458 passed. The docstring's claim (an answer's
  hash does not depend on its part split, `ingest.py:22`) is tested only with parts that end in
  newlines.
- M8 drops the `\r` strip: 458 passed.

**Fix:** a test where the first part has its final newline stripped and `prior` is the whole
answer's hash, which must give `AlreadyIngested`. I ran it: green live, red under M3. Add a CRLF
answer that ingests, for M8.

**P5-F4 — minor — idempotence at the driver falls in no unit's row.** *Owner: `architect`, for
plan4 §5's U5.6 row, via `teco`. Verified for the code path; the U5.6 consequence is inferred.*
- U5.4 proves idempotence against a stand-in ledger, which is all the module can do.
- Plan4 §3.2 moves the answer parts to `store/` on ingest. A repeated `driver ingest` that reads
  `exchange/ans/` then finds no parts, and `ingest_answer(…, [], ingested_answer_sha256=<sha>)`
  raises `AnswerRefused`. So the plan's "no-op" turns into a refusal unless the driver reads from
  `store/`.
- A crash between recording a `Void` and moving the files means the void is re-ingested with
  `ingested_answer_sha256=None`. That could count a second attempt.

**Fix:** add a U5.6 row with three cases:
- `driver ingest` repeated on an ingested batch leaves the ledger byte-equal;
- repeated on a voided batch, it counts one attempt;
- a different second answer is refused.

The audit-precedence seam itself is sound. Ingest emits only run-attributable outcomes, and U5.5's
row already requires that "a fixture matching a halt row and a counted row halts".

**P5-F5 — minor — the 40,000-character budget assumes prose token density, and the costliest
line is not prose.** *Owner: `architect`, for plan4's U5.7 smoke row, via `teco`. Inferred: I did
not measure tokens.*
- Plan4 §2 measured 2.0 characters per token on raw pt-BR chat text.
- The densest line is a guard `understanding` item. The renderer ASCII-escapes it and the request
  escapes it again: 197 `\\u00XX` sequences, about 22% of the line's 6,331 characters.
  Escape-heavy text tokenizes worse than prose.
- A page over 25,000 tokens makes Read fail outright.

**Fix:** require the U5.7 smoke fixture to include this shape (`tests/test_exchange.py`'s
`pt_br_item_at_cap`), not only `turns` prose, or measure it once.

**P5-F6 — nit — the validator accepts degenerate pages and batches.** *Owner: `tdd-engineer`.
Verified by running it.*
- `[[1,1],[0,0],[2,2]]` and `[[1,1],[2,2],[7,0]]` both pass `request_errors`, because a
  zero-limit page covers nothing.
- `build_label_request(G, [], …)` writes a request with no items, which costs a spawn for nothing.

**Fix:** refuse a `limit < 1` and an empty batch.

**P5-F7 — nit — docs.** *Owner: `tdd-engineer`. Verified.*
- `HISTORY.md` says 25 mutants; the brief reports 26.
- `HISTORY.md` says keys are "redrawn against a caller-held set", which is true in code but
  untested (P5-F2).
- `AGENTS.md`'s new `bodySha256`/page sentence restates `exchange.py`'s docstring. Citing it would
  be enough.

### Checks the brief asked for

1. **Request format.** Every §3.2 element matches line by line:
   - the header's six keys, `[1, 1]` first, `answerPath` under `ans/<opaque>.jsonl`;
   - line 2 per kind, with `{key, text}` or `{key, <text axes>}` per entry;
   - closed sets on every line, and canonical `bodySha256`;
   - newline-inclusive page counting, checked at exactly 40,000 and 40,001;
   - 16-character opaque names.

   The delegate applies the `*Id` rule to every generation line, which is a superset of plan5's
   wording. It over-refuses nothing: no header key, line-2 key or design axis ends in `Id`. On the
   header and line 2 it changes only the error message, since those sets are closed anyway.
   Every point's writer output validates across 30 RNG seeds × each stratum (one axis shape per
   point).
2. **Page slack.** I reproduced every figure: rubric line 5,099, item 6,331, 50 items in exactly 10
   pages (page 2 holds the rubric and 5 items), and 70 items cut at 53. Three spare items is the
   wrong measure. Page 2 holds the rubric, so a worst-case 50-item batch still fits until the
   rubric line reaches 27,335 characters, 5.4× today. Past that, the cut is still correct:
   overflow goes to the next batch, and only the batch size changes. Neither plan nor code needs
   an item margin. The margin worth having is in tokens (P5-F5).
3. **Ingest.**
   - Type-strict answers hold (M7: 2 failed).
   - Parts are ordered numerically (M4: 1 failed).
   - Earlier-ingest is checked before validation, and invalid request text raises
     `RequestRefused`, never a void.
   - The reason order is an internal tiebreak inside plan4 §3.3's single run-attributable class,
     so it matches. Plan5 §3.5's "schema first" is honored.
   - The gaps are P5-F1 and P5-F4.
4. **Line separators.** With raw U+2028, U+2029 and U+0085 in a turn, the header hash equals a
   recomputation over the raw-character body and the independent helper's. `splitlines()` and
   `split("\n")` agree (3 lines). The body hash is unchanged.
5. **Docs.** The `HISTORY.md` paging and suite figures reproduce. `AGENTS.md` grows by 71 words for
   two new modules, which is new scope. It was rewritten in place: the state line was replaced,
   the U5.2b bullet merged and the FR-8 bullet shortened. No line is over 700 characters. It
   passes.
6. **My plumbing mutants.** M9 shifts the store's key→seed pairing by one: 2 failed. M1, the
   within-request mint, is P5-F2.

### What's solid

- The writer validates its own output twice, before it hands the request out and before it writes.
  `write_request` never overwrites and writes through a `.partial` rename.
- `bodySha256` is computed from values, not lines, so escaping and layout cannot move it.
- Tests take their expected values from the plans, with an independent hash recomputation and the
  exact-budget boundary on both sides. The second-answer refusal also covers a would-be-void
  answer.

### Open questions

- Is a `Void` an "ingested" outcome for the ledger's `ingested_answer_sha256`? The module allows
  either reading, and P5-F4's U5.6 row should state which.

**Growth-stops check on this document:** Pass 5 adds new findings only. Passes 1–4 are
byte-identical to their pre-Pass-5 text (`cmp` on the first 769 lines).

### Appendix E — Pass 5 mutants

These ran in the private copy with the full suite, the `input-hash.jsonl` node deselected.
Baseline: 458 passed, 4 xfailed.

| ID | Mutation | Result |
|---|---|---|
| M0 | `mint_key(minted)` (coordinator's) | **458 passed** (P5-F2) |
| M1 | `mint_key(set(held_keys))` | **458 passed** (P5-F2) |
| M3 | `answer_sha256` hashes parts raw | **458 passed** (P5-F3) |
| M4 | parts ordered by `str(k)` | 1 failed |
| M6 | line-separator escaping removed | 1 failed |
| M7 | answer space not type-strict | 2 failed |
| M8 | `\r` strip removed | **458 passed** (P5-F3) |
| M9 | store key→entry pairing shifted by one | 2 failed |

## Pass 6 — U5.4 fix round (2026-11-01)

**Verdict: approve with suggestions.** There is no blocker or major. What I ran:
- **Suite:** 476 passed, 1 deselected (`input-hash.jsonl`), 4 xfailed. `ruff check .` is clean, and
  `AGENTS.md` is 581 words.
- **Mutants:** they ran in a fresh private copy (`scratchpad/p5u54/repo/`, with `model-bench/`
  symlinked in). Each was restored and checked with `cmp`, and the copy's package matches the live
  tree with `diff -rq`.
- **Live tree:** I never wrote to it, and its sha256 matched at the end. No git command changed the
  tree.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; the fix-round diff of both modules was read in full.

### Pass 5 findings

- **P5-F1:** fixed.
  - List and dict keys are `malformed` voids. A repeated name is caught at any depth, including on
    line 1 (`answer line 1: repeated name 'bodySha256'`).
  - A 5,000-digit integer (a plain `ValueError`) and 100,000-deep nesting are `malformed` voids.
  - Mutant N1 (key check removed) gives 6 failed, N2 (hook removed) 3, N3 (`except` narrowed to
    `JSONDecodeError`) 3 and N4 (`RecursionError` dropped) 1.
- **P5-F2:** fixed. M0 and M1 each turn 1 test red (both were green).
- **P5-F3:** M3 is fixed (1 failed). The `\r` strip was deleted rather than tested; see P6-F2.
- **P5-F4:** ruled in the coordination ledger (2026-11-01, "Architect inline ruling"). Executed
  against the module:
  - an identical void re-delivered with its recorded sha gives `AlreadyIngested`;
  - a different answer after a void, or after `answerMissing`, gives `AnswerRefused`;
  - an `answerMissing` void's sha equals `sha256(b"")`;
  - the delegate's description ("voids a re-read with no prior hash, refuses one with") holds.

  The residue is P6-F1.
- **P5-F5:** ruled. The U5.7 brief requires 50 `pt_br_item_at_cap` items, with the largest page
  measured at 22,500 tokens or fewer.
- **P5-F6:** fixed. Removing the empty-batch check gives 1 failed on the writer side (N5) and 1 on
  the validator side (N6). Removing the `limit < 1` check (N7) gives 3 failed.
- **P5-F7:** fixed. `HISTORY.md` reports 36 mutants, the key redraw is now tested, and the
  restated `AGENTS.md` sentence is gone.

**The request-side hook** cannot refuse a request the driver wrote. The writer serializes Python
dicts, which cannot repeat a name at any depth, and item text is a string. 151 writer outputs
validate: 30 RNG seeds × 3 points unrestricted, 30 × 2 guard strata, and a 70-item at-cap label
batch.

**The fixes work together.** `RepeatedName` subclasses `ValueError`, so the ingest-side
`except (ValueError, RecursionError)` maps the hook's error, a decode error and a digit-limit error
all to `malformed`. That stays below `answerMissing` and above every key reason. A line that both
repeats a name and has an unknown key voids as `malformed`, which is correct.

### New findings

**P6-F1 — minor — a re-read under the ruling can refuse a benign re-ingest in two cases.**
*Owner: `architect`, for the U5.6 brief, via `teco`. Verified by running it.*
- **Layout void.** A layout void (both a whole file and parts, or a gap) records `sha256(b"")`,
  because `ingest_answer_files` hashes `()` (`ingest.py:241`).
  - A re-read through `ingest_answer_files` gives `AlreadyIngested`.
  - Reading the same files' bytes and passing them to `ingest_answer` gives `AnswerRefused`.
- **Mid-move crash.** A 2-part answer was ingested, then part 2 moved to `store/` before a crash.
  The ruling's "`exchange/ans/` if present, else `store/`" reads only part 1 and gives
  `AnswerRefused`. That fails closed, but it needs a human. The ruling's case (b) covers only a
  crash before the move.

**Fix:**
- In U5.6: always re-read through `ingest_answer_files`, against a directory that keeps the
  original part names.
- Make the move atomic as a set: copy every part to `store/`, fsync, then unlink. The re-read
  takes `store/` when it holds the complete set.
- Add case (e) to the U5.6 row: a crash mid-move, then re-ingest, leaves the ledger byte-equal.

No module change is required.

**P6-F2 — nit — the `\r` strip was not entirely dead code.** *Owner: `tdd-engineer`. Verified by
running it.*
- These ingest exactly as before:
  - CRLF answers;
  - a `\r` before a closing brace;
  - a final `\r` with no LF;
  - a double `\r`.
- A raw `\r` inside a string is still `malformed`.
- But a `\r`-only line in position 1 now voids as `bodyHashMissing`, where an LF blank line voids
  as `malformed` (the strip made the two agree). A `\r`-only line in the middle stays `malformed`,
  with a different detail. Only the void reason changes, and the class is the same.

**Fix:** either restore the strip, or accept the change and replace "dead code" in `HISTORY.md`
with "changes only the void reason of a CR-only first line".

**P6-F3 — nit — two request-side checks cannot fail.** *Owner: `tdd-engineer`. Verified by
running it.*
- Mutant N8 (the `offset < 1` check removed) passes the full suite. The coverage check already
  refuses any offset below 1, so the check is redundant: delete it.
- Mutant N10 (the request-side `RecursionError` catch disabled) also passes the full suite.
- `_parse_lines` does not catch the digit-limit `ValueError`, so a request line with a 5,000-digit
  integer raises a bare `ValueError` from `request_errors`. That predates this round, and the driver
  cannot write such a line.

**Fix:** catch `ValueError` beside `RecursionError` in `_parse_lines`, so it reports errors the way
the ingest side does, and pin both.

**Growth-stops check on this document:** Pass 6 gives one line per Pass 5 finding and three new
findings. Passes 1–5 are byte-identical to their pre-Pass-6 text (`cmp`).

### Appendix F — Pass 6 mutants

These ran in the private copy with the full suite. Baseline: 476 passed, 4 xfailed.

| ID | Mutation | Result |
|---|---|---|
| M0 / M1 / M3 | Pass 5's survivors | 1 failed each |
| N1 | non-string key check removed | 6 failed |
| N2 | repeated-name hook removed | 3 failed |
| N3 / N4 | ingest `except` narrowed to `JSONDecodeError` / `ValueError` | 3 / 1 failed |
| N5 / N6 | empty batch allowed (writer / validator) | 1 / 1 failed |
| N7 | `limit < 1` check removed | 3 failed |
| N8 | `offset < 1` check removed | **476 passed** (P6-F3) |
| N9 | line-1 repeated name read as missing | 1 failed |
| N10 | request-side `RecursionError` catch disabled | **476 passed** (P6-F3) |

## Pass 7 — U5.2b-hash (2026-11-01)

**Verdict: approve with suggestions.** There is no blocker or major. I re-ran the suite without
deselection: 487 passed, 2 xfailed. `ruff` is clean, and `tests/test_fr8_isolation.py` gives 24
passed. The vector is `cmp`-equal to model-bench's. Model-bench's `decisions.py` is unmodified
against HEAD, and its row equals the one at `1554d560`.

Mutants ran in a private copy (`scratchpad/p7u52bh/repo/`), with model-bench symlinked in and,
where marked, removed. The copy matches the live tree (`diff -rq`), the live sha256 is unchanged,
and model-bench was only read. CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; the diff (one table, two tests) was read in full.

**The AST reader (question 1) fails loudly in every case but one.** I ran it against planted
`decisions.py` variants:
- A renamed symbol, a plain `=` assignment, or a re-export from another module raises
  `AssertionError` ("defines no … literal").
- A value bound through a name raises `ValueError`.
- A file that is mid-edit and does not parse raises `SyntaxError`.
- A missing `decisions.py` raises `FileNotFoundError`.
- Only an absent `model-bench/` skips. Under `-rs` that skip is listed beside the 3 existing guarded
  skips: 483 passed, 4 skipped.
- **The silent case is P7-F2.**

**P7-F1 — minor — the exact-order comparison is a false-red coupling.** *Owner: `coder`. Verified
by running it.*
- `canonical_input_json` builds a dict and dumps it with `sort_keys=True`. Tuple order changes no
  hash, and nothing in decision-models reads the order.
- Mutant H2 reorders join-in's fields, which is hash-neutral. Under
  `test_input_fields_by_role_equals_model_benchs` it gives 1 failed.
- A pure reorder refactor in model-bench (U4.2 is editing it now) would therefore redden
  decision-models' suite. Plan5 §3.4 puts that suite in every U4.x gate, so the cost would be a
  pointless re-copy.
- This repeats the coupling `HISTORY.md` itself calls "a hash-neutral reorder falsely red" for the
  points test, and the point test was fixed accordingly.

**Fix:** compare `{role: sorted(fields)}` on both sides. Sorted lists, not sets, still catch a
duplicated field. I ran this:
- H2 (reorder) becomes green: 487 passed.
- H1 (drop `priorTurn`) gives 5 failed.
- H3 (an extra field) gives 2 failed.
- H4 (a duplicated field) gives 1 failed.

**P7-F2 — nit — the reader takes the first top-level binding.** *Owner: `coder`. Verified by
running it.* If `decisions.py` gains a later `INPUT_FIELDS_BY_ROLE = …` override, the test passes
against the stale first literal.

**Fix:** collect every top-level `Assign`, `AnnAssign` or import that binds the name, and require
exactly one.

**P7-F3 — nit — `README.md:44` is 167 characters.** *Owner: `coder`.* The edit joined two lines
without reflowing them.

**Obligations (question 3): nothing is dropped.** Plan5 §3.4's U5.2b-hash list is complete:
- both rows are present;
- `input-hash.jsonl` is re-copied with 10 rows (4 guard, 3 join-in, 3 message type);
- `inputFields` equals the rows as sets for every role;
- the xfails, `PLANNED_INPUT_FIELDS` and the `.get` fallback are gone. A sweep finds them only in
  `HISTORY.md`.

The deleted tuple test was P3-F6's order-sensitive test, which plan5 slated for deletion.

The new checks can fail without model-bench:
- H1 gives 4 failed with `model-bench/` absent, through the copied vectors and the points test.
- H3 gives 1 failed, through the points test.
- Only the hash-neutral H2 goes green without model-bench, which is correct.

**Docs (question 4): the figures reproduce.** They are 487 passed / 2 xfailed, FR-8 24, 6 new
vector rows and the `1554d560` row. `AGENTS.md` shrinks from 581 to 574 words, rewritten in place:
the U5.2b bullet is narrowed to the renderers. I did not re-run all of the delegate's ten mutants.
My H1 matches its "field dropped" class (5 failed).

**Growth-stops check:** Pass 7 is new findings only. Passes 1–6 are byte-identical (`cmp`).
