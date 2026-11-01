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

## Pass 8 — U5.5 transcript audit (2026-11-01)

**Verdict: needs changes.** There is no blocker. There are two majors, both on how real runs end on this
machine. Neither is visible on the hand-written fixtures. What I ran:
- **Suite:** 568 passed, 2 xfailed, with no deselection. `ruff` is clean, FR-8 gives 24 passed and
  `tests/test_audit.py` gives 76.
- **Mutants and probes:** these ran in `scratchpad/p8u55/repo/`, with `model-bench/` symlinked and
  only read. Each mutant was restored and checked with `cmp`. The probes are planted fixture edits
  driven through the suite's own helpers (Appendix G).
- **Live tree:** the sha256 of all 45 files was unchanged at the end. No git command changed the tree.
- **Real transcripts:** I read keys, types and counts with Python, across 238 subagent runs. I
  copied nothing out of them.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; `audit.py`, its tests and fixtures were read in full.

### Findings

**P8-F1 — major — under background spawning, a dead run and a still-running run are both counted.**
*Owner: `architect` (the ruling already asked for), then `tdd-engineer`. Verified by running it.*
- All 236 `Agent` results on this machine are `async_launched`. This holds with or without a
  `run_in_background` key. So `has_result` (`audit.py:336`) is always true, and the `incomplete`
  row never fires.
- On the fixture with `toolUseResult.status: "async_launched"`, a run cut after its first Read
  gives `COUNT maxTurns`. So does a run cut before its final reply, which is what a run still
  working when `driver ingest` runs looks like. The delegate's flag says these runs are
  `answerMissing`. The actual reason is `maxTurns`, but the class is the same: counted where the
  plan says no attempt, with items dropping as `retryCapExceeded`.
- **Minimal audit change, run in the copy:** a `tool_result` whose `toolUseResult.status` is
  `async_launched` is not a completion. For such a spawn, completion is `.meta.json`
  `finished: true`. A foreground result keeps today's rule. The suite stays at 568 passed. A
  background run that is finished and complete is `CLEAN`. A background run that is unfinished,
  whether killed or still running, is `REISSUE incomplete`. A background run that is finished but
  cut short is `COUNT maxTurns`.
- Add fixtures in both shapes. Today the fixtures carry no `toolUseResult`, and the meta carries no
  `requestShape`.
- *Inferred:* a killed run's meta lacks `finished`. The 2 in-flight runs lack it, and I have no
  sample of a killed run.
- **U5.6 brief:** run `ingest` only after the run hands back. A re-issued opaque name must be
  retired in the ledger, so a late answer to it is never ingested.

**P8-F2 — major — real subagent runs end with a `SubagentHandback` call, and the audit voids every
such run.** *Owner: `architect`, together with P8-F1. Verified by running it.*
- 206 of 207 finished runs here end with a `SubagentHandback` `tool_use`. Their last conversation
  entry is its `tool_result`, not an `end_turn` reply.
- The call is injected even when `tools:` omits it: all 71 `analyst` runs end with it.
- On the clean fixture, I replaced the final `done` reply with that call:
  - unhooked, it gives `HALT hookMissing`, with `maxTurns` and `toolOutOfBounds` also found;
  - with the guard allowing it, it gives `COUNT maxTurns`.
- If `dm-labeler` runs end this way, every batch voids. The U5.7 smoke would catch this, but only
  in a usage window.

**Fix:**
- The architect rules on whether dm runs end with a hand-back.
- If they do, exactly one final `SubagentHandback` counts as the reply. It is exempt from the
  bounds and hook checks, and it has its own fixture.

**P8-F3 — minor — two mutants survive, and the §5 halt-over-counted test cannot go red alone.**
*Owner: `tdd-engineer`. Verified by running it.*
- **A6** ties counted with re-issue (`COUNT: 1`) and passes all 568 tests. Every precedence test
  pairs rows whose table order already agrees with the tiers. A planted `hookDenied` plus
  `rate_limit` gives `WAIT`, and `COUNT` under A6.
- **P1** (`has_result` ignores `tool_use_id`) passes all 568. The fixture driver holds no other
  `tool_result`, but a real driver always does: `driver next` is a Bash call. A planted Bash
  `tool_result`, with the spawn's result dropped, gives `REISSUE`, and `CLEAN` under P1.
- `test_a_fixture_matching_a_halt_row_and_a_counted_row_halts` stays green under A1 and A2 (only
  its siblings catch them), because `agentTypeMismatch` precedes `toolOutOfBounds` in the table.

**Fix:**
- Pin the two planted cases.
- Re-pair the §5 clause test as `hookDenied` plus `authentication_failed`, where the counted row
  comes first in the table.

**P8-F4 — minor — any hook's JSON decision passes as the guard's.** *Owner: `tdd-engineer`; U5.7
supplies the command. Verified by running it.*
- All 5,198 real `hook_success` entries carry `command`, and every agent guard here prints a
  `permissionDecision`. So the stdout parse does not single out the guard.
- With every hook entry's `command` set to a non-guard script, the fixture stays `CLEAN`.
- Example: a user-level hook stays active while the project guard is off.

**Fix:** add `Expected.guard_command`, and require `command` to equal it.

**P8-F5 — minor — flag (c): a run killed between a `tool_use` and its hook entry halts instead of
re-issuing.** *Owner: `tdd-engineer`. Verified by running it.*
- Hook entries always follow their `tool_use` (5,207 of 5,207).
- The fixture cut there, with no `tool_result`, gives `HALT hookMissing` above `incomplete`.
- A truncated last line likewise gives `transcriptUnreadable`.

Both fail closed, but cost a `resolve`.

**Fix:**
- When the run is incomplete, exempt the trailing unanswered `tool_use` from `hookMissing`.
- Read an unparseable final line as truncation.

**P8-F6 — nit — `README.md:45` is 142 characters**, the same unreflowed join as P7-F3. *Owner:
`tdd-engineer`.*

**Flag (a)** needs no new finding: the provisional `maxTurns` shape is carried to U5.7 in the ledger.
Settle it together with P8-F2, which changes what the last entry of a clean run is.

### Questions 1–7

**1. §5 clauses.** Every U5.5 clause has a test, and each test goes red under some mutant. The
exception is the halt-over-counted clause on its own (P8-F3).
- The 3/900/2087 fixture is literal and written by hand: `cacheWrite` 500/800/650 makes the last
  entry neither the first nor the maximum. The first-entry mutant A4 gives 5 failed.

**2. Judgement calls.** Each one is stricter than §3.3 or equal to it.
- Table order breaks ties within a tier. Precedence matches §3.3, though A6 is unpinned (P8-F3).
- One cap shared across never-spawned and incomplete is the §5 U5.6 wording.
- `Write` to the request becomes counted, never a halt.
- `toolUseRepeated` and `transcriptUnreadable` are halts that never displace a counted void.
- Effort: all 238 real runs have `api-request` entries with `output_config.effort`.
- Locating the spawn only through `assistant` entries is sound: replays sit only in
  `api-request-blob` entries.
- **Where a class does change:** P8-F1 (no-attempt → counted), P8-F2 (clean → halt or counted) and
  P8-F5 (re-issue → halt).

**3. The `combine` seam.** It matches the ruling.
- A `COUNT` audit over a valid answer keeps that answer's hash, so a re-delivery is
  `AlreadyIngested` and the attempt is counted once.
- Raising on `AlreadyIngested` fits, provided the U5.6 re-ingest path checks the ledger first and
  never calls `audit_run` or `combine`. It finishes the move and appends nothing.
- **Add both points to the U5.6 brief.**

**4. Fixture fields.** I spot-checked `.meta.json` `toolUseId`/`agentType`, the `hook_success`
keys, the first `user` content (a string, 238 of 238) and `api-request` `params`. All match. The
gaps are completion shapes (P8-F1, P8-F2), not field names.

**5. Flags.** (a) is carried to U5.7. (b) is P8-F1, and the audit is unsound under background
spawning. (c) is P8-F5.

**6. Docs.** The figures reproduce: 568/2, 76 tests, FR-8 24, and 20 mutants plus 1 equivalent.
`AGENTS.md` grows by 57 words of new scope, and no line exceeds 700 characters. It passes the
context-file rule.

**7. My data-plumbing mutants.**
- P1 survives (P8-F3).
- P3 takes the `effort` fact from the last `api-request` and survives. That is equivalent on a
  constant-effort run.
- P4, a corrupted `agentId`, gives 1 failed.

**Growth-stops check:** Pass 8 is new findings only. Passes 1–7 are byte-identical (`cmp`).

### Appendix G — Pass 8 mutants

These ran with the full suite. Baseline: 568 passed, 2 xfailed.

| ID | Mutation | Result |
|---|---|---|
| A1 | `COUNT` tier 0 (tied with halt) | 2 failed (re-issue and usage stop over counted) |
| A2 | table order only | 1 failed (halt over re-issue) |
| A3 | `syntheticError` counted | 4 failed |
| A4 | first-entry tokens | 5 failed |
| A6 | `COUNT` tier 1 (tied with re-issue) | **568 passed** (P8-F3) |
| P1 | `has_result` ignores `tool_use_id` | **568 passed** (P8-F3) |
| P2 | instructions as a subset (teco's mutant) | 1 failed |
| P3 | `effort` fact from the last `api-request` | 568 passed (equivalent here) |
| P4 | `agentId` reversed in `run_record` | 1 failed |
| fix | P8-F1's background rule, applied | 568 passed |

Probes, as fixture edits:
- deny + rate limit → `WAIT`;
- deny + authentication failure → `HALT`;
- dropped spawn result plus a Bash result → `REISSUE`;
- background, killed or still running → `COUNT maxTurns`;
- killed before the hook entry → `HALT hookMissing`;
- a non-guard `command` → `CLEAN`;
- a terminal `SubagentHandback` → `HALT`, or `COUNT` when the guard allows it.

## Pass 9 — U5.5 re-gate against plan6 v1.5 (2026-11-01)

**Verdict: approve with suggestions.** There is no blocker or major. Every §10.3 shape gives plan6's class, on the fixtures and on the raw probe transcripts. What I ran:
- **Suite:** 656 passed, 2 xfailed, with no deselection; `ruff` is clean; `tests/test_audit.py` has 164 tests.
- **Mutants:** 22, in `scratchpad/p9u55/repo/` with `PYTHONPYCACHEPREFIX` set. Each file was copied aside, restored and checked with `cmp`.
- **Raw transcripts:** I ran `audit_run` read-only over all six headless sessions. Ids are cited; no content was copied.
- **Live tree:** the sha256 of all 46 files was unchanged at the end, and no git command changed the tree. I did not touch `model-bench/`.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; `audit.py` and the fixture builders were read in full.

### Pass 8 findings

- **P8-F1:** superseded by plan6 §3.1/§3.4 (foreground only). Raw H2 (`190af10c`) gives `HALT backgroundSpawn`, and raw H3a (`43b56824`) finds `incomplete`.
- **P8-F2:** superseded by plan6 §3.4 "Terminal shape". A hand-back, hooked or not, is `CLEAN`. M14 and M15 each give 1 failed.
- **P8-F3:** fixed as plan6 §5 re-plants it. P1 gives 2 failed, A6 (counted tied with the usage stop) 2 and A6b (counted tied with halt) 4. The halt-over-counted test now uses a counted row that comes first in the table.
- **P8-F4:** fixed (`Expected.guard_command`, plan6 §3.4 "guard entry").
- **P8-F5:** fixed. Plan6 replaces my "trailing call" fix with "every in-flight call", which is sound for parallel calls (P3-m1). Truncation is read only on a final line with no newline, narrower than I proposed. M2 gives 20 failed and M7 1.
- **P8-F6:** fixed. No line in `README.md` exceeds 100 characters.

### New findings

**P9-F1 — minor — `spawnShapeUnknown` is never applied to an in-flight run.** *Owner:
`tdd-engineer`. Verified by running it.*
- `_check_completion` (`audit.py:466`) tests the meta's shape only when a result is matched.
- Plan6 §3.4's row ("a meta `requestShape` other than `foreground`") has no such condition, and
  in the precedence it sits above the re-issues.
- An in-flight run whose meta says `detached`, or has no `requestShape`, gives
  `REISSUE incomplete` where plan6 gives `HALT`.
- The meta is written at spawn: the raw H3a meta has `requestShape: foreground` without
  `finished`. So the plan's rule costs a killed run nothing.

**Fix:**
- Check `requestShape` whenever a meta exists, and keep the `status` and `finished` clauses for
  matched runs.
- Pin it with an in-flight run and a `detached` meta → `HALT spawnShapeUnknown`.

**P9-F2 — minor — the order of the halt rows is unpinned.** *Owner: `tdd-engineer`. Verified by
running it.*
- Mutant M1 reverses the first four plan6 halt rows in `ROWS`, and all 656 tests still pass.
- The order decides the halt **reason**, and the reason decides what the stakeholder may do at
  `resolve`: `--skip-batch` exists for a safeguards flag (plan6 §3.5).
- Today a refusal plus a guard deny correctly gives `safeguardStop`. No test pins that.

**Fix:** pin two pairs in plan6's table order:
- refusal + `guardRejected` → `safeguardStop`;
- `backgroundSpawn` + `guardRejected` → `backgroundSpawn`.

**P9-F3 — nit — a hand-back refused by input validation is `CLEAN`.** *Owner: `architect` (a
one-line ruling), else `tdd-engineer` follows the row. Verified by running it.*
- The `HANDBACK` branch (`audit.py:591`) comes before the `inputInvalid` branch.
- Plan6 exempts the hand-back only from the bounds checks and `hookMissing`, so the validation row
  would count this case.
- No label is at stake: the answer is ingested from its file.

**P9-F4 — nit — boundaries and caller contracts that nothing pins.** *Owner: `tdd-engineer`.
Verified by running it.*
- The behaviour matches the plan in each case, but no test pins it:
  - M9 (`>=` for `>`) survives. A launch at exactly 6 h gives `launchFailed` and one at 6 h 1 s
    gives `launchFailing`.
  - M17 (an empty driver transcript treated as missing) survives. Today such a file gives
    `neverSpawned` at exit 0 and `launchFailed` otherwise, as the plan says.
- `Launch.fail_bound` defaults to 6 h. If U5.6 omits the bound recorded at `init`, nothing says
  so. Make the field required.
- M19 (a relative `cwd` accepted) is equivalent: a real `cwd` is always absolute.

**P9-F5 — nit — plan6 §5 cites attempt 1 `c5ce5291` "line 46".** *Owner: `architect`.* The
validation entry is line 48; line 46 is an `api-request` entry. The fixture's key sets match
line 48 exactly.

**Carry to the U5.6 brief (`teco`):**
- Leave `interrupted` outcomes out of `earlier_no_attempt` (plan6 §3.2: they do not count toward
  the cap). The flag alone exempts only the current outcome.
- Build `Expected` with absolute paths. With relative paths every run gives
  `HALT guardBreach`, which is loud but wrong.
- Pass 8's seam point stands: the re-ingest path never calls `audit_run` or `combine`.

### Questions 1–6

**1. §10.3 shapes.**
- On raw `b0bbcc3d`, D1 and D2 give `guardRejected` and D3 gives `hookMissing`, so the run halts
  `guardRejected`.
- Raw H6 (`973830b1`) gives `guardBreach`.
- Raw H1 and H4 find only `pathOutOfBounds`, because the kit's answer files end in `.txt`. Under
  dm bounds both are clean.
- The fixture `tool_result` entries for D1–D3 have exactly the raw key sets, and the same
  `permissionDecision`, `toolDenialKind`, block keys, `is_error` and `toolUseResult` type. The
  validation entry matches raw line 48 (P9-F5).

**2. Precedence.** The tiers match plan6 §3.4, and A6, A6b, M8 and M16 each go red. `ROWS` order
does carry meaning the tests cannot see: the halt reason (P9-F2).

**3. The implementer's calls.** None of them needs `architect`.
- `Launch`: the clock resets from the last launch that produced something, read from ledger
  timestamps. M10 gives 1 failed. Bound and default: P9-F4.
- `inputInvalid` is a name the plan leaves open.
- `ENVIRONMENTAL` versus `DRIVER` changes no action: both halt and count nothing.
- Truncation is P8-F5.
- The empty-transcript cases follow the plan's rows: P9-F4 for the driver; the subagent case is
  tested.

**4. Bounds.** All checked by running them:
- `realpath`: M5 gives 3 failed.
- Joining to `cwd`: M6 and M12 give 5 each.
- The separator and sibling prefix: M4 gives 6.
- The driver bound: M18 gives 1.
- Swapping the bound folders gives 115, as teco found.
- `audit.py` imports only the standard library and `.ingest`; `guard_hook.py` is named only in a
  docstring.

**5. Interactions.**
- A call in flight never reaches `inputInvalid`, because that row needs a `tool_result` entry.
- `hookMissing` skips calls in flight (M2: 20 failed).
- An in-flight hand-back leaves the run `incomplete`, and the terminal rule runs only on a matched
  result.
- The one gap is P9-F3.

**6. Docs.** The figures reproduce: 656/2, 164 tests, and the mutants I re-ran match their
`HISTORY.md` counts.
- `AGENTS.md` is 665 words (631 at Pass 8). The audit bullet was rewritten in place, not appended
  to, and no line exceeds 700 characters, so it passes the context-file rule.
- The `README.md` and `HISTORY.md` additions are new scope.

**Growth-stops check:** Pass 9 has one line per Pass 8 finding and new findings only. Passes 1–8
are byte-identical (`cmp`).

### Appendix H — Pass 9 mutants and probes

Each mutant ran with the full suite. Baseline: 656 passed, 2 xfailed.

| ID | Mutation | Result |
|---|---|---|
| M1 | first four halt rows reversed in `ROWS` | **656 passed** (P9-F2) |
| M2 | in-flight calls read | 20 failed |
| M3 | `inputInvalid` ignores attachments | 1 failed |
| M4 / M5 / M6 | prefix without separator / `normpath` / relative path not joined | 6 / 3 / 5 failed |
| M7 | truncation accepted on any line | 1 failed |
| M8 | `spawnShapeUnknown` tier above the usage stop | 1 failed |
| M9 | launch bound `>=` | **656 passed** (P9-F4) |
| M10 / M11 | `last_produced` ignored / `interrupted` ignored | 1 / 1 failed |
| M12 | `cwd` plumbing dropped | 5 failed |
| M13 | Read/Write bound folders swapped | 115 failed |
| M14 / M15 | hand-back exempt from `guardRejected` / hand-back not last allowed | 1 / 1 failed |
| M16 | `finished` not required | 2 failed |
| M17 | empty driver transcript read as missing | **656 passed** (P9-F4) |
| M18 | driver's non-`Agent` bound dropped | 1 failed |
| M19 | relative `cwd` accepted | 656 passed (equivalent) |
| A6 / A6b / P1 | Pass 8 survivors, on the new code | 2 / 4 / 2 failed |

Probes, as fixture edits run through the suite's helpers:
- in flight with a `detached` meta, or with no `requestShape` → `REISSUE` (P9-F1);
- a hand-back refused by validation → `CLEAN` (P9-F3);
- refusal + deny → `safeguardStop`;
- relative `Expected` paths → `guardBreach`;
- launch at 6 h / 6 h 1 s → `launchFailed` / `launchFailing`;
- an empty driver transcript at exit 0 / exit 1 → `neverSpawned` / `launchFailed`;
- the six raw headless sessions → as in question 1.

## Pass 10 — U5.6 campaign ledger and driver CLI (2026-11-01)

**Verdict: needs changes.** There is no blocker, but there are two majors. Both concern a contract that a later unit reads: the per-run provenance and the key → item binding, and the pilot report's keys. Both fixes are small. Re-check them with the suite and the mutants named below; no Pass 11 is needed unless the P10-m1 option (b) is taken.

### Scope

- **Reviewed:** the uncommitted U5.6 diff against `5f1faeb2`. That is `campaign.py`, `driver.py`, `test_campaign.py` and `test_driver.py`, all read whole, plus the `README.md`, `AGENTS.md` and `HISTORY.md` edits.
- **Specs:** plan6 v1.7 §3.2, §3.4, §3.5 and the §4/§5 U5.6 rows; plan5 §3.2, §3.6 and the §4 U5.6 row; plan4 §3.2, §3.3, §3.5, §3.7, §3.8 and the §4.1/§5 U5.6 rows; the carried rulings (P5-F4 cases (a)–(d), P6-F1 case (e), Pass 8 q3, the retired names) and Pass 9's carry. I read `audit.py`'s `audit_run`, `combine`, `run_record`, `Launch`, `Expected` and `_check_bounds`, and `ingest.py`'s `ingest_answer[_files]`.
- **Suite:** run on the live tree, `.venv/bin/pytest -q` from `decision-models/`, with no deselection: **761 passed, 2 xfailed**. `ruff check .` is clean. `test_campaign.py` + `test_driver.py` give 90 tests, which matches HISTORY's "81 and 9".
- **Mutants:** run in a private copy, `scratchpad/p10/repo/`, with a fresh `PYTHONPYCACHEPREFIX`. A probe test confirmed that the copy imports its own package. That copy gives 756 passed + 5 skipped, because the model-bench byte-copy tests skip there. Each mutated file was restored and checked with `cmp`, which reported equal.
  - I re-ran three of the coder's mutants and reproduced its counts: M1 event-after-move gives 6, M2 re-ingest calling audit/combine gives 8, M6 pause ignored gives 1.
  - I also ran 30 mutants of my own (Appendix I).
- **Live tree:** I ran no git command that changes the tree. Outside the scratchpad I wrote only this review, plus gitignored `__pycache__` bytecode under `decision-models/`. The bytecode came from the probes: `python -I` ignores `PYTHONPYCACHEPREFIX`. The sha256 of every reviewed file is unchanged.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded (`GRAPHS` lists only `cpg_falkorchat`); the unit's files were read in full.

### New findings

**P10-M1 — major — the values the ledger carries to `runs[]` and to the key → item binding are pinned by no test.** *Owner: a fresh `coder`/`tdd-engineer` (test-only). Verified by running mutants.*
- **teco's premise is true.** Traced: `issue` writes `attempt` and `requestSha256` into `issued` (`campaign.py:623-626`). `fold` copies both into `Attempt` (`:199-203`). `_outcome_event` passes `attempt.attempt` and `attempt.request_sha256` to `audit.run_record` (`:677-679`), which writes them verbatim into the event's `run` (`audit.py:736-739`).
  - Plan4 §3.7 makes that `requestSha256` the item's generator provenance, replacing v1.3's call hash. U5.10's finalize, which is not yet built, is its only reader. U5.10's §3.7 vector check compares JSON *types* only, so no later check catches a wrong value either.
- **The same class, by mutant.** Each of these gives 0 red on the full suite (Appendix I):
  - teco's two (T1, T2);
  - the same two at the `run_record` call site (T25, T26);
  - `role` fixed to `"label"` (T3). No generation run is ever ingested in a test;
  - `runId` fixed (T4), or numbered over all outcomes rather than located runs (T24);
  - **the worst: `issued.keys` bound to the reversed refs (T5).** That mapping is the only record of which item each answer key labels. Plan4 §3.2: keys are random per (item, vote). Today only `set(keys)` is asserted (`test_campaign.py:221`). A misbinding would attach votes to the wrong items with no error anywhere.
- **Fix** (serves plan4 §3.7 `runs[]` and the §3.2 keys; Appendix I has the tests verbatim). Three tests, each computing its expected value itself:
  - (1) a launchFailed, then a counted void, then an ingested attempt gives `issued.attempt` [1, 2, 3]; `run` is `None`, then (`run_0001`, 2), then (`run_0002`, 3); and `requestSha256` equals sha256 of `store/req/<opaque>.jsonl`'s bytes;
  - (2) every request line's `text` renders the item its `issued.keys[key]` names;
  - (3) a located generation run records `role: "generate"`.
- **Run:** all three pass on the live code, and together they redden all eight survivors, T1–T5 and T24–T26, with 1 red each.

**P10-M2 — major — `init --phase full` reads the pilot report's `cap` and `rngSeeds` with silent defaults.** *Owner: `coder` (fix); `teco` (carry the key names into U5.8's brief). Verified by a probe.*
- The code reads `report["stakeholderDecision"].get("cap", 50)` (`campaign.py:441`) and `report.get("rngSeeds", [])` (`:448`).
- **Probe** (`probes/probe_behaviour.py`): a report that passes the spend gate but spells the keys `capChosen` and `pilotRngSeeds`. `init` succeeds and records `labelBatchCap: 50` and `held-rng-seeds: []`. `fresh_rng_seed` can then return a pilot seed.
- **Why it matters.** The key names are the coder's call 4, and U5.8's row does not fix them. A naming drift in U5.8 would therefore have two silent effects:
  - it undoes plan5 §3.2's addition ("seeds are not reused", made mechanical), which exists only because the pilot store is deleted;
  - it overrides the stakeholder's `--cap 25` (plan4 §3.8).
- **Fix** (serves plan5 §3.2 and plan4 §3.8): refuse unless `stakeholderDecision.cap` is 25 or 50 and `rngSeeds` is a non-empty list of ints. A pilot always draws, so an empty list is wrong. Add one planted-report test per key: the key absent, then mistyped → refused, and no root is created. Mutant T13 already reddens the cap's *value*. These tests cover the key's *absence*.

**P10-m1 — minor — plan6's runner loop, taken literally, re-launches in a spent window.** *Owner: `teco` (choose); then `coder` (U5.6 fix round, option b) or the U6.3 brief (option a). Verified by reading; the tests show the behaviour.*
- Plan6 §3.2 says "`run` exits 0 when `next` returns `stop`", including on a usage stop. But `next` returns `Stop("usage-stop")` only when its *own* reconciliation meets one. After `run`'s step-5 `ingest` records a usage stop, the next `next` serves the same batch again. `test_next_stops_on_a_usage_stop_its_reconciliation_meets` asserts that `next` serves again.
- A U6.3 that follows the plan's text therefore launches `claude -p` again and again inside a spent window. Ten usage stops never halt (plan4 §5), and the ledger grows on every pass.
- **Option (a):** keep the design. Carry "exit on `IngestResult.stop`" into U6.3's brief, with a U6.3 §5 test: a fake child that leaves a `rate_limit` → exactly one launch, exit 0.
- **Option (b), which I recommend:** the `Campaign` instance remembers a usage stop its own `ingest` recorded, and its `next` returns `Stop("usage-stop")`. Plan6's loop is then correct as written, and U6.3 gains no new contract.
  - The assertions for (b): after `ingest` returns a usage stop, the same instance's `next()` is `Stop("usage-stop", …)`, and a fresh `Campaign(root)` serves.

**P10-m2 — minor — an unparsable `store/runs/<sessionId>.exit` makes every `next` raise.** *Owner: `coder`; `teco` (carry the contract to U6.3). Verified by a probe.*
- `_exit_code` calls `int(path.read_text().strip())` (`campaign.py:790`).
- **Probe:** an empty `.exit`, as from a runner killed mid-write, gives `ValueError: invalid literal for int()` out of `next()`. Every timer fire then crashes, and nothing is written to `status` or `store/refusal`.
- **Fix:** read a non-integer file as unobserved. The code already has that branch: exitCode `None`, then `launchFailed`. Test with an empty `.exit` → `launchFailed`, `launch.exitCode is None`.
- U6.3's brief should fix the format: a decimal integer, negative for a signal (Python's `returncode`), written atomically after `wait()` and before `ingest`.

**P10-m3 — minor — the campaign root is `abspath`, not `realpath`.** *Owner: `coder`. The bypass is verified; the slug mismatch is inferred.*
- `Campaign.root = os.path.abspath(root)` (`campaign.py:379`), and two things follow from that.
- **The `root_refusal` bypass (verified).** A root reached through a symlink whose target is inside a work tree with an `AGENTS.md` gives `root_refusal(...) is None`.
- **The slug mismatch (inferred).** `transcript_slug` is built from the symlinked path. `getcwd()` in a child started with `cwd=` a symlink returns the physical path (verified with `pwd -P` and `os.getcwd()` under an empty environment). So Claude Code would write transcripts under the *physical* slug, and every launch would read as `launchFailed` until `launchFailing` fires, 6 h later and for the wrong reason. Claude's own `cwd` handling was not run, because the brief rules out live calls.
- `_check_bounds` already uses `realpath` on both sides, so the guard checks are unaffected.
- **Fix:** `Path(os.path.realpath(root))`. Test: a symlinked root whose target is under a planted `.git` is refused, and `transcript_path` uses the target's slug.
- **Serves:** plan4 §3.5's root refusal and plan6 §3.2's "reconciliation reads only that transcript".

**P10-m4 — minor — call 2's launch-clock restart is unpinned.** *Owner: `coder`. Verified by a mutant.*
- Dropping the epoch filter from `_launch` (T10) gives 0 red. Only the cap's half has a test (T11 gives 1 red).
- **Fix:** with a bound of 1 h, launchFailed at t0, then a `launchFailing` halt at t0 + 61 min, then `resolve`, then launchFailed at t0 + 62 min → `reissued`, not `halted`.

**P10-n1 — nit — other branches that nothing pins** (each gives 0 red; Appendix I). *Owner: `coder`, in the same round.*
- `issue`'s two refusals, the stale request (T14) and the already-issued opaque (T15).
- `launchFailing` keeping its stdout/stderr evidence (T19). Plan6 §3.4 records it for the launch rows, and at a halt it is what the human reads.
- The non-ASCII half of `transcript_slug` (T17). Add a path with `ç`.
- `status`'s "last issued" line (T28).
- `countsAttempt` (T8) and `details` (T7). `fold` counts on `outcome == "void"` and never reads `countsAttempt`, so either `fold` reads it or the event drops it. Today it is a second copy of a fact that can drift.

**P10-n2 — nit — a torn final ledger line bricks every command.** *Owner: `coder`. Verified by a probe.*
- **Probe:** appending `{"event":"queued","at":"2026` gives a `JSONDecodeError` traceback from `status` (probed). `resolve`, `next` and `ingest` read through the same `events()` (`campaign.py:388-392`, traced).
- `audit.py` already treats a final line with no newline as truncation (P8-F5).
- **Fix:** refuse with a `CampaignRefused` that names the byte offset. Never drop the line silently, because it may be a commit point.


### The implementer's calls

1. **`.exit` file, missing → non-zero → `launchFailed`.** This is right. Plan6 §3.4's never-spawned row needs "a child that exited 0", so an unobserved exit cannot claim it. The cost is a bounded, uncapped re-issue rather than a cap. U6.3's row does not own the file name, so `teco` carries the name and format into U6.3's brief (P10-m2).
2. **`resolve` restarts the cap and the clock.** This is acceptable, and no plan row owns it. The trade-off: in an outage that persists after a retry, the re-halt comes one full bound later, not at once. That is cheap, because a launch that produces nothing spends nothing. The clock half is unpinned (P10-m4).
3. **`--skip-batch` only after `safeguardStop`.** This is right. Plan6 §3.5 offers the three actions "at the halt" of a safeguards flag, and the key is `dropped.safeguardSkipped`. `test_skip_batch_is_for_a_safeguard_stop` asserts both the refusal and that the ledger is byte-equal (I read it). The coder reports its M13 at 1 red; I did not re-run it.
4. **Pilot-report keys.** The names are reasonable. `cap` and `stakeholderRef` follow plan4 §3.8's and plan5 §3.6's flag names; `rngSeeds` is the coder's own. U5.8's row does not own them, so `teco` carries them. The silent defaults are P10-M2.
5. **`IngestResult.stop`.** This works, but it departs from plan6 §3.2's exit wording. See P10-m1 for the option I recommend.
6. **Plan6's headless `run` over "ingest only after the hand-back".** This is right. Plan6 §1 carries only the idempotence and Void-is-ingested rulings. §3.2 steps 4–5 ingest after the wait or the kill, and §3.4's `incomplete` row classes an unfinished run. Nothing in `campaign.py` depends on the old rule.

### What's solid

- **Commit point and re-ingest.** I reproduced the coder's M1 (6) and M2 (8). Cases (a)–(e) are planted with the ledger compared byte-equal. A probe crash after the event and before the move also re-ingests as a no-op that finishes the move for the two void shapes the tests do not plant: a part-layout void (`.1` + `.3`, `malformed`) and an `answerMissing` void (`probes/probe_layout_void.py`).
- **Retired names.** A late part (T20: 3 red) and a part that differs from its stored copy (T21: 1 red) are both refused, and nothing is appended.
- **Pass 9's carry is honoured:**
  - `interrupted` stays out of `earlier_no_attempt` (coder's M5);
  - `Expected` paths are absolute (M11);
  - the re-ingest path never reaches `audit_run`/`combine` (M2);
  - `Launch.fail_bound` comes from `init` (M10).
- **Plan rows covered:**
  - plan6 §5 U5.6: session id, a planted second session (M17: 3 red), pause byte-equal, durations, skip-batch;
  - plan5 §4 U5.6: held RNG seed, held `seedId`, the redraw, and the spend-gate branches;
  - plan4 §5 U5.6: resume, the caps, the halt gate and the `init` refusals.
  - Mutants T6, T9, T11, T12, T13, T16, T18, T22, T23, T29 and T30 all redden.
- **Docs:**
  - `AGENTS.md` is 721 words (665 before). The ingest bullet was rewritten in place, not appended to, and no line is over 700 characters.
  - The `README.md` campaign section and the `HISTORY.md` entry are new scope, and their claims match the code.

### Open questions

- **`runs[].attempt` semantics** (for `architect` if it matters to U5.10). The code numbers every issue of a batch, so uncounted re-issues are included. Plan4 §3.7 does not say whether `attempt` means that or the counted-attempt number. P10-M1's test pins today's reading. If it is the wrong one, that test is the place to change it.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*. Source: P10-M1 (T1–T5, T24–T26) and teco's two survivors in the coordination ledger's U5.6 entry.

- Addition: "For a record another unit consumes (a ledger event, a dataset row, a key binding), mutate each field's value at its write site. A field whose value no test pins, with the expected value computed independently, is a finding."

No existing line is now enforced elsewhere, so no prune is proposed.

### Appendix I — Pass 10 mutants and probes

The runner, mutant list and probes are in `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/p10/`: `mutate.py`, `mutants.json`, `probes/`. Baseline: 756 passed + 5 skipped, 2 xfailed. Every mutant ran the full suite, and every file was restored and checked with `cmp`, which reported equal.

| ID | Mutation (`campaign.py`) | Red |
|---|---|---|
| T1 / T2 | `issued.attempt` → `1` / `requestSha256` reversed (teco's) | **0 / 0** |
| T3 | `run_record(role="label")` | **0** |
| T4 / T24 | `runId` constant / numbered over every outcome | **0 / 0** |
| T5 | `issued.keys` bound to the reversed refs | **0** |
| T25 / T26 | `attempt=1` / wrong `request_sha256` at the `run_record` call | **0 / 0** |
| T6 | `voidClass` → `None` | 2 |
| T7 / T8 | `details` → `[]` / `countsAttempt` → `False` | **0 / 0** (P10-n1) |
| T9 | `produced` always true | 2 |
| T10 | launch clock ignores the epoch | **0** (P10-m4) |
| T11 | no-attempt cap ignores the epoch | 1 |
| T12 | unobserved exit recorded as `-1` | 1 |
| T13 | the report's `cap` ignored | 3 |
| T14 / T15 | `issue`'s stale / duplicate-opaque refusals dropped | **0 / 0** (P10-n1) |
| T16 | request written before `issued` | 46 |
| T17 | non-ASCII letters kept in the slug | **0** (P10-n1) |
| T18 | dropped seeds counted as items | 1 |
| T19 | `launchFailing` without launch evidence | **0** (P10-n1) |
| T20 / T21 | late part accepted / differing stored copy accepted | 3 / 1 |
| T22 / T23 | `interrupted` without an open attempt / pilot seeds not held | 1 / 1 |
| T27 / T28 | `resolved.batch` dropped / status "last issued" dropped | 0 / **0** (informational / P10-n1) |
| T29 / T30 | overflow id skips one / `.git` checked on the root only | 1 / 2 |
| coder M1 / M2 / M6 | re-run from the coder's `mutants.json` | 6 / 8 / 1 (as claimed) |

P10-M1's positive control (`probes/test_p10_pins.py`, copied into the scratch repo's `tests/` and then removed): the 3 tests pass on the live code, and T1, T2, T3, T4, T5, T24, T25 and T26 each give 1 red with it present.

Probes (`probes/probe_behaviour.py`, `probes/probe_layout_void.py`; live package imported read-only, writes under `scratchpad/p10/tmp/`):
- an empty `.exit` → `ValueError` from `next()` (P10-m2);
- a torn final ledger line → `JSONDecodeError` from `status()` (P10-n2);
- misnamed pilot-report keys → `labelBatchCap: 50`, held `[]` (P10-M2);
- a symlinked root → the slug differs from the physical cwd's; a symlink into a work tree → `root_refusal` is `None` (P10-m3);
- crash before the move for a layout void and an `answerMissing` void → the re-ingest appends nothing and finishes the move.

**Growth-stops check:** Pass 10 holds only new findings; no earlier finding is re-argued. Passes 1–9 are byte-identical to the pre-pass copy (`cmp` on the prefix).

## Pass 11 — U5.6 fix round (2026-11-01)

**Verdict: approve.** Every Pass 10 finding is fixed, and the fixes hold together. There are no new findings beyond one optional nit.
- **Suite:** 819 passed, 2 xfailed on the live tree; ruff is clean.
- **Mutants:** in a fresh private copy (`scratchpad/p11/repo/`, 814 passed + 5 skipped), with a fresh `PYTHONPYCACHEPREFIX` and without `-I`. Each file was restored and checked with `cmp`.
  - All 26 applicable Pass 10 mutants now redden. T8, T12, T13 and T23 no longer apply because their code changed; T13 and T23 were re-planted on the new code and redden.
  - All 11 new mutants (N1–N11) redden.
  - Two of the coder's 32 (A1, A32) were re-run and give 1 red each, as claimed.
- **Live tree:** the reviewed files' sha256 values are unchanged, and I ran no git command that changes the tree.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded.

### Pass 10 findings

- **P10-M1:** fixed. T1–T5 and T24–T26 each redden: T5 gives 2, through the item binding and the seed binding; the rest give 1. T7 (`details`) and T27 (`resolved` fields) are now pinned too.
- **P10-M2:** fixed. The misnamed-key probe is refused (`cap is None`). N7 gives 15, T13b 3, T23b 1 and N10 3.
- **P10-m1:** fixed with option (b). N1 and N2 give 2 each, and N3 (a re-ingest that sets the flag) gives 1.
- **P10-m2:** fixed. The empty `.exit` probe now gives `launchFailed` with `exitCode: null`. N8 gives 2 and N9 10.
- **P10-m3:** fixed. The symlink-into-tree probe is refused. N6 gives 1, and T30 gives 3.
- **P10-m4:** fixed. T10 gives 1.
- **P10-n1:** fixed. T14, T15, T17, T19 and T28 each redden. `countsAttempt` is dropped; see ruling 1.
- **P10-n2:** fixed. The torn-line probe gives `CampaignRefused` with the byte offset. N4 gives 5 and N5 1.
- **Open question (`runs[].attempt`):** answered by plan6 v1.8 §3.7: the issue ordinal, which matches the code and the P10-M1 test.

### Rulings on the fix round's calls

1. **`countsAttempt` dropped:** right. No reader exists. I swept every non-deprecated `*.py`, `*.md` and `*.jsonl` file. Plan4 §3.7's `runs[]`, plan6 §3.7 and the U5.10 row never name it, and `fold` counts on `outcome == "void"`. `audit.Verdict.counts_attempt` stays, and it is pinned in `test_audit.py`.
2. **Usage stop remembered by the instance only:** this is a contract change, not a weaker test. It matches plan6 §3.2 "Scheduling", where a fire costs at most one short driver turn. The three edited tests now model one fire as one instance:
   - `test_ten_usage_stops…` keeps all its assertions, with a fresh instance per fire;
   - `test_next_stops_on_a_usage_stop…` gains assertions: the same instance stops twice, and a fresh one serves;
   - `test_a_launch_that_produced_something…` moves only the later launch to a fresh instance.
   N1 and N2 redden the loop test, which counts launches.
3. **Splitting on `\n` only:** right. `ensure_ascii=False` writes U+2028 raw, and N11 (`splitlines`) reddens the round-trip test plus all five torn-line tests. `_append` refusing to write after a torn line is right too: anything appended later would sit behind a line no reader can parse.

### Interactions

- **The torn-line refusal and the commit point:** `probes/probe_p11.py` runs this sequence. A crash mid-append of the `outcome` line leaves the line torn, and `next` is refused. Truncating the torn bytes is the human's repair. Re-ingest then gives one `ingested` outcome, and the parts move to `store/` once. Cases (a)–(e) are unaffected because they crash only after a complete append, and the suite is green.
- **`realpath` and the audit:** `Expected` paths, the spawn prompt and the transcript slug all derive from the one physical root. `test_a_symlinked_root_names_the_physical_transcript_folder` runs a clean batch through a symlinked root to `ingested`, so the audit's comparisons agree. `_check_bounds` already used `realpath`.
- **The usage-stop flag and re-ingest:** only a first ingest sets the flag (N3 reddens otherwise). `next` checks the flag after reconciliation and the stray check, so a fire still clears `exchange/` before stopping.
- **The usage-stop flag and pause:** a probe shows pause wins (`Stop("paused")`), and once the pause file is removed the same instance gives `Stop("usage-stop")`.

**P11-n1 — nit — the README's "repair by hand" could name the safe repair.** *Owner: `coder`, optional. Verified by tracing and by a probe.*
- Every writer appends its event before acting on it: `issue` writes only `store/req` first, which is harmless, and the move follows the outcome append. So truncating the torn last line is always safe; the probe above shows it.
- Saying so in the README bullet would spare the human a judgment call.

### Appendix J — Pass 11 mutants and probes

Files are in `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/p11/`: `all-mutants.json`, `re-mutants.json`, `coder-sample.json`, `mutants-run.txt`, `probes/probe_p11.py`.

| ID | Mutation | Red |
|---|---|---|
| N1 / N2 | usage flag never set / never read | 2 / 2 |
| N3 | a re-ingest of a usage stop sets the flag | 1 |
| N4 / N5 | torn last line dropped / `_append` skips the torn check | 5 / 1 |
| N6 | `abspath` root | 1 |
| N7 | `pilot_report_refusal` bypassed | 15 |
| N8 / N9 | exit regex `match` / evidence records `-1` | 2 / 10 |
| N10 | a `stakeholderRef` that is merely truthy passes | 3 |
| N11 | the ledger read with `splitlines` | 6 |
| T13b / T23b | cap ignored / pilot seeds not held (re-planted) | 3 / 1 |
| T1–T7, T9–T11, T14–T22, T24–T30 | Pass 10 set | all red (T10 1, T19 2, T28 1) |
| A1 / A32 (coder's) | `attempt` → 1 / a non-event line accepted | 1 / 1 |

**Growth-stops check:** Pass 11 gives one line per Pass 10 finding, and its only new finding is a nit. Passes 1–10 are byte-identical to the pre-pass copy (`cmp` on the prefix).

## Pass 12 — U6.3 driver run (2026-11-01)

**Verdict: approve with suggestions.** The runner implements plan6 §3.2 and every §5 U6.3 row, and the code has no blocker and no major.
- **The one major, P12-M1, is the plan's,** not the code's: the user-settings refusals miss keys the plan's own threat model excludes elsewhere. It goes to `architect` and should be settled before U5.7 builds `install`, because `install` records the hash.
- **The three minors have tested fixes,** each a few lines plus its probe as a test: a group SIGKILL the runner skips, a paused fire that refuses, and a root refusal `status` never shows. They suit one short `coder` round. A `teco` line check suffices for that round if the fixes are taken as written: the probes become tests and turn red against today's code (Appendix K).
- **Suite:** 854 passed, 2 xfailed, as claimed.
- **Mutants:** 30 in all, my 29 plus the coder's M25. 16 redden on the suite as it stands, and the probes kill 7 of the 14 survivors. Of the other 7:
  - R16, the group SIGKILL, is killed only by P1 inverted against the fixed code;
  - R6 needs P12-n2's timestamp assertion, which I did not run;
  - R3, R22, R27, R36 and R37 are harmless or equivalent (Appendix K).

### Scope

- **Reviewed:** the uncommitted U6.3 diff against `c2328164`. I read `runner.py` and `test_runner.py` whole, and the `campaign.py`, `driver.py`, `README.md`, `AGENTS.md` and `HISTORY.md` diffs. I also read `campaign.py`'s `next`, `issue`, `ingest`, `_launch`, `_observed_exit`, `locked`, `root_refusal` and `status`, and `audit.py`'s interrupted cap.
- **Specs:** plan6 v1.8 §3.1–§3.3, the §4 U6.3 row and the §5 U6.3 row (the acceptance list); plan4 §3.5 and §4.3; Passes 10–11 and the coordination ledger's U6.3 carries. For the settings keys, I checked the Claude Code settings docs (`code.claude.com/docs/en/settings` and `/settings-reference`).
- **Suite:** run on the live tree from `decision-models/` with a fresh `PYTHONPYCACHEPREFIX`: **854 passed, 2 xfailed**, which reproduces the claim. `ruff check` is clean.
  - `tests/test_runner.py` alone: 5/5 green unloaded, 3/3 green under 16 busy loops on 8 CPUs, 4/4 green as four parallel runs under 32 busy loops, and red under 64 busy loops (P12-n1).
- **Mutants:** run in a private copy (scratchpad `p12/repo/`, since lost in a session crash, Appendix K), with a fresh `PYTHONPYCACHEPREFIX` per run, without `-I`, and `.venv/bin/pytest` run from the copy's `decision-models/`. A probe test confirmed that the copy imports its own package. Baseline: 849 passed + 5 skipped (the model-bench byte-copy tests skip there). Every mutated file was restored and checked with `cmp`, which reported equal. I ran 29 mutants of my own plus a re-run of the coder's M25 (Appendix K).
- **Suggested fixes** were applied in a second private copy (`repo2`, also lost in the crash). There the full suite stays green and the matching probes turn red (Appendix K).
- **Live tree:** I ran no git command that changes the tree, and I wrote only this review. Before the crash, the sha256 of every reviewed file matched the snapshot I took at the start of the pass; that snapshot was lost with the scratchpad, and teco's post-crash re-run (854 passed, 2 xfailed, ruff clean) is the only later check. I never touched the real `~/.claude`: every probe runs under the tests' fake `HOME` and fake `claude`.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded (Pass 10/11 checked `GRAPHS`); the unit's two new files were read in full.

### Findings

**P12-M1 — major — plan6 §3.2's user-settings refusals miss the settings-file twins of what the plan itself excludes.** *Owner: `architect` (plan6 §3.2 and the §5 U5.7 row), before U5.7 builds `install`. Not a U6.3 code defect: the code implements the plan's list exactly (ruling 2). Checked against the docs; the effect on the child is inferred, because the brief rules out live CLI calls.*
- Plan6 drops `--fallback-model`, every `ANTHROPIC_*` variable (billing and routing) and `CLAUDE_CODE_EFFORT_LEVEL` (effort). It calls the guard "the only bound" (§3.3) and refuses only an `env`/`hooks` key, `autoContinueAtUsageLimit: true`, and a `permissions`/`modelSettings` hash.
- The settings reference lists user-settings keys that do the same things, and none of them is refused or hashed:
  - `fallbackModel` ("Scope: Any file"; "`--fallback-model` takes precedence over this key");
  - `apiKeyHelper` ("Generate the API credential with your own command");
  - `disableAllHooks` ("Turn off hooks … at once");
  - top-level `effortLevel` (`CLAUDE_CODE_EFFORT_LEVEL` and `--effort` override it);
  - `outputStyle` ("changes Claude's role, tone, and output format").
- **Why it matters.** The audit catches some of these only after the run has happened. A guard switched off by `disableAllHooks` shows as `hookMissing`, but only after an unguarded run. A model change shows as a model-pin halt. An `apiKeyHelper` billing switch leaves no trace in the transcript, so nothing ever catches it.
- **Suggested direction, for the architect to choose:** invert the hash. Hash the whole settings object minus a named set of cosmetic keys (`model`, `theme`, `editorMode`, `verbose`, …), which keeps the stakeholder's "a `theme` or `model` edit does not refuse". Or add the five keys above to the refusals. Either way, U5.7's `install` records whatever `run` compares, and the §5 U6.3 row gains one planted case per key. I have not run either design: the runner's hash function is the only code that changes, and the plan decides which keys it covers.

**P12-m1 — minor — the timeout and `pause --now` kill skip the group SIGKILL whenever the leader exits within the grace, and nothing pins that SIGKILL or the grace.** *Owner: `coder`. Verified by a probe and by mutants.*
- `_kill_group` (`runner.py:276-282`) sends SIGTERM to the group, then waits on the **leader** only, and SIGKILLs the group only if the leader is still alive at the grace. A descendant that survives SIGTERM therefore outlives the run. It holds the lock too, since it inherits the `pass_fds` descriptor, as a libuv child inherits any non-CLOEXEC fd. So every later fire exits on the lock.
- **Probe P1:** the leader dies on SIGTERM; its grandchild ignores SIGTERM and inherits fds. After `run` returns, the grandchild is alive and the next `run` prints `LOCK_HELD`. Plan6 §6 accepts an orphan as rare *because it needs the runner to die*; this path needs no runner death.
- Plan6 §3.2 step 4 says "SIGTERM to the group, then SIGKILL after 30 s". The module docstring, `README.md` and `HISTORY.md` repeat that, so all three overstate what the code does.
- **The tests miss it** because the timeout test's grandchild dies on SIGTERM:
  - R16 (SIGKILL to the leader only) gives 0 red;
  - R35 (grace ignored, so SIGKILL straight after SIGTERM) gives 0 red. The test's `1 <= elapsed` bound is the batch timeout, not the timeout plus the grace.
- **Fix** (serves plan6 §3.2 step 4 and the §5 "killed as a group" row; tested in `repo2`): after the SIGTERM, poll until the grace ends, reaping the leader with `child.poll()` and probing `os.killpg(pgid, 0)`. If the group is not empty by then, SIGKILL it.
  - Tests: P1 inverted (the TERM-ignoring grandchild is dead after `run` returns) and P11 (`elapsed >= timeout + grace`).
  - With the fix the suite stays green and P1 turns red. P11 kills R35 (Appendix K).
- Carry for U6.4: under a systemd unit with `KillMode=control-group`, leftover processes are killed when the service stops. Under cron, the WSL2 fallback, nothing kills them. That makes this fix the only guard there.

**P12-m2 — minor — a paused campaign with drifted user settings refuses (exit 1) instead of exiting 0 at once.** *Owner: `coder`. Verified by probe P2.*
- The settings check runs before `next` (call 4), so it also runs before the pause check. Probe P2 sets `store/pause` and adds a `hooks` key. `run` then exits 1 and writes `store/refusal`.
- That contradicts plan6 §3.2: "A fire while paused exits at once" and "`run` exits 0 when … `store/pause` exists". Pausing in order to edit your settings is the natural sequence. Under U6.4's timer, each fire in that window would then fail and overwrite `REFUSED`.
- **Fix** (tested in `repo2`): check `camp.pause_path.exists()` at the top of each iteration, before the settings check, and stop with `stopped: paused`. That check reads nothing, so call 4's "a refusal writes nothing" still holds. With the fix the suite stays green and P2 turns red.
- **Related gap (R28 gives 0 red):** the call-4 placement itself is unpinned. In the fixture, `next` writes nothing, so moving the settings check after `next` keeps the ledger byte-equal. Probe P12 plants an issued attempt that `next` would reconcile, and R28 then reddens it (Appendix K). Add P12 to serve the §5 row's "no ledger write".

**P12-m3 — minor — a root refusal never reaches `status`, although `README.md` and `driver.py`'s docstring say it does.** *Owner: `coder`. Verified by probe P3.*
- `Runner.run` raises the root and no-campaign refusals before `_refusing`, so they go to stderr only (call 5). The no-campaign case is fine, because `status` refuses the same way.
- The root case is not. Probe P3 plants `AGENTS.md` in the campaign's parent, and `status` then prints `batches: 1 pending` with no `REFUSED` line.
- A timer-driven campaign stalls silently, with the reason only in the scheduler's log, if a `.git`, `CLAUDE.md` or `AGENTS.md` later appears in any ancestor, the home directory included.
- `README.md` lists the root refusal with "The reason, with a timestamp, goes to `store/refusal`". The docstring says "`run` … leaves the reason in `store/refusal`". Both claims are false for this case.
- **Fix** (tested in `repo2`): `status` computes `root_refusal(self.root)` live and prints it as a `REFUSED:` line. That is read-only and always current, with no write before the lock. With the fix the suite stays green and P3 turns red. Then correct the README sentence.

**P12-n1 — nit — the timeout test is load-sensitive.** *Owner: `coder`. Verified by running it.*
- The test fails under 64 busy loops on 8 CPUs, 4 of 4 runs. It passed every run under up to 32 busy loops.
- Cause, from the failed run's ledger: the fake `claude` is still starting when the 1 s `batchTimeout` expires. It has not yet set `SIG_IGN`, because it first imports `test_campaign` to stage. So the SIGTERM kills it before it records the call. The outcome is `launchFailed {exitCode: -15}`, and `made.calls()[0]` raises `IndexError`.
- **Fix:** have the fake apply `ignore_term` as its first statement, and give the test a 3 s `batchTimeout` with the watchdog and the upper bound scaled to match (Appendix K reports the run).

**P12-n2 — nit — branches no test pins.** *Owner: `coder`, in the same round. Each verified by a mutant giving 0 red.*
- **SIGINT** (R20). The README promises "Ctrl-C … does the same", and probe P4 shows it does: one `interrupted`, exit 0. P4 kills R20.
- **A signal that lands between batches** (R29). Probe P5 shows `stopped: signalled` with no `interrupted`, and kills R29.
- **M25** (see ruling below). Probe P8 kills it.
- **The hash's canonical form** (R10 `sort_keys=False`, R11 default separators):
  - plan6 §3.2 says "canonical JSON", and `install` (U5.7) will record this value;
  - P9 (reordered `permissions` keys hash equal) and P10 (the sha256 of a hand-written canonical string) kill both;
  - without P9, a hand edit that reorders keys would refuse every launch.
- **The `store/refusal` timestamp format** (R6: epoch seconds pass `refusal[:4].isdigit()`). Assert `datetime.fromisoformat` on the first token.
- **Never-pinned, harmless:** R3 (`runner.json` written non-atomically; `pause --now` reads a torn record as "no runner"), R22 (`pause --now` sends SIGINT) and R27 (`store/refusal` deleted before `Popen` succeeds). I propose no test for these.

**P12-n3 — nit — small behaviours worth a line.** *Owner: `coder`, optional.*
- **A signal during `next`/`issue` still launches a child,** which is then killed about 50 ms later. Probe P6 shows `launched session …` and then `interrupted` with no call recorded. Checking `self.signalled` after `issue` and before `Popen`, and returning after `record_interrupted`, would skip the launch. It is harmless today: no `.exit` means `launchFailed`, which is uncapped.
- **`_signal_runner` calls `os.kill` unguarded** (`runner.py:322`). A runner that exits between the start-time check and the kill gives `ProcessLookupError` and a traceback after `store/pause` was already created. This is inferred from reading; I did not race it.
- **The halt reason is lost on the ingest path.** `_batch` returns the bare `result.stop`, so the run prints `stopped: halted ` with no reason (probe P8's output). The reconciliation path prints `halted (<reason> (<class>) on <batch>)`.

### Rulings on the coder's calls

1. **`driverTemplate` and `userSettingsSha256` in the `expectations` event; `run` refuses without them.** Right.
   - The `expectations` event is plan4 §3.5's "declared expectations, written at `install`, never learned from a run". U5.7's row owns writing both values ("the driver template …", "the user-settings record at install").
   - Both are pinned at their write site: R8 (template value) gives 1 red, and R9 (hash value) gives 21.
   - **Carry to U5.7:** compute `promptSha256.spawnTemplate.driver` from the same `driverTemplate` string it records, so the text has one source. Record the hash with `runner.user_settings_sha256`, not a second implementation.
2. **The hash covers `permissions` and `modelSettings` only; an absent key is not `null`; a missing file is `{}`.** This matches plan6 §3.2's list exactly. It fits the stakeholder's rule: `theme`, `model`, `verbose` and `autoContinueAtUsageLimit: false` edits launch (the test is parametrized).
   - "Missing file = `{}`" matches Claude Code, which creates no settings file until one is needed (settings docs).
   - Refusing a non-object or unparsable file is right. Under `-p`, Claude Code silently skips a broken file (docs), and the child would then run with none of the user's rules.
   - The list itself is too narrow: P12-M1, owned by the plan.
   - The canonical form is unpinned: P12-n2, P9/P10.
3. **`run` stops after a `launchFailed` outcome.** Right. It follows plan6 §3.2 "Scheduling", where a fire costs at most one short turn.
   - It keeps `launchFailing` reachable: one fire per 30 minutes crosses the 6 h bound.
   - Pinned: the coder's M24 gives 2 red. No later unit owns it, but U6.4's cadence sets the retry interval.
4. **Settings checked at the top of each iteration, before `next`.** Right for "a refusal appends nothing". `next` can append a reconciliation outcome or a `cut`, so the check has to come first.
   - Two gaps remain: the placement is unpinned (R28, probe P12), and the pause must be honoured before the check (P12-m2).
5. **`store/refusal` only after the lock; the root and no-campaign refusals go to stderr only.** Acceptable for the no-campaign case, because `status` refuses the same way. Not for the root case: P12-m3.
   - With that fix, `status` shows every refusal a timer-driven user can hit, apart from an uncaught exception. An example is a `PermissionError` reading `settings.json`, which ends in a traceback with no `REFUSED` line; this is inferred from reading, and I did not run it. U6.4's scheduler log is the backstop.
6. **Exit codes: a refusal exits 1; every stop exits 0** (a held lock, a pause, `launchFailed`, a signal). Right.
   - Exit 1 is the case a human must act on, and systemd marks the unit failed for it.
   - A signal is an operator's stop: SIGTERM from `systemctl stop` or `pause --now`.
   - Pinned: R25 gives 2 red. P12-m2 is the one place where this rule and the code disagree.
7. **SIGINT is treated as SIGTERM; no ingest after `interrupted`.** Right. Ctrl-C reaches only the runner's process group, because the child has its own session, so the runner must kill the child itself.
   - Probe P4 shows one `interrupted` and exit 0. It is unpinned in the suite (R20; P12-n2).
   - Not ingesting is right, because the next `run` reconciles with the real `.exit` value.
8. **`runner.json` never deleted; child stdin `/dev/null`; an `OSError` at launch writes `.err` and no `.exit`.** All three are right.
   - The start-time check is what makes a stale `runner.json` safe (row test). Deleting the file at exit would add a write that a crash skips anyway.
   - `/dev/null` is pinned by the coder's M31.
   - No `.exit` means "never observed", which is U5.6's `launchFailed` with `exitCode: null`. The missing-binary test pins it.
   - **`pause --now` racing a child that is just exiting** (probe P7, made deterministic): `.exit` holds the true code `0`, one `interrupted` is appended, and the next `run` ingests the batch normally. That is correct: `interrupted` only lifts the cap.
- **M25 is not strictly equivalent.** Without the explicit exit, the next iteration re-runs the settings check before `next` reports the halt. So a halting batch whose child (or the user) edits the settings exits 1 with `REFUSED` instead of `stopped: halted`, exit 0 (probe P8 kills M25). Keep the branch and add P8 (P12-n2).
- **The ledger's carries are right.**
  - U5.7: `install` records the template and the hash, under `Campaign.locked()`.
  - U6.4: set `PATH` and `LANG` explicitly; refusals exit 1. U6.4 should also record the unit's `KillMode` (P12-m1).

### What's solid

- **Every §5 U6.3 row has a test, and runs against a fake `claude` under a fake `HOME`.**
  - Environment and argv: the planted `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `CLAUDE_CODE_EFFORT_LEVEL` and `CLAUDE_CODE_FORK_SUBAGENT` are all absent, and the expected argv is built independently from the template.
  - The settings refusals, each with the ledger byte-equal and the `REFUSED` line in `status`.
  - Single flight, pause in flight, `pause --now` with exactly one `interrupted` and the uncounted reconcile, the orphan, the stranger PID and the root.
  - These mutants redden: R1, R2, R4, R5, R7, R8, R9, R14, R15, R17, R21, R25, R30, R31, R32 and R34.
- **The U5.6 contracts are honoured, each pinned:**
  - one `Campaign` per process (the coder's M6);
  - `.exit` written atomically, in decimal, before `ingest` (R4, R5 and the coder's M4/M5);
  - `cwd=exchange_dir` (M30);
  - `interrupted` written once (M7);
  - the lock passed through `pass_fds` (M8).
- **The values a later unit reads are pinned against independently computed values:**
  - `runner.json` against `/proc/<pid>/stat` read by the test (R1, R2);
  - `.exit` against `-signal.SIGTERM`/`-SIGKILL` and the planted exit 3 (R4, R5);
  - the `expectations` keys through the argv and the refusal paths (R8, R9).
- **Signal handling is sound.** The handler only sets a flag, so no ledger append can be torn by a signal. Handlers are installed before `runner.json` is written, so `pause --now` can never kill the runner with SIGTERM's default action. The child is in its own session, so Ctrl-C at a terminal never hits it directly.
- **Docs:**
  - `AGENTS.md` grew 721 → 792 words, edited in place, with no line over 700 characters.
  - The README section and the HISTORY entry are new scope.
  - Their claims match the code except the two P12-m1/m3 sentences named there.

### Open questions

- **P12-M1's direction,** for `architect` via `teco`: hash everything except a cosmetic allowlist, or extend the refusal list? Either way it must be settled before U5.7's `install` fixes the recorded value. Whether a user-level `disableAllHooks` also disables the project-level guard wiring was not confirmed in the docs excerpt I read; U5.7's smoke could plant it.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*. Source: P12-m1 (R16 and R35 survive because the fixture's descendant dies on the first signal) and P12-n1 (a 1 s timeout races the fixture's own start-up).

- Addition: "For code that stops a process tree (a group kill, a grace period, a timeout), plant a descendant that survives the first signal and assert it is gone afterwards; assert the grace by elapsed time. Run timing-dependent tests under CPU oversubscription before calling them stable."

No existing line is now enforced elsewhere, so no prune is proposed. Pass 10's proposed field-mutation line is still pending feature close; this pass applied it (R1–R9).

### Appendix K — Pass 12 mutants and probes

**Files lost.** The scratchpad holding this pass's files was wiped in a session crash after the runs finished: `p12/mutants.py`, `p12/mutate.py`, `p12/probes/test_p12_probes.py`, `p12/probes/test_p12_fixcheck.py`, the two private copies, the logs and the sha256 snapshot. The tables below are transcribed from the run output recorded in my session transcript. Nothing was re-run after the crash.
- To reproduce, rebuild a private copy as Scope describes, then re-plant each mutant from its description and each probe from the list below.
- Rules for every run: each mutant ran the full suite in the copy and was restored, and `cmp` reported equal. Baseline: 849 passed + 5 skipped, 2 xfailed. With the probe file present: 861 passed + 5 skipped.

| ID | Mutation (`runner.py` unless noted) | Red (suite) | Red (+ probes) |
|---|---|---|---|
| R1 | `runnerPid` + 1 | 2 | — |
| R2 | start time read from field 23 (write and check) | 1 | — |
| R3 | `runner.json` written non-atomically | **0** | 0 (harmless) |
| R4 / R5 | `.exit` without newline / `abs(returncode)` | 4 / 2 | — |
| R6 | `store/refusal` stamped in epoch seconds | **0** | 0 (P12-n2) |
| R7 | stamp after the reason | 5 | — |
| R8 / R9 (`campaign.py`) | `driverTemplate` + `" "` / `userSettingsSha256` reversed, at the write site | 1 / 21 | — |
| R10 / R11 | hash `sort_keys=False` / default separators | **0 / 0** | 2 / 1 (P9, P10) |
| R14 | refused keys by truthiness | 3 | — |
| R15 | missing settings file refuses | 1 | — |
| R16 | SIGKILL to the leader only | **0** | 0; red against P1 inverted on the fixed code |
| R17 | SIGTERM to the leader only | 1 | — |
| R20 | SIGINT not handled | **0** | 1 (P4) |
| R21 | `process_group=0` instead of a new session | 1 | — |
| R22 | `pause --now` sends SIGINT | **0** | 0 (harmless) |
| R25 | held lock exits 1 | 2 | — |
| R27 | `store/refusal` deleted before `Popen` succeeds | **0** | 0 (harmless) |
| R28 | settings checked after `next()` | **0** | 1 (P12) |
| R29 | `while True` (a signal between batches is ignored) | **0** | 1 (P5) |
| R30 | straight to SIGKILL | 1 | — |
| R31 | `batchTimeout` × 60 | 1 | — |
| R32 | `LC_ALL` dropped | 1 | — |
| R34 | stdout/stderr swapped | 1 | — |
| R35 | grace ignored (`wait(timeout=0.001)`) | **0** | 1 (P11) |
| R36 | signalled after `issue`: return without launching | **0** | not re-run (P12-n3's design, not a defect) |
| R37 | stdout opened in append mode | **0** | not re-run (equivalent: fresh session ids) |
| M25 (coder's) | no exit on `IngestResult.stop` | **0** | 1 (P8) |

**Probes** (pytest functions in the copy's `tests/`, using `test_runner`'s `Bench`, fake `claude` and fake `HOME`). Each asserts today's behaviour, so a fix turns it red:
- P1: leader dies on SIGTERM; a grandchild spawned with `close_fds=False` ignores SIGTERM. After `run` (timeout 1 s, grace 0.5 s), the grandchild is alive 1 s later, and the next `run` prints `LOCK_HELD`. Observed: alive, `LOCK_HELD`.
- P2: `pause`, then a `hooks` key, then `run` → exit 1 and `store/refusal` written.
- P3: `AGENTS.md` in the root's parent, then `run` → exit 1. `status` then prints `point …`, `batches: 1 pending` and no `REFUSED` line.
- P4: SIGINT to a `driver run` process mid-batch → exit 0, `stopped: interrupted bat_0001`, one `interrupted`.
- P5: SIGTERM delivered inside `ingest` (monkeypatched) with two batches → `stopped: signalled`, one launch, no `interrupted`.
- P6: SIGTERM delivered inside `issue` → `launched session …`, then `interrupted: killing …`, with 0 calls recorded by the fake.
- P7: `Popen.wait` patched to report one timeout after the child exited 0, with SIGTERM raised at that moment → `.exit` `0`, one `interrupted`; the second run's outcomes are `[ingested]`.
- P8: the child halts the batch (`opus-driver`) and writes a `hooks` key → exit 0, `stopped: halted ` (no reason), no `store/refusal`.
- P9: `permissions`/`modelSettings` with reordered keys hash equal.
- P10: the hash equals the sha256 of `{"modelSettings":{"effort":"high"},"permissions":{"allow":["Read"],"deny":["Bash"]}}` for a settings object that also holds `theme`.
- P11: leader ignores SIGTERM; timeout 1 s, grace 1.5 s → elapsed ≥ 2.5 s.
- P12: an `issued` attempt planted with `next()` + `issue()`, then a `hooks` key, then `run` → exit 1 and the ledger byte-equal.

**Suggested fixes, applied in `repo2`** (the code is in P12-m1/m2/m3):
- the polling group kill;
- the pause check before the settings check;
- `status` printing `root_refusal`.

Results:
- Full suite plus probes: 856 passed + 5 skipped. Exactly P1, P2 and P3 failed, each for the intended reason ("grandchild was killed", `0 == 1`, a `REFUSED:` line present).
- P1 inverted (the grandchild is dead and the next run is not `LOCK_HELD`) passes on the fixed code.
- With the fixed code's group SIGKILL changed to `os.kill(child.pid, …)`, P1 inverted fails and P11 passes. P1 inverted thus kills R16.
- A stray probe grandchild left alive by one mutant run was found with `pgrep` and killed.

**P12-n1's fix,** applied in `repo2`'s `test_runner.py`:
- the fake sets `SIG_IGN` right after reading its plan;
- `batchTimeout` 3 s;
- watchdog 20 s;
- bounds `3 <= elapsed < 18`.

The timeout test passed 4 of 4 runs under 64 busy loops, at 5.8–8.1 s each. Unfixed, it failed 4 of 4 under the same load, with ledger `outcome reissued launchFailed {exitCode: -15}`.

**Growth-stops check:** Pass 12 gives no disposition lines because no earlier finding is re-opened, and every finding in it is new. Passes 1–11 are unchanged: `git diff` against `HEAD` shows a single hunk, purely additive, starting after Pass 11's last line.

## Pass 13 — U6.3 fix round re-gate (2026-11-01)

**Verdict: approve.** Every Pass 12 code finding is fixed and each fix is pinned by a test that turns red under the mutant that undoes it. I found no new defect in the way the round's fixes interact. There are two nits.

### Scope

- **Reviewed:** the uncommitted fix round against `c2328164`, diff-scoped: `runner.py`'s `_loop`, `_stopped`, `_batch`, `_kill_group`, `_group_alive` and `_signal_runner`; `campaign.py`'s `status`; `driver.py`'s docstring; `tests/test_runner.py`'s fixture and the new tests; and the README and HISTORY claims about them. Settings-key scope (P12-M1) is out of scope: plan6 v1.10 / U6.3b owns it.
- **Suite:** run on the live tree from `decision-models/` with a fresh `PYTHONPYCACHEPREFIX`: **864 passed, 2 xfailed**; `ruff check` clean. `tests/test_runner.py` alone: 3/3 green unloaded (45 tests, 11 s), and 2/2 green under 64 busy loops on 8 CPUs (about 57 s each). Pass 12's P12-n1 failure under that same load is gone.
- **Mutants and probes:** private copy `scratchpad/p13/repo/` (this pass's own, built fresh), a fresh `PYTHONPYCACHEPREFIX` per run, no `-I`. Baseline 859 passed + 5 skipped. Every file was restored and `cmp` reported equal. 19 mutants, and 6 probe tests that I added to the copy only (Appendix L).
- **Live tree:** no git command that changes it; I wrote only this review. The reviewed files' sha256 values are in `scratchpad/p13/live-sha256.txt`.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; the changed hunks were read in full.

### Disposition of Pass 12

- **P12-m1** (group SIGKILL): fixed. L1 (SIGKILL to the leader only) gives 1 red, L3 (`_group_alive` always false) 2, L4 (always true, so the grace is always waited out) 2, L5 (leader never reaped) 2, and L20 (grace ignored) 2.
- **P12-m2** (pause before settings): fixed. L6 gives 1 red, and L15 (settings after `next`) gives 1.
- **P12-m3** (root refusal in `status`): fixed. L10 gives 1 red. The README and the `driver.py` docstring now say that the root refusal writes nothing and `status` computes it.
- **P12-n1** (load-sensitive timeout test): fixed. It passes 2/2 under 64 busy loops.
- **P12-n2:** fixed, with these results:
  - L19 (no SIGINT) gives 1 red.
  - L17 and L18 (hash canonical form) give 1 red each.
  - L16 (epoch timestamp) gives 5.
  - M25 gives 1.
  - R29 survives the suite: see question (b).
  - R3, R22 and R27 stay unpinned, as I advised.
- **P12-n3:**
  - The signal during `next` is fixed (L7 gives 1 red); see question (a).
  - The unguarded `os.kill` is fixed (L11 gives 1).
  - The lost halt reason is fixed (L13 gives 1). The usage-stop reason is the same code path and is unpinned (P13-n1).

### Answers to the gate's questions

**(a) Checking `signalled` after `next()`, before `issue`: sound, and better than my suggestion.** Verified by running.
- It leaves no `issued` event to unwind, so there is no `interrupted` to write and nothing for the next run to reconcile. `test_a_signal_while_next_runs_launches_nothing` pins it: 0 calls, 0 `issued`, and L7 reddens it.
- **The window that remains.** A signal landing after the check and before `issue` still launches, and then kills, a child. Probe A1 plants the signal just before `issue`: exit 0, one `interrupted`, one launch, and the next run ingests the batch to `ingested`. The residue is a harmless, bounded launch (plan6 §3.2 counts an `interrupted` re-issue toward nothing). No test pins this window, and I do not ask for one.
- **A `next()` that appended events.** Breaking after `next` leaves whatever `next` reconciled. Probe A2 (a signal after `next` on a campaign with a pending batch, then a second run) gives `ingested` with the first run's ledger bytes as a prefix. `next` appending a `cut` was not exercised: the fixture has no overflowing batch. That `prepare` is idempotent after a `cut` is inferred from `fold`, where the batch's entries shrink to `keep`.

**(b) R29 (`while True`) is not equivalent, but the difference is benign.** The coder is right that the suite cannot see it.
- **Same:** with a second batch pending, the new post-`next` check breaks at once, so `stopped: signalled`, no launch and no ledger change are identical.
- **Different, probe B1:** a signal lands in the last ingest, and the user settings have drifted by then. With the guard, `run` stops `signalled` and exits 0. Under R29 the next iteration re-checks the settings and exits 1 with `REFUSED`. The probe kills R29 (1 red, in a copy that holds the probe).
- **Different, message only:** with nothing pending after a signal, R29 prints `stopped: done`; the guard prints `signalled`.
- **Ruling:** an exit-0 operator stop turning into a refusal is the same class as P12-m2, but it needs a signal and a settings edit in the same few milliseconds. Taking B1 as a test is optional; I report no finding.

**(c) The fixture change leaves no test vacuous.** I checked each kill-path mutant against the new fixture.
- The `grandchild` plan now resets SIGTERM to the default, because the leader's `SIG_IGN` is set first. The timeout test's grandchild therefore dies on a group SIGTERM, but it no longer distinguishes a group SIGTERM from a leader-only one. That job belongs to the pause-now test, and L2 (SIGTERM to the leader only) gives 1 red there.
- `stubborn_grandchild` ignores SIGTERM from before its `exec` and holds the lock fd, which the test asserts through `/proc/<pid>/status` (`SigIgn`) and `/proc/<pid>/fd`. That makes the test meaningful: L1 reddens it, and it fails if the descendant ever stops being stubborn.
- The `ignores_sigterm` assertion is independent of the runner.
- The timeout test's `3 + 1 <= elapsed` bound is `batchTimeout` plus grace, and L20 reddens it.

**(d) Interactions of the round's fixes: none defective.** Checked by running or by L-mutants:
- **Pause check × settings check × `next`'s own pause check.** The loop-top check returns `stopped: paused (store/pause exists)`, the same text `next` would give, so nothing a caller reads changed. A paused campaign with no install record now exits 0 instead of refusing, which fits "a fire while paused exits at once".
- **`_group_alive` × `child.poll()`.** The loop reaps the leader before probing, so a zombie leader is not counted as a member forever. Without it (L5), a leader that dies on SIGTERM costs the whole grace. L4 and L5 each give 2 red, through the default-grace tests `pause_now` and SIGINT (their `finish(timeout=15)` is shorter than the 30 s grace), not through a timing assertion. Those two tests therefore guard the fast path only while the grace default stays above 15 s; the 30 s constant is pinned.
- **Probe D1** (the leader never exits, a stubborn descendant, a 2 s timeout, a 0.5 s grace) gives the second launch and then a `halted incomplete` stop at 5.1 s, and the descendant is dead. That is the group kill working through the timeout path, not only the signal path.
- **`status` root refusal × `init`.** A valid campaign never has a root refusal, because `init` refuses the same root, so the line cannot appear on a healthy campaign. No test pins the absence of the line; I report none.

### New findings

**P13-n1 — nit — the usage-stop reason in `stopped: usage-stop (...)` is unpinned.** *Owner: `coder`, optional. Verified by mutant L14, which gives 0 red.*
- `_batch` builds the stop detail from `result.outcome["reason"]` for a non-halt stop. Dropping it leaves the suite green, since `test_a_usage_stop_launches_once_and_exits` asserts only `"usage-stop" in out`. One `==` on the printed line would pin it.

**P13-n2 — nit — the README's kill sentence says "30 s later, SIGKILL to any process of the group still alive", and the code stops waiting early when the group is empty.** *Owner: `coder`, optional. Verified by reading `_kill_group`.*
- The sentence is true. Saying "once every process of the group has gone, the run does not wait out the 30 s" would also state the fast path that L4 and L5 guard.

### Appendix L — Pass 13 mutants and probes

Files are in `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/p13/`: `mutants.py`, `mutate.py`, `probes/test_p13_probes.py`, `logs/mutants-run.txt`, `logs/probes.txt`. Baseline in the copy: 859 passed + 5 skipped, 2 xfailed (865 with the probe file). Each mutant ran the full suite; each file was restored and `cmp` reported equal.

| ID | Mutation (`runner.py` unless noted) | Red |
|---|---|---|
| L1 | group SIGKILL becomes `child.kill()` | 1 |
| L2 | group SIGTERM becomes leader only | 1 |
| L3 / L4 | `_group_alive` always false / always true | 2 / 2 |
| L5 | `child.poll()` removed from the poll loop | 2 |
| L6 | the loop-top pause check removed | 1 |
| L7 | the post-`next` `signalled` check removed | 1 |
| L9 (R29) | `while True` | **0**; 1 with probe B1 |
| L10 (`campaign.py`) | `status` root-refusal line removed | 1 |
| L11 | `ProcessLookupError` guard in `_signal_runner` removed | 1 |
| L13 / L14 | halt reason / usage-stop reason dropped from the ingest stop | 1 / **0** (P13-n1) |
| L15 | settings checked after `next` | 1 |
| L16 | refusal stamp in epoch seconds | 5 |
| L17 / L18 | hash `sort_keys=False` / default separators | 1 / 1 |
| L19 | SIGINT not handled | 1 |
| L20 | grace ignored | 2 |
| M25 | no exit on `IngestResult.stop` | 1 |

**Probes** (pytest functions added to the copy's `tests/` only, using `test_runner`'s `Bench`, fake `claude` and fake `HOME`; all pass on the live code):
- **A1:** SIGTERM delivered just before `issue` → exit 0, one `interrupted`; a second run ingests the batch.
- **A2:** SIGTERM after `next`, then a second run → the batch is ingested, and the first run's ledger bytes are a prefix of the final ledger.
- **B1:** a signal in the last ingest plus a settings drift → `stopped: signalled`, exit 0. Fails under L9.
- **B2:** a signal in the last ingest, one batch → `stopped: signalled`.
- **C1:** `pause --now` with the default 30 s grace → the runner returns in 0.09 s when the leader dies on SIGTERM.
- **D1:** a leader that never exits plus a stubborn descendant, 2 s timeout, 0.5 s grace → the descendant is dead and the run stops after the second incomplete.

**Growth-stops check:** Pass 13 gives one disposition line per Pass 12 finding and two new nits. Passes 1–12 are unchanged: `git diff` of this review shows one purely additive hunk at the end.

## Pass 14 — U6.3b user-settings record (2026-11-01)

**Verdict: approve with suggestions.** U6.3b implements plan6 v1.10 §3.2 and the §4 and §5 U6.3b rows. I found no blocker and no major, and the two findings are small.
- **P14-m1:** the first `permissions.allow` rule, or a `permissions` block that holds only allow rules, is refused when `install` saw no such block. That is a literal reading of §3.2's digest, and it contradicts §5's "not refused: a new `permissions.allow` entry" and the README's claim.
- **P14-n1:** two documentation points.
- **The plumbing is solid.** 25 mutants of my own all turn the suite red, including every write and read of `userSettingsDigests` (Appendix M).

### Scope

- **Reviewed:** the uncommitted U6.3b diff against `49636569`, which is `runner.py`, `campaign.py`, `tests/test_runner.py`, `README.md` and `docs/HISTORY.md`. Spec: plan6 v1.10 §3.2 "User settings", the §4 U6.3b row and the §5 U6.3b row. U5.7's `install` is out of scope.
- **Suite:** run on the live tree from `decision-models/` with a fresh `PYTHONPYCACHEPREFIX`: **894 passed, 2 xfailed**; `ruff check` clean.
- **UNRECORDED list:** I extracted the plan's list from §3.2 by script and compared it with the code's `UNRECORDED_SETTINGS`. Both have 35 keys: 5 named in the bullets (`model`, `switchModelsOnFlag`, `autoContinueAtUsageLimit`, `disableAllHooks`, `$schema`) and the 30 display and input keys. The difference is empty both ways.
- **Mutants and probes:** private copy `scratchpad/p14/repo/`, fresh `PYTHONPYCACHEPREFIX` per run, no `-I`. Baseline 889 passed + 5 skipped. Each file was restored and `cmp` reported equal. 25 mutants and 11 probe tests that I added to the copy only (Appendix M). My suggested fix was applied in a second copy, `repo2/`.
- **Live tree:** no git command that changes it; I wrote only this review. The reviewed files' sha256 values are in `scratchpad/p14/live-sha256.txt`.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; the changed hunks were read in full.

### Answers to the gate's questions

**(a) Conformance to §3.2: conforms, with one edge (P14-m1).**
- **Forbidden set.** `FORBIDDEN_SETTINGS` is exactly `env`, `hooks`, `fallbackModel` and `apiKeyHelper` (any value), plus `disableAllHooks` and `autoContinueAtUsageLimit` (only `true`). The coder's M-mutants and my N11, N18 and N19 each redden. The check is applied before the digest comparison (N10 reddens), and a forbidden key still refuses when it is also in the recorded digests (a test per key).
- **Value shapes (probe F2).** `autoContinueAtUsageLimit: 1` and `disableAllHooks: 1` refuse, because `1 == True` in Python. That is conservative and harmless. The strings `"true"` pass. They are not booleans, so I take them as outside the schema and report nothing.
- **UNRECORDED_SETTINGS:** exact against the plan, as stated in Scope. The test pins it against a hand-copied literal; I checked that literal against the plan too.
- **Permissions digest.** `allow` and `defaultMode` are left out (N12 and N13 each give 2 red), and `deny`, `ask` and `additionalDirectories` stay in (the test per class). One edge is P14-m1.
- **No values in a refusal.** A marker test covers the digest refusal (N16 reddens) and a second covers the forbidden refusal (N15 reddens). The forbidden message prints the refusing constant, `: true`, never the user's value.

**(b) The forbidden refusal ends "remove it": the coder is right, and the plan's wording is the loose part.**
- §3.2 places the recovery sentence under "*Accepting a change*", and says re-running `install` "repeats its checks". The same section says `install` refuses to record a forbidden key. A forbidden refusal that said "run `driver install` to accept it" would send the person to a command that refuses them, the same message again.
- The §4 row ties the sentence to the digest comparison ("then names the added, removed and changed keys and ends with §3.2's recovery sentence"), and the §5 row's recovery bullet sits with the recorded-key cases. So the plan's text requires the sentence on digest refusals only, and the code does exactly that (N17 reddens it, 10 red).
- **Wording to tighten in the plan,** which is P14-n1: "A refusal's text ends…" should read "A recorded-key refusal's text ends…".
- **A smaller point.** For `autoContinueAtUsageLimit` and `disableAllHooks`, "remove it" is correct, but "or set it to false" is also accepted. That is optional text.

**(c) Fixtures, vacuity and mutant adequacy: adequate.**
- Every plumbing argument has a mutant that turns the suite red:
  - `record_expectations` writing `{}` (N1) gives 34 red, and writing reversed values (N2) gives 32;
  - writing the retired key name (N3) gives 60;
  - `run` reading a misspelt key (N4) gives 60, and ignoring the record (N5) gives 34;
  - hashing the key's name instead of its value (N22) gives 8, and the unsorted and default-separator variants (N23, N24) give 2 and 1.
- **Set and refusal plumbing:** each of N6 through N9, N20 and N25 is red, and N21 (only a fixed key list recorded, so `futureKey` launches) gives 2.
- **No test is vacuous.** The expected digests are independently computed hand-written canonical strings: `b'{"deny":["Bash"]}'` and `'{"a":"é","effort":"high"}'`, with `é` pinning `ensure_ascii=False` (N14 reddens). `Bench` records `runner.user_settings_digests(recorded)` and the run side recomputes it, so the pair agrees only if both sides are right; the independent canonical-string tests close that loop.

**(d) Ordinary user actions: mostly right; refusals show and leave the ledger byte-equal.**
- Not refused, and tested: `theme`, `model`, `verbose`, `$schema`, both `switchModelsOnFlag` values, `autoContinueAtUsageLimit: false`, `disableAllHooks: false`, a new `allow` entry on an existing `permissions` block, a `defaultMode` change, and reordered or re-indented files.
- **The gap:** the first allow rule when `install` saw no `permissions` block (P14-m1).
- **`status` and the ledger.** The shared `refused()` helper asserts, for every refusal, that nothing launched, the ledger is byte-equal, `store/refusal` begins with a timestamp, and `status` prints the reason. Probe F4 shows the printed line for a forbidden key.

**(e) Interaction with U6.3: clean.** Verified by probe.
- **Pause before the settings check.** E1: a paused campaign with an `env` key exits 0 with `stopped: paused`, and no `store/refusal` is written. The pause check still precedes the forbidden check.
- **The install-record requirement.** E2: a record with a template but no digests refuses with "no driver template or user-settings digests (run `driver install`)", before the settings are read, exit 1. E3: an old `userSettingsSha256` record does not satisfy it. F1: a malformed digest value reports a change, not a crash. N4 and N5 redden the same path.
- **The settings are still read before `next`.** The coder's L15 mutant from Pass 13 stays red in the suite.

### Findings

**P14-m1 — minor — a `permissions` block that holds only `allow` rules or a `defaultMode` is not the same as no block, so the first allow rule refuses.** *Owner: `architect` (§3.2 states the digest, §5 states the case), then `coder`. Verified by probes D1–D4 and by a tested fix.*
- `user_settings_digests` strips `allow` and `defaultMode` and then digests what is left, so `permissions: {"allow": ["Read"]}` digests as `{}` and still produces a `permissions` entry. When `install` saw no `permissions` key, the first allow rule the user adds makes the key appear, and the refusal reads `added: permissions`. I ran it:
  - D1: first allow rule added → exit 1, "added: `permissions`";
  - D2: a `defaultMode` alone → exit 1;
  - D3: the last allow rule and its block removed → exit 1 ("removed");
  - D4: the digests of `{permissions: {allow: [x]}}` and of `{}` differ.
- **Why it matters.** §5 says "Not refused: a new `permissions.allow` entry", §3.2 says an allow rule "cannot loosen a deny, an ask or the guard", and the README repeats that a new allow entry does not refuse. The unattended campaign would stall on the user's first "always allow" in the user file, which is one of the ordinary edits the stakeholder asked not to stop it. The tests start from `SETTINGS`, whose `permissions` already holds an allow rule, so they never reach this case.
- **Fix** (serves §5's "not refused" row), tested in `repo2/`: in `user_settings_digests`, after stripping, `continue` when the `permissions` value is empty, so an allow-only or `defaultMode`-only block digests as absent. D1–D4 then pass, and the full suite stays green except two tests whose fixtures assume the block is recorded:
  - `test_the_digests_cover_each_recorded_key_and_only_those` expects `permissions` in the digests of `SETTINGS`;
  - `test_a_refusal_names_added_removed_and_changed_keys_and_shows_no_value` expects `changed: permissions`.
  - Both need a `deny` rule in the recorded `SETTINGS`, or a changed expectation.
- **The plan decides the rule,** because `permissions: {}` versus absent is a design choice. Whoever takes it adds D1 and D3 as tests and makes the README sentence true. `install` calls the same function, so the fix lands there with no further change.

**P14-n1 — nit — two documentation points.** *Owner: `architect` (the first), `coder` (the second). Verified by reading.*
- **Plan wording:** §3.2's "A refusal's text ends 'revert it, or run `driver install` to accept it'" applies to recorded-key refusals only (question (b)). Saying so removes the ambiguity that the coder's deviation exposed.
- **`docs/HISTORY.md`'s U6.3b entry lists the unit's work and mutants, but not the call it made:** forbidden refusals end "remove it", where the plan's sentence would mislead. A one-line "Calls made" item records it.

### Appendix M — Pass 14 mutants and probes

Files are in `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/p14/`: `mutants.py`, `mutate.py`, `probes/test_p14_probes.py`, `logs/mutants-run.txt`, `logs/probes.txt`. Baseline in the copy: 889 passed + 5 skipped, 2 xfailed. Each mutant ran the full suite; each file was restored and `cmp` reported equal.

| ID | Mutation | Red |
|---|---|---|
| N1 / N2 | `record_expectations` writes `{}` / digest values reversed | 34 / 32 |
| N3 / N4 | key written under the old name / `run` reads a misspelt key | 60 / 60 |
| N5 | `run` compares against `{}` | 34 |
| N6 | added and removed swapped | 2 |
| N7 / N8 / N9 | removed / changed / added keys ignored | 1 / 4 / 5 |
| N10 | digest check ahead of the forbidden check | 1 |
| N11 | `disableAllHooks` refuses on any value | 2 |
| N12 / N13 | `defaultMode` / `allow` digested | 2 / 2 |
| N14 | `ensure_ascii=True` | 1 |
| N15 / N16 | a value shown in the forbidden / digest refusal | 1 / 1 |
| N17 | recovery sentence reworded | 10 |
| N18 / N19 | `apiKeyHelper` / `fallbackModel` not forbidden | 2 / 2 |
| N20 | `outputStyle` unrecorded | 4 |
| N21 | only a fixed key list recorded | 2 |
| N22 | digest over the key's name | 8 |
| N23 / N24 | `sort_keys=False` / default separators | 2 / 1 |
| N25 | `$schema` recorded | 2 |

**Probes** (pytest functions added to the copy's `tests/` only, using `test_runner`'s `Bench`, fake `claude` and fake `HOME`):
- **D1:** install saw no `permissions` block; the first allow rule is added → exit 1, "added: `permissions`". Fails against the plan's "not refused" row.
- **D2:** a `defaultMode`-only block is added → exit 1. Fails.
- **D3:** the last allow rule and its block are removed → exit 1. Fails.
- **D4:** the digests of `{permissions: {allow: [x]}}` and `{}` differ. Fails.
- **E1:** a paused campaign with an `env` key → exit 0, `stopped: paused`, no `store/refusal`.
- **E2:** a record with a template and no digests → exit 1, "no driver template or user-settings digests".
- **E3:** a record carrying only `userSettingsSha256` → the same refusal.
- **F1:** a non-string digest value in the record → exit 1, "changed".
- **F2:** forbidden value shapes (`1`, `"true"`, `None`, `[]`, `""`), observed as described in (a).
- **F3:** a non-dict `permissions` digests whole, without a crash.
- **F4:** `status` prints the forbidden refusal.

D1–D4 are the only probes that fail on the live code. In `repo2/` (the fix in P14-m1) the suite plus all 11 probes gives 898 passed, with exactly the two fixture-dependent tests above failing.

**Growth-stops check:** Pass 14 adds one new finding and one nit, and re-opens no earlier finding. Passes 1–13 are unchanged: the pre-pass copy and the first part of the live file are `cmp`-identical.

## Pass 15 — U5.7 agent definitions, guard and install (2026-11-01)

**Verdict: approve with suggestions.** U5.7 delivers the three definitions, the guard, the settings wiring and `driver install` as plan6 v1.11 §3.3 and the U5.7 rows describe them. I found no blocker and no major.
- **P15-m1 (minor):** re-running `install`, the sanctioned way to accept a refused settings edit, resets a `resolve --repin-model` or `--repin-instructions` decision to the definition's values, because the ledger fold lets each `expectations` event overwrite the pins.
- **P15-n1 and P15-n2 (nits):** a missing `effort` in an `--agents-dir` copy gives a traceback after files are written, and `install` validates no frontmatter field but `hooks`; and the guard's template map is untested because the two role templates are byte-identical.
- **Mutants:** 32 of mine, 30 red. The two survivors are the one the coder reported (equivalent in effect) and P15-n2.
- **For the cobb consult:** Answer (e) lists which frontmatter fields no probe has exercised.

### Scope

- **Reviewed:** the uncommitted U5.7 diff against `bfb34dc1`. New: `agents/` (the three definitions, three spawn templates, `settings.json`, `guard_hook.py`), `decisionmodels/install.py`, `decisionmodels/smoke.py`, `tests/test_install.py` and `tests/test_guard_hook.py`. Edited: `campaign.py` (`record_expectations`), `driver.py`, `README.md`, `AGENTS.md` and `docs/HISTORY.md`.
- **Specs, the later one winning:** plan6 v1.11 §3.2 "User settings", §3.3 and the U5.7 rows of §4 and §5; plan5 v1.5; plan4 v1.3 §3.4, §3.5, §3.7 and the U5.7 rows. For the Claude Code behaviour, I read the subagents and hooks documentation (`code.claude.com/docs/en/sub-agents` and `/hooks`) and the U6.1, U6.2 and U6.7 probe report.
- **Suite:** run on the live tree from `decision-models/` with a fresh `PYTHONPYCACHEPREFIX`: **1085 passed, 2 xfailed**; `ruff check` clean.
- **Mutants and probes:** private copy `scratchpad/p15/repo/`, fresh `PYTHONPYCACHEPREFIX` per run, no `-I`. Baseline 1080 passed + 5 skipped. Each file was restored and `cmp` reported equal. 32 mutants and 7 probe tests, which I added to the copy only (Appendix N). My two suggested fixes were applied in a second copy, `repo2/`.
- **Live tree:** no git command that changes it; I wrote only this review. The reviewed files' sha256 values are in `scratchpad/p15/live-sha256.txt`. Out of scope, as briefed: the live smoke and anything that calls the real `claude`.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; the new files were read in full.

### Answers to the gate's questions

**(a) Conformance: conforms.** Checked by reading each file against the rows, and by the mutants in (c).
- **Frontmatter against plan4 §3.4's table as plan6 §3.3 amends it:**
  - `dm-driver`: `claude-sonnet-5-5`, `tools: Agent(dm-generator, dm-labeler)`, `maxTurns: 3`, no `initialPrompt`, no `hooks`, no `Bash`.
  - `dm-generator`: `claude-opus-5-5`, effort `medium`, `Read, Write`, `maxTurns: 6`, `omitClaudeMd`, `permissionMode: acceptEdits`.
  - `dm-labeler`: `claude-sonnet-5-5`, effort `high`, `maxTurns: 14`, same flags.
  - `P_max + 4` gives 6 and 14, as the plan says.
- **Driver template and hash:** `spawn-driver.txt` has the `{agent_type}` and `{spawn_prompt}` slots and the `PROMPT:` line that the `dm-driver` body names. `promptSha256.spawnTemplate.driver` is recorded beside `generator` and `labeler`. The templates carry no trailing newline, and install refuses one that does (I11 and the template test).
- **Guard rules, each verified by a mutant or a test:**
  - **No `Bash`:** the driver may call only `Agent`.
  - **Canonical paths:** `realpath` of the joined path, strictly inside `realpath(exchange/req)` or `realpath(exchange/ans)`. G3 (sibling prefix) gives 4 red, G4 (no `realpath`) 4, and G16 (the guard's own `cwd`) 3.
  - **Fail closed:** G9 (exit 0 on a failure) gives 35 red.
  - **`onFailure: "block"`** is required by install (I15 gives 4 red).
  - **The bounds derive from the install location**, `realpath(__file__)`, never from the input (G1 gives 64 red).
- **Install steps, in order:** the lock; the `runner.forbidden_settings_refusal` check, which is the same function `run` uses (I12 gives 8 red); `runner.user_settings_digests` (I6 gives 2); the source checks; the installed-versus-source sha256 compare (I11 gives 9); the fresh `expectations` event. The output names keys and never a value, and the project pin `autoContinueAtUsageLimit: false` is required before anything is copied (I14 gives 1).
- **The "refusals precede every write" claim has one exception,** P15-n1.

**(b) The coder's deviations: all five stand.**
1. **Exit 1 with no stdout when the guard cannot judge: plan6 requires it, and the brief was stale.** Plan4 §3.4 said "always exits 0"; plan6 §3.3 amends that to exit 1 plus `onFailure: "block"`, and plan6 is later. The hooks documentation agrees: exit 1 without valid JSON is a non-blocking error, so the action goes ahead unless `onFailure` is `"block"`. That is why install refuses a wiring without it.
2. **The Agent key whitelist (`subagent_type`, `prompt`, `description`) is stricter than plan4's four named keys, and conflicts with nothing.** Every Agent input the probe report lists for the driver has exactly those three keys, plus `run_in_background` where a driver on Opus backgrounded. A real shape cannot halt unless a future CLI adds a key the model did not send. A deny there halts loudly and cheaply, and the smoke should record the hook's actual `tool_input` key set.
3. **`SubagentHandback` allowed for the two subagents, with no condition: acceptable.** Plan6 makes the allow conditional on the smoke showing a hook sees it. Allowing a path-free tool that the subagents' `tools: Read, Write` list does not grant is harmless, and G13 (also allowed for the driver) gives 2 red. The hooks documentation does not mention the tool, so the smoke settles whether the line is ever used.
4. **The guard command `python3 "$CLAUDE_PROJECT_DIR/.claude/guard_hook.py"`, recorded unexpanded:** plausible, unverified. The hooks documentation says a shell-form command is expanded by the shell, which would leave `hook_success.command` as the configured string, equal to `guardCommand`. No probe used a variable: every probe wired an absolute path. If the recorded string differs, the first smoke batch halts as `hookMissing`, which is loud and cheap. The fallback is install writing the absolute path into the installed `settings.json`, at the cost of the byte-copy. `python3` must also resolve on the child's allowlisted `PATH`, which is a U6.4 point.
5. **`settingsSha256` beside `promptSha256`: acceptable.** Nothing reads it, and I8 (1 red) pins it. It sits outside `promptSha256`, so U5.10's "matches §3.7's keys exactly" is unaffected, provided U5.10 does not copy it into that object.

**(c) Mutants: adequate.** 32 mutants of mine, 30 red (Appendix N).
- **Plumbing, the install side.** Wrong model, effort, spawn template, driver template, guard command, digests, definition hashes, template hashes and settings hash each redden 1 to 12 tests. An unread earlier-hash check, a skipped installed compare, a skipped forbidden check and a skipped `onFailure` check do too.
- **Plumbing, the guard side.** The Read and Write folders swapped (7 red), the role map swapped (5), `startswith` without the separator (4), no `realpath` (4), and the guard's directory one level off (64) are each caught.
- **G8, dropping the `subagent_type in SUBAGENTS` check: I agree it is equivalent in effect.** With the check gone, an unknown role and a missing `subagent_type` end in a `KeyError`, exit 1 and a block (probe A4), where the check gives a deny. Both block and both halt; the U6.7 probe found that a deny and a blocked failure are structurally identical in the transcript.
- **G6 is a survivor of mine (P15-n2).** Swapping the guard's template-file map is invisible because `spawn-generator.txt` and `spawn-labeler.txt` are byte-identical (`cmp` says so).

**(d) Refusals and the interaction with U6.3 and U6.3b: sound, with one interaction (P15-m1).**
- Each forbidden key refuses `install` with the ledger byte-equal. Safe values install. A `theme` or `model` edit changes no digest. Re-running after a refused `outputStyle` edit appends a second event and the next `run` launches (the test per row).
- The ordering holds: lock first, settings and sources next, writes last. Probe A1 is the one refusal that arrives after files exist (P15-n1).
- `run` still checks pause before settings, and still requires the install record. `userSettingsDigests` round-trips through `runner.user_settings_digests` (I6 reddens).
- **The interaction is P15-m1.**

**(e) Version-sensitive fields: which no probe has exercised.** From the probe report (`claude` 2.1.316) and the documentation.
- **Probed, with the same field and value:**
  - `model: claude-sonnet-5-5` on a driver and a subagent;
  - `tools: Agent(<type>)` on the driver;
  - `maxTurns: 3` on the driver;
  - `omitClaudeMd: true` and `permissionMode: acceptEdits` on a subagent (the documentation says `omitClaudeMd` needs 2.1.271 or later);
  - `effort: high` on a subagent;
  - `onFailure: "block"` with a guard exit of 1 (D2). The documentation says it needs 2.1.295 or later.
- **Not probed:**
  - `model: claude-opus-5-5` and `effort: medium` as a **subagent** pin;
  - `maxTurns: 6` and `14`, and the cut-short shape (the smoke's `maxTurns: 2` row);
  - `permissionMode` when another agent spawns the subagent: the documentation ties it to the parent's mode, and the guard is the bound under `-p`;
  - `onFailure` for a guard **timeout** or **cannot-start** (the probe report calls these inferred);
  - the `$CLAUDE_PROJECT_DIR` expansion in the shell-form `command`, and the recorded `hook_success.command`;
  - whether a hook ever sees `SubagentHandback`;
  - the hook's `timeout: 30` field;
  - the project-level `autoContinueAtUsageLimit: false`: the settings reference gives that key "User or managed" scope, so a project value is probably ignored, which plan6 says costs nothing;
  - `python3` on the child's allowlisted `PATH`, which the hook command needs.

### Findings

**P15-m1 — minor — re-running `install` resets a repinned model or instruction set.** *Owner: `architect` (plan6 §3.2's "accept by re-running `install`" says nothing about the pins), then `coder`. Verified by probe A2 and by a tested fix.*
- `fold` takes `state.pins[role]` and `state.instructions[role]` from **every** `expectations` event (`campaign.py` `fold`), and `install` writes each event's `model` from the definition and `instructions: []`. A `resolve --repin-model` or `--repin-instructions` therefore lasts only until the next `install`.
- **Probe A2:** the ledger gets a `labelling-restart` repin to `claude-haiku-9` and a `repin-instructions` set, then `install` runs again. Before: `claude-haiku-9` and the planted pair. After: `claude-sonnet-5-5` and `[]`.
- **Why it matters.** The re-run is the only way to accept a user-settings edit that `run` refused. After any earlier repin, the next batch audits against the old pin or the empty set and halts again, and a human re-resolves. It is loud and not silent, but it costs a halt, and `resolve` records a decision that the next `install` undoes without telling anyone.
- **Fix** (serves plan6 §3.2 "Accepting a change"), tested in `repo2/`: in `fold`, `setdefault` the pins and instructions from an `expectations` event, so only the first event sets them. That loses nothing, because a later install of the same sources writes the same pins. The suite stays green (1087 passed with probes A1 and A2), and A2 passes. Add A2 as a test, and state the rule in §3.2.

**P15-n1 — nit — a definition without `model` or `effort` fails after files are written, and `install` validates no field but `hooks`.** *Owner: `coder`. Verified by probes A1 and A3.*
- A1: an `--agents-dir` copy of `dm-labeler.md` without `effort:` gives `KeyError: 'effort'` from `install()`, a traceback, and all seven files already copied under `exchange/.claude/`. The ledger is unchanged. The README and the `install.py` docstring say that every refusal comes before any write.
- A3: a copy of `dm-driver.md` with `Bash`, `claude-opus-5-5` and `maxTurns: 99` installs. The §5 U5.7 pins (`claude-sonnet-5-5`, no `Bash`, `maxTurns: 3`) are asserted on the repository's own sources only.
- **Fix,** tested in `repo2/` for the first half: in `_read_sources`, refuse a generator or labeler definition that lacks `model` or `effort`, before any copy. The suite stays green. Optionally refuse a driver definition whose `tools` contains `Bash`.

**P15-n2 — nit — the guard's template map is untested.** *Owner: `coder`. Verified by mutant G6 (0 red) and `cmp`.*
- Swapping `TEMPLATE_FILES` in the guard changes nothing the suite can see, because `spawn-generator.txt` and `spawn-labeler.txt` are byte-identical. If the two ever diverge, the guard would deny every prompt of one role, or allow the other role's.
- **Fix:** one test in `tests/test_guard_hook.py` that installs from an `--agents-dir` copy whose generator template differs in one word, then asserts that the guard allows the generation request's own prompt and denies the labeler template's.

### Appendix N — Pass 15 mutants and probes

Files are in `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/p15/`: `mutants.py`, `mutate.py`, `probes/test_p15_probes.py`, `logs/mutants-run.txt`, `logs/probes.txt`. Baseline in the copy: 1080 passed + 5 skipped, 2 xfailed. Each mutant ran the full suite; each file was restored and `cmp` reported equal.

| ID | Mutation (`install.py` I, `guard_hook.py` G) | Red |
|---|---|---|
| I1 / I2 | generator model recorded as the labeler's literal / effort a constant | 1 / 1 |
| I3 | role spawn template is the driver template | 12 |
| I4 | guard command recorded as a literal | 3 |
| I5 | driver template is the labeler template | 2 |
| I6 | user settings digests recorded empty | 2 |
| I7 | definition hashes assigned to the wrong roles | 2 |
| I8 | `settingsSha256` over the guard source | 1 |
| I9 | declared `instructions` not empty | 7 |
| I10 / I11 | earlier-hash check / installed-versus-source compare removed | 1 / 9 |
| I12 | forbidden user settings not refused | 8 |
| I13 | `spawnTemplate` hashes all the generator's | 2 |
| I14 / I15 | autoContinue pin / `onFailure: "block"` not required | 1 / 4 |
| I16 | the recorded command not read from the installed copy | 1 |
| G1 | the guard's exchange directory one level off | 64 |
| G2 | Read and Write folders swapped | 7 |
| G3 / G4 | sibling prefix not excluded / no `realpath` | 4 / 4 |
| G5 | the role-to-agent map swapped | 5 |
| G6 | the template-file map swapped | **0** (P15-n2) |
| G7 | `model` key allowed on `Agent` | 1 |
| G8 | `subagent_type` membership check dropped | **0** (equivalent, answer (c)) |
| G9 | exit 0 on a failure | 35 |
| G10 | event name not checked | 1 |
| G11 | an unknown `agent_type` denied, not failed | 5 |
| G12 | prompt compared by prefix | 1 |
| G13 | hand-back allowed for the driver | 2 |
| G14 | the request's role not compared | 3 |
| G15 | the pending request need not be unique | 1 |
| G16 | a relative path joined to the guard's own `cwd` | 3 |

**Probes** (pytest functions added to the copy's `tests/` only; the guard probes use `test_guard_hook`'s `tree` fixture):
- **A1:** an `--agents-dir` labeler without `effort` → `KeyError: 'effort'`, seven files written, ledger unchanged. Fails the "refuses before any write" assertion.
- **A2:** a repin of the labeler's model and instructions, then `install` → pins reset to `claude-sonnet-5-5` and `[]`. Fails against the repin's lasting.
- **A3:** a driver copy with `Bash`, Opus and `maxTurns: 99` → installs.
- **A4:** an unknown role in the request header and no `subagent_type` → a deny, or with G8 an exit 1 and `KeyError: None`.
- **A5:** an arbitrary `description` string is allowed.
- **A6:** the three definition hashes are distinct per role.
- **A7:** the hook log's keys are `agent_type`, `at`, `decision`, `event`, `tool_name`, `tool_use_id`, with no path.

**Fixes tried in `repo2/`:** `fold` takes pins and instructions with `setdefault`; `_read_sources` refuses a generator or labeler definition without `model` and `effort`. Full suite plus probes: 1087 passed + 5 skipped, 2 xfailed, so A1 and A2 turn green and no existing test pins the old behaviour.

**Growth-stops check:** Pass 15 adds one minor and two nits and re-opens no earlier finding. Passes 1–14 are unchanged: the pre-pass copy and the first part of the live file are `cmp`-identical.

## Pass 16 — U5.7 fix round: version floor and definition check (2026-11-01)

**Verdict: approve.** The Pass 15 findings are fixed and re-run red against the mutants that undo them. The version floor and the definition check conform to plan6 v1.13 §3.3 as written, and nothing in the round's own interactions is defective. There are two nits.
- **P16-n1:** the stderr half of the version probe and its timeout path are untested.
- **P16-n2:** the definition check does not reject a duplicated frontmatter key, and does not pin a definition's `name`.
- **Mutants:** 27 of mine against this round, plus a re-run of Pass 15's G6. 25 redden on the suite as it stands, and the probes kill the other two.

### Scope

- **Reviewed:** the uncommitted U5.7 tree after the fix round, against my Pass 15 reading and, for committed code, `bfb34dc1`. In detail: `runner.py` (`MIN_CLAUDE_VERSION`, `claude_version_refusal`, its call in `Runner._loop`), `install.py` (`definition_error`, the version call, `_read_sources`), `campaign.py`'s `fold`, `tests/conftest.py`, and the diffs of `test_runner.py`, `test_install.py` and `test_guard_hook.py`. Spec: plan6 v1.13 §3.3 "Definition check" and "Version floor", and the §4 and §5 U5.7 rows. For context I read cobb's consult, `decision-models/docs/reviews/typed-decision-models-frontmatter.md`.
- **Suite:** run on the live tree from `decision-models/` with a fresh `PYTHONPYCACHEPREFIX`: **1138 passed, 2 xfailed**; `ruff check` clean. I also ran it with `PATH=/usr/bin:/bin`, so that the real `claude` (at `~/.local/bin/claude`) is unreachable: the same 1138 pass, so no test depends on the real CLI.
- **Mutants and probes:** private copy `scratchpad/p16/repo/`, fresh `PYTHONPYCACHEPREFIX` per run, no `-I`. Baseline 1133 passed + 5 skipped. Each file was restored and `cmp` reported equal. 27 mutants (and a re-run of G6) and 13 probe tests, which I added to the copy only (Appendix O).
- **Live tree:** no git command that changes it; I wrote only this review. The reviewed files' sha256 values are in `scratchpad/p16/live-sha256.txt`. Out of scope, as briefed: the live smoke.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; the changed hunks were read in full.

### Disposition of Pass 15

- **P15-m1** (re-install resets a repin): fixed. `fold` uses `setdefault` for the pins and the instructions, so only the first `expectations` event sets them. F1 and F2, which restore the overwrite, give 1 red each; F3 (pins never set) gives 126. The new test drives a real `resolve --repin-model` and `--repin-instructions` through a halt, then `install`, and compares with an independently written pair and model. My own probe A2, reduced to the model, also passes.
- **P15-n1** (a definition missing `effort` fails after files are written): fixed. The same input refuses with the ledger unchanged and no `exchange/.claude` directory (my A1 re-run, and the 22 planted-definition cases, each of which asserts the directory does not exist). The driver-with-`Bash` copy of my A3 is refused as well.
- **P15-n2** (the guard's template map): fixed. `test_each_role_is_checked_against_its_own_template` installs a copy whose two templates differ, and checks each role's own prompt allowed and the other's denied. Mutant G6, which survived Pass 15, gives 1 red.

### Answers to the gate's questions

**(a) Each Pass 15 finding is fixed:** see the disposition above. All three were re-run, not read.

**(b) Conformance to §3.3: conforms.**
- **Definition check, key sets and pins.** `definition_error` matches the plan's text: subagents' keys `name`, `description`, `model`, `effort`, `tools`, `maxTurns`, `omitClaudeMd`, `permissionMode`; the driver's `name`, `description`, `model`, `tools`, `maxTurns`. The pinned values are `tools: Read, Write`, `omitClaudeMd: true`, `permissionMode: acceptEdits`, an integer `maxTurns`, and non-empty `model` and `effort` for the subagents; for the driver `model: claude-sonnet-5-5`, `maxTurns: 3` and `tools: Agent(dm-generator, dm-labeler)`, so `Bash` cannot appear. `hooks` is refused by name in all three. Each of D1 to D9 (unknown key, missing key, driver pins, integer check, `hooks`, driver `effort`, the `tools` and `omitClaudeMd` pins, empty value) reddens the suite, and D10 (the check skipped for the driver) gives 3 red.
- **Floor-only.** `claude_version_refusal` compares an integer tuple with `(2, 1, 295)` and keeps no recorded version. W2 (`<=`) gives 2 red, W3 (a string compare) 3, W6 and W7 (the floor one patch off) 6 and 7. A version change between two runs above the floor changes nothing (its test).
- **The child's allowlisted environment.** The probe runs under `child_env(environ)`. W1, which passes the full environment, gives 1 red, and my probe V1 (a fake `claude --version` that exits 3 when `ANTHROPIC_API_KEY` or `CLAUDE_CODE_EFFORT_LEVEL` is present, and 4 without the background-task variable) passes on the live code.
- **Parsing.** The first `x.y.z` in stdout plus stderr, as the plan says: output with none refuses, and so does a non-zero exit with a version (W4 gives 1 red). Probe V3 shows that a banner carrying an earlier `x.y.z` (`node 20.1.0` before the CLI's own) would decide; that is what "first" means in the plan, and the real CLI prints one line.
- **The refusal path.** A version refusal on `run` exits 1 with `refused: launch refused: …`, writes `store/refusal`, shows in `status` and leaves the ledger byte-equal (its test, under `_refusing`, like the settings refusal).
- **Install refusals precede any write.** The version check is the first step under the lock, and the definition check runs in `_read_sources`, before `_check_installed` and before any copy. Probe V2 (2.1.294 at install) leaves the ledger unchanged and no `exchange/.claude`. The old `KeyError` case refuses (Pass 15's disposition).

**(c) Interaction within the round: sound.**
- **Order.** `Runner._loop` checks pause, then the version, then the install record and the settings. W8 (version before pause) and W9 (version after settings) each give 1 red, and the two order tests state the plan's order. The version refusal precedes the settings refusal in the text a person sees (the test asserts "forbidden" is absent).
- **A missing `claude` binary.** It is now a refusal (exit 1, nothing issued, ledger byte-equal), where U6.3 made it a `launchFailed`. That follows §5's row ("a missing binary refuse[s]") and plan6 §3.3. The U6.3 launch-cannot-start path is reachable only if the binary vanishes between the version check and the launch.
- **The changed U6.3 test: intent preserved.** The old test's intent was that an unstartable launch is a `launchFailed` outcome (`exitCode: None`, "the launch failed" in the evidence), and that the run stops after one launch instead of looping. The new test `test_a_launch_that_cannot_start_is_a_launch_failure_not_a_crash` keeps every assertion (code 0, no calls, one outcome with that reason and evidence) and reaches the path by making `Popen` raise only for the `-p` launch. Its comment says why. The old test's PATH-restoring timer, which guarded against a run that kept launching, is gone, but the single-outcome assertion covers the same loop. A second new test covers the missing-binary refusal.
- **`fake_claude` is not vacuous.** It is applied module-wide to `test_install.py` and `test_guard_hook.py` by `pytestmark`. Its `--version` text and exit code are set per test, and it can remove the binary. The suite passes with the real CLI off `PATH`, and the version tests reach both refusals and passes through it, with W11 (install never checks the version) giving 9 red.

**(d) Mutants: adequate.**
- **Plumbing I mutated:** the version probe's environment (W1), what `definition_error` reads against what `install` copies (D10), the fold's pin source (F1, F2, F3), and what `record_expectations` writes (F4, 5 red).
- **Survivors of the suite, both closed by a probe:**
  - W5 (stderr not searched) and W12 (a `TimeoutExpired` from a hung `claude --version` not caught) leave the suite green, and probes V6 and V5 kill them (P16-n1).
- **The first run of D9 was a malformed mutant** (a syntax error, which my runner reported as 0 failures). I re-ran it correctly: 1 red. All figures in this section come from that corrected run.

**(e) Pass 15's survivors.** G6 now gives 1 red. I did not re-run G8, which Pass 15 showed equivalent in effect, and the guard is unchanged.

### Findings

**P16-n1 — nit — two branches of `claude_version_refusal` are unpinned.** *Owner: `coder`. Verified by mutants W5 and W12 (0 red each) and by probes that kill them.*
- **W5:** dropping `done.stderr` from the search leaves the suite green. A CLI that printed its version on stderr would then refuse every launch. Probe V6 plants it.
- **W12:** narrowing the `except` to `OSError` leaves the suite green. A `claude --version` that hangs past 30 s would then raise `TimeoutExpired` out of `run` and `install` as a traceback, not a refusal. Probe V5 plants it (a `sleep 100` fake, with the timeout cut to 1 s).
- **Fix** (serves the §5 row's "output with no `x.y.z` … refuse"): add V5 and V6 as tests, each with the expected refusal text written in the test.

**P16-n2 — nit — `definition_error` rejects neither a duplicated frontmatter key nor a wrong `name`.** *Owner: `architect` (§3.3 lists keys and pins, not these), then `coder`. Verified by probes D1 and D2.*
- **D1:** a labeler with `maxTurns: 14` and a second `maxTurns: 2` installs, because `frontmatter()` keeps the last value. Which value Claude Code keeps is unknown to me, and the hash records the file either way.
- **D2:** a labeler named `dm-labelerx` installs. `agent_type` is the agent's `name`, and the guard and the audit key on the constants `dm-generator` and `dm-labeler`. The result would be an unknown `subagent_type` and a halted first batch, loud and cheap.
- Both reach only an `--agents-dir` copy, since the shipped files pass. If the plan wants them closed, the rule is one line in §3.3: "a key appears once, and `name` equals the role's agent type". Quoted pin values are already refused (probe D3).

### Appendix O — Pass 16 mutants and probes

Files are in `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/p16/`: `mutants.py`, `mutate.py`, `probes/test_p16_probes.py`, `logs/mutants-run.txt`, `logs/probes.txt`. Baseline in the copy: 1133 passed + 5 skipped, 2 xfailed. Each mutant ran the full suite; each file was restored and `cmp` reported equal.

| ID | Mutation | Red |
|---|---|---|
| W1 | version probe under the full environment | 1 |
| W2 / W3 | floor `<=` / compared as a string | 2 / 3 |
| W4 | a non-zero exit with a version accepted | 1 |
| W5 | stderr not searched | **0**; 1 with probe V6 |
| W6 / W7 | floor 2.1.294 / 2.1.296 | 6 / 7 |
| W8 / W9 | version check before the pause check / after the settings check | 1 / 1 |
| W10 / W11 | `run` / `install` never checks the version | 6 / 9 |
| W12 | `TimeoutExpired` not caught | **0**; 1 with probe V5 |
| W13 | `run` checks the version once, before the loop | 6 |
| D1 / D2 | unknown / missing frontmatter keys allowed | 3 / 2 |
| D3 / D6 | driver held to the subagent pins / allowed an `effort` key | 60 / 60 |
| D4 / D5 | the `maxTurns` integer check / the `hooks` check removed | 1 / 1 |
| D7 / D8 | the driver `tools` pin / the `omitClaudeMd` pin dropped | 1 / 1 |
| D9 | empty `model` or `effort` allowed (re-run after a malformed first try) | 1 |
| D10 | the definition check skipped for the driver | 3 |
| F1 / F2 | `fold` overwrites the pins / the instructions | 1 / 1 |
| F3 | `fold` never sets the pins | 126 |
| F4 | `record_expectations` drops `promptSha256` | 5 |
| G6 | the guard's template map swapped (Pass 15's survivor) | 1 |

**Probes** (pytest functions added to the copy's `tests/` only, under the `fake_claude` fixture):
- **Pass 15 re-runs:** a missing `effort` refuses before any write; a re-install keeps a repinned model; a driver with `Bash` is refused; the first install still sets the pins from the definitions.
- **V1:** the version probe's environment is the allowlist: a fake that exits non-zero if `ANTHROPIC_API_KEY` or `CLAUDE_CODE_EFFORT_LEVEL` is present, or if the background-task variable is absent. Passes.
- **V2:** 2.1.294 at install → refused, ledger unchanged, no `exchange/.claude`.
- **V3:** a banner `node 20.1.0` before `2.1.316` → accepted (the first `x.y.z` decides).
- **V4:** `2.1.295-beta.1` → accepted.
- **V5:** a hung `claude --version` (timeout cut to 1 s) → a refusal naming `TimeoutExpired`.
- **V6:** a version printed on stderr only → accepted.
- **C1:** a `run` refused at 2.1.294 writes `store/refusal`; at 2.1.296 it launches and the file is gone.
- **D1:** two `maxTurns` lines → installs. **D2:** `name: dm-labelerx` → installs. **D3:** `omitClaudeMd: "true"` → refused.

**Growth-stops check:** Pass 16 adds two nits and re-opens no earlier finding. Passes 1–15 are unchanged: the pre-pass copy and the first part of the live file are `cmp`-identical.

## Pass 17 — P16-n2 fix, U5.5 `maxTurns` fixture swap, plan6 v1.14 (2026-11-01)

**Verdict: approve with suggestions.** All four parts conform to what they cite, and the new code and fixture are pinned. The one finding is stale prose.
- **P17-m1 (minor):** `README.md` and the always-loaded `AGENTS.md` still say the smoke has not run, that `$CLAUDE_PROJECT_DIR` is unverified and that the `maxTurns` row is provisional. The same tree's HISTORY, plan6 v1.14 and the swapped fixture say the opposite.
- **P17-n1 and P17-n2 (nits):** the unstripped-key variant of the repeat check is unpinned, and plan6 v1.14's "margin stands" wording hides that `P_max + 4` has no slack at three answer `Write`s.

### Scope

- **Reviewed:** the uncommitted tree against the commits it follows. That is (1) `install.py` and `tests/test_install.py` for P16-n2, against plan6 v1.14 §3.3 and the §5 U5.7 row; (2) `tests/test_audit.py`'s `max_turns_run`, against `decision-models/docs/test-reports/live-smoke-report.md` Batch 2; (3) plan6 v1.14's diff (§2 L1–L6, §3.3 "`maxTurns` margin", the §3.4 row and the §4/§5 U5.7 rows); (4) the `audit.py` docstring, the `README.md` line and the `HISTORY.md` entries.
- **Suite:** run on the live tree from `decision-models/` with a fresh `PYTHONPYCACHEPREFIX`: **1146 passed, 2 xfailed**; `ruff check` clean.
- **Mutants and probes:** private copy `scratchpad/p17/repo/`, fresh `PYTHONPYCACHEPREFIX` per run, no `-I`. Baseline 1141 passed + 5 skipped. Each file was restored and `cmp` reported equal. 12 mutants and 6 probe tests, which I added to the copy only (Appendix P).
- **Live tree:** no git command that changes it; I wrote only this review. The reviewed files' sha256 values are in `scratchpad/p17/live-sha256.txt`. The smoke report is `qa-engineer`'s and was read, not run.

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; the changed hunks were read in full.

### Answers to the gate's questions

**(1) P16-n2 conforms, and it is pinned.**
- `frontmatter()` refuses a repeated key and `definition_error` pins `name` to `dm-<role>`, which gives `dm-driver`, `dm-generator` and `dm-labeler` as the plan says. `_read_sources` turns a parser refusal into `<file> <reason>`, so the refusal names the file.
- Every refusal comes before any write. Probe P3 (a driver with `name` twice) refuses with the file name and leaves no `exchange/.claude`, and the planted-definition test asserts the same for each of its cases.
- **Mutants:** dropping the name check (Q1) gives 3 red, a prefix-only check (Q2) 3, a labeler-only check (Q3) 62, no repeat check (Q5) 2, an identical repeat tolerated (Q6) 1, the file name lost (Q7) 2, and the wrong key named (Q8) 2. The one survivor is Q4 (P17-n1).
- Probes P1, P2 and P4: a repeat written `maxTurns : 2` is refused, `maxturns` is refused as unknown, and `name: dm-labeler # x` is refused.

**(2) The `max_turns_run` fixture matches the report's shape; the old shape classifies the same.**
- **Against Batch 2:** a header `Read` for turn 1, nine parallel page `Read`s for turn 2 (the last at offset 50, as the report says), each guard-allowed and answered, no `Write`, no `SubagentHandback`, and the transcript ending on a user `tool_result`. The driver's `Agent` result is `completed` with `canContinueAgent: false`, `toolsUsed: ["Read"]` and the NOTE text. The meta has `finished: true`, and the test asserts all of that. The nine tool uses share one message id, as the Pass 5 reading says a real parallel message does.
- **One inconsistency, a nit within the report:** the fixture copies the report's `tool_uses: 11` and `totalToolUseCount=11`, but it holds 10 tool uses (1 + 9), and the report itself describes 1 + 9. The audit reads neither number.
- **The old provisional shape:** probe O1 runs the baseline fixture (`clean_run().agent[:-2]`) against the current code: `(count, run-attributable, maxTurns)`, one finding, ending on a user entry, as before. `audit.py`'s logic did not change, only its docstring, and both shapes go through `_terminal`, which reads the last assistant entry. For the real shape that is one of nine same-id entries holding one `Read`, with nine `tool_result`s after it. So it is not a hand-back and not terminal.
- **Mutants:** reading the first assistant entry instead of the last (R6) gives 156 red, and gating the row on the driver's `canContinueAgent` (R5) gives 2. The row's existing mutants stay red.

**(3) Plan6 v1.14: it holds, with one wording point (P17-n2).**
- **L1 to L6 are faithful to the smoke report,** each checked against it: the verbatim and unexpanded `hook_success.command` and the accepted `timeout: 30` (L1); the clean batch, no `is_error`, the nine parallel Reads and 4 turns of 14 (L2); no `SubagentHandback` and a harness text frame that is not a tool call (L3); the cut-short shape under `maxTurns: 2` and its void classification (L4); three spawns before `retryCapExceeded` (L5); the driver's `Agent` input keys (L6).
- **The F-1 acceptance rewrite holds up.** It replaces "usage deltas around each Read", which a parallel reader makes unmeasurable, with "no `Read` `is_error` at the 25,000-token cap", the recorded aggregate, and a proportional estimate of at most 22,500. Each of these can fail. I recomputed the estimate: 164,763 × 38,004 / 321,716 is about 19,460, and the page sizes (37,991 for six pages, 38,004 at most) make the proportion nearly exact. The aggregate also contains the turn's output tokens, so the true figure is lower. The measured 1.95 characters per token agrees with plan4's 2.0, which is the source of the 40,000-character page budget.
- **The F-3 ruling, "the margin stands", is right but loosely worded.** `P_max` is 10 for labelling with the header page included, so a serial reader makes at most 10 `Read`s, 3 `Write`s and 1 reply: exactly 14, the pinned `maxTurns`, with no slack. The smoke's 4 of 14 is the parallel case, and a typical serial 50-item batch is 12. The report's own "header + 10 pages + Write + reply ≈ 12" miscounts (13), and the plan quietly corrects it to nine pages. The "three `Write`s" are an assumption nothing bounds, because parts are cut "when one Write would be too long". A cut-short run is a counted void, so the plan's "not a halt" holds.

**(4) Docs: the edits are accurate, and the sweep found stale prose (P17-m1).**
- The `audit.py` docstring matches the code (the row reads the last assistant entry and does not read the driver's note), and the README line states the same rules as `definition_error`. HISTORY records the swap, the new rows and the mutants.

### Findings

**P17-m1 — minor — five statements in `README.md` and `AGENTS.md` are false after the smoke and the fixture swap.** *Owner: `coder` (or `teco` in the close pass). Verified by an unfiltered sweep for `provisional`, `smoke`, `unverified` and `has not run` over `decision-models/` and plan6.*
- `AGENTS.md:19` says "U5.7's live smoke has not run". `AGENTS.md:45` says the guard's command "uses an unexpanded `$CLAUDE_PROJECT_DIR`, unverified in a live transcript until the smoke". `AGENTS.md:77` says "`maxTurns` is provisional until U5.7's smoke".
- `README.md:138` titles a section "The label smoke (human-started, not yet run)", and `README.md:154–161` still states the old acceptance (the largest page's tokens from "the transcript's usage deltas around each Read", and a "provisional `maxTurns` fixture").
- **Why it matters.** `AGENTS.md` is loaded in full into every session, and root `AGENTS.md` makes it answer "what is true now": all three claims are now untrue, and they contradict the same tree's HISTORY, plan6 v1.14 and `audit.py`'s docstring. A later agent would doubt a verified fact (L1) or re-run the smoke.
- **Fix:** replace each entry in place, citing `docs/HISTORY.md` and `docs/test-reports/live-smoke-report.md` in a clause, not adding a new paragraph. Reword the README's smoke section to plan6 v1.14's acceptance (no `is_error`, the aggregate, the estimate).

**P17-n1 — nit — the repeat check's key normalisation is unpinned.** *Owner: `coder`. Verified by mutant Q4 (0 red) and a probe that kills it.*
- Comparing the raw key, while storing the stripped one, leaves the suite green: a repeat written `maxTurns : 2` would then overwrite the first value silently. The planted cases all repeat the identical spelling.
- **Fix:** add the space-before-colon variant to `DEFINITION_PLANTS`, with the expected refusal text written in the test (probe P1).

**P17-n2 — nit — plan6 v1.14's "margin" wording, and one fixture count.** *Owner: `architect`; `coder` for the second part. Verified by arithmetic from §3.2 of plan4 and the report.*
- Say that `P_max + 4` is the serial worst case with no slack at three answer `Write`s, that the `Write` count is an assumption (parts are cut when one `Write` would be too long), and that the measured 4 of 14 is the parallel case. The rule does not change.
- `max_turns_run` sets `tool_uses: 11` and `totalToolUseCount=11` while holding 10 tool uses. Make the fixture agree with itself, or say in a comment that the numbers are the report's.

### Appendix P — Pass 17 mutants and probes

Files are in `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/p17/`: `mutants.py`, `mutate.py`, `probes/test_p17_probes.py`, `logs/mutants-run.txt`, `logs/probes.txt`. Baseline in the copy: 1141 passed + 5 skipped, 2 xfailed. Each mutant ran the full suite; each file was restored and `cmp` reported equal.

| ID | Mutation | Red |
|---|---|---|
| Q1 / Q2 / Q3 | name check dropped / prefix-only / compared with the labeler's for every role | 3 / 3 / 62 |
| Q4 | repeat check on the raw key (no strip) | **0**; 1 with probe P1 |
| Q5 / Q6 | repeat check dropped / an identical repeat tolerated | 2 / 1 |
| Q7 / Q8 | the refusal loses the file name / names the wrong key | 2 / 2 |
| R1 | `_terminal`: no assistant entry counts as terminal | **0** (outside this diff; the row's code is unchanged) |
| R4 | the `maxTurns` row ignores a synthetic error | **0** (outside this diff) |
| R5 | the row gated on the driver's `canContinueAgent` | 2 |
| R6 | `_terminal` reads the first assistant entry, not the last | 156 |

**Probes** (pytest functions added to the copy's `tests/` only, under `fake_claude`):
- **O1:** the baseline provisional fixture against the current audit → `(count, run-attributable, maxTurns)`, one finding.
- **O2:** the entry kinds of `clean_run` (context for the fixture's shape).
- **P1:** a repeat written `maxTurns : 2` → refused with "repeats the frontmatter key `maxTurns`".
- **P2:** `maxturns: 2` → refused as an unknown key.
- **P3:** `name` twice on the driver → refused with the file name first, and no `exchange/.claude`.
- **P4:** `name: dm-labeler # x` → refused.

R1 and R4 survive but sit in code this round did not change (the row's condition already existed); I report them as context, not as findings.

**Growth-stops check:** Pass 17 adds one minor and two nits and re-opens no earlier finding. Passes 1–16 are unchanged: the pre-pass copy and the first part of the live file are `cmp`-identical.
