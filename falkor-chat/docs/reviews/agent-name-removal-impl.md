# K-095 `config.AGENT_NAME` removal — implementation review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-095

## Scope & verdict

**Reviewed:** the uncommitted working-tree diff (`git diff --`) for the seven files in the brief:
`server/falkorchat/config.py`, `server/.env.example`, `scripts/start_server.sh`,
`scripts/start_demo.sh`, `server/tests/test_start_demo_script.py`,
`server/tests/test_opencode_config_resolution.py`, and the new
`server/tests/test_agent_name_handoff.py`. **Baseline:** `docs/BACKLOG.md` `### K-095` (removal
by default; done-condition "no unread config key; SERVER.md §1.3's row updated at close"). I did
not look at any other modified file in the tree.

**Verdict: approve with suggestions.** The removal is correct and complete. Nothing in the server
or anything it launches reads the variable (an unfiltered sweep, mapped below). The seed handoff is
kept and pinned. Every claimed mutant reproduces, and so does teco's. One major issue: the new
general guard `test_every_config_key_is_read` counts text matches, not reads. A comment or a
docstring satisfies it, and that is a shape this codebase already has (F1). Fix it, or narrow its
claim, before the close leans on it as proof of the done-condition.

CPG: considered, not relevant — a one-key removal plus shell-export edits; an unfiltered text
sweep and an AST read census over `falkorchat/` answer "who reads it" more directly than the
graph, and the shell scripts aren't in a Python CPG.

**Checks run** (all FalkorDB-free, from `falkor-chat/server`; no `wf_repo` test and no
`test_queries.sh` ran, so `reference` was not touched):
- `pytest tests/test_agent_name_handoff.py tests/test_start_demo_script.py
  tests/test_opencode_config_resolution.py`: **37 passed**. Adding `tests/test_services.py`
  (the documented review-safe subset) gives **359 passed**. I did not reproduce coder's exact
  "128" set, so that count is unverified, but every subset I ran is green. The full suite did not
  run (it includes the tests that wipe `reference`).
- Mutants, each run against a scratch copy of `scripts/` + `server/{falkorchat,tests}` (the
  harness resolves `_SCRIPTS` from `__file__`, so the copy is what runs). **Killed:** the
  `config.AGENT_NAME` line put back; the export put back in `start_server.sh` and in
  `start_demo.sh`; the seed prefix assignment dropped in each script; the `:-Assistant` default
  changed in each script. That is coder's 5 plus teco's, and the same default mutant in
  `start_server.sh`. **Survived:** see F1.

## Findings

### F1 — major — `test_every_config_key_is_read` counts word matches in source text, so a comment or docstring mention passes it (verified)

`test_agent_name_handoff.py` `test_every_config_key_is_read` requires at least 2 `\bKEY\b`
regex hits across the raw text of `falkorchat/**/*.py`. The definition is one hit. Any second hit
counts, whether it is code, a comment, a docstring or a string. The `\b` handling of substrings
is sound: `_` is a word character, so `FALKORCHAT_AGENT_ID` does not match `AGENT_ID`. The
problem is the comments and strings. Mutants run on copies:
- `AGENT_NAME` put back with a trailing `# AGENT_NAME: display name` on its own line: **passes**.
- `AGENT_NAME` put back, plus `# see config.AGENT_NAME` in `app.py`: **passes**.
- **A live case:** `USER_ID`'s only real read is `config.py:289` (`get_context`). After
  replacing it with `actor="u1"`, the key is dead, yet the test **passes** on prose alone:
  `repository.py:4056/4487/4603` and `storefront.py:866` cite `config.USER_ID` in comments and
  docstrings.

So the test proves "every key name is mentioned twice", not the done-condition "no unread config
key". Today the done-condition does hold. An AST load census (Appendix A) shows all 26 keys
genuinely loaded: 24 outside `config.py`, and `USER_ID`/`LEGACY_MODEL_ENV_VARS` inside
`config.py`'s own functions. But the regression guard is weaker than its docstring says.

**Suggested improvement:** count AST reads instead of text. Collect every `ast.Name` with `Load`
ctx, every `ast.Attribute` with `Load` ctx (its `.attr`), and `from .config import X` aliases
across `falkorchat/**/*.py`. A key is read if it is in that set. The definition is a `Store`, so
it never counts itself. This matches an existing convention, the AST guard in
`test_modelconfig.py` (`docs/SERVER.md` line ~735). I ran it (Appendix B): baseline `[]`. It flags
`USER_ID` under the dead-`USER_ID` mutant and `AGENT_NAME` under both comment mutants and a
string-literal mutant. Its remaining bound: an unrelated local or attribute that shares a key's
name still satisfies it. Say so in the docstring rather than claim proof. If coder keeps the
regex, the docstring must stop claiming the done-condition "in general form".

### F2 — nit — the one theoretical behaviour change is a `{env:FALKORCHAT_AGENT_NAME}` model-config substitution (inferred; no instance exists)

Before this change, uvicorn always got the variable from the start scripts. Now it gets it only
if the operator's own shell exports it. The server spawns no subprocesses (no
`subprocess`/`create_subprocess`/`os.system`/`exec*` in `falkorchat/`; `mcp.py` "spawns" threads).
In both scripts, the only command after the export block is `exec uvicorn`. That leaves one
remaining reader of the uvicorn environment: `modelconfig._substitute`'s `{env:NAME}`. A model
config that references `{env:FALKORCHAT_AGENT_NAME}` would now fail startup loudly
(`ModelConfigError`) under the start scripts. Neither `falkor-chat/config/*.json` nor the
operator's `~/.config/opencode/opencode.json`/`opencode.local.json` contains it. No action needed.
A one-line mention in the HISTORY entry would be enough.

## Sweep — "nothing else reads the uvicorn-side variable"

`/usr/bin/grep -rnI AGENT_NAME` over all of `falkor-chat/`, including dotfiles and `server/.venv`
(the shell's `grep` is a wrapper, so I bypassed it), plus a repo-wide sweep. Every hit is mapped:

| Hit | Owner / ruling |
|---|---|
| `scripts/seed_demo.sh:33,47,80,82` | the live consumer (stakeholder: keep). No change. |
| `scripts/start_server.sh:36,91,117,203`; `start_demo.sh:48,161,276` | header (reworded, accurate), usage text (still true: the override is honoured for seeding), default, seed prefix. No change. |
| `server/tests/*` (3 files) | this unit's tests. |
| `scripts/seed_eval_corpus.{py,sh}` `EVAL_AGENT_NAME` | a different variable. No change. |
| `docs/SERVER.md:171` | live-spec close (below). |
| `docs/BACKLOG.md:916-925` | closeout removes K-095. |
| `docs/HISTORY.md:206,6585`; `docs/plans/live-spec*-coordination.md`; `docs/reviews/*` | records. No change. |
| `docs/test-plans/workflows.md:24` | an executed test plan's recorded env (`qa-engineer`). Setting the variable on a hand-run uvicorn is harmless. No change. |
| `cpg/spikes/frontend-fitness/**`, `cpg/.ccg-work/**`, `cpg/.cpg-artifacts/**` | pinned-corpus snapshots (`config.py:87`, an older line number). Not consumers. No change. |
| `server/.venv`, `config/`, `compose.yaml`, `Dockerfile`, `.claude/` | no hits. |

No `config.AGENT_NAME` read exists anywhere, and no test monkeypatches it (a
`setattr(config, "AGENT_NAME")` would now raise).

## What's solid

- The removal is minimal and exact. The `.env.example` header ("the env vars the served app
  reads") now matches its content.
- The handoff tests drive the real scripts in a hermetic env (`HOME` + `PATH` only), so an
  operator's exported variable can't leak into the `uvicorn == None` assertion. Each assertion
  has a mutant that turns it red, with expected values (`"Assistant"`, `"Helper"`) independent of
  the scripts.
- The reworded headers state the truth: "handed to seed_demo.sh only (the server never reads it)".

## Live-spec close — what must change

- **`docs/SERVER.md` §1.3, line 171 (the `FALKORCHAT_AGENT_NAME` row):** delete it. The table's
  intro (line 154: "Every key is an environment variable read once … by `config.py`") now
  contradicts a row for a variable `config.py` no longer reads. If the seed fact should stay
  visible, it belongs in the seed script's own documentation (`seed_demo.sh:33` already has it),
  not in the server-config table. If the row is kept as a pointer instead, it must say "not read
  by the server" and name both `start_server.sh` and `start_demo.sh` (today it names only the
  former).
- **Nothing else:** no other live-spec mention (`DESIGN.md`, `QUERIES.md`, `README.md` and
  `AGENTS.md` swept). Line 154's "reads no other environment variable apart from `{env:NAME}`"
  stays true.

## Proposed REVIEW.md lines

Target: `falkor-chat/REVIEW.md`, `## Always check`. Source: F1. This codebase cites
`config.X` in comments and docstrings, so text-match guards false-green here specifically.

```
- A guard that proves a name is read (or unread) is checked with a planted comment-only and a
  string-only mention of the name. A regex or grep over source text counts prose: `config.X`
  is cited in docstrings across `falkorchat/`. A text match is a bound; count AST loads.
```

No prune: no line in `falkor-chat/REVIEW.md` is now enforced elsewhere.

## Open questions

None.

## Appendix

### A — AST read census of `config.py` keys (current tree)

All 26 module-level public keys have at least one `Load`. `USER_ID` (`get_context`, `config.py:289`) and
`LEGACY_MODEL_ENV_VARS` (`assert_no_legacy_model_env`) are loaded only inside `config.py`. The rest
are loaded outside it. Regex counts exceed load counts for most keys (for example `ENABLE_AGENT`
has 5 regex hits and 1 load, and `FALKORDB_SOCKET_TIMEOUT` 7 and 1), which measures how much prose the regex
counts.

### B — the AST guard I ran

```python
reads = set()
for p in pkg.rglob("*.py"):
    for n in ast.walk(ast.parse(p.read_text())):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load): reads.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load): reads.add(n.attr)
        elif isinstance(n, ast.alias): reads.add(n.name)
unread = sorted(keys() - reads)   # keys(): the test's own _config_keys()
```

Results: baseline `[]` · dead-`USER_ID` → `['USER_ID']` · `AGENT_NAME` + self-comment →
`['AGENT_NAME']` · `AGENT_NAME` + string literal in `app.py` → `['AGENT_NAME']`.

## Pass 2

**Verdict: approve.** I re-checked by executing the rewritten `test_every_config_key_is_read`
(an AST load census with alias resolution) against mutants on scratch copies of
`scripts/` + `server/{falkorchat,tests}`. FalkorDB-free subset: the three K-095 test files +
`tests/test_services.py`, **359 passed**. Nothing that wipes `reference` ran.

- **F1: fixed.** Baseline green. Each of these mutants turns the test red:
  - the key restored;
  - the key restored with a trailing comment;
  - the key restored with a `# see config.AGENT_NAME` comment in `app.py`;
  - the key restored with a string literal `"config.AGENT_NAME"`;
  - `USER_ID` made dead, with only its docstring and comment `config.USER_ID` mentions left;
  - the key restored with an unused `from .config import AGENT_NAME as _AN`;
  - the key restored with an unused plain `from .config import AGENT_NAME`.

  Positive controls stay green: a used relative alias (`print(_AN)`) and a used absolute
  `falkorchat.config` alias inside a function. The docstring now states the bound.
- **My Pass 1 recipe: refuted, and the coder's departure is correct.** My recipe counted every
  `ast.alias` as a read. Against the unused-aliased-import mutant it returned `[]`, so it would
  have kept a dead key green (coder's M-e).
- **F2: unchanged** (nit, no action).
- **New, nit (verified): the alias check is package-wide, not per file.** An unused
  `from .config import AGENT_NAME as _AN` in `app.py` stays green when `mcp.py` happens to load
  its own `_AN`. That is the same name-collision class the docstring already declares as a bound,
  so no change is needed. Resolving aliases per file would close it cheaply if wanted.
- **Proposed REVIEW.md line:** unchanged from Pass 1.

## Close check

**Verdict: approve.** Checked the uncommitted `docs/SERVER.md` change against the shipped diff
`940dc1e2` (`config.py`, `.env.example`, both start scripts). The spec change is a one-line
deletion of the §1.3 `FALKORCHAT_AGENT_NAME` row.

- **The table intro is still true (verified).** The intro at line 154 says every key is an
  environment variable read once by `config.py`, and that the server reads no other environment
  variable apart from `{env:NAME}`. I compared the env-var literals in `config.py` with the
  table's rows. Every row is read by `config.py`. The only `config.py` names with no row are the
  four `LEGACY_MODEL_ENV_VARS`, which are refused rather than read, and §1.8 (line 788) documents
  them. Outside `config.py`, the only environment read in `falkorchat/` is
  `modelconfig._substitute`, which is the `{env:NAME}` exception the intro already names. The
  deletion removed the one row that contradicted the intro.
- **No other live-spec mention (verified).** `/usr/bin/grep -nI AGENT_NAME` over
  `docs/SERVER.md`, `docs/DESIGN.md`, `docs/QUERIES.md`, `README.md` and `AGENTS.md` finds no hits.
- **No change history (verified).** The diff adds 0 lines and removes 1. No section is
  renumbered and no "removed"/"formerly" note was added.
- **The shipped diff matches the reviewed one.** `940dc1e2` has the same `config.py`,
  `.env.example` and script hunks I gated in Pass 1.

Final `falkor-chat/REVIEW.md` proposal, unchanged from Pass 1 (target `## Always check`; source F1
and the Pass 2 result; no prune):

```
- A guard that proves a name is read (or unread) is checked with a planted comment-only and a
  string-only mention of the name. A regex or grep over source text counts prose: `config.X`
  is cited in docstrings across `falkorchat/`. A text match is a bound; count AST loads.
```
