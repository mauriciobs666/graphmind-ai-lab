# TESTING.md — testing standards for graphmind-ai-lab

> Cobb's reference note. Captures the project's testing conventions and the
> reusable **agent eval harness** pattern, distilled from the Severino work
> (2026-05-31). Not an agent definition — a knowledge artifact for future cobb
> sessions. Keep it in sync when the pattern evolves.

## The two-altitude standard

There is **no single repo-wide test runner** in `graphmind-ai-lab`, and that's
deliberate (the root `AGENTS.md` says so: *"no root-level build/test scripts …
manual code checks only"*). Instead there are **two kinds of test, chosen by what
you're testing:**

| You are testing… | Use | Why |
|---|---|---|
| **Deterministic code** (functions, converters, parsers) | **pytest** | Output is exact and assertable. Red/green/refactor; matches the `tdd-engineer` agent's discipline. |
| **Agent behavior** (a persona prompt, a model swap) | **eval / bless harness** | A local LLM is non-deterministic — you can't assert byte-for-byte. You review diffs against a blessed reference and judge better/worse. |

### Standard 1 — pytest (deterministic code)

Canonical exemplar: `excel_extractor/`.

- `tests/` package with `__init__.py`, files named `test_*.py`.
- Config in `pyproject.toml`:
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  python_files = ["test_*.py"]
  ```
- `pytest>=8.0` in `requirements-dev.txt`.
- Idioms: `tmp_path` fixtures, small `_make_*` helpers, one behavior per test,
  clear arrange/act/assert.

Use this for any future library/utility code (`excel_extractor`, `salesperson`, etc.).

### Standard 2 — agent eval / bless harness (agent behavior)

Canonical exemplar: `opencode/agents/severino/tests/`. This is the reusable
pattern documented below.

---

## The agent eval harness pattern

### What it is

A `bash` runner (`run.sh`) that, for each **case**, sends a fixed prompt (plus
optional fixture files) through `opencode run --agent <AGENT>`, captures the
answer as markdown in `outputs/`, and diffs it against a blessed reference in
`baseline/`. Because the backend is a local LLM, this is a **human-review aid,
not a pass/fail CI gate** — you read diffs to spot regressions/improvements,
then `--bless` to accept a new baseline.

### Layout

```
tests/
├── run.sh           # the runner (agent-agnostic)
├── cases/           # one directory per case
│   └── NN-name/
│       ├── prompt.md   # the message sent to the agent (required)
│       ├── expect.md   # optional deterministic substring assertions
│       └── *           # any other file is attached as context via -f
│                       #   (prompt.md, notes.md, expect.md are never attached)
├── baseline/        # blessed reference answers (committed to git)
└── outputs/         # latest actual answers (gitignored, regenerated each run)
```

### The kaizen loop

1. **Seed a baseline.** Run the suite, eyeball `outputs/`, then `./run.sh --bless`.
2. **Change something** — edit the persona `prompt` in `opencode.json`, or swap `MODEL`.
3. **Re-run** `./run.sh`. Per case it prints `unchanged` / `changed` (+ unified
   diff). Volatile header lines (Generated/Duration/Model) are stripped before
   comparing, so the diff is signal.
4. **Judge** the diff: better → `--bless`; worse → revert. Repeat.

### Invocation

```bash
./run.sh                      # run all cases, diff vs baseline
./run.sh 02-review-bug-js     # run a single case
./run.sh --list               # list discovered cases
./run.sh --bless [case...]    # accept current outputs/ as new baseline/
MODEL=lmstudio/mistralai/ministral-3-3b ./run.sh   # A/B a different model
ENDPOINT=http://localhost:5000/v1 ./run.sh         # different server
AGENT=rpg ./run.sh            # override the auto-derived agent name
```

### Assertions (`expect.md`) — the deterministic gate

The diff is advisory and noisy (a local LLM rewords every run, so nearly
everything reads as `changed`). For the non-negotiable parts of an answer, a
case may carry an optional `expect.md` with literal substring directives:

```
require: random.shuffle      # response MUST contain it (case-insensitive, literal)
reject:  my_list.shuffle(    # response must NOT contain it
```

- One directive per line; blank lines and `#` comments ignored.
- Matched against the **response body only** (not the echoed prompt), so a
  prompt that quotes forbidden text won't trip a `reject:`. Matching is
  `grep -iF` (case-insensitive, fixed-string).
- Deterministic, so unlike diffs they **gate**: any `FAIL` makes `run.sh` exit
  non-zero. Cases without an `expect.md` stay advisory-only (exit 0). This is
  what makes the harness CI-usable for the parts that *are* deterministic.
- **Author at the concept/API level** (`closure`, `forEach`, `random.shuffle`)
  so substrings survive rewording. *Limitation:* a plain substring can't tell
  "use `list.shuffle()`" from "there is no `list.shuffle()`" — for negative
  intent, pick a signature that only appears in the *wrong* answer (e.g.
  `my_list.shuffle(`), or lean on a `require:` of the correct API instead.

### Design decisions worth preserving

- **`set -euo pipefail`**, tty-aware colors (no-op when piped), null-delimited
  fixture globbing (`find -print0` / `read -d ''`) — handles spaces in names.
- **Volatile-line stripping** before diff (`strip_volatile`) so timestamps/
  duration don't show as spurious changes.
- **Greedy `-f` gotcha:** `opencode run`'s `-f/--file` is a greedy array option;
  the positional **prompt must come before** the `-f` flags or it's parsed as a
  filename. (Documented inline in `run.sh`.)
- **Health check** pings `${ENDPOINT}/models` and fails fast with a fix-it hint
  if LM Studio isn't up.
- **Agent name auto-derives** from the parent directory name
  (`AGENT="${AGENT:-$(basename "$PROJECT_DIR")}"`), so the same script drops into
  any `opencode/agents/<name>/tests/` unchanged; `AGENT=` overrides.
- **Per-run stderr** goes to a `mktemp` file (not a fixed `/tmp/<name>.stderr`),
  so concurrent runs of different agents don't collide; on non-zero exit the
  stderr is echoed indented.
- **Non-zero `opencode` exit only warns** and continues (kaizen-friendly). A
  strict/CI mode that exits non-zero is a future option (see roadmap).
- **No absolute paths in committed reports.** Fixtures are attached as paths
  *relative* to `PROJECT_DIR` (the script `cd`s there first), so the agent sees
  and echoes relative paths — not the home dir / username. A `sed` scrub of the
  captured body is the safety net: `PROJECT_DIR/` → relative, remaining
  `$HOME/` → `~/`. Keeps baselines safe to commit to a shared repo.

### LM Studio / WSL prerequisites

- LM Studio server at `http://localhost:1234/v1`, a model loaded with
  **Context Length ≥ 16K** (32K recommended) or OpenCode's system prompt
  overflows with `n_keep >= n_ctx`.
- LM Studio runs on Windows; reach it from WSL2 via mirrored networking
  (localhost) or the gateway IP fallback.

---

## Promoting the harness to a shared pattern — status & plan

**Done (2026-05-31):**
- ✅ **#1 Proven green.** Ran all three Severino cases, blessed the baseline,
  confirmed the loop end-to-end (a re-run correctly reported `changed` when the
  model added a "TL;DR" section — diff mechanism works).
- ✅ **#3 Decoupled from "severino".** Agent name now derives from the parent
  dir; stderr uses `mktemp`; banner/help/comments are agent-agnostic.
- ✅ **#2 Lightweight assertions.** Optional per-case `expect.md` with
  `require:`/`reject:` literal substrings, checked against the response body,
  printing per-assertion `PASS`/`FAIL` and **gating the exit code** (any fail →
  non-zero). Seeded the three starter cases with assertions. Verified live: a
  clean run reports `4 passed, 0 failed` / exit 0, and a deliberately failing
  `reject:` reports `1 passed, 1 failed` / exit 1. Also hardened `--help` to
  print only the leading comment block (awk) instead of overshooting into code.
  Motivating evidence remains `02-review-bug-js`, where the model **diagnosed
  the bug correctly but emitted a broken fix** (`await` outside a non-`async`
  function) — a future `expect.md` can assert the correct fix shape.

**Roadmap (in priority order):**

1. **Pin temperature** (env `TEMP`, default low/0) to reduce sampling-driven
   diff noise. *med*
2. **N samples per case** (`RUNS=3`) to surface variance instead of a single
   coin-flip. *med*
3. **Stricter assertions** — optional regex directives (`require_re:` /
   `reject_re:`) for intent a literal substring can't capture, and per-case
   assertion counts in the summary. *low*
4. **`--no-gate` / advisory mode** — env or flag to downgrade assertion failures
   to warnings (exit 0) for pure exploration runs; the gate is on by default. *low*

**Distribution decision:** extract the shared template on the **second** consumer,
not the first. Candidates already in the repo: `rpg`, `coding-senior` (per
`opencode/agents/`). Two viable forms — decide when extracting:
- *Template/copy* per agent (`agents/_eval-template/`): simplest, but copies drift.
- *One shared `run.sh`* each agent's `tests/` calls: single source of truth, a
  little more wiring. Lean shared once a second agent proves the generality.
