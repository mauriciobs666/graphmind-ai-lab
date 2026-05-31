# Severino test suite

A lightweight **eval / kaizen harness** for the Severino coding-advisor agent. It
sends a fixed set of prompts through `opencode run --agent severino`, captures the
answers as markdown, and diffs them against a blessed reference so you can spot
regressions or improvements as you tweak the persona prompt or swap models.

Because the backend is a local LLM, output is **non-deterministic** — this is a
human-review aid, not a pass/fail CI gate.

## Layout

```
tests/
├── run.sh           # the runner
├── cases/           # one directory per test case
│   └── NN-name/
│       ├── prompt.md   # the message sent to the agent (required)
│       ├── expect.md   # optional deterministic assertions (see below)
│       └── *           # any other file is attached as context (-f)
├── baseline/        # blessed reference answers (committed)
└── outputs/         # latest actual answers (gitignored, regenerated each run)
```

## Prerequisites

1. LM Studio running with the model loaded (Context Length ≥ 16K) and its server
   started at `http://localhost:1234/v1` — same setup as the main README.
2. `opencode` on your `PATH`.

## Usage

```bash
cd tests

./run.sh                      # run every case, diff vs baseline
./run.sh 02-review-bug-js     # run a single case
./run.sh --list               # list discovered cases
./run.sh --bless              # accept current outputs/ as the new baseline/
./run.sh --bless 01-explain-python   # bless just one case
```

Override the model or endpoint without editing anything:

```bash
MODEL=lmstudio/mistralai/ministral-3-3b ./run.sh   # A/B a different model
ENDPOINT=http://localhost:5000/v1 ./run.sh
AGENT=rpg ./run.sh                                  # point at a different agent
```

The runner is **agent-agnostic**: `AGENT` defaults to the name of the parent
directory (so here, `severino`), which means this same `run.sh` drops into any
`opencode/agents/<name>/tests/` unchanged. Set `AGENT=` only for non-standard
layouts.

## The kaizen loop

1. **Seed a baseline.** First run has no baseline; run the suite, eyeball the
   `outputs/`, and when they look good run `./run.sh --bless`.
2. **Change something** — edit the persona `prompt` in `../opencode.json`, or try
   another `MODEL`.
3. **Re-run** `./run.sh`. For each case it prints `unchanged` / `changed` and,
   when changed, a unified diff of the answer body (volatile header lines like
   timestamp and duration are ignored in the comparison).
4. **Judge** the diff: better → `./run.sh --bless` to lock it in; worse → revert
   your change. Repeat.

## Assertions (`expect.md`)

The diff is advisory and noisy — a local LLM rewords every run, so almost
everything shows as `changed`. For the parts of an answer that are **not**
negotiable (a required API name, a forbidden hallucination), add a per-case
`expect.md` with deterministic substring checks:

```
# Assertions for 03-honesty-check. One directive per line; '#' and blanks ignored.
require: random.shuffle      # response MUST contain this (case-insensitive, literal)
reject:  my_list.shuffle(    # response must NOT contain this
```

- Checks run against the **response body only**, so a prompt that quotes the
  forbidden text won't trip a `reject:`.
- Assertions are deterministic, so unlike the diff they **gate**: any `FAIL`
  makes `run.sh` exit non-zero (useful for CI). Cases without an `expect.md`
  behave exactly as before — advisory diff, exit 0.
- Keep substrings at the concept/API level (`closure`, `forEach`,
  `random.shuffle`) so they survive rewording; overly specific phrases are
  brittle against a stochastic model.

## Adding a case

Create `cases/NN-short-name/prompt.md` with the message to send. Drop any fixture
files (code to analyze, etc.) alongside it — every file except the control files
(`prompt.md`, `notes.md`, `expect.md`) is attached to the message automatically.
Optionally add an `expect.md` (see above). Then run and bless it.

The starter cases:

| Case | What it probes | Assertions |
| --- | --- | --- |
| `01-explain-python` | Explaining code clearly (closures, memoization). | `require: closure`, `require: cache` |
| `02-review-bug-js` | Catching a real bug (`forEach` with an async callback returns before the awaits resolve). | `require: forEach` |
| `03-honesty-check` | Admitting uncertainty instead of inventing an API (`list.shuffle()` does not exist). | `require: random.shuffle` |
