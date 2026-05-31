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
│       ├── prompt.md   # the message sent to Severino (required)
│       └── *           # any other files are attached as context (-f)
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
```

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

## Adding a case

Create `cases/NN-short-name/prompt.md` with the message to send. Drop any fixture
files (code to analyze, etc.) alongside it — every file except `prompt.md` and
`notes.md` is attached to the message automatically. Then run and bless it.

The starter cases:

| Case | What it probes |
| --- | --- |
| `01-explain-python` | Explaining code clearly (closures, memoization). |
| `02-review-bug-js` | Catching a real bug (`forEach` with an async callback returns before the awaits resolve). |
| `03-honesty-check` | Admitting uncertainty instead of inventing an API (`list.shuffle()` does not exist). |
