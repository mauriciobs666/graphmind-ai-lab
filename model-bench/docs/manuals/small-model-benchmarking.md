# Small-Model Benchmarking — User Manual

> **Status:** active · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-09-20

## Who this is for

Anyone who wants to know, on their own machine, whether one local model does a job better than
another — "does this new model answer support questions more accurately than the one we're using
now?", "which of these embedding models retrieves better for our data?" — and wants that answer
tied to real evidence instead of a published leaderboard number. You need LM Studio running
locally with at least one model downloaded; no cloud account, no CI, nothing else running.

## Overview

`model-bench` measures **one local model** against **one versioned task pack** (a fixed set of
test items for one job — e.g. "call the right tool in a shop-assistant conversation," or "embed
these documents so the right one comes back first"). Every result is stored together with a
fingerprint of the exact environment it ran under, so a result from today can be honestly compared
against a result from three months ago on the same hardware — never against a different pack, a
different role, or a different machine's numbers.

Three things it deliberately does **not** do, because it's easy to expect them from a "benchmark"
tool: it never runs on its own (no CI hook, no scheduler — only a person typing a command starts
it); it never declares a model "passing" or "failing" (no threshold, no gate); and it never adds
scores across different jobs into one leaderboard number (a `tool-caller` score and an `embedder`
score aren't the same kind of thing, so nothing adds them together).

```mermaid
flowchart LR
    A[Attest your host] --> B[Pick a pack + a model]
    B --> C{Just checking the\npack is well-formed?}
    C -->|yes| D["./run.sh validate"]
    C -->|no, run it for real| E["./run.sh run"]
    D --> B
    E --> F[Result stored under results/runs/]
    F --> G["./run.sh compare"]
    G --> H[Markdown report in reports/ + on screen]
```

## Walkthroughs

### 1. One-time setup

```bash
cd model-bench
./setup.sh                        # creates .venv, installs the tool
./run.sh --help                   # confirms it's working
```

### 2. Attest your host (once per machine, redo if hardware/LM Studio changes)

A few facts about your setup — LM Studio's own app version, its KV-cache setting, how much RAM
your machine has, what else is running alongside it — can't be read from LM Studio automatically,
so you tell the tool once and it remembers:

```bash
./run.sh attest
```

This writes `host.json` (not checked into git — it's specific to your machine) and prompts you for
anything it can't fill in on its own. If you already know the values you want to set, pass them
directly: `./run.sh attest --set hostRamGb=16`. Every later run copies these facts into its own
result record, so a result stays self-contained even if you attest again later. If `host.json` is
missing or stale (LM Studio itself changed since you last attested), `run` refuses rather than
silently recording something that's no longer true.

### 3. Pick a pack and a model

A **pack** is one job to test. Right now there are five:

| Pack id | Tests the job of… | Needs a model that can… |
|---|---|---|
| `embedder-graphrag-retrieval` | embedding text for retrieval | produce embeddings |
| `guard-judge-understanding` | judging/classifying content | chat |
| `nlq-structured-query` | turning a question into a structured query | chat |
| `tool-caller-shop-assistant` | holding a conversation and calling tools correctly | chat — a model that explicitly advertises no tool-calling capability is refused; most models don't declare either way, and for those the pack also recognizes tool calls written out in prose |
| `chat-responder-grounded-answers` | answering questions grounded in given context | chat |

Ask Tico ("list all available models for the bench mark") at any time for the live list of what
LM Studio currently has on hand, since that list is whatever's loaded in LM Studio on your machine
right now — model-bench doesn't keep its own copy of it. Match the model's type to the pack: the
embedder pack needs one of LM Studio's `embeddings`-type models, everything else needs a chat
(`llm`/`vlm`-type) model.

### 4. (Optional) Validate a pack before spending a live run on it

No LM Studio connection needed — just checks the pack's own files are well-formed:

```bash
./run.sh validate --pack packs/tool-caller-shop-assistant
```

Useful mainly if you're building a new pack; skip it for an existing shipped pack, it's already
valid.

### 5. Run one model against one pack

```bash
./run.sh run --pack tool-caller-shop-assistant --model qwen/qwen3-4b-2507
```

This drives that model through every item (or, for `tool-caller`, every scripted conversation) in
the pack, talking to your live LM Studio, and — if it succeeds — stores the result under
`results/runs/`. A few optional flags:

- `--session <id>` — tag this run so you can pull it back out later with `--session` on `compare`.
- `--reference <key>` — mark a second model as the thing you're comparing *against*, if you already
  know which pair you want to compare.
- `--warmup <n>` — extra throwaway calls before the timed ones, if you want the model "warmed up"
  first.
- `--first-call-timeout <s>` / `--request-timeout <s>` — how long to wait for the first (warm-up)
  call and for each scored call, if the defaults are too tight for your hardware.

Nothing about the score decides whether this command "succeeds" — a run only fails for operational
reasons (see Troubleshooting below), never because a model scored badly.

### 5a. Testing several models in one sweep

There's no built-in "run N models" command — `run` is deliberately one model per invocation (see
Overview: no scheduler, no batch mode). The easiest way to sweep a subset is a plain shell loop,
tagging every run with the same `--session` so you can pull the whole batch back out together
afterward:

```bash
MODELS=("qwen/qwen3-4b-2507" "mistralai/ministral-3-3b" "qwen2.5-3b-instruct")
for m in "${MODELS[@]}"; do
  ./run.sh run --pack tool-caller-shop-assistant --model "$m" --session batch-20260918 || \
    echo "!! $m failed operationally, continuing"
done
./run.sh compare --pack tool-caller-shop-assistant --session batch-20260918
```

The `|| echo ...` matters: a `run` only exits non-zero for an operational reason (see
Troubleshooting), never a bad score, but that's still enough to stop a loop that isn't guarded
against it. Every model in the loop needs to be the right type for the pack (see the table in step
3), and each run is a full live pass through the pack, so N models costs roughly N× one pack's
runtime.

### 6. See what's already been tested

Before running a model again, check whether it already has a stored result:

```bash
./run.sh models --tested --pack tool-caller-shop-assistant
```

### 7. Compare two runs

```bash
./run.sh compare --pack tool-caller-shop-assistant --models qwen/qwen3-4b-2507,mistralai/ministral-3-3b
```

This reads the stored runs for that pack, renders a markdown report to both `reports/` and your
screen, and never overwrites an earlier same-day report for the same pack (it appends a sequence
number instead). Leave off `--models` and it picks from whatever's stored for that pack; use
`--session <id>` to compare only runs you tagged together in step 5.

```mermaid
sequenceDiagram
    participant You
    participant CLI as model-bench CLI
    participant LM as LM Studio
    participant Store as results/runs/

    You->>CLI: ./run.sh run --pack P --model M
    CLI->>LM: warm-up call
    LM-->>CLI: ack
    CLI->>LM: one call per pack item / conversation turn
    LM-->>CLI: replies + timing stats
    CLI->>Store: write result (score + fingerprint, never the raw reply text)
    You->>CLI: ./run.sh compare --pack P
    CLI->>Store: read stored runs for P
    CLI-->>You: markdown report (reports/ + screen)
```

### 7a. Ranking every model tested against a pack, not just two

`compare` is always a two-arm comparison. Once three or more models have a stored result for the
same pack, `rank` gives you the whole field in one table instead of running `compare` repeatedly:

```bash
./run.sh rank --pack embedder-graphrag-retrieval
```

This prints (and saves to `reports/`) one row per model with a stored result — its point estimate,
a descriptive confidence interval, p95 latency, and footprint — sorted, never as a pairwise
matrix. A pack with two co-equal metrics and no single headline (e.g. `guard-judge-understanding`)
gets one table per metric instead of one overall table.

Two optional flags change what the table can claim:

- **`--reference <modelKey>`** adds a statistically-corrected verdict column against that one named
  model — `distinguishable` or `not distinguishable` (direction, when distinguishable, comes from
  the row's own signed diff column, not from the word "better"/"worse" itself), corrected for
  testing every other model at once (a Holm-Bonferroni family), never a raw uncorrected p-value per
  row. Leave it off and the table states only descriptive intervals — no verdict column, no p-value
  anywhere. Naming a model with no stored result for this pack is a usage error (exit `2`) —
  nothing is written.
  > ⚠️ **Known bug:** `--reference` currently crashes (exit `2`, an internal error naming
  > `scored_value`) against any pack whose headline metric is *continuous* rather than a plain
  > success/failure rate — `embedder-graphrag-retrieval`'s `mrr` is exactly this case, so
  > `./run.sh rank --pack embedder-graphrag-retrieval --reference <key>` does not work today. It's
  > fine on a boolean-outcome pack (e.g. `guard-judge-understanding`). Until this is fixed, use
  > `compare` instead for a reference-style question against a continuous-metric pack — `compare`
  > already handles `mrr` correctly.
- **`--footprints <path.json>`** — a flat `{"<modelKey>": "<display string>"}` map, e.g.
  `{"qwen/qwen3-4b-2507": "4B, Q4_K_M, 2.4 GB"}`, rendered verbatim in its own column. A missing or
  malformed value just shows as a dash or its raw string — never parsed as a number; a malformed
  *file* is a usage error, exit `2`.

Like `compare`, a same-day re-run never overwrites an earlier report — `rank` uses its own
`-rank-` sequence in the filename, so `compare` and `rank` reports for the same pack and day don't
collide.

### 8. Recovering the results index

`results/index.csv` is an **optional, on-demand** flat summary of everything under
`results/runs/` — nothing else writes it or reads it. `compare`, `rank`, and `models --tested` all
read the actual result files under `results/runs/` directly, never this file; the only way
`index.csv` ever gets created or refreshed is by asking for it explicitly:

```bash
./run.sh index rebuild
```

This reads every file in `results/runs/` and regenerates `results/index.csv` from scratch — it
never edits a stored run record, only this derived file. Useful if you want one flat file to skim
or feed to something else (a spreadsheet, a quick `grep`); model-bench itself never needs it to
exist.

## On-disk data shapes

`model-bench` has no database and no server — every "record" is a plain JSON (or CSV) file. The
three that matter, and how they relate:

```mermaid
flowchart LR
    subgraph "packs/<pack-id>/"
        Pack["pack.json<br/>role, scorer, sampling,<br/>metrics, packContentHash"]
        Golden["golden data<br/>(queries.jsonl, corpus.jsonl, ...)<br/>+ PROVENANCE.md"]
    end
    Host["host.json<br/>(gitignored, per-machine)<br/>operator-attested facts"]
    Run["results/runs/*.json<br/>one RunResult per<br/>model x pack x time"]
    Idx["results/index.csv<br/>derived summary,<br/>rebuilt from Run files"]

    Pack -- "declares" --> Golden
    Pack -- "packContentHash + packId/packVersion<br/>stamped into" --> Run
    Host -- "attested fields copied into" --> Run
    Run -- "./run.sh index rebuild" --> Idx
    Run -- "read by" --> Compare["compare / rank"]
```

- **`pack.json`** — one job's declaration. Key fields: `packId`, `packVersion`, `role` (one of the
  five FR-21 roles), `scorer` (the module that judges this pack's results), `environment.requires`
  (what the model must support — e.g. `lmstudio-embeddings`), `data` (paths to the golden files
  under the same pack directory), `sampling` (`seed`, `pairingKey`, `analysisUnit` — what makes two
  runs comparable at all), and `metrics` (`verdictMetrics`, `headlineMetric`). `packContentHash`
  covers the pack directory's content — a golden item, a prompt, or a tool schema changing all
  produce a new, distinguishable hash (its one deliberate exclusion is `PROVENANCE.md` itself) — a
  stored result always names the exact pack version it was measured against, never just a pack id.
- **`host.json`** — one file per machine, never checked into git (it describes *your* hardware, not
  the project). Its `attested` block holds four fields (`lmStudioAppVersion`, `kvCacheSetting`,
  `hostRamGb`, `otherResidentWorkloads`) that nothing in LM Studio exposes programmatically, so
  `attest` asks you once and remembers. Written by `./run.sh attest`; every later `run` copies these
  into its own result record.
- **`results/runs/*.json`** (one file per model × pack × timestamp) — the result itself. Three parts:
  a **`fingerprint`** block (pack identity + version, the model's own reported capabilities, every
  `host.json` field, timing of the run) that makes the record self-contained and honestly
  comparable later; an **`aggregates`** block (the scored metrics — shape depends on the pack's
  scorer, e.g. `mrr`/`recallAtK`/`precisionAt1` for an embedder pack); and an **`items`** list (one
  scored entry per pack item — never the model's raw reply text, only outcome/score detail). A
  top-level `sessionId` (not inside `fingerprint`) is what `--session` groups runs by.
- **`results/index.csv`** — a flat, derived summary of every file in `results/runs/`, written only
  by `index rebuild`, on demand — no other command creates, refreshes, or reads it. `compare`/
  `rank`/`models --tested` all go straight to `results/runs/`, never this file. Purely an optional
  convenience export; model-bench itself works the same whether it exists or not.

## Configuration & integration

**There is no environment-variable configuration** — deliberately, per the project's "zero runtime
dependencies, standalone" design: every setting is either a CLI flag (this manual's Walkthroughs)
or a file you can inspect directly (`host.json`, a pack's `pack.json`). Nothing here reads
`falkor-chat`'s own configuration, its model gateway, or its golden data live — any data that
originated in `falkor-chat` (e.g. the eval corpus behind `embedder-graphrag-retrieval`) was copied
in once, with its origin recorded in that pack's `PROVENANCE.md`, and stays a versioned, static
copy from then on.

**Integration with `falkor-chat`'s embedding-model migration workflow.** Before committing a
workspace to a new embedding model (see `falkor-chat`'s embedding-migration manual), the
recommended path is to validate the candidate here first, using the `embedder-graphrag-retrieval`
pack — no real workspace touched, no migration risked on a model that turns out to underperform.
This has already been exercised for real: a catalog sweep run against this pack found that
`granite-embedding-278m-multilingual` (the embedding-migration feature's lead replacement
candidate) scores within ~2-3 percentage points of MRR of the top-ranked model in the sweep, while
running at roughly **1/8th the footprint and ~4.6x lower p95 latency** — well inside the pack's own
statistical resolving power, so no model in the top cluster can be called definitively better or
worse than another from that data alone, but footprint/latency become the deciding factor once
quality is indistinguishable (`reports/catalog-sweep-2026-09-19-consolidated.md`, "embedder"
section). That is exactly the kind of evidence FR-6 of the embedding-migration feature asks for
before touching production data.

## FAQ / troubleshooting

**Should I use `compare` or `rank`?** `compare` is for a specific two-model question ("is A better
than B for this job?"). `rank` is for "how does everything I've already tested stack up?" — one
table across every model with a stored result, and it's the only one of the two that can name a
single `--reference` model and get corrected verdicts against the whole rest of the field at once
— **except on a continuous-metric pack like `embedder-graphrag-retrieval`, where `--reference`
currently crashes; use `compare` there instead** (see the callout in Walkthrough 7a).

**What are all the exit codes?** `0` — ran and reported, whatever the scores (a `compare`/`rank`
that finds every stored record invalid still exits `0`; that's a report, not a failure). `2` — bad
arguments/usage (including a `rank --reference` naming a model with no stored result, or a
malformed `--footprints` file). `3` — LM Studio unreachable, or didn't answer within
`--first-call-timeout`. `4` — an invalid pack, an environment requirement the model doesn't meet,
or (for `tool-caller`) a conversation cut short by a tool-dispatch failure. `5` — `host.json`
missing or stale. Nothing here is score-driven — see Overview.

**How do I view a report after it's written?** You don't need to hunt for it — `compare` prints
the same markdown to your screen as it runs. The saved copy is plain text at
`reports/<pack-id>-<date>-<NN>.md` (the two-digit suffix is a same-day sequence number, so a
re-run never overwrites an earlier report); open it in any markdown viewer or editor, or `cat` it.
Pass `--out <path>` on `compare` if you'd rather it land somewhere else entirely.

**"LM Studio unreachable" / exit code 3.** LM Studio isn't running, isn't answering its
`/api/v0/models` catalog endpoint, or didn't respond to the warm-up call in time. Start LM Studio
(or load the model) and try again; if it's slow to warm up, raise `--first-call-timeout`.

**"Invalid pack" / exit code 4.** Usually means the model you picked doesn't match what the pack
needs (e.g. you pointed the embedder pack at a chat model), or — for `tool-caller` specifically —
a conversation had to be cut short because a tool call couldn't be dispatched. In that last case
the partial result is still written to `results/runs/` before the tool reports the failure, so
nothing is silently lost.

**"Fingerprint incomplete" / exit code 5.** Run `./run.sh attest` again — either you haven't
attested this host yet, or something about your LM Studio setup changed since you last did.

**Why doesn't a bad score make the command fail?** By design. There's no "passing" score in this
tool — see Overview. A non-zero exit always means something operational went wrong, never that a
model did poorly.

**What does `--negative-control` on `compare` actually prove?** It compares one stored run against
a copy of *itself*, so the two "arms" are guaranteed identical — it's a smoke check that the
comparison machinery is wired correctly, not a real result. Two genuinely independent runs of the
same model is the real version of that check.

**Can I see the model's actual replies afterward?** No — a stored result never keeps the raw reply
text, only the score and a small scoring detail block. To see what a model actually said for a
particular item, you'd need to run it again live.

**Where do the numbers in a report come from — is a small difference between two models
meaningful?** The report accounts for run-to-run noise with confidence intervals and flags when a
difference isn't large enough to be sure of; if you want the statistical reasoning behind that,
ask Tico or see `docs/plans/small-model-benchmarking-ml.md` at the repo root.
