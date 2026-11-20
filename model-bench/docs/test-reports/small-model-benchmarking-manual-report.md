# Small-Model Benchmarking Manual — Walkthrough Test Report

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** — · **Last updated:** 2026-11-20

Subject: `model-bench/docs/manuals/small-model-benchmarking.md` (as updated 2026-11-20 by `tico`: Walkthrough 9, `run --reference` "accepted and ignored", two-arm `compare` verdict rule, rewritten exit-code FAQ). The manual's own steps are the spec. Run date 2026-11-20, repo commit `84c0012f` plus the uncommitted manual edit.

CPG: considered, not relevant — black-box walk of a manual against the CLI; no code-level graph question.

## Method

Commands run from `model-bench/` with the repo `.venv/bin/python -m modelbench` (what `./run.sh` execs) and `--root` pointed at a scratch copy under `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/ec6fdb1e-9727-4ad1-8812-4897d433fff3/scratchpad/mb` (copy of `model-bench/` without `.venv`, including the 130 pre-existing stored runs). Repo `results/` and `reports/` were never written. `validate` and `models --tested` (read-only) ran against the repo directly. LM Studio state found at start: LLM `qwen/qwen3-4b-2507` loaded (`lms ps`), embedding `text-embedding-qwen3-embedding-0.6b` loaded (`lms ls`), no decision model loaded.

## Results

Legend: PASS = manual's stated output/exit/behavior observed. Exit codes are from this run.

### Offline / read-only steps

| Step | Command (abridged) | Observed | Result |
|---|---|---|---|
| 1 | `./run.sh --help` | usage lists `compare,rank,index,models,attest,validate,run` | PASS |
| 3 | pack count | `packs/` holds 8 dirs: 5 plain + `guard-judge-synthetic`, `join-in-judge-synthetic`, `message-type-classifier-synthetic`; all 8 `validate` exit 0 | PASS |
| 4 | `validate --pack packs/tool-caller-shop-assistant` | `tool-caller-shop-assistant 0.2.0 (tool-caller): valid`, exit 0 | PASS |
| FAQ | `validate --pack ... --strict` | `NotImplementedError` traceback, exit 1 (as the FAQ says) | PASS |
| 6 | `models --tested --pack tool-caller-shop-assistant` | 16 model keys, one per line, exit 0 | PASS |
| 9 | `models --tested --pack guard-judge-synthetic` | `guard-judge-v1-handoff-f1`, `guard-judge-v1-only-f1`, `qwen/qwen3-4b-2507`, exit 0 (candidates and chat model alike) | PASS |
| 7 | `compare --pack tool-caller-shop-assistant --models qwen/qwen3-4b-2507,mistralai/ministral-3-3b` | exit 0; report to screen and to `reports/tool-caller-shop-assistant-20261120-01.md`; a second run wrote `-02.md` (no overwrite) | PASS |
| 7 | `--models` naming a model with no stored run | `no stored run for 'nonesuch/x'...`, exit 2, no report written | PASS |
| 7 | `compare` without `--models` | every stored run is an arm, alphabetical by model key (first arm `gemma-3-4b-vl-...`, then `google/gemma-3-4b`...) | PASS |
| 7 | "verdict covers only the first two arms" (17 arms, `--session catalog-sweep-2026-09-19`) | arms table lists all 17; the single Verdicts section reports one paired comparison with `paired n: 12 of 12 ... gemma-3-4b-vl-... only / google/gemma-3-4b only`, i.e. the first two arms | PASS |
| 7 | `--out <path>` | report written to the given path, exit 0 | PASS |
| 7a | `rank --pack embedder-graphrag-retrieval` | one table, 5 rows, mean + 95% CI + p95 + footprint, saved to `reports/`, exit 0 | PASS |
| 7a | `rank ... --reference bm25` | line `alpha_family=0.05, k=4, alpha_used=0.0125 (98.75% CI)` printed once above a candidate/diff/CI/decision table, no Holm column, exit 0 | PASS |
| 7a | `rank --pack guard-judge-understanding` | two tables (`falseAdvanceRate`, `falseSuspendRate`) | PASS |
| 7a | `rank --reference nonesuch` | `reference model 'nonesuch' has no stored, consistent run...`, exit 2, nothing written | PASS |
| 7a | `rank --reference qwen/qwen3-4b-2507` on `guard-judge-understanding` | decision cells include `distinguishable`, `not distinguishable` and also `not tested (Holm stops here)` | DEFECT D-2 (minor) |
| 7a | `rank --footprints footprints.json` / malformed file | valid: exit 0; malformed: `not valid JSON ...`, exit 2, no report | PASS |
| 7a | `rank` on each decision-point pack | `rank is refused on decision-point pack <id>; use compare --session ...`, exit 2, nothing written | PASS |
| 8 | `index rebuild` (index.csv removed first) | recreated, 131 lines for 130 run files (header + rows), exit 0 | PASS |
| 2 | `attest --set hostRamGb=32` alone | writes nothing until the three other fields are given; blocks on an interactive prompt (stdin was a terminal-less shell: hung until killed) | NOTE N-1 |
| 2 | `attest` with all four `--set` | `wrote .../host.json`, hostRamGb 32, exit 0 | PASS |
| 2 | `run` with no `host.json` | `no host.json at ... run model-bench attest first`, exit 5 | PASS |

### Step 5 / 9 argument and refusal behavior (no model call made)

| Claim | Observed | Result |
|---|---|---|
| exactly one of `--model` / `--candidate` | both: argparse error exit 2; neither: argparse error exit 2 | PASS |
| LM Studio unreachable -> exit 3 | `host.json` pointing at `localhost:9`: `LM Studio is not reachable`, exit 3 (plain `--model` and `--candidate`) | PASS |
| embedder pack + chat model -> exit 4 | `callSurface='embeddings' but model ... has catalog type 'llm'`, exit 4 | PASS |
| chat pack + embeddings model -> exit 4 | exit 4 | PASS |
| unknown model -> exit 4 | `not in the LM Studio catalog`, exit 4 | PASS |
| unknown pack -> exit 4 | `pack.json is absent`, exit 4 | PASS |
| `--request-timeout` other than 180 on a decision pack -> exit 4 | `--request-timeout 60 ... drop the flag`, exit 4 | PASS |
| wrong-role candidate -> exit 4 | `it is a 'guard-judge' candidate and the pack's role is 'join-in-judge'`, exit 4 | PASS |
| unreadable candidate file -> exit 4 | `cannot read candidate ...nonesuch.json`, exit 4 | PASS |
| decision `compare` refusals (exit 2, nothing written): missing `--session`; missing `--models`; both missing; name with no run in session; repeated name; `--negative-control` | all exit 2 with the stated cause, no output file | PASS |
| decision `compare` on stored session `u7-guard-2026-11-19` (incumbent + `guard-judge-v1-handoff-f1`) | exit 0 although every bar reads `not assessable` | PASS |
| decision `compare` for `join-in-judge-synthetic` / `message-type-classifier-synthetic`, candidates only, and with one chat model added | exit 0; `FR-10 per-class agreement ... not met`, `mean latency ... not assessable (only 1 load replicate(s) (FR-11a needs 2))`; basis `by-construction` | PASS |

Further step-9 refusals, exercised with candidate files edited in the scratch copy only (all refused before any model call, no run stored):

| Claim | Observed | Result |
|---|---|---|
| decision model not `decision`-type -> exit 4 | `its decision model 'qwen/qwen3-4b-2507' has catalog type 'llm', not 'decision'`, exit 4 | PASS |
| decision model missing -> exit 4 | `model 'nonesuch@q8_0' is not in the LM Studio catalog`, exit 4 | PASS |
| hand-off model not the guard's incumbent -> exit 4 | `its hand-off model 'mistralai/ministral-3-3b' is not the speed-up point's incumbent`, exit 4 (an embeddings model gets the same incumbent message, not a "not a chat model" one) | PASS |
| mismatched fit list -> exit 4 | `sha256 ... is not the declared fitManifest.sha256`, exit 4 | PASS |
| leakage -> exit 4 | fit list holding the input hash of `gj-0001`: `fitted on an input it would be measured on`, exit 4 | PASS |

### Live runs (LM Studio shared and contended by other sessions during this run)

| Step | Command | Observed | Result |
|---|---|---|---|
| 5 | `run --pack guard-judge-understanding --model qwen/qwen3-4b-2507 --session qa-plain-1 --reference bogus-key --warmup 1` | printed `stored: .../guard-judge-understanding-qwen_qwen3-4b-2507-2026-11-20T14:40:48Z.json`; the record has top-level `sessionId: qa-plain-1`; the string `bogus-key` and any reference field appear nowhere in the record; no error for the unknown reference key. Took 18m51s (the normal earlier runs of this pack took about 1 minute; LM Studio was busy with another client) | PASS: `--reference` accepted and ignored, `--session` recorded |
| 9.1 | `run --pack guard-judge-synthetic --model qwen/qwen3-4b-2507 --session qa-guard-1` | exit 0, stored | PASS |
| 9.2 | `run --pack guard-judge-synthetic --candidate guard-judge-v1-handoff-f1 --session qa-guard-1` (laya not loaded beforehand) | exit 0, stored; `lms ps` afterwards shows `laya@q8_0` loaded: the tool loaded the absent decision model itself | PASS |
| 9.2 | same, with stderr lines `the between-item residency probe before item 'gj-0546' failed (LMStudioUnreachable ...)` (contention) | replicate header `basis assumed`; bars `not assessable (load changed mid-run)` | consistent with the manual's basis text (the probe failures were caused by another client loading LM Studio) |

| 9.2 | second replicate: `lms unload laya@q8_0`, then the same `run --candidate` | first attempt: exit 3, `LM Studio serves an OpenAI-compatible API but not its native /api/v0 catalog` (a catalog call timed out under contention; nothing stored, matches the FAQ's exit 3 description). Retry: exit 0, stored, replicate header `basis by-construction`, FR-1 `met`, FR-2 `not met` (speed ratio 0.70) | PASS |
| 9.3 | `compare --pack guard-judge-synthetic --session qa-guard-1 --models qwen/qwen3-4b-2507,guard-judge-v1-handoff-f1` | exit 0 although a bar is `not met`; one `#### replicate N ... basis ...` block per replicate (1 `assumed`, 2 `by-construction`) and an `#### Across the replicates` block: `FR-1 parity, over 2 replicate(s) (1 load(s)): not assessable (only 1 load replicate(s) (FR-11a needs 2))`, `FR-2 speed-up ... not met`. Matches the state table and the rule that an `assumed` replicate is not counted as a load | PASS |
| 9.3 | what follows the bar in that report | `## Arms` lists only the incumbent; `## Verdicts` reads `None: fewer than two arms were selected ... a comparison needs two stored runs for this pack` | DEFECT D-1 |
| FAQ | `compare`/`rank` over a root holding one valid and two invalid stored records | both exit 0, report written | PASS (partial: the "every record invalid" extreme was not run) |
| FAQ | stale `host.json` (runtimeVersion altered in a scratch copy) | `LM Studio changed since you last attested host.json ... attest`, exit 5 | PASS |

Not run live (cost / environment): steps 5a (the 3-model loop; its `compare`-covers-two-arms claim was verified on stored data in step 7 instead), `join-in-judge` and `message-type-classifier` live runs (compare verified on stored runs only), a `same-load` replicate, a `-handoff-` hand-off failure at exit 3, `--first-call-timeout` expiry, exit 5 from an unreadable end-of-run model list, and `--negative-control` on a plain pack.

## Defects

All are documentation defects unless marked product. None blocks following the manual.

| ID | Severity | Where | Expected (manual) | Observed | Repro |
|---|---|---|---|---|---|
| D-1 | Low (doc gap; the tool wording is also misleading) | Step 9, "Reading the report" | Report is the requirement bar with states `met` / `not met` / `not assessable`; step 7 says a decision-point pack "works differently". | The bar section is as described, but the same report then shows an `## Arms` table with only the incumbent and `## Verdicts` = `None: fewer than two arms were selected ... Check --models and --session ... a comparison needs two stored runs for this pack`. A reader who passed both names correctly is told to recheck them. The manual never says the Verdicts section is empty by design on these packs. | `compare --pack guard-judge-synthetic --session qa-guard-1 --models qwen/qwen3-4b-2507,guard-judge-v1-handoff-f1` (also seen on `join-in-judge-synthetic` / `message-type-classifier-synthetic`). Fix: one sentence in step 9 saying "ignore the Arms/Verdicts sections, the bar is the result"; product wording could be suppressed for decision packs. |
| D-2 | Low | Step 7a, `--reference` | Verdict column is `distinguishable` or `not distinguishable` (plus the two `no verdict` states). | On a boolean pack the Decision column also shows `not tested (Holm stops here)` | `rank --pack guard-judge-understanding --reference qwen/qwen3-4b-2507` (rows for `google/gemma-3-4b`, `google/gemma-4-e2b`). |
| D-3 | Low (unclear) | Step 2 | "If you already know the values you want to set, pass them directly: `--set hostRamGb=16`." | One `--set` still prompts for the other three fields (`lmStudioAppVersion:` prompt appeared and blocked a non-interactive shell); the other key names (`lmStudioAppVersion`, `kvCacheSetting`, `otherResidentWorkloads`) appear only in `attest --help`. Non-interactive use needs all four `--set` flags. | `attest --set hostRamGb=32 </dev/null` blocks; with all four `--set` flags it writes `host.json`, exit 0. |
| D-4 | Info (unverified) | Step 9 intro | "every command below uses one shared `--session` tag, which is required" | The `compare` requirement is confirmed. For `run`, `--session` is optional at argument-parse time (no usage error); whether a session-less decision-point run is later refused was not determined, because the refusals I could trigger came first. | `run --pack guard-judge-synthetic --candidate <c>` without `--session`. |

No defect found in: the changed `run --reference` text, the two-arm verdict rule, `rank` refusal on decision packs, the exit-code FAQ for exits 0/2/3/4/5 (each exercised at least once, except where listed under "Not run live"), the stored-result behavior of `--session`.

## Coverage, gaps, environment notes

- Covered: every command named in the manual (`attest`, `validate` incl. `--strict`, `run` for plain and decision packs, `compare` for plain and decision packs, `rank` incl. `--reference`/`--footprints`, `models --tested`, `index rebuild`), every changed passage (Walkthrough 9, `--reference` ignored, two-arm verdict, exit-code FAQ).
- Not covered: see "Not run live" above. Latency-dependent states (`met` for the new-ability bars) were not reproduced live; `by-construction` was reproduced live on the guard pack.
- LM Studio was shared with other sessions' test runs, so runs were slow (the `guard-judge-understanding` run took 18m51s against about 1 minute historically) and the first replicate came out `assumed`. The manual's advice to unload the decision model between replicates worked, but it says nothing about a busy LM Studio degrading a replicate to `assumed`; worth one FAQ line.
- Environment restoration: LM Studio found with `qwen/qwen3-4b-2507` and `text-embedding-qwen3-embedding-0.6b` loaded; `laya@q8_0` was loaded by the tool during the runs and unloaded again by me; at the end the loaded set matched the one found.
- The repo's `model-bench/results/` (130 runs) and `reports/` (52 files) were unchanged; all writes went to the scratch copy.

## Verdict

PASS with minor documentation findings: every walkthrough step ran as described; 0 high, 0 medium, 3 low (D-1, D-2, D-3), 1 informational/unverified (D-4).
