# Review — `model-bench/docs/manuals/small-model-benchmarking.md` (factual/architectural half)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** —

## Scope & verdict

Scope: the uncommitted `tico` update of the manual (`git diff` against HEAD, Last updated 2026-11-20), checked against `model-bench/docs/DESIGN.md` (§2.3, 2.4, 2.8, 2.9, 3.2, 4.1, 7.1-7.8, 9.4), `FORMATS.md` §5, `README.md`, and the code (`modelbench/cli.py`, `runner.py`, `roles.py`, `decision_stats`/`report` replicate paths), plus `packs/*/pack.json` and `candidates/`. Read-only commands re-run: `./run.sh --help`, `run --help`, `models --help`, `validate --pack packs/guard-judge-synthetic` (and `--strict`), `rank` / `compare` refusals on `guard-judge-synthetic`, `models --tested`. No LM Studio run; no file other than this review written (`git status` on `model-bench/` shows only the manual modified). Not covered: whether the walkthroughs work live (`qa-engineer`). Root and `model-bench/REVIEW.md` read; the LM-Studio-load-state line there decides finding M1.

Verdict: **approve with suggestions** — no blocker; one major that makes the step-9 example unable to reach `met`, plus a header-claim contradiction.

CPG: considered, not relevant — a manual claim check read straight from `cli.py`/`runner.py` and the spec; no impact or data-flow question.

Counts: blocker 0 · major 2 · minor 4 · nit 2.

## Findings

### M1 (major, verified) — step 9's example replicates never reach `met`; the unload advice is correct but buried and incomplete (flagged item a)

Evidence: `cli.py:_replicate_groups` (lines ~318-324): a run is `assumed` if its basis is, else `load` if it is the group's first non-assumed run **or** its decision model (`fingerprint.modelKey`) is absent from `residentModelsAtStart` (`_resident_at_start`), else `same-load`. DESIGN §7.5 says the same; `combine_replicates` needs `>= 2` loads for `met`. DESIGN §4.1: "There is no unload operation" — the harness never unloads; `runner._load_absent_models` loads only what is absent and uses a resident model as found.

So tico's inference is right: a replicate counts as a "load replicate" iff the decision model was freshly loaded by that run, i.e. not resident when the run started. The end-user instruction is therefore: eject the **decision model** in LM Studio before every replicate after the first. The chat (hand-off) model's residency does not enter the kind, so it need not be ejected.

Problem: the manual's own step-by-step block runs `--candidate guard-judge-v1-handoff-f1` twice back to back, with the comment "run it twice or more (replicates, see below)" and no unload. Run as written, the second run starts with the model resident and is `same-load`; the report says `not assessable (only 1 load replicate)` and cannot say `met`. The unload instruction appears only as the last sentence of the "Basis" paragraph, after the reader has finished the walkthrough. The Mermaid diagram (R1 / R2+) omits it too.

Fix: put the step in the command block (a comment line between the two `run`s: "eject `laya@q8_0` in LM Studio so the next run loads it fresh; model-bench never unloads for you"), add it to the diagram between R1 and R2, and phrase the rule as "before each replicate after the first". The `met` table row ("at least two ... separate loads") and the FAQ ("run the candidate again after unloading its model") are already right. Caveat to state: the first replicate counts as a load even if the model was pre-loaded; a later idle auto-unload by LM Studio would also make a run a load, but cannot be relied on.

### M2 (major, verified) — Overview and FAQ still say there is no threshold; step 9 reports requirement bars

Evidence: manual Overview ("it never declares a model 'passing' or 'failing' (no threshold, no gate)") and the FAQ ("There's no 'passing' score in this tool") unchanged, vs. step 9 (`met` / `not met`, 0.90, 350 ms, 2x). `README.md` line 22 reconciles it: "no threshold **of its own**; a decision-point report states the requirement's bar and never acts on it". The manual does say "never acts on it ... exits `0`" in step 9 but never revises the headline promise. A reader of the Overview will be surprised by `not met`.

Fix: reword both sentences to the README's form ("no gate: nothing fails the command; on the three decision-point packs the report states a requirement bar and records whether it was met, but never changes the exit code").

### m1 (minor, verified) — candidate-name suffix is `-only-f<n>` / `-handoff-f<n>`, not `-only-<n>` (flagged item b)

Evidence: `candidates/` listing (`guard-judge-v1-only-f1.json`, `join-in-judge-v2-handoff-f3.json`, …) and `FORMATS.md` §5 (`<point>-<version>-only-f<n>.json`). The claim is right in kind (`only` = no hand-off model, `handoff` = has `handoffModel`; `Candidate` JSON `handoffModel` confirmed in `guard-judge-v1-handoff-f1.json`) but the printed pattern omits the `f`. Fix: "ends `-only-f<n>` ... `-handoff-f<n>`" (the `f` is the fit number; the user needs not know more). The candidate id is the file's `candidateId`, equal to its basename; `models --tested` prints it (verified).

### m2 (minor, verified) — "a chat model other than the pack's incumbent" is refused only on the guard

Evidence: `cli._decision_point_arms`: the refusal is `incumbent is not None and key != incumbent`; `join-in-judge-synthetic` and `message-type-classifier-synthetic` have `"incumbent": null` in `pack.json`, so one chat model of any name is accepted as a reference. The exit-2 list (otherwise matching DESIGN §2.3 item for item) states it generally. Fix: "(on the guard) a chat model other than the incumbent". Same list: all other entries match code (`--negative-control` refusal re-run, exit 2, nothing written; missing `--session`/`--models` re-run, exit 2).

### m3 (minor, verified) — exit 5 at end of run: add that the finished run is lost (flagged item c)

Evidence: `runner.py:2076-2084` raises `RunRefused(exitCode=5)` after three 1 s retries of `lmstudio.residency()`; DESIGN §3.2 step 11 agrees. Cause stated by tico ("loaded-model list couldn't be read at the end of the run") is correct. The raise precedes `store` in `cli._cmd_run`, so nothing is saved: the whole run's scoring is discarded. The manual's "retry once LM Studio is responsive" is right but should say "rerun the whole thing; nothing was saved". The quirk note (message cites `results/transcripts/<runId>.jsonl`, which no run writes) is confirmed: `runner.py:2082` vs DESIGN.md:407 ("`results/transcripts/` is gitignored and no run writes raw model ..."). Accurate.

### m4 (minor, verified) — "as accurate" overstates the guard's parity bar

Evidence: DESIGN §7.5 FR-1: per error-kind cell, a one-sided exact McNemar test that the candidate's error rate is higher; `worse` iff `p <= 0.05`; `met` means "not shown worse", not "as accurate", and requires paired scored n >= 200. The table row says "must be as accurate and at least twice as fast"; the later paragraph says "quality parity". Fix: "must not be shown less accurate (per kind of error) and must be at least twice as fast (measured ratio >= 2.0)". Speed bar (`R = sum x / sum y >= 2.0`, measured, interval never decides), 90 % per class (`k/n >= 0.90`), mean latency `<= 350` ms, fixed 180 s — all verified against DESIGN §7.5, `runner._resolve_request_timeout` (exit 4 on any other value), `report.py` comparison-wide note.

### n1 (nit) — diagram

Mermaid `flowchart LR` in step 9: syntax valid by inspection (quoted labels, `<br/>`, no unescaped brackets); not rendered (`mmdc` not installed). Content matches the described flow (incumbent run only for speed-up packs, replicates, `compare --session --models`, bar report with the three states; heading `## Requirement bar` matches DESIGN §9.4). Gap is the unload step (see M1).

### n2 (nit, pre-existing, not in the diff) — "On-disk data shapes" and `FR-6`/`FR-21`-style references

The `pack.json`/record-field section and bare FR numbers sit uneasily with root `AGENTS.md`'s rule that a manual never covers internal architecture or layout. Unchanged by this update; flag for `tico`'s next pass only.

## Verified items (no finding)

- (d) "synthetic": `pack.json` descriptions and DESIGN lines ~733-741 (`labels.source llm-unreviewed`, `inputs synthetic`, imported from `decision-models`); the caveat wording matches DESIGN §9.4 / line 1030. Pack counts: 8 packs on disk, 5 plain, 3 with `decisionPoint`; "seven roles" matches `roles.ROLES`.
- (e) exit codes: 3 (explicit load failure `runner.py:1590`, not resident after warm-ups `1605`, decision config unreadable `1627-1641`, warm-up timeout `1821`), 4 (candidate refusals `1693-1697`, catalog type, hand-off type/incumbent `1715`, leakage, timeout pin `1789`), 5 (host.json, stale attestation, end residency) all match. Compare exit-2 list per m2.
- (g) `run --reference`: DESIGN §2.9 "accepted and ignored by `run`"; `referenceKey` is set on `RunConfig` and read nowhere else (grep). `rank` on a decision-point pack: re-ran, exit 2 with a `use compare --session ... --models ...` message, nothing written.
- Compare arm selection: newest stored run per `--models` key, unknown key exit 2, no `--models` = every valid run in file-name order, verdict on first two only — `cli._select_arms`, `report._comparison_pair`, DESIGN §2.3. `validate --strict` traceback, exit 1 — re-run, confirmed.
- `run` takes exactly one of `--model` / `--candidate` (`run --help` usage group). Basis mechanics (warm-up re-ask, `assumed` triggers) match DESIGN §7.6.
- No history language in the diff beyond neutral "not implemented yet"; internal layout not added by the diff. README: only M2 differs.

## Open questions

None blocking. Decision made without escalation: treated the unload advice as "keep and move/extend" rather than drop, since spec and code agree it is the only way to produce a second load replicate.

## Pass 2 (2026-11-20)

Re-checked the revised manual against `cli.py`, `runner.py`, `lmstudio.py`, `README.md`, `attest --help`.

Dispositions of Pass 1: **M1 fixed** (eject comment in the step-9 block, diagram node `U`, Basis paragraph "before each replicate after the first"; chat-model state called irrelevant; FAQ line consistent). **M2 fixed** (Overview now "no threshold of its own, no gate ... never changes the exit code"; same meaning as `README.md:22`; no contradiction among Overview, step 9 block, Basis paragraph, diagram). **m1 fixed** (`-only-f<n>`/`-handoff-f<n>`). **m2 fixed** ("on the guard, the only pack with an incumbent"; matches `pack.json` nulls). **m3 fixed** (exit 5 "finished run is not stored": `run_pack` raises before `store` in `cli._cmd_run`, verified). **m4 fixed** ("not shown less accurate"). n1 diagram, n2 unchanged (pre-existing).

Checks requested:
1. Consistency M1/M2: verified by reading all four sites; none contradicts README.
2. "First replicate counts as a load even if already loaded": verified, `cli._replicate_groups` (`"load" not in kinds or not _resident_at_start`). Precision nit: it is the first *non-assumed* replicate (an earlier `assumed` run does not use up that status); harmless to the user.
3. D-4 verified: `--session` is optional on `run` (`RunConfig.sessionId: str | None`, stored top-level as null), and `compare` on decision packs requires it (re-run, exit 2). New FAQ lines: "assumed on a busy machine" is consistent with DESIGN §7.6 and `_between_item_residency` (a failed snapshot that still answers a re-probe reads as "not resident" and degrades to `load`/assumed). "Slow catalog response can give exit 3": **partly verified**, see m5.
4. Exit-5 "not stored": verified (m3).
5. Mermaid by inspection: valid. `C` is referenced before its labelled definition, which Mermaid allows; quoted labels, `<br/>`; no stray brackets. Not rendered (`mmdc` absent).
6. `--set` keys: `lmStudioAppVersion, kvCacheSetting, hostRamGb, otherResidentWorkloads` match `attest --help` exactly, and the on-disk section's four fields.

### m5 (minor, verified by reading) — "a slow catalog response can give exit 3" is only true for some steps

`runner.run_pack` step 2 (`probe()`, catalog budget 10 s) and the between-item residency re-probe refuse with exit 3. But step 3 (`lmstudio.residency()`) and step 3a (`lmstudio.catalog()`, `lmstudio.py:622-629`) are not wrapped; they raise `LMStudioUnreachable`, which is not a `RunRefused`, and `cli._cmd_run`/`main` catch only `RunRefused`. A catalog/residency timeout *after* a successful probe therefore ends as an uncaught traceback, exit `1` (not run live; from reading the two call sites and `main`). Manual fix: say "can give exit `3`, or occasionally a Python traceback with exit `1`; either way nothing is stored — re-run". The unhandled path itself is a code gap (owner: `coder`/`tdd-engineer`, wrap steps 3/3a in the exit-3 refusal); not a manual defect.

Verdict Pass 2: **approve with suggestions** — the manual may be accepted. Counts open: major 0, minor 1 (m5), nit 2 (n1 diagram not rendered, n2 pre-existing).

## Pass 3 (2026-11-20)

Scope: the uncommitted manual diff (20 insertions, 17 deletions) for the five behaviour changes, against `modelbench/cli.py`, `runner.py` and `DESIGN.md` §2.2, §2.3, §2.8, §2.9 (working tree). Source and manual not edited; no contact with the real LM Studio (a local fake HTTP server and a copied-fixture root in the scratchpad). Root and `model-bench/REVIEW.md` applied. CPG: considered, not relevant — manual claim check against the CLI.

Verdict: **approve with suggestions** — no blocker, no major.

Claims, each executed:
- `run --reference y` → argparse "unrecognized arguments", exit 2; `run --help` lists no `--reference`. Verified. `rank --reference` untouched (manual lines 219, 236, 445, 451).
- `validate --strict` → one-line "not implemented ... run `validate` without --strict", exit 2. Verified.
- Plain `compare`, scratch root with real run files: 3 models → exit 2, message names `--models a,b`, no report written (`reports/` empty). 2 models where one has two runs → exit 0, one row per model; qwen shows 23/30, which is the 2026-09-19 run (the older run has 15) — newest wins. 1 model and 0 models → exit 0, report says "fewer than two arms were selected". Verified.
- Start-of-run read failures: fake server answering the probe, then HTTP 500 (and separately a non-JSON 200 body) → `run` exit 3, plain one-line message, nothing stored (0 files in `results/runs`). Verified. A timeout case was not run; `lmstudio._raw_get` folding timeouts to unreachable is taken from Pass 2's reading plus `runner.py:371-409` docstrings.
- Exit-5 message: `grep transcripts modelbench` finds no message mentioning a transcript path; the quirk sentence removal is correct.
- Pass 2 m5 (traceback after slow catalog): disposition **fixed in code and manual** — manual now says exit 3 only; DESIGN §2.2 agrees (steps 3/3a/7/8/9 wrapped, `_start_of_run_read`).

Unfiltered sweep of the whole manual (`--reference`, `traceback`, `transcript`, `first two`, `strict`, `exit 1`, `fewer than two`, `newest`, `now`, `no longer`, `fixed`, `yet`): no passage left false. `first two`, `transcript`, `exit 1`, "not implemented yet" no longer occur. Remaining hits are consistent: line 336 "fewer than two arms" (decision-point report, still true), lines 70/74/91/454 (`no longer`/`right now`, generic prose and pre-existing), line 295 "now eject" (an instruction, not history). Mermaid blocks: 4 before, 4 after, byte-identical (checked against `git show HEAD:`). No internal architecture added; the new history-style words are absent.

### m1 (minor, verified by reading `cli._select_arms`) — "newest" is newest within the session

With `--session`, runs are filtered to the session before the newest-per-model pick (`cli.py:231-254`). Manual lines 169-171 say each model uses "its newest stored run" and only separately mention `--session`. Fix: "its newest stored run (within the session, if you gave `--session`)".

### m2 (minor) — the exit-3 FAQ sentence is long and the "newest" aside leaks implementation

Line 180-181 ("decided by the run files' names ... renamed a run file by hand") describes storage layout; the user-facing fact is that hand-renamed or copied run files can make `compare` pick an older run. Keep the warning, cut the mechanism: "If you rename or copy run files by hand, `compare` may pick an older run." Also line 452-453 grew past the wrapped width (cosmetic).

### n1 (nit) — exit-3 summary, wording

"answering badly when `run` first reads its state" is vague against the FAQ's "gave a bad or timed-out answer when `run` first checked its state"; use the FAQ wording in both.

Growth-stops: the diff's net growth is new scope (the >2-model refusal, the 0/1-arm report, the newest-run rule) offset by deletions of the obsolete quirk notes; no rework growth or restated history found. Counts open: blocker 0 · major 0 · minor 2 · nit 1 (plus Pass 2 n2, pre-existing).
