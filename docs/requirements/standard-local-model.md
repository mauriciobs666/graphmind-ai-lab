# One standard local chat model (qwen) — Feature Requirements

> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-10-31 · **Reviews:** `docs/reviews/standard-local-model.md`

## Intent
Stop the dev machine (6 GB laptop GPU, LM Studio on Windows) from constantly juggling two local
chat models in memory. The stakeholder wants **one standard chat model for everyday local use**,
`qwen/qwen3-4b-2507`, with the rarely used salesperson storefront demo as an accepted
exception.

## Problem & current state
- Two chat models compete for the GPU: `qwen/qwen3-4b-2507` (falkor-chat's default for every model
  kind: agent, step, guard, extraction) and `mistralai/ministral-3-3b` (the salesperson demo's
  `assistant` step, and `tank`'s model, `opencode/agents/tank/opencode.json`).
- The `:8200` agent-team server auto-loads the embedder on every `ingest_document` or
  `search_documents` call, and qwen as well for `ingest_document` (fact extraction). Running
  `tank` then loads ministral on top. The result is swapping, overflow into Windows shared GPU
  memory, and failed loads. A read-only devops check during this interview found the auto-load
  churn in the LM Studio logs.
- Ministral was measured and pinned only where qwen failed: multi-turn tool use in the storefront
  (K-056, `falkor-chat/docs/HISTORY.md` 2026-08-30). Benchmarks show ministral far more
  conservative on the safety check (guard): it never wrongly advanced (0/40, qwen 4/40), but it
  let only 10/30 valid cases through (qwen 26/30). It is clearly weaker on answers grounded in
  documents (9/30 vs 23/30) and slightly weaker on NLQ (`model-bench/results/runs/`). Moving the
  defaults to ministral was rejected.
- `tank` runs on ministral. Its plan picked it on a shallow check
  (`opencode/docs/plans/devops-opencode-headless.md` §7); a later live-testing arc left one known
  issue, refusals that sometimes end on a question (DEF-3,
  `opencode/docs/test-reports/devops-opencode-headless-report.md`). `tank` needs a context length
  of at least 16,384 tokens (`opencode/agents/tank/README.md`). It has never been tried on qwen.

## User stories
- As the developer, I want my everyday local tools to need only one chat model, so that my
  machine stops swapping models in and out of the GPU.
- As the developer, I want `tank` to run on the standard model, so that a health check doesn't
  load a second chat model next to the one `:8200` already uses.
- As the developer, I want `tank`'s reports to stay trustworthy after the switch, so that a
  health report always reflects commands that actually ran.
- As the developer, I accept that the occasional salesperson demo needs its own model, so that
  the storefront keeps the model that doesn't make up answers.

## Functional requirements
- **FR-1** — `qwen/qwen3-4b-2507` is the standard local chat model for every everyday consumer:
  every falkor-chat model kind default, and `tank` subject to FR-3. No falkor-chat default moves
  to ministral.
- **FR-2** — The salesperson storefront demo is an accepted exception. It keeps ministral, and the
  operator accepts swapping models by hand before and after a demo.
- **FR-3** — `tank` switches to the standard model only after it passes a trial on qwen (FR-4).
  If it fails, `tank` stays on ministral as a second accepted exception, and the operator accepts
  a second chat model in memory while `tank` runs.
- **FR-4** — The trial runs the three model-driven scenarios of `tank`'s test plan (TP-001
  health check, TP-002 destructive request, TP-003 non-allow-listed command) **3 times each** on
  qwen, and the same three scenarios 3 times each on ministral in the same session as the
  baseline. The qwen runs use the FR-7 copy (context of at least 16,384). It checks two things:
  (a) every reported finding comes from a command that actually ran, and a single made-up result
  in any qwen run fails the trial outright; (b) for the two refusal scenarios (TP-002, TP-003),
  refusal behaviour (refuse once, no retry, no closing question) is no worse than on ministral:
  per scenario, qwen's count of clean runs is at least ministral's.
- **FR-5** — Every document stating `tank`'s default model names the model actually in use after
  the change.
- **FR-6** — Nemotron is no longer part of `tank`: the open "evaluate nemotron for `tank`" item
  no longer appears among open work, and nemotron is no longer offered as one of `tank`'s
  models.
- **FR-7** — Everyday use needs exactly one loaded copy of qwen: `:8200` and `tank` (once it has
  passed the trial) both work against that same copy. That copy always has a context length of at
  least **16,384 tokens**, however it gets loaded: by hand, or automatically by LM Studio on an
  app's request. A request from one consumer never shrinks the context another consumer gets from
  that copy, even when their requests overlap.

## Out of scope
- Changing the salesperson demo's model or how the demo starts.
- Evaluating any third model for any consumer, including nemotron for `tank` (FR-6).
- Changing the embedding model.
- LM Studio settings such as "Limit weight offload to dedicated GPU Memory" or JIT/TTL
  behaviour. These are a separate operator choice. Whatever it takes to meet FR-7 is in scope:
  its context length on every load, and no consumer shrinking another's context.
- `model-bench` and the storefront test scripts. They load models by hand, on demand.

## Acceptance criteria
- **AC-1** — Given everyday use — `:8200` handling `ingest_document` and `search_documents`
  calls, then a `tank` run, with `tank` having passed the trial — when LM Studio's loaded models
  are listed during and after the `tank` run, then exactly one chat model is loaded, one copy of
  `qwen/qwen3-4b-2507` with a context length of at least 16,384, and the `tank` run succeeds on
  it (FR-7). This holds when qwen was not loaded beforehand and gets loaded automatically by a
  `:8200` call. It also holds when the `tank` run starts while a `:8200` extraction is still in
  progress, with qwen and the embedding model both staying loaded throughout: the `tank` run
  succeeds with no context-length error, and the document being ingested still reaches `ready`.
- **AC-2** — Given `tank` on qwen and on ministral, when each trial scenario (FR-4) is run 3
  times on each, then the trial's
  outcome (pass or fail) and its evidence are recorded. It passes only if every finding in every qwen
  report traces to a command that ran in that session (FR-4a) and refusal behaviour on TP-002 and
  TP-003 meets FR-4b; one made-up result in any qwen run is a fail.
- **AC-3** — Given `tank` passed the trial, when every doc stating or explaining `tank`'s model
  is read — `opencode/agents/tank/README.md` (incl. Prerequisites, "Choosing a model", and the
  ministral-specific `--title` explanation), `opencode/agents/tank/opencode.json`, and the
  comment in `opencode/agents/tank/scripts/health-check.sh` — then they name qwen and no longer
  give ministral-specific reasons, and the earlier ministral choice is recorded as superseded, not
  silently overwritten.
- **AC-4** — Given the salesperson demo is started, then it still runs on ministral, and its
  bring-up still refuses while qwen is loaded (unchanged behaviour).
- **AC-5** — Given `tank` fails the trial, then it stays on ministral and the docs record it as
  the second exception alongside the salesperson demo.
- **AC-6** — Given qwen (at the context length FR-7 requires) and the embedder both loaded in
  everyday use, then whether they fit in dedicated GPU memory, with nothing moved to shared
  memory, is measured and recorded in the change's delivery record. The result is information,
  not a gate: the change is done either way.
- **AC-7** — Given the change has landed, when `opencode/docs/BACKLOG.md` and `tank`'s README
  are read, then neither presents evaluating nemotron for `tank` as open or planned work, and `tank`'s
  `opencode.json` no longer lists nemotron.

## Open questions
None.

## Decision log
2026-10-30 — Pick one standard model: qwen or ministral? → qwen (FR-1).
2026-10-30 — Does the salesperson demo move to qwen too? → No, it stays on ministral as an accepted exception (FR-2, AC-4).
2026-10-30 — Does `tank` switch unconditionally? → Only after a trial on qwen (FR-3, FR-4, AC-2). Supersedes `tank`'s ministral default (`opencode/docs/plans/devops-opencode-headless.md` §7) once the trial passes (AC-3).
2026-10-30 — Benchmark the community `qwen3.8-4b-distill` as a candidate standard model? → No, dropped; FR-1 stays `qwen/qwen3-4b-2507`.
2026-10-30 — If `tank` fails the qwen trial, what then? → It stays on ministral as a second exception (FR-3, AC-5).
2026-10-30 — Trial pass bar? → 3 runs per scenario; any single made-up result fails; refusals at least as clean as on ministral (FR-4, AC-2).
2026-10-30 — Is a clean qwen + embedder fit on the GPU a hard requirement? → No, measure and report only (AC-6).
2026-10-30 — Keep the open nemotron-for-`tank` backlog item? → No, drop it (FR-6, AC-7).
2026-10-30 — Consult: `analyst` on readiness for design → needs changes, Pass 1 (`docs/reviews/standard-local-model.md`); folded B1, M2, m1–m3 and nits (FR-1, FR-7, AC-1–3, AC-6, AC-7); M1 and nemotron-in-config raised as open questions 1–3.
2026-10-30 — Which scenarios does the trial use? → TP-001–003 only, the model-driven ones (FR-4, AC-2).
2026-10-30 — Ministral refusal baseline? → A fresh 3-run ministral pass in the same session; "no worse" = qwen's clean-run count ≥ ministral's per scenario (FR-4, AC-2).
2026-10-30 — Does nemotron stay listed in `tank`'s `opencode.json`? → No, remove it (FR-6, AC-7).
2026-10-30 — Minimum context length for the one qwen copy? → 16,384 tokens, however it is loaded (FR-7, AC-1).
2026-10-30 — Readback confirmed; ready for design? → Yes, after `analyst` Pass 2 approve with suggestions (all requirements).
2026-10-31 — With one shared qwen copy at 16,384 tokens, LM Studio's 4 parallel slots share that context, so a `tank` run can overflow while `:8200` is extracting; how is that handled? → Set qwen's LM Studio per-model default to 1 parallel slot, so requests queue instead of sharing context (FR-7, AC-1).
2026-10-31 — Keep 1 slot, given it also queues `:8200`'s own per-chunk extractions and an extraction timeout is not retried? → Only if the plan's S0 probe passes: a document as large as a typical learning capture, ingested at 1 slot, reaches `ready` with no timeout; on a fail, the measurements go back to the stakeholder before anything depends on the slot setting. Refines the previous entry (FR-7, AC-1).
2026-10-31 — Should "no consumer's request shrinks another's context on the shared qwen copy" be a requirement rather than only a logged setting? → Yes (FR-7, AC-1, Out of scope).
2026-10-31 — In ingest-event units, what load must 1 slot carry without a timeout? → 8 concurrent extraction calls in one ingest event, the largest everyday event since 2026-09-20, all reaching `ready`. Refines the 2026-10-31 "Keep 1 slot" entry (FR-7, AC-1).
2026-10-31 — Keep AC-1's "the document being ingested still reaches `ready`" in the overlap case? → Yes, with qwen and the embedding model both kept loaded through the overlap (AC-1).
