# Kiro demo agent — coordination log

> **Status:** active · **Owner:** `teco` · **Tracks:** — (—)
>
> `teco` coordination doc for the **minimal Kiro demo agent** feature (a repo-checked-in Kiro
> agent config that reaches falkor-chat's MCP server as a client, for a live demo).
> Requirements: [`../requirements/kiro-demo-agent.md`](../requirements/kiro-demo-agent.md)
> (Status: Ready for design, no open questions).
> Related, not in scope here:
> [`../requirements/kiro-vision-followups.md`](../requirements/kiro-vision-followups.md)
> (item 4 — this feature is that item's first slice).
> Started 2026-08-01.

## Goal / definition of done

AC-1…AC-4 of the requirements met live:
- AC-1 — typing a message `@mention`-ing `@assistant` in a live Kiro chat session lands it in the
  falkor-chat demo thread (visible in the web UI).
- AC-2 — having the Kiro agent read messages back returns and shows `@assistant`'s reply.
- AC-3 — a fresh clone + falkor-chat set up per its own docs + loading the repo-committed Kiro
  agent config needs no manual MCP wiring beyond what's checked in.
- AC-4 — the demo agent's reachable tool set is exactly `send_message` + `read_messages` — no
  other falkor-chat MCP tool is reachable through it.

Plus doc-curation done-conditions (see scan below).

## Pre-existing state noted at decomposition (teco, 2026-08-01)

- `kiro/` exists today as a single file, `kiro/DESIGN.md` (Status: Draft, predates falkor-chat) —
  **not yet a structured component** (no `kiro/docs/` tree, not listed in root `AGENTS.md`
  Structure or Component-docs tables). This feature is the first real build-out of `kiro/`.
- The two requirements docs for this feature area were filed by `tico` at repo-root
  `kiro/docs/requirements/kiro-demo-agent.md` and `kiro/docs/requirements/kiro-vision-followups.md` —
  **not** `kiro/docs/requirements/…`. Root `docs/` is otherwise scoped entirely to the CPG
  component (its `BACKLOG.md`/`HISTORY.md` headers say so explicitly) — these two files are
  presumably a stopgap filed before `kiro/` had its own `docs/` tree, and likely belong at
  `kiro/docs/requirements/` per the filename-grammar rule (`<component>/docs/<kind>/<slug>.md`).
  **Flagged to `architect` to decide and recommend** (not decided by `teco` unilaterally) —
  relocation, if approved, executes as part of the implementation unit.
- `falkor-chat/` currently has a large **unrelated, uncommitted, in-flight change** (K-034
  re-publish semantics — `services.py`/`executor.py`/`repository.py`/`app.py`/tests/docs). This
  feature makes **no changes to falkor-chat** (explicitly out of scope in the requirements — MCP
  server itself is untouched), so no collision is expected. Every unit's brief says so explicitly
  as a hard constraint: do not touch any currently-modified falkor-chat file.
- `kiro-cli` (Kiro CLI) is installed locally (`kiro-cli`, subcommands `chat`/`agent`/`doctor`/
  `settings`). `~/.kiro/agents/` (global) is empty. `kiro-cli agent create/edit/validate/list`
  exist; `agent list` help states "local agents are only discovered if the command is invoked at a
  directory that contains them" — i.e. there is a project-local discovery mode, which is almost
  certainly how AC-3 ("no manual MCP wiring beyond what's checked in") is meant to be satisfied.
  **Nobody has yet verified the actual on-disk JSON schema or the exact local-discovery directory
  convention against the real CLI** — `kiro/DESIGN.md`'s JSON sample is an illustrative vision
  sketch predating this CLI and must not be trusted as ground truth.

## Documentation-impact scan

- `AGENTS.md` (root) — Structure table + Component docs table need a `kiro/` row (implementer).
- `kiro/docs/` tree — likely net-new (`plans/`, `reviews/`, `test-plans/`, `test-reports/`,
  `HISTORY.md`; `BACKLOG.md` only if architect judges it warranted for a feature this size).
- `kiro/docs/requirements/kiro-demo-agent.md` — candidate relocation to `kiro/docs/requirements/`
  (architect recommends, implementer executes if approved); Status flips `archived` at close,
  owner `tico`, once its own path is settled.
- `kiro/docs/requirements/kiro-vision-followups.md` — item 4 gets a factual update noting this slice
  shipped (owner `tico`); same relocation question applies to this file for consistency.
- No end-user manual judged necessary (a live, manually-run demo config is not an end-user
  product surface) — flagged in the final report as an open call for the stakeholder, not decided
  unilaterally.

## Units

| # | Unit | Owner | Depends on | Deliverable |
|---|---|---|---|---|
| 1 | Design | `architect` | — | `kiro/docs/plans/kiro-demo-agent.md` (or path architect justifies) |
| 2 | Plan review | `analyst` | 1 | `.../docs/reviews/kiro-demo-agent.md`, verdict |
| 3 | Implementation | `coder` | 2 (approved) | agent config + doc/scaffolding changes |
| 4 | Code review | `analyst` | 3 | review of delivered change |
| 5 | Acceptance QA (live) | `qa-engineer` | 3 (parallel w/ 4) | test plan + report, AC-1…AC-4 executed |
| 6 | Doc freeze / follow-up notes | `tico` | 4, 5 pass | requirements Status flips, vision-followups note |

## Status

- [x] Unit 1 — architect dispatched 2026-08-01. **Session crashed before delivery** — no plan
  file was produced. Re-dispatched 2026-08-01 (same brief, fresh run); delivered
  `kiro/docs/plans/kiro-demo-agent.md`, grounded in empirically-verified `kiro-cli` v2.14.1
  behavior (not `kiro/DESIGN.md`'s stale sketch). Recommends relocating both root
  `docs/requirements/kiro-*.md` files into `kiro/docs/requirements/` (§3.5) — flagged for unit 2
  to weigh in on, executes in unit 3 if the review doesn't object.
- [x] Unit 2 — analyst plan review, dispatched 2026-08-01. Verdict: approve with suggestions, no
  blockers (1 Major — a rationale claim contradicted the plan's own evidence, chosen config value
  unaffected — + 2 minors). All three folded back into the plan by `architect` in place. Plan +
  review committed together (`0e20fb1`).
- [x] Unit 3 — coder, dispatched 2026-08-01. Delivered exactly per plan §4 (config byte-matches
  §3.1, README/HISTORY.md/AGENTS.md rows written, requirements docs relocated + cross-refs fixed,
  `kiro-cli agent validate`/`agent list` both green, falkor-chat self-check clean). Verified
  independently by `teco` (byte-diff on the config, live `kiro-cli` re-run, grep re-run) before
  committing (`d33a8af`).
- [x] Unit 4 — analyst code review, dispatched 2026-08-01. Verdict: **approve**, no blockers/
  majors/minors — independently re-verified byte-match, relocation `git log --follow` history,
  live `kiro-cli` re-run, falkor-chat no-touch constraint, doc consistency. Committed (`15f9b6e`).
- [x] Unit 5 — qa-engineer acceptance QA (live), dispatched 2026-08-01. **3 of 4 ACs PASS**
  (AC-1, AC-3, AC-4, live). **AC-2 FAILS** — not a Kiro-side defect, but a real, code-confirmed
  `falkor-chat` bug (D-1, High): MCP `send_message` never schedules the `assistant` responder/
  workflow-trigger background task the REST route does, so a message posted through the Kiro
  agent (or any real MCP client) never gets a reply to read back. Test plan + report committed
  (`3c7ed6d`). **Paused here — see note below, awaiting a stakeholder call before unit 6.**
- [ ] Unit 6 — blocked on the D-1 decision below.

## Open decision (2026-08-01, paused for the stakeholder — not decided by `teco` unilaterally)

QA's finding changes what "done" means for this feature and reaches outside its own build scope
(the requirements explicitly rule out changing falkor-chat's MCP server), so this isn't `teco`'s
call to make silently. Full defect writeup: `kiro/docs/test-reports/kiro-demo-agent-report.md`
§"Defects" (D-1). Options:

1. **Fix D-1 now, as its own falkor-chat unit** (not part of this feature's build), then re-run
   AC-2. Likely `tdd-engineer` (bug fix, reproduction test first) — wire the same
   `BackgroundTasks` scheduling `api.py`'s REST route already does into `mcp.py`'s `send_message`
   (or, per the report's recommendation, move the scheduling into `Services.post_message()` so
   REST and MCP share one code path instead of two policies that must stay hand-in-sync). Delays
   this feature's close but ships a demo that actually round-trips.
2. **Ship the Kiro feature as-is, file D-1 as a new `falkor-chat/docs/BACKLOG.md` K-item**, and
   proceed to unit 6 with AC-2 documented as blocked-not-met, not silently dropped. The demo would
   "hang" after the read-back step exactly as the report warns — acceptable if the near-term need
   is the checked-in config existing, not a fully working live demo yet.
3. Something else the stakeholder prefers (e.g. a narrower same-day patch vs. a fuller
   REST/MCP-unification fix, per the report's two-copies-risk note).

Recommendation: **option 1**, scoped to the minimal mechanical fix (mirror `api.py`'s scheduling
into `mcp.py`), given D-1 is the single most audience-visible failure mode this feature could hit
and the fix is narrow and well-diagnosed already.

## Note on the parallel falkor-chat K-034 work (2026-08-01, post-crash resume)

At the same crash, `falkor-chat`'s K-034 (re-publish semantics) implementation was found
**fully built and self-verified** but uncommitted (`pytest` 691/691, `test_queries.sh` 282/282 —
both match the already-written `falkor-chat/docs/HISTORY.md` entry exactly) with only its
pre-implementation plan review on file, no post-implementation code review. `analyst` was
dispatched for that code-review gate in parallel with this feature's re-dispatched architect unit
— no file overlap between the two threads, confirmed against this doc's own no-collision note
above.
