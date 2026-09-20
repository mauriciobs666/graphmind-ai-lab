# Kaizen — Improvement Plan: analyst

> Forward-looking backlog for the `analyst` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-09-18 (kaizen-team distillation pass 2, units U1–U2 — both chunks of `analyst`'s inbox)

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-09 | med | 🟡 | Shakedown — RCA mode only remaining: plan-review ✅ (2026-07-11) + code-review ✅ (K-022 impl review, 2026-07-12); RCA run still open |
| K-002 | 2026-07-09 | low | 🔵 | Reciprocal mentions in producer prompts (architect/coder) |
| K-003 | 2026-08-24 | low | 🔵 | Progressive disclosure: move the evidence-traps *mechanisms* to `review-techniques.md`, keep trigger stubs |

### K-001 — First-run shakedown: RCA mode remaining
- **Status:** 🟡 in-progress — plan-review ✅ + code-review ✅; **RCA mode only remaining**
- **Priority:** medium
- **Rationale:** The prompt is untested against a live run. The likely weak spots: verdict calibration (does it rubber-stamp or nitpick-flood?), whether it actually runs suites for evidence, and whether the review doc lands at `docs/reviews/<slug>.md` with the hook staying silent. Two of the three review modes have now cleared these on real artifacts; the RCA mode has not run.
- **Proposed change:** Run an **RCA of a real (or seeded) failing test** end-to-end — assess whether it delivers a clean causal chain + suggested fix at `docs/reviews/<slug>-rca.md`, hook silent; fold any verdict/structure findings back into the prompt. Then close K-001.
- **Progress:**
  - **Plan-review ✅ 2026-07-11** — `falkor-chat/docs/archive/reviews/m3-executor.md` (K-022 design review; majors M1–M4 raised and closed into the approved plan; right path, hook silent).
  - **Code-review ✅ 2026-07-12** — `falkor-chat/docs/archive/reviews/m3-executor-impl.md` (K-022 impl review; approve-with-suggestions, 0 blockers / 1 major / 3 minor / 3 nit; calibration healthy). Counterpart to teco K-003 (now closed). See history.md.
  - **RCA ⬜ open** — no RCA run yet; this is the sole remaining piece of the shakedown.

### K-002 — Reciprocal mentions in producer prompts (architect/coder)
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** The analyst names the owners it routes findings to (coder/tdd-engineer/architect/qa-engineer), but no producer prompt mentions the analyst as an available review gate — teco's roster is currently the only router. Fine while teco mediates everything; worth revisiting if plans/code should advertise "reviewable by analyst" themselves.
- **Proposed change:** If review gates become a standing part of the pipeline, add a one-line mention in architect's handoff section (plan may be routed through analyst) — keep it minimal to avoid roster sprawl in specialist prompts.

### K-003 — Progressive disclosure for the evidence-traps list
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** Opened out of the C3 compression unit (`claude/docs/plans/prompt-waste-reduction.md`), which measured this prompt at its **editorial floor**: after Stage B and C3, the residual class-6/7 inventory across the whole file is under 25 words. The evidence-traps sub-list plus the pre-existing-deliverable bullet is ~350 w of a ~2,300-w body, and none of it is narrative waste — it is distilled class-3/4 lesson payload, so no further prose editing can reach it. The remaining lever is structural, and this file already has the affordance: `claude/analyst/review-techniques.md`, loaded on demand. The stronger argument is **consistency, not tokens** — `review-techniques.md` already holds entries of the same genre (the `pytest -k`/`-m` marker trap, live-service reachability before trusting a live-test report), so near-identical material is currently split across two locations with no stated criterion for which goes where. That, more than the word count, is what will keep generating drift. Secondary benefit: entries there carry `Origin:` provenance blocks, legitimate in an on-demand file and forbidden in the always-loaded prompt — so a moved trap keeps the evidence the doctrine forces it to strip.
- **Proposed change:** Move each trap's *mechanism and consequence* into `review-techniques.md`; keep a one-line **trigger stub** for each in the prompt, pointing there. **The trigger stubs are non-negotiable** — all six traps were checked and each has a trigger recognizable from the task surface without already knowing the trap's content ("I'm about to cite a grep count as a baseline", "this plan prescribes a check command", "this doc contains a hold note"). The tempting failure mode is a single vague pointer ("consult `review-techniques.md` for evidence traps"), which requires recognizing a trap as a trap in order to know to load the file that names the traps — circular, and it fails silently. Estimated net ~230 w (~9%) at the cost of one on-demand read per triggered review. Also settle the split criterion between the two files while in there.
- **Blocked on:** nothing, but it is the analyst-side analogue of `K-016` progressive disclosure and deliberately out of scope for the prompt-waste plan (its finding 6 routes floor-bound files here rather than to cutting rules to reach a number).

## Parking lot / ideas
- **No `analyst` backlog item opened by the 2026-09-20 sweep (U1 of
  `docs/plans/kaizen-distillation3-coordination.md`, all 16 remaining `analyst` `kaizen_team`
  entries).** 7 promoted, 9 discarded as already documented elsewhere (verified, not just cited);
  nothing kept open, so nothing needs a `K-`number. Dedup check run on all 16 `entryId`s — none
  appears anywhere in this file. Dispositions: `kaizen/history.md`, 2026-09-20. This sweep also
  drains `analyst`'s queue in `kaizen_team` to zero — see that history entry for the graph-clear
  confirmation; `analyst`'s learnings capture moved to `ws:agent-team` team-wide since 2026-09-19,
  so no further entries are expected here.
- **No `analyst` backlog item opened by U2 of the 2026-09-18 sweep (6 code facts).** Five promoted
  (two merged into `review-techniques.md`, one into `model-bench/AGENTS.md`, two merged into
  `falkor-chat/docs/SERVER.md` §1.3); the one kept-open item — `validate_pack`'s
  `_answerability_stamp_problems` still crashing on a not-yet-authored `items.jsonl` (raw entry
  `b3f1b8b4-6f3c-4b6a-9a1a-2f0f9a2b6d31`) — is tracked in `model-bench/docs/BACKLOG.md`, that
  component's own backlog, not here. Dedup check run on all six `entryId`s — none appeared in this
  file before this line was written. Dispositions: `kaizen/history.md`, 2026-09-18 (chunk B).
- **No `analyst` backlog item opened by U1 of the 2026-09-18 sweep (8 meta-lessons from gating the
  09-16 sweep).** Seven promoted (two here, five into `agent-maintenance` §5), one discarded as
  already published; nothing kept open. Dedup check run on all eight `entryId`s — none appears in
  this file. Dispositions: `kaizen/history.md`, 2026-09-18.
- **No `analyst` backlog item opened by U37 (2026-09-09, 5 entries).** Four promoted, one
  discarded as already published; nothing unresolved, so nothing needs a `K-`number. Dedup check
  run on all five `entryId`s — none appears in this file. Dispositions: `kaizen/history.md`,
  2026-09-09.
- **Idea, not a defect: "a done-condition that cannot fail" rules now sit in two agents' knowledge
  bases with no stated criterion for which.** `review-techniques.md` holds the median-latency rule
  (U37) and the grep-residual rules; `qa-engineer`'s `qa-testing-techniques.md` holds the
  survivor-label rule, under a catalog line that calls that file *environment/tooling techniques*.
  U37 ruled its own entry by activity — the rule fires where a done-condition is **written or
  gated**, not where a test is executed — which is a usable criterion but is currently recorded
  only in a history entry. Worth stating in whichever file is touched next, or in `claude/AGENTS.md`
  if the two catalog lines are ever revised together.
- **No `analyst` backlog item opened by the 2026-09-08 distillation (U19-U24, six chunks, 69
  entries).** Every entry resolved to a promotion, a discard, or — once, in U24 — a kept-open item
  filed against **another** agent's plan, so nothing here needs a `K-` number. Dedup check run as
  required for each; none of the 69 `entryId`s appears anywhere in this file. Dispositions:
  `kaizen/history.md`, 2026-09-08.
- **One U24 finding routed outward, tracked as `graph-dba` K-009, not here.**
  `skills/joern-cpg/scripts/pipeline.sh`'s `rq()` helper (the prefix `case` at line 304) returns
  **0** on a bare FalkorDB runtime-error reply, so a call site with no expected-substring argument
  reads a failed query as success — confirmed by executing the helper verbatim against a live
  graph. The general reply-shape behaviour is published in `claude/graph-dba/falkordb-quirks.md`;
  the code fix is `graph-dba`'s, outside `cobb`'s write remit. Raw entries
  `b7f3c2a1-9d4e-4c11-8a52-6e0f1d3b7c94` and `4f9c21ae-7b30-4d62-9c18-6ea5d0b73c41`, both tagged
  `MENTIONS`→`graph-dba` and alive on that edge until its pass runs.
- **A second finding routed outward, not tracked here (U21, 2026-09-08).** `falkor-chat/server`'s
  pytest suite **cannot be parallelised across processes** — `tests/conftest.py:86-91`'s `conn`
  fixture wipes the single shared `ws:test` graph (`MATCH (n) DETACH DELETE n`) at *setup*, once per
  test, and `repo`/`wf_repo` both depend on it, so two concurrent runs destroy each other's fixtures
  mid-test and produce large, scattered, non-reproducible failure counts. This hazard is **absent
  from `falkor-chat/docs/SERVER.md` §1.7**, which is where that component's testing hazards live and
  which already documents the sibling `wf_repo` `reference`-graph wipe. The reviewer-facing half is
  published (`review-techniques.md`, folded onto the pytest-plugin ablation section); adding the
  component-facing half to §1.7 is `falkor-chat`'s owner's call, not `analyst`'s or `cobb`'s.
- **One finding routed outward, not tracked here:**
  `falkor-chat/docs/plans/oversized-indexed-property-guard-graph.md:205` (`Status: active`, owner
  `graph-dba`) still cites `grep -n RELATIONSHIP scripts/bootstrap_schema.sh` → "no matches" as
  evidence, which `8d7dcfb` falsified on 2026-08-24 (`bootstrap_schema.sh:265`). The document's
  owner, not `analyst`, decides whether the surrounding conclusion survives. Chunk B added nothing
  to it — no entry in that chunk touches `RELATIONSHIP` constraints or that document.
- **Re-review vs. `## Pass N` (noted 2026-07-27).** The doc convention (`docs/plans/doc-reference-convention.md` §9.5 rule 5) now rules that a second review of the *same* artifact is a dated `## Pass N` section appended to the existing review, not a new file — which is exactly the "re-review mode" idea below, now with a house rule behind it. If that mode is ever written into the prompt, it must produce `## Pass N`, and the ordinal-on-the-role escape (`x-impl2.md`) is explicitly withdrawn.
- A severity rubric calibrated on real reviews (examples of blocker vs major from this repo) once a few reviews exist — only if verdicts prove inconsistent.
- Re-review mode: given a prior review doc + a revised artifact, verify each finding was addressed and append a dated re-review section instead of writing a fresh doc.
- **Two findings routed outward, not tracked here (2026-09-16 distillation, scoped to `analyst`'s
  `kaizen_team` entries).**
  - Root `AGENTS.md`'s own prescribed budget-check command
    (`awk 'length($0)>700{print FILENAME": "NR}' $(git ls-files '*AGENTS.md')`) misreports line
    numbers across multiple files because `awk`'s `NR` is cumulative across every file `awk`
    processes, not per-file — `FNR` is needed instead. Verified: re-ran the documented command
    verbatim against all `*AGENTS.md` files (2026-09-16) and it printed line 698 for a length-980
    line actually at line 83 of `falkor-chat/AGENTS.md`; re-run with `FNR` gave the correct
    location. (Re-checked again at U1's re-verification, same day: still 698 — the cited number
    tracks the current tree, not a stale capture-time snapshot.) Root
    `AGENTS.md` is outside both `analyst`'s and `cobb`'s write remit (cobb's guard allowlists
    `claude/AGENTS.md`, not the repo-root file) — whoever next touches that file's Context-file
    convention bullet should swap `NR` for `FNR`. No action needed from `analyst`.
  - `falkor-chat` `services._drive_or_fault` (used by `start_workflow_run`/`submit_workflow_input`/
    `sweep_due_workflow_runs`) swallows four named exception types into a normal
    `{"status":"failed"}` return, but `services.resume_workflow_run` bypasses `_drive_or_fault`
    entirely (calls `executor.resume` directly) and lets the same exceptions propagate —
    re-verified directly against the current `services.py` (`resume_workflow_run`, `_drive_or_fault`)
    2026-09-16: confirmed true. So a caller that only catches exceptions (not inspecting return
    values) is safe on the resume path but silently blind on start/submit-input/sweep, and
    separately blind to `_fail_budget` (always a normal return, never a raise, on every path
    including resume). Worth a line in `falkor-chat`'s own exception-handling documentation
    (`SERVER.md` or `DESIGN.md`); outside `analyst`'s and `cobb`'s write remit — `falkor-chat`'s
    owner's call, not tracked here.
