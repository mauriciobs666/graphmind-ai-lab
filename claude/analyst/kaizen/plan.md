# Kaizen — Improvement Plan: analyst

> Forward-looking backlog for the `analyst` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-09-08 (kaizen_team distillation pass 2, chunks A-F / U19-U24)

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
