# Kaizen-team distillation — pass 2, one producing agent per turn

> **Status:** active · **Owner:** `teco` · **Tracks:** — (stakeholder-triggered `cobb` sweep of the shared `kaizen_team` graph) · **Extends:** `docs/plans/kaizen-team-distillation-coordination.md`

## Context

Stakeholder asked (2026-09-18) to distill the shared `kaizen_team` graph again, dispatching `cobb`
per producing agent. Procedure, conventions and lessons are those of the 2026-09-16 sweep this
document extends — read that document's Context and Close-out rather than this one restating
them. Carried forward unchanged: **fresh `cobb` instance per unit**, **strictly sequential** (units
share `claude/cobb/kaizen/history.md` and cross-agent KBs), **`analyst` diff-scoped gate on every
unit before commit**, largest inbox first, `teco` commits each accepted unit by explicit path.

Two conventions corrected against that precedent, both documentation-grammar:

- Gate reviews land in **one** family document, `docs/reviews/kaizen-team-distillation2.md`, one
  `## U<n>` section per unit (root `AGENTS.md` collision rule 2 — the per-unit `-u<n>` basenames of
  the earlier sweep invented a slug per unit).
- `claude/docs/plans/kaizen-distillation2-coordination.md` (the 2026-09 pass that stayed `active`
  under a "keep going indefinitely, re-query before close" instruction) is flipped to `archived`
  at this open: that instruction is honoured by this lineage — each sweep re-queries fresh at open
  and drains what is there — not by an ever-open ledger. Its own close condition (graph at only the
  one deliberate orphan) was in fact observed 2026-09-10 and again by the 09-16 sweep's U7 sweep.

**Snapshot at open (2026-09-18)**, all current-shape, no legacy `author` entries:

| Agent | Produced | Notes |
|---|---|---|
| `analyst` | 14 | 8 are meta-lessons captured while *gating* the 09-16 sweep; 6 are code facts (model-bench, falkor-chat) |
| `architect` | 2 | one is the falkor-chat `CallContext` actor/ws pin — same seam as two `analyst` entries and the `data-scientist` one |
| `qa-engineer` | 2 | |
| `cobb` · `coder` · `data-scientist` · `tdd-engineer` · `teco` · `tico` | 1 each | |
| `MENTIONS`-only orphans | 3 + 1 | `a3f1c8e2…`→`devops`, `c1f3a9e2…`/`a1c2e3f4…`→`graph-dba`; **`e1a6c4d2…`→`tico` is the deliberate survivor (routing signal for `tico` K-016) and is never cleared** |

Total 28. The graph is live — re-query immediately before each dispatch, pin every brief to explicit
`entryId`s, never "everything this agent has". The `analyst` inbox is split by **theme** rather than
date so each chunk has one discard bar and a disjoint target-file set.

**Documentation-impact scan**: per unit, the producing agent's `kaizen/{history,plan}.md`, its
prompt/KB, and — only when a prompt's routing contract or a KB's existence changes — the
`claude/README.md` catalog row / `claude/AGENTS.md` roster; project-docs promotions land in the
component's own tree (`model-bench/`, `falkor-chat/docs/`), never root `docs/`. No `HISTORY.md`/
`BACKLOG.md`/manual in scope (`claude/` has none; the earlier sweep set that precedent).

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `cobb` — `analyst` chunk A, 8 meta-lessons from gating the 09-16 sweep: `b4f6c1a2` `a1e2c3d4` `f4b8c2d1` `7f3c9a2e` `f3d9a1c2` `b7e4a1f2` `e2f1a8c3` `f0e8b2a4` | `a958e5e05212be5a1` | accepted — committed `48e8a84` (8 files, matches) | 8/8 dispositioned: 2 sharpenings `claude/analyst/review-techniques.md` (18,881→19,039), 4 sharpenings `skills/agent-maintenance/SKILL.md` §5 (6,366→6,603; 5 entries, 2 merged), 1 discarded (already published twice), 0 kept open; `claude/analyst/kaizen/{history,plan}.md`, `claude/cobb/kaizen/history.md`. Post-clear: `analyst` 6 (exactly U2's ids), orphans 4 — teco re-derived counts and dup-heading scan | `analyst` (`af5a0203abc6441c1`, 111.3k tok · 28 tools) → **needs changes** (`docs/reviews/kaizen-team-distillation2.md` §U1): 1 Major (§5 step-1 timestamp instrument mis-states the clear→gate→commit window; `date` is day-granular), 2 Minor, 2 Info — all 5 fixed by the same `cobb` (Major applied with its own evidence-backed correction to the reviewer's fix: `createdAt` is itself placeholder-midnight for 5 of 6 survivors); teco re-verified each fix against the tree and corrected one figure in `cobb`'s history (4→5 of 6). Post-fix 19,056 / 6,705 words. **Accepted without a Pass-2 re-gate** (disjoint wording fixes; reviewer pre-stated none needed). The gate's own capture `bb058e98…` (`analyst`, 09-18) is a future sweep's | 160.3k+183.1k tok · 39+10 tools (cobb) · 111.3k · 28 (gate) |
| U2 | `cobb` — `analyst` chunk B, 6 code facts: `b3f1b8b4` `a1f3c9e2-6b7d` `a1e6c9d4` `c7f2a815` (model-bench) · `c7e2a814` `a1f3c9e2-7b4d` (falkor-chat `CallContext`) | `a5fd9879367617dd6` | accepted — committed `49ba441` (9 files, matches) | 6/6 dispositioned: 1 new section `claude/analyst/review-techniques.md` (2 entries merged; 19,056→19,399), 1 invariant paragraph `model-bench/AGENTS.md` (1,950→2,035 — written so U6's `tdd-engineer` entry folds onto it), 1 kept open in `model-bench/docs/BACKLOG.md` (the one remaining unguarded `validate_pack` axis), 2 merged into `falkor-chat/docs/SERVER.md` §1.3 (8,829→8,955 — U3's `architect` entry belongs as a clause on its "same actor" sentence); 4 model-bench *instances* found already fixed (U163/U170/U175/U177), only technique/invariant forms promoted; `claude/analyst/kaizen/{history,plan}.md`, `claude/cobb/kaizen/history.md`. Post-clear: `analyst` 1 (`bb058e98…`, out of scope) — teco re-derived word counts, 7-file diffstat, dup scans | `analyst` (`af430950cce150ef7`, 125.1k tok · 27 tools) → **approve with suggestions** (§U2): 3 Minor (history axis count 2→3; SERVER.md "every REST and MCP call" contradicts the storefront sentence below it; BACKLOG quotes a contract not in the source), 5 Info — 2 are sequencing notes: U3 must land after/around the concurrent `mcp.py` `produced_by` work; U6 must discard the `tdd-engineer` entry's `rfind` advice (superseded by U177). All 3 Minors + 2 file Infos fixed by the same `cobb`, each re-derived; teco re-verified (axis count 3 at `packs.py:672/890/930`, SERVER.md single-hunk, 8,971 / 1,038 words). **Accepted without a Pass-2 re-gate** (one-sentence disjoint fixes; reviewer pre-stated none needed). Pre-existing drift noted by the gate, not chased: `model-bench/docs/BACKLOG.md` L34-39 is a delivered item (`_scorer_problems`) still in the backlog — model-bench closeout list | 197.5k+216.4k tok · 50+9 tools (cobb) · 125.1k · 27 (gate) |
| U3 | `cobb` — `architect` (2): `a1e6d9f4` `a1f3d9c2` | `a3014a07293725567` | accepted — committed `2513f0e` (5 files, matches) | 2/2 dispositioned: `a1e6d9f4` **discarded** (fixed same day by `9ef89d7` — `_render_speed` wired into `compare_report`, in `model-bench/docs/HISTORY.md`; "13 fields" enumerated, not trusted — teco's own first count of 10 was a bad regex, class has 13 field lines); `a1f3d9c2` **promoted** as one sentence onto U2's §1.3 paragraph in `falkor-chat/docs/SERVER.md` (8,971→9,076, single hunk), worded at `49ba441` only — no reference to the concurrent `produced_by` work. `claude/architect/kaizen/history.md`, `claude/cobb/kaizen/history.md`. Post-clear `architect` 0 | `analyst` (`a297245519ae85a57`) → **approve with suggestions** (`docs/reviews/kaizen-team-distillation2.md` §U3): 1 Minor (the promoted sentence's middle clause restates `agent-knowledge-base-strategy.md` §4.1's *design decision* as a code fact, and reads as contradicting §1.3's own "only `get_context` changes" line two sentences below), 2 Info (1 pre-existing/out-of-scope: §2.2's tool table omits `ingest_document`), 2 nits (a line-number off-by-one in `architect`'s history bullet; a run-on sentence). `LatencyBlock` reconfirmed at 13 fields by direct read (`callAttemptedCount` is a `@property`, the likely 10-vs-13 trap). **Gate task's own status notification reported `failed` (HTTP 429) with a stale mid-task `<result>`, but the relayed report was fully detailed and the on-disk diff (`git diff 49ba441 --stat`) shows a clean 177-line pure-insertion §U3 section ending in a well-formed close, matching U1/U2's structure exactly — teco treated this as conclusive evidence of a genuine, complete deliverable (write succeeded; the 429 hit on a later, non-durable call) rather than re-dispatching.** No Pass-2 re-gate needed (reviewer pre-stated so). Fix round (same `cobb` instance, resumed by `SendMessage`): reworded the Minor's clause to state only present-tense code facts (no call through `get_context` reads per-caller identity off `ctx.actor`; `Storefront.context_for` is the one existing precedent), leaving §4.1 sole owner of the "must be additive" design ruling; both nits applied; teco independently re-verified the diff, word count (8,971→9,066), id-occurrence counts (each id once in `architect`'s history; `a1f3d9c2` twice in `cobb`'s — one is U2's own pre-existing forward-reference, confirmed legitimate), dup-heading scans (0), and post-clear graph (`architect` `PRODUCED` → 0) before committing | 159.2k tok · 38 tools · 377s (cobb, U3 run) · 172.1k tok · 12 tools · 156s (cobb, fix round) · gate cost unrecorded (task notification reported `failed` before usage; deliverable independently confirmed complete on disk) |
| U4 | `cobb` — `qa-engineer` (2): `c2e40890-5f35-45cb-9c6c-9501f93e3959` `5a2b5130-8c8a-4ca3-af19-e16f1dbec024` | `ade8892d1c5818db7` | gated (pending) | 2/2 dispositioned: `c2e40890` **promoted** as 2 paragraphs onto `model-bench/AGENTS.md`'s "Load-bearing invariants" (2,035→2,244) — one claim corrected on re-derivation: the entry's (and the S7 test-report's own) "`--negative-control` always picks newest" claim is false without `--models`; `cobb` ran `_select_arms` live and found it's the *oldest* record by ascending filename sort, newest only when `--models` dedups — teco independently re-derived this from `cli.py`/`results.py` source and confirms; `5a2b5130` **promoted** as a new section in `claude/qa-engineer/qa-testing-techniques.md` (3,027→3,210) — the 24/30 (0.80) estimate vs. 23/30 (0.767) measured figures teco cross-checked exactly against `model-bench/docs/HISTORY.md`'s S7 close-out and the S7 test-report. `claude/qa-engineer/kaizen/history.md`, `claude/cobb/kaizen/history.md`. Post-clear `qa-engineer` 0 (teco-reverified). Dup-heading scans clean on all 4 files | `analyst` (`a0190e96227af7763`) → pending | 207.5k tok · 61 tools · 839s (cobb) |
| U5 | `cobb` — `data-scientist` (1): `7a3e9c1b` | — | queued | — | `analyst` → — | — |
| U6 | `cobb` — `tdd-engineer` (1): `a1f3c2e4` | — | queued | — | `analyst` → — | — |
| U7 | `cobb` — `coder` (1): `c2b1f6b4` | — | queued | — | `analyst` → — | — |
| U8 | `cobb` — `teco` (1): `b2f6a1d4` | — | queued | — | `analyst` → — | — |
| U9 | `cobb` — `tico` (1): `f3d8a1c2` | — | queued | — | `analyst` → — | — |
| U10 | `cobb` — `cobb` self-produced (1): `9f2b6e1a` | — | queued | — | `analyst` → — | — |
| U11 | `cobb` — the 3 `MENTIONS`-only orphans (`devops` 1, `graph-dba` 2), one unit as the 09-10 pass's orphan-backlog precedent | — | queued | — | `analyst` → — | — |
