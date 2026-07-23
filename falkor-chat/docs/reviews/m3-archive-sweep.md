# Review — M3-close documentation-archival sweep (doc-only)

- **Reviewer:** analyst
- **Date:** 2026-07-22
- **Artifact under review:** staged/working (uncommitted) doc-archival change in `falkor-chat/`
- **Baseline:** working tree vs `HEAD` (`bdc6dc6`)
- **Scope:** focused link-integrity + set-correctness verification of a doc-only sweep that moves
  completed-milestone (M3, plus one M1 leftover) planning docs into `docs/archive/` and repaths
  inbound links. Not a broad content audit of the moved docs.
- **Verdict: APPROVE.** 0 blocker / 0 major / 0 minor. Two informational observations below, neither
  attributable to this change.

---

## Method (what was run)

- `git status` + `git diff --staged --name-status` — enumerated the renames.
- `git diff` on every modified live doc (`AGENTS.md`, `BACKLOG.md`, `DESIGN.md`, `HISTORY.md`,
  `docs/plans/local-model-ram-budget-ml.md`) — inspected each link edit.
- `ls`/`find` — confirmed the "stay active" set and the archive destinations.
- Two grep sweeps for dangling OLD-path references (component-wide excluding `docs/archive/`, then
  *inside* `docs/archive/` for broken cross-refs between moved docs).
- Existence check on every `docs/archive/...md` link referenced anywhere in the component.
- Repo-wide grep for external references into the moved paths.

---

## Verification results (against the brief's 7 checks)

1. **Move set is exact — PASS.** `git diff --staged --name-status` shows exactly **19 renames**, all
   `R100` (pure renames, content-identical, history preserved): 12 plans, 5 reviews, 1 test-plan,
   1 test-report — matching the intended set precisely, with no extras. Every "stay active" doc is
   still in its original location: `docs/plans/{demo-environment-bringup,graphrag-eval-ml,local-model-ram-budget-ml,wsl2-memory-diagnostic}.md`
   and all of `docs/requirements/` (`llm-native-workflows.md`, `summary-nodes.md`).

2. **No dangling links to moved docs — PASS.** Grep across all component `.md` for the OLD paths of
   the moved set returns **zero hits outside `docs/archive/`**, and a second grep *inside*
   `docs/archive/` returns **zero broken cross-references between moved docs**. The collision-prone
   pairs were repathed to the correct subdir throughout: `docs/plans/m3-executor.md` →
   `docs/archive/plans/…` and the distinct `docs/reviews/m3-executor.md` → `docs/archive/reviews/…`
   are never conflated (verified in `BACKLOG.md` and `HISTORY.md` diffs, where plan citations land in
   `archive/plans/` and review/gate citations in `archive/reviews/`). The `-ml`/`-impl`/`-landing2`/
   `-coordination` siblings are each repathed to their own target.

3. **Every introduced `docs/archive/...` link resolves — PASS.** All 29 distinct `docs/archive/...md`
   targets referenced across the component resolve to files that exist (includes pre-existing M1/M2
   archives; every newly introduced M3 target is present).

4. **Kept-doc references untouched — PASS.** References to the four stay-active plans still read
   `docs/plans/...` — confirmed for `docs/plans/local-model-ram-budget-ml.md`, which is cited twice
   in `BACKLOG.md` and both citations were correctly *left* unrepathed. That doc was itself modified,
   but only to fix its own outbound links to two now-moved executor docs (a correct inbound-link fix,
   not a move).

5. **Scope discipline — PASS.** The union of staged and unstaged changed paths contains **only `.md`
   files**. No `server/**.py`, no `scripts/*.sh`, no config touched. Bare-filename script/source
   citations (e.g. in `AGENTS.md`'s `seed_workflows.sh` row) are prose, not path links, and were
   correctly left alone.

6. **HISTORY.md sweep entry — PASS.** A dated **`## 2026-07-22 — M3-close documentation-archival
   sweep (doc-only)`** entry leads the file, in the established style: it lists all 19 moves grouped
   by subdir, records the inbound-link-fix count and collision-handling approach, and explicitly
   names the deliberately-kept-active set. It also correctly repaths the moved-doc citations inside
   the *older* HISTORY entries (K-025/K-024/K-022/slice-1) in the same change, so historical entries
   still resolve.

7. **Pre-existing dangling ref — CONFIRMED pre-existing (observation, not a defect of this change).**
   See O-1.

---

## Observations (informational — not blockers, not owned by this change)

### O-1 · Pre-existing dangling reference to a never-created `m3-workflow-def-queries.md`
`docs/archive/plans/m3-workflow-engine.md:254` (a table row) links
`docs/plans/m3-workflow-def-queries.md` — a planned graph-dba query deliverable that was never
created (`find` confirms no such file exists anywhere). **Confirmed pre-existing and untouched by
this sweep:** the identical line is present verbatim in `HEAD` (`git show HEAD:…m3-workflow-engine.md`
line 254), and it appears in the rename diff only as an unchanged *context* line. It is not in the
move set and not introduced here. Flagged only so it isn't mistaken for sweep breakage. If ever
cleaned up, it belongs in a separate content-fix, not this archival move.

### O-2 · Eight cross-component references in `claude/*/kaizen/` now point at moved paths
Outside the reviewed component, these agent-kaizen files still cite the pre-move falkor-chat paths:
`claude/analyst/kaizen/{history,plan}.md`, `claude/architect/kaizen/{history,inbox,plan}.md`,
`claude/coder/kaizen/inbox.md`, `claude/teco/kaizen/{history,k001-run-brief}.md`. These are **out of
this sweep's declared scope** (the module doc convention governs inbound links *within the module*;
cross-component agent logs are separately owned) and most are **point-in-time records**
(`history.md`, `inbox.md`, `k001-run-brief.md`) where preserving the path as-it-was-then is arguably
correct rather than wrong. Surfaced for awareness only; no action required of this change. If desired,
the two forward-looking `plan.md` files (analyst, architect) are the only candidates worth a
follow-up repath, and that is a `claude/`-owner decision.

---

## What's solid

- Renames are pure (`R100`) — content untouched, git history preserved via `git mv`.
- Link repathing is full-path-anchored, so the plans-vs-reviews `m3-executor.md` /
  `m3-process-flow.md` name collisions are resolved to the correct subdir every time — the single
  highest-risk failure mode for this kind of sweep, and it is clean.
- The HISTORY entry is precise and honest (doc-only claim is verifiably true; it even repairs older
  entries' links in the same change so nothing rots).
- Kept-active set is exactly right and left in place; requirements untouched.

---

## Open questions

None blocking. The only decision available is O-2's optional cross-component cleanup, which is not
this change's responsibility.
