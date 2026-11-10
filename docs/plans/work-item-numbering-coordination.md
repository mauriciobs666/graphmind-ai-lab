# Work-item numbering and terminology — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** — · **Item:** `GAL-F0001`

Requirements: `docs/requirements/work-item-numbering.md` (Ready for design; analyst review `docs/reviews/work-item-numbering.md`, 3 passes). Chain so far is docs-only (design → review); verified deliverables are held uncommitted and land in one commit at the chain's terminal state. Build units (allocator/lookup/check tooling, skill) are drawn after the plan gate.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 design plan | `architect` | `a7af14139c4b90236` | accepted (rev 3; 3615 words) | `docs/plans/work-item-numbering.md` | `analyst` → — | ~105k+136k tok (rev), 28 tool uses |
| G1 plan gate | `analyst` | `ad7deab2cf57f6c0f` | accepted | `docs/reviews/work-item-numbering.md` (new Pass) | Pass 4: needs changes (1 blocker, 5 major); Pass 5: needs changes (narrow, 3 major); Pass 6: approve with suggestions | — |

## Checkpoint 2026-11-09 (session paused at usage limit)

- State: U1 revision 3 in-flight (`architect` `a7af14139c4b90236`) against `docs/reviews/work-item-numbering.md` §Pass 5 (needs changes, narrow). Done-condition = Pass 5's stopping condition: fix N1, N2, N3 + the two growth cuts → approve with suggestions, no further pass; n1–n4 fold without re-review.
- On resume: `git status` / `wc -w docs/plans/work-item-numbering.md` (was 4549 before rev 3; expect it to fall). If the plan is unchanged since Pass 5, rev 3 did not land.
- Resume path: `SendMessage` `a7af14139c4b90236`; **cold-start fallback** (agentId may not survive a session reboot): fresh `architect` dispatch with a state-recovery brief — read plan + `docs/reviews/work-item-numbering.md` §Pass 5, apply N1–N3, growth cuts, n1–n4, in place.
- Then: verify N1–N3 by the commands in Pass 5 (re-run, don't trust), send a compact confirmation to `analyst` `ad7deab2cf57f6c0f` (cold fallback: fresh `analyst`, brief = Pass 5 stopping condition) → on approve-with-suggestions, mark G1 accepted, draw build units from the plan's 13-step table (steps 1–8 `tdd-engineer`, 9–10 `cobb`, 11 `teco`, 12 `coder`, 13 `teco` routing). Build chain is code → commit per unit from then on.
- Uncommitted docs-only set (commit together by explicit path at chain end or handoff to build): `docs/plans/work-item-numbering.md`, `docs/plans/work-item-numbering-coordination.md`, `docs/reviews/work-item-numbering.md`.

- Update: rev 3 landed (architect `a7af14139c4b90236`, plan 3615 words, was 4549). Next: send compact confirmation to analyst `ad7deab2cf57f6c0f` (Pass 5 stopping condition: N1–N3 + growth cuts); not yet re-verified by teco or analyst.

- Update: Pass 6 approve with suggestions; design chain (U1/G1) closed. Nits to fold in the build (no re-review): move 3.1's duplicate example into test 6; SKILL.md flush rule: `wi flush` runs only after a merge completes. Build units (plan steps 1–13) not yet drawn — awaiting stakeholder go-ahead to implement.

## Build units (plan steps; code chain — commit per verified unit by explicit path; tree carries unrelated in-flight changes from other sessions, so commits are pathspec-only under `skills/work-items/`)

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| B1 (plan step 1) | `tdd-engineer` | `ac7238baf44ef7314` | accepted (committed; diff gate batched) | `skills/work-items/scripts/wilib/{rules,events}.py` + tests | `analyst` (diff gate, batched per ≤3 steps) → — | ~103k tok, 16 tool uses |
| B2 (step 2) | `tdd-engineer` | `a6cfead2df12dcbe8` | accepted (committed) | `skills/work-items/…` | — | ~95k tok, 11 tool uses |
| B3 (step 3) | `tdd-engineer` | `ac3a40cd64f27380a` | accepted (committed) |
| B4 (step 4 + CLI `flush`/`init` gap from step 3) | `tdd-engineer` | `a798c99c101250269` | accepted (committed) |
| B5 (step 5) | `tdd-engineer` | `adca059df71547b9f` | accepted (committed) | `skills/work-items/…` | — | ~91k tok, 12 tool uses |
| G2 diff gate steps 1–5 | `analyst` | `ad341f66ca9cc17cb` | delivered | `docs/reviews/work-item-numbering.md` Pass 7 | approve with suggestions (F1 resolve FR/AC parent defect, F2 silent-corruption test gap, F3 links shape decision) | — |
| B5b fixes F1/F2/minors | `tdd-engineer` | `a798c99c101250269` | accepted (committed 28ce49c5; ~132k tok) | `skills/work-items/…` | re-confirm by `analyst` after step 7 | — |
| U1b plan sync + F3 decision | `architect` | `a7af14139c4b90236` | delivered (F3: concerns functional; fixed-by/relates multi-valued via link op; step 6 split into 6 + 6b) | `docs/plans/work-item-numbering.md` | — | — |
| B6 (step 6 links/attrs) | `tdd-engineer` | `acbba0473b46da2aa` | accepted (committed; ~86k tok) | `skills/work-items/…` | — | — |
| B6b (step 6b headers/options) | `tdd-engineer` | `a0b298174a600c561` | accepted (committed; ~91k tok) | — | — | — |
| B7 (step 7 rebuild/check) | `tdd-engineer` | `aa72975dda9e9acd9` | accepted (committed; ~126k tok). Carry-forward: step 12 must populate `project.json` `legacy_paths` (pattern→home, `$1` capture) — init doesn't write it; check.py uses private Store/headers members (follow-up: public `Store.read_events()`) | `skills/work-items/…` | — | — |
| B8 (step 8) | `tdd-engineer` | `aaba6a1fbbdbf501f` | accepted (committed; ~75k tok) | keys | — | — |
| G4 final impl gate steps 7b–12 + close readiness | `analyst` | `ad341f66ca9cc17cb` | accepted | `docs/reviews/work-item-numbering.md` Pass 9 | approve with suggestions. P9-1 (suite red after GAL claim) fixed by teco (trivial single-test edit; 500 pass; mutant — duplicate key in keys.json — killed). Open: P9-2 rebuild guard runs after reseed reads stale ref tips (minor, follow-up); P9-3 symlink/PATH machine step; P9-4 bare `K-` can never resolve (nit); N5 public `Store.read_events()` follow-up |
| G3 diff gate steps 6–8 (+ re-confirm Pass 7 F1/F2) | `analyst` | `ad341f66ca9cc17cb` | delivered | `docs/reviews/work-item-numbering.md` Pass 8 | approve with suggestions (N1 major: rebuild can reissue numbers with `check` clean; N2–N5 minor) | — |
| U1c N1 decision + 7 stale plan statements | `architect` | `a7af14139c4b90236` | delivered (N1: git-history recovery + --floor + --accept-reissue guard + `rebuilt` marker = step 7b; N2/N3 = step 8b, already with B8b) | `docs/plans/work-item-numbering.md` | — | — |
| B8b N2/N3/N4 fixes | `tdd-engineer` | `a798c99c101250269` | accepted (committed c321abea; ~162k tok). One of my mutants (own-uid exemption in `check_slug_free`) survived — likely equivalent since `set_attr` short-circuits an unchanged value; unverified. Plan §3.9 to reconcile: init refuses reserved/malformed keys only (cannot tell a key claimed for this repo from another's); retired items keep slugs; `.resolve()` in `keys_path()` is load-bearing | `skills/work-items/…` | — | — |
| B7b N1 rebuild-reissue guard | `tdd-engineer` | `aaeb2eea10f793eae` | accepted (committed; ~102k tok; my mutant reversing history-commit order survived — equivalent: union by evid + fold sorts by (ts,evid)) | `skills/work-items/…` | — | — |
| B9 (step 9 SKILL.md/glossary/catalog/drift test) | `cobb` | `aaaccc46fa9e0361a` | accepted (committed; ~84k tok) |
| B10 (step 10 prompts + claude/AGENTS.md) | `cobb` | `a6160aa3331ab4172` | in-flight — touches shared files (`claude/AGENTS.md`, agent prompts): do three-way diff check before commit | — | — |
| B11 (step 11) | `teco` (trivial 2-line edit, done directly) | — | accepted (committed) | — | — |
| B12 (step 12 bootstrap GAL) | `coder` | `a6aa601306d99e78c` | accepted (committed 2cc4c4ee; ~66k tok; I re-ran `wi check` rc 0, resolve, header diffs = 1 line per doc) |
| B13 (step 13 follow-ups) | `teco` → stakeholder | — | handed off: (a) `tico` amends `claude/docs/requirements/team-engagement-in-other-repos.md` AC-1 so the kickoff set includes `.work-items/` and `.gitattributes` (`tico` is user-run, not a delegation target); (b) cross-initiative plan author decides whether `needs` cites scheme IDs and confirms its parser treats them as opaque strings; (c) machine step: `ln -s` the skill into `~/.claude/skills/` (human/`devops`) | — | — |

## Checkpoint 2 (usage limit reached)

- Committed: steps 1–6 + Pass 7 fixes (latest code commit `5e8076e9`, 360 tests). G2 (analyst Pass 7 steps 1–5) approve with suggestions; F1/F2 fixed, F3 decided.
- In flight: B6b `tdd-engineer` `a0b298174a600c561`. On resume: `git status --short skills/work-items`, run the suite (`cd skills/work-items && /home/mauricio/prg/graphmind-ai-lab/model-bench/.venv/bin/python -m pytest tests -q`), mutate one argument, commit by explicit path. Cold fallback: fresh `tdd-engineer`, brief = plan step 6b row.
- Queued: B7 (step 7 rebuild/check), B8 (step 8 keys), analyst diff gate for steps 6–8 (also re-confirm F1/F2; step-4 mutants were not re-run), B9–B10 `cobb` (shared files: three-way diff check), B11 root AGENTS.md header fields, B12 bootstrap `GAL`, B13 follow-ups (tico AC-1 amendment; cross-initiative `needs` parser question).

## Close status (2026-11-10)

Build complete: steps 1–12 committed, 500 tests green, `wi check` clean on the real repo. NOT yet closed. Open items for the stakeholder: (1) spec home — `skills/work-items` has no docs tree and no live spec; root AGENTS.md requires a whole-component spec checked by `analyst` before the first close (where, and who owns it?); (2) HISTORY entry home — `skills/` has no `docs/`, root `docs/HISTORY.md` is cpg's; (3) which review document is the close check (Passes 7–9 live in `docs/reviews/work-item-numbering.md`); (4) step 13 handoffs (tico AC-1 amendment; cross-initiative `needs` parser; skill symlink into `~/.claude/skills/`); (5) follow-ups P9-2, P9-4, N5; (6) archived-status flips after close.

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| C1 symlink machine step | `devops` | `a0471854ac25bf516` | accepted (I verified the link exists; // ~/.claude/skills/work-items -> repo skill) | `~/.claude/skills/work-items` (outside repo; may need human approval) | — | — |
| C2 live spec + HISTORY draft (close step 2; stakeholder chose `skills/work-items/docs/`) | `coder` | `aa126193922242150` | delivered (uncommitted until the close check; 6 plan/code divergences reported) | `skills/work-items/docs/SPEC.md`, `HISTORY.md` | `analyst` close check (step 3) → — ; HISTORY entry must be re-checked after it (step 5 is last) | — |

| C3 close check (step 3) | `analyst` | `ad341f66ca9cc17cb` | delivered | `docs/reviews/work-item-numbering.md` Pass 10 | needs changes (S-1 link/set miss legacy fallback; C-1 mint-legacy --home unvalidated; C-2 key suggest missing keys.json; spec gaps; no context file names SPEC.md). Analyst agent now ~256k tok: further gate passes → fresh analyst |
| C4 fixes S-1/C-1/C-2 | `tdd-engineer` | `adf789a10751b7507` | accepted (committed 36b7c392; 505 tests; my mutant on mint --home check killed) | `skills/work-items/…` | — | — |
| C5 plan 3.9/8b restated | `architect` | `a7af14139c4b90236` | accepted (2-line diff verified; committed) | `docs/plans/work-item-numbering.md` | — | — |
| C6 spec edits + pointers + HISTORY (last) | `coder` | `aa126193922242150` | delivered (uncommitted: SPEC.md, HISTORY.md, README row, root AGENTS.md bullet — diffs verified as only ours) | `skills/work-items/docs/{SPEC,HISTORY}.md`, `skills/README.md` row, root `AGENTS.md` skills line | — | — |

Note: the real journal already holds a foreign event `GAL-I02-K0001` (python-web-quirks session, home `claude`) — first adoption by another session; NOT this coordination's to flush/commit.

| C7 close re-check (fresh analyst) | `analyst` | `a2cb1cc0e27bf29d0` | in-flight | `docs/reviews/work-item-numbering.md` Pass 11 | — | — |
