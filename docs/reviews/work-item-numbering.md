# Review — work-item-numbering requirements

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/requirements/work-item-numbering.md`

## Pass 1

**Scope & verdict.** `docs/requirements/work-item-numbering.md` (untracked, Status `Interviewing`), read whole, against root `REVIEW.md`, root `AGENTS.md` (header block, collision rules, growth-stops), `docs/requirements/cross-initiative-coordination.md`, `claude/docs/requirements/coordination-graph.md`, `claude/docs/requirements/team-engagement-in-other-repos.md`, and ID usage in `docs/BACKLOG.md`, `falkor-chat/docs/BACKLOG.md` and `claude/*/kaizen/plan.md`. Static review; nothing run beyond greps.

**Verdict: needs changes.** (Two blockers on the legacy-ID contract and the child-rename cascade; a small, fixable set. Re-review after one revision should clear it.)

CPG: considered, not relevant — a requirements document with no code under review.

**Growth-stops check:** first draft, no prior revision to compare; nothing to report. Open questions "None" is consistent with the stakeholder confirming the readback.

## Findings

### Blockers

**B1. FR-12 / FR-20 / FR-6 / AC-6: legacy `K-` IDs are ambiguous, and "resolves to the item" has no target ID.** Verified: `K-004`, `K-005` exist in `claude/architect/kaizen/plan.md:11-12` as the architect's kaizen, and `K-0xx` also numbers `falkor-chat/docs/BACKLOG.md` (38 `### K-`/`### C-` headings) and every other agent's `kaizen/plan.md`. A bare `K-082` is therefore not unique across the repo, yet FR-6 promises "every old ID resolves" and AC-6 resolves `K-082` with no qualifier. Second, a legacy item gets no `GAL-` ID (FR-12), but FR-20 says following an alias "resolves to the item's current ID", and FR-9 flags "an item with no ID". The three requirements cannot all hold. Fix: state whether each legacy item receives a new-scheme ID lazily (on first lookup, without editing its document) or stays outside the scheme; require legacy aliases to be qualified by home (component or agent) and say what an unqualified, ambiguous alias returns (all candidates, never a guess). Add an AC with a deliberately colliding pair (`K-004` in two homes). Source: verified by grep.

**B2. A rename of a parent has undefined effect on descendants (FR-4, FR-13, FR-22, AC-3, AC-9).** The path is the identity, so moving a feature or promoting kaizen `K0015` renames it; its stories and tasks contain the old path. Nothing says whether children are renamed (and every old child ID aliased), or keep a stale prefix. FR-22 covers only story/task moves, and a moved story's tasks are likewise silent. AC-3 and AC-9 test only the moved item. The architect cannot choose without this, because it decides alias volume and the FR-9 check. Fix: add an FR (descendants are re-pathed, each old path aliased, local numbers kept) and extend AC-3/AC-9 to assert a child's old ID resolves. Inferred from reading.

### Majors

**M1. Tension between FR-2a and FR-4.** FR-2a makes `F0012` unique in the whole project "whatever its parent", which already identifies the feature; FR-4 nevertheless renames it on every move. The doc never says why the path rather than the segment is the identity (decision log entry only restates it). Either justify it in Intent (human-readable containment) or accept it as a stakeholder-confirmed decision and note that two live IDs can share `F0012`; FR-9 must then check segment-level uniqueness, not only whole strings (AC-7 tests only whole-ID duplicates).

**M2. FR-10 is under-specified and collides with the segment grammar.** `GAL-F0012-AC3`: "AC"/"FR" are not single-letter typed segments (FR-2), `FR3` begins with `F` (the feature letter), the counter scope (per feature? per story?) and width are absent from FR-2a/FR-21, and "attached to the feature (or story)" gives no rule for choosing. Also unsaid: whether existing documents' FR/AC numbers (and this document's own FR-1..23) are retrofitted — FR-12 says no rewrite, FR-10 says "numbered under the same scheme". No AC covers FR-10 (see M5).

**M3. Parent rules for D and K items are incomplete.** FR-3 says a story/task "always has a parent" and FR-11 allows `D…-T01`, but nothing says whether a story may sit under a defect, whether a kaizen item can have tasks (`GAL-I02-K0015-T01`), whether a feature may be created under `I01`/`I02`, and what happens to a kaizen item's tasks on promotion (B2). Add a short parent/child legality table (as requirement, not design).

**M4. Conflicts with repo conventions not acknowledged.** (a) FR-20 `Aliases:` and FR-23 "feature ID in each family document's header" add header fields; root `AGENTS.md` header block lists only `Version/Supersedes/Superseded by/Extends/Extended by/Last updated/Reviews` as optional. The requirement must say the header convention is amended (and by whom), otherwise every conforming document is non-conforming. (b) Root `AGENTS.md` still says `BACKLOG.md` items are `K-` numbered; out of scope lists "rewriting existing documents" but not the convention text — name it as an in-scope follow-up or an explicit out-of-scope. (c) FR-23 requires the feature ID in family headers but never says when the ID is allocated for a requirements doc born `Interviewing`, or for a feature with no requirements doc.

**M5. AC coverage gaps.** Every FR except these has at least one AC: FR-1 (glossary/alias — AC-1 cites it but tests only IDs), FR-2a (nothing asserts story counters restart per parent — add: two features each get `S01`), FR-3 (the "no parent → rejected / use D or K" half untested; AC-1 covers only the skip), FR-10 (none), FR-15 (only the defect link in AC-8). FR-18's "a key already in use is refused" (AC-10) has no stated source of truth for "in use across the shared pool"; FR-8 covers items, not keys.

**M6. FR-5 scope is narrower than the Problem.** Problem names "concurrent agents or branches"; FR-5/AC-2 test only simultaneous agents. Parallel git branches/worktrees and cross-repo green-field projects are the realistic failure; state whether FR-5 must hold across unmerged branches, and which side of an unmerged collision is renamed (ties to FR-6). Edge case the architect will otherwise trip on.

### Minors

- **m1. Solution language.** FR-20 "the lookup can be rebuilt from those headers" mandates a storage design (headers as source of truth) while Out of scope defers "how IDs are stored"; restate as the outcome AC-12 already tests (recoverability). FR-21 widths are acceptable as spec. FR-5 "at the same time" is untestable as worded; say "overlapping allocation requests".
- **m2. Terminology.** User story and Intent list "requirement" as an item type, FR-1 glossary has four levels and D/K are not defined there; FR-1 should define defect, kaizen, requirement, and say "feature" also names the requirements→plan→review family (FR-23), which is the same word used for the document family.
- **m3. Decision log order.** Entry at line 86 "Overturns … Counter scope: per parent or per project?" precedes the entry it overturns (line 92); the log is meant append-only in time order. Reorder, or note that 92 is overturned by 86 as to story/task only. The `GML` entry (83) is correctly overturned by 94.
- **m4. Unverified claims in Problem.** I did not find the literal "stable key" need in `docs/requirements/cross-initiative-coordination.md` or the coordination-graph doc (the latter keys overlap on files/graph keys, FR-14); cite the FR in those documents that depends on IDs. The claim of "no consistent prefix" in `mcp-monitor`, `opencode`, `model-bench` was not checked.
- **m5.** FR-2 example `GAL-F0012` shows only feature; add one full example per type (`GAL-I03-F0012-S04-T07`, `GAL-I01-D0015-T01`).

## What's solid
Header block and filename are conformant; the FR-14 reserved `I01/I02` design cleanly resolves the defect/kaizen placement and the overturned entries are named with their FR impact; FR-4/AC-3 test a double move; FR-6/AC-4, FR-9/AC-7 and FR-21/AC-13 are concrete and falsifiable; scope boundaries (no migration, no file renames) are clear.

## Open questions
1. B1: do legacy items get a lazily minted `GAL-` ID, or remain a separate namespace that the lookup merely resolves?
2. M1: confirm the path-as-identity decision still stands given globally unique feature segments.
3. M4: who amends the root `AGENTS.md` header-block convention (`Aliases:`, feature ID field)?

## Pass 2

**Verdict: needs changes** (one major, several minors; nothing blocks the architect once M-A is fixed — a single in-place revision should clear it). Re-read the revised `docs/requirements/work-item-numbering.md` whole. CPG: considered, not relevant — requirements document, no code under review.

**Growth-stops check.** New scope (acceptable): FR-24, FR-25, AC-15 to AC-17, FR-12's qualified-alias rule, decision entries 102-106. Rework done in place (Problem, FR-1, FR-5, FR-10, FR-11, FR-12, FR-20, FR-23, AC-2, AC-3, AC-6, AC-9, AC-12): no finding→resolution table, no disposition list; good. Reported: (g1, minor) log entry 101 is a review-outcome narrative that lists the sections changed, not one question → one answer with FR/AC affected; it restates review history the review file already holds — drop it or reduce to the one stakeholder question. (g2, nit) the log was reordered into time order, whereas it is append-only; content is intact (all Pass 1 entries present).

**Pass 1 dispositions**
- B1 fixed: FR-12, AC-6, Problem line 9. Rechecked: `K-004` exists in architect, data-scientist, devops, qa-engineer, security-expert and tico `kaizen/plan.md` (grep), so the AC-6 pair `architect:K-004` / `devops:K-004` is real. Residual in M-B below.
- B2 fixed: FR-24, AC-3, AC-9 — but see M-A (FR/AC descendants).
- M1 accepted as stakeholder decision (FR-4, log 104); no new text needed. Residual: FR-9/AC-7 still whole-string only; fine now since a rename aliases the old segment.
- M2 mostly fixed: FR-10 now fixes letters, per-feature counters, no renumbering; AC-17 added. Residual in M-A.
- M3 fixed: FR-25 and AC-16 (coverage residual m-C).
- M4 fixed: header fields moved to Out of scope with "when the feature closes"; FR-23 now says ID assigned when the requirements document is created.
- M5 fixed: AC-15 (FR-1), AC-16 (FR-2a, FR-3), AC-17 (FR-10). FR-18's "key in use" source of truth still unstated (minor, m-D).
- M6 fixed: FR-5 and AC-2 now cover separate branches merged.
- m1 fixed (FR-20 "recovered"; FR-5 "concurrently"). m2 fixed (FR-1). m3 fixed (log now time-ordered). m4 fixed (Problem line 14 rephrased to what the coordination document actually records; line 9 now names the ambiguity). m5 not done (no full-path example for a defect/task; FR-11 and AC-1 now give some) — dropped.

**New findings**

- **M-A (major). FR-10/AC-17 contradict FR-24 and FR-2.** FR-10 and AC-17 cite `GAL-F0012-AC3` / `GAL-F<n>-FR<k>`, i.e. a feature path with no initiative. A feature under `I03` has the ID `GAL-I03-F0012`, so its AC is `GAL-I03-F0012-AC3`, and FR-24 renames every descendant — including FR/AC, which FR-25 makes children of a feature — when the feature moves. Either FR-10/AC-17 hold only for initiative-less features (then say so) or the citation form is wrong for most features. Decide whether FR/AC are path descendants (renamed on a move, aliased, citation churn across documents) or recorded links keyed by the feature's stable segment; the doc must pick one, and AC-3/AC-17 must test it. Verified by reading FR-2, FR-10, FR-24, FR-25.
- **M-B (major). FR-12 "first cited or worked on" is not decidable.** Who detects a citation, and is any mention in prose enough? Without a boundary, two agents can mint two `GAL-` IDs for one legacy item. State the trigger as an observable event (e.g. the first time an item is referenced from a new-scheme document or given a link/attribute), and add that minting is idempotent, including under FR-5 concurrency; AC-6 tests only two distinct items. Also state that an unambiguous bare `C-310` resolves (the text implies it; no AC checks it) and that FR-6/FR-9 apply only to items already in the scheme (FR-6's "every old ID resolves" otherwise covers untouched legacy items).
- **m-A. FR-11 is internally redundant and drifts from FR-15/AC-8.** It says a fix is "a task under the defect, or a new story under the feature" and then "The fix is a task under the defect". FR-11 now allows the link target to be a feature, story or task, while FR-15 and AC-8 still say "introducing feature"; AC-8 should exercise a story/task target and a change of target type. Log entry 105 narrows entry 91 but does not name it as the entry it overturns (root rule: a reversing entry names it).
- **m-B. FR-24/FR-13 gaps.** Promotion of a kaizen item to a feature does not say under which initiative the feature lands (FR-25 allows initiative-less or I03+). FR-22 does not cite FR-25: a story moved to a non-feature parent must be refused. FR-24 and FR-25 sit before FR-23 (numbering order).
- **m-C. AC-16 samples, does not probe.** FR-25 is a parent×child matrix; AC-16 names three refusals and one acceptance. Restate as: for every child type and every parent type, the pair is accepted exactly when FR-25 lists it (kaizen under I02 only, defect under I01 only, FR/AC under a feature only, story under initiative/kaizen/task refused). The shape list is otherwise incomplete (story under kaizen is untested).
- **m-D.** FR-18/AC-10 "a key already in use is refused": name where the used keys are known (e.g. the FR-8 lookup extended to project keys) or accept that it is design.

## Pass 2 — stakeholder answers folded in (question → answer, verified present)
Legacy IDs → GAL- on first touch, qualified alias (FR-12, AC-6, log 102) · descendants follow (FR-24, AC-3, AC-9, log 103) · rename-on-move one rule (FR-4, log 104) · defect link-only, never a child (FR-11, log 105) · parent legality (FR-25, AC-16, log 106).

### What's solid (Pass 2)
FR-25's table is consistent with FR-3, FR-13, FR-14 and FR-11 (task under kaizen, defect and feature all legal; story only under feature); AC-9 and AC-3 now test descendants; AC-2 is falsifiable across branches; the open-questions list is genuinely empty.

### Open questions (Pass 2)
1. M-A: are FR/AC path descendants of the feature (renamed on a move) or links keyed by its stable segment?

## Pass 3

**Verdict: approve with suggestions.** No blocker or major remains; the latest verdict is not "needs changes", so the `Ready for design` flip is open to `tico` on stakeholder confirmation. CPG: considered, not relevant — requirements document, no code under review. Re-read the revised `docs/requirements/work-item-numbering.md` whole.

**Growth-stops check.** New scope: none beyond Pass 2's. Growth since Pass 2 is rework or folded answers: FR-10 (path-descendant form), FR-12 (trigger, idempotency, bare-name rule, scope sentence), FR-13/FR-18/FR-22 (one clause each), AC-6, AC-16, AC-17, log entries 107-108 (each one question → one answer with FR/AC). No finding→resolution table, no disposition list, no restated spec facts. Residual: entry 101 still names review passes and a section list (restated review history, g1 from Pass 2, now shorter; drop it at the next revision). Entry 91 was edited in place ("as to placement"); a log is append-only, so a nit. Log reordering: not required to undo, per the coordinator.

**Pass 2 dispositions**
- M-A fixed: FR-10 now gives `GAL-I03-F0012-AC3` / `GAL-F0012-AC3`, renamed with the feature; FR-24, FR-25 (FR and AC have a feature) and AC-17 agree (read all four).
- M-B fixed: FR-12 states the trigger (links/cites, or work first opened), idempotent minting, bare `C-310` resolves, and scopes FR-6/FR-9 to items in the scheme; AC-6 tests two `K-004` homes, repeat citation, bare ambiguity and bare resolution. Residual m-1 below.
- m-A fixed: FR-11 trimmed, FR-15 and AC-8 now say feature, story or task.
- m-B fixed: FR-13 names the initiative at promotion, FR-22 cites FR-25, FRs are in numeric order.
- m-C fixed: AC-16 now states the whole parent/child matrix (accepted exactly when FR-25 lists it).
- m-D fixed: FR-18 names one shared key list.
- g1 partly fixed (shortened, see above); g2 waived.

**Stakeholder answers folded in (verified present):** FR/AC as path descendants → FR-10, FR-24, AC-17, log 107 · legacy mint trigger and idempotency → FR-12, AC-6, log 108.

**New findings (all minor)**
- **m-1.** FR-12's second trigger ("work on it is first opened under the scheme") has no AC; AC-6 exercises only the citation trigger. Add it to AC-6, or drop the clause.
- **m-2.** FR-21 gives no width for `FR`/`AC` segments and FR-2a does not list FR/AC among the counted types; the examples (`AC3`) imply unpadded and per-feature (FR-10). One clause in FR-21 settles it, and the architect should not have to infer it. Also FR-3 says "the type letter of each segment" while `FR`/`AC` are two letters.
- **m-3.** No AC asserts FR-2's cross-project uniqueness of the whole string beyond the key refusal in AC-10; acceptable, since uniqueness follows from FR-17/FR-18, but say so if the architect asks.
- **n-1.** Log entry 105 broadens entry 91's link target (feature → feature, story or task) without naming 91.

### What's solid (Pass 3)
FR-25/AC-16 is a real matrix probe and agrees with FR-3, FR-13 and FR-24 (promoted kaizen tasks land directly under the feature, which FR-25 allows); every FR has an AC (FR-2 and FR-12's second trigger noted above); rename, alias and descendant rules now form one coherent rule set; open questions are empty and the header block is conformant.

### Open questions (Pass 3)
None blocking.
