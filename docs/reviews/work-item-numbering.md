# Review — work-item-numbering requirements

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/requirements/work-item-numbering.md`, `docs/plans/work-item-numbering.md` (Pass 4) · **Item:** `GAL-F0001`

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

## Pass 4 — implementation plan `docs/plans/work-item-numbering.md`

**Scope & verdict.** The plan (untracked, `active`, first draft) read whole against `docs/requirements/work-item-numbering.md` (FR-1..25, AC-1..17), root `AGENTS.md`/`REVIEW.md`, the two consumer requirements docs, `skills/README.md`, the coder/tdd write guards and the repo's real K-/C- usage. Run: three git prototypes (union merge, dirty-tree merge, newline glue, worktree common-dir), a lock-free allocator race (8 procs x 25, three runs), and recounts of the legacy homes. Nothing in the plan or requirements was edited.

**Verdict: needs changes** (1 blocker, 5 majors; all fixable in place, none requires a new design).

CPG: considered, not relevant — a plan for a new stdlib script package and docs; no existing code is changed, so there is no call graph to query.

**Growth-stops check.** First revision of the plan; no prior text to compare. The plan states only its own design; §3.3's parent table restates FR-25 by necessity (it is the code table, guarded by the drift test); no finding-resolution table. Nothing to report.

### Blocker

**B1. The FR/AC grammar cannot express the numbers FR-10 says are kept (AC-17 satisfied only on paper).** Plan §3.2 gives FR/AC "unpadded" digits, parsed as `\d+`, and §3.8 verifies "the cited number exists in the requirements document". Existing documents number requirements with letter suffixes: `grep -oh '(FR|AC)-[0-9]+[a-z]'` over `docs/requirements`, `claude/docs/requirements`, `falkor-chat/docs/requirements` gives FR-2a x12, FR-7a x9, FR-11a x9, FR-8a x7, FR-2b x7, AC-21a x4 ... and the two documents this plan itself cites (FR-5a/FR-5b/FR-2b in team-engagement; FR-2a here). FR-10 requires "none is renumbered". Verified by grep. Test strategy item 1 and 11 use only `AC3`-shaped numbers, so no test would redden. Fix: define the FR/AC suffix as `\d+[a-z]?` in `rules.py` (and FR-21 "unpadded" for the digit part), add `FR2a`/`AC21a` to test 1 and 11, and make the check's document parser read `FR-2a` headings/bullets to the same shape.

### Majors

**M1. Journal/tracked asymmetry leaves dangling IDs across branches (FR-8, FR-9, FR-12).** §3.1 copies only tracked→journal on open; a *new* event is written to both, but an event that already exists only in the journal (branch A's `new`, returned to branch B by an idempotent `mint-legacy`, or by a `resolve` B then cites) is never written into B's tracked file. B merges first, and main holds documents citing an ID with no `new` event; if A is abandoned the ID dangles permanently (the informational "journal not in tracked" line only helps in A's clone). The same one-way gap means deleting `<common-dir>/work-items/` (journal loss) silently drops the numbers of every unmerged branch — duplicates later. Reasoned from §3.1/§3.6; the union-merge half verified by prototype. Fix: any mutating op (and `headers sync`) first copies into the current tracked file the events needed by the uids it touches or cites (closure over parents); and journal reseed on open also folds `git show <ref>:.work-items/registry.jsonl` for every local branch and `refs/remotes/*` ref (O(branches), reads no backlog, so FR-7 holds) — which also narrows the declared cross-clone residual to unpushed work. Add a test: mint on A, mint again on B, merge only B, assert the `new` is present in main's tracked file.

**M2. Every allocation dirties the tracked file and blocks merges (operability of FR-5's own use case).** Prototype: with an uncommitted `.work-items/registry.jsonl` append in the checkout, `git merge B` aborts ("Please commit your changes or stash them before you merge"), although B touches the same file only by appending. Teco's parallel-branch flow merges into a tree where another agent may have just allocated. Also verified: a tracked file whose last line lacks `\n` glues the next branch's first line to it after a union merge (`{"evid":"n1"}{"evid":"a9"}`) — a corrupt line the fold must survive. Fix: writer always ends records with `\n` and the reader tolerates/repairs a glued line (test with the glued case); SKILL.md states the rule "commit the registry with the unit that allocated"; consider making the tracked file a flush target (`wi flush`, run by the commit step) with the journal authoritative, which removes the per-allocation dirtying. Add a test for merge with a dirty registry.

**M3. FR-23 is satisfied only at sync time, and the family glob misses this repo's own documents.** (a) §3.7 finds family documents under `*/docs/*/`; this feature's family lives at root `docs/requirements|plans|reviews/` (no leading component), as does any green-field repo with a single `docs/`. Verified by pattern reasoning against the paths in this plan's header; step 8's dogfood ("`Item:` header on this feature's family documents") would find nothing. Use `docs/<kind>/` and `*/docs/<kind>/` (kinds = the closed list in root `AGENTS.md`). (b) A plan/review/test-report created *after* `new F` gets `Item:` only if an agent remembers `headers sync`; §3.8's check flags a missing `Item:` only for `requirements/` documents. Extend the check: for every registered slug, every family document must carry an `Item:` equal to the registry ID (this is the testable half of AC-14), and add that to test 17.

**M4. Step order: steps 2-4 cannot be tested without step 5.** `wi new` needs `project.json` (key, components), `D`/`K` need `I01`/`I02` seeded by `init` (AC-1, AC-8 in steps 2/4), home validation needs `components`. `init` is step 5. Either move `init` (minus the shared-key claim) before step 2, or have each earlier step state the fixture it hand-writes. Also step 4 is oversized against the ~3 steps/~5 files norm (4 modules + 2 test files, 5 ACs); split `rebuild`+`check` from `links`+`headers`.

**M5. Consumer contract and follow-ups have no owning step (§3.9, plan risks 5-6).** `wi resolve --json` shape (`touched[]`, `milestone`, `where`, `links`) spans step 3 (`resolve`) and step 4 (`set`/`link`); no step owns it and no test in §5 asserts the keys. Risks 5-6 say "confirm"/"must list" but nothing assigns them, and the documentation-impact list names neither `docs/requirements/cross-initiative-coordination.md` (ledger unit = `T` under a feature is a decision about that document's data, only "non-blocking, defaulted" here) nor `claude/docs/requirements/team-engagement-in-other-repos.md` (its AC-1 says a kickoff ends with *exactly* two files; FR-23 needs `.work-items/` and `git init` at kickoff — a real conflict, see plan risk 6). Add a step-3/4 test pinning the JSON keys, and a flagged follow-up item for each consumer owner (route via `teco`: `tico` amends both requirements). Also: root `AGENTS.md` amendments have no step; if they land only "at close", step 8 writes `Item:` into headers that the header convention does not yet permit — land the header-field amendment before step 8 or state the transient nonconformance.

### Minors

- **m1. Legacy home count is wrong; spread 13 / 14 / 16.** Recomputed from the working tree: 13 `claude/*/kaizen/plan.md` define K- items; `falkor-chat/docs/BACKLOG.md` defines 38 `### K-` headings (14 homes by *definition*). `mcp-monitor/docs/BACKLOG.md` and `model-bench/docs/BACKLOG.md` define none: each mentions a falkor-chat item (`K-018` at mcp-monitor:34,37; `pre-K-066` at model-bench:183, both falkor-chat's), so they are 16 only under "any mention". C- : one home (`docs/BACKLOG.md`, 15 distinct ids; 107 more in `docs/HISTORY.md`), `falkor-chat` and `opencode` zero — plan correct. Consequence: listing the two non-homes makes bare `K-018` report a phantom `mcp-monitor:K-018` candidate; and cross-component citations of falkor-chat K- ids exist (K-030 is both falkor-chat's and cobb's cited "K-030 Track 2"), which the plan's resolver handles only if the namespace list is complete. Set the list to 14, and add a rule/check that a new `claude/<agent>/kaizen/` or backlog adds its home to `legacy_namespaces` (otherwise a new agent's `K-001` is wrongly "unambiguous"). The "known legacy, not minted" outcome needs its own exit code (§3.6 defines 0/3/4 only).
- **m2. Tests that cannot fail.** (i) Test 7 / AC-5 ("documents unreadable/absent, allocation succeeds") passes for an allocator that scans documents and swallows errors; assert no file other than the registry/journal/project.json is opened (patched `open`, or run under a tree with a poisoned `docs/`). (ii) Test 16 / AC-12 must delete *both* copies; deleting only the tracked file passes vacuously because the fold reads the journal. (iii) Test 2 expected matrix must be written out from FR-25's text, not iterated from `ALLOWED_PARENT`, and the drift test (19) compares only two copies of the plan's own table. (iv) AC-6 "no existing document is edited" has no assertion — add a clean-`git status` check after mint; the "first opened" trigger as `--reason opened` is the same code path, so only the prompt side (step 7) carries it — say so. (v) AC-2's "two tasks under one story on two branches" is not in §5 (test 4 covers stories under different features only).
- **m3.** `(ts, evid)` ordering uses wall-clock `time_ns`; a backward clock step (common under WSL2) can order a `move` before its `new`. Record an append sequence number under the lock and order by `(branch-local seq, ts, evid)`, or at least tolerate it in the fold.
- **m4.** `wi check` malformed-ID rule ("token beginning with a claimed key and hyphen") flags prose such as `GAL-` and `GAL-F<4 digits>` (14 such tokens already in these two documents). Scope the scan to header lines (`Item:`, `Tracks:`, `Aliases:`) and `--set`, or exempt backticked placeholders. Also the check never reads ledgers, so an unnumbered ledger unit (the cross-initiative consumer) is invisible to FR-9.
- **m5.** `resolve <alias>` must report that the input was an alias and the current ID (FR-20); the CLI/`--json` spec does not say so, and no test asserts it.

### Answers to the brief's questions

1. **FR/AC coverage.** All FRs have a design home except where B1 and M3 bite. ACs satisfied only on paper: AC-17 (B1), AC-14 (M3b), AC-5 and AC-12 (m2), AC-6's "opened" trigger and AC-10's stakeholder agreement (prompt-side, verified by `cobb` review — acceptable if labelled so). AC-2 test 5 is genuine: a lock-free read-then-append allocator yielded 199, 193, 195 distinct of 200 across three 8x25 subprocess runs, so a removed `flock` reddens it (keep it process-based, not threads). Test 6 reddens a tracked-file-only mutant if both worktrees branch from one commit before either allocates; say so.
2. **Concurrency.** Same clone, any number of worktrees/branches: `git rev-parse --git-common-dir` is the shared dir (verified from a linked worktree; relative `.git` from the main tree, absolute from the worktree — resolve before use). Union merge of two branches each appending one line keeps both (verified). Residuals beyond the plan's "separate clones": M1 (journal loss, journal-only events at merge), M2 (dirty tree, newline glue), a remote-isolation or scratch-clone agent (separate `.git`; the plan should say agents must use worktrees, not clones, in this repo), and a GitHub web merge, which does not honour a repo's `merge=union` attribute (inferred, not tested; no GitHub flow is in use here).
3. **Legacy counts.** See m1: 13 (agent plans only) / 14 (definitions) / 16 (any mention in a backlog). The plan's 16 overcounts by two non-homes; `K-004` is a heading in 5 kaizen plans and in security-expert's in another format (6 agents, as stated).
4. **Rename/alias.** Counter-by-parent-uid and never-reduced numbering hold; chained moves resolve by uid; promotion keeps tasks (K uid retyped, T counter continues). Suffix resolution is sound except B1. Move-back returns a fresh local number and the old alias string maps to the same uid, so no clash. Concurrent move vs. descendant creation on an unmerged branch is only partly covered by risk 3.
5. **Seams.** M5, M4 (step sizing), m4. Doc-impact list is otherwise complete for root `AGENTS.md` (header, filename/collision, `K-` numbering, Structure) and `skills/README.md`; add the two consumer requirements docs and `claude/AGENTS.md`. Hooks: `guard-coder-broad-write.sh` and the tdd guard deny `docs/plans|reviews|requirements/*`, so step 8's `Item:` edits must go through the CLI's Bash path (or `teco`), not a coder `Edit`; the plan says "with `teco`" — make the CLI path explicit.
6. **Skill + stdlib CLI** is the right home (reaches an empty repo through the symlink; no service dependency; precedent `skills/joern-cpg/scripts/`); the rejected alternatives are fairly argued. Dependencies a green-field repo lacks: the root-`docs/` glob (M3a), `keys.json` only in the team repo (acceptable, by design), the team repo's working tree being writable and clean for `key claim` (risk 2; worktrees of the team repo resolve their own `keys.json` unless located through the symlink — `Path(__file__).resolve()` does that, keep it), and the symlink existing per machine (no skill on CI or another machine).

### What's solid (Pass 4)
Event-log fold with uid = birth ID is a clean answer to FR-4/FR-6/FR-24 (aliases fall out of the fold; counters never decrease); suffix-resolved FR/AC avoids registry rows; the legality table matches FR-25 line for line; the lock-then-allocate path makes idempotent minting concurrent-safe; rejected alternatives and their trade-off are stated; legacy-namespace ambiguity is decided from a declared list rather than from what was minted; steps carry owners, files, done-conditions and an AC→step→test map.

### Open questions (Pass 4)
1. M1: accept reading other branches' registry files on reseed (small cost, closes journal loss), or document journal loss as a residual?
2. M5: does the stakeholder want ledger units as `T` under the feature settled here, or left to the cross-initiative owner?

## Pass 5 — revised `docs/plans/work-item-numbering.md` (13 steps)

**Verdict: needs changes (narrow).** No blocker. Three majors (N1-N3), each a one-line design fix; every Pass 4 finding is fixed or fixed-with-residual (below). CPG: considered, not relevant — plan for a new script package and docs; no call graph to query. Re-read the plan whole; re-ran the union-merge/flush prototypes, the FR/AC suffix grep, the glob test and the K- recount against the tree.

**Pass 4 dispositions (one line each)**
- B1 fixed: `(FR|AC)\d+[a-z]?` in 3.2/step 1/tests 1, 9, 14; recheck grep over `git ls-files '*.md'` finds no multi-letter, uppercase or `a1`/`.1` suffixes (34 distinct `FR-/AC-<n><letter>` labels), so the claim "single letter" holds.
- M1 fixed with residual N1: the journal-authoritative design removes the cites-a-journal-only-ID hole (flush copies all events; test 12 reddens an own-events-only flush); reseed-from-refs is specified and tested (test 18).
- M2 fixed: allocation no longer touches the tracked file; `raw_decode` reader and always-newline writer specified (test 4). Re-ran: flush-then-`git merge` still aborts on a dirty registry; the plan now states this and puts it in the skill rule (test 6's last clause is documentation and cannot fail — nit).
- M3 (a) fixed in 3.7 (`docs/<kind>/` and `*/docs/<kind>/`) but the same defect recurs in 3.8, see N3; (b) fixed (check rule, test 15).
- M4 fixed: init precedes allocation (step 2); steps are at most 5 files. Step 2 is dense (journal, flush, reseed, refs, init) but within the norm.
- M5 fixed: JSON key set pinned (test 14), steps 11 and 13 own the header amendment and both consumer follow-ups.
- m1 fixed except the detector (N3): K- home list is 14, exit 5 added. Recount re-run: 13 agent plans define `K-`; 14 files define (13 + `falkor-chat/docs/BACKLOG.md`); 16 mention (+ `mcp-monitor`, `model-bench` backlogs, citations only). Plan's 13/14/16 confirmed; 14 is the right driver. C-: 1 home.
- m2 fixed (i literal-open recorder, ii both copies, iii literal matrix, iv `git status`, v two tasks on two branches) except AC-5 vs the `new F` hook (N2).
- m3 fixed (`ts = max(now, last+1)`, `new` events first); `last` must be the max over the folded set, not the journal alone (nit — an imported higher `ts` otherwise orders later local events early).
- m4 fixed for prose; the `Tracks:` scope creates N3. m5 fixed (`matched_as`, test 9).

### New findings

**N1 (major). `wi flush` republishes other branches' events in a different order, so union merges duplicate lines (verified).** Prototype: branch B flushed `b1` earlier; A flushed `a1 b1 a2`; B later flushed `a1 a2 b2` (journal order). Merging A then B gave `e0 a1 b1 a2 b1 a1 a2 b2`: 8 lines, 5 unique. When both flush the same events in the same order, git merges clean (no growth; also verified). The fold dedups by `evid`, so results stay correct, but each concurrent branch re-appends everything created since its base, so the file grows with branches x in-flight events, permanently. Fix: `wi flush` also removes duplicate `evid` lines (keep first, one deterministic rewrite) and `check` reports duplicates informationally; test 6 gains the differing-flush-order case and asserts unique lines == `evid` count after the post-merge flush.

**N2 (major). AC-5 test 7 contradicts 3.7's hook.** `new F`, `move` and `promote` "call `headers sync` automatically", which scans `docs/<kind>/` and opens family documents. AC-5 requires allocation to read none, and test 7's open-recorder would redden on `new F --slug` once step 6 lands, or be run only on `new T` and then say nothing about the path `tico` actually uses. Fix: make header writing a second phase after the lock is released (`wi new F` prints the ID, then runs the sync, or the agent runs `wi headers sync`), and state that test 7 covers the allocation phase of `new F` too, while the header phase may touch only files matching the family regex.

**N3 (major). `wi check` cannot be "clean" at step 12 as specified.** (a) 3.8's namespace rule scans `*/docs/BACKLOG.md`; verified `fnmatch('docs/BACKLOG.md','*/docs/BACKLOG.md')` is False, so the one C- home (root `docs/BACKLOG.md`, home `cpg`, not derivable from the path, which maps to `repo`) is never checked, the Pass 4 M3a glob defect again. "Defines" is also unspecified: C- items there are `- **C-310 — …**` bullets, K- ones `###` headings or table rows; a detector keyed on headings would miss them, one keyed on any mention would flag the two citing backlogs. (b) The dangling rule covers `Tracks:`, but existing headers carry legacy ids (`C-1001` x18, `K-030` x17, `K-055` x9 in a sample of the first 8 values); unless limited to scheme-shaped tokens and to documents carrying `Item:`, step 12's done-condition "`wi check` clean on the repo" fails on hundreds of unchanged documents. Fix: glob `docs/BACKLOG.md` too, take the home from `legacy_namespaces` values (a path→home map in `project.json`), define "defines" as a line-start `K-nnn` heading/bullet/row, scope dangling to `KEY-` tokens, and give test 17 one planted citing-only backlog (must not fire) and one root-level definer.

**n1 (minor). Family exclusion is incomplete (3.7).** "A file whose name equals another registered slug exactly is excluded": `x2-coordination.md` where `x2` is registered does not equal `x2`, so `x`'s sync stamps its `Item:` on `x2`'s document. Exclude any file that matches a longer registered slug's family regex; test with `x`, `x2` registered, `x2-coordination`.
**n2 (minor). AC-1 and AC-13 have no behavioral test.** Test 1 parses/formats and test 5 counts distinct IDs; nothing creates `I03`, a feature, story and task and asserts `GAL-I03-F####`, `-S01`, `-T01`, or the 100th story as `S100` by allocation. Add both to test 5/step 3 (this mapping was cleared in Pass 4; the renumbering moved it to tests that don't assert it).
**n3 (minor). Step 12 omits registering the feature.** It runs `wi headers sync` and expects `wi resolve work-item-numbering`, but never `wi new F --slug work-item-numbering`; and prompt-side commit rules (step 10) name `tico`/`teco` only, while `qa-engineer`/`analyst` file defects and implementers commit.
**n4 (nit).** The plan header's `Reviews:` still reads "(requirements review)".

### (1) Same-revision interactions
Journal-only paths: an ID is journal-only until its unit's `wi flush` is committed (named risk 5, caught only if `check` runs at commit); that is the one path and it is now explicit. Reseed-from-refs runs only when the journal is missing or empty, so a long-lived second clone never re-reads other clones' later pushes: the declared cross-clone residual stands and is stated correctly. `ts = max(now,last+1)` and new-first ordering do not interact badly with flush. The glued-line reader is not exercised by flush (writer always ends `\n`), only by external edits. The one real interaction defect is N1.

### (2) Growth-stops check (3127 to 4549 words, +45%)
By section, 3.1 (628 words) and 3.9 (417) carry most of it. **New scope (acceptable):** flush/reseed/ordering design (3.1), steps 11 and 13 and the `docs/<kind>/` glob, the JSON contract key set and `matched_as`, extra check rules, tests 4, 12, 14-18. **Rework prose and finding leakage to cut:** (a) the whole "Decisions taken on questions the review raised" paragraph in section 6: answers to review questions, already designed in 3.1/3.9; (b) 3.1's "Decided against documenting loss as a residual: it costs one `git show`…", "Interaction checked: …", "(prototype: …)", and "Rule carried by…" justification sentences, and 3.2's "(FR-10; B1)": design is stated once, finding IDs at most cited; (c) section 2's first bullet narrates three counting definitions and a decision (about 250 words); one line ("14 defining homes; the 2 citing backlogs are not homes") is the fact; (d) the 14-home fact is stated three times (2, 3.6, "Step 12 values"): keep one, cite it; (e) "(not read by me)" and "decided here" first-person asides in 3.9; (f) section 5's AC to step to test map restates the ACs column of the step table: keep one. **Restated facts:** the section 6 documentation list re-lists items in 3.7/3.10/the steps. Estimated saving 500-700 words with no design loss.

### (3) Re-audit of items cleared in Pass 4
AC-2 (S3, tests 5, 6), AC-3 (S4, 9), AC-9 (S4, 10), AC-16 (S1, 2 and 3), AC-4 and AC-8 map to a real step and a test that can fail. AC-1 and AC-13: weaker after renumbering (n2). AC-5: n/a until N2 is fixed. AC-17 relies on test 9's `AC21a` citation: good.

### (4) New seams
Step 11 (header-only amendment before step 12) is correct and narrow. Step 12 via the CLI (Bash): the coder Bash guard (`guard-coder-broad-bash.sh` to `claude/scripts/guard-broad-bash.sh`) is a confirm-prompt guard, not a path deny-list, so a `wi` run is not blocked by the docs/* Write guard, as the plan states; owners of the edited requirements/review documents get a header-only change (permitted as metadata). Step 13 is delegation only; its done-when is the two follow-ups being routed, fine. Glob: `docs/<kind>/` and `*/docs/<kind>/` exclude `docs/archive/<kind>`; see n1 for the slug exclusion.

### (5) Stopping condition
Findings in the fixes (3 majors, 4 minors) are fewer than in the original (1 blocker, 5 majors, 5 minors) and none is structural. Next revision's done-condition, each checkable by command: (i) N1: a prototype or test merging two differing-order flushes ends with `sort file | uniq -d` empty after the post-merge flush; (ii) N2: test 7 text names `new F --slug` and the header phase as separate; (iii) N3: `python3 -c "import fnmatch; ..."`-style check that the plan's BACKLOG glob matches root `docs/BACKLOG.md`, and test 17 lists the planted citing-only backlog; (iv) the section 6 "Decisions taken" paragraph and the 14-home duplicates are removed. If those four hold, **approve with suggestions** with no further pass; n1-n4 may be folded without re-review.

### Open questions (Pass 5)
None blocking.

## Pass 6 — plan revision 3 (3615 words)

**Verdict: approve with suggestions.** The Pass 5 stopping condition holds: N1–N3 fixed and re-executed, n1–n4 fixed, growth cuts real. No blocker or major remains; the plan may go to step 1. CPG: considered, not relevant — plan for a new script package; no call graph to query.

**Dispositions**
- **N1 fixed.** Re-ran the differing-order prototype with the specified flush (append missing, then keep-first dedup): after A and B merged 8 lines (5 unique); a post-merge flush gave 6 lines, 0 duplicates. Extended it with a third branch D forked before the dedup that republished everything: its merge into the deduped main kept 6 unique lines and a final flush left 0 duplicates, so the dedup rewrite does not conflict with other branches' appends and converges. Reseed-from-refs only feeds events into the journal, which the fold dedups by `evid`, so it does not interact. Test 6 states the `uniq -d` assertion.
- **N2 fixed.** 3.7 makes header writing a second phase (`new F` prints the ID and reads no document; `--sync` runs after the lock is released); test 7 runs plain `new` and `new F --slug` without `--sync` under the open recorder and bounds the `--sync` phase to family-pattern files. `headers sync` reads the registry only, so running it unlocked has no ordering hazard.
- **N3 fixed.** Re-ran the dry check against the tree with the plan's definer pattern (`#{2,4} K-nnn`, `- **K-nnn`, `| K-nnn`) over `docs/BACKLOG.md`, `*/docs/BACKLOG.md`, `claude/*/kaizen/plan.md`: 18 candidate files, exactly 15 define (13 agent plans, `falkor-chat`, root `docs/BACKLOG.md`); `mcp-monitor`, `model-bench`, `opencode` backlogs do not fire; no BACKLOG file outside the candidates defines. Root file matched by its own pattern (`fnmatch('docs/BACKLOG.md','*/docs/BACKLOG.md')` is False, as the plan states). Legacy `Tracks:` ids no longer fire: no header line in the tree begins with `GAL-` (0 of the headers read), and the malformed/dangling rules are scoped to claimed-key tokens.
- **n1 fixed.** Simulated the longer-slug rule with `x`, `x2` registered: `x2-coordination.md`, `x2.md`, `x2-ml.md` go to `x2`; `x-coordination.md` and `x3.md` to `x`. Test 15 covers it.
- **n2 fixed** (test 5 creates `I03`/F/S/T, a no-initiative feature, and 100 stories, asserting `S100`); **n3 fixed** (step 12 runs `wi new F --slug … --sync`; flush rule moved to `claude/AGENTS.md` so it binds every committing role); **n4 fixed** (header `Reviews:` plain path).
- **Growth.** 4549 to 3615 words. Verified absent: "Decisions taken", "three definitions", "not read by me", "decided here", inline "B1", the 14-home triplication (now stated once in section 2, cited in 3.6/3.8; "Step 12 values" keeps only components). Still +15% over the original 3127: new scope (flush/reseed, steps 11 and 13, JSON contract, check rules, tests 4–5, 12, 14–18), no finding→resolution prose found. Pass 4's section-5 AC map duplicate is removed (the step table's ACs column is the single map).

**New findings (nits only)**
- 3.1 still carries a worked example (`e0 a1 b1 a2 b1 a1 a2 b2 → 5 unique`) that belongs in test 6; cut it at the next ordinary edit.
- Dedup is a mid-file rewrite: mention that `wi flush` must run after the merge commit is complete (verified order: merge, flush, commit), otherwise an in-progress union merge is rewritten; one line in the SKILL.md rule suffices.

**Open questions.** None.

## Pass 7 (impl, steps 1–5)

**Scope.** `skills/work-items/scripts/wi.py` + `wilib/{rules,events,store,project,alloc,resolve,move,legacy,cli}.py` and `tests/` at 2e8635fd (steps 1–5, commits 2353ab33, 5ca9ccc8, 59a1911c, c28471f4, 2e8635fd), against `docs/plans/work-item-numbering.md` and `docs/requirements/work-item-numbering.md` (AC-1, 2, 3, 4, 5, 6, 9, 13, 16, 17), root `REVIEW.md` (no component `REVIEW.md` applies). Ran: suite (`cd skills/work-items && …/model-bench/.venv/bin/python -m pytest tests -q`: **310 passed**), real `scripts/wi.py` scenarios in temp git repos, 10 mutants on a copy under scratchpad. Scripts: `…/scratchpad/s1.sh`–`s4.sh`, `r4`, `ng` (same dir as `env.sh`).

**Verdict: approve with suggestions.** No blocker: every owned AC held end to end through the CLI (below). Two majors are test/contract gaps with a verified defect or a verified silent-corruption window; neither needs redesign. CPG: considered, not relevant — a new standalone script package, no existing code changed and no loaded graph for it.

### Executed AC evidence (all via `wi.py`)
- **AC-1/13/16** `I03`→`F0001`→`S01`→`T01`; no-initiative feature `GAL-F0001`; matrix in `tests/test_rules.py:74-130` is a literal table matching FR-25 (read against the requirement text; `rules.validate_parent` matches it, no grammar drift; `F` under `I01/I02`, `S/T` with no parent, story under `D/K` refused via CLI, exit 1).
- **AC-2** Two linked worktrees interleaving `new S`, `promote`, `mint-legacy` (same item from both sides, and the second home), `new T` under the *promoted* kaizen's old ID, `move`, `new F`; flush+commit on each, merge both into main: no conflict, 0 duplicate lines, 14 distinct `new` ids, a post-merge `flush` leaves 15 lines, next `new` is unique.
- **AC-3/17/9** Story moved out (`F2-S01`), new story under old parent = `S03` (no reuse), moved back via its *old alias* = `S04`; task keeps its alias chain through 2 moves; feature `I03→none→I03` and aliases resolve; story created *after* a feature move continues numbering and follows a second move; kaizen+task promoted: `K` and `K-T01` resolve as aliases of `F0003`/`F0003-T01`. `FR2a` citation resolves after move.
- **AC-4** retire → number not reissued (`test_cli.py` `test_cli_retire_and_no_reissue`: retired `S01`, next is `S02`; the ID still resolves). Not re-run by me beyond the suite.
- **AC-5** Read `alloc.py`, `move.py`, `legacy.py`, `store.py`: none opens a document (only `init` globs `requirements/` for the baseline, as planned). The recorder tests cover `new` and `mint` only (see F2).
- **AC-6** architect/devops `K-004` → two IDs, re-mint returns the first (also from the other worktree), `K-004` exit 3 listing both with minted IDs, `C-310` exit 5 before and 0 after mint, qualified name text says "alias of …". Ambiguity is driven by declared homes, so minting later cannot change it.
- **Journal loss** Deleted `.git/work-items` after an unflushed `new` on a branch ref: reseed folded the other branch's flushed events (no reissue of `F0003/F0004`) and reissued only `F0005`, which lived solely in the lost journal. This is exactly the plan's stated residual (3.1), not a defect.

### Findings

**F1 (major, verified) — `resolve` accepts FR/AC citations under any parent.** `resolve.py:_requirement` splits the suffix and resolves the prefix as an item without checking it is a feature. `wi resolve GAL-I04-FR1` → rc 0 `GAL-I04-FR1`; `wi resolve GAL-F0001-S02-AC3` → rc 0 (alias of the story-path form). FR-25 says FR/AC have a feature; "any other parent is refused" (AC-16). Step 7's "FR/AC cited but absent" check will inherit a resolver that blesses illegal paths. Fix: require `feature["type"] == "F"` (else unknown, exit 4); the test should parametrize the `FR`/`AC` rows of `test_rules.REFUSED` (every parent type) through `resolve`, not one example.

**F2 (major, verified by mutation) — tests that cannot fail on silent corruption.** Planted on a copy; each survived all 310 tests:
- `alloc.py` writes `"parent": parent_item.id` instead of `.uid`. Identical until a parent has moved; then a child created under a moved feature restarts at `S01`: ran it, `new S` twice under a moved feature printed `GAL-I04-F0001-S02` **twice** (FR-5/FR-6 break) and the child is not renamed by a later subtree move. Missing tests: a `new` under an already-moved parent continues numbering, and a *second* move renames that post-move child (CLI or `alloc`+`move`; assert old-ID alias too).
- `legacy.mint` dropping `parent=` or `title=` (and `cli.py` dropping `parent=args.parent` for `mint-legacy`): no test mints with `--parent`/`--title` and reads the event back. Add one CLI test with `--type F --parent <I03>` and `--title`.
- Plan test 11's "`git status` clean afterwards" and test 12 (mint on A, mint on B, flush B, merge only B: the `new` reaches main) are not in the suite (only `test_init`/`test_store` assert clean status). Behaviour is right (my run above), the pins are missing; put test 12 in `test_legacy`/`test_concurrency` and require the clean status after `mint`.
Killed (so the suite does catch): journal ref scan for remotes, flush not copying journal events, `flush` CLI no-op, promote `--parent` dropped, `_see_local` on move, mint `--type` dropped.

**F3 (major, design decision before step 6) — link model is single-valued; the JSON contract pins that shape.** `events._apply_link` keys `(rel, from)` last-wins and `resolve._record` emits `links: {rel: id}`. `concerns` (a defect concerns one thing) fits. `relates` (FR-15 "every other relation") and `fixed-by` (a defect may have several fix tasks) hold one target. Step 6 will either live with that or change the pinned `links` shape. Decide now, in plan 3.5: either state "links are functional (one target per relation, last wins)" and drop `relates` multiplicity, or fold to `rel → [ids]` with `to:null` clearing all (and an add/remove op if single removal is needed) before `test_resolve`'s key/shape pins grow. The event schema is otherwise compatible (links reference uids, survive moves; unknown `from`/`to` are silently kept, so step 7's dangling check must read events, not `State.orphans`).

**Rulings on the soft spots**
- **(a) `move`/`promote`/`retire` take IDs and aliases only — not a defect.** Plan 3.4 types `<id>`; old IDs work (verified, incl. moving back by alias). FR-12/FR-20 are met by `resolve`. But the failure is a trap: `wi move architect:K-004 …` on a minted item prints `unknown item 'architect:K-004'` (exit 1, not 4). **Minor**: either add a shared qualified-name lookup (not bare: refusing a mutation on an ambiguous name is correct) or make the error say "resolve it first / use its ID". 
- **(b) Minted items get `home="repo"` — minor, FR-19.** Verified (`resolve C-310 --json` → `"home": "repo"`). Legacy homes are namespace homes (`architect`) that are not always components, so copying blindly is wrong, but `cpg`/`falkor-chat` are. Add `--home` to `mint-legacy`, defaulting to the legacy home only when it is a declared component. Also `wi new F --home nonexist` is accepted today; plan 3.5/step 6 owns the component validation, so this is expected until then, but the step 6 row should say it applies to `new --home` and `mint-legacy --home`.
- **(c)** see F3.
- **(d) `--sync` stub — fails safe, minor sharpening.** Only `cli.py` calls it; repo-wide grep finds no caller outside the plan docs and the skill is not yet symlinked or referenced by any agent (steps 9–10 pending). It writes nothing. But it raises *after* allocating and printing: `new F --slug x --sync` prints `GAL-F0006`, then a traceback, rc 1, and a retry burns `F0007`. Reject `--sync` before allocation (argparse error) until step 6; `test_sync_is_a_stub_that_writes_nothing` pins `NotImplementedError` and must flip with step 6. `move`/`promote` have no `--sync` flag at all.
- **(e) Plan text to match the code** (plan edits not made): 3.4 and step 4: `wi move <id> --to <parent|none>` is actually `wi move <id> <newparent|none>` (positional); `wi promote <K-id> --slug S [--initiative I03]` is `wi promote <K-id> --to F [--parent I03]`, and there is no `--slug` (a promoted feature keeps the kaizen's slug, usually none, so FR-23's header phase needs `wi set slug` or a `--slug` option); 3.7: `--sync` on `move`/`promote` does not exist yet. Plan gap: 3.5 names `set` attributes (milestone/home/touched/slug/where) but step 6 lists no `wi set`/`wi link` CLI owner; assign it.
- **(f) alloc refactor — no lock regression.** `allocate(tx, …)` takes no lock; `new`, `retire`, `mint`, `move`, `promote` each open exactly one `transaction()`; `init` calls `flush()` after leaving its transaction; `project.load_namespaces`/`load_key` read before locking; no nested `_locked()` path (read `alloc.py`, `legacy.py`, `move.py`, `project.py`, `store.py`; 8-process and 8-process-mint tests pass).

### Minor / nit
- **m1** Non-git mode creates `.work-items/.lock` (verified); after a later `git init`, `git add .work-items` commits a lock file. Ignore it or keep the lock outside the tracked dir.
- **m2** Retired items stay fully usable: `new S --parent <retired F>` and `move <retired>` succeed (verified). FR-6 only needs the ID to resolve; say in the plan whether that is intended.
- **m3** `IllegalParent` text prints `I3` (`parent_num` unpadded), e.g. `T may not have I3 as parent`; use `format_segment`.
- **m4** `flush` leaves a `registry.jsonl.tmp` after a crash between write and `os.replace`; harmless, `check` could ignore it.

### What's solid
Fold/ordering design: counters never reduced by a move, local counters keyed by parent uid, `new`-first ordering; journal-shared worktree allocation and union-merge dedup behave as the plan states in executed multi-branch scenarios; legacy ambiguity from declared homes; `resolve --json` key set pinned; mutation-killed coverage of flush, reseed and promote paths is real.

### Open questions
None blocking. F3 needs the architect's call before step 6 starts.

### Proposed REVIEW.md lines
None (F2 is project-specific test gaps; F1 is a grammar-drift instance already covered by "check can both fail and pass").

## Pass 8 (impl, steps 6–8)

**Scope.** `skills/work-items/` at 0dd7fd0b (5e8076e9 links, 99345898 headers, 42e46051 rebuild/check, 0dd7fd0b keys, plus fix commit 28ce49c5 for Pass 7), against the current plan (re-read in full: 3.4–3.9 and steps 6, 6b, 7, 8 changed since Pass 6). Ran: suite (**446 passed**), real `wi.py` scenarios in temp git repos and worktrees (`…/scratchpad/p8a.sh`–`p8d.sh`, `q1`–`q6`, `kk`), `check` over a copy of this repo's docs, 26 mutants on a scratch copy (5 re-run from Pass 7, 21 new on steps 6–8). Root `REVIEW.md` applied; no component `REVIEW.md` for `skills/`.

**Verdict: approve with suggestions.** No blocker. One major (N1: a rebuild leaves the registry able to reissue numbers, undetected, with the consequence unstated). Pass 7's three findings are fixed and re-verified. CPG: considered, not relevant — a standalone script package with no loaded graph.

### Pass 7 dispositions (one line each, re-executed)
- **F1 fixed.** `wi resolve GAL-I03-FR1` and `GAL-F0001-S01-AC3` now exit 4; `GAL-F0001-FR2a` and the old-path `GAL-I03-F0001-AC1` resolve (alias). Mutant `feature["type"] != "F"` → `False` is killed.
- **F2 fixed.** Re-ran the Pass 7 mutants on the new tree: `parent: .id`, mint dropping `parent`/`title`, CLI dropping `--parent`: all killed (1 failed each). Plan tests 11 (`git status` clean) and 12 (mint on A and B, merge only B) now exist (`test_legacy.py:219,223`).
- **F3 fixed as the plan now states.** `concerns` single-valued (replace, `--clear`), `fixed-by`/`relates` add/`--remove`, `resolve --json` `links` = `{rel: [ids]}`; refusals verified (`remove` on `concerns`, `--clear` on `fixed-by`, link to an FR citation); mutants dropping `remove`, add-dedup and CLI `--remove` killed.
- **(a) fixed** (error now says "resolve it first"); **(b) fixed** (`mint-legacy --home a` → `"home": "a"`; `new --home zzz` refused); **(d) fixed** (`--sync` implemented on `new`/`move`/`promote`; `headers sync [id]`); **(e) partly**: 3.4/3.5/3.7 now match the CLI, the remaining stale text is listed below.
- **m1 fixed** (`init` writes `.work-items/.gitignore` with `.lock` and `*.tmp`; after a later `git init`, `git add -A` does not stage the lock); **m3 fixed** (`T may not have I02`); **m4 fixed** (same `.gitignore`). **m2 unchanged**: `new S --parent <retired F>` still succeeds (verified); plan still silent, so decide and state it.

### New findings

**N1 (major, verified) — rebuild can reissue numbers, `check` cannot see it, and the plan states only part of the residual.** Plan 3.7 says "stories, tasks and undocumented deleted IDs are not recoverable"; the CLI note says the same. Executed (`p8c.sh`): features F1–F6 (F3/F4 undocumented, F4 retired, F6 a promoted kaizen item with no document), 3 stories, one minted legacy item; delete journal and tracked registry (registry never committed); `rebuild --from-headers . --write`; `check` → **clean, rc 0**; then `new F` → `GAL-F0006` (**reissued**: the promoted feature), `new S --parent GAL-F0001` → `GAL-F0001-S01` (**reissued**), `new K` → `GAL-I02-K0001` (reissued). The `architect:K-004` mapping is also lost, so a later `mint-legacy` would give a second ID (FR-12). The loss is therefore every undocumented live or retired F/D/K, every S/T counter, promotion aliases and legacy minting without a header — not just "deleted IDs" — and the consequence (FR-6/FR-5 violated, silently) is not stated. AC-12 itself (header aliases recoverable) holds. Fixes (architect's call; not run): (i) state the full consequence in 3.7 and in the `NOT_RECOVERABLE` text; (ii) prefer recovery from the registry's git history over headers (`_ref_events` reads only ref tips, so a rebuild is refused only while some tip carries the file — `p8c` first attempt hit that); (iii) let `rebuild --write` take a counter floor the operator knows (`--floor F=N,S:<parent>=N`) or refuse without an explicit `--accept-reissue`. Test: the `p8c` scenario asserting the new F number is above the pre-loss maximum when a floor is given, and refusal otherwise.

**N2 (minor, verified) — `new F --slug S` accepts an already-registered slug.** Two worktrees each ran `new F --slug same --sync`: `GAL-F0001` and `GAL-F0002`, both stamped, merge conflicted add/add on `same.md`; `check` then reports `duplicate: slug 'same' is held by …` and `resolve same` exits 3. The journal is shared, so B can see A's slug at allocation: refuse (or require an explicit flag) there; it also saves the burned number.

**N3 (minor, verified) — `wi init` ignores the key rules.** `wi init --key FR` succeeds, and `init --key GALM` in a second repo succeeds though `keys.json` lists `GALM` for another repo (FR-18/AC-10: never accepted). Only `key claim` checks. Plan 3.9 lists `init` under AC-10; either have `init` call `keys.check_key` (reserved always; claimed-by-other-repo refused) or say the `tico` flow `suggest → claim → init` is the only gate. Also: a missing or corrupt `keys.json` (or `WI_KEYS_FILE`) is an uncaught `FileNotFoundError`/`JSONDecodeError` traceback (verified); `LEGACY_PREFIXES={"K","C"}` is dead against `[A-Z]{2,6}` and ignores the declared prefixes the plan names.

**N4 (minor) — survivors: tests that do not notice.** `headers.KINDS` without `test-reports` (446 pass) and `rebuild` dropping `retype` on a promoted K→F alias chain (446 pass; inferred effect: the rebuilt item keeps type `K` under an `F` ID, not traced further). Of the 21 new mutants, 17 were killed; a third survivor (`rebuild` parent uid via `birth_of`) looks equivalent for documented items, and one mutant I wrote hit a docstring and is discarded. Killed: incl. longest-slug owner, legacy aliases in `Aliases:`, baseline grandfathering, definer pattern, duplicate-Item, keys lock re-check and reserved set. Add: a header-sync case per kind, and a rebuild test asserting `type` after a promotion chain.

**N5 (minor, ruling) — `check.py` using private members is a follow-up, not a defect.** Nine cross-module private references (`Store._locked`, `_reconcile`, `store._read`, `headers._family_files/_owner/_header_index`, `rebuild._tokens`); behaviour is correct and covered by the suite. Make `Store.read_union()` (lock + reconcile) and public `headers.family_files/owner/header_index`, `rebuild.tokens` before step 9+ consumers copy the pattern. Note `check`, documented read-only, appends missing tracked events to the journal via `_reconcile`, as `read_state` does; say so.

### ACs end to end (real CLI)
- **AC-14/AC-2** two worktrees each `new F --slug … --sync` (x/x2 and y/x3), flush, commit, merge both: no conflict, IDs distinct, headers merge, `check` clean, slug↔ID both ways. With `x` and `x2` both registered `x2.md` and `x2-coordination.md` carry `x2`'s ID only (a pre-registration `x` sync stamped `x2.md` and the later `x2` sync correctly replaced it). A document created after `new F` is reported (`family-no-item`) until synced.
- **AC-7** `--set` with no-ID, malformed, duplicate, dangling, illegal-parent and a legacy id: all reported (legacy id informational); headers with `GAL-F12`, duplicated `Item:` across slugs, two `new` events for one ID, duplicate registry lines (informational) reported; prose `GAL-F0777`, `K-030`, `C-1001` silent.
- **AC-8/11** link set/changed/cleared and home/touched/milestone changes leave the ID constant; unknown component refused for `home`, `touched`, `new --home`.
- **AC-10** keys: suggest, claim, reserved/malformed/duplicate refusals; N3 for `init`. **AC-12** rebuild from headers restores `GAL-I03-F0001` as alias (N1 for what it cannot).

### Dry run over a copy of this repo (`q4`)
Copied `docs/`, `claude/*/kaizen/plan.md`, every component `docs/`, `init` with the step-12 components, `legacy_namespaces` K-: 13 agents + `falkor-chat`, C-: `cpg`, `legacy_paths` `docs/BACKLOG.md`→`cpg`, `claude/*/kaizen/plan.md`→`$1`, `*/docs/BACKLOG.md`→`$1`: **`wi check` clean, rc 0.** With `legacy_namespaces` emptied it reports exactly **15** `namespace` findings (13 agent plans, `falkor-chat`, root `docs/BACKLOG.md`), equal to an independent `grep -lE` count of 15. Caveat: no document there carries `Item:` yet, so the header rules are idle in this run; they were exercised only in the temp repos above.

### Keys through the symlink (`kk`)
Skill copied to `team/work-items`, symlinked at `home/skills/work-items`, `wi key claim GALM` run from a different empty git repo via the symlink: wrote `team/work-items/keys.json` (the team copy), dup claim refused. Note the reach is guaranteed by `wi.py`'s `Path(__file__).resolve()` (modules load from the real path); `keys_path()`'s own `.resolve()` is redundant. `WI_KEYS_FILE` silently redirects claims; keep it test-only.

### Plan statements now stale against the code (for routing)
1. 3.7: "committed code leaves `--sync` a stub that raises; step 6b implements it" — implemented.
2. 3.9: `wi init` writes also `.work-items/.gitignore`; "refused if … equal to a declared legacy prefix" — the code hardcodes `K`/`C`, which the `[A-Z]{2,6}` rule already excludes (N3); `claim` has `--no-pool`; init does not call the key check.
3. 3.2: `link` event lacks `op`; 3.6: `mint-legacy` lacks `--home`.
4. 3.8/step 12: the path→home map is `project.json` `legacy_paths` with `*` captured as `$1`; the plan never names it. `check` exits 1 on any non-informational finding; `--set` accepts a list or `{items: […]}`; extra kinds `family-no-item`, `absent`, `item-slug`, `parent: parent is X, its path says Y`.
5. 3.7 rebuild: builds one `new` at the oldest alias plus one `move` per alias step (not one `new` plus aliases), refuses `--write` over a non-empty registry, and its residual is wider than stated (N1). `headers sync` takes an optional ID (default: every slugged item).
6. 3.3: `ALLOWED_PARENT` is not a table in `rules.py` (an if-chain in `validate_parent`); the literal matrix is in the tests.
7. 3.4/m2: retired items remain usable as parents/moves — state it or forbid it.
Plan growth (3615 → 3910 words) was not gated here; it tracks new scope (steps 6, 6b, 3.5 link model).

### Open questions
N1's remedy (history-first recovery vs floor flag) needs the architect before step 12 writes the real registry.

### Proposed REVIEW.md lines
None.

## Pass 9 (impl, steps 7b–12; close check)

**Scope.** `skills/work-items/` and the real registry at HEAD (c321abea, 20db8176, e5fa1fe1, d70103b2, 9da81acb, 2cc4c4ee), against the plan as the architect revised it after Pass 8. Executed: the suite, `wi.py` in three scratch clones of this repo (`…/scratchpad/p9a.sh`–`p9d.sh`, `c1`–`c5`; the real tree was only read, `check` and `resolve`), header rules against a scratch copy, 5 drift-test mutants. Root `REVIEW.md` applied.

**Verdict: approve with suggestions — but the suite is red at HEAD (P9-1); fix that before close.** No design blocker. Pass 8's major (N1) is fixed for the cases the plan scopes. CPG: considered, not relevant — script package and prompts, no graph.

### Pass 8 dispositions
- **N1 fixed in scope.** In a scratch clone with the registry deleted from the tip and the journal gone: `rebuild --write` restored 12 events from 2 committed versions; next F/K came back above the committed maxima (`F0006`, `K0003`), `mint-legacy architect:K-004` and `cpg:C-310` returned the original IDs (`K0002`, `F0005`), the promoted alias resolved, `check` rc 0 with `info rebuilt`. IDs allocated after the last commit of the registry (flushed or journal-only) were reissued (`F0006`–`F0008`, `S03`, `K0003`), which is the plan's stated residual (3.7 "What this guarantees"); `--floor` covers it. No history (registry never committed): `--write` refuses with the reissue warning; with `--accept-reissue --floor F=3` the next F was `F0004` and `check` reports `rebuilt`. A `--depth 1` clone of the real repo keeps the tip's registry (1 version, `check` rc 0), so the guard cannot fire there. One gap: P9-2.
- **N2 fixed** (refusal of a registered slug is in c321abea; not re-run beyond the green tests `test_alloc`). **N3 fixed** (`init --key FR` and a claimed key refused per plan 3.9/8b; tests green). **N4 fixed** (header-kind and promotion-type tests added; suite green except P9-1). **N5** follow-up not taken; still a follow-up, no consumer copied the pattern (step 7b added `check._rebuilt_findings` only).
- **Stale statements:** 1 (--sync stub) fixed, 2 (`init`/keys text) fixed, 3 (`op`, `--home`) fixed, 4 (`legacy_paths`, finding kinds, `--set` shapes) fixed, 5 (rebuild description, residual) fixed, 6 (`ALLOWED_PARENT`) fixed in 3.3, 7 (retired items stay usable parents) fixed in 3.4.

### New findings
**P9-1 (major, verified) — the committed suite fails.** `cd skills/work-items && …python -m pytest tests -q` → **1 failed, 499 passed**: `tests/test_keys.py:190 test_shipped_keys_json_is_an_empty_list` asserts `keys.json == []`, and step 12 (2cc4c4ee) legitimately claimed `GAL` there. The step's message reports `wi check` clean, not the suite. Replace the pin with what it was guarding (the shipped file parses and each entry has `key/repo/claimed/feeds_pool`; no fixture key such as `ZZZ` leaked into it); do not delete it, and do not re-empty the real file.

**P9-2 (minor, verified) — the rebuild guard looks at the post-reseed state, so history is unreachable when any ref tip carries an older registry.** In a clone where the registry was removed at the tip but `origin/main` still held the 3-event step-12 copy, `rebuild --write` answered "the registry is not lost (3 items present)": opening the transaction had already reseeded from the stale tip, so the committed versions in history were never applied, and allocation continued from the stale counters (`new F` → `F0002` while history had up to `F0005`). The 3.7 sentence "reseed covers the rest" is only true when a tip holds the newest copy. Fix: judge "lost" from the on-disk journal and tracked file (both absent/empty), and apply history ∪ tips; test: the stale-origin clone above, asserting next F is above the committed maximum.

**P9-3 (minor, verified) — the skill is not reachable yet.** `~/.claude/skills/work-items` does not exist and no `wi` is on `PATH`; the prompts say `wi …` while `SKILL.md` says to alias `python <skill-dir>/scripts/wi.py`. This is the plan's machine step; list it as an open close item and have the prompts' first mention carry the invocation form or a pointer to `SKILL.md`'s.

**P9-4 (nit) — bare `K-004` lists all 14 declared homes**, 8 of which do not define it (6 do: independent `grep` of definers). Bare `K-` names therefore can never resolve (by the plan's declared-homes design); FR-12's "unambiguous bare name resolves" only ever applies to `C-`. Say so in `SKILL.md`, and consider listing defining homes first.

### Executed checks
1. **N1/shallow:** above. 2. **Real tree:** `wi.py check` rc 0, `flush` leaves `git status` clean for `.work-items`. Header rules on a scratch clone of HEAD with `second-feature.md`: no `Item:` → `no-id`; `new F --slug … --sync` → clean; wrong `Item: GAL-F0009` → `dangling` + `family-mismatch`; `Tracks: GAL-F2, GAL-F0002-FR7` → `malformed` + `absent`; same `Item:` on two slugs → `duplicate`; the real plan/review headers (`Item: GAL-F0001`, four documents) resolve through `resolve work-item-numbering`. 4. **Step-12 data:** independent grep finds 14 `K-` definer files (13 agent plans + `falkor-chat/docs/BACKLOG.md`) and 1 `C-` (`docs/BACKLOG.md`), equal to `project.json` (14 and `["cpg"]`); real tree: bare `K-004` exit 3, `C-310` exit 5, `cpg:C-310` exit 5.

### Prompt-side ACs (diff of d70103b2; verified by reading, no test exists)
- **AC-10 agreement:** `tico.md` — propose with `wi key suggest`, run `wi key claim` then `wi init` "only after the stakeholder agrees — ask, never claim on your own". Carried, by wording only.
- **AC-6 "opened" trigger:** `tico.md` and `teco.md` — mint "on its first cite or when work on it is first opened … never edit the old document". Carried, by wording only. `architect.md`: cite by full ID, `wi new K`; `claude/AGENTS.md`: flush with the allocating unit and not mid-merge, binding every committing role.
- Root `AGENTS.md` (9da81acb): the two optional header fields added in the existing bullet; no line over 700 chars. The file is 3186 words (smell ~2,500, already exceeded before; this adds ~25); `claude/AGENTS.md` 3476. Not a finding for this feature.

### AC coverage
| AC | Passing test (suite) | Other |
|---|---|---|
| 1, 13, 16 | `test_alloc`, `test_rules` (literal matrix), `test_fold` | CLI Pass 7 |
| 2 | `test_concurrency`, `test_store` (8 processes, worktrees, differing-order merge) | executed Pass 7/8 |
| 3, 17 | `test_move`, `test_resolve` | executed Pass 7 |
| 4 | `test_alloc` | — |
| 5 | recorders in `test_alloc`, `test_legacy`, `test_store`, `test_headers` | — |
| 6 | `test_legacy` (idempotent, ambiguity, concurrent, merge only B) | **"opened" trigger: prompt wording only** |
| 7 | `test_check` | real-tree and header runs above |
| 8, 11 | `test_links` | executed Pass 8 |
| 9 | `test_move` | — |
| 10 | `test_keys`, `test_init` | **agreement: prompt wording only**; `test_shipped_keys…` red (P9-1) |
| 12 | `test_rebuild` | executed above |
| 14 | `test_headers`, `test_check` | executed Pass 8 |
| 15 | `test_skill_drift` (5 mutants killed: epic, alias, name, two matrix rows) | — |
No AC is without a test except the two prompt-side halves (AC-6 "opened", AC-10 agreement), verified on paper by wording; the plan assigns them to `cobb`'s review.

### Close readiness (list only, per root `AGENTS.md`)
1. Fix P9-1; re-run the suite (expect 500 passed). Decide on P9-2/P9-3.
2. **Spec:** `skills/work-items/` has no `AGENTS.md`/`docs/`; root rule says a spec-less component gets a whole-component spec, checked against the shipped diff by `analyst` (`reviews/<slug>-impl.md`) before its first close. Decide the spec's home (a `skills/work-items/` spec or the skill's `SKILL.md`, which today is a 713-word agent guide, not a contract) and its owner.
3. **Naming:** Passes 7–9 sit in `reviews/work-item-numbering.md` (the plan review); the `-impl` role suffix would have put them in `reviews/work-item-numbering-impl.md`. `teco` to say which is the close-check document.
4. **Documentation impact** in plan §6 still open: root `AGENTS.md` filename-grammar/collision rule 2 and the "`K-` numbered" line, the Structure `skills/` bullet, a `.work-items/` note; `claude/README.md`, `claude/docs/manuals/agent-team.md`; kaizen history for agents beyond the four touched in d70103b2; component backlogs' numbering sentences; root `REVIEW.md` optional `Item:` lint; step 13 (team-engagement AC-1 amendment; cross-initiative `needs` decision); the machine symlink.
5. **`REVIEW.md` additions:** none proposed in Passes 7–9; nothing to apply.
6. **HISTORY entry:** last, citing the updated sections and the check; the root `docs/HISTORY.md` is `cpg`'s, `skills/` has no docs tree, so its home needs a decision. `Tracks: —` (no backlog item).
7. Milestone close flips `Status:` of requirements, plan, coordination ledger and reviews to `archived` (`teco`, mechanical).
