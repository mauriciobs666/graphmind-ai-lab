# K-030 Track 2 Stage 9 — distillation-ingestion hook review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (Stage 9)

**CPG:** considered, not relevant — `GRAPHS` shows no `cpg_claude` graph loaded (`cpg_falkorchat`,
`kaizen_team`, `reference`, `ws:acme`, `ws:agent-team`, `ws:demo`, `ws:eval`, `ws:nlq-eval`,
`ws:test`, live-queried). This unit edits a skill procedure's markdown and two kaizen files, no
application source — matches Stage 6/7/8's own finding.

## Scope & verdict

Diff-scoped review of commit `2f0b367` (U7 in
`claude/docs/plans/agent-knowledge-base-strategy4-coordination.md`): `cobb`'s new step 5 in
`skills/agent-maintenance/SKILL.md` (Track 2 Stage 9), plus `claude/cobb/kaizen/history.md`'s new
dated entry and `claude/cobb/kaizen/plan.md`'s K-030 rewrite. Confirmed via `git diff --stat` that
the SKILL.md change is exactly the claimed one addition (62 insertions, 1 deletion — the deletion
is the Origin blockquote's trailing sentence being extended, not a content change); nothing else
in steps 1–4 moved. I did not repeat `teco`'s live smoke test (already independently reproduced by
`teco`) or re-run `claude/scripts/audit-team.sh` (already re-run clean by `teco`).

**Verdict: needs changes.** One blocker: a citation `cobb` uses as supporting evidence for the
manifest-reuse decision (closing item c, leg (c) of three) does not say what `cobb` claims it
says — the word "recommend" does not appear anywhere in the cited document, and its only Stage-9
content addresses a different artifact entirely. One major: the new step 5's own text has no
answer for the partial-completion failure mode the brief asked me to check, and Stage 6's own
migration history shows this failure mode is not hypothetical for this exact ingest path. Neither
finding requires reopening the underlying design (manifest reuse over a fresh
`list_documents`+title-match scan remains sound on its other two legs); both are concrete,
bounded text/design fixes.

## Findings

### Blocker — the Stage 6 review is cited as "recommending" manifest reuse; it never does

**Evidence.** `skills/agent-maintenance/SKILL.md:755` ("`claude/docs/reviews/agent-knowledge-base-
strategy4-stage6.md` recommends it"), `claude/cobb/kaizen/history.md:26-28` ("independently
recommends it"), and `claude/cobb/kaizen/plan.md`'s K-030 row (`Stage 6's own review recommended
it as the Stage 9 mechanism`) all make the same claim, three times. I read the full cited review
(`claude/docs/reviews/agent-knowledge-base-strategy4-stage6.md`, 188 lines) and `grep -n -i
"recommend"` against it returns **zero matches**. The review's only two Stage-9-relevant passages
(lines 28, 77) both name `check_content_loss.py` — the fidelity *checker script* — as what "Stage
9 relies on" / "Stage 9's hook leans on ... unattended," warning about that script's undocumented
false-positive mode on shared-duplicated-label splits. The review never discusses, let alone
recommends, `kb-claim-manifest.json` as Stage 9's `documentId`-tracking mechanism — that question
is simply out of its scope (it gates the migration, not Stage 9's design). The manifest's own
`_comment` (written during Stage 6, commit `a1eb03d`, before the review existed) poses this as an
open question — "see cobb's Stage 6 report for whether it is recommended as-is" — and the review
that followed (commit `7c0caa1`) never answers it.

**Why this matters.** The decision text presents three "converging reasons, not just one" for the
manifest-reuse choice; one of the three is a citation to an independent authority that, on
inspection, offers no opinion on the question at all. A future reader (or `cobb` itself, revisiting
this later) who trusts the citation without re-reading the review — reasonable, since the citation
is stated as settled fact in three separate documents — will believe this design was externally
vetted when it was not. The other two legs (manifest already exists, already keyed on the edit
axis; `check_content_loss.py` treats it as canonical) still stand on their own and remain a
reasonable basis for the decision — this is a citation-accuracy defect, not evidence the decision
itself is wrong.

**Suggested fix.** `cobb`: correct all three occurrences (`SKILL.md:755`, `history.md:26-28`,
`plan.md`'s K-030 row) to either (a) drop the Stage-6-review citation from the reasoning and stand
on the two legs that are actually true, or (b) if `cobb` wants a citation for this exact design
question, get one — a short, explicit sentence added to this review's own "Open questions" section
answering the manifest's `_comment` question, since that question still has no source of record.
Option (a) is the smaller, faster fix and closes the gap without inventing new review scope.

### Major — the new step 5 has no answer for delete-succeeds/ingest-fails, and Stage 6's own history shows this is not hypothetical

**Evidence.** `SKILL.md:768-774` ("Existing claim, text changed") reads, in sequence:
`delete_document(documentId)` first, then `ingest_document(...)`, then — only afterward —
"Overwrite the manifest entry's `documentId` with the new id and set `verified: false`." Nothing
in the step addresses what happens if `ingest_document` raises after the delete already succeeded
(a real, auditable state — `delete_document` is a hard delete with a
`get_document_deletion`-readable audit trail per `falkor-chat/docs/plans/document-ingestion2.md:
585`). On the literal sequence as written, a mid-sequence failure leaves the manifest entry
pointing at a `documentId` that `get_document` now returns `null` for, still carrying whatever
`verified` value a prior successful cycle left it at (commonly `true`) — a claim silently vanishes
from `ws:agent-team` with no flag anywhere recording that it happened. This is not a remote
scenario: Stage 6's own migration hit exactly this class of failure repeatedly (31 documents
transiently `status:"failed"` from LM Studio embedding-backend contention/crashes — ledger rows
U1/U3b/U3c, `claude/docs/plans/agent-knowledge-base-strategy4-coordination.md:27,35-36`) — the
same embedding backend Stage 9's `ingest_document` call depends on for every future revision.

**Why this matters.** The plan's own design rationale for delete-then-ingest (over the
auto/suggested-tier `SUPERSEDES` path) is explicitly to avoid "old and new both searchable until
`confirm_document_update`" (`agent-knowledge-base-strategy.md`, §4 area cited in the diff) — a
staleness window. The chosen design trades that for the opposite and worse window: neither old
nor new content present, unflagged. The `verified: true`/`false` field as specified can only ever
catch a claim that *is* being tracked with a fresh entry — not the case here, since the manifest
write that would set `false` never runs if `ingest_document` is the step that fails.

**Suggested fix.** Two independent, compatible options for `cobb` to choose from, not both required:
1. Reorder to ingest-new-first, delete-old-second. This reintroduces a brief duplicate-searchable
   window (recoverable — worst case is a stale duplicate, not a silent gap) instead of a data-loss
   window, and is a one-clause change to the bullet's own sequence.
2. Whichever order ships, write the manifest's `verified: false` (or a distinct third marker,
   e.g. `"pending"`) **before** calling `delete_document`/`ingest_document`, not after — so a
   manifest read at any point mid-sequence shows an edit in flight rather than stale confidence,
   and add one sentence instructing a future `cobb` to treat any entry it finds `verified: false`
   at the *start* of a fresh distillation pass as a sign the prior cycle didn't complete and needs
   investigation (a `get_document(documentId)` check) before trusting it.
Either belongs in `skills/agent-maintenance/SKILL.md`'s step 5 text itself, not only as a kaizen
note — the step 5 procedure is what a future dispatch actually follows.

### Minor — the manifest's own per-file narrative fields (`_DONE`-suffixed keys, "FILE COMPLETE" notes) will go stale under Stage 9's ongoing edits, and step 5 never says to touch them

**Evidence.** Several of the manifest's top-level keys are Stage-6-specific completion markers —
e.g. `_graph_dba_falkordb_quirks_DONE` with `_note: "FILE COMPLETE (all 5 headings, ...86 claims)"`
— written once, at migration close, to narrate that stage's own finish state. `SKILL.md`'s new
step 5 (`:765-784`) only instructs updating a claim's own `{title, documentId, verified}` entry;
it says nothing about these enclosing `_DONE`/`_note`/`_status` fields. The first real post-Stage-6
edit to, say, `falkordb-quirks.md` (an added/removed/re-split claim) will change the true claim
count under a heading whose parent key still says "FILE COMPLETE (... 86 claims)" and whose key
name still says `_DONE` — both now describing a one-time migration state that Stage 9's ongoing
edits have silently outgrown. This is exactly the "assumption the manifest always reflects a
completed migration state" risk the brief asked me to check for (item 1) — real, though narrow in
blast radius (it only misleads a human/agent reading the manifest's prose, not the structured
`claims[]` data the actual sync logic reads).

**Suggested fix.** `cobb`: add one sentence to step 5 — after a claim add/remove (not a same-count
text revision), also strike or update the enclosing file's `_note`/`_status` narrative if it states
a specific claim count or "complete" status that the edit has just made stale. Low cost, prevents
a future reader from trusting a frozen migration-era summary over the live data beneath it.

### Minor — no concurrent-write guard on `kb-claim-manifest.json`, and the new step 5 doesn't name the risk

**Evidence.** `kb-claim-manifest.json` is a flat JSON file with no locking; step 5's
read-manifest-entry → mutate → write-manifest-entry sequence (`SKILL.md:765-784`) is a classic
read-modify-write race if two `cobb` dispatches (or a dispatch plus a stray concurrent process)
ever touch the file inside the same window — a lost update, not a crash, so it would fail silently.
`claude/AGENTS.md` already documents an analogous git-index race for shared-tree commits in
significant detail; no equivalent note exists for this shared JSON file. In practice `cobb`
distillation passes run sequentially (per the coordination ledger's own dispatch-one-at-a-time
pattern), so likelihood is low — this is why it's Minor, not Major.

**Suggested fix.** One sentence in step 5 or in the manifest's own `_comment`: "not safe for
concurrent writers; `cobb` dispatches touching this file must be serialized," mirroring the
existing git-index-race convention's spirit. No mechanism change needed at current usage patterns.

## What's solid

- **The manifest's actual on-disk structure matches what `SKILL.md` describes.** Read the file
  directly (`claude/cobb/scripts/kb-claim-manifest.json`): `files[<path>].headings[]` →
  `{heading, flagged, claims[{title, documentId, verified}]}`, confirmed by sampling multiple
  files. The "family-slug — claim-title" split-sibling title convention `SKILL.md:770-771`
  describes is real and consistently applied (spot-checked `falkordb-reference.md`'s
  "Cypher on FalkorDB" and "Indexing & constraints" headings, `frontend-quirks.md`'s "i18next" and
  "Testing" headings, `qa-testing-techniques.md`'s model-bench heading).
- **The `MENTIONS`-equivalent framing (item 3) reads as clearly flagged, not solved.** `SKILL.md:
  786-796` states plainly there is no `ws:agent-team` analogue, names the plan's own closing bullet
  as the source, and gives the fallback (claim prose, or a `kaizen_team` entry) without inventing a
  workaround — no risk of a future reader mistaking this for a closed decision.
- **The scope boundary held.** `git diff --stat` on `SKILL.md` confirms exactly one addition (62
  insertions, 1 trailing-sentence edit) — steps 1–4 and the rest of the file are untouched, matching
  the dispatching brief's explicit limit.
- **`kaizen/history.md`'s and `kaizen/plan.md`'s bookkeeping otherwise matches what landed** —
  spot-checked against the coordination ledger's U1–U9 rows and against `SKILL.md`'s actual new
  text; no overclaiming found beyond the Blocker above. The plan's Track 2 stage table (`agent-
  knowledge-base-strategy.md:272-278`) and closing item (c) (`:518-526`) match what `cobb` cites.
- **`check_content_loss.py`'s LIMITATIONS fix and the falkordb-quirks.md 83→86 count correction —
  the Stage 6 review's own suggested-fix items 1 and 2, explicitly timed to "land before Stage 9's
  hook leans on this checker unattended" — are already committed** (`1b9d1c0`, 2026-09-19
  12:16:47, well before this unit's `2f0b367` at 18:07:18). The coordination ledger's row 40 (U3l)
  still shows `in-flight` despite this; see Open questions — that staleness is the ledger's own
  bookkeeping, not `cobb`'s deliverable.

## Open questions

- **For `teco`, not `cobb`:** the coordination ledger's U3l row (`claude/docs/plans/agent-
  knowledge-base-strategy4-coordination.md:40`) is marked `in-flight`, but its described deliverable
  is already fully committed (`1b9d1c0`) — a stale status marker, not missing work. Separately, U10
  (row 55, "apply U9's ready-to-paste SKILL.md floor-section reframe," touching
  `skills/agent-kb-retrieval/SKILL.md` — Stage 7/8's skill, not Stage 9's) is genuinely still
  in-flight as of the most recent commit (`dface48`) — non-blocking per the ledger's own note ("not
  a Stage 8 blocker, does not gate Stage 9's start"), but worth knowing before treating the whole
  K-030 Track 2 arc as having literally nothing left outstanding, since `plan.md`'s rewritten K-030
  row (accurate as authored, at `2f0b367`'s timestamp, before U9/U10 existed) reads as if it does.
- Whether `cobb` wants to pursue Blocker-finding option (b) — actually answering the manifest's own
  open `_comment` question about the reuse decision — or just take option (a) (drop the miscited
  leg) is `cobb`'s call; either resolves the blocker.

## Pass 2 — 2026-09-19 (U7c, focused re-check of commit `7cc266d`)

**Verdict: needs changes.** 3 of 4 findings are cleanly fixed. The major is only partially
fixed: the fix correctly closes the originally-reported scenario but its own recovery logic
introduces a new, unaddressed failure mode of the same severity and character — a claim can still
end up silently unsynced and marked `verified: true`, which is worse than being left `"pending"`,
since `"pending"` at least reads as suspect.

- **Blocker (miscited Stage 6 review) — fixed.** `grep -i "recommend"` and
  `grep "agent-knowledge-base-strategy4-stage6"` against `skills/agent-maintenance/SKILL.md`
  return zero matches — the citation and any mention of that review are both gone from the
  operative file. `history.md:13,78` and `plan.md:471` still say "recommend[ing]" but only inside
  the correction's own past-tense description of the false claim being removed — not a live
  restatement. No new false claim introduced in its place; the design now stands on the two true
  legs, stated plainly in all three files.
- **Major (delete-succeeds/ingest-fails) — partially fixed; see new finding below.** The
  originally-reported scenario (delete succeeds, then `ingest_document` fails) is now correctly
  caught: `documentId` in the manifest stays at the old, now-deleted id until `ingest_document`
  succeeds (`SKILL.md:772-780`), so a future pass's `get_document(documentId)` call correctly
  returns `None` and correctly triggers a full re-run. That part of the fix is sound and verified
  by re-reading the bullet's actual write ordering.
- **Minor 1 (stale narrative fields) — fixed.** `SKILL.md:801-809` adds exactly the instruction
  requested: refresh a file's `_DONE`/`_note`/`_status` narrative after a claim add/remove.
- **Minor 2 (no concurrent-write guard) — fixed.** `SKILL.md:811-816` names the no-locking risk
  and instructs serializing `cobb` distillation passes against the manifest, mirroring the git-index
  race convention as suggested.

### New (Major) — the `"pending"`-recovery check can't tell "completed, unflipped" from "never started," and treats both as safe

**Evidence.** `SKILL.md:794-801`, the new recovery bullet: on finding `verified: "pending"` at the
start of a pass, call `get_document(documentId)` — `None` means re-run the full sequence; **"a
real document back means the prior cycle actually completed and only its manifest flip was left
undone — safe to just set `verified: true` and move on."** This binary check conflates two
genuinely different prior states that both return a non-`None` document:
1. **Truly completed, only the flip forgotten** — `ingest_document` succeeded, so per the bullet's
   own ordering (`:772-778`) `documentId` was already overwritten to the *new* id before the
   interruption. `get_document(new_id)` correctly returns the new content — safe to flip.
2. **Interrupted between writing `"pending"` and calling `delete_document`** (the very first, and
   arguably most likely, interruption point in the whole sequence — before any network/MCP call
   has even run) — `documentId` is untouched, still the *old* id, and the old document still
   exists (delete never ran). `get_document(old_id)` **also returns a real, non-`None` document**
   — but it's the stale, pre-edit text, not the intended new claim. The bullet's instruction says
   to flip this to `verified: true` and move on too — which is wrong: the `.md` file's edit that
   triggered step 5 in the first place is never actually synced to `ws:agent-team`, and the `true`
   flag now actively hides that gap from every future check, which is worse than leaving it
   `"pending"` (at least visibly suspect).

The recovery bullet never compares the returned content against the current `.md` text, and never
tells the reader to check whether `documentId` is the pre- or post-edit id — the two cases are
indistinguishable under the check as written, even though the information needed to distinguish
them (byte-exact match against the source, the same check the pre-existing "Verify, then flip"
bullet already performs) is one clause away.

**Why this matters.** This is the same failure class as the original finding (a claim silently
drops out of sync with its `.md` source, undetected) — the fix moved the blind spot earlier in the
sequence rather than closing it, and made the failure mode strictly worse in one respect: the
false-safe path ends in `verified: true`, which is the state every other check in this procedure
treats as trustworthy, so nothing will ever re-examine that entry again.

**Suggested fix.** `cobb`: in the recovery bullet, replace the bare `None`-check with the same
byte-exact discipline the "Verify, then flip" bullet already uses: `get_document(documentId)` →
`None` → re-run the full sequence (unchanged); **document returned** → additionally confirm it is
byte-exact against the claim's current text in the `.md` file — matches → safe to flip to `true`;
does not match (still old/stale content) → treat identically to the `None` case, run the full
"Existing claim, text changed" sequence to actually apply the edit. One added clause, reusing
existing text/tooling already present two bullets up — no new mechanism required.

### Note — `check_content_loss.py`'s supporting fact was dropped from `SKILL.md`, not merely restated (self-report imprecision, matches `teco`'s own catch)

**Evidence.** `grep -n "check_content_loss" skills/agent-maintenance/SKILL.md` returns zero
matches — the fact is gone from the operative procedure text entirely, present only in
`history.md:84` and `plan.md:469`. `teco`'s message already caught this and judged it
non-blocking; independently confirmed the same via direct grep. Agreed: doesn't affect
correctness, not worth a fix cycle on its own — but if `cobb` is already touching this section for
the recovery-bullet fix above, restoring the one clause ("`check_content_loss.py` already treats
this file as canonical") to `SKILL.md` itself would cost nothing extra.
