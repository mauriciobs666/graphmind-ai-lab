# Review: salesperson-ui S16 (Docs close-out) — implementation

> **Status:** active · **Owner:** `analyst` · **Tracks:** S16 (`docs/plans/salesperson-ui.md` §5.1, v1.40)

## Scope & verdict

Reviewed the 8 uncommitted docs-only files implementing `docs/plans/salesperson-ui.md` §5.1's
**S16** row (`coder`, agent `a7414a41ff6a70a4e`): root `AGENTS.md`, `falkor-chat/docs/HISTORY.md`,
`falkor-chat/docs/BACKLOG.md`, `falkor-chat/README.md`, `falkor-chat/AGENTS.md`,
`falkor-chat/docs/SERVER.md`, `salesperson/README.md`, `salesperson/AGENTS.md`. Baseline: the S16
plan row and its acceptance-command block, the root `docs/HISTORY.md`/`docs/BACKLOG.md` headers,
`docs/plans/salesperson-ui2-coordination.md`'s ledger, `docs/test-reports/salesperson-ui-report.md`,
`docs/plans/salesperson-ui-ml.md`. No source or test code is in scope — this is the milestone's
final, docs-only step.

**Verdict: approve with suggestions.** No blocker. One Major (a stale sentence left in root
`AGENTS.md`, outside this diff's own hunks, that now directly contradicts content this same diff
added a few lines above it). Everything else checked — the acceptance command, the
`docs/HISTORY.md`/`BACKLOG.md` scope redirect, every fact-checked commit sha/number/verdict in the
new `HISTORY.md` entries and `K-065`, the word-count/line-length budget — held up under independent
re-verification.

**CPG:** not applicable — this is a documentation-only close-out step with no code-level component
(no source or test files changed; the reviewed artifacts are prose docs).

## Findings

### Major — root `AGENTS.md`'s "Retired components" bullet still calls the shipped app "not-yet-built", contradicting this same diff's own new `salesperson/` bullet 98 lines above it

`AGENTS.md:115-117` (untouched by this diff): *"The retired Streamlit chatbot is
`deprecated/salesperson/`; "the salesperson app" almost certainly means its **not-yet-built
replacement** (`docs/requirements/salesperson-ui.md`) — confirm before touching anything under
`deprecated/`."* This diff's own new Structure bullet (`AGENTS.md:17-23`) and corrected
`deprecated/` bullet (`AGENTS.md:73-76`) both now say the replacement is **delivered**, at
`salesperson/`. A reader who reaches the "Retired components" bullet first — it comes later in the
same always-loaded file — gets told the opposite of what the file says a page earlier, and is sent
to read `deprecated/salesperson/` to "confirm" a replacement that already exists and is documented
two bullets up. This is exactly the staleness class S16's own acceptance command and sweep were
built to catch, but it sits in a bullet the diff's hunks never touch, so a diff-only pass missed
it — evidence-checked by reading `AGENTS.md` whole (not just the diff) and confirming the sentence
is unchanged from `HEAD`.

**Suggested fix:** rewrite the clause to point at the now-delivered app, e.g. *"the retired
Streamlit chatbot is `deprecated/salesperson/`, superseded by the delivered `salesperson/`
storefront (see the Structure bullet above); "the salesperson app" almost certainly means the
current one — confirm before touching anything under `deprecated/`, which is unrelated history,
not a fallback."* Small, single-sentence, in-place edit — same file, same section, no new content
needed elsewhere.

### Minor — the plan's own AC-10 row still cites "root `docs/HISTORY.md`", unamended by S16's justified redirect

`docs/plans/salesperson-ui.md:2897` (§6, AC-10 row, untouched by this change and out of S16's own
file list): *"Not a build gate. S16 records it in `docs/HISTORY.md`..."* — this still names the
plan's original (wrong) target, the one S16 correctly redirected to
`falkor-chat/docs/HISTORY.md`/`BACKLOG.md` after confirming root `docs/HISTORY.md`/`BACKLOG.md` are
CPG-scoped-only. The redirect itself is sound (independently re-verified below), and this row's
edit is `architect`'s to make in a plan amendment, not `coder`'s S16 unit — flagging it only so the
plan doesn't keep pointing the next reader at the wrong file. Low stakes: `K-065` and the delivery
record are already correctly placed regardless of what this one plan sentence says.

**Suggested fix:** a small `architect`-owned plan amendment (or a dated note in the plan's own
change log) correcting the AC-10 row's file citation to `falkor-chat/docs/HISTORY.md` — no urgency,
can ride with any other future plan touch.

### Nit — the "context-file budget" convention's own prescribed `awk` command misreports line
numbers when run across more than one file, as its own usage example does

`AGENTS.md`'s "Context-file convention" bullet prescribes `awk 'length($0)>700{print
FILENAME": "NR}' $(git ls-files '*AGENTS.md')` — run verbatim across every `AGENTS.md` in the repo
(which is exactly how I first ran it for this review, before catching it). `awk`'s `NR` is the
*cumulative* record count across all input files, not the per-file line number; only `FNR` resets
per file. Against this repo's `*AGENTS.md` set, the command silently reports the wrong line for
any hit after the first file (verified: it printed line `697` for a length-980 line that is
actually at line `83` of `falkor-chat/AGENTS.md`). Not S16's bug to fix and not part of this diff,
but it sits in the same file under review and is exactly the kind of "prescribed check that
doesn't do what it says" this convention itself warns against.

**Suggested fix:** swap `NR` for `FNR` in the documented command.

## What's solid

- **S16's acceptance command re-run independently**: `! grep -rEn --exclude-dir=.git
  --exclude-dir=deprecated --exclude-dir=docs --exclude-dir=kaizen --exclude-dir=node_modules
  --exclude-dir=.venv --exclude-dir=dist 'salesperson/(chatbot|cart|customer_profile|
  session_manager|diagnostics|agent|graph|cypher|prompts|utils_common)\.py' .` — exit 0, zero
  matches, confirmed myself, not taken on the implementer's or coordinating session's word.
- **The `docs/HISTORY.md`/`BACKLOG.md` redirect is correct and well-evidenced**: both root files'
  own headers ("Change History/Backlog — CPG code-graph component") confirmed by direct read —
  neither was ever salesperson-ui's target, and the plan's S16 row is simply wrong on this point.
  Root `AGENTS.md`'s own S16 asks (Structure bullet, corrected `deprecated/` bullet, table row,
  "Working in this repo" bullet) are unaffected by the redirect, as they should be — those are
  structural facts about `AGENTS.md` itself, not history/backlog content.
- **Every fact-checked claim in the new `falkor-chat/docs/HISTORY.md` entries and `K-065` holds
  up**: all 13 cited commit shas resolve to the commits the entries describe, with matching
  subjects (`git log -1` on each); the DEF-6/`K-065` percentages (2/10 at 3-way, 5/10 at 40-way,
  never at concurrency=1) and the D→C→B mitigation priority order match
  `docs/plans/salesperson-ui-ml.md`'s own findings/recommendation verbatim; the "230 graph-confirmed
  failed `WorkflowRun`s" DEF-3 figure matches `docs/test-reports/salesperson-ui-report.md` exactly;
  the S12d/S17/welcome-turn-followup review-finding counts (3 Majors, 6 findings, 17 files) all
  match the coordination ledger's own rows. No overclaiming found.
  `K-065` confirmed the next-available number (`K-016…K-064` pre-existing, no gap or collision).
- **`salesperson/README.md` and `salesperson/AGENTS.md` read as present-tense, shipped-app
  documentation throughout** (read whole, not just diffed) — the only "S5"/"scaffold" references
  left are in the file-ownership history table, explicitly framed as history ("the step labels
  below are history... not live parallel-work coordination"), which is the correct place for that
  information per the module-docs convention. A broader sweep for `streamlit`/`kg_pastel`/`scaffold
  only`/`TBD`/`coming soon` across the repo, filtered to outside `deprecated/`/`docs/`/`claude/*`
  kaizen files, found nothing else stale.
- **Root `AGENTS.md`'s own budget check is clean**: `awk 'length($0)>700{...}' AGENTS.md` — zero
  hits (independently re-run); word count 2730→2859 (independently re-run), matching the
  coordinating session's figure. The growth is proportionate to what S16's row asks for — the new
  content cites `falkor-chat/docs/HISTORY.md`/`BACKLOG.md` for delivery detail rather than inlining
  it, which is the right instinct for an always-loaded file. The file was already over its own
  ~2,500-word guideline before this change; this pass had no obligation to right-size an unrelated
  section while adding required content, and didn't try to — a defensible call, not a gap.
  (`falkor-chat/AGENTS.md`'s one line >700 chars, found by running the check against every
  `*AGENTS.md` file, predates this change — line 83 in both `HEAD` and the working tree, byte-for-byte
  unrelated to S16's diff.)

## Open questions

- Should `docs/plans/salesperson-ui.md`'s own `Status:` flip to something other than `active` now
  that S16 — its last row — is delivered? That's an `architect`-owned lifecycle action per the
  document-convention table, not part of S16's own deliverable list, and not something this review
  gates on; noting it only so the milestone close-out doesn't leave the plan's own header stale next.
