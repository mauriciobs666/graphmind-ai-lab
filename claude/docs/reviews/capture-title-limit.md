# Review: raw-capture template states the 200-character `title` limit

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-101 (follow-up)

## Scope & verdict

**Reviewed:** the uncommitted `cobb` change read with `git diff -- claude skills` (28 files, +94/−15):
the Learning-capture write line in 13 `claude/<agent>/<agent>.md`, 13 `kaizen/history.md`
entries, the capture summary in `claude/AGENTS.md`, and `skills/agent-maintenance/SKILL.md` §5.
**Baseline:** falkor-chat HEAD (`7490f634`, K-101). **Standards:** root `REVIEW.md` and
`claude/AGENTS.md`. Neither `claude/` nor `skills/` has its own `REVIEW.md`, and the change does not
touch falkor-chat. No pytest suite was run. The brief forbids FalkorDB-touching suites, and the
change has no tests.

**Verdict: approve with suggestions.** There are no blockers and no majors. The limit and the
"refused" wording are accurate. All 13 copies are identical, and each has its own `produced_by`.
The scope is clean. The open items are follow-ups beyond the declared fence (F1, F2) plus two nits.

CPG: considered, not relevant — the accuracy question is one parameter annotation, settled by reading `mcp.py`/`schemas.py` and calling FastMCP `list_tools()`; no call-graph or data-flow question arose.

## Checks run (evidence)

- **Limit and refusal (verified by running).** `schemas.py:29` `MAX_NAME_LEN = 200`, and
  `mcp.py:57` `OptionalName = Annotated[str | None, Field(max_length=MAX_NAME_LEN)]` applies to
  `ingest_document`'s `title` (`mcp.py:329`). I ran it in `falkor-chat/server/.venv`.
  `list_tools()` gives `title` as `{"anyOf":[{"maxLength":200,"type":"string"},{"type":"null"}]}`.
  `call_tool('ingest_document', {'text':'x','title':'a'*201})` raises `ToolError ... String should
  have at most 200 characters [type=string_too_long]`. Argument validation rejects the call before
  the function body runs, so the probe made no database write. FastMCP returns that error to the
  client as an `isError` result. So "refused" is accurate, and nothing is written or truncated.
  Pydantic counts code points, so "characters" is the right unit.
- **Completeness (verified, unfiltered).** `grep -rn "ingest_document"` and `grep -rn "title=<"`
  over `claude/`, `skills/`, `opencode/`, `kiro/` and root `AGENTS.md`. For each `<agent>`, the full
  new line, with that agent's own `produced_by='<agent>'`, matches exactly once in
  `claude/<agent>/<agent>.md`. That holds for all 13 agents. Every hit outside the change is mapped
  in F1, F2 or the no-change list below.
- **Kaizen entries (verified).** After masking agent names, the 12 non-`cobb` entries hash
  identically. `cobb`'s entry adds the AGENTS.md/SKILL.md scope and flags `SKILL.md:595`. Each entry
  follows the file's existing format (dated `##` heading, **What / Why / Plan items**, newest
  first). The "7 documents over 200" claim is verified: `ws:agent-team` has 438 titled documents,
  7 of them over 200 characters, the longest 243.
- **Scope (verified).** Each prompt diff is exactly one line. The `claude/AGENTS.md` add/edit rule
  (l.309) is met: source and `kaizen/history.md` changed together. The README catalog and the roster
  needed no change, because no agent's role or description changed. `plan.md` was correctly left
  untouched, since no plan item is involved.

## Findings

### F1 — minor — The curator re-ingest title (`SKILL.md:595`) should also carry the limit

`SKILL.md:595` builds `title=<"<family-slug> — <claim-title>">`. A composite title like that is the
likeliest to go over 200. A failure there is worse than one on a raw capture:
- The call comes after the hard `delete_document`. The manifest's `"pending"` marker keeps the claim
  from vanishing silently.
- The refusal is deterministic, not transient. The recovery path (`SKILL.md` ~l.633–706) re-runs the
  same `ingest_document` with the same title, so it fails the same way on every pass. A curator
  following the procedure literally could loop or stall.

Whether any current family title would actually exceed 200 is **inferred, not measured**. The
existing titles mix raw and distilled entries, and I could not separate them by producer.

**Recommendation: yes, bound it.** Add one clause at `SKILL.md:595`: "≤ 200 characters — shorten
`<claim-title>`, never the family slug, if it would go over". This is a change to `cobb`'s
distillation procedure, so `cobb` should log it in its own history entry. `cobb`'s current entry
already flags the gap, so this can be a separate follow-up.

### F2 — minor — The team-KB manual describes `title` without the limit

`claude/docs/manuals/team-knowledge-base.md:62` gives the API signature as `title: str | None =
None,  # a short label`. Its raw-capture bullet (l.69) says "`title` is the one-line fact". Neither
mentions the bound or the refusal, and this manual is the user-facing walkthrough of exactly this
call. It is outside `cobb`'s fence and owned by `tico`. **Suggest:** `tico` adds "≤ 200 characters;
a longer title is refused" to the l.62 comment and the l.69 bullet. Its troubleshooting section can
also name the `string_too_long` error. Verified by reading.

### F3 — nit — The refusal clause is missing from SKILL.md §5

`SKILL.md:341–342` gains "≤ 200 characters" but not the refusal/retry clause that the 13 prompts and
`claude/AGENTS.md` carry. §5 is a summary of the write shape, so this is acceptable. Appending
"(refused if longer)" would keep the three statements of the shape aligned.

### F4 — nit — "Keeping the detail in `text`" could name the `Fact:` line

The `text` template's `Fact:` line is also "the fact, one line". A literal-minded agent could shorten
`Fact:` to match the title. That would lose detail, which is the opposite of the intent. **Suggest**
"keeping the full fact in `text`'s `Fact:` line" instead. Inferred from the wording; this is low
risk.

### No-change rulings (sweep hits outside the change)

These sweep hits need no change:
- `claude/README.md:26` and `claude/AGENTS.md:13`: the call is named without the title shape.
- `teco.md:147`: a handoff check, not a template.
- `cobb.md:78`: a distillation summary.
- `skills/agent-kb-retrieval/SKILL.md`: query side only.
- `devops/ops-quirks.md:193ff`: a probe quirk, with no title-length dependence.
- `claude/docs/plans/*`, `requirements/*`, `test-plans/*`, `cobb/kaizen/plan.md`: historical design
  records; none is a live template.
- `SKILL.md:603` (new-claim ingest): uses the same title convention as l.595, so F1 covers it.

## Prompt safety

The new wording is clear and narrowly triggered. It applies to an over-length `title`, and the
server's error names the limit explicitly, so an agent can recognise the case. It does not invite
retrying on other errors. Shortening below 200 cannot fail validation again, so a retry loop has no
realistic path. It also does not invite dropping facts: it routes the overflow into `text`
(unbounded up to 500,000 characters), so the worst case is a terse title. F4 tightens that further.
Each prompt grows by about 25 words. That is a fair price for keeping a capture from being lost
silently, and every session pays it.

## What's solid

- The change does one thing: 13 byte-identical edits, each with the correct agent's `produced_by`.
- The accuracy claims hold under execution, not just reading.
- `cobb` drew the fence explicitly and flagged the excluded site in its own history, not silently.
- The `claude/AGENTS.md` summary entry was rewritten in place, not appended to. That follows the
  context-file convention.

## Open questions

- **The running agent-team server may predate K-101 (observation, not a defect in this change).**
  This review session's `mcp__falkor-chat-agent-team__ingest_document` schema has no `maxLength` on
  `title`. Its description also lacks the bounds sentence that HEAD `mcp.py:338` carries. So the
  instance serving `ws:agent-team` appears to be running pre-`7490f634` code and would still accept
  long titles until it restarts. The prompts are correct for the code as committed. Whether and when
  to restart that instance is for `teco` or `devops`, not a fix to this change. This is inferred
  from the session's tool schema; I did not probe the live server with an over-length write.

No `REVIEW.md` additions are proposed. The relevant discipline is already the "no X remains is an
unfiltered sweep" line, and applying it produced F1 and F2.

## Pass 2

**Rechecked:** the uncommitted `git diff -- claude skills` (28 files, +100/−18) and `c6e33c9d`.
**Verdict: approve with suggestions.** All four findings are fixed. One new minor finding (F5) is
latent today.

- **F1 — fixed, with a gap (see F5).** Lines `SKILL.md:598–600` and `:607` now carry the bound and
  "shorten `<claim-title>`, never the family slug". Verified by reading.
- **F2 — fixed** in `c6e33c9d`. `team-knowledge-base.md` l.62 and l.69 state the limit, the refusal
  and the retry.
- **F3 — fixed.** `SKILL.md:342–343` carries the refused/retry clause. Backticks are correctly left
  out inside the open code span.
- **F4 — fixed.** The 13 prompts and `claude/AGENTS.md:15` now say "the full fact stays in
  `text`'s `Fact:` line". Rechecked: with `produced_by` masked, all 13 template lines collapse to one
  unique line (`sort | uniq -c` → 13). `grep -rn "keeping the detail\|keeping detail"` finds no
  remaining hit in `claude/` or `skills/`, excluding history and reviews.
- **Kaizen entries — conform.** There is one new dated **What / Why / Plan items** entry per agent,
  newest first. The 12 non-`cobb` entries are byte-identical, and `cobb`'s entry adds the SKILL.md
  scope. A second same-day entry, not a rewrite of the first, is acceptable in an append-only log.
- **Manual consistency — consistent in substance.** `c6e33c9d` says "keep the detail in `text`" and
  does not name the `Fact:` line. That is fine at an end-user manual's altitude; no change is needed.

### F5 — minor (new, latent) — A shortened claim title is never recorded, so recovery can't find it by title

Each claim in the manifest stores `{title, documentId, verified}` (`SKILL.md:566–568`). The
split-sibling lookup goes by that `title` (`:588–590`). Orphan recovery scans `list_documents` for a
document whose title "exactly matches this claim's expected title". It rebuilds that title from the
`"<family-slug> — <claim-title>"` convention (`:674–678`).

F1's shortening step is free-form and cannot be reproduced. The existing-claim bullet tells the
curator to overwrite only `documentId`, never `title` (`:601–602`). So a shortened re-ingest leaves
a live document whose title matches neither the manifest nor the rebuilt convention.

If that ingest is interrupted, the orphan scan reports "Found none" and re-ingests. The result is the
duplicate that this step exists to prevent.

The risk is latent today. The longest of the manifest's 90 claim titles is 185 characters, and none
exceeds 200 (verified by script over `claude/cobb/scripts/kb-claim-manifest.json`). The trace itself
was inferred by reading.

**Suggest (`cobb`):** decide the shortened title once and write it to the manifest entry's `title`
in the same pre-pair write that sets `verified: "pending"`. For a new claim, compute it before the
ingest. Then have the orphan scan match the manifest's recorded `title`, not a rebuilt one. No
existing assertion would catch a wrong version of this design, so the exact wording is `cobb`'s
call.

### Nit — The F1 inline code renders badly

At `SKILL.md:599`, the backticks around `<claim-title>` close the code span that opens at `:596`.
In rendered view, `<claim-title>` becomes a bare HTML-like tag. The raw text agents read is
unaffected. Dropping those backticks, as F3 did, would make the two consistent.
