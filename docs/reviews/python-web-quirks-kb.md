# Review — python-web-quirks on the knowledge base (requirements)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/requirements/python-web-quirks-kb.md`

## Pass 1

**Scope & verdict.** Static review of `docs/requirements/python-web-quirks-kb.md` (Status Interviewing, untracked, first version) against root `REVIEW.md`, the live files it cites, and `docs/requirements/agent-reading-cost.md`. OQ-1 (golden set and recall target) is deliberately pending and not counted. **Verdict: needs changes** (1 blocker, 5 majors; none requires design work, all are requirement edits or stakeholder answers).

CPG: not applicable — requirements/documentation review with no code-level component.

**Growth-stops check (root `REVIEW.md`).** First version of the document, no prior revision, so there is no rework growth, restated history or restated fact to report. The decision log is append-only question to answer with FR/AC refs. One note: the "Problem & current state" bullets restate `skills/agent-kb-retrieval/SKILL.md` (the 0.43 floor, top-5) rather than citing only. That is minor and the numbers are correct, but they will drift. Cite without the numbers.

**Facts verified against files (check 3).**
- `wc -c skills/python-web-quirks/SKILL.md` = 76,291; 42 `^## ` sections; `description:` = 973 characters after folding. All confirmed.
- `grep -rn python-web-quirks claude/*/*.md` has no agent-prompt hits (only `claude/README.md`). Confirmed. The `~/.claude/agents` copies were not inspected, so "no consumer prompt names the skill" is a bound for the repo copies.
- K-100 and K-102 exist in `falkor-chat/docs/BACKLOG.md` (lines 69, 84) and are described correctly (head-only embedding past ~2,048 tokens; 11 `failed` documents with unembedded chunks).
- `claude/cobb/scripts/check_content_loss.py` and `kb-claim-manifest.json` exist. The checker locates each claim's text as an exact substring of a source file's headings, so a source file must still hold the content when AC-1 runs (see M4).
- `skills/agent-kb-retrieval/SKILL.md` matches the doc's summary (0.43, top-5, prefix, `get_document` follow-up).

## Findings

### B1 (blocker) — FR-5 / AC-8 cannot be met under the "no content change" rule; the source does not carry what FR-5 demands
Verified by script over the 42 sections: only 16 contain an ISO date and 32 a version-like token. FR-5 requires every gotcha to expose "last verification date" and AC-8 requires "version pin and verification date visible on the hit". The Out-of-scope bullet forbids changing gotcha content beyond formatting and forbids re-verification. For about 26 gotchas the date does not exist, so either the migration invents one (content change, and a false verification claim), or AC-8 fails. Needs a stakeholder answer, then an edit: for example, "where the source states no date or pin, the hit shows 'unverified date, migrated 2026-11-09' (explicit, not fabricated)", with AC-8 stating the same. Also say where the metadata lives (title line, header block) only as a visible-on-hit requirement, not a layout.

### M1 — The "cache, not the source of truth" preamble and the non-`##` content are unaccounted for (FR-2/AC-1/FR-8)
`SKILL.md` lines 18-32 (H1, scope paragraph, the verified-version caveat, the "This is a cache, not the source of truth. Origin: distilled 2026-08-09 from the analyst inbox" block) sit outside all 42 `##` sections. AC-1 counts only the 42, so this text can be lost with AC-1 green, and FR-8 is in tension with the sentence "this is a cache": after migration the git-tracked copy IS the source of record, and the analyst-inbox lineage is the FR-10 authoring path. Add: the preamble is migrated or explicitly dispositioned (retained in the stub, moved to the durable copy header, or retired, with the "cache" statement corrected), and AC-1 is stated over the whole file, not a section count. Also drop the hard-coded "42": derive it at migration start (the claim is true today, 2026-11-09, but the AC should fail on a 43rd).

### M2 — Cross-references between sections break FR-3's "self-contained" premise
Verified: 18 lines in `SKILL.md` contain "above/below/see/§". Several are positional references between sections (line 587 "the section above"; line 624 heading "ahead of the two sections above"; line 633 "the two sections above are pinned"). One gotcha retrieved alone makes these meaningless or misleading, and the doc's own premise ("each a self-contained gotcha") is not true of all 42. Require an inventory of positional cross-references at migration start with a disposition per case (reword to a title reference, or co-locate the two as one unit), and have FR-3 say what a hit does when a gotcha depends on another. This is the "a hit is usable without a second lookup in the normal case" clause with no handle on the abnormal case.

### M3 — FR-11 / AC-8 "all titles listable in one call" is not true of the existing tools as stated
`list_documents(current_only=True, limit=50)` (`falkor-chat/docs/SERVER.md` lines 280, 1095) is oldest-first over the whole workspace and `skills/agent-kb-retrieval/SKILL.md` calls it a "manual scan"; it is not in that skill's `allowed-tools` and `ws:agent-team` also holds the distilled KB plus raw captures, far over 50 documents. A 42-title list "in one call" would need a title-prefix filter that does not exist. State the requirement as the outcome (an agent can see all 42 titles in one bounded read, without reading bodies) and let design choose between a git-tracked index, a document, or a tool. Whatever it is, AC-8 must name the exact call, and it must still work in the degraded mode (FR-9).

### M4 — AC-1 and AC-5 conflict on what the "source" is once the skill is gutted (FR-2, FR-6, FR-8)
`check_content_loss.py` verifies claims against a source markdown file. At AC-1 time the source is `SKILL.md`; after FR-6 it is a stub, so the check cannot be re-run from the repo. AC-5 says "AC-1 and AC-2 pass again" after a rebuild from the durable copy; that only holds if the checker's source is the durable copy and the manifest's `documentId`s (which change on rebuild) are regenerated. Say: AC-1 runs once against the pre-migration `SKILL.md` (pinned by commit), AC-5 reruns against the durable copy, and the durable copy's location is not an auto-loaded skill body (else FR-6 and FR-9 "never the whole file" are violated). Also pin the baseline by commit in AC-1 ("as it stands at migration start" is time-dependent while the file is open to edits; the file was edited in recent history).

### M5 — FR-9 / AC-6 are not testable as written, and the unavailable-by-grant case is missing
(a) "An agent is told so explicitly" has no producer: if the KB is down the agent sees a tool error from `search_documents`; nothing in the requirement says what turns that into the "KB unavailable" signal, or whether the stub instructs it. State the observable (the agent reports unavailability and reads the fallback) and which party owns it. (b) Not every consumer has the tool at all: from the agent tool lists, `tico` and `security-expert` carry only `ingest_document`, not `search_documents`/`get_document`; a skill's `allowed-tools` also governs MCP access. Today those agents get the content by activating the skill; after FR-6 they get nothing, and "unreachable" does not cover "not granted". Extend FR-9 to "no retrieval tool available", or decide the stub's `allowed-tools` and record which agents lose access. (c) "Read by section" needs a defined way to do so (grep by heading, offset/limit); AC-6 should say what the test proves. Verified from the tool lists given to this run; the agent definition files were not read.

### M6 — Route inventory for AC-9 is incomplete, and inbound citations are not addressed
AC-9 names `skills/README.md` and "any document describing the skill as holding the gotchas". Verified hits for the skill path outside reviews/kaizen: root `AGENTS.md` line 69, `claude/README.md`, `falkor-chat/docs/SERVER.md` line 638 (cites the skill as the home of the `--deselect` gotcha), `skills/agent-standards/claude-code.md` line 165, `docs/requirements/cpg-agent-adoption.md`. Several cite a gotcha by file path, which after migration points at a stub. Make AC-9 a sweep: every file citing the path is listed, and each hit is either re-pointed (to the title or durable copy) or has an explicit no-change ruling. Run unfiltered (root `REVIEW.md` "no X remains" rule). Frozen reviews and kaizen files need only a one-line ruling.

### m1 (minor) — Chunking and K-100/K-102 handling are stated per gotcha, but one gotcha may be several chunks
AC-2 says "zero have unembedded chunks" and the decision log says "check chunk behaviour on the largest sections first". Verified section sizes: 10 sections exceed 2,000 characters, the largest 6,147 (code-dense, so potentially past the ~5.7k-char window K-100 cites for URL-dense text). FR-3 says "retrievable unit is one gotcha" but ingestion chunks documents, so a unit may be 2+ chunks, with the head-only embedding risk on any that stay single. State the unit as "one document per gotcha" or "one or more chunks, all embedded", and make AC-2 check the tail (query by a phrase from the last paragraph of each section, not only title and situation). AC-2 as written is satisfiable by a head-only embedding that never covers the tail.

### m2 (minor) — Mild design leakage
FR-3 (one gotcha per unit), FR-6 ("pointer" stub body), FR-8 (rebuild procedure), FR-10 (raw capture then `cobb`) and FR-11 ("table of contents") decide shape. Those that are stakeholder decisions (decision log) can stay; FR-3's unit shape and FR-11's "table of contents" should be phrased as outcomes. Not blocking.

### m3 (minor) — FR-10's "without a manual second step being forgotten" is an unfalsifiable negative
AC-7 tests one add. Add the other two verbs the FR names (edit and retire), and the divergence case: the durable copy changes but the KB does not (or the reverse) must be detectable (a drift check, e.g. a per-gotcha id/hash comparison), otherwise "the KB and its source never disagree" (user story 3) has no check. Note this check must be able to fail: name a planted divergence for it.

### n1 (nit) — Overlap with `docs/requirements/agent-reading-cost.md`
No conflict. That document is Ready for design and its FR-6 staleness rule is correctly cited; its retrieval scope (documents and code) does not claim skills. The python-web-quirks content would, however, become a second retrieval mechanism beside the reading-cost one, so say whether the reading-cost design may reuse the same unit shape. Optional. Also correct a small point: `claude/docs/requirements/agent-knowledge-base-strategy.md` records that distilled knowledge as a git-tracked file counterpart was an open question (2026-09-10 decision log), so FR-8 is an answer to it for this one content; cite that.

## What's solid
- Current-state facts (size, section count, description length, K-100/K-102, no consumer prompt names the skill) check out.
- FR-7 (do not fork the retrieval convention) and the Out-of-scope list are crisp; scope decisions are logged with FR/AC refs.
- Treating the KB as a rebuildable index with a git-tracked source of record (FR-8, AC-5) is the right invariant, and the degradation requirement and the authoring path are consistent in principle: both route through the durable copy. Combined with the activation facts, the only inconsistency is M5 (consumers that cannot call the tools).
- The checker precedent for FR-2 exists and is a real tool.

## Open questions
1. B1: for gotchas without a stated verification date, accept "unverified, migrated <date>" on the hit, or require re-verification (which Out-of-scope forbids)? Stakeholder.
2. M5(b): is it acceptable that agents without `search_documents`/`get_document` get content only via the git copy, or must the stub grant tool access?
3. M2: reword cross-referencing sections (a content change Out-of-scope forbids) or co-locate them as one unit?

### Proposed REVIEW.md lines
None.

## Pass 2

Re-review of the in-place revision of `docs/requirements/python-web-quirks-kb.md` (Status still Interviewing, Open questions "None"), read in full against the files. **Verdict: needs changes**, but narrowly: the Pass 1 findings are fixed; the revision's own interactions leave 1 major conflict and 3 minors. CPG: not applicable — requirements review, no code-level component.

### Pass 1 dispositions
- **B1** fixed. FR-5/AC-8 and the decision log (line 167) give "unverified, migrated <date>", "no date or version is invented"; the 16-of-42 fact is stated (line 22). But see N1 for how it can be satisfied.
- **M1** fixed. Preamble described (lines 18-21), FR-2 dispositions it and corrects the "cache" sentence, AC-1 covers the whole file at a pinned commit, count derived, "43rd section would fail". One residue: see m-N3 (FR-12/AC-3 still say 42).
- **M2** fixed. Cross-references stated (line 23), FR-3 co-locates verbatim, AC-2 retrieves each unit as one. Interaction in N2.
- **M3** fixed, tico's variant accepted (judgment below).
- **M4** fixed. AC-1 pins a commit; AC-5 checks "against that copy (not the gutted skill)". The two do not conflict: AC-1 runs once against the pin, AC-5 against the durable copy. Residue: the manifest `documentId`s change on rebuild (inferred from the tool's design, not run); the plan must regenerate it.
- **M5** fixed in substance. FR-9 covers server-down and no-tool, "skill's instructions tell the agent", AC-6 tests both cases separately; FR-13/AC-10 add the grants (N3 for AC-10).
- **M6** fixed, AC-9 is now an unfiltered sweep with a minimum list. See N4: the minimum list is incomplete and the sweep is wider than it assumes.
- **m1** fixed. FR-4 covers "every part", AC-2 queries by the last paragraph and requires zero unembedded chunks.
- **m2** not fixed, left as-is (FR-3 unit shape and FR-11 phrasing are still partly design). Accepted, not worth a round.
- **m3** fixed. FR-10 names add/edit/retire, a detectable divergence, and AC-7 plants one (the check can fail).
- **n1** fixed. The strategy-doc open question is cited (line 35-37).

### Judgment on M3 (AC-8 requires the plan to name the call)
Acceptable. AC-8 states the observable outcome (all titles, one bounded read, normal and degraded) and defers the mechanism, which is the right altitude for requirements; naming a tool here would have been design. The one gap: "bounded" has no bound, so AC-8 cannot fail on size. Require the plan to state the bound (a byte or token figure for the listing) and a planted case (e.g. a 43rd title must appear in it). Minor, folded into the plan gate rather than a requirements edit.

### Growth-stops check
Revision shows growth in the decision log (appended entries, allowed: append-only, one question to one answer with FR/AC refs) and in FR/AC text for new scope (FR-13, AC-10, AC-11, FR-5 wording, the preamble). Rework in place: FR-2/3/4/9/10/12 and AC-1/2/3/5/6/7/8/9 replaced rather than stacked, no finding-to-resolution table, no Pass 1 IDs cited in the document. Two findings: (1) the log entry of line 162 ("No number now; ... OQ-1") and line 164 ("no consumer prompt edits") are overturned by later entries (lines 168, 170). The rules say a reversing entry names the entry it overturns: line 168 does ("partially overturns the ... 'no consumer prompt edits' entry"); line 170 does not name line 162, and 162/OQ-1 still read as open. Add "supersedes the retrieval-quality entry". (2) Line 164's "Skill trigger description only" is now only partly true (FR-13); acceptable as the log is a record. No restated spec content; FR-12 restates the ML plan's numbers (0.90, 76/84, 27/30, 15/15, 12/15) rather than citing only, and those are the duplicated facts most likely to drift: verified they match `docs/plans/python-web-quirks-kb-ml.md` lines 16-20 today (84 positives, 30 negatives, 0.90, 76 of 84, 27 of 30, 15/15, 12/15). Prefer "targets as approved in the plan, §2" to avoid a second copy (minor).

### New findings

**N1 (major) — FR-2 "text verbatim" vs FR-3/FR-5 annotations carried on the hit.**
`check_content_loss.py` locates each stored claim's text as an exact substring of the source (module docstring, "locate, don't diff"), and `search_documents` returns only stored chunk text plus `documentId` (ML plan §2 lines 36-37; `repository.search_chunks`). FR-5 needs "unverified, migrated <date>" on the hit for 26 sections, and FR-3 needs the title on a hit "whichever part matched" while the ML plan itself flags tail chunks that carry no title. Neither can be produced by the retrieval layer, so both must be added text in the stored document, which makes the claim text not a substring of the source and fails the AC-1 check, or the check is changed (out of scope: "content-loss check precedent"). As written the requirements can pass individually and fail together. Fix: FR-2 states that a fixed, machine-delimited header (title, versions, date-or-"unverified, migrated <date>") is added to each stored unit and excluded from the verbatim comparison, and that the check compares the body only; or says where else the annotation comes from. Verified by reading the tool docstring and the plan; not run.

**N2 (minor) — merged units vs the 42-document assumption in FR-12/AC-3.**
The ML plan scores "the expected `documentId` of the migrated gotcha", sets 42 positives (one per section), and counts false accepts against "the 42 gotcha documents" (lines 19, 55, 185-186); its chunk arithmetic (108 chunks) assumes one document per section. The co-location decision (cross-referencing sections as one unit) reduces the document count and changes chunk counts and the confusable lists. FR-12 "every one of the 42 gotchas hit at least once" stays well-defined only if a hit on a merged unit counts for each section in it, and a negative's "none of the gotcha documents" stays defined. State this, and have the plan refresh the ML plan's counts after the unit list is derived. Same cause: AC-3/FR-12 hard-code 42 where AC-1 deliberately derives the count.

**N3 (minor) — AC-10 is not testable for three of the seven agents.**
Verified: `coder`, `tdd-engineer`, `qa-engineer` have no `tools:` line in `claude/<a>/<a>.md` (all tools); `architect`, `analyst`, `security-expert` already list the MCP tools (the `tools:` line is cut at 120 characters in my check, so listing of `search_documents`/`get_document` there was not seen in full); `tico` does not (its line ends at `mcp__cypher__query` in the visible text). "each lists" cannot be true for an all-tools agent. Restate: each agent can call both tools (listed, or tools unrestricted), and name `tico` and `security-expert` as the only edits expected; AC-10 then checks effective access.

**N4 (minor) — AC-9's minimum list is incomplete, and one hit is functional.**
Unfiltered `grep -rl python-web-quirks` (excluding `.venv`, kaizen, reviews) also hits `falkor-chat/server/falkorchat/app.py:220,452`, `storefront_api.py:246` and `falkor-chat/docs/plans/workflow-timers.md` (comment/plan citations of the skill), `cpg/eval/kit-manifest.json` (a hash manifest of `python-web-quirks/SKILL.md`, so editing the skill can break a pin), and `claude/cobb/hooks/guard-cobb-topic-writes.sh:82,105` (a write-guard list of paths `cobb` may write). The last matters to FR-10: if the durable copy leaves `skills/python-web-quirks/`, `cobb`'s guard must still allow writing it, or the authoring path cannot complete. Add these to the minimum list or state "the sweep decides"; add the guard and the manifest as explicit obligations.

### Cross-checks and verification the coordinator asked for
- **FR-9 vs FR-8 vs FR-11 vs FR-13:** consistent. FR-8 keeps the copy out of the whole-load path, FR-9 reads it section by section, FR-11 requires the listing in degraded mode (needs a git-side index, a plan matter), FR-13 shrinks the no-tool set to future agents. No contradiction; the plan must give the degraded listing and the by-section read a concrete mechanism (grep by heading or offset/limit). Not run.
- **Stakeholder decisions consistent:** undated wording (FR-5, AC-8, log 167) yes, subject to N1; tool grants (FR-13, Out-of-scope line 105-106 reconciled with log 168) yes; verbatim co-location (FR-3, AC-2, log 169) yes, subject to N2; AC-3 negative scoring matches ML plan lines 184-186 and the 0.43 floor retained by that plan; approved numbers match the plan (above). Log line 162 is now stale (growth-stops item 1).
- **Raw captures (check 5):** the claim "four raw captures mention the skill; one restates section 34" is only partly verified. A read-only `search_documents` for the skill turned up one such capture, document `38cd49cddb004207a5c7abd10181a4eb` (K-067 backoff, mentions the skill's `BackgroundTasks` section and says the skill "covers the 40/100-token limiter bound but not this per-request serialization"); section 34 is indeed "Starlette `BackgroundTasks` runs one request's tasks sequentially", so the "restates section 34" claim is consistent with that document. The other three could not be confirmed: semantic search cannot enumerate by a literal string and no listing tool is granted to me. This is a bound, not a refutation. The requirements should list the four `documentId`s (or say `cobb` derives the list), since AC-11 tests "the four raw captures" and an unlisted count is not checkable. Note also that `cobb`'s coverage check requires that deleted captures' content is in a promoted unit, and that "restates" does not mean "adds nothing": the quoted capture contains a gap statement about the skill, which should be considered for promotion, not merely deleted.
- **AC-9 sweep hits:** run unfiltered; findings in N4. The six named files do cite the skill; `skills/README.md`, root `AGENTS.md:69`, `claude/README.md`, `SERVER.md:638`, `claude-code.md:165`, `cpg-agent-adoption.md` confirmed.

### Open questions
None that need the stakeholder. N1 needs an owner decision (header vs tolerant check), routable to `architect` with `cobb` for the checker. All four are fixable in the requirements without a new stakeholder question, except N1's choice, which tico may state as an outcome.

### Proposed REVIEW.md lines
None.

## Pass 3

Re-review of the second in-place revision (16.5 KB; Status Interviewing; Open questions "None"), including the stakeholder reversal (no stub; skill deleted; every consumer searches the KB directly). Read in full. CPG: not applicable — requirements review, no code-level component. **Verdict: approve with suggestions** (no blocker; 2 majors and 4 minors, all editable in the requirements without a new stakeholder question). **Stopping signal:** another static pass has low expected value. What remains (degraded-guidance location, pointer literal, the manifest ruling, header format, the degraded title listing) is plan-time design; verify it at plan review and at the AC-9/AC-10/AC-12 executions, not by re-reading this document.

### Dispositions
- **N1** fixed. FR-2 now defines a delimited metadata header plus verbatim body; AC-1 and AC-5 compare the body only and AC-5 regenerates the manifest. FR-3/FR-5 correctly route title and date through that header.
- **N2** fixed. FR-12 counts over the derived unit list ("a hit on a merged unit counts for each section in it"), AC-3 follows, the method-note re-derivation is a stated plan-time obligation; AC-1 no longer hard-codes 42. Residue: the decision log line 198 still says "all 42 hit" (history, acceptable).
- **N3** fixed. AC-10 now reads "listed, or tools unrestricted". Re-verified by `grep -m1 '^tools:' claude/<a>/<a>.md` for every agent definition: `coder`, `tdd-engineer`, `frontend-engineer`, `qa-engineer`, plus `cobb`, `devops`, `graph-dba`, have no `tools:` line; `analyst`, `architect`, `data-scientist`, `teco` enumerate and name `search_documents` (counted twice each); `tico` and `security-expert` enumerate and lack it (0 occurrences). The AC's checked-2026-11-09 sentence is true as written.
- **N4** partly fixed. All hits I named are now in AC-9's list and the guard is an explicit obligation (FR-10, FR-15, AC-12); but the manifest instruction is wrong, see P3-1.
- **Growth-stops notes** fixed. Log line 190 now says it is resolved by the later entry and line 198 names the entry it supersedes; FR-12 cites the plan §1/§4 instead of copying the figures; the raw-capture list is derived by `cobb` and the verified capture's id is recorded. No finding table or Pass IDs in the document; Pass-1/2 rework replaced text in place.

### Growth-stops check on the whole document
New scope accounts for most growth (FR-13 to FR-15, AC-9 to AC-13, the reversal, five log entries; the log is append-only and the two reversals name the entries they overturn: lines 190/198 and 202; line 203 states the plan-time choice). Rework growth to report: (a) line 24 (`description:` is 973 characters, limit 1,024) is now dead, since the skill is deleted; it was only relevant to the stub. Remove it. (b) Line 196 (the tico/security-expert-only grant, "no prompt text") and lines 192-193 are overturned by 202; 202 names 192, 193 and the grant scoping, but 196 still reads as current. Mark 196 "superseded by the 202 entry". (c) The log restates the golden-set figures (198) that FR-12 now cites by reference; log history may keep them. No restated live-spec content.

### New findings

**P3-1 (major) — FR-15/AC-9 instruct the implementer to update a manifest pin that must not be touched.**
`cpg/eval/kit-manifest.json` lists `skills/python-web-quirks/SKILL.md` with sha256 `4728dd86...ddc45` inside its `files` array. The manifest is an export of the kit at the pinned eval commit (`cpg/eval/EVAL_COMMIT` = `cf7b2042...`; `cpg/eval/S40-NOTES.md` lines 13-21: the manifest hashes the kit snapshot, "re-hash of every file on disk" is against `kit-src`, produced by `export_kit.py --repo . --eval-commit <E>`). I confirmed the skill file exists at that commit (`git cat-file -e cf7b2042:skills/python-web-quirks/SKILL.md`); I did not run the verifier. So deleting the skill at HEAD does not invalidate the pin, and "updating" it to the copy's location would falsify a record of what the eval kit contained. FR-15 ("every pin ... updated to the copy's location or retired") and AC-9 ("must stay valid") steer the implementer toward editing it. Restate: the manifest gets a no-change ruling and the AC check is that its verifier (the `cpg/` test or export check that covers it) passes after the deletion; if that verifier re-hashes the working tree rather than the commit, the plan decides, with `cpg/` owner input. Confidence: inferred from the notes and the export tool's role, not from running the verifier.

**P3-2 (major) — FR-9's degraded mode: reachability and AC-6's second leg.**
(a) Every consuming agent now has both retrieval tools (FR-13), and FR-14's pointer exists only in consuming agents, so "an agent without the tools ... following the guidance reachable from its pointer" (AC-6) has no subject: an agent with no tool has no pointer, and one with the pointer has the tool. Either AC-6's second leg is vacuous or it tests an agent outside the eight; say which (for example, a tool-denied session of a consuming agent). (b) The guidance must not live where the failure is: it cannot be a `ws:agent-team` document (the server being down is the case). Today nine agent definitions already point at `skills/agent-kb-retrieval/SKILL.md` ("see ... before calling it"; verified in `claude/{analyst,architect,data-scientist,devops,frontend-engineer,graph-dba,qa-engineer,tdd-engineer,teco}/*.md`), and that skill is the natural file-backed home, since an agent that follows its pointer reads it before the first call. State the constraint, not the file: guidance is file-backed, reachable from the pointer in at most one read, and carries the by-section read method. (c) AC-6's "reachable" is unbounded; fix it as "at most one file read from the pointer".

**P3-3 (minor) — FR-14's pointer largely exists already; AC-13 is not checkable as written.**
The nine definitions listed above already carry a line citing `agent-kb-retrieval`; `coder`, `tico`, `security-expert`, `cobb` do not (0 occurrences for coder/security-expert/tico/cobb in `grep -c`, `search_documents` or the skill path). FR-14 should say the pointer extends that existing line (names the Python/web gotcha class) where one exists, adds it where one does not, and does not create a second competing line. AC-13 "contains the discovery pointer (checkable by search)" needs a fixed literal or token chosen in the plan and grepped across the eight (a drift check of the shape `claude/scripts/audit-team.sh` already uses for the prefix), and the control: removing it from one agent turns the check red.

**P3-4 (minor) — FR-7's second sentence has no AC; FR-6's "listing budget" is unsupported.**
"That convention names this content class" requires an edit to `skills/agent-kb-retrieval/SKILL.md`, whose exact prefix string is guarded by `audit-team.sh`; no AC checks the edit, and the edit must not touch the guarded literal. Add a check (grep for the class name; drift check still passes). Separately, FR-6 cites "competing for the listing budget"; I found no budget document that requires or measures the skill's listing (`skills/README.md`, `claude/AGENTS.md`, `skills/agent-standards/` searched for budget/listing/trim terms), and I found none requiring the skill to be listed either, so the deletion has no such constraint to conflict with. Drop the budget clause or cite its source.

**P3-5 (minor) — The eight-agent set is unexplained.**
The old skill's audience was `coder`, `tdd-engineer`, `architect`, `analyst`, `qa-engineer`; the set adds `frontend-engineer`, `tico`, `security-expert` and omits `data-scientist` (writes Python, already has the pointer and tools), `devops`, `graph-dba`, `teco`. Log 202 says "every consuming agent" without naming them. Record the selection rule (for example "agents that write or review Python") so a later agent is not added or dropped by accident.

**P3-6 (minor) — Ordering and a stale entry.** AC-13 sits before AC-11 and AC-12 (AC-13 follows AC-10); renumbering would break citations, so reorder rather than renumber. FR-11's degraded-mode listing has no stated source of truth: the copy's index must exist without the KB, and the divergence check (FR-10) should cover the index as well as the bodies.

### Cross-checks
- **FR-9/FR-8/FR-11/FR-13/FR-14 together:** consistent apart from P3-2. FR-8's fixed location does not auto-load; FR-9's one-section reads fit it; FR-13 shrinks the no-tool set to a test condition.
- **Guard and AC-12:** `claude/cobb/hooks/guard-cobb-topic-writes.sh` line 82 documents the `skills/python-web-quirks/*` allowance, and it appears in the glob list at line 105. AC-12 (cobb writes to the copy, guard allows) is testable by feeding the guard a synthetic write path, and passes only if the new location is added; the stale `skills/python-web-quirks/*` entry should be retired in the same change (the AC does not say so; add it to FR-15).
- **Leftover contradictions to the reversed decisions:** none in the body outside the expected log entries 192, 193, 196. Problem-statement line 25 ("no consumer agent prompt names the skill") is a current-state fact and remains true.

### Open questions
None for the stakeholder. P3-1's verifier question goes to the `cpg/` owner at plan time.

### Proposed REVIEW.md lines
None.
