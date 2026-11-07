# Team learnings distillation 5 — gate review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-063 · **Extends:** `docs/reviews/team-learnings-distillation4.md`

## Scope & verdict

Reviewed: `cobb`'s uncommitted working-tree diff (16 modified files, the untracked sweep report `docs/reviews/team-learnings-distillation5-sweep.md` and ledger `docs/plans/team-learnings-distillation5-coordination.md`). `resume_live_spec.sh` ignored. Baseline: HEAD `24c0ffce`. Root `REVIEW.md` applied; no component `REVIEW.md` is touched.

Result gated: 40 raw entries → 15 promoted or merged, 24 rejected, 1 deferred (`2885564b`). No raw deletions, no git mutations on my side.

**Verdict: approve with suggestions.** No blocker, but F1 and F2 are wrong or unobserved statements inside a skill file other agents will read, and each is a one-sentence edit; they should land before the commit. The cleared (rejected) set held up: every sampled citation carries the fact, and I found no rejection that dropped a live learning.

CPG: considered, not relevant — the diff is documentation, skill prose, a JSON manifest and kaizen rows; the one code-level check (finding F3) was a static read of three small `falkor-chat` functions, which I read directly.

## Findings

### F1 — major — the Read-tool bullet's "did not reproduce / refused" negative is contradicted by a plain `Read` (verified, executed)

`skills/agent-standards/claude-code.md` lines ~826-836 (bullet "The Read tool refuses a page over 25,000 tokens...") end: the earlier-reported truncation of a one-line file "did **not** reproduce on 2.1.319", and a 233,928-char one-line file "was refused the same way (161,817 tokens)". The sweep (row 30, "Claims narrowed") and `claude/cobb/kaizen/history.md` repeat it as a narrowed claim.

I ran it on the same CLI (2.1.319), same size (233,928 chars, one line, under the 256 KB cap):
- `Read(file)` with **no** `offset`/`limit`, high-entropy ASCII: returned a partial view, "showing the first 22390 of 233928 characters (222011 tokens, cap 25000); this file has very long lines and cannot be paginated by line". Not refused.
- Same, escape-heavy JSON-string text (cobb's likely shape): partial view, first 23,896 of 233,928 characters (208,017 tokens).
- The same escape-heavy file with `limit=1`: refused, "File content (208017 tokens) exceeds maximum allowed tokens (25000)".

So both behaviours exist and the switch is whether `offset`/`limit` is passed. The original entry (`50e13c02`) was right for the plain call; the narrowing is right only for the `limit` form. The bullet as written sends a reader to expect an error where a plain `Read` silently returns ~10% of the file plus a notice. Consequence: an agent that reads a one-line JSON dump without `limit` reasons from a prefix.

Fix: replace the last sentence with both observations, keyed on `offset`/`limit` (partial view with a "cannot be paginated by line" notice without it; refusal with it), and correct the "Claims narrowed" line in the sweep report and the history entry. Separately, "budget by characters well under 40 k" rests on one text: a ~75 k-char ASCII file of common Portuguese words (about 7 k chars per line, ten lines) read in full under the cap in my run, so say "for escape-heavy text" or give no number.

### F2 — major — the transcript-accounting bullet asserts one thing nobody observed (inferred; scan negative)

`claude-code.md` line ~816: "A **resumed** subagent appends to the same file, so summed usage is a cost figure, not a context-size figure." Neither the source entry (`7cd73007`, read in full) nor any sweep row reports it. I scanned all 465 `subagents/agent-*.jsonl`: zero files contain a second real user prompt, and zero have a `queued_command` after a gap over 10 minutes, so I could not find one resumed instance either. The sentence sits in a bullet headed "measured 2026-11-07", so it reads as measured.

Fix: delete the sentence, or keep it marked "not observed in this scan". A second wording point in the sibling hooks bullet: it lists `cypher` among servers whose instructions arrive "though the architect's allowlist names only `mcp__cypher__query`..." — `cypher` is in that allowlist; only `claude-artifacts` and `claude.ai Claude Docs` are tool-less. The headline is correct, the list is loose.

### F3 — minor — cobb's finding 1 / the `2885564b` deferral is overstated: "no coverage" is not true (static read; not executed)

The fact itself is confirmed by reading: `_LlmGuardJudge.__call__` (`falkor-chat/server/falkorchat/app.py:733-810`) calls `self._models.llm(...)` and `llm.complete(...)` with no `try`; only unparseable text becomes `decision=False`. `guards.evaluate_guard` (`guards.py:229-262`) calls `judge(...)` unwrapped. `executor._drive` (`executor.py:686-692`) catches `Exception` → `_fail_with_note` → re-raise. So a judge call that raises fails the run; it does not suspend.

"Executed" extent: none. Existing tests cover only the no-judge case (`tests/test_executor.py:348`); a raising-judge run needs the `wf_repo` fixture, which wipes `reference` state, so I did not run one.

Refuted part: the live-spec set already says it. `falkor-chat/docs/SERVER.md:430` (error map): "a fault during the drive: ... `ModelResolutionError` or `ProviderCallError` (any agent node or LLM guard) → 201/200 `{"status":"failed",...}`", and `falkor-chat/docs/DESIGN.md:638` (fault net: "any other exception fails the run"). Only the sentence at `DESIGN.md:501` omits the case. So there is no spec hole, at most a missing cross-reference, and the deferral to a feature close (K-063 item 2, manifest `deferred_to_owner`) is heavier than the evidence warrants.

Fix: re-word row 6 and K-063 item 2 as "doc'd in SERVER.md:430 and DESIGN.md:638; DESIGN.md:501 could cross-reference", and drop it from the owed items (rejected-as-doc'd is the honest disposition). The manifest counts then become 25 rejected / 0 deferred; do this only if cobb agrees, since the ids are already recorded.

### F4 — minor — three rejection rows cite line numbers the sweep's own edits moved

Row 8 cites `claude/teco/coordination-techniques.md` "line 218" for the heading "A `completed` notification's result can be a stale mid-task placeholder"; it is at line 238 (the sweep inserted 20 lines above it). Row 17 cites `skills/python-web-quirks/SKILL.md` line 860 for "A same-second edit..."; it is at 862 (+2 from the description edit). Row 18 cites `python-web-quirks` line 531 for "passing for the wrong reason"; it is at 533. Content is correct in all three; the numbers were taken before the edits. Every other cited line I opened matched exactly (see Checks run).

### F5 — minor — row 8 (`3bc8105b`) rests partly on a non-repo home

The placeholder half is doc'd (`coordination-techniques.md:238`) and the transcript facts reproduce (entry 647 "Waiting for the notification." at 22:23:13Z, entry 648 a system notification at 00:56:58Z next day). The entry's other lesson, "give mutation harnesses a per-run timeout", is not in any `claude/tdd-engineer`, `claude/teco`, `claude/coder` or skill file I grepped; the row's support is "fixed in that agent's final message", which is a one-off repair. Rejecting it as too generic is defensible, but the row should say that instead of leaning on "Doc'd".

### F6 — minor — an already over-budget always-on description grew by appending

`skills/python-web-quirks/SKILL.md` `description:` goes from 6,256 to 6,493 chars, via an appended "Also: ..." clause; `skills/README.md` line 13 states `description` ≤ 1024 and line 43 says the description is always-on. The `skills/README.md` catalog row goes 6,793 → 6,975 bytes, again appended. The breach pre-exists, so this is not a new defect, but each promotion makes it worse and root `AGENTS.md`'s rewrite-not-append rule is the same logic. Suggest the next owner pass rewrite the description to its triggers instead of enumerating every section. Size bar on steering files: no `AGENTS.md`/`CLAUDE.md` is touched; the awk bar prints only the two pre-existing `falkor-chat/AGENTS.md` lines (87, 88). Added lines over 700 chars are kaizen history/plan rows (not always-loaded) and the README row above.

### F7 — minor — KB claim `fd27e4a9` already contains a falsifiable-by-time statement

It says "The one file under `.lmstudio/server-logs/` that contains the string `Num Offload Layers` is a logged chat request". Today there are two such files (`server-logs/2026-10/2026-10-30.1.log` and `server-logs/2026-11/2026-11-07.1.log`); the second appeared today and its content is an entity-extraction request, so it is most likely the ingest of this very claim. I did not confirm that. The claim is dated and the substance (placement lines live in `main.log`, not `server-logs/`) still holds: `main.log` now has 128 `Num Offload Layers` and 58 `Strict GPU VRAM cap` lines (claim: 127 and 58), and `main.old.log` has 159. Nothing to fix before commit (a fix means a re-ingest plus a user delete); note it so nobody reads "the one file" as current.

### F8 — nit — FastMCP section caveat is loose and first-person

`python-web-quirks/SKILL.md` ends the section with "Not observed here: the FastAPI Enum-class *query* parameter as `anyOf:[{$ref}]` (my probe printed a plain `$ref`)". With fastapi 0.139 I get a plain `$ref` for `c: C = C.a` but `anyOf:[{$ref}, {type:null}]` for `Optional[C]`. Say "plain `$ref` for a required/defaulted Enum, `anyOf` only when Optional", and drop "my probe".

### F9 — nit — hooks bullet should say what "fire" means

The new bullet ("A custom main agent's frontmatter hooks also fire...") sits near notes that a `PreToolUse` `ask` did not pause execution under Auto Mode (`claude-code.md` lines 26-34, 128-140). The new evidence is `hook_success` attachments, i.e. the hook ran, not that its decision was enforced. One clause avoids the two being read as contradictory.

## What's solid

- **Rejection set (a).** 22 of 24 rejected rows' citations opened and matched at the cited file and line (rows 1, 2, 3, 4, 5, 7, 10-13, 15, 16, 18, 19, 20, 23, 36-38, 40, plus 8 and 17 with the shifted numbers in F4). Cheap commands re-run: `awk` bar (prints `falkor-chat/AGENTS.md: 87` and `: 88`), unknown-route GET → `200` + `{"error":"Unexpected endpoint or method. ..."}`, `laya` still `max_context_length: 8192`, `git merge-base --is-ancestor 09e30282 7a270329` true, `DESIGN.md:809` documents `migrate_embeddings.sh`, `guard-judge-understanding/items.jsonl` has 85 items.
- **Embedder-truncation row 19** reproduces: with the embedder loaded at context 2048, inputs of 30,000, 60,000 and 100,000 chars returned identical 1024-dim vectors, 10,000 vs 30,000 chars did not, and `usage` is zeros. (The existing report's 2,003-vs-4,463-char equality does not hold at today's context; the doc'd conclusion, silent head truncation, does.)
- **`3bc8105b`** transcript entries 647/648 match the row's description.
- **Promoted claims re-run (b):** FastMCP (mcp 1.28.1, pydantic 2.13.4, scratch server): `str` `max_length`, `int` `le`, list `max_length`, list-element and dict-key violations each `isError=True` with `Error executing tool t: 1 validation error for tArguments`; schema placement confirmed (`items`, `anyOf` branch for Optional, `propertyNames`, `$ref` for Enum, inline `enum` for Literal); FastAPI puts the list-element bound under `items` too. `pytest -q -q` (9.1.1): `-q` ends `1 failed, 1 passed`; `-q -q` prints the `FAILED` line and no counts line, with and without `-ra`. `PYTHONPYCACHEPREFIX` (CPython 3.12.3): same-length rewrite with the mtime reset printed `AAAA` under the default cache and `BBBB` under a fresh prefix. message.id dedupe over 465 files / 23,972 ids: 18,445 multi-entry, 120 differ (all in `output_tokens` only), the last entry is the maximum for every id, first-vs-last undercount 1.25%, summing every entry gives 52.8 M vs 25.6 M. The bullet's counts (464 files, 23,877 ids, 18,364) are one transcript earlier, which is fine as a dated snapshot. All 20 `<synthetic>` entries carry `error: rate_limit` with session/5-hour/weekly-limit text. Hooks bullet: six tool uses in the cited architect transcript carry both guard hooks; 28 transcripts show the stacking. `mcp_instructions_delta` adds the three servers at entry 20. Read tool: the size-first 256 KB refusal and the `limit` token refusal reproduce (F1 covers the part that does not).
- **Narrowing is real where claimed:** 1.3% (not 3%), 31/33 and 5/18 side-check counts, the `-ra` non-requirement, the "even without MCP tools" narrowing.
- **KB claims (c):** `fd27e4a9` and `ab95ba9f` are `ready`, each with 2 of 2 chunks carrying an embedding (cypher read), and `get_document` text is byte-equal to the `.md` section body (full compare run: 1416 and 1407 chars, equal) — the manifest note says a full compare was not run; it now has been.
- **Manifest (d):** both new keys parse. `dispositioned_not_deleted` = 40 unique ids, all 40 exist in `ws:agent-team` as `Fact:` documents; `promoted` 15 + `rejected` 24 + `deferred_to_owner` 1 partition the 40 with no overlap. The 120 unique ids over the three `_dispositioned_raw_*` keys excluded from the pending query leave exactly **18** (first `98a4de01`). `480ef825`, `a1d388f4`, `a93905a6` are cited by id in `docs/plans/typed-decision-models3.md`, `docs/plans/typed-decision-models-coordination.md`, `model-bench/docs/HISTORY.md`, `model-bench/modelbench/decisions.py`, `model-bench/tests/test_decisions.py` and the two model-bench fixtures; the do-not-delete note is right.
- **Leaks (g):** `git diff -U0` additions and both untracked docs contain no home path, user name, hostname or `/mnt/c/Users/<name>` (only the `<user>` placeholder). The sweep's own line 79 mentions the words "home-path leak" as audit vocabulary.
- **Contradictions (h):** no promoted text depends on a rejected row; row 27 (unreproducible) is not restated; row 17's duplicate-plus-narrow-promotion cross-references the section it complements. The only tension is F9.

## Checks run

Executed: items listed under *What's solid* (scratch scripts under the session scratchpad); `curl` against the local LM Studio; transcript scans of `~/.claude/projects/*/subagents`; cypher reads on `ws:agent-team`. Static only: F3 (code read), the hooks-in-subagents interpretation, spot cites not listed above. Not run: the model-bench or falkor-chat suites (the diff touches no code), any step that loads/unloads a model or wipes graph state, and `claude/scripts/audit-team.sh` (I took cobb's reported counts as given; the diff adds no leak lines by my own grep). I did not verify the sweep's claim that every `ingestedById` is a real agent (the property is not queryable from the document node here).

## Open questions

1. F3: does `cobb` keep `2885564b` as a deferral to the live-spec owner, or re-classify it as rejected-as-doc'd now that `SERVER.md:430` and `DESIGN.md:638` cover it? My read is the latter; it changes K-063 item 2 and two manifest counts.
2. F6: who owns the `python-web-quirks` description budget (README says ≤ 1024, file is 6.5 k)? Out of scope for this gate, but it will be hit again by the next promotion.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, heading *Always check*. Source: F1.

- Add: A narrowing based on "did not reproduce" re-runs the entry's original invocation, with the same parameters (for a tool behaviour: the same flags, `offset`/`limit` included), and records which form was run. A negative from a different call form is a bound, not a refutation.

No prune: nothing in that file is enforced elsewhere by a test or hook.
