# Review — `docs/plans/python-web-quirks-kb.md` (python-web-quirks on the knowledge base)

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## Pass 1

### Scope and verdict

Reviewed the architect plan `docs/plans/python-web-quirks-kb.md` (479 lines, untracked) against
`docs/requirements/python-web-quirks-kb.md` (FR-1..15, AC-1..14), `docs/plans/python-web-quirks-kb-ml.md`,
and the repo. Static review plus read-only commands: grep, the guard script fed synthetic JSON, the real
chunker, one read-only Cypher against `ws:agent-team`, `get_document`, one docs fetch. Root `REVIEW.md`
read and applied (no component `REVIEW.md` exists for `claude/`, `falkor-chat/`, `skills/`). I did **not**
run `export_kit.py` (no scratch-dir run attempted), did not run `GRAPH.COPY`, and did not run the
`check_content_loss.py` selftest; those items are marked unverified below.
I did not rely on `docs/reviews/python-web-quirks-kb.md`.

**Verdict: needs changes.** One blocker (the merged-unit derivation misses a cross-reference, so the unit
list, the 37+1 count and tests T2/T3/T30 are wrong), plus several majors that an implementer cannot resolve
without re-deciding the design (a production-write guard that is bypassable by flag choice, a checker that
cannot run against the test workspace, a missing probe mode that burns the sealed set, contradictory step
numbering/H1 ordering, an AC-6 test that precedes its own deliverable, oversized steps).

CPG: considered, not relevant — docs, scripts, agent definitions and one comment-only code edit; the questions here are about text cross-references, hooks and ingest mechanics, not call graphs.

Growth-stops check: first draft of a new plan, no prior revision to compare; no restated history found
beyond two small items (see F13). Live-spec check: the plan cites `falkor-chat/docs/SERVER.md` §14.9 and
`:638` only and states no spec change; no restated spec content found.

### Findings

**F1 — blocker — the unit derivation misses §34→§2 (verified).** `skills/python-web-quirks/SKILL.md:891`
(§34, `Starlette BackgroundTasks runs one request's tasks sequentially…`) opens "**The bounded-pool entry
above** is about *total* parallelism; this is the per-request half." The referent is §2 (line 55,
`BackgroundTasks (sync callable) is bounded…`, the 40-thread limiter), 32 sections away. The plan's regex
(`sections? above|below`, D2) cannot match "entry above", and T2 pins "exactly the six lines 587, 624, 633,
670, 951, 974", so the plan's own check encodes the miss. I swept independently (all of
`above|below|earlier|previous|preceding|next|elsewhere|half|sibling|companion|section|entry` over the pin,
then read each hit): the only additional inter-section reference is 891; 744/660/916/754 are intra-section,
439/861/987/999 point outside the file (Appendix A). Consequence: 5 merged units (11 sections), not 4;
31 singletons; 36 gotcha units + `u00` = **37 documents**, not 38. Every "38", "37+1", "42 headings … 37+1
fences" and the D11 recount (T24, T30, "38 documents") changes; a retrieval of §34 without §2 (or vice versa)
breaks FR-3 for the unit pair. Note the 891 sentence also shows the failure class: T3's mutant injects
"the section above", the exact shape the regex already knows, so it cannot catch a differently-worded
reference (REVIEW.md "a check that cannot fail").
Fix: (1) add §2+§34 to D2's table and rederive counts everywhere they appear; (2) change D2's rule from a
regex whose hits are pinned to a **coverage probe over the axes the sources vary on**: noun
(`section|entry|gotcha|note|item|paragraph`) × direction (`above|below|earlier|previous|preceding|next|following`)
× distance (adjacent / non-adjacent) × self-reference; `units` prints every line matching the union of those
words with its enclosing section, and the pinned expectation (T2) is "every printed line has a committed
ruling (referent or `intra`/`external`)", with a planted test: a scratch section saying "the bounded-pool
entry above" with no ruling must fail the build. Plan T2 then cannot be satisfied by a list copied from the plan.

**F2 — major — the production-write guard is keyed to a label the REST call does not use (verified).**
D8/D9: `ingest`/`retire`/`--allow-replace` "refuse `--ws agent-team` unless `--confirm-production`".
But the REST client talks to `--base-url`; the workspace is fixed per server process
(`FALKORCHAT_WS_ID`, `falkor-chat/scripts/start_agent_team.sh`), `--ws` is used only for the Cypher reads
(`gate`/`diverge`), and `GET /health` returns only `{"status":"ok"}` (`falkor-chat/server/falkorchat/app.py:498-512`,
`api.py:58-67`) so the client cannot confirm which workspace it is writing to. `ingest --ws pwq-test
--base-url http://localhost:8200` writes the test units to production with no flag; T19 only tests the
`--ws agent-team` spelling. Separately, `--confirm-production` is one flag for both ingest and the delete paths:
`--allow-replace --confirm-production --ws agent-team` is a production delete by an agent, which the plan's
hard constraint ("No agent deletes a document in `ws:agent-team`") forbids in prose only; D8 also describes
this production `--allow-replace`/`retire` path as the steady-state edit route for `cobb`.
Fix: derive the production determination from the target (port/base-url equals the configured production
URL, or a server-reported workspace id if the plan adds one — a falkor-chat change that is out of scope, so
use the URL), refuse any base-url that is not in an explicit `{test, production}` pair; make the delete
paths (`--allow-replace`, `retire`) refuse production outright with no flag (the stakeholder's `curl` stays the
only production delete, as H1 already is) and state the steady-state edit route accordingly (print the
`curl` line). Tests (new): `--ws pwq-test --base-url <8200>` refuses; `--allow-replace --confirm-production`
against 8200 refuses; mutant: drop the URL check.

**F3 — major — `check_content_loss.py` is hard-wired to `ws:agent-team` (verified, `:177`).**
`LIVE_FETCH_SNIPPET` does `db.select_graph("ws:agent-team")`; none of D4's five changes adds a workspace
parameter. T32 (AC-5) and the S10/S11 checker runs are in `ws:pwq-test`: `--live` would read
`ws:agent-team`, report every manifest id `missing_document`, and cannot pass (or, if ids ever collide, pass
on the wrong data). Fix: add `--ws` (default `agent-team`, byte-unchanged for old calls) to D4 as change 6 with
a selftest row using a non-default name; or require `--dump` for test-workspace runs and say so. Same
for `diverge`/`gate`, which D8/D9 already parameterise.

**F4 — major — (open question a) the captures cannot all be gone before S14, by the plan's own text.**
D11 keeps verified capture `38cd49cd…` "until it is promoted"; that capture restates §34 (I read it:
`get_document` returns "Starlette BackgroundTasks runs one request's tasks sequentially…"). It will remain in
production during the sealed B run and competes for the five chunks on every §34/§2 query, exactly the
confound ("a correct answer from a duplicate would score as a miss", ML §5 channel i) that H1 exists to
remove. D11 also says "S13 does not start before H1" (`:340`) while the table runs H1 after S13 and the
open question says "before S14" (see F8). Fix: decide the capture's fate before S14: either H1 includes it
(promotion text recorded in the BACKLOG item first), or scoring is defined with that one documentId excluded
from the ranked list (a documented post-filter, recorded in the report). State the consequence of "no" to (a).
Also re-derive the S7 list immediately before H1: agents write raw captures at the end of every run (every
agent's Learning-capture duty), so "No agent writes to `ws:agent-team` except S13" (§1) is false for the
chain as run and the S7 snapshot goes stale (new captures about `TestClient`/`BackgroundTasks` are likely).

**F5 — major — Set B: no probe-only mode, and the lock can be consumed by the probe (inferred from S3/T23-T27 vs S8).**
S8 requires a "probe-only pre-ingest run in `ws:pwq-test`" of B, but S3's `eval` modes (A/B/AC-2, floor
sweep, BM25, sealed-once) and T23-T27 contain no probe mode (nor does S9's `eval --set A --before` appear in
S3's list). If `qa-engineer` uses `eval --set B`, it must pass `--sealed-once` and writes `pwq-set-b.scored`,
so S14 "a second run refuses" — the one scoring is spent before the data exist. Fix: specify
`eval --set B --probe` (no recall, no sentinel write, prints rank-1 source and score only) in S3 with a test
that it does not create the sentinel; mutant: probe writes the sentinel -> red. Also: the probe rewrites
"any query that already surfaces from another document" — that removes exactly the hard
duplicate-coverage positives from B and inflates recall; `data-scientist` should rule on it (I flag it, I did
not judge the method). On the seal itself: nothing mechanical keeps B out of an agent's reading path (the
path is under `$HOME`, no hook denies reading it, and S3's `eval` source carries the default path); the
timeline is what protects it (S5 `coder` runs before B exists, S16 `coder` after it is scored; the exposure is
`data-scientist` S12 and `cobb` S13/S15 between freeze and score). Adequate for accidental leakage, not
a lock: say so, keep the path out of S3's source (argument or env var only), and fix S8/S13/S14: §D11 says
"no step brief other than S8 and S13 names the path" but S13 is a production ingest that does not need it
and S14, which scores B, does.

**F6 — major — T35/AC-6 cannot run where the plan puts it.** T35 sits under S11 (live, `ws:pwq-test`), but
(i) the guidance it tests lands in S15a and the pointers in S15b, both after S11; (ii) "with the server
stopped" would stop the shared `:8200` server every agent uses (consumers' MCP is wired to it, not to
`:8210`), a shared-state action the plan never names; (iii) the map assigns AC-6 to "S11/S16", S16 says
"tool-denied session". Fix: move T35 to after S15b, run it as a tool-denied session (the plan's own
S16 wording), and drop "server stopped" or specify a throwaway MCP config pointing at a dead port.

**F7 — major — S6 ordering is self-contradictory and the second server's lifecycle is unspecified.**
`start_agent_team.sh` itself (steps 3-4) runs `create_workspace.sh $FALKORCHAT_WS_ID` and `seed_agent_team.sh`,
so starting the `:8210` process creates `ws:pwq-test`. FalkorDB's `GRAPH.COPY` errors when the destination
exists (docs.falkordb.com `GRAPH.COPY`: "You must run `GRAPH.DELETE` on the destination key first"); the plan
says copy first, but also that the workspace "is created with `create_workspace.sh`" and the script that starts
the server does that. Order must be: `GRAPH.COPY` -> then start the server (its bootstrap/seed steps are
idempotent) -> verify. Whether `GRAPH.COPY` carries the `VECTOR` index on `Chunk.embedding` is **unverified**:
the docs say "indexes" and do not name vector indexes, and I did not write to the instance; S6's "verifies
embeddings are present" must also verify the vector index with `CALL db.indexes()` (I confirmed
`ws:agent-team` has `Chunk | embedding | VECTOR`) and run a known-answer `search_documents` against `:8210`.
Production corpus today: 476 current documents, 767 chunks, **751 embedded** (my read-only Cypher), so the copy
inherits 16 unembedded chunks (K-102); S6 records them (good), but S12's floor decision then runs on a corpus that
differs from production after H1. Lifecycle: `start_agent_team.sh` ends in `exec uvicorn` (foreground, Ctrl+C
to stop); S6 -> S17 spans eight dispatches and no step says who keeps the process alive or restarts it
(`teco`'s `ensure-environment.sh` only covers `:8200`); it also runs `pip install -e` into the shared
`falkor-chat/server/.venv`. S17 "key gone" is `GRAPH.DELETE`, which `.claude/settings.json` asks on: expected,
say so. Compatibility with the shared `falkordb-dev` rule is fine otherwise: the script reuses the running
instance on 6379 and Ctrl+C stops only uvicorn (`start_agent_team.sh` header).

**F8 — major — step numbering drifted; the ordering text contradicts the table.** D11's "Order" paragraph
uses an older numbering: "S9-S11: baseline, AC checks, floor sweep" (floor sweep is S12), "Production ingest
(S12)" (S13), "B scored once … (S13)" (S14), "BM25 … on B inside S13" (S14), "S13 does not start before H1"
(`:340`; the table has H1 after S13). Risk 3 says a floor change "lands in S15a before B" while S15a runs after
S14, and "Sequencing facts" says it "lands before the production run and before B". An implementer
following D11 gets a different chain than the table. Fix: one pass renumbering D11/§6 against the table; decide
where the floor value lives while B is scored (pass the floor to `eval` explicitly and record it in the report,
then S15a codifies the same number), so B is never scored under an edit that has not landed.

**F9 — major — step sizing breaks the unit-sizing rule.** S3 is one dispatch holding: `ingest` (REST,
idempotent, adopt, replace, confirm), `gate`, `diverge`, `retire`, `eval` (recall, floor-applied recall, MRR,
distinct docs, coverage with merged units, negatives, BM25, sealed lock, prefix extraction), manifest I/O, a
fake REST server, a fake Cypher fetcher and T16-T27: well over 3 dependent steps. Split S3a
(ingest/gate/diverge/retire/manifest, T16-T22) and S3b (`eval`, T23-T27 plus the probe mode of F5), with G1
after both. S15b (8 agent files + `audit-team.sh` check 12 + scratch-copy mutant) and S16 (about 14 files
across `AGENTS.md`, `claude/README.md`, `skills/README.md`, `claude-code.md`, `SERVER.md`, `app.py`,
`storefront_api.py`, `test_storefront_api.py`, `workflow-timers.md`, `kaizen/plan.md`, `claude/AGENTS.md`,
kit re-export, `rm -r`, final sweep, AC-6) exceed ~5 files per dispatch: split S15b into grants/pointer
edits and audit check 12; split S16 into the `coder` code/doc edits, the `cobb` doc edits, and a last unit that
performs the deletion and the final sweep. Shared-live-key serialisation is fine (only S13 writes
`ws:agent-team`, subject to F4's note on learning captures); S1/S3a/S3b share `pwq_kb.py` and are ordered, S2 is
independent and could run beside S1.

**F10 — major — test checks that cannot fail, or whose expected values come from the thing checked.**
The plan states "every check below names the mutant" (§5); of T1-T36 only T3, T5, T8, T12, T13, T21, T25,
T31, T34 name one. Sampled (twelve) and judged:
- T2 (expected set = the plan's own six lines) and T30 (expected "42 headings and 37+1 fences", numbers
  taken from D2) read expectations from the derivation under test; expected values should come from the pin
  (`grep -c '^## ' pin` = 42; fences = 42 − Σ(members−1)). Mutant for T2/T3: see F1's planted "entry above".
- T3's named mutant exercises only the regex-known wording: cannot turn red on the real miss (F1).
- T5's mutant ("add a date to the header the body lacks -> `verify-copy` fails") does not follow from D3/D9:
  the header is outside the verbatim comparison (D3, D4) and `verify-copy` is specified as "bodies equal the
  pinned bodies". State that `verify-copy` recomputes every header (`Versions`, `Verified`, `Gotcha:`,
  member order) and compares; otherwise T5/T6 are unreachable by their mutants.
- T8 (change one body char -> names the unit): sound. T12 (widen the blockquote -> gap): sound, since a gap is
  allowed only when its text equals the waived span exactly (D4.3); add the positive control (unchanged
  source passes).
- T13: planting an extra source heading turns it red; sound. T21 (null embedding): sound; I confirmed
  `MATCH (d)-[:HAS_CHUNK]->(c:Chunk) … count(c), count(c.embedding)` runs and distinguishes unembedded
  chunks on the live graph (767 vs 751) and `Document.status` takes `processing/ready/failed`
  (`repository.py:1080,1150-1186`).
- T25: sound, but say how the f-string literal in the fence (`f"…\nQuery: {situation}"`) is parsed (not `eval`).
- T16: the fake REST server must be derived from the real request model (`producedBy`, `sourceFormat`; the
  MCP form is `produced_by`); mutant: send the snake_case name -> fake rejects. T17/T18/T19/T20/T22/T23/T24/T26:
  no mutant named; suggested ones: skip-check removed (second POST), adopt path re-ingests, count counts any
  document (T24), merged-unit coverage counts the first member only (T23), sentinel file never written (T26).
- T31's "43rd title appears in the listing" is true by construction (the listing is a `grep` of the copy):
  the mutant that matters is a unit whose title a different heading form (`###`, missing fence) hides from
  the grep while `diverge` also misses it; add it.
Fix: put a "Mutant" column on the T-table, not just prose; G1 rejects a row with none.

**F11 — minor — (open question b) is framed as one decision but is three, and the fallback invalidates the tested configuration.**
See "Open questions" below.

**F12 — minor — `Verified:` over-claims for incidental dates (verified).** D3: `Verified` = every ISO date
in the member's body. Of the 16 dated sections, three carry a date that is provenance, not verification:
§16 "Observed graphmind-ai-lab 2026-08-26", §26 "(`architect`, 2026-09-08)", §28 "confirmed absent from
`falkor-chat/server/.venv`, 2026-09-08". FR-5 says "its verification date where the section states one";
a header that says `Verified: 2026-09-08` for a non-verification date is a small invention. Fix: restrict the
extraction to dates within a verification phrase (`verified|measured|confirmed|re-run|reproduced`) and let
`cobb`'s per-header review (S4) rule the remainder; T29's "16 dated" would then be a derived number.

**F13 — minor — the retrieval skill and K-026.** (a) `skills/agent-kb-retrieval/SKILL.md` front matter lists
its consumers ("teco, architect, tdd-engineer, frontend-engineer, qa-engineer, analyst, data-scientist, graph-dba,
devops"); adding `coder`, `tico`, `security-expert` to the pointer makes it stale: S15a should refresh it.
(b) K-026's "cheap check" `grep -c '^## .*TestClient'` already returns **4** on the pin (not "currently 2"),
so its own trigger fired before this plan; D10's "u-tc is the parent" does not match what K-026 specifies
(a parent heading). Close K-026 with a pointer to this migration rather than rewriting it. (c) The plan's
"Ready to implement" paragraph restates D2/D1; one cite would do (growth-stops: restated facts).

**F14 — minor — listing bound and write-guard prompts.** The title listing measures about 5.8 KB on the pin
(`grep -n -E '^## '` = 5,767 bytes; plus 37 fence lines at ~18 bytes), so 10,000 passes with ~3.5 KB of headroom, about
25 more gotchas; T9 has no failing mutant (a 43rd 200-char title near the bound). The plan's risk 7 can be exact:
by running the guards on synthetic input, `cobb` gets `ask` for `skills/agent-kb-retrieval/SKILL.md`,
`claude/scripts/audit-team.sh`, `claude/cobb/hooks/guard-cobb-topic-writes.sh` and any `.json` (the manifest);
`allow` for `claude/cobb/python-web-quirks.md`. `tdd-engineer`/`coder` deny-lists (`claude/*/*.md`,
`*/docs/plans/*`) do not match `.py`/`.json` under `claude/cobb/scripts/`, so S1-S3/S5 are not blocked, but
`coder` editing `falkor-chat/docs/plans/workflow-timers.md` (D10) and any implementer editing the copy `.md`
will get an `ask`.

### Checks by brief item

1. **Merged units** — independently derived; one miss (F1). The six lines in the plan are all genuine
   (587→§21, 624/633→§21+§22, 670→§24, 951→§29, 974→§30); adjacency claims correct.
2. **FR/AC traceability** — every FR-1..15 and AC-1..14 maps to a step; no orphan steps (S5/S6/S9/S17 are
   support steps, named under FR-12 or the test workspace). Gaps are in executability (F6, F10), not mapping.
3. **Guard** — verified by feeding the real `guard-cobb-topic-writes.sh` synthetic JSON: `claude/cobb/python-web-quirks.md`
   (relative and absolute) -> `allow` via `claude/*/*.md`/`*/claude/*/*.md`; `skills/python-web-quirks/SKILL.md`
   currently `allow`. Retiring that one OR-list entry cannot change any other path (patterns are independent);
   remaining patterns do not match the old path, so `ask` (my inference from the glob list, not a run of the
   edited script). `tdd-engineer`/`coder` hooks: see F14 (verified from the wrapper globs).
4. **Checker** — plan's reading of `parse_sections` (preamble ignored, heading excluded) and of manifest mode
   (only listed headings examined, so an unlisted section is never looked at) is correct
   (`flag_split_candidates.py:183-197`, `check_content_loss.py:628-640`). D4's design (header strip, parts per
   heading, preamble pseudo-section, waived span, heading-set equality) is workable; gaps: F3 (workspace),
   F10 (`verify-copy` header recompute). Selftest not run; `locate_claim`'s first-occurrence limitation is
   harmless because each part is located inside one section.
5. **Ingest mechanics** — confirmed: auto-supersede on identical normalized text
   (`repository.py:1793-1863`, SERVER.md §14.9); `produced_by` resolves only an existing `Agent`;
   `search_chunks` returns `chunkId,text,documentId,seq,score`, no title (`repository.py:1266-1316`);
   `get_document` returns `title` and full `text` (called on a live capture). The 10,000-byte bound is the
   plan's own, not a falkor-chat limit (F14). REST/`/health` workspace blindness: F2.
6. **K-102 gate** — the Cypher names exist and work (F10, T21 note).
7. **Test workspace** — F7. `GRAPH.COPY` carrying the vector index: unverified.
8. **Sealed B** — F5.
9. **Sizing / order / gates** — F8, F9. Gate placement is sensible (G1 after code, G2 before test-ws ingest,
   G3 at close); G1 should follow S3b.
10. **AC-9 sweep** — verified: `grep -rIl python-web-quirks .` (excluding `.git`, `.venv`, `node_modules`) = **65**
    files; every one maps to a D10 row or the last-line globs. Both added hits are real
    (`claude/cobb/kaizen/plan.md:25,298,314`; `test_storefront_api.py:1344`). Two rulings need a note: K-026
    (F13b) and `claude/cobb/kaizen/plan.md` K-029 (open item about the skill's listing budget: "historical"
    leaves an open item citing a deleted skill). The sweep is name-based only; I found no citation phrased without
    the hyphenated name (`Python web/async` hits are in kaizen history). Kit re-export: **not run by me**; the
    plan's command is untested (risk 9 admits it).
11. **Mutants** — F10 (twelve sampled).
12. **Safety** — no step passes a delete flag to production by design; no agent deletes in `ws:agent-team` in
    the step table. But the tooling has an agent-reachable production delete behind `--confirm-production`
    and a flag/target mismatch (F2), the learning-capture writes (F4), and `GRAPH.DELETE` in S17 (F7).

### What's solid

- The decision to keep one file as the copy and derive the listing from it by `grep` (FR-11's "stays in
  step" by construction); the header/body split and the delimiter; `claude/*/*.md` already allowing the copy.
- Procedure for the tail-chunk problem (take distinct `documentId`s, then `get_document`) matches how
  `search_chunks` really works (no title in hits), and the chunk estimate (128-135) matches my run of the real
  chunker (135 over per-section units with a ~330-char header; merging lowers it slightly).
- Adopt-not-reingest and the duplicate check in `diverge` correctly defend against the auto-supersede
  behaviour; the gate's use of `count(c.embedding)` is correct for this build.
- Consumer-agent facts (`tools:` lines, retrieval-line locations) all check out.

### Open questions (the three stakeholder questions)

- **(a) H1 before the sealed run — framed wrongly.** It is presented as a stakeholder choice with a default
  of yes, but it is a methodological requirement (ML §5 channel i), and the plan's own retained capture
  `38cd49cd…` means "deleted before B" is not achievable as worded (F4). D11 `:340` says the opposite ordering
  (before S13) from the question (before S14). The real question: what to do with the one capture kept until
  promotion (delete after recording promotion text, or exclude its id from scoring), and the stated consequence
  of answering "no".
- **(b) `u00` — compound and mis-timed.** It bundles: (1) ingest the preamble at all (FR-2 allows "explicitly
  dispositioned"; the plan's `waivedSpans` mechanism could disposition it wholly in the copy), (2) count it as a
  gotcha document for false accepts (stricter than AC-3's "gotcha documents"), (3) the fallback "keep it only in the
  copy, a one-unit change before S13". The fallback changes the document count (37 vs 38 after F1), T24/T28, the
  AC-1 mechanics and the very configuration S9-S12 measured, so it should be decided **before S10**, not after
  the floor sweep. A generic scope document in the shared `ws:agent-team` index also costs every KB user near-domain
  noise, a cost the question does not mention.
- **(c) copy location — the alternative offered is a straw man.** `claude/coder/python-web-quirks.md` is
  not a real option; the live alternatives are a non-agent directory (rejected in D1 for audit check 1, which
  I did not verify) versus `claude/cobb/`. What the question omits: `claude/cobb/` then holds a KB read by
  all eight consumers, not by its owner (the precedents, `graph-dba/falkordb-quirks.md` etc., are consumed by
  their folder's agent), and implementers' deny-lists will prompt on edits to it (F14). The `cobb` choice is
  still sound (the guard already allows it; verified), so this is a framing defect, not a design defect.
- Not a question but needs an owner: `data-scientist` to rule on the probe rewrite rule in F5 and on scoring
  with the retained capture in F4.

### Proposed REVIEW.md lines

Target `REVIEW.md`, heading "Always check" (source: F1, this review): add —
"A plan that derives a set by pattern (cross-references, call sites, citations) is checked by an independent
sweep over a broader vocabulary than the plan's pattern; the plan's own list is a bound. Its test must pin 'every
printed hit has a ruling', not a list copied from the plan." No existing line becomes redundant.

## Pass 2

Scope: the revised plan (464 lines) with the three stakeholder decisions (36 documents, no `u00`, copy at
`claude/knowledge/python-web-quirks.md`). Static; read-only commands, the real guard script on synthetic input,
a re-run of my own sweep.

**Verdict: approve with suggestions.** No blocker remains. Two new majors (N1, N2) are one-sentence plan
fixes and must be folded into the S3a/S3b briefs (or the plan) before dispatch; the rest are minor.

### Dispositions of Pass 1 (rechecked in the plan)

- **F1 fixed.** D2 now has the five merged units (u-bt §2+§34, u-tc, u-st, u-pyc, u-in), 11 sections, 31
  singletons, 36 documents; the probe prints every hit and the build fails on an unruled one (D2, T2, T3). I
  reimplemented the probe vocabulary (noun x direction within 40 characters, DOTALL) over the pin: it prints
  exactly 10 hits — 587, 624, 633, 660, 670, 744, 754, **891**, 951, 974 — all seven inter-section lines
  including "entry above", and no noise; the T2 mutant (planted "the bounded-pool entry above") is reachable
  and T3's mutant (drop the 891 ruling) removes u-bt. The independent-sweep duty at G1 is kept. Residual: see N3.
- **F2 fixed.** D8 target table `{agent-team: :8200, pwq-test: :8210}` keyed on the URL written to; `replace`/`retire`
  test-only and print the `curl` for production; T19/T20 carry URL-check and allow-with-flag mutants. S13's flow
  (`ingest --ws agent-team --base-url :8200 --confirm-production`) matches the table, so production ingest stays
  reachable and needs no replace. Interaction left open: N4.
- **F3 fixed.** D4 change 6 `--ws`, T15 (mutant: ignore `--ws`).
- **F4 fixed in design.** Capture `38cd49cd` recorded in a kaizen item (S7, `claude/cobb/kaizen/plan.md`, a path
  cobb's guard allows) then deleted by the stakeholder at H1; S13b re-derives before H1 and S14 re-derives at
  start; exclusion rule at S14 (flaw: N5). The "no agent writes except S13" claim is reworded to exclude routine
  learning captures.
- **F5 fixed, with a regression.** `--set B --probe` exists (D9, T28: no sentinel, no recall); the path comes only
  from `--set-b-path`/`PWQ_SET_B`; only S8/S14 name it; the seal is stated as procedural. But D9's "`eval` refuses
  to run in a workspace whose gate has not passed" blocks the pre-ingestion `--probe` (S8) and `--set A --before`
  (S9): N1.
- **F6 fixed.** T39 is after S15b1 (step S15c), a tool-denied session plus a dead-port MCP config, `:8200` never stopped.
- **F7 fixed in design.** S6a (throwaway probe of `GRAPH.COPY` vector-index carry) then S6b; copy before server start;
  `teco` keeps `:8210` alive; `:8200` untouched; the unembedded-chunk inheritance is recorded. Whether `GRAPH.COPY`
  carries the vector index remains unverified by design (S6a decides); execution-time.
- **F8 fixed.** D11 order now reads S13 -> S15a -> S13b/H1 -> S14, matching the table; `--floor` explicit; no
  leftover "S13 does not start before H1" (grep).
- **F9 fixed.** S3a/S3b, S15b1/b2/c, S16a-d. S16a still touches about 7 files (comment-only edits plus
  `AGENTS.md`/`SERVER.md`/`workflow-timers.md`): acceptable because mechanical, flagged minor.
- **F10 fixed.** Every T-row has a mutant column; sampled and reachable: T2, T3, T5 (verify-copy now recomputes
  headers), T9 (25 x 200-char titles pushes 6.5 KB past 10,000), T12, T19, T20, T23, T28, T29. T32 says "n/a (derived)"
  against the stated rule: nit.
- **F11 resolved** by the stakeholder decision (no `u00`; no leftover `u00`/38 in the plan, grep).
- **F12 fixed** (`Verified` phrase-qualified within 100 chars, else `Other-dates`; §16's and §26's provenance dates
  go to `Other-dates`; §28's "confirmed absent" counts as verified, defensible).
- **F13 fixed** (consumer list refresh in S15a; K-026 closed, K-029 rewritten; Ready paragraph shortened).
- **F14 fixed** (6.5 KB / headroom stated, T9 mutant, risk 7 exact).

### Interaction checks (brief item 2)

- F5 vs sentinel: T28 separates the two correctly; the conflict is with the gate refusal (N1), not the sentinel.
- S15a before S14 vs S12: consistent (floor decided S12 on the test ws, codified S15a, passed as `--floor` at S14).
- S7 kaizen item vs H1/S14: consistent; S14's exclusion math is the defect (N5).
- Preamble in the copy vs checker vs D6 grep: N6.
- F7 ordering vs lifecycle vs `:8200`: consistent on paper; only the process-lifetime assumption is execution-time.

### Re-run clears (brief item 3)

- **Guard, new path:** the real `guard-cobb-topic-writes.sh` returns `allow` for `claude/knowledge/python-web-quirks.md`
  (relative and absolute) and `ask` for `claude/knowledge/other.json`. With the `skills/python-web-quirks/*` pair
  removed (a scratch copy of the script), the old skill path returns `ask` and the new path stays `allow`.
- **audit-team:** the only `claude/*/` loop is `audit-team.sh:107`, filtered by `[ -f $d$name.md ]`, so
  `claude/knowledge/` is not enumerated; no other script iterates `claude/*/`. `cpg/eval/tools/export_kit.py:47`
  globs `claude/**/*.md` as prompt files, but the re-export reads `git archive` of `cf7b2042`, which predates
  the new directory: no effect.
- **Sweep:** `grep -rIl python-web-quirks .` (excluding `.git`, `.venv`, `node_modules`) = 66 files, the Pass 1 set
  plus this review; D10's rows still cover every hit.
- **Pass 1 "Verified and fine" items whose premises did not change** (Chunk/HAS_CHUNK names, auto-supersede,
  `get_document` fields) were not re-run; the plan's `schemas.IngestDocumentIn` (`producedBy`, `sourceFormat`) exists.
- Layout claims the plan will edit exist: `claude/AGENTS.md:5` and root `AGENTS.md:44` ("one folder per agent").

### New findings

**N1 — major — the gate refusal blocks the pre-ingestion runs.** D9: "`eval` refuses to run in a workspace whose
gate has not passed". S8's `--set B --probe` and S9's `--set A --before` run before any ingest, so no gate has
passed: both are refused as written (and `--before` exists only in S9's row, not in D9's mode list beyond a
bracket). Fix: the refusal applies to modes that score the feature documents (`--ac2`, `--sweep`, `--set B
--sealed-once`, A without `--before`); `--probe` and `--before` are exempt, with a test (mutant: exempt mode
refused -> red).

**N2 — major — S14's exclusion rule under-fetches.** D11: "`limit = 5 + |excluded|`, drop excluded ids, score the
first five remaining". `limit` counts chunks and one excluded document can hold several of the top chunks (the
captures run to 3-4 chunks), so five remaining may not be reached and the harness silently scores a shorter list
(T29 only tests ordering of exclusion vs truncation). Fix: request a fixed larger limit (the API allows 200) and
truncate after filtering, or sum the excluded documents' chunk counts; T29 mutant: an excluded document holding
3 chunks -> fewer than five scored -> red. Record in the S14 report that the convention's `limit=5` is deliberately
exceeded here.

**N3 — minor — ruling key.** D2 keys rulings "by the hit's text snippet": "section above" occurs in four different
sections (587, 670, 951, 974), so a snippet-only key collides. Key on (enclosing heading, snippet).

**N4 — minor — stale unit with its old document still current.** D8's production edit path is edit copy -> `diverge`
-> stakeholder `curl` -> `ingest`. If `ingest` runs first, an edited text creates a second current document under the
same title (a duplicate, or a pending suggestion), and an identical-text failed document would be *adopted* by the
adopt rule and fail the gate again. Add a rule and a T-row: `ingest` refuses a unit whose manifest entry or title
already has a current document with different text ("delete it first: `curl …`"), and adopt skips documents with
`status=failed`.

**N5 — minor — preamble block grammar.** D3 defines only the unit grammar; `<!-- preamble -->`/`<!-- preamble-note -->`
have no stated end marker, the checker's preamble mode (D4.3, T12) is invoked as `--stored-format pwq <pinned source>`
with no argument naming the copy, and the D6 listing regex `^(<!-- unit |## )` does not match either preamble fence
although D6 says the listing "includes its fence" (and T33 counts only unit fences). Fix: add the preamble grammar to
D3, add `--copy <path>` to the preamble mode, and either change the regex to `^(<!-- (unit|preamble) |## )` or drop
the claim.

### Growth-stops (brief item 4)

Net shorter (464 lines vs 479) with new scope (target table, probe design, mutant column, S-split) and reworked
findings replaced in place; no finding-to-resolution table or disposition list; findings cited by ID only (`F1`,
`F3`, `F12`, `F13`: allowed). One restatement to trim: §1's "Stakeholder decisions folded in" sentence announces
history; the decisions are already stated as design in D1/D2/D11. Not blocking.

### Another static pass?

No: a third plan review has low expected value. Everything checkable statically has been checked, and N1/N2 are
sentence-level. The residual risk is execution-time: whether `GRAPH.COPY` carries the vector index (S6a), the
`:8210` process lifetime, the unrun kit re-export (S16c), that the probe vocabulary holds on a future copy (G1's
independent sweep), and mutant reachability in real code (G1). Dispatch S1-S3b with N1, N2, N4 added to the briefs.

## Appendix A — the cross-reference sweep (F1)

Pin `3fcb171c`; line numbers in `skills/python-web-quirks/SKILL.md` (§ = ordinal of `## ` heading).

| Line | In § | Wording | Referent | Class |
|---|---|---|---|---|
| 587 | 22 | "the section above" | §21 | inter-section (plan has it) |
| 624, 633 | 23 | "the two sections above" | §21, §22 | inter-section (plan has it) |
| 670 | 25 | "the `app.state` section above" | §24 | inter-section (plan has it) |
| 951 | 39 | "the same-second section above" | §29 | inter-section (plan has it) |
| 974 | 40 | "the `input()` section above" | §30 | inter-section (plan has it) |
| **891** | **34** | "**The bounded-pool entry above** … the per-request half" | **§2** (line 55) | **inter-section, missed** |
| 660, 744, 754, 916 | 24, 26, 26, 36 | "probe/table/rule above", "earlier-reported" | same section | intra |
| 439, 861, 987, 999 | 15, 31, 41, 42 | "elsewhere in the same codebase", "source entry" | outside the file | external |

Resulting units: u-tc (§21-23), u-st (§24-25), u-pyc (§29+39), u-in (§30+40), **u-bt (§2+34)**; 11 sections merged,
31 singletons, 36 gotcha documents + `u00` = 37.

## Appendix B — guard runs (F14, brief item 3)

`claude/cobb/hooks/guard-cobb-topic-writes.sh` fed `{"tool_input":{"file_path":P}}`:
`claude/cobb/python-web-quirks.md` allow; absolute form allow; `skills/python-web-quirks/SKILL.md` allow (today);
`skills/agent-kb-retrieval/SKILL.md` ask; `claude/scripts/audit-team.sh` ask; `claude/cobb/scripts/pwq-kb-manifest.json` ask;
`claude/cobb/hooks/guard-cobb-topic-writes.sh` ask.
