# Kaizen agent/learning-note ontology — Plan Review (M8)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (M8)

**CPG:** not applicable — this is a plan-gate review of a design document; the plan under review
itself already carries and I independently confirmed the code-level `CPG:` line (`considered, not
relevant`, `cypher-mcp`/`claude` carry no CPG — `GRAPH.LIST` against the live instance, re-run for
this review, shows only `cpg_salesperson` and `cpg_falkorchat` loaded).

## Scope

Independent pre-implementation review of `docs/plans/kaizen-agent-ontology.md` (`architect`, U2),
gating S3/S4 per `docs/plans/kaizen-agent-ontology-coordination.md` (U3), before any implementation
unit is dispatched. Read in full, by path: the requirements doc
(`docs/requirements/kaizen-agent-ontology.md`, FR-1…FR-8/AC-1…AC-7 and full Decision log), the
graph-dba design note (`docs/plans/kaizen-agent-ontology-graph.md`), the plan itself, the
coordination ledger, `cypher-mcp/server.py` in full (`authorize_write()`, `_author_claims()`,
`_kaizen_entry_create_map_spans()`, and the surrounding module), `cypher-mcp/tests/test_server.py`'s
full write-authorization section (lines 562–815, all 16 existing cases) plus the live section, the
13 agent prompt files' Learning-capture blocks (grepped and spot-checked), `skills/agent-maintenance/
SKILL.md` §5 in full, and `claude/README.md`/`claude/AGENTS.md`. Also independently re-ran `GRAPH.LIST`
and a label count against the live `kaizen_team` graph. Focus, per the brief: security-gate
correctness, sequencing, requirements coverage, and the qa-engineer judgment call — not code style
(no code exists yet).

**Verdict: approve with suggestions.** Grounding, requirements coverage, and regex-vs-recipe fidelity
are all independently verified accurate. One design gap (finding 1) is significant enough that it
should be explicitly resolved — accepted in writing or closed — before S1 starts, but it does not by
itself invalidate the plan's design, and the plan's own stated trust bar (FR-8: "well-behaved callers
can't do this by accident, not hardened against a malicious one") gives a legitimate, if debatable,
basis for accepting it rather than fixing it. I am not marking this a blocker because the plan is
otherwise sound and the gap is inherited from already-shipped code, not introduced fresh by this
plan's own new logic — but teco/the stakeholder should make that call consciously, not by default.

## Findings

### 1 (Major, action required before S1 — decide, don't inherit silently) — a self-attributed decoy `CREATE` lets any agent smuggle a curator-only or attribution-mismatched clause past `authorize_write()` in one statement

`authorize_write()`'s control flow — unchanged by this plan for shapes 1–2, and the new
`_producer_write_agent_id()` check is bolted onto the *same* structure — returns **immediately** the
first time `_author_claims(cypher)` finds a non-empty, self-matching claim (`cypher-mcp/server.py:355–365`,
preserved verbatim per the plan's own "shapes 1-2 keep their exact existing code" statement, §3.1). Coupled
with `_kaizen_entry_create_map_spans()` scanning the **entire** cypher text for *every* `CREATE
(...:KaizenEntry {...})` occurrence (not just the first), a caller can chain a harmless,
self-attributed `CREATE` clause with an *unrelated* second clause in the same `GRAPH.QUERY` call —
completely valid, ordinary multi-clause Cypher, the same chaining style the producer-write recipe
itself relies on (`MERGE` then `CREATE` in one call) — and the whole statement authorizes on the
strength of the first clause alone, with the second clause never checked against anything.

Traced end-to-end, not hypothetical. Today, pre-M8, this already lets a non-curator agent delete an
entry it doesn't own:
```cypher
CREATE (junk:KaizenEntry {entryId:'z1', date:'2026-08-22', fact:'f', evidence:'e', context:'c',
  suggestedHome:'unsure', author:'analyst', createdAt:'t'})
MATCH (victim:KaizenEntry {entryId:'not-mine'}) DETACH DELETE victim
```
called with `agent='analyst'` (not a curator): `_author_claims()` finds exactly one claim
(`'analyst'`, from the first `CREATE`, since the second clause has no `CREATE ...:KaizenEntry {`
shape and produces no span) matching the declared agent, so `authorize_write()` returns `None`
*before ever evaluating `_CURATOR_CLEAR_RE` against the full text* — the entire statement, including
the `DETACH DELETE` of an entry `analyst` neither owns nor curates, executes as one write.

This plan **extends the same structural gap onto its own new, more security-critical check**. The
producer-write mismatch check (§3.1 step 3, "One agent's write cannot be accepted as another's
(FR-8)") only ever runs when `_author_claims()` finds nothing — i.e., it is reachable *only* by a
"clean" producer-write with no decoy `CREATE` anywhere in the text. An agent that wants to forge
another agent's provenance simply prepends the same kind of self-attributed decoy:
```cypher
CREATE (junk:KaizenEntry {entryId:'z1', date:'2026-08-22', fact:'f', evidence:'e', context:'c',
  suggestedHome:'unsure', author:'analyst', createdAt:'t'})
MERGE (a:Agent {agentId: 'cobb'})
CREATE (a)-[:PRODUCED {sessionId:'s'}]->(k:KaizenEntry {entryId:'forged', date:'2026-08-22',
  fact:'fabricated, attributed to cobb', evidence:'e', context:'c', suggestedHome:'unsure',
  createdAt:'t'})
```
called with `agent='analyst'`: authorized the same way, and the forged `(:Agent
{agentId:'cobb'})-[:PRODUCED]->(:KaizenEntry {...})` edge lands in `kaizen_team` — a permanent,
queryable, *wrong* provenance record, exactly the failure mode this whole feature exists to make
impossible (per the requirements doc's own framing: "one-query, team-wide attribution lookups" and
"cross-agent provenance tracing" both depend on `PRODUCED` being trustworthy). The
`producer_agent_id != agent` check the plan is proudest of (§3.1 step 3, graph-dba's "crux") is never
even reached in this scenario.

Neither graph-dba's note, the plan's §3.1 design, nor its §5 test list (17 new cases) or mutation
directive covers this class of statement — every new test is single-shape, single-clause. This is a
real gap in the plan's own stated highest-risk area (§6: "regex/parsing correctness is the
highest-risk part of this delivery").

**Why I'm calling this Major, not Blocker:** it is a property of already-shipped code (shape 1's
early-return + whole-text scan), not new logic this plan introduces; FR-8 explicitly disclaims
hardening against a malicious caller ("well-behaved callers can't do this by accident"); and building
this exact multi-clause statement requires deliberate construction, not an accident of the recipe
templates every agent is instructed to copy-paste verbatim. But this plan is the one that raises the
stakes on the SAME gap (attribution forgery to any agentId, not just an unrelated delete), and it
does so silently — the plan's Risks section (§6) never mentions it.

**Suggested resolution — pick one, but pick one explicitly before S1:**
- **Accept explicitly**: add a line to §6 stating this residual risk is accepted under FR-8's trust
  bar (extending "well-behaved callers, not hardened against malicious ones" in writing to cover
  cross-clause smuggling, not just single-clause decoys), and add one pinned regression test (either
  offline suite) that documents current behavior is unchanged by this plan, not silently inherited.
- **Close it**: generalize the "nothing else follows" anchoring `_producer_write_agent_id` already
  gets (§3.1 step 2e) onto shape 1 too — reject a statement if, after accounting for the matched
  `CREATE ...:KaizenEntry {...}` span(s), whitespace-collapsed leftover text still contains another
  recognized shape's trigger keywords (`DETACH DELETE`, a second `MERGE ...:Agent`, etc.). Worth
  noting for whoever designs the fix: the one already-accepted multi-clause use case (the `§3.4`
  migration batch, `_MIGRATION_CYPHER` in the test suite) is actually a *single* `CREATE` clause fed
  by `UNWIND`, not multiple independent `CREATE` clauses — so this tightening would not appear to
  regress it, though that should be confirmed by whoever implements the fix, not assumed from this
  review alone.

### 2 (Major) — FR-6/AC-3's edge-count-then-decide sub-step ordering is correct only by accident of the existing step numbering, and neither doc states the invariant

Within one distillation pass on one entry, if `cobb` decides (per the new §3.3 Step 3 branch) that the
entry also mentions another agent, the new `MENTIONS` edge must be **durably committed to the graph
before** graph-dba's §4.1 count-and-decide query runs for that same entry. If the count ran first (or
if Step 3's tagging and Step 4's count/delete were ever parallelized or reordered), the just-added
`MENTIONS` edge would not be counted, `otherRemaining` could evaluate to `0` when it should be `>0`,
and the entry's own node would be fully `DETACH DELETE`d (§4.3) *before* the `MENTIONS` edge was ever
attached — silently discarding the very cross-agent link FR-3/FR-4 exist to create, with no error.

The plan happens to get this right, because it describes the change as "Step 3 gains a branch... Step
4 replaces its clear logic" — preserving the SKILL.md's existing 1→2→3→4 order, which is the correct
order. But neither the plan (§3.3) nor graph-dba's note (§4.1, whose own correctness caveat is only
about `OPTIONAL MATCH` fan-out, not step sequencing) states this as an explicit invariant. A reader of
S4's done-condition alone (table row, §4) would not know that this ordering is load-bearing, not
incidental — a future edit to `SKILL.md` §5, or a `cobb` run that decides to batch/parallelize
per-entry work for speed, could reorder them without anything flagging the regression.

**Suggested improvement:** add one sentence to the plan's §3.3 (or S4's done-condition) stating the
invariant explicitly: *"the MENTIONS-tag (if any) for a given entry must be committed before that
entry's §4.1 count-and-decide read runs in the same pass — the count must reflect any edge just added
this pass."* Cheap to state now, expensive to debug later as a silently-vanishing note.

### 3 (Minor) — §4's dependency table overstates S3/S4's dependency on S2 as a hard gate

The step table's "Depends on" column reads `S2 approved` for both S3 and S4 — read in isolation, this
says implementation cannot even *start* until S2 finishes. The prose immediately below the table
says the opposite is also viable: "drafting the text itself has no such dependency... this plan
recommends sequencing S3/S4 after S2's approval anyway, not just before go-live... The cost of waiting
one review cycle is small." These are two different claims (hard predecessor vs. a recommended
ordering with a stated, accepted trade-off) and only the softer one is what the plan actually argues
for. A reader who consults only the table (which is what the coordination ledger's own summary row
does: "S3+S4 (`cobb`, 13 prompts + skill doc, both after S2 approves)") will draw the harder
conclusion.

**Suggested improvement:** soften the table cell to `S2 approved (recommended — drafting can start
earlier; see §4 rationale)`, so the table and prose agree without requiring the reader to reconcile
them.

### 4 (Minor, suggestion) — pin finding 1's scenario as an explicit test case regardless of which resolution is chosen

Whatever teco/the stakeholder decides for finding 1 (accept or close), add the exact reproduction
above as a new, explicitly-labeled test case in `cypher-mcp/tests/test_server.py`'s section 8 — a
compound statement chaining a self-attributed decoy `CREATE` with an unrelated curator-shaped or
mismatched-producer-write clause, asserting whichever behavior was decided on. Today this is
undocumented in the suite: the closest existing tests (14, 16) cover decoys *inside* one clause's
free-text fields, not a second, independent top-level clause chained after a legitimate one.

## Recommendation on the open judgment call (§6: is a `qa-engineer` acceptance pass warranted?)

**Yes — recommend adding one**, specifically *because of* finding 1. A static review (this one) can
describe and trace an adversarial multi-clause scenario, but whether the eventual accepted resolution
(accept-and-pin, or a code-level close) actually holds against the *deployed*, rebuilt container is
exactly what a black-box pass is suited to confirm dynamically that a static read cannot. Concretely:
after S1 lands (and again after S3/S4, per the plan's own suggestion of "a dry-run distillation pass
against real, disposable entries"), have `qa-engineer` drive the live `cypher-mcp` container with (a)
the finding-1 reproduction under whatever agent identity was decided, confirming the resolved
behavior, and (b) one real dry-run of `cobb`'s updated distillation procedure (tag → count → partial
or full delete) against disposable entries in the real `kaizen_team` graph, complementing — not
duplicating — S1's own scripted live acceptance sub-step. This is a narrower ask than M5's full
`qa-engineer` pass (matching the plan's own scoping argument that M8 is narrower), so it should be
cheap relative to the confidence it buys on the plan's own self-identified highest-risk area.

## What's solid

- **Grounding is accurate throughout, independently re-verified, not just trusted.** Every line
  citation into `cypher-mcp/server.py` checked against the real file (`_kaizen_entry_create_map_spans`
  at line 291, `_author_claims` at 334, `_CURATOR_CLEAR_RE` at 265, `authorize_write` at 348, the
  module docstring's "Only two write shapes are ever authorized" at line ~21) — all correct. The
  claim of "exactly two currently-recognized write shapes" is exactly what the shipped code does.
- **All three new curator regexes were hand-traced against graph-dba's literal recipes,
  whitespace-collapsed, and they match verbatim** — `_MENTIONS_WRITE_RE`,
  `_PRODUCER_EDGE_RESOLVE_RE`, `_MENTION_EDGE_RESOLVE_RE` all correctly bind and backreference their
  variables against the exact Cypher graph-dba's §3/§4.2 specify. No ReDoS risk in any of the four new
  patterns (no nested/ambiguous quantifiers; the brace-matching is a manual linear scan, not regex
  backtracking).
- **`_producer_write_agent_id`'s recognition algorithm (§3.1 steps 2a–2f) is sound** for the case it's
  designed for (a single, well-formed producer-write statement, no decoys) — the optional-`sessionId`
  handling, the variable-binding check between `MERGE` and `CREATE`, and the "nothing else follows"
  end-anchor were all traced by hand against the recipe and hold up.
- **Requirements coverage is complete.** FR-1…FR-8 and AC-1…AC-7 all map to a concrete plan step or
  design decision; FR-4/FR-5/FR-6 and AC-3/AC-4 (the most intricate behavioral requirements) are
  correctly reflected in §3.3's three changes and S4's done-condition (modulo finding 2's missing
  invariant statement). FR-7/AC-6 (M7 gate) independently confirmed: `docs/plans/generic-cypher-mcp2.md`
  and its coordination doc are both `Status: archived`.
- **The 13-file count and "structurally uniform Cypher block, non-uniform prose" claim (§2.6) are
  independently confirmed** — `grep -l "Learning [Cc]apture"` finds exactly the same 13 agent files
  plus `claude/README.md` (a non-agent, expected match); the "called as ..." line is stable across
  all of them.
- **Sequencing is otherwise well-reasoned**: S0/S1's parallel-independence claim, S1's own internal
  split (offline suite has no S0 dependency; the one live acceptance sub-step does), and the
  no-migration/no-rollback-risk assessment (§6) are all correctly argued and match what the code and
  coordination ledger actually show.
- **The existing regression suite's characterization (16 cases, migration-batch shape, decoy tests
  14/16, SET-rejection 15/15b) is accurate** — read in full and cross-checked line by line against
  `test_server.py`; the plan's proposed 17 new cases are each concretely tied to a specific boundary
  in the new logic, and the mutation-testing directive correctly targets load-bearing checks (the
  backreference, the trailing-content anchor) rather than cosmetic ones.

## Open questions

- Whether finding 1 is accepted or closed is a call for `teco`/the stakeholder, not something this
  review resolves unilaterally — see the Suggested resolution under finding 1.
- `kaizen_team` currently holds 21 `:KaizenEntry` nodes (re-checked live for this review; the plan's
  own 2026-08-22 count was 20) — one more entry landed since the plan was written (plausibly a normal
  kaizen write in the interim, this review's own included). This does not affect S0's DDL step (still
  zero `:Agent` nodes, confirmed) or any other part of the plan; noted only so the discrepancy isn't
  mistaken for a grounding error on re-read.

## Pass 2 — 2026-08-22

Re-review of Version 2 (`docs/plans/kaizen-agent-ontology.md`, read fresh in full, not from memory of
Version 1 — section numbers shifted: the closure is now §3.1a, the ordering invariant is in §3.3, the
new step is S6). Scope, per the coordinator's brief: trace §3.1a's closure against the real,
unmodified `cypher-mcp/server.py` (S1 hasn't run — this is still a design-level trace, same posture
as Pass 1), re-confirm findings 2–4's resolutions, and hunt specifically for any other combination
that defeats the new check or smuggles something past shapes 1–6 collectively.

**Verdict: needs changes.** Findings 2–4 are correctly resolved, verified below. §3.1a's closure is
real and correctly closes both traced attacks (A and B) — I re-traced each by hand against the
regexes and functions exactly as specified and both are rejected as designed. But the closure is
**under-scoped**: it screens only for a bare `MERGE` or `DELETE` token, not the full write-keyword
set this same file already enumerates elsewhere (`_WRITE_KEYWORD_RE`: `CREATE|MERGE|SET|DELETE|
REMOVE`). A third attack — a decoy `CREATE` chained with a `SET` (or `REMOVE`) clause against a node
the caller doesn't own — passes straight through §3.1a's check untouched, including a variant that
re-opens the *exact* SET-based author-reassignment attack the existing suite's tests 15/15b were
written specifically to keep closed. The plan's Revision note and §3.1a both assert Finding 1 is
"closed" without qualification; that assertion is not yet true as designed.

### New finding (Blocker) — §3.1a's `_has_foreign_trigger_outside_strings()` omits `SET`/`REMOVE`, reopening a SET-based tampering path through the same decoy-`CREATE` mechanism

**Traced against the plan's own code, not hypothetical.** `_FOREIGN_TRIGGER_RE = re.compile(r"\b(?:
MERGE|DELETE)\b", ...)` is deliberately narrower than `_WRITE_KEYWORD_RE` (already defined in
`cypher-mcp/server.py:239`, used for the empty-key pre-classification branch): `CREATE|MERGE|SET|
DELETE|REMOVE`. §3.1a's own "why a bare keyword scan" rationale argues correctly that `CREATE` must
stay excluded (every legitimate shape-1 statement legitimately contains a bare `CREATE`), but gives no
reasoning for excluding `SET`/`REMOVE` specifically — and, by the same logic §3.1a already applies to
`MERGE`/`DELETE`, neither `SET` nor `REMOVE` ever legitimately appears (bare, outside a string) in
either accepted shape-1 statement (a plain author-write or the `_MIGRATION_CYPHER` batch) either:
property keys are `entryId, date, fact, evidence, context, suggestedHome, author, createdAt,
sessionId` — `\bSET\b`/`\bREMOVE\b` word-boundary-match none of them (`suggestedHome` does not
contain "SET" as a contiguous substring, and even if it did, `\b` requires a token boundary the
identifier's own internal characters don't provide). So including them is exactly as safe as including
`MERGE`/`DELETE` already is, and closes a gap the "one narrow fix" framing didn't need to leave open.

**Attack C — SET-chained tampering, traced end to end** (called with `agent='analyst'`, not a curator,
not the victim's own agent):
```cypher
CREATE (junk:KaizenEntry {entryId:'z1', date:'2026-08-22', fact:'f', evidence:'e', context:'c',
  suggestedHome:'unsure', author:'analyst', createdAt:'t'})
MATCH (victim:KaizenEntry {entryId:'not-mine'})
SET victim.author = 'nobody', victim.fact = 'tampered'
```
Trace: `_author_claims()` finds one span (the first `CREATE`'s own map — the `MATCH...SET` clause has
no `CREATE` keyword at all, so `_kaizen_entry_create_map_spans()` never even looks at it), claim
`'analyst'`, matching the declared agent → `mismatched = []`. §3.1a's new check,
`_has_foreign_trigger_outside_strings(cypher)`, scans the *whole* text for a bare `MERGE` or `DELETE`
— finds neither (the text contains `MATCH` and `SET`, not either trigger word) — returns `False`. The
existing `if claims: ... return None` path is reached exactly as before the fix: **authorized.** The
whole statement executes as one write: the harmless decoy entry is created, *and* `victim` (an entry
`analyst` neither produced nor curates) has its `author` and `fact` silently rewritten. A `REMOVE
victim.author` variant is symmetric and equally unguarded. This is a strictly more dangerous
primitive than Attack A (arbitrary property tampering on any node, not just whole-node deletion) and,
for the `victim.author = '<other-agent>'` sub-case, is precisely the SET-based author-reassignment
scenario `test_set_based_author_reassignment_is_always_rejected` (test 15) and
`test_set_map_merge_author_reassignment_is_always_rejected` (test 15b) exist to keep closed — those
two tests only pin a **standalone** `SET`, with no `CREATE` clause anywhere in the same statement
(`claims` is trivially empty, so the statement never reaches the "already accepted, check for
chaining" branch at all); neither pins the chained case §3.1a introduces a branch for. Item 20 in the
plan's own §5 test list ("a legitimate single-clause author-write... still succeeds") does not cover
this either — it only re-confirms non-regression of the accepted shapes, not non-authorization of a
new attack variant.

**Suggested fix — mechanical, matches the plan's own reasoning, no design rework needed:** widen
`_FOREIGN_TRIGGER_RE` to `r"\b(?:MERGE|DELETE|SET|REMOVE)\b"` (equivalently: reuse
`_WRITE_KEYWORD_RE`'s own keyword set minus `CREATE`, rather than hand-picking two of its four
members) — the surrounding string-literal-exclusion logic, the `if claims:` gating (so a genuine
producer-write, which legitimately contains a bare `MERGE`, is never subjected to this check at all,
confirmed by re-tracing `_producer_write_agent_id`'s own path — it returns before `claims` is ever
consulted since a clean producer-write yields zero author-claims by construction), and the rejection
message all need no other change. Add two more pinned regression tests alongside 18/19 (Attack-C's
`SET`-chained variant, explicitly including the `victim.author = '<other>'` sub-case that reopens
15/15b's exact concern, and one `REMOVE`-chained variant), and extend the mutation-testing directive
to also confirm dropping `SET|REMOVE` from the widened regex is caught by at least one of them.

### Re-confirmed: findings 2–4 are resolved as requested

- **Finding 2 (ordering invariant)** — §3.3 now carries an explicit "Explicit ordering invariant
  (analyst review, Finding 2)" paragraph stating, verbatim to what I asked for, that the MENTIONS-tag
  must be committed before the same-pass count-and-decide read, with the exact failure mode
  (premature full-`DETACH DELETE` before the just-added edge lands) spelled out. S4's done-condition
  cross-references it explicitly ("not left implicit"). **Resolved.**
- **Finding 3 (table/prose inconsistency)** — S3 and S4's "Depends on" cells now read `S2 approved
  (recommended — drafting can start earlier; see §4 rationale)`, matching the softer framing the
  surrounding prose already argued. **Resolved.**
- **Finding 4 (pin the scenario as a test)** — §5 items 18 and 19 reproduce Attacks A and B verbatim
  as new pinned offline tests, regardless of which resolution was chosen (here, closure); item 20
  independently confirms — and I independently re-verified by inspecting `_MIGRATION_CYPHER`'s literal
  fixture text myself — that its `UNWIND` list and `CREATE` clause contain neither `MERGE` nor
  `DELETE` as bare tokens, so the closure does not regress it; item 21 extends decoy-robustness (a
  free-text field that itself quotes "MERGE"/"DELETE") to the new check, and I confirmed by tracing
  `_string_literal_spans()` that such a quoted occurrence is correctly excluded — a real instance of
  this already exists in the live `kaizen_team` graph today (this reviewer's own Pass-1 kaizen entry,
  whose `evidence` field quotes both "DETACH DELETE" and "MERGE" inside a single-quoted string
  literal), which is a small, free, real-world confirmation that the exclusion logic behaves as
  designed. **Resolved**, with the caveat that none of 18–21 (nor the mutation directive) exercises
  the `SET`/`REMOVE` gap above, since that gap wasn't identified until this pass.

### Everything else re-checked in this pass, no new issues found

- Re-traced §3.1a's "scope of the fix" argument (shapes 2/4-6 already fully `^...$`-anchored against
  the whole whitespace-collapsed statement; shape 3 immune via its own step-2e end-anchor) against
  three additional combinations I constructed myself: a decoy prepended before a producer-write
  (breaks `_producer_write_agent_id`'s `\A` anchor, and no other shape's fixed skeleton matches the
  resulting text either → correctly rejected overall), a decoy suffix appended *after* a clean
  producer-write (breaks step 2e's trailing-content check the same way → correctly rejected), and two
  independent, both-self-attributing `CREATE (...:KaizenEntry {..., author:'<same-agent>',...})`
  clauses in one statement (both claims match, no foreign trigger — authorized, but this is a benign
  multi-entry self-write, structurally the same trust boundary as the already-accepted migration
  batch, not a new hole). None of these defeat the design beyond the SET/REMOVE gap above; the
  "scope of the fix" reasoning about *which shapes are chainable onto* is correct — the omission is
  narrowly in *which trigger keywords are screened for*, not in that reasoning.
- S6's dependency wiring (`S1` deployed, `S3`, `S4` landed) and its stated non-duplication of S1's own
  live acceptance sub-step both check out against §4's dependency-graph section — no gap found.

**Not yet ready for S0/S1 dispatch.** Once `_FOREIGN_TRIGGER_RE` is widened to include `SET`/`REMOVE`
(or reuses `_WRITE_KEYWORD_RE` minus `CREATE`) and two corresponding regression tests are added to
§5's list, I have no further findings and would expect to approve on a Pass 3 pass over just that
delta — this does not require re-reviewing the rest of the plan, which stands as verified in Pass 1
and this pass.

## Pass 3 — 2026-08-22

Re-review of Version 3 (`docs/plans/kaizen-agent-ontology.md`, read fresh in full — structure
confirmed stable versus Version 2, changes confined to §3.1a's regex/prose, §5 items 22-23, the
mutation directive, and the two revision notes, exactly as reported). Scope, per the coordinator's
brief: verify the widened `_FOREIGN_TRIGGER_RE` and its message are actually in place; re-trace Attack
C against the widened regex myself; confirm items 22/23 pin what they claim; one more adversarial
pass for any remaining gap or new false-positive risk; a conclusive final verdict.

**Verdict: approve. Ready for S0/S1 dispatch.**

### Verified in place

- **§3.1a, line 431**: `_FOREIGN_TRIGGER_RE = re.compile(r"\b(?:MERGE|DELETE|SET|REMOVE)\b", re.IGNORECASE)`
  — widened exactly as reported, comment correctly frames it as "the same keyword set as
  `_WRITE_KEYWORD_RE`... minus `CREATE`."
- **Rejection message (lines 452-458)** updated to name all four: *"a bare MERGE, DELETE, SET, or
  REMOVE elsewhere in the same statement."*
- **Re-traced Attack C by hand against the widened regex, not trusted from the plan's own claim.**
  `CREATE (junk:KaizenEntry {..., author:'analyst', ...}) MATCH (victim:KaizenEntry {...}) SET
  victim.author = 'nobody', victim.fact = 'tampered'`, `agent='analyst'`: `_author_claims()` still
  finds the one claim (`'analyst'`, matching), `mismatched=[]`; `_has_foreign_trigger_outside_strings`
  now scans for `MERGE|DELETE|SET|REMOVE` and finds the bare `SET` token (not inside any of
  `_string_literal_spans()`'s ranges — it's a real keyword, not quoted text) → returns `True` →
  **rejected**. The `REMOVE`-variant (`REMOVE victim.author`) traces identically — bare `REMOVE`,
  outside any string span, detected, rejected. Both now closed.
- **Items 22/23, read directly in §5 (lines 670-679):** item 22 is explicitly the `SET`-chained
  variant and explicitly calls out asserting the `victim.author = '<other-agent>'` sub-case "not just
  `victim.fact = 'tampered'`," matching exactly what Pass 2 asked for — this is not left to be merely
  implied by a more general assertion. Item 23 is the symmetric `REMOVE`-chained variant. The
  mutation directive (lines 684-694) correctly requires the `SET|REMOVE`-narrowing mutation be caught
  specifically by 22/23, not by 18/19 (which only exercise `MERGE`/`DELETE`) — this is the right
  granularity: a mutation that only removes `SET`/`REMOVE` from the regex would leave 18/19 passing
  (they never contained those keywords), so requiring 22/23 specifically to catch it is the only way
  this mutation check is actually meaningful, and the plan gets this right.

### Adversarial pass — no further gap found, one non-blocking nit

- **Non-regression re-checked against both accepted shape-1 templates, myself, not just the plan's
  own confirmation.** Re-scanned `_MIGRATION_CYPHER`'s literal fixture text (`UNWIND [...] AS e
  CREATE (k:KaizenEntry {entryId: e.entryId, date: e.date, fact: e.fact, evidence: e.evidence,
  context: e.context, suggestedHome: e.suggestedHome, author: 'graph-dba', createdAt:
  e.createdAt})`) and the plain single-clause author-write template's property-key vocabulary
  (`entryId, date, fact, evidence, context, suggestedHome, author, createdAt, sessionId`) against all
  four widened keywords: none contains `SET`, `REMOVE`, `MERGE`, or `DELETE` as a `\b`-bounded bare
  token (confirmed `suggestedHome` — the one property key that visually rhymes with `SET` — has no
  internal token boundary for `\bSET\b` to land on, same reasoning already applied to `MERGE`/
  `DELETE` in Pass 1/2, now correctly extended). No false-positive-rejection risk introduced by the
  widening.
- **Re-checked the "scope of the fix" argument (shapes 2-6 already fully anchored) once more under
  the wider keyword set specifically** — widening `_FOREIGN_TRIGGER_RE` changes nothing about *which*
  shapes are reachable via chaining (that argument was about anchoring, not about which keywords are
  screened), so it still holds; re-confirmed no new combination opens up by re-running the same three
  constructions from Pass 2 (decoy-before-producer-write, decoy-after-producer-write, two independent
  self-attributing `CREATE`s) mentally against the wider trigger set — none change outcome.
- **No further chaining gap found.** Considered and ruled out: `FOREACH (...| DELETE ...)`-wrapped
  writes (the bare `DELETE`/`MERGE`/`SET`/`REMOVE` inside a `FOREACH` body is still outside any string
  literal, so it's still caught — the check operates on bare tokens anywhere in the text, not on
  clause structure, so wrapping a trigger keyword in `FOREACH` doesn't hide it); a `CALL {}` subquery
  primitive (not part of `_WRITE_KEYWORD_RE`'s own enumeration either, so out of this tool's modeled
  write-keyword surface entirely — consistent with, not a gap relative to, the codebase's existing
  scope). Two independent review passes plus this one have now looked specifically for a chaining
  gap in this mechanism and found the full set the codebase itself already enumerates
  (`_WRITE_KEYWORD_RE`) is what's covered.
- **One non-blocking nit, worth a cheap follow-up, not a gate:** item 21 (Pass-1-era) only pins the
  string-literal-exclusion behavior for quoted `"MERGE"`/`"DELETE"` decoys, not `"SET"`/`"REMOVE"`.
  The underlying mechanism (`_string_literal_spans()`) is keyword-agnostic — it excludes a match
  by *position*, not by which of the four words matched — so this is expected to generalize
  correctly without further design work, and I found no reason to doubt it; still, a symmetrical
  assertion (a free-text field quoting "SET" or "REMOVE" in an otherwise-legitimate author-write
  still authorizes) would make the test suite's coverage match its own stated justification exactly,
  rather than leaving two of the four keywords one degree less directly pinned than the other two.
  Cheap enough for `tdd-engineer` to fold into item 21 itself during S1, not worth another plan
  revision cycle for.

### Conclusion

Three passes on `authorize_write()`'s shape-1 closure (Pass 1's initial trace, Pass 2's under-scoping
finding, this pass's re-verification) now converge: the mechanism correctly closes Attacks A, B, and
C against the widened `MERGE|DELETE|SET|REMOVE` set, does not regress either accepted shape-1
template, and no further chaining gap was found against the full write-keyword surface this codebase
itself already models. Findings 2 and 3 (from Pass 1) remain correctly resolved, unaffected by this
revision. The plan is sound, complete against FR-1…FR-8/AC-1…AC-7, and its highest-risk area has now
had the adversarial scrutiny its own risk section calls for. **No further findings. Approved —
ready for S0/S1 dispatch.**
