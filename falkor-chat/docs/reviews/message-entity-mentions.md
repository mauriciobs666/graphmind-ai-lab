# Review — message entity mentions (K-093, option B plan gate)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-093

## Scope & verdict

**Reviewed:** `falkor-chat/docs/plans/message-entity-mentions.md` (untracked, first revision), §1, §3–§6
and §7 B. I read it against K-093's done-condition, DESIGN §1.2/§5.1/§5.2/§8, QUERIES §4/§6/§10/§14.3,
SERVER.md, `Repository.hybrid_search`/`search_chunks`, `Services.hybrid_search`, `test_graphrag.py`,
`test_services.py`, `scripts/test_queries.sh` §6, and `hybrid-search-bounded.md` F1. Option A was
read only where it bears on the reversal trigger. Growth-stops check: not applicable, because this is
the plan's first revision.

**Verdict: needs changes.** B is the right call and the premises hold. But the unit as written has a
contradictory test step (M1) and no check that the removal happened (M2). Its close list also misses
three spec sections (M3). Each fix is a small plan edit.

CPG: considered, not relevant — `cpg_falkorchat` is stale (16 server commits), so I used grep and live EXPLAIN instead.

## Findings

### M1 (major) — deleting `_mention` breaks the test the plan says to keep green

`test_hybrid_search_finds_the_seed_thread_by_index_not_by_walk` (`test_graphrag.py:308`) builds its
fixture with `_seed_hub_thread` (`:317`), and that helper calls `_mention` (`:201`). §4 deletes `_mention`
while §7 keeps this test green as the F1 guard, so the guard fails with a `NameError`. Deleting the five
tests also orphans `_seed_hub_thread`, `_plan_ops` and `_records_produced` (`:188`, `:356`, `:360`; I
grepped their callers). *Verified by reading and grep.*
**Fix:** in §4, rebase the index test on `_seed_thread_partly_embedded` or `_seed_thread`, and delete
the three orphaned helpers.

### M2 (major) — no gate can fail if the expansion stays

The §7 gates are `pytest`, `test_queries.sh` and the impl review. Every edit to a test in §4 is a
deletion or a fixture-key removal. A diff that deletes the tests but leaves the `CALL {}` subquery and
the column in place passes both suites. A `tdd-engineer` unit needs a red-first pin.
**Fix:** name three pins in §7:
- repository rows have exactly the keys `{msgId, text, role, score}`, in both variants (this replaces `…related_context_is_empty_list`);
- `"relatedContext" not in` the merged Chunk row (this replaces `test_services.py:2361`);
- `"MENTIONS" not in` the sent text, checked through the existing `_sent_hybrid_query`.

The mutant is to restore the subquery or the `[]` stamp, and each pin turns red under it.

### M3 (major) — the close list misses spec sections that describe the expansion

The close list is built from a `relatedContext` grep, so it misses the prose synonyms. My unfiltered
grep for `MENTIONS[^_]`, `co-occurrence` and `expansion` also finds these:
- DESIGN §5.2 `:326`: "reaches … entity co-occurrence — hybrid retrieval in one round trip";
- QUERIES §4 `:364` and §10 `:830`: both say `MENTIONS`→`Entity` is "GraphRAG co-occurrence (§6)";
- QUERIES §14.3 `:2018`: "no Entity expansion", which is harmless but now a contrast with nothing;
- code docstrings: `repository.py:1370` (`search_chunks`) and `services.py:1203-1205`.

Without these edits the spec still describes a dropped read, which is a blocker at the close check (root `REVIEW.md`). SERVER.md is clean:
its only `hybrid_search` text (`:940`, `:1028`) describes the `graphrag_retrieve` projection and the
responder, and neither carries the field. *Verified by grep.*
**Fix:** add these sections and docstrings to §4 and to the §7 close list.

### M4 (major) — reversal trigger: wrong home and not falsifiable

**Home.** `falkor-chat/AGENTS.md` names DESIGN §1.2 as the single authoritative decision register. It already holds a rejected-option row, "No DayBucket *(rejected alternative)*", as precedent. B also deletes §5.1's dormant-schema block, so a trigger placed there sits in the schema notation with nothing around it.

**Falsifiability.** "An eval shows entity co-occurrence beats plain ANN" names no eval, no metric and no margin. It also cannot be measured without first building A's writer, so it never fires on its own.

**Fix:** add a §1.2 row, "No `Message→Entity` co-occurrence expansion *(rejected alternative)*", detailed in §8. Make the trigger observable: "a consumer (responder prompt or a tool schema) needs per-seed related context, **or** a prototype canonical-entity expansion improves `tests/eval/test_retrieval_eval.py` recall@5 or MRR on `ws:eval` by ≥ X". `data-scientist` sets X.

### m1 (minor) — the `test_queries.sh` neighbour assertion can never fail

`assert_contains "§6 Entity graph is empty …" "0"` (`test_queries.sh:430-431`) matches redis-cli's
`Cached execution: 0` footer. A read-only probe on `ws:acme` returned `544` entities and the footer
still contained `0`. Its comments (`:379`, `:419`, `:427`) restate the same false "no Entity
pipeline in M2" premise. **Fix:** delete the assertion and those comments together with the expansion assertion, and list them in §4.

### m2 (minor) — the target Cypher is not stated

§4 says what to delete but never gives the resulting text. Put the two target variants in §7, with
`ORDER BY score ASC, msgId ASC LIMIT $limit` kept verbatim, so that the QUERIES §6 block and the three
`test_queries.sh` copies are compared against one source.

## What's solid

- **Premise 1 holds.** My unfiltered grep found 59 files (the plan says 62; the count is not load-bearing). The only live readers are the files in the §4 table. The remaining hits are in archived plans, review evidence or kaizen. No reader passes the field through generically:
  - responder `_build_prompt` reads only `s['text']`;
  - `GraphragRetrieveTool.run` projects four keys;
  - both eval harnesses read `msgId`/`text` only;
  - `test_background.py` `FakeServices` raises before it builds any rows;
  - `mcp.py`/`api.py` expose no hybrid search;
  - `kiro/`, `salesperson/` and `web/` have zero hits.
- **Dropping the edge type leaves nothing dangling.** It has no DDL, no writer and no fusion code; the only non-test `MENTIONS` code hits are cypher-mcp's kaizen ontology. Live `MENTIONS` count is `0` in all seven `ws:*` graphs (I re-ran this).
- **Planner hazard: covered.** I EXPLAINed the subquery-free text on `ws:acme`. The scoped variant plans as `Node By Index Scan | (t:Thread)` → `Conditional Traverse | (t)->(c:Channel)`, and the unscoped one as `Node By Index Scan | (t:Thread)`. With the `WITH` dropped (the mutant), the plan anchors on `(c:Channel)`, so the kept index test can fail. The 13 `hybrid` tests in `test_graphrag.py` pass at HEAD.
- The recommendation is proportionate, and §5 rightly rejects the hard-coded-`[]` variant.

## Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*. Source: M1 and M3 of this review.

- (addition) When a plan deletes a test helper, check every caller of that helper, including calls made through other helpers, and check it hardest for the tests the plan keeps as guards.
- (addition) When a plan drops a named element, build its close list from an unfiltered grep of the element's name **and** its prose synonyms across the live spec and docstrings. A grep for the field name alone is a bound.

## Open questions

- None that block. M4's X is a `data-scientist` call, which the plan can defer by naming the owner.

## Pass 2

**Reviewed:** plan v2, revised in place, 1,342 words. **Verdict: approve with suggestions.**

All checks below ran on a scratch copy of `server/` (package, tests, `pyproject.toml`; `scripts/` and `config/` symlinked) using the real venv. The real tree was not touched. I applied §3.1's Cypher and §3.3's test edits to the copy, ran the three pins and the rebased F1 guard at HEAD, at v2, and under each named mutant, then ran the four affected modules at v2.

**Dispositions**
- **M1 — fixed.** With the single-message `_seed_thread` fixture, the rebased index test passes on §3.1's text. Under the no-`WITH` mutant, `[c1]` goes red with `Conditional Traverse | (c)->(t:Thread)`. `_seed_thread_partly_embedded` is in fact orphaned: its only callers are `_seed_hub_thread` and `…lists_a_message_sharing_two_entities_once`, both deleted. Pass 1's orphan list missed it.
- **M2 — fixed.** At HEAD, pins 1 and 2 are red in both variants, and pin 3 is red. At v2 they are green. The restored `CALL {}` mutant (HEAD repository) turns pins 1 and 2 red. The restored Chunk-row `[]` mutant turns only pin 3 red. `_sent_hybrid_query` captures the text actually sent: on the copy, `Channel` and `channelId` appear only in the `c1` variant. At v2, `test_graphrag`, `test_services`, `test_tools` and `test_responder` give 500 passed.
- **M3 — fixed.** §4 covers DESIGN §5.2, QUERIES §4/§10/§14.3, and both docstrings. My re-grep of the context files, README and SERVER.md found nothing more. The two exclusions are correct: QUERIES §14.5's "harmless extra co-occurrence edge" and `link_chunk_about_entity`'s docstring both describe `ABOUT`, not the dropped read.
- **M4 — fixed, and Pass 1's recommendation was the weaker one.** Placing the trigger in the §1.2 row is right. The rejection of the eval-margin trigger is sound: that trigger can only be measured after paying for A's writer, as Pass 1 itself noted. A requirement or backlog item that names a consumer is observable and can be grepped, so it is falsifiable.
- **m1 — fixed.** The probe, the assertion that can never fail, and the three comments are all on the deletion list.
- **m2 — fixed.** §3.1 states both variants. `ORDER BY score ASC, msgId ASC` is verbatim. The QUERIES §6 header comment ("workspace-wide drops the `WITH` + `MATCH (c…)` pair") still matches.

**Cross-fix interaction.** The pins and the rebased test use only helpers that are kept (`_seed_thread`, `_sent_hybrid_query`). The §3.1 text matches what DESIGN §8 and the QUERIES §6 bullets will say once they are edited. The `test_queries.sh` edits carry the column out of exactly the two copies that have it (`:422`, `:440`), not the PROFILE copy (`:435`).

**New findings**
- **N1 (minor, growth-stops).** The 1-word drop comes from rework: Option A's cost, value and unit sections were replaced by the B unit. No new scope was added. But §2's paragraph "I rejected the eval-margin form the review proposed…" is a finding disposition, which the growth-stops bullet bars from a plan. **Fix:** state the reason as design ("an eval margin cannot be measured before A's writer exists, so the trigger is a consumer, not a margin") and drop the reference to the review.
- **N2 (nit).** Three stale comments sit outside §3.3's deletion list:
  - `test_graphrag.py:123`, the "Entity layer dormant (M2)" section header, which is orphaned once its only test goes;
  - `test_graphrag.py:416`, "no Entity expansion (ABOUT is dormant…)", which is the same contrast the plan already drops from the `search_chunks` docstring;
  - `test_services.py:2360`, the "dormant relatedContext:[] convention" comment above pin 3.

  **Fix:** add them to §3.3.
- **N3 (nit).** The new §1.2 row says "Detailed in: §8". After the close, §8 does not mention the expansion. The DayBucket row has the same pattern (it points to §5.2, which never names DayBucket), so this is tolerable. A pointer to "—" or "QUERIES §6" would be no better. Leave it, or let the §8 edit keep one clause saying the message read has no entity hop.

**Proposed REVIEW.md lines.** Pass 1's proposals stand. N1 is already covered by root `REVIEW.md`'s growth-stops check, so I propose nothing new.
