# Gate: keyless `candidate_count` fix (ccg engine reader)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (P1 engine, `candidate_count`)

**Scope.** Uncommitted diff of `cpg/ccg/engine/reader.py` (`candidate_count` text), `cpg/ccg/engine/bfs.py` (docstring), tests `test_engine_flow.py`, `engine_walk_helpers.py` (MemoryReader twin), `test_engine_flow_live.py`, `test_engine_answer.py`, `test_engine_reader.py`, and `docs/plans/connected-code-graph-p1-graph.md` §5 / acceptance (d). Baseline: HEAD, `docs/plans/connected-code-graph-p0-graph.md` D6, `docs/reviews/connected-code-graph-p1-21.md` Pass 2 G1. Concurrent fixture/plan edits ignored. `REVIEW.md` (root) applied; no `cpg/REVIEW.md` exists.

**Verdict: approve with suggestions** — the crash is fixed and the count is never 0/1-by-default, but the new group is wider than the semantics it claims (F1, major) and the unit tests cannot see the Cypher (F2, minor). Neither blocks; F1 should be decided before the keyless count reaches a user-facing answer.

CPG: considered, not relevant — a one-line Cypher text change in a graph-reading engine; no call-graph/impact question needed answering.

## Findings

### F1 (major, verified live) — grouping by owner over-counts when one owner has several keyless sites
The linker gives a keyless consumer a `VIA` edge to **every provider of its kind** (`ccg/linker/link.py` "Unknown keys" paragraph; `_candidates`, `c["key"] is None`). So the true "may call any of N" for a keyless edge is the fan-out **of that call site**. The fix groups by *owner symbol* + absent key, which merges different keyless sites of the same function (different kinds, or one binding-narrowed and one not).
Probe on a throwaway store (owner `F`; keyless http site `h` -> `H1,H2`; keyless shell-dynamic site `q` -> `Q1`; keyed site `k` -> `K1,H1`):
the shipped text returns `h/H1 = 3`, `q/Q1 = 3` (true: 2 and 1); keyed `k/K1 = 2`, `k/H1 = 2` correct. The answer would say "may call any of 3" for a site that has one candidate.
The tests do not catch it: `via_graph` puts both keyless sites in one kind-agnostic fan-out, so the two views coincide.
Suggested text (run on the same store: `h/H1 = 2`, `q/Q1 = 1`, keyed unchanged, still `>= 1`, still no row for an ownerless site):
`MATCH (s)-[:HAS_CALL]->(c2:CallSite)-[v2:VIA]->(l2:Link) WHERE (v.consumerKey IS NULL AND c2 = c AND v2.consumerKey IS NULL) OR (v.consumerKey IS NOT NULL AND v2.consumerKey = v.consumerKey)`.
Mirror it in `MemoryReader._candidate_count`, the plan §5 text and the `EXPECTED` pin. Add a probe per axis the group varies on: two keyless sites of one owner with disjoint providers; a keyless and a keyed edge to the same provider; shell-dynamic and unknown-key mixed in one owner. If the owner-wide group is the intended reading, say so in D6/§5 and name the over-count as accepted; the current text claims "the providers the key-less consumer may call", which it does not guarantee. Whether to change D6's grouping is the owner's (architect) call; I did not change scope.

### Answers to the brief's other semantic questions
- Same owner, keyed + keyless to one provider: separate groups (keyed uses the key, keyless the `0` arm) — verified (`k/H1 = 2` counts `K1,H1` only). Right.
- `>= 1` guarantee: the pair's own edge always joins its own group (key equals itself, or both absent), so a row has `count >= 1`; the no-row case (ownerless site / no such edge) stays `None`. Verified by the live test and the probe.
- Per consumer symbol only (ignoring key) would be wrong for keyed edges; the D6 text is explicit about equal `consumerKey`.

### F2 (minor, verified) — three of the new tests exercise only the Python twin
Run against HEAD's reader with the working-tree tests: the `test_engine_flow.py` and `test_engine_answer.py` keyless tests still pass (they use `MemoryReader`, which the diff itself changed). Only two things fail on the old text: the `EXPECTED` text pin (`test_every_text_of_the_plan_is_the_one_in_the_code_character_for_character`) and the live test `test_live_candidate_count_groups_...` (live, throwaway). So the default suite proves the twin and the pin, not the Cypher; the Cypher is guarded only by `-m live`. Acceptable if the gate runs `-m live` with `CCG_REQUIRE_LIVE=1`; say so in the unit's close.
Mutation on a copy: `coalesce(v2.consumerKey, 0) = v.consumerKey` (one arm dropped) -> live test red, text pin red (killed). Old reader -> live test red (killed).

### F3 (minor, verified) — `coalesce(..., 0)`
Only strings are written to `consumerKey` (`assemble/rows.py:666-667`). Probed on v4.18.11: `coalesce(null,0)=coalesce('0',0)`, `=coalesce('',0)` and `0='x'` are all false without an error; null/null is true. No collision possible with string keys, including `""` and `"0"`. The read-only lint constraints are unchanged by the edit (`test_every_registered_text_runs_as_ro_query...` is green on the working tree). A `v.consumerKey IS NULL` form (F1's suggestion) avoids the numeric sentinel altogether.

### F4 (minor, inferred) — cost
`EXPLAIN` (empty store): `Node By Label Scan (l:Link)` -> filter -> traverse; unchanged in shape by the edit. `coalesce` makes the `v2` predicate a post-traversal filter, but it was one before (`v2.consumerKey = v.consumerKey` is not indexable either). I did not measure at ~230k nodes; the cost class is that of the existing text. If the graph-dba note's `linkId` anchoring assumes an index, confirm it exists on `Link.linkId` (the scratch graph had none).

### F5 (nit) — docs
`-p1-graph.md` §5 and acceptance (d) are consistent with the code text (verified by the pin). The §5 comment line says "counts the keyless links of its owner"; revise it if F1 is taken. No growth issue: the plan edit is a rework of two lines plus one comment line (rework in place, no history, no table).

## (4) Suite: `test_tp006_...` and the scip group
- **tp006: not environmental and not the candidate-count change; caused by the concurrent fixture re-plant.** Run solo with `-m live` (deselected by default via `addopts -m "not live"`): working tree fails `assert rep["expected_callable"] == want_n` -> `('svc', {... 'expected_callable': 68, 'expected_found': 67})` against `UNIT_FLOORS["svc"] == (67, 66)` (`tests/test_fixtures.py:2164`). A HEAD export (cpg + docs, own root, same docker toolchain): `1 passed`. The re-plant (nested `build_router.*` functions in `fx-svc/api/*.py`) adds one named callable; `UNIT_FLOORS` was not updated (the `test_fixtures.py` diff has no `FLOORS` hunk). Confirmed. Owner: `qa-engineer` (fixtures wave) — set `svc` to `(68, 67)` after checking the extra callable is intended. Not part of this fix.
- **scip group:** `-k scip -m "live or not live"` on the working tree: 136 passed solo. The failures in the implementer's full run are therefore not reproducible in isolation (inferred: the known scip memory/resource flake under a full run); not attributable to this diff, which touches nothing in scip.

## What's solid
The crash is fixed and the new live test is a real positive control (red on HEAD, red under a one-arm mutation, green on the fix). The `None`/BuildError contract for an ownerless site is intact and still tested (`NoCountRows`). Edits are local to the text, twin, docstring, plan §5.

## Open questions
1. F1: owner-wide keyless group (as shipped) or per-site (suggested)? Needs the architect's ruling on D6's wording for the keyless case.

## Throwaway store
`ccg-falkordb-tw-16391` started and removed; scratch credentials copy deleted. The existing `ccg-falkordb-tw-16390` is not mine and untouched.

## Pass 2 (per-site text, `CASE WHEN v.consumerKey IS NULL THEN c2 = c ELSE true END`)

**Verdict: approve.** Executed on a throwaway store (`ccg-falkordb-tw-16391`, removed), the exact text from `QUERIES["candidate_count"]`, my Pass 1 owner `F` probe plus an ownerless site:

| pair | result | expected |
|---|---|---|
| keyless http site -> `H1`, `H2` | 2, 2 | 2 |
| keyless shell site -> `Q1` | 1 | 1 |
| keyed `K1`, keyed + `H1` (same key) | 2, 2 | unchanged |
| ownerless site, pair with no edge | no row | no row |

Keyed and keyless edges to the same provider (`H1`) stay in separate groups (`h/H1 = 2`, `k/H1 = 2`); every returned count is >= 1. HEAD's text on the same store returns rows only for the keyed pairs (the original crash class). Suites with `-m "live or not live"`: flow, answer, reader, flow-live 546 passed.

- **F1: fixed** (above).
- **F2: fixed.** New `test_candidate_count_text_groups_a_keyless_edge_by_its_own_site_and_a_keyed_edge_by_its_key` fails on HEAD's reader (checked), and the live test is red under each mutant run on a copy: CASE arm dropped (owner-wide), `ELSE true` -> `ELSE false`, `c2 = c` -> `c2 <> c`. The text guard and the plan-text pin are red under all three too. The text guard is a substring wiring check, so the live test is the semantic guard; run it with `CCG_REQUIRE_LIVE=1`.
- **F4: not fixed, still open** (no scale measurement; confirm a `Link.linkId` index). `CASE` and `c2 = c` add a post-traversal filter only.
- F3/F5: unchanged; the CASE form removes the numeric sentinel concern only partly (`coalesce(..., 0)` remains, still safe for string keys). No growth finding.
