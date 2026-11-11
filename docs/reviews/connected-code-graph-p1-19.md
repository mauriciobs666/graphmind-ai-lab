# Review — P1-19 dispatch 1 (store loader, swap, DDL)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope & verdict

Uncommitted `cpg/ccg/store/loader.py`, `swap.py`, `ddl_v1.cypher` (+2 indexes, +1 constraint), `tests/loader_helpers.py`, `test_loader_unit.py`, `test_loader_swap_unit.py`, `test_loader_live.py`, the `test_ddl_live.py` re-base and one `cpg/README.md` row, against `docs/plans/connected-code-graph-p1-linking.md` P1-19 (dispatch 1: Behaviours 1, 3, 5, 6, swap/lock), `-p1-graph.md` §4, §7, §8 P1-19, G5, G6, D8, and `-p0-graph.md` §6. ACL, fuzz and measurements (dispatch 2) are out of scope.

**Verdict: approve with suggestions.** No blocker. The protocol is correct against §7 and every behaviour I mutated is caught, except the four items under Findings (one needs a plan ruling before P1-19 closes).

CPG: considered, not relevant — a new, uncommitted module with no caller yet; judged by reading the code and by running mutants and a live throwaway.

Run (verified): `test_loader_unit.py` + `test_loader_swap_unit.py` + `test_ddl_live.py` 127 passed; `-m live tests/test_loader_live.py` 16 passed on a throwaway (`ccg-falkordb-tw-16390`, scratch `CCG_CONFIG_DIR`; removed by name afterwards, none of mine remain; `falkordb-dev`/`ccg-falkordb`/`serene_gagarin` never contacted). 43 argument mutants run on a scratch copy `p119a_gate_copy` (unit + live): 40 killed, 1 killed only after a 300 s spin (F5), 2 survived (F2, F3). Mutants of `ddl_v1.cypher` (drop `ConfigEntry.path` index, drop its constraint, typo the property, move the index after the constraints) are all killed by `test_ddl_live.py`.

## Findings

### F1 (major, plan gap) — G5's "the loader refuses a `Unit` row without `flowAvailable`" is unmet and the deviation is recorded nowhere but the brief
`loader.py` has no property check beyond the id (`_preflight`); no test feeds a `Unit` row without the flag; the loader's docstring and the README row do not say so. Verified: `tests/test_assemble_rows.py:297-304` forbids the string `"flowAvailable"` in any `ccg/`/`clients/` module but the assembler's two.
- A **type-generic** check does not remove the need to name the property: "carries every property the schema lists" needs a per-label required-property table, and that table holds `flowAvailable`. So it cannot live as string constants in `loader.py` without tripping the guard, and loosening the guard to fit is wrong. It also would not be independent if imported from the assembler (the loader keeps its own closed tables on purpose).
- What actually enforces G5 today: `-p1-graph.md` §8 P1-18 (b), tested at `test_assemble_rows.py:835` (`isinstance(..., bool)` on every `Unit` row).
- **Suggested ruling:** amend G5 (graph-dba, plan text only) to "the assembler guarantees it (P1-18 (b)); readers keep the fail-closed backstop", dropping the loader sentence. If a loader-side backstop is wanted, put the required-property table in a data file (not a `.py` string) read by both, and add a planted-`Unit`-row-without-flag test that must fail the load before any store call. Do not close P1-19 with G5 as written.

### F2 (major) — the "unknown label" refusal test cannot fail
Mutant `if unknown_nodes:` → `if False:` survives all 132 tests. `test_an_unknown_node_label_fails_before_any_call` (`test_loader_unit.py:211`) plants `SchemaEntry: [{"schemaId": "x"}]`, which is then refused by the *id-property* check (`SchemaEntry row 0 lacks unitId`) and the regex `SchemaEntry` matches that message too. Verified: with the check removed, an unknown-label row that has `unitId` and `path` loads and is **silently dropped** (never sent, absent from the report), the exact failure the check exists to stop. **Fix:** plant `{"unitId": "u", "path": "p"}` and match the "no load statement" text.

### F3 (major) — `LiveStore.query` dropping `timeout` goes unnoticed
Mutant `graph.query(cypher, params, timeout) if timeout else ...` → `graph.query(cypher, params)` survives. The unit tests assert the timeout reaches `Store.query` on a fake; the live test with `TIMEOUT_MAX` 3000 loads a few rows and cannot hit the cap. The one real store therefore has no test that Behaviour 1's per-query `timeout` is forwarded. **Fix:** a unit test over `LiveStore` with a recording `db.select_graph(key).query` (the `_RecDb` pattern in `test_ddl_live.py`) asserting the third argument equals the cap, and `None` when the cap is 0.

### F4 (major) — a gate or CI cannot see a silent skip
Verified: with no throwaway configured, `pytest -m live tests/test_loader_live.py` reports `16 skipped` and **exits 0**; `throwaway_port()` also returns `None` whenever the `docker` CLI cannot start (the `ulimit -v 1500000` case; it passed here with 1.5 GB because docker started). The skip text is good but only `-rs`/`-ra` shows it. `pyproject` has `addopts = -ra -m "not live"`, so the reason prints, yet nothing turns it red. **Fix:** add an opt-in `CCG_REQUIRE_LIVE=1` that makes the `instance` fixture `pytest.fail` instead of `skip` (and the gate command in the plan's P1-19 Gate line sets it), or have the gate assert `0 skipped` in the summary. The mutation list below shows the live tests carry real weight (6 mutants are killed only by them), so they must not vanish silently.

### F5 (minor) — a constraint-FAILED test passes the mutant only after a 300 s real-time spin
Mutant `if failed:` → `if False:` is not reddened by assertion but by the loop running to the real `CONSTRAINT_POLL_TIMEOUT_S` (300 s; unit test `test_loader_unit.py:177` passes no `poll_timeout`). Pass `poll_timeout=1, poll_interval=0` (as the neighbouring test does) so the mutant fails fast.

### F6 (minor) — node preflight checks the id key, not its value
`_preflight` tests `prop not in row` only (read, not run): `{"unitId": None}` or `""` passes and loads a node no edge can match. Add `row[prop]` non-empty string (messages already avoid values).

### F7 (minor, for dispatch 2) — lock-ttl inputs
`swap` defaults `timeout=None`, so the stamp reads under the lock are not capped by `TIMEOUT_MAX` (10,000 ms) while the lock ttl is 2,000 ms; and the post-`RENAME` step checks the stamp only, not that the lock is still held (the P2-3 cover, a no-op until P4). Both are consistent with `-p0-graph.md` §6 rows 8-9 and the spike (`spikes/store/lock_replay.py` had no pre-`RENAME` held-check at all); the dispatch 2 ttl re-derivation should include the capped stamp reads and say that the lost-lock window after the check is covered only by the remainder replay.

### F8 (nit) — ACL lines in the live tests are a hand transcription
I compared `_acl_lines` (`test_loader_live.py:63-73`) with `-p1-graph.md` §7 for `build` and `verdict`: identical (`%W~live ~live__*`, lock selector `+set +get +del +eval`, `+del +expire` only on `__build_*`/`__ov_*`, `+graph.config|get`, `+rename`, `+exists`; verdict `+graph.query +info ~live` plus the lock selector). They are not parsed from §7, so they can drift; dispatch 2's `acl.create_users` replaces them, so no action beyond making that swap.

## Behaviour-by-behaviour (mutants run; "unit"/"live" = who reddens it)

| Behaviour | Mutants (all on `p119a_gate_copy`) | Result |
|---|---|---|
| Lock release compare | drop compare; invert it; plain `DEL` instead of `EVAL` | killed: unit (script text pin + fake semantics) **and** live `…never_deletes_a_lock_another_holder_took` |
| `SET NX PX` | drop `px`; drop `nx`; no release | killed (unit; live for `nx`) |
| RENAME / post-RENAME credential | `rename` as verdict; stamp as build; replay-after as build | killed by unit and, for the first two, by the real grants (live NOPERM) |
| Lost-lock check; pre/post stamp checks; `renamed` flag | removed / half-compared / flipped | killed (unit; each stamp component separately) |
| Batch ceiling | `BATCH_MAX` 100k; batches of `size+1`; size gate removed | killed (unit + live 45,000-row group) |
| Count check | removed; only `<`; only `>`; `None` not fail-closed | killed (unit; live for both directions) |
| `MOUNTED_BY` CREATE | `MERGE` | killed (27 tests, incl. live G6 pair `ord 0/1`) |
| Key ownership | always owned; no round-trip; loose prefix; each `LiveStore` method unguarded | killed (the 6 cases `<s>` live, `<s>2`, `<s>2__build_`, `<s>__x`, `<s>___build_`, bare `shop`) |
| Cleanup / ordering | cleanup removed; Snapshot not last; indexes skipped | killed |
| Timeout | `timeout=None`; not forwarded by `LiveStore` | killed / **survived (F3)** |
| Preflight | system, model, edge group, id property; unknown label | killed / **survived (F2)** |
| Zero retry | `retry=None` | killed (live) |

Other checks: EVAL is permitted to both users by §7's lock selector (`+eval` on `~ccg_<s>:verdict_lock` only) and the live swap exercises it as `ccg_build_*` (released True); the verdict user's EVAL is not exercised in dispatch 1 (P4). Failure cleanup and the mid-load SIGKILL test use plain `DEL`; `grep` shows no `GRAPH.DELETE`/`graph.delete` in `loader.py` or `swap.py`, and the build credential's `+del` works live while `GRAPH.DELETE` is in nobody's grant. falkordb-py: `relationships_created` is cast with `int()` and compared (a `1.0` would equal `1` in Python either way, so the cast is cosmetic but harmless); the lock check handles `bytes` and `str`; `FalkorDB(... retry=zero_retry())` is asserted live.

## What's solid

- The sequence (indexes on the empty key, nodes, edges, constraints polled, `:Snapshot` last), the closed 34-group list (plus the excluded `PROVIDES`/`CONSUMES`/`READS`/`WRITES` refusals before any store call) and the ownership check, which round-trips through `sysname.build_key` rather than a second regex.
- The swap docstring, code and tests agree on who does what with which credential; `SwapError.renamed` is correct in both phases; release never masks the propagating error.
- Live tests run under the real grants with a negative control (`build_db` refused on live) and cover kill-leaves-live-intact, held lock, foreign neighbours, first build, lock ttl lapse.
- `test_ddl_live.py` now parses both plan tables instead of transcribing them; all four DDL mutants are killed.

## Open questions

- F1: does graph-dba amend G5, or add a data-file backstop? (Plan owner's call; I recommend the amendment.)

### Proposed REVIEW.md lines

None. (F2/F3 are the standing "check that can fail" rule applied; F4 is a per-test fix.)
