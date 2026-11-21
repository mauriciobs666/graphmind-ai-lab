# Connected code graph P1-56 — implementation gate and test-plan re-gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** P1-56 (ccg P1)

## Scope & verdict

Gate on commit `e48b0916` (56a `qa-engineer` + 56b `tdd-engineer`, one commit): `cpg/fixtures/system1/fx-svc/{.mcp.example.json,.claude/agents/remote.md,.claude/agents/orders.md}`, the manifests (`expected/{fx-svc,fx-web,cross}.yaml`), `cpg/tests/{test_fixture_binding_offline,test_fixture_binding_live,test_fixtures,test_manifest}.py`, and the v1.11 re-gate of `docs/test-plans/connected-code-graph-fixtures.md` and `…-p1.md`. Baseline: `docs/plans/connected-code-graph-p1-linking.md` P1-56 (with the P1-53 "Rows" paragraph and P1-49), QA report `docs/test-reports/connected-code-graph-p1-report.md` D1/D2, style of `docs/reviews/connected-code-graph-p1-55-impl.md`. Root `REVIEW.md` applied; `cpg/REVIEW.md` does not exist. Later commits (`201f4773`…`943c7c09`) touch only P1-55 files and ledgers, not this diff.

**Verdict: approve with suggestions.** No blocker. The fixtures match the plan, the 56 pins are independent of the implementation and every planted mutant I ran was killed, and the live build changes exactly two Links. One real weakness (M1): the `resolve_ts` fallback lets a wrong single-segment locator resolve. The rest is stale text and unpinned corners.

CPG: considered, not relevant — the questions are fixture content, test-pin strength and manifest arithmetic, answered by running the suites, planted mutants and the live fixture build; `cpg_*` graphs describe other components.

**Observed runs** (from `cpg/`, `.venv/bin/python`):
- `pytest tests/test_fixtures.py tests/test_manifest.py tests/test_fixture_binding_offline.py -q`: **273 passed, 3 deselected** (the live tier), no failures.
- Live tier, `pytest tests/test_fixture_binding_live.py -m live -q`: **3 passed** in 102 s (Docker sandbox, no store; port 6390 / `falkordb-dev` untouched).
- Link-multiset diff, same live build (my dump test in a scratch copy), fixtures at `eb930b8b` vs `e48b0916`: **84 → 86 Links, 2 added, 0 removed, 0 changed** (consumer, provider, status, reason, matcher, BOUND_BY ords compared). The added two: `svc:.claude/agents/orders.md::<document>` → `svc:mcp/server.py::list_orders` `possible`/`prompt-mention`/`exact`, no BOUND_BY; `svc:.claude/agents/remote.md::<document>` → same provider `possible`/`example-config`/`exact`, BOUND_BY ord 0 and 1. That meets the plan's Done-when ("exactly two Links added").
- Mutants ran in scratch copies under the session scratchpad (`w/`), reset after each; the working tree and git were not touched.
- Not run: the full repo suite; the known reds listed in the brief.

## Findings

### M1 (major) — the `resolve_ts` single-segment fallback resolves a call statement as a declaration

*Verified by mutant and by direct calls.* `tests/test_fixtures.py:600-604` (new): when no declaration matches and the locator has one segment, any line matching `^\s*(async\s+)?<name>\s*(…|\()` counts as a member declaration. The trailing `\(` alternative matches a bare call: `resolve_ts("request(1);\n","request")` → `(1, 1)`; `"function a(){\n  query(1);\n}\n"` → `(2, 2)`; `"if (x) {\n  query (y)\n}\n"` → `(2, 2)`. The three controls the qa added (`queryy`, `{ quer: 1 }`, a comment) are near-names and a comment; none is a call.
Consequence: `test_tp002_every_locator_resolves_at_a_pinned_commit` accepts a manifest locator that names a function the file only calls. Planted: `fx-web:app/src/ui/Health.tsx::Health` → `::useEffect` in every holder and both test plans (the line `  useEffect(() => {` matches). Result: TP-002 **green**; only `test_tp018_every_planter_site_is_explained_by_a_row…` went red (observed), a different check with a different job. With the fallback disabled TP-002 reddens (3 failed).
Suggestion: tighten the member regex so it needs a definition shape, not a call. I ran this one against eleven lines: `…{head}\s*(?::\s*(?:async\s*)?(?:\([^)]*\)\s*(?::[^=]+)?=>|function\b|[A-Za-z_$][\w$]*\s*=>)|\([^)]*\)\s*(?::\s*[\w<>\[\]|. ]+)?\s*\{\s*$)`. It accepts `query: () => …`, `query: async () => …`, `query: function() {}`, `query: foo => …`, `query(x) {`, `async query(x: string): Promise<void> {`, and rejects `query(1);`, `query(() => {`, `query (y)`, `query: (1),`, `query: foo,`. Add the call-statement, callback-opening and `if (` lines as controls beside the existing three in `test_tp002_resolvers_distinguish_overloads_and_nested_members`. The LNK-10 locator (`query`, which the extractor emits for a nested object method, P1-55 D6) still resolves.

### M2 (minor) — TP-004 in the fixtures plan still says "20 cells"

`docs/test-plans/connected-code-graph-fixtures.md:613` (TP-004): "floors, **20 cells**, per-gate shape counts". §6 line 505 and §7 line 551 now say 19 of 20 (`launch` × `possible` excepted). The script (`census_violations`) requires 19. A reader of the row expects 20.
Suggestion: "floors, the 19 kind × status cells §7 requires, per-gate shape counts".

### M3 (minor) — `P1_CELLS_WITHOUT_A_ROW` is narrow, but nothing plants the exception, and it is gate-agnostic

*Verified.* The exception is one tuple and the check is equality (`fig["cells"] != cells`), so it cannot hide any other missing cell (the other 19 are still required; `launch` × `confirmed` has LNK-21/22). Two gaps: (a) `census()` collects cells from every row at every gate (`test_fixtures.py:920`), so the constant's comment ("no row at P1") and §7's wording are narrower than the code: a **P2** `launch`/`possible` row would also turn the census red, which is the opposite of "P2 re-adds the row". Whoever adds that row must edit this constant; say so. (b) Planting `launch` × `possible` (LNK-22 flipped to `possible`) turns it red with the message `kind x status cells missing: []` — red but empty, because the message lists only missing cells. No test plants it.
Suggestion: report `sorted(fig["cells"] ^ cells)` in the message, add the planted `launch`/`possible` row to `test_tp004_planted_failing_inputs`, and word the comment "at any gate; P2 removes the constant with the launch-target export".

### M4 (minor) — the AC-6 example-config floor counts rows by gate; the reason is pinned elsewhere, and EX-1's BOUND_BY ord 1 is unpinned

*Recomputed from the manifests (held rows, merged across holders).* Rows carrying `reason: example-config`, all gates: AC6-EX-1 (P1), EX-2 (P2), EX-3 (P3), EX-4 (P1). At P1: **2** (EX-1 `mcp_tool`, EX-4 `http`), no `p1Form` overrides either. The plan (P1-53c, P1-56) defines the floor as instances "whose P1 form carries the `example-config` reason"; `census_violations` reads `AC6_GROUP_FLOOR` by `gate` and `item` text only. Planted EX-4 `reason: partial-key`: `test_tp004_*` stayed green; red came from F3 (plan-row equality) and the binding pin (4 failed, observed). EX-1 `reason: partial-key`: F3 and the synthetic Link tier red (3 failed). So both reasons are pinned, by checks other than the floor; the floor itself would pass a P1 EX group with no `example-config`. Acceptable; say it in TP-004's row ("the reason is pinned by F3 and the Link tier, not by the floor").
Also observed: the live EX-1 Link carries BOUND_BY ords 0 **and** 1; `p56_link_violations` pins ord 0 only. Ord 1 is not in the manifest row. Either state it in the row's `aux` or accept it as unspecified.

### M5 (nit) — test-code leftovers

- `test_fixture_binding_offline.py:92` cites `p56_file_violations`, which does not exist (the checks are `p56_entry_violations`/`p56_prompt_violations`); `tool` is assigned and unused in `p56_entry_violations`.
- The `p56_root` control docstring says it holds "whatever the committed fixtures hold (the edits overwrite them)", but `p56_edits` does not overwrite `.mcp.json`: planting `fx-svc-remote` into the committed `.mcp.json` also reddens `test_the_p56_pins_accept_the_tree_the_plan_prescribes` (observed, mutant C), so a fixture regression shows as a control failure ("the pin, not the fixtures, is wrong" per the docstring of its sibling). Add `.mcp.json` to `p56_edits` (copy of the base) or reword.

### M6 (nit) — test-plan notes stack instead of replacing

Root `AGENTS.md` rule 5 says a dated revision note "replaces the previous line". `…-p1.md` gained a third top note (line 7, beside lines 5 and 9) and `…-fixtures.md` keeps v1.10, v1.9, v1.7, v1.4 under the v1.11 note. `test-plans/` is not a growth-stops kind, so this is pre-existing and not a growth-stops finding; word counts for the record: fixtures 12698 → 12916 (+218), p1 9307 → 9424 (+117), all scope-driven (EX-1/PM-3/URL-2/LNK-10 rework plus the §7 exception). No restated history in the new text.

## Check-by-check

**(1) Fixtures vs plan — pass.** `.mcp.example.json` has `fx-svc` and `fx-svc-remote` `{"url": "http://localhost:8200/mcp"}`; `.mcp.json` holds `fx-svc` and `fx-svc-exec`, **not** `fx-svc-remote` (read). `remote.md`: one tool mention, `` `mcp__fx-svc-remote__list_orders` ``, no plain `list_orders` (read; pin below). `orders.md`: one list item `- Use list_orders to see what a customer has bought.` Both prompt files classify `prompt-md` (`test_manifest` counters 5 and 6 pass). `prompts/system.py` stays, unreferenced by any manifest row or test-plan (unfiltered grep: 0 hits in `docs/test-plans/*.md`, `expected/*.yaml`, `cpg/tests/*.py`).

**(2) Pin independence and mutants — pass.** `p56_entry_violations`/`p56_prompt_violations`/`p56_link_violations` take server, tool, keyPath, providers, status and reason from the manifest rows (`row_locators`, `merged_row`, `loc_of_row`, `row_locators_true`); no fixture constant is mirrored (the control tree `REMOTE_MD`/`ORDERS_MD` is the plan's text, used only to prove the pins accept it). Mutants on scratch copies of the fixtures/manifests, each run against `test_fixture_binding_offline.py` unless noted:

| Mutant | Killed by |
|---|---|
| example `fx-svc-remote` entry dropped | `test_the_fixtures_carry_the_example_url_entry…` |
| url port 8201 | same |
| `fx-svc-remote` also in `.mcp.json` | same, and (M5) the control |
| plain `list_orders` line appended to `remote.md` | `test_the_fixtures_carry_the_prompt_files…` |
| `remote.md` deleted | same |
| `orders.md` as prose, not a list item | same |
| EX-1 manifest `confirmed` (full `test_fixtures`+binding set) | 8 failed: TP-003, TP-004 ×3, F3 ×2, synthetic Link tier ×2 |
| EX-1 manifest reason `partial-key` | F3, synthetic Link tier ×2 |
| EX-1 manifest gate `P2` | TP-004 ×3, F3 |
| EX-4 manifest reason `partial-key` | F3, binding pin ×3 |
| URL-2 `p1Form` reason `partial-key` | `test_p1_form_equals_the_plan_table…` |

The Link-tier mutants are synthetic (`_fake_world`); the real-build assertion is the live test, which I ran green. One caveat on the synthetic tier: it proves `p56_link_violations` detects a wrong Link, not that the linker produces the right one; the live run does the latter.

**(3) qa edits to `test_fixtures.py`.** Cell exception: M3. Resolver fallback: M1. P1-form parser regex (`AC6-URL` added, `p1_form_plan_rows`): scoped to the "P1 form" table slice, two-sided (`set(plan) - set(seen)` and the converse); URL-2 plan-cell mutant reddened (above); no other AC6-URL row sits in that table. PM row counts 2→3 and `tools=()`/globs mutants 2→3: correct (PM-1, 2, 3 are `.md`); the new `is_file` guard in `pm_md_violations` is exercised by deleting the fixture (`remote.md` deletion reddens it).

**(4) EX-1 floor and Link-reason stability.** Recomputed: M4 (2 at P1, 4 across gates). Reason stability: the Link `(provider, consumer symbol)` for `remote.md::<document>` has one site, so "best site sets the reason" cannot be contested; the live build reads `possible`/`example-config`. The stability rests on `remote.md` staying single-site, which `p56_prompt_violations` pins (mentions list exactly `[mcp__fx-svc-remote__list_orders]`, zero plain mentions; mutants D and the "naming another server" mutant red). The tie rule itself (which reason wins at equal status) is read from the plan's probe, not asserted by any test; that is the unpinned premise, and the single-site pin is what contains it. Judged stable.

**(5) Test-plan docs — every hit ruled** (greps over both whole files, unfiltered):
- `EX-1`: fixtures 5 (v1.11 note), 7 (v1.10 history note, dated), 238 (§5 row), 497, 556, 558 (§6/§7); p1 7, 168, 191, 297, 310 — all current. `PM-3`: fixtures 5, 244; p1 297 — current. `launch`: fixtures 82, 167, 170-172 (LNK-18/21/22/45, unrelated), 247, 505, 551, 558 — current; p1 132, 165, 233, 294 — unrelated (the `launch` kind still has P1 rows LNK-21/22). `FX_PEER_BASE`: fixtures 241, p1 297 — EX-4, unchanged. `partial-key`: 141, 166, 215, 347-350, 555, 568; p1 114, 167, 291-294 — other rows, unchanged. `prompts/system.py`, `SYSTEM_PROMPT`: no hits (removed cleanly).
- Stale: **fixtures 613 "20 cells"** (M2). Nothing else stale about EX-1/PM-3/launch×possible/URL-2/LNK-10: LNK-10's `C` is `…index.ts::query` in §5 (159) and P1T-06; `api.search.query` remains only as the field-path description (fixtures 159, p1 170), correct.
- Counts: P1 AC6 rows 11 (URL-1..3, DYN-1..3, EX-1, EX-4, PM-1..3; p1:191) = manifests; family table 2+11+4+4+2+5+27+2+6+3+18+3 = 87 (p1:199) = manifests (`LNK 27, SK 18, AC6 11, NC 6, KN 5, DF 4, CD 4, TW 3, RV 3, MR 2, AB 2, FT 2`); PM `.md` rows 3; cells 19 of 20 (equal to `census()` observed: 19 cells present).

**(6) Census 87 / 134, two definitions.** Held rows merged across manifests (the census's definition, `M.holders`): **134 rows, 87 at P1** (P2 18, P3 25, P4 1, P5 3). Raw manifest entries (a row is written once per holder file; 40 ids have more than one holder): **175 entries, 111 at P1**. Per file: fx-svc 92, fx-web 50, fx-lib 6, fx-job 27. Subtracting the 12 negative controls (10 at P1): 77 non-negative P1 rows, 122 positives overall (plan: 122). Spread 87 vs 111 is entirely the holder copies; 87/134 is the figure §6, TP-004 and the P1 test plan use, and it reproduces. Also: the plan's "census stays 134 and 87" holds after the PM-3/EX-1 reshape.

## What's solid

- The fixture edits are minimal and match P1-56 to the character; the example server is bound only by the example config (not in `.mcp.json`).
- The 56a pins are manifest-driven, red-first by design, and killed on all eleven mutants above, including the planted-neighbour-mention mutant that protects the EX-1 reason.
- The live fixture build does what the plan said and nothing else: two Links added, none removed or altered.
- `P1_CELLS_WITHOUT_A_ROW` is one tuple under an equality check, not a loosening.
- Counters (`test_manifest` 4→6 md files, 3→5 prompt-md) and SHAs regenerated; suite green.

## Open questions

None that change scope.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, heading **Always check** (addition; no prune: nothing in that file is now enforced elsewhere). Source: M1.

- A resolver or matcher loosened to accept a new shape (a declaration regex, a locator fallback) is checked with one planted wrong input per axis it now accepts (a call site, a comment, a near-name), not only the new shape and its near-names; a loosened matcher whose controls are all near-names is a bound, not proof.
