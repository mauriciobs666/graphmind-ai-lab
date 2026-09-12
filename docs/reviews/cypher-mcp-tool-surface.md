# Cypher MCP tool surface — Plan Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** C-901, C-902 (M9)

## Scope & verdict

Plan-gate review (pre-implementation) of `docs/plans/cypher-mcp-tool-surface.md` (v1,
2026-09-12, `architect`) against `docs/requirements/cypher-mcp-tool-surface.md` (Status: Ready for
design, FR-1…FR-4, AC-1…AC-3). Checked: grounding of every server.py/README.md/requirements-doc
citation against the real files, the `GRAPHS`-directive design and its collision-safety argument,
the FR-3 blast-radius claim, completeness against FR-1…FR-4/AC-1…AC-3, the archived-document
header-pointer edit, the test strategy (including the two regression-pin tests and the live test's
safety), and the `split_directive()` diff's non-regression on existing `query`/`explain`/`profile`
classification. Nothing has been implemented; this review does not touch `cypher-mcp/server.py`,
its tests, or either markdown doc.

**Verdict: approve with suggestions.** No blockers. The design (a standalone `GRAPHS` directive,
detected by requiring the entire trivia-stripped input to be the bare word, `\Z`-anchored) is sound
and its central risk claim — no real Cypher statement can be misclassified — holds up under direct
scrutiny of Cypher's grammar (every real statement opens with a clause keyword; a bare
identifier can never be a complete, self-contained statement). Every server.py line-number citation
I checked was accurate. One finding is worth fixing before or during implementation, not blocking
the plan itself.

**CPG:** considered, not relevant — this is a review of a plan over `cypher-mcp/server.py`, its
test file, and two markdown docs; no Joern CPG is loaded for this repo's own MCP tooling
(confirmed live: `mcp__cypher__query` against a deliberately-wrong graph name lists every loaded
graph today — `cpg_falkorchat`, `cpg_deprecated_salesperson`, `kaizen_team`, various `ws:*`,
`reference`, `test` — no `cpg_cyphermcp`-shaped entry). A call-graph tool would add nothing to a
review of ~150 lines of already-well-documented Python plus prose.

## Findings

### Major — the plan's cited "current" test-gate pass counts are already stale, by a wide margin

`docs/plans/cypher-mcp-tool-surface.md` §2 states the offline/live in-container gate counts as
"74 passed, 7 deselected" / "7 passed, 74 deselected" — verbatim what `cypher-mcp/README.md` (lines
621, 623) states today — and treats this as the accurate pre-delivery baseline that "will shift once
§5's new tests land" (step 3's housekeeping instruction). I ran the actual host-venv suite instead
of trusting the citation:

```
cypher-mcp/.venv/bin/pytest tests -q          → 113 passed, 10 deselected
cypher-mcp/.venv/bin/pytest tests -q -m live  → 10 passed, 113 deselected
```

The real current counts are 113/10 and 10/113 — not 74/7 and 7/74. The gap (39 offline tests) is
almost certainly the M8 kaizen-agent-ontology test additions (producer-write, MENTIONS-write, the
two edge-resolve shapes, the zero-space tolerance tests — all present in `test_server.py` today)
landing after `README.md`'s count was last updated, an existing "the bar" (root `AGENTS.md`)
violation the plan inherited rather than caught. This doesn't change the design and step 3's own
done-condition ("recount and update both numbers... a stale count in this file is exactly the kind
of drift `AGENTS.md`'s 'the bar' convention warns against") is, read literally, self-correcting —
but the plan's §2 "Context & findings" section presents 74/7 and 7/74 as a verified fact about the
current file, not an example, and that fact is wrong by more than 50%. Since §2 is exactly the
section this kind of review exists to gate ("everything below was verified" is the standing bar
other plans in this repo set for themselves — see `cpg-query-access.md`'s §2 preamble), flag it as
a factual defect, not a mere illustration.

**Suggested fix:** before dispatch, correct §2's cited counts to the real current baseline (113/10,
10/113) or drop the specific numbers from §2 and let step 3's recount instruction stand alone as the
only place a count is asserted. Either way, don't let a stale "current fact" ship inside an
otherwise-accurate grounding section.

### Minor — the `GRAPHS` directive has no tolerance for trailing trivia (only leading)

`_GRAPHS_DIRECTIVE_RE = re.compile(r"GRAPHS\b\s*\Z", re.IGNORECASE)` requires only whitespace
between the word and end-of-string; `EXPLAIN`/`PROFILE`'s existing `_scan_leading_trivia()` only
ever strips *leading* comments/whitespace (by design — there is a "rest of the statement" after
those keywords for a trailing comment to attach to). `GRAPHS` has no such rest, so a caller who
appends an explanatory trailing comment — `GRAPHS // just checking` or `GRAPHS /* why */` — falls
through to `"query"` classification and gets FalkorDB's raw syntax error instead of the intended
directive. This is a plausible shape for an agent that habitually annotates its own calls, and nothing in the plan's design, docstring, or `README.md` text warns against it.

**Suggested fix:** either extend `_GRAPHS_DIRECTIVE_RE`'s tail to tolerate a trailing comment
(mirroring the leading-trivia scan, symmetric on both sides), or — cheaper, and probably sufficient
given `GRAPHS` is meant to be sent as an exact, documented literal — add one sentence to
`cypher-mcp/README.md`'s "Graph discovery" section and the tool description: "send exactly
`GRAPHS`, nothing else — a trailing comment is not tolerated and will be sent to FalkorDB as an
ordinary (invalid) query." Either fix is small; pick the first only if a real caller is expected to
decorate the call, which nothing in the requirements doc suggests.

### Nit — §6's citation of `cpg-query-access.md`'s AC-3 equivalence reconciliation as "D5" is a mislabel

Plan §6 (Sort order vs. AC-2) writes: *"consistent with the precedent `cpg-query-access.md`'s AC-3
reconciliation (D5) already established for this exact class of question."* I read the cited
passage (`docs/plans/cpg-query-access.md` §7.2, "AC-3 — equivalence, plus the fresh baseline (D1)",
lines 1109–1134): it does establish exactly the claimed precedent (diff the *value sets*,
order-insensitive unless the query has `ORDER BY`) — but that document's own `D5` label names a
different decision entirely (§4.4 D5, "EXPLAIN-only, PROFILE removed" — the stakeholder decision
table at its §1.1). The substantive claim is accurate and well-grounded; only the parenthetical
label is wrong. Low stakes (nothing downstream keys off it), but worth a one-word fix — cite it as
"§7.2 AC-3" rather than "(D5)" — since a reader chasing the citation into the wrong section would be
briefly misled.

## What's solid

- **The central risk claim holds.** I independently worked through whether any real Cypher
  construct could open with the bare token `GRAPHS` and be misclassified: Cypher statements always
  open with a clause keyword (`MATCH`/`CREATE`/`MERGE`/`WITH`/`UNWIND`/`CALL`/`RETURN`/…), never a
  bare identifier standing alone as a complete statement, and the `\s*\Z` anchor additionally
  requires the *entire* input to be that one word — so even a query that legitimately uses `graphs`
  as an identifier (e.g. `WITH graphs` or `MATCH (graphs:Foo)…`) can never collide, since neither
  starts with the bare token nor ends immediately after it. The `\b` boundary correctly excludes
  `GRAPHS_LIST`/`GRAPHSTATS`-shaped decoys, matching the established `EXPLAIN_ME`/`PROFILER`
  protection class. This is the plan's single highest-risk claim and it is correct.
- **FR-3 blast-radius mechanism verified against the real code.** `run_query()`'s proposed
  `"graphs"` branch (step 1e) is placed *before* `client.select_graph(graph)` is ever called
  (confirmed against the actual control flow, `server.py:949-959`), so `graph` is genuinely never
  read on this path, and it calls `get_client().list_graphs()` directly — the same call
  `graph_not_found_message()`'s two existing sites already make (`server.py:967`, `1015`) — bypassing
  `_list_graphs()`'s swallow-on-failure wrapper (`server.py:907-911`, used only at the outer
  `except`, line 1021). A `list_graphs()` failure therefore propagates as an uncaught exception into
  the outer `except Exception` handler exactly as the plan claims, and `_is_missing_graph()` correctly
  returns `False` for a `RedisConnectionError` so `explain_error()` renders the curated "unreachable"
  message rather than a silent empty list — traced end-to-end, not just asserted.
- **The `split_directive()` diff is genuinely additive.** The new `_GRAPHS_DIRECTIVE_RE` check can
  only match an input that is *exactly* the word `GRAPHS` (plus trailing whitespace); since
  `_DIRECTIVE_RE` only ever matches `EXPLAIN`/`PROFILE`, the two checks are mutually exclusive by
  construction — no existing `query`/`explain`/`profile` classification changes for any of the 15
  parametrized cases already in `test_split_directive_classification`.
- **Every server.py/README.md line-number and character-count citation I spot-checked was
  accurate** — `SERVER_INSTRUCTIONS` is exactly 1254 chars today (confirmed by import), the quoted
  anchor phrases in `TOOL_DESCRIPTION`/`SERVER_INSTRUCTIONS`/the module docstring exist verbatim
  where cited, `format_plan()`'s location (~line 792) is correct, and the `docs/requirements/
  cpg-query-access.md` "Old header note" quoted in §3.2 is a byte-for-byte match of the real file
  (lines 6–9) — this is the one place a plan touches an *archived* document, and the proposed edit
  is genuinely header-metadata-only, correctly following the `generic-cypher-mcp.md` §5 precedent
  for the same situation (a plain `**Note:**` line, not the `Supersedes:`/`Superseded by:` pair,
  which the doc-lifecycle convention reserves for same-slug ordinal succession — this is a
  cross-slug pointer).
- **Completeness against the requirements doc is clean.** FR-1/FR-2/FR-3/FR-4 and AC-1/AC-2/AC-3
  are each mapped to a concrete plan section, nothing in the requirements doc's Out-of-scope list
  (schema discovery, blast-radius change, build-vs-buy redo, `delete_graph`, the write path) is
  touched, and the two frozen-shape regression pins
  (`test_input_schema_has_two_required_params_and_one_optional_agent`,
  `test_exactly_one_tool_named_query`) are correctly identified as needing **no** edit, since the
  mechanism is a directive, not a parameter — verified: no existing test or fixture in
  `cypher-mcp/tests/test_server.py` uses the names `GRAPHS`/`format_graph_list`/
  `_GRAPHS_DIRECTIVE_RE`/`list_graphs_error` today, so the plan introduces no naming collision.
- **The live test (§5.2) is safe.** It only reads (`client.list_graphs()`, and `run_query(...,
  "GRAPHS")`, itself read-only) — no graph is created or mutated by this specific test; it borrows
  the existing module-scoped `live_graph` fixture purely to guarantee the instance isn't empty at
  test time, which is exactly what the plan says it does.

## Open questions

None that block dispatch. The Major finding (stale test-gate counts) is a correction to make before
or during implementation, not a design question for the stakeholder.

## Pass 2 — 2026-09-12, diff-scoped re-gate

Scope: the real, uncommitted diff (`git status`/`git diff`) across `cypher-mcp/server.py`,
`cypher-mcp/tests/test_server.py`, `cypher-mcp/README.md`, `docs/requirements/
cpg-query-access.md`, `docs/HISTORY.md` — not the plan. Everything below was independently
re-derived (diff read in full, suite re-run from a cold shell, arithmetic cross-checked), not
transcribed from the implementer's report.

**Verdict: approve.** No blockers, no majors, no new minors. The diff is a clean, unadorned
implementation of the plan; both Pass-1 findings are correctly closed.

### Pass-1 findings, disposition

- **Major (stale test-gate counts) — fixed, independently reconfirmed.** I re-ran the host suite
  from scratch: `125 passed, 11 deselected` offline, `11 passed, 125 deselected` live — matching
  the implementer's reported figures exactly (not merely trusting them). `cypher-mcp/README.md`'s
  diff now states `116 passed, 11 deselected` / `11 passed, 116 deselected` for the **container**
  gate, a different (smaller) number by design: `tests/test_build_inputs.py` is host-only and not
  collected in the image. I verified this arithmetically — that file collects exactly 9 test items
  (`pytest --collect-only`: 7 `def`s, one parametrized into 3 cases = 9), and 125 − 9 = 116 exactly
  — so the two counts are consistent, not contradictory. I did not run the actual Docker build
  (out of scope for this pass), but the arithmetic leaves no room for the container figure to be a
  guess. `grep`ed both `cypher-mcp/README.md` and `docs/HISTORY.md`'s diff for any leftover
  `74 passed`/`7 deselected` — none remain in either file (the only surviving occurrence anywhere is
  inside `docs/plans/cypher-mcp-tool-surface.md` §2 itself, a plan that has now been executed
  against and is therefore not revised in place per the doc-lifecycle convention — harmless, since
  neither `README.md` nor `HISTORY.md`, the artifacts a reader actually consults for current state,
  carry the stale figure anymore).
- **Minor (no trailing-trivia tolerance after `GRAPHS`) — disposed acceptably, not "fixed" by
  changing behavior.** The implementer chose the doc-caveat option I offered as an alternative to a
  regex change: a one-sentence README addition ("Send exactly `GRAPHS` and nothing else — a
  trailing comment... is **not** tolerated... and is sent to FalkorDB as an ordinary, invalid
  query"), plus a matching inline comment on `_GRAPHS_DIRECTIVE_RE` in `server.py`. This is an
  acceptable disposition of the finding, not a lesser one: the underlying behavior (fail-safe —
  FalkorDB's own syntax error, never silent misclassification) is exactly what made this a Minor
  rather than a Major in Pass 1, and the stated reasoning (`GRAPHS` is meant as an exact literal,
  unlike `EXPLAIN`/`PROFILE` which prefix continued text, so there's no natural "rest of the
  statement" for a trailing comment to belong to) is sound and consistent with the plan's own §3.1
  framing of `GRAPHS` as categorically different from the other two directives. Judged on the
  question asked (is this an acceptable disposition of *my* finding), not on whether I'd have picked
  the same fix: yes.
- **Nit ("D5" mislabel in the plan's §6) — not applicable to this diff.** That citation lives in
  `docs/plans/cypher-mcp-tool-surface.md`, which is not one of the five changed files; nothing to
  recheck here.

### New checks this pass

- **Diff matches the plan's §4 design exactly.** Read the full `git diff` for `server.py`: the
  module-docstring bullet, the `_GRAPHS_DIRECTIVE_RE` regex (`GRAPHS\b\s*\Z`, anchored, plus the new
  trailing-trivia caveat comment), the `split_directive()` branch (checked first, returns
  `("graphs", "")`), `format_graph_list()` (sorted, `graphs=0\n(none loaded)` on empty), and the
  `run_query()` branch (placed before the `profile` check, before `client.select_graph()`, calling
  `get_client().list_graphs()` directly) are all present, unmodified from the plan's exact code
  blocks, in the right locations. `TOOL_DESCRIPTION`/`SERVER_INSTRUCTIONS` gained exactly the one
  sentence/clause the plan specified, at the cited anchor points. No drift.
- **FR-2 regression pins verified byte-for-byte unedited in the diff itself** (not just "still
  passing"): `git diff -- cypher-mcp/tests/test_server.py` shows exactly 5 hunks (62 insertions, 1
  deletion — the `FakeClient.__init__`/`list_graphs` additions, the new parametrize cases, the 4 new
  offline tests, the 1 new live test); grepping that diff for
  `test_input_schema_has_two_required_params_and_one_optional_agent` and
  `test_exactly_one_tool_named_query` returns nothing — neither test appears in the diff at all, so
  their assertions are provably untouched, not merely still green.
- **`docs/requirements/cpg-query-access.md`'s `git diff` shows only the header-note paragraph
  changed** (5 lines added, 2 removed, all inside the `> **Note:**` block) — nothing else in the
  archived document is touched, confirming the plan's own done-condition.
- **Mutation-restore sanity check.** `server.py`'s diff (49 insertions, 3 deletions total) shows the
  anchored regex (`\s*\Z`) present and the `"graphs"` branch present in `run_query()`, both exactly
  once; a repo-wide grep for `TODO|FIXME|XXX|pdb.set_trace|print(` in `server.py` turns up only the
  pre-existing `_log()` helper's stderr `print()` (unrelated, predates this diff) — no leftover
  debug artifacts or partial mutations.
- **`docs/HISTORY.md`'s new entry accurately describes what was built**, not just what was planned:
  it states the real, independently-reconfirmed suite counts (125/11, 11/125, 116/11, 11/116), names
  the two regression pins by name as having been run and passing unedited, and describes the two
  mutations actually performed (dropping the `\s*\Z` anchor; deleting the `"graphs"` branch) — all
  consistent with the diff and with my own re-derivation.
- **`SERVER_INSTRUCTIONS`/`TOOL_DESCRIPTION` length re-verified post-edit**: 1326 / 1105 chars
  (imported and measured directly), both comfortably within the 2000-char pin
  `test_server_instructions_are_present_and_bounded` asserts.
- **Convention fit.** The new code matches the file's existing style throughout: the `#:`-prefixed
  comment before the new regex mirrors `_DIRECTIVE_RE`'s own comment, the new formatter
  (`format_graph_list`) mirrors `format_plan`'s docstring register, and the new tests follow the
  existing `fake_client`/`FakeClient` fixture pattern rather than inventing a new one.
