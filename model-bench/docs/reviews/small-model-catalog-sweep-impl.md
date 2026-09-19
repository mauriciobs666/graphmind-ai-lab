# Small-Model Catalog Sweep — Unit C Implementation Review (`consolidate_sweep_reports.py`)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M9)

**Filename note:** the dispatching brief asked for this to land as a new section inside the
existing `docs/reviews/small-model-catalog-sweep.md` (the plan review). I deviated: that document
is the review of the **plan** (an `architect` artifact); this is a review of **code** claiming to
implement one unit of it, which this repo's own filename convention (`AGENTS.md`'s closed role
set, `-impl`) and this component's own precedent
(`docs/reviews/small-model-benchmarking-impl.md`, plus a dozen more across the monorepo —
`git ls-files '*-impl.md'`) route to a separate `-impl` document on the same slug, never grown
into the plan review. Cross-referencing rather than merging.

## 1. Scope & verdict

Reviewed `model-bench/scripts/consolidate_sweep_reports.py` and
`model-bench/tests/test_consolidate_sweep_reports.py` (new, 16 tests) — Unit C of
`docs/plans/small-model-catalog-sweep.md` **Version 2** (§3.5, §4/§5 Unit C) — against that plan,
`docs/requirements/small-model-catalog-sweep.md` FR-9/FR-10, and `scripts/refresh_golden.py`'s
existing house style (this component's own precedent for a "light touch," non-mutation-tested
utility script). Explicitly a light-touch, static/fixture-based review per the brief and per the
plan's own §4 Unit C note — no live `rank` output exists yet (Units A/B in-flight), so end-to-end
integration is out of scope here and expected as a follow-up once they land.

**Verdict: approve.**

**CPG:** considered, not relevant — queried `GRAPHS` directly: `cpg_falkorchat`, `kaizen_team`,
`reference`, `ws:*` are loaded; no `cpg_model-bench` graph exists. A ~270-line script plus its test
file were read in full; no call-graph tooling was needed beyond direct reading.

## 2. Findings

### 2.1 [MINOR] The unknown-pack refusal is verified at the function level only, not at the CLI (`main()`) level

The plan/brief's required refusal path — a marker naming a pack this script has no role mapping
for — is asserted only against `build_consolidated_document` directly
(`tests/test_consolidate_sweep_reports.py:280-291`, a hand-built `RankMarker`), never through
`main()`. I independently drove it through the CLI to confirm the wiring itself is correct:

```
$ report.md carries: <!-- rank-report: pack=some-future-pack metric=x top=model-a value=0.5 ci=[0.4,0.6] -->
exit code: 1
out exists: False
```

This passes today, but nothing pins it — a future change to `main()`'s try/except ordering
(e.g. reordering the `--out` mkdir before the `build_consolidated_document` call) could silently
reintroduce a partial-write bug with no test catching it. **Suggested fix:** add
`test_main_exits_1_on_unknown_pack_marker`, mirroring the existing
`test_main_exits_1_when_no_reports_carry_any_marker` shape, asserting exit code 1, the pack name in
stderr, and `not out_path.exists()`.

### 2.2 [NIT] `docs/HISTORY.md` carries no Unit C entry yet

The plan's own Documentation section (§4, "whoever lands each unit updates... `docs/HISTORY.md` (a
dated entry per unit landed)") applies to this unit; `docs/HISTORY.md` has no
`consolidate_sweep_reports.py`/Unit C entry today. The coordination ledger
(`docs/plans/small-model-catalog-sweep-coordination.md:56`) marks U3c `delivered`, so this may be
intentionally deferred to a coordinated close alongside Units A/B rather than dropped — flagging so
`teco` doesn't lose track of it before this ledger row gates closed. Not a code defect, not
gating.

## 3. Verification performed

- **Marker-parsing contract vs. plan §3.5**, including the guard-judge two-marker case: read
  `_MARKER_RE`, `parse_markers`, `load_markers` (`scripts/consolidate_sweep_reports.py:64-125`) and
  the matching fixtures (`tests/test_consolidate_sweep_reports.py:34-127`) — confirmed the
  `five_reports` fixture yields exactly 6 markers across 5 files, guard-judge contributing 2 kept
  adjacent in file order, matching `docs/plans/small-model-catalog-sweep.md:499-519` exactly.
  `.venv/bin/python -m pytest -q tests/test_consolidate_sweep_reports.py` → **16 passed**.

- **No-cross-pack-arithmetic invariant (FR-9), independently verified, not taken on trust:**
  - Read the whole script for numeric parsing: `grep -n "float(\|int(\|Decimal\|eval("
    scripts/consolidate_sweep_reports.py` matches nothing in executable code — every hit is in a
    docstring/comment describing the invariant. Every `RankMarker` field is typed `str` and only
    ever f-string-interpolated (`_index_table`, `:154-165`), never parsed.
  - Rather than trust the coder's reported mutation (temporarily edited and reverted during their
    own session, which I cannot re-inspect after the fact), I constructed an **independent
    equivalent mutant without touching the repo file**: imported the real module into a scratch
    script and monkeypatched `_index_table` in-memory to inject a genuine cross-pack sum
    (`sum(float(m.value) for m in markers)`) into the rendered document, then called the actual
    test function `test_assembler_never_combines_two_packs_values` against the mutated module.
    Result: the real test **failed** with `AssertionError: consolidated document contains a
    cross-pack sum ('1.0779999999999998')...` — confirming the guard is real, not decorative.
    Reverting the monkeypatch (i.e., the shipped code) passes clean. See Appendix for the full
    scratch script.

- **Error paths:** `main()`'s missing-`--reports`-file check (`:246-252`, exit 2) and the
  zero-marker/unrecognized-pack refusal (`ConsolidateSweepReportsError` → exit 1, `:256-260`) are
  both real (read) and both tested at the CLI level for the missing-file and zero-marker cases
  (`test_main_exits_2_on_a_missing_report_file`, `test_main_exits_1_when_no_reports_carry_any_
  marker`). The unrecognized-pack case is real (I drove it through `main()` myself, §2.1) but only
  pinned at the function level, not the CLI level — finding 2.1.

- **`_ROLE_BY_PACK_ID` design choice:** cross-checked its five entries against each pack's own
  `pack.json` `"role"` field directly (`grep -n '"role"' packs/*/pack.json`) — verbatim match on
  all five (`embedder`, `guard-judge`, `nlq-generator`, `tool-caller`, `chat-responder`), and
  against `modelbench/roles.py`'s canonical `ROLES` tuple (same five, FR-21's closed set — "there
  is no sixth, and no default"). Confirmed no existing `packId -> role` constant already exists
  elsewhere in `modelbench/` for this to duplicate (`packs.py` only ever reads `role` off a loaded
  `pack.json`, never exposes a bare mapping). The reasoning holds: this is a genuinely new,
  complete, closed-set constant, not a re-derivable duplicate, and matches this component's own
  "name a closed set explicitly" convention (`AGENTS.md`).

- **Suite/lint, run myself:**
  `.venv/bin/python -m pytest -q tests/test_consolidate_sweep_reports.py` → 16 passed.
  `.venv/bin/ruff check scripts/consolidate_sweep_reports.py tests/test_consolidate_sweep_reports.py`
  → **All checks passed!** (zero findings in Unit C's own two files).

- **The reported `ruff` finding's attribution, checked rather than assumed:** `git diff --stat`
  shows only `modelbench/stats.py` and `tests/test_stats.py` as modified (`M`); Unit C's own two
  files are new/untracked (`??`). Running `ruff check tests/test_stats.py` directly reproduces 5
  `E501` (line-too-long) findings, all inside test bodies/docstrings added by the concurrent Unit A
  work (`tests/test_stats.py:969-1031`), none touching anything Unit C wrote. Attribution confirmed
  correct.

## 4. What's solid

- Zero cross-pack arithmetic, verified two independent ways (static read + live mutation test),
  matching FR-9's letter and purpose exactly — this is the property the unit exists for, and it
  holds.
- Explicit `--reports` paths (never glob-discovered), matching the plan's stated rationale
  (no silent stale/wrong-session pickup) and consistent with `refresh_golden.py`'s own
  explicit-input philosophy.
- The refuse-rather-than-silently-partial discipline is real: no output file is written on either
  error path (confirmed for the missing-file, zero-marker, **and** unknown-pack cases — the last
  one my own addition to the check, §2.1).
- Docstrings do real work (contract citations back to plan section numbers, explicit "why not
  X" notes) without narrating obvious code — matches this component's established documentation
  density (`refresh_golden.py`).
- Test fixtures explicitly exercise the variable-marker-count shape the plan calls out as
  easy to get wrong (six markers/five files, guard-judge's two), including the
  role-count-vs-marker-count distinction (five roles from six markers).

## 5. Open questions

- None blocking. §2.2 (the pending `HISTORY.md` entry) is for `teco` to resolve at whatever point
  this ledger row is judged fully closed — possibly batched with Units A/B rather than per-unit.

## Appendix — independent mutation-test script

```python
import sys
from pathlib import Path

sys.path.insert(0, "/home/mauricio/prg/graphmind-ai-lab/model-bench/scripts")
sys.path.insert(0, "/home/mauricio/prg/graphmind-ai-lab/model-bench/tests")
import consolidate_sweep_reports as csr
import test_consolidate_sweep_reports as t

_orig_index_table = csr._index_table
def _mutated_index_table(markers, *, out_dir):
    base = _orig_index_table(markers, out_dir=out_dir)
    total = sum(float(m.value) for m in markers)
    return base + f"\n\n<!-- injected: {total} -->"
csr._index_table = _mutated_index_table

try:
    t.test_assembler_never_combines_two_packs_values(Path("/tmp"))
    print("TEST PASSED (mutation NOT caught) -- unexpected")
except AssertionError as e:
    print("TEST FAILED as expected (mutation caught):")
    print(e)
```

Output: `TEST FAILED as expected (mutation caught): consolidated document contains a cross-pack sum
('1.0779999999999998') that no single marker's own value could have produced — the assembler
combined two packs' numbers`

No repository file was edited to produce this — the mutation was applied to the imported module
object in-process only.
