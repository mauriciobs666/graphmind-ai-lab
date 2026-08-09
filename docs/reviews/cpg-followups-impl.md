# Review — CPG follow-ups implementation (U3 + U4 + U5)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** C-314, C-315, C-318, C-321 (post-M2/M3 follow-up)

> **Split note:** this path was written to concurrently by two parallel Wave-2 review units
> (`analyst` reviewing U3–U5, `cobb` reviewing U1/U2/U6) — the later write won and this file now
> holds only `analyst`'s content. `cobb`'s review of U1 (C-308) + U2 (C-312) + U6 (C-319) lives at
> `docs/reviews/cpg-followups-skills-impl.md` instead, per the collision fallback in both units'
> briefs.

## 1. Scope & verdict

Reviewed the uncommitted working-tree diffs for three parallel units dispatched by
`docs/plans/cpg-followups-coordination.md`:

- **U3** (`coder`, C-314 + C-315) — `cpg/mcp/server.py`'s `render_cell()`: map cells coerced
  through `dict(value)` to stop leaking `falkordb`'s `OrderedDict` class name; booleans given an
  explicit `isinstance(value, bool)` branch rendering lowercase `true`/`false`.
- **U4** (`tdd-engineer`, C-318 + C-321 core) — `cpg/mcp/tests/test_server.py`: a new pin on
  `mcp.instructions` (non-empty, ≤2000 chars) and a `uuid4().hex[:8]`-derived
  `_scratch_graph_name()` helper replacing an `os.getpid()`-based live-suite scratch-graph key.
- **U5** (`devops`, C-321 deferred sub-items) — `cpg/mcp/docker-run.sh`, `cpg/mcp/build.sh`,
  `cpg/mcp/image-tag.sh`: `CPG_MCP_NO_PULL=1` on the autobuild path, `image-tag.sh` hardening
  (missing-dir/symlink/failed-`find` hard errors, `.pytest_cache` exclusion, file-mode left
  deliberately unhashed), and `build.sh`'s `test`-before-`runtime` build reordering to close the
  `--no-cache` dependency-drift risk (m-23).

Baseline: current `main` (working tree diffs only, nothing committed). Verified by reading every
diff line, tracing the actual `falkordb` client source in `.venv/` for the map-cell claim, running
the offline suite, running `build.sh --verify-inputs`, `bash -n` on all three shell scripts, and
directly exercising `render_cell()` and `image-tag.sh`'s functions from a Python/bash REPL against
scenarios the new tests don't cover.

**Verdict: needs changes.** One blocker-adjacent finding (rated Major below, see rationale) means
U3's fix does not achieve C-314/C-315's own stated goal for the most realistic query shape those
items were filed against; the rest of U3, and all of U4 and U5, are sound and well-evidenced. This
should loop back to `coder` for U3 specifically — U4 and U5 do not need rework.

## 2. Findings

### Major — U3's fix is shallow: nested maps/lists still leak `OrderedDict` and `True`/`False`

`render_cell()`'s new branches only look at the **top-level** cell value:

```python
elif isinstance(value, dict):
    text = repr(dict(value))
elif isinstance(value, (list, tuple)):
    text = repr(value)
```

`dict(value)` converts the *outer* `OrderedDict` to `dict`, but any **value inside** that dict
that is itself an `OrderedDict` (a nested map) is untouched — `repr()` on the outer `dict` still
calls the nested value's own `__repr__`, which is `OrderedDict`'s. Same story for the `list`/`tuple`
branch: it never touches map or bool values inside the collection at all. Verified directly against
the running module (`cpg/mcp/.venv`):

```
>>> render_cell({'NAME': 'foo', 'IS_EXTERNAL': False}, 300)
"{'NAME': 'foo', 'IS_EXTERNAL': False}"        # capitalized, not lowercase
>>> render_cell(OrderedDict([('a', 1), ('nested', OrderedDict([('b', 2)]))]), 300)
"{'a': 1, 'nested': OrderedDict({'b': 2})}"    # class name still leaks
>>> render_cell([1, OrderedDict([('a', 1)])], 300)
"[1, OrderedDict({'a': 1})]"                   # class name leaks in a list too
```

This is not a contrived edge case. `IS_EXTERNAL` is the *literal example* C-315's own backlog entry
cites for why booleans matter (`docs/BACKLOG.md:327-331`, quoting the `SKILL.md` gotcha "CPG
booleans are real booleans"), and `RETURN properties(m)` — a single map cell whose values include
that exact boolean field — is confirmed (`falkordb/query_result.py:__parse_map`, which recurses via
`parse_scalar` for every value) to be the shape the client library actually returns for any
map-valued or list-of-maps query. So the fix's own docstring overclaims: "maps are coerced to a
plain `dict` first so a client-library subclass … never leaks its class name into the rendering"
and "booleans render lowercase … rather than Python's" are both false once a map or list is anything
but flat.

Why it's Major rather than Blocker: it is cosmetic (FalkorDB accepts boolean literals
case-insensitively per C-315's own "verified cosmetic" note, so nothing round-trips incorrectly),
and both backlog items are already filed 🔵 Low. But shipping this as closing C-314/C-315 in Wave 3
would misrepresent what was actually fixed — the new tests
(`test_render_cell_renders_map_valued_cells_without_leaking_client_type`,
`test_render_cell_renders_booleans_lowercase_like_cypher_json_not_python`) only exercise the flat
case, so nothing pins the gap and a future reader has no signal it exists.

**Suggested fix:** make the coercion recursive — walk `dict`/`list`/`tuple` values and re-apply the
same `None`/`bool`/`dict`/`list` handling to each element before calling `repr()`, or write a small
custom pretty-printer instead of leaning on Python's built-in `repr()` for collections (which will
always call each element's own `__repr__`, so no amount of top-level-only coercion closes this
class of gap). At minimum, if a recursive fix is judged out of scope for a 🔵 Low item, the backlog
closing note and the docstring should explicitly say "top-level cells only" rather than the current
unqualified claim, and a test should pin the known limitation (`pytest.mark.xfail` or an explicit
"nested — not fixed" assertion) so it isn't rediscovered as a surprise later.

### Minor — `cpg/mcp/README.md`'s C-321 warning is now stale

`README.md:434-439` still describes the `os.getpid()` bug as present tense ("`os.getpid()` is
**1**... Backlog **C-321** fixes the root cause") and instructs operators to manually check
`GRAPH.LIST` for residue before every concurrent live run. U4's diff lands exactly that fix
(`_scratch_graph_name()` via `uuid4().hex[:8]`), but `README.md` is untouched by any of the three
diffs under review — confirmed via `git status` (not in the modified-files list) and `grep -rn
getpid` (only hits are the README prose and `test_server.py`'s own explanatory comments about the
*old* behavior, which are appropriately past/historical in framing). This isn't in U4's stated
scope (`cpg/mcp/tests/test_server.py` only) or its done-condition, but the brief specifically asked
to check for it, and a reader hitting this section after the fix lands would get instructions to
work around a bug that no longer exists.

**Suggested fix:** fold a doc line into whichever unit's Wave-3 backlog-closing note touches
`docs/HISTORY.md`/`docs/BACKLOG.md`, or file a one-line follow-up: update `README.md:434-439` to
past tense noting the fix landed (uuid4-derived names no longer collide across containers), while
keeping the `GRAPH.LIST` residue-check habit for interrupted runs (that part is still valid advice
independent of the collision fix).

### Informational — `.pytest_cache` exclusion in `image-tag.sh` is currently a no-op, but the right kind

The new `! -path '*/.pytest_cache/*'` filter is added to the `find "$d" ...` walk, where `$d` is
always `tests` (`cpg_mcp_input_dirs` only ever yields `tests`). Verified `.pytest_cache` is created
at `cpg/mcp/.pytest_cache` (top level, both on the host and inside the container — `pytest.ini`
lives at `cpg/mcp/`, `WORKDIR /app` in the `test` stage, so pytest's rootdir is always the
top-level directory, never `tests/`), which the directory walk never reaches regardless of this
exclusion. So today, this specific line changes no hash. Not a defect: `docs/BACKLOG.md:410`
(m-21) frames this as restoring an *invariant* the code comments already claimed ("three places
claim it mirrors [`.dockerignore`]"), and `.dockerignore`'s blanket `**/.pytest_cache` pattern is a
real defense against a future test-runner invocation that *does* create a nested cache dir. Correct,
defensive, low-cost — just noting it so nobody mistakes the offline suite's unchanged hash-related
output as evidence this line does anything observable today.

## 3. What's solid

- **U3 — boolean/isinstance ordering is correct.** `isinstance(value, bool)` is checked before any
  other branch (and there's no separate `int` branch to worry about since `int` falls through to
  `str()` regardless), so the well-known `bool`-is-a-subclass-of-`int` footgun is correctly avoided.
- **U3 — the flat-case fix is exactly right and the client-type claim is verified, not assumed.**
  Traced `falkordb/query_result.py:__parse_map` directly (`.venv/lib/python3.12/site-packages/`):
  `OrderedDict` really is the only map type the pinned `falkordb>=1.6,<1.7` client returns, and it
  really is a `dict` subclass, so `isinstance(value, dict)` genuinely catches every map shape the
  client can produce at the top level — the flat-case fix does exactly what it claims for that case.
- **U4 — both new tests are meaningful, not tautological.** The instructions pin checks real
  properties (non-empty, ≤2000 chars) rather than re-asserting the source string; the
  `_scratch_graph_name` uniqueness test is a genuine offline regression pin for a bug that could
  otherwise only be caught by an expensive/flaky concurrent-container reproduction.
- **U4 — scratch-graph naming stays recognizable.** The `_cpg_mcp_selftest_` prefix is preserved
  ahead of the uuid suffix, so `GRAPH.LIST` residue-hunting (documented in `README.md`) still works
  unchanged.
- **U5 — the SIGPIPE-vs-real-`find`-failure distinction is robust, verified empirically, not just
  plausible.** Reproduced both cases directly: a `find | head -n1` truncation reliably produces a
  non-zero pipeline status under `pipefail` with **empty** stderr (confirmed twice on this box), and
  a genuine `find` error (planted a `chmod 000` directory) reliably produces **non-empty** stderr
  alongside the same non-zero status. Judging failure by `errfile` content rather than exit code is
  the right call and it holds up.
- **U5 — `build.sh`'s reordering is a verified no-op when `--no-cache` isn't requested.** Traced
  both branches: `runtime` is unconditionally built in the `else` arm regardless of `$NO_CACHE`, and
  when `NO_CACHE=0` the `CACHE_FLAG` array is empty in both the old and new code paths, so the
  observable Docker invocation is identical. `--verify-inputs` and `bash -n` on all three scripts
  pass.
- **U5 — the `CPG_MCP_NO_PULL` scoping is correct.** It's set only at the `docker-run.sh` autobuild
  call site (`CPG_MCP_NO_PULL=1 "$HERE/build.sh" --runtime-only`); `build.sh` itself only reads it
  (`${CPG_MCP_NO_PULL:-0}`), so a manual `cpg/mcp/build.sh` run still pulls by default, exactly as
  the brief asked to confirm.
- **U5's m-18 through m-23 sub-items are all present**, matching `docs/reviews/cpg-mcp-containerization.md`'s
  §17–§18 findings one-for-one (verified by grepping both documents' `m-1*`/`m-2*` labels).
- **Cross-cutting: no interaction/regression between U3 and U4 in the shared `test_server.py`.**
  Both diffs land in the same file (test additions interleave, e.g. the boolean test split right
  where U3's map test was inserted) with no structural conflict. Re-ran the offline suite after
  reading both diffs together: `66 passed, 7 deselected` — matches the reported figure exactly.
  `bash build.sh --verify-inputs` also passes clean against the current tree.

## 4. Open questions

- Does `teco` want U3's Major finding fixed before Wave 3 closes C-314/C-315, or is a scope-narrowed
  closing note ("flat top-level cells only") an acceptable outcome for two 🔵 Low items? Either is
  defensible; the finding above exists so that choice is made deliberately rather than by omission.
- Is the `README.md` staleness (Minor finding) folded into U4's existing scope, a new one-line
  follow-up, or absorbed into Wave 3's consolidated doc pass? Not blocking either way.

## Pass 2 — 2026-08-09 (re-review of the Major finding's fix)

Scope: `coder` fixed Pass 1's Major finding (nested maps/booleans still leaking `OrderedDict`/
`True`/`False`). Re-reviewed only the resulting diff — a new `_ReprAsIs` sentinel class and
`_normalize_for_repr()` recursive helper in `cpg/mcp/server.py`, `render_cell()`'s collection
branch now routed through it, the corrected docstring, and the new test
`test_render_cell_normalizes_booleans_and_maps_at_any_nesting_depth` in
`cpg/mcp/tests/test_server.py` — per `teco`'s request. Did not redo the U4/U5 portions; spot-checked
they're undisturbed (below).

**Independently re-verified every claimed case** (ran directly against the module, not trusted from
the report):

```
>>> render_cell({'NAME': 'foo', 'IS_EXTERNAL': False}, 300)
"{'NAME': 'foo', 'IS_EXTERNAL': false}"          # Pass 1's exact repro — now fixed
>>> render_cell(OrderedDict([('a', 1), ('nested', OrderedDict([('b', 2)]))]), 300)
"{'a': 1, 'nested': {'b': 2}}"                   # map nested in map — fixed
>>> render_cell([1, OrderedDict([('a', 1)])], 300)
"[1, {'a': 1}]"                                  # map nested in list — fixed
>>> render_cell((1, True, {'x': False}), 300)
"(1, true, {'x': false})"                        # tuple w/ bare bool + nested map+bool, tuple-vs-list preserved
```

All four match the claimed output exactly.

**Additional edge cases probed beyond the new test** (all passed, no issues found):

- Dict key containing an apostrophe (`{"it's": True, ...}`) — Python's native quote-switching in
  `repr()` for the key is untouched by the normalization (only *values* are normalized; Cypher map
  keys are always strings via `__parse_string`, never bool/dict), so no interaction bug.
- 10-level-deep nested map ending in a bare `bool` — renders correctly (`true` at the bottom), no
  `RecursionError`.
- Single-element tuple containing a bool (`(True,)`) — correctly renders `(true,)` with the
  required trailing comma; `type(value)(...)` reconstruction preserves list-vs-tuple identity
  faithfully, confirmed for both the multi-element and singleton case.
- Empty `dict`/`list`/`tuple` — no crash, render as `{}`/`[]`/`()`.
- A `set` nested inside a dict — falls through `_normalize_for_repr` unchanged (not `bool`/`dict`/
  `list`/`tuple`) and renders via its own native `repr()`. Not a defect: FalkorDB's client never
  returns a `set` (Cypher has no set type distinct from list), so this is out of scope, and behavior
  is unchanged from before the fix either way.
- A non-`Mapping` custom object without its own `__repr__`, nested inside a dict — falls through to
  default `object.__repr__` (a memory address). **Confirmed pre-existing, not a regression**: `git
  show HEAD:cpg/mcp/server.py` shows the list/dict branch has *always* called plain `repr()` on its
  top-level value, so any nested value without a custom `__repr__` got this treatment before U3
  touched anything at all — including FalkorDB's own `Node`/`Edge` classes, which define only
  `__str__` (confirmed in `.venv/lib/python3.12/site-packages/falkordb/{node,edge}.py:__str__`,
  no `__repr__`). Out of scope for this fix; noted for the record, not a finding against this diff.
- `None` nested inside a dict/list still renders as Python's `None`, not `"null"`
  (`render_cell({'a': None}, 300)` → `"{'a': None}"`) — same as before this diff. The corrected
  docstring's "any nesting depth" claim is scoped to booleans and maps only and makes no claim about
  `None`, so this is not an overclaim, just a pre-existing quirk of the same shape. Possible future
  backlog item, not this fix's job.

**`_ReprAsIs` correctness:** no issue found. It's constructed and consumed entirely inside
`_normalize_for_repr` → `repr(...)`, in the same expression (`text = repr(_normalize_for_repr(value))`);
its instances are never persisted, returned, or inspected by anything else, so its unusual
"`__repr__` returns raw unquoted text" identity never leaks past that one call. No interaction with
dict-key string-escaping (keys are never normalized, only values). No cycle-detection concern:
`_normalize_for_repr` builds brand-new `dict`/`list`/`tuple` objects from a linearly-parsed
FalkorDB result, so self-referential structures cannot arise.

**Docstring:** re-read against the verified behavior — accurate. It now says the recursive
normalization means a map renders as a plain `dict` literal and a boolean lowercase "at *any*
nesting depth... however deeply nested," which matches what was independently verified above; no
overclaim remains.

**Test suite:** re-ran offline — `67 passed, 7 deselected`, exactly as claimed (Pass 1 was 66; the
one new test, `test_render_cell_normalizes_booleans_and_maps_at_any_nesting_depth`, accounts for the
delta, and it subsumes rather than replaces the flat-case test kept alongside it).

**U4/U5 spot-check:** `git diff --stat` for `cpg/mcp/docker-run.sh`, `cpg/mcp/build.sh`,
`cpg/mcp/image-tag.sh` is unchanged from Pass 1 (identical line counts); `test_server.py`'s
instructions-pin and `_scratch_graph_name` sections are untouched — only the `render_cell` test
block grew. Neither needed re-review, matching `teco`'s note that nothing there looks disturbed.

**Updated verdict: approve.** Pass 1's Major finding is closed, verified independently rather than
taken on the report's word. No new issues found in the fix itself. Pass 1's two non-blocking items
stand, unaffected by this diff, and remain open per the Pass 1 open questions above:

- Minor — `cpg/mcp/README.md`'s C-321 warning is still stale (present-tense description of a bug
  that U4 already fixed).
- Informational — `image-tag.sh`'s `.pytest_cache` exclusion is still a no-op given the real
  directory layout, but is the right defensive addition regardless.
