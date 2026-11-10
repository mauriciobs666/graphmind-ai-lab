# C-1001 P1-05 dispatch 5a — Python normalizer port: review

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope & verdict

Diff-scoped review of the uncommitted dispatch 5a under `cpg/`: `extractors/joern/normalize/{__init__,core,python}.py`,
`tests/test_normalize_core.py`, `tests/test_normalize_python.py`, `tests/normalize_helpers.py`,
`tests/slice/test_normalize_equivalence.py`, and the `extractors/joern/normalize/` row of `cpg/README.md`. Baseline: the
P1-05 row of `docs/plans/connected-code-graph-p1.md`, its section in `docs/plans/connected-code-graph-p1-extraction.md`
(5a/5b, behaviours (1), (2) Python half, (4)), the bundle spec chain `-spec3`/`-spec4`/`-spec5` (§1.5 `import`, §4),
the P1-04 review `docs/reviews/connected-code-graph-p1-04.md`, root `REVIEW.md` (no `cpg/REVIEW.md` exists). Oracle: the
slice normalizer `cpg/spikes/vertical-slice/normalize.py` and the stored bundle (sha256 below). Not reviewed: `ccg/render/*`,
`tests/test_render.py`, `cpg/sandbox/*`, `cpg/infra/*`, `ccg/broker/*`, `ccg/packrt/*`.

**Verdict: approve with suggestions.** The Python half is a faithful port: all 4,686 symbols and all 926 imports of the
stored `falkor-chat/server` bundle reproduce under my own comparison, and the same-name-sibling assumption is settled live
in the delegate's favour for properties and overloads. No blocker (every defect below is inherited from the oracle, and the
plan's "equivalence before improvement" rule forbids fixing it in this dispatch). But two latent defects the port carries
are now evidenced and are **not** pinned by any test, the delegate's fixture assumption hides one of them, and the mutation
claim does not hold up under an independent operator set. These need to land before 5b builds on `build_scopes`.

CPG: considered, not relevant — the unit under review is new, uncommitted code (and the oracle is a frozen spike) that no loaded `cpg_*` graph indexes; verification was by execution and a live Joern probe.

Checks run (all by execution): `pytest tests/test_normalize_core.py tests/test_normalize_python.py
tests/slice/test_normalize_equivalence.py` in `cpg/` — 108 passed, 0 skipped (the slice tests ran: artifact present, sha256
`a0cae513…1c86` re-hashed by me and equal to the pin). Full default `cpg` suite with `-x` (render tests excluded): ran to 100%, no failure, 1 skip (`test_rule_pack_leaf_mutation.py:758`, the opt-in slow run, unrelated).

## Findings (ranked)

### F1 — major — a class redefined in one scope is mis-built, and the result depends on the order of the raw facts (verified)

`build_scopes` keys class bodies by `body_fulls = {m["full"]: mid ...}` (`python.py:41`, looked up at `:45`) and decorators by `d["target"]`/`full`
(`python.py:27`, `:105`). Real `pysrc2cpg` suffixes a redefined **function** with `$redefinitionN` but gives a redefined
**class** the same `fullName`, and so the class's methods too. Live probe (real runner, toolchain image, throwaway
container, `extract.sc`): a file with `class B` defined twice, the second with `m` and `n`, produced two
`B.<body>` methods and two `B.m` methods with identical `full`. Feeding those real facts to `build_scopes` gives `B` and `B#2`
**both** with span 13-18 (the first class is at 8-11), `B.m` (line 9) attached to the wrong class, and under 200 random
fact orders only **55/200** give the same scopes. This breaks the plan's stated premise that "raw facts are not byte-stable,
and bundle determinism rests on the normalizer re-sorting" (P1-04 review): here the output depends on dict insertion order
(`m2s[bid]` and `body_fulls` are last-writer-wins). On the monorepo's 569 tracked `.py` files there is no such class (verified
with an `ast` sweep), so it is latent, and it is in the oracle too (D7/"equivalence before improvement": do not fix in 5b
before byte equivalence). Actions: (a) pin it now with a characterization test over the **real** fact shape (two `B.<body>`
with equal `full`), marked as a known defect and asserting at least that the result is order-independent once fixed; (b) add
a backlog/delta item "redefined class: key bodies by (full, line, col)", to be pinned as a named delta at the dispatch
that changes behaviour; (c) correct the README claim "raw-fact record order is not an input" until then.

### F2 — major — `py_resolve` emits an un-normalized submodule path; `lstrip` is applied to only one of its two outputs (verified)

For a bare-dot import at the unit root (`from . import sibling` in a file directly under the unit root; also
`from .. import x` one level down) `d` is `Path(".")`, so the submodule candidate is `./sibling.py` and is **not** passed through
`c.lstrip("./")` (`python.py:191` strips only `res`). Executed: unit root `.` gives `resolvedPath = "__init__.py"` but
`_subs = {"sibling": "./sibling.py"}`; unit root `R` gives `R/./sibling.py`. `Resolver.element` then emits
`{"ref": "./sibling.py::X"}` (`core.py:260`, the `attr` branch), a dangling `ref`: the later linking steps that join a `ref` to a
`file.path` will silently miss it. The slice has 0 such imports, so equivalence holds. The `lstrip("./")` quirk itself
is a character-set strip, not a prefix strip, but on every reachable candidate (`./__init__.py` only) it is harmless; the
real corruption is the *asymmetry*. Fix at the behaviour-changing dispatch: normalise `c` once (`PurePosixPath(c)`) before both outputs. Pin
today: a test with an importer directly under the unit root for both roots (`.` and `R`) asserting the current
`./sibling.py`/`R/./sibling.py`, labelled "slice quirk, delta pending".

### F3 — major — the mutation evidence does not support "7 survivors, all equivalent/ill-formed" (verified by independent run)

The delegate's ledger line (`docs/plans/connected-code-graph-coordination.md:732`) reports 122/129 killed and names no
survivor, so none of the seven claims can be checked as stated. I ran my own AST mutator (comparison, boolean, `not`, `+/-`,
int/bool constants, `min/max`, `.strip/.lstrip/.rstrip/.sort` drops, return values) over `core.py` and `python.py` on a
copy, 214 mutants, same suite including the slice tests: **197 killed, 17 survived**. Triage by execution:

| Survivor (file:line, mutant) | Verdict |
|---|---|
| `python.py:220` `asn or ""` → `and` | **not equivalent.** Source `from m import a as b, a as c` yields both aliases today; the mutant keeps one (`q as r, q as r2` loses `r2`). No test has two different aliases of one name. Spec3 §1.5 says `a` and `a as b` are two entries |
| `core.py:277` `"template" in p` / `"concat" in p` → `not in` (2) | **not equivalent.** Depth-3 `fmt(fmt(fmt(lit)))` gives `template[template[template[const]]]` today and a `value-set-cap` under the mutant. The depth-cap rule has no test |
| `python.py:191` drop `.lstrip` | **not equivalent.** `./__init__.py` vs `__init__.py` for `from . import x` at the unit root. The quirk the delegate says is recorded in tests is in fact unpinned (see F2) |
| `python.py:98` `and` → `or` | not equivalent in a multi-file unit (the orphan fallback may pick another file's module); every test has one file |
| `python.py:134` `1 <= ln` boundary | decorator on line 1 of a file untested (harmless for `overload`, wrong name for a plain decorator whose CPG name differs) |
| `core.py:41,42` `collapse`: limit, `strip`, `rstrip`, `return` (4) | **not equivalent.** `collapse` has no test at all (`grep` over `tests/` finds none). Consumers arrive in 5b, so either test it now or move it to 5b |
| `core.py:33` message index `vi[0]`→`vi[1]` | equivalent in behaviour (the error message text only, which the test does not assert) |
| `core.py:228`, `core.py:235` `return None` → `return None` | equivalent (artifact of my mutator: the original is already `return None`) |
| `python.py:181` drop `.lstrip(".")` | equivalent: `parts` filters empty segments (probe outputs identical; by reading) |
| `python.py:228` sort-key flag `0→1`, `1→2` (2) | equivalent: an absent alias compares as `""` which sorts first anyway; the flag is redundant (probe identical) |
| `python.py:34` module `col=1→2` | equivalent: `Scope.col` of a module is read nowhere (`grep -n "\.col\b"`: only the lambda key and the sort, both skip modules) |

So 7 of my 17 are equivalent (I confirmed `python.py:34/181/228` by reading plus an identical probe, `228`
twice; the two `return None` rows by inspection) and **10 are genuine test gaps**. This is not a different
operator set explaining a different count: the delegate's equivalence claims may be sound, but the claim that the contract
boundary is fully pinned is not. The plan (`-p1.md` §8 TDD ordering) requires surviving mutants "listed in the step's
report"; the list must be named, not counted. Required: name the delegate's seven, add the five tests above
(`as`-alias pair, depth-3 value-set cap, bare-dot root import, two-file orphan, line-1 decorator), and test or relocate `collapse`.
My own plumbing mutants (15 hand-written data-path mutations: decorator lookup key, parent id, span end source, decorator column,
level loop, fingerprint blank filter, numbering key order, accessor in group key, `<init>` name, casefold, decode, `splitlines`) were all
killed, and also killed by the golden tests **alone** with the slice tests excluded, except one (see F6). So the golden
layer is real, which is the good half of this finding.

### F4 — minor — the unit test file and `import.symbolId`: a Python-half-decidable field is omitted from the slice test

`tests/slice/test_normalize_equivalence.py` drops `import.symbolId`, but for Python it depends only on scopes, spans
and `py_imports` — all in 5a. I recomputed it from `build_scopes` (innermost enclosing symbol by
`span_start <= line <= lineEnd`, the oracle's rule): **0 mismatches over 926 imports, 87 carry a `symbolId`**. `complexity` needs conds
and `lastChanged*` needs blame, both 5b (the plan's 5a file list excludes them; fine). Add `symbolId` now, cheaply, so 5b's byte comparison is not the
first time it is checked. The plan lists the slice test as a **5b** file; adding a partial 5a version is sensible and is
the right D9 posture (equivalence green at the step that ports it), provided 5b keeps extending the same file.

### F5 — minor — `Builder` requires CPython 3.12 for every language

`require_runtime()` runs in `Builder.__init__` for `lang == "javascript"` as well, but spec4 §3 only concerns the Python
normalizer (`ast`-derived spans and ids). It is fail-closed and conservative, but it will refuse a JS bundle on a host with
another Python, and `Resolver.lit` uses `ast.literal_eval` on JS code, so the claim "TS/JS does not depend on the host
`ast`" is itself not clean. Not a defect today. Decide in 5b which it is and say so in the docstring.

### F6 — nit — one plumbing mutant survives; recorded for the owner

`line=bm["l"], col=bm["c"]` → `t["l"], t["c"]` (class scope position from the `<body>` method vs its `TYPE_DECL`) survives
both the goldens and the slice. The facts agree for every class I probed (undecorated), and `apply_py_spans` re-keys classes by
source `(name, lineno)` anyway, so it is likely equivalent. Unverified for a decorated class: I did not probe one.

### F7 — nit — citation: "P1 index D9"

`core.py`'s docstring cites D9 for "equivalence before improvement". D9 is the skip rule (a skipped oracle test is unrun,
not green); the equivalence rule is `-p1.md` §8 "Equivalence before improvement" (and D7). Cite those.

## What the checks settled

**1. Equivalence (independent).** I built symbols and imports from the production scope tree but compared with my own
definition, and recomputed lineStart/lineEnd/fingerprint/parentId/name/nameNorm/fileId and import ids myself rather than calling the
test's helper. Results: 4,686 stored symbols = 4,686 built, id sets equal, 0 differing; multiset of full tuples equal with **0**
duplicate tuples; multiset *ignoring ids and parents* (qname, kind, lineStart, lineEnd, fingerprint) equal; per-file symbol counts
equal across 107 files; `fileId` and `nameNorm` recomputed (0 bad). 926 stored imports = 926 built, multiset equal, ids
`import_id(fileId, ordinal)` all recomputed (0 bad). Kinds: 3,141 function, 875 method, 348 type, 185 lambda, 107 module, 30
accessor. Omitted by the test: `complexity`, `lastChangedAt/Commit` (5b: need conds and blame), `import.symbolId` (F4). **Weakness
of my own check:** spans and decorators still come from the production `py_spans`/`decorator_name`, so those two are checked only against the stored
bundle, not re-derived.

**2. Same-name siblings (live).** I ran `offline-run.sh` (container backend, offline mode, `CCG_AUDIT_DIR` pointed at a scratch dir so
`cpg/.audit` is untouched) on a planted file with a property getter/setter/deleter, an overload pair plus implementation, and
duplicate methods: `joern-parse /src --language pythonsrc`, then `extract.sc`. `pysrc2cpg` gives each redefinition its own
`fullName`: `A.x`, `A.x$redefinition1`, `A.x$redefinition2`; the decorators are keyed to the right target
(`property`, `x.setter`, `x.deleter` with code and 1-based column as the helper assumes). Production `build_scopes` returned
`A.x.<get>`, `A.x.<set>`, `A.x.<del>`, dropped both overload stubs, numbered `A.g`/`A.g#2`, `f`→`A.f`. Stable in 200/200
shuffles. So the delegate's assumption holds for **functions and methods** (suffix `$redefinitionN`, not the fixture's `$n`, immaterial);
it fails for **classes** (F1). The helper reproduces the real result for the property/overload probe exactly (identical
qname/kind/span lists), and diverges only on the redefined class: the fixture is a self-confirming stand-in for exactly that shape.

**3. Determinism.** Fact kinds 5a consumes: `f`, `m`, `t`, `deco`. On the real 4,686-symbol facts: 8 full shuffles of every kind, each
kind alone ×4, and a full reversal all give identical scopes (qname, kind, span, parent, decorator lines). The in-repo
parametrized test checks `describe()` = (qname, kind, parent) only, not spans, `cpg_ids` or decorators, and its fixture has no
redefined class (F1). **5b must assert** (i) the *whole bundle* (not just scopes) byte-identical under shuffles of every kind
`assemble` reads (`m,t,p,r,c,cs,const,d,cb,deco,f`, with `stats` ignored), over a fixture containing a redefined class, duplicate
`(point, cond)` pairs and a `use == def` edge; (ii) `d`/`cb` are sorted and de-duplicated *by the spec's key* in `assemble_bundle`, with the mutant "drop the sort" red;
(iii) a live pair: two real Joern runs on the same stage give identical bundles, since the raw files are known not to be byte-equal.

**4. Quirks.** `import a.b as c` drops `c`: spec3 §1.5 has no place for a whole-module alias (`names` is `[]`), plain imports
never get `resolvedPath`, and `Resolver.alias` is built from `names` only, so the consequence is **under**-linking (`c.x` stays
`point-derived`), never a wrong link. Kept: correct under the plan. `py_resolve` absolute imports resolve against the unit root: a unit-local
`json.py` would shadow stdlib `json` (not checked against the stored data; flag for P1-09's join review). `lstrip("./")`: F2.

**5. Dead parent branches.** Confirmed on the real facts: **all 4,579** non-module symbols reach their parent through the first
branch (`pk == "METHOD"` and `m2s.get(pid)`), and none through the module fallback (my probes: 8 and 13 of 8 and 13). **Keep** branches
2-3 (cheap, the tests are honest hand-built facts, and a `pysrc2cpg` bump could surface them); but the final fallback
(`parent = file's module`) turns a lookup failure into a plausible-looking wrong qname silently. Have `build_scopes` count it in
`B.report` (as `pythonSpanLookupMisses` is) so 5b can fail or flag; test the multi-file case (F3, `python.py:98`).

**6. CPython 3.12 check.** Fail-closed: `Builder.__init__` calls it before any state; `RuntimeRefused` is a hard error; not `-O`
dependent. It can fail: the parametrized refusals (3.13, 3.11, pypy, 2.x) go red under any mutant of the predicate (my `cmp`/`bool`
mutants on lines 29-33 were all killed except the message-index one), and `test_a_builder_asserts_the_runtime_at_start`
monkeypatches `sys.version_info`. The positive test `require_runtime()` runs the real interpreter, so the suite itself is red on
any host but 3.12 — intended. Gap: the Builder-level test varies version only, not implementation. F5.

**7. Fixtures.** Hand-written values; the helper mimics `pysrc2cpg` shapes. I compared it to real facts: identical on the
property/overload/duplicate-method probe, divergent on a redefined class (F1). The goldens are stronger than they look: all 15 of my
plumbing mutants die with the slice tests excluded (one was a control that I intend to survive; it survived). The self-confirming
risk is therefore confined to shapes the helper normalizes away (`$n` suffix).

**8. Test-first order.** `-p1.md` §8 says `tdd-engineer` steps follow the red-to-green list literally; the delegate wrote the
code first. For a port whose bodies are the oracle's, that is defensible, and the evidence is good where it is good (goldens
kill every data-path mutant I wrote). It is not enough where F3 shows ten unpinned branches: a test-first run would have had
to state an expectation for the alias pair and the depth cap. Accept the order; require the F3 tests.

**9. My own mutation.** Fresh scratch copy (`.../scratchpad/rv/mut_base`, workers `mut_w*`), nothing touched in place: 15
hand-written data-plumbing mutants plus the 214 generated ones, results above; each ran with a positive control (a no-op
edit that must survive and did).

## What's solid

- The port is genuinely verbatim, and the equivalence against the stored bundle is airtight on every field in scope (checked
  four ways above); the test's expected values come from the stored bundle, never from the code under test; the sha256 pin and
  the "skip is unrun" convention (`need()` / `0 skipped`) follow D9.
- Goldens independently kill every data-path mutant I wrote, including ones the slice test also kills; the file's
  hand-counted `WANT` table is checked against the source by `test_the_golden_file_has_the_lines_...`.
- The getter/setter/deleter, overload and same-name-method rules hold on real `pysrc2cpg` output, not only on the
  fixture.
- `Builder` refuses a non-3.12 host at construction.

## Open questions

1. F1/F2 are inherited oracle defects: should they be fixed in a named later dispatch (with `Supersedes`-style delta pins) or
   backlogged? The plan implies the former (P1-09's projection fix is the nearest owner, not an obvious fit; none is named for scope-key fixes).
2. F5: is TS/JS meant to be 3.12-bound?


## Pass 2 (fixes re-checked against the tree)

**Verdict: approve with suggestions — 5b may be released.** F1-F5 and the parent-fallback count are fixed; one residual
test gap (P2-1) and one 5b obligation (P2-2) remain, neither a blocker. Run: the 3 test files, 342 passed, 0 skipped
(slice artifact re-hashed, sha256 prefix `a0cae513fb67…` unchanged).

**Dispositions**
- **F1 — fixed (verified live).** Re-ran the runner (`joern-parse` then `extract.sc`, throwaway containers, scratch
  `CCG_AUDIT_DIR`) on the redefined-class file and on a new file with: a *decorated* class defined twice (`@deco class D`,
  one `@property`, then `@property`+`@x.setter`), `@functools.cache @staticmethod` stacked, a nested `class In` redefined inside
  `E`, a class redefined inside a function, and `@property @functools.cache` / `@p.setter @functools.cache` stacks. Result:
  `B`/`B#2` now 8-10 / 13-18 with members under the right class; every accessor lands on the right class
  (`D.x.<get>`, `D#2.x.<get>`/`<set>`, `E.In#2.m.<get>/<set>/<del>`, `f.L#2.v.<get>/<set>`), multi-decorator lists in source order. **200/200 stable under
  shuffle on each of the redefined-class facts, the decorated/nested facts and the real 4,686-symbol facts**; 0 parent fallbacks, 0 span misses.
  Same-name functions with decorators on one line cannot occur (a `def` takes one line). `o` is **source order** on the real facts
  (`property` o=0 line 72, `functools.cache` o=1 line 73; likewise `cache` 0 / `staticmethod` 1), which is what spec3 §5.12 says
  ("0-based, in source order on that target"), so sorting by `(l, c, o)` is consistent with it and with the `<get>/<set>/<del>` rule (any-match, one decorator each).
  Caveat: `o` restarts at 0 per owner, not per shared `fullName`, which is why ownership must be settled first (P2-2).
- **F2 — fixed (verified).** `_norm` (`PurePosixPath`) is applied to the module candidates and every submodule candidate. Bare-dot and
  `..`/`...` imports from the unit root and from `pkg/`, under roots `.` and `R`, all give dot-free paths
  (`sibling.py`, `R/sibling.py`, `pkg/__init__.py`); a dotted directory (`.h/y.py`) is kept intact. Over-climbing past the root
  (`from ... import x` at depth 0) still resolves to the root — an inherited oracle quirk that is invalid Python, untouched.
- **F3 — fixed, one new gap (verified by re-running my mutator).** 238 mutants on the new code: **228 killed, 10 survive**. My previous 10 gaps
  (alias pair `:220`, depth cap `:277`×2, `lstrip` `:191`, orphan fallback `:98`, line-1 decorator `:134`, `collapse`×4) are **all killed**. Rulings on the 10, by execution
  (probe output identical to base) or reading: equivalent — message index `core.py:33`; three `return None` that are already `return None` (`core.py:210,239,246`);
  module `col` `python.py:34`; redundant `module.lstrip(".")` `:219` (empty segments are filtered); the redundant sort flag `:266` ×2; and
  `max`↔`min` at `python.py:152` (the no-candidate-after branch is unreachable for a real decorator, which precedes its `def`). **Not equivalent: `python.py:151`**, see P2-1.
  I cannot check the implementer's own "4 equivalent by reasoning"; the survivors I found all have rulings above.
- **F4 — fixed (verified).** `core.import_symbol_id` agrees with my independent innermost-span derivation on all 926 imports (87 with a `symbolId`, 0 mismatches), and the slice test compares to the stored value and counts the 87.
- **F5, F6, F7 — accepted** (not re-probed beyond the suite: F6 survivor no longer listed in my 238-set, F5/F7 are docstring-level).
- **README claim** "scope construction does not depend on the order of the raw facts, a class defined twice included" — **holds** for everything probed above (3 fact sets × 200 orders). Claim is about scope construction only, which matches what was tested.

**P2-1 — minor — the decorator ownership rule's main branch is unpinned.** Mutating `after or cands` to `after and cands` (`python.py:151`)
makes every decorator go to the *first* function sharing a `fullName`. Executed on the live decorated-redefined-class facts: `D#2.x` loses `<get>`,
`D.x.<get>` gets two `property` decorators, `E.In#2.m` and `f.L#2.v` lose their getter — and all 342 tests pass. The new tests exercise redefined classes
but not one whose members *carry decorators in both definitions*. Add a golden: two definitions of a class, each with a `@property` member of the same name (second also with `@x.setter`), asserting
accessors and `decos` per class, under the shuffle parametrization. (`after or cands` is also dead code: it is only evaluated when `after` is non-empty.)

**P2-2 — 5b obligation (not a defect of 5a).** The slice's decorator emission maps `d["target"]` through `full2mid = {m["full"]: m["id"]}`
(`cpg/spikes/vertical-slice/normalize.py:1097-1100`): last writer wins and, with a shared `fullName`, two owners' `o=0` decorators would
land on one `targetId` with colliding `decorator_id(sid, 0)`. `build_scopes` still returns `deco_by_target` keyed by `fullName`; 5b must emit `decorator`
records from each scope's `s.decos` (already owner-resolved), not from that map, and test it on the redefined-class facts under shuffles of `deco`, `m` and `t`.
