# ccg P1-05 step 5d: how many Python call edges does name-link demotion hit, and is a self/cls rule worth P1?

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1)

**Decision served.** `docs/reviews/connected-code-graph-p1-5d-49.md` M3 and open question 3: accept 5d's demotion of every Python `self.m()` to `possible/name-match` with no P1 enclosing-class rule (and with SCIP possibly only listed, P1-15)? Method note, figures as observed in this session's commands; no refresh, no FalkorDB, no `ccg ask`, no held-out data, no tree-mutating git. Scratch scripts are in the session scratchpad, not committed.

## Recommendation

1. **Pull a `self`-only enclosing-class rule into P1** (stated in 5d as a rule over the symbol table, `cls.` excluded). It costs little against 553 sites and its precision is measurable; it also keeps the 8 Python flow dev items from changing status. Rule: a `self.m()` call inside method of class `C`, where `C` defines `m` itself and no class in the unit that names `C` as a base redefines `m`, becomes `confirmed` with target `C.m`. Anything else stays `possible/name-match`. Measured eligibility: 524 of 557 `self.m()` sites (94%). It is an architect/coder choice whether it lives in `assemble_bundle` (needs the class symbol table there) or in the SCIP-independent link stage; I did not design that.
2. **Block 5d's literal rule on bare identifiers.** If the rule is implemented as "DYNAMIC_DISPATCH and the receiver's `typeFullName` unknown" and the receiver of a bare call `helper(x)` (an `Identifier`) is read, **14,618 of 20,329** Python in-repo bare calls (72%) have an `ANY` receiver type and would be demoted, on top of the 3,172 attribute calls. The plan wording ("receiver base ... absent for a call with no receiver") does not say what happens to them. This is the analyst's M1 point, now with a number: the rule must be member-access-only for Python, and a test must pin a bare imported-function call as `confirmed`.
3. If neither is wanted, the honest P1 statement is: Python `self.m()` is `possible/name-match`, 553 sites in this repo, 498 in non-test code.

## Q1. Number of Python call sites 5d would demote (graphmind-ai-lab, bundles built at 8377c32)

Data: stored Joern facts `cpg/.ccg-work/joern/*/cpg/*.facts.jsonl` (10 Python units), their `cpg.bin`, and the retained bundles `cpg/.ccg-work/bundles/graphmind-ai-lab/` (1138 bundles, one commit). Bundles pre-date 5d and carry no dispatch or receiver field; facts carry `dt` and `in` but no `rt`. Four independent counts:

| Method | What it counts | Count |
|---|---|---|
| M-A Joern `cpg.bin`, read-only re-query (receiver base `typeFullName`, descending `fieldAccess` to the base identifier) | in-repo-linked DYNAMIC_DISPATCH attribute calls whose base type is `ANY`: **the 5d rule as written, member-access shape** | **3,172** of 7,148 attribute calls (4,976 typed stay confirmed) |
| M-B bundle `resolved/confirmed` Python `call` sites joined to facts `(path,line,name)` and split by code shape | attribute calls of any base type (upper bound if a type test did nothing) | **7,395** of 27,804 confirmed (26.6%); 20,407 bare; 2 unjoined |
| M-C facts only, same split by regex | same quantity before the bundle filter | 8,307 = 7,395-class + 658 synthetic `cls.__init__(__newInstance)` + 254 other (tmp lowering and unmatched); consistent with M-B |
| M-D `ast` over source at 8377c32 (attribute calls whose attr name equals some unit-defined function name, any receiver) | upper bound ignoring Joern's resolution | 12,507 (includes names Joern did not link, e.g. `str.replace`) |

**Spread: 3,172 (type-aware, the quantity 5d would act on) to 7,395 (shape-only) to 12,507 (name-only).** The first is the best estimate; the others bound it. The bundle's confirmed Python edges are 27,804; demotion would remove **11.4%** of them (3,172/27,804). Restricted to non-test paths (a `test` substring in the path, which also catches e.g. `latest`; a coarse split): 882 of 4,272 confirmed non-test edges, **20.6%**; test files carry 2,290 of the 3,172.

Composition of the 3,172 (M-A, labels for non-self bases from the AST join; `ident:` rows are the 2,617 identifier bases with `ANY` type, the tmp row is Joern's lowering of `a.b.m()` / `f().m()` through a `tmpN` base):

| Base | Demoted | non-test | Share |
|---|---|---|---|
| `self.m()` direct | 553 | 498 | 17.4% |
| `cls.m()` direct | **0** (the 658 `cls.__init__(__newInstance)` are synthetic constructor lowering, not source; a further 476 `tmpN.append` are comprehension lowering) | 0 | 0% |
| `self.x.m()` / `f().m()` chains (tmp base) | 255 | 166 | 8.0% |
| local assigned from a repo-class call (`x = C()`; the plan expected these typed) | 633 | 16 | 20.0% |
| local, other | 978 | 67 | 30.8% |
| unannotated parameter | 489 | 35 | 15.4% |
| annotated parameter | 127 | 66 | 4.0% |
| loop/with/except locals, unresolved names, module globals | 135 | 34 | 4.3% |

Bare-identifier calls: **0 demoted** under the member-access rule (20,329 stay confirmed, 74% of edges); see recommendation 2 for the literal reading. Direct `self.m()` agrees across counts: 553 (M-A), 551 (M-B and M-C), 554 (AST), 557 (the class-aware AST in Q2). The share the questioner asked for, "bare identifiers", is 0 demoted under the plan's intent and 14,618 under its literal wording.

Surprise worth a plan edit: 633 constructor-assigned locals have an unknown receiver type in this CPG (and 241 are typed), so 5d's "`a = A(); a.dup()` is typed" holds only sometimes. The probe table should include `x = C()` in a function with other assignments, and the P1-49 wrapper case via `self.`.

## Q2. Effect on P1 measures

- **What changes.** Demoted sites become `unresolved/possible/name-match`, `targets: []`; the read-time `CALLS_NAME` hop still lists candidates. Nothing is removed from an impact list; a hop that was `confirmed/resolved-call` becomes `possible`. For a Python impact list through a `self` call that is a confirmed-to-possible status change on 553 sites; the recall of "listed at any status" is unchanged, the **confirmed-only** precision/recall and the counts of confirmed edges fall by up to the figures above.
- **P1-39 flow-fallback measurement (dev data `cpg/eval/dev/ac5-flow.jsonl`, 28 items; py->py 8).** All 8 py->py chains are built from `self.resolve_llm(...)` / `self._models.llm(...)` / `self.resolve(...)` / `self._warn_once(...)` hops (I read the cited source lines). Under 5d these become `possible` call edges, so the share of flow items "call-level reach alone lists" changes with the status the measurement counts (any status: unchanged; confirmed only: down). The 8 items share one anchor family (`StaticModelGateway.resolve_llm`), so they are not 8 independent observations; this dev set cannot show a difference either way. The measurement's status basis must be pre-registered as "reach at any status" or this change is a silent re-definition (`ccg-p1-flow-fallback-v1`).
- **AC-5 dev strata that read Python CALLS.** http, mcp, wiring and scripts chains are `link`/`config`/`caller` hops (scripts `caller` hops are shell); the only Python-call-dependent dev items are the 8 flow items above and 1 ts->py flow item. AC-24 dev (14 items) has no path fields I could classify. AC-1/AC-25 planted fixtures `cpg/fixtures/system1`: 0 `self`/`cls` calls to fixture-defined names (51 other-receiver, 23 bare, counted by `ast`, working tree). So **the fixtures cannot detect the self demotion in either direction**: they will stay green whichever way the policy goes, and the only evidence about it is the real-repo count above.
- **P1-49 role lifting (analyst M2).** A client wrapper reached through `self._get(path)` is among the 553; lifting would stop there under 5d without the self rule. Recommendation 1 removes that exposure for 94% of such sites.
- **Self rule precision (Q2 of the recommendation).** AST over the same files: of 557 `self.m()` sites, 524 are `m` defined by the enclosing class itself with no subclass override in the unit (375 where the name is unique in the unit, 149 where Joern links several same-name methods and the rule picks the right one); 1 is own-but-overridden (stays possible); 29 are not own and have no base class (closure, callable attribute, dynamic; stay possible); 3 inherited. Inheritance is the rule's only systematic miss: it is conservative (stays possible), never mints a wrong confirmed edge unless a subclass outside the unit overrides `m` (not measurable here).

## Evaluation of the rule, if adopted

Metric: of `self.m()` sites, the share resolved `confirmed` whose target is the enclosing class's `m`; and precision of those targets. Data: the 557 sites above (development data, same repo), labels by `ast`, plus a hand audit of 30 randomly drawn (fixed seed) `confirmed` targets for subclass/mixin cases. Acceptance: audited precision lower Wilson bound >= 0.90 (30/30 gives about 0.886, so draw 40 or accept 0.88; state which before drawing); eligibility reported as measured, not thresholded. A test pins `self.m()` with a subclass override as `possible`.

## Could not measure

- Post-5d demotion actually produced by the implemented rule: `rt` does not exist in the stored facts; M-A emulates it from Joern's own `typeFullName` at 8377c32 and may differ from the implementation (`extract.sc` reads the receiver base; I descended `fieldAccess` and read the first argument, with `tmpN` as its own identifier).
- Whether P1-15 folds `scip-python` (the fold-or-list result is not in the inputs I read); the whole argument for recommendation 1 is that it does not depend on it. Any SCIP upgrade would apply to the other 2,619 attribute sites regardless.
- Test versus non-test split is a path-substring heuristic. The `JS` units and `.sh`/config units were not counted. Held-out items and cross-language calls were not read.
- Subclass overrides outside the unit; dynamic `getattr`/mixin dispatch.
