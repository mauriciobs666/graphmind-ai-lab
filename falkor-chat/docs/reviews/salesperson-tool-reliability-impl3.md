# `salesperson-tool-reliability` — implementation review, K-057 fix (Pass 3)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-057 (M6)

## Scope & verdict

Reviewed commit `d58125a` ("feat(falkor-chat): K-057 — filter_products inclusive-bound wording
fix, salesperson@v4->v5") against its own spec, `docs/reviews/salesperson-tool-reliability-ml.md`
§11 (§11.5 in particular). In scope: `server/falkorchat/proof_defs.py`, `server/falkorchat/
tools.py`, `server/tests/test_salesperson_scaffold.py`, `server/tests/eval/nlq_golden_set.jsonl`,
`scripts/seed_salesperson.sh`, `scripts/verify_salesperson.sh`, plus a consistency check (not a
substantive review — already checked by `teco`) that `docs/HISTORY.md`/`docs/BACKLOG.md` don't
disclose anything the diff contradicts. This is a code-diff conformance review against a
prior ml diagnosis, not a fresh ml methodology review.

**Verdict: approve.**

**CPG:** considered, not relevant — `cpg_falkorchat` was confirmed stale at coordination-dispatch
time (built 2026-08-26T22:27:22Z, predates every commit in this diff's history including the
`v4`→`v5` bump itself); read `server/falkorchat/proof_defs.py`/`tools.py` and the live `reference`/
`ws:acme` graph state directly instead (`mcp__cypher__query`, `verify_salesperson.sh`).

## Verification performed

- Read §11 (all subsections) and confirmed the diff's two wording additions match §11.5's
  recommendations 1 and 2 in substance and placement (guidance on `tools.py`'s `minPrice`/
  `maxPrice` descriptions, not `systemPrompt`; the non-revision instruction added to
  `systemPrompt`, not the tool description) — `server/falkorchat/tools.py:467-483`,
  `server/falkorchat/proof_defs.py:330-334`.
- Grepped the diff and the resulting `tools.py`/`proof_defs.py` for the reverted second
  iteration's two named phrases ("always pass... category", "check every returned item") — the
  only hit is inside `proof_defs.py`'s own historical-narration docstring (line 260, describing
  what was *tried and reverted*), not in any shipped schema/prompt string. Confirmed: no residue.
- Confirmed `filter_products` is granted only by `SALESPERSON_DEF` (`proof_defs.py:364`) — no
  other def in `proof_defs.py` lists it in `config.tools` — so the `tools.py` description change,
  though code-level and version-independent, has a blast radius of exactly one def family; `triage`/
  `access-request` are unaffected.
- Ran the live ground-truth query for `nlq-40` against `reference`: `MATCH (p:Product) WHERE
  p.category='Peripherals' AND p.price<60 RETURN p.name,p.price` returns exactly `Gaming Mouse Pad
  XL` (19.99), `Wireless Mouse Pro` (29.99), `Webcam HD 1080p` (59.99) — matches the golden-set
  entry's `expected.values` and rationale exactly.
- Ran the full offline suite (`server/.venv/bin/python -m pytest -q`): **2302 passed, 14
  deselected, 0 failed** — matches the commit message and `HISTORY.md`'s claim exactly (not just
  plausible — reproduced).
- Restored shared state per the destructive-run protocol
  (`bootstrap_schema.sh acme` → `seed_demo.sh acme` → `seed_workflows.sh acme` →
  `seed_catalog.sh acme` → `seed_salesperson.sh acme`) and confirmed `verify_workflows.sh acme`,
  `verify_catalog.sh`, `verify_salesperson.sh acme` all report `OK` (`salesperson@v5`, 2/1
  topology unchanged). `reference` now holds exactly one `salesperson` `WorkflowDef` node,
  `version='v5'`. `ws:k057-fix-eval` confirmed torn down (`Graph 'ws:k057-fix-eval' does not
  exist`).
- Spot-checked two of `HISTORY.md`'s Wilson 95% CIs by direct computation (n=20, p=1.0 and
  n=20, p=0.8): both reproduce exactly (`83.9–100%` and `58.4–91.9%`) — the reported statistics
  are genuine computations, not invented numbers, which supports trusting the rest of the
  live-regression figures I could not independently re-run (no LM Studio access in this session).
- Confirmed `test_salesperson_scaffold.py`'s only content-bearing change (the version-pin
  assertion, `SALESPERSON_DEF["version"] == "v5"`) is the file's own stated scope (topology
  validity / no-op republish / end-transition guard, none of which is about prompt wording) — no
  other test in the offline suite needed updating, and none broke, which is consistent with that
  scope rather than a gap.

## Findings

### Minor — `minPrice`'s symmetric "more than $X" guidance ships unverified by the live regression

`server/falkorchat/tools.py:467-474` adds inclusive-bound guidance to `minPrice` symmetric to the
`maxPrice` fix, but §11.5 diagnosed and the n=20 regression exercised only the `maxPrice` "less
than $X" translation error (§11.3's confirmed 56.2%-of-single-call-runs rounding defect). The one
`minPrice`-related anomaly §11.3 actually observed was a different mechanism — a direction/framing
slip (`{"minPrice": 60}` for a "less than" question, in 4/20 runs) — not a boundary-rounding error
on `minPrice`, so this addition is a reasonable-by-analogy extension, not something the shipped
evidence measured. Risk is low (documentation-only, doesn't remove or alter any code path, and
`HISTORY.md`'s own phrasing already says "minPrice/maxPrice parameter descriptions" plural, so it
isn't misrepresented as untouched) — but it is presented in the same breath as the *measured*
`maxPrice` fix without flagging that only one half of the pair was live-verified.

**Suggested improvement:** either fold a "more than $X" live probe into K-060's follow-up n (cheap
to add to the same harness), or add one clause to `proof_defs.py`'s own `v5` docstring noting the
`minPrice` wording is untested-by-analogy — a one-line disclosure, not a blocker to shipping.

## What's solid

- Faithful, minimal implementation of exactly §11.5's two recommendations, correctly placed
  (parameter-schema guidance on the tool, not `systemPrompt`; turn-level guidance on `systemPrompt`,
  not the tool) and correctly reasoned in the commit's own docstrings about why that split makes
  sense (schema resent fresh every turn vs. `systemPrompt` bloat).
- Version-bump discipline is followed correctly and consistently across every file that needed it:
  `proof_defs.py`'s `version` field, its own docstring history, `seed_salesperson.sh`'s doc comment
  *and* its executable default (`SALESPERSON_DEF_VERSION="${...:-v5}"`), `verify_salesperson.sh`'s
  doc comment *and* executable default, and `test_salesperson_scaffold.py`'s assertion — no
  instance of the comment-updated-but-default-stale drift that caused the earlier shared-state
  collision (`docs/reviews/salesperson-tool-reliability-impl2.md` Appendix). `config.model` is
  correctly carried forward unchanged (create-only discipline).
- No residue anywhere of the reverted second-iteration wording (`category`-parameter nudge,
  synthesis-time safety-net sentence) — verified by direct grep of the shipped code, not just
  trusting the commit message.
- `nlq-40`'s ground truth and rationale are accurate against the live catalog, and the entry is
  structurally identical to its sibling `compound-filter` entries (confirmed via the golden-set
  integrity test suite, 255 passed).
- `HISTORY.md`'s disclosed K-060 third-mechanism gap is mechanistically consistent with the code
  change: nothing in this diff touches the no-`category` mixed-result synthesis path, so its
  persistence at the same order of magnitude is exactly what should be expected, not a red flag.
- Shared-state hygiene: throwaway `ws:k057-fix-eval` confirmed torn down; `reference`/`ws:acme`
  verified in sync post-fix, independent of my own destructive-suite run and its own restore.

## Open questions

None — no finding here needs the caller's input to resolve; the one minor finding has a
self-contained suggested fix for whoever picks it up (optionally, folded into K-060 rather than a
new item).
