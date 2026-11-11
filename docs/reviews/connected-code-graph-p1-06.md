# C-1001 P1-06 — `Cond.code` literal masking (F-9): gate review

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-06)

**Verdict: needs changes** (one blocker, three majors, all small). Scope: `cpg/extractors/joern/normalize/mask.py` (new), the `assemble_bundle.py` diff (call site, `NORMALIZER_VERSION = "0.3"`), `cpg/tests/test_mask_cond.py` (new), the `tests/slice/test_normalize_equivalence.py` diff (`P1_06_MASKED`, delta `p1-06-mask-cond-code`), against `docs/plans/connected-code-graph-p1-extraction.md` § P1-06 and 5c's delta mechanism. `extract.sc` and other units' files not looked at.

CPG: considered, not relevant — a pure Python function plus one call site, fully read; the claims were settled by executing it.

Executed vs read: **executed** — the property probes, the mutants, the stored-data counts, `tests/test_mask_cond.py` in the mutant copies, the slice file once (`20 passed, 1 deselected`, 113 s). **Read only** — the delegate's 21-mutant table and the 852-passed unit-set figure (the caller re-ran it).

## Findings

### B1 (blocker) — the property "stored `Cond.code` never holds a flagged string" is false at the 200-char cut

Mask-before-cut leaves a passing literal that straddles char 200 as an unterminated prefix, and a **prefix can fail where the whole passes** (`name_fails` whole-token words are not prefix-monotone, and T3 needs an all-hex run that a longer run breaks). Executed (`scratchpad/probe.py`, then `prop.py`, via `collapse(mask_cond_code(code), 200)` as `assemble_bundle.py:254` does):

| literal (passes whole) | fragment stored | fails |
|---|---|---|
| `pinned please` | `"pin` | `name_fails` (`pin`) |
| `seeded value` / `salted caramel` / `patient zero` / `vaulted` / `sshd running` | `"seed` / `"salt` / `"pat` / `"vault` / `"ssh` | `name_fails` |
| `deadbeefcafexyz tail` | `"deadbeefcafe` | `screen_fails` T3 |

Randomized (seed 7, 2,335 passing literals cut at a random point): **108 stored fragments are flagged** under the current order, 0 under the fix below. The leak is limited to a prefix of a literal that passed whole, so exposure is small. But the unit's stated done-property (plan P1-06 Behaviours) is false, the 40x60 property test only covers whole literals, and this is the F-9 control that gates every non-proving build. The delegate's residual note said "not tested/pinned"; it is a stated behaviour and needs a test.

**Order.** Cut-then-mask is unsafe: a flagged literal cut to a passing prefix (10 of 32 hex chars) would leak. Mask, cut, **mask again** is safe: `mask_cond_code(collapse(mask_cond_code(code), 200))`. The second pass sees the truncated tail as an unterminated literal and judges its prefix; terminated literals re-scan identically. Executed: 0 leaks over the 2,335 cases and `mask(mask(x)) == mask(x)` held. Applying a second pass to all 43 `P1_06_MASKED` values changes none of them, so the delta and the two long conds stand.

**Id stability survives.** The sort key keeps the raw code (`assemble_bundle.py` untouched there), and the extra pass touches only the stored string, so ordinals and point ids do not move. Fix, with its test placed *before* the call-site change: a test (in `test_mask_cond.py`, driving `assemble`) that cuts a passing `pinned please`-style literal at each offset and asserts the stored tail passes `screen_fails`/`name_fails`. The probe's axes are the cut offset, the literal's rule family (whole-token word, hex run, T5 prefix) and quote kind; use a coverage probe over those, not a list of literals.

### M1 (major) — a quote-parity flip fails open: the whole secret is stored

`mask.py` documents regex literals and comments as "not handled", but the failure is not an over/under-mask of the odd text; it inverts which text counts as a literal. Executed:
`/"/.test(a) && b === "<32-hex>"` and `it's fine and k == '<32-hex>'` come back **unchanged**: the regex quote and the apostrophe open a fake literal, and the real one is then scanned as non-literal text. TS `/"/` in a condition is plausible. Suggestion: add a narrow backstop on the masked output's non-literal text, limited to T3/T4 (a hex run of 12 or more, or a UUID), and pin the regex case with a planted-secret test. Do not use `screen_fails` on the whole output: executed, it flags 1,308 of the 4,034 stored conds (T2 on long identifiers such as `iteratorNonEmptyOrException`). I did not run the T3/T4 backstop; its false-positive rate on the stored data is for the implementer to measure first. If the owner rules the residual acceptable, the plan's property sentence must say "per well-formed literal".

### M2 (major) — a mutant on the `screen_fails` argument survives every test

Mutant B (fresh copy under `cpg/.ccg-work/gate-p1-06/B`): `screen_fails(body[::-1])`. `tests/test_mask_cond.py` 73 passed (base 73 passed), and the slice delta also stays green because the stored data has **0** literals that `screen_fails` flags (every one of the 46 masks is `name_fails`). The secrets used (32-hex, UUID, `ghp_` token) are each caught by a second rule on the reversed text (T3, T6, T1), so the call is never pinned. Add the rule-isolated literals: a T5-only (`"sk-abcdefghijklmnopq"`, executed: masks in base, not in B), a T4-only and a T2-only. Corollary: until then the slice delta is no evidence for the screen half of `_flagged`.

### M3 (major) — the unit-level mutation table cannot be reproduced from `tests/test_mask_cond.py` alone, and the wiring is checked only by the 3 helper-dependent tests

Minor mechanics, but the load-bearing check for the screen path is the unit file. Fix M2 and the claim "21 killed" holds for the screen arguments. (Mutants A, C, D, E, executed, are killed: A by the slice delta only — the 7 `empty key`-style conds, not by the unit tests, which pass 73/73 under A; C by the property tests; D by `test_template_nested_template_and_braces`; E by `test_template_literal_with_secret_text_is_masked_whole`. So A's argument (`name_fails` receives only the first word) is also unpinned at unit level: add a multi-word literal whose name word is not first, e.g. `"empty key"`.)

### m1 (minor) — plan file list not followed

The plan lists `cpg/tests/test_normalize_assemble.py` for the planted-32-hex wiring test; the delegate put it in `test_mask_cond.py` (3 wiring tests). Acceptable and cheaper than a cross-file edit; say so in the HISTORY entry.

## Answers to the brief

**(1) Conformance.** Executed on a probe set: failing literals become `"…"`; passing literals verbatim; no-literal conds byte-identical; Python triple quotes, escapes, `b"…"` prefixes (prefix kept outside the mask), TS template literals with `${…}` recursion and nested braces all behave as specified (mutants D, E killed). The cut finding is B1; the parity flip is M1.

**(2) The 43 changed conds.** `P1_06_MASKED` is a literal dict of 43 id → code (`tests/slice/test_normalize_equivalence.py` diff), not built from the masker; the delta takes `want` from it and asserts `r["code"] != want`. My independent recomputation over the stored bundles gives exactly the same 41 + 2 ids, and every value equals `mask_cond_code(stored code)` except the two long conds, whose expected values rest on the pre-cut raw text (283 and 218 chars) — consistent with mask-before-cut, and unchanged by my B1 fix. Real check: planted mutant (an extra mask of `"stderr"`, one unlisted cond, injected into `assemble_bundle.mask_cond_code` via a scratch pytest plugin) → `test_whole_bundle_is_byte_identical_to_the_stored_one[fcs]` **failed**, killed. The loosened 5c assertion is still a real check: it now pins per-name touched counts (3/0 and 41/2), a delta that touches a different count fails. The remaining weakness is only that I cannot prove the table was hand-typed; its text carries hand comments and agrees with an independent recomputation, so I treat it as sound.

**(3) Lossiness.** `name_fails` runs over the whole literal body, as plan and spec both state (the plan's "fails `screen_fails` or `name_fails`" per literal). On the stored data: **conds changed 43 of 4,034 (1.1%)**; **literals changed ≈46** (44 + 2, counted as `"…"` in the output; top: `key` x16, `empty key` x6, `seedKind` x5, `startKeys` x3). Zero from `screen_fails`. `Cond.code` is display text only (spec3 §5.5; `pointId` is `symbolId` + ordinal, sort key raw; the graph plan's `has_out:Cond` just returns `p.code`), so no match key, flow-fallback or id is hurt; answer rendering shows `"…"` where a dictionary key was. Recommendation for the architect: **keep it and do not block**: fail-closed, 1% of conds, display-only. If the over-masking bothers answer quality, a segment-wise judgement of identifier-shaped literals is a spec change with its own leak surface; defer it to P2 with the count above as the evidence.

**(4) Mutants (fresh copies, `unit-file only` unless stated).** A `name_fails(first word)`: survives unit, killed by slice. B `screen_fails(reversed)`: survives everything (M2). C static text judged minus first char: killed. D template brace depth ignored: killed. E `${}` expression not recursed (`in_expr=False`): killed. Delta mutant: killed. Base copy 73 passed.

**(5) Version and ordering.** `NORMALIZER_VERSION = "0.3"`; delta list is `drop-lambda-decorators` "0.2" then `p1-06-mask-cond-code` "0.3", matching plan order; the last-literal-equals-constant test (`test_normalize_equivalence.py:373-378`) is green in the 20-passed run. Counts untouched (a cond edit only). Fine.

## What's solid
Linear single pass with a recursion cap; no regex backtracking exposure; raw-code sort key preserved; the delta table is independent of the code and the byte test genuinely fails on an unlisted change; wiring touches one line; slice file 20 passed.

## Open questions
- B1/M1: does the owner accept the parity-flip residual (M1) with a narrowed property sentence, or want the T3/T4 backstop? (Route: `architect` for the sentence, `tdd-engineer` for the code.)

## Routing
`tdd-engineer`: B1 (mask-cut-mask plus test first), M2/M3 tests, optionally M1. `architect`: lossiness ruling (keep) and the M1 wording. Re-gate by `analyst` after the fixes (B1 is a one-line change; the delta table needs no edit).

## Pass 2

**Verdict: approve with suggestions.** Re-gate of the `GATE FIXES` in `cpg/.ccg-work/p1-06/REPORT.md`: the call site is now `mask_cond_code(collapse(mask_cond_code(c["code"]), 200))`, and `mask.py` gained `_backstop` (UUID or all-hex alphanumeric run of 12 or more becomes `…`, applied to the whole output). Executed: probes `re.py`, three mutants in fresh copies (`cpg/.ccg-work/gate-p1-06/{base,P,Q,R}`, rebuilt from the new files; the delegate's overwrite of the old `A..E` scratch copies is confirmed harmless, I deleted and rebuilt the directory), `test_mask_cond.py` 100 passed in the base copy. Read: the 879 and slice 20-passed figures (the delegate's/coordinator's).

- **B1 fixed.** 3,939 random passing literals (quotes `"`, `'`, `` ` ``, escapes, `${}`, cut at a random offset): 0 flagged fragments, 0 non-idempotent. `pinned please` cut at 3 gives `"…"`, at 4 `"pinn` (kept, passes); `deadbeefcafexyz tail` cut at 12 gives `"…"`.
- **M1 fixed, residual honest.** `/"/.test(a) && b === "<32-hex>"` and `it's fine and k == '<32-hex>'` now give `"…"` for the hex. A 64-hex after a parity flip is masked too. The `mask.py` docstring (lines 10-11) says exactly "any other secret shape there is not masked (the guarantee is per well-formed literal)", which matches the architect's wording. 0 false positives on the 4,034 stored conds: all but 2 agree with the table or their unchanged stored code; the 2 are the two long conds (`a2a06f12…`, `acff4327…`), which the hand table derives from the raw 283/218-char text, so they cannot be compared with the already-cut stored code (the slice run is their check; same as Pass 1).
  - **m2 (minor, executed):** the backstop also masks any digit-only run of 12 or more (`ts > 1700000000000` gives `ts > …`) and a hex-shaped identifier (`a = deadbeef1234` gives `a = …`; `foo_deadbeefcafe` gives `foo_…`). That equals the spec T3 rule applied to code, 0 in the stored data, so acceptable; note it in the HISTORY entry. `0x1234567890ab` is untouched.
- **M2 fixed.** Delegate's reversed-screen mutant killed (2 failed); T5-only, T2-only cases present. I did not re-run it (read, not executed).
- **M3 fixed.** First-word mutant killed by the unit file (18 failed per REPORT; not re-run by me).
- **Delta table unchanged.** `git diff` of `tests/slice/test_normalize_equivalence.py` still shows 43 added table rows (41 fcs, 2 sp) and the same 74+/1- stat; hand-written, no new generation path.
- **New mutants (unit file only):**
  - **R** (`all` hex check becomes `any`): killed (30 failed).
  - **Q** (backstop boundary: run of 12 to 39 only, no upper bound): **survives** (100 passed). Executed consequence: `/"/ and k == "<64-hex>"` stores the whole 64-hex SHA-256 under Q, masked in base. **M4 (major-low):** add a 64-hex (and a 40-hex) parity-flip test.
  - **P** (silent argument: `screen_fails(body[:64])`): **survives** (100 passed). Not shown to leak on the probe I tried (the T1/T2 runs I built started inside 64); the argument is simply unpinned. **m3 (minor):** add a literal whose only screen failure (a 24-letter run or a T5 prefix after `=`) lies beyond char 64.

No new blocker. Routing: `tdd-engineer` for M4 and m3 (tests only, no code change expected); the HISTORY entry notes m1 and m2. No re-gate needed unless the tests expose a code change.
