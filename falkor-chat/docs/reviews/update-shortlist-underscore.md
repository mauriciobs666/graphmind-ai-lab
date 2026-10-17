# Review: K-075 — underscore-only title token in `find_update_shortlist`

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-075

## Scope & verdict

**Reviewed:** the uncommitted working-tree change against `4f978ddf`, `git diff 4f978ddf -- falkor-chat/`.
That covers `server/falkorchat/repository.py` (`_escape_fuzzy_token`, the `find_update_shortlist`
docstring), the four new tests in `server/tests/test_repository.py`, `docs/QUERIES.md` §14.6, the
top `docs/HISTORY.md` entry, and `docs/BACKLOG.md`. I reviewed it against two baselines. The
contract is the `### K-075` body in `git show 4f978ddf:falkor-chat/docs/BACKLOG.md`. The
done-condition is in `docs/plans/update-shortlist-underscore-coordination.md`. The precedent is
`fusion._fuzzy_query` and `docs/reviews/fusion-fuzzy-escape.md`.

**Verdict: approve with suggestions.** There are no blockers and no majors. The fix is one line,
it is correct, and it matches the K-068 precedent. All four of the implementer's premises held up
when I executed them. The tests kill every mutant I and the implementer tried, including a
plausible alternative fix that swallows the exception. The two remaining findings are wording
problems in the docs.

CPG: considered, not relevant — a one-function, call-site-local change, with the single caller and the background path traced by reading; `cpg_falkorchat` is also stale per the coordination doc.

### Premises, each checked by execution

I ran my own probe on a throwaway graph, `ws:probe-k075-review`, deleted afterwards. It used the
`Document.title` full-text DDL from `scripts/bootstrap_schema.sh:373`. The script and its output
are in the session scratchpad (`k075-review/probe.py`, `probe.out`).

| Premise | Result | Evidence |
|---|---|---|
| (a) Only a term that is exactly `_` errors. `%__%` parses but ANDs the title signal to nothing. | **Confirmed** | `%_%` → `Syntax error at offset 2`. `%__%` → ok, `[]`. `%Report% %__% %draft%` → ok, `[]`. `%Report% %draft%` → `['d1']`. |
| (b) An empty full-text query returns 0 rows rather than erroring. | **Confirmed** | `''` → ok, `[]`. `'   '` → ok, `[]`. |
| (c) Every character the helper keeps (`\w`) parses inside a term. | **Confirmed**, with a wider net than the implementer's | (1) Python `\w` holds exactly one connector-punctuation character, ASCII `_`; the other nine `Pc` code points (`‿`, `＿`, …) and every `Mn`/`Mc` mark are not `\w`, so `strip("_")` covers the whole class. (2) I ran a coverage probe over 17 symbols (`_ a - * ( . 1 é \ ' " \| ~ @ $ { %`) at every length from 1 to 3. That is 5,219 tokens, each placed after `Report` and passed through the **fixed** builder, with 0 errors. (3) Every ASCII code point 0–127, both inside a word and as a whole token, gave 0 errors. (4) The implementer's `probe.out` covers 52,838 non-ASCII `\w` code points (all categories `L*`/`N*`), with 0 errors. |
| (d) The caller already skips empty tokens, and skips the title query when no token is left. | **Confirmed** by reading `repository.py:1997-2013` and by execution | `Report _ draft` → `%Report% %draft%` → `['d1']`. `_` and `*_*` → no query issued. |

## Findings

### F-1 (minor): HISTORY's red-first figure covers only part of the new tests

`docs/HISTORY.md` K-075 entry, **Proof** bullet: "The new tests went red first (6 syntax errors, 2
empty results)." All the other figures in that bullet are true. I re-ran the 36 passed, and the
18/7/4/9 mutant counts match `mutants.out`. The red-first figure does not describe the full new
selection, though. With the pre-fix helper (mutant a, "no filter"), the four new tests fail **18**
cases: 9 syntax errors, 2 empty title results, the `__` skip assertion, and the 6 pure-helper
assertions. "6 syntax errors, 2 empty results" matches only the first and third tests: the band
test's 5 errors plus the title-signal test's 1 error and 2 empties. Most likely those two were
written red first, and the other two came later.

**Why it matters:** `HISTORY.md` is the lookup record, and it should state only figures that are
true as written. **Fix (artifact: `docs/HISTORY.md`):** either scope the sentence ("the band and
title-signal tests went red first (6 syntax errors, 2 empty results)") or cite the full-selection
figure: "against the pre-fix helper the new tests fail 18".

### F-2 (nit): Two doc wording problems

- `docs/HISTORY.md`, **Characterized live** bullet. The lead-in says every point was characterized
  "on a throwaway `ws:probe-k075` graph", but the third sub-bullet says "on `ws:test`". The claim
  is true; I reproduced it on my own throwaway graph. The location is inconsistent, though, and the
  implementer's `probe.py` does not contain the empty-query check. **Fix:** move that sub-bullet
  out of the lead-in's scope, or drop "on `ws:test`".
- `docs/BACKLOG.md:728`. The rewritten K-064 "Related" line runs to about 160 characters, while the
  rest of the paragraph wraps at about 100. **Fix:** reflow it.
- `repository.py:1957-1961`. The `find_update_shortlist` docstring nests an em-dash aside inside a
  parenthetical: "(no degenerate query is issued — on this build … — the same posture …)". Its
  content is accurate, and (b) was verified. This is take-or-leave.

### Checked and found sound (no finding)

- **Tests pin the behaviour.** The band test really needs the band match. `d1`'s title is
  `Unrelated`, so for every parametrized title the only way `d1` can arrive is the LSH band, and
  the assertion is exact equality, `== ["d1"]`. I checked whether the tests would fail if the
  exception were swallowed some other way. For that I ran **reviewer mutant M1**: the helper
  reverted, plus a `try/except` around the title query. The band test alone would pass under M1,
  which is the correct behaviour for band survival. The selection still kills it: **15 failed**.
  The title-signal test fails because the title match is lost. The skip test fails because the
  query is still issued. The pure test fails too. **Mutant M2**, which drops underscore-only
  tokens only when the raw token had no punctuation, is killed with **9 failed**. I copied
  `repository.py` aside for each mutant and restored it afterwards, confirmed with `cmp`.
  `--setup-plan` showed 0 `wf_repo` uses, and the selection passed 36 before and after.
- **Contract conformance.** The K-075 test strategy asks for a live call, the `repo` fixture,
  never `wf_repo`, a lone-`_` title and a band-matching candidate, with no exception and the band
  candidate returned. Every point is met. The scope boundary is also respected: titles still strip
  per token, and index-like splitting is left to K-064.
- **Docs match the code.** The `_escape_fuzzy_token` docstring, the `find_update_shortlist`
  docstring, the closing sentence of `QUERIES.md` §14.6 and the BACKLOG K-064 "Related" paragraph
  all describe the shipped behaviour. The `fusion.py:66` mention of the title path, which says it
  strips and glues, is still accurate.
- **No stale K-075 cross-reference.** I grepped the whole component for `K-075`, `lone _`, `%_%`,
  `still crash` and `_escape_fuzzy_token`, unfiltered by extension. What remains is either
  historical and dated (`HISTORY.md` 2026-10-17 K-068 entry, `reviews/fusion-fuzzy-escape.md`,
  `plans/fusion-fuzzy-escape-coordination.md`) or current. K-076's body does not mention the title
  path. The `scripts/test_queries.sh:1904-1920` assertion uses `Report (Draft)`, which the change
  does not affect.
- **`background._safe_detect_update` path (static trace, not executed).** `background.py:270`
  reads `document.get("title") or ""` verbatim, and `:274` calls
  `repo.find_update_shortlist(ws, bands=bands, title=title)`. For `Report _ draft`, the fixed
  helper maps `_` to `""`, the caller's `if escaped` drops it, and the query becomes
  `%Report% %draft%`. My probe executed that exact query string without error. The method then
  returns the band and title union to the Jaccard ranking, and the `except` at the end of
  `_safe_detect_update` is not reached.

## What's solid

- The fix is minimal and puts the rule in the one place both cases flow through. `strip("_")`
  matches `fusion._fuzzy_query`'s `tok.strip("_")` word for word, so the two paths share one rule.
- Dropping `__`/`___` as well, rather than only the literal `_`, is the right call. Under the
  implicit AND they parse but silently zero the title signal, and mutant b shows the tests pin it.
- The new docstring removed a false claim, that an empty query is "a caller error", and replaced
  it with a probed fact.
- The implementer's evidence is reproducible, and it cleaned up after itself. `ws:probe-k075` is
  gone.

## Open questions

None.
