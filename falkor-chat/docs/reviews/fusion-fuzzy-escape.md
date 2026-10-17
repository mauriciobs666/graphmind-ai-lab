# Review: K-068 entity-name fuzzy-query fix (`fusion._fuzzy_query`)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-068

## Scope & verdict

**Reviewed:** the uncommitted diff `git diff 3d91f750 -- falkor-chat/`, covering
`server/falkorchat/fusion.py`, `server/falkorchat/repository.py` (docstrings only),
`server/tests/test_fusion.py`, `docs/QUERIES.md` §14.6, and the `HISTORY.md`/`BACKLOG.md` edits.
The baseline is K-068 as written at `3d91f750`, plus the implementer's scratchpad evidence
(`…/scratchpad/k068/`). The coordination doc is out of scope.

**What I ran:**
- `tests/test_fusion.py`: 45 passed. `--setup-plan` shows no `wf_repo`.
- Two throwaway graphs (`ws:probe-k068-review`, `ws:probe-k068-cov`/`-cov2`). I created and
  deleted each one, and confirmed the deletion.
- Read-only `ro_query` sweeps over every distinct `Entity` name in `ws:agent-team`.
- A read-only title-path probe against `ws:agent-team`.
- Two `pytest -p <plugin>` runs that monkeypatch `fusion._fuzzy_query` in memory. No file was
  edited.

My scripts and outputs are in
`/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/cc651677-ac22-4370-9988-76881753007e/scratchpad/k068-review/`.

**Verdict: needs changes.** The fix removes the crash, and the split approach is right for ASCII
punctuation. But `[^\w]+` also splits on non-ASCII punctuation and symbols, and the index does
**not** separate on those. Real names that matched before the fix now miss (F-1). The docs'
central claim that splitting "yields exactly the indexed terms" is false for the same reason. The
fix is one regex, and I verified it below. Separately, the title-path follow-up is not filed
anywhere (F-2).

CPG: considered, not relevant — the change is a one-function query-string builder whose behaviour is decided by the live RediSearch tokenizer, which only execution against FalkorDB can show; call-graph impact (`find_fuzzy_candidates` → `Repository.find_fuzzy_candidates`) was read directly.

## Findings

### F-1 (major) — splitting on every non-word character breaks recall for non-ASCII punctuation the index keeps inside a term

**Evidence.**

- **Throwaway-graph probe (`probe_index.py`).** Stored names `alpha—beta`, `gamma–delta`,
  `don’t panic`, `foo→bar`, `2×3 grid`, `deg°C`, `a…b` and `minus−sign` were each **found** by
  the old whitespace build, for example `%alpha—beta%`. The new build, for example
  `%alpha% %beta%`, finds **none** of them.
- **Coverage probe (`coverage.py`), over 134 characters.** The set was all ASCII punctuation, up
  to 6 characters from every non-word Unicode category, Unicode whitespace, and emoji. The index
  separates on **ASCII punctuation (except `_` and `\`) and ASCII whitespace only**.
  `_fuzzy_query` misses the stored `aa<c>bb` for 102 of the 134 characters.
- **Real data, read-only (`real_regress.py`).** Among the 5134 distinct `ws:agent-team` names whose
  query changed and whose old query parsed, 8 **lost** their identical-name hit and 37 gained one.
  The 8 losses are all non-ASCII: `A1–A15`, `B1–B15`, `C1–C6`, `Passes 5–9`,
  `review passes 15–16`, `producer→store→load→consumer round trip`, `A∩B`, and `∅`, which now
  produces an empty query.

**Why it matters.** This is a regression on names that worked before the fix. En-dashes in ranges
are common in LLM-extracted text. The replay counted only errors, not recall (`replay.py`, "clean"
means no exception), so it could not see this. The claim appears in three places, and it is false
in each:
- `QUERIES.md` §14.6: "yields exactly the indexed terms";
- the `fusion._fuzzy_query` docstring: "splitting reproduces the indexed terms";
- the HISTORY entry.

**Suggested improvement.** Split on the measured separator set instead of `[^\w]`. That set is
ASCII whitespace plus ASCII punctuation minus `_` and `\`. Delete `\` before splitting, because the
tokenizer treats it as an escape. Keep the `tok.strip("_")` filter. The tested form is in
`k068-review/k068prop.py` (verified):
- coverage probe: the only remaining miss is the emoji, which the old build also missed;
- all 5134 real names: 5128 hit, **0 miss**, 6 empty, against 5120/7/7 for the current build
  (`real_prop.out`);
- `test_fusion.py`: 44/45 pass. The only failure is the `"…"` all-punctuation pin, which
  encodes the current split and should flip (F-4).

**Guard placement.** Add a live char-class coverage test **as a red step before the regex
change**. Seed `aa<c>bb` for one sample per Unicode non-word category, plus all ASCII punctuation,
and assert that `find_fuzzy_candidates` finds it. Do not hand over a fixed list of shapes.
Emoji gets an explicit, reasoned `xfail`.

### F-2 (major) — title-path `%_%` crash confirmed, wider than reported, and filed nowhere

**Evidence.** I ran `title_probe.py`, which builds the query exactly as `repository.py:1985-1990`
does and runs it read-only against `ws:agent-team`'s `Document` index:
- `Report _ draft` → `%Report% %_% %draft%` → `RediSearch: Syntax error at offset 19`;
- `notes (v2) _` errors the same way;
- `a __ b` and `foo_ bar` parse.

The impact is wider than HISTORY says. In `background._safe_detect_update`, `background.py:274` calls
`find_update_shortlist`, which runs the band query and then the title query inside one call. The
title query's exception therefore throws away the already-computed LSH-band candidates. That
document loses **all** update detection, which the same function's `except` then logs and
swallows. The HISTORY entry says "Left for a follow-up" and also "K-068 is removed from
`BACKLOG.md`". No BACKLOG item tracks the follow-up (`grep '%_%\|Report _' docs/BACKLOG.md` finds
nothing).

**Suggested improvement.** File a new `K-0NN` in this same change. It should not be fixed inside
K-068: the path is different, its live tests belong in `test_repository.py` (and its `wf_repo`
hazard), and K-064 owns the same builder's next redesign. Write its body so the whole update
detection is lost, not just the title signal. Its test is a live `find_update_shortlist` call with
a lone-`_` title that still returns the band candidate.

### F-3 (minor) — K-064 "Related" rewrite is inaccurate in two places

**Evidence.**

**(a)** "The syntax-error crash on punctuation is fixed on both RediSearch fuzzy paths." This is
false for the title path's lone `_` (F-2). The HISTORY entry in the same diff says so itself.

**(b)** "On the entity path the AND is now stricter per name." Measured against pre-fix behaviour,
the opposite is true:
- every ASCII-punctuated name used to crash, so its recall was zero;
- among names that parsed before, 37 gained their self-hit and 8 lost it;
- all 8 losses come from F-1, not from the AND.

The extra terms are ones the index already holds for any stored name with the same punctuation. The
AND is stricter only against the strip alternative, and only for glued-vs-punctuated pairs.

For your example, I seeded `model-bench` and `model bench` on a throwaway graph:
- `model bench FX1` and `model-bench FX1` → `[]`;
- `model bench` and `model-bench` → both entities;
- `%modelbench%` → `[]`.

So the stored `model-bench` against the new `model bench FX1` misses both before and after. That
miss is K-064's implicit AND (`FX1`), unchanged by K-068. The one realistic pair that split loses
and strip would win is a stored glued `modelbench` against a new `model-bench`. That is inferred;
I did not seed it.

**Suggested improvement.** Rewrite the paragraph as one present-tense statement:
- the entity path no longer crashes;
- the title path still crashes on a lone `_` (cite the new item);
- both paths still AND their terms;
- splitting adds terms only where the index holds them too.

Then cite the glued-vs-punctuated pair as the OR-vs-subset input.

### F-4 (minor) — the tests are live and use real names, but cannot see F-1, and one pin encodes it

**Evidence.**

The live tests hit real RediSearch: `repo` → `conn` → `_schema` bootstrap on `ws:test`
(`tests/conftest.py:62-91,133`). I checked all 17 "real" names against `names.json`, and all 17
are present. `_schema` **skips** rather than fails when FalkorDB is down, so "45 passed" holds only
with FalkorDB up. That is the suite's existing convention.

The fixtures do not pass by accident in the strip/escape sense. M2 and M4 kill 19 and 35 tests, and
the logs agree. But three limits remain:

- **Identity tests cannot see F-1.** Each one seeds the **identical** name in a one-entity graph,
  and the only non-ASCII case is the trailing `…`, where splitting is harmless.
- **The `"…"` all-punctuation pin encodes the regression.** It asserts that a non-ASCII name
  gets no call.
- **No live precision assertion exists.** An OR-joined mutant (`k068or.py`) is killed only by 6
  pure string-shape tests (`or-mutant.out`).

**Suggested improvement.**
- Add the F-1 coverage test, red first.
- Flip the `…` pin to expect a query.
- Add one live negative case: a punctuated name must not return an unrelated same-type entity.

### F-5 (nit) — two stale or over-general doc sentences

**Evidence.**
- `repository.py:1953-1954` still says the title query is "built the same way `fusion._fuzzy_query`
  builds one". Two sentences later it says the entity builder splits instead.
- HISTORY and `QUERIES.md` say RediSearch rejects "every ASCII punctuation character except `_`
  inside a `%term%`". `probe_chars.py` measured only the mid-token position (`%x<c>y%`), and
  `%C++%` and `%C#%` parse (`probe_index.py`, old build).

**Suggested improvement.**
- Delete the "built the same way" clause.
- Qualify the rejection claim as "between word characters".

**Figures re-checked and correct:**
- 157/157 replay (before: 157 errors; after: 157 clean);
- 7450 / 2403 / 7436 / 14 (`replay-*.out`, `teco-replay.out`);
- strip 22 vs split 157 (`compare.out`);
- mutant kill counts 36/19/4/35;
- 19 names in the test list;
- the working-tree `fusion.py` is identical to `fusion.py.fixed`.

## What's solid

- The premise holds for ASCII punctuation and for `_`. `_` is **not** an index separator: for a
  stored `mcp__cypher__query`, `%mcp%`, `%cypher%` and `%query%` → `[]`, and
  `%mcp__cypher__query%` → hit. Splitting on ASCII punctuation fixes every real crash, and recall
  matches or beats strip (157 vs 22).
- Dropping underscore-only tokens is correct and evidenced: `probe_underscore.out` shows `%_%`
  errors and `%__%` parses.
- The empty-query posture is reused, so no degenerate call is made.
- Keeping the helper private to `fusion.py` and leaving the title path's code alone keeps the scope
  right.
- The mutation testing (M1–M4) is real and discriminating.

## Open questions

- Is the 8-in-5134 real-name regression enough to hold the change? My recommendation is no: apply
  F-1's one-regex fix before commit, since it is verified and strictly better on every measure I ran.
- Emoji inside a token misses under every builder I tried. Should it be an accepted limitation
  (`xfail`) or a backlog note?
- Process: `k068/mutate.py` rewrites the file at the path it is given, and `teco-fusion.bak` sits
  next to it. If that path was the working-tree `fusion.py`, mutants were run in the user's tree.
  The current file is byte-identical to `fusion.py.fixed`, so nothing leaked. Only `teco` can say
  which path was used.

## Pass 2

**Scope:** the pass-2 diff `git diff 3d91f750 -- falkor-chat/` and the implementer's
`…/scratchpad/k068/pass2/`.

**What I ran:**
- `test_fusion.py`: 141 passed, 1 xfailed.
- Three throwaway graphs: `ws:probe-k068-p2`, `ws:probe-k068-bs`, `ws:probe-k068-bs3`. Each was
  created and deleted, and the deletion confirmed.
- Read-only sweeps of `ws:agent-team`.
- `pytest -p` in-memory monkeypatch runs.

My scripts are in `…/scratchpad/k068-review/` (`pass2_probe*`, `bs_probe*`, `k068optc.py`,
`k068alnum.py`, `real_optc.out`).

**Verdict: approve with suggestions.** The control-character overrule is confirmed. The two new
findings are minor edge cases, both with tested remedies.

**The implementer's overrule of my F-1 regex is confirmed** (`pass2_probe.out`, my own probe on
`ws:probe-k068-p2`):
- Of 0x01–0x1F and 0x7F, only tab is an index separator.
- A raw `%aa<c>bb%` parse-errors for **every** one of them, including `\n`, `\v`, `\f` and `\r`.
- The pass-2 builder finds all 32 stored `aa<c>bb`.
- `aa\r\nbb` and `aa\x01\x02bb` are found by `%aabb%`. So the index drops control characters
  rather than keeping them; 2 edits would exceed fuzzy distance 1.
- NUL cannot be stored at all (`Failed to parse query parameter`).

My pass-1 regex was wrong on both counts. Its coverage probe sampled no control character except
tab.

**Interactions among the pass-2 fixes** (`pass2_probe.out` "degenerate"). Every all-control/`_`/`\`
name yields `''`, so no call is made and nothing crashes. That covers `\x01_\x02`, `_\n_`, `\_\`,
`_\t_`, `\x7f\x7f`, `__\__`, `%`, `_%_` and `\%`. Deleting `\` before splitting interacts badly
only with an escaped separator (P2-2). The flipped `…` pin interacts badly with fuzzy matching
(P2-1). The emoji strict-xfail is consistent with the code: `%😀%` parses, and `aa😀bb` misses.

### Pass-1 dispositions

- **F-1:** fixed. I re-ran both checks:
  - real names: 5147 hit / 0 miss / 10 empty over 5157 distinct names (`real_optc.out`, "current"
    = tree);
  - the per-class live test is present (`test_fusion.py:242`).
- **F-2:** fixed. K-075 is filed, and its text matches my probe: `Report _ draft` errors at offset
  19, while `__` and `foo_` parse. The band-candidates loss is stated correctly, and the test is
  `repo`-only.
- **F-3:** fixed. K-064 "Related" no longer claims both paths are fixed, and it cites K-075. The
  glued-vs-punctuated input is phrased as "can be missed". That is still inferred, unseeded, and
  acceptable.
- **F-4:** fixed:
  - the `…` pin is flipped;
  - a precision test exists (`:284`);
  - a coverage test exists.

  The single-entity-fixture weakness persists for symbol-only names (P2-1).
- **F-5:** fixed. The "built the same way" clause is gone, and the rejection claim now says
  "between word characters".

**HISTORY figures:**
- The 27 control characters check out: 8 + 18 + 1.
- 141/1 and 157/157 match.
- 7480 / 7464 / 0 / 16 match `replay2.out`.
- The mutant kill counts match `pass2/mutant-*.out`.
- One wording nit: "16 all-ASCII-punctuation names yield no query". The 16 also include
  underscore-only and backslash-only names (`_`, `\`), so "16 names with no token" is accurate.

### P2-1 (minor) — a symbol-only name now issues a fuzzy term that matches every short term, and writes a junk suggestion

**Evidence.** Pass 2 flipped the `…` pin, as my F-4 asked, so `∅`, `…`, `→` and a lone NBSP now
build `%∅%`-style terms. Run read-only against `ws:agent-team`, each matches **802** entities: a
1-edit fuzzy term on one character matches any 1–2 character indexed term.

For the real `∅` (type `Concept`), the ranked top-5 is `Auth ✔`, `baseline ≥ 0.90`, `h-full`,
`if ! f; then rc=$?`, and only then `∅`. `classify_fuzzy` takes any candidate as `suggested`, so a
pending `SAME_AS` to `Auth ✔` would be written.

The `∅` test passes only because `ws:test` holds one entity, which is exactly the fixture artifact
F-4 warned about. This is not a regression against the baseline: the old whitespace build sent the
same `%∅%`. A 1-character word name (`%K%` → 808 hits) already behaves this way.

**Suggested improvement.** Drop tokens with no letter or digit, `re.search(r"[^\W_]", tok)`. Then
a symbol-only name gets no call.

Verified in memory (`alnum-tests.out`): 139 passed. The only failures were the two pins that encode
symbol-only behaviour, the `…` shape pin and the `∅` self-find, and both should flip. Add the guard
first as a red step: a live test that seeds `∅` together with a same-type `Auth ✔` and asserts
that `Auth ✔` is not returned. This is a design call, recall of symbol-only names against junk
suggestions, so the implementer decides.

### P2-2 (minor) — deleting `\` before splitting misses names where `\` escapes a separator

**Evidence.** `bs_probe.out`, on a throwaway graph. The index treats `\<punct>` as a literal. For
example, `a\-b` is found by `%a\-b%` and by `%ab%`, while `%a-b%` errors. The builder instead
emits `%a% %b%`, `%foo% %bar%`, `%re% %compile%` and `%a% %b%`, and each misses its stored
`a\-b`, `foo\.bar`, `re\.compile` or `a\%b`.

This is realistic in a developer knowledge base: `ws:agent-team` already holds
`/GRAPH\.[A-Za-z_.]*/`, which hits by luck. Real data shows 0 misses today. The F-1 coverage test
samples one character per name, so `\`+punctuation pairs are never exercised. The docs'
"each term is one the index can hold" overclaims for this axis.

**Suggested improvement.** Keep a `\` that precedes ASCII punctuation inside the term, delete any
other `\`, then tokenize (`k068optc.py`). I verified it three ways:
- `test_fusion.py`: 141 passed, 1 xfailed;
- real names: 5147 / 0 miss / 10 empty, identical to the tree;
- `bs_probe3.out`: every case hits except `x\ y`, which the tree misses too.

`%\-%`, `%a\%%` and `%\%%` parse. Add a second coverage axis first: `aa\<p>bb` for each ASCII
punctuation `p`, red against the tree. Optional within K-068; fine as a follow-up if the owner
prefers.
