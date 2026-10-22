# Review — F1 fix: `filter_products` treats blank arguments as omitted

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-079

## Pass 1

**Scope.** Code gate on the uncommitted working-tree diff that implements F1 from
`falkor-chat/docs/reviews/salesperson-fabricated-prices-rca.md` (CF-1). Files:
`server/falkorchat/repository.py`, `server/falkorchat/tools.py`, `server/tests/test_repository.py`,
`server/tests/test_tools.py`, `docs/QUERIES.md` §15.2. The baseline is `HEAD` `41ce1c2b` plus the
RCA's F1. F2–F5 are out of scope.

**Verdict: approve with suggestions.** Both departures from F1 hold up when run. Departure 1 is an
improvement on what the RCA recommended. No blockers and no majors.

CPG: considered, not relevant — a two-function diff, verified directly against source with targeted
tests, mutations of a scratch copy and read-only Cypher. `cpg_falkorchat` is also stale.

### What was run

- **Targeted tests:** both new tests, plus all of `tests/test_tools.py` (offline stubs, no `wf_repo`):
  87 passed. I did not run the full suite or any `wf_repo` test.
- **Mutations:** made on a scratch copy of `server/` with `PYTHONPATH` pointed at it. I confirmed via
  `__file__` that the copy was the module imported, and the real tree was untouched (`git status`
  unchanged). Each mutation turned its test red:
  - M1: drop `or None` in `Repository.filter_products` → the repo test fails.
  - M2: make `_blank_as_none` return its input unchanged → the tool test fails.
  - M3: `_blank_as_none` catches only `""`, not whitespace → the tool test fails.
  - M4: wrap only `minPrice`, not `maxPrice` → the tool test fails.
- **Read-only Cypher on `reference`:** over 15 products, `p.price >= coalesce('', -1.0)` matches 0,
  `p.price <= coalesce('', 1e9)` matches 0, `<= coalesce('50', …)` matches 0, and `<= coalesce(50, …)`
  matches 6.
- **Callers of `filter_products`:** `tools.py:513` and `storefront.py:1520`. The storefront always
  passes `category=None`.

### Departure 1 — blank-category rule in the repository, not the tool: **confirmed, and better than F1**

The RCA put this rule in the tool, and that was the weaker placement. "Blank" is defined by the
normalizer's output (`normalize_name(x) == ""`). `Repository.filter_products` is where
`normalize_name` already runs. A tool-layer rule would have had two options, both worse: call the
normalizer a second time, or approximate it with `strip()` and risk drifting from it.

Putting the rule in the repository covers every caller, including any future one. The only caller
today that can send a non-`None` category is the tool. The repository test probes the right axis: it
asserts `normalize_name(blank) == ""` before each case, and it compares the full parameter dict
against the omitted call's parameters. M1 shows the test bites.

### Departure 2 — blank `minPrice`/`maxPrice` treated as omitted at the tool: **confirmed**

The implementer's live claim reproduces: a `""` bound survives `coalesce` and matches 0 of 15. The
tool layer is the right place for this, because past it everything is typed `float | None` and only
the JSON boundary can carry a string. M2–M4 show the test pins both bounds and the whitespace case.

Caveat: this is preventive. In the run's LM Studio logs (`lms-server.log` and `lms-server-q3.log`),
all four priced `filter_products` calls sent JSON numbers, and none sent `""` for a bound. Cheap and
consistent with the category fix, so it is worth keeping, but no observed failure motivated it.

### Findings

**m1 (minor) — a numeric-string bound still silently matches nothing.** `_blank_as_none` handles only
blank strings. `maxPrice: "50"` reaches `coalesce('50', …)`, which matches 0 rows where `50` matches
6 (verified above). The result is the same silent "no matching products found" CF-1 produced. It was
not observed: every logged call sent a number. Two options, the owner's choice:
- Coerce numeric strings to `float` in the same helper, renamed to describe the new job. A
  non-numeric non-blank string still falls through, so it would need its own test.
- Leave it as is, and note in `_blank_as_none`'s docstring that a numeric string is deliberately not
  handled.

**n1 (nit) — `normalize_name(category) or None if category is not None else None`** depends on
`or` binding tighter than the conditional expression. It is correct (M1 and the test prove the
behaviour), but parenthesize `(normalize_name(category) or None)` so the next reader doesn't have to
re-derive that.

**n2 (nit) — the tool description change is visible to the model.** "(an empty value counts as
omitted)" now reaches ministral on every turn. It is accurate and harmless, and there is no
measurement behind it. Mention it in the `HISTORY.md` entry so a later behaviour shift in
`filter_products` calls has an obvious suspect.

### What's solid

- Both tests are offline. The repository test uses a recording connection rather than `wf_repo`,
  which keeps it safe against the shared `reference` graph.
- The `QUERIES.md` §15.2 note matches the code exactly, and the docstrings cite the RCA.
- No existing test relied on the old `""` → 0-rows behaviour (`grep` of `tests/`).

### Open questions

- None blocking. K-079 stays open for F2–F5. Per the RCA, F2 waits for K-078's §4.

## Pass 2

**Verdict: approve.** The tree was re-checked: `tests/test_tools.py` plus the repository blank-category
test, 89 passed. Six mutants of `_coerce_price_bound` were run on a scratch copy of `server/`, imported
via `PYTHONPATH` (confirmed with `__file__`), and every one was killed:
- drop the finite check
- drop the float coercion
- drop `strip()`
- blank passes through instead of becoming `None`
- `maxPrice` left unwrapped
- non-numeric string becomes `None`

The real tree's `git status` was unchanged afterwards.

- **m1 — fixed.** A numeric string becomes a `float`, so `"50"` reaches the query as `50.0`, which
  matches 6 rows (Pass 1's live check) where the string matched 0. `nan`/`inf` and non-numeric strings
  pass through unchanged, with no new error path. Residual: `"$50"` still silently matches nothing.
  That is accepted and documented in the helper's docstring, and no logged call sent it.
- **n1 — fixed.** `repository.py` now reads `(normalize_name(category) or None) if …`, and the
  repository test still passes.
- **n2 — not fixed, deliberately.** It stays a closeout item for the `HISTORY.md` entry.
