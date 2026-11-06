# Connected code graph — evaluation protocol, key-normalizer amendment (step 8a)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (M10) · **Extends:** `docs/plans/connected-code-graph-eval-protocol-ml.md` · **Version:** 1.5 · **Reviews:** `docs/reviews/connected-code-graph-eval-protocol2-ml.md`

*2026-11-06 (v1.5): §1.5 pins stated as a format with a build-time rule (S50); v1.4's K48 and idempotence changes stand.*

## 0. What this is, and how to read it

**Read with `docs/plans/connected-code-graph-eval-protocol-ml.md` v1.2 ("EP").** EP stays
authoritative everywhere except the clauses §1 below names. Where §1 names a clause, §1 controls.
Section numbers (§2.2, §13.4, …) are EP's. "Spec3" is
`docs/plans/connected-code-graph-bundle-spec3.md`; its §6 `http` rule and §1.7.1/§1.8 are the
rule this note implements, and **its step numbers are the ones used here** (1 verb, 4 leading `/`,
5 percent-decode, 6 collapse and dots, 7 parameters, 8a mask).

**The question.** Spec3 §1.8 adds step 8a to the `http` `keyNorm` rule. A literal path segment that
fails the token screen (spec3 §1.7.1) becomes `{#}` in a provider key and `{?}` in a consumer key.
EP §13.4's `normalize` must produce the same `keyNorm` the system produces. Otherwise:
- a writer's contract anchor converts to a `cpId` that is not in the graph;
- or golden-key writers would have to screen literals themselves, which Appendix B never asks of
  them and which F-8 rules out.

**The decision it serves.** S49 (`match_keys.py`) encodes §1 and the oracle in §2. Nothing a
writer sees changes. §3 lists what this places on `docs/plans/connected-code-graph-p0-2.md`.

**Why a successor, not an in-place edit of EP.** EP v1.2 passed its S06 gate, so collision rule 5
sends a change to a successor. This change is not errata:
- EP v1.2's `normalize` lists its `http` steps as a closed set: verb, leading `/`, collapse,
  percent-decoding and parameters.
- Neither step 8 nor step 8a is in that set, and EP's list also omits dot-segment resolution and
  the RFC 3986 order (§1.1).
- V3 ("a changed one is replaced and logged") only applies what `normalize` lists, so 8a is not
  implied there either.
- Adding 8a makes some keys equal that EP v1.2 kept distinct (§1.3).

**Does it move AC-5's thresholds or scope? No.**
- The 75 counting items, the per-stratum 8, the strata, census scope and the hit rule are unchanged.
  The change only affects how a contract locator's text becomes its id.
- **Exposure today is zero.** Extraction definition: census-scope files (`git ls-files`, minus
  `deprecated/`, `cpg/`, `node_modules/` and EP §3.3's test globs). From these, take the string
  argument of every Python route decorator (`get`, `post`, `put`, `patch`, `delete`, `api_route`,
  `route`, `websocket`), every `prefix=` and `mount` path (AST), and every `/`-leading string
  literal in `salesperson/src/api/`. Split on `/`, drop `${…}` interpolations and parameter
  segments, and keep the distinct remainder. That gives **51 paths and 39 distinct literal
  segments, of which 0 fail** `screen_fails` (run 2026-10-30). The analyst's three census-scope
  definitions give 39–50 segments, also with 0 failing.

**What it does change: AC-24 and the contact set become more lenient, monotonically.** §1.2 runs
two kinds of contract locator through all of `normalize` (steps 1 and 4–7, 8a, and the `mcp__`
strip): AC-24 gold anchors and the contact set C. EP's §2.2 converted both without
normalizing them. As a result:
- a gold writer's slip, such as a lowercase verb or `:id`, is now corrected, not scored as not
  understood;
- a contact entry can now catch an anchor written in a different but equivalent spelling.

Both changes are **monotone**. The function only coarsens EP v1.2's identity (§1.3), so any pair
that was equal stays equal: understood items and contacts can only be added. More contacts are
conservative for the split. Every gold rewrite is logged and counted in S47's report (§1.2), as
AC-5's V3 rewrites are, so the leniency is visible. It moves no AC-24 threshold or cell.

## 1. Amendments (normative)

### 1.1 `http` `keyNorm` in `normalize` (replaces EP §13.4's "http `keyNorm`" bullet)

- **http `keyNorm`:** apply these to the writer's string, in spec3's order:
  - step 1: an uppercase verb;
  - step 4: a leading `/`;
  - step 5: RFC 3986 percent-decoding of the unreserved characters, with the hex of every other
    escape uppercased;
  - step 6: runs of `/` collapsed, **and `.` and `..` segments resolved** as RFC 3986 §5.2.4
    does: a trailing `.` or `..` leaves the directory form, so the result ends in `/`, and `..` at
    the root stays at the root (K43–K47);
  - step 7: every parameter syntax rewritten to `{}`/`{**}`;
  - **then spec3 §1.8's step 8a, in its provider form:** every remaining literal segment `s` with
    `screen_fails(s)` replaced by `{#}`.
- Step 3, mount composition, is still the writer's job.
- Step 8, which turns an unresolved consumer segment into `{?}`, does not apply. Writers write the
  route the server serves (Appendix B.0). The one consumer-form case is §1.2's unserved gold.

**The details that fix the function:**
- **Provider form, by default.** Every `http` contract locator in a key item names a **served**
  route:
  - B.0 asks for "the full path the server serves";
  - EP §4.1 requires `provider` for every contract anchor;
  - `serves[]` lists points the unit serves.

  The system's contract point for a served route with a failing literal is the provider key with
  `{#}`. The consumer form `{?}` was **rejected** for key items. It would name a different `cpId`,
  which exists only if some consumer sends that literal and links to the provider only as
  `possible`. The gate would then ask from the wrong point.
- **One 8a implementation, in `cpg/ccg/`.** It is `mask_http_segments(path, placeholder)` in
  `cpg/ccg/keynorm.py`: a pure function over a post-step-7 path. `placeholder` is `{#}` (provider
  form) or `{?}` (consumer form). `assemble.py` (S12), `match_keys.py` and the P1 assembler import
  it, and none reimplements it, as with `ids.py`. It does step 8a only; steps 1 to 7 belong to
  the caller and follow the order above, so the earlier steps are tested by §2 too.
- **What is screened.** Each `/`-separated segment of the path, as it stands after step 7. That
  includes the percent-escapes step 5 left in place, uppercased. Not screened:
  - the verb;
  - an empty segment, such as the one after a trailing `/`;
  - the four placeholders `{}`, `{**}`, `{?}` and `{#}`.
- **A partly parameterized segment is a literal.** Step 7 rewrites only a segment that is wholly a
  parameter. So `{x7k2q9w4m1p8z3v6}.json` is screened as written and masked (K30). This
  over-masks, which spec3 accepts, and the shared function makes S12 agree.
- **The placeholders are fixed points of step 7 as well.** This is a clarification: `{?}` and
  `{#}` are no framework's parameter syntax.
- **Idempotence.** Spec3 decodes (step 5) before it resolves dots (step 6), so an encoded dot
  segment such as `%2E%2E` resolves in the same pass (K42). The order is spec3's, and this note
  must equal the system, so it never reorders the steps. `normalize` is idempotent **except** when
  step 5, a single pass, forms a new escape of an unreserved character from a stray `%` before an
  escape: `%2%45` becomes `%2E`, which a second pass decodes (K48). Equivalently, an output is a
  fixed point unless one of its segments holds an escape of an unreserved character. Nothing here
  depends on idempotence: every id is computed once from the writer's original object (§1.4,
  §1.5), and the coarsening in §1.3 holds regardless. No real route is written this way.
- **`mcp_tool` keys are not screened.** 8a is an `http` rule (spec3 §1.8). The `mcp__<server>__`
  strip is unchanged.
- **Logging.** A rewrite by 8a is a V3 change: the item is **replaced and logged, never
  invalidated**. The log entry gives the item's `seq`, the rule `8a` and the count of masked
  segments. The method owner reviews it at S45. Like every GC tool, it prints no item content to
  stdout (EP §13).

### 1.2 Every contract locator the protocol converts to a `cpId` (adds to EP §2.2)

**`normalize_contract(loc, form="provider")`** is exported from `match_keys.py`. It is a pure
function over a plain dict, and it returns the normalized locator: `http` by §1.1 with 8a in the
given `form`, and `mcp_tool` by EP §13.4, where `form` has no effect. Every
`contract` locator goes through it **before** EP §2.2's conversion to a `cpId`:
- **key items:** `anchor` and every `serves[]` entry, at S45 (EP §13.4, `normalize`);
- **AC-24 gold:** every `contract` locator in `gold.anchors` and `goldOptions[*].anchors`, **at
  S47, once**.
  - Each gold `http` locator is stored with its **`form`** (`provider` | `consumer`). The form is
    `provider` unless the unserved-gold check below set it to `consumer`. It is logged in S47's
    rewrite log beside the `qid`.
  - The held-out and dev `ac24` files store each question's gold with its normalized locators and
    keep the writer's original object beside it, as the AC-5 files do (EP §9.2). The gate harness
    reads the stored normalized gold and never re-normalizes it for scoring (§1.5).
  - Each rewrite is logged by `qid`, and S47's report counts them per cell.
  - The interpreter's resolved anchors are system ids, taken as emitted.
- **Unserved gold.** A gold `http` anchor may name a route nobody in the repository serves, such
  as an external API; B.2 allows that. The graph holds such a route only in **consumer** form. So
  at S47, for each gold `http` anchor:
  - **if** its provider-form `cpId` matches no key-item anchor, no `serves[]` entry and no listed
    unit's served point, **and** the anchor has a masked segment;
  - **then** the method owner checks it against source, and if the route is unserved, sets the
    consumer form `{?}` (K39), logged.

  Exposure is 0 today, because no census-scope literal fails.
- **The contact set C:** every `contract` locator from every source in EP §8.2's table goes through
  it at S47. That covers S14, S30, S56 and `cpg/tests/golden/*`, including S34's
  `answer-ac1.json`. Without this, a contacted masked route would convert to a different
  `anchorId` from its key items, escape contact, and could land in held-out: a leak (K27).

### 1.3 What changes in golden-key equality

**No pair of keys equal under EP v1.2 becomes unequal.** §1.1's function is EP v1.2's function
followed by dot-segment resolution and then 8a, so equal inputs still give equal outputs.

**Which unequal pairs become equal.** Two keys merge if and only if all of these hold:
- they have the same verb and the same segment count after steps 1 and 4–7 (dot segments
  resolved);
- they are equal after those steps, except at some positions;
- at every position where they differ, each side holds either a screen-failing literal or `{#}`.

So `POST /hooks/x7k2q9w4m1p8z3v6` ≡ `POST /hooks/a1b2c3d4e5f6g7h8` (K24), and a failing literal
≡ a writer-written `{#}` (K34). Two keys that differ only by a resolvable dot segment, such as
`GET /a/../b` and `GET /b`, also merge, because EP v1.2 did not resolve dots.

**The merge is intended.** The graph makes the same merge, so the eval's anchor identity now
equals the system's, where EP v1.2's was finer. These are the consequences, each handled by EP's
existing machinery:
- **Split.** The merged keys are one anchor, so one split unit (EP §9.1).
- **Matching.** Writers' records of (masked route 1, X) and (masked route 2, X) match mechanically,
  as they should: the system cannot tell them apart either.
- **Near-misses.** EP §6.3's N2 rewrites every `{…}` segment to `{}`, so a `{#}` key and a `{}` key
  with the same `impactedId` can become N2 candidates. That only adds a candidate for
  adjudication. Mechanical matching is unaffected, because `{#}` ≠ `{}` (K25).

### 1.4 The leak check's `keyNorm` probes (clarifies EP §9.4, "every `keyNorm`")

For a held-out item, "every `keyNorm`" means **both** of these:
- the **normalized** form (§1.1), removed when it equals the normalized form of some dev or C
  value;
- the **pre-8a** form, `normalize`'s output through step 7 (without 8a), removed when it
  equals the pre-8a form of some dev or C value.

**Both probes, and the dev and C values they are removed against, are computed from the
writer's original object.** That holds for AC-5 records and AC-24 gold alike, and never
for a stored normalized locator. The reason: the pre-8a form of a stored `POST /hooks/{#}` is
`POST /hooks/{#}` again, which carries no literal. Computed from stored gold, a report leaking a
held-out gold literal would pass, and a report naming a public dev gold literal would fail (K41).

The writer's raw spelling is **not** a probe.

**Why the second probe is needed.** Distinct masked routes share one normalized string. So a dev
value masked at the same position removes the normalized probe. The pre-8a form still carries the
literal, and it is the one that catches the literal in a report (K29a).

**Why it is the pre-8a form and not the raw spelling.** The raw spelling makes the check fail
clean reports:
- a held-out `get /shop/api/products` does not equal dev's `GET /shop/api/products`, so it was
  never removed, and the casefolded report then hit it (K36);
- it would also probe a writer's parameter names, such as `{item_id}`, which EP §9.4 never
  probes (K37).

The pre-8a form has neither problem, and it differs from the normalized form only where 8a
masked something.

**How dev can share a held-out anchor's normalized key.** A dev **key item** anchored at the same
`cpId` cannot: it has the same `anchorId`, so it is in the same cluster. The states that do occur:
- a dev AC-24 gold anchor (contact does not apply to AC-24, EP §10.2);
- a dev unit's `serves[]` entry (the served-anchor rule is one-way, EP §8.2);
- a held-out anchor merged with a different masked route (§1.3).

### 1.5 Versioning, the screen record, and the drift guard

- **Versioning.** `eval-manifest.yaml`'s "this note's path and version" (EP §13.1 step 8, written
  by S40) records both documents. So does the held-out manifest's `protocol` field (EP §9.3), as
  `<EP path>@<version>+<EP2 path>@<version>`. Each records the versions in force when it was written: `eval-manifest.yaml` at S40, the held-out manifest at S47. Both are build-time records, so a later revision never bumps them.
- **The screen record.** `eval-manifest.yaml` also records `screen: {specVersion, bundlePySha256}`:
  spec3's version, and the SHA-256 of `cpg/ccg/bundle.py` at `EVAL_COMMIT`. The record is for
  diagnosis. It is **not** a stop condition, because any unrelated edit to `bundle.py` would
  change the hash and stop a sound gate.
- **The drift guard (`match_keys.py renormalize-check`).** Before scoring, at every gate:
  - the harness re-runs `normalize_contract`, with the gate's `bundle.py`, over every stored
    writer-form contract locator: key-item `anchor` and `serves[]`, AC-24 gold, and C. **Each gold
    locator is re-run with its stored `form`** (§1.2), and every other locator in provider form.
    So a consumer-form gold id is reproduced (K40). A screen change that unmasks or newly masks
    its segment is still caught;
  - it compares the resulting ids with the ids stored at S45/S47;
  - **any difference stops the gate** before a question is asked. The output is counts and ids
    only.

  The method owner then decides on any re-keying, as a protocol revision recorded before the run
  (EP §12.3). The re-keying uses writer forms and the screen only, never the system's answer, so
  it adds no bias toward what the system finds (the same argument as EP §5.4).
- **A contract anchor that is absent from the gate graph.** After the guard passes, a held-out
  contract anchor (or probe) whose `cpId` is absent from the gate snapshot's contract points is
  still a **miss**: the system made no point for a route that exists. The miss is reported with the
  flag `anchor-not-in-graph`, beside EP §5.3's depth-bound and extraction-miss flags. That way it
  is never a miss with no signal.

## 2. Test oracle (S49)

**Unit function:** `normalize_contract({"type":"contract","kind":k,"keyNorm":input})["keyNorm"]`.
"Decided by" names at least one rule. S49's tests assert the output, not the rule.

| # | kind | input | normalized `keyNorm` | decided by |
|---|---|---|---|---|
| K1 | http | `GET /shop/api/health` | `GET /shop/api/health` | all segments pass (SL-1 unchanged) |
| K2 | http | `POST /hooks/x7k2q9w4m1p8z3v6` | `POST /hooks/{#}` | **provider literal masked to `{#}` (BS-m7)**; T1 |
| K3 | http | `POST /hooks/a1b2c3d4e5f6g7h8` | `POST /hooks/{#}` | T1 |
| K4 | http | `POST /hooks/{}` | `POST /hooks/{}` | a placeholder, not screened |
| K5 | http | `POST /notify/x7k2-q9w4-m1p8-z3v6` | `POST /notify/{#}` | T6 (dash-chunked) |
| K6 | http | `POST /notify/x7k2+q9w4+m1p8+z3v6` | `POST /notify/{#}` | T6 (the widened class, spec3 §1.7.1) |
| K7 | http | `GET /builds/3f2a9c1e5b7d` | `GET /builds/{#}` | T3 (12 hex) |
| K8 | http | `DELETE /items/550e8400-e29b-41d4-a716-446655440000` | `DELETE /items/{#}` | T4 |
| K9 | http | `POST /cb/sk-abcdefghijklmnopqrs` | `POST /cb/{#}` | T5 alone |
| K10 | http | `POST /services/T024BE7LD/B01ABCDEFGH/abcdefghijklmnopqrstuvwx` | `POST /services/T024BE7LD/B01ABCDEFGH/{#}` | T2, last segment only |
| K11 | http | `GET /t/3f2a9c1e5b7d/x7k2q9w4m1p8z3v6/info` | `GET /t/{#}/{#}/info` | each segment on its own |
| K12 | http | `post hooks//x7k2q9w4m1p8z3v6/` | `POST /hooks/{#}/` | steps 1, 4, 5 and 6 first; the trailing `/` is kept, and its empty segment is not screened |
| K13 | http | `GET /hooks/x7k2q9w4%6d1p8z3v6` | `GET /hooks/{#}` | step 5 decodes `%6d` before the screen |
| K14 | http | `GET /a/x7k2q9w4%2fm1p8z3v6` | `GET /a/x7k2q9w4%2Fm1p8z3v6` | step 5 keeps `%2F`, uppercased; it splits the runs, so the segment passes |
| K15 | http | `GET /x/{x7k2q9w4m1p8z3v6}` | `GET /x/{}` | step 7 before 8a; a parameter's name is not screened |
| K16 | http | `GET /static/{p:path}` | `GET /static/{**}` | step 7, catch-all |
| K17 | http | `GET /ntfy/kxqzmwvptryhbnjdlc` | unchanged | passes: spec3's letters-only residual (§1.7.6) |
| K18 | http | `GET /wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` | unchanged | passes piece by piece: spec3's base64 residual (§1.7.6) |
| K19 | http | `GET /release-2024-notes/s3-bucket-v2-us-east-1` | unchanged | passes (2 and 5 transitions) |
| K20 | http | `POST /hooks/{#}` | `POST /hooks/{#}` | idempotence: a placeholder is a fixed point |
| K21 | http | `GET /x/{?}` | `GET /x/{?}` | idempotence |
| K22 | http | `GET /x7k2q9w4m1p8z3v6` | `GET /{#}` | the first segment is screened like any other |
| K23 | mcp_tool | `mcp__falkor__x7k2q9w4m1p8z3v6` | `x7k2q9w4m1p8z3v6` | the namespace strip only; 8a is `http` only |
| K30 | http | `GET /f/{x7k2q9w4m1p8z3v6}.json` | `GET /f/{#}` | a partly parameterized segment is a literal (§1.1) |
| K31 | http | `GET /a/x7k2q9w4m1p8z3v6/../b` | `GET /a/b` | step 6 resolves `..` before 8a |
| K32 | http | `GET /a/./x7k2q9w4m1p8z3v6` | `GET /a/{#}` | step 6 resolves `.` |
| K33 | http | `POST /hooks/x7k2q9w4m1p8` | unchanged | 12 characters: below T1, T6 under 16, not hex. The drift fixture, K35 |
| K42 | http | `GET /a/%2E%2E/b` | `GET /b` | step 5 decodes `%2E%2E` to `..`, and step 6 then resolves it: spec3 §6's example. A fixed point |
| K43 | http | `GET /a/b/..` | `GET /a/` | step 6 as RFC 3986 §5.2.4: a trailing `..` leaves the directory form (spec3 §6) |
| K44 | http | `GET /a/b/.` | `GET /a/b/` | a trailing `.` leaves the directory form |
| K45 | http | `GET /a/b/%2E%2E` | `GET /a/` | step 5 decodes first, then K43 |
| K46 | http | `GET /a/..` | `GET /` | the root |
| K47 | http | `GET /..` | `GET /` | `..` at the root stays at the root |
| K48 | http | `GET /a/%2%45%2%45/b` | `GET /a/%2E%2E/b` | step 5 is one pass: it forms `%2E%2E` from `%2%45%2%45` and does not decode it again. **Not a fixed point**: a second pass gives `GET /b` (§1.1) |

Every output in K1–K33 and K42–K47 must also be a fixed point of the function. K48's is not, by §1.1.

**Integration cases (S49):**
- **K24, merge.** KA holds (K2's key, X) and KB holds (K3's key, X). They have one `itemId` and
  match mechanically.
- **K25, no false match.** KA holds (K2's key, X) and KB holds (K4's key, X). They do not match,
  and they form an N2 candidate pair.
- **K26, V3 log.** An item with K2's input is valid and carries `POST /hooks/{#}`. The log names
  rule `8a` with one masked segment, and stdout shows no literal.
- **K27, contact.** C holds `POST /hooks/x7k2q9w4m1p8z3v6`, written raw by S14 or S30, and a key
  item's anchor is `POST /hooks/a1b2c3d4e5f6g7h8`. They share an `anchorId`, so the item is
  **contacted** and goes to dev.
- **K28, AC-24 gold.** A gold anchor with K2's input, stored normalized at S47, equals
  `contract_point_id("http", "POST /hooks/{#}")`.
- **K29, leak probe, masked twin.** A held-out item with K2's input, and a **dev unit's `serves[]`
  entry** (or a dev AC-24 gold anchor) with K3's input.
  - **(a)** A report containing `missed post /hooks/x7k2q9w4m1p8z3v6 here` **fails**: the pre-8a
    probe survives.
  - **(b)** A report containing only `missed post /hooks/{#} here` **passes**: the normalized
    probe was removed by dev equality.
- **K34, `{#}` identity.** K2's input and K20's input have one `cpId`.
- **K35, drift guard.** Store ids under the working tree's screen. Then:
  - re-running `renormalize-check` under the same screen gives 0 differences, and the gate
    proceeds;
  - re-running it under a stricter screen (T1 lowered to 12, a test double) changes K33's id, and
    the gate **stops**.
- **K36, writer spelling is not a false positive.** A held-out item written
  `get /shop/api/products`, and a dev value `GET /shop/api/products`. A report naming
  `GET /shop/api/products` **passes**.
- **K37, no new probe class.** A held-out item written `GET /api/items/{item_id}`, and no dev twin.
  A report containing `GET /api/items/{item_id}` **passes**, as under EP §9.4.
- **K38, normalized probe without a twin.** A held-out item with K2's input, and no dev twin. A
  report ending `missed post /hooks/{#}.` **fails**, under EP §9.4's trailing-punctuation rule.
- **K39, unserved gold.** For an unserved gold route with K10's input, the method owner's
  consumer form is `POST /services/T024BE7LD/B01ABCDEFGH/{?}`. For K1's input, which has no
  masked segment, it is never applied.

- **K40, consumer-form gold under the drift guard.** K39's gold is stored with
  `form: consumer`. Under the same screen, `renormalize-check` gives **0 differences**. Re-running it
  in provider form would differ, which is why the form is an input.
- **K41, gold leak probes come from the writer original.** A held-out AC-24 gold anchor written
  `post /hooks/x7k2q9w4m1p8z3v6`:
  - **(a)** A report containing `q12 gold post /hooks/x7k2q9w4m1p8z3v6 not understood`
    **fails**.
  - **(b)** The same literal as a **dev** gold anchor's writer original, with the held-out item
    above. A report containing `dev q3 gold post /hooks/x7k2q9w4m1p8z3v6` **passes**: the probe
    is removed by dev equality, computed from both writer originals.

**Count: 48 oracle cases.**
- **Unit cases (34):** K1–K23, K30–K33 and K42–K48.
- **Integration cases (14):** K24–K29 and K34–K41.

**Reproduced** against the working tree's `cpg/ccg/keynorm.py` and `bundle.py` by the analyst's Pass 3 and by my emulation (2026-11-01): see `docs/reviews/connected-code-graph-eval-protocol2-ml.md`. K26 awaits S49's logger, and S49 reruns every row on the final `bundle.py`.

## 3. Obligations on `docs/plans/connected-code-graph-p0-2.md`, by step

| p0-2 location | Obligation |
|---|---|
| **§3 S06 row; §14** | The accepted contract is EP v1.2 **plus** this note. §14's consumers gain this note. |
| **§3 S10** (in flight) | Nothing new. The final `screen_fails` keeps spec3's outcomes, and S49 reruns §2 on it. |
| **§3 S12** | Imports `mask_http_segments` from `cpg/ccg/keynorm.py` (§1.1). Its steps 1 to 7 follow spec3's order, and its `http` rule passes K1–K22, K30–K33 and K42–K48 in provider form, in addition to "SL-1/SL-2 unchanged". |
| **§3 S14, S30, S34** | **None.** Their lists and fixtures stay in the raw spec §3.5 form, and S47 normalizes them. |
| **§3 S40** | `eval-manifest.yaml` records both documents' paths and versions, plus `screen: {specVersion, bundlePySha256}` (§1.5). |
| **§3 S41–S44, S52** | **None.** Appendix B is unchanged, and writers never screen. |
| **§3 S45** | `normalize` applies §1.1: decoding at step 5, collapse and dot resolution at step 6, and 8a after step 7. 8a rewrites are V3 replace-and-log entries, and the method owner reviews the log. |
| **§3 S47** | Every C source goes through `normalize_contract` before `anchorId` (§1.2). AC-24 gold is stored normalized **with each `http` locator's `form`** (`provider` | `consumer`), each rewrite is logged and counted, and the unserved-gold check is run (§1.2). The manifest's `protocol` string follows §1.5. `--check-report` uses §1.4's two probes. |
| **§3 S49** | `normalize` gains §1.1, importing the shared 8a function. Exports `normalize_contract(loc, form)`. Adds the `renormalize-check` verb, which re-runs each gold locator with its stored `form` (§1.5). `--check-report` computes both probes and their removal values from writer originals (§1.4). Oracle **K1–K48** joins "Oracle cases pass". **Dependencies gain S10 and S12.** It imports `mask_http_segments`; its steps 1 to 7 follow spec3's order. |
| **P1 assembler** (replaces `assemble.py`) | **Imports `mask_http_segments` (§1.1), and never reimplements it.** The eval-side guard (§1.5) cannot see the graph side, so this import is what keeps the two equal. |
| **P1 gate harness** (EP §14 row) | Reads stored normalized gold and ids. Runs `renormalize-check` before scoring, and stops on any difference. Flags `anchor-not-in-graph` misses (§1.5). Imports, and never reimplements. |
| **§15 follow-ups** (F-8 line) | F-8 is delivered by this note and encoded at S49. |
| **§16.1 row BS2-21** | Placed: S49, with S40, S45 and S47 as above. |

## 4. Risks and open questions

- **R1, divergence from `assemble.py`.** Removed in kind by the shared 8a function and the
  imported screen (§1.1). Drift between epochs is caught by §1.5's guard. Steps 1 to 7 are not
  shared code: `assemble.py` and `match_keys.py` each implement them, and §2 is their only common
  check.
- **R2, the raw literal stays in eval artifacts.** The writer objects in `work/`, `heldout/` and
  the committed `cpg/eval/dev/` keep the literal, by EP §9.2. They hold it from the proving repo's
  own source, which has the same readers, so this adds no exposure beyond the repository. Not
  changed. A literal that dot resolution removes (`/a/x7k2q9w4m1p8z3v6/../b`) is in neither leak
  probe, consistent with §1.4: the raw spelling is not a probe.
- **Q1 (`architect`, P1):** EP §2.2's `contract` row may cite spec3 §6 and §1.8 for `keyNorm`. Not
  blocking.
