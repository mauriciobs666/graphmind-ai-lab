# Review — connected-code-graph-bundle-spec3 (consolidation of the bundle spec and `-spec2`)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (M10) · **Reviews:** `docs/plans/connected-code-graph-bundle-spec3.md`

## Pass 1

**Scope.** `docs/plans/connected-code-graph-bundle-spec3.md` v1.0 (untracked, 2,155 lines), the document half of S16 (`docs/plans/connected-code-graph-p0-2.md` §3 S16 row, §2 D5). Baselines: `docs/plans/connected-code-graph-bundle-spec.md` v1.3 and `docs/plans/connected-code-graph-bundle-spec2.md` v1.3. Root `REVIEW.md` applied; there is no `cpg/REVIEW.md`. Not reviewed: the code half of S16 (the `test_bundle.py` re-point), which has not run.

CPG: considered, not relevant — a document consolidation; the code facts were checked by running `cpg/ccg/bundle.py` and its suite, and no `cpg_*` graph covers `cpg/`.

**Verdict: approve with suggestions.** No blocker: every normative table, vector and regex the two base specs carry arrived byte-identical, and the seven authorised changes are present and execute correctly. One major (M1): an open follow-up (F-13's consumer list) was dropped with the history and now lives only in a ledger. Close M1 before `teco` accepts the document, because the ledger forbids dispatching S15 until `-spec3` is accepted.

### How I checked

Everything below was run unless marked *inferred*. Scratch: `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/13504717-4242-422a-b7d8-a910a6cc9348/scratchpad/`.

| Check | Result |
|---|---|
| `cpg/` suite, `cd cpg && .venv/bin/python -m pytest -q` | **1311 passed** in 11.36 s |
| Line diff of the bases' §2–§12 against spec3 §2–§12 (`base_vs_s3.diff`, 138 lines) and of `-spec2` §1 against spec3 §1 (`spec2_vs_s3.diff`, 442 lines); both read in full | Every hunk is one of: a version/dated tag removed, a pointer replacing a moved table, or one of the seven items, plus the items listed in M1/m1–m7 below |
| GV/AV/BV rows extracted from each file and compared as strings (`rows.py`) | GV 22/22 identical; AV 68/68, only AV-6/16/18/21 differ (a `(v1.2)`-style tag removed); BV 28/28, only BV-27/28 differ (same). IDs missing or extra: none |
| GV-1–22 recomputed from the tuples in spec3's table with my own hash (`gv.py`) | **22/22 match**, including GV-4=GV-3, GV-6=GV-5, GV-10 (5,032 bytes), GV-17≠GV-18 |
| Existing `test_bundle.py`/`test_ids.py` copied to scratch with `SPEC2_TEXT` pointed at spec3 (the table parsers read AV/BV/§1.x from spec3) | 1210 passed, **3 failed** (`_probe_counts` `KeyError`, see m5) |
| Same copy with `SPEC_TEXT` also pointed at spec3 | Collection error at `_spec_secret_tokens` (see m5) |
| Internal `§` references in spec3 | Every internal reference resolves to a heading; the non-matching ones are external (proposal, graph note, P0 plan, RFC) |
| `git diff` of the two base specs | One header line each, `Superseded by:` added; nothing else |

### Findings

#### M1 (major) — F-13's "no step may encode the old order" follow-up was dropped, and no durable copy remains

- **Evidence.** `-spec2` §3 row F-13 carried both the rule and its consumer list: "No step may encode the old order: S12's `http` rule and the shared 8a function, S35/S39 contract keys, the S40/S52 writer briefs, and the eval normalizer (K42 expects `GET /b`)". spec3 §13 keeps F-1, 3, 4, 6, 7, 9, 10, 11, 12 and drops F-2, F-5, F-8 and F-13. F-2 (closed), F-5 (this document) and F-8 (delivered by EP2, `docs/plans/connected-code-graph-p0-2.md` §16.1 BS2-21) are legitimately gone. F-13's *rule* is now normative text in §6; its *consumers* are an open obligation. `grep -n "F-13" docs/plans/connected-code-graph-p0-2.md` returns nothing, and p0-2 EP2-16 says only "do not encode" the order. The only other holder is `docs/plans/connected-code-graph-coordination.md:220-223`, a ledger.
- **The consumers do disagree with spec3 today** (grep of `step 5|step 6`): `docs/plans/connected-code-graph-eval-protocol2-ml.md` lines 70-72 (step 5 = collapse and dot resolution, step 6 = decode), 116-121, 271 (K13 "step 6 decodes"), 283-284 (K31/K32 "step 5's `..`"), 286 and 288 (K42 expected `GET /a/../b`, "pins the spec's order", exempt from the fixed-point rule), 357, 376, 395-400 (Q3 open). `cpg/tests/test_keynorm.py` lines 34, 57, 130, 149, 157, 173 use the old numbers. In spec3 step 5 is decode and step 6 is collapse-and-dots, so every one of these labels is now wrong. K42's expected value is also wrong. I ran both orders (`fuzz_keynorm.py`): old order gives `GET /a/../b` and is non-idempotent on 17,794 of 200,000 random inputs; the new order gives `/b` and 0 of 200,000.
- **Why it matters.** S45/S47/S49 build `normalize` and the K1–K42 oracle from EP2 as written; S49's oracle would pin the order F-13 exists to remove, and `keyNorm` is in `cpId`.
- **Fix.** (a) Add a `F-13` row to spec3 §13 listing the consumers above with the renumbering (decode = step 5, collapse and dots = step 6, K42 = `GET /b`, now a fixed point). It is an open follow-up with an owner, so §13 is its home, not history. (b) `teco` routes a note to EP2's owner (`data-scientist`) for the lines above; this needs EP2 revised in place, since it has not been executed against. (c) `tdd-engineer`'s S16 re-point renumbers the `test_keynorm.py` comments.

#### m1 (minor) — the header's "section numbers resolve here to the same clause" is false for five places

- **Evidence.** spec3 lines 6-8. Not carried: the spec's §13.1–§13.3 (S03 dispositions) and "Ready to implement"; `-spec2` §0, §2, §3, §4, §5. Worse, spec3 §13 is now "Follow-ups with owners", which was `-spec2` §3, so a citation of "spec §13.2" lands on a different table. Live citations exist: `docs/plans/connected-code-graph-p0-2.md` cites `-spec2` §2–§5 or "BS2 §2–§5" on 33 lines (grep); `cpg/tests/test_keynorm.py:7` cites "BS2 §3 F-13".
- **Fix.** Replace the sentence with the true statement: "§1–§12 of the spec and §1.1–§1.13 of `-spec2` resolve here; the spec's §13 and `-spec2` §0 and §2–§5 are not carried (history), and `-spec2` §3 is §13 here." The cited documents stay on disk, so the citations still resolve there; nothing needs rewriting.

#### m2 (minor) — the T4 paragraph states a condition that is wrong (verified by execution)

- **Evidence.** spec3 lines 586-593 say a UUID "whose last group runs on into more hex digits fails T3 too", and that T4 decides alone only when the last group is "followed directly by a non-hex letter". Counterexample (`t4.py`): `aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa1z` fails with T4 and passes with T4 disabled; so does `aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaaaz`. The last group is followed by hex and *then* a non-hex letter, which makes the run not all hex, so T3 does not fire. A 300,000-string fuzz found 4 T4-alone cases, 2 of which fall outside the paragraph's characterization. The spec's own example `…aaaaaaaaaaaaz` and its 3-character-group twin are right.
- **Why it matters.** Low: the rule is unchanged and T4 correctly stays (loosening is a major change, §1.10). But it is a new normative-looking sentence that a reader will use to decide whether T4 can ever be retired.
- **Fix.** State it as: T4 decides alone only when the alphanumeric run holding the last group is not all hex (a non-hex letter after the group, directly or after more hex digits), no T1/T2/T6 condition applies, and the string is not caught by T3. Add the `…1z` example.

#### m3 (minor) — finding IDs are used without a legend, and two reviews' `P3-1`/`P3-2` collide

- **Evidence.** The dropped `-spec2` §0 "Sources" list and §2 preamble were the only place that mapped `BV`, `S10`, `EP` to documents. spec3 now cites `SA-1…13`, `BS-*`, `P2-n`, `P3-n`, `G-n` and cites none of the source reviews except `docs/reviews/connected-code-graph-p0-impl.md` in the header (the bindValue advisory `docs/reviews/connected-code-graph-bindvalue.md` is not named anywhere). `P3-2` means the analyst's C# static-initializer ruling at line 1193 and the security reviewer's T6 separator class at lines 392, 575, 594, 616; `P3-1` is both the analyst's `ownedRegions` finding (lines 1609, 2025) and security's argv finding (ten lines, e.g. 247, 303, 468).
- **Fix.** Add a four-line legend after the header: prefix, source review path, section. Qualify the colliding IDs where they appear (`analyst P3-2` versus `security P3-2`). New scope, but a few lines.

#### m4 (minor) — carried-over text says the security advisory has not happened

- **Evidence.** §10 line 2101: "`configEntry.bindValue` (P1, after the `security-expert` advisory)". §12 BF8 (line 2133): "Recommend a `security-expert` advisory pass on §5.17 before P1 relies on it". The advisory ran (S58) and its dispositions are §1.7; §5.17 is now a pointer. Root `REVIEW.md`: a contradiction carried over from the previous text is a finding.
- **Fix.** §10: "`configEntry.bindValue` (P1; gated by §1.7) and `launchFact` (P1; §1.6)". BF8: replace the recommendation with "Done: §1.7 (S58)".

#### m5 (minor) — "Observed on" was not purely history: three tests parse it, and the S16 code half needs a hand-off

- **Evidence.** `cpg/tests/test_bundle.py:1855-1862` (`_probe_counts`) parses the removed table for `Grid`, `Separator axis` and `Transplant`; with `SPEC2_TEXT` on spec3, `test_probe_grid_has_no_value_in_clear`, `test_probe_separator_axis_has_no_value_in_clear` and `test_probe_bs_m1_transplant_leaves_only_the_bare_positional_residual` fail with `KeyError`. A full re-point also fails at `_spec_secret_tokens` (`test_bundle.py:1468`), because the 25 substring words now live in §1.7.3, not in the §5.17 row. The numbers survive in §1.6.3 as formulas (17,280; 6,480; 675; 85,536) and in "A green probe", so no normative content was lost; the parsers must change.
- **Fix.** Record in the S16 test-re-point brief: parse the pass-condition formulas from §1.6.3 (do not retype them), and read the substring list from §1.7.3. Expected values must come from the text, not from the code (root `REVIEW.md`).

#### m6 (minor) — F-10's writer obligation is stated twice and is untagged

- **Evidence.** §1.6.1 line 336 ("Writer obligation (F-10)") and §13 F-10 (line 2152) both state "`command` and `args` only through `ccg.bundle.gate_argv`, any further withholding writes the element's own sentinel". Growth-stops: this is new scope (item 7), but a fact has one home. The §1 tag legend (`[pre-freeze]`/`[writer]`/`[doc]`) is also not applied to it; it is a `[writer]` rule.
- **Fix.** Keep it in §1.6.1, tagged `[writer]`, and let the §13 F-10 row cite it ("§1.6.1") instead of restating.

#### m7 (minor) — step 6 is silent on a trailing dot segment, and it is id-bearing

- **Evidence.** §6 step 6 "Collapse runs of `/`; resolve `.` and `..` segments", with "a trailing `/` is **kept**". It does not say whether `GET /a/b/..` gives `/a` or `/a/` (RFC 3986's `remove_dot_segments` gives `/a/`). I ran both variants through the new order: each is idempotent over 200,000 inputs, so this is a definition gap, not an idempotence bug. But `assemble.py` (S12) and the eval `normalize` (S49) must agree on a `cpId` input, and both implement it from this one sentence.
- **Fix.** One worked example in the step-6 text (`GET /a/b/..` → `GET /a/` or `GET /a`, whichever the architect rules) and one oracle row.

#### nits

- n1. §1.12 says the `(?!\n)` guard "rejects nothing a spec-valid value can contain". The schema `path` pattern already rejects a path with an embedded or trailing newline, and §8.2 does not forbid one (checked: `a/b\nc.py` and `a/b.py\n` both fail). Either narrow the sentence to "trailing newline on the identifier-shaped patterns" or add "no control characters" to §8.2.
- n2. BV tables run BV-21–28 first, then BV-1–20 (lines 767-790), and §1.6.3 precedes §1.6.2; reorder in a consolidation. The sub-bullets under the 25 substring words (lines 686-690) are indented four spaces under a two-space parent.
- n3. §1.6.1's "the word `pass` inside `htpasswd` does not make `admin` a value" reads as if `admin` were being judged; say "does not make `admin` a secret-named context".
- n4. The `-spec2` SA-10 rejection ("HMAC-ing a keyPath segment would make `config` locators unwritable from source") is a rejected option with a reason that still binds; it survives only as a half-sentence in §1.11. One sentence under §5.17 or F-4 would keep it.

### The seven authorised changes

1. **F-13 (swap).** Present: §6 `http` step 5 = decode, step 6 = collapse and dots; §1.8 says 8a runs after both and "screens decoded segments". Executed (`fuzz_keynorm.py`, my own emulation of steps 5–6; `cpg/` has no steps 1–7 implementation, `ccg/keynorm.py` is 8a only): the worked examples hold: `GET /a/%2E%2E/b` → `GET /b`; `GET /a/%7Euser/%2f` → `GET /a/~user/%2F`. New order: 0 non-idempotent of 200,000 random strings (alphabet `/ . .. %2E %2e %2F %252E %25 a b %7E %41 %5C`), under both trailing-slash variants; old order: 17,794 and 18,036. Whole-`keyNorm` idempotence (steps 7 and 8a over the output) was not run; no reference exists yet. See M1 and m7.
2. **Depth example.** Verified with `bundle._path_head` (`kw.headers.a.b.c` valid, depth 3; `kw.headers["a.b"]["c d"]["é"]` valid, depth 3; the old `kw.headers["a"]["b"]["c"]` INVALID). Correct.
3. **Lookaround.** Verified on the schema: 16 patterns; 15 anchored ones end in `(?!\n)$`; `bindValue` does not; no lookbehind anywhere. **The architect's claim holds and the exemption is acceptable under "necessary, not sufficient":** `bindValue` `"8000\n"` is accepted by the schema under Python `re` (jsonschema 4.26.0), rejected by an ECMA-style `\Z` emulation, and rejected by `bundle.bind_value_allowed`. Suggestion, not a finding: adding the guard to `bindValue` costs nothing and removes the one pattern that gives two answers in two engines, which §1.12 says the guards exist to prevent.
4. **T4.** Kept as a rule; the redundancy claim is mostly right but see m2.
5. **Grid 85,536.** `grid.py` runs `gate_argv` over the §1.6.3 placements (11), words (36 parsed from §1.7.3), cases (3), suffixes (4), after one prefix entry: 18 representatives (spec3's list) → **85,536 cells, 0 leaks**; the `-spec2` "15" (12 exact + `-Xmx/-Xms/-Xss`) → 71,280 cells, 0 leaks, which reproduces `-spec2`'s old figure and shows its "15" omitted `-XX:±` and `-verbose:`; 21 entries (each `-verbose:` word separately) → 99,792 cells, 0 leaks, so one representative for `-verbose:` is sufficient. All 18 entries pass `gate_argv(["x", e]) == ["x", e]`. The definition ("one for each exact string and for each closed shape": 12 + 3 + 2 + 1) is correct, and `-spec2`'s "15" is correctly dispositioned.
6. **S10 notes.** m1 (input sentinels kept, AV-32 a fixed point), m2 (argv[0] supplies no context, `htpasswd admin hunter2` unchanged) and n1 (depth example) are all present. Executed: `gate_argv` on AV-32's output returns it unchanged; `htpasswd admin hunter2` unchanged; `x --<redacted> hunter2` → `x --<redacted> <redacted>`.
7. **F-10.** Present in §1.6.1 and §13; see m6.

### Growth-stops check

Combined bases 32,297 words; spec3 24,918 words (−7,379). The reduction is the history that was removed: dispositions, `Observed on` table, version tags, dated "run on" claims. A grep for `v1.[0-9]` and `2026-10` left only the header and the stakeholder-decision dates, which are facts. **New scope** (allowed): the seven items, the §1 scaffolding (tag legend, clause table), the pointer stubs. **Rework or restated facts** (findings): F-10 twice (m6); the `(§1.x)` pointer bullets in §5.1 restate §1.1's one-encoding sentence in a short form (acceptable, cites its home). The §1.11/§1.10/§1.12/§1.13 "amends/adds to" structure keeps two homes for the same row (§9.1 and §1.11), so a reader still reads two places for a record's graph mapping; I found no contradiction between them, so I report it as the same bounded cost `-spec2` §0 named, not as a defect.

### Citations and header

- Code and test citations of the bases (`BS2 §1.1`, `bundle-spec2.md §1.8`, `spec §5.1`, `bundle-spec.md §3`) all resolve to the same clause in spec3 because §1–§12 are preserved. The base-spec citations in `cpg/` will be re-pointed by S16's code half.
- Header: `Supersedes:` both bases, `Extends:` the proposal (inherited from the base, a different document, so the pairs are not mixed), `Version: 1.0`. The two bases gained `Superseded by:` and nothing else. The `Status:` flip to `superseded` is `teco`'s at acceptance.

### What's solid

All 22 GVs, 68 AVs, 28 BVs, every regex and enumeration in §1.1–§1.13 and §2–§12 came across unchanged except for tag removal. The depth example, the lookaround statement and the allow-list grid are correct and verified. Internal references resolve. The suite is green (1311). The history cut (−23 %) is clean: every dropped disposition that I checked has its rule carried in the text (`= delete`, the C# static ruling, the vendored-path test base, the `.h` known limit, the record accessor `qname`).

### Open questions

1. `docs/plans/connected-code-graph-p0-2.md` §3 S16 still describes a "change mode" in which `-spec3` encodes SL-7 deltas, while spec3 §1 and the ledger (`-spec3` before S15) make any SL-7 change a `-spec4`. `teco` should align the S16 row and D5 with spec3 §1.
2. M1's EP2 revision: does `data-scientist` revise EP2 in place (Status `active`, gated but not executed against) or via a note? Root `AGENTS.md` collision rule 5 decides; I did not check whether S45–S49 have started.

### Stopping condition (falsifiable)

Pass 2 closes the review when all of these hold:
- `grep -c 'F-13' docs/plans/connected-code-graph-bundle-spec3.md` is at least 2 and the §13 row names S12, S35/S39, S40/S52, S45/S47/S49 and EP2 with decode = step 5.
- No line of `docs/plans/connected-code-graph-eval-protocol2-ml.md` still gives `GET /a/../b` as K42's expected value, or a note routed from `teco` is on record.
- spec3 lines 6-8 no longer claim that all earlier section numbers resolve here.
- `python3 scratchpad/rows.py` still reports zero unexplained GV/AV/BV differences, and `gv.py` zero mismatches.
- m2's `…1z` string either fails with T4 alone and the paragraph says why, or the paragraph no longer states the condition.

## Pass 2

**Scope.** `-spec3` v1.1 (25,248 words, +330 over v1.0) and `docs/plans/connected-code-graph-p0-2.md` v1.3 (29,155 words, −26 against HEAD). EP2 is out of scope (revised in parallel by `data-scientist`).

**Verdict: approve with suggestions.** All Pass 1 findings are fixed except M1's EP2 half, which is correctly parked. One new minor in p0-2 (N1) and one nit (N2); neither blocks acceptance of `-spec3`.

**Re-run (executed):** `cpg/` suite **1311 passed**. `rows.py`: GV 22/22 identical, AV/BV 68/68 and 28/28 with the same six tag-only differences as Pass 1. `gv.py`: 22/22 recomputed. `grid.py`: 18 entries → 85,536 cells, 0 leaks. The two base specs still differ from HEAD by one `Superseded by:` line each.

### Dispositions

- **M1** (F-13 consumers): fixed in `-spec3`. §13 now has the F-13 row (line 2167) naming S12/P0-11a, S35/S39, S40/S52, EP2 (numbering, K13, K31/K32, K42 = `GET /b`, Q3), S45/S47/S49 and `test_keynorm.py`, owner and deadline "before S15's freeze record". p0-2 S16 row carries the same list and the renumbering (decode = step 5). EP2's edit is `data-scientist`'s, not gated here.
- **m1** (header claim): fixed; the header now lists what resolves and what is not carried, and says `-spec2` §3 is §13 here. Matches my grep.
- **m2** (T4): fixed. I re-derived it: `…aaaaaaaaaaaa1z` fails with T4 and passes without (`t4b.py`); the 3-character-group twin fails neither; `550e8400-e29b-41d4-a716-446655440000z` fails with and without T4 (T6), as the text says. A 400,000-string fuzz found 103 T4-alone strings, 0 outside "the run holding the last group has a non-hex letter after it".
- **m3** (IDs): fixed by a legend in the header. It maps unprefixed `P2-n`/`P3-n`/`SA-n`/`BV-n` to the security advisory and "analyst"-marked ones to S03; the C# `P3-2` is marked "analyst" in the text.
- **m4** (stale advisory): fixed. BF8 reads "Done: the advisory (S58) ran, dispositions are §1.7"; the §10 "after the advisory" phrase is gone (grep).
- **m5** (probe parsers): fixed as a hand-off in the p0-2 S16 row (parse §1.6.3 formulas, read the substring words from §1.7.3). Not yet executed; it is the code half.
- **m6** (F-10 twice): fixed. §1.6.1 holds it, tagged `[writer]`; the §13 row cites it.
- **m7** (trailing dot): fixed and consistent. Step 6 now follows RFC 3986 §5.2.4; examples `GET /a/b/..` → `GET /a/`, `/a/b/.` → `/a/b/`, `/a/..` → `GET /`. My emulation gives `/a/`, `/a/b/`, `/` and each result is a fixed point; the earlier §6 examples (`/a/%2E%2E/b` → `/b`, `/a/%7Euser/%2f` → `/a/~user/%2F`) still hold. The row does not contradict the retained "a trailing `/` is kept".
- **Nits:** n1 fixed (the guard is scoped to identifier-shaped patterns, line 910). n2 not fixed (§1.6.3 still precedes §1.6.2 and BV-21–28 still precedes BV-1–20, lines 497/552 and 777/790); cosmetic, dropped. n3 fixed (line 267). n4 fixed (line 2169, "Why `keyPath` segments are not HMAC'd").
- **Open question 1** (S16 change mode): fixed in p0-2 D5, the S16 row and the S15 gate text (`-spec4` for an SL-7 change).

### Same-revision cross-checks

F-13 row, §6 step 5/6, §1.8 ("screens decoded segments") and the p0-2 S16/D5 text all agree: decode = step 5, collapse and dots = step 6, K42 = `GET /b`. The T4 text, the legend and the BF8 edit do not touch each other. Growth: `-spec3` +330 words is new scope (F-13 row, legend, trailing-dot examples, T4 rewrite, rejected-option note); the F-10 duplication is gone. p0-2 shrank by 26 words; its edits replace in place, with a one-line revision note.

### New findings

- **N1 (minor), p0-2 does not make S16 a dependency of S15.** The new rule is "S15 is not dispatched until `-spec3` is accepted", but the S15 row's Dependencies column is still `S14` (§3 line 380). The dependency column is what a scheduler reads. Add `S16`. Falsifiable: `awk -F'|' '/^\| S15 \|/{print $8}'` shows S16.
- **N2 (nit), leftover "mode" wording in p0-2.** §5.2 line 888 ("in both modes"), §9.2 line 1343 ("in whichever mode it ran"), §15 line 1561 ("`-spec3` (S16, or before P0 exit; required)"), and the §5.3 row for S15 (line 910) which lists S16 among the steps that wait on S15's decision, although S16 now precedes S15. The §16.1 traceability rows (1740, 1746) are history and fine. Reword to "S16 always runs before S15".

### Stopping condition

Met for `-spec3`: F-13 row present with the consumer list; header claim corrected; `rows.py` and `gv.py` report no unexplained difference; the T4 paragraph is true on 103 of 103 fuzzed cases. Not met for EP2 (K42 still to be revised), which is outside this gate. N1 should land with `-spec3`'s acceptance.
