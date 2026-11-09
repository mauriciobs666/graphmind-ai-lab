# Review — fact-bundle spec `-bundle-spec5.md` (`bindParts`, v1.1)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Pass 1 — 2026-11-09

### Scope & verdict

**Reviewed:** `docs/plans/connected-code-graph-bundle-spec5.md` (uncommitted), the gate of unit P1-13. The baseline is
step `## P1-13` of `docs/plans/connected-code-graph-p1-extraction.md` (with P1-14, which implements it, and P1-07/P1-17/
P1-18/P1-20 where the spec relies on them), index D4 (`docs/plans/connected-code-graph-p1.md`), SA-4(b)/P2-6 of
`docs/reviews/connected-code-graph-bindvalue.md`, the extended specs `-bundle-spec3.md` and `-bundle-spec4.md` (including
their new `Extended by:` lines), `cpg/ccg/bundle.py`, `cpg/ccg/schemas/bundle-v1.schema.json` and `cpg/tests/test_bundle.py`
at HEAD, and root `REVIEW.md` (`cpg/REVIEW.md` does not exist).

**Verdict: needs changes.** There are no blockers. Three majors: (M1) the `bindKeyCheck` refusal has no defined outcome
on the path it guards and no step owns it; (M2) the normative vectors leave four validator conjuncts uncatchable; (M3,
against `-p1-extraction.md`, not this spec) P1-07's `valueHash` formula contradicts the frozen §1.7.5. Each fix is small.
Every vector is correct, and the design meets P1-13's scope.

CPG: considered, not relevant — a contract review; no loaded graph indexes `cpg/` (`GRAPHS` lists only `cpg_falkorchat`), and the few code facts were read directly from `bundle.py`, the schema and `test_bundle.py`.

**Checks run (verified):**
- An independent implementation in the scratchpad (`vec.py`, `bp.py`, `bk.py`, `bl.py`; Appendix A). It parses every
  §9 table from the spec's text and recomputes all 25 EH, 38 BP, 32 BK and 15 BL rows: **0 mismatches**.
- EH-1, EH-14, EH-24 and EH-25 also reproduced with `openssl dgst -sha256 -mac HMAC -macopt hexkey:…`.
- Varied definitions are rejected: no `0x00` separator, or the hex text used as the key, both give different hex. BP-34
  has 33 path segments, but the total `/` count is 35, so the vector needs the §3 step 4 definition, and only that one
  matches.
- Mutation runs: Appendix B.
- `cpg` suite slice `tests/test_bundle.py tests/test_ids.py tests/test_answer.py`: 1873 passed. The `Extended by:`
  header edits break no spec-text parser.

### Findings

#### M1 (major) — `bindKeyCheck`: the refusal outcome is undefined on its own path, and no step owns it

§5.3 says the assembler refuses a bundle whose `bindKeyCheck` differs, "with the outcome `-spec3` §8.6 gives",
`stale-kept`.
- **The outcome loops.** P1-20 (1) defines `stale-kept` as "keeps the unit's last good bundle". The only realistic
  mismatch is after a key rotation (P1-02(d)) whose full rebuild failed for that unit. The kept bundle was then written
  under the *old* key, so it fails the same check. A faithful implementer has no rule for that case. They will either
  crash, load nothing silently, or compare hashes across keys, which is the case §2 says must "fail loudly".
- **No owner.** The §8 table has no assembler row. P1-18's behaviours do not include the check, P1-18 does not depend
  on P1-14, and P1-14's Files list has no assembler module.

*Verified by trace* (§5.3, `-spec3` §8.6, P1-18, P1-20 (1)). The failure outcome is inferred.

**Fix (design → `architect`):**
1. Define the outcome when the kept bundle also mismatches. One option: the unit is `stale-kept` and its `bindParts`
   are never compared. Its URL entries bind as a `1.0` `bindValue` would (§7 "Which parts"), or not at all. State
   which, and the report line that names `bindKeyCheck`.
2. Add a §8 row for `cpg/ccg/assemble/` (P1-18) or the linker, and the index dependency on 14.
3. Add a test: a bundle planted under a second test key (positive control: the same bundle under the §9 key is
   accepted).

#### M2 (major) — four validator conjuncts survive mutation, so a §6.4 check can be omitted with every vector green

I deleted one conjunct at a time from the independent validator. Four deletions left all 32 BK rows passing
(Appendix B):
- §1's "a `"1.1"` bundle with none" (masked by (1) in BK-5);
- (1)'s converse, `bindKeyCheck` ⇒ `bindParts`;
- (4)'s `service` ≠ `localhost`;
- (5)'s `host == H`. BK-9 changes the host to one outside the loopback class, so (4) rejects it first.

The boundary positives are also missing: 32 segments or 32 `pathHmacs` items valid, port 65535 valid, a 16-character
non-token `scheme` valid. The mutants `>31` and `<65535` survive.

Validator rules are fixed in both directions (`-spec3` §1.10). A check that P1-14 omits and ships becomes a loosened
rule, and only a major bump can restore it.

**Fix:** don't just add the four rows. Require the probe: §6.4 numbers each conjunct (1a, 1b, 4a…, 5a…). Every
reject row's rule cell names the **one** conjunct it violates while all the others hold, and every conjunct has such a
row plus a boundary positive where it has a bound. P1-14 runs a mutation pass, deleting each conjunct in turn: every
deletion must turn at least one BK row red. The four survivors above are this pass's evidence that the probe is
needed. They are not the list that closes the gap.

#### M3 (major, in `-p1-extraction.md` P1-07, not in this spec) — `entry_hmac(key, "valueHash", v)` contradicts §1.7.5

The spec's §10 flag is right. P1-07 line 159 passes the constant `"valueHash"` as the domain. Under §2/`-spec3` §1.7.5
the domain is the **record's own 32-hex id**. With a constant domain the hash is equal across `keyPath`s, so the
same sentence's test ("differs across `keyPath`s") cannot pass, and an implementer who "fixes" the test instead
breaks the frozen field's semantics (SA-11). Line 152 already says "per-entry domain separation", so P1-07 contradicts
itself. *Verified by reading.* P1-07 line 159 should read:

> (§1.7.5) `valueHash = entry_hmac(key, r["id"], text)`, where `r["id"]` is the `configEntry` record's own 32-hex id
> (`ids.config_entry_id(repoId, path, keyPath)`) and `text` is the value as the format's unescaping yields it.
> `launchFact.defaultHash = entry_hmac(key, <the launchFact's id>, <default text>)`. The same value under two
> `keyPath`s gives two hashes. `entry_hmac` reproduces `-bundle-spec5.md` §9.1, EH-25 for the id-domain form, with
> expected hex read from the spec text. The key is read only through `ccg.atrest.key_path` (P1-02).

Citing EH-25 needs P1-13 accepted before P1-07 is dispatched. P1-13 has no dependencies, so adding `13` to P1-07's
row is cheap. Whether to add it is `teco`'s sequencing call.

#### m1 (minor) — an IPv4-mapped IPv6 host makes `H`, and therefore `hostHmac`, depend on the Python version

§3 sets `H = "[" + str(IPv6Address(inner)) + "]"`. `cpg/pyproject.toml` allows `>=3.12`. On 3.12,
`str(IPv6Address('::ffff:7f00:1'))` is `::ffff:7f00:1` (*run*). On 3.13 the textual form of IPv4-mapped addresses
changed to dotted-quad (*inferred* from CPython gh-123409, which cites "the new improved textual representation of
IPv4-mapped IPv6 addresses in 3.13"). That has two effects:
- A 3.13 writer emits `host` with a `.`, which §6.4 (4) rejects.
- `hostHmac` differs across interpreters, so two bundles of one system never match.

**Fix:** in §3's bracketed branch, return `none` when `IPv6Address(inner).ipv4_mapped is not None`. That attribute
behaves the same on both versions. Add a BP row (`http://[::ffff:7f00:1]:80/` → `none`). This is a writer rule, so
it is loosenable later.

#### m2 (minor) — sentences of `-spec3` that `bindParts` falsifies are left unamended

§2 amends one sentence of `-spec3` §1.7.5. The next sentence, "Binding uses clear fields and env-var **names**", is
now false, because the linker binds `external` hosts and every path by hash (§7). `-spec3` §5.17's `bindValue` row
("URL, env and Compose binding read it") is also now partial (W6). Root `REVIEW.md`: a contradiction left in
unchanged text is a finding.

*Verified:* `git diff` shows `-spec3` untouched apart from its header pointer, so amending by citation is the only
route. That route is correct: `-spec4` §0, and `-spec3` §8.6 ("a minor version is written into a revised or
successor spec").

**Fix:** extend §2's amendment bullet to these two sentences.

#### m3 (minor) — §1's "every later minor" rule is misstated and too narrow

The text says "`"1.n"` is declared iff the bundle uses a field or record type first admitted in `1.n`". A bundle that
uses both a 1.1 field and a 1.2 field then satisfies the iff for both `"1.1"` and `"1.2"`. The rule also ignores
§8.6's other minor additions (a new reason code, a new contract kind), which are enum values, not fields.

**Fix:** "declares the maximum, over every field, record type and enum value it carries, of the minor that first
admitted it" (the first bullet's "lowest minor that admits every field" already says this; make the generalization
match it).

#### m4 (minor) — P1-14's step text doesn't cover the §8 table

§8 places code in `cpg/ccg/valuehash.py` (`bind_parts`, `bind_key_check`, `mount_prefix_hmac`, the tags),
`cpg/extractors/common/writer_rules.py` (`bind_parts_allowed`) and `cpg/tests/test_answer.py`. P1-14's Files list
names none of them, and P1-07, which owns those modules, runs before the spec exists.

W6 also changes the writer outcome P1-11 ships: `ip`/`external` URLs lose `bindValue`. Any P1-11 test that expects
that outcome changes in P1-14, and neither text says so.

**Fix:** add the three files (and their tests) to P1-14's Files line, plus one clause in §8's `formats.py` row about
P1-11's changed expectations.

#### n1 (nit) — wording and small gaps

- §6.2's first bullet ("decided by `-spec3` §1.7 alone") reads against W6 two lines later. Say W6 is an instance of
  W4.
- §7 does not say that the system-wide `s` is ASCII-lowercased, as §4's set is. `H` is lowercased, so an unlowered
  `container_name` would never match.
- P1-17 (4) says "with `bindParts` the comparison is on `hostHmac`/`pathHmacs`". §7 uses the clear `host` for three
  kinds. Align P1-17 by citing §7.
- §6.3 forbids `bindParts` in "a log line", but `bundle.py:905` quotes the offending `bindValue` in its `BundleError`.
  Say whether a validator error may quote a `bindParts` field. Recommended: name the field, never its value.

#### n2 (nit) — growth-stops and single-home

This is a new document, so there is no revision growth to judge, and it has no finding→resolution table. There are
small restatements:
- E2 restates W1's definition (cite it).
- §6.4 (4) restates §4's clear classes (cite §4's table).
- §10's "Flag for `teco`" is a process note in a normative spec. Remove it when P1-07's line is fixed, or it becomes
  stale history.

### Judgments the brief asked for

- **P1-13 scope:** met item by item (field shape, own domain tags, `host` only for three kinds, the minor bump rule,
  the keep-`bindValue`-only-if-§1.7 rule, never rendered, the linker comparison, the vectors, and the `bundle.py`/
  schema/`ids.py`/graph-note changes).
- **Implementable from the text:** yes, apart from M1's outcome. I implemented §3, §4, §6.4 and §7 from the prose
  alone, without reading the architect's figures first, and got every row. The cell and mutation syntaxes parse with
  single regexes (Appendix A).
- **Version rule vs `bindKeyCheck`:** consistent. `bindKeyCheck` is present iff `bindParts` is, so "1.1 iff
  `bindParts`" equals "lowest admitting minor". The gaps are only M2's untested directions and m3's generalization.
- **W6 vs the older rules:** sound. Every §1.7.2 `url` parses under §3 (its grammar is a subset), so W6 reaches every
  URL `bindValue`. It only withholds, so it is a writer-side tightening (`-spec3` §1.10). The validator outcome of every
  BV row is unchanged. The stakeholder's internal-DNS-in-clear allowance survives for `hostport` (BV-27), and no P1
  linker rule binds a clear dnsname URL, so no binding is lost.
- **Prefix match vs external host comparison; optional `port`:** independent and consistent. BL-13–BL-15 exercise the
  host path, and BK-13/BK-14 with §6.4 (5) pin port presence.
- **The §1.7.5 sentence edit:** made in `-spec5`'s text only. `-spec3`'s bytes are unchanged except the
  `Extended by:` pointer, which is permitted metadata. Amending a sentence by citation is the established reading rule,
  and the sentence is about the assembler, not a validator rule, so no `1.0` behaviour moves. See m2 for what it
  missed.
- **`header.bindKeyCheck`:** justified and proportionate. It is a cheap, loud guard against the one silent failure
  keyed equality introduces (cross-key comparison binds nothing). It reveals nothing beyond what `valueHash` already
  gives (a known-input MAC under a 32-byte random key). It is a small self-added scope step, flagged in the opening
  paragraph. It is acceptable once M1 defines its outcome and owner. The cost is one more field frozen in both
  directions.

### What's solid

- The domain-tag design (`.` in every tag, so it can never equal a record id) and prefix hashes in `pathHmacs`. One
  comparison gives the O-5 prefix match, and no segment matches out of position.
- §3 closes userinfo, query, fragment, percent-encoding, interpolation and the backslash differential with one
  character rule, and avoids WHATWG IPv4 shorthand by construction.
- The validator, which holds no key, still pins `bindParts` against a co-present `bindValue` (§6.4 (5)).
- The §9.2 commentary column states, for every row, which rule decides it.

### Open questions

- None that block. M1's outcome choice is the architect's.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*. Source: M2, plus the single-fault principle already used in
`docs/reviews/connected-code-graph-bindvalue.md` (SA-12).

- Addition: "Normative validator vectors are mutation-checked: deleting any one conjunct of a validator rule turns at
  least one vector red, and each bounded conjunct has a positive at its bound. A conjunct that a sibling check masks
  in every vector is a finding."

No line in root `REVIEW.md` is now enforced elsewhere, so no prune is proposed.

## Pass 2 — 2026-11-09

### Scope & verdict

**Re-gated:** `-bundle-spec5.md` as revised in place (uncommitted, 7,070 words); the `-p1-extraction.md` diff at
P1-07/P1-14; the index's P1-07 dependency row; `cpg/infra/README.md` § Bundle-at-rest (uncommitted, read only for
S5-6); `-p1-graph.md` G4/G13/§3, read for cross-checks. Spec5 now cites the security advisory
`docs/reviews/connected-code-graph-bundle-spec5-security.md` (S5-n IDs), so I read that too.

**Verdict: needs changes.** All seven Pass 1 findings are fixed. One new **major** (P2-1): BP-39–BP-41 contradict the
rules as written, and W7's "holds by construction" claim is false since the range exclusion landed. The fix is one
sentence.

CPG: considered, not relevant — a contract re-gate; no loaded graph indexes `cpg/` (Pass 1's `GRAPHS` check), and the code facts were read and run directly.

**Checks run (verified):**
- My Pass 1 implementation, updated only for the revised rules: host-scoped path input, §3's excluded ranges, the
  `#32` cell, the V-conjunct table, and the dotted-name rule in §7. Scripts are in `scratchpad/p2/`.
- All 31 EH rows: 0 mismatches. `openssl` reproduces EH-18, EH-26 and K2's check.
- BP: 46 rows, 3 mismatches (BP-39–BP-41, P2-1).
- BK: 48 rows, 0 mismatches. Every reject row fails **exactly** the one conjunct its rule cell names.
- BR: K2's `bind_key_check` matches the spec.
- BL: 16 rows, 0 mismatches.
- One-rule-at-a-time deletion over all 30 conjuncts V1a–V6f: **every deletion turns at least one BK row red**
  (Appendix C).
- Two linker mutants: dropping the host from the path input turns BL-1–BL-5, BL-11 and BL-13 red; admitting dotted
  names turns BL-16 red.

### Pass 1 dispositions

- **M1** — fixed. §5.4 loads the bundle instead of refusing it, so the loop is gone. BR-1–BR-3 pin the outcome.
  P1-14 supplies `bind_parts_usable` and P1-17 owns the report line. Residual: P2-3.
- **M2** — fixed. A 30-conjunct table with single-fault rows (verified 48/48). Bound positives are BK-14, BK-19,
  BK-20, BK-25, BP-45 and BP-46. Every deletion is caught. Residual: P2-4.
- **M3** — fixed. P1-07 carries the Pass 1 wording verbatim, with EH-25 renumbered to EH-31. `Needs: P1-13` is added,
  and index row P1-07 now reads `05, 02, 13`.
- **m1** — fixed by range (§3, V5e, BP-39–BP-41, BK-34/35). It introduced P2-1.
- **m2** — fixed: §0's list amends all five `-spec3` sentences.
- **m3** — fixed: §1 is now "the maximum … field, record type and enum value".
- **m4** — fixed: P1-14's Files list adds `valuehash.py`, `writer_rules.py`, `test_answer.py` and the P1-11 tests
  that W6 changes. Residual for W7: P2-3.
- **n1** — fixed, all four parts: W6 is now "an instance of W4"; §7 lowercases `s`; §7 opens by reading P1-17 (4);
  §6.3 gives the message rule and its test.
- **n2** — partly fixed. E2 cites W1, §4 is declared "the one home", and the `teco` flag is gone. New restatements:
  P2-6.

### Same-revision cross-checks (brief)

- **S5-2 host-scoped paths vs the prefix match and the BL rows: consistent.** The linker always hashes the mount under
  the consumer's decided `H`, and a `hostHmac` match makes `s == H`. So host scoping never splits a match P1-17 (4)
  needs, including the 8000/8200 `localhost` deployments. The one gap is a missing vector (P2-4).
- **M1 "load, don't refuse" vs `bind.py` and P1-20 `stale-kept`: sound.** A stale bundle under the old key loads.
  Its clear `url` entries still bind through derived parts, and only the parts-only entries go unbound. P1-18 needs no
  change. The ownership of the report line is incomplete (P2-3).
- **V3 (S5-4) vs W6/W7 vs P1-11's tests: consistent.** The validator's name rule equals E4's non-alias half, and
  `bindValue` already required it. W7 changes P1-11's expectations (P2-3) and W6 changes them too; P1-14 names only
  W6.
- **S5-5 single-label names vs the `service` kind: consistent.** Both sides admit only single labels, lowercased. One
  thing is unstated: whether `s` must pass the token screen (P2-5).
- **V5e vs `-spec3`'s frozen `ipv6` check: consistent for `bindParts`.** An IPv4-mapped `ip` host is rejected on
  every patch level: by V5d where `str()` prints it dotted, by V5e elsewhere. The frozen `bindValue` check still
  admits all three excluded ranges on this interpreter. That is what breaks W7 (P2-1).

### New findings

#### P2-1 (major) — BP-39–BP-41 contradict the rules, because W7 no longer "holds by construction"

The vectors say `bindValue` is **absent** for `http://[::ffff:7f00:1]:80/`, `http://[::7f00:1]/` and
`http://[64:ff9b::7f00:1]/`. But §6.2 makes `bindValue`'s admission "`-spec3` §1.7's, unchanged … and W6", and W6
applies only "when `bindParts` is emitted". It is not emitted here, because §3 returns `none` for these ranges. §1.7
admits all three on this interpreter: shipped `bundle.bind_value_allowed('x.url', v)` returns `True` for each
(Python 3.12.3; *run*).

So the rules give `bind`, the vectors say `absent`, and my implementation follows the rules. W7's justification,
"§1.7.2's `url` is inside §3's domain", has been false since §3 excluded ranges that §1.7.2's `ipv6` admits. A writer
that follows the text keeps a clear `ip` URL, path included, with no parts: the W6 bypass W7 exists to prevent.

**Fix:** state W7 as a rule, not a property. "A `url`-form `bindValue` is emitted only if `bindParts` is emitted for
the entry; otherwise the writer withholds it (W4)." Then delete the "by construction" sentence. BP-39–BP-41 then
follow, and P1-14's W7 test over §9.2 becomes able to fail (today it would pass vacuously on these rows, or fail
against the spec).

#### P2-2 (minor) — the test-key refusal has three homes that disagree

- Spec5 §2: `ccg.valuehash.system_key` refuses both §9 keys "unless … `allow_test_key=True`".
- `cpg/infra/README.md`: `atrest.hmac_key` refuses the first key only, says "There is no test-harness bypass", and
  cites its check as EH-24, which is now EH-30.
- P1-07: the key is read through `ccg.atrest.key_path`, and P1-07 runs before `system_key` exists (§8: P1-14 adds it).
  So `valueHash` would bypass the refusal.

No test needs `allow_test_key`: §8 says the conformance tests pass the key to `entry_hmac` directly.

**Fix:** make one home, `atrest` (P1-02, `devops`), refusing **both** §9 keys with no bypass. §2 then cites it and
drops `system_key`/`allow_test_key`, and P1-07 and P1-14 read the key only through it. The README's EH-24 → EH-30 goes
to the separate devops gate.

#### P2-3 (minor) — W7's effect and the key-epoch report line are not carried into the steps

- **W7 adds `bindParts` to every `url` entry that P1-11 emits**, so those bundles become `"1.1"` with `bindKeyCheck`.
  P1-14's Files line names only W6's change to P1-11's tests.
- **The report line has no owner in P1-17's text.** P1-17's step text was not edited. Its interface,
  `link(facts, bundles) -> Links`, has no channel to P1-20's `RefreshReport`. And §8 assigns all of §9.4 to
  `test_bindparts.py` (P1-14), although BR-2/BR-3's binding outcomes and report line need the P1-17 linker.

**Fix:**
- P1-14 Files: add "W7 adds `bindParts` (and `"1.1"`, `bindKeyCheck`) to P1-11's URL entries".
- P1-17 Behaviours: one clause, "§5.4: `bind_parts_usable`, derived parts, and `bind-key-mismatch` returned to the
  refresh report".
- §8: split §9.4 into a "usable" column (P1-14) and outcomes (P1-17).

#### P2-4 (minor) — two clauses that a sub-mutant can drop

- **No validator row has the IPv4-mapped range.** BK-34 and BK-35 cover the IPv4-compatible and NAT64 ranges, so a
  V5e that omits `::ffff:0:0/96` passes every BK row. That is the exact class S5-3 is about. On a patch level that
  prints the address dotted, V5d also rejects it. The rule cell can say "V5e (V5d where it prints dotted)", and the
  outcome is reject on every interpreter.
- **The path hash under a matched `s` is never exercised.** BL-14 has no path. Add `http://worker:8000/mcp` with
  `service=worker`, `mount=/mcp` → match, with the expected hex for `worker/mcp`.
- **V5c's class members are untested beyond `127.0.0.2`.** Dropping `[::1]` or `host.docker.internal` from V5c
  survives the BK rows.

The probe this calls for is the M2 one, extended. Every rule that is a list of members or ranges gets a reject or
positive row per member, on top of the per-conjunct deletion.

#### P2-5 (nit) — §7's `s` pattern: with or without the token screen?

§7 says "§1.7.2's single `label` pattern". §1.7.2's `label` includes the screen, while §3 deliberately uses the pattern
"**without** the token screen". So a declared service whose name fails the screen is matchable in one reading and not
in the other. Say which. I recommend "without": a match reveals only equality with a name that is already declared.

#### P2-6 (minor) — growth-stops

The document grew +1,483 words (5,587 → 7,070).
- **New scope:** the V-table, §5.4, §6.3's failure paths, the ranges, the key and where-keyed-work-runs bullets,
  BK-33–BK-48, BP-39–BP-46, BR, and BL-13/BL-16. Each adds a normative obligation that a finding required.
- **Rework growth and restated facts, to remove or cite:**
  - **(a) Three copies of one plan instruction.** "P1-11's tests … change with P1-14" appears in §6.2, in §8's
    `formats.py` row and in P1-14. Keep it in P1-14 and the §8 row only.
  - **(b) W7's test is stated twice**, in §6.1 and in §8. Keep §8.
  - **(c) §10's "Open for `devops`" bullet is answered** (below), and its own instruction says to fold it in and
    remove it.
  - **(d) §10's "Open for `graph-dba`" is answered.** `-p1-graph.md` §3 rules "the highest minor loaded", with
    per-unit G4. Cite it in §8 item (4) instead of "recommended", once that note is gated.
  - **(e) §10's "Outside this spec" bullet is a finding against frozen `-spec3`.** Its home is a cpg backlog item
    (root `docs/BACKLOG.md`); a one-clause pointer may remain.
  - **(f) §2's "Where keyed work runs" restates a custody fact** whose home is `cpg/infra/README.md` § Bundle-at-rest.
    Cite it.

### Ruling on S5-6 (spec5 §10, key custody)

Spec5 §10's "Open for `devops` (P1-02)" bullet should be **deleted**, not reworded. Both of its points are answered
by `cpg/infra/README.md` § Bundle-at-rest:
- the key is at `~/.config/ccg/keys/<system>.hmac`, outside the repo, which agrees with `-spec3` §5.17;
- it is 32 bytes from `secrets.token_bytes`.

The one fact spec5 needs goes into §2's Key bullet, by citation:

> The per-system key of `-spec3` §5.17: 32 bytes, held outside the repository per P1-02
> (`cpg/infra/README.md` § Bundle-at-rest), and read only through `ccg.atrest`, which refuses both §9 test keys.

This ruling is conditional on the devops gate of that README passing.

`-p1-extraction.md` P1-02 (d)/(e) (lines 59–60) still say `cpg/.ccg-work/keys/`. That is the same stale fact in a
second home. The P1-02 step's owner should update it in the same change.

### Proposed REVIEW.md lines

Pass 1's proposal stands unchanged.

## Pass 3 — 2026-11-09

### Scope & verdict

**Re-gated:**
- `-bundle-spec5.md`, revision 3 (uncommitted, 6,955 words);
- the `-p1-extraction.md` diff at P1-07 and P1-14;
- `-p1-linking.md` P1-17 clause (13);
- index row P1-07.

I read `cpg/ccg/atrest.py` (uncommitted, `devops`) only to check what spec5 relies on. I did not judge it, nor
`cpg/infra/README.md`.

**Verdict: approve with suggestions.** Every Pass 2 finding is fixed. One new minor (P3-1) and two nits remain.

CPG: considered, not relevant — a contract re-gate; no loaded graph indexes `cpg/`.

**Checks run (verified):** my own implementation again, in `scratchpad/p3/`.
- **What I changed in it:** W7 as a withholding rule, BL-17's compound provider side, and the new rows.
- **EH:** 32 rows, 0 mismatches.
- **BP:** 47 rows, 0 mismatches. Turning W7 off reddens BP-39–BP-41, so the rows now follow from the text.
- **BK:** 52 rows, 0 mismatches. Every reject row fails exactly the conjunct its rule cell names. BK-49 is
  single-fault on 3.12.3.
- **BR:** the second key's `bind_key_check` matches.
- **BL:** 17 rows, 0 mismatches.
- **Deletion test, 30 conjuncts:** every deletion reddens at least one BK row (V5e now reddens BK-34, BK-35 and
  BK-49).
- **Pass 2's sub-mutants:**
  - dropping the mapped range from V5e reddens BK-49;
  - dropping `[::1]` or `host.docker.internal` from V5c reddens BK-50 or BK-51;
  - dropping V5d's IPv4 branch reddens BK-52;
  - allowing dotted names reddens BL-16 and BL-17.

### Pass 2 dispositions

- **P2-1** — fixed. W7 is now a writer rule, "emitted only if `bindParts` is emitted; otherwise withheld" (§6.1).
  BP-39–BP-41 now follow from it (rechecked above).
- **P2-2** — fixed in spec5. §2 makes `ccg.atrest.hmac_key` the one key path: it refuses both §9 keys, with no
  bypass. `ccg.atrest.bind_key_check` is "the one implementation", and `system_key`/`allow_test_key` are gone from
  §2 and §8. P1-07 now reads the key through `atrest.hmac_key`. The `devops` side is a dependency, not judged here:
  `atrest.py` on disk refuses only the first key. Residual: P3-1.
- **P2-3** — fixed.
  - P1-14's Files line names W7's effect.
  - P1-17 clause (13) owns the derived-parts outcome and returns the `bind-key-mismatch` lines beside `link()`'s links
    for P1-20.
  - §9.4 is split into a `usable` column (P1-14) and an outcome column (P1-17), and §8's linker and `test_bindparts`
    rows match that split.
- **P2-4** — fixed: BK-49–BK-52, BP-47/EH-32 and BL-17 (rechecked by the sub-mutants above).
- **P2-5** — fixed: §7 now says "without the token screen".
- **P2-6** — fixed. The document fell from 7,070 to 6,955 words while adding rows, so this pass's work replaced text
  in place.
  - (a)/(b): the P1-11 sentence and W7's test each have one home (§8, plus P1-14).
  - (c): §10's devops bullet is deleted, and its fact is cited in §2.
  - (d): the graph-dba open point now cites `-p1-graph.md` §3/G4.
  - (e): moved to the coordination ledger as a cpg backlog candidate (line 647); it is owed by `teco`.
  - (f): "Where keyed work runs" is deleted, and custody is cited to the README.

**Growth-stops:** passes. There is no rework growth, and no restated history or facts were found in this revision.

### Same-revision cross-checks (brief)

- **W7 withholding vs the P1-11 tests and V1a–V1d: consistent.** A W7-withheld entry carries neither field, so it
  never forces `"1.1"`. Any entry with parts forces `"1.1"` plus `bindKeyCheck`. One wording overstatement: n1.
- **`usable`/outcome owners vs P1-14/P1-17: consistent** across §5.4, §8, §9.4, P1-14's Key-epoch line and P1-17
  (13).
- **atrest as the single home vs spec5's own §8:** one stale line in P1-14 (P3-1), and a duplicated tag constant
  (n2).

### New findings

#### P3-1 (minor) — P1-14's Files line still lists the removed accessor in `valuehash.py`

P1-14 says `cpg/ccg/valuehash.py` (tags, **`system_key`**, `bind_parts`, **`bind_key_check`**, `bind_parts_usable`,
`mount_prefix_hmac`). Spec5 §2 and §8 now say:
- `system_key` does not exist;
- `bind_key_check` lives in `ccg.atrest` and is "the one implementation";
- `valuehash.bind_parts_usable` only uses it.

A P1-14 implementer who follows the Files line builds a second key accessor and a second check, the two homes that
P2-2 removed. *Verified by reading both texts.*

**Fix:** P1-14 Files should read `cpg/ccg/valuehash.py` (tags, `bind_parts`, `bind_parts_usable` using
`ccg.atrest.bind_key_check`, `mount_prefix_hmac`; spec5 §8). Add one test that no module other than `ccg.atrest`
defines `bind_key_check` or reads the key file. Its positive control is a planted second definition.

#### n1 (nit) — "every P1-11 URL entry" gets `bindParts` is overstated, and the linker case for a refused URL is unstated

P1-14 says "W7 plus spec5 §1 give every P1-11 URL entry `bindParts`". §8 says "its URL entries". But a URL that §3
refuses (the excluded ranges, a `_` host) gets **neither** field under W7. That changes P1-11's expectations too.

Add "or withhold its `bindValue` where §3 refuses it" to both. In §7 "Which parts", say that a `url`-form `bindValue`
that `parse_bind_url` refuses binds by nothing. Only a pre-P1-14 bundle can hold one, and today the rule is silent
on it.

#### n2 (nit) — cell syntax and constant homes

- **§9.5's compound provider cell needs a separator rule.** It is `service=s; mount=M`, split on `"; "`, while BL-12's
  mount `/mcp;v=1` holds a bare `;`. A parser that splits on `;` mis-reads BL-17 (run: BL-17 goes red). Say
  "tokens are separated by `; ` (semicolon, space)".
- **The `BIND_KEYCHECK_TAG` constant has two homes.** §8 puts "the three tags" in `valuehash.py`, while `atrest.py`
  defines its own `BIND_KEYCHECK_TAG`. Pick one home, with the other importing it, or add a test that both equal
  §2's text.

### Proposed REVIEW.md lines

Pass 1's proposal stands unchanged.

## Appendix C — Pass 2 deletion runs

`scratchpad/p2/bk2.py <id>` disables one conjunct:

| Conjunct | Rows that turn red |
|---|---|
| V1a | BK-4 |
| V1b | BK-5 |
| V1c | BK-6 |
| V1d | BK-7 |
| V2 | BK-8 |
| V3 | BK-9 |
| V4a | BK-10 |
| V4b | BK-11 |
| V4c | BK-12, BK-13 |
| V4d | BK-15 |
| V4e | BK-16, BK-17, BK-18 |
| V4f | BK-21 |
| V4g | BK-22 |
| V4h | BK-23 |
| V4i | BK-24 |
| V5a | BK-26 |
| V5b | BK-28 |
| V5c | BK-29 |
| V5d | BK-31, BK-32 |
| V5e | BK-34, BK-35 |
| V5f | BK-36 |
| V5g | BK-37 |
| V5h | BK-38 |
| V5i | BK-39 |
| V6a | BK-41 |
| V6b | BK-42 |
| V6c | BK-43, BK-44 |
| V6d | BK-45 |
| V6e | BK-47 |
| V6f | BK-48 |

No conjunct survives. This proves only that my decomposition is caught. An implementation that merges conjuncts
behind one early return can still mask one, which is why P1-14's own mutation pass (§8) is required.

## Appendix A — independent recomputation

Scripts, all under `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/a56a19ba-3e84-44da-9d76-47eef6534cc4/scratchpad/`:
- `vec.py`: HMAC and the EH rows.
- `bp.py`: the token screen, the §1.7.2 grammar, the name rule, W1, role, W2, §3, §4, W6.
- `bk.py`: §1 and the §6.4 validator, over the base bundle's fields.
- `bl.py`: the §7 mount and host matching.

Outputs: `EH mismatches: 0`, `BP diffs: 0`, `BK diffs 0`, `BL diffs 0`. The BK base reproduces only the `header`
and `configEntry` fields the mutations touch. `rich_bundle()`'s other records are outside §6.4.

## Appendix B — mutation runs

| Mutant | Rows that turn red |
|---|---|
| segment cap 32 → 33 | BP-34 |
| W6 removed | BP-6, BP-7, BP-9, BP-12, BP-18 |
| no trailing-`/` strip | BP-2, BP-18 |
| loopback IPv4 = `127.0.0.1` only | BP-13 |
| `host` emitted for `external` | BP-5–9, BP-15, BP-17 |
| (5) `len(pathHmacs)` check removed | BK-17 |
| (2) W1 removed | BK-22 |
| §1: only "1.0 with `bindParts`" rejected | **none** |
| (1): only `bindParts` ⇒ `bindKeyCheck` | **none** |
| (4): `service` may be `localhost` | **none** |
| (5): `host == H` removed | **none** |
