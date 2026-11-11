# Review — P1-14 `bindParts` encode (uncommitted diff)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope and verdict

Reviewed: the uncommitted P1-14 diff (`cpg/ccg/bundle.py`, `cpg/ccg/schemas/bundle-v1.schema.json`, `cpg/ccg/valuehash.py`,
`cpg/extractors/common/writer_rules.py`, `cpg/extractors/config/extract.py`, tests `test_bundle`, `test_writer_rules`,
`test_config_extractor`, `test_answer`, new `test_bindparts.py`) against the step `docs/plans/connected-code-graph-p1-extraction.md`
§ P1-14 and `docs/plans/connected-code-graph-bundle-spec5.md` §1–§9. Read root `REVIEW.md` (no `cpg/REVIEW.md` exists).
Not looked at: the other modified files in the tree (sandbox, decision-models, plans), `launch.py`/`formats.py` (the plan lists them for
"emit"; the diff does not touch them and the emit lives in `extract.py`, which is where the entry records are built).

**Verdict: approve with suggestions.** No blocker. The implementation conforms to spec5 §3–§6 and the vectors; two majors are a test-strength
gap on a leak-relevant validator arm and an unowned plan obligation.

CPG: considered, not relevant — a contract/encoder change reviewed from the diff and the spec text; no `cpg_*` graph answers it.

Evidence run (all foreground, `ulimit -v 3000000`, `timeout -s KILL 160`, cwd `cpg/`): `tests/test_bindparts.py test_bundle.py
test_writer_rules.py test_config_extractor.py test_answer.py` = **2685 passed** (55 s). The rest of the suite was not run.
Mutation runs were done on a scratch copy of `cpg/` (the tree was not edited).

## Findings

### M1 (major, verified) — V5d's bracketed-IPv6 format arm has no reject row; spec5 §8's in-suite mutation pass is absent

`bundle.py` `_bind_parts`: `if not _ipv6(host[1:-1]) or host == "[::1]": fail("V5d")`. Mutating it to `if host == "[::1]":` leaves
**2046 passed**: no BK row has a bracketed `ip` host that is non-canonical or not IPv6 (BK-31/32/34/35/49/52 hit the IPv4, `[::1]` and
V5e arms). The mutated validator then accepts `hostKind: ip, host: "[zz-anything]"` as a **clear** `host`, which is the field spec5 §4
reserves for non-secret classes. The real code is correct (probed: `[2001:0db8::5]`, `[2001:DB8::5]`, `[2001:db8:0:0:0:0:0:5]`, `[zz]`,
`[2001:db8::5` each reject V5d), so this is a guard that cannot fail, not a bug. Spec5 §8 also assigns `test_bindparts.py` "a mutation pass:
disabling any one V-conjunct ... turns at least one BK row red"; no such test exists (`grep -i mutat` finds only the `mutate()` helper), and
this survivor is the proof it would have caught something.
**Fix:** add a BK-style row (or a parametrised test) with `hostKind=ip` and `host` in `{"[2001:0db8::5]", "[zz]"}` expecting V5d; ideally
implement the spec's mutation pass (apply each conjunct-disable to a copy of the validator) or state it deferred.

### M2 (major, verified by grep) — the `header_fields` merge has no owner in the plan (orphan obligation)

`ConfigResult.header_fields` / `writer_rules.bind_header_fields` yield `"1.1"` + `bindKeyCheck`, but only the test helper `_bundle_for`
merges them. Grep of `docs/plans/connected-code-graph-p1*.md` for `header_fields`, `bind_header_fields`, `bindKeyCheck`: only P1-14's own row
and the graph note (G13). The natural owner is P1-44 (`-p1-linking.md`, "each non-Joern extractor writes one bundle per unit through
`ccg.atrest.write_bundle`... merged into the unit's bundle"), which does not mention it; P1-18 (assembler) reads bundles and is not the writer.
Failure mode if left: fail-closed (`bundle.validate` V1a/V1c raises, unit becomes `bundle:invalid`) rather than silent, so it is loud but
would block every config unit with a URL entry.
**Fix (architect):** add to P1-44 behaviour (1): the unit header takes `bind_header_fields(key, <merged records>)` over the **merged** record
set (not a per-extractor `header_fields` union), with a test that a config unit with a URL entry validates as `"1.1"` and a code-only unit as `"1.0"`.

### m1 (minor, verified) — float values still echo in two pre-existing messages

Spec5 §6.3: a message never includes the value of any `bindParts` field. Probed `bindParts.port = 8200.5`, `pathHmacs[0] = 31337.5`,
`header.bindKeyCheck = 31337.5`, `bindValue = 31337.5`: all reply `... numbers are integers only (§8.1), got <value>` from
`_check_values` (`bundle.py:230`, `{value!r}`); `_no_float` (`bundle.py:139`) echoes `text` the same way. All string/other-type corruptions
are value-free (probed 18 shapes: wrong-typed host/scheme/hostHmac, extra key, non-NFC host/bindValue/scheme, over-long host).
Severity: low. A float cannot be a bearer capability string, the cell holding it is a number field, and a hash/host cannot be a float in a
conforming bundle; the realistic leak is a port-like number. But it is a literal §6.3 breach and the test (`strings` of length >= 6) cannot see
numbers. **Fix:** drop `got ...` from the `_check_values` message when `key` is `bindParts`-scoped (or everywhere: the integer rule needs no value),
and add the 8200.5 row to the §6.3 test. Delegate's own disclosure matches what I found.

### m2 (minor, verified) — mislabelled rule ids in value-free messages

`_value_free_schema_message` maps any `host` schema error to `V5b` (a non-string `host`, probed `host=123456`, reports V5b "host present") and a
malformed `bindKeyCheck` to `V1c/V1d`. The message stays value-free, only the id misleads. Fix: map by `err.validator` (`type` -> a V4-class id or
say "not a string"), and name the bad-format check without borrowing V1c/V1d.

### m3 (minor, verified) — loopback `127/8` classification has a test hole

Mutating `_bind_authority`'s `host.startswith("127.")` to `"127.0"` leaves the suite green: only `127.0.0.1`/`127.0.0.2`/`127.000.0.1` are used.
`http://127.1.2.3/` would become `ip` (clear host, W6 withholds `bindValue`), contradicting spec5 §3/§4. Add `127.1.2.3` and `127.255.255.254`
to the writer parametrisation (BP-13's neighbours).

### m4 (minor, plan text) — guard wording vs the implementation split

Plan: "only `writer_rules.bind_parts_allowed` writes it". Implemented: `bind_parts_allowed` (E1–E4 gate, pure bool), `bind_parts_fields` (the single
write site, always calls the gate), with W6/W7 inside `bind_value`. **Judged sound and better than the single-function wording**: spec5 §8 itself
describes `bind_parts_allowed` as the E2–E4 decision, and the new test asserts the dict-literal writes of `bindParts` in `writer_rules.py` all lie
inside `bind_parts_fields`, which calls `bind_parts_allowed` (and `_OWNED_FIELDS` plus `_SEVEN` literal pin the set, with a planted-offender
`owned_field_writes` row for `bindParts`). Mutation: dropping `withhold` from the gate, the alias half, or `force_external` each reddened a test.
Action for architect: reword the plan line to "only `bind_parts_fields`, gated by `bind_parts_allowed`". Residual (documented guard limit, no action):
`valuehash.bind_parts` is not in `_RULE_NAMES`, so another extractor could call it, but writing the result under a literal `bindParts` key is flagged.

## Checks asked for, and outcome

- **Spec §9.1–9.5 conformance:** `test_bindparts.py` parses EH-1..32 (32 rows), BP-1..47, BK-1..52 (all 30 V-ids have a reject row, asserted), BR, BL-1..17
  from spec5's own text; independent `ref()` HMAC reimplementation. All pass. `parse_bind_url` read line by line against §3 steps 1-4: matches
  (length < 512, 0x21-0x7E, forbidden `@?#;\%$`, last-colon split, canonical IPv6 with range exclusion, 253 cap, all-digit last label, 32 segments, empty/dot segments).
- **Schema closedness:** `bindParts` closed (`additionalProperties:false`, `if/then/else` on `hostKind`); probed: `bindParts`/`bindKeyCheck` on a `file` record rejected,
  `bindParts: null` rejected, upper-case and trailing-newline hex/scheme rejected, `"1.2"` rejected. `bindKeyCheck` only in the header. No `required` changes.
- **hostKind external/link-local withholding:** `_forbidden_target` forces `external` (no `host`, hash only) in `_parts_for`, and `bind_value` withholds the clear
  URL by W6; tests for 169.254.169.254, `[fe80::1]`, `[fd00:ec2::254]`, 100.100.100.200 incl. "even if declared a service". Mutating `_parts_for`'s `force_external` reddens;
  mutating the same flag in `bind_value` survives and is **equivalent** (a link-local host is class `ip`, withheld by W6 either way).
- **Mutants:** I ran 22 mutants on a scratch copy; 19 killed, 1 equivalent (above), 2 live = M1 and m3. The delegate's "4 equivalent mutants" are not recorded in the repo, so I cannot confirm that list; I confirmed one equivalent and found two that are not.
- **Changed existing tests, legitimately?** Yes. `test_writer_rules`: the `http://10.0.0.5:8080/api` row moved to `10.0.0.5:8080` and the neighbouring-address test now expects `None` for `://` forms, both required by W6 (spec5 §6.2), with new W6 tests asserting the withholding; `test_bundle`: the `"1.1"` header flips to valid and `"1.2"`/`"2.0"` are added, the constant check now reads the enum from spec5's text; `test_answer`: three fields added to the closed `launch` object list. No assertion was weakened without a spec reason.
- **Assembler seam:** M2.
- **Growth-stops / plan conformance:** `bundleSchemaVersion`/`bindKeyCheck` writer side, `bind_parts_usable`, `mount_prefix_hmac` and the key-epoch tests are present; the one-definition (`hmac_key`/`bind_key_check`) test exists.

## What's solid

Value-free failure paths for every non-number shape; `parse_bind_url` as one shared implementation for writer and validator (no URL library); validator never touches a key;
tests read the spec text rather than copy constants; the `withhold`/alias/link-local gate arguments are each pinned by a failing test.

## Open questions

None blocking. For architect: whether the in-suite mutation pass of spec5 §8 stays a requirement or is dropped in favour of this review's scratch-copy run.

## Pass 2

**Verdict: approve.** No new blocker or major. Run: the five-file set (`test_bindparts`, `test_bundle`, `test_writer_rules`, `test_config_extractor`, `test_answer`) = **2863 passed** (99 s), venv, cwd `cpg/`, `ulimit -v 3000000`.

- **M1: fixed.** `V5d-v6-format`, `-v6-loopback`, `-v4`, `-v4-127(-prefix)`, `V5c-prefix` and the whole V/S mutant list now run in-suite. The harness `exec`s the mutated `ccg/bundle.py` source as a private module (`_mutant_module`) and a mutant is red only on a row disagreement with the wrong or missing rule id (a crash does not count); `test_the_unmutated_module_agrees_with_every_row` is the positive control. Verified it executes mutated source: I broke arguments on a scratch copy (outside the suite harness) and each turned a test red: `2002::/16`->`/17`, `192.0.0.192/32`->`/31`, `168.63.129.16/32`->`/31`, `BIND_PATH_TAG`/`BIND_HOST_TAG` `v1`->`v2`, path cap 32->31, `127.`->`127.0`.
- **M2: fixed (plan).** `-p1-linking.md` P1-44 (8) assigns the merged-records `bind_header_fields` call and V1a-V1d, "never a union of per-extractor `header_fields`".
- **m1: fixed.** Both float messages are value-free; tests for the validator and parser paths (8200.5, 31337.5).
- **m2: fixed.** `host` non-string reports "not a string", `bindKeyCheck` no longer borrows V1c/V1d; the corpus's `rule in msg` check passes with the remapped ids, so the remapping and the new reject rows agree.
- **m3: fixed.** 127/8 neighbours covered (`V5d-v4-127-prefix`, `V5c-prefix`, and the writer mutant above is red).
- **m4: open, plan wording only** (architect), unchanged.
- **Equivalent mutants (`EQUIVALENT_MUTANTS.md`):** the four are dropping one of `$ @ % \` from `_BIND_URL_FORBIDDEN`. Argument is sound by inspection (host label, segment, port and scheme classes each refuse those characters: `u:p@db` gives port text `p@db`); evidence is a 40,186-case fuzz with 0 differences, which is a bound, not proof. Accepted, with the stated caveat; `;?#` are redundant the same way and unmutated.
- **New writer set vs W6/W7:** 168.63.129.16, 192.0.0.192 and 2002::/16 force `external` in `_parts_for` and `bind_value`, so a URL to them keeps hash-only parts and no clear `bindValue`; neighbours (`.17`, `.15`, `192.0.0.193/.191`, `[2003::1]`) and RFC1918 stay clear. Test `texts` coverage (`audit_records` over `a_url=168.63.129.16`, `b_url=[2002:a00:1::1]`) found no disagreement. The validator has no link-local rule (writer-only), consistent with spec5 §10.
