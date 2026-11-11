# Review: connected-code-graph-bundle-spec6 (`sandboxAuditIds` of host-extractor bundles)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

**Scope.** `docs/plans/connected-code-graph-bundle-spec6.md` and the two header-pointer edits (`-bundle-spec5.md`, `-egress-spec3.md`; `git diff` read: both correct and minimal). Read against `-bundle-spec3/4/5`, `-egress-spec3.md` §5.3–§5.4, `cpg/ccg/schemas/bundle-v1.schema.json`, `cpg/ccg/bundle.py` (`_header`, `_Semantics`, `loads`/`validate`), `cpg/ccg/sandbox_audit.py` and `cpg/ccg/assemble/rows.py` (both are uncommitted work ahead of the plan), P1-18/P1-44 in `docs/plans/connected-code-graph-p1-linking.md`, root `REVIEW.md` and the growth-stops rules. The existing `docs/reviews/connected-code-graph-bundle-spec6-security.md` was not folded in.

**Verdict: approve with suggestions.** No blocker. The decision is sound and the schema delta works as stated. Three majors concern an overclaim, a unit that is in neither plan row nor ownership, and a stale base citation.

CPG: considered, not relevant — a spec-text and schema-constraint review settled by reading and by validating headers against a scratch schema; no call or data-flow question.

## Findings

### Major

**M1. "No evasion by label" overclaims; the residual trust boundary is unstated** (spec6 §1 last bullet; verified by probe and by reading `bundle.py`).
The file-language rule is decidable only over labels. `_Semantics` has no `file.lang == header.unit.lang` rule (only `symbol.lang == file.lang`, `bundle.py:977`). So:
- A code unit labelled `shell` whose `file` records are also labelled `shell` passes every rule: schema, the `_header` rule and `check_sandbox_ids`. `unitId` is derived from `unit.lang` (`bundle.py:894`), so the forgery is self-consistent. The rule defeats only the *inconsistent* label (a `python` file in a `shell` unit).
- A `shell` unit with zero `file` records and `[]` passes vacuously, and a `shell` unit may hold `config`/`markdown` files. Both are harmless but undeclared.
- Case/Unicode variants are closed: I validated `Shell`, `sheLL`, `shell​` against the proposed schema and all fail on the `lang` enum (a closed enum, exact match).

*Fix.* Reword the bullet to "a bundle with `[]` cannot *contain* a file labelled with a code language", and state that the label's truth rests on the builder: P1-44 passes `[]` only from its host-extractor branch, chosen from the manifest unit's kind, never from a writer-supplied header. Add the P1-44 test: a python/javascript unit built with no `audit_ids` still raises `MissingAuditIds`, and `header.unit.lang` equals the manifest unit's language. Do not extend the validator with extension heuristics.

**M2. P1-44a is not in the plan and `HOST_LANGS` ownership is circular** (spec6 §2 rows 3–4, §4; plan P1-18 and the ruling at `-p1-linking.md:274`; grep "P1-44a" over `docs/` outside reviews hits only spec6).
- The plan ruling says P1-18 (`graph-dba`) owns `check_sandbox_ids` *and* the `HOST_LANGS` constant. Spec6 §2 says P1-44a (`coder`) lands `HOST_LANGS` and the `bundle.py` rule, importing it from `ccg.sandbox_audit` (P1-18's file). §4 says P1-44a lands `HOST_LANGS` but §2 defines it in P1-18's module, so each unit waits on the other.
- A format validator (`bundle.py`, pure) would then depend on an audit module that does I/O and imports `jsonschema`.

*Fix.* Define `HOST_LANGS` once in `ccg/bundle.py` (P1-44a's file); `sandbox_audit.py` imports it. Add the P1-44a row to the linking plan (files, gate, order) and amend the ruling's "P1-18 owns ... `HOST_LANGS`". The ordering then reads P1-44a → P1-18's `check_sandbox_ids` change → P1-44.

**M3. The base citation is false** (spec6 header, line 5 and §2 row 1).
"The `sandboxAuditIds` row ... as carried by spec4/spec5": `grep sandboxAuditIds` over spec4 and spec5 returns nothing. The row lives only in `-bundle-spec3.md:959` (§2.2), a `superseded` document; spec4 supersedes spec3 and spec5 adds `bindParts` rows. The ruling said `Extends:` spec3, which is the one place the live row sits, so the header lists the wrong documents and the feature-close spec update will not know which section to edit.

*Fix.* Cite `-bundle-spec3.md` §2.2 as the row's home ("not restated by spec4/spec5"), and say which header edit the successor carries (spec3 is `superseded`, so it takes no `Extended by:` pointer, but spec4 does).

### Minor

**m1. The text and the tree disagree on the empty-list branch** (spec6 §2 row 4 vs `sandbox_audit.py:79-82`).
The spec says the `egressMode` check applies "to a non-empty list" and gives the error string `"sandboxAuditIds: none named for a non-host language"`. The in-tree function also requires `egressMode == "offline"` for a host `[]`, and uses a different string. Pick one: the stricter form is reasonable since P1 accepts only `offline`. State it, and add the test (host `[]` with a non-offline mode).

**m2. The test "`HOST_LANGS` equals the schema's `then` enum" names the wrong branch** (spec6 §2 row 2, §3 test 2). The enum is in the `if`. The `then: {minItems: 0}` branch is a no-op (I confirmed that `[]` passes without it). Drop `then`, and read the `if` enum in the test.

**m3. Frozenset vs tuple** (spec6 §2 row 4; the tree has a tuple). `lang in frozenset` raises `TypeError` on an unhashable `lang` from a header that was not schema-validated first (P1-19 calls the check directly). Either keep the tuple or guard `isinstance(lang, str)`; add a planted `lang: ["shell"]` case.

**m4. Validator message convention** (spec6 §2 row 3). Sibling `_header` messages cite a rule id or a section (`V1a`..`V1d`, `§2.2`). Give this rule an id and cite it, still value-free.

**m5. Tests and mutants** (spec6 §3). Add:
- (a) the non-host file is not the first file record, so a "check only the first file" mutant dies;
- (b) a host unit naming ids *and* holding a non-host file stays valid, killing a mutant that applies the file rule to non-empty lists;
- (c) a `schema`-only mutant, where the conditional is removed and `bundle.py` alone refuses a python `[]`. Test 1 covers it, but it needs the schema validated independently of `bundle.py`.

No stored bundle exercises a host `[]`, so the positive case must come from a synthetic bundle.

**m6. A stale sentence will outlive the change** (`rows.py:174-176`, the `minItems` pop and "until the spec6 schema amendment lands"). Once the schema carries the conditional, the pop is dead and the comment false. Add it to P1-44a/P1-18's sweep as an owner.

### Verified (checks I ran)

- **Stored bundles.** I validated every stored bundle header with the proposed schema delta, on a scratch copy (`$defs.header` with `minItems` dropped and the `allOf` if/then/else added). All 10 stored bundles are valid: `tests/data/scip/real_ts.bundle.jsonl` plus 9 under `.ccg-work/vertical-slice/` (python 2 ids, javascript 3). All are code bundles, so **no `bundleSchemaVersion` bump is needed**.
- **Schema behaviour.** Synthetic headers behave as intended: `shell` with `[]` is valid, `python` with `[]` fails `minItems`, a missing `unit` fails `required`, and `Shell`/`sheLL`/`shell​` fail the `lang` enum.
- **Enum spelling.** `$defs.lang` is `python, javascript, java, kotlin, csharp, go, c, cpp, shell, config, markdown`, so the three `HOST_LANGS` names are the exact spellings.
- **Where each rule runs.** The file-language rule can run only in `bundle.validate`, because `classify_bundle` is header-only. `build.py:383` and `read_bundle` go through `loads`→`validate`, so the load path covers it. Spec6 §3 test 3, "a code bundle with `[]` refused at assembly", is satisfied by the header-only check alone; the spec should not imply `classify_bundle` enforces the file rule.

## What's solid

The decision and the rejected alternatives (no forged host `sandbox` record). The empty list is read as "no sandboxed invocation", never "succeeded". The relaxation is one closed set, with no id, tuple or version change. The §5.4 "presence-only" amendment is accurate: the check reads `sandbox.jsonl` records only. The spec is short (about 640 words) and holds only the change; there are no changelog or history passages. Growth-stops: a new document, so there is no revision growth to report, and the only restated fact is the one-sentence recap of the §2.2 row, which is acceptable.

## Open questions

1. Does the stakeholder want a cheap consistency bound, e.g. `[]` requires every `file.kind` to be non-`code` (`script`/`config`/`schema`/`prompt`)? It is still only a label, so I recommend against it (M1).
2. Should the stricter `egressMode` check for host `[]` (m1) be the specified behaviour? The tree already does it.
