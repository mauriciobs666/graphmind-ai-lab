# Connected code graph — fact-bundle spec, `sandboxAuditIds` of host-extractor bundles

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1) · **Extends:** `docs/plans/connected-code-graph-bundle-spec4.md`, `docs/plans/connected-code-graph-egress-spec3.md` · **Reviews:** `docs/reviews/connected-code-graph-bundle-spec6.md`, `docs/reviews/connected-code-graph-bundle-spec6-security.md`

Step P1-44a (`docs/plans/connected-code-graph-p1-linking.md`). It states only the change. The `sandboxAuditIds` row lives only in `-bundle-spec3.md` §2.2 (superseded; spec4 supersedes it without restating the row, spec5 adds only `bindParts` rows), so that row is the one amended; spec4 carries the `Extended by:` pointer. No id, tuple, record field or `bundleSchemaVersion` changes: one header constraint is relaxed for a closed set of units, so every stored bundle stays valid (all ten stored headers validate under the new schema).

## 1. Rule

`header.sandboxAuditIds` names one `sandbox` record per `offline-run.sh` invocation that produced the bundle's inputs. The shell, config and markdown extractors run on the host, never in the sandbox, so such a unit has no such invocation and **may carry `[]`**.

- `HOST_LANGS = ("shell", "config", "markdown")`, defined **once**, in `ccg/bundle.py`, as a tuple (`in` on a tuple never hashes, so an unhashable `lang` in an unvalidated header cannot raise `TypeError`); `ccg/sandbox_audit.py` and the tests import it, and `bundle.py` imports nothing from the audit module.
- `[]` is valid only when `header.unit.lang` is in `HOST_LANGS`; a code language with `[]` is invalid. A host bundle that names ids stays valid and each id is checked as before.
- **Rule S6a** (a validator rule in `bundle.py` `_header`, sibling of V1a–V1d): an empty `sandboxAuditIds` requires every `file` record's `lang` to equal `header.unit.lang`. Message, value-free: `line 1: header.sandboxAuditIds: S6a (an empty list needs every file in the unit's host language)`. It runs in `bundle.validate`, so `read_bundle` and `loads` enforce it; the header-only `classify_bundle` does not.
- **Trust boundary (stated, not overclaimed).** S6a and the schema can compare labels only; nothing proves a label true. A code unit mislabelled `shell` with `shell`-labelled files would pass them. The control is the builder: P1-44 passes `[]` only from its host-extractor branch, chosen from the manifest unit's kind and never from a writer-supplied header, and `header.unit.lang` is the manifest unit's language.
- **Provenance meaning.** An empty list says *no sandboxed invocation produced this bundle*, never that one succeeded. No `sandbox` record is minted. `egressMode` on a host bundle is a constant label (`offline`), not an attestation.

## 2. Deltas

| Where | Change |
|---|---|
| `-bundle-spec3.md` §2.2 row `sandboxAuditIds` | "at least 1" becomes "at least 1, or `[]` when `unit.lang` is in `HOST_LANGS`"; the "one per invocation" sentence is unchanged |
| `cpg/ccg/schemas/bundle-v1.schema.json` | the header's `sandboxAuditIds` drops the unconditional `minItems: 1`; an `allOf` of `if: {properties: {unit: {properties: {lang: {enum: [shell, config, markdown]}}}, required: [unit]}}` / `else: {properties: {sandboxAuditIds: {minItems: 1}}}` carries it (no `then`); the item pattern is unchanged |
| `cpg/ccg/bundle.py` | `HOST_LANGS` and rule S6a in `_header` |
| `-egress-spec3.md` §5.4 and `cpg/ccg/sandbox_audit.py` | `check_sandbox_ids` returns `[]` for an empty list iff `header.unit.lang in HOST_LANGS` **and** `egressMode == "offline"` (otherwise the existing egress error `egressMode: the header's mode is not 'offline' (the only mode P1 accepts)`); for any other empty or non-list value it returns `sandboxAuditIds: none named (only a host-extractor unit may name none)`. A new `attestation(header, audit_dir) -> "sandboxed" \| "host-unattested" \| "invalid"` wraps it: `host-unattested` for the valid empty case, `sandboxed` for a valid non-empty one. §5.4 gains: **an empty list is unattested, not attested; `[]` from `check_sandbox_ids` means "valid", never "attested"** |
| `cpg/ccg/assemble/rows.py` | where `audit_ids += header["sandboxAuditIds"]` (~:437) the assembly report also records `unattested_units` (sorted unit ids whose `attestation` is `host-unattested`) and its count, so an audit report never lists a host unit beside attested ones as if equal; no graph property (graph model unchanged). The `minItems` pop in `_header_validator` (~:174) and its "until the spec6 schema amendment lands" comment are removed once the schema carries the conditional |

## 3. Tests

1. `test_bundle.py`: a synthetic `shell` bundle with `[]` validates; the same with a `python` `file` record, placed **not first**, is refused with S6a; a `python` unit with `[]` is refused by the schema (the schema validated on its own, so a missing `bundle.py` rule cannot hide it); a `shell` bundle naming one id **and** holding a `python` file stays valid (S6a applies to the empty list only); a `shell` unit holding a `config` file with `[]` is refused (S6a is equality).
2. A drift test reads the schema's `if` enum and asserts it equals `set(HOST_LANGS)`.
3. `test_sandbox_audit` and `test_assemble_rows.py`: `[]` is accepted for `shell`, `config`, `markdown` with `offline`, refused with a non-offline mode, refused for `python`, `javascript`, `java`; a planted `unit.lang: ["shell"]` returns an error and does not raise; `attestation` returns the three outcomes; a host `[]` bundle assembles, lands in `unattested_units`, and a code bundle with `[]` is refused at assembly and at load.
4. P1-44 (`test_build_dispatch.py`): a python or javascript unit built with no `audit_ids` still raises `MissingAuditIds`; a host unit's header `unit.lang` equals the manifest unit's language and its `sandboxAuditIds` is `[]`.
5. Mutants: widening `HOST_LANGS` by a code language; dropping S6a; checking only the first `file`; applying S6a to non-empty lists; accepting `[]` regardless of language; accepting a host `[]` with a non-offline mode.
