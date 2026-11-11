# Security review: bundle-spec6 (`sandboxAuditIds` of host-extractor bundles)

> **Status:** active · **Owner:** `security-expert` · **Tracks:** C-1001 (P1)

**Lens:** code/app + plan provenance (advisory). **Artifact:** `docs/plans/connected-code-graph-bundle-spec6.md`, read against egress-spec3 §5.4, `cpg/ccg/sandbox_audit.py`, `cpg/ccg/assemble/rows.py`, `cpg/ccg/bundle.py`, `cpg/ccg/schemas/bundle-v1.schema.json`, `cpg/ccg/build.py` (uncommitted), the shell/config/markdown extractors. Static reading only; no code was run and nothing was exploited.

**Verdict: approve with suggestions.** No blocker. The empty list does not weaken what `sandboxAuditIds` attests, provided the three majors/minors below are closed in P1-44a/P1-18.

CPG: considered, not relevant: the question is a header-constraint relaxation plus parse-only extractors, answered by reading the check and the parsers; no input→sink flow through a CPG-indexed component was in question.

## Findings

### M1 (major, wording/consumer risk): the "empty list proves nothing" text is only in a plan, and `check_sandbox_ids` returns `[]` for host bundles
`check_sandbox_ids` (`sandbox_audit.py:76-82`) returns `[]` ("valid") for a host `[]` bundle exactly as for a fully-attested code bundle. `classify_bundle` (`rows.py:202`) and any later consumer see no distinction. The only warning is spec6 §1/§2's "presence-only" sentence, which lives in a plan, not at the point of use. A later consumer (an audit report, a "bundle is sandbox-attested" badge, the `rows.py:437` `audit_ids += header["sandboxAuditIds"]` aggregation) can read "check passed" as "attested". The aggregation at `rows.py:437` silently contributes nothing for host units, which is correct but unlabelled.
Suggestion: give the host case a distinct, loud, non-error outcome rather than a bare `[]`. E.g. a sibling `attestation(header, audit_dir) -> "sandboxed" | "host-unattested" | "invalid"` (or a returned tag next to the error list), used by the assembler to record per-unit `attested: false` provenance (a `Unit` property or an audit-report line "N host units: no sandbox attestation by design"). Keep `check_sandbox_ids`' `[]`-means-valid contract for the refusal path, but have the docstring and egress §5.4 state, in the same sentence as the `HOST_LANGS` clause, that `[]` is "unattested, not attested".

### m1 (minor): `HOST_LANGS` type and drift guard
Spec §2 says `frozenset`; code is a tuple (`sandbox_audit.py:34`). Harmless for `in`, but the spec/code disagree, and a tuple permits order-based accidental equality with a list in tests. The spec's drift test (HOST_LANGS == schema `then` enum) is good but the schema half does not exist yet: `bundle-v1.schema.json` still has unconditional `minItems: 1`, `rows.py:176` pops it as a stopgap. Until P1-44a lands, a host `[]` bundle passes assembly but fails `bundle.read_bundle`. Make the stopgap fail-closed-tested (a code `[]` bundle must be refused at assembly by `check_sandbox_ids`, which it is) and remove the pop when the schema lands; spec §4 ordering should say "P1-44 does not ship before P1-44a" explicitly.

### m2 (minor): the anti-evasion rule checks `file` records only; unit.lang and file.lang are independent
`bundle.py:975-978` only ties `symbol.lang` to its file's lang; nothing ties `file.lang` to `header.unit.lang`. Traced consequences: (a) a hostile writer labelling a code unit `shell` with python `file` records and `[]` is caught by the new rule (good); (b) a `shell`-labelled bundle with zero `file` records and `[]` passes, but is vacuous (symbols need a file; no content), so no exposure; (c) the schema `lang` enum is closed (`bundle-v1.schema.json:375`), so case/Unicode variants (`Shell`, fullwidth) fail the schema before the check, and `lang in HOST_LANGS` is exact-match on a validated string; (d) records other than `file`/`symbol` carry no `lang` that could smuggle code semantics. Suggestion: state in the spec that the rule relies on the schema's closed `lang` enum running first (true in `classify_bundle` and `read_bundle`, but `check_sandbox_ids` alone does not validate; it silently treats an unhashable/odd `lang` as non-host, which is fail-closed, fine). Optionally also require `file.lang == unit.lang` for host units (stronger and simpler than a set membership), but that is a hardening, not a gap.

### m3 (minor, pre-existing, theoretical): host path runs `git` on the hostile repo without the hardened env in one place
`blame.git` (`extractors/joern/normalize/blame.py:33-48`) strips `GIT_*`, disables global/system config and is timeboxed. `ccg/units.py:80` (`_git_files`: `git ls-files -z`) uses a bare `subprocess.run` with the ambient environment and no timeout parity beyond 60 s. `ls-files` does not run hooks or fsmonitor in normal operation, so I rate this theoretical, but since the host path is precisely the one with no sandbox, route it through `blame.git` (or the same `_env()`) so the host extractors have one git-hardening point. Not caused by spec6; spec6 makes the host path's trust matter more.

## Does a host extractor ever execute repo content? (confirmed by reading)
- **Shell:** tree-sitter-bash parse only (`shell/extract.py:12` "parsed, never run, never sourced or eval-ed"); `eval` is modelled as a call site and its argument parsed as shell text, capped (`MAX_EVAL_CHARS`, `MAX_EVAL_COMMANDS`); file read bounded (2 MiB), depth-bounded, symlink-resolved and confined to the unit root (`_read`, lines 141-157); build.py reads blobs by `git cat-file` of the built commit, not the work tree. No `subprocess`/`os.system`/`exec`/`eval` import in the extractor.
- **Config:** `formats.py` uses `json`, `tomllib`, and a `yaml.SafeLoader` subclass used for composition only (`get_node`, never `construct_*`), so custom tags and `!!python/*` are never constructed (SafeLoader would reject python tags anyway); alias/merge bombs are bounded by `MAX_NODES` 50,000, `MAX_DEPTH` 32, `MAX_BYTES` 512 KiB, one document, `RecursionError` caught. `launch.py` only tokenises (`shlex.split`) command strings, "Nothing here executes".
- **Markdown:** imports are `hashlib/re/unicodedata` plus `ccg.ids`; no parser with execution semantics.
So the property attested by `sandboxAuditIds` (no repo-driven code execution outside the sandbox) is, for these three, preserved by construction, not by the id. Residual risk is parser DoS/memory (bounded) and parser CVEs (tree-sitter, PyYAML), which the sandbox would mitigate and the host path does not; this is the real cost of the host path and independent of whether the list is `[]`.

## Instruction-shaped text hazard
None. Spec6 is declarative and its "No evasion by label" bullet states the safety property (a code unit cannot escape the sandbox check), not a workaround. The value-free error message rule is good.

## What's solid
Closed `HOST_LANGS` set; conditional schema (`if/then/else`) rather than a blanket `minItems: 0`; the file-language rule closing the labelling bypass; mutant list includes the right three (widened set, dropped file rule, language-blind `[]`); no new record minted so nothing claims an offline run that did not happen; `[]` bundle with ids still checked as before; `egressMode` still checked for `[]` host bundles (`sandbox_audit.py:80`).

## Open questions
1. Should assembled output (Unit node or audit report) carry an explicit `attested`/`sandboxed` flag so consumers need not infer it from `sandboxAuditIds` (M1)? Owner: `architect`/`graph-dba`.
2. Is a host-bundle `egressMode: "offline"` meaningful when nothing was offline-run? It is a constant label here; consider whether it too should not read as attestation.
