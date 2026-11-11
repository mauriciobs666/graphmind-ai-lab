# C-1001 P1-14 security review: `bindParts` encode (URL parse, host classification, keyed parts, writer gate)

> **Status:** active · **Owner:** `security-expert` · **Tracks:** C-1001 (P1-14)

**Lens:** code/app security (advisory, not a gate). **Artifact:** uncommitted P1-14 diff: `cpg/ccg/bundle.py` (`parse_bind_url`, `_bind_parts`, `_bind_version`, value-free schema messages), `cpg/ccg/schemas/bundle-v1.schema.json`, `cpg/ccg/valuehash.py` (`host_kind`, `bind_parts`, `bind_parts_usable`, `mount_prefix_hmac`), `cpg/extractors/common/writer_rules.py` (`bind_parts_allowed`, W6/W7, `bind_header_fields`), `cpg/extractors/config/extract.py`, `cpg/tests/test_bindparts.py`. Baseline: `docs/plans/connected-code-graph-p1-extraction.md` § P1-14 and `docs/plans/connected-code-graph-bundle-spec5.md` (§3 parse, §4 host kinds, §6 gate and messages, §7 mount prefix).

CPG: considered, not relevant — the files are uncommitted, so any built `cpg_cpg` graph predates them; paths were traced by reading and by live throwaway probes (cwd `cpg/`, `ulimit -v 1500000`, `timeout -s KILL 45`, made-up keys, no network, no service touched).

**Verdict: approve with suggestions.** No blocker, no confirmed leak. The URL parse is a strict whitelist that fails closed, every link-local/metadata encoding I tried ends up `external` (hash only) or refused, messages never echo values, and `bindParts` has exactly one writer. Two minors concern *where* the link-local rule lives (writer only, not the validator) and a coverage gap in the metadata list; the rest are nits.

`cpg/tests/test_bindparts.py` + `test_writer_rules.py`: 689 passed.

## Findings

### minor-1 — the link-local/metadata rule is writer-only; the validator accepts a clear link-local `host` (confirmed live)
`bundle._bind_parts` (V5c-V5i) checks that `hostKind: ip` is a well-formed non-loopback address, but not that it is outside link-local/metadata space. Forged bundle (base bundle of `test_bindparts.base()`, `bindValue` removed): `hostKind: ip, host: 169.254.169.254` and `host: [fe80::1]` and `hostKind: service, host: metadata` all **validate**. The plan (§P1-14 "Link-local hosts") puts the rule on the writer (`_forbidden_target` in `writer_rules`), and spec5 never mentions it, so this is a design choice, not a defect in the diff. Exposure: a different producer or a writer regression emits a clear metadata IP, and only `audit_records` (needs the writer's own test to run it) would notice. Suggest: move the CIDR set into `ccg.bundle` (it already holds `v6_excluded`) and add a validator rule (V5-class id) rejecting `ip` hosts in it, so the contract is enforced where bundles are read; `writer_rules._forbidden_target` then imports it. If deliberately writer-only, say so in spec5 §6.4.

### minor-2 — metadata coverage is the fixed list; a few well-known targets stay clear `ip` (confirmed live)
`http://168.63.129.16/` (Azure wireserver) and `http://192.0.0.192/` (Oracle) yield `hostKind: ip` with the address in clear; `[2002:a9fe:a9fe::1]` (6to4 embedding 169.254.169.254) likewise. They are host addresses, not secrets, and the plan lists only the set implemented, so this is a coverage question rather than a bug. Same for RFC1918 addresses (`10.0.0.5` stays clear by spec §4): `bindParts.host` therefore discloses internal topology. Suggest: decide explicitly whether `ip` hosts are clear by policy; if the aim is "metadata → hash only", add 168.63.129.16, 192.0.0.192 and 2002::/16-embedded v4 checks to the one shared list (with minor-1).

### minor-3 — `audit_records` cannot re-derive `bindParts` for an entry whose text it does not hold
`_audit_config_entry` compares `_parts_for(key, text, declared) != bp` only when `texts` has the id or a `bindValue` is present. W6/W7 withhold `bindValue` for `ip`/`external`, which are exactly the entries carrying parts that matter, so a writer test that does not pass `texts` gets no check of host classification for them. Suggest: have the P1-11 writer test always supply `texts`, or make the audit report "unverifiable" rather than silent.

### nit-1 — `mount_prefix_hmac(host, ...)` does not validate `host`
`host` is concatenated unescaped: `mount_prefix_hmac(k, "h/a", "/b")` equals the second `pathHmacs` entry of `http://h/a/b` (confirmed). Callers (P1-17) pass the decided host, so theoretical, but a one-line `"/" in host` refusal makes the function total. Surrogate hosts raise `UnhashableValue` with a value-free message (fine).

### nit-2 — wrong rule id for a bad `hostKind` value
A `hostKind` outside the enum (e.g. a secret string) reports `configEntry.bindParts: V5b` (the `else: required host` branch wins in `best_match`) rather than V4f. No value is echoed; only the diagnostic is misleading.

### nit-3 — `bindKeyCheck` is a stable per-key fingerprint
`HMAC(key, tag\0)` is identical in every bundle of a system; it reveals nothing about the key but links bundles. By spec; noted only because P1-17 will compare it with `compare_digest` (as `bind_parts_usable` does, fine).

## Requested probes, results (all confirmed live unless marked)

- **Userinfo / query / fragment / `;` / `\` / `%` / `$`:** forbidden characters in step 1 of `parse_bind_url`; `http://user:pw@host/`, `?a=b` → `None`, so no parts, no `bindValue`. Zone ids (`[fe80::1%eth0]`, `%25eth0`) → `None`.
- **Link-local and metadata encodings:** dotted, uppercase scheme (`HTTP://`), with port, `0xa9fea9fe`, `0xa9.0xfe.0xa9.0xfe` → `external`, no `host`, hash only; decimal `2852039166`, octal `0251.0376.0251.0376`, `127.1`, trailing dot, `1.2.3.4.5` → refused (all-digit last label). IPv6: `[fe80::1]`, `[fd00:ec2::254]` and its long uppercase form, `[fec0::1]`, `[100::1]` → `external`; IPv4-mapped (both notations), `::/96`, NAT64 → refused; `100.100.100.200` → `external`. Writer: `bind_value` is `None` for all of these; `HTTP://169.254.169.254` cannot slip through `_URL_AUTHORITY` (lowercase-only) because `_forbidden_target` runs on the parsed `url.host`, not the raw text.
- **Unicode / IDNA lookalikes (`１６９.…`, `ａ.com`), lone surrogates:** step 1 admits only `\x21`-`\x7e`; all `None`, no exception.
- **Robustness:** length bound 511; 600-char host, 300-label host, 254-segment path, unterminated/odd brackets all `None` in under a millisecond; regexes are linear (`fullmatch` on `[^/]*(/.*)?`); no URL library involved.
- **Fail-closed divergence between grammar and parse:** a value the §1.7.2 grammar accepts but the parse refuses gets no parts and (W7) no clear `bindValue`; reverse direction cannot occur because W7 requires parts for any `url`-form value. `http://localhost:8000/api/v1` keeps parts but no `bindValue` (pre-existing grammar limit).
- **HMAC value hashing:** `entry_hmac` uses ASCII tag + NUL + value; the three tags contain `.` so they cannot equal a 32-hex `valueHash` domain; host and path messages are separated by tag and by `/`; an external host's `host` is never written; `pathHmacs` hide capability segments (the `https://ntfy.sh/x7k2-…` case gives no clear value). `bind_key_check` is a keyed MAC of a constant (no key bytes), published test keys are refused by `atrest`. `bind_parts_usable` uses `compare_digest` and never raises on a hostile header.
- **Error echo:** 11 forged records with a marker string in `scheme`, `hostKind`, `host`, `hostHmac`, `pathHmacs[0]`, `port`, an extra property *name*, `bindParts` itself, and `bindValue` as str/dict/list: every message was field + rule id, none carried the marker. `_bind_parts` and writer messages are value-free by construction.
- **Writer-rule guard:** `bindParts` joins `_OWNED_FIELDS`; the planted-offender tests cover literal and folded keys; the only write site is `bind_parts_fields`, which calls `bind_parts_allowed` (name rule on every `keyPath` segment, W1, binding role, alias anchor; `db.password.url`, `x.args[0]`, `a.token` → `{}` live). Residual: the tripwire is an AST pattern match (`dict(zip())`, `update(var)`, computed keys are invisible), as `writer_rules.py` already states; the validator's V2/V3 re-check is the backstop.

## What's solid
Whitelist parse with one function shared by writer and validator; fail-closed W6/W7 ordering (parts gate first, then clear value); `force_external` takes precedence even over a caller-declared service; value-free diagnostics with dedicated tests; version/key-check pairing (V1a-V1d); single-definition of the key and check.

## Open questions
1. (`cobb`/spec owner) Is the link-local/metadata rule intended to stay writer-only (minor-1)?
2. Should `ip` hosts, including RFC1918 and the uncovered metadata addresses, be clear at all (minor-2)?
