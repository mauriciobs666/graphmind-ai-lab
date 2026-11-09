# Connected code graph — fact-bundle spec, keyed binding parts (`bindParts`, v1.1)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1) · **Extends:** `docs/plans/connected-code-graph-bundle-spec3.md`, `docs/plans/connected-code-graph-bundle-spec4.md` · **Reviews:** `docs/reviews/connected-code-graph-bundle-spec5.md`, `docs/reviews/connected-code-graph-bundle-spec5-security.md`

Step P1-13 of `docs/plans/connected-code-graph-p1-extraction.md`, decided by index D4
(`docs/plans/connected-code-graph-p1.md`), closing obligation F-7 (`-bundle-spec3.md` §13). It specifies
SA-4(b) of `docs/reviews/connected-code-graph-bindvalue.md` (Pass 1, SA-4; Pass 2, P2-6) as an
**additive minor version, `"1.1"`**: one optional record field (`configEntry.bindParts`), one
optional header field (`header.bindKeyCheck`) and the `bundleSchemaVersion` rule that admits them.
No id, tuple, `qname` rule or existing field changes. A bundle without the new fields is the same
bytes and stays valid, so every stored slice bundle is unaffected.

CPG: considered, not relevant — a contract amendment; no `cpg_*` graph was leaned on (the brief excludes it), and every fact below was read from the cited specs, the advisory and `cpg/ccg/bundle.py`/`cpg/ccg/schemas/bundle-v1.schema.json` at HEAD.

## 0. Reading rule

- **Base text.** Schema v1 is `-bundle-spec3.md` read with `-bundle-spec4.md` (`-spec4` §0). This
  document is read with both. A sentence below that names a `-spec3` or `-spec4` section amends
  that sentence and nothing else; every other rule (the token screen §1.7.1, the value grammar
  §1.7.2, the name rule §1.7.3, W1–W4 §1.7.4, `valueHash` §1.7.5, the gate vectors §1.7.6) applies
  unchanged.
- **Sentences of `-spec3` this document amends** (by citation; `-spec3`'s text is not edited):
  - §1.7.5, "So the assembler needs the key only to detect change. It never needs it to bind",
    reads: "The assembler needs the key to detect change and, for URL values, to bind through
    `bindParts` (`-spec5` §7). It never matches `valueHash` or `defaultHash` across entries."
  - §1.7.5, "Binding uses clear fields and env-var **names**", reads: "Binding uses clear fields,
    env-var names and, for URL values, the keyed parts of `-spec5` §5.3."
  - §5.17's `bindValue` row, "URL, env and Compose binding read it", reads: "env and Compose
    binding read it; URL binding reads `bindParts`, or parts the linker derives from a `url`-form
    `bindValue` (`-spec5` §7), and the writer withholds it under W6 (`-spec5` §6.2)."
  - §1.7.2, "The scheme list is closed. A further scheme is a major change, decided with F-7":
    decided in §6.2.
  - §1.11's `configEntry` row, "Until F-7, a renderer shows the provider's mount path, never a
    consumer's config URL": see §6.3.
- **Tags.** As in `-spec3` §1: **[validator]** = `bundle.py` checks it, fixed in both directions
  for every `1.n` (`-spec3` §1.10); **[writer]** = a rule for the P1 writers that the validator does
  not check, tightenable or loosenable with no bump.
- **Vector prefixes.** `EH-`, `BP-`, `BK-`, `BR-`, `BL-`. None collides with `-spec3`'s
  `AV-`/`BV-`/`GV-`, which `cpg/tests/test_bundle.py` parses from `-spec3`'s text.

## 1. Version rule (adds to `-spec3` §8.6 and §2.2's `bundleSchemaVersion` row) [validator]

- **`bundleSchemaVersion` is `"1.0"` or `"1.1"`.** A bundle declares **the maximum, over every
  field, record type and enum value it carries, of the minor that first admitted it**. In `1.1`
  the only new elements are `configEntry.bindParts` and `header.bindKeyCheck`, so a bundle is
  `"1.1"` iff it carries either, and `"1.0"` otherwise (§6.4 V1a–V1d). Bytes stay a function of
  content (`-spec3` §8.4): a P1 writer that emits no `bindParts` (every Joern code unit; the `slice`
  profile of P1-05) writes `"1.0"` exactly as today.
- **Readers.** A `1.1` validator accepts both minors. A `1.0`-only reader refuses a `"1.1"` bundle,
  and the unit becomes `stale-kept` (`-spec3` §8.6, unchanged). The P1 assembler is a `1.1` reader.

## 2. The HMAC primitive, the domain tags and the key [validator: field semantics]

```
entry_hmac(key, domain, value) =
    lowercase-hex( HMAC-SHA-256( key, UTF-8(domain) || 0x00 || UTF-8(NFC(value)) ) )
```

- This is `-spec3` §1.7.5's construction with the record id generalized to a **domain** string.
  `valueHash` is `entry_hmac(key, <the configEntry's own id>, <text>)` and `defaultHash` is
  `entry_hmac(key, <the launchFact's own id>, <default text>)`, unchanged (EH-31 pins the form).
  `domain` is non-empty ASCII without `0x00`, so the split at `0x00` is unique.
- **`bindParts` uses three tags of its own:**

  | Constant | Tag | Hashes |
  |---|---|---|
  | `BIND_HOST_TAG` | `ccg.bindParts.host.v1` | the canonical host text `H` (§3) |
  | `BIND_PATH_TAG` | `ccg.bindParts.path.v1` | `H + "/" + seg₁ + "/" + … + segᵢ`, a path prefix **under its host** (§5.3) |
  | `BIND_KEYCHECK_TAG` | `ccg.bindParts.keycheck.v1` | the empty string (§5.4) |

  A tag contains `.`, so it can never equal a record id (32 lowercase hex): no `bindParts` hash
  collides with a `valueHash` or `defaultHash`, and host and path hashes are separated by
  construction. `H` contains no `/`, so the path input splits uniquely at its first `/`.
- **Deliberately not per-entry.** These hashes exist for **cross-entry equality** (the linker's
  comparison, §7), which `-spec3` §1.7.5 reserves for exactly this case: "a minor bump, and
  bundle-only … never becomes a graph property". Equal hosts, or equal path prefixes under one
  host, hash equally across one system's entries; scoping a path to its host keeps a word W6 left
  clear on a `loopback` path from revealing the same word on an `external` one.
- **Key.** The per-system key of `-spec3` §5.17, 32 bytes, held outside the repository per P1-02
  (`cpg/infra/README.md` § Bundle-at-rest), read only through `ccg.atrest.hmac_key`, which refuses
  both §9 test keys (checks EH-30 and §9.4's `K2` check) with no bypass. `ccg.atrest.bind_key_check`
  is the one implementation of the check. Rotation is a full rebuild (P1-02(d)).

## 3. The URL parse (`parse_bind_url`) [writer; reused by the validator for V6]

One function of the value text, keyless, implemented once in `bundle.py`. Its input is the exact
string the writer would judge for `bindValue`: after the format's own unescaping and §8.1's NFC
(`-spec3` §1.7.2). No URL library parses it. It returns `none` or
`(scheme, port?, H, hostClass, segs)`, where `hostClass` is one of `loopback`, `ip`, `label`,
`dnsname`.

1. **Characters.** `none` unless the value has fewer than 512 characters, every character is in
   `0x21`–`0x7E`, and none of `@ ? # ; \ % $` occurs anywhere. That excludes userinfo, query,
   fragment, `;params`, the backslash differential (SA-5), percent-encoding, interpolation (W3) and
   whitespace in one rule.
2. **Shape.** `none` unless the value fully matches `([A-Za-z][A-Za-z0-9+.-]*)://([^/]*)(/.*)?`.
   `scheme` = group 1, ASCII-lowercased; `none` unless it has 1–16 characters and passes the token
   screen.
3. **Authority** (group 2).
   - **Bracketed:** it starts with `[` and holds a `]`; `inner` = the text between them,
     ASCII-lowercased, must match `[0-9a-f:]{2,39}` and parse as `ipaddress.IPv6Address`. It is
     `none` if the address is IPv4-mapped (`::ffff:0:0/96`), IPv4-compatible (`::/96` other than
     `::1`) or NAT64 (`64:ff9b::/96`): these are excluded **by range**, so every interpreter that
     prints them differently agrees. `H = "[" + str(IPv6Address(inner)) + "]"` (RFC 5952). The
     text after the first `]` is the port part. `hostClass` = `loopback` if `H == "[::1]"`, else
     `ip`.
   - **Otherwise:** if the authority holds a `:`, split at the **last** one into host text and
     port text; ASCII-lowercase the host text. If it matches `-spec3` §1.7.2's `ipv4` (no leading
     zeros), `H` is it and `hostClass` = `loopback` when its first octet is `127`, else `ip`.
     Else it must be a dot-separated name: every label matches §1.7.2's `label` pattern **without**
     the token screen, the whole is at most 253 characters, and the last label is not all digits
     (so `1.2.3` and `99999` are `none`, avoiding WHATWG's IPv4 shorthand). `hostClass` =
     `loopback` for exactly `localhost` and `host.docker.internal`, `label` for any other single
     label, `dnsname` otherwise. Forms a resolver might read as an IP (`1.2.3.0x4`,
     `0x7f.0.0.0x1`) and `foo.localhost` are therefore `dnsname`, hence `external`: never a clear
     host (BP-42–BP-44).
   - **Port:** after the host, either nothing or `:` plus a text matching §1.7.2's `port`
     (`[1-9][0-9]{0,4}`, value ≤ 65535); anything else is `none`. `port` is the integer, absent
     when the URL states none. The writer never fills a scheme default.
4. **Path** (group 3, or empty). `""` and `"/"` give `segs = []`. Otherwise strip **one** trailing
   `/`, split the rest after the leading `/` on `/`; every segment must match
   `[A-Za-z0-9._~-]+` and must not be `.` or `..`, and there are at most 32 segments, else `none`.
   Segment case is kept (paths are case-sensitive). An empty segment (`/a//b`, `/a//`) is `none`.

Only `scheme` is screened here; §4 screens a host only where it is emitted in clear. Every
exclusion above is a writer rule (the validator never sees a withheld raw value), so loosening one
needs no bump (`-spec3` §1.10).

## 4. `hostKind` and `host`

This table is the one home of the clear classes; §6.4 V5 checks it.

| `hostClass` | `hostKind` | `host` in clear |
|---|---|---|
| `loopback` | `loopback` | `H`: `localhost`, `host.docker.internal`, `[::1]`, or an IPv4 in `127.0.0.0/8` |
| `ip` | `ip` | `H`: any other IPv4 literal, or a bracketed canonical IPv6 other than `[::1]` and outside §3's excluded ranges |
| `label` | `service` **iff** `H` is a declared service (below), contains a letter and passes the token screen; else `external` | `H` when `service`; absent when `external` |
| `dnsname` | `external` | absent |

- **Declared service** = the set W2 already uses (`-spec3` §1.7.4): a Compose service name or
  `container_name` in a Compose file of the **same repository** at `builtCommit`, ASCII-lowercased.
  A single label declared only in another repository is `external` here; the linker can still
  resolve it by hash (§7).
- **`host` is present iff `hostKind` ∈ {`loopback`, `ip`, `service`}** (SA-4(b)). Those are the
  advisory's clear classes (Pass 1, "What may pass in clear"), IP literals included by the
  stakeholder decision of 2026-10-29 (SA-13, Q4). An `external` host, the place a capability host
  lives (BV-13's Lambda id), is never in clear in `bindParts`.
- **`hostHmac = entry_hmac(key, BIND_HOST_TAG, H)` for every `hostKind`**, so one comparison serves
  all four (§7).

## 5. The fields

### 5.1 `configEntry` row (adds to `-spec3` §5.17's table)

| Field | Type | Req | Meaning |
|---|---|---|---|
| `bindParts` | object, §5.3 | no | first admitted in `1.1`; keyed binding parts of a URL value, emitted per §6.1. Matching input only: never in the graph and never rendered (§6.3). **[P-later]** until P1-14's writer emits it. |

### 5.2 `header` row (adds to `-spec3` §2.2's table)

| Field | Type | Req | Meaning |
|---|---|---|---|
| `bindKeyCheck` | 64-hex | iff the bundle carries `bindParts` | `entry_hmac(key, BIND_KEYCHECK_TAG, "")`, §5.4 |

### 5.3 The `bindParts` object (closed: no other field)

| Field | Type | Req | Meaning |
|---|---|---|---|
| `scheme` | string `^[a-z][a-z0-9+.-]{0,15}$`, passes the token screen | yes | §3 step 2 |
| `port` | int 1–65535 | iff the URL states a port | §3 step 3; never a scheme default |
| `hostKind` | `loopback` \| `ip` \| `service` \| `external` | yes | §4 |
| `host` | string | iff `hostKind` ≠ `external` | `H`, §4 |
| `hostHmac` | 64-hex | yes | `entry_hmac(key, BIND_HOST_TAG, H)` |
| `pathHmacs` | array of 64-hex, 0–32 items, **in path order** (never sorted; `-spec3` §8.1) | yes, possibly empty | item *i* = `entry_hmac(key, BIND_PATH_TAG, H + "/" + "/".join(segs[0..i]))`, the hash of the **prefix** ending at segment *i*, under host `H` |

Prefix hashes make the linker's prefix match one comparison (§7), and an item's equality implies
the whole prefix's under the same host, never one segment's at another position.

**Determinism.** `bindParts` is a function of the value, the declared-service set (already a W2
input) and the key (already a `valueHash` input). Two builds with the same key are byte-identical;
neither new field is exempt under §8.5.

### 5.4 The key epoch (`bindKeyCheck`)

`bindKeyCheck` identifies the key epoch and reveals nothing about a 32-byte random key. The linker
(P1-17) computes `ccg.atrest.bind_key_check(key)` once per run; a bundle's
`bindParts` are **usable** iff its `header.bindKeyCheck` equals that value
(`ccg.valuehash.bind_parts_usable(header, key)`).

A bundle whose check differs is **not refused**: its code facts do not depend on the key, and
refusing it would loop, because the realistic case is a unit whose rebuild failed after a rotation
and whose kept bundle is itself under the old key. Instead:
- the bundle loads as any valid bundle does, and none of its `bindParts` is ever compared;
- its entries with a `url`-form `bindValue` bind through parts the linker derives under its own key
  (§7, "Which parts"); entries with only `bindParts` stay unbound;
- the refresh report (P1-20) carries one line per such unit, `bind-key-mismatch <unitId>`, with no
  hash value, which `link()` returns beside its links;
- its `valueHash`es differ from the next good build's, which a rotation's full rebuild expects.

## 6. Emission and validation

### 6.1 When the writer emits `bindParts`

A `configEntry` carries `bindParts` iff **all** hold:

- **E1** [writer] `parse_bind_url(value)` is not `none` (§3);
- **E2** [validator: V2] W1 holds (`-spec3` §1.7.4). Argv stays `launchFact`'s; a `launchFact`
  gains no field;
- **E3** [writer] the entry is in a binding role, by P1-11's allow-by-role rule (SA-3, F-6);
- **E4** [validator: V3, plus a writer half] the name rule (`-spec3` §1.7.3) passes on every named
  `keyPath` segment; for a YAML alias or merge-key value the writer also judges the anchor's
  `keyPath` (W3's two-ends rule), which the validator cannot see.

E4 guards the clear `port` and `host`: SA-7's numeric secrets live in the port class, and the
validator rejects `x.ADMIN_PIN` as `bindValue` (BV-11) and must reject it as `bindParts` too. Its
over-redaction (`x.auth_url` fails on `auth`, BP-38) is accepted; loosening it is a major change.

**W7 [writer, an instance of W4], no clear URL without parts.** A `url`-form `bindValue` is emitted
only if `bindParts` is emitted for the entry; otherwise the writer withholds it. So W6 cannot be
bypassed by a URL that §1.7 admits and §3 refuses (the excluded IPv6 ranges, BP-39–BP-41).

### 6.2 `bindValue` beside `bindParts` (amends `-spec3` §1.7.6's residual) [writer]

- **`bindValue`'s admission is `-spec3` §1.7's**, unchanged: a value with `bindParts` keeps its
  `bindValue` only if it passes §1.7 (grammar, screen, name rule, W1–W3), W6 and W7.
- **W6 (an instance of W4).** When `bindParts` is emitted, a `url`-form `bindValue` is kept only if
  `hostKind` ∈ {`loopback`, `service`}: the two kinds whose target is on the same host or the same
  Compose network, where a path word is not a remote bearer capability. For `ip` and `external` the
  parts carry the binding and the clear copy is withheld. The validator still admits such a
  `bindValue` (`-spec3` §1.10: the writer is the stricter side).
- **Effect on `-spec3` §1.7.6's accepted residuals, for `url`-form values under a binding role:**
  the word-only capability (`https://ntfy.sh/deploy-alerts-team`, BP-6), the letters-only token
  (`https://ntfy.sh/kxqzmwvptryhbnjdlc`, BP-7) and the chunked token across host labels (BV-28,
  BP-9) no longer reach the bundle in clear. They remain residuals for a non-URL value (`host`,
  `hostport`), which has no `bindParts`, and for a `loopback`/`service` URL path.
- **Every `-spec3` BV row keeps its validator outcome.** BV-1, BV-5 and BV-21 keep their writer
  `bindValue`; BV-28 is now withheld at the writer.
- **The scheme decision (`-spec3` §1.7.2):** `bindValue`'s scheme list stays closed and unchanged.
  A URL of any other scheme binds through `bindParts`, whose `scheme` is open syntax (BP-17), so no
  future scheme needs a bump.

### 6.3 Never rendered, including on failure paths (amends `-spec3` §1.11's `configEntry` row; P2-6)

- `bindParts`, every field of it **including the clear `host`**, and `header.bindKeyCheck` are
  matching input only. They never become a graph node, property, index or edge property, never
  appear in an answer, a `raw_query` result, the broker audit or a log line, and are never sent to
  an LLM-backed tool (P1-02(c): no such tool reads a bundle).
- **Failure paths.** A validation error, an assembler refusal and a linker or build-report message
  name the record line, the field and the §6.4 rule id (for example
  `line 7: configEntry.bindParts.host: V5g`). They never include the value of `bindValue`, of any
  `bindParts` field or of either key check. `bundle.py` replaces jsonschema's message for these
  fields with that form, and its existing `bindValue` message (`_Semantics._configEntry`) drops the
  value. Any stored failure reason derived from such an error (P1-20's `stale-kept` reason,
  `Unit.failReason`) carries that value-free text or a reason code, never raw exception text.
- P2-6's `bindValue` rule is unchanged and F-7 does not relax it: `-spec3` §1.11's sentence reads
  "A renderer shows the provider's mount path, never a consumer's config URL or any part of it".
  The answer schema's closed `launch` hop (`docs/plans/connected-code-graph-answer-spec.md` §7.1)
  already cannot represent these fields.

### 6.4 Validator checks (adds to `-spec3` §1.12's `bundle.py` list) [validator]

Each id is one conjunct. §9.3 has, for every id, a reject row in which **only that conjunct** fails,
and a valid row at each bound. V2–V6 apply to an entry carrying `bindParts`.

| Id | Check |
|---|---|
| V1a | a `"1.0"` bundle carries neither `bindParts` nor `bindKeyCheck` |
| V1b | a `"1.1"` bundle carries at least one of them |
| V1c | some entry has `bindParts` ⇒ `header.bindKeyCheck` is present |
| V1d | `header.bindKeyCheck` is present ⇒ some entry has `bindParts` |
| V2 | W1 (`-spec3` §1.7.4) |
| V3 | the name rule (`-spec3` §1.7.3) on every named `keyPath` segment |
| V4a | closed object |
| V4b | `scheme`, `hostKind`, `hostHmac` and `pathHmacs` present |
| V4c | `scheme` matches `^[a-z][a-z0-9+.-]{0,15}$` |
| V4d | `scheme` passes the token screen |
| V4e | `port`, when present, is an integer 1–65535 |
| V4f | `hostKind` is in the enum |
| V4g | `hostHmac` is 64 lowercase hex |
| V4h | every `pathHmacs` item is 64 lowercase hex |
| V4i | `pathHmacs` has at most 32 items |
| V5a | `hostKind` `external` ⇒ no `host` |
| V5b | `hostKind` ≠ `external` ⇒ `host` present |
| V5c | `loopback`: `host` is in §4's `loopback` class |
| V5d | `ip`: `host` is a §1.7.2 `ipv4` not in `127.0.0.0/8`, or `"[" ipv6 "]"` (§1.7.2's canonical check) other than `[::1]` |
| V5e | `ip`: a bracketed `host` is outside §3's excluded IPv6 ranges |
| V5f | `service`: `host` matches §1.7.2's `label` pattern |
| V5g | `service`: `host` passes the token screen |
| V5h | `service`: `host` holds a letter |
| V5i | `service`: `host` is not `localhost` |
| V6a | with a `bindValue`: `parse_bind_url(bindValue)` is not `none` (so a `port`, `host` or `hostport` `bindValue` cannot sit beside `bindParts`) |
| V6b | with a `bindValue`: `scheme` equals its parse's |
| V6c | with a `bindValue`: `port` equals its parse's (both absent, or equal) |
| V6d | with a `bindValue`: `hostKind` follows its `hostClass` (`loopback`→`loopback`, `ip`→`ip`, `label`→`service` (W2), `dnsname`→`external`) |
| V6e | with a `bindValue` and a `host`: `host == H` |
| V6f | with a `bindValue`: `len(pathHmacs) == len(segs)` |

The validator holds no key, so it never recomputes a hash, and it cannot check E1, E3, E4's alias
half, W6, W7 or that a `service` is truly declared.

## 7. The linker's comparison (for P1-17, `cpg/ccg/linker/bind.py`)

P1-17 behaviour (4)'s sentence "with `bindParts` the comparison is on `hostHmac`/`pathHmacs`" is
read as this section. The linker consumes only bundles that passed `bundle.validate`, holds the
system key (§2), and compares **only** through the functions below; it never decodes, logs or
stores a hash.

- **Which parts.** A consumer `configEntry` with usable `bindParts` (§5.4) binds by them. One with a
  `url`-form `bindValue` and no usable `bindParts` (a `1.0` bundle, a writer that withheld, or a
  key-epoch mismatch) is turned into parts by the linker itself, with `parse_bind_url` and
  `entry_hmac` under its key, so URL binding has one code path. A non-URL `bindValue` binds by its
  clear text, as today.
- **Host first.** `loopback`, `ip` and `service` use the clear `host` exactly as the existing URL
  and Compose binding rules use a clear host (P1-17 behaviours (4) and (5)). For `external`, the
  linker compares `hostHmac` with `entry_hmac(key, BIND_HOST_TAG, s)` for every Compose service or
  `container_name` `s` declared **anywhere in the system** (the system-wide unit index of P1-17
  (5)), ASCII-lowercased, **and matching §1.7.2's single `label` pattern, without the token screen** (a match reveals only equality with an already-declared name). A dotted declared name
  is never compared: it would let one repository confirm a guess at another's withheld external
  host (BL-16). A match proceeds as `service` `s` under P1-17's rules (a name declared in several
  places is handled as P1-17 handles that today); no match means the target is outside the system,
  the outcome an unbound URL has now.
- **Path, prefix-only (O-5), under the decided host `H`** (the clear `host`, or the matched `s`). A
  provider mount path `M` (from code or a pack) is normalized like §3 step 4: it must start with
  `/`; strip one trailing `/` unless `M` is `/`; split; every segment in `[A-Za-z0-9._~-]+`, not
  `.`/`..`. A mount outside that class matches no consumer by parts. With `d` segments: `d == 0`
  matches every consumer; otherwise it matches iff `len(pathHmacs) ≥ d` and
  `pathHmacs[d-1] == entry_hmac(key, BIND_PATH_TAG, H + "/" + "/".join(M's segments))`. Exact
  equality, where a rule needs it, adds `len(pathHmacs) == d`. Comparison is case-sensitive.
- **Port.** Unchanged: `port`, when present, is matched against launch facts as P1-17 (4) states;
  an absent `port` is the scheme default the pack supplies, decided in the linker, never written
  into the bundle.
- **No new reason or status.** A link bound by parts carries the same `status`/`reason` as one bound
  by clear fields (`binding-confirmed`, `port-shared`, …); its `BOUND_BY` evidence names the
  consumer `ConfigEntry` (`entryId`) and renders the provider's mount path (§6.3).

## 8. What changes where (P1-14 and others)

| Where | Change |
|---|---|
| `cpg/ccg/schemas/bundle-v1.schema.json` | `header.bundleSchemaVersion`: `const "1.0"` → `enum ["1.0","1.1"]`; `header.bindKeyCheck` (`hex64`, optional); `configEntry.bindParts` per §5.3 as a closed object (V4a–V4i, V5a/V5b as an `if`/`then`/`else` on `hostKind`). No `required` list changes. The top-level `description` cites `-spec5` and adds the §6.4 checks to its `bundle.py` list. The schema stays necessary, not sufficient (`-spec3` §1.12). |
| `cpg/ccg/bundle.py` | Public `parse_bind_url(value) -> BindUrl \| None` (a `NamedTuple` `scheme, port, host, host_class, segments`; §3), one implementation, reusing `_ipv4`, `_ipv6`, `_label`, `_port` and `screen_fails`. `_Semantics._header` adds V1a–V1d (cross-record); `_Semantics._configEntry` adds V2–V6. Messages per §6.3, the existing `bindValue` message included. No key, no HMAC. |
| `cpg/ccg/valuehash.py` (created by P1-07, extended by P1-14) | `entry_hmac` as §2; the three tags; `bind_parts(key, value, declared_services) -> dict \| None` (§3–§5.3); `bind_parts_usable(header, key)` (using `ccg.atrest.bind_key_check`) (§5.4); `mount_prefix_hmac(key, host, mount_path) -> tuple[int, str \| None] \| None` (§7). |
| `cpg/extractors/common/writer_rules.py` (P1-07, extended by P1-14) | `bind_parts_allowed(...)`: E2–E4 (§6.1), W6 (§6.2) and W7, so a writer decides them only through this import (P1-07 Done). |
| `cpg/extractors/config/formats.py` | Emit `bindParts` through `writer_rules` and `valuehash.bind_parts`; write `bindKeyCheck` and `"1.1"` exactly when one is emitted. P1-11's tests change: W6 withholds `bindValue` for `ip`/`external` URLs, and W7 plus §1 give its URL entries `bindParts`, `"1.1"` and `bindKeyCheck`. `launch.py` changes only if it emits `configEntry` records; a `launchFact` gains nothing. |
| Linker (P1-17, `bind.py`) | §5.4 and §7, through `valuehash`; the `bind-key-mismatch` line returned beside `link()`'s links to P1-20's refresh report; BR-1–BR-3's binding outcomes. |
| `cpg/ccg/ids.py` | **Nothing.** `bindParts` and `bindKeyCheck` are in no identity tuple; `entryId` is unchanged (`-spec3` §5.17). |
| `cpg/ccg/registry/reasons.yaml` | Nothing: no new reason code. |
| `cpg/tests/test_bundle.py` (P1-14) | Read `-spec5`'s text as `SPEC4_TEXT` is read: `_spec_fields` adds §5.1's row to `configEntry` and §5.2's to `header`; `test_header_constants_match_spec` checks the §1 enum. The stored slice bundles and the unmodified `rich_bundle()` stay valid as `"1.0"` (positive control); failing tests first, the pin and the field not authored in one step (`-spec4` §4). |
| `cpg/tests/test_bindparts.py` (P1-14) | Runs §9.1–§9.3, §9.4's `usable` column and §9.5's hash comparisons, parsed from this text; W7 over §9.2; the §6.3 message rule (for every BK reject row, no mutated string value of six or more characters appears in `str(BundleError)`; BV-12's value through the old `bindValue` message is the positive control that turns red today); and a **mutation pass**: disabling any one V-conjunct in `bundle.py` or the schema turns at least one BK row red. |
| `cpg/tests/test_answer.py` (P1-14) | Add `bindParts`, `hostHmac` and `pathHmacs` to the parametrized "not representable" field list beside `bindValue`. |
| Graph note (`graph-dba`) | Must carry: (1) `-spec3` §9.1's `configEntry` row gains `bindParts` and its `header` row gains `bindKeyCheck` under "Not carried"; (2) no `bindParts` hash or `host` becomes a graph property, edge property or index, `BOUND_BY` evidence included (§2, §6.3); (3) a link bound by parts is represented like one bound by clear fields (§7); (4) a snapshot mixing `"1.0"` and `"1.1"` units: ruled in `docs/plans/connected-code-graph-p1-graph.md` §3 and G4; (5) `Unit.failReason` is value-free (§6.3). |

## 9. Conformance vectors (normative; P1-14 parses them from this text)

**Test key.** Every hash below uses the 32-byte key `000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f`
(hex; bytes `0x00`–`0x1f`); §9.4 also uses a second key,
`202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f` (bytes `0x20`–`0x3f`). Both are
test constants that `ccg.atrest.hmac_key` refuses (§2); the tests pass them to `entry_hmac` directly. Every hex value was computed with Python's `hmac` and recomputed
with `openssl dgst -sha256 -mac HMAC`.

### 9.1 `entry_hmac` vectors

An `(empty)` value is the empty string. EH-31 is §1.7.5's `valueHash` for a record whose id is the
domain shown.

| # | domain | value | `entry_hmac` |
|---|---|---|---|
| EH-1 | `ccg.bindParts.host.v1` | `localhost` | `d8b6feef1935e01b556c286cf51a871df1e71901bdbdd795bd73e2ce1f88d6ba` |
| EH-2 | `ccg.bindParts.host.v1` | `falkordb` | `eb7ef31ea6e40cff1909542d1c4046649993de123a0ba05fd1b16ab31e0441a8` |
| EH-3 | `ccg.bindParts.host.v1` | `ntfy.sh` | `90ab24c6873de6ecde5330b34ef3eaf62e14567af501ff3d5b1e0d73367e8f39` |
| EH-4 | `ccg.bindParts.host.v1` | `abcdefghij0123456789klmnopqrstuv.lambda-url.us-east-1.on.aws` | `3951b189126a466656b16c5adf3479b92e2ca76e273d0a26f1b140ccd4f1fe6b` |
| EH-5 | `ccg.bindParts.host.v1` | `x7k2.q9w4.m1p8.z3v6.example.com` | `fae108d1b7207ea8efd01f6e4031998212f34de4a6856eb21687d18de49b1554` |
| EH-6 | `ccg.bindParts.host.v1` | `[::1]` | `acf4ee9b5936c6073828f2dd90d428c8b54bc18863241509a3af7d7d59723069` |
| EH-7 | `ccg.bindParts.host.v1` | `10.0.0.5` | `9f6f09e8272f9c5305f6de7464181ebf4d7b7affb7c26ef56c63dea94936f415` |
| EH-8 | `ccg.bindParts.host.v1` | `127.0.0.2` | `3013a4c7bb0aa53b2e08e69c88a37af168938017e8cafd8270f3ce48d2d9687b` |
| EH-9 | `ccg.bindParts.host.v1` | `api` | `ba3a9c22347d85e2f4ad47f764ec49b575dba5517eceb0a90fe2717c81a24108` |
| EH-10 | `ccg.bindParts.host.v1` | `worker` | `ba9a06a6c4b0ffa30dc5ff4c7eb4f63be07dedbcb528d9a991245a7fc4d934f1` |
| EH-11 | `ccg.bindParts.host.v1` | `host.docker.internal` | `97d2dfa3d9621a30d78606fdd2e16e567e353b2eff3ce9c3e922dfe9c2921482` |
| EH-12 | `ccg.bindParts.host.v1` | `neo4j.internal` | `410fe1eafbd157b789a8541f60c20289e81217bfbd4902d3a7fbbe367bfd1342` |
| EH-13 | `ccg.bindParts.host.v1` | `[2001:db8::5]` | `613ef3f9092bf0ee0ecfa0741983c728dab3619649aba3530f6eb9dfdfbc3a50` |
| EH-14 | `ccg.bindParts.host.v1` | `hooks.slack.com` | `a41cac1b692bfe9d470a075c8fc7ad031c12ab5f7d9922f57aefeebe1cff7b78` |
| EH-15 | `ccg.bindParts.host.v1` | `1.2.3.0x4` | `5dee0f2ecb82aff859d958e6fbb26367316682fc6e7ec1f813191b9022dd174d` |
| EH-16 | `ccg.bindParts.host.v1` | `0x7f.0.0.0x1` | `958691949b848904b6e804cba86f30003555208af98a5a9064a10fbd89d484cb` |
| EH-17 | `ccg.bindParts.host.v1` | `foo.localhost` | `9041c3be8068d2a40324ec76e162b7d71a57fd87ef8ad2d578df8c68c5dcf618` |
| EH-18 | `ccg.bindParts.path.v1` | `localhost/mcp` | `d11e9ff968c1f54e47492c0cb78353903cf274ab8c951d5eda0ac4882dd8505f` |
| EH-19 | `ccg.bindParts.path.v1` | `localhost/api` | `c2f24641887b90a68fce7fbd2b7cd61c775ec659065997e61bc8529a82ed70cf` |
| EH-20 | `ccg.bindParts.path.v1` | `localhost/api/v1` | `7badba10ad8a60b0824a1e1a20cd89d243e5cc12c0fbd4e4afc9a7292522b139` |
| EH-21 | `ccg.bindParts.path.v1` | `ntfy.sh/x7k2-q9w4-m1p8-z3v6` | `6d09ecca2c3291e62b1c17b6e379a523269860ba3a7a92bd7388333e38519dc5` |
| EH-22 | `ccg.bindParts.path.v1` | `ntfy.sh/deploy-alerts-team` | `0a8e7fe43b3883a64c744bca99f7414e29b3047a3e8b555b908dce79bd71e3a7` |
| EH-23 | `ccg.bindParts.path.v1` | `ntfy.sh/kxqzmwvptryhbnjdlc` | `7ecdc4f0628558fb9b65b432caee5cd4e759ce11d7f5c6fa6b7fc864044af7d1` |
| EH-24 | `ccg.bindParts.path.v1` | `10.0.0.5/api` | `43a9ba1dd91cd6fceee125bdea6c6d467fbfc2db614f483235efb6ebf27b085d` |
| EH-25 | `ccg.bindParts.path.v1` | `host.docker.internal/mcp` | `d57bde01d876b4b91f0e98573b319ea44e7251fc7fa75f2588108bb7c292e3e7` |
| EH-26 | `ccg.bindParts.path.v1` | `[2001:db8::5]/v1` | `0bfee328d8f3bf246cb93ffce200fe334306bf74ee6bc1a0487032e0d02afa9d` |
| EH-27 | `ccg.bindParts.path.v1` | `localhost/api/v2` | `3589d43a13ef1cb148609828ddbb5355901e455b85bc45f660881452babc8b93` |
| EH-28 | `ccg.bindParts.path.v1` | `localhost/mcp/extra` | `c834fdae61b3cb9e968913f63e96450b5cb4fe6cfc78b41d8b1b0a111d6d07ca` |
| EH-29 | `ccg.bindParts.path.v1` | `localhost/MCP` | `92d4dbb7d3c583936aa0ea792af1e0bc96641bf300c2361910b37efef536111b` |
| EH-30 | `ccg.bindParts.keycheck.v1` | (empty) | `aca11439350046b0f12310cffc38748816b211bcbe3cb4bfc1b783d8f98f090a` |
| EH-31 | `0123456789abcdef0123456789abcdef` | `8200` | `e807601590ba94c09fd5ff44ad2a3084476803e1a669fbfe9d5b82411d6da93f` |
| EH-32 | `ccg.bindParts.path.v1` | `worker/mcp` | `208cd2ca1c1bb23055a8a0e142742261fc4828efef5aca8f281e36f81b65597c` |

### 9.2 Writer vectors (`bindValue` and `bindParts` per entry)

**Inputs.** Declared services (§4) for every row: `falkordb`, `server`, `api`. Every row's `keyPath`
is a binding role (E3) except BP-37's. No row is a YAML alias.

**Cell syntax.** The `bindValue` column is the writer's outcome: `bind` (emitted, equal to the
value) or `absent`. The `bindParts` cell is `none` or `;`-separated `field=value` tokens: an absent
`port` or `host` has no token; `hostHmac` and `pathHmacs` name §9.1 rows, whose hex is the expected
field value; `pathHmacs=` with nothing after it is the empty array (items comma-separated, in
order); `pathHmacs=#32` is an array of 32 items, each per §5.3, not pinned row by row. The last
column is commentary.

| # | `keyPath` | value | `bindValue` | `bindParts` | decided by |
|---|---|---|---|---|---|
| BP-1 | `mcpServers.falkor-chat-agent-team.url` | `http://localhost:8200/mcp` | bind | `scheme=http; port=8200; hostKind=loopback; host=localhost; hostHmac=EH-1; pathHmacs=EH-18` | BV-1; loopback keeps `bindValue` (W6) |
| BP-2 | `x.url` | `http://localhost:8000/mcp/` | bind | `scheme=http; port=8000; hostKind=loopback; host=localhost; hostHmac=EH-1; pathHmacs=EH-18` | BV-5; trailing `/` dropped: same `pathHmacs` as BP-1 |
| BP-3 | `x.base_url` | `http://localhost:8000/api/v1` | absent | `scheme=http; port=8000; hostKind=loopback; host=localhost; hostHmac=EH-1; pathHmacs=EH-19,EH-20` | §1.7.2 (two segments); multi-segment base binds by parts |
| BP-4 | `services.x.environment.FALKORDB_URL` | `redis://falkordb:6379` | bind | `scheme=redis; port=6379; hostKind=service; host=falkordb; hostHmac=EH-2; pathHmacs=` | BV-21; `falkordb` declared (W2) |
| BP-5 | `notify.url` | `https://ntfy.sh/x7k2-q9w4-m1p8-z3v6` | absent | `scheme=https; hostKind=external; hostHmac=EH-3; pathHmacs=EH-21` | BV-24 (T6); the capability URL of P1-14 |
| BP-6 | `notify.url` | `https://ntfy.sh/deploy-alerts-team` | absent | `scheme=https; hostKind=external; hostHmac=EH-3; pathHmacs=EH-22` | W6 (validator admits it); the word-only residual |
| BP-7 | `notify.url` | `https://ntfy.sh/kxqzmwvptryhbnjdlc` | absent | `scheme=https; hostKind=external; hostHmac=EH-3; pathHmacs=EH-23` | W6 (validator admits it); the letters-only residual |
| BP-8 | `fn.url` | `https://abcdefghij0123456789klmnopqrstuv.lambda-url.us-east-1.on.aws/` | absent | `scheme=https; hostKind=external; hostHmac=EH-4; pathHmacs=` | BV-13 (T1 on a label); host only hashed |
| BP-9 | `x.url` | `https://x7k2.q9w4.m1p8.z3v6.example.com/` | absent | `scheme=https; hostKind=external; hostHmac=EH-5; pathHmacs=` | W6 (BV-28 validator-admitted) |
| BP-10 | `x.url` | `http://LOCALHOST:8000/mcp` | absent | `scheme=http; port=8000; hostKind=loopback; host=localhost; hostHmac=EH-1; pathHmacs=EH-18` | BV-19 (grammar); host lowercased: same hashes as BP-2 |
| BP-11 | `x.url` | `http://[0:0:0:0:0:0:0:1]:8000/` | absent | `scheme=http; port=8000; hostKind=loopback; host=[::1]; hostHmac=EH-6; pathHmacs=` | §1.7.2 (non-canonical `ipv6`); canonicalized to `[::1]` |
| BP-12 | `x.url` | `http://10.0.0.5:8080/api` | absent | `scheme=http; port=8080; hostKind=ip; host=10.0.0.5; hostHmac=EH-7; pathHmacs=EH-24` | W6 (`ip`) |
| BP-13 | `x.url` | `http://127.0.0.2:9000` | bind | `scheme=http; port=9000; hostKind=loopback; host=127.0.0.2; hostHmac=EH-8; pathHmacs=` | `127.0.0.0/8` is loopback |
| BP-14 | `x.url` | `http://api:8000` | bind | `scheme=http; port=8000; hostKind=service; host=api; hostHmac=EH-9; pathHmacs=` | `api` declared (W2) |
| BP-15 | `x.url` | `http://worker:8000` | absent | `scheme=http; port=8000; hostKind=external; hostHmac=EH-10; pathHmacs=` | W2 (`worker` undeclared): `external` |
| BP-16 | `x.url` | `http://host.docker.internal:8200/mcp` | bind | `scheme=http; port=8200; hostKind=loopback; host=host.docker.internal; hostHmac=EH-11; pathHmacs=EH-25` | loopback; path hash differs from BP-1's (host-scoped) |
| BP-17 | `x.url` | `bolt://neo4j.internal:7687` | absent | `scheme=bolt; port=7687; hostKind=external; hostHmac=EH-12; pathHmacs=` | §1.7.2 (scheme not listed); open scheme in parts |
| BP-18 | `x.url` | `http://[2001:db8::5]:8080/v1/` | absent | `scheme=http; port=8080; hostKind=ip; host=[2001:db8::5]; hostHmac=EH-13; pathHmacs=EH-26` | W6 (`ip`) |
| BP-19 | `DATABASE_URL` | `postgres://u:p@db/app` | absent | `none` | §3 step 1 (`@`: userinfo) |
| BP-20 | `x.url` | `https://svc.example.com/x?sig=abc` | absent | `none` | §3 step 1 (`?`: query) |
| BP-21 | `x.url` | `http://localhost:8000/a#frag` | absent | `none` | §3 step 1 (`#`) |
| BP-22 | `x.url` | `http://localhost:8000\abc` | absent | `none` | §3 step 1 (`\`) |
| BP-23 | `x.url` | `http://localhost:8000/a//b` | absent | `none` | §3 step 4 (empty segment) |
| BP-24 | `x.url` | `http://localhost:8000/a/../b` | absent | `none` | §3 step 4 (dot segment) |
| BP-25 | `x.url` | `http://localhost:8000/a%2Fb` | absent | `none` | §3 step 1 (`%`) |
| BP-26 | `x.url` | `http://localhost:0/` | absent | `none` | §3 step 3 (port 0) |
| BP-27 | `x.url` | `http://localhost:08000/` | absent | `none` | §3 step 3 (leading zero) |
| BP-28 | `x.url` | `http://localhost:${PORT}/mcp` | absent | `none` | §3 step 1 (`$`; W3) |
| BP-29 | `x.url` | `http://127.000.0.1:8000/` | absent | `none` | §3 step 3 (non-canonical IPv4) |
| BP-30 | `x.url` | `http://1.2.3:80/` | absent | `none` | §3 step 3 (all-digit last label) |
| BP-31 | `x.url` | `mailto:ops@example.com` | absent | `none` | §3 step 1 (`@`) and step 2 (no `://`) |
| BP-32 | `x.url` | `http://[fe80::1%25eth0]:80/` | absent | `none` | §3 step 1 (`%`: zone) |
| BP-33 | `x.url` | `http://under_score:80/` | absent | `none` | §3 step 3 (`_` in a label) |
| BP-34 | `x.url` | `http://localhost:8000/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a` | absent | `none` | §3 step 4 (33 segments) |
| BP-35 | `notify.webhook` | `https://hooks.slack.com/services/T0/B0/XXXX` | absent | `none` | E4 (`hook`); parses, refused by the name rule |
| BP-36 | `services.app.command[2]` | `http://localhost:8000/mcp` | absent | `none` | E2 (W1) |
| BP-37 | `x.timeout` | `http://localhost:8000/mcp` | absent | `none` | E3 (no binding role) |
| BP-38 | `x.auth_url` | `http://localhost:9000` | absent | `none` | E4 (`auth`): accepted over-redaction |
| BP-39 | `x.url` | `http://[::ffff:7f00:1]:80/` | absent | `none` | §3 step 3 (IPv4-mapped range) |
| BP-40 | `x.url` | `http://[::7f00:1]/` | absent | `none` | §3 step 3 (IPv4-compatible range) |
| BP-41 | `x.url` | `http://[64:ff9b::7f00:1]/` | absent | `none` | §3 step 3 (NAT64 range) |
| BP-42 | `x.url` | `http://1.2.3.0x4/` | absent | `scheme=http; hostKind=external; hostHmac=EH-15; pathHmacs=` | not `ipv4` by §1.7.2: `external`, host withheld (resolvers may read it as an IP) |
| BP-43 | `x.url` | `http://0x7f.0.0.0x1/` | absent | `scheme=http; hostKind=external; hostHmac=EH-16; pathHmacs=` | as BP-42; never clear `loopback` |
| BP-44 | `x.url` | `http://foo.localhost:8000/` | absent | `scheme=http; port=8000; hostKind=external; hostHmac=EH-17; pathHmacs=` | only `localhost` itself is `loopback`; W6 |
| BP-45 | `x.url` | `http://localhost:65535/mcp` | bind | `scheme=http; port=65535; hostKind=loopback; host=localhost; hostHmac=EH-1; pathHmacs=EH-18` | port at its limit, 65535 |
| BP-46 | `x.url` | `http://localhost:1/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a/a` | absent | `scheme=http; port=1; hostKind=loopback; host=localhost; hostHmac=EH-1; pathHmacs=#32` | §1.7.2 (multi-segment); valid at the limits: port 1, 32 segments |
| BP-47 | `x.url` | `http://worker:8000/mcp` | absent | `scheme=http; port=8000; hostKind=external; hostHmac=EH-10; pathHmacs=EH-32` | W2 (`worker` undeclared); path hashed under `worker` |

### 9.3 Validator vectors (`bundle.validate`)

**Base bundle.** `cpg/tests/test_bundle.py`'s `rich_bundle()`, with header `bundleSchemaVersion`
`"1.1"` and `bindKeyCheck` = EH-30, and its `configEntry` given `keyPath`
`mcpServers.falkor-chat-agent-team.url` (id recomputed through `ids.config_entry_id`), `bindValue`
`http://localhost:8200/mcp` and BP-1's `bindParts`. Counts are recomputed after each mutation.

**Mutation syntax.** One backticked span of `;`-separated `target.field = value`, applied in order:
`target` is `header`, `configEntry` or `bindParts`; `value` is JSON, or `<absent>` (delete the
field), or `EH-n` (that row's hex as a JSON string), or `k × EH-n` (an array of `k` copies). A
`configEntry.keyPath` change recomputes the id. A reject row's rule cell starts with the **one**
§6.4 conjunct it violates; every other conjunct holds.

| # | mutation | outcome | rule |
|---|---|---|---|
| BK-1 | (none) | valid | positive control |
| BK-2 | `configEntry.bindValue = <absent>` | valid | `bindParts` alone |
| BK-3 | `configEntry.bindParts = <absent>; header.bindKeyCheck = <absent>; header.bundleSchemaVersion = "1.0"` | valid | the `1.0` shape |
| BK-4 | `header.bundleSchemaVersion = "1.0"` | reject | V1a |
| BK-5 | `configEntry.bindParts = <absent>; header.bindKeyCheck = <absent>` | reject | V1b |
| BK-6 | `header.bindKeyCheck = <absent>` | reject | V1c |
| BK-7 | `configEntry.bindParts = <absent>` | reject | V1d |
| BK-8 | `configEntry.keyPath = "mcpServers.x.args[0]"; configEntry.bindValue = <absent>` | reject | V2 |
| BK-9 | `configEntry.keyPath = "x.auth_url"; configEntry.bindValue = <absent>` | reject | V3 (`auth`) |
| BK-10 | `bindParts.extra = "x"` | reject | V4a |
| BK-11 | `bindParts.hostHmac = <absent>` | reject | V4b |
| BK-12 | `configEntry.bindValue = <absent>; bindParts.scheme = "HTTP"` | reject | V4c |
| BK-13 | `configEntry.bindValue = <absent>; bindParts.scheme = "abcdefghijklmnopq"` | reject | V4c (17 characters) |
| BK-14 | `configEntry.bindValue = <absent>; bindParts.scheme = "abcdefghijklmnop"` | valid | V4c bound: 16 characters |
| BK-15 | `configEntry.bindValue = <absent>; bindParts.scheme = "x7k2q9w4m1p8z3v6"` | reject | V4d (T1) |
| BK-16 | `configEntry.bindValue = <absent>; bindParts.port = 0` | reject | V4e |
| BK-17 | `configEntry.bindValue = <absent>; bindParts.port = 65536` | reject | V4e |
| BK-18 | `configEntry.bindValue = <absent>; bindParts.port = "8200"` | reject | V4e (not an integer) |
| BK-19 | `configEntry.bindValue = <absent>; bindParts.port = 1` | valid | V4e lower bound |
| BK-20 | `configEntry.bindValue = <absent>; bindParts.port = 65535` | valid | V4e upper bound |
| BK-21 | `configEntry.bindValue = <absent>; bindParts.hostKind = "remote"` | reject | V4f |
| BK-22 | `bindParts.hostHmac = "00"` | reject | V4g |
| BK-23 | `bindParts.pathHmacs = ["D11E9FF968C1F54E47492C0CB78353903CF274AB8C951D5EDA0AC4882DD8505F"]` | reject | V4h (upper-case hex) |
| BK-24 | `configEntry.bindValue = <absent>; bindParts.pathHmacs = 33 × EH-18` | reject | V4i |
| BK-25 | `configEntry.bindValue = <absent>; bindParts.pathHmacs = 32 × EH-18` | valid | V4i bound |
| BK-26 | `configEntry.bindValue = <absent>; bindParts.hostKind = "external"` | reject | V5a |
| BK-27 | `configEntry.bindValue = <absent>; bindParts.hostKind = "external"; bindParts.host = <absent>` | valid | V5a satisfied |
| BK-28 | `bindParts.host = <absent>` | reject | V5b |
| BK-29 | `configEntry.bindValue = <absent>; bindParts.host = "10.0.0.5"` | reject | V5c |
| BK-30 | `configEntry.bindValue = <absent>; bindParts.host = "127.0.0.2"` | valid | `127.0.0.0/8` is loopback |
| BK-31 | `configEntry.bindValue = <absent>; bindParts.hostKind = "ip"; bindParts.host = "127.0.0.1"` | reject | V5d |
| BK-32 | `configEntry.bindValue = <absent>; bindParts.hostKind = "ip"; bindParts.host = "[::1]"` | reject | V5d |
| BK-33 | `configEntry.bindValue = <absent>; bindParts.hostKind = "ip"; bindParts.host = "[2001:db8::5]"` | valid | `ip` class |
| BK-34 | `configEntry.bindValue = <absent>; bindParts.hostKind = "ip"; bindParts.host = "[::7f00:1]"` | reject | V5e (IPv4-compatible) |
| BK-35 | `configEntry.bindValue = <absent>; bindParts.hostKind = "ip"; bindParts.host = "[64:ff9b::7f00:1]"` | reject | V5e (NAT64) |
| BK-36 | `configEntry.bindValue = <absent>; bindParts.hostKind = "service"; bindParts.host = "Api"` | reject | V5f |
| BK-37 | `configEntry.bindValue = <absent>; bindParts.hostKind = "service"; bindParts.host = "x7k2q9w4m1p8z3v6a"` | reject | V5g (T1) |
| BK-38 | `configEntry.bindValue = <absent>; bindParts.hostKind = "service"; bindParts.host = "12345"` | reject | V5h |
| BK-39 | `configEntry.bindValue = <absent>; bindParts.hostKind = "service"; bindParts.host = "localhost"` | reject | V5i |
| BK-40 | `configEntry.bindValue = <absent>; bindParts.hostKind = "service"; bindParts.host = "worker"` | valid | declared-ness is a writer fact |
| BK-41 | `configEntry.bindValue = "8200"` | reject | V6a |
| BK-42 | `bindParts.scheme = "https"` | reject | V6b |
| BK-43 | `bindParts.port = 8000` | reject | V6c |
| BK-44 | `bindParts.port = <absent>` | reject | V6c |
| BK-45 | `configEntry.bindValue = "redis://falkordb:6379"; bindParts.scheme = "redis"; bindParts.port = 6379; bindParts.hostKind = "external"; bindParts.host = <absent>; bindParts.hostHmac = EH-2; bindParts.pathHmacs = []` | reject | V6d (a single-label `bindValue` host is `service`) |
| BK-46 | `configEntry.bindValue = "redis://falkordb:6379"; bindParts.scheme = "redis"; bindParts.port = 6379; bindParts.hostKind = "service"; bindParts.host = "falkordb"; bindParts.hostHmac = EH-2; bindParts.pathHmacs = []` | valid | BP-4's record |
| BK-47 | `configEntry.bindValue = "http://127.0.0.2:8200/mcp"` | reject | V6e |
| BK-48 | `bindParts.pathHmacs = []` | reject | V6f |
| BK-49 | `configEntry.bindValue = <absent>; bindParts.hostKind = "ip"; bindParts.host = "[::ffff:7f00:1]"` | reject | V5e (IPv4-mapped; V5d also fails where the interpreter prints it dotted) |
| BK-50 | `configEntry.bindValue = <absent>; bindParts.host = "[::1]"` | valid | V5c member |
| BK-51 | `configEntry.bindValue = <absent>; bindParts.host = "host.docker.internal"` | valid | V5c member |
| BK-52 | `configEntry.bindValue = <absent>; bindParts.hostKind = "ip"; bindParts.host = "10.0.0.5"` | valid | V5d IPv4 member |

### 9.4 Key-epoch vectors (§5.4)

`K1` is §9's first key, `K2` the second. `bind_key_check(K1)` is EH-30;
`bind_key_check(K2)` is `4d6dd0974c662bfb2eafcf965b6a69b09f6689dc8cd85e85368fc44ed02414ac`. Each row
is BK-1's bundle with the `bindKeyCheck` shown.

| # | bundle `bindKeyCheck` | linker key | `usable` (P1-14) | outcome (P1-17) |
|---|---|---|---|---|
| BR-1 | EH-30 | `K1` | true | binds by `bindParts` (BL-1) |
| BR-2 | `4d6dd0974c662bfb2eafcf965b6a69b09f6689dc8cd85e85368fc44ed02414ac` | `K1` | false | bundle loaded; binds by parts derived from `bindValue` under `K1`; one `bind-key-mismatch` line |
| BR-3 | `4d6dd0974c662bfb2eafcf965b6a69b09f6689dc8cd85e85368fc44ed02414ac` | `K1`, `configEntry.bindValue` absent | false | the entry stays unbound; one `bind-key-mismatch` line |

### 9.5 Linker vectors (§7)

`consumer` names a §9.2 row's `bindParts`. `mount=M` is a provider mount path, hashed under the
consumer's decided host; `service=s` is a Compose name declared elsewhere in the system, compared
through `hostHmac` (the consumer is `external`); `service=s; mount=M` decides the host, then hashes `M` under `s`.

| # | consumer | provider side | outcome | why |
|---|---|---|---|---|
| BL-1 | BP-1 | `mount=/mcp` | match | `d` 1, EH-18 |
| BL-2 | BP-2 | `mount=/mcp` | match | consumer trailing `/` |
| BL-3 | BP-10 | `mount=/mcp` | match | case-folded host, same path |
| BL-4 | BP-3 | `mount=/api` | match | prefix, EH-19 |
| BL-5 | BP-3 | `mount=/api/v1/` | match | provider trailing `/`, `d` 2, EH-20 |
| BL-6 | BP-3 | `mount=/api/v2` | no match | EH-27 ≠ EH-20 |
| BL-7 | BP-1 | `mount=/mcp/extra` | no match | `d` 2 > 1 item |
| BL-8 | BP-1 | `mount=/MCP` | no match | case-sensitive, EH-29 |
| BL-9 | BP-4 | `mount=/` | match | `d` 0 |
| BL-10 | BP-1 | `mount=/api` | no match | EH-19 ≠ EH-18 |
| BL-11 | BP-12 | `mount=/api` | match | EH-24, under `10.0.0.5` |
| BL-12 | BP-1 | `mount=/mcp;v=1` | no match | mount outside the segment class |
| BL-13 | BP-16 | `mount=/mcp` | match | EH-25, under `host.docker.internal` (≠ EH-18) |
| BL-14 | BP-15 | `service=worker` | match | EH-10: proceeds as `service` `worker` |
| BL-15 | BP-9 | `service=worker` | no match | EH-5 ≠ EH-10 |
| BL-16 | BP-6 | `service=ntfy.sh` | no match | a dotted declared name is never compared, although EH-3 would equal |
| BL-17 | BP-47 | `service=worker; mount=/mcp` | match | host EH-10, then EH-32 under the matched `worker` |

## 10. Risks and open points

- **What equality reveals.** Inside the bundle, equal hosts and equal path prefixes under one host
  show as equal hashes; without the key nothing else is recoverable. The graph learns, per bound
  link, the equality its binding asserts (host = a declared service, path prefix = a provider
  mount) and nothing else; an unbound consumer reveals nothing (§6.3, §7).
- **Residuals that remain:** `-spec3` §1.7.6's residuals for non-URL values and for a
  `loopback`/`service` URL's path; and an `external` entry's clear `scheme`, `port` and number of
  path segments, which can fingerprint a third-party service's URL shape without revealing a
  credential (accepted). A key compromise turns every low-entropy host and path into a dictionary
  target; custody is P1-02's.
