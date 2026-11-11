# C-1001 P1-07 security review: writer rules, per-entry HMAC, bundle file-write hardening

> **Status:** active · **Owner:** `security-expert` · **Tracks:** C-1001 (P1-07)

**Lens:** code/app security (advisory, not a gate). **Artifact:** uncommitted P1-07 delivery: `cpg/ccg/valuehash.py`, `cpg/extractors/common/writer_rules.py`, `git diff cpg/ccg/bundle.py` (`write_bundle`, `value_grammar_ok`), `cpg/ccg/atrest.py` `hmac_key`, tests `cpg/tests/test_valuehash.py`, `cpg/tests/test_writer_rules.py`. **Baseline:** `docs/plans/connected-code-graph-bundle-spec3.md` §1.6.1/§1.7.4/§1.7.5, `-bundle-spec5.md` §2/§9, plan section P1-07. No `cpg/REVIEW.md` exists; root `REVIEW.md` applied (growth-stops does not apply to a new review; mutation-check line applied in F-6).

CPG: considered, not relevant — the unit's files are new and uncommitted, so any built `cpg_cpg` graph predates them; paths were traced by reading and by live throwaway experiments instead.

**Verdict: approve with suggestions.** No blocker. The HMAC construction, key custody and `write_bundle` hardening are sound (verified live). Two majors concern the writer rules as an *injection control* for the extractors that P1-10/11/12 will build over untrusted repo content.

Experiments run (tmp dirs under the session scratchpad, `ulimit -v 1000000`, `timeout -s KILL`, made-up keys only, no network): collision/encoding probes, ReDoS/size probes, `write_bundle` symlink/hardlink/umask/name-length/dir-target probes, `hmac_key` refusal of both §9 keys via a throwaway key root. Nothing exploitation-shaped against a live system.

## Findings

### Major

**M-1. Only `command`/`args` have an enforced writer-side backstop; W2, `valueHash`, `defaultHash` and `default` rest on convention.** (verified)
- `dumps` -> `validate` -> `_launchFact` (`cpg/ccg/bundle.py:926-937`) re-runs `gate_argv` and rejects any non-fixed-point argv, so an extractor that bypasses `launch_argv_fields` (the AST-guard gaps below) is still stopped before bytes are written. SA-1 is therefore defended in depth. Good.
- W2 is not: `bundle.bind_value_allowed('svc.host', 'hunter')` and `'http://hunter'` return `True` (live). The validator cannot know the declared services, so a later extractor that writes `{"bindValue": ...}` or `"default"` itself, without `writer_rules.bind_fields`/`default_fields`, passes validation and leaks a single-label host (the spec's `${ROOT_PHRASE:-hunter}` case). Same for a wrong `valueHash` domain, a `null` instead of an absent withheld field, or a missing `defaultHash`: the validator cannot recompute the hash (no key).
- The guard (`test_only_writer_rules_writes_argv_fields_in_the_extractors`, `tests/test_writer_rules.py:241`) scans only `command`/`args` keys and the three names `bind_value_allowed`/`value_grammar_ok`/`gate_argv`. No extractor under `cpg/extractors/` writes `bindValue`/`default`/`valueHash`/`defaultHash` yet (`grep` confirms only `writer_rules.py` mentions them), so the gap is latent, but P1-10/11/12 are exactly where it opens.
- Why it matters: those extractors read attacker-controlled repos; a single hand-built `bindValue` is the injection path W2 exists to close.
- Fix: widen `_ARGV_FIELDS` in the guard to `{"command","args","bindValue","default","valueHash","defaultHash"}` (keeping the existing `"type"`-constant exemption, which must then also exempt `decorator`'s `args` as today), and add the `bindValue` word to the names-through-`writer_rules` parametrisation (`bind_value_allowed` is already there). Per root `REVIEW.md`, add a planted-offender row per new key (dict literal, keyword, subscript store) so the check is shown to fail.
- Residual, label theoretical: the guard is still blind to dynamic keys (`dict(zip(names, vals))`, a variable key, `setattr`, `r.update(other)`); the delegate already notes this. It is acceptable only because of the validator backstop for argv. For the W2/hash fields the guard is the only control, so say so in `writer_rules.py`'s docstring, and consider a runtime post-check (below, S-1).

**M-2. Hostile input makes the writer helpers raise instead of withholding, and some exception messages carry repo-derived text.** (verified)
- W4 says withholding is always allowed, but `bind_value`/`bind_fields` call `bundle._keypath_segments` on the raw keyPath, which raises: `'a['`, `'[01]'` -> `BundleError` whose message embeds `{key_path!r}` in full (a 300 037-char message for a 300 kB key, live), and `'["\\a"]'`, `'["\\u00"]'` -> a bare `json.JSONDecodeError` (`bundle.py:469`, `json.loads(m.group("q"))` after a regex that admits `\\.`). Any YAML/JSON/properties file with such a key can abort the whole extractor run (per-file DoS), and the key text, which may be the secret-bearing part of the entry, lands in whatever logs the exception.
- `entry_hmac`/`default_hash` raise `UnicodeEncodeError` for a lone surrogate value (`"\ud800"`, producible from a YAML `"\uD800"` escape), whose message contains the offending character. `default_fields` calls `default_hash` first, so a hostile default also aborts. A single character of a would-be secret in a message is minor in itself; the abort is the issue.
- Not a key leak: `entry_hmac` messages are `key must be bytes`/`key must not be empty`/`domain must be…` with no key material (live; `_record_id` echoes only the id passed in).
- Fix (route to tdd-engineer, decision to architect): in `bind_value`/`launch_default`, catch `(BundleError, ValueError)` from the keyPath parse and return `None` (a keyPath that does not parse cannot be judged, so withhold); in `bundle._keypath_segments` wrap the `json.loads` into `BundleError` and cap the repr (`{key_path[:80]!r}`). For surrogates, the spec's hash is silent: either define the hash of an unencodable value (e.g. withhold the entry and emit no hash) or reject it at the extractor's decode step; architect should pick, not the implementer.

### Minor

**m-1. `write_bundle` name budget regression.** (verified) The temp name is `.<name>.<16 hex>.tmp` (`bundle.py:98`): a 240-char bundle file name now fails `OSError` (ENAMETOOLONG) where `<name>.tmp` fit; 230 chars still works. Practically irrelevant for `<system>.bundle.jsonl` (system names are capped at 64 by `atrest._NAME_RE`), but a one-line `if len(name) > 230` clear error, or documenting the limit, avoids a confusing OSError.

**m-2. Crash residue.** (verified by reading, behaviour not crash-tested) A kill between `os.open` and `os.replace` leaves `.<name>.<hex>.tmp`, 0600, holding the complete post-gate bundle bytes (already screened, so no extra secret exposure beyond the final bundle itself). It ends in `.tmp`, not `.bundle.jsonl` (`atrest.BUNDLE_SUFFIX`), so bundle globs do not pick it up as a bundle. Nothing sweeps them, so they accumulate silently in the work area; a stale-tmp sweep (or a note in the README/infra doc) is worth one line. A tiny window also exists between `os.open` and the `try:` (a `KeyboardInterrupt` there leaks the fd and the file); moving the `try` to cover `os.open`'s result is free.

**m-3. No `fsync` before `os.replace`.** Pre-existing behaviour (the old code had none either); after a power loss the target can be a zero-length file. `atrest._write_new` fsyncs, so the idiom is in the repo; consider matching it, since a truncated bundle is rejected by the validator but costs a rebuild.

**m-4. `entry_hmac` accepts any key length >= 1 byte.** (verified: a 1-byte key hashes) Spec §2 fixes 32 bytes, and `hmac_key` only ever returns 32. A `len(key) == 32` (or `>= 16`) check makes a caller bug (passing a hex string's bytes, or a placeholder) fail loudly. Check first that the §9 vectors and tests use 32-byte keys; they appear to.

### Nits

- `write_bundle` does not check its parent directory's mode or ownership; that is by design (the at-rest area check is `atrest.ensure_private`'s job, called by whoever chooses the path). State that in the docstring so a reader does not assume `write_bundle` enforces custody.
- `atrest.hmac_key`'s temp suffix is `token_hex(4)` (32 bits). It is `O_EXCL|O_NOFOLLOW` in a 0700 dir, so safe; only inconsistent with `write_bundle`'s 64 bits. Not this unit's file.
- `writer_rules.single_label_host` is exported in `__all__` but is only meaningful after the grammar passed (it returned the 1 MB string for `'http://' + 'a'*10**6` live); the docstring says so, fine.

## Answers to the review focus

**(1) HMAC construction. Sound.**
- Framing: `ascii(domain) || 0x00 || utf8(NFC(value))`; `entry_hmac` rejects a domain that is empty, non-ASCII or contains NUL (live: `'a\x00b'` and `''` raise), so the first `0x00` splits uniquely; a value containing NULs or being empty cannot move the split. No length-prefix ambiguity. `writer_rules._record_id` additionally pins the domain to 32 lowercase hex, so case variants of an id cannot produce a second hash for one record, and `bindParts` tags contain `.`, so they cannot collide with a record id (spec5 §2).
- NFC: `é` and `e` + U+0301 hash equal (live), intended by spec; NFKC is not used (test added by the delegate after a surviving mutant).
- Truncation: none (full 64 lowercase hex). No comparison in `valuehash.py`; `atrest._refuse_test_key` uses `hmac.compare_digest` (over public checks, so timing is moot anyway).
- Key hygiene: the key is only an argument; no log, no repr, no write; `entry_hmac` messages carry no key (live). `hmac_key` refused both §9 test keys (bytes 00..1f and 20..3f) from a throwaway key root with `AtRestError`, and a fresh key is 32 bytes, mode 0600 (live).
- Low-entropy values (a port, a boolean, a short token): without the key the HMAC gives no offline dictionary oracle, and per-entry domains (the record id, which is public in the bundle) mean equal values do not even correlate across entries. Whoever holds both the bundle and the system key can brute-force any low-entropy hashed value trivially; spec3 §1.7.5 relies on key custody outside the repo (P1-02) and on "no consumer matches hashes", and explicitly refuses a cross-entry hash because it would hand every raw-query reader an oracle. I found no contradiction. The practical consequence to keep in mind: `defaultHash` is always present even when `default` is withheld (`${ROOT_PHRASE:-hunter}`), so the key protects exactly those withheld low-entropy secrets; the key custody tests (P1-02) are what carry that.

**(2) `write_bundle`.** (verified live under `os.umask(0)`)
- Mode 0600 regardless of umask (`fchmod`); `O_EXCL|O_NOFOLLOW` with a name from `secrets.token_hex(8)` (CSPRNG, 64 bits, unpredictable); so a pre-planted symlink or file at the temp name cannot be written through, and TOCTOU on the temp name is closed (the attacker would need the 64-bit suffix, and `O_EXCL` fails if guessed-and-planted).
- Destination symlink: `os.replace` replaces the link itself, never writes through it (victim file unchanged, link became a regular file). Destination hard link: the original inode is untouched (rename swaps the directory entry). Destination a directory: `IsADirectoryError`, temp unlinked, none left. Validation failure happens before any file is created.
- The remaining race is a hostile writer in the *parent directory* swapping the target name between validation and rename, which any rename-based atomic write has; the directory's custody (`atrest.ensure_private`) is the control, outside this unit.
- Pre-change name `<name>.tmp` was predictable; the change is a strict improvement.

**(3) Injection control.** (verified)
- Through `bind_value` the `$`, `${…}`, backtick, newline, NUL, uppercase-host and fullwidth/Cyrillic-confusable inputs all returned `None`: the value grammar admits only ASCII host/port/url shapes (no `$`), and `len < 512` is checked first. `db`, `db:5432`, `redis://db`, `http://db@evil/`, `db.`, `LOCALHOST`, `DB` withheld when `db` is not a declared service; `localhost`, `localhost:80`, declared services (case-folded on the declared side) and dotted names pass as W2 specifies.
- Argv: `launch_argv_fields` output is the `gate_argv` output; a 20 000-element argv and a 1 MB element complete in ~0.2 s and ~0.08 s, with elements redacted.
- AST guard bypasses (`dict(command=...)` is caught as a keyword; `rec.update({...})` caught via the dict literal; `r.update(args=...)` caught via the keyword). Not caught: `dict(zip(...))`, variable keys, `setattr`, `r.update(other_dict)`, string concatenation of a key. For argv this is covered by the `validate` backstop (M-1); for W2/hash fields it is not (M-1).

**(4) DoS.** (verified) No super-linear pattern found. Adversarial 511-char grammar inputs (dots, colons, separators, hostish, URL) and 1 MB inputs to `value_grammar_ok`, `screen_fails`, `name_fails`, `_keypath_segments` (1 MB keyPath, 200 000 index segments, 400 000 backslashes), `single_label_host` all finished in under 0.5 s with a 1 GB address-space cap and no recursion. Remaining availability issue is the exception path (M-2), not CPU.

## What's solid

- Encoding and domain separation of `entry_hmac`; the delegate's mutant set kills the separator, swap, NFC and domain-check mutants, and the spec §9 vectors are parsed from the spec text, not copied.
- `hmac_key` custody: CSPRNG key, atomic link-create, refusal of the published keys on generation and on every load, `O_NOFOLLOW` chain reads.
- `write_bundle` temp handling and cleanup, and the `validate`-before-write ordering.
- The decision to treat the validator as the SA-1 backstop and make `launch_argv_fields` the sole argv producer.

## Open questions

1. (architect) Spec decision for M-2: what is `valueHash`/`defaultHash` of text that has no UTF-8 encoding (lone surrogate), or should the extractor reject it at decode time?
2. (architect/teco) Should P1-14's `bindParts` writer (the plan says `writer_rules` is extended there) also add the key names it writes (`bindParts`, host/path hashes) to the guard from M-1?
3. Is a stale-`.tmp` sweep owned by P1-07, by the at-rest tooling (P1-02), or accepted as a known leftover (m-2)?

## Routing

- **tdd-engineer:** M-1 guard widening with planted-offender rows (mutation-check per root `REVIEW.md`); M-2 `bind_value`/`launch_default`/`_keypath_segments` fail-closed handling with tests for `'["\\a"]'`, `'a['`, a 300 kB bad key (assert no keyPath text beyond a cap in the message); m-1, m-2 (`try` placement), m-4.
- **architect:** Open question 1 (hash of unencodable text); confirm whether the fail-closed rule in M-2 is a W4 spec statement.
- **devops:** nothing.

## Pass 2

**Verdict: approve.** No blocker, no major open. Re-checked on the fix-pass tree with the same throwaway probes (made-up 32-byte keys, no custodian root, `ulimit -v 1000000`).

- **M-1 (guard widening): fixed.** `tests/test_writer_rules.py:owned_field_writes` flags the six keys: dict literal, subscript store, `dict(k=)`, `.update(k=)`, `setdefault`, and a `"bind"+"Value"` concatenation; `{"type":"decorator","args":..}` is exempt but `{"type":"decorator","command":..}` is flagged. Still blind (live, documented and pinned): `r.update(other)`, `dict(zip(..))`, `r[k]=v`.
- **M-1 backstop `audit_records`: works, with two stated limits.** Live: a `valueHash` under a keyPath domain or another record's id is reported; `hunter` bindValue (W2), a secret-named keyPath, `null` or missing hash, a W2-violating `default`, a raw-secret argv are all reported; correct records give `[]`; messages carry only type, capped id and obligation (no value). Limits: a hash of a *withheld* value with a wrong domain (or a garbage 64-hex) passes unless the caller supplies `texts` (live; documented). Nothing calls it yet: the only references are its own tests and the README row, and no extractor writes these fields yet. Its value is entirely a P1-11/P1-12 done-condition; route to architect/teco to put "the writer's test runs `audit_records` with `texts`" into those steps' Done lines.
- **M-2 (hostile keyPath): fixed.** `'a['`, `'[01]'`, `'["\\a"]'`, `'["\\u00"]'` and a 300 kB key all return `None` (withheld) from `bind_value`; no exception. `bundle._keypath_segments` now raises a capped message (116 chars for a 400-char input via `cap_repr`). Nit: the cap bounds the echo to 80 chars but a secret inside those 80 chars is still echoed (live: `SECRET` appeared); acceptable for a validator error, do not log it from the extractor.
- **M-2 (lone surrogate): fixed.** `bind_value` withholds a surrogate value or keyPath, `default_fields` returns `{}`, `entry_hmac`/`value_hash` raise `UnhashableValue("value has no UTF-8 encoding")` with a constant message (live).
- **m-1 (name budget): accepted as documented** (docstring note only; the 64-char system-name cap makes it moot).
- **m-2 (crash residue / `try` placement): fixed in code** (`try` begins before the `os.open` result is used, fd closed if `fdopen` fails, foreign names never unlinked); stale-tmp sweep ownership is still the architect's call (open question 3 stands). Not crash-tested, read only.
- **m-3 (no `fsync`): not fixed**, pre-existing, still minor.
- **m-4 (exact key length): fixed.** `entry_hmac` rejects 0, 16, 31, 33 and 64-byte keys (live). No legitimate caller broken: the only production path is `writer_rules.system_key` -> `atrest.hmac_key`, always 32 bytes (`KEY_BYTES`); `atrest.bind_key_check` inlines its own HMAC and does not go through `entry_hmac`; every other `entry_hmac` call is in tests, which use 32-byte keys.
