# Review: C-1001 P1-29, broker core (diff-scoped)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope and verdict

Reviewed, all uncommitted under `cpg/`: `ccg/broker/*` (9 modules), `ccg/schemas/egress-audit-v1.schema.json`,
`tests/test_broker_{policy,audit,server}.py`, `tests/broker_helpers.py`, and the `pyproject.toml` and `README.md`
edits. Baseline: the P1-29 row of `docs/plans/connected-code-graph-p1.md`, its section in
`docs/plans/connected-code-graph-p1-gates.md`, and `docs/plans/connected-code-graph-egress-spec3.md` v1.4 (§2, §3,
§10.1 tests (14)–(20), §10.2, D8). Root `REVIEW.md` applied (no `cpg/REVIEW.md` exists). Not reviewed: `cpg/sandbox/*`,
`cpg/infra/*`, the toolchain README (P1-48b), and the deep security pass (a separate advisory runs).

**Verdict (pass 1): needs changes; latest verdict: see Pass 2.** One blocker (F1): caller-controlled content can latch the broker closed, before any token
check, contrary to §5.1 and test (14). Everything else is major or below. The core is otherwise strong: I ran the
three broker test files (280 passed, 9 s) and 36 mutants of mine, of which 33 were killed, 1 is equivalent, 2 are
real gaps (F6, F7); see the mutation section.

CPG: considered, not relevant — the code under review is new and uncommitted, so no loaded graph contains it; read directly and executed.

## Findings

### F1 (blocker, verified by execution): caller input latches the broker or crashes the audit path

§5.1: "a caller-supplied value that does not fit its field is recorded as `null` ... so caller input can never make a
record invalid"; §10.1 (14) tests exactly that. Three gaps, each run against `Broker.handle` on a scratch copy of the
test harness:
- **(a) Lone surrogate in `target.path`** (`"/\ud800"`). `policy.sanitize` accepts any `str` of at most 256 characters
  (`policy.py:167`). `parse_strict` accepts the escape (`'{"a":"\ud800"}'` parses). The decision record then fails in
  `json.dumps(...).encode("utf-8")` (`audit.py:169-170`), which is inside `append`'s `except Exception`, so the writer
  **latches**. It works with **no token** (`token: null`): any same-uid process can make the broker answer
  `audit-unwritable` to every run until restart.
- **(b) Lone surrogate in any junk field** (e.g. `component: "\ud800"`). `_digest` does
  `json.dumps(..., ensure_ascii=False)` then `.encode()` (`policy.py:132-133`) and raises `UnicodeEncodeError`
  **outside** any handler: no decision record, no reply, the handler thread dies silently
  (`server.py:298` catches only `OSError`). §5.1 requires null plus a digest. `RunTable.close` has the same shape
  for a surrogate token (`runs.py:71-75`, `token.encode()` with no regex guard).
- **(c) Opaque body with a long key under `R1`/`R7`** (§2.3: "any body"). `payload_fields` lists top-level keys
  verbatim (`server.py:184`), the schema caps each at 64 characters (`egress-audit-v1.schema.json`, `payloadFields`),
  so `{"k"*65: 1}` to an allowed local endpoint makes the broker-built record invalid and latches. Same for an
  `identity.<70-char key>`. Failing closed, but a caller-triggered outage; the spec names no cap on a body key.

**Fix.** (i) `parse_strict` rejects any string (key or value) that is not UTF-8 encodable (a walk beside `_depth_ok`):
a malformed frame, closed with no record, which is §2.2's strictness. (ii) `sanitize`'s predicates also require
encodability, as defence in depth. (iii) For `payloadFields`, when any key does not fit the schema item, record
`payloadFields: null` and `rejected["payloadFields"] = <digest of the key list>` (the §5.1 mechanism), or lift the
schema cap and bound the list; pick one and state it. (iv) Guard `RunTable.close` like `lookup`.
**Test, and where it goes:** a round before the fix lands a red test; it must not be authored in the same change as
the fix. It enumerates every caller-controlled string slot (the `sanitize` picks, `target.path`, each body key and
`identity` subkey, both token slots, `open_run`'s `systemId`/`codebases`/`override`, `close_run`'s token) crossed with
hostile classes (lone surrogate, 65/257/301/10k characters, NUL and other control characters, non-string, nested), and
asserts for each: broker not latched, and either every record on disk validates or no record exists and no upstream
connection was made. Derive the slot list from `sanitize` and the message shapes, so a new field joins the sweep.

### F2 (major, verified): an exception between the durable allow and the `close` record leaves an allow with no close

`_upstream` calls `self.tls_factory()` and `self.exchanger(...)` unguarded (`server.py:202-204`);
`structured.exchange` catches its own socket errors but `make_tls_context` runs outside it, and
`ssl.create_default_context(cafile=...)` raises `FileNotFoundError` when `/etc/ssl/certs/ca-certificates.crt` is
absent. Run: a hosted-only system, `tls_factory` raising; result: records `policy`, `decision allow`, **no close**,
handler thread dead, client sees EOF and reports `BrokerUnavailable`. §5.2 and §8.2 define an allow without a close as
"the broker died mid-exchange" (AC-8 fails the run), and it is untrue here and misleading to the caller.
**Fix:** wrap factory and exchange; on any `Exception`, write the `close` with `closeReason = upstream-error`, return
that. In `_serve_one`, catch `Exception` (log the type, close the connection) so a handler bug is visible in the
stderr log. **Test:** injected `tls_factory` raising, assert exactly one `close` with `upstream-error`.

### F3 (minor, verified): junk `codebases` in a `vulndb` run gives `codebases-not-subset`, not `codebases-in-vulndb`

§3.3 R0 check 6 (non-empty in a `vulndb` run) precedes check 7. `sanitize` nulls `["UPPER"]` and a 9-element list, and
`_r0` (`policy.py:216`) then reports check 7. Run: `["UPPER"]` and `["a"]*9` both give `codebases-not-subset`; a clean
`["front"]` gives `codebases-in-vulndb`. Only the reason differs (still a deny). **Fix:** `sanitize` keeps a flag
"raw value was a non-empty list" in `clean`, and `_r0` uses it for check 6. Test (10)'s order mutant does not cover
this because it feeds clean values; add the junk case.

### F4 (minor, fit): `BrokerClient` imports the upstream-connection module

`client.py:18` imports `structured` for `max_reply_frame` and `UPSTREAM_TIMEOUT_S`. Everything `ccg/refresh.py` and
`ccg/front/*` import from the broker therefore also carries `structured.exchange` (`socket.create_connection`) and
`make_tls_context`. I2's per-module allow-list (§10.1) then passes while the answer path can reach a direct-connection
function. The in-repo check (`test_broker_server.py:285`) forbids only `server` and `policy`. **Fix:** move the frame
constants and `max_*_frame` into `framing.py` (re-export from `structured`), and extend that test to forbid
`structured`. Verified by reading `client.py` and `structured.py`; no outbound connection exists in `client.py` itself
(see below).

### F5 (minor): shipped behaviour that differs from, or is silent against, the spec text

Not defects; each needs one sentence in the spec (a v1.5 line) so the next reader is not surprised. Rulings in the
table below. Items 1, 2, 7 and 9 are the ones that change a stated number, check or signature.

### F6 (minor, verified by mutant): the owner conjunct of the sticky exemption cannot fail

`st.st_uid in (0, uid)` (`audit.py:64`): replacing it with `True` (mutant Q1: any sticky ancestor is accepted whatever
its owner) leaves all tests green, because no test supplies a sticky ancestor owned by another uid (it needs root).
The sticky-bit conjunct is covered (mutant removing it fails `test_the_directory_check_is_the_pass_to_a_real_audit_dir_under_tmp`).
**Fix:** inject the `stat` reader into `_ancestors_ok`, as the peer-uid reader already is, and plant a sticky ancestor
with `st_uid = uid + 1`: it must raise. This is the check §2.0's rule rests on once ruling 2 is settled.

### F7 (nit, verified by mutant): the response `contentType` is never asserted through the reply

Mutant M10 (`server.py:215` sends `"contentType": None`) survives: `BrokerClient.request` returns
`Response.content_type`, and no test reads it. Assert it in the round-trip test (`test_client_round_trip_and_refusal_text`)
against the upstream's header. (Mutant N10, dropping `run.run_kind == "vulndb"` from `R3`, also survives, but it is
equivalent: `R0`'s component-to-run-kind table already admits `vulndb.refresh` only in a `vulndb` run, so the conjunct is
redundant, not untested.)

## Rulings on the delegate's ten decisions

| # | Decision | Ruling |
|---|---|---|
| 1 | Reply cap `ceil(MAX*4/3)+64 KiB` (`structured.py:33-36`) | **Equivalent reading; add a spec sentence.** The reply carries bytes as base64 inside JSON (the spec never says how), so the literal `MAX_RESPONSE_BYTES+64 KiB` (16,842,752) would make any response above about 12.6 MiB undeliverable although §2.3 admits 16 MiB. The code uses `(n+2)//3*4`, the exact size, not an approximation. State the encoding and the formula in §2.2. |
| 2 | Sticky world-writable ancestor accepted if owned by root or the user (`audit.py:59-66`) | **A deviation from §2.0's literal text; behaviourally sound; amend the spec or tighten.** In a sticky directory a different uid cannot rename or remove our entry, so the replacement attack the rule guards is closed. The default broker path (`~/.local/state/...`) never meets it; it exists so `tmp_path` works. Either amend §2.0 ("a sticky ancestor owned by root or the broker's uid is accepted") or keep the strict rule in production with the exemption injected by tests. The user-visible choice is the owner's; F6 notes the missing negative test. Depth belongs to the security advisory. |
| 3 | Exit 64 when interface data is unreadable at start | **Correct.** §2.0: configuration load ("S6, which needs its interface reader") exits 64 on any failure; 69 is the closed list of lock, live socket, directory, audit directory. S6 reads interface data only for a non-loopback literal (`addr.py:104-110`), so a loopback-only manifest needs no `/usr/bin/ip`. At request time the same condition is the §3.4 `address-check-unavailable` deny. |
| 4 | Resolver bypassed for IP literals, forbidden set still runs | **Accept.** `host + "."` on `::1` or `10.0.0.5` is no resolvable name, so the bypass is the only coherent reading. Run: `https://127.0.0.1:443`, `10.0.0.5`, `[::ffff:7f00:1]` each deny with `forbidden-address` and zero resolver calls; `93.184.216.34` allows. One spec sentence. |
| 5 | Null-after-sanitise `codebases` and `requestMode` fallback | **Accept, with F3.** Null maps to `codebases-not-subset` (right outside `vulndb`). For a refresh run the mode falls back to the run's full codebases (`policy.py:196`), the conservative side; it is informational, since the same request is an R0 deny and the rows that read `requestMode` (8.1, 8.8) select answer runs. Run: a refresh run with `["lib"]` (not in scope) records `requestMode = local-only`. State it in §3.3. |
| 6 | ±2⁶³ bound inclusive (`framing.py:30`) | **Accept.** "Outside ±2⁶³" is read literally; floats are not integers and overflow to infinity is rejected (`_float`). |
| 7 | One request per connection | **Accept; state it.** §2.2 speaks of "the connection" without fixing a count. `BrokerClient` opens one per call (`client.py:80`); the cost on a Unix socket is negligible and the framing has no resynchronisation to get wrong. |
| 8 | 256-character `repoId` cap | **Accept.** The spec names none; the manifest schema caps a repoId at 200 and normalization yields `[a-z0-9._-]`, so the audit cap is never reached by a loaded id (the redundant check in `LoadedSystem.from_manifest` is harmless). |
| 9 | `auditId` in the grant | **Accept, and it is required.** §2.1 says the overridden refresh's report names the `policy` record's `auditId`, but the `RunGrant` signature omits it. Add it to the signature. |
| 10 | `payloadFields` for R4 | **Accept.** `identity` is replaced by `identity.subject`/`identity.issuer`, which is what §8.2 row 8.3's subset `{repoNames, identity.subject, identity.issuer}` requires (a literal "plus" would put `identity` in the list and fail it). An allow with no body records `[]`, a deny `null`, as §5.2 says. |

## Policy rows, order and reachability (verified)

`evaluate` (`policy.py:241-271`) follows §3.3's order: `R0` (11 checks in §3.3's order, except F3), then `R2`/`R8`/`R9`
by purpose (before `R1`, even for a `local` endpoint), `R1`, `R3` (all five conditions, including `GET` and
`payloadClass none`), `R4` (deny `payload-shape` on a bad shape, fall through on no matching entry), `R5`, `R6`, `R7`,
`R99`. Reachability, each confirmed by an existing test or my run: `R5` (hosted endpoint, system with a local-only
repository), `R6` (non-model purpose, endpoint that matches no named entry, local-only scope: `R5` always catches a
`model-call` in that mode, so `R6` is reachable only through `access.*` purposes), `R7` (hosted scope), `R99` (hosted
scope, unknown endpoint). `R3` with `POST` falls through to a deny (mutant N7 killed).

## No outbound or direct path (verified)

Sweep of every import and call in `ccg/broker/*.py`: `AF_UNIX` only in `server.py` and `client.py`; the single
network connection is `structured.exchange` (`socket.create_connection` to the address `addr.validate` returned,
never a name; mutant M5 shows a hostname instead of the validated address fails a test); `getaddrinfo` only in
`addr.default_resolver`; `subprocess` only for the absolute `/usr/bin/ip`. No module outside `ccg/broker/` and the tests
imports `ccg.broker` yet (P1-20/-26 are not built). `BrokerClient` contains no connection code to a non-Unix address and
no fallback: a missing broker raises `BrokerUnavailable`, a latched one `BrokerError("audit-unwritable")`
(`test_client_with_no_broker_fails_and_never_connects_directly`, `test_client_with_a_latched_broker_fails_with_the_reason`,
both in the 280 green). Wall-clock deadline verified separately: a TLS peer that sends a record header then one byte per
0.4 s ends in `timeout` at 2.0 s with `UPSTREAM_TIMEOUT_S = 2`.

## Mutation evidence (own run, on a copy under the scratchpad, never in place)

Copy of `ccg/`, `tests/`, `pyproject.toml`. Two controls on the copy: the one test that spawns `python -c` against the
*installed* package (`test_the_default_audit_dir_is_the_codes_own_location_not_the_working_directory`) fails on any copy
and was deselected, and the process tests (they re-exec the installed original) were deselected, since they cannot see
a mutant. My first pass used `-x` and counted that unrelated failure as kills; I discarded it and re-ran every mutant
against the corrected baseline (the numbers below).
- 15 data-plumbing mutants (systemLocalOnly into the run, `resolvedAddr`, `privateAddress`, `bytes`, connect by name,
  `bytesIn`/`bytesOut` swap, decision `runId`, policy `codebases`, `requestMode` from defaults, `contentType`, raw-token
  lookup, `payloadFields` expansion, `allowListDigest`, `auditDir`, grant `effective`): **14 killed, 1 survived (M10, F7).**
- 17 gate-named and spec mutants (R5/R6 order, refresh and answer scope set, `R1` before `R2/R8/R9`, a component-purpose
  row, R0 checks 9/10 swapped, R3 admitting `POST`, R7 ignoring purpose, R4 any body, R3 outside `vulndb`, R5 removed,
  refused override honoured, issuer unchecked, body-size off, bridge nets, gateways, only the first resolved address
  checked): **16 killed, 1 survived (N10, equivalent).**
- 4 audit and process mutants (sticky owner, ancestor check off, peer check inverted, token expiry never): **3 killed,
  1 survived (Q1, F6).** A mutant that dropped the sticky-bit test was killed once rewritten correctly.

## What is solid

Audit writer (single `write(2)` under `flock`, `fsync`, latch on validation, write and fsync failure, FIFO-safe open,
`O_NOFOLLOW`, `nlink`, mode and owner checks, newline repair); token hashing and expiry; closed environment re-export
and isolation; the strict frame parser and its length-first read; exact upstream request construction and the
wall-clock deadline; the 21-reason closed vocabulary shared by `policy.py` and the schema; and a test suite whose
expected values are literals, not read from the code under test.

## Pass 2 (re-check of the fix revision)

**Latest verdict: approve with suggestions.** Run: the four broker test files, 991 passed (22 s); probes of F1-F3 on the
tree; 13 new mutants on a fresh copy; a real broker process for the sticky route.

Dispositions:
- **F1: fixed.** On the tree: lone surrogate in `target.path` (with and without a token), in `component`, and a 65-char
  body key give no latch, valid records (`payloadFields: null` plus `rejected.payloadFields`, `target.path` null plus
  digest); the wire parser rejects `"\ud800"`; `RunTable.close` is regex-guarded. Mutants B3 (parser check off), B4
  (`surrogatepass` off), B5 (`close` unguarded), B6 (field cap off), B2 (`ensure_ascii` off) all killed.
- **F2: fixed.** `tls_factory` raising now gives `policy, decision, close(upstream-error)`; `_serve_one` catches
  `Exception`. Mutant B7 killed.
- **F3: fixed.** `["UPPER"]`, nine elements and `["front"]` give `codebases-in-vulndb`; absent gives `not-subset`; `[]`
  allows. Mutant B8 killed.
- **F4: fixed, shim acceptable.** `client.py` reads `framing` only. Every consumer of the constants reads `framing.X`
  (`structured.py:111,123,126`, `policy.py:253`, `client.py:21`) and tests patch `framing`, so the shim in
  `structured.__getattr__` hides no drift in the code; a test pins its values against literals (`test_broker_server.py:459`,
  mutant B12 killed). Residual: `monkeypatch.setattr(structured, "MAX_...")` would be a silent no-op. Prefer deleting the shim.
- **F5: open, by design.** The spec file is untouched. Ruling 2 is now settled in code (strict), so §2.0 stands as
  written; sentences are still owed for rulings 1, 4, 5, 7, 9, 10 at the feature's close.
- **F6: fixed.** `allow_sticky=False` default, injectable `stat_fn`; mutants B9 (default true) and B10 (owner ignored)
  killed. Production route: the only caller of the ancestor check is `server.py:380` (strict), `AuditWriter` passes
  `check_ancestors=False`, and no argv, environment or config reaches the flag. Run: a real broker with its directory
  under `/tmp` exits 69 (`ancestor /tmp is group- or world-writable (1777)`). **New minor (N1):** mutant B11 (that call
  site passing `allow_sticky=True`) survives: the process tests use `~/.cache`, so strict and sticky behave alike
  there, and the unit tests call `require_dir` directly. Pin the call site, e.g. a process test whose broker directory
  is under `/tmp` expecting exit 69.
- **F7: fixed.** Mutant A6 (`contentType` dropped) killed.

Judgements:
- **Red and fix in one run: nothing material lost.** I ran the new `test_broker_caller_input.py` against a copy of the
  pre-fix `ccg/` (current tests, old code): **124 failed, 567 passed**, so the sweep does go red on the defects. The
  count differs from the delegate's 62 of 346 only because the file and the case grid grew; the evidence that matters,
  red against the old code, reproduces. What is lost is provenance only.
- **`ensure_ascii=True` vs §5.1: no conflict.** §5.1 says "UTF-8"; ASCII output is valid UTF-8 JSON, readers use
  `json.loads`, and it keeps U+2028/U+0085 out of the line. The body digest (`structured.serialize_body`, ensure_ascii False)
  is separate and unchanged.
- **Sanitiser encodability survived mutation (B1): equivalent for safety, behaviourally different.** With the check
  removed, a surrogate in `target.path` reaches the record as the escape `"\ud800"` (valid, no latch) instead of null plus
  a digest. No test pins which; and the wire path never delivers it. Acceptable as defence in depth; one assertion on
  the recorded path of a direct-call surrogate would pin the choice.
- **`~/.cache/ccg-broker-tests`:** acceptable (private, removed after each test, `ignore_errors`), and it keeps the
  process tests honest about the strict check.

New findings: N1 (minor, above), the shim (nit), B1's unpinned choice (nit). No blocker or major remains.
