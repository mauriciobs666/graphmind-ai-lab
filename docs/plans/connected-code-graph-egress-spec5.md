# Connected code graph — egress spec, delta for the key-holder forward (U-S5)

> **Status:** active · **Owner:** `architect` · **Tracks:** GAL-F0046 · **Item:** `GAL-F0046` · **Extends:** `docs/plans/connected-code-graph-egress-spec3.md`, `docs/plans/connected-code-graph-egress-spec4.md`

This states only what changes against `docs/plans/connected-code-graph-egress-spec3.md` (spec3, the base) as amended by
`docs/plans/connected-code-graph-egress-spec4.md` (spec4, the latest egress delta). Both stay authoritative everywhere not
named here; section numbers are spec3's. The design, wire and threat model live in `docs/plans/llm-gateway.md` (the plan):
§3.3 (refusal exchange), §3.4 (holder, credential, paths into the holder), §3.6 (installed manifest), §1 (what is and is
not guaranteed). They are cited, not restated. This is unit U-S5 of the plan (§4); build unit D9 implements it. Each fact
below names the code symbol it was read from (`cpg/ccg/broker/`: `server.py`, `structured.py`, `policy.py`, `audit.py`,
`client.py`; `cpg/ccg/schemas/egress-audit-v1.schema.json`; the holder side is `llm-gateway/src/llmgateway/framing.py` and
`keyholder.py`).

## 1. §2.0 "Inputs, closed": new options, credential, owner protection

**Argv gains** (spec3 §2.0 listed `--system`, `--socket` only). All are parsed by `server.py:_Parser`; a violation is exit 64.

| Option | Meaning |
|---|---|
| `--key-holder-socket <path>` | Enables forwarding of R7 allows (§3). Absent: no hosted exchange is ever made (§3, last row). Needs `--holder-uid` and `--require-owner-protected`. |
| `--holder-uid <int>` | The uid that must own the socket's directory (plan §3.4: a caller refuses a socket whose directory is not holder-owned). The plan names `holder_uid` in `[deployment]` but gives the broker no source for it, and the broker reads no `gateway.toml`; it is an option. A value equal to the broker's own `os.getuid()` is exit 64: the holder runs under another account, and a same-uid "holder" would let any agent bind a fake one. |
| `--holder-cred <file>` | The broker's credential (plan §3.4, caller `broker`). Default `<passwd home>/.config/llmgateway/callers/broker.cred` (the plan §3.4 artifact table's path). Valid only with `--key-holder-socket`. |
| `--require-owner-protected` | Every `--system` manifest must pass the owner-protection check below before it is loaded. Mandatory with `--key-holder-socket`: giving the socket without it is exit 64, before anything is bound or locked. Allowed alone. |

`--system` is unchanged in form; in a hosted deployment its value is the installed manifest `/etc/llmgateway/ccg/system.yaml`
(plan §3.6), never the tracked `cpg/ccg/system.yaml`. The broker still loads manifests once, in `main`; a changed manifest is
a restart (spec3 §2.0, plan D-4).

**Static check of hosted entries (resolution-free).** Because R7 skips address validation (§3 item 1), `server.load_systems` checks
each `egress.hosted[]` entry after `check_local_entries` and before accepting the system, with no DNS and no connection: the
scheme is `https`, and when the host is an IP literal (`addr._literal`) it is not forbidden by `addr._forbidden(addr, data, local_addrs, False)`, where `data` is the interface data already read for S6 and
`local_addrs` is the set of IP literals of the manifest's `local` entries (built as `addr.validate` builds it); `private_ok` is
always false, so a private literal on a hosted entry fails at load even with `privateAddress: true` (the flag has no effect on R7). Failure is `ManifestError` naming the file, exit 64. A name host
passes (only the holder can resolve it, plan §3.4).

**Owner-protection check** (new module `cpg/ccg/broker/ownerprot.py`; the broker must not import `llmgateway`, so this is
its own implementation, held to parity with `llmgateway.ownercheck.read_owner_file` by a test that feeds both the same planted
files). A manifest passes only when: opened `O_NOFOLLOW`; `fstat` on the descriptor shows a regular file; owner uid in
`trusted_uids`; `mode & 0o022 == 0`; every ancestor, walked by descriptor, is a non-symlink directory owned by a trusted uid
and not group/other-writable. The bytes are read from that descriptor and **parsed from those bytes**, not re-read by path:
`ccg.manifest.load_manifest(path)` reads by path (`Path(path).read_text()`), so D9 adds `ccg.manifest.load_manifest_text`
(`load_manifest` delegates to it) and `server.load_systems` calls it with the verified bytes. `trusted_uids` is `{0}` in
production and an injected parameter of `main`/`load_systems` for tests only; there is no option or environment variable for
it. A failure is `PreconditionError`, exit 69, naming the file and the failed condition (spec3 §2.0's exit-code class for a
failed start precondition).

**Credential intake.** `main` reads the file once, and never again. `reexec_if_needed` runs only in `server.py`'s `if __name__ == "__main__"` guard,
before `main`, so a process launched as `python -m ccg.broker.server` has re-executed under the closed environment before `main`
reads the file (the environment keeps `HOME` and the path arrives in argv, so the re-exec loses nothing); a test calling `main`
directly skips the re-exec and reads the same way. Rotation is a broker restart: a rotated credential is a broker restart (plan §3.4,
`--rotate`). The file must be opened `O_NOFOLLOW`, a regular file, owned by the broker's uid, `mode & 0o077 == 0`,
`nlink == 1` (else `PreconditionError`, exit 69); its content is the hex of exactly 32 bytes (the form of
`llmgateway.framing.load_secret`), else exit 64. No message, log line or record carries any byte of it (§5.1's rule for the run
token applies). Startup does **not** require the holder to be reachable: the holder may start later or restart under it.

## 2. §2.2: one new request field, `gatewayRunId`

`EgressRequest` gains `gatewayRunId: str | absent` (plan §3.4, "paths into the holder"). It is the run id the **holder's
ledger** bounds (plan §3.4 "FR-10 ledger"); it is unrelated to the broker's own `runId` (the `policy` record's uuid) and is
never derived from it.

- **Validation**, in `policy.sanitize`, **outside** its `pick` helper (`pick` records a `rejected` entry for any value failing
  its predicate, `None` included, which would add `rejected["gatewayRunId"]` to every request that lacks the field). **Absent**
  (key missing or JSON `null`) is `None` with **no** `rejected` entry, so every existing record, including `rejected == {}` for a
  clean request, is unchanged. **Present** must be a string, UTF-8 encodable, and full-match `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}`;
  anything else (wrong type, `""`, over-long, a character outside the set) becomes `None` and its SHA-256 goes to
  `rejected["gatewayRunId"]` (`decision.rejected` admits any key with a sha256 value; spec4 §4 "caller input never invalidates a
  record"). It is neither required nor denied on: no `R0` check, no deny reason (spec3 §3.5's 21 values stay). The consequence
  arrives from the holder, as `run_id_missing` or the owner's `unmatched` rule (§4).
- **Use.** Only an R7 forward reads it (§3); it is the holder request's `run_id` (empty string when `None`, as
  `llmgateway.framing.build_request` does). R1, R3, R4 ignore it. It is not copied to any audit record (§5).
- **Unknown request keys** are still silently ignored, as today (`sanitize` reads a fixed key set). "Closed" means the read set
  is fixed. The plan §3.6 rule that a request carrying a marker or scope field is refused is the gateway `Request` type's, not
  the broker's; the broker reads no marker from a request, so the extra key is inert here, and a test pins that (§7, item 7).

## 3. §2.3 and §3.4: an R7 allow is forwarded to the holder, never sent by the broker

**Replaces, for R7 only, spec3 §2.3's "the broker builds the whole upstream request", §2.3's "Connection", and §3.4's naming of
R7.** R1, R3 and R4 are exactly as before (`Broker.exchanger`, `structured.exchange`, §3.4 address validation).

1. **No broker-side resolution or connection for R7.** In `Broker.request`, `addr_mod.validate` runs only when
   `verdict.section != "hosted"`; for R7 the decision record has `resolvedAddr: null` and `privateAddress: false` (the schema
   allows null on an allow; `resolvedAddr`'s meaning stays "the address connected to", and the broker connects to none). The
   resolver, interface reader and TLS factory are not called. A manifest `hosted[].privateAddress` has no effect on R7; the
   holder's `allow_private` governs (plan §3.5). *Why:* the holder resolves, pins and verifies the address itself (`Refusal`
   `address_forbidden`, plan §3.4); a second resolution at the broker could disagree with it (split DNS) and adds a lookup for
   nothing. *Rejected:* keeping the broker's check as an advisory pre-deny; it would keep spec3 §3.4's forbidden-set coverage
   for R7 at the price of that disagreement. Reversal trigger: a reviewer wants defence in depth over single ownership.
2. **`Broker.__init__` gains `hosted_exchanger`** (default `None`); `Broker.exchanger` keeps its signature and its R1/R3/R4
   role. `_upstream` calls `hosted_exchanger` when `verdict.rule == "R7-HOSTED-SERVICE"`, with
   `(endpoint, method, path, body_bytes, gateway_run_id)`: `endpoint` is the matched entry's canonical origin `scheme://host:port` (`ccg.manifest.canonical_endpoint`; the same
   string `policy.LoadedSystem.lookup` matched, explicit default port included), and the holder must key providers by the canonical
   origin of `base_url` (D7b: `keyholder.canonical_origin`, applied to each provider's `base_url` when `Policy.providers` is built, so a `base_url` written without the default port still matches), `path` is the entry's stored string (never the caller's, spec3 §2.3), and
   `body_bytes` is the `structured.serialize_body` output whose digest is the decision's `bodySha256`. It returns a
   `structured.Exchange`; any exception it raises is caught by `_upstream`'s existing `except Exception` and becomes
   `Exchange("upstream-error", None, None, None, 0, 0)`, so spec4 §5's "every allow is followed by exactly one `close`" holds.
3. **The exchanger** is `cpg/ccg/broker/holderclient.py` (new). It imports nothing from `llmgateway` (plan §3.1 test:
   `ccg.broker` never imports it), so it re-implements the wire of `llmgateway.framing` byte for byte: `request_mac`,
   `build_request` (method `POST`, caller `broker`, a fresh 128-bit nonce per call, `ts` from the wall clock via an injected
   `now`), `parse_reply`, `check_socket_dir(socket, holder_uid)` before every connect, one request and one reply per
   connection. A parity test (§7, item 11) pins it to the real module on the same vectors.
4. **Reply to `Exchange`**, closed table (`llmgateway.framing.parse_reply` semantics; the reply-frame cap is
   `ccg.broker.framing.MAX_RESPONSE_BYTES + FRAME_SLACK` passed as `max_reply_bytes`, the parameter of
   `llmgateway.framing.exchange`; the whole exchange is bounded by `ccg.broker.framing.UPSTREAM_TIMEOUT_S` (300) as a wall-clock
   deadline, longer than the holder's `Policy.upstream_timeout`, default 120 from `upstream_timeout_seconds`). **Holder start check
   (D7b):** `keyholder.build_policy` refuses to start (`HolderStartError`) when `max_response_bytes` exceeds 16 MiB
   (`ccg.broker.framing.MAX_RESPONSE_BYTES`; the holder's default is 8 MiB), so a holder reply can never be over the broker's cap.

   | Holder outcome | `Exchange` |
   |---|---|
   | MAC-verified reply, `closeReason = completed`, status in 100–599 | `completed`, that status, `contentType` (`None` when empty), the body |
   | MAC-verified reply, `closeReason` ∈ `upstream-error`, `timeout`, `response-too-large` | that reason; status, content type and body `None` (the holder sends status 0) |
   | unsigned reply whose `refusal_code` is `unknown_caller` or `bad_credential` (the two pre-authentication refusals the holder cannot sign) | `completed`, status 499, as carried |
   | any other unsigned reply; a MAC that does not verify; a malformed header; a `closeReason` outside the four; `completed` with a status outside 100–599 | `upstream-error` |
   | socket absent or refused, directory not owned by `--holder-uid` or group/other-writable, any `OSError` other than a timeout | `upstream-error` |
   | reply frame over the cap | `response-too-large` |
   | deadline reached (`TimeoutError`/`socket.timeout`, tested before the `OSError` row, as `structured.exchange` orders its handlers) | `timeout` |

   A status outside 100–599 is turned into `upstream-error` here because `close.httpStatus` is bounded 100–599 and a
   broker-built record that fails validation latches the broker (spec3 §5.1, `AuditWriter.append`).
5. **Holder refusals are completed exchanges** (plan §3.3). A signed reply `status = 499`,
   `contentType = application/vnd.llmgateway.refusal+json`, body `{"holderRefusal": "<code>"}` is row 1 of the table: `_upstream`
   builds `response = {status: 499, contentType, bodyB64}` for any `close_reason == "completed"` and never inspects status,
   content type or body, so the reply to the caller is `{"ok": true, "decision": <allow>, "response": {...}, "closeReason": "completed"}`
   and `BrokerClient.request` returns it as a `Response` unchanged. It is therefore never `upstream-error`. The broker neither
   parses nor classifies the code (the plan's code→class table is the gateway's, plan §3.3); a provider that itself answers
   with status 499 or that content type is forwarded identically and fails closed at the gateway's classifier.
6. **No holder configured** (`--key-holder-socket` absent, `hosted_exchanger is None`): an R7 allow is decided and recorded as
   today, then `_upstream` closes it `upstream-error` with `bytesOut = 0` and makes no connection of any kind, direct
   included. This **narrows spec3 §2.3**: the broker never sends a hosted request itself (it holds no key and the plan has no
   direct hosted path).

## 4. §3.3: rule table

No row, order, reason or deny vocabulary changes. In particular `R5-SYSTEM-LOCAL-MODEL` and `R6-STICKY-LOCAL` are evaluated by
`policy.evaluate` from the broker's loaded manifest and the run alone; **no input from the gateway, the request or the
holder reaches them**, so they refuse independently of whatever the gateway's marker source said (plan §3.6 "the broker's R5/R6
decision on the same manifest stays authoritative"; AC-21). A refused request reaches neither `hosted_exchanger` nor a socket. R7's
condition is unchanged; only what an R7 allow does is (§3). spec3 §3.4 now applies to R3 and R4 only (and R1's local re-check).

## 5. §5.1/§5.2: audit

**No schema change.** `egress-audit-v1.schema.json` stays as is: it is closed (`additionalProperties: false`, spec3 §8.14), and
every field the forward needs already exists.

- **Hosted marking.** A forwarded hosted exchange is the `decision` with `rule = R7-HOSTED-SERVICE`, `resolvedAddr = null`,
  `requestMode = hosted`, `purpose = model-call`, followed by its `close`. R7 never connects directly any more (§3, items 1 and 6), so
  `R7` is exactly "forwarded to the holder". No `hosted` flag or `gatewayRunId` field is added; the join between a broker
  record and the holder's `calls.log` is `ts`, `endpoint` and status. *Rejected:* a new `gatewayRunId` field (a schema
  revision, and it would put a caller-chosen string in the audit); reversal trigger: an auditor needs the join.
- **`close` for a refusal.** `httpStatus = 499`, `closeReason = completed`; the code is not in the record (no field, and the
  body is not hashed by the close record). `bytesOut`/`bytesIn` now count bytes written to and read from the **holder
  socket**, frame prefix included (spec3 §5.2 said "the upstream socket"); for a forward the broker never sees the provider
  socket.
- **`decision.endpoint`, `target`, `payloadFields`, `bytes`, `bodySha256`** keep their spec3 meaning: the body the holder is sent
  is the body digested.
- **Resolved address is in the holder's log, not here.** Because the broker records no address for R7, the holder's `calls.log`
  line (`keyholder.Holder._log_call`: `ts, caller, run_id, endpoint, status, tokens`) gains the address it connected to (D7b; plan §3.4
  names the log's fields, so that line needs the addition).
- **Residual, stated as the plan states it (§1, FR-8's limitation).** The audit directory is writable by the broker's uid
  (spec3 §2.6, R-E7), so `egress.jsonl` is evidence of the broker's behaviour, not tamper-proof. A same-uid process holding the
  broker credential can call the holder directly: that call leaves **no** broker record (the holder's `calls.log` has one). So
  "no `allow` in the audit" does not by itself show "nothing was sent"; an AC-21 assertion therefore pairs the audit with the
  provider stand-in's own request log (§7).

**Stated limit: the run-id namespace is unauthenticated.** `gatewayRunId` is caller-chosen and the holder's bound is per id, so a
caller that names another run's id consumes that run's bound; the broker cannot prevent it (the token proves membership of a broker
run, not ownership of a holder run id). The window ceiling still bounds total spend (plan §1, §3.4).

## 6. Unchanged (so reviewers need not look)

The peer-uid check, the socket mode and directory checks, the closed-environment re-exec, the singleton lock, the audit
writer's hardening and latch (spec3 §2.0, §5.1, §2.5), framing and frame caps toward callers (spec3 §2.2, spec4 §3), run
tokens and `open_run`/`close_run`, R0 checks and their order, the closed enumerations and `answer.*` components, mode
resolution (§3.2), `policy` records, the schema, `BrokerClient` (the new field rides inside the existing `request` dict),
and R1/R3/R4 transport including §2.3's header set and test (18). AC-8 (spec3 §8): hosted rows are vacuous in P1 as before; row
8.16 stays computable, and a hosted forward adds only `AF_UNIX` connects, which 8.16 ignores.

## 7. §10.1: tests the delta requires (D9; each is red before its code)

Altitudes: unit (`Broker.handle` with injected exchangers, as `cpg/tests/broker_helpers.py:Harness` does), process (`main` and
`BrokerServer`, as `cpg/tests/test_broker_server.py`), integration (the real broker subprocess against the real
`llm-gateway` `keyholder` and a logging HTTPS stand-in for the provider, plan §5.1). Expected values are literals in the test.
In order:

1. **R7 forward (unit).** A hosted allow calls `hosted_exchanger` exactly once with the entry's canonical `endpoint`, `"POST"`,
   the entry's path string (a caller path that differs only by an equal-looking spelling is already `path-not-allowed`), the
   serialized body whose SHA-256 equals `decision.bodySha256`, and the validated `gatewayRunId`; the resolver, interface reader,
   TLS factory and `Broker.exchanger` record zero calls; the decision has `resolvedAddr: null`, `privateAddress: false`; every
   written record validates against the unchanged schema. R1, R3 and R4 still use `Broker.exchanger` and the resolver.
2. **Exchange mapping.** A scripted `AF_UNIX` double (directory owned by the test uid, passed as `--holder-uid`) answers each row
   of §3 item 4, including an unsigned `bad_credential`, an unsigned reply of any other kind, a bad MAC, a `closeReason` outside
   the set, `completed` with status 0 and with status 600, an over-cap frame, a hang past an injected deadline, and a refused
   connect; assert the `Exchange` and the resulting `close` record.
3. **Refusal passthrough (the claim checked in plan review, executed).** Through `Broker.handle` and through a served
   `BrokerServer` + `BrokerClient`: a signed 499 refusal yields a reply with `response.status == 499`,
   `contentType == "application/vnd.llmgateway.refusal+json"`, the body bytes intact, `closeReason == "completed"`; exactly one
   `decision` allow and one `close` with `httpStatus: 499`, `closeReason: completed`; the audit writer is not latched. A mutant
   that lets `_upstream` answer a non-2xx completed exchange with `response: None` fails this test.
4. **No holder.** Without `hosted_exchanger`: an R7 allow is followed by a `close` `upstream-error`, `bytesOut: 0`, `response: None`;
   no socket is opened and `Broker.exchanger` is not called. A mutant that falls back to `Broker.exchanger` fails.
5. **`gatewayRunId`.** A valid id reaches the exchanger; absent and `null` reach it as `None`/empty with **no** `rejected` entry
   (`rejected == {}`); an integer, `""`, 129 characters, a space, `|`, a newline and a non-ASCII letter reach it as `None`/empty
   with `rejected["gatewayRunId"]` equal to the digest of the raw value; every record is valid; an R1 call carrying it ignores it; a denied request never calls the exchanger.
6. **R5/R6 refuse independently of the gateway (AC-21).** With a loaded manifest where a codebase is `local-only`: (a) a
   `model-call` to a hosted endpoint, in a run that touches only hosted codebases of that system (`systemLocalOnly`), is
   `deny R5 system-local-only-model`, and so is one in a run that touches the local-only codebase (R5 precedes R6); (b) a
   non-`model-call` request (`answer.pr-adapter`, `pr-addon`) to the hosted endpoint in a run whose mode `M` is `local-only`
   is `deny R6 sticky-local-only` (R6 is unreachable for a `model-call` in a `systemLocalOnly` run); in each a `decision` deny
   record exists, there is no `close`, `hosted_exchanger` has zero calls and a counting listener on the holder socket path
   sees zero connections. The request carries, in addition, fields a gateway might send (`marker`, `scope`, `unrestricted: true`,
   `gatewayRunId`): the verdict is unchanged. Mutants: R5 deleted (the R7 allow appears); R6 deleted (the deny's rule
   becomes `R99-DEFAULT-DENY`); each fails the test.
7. **Unknown request keys are inert** (§2): an extra key changes no verdict and no record field.
8. **Options and owner protection (process).** `--key-holder-socket` without `--require-owner-protected`, without `--holder-uid`,
   `--holder-cred` without the socket, `--holder-uid` equal to the own uid, a hosted entry with scheme `http` or an IP-literal host
   that is loopback, private or a host address: exit 64, no socket file, no lock held, no audit file created. With
   `--require-owner-protected` and an injected `trusted_uids`: a manifest that is user-writable, group-writable, owned by another
   uid, own-uid-owned when own uid is not trusted, a symlink, under a writable or untrusted ancestor, or a FIFO each exit 69
   naming the file; a conforming one starts. A test edits the tracked `cpg/ccg/system.yaml` after start and sees the same
   decisions (load once); the manifest is parsed from the verified bytes (a file swapped between check and parse is not
   loaded; a test double hooks the gap).
9. **Credential intake (process).** Missing, a symlink, mode `0640`, another owner, two links, 31 bytes, non-hex, 33 bytes: exit
   69 or 64 as §1; a good file starts the broker. After a request, neither the credential's hex nor its bytes appear in the audit
   files, stderr, `/proc/<pid>/cmdline`, or the environment.
10. **Socket directory check.** A directory owned by another uid than `--holder-uid`, or group/other-writable, closes the allow
    `upstream-error` with no connect (the double's listener counts zero).
11. **Wire parity, pinned by shared literals.** One committed file `llm-gateway/tests/holder_wire_vectors.json`, **generated and committed first by D7b**
    (once, from `llmgateway.framing`, then reviewed; never computed at test time; D9a only consumes it). Schema, one JSON object:
    `{"version": 1, "secret_hex": <64 hex>, "requests": [{"name", "caller", "ts": int, "nonce", "method", "endpoint", "path",
    "run_id", "body_hex", "mac", "frame_hex"}], "replies": [{"name", "request_nonce", "close_reason", "status": int,
    "content_type", "body_hex", "mac", "frame_hex"}], "origins": [{"input", "canonical": <string or null for refused>}]}`; requests
    include a `run_id` containing `|`, the empty run id and a non-empty body; replies include a signed 200, a signed 499 refusal and
    an unsigned `bad_credential` (`mac: ""`); origins include a default-port omission, an uppercase host, an IPv6 literal and a
    refused form. A missing or unparsable file **fails** every consumer, `ccg.manifest.canonical_endpoint`'s test included. Both sides consume it: D9a's broker tests drive `holderclient` against it, and the
    holder's suites (`llm-gateway/tests/test_framing.py`, `test_holder_core.py`, via D7b) drive `llmgateway.framing`,
    `keyholder.canonical_origin` and `ccg.manifest.canonical_endpoint` (a ccg test) against the same pairs, so the three copies of
    the wire and of canonicalisation are each tied to the literals, not to each other. A live round trip of `holderclient` against
    the real `llmgateway.framing.parse_reply` is added. Named mutants: `holderclient` reordering two MAC fields; dropping the
    explicit default port in either canonicaliser; each fails the vectors. D9a's tests import `llmgateway` through the
    `cpg/pyproject.toml` extra `gateway`, which D9a adds and installs (§8); an unimportable `llmgateway` makes these tests **fail**, never skip.
12. **Real holder, through the real broker (integration).** Two additional cases: a hosted entry at the default port (`https://host:443`
    in the manifest, `base_url` spelled without the port) is served, proving the canonical-origin keying; and an absent
    `gatewayRunId` is refused `run_id_missing` under `unmatched = refuse` and served within the shared bucket under `unmatched = cap`. For each code of plan §3.3's table that one ccg-shaped call can
    provoke, assert the 499 refusal reaches `BrokerClient.request` intact, one `decision` and one `close` (`499`, `completed`),
    the stand-in's request log empty, and only a stopped holder gives `closeReason: upstream-error`. A 200 settles and returns the
    provider body. The fake key (`sk-ant-FAKE-…`) appears in none of the broker's files, logs, argv, environment or records
    (positive control: planted copy is found). The class mapping and the no-retry rule are the gateway's tests (plan §5.2 FR-8 (6)), not
    this unit's.
13. **Import isolation.** No module under `cpg/ccg/broker/` imports `llmgateway` (AST scan, plan §3.1).
14. **Existing tests re-pointed.** The §3.4 address tests in `cpg/tests/test_broker_policy.py` that drive a hosted entry
    (`test_the_absolute_name_is_resolved_once_and_the_first_address_is_used`, `test_each_forbidden_form_is_forbidden_address`,
    `test_one_bad_address_among_several_denies`, `test_a_local_entry_address_is_forbidden_for_a_non_local_name`,
    `test_private_address_opt_in_admits_private_but_never_a_host_address`, `test_resolve_failed_and_address_check_unavailable`,
    `test_an_ip_literal_remote_entry_needs_no_resolution_but_is_still_checked`, and
    `test_hosted_model_call_is_r7_when_no_repository_is_local_only`, which asserts `hh.resolver.calls` and `hh.exchange.calls[0]["addr"]` for
    an R7 allow and is rewritten to assert the forward instead) move to an R4 `outbound` entry with a valid
    identity body so forbidden-set coverage is kept; `cpg/tests/test_broker_server.py:test_a_failing_tls_factory_still_ends_the_allow_in_an_upstream_error_close`
    and `test_a_failing_exchange_still_ends_the_allow_in_an_upstream_error_close` are re-pointed likewise, and a new hosted twin
    asserts the §3 item 2 `except` path. `cpg/tests/broker_helpers.py:Harness` gains a `hosted_exchanger` keyword (a recording double returning a configurable `Exchange`,
    passed to `Broker`) and a hosted-entry helper. Every other existing broker test passes unmodified.

**Done:** the suite of `cpg/` is green with `cpg/` as the working directory; items 1-5, 8-12 are red against today's code, and items 6, 7 and 13 are guards that pass today and are proven by their named mutants (item 6 R5/R6 deleted; item 7 and 13 by a mutant that reads an extra request key or imports `llmgateway`); the plan's D9
dependency (D7's holder) is exercised by item 12 only.

## 8. Dispatch: D9 splits into D9a and D9b

Production files: `cpg/ccg/broker/holderclient.py` (new), `cpg/ccg/broker/ownerprot.py` (new), `cpg/ccg/broker/server.py`,
`cpg/ccg/broker/policy.py`, `cpg/ccg/manifest.py`, `cpg/pyproject.toml`: six, and nine items of work, past the ≤5-file, ≤3-step
cap, so two serial dispatches (both edit `server.py`, and D9b's `main` wires D9a's exchanger):

- **D9a (forward path):** `holderclient.py`, `policy.py` (`sanitize`: `gatewayRunId`), `server.py` (`Broker.__init__`, `request`,
  `_upstream`), `cpg/pyproject.toml` (the `gateway` extra, §7 item 11), `broker_helpers.py`; needs D7b's vectors file; tests 1-5, 7, 10-11,
  13, and 14's `test_broker_policy.py` re-points and the `Harness` change.
- **D9b (start-up):** `ownerprot.py`, `manifest.py` (`load_manifest_text`), `server.py` (`load_systems`, `main`: options, static
  hosted check, credential intake); tests 6, 8, 9, 12, and 14's `test_broker_server.py` re-points. Needs D9a and D7/D7b.

D9a adds the `gateway` extra (naming the `llm-gateway` distribution (import name `llmgateway`)) and its implementer then runs, in its own dispatch and not
under `sudo`, `cd cpg && .venv/bin/pip install -e ../llm-gateway` (editable, inside `cpg/.venv`, where `import llmgateway` fails today);
nothing else in the venv changes. D10's `BrokerTransport` imports `llmgateway`, so D10 depends on D9a, and D10's row ("extra") is wrong:
D10 neither edits `pyproject.toml` nor installs the extra.

Plan change needed (`docs/plans/llm-gateway.md` §4): replace the D9 row by D9a and D9b as above; D10 loses `cpg/pyproject.toml`
and its "Needs" gains D9a (it is then parallel with D9b and D8, not with D9a); D9a is parallel with D8 and needs D7b's vectors file;
D11 and D12 need D9b, not D9.
