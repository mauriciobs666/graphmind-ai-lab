# Connected code graph P1-29 (egress broker core), security advisory review

> **Status:** active · **Owner:** `security-expert` · **Tracks:** C-1001 (P1-29)

## Scope & verdict

- **Lens:** code/app security review, advisory. Static reading plus offline unit-level probes run from the session
  scratchpad against in-process `Broker`/`AuditWriter` instances (no network, no FalkorDB, no tree writes, no
  exploitation of a live target; no stray process remains).
- **Reviewed:** uncommitted `cpg/ccg/broker/*` (`addr`, `audit`, `framing`, `policy`, `runs`, `server`, `structured`,
  `client`), `cpg/ccg/schemas/egress-audit-v1.schema.json`, `cpg/tests/broker_helpers.py`, the `broker` extra of
  `cpg/pyproject.toml`. Baseline: `docs/plans/connected-code-graph-egress-spec3.md` §2-5 and the two earlier advisories.
  `cpg/sandbox/*` and the analyst's correctness pass are out of scope. Root `REVIEW.md` applied; `cpg/` has none.
- **Verdict: needs changes** (analyst scale). No blocker: nothing lets a caller cause an outbound connection the policy
  would deny, and every failure I found fails closed. Two majors break spec invariants that the spec states as
  guarantees (caller input never invalidates a record; every allow is followed by a close), both reproduced.
- `CPG: not applicable — no cpg_ccg graph covers this uncommitted new code, and the paths of interest (socket frame to
  audit record, manifest path to request line) are short enough to trace by reading; offline probes confirm them.`

## Findings

### M1 (major, reproduced) · One tokenless frame latches the audit writer for good

Caller strings that pass `sanitize` but cannot be UTF-8 encoded make `AuditWriter.append` latch. Evidence
(`audit.py:164-173`, `policy.py:131-160`, `framing.py:47-60`): `parse_strict` accepts the JSON escape `"\ud800"` and
yields a lone-surrogate `str`; `sanitize.pick("path")` accepts any `str` of at most 256 chars; `append` does
`line.encode("utf-8")` inside `except Exception`, which latches. Probe (`Harness`, no token at all):
`{"op":"request","token":"a","request":{"target":{"path":"\ud800"}},"body":null}` returned `audit-unwritable`,
`latched=True`, and a following legitimate `open_run` returned "the broker's audit writer is latched; restart it".
Same outcome for an allowed hosted call whose body has a top-level key longer than 64 chars
(`payloadFields` items have `maxLength: 64`, schema line for `payloadFields`; legitimate model bodies reach this
path). This contradicts §5.1 ("caller input can never make a record invalid"). Impact: any same-uid process with the
socket (the peer check does not separate processes of one user) denies egress to every component until a manual
restart. Fail-closed, so availability only, but it is a one-frame, no-token kill switch.

Two sibling crashes, same root (surrogates not handled): `sanitize._digest(value)` and `structured.serialize_body`
call `.encode()` and raise `UnicodeEncodeError` for a surrogate in a rejected field (e.g. `component`) or in the
body (both reproduced); `_serve_one` catches only `OSError`, so the handler thread dies with a traceback and the
caller sees EOF.

*Fix:* (1) in `sanitize`, require `path` to match the manifest's own path pattern
`^/[A-Za-z0-9._~!$&'()*+,;=:@%/-]*$` (this also stops verbatim logging of caller junk, see m3); (2) make
`_digest`/`serialize_body` encode with `errors="surrogatepass"` or reject a body containing a lone surrogate with a
`payload-shape`/`R0` deny before any record is built; (3) in `payload_fields` drop or hash keys that exceed 64 chars
or are not printable ASCII, so an allow can never produce an invalid record; (4) add a catch-all in `_serve_one`
that logs the exception type and closes. Add a test per input class named above (each is a planted input that turns
the latch red).

### M2 (major, reproduced) · An allow with no `close` record when the TLS context or exchange raises unexpectedly

`Broker._upstream` (`server.py:198-217`) calls `self.tls_factory()` and `self.exchanger(...)` outside any
`try`. `exchange()` catches the network errors, but `make_tls_context()` runs before it: with a missing or unreadable
`/etc/ssl/certs/ca-certificates.crt`, `ssl.create_default_context(cafile=...)` raises `FileNotFoundError`/`SSLError`.
Probe (hosted allow, `tls_factory` raising): the `decision` (allow) record is durable, no `close` record is ever
written, the handler thread dies, and the caller gets EOF. The audit trail then shows an allow that never closed,
and the §2.5 "every allow is followed by a close" property is lost. Fix: move context creation inside the exchange's
`try` (or wrap `_upstream` and write a `close` with `closeReason: upstream-error`), and add a catch-all that also
writes a close for any other exception.

### m1 (minor) · Sticky-ancestor relaxation (delegate's ruling requested)

`audit._ancestors_ok` accepts a group/world-writable ancestor if it is sticky and owned by root or the user
(`audit.py:49-56`). Spec §2.0 says "no group- or world-writable ancestor" with no exception. My ruling: **the
relaxation is sound for the directory that is a direct child of the sticky directory** (the final directory must be
uid-owned, `0700`, so nobody else can rename or unlink it), **but it is not sound as written for deeper paths**, because
the check looks only at write bits, never at *ownership* of the non-sticky intermediates: `/tmp/other/…/broker`
with `/tmp/other` owned by another uid and mode `0755` passes, and that uid can rename our subtree and put its own
socket at the same path (the client never verifies the peer, see m4). The spec-strict check cannot be tripped this
way only because it rejects `/tmp` entirely. The production default (`~/.local/state/ccg/broker`) never meets a sticky
ancestor, so the exposure is confined to `--socket` (a test option). Recommendation, in order of preference: (a) keep
spec-strict in `server.main` and let tests inject a relaxed predicate; (b) keep the relaxation but add "every
ancestor is owned by root or the uid" and amend §2.0 to say so. §7's backend directory rule (not under `/tmp`) is
stricter still, so (a) is also the consistent choice. Either way the spec and the code must say the same thing; today
they differ.

### m2 (minor) · No resource bounds on the listener (same-uid DoS)

`serve_forever` starts one unbounded thread per connection; the 30 s read timeout is per `recv`, not a wall clock, so
a trickle of one byte every 29 s holds a thread and up to 4 MiB of frame buffer; a parsed 4 MiB JSON can expand to
hundreds of MB. `RunTable` never sweeps expired runs (only a lookup of that token deletes it) and `open_run` has no
cap, each also appending a `policy` record to a file that is never rotated. `AuditWriter._append_raw` takes a
**blocking** `flock` while holding the writer lock, so any same-uid process holding `LOCK_EX` on `egress.jsonl` hangs
every broker call. All are fail-stop DoS by a process of the same user, i.e. not a boundary the design claims.
Fix at low cost: cap concurrent connections (a semaphore, close extras), a wall-clock deadline for frame read,
`LOCK_NB` with a bounded retry that latches, a run-count cap, and a sweep of expired runs in `create`.

### m3 (minor) · Caller text lands verbatim in the audit file and stderr

(a) A denied request's `target.path` (any string up to 256) is written verbatim; a caller can place a secret or
a URL query there. (b) U+2028/U+0085 are written raw (`ensure_ascii=False`): probe showed `str.splitlines()` splits
one record into 3 lines while `\n` splitting sees 1. `read_records` splits on `b"\n"`, so it is safe, but other
readers (`jq -c` is fine, `splitlines()` consumers and some pagers are not). (c) `FrameError` messages repr a
duplicate key and `unknown op {op!r}` repr the whole `op`, both attacker-sized (up to 4 MiB) into stderr.
Fix: the path allowlist from M1 closes (a); consider `ensure_ascii=True` for the audit line (spec text says UTF-8
JSON, escaping is still valid JSON) or document it; truncate reprs in log lines to 80 chars.

### m4 (minor, theoretical) · Client never authenticates the broker

`BrokerClient._call` connects to `CCG_BROKER_SOCKET` (or the default) and sends the token and body with no
`SO_PEERCRED` check on the server side of the pair and no owner check on the socket path. Anyone who controls that
environment variable already runs as the user, so no privilege is crossed, but a same-uid sandboxed component with an
env override could point a client at a look-alike that answers "allow". Cheap hardening: in `_call`, read
`SO_PEERCRED` after `connect` and require the uid to equal `os.getuid()`.

### m5 (minor) · Forbidden-address set gaps

Probed `_forbidden`: not forbidden are `fec0::/10` (deprecated site-local, still routed on some stacks) and `100::/64`
(discard). With `privateAddress: true` the cloud metadata addresses `fd00:ec2::254` (AWS IPv6 IMDS) and
`100.100.100.200` (Alibaba) are accepted, because they fall in the private ranges; `169.254.169.254` is correctly
always forbidden. Add `fec0::/10`, `100::/64`, `fd00:ec2::254/128`, `100.100.100.200/32` to the always-forbidden
lists (spec §3.4 lists the set and would need the same addition). Documentation prefixes (`192.0.2.0/24`,
`2001:db8::/32`) are unroutable and can stay.

### n1 (nit) · Smaller items

- `_bind` treats **any** `OSError` from the probe `connect` as "nothing answered" and unlinks (`server.py:243-247`); the
  spec says a live socket is never unlinked. The `flock` already prevents a live peer from the same directory, so this
  is defence in depth only; unlink only on `ECONNREFUSED`/`ENOENT`, and give the probe a short timeout.
- `exchange` never sends `Connection: close`, so a keep-alive upstream that answers without a length would be read until
  the 300 s deadline; `rest += chunk` in `_read_response` is quadratic under a 1-byte-chunk drip (bounded by the deadline
  and by the allow-listed upstream).
- `RunTable` uses `time.monotonic`, which does not advance across WSL suspend, so a run can outlive its 24 h wall time.
- `reexec_if_needed` with an empty `sys.executable` raises a traceback instead of exit 64/69.

## Items checked and found solid

- **Re-exec guard.** Confirmed live: with `PYTHONPATH` pointing to a planted dir holding `argparse.py` (would exit 99)
  and `sitecustomize.py`, `python -m ccg.broker.server` printed the sitecustomize marker once and ran the real argparse.
  So the closed re-exec works: the planted `argparse` never loaded, and the second interpreter (`-I`, closed env, `HOME`
  from passwd) did not run `sitecustomize` again. Residual, by design and cross-uid-irrelevant: whatever the launching
  interpreter loads (`sitecustomize`, `.pth`, the `ccg` and `ccg.broker` package `__init__`, which runpy imports first)
  runs *before* the guard, and `-I` in the second process needs `ccg` installed (it is, editable, in `cpg/.venv`). Only
  stdlib `os`, `pwd`, `sys` are imported before the guard.
- **Peer uid / socket / lock.** Socket bound under `umask(0o177)` in a `0700`-checked directory; `SO_PEERCRED` uid check
  before any read; lock `O_NOFOLLOW|O_CLOEXEC`, `flock` non-blocking; stale-socket handling per spec (see n1).
- **Audit file.** `O_NOFOLLOW`, `O_NONBLOCK` against a planted FIFO, `fstat` regular/owner/mode/`nlink == 1`, one `write`
  plus `fsync` under `flock`, schema-validated on both sides, latch on failure. The audit directory takes no ancestor
  check, which matches §5.1 (my earlier F-7 suggestion was not adopted in the spec; unchanged by this step).
- **Tokens.** 32 bytes from `secrets`, only the SHA-256 held, strict 64-hex gate before hashing, 24 h expiry enforced on
  lookup, token never in a record or log line.
- **Request-line and header smuggling.** The upstream path is the manifest's own string (`next(p for p in entry["paths"]
  if p == clean["path"])`), and the manifest schema (`egress-v1.schema.json` `paths` pattern, plus S5's control-character
  sweep, which also covers Python's `$`-before-trailing-newline quirk) excludes CR, LF, space and `?`. Host header from
  the canonical endpoint; headers fixed; the only caller bytes upstream are the JSON body with a matching
  `Content-Length`. One request per connection, so response-side CL/TE ambiguity cannot desynchronise anything.
- **Response bounds.** Bytes counted at the socket with read aborted at bound+1; a wall-clock `monotonic` deadline
  re-applied before connect, handshake, send and every recv; `Content-Encoding` other than identity refused; chunk
  size capped to 8 hex digits; 1xx refused; no redirect following.
- **TLS.** `create_default_context(cafile=...)` skips default verify paths (so `SSL_CERT_*` has no effect), keeps
  `CERT_REQUIRED` and hostname checking, SNI and verification against the entry host; connects to the validated address,
  never re-resolves.
- **Address policy.** Denied requests cause no resolution; absolute-name lookup; every returned address checked;
  IP literals skip the resolver but not the check; IPv4-mapped, NAT64, 6to4, Teredo, link-local and zone ids forbidden;
  host, bridge-network and gateway addresses always forbidden; `local` entries re-checked per request (S6).
- **Frames.** Length checked before payload; 4 MiB + 64 KiB request cap; duplicate keys, NaN/Infinity, int bounds, nesting
  depth, over-long int literals (Python's digit limit) all map to `FrameError`; `RecursionError` caught.
- **Rule table.** I traced the rule table for a path to an outbound connection the policy would deny and found none:
  an allow requires a manifest entry matched by exact canonical endpoint and exact path, so neither the caller's
  `endpoint` nor `path` reaches the socket layer unvalidated.

## Open questions

1. (`architect`/`devops`) Spec amendment for m1: tighten the code to §2.0, or amend §2.0 to the narrower relaxation
   named above? Tests need a non-`/tmp` base or an injected predicate either way.
2. (`architect`) Should `privateAddress` ever cover cloud-metadata addresses (m5)? I recommend no.
3. (`architect`) Is `ensure_ascii=False` in the audit line a requirement, or may the writer escape (m3b)?

## Proposed REVIEW.md lines

### Root `REVIEW.md`, "Always check"
- Add: for any component that writes caller-influenced strings into an audit/log record, enumerate each caller-supplied
  field and run a probe with a lone surrogate, an over-length value and a control character against the *record
  writer*, not only the sanitiser (source: M1, which every existing `sanitize` test missed).

## Pass 2

Re-run against the current tree with the original probes (offline, scratchpad, in-process; no live target). Verdict:
**approve with suggestions**. Remaining items are minor/nit and deferred.

- **M1: fixed.** Lone surrogate in `path`, `component`, op, body value or body key: the wire parser now raises `FrameError`
  (all four frames, `framing._walk_ok`); the same inputs injected past the parser are handled (`path`, `component`) or
  raise `FrameError` (body), writer never latched. Over-length `payloadFields` key (100 chars): allow, no latch (clipped
  at `MAX_FIELD_CHARS = 64`, digest in `rejected["payloadFields"]`). Control/U+2028 key: no latch.
- **M2: fixed.** `tls_factory` raising `FileNotFoundError`: `closeReason: upstream-error`, records
  `policy, decision, close`, writer not latched.
- **m1: fixed.** `main` calls `require_dir(..., check_ancestors=True)` with no `allow_sticky`; `/tmp` is refused
  (strict probe: "ancestor /tmp is group- or world-writable"). `allow_sticky` is reachable only by a Python caller of
  `audit.require_dir`/`_ancestors_ok`: grep finds it nowhere in `server.py`, no CLI flag, no env var. It still has no
  intermediate-ownership check (`/tmp/other/x/broker`, `other` uid 4242 mode 0755, passes under `allow_sticky=True`), so
  keep it test-only; the audit test's "owner conjunct" covers the sticky dir itself, not intermediates.
- **m3: fixed.** `ensure_ascii=True` (record line is pure ASCII; `splitlines()` count equals newline count after a U+2028
  key) answers open question 3: agreed, it is valid JSON and removes the splitter class. Verbatim path (a): now matched
  to the manifest pattern in `sanitize`. (c) reprs: `framing.clip` is used for the duplicate key and `op`.
- **`surrogatepass` rejection: agree.** `surrogatepass` would send an invalid-UTF-8 body (`ED A0 80`) upstream and sign
  a digest over bytes no receiver can decode; rejecting at the parser, with a `FrameError` guard in `request()` for
  in-process callers, is the safer design. `_digest` using `surrogatepass` is fine: it only hashes a rejected value.
- **Deferred, my call.** None needs promoting to this step: m2 (listener bounds), m4, m5 (extra prefixes, including the
  two cloud-metadata addresses that `privateAddress: true` still accepts), n1 are all same-uid DoS, theoretical or
  defence in depth, none moves an allow/deny decision. Two conditions: keep m5 in the ledger tied to the spec §3.4
  prefix-list question so it is not lost, and do not enable `privateAddress` on a cloud-hosted machine before it lands.
- Stray processes: none of mine; two `bash` processes matching `ccg.broker` belong to another agent's pytest run.
