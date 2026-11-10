# Connected code graph — egress spec 3: security advisory review

> **Status:** active · **Owner:** `security-expert` · **Tracks:** C-1001 (P1)

## Scope & verdict

- **Lens:** code/app security review of a design document (static; no exploitation, no network
  experiments, no FalkorDB writes), plus the secrets/infra-hardening lens for the socket, lock and
  audit files. Step P1-28, index `docs/plans/connected-code-graph-p1.md` §9 row N-ADV. Advisory:
  `architect` owns the spec and `devops` owns P1-29/P1-31. Neither is bound by this review.
- **Reviewed:** `docs/plans/connected-code-graph-egress-spec3.md` (uncommitted working copy,
  2026-11-09), concentrating on the targets the architect named: §2.0, §2.3, §2.6 and §3.4 (the
  structured-call trust boundary), I4, I6, §2.5 and §5.1 (the single audit writer), and R-E1 and
  R-E3. **Baseline:** index §6 (option C), `docs/plans/connected-code-graph-p1-gates.md` P1-29 and
  P1-31, and the built runner `cpg/sandbox/offline-run.sh` (its record writer, lines 517–584).
- **CPG:** not applicable. The artifact is a design spec, and the broker it specifies
  (`cpg/ccg/broker/`) does not exist yet, so there is no data flow to trace. The only code read is
  the runner's audit writer, by hand.
- **Growth-stops:** not applicable. This is a new successor document, not a revision. The §0
  "Material differences from ES1 + ES2" paragraph is shaped like a change log. That is `analyst`'s
  call, not a security one.
- **Verdict: needs changes.** There are no blockers. Nothing here is exploitable in P1 as
  configured: the repository's manifest has no `outbound` or `hosted` entry, and no P1 caller uses
  `R3` or `R4`. But six majors make a security property the spec claims false, or leave it with no
  mechanism: "nothing beyond names and identity, by construction", the forbidden address set,
  "local = the machine", the closed input set, the singleton, and P-a. Each one has a small,
  specific fix.

**Confidence labels used below:** *verified* means run on this host or read in code; *traced*
means followed end to end through the spec's own rules; *inferred* means reasoned from
documentation or memory and not probed.

## Findings

### Major

**F-1 · `R4`'s "closed shape" bounds the keys, not the content, so local-only data can leave
through `repoNames`.** *Traced.*
§2.3 says the body is `{repoNames: [str], identity: {subject: str, issuer?: str}}`, so that "AC-8's
'nothing beyond repository names and identity' holds by construction". `R4` sits before `R5`/`R6`
(§3.3), so it is the one outbound allow that **local-only** codebases keep. No rule checks what
the strings contain. Take `repoNames: ["<base64 of a source file>"]` or a 3 MB `identity.subject`
in an `answer` run over a local-only codebase: it passes R0 (size ≤ 4 MiB), matches `R4` and is
allowed. `payloadFields` records only key names (§5.2), so row 8.3 passes as well. A related gap:
R0 has no component↔purpose check. `component = answer.interpret, purpose = idp` passes R0 in an
`answer` run (§2.4a admits `answer.*`), so any answer-path code can reach `R4`, not only `access.*`.
*Why it matters:* this is the one sanctioned path out of local-only, and its content guarantee is
only declared. A client bug or a prompt-injected summary that lands in a field would leave the
machine while the audit reports compliance.
*Fix:* (a) every `repoNames` element must be a `repoId` of the run's system (better: of the run's
scope set). That makes it a closed value, not a free string. (b) bound `identity.subject` (for
example ≤ 256 characters, `^[\x21-\x7e]+$`) and require `identity.issuer`, when present, to be a
canonical https URL of ≤ 256 characters. (c) add an R0 check `component-purpose-mismatch` from a
fixed table (`access.idp`→`idp`, `access.check`→`access-check`, `answer.pr-adapter`→`pr-addon`,
`answer.interpret|summarize`→`model-call`, `vulndb.refresh`→`vulndb-refresh`). It closes this and
the vulndb-run model-call path in F-14. (d) record `bytes` per field, or a digest of the body. The
P1-29 unit test plants a non-repoId `repoNames` element and an oversized subject, and expects
`payload-shape` for both.

**F-2 · The forbidden address set, implemented the obvious way, lets IPv4-mapped IPv6 through.**
*Verified on this host's Python 3.12.3.*
§3.4 lists the classes: loopback, link-local, unspecified, multicast, RFC 1918 and so on. With
Python's `ipaddress`, `IPv6Address('::ffff:127.0.0.1').is_loopback` is **False**, and so is
`::ffff:169.254.169.254`'s `is_link_local`. `::127.0.0.1` (IPv4-compatible) and `64:ff9b::7f00:1`
(NAT64) even report `is_global = True`. An `AAAA` answer of `::ffff:127.0.0.1` reaches loopback
on Linux through a dual-stack socket. So does a schema-legal literal `https://[::ffff:127.0.0.1]:443`
(the `remoteEndpoint` IPv6 alternative is `\[[0-9a-f:.]+\]`). For `privateAddress: true` entries,
the RFC 1918 check is skipped by design, which leaves only the class checks above as the guard. A
check against explicit IPv4 networks also misses mapped forms of 10/8 and the other private
ranges.
*Fix:* make the spec normative here. Before classifying, map `ipv4_mapped`, IPv4-compatible
(`::/96`), NAT64 (`64:ff9b::/96`, `64:ff9b:1::/48`) and 6to4 (`2002::/16`, its embedded address)
to the embedded IPv4 address and check **both** addresses. Alternatively, deny those prefixes
outright. Classify against an explicit network table that the spec lists, not against
`is_*` predicates, whose meaning changes between Python versions. Add to the table `0.0.0.0/8` (on
Linux, a connect to `0.x` can reach the host), `240.0.0.0/4`, `192.0.0.0/24`, `198.18.0.0/15` and
`100.64.0.0/10`, which is already listed. Test (12) gains one planted address per form, each
expected to give `forbidden-address`.

**F-3 · Nothing makes a `local` entry local.** *Traced.*
I3 and `R1` rest on "local endpoint = the machine" (FR-9). `R1` allows **any** payload, including
`repo-derived`, from a local-only codebase, over cleartext `http`. Yet `localEndpoint` admits any
IPv4 literal, and S4 bans only unspecified and multicast. So `local: [{endpoint:
http://203.0.113.7:80}]` is a valid manifest. With it, every local-only codebase's source goes to
a public address in clear text, and every AC-8 row stays green (8.1, 8.8 and 8.9 accept
`rule = R1`).
*Fix:* add semantic check **S5**: a `local` IP literal must be loopback (`127.0.0.0/8`, `::1`), an
address of a host interface, or the default gateway of a host interface. The last case is the WSL
NAT route to the Windows host that §2.0 itself names. All three come from the same `ip -j -d
address show` / `ip -j route` read that §3.4 already does. Anything else is a startup refusal,
exit 64. Under mirrored networking, the Windows host's addresses are host-interface addresses, so
LM Studio keeps working.

**F-4 · The broker's "closed inputs" are not closed: the interpreter, OpenSSL and the libc
resolver read the environment.** *Partly verified.*
§2.0 says the broker reads only `CCG_AUDIT_DIR` and that `http.client` "reads none". On this host,
`ssl.get_default_verify_paths()` consults `SSL_CERT_FILE`/`SSL_CERT_DIR` **even under
`python -I`** (verified). Two more are *inferred* from documented behaviour: glibc's resolver
honours `LOCALDOMAIN`, `RES_OPTIONS` (including `ndots`) and `HOSTALIASES`, and `python -m`
without `-I` honours `PYTHONPATH` and the user site, and puts the working directory first on
`sys.path` (a planted `json.py` or `ssl.py` there). Through these variables a launcher's
environment can change §3.4's resolution and the https trust anchor. These are the two controls
the trust boundary depends on. The runner solved the same problem (§7.7: re-exec under `env -i`,
`#!/bin/bash -p`); the broker spec does not.
*Fix:* the broker re-executes itself under `env -i` with a fixed `PATH`, `HOME` (from passwd),
`LANG=C.UTF-8` and `CCG_AUDIT_DIR`, as `python -I -m ccg.broker.server`, unless its environment is
already exactly that. Its TLS context loads a fixed CA bundle path rather than the default verify
paths. P1-29 adds a test that starts the broker with `SSL_CERT_FILE`, `LOCALDOMAIN`, `RES_OPTIONS`,
`HOSTALIASES` and `PYTHONPATH` planted, and asserts that none takes effect (an injected resolver
and an injected `ssl` context record what they received).

**F-5 · The singleton is per audit directory, but the socket is per home, so a second broker can
take over the socket.** *Traced.*
The lock is `<auditDir>/broker.lock` (§2.0), but the socket path comes from the passwd home. The
default `auditDir` is `cpg/.audit/`, which is per checkout, and §5.1 does not say what it is
relative to. A broker started from **another worktree** gets its own audit directory, so its own
lock, so the lock check passes. It also holds "the singleton lock" under which a leftover socket
may be unlinked. It then unlinks the first broker's live socket and binds its own. This team
routinely runs agents in git worktrees (`isolation: worktree`). From then on every client reaches
the second broker, with that checkout's manifest, and its records go to that checkout's
`egress.jsonl`. An AC-8 or operator reader of the first directory sees fewer decisions, and rows
quantified "every allow …" pass vacuously over the missing ones.
*Fix:* take the singleton lock in the **socket directory** (`<home>/.local/state/ccg/broker/broker.lock`,
under the same 0700 checks), so lock and socket share one key. Before unlinking a leftover socket,
also attempt a `connect()` and refuse (exit 69) if a live broker answers. Anchor the default
`auditDir` to the package location, as the runner does (`$SBX_DIR/../.audit`, `offline-run.sh:580`),
and record its absolute path in each `policy` record so a reader can see which directory a run
used. The same applies to `--socket`.

**F-6 · P-a's check is narrower than P-a's claim (R-E3).** *Inferred, not probed.*
P-a claims "the only endpoints that accept are the store's port and the detector's". The check
connects only to **the IPv4 gateway**, only on `ss -ltn`'s TCP port snapshot, and to `local` ports.
Three gaps remain. UDP listeners are not checked (`ss -lun`). The host side of the bridge may also
have an IPv6 link-local address (`fe80::…`), which reaches dual-stack `::` listeners even with no
IPv4 gateway. And a listener that starts after the snapshot is never probed. Under REVIEW.md's
check rule this is a bound, not proof. On the mechanism: Docker's bridge driver documents
`com.docker.network.bridge.inhibit_ipv4` ("prevents Docker from assigning an IP address to the
bridge"). That is a third candidate the spec does not list, and it needs no host firewall change.
What it does to the bridge's IPv6 link-local address, and to the engine's own 29.7.2 internal-mode
rules, is unprobed. A fourth option removes R-E3 entirely. Run the answer components
`--network none`, and give them the broker's socket and a **Unix-socket** store endpoint (FalkorDB
inherits Redis's `unixsocket`) as their only channels. P-a then reduces to A1 inside the
container. The cost: the detector stops seeing attempted connections, because they fail locally.
*Fix:* (a) state P-a's check as a probe set: every host address, IPv4 **and** IPv6 (link-local
included, from the same interface read as §3.4), × every TCP and UDP listener port from `ss -ltun`
taken **after** scenario start, plus no `::/0` route, all probed from each component container,
with a positive control (one deliberately reachable listener, so the probe can turn red). (b) add
`inhibit_ipv4` (with `--ipv6=false`, or an explicit check of the bridge's IPv6 state) to the
candidate list. (c) require component containers to run `--cap-drop ALL`, so they cannot add a
route or an address. I agree with R-E3's "escalate rather than relax".

### Minor

**F-7 · Audit-file and directory hardening is not specified, though the socket directory's is.**
*Traced; runner part verified in code.*
`auditDir` has no owner, mode or symlink check, although it holds `broker.lock` and both audit
files. `egress.jsonl` is created `0600` only if absent: an existing file is used as found (a
symlink, mode `0644`, another owner, `nlink > 1`). The runner creates `sandbox.jsonl` with
`O_CREAT, 0o644`, follows symlinks, and runs `mkdir -p` with default mode (`offline-run.sh:519-524`).
*Fix:* open with `O_WRONLY|O_APPEND|O_CREAT|O_NOFOLLOW|O_CLOEXEC`, then `fstat` and require a
regular file, owner = uid, `mode & 0o077 == 0`, `nlink == 1`, or latch. Give `auditDir` the same
plain-directory/owner/`0700` check as the socket directory, plus "no group- or world-writable
ancestor", as §7.7 already requires for the Docker config. The same applies to `broker.lock`,
whose path moves under F-5. Ancestor checks belong on the socket directory as well: §2.0 checks
only `broker/` itself.

**F-8 · "Append-only" is a convention, and the records are the only witness (R-E1).** *Traced.*
I6 and §5.1 say who should write, but `flock` is advisory and nothing detects an edit or deletion.
That matters more than usual here, because R-E1 makes these records the sole attestation of the
broker's upstream. *Fix:* hash-chain each file, so each record carries `prev` = SHA-256 of the
previous line, and anchor the chain at the AC-8 flush marker (P-d), which records the last hash.
Readers (§5.4, §8) verify the chain. This detects tampering; it does not prevent it, which suits a
same-uid threat model the spec scopes out.

**F-9 · A crash can glue a partial line onto the next record, mid-file.** *Traced.*
"A reader skips a trailing partial line" covers only the tail. After a crash, the next broker's
first `O_APPEND` write continues the partial line. The result is one invalid line in the middle of
the file, and row 8.14 fails it, or a reader drops a good record. *Fix:* at start, under the lock,
if the file does not end in `\n`, write a `\n` (and count it on stderr). Treat a short `write(2)`
as unwritable (latch). For a newly created file, `fsync` the directory as well.

**F-10 · The detector's mount and the answer containers' socket mount over-grant.** *Traced.*
The detector writes `<auditDir>/conn.jsonl` (§8.1), so as written it gets `auditDir`, including
`egress.jsonl` and `sandbox.jsonl`, writable from a network-facing container. The socket
**directory** is bind-mounted into several component containers. Under the same uid, any of them
can unlink `broker.sock` and bind its own, and so intercept the tokens and requests of the
components that connect next. *Fix:* mount the detector on a dedicated `<auditDir>/detector/`
only. Bind-mount the socket directory `:ro`: `connect()` to a Unix socket on a read-only mount
still works, and unlink/bind there does not (*inferred*, the same reason a `:ro` `docker.sock`
gives no protection). State that component containers run `--user <broker uid>:<gid>`, without
which `SO_PEERCRED` rejects them, and with `--cap-drop ALL`.

**F-11 · The local-socket framing and JSON parsing are unspecified.** *Traced.*
§2.2 defines the call but not the frame. A frame-size cap applied before parsing, the parser's
duplicate-key behaviour, nesting depth, `NaN`/`Infinity` (Python's `json` accepts and emits them
by default, producing invalid JSON upstream) and very large integers are all left to the
implementer. *Fix:* name the frame (for example a 4-byte length prefix, max `MAX_REQUEST_BYTES +
64 KiB`, checked before reading), and set `object_pairs_hook` to reject duplicate keys,
`allow_nan=False` on both sides, and a depth limit. A malformed frame closes the connection
without a decision. That is not an egress request, so it is counted on stderr like the peer check.

**F-12 · The response bounds can be bypassed by the HTTP client's own behaviour.** *Inferred.*
`httpx` (allowed by §2.0) decodes `Content-Encoding: gzip` even when not asked to, so a 16 MiB
bound applied after decoding admits a decompression bomb. `http.client`'s timeout is per socket
operation, so a server that sends one byte every 299 s keeps "the whole exchange" alive
indefinitely. *Fix:* send `Accept-Encoding: identity`, refuse any non-identity encoding, count
`MAX_RESPONSE_BYTES` on the wire and abort at bound + 1, and enforce `UPSTREAM_TIMEOUT_S` as a
wall-clock deadline. Test (17) gets one case each.

**F-13 · `.invalid` names are not guaranteed to be unresolvable through the system resolver (P-e).**
*Inferred.* The schema forbids a trailing dot, so `getaddrinfo("idp.x.invalid")` is subject to
`/etc/hosts`, the `search` list (after a failed absolute lookup), `HOSTALIASES` (F-4), and on WSL
to the Windows resolver behind DNS tunnelling (this host's resolver is `10.255.255.254`). A
wildcard search domain turns "never resolves" into "resolves". As stated, P-e also has no check
that can fail. *Fix:* the broker resolves `host + "."`, which disables search. P-e gains a
pre-run assertion that the broker's own resolve function returns `resolve-failed` for every
non-local host, plus a positive control: one name the harness deliberately maps in a test hosts
file, which the assertion must flag.

**F-14 · Some AC-8 checks cannot fail, because the runner writes their values as constants.**
*Verified in code.* `sbx_write_record` hard-codes `"network":"none"`, `"networkName":null`,
`"runId":null` and `"liftedHits":[]` (`offline-run.sh:564, 567, 571`). Row 8.5's "`network =
none`" and row 8.4's "`runId = null`, so no `vulndb` run reaches a sandbox" are therefore true of
every record the runner can write, and are no evidence of either claim (REVIEW.md, a check that
cannot fail). On the same row: §2.4a admits `vulndb.refresh` in a `vulndb` run with any `purpose`
and `payloadClass`, so (without F-1(c)) a `vulndb` run can make a `model-call` with a
`repo-derived` body, allowed by `R7` in an all-hosted system. And `vuln-feed-request` bodies are
opaque, so row 8.4's "never sees repository content" is also only declared. *Fix:* base 8.5 on
`launcherArgv` (`--network none`) and `preconditions.netnsDistinct`, which 8.6 already partly
does. Report 8.4's sandbox clause as vacuous. In P1, restrict `R3` to `GET`/`payloadClass = none`
until P5 defines the feed shape.

**F-15 · Forbidden-set coverage under `privateAddress: true`, and staleness.** *Inferred.*
In WSL NAT mode the Windows host is the eth0 **gateway**, which is not a host-interface address
and not a bridge network. So an entry with `privateAddress: true` that resolves to it is
admitted. A Docker network created within the 60 s re-read window is not yet in the set. *Fix:*
add each host interface's default gateway to the forbidden set (unless it is a `local` entry's
literal). Re-read the interface list on every validation: the read is cheap and validations are
rare, since there is one per non-local allow.

**F-16 · The `sandbox` record is not tied to what it produced (§5.4).** *Traced.*
`check_sandbox_ids` proves that a successful, offline record with that id exists. It does not
prove that the bundle's inputs came from it: the record carries no output digest. Any successful
id can be cited by a bundle built elsewhere. *Fix (P1 changes no runner behaviour, so this is an
open question for `architect`):* record `outputs: [{path, sha256}]` in the `sandbox` record, and
have §5.4 compare it with the bundle header's input digests. Until then, say in §5.4 that the
check is presence-only.

**F-17 · R-E1 has a cheap independent observer in P1.** *Inferred.*
The broker's own upstream is unobserved. In the scenario, the harness can launch the broker under
`strace -f -e trace=connect,sendto,sendmsg -o <auditDir>/broker-syscalls`. That is not root, and
it is the harness's own child, so Yama `ptrace_scope=1` allows it. Add a row: every non-`AF_UNIX`
`connect` equals some allow's `resolvedAddr:port`, with an injected rogue dial as the positive
control. DNS to the stub resolver, or the `io.systemd.Resolve` varlink socket, appears only after
an allow. It does not change the production shape, and it closes most of R-E1 for P1-40.

### Nit

- **N-1.** §2.0 "creates the socket `0600`": `bind()` takes its mode from the umask, so set
  `umask(0o177)` around `bind`, not a later `chmod`. The 0700 directory makes this moot in
  practice.
- **N-2.** `open-run-invalid` writes no record (§2.1). That is consistent with "no run", but probes
  of unknown systems or repositories leave no trace. Count them on stderr, as the peer check does.
- **N-3.** A caller value recorded as `null` (§5.1) loses the forensic trace. Add a bounded
  `rejectedDigest` (SHA-256 of the raw value) next to the null. Bound `codebases`' length in R0 so a
  valid-pattern list cannot make a multi-megabyte record line.
- **N-4.** Header injection is closed twice: the `paths` pattern excludes CR, LF and space, and
  `http.client` refuses control characters in the URL and header values. Put the stated
  equality test on the **entry's** stored string (send that string, not the caller's) so that no
  later normalization step can come between the check and the write.

## What's solid

- **A Unix socket, with no TCP listener**, rejected for the right reason (mirrored `localhost` and
  gateway exposure), plus a `0700` directory as the control and `SO_PEERCRED` as defence in depth.
- **Token handling:** 256-bit CSPRNG, stored only as a SHA-256 (no timing oracle on lookup), never
  in a record, log or argv, and kept off the sandbox by the closed pass-list. Test (18) can fail.
- **Deny before DNS, resolve once, connect to the validated address, verify against the entry
  host:** no rebinding window, and a denied request leaks nothing to the resolver. Test (12)
  injects the resolver.
- **The broker builds the whole request:** fixed headers, exact-equality path matching (no
  normalization surface, so encoding tricks have nothing to bypass), no redirects.
- **Record before acting, plus the latch:** no upstream bytes without a durable decision, and a
  `decision` allow with no `close` fails AC-8.
- `R2`/`R8`/`R9` deny before `R1`, with their own reasons; closed enums and deny vocabulary;
  `allowListDigest` binds every run to its policy.
- §2.6 "What this boundary does not cover" and R-E1/R-E2 are stated honestly rather than hidden.

## Open questions

1. **`architect`:** F-16. Should the `sandbox` record gain output digests now (a runner change P1
   says it will not make) or at the next runner revision, with §5.4 saying "presence-only" until
   then?
2. **`devops` (P1-31):** F-6. Choose between `inhibit_ipv4`, a `DOCKER-USER` rule and
   `--network none` with a Unix-socket store, after probing each. The last gives up the detector's
   view of attempted connections.
3. **`architect`:** F-1(a). Should `repoNames` be limited to the run's scope set or the whole
   system's repositories? An access check for a question may legitimately name repositories
   outside the touched set.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`. Source: F-2 (verified on Python 3.12.3).

- A check that classifies an IP address (forbidden set, SSRF guard, "is local") maps IPv4-mapped,
  IPv4-compatible, NAT64 and 6to4 forms to the embedded IPv4 address before classifying, and its
  test plants `::ffff:127.0.0.1`. Python's `ipaddress` `is_*` predicates do not do this.
