# Connected code graph — egress broker and offline-build sandbox spec (P1, option C)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1) · **Version:** 1.4 · **Supersedes:** `docs/plans/connected-code-graph-egress-spec.md`, `docs/plans/connected-code-graph-egress-spec2.md` · **Extended by:** `docs/plans/connected-code-graph-egress-spec4.md`, `docs/plans/connected-code-graph-bundle-spec6.md` · **Reviews:** `docs/reviews/connected-code-graph-egress-spec3.md`, `docs/reviews/connected-code-graph-egress-spec3-security.md`

*2026-11-10 (v1.4): §7.4's closed frontend set gains the two indexer entries.*

## 0. Scope, sources and how to read this

**What this is.** The current egress and sandbox contract of the connected code graph, written for
**option C** (`docs/plans/connected-code-graph-p1.md` §6: no in-system git fetch in P1; the
git-fetch path is deferred with the trigger "a manifest first names an https `remote`"; AC-8's proxy
is a detector). It is step P1-28 of `docs/plans/connected-code-graph-p1-gates.md`. It replaces
`docs/plans/connected-code-graph-egress-spec.md` v1.3 ("ES1") and
`docs/plans/connected-code-graph-egress-spec2.md` v1.2 ("ES2") as the one normative text. Those two
stay the record of **evidence and rationale** (host probes, gate dispositions, rejected
alternatives); this document cites them for *why*, never for *what*.

**Who builds against it.**

| Section | Step and owner |
|---|---|
| §2, §3, §5.1–§5.2 (broker, policy, broker records) | P1-29 broker core, `devops` |
| §4 (allow-list, `egress-v1.schema.json`, per-repository mode) | P1-03 manifest loader, `tdd-engineer` (creates the schema file); P1-29 consumes it |
| §5.3–§5.4 (the `sandbox` record schema and the `sandboxAuditIds` check) | P1-18 assembler, `graph-dba`; P1-19 calls the same check |
| §1 I2 (the import boundary) | P1-25's `cpg/tests/test_no_network.py`, extended (§10.1) |
| §7 (sandbox) | built and gated in P0 (S08, `cpg/sandbox/offline-run.sh`); **P1 adds only the two indexer frontends of §7.4**, built in P1-48b (`devops`) |
| §8 (AC-8 P1 scenario) | P1-31 harness, P1-40 run, `devops` |
| §2.1, §2.5 (who opens runs; never fall back) | P1-20 refresh, P1-26 front |

**What this does not decide.** The rest of the system manifest (P1-03). The bundle header's fields
(`docs/plans/connected-code-graph-bundle-spec3.md` §2.2). The proxied-stream design (§6 lists its
inputs; it is designed when its trigger fires).

CPG: considered, not relevant — no graph is loaded for `cpg/` (and the brief forbids leaning on one); the spec rests on ES1, ES2, the P1 plan and the built runner `cpg/sandbox/offline-run.sh`, read directly.

## 1. Terms and invariants

| Term | Meaning |
|---|---|
| **Egress** | Any connection that leaves the process's own network namespace or reaches another process through a host channel: TCP/UDP to an address other than the process's own loopback, DNS, a host Unix socket, `AF_VSOCK`, WSL interop. The store connection of I2 is not egress. |
| **Mode** | A codebase's egress mode for one run: `local-only` or `hosted` (FR-9). **Default:** the mode in its manifest entry (§4.4). **Override:** a per-run request, `local` or `hosted`. |
| **Local endpoint** | An entry of the allow-list's `local` section: "the machine" (FR-9), checked to be on the machine (§4.3, S6). |
| **Broker** | The one host process per user that makes egress on behalf of components (§2.0). |
| **Run** | One refresh, answer or vulnerability-database job, opened with `open_run` and identified by `runId`. |
| **Structured call** | A request the broker serializes and sends itself (§2.3). The only transport in P1. |
| **Sandbox** | The runner `cpg/sandbox/offline-run.sh` that executes Joern frontends and extractors with no route out (§7). |
| **Escape axis** | One channel by which a sandboxed process could reach beyond the sandbox (§7.3). |

**Invariants.** Every section below serves one of these.
- **I1. Default deny.** A request no rule allows is denied and recorded.
- **I2. One route.** In P1 the broker is the only component of ours that opens a connection for
  egress. The one other permitted connection is the store client's to the `ccg` FalkorDB instance on
  the machine (its loopback-published port; in the AC-8 scenario, the store container, §8.1), which
  is not egress. Sandboxed tools have no network at all: P1 has `offline` mode only.
- **I3. Local-only is sticky.** Nothing derived from a local-only codebase reaches an endpoint outside
  `local`, except through `R3` and `R4` with their content bounds (§2.3, §3.3), whatever the run asks.
- **I4. Recorded before acted on.** Every mode resolution, every egress decision and every sandbox
  invocation, refusals included, writes exactly one record (§5). A decision is durable on disk before
  any upstream connection is attempted. The one act that precedes its record is the DNS resolution of
  an **allowed** non-local entry's host (§3.4), which is part of the allowed act and carries only a
  name from the allow-list. A broker that cannot write a record allows nothing (§2.5).
- **I5. Properties are executable.** Every escape axis and every AC-8 row is a property a test checks
  by running something, and a check is labelled vacuous where nothing in P1 can make it fail (§8.2).
- **I6. One writer per audit file.** `egress.jsonl` is written only by the broker, of which one runs
  per user (§2.0). `sandbox.jsonl` is written only by runner invocations, one record each (§7.8).
- **I7. Server-held state is never taken from the caller.** The run id, the modes, `systemLocalOnly`,
  the scope set, the matched allow-list entry, the resolved address, the request path sent and every
  upstream header are the broker's own. What the caller declares is checked against them (§2.6).

## 2. Broker: process, interface and trust boundary

### 2.0 Process split (P1)

**One host process per user**, `ccg-broker`, owns the loaded allow-lists and modes, the run table and
tokens, rule evaluation, the upstream connection, and the **single audit writer** for
`<auditDir>/egress.jsonl`. It serves structured calls only.
- **Host network namespace.** It must reach `local` endpoints such as LM Studio on the Windows host:
  through WSL mirrored networking's `localhost` (WSL NAT mode is not supported, R-E8). A container
  cannot (ES1 §6 step 1).
- **Closed environment.** Started as `python -m ccg.broker.server …`, it re-executes itself, before
  reading anything else, as `<sys.executable> -I -m ccg.broker.server …` under `env -i` with exactly:
  a fixed `PATH=/usr/bin:/bin`, `HOME` from the passwd entry of its uid, `LANG=C.UTF-8`, and
  `CCG_AUDIT_DIR` if the caller set it. It skips the re-exec only when its environment and flags are
  already exactly that. So `PYTHONPATH`, the user site, the working directory on `sys.path`,
  `SSL_CERT_FILE`/`SSL_CERT_DIR`, and the resolver's `LOCALDOMAIN`, `RES_OPTIONS` and `HOSTALIASES`
  have no effect. Its TLS context loads the fixed bundle `/etc/ssl/certs/ca-certificates.crt`, never
  the default verify paths. It runs the interface reader as the absolute `/usr/bin/ip`. Its HTTP
  client reads no proxy variable and no `.netrc` (the standard library's `http.client` reads none).
- **The broker directory** is `<home>/.local/state/ccg/broker/`. At start the broker creates it if
  absent (mode `0700`) and then requires it to be a plain directory (not a symlink), owned by its uid,
  mode `0700`, with no group- or world-writable ancestor; otherwise it exits 69.
- **Singleton per user.** In that directory it opens `broker.lock` (`O_NOFOLLOW`, mode `0600`) and
  takes `fcntl.flock(LOCK_EX | LOCK_NB)`, held for its lifetime; if the lock is held it exits 69. The
  lock and the socket share one key, the user's broker directory, whatever `CCG_AUDIT_DIR` says, so a
  broker started from another checkout or with another audit directory cannot take over a live one.
- **Listener: one Unix-domain stream socket, `broker.sock` in that directory, and nothing else.** No
  TCP or UDP listener. Holding the lock, the broker first tries to `connect()` to an existing
  `broker.sock`; if something answers, it exits 69; otherwise it unlinks the leftover file and binds
  with `umask(0o177)` in force, so the socket is created `0600`.
- **Client discovery.** A client uses `CCG_BROKER_SOCKET` if set, else
  `<its own passwd home>/.local/state/ccg/broker/broker.sock`. The broker's `--socket <path>` option
  exists for tests and is subject to the same directory checks.
- **Peer check.** On every accepted connection the broker reads `SO_PEERCRED` and closes the
  connection, with no reply, unless the peer uid equals its own. That is not an egress request and
  writes no record; it is counted on the broker's stderr log, as are malformed frames (§2.2) and
  `open-run-invalid` errors (§2.1). The directory mode is the control; the peer check is defence in
  depth.
- **Inputs, closed.** Argv: `--system <manifest path>` (at least one, repeatable) and `--socket`.
  Environment: `CCG_AUDIT_DIR` only (§5.1).
- **Configuration load.** Each manifest goes through P1-03's `ccg.manifest.load_manifest`, which
  validates the `egress` section against `egress-v1.schema.json` and checks S1–S5 of §4.3; the broker
  then checks S6, which needs its interface reader. Any failure exits 64 naming the file and the
  violation. There is no reload: a changed allow-list is a
  restart. Every `policy` record carries the digest of what was loaded (`allowListDigest`, §5.2).
- **Exit codes.** 0 on a clean stop; 64 on a usage or configuration error; 69 on a precondition
  failure (lock held, live socket, directory check, audit directory check of §5.1).

**Why this split.** Option C removes the one reason for a second process: the proxy listener, which
had to live in a container attached to per-invocation networks (ES1 §6 step 1). What is left needs
the host network and one audit writer. Rejected:
- **A broker container on a bridge network** cannot reach mirrored-network `localhost`, and with
  `--network host` it is a host process with more moving parts.
- **A policy library linked into each component** gives one audit writer per process and makes the
  run scope a value each caller holds, so I6 and I7 fail by construction.
- **A TCP loopback listener.** Under WSL mirrored networking a Linux `localhost` listener is also
  reachable from Windows processes (from Microsoft's mirrored-mode documentation, not probed), and a
  container network reaches host listeners through its gateway (S04 gate M4). A Unix socket has
  neither exposure, and gives the filesystem mode and the peer uid as controls.

### 2.1 Run credential

```
open_run(runKind: "refresh" | "answer" | "vulndb", systemId: str,
         codebases: list[repoId] | "*", override: "local" | "hosted" | None)
  -> RunGrant {runId: uuid, token: str, effective: {repoId: mode}, systemLocalOnly: bool}
close_run(token) -> None
```

- `systemId` is the system name of a loaded manifest; each `repoId` must be one of its repositories.
  `"*"` means every repository of the system; interpretation uses it, because the touched
  repositories are not known when a question is interpreted (FR-9, V3-Q3).
- **Cardinality.** A `refresh` or `answer` run has `"*"` or at least one distinct codebase. A `vulndb`
  run has `[]` exactly. Any violation, an unknown system or an unknown repository is the error
  `open-run-invalid`: no token is issued and no record is written, because there is no run.
- `open_run` performs §3.2's mode resolution and writes one `policy` record. The grant is returned
  only after that record is durable (I4); if it cannot be written, no token is issued (§2.5).
- **The token** is 32 bytes from the operating system's CSPRNG (`secrets.token_bytes`), presented as
  64 lowercase hex characters. The broker stores only its SHA-256 and looks a presented token up by
  that hash. It expires at `close_run` or 24 h after `open_run`; a broker restart drops every run.
- **The token is never written** to a record, a log line or an argv. A client holds it in memory and
  passes it to a child process only through the environment variable `CCG_RUN_TOKEN`. It never
  reaches a sandboxed tool: the runner's environment is closed (§7.7).
- `close_run` is idempotent and writes no record.
- **Who opens runs in P1.**
  - **Refresh (P1-20)** opens one `refresh` run **only when the operator gives an override**, carrying
    it, so that the resolution and any refusal are recorded (rows 8.10, 8.11). Without an override
    the effective modes are the defaults, every P1 build is `offline` whatever the mode, and there is
    nothing to resolve, so a refresh with no override needs no broker. A refresh given an override
    fails if the broker is absent (§2.5); its report names the `policy` record's `auditId`.
  - **The plain-language front (P1-26)** opens at most one `answer` run per question, with
    `codebases = "*"`, at the question's first model call (from `ccg/front/interpret.py` or
    `ccg/front/summary.py`, whichever comes first), and uses that token for every later model call of
    the question. A question the deterministic parsers resolve without a model call opens no run.
  - **The clients (P1-25)** open no run and import no broker module: an MCP or CLI `ask` that reaches
    the front gets the front's single run.

### 2.2 Egress request and framing

```
EgressRequest {
  component: Component          # §2.4
  purpose: Purpose              # §2.4
  payloadClass: PayloadClass    # §2.4
  endpoint: "scheme://host:port"  # canonical form, §4.3
  target: {method: "GET" | "POST", path: str}   # checked against the entry, §3.3 R0
  codebases: list[repoId]       # the codebases this request serves; ⊆ the run's codebases
}
request(token, EgressRequest, body: JSON value | null)
  -> {decision: Decision, response: {status: int, contentType: str | null, body: bytes} | null}
Decision { decision: "allow" | "deny", rule: RuleId, reason: DenyReason | null, auditId: uuid }
```

There is no `runId` in the request: the run is the token's (I7). `body` is `null` exactly when
`target.method = GET`. `response` is non-null only for an allow whose upstream exchange completed
(§2.3).

**Framing on the socket.** Each message, in either direction, is a 4-byte big-endian length followed
by that many bytes of UTF-8 JSON. The broker reads the length first and closes the connection, with
no decision, if it exceeds `MAX_REQUEST_BYTES + 64 KiB` (inbound) or `MAX_RESPONSE_BYTES + 64 KiB`
(outbound). The JSON is parsed strictly: a duplicate object key, `NaN` or `Infinity`, nesting deeper
than 64 levels, or an integer outside ±2⁶³ makes the frame malformed, and a malformed frame closes the
connection with no decision. The broker serializes with `allow_nan=False`.

### 2.3 Transport: structured calls only

- **The broker builds the whole upstream request.** The method comes from `target`. The path sent is
  **the matched entry's stored `paths` string** that equals `target.path`, never the caller's string.
  The broker sets the only headers sent: `Host`, `Content-Type: application/json` and
  `Content-Length` (for `POST`), `Accept: application/json`, `Accept-Encoding: identity`, and
  `User-Agent: ccg-broker/1`. The caller supplies no header, query string, fragment or user-info. P1
  sends no credential header; a later phase's credentials are broker-held (proposal §3.12).
- **Body.** For `POST` the broker serializes `body` as UTF-8 JSON. `decision.payloadFields` records
  the top-level keys sent, plus `identity.*` keys for `repo-names-and-identity`, and
  `decision.bodySha256` the digest of the serialized bytes.
- **The `repo-names-and-identity` shape, under `R4`** (`R1` and `R7` admit any body). The body must be
  exactly `{repoNames, identity: {subject, issuer?}}` where:
  - `repoNames` is a non-empty array of **distinct repository ids of the run's system**. The whole
    system rather than the run's scope set, because an access check for one question may name
    repositories outside the touched set; the system's ids are what FR-9 lets leave, and the caller
    can only choose among them.
  - `identity.subject` full-matches `^[\x21-\x7e]{1,256}$`.
  - `identity.issuer`, when present, equals the canonical endpoint of an `outbound` entry of
    `kind: idp` in the same manifest.

  Anything else is `R4`, `payload-shape`. The bound leaves `identity.subject` as the one free string
  (at most 256 printable ASCII characters per request, §11 R-E6); P4's identity adapter is where the
  broker derives the identity itself instead of taking it from the caller.
- **Bounds** (module constants of `ccg.broker.structured`): `MAX_REQUEST_BYTES = 4 MiB` of serialized
  body (over it: `R0`, `body-too-large`); `MAX_RESPONSE_BYTES = 16 MiB`, counted in bytes read from
  the socket, the read aborted at the bound plus one (the response is discarded, the caller gets none,
  `closeReason = response-too-large`); `UPSTREAM_TIMEOUT_S = 300` as a **wall-clock deadline** for the
  whole exchange, not a per-read timeout (`timeout`). A response with any `Content-Encoding` other
  than `identity` is discarded (`upstream-error`). The response is buffered whole: P1 has no
  streaming.
- **Connection.** The broker connects to the address §3.4 validated and never re-resolves. `https`
  uses the fixed CA bundle (§2.0) with hostname verification against the entry's host (SNI = that
  host). `http` is admitted only for `local` entries (§4.3).
- **No redirects.** A 3xx response is returned to the caller as the response. Following it would be
  a new request, evaluated from the start.
- **Proxied streams** do not exist in P1 (§6).

### 2.4 Closed enumerations (v1)

| Field | Values |
|---|---|
| `Component` | `build.git`, `build.frontend`, `build.indexer`, `build.pkgmgr`, `answer.interpret`, `answer.summarize`, `answer.pr-adapter`, `access.idp`, `access.check`, `vulndb.refresh` |
| `Purpose` | `git-fetch`, `dependency-download`, `model-call`, `idp`, `access-check`, `pr-addon`, `vulndb-refresh`, `import-name-lookup` |
| `PayloadClass` | `none` (no body), `repo-names-and-identity` (§2.3's shape), `repo-derived` (anything derived from repository content: source, names, paths, messages, traces, dependency lists; FR-9), `vuln-feed-request` (feed protocol only; no repository content) |
| `Method` | `GET`, `POST` |

A value outside these sets is `R0`. Adding a value is a revision of this spec, which is also AC-8's
re-run trigger (row 8.14). The `Purpose` set keeps the deferred path's values so that a request
naming one is denied by its own row with its own reason (§3.3) rather than as malformed.

### 2.4a Components per run kind, and per purpose

`R0` denies a component outside its run kind, and a purpose its component may not declare.

| Run kind | Components | P1 caller (informational: what the AC-8 scenario exercises) |
|---|---|---|
| `refresh` | `build.*`, `access.*` | **none.** Sandboxed tools have no network (§7); refresh orchestration reads local heads only (index §6). The run exists to record an override's resolution. |
| `answer` | `answer.*`, `access.*` | `answer.interpret` and `answer.summarize`, from the front (P1-26), to a `local` model. `answer.pr-adapter` is P3; `access.*` is P4. |
| `vulndb` | `vulndb.refresh` | none: the vulnerability-database job is P5. `R3` is built and unit-tested now. |

| Component | Purposes it may declare |
|---|---|
| `build.git` | `git-fetch` |
| `build.frontend`, `build.indexer`, `build.pkgmgr` | `dependency-download`, `import-name-lookup` |
| `answer.interpret`, `answer.summarize` | `model-call` |
| `answer.pr-adapter` | `pr-addon` |
| `access.idp` | `idp` |
| `access.check` | `access-check` |
| `vulndb.refresh` | `vulndb-refresh` |

### 2.5 Errors and failure behaviour

- **A denial is a normal `Decision`**, not an error. The caller surfaces it as "refused by egress
  policy (`<rule>`: `<reason>`)" and never retries through another path.
- **Audit failure latches the broker closed.** If a `policy` or `decision` record cannot be written
  and made durable, the broker answers that call with the error `audit-unwritable` (no `Decision`,
  since no record exists), sends no upstream byte for it, and from then on refuses every `open_run`
  and `request` the same way until it is restarted. A `close` that cannot be written after a
  completed exchange still returns the response, then latches. An exchange already in flight when
  another call trips the latch completes, attempts its `close`, and returns its response.
- **A broker that is absent or latched fails the operation.** A client that cannot reach the broker,
  or receives `audit-unwritable`, fails the model call (or the overridden refresh) with that reason.
  **It never falls back to a direct connection**; P1-20 and P1-26 each test it (§10.2).

### 2.6 Trust boundary of the structured-call path

The caller is any process of the broker's uid that holds a run token. Each input it supplies, and
what the broker does with it:

| Input | Trusted? | Checked by |
|---|---|---|
| frame | untrusted bytes | length cap before reading; strict JSON (§2.2) |
| token | proves membership of one run, nothing more | lookup by hash; `R0` `token-invalid` |
| `component`, `purpose`, `payloadClass`, `target.method` | declared, not trusted | closed enums, §2.4a's two tables (`R0`); each allow rule names the values it admits |
| `codebases` | declared, never allowed to narrow a run's scope | `R0` subset, distinctness and cardinality checks; an `answer` run's scope is the run's codebases, not the declaration (§3.3) |
| `endpoint` | declared | must be canonical; allowed only on exact equality with an allow-list entry |
| `target.path` | declared | must equal one of the matched entry's `paths`; the entry's string is what is sent (`R0`, §2.3) |
| `body` | opaque, except under `R4` | size bound; under `R4`, closed keys and closed or bounded values (§2.3); a `repo-derived` body is allowed only to `local` (`R1`), or to `hosted` when the whole scope is hosted (`R7`) |
| run id, modes, `systemLocalOnly`, scope set, resolved address, headers, credentials | never from the caller | broker state (I7) |

**What this boundary does not cover, stated plainly.**
- **A process of the same uid that bypasses the broker** by opening its own socket. Host processes
  have the host's network. In P1 that is closed for our own code by I2's import-boundary test
  (§10.1), and shown for the answer path by the AC-8 scenario, whose answer components run in
  containers whose only ways out are the broker's socket and the store (§8.1). It is not closed
  against a hostile same-uid process.
- **A same-uid process opening its own run, replacing the socket file, or editing the audit files.**
  The token separates runs; it does not authenticate a principal. The records carry no tamper
  evidence (§11, R-E7). Root and same-uid compromise are out of scope.
- **The broker's own upstream traffic** is attested by its records, and in the scenario by an
  optional syscall trace (§8.2, row 8.16; §11, R-E1).

## 3. Policy evaluation

### 3.1 Inputs

The run's system manifest (its `egress` section, §4, and each repository's mode, §4.4); the run's
`override`; the request (§2.2).

### 3.2 Mode resolution (once per run, in `open_run`)

For each codebase `c` of the run:

| `default(c)` | override | `effective(c)` | `policy.modes[].rule` |
|---|---|---|---|
| `local-only` | none or `local` | `local-only` | `P1-DEFAULT` |
| `local-only` | `hosted` | **`local-only`** | **`P2-OVERRIDE-REFUSED`**; `overrideRefused: true` |
| `hosted` | none or `hosted` | `hosted` | `P1-DEFAULT` |
| `hosted` | `local` | `local-only` | `P3-OVERRIDE-LOCAL` |

- **A refused override does not abort the run.** The codebase proceeds as `local-only`, and the
  refusal is visible in the `policy` record and in the run's report (ES1 §11 records why).
- `systemLocalOnly` is true if **any repository of the system** is `local-only` for this run: its
  default is `local-only`, or it is a run codebase with `effective = local-only`. It is computed over
  the whole system (FR-9: "in a system containing any local-only repository").

### 3.3 Request evaluation (per request, first match wins)

Let `M = local-only` if any codebase in the **scope set** has `effective = local-only`, else `hosted`.
- **`answer`:** the scope set is the run's codebases. FR-9 is sticky per answer, so a call that
  declares fewer codebases than its answer touches cannot leave local-only.
- **`refresh`:** the scope set is `request.codebases`. That declaration cannot widen any allow: `R7`
  is the only allow that reads `M`, it needs `purpose = model-call`, which no `refresh` component may
  declare (§2.4a).
- **`vulndb`:** the scope set is empty, so `M = hosted`; only `R3` can allow anything in such a run,
  because `vulndb.refresh` may declare only `vulndb-refresh`.

| Order | Rule | Condition | Decision | `reason` |
|---|---|---|---|---|
| 1 | `R0-INVALID` | any check of the `R0` list below fails | deny | the first failing check's reason |
| 2 | `R2-GIT-OWN-REMOTE` | `purpose = git-fetch` | **deny: not implemented in P1** | `not-in-p1:git-fetch` |
| 3 | `R8-CONTROLLED-REGISTRY` | `purpose = dependency-download` | **deny: not implemented in P1** | `not-in-p1:controlled-download` |
| 4 | `R9-IMPORT-NAME-SERVICE` | `purpose = import-name-lookup` | **deny: not implemented in P1** | `not-in-p1:import-name-service` |
| 5 | `R1-LOCAL-ENDPOINT` | `endpoint` ∈ `local` | allow | — |
| 6 | `R3-VULNDB` | the run's `runKind = vulndb`, `purpose = vulndb-refresh`, `component = vulndb.refresh`, `target.method = GET`, `payloadClass = none`, `endpoint` ∈ `outbound` with `kind: vulndb` | allow | — |
| 7 | `R4-NAMED-SERVICE` | `purpose ∈ {idp, access-check, pr-addon}` and `endpoint` ∈ `outbound` with the same `kind` | allow iff `payloadClass = repo-names-and-identity` and the body has §2.3's shape; otherwise deny | `payload-shape` |
| 8 | `R5-SYSTEM-LOCAL-MODEL` | `purpose = model-call` and `systemLocalOnly` | deny | `system-local-only-model` |
| 9 | `R6-STICKY-LOCAL` | `M = local-only` | deny | `sticky-local-only` |
| 10 | `R7-HOSTED-SERVICE` | `M = hosted`, `purpose = model-call`, `endpoint` ∈ `hosted` | allow | — |
| 11 | `R99-DEFAULT-DENY` | nothing above matched | deny | `no-rule-matched` |

**The `R0` checks, in order** (the first that fails is the recorded reason):
1. `token-invalid`: no token, or an unknown or expired one.
2. `enum-invalid`: `component`, `purpose`, `payloadClass` or `target.method` outside §2.4.
3. `component-not-in-run-kind`: §2.4a's first table.
4. `component-purpose-mismatch`: §2.4a's second table.
5. `codebases-empty`: `request.codebases = []` in a run other than `vulndb`.
6. `codebases-in-vulndb`: non-empty `request.codebases` in a `vulndb` run.
7. `codebases-not-subset`: `request.codebases` has a repeated element or one outside the run's
   codebases.
8. `endpoint-not-canonical`: `endpoint ≠ canonical(endpoint)` (§4.3), or unparseable.
9. `path-not-allowed`: `endpoint` equals an entry and `target.path` is not in that entry's `paths`.
10. `method-body-mismatch`: `GET` with a non-null body or a `payloadClass` other than `none`, or
    `POST` with a null body.
11. `body-too-large`: §2.3's request bound.

**Notes on the order.**
- **`R2`, `R8` and `R9` sit before `R1` in P1.** Their purposes belong to proxied streams, which P1
  does not have (§6). A structured request naming one is a caller defect, so it is denied whatever
  its endpoint, `local` included. Because `R0` runs first, a test or control that wants the
  `not-in-p1:*` reason must pass `R0`: the component of §2.4a's second table, a run kind admitting it
  (`refresh`), and, for a `local` endpoint, a listed path. The rule ids are kept so records stay
  comparable when the deferred path restores the designed conditions and positions (§6, input 5).
- **`R1` precedes every other allow:** "the machine" may receive anything, in either mode; S6 (§4.3)
  is what makes `local` mean the machine.
- **`R3` and `R4` precede `R5`/`R6`,** so local-only can use them, with their content bounds. They are
  the only outbound entries local-only allows in P1 (FR-9, V3-Q4, SQ-E; the third, a repository's own
  git remote, is the deferred `R2`). `R3` admits only a bodiless `GET` until P5 defines the feed
  request.
- **`R5` precedes `R7`,** so a hosted-only question in a system with any local-only repository is
  still denied a hosted model (AC-8).

### 3.4 Address validation (after an allow)

Rule evaluation needs no network. Only when `R3`, `R4` or `R7` would allow does the broker resolve the
entry's host, so **a denied request causes no DNS query**.
- **`local` entries are never resolved.** Their host is an IP literal, used as is, or `localhost`,
  which the broker maps to `127.0.0.1` itself. Before each `R1` connection the broker re-applies S6
  (§4.3) to that address with interface data read as below, so an address the host has since lost is
  no longer `local`; a failure is a deny, as in the last bullet.
- **Other entries** are resolved once, with the system resolver, as the absolute name `host + "."`, so
  no search domain applies. **Every** returned address must pass the forbidden set below, or the
  request is denied; otherwise the broker connects to the first address returned, with no second
  resolution.
- **The forbidden set** is checked against explicit networks, not against a library's address-class
  predicates. The interface data is re-read from `/usr/bin/ip -j -d address show` and
  `/usr/bin/ip -j route show default` (and `-6`) at each validation.
  - **Always forbidden, IPv4:** `0.0.0.0/8`, `127.0.0.0/8`, `169.254.0.0/16`, `192.0.0.0/24`,
    `198.18.0.0/15`, `224.0.0.0/4`, `240.0.0.0/4` (which includes `255.255.255.255`).
  - **Always forbidden, IPv6:** `::/128`, `::1/128`, `fe80::/10`, `ff00::/8`, and every prefix that
    embeds an IPv4 address: `::ffff:0:0/96` (IPv4-mapped), `::/96` (IPv4-compatible),
    `64:ff9b::/96` and `64:ff9b:1::/48` (NAT64), `2002::/16` (6to4), `2001::/32` (Teredo). These are
    denied outright rather than mapped and re-checked.
  - **Always forbidden, from the host:** every address of a host interface; every address inside the
    network of each bridge-kind host interface (Docker's `docker0` and `br-*`, so every Docker subnet
    and gateway); each default gateway of a host interface; every
    `local` entry's address.
  - **Forbidden unless the entry declares `privateAddress: true`:** `10.0.0.0/8`, `172.16.0.0/12`,
    `192.168.0.0/16`, `100.64.0.0/10`, `fc00::/7`. The opt-in is per entry and visible in review; it
    never admits an address of the bullets above (ES1 §12.1, m4, records why a blanket ban was
    rejected).
- **A failure turns the allow into a deny** with `rule = R0-INVALID` and reason `resolve-failed`
  (no address), `forbidden-address`, or `address-check-unavailable` (the interface data could not be
  read). The record's `endpoint` shows which entry it was.

### 3.5 Deny reasons: the closed vocabulary

`DenyReason` is exactly the reasons named in §3.3 and §3.4: `token-invalid`, `enum-invalid`,
`component-not-in-run-kind`, `component-purpose-mismatch`, `codebases-empty`, `codebases-in-vulndb`,
`codebases-not-subset`, `endpoint-not-canonical`, `path-not-allowed`, `method-body-mismatch`,
`body-too-large`, `resolve-failed`, `forbidden-address`, `address-check-unavailable`,
`not-in-p1:git-fetch`, `not-in-p1:controlled-download`, `not-in-p1:import-name-service`,
`payload-shape`, `system-local-only-model`, `sticky-local-only`, `no-rule-matched` (21 values). An
allow has `reason: null`; a deny never does. Adding a reason is a spec revision.

## 4. Allow-list format and `egress-v1.schema.json`

### 4.1 Location and ownership

The allow-list is the `egress` section of each system manifest (YAML, P1-03's file); it is the
system's only allow-list (P1-03's former `endpointAllowList` is removed). Each repository entry of the
same manifest carries its mode (§4.4). **`cpg/ccg/schemas/egress-v1.schema.json` is created by P1-03
(`tdd-engineer`), byte-for-byte the JSON of §4.2**, beside P1-03's own
`system-manifest-v1.schema.json`; P1-03's `load_manifest` validates the section against it and applies
§4.3's semantic checks. P1-29 consumes the loaded result. This document owns the schema's content; a
change to either is a revision of this spec.

**Not in the P1 shape, and rejected as unknown keys:** `registries`, `importNameServices`,
`gitFetchMaxBytesOut` (all deferred, §6). A repository's `remote` key is rejected by P1-03's loader
with "git fetch is not enabled in P1" (index §6).

### 4.2 The schema (normative)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "urn:ccg:egress-v1",
  "title": "ccg system manifest: the egress section, v1 (P1 shape)",
  "type": "object",
  "additionalProperties": false,
  "required": ["schemaVersion", "local", "outbound", "hosted"],
  "properties": {
    "schemaVersion": {"const": 1},
    "local": {"type": "array", "items": {"$ref": "#/$defs/localEntry"}},
    "outbound": {"type": "array", "items": {"$ref": "#/$defs/outboundEntry"}},
    "hosted": {"type": "array", "items": {"$ref": "#/$defs/hostedEntry"}}
  },
  "$defs": {
    "egressMode": {"enum": ["local-only", "hosted"]},
    "id": {"type": "string", "pattern": "^[a-z][a-z0-9-]{0,62}$"},
    "note": {"type": "string", "maxLength": 500},
    "paths": {
      "type": "array", "minItems": 1, "uniqueItems": true,
      "items": {"type": "string", "maxLength": 256, "pattern": "^/[A-Za-z0-9._~!$&'()*+,;=:@%/-]*$"}
    },
    "localEndpoint": {
      "type": "string",
      "pattern": "^https?://(?:localhost|(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])(?:\\.(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3}|\\[[0-9a-f:.]+\\]):[1-9][0-9]{0,4}$"
    },
    "remoteEndpoint": {
      "type": "string",
      "pattern": "^https://(?:(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\\.)+[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?|(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])(?:\\.(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3}|\\[[0-9a-f:.]+\\]):[1-9][0-9]{0,4}$"
    },
    "localEntry": {
      "type": "object", "additionalProperties": false,
      "required": ["id", "kind", "endpoint", "paths"],
      "properties": {
        "id": {"$ref": "#/$defs/id"},
        "kind": {"enum": ["model", "service"]},
        "endpoint": {"$ref": "#/$defs/localEndpoint"},
        "paths": {"$ref": "#/$defs/paths"},
        "note": {"$ref": "#/$defs/note"}
      }
    },
    "outboundEntry": {
      "type": "object", "additionalProperties": false,
      "required": ["id", "kind", "endpoint", "paths"],
      "properties": {
        "id": {"$ref": "#/$defs/id"},
        "kind": {"enum": ["vulndb", "idp", "access-check", "pr-addon"]},
        "endpoint": {"$ref": "#/$defs/remoteEndpoint"},
        "paths": {"$ref": "#/$defs/paths"},
        "privateAddress": {"type": "boolean"},
        "note": {"$ref": "#/$defs/note"}
      }
    },
    "hostedEntry": {
      "type": "object", "additionalProperties": false,
      "required": ["id", "kind", "endpoint", "paths"],
      "properties": {
        "id": {"$ref": "#/$defs/id"},
        "kind": {"const": "model"},
        "endpoint": {"$ref": "#/$defs/remoteEndpoint"},
        "paths": {"$ref": "#/$defs/paths"},
        "privateAddress": {"type": "boolean"},
        "note": {"$ref": "#/$defs/note"}
      }
    }
  }
}
```

Example, this repository's manifest (`cpg/ccg/system.yaml`, P1-03's file; the `paths` list is
P1-26's to fix to what its client calls):

```yaml
egress:
  schemaVersion: 1
  local:
    - id: lmstudio
      kind: model
      endpoint: http://127.0.0.1:1234
      paths: [/v1/chat/completions]
      note: LM Studio on the Windows host, through WSL mirrored networking
  outbound: []
  hosted: []
repos:
  - repoId: graphmind-ai-lab
    policy:
      egress: local-only
```

### 4.3 Matching and semantic rules

- **Canonical endpoint:** `scheme://host:port`, scheme and host lowercase, port always explicit,
  no path, query or user-info, an IPv6 literal bracketed in RFC 5952 form. The manifest must already
  be canonical (the schema's patterns plus S5); a request's endpoint must equal `canonical(endpoint)`
  (`R0`). `canonical()` is one function, `ccg.manifest.canonical_endpoint` (P1-03's), which the broker
  imports.
- **Exact match only.** No wildcards, CIDR ranges or suffix matches. A request matches an entry when
  the canonical endpoints are equal.
- **Schemes.** `local` entries may be `http` or `https`; `outbound` and `hosted` entries are `https`
  only, so nothing leaves the machine in clear text.
- **Semantic checks the schema cannot express** (each refuses the manifest, and so the broker's start,
  exit 64):
  - **S1** every `id` is unique across the three sections;
  - **S2** every endpoint appears in at most one entry across the three sections;
  - **S3** every port is ≤ 65535, and every IPv6 literal equals its RFC 5952 compressed form;
  - **S4** no `local` IP literal is unspecified or multicast (`0.0.0.0`, `::`, `224.0.0.0/4`, `ff00::/8`);
  - **S5** every endpoint equals `canonical(endpoint)`, and no string in the section contains a
    character ≤ U+001F or U+007F (Python's `re` lets `$` match before a final newline, so the
    patterns alone admit one);
  - **S6 `local` is the machine:** a `local` IP literal is in `127.0.0.0/8` or `::1`, or is an address
    of a host interface, read as §3.4 reads them. A default gateway is never admitted: on native
    Linux and under WSL mirrored mode it is the LAN router. S6 is the broker's check, at its start and
    before each `R1` connection (§3.4, P1-29); S1–S5 are P1-03's loader's.

### 4.4 The per-repository mode

Each repository entry of a system manifest carries `policy.egress`, **mandatory**, whose value set is
`egress-v1.schema.json`'s `$defs/egressMode` (`local-only` | `hosted`). There is no implicit default
and no boolean twin (it replaces P1-03's `policy.localOnly`). P1-03's `system-manifest-v1.schema.json`
states the enum inline, and a P1-03 test asserts it equals `$defs/egressMode` of the egress schema;
both files are P1-03's, so the test has no outside dependency.

## 5. Audit records

### 5.1 Common rules

- **Storage.** Append-only JSONL, one record per line, UTF-8. Broker records go to
  `<auditDir>/egress.jsonl`, sandbox records to `<auditDir>/sandbox.jsonl`. `auditDir` is
  `CCG_AUDIT_DIR` when set, else `cpg/.audit/` resolved **from the code's own location**, never the
  working directory: the runner uses `<its directory>/../.audit`, the broker and §5.4's check use
  `<directory of the ccg package>/../.audit`, which are the same directory in one checkout. Every
  `policy` record carries the absolute `auditDir` it used. The runner never passes it into the
  sandbox. The system never rotates or deletes either file.
- **One writer per file (I6).** Only the broker writes `egress.jsonl`; inside it, records go through
  one writer. Only runner invocations write `sandbox.jsonl`, each exactly one record (§7.8). Records of
  a run all go to the one broker's `auditDir`, whichever checkout opened the run.
- **Broker-side hardening.** At start the broker requires `auditDir` to be a plain directory (not a
  symlink) owned by its uid and not group- or world-writable, else exit 69. It opens `egress.jsonl`
  with `O_WRONLY | O_APPEND | O_CREAT | O_NOFOLLOW | O_CLOEXEC`, mode `0600`, then `fstat`s it and
  requires a regular file, owned by its uid, `mode & 0o077 == 0`, `nlink == 1`, else exit 69; a newly
  created file is followed by an `fsync` of the directory. If the file does not end in `\n`, the
  broker appends one before its first record and counts it on stderr. The runner's own file handling
  is as built (§11, R-E7).
- **Atomicity and durability.** Each record is one `write(2)` of the whole line under an exclusive
  `flock` on the file; a short write counts as a failed write. A broker record is followed by
  `fsync`: before the grant returns (`policy`), before any upstream connection (`decision`), before
  the response returns (`close`, with §2.5's exception when it fails). A reader skips a trailing
  partial line.
- **Validated on both sides.** `cpg/ccg/schemas/egress-audit-v1.schema.json` (P1-29) holds one `$defs`
  entry per broker `recordType` of §5.2; `cpg/ccg/schemas/sandbox-record-v1.schema.json` (P1-18) holds
  §5.3's. In each, every listed field is required, `additionalProperties` is false, and enums admit
  **only the values P1 can write**; a deferred-path value is a schema revision. The broker validates
  each record before writing it; every reader (§5.4, §8) validates on reading. **A caller-supplied
  value that does not fit its field is recorded as `null`**, with the SHA-256 of its raw JSON in the
  record's `rejected` map, so caller input can never make a record invalid. A broker-built record
  that fails validation is a broker defect and latches the broker closed (§2.5).
- **Common fields:** `schemaVersion: 1`, `recordType`, `auditId` (lowercase UUIDv4), `ts` (RFC 3339
  UTC with milliseconds).
- **Content.** Records hold repository names, paths, argv and endpoints. They are local files that
  never leave the machine; the broker never sends a record anywhere. A run token is never in a record.

### 5.2 Broker records

**`policy`**: one per successful `open_run` (§3.2).

| Field | Type | Meaning |
|---|---|---|
| `runId` | uuid | |
| `runKind` | `refresh` \| `answer` \| `vulndb` | |
| `systemId` | str | |
| `codebases` | [repoId] | the run's codebases, `"*"` expanded |
| `override` | `local` \| `hosted` \| null | as requested |
| `modes` | [{`repoId`, `default`, `effective`, `rule`}] | one per run codebase; `rule` ∈ `P1-DEFAULT`, `P2-OVERRIDE-REFUSED`, `P3-OVERRIDE-LOCAL` |
| `overrideRefused` | bool | true iff any row is `P2-OVERRIDE-REFUSED` |
| `systemLocalOnly` | bool | §3.2 |
| `allowListDigest` | str | lowercase hex SHA-256 of the canonical JSON (sorted keys, no whitespace) of `{egress, modes: {repoId: default}}` as loaded for this system |
| `auditDir` | str | the absolute audit directory (§5.1) |

**`decision`**: one per `request`, written and made durable before any upstream connection.

| Field | Type | Meaning |
|---|---|---|
| `runId` | uuid \| null | from the token; null when `token-invalid` |
| `component`, `purpose`, `payloadClass` | §2.4 enum \| null | as declared; null when outside the enum |
| `target` | {`method`: §2.4 enum \| null, `path`: str \| null} | as declared; `path` null when not a string or over 256 characters |
| `endpoint` | canonical endpoint \| null | null when not canonical |
| `codebases` | [repoId] \| null | as declared; null when not a list of repoId strings no longer than the system's repository count |
| `rejected` | {field: sha256} | the digest of each caller value recorded as null; `{}` when none |
| `resolvedAddr` | str \| null | the address connected to (`local`: the literal, or `127.0.0.1` for `localhost`); null on deny |
| `privateAddress` | bool | true iff the matched entry declares `privateAddress: true`; false on deny and for `local` |
| `requestMode` | `local-only` \| `hosted` \| null | `M` of §3.3; null when `token-invalid` |
| `decision` | `allow` \| `deny` | |
| `rule` | RuleId | one of the eleven ids of §3.3's table; §8's checks abbreviate them `R0` … `R99` |
| `reason` | DenyReason \| null | §3.5; null iff allow |
| `transport` | `structured` | the only P1 value |
| `payloadFields` | [str] \| null | the top-level keys sent, plus `identity.*` for `repo-names-and-identity`; null on deny |
| `bytes` | int | serialized request body bytes; 0 when there is no body |
| `bodySha256` | str \| null | SHA-256 of the serialized body; null when there is none |

**`close`**: one per allowed request, when its exchange ends.

| Field | Type | Meaning |
|---|---|---|
| `decisionAuditId` | uuid | the allow it closes |
| `runId` | uuid | |
| `bytesOut`, `bytesIn` | int | bytes written to and read from the upstream socket, framing included |
| `httpStatus` | int \| null | null when no response arrived |
| `durationMs` | int | |
| `closeReason` | `completed` \| `upstream-error` \| `timeout` \| `response-too-large` \| `broker-shutdown` | |

A `decision` allow with no `close` means the broker died mid-exchange; AC-8 fails on it (§8.2).

### 5.3 The `sandbox` record

Exactly one per invocation of `cpg/sandbox/offline-run.sh`, refusals included, except `-h`/`--help`
(exit 0, no record) and an invocation killed by a signal (exit `128+n`, no record; §7.8). The runner
allocates `auditId` before launch, makes it the stem of the tool's own output files, and writes the
record when the invocation ends. Fields marked "constant in P1" are written as literals by the
runner's `sbx_write_record`, so no check on them is evidence of anything (§8.2).

| Field | Type | Meaning |
|---|---|---|
| `backend` | `container` \| `namespace` | §7.1 |
| `toolchain` | {`image`, `imageId`} \| {`joernVersion`, `joernPath`, `javaVersion`} | container: tag and `sha256:` id; namespace: host Joern and JDK |
| `job` | `parse` | constant in P1 (the deferred path adds `git-fetch`) |
| `egressMode` | `offline` \| `controlled` | the validated `--mode`; `offline` when `--mode` is missing or invalid; `controlled` occurs only on a refused record |
| `network` | `none` | constant in P1 |
| `networkName` | null | constant in P1 |
| `launcherArgv` | [str] | the full `docker run …` / `bwrap …` argv as executed |
| `toolArgv` | [str] | launched: the argv as executed, after `@AUDIT_ID@` substitution and the runner's additions (§7.8); refused: the argv the caller passed after `--` |
| `frontend` | str | the §7.4 closed set; for `joern-parse`, the frontend its `--language` derives |
| `script` | {`path`, `sha256`} \| null | `frontend = joern` only: the path inside the sandbox and the SHA-256 of the staged copy |
| `flagCheck` | {`listDigest`, `forbiddenHits`: [str], `allowedHits`: [str], `liftedHits`: []} | `listDigest` = SHA-256 of `forbidden-flags.txt` and `allowed-flags.txt` concatenated; `liftedHits` constant `[]` in P1 |
| `envKeys` | [str] | the keys of the `K=V` arguments to `env -i` in `launcherArgv` (§7.6) |
| `mounts` | [{`target`, `source`, `mode`}] | every host path visible inside, each `/usr` entry and `/etc` file separately on the namespace backend, with `ro`/`rw` and `exec`/`noexec` |
| `preconditions` | {`interopPE`: `enabled`\|`disabled`\|`unreadable`, `binfmtPinned`: bool, `seccomp`: str \| null, `vsock`: `blocked`\|`present`\|`absent`, `netnsDistinct`: bool} | read before launch (§7.1); `netnsDistinct` is constant in P1: `true` on every launched record, set once a launcher exists (A1(c) is its probe) |
| `refused` | bool | |
| `refusalReason` | null \| `usage` \| `forbidden-flag` \| `forbidden-directive` \| `unknown-frontend` \| `mode-not-implemented` \| `backend-precondition` \| `backend-unavailable` | §7.5 |
| `exitCode` | int | the tool's exit code; 64 for a usage, argv or directive refusal; 69 for a backend refusal |
| `codebases` | [repoId] | the `--codebase` values |
| `runId` | null | constant in P1: the runner takes no run id (§6, input 10) |
| `durationMs` | int | |

Records of the test-only launch path carry `frontend: lib-launch` and go to a separate test audit
file, never to `sandbox.jsonl` (`cpg/sandbox/README.md`, deviation 9); `sandbox-record-v1.schema.json`
rejects `lib-launch`.

### 5.4 The `sandboxAuditIds` loader check

A bundle header's `sandboxAuditIds` (`docs/plans/connected-code-graph-bundle-spec3.md` §2.2) is checked
at assembly (P1-18) and again at load (P1-19), by one function both call, in a module with no broker
dependency:

```
ccg.sandbox_audit.check_sandbox_ids(header: Mapping, audit_dir: Path) -> list[str]   # [] = valid
```

`cpg/ccg/sandbox_audit.py` and `cpg/ccg/schemas/sandbox-record-v1.schema.json` are P1-18's. The bundle
is invalid, and refused, unless for **every** id in `header.sandboxAuditIds`:
1. `sandbox.jsonl` holds **exactly one** record with that `auditId` (zero or several is invalid);
2. that record validates against `sandbox-record-v1.schema.json`;
3. it has `refused: false` and `exitCode: 0`;
4. its `egressMode` equals `header.egressMode`, and in P1 both are `offline`.

Each error string names the id and the failed condition. **The check is presence-only:** it proves a
successful offline invocation with that id exists, not that the bundle's inputs came from it, because
the record carries no output digest (§6, input 13). **An empty `sandboxAuditIds`** is valid only for a host-extractor language (`shell`, `config`, `markdown`; `-bundle-spec6.md`) with `egressMode` `offline`, and proves nothing about any sandboxed invocation: `check_sandbox_ids` returning `[]` for it means *valid*, never *attested*. `ccg.sandbox_audit.attestation(header, audit_dir)` returns `sandboxed`, `host-unattested` or `invalid`; `host-unattested` applies to any valid host-language header, named ids included, and the assembly report lists those units as `unattested_units`.

## 6. Deferred: proxied streams (git fetch and the controlled download path)

**Status: not built in P1, and not P1 text.** Two deferred features share one transport, the
**proxied stream** (a sandboxed tool's connection carried through the broker):
- **git fetch** (`R2`), built in the first phase in which a manifest names an **https** `remote` for a
  repository (index §6; a backlog item at P1 close, index §11);
- **the controlled download path** for hosted-default builds (`R8`, `R9`), which is P2's (proposal §4,
  P2 row).

Whichever comes first builds the transport. Until then P1 enforces the absence four ways: P1-03
rejects `remote`; the egress schema has no `registries` or `importNameServices`; the runner refuses
`--mode controlled` (`mode-not-implemented`, §7.5); `R2`, `R8` and `R9` are deny rows (§3.3).

**Design inputs of the deferred path.** Recorded so that its designer starts from them; none binds
P1, and the designer re-derives each against the system as it then is.

| # | Input | Where it is worked out |
|---|---|---|
| 1 | The proxy listener runs in a container (a container cannot join `host` and another network), while the structured-call listener stays on the host; both share one policy state and **one audit writer**, so §2.0's split must be extended, for example with the proxy relaying each decision to the host broker over its socket. The proxy technology is chosen then (index §3.2 rejects a third-party proxy such as mitmproxy). | ES1 §6 step 1; index §4.2 |
| 2 | Binding by network attachment: `bind_proxy(runId, repoId, purpose, auditId, liftedEntryIds)`, no credential in the build's environment, a binding identified by receiving address **and** peer IP. | ES1 §2.3 |
| 3 | One `--internal` network `ccg-egress-<auditId>` per invocation, its label, the age-bounded reaper, refusal on address-pool exhaustion, never reuse. | ES1 §2.3 |
| 4 | **`--internal` reaches the gateway**, and with it host listeners (S04 gate M4); the A1c probe set, including no cross-binding reach, is the ship criterion; §8.1's P-a probe and mechanism candidates are the P1 precedent. | ES1 §7.2 (A1c) |
| 5 | `R2`, `R8`, `R9` with their designed conditions (`transport = proxy` through a binding) and positions (`R2` after `R1`; `R8`, `R9` after `R7`); the refresh-scope argument that rests on them. | ES1 §3.3 |
| 6 | The git-fetch job: runner-only, fixed argv, read-only deploy credential mounted per invocation, no hooks, a runner-owned bare mirror, the `gitFetchMaxBytesOut` bound and its calibration, `remote` https-only, possibly a list. This repository's `origin` is SSH-form, which a `CONNECT` broker cannot carry. | ES1 §6.1, §4.3; index §6 |
| 7 | **The host-granular tunnel** and TLS interception for `R2` (a broker-held CA trusted by the fetch container only). Option C removed the question for P1; it returns to the stakeholder with the trigger. | ES1 §6.1, §11 Q1 |
| 8 | The controlled download path: `registries` and `importNameServices` sections, `privateAddress` on registries, `liftedEntryIds` and the argv-audit lift of `--jar4import-url`, the controlled pass-list entries and `JAVA_OPTS` pattern, the controlled launcher clause, the cache volume keyed by lock-file digest, the rejected pre-fetch alternative. | ES1 §6, §7.3.4, §7.4 step 5 |
| 9 | AC-8 rows 8.12 and 8.13 and row 8.2's `R2` clause, with the `conn` ↔ `decision` join over proxied streams and the `decision.upstream.localPort` field it joins on (P1 builds neither). | ES1 §5.2, §8 |
| 10 | A runner `--run-id` option, so `sandbox` records name their run; P1 records `runId: null`. | §5.3 |
| 11 | The broker's own non-local upstream routed through the AC-8 logging proxy, an independent observation beyond P1's optional syscall trace (§11, R-E1). | ES1 §8 |
| 12 | The JDK's default refusal of Basic authentication on `CONNECT` tunnels (from memory of JDK documentation, never probed). | ES1 §2.3 |
| 13 | **Output digests in the `sandbox` record** (`outputs: [{path, sha256}]`), compared by §5.4 with digests the bundle header would carry, so the check proves provenance and not only presence. Both the runner and the bundle header change; it rides with input 10's runner revision. | §5.4 |

## 7. Sandbox contract

The runner is built and gated (P0 S08: `cpg/sandbox/offline-run.sh`, `forbidden-flags.txt`,
`allowed-flags.txt`, `tests/test_sandbox.sh`, `tests/probe.pl`, `tests/mutants.sh`). This section is
its contract as built; P1 changes only §7.4's frontend set (the indexer entries), §7.5's output refusal and `usage` list, §7.6's forbidden names and indexer pass-list entry, and §7.8's output convention. The evidence behind its rules is in §9.

### 7.0 Interface and guarantees

```
cpg/sandbox/offline-run.sh --backend container|namespace|auto --mode offline|controlled \
    --frontend <name> --source <dir> --output <dir> [--codebase <repoId>]... \
    [--script-file <host path>] -- <tool args...>
```

- `--script-file` is accepted only with `--frontend joern`, and is then mandatory (§7.4).
- `auto` picks `container` when Docker, the image and §7.1's preconditions hold, else tries
  `namespace`; if neither can start it exits 69 with one record.
- `--mode controlled` is refused (exit 64, `mode-not-implemented`) in P1 (§6).
- **Guarantees to the tool:** the source is readable at `/src`; `/out` is writable; `/tmp` (=
  `$TMPDIR` = `java.io.tmpdir`) and `$HOME` (`/home/ccg`) are writable scratch, and `/tmp` is the one
  writable location that allows executing what is written there; `localhost` resolves and is the
  hostname; the toolchain is at the image's paths (`cpg/infra/toolchain/Dockerfile`, stage
  `toolchain`); the environment is exactly §7.6's pass-list. Nothing else is promised.
- **The tool is always started through `env -i`**: the container runs `--entrypoint /usr/bin/env`
  with image arguments `-i K1=V1 … Kn=Vn <tool> <toolArgv…>`; the namespace backend uses
  `--clearenv` with the same wrapper. Docker cannot drop an image's `ENV` keys and adds `HOSTNAME`;
  `env -i` replaces the environment before it `exec`s the tool.
- The runner never builds `toolArgv` from environment variables or source-tree files, and builds
  `launcherArgv` from fixed templates, recorded verbatim.

### 7.1 Backends and their preconditions

| | **Container** (primary) | **Namespace** (fallback, hosts without Docker) |
|---|---|---|
| Launch | `docker run --rm --pull=never --name ccg-sbx-<auditId> --network none --hostname localhost --read-only --cap-drop ALL --security-opt no-new-privileges --user <uid>:<gid> --pids-limit 4096 --entrypoint /usr/bin/env` on `ccg-toolchain:<hash>`, with the daemon's built-in seccomp profile; container and volumes labelled `ccg.sandbox.auditId=<auditId>` | `bwrap --unshare-all --die-with-parent --new-session --clearenv --hostname localhost`, with the host Joern and JDK 21 |
| Mounts | §7.2 | §7.2 |
| Never present | `/mnt/*`, `/run/WSL`, `/init`, any `docker.sock`, `/usr/lib/wsl` | the same |
| Refuses, exit 69 `backend-precondition` | **C1:** an enabled PE `binfmt_misc` entry (matched as in N1 (a)–(d)) has flag `F`. **C2:** `docker info`'s security options lack `name=seccomp,profile=builtin`. | **N1:** `interop_state` is `enabled`, or `unreadable` with a kernel release containing `microsoft` (case-insensitive). **N2:** `/dev/vsock` exists on the host. |
| Unavailable, exit 69 `backend-unavailable` | no Docker daemon, or no image with the expected tag | no `bwrap`, or unprivileged user namespaces disabled |

**`interop_state(binfmt_dir, kernel_release) → enabled | disabled | unreadable`**, `binfmt_dir`
defaulting to `/proc/sys/fs/binfmt_misc`: `unreadable` if `binfmt_dir/status` is missing; `disabled`
if `status` reads `disabled`; otherwise `enabled` if any entry file (other than `status` and
`register`) whose first line is `enabled` has (a) a `magic` line starting `4d5a`, (b) an `extension`
of `exe`, case-insensitively, (c) an `interpreter` of `/init`, or (d) a file name matching
`WSLInterop*`; else `disabled`. The literal path `WSLInterop` must not be relied on: this host's entry
is `WSLInterop-late`. Test 8's fixtures F1–F12 and its Host row are ES1 §7.1.1's table, encoded in
`test_sandbox.sh`. **On WSL2 only the container backend can start**; N2 alone refuses the namespace
backend there.

### 7.2 Mounts

| Target | Container | Namespace |
|---|---|---|
| `/src` | `local`-driver volume, `type=none`, `o=bind,ro,noexec,nosuid,nodev` | `--ro-bind` |
| `/out` | `local`-driver **overlay** volume, `noexec,nosuid,nodev`; `upperdir` = the `--output` directory, `lowerdir` empty, `workdir` runner-owned scratch beside it on the same filesystem, removed after the run | `--bind` |
| `/scripts` | for `frontend = joern` only: a runner-owned directory holding `<name>.sc`, as a `local`-driver volume `o=bind,ro,noexec,nosuid,nodev` | `--ro-bind` |
| `/tmp` | `--tmpfs /tmp:rw,exec,nosuid,nodev`, **`exec` written out** (Docker's default is `noexec`) | `--tmpfs` |
| `/home/ccg` | `--tmpfs /home/ccg:rw,noexec,nosuid,nodev,uid=<uid>` | `--tmpfs` |
| Toolchain | in the image | `/usr` bound **entry by entry around every host submount**, so `/usr/lib/wsl` is absent, not masked; the Joern bind at `/opt/joern/joern-cli`; `/lib*` and `/bin`; `--proc /proc`; `--dev /dev`; a minimal generated `/etc` (`alternatives`, `java-21-openjdk`, `ld.so.cache`, `hosts`, `passwd`, `group`, `nsswitch.conf` with `files`); then `--remount-ro /` |

`/tmp` is exec-capable because a `noexec` `/tmp` breaks zstd-jni and astgen extraction and the forked
JVM of `joern --script`, and the narrow alternative buys almost nothing (ES1 §7.1.2). Every other
writable mount is `noexec`.

### 7.3 Escape axes, as testable properties

The escape-axis probe is launched through the runner's **launch function**, which builds
`launcherArgv` exactly as for a tool; a check pins that a production record's `launcherArgv` equals
the probe launch's, modulo `auditId`, paths, the `/scripts` mount and the tool tail. A backend that
refuses is tested by asserting the refusal and its preconditions instead. Every connect check is
local, and **a timeout counts as a failure**.

| Axis | The test passes iff… |
|---|---|
| **A1 IP** | (a) `/proc/net/dev` lists only `lo`; (b) no default route: no `0.0.0.0/0` entry in `/proc/net/route` and no `::/0` entry in `/proc/net/ipv6_route` without the `RTF_REJECT` flag (`0x0200`); (c) the sandbox's `/proc/self/ns/net` inode differs from the host's; (d) to `1.1.1.1:443`, `[2606:4700:4700::1111]:443`, each host nameserver on 53, and each host non-loopback address and the Docker bridge gateway on every `local` port: TCP connect and unconnected UDP `sendto` each fail within 2 s with `ENETUNREACH`, `EHOSTUNREACH` or `EADDRNOTAVAIL`; (e) to `127.0.0.1` and `::1` on every `local` port (so LM Studio's 1234 is unreachable): TCP connect fails with `ECONNREFUSED`, `ENETUNREACH` or `EADDRNOTAVAIL`; UDP `connect`, `send` one byte, `recv` with a 2 s timeout passes on `ECONNREFUSED`, or `ENETUNREACH`/`EADDRNOTAVAIL` from `connect`/`send`, and fails on a datagram or a timeout. |
| **A2 DNS** | `getent ahosts example.com` exits non-zero within 5 s; `getent hosts localhost` succeeds. |
| **A3 Host Unix sockets** | no socket file outside `/proc` and `/sys`; `/proc/net/unix` holds only its header (no abstract host socket); the named host sockets (`docker.sock` in `/run` and `/var/run`, `/run/WSL/*_interop`, `/tmp/.X11-unix/X0`, `/mnt/wslg/PulseServer`, `/run/user/<uid>/bus`, `/run/dbus/system_bus_socket`, `/run/systemd/resolve/io.systemd.Resolve`) are absent. |
| **A4 WSL interop exec** | a PE staged in `/src`, `/out`, `/tmp` and `/home/ccg` fails to execute (non-zero, no stdout, `ENOENT`, `EACCES` or `ENOEXEC`; in `/tmp` exactly `ENOENT` when the host has a PE binfmt entry, `ENOEXEC` when not); the same PE executed from a `memfd_create` file fails with `ENOENT`; `WSL_INTEROP` is unset, `/run/WSL` absent, the PE entry's interpreter path absent inside and not creatable. |
| **A5 Host and Windows paths** | every mount point in `/proc/self/mountinfo` is in the backend's allowed set (§7.2, plus Docker's `/etc/{hosts,hostname,resolv.conf}`, `/proc`, `/dev`, `/sys` and below); no mount of fstype `9p`, `drvfs` or `virtiofs`, and **no mount point of any fstype** under `/mnt`, `/usr/lib/wsl` or `/run/WSL`, nor `/init`; sentinel files the runner creates in a fresh host temp dir and in the host `$HOME` are invisible; `/proc` lists no host process. |
| **A6 Environment** | (a) the `env -i` target's own environment keys equal `sandbox.envKeys`; (b) `envKeys` equals the mode's pass-list (§7.6, with `PYTHONSAFEPATH` for indexer entries) with no forbidden name; (b′) every pass-list value equals §7.6's (literal values literally; `PATH`, `JAVA_HOME`, `GOROOT`, `DOTNET_ROOT` equal the image's `Config.Env` on the container backend); (c) `JAVA_OPTS` full-matches the mode's pattern; (d) image `ENV` keys off the pass-list, and `HOSTNAME`, are absent; (e) a child started by `/bin/bash -c` and by `/bin/sh -c` (including a program the shell starts, such as `cat`) sees keys ⊆ `envKeys` ∪ {`PWD`, `SHLVL`, `_`}. |
| **A7 `AF_VSOCK`** | `socket(AF_VSOCK, SOCK_STREAM, 0)` fails with `EPERM`, `EAFNOSUPPORT` or `EACCES`. `preconditions.vsock` is `blocked` on the container (C2 passed), and `present`/`absent` on the namespace backend from the host's `/dev/vsock`. |

**The launcher clause** (test 5, on `launcherArgv`, container). It must contain `--hostname localhost`,
`--read-only`, `--cap-drop ALL`, `--security-opt no-new-privileges`, `--entrypoint /usr/bin/env` with
`-i` as the first image argument, exactly one `--network` and it `none`, and `--workdir /tmp` when
`frontend = joern`. It must contain none of: `--privileged`; `--pid`, `--ipc`, `--uts` or `--userns`
set to `host`; `--cap-add`; `--device`; a `--security-opt` with `seccomp=` or `apparmor=unconfined`;
`-e`, `--env`, `--env-file`; `--add-host`; `--dns`; `--publish`/`-p`; any `-v`/`--mount` other than
§7.2's `/src`, `/out` and (for `joern`) `/scripts`. If a later daemon's built-in profile admits
`AF_VSOCK`, the only admissible `seccomp=` value is a runner-supplied copy of the default that also
blocks family 40 (ES1 §11 R4); that is a revision of this spec.

### 7.4 Frontends and flags

- **The closed frontend set** (`SBX_FRONTENDS`; this list is the contract, and a sandbox test asserts the
  runner's array equals it, in order): `pysrc2cpg`, `jssrc2cpg`, `javasrc2cpg`, `jimple2cpg`, `kotlin2cpg`,
  `csharpsrc2cpg`, `gosrc2cpg`, `c2cpg`, `joern-parse`, `joern` in script mode, `scip-typescript` and
  `scip-python`. Anything else is `unknown-frontend`. The ten Joern names map to `$JOERN_HOME` executables
  (`c2cpg` → `c2cpg.sh`, `jssrc2cpg` → `jssrc2cpg.sh`, the others to the file of the same name); the two
  indexer names map to fixed absolute paths of the executables the `indexer` image installs under
  `/opt/indexer` (the runner does not resolve them through `PATH`; their `#!/usr/bin/env node` line then finds the
  interpreter on the image's `PATH`).
- **Indexer entries** (`scip-typescript`, `scip-python`), `offline` only, container backend only. With
  `controlled` they are `unknown-frontend`; on the namespace backend, `backend-unavailable`. The caller selects
  the image that carries them through `CCG_TOOLCHAIN_IMAGE` (the `indexer` image of
  `cpg/infra/toolchain/README.md`); the runner refuses an indexer entry (`backend-unavailable`, exit 69) when that
  image lacks the label `ccg-toolchain-indexer=1`, so a wrong image is a recorded refusal, not a tool error.
  `toolArgv` must full-match `index` followed by these elements, in any order, each at most once, each option and
  its value as two separate elements (an `--opt=value` form is `forbidden-flag`):
  - `--cwd <p>`, **mandatory** for both tools (both default to the image's `/work`, which would index nothing and
    look like success);
  - `--environment <p>`, `scip-python` only and **mandatory** there, because without it `scip-python` runs `pip`
    to discover packages;
  - `--project-name <n>`, `scip-python` only (`scip-typescript` has no such option), `<n>` matching
    `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`;
  - `<p>` is `/src` or `/src/` followed by components none of which is empty, `.` or `..`.

  A missing mandatory element is `usage`; any other element, including an option the tool does not take, is
  `forbidden-flag`. The runner owns the output: it appends `--output /out/<auditId>.scip` (§7.8) and refuses a
  caller-supplied `-o`/`--output` (§7.5 (4)). The classification rule below extends to both tools' `index --help`,
  as a drift alarm for network-shaped flags only; for indexers the closed grammar, not the sweep, is the control. An
  indexer is **never** launched through `--frontend joern`: script mode bounds a script's arguments, not what the
  script executes, so it cannot enforce an indexer's argv. **An indexer run has its own staged source directory**,
  which is the `/src` mount, and a parse frontend never sees provisioned dependencies. Provisioned dependencies
  (`node_modules`, a venv, the `env.json` of `--environment`) are placed in it by the caller before the call and
  arrive through the existing read-only `noexec` `/src` mount (§7.2 is unchanged; no runner option or mount is
  added). Before the call the stager (P1-15) rejects, anywhere in that directory, a symlink that points outside it,
  a hardlink, and a device or FIFO file; provisioning installs dependencies only (never the target's own build
  backend: `--no-deps` for the target, `--only-binary :all:` or `--no-build` where the corpus allows,
  `npm ci --ignore-scripts`). **Provisioning is an operator step outside the system**,
  like the operator's own `git fetch` (option C): an indexer runs only if the operator already provisioned the
  dependencies, otherwise Joern runs alone, and no `cpg/` module runs `npm ci`, `pip` or `uv` (I2: the broker is the
  only component of ours that opens a connection for egress). No egress path is added in P1. Indexer runs are for the
  lab's trusted, pinned sources only (R-E9). Nothing in the
  runner limits an indexer container's memory, CPU or wall-clock time (R-E10).
- **`joern-parse --language`:** exactly one, `--language X` or `--language=X`, compared
  case-insensitively, from: `pythonsrc` → `pysrc2cpg`; `jssrc`, `javascript` → `jssrc2cpg`; `javasrc`
  → `javasrc2cpg`; `kotlin` → `kotlin2cpg`; `csharpsrc`, `csharp` → `csharpsrc2cpg`; `golang` →
  `gosrc2cpg`; `newc`, `c` → `c2cpg`. Absent, repeated or other (`python` and `java` included) is
  `unknown-frontend`. The table assumes the commercial `java2cpg.sh`, `csharp2cpg.sh`, `go2cpg.sh`,
  `py2cpg.sh` and `js2cpg.sh` are absent; test 9 asserts it.
- **`joern` script mode, `offline` only** (with `controlled`: `unknown-frontend`). `toolArgv` must
  full-match `--script /scripts/<name>.sc` followed by zero or more `--param <k>=<v>`, `<name>` the
  basename of `--script-file`, `<k>` matching `^[A-Za-z][A-Za-z0-9_]*$`; any other element is
  `forbidden-flag`. The runner stages a copy of `--script-file` in a runner-owned directory, checks
  and hashes **that copy** (`sandbox.script`), and binds it at `/scripts` (§7.2); a script never comes
  from `/src`.
- **The `//>` directive refusal.** Split the staged script at every `\n`, `\r` and `\r\n` (the runner
  also splits at `\f` and `\v`, which fails closed). Refuse (`forbidden-directive`, exit 64, nothing
  launched) if any line, after removing every leading character ≤ U+0020 and every leading U+007F,
  starts with `//>`, in any case. The oracle is replpp's `line.trim()`.
- **Classification.** Every flag of a launchable frontend whose name or `--help` description matches,
  case-insensitively,
  `fetch|download|url|https?:|dependenc|resolver|gradle|maven|nuget|proxy|registry|remote|jar4`
  appears in exactly one of `cpg/sandbox/forbidden-flags.txt`
  (`<frontend>\t<flag>\t<forbiddenIn>\t<reason>`, `forbiddenIn` ∈ `offline`, `all`) and
  `cpg/sandbox/allowed-flags.txt` (`<frontend>\t<flag>\t<reason>`, the reason stating the evidence).
  A flag is forbidden unless there is evidence that it opens no network path. **The files are the
  normative lists** (13 forbidden, 3 allowed, for Joern v4.0.579); an addition to
  `allowed-flags.txt` passes an `analyst` gate. The sweep (test 6) runs every frontend's `--help` in
  the sandbox and fails on an unclassified hit, a stale entry or a double entry, at every Joern
  upgrade.
- **Launcher-level options** (consumed by the sbt-native-packager launcher, invisible to `--help`):
  any element that starts with `-D`, `-J`, `-agentlib`, `-agentpath`, `-javaagent` or `-XX`, or equals
  `-jvm-debug`, `-main` or `-java-home`, is refused, unless it full-matches `^-J-Xmx[0-9]+[mg]$` or
  `^-J-Xss[0-9]+[km]$`. This list is fixed in the runner, not in the flag files.

### 7.5 Argv audit, refusals and precedence

Before launch, for every element of `toolArgv`, including those after `--` and `--frontend-args`:
1. the frontend is in §7.4's set (for `joern-parse`, the derived frontend's entries and
   `joern-parse`'s own both apply; for `joern`, the grammar and directive checks of §7.4);
2. an element hits a forbidden entry applicable in this mode (for this frontend or `*`) if it equals
   the flag, starts with `flag=`, or contains the flag as a substring, case-insensitively;
3. launcher-level options are matched case-sensitively by prefix on every element that starts with a
   single `-` followed by a non-`-` character;
4. for a parse frontend or an indexer, a caller-supplied `-o`/`--output` is refused: the runner owns `--output`
   (§7.8);
5. any hit refuses with exit 64, `forbidden-flag` (or `forbidden-directive`), the hits in
   `flagCheck.forbiddenHits`, and no container or namespace created; otherwise `allowedHits` records
   the elements matching `allowed-flags.txt`, and nothing is ever lifted in P1.

**`usage`** (exit 64) covers a missing, unknown or malformed runner option, a missing `--`, a bare
invocation, a missing mandatory indexer element (§7.4), a path the mount syntax cannot express (`,` `"` `:` `=` or a newline), equal `--source`
and `--output`, a malformed flag list, `--script-file` without `--frontend joern`, and an `--output`
in use by a live invocation. **Precedence:** backend resolution is read-only and may run first, but an
invocation refused both with 64 and with 69 is recorded as 64. Each refusal writes exactly one record.
`flagCheck.listDigest` ties a record to the list version, so AC-8 re-checks a historical record
against the lists current at that digest.

### 7.6 The tool's environment (pass-list, `offline`)

| Name | Value |
|---|---|
| `PATH` | container: the image's `PATH` verbatim; namespace: `/opt/joern/joern-cli:<host JDK>/bin:/usr/bin:/bin` |
| `JAVA_HOME` | the image's (`/opt/java/openjdk`); namespace: the host JDK 21 |
| `HOME` | `/home/ccg` |
| `TMPDIR` | `/tmp` |
| `LANG`, `LC_ALL` | `C.UTF-8` |
| `JAVA_OPTS` | full-matches `^-Xmx[0-9]+[mg]( -Xss[0-9]+[km])?$` (the runner passes `-Xmx2g`) |
| `GOTOOLCHAIN` | `local` (safety-positive: `auto` would try a toolchain download) |
| `GOROOT`, `DOTNET_ROOT` | the image's (`/opt/go`, `/opt/dotnet`) |
| `DOTNET_CLI_TELEMETRY_OPTOUT`, `DOTNET_NOLOGO`, `DOTNET_SKIP_FIRST_TIME_EXPERIENCE` | `1` (safety-positive) |
| `PYTHONSAFEPATH` | **indexer entries only:** `1` (safety-positive: a guard against a future tool version; `scip-python` runs `python3 -c` with the staged tree as its working directory) |

Not passed: `JOERN_HOME`, `JAVA_VERSION`, `LANGUAGE`, `MAVEN_HOME`, `GRADLE_HOME`, `HOSTNAME`, `PWD`,
`SHLVL`, `_`. **Forbidden names**, which must never be on the pass-list: `ENABLE_DEPENDENCY_RESOLVER_V2`,
`JAVASRC_FETCH_DEPENDENCIES`, `JAVASRC_JDK_PATH`, `JAVA_TOOL_OPTIONS`, `_JAVA_OPTIONS`,
`JDK_JAVA_OPTIONS`, `*_PROXY`, `NO_PROXY`, `GOPROXY`, `GOFLAGS`, `GOPATH`, `NUGET_*`, `WSL_INTEROP`,
`WSLENV`, `CCG_RUN_TOKEN` (§2.1), and for the indexer entries `NODE_OPTIONS`, `NODE_PATH`,
`NODE_EXTRA_CA_CERTS`, `NODE_PRESERVE_SYMLINKS`, `npm_config_*`, `NPM_CONFIG_*`, `PYTHONPATH`, `PYTHONHOME`,
`PYTHONSTARTUP`, `PYTHONUSERBASE`, `PIP_*`, `UV_*`, `LD_PRELOAD` and `BASH_ENV` (illustrative). The list is documentary, because the pass-list is closed:
`CCG_RUN_TOKEN` is named so a reviewer sees it cannot reach a tool, and needs no runner or test change
(§7.7 drops it before launch). Test 7 is the text half (pass-list ∩ forbidden = ∅); A6 is the runtime
half. The pass-list's controlled-mode additions are deferred (§6, input 8).

### 7.7 The runner's own environment

Apart from `CCG_AUDIT_DIR` and `CCG_TOOLCHAIN_IMAGE`, nothing in the caller's environment may influence
the runner, its record or its `launcherArgv`. The runner's `#!` line is `#!/bin/bash -p`; before
anything else it re-executes itself under `env -i` unless every variable it was started with is one
of: a fixed `PATH`; `DOCKER_HOST=unix:///var/run/docker.sock`; `DOCKER_CONFIG` set to its own
directory `<home>/.local/state/ccg/docker-config`; `HOME` from the passwd entry of its uid; the two
`CCG_*` names; and `PWD`, `OLDPWD`, `SHLVL` and `_`, which a shell sets and which the runner does not
read. The Docker-config directory is created `0700` if absent and must be a plain directory owned by
the uid, mode `0700`, empty, not under `/tmp`, `/var/tmp` or `/dev/shm`, under no world-writable
ancestor; otherwise exit 69, `backend-precondition`. The contract covers the runner launched through
its `#!` line; `bash offline-run.sh` with a hostile `BASH_ENV` is outside it (ES2 §1.7).

### 7.8 Runner conventions

- For a parse frontend the runner appends `--output /out/<auditId>.cpg.bin`; for an indexer, `--output /out/<auditId>.scip`. In every tool argument
  the token `@AUDIT_ID@` is replaced by the invocation's `auditId`. Together they make the tool's own
  output files carry the `auditId` stem; the bundle file itself is named by the normalizer.
- **One live invocation per `--output`:** `flock -n` on a descriptor of the output directory itself,
  outside the overlay `upperdir`; a second is refused `usage`.
- **Signals.** On INT, TERM or HUP while the sandbox runs, the runner kills the Docker client, removes
  every container and volume labelled with its `auditId` until three checks find none, cleans its
  scratch, writes **no record** and exits `128+n`. A `SIGKILL` cannot be trapped: the container then
  runs to its end under `--rm`.

## 8. AC-8, P1 scenario

AC-8 part 1 (proposal §5.3; P1-31 builds the harness, P1-40 runs it): one scenario, a **full refresh
plus queries** of a mixed fixture system, judged from records. Under option C **the scenario's
network has no route out, and the logging proxy is a detector: the main run passes only with zero
`conn` records**, because no allow in P1 can produce a proxied connection (index §6). There is no
`conn` ↔ `decision` join in P1 (§6, input 9).

### 8.1 Topology, preconditions and controls

| Element | Runs as | Network and mounts |
|---|---|---|
| Broker | host process, production shape (§2.0) | host; `local` entries only can complete (P-e) |
| Refresh orchestration and the runner | host processes: the runner needs the Docker daemon, and a container given the daemon's socket would be an escape | sandbox invocations `--network none` (§7) |
| Answer-path components: `ccg` MCP server and CLI (P1-25), plain-language front (P1-26) | containers, `--user <broker uid>:<broker gid>`, `--cap-drop ALL`, on a rootful engine without userns-remap (under remapping or rootless Docker `SO_PEERCRED` reports another uid, and P-c fails) | the scenario network `ccg-ac8-<scenarioId>` (`--internal`, see P-a); the broker directory bind-mounted **read-only** at a fixed path with `CCG_BROKER_SOCKET` naming the socket in it (a `connect()` on a read-only mount works, an unlink or bind does not: inferred, which P-c's connect verifies); `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` naming the detector |
| Store | the throwaway instance (`infra/ccg-falkordb.sh --throwaway`, P1-48) | attached to the scenario network for the answer path; the host refresh reaches it on its loopback-published port |
| Detector | a container | the scenario network only; it accepts TCP on 80, 443 and 3128, writes one `conn` record `{ts, srcAddr, srcPort, dstAddr, dstPort, bytesOut, bytesIn}` per connection, and closes without forwarding. Its only writable mount is `<auditDir>/detector/`, where it writes `conn.jsonl` and the markers of P-d |

**Preconditions.** A scenario that cannot establish one is a **failed** run, never a skipped one.
- **P-a, no route but the two (the acceptance probe).** From each component container, after the
  broker, the store and the detector are up, and again at the end of the run:
  - no default route, IPv4 or IPv6 (A1(b)'s definition);
  - for every host address — every address of every host interface, IPv4 and IPv6 including
    link-local, from `/usr/bin/ip -j -d address show` on the host, except `127.0.0.0/8` and `::1`
    (inside a container those are its own loopback, A1(e)'s case; `lo`'s other addresses, such as
    mirrored mode's `10.255.255.254`, stay in) — and every listening port from `ss -ltun` on the
    host, taken at that moment (TCP and UDP): a TCP connect and a UDP `sendto` each fail as in A1(d),
    a timeout counting as failure. A link-local address is probed with the scope id of each container
    interface other than `lo`: without one Linux rejects the connect with `EINVAL` before sending
    (kernel `tcp_v6_connect`, not probed), which proves nothing;
  - `1.1.1.1:443` and `[2606:4700:4700::1111]:443` fail the same way;
  - **positive control:** the same probe code reports the store's port and the detector's ports as
    accepting; if it does not, the probe is broken and the run fails.

  `--internal` alone fails this, because containers on an internal network still reach host services
  through the bridge's address (S04 gate M4). P1-31 (`devops`) chooses the mechanism; P-a is the
  acceptance criterion whatever is chosen. Candidates, in the order to verify: `--internal` with
  `-o com.docker.network.bridge.gateway_mode_ipv4=isolated` (and `…_ipv6=isolated` where IPv6 is
  enabled), which Docker documents as assigning no address to the bridge; the bridge option
  `com.docker.network.bridge.inhibit_ipv4`, whose effect on the bridge's IPv6 link-local address is
  unprobed. A `DOCKER-USER` rule is not a candidate: that chain is reached from `FORWARD`, and traffic
  to the host's own addresses is delivered through `INPUT` (inferred). **No fallback in P1:** if
  neither candidate passes, P1-31 escalates for a plan revision (R-E3). Answer containers on
  `--network none` would fail P-a's positive control and, through a Unix-socket store endpoint, P-c,
  and no step builds that store path.
- **P-b, no DNS out.** `getent ahosts example.com` fails in each component container (A2).
- **P-c, one host socket.** The broker's socket is the only host Unix socket visible in a component
  container (A3), a `connect()` to it from the container succeeds, and `AF_VSOCK` fails there (A7).
- **P-d, the window.** The detector writes a start marker before the first `policy` record and a flush
  marker after the last record; a `conn` outside the window fails the run, and so does a missing
  marker.
- **P-e, the allow-list.** Every `outbound` and `hosted` host is under `.invalid` (RFC 6761), so no
  non-local allow can connect. Before the run, the harness calls the broker's resolve function
  (`ccg.broker.policy.resolve`) on every non-local host of the scenario manifest and requires
  `resolve-failed` for each; **positive control:** the same assertion, run against a stub resolver
  that answers one of those names, must flag it. The manifest has at least one `hosted` model entry
  and one `outbound` entry of each kind the controls use; `local` is LM Studio or a stub model on
  loopback. The harness checks the manifest against the `allowListDigest` of every `policy` record.
- **P-f, a mixed system.** At least one local-only and one hosted-default repository
  (`cpg/fixtures/system1`, P1-40).
- **P-g, witnesses.** The main run must produce: at least one `R1` allow with `component =
  answer.interpret` (a free-prose question the deterministic parsers do not resolve); at least one
  with `answer.summarize` when the summary is enabled; one non-refused `sandbox` record per
  repository of the system; and the two override `policy` records of the controls. Without them the
  universal checks of §8.2 run over empty sets, so the run fails.

**Controls inside the main run**, injected by the harness; each must appear as stated or the run
fails: an `R4` request from `access.idp` with an extra body field, and one whose `repoNames` names a
repository outside the system (`R4`, `payload-shape`); a hosted model call from `answer.interpret` in
the mixed system (`R5`); a `git-fetch` request from `build.git` and a `dependency-download` request
from `build.pkgmgr`, in a `refresh` run, to a `local` endpoint with a listed path (`R2`, `R8`,
`not-in-p1:*`); a request with no token, and one with `codebases = []` (`R0`); a forbidden frontend
flag, `kotlin2cpg --jar4import-url` (`sandbox` refused, 64, no container per `docker events`); a
refresh with `--override hosted` on a local-only repository (`P2-OVERRIDE-REFUSED`) and one with
`local` on a hosted-default repository (`P3-OVERRIDE-LOCAL`).

**Detector control scenario**, run separately before the main run, because any `conn` fails a run:
the detector is started, a component container connects to it once, and the harness's verdict
function must fail on the resulting `conn` record; a `conn` planted before the start marker and one
after the flush marker must each be flagged too.

### 8.2 The rows, as P1 computes them

*L* is the set of codebases with `effective = local-only` in a run's `policy` record; *W* is the
scenario window (P-d). Rows keep ES1 §8's numbering. **Computed** means the check can fail in this
scenario; **vacuous** means nothing in P1 can make it fail (no such allow can exist under P-e, or the
value is a runner constant, §5.3), and the report says so; P1-29's unit tests carry the positive cases
of the rules involved.

| # | Assertion (proposal §5.3, requirements AC-8) | P1 status | Check |
|---|---|---|---|
| 8.1 | A local-only codebase makes zero connections outside the allow-list | computed | No `conn` record in *W*. Every allow with `codebases ∩ L ≠ ∅` has `rule ∈ {R1, R4}` (P-g makes the set non-empty). |
| 8.2 | …the only allowed outbound ones are its own git remotes, the vulnerability-database job, allow-listed IdP, access-check or PR services | **vacuous** | No `R2`, `R3` or `R4` allow can exist under option C and P-e; the `git-fetch` control appears as an `R2` deny (counted under the controls). The report lists every allow with `privateAddress = true` (expected: none). |
| 8.3 | …request bodies carry nothing beyond repository names and identity | computed (negative); positive half vacuous | The two `R4` controls are denies with `payload-shape`. Every `R4` allow would need `transport = structured`, `payloadClass = repo-names-and-identity`, `payloadFields ⊆ {repoNames, identity.subject, identity.issuer}`; none can exist under P-e. |
| 8.4 | The vulnerability-database job is separate and never sees repository content | **vacuous** | No `vulndb` run has a P1 caller (§2.4a), and `sandbox.runId` is a runner constant. |
| 8.5 | A local-only build makes no dependency download | computed; proxied half vacuous | The `dependency-download` control is an `R8` deny, and no allow has that purpose. Every non-refused `sandbox` record in *W* has its backend's network-namespace flag in `launcherArgv`: `--network none` for `container`, `--unshare-all` without `--share-net` for `namespace` (§7.1). All records, not only *L*'s, since P1 has no controlled mode; `network` and `netnsDistinct` are constants and prove nothing. |
| 8.6 | Every fetch-capable frontend is invoked with its fetch flags off, from the command lines | computed | For every non-refused `sandbox` record in *W*: §7.5 re-run on `toolArgv`, against the lists at `listDigest`, gives no hit; `forbiddenHits = []`; `launcherArgv` has 8.5's backend flag; `envKeys` has no §7.6 forbidden name. The forbidden-flag control is `refused`, `exitCode = 64`. |
| 8.7 | In a mixed system, a question touching only hosted repositories is still interpreted by a local model | computed | In a run with `systemLocalOnly = true`, every allow with `component = answer.interpret` has `rule = R1` (P-g makes the set non-empty); the hosted control is a deny with `rule = R5`. |
| 8.8 | An answer touching a local-only repository is handled entirely locally | computed | Selected by `runId` alone: for every `answer` run whose `policy.codebases ∩ L ≠ ∅`, every `decision` of that run has `requestMode = local-only` and every allow `rule ∈ {R1, R4}`. |
| 8.9 | In a system containing a local-only repository, every model call goes to a `local` endpoint | computed | Every allow with `purpose = model-call` in a run with `systemLocalOnly = true` has `rule = R1`. |
| 8.10 | A hosted override on a local-only codebase is refused | computed | The control's `policy` record: `override = hosted`, that codebase's row `P2-OVERRIDE-REFUSED` and `effective = local-only`, `overrideRefused = true`. |
| 8.11 | A hosted-default codebase run with a local override stays local | computed | The control's row: `default = hosted`, `effective = local-only`, `P3-OVERRIDE-LOCAL`. |
| 8.12 | A hosted-default build downloads dependencies only through the controlled path | **vacuous** (§6) | — |
| 8.13 | …with no import-name service | **vacuous** (§6); residue computed | The `--jar4import-url` control is refused, and so is every record in *W* whose `toolArgv` contains that flag. (`liftedHits` is a runner constant.) |
| 8.14 | Re-checked as each phase adds outbound traffic | computed | Every record in *W* validates against `egress-audit-v1.schema.json` or `sandbox-record-v1.schema.json` (closed enums). A new outbound path needs a new `Component` or `Purpose`, which revises this spec and re-runs AC-8. |
| 8.15 | No request can opt out of the sticky rule by under-declaring | (i) computed; (ii), (iii) vacuous | (i) No allow in a run with *L* ≠ ∅ has `codebases = []` (checked over the answer runs' allows). (ii) No `R8`/`R9` allow can exist. (iii) A `refresh` run has no allows in P1. |
| 8.16 | (P1) The broker's own connections match its decisions | computed when provisioned; otherwise **unrun** | The harness starts the broker under `strace -f -e trace=connect,sendto,sendmsg`; every non-`AF_UNIX` `connect` in *W* equals the `resolvedAddr` and port of some allow, and no resolver traffic appears before an allow. **Positive control:** the trace parser, run on a trace of a small program that dials an unlisted address, must flag it. `strace` is not installed on this host; provisioning it is `devops`'s call, and an unprovisioned row is reported unrun, not passed. |

**Added for P1:** every allow in *W* has exactly one `close` (an allow without one is reported as an
interrupted exchange and fails the run).

### 8.3 What the scenario cannot see

- **The broker's own non-local upstream**, except through row 8.16. P-e makes every allow-listed
  non-local name unresolvable, so a correct broker connects nowhere non-local.
- **The host refresh orchestration's own network use**, which I2's import-boundary test covers
  (§10.1), not the scenario.
- **`R1` traffic** never reaches the detector, by design: it is permitted and audited by `decision`
  records.
- **UDP and DNS**, which `conn` cannot record: P-a and P-b cover them. The sandbox-side escape probes
  (§7.3) run inside the scenario on the backend it uses.

## 9. Evidence

- **For §7:** ES1 §9 (host probes H1–H15, 2026-10-29), ES2 §1.1 (directive probes c2–c11), and
  `cpg/sandbox/README.md` "Deviations"; the S08 gate is `docs/reviews/connected-code-graph-p0-impl.md`.
- **Read-only checks for this document** (2026-11-09; nothing created, no request sent off the
  machine): the Docker daemon is the native Linux engine (Docker Engine – Community 29.7.2), so a
  Unix socket bind-mounted into a container reaches the host broker; `ip -j -d address show` reports
  the Docker bridge `docker0` with `linkinfo.info_kind = bridge` and its prefix; `/usr/bin/ip` and
  `/etc/ssl/certs/ca-certificates.crt` exist; `strace` is not installed; the runner's record writer
  emits exactly §5.3's field set and writes `network`, `networkName`, `runId` and `liftedHits` as
  literals; its re-exec check admits `PWD`, `OLDPWD`, `SHLVL` and `_` (§7.7).
- **Not probed, and marked where used:** WSL mirrored networking's sharing of Linux `localhost`
  listeners with Windows (§2.0); either P-a candidate on this engine; a `connect()` through a
  read-only bind mount (§8.1); the `DOCKER-USER` chain's reach (§8.1).

## 10. What gets built, and the consequences for other steps

### 10.1 Tests the spec requires

**P1-29 (broker core).** Behaviours (1)–(9) in `docs/plans/connected-code-graph-p1-gates.md` are
satisfied by §2–§5; behaviour (6)'s cases are §3.3's `R0` checks 1, 2, 3 and 5, and behaviour (8)'s
allow-list refusal arrives through P1-03's loader. This spec adds, each as a unit test over audit
records, the frame, or the upstream test server's view:
- (10) every `R0` reason, in order (a mutant that reorders two checks fails), including
  `component-purpose-mismatch` for each row of §2.4a's second table;
- (11) `R2`, `R8`, `R9` deny with their reasons even to a `local` endpoint, with requests that pass
  `R0` (§3.3's note);
- (12) no resolver call for any deny (an injected resolver asserts it is never called); the absolute
  name `host.` is what is resolved; one planted address per §3.4 forbidden form, each giving
  `forbidden-address` (`::ffff:127.0.0.1`, `::127.0.0.1`, `64:ff9b::7f00:1`, `2002:7f00:1::`,
  `0.1.2.3`, `240.0.0.1`, a bridge-subnet address, a default gateway); `resolve-failed` and
  `address-check-unavailable`;
- (13) `R4`'s shape: an extra key, a `repoNames` element outside the system, a repeated element, an
  over-long or non-printable `subject`, and an `issuer` that is not an `idp` entry each give
  `payload-shape`; `R3` with a `POST` falls through to a deny;
- (14) caller junk in each field is recorded `null` with its digest in `rejected`, and the record
  still validates; malformed frames (over-length, duplicate key, `NaN`, depth 65) close the
  connection with no record;
- (15) the audit latch: a writer that fails makes the broker attempt no upstream connection and refuse
  every later call with `audit-unwritable`; a short write latches; a file not ending in `\n` is
  repaired at start; a symlinked, group-readable or hard-linked `egress.jsonl` refuses the start;
- (16) the singleton: a second broker with a different `CCG_AUDIT_DIR` and the default socket exits
  69; a live socket is never unlinked; the directory and ancestor checks; the peer-uid check, through
  an injectable credential reader, since a second uid needs root;
- (17) the closed environment: started with `SSL_CERT_FILE`, `LOCALDOMAIN`, `RES_OPTIONS`,
  `HOSTALIASES`, `PYTHONPATH` and a planted module in the working directory, the broker shows none
  in effect (an injected resolver and TLS context record what they received);
- (18) the upstream test server sees exactly §2.3's request line and headers, with the entry's path
  string; a 3xx is returned, not followed;
- (19) the bounds: an over-size request, a response one byte over the bound on the wire, a gzip
  `Content-Encoding`, and a server that trickles a byte under each per-read timeout until the
  wall-clock deadline;
- (20) the token: absent from every record and from captured stderr after a run; expired after
  `close_run` and after 24 h (injected clock).

The gate's mutation check covers `R5`/`R6` precedence, the scope set, the `R2`/`R8`/`R9`-before-`R1`
placement and the component↔purpose table.

**P1-03.** S1–S5 (§4.3) each refuse a planted manifest; `policy.egress` is mandatory; its enum equals
`$defs/egressMode`. **P1-29** adds S6, through an injected interface reader: a host interface address
is admitted; a public literal, a LAN literal that is no host address, and the default gateways of
this host's reading (`192.168.0.1` and `fe80::1` beside the host address `192.168.0.64`) are refused;
an entry admitted at start whose address the reader then drops gives `forbidden-address` on its next
`R1` request, with no connection attempted.

**P1-25: the one import-boundary test (I2).** P1-25's `cpg/tests/test_no_network.py` widens its scope
to every module under `cpg/ccg/` except `cpg/ccg/broker/`, plus `cpg/extractors/` and `cpg/clients/`,
keeping its named mutants (an import inside a function, `from … import`, `__import__`,
`importlib.import_module`, `subprocess` of `curl`/`wget`). Its per-module allow-list is:
`falkordb`/`redis` for exactly the modules of `cpg/ccg/store/` (P1-19), `cpg/ccg/engine/reader.py`
(P1-21's `CypherReader`) and `cpg/clients/common.py`, the one client module that opens `raw_query`'s
`ccg_raw_<system>` connection (P1-25); `ccg.broker.client` for
`ccg/refresh.py`, `ccg/front/interpret.py` and `ccg/front/summary.py` only. `ccg/sandbox_audit.py`
imports nothing from `ccg.broker`.

**P1-18.** `check_sandbox_ids`' four conditions, each with a planted record set that fails only it.

**P1-48b (sandbox tests for the indexer entries, in `cpg/sandbox/tests/test_sandbox.sh`).** (21) `SBX_FRONTENDS`
equals §7.4's twelve names in order (a mutant that drops an indexer entry fails). (22) The grammar, each case a
launch or a refusal. **Launches:** `index --cwd /src` under `scip-typescript`; `index --cwd /src --environment
/src/env.json --project-name p` under `scip-python`; the same elements in another order. **`usage`:** either tool
without `--cwd`; `scip-python` without `--environment`. **`forbidden-flag`:** a first element `npm`, `pip` or `uv`;
a bare `--version`; `--project-name` or `--environment` on `scip-typescript`; a repeat of each flag; the `--cwd=/src`
and `--project-name=x` forms; the abbreviations `--proj` and `--env`; `--cwd` values `/srcfoo`, `/src//a`,
`/src/./a`, `/src/../x`; a `--project-name` value beginning with `-` or `--`; a caller `--output`. **Image and
mode:** an indexer entry on the default `toolchain` image is `backend-unavailable` (no label); `controlled` is
`unknown-frontend`; the namespace backend is `backend-unavailable`. (23) A launched indexer record's `launcherArgv`
passes the §7.3 launcher clause and equals the probe launch's modulo the tool tail, and the escape-axis probe run
through the same launch function on the `indexer` image passes A1–A7 (the boundary is unchanged). (24) The sweep of
test 6 covers both `index --help`s through the test-only launch path (`lib-launch`, §5.3), since the §7.4 grammar
refuses `--help`; for the indexer sweep the classification regex also matches `pnpm|yarn|npm|pip|workspace|infer`; an
unclassified network- or spawn-shaped flag fails it, and it is an alarm only (the grammar is the control). (25)
`ccg-versions` on the `indexer` image names both tools (already asserted by
`cpg/infra/toolchain/tests/test_toolchain.sh`). (26) **Planted `json.py`:** through `lib-launch` under the indexer pass-list, `python3 -c 'import json'` runs with
working directory `/src`, which holds a marker-writing `json.py`: the marker is absent with the pass-list as specified
and present with the mutant pass-list lacking `PYTHONSAFEPATH`; A6 and test 7 pin the variable's presence. An
end-to-end `scip-python` run is not the oracle: at 0.6.8 its probe strips the working directory from `sys.path` itself.

### 10.2 Consequences for other steps

| Step | Consequence |
|---|---|
| P1-03 | Creates `cpg/ccg/schemas/egress-v1.schema.json` (§4.2, verbatim) and `canonical_endpoint`; validates the `egress` section with S1–S5; the per-repository field is `policy.egress`, mandatory, replacing `policy.localOnly`; `endpointAllowList` is removed. |
| P1-48b | **Needs P1-28 and P1-48.** Owns `cpg/sandbox/offline-run.sh` (the two indexer entries, the image-label check, `PYTHONSAFEPATH`, `--output` for an indexer), `cpg/sandbox/allowed-flags.txt`, `cpg/sandbox/forbidden-flags.txt` and `cpg/sandbox/tests/test_sandbox.sh` (tests (21)–(26)); `analyst` plus `security-expert` advisory on the boundary, and the sandbox suite and its mutants re-run in full. Blocks P1-15. |
| P1-29 | **Needs P1-03** (manifests and the schema come from it). Files beyond its list: `cpg/ccg/broker/client.py` (`BrokerClient.open_run`, `request`, `close_run`) and `cpg/ccg/schemas/egress-audit-v1.schema.json`. Replaces `cpg/pyproject.toml`'s `broker = []` with `jsonschema==4.26.0` and `pyyaml==6.0.3`, the `dev` pins verbatim (manifests load through `load_manifest`, records validate against the audit schema): the decision that line's comment reserves to P1-28, made by the owner of D8's one edit (`devops`), with no new pin. Blocks P1-20 and P1-26 (and P1-31, as before). |
| P1-18, P1-19 | P1-18 creates `cpg/ccg/sandbox_audit.py` and `sandbox-record-v1.schema.json` (§5.4); P1-19 calls the same function. No dependency on P1-29. |
| P1-20 | **Needs P1-29**: `refresh(..., override=None)` opens a `refresh` run only when an override is given (§2.1), fails if the broker is absent then, and names the `policy` record in its report; a test shows no direct-connection fallback. |
| P1-25 | No broker call and no dependency on P1-29; widens `test_no_network.py` (§10.1). |
| P1-26 | One `answer` run per question at its first model call (§2.1); fixes the `local` entry's `paths`; a test shows a missing or latched broker fails the call with no direct connection attempted. |
| P1-31, P1-40 | P1-31 **needs P1-18**: `assertions.py` validates against its `sandbox-record-v1.schema.json` (8.14). §8: the topology, P-a–P-g, the controls, the detector control scenario, the row table; no `conn` ↔ `decision` join in P1. |
| P1-32, P1-37, P1-39, P1-41, P1-42, P1-47 | **Unaffected:** they refresh without an override, which needs no broker (§2.1). A step that adds an override needs a broker loaded with that system's manifest. |

### 10.3 Documentation follow-ups (not build units)

The runner's header comment, `cpg/sandbox/README.md` and `cpg/sandbox/tests/test_sandbox.sh` cite
ES1/ES2 and move to this document at their next edit (no behaviour change).
`docs/plans/connected-code-graph-bundle-spec3.md` §2.2 has been executed against and keeps its
citation of ES1 §5.3; §5.4 here is the check's current home.

## 11. Risks and open questions

- **R-E1: the broker's own upstream is attested by its records, and by row 8.16 only if `strace` is
  provisioned.** P-e removes every reachable non-local name in the scenario, and tests (12) and (18)
  test the dialler. The deferred path's input 11 adds an independent network observer.
- **R-E2: host processes are not network-confined in production P1.** I2 holds for our code by the
  import-boundary test and review; a third-party library that opens its own connection would pass
  it. The AC-8 scenario confines the answer path only.
- **R-E3: P-a's mechanism is documented but unverified on this engine.** If neither candidate passes
  the probe, `devops` escalates for a plan revision (§8.1) rather than relaxing P-a.
- **R-E4: the scenario needs a rootful native engine without userns-remap** (the native engine is
  verified here); Docker Desktop's WSL integration or rootless Docker would need another path to the
  broker.
- **R-E5: `UPSTREAM_TIMEOUT_S = 300` and full buffering** may not suit a slow local model. Changing a
  bound is a revision of this spec, made on a measured figure.
- **R-E6: `identity.subject` is a bounded free string** (≤ 256 printable ASCII characters per `R4`
  request) on the one path out of local-only. No P1 caller uses `R4`; P4's identity adapter should
  have the broker supply the identity itself.
- **R-E7: the audit files have no tamper evidence, and `sandbox.jsonl` has the runner's built
  handling** (created `0644`, symlinks followed, `mkdir -p` with the default mode). Both matter only
  against a same-uid actor, which §2.6 places out of scope; a hash chain was weighed and not adopted
  because it would detect that actor without stopping it, at the cost of a runner change and a chain
  that a crash breaks.
- **R-E8: WSL NAT mode cannot reach a Windows-host `local` endpoint.** There the Windows host is the
  default gateway, which S6 never admits, so the broker refuses to start (exit 64); NAT hosts switch
  to mirrored mode. Admitting a gateway on detected NAT was rejected: a false detection sends
  `repo-derived` bodies in clear text to the LAN router, and the heuristics (kernel release, no
  `loopback0`, gateway equal to `resolv.conf`'s nameserver) are not WSL contracts. Reversal: admit it
  only when `wslinfo --networking-mode` reports `nat` (it reports `mirrored` here), keeping the
  router case as a refused test.
- **R-E9: an indexer's children are not argv-audited, and the staged tree is their working directory.** The closed
  argv keeps `npm`, `pip` and `uv` out of the argv; it does not bound what the indexer runs. At the pinned versions
  (`scip-typescript` 0.6.0, `scip-python` 0.6.8), by trace and by run: `scip-typescript` spawns only for
  `--pnpm-workspaces` and `--yarn*-workspaces`, outside the grammar; `scip-python` with `--environment` spawns
  `python3 -c …` and `git rev-parse HEAD` in the staged tree (`git` opens no connection), and its probe strips the
  working directory from `sys.path`, so a staged `json.py` does **not** execute. `PYTHONSAFEPATH=1` guards against a
  future version that does not. The probe still reads `pyrightconfig.json` and `pyproject.toml` from the staged tree.
  The A1–A7 boundary and the launcher clause limit the rest, which is why indexer runs are for trusted, pinned
  sources only. A change of `truth-npm/` needs a re-trace of `child_process` use, `python3` and `git` included.
  `--environment`'s value is read as given (absolute); only the format of an `env.json` generated from a venv built
  at another host path is unmeasured (P1-15's first measurement).
- **R-E10: no resource limit on an indexer container.** The launcher sets `--pids-limit 4096` and nothing for
  memory, CPU or wall-clock time, for Joern and indexers alike. P1-15's `run.py` runs one indexer at a time (index
  R-4) and bounds the run by a wall-clock timeout that sends TERM to the runner, whose handler removes the container
  (§7.8). A runner-side limit is a launcher-clause revision, taken on a measured need.
- **Open question, carried, not new:** how AC-8 closes without the fetch path (index §10, Q-E) is the
  stakeholder's, through `tico`.
