# Connected code graph — egress broker and offline-build sandbox spec

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (M10) · **Extends:** `docs/plans/connected-code-graph.md` · **Version:** 1.3 · **Reviews:** `docs/reviews/connected-code-graph-p0-impl.md` (§ S04)

*Revision 2026-10-29 (v1.1): S04 gate findings M1–M5, m1–m6 and n1 addressed; dispositions in §12.*

*Revision 2026-10-29 (v1.2): S04 gate Pass 2 (P2-M1, P2-m1–m3, P2-n1, m4 nit) and S02 gate M1 addressed; dispositions in §12.2.*

*Revision 2026-10-29 (v1.3): R10 closed: `joern --script` admitted to the runner in one fixed, offline-only form; dispositions in §12.2.*

## 0. Scope and sources

**What this is.** The contract for P0 deliverable (e), step S04 (unit P0-03b) of
`docs/plans/connected-code-graph-p0.md` (the "P0 plan"). It turns proposal §3.12 and §5.3, FR-9
and AC-8 (`docs/requirements/connected-code-graph.md`) into a design an implementer can build. It
also folds in review findings M1 and m4 (`docs/reviews/connected-code-graph-p0.md`, Pass 1).

**Who encodes it.**
- **S08** (`devops`, P0) builds the sandbox: `cpg/sandbox/offline-run.sh`,
  `forbidden-flags.txt`, `allowed-flags.txt` and `tests/test_sandbox.sh`. §7, §5.3 and §10 are
  what S08 encodes (P0 plan §5.3).
- **P1–P2** build the broker and AC-8's logging proxy. They are out of P0's scope (P0 plan §0).
  §2–§6 are their contract.

**What this does not decide.** The rest of the system manifest (FR-0; only its `egress` section
is specified here, in §4). The bundle header's own field names (S03,
`docs/plans/connected-code-graph-bundle-spec.md`); it carries this spec's `auditId` (§5.3).
Proxy implementation technology (P1).

CPG: considered, not relevant — this is a contract spec for code that does not exist yet
(`cpg/` holds no code), and the facts it rests on come from Joern's `--help` and jar strings, and
from host probes (§9), not from a call graph.

## 1. Terms and invariants

| Term | Meaning |
|---|---|
| **Egress** | Any connection that leaves the process's own host network namespace or reaches another process through a host channel: TCP/UDP to any address other than the sandbox's own loopback, DNS, a host Unix socket, AF_VSOCK, or a Windows process through WSL interop. |
| **Mode** | A codebase's egress mode for one run: `local-only` or `hosted` (FR-9). |
| **Default** | The mode in the codebase's manifest entry (§4). |
| **Override** | A per-run request: `local` or `hosted`. |
| **Sticky local-only** | A `hosted` override never changes a `local-only` default (§3.2). |
| **Local endpoint** | An entry in the allow-list's `local` section: "the machine" (FR-9). |
| **Broker** | The one process that holds a route out (proposal §3.12). Every other component reaches the network only through it. |
| **Sandbox** | The runner that executes Joern frontends, extractors and indexers with no route out (`offline`), or with a route only to the broker (`controlled`). |
| **Escape axis** | One channel by which a sandboxed process could reach beyond the sandbox (§7.2). |

**Invariants.** Every section below serves one of these.
- **I1. Default deny.** An egress attempt that no rule allows is denied and logged.
- **I2. One route.** Only the broker has a network route out. A sandboxed process has none in
  `offline` mode, and only the broker's address in `controlled` mode.
- **I3. Local-only is sticky.** Nothing derived from a local-only codebase reaches an endpoint
  outside `local`, except the named permitted outbound entries (§3.3), whatever the run asks.
- **I4. Everything is recorded.** Every policy resolution, every egress decision and every
  sandbox invocation, including refusals, writes exactly one audit record (§5).
- **I5. Properties are executable.** Every escape axis is stated as a property that a test can
  check by running something (§7.2). A property that can only be checked by reading text is a
  defect in this spec.

## 2. Broker interface

The broker is a local service. Components never open sockets to the outside themselves. There
are two ways to reach it. Structured calls carry the **run token**. Proxied streams are bound by
**network attachment** instead, so no secret sits in a sandboxed build's environment (§2.3).

**2.1 Run credential.** At the start of each run (a refresh, an answer, or a vulnerability-database
job), the orchestrator asks the broker for a credential:

```
open_run(runKind: "refresh" | "answer" | "vulndb", systemId: str,
         codebases: list[repoId] | "*", override: "local" | "hosted" | None)
  -> RunGrant {runId: uuid, token: str, effective: {repoId: mode}, systemLocalOnly: bool}
```

- `open_run` performs the policy resolution of §3.2 and writes one `policy` record (§5.2).
- `codebases: "*"` means every codebase in the system. Interpretation uses it, because the
  touched repositories are not known yet when a question is interpreted (FR-9, V3-Q3).
- **A `refresh` or `answer` run has at least one codebase. A `vulndb` run has none:** `open_run`
  rejects `vulndb` with any codebase, and `refresh`/`answer` with an empty list. So a `vulndb`
  grant can never carry repository codebases, and no other run can obtain `R3` (§3.3).
- The token is bound to `runId`, and the broker scopes every later request to that run's
  `effective` modes. The token is an opaque random 256-bit value, lives only in the broker's
  memory and the run's process environment, and expires at `close_run(runId)` or after 24 h.
- A request with no token, or with an expired or unknown one, is denied by rule `R0` (§3.3).

**2.2 Egress request.** Every outbound attempt declares:

```
EgressRequest {
  runId: uuid               # from the token; never trusted from the body
  component: Component      # §2.4
  purpose: Purpose          # §2.4
  endpoint: "scheme://host:port"   # canonical form, §4.3
  payloadClass: PayloadClass       # §2.4
  codebases: list[repoId]   # the codebases this attempt serves; ⊆ the run's codebases
}
-> Decision { decision: "allow" | "deny", rule: RuleId, auditId: uuid }
```

**2.3 Two transports.**
- **Structured calls** (model calls, IdP, access check, PR add-on, vulnerability-database
  refresh): the component calls `request(EgressRequest, body)`. The broker evaluates the policy,
  serializes and sends the body itself, and returns the response. For the `repo-names-and-identity`
  payload class, `body` must match the closed shape `{repoNames: [str], identity: {subject: str,
  issuer?: str}}`. Any extra field is a denial (`R4`), so AC-8's "nothing beyond repository names
  and identity" holds by construction, not only by observation.
- **Proxied streams** (§6.1's git fetch, §6's controlled dependency download): only a **sandbox
  invocation** opens one.
  - **Binding.** For each such invocation, the runner asks the broker for a **proxy binding**
    `bind_proxy(runId, repoId, purpose ∈ {git-fetch, dependency-download}, auditId,
    liftedEntryIds)`. It creates
    one Docker network `ccg-egress-<auditId>` with `--internal`, and attaches exactly two
    containers to it: the broker and that invocation's container. The broker identifies the
    binding by the receiving network and the peer address, so the build's environment carries
    `HTTPS_PROXY=http://broker:3128` with **no credential**. That also matters for Java: the JDK
    disables Basic authentication for `CONNECT` tunnels by default
    (`jdk.http.auth.tunneling.disabledSchemes`), so a token in the proxy URL would not work
    reliably anyway. That JDK default is from memory of its documentation, and was not probed. The
    binding is released when the invocation ends.
  - **Binding identity (conditions of the network-attachment design; S04 gate P2-m3).**
    - The broker attributes an accepted connection to a binding only when **both** hold: it
      arrived on the broker's own address on `ccg-egress-<auditId>`, and its peer IP is that
      invocation's container IP on that network (recorded at `bind_proxy` time from `docker
      network inspect`). It runs one listener per binding network bound to that address, or one
      listener with that accept-time check. A connection that fails either test gets an `R0` deny
      and is closed.
    - A1c(c) and (f) (§7.2) are what prove that the broker's multi-homing cannot route one
      binding's traffic into another.
  - **Lifecycle.** The runner creates each network with the label `ccg.egress.auditId=<auditId>`
    and removes it when the invocation ends. A runner killed mid-invocation leaves it behind, and
    Docker's default address pools hold only about 30 bridge networks, so:
    - **reaper:** before creating a network, the runner removes every network whose name starts
      `ccg-egress-`, that carries the `ccg.egress.auditId` label, whose `Created` time is more
      than 10 minutes old, and that has no container attached other than the broker (which it
      disconnects first). The age bound keeps it off a concurrent invocation's network between
      creation and attach. It never touches a network that lacks the label.
    - **pool exhaustion:** if network creation still fails, the invocation is refused with
      `backend-unavailable` (exit 69). The runner never reuses another invocation's network.
  - **Derivation.** `codebases = [repoId]`, exactly one per binding, which resolves the multi-codebase
    case of a mixed refresh (S04 gate m4). `component` is `build.git` for `git-fetch`, and
    `build.pkgmgr` or `build.frontend` for `dependency-download` (whichever the invocation's
    `frontend` is). `payloadClass` is `none` for `git-fetch` and `repo-derived` for
    `dependency-download`, because dependency coordinates are a dependency list (FR-9).
  - **Purpose of a request on a `dependency-download` binding.** `liftedEntryIds` is the
    invocation's `flagCheck.liftedHits` entry ids (§7.4 step 5; `[]` unless an
    `importNameServices` entry was lifted). A request whose endpoint equals a lifted entry's
    endpoint derives `purpose = import-name-lookup`; every other request derives
    `dependency-download`. Section exclusivity (§4.3) keeps the two disjoint. A `git-fetch`
    binding always has `liftedEntryIds = []`. This is what makes `R9` reachable now that it
    requires `transport = proxy` (§3.3).
  - **Endpoint.** A `CONNECT host:port` request derives `https://host:port`; `CONNECT` is used only
    for TLS. A plain-HTTP absolute-URI request derives `http://host:port`. The decision uses
    `scheme://host:port` only, because `CONNECT` exposes no path. There is no TLS interception
    (see §6.1 for what that leaves open).
- Either way the broker connects upstream itself, and writes the `decision` and `close` records
  of §5.2.

**2.4 Closed enumerations (v1).**

| Field | Values |
|---|---|
| `Component` | `build.git`, `build.frontend`, `build.indexer`, `build.pkgmgr`, `answer.interpret`, `answer.summarize`, `answer.pr-adapter`, `access.idp`, `access.check`, `vulndb.refresh` |
| `Purpose` | `git-fetch`, `dependency-download`, `model-call`, `idp`, `access-check`, `pr-addon`, `vulndb-refresh`, `import-name-lookup` |
| `PayloadClass` | `none` (no request body beyond protocol framing), `repo-names-and-identity` (closed shape above), `repo-derived` (anything derived from repository content: source, names, paths, messages, traces, dependency lists; FR-9), `vuln-feed-request` (feed protocol only; no repository content) |

A value outside these sets is a denial (`R0`). Adding a value is a spec revision.

**2.4a Components allowed per run kind.** `R0` also denies a component outside its run kind:
`refresh` → `build.*` and `access.*`; `answer` → `answer.*` and `access.*`; `vulndb` →
`vulndb.refresh` only.

**2.5 Errors.** A denial is a normal `Decision`, not an exception. The caller surfaces it as
"refused by egress policy (`<rule>`)". It never retries through another path. A broker that cannot
write its audit record denies the request (I4 before I1: no unrecorded allow).

## 3. Policy evaluation

### 3.1 Inputs
- the system manifest's `egress` section and per-codebase `policy.egress` (§4);
- the run's `override`, if any;
- the request (§2.2).

### 3.2 Mode resolution (once per run, in `open_run`)

For each codebase `c` in the run:

| `default(c)` | override | `effective(c)` | Policy record |
|---|---|---|---|
| `local-only` | none | `local-only` | `rule: P1-DEFAULT` |
| `local-only` | `local` | `local-only` | `P1-DEFAULT` |
| `local-only` | `hosted` | **`local-only`** | **`P2-OVERRIDE-REFUSED`**, `overrideRefused: true` |
| `hosted` | none | `hosted` | `P1-DEFAULT` |
| `hosted` | `local` | `local-only` | `P3-OVERRIDE-LOCAL` |
| `hosted` | `hosted` | `hosted` | `P1-DEFAULT` |

- **A refused override does not abort the run.** The codebase proceeds as `local-only`, and the
  refusal is visible in the `policy` record and in the run's answer or refresh report.
- `systemLocalOnly` is true if **any codebase in the system** has `effective = local-only` for
  this run, counting a `local` override. It is computed over the whole system, not only the run's
  codebases (FR-9: "in a system containing any local-only repository").

### 3.3 Request evaluation (per egress request, first match wins)

Let `M = local-only` if any codebase in the **scope set** has `effective = local-only`, else
`hosted`. The scope set depends on the run kind:
- **`answer`:** the run's codebases. FR-9 is sticky **per answer**, so a call that declares fewer
  codebases than its answer touches cannot drop out of local-only.
- **`refresh`:** `request.codebases`. A proxied stream has exactly one, fixed by its binding
  (§2.3). A structured call declares its own, and that declaration is **not trusted to narrow
  the scope of any allow**: no rule that depends on `M` can allow a structured call in a
  refresh run that contains a local-only codebase. `R8` and `R9` require `transport = proxy`, so
  their `M` comes from a runner-fixed binding. `R7` requires `purpose = model-call`, and any run
  with a local-only codebase has `systemLocalOnly = true` (§3.2), so `R5` denies first. The
  remaining allows (`R1`, `R2`, `R4`) do not read `M` (S04 gate P2-m2).
- **`vulndb`:** empty, so `M = hosted`, but only `R3` can allow anything in a `vulndb` run (§2.4a).

One local-only codebase in the scope set makes the whole request local-only (FR-9 × FR-22).

| # | Rule | Condition | Decision |
|---|---|---|---|
| `R0` | `R0-INVALID` | any of: no valid token or binding; an enum value outside §2.4; a component outside its run kind (§2.4a); `request.codebases` ⊄ the run's codebases; **`request.codebases = []` in a run other than `vulndb`**; non-empty `codebases` in a `vulndb` run; a forbidden resolved address (§4.3) | deny |
| `R1` | `R1-LOCAL-ENDPOINT` | `endpoint` ∈ `local` | allow |
| `R2` | `R2-GIT-OWN-REMOTE` | `purpose = git-fetch`, `component = build.git`, `transport = proxy` through a `git-fetch` binding (§6.1), `codebases = [c]` (exactly one), `endpoint` = canonical(`remote(c)`) | allow |
| `R3` | `R3-VULNDB` | **the run's `runKind = vulndb`** (so its codebases are `[]`), `purpose = vulndb-refresh`, `component = vulndb.refresh`, `payloadClass ∈ {none, vuln-feed-request}`, `endpoint` ∈ `outbound` with `kind: vulndb` | allow |
| `R4` | `R4-NAMED-SERVICE` | `purpose ∈ {idp, access-check, pr-addon}`, `endpoint` ∈ `outbound` with the matching `kind`, `payloadClass = repo-names-and-identity`, and the body matches the closed shape (§2.3) | allow; if the endpoint matches but the payload does not, **deny** |
| `R5` | `R5-SYSTEM-LOCAL-MODEL` | `purpose = model-call` and `systemLocalOnly` | deny (only `R1` could have allowed it) |
| `R6` | `R6-STICKY-LOCAL` | `M = local-only` | deny |
| `R7` | `R7-HOSTED-SERVICE` | `M = hosted`, `purpose = model-call`, `endpoint` ∈ `hosted` | allow |
| `R8` | `R8-CONTROLLED-REGISTRY` | `M = hosted`, `purpose = dependency-download`, **`transport = proxy` through a `dependency-download` binding** (§2.3), `endpoint` ∈ `registries` | allow |
| `R9` | `R9-IMPORT-NAME-SERVICE` | `M = hosted`, `purpose = import-name-lookup`, **`transport = proxy` through a `dependency-download` binding whose `liftedEntryIds` names this endpoint's entry** (§2.3), `endpoint` ∈ `importNameServices` | allow |
| `R99` | `R99-DEFAULT-DENY` | nothing above matched | deny |

Notes on the order:
- `R1` comes first: "the machine" may receive anything, in either mode.
- `R2`–`R4` are the only outbound entries local-only allows (FR-9, V3-Q4, SQ-E). They come before
  `R5`/`R6` so local-only can use them, and they carry their own payload restrictions.
- `R5` sits before `R7`, so a hosted-only question in a system with any local-only repository is
  still denied a hosted model (AC-8).
- `R9` matches only an endpoint the manifest names explicitly. Kotlin's `--jar4import-url` is the
  known case (proposal §3.12), and the sandbox forbids that flag in every mode unless its value
  equals such an entry (§7.3).

## 4. Allow-list format

### 4.1 Location
The `egress` section of the system manifest, plus `remote` and `policy.egress` on each codebase
entry. The rest of the manifest (FR-0) is P1's. YAML, matching the repo's `pack.yaml` convention.
The loader validates it against a JSON Schema (P1: `cpg/ccg/schemas/egress-v1.schema.json`) and
refuses to start the broker on any violation.

### 4.2 Shape

```yaml
egress:
  schemaVersion: 1
  local:                         # "the machine" (FR-9)
    - id: lmstudio
      kind: model                # model | service
      endpoint: http://127.0.0.1:1234
      note: LM Studio on the Windows host, reached through WSL mirrored networking
  outbound:                      # the only entries local-only may use besides git remotes
    - id: osv
      kind: vulndb               # vulndb | idp | access-check | pr-addon
      endpoint: https://osv-vulnerabilities.storage.googleapis.com:443
  hosted:                        # hosted-mode only
    - id: hosted-llm
      kind: model
      endpoint: https://api.example-llm.com:443
  registries:                    # controlled download path; hosted-mode builds only
    - id: maven-central
      ecosystem: maven           # maven | gradle | go | nuget | npm | pypi
      endpoint: https://repo.maven.apache.org:443
      # privateAddress: true     # only if it resolves to RFC 1918 / ULA space (§4.3)
  importNameServices: []         # each entry: id, endpoint, reason (mandatory)
codebases:
  - repoId: graphmind-ai-lab
    remote: https://github.com/<owner>/graphmind-ai-lab.git
    policy:
      egress: local-only         # local-only | hosted
```

### 4.3 Matching and validation rules
- **Canonical endpoint:** `scheme://host:port`. The scheme and host are lowercased, the port is
  always explicit (the default port is filled in), and there is no path, query or user-info.
  `remote` is canonicalized the same way.
- **`remote` must be `https://` in v1.** `ssh://` and scp-style `git@host:path` remotes are
  rejected by the loader, because ssh ignores `HTTPS_PROXY` and has no path through an
  HTTP-`CONNECT` broker. A later revision can add an ssh path, for example through a `ProxyCommand`
  that speaks `CONNECT`.
- **Exact match only.** No wildcards, CIDR ranges or suffix matches in any section. A request
  matches an entry when the canonical endpoints are equal.
- **`local` hosts** must be IP literals or `localhost`. The broker resolves `localhost` from
  `/etc/hosts` only, never through DNS, so a DNS answer cannot turn a remote host into "the
  machine".
- **Hosts in other sections** are resolved by the broker at decision time. The broker connects
  to **the address it validated, and never re-resolves**, so there is no DNS-rebinding window
  between check and connect. `decision.resolvedAddr` records that address.
- **Forbidden resolved addresses** (`R0`) stop a non-`local` entry from reaching local services:
  - loopback, link-local, unspecified and multicast addresses;
  - every address of the host's own interfaces;
  - every Docker network's subnet and gateway (`docker network inspect`, refreshed whenever the
    broker starts a binding);
  - any `local` entry's address;
  - **RFC 1918, RFC 6598 (`100.64.0.0/10`) and ULA (`fc00::/7`) addresses, unless the entry
    declares `privateAddress: true`.**

  The opt-in is a deliberate departure from the S04 gate's recommended blanket ban (§12.1, m4): a
  workplace registry, IdP or access check commonly lives on a private address, and a blanket ban
  would make the controlled path and `R4` unusable there. The opt-in is per entry and visible in
  review, and the other bullets still apply to such an entry: no host, Docker or `local` address
  even when `privateAddress` is set.
- **Section exclusivity:** an endpoint may appear in only one section. `outbound` kinds are
  closed (`vulndb`, `idp`, `access-check`, `pr-addon`). An `importNameServices` entry without a
  non-empty `reason` is rejected.
- **`remote`** is mandatory for every codebase, and it is the only git remote `R2` allows for it.
  A codebase with several remotes lists the one to fetch from; P1 may extend `remote` to a list.

## 5. Audit-record format

### 5.1 Common rules
- **Storage:** append-only JSONL, one record per line, UTF-8. Broker records go to
  `<auditDir>/egress.jsonl`; sandbox records go to `<auditDir>/sandbox.jsonl`. `auditDir`
  defaults to `cpg/.audit/` (gitignored; S08 adds `.audit/` to `cpg/.gitignore`) and can be
  overridden by `CCG_AUDIT_DIR` in the **launcher's** environment. That variable is never passed
  into the sandbox.
- **Atomicity:** each record is one `write(2)` of the full line under an exclusive `flock` on the
  file. A reader skips a trailing partial line.
- **Common fields:** `schemaVersion: 1`, `recordType`, `auditId` (lowercase UUIDv4), and `ts`
  (RFC 3339 UTC with milliseconds).
- **Content:** records hold repository names, paths, argv and endpoints. They are local files and
  never leave the machine; the broker never sends audit records anywhere.

### 5.2 Broker records

**`policy`** — one per `open_run` (§3.2):

| Field | Type | Meaning |
|---|---|---|
| `runId` | uuid | |
| `runKind` | `refresh` \| `answer` \| `vulndb` | |
| `systemId` | str | |
| `codebases` | [repoId] | the run's codebases, expanded from `"*"` |
| `override` | `local` \| `hosted` \| null | as requested |
| `modes` | [{`repoId`, `default`, `effective`, `rule`}] | one per codebase, rule ∈ `P1`–`P3` |
| `overrideRefused` | bool | true if any row is `P2-OVERRIDE-REFUSED` |
| `systemLocalOnly` | bool | §3.2 |

**`decision`** — one per egress request, written **before** any upstream byte is sent.
`{ts, component, purpose, endpoint, decision, rule, codebases, payloadClass, bytes}` is the P0
plan's §4.2 minimum. The full set:

| Field | Type | Meaning |
|---|---|---|
| `runId` | uuid | |
| `component`, `purpose`, `payloadClass` | §2.4 enums | |
| `endpoint` | canonical endpoint | |
| `resolvedAddr` | str \| null | the address the broker connected to; null on deny |
| `privateAddress` | bool | true iff the matched entry declares `privateAddress: true` (§4.3), so AC-8 can list every allow that reached private address space; false on deny and for `local` entries |
| `codebases` | [repoId] | as declared, after the `R0` subset check |
| `requestMode` | `local-only` \| `hosted` | `M` of §3.3 |
| `decision` | `allow` \| `deny` | |
| `rule` | RuleId | §3.3 |
| `transport` | `structured` \| `proxy` | §2.3 |
| `payloadFields` | [str] \| null | for `structured`: the top-level and `identity.*` keys actually sent; null for `proxy` |
| `bytes` | int | request body bytes for `structured`; 0 for `proxy` (see `close`) |
| `upstream` | {`localPort`: int} \| null | the source port of the broker's upstream socket, for the AC-8 join (§8) |

**`close`** — one per allowed proxied connection, when it closes: `{auditId` (its own),
`decisionAuditId`, `bytesOut`, `bytesIn`, `durationMs`, `closeReason}`.

### 5.3 Sandbox record

**`sandbox`** — exactly one per invocation of `cpg/sandbox/offline-run.sh`, refusals included.
The runner allocates `auditId` **before** launch. It passes it to the tool as the stem of the
**tool's own output files**, not of the bundle file, which the normalizer names
`<unitId>.bundle.jsonl` and may write outside the sandbox. The record is written when the
invocation ends, so a killed runner leaves no record.

**Seam with the bundle spec.** One bundle can come from several invocations, for example the
Joern parse plus the `extract.sc` run (P0 plan §4.4). The bundle header therefore carries
`sandboxAuditIds`, an array, as S03 decided (`docs/plans/connected-code-graph-bundle-spec.md`
§2.2, BF15). The P1 loader check is: **every id in `sandboxAuditIds` has a `sandbox` record with
`refused: false`, `exitCode: 0`, and `egressMode` equal to the header's `egressMode`.** Otherwise
the bundle is invalid. The bundle spec owns that check; this spec owns the record it reads.

| Field | Type | Meaning |
|---|---|---|
| `backend` | `container` \| `namespace` | §7.1 |
| `toolchain` | {`image`, `imageId`} \| {`joernVersion`, `joernPath`, `javaVersion`} | container: tag and `sha256:` id; namespace: host Joern and JDK |
| `job` | `parse` \| `git-fetch` | `parse` is a frontend, extractor or indexer run; `git-fetch` is §6.1's fixed-argv fetch (P1) |
| `egressMode` | `offline` \| `controlled` | P0 implements `offline` only; `git-fetch` is always `controlled` |
| `network` | `none` \| `broker` | `none` iff `offline` |
| `networkName` | str \| null | `ccg-egress-<auditId>` when `network = broker`, else null (§6) |
| `launcherArgv` | [str] | the full `docker run …` / `bwrap …` argv, as executed |
| `toolArgv` | [str] | the full argv of the tool inside the sandbox |
| `frontend` | str | from the closed set in §7.3 |
| `script` | {`path`, `sha256`} \| null | for `frontend = joern`: the script's path inside the sandbox and the SHA-256 of the exact bytes bound there (§7.3.1); null otherwise. The bundle header's `extractScriptHash` can be checked against it; the bundle spec owns any such check. |
| `flagCheck` | {`listDigest`, `forbiddenHits`: [str], `allowedHits`: [str], `liftedHits`: [{`element`, `entryId`}]} | `listDigest` = SHA-256 of `forbidden-flags.txt` and `allowed-flags.txt` concatenated; `liftedHits` per §7.4 step 5, always `[]` in `offline` |
| `envKeys` | [str] | the names (never values) of the **environment the tool process receives**: the keys of the `K=V` arguments to `env -i` in `launcherArgv`. They are derived from the argv, not declared separately, and A6 verifies them against the tool's real `/proc/self/environ`. |
| `mounts` | [{`target`, `source`, `mode`}] | every host path visible inside, with `ro`/`rw` and `exec`/`noexec` |
| `preconditions` | {`interopPE`: `enabled`\|`disabled`\|`unreadable`, `binfmtPinned`: bool, `seccomp`: str \| null, `vsock`: `blocked`\|`present`\|`absent`, `netnsDistinct`: bool} | what the runner checked before launch. `interopPE` is §7.1.1's result; `binfmtPinned` is C1; `seccomp` is the profile from `docker info` (container) or null; `vsock` is derived as in A7. |
| `refused` | bool | |
| `refusalReason` | null \| `forbidden-flag` \| `forbidden-directive` \| `unknown-frontend` \| `mode-not-implemented` \| `backend-precondition` \| `backend-unavailable` | `forbidden-directive`: a `joern` script carries a `//>` directive line (§7.3.1) |
| `exitCode` | int | tool exit code; 64 for an argv or directive refusal; 69 for a backend refusal |
| `codebases` | [repoId] | the codebase(s) whose source is mounted |
| `runId` | uuid \| null | the refresh run, when one exists (P1+); null in P0 |
| `durationMs` | int | |

## 6. Controlled-download path

**For hosted-default codebases only** (FR-9, V3-Q5). A local-only build never uses it: the
runner refuses `controlled` mode with `job: parse` for any codebase whose `effective` mode is
`local-only`. The one `controlled` job local-only allows is `git-fetch` (§6.1). In P0
the runner refuses `controlled` for every codebase (`mode-not-implemented`); P2 implements it.

**Mechanism: the frontend fetches, and the network reaches only the broker.**
1. **Broker topology.** The broker's **proxy listener** runs in a container. That container is
   attached to one ordinary (non-internal) bridge network for its upstream traffic, and to each
   active per-invocation network. It publishes no port. A container cannot join `host` together
   with another network (S04 gate M4), so v1.0's "attaches to it and to the host network" was
   wrong. The **structured-call listener** may run on the host, because it must reach `local`
   endpoints such as LM Studio through mirrored-network `localhost`. Both listeners share one
   policy state and one audit log. P1 decides the process split within those constraints.
2. **Per-invocation network.** For each `controlled` invocation, the runner calls
   `bind_proxy(runId, repoId, dependency-download, auditId, liftedEntryIds)` (§2.3) and creates
   `ccg-egress-<auditId>` with `--internal`. Only the broker and this invocation's container
   attach to it. The build container runs with `--network ccg-egress-<auditId>` instead of
   `--network none`.
3. **Environment.** The pass-list adds the controlled-only entries of §7.3.4:
   `HTTPS_PROXY`/`HTTP_PROXY=http://broker:3128`, with **no token**, and `JAVA_OPTS`'s proxy
   properties. The binding is the credential, so repo-driven build code (for example a Gradle
   script running under `--download-dependencies`) finds no secret to reuse elsewhere. Within its
   own invocation it can still reach the same registries, which is what hosted-default allows.
4. **Fetch flags.** The frontend runs **with** its fetch flags (§7.3, `forbiddenIn = offline`).
   Every `CONNECT` is decided by `R8`: registries only.
5. **No other route.** This is a property, not a claim: §7.2's **A1c** must pass in `controlled`
   mode before P2 ships. `--internal` alone does not make it true, because the gateway IP stays
   reachable, and with it any host listener on `0.0.0.0` (S04 gate M4). A tool that ignores the
   proxy settings then fails closed.
6. **Import-name service.** `--jar4import-url` is refused unless §7.4 step 5 lifts it. Then `R9`
   decides each lookup, and `flagCheck.liftedHits` records the lift.
7. **Cache.** The fetched dependencies land in a cache volume keyed by the lock-file digest.
   Later offline builds of the same codebase may mount it read-only. That is how "dependency
   caches provisioned in advance" (proposal §3.12) are produced for a hosted-default codebase.
   For a local-only codebase, caches are provisioned by hand outside this system.

**Rejected alternative:** pre-fetch with each ecosystem's package manager in a separate job, then
always build offline. It would keep the Joern sandbox uniformly offline. But `csharpsrc2cpg
--download-dependencies` talks to the NuGet API itself (the jar carries `NUGET_BASE_API_V2`/`_V3`;
§9), and the Java, Kotlin and Go frontends consume their own fetch output. A pre-fetch would have
to reproduce four frontend-internal resolution schemes, and S25 measures fetch-on runs, not
pre-fetch. Revisit if S25 shows that the frontends accept pre-fetched caches through
`--inference-jar-paths`, `--classpath` and `--external-summary-paths` with the same type
resolution.

### 6.1 The git-fetch path (both modes; P1)

FR-9 lets local-only fetch "each repository from its own git remote". The broker can only decide
`CONNECT https://host:443`, so `R2` admits an **opaque TLS tunnel to the whole host**. Any process
holding that tunnel could push to another repository on the same host, or create a gist or an
issue carrying source, and still be within the rules (S04 gate M5). The mitigations that follow
narrow who holds the tunnel and what passes through it.
- **Only the runner fetches.** `git-fetch` is a `sandbox` invocation with `job: git-fetch`, run by
  the runner. Its argv is fixed:
  `git -c credential.helper= -c protocol.version=2 -c http.proxy=http://broker:3128 fetch
  --no-write-fetch-head --prune <remote(c)> +refs/heads/*:refs/remotes/origin/*`.
  Only `<remote(c)>` varies, and it comes from the manifest. No component, extractor or tool can
  open a `git-fetch` binding (`R2` requires `component = build.git` and a `git-fetch` binding).
- **Read-only credentials only.** The fetch container gets no credential helper, no SSH agent,
  no `~/.gitconfig` and no token, beyond what a public or read-only remote needs. For a private
  remote, P1 supplies a **read-only deploy credential**, scoped to that one repository, through a
  file mounted only for this invocation. No push-capable credential exists anywhere in the
  sandbox.
- **The repo's own hooks and config are not consulted.** `fetch` runs no hooks. The runner
  applies `-c core.hooksPath=/dev/null`, and runs against a bare mirror owned by the runner, not
  against the working checkout.
- **Upload volume is bounded and audited.** A fetch sends only negotiation, so very little goes
  out. Every `R2` connection's `close.bytesOut` must be ≤ `egress.gitFetchMaxBytesOut` (manifest,
  default 1 MiB; P1 calibrates it on the proving repo). An outlier is flagged, and fails AC-8
  (§8, 8.2).
- **Residual risk, stated plainly.** Even with all of the above, the tunnel is host-granular, and
  the broker has no visibility of paths. A compromised `git` binary in the image, or a git
  protocol feature that sends data, could reach any repository on the remote's host within the
  byte bound. Closing that requires TLS interception for `R2` only, where the broker terminates
  TLS, checks the path is `<repo>/info/refs` or `<repo>/git-upload-pack`, and re-encrypts. That
  adds a broker-held CA, which the fetch container must trust.
  **This is an open stakeholder decision for P1, not decided here (§11 Q1):** accept the residual
  risk with the mitigations, or require TLS interception for `R2`.

## 7. Sandbox contract

**Interface (S08 implements it):**

```
cpg/sandbox/offline-run.sh --backend container|namespace|auto --mode offline|controlled \
    --frontend <name> --source <dir> --output <dir> [--codebase <repoId>]... \
    [--script-file <host path>] -- <tool args...>
```

`--script-file` is accepted only with `--frontend joern`, and is then mandatory (§7.3.1).

- `auto` picks `container` when Docker and the image are present and §7.1's preconditions hold.
  Otherwise it tries `namespace`. If neither can start, it exits 69 with one `sandbox` record.
- The runner takes **no input from its own environment** except `CCG_AUDIT_DIR` and
  `CCG_TOOLCHAIN_IMAGE`. It reads no configuration from the source tree.
- `--mode controlled` exits 64 with `refusalReason: mode-not-implemented` in P0 (§6).
- **Guarantees to the tool:** the source is readable at `/src`; `/out` is writable; `/tmp` and
  `$HOME` (`/home/ccg`) are writable scratch space, and `/tmp` (= `$TMPDIR` = `java.io.tmpdir`)
  is the one writable location that allows executing what is written there (§7.1.2); `localhost` resolves and is also the sandbox's
  hostname (S08 test 3); the toolchain is at the image's paths (S02, `cpg/infra/toolchain/Dockerfile`
  stage `toolchain`); the environment is exactly §7.3.4's pass-list. Nothing else is promised.
- **The tool is always started through `env -i`** (both backends). The container runs with
  `--entrypoint /usr/bin/env`, and the image arguments are `-i K1=V1 … Kn=Vn <tool> <toolArgv…>`.
  Docker cannot drop an image's `ENV` keys and always adds `HOSTNAME` (S04 gate M1, verified), but
  `env -i` replaces the environment before it `exec`s the tool, so the tool receives exactly the
  listed pairs. The namespace backend gets the same through `--clearenv` plus `--setenv`, and the
  runner applies the same `env -i` wrapper so that both backends share one code path.

### 7.1 Backends

| | **Container** (primary) | **Namespace** (fallback, hosts without Docker) |
|---|---|---|
| Launch | `docker run --rm --network none --hostname localhost --read-only --cap-drop ALL --security-opt no-new-privileges --user <uid>:<gid> --pids-limit 4096 --entrypoint /usr/bin/env` on `ccg-toolchain:<hash>` (S02), with the daemon's built-in seccomp profile | `bwrap --unshare-all --die-with-parent --new-session --clearenv --hostname localhost`, with the host Joern and JDK |
| Source `/src` | bind, `ro`, `noexec` | `--ro-bind` |
| Output `/out` | bind, `rw`, `noexec` | `--bind` |
| `/tmp` | `--tmpfs /tmp:rw,exec,nosuid,nodev`. **`exec` is written explicitly**: Docker's `--tmpfs` defaults to `noexec` (S02 gate M1, verified). See §7.1.2 for why `/tmp` is exec. | `--tmpfs` |
| `/home/ccg` | `--tmpfs /home/ccg:rw,noexec,nosuid,nodev,uid=<uid>` | `--tmpfs` |
| Toolchain | in the image | `--ro-bind /usr /usr` with `--tmpfs /usr/lib/wsl` over it, `--ro-bind <joern-cli> /opt/joern/joern-cli` (the image's path), `/lib*`, `/bin` symlinks, `--proc /proc`, `--dev /dev`, and a minimal `/etc` (`alternatives`, `java-21-openjdk`, `ld.so.cache`, a generated `hosts` and `passwd`) |
| Never present | `/mnt/*`, `/run/WSL`, `/init`, any `docker.sock`, `/usr/lib/wsl` | the same |
| **Refuses (exit 69, `backend-precondition`) when** | **C1:** an enabled `binfmt_misc` entry with magic `4d5a` (PE) has flag `F`, which pins the interpreter across mount namespaces; **C2:** `docker info`'s security options do not include `name=seccomp,profile=builtin`. (A runner-supplied profile is the only alternative, and §11 R4 governs it. A daemon-level `seccomp-profile` change would otherwise go unseen on every invocation.) | **N1:** §7.1.1. **N2:** `/dev/vsock` exists on the host. |
| Unavailable (exit 69, `backend-unavailable`) when | no Docker daemon, or no image with the expected tag | no `bwrap`, or unprivileged user namespaces are disabled |

**7.1.1 N1, interop detection. This controls over the P0 plan's literal path.** The P0 plan's
§2 D6 and §4.1 test 1, and review M1, name `/proc/sys/fs/binfmt_misc/WSLInterop`. On this host
that file **does not exist**; the entry is `WSLInterop-late` (§9 H1). A check of the literal path
would conclude "interop disabled" and let the namespace backend start, which is exactly the M1
escape. `teco` routes this correction through S08's brief (P0 plan §5.3, S04→S08); the P0 plan
is not edited.

The check is a function `interop_state(binfmt_dir, kernel_release) → enabled | disabled |
unreadable`, with `binfmt_dir` defaulting to `/proc/sys/fs/binfmt_misc`.
1. **Readable** means `binfmt_dir/status` exists. If it is missing (no directory, an empty
   directory, or `binfmt_misc` not mounted in the runner's mount namespace, while entries may still
   exist kernel-wide on 5.15), the result is `unreadable`.
2. If `status` reads `disabled`, the result is `disabled`: the kernel ignores every entry.
3. Otherwise it reads every entry file other than `status` and `register`. The result is
   `enabled` if any entry whose first line is `enabled` matches any of:
   - (a) a `magic` line starting `4d5a` (PE, `MZ`), at any offset;
   - (b) an `extension` line equal to `exe`, case-insensitively, for an extension-type entry (`E`);
   - (c) an `interpreter` line equal to `/init`;
   - (d) a file name matching `WSLInterop*`.
4. Otherwise the result is `disabled`.

The namespace backend **refuses** when the result is `enabled`. It also refuses when the result
is `unreadable` and `kernel_release` contains `microsoft`, compared case-insensitively (WSL2 is
`microsoft-standard-WSL2`, WSL1 `Microsoft`). The runner records the result
as-is in `preconditions.interopPE` (§5.3). C1 uses the same scan and refuses if any entry matched by (a)–(d) has `F` in its
`flags:` line.

**Fixtures for S08 test 8, with the expected result of each.** In every row, `/dev/vsock` is
taken as absent, so that the row isolates N1. The real host is a separate row.

| # | Fixture `binfmt_dir` contents | `kernel_release` | `interop_state` | Namespace backend | C1 (container) |
|---|---|---|---|---|---|
| F1 | `status`=`enabled`; `WSLInterop-late` = this host's (`enabled`, `interpreter /init`, `flags: P`, `magic 4d5a`) | `5.15.167.4-microsoft-standard-WSL2` | `enabled` | refuse, 69 | pass |
| F2 | `status`=`enabled`; `WSLInterop` with the same body | same | `enabled` | refuse | pass |
| F3 | `status`=`enabled`; `winexe`: `enabled`, `interpreter /usr/bin/wine`, `flags: `, `magic 4d5a` | `6.8.0-generic` | `enabled` | refuse | pass |
| F4 | `status`=`enabled`; `pe`: `enabled`, `interpreter /opt/x/run`, `flags: `, `extension .exe` | `6.8.0-generic` | `enabled` | refuse | pass |
| F5 | `status`=`enabled`; `x`: `enabled`, `interpreter /init`, `flags: `, `magic 7f454c46` | `6.8.0-generic` | `enabled` (rule c) | refuse | pass |
| F6 | F1 with `flags: PF` | WSL2 | `enabled` | refuse | **refuse, 69** |
| F7 | F1 with the entry's first line `disabled` | WSL2 | `disabled` | start | pass |
| F8 | F1 with `status`=`disabled` | WSL2 | `disabled` | start | pass |
| F9 | `status`=`enabled`; `python3.12` only (`enabled`, `interpreter /usr/bin/python3.12`, `flags: `, a bytecode magic that is not `4d5a`) | `6.8.0-generic` | `disabled` | start | pass |
| F10 | empty directory | WSL2 | `unreadable` | refuse | pass |
| F11 | directory missing | `4.4.0-19041-Microsoft` (WSL1) | `unreadable` | refuse | pass |
| F12 | directory missing | `6.8.0-generic` | `unreadable` | start | pass |
| Host | the real `/proc/sys/fs/binfmt_misc`, `/dev/vsock` and kernel release | — | `enabled` | refuse, with `preconditions.interopPE = enabled` and `vsock = present` | pass |

"Start" means only that N1 does not refuse. The fixture test calls the function, or the runner
with an injected `binfmt_dir` through a test-only entry point that the production argv parser does
not expose, and asserts the result. It does not launch a sandbox.
**7.1.2 Notes.**
- **On this host only the container backend can run.** The namespace backend refuses on both
  N1 and N2. In practice N2 alone already refuses on every WSL2 host. This is the outcome the P0
  plan accepted (§2 D6).
- **`--hostname localhost` is required** (S02, `cpg/infra/toolchain/README.md`). Under
  `--network none`, Docker does not write the generated hostname into `/etc/hosts`. The JVM's
  `InetAddress.getLocalHost()` then fails immediately, with a log4j `UnknownHostException`. The
  parse still succeeds, but every run carries a misleading stack trace and a resolver attempt.
  With the hostname set to `localhost`, the existing `/etc/hosts` line answers it. It costs
  nothing and adds no channel.
- **`/tmp` is exec-capable; every other writable mount is `noexec`.** This settles S04 gate P2-M1
  and S02 gate M1. All probes are §9 H12–H14, on the S02 `toolchain` stage image
  (`ccg.joern.version=4.0.579`).
  - **What breaks under a `noexec` `/tmp`.** Two extraction routes, and one way a redirect fails
    to reach. zstd-jni (every frontend's CPG write, and
    `joern` loading a CPG) maps its `.so` from `java.io.tmpdir`. jssrc2cpg's `astgen-linux` is a
    gzexe wrapper that unpacks into `$TMPDIR` and runs from there. And `joern --script` forks a bare `java` child
    (`replpp.scripting.NonForkingScriptRunner`) that receives none of `JAVA_OPTS`'s properties.
  - **Why not the narrow shape** (a `noexec` `/tmp` plus an exec `/tmp/native`, with
    `TMPDIR=/tmp/native` and `JAVA_OPTS=… -Djava.io.tmpdir=/tmp/native`, as S04 gate P2-M1
    proposed). It passes `joern-parse` for all seven FR-1 languages, but **`joern --script` still
    fails** on zstd-jni in `/tmp`, because the forked JVM ignores `JAVA_OPTS`. S11 runs `extract.sc`
    that way (P0 plan §4.4). Repairing it would need `JDK_JAVA_OPTIONS`, which every `java`
    reads, so it reaches forked children. That works (H14), but it moves a §7.3.4 forbidden name
    onto the pass-list. More to the point, the narrow shape buys almost no security. With
    `$TMPDIR` and `java.io.tmpdir` both pointing at `/tmp/native`, all temp files land in the exec
    location anyway, and a hostile process simply writes there. What remains is a per-tool hunt
    for extraction routes. Each one missed fails **silently**: with only `JAVA_OPTS` redirected,
    `joern-parse --language jssrc` exits 0 and writes a CPG with **no methods** (H12). That is
    how the two gates came to disagree on whether astgen needs `TMPDIR`: the S04 gate saw exit 0,
    and the S02 gate's reading was right.
  - **Why not image-side pre-extraction** (pre-gunzip astgen, pre-extract zstd-jni at build
    time). It is unverified, it puts an S08 runtime property into S02's image, and it would still
    leave the forked-JVM route and any S24 tool to rediscover.
  - **What it costs, and why that is acceptable.** A4(a)'s `noexec` mechanism no longer covers
    `/tmp`. There, A4 rests on interpreter absence (`/init` is absent; A4(a′)'s memfd exec proves
    that no mount flag is needed) and on interop-socket absence, each sufficient alone (§9 H3). An
    ELF run from `/tmp` is still inside `--network none`, the seccomp profile and `--cap-drop ALL`,
    so A1, A3 and A7 are unaffected. Anything able to stage and run a binary in `/tmp` already
    has code execution in the tool's process, so an exec `/tmp` gives it nothing that the
    boundaries A1–A7 rely on withholding. `nosuid,nodev` stay.
  - **Consequences.** `TMPDIR` stays `/tmp` and `JAVA_OPTS` stays heap-only (§7.3.4), so A6(c)'s
    pattern is unchanged. The `-J-Djava.io.tmpdir` launcher exception is removed (§7.3.5), because the
    S02 and S04 gates both showed it does not reach the frontend's JVM. Test 3 checks CPG
    **content**, not the exit code alone (§10).
- **Noexec on Docker bind mounts.** `docker run -v`/`--mount type=bind` has no `noexec` option.
  S08 must achieve it, for example with a `local`-driver volume using `o=bind,ro,noexec`, and
  prove it from `/proc/self/mountinfo` (A5). If it cannot, the interop property still holds
  through A4's other two mechanisms. S08 then records that in its README and the S08 gate decides
  whether that is acceptable (§11 R2).

### 7.2 Escape axes, as testable properties

Each property is checked by S08's escape-axis probe: a script launched **through the runner,
exactly as a tool is launched**, on each backend that can start. When a backend refuses, the test
instead asserts the refusal and its precondition values in the `sandbox` record. Every connect
check is local: no packet may leave, and a **timeout counts as a failure**, because it means
something routed the packet.

| Axis | Property: the test passes iff… | Container closes it by | Namespace closes it by |
|---|---|---|---|
| **A1 IP** | (a) `/proc/net/dev` lists only `lo`. It is per network namespace and needs no `/sys`, which the namespace backend does not mount. (b) `/proc/net/route` and `/proc/net/ipv6_route` have no default route. (c) The sandbox's `/proc/self/ns/net` inode differs from the host's; the runner passes the host value in, and records the result as `preconditions.netnsDistinct`. (d) **Non-loopback targets:** `1.1.1.1:443`, `[2606:4700:4700::1111]:443`, each host nameserver from the host's `/etc/resolv.conf` on port 53, and each of the host's non-loopback addresses and the Docker bridge gateway on every `local` allow-list port. A TCP connect, and an unconnected UDP `sendto`, must each fail within 2 s with `ENETUNREACH`, `EHOSTUNREACH` or `EADDRNOTAVAIL`; a successful `sendto` fails the test. (e) **Loopback targets:** `127.0.0.1` and `::1` on every `local` allow-list port. Under mirrored networking the host's `localhost` reaches Windows, so LM Studio's 1234 must be unreachable here. A TCP connect must fail with `ECONNREFUSED` (when `lo` is up) or `ENETUNREACH`/`EADDRNOTAVAIL` (when `lo` is down). For UDP: `connect`, `send` one byte, then `recv` with a 2 s timeout. It passes on `ECONNREFUSED` from `send` or `recv`, or `ENETUNREACH`/`EADDRNOTAVAIL` from `connect`/`send`; a received datagram or a timeout fails. A plain loopback `sendto` succeeds when `lo` is up, so it proves nothing (S04 gate M2, verified). | `--network none` | `--unshare-net` |
| **A2 DNS** | (a) `getent ahosts example.com` exits non-zero within 5 s; (b) `getent hosts localhost` succeeds. DNS has a Unix-socket path too (systemd-resolved's varlink socket), so A2 also depends on A3. | no route (A1) and no host sockets (A3) | the same |
| **A3 Host Unix sockets** | (a) `find / -path /proc -prune -o -path /sys -prune -o -type s -print` prints nothing; (b) `/proc/net/unix` holds only its header line (so no **abstract** host socket is reachable); (c) for readable failure messages, each of these named host sockets is absent: `/run/docker.sock`, `/var/run/docker.sock`, `/run/WSL/*_interop`, `/tmp/.X11-unix/X0`, `/mnt/wslg/PulseServer`, `/run/user/<uid>/bus`, `/run/dbus/system_bus_socket`, `/run/systemd/resolve/io.systemd.Resolve` | nothing from the host is mounted except `/src` and `/out`; new netns (b) | minimal bind set; new netns (b) |
| **A4 WSL interop exec** | (a) a PE staged in each of `/src` (staged by the test before launch), `/out`, `/tmp` and `/home/ccg` **fails to execute**: non-zero exit, no stdout, with `ENOENT`, `EACCES` or `ENOEXEC`; in `/tmp`, which is exec (§7.1.2), it must be `ENOENT` when the host has a PE binfmt entry (its interpreter is absent inside) and `ENOEXEC` when it has none, never `EACCES`, so that leg isolates interpreter absence on a real file. The PE is `whoami.exe` copied from `/mnt/c/Windows/System32/` when the host has it; otherwise a two-byte `MZ` stub, which still matches the binfmt magic. (a′) **Mechanism-isolating variant:** the same PE written to a `memfd_create` file and executed through `/proc/self/fd/<n>` fails with `ENOENT`. No `noexec` mount can stop a memfd exec, so this exercises interpreter absence alone (verified by the S04 gate: `ENOENT`). (b) `WSL_INTEROP` is unset, `/run/WSL` is absent, and the PE binfmt entry's interpreter path (read from the host, `/init` here) is absent inside; (c) creating a file at that interpreter path fails, because the root is read-only | three independent mechanisms: no interpreter (the binfmt entry has flag `P`, not `F`, so `/init` is resolved inside the container's mount namespace), no interop socket, and `noexec` on `/src`, `/out` and `/home/ccg` (not on `/tmp`, which rests on the first two; §7.1.2); plus C1 | **refusal N1** on any host with a PE binfmt entry. Where N1 does not fire, (a)–(c) are asserted in full. |
| **A5 Mounted host and Windows paths** | (a) every mount point in `/proc/self/mountinfo` is in the backend's allowed set (§10, test 5); (b) no mount has fstype `9p`, `drvfs` or `virtiofs`, and no mount point is under `/mnt`, `/usr/lib/wsl` or `/run/WSL`, or is `/init`; (c) a sentinel file the runner creates in a fresh host temp dir outside the binds, and one in the host `$HOME`, are not visible at their host paths; (d) `/proc` lists no host process: a new PID namespace, so `/proc/<pid>/root` cannot reach the host filesystem | image root plus the two binds; Docker's PID namespace | the bind set, with `--tmpfs` masking `/usr/lib/wsl`; `--unshare-pid` |
| **A6 Environment** | (a) the probe, launched through the same `env -i` path as the tool, reads **its own** initial environment, and its keys **equal** `sandbox.envKeys` exactly. "Its own" means the process that `env -i` `exec`s: a compiled probe reads `/proc/self/environ`; a shell probe reads `/proc/$$/environ`, never the environ of a child it starts (such as `cat /proc/self/environ`), because a shell adds keys to its children (S04 gate P2-n1). `envKeys` is defined as the keys of the `K=V` arguments to `env -i` in `launcherArgv` (§5.3), which is the environment the tool process actually receives, not what the runner meant to pass. (b) `envKeys` equals the mode's pass-list (§7.3.4) and contains no name that is forbidden in that mode. (c) `JAVA_OPTS` matches the mode's pattern in §7.3.4. (d) Image `ENV` keys not on the pass-list (for example `JOERN_HOME`, `MAVEN_HOME`, `JAVA_VERSION`, `LANGUAGE`), and Docker's `HOSTNAME`, are absent. A later image layer that adds `ENV JAVA_TOOL_OPTIONS=…` therefore cannot reach the tool, and (d) catches the case where the wrapper is bypassed. (e) **Descendants of a shell launcher.** Joern's launchers are shell scripts, so the JVM they start also sees what the shell adds: `PWD` (dash, verified), and `PWD`, `SHLVL` and `_` (bash, verified by the S04 gate). The probe starts `/bin/bash -c` and `/bin/sh -c` children that read their own environ, and passes iff each child's keys ⊆ `envKeys` ∪ {`PWD`, `SHLVL`, `_`}. Those three are expected and harmless; anything else in a descendant fails. | `--entrypoint /usr/bin/env` with `-i` and the pass-list (§7 interface). No `-e`, `--env` or `--env-file`. | `--clearenv` plus `--setenv`, and the same `env -i` wrapper |
| **A7 AF_VSOCK** | `socket(AF_VSOCK, SOCK_STREAM, 0)` fails with `EPERM`, `EAFNOSUPPORT` or `EACCES`. The `preconditions.vsock` value is derived as follows. Container: `blocked` iff C2 passed, `present` is impossible because C2 would have refused, and the probe verifies `blocked`. Namespace: `present` if the host has `/dev/vsock` (then N2 refuses), else `absent`. | Docker's default seccomp profile allows `socket()` only for an enumerated list of families, and family 40 (`AF_VSOCK`) is not on it; C2 refuses without seccomp | **refusal N2** wherever `/dev/vsock` exists, because a network namespace does not isolate vsock on this kernel (§9 H5) |

**Why A7 is here although the brief lists five axes.** WSL2 reaches its Windows host over Hyper-V
sockets, which Linux exposes as `AF_VSOCK`. On this host an `AF_VSOCK` socket can be created
inside `unshare -rn` (§9 H5), so a network namespace does not close this channel. What I did
**not** verify is that a connect over it reaches a Windows listener that would relay traffic; that
part is inference. The container closes it anyway, and the test costs one syscall, so the axis is
contracted rather than argued.

**The launcher argv is part of the contract, and it depends on the mode.** S08 test 5 asserts
the following of `sandbox.launcherArgv` (container).
- **Every mode** must contain `--hostname localhost`, `--read-only`, `--cap-drop ALL`,
  `--security-opt no-new-privileges` and `--entrypoint /usr/bin/env`, with `-i` as the first image
  argument.
- It must contain none of:
  - `--privileged`;
  - `--pid`, `--ipc`, `--uts` or `--userns` set to `host`;
  - `--cap-add`; `--device`;
  - `--security-opt` with `seccomp=…` (other than §11 R4's runner-supplied profile) or
    `apparmor=unconfined`;
  - `-e`, `--env` or `--env-file`;
  - `--add-host`; `--dns`; `--publish`/`-p`;
  - any `-v`/`--mount` other than `/src` and `/out`; plus, in `controlled` mode, the cache volume
    of §6 step 7; plus, for `job: git-fetch`, the runner's bare mirror (as `/out`) and the
    read-only deploy-credential file (§6.1); plus, for `frontend = joern`, the one read-only
    script bind at `/scripts/<name>.sc` (§7.3.1).
- **`frontend = joern`** must also contain `--workdir /tmp`.
- **`offline`** must contain `--network none`, and no other `--network`.
- **`controlled`** (P2) must contain exactly one `--network ccg-egress-<auditId>` and no
  `--network none`.

**A1c, the controlled-mode probe set (P2 contract; P0 refuses `controlled`).** `--internal` alone
is **not** "no route except the broker". Docker documents that containers on an internal network
can reach the gateway IP, and so any host service listening on `0.0.0.0`; here that includes
`falkordb-dev`'s 6379 (S04 gate M4, from the CVE-2024-29018 advisories, not probed). So
controlled mode is contracted by properties, which P2's probe runs through the runner in
`controlled` mode:
- (a) the only TCP endpoint that accepts a connection is `broker:3128`. A connect to the network's
  gateway IP fails, on every port the host has listening (the runner passes `ss -ltn`'s port list
  in), and on `local` allow-list ports. So does a connect to `1.1.1.1:443`.
- (b) `getent ahosts example.com` fails; `getent hosts broker` succeeds.
- (c) A1(a)–(c) hold with `eth0` added: there is **no default route** in `/proc/net/route` or
  `/proc/net/ipv6_route`, and the only route is the internal subnet.
- (d) a `CONNECT` to a non-registry host through the broker is answered `403`, and writes a `deny`
  decision with rule `R6` or `R99`.
- (e) A3–A7 hold unchanged.
- (f) **No cross-binding reach** (S04 gate P2-m3). The probe runs with a second, concurrent
  `controlled` invocation holding its own binding. A connect from the first to the broker's
  address on the second's `ccg-egress-*` network fails, on 3128 and on every port the broker
  listens on. So does a connect to that second container's IP. Forging the peer address §2.3
  checks would need raw sockets, which `--cap-drop ALL` removes (`CAP_NET_RAW`; inferred, not
  probed).

P2 picks the mechanism that makes (a) true, and the probe proves it. The candidates, none of them
verified: a host-side `DOCKER-USER` rule dropping traffic from the internal subnet to the gateway
address; or a broker container that publishes no port, with the internal network created without
a host-reachable gateway. A design that cannot pass (a) does not ship.

### 7.3 Frontend flags: forbidden and reviewed-allowed

**7.3.1 The closed frontend set.** The runner launches only `pysrc2cpg`, `jssrc2cpg`,
`javasrc2cpg`, `jimple2cpg`, `kotlin2cpg`, `csharpsrc2cpg`, `gosrc2cpg`, `c2cpg` (the FR-1 set)
and `joern-parse`, plus `joern` in script mode only (below). Anything else is refused
(`unknown-frontend`, exit 64).

**`joern`, script mode (v1.3).** S11 runs `extract.sc` through the runner (P0 plan §4.4), and
that is `joern --script`. It is admitted in exactly one form, in `offline` mode only:
- **Argv grammar.** `toolArgv` must full-match `--script /scripts/<name>.sc` followed by zero or
  more `--param <k>=<v>` pairs, where `<name>` is the basename of `--script-file` and `<k>`
  matches `^[A-Za-z][A-Za-z0-9_]*$`. Any other element is refused (`forbidden-flag`), whatever it
  is. That covers the positional `<cpg.bin>`, `--import`, `--runBefore`/`--runAfter`,
  `--classpathEntry`, `-d/--dep`, `-r/--repo`, the plugin options, `--server*` and every
  launcher-level option. `--param` values are still checked by §7.4 steps 2 and 3, like every
  element.
- **Script staging.** The runner copies the `--script-file` host file into a fresh runner-owned
  temp directory. It checks and hashes **that copy**, and binds it read-only and `noexec` at
  `/scripts/<name>.sc` (`--ro-bind` on the namespace backend), so the bytes checked are the bytes run. `sandbox.script` records the path
  and SHA-256. The script never comes from `/src`, because the repository is untrusted input.
- **In-file directives are refused.** replpp honours `//> using …` directives at the head of a
  script. `using dep` and `using resolver` make the runner download a resolver from `github.com`
  and then resolve Maven artifacts, a fetch that no argv check can see. `using file` pulls
  another file into the compilation (§9 H15). The runner refuses
  (`forbidden-directive`, exit 64, no container created) any script with a line that starts
  with `//>` once leading whitespace is stripped. The match is case-insensitive, and any
  directive counts, not only `using dep`. An indented directive is inert today (H15), but it is
  refused anyway, so the check doesn't depend on replpp's parser. `flagCheck.forbiddenHits`
  records the offending lines. Refusing is chosen over neutralizing (rewriting the line),
  because the script is runner-owned code that must not change between review and run.
- **Mode.** `--mode controlled` with `--frontend joern` is refused (`unknown-frontend`). An
  extractor reads a CPG and needs no network, and offline mode leaves any fetch nothing to
  reach.
- **Working directory.** `joern` creates its `workspace/` in the current directory, and fails on
  the read-only image `WORKDIR` (`/work`). The runner adds `--workdir /tmp` (H15).

**Names and executables.** The lists, `--frontend` and `sandbox.frontend` all use the frontend
name without a suffix. The runner maps that name to the executable under `$JOERN_HOME`
(`/opt/joern/joern-cli` in the image, and the same bind path in the namespace backend):
`c2cpg` → `c2cpg.sh` and `jssrc2cpg` → `jssrc2cpg.sh`. Each of the others maps to the file of the
same name, and `joern-parse` maps to `joern-parse`. The executable path is part of `launcherArgv`.

**`joern-parse --language`.** Exactly one `--language` must be given, in either the
`--language X` or the `--language=X` form. Joern compares the value case-insensitively, and the
runner does too. It is refused (`unknown-frontend`) when it is absent, because `joern-parse` would
then guess the language and may pick `rubysrc2cpg`, when it is given twice, or when it is not in
this table:

| `--language` value | Frontend checked and recorded |
|---|---|
| `pythonsrc` | `pysrc2cpg` |
| `jssrc`, `javascript` | `jssrc2cpg` |
| `javasrc` | `javasrc2cpg` |
| `kotlin` | `kotlin2cpg` |
| `csharpsrc`, `csharp` | `csharpsrc2cpg` |
| `golang` | `gosrc2cpg` |
| `newc`, `c` | `c2cpg` |

**Refused aliases** (`unknown-frontend`; S04 gate P2-m1, both verified, §9 H13):
- **`python`** reaches `PyCpgGenerator`, which runs the commercial `py2cpg.sh`. That script is
  not in the image, so the parse exits 1. It never reaches `pysrc2cpg`; mapping it there would
  record a frontend that did not run.
- **`java`** is a redundant alias. With the commercial `java2cpg.sh` absent, it runs
  `jimple2cpg`, the **bytecode** frontend, deterministically. The name invites passing a Java
  source tree, and on one it exits 0 with a CPG holding no methods. Bytecode goes through
  `jimple2cpg` directly, and Java source through `javasrc`. (`csharp`, `golang` and `javascript`
  follow the same commercial-else-open-source pattern, with `csharp2cpg.sh`, `go2cpg.sh` and
  `js2cpg.sh`. They are kept, because their open-source targets are the source frontends the
  table names.)

**The table assumes the commercial scripts are absent.** If `java2cpg.sh`, `csharp2cpg.sh`,
`go2cpg.sh`, `py2cpg.sh` or `js2cpg.sh` appeared under `/opt/joern/joern-cli`, `csharp`,
`golang` and `javascript` would silently reach a different, unclassified frontend. Test 9 (§10)
asserts their absence.

`llvm`, `ghidra`, `php`,
`rubysrc`, `rust`, `swiftsrc`, `abap` and `fuzzy_test_lang` are outside FR-1. The language list
comes from `joern-parse --list-languages` (§9 H6); the alias pairs are the S04 gate's reading of
the console jar (`cpgcreation/package$`). The generators spawn the frontend's script, so
`--frontend-args` reach its launcher, and §7.4 checks them against both. The installed `abap2cpg`, `php2cpg`,
`rubysrc2cpg`, `rust2cpg`, `swiftsrc2cpg` and `ghidra2cpg` are outside FR-1 and stay unlaunchable.
That keeps the sweep's scope finite: `rubysrc2cpg`'s jar, for example, carries `RESOLVER_BASE_URL`
(§9 H7), and nobody has classified it.

**7.3.2 Classification rule.** Every flag of a launchable frontend whose **name or `--help`
description** matches, case-insensitively,

```
fetch|download|url|https?:|dependenc|resolver|gradle|maven|nuget|proxy|registry|remote|jar4
```

must appear in exactly one of the two files. It is **forbidden** unless there is evidence that it
opens no network path and fetches nothing. "It probably doesn't" is not evidence. The cost of
forbidding a harmless flag is a less precise offline parse, which FR-9 accepts. The cost of
allowing a fetching flag is an AC-8 failure. The regex extends m4's
`fetch|download|url|dependenc|resolver|gradle` with `https?:|maven|nuget|proxy|registry|remote|jar4`.
On today's `--help` output, the extension adds exactly one hit, `c2cpg --compilation-database`,
whose description contains an `https://` link.

**7.3.3 File formats.** Both files are tab-separated UTF-8. `#` starts a comment line. One entry
per line, with no blank fields.

- `cpg/sandbox/forbidden-flags.txt`: `<frontend>\t<flag>\t<forbiddenIn>\t<reason>`. `frontend` is
  a name from §7.3.1 or `*`. `forbiddenIn` is `offline` (allowed in `controlled` mode, P2) or
  `all`.
- `cpg/sandbox/allowed-flags.txt`: `<frontend>\t<flag>\t<reason>`. The reason is mandatory and
  states the evidence. An entry is "reviewed" when it has passed the S08 analyst gate. A later
  addition needs the same gate.

**Initial contents, from the `--help` of every frontend in §7.3.1 under Joern v4.0.579
(2026-10-29, §9 H6, H14).** The sweep regex matches 16 flags: 13 forbidden, 3 allowed. (v1.2 and
earlier said "13: 11 forbidden, 2 allowed", which did not match the table's own 13 rows: 10
forbidden and 3 allowed. v1.3 adds `joern`'s 3.)

| Frontend | Flag | List | `forbiddenIn` | Reason |
|---|---|---|---|---|
| `javasrc2cpg` | `--fetch-dependencies` | forbidden | `offline` | fetches dependency jars |
| `kotlin2cpg` | `--download-dependencies` | forbidden | `offline` | downloads dependencies |
| `kotlin2cpg` | `--jar4import-url` | forbidden | `all` | sends import names to a service (proposal §3.12). Only an `importNameServices` entry lifts it (§7.4 step 5, §6 step 6). |
| `kotlin2cpg` | `--gradle-project-name` | forbidden | `offline` | `--help`: "used to download dependencies" |
| `kotlin2cpg` | `--gradle-configuration-name` | forbidden | `offline` | `--help`: "used to download dependencies" |
| `kotlin2cpg` | `--enable-dependency-resolver-v2` | forbidden | `offline` | experimental dependency resolver; it has an environment twin (§7.3.4) |
| `kotlin2cpg` | `--generate-nodes-for-dependencies` | forbidden | `offline` | there is no evidence that it stays offline; S25 may move it with evidence |
| `csharpsrc2cpg` | `--download-dependencies` | forbidden | `offline` | downloads from NuGet; the jar carries `NUGET_BASE_API_V2`/`_V3` |
| `gosrc2cpg` | `--fetch-dependencies` | forbidden | `offline` | fetches modules |
| `gosrc2cpg` | `--include-indirect-dependencies` | forbidden | `offline` | F11; works together with `--fetch-dependencies` |
| `joern` | `--dep` | forbidden | `all` | resolves Maven artifacts and their transitive dependencies; the short form `-d` is refused by §7.3.1's argv grammar |
| `joern` | `--repo` | forbidden | `all` | adds a Maven repository to resolve from; short form `-r` as above |
| `joern` | `--verbose` | forbidden | `all` | it matched only on "resolved dependency jars" in its description, and it is harmless, but §7.3.1's grammar refuses it anyway, so it is listed for the sweep's sake |
| `c2cpg` | `--compilation-database` | **allowed** | — | reads a local `compile_commands.json`; it matched only on an `https://` link in its description |
| `javasrc2cpg` | `--delombok-mode` | **allowed** | — | runs delombok locally on the input; it matched only on "lombok dependency" in its description. Delombok is a child JVM inside the same sandbox. |
| `jimple2cpg` | `--full-resolver` | **allowed** | — | Soot's transitive resolution over classes in the input, per its `--help`. This is inferred, not probed; it matched on "resolver". |

The sweep regex does not match these flags, so they need no entry. They are listed here because
they are the **intended route for pre-provisioned caches** under local-only: `javasrc2cpg
--inference-jar-paths`, `kotlin2cpg --classpath`, `csharpsrc2cpg --external-summary-paths`,
`pysrc2cpg --venvDirs`, `c2cpg --include`. Their paths must lie under `/src` or a read-only cache
mount; a path outside the binds simply fails to resolve.

**7.3.4 Environment pass-list and forbidden names.** The tool's environment is exactly this
list, delivered through `env -i` (§7 interface, A6). No value is inherited from the host. Values
marked "image" are read by the runner from the image config (`docker image inspect`, `Config.Env`)
of the `toolchain` stage of `cpg/infra/toolchain/Dockerfile`. Those values are reviewed with S02's
and S24's image, and are recorded through `launcherArgv`.

| Name | Mode | Value | Why |
|---|---|---|---|
| `PATH` | both | container: the image's `PATH`, verbatim (today it starts `/opt/joern/joern-cli` and includes `/opt/java/openjdk/bin`); namespace: `/opt/joern/joern-cli:<host JDK>/bin:/usr/bin:/bin` | the launchers find `java` through `JAVA_HOME` or `PATH` |
| `JAVA_HOME` | both | image (`/opt/java/openjdk`); namespace: the host JDK 21 | the launchers read it (§9 H9) |
| `HOME` | both | `/home/ccg` | the launchers read it |
| `TMPDIR` | both | `/tmp` | exec-capable scratch (§7.1.2). jssrc2cpg's `astgen` unpacks itself here, and `java.io.tmpdir` keeps its default `/tmp`, so no JVM property is needed. |
| `LANG`, `LC_ALL` | both | `C.UTF-8` | replaces the image's `LANG`/`LANGUAGE`/`LC_ALL` |
| `JAVA_OPTS` | both | offline must full-match `^-Xmx[0-9]+[mg]( -Xss[0-9]+[km])?$`; controlled must full-match the offline pattern followed by ` -Dhttps\.proxyHost=broker -Dhttps\.proxyPort=3128 -Dhttp\.proxyHost=broker -Dhttp\.proxyPort=3128` | heap only; proxy settings only in controlled mode (§6) |
| `GOTOOLCHAIN` | both | `local` | **safety-positive.** If `env -i` dropped it, Go would default to `auto` and could try to download a toolchain. In offline mode that would only fail, but it would fail in a way that looks like a fetch. |
| `GOROOT` | both | image (`/opt/go`) | the image's Go, with no lookup |
| `DOTNET_ROOT` | both | image (`/opt/dotnet`) | |
| `DOTNET_CLI_TELEMETRY_OPTOUT`, `DOTNET_NOLOGO`, `DOTNET_SKIP_FIRST_TIME_EXPERIENCE` | both | `1` | **safety-positive.** Telemetry is egress in controlled mode, and first-run work is avoided. |
| `MAVEN_HOME`, `GRADLE_HOME` | controlled only | image | needed only by fetch-on frontends |
| `HTTPS_PROXY`, `HTTP_PROXY` | controlled only | `http://broker:3128`, with **no credential** (§6) | |

**Image `ENV` keys deliberately not passed:** `JOERN_HOME` (the launchers resolve their own
directory through `realpath`; §9 H9), `JAVA_VERSION`, `LANGUAGE`, `MAVEN_HOME` and
`GRADLE_HOME` (offline), plus Docker's `HOSTNAME`. That `JOERN_HOME` is unused is now verified:
both gates parsed every FR-1 language without it (§9 H12). `PWD`, `SHLVL` and `_` are not passed
either; shells add them to their own children (A6(e)). `JRE_CACERTS_PATH` is not an image `ENV`
key at all: Temurin's `/__cacert_entrypoint.sh` sets it at run time, and `--entrypoint
/usr/bin/env` bypasses that script (S04 gate P2-n1).

Forbidden names must never be on the pass-list for that mode. `*_PROXY` is forbidden in
`offline` only.
`ENABLE_DEPENDENCY_RESOLVER_V2` (it lives in the shared `x2cpg` jar, so any frontend may read it,
not only `kotlin2cpg`), `JAVASRC_FETCH_DEPENDENCIES` and `JAVASRC_JDK_PATH` (`javasrc2cpg`'s jar;
§9 H7), `JAVA_TOOL_OPTIONS`, `_JAVA_OPTIONS`, `JDK_JAVA_OPTIONS`, `*_PROXY` and `NO_PROXY` in
`offline` mode, `GOPROXY`, `GOFLAGS`, `GOPATH`, `NUGET_*`, `WSL_INTEROP`, `WSLENV`. The list is
documentary, because the pass-list is closed. It exists so test 7 can name the dangerous cases.
Test 7 is the **text half** of the check: this table, per mode, ∩ the forbidden names = ∅. The
**runtime half** is A6, which reads the tool's actual environment. Test 7 alone proves nothing
about the runtime (I5).

**7.3.5 Launcher-level options.** Every Joern frontend script is an sbt-native-packager launcher
(`frontends/<fe>/bin/<fe>`). It consumes these options **before** the application parses its
flags, and they never appear in `--help`. A `--help` sweep therefore cannot see them.
- `-D*` sets a JVM system property, for example `-Dhttps.proxyHost`.
- `-J*` passes a JVM option.
- `-agentlib*`, `-agentpath*` and `-javaagent*` load native or Java agents.
- `-XX*` passes a JVM `-XX` option.
- `-jvm-debug <port>` opens a JDWP listener.
- `-main <class>` runs an arbitrary main class.
- `-java-home <path>` replaces the JVM, and passes `<path>` through `eval`.

The runner refuses any `toolArgv` element matching one of these, except an element that
**full-matches** one of the runner's own options: `^-J-Xmx[0-9]+[mg]$` and `^-J-Xss[0-9]+[km]$`.
There is no `-J-Djava.io.tmpdir` exception: a `joern-parse -J…` option configures only the
`joern-parse` JVM, never the frontend it spawns (S02 and S04 gates; §7.1.2). Prefix exceptions would
let `-J-Xmx1g -Dx` style joins through. This list is **fixed in the runner, not kept in
`forbidden-flags.txt`**, for two reasons. Under §7.4's case-insensitive substring rule, `-d` would
match `--delombok-mode` and `-j` would match `--jdk-path`. And §7.3.6's stale-entry check would
fail on options that `--help` never shows. They are matched by §7.4 step 3 instead.

**7.3.6 The sweep (S08 test 6).** For each frontend in §7.3.1, run `<fe> --help` inside the
sandbox. Split the output into flag blocks (a line starting with whitespace and `-`, plus its
continuation lines). Apply §7.3.2's regex to each block. The test fails on any matching flag that
is in neither file, and on any entry in either file that no longer exists in `--help` (so the
lists cannot rot). It also fails if a flag is in both files. It runs in S08's gate, before S25
uses the lists, and again on any Joern upgrade.

### 7.4 Argv audit

Before launch, the runner checks `toolArgv`, the complete argv passed to the tool, as follows.
1. **Frontend:** it must be in §7.3.1. For `joern-parse`, the frontend is derived from the single
   `--language` through §7.3.1's table, and **every** element is checked against both that
   frontend's entries and `joern-parse`'s own. For `joern`, `toolArgv` must full-match §7.3.1's
   argv grammar (else `forbidden-flag`), and the staged script copy must carry no `//>` line
   (else `forbidden-directive`). Both are checked before any container exists, and steps 2–3
   still run on every element.
2. **Every element is checked, including those after `--` and `--frontend-args`.** The match is
   case-insensitive. An element `e` hits a forbidden entry `f` (for this frontend or `*`, and
   applicable in this mode) if `e == f`, or `e` starts with `f=`, or `e` contains `f` as a
   substring. The substring rule covers the `--flag=value` form, quoted and joined forms, and
   wrapped forms. Over-matching a path that happens to contain a flag name is accepted: it refuses,
   which fails closed, and the record shows why.
3. **Launcher-level options** (§7.3.5) are matched **case-sensitively, by prefix**, on every
   element that starts with a single `-` followed by a non-`-` character, wherever it is in argv:
   `-D`, `-J`, `-agentlib`, `-agentpath`, `-javaagent`, `-XX`, and the exact elements
   `-jvm-debug`, `-main` and `-java-home`.
4. **Any hit refuses:** exit 64, `refused: true`, `refusalReason: forbidden-flag` (or
   `forbidden-directive` for step 1's script check),
   `flagCheck.forbiddenHits` = the matching elements, and no container or namespace is created.
5. **Lift (`controlled` mode only, P2).** A hit on `kotlin2cpg --jar4import-url` (`forbiddenIn
   = all`) is lifted, and is not a refusal, iff its value canonicalizes (§4.3) to the endpoint of
   an `importNameServices` entry. A lifted element is recorded in `flagCheck.liftedHits` as
   `{element, entryId}`, never in `allowedHits`. In `offline` mode nothing is ever lifted. This is
   the only lift. Being on a list is not a lift, so §7.3.6's "never in both files" rule stands.
6. **No remaining hit:** `flagCheck.allowedHits` records the elements that match
   `allowed-flags.txt`. Then the runner launches and writes the record with the exit code.
7. `flagCheck.listDigest` ties the record to the exact list version. AC-8 re-checks a historical
   record against the list current at that digest.

The runner never builds `toolArgv` from environment variables or from files in the source tree.
`launcherArgv` is constructed by the runner from fixed templates. It is recorded verbatim and
checked by §7.2's launcher clause.

## 8. AC-8 assertion → audit field map

**The scenario's topology** (proposal §5.3; the harness is P1). AC-8's logging proxy is the only
route out of the scenario's network, and only the broker is attached to it. It writes one `conn`
record per upstream connection: `{ts, srcAddr, srcPort, dstAddr, dstPort, bytesOut, bytesIn}`.

**The join.** A `conn` record matches a `decision` record when `conn.srcPort =
decision.upstream.localPort`, `conn.dstAddr:dstPort = decision.resolvedAddr` plus the endpoint's
port, and `|conn.ts − decision.ts| ≤ 5 s`. **Every `conn` must match an `allow` decision**, or AC-8
fails. This is what makes "zero connections outside the allow-list" checkable from records rather
than from the broker's word.

Below, *L* is the set of codebases with `effective = local-only` in a run's `policy` record. Each
assertion comes from proposal §5.3, with requirements AC-8's wording where it adds something.

| # | AC-8 assertion | Audit fields | Check |
|---|---|---|---|
| 8.1 | A local-only codebase makes **zero connections outside the allow-list** | `conn.*`; `decision.{codebases, decision, rule, endpoint}`; `policy.modes` | Every `conn` joins an `allow`. For every `decision` with `codebases ∩ L ≠ ∅` and `decision = allow`: `rule ∈ {R1, R2, R3, R4}`. |
| 8.2 | …and the only allowed outbound ones are its own git remotes, the vulnerability-database job, and allow-listed IdP, access-check or PR services | `decision.{rule, endpoint, codebases, purpose, component}`; `close.bytesOut`; `sandbox.{job, toolArgv}` | For each allow with `rule ≠ R1`: `R2` ⇒ `component = build.git`, `codebases = [c]`, `endpoint = canonical(remote(c))`, every linked `close.bytesOut ≤ gitFetchMaxBytesOut`, and the run's `job: git-fetch` record's `toolArgv` equals §6.1's fixed argv; `R3` ⇒ `purpose = vulndb-refresh`; `R4` ⇒ `purpose ∈ {idp, access-check, pr-addon}`. The byte bound narrows §6.1's residual risk but does not remove it. The report lists every allow with `privateAddress = true`, for review. |
| 8.3 | …whose request bodies carry **nothing beyond repository names and identity** | `decision.{payloadClass, payloadFields, transport, bytes}`; `close.bytesOut` | Every `R4` allow has `transport = structured`, `payloadClass = repo-names-and-identity` and `payloadFields ⊆ {repoNames, identity.subject, identity.issuer}`. A deny with `rule = R4` for an extra field is the negative case. |
| 8.4 | The vulnerability-database job is **separate and never sees repository content** | `policy.{runKind, codebases}`; `decision.{component, codebases, payloadClass, runId}` | Every `R3` allow has `runId` equal to a `policy` record with `runKind = vulndb` and `codebases = []`, plus `payloadClass ∈ {none, vuln-feed-request}`. No `vulndb` run's `runId` appears on any `sandbox` record. |
| 8.5 | A local-only build makes **no dependency download** | `decision.{purpose, decision, codebases}`; `sandbox.{egressMode, network, codebases}` | No `decision` has `purpose = dependency-download`, `decision = allow` and `codebases ∩ L ≠ ∅`. Every `sandbox` record whose `codebases ∩ L ≠ ∅` has `egressMode = offline` and `network = none`. |
| 8.6 | Every fetch-capable frontend is invoked **with its fetch flags off** (asserted from the command lines) | `sandbox.{toolArgv, launcherArgv, flagCheck.listDigest, flagCheck.forbiddenHits, envKeys, frontend, refused}` | For every non-refused `sandbox` record of a codebase in *L*: re-running §7.4 on `toolArgv`, against the lists at `listDigest`, yields no hit; `forbiddenHits = []`; `launcherArgv` contains `--network none`; `envKeys` contains no §7.3.4 forbidden name. The negative case is a forbidden flag on purpose, giving `refused = true`, `exitCode = 64`. |
| 8.7 | In a mixed system, a question touching **only hosted repositories is still interpreted by a local model** | `policy.systemLocalOnly`; `decision.{component, purpose, rule, endpoint}` | In a run with `systemLocalOnly = true`, every `decision` with `component = answer.interpret` and `decision = allow` has `rule = R1`. A hosted attempt appears as `deny`, `rule = R5`. |
| 8.8 | An answer touching a local-only repository is **handled entirely locally** | `policy.{runId, runKind, codebases}`; `decision.{runId, requestMode, rule}` | Select by `runId` alone, never by `decision.codebases`: for every `answer` run whose `policy.codebases ∩ L ≠ ∅`, **every** `decision` with that `runId` has `requestMode = local-only`, and every allow has `rule ∈ {R1, R4}`. This verifies §3.3's run-scoped `M` rather than trusting each call's declaration (S04 gate P2-m2). |
| 8.9 | In a system containing a local-only repository, **every model call** goes to an allow-listed local endpoint (requirements AC-8) | `policy.systemLocalOnly`; `decision.{purpose, decision, rule}` | Every `decision` with `purpose = model-call`, `decision = allow` and a `runId` whose `policy.systemLocalOnly = true` has `rule = R1`. |
| 8.10 | A **hosted override on a local-only codebase is refused** | `policy.{override, modes[].rule, modes[].effective, overrideRefused}` | `override = hosted`; that codebase's row has `rule = P2-OVERRIDE-REFUSED` and `effective = local-only`; `overrideRefused = true`. Then 8.1 and 8.5 hold for that run. |
| 8.11 | A hosted-default codebase **run with a local override stays local** | `policy.modes[]` (`default = hosted`, `effective = local-only`, `rule = P3-OVERRIDE-LOCAL`); `sandbox.egressMode` | The row is as stated, and 8.1, 8.5 and 8.6 hold for that codebase in that run (`sandbox.egressMode = offline`). |
| 8.12 | A hosted-default build downloads dependencies **only through the controlled path** | `sandbox.{job, egressMode, network, networkName, launcherArgv, toolArgv, codebases, flagCheck.listDigest}`; `policy.modes`; `decision.{purpose, rule, endpoint}`; `conn` | Two checks on `sandbox` records. (i) Every record with `egressMode = controlled` has `network = broker`, `networkName = ccg-egress-<auditId>`, and that `--network` in `launcherArgv`. (ii) Every record whose `toolArgv` contains a flag with `forbiddenIn = offline` (in the lists at `listDigest`) has `egressMode = controlled` and a codebase with `effective = hosted`. On the broker side, every `dependency-download` allow has `rule = R8`, and every `conn` in the run joins a decision. A1c passing in the scenario is the precondition. |
| 8.13 | …**with no import-name service** | `decision.{purpose, rule}`; `sandbox.flagCheck.{forbiddenHits, liftedHits}` | No allow with `purpose = import-name-lookup` unless `rule = R9`. Every `toolArgv` element containing `--jar4import-url` appears in that record's `liftedHits`, with an `entryId` naming an `importNameServices` entry and `egressMode = controlled`. Otherwise the record must have been refused. |
| 8.14 | Re-checked as each phase adds outbound traffic | `decision.component`, `decision.purpose` (closed enums, §2.4); `schemaVersion` | A new outbound path needs a new `Component` or `Purpose` value, which is a revision of this spec. That revision is the trigger to re-run AC-8. A record whose value is outside the enums fails schema validation. |
| 8.15 | (completeness of 8.1, 8.5 and 8.8) No request can opt out of the sticky rule by under-declaring | `decision.{codebases, decision, rule, transport, runId}`; `policy.{runKind, modes}` | (i) No allow, in any run whose *L* ≠ ∅, has `codebases = []` unless that run's `runKind = vulndb`. (ii) Every allow with `rule ∈ {R8, R9}` has `transport = proxy`. (iii) In a `refresh` run whose *L* ≠ ∅, every `structured` allow has `rule ∈ {R1, R4}`. With §3.3 none of these can fail, so the checks confirm the rules hold in practice. |

**What the join cannot see** (S04 gate Q1). The harness covers each gap explicitly:
- **UDP and DNS:** `conn` is TCP-only. Each component's container must pass A1/A2-style probes
  inside the scenario.
- **`local` (`R1`) traffic** never reaches the logging proxy. It is audited by `decision` records
  only, and it is permitted by definition.
- **Connections outside the window:** the logging proxy must start before the first `policy`
  record, and it writes a closing flush marker. A `conn` before the start or after the marker fails.
- **Non-broker components:** each must have no route other than the broker's listeners. That is
  shown by an A1-style probe from each component's container, not by the join.

**Checked from the sandbox side as well.** 8.1 and 8.5 assume that a sandboxed build has no
route. That is §7.2's escape-axis probe, which runs inside the AC-8 scenario on the backend the
scenario uses.

## 9. Host probe record

Read-only probes on this host on 2026-10-29: WSL2, kernel `5.15.167.4-microsoft-standard-WSL2`,
`networkingMode=mirrored`. Nothing was installed or changed, and no request was sent to an
external host. Raw `--help` output is in the session scratchpad and is not committed.

| # | Probe | Result | Used in |
|---|---|---|---|
| H1 | `ls /proc/sys/fs/binfmt_misc/`; `cat …/WSLInterop-late` | Entries: `WSLInterop-late`, `python3.12`. **No `WSLInterop`** (`cat` → "No such file"). `WSLInterop-late`: `enabled`, `interpreter /init`, `flags: P`, `magic 4d5a`. | N1, C1, A4 |
| H2 | `unshare -rn sh -c …` | `lo` DOWN and the only link; `getent hosts example.com` → rc 2; `/init` visible; `/run/docker.sock` visible; `/mnt` = `c wsl wslg`; **`/mnt/c/Windows/System32/whoami.exe` → rc 0 (runs)** | confirms M1; A1–A5 |
| H3 | `bwrap --unshare-all --clearenv` with a minimal `/usr` bind, the PE bound in at `/tmp/w.exe`, three variants | No `/init` → `execvp: No such file or directory` (rc 1). `/init` bound → WSL error, rc 1. `/init` + `/run/WSL` + `WSL_INTEROP` → **rc 0 (runs, even with a new netns)**. | A4 mechanisms: interpreter absence and socket absence are each sufficient; the network namespace is irrelevant |
| H4 | `find /run /var/run /tmp -maxdepth 3 -type s`; `/proc/self/mountinfo` | Host sockets include `/run/docker.sock`, `/run/WSL/{2,844,56393}_interop`, `/tmp/.X11-unix/X0`, `/run/user/1000/bus`, `/run/dbus/system_bus_socket`, `/run/systemd/resolve/io.systemd.Resolve`, plus `/mnt/wslg/PulseServer`. Windows-backed mounts are `/mnt/c` (`C:\`), **`/usr/lib/wsl/drivers` and `/usr/lib/wsl/lib`**, and `/mnt/wslg/*`. `/init` is its own `ro` mount. | A3, A5 (why `/usr/lib/wsl` is masked) |
| H5 | `unshare -rn python3 -I -c 'socket(AF_VSOCK, SOCK_STREAM)'`; `ls /dev/vsock` | **The socket is created inside the netns.** `/dev/vsock` exists (`crw-rw-rw-`). No connect was attempted. | A7, N2 |
| H6 | `<fe> --help` for the 8 FR-1 frontends and `joern-parse` (`~/joern/joern-cli`, v4.0.579) | All exit 0. The sweep regex (§7.3.2) gives the 13 flags of §7.3.3; m4's narrower regex gives 12. **F11 confirmed:** `gosrc2cpg --include-indirect-dependencies` ("works along with flag 'fetch-dependencies'"). `jssrc2cpg` and `pysrc2cpg` have no network-shaped flag. `joern-parse` passes everything after `--frontend-args` verbatim. | §7.3 |
| H7 | `strings` over `frontends/<fe>/lib/io.joern.<fe>-4.0.579.jar` and `lib/io.joern.x2cpg-4.0.579.jar`, filtered for upper-case constant names | `javasrc2cpg`: `JAVASRC_FETCH_DEPENDENCIES`, `JAVASRC_JDK_PATH`. `x2cpg` (shared): `ENABLE_DEPENDENCY_RESOLVER_V2`. `csharpsrc2cpg`: `NUGET_BASE_API_V2`/`_V3`. `kotlin2cpg`: `JDK_HOME`. `rubysrc2cpg` (outside FR-1): `RESOLVER_BASE_URL`, `GEM_PATH`. These are constant names in class files. That each is read as an environment variable is **inferred** from its name, and from `--help`'s statement for `ENABLE_DEPENDENCY_RESOLVER_V2`. | §7.3.4 |
| H8 | `javasrc2cpg --show-env <empty dir>` | **Prints no variables.** It fails on `<input_dir_placeholder>` whatever the argument, so the documented way to list `javasrc2cpg`'s environment variables does not work in v4.0.579, and H7 was used instead. | §7.3.4 |
| H9 | `frontends/kotlin2cpg/bin/kotlin2cpg`, `process_args` (the same native-packager template in every frontend) | It consumes `-D*`, `-J*`, `-agentlib*`, `-agentpath*`, `-javaagent*`, `-XX*`, `-jvm-debug <port>`, `-main <class>` and `-java-home <path>` (passed through `eval echo`), and reads `JAVA_OPTS`. None of these appears in the app's `--help`. | §7.3.5 |
| H10 | (verified by the S04 gate on Docker 29.7.2: built-in profile → `EPERM` for `AF_VSOCK`; `seccomp=unconfined` → socket created) `docker info --format '{{.SecurityOptions}}'`; moby `profiles/seccomp/default.json` (`main`, fetched 2026-10-29) | `[name=seccomp,profile=builtin name=cgroupns]`. The profile allows `socket` only for families `< 3` and an enumerated list from 7 to 45 that omits 40 (`AF_VSOCK`), so the default is `EPERM`. That Docker 29.7's built-in profile equals `main` is **assumed**; S08's A7 probe is what verifies it. No container was started for this spec. | A7, C2 |
| H11 | `grep hosts /etc/nsswitch.conf` | `hosts: files dns`. There is no `resolve` module, so glibc does not use the systemd-resolved socket, but a process could still talk to it directly (A3). | A2 |

**v1.2 probes (2026-10-29), on the S02 `toolchain` stage image** (`ccg.joern.version=4.0.579`).
S02 was rebuilt during the probes: H12 shapes (2) and (3) and H14 shape (3) ran on the build the
S02 gate reviewed; H12 shapes (1) and (4) and H14 shape (4) ran on the rebuild; H13 ran on both; H15 (v1.3) ran on the rebuild.
Both builds carry the same Joern label.
Every run used `docker run --rm --pull=never --network none` with the §7.1 launch shape (`env -i`
pass-list, `--read-only`, `--cap-drop ALL`, `no-new-privileges`, `--user 1000:1000`,
`--hostname localhost`), one-file fixtures per FR-1 language bound read-only at `/src`, and a
`/tmp` shape varied per row. "Methods" means the non-external method names in the CPG, read by a
`joern --script` that loads it.

| # | Probe | Result | Used in |
|---|---|---|---|
| H12 | `joern-parse --language <l>` for `jssrc`, `golang`, `pythonsrc`, `javasrc`, `csharpsrc`, `newc`, `kotlin`, in four `/tmp` shapes | (1) `/tmp` `noexec`, `JAVA_OPTS=-Xmx2g`, `TMPDIR=/tmp` (v1.1 literal): `pythonsrc` and `jssrc` exit 1 (zstd-jni `failed to map segment`; the S02 gate saw all 7 fail). (2) `/tmp` `noexec` plus an exec `/tmp/native`, `JAVA_OPTS=-Xmx2g -Djava.io.tmpdir=/tmp/native`, `TMPDIR=/tmp`: all 7 exit 0, **but `jssrc`'s CPG has no methods** (9,195 B against 83,611 B). Running `jssrc2cpg.sh` directly in that shape logs `running astgen failed! … /tmp/gztmp…/astgen-linux: Permission denied`; `joern-parse` hides it. (3) As (2) with `TMPDIR=/tmp/native`: all 7 exit 0, `jssrc` methods `:program,f`. (4) **v1.2 shape**, `/tmp` `rw,exec,nosuid,nodev`, `JAVA_OPTS=-Xmx2g`, `TMPDIR=/tmp`: all 7 exit 0 in 6–13 s, each CPG holds the fixture's method (`f`/`F`, plus module or global methods). | §7.1, §7.1.2, §7.3.4, §10 test 3 |
| H13 | `joern-parse --language python` and `--language java`; `test -e` of the commercial scripts | `python`: exit 1, `CPG generator does not exist at: /opt/joern/joern-cli/py2cpg.sh` (from `PyCpgGenerator`). `java`: runs `/opt/joern/joern-cli/jimple2cpg`, exit 0, and on a Java source fixture the CPG has no methods. `java2cpg.sh`, `csharp2cpg.sh`, `go2cpg.sh`, `py2cpg.sh` and `js2cpg.sh` are all absent. | §7.3.1, §10 test 9 |
| H14 | `joern --script` (loads a CPG) under shapes (3) and (4) of H12; process listing during the run | Shape (3): exit 1, zstd-jni `UnsatisfiedLinkError` in `/tmp`. The `ReplBridge` JVM has `-Djava.io.tmpdir=/tmp/native` from `JAVA_OPTS`, but it forks a child JVM (`replpp.scripting.NonForkingScriptRunner`) whose command line carries no `JAVA_OPTS` properties. Adding `JDK_JAVA_OPTIONS=-Djava.io.tmpdir=/tmp/native` makes it pass. Shape (4): exit 0 with no extra variable. `joern --help` matches §7.3.2's regex on `-d/--dep` and `-r/--repo`, which resolve Maven artifacts. | §7.1.2, §11 R10 |
| H15 | `joern --script` of small scripts with `//>` directives, under H12 shape (4); then the default `WORKDIR` | No directive: exit 0. `//> using dep com.michaelpollmeier:versionsort:1.0.7`, and `//> using resolver …` plus `using dep`: exit 1 with `UnknownHostException: github.com` from `replpp.util.Cache.getOrDownload` via `replpp.Dependencies.resolve`, so the parent JVM tries a download before compiling. `//> using file other.sc`: exit 0, and the other file's code is compiled in. The same `using dep` indented by two spaces: exit 0 with no download attempt (inert). Without `--workdir /tmp` (the image's `WORKDIR` is the read-only `/work`): exit 1, "Error during compilation". | §7.3.1, §10 test 10 |

## 10. What S08 encodes

The P0 plan's §4.1 S08 tests, as this spec amends them. Every one runs per backend; when a
backend refuses on this host, the test asserts the refusal record instead (§7.1).

| Test | What it asserts | Spec |
|---|---|---|
| 1 | The escape-axis probe: **A1–A7**, every sub-property including A1(e)'s loopback UDP procedure, A4(a)'s `ENOENT` leg in `/tmp`, A4(a′)'s memfd variant and A6(e)'s shell-descendant check, with timeouts treated as failures | §7.2 |
| 2 | Each forbidden entry is refused before launch, in four forms: `--flag`, `--flag=value`, after `--`, and after `joern-parse … --frontend-args`. Each launcher-level option (§7.3.5) is refused too. So are near-misses of the exceptions, for example `-J-Xmx1g-x` and `-J-Xmx1g -Dx` as one element (n1). `joern-parse` with no `--language`, with two, with `java`, with `python`, and with an alias outside FR-1 is refused (`unknown-frontend`). A refusal is exit 64 plus `refused: true` and `forbiddenHits`. **No container is created:** the test runs `docker events --filter type=container --filter event=create --since <t0> --until <t1>` over the test window and expects no event. `docker ps -a` cannot see a `--rm` container created and removed between two snapshots. | §7.3.1, §7.3.3, §7.3.5, §7.4 |
| 3 | `joern-parse` of a one-file project in **each FR-1 language** (`pythonsrc`, `jssrc`, `javasrc`, `kotlin`, `csharpsrc`, `golang`, `newc`), through the runner in `offline` mode, exits 0 in under 120 s with no `UnknownHostException` (JVM `localhost` resolution), **and its CPG contains the fixture's method**. Exit 0 alone is not a pass: with a misconfigured temp directory, `jssrc` exits 0 with an empty CPG (§9 H12). The content check loads `/out`'s CPG with a fixed `joern --script` that prints the non-external method names, run through the runner (`--frontend joern`, §7.3.1). All seven languages, because each temp-directory failure found so far was specific to some frontends (§7.1.2). | §7 interface, §7.1.2 |
| 4 | Every invocation, refusals included, appends exactly one `sandbox` record carrying every §5.3 field with a valid type. `auditId` is the stem of the tool's own output files, and `envKeys` equals the `env -i` keys in `launcherArgv`. | §5.1, §5.3 |
| 5 | Mounts: `/proc/self/mountinfo` mount points equal the backend's allowed set. In the container, `/src`, `/out` and `/home/ccg` are `noexec`, and `/tmp` is `nosuid,nodev` **without** `noexec` (§7.1.2); `launcherArgv` carries `--tmpfs /tmp:rw,exec,nosuid,nodev` with `exec` written out. `launcherArgv` satisfies §7.2's launcher clause. The allowed set for the container is `/`, `/src`, `/out`, `/tmp`, `/home/ccg`, `/proc` and `/proc/*`, `/dev` and `/dev/*`, `/sys` and `/sys/*`, and Docker's `/etc/{hosts,hostname,resolv.conf}`, plus `/scripts/<name>.sc` (read-only, `noexec`) for `frontend = joern` only; for the namespace backend it is the §7.1 bind set. | §7.1, §7.2 A5, §7.3.1 |
| 6 | The flag sweep of §7.3.6: an unclassified hit, a stale entry, or a double entry each fails | §7.3.2, §7.3.6 |
| 7 | **Text half only:** for each mode, §7.3.4's pass-list ∩ that mode's forbidden names = ∅ (`HTTPS_PROXY` is allowed in `controlled` and forbidden in `offline`). The runtime half is A6 in test 1, which reads the tool's real environment and checks that the image's other `ENV` keys and `HOSTNAME` are absent. | §7.3.4, §7.2 A6 |
| **8** (new) | **Refusal detection.** `interop_state` (§7.1.1), N2, C1 and C2 take the `binfmt_misc` directory, the `/dev/vsock` path, the kernel release and the `docker info` output as parameters. Each fixture F1–F12 of §7.1.1 must give exactly the listed `interop_state`, namespace outcome and C1 outcome. C2 fixtures: `name=seccomp,profile=builtin` passes; a `name=seccomp` entry with any other `profile=` value (a daemon-level profile) refuses, and so does an output with no seccomp entry. The **Host** row runs for real. | §7.1, §7.1.1 |
| **9** (new, v1.2) | **Frontend identity in the image.** `java2cpg.sh`, `csharp2cpg.sh`, `go2cpg.sh`, `py2cpg.sh` and `js2cpg.sh` are absent from `/opt/joern/joern-cli` (§7.3.1). If one appears, `csharp`, `golang`, `javascript` or `java` would silently reach a different frontend. The check runs inside the sandbox, and again on any image rebuild. | §7.3.1 |
| **10** (new, v1.3) | **`joern` script mode.** (a) Refused before launch, with `docker events` showing no container: each option §7.3.1's grammar excludes (`--import`, `--runBefore`, `--classpathEntry`, `-d`, `--dep`, `-r`, `--repo`, `--server`, `--add-plugin`, a positional CPG, a `-J`/`-D` option), a `--script` path other than `/scripts/<basename of --script-file>`, a malformed `--param`, a missing `--script-file`, and `--mode controlled` (`unknown-frontend`). (b) Refused with `forbidden-directive` and the line in `forbiddenHits`: scripts carrying `//> using dep …`, `//> using resolver …`, `//> using file …`, an indented `  //>   using dep …`, and `//> USING DEP …`. (c) Accepted: a directive-free script with `--param cpgFile=/out/<x>.bin` runs, prints the fixture's methods, and its record carries `script.sha256` equal to the file's hash and `launcherArgv` containing `--workdir /tmp`. | §7.3.1, §7.4, §5.3 |

S08 also adds `.audit/` to `cpg/.gitignore` (§5.1).

## 11. Open questions and risks

None blocks S08. Each one below has an owner and a fallback.

- **Q1 (open stakeholder decision, P1, via `tico`) — the host-granular git tunnel (§6.1, S04
  gate M5).** Should the residual risk of an `R2` tunnel that can reach any repository on the
  remote's host be accepted, given §6.1's mitigations (runner-only fixed argv, read-only deploy
  credential, no hooks, a byte bound checked by AC-8)? Or does local-only require TLS interception
  for `R2`, with a broker-held CA trusted only by the fetch container? Not decided here. P0 is not
  affected, because it builds no broker.
- **R1 — Resolved: N1 placement.** The P0 plan's literal `WSLInterop` path is superseded by
  §7.1.1, which `teco` accepted. S08's brief states that §7.1.1 controls. The P0 plan is not edited
  (collision rule 5).
- **R2 — Docker bind mounts have no `noexec` option.** The local-volume route (§7.1) is not
  verified here, because no container or volume was created. If it fails, A4 still holds through
  interpreter absence and socket absence (§9 H3), but the P0 plan's test 5 wording (`ro,noexec`)
  does not. **Owner:** S08 reports it; the S08 gate accepts or rejects it.
- **R3 — Resolved (v1.2): `/tmp` is exec-capable.** A `noexec` `/tmp` broke every frontend, and
  the `-J-Djava.io.tmpdir` fallback this item used to permit does not reach the frontend's JVM
  (S02 gate M1, S04 gate P2-M1). The decision and its cost to A4 are in §7.1.2; there is no
  fallback option left to add.
- **R4 — A7's route is inferred; its closure is verified.** That `AF_VSOCK` leads to a Windows
  listener that would relay traffic is still not probed. The S04 gate verified the closure on
  Docker 29.7.2: the built-in profile gives `EPERM`, and `seccomp=unconfined` creates the socket
  (§9 H10). If a later daemon's built-in profile allows `AF_VSOCK`, C2 becomes "a runner-supplied
  profile", and the runner passes `--security-opt seccomp=<file>`, a copy of the default that also
  blocks family 40. That is the only `seccomp=` value §7.2's launcher clause admits.
- **R5 — The namespace backend's full-probe branch cannot run on this host**, because it refuses
  (N1, N2). Test 8 covers the refusal logic. A1–A3, A5 and A6 for the namespace backend can be
  exercised by running its bwrap template **directly** from the test, not through the runner, so
  the runner never gains a bypass option. A4 and A7 for that branch stay unexercised until a
  non-WSL host exists, and S08's README must say so.
- **R6 — Resolved: the bundle seam.** S03 chose `sandboxAuditIds`, an array, and §5.3 now states
  the loader check over every id.
- **R8 — A1c's mechanism is unproven (P2).** No Docker network was created for this spec or its
  gate. §7.2 lists the candidate mechanisms; A1c(a) is the ship criterion.
- **R9 — Resolved (v1.2): the pass-list exclusions.** `JOERN_HOME` is unused: both gates parsed
  every FR-1 language without it (§9 H12). `JRE_CACERTS_PATH` is not an image `ENV` key (§7.3.4).
- **R10 — Resolved (v1.3): `joern` script mode is launchable.** §7.3.1 admits `joern --script` in
  one fixed, offline-only form, and refuses any script with a `//>` directive (§9 H15). Two
  residuals remain. The script is runner-owned code that can itself do anything a JVM can, inside
  the offline sandbox; its review is S11's gate, and `sandbox.script.sha256` ties each run to
  the reviewed bytes. replpp may accept other inline mechanisms that this spec doesn't know about;
  the directive check covers the documented one, and offline mode leaves any other nothing to
  reach.
- **R11 — Exit 0 does not mean a parse produced anything.** `joern-parse` hides a frontend's
  internal failure (§9 H12: `jssrc` exits 0 with an empty CPG). §5.3's loader check reads
  `exitCode: 0` only. Test 3 closes this for the sandbox's own configuration. A content check
  in the bundle pipeline (for example, a unit with source files but no methods is invalid) is for
  S11 and the bundle spec's owner to decide; it is flagged here, not specified.
- **R7 — Three classifications rest on `--help` text only:** `jimple2cpg --full-resolver` and
  `javasrc2cpg --delombok-mode` (allowed), and `kotlin2cpg --generate-nodes-for-dependencies`
  (forbidden). The sandbox closes the network either way, so the risk is precision for the first
  two, not egress. S25's fetch-off runs are where the evidence for moving one appears.
- **Decision taken: a refused hosted override does not abort the run** (§3.2). The proposal says
  "refused and logged". Proceeding as local-only satisfies that and the AC. Aborting would be
  stricter but would turn a typo into a failed refresh. This can be reversed later without changing
  any record shape.

## 12. Review dispositions

### 12.1 S04 gate, Pass 1 (v1.1)

Gate: `docs/reviews/connected-code-graph-p0-impl.md`, `## S04 (P0-03b)`, verdict **needs
changes, no blockers**.

| Finding | Disposition | Where |
|---|---|---|
| **M1** env not removable; `envKeys` records intent; `PATH` wrong | **Fixed as recommended.** The tool is started through `--entrypoint /usr/bin/env -i`. `envKeys` = the `env -i` keys, and A6(a)/(d) verify them against the real environment. The pass-list is explicit per mode: `GOTOOLCHAIN=local` and the `DOTNET_*` opt-outs are kept as safety-positive, and `PATH`/`JAVA_HOME`/`GOROOT`/`DOTNET_ROOT` come from the `toolchain` stage's image config. Also adopted from S02: `--hostname localhost` is required. | §7, §7.1.2, §7.2 A6, §7.3.4, §5.3 |
| **M2** loopback UDP `sendto` succeeds | **Fixed as recommended.** A1 is split into non-loopback targets (d: `sendto` must fail) and loopback targets (e: connect + send + `recv`, which passes on `ECONNREFUSED`). | §7.2 A1 |
| **M3** empty or under-declared `codebases` escapes the sticky rule; `R3` not bound to `vulndb` | **Fixed as recommended, plus §2.4a.** `R0` denies `codebases = []` outside `vulndb`; `R3` requires `runKind = vulndb`; `open_run` enforces codebase cardinality per run kind; `M` for `answer` runs is computed over the run's codebases; components are restricted per run kind. Check 8.15 is added. | §2.1, §2.4a, §3.3, §8 |
| **M4** `--internal` reaches the gateway; broker on `host` + another network; token in the build env; mode-dependent clauses | **Fixed, with one change to the recommendation.** The A1c probe set is the controlled-mode ship criterion, and the mechanism is left to P2 with candidates listed. The broker topology is corrected. The launcher clause and `JAVA_OPTS` are mode-dependent. Instead of a proxy-only sub-token, I chose **binding by network attachment**: one `--internal` network per invocation, with the broker and one build container. That puts no secret in the build env at all, and it avoids Java's default refusal of Basic proxy authentication on tunnels (from memory of JDK docs, not probed). | §2.3, §6, §7.2, §7.3.4 |
| **M5** `R2` is a host-wide tunnel | **Mitigations written, decision left open.** Only the runner fetches, with a fixed argv; read-only deploy credentials only; no hooks; a bare mirror; a `bytesOut` bound checked in 8.2. The residual risk and the TLS-interception alternative are stated as stakeholder decision **Q1** for P1. | §6.1, §8 8.2, §11 Q1 |
| **m1** `joern-parse --language` mapping | **Fixed, one stricter than suggested.** Explicit alias table; exactly one `--language`. **`java` is refused** rather than mapped to `javasrc2cpg`, because the gate's own reading is that its generator references `java2cpg.sh`/`jimple2cpg`, so the target frontend isn't one known script. | §7.3.1 |
| **m2** N1 fixtures and outcomes | **Fixed.** `interop_state` is defined: readability keyed on `status`; magic `4d5a`, extension `exe`, interpreter `/init`, and the `WSLInterop*` name; `microsoft` matched case-insensitively. There are 12 fixtures plus the host row, each with its expected result. §7.1.1 is self-sufficient and controls over the P0 plan's literal path (teco accepted open question 1). | §7.1.1, §10 test 8 |
| **m3** A4 memfd variant; C2 `profile=builtin`; `vsock` derivation | **Fixed as recommended.** | §7.2 A4(a′), A7; §7.1 C2; §5.3 |
| **m4** multi-codebase profile; no re-resolve; private ranges; CONNECT scheme; ssh | **Fixed, with one disagreement.** Bindings are per `(runId, repoId)` per invocation. No re-resolution. CONNECT → `https`. `ssh` remotes are rejected in v1. **Disagreement:** I did not adopt a blanket ban on RFC 1918/ULA. Workplace registries and IdPs commonly sit on private addresses, so a blanket ban would make the controlled path and `R4` unusable there. Private ranges need a per-entry `privateAddress: true`. Host, Docker and `local` addresses stay forbidden unconditionally. | §2.3, §4.2, §4.3 |
| **m5** 8.12 and 8.13 not computable | **Fixed as recommended.** 8.12 is restated as two record checks. 8.13 uses the new §7.4 step 5 lift and `flagCheck.liftedHits`, so `--jar4import-url` never needs to be in both lists. | §7.4, §5.3, §8 |
| **m6** test 2 `docker ps`; test 7 is text-only; `/sys` absent in namespace; executable names | **Fixed as recommended.** Test 2 uses `docker events … event=create`. Test 7 is labelled as the text half, with A6 as the runtime half. A1(a) uses `/proc/net/dev` on both backends. A frontend-name → executable mapping is added (`c2cpg.sh`, `jssrc2cpg.sh`). | §10, §7.2 A1, §7.3.1 |
| **n1** anchored exceptions | **Fixed.** The exceptions are full-match regexes, and test 2 includes near-misses. | §7.3.5 |
| Q6 / open question 3: bundle seam | **Fixed.** §5.3 states the loader check over every id in `sandboxAuditIds`, and the output-stem convention applies to tool outputs only. | §5.3, §11 R6 |

### 12.2 S04 gate Pass 2, S02 gate M1, and R10 (v1.2–v1.3)

Gates: `docs/reviews/connected-code-graph-p0-impl.md`, `## S04 (P0-03b)` → `### Pass 2` (verdict
**needs changes**, for P2-M1 only), and `## S02 (P0-02)` finding M1 (owner `architect` for the
spec side). Probe evidence is §9 H12–H14.

| Finding | Disposition | Where |
|---|---|---|
| **S04 P2-M1** / **S02 M1** — a `noexec` `/tmp` breaks every frontend; R3's `-J` fallback cannot fix it | **Fixed, with a different shape from the one recommended.** Both gates' diagnosis is confirmed. I did not adopt the narrow shape (`noexec` `/tmp` plus an exec `/tmp/native` reached through `JAVA_OPTS`). It passes `joern-parse`, but `joern --script`, which S11 needs, still fails: its forked JVM ignores `JAVA_OPTS` (H14). And with `TMPDIR` and `java.io.tmpdir` both redirected, the narrow shape buys almost no security over an exec `/tmp`, at the price of a per-tool hunt for extraction routes whose misses fail silently (H12). Instead, `/tmp` is `rw,exec,nosuid,nodev` with `exec` written out, `/home/ccg` stays `noexec`, `TMPDIR=/tmp`, `JAVA_OPTS` stays heap-only, the `-J-Djava.io.tmpdir` exception is gone, and R3 is resolved rather than made unconditional. The S04 gate named this shape as acceptable, and the S02 gate recommended it. | §7 interface, §7.1, §7.1.2, §7.2 A4, §7.3.4, §7.3.5, §10 tests 1/3/5, §11 R3 |
| P2-M1 sub-question: does jssrc's astgen need `TMPDIR`? | **Settled by probe: yes. The S02 gate was right**, and the S04 gate's "jssrc succeeded" was an exit-0 false green, with a CPG holding no methods (H12 shape 2). Under v1.2's shape it is moot, because `TMPDIR=/tmp` is exec. Consequence: test 3 now checks CPG content for all seven FR-1 languages, and R11 flags exit-0-means-nothing for the bundle pipeline. | §9 H12, §10 test 3, §11 R11 |
| **S04 P2-m1** — `python` mismapped; `java` refusal's reason wrong | **Fixed.** `python` is refused (it runs the absent `py2cpg.sh`, so exit 1). `java` stays refused, now as a redundant alias of the bytecode frontend `jimple2cpg`, whose name invites a source tree and yields an exit-0 CPG with no methods (H13). Mapping it would add a second name for `jimple2cpg` with a misleading one. `csharp`, `golang` and `javascript` are kept, and the reason is stated. New test 9 asserts that the five commercial scripts are absent. | §7.3.1, §9 H13, §10 tests 2/9 |
| **S04 P2-m2** — refresh-run structured calls can under-declare; 8.8 filters on declared codebases | **Fixed as recommended, plus the gap the fix opened.** `R8` and `R9` require `transport = proxy`. 8.8 selects by `runId` alone. Not in the recommendation: under the fix, `R9` could never match, because a binding derived only `dependency-download`. `bind_proxy` now takes the invocation's lifted `importNameServices` entries, and a request to one of those endpoints derives `import-name-lookup`. §3.3 now states why no other allow depends on a structured call's declaration in a refresh run (`R7` is pre-empted by `R5`), and 8.15 checks it in practice. | §2.3, §3.3, §6 step 2, §8 8.8, 8.15 |
| **S04 P2-m3** — conditions on network-attachment binding | **Fixed as recommended.** The binding is identified by the listening address **and** the peer IP. A1c(c) states the no-default-route check, and new A1c(f) checks for no cross-binding reach. There is a label-guarded reaper (with an age bound, so it cannot race a concurrent invocation) and a `backend-unavailable` refusal on pool exhaustion, never reuse. | §2.3, §7.2 A1c |
| **S04 P2-n1** — A6 reads its own environ; `JRE_CACERTS_PATH` | **Fixed as recommended.** A6(a) defines "its own" (`/proc/self/environ` for a compiled probe, `/proc/$$/environ` for a shell). New A6(e) bounds shell descendants to `envKeys` ∪ {`PWD`, `SHLVL`, `_`}. `JRE_CACERTS_PATH` is dropped from A6(d) and from the not-passed list, and R9 is resolved. | §7.2 A6, §7.3.4, §11 R9 |
| S04 Pass 2, m4 nit — the private-address flag on `decision` | **Fixed as recommended.** `decision.privateAddress`; 8.2's report lists those allows. | §5.2, §8 8.2 |
| Found while probing — `joern --script` is not launchable through the runner | Flagged as R10 in v1.2; **fixed in v1.3** (next row). | §11 R10 |
| R10 (v1.3, at `teco`'s request, so the re-gate sees it together with v1.2) — admit `joern --script` | **Fixed, as `teco` specified, with three additions.** `joern` joins the closed set in one form, `--script /scripts/<name>.sc [--param k=v]…`, with every other element refused. In-file directives are **refused rather than neutralized**: any line starting `//>` after leading whitespace gives `forbidden-directive`. The probe showed that `using dep` really attempts a download and `using file` pulls in code (H15); refusing keeps the reviewed bytes equal to the run bytes. The additions: (1) offline mode only; (2) the script is staged from a runner-owned copy, hashed into the new `sandbox.script`, and never read from `/src`; (3) `--workdir /tmp`, without which `joern` fails on the read-only `WORKDIR` (H15). `--dep`, `--repo` and `--verbose` join `forbidden-flags.txt` for the sweep. The §7.3.3 flag count, which was already wrong in v1.1, is corrected. New test 10; test 3's content check now runs through the runner. | §5.3, §7 interface, §7.2 launcher clause, §7.3.1, §7.3.3, §7.4, §9 H15, §10 tests 3/5/10, §11 R10 |
| R11 — exit 0 does not mean a parse produced anything | **Unchanged: still a flagged risk**, which `teco` is routing to S10/S11. | §11 R11 |
| §11 Q1 (M5, host-granular git tunnel) | **Unchanged: open for the stakeholder.** | §6.1, §11 Q1 |
