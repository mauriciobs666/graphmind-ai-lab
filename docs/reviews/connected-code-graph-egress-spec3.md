# Review: connected code graph egress spec 3 (P1-28 gate)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Pass 1 (2026-11-09)

### Scope and verdict

**Reviewed:** `docs/plans/connected-code-graph-egress-spec3.md` (uncommitted, 11,576 words), the P1-28 step it
implements. **Baseline:** P1-28 and P1-29 in `docs/plans/connected-code-graph-p1-gates.md`; index
`docs/plans/connected-code-graph-p1.md` §3, §4.2, §6, §7, §8; the sibling steps it touches (P1-03 in
`-p1-extraction.md`, P1-18/19/20 in `-p1-linking.md`, P1-25/26 in `-p1-answers.md`); the two superseded specs (I
checked only their header edits, which are header-only: `git diff` shows one changed line each); `cpg/sandbox/` as
built (`offline-run.sh`, `README.md`); root `REVIEW.md` (there is no `cpg/REVIEW.md`). I extracted the §4.2 schema
and ran 44 cases against it with `jsonschema` 4.26 (`cpg/.venv`). I read the host's interfaces and Docker
version (read-only). I read the Docker documentation on gateway modes. I sent no traffic and wrote to no database.

**Verdict: needs changes.** There are two blockers: the singleton lock and the socket are scoped differently (B1),
and the step graph has a P1-03 ↔ P1-29 cycle (B2). There are seven majors. The core is sound: the rule table, the
deny rows for option C, the closed reason vocabulary, the trust-boundary table and the schema (it rejects every
negative case it claims to reject, except a trailing newline). Most fixes are local edits. The cross-document edits
in M7 are `teco`'s list to dispatch.

CPG: considered, not relevant — no `cpg_*` graph covers `cpg/` (the loaded graphs are `cpg_falkorchat` and non-CPG workspaces); this is a spec review checked against the runner and sibling plans, read directly.

### Findings

#### Blockers

**B1. The singleton lock and the socket are scoped differently, so a second broker can take over a live socket.**
§2.0 puts the lock at `<auditDir>/broker.lock`, and `auditDir` can be changed per process with `CCG_AUDIT_DIR`.
The socket is fixed per user at `<home>/.local/state/ccg/broker/broker.sock`. The rule "unlinks a leftover socket
file only while holding the singleton lock" therefore protects nothing across audit dirs. Here is the case. A dev
broker is running. P1-40 then starts a scenario broker with `CCG_AUDIT_DIR=<scenario dir>`. The second broker gets
*its* lock, unlinks the first broker's live socket and binds its own. New clients now reach the second broker.
Tokens issued by the first broker come back `token-invalid`. The run's records are split across two `egress.jsonl`
files. I6 claims "the broker is a singleton", and that claim is false per user. (Traced from §2.0's text; not run,
since no broker exists.)
*Fix:* keep the lock beside the socket, in `broker/broker.lock` inside the `0700` directory. Alternatively, derive
the socket path from `auditDir`. Either way, one key must identify one broker. Add the case to test (15): two
brokers with different `CCG_AUDIT_DIR` and the default socket; the second must exit 69.

**B2. P1-03 and P1-29 form a cycle in the step graph.** §4.4 and §10.2 make P1-03 own a test that asserts its enum
equals `$defs/egressMode` of `cpg/ccg/schemas/egress-v1.schema.json`. §4.1 and §10.1 say P1-29 creates that file.
§2.0 and §10.2 also make P1-29 need P1-03 (it loads manifests through `load_manifest`). So P1-03 cannot go green
until P1-29 lands, and P1-29 cannot start until P1-03 lands. P1-03 could not ship that test skipped either, because
index D9 says a skipped test counts as unrun. (Traced through §4.1, §4.4, §10.1 and §10.2.)
*Fix:* choose one of two options. (a) P1-03 creates `egress-v1.schema.json` verbatim from §4.2 (it is a data file,
and P1-03 already owns `system-manifest-v1.schema.json`). P1-29 then only consumes it. (b) The enum-equality test
moves to P1-29, the later step. (a) is smaller and also removes M5's file-ownership churn.

#### Majors

**M1. Several AC-8 checks that are marked "computed" can never fail in the P1 scenario, and none of the rows
requires a witness.** Every row of §8.2 is a universal ("every allow with …"), so each one passes over an empty
set. Under P-e, every allow in the scenario is an `R1` allow. These checks are vacuous without saying so (traced
from §2.4a, §3.3, §5.3 and §8.1):
- **8.4, second sentence:** `sandbox.runId` is `null` by construction (§5.3; `offline-run.sh:571` writes the
  literal). "So no `vulndb` run reaches a sandbox" infers that from a constant.
- **8.10 and 8.11, "8.1/8.5/8.6 hold for that run":** a `refresh` run has no P1 caller and therefore no allows
  (§2.4a), and `sandbox` records cannot be attributed to a run (`runId: null`). Both clauses collapse into the
  global checks.
- **8.15 (i):** `R0` check 4 makes it true by construction. **8.15 (iii):** a `refresh` run has no allows.
- **8.2:** every clause except the `R2` one also runs over zero allows (the check text says so, but the status
  column says "computed").
- **8.7, 8.8, 8.9:** these rows are real only if at least one free-prose question reaches `interpret.py`. P1-26's
  deterministic parsers make no model call for a question they resolve, and nothing in §8 requires such a
  question.

Root `REVIEW.md` requires a check that can fail. *Fix:* add a witness precondition **P-g**. The scenario must
produce at least one `R1` allow with `component = answer.interpret`, at least one with `answer.summarize` if the
summary is enabled, and at least one non-refused `sandbox` record per local-only and per hosted-default
repository, or the run fails. Relabel 8.2, 8.4's second sentence, 8.10/8.11's "for that run" clauses and 8.15
(i)/(iii) as vacuous in P1, or delete them.

**M2. The negative controls contradict the run verdict, and the "unlogged/logged" split no longer means anything.**
§8.1 says each control "must appear as stated or the run fails". It then lists two detector connections that
"both fail the run". If they are injected into the scenario, the scenario can never pass. "Unlogged vs logged"
was ES1's `conn` ↔ `decision` join. Under option C every `conn` fails the run, so the join has no role. P1-31
still lists `cpg/sandbox/ac8/join.py` and `test_ac8_join.py` (`-p1-gates.md` P1-31). §2.3 keeps
`upstream.localPort` "for the AC-8 join", and §8.3 itself says the detector cannot see the broker.
*Fix:* say that the two detector controls run as a **separate control scenario**, or against a planted record
set, and that the main run must have zero `conn` records. Drop the logged/unlogged distinction. Tell P1-31
(§10.2) that the join is not built in P1. Keep `localPort` as a reserved field, or drop it with the deferred
path's input 9.

**M3. P-a names one mechanism that cannot work, and leaves out a documented one that may.** "A host-side
`DOCKER-USER` drop rule" cannot block container → host listeners. `DOCKER-USER` is called from the `FORWARD`
chain. Packets addressed to the bridge's own host address are delivered locally, through `INPUT`. (Inferred from
netfilter chain semantics; not probed, per the brief.) The first candidate, "an internal network created without
a host-reachable gateway, if this engine offers one", is documented. Docker's port-publishing page says gateway
mode `isolated` "can only be used when the network is also created with … `--internal`", and "No address is
assigned to the bridge when the network is created with gateway mode `isolated`". The default case is also
documented there: "containers in the network can access host services". Engine 29.7.2 is installed here; I did
not probe it. With no bridge address there is also no "gateway address" for P-a's probe to target.
*Fix:* name `-o com.docker.network.bridge.gateway_mode_ipv4=isolated` (and `_ipv6`, if IPv6 is enabled) as the
candidate to verify first. Restate P-a's probe target as every host interface address plus `docker0`'s, as A1(d)
does, rather than "the gateway". Replace the `DOCKER-USER` candidate with an `INPUT` rule on the scenario bridge,
or drop it. Lower R-E3 from "may need a host change" to "unverified, documented mechanism".

**M4. P1-29 and P1-31 cannot implement the container path from the text alone.** (Traced from §2.0, §2.1 and
§8.1.)
- **How a client finds the socket.** §2.0 derives the path from the *broker's* passwd home. A client in a
  container has a different home and passwd, and `--socket` is broker-side only. `BrokerClient` (§10.1) needs a
  stated discovery rule, for example `CCG_BROKER_SOCKET` with the default path as fallback.
- **The uid inside the containers.** The peer check rejects any uid other than the broker's, and the directory is
  `0700`. The answer containers must therefore run with `--user <broker uid>:<gid>` on a rootful engine without
  userns-remap. Under userns-remap or rootless Docker, `SO_PEERCRED` reports a shifted uid. Container default
  root, uid 0, is rejected. §8.1 states none of this.
- **Where the audit dir is.** The default `cpg/.audit/` is relative. The runner resolves it as
  `$SBX_DIR/../.audit` (`offline-run.sh:580`). The broker and `check_sandbox_ids` must resolve to the same
  directory, and §5.1 does not say against what (cwd, package or repository root).
- **A PATH dependency.** §2.0 says "no other environment input", yet §3.4 runs `ip -j -d address show`, which
  depends on `PATH`. Pin `/usr/sbin/ip` or read netlink.

*Fix:* add the four statements to §2.0, §5.1 and §8.1's topology row.

**M5. The spec's import rules contradict P1-25's no-network test, and it does not name one owner for opening the
answer run.** P1-25's `test_no_network.py` (`-p1-answers.md`, P1-25 behaviour 7) scans `ccg/refresh.py` and
`clients/`. It fails on "an import of `ccg.broker` outside `ccg/front/interpret.py`'s one broker client". The spec
has P1-20 (`refresh.py`) open a run (§2.1), and it has "the answer clients (P1-25, P1-26)" open runs and import
`BrokerClient` (§2.1, §10.2). Both of those fail P1-25's test as written. P1-26 already allows `summary.py` as
well, which is a third allow-list. §10.1 (19) then adds a second AST import-boundary test with a different scope
and different exceptions. Its "only P1-19's store modules" exception would also fail P1-25's `raw_query` client
if that client imports `falkordb` itself. Unlike P1-25's test, (19) names no mutants. Root `REVIEW.md` requires
one planted import per composition axis: import inside a function, `from … import`, `__import__`,
`importlib.import_module`.
*Fix:* keep one boundary test, P1-25's, extended. State a single allow-list for `ccg.broker.client` imports
(`refresh.py`, `front/interpret.py`, `front/summary.py`). Name one opener per question, the front's entry point,
so that an MCP `ask` that routes to the front opens one run, not two. Add "P1-25 needs P1-29" to §10.2 if P1-25
keeps any broker call. Otherwise remove P1-25 from §2.1 and §10.2.

**M6. Every refresh now fails unless a broker is running, and §10.2 does not tell the steps this affects.** §2.1
makes P1-20 open a `refresh` run for every refresh. §2.5 says a client that cannot reach the broker "fails … the
run". Refresh makes no egress in P1. The run exists only to record mode resolution. Several steps run refresh but
do not have the broker in their briefs: P1-32 (the AC-5 harness builds at `EVAL_COMMIT`), P1-37, P1-41, P1-42,
P1-47 and P1-39. Each of them now needs a running broker loaded with that system's manifest (`--system`), and
must avoid B1's collision. (Traced through §2.1, §2.5 and the index §7 Needs column.)
*Fix:* list this runtime dependency in §10.2 for each of those steps. Have P1-20's report name the `policy`
record's `auditId`. Whether a refresh without an override could skip the run is a design decision for
`architect`, not a fix I can back with a failing assertion: dropping the run loses 8.10/8.11's evidence only when
an override is given.

**M7. The cross-document edits in §10.2/§10.3 are real but incomplete.** I confirmed each requested change
against the text:
- **P1-29 needs P1-03:** real (§2.0). Index §7's row says `28`. Index line 366 says the broker core needs "only
  those" (00, 01, 02, 13, 28, 48). Both need editing.
- **P1-20 needs P1-29:** real (§2.1). Index row: `19, 44, 02, 03`.
- **P1-18/19 need P1-29:** real only because §5.4 puts `check_sandbox_ids` in `cpg/ccg/broker/audit.py`. That
  function reads runner-written `sandbox.jsonl`, not broker state. If it lives in a broker-free module (for
  example `cpg/ccg/sandbox_audit.py`, owned by P1-18, with the `sandbox` record's `$defs` in a schema file P1-18
  owns or P1-03 creates per B2(a)), the edge disappears, and so does the "lands first, then switches" clause.
  Consider that before adding the edge to the critical-path steps 18 → 19.
- **`policy.egress`, not `policy.localOnly`:** real (`-p1-extraction.md:75`). That same line also lists
  `endpointAllowList`, a second allow-list concept the spec does not mention. Say "removed: the `egress` section
  is the allow-list".
- **The schema is owned by P1-29, not P1-28:** this contradicts index §4.2 (line 165: "`egress-v1.schema.json` is
  P1-28") and `-p1-gates.md` P1-28's second bullet. §10.3 lists neither, and B2 may reverse the owner again.
- **Not listed anywhere:**
  - `-p1-gates.md` P1-29 Files, Behaviours ("rows of `-egress-spec.md` §8") and Blocks (add 20, 18/19 or per the
    alternative above, and 25 per M5).
  - P1-31 behaviour 4, which cites `-egress-spec.md` §7.2.
  - Index §8's policy bullet: "all egress through a logging proxy; `-egress-spec.md` §8".
  - Index §3.2 rejection (b): "proxy technology … decided in P1-28". §6 input 1 defers it, consistent with index
    §6.
  - `-p1-linking.md` P1-18 (7), which cites "egress §5.3".

*Fix:* replace §10.3 with a complete table (document, line or section, edit) so `teco` can dispatch it in one
pass.

#### Minors

**m1. The DNS lookup is egress, and it happens before the decision is durable.** §1 defines DNS as egress. I4
says a decision is durable before any upstream connection. §3.4 resolves after a rule allows but before the
`decision` record (which carries `resolvedAddr`) is written. So an allowed entry's DNS query is egress that
precedes its record, and it still happens when that record write then fails. The query is only the allow-listed
name, so the risk is low. *Fix:* state the exception in I4 ("the resolution of an allowed entry's host is part of
the allowed act"). Alternatively, write the `decision` before resolving and add a `resolvedAddr` field to `close`.

**m2. Python's schema validation accepts a trailing newline.** *Verified:* `jsonschema` uses `re.search`, and
Python's `$` matches before a final `\n`. So `paths: ["/v1\n"]` and `endpoint: "http://127.0.0.1:1234\n"` both
validate (2 of 44 cases; script in the Appendix). An entry like that cannot be matched, or crashes the client. It
is not an injection: `http.client` rejects control characters in the URL. But "rejected at load" is not true for
it. *Fix:* add **S5** to §4.3: every endpoint equals `canonical(endpoint)`, and no string in the section contains
a character ≤ U+001F or U+007F.

**m3. §5.1 and §2.5 disagree about the `close` record.** §5.1 says `close` is made durable "before the response
returns". §2.5 says a failed `close` write "returns the response and latches". Pick one; §2.5's version is the
safe and consistent one. Also state what an in-flight exchange does when another call trips the latch. I read it
as: finish, try `close`, return the response.

**m4. The order of `R0` and the deny rows interacts with test (11).** `R0` check 8 (`path-not-allowed`) runs
before `R2`/`R8`/`R9`. A `git-fetch` request to a `local` endpoint with a path not in the list is therefore
recorded as `R0`, not as `not-in-p1:git-fetch`. Test (11) must use an allowed path, and the AC-8 controls likewise.
Separately, §2.3 says a `repo-names-and-identity` body "must have the closed shape; any other field is a denial".
That is true only under `R4`: `R1` and `R7` admit any body. Say "under `R4`".

**m5. §7.7 does not describe the runner as built.** `offline-run.sh:705` also accepts `PWD`, `OLDPWD`, `SHLVL` and
`_` without re-executing. §7.7 says "exactly" five names plus `CCG_*`. Add the four names; the runner's behaviour
is the contract.

**m6. Growth-stops check (a new successor, so this is about restating rather than rework growth).**
- §0's "Material differences from ES1 + ES2" paragraph is a change list inside a current-design document. That
  belongs in the gate brief or this review. Delete it.
- §7 (2,434 words) restates the P0-gated runner. It is justified: P1-28 and index §6 mandate consolidation, and
  with ES1/ES2 superseded this section is the contract's only normative home. It is not justified where it
  repeats evidence: the parenthetical probe citations (H1, H12–H14, …) can go to §9's single pointer.
- §6's table is mandated (P1-28's first bullet) and cites rather than restates. That is fine.
- Keeping the deferred values (`transport: proxy`, `job: git-fetch`, `network: broker`) "so records stay
  `schemaVersion: 1`" is speculative. The deferred path adds fields anyway (input 9). This is a nit.

#### Nits

- §2.0 says "`flock -n`" (the CLI). The broker is Python, so write `fcntl.flock(LOCK_EX | LOCK_NB)`.
- §10.1 brings P1-29 to about 12 files and 19 behaviours, against index §7's "at most about five files" per
  dispatch. Moving §5.4 out (M7) and moving the schema to P1-03 (B2(a)) both help.

### What's solid

- **Option C is honoured.** Nothing of the fetch path remains as P1 text. `R2`/`R8`/`R9` are deny rows with
  reasons from the closed vocabulary, placed before `R1`, with the ordering argued (§3.3). `remote`, `registries`,
  `importNameServices` and `gitFetchMaxBytesOut` are rejected (I verified the latter three against the schema).
  The deferred inputs carry every item P1-28's first bullet names (§6 inputs 1–3, 6, 7).
- **The rule table.** The scope-set argument for `refresh` holds: `R7` needs `model-call`, and `R5` precedes it
  whenever `systemLocalOnly` is set. "Deny before DNS" holds for every deny row, because rule evaluation reads no
  address.
- **The trust-boundary table (§2.6)** says plainly what it does not cover.
- **The schema.** 42 of 44 cases behaved as claimed: unknown keys, `http` outbound, uppercase hosts, wildcards,
  user-info, a path in the endpoint, port 0, leading-zero IPv4, `privateAddress` on `local`, wrong `kind`.
  Out-of-range ports, junk IPv6 and duplicate ids pass the schema, as §4.3's S1–S4 intend.
- **The grounding claims checked out.** `sbx_write_record` writes `runId` as null (`offline-run.sh:571`).
  `docker0` reports `info_kind = bridge`. The engine is 29.7.2. `mode-not-implemented` is the runner's refusal
  (`:775`). The superseded headers are header-only edits.

### Open questions

- B2 needs a choice: does the schema file go to P1-03, or the test to P1-29? This affects M7's ownership edits.
  It is `architect`'s call, with `teco` scheduling.
- M6: should a refresh with no override still open a run, and so require a live broker for every build-using
  gate? It is a design call (`architect`), not a defect.

## Pass 2 (2026-11-09)

### Scope and verdict

**Reviewed:** spec v1.1 (uncommitted, 13,560 words), with the security advisory
(`docs/reviews/connected-code-graph-egress-spec3-security.md`) folded in. Also the cross-document edits, read as
`git diff` of the index, `-p1-gates.md`, `-p1-extraction.md`, `-p1-linking.md` and `-p1-answers.md`.

**Executed:**
- Re-extracted §4.2 and reran the 44 cases. The schema and the example are byte-identical to Pass 1, with the
  same 42/44 result.
- Wrote a reference reading of S5 and S6 from the §4.3 text and applied it to this host's `ip` output.
- Built the step graph from the index §7 Needs column plus every sub-plan `**Needs:**` line, checked it for cycles
  and computed the longest path (A2).
- Read `offline-run.sh:569/588/626`, the existing `cpg/.audit/sandbox.jsonl` (563 records) and the extras in
  `cpg/pyproject.toml`.

**Verdict: needs changes.** All but small residues of Pass 1's findings are fixed, and the step graph is acyclic.
The revision brings three new defects, each a narrow edit:
- S6 admits the LAN router as `local` on this host (P2-B1).
- P-a's probe fails on every run by construction (P2-M1).
- The `--network none` fallback cannot pass P-a or P-c (P2-M2).

A re-gate can be limited to those three and the minors below.

CPG: considered, not relevant — no `cpg_*` graph covers `cpg/`; the revision was checked against the runner, the audit file and the sibling plans, read directly.

### Dispositions of Pass 1 findings

- **B1: fixed.** The lock and the socket share the broker directory, and the broker refuses to start when a live
  socket answers a `connect()` (§2.0). Test (16) is the planted case.
- **B2: fixed.** P1-03 creates the schema file and the equality test (§4.1, §4.4; extraction P1-03 Files). The
  graph has no cycle: P1-29 needs 28 and 03, and its only ancestors are 01, 03 and 28 (A2).
- **M1: fixed, with residues P2-m3 and P2-n1.** P-g requires witnesses. 8.2, 8.4 and 8.15 (ii)/(iii) are
  relabelled vacuous, and §5.3 marks the runner's constants.
- **M2: fixed.** There is a separate detector control scenario, the main run requires zero `conn`, and there is no
  join (P1-31 now has `window.py`).
- **M3: fixed.** `gateway_mode_ipv4=isolated` is named first and `DOCKER-USER` is ruled out. The rewritten probe has
  a new defect (P2-M1).
- **M4: fixed.** `CCG_BROKER_SOCKET` is defined. The answer containers run with `--user <broker uid>`,
  `--cap-drop ALL`, a rootful engine and no userns-remap. `auditDir` is anchored to the package location and
  recorded. `/usr/bin/ip` is absolute.
- **M5: fixed, with a residue.** There is one test (P1-25's, widened) and one allow-list (`refresh.py`,
  `front/interpret.py`, `front/summary.py`). The front is the single opener, and the clients open no run (answers
  P1-25 diff). Residue: the P1-25 `raw_query` client must reach FalkorDB through a P1-19 store module, or the
  allow-list must name it. This is not stated (minor, carried).
- **M6: fixed.** The architect's decision: a refresh opens a run only when given an override (§2.1). I checked it
  against rows 8.10 and 8.11. Their evidence is the two override control records, which P-g requires, so nothing is
  lost. A default refresh performs no resolution, so I4 still holds. The §10.2 "unaffected" row is correct.
- **M7: fixed, with two gaps.** Every edit I listed is applied: index lines 36, 96, 165, 338, 349, 366 and 397,
  gates P1-28/29/31, extraction P1-03, linking P1-18/20, answers P1-25. Two gaps remain: P2-m1, and
  `-p1-extraction.md:8`, which still names `-egress-spec.md` + `-egress-spec2.md` as the sandbox contract. An
  unfiltered sweep of the P1 plans finds the remaining hits at index lines 104, 277, 316 and 489. Those are
  historical source citations and correct.
- **m1: fixed** (I4 states the DNS exception). **m2: fixed by S5.** The reference S5 refuses both newline cases
  and a non-RFC 5952 IPv6 literal, and passes the example (A1). **m3: fixed** (§2.5, §5.1). **m4: fixed** (§2.3
  "under `R4`", §3.3 note, test (11)). **m5: fixed** (§7.7 lists the four shell names). **m6: fixed.** §0's
  change list and §7's probe parentheticals are gone, and §5.1 restricts enums to values P1 can write.
- **Nits: fixed** (`fcntl.flock`). On the P1-29 dispatch size, the gates diff leaves it to `teco` ("re-checks the
  dispatch split"), which is acceptable.

**Decisions taken in the revision, checked:**
- **F-1(a): accepted.** `repoNames` is limited to the system's distinct repository ids. It is a closed value set,
  and the rationale (an access check names repositories outside the touched set) is stated. R-E6 records the
  256-character `subject` channel, per request and unbounded in count. That is honest, and P1 has no `R4` caller.
- **F-16: accepted, deferred.** §5.4 states presence-only, and §6 input 13 holds the digest design.
  `-bundle-spec3.md:959` claims only "one per invocation that produced this bundle's inputs", which is a writer
  obligation, not a check, so nothing contradicts it.
- **F-8 (hash chain) rejected: accepted.** It detects but does not stop an actor that §2.6 puts out of scope, and
  a crash breaks it. R-E7 records this.
- **F-7 (`sandbox.jsonl` hardening) rejected: accepted.** The rejection keeps "P1 changes no runner behaviour".
  The existing file is `0644` in a `0755` directory, which passes the broker's directory check (not group- or
  world-writable). R-E7 records the runner's handling.

### New findings

**P2-B1 (blocker). S6 admits off-machine addresses, the LAN router among them, as `local`.** S6 admits "a host
interface's default gateway (the WSL NAT route to the Windows host)". That gateway is the Windows host only under
WSL NAT. On this host (mirrored mode, `.wslconfig` `networkingMode=mirrored`) and on any native Linux host, the
default gateway is the LAN router. *Verified:* `ip -j route show default` reports `192.168.0.1` and `fe80::1`, and
the reference S6 admits `http://192.168.0.1:80` and `http://[fe80::1]:80` (A3). `R1` then sends `repo-derived`
bodies from local-only codebases in clear text to the router, and 8.1, 8.8 and 8.9 stay green. That is exactly
F-3's failure, which S6 exists to close. §1 ("checked to be on the machine") and §3.3's note claim the opposite. A
related gap: S6 is evaluated once, at broker start, so a DHCP-assigned LAN address that the host later loses
stays admitted.
*Fix:* admit a gateway only when WSL NAT is detected. One possible test: the kernel release contains `microsoft`,
the host has no `loopback0` interface (mirrored mode creates it; this host has one), and the gateway is
`/etc/resolv.conf`'s nameserver, which WSL NAT sets to the Windows host. Alternatively, drop the gateway clause and
have NAT-mode operators use mirrored mode. Either way, add the router case to P1-29's S6 test as the planted
input that must be refused, and re-run S6 before each `R1` connection with the same interface data §3.4 reads.

**P2-M1 (major). P-a's probe fails on every run, because it includes the host's loopback addresses.** P-a probes
"every address of every host interface … from `ip -j -d address show` on the host" and requires each probe to
"fail as in A1(d)", that is, with `ENETUNREACH`, `EHOSTUNREACH` or `EADDRNOTAVAIL`. Host `lo` carries `127.0.0.1`,
`::1` and, in mirrored mode, `10.255.255.254` (A3). Inside a container, `127.0.0.1` and `::1` are the container's
own loopback. A TCP connect there gets `ECONNREFUSED`, which is a failure under A1(d), so the precondition can
never hold. (Traced from §8.1 and §7.3's A1(d)/(e); not run.)
*Fix:* exclude `127.0.0.0/8` and `::1` from P-a's address set. The container's own loopback is not a host channel,
and A1(e)'s semantics already cover it. Keep every other `lo` address in the set, `10.255.255.254` included.

**P2-M2 (major). The `--network none` fallback contradicts P-a's positive control and P-c, and nobody owns it.**
Under the fallback, no TCP port accepts. P-a's positive control ("the store's port and the detector's ports as
accepting; if it does not, the probe is broken and the run fails") therefore fails the run. The fallback's
"Unix-socket store endpoint" is a second host Unix socket in each container. P-c requires the broker's socket to
be "the only host Unix socket visible", so P-c fails as well. No step builds Unix-socket support either: not P1-48
(instance flags), not P1-19 (store connection), not P1-25 (clients). (Traced from §8.1 against the extraction and
gates step texts.)
*Fix:* give the fallback its own preconditions:
- P-a′: A1 inside each container, with the store socket's `connect()` as the positive control.
- P-c′: exactly the broker's socket and the store's socket are visible.

Name the steps that add `unixsocket` to the throwaway instance and the store client. Alternatively, mark the
fallback "requires a plan revision", so P1-31 escalates rather than improvises (that matches R-E3's posture).

#### Minors and nits

- **P2-m1. P1-31 needs P1-18.** Its `assertions.py` validates against `sandbox-record-v1.schema.json` (8.14), and
  P1-18 creates that file. P1-31's only Need is 29, whose ancestors are 01, 03 and 28 (A2). Add 18 to the index
  row and to §10.2's P1-31 row.
- **P2-m2. The `broker` extra is still empty, but the broker needs third-party packages.** `cpg/pyproject.toml:52`
  has `broker = []` ("stdlib only unless P1-28 decides otherwise"). The broker loads YAML through `load_manifest`
  and validates every record against JSON Schema, so it needs `pyyaml` and `jsonschema`, which are only in `dev`.
  Under D8, P1-29 would stop and ask. Let §10.2 state the extra: `pyyaml` and `jsonschema`, at the `dev` pins.
- **P2-m3. 8.5's `netnsDistinct = true` is a constant for every launched record.** `offline-run.sh:626` sets it
  unconditionally once a launcher exists ("holds by construction"), which is the class §5.3 now marks for other
  fields. Base 8.5 on `launcherArgv` alone, or mark `netnsDistinct` constant in §5.3. Also, `--network none` exists
  only on the container backend. A namespace-backend record carries `--unshare-all` and would fail 8.5 and 8.6 as
  written. Either check the backend's own clause, or state that AC-8 runs on the container backend only, which is
  true on WSL2 (§7.1).
- **P2-n1.** 8.10's clause "that refresh's `sandbox` records satisfy 8.5 and 8.6 (they are within *W*)" is the
  global check under another name, because records carry no run id. Delete it.

### Growth-stops check

The document grew by 1,984 words. By section: §0 −89, §1 +57, §2 +587, §3 +118, §4 +66, §5 +200, §6 +53, §7 +4,
§8 +498, §9 +67, §10 +324, §11 +81.

- **Justified: new normative content.** Nearly all the growth specifies contracts that Pass 1 or the advisory found
  unspecified. In §2: the closed environment, the per-user singleton, client discovery, framing, the `R4` value
  bounds, the wire-level bounds and the component↔purpose table. In §3.4: the explicit forbidden set. In §4.3: S5
  and S6. In §5.1: the hardening rules and the `rejected` map. In §8: P-g, the controls and 8.16. In §10.1: tests
  (10)–(20), each tied to one rule. These are additions to scope that the gate required, not restated rework. The
  wording replaced the old text in place, and §0's change list was deleted.
- **Restated, to cut.** In §9, the bullet "From the reviews of this document" (about 45 words) narrates who ran
  what. The header's `Reviews:` field already cites both reviews, so cut it.
- **No section needs a structural cut.** §7 is still mandated consolidation, with its evidence parentheticals
  gone. §10.2's "Unaffected" row costs one line and saves `teco` a question, so keep it.
- **Revision note: compliant.** It is one dated line.

## Pass 3 (2026-11-10)

**Reviewed:** spec v1.2 (uncommitted, 13,795 words), the index §7 rows for P1-29 and P1-31, `cpg/pyproject.toml`
and `-p1-extraction.md:8`. I re-read the sites of the Pass 2 findings and ran `wslinfo --networking-mode` (it
reports `mirrored`). Nothing was written apart from this review.

**Verdict: approve with suggestions.** Every Pass 2 finding is fixed. One new minor (P3-m1) and one nit remain.

CPG: considered, not relevant — no `cpg_*` graph covers `cpg/`; the check reads the spec, the index and `pyproject.toml` directly.

**Dispositions:**
- **P2-B1: fixed.** S6 admits only `127.0.0.0/8`, `::1` and host-interface addresses, and never a default gateway
  (§4.3). §3.4 re-applies S6 before each `R1` connection and turns a failure into a deny. §10.1's P1-29 test plants
  this host's `192.168.0.1` and `fe80::1` as cases that must be refused, plus an address that is dropped after
  start. R-E8 declares WSL NAT unsupported and gives a reversal trigger based on `wslinfo`, which exists here and
  reports `mirrored`, as R-E8 says.
- **P2-M1: fixed.** P-a excludes `127.0.0.0/8` and `::1` and keeps `lo`'s other addresses (`10.255.255.254`). It
  probes link-local addresses with each container interface's scope id, so an `EINVAL` from a connect with no
  scope no longer counts as a false pass. That kernel behaviour is cited as not probed, which is acceptable.
- **P2-M2: fixed.** There is no fallback in P1: P1-31 escalates for a plan revision (§8.1, R-E3), and the text
  says why the `--network none` variant fails.
- **P2-m1: fixed.** Index row P1-31 needs `29, 18`, and §10.2 says so.
- **P2-m2: fixed in the spec; P3-m1 records the residue in the plan and the file.** §10.2 gives the `broker`
  extra as `jsonschema==4.26.0` and `pyyaml==6.0.3`, the `dev` pins verbatim, so no new pin is introduced. I agree
  with your reading that `devops` owns both steps.
- **P2-m3: fixed.** 8.5 and 8.6 check each backend's own namespace flag (`--network none`, or `--unshare-all`
  without `--share-net`). §5.3 marks `netnsDistinct` constant, and 8.5 says it proves nothing.
- **P2-n1: fixed.** 8.10's clause is gone.
- **M5 residue: fixed.** The import allow-list names `cpg/clients/common.py` for `raw_query`'s connection, beside
  `cpg/ccg/store/` and `engine/reader.py`.
- **M7 residue: fixed.** `-p1-extraction.md:8` now cites `-egress-spec3.md`.
- **§9 restatement: fixed.** The "From the reviews of this document" bullet is gone.
- **Same-revision cross-checks: consistent.**
  - The per-`R1` S6 re-check needs no network and reuses the deny reason `forbidden-address`, so §3.5's
    vocabulary is unchanged.
  - S6 refusing gateways agrees with §3.4's forbidden set for non-local entries.
  - With no fallback, P-a's positive control and P-c stand as written.
- **Growth-stops: compliant.** The document grew by 235 words. R-E8 is new scope, an explicit unsupported
  mode, and states its rejected alternative as a design reason. The remaining growth replaces text in place at the
  finding sites. The revision note is one dated line.

**P3-m1 (minor). Index D8 and `cpg/pyproject.toml` still forbid P1-29's edit.** Index D8 says
"`pyproject.toml` is edited once, by P1-01". The file's own comment (`cpg/pyproject.toml:9-10`) says "No other
step edits this file: a step that finds a dependency missing stops and asks `teco`", and line 53 keeps
`broker = []` with "stdlib only unless P1-28 decides otherwise". Taken literally, P1-29 must stop and ask before
making the edit §10.2 assigns it. *Fix:* add one clause to D8 ("and the `broker` extra, by P1-29, with the `dev`
pins of `-egress-spec3.md` §10.2"). In the same P1-29 edit, reword the file's comment and the line 53 comment, so
that the file does not contradict itself afterwards.

**Nit.** Index row P1-31 still reads "logging detector, join, rows 8.1–8.15". In P1 there is no join and the rows
run to 8.16 (§8.2): change it to "detector, window, rows 8.1–8.16".

## Pass 4 (2026-11-10)

**Reviewed (scoped):** spec v1.3 (14,382 words). Sections: §7.4's closed set and its "Indexer entries" bullet;
the §7.6 additions; §7.8's `.scip` sentence; §10.1 tests (21)–(25); §10.2's P1-48 row; R-E9. Also their
interaction with §7.1–§7.3 and §7.5 (unchanged), and with Pass 3's fixes.

**Checked against:**
- the uncommitted `cpg/infra/toolchain/Dockerfile` stage `indexer` and its `README.md`;
- the index row for P1-48 and `-p1-extraction.md` P1-48;
- the locally built `ccg-toolchain-indexer:73b9fb76bd2a`. I ran both indexers' `index --help`, and one
  `scip-typescript` argument-parse case, under `--network none --read-only --cap-drop ALL` (A4).

I did not touch 6379 or 6390, and no store was involved.

**Verdict: needs changes**, on one major (P4-M1). It is a two-clause edit to the grammar. Everything else in the
change is sound or minor.

CPG: considered, not relevant — no `cpg_*` graph covers `cpg/`; the indexer entry was checked against the image and the tools' own `--help`, executed offline.

**Pass 3 items:** unaffected. P3-m1 and the P1-31 row nit are outside this scope and were not rechecked.

**What holds (verified):**
- `/opt/indexer/bin/scip-{python,typescript}` exist as symlinks into the lockfile-installed `node_modules`.
- Both tools take `--cwd`, `--output` (absolute paths are honoured) and, for `scip-python`, `--environment`.
- The image's `ENV` sets only `PATH`, so A6(d) is unaffected.
- The default `toolchain` image has no `/opt/indexer` (README, test section 3).
- Keeping §7.2 unchanged works: nothing the indexers need executes from `/src`, and `/tmp` stays the one exec
  mount.
- §0's and §7's "P1 changes only …" now say what P1 changes, so the earlier "changes no runner behaviour" claim no
  longer contradicts P1-48.
- Growth of 587 words: new scope (the indexer entries), not rework.

**P4-M1 (major). The `index` grammar admits an option one tool rejects, and does not fix the working directory.**
- *Verified:* `scip-typescript` has no `--project-name`. Running
  `scip-typescript index --cwd /tmp --project-name p --output /tmp/x.scip` prints
  `error: unknown option '--project-name'` and exits 1. §7.4 admits `--project-name` for both tools. Test (22)'s
  one positive launch case is `index --cwd /src --project-name p` under `scip-typescript`, so it cannot pass.
- *Inferred from `--help`:* `--cwd` is optional in the grammar, and both tools default to `/work`, the image's
  `WORKDIR`. §7.3's launcher clause sets `--workdir /tmp` only for `joern`. So an `index` with no `--cwd` indexes
  the image's `/work`, not `/src`. That would plausibly give a successful run with an empty index: a wrong result
  with nothing refused.

*Fix:*
- Admit `--project-name` for `scip-python` only, and add `--project-name` on `scip-typescript` to test (22)'s
  `forbidden-flag` cases.
- Make `--cwd <p>` mandatory for both tools; a missing one is `usage`, like the missing `--environment`. The
  alternative is for the runner to append `--cwd /src` itself.
- Change test (22)'s launch case to `index --cwd /src` for `scip-typescript`, and add `scip-python index --cwd /src
  --environment /src/env.json --project-name p`.

**P4-m1 (minor). §7.5 has not caught up with the indexer entries.**
- §7.5 (4) still limits the output refusal to "a parse frontend", and the `usage` list does not name a missing
  mandatory indexer element. Extend both.
- Test (24)'s sweep cannot launch `index --help` through the runner. The §7.4 grammar makes `--help`
  `forbidden-flag`, which test (22) asserts for the similar case `--version`. State that the sweep uses the
  test-only launch path (`lib-launch`, §5.3).
- On today's `--help` the sweep finds no hits: no flag of either tool matches the classification regex. Meanwhile
  `scip-typescript`'s `--pnpm-workspaces` and `--yarn-workspaces` run a package-manager listing. That is inferred
  from the flags' purpose, and the closed grammar already refuses both. Say that for indexers the grammar is the
  control and the sweep is a drift alarm. Optionally, add `pnpm|yarn|npm|pip|workspace` to the regex, so that test
  (24)'s planted unclassified flag is a realistic one.

**P4-m2 (minor). P1-48 needs P1-28.** P1-48 now implements §7.4 v1.3, but the index row for P1-48 lists Needs
"—". Add 28. Also, P1-48's file list is now about ten files across image, instance and runner. It is `teco`'s call
whether to split the runner part into its own dispatch, as P1-29's was.

**P4-m3 (minor). The staging location for dependencies is not stated.** "Staged by the caller into the staged
source directory" does not say whether that is the same directory a Joern parse of the unit reads. If it is,
`jssrc2cpg` or `pysrc2cpg` would parse `node_modules` or a venv. State that an indexer run gets its own staged
directory, and that a parse frontend never sees provisioned dependencies.

**Nits.**
- "Never a `PATH` lookup" holds for the indexer's own path. Both entry points start `#!/usr/bin/env node`, though,
  so the interpreter is found on the image's `PATH`. Reword the sentence.
- The runner could refuse an indexer frontend when the image named by `CCG_TOOLCHAIN_IMAGE` lacks the label
  `ccg-toolchain-indexer=1` (Dockerfile:455). Without that, the default image fails at exec with a tool error,
  instead of with a refusal that is recorded as such.

## Pass 5 (2026-11-10)

**Reviewed (scoped):** spec v1.4 (14,950 words): §7.4, §7.5, §7.6 with A6(b), §7.8, §10.1 tests (21)–(26), the
§10.2 P1-48b row and R-E9/R-E10. Also the index rows P1-15/48/48b, §3.1, the core-freeze list and N-DEVOPS-1,
and `-p1-extraction.md` P1-48/48b/15.

**Executed, offline:** every container run was on `ccg-toolchain-indexer:73b9fb76bd2a` with `--network none
--read-only --cap-drop ALL` (A5). No store was involved.

**Verdict: needs changes**, on one major (P5-M1). Its fix is local to test (26). The grammar, the step edits and
the other Pass 4 fixes are sound.

CPG: considered, not relevant — no `cpg_*` graph covers `cpg/`; the indexer change was checked by running the image's tools offline.

**Dispositions:**
- **P4-M1: fixed.**
  - `--project-name` is `scip-python` only.
  - `--cwd` is mandatory for both tools; a missing one is `usage`.
  - `--opt=value` forms are refused.
  - The path rule closes `/srcfoo`, `//`, `.` and `..`.
  - *Verified:* `scip-typescript index --cwd /tmp/src --output …` exits 0 and writes the index. Each option the
    grammar admits exists in that tool's `--help`, and none it refuses for a tool is needed. The tool itself accepts
    `--cwd=/tmp/src` (exit 0), so refusing the `=` form is the runner's work and not redundant.
- **P4-m1: fixed.** §7.5 (4) covers indexers, and the `usage` list names the missing mandatory element. Test (24)
  runs via `lib-launch`, and the grammar is stated to be the control.
- **P4-m2: fixed.** P1-48b (index row) needs 28 and 48, and P1-15 needs 48b. P1-48b is in the core-freeze list.
  "49 steps" (index line 64) counts 48b. N-DEVOPS-1 names it. There is no cycle, because 48b's ancestors are only
  28 and 48.
  - Nit: index §2's step-id paragraph still says ids are two-digit `P1-NN`. Add `48b`.
- **P4-m3: fixed.** An indexer run gets its own staged directory, and the P1-15 text agrees.
- **Nits: fixed.** The `PATH` wording is corrected. The image-label check is specified: the indexer image carries
  `ccg-toolchain-indexer=1` (I inspected it), and the default image does not.
- **Security M1 (`PYTHONSAFEPATH`): sound as a mechanism, but its premise is false at the pinned version.** See
  P5-M1 and P5-m1.
- **Dependency provisioning (pending stakeholder decision): no contradiction.** "How dependencies are obtained is
  not specified here" sits beside install-mode constraints (`--no-deps`, `npm ci --ignore-scripts`), which say how
  dependencies are installed, not where from, and P1-15 says "pre-provisioned".
  - Suggestion: add one clause until the decision lands, saying that provisioning is not performed by the system's
    own code. A `cpg/` module running `npm ci` or `pip install` would contradict I2 ("the broker is the only
    component of ours that opens a connection for egress"), and P1-25's boundary test names `curl`/`wget`
    mutants, not `npm`/`pip`.
- **Growth-stops: compliant.** The document grew by 568 words: new scope (the indexer grammar, tests (26), R-E10)
  plus rework at the finding sites. The trace narrative in R-E9 restates more than it needs to (P5-m1).

**P5-M1 (major, verified). Test (26) cannot fail. At `scip-python` 0.6.8 a staged `json.py` is never imported, with
or without `PYTHONSAFEPATH`.** I captured the probe `scip-python` runs (A5). It is `python3 -c` with the staged
tree as working directory, and it removes `""` and the working directory from `sys.path` before
`import sys, json`. I planted `json.py`, `os.py`, `posixpath.py`, `re.py`, `sitecustomize.py` and
`usercustomize.py` in the working directory. Each writes a marker. No marker appeared, either from a real
`scip-python index` run or from the captured script re-run, with the variable or without it. So test (26)'s
mutant (the variable removed from the pass-list) survives. Root `REVIEW.md` requires that a check can fail.
The variable itself works on a generic `python3 -c`. Without it, a `json.py` in the working directory is imported
(the planted file raised its own `SyntaxError`); with it, `sys.path[0]` is `/usr/lib/python312.zip`. It is
therefore a sound guard against tool drift, but the end-to-end test is the wrong oracle.
*Fix:* replace (26) with a `lib-launch` probe. Run `python3 -c 'import json'` with working directory `/src`, which
holds a marker-writing `json.py`, under the indexer pass-list. Expect the marker absent with the pass-list as
specified, and present with the mutant pass-list. I ran both halves by hand (A5). A6 and test 7 already pin the
key's presence.

**P5-m1 (minor, verified). Part of R-E9's trace is wrong.** R-E9 says that "without `PYTHONSAFEPATH=1` a staged
`json.py` … executes"; P5-M1 shows it does not at 0.6.8. It also says `scip-python` uses `child_process` "for `pip
list` only when `--environment` is absent". With `--environment` given, I logged two children: `python3 -c …` and
`git rev-parse HEAD` (cwd = the staged tree, A5). `git` reads the staged tree's `.git` and opens no connection.
*Fix:* restate R-E9 at its measured strength: the probe sanitizes `sys.path`, `PYTHONSAFEPATH` guards drift, and
the children are `python3` and `git`. Add `git` to the list that a `truth-npm/` bump must re-trace. Separately,
`scip-python`'s fatal `TypeError` on a hand-written `env.json` (A5) is evidence for R-E9's "format unmeasured"
line. It belongs to P1-15's first measurement, not to this gate.

## Pass 6 (2026-11-10)

**Verdict: approve.** P5-M1 is fixed and verified. I ran test (26)'s two halves in the image, offline (`--network
none --read-only --cap-drop ALL`): `python3 -c 'import json'` with working directory set to a directory holding a
marker-writing `json.py` gives `with var: absent` and `without var: PRESENT`, so the mutant is killed.
CPG: considered, not relevant — no `cpg_*` graph covers `cpg/`; verified by running the image offline.
- **P5-m1: fixed.** R-E9 now states the measured children (`python3`, `git`) and that the probe strips the working
  directory, and it puts `git` on the re-trace list.
- **Provisioning suggestion: applied.** Provisioning is an operator step outside the system, and no `cpg/` module
  runs `npm`, `pip` or `uv`. That is consistent with I2 and with P1-15's text.
- **Step-id nit: fixed.** Index §2 covers `48b`.
- **Not rechecked here:** P3-m1 (the wording of D8 and the `pyproject.toml` comments) and the stale P1-31 index row
  were not in this scope and remain open for `teco`.

## Appendix

**A1. Schema check.** I extracted the schema to `<scratchpad>/egress-v1.schema.json` and the example to
`example.yaml`, then ran `cpg/.venv/bin/python -I check.py` (Draft 2020-12, `check_schema` passed). The example has
no errors. These were unexpected:

```
valid   expect invalid path trailing newline
valid   expect invalid endpoint trailing newline
unexpected: 2
```

These passed the schema as intended, left to S1–S4: port 65536 and 99999, `[:::.]`, `0.0.0.0`, a duplicate id
across sections. `https://foo.localhost:443` also passes as a remote entry, and §3.4's loopback ban catches it at
resolution.

**A2. Step graph (Pass 2).** I ran `<scratchpad>/graph.py`. It parses the index §7 table's Needs column and every
`**Needs:**` line in `-p1-{answers,extraction,gates,linking}.md`, attributing each to the nearest preceding
`## P1-NN` heading. No sub-plan line adds an edge that the index lacks. The output:

```
29 needs ['P1-03', 'P1-28']      20 needs ['P1-02', 'P1-03', 'P1-19', 'P1-29', 'P1-44']
31 needs ['P1-29']               29 ancestors ['P1-01', 'P1-03', 'P1-28']
cycles []                        25 transitively needs 29: True (via 20)
longest 19: 01 03 04 05 07 10 46 17 18 19 21 22 23 39 25 32 36 37 43
```

P1-29 is not on the longest path, so the new edges add no length. The graph is only as complete as the Needs
columns: a dependency that appears only in prose, like P2-m1's, is invisible to it.

**A3. S5/S6 reference reading (Pass 2).** I wrote `<scratchpad>/p2/sem.py` from the §4.3 text and ran it with
`python3 -I` against this host's `/usr/bin/ip` output:

```
S5 path-newline: ["control char in '/v1\n'"]
S5 endpoint-newline: [control char, non-canonical]
S5 clean example: []         S5 [0:0:0:0:0:0:0:1]: non-canonical
default gateways on this host: ['192.168.0.1', 'fe80::1']
S6 127.0.0.1 admitted   192.168.0.64 admitted   192.168.0.1 admitted   [fe80::1] admitted
S6 203.0.113.7 refused  172.17.0.1 admitted
```

Host `lo` carries `127.0.0.1/8`, `10.255.255.254/32` and `::1`, which is the input to P2-M1.

**A4. Indexer checks (Pass 4).** All runs used the image `ccg-toolchain-indexer:73b9fb76bd2a` (id `fd3a8716a5f0`)
under `docker run --rm --network none --read-only --cap-drop ALL --entrypoint /usr/bin/env … -i
PATH=/opt/indexer/bin:/opt/indexer/node/bin:/usr/bin:/bin HOME=/tmp`.
- **`scip-python index --help`:** options `--project-name`, `--project-version`, `--project-namespace`,
  `--cwd` (default `/work`), `--target-only`, `--output` (relative to `--cwd`), `--quiet`,
  `--show-progress-rate-limit`, `--environment`, `--dev`, `-h`.
- **`scip-typescript index --help`:** usage `index [options] [projects...]`; options `--cwd` (default `/work`),
  `--pnpm-workspaces`, `--yarn-workspaces`, `--yarn-berry-workspaces`, `--infer-tsconfig`, `--output`,
  `--progress-bar`, `--no-progress-bar`, `--no-global-caches`, `--max-file-byte-size`, `-h`. There is no
  `--project-name`.
- **`scip-typescript index --cwd /tmp --project-name p --output /tmp/x.scip`:** printed
  `error: unknown option '--project-name'` and gave `exit=1`.
- **Entry points:** both resolve to scripts whose first line is `#!/usr/bin/env node`. The image `PATH` starts with
  `/opt/indexer/bin:/opt/indexer/node/bin`.

**A5. Indexer and `PYTHONSAFEPATH` runs (Pass 5).** All runs used the image `ccg-toolchain-indexer:73b9fb76bd2a`
(Python 3.12.3) with `--network none --read-only --cap-drop ALL --tmpfs /tmp:rw,exec`, under `env -i`.
- **The probe `scip-python index --cwd /tmp/src --environment /tmp/src/env.json --project-name p` runs.** I put a
  logging shim ahead on `PATH` to record it. The probe passed to `python3 -c` is:
  `import os, os.path, sys; … orig_sys_path = [p for p in sys.path if p != ""]; sys.path[:] = [p for p in sys.path if p != "" and normalize(p) != cwd]; import sys, json; json.dump(...)`.
- **Children logged:** `git cwd=/tmp/src args=rev-parse HEAD` and `python3 cwd=/tmp/src args=-c …`.
- **Planted modules in cwd:** `json`, `os`, `posixpath`, `re`, `sitecustomize` and `usercustomize`, each writing a
  `/tmp/HIT_<m>` marker. Re-running the captured script gave no hits with `SAFE=0` and no hits with `SAFE=1`.
  Separately, the full `scip-python` run with a marker-writing `json.py` gave `marker-not-executed` in both
  settings. That run itself exited 1 with `TypeError: Cannot read properties of undefined (reading 'indexOf')` on a
  hand-written `env.json`.
- **Generic `python3 -c "import json"` in a cwd holding `json.py`:**
  - without `PYTHONSAFEPATH`, the cwd `json.py` was imported: it raised its own `SyntaxError`;
  - with `PYTHONSAFEPATH=1`, it was not imported, and `sys.path[0]` was `/usr/lib/python312.zip`.
- **`scip-typescript`:** `index --cwd /tmp/src --output /tmp/o.scip` on a one-file TS project gave exit 0 and a
  302-byte index. `index --cwd=/tmp/src …` also gave exit 0, so the tool accepts the `=` form.
- **Image labels:** `ccg-toolchain=1` and `ccg-toolchain-indexer=1` are both present.
