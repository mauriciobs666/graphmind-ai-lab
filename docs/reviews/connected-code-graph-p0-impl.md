# C-1001 P0 implementation review

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (M10)

Per-step implementation gates for `docs/plans/connected-code-graph-p0.md`. Each P0 step's review
appends its own `## S<n> (P0-nn)` section; a re-review of a step adds a `### Pass N` under it.

## S01 (P0-01)

**Reviewed (2026-10-29):** `cpg/infra/ccg-falkordb.sh` and `cpg/infra/README.md` (both
untracked), plus the files the script generated in `~/.config/ccg/`, checked against
`docs/plans/connected-code-graph-p0.md`: the §3 S01 row, §2 D7, §4.1, §5.1, and the S01 test-strategy row.

**Verdict: approve with suggestions.** The instance meets D7 and every part of the S01
done-condition I could check without stopping it. One major finding should be fixed before
any later step runs `down`: M1, the `down` and `up` guards can be pointed at another container.

CPG: considered, not relevant — the artifact is a Bash ops script and its README, with no
Joern-built graph of `cpg/infra/`. It was reviewed by reading the code and by read-only runs
against the live instance.

**What I ran.** Everything below is read-only.
- `ccg-falkordb.sh status`, twice. Both runs exited 0.
- `ccg-falkordb.sh up` against the running instance. The code path for that case only checks the
  state and sends `PING`. Container Id and `StartedAt` were the same before and after.
- `docker port` and `docker inspect` on both instances, `docker stats --no-stream`, and `free -m`.
- Through `redis-cli`: `CONFIG GET` (requirepass length only), `ACL USERS`,
  `GRAPH.CONFIG GET TIMEOUT_MAX`, `INFO keyspace` and `MODULE LIST`.
- Password-leak scans with `/usr/bin/grep -F -f <passfile>`, so the pattern never appeared on
  argv. They covered `docker inspect`, `docker logs`, every `/proc/*/cmdline`, the `status`
  output, the README and the script.
- Three redis-cli behaviour probes against `falkordb-dev`: `AUTH` with a bogus password, followed
  by `PING` and `CONFIG GET`, and an unknown command to read redis-cli's exit code.
- `bash -n` on the script.

I also ran one `docker exec ccg-falkordb true` that I didn't need. It wrote nothing.

**What I did not run.** Per the brief, I did not execute `down` or `reset-p0`. I reasoned about
them from the code only. This means the done-condition items "`down` is idempotent" and "`down`
then `up` restores it" are **not verified by me**. The `ccg_smoke` round trip happened before
this review. The only evidence I have for it is `DBSIZE` = 0 now, which fits with the key having
been deleted.

### Q1 — D7 and the S01 done-condition

| Item | Result | Evidence |
|---|---|---|
| Binding only `127.0.0.1:6390` | **verified** | `docker port ccg-falkordb` → `6379/tcp -> 127.0.0.1:6390`, one line, no `0.0.0.0` or `[::]` (`falkordb-dev` shows both) |
| `noeviction` / `1536mb` / `save ""` / `appendonly no` | **verified** | `status`: `maxmemory 1610612736`, `noeviction`, `save ""`, `appendonly no`, "config: matches D7" |
| Image pinned `v4.18.11`, module version printed (K-P0-9) | **verified** | `.Config.Image` is `falkordb/falkordb:v4.18.11`; `MODULE LIST` shows graph ver 41811, the same version and args as `falkordb-dev` |
| Password file mode 600, outside the repo | **verified** | `stat` gives 700 for `~/.config/ccg`, 600 for `falkordb-admin.pass` and 600 for `ccg-falkordb.redis.conf`; requirepass is 48 chars |
| `up` / `status` idempotent | **verified** | same Id and `StartedAt` (`2026-10-29T19:46:47Z`) before and after `up`; `status` gave rc 0 on both runs |
| `ccg_smoke` round trip, then the key deleted | consistent with the claim, not re-run | `DBSIZE` 0, `INFO keyspace` empty |
| `down` idempotent; `down` → `up` restores | **not verified** (brief forbids running it) | the code looks right for the default name and port (see M1 for overrides) |
| `falkordb-dev` unchanged | **verified** | `StartedAt 2026-10-28T20:20:01Z` (up 23 h), mapping and volume as before |
| No ACL users (§4.1) | **verified** | `ACL USERS` → `default`; `TIMEOUT_MAX` 0 |
| Password never in argv, `docker inspect` or output | **verified** | 0 hits in every scan listed above; `Config.Env` holds only `BROWSER=0` and `REDIS_ARGS=/etc/ccg/redis.conf` |

### Findings

#### M1 (major) — `down` and `up` act on any container `CCG_FALKORDB_NAME` names; the `down` guard checks a different instance from the one it removes

*Traced in code, not executed.*

- `cmd_down` (`ccg-falkordb.sh:196-213`) stops and removes the container called `$NAME`. Its
  key-count guard, though, asks `cli DBSIZE`, and `cli` connects to `127.0.0.1:$PORT` (`:107`),
  not to whatever `$NAME` publishes. So `CCG_FALKORDB_NAME=falkordb-dev ccg-falkordb.sh down`
  works like this today:
  1. The state is `running`.
  2. `DBSIZE` goes to `ccg-falkordb` on 6390, which holds 0 keys right now.
  3. The guard passes, and the script runs `docker stop falkordb-dev` and then
     `docker rm falkordb-dev`. No `--force` is needed.
- The `falkordb-data` volume would survive, but the shared service, its `0.0.0.0` and port-3000
  publication, and every other session's connection would not. A stopped container of any name
  gets removed with no check at all, because the guard runs only when the state is `running`.
- `cmd_up`'s recreate branch (`:133-137`) does the same thing to any *non-running* container
  called `$NAME`.
- The script's own invariant (`:16`, "Never touches any other container") is therefore false.
  The README's sentence "`reset-p0` ignores any override" suggests the other verbs are safe,
  which they are not.
- The script already sets `--label ccg-falkordb=1` (`:150`) and never reads it back.

**Suggested improvement:**
- Add an `own_container` check that refuses unless
  `docker inspect -f '{{index .Config.Labels "ccg-falkordb"}}' "$NAME"` is `1`. Call it before
  every `docker stop` or `docker rm`, in `up`'s recreate branch and in `down` whatever the state,
  and in `reset-p0` too.
- In `down`, take the port to probe from `docker port "$NAME" 6379/tcp` instead of `$PORT`, so
  the guard counts the keys of the instance it is about to remove.
- Write the regression check as a **separate test written first**, a small stub-`docker`/stub-`redis-cli`
  harness on `PATH` (no real Docker). Run it red against the current script before the fix
  lands. Don't add it as an assertion inside the fix itself.
- Make it a **coverage probe over the axes**, not a list of cases:
  - verb: `up`'s recreate, `down`, `down --force`, `reset-p0 --yes`;
  - override: none, `NAME`, `PORT`, `CONFIG_DIR`, and the combinations;
  - target state: running, exited, absent;
  - label: present, absent.
- It asserts that no `stop`, `rm` or `FLUSHALL` is ever sent to an unlabelled container, and
  never to `falkordb-dev`.

#### m1 (minor) — `status` passes an instance that has no password, a mismatched image, or a missing binding

*`requirepass` behaviour verified, the rest traced.* `status` checks only the four D7 values
(`:177-188`).

- **No password.** I ran `REDISCLI_AUTH=bogus redis-cli … PING` against the password-less
  `falkordb-dev`. It printed "AUTH failed" on stderr, then `PONG`, and exited 0, and
  `CONFIG GET` answered normally. So an instance started by hand or by an older script, with
  D7's four values but no `requirepass`, reports "config: matches D7".
- **Image.** The image and module version are printed but never compared. Under
  `CCG_FALKORDB_IMAGE=…:latest`, `up` starts a different image and `status` still passes,
  which is K-P0-9's risk.
- **Binding.** The binding test (`:173`) uses `grep -qv`, which exits 1 on empty input
  (checked with GNU grep). An instance with no published port gets no `DRIFT` line, and the
  later `cli` failure exits through `set -e` with only redis-cli's connection error.
- **Label.** The label isn't checked either.

**Suggested improvement:**
- Add `DRIFT` checks for the following, each counted in `rc`:
  - an unauthenticated `redis-cli -h 127.0.0.1 -p $PORT PING` must reply `NOAUTH` (this tests
    that auth is on without the password passing through a variable);
  - `.Config.Image` must equal the pinned default, not `$IMAGE`;
  - the module `ver` must be `41811`;
  - the label must be `1`;
  - `docker port` must return exactly one line (`[ "$(docker port "$NAME" | wc -l)" = 1 ]`
    alongside the existing match).

#### m2 (minor) — `status` doesn't check §5.1's exit-hygiene state

§5.1 makes each exclusive step (S21, S22) restore `TIMEOUT_MAX` to 0 and delete its probe ACL
users before releasing its reservation. `status` is the obvious tool for `teco` to check that
at release time, but it reports neither. Both are clean right now (`TIMEOUT_MAX` 0, `ACL USERS`
= `default`).

**Suggested improvement:** print `GRAPH.CONFIG GET TIMEOUT_MAX` and `ACL USERS`, and flag a
non-default value as `DRIFT`.

#### m3 (minor) — `reset-p0` can report "done" when nothing was flushed, and it empties only the keyspace

*Exit codes verified; the rest traced.* redis-cli 7.0.15 exits 0 on an error reply. I checked
this two ways:
- an unknown command against `falkordb-dev` exited 0;
- `REDISCLI_AUTH=wrong … DBSIZE` against 6390 printed `NOAUTH Authentication required.` and
  exited 0.

With a wrong password (for example a stray `CCG_CONFIG_DIR`), `cli FLUSHALL >/dev/null`
(`:224`) fails silently. The script then logs `done: NOAUTH… key(s) after`. That fails in the
safe direction, but the message is misleading.

Separately, `FLUSHALL` leaves ACL users and `GRAPH.CONFIG` changes in place. So §5.1's
"P1 starts clean" holds for the keyspace only.

**Suggested improvement:**
- After the flush, require `cli DBSIZE` to equal `0` exactly, or `die`.
- Either make `reset-p0` also assert the m2 defaults, or state in the README that a full clean
  start is `down --force` followed by `up`. That route works because nothing persists, including
  config.

#### n1 (nit) — small robustness gaps

- `preflight` doesn't check `openssl` or `ss`. If `openssl` is missing, `up` aborts and leaves
  an empty password file, which the next run regenerates because of the `-s` test. If `ss` is
  missing, `port_in_use` always answers "free", and `docker run` then fails loudly on the bind.
  Neither is unsafe.
- `down`'s guard counts db 0 only (`DBSIZE`). `INFO keyspace` covers every db.
- A `paused` container makes `up`'s plain `docker rm` fail. That is loud and harmless.

### Q2 — can the safety guards be got around?

- **`reset-p0`:** no practical way around it (traced).
  - It checks the literal `$NAME = ccg-falkordb` and `$PORT = 6390` *before* doing anything
    (`:217-218`), so neither override can move it.
  - It then requires `docker port` to show exactly `6379/tcp -> 127.0.0.1:6390` for that
    container (`:221`). `docker rename falkordb-dev ccg-falkordb` fails that check, because the
    mapping would be `0.0.0.0:6379`.
  - While the real container holds 6390, docker-proxy owns `127.0.0.1:6390`, so the `FLUSHALL`
    that `cli` sends there reaches that container.
  - `CCG_CONFIG_DIR` only changes the password, which fails closed (m3).
  - The one hardening left is the label check from M1.
- **`down`:** yes, as M1 describes. A `NAME` override removes another container. A `PORT`
  override changes which instance the guard counts. A stopped container of any name is never
  guarded. With the default name and port the guard fails closed: a missing password file or
  `NOAUTH` makes `n` something other than `0`, so the script refuses.
- **`up`:** yes, its recreate branch does the same to a non-running container (M1).

### Q3 — can an instance with the wrong config pass?

- **An older conf on a replaced stopped container:** no (traced). Whenever the container is not
  running, `up` calls `write_conf` before `docker run` (`:140-156`). That rewrites the file from
  the script's constants on every recreate, and the file name includes `$NAME`. A recreated
  container always mounts the current D7 conf. A running container is left alone, and `status`
  catches drift in the four D7 values, including a later `CONFIG SET` by S21 or S22.
- **What does pass:** drift that `status` doesn't check (m1 and m2). That covers an instance with
  no password, a different image or module, no published port, no label, or a leftover
  `TIMEOUT_MAX` or ACL user.

### Q4 — the two deviations, for a 6 GB host

- **The `down` key guard: keep it.** It costs nothing on an empty instance. It protects the
  non-exclusive R-B and R-D reservations, which §5.1 allows to overlap. It fails closed. It
  doesn't break the done-condition's idempotency, because `down` on an absent container is a
  no-op, and refusing on a populated instance is still repeatable. It is only sound once M1
  points it at the right instance.
- **No `--memory` cap: I'd add one. The case for it is reasoning, not something I ran.**
  - `maxmemory` limits Redis's accounted `used_memory`, and only when each command starts. It
    doesn't limit allocator fragmentation, client buffers, or a single `GRAPH.QUERY` that
    overshoots after it has been admitted.
  - LM Studio is on the Windows side and is already protected by the `.wslconfig` cap. What a
    cgroup cap protects is everything *inside* the VM. Without one, a runaway would hit the VM's
    OOM killer, and that can take `falkordb-dev` (shared), the `cypher-mcp` containers, or the
    Joern JVM a step is timing.
  - With a cap, the failure stays in `ccg-falkordb`, and since nothing persists there, losing it
    costs nothing.
  - The VM is already tight. At review time `free -m` showed 165 MB available and 2.1 GB of swap
    in use, with a second FalkorDB (`gdba-s07-probe`, 6394) running. I didn't investigate why.
  - **Suggested improvement:** `--memory=2g --memory-swap=2g`, which is about 1.3× `maxmemory`.
    Setting the swap limit equal to the memory limit keeps the container out of swap, so a
    distorted S20 timing becomes a loud OOM instead. Have `status` print `HostConfig.Memory`.
  - The exact headroom is `devops`'s call. I haven't run it, and no assertion I could write
    today would show that 2g is right rather than merely large enough.

### Q5 — secrets hygiene

- **No create/chmod race (traced and checked).**
  - The password file and the conf file are both created inside `(umask 077 && …)`
    (`:82`, `:93`), so they are 600 from the start. The `chmod 600` after each (`:85`, `:102`)
    is a second safeguard, not the protection.
  - `mkdir -p` under the same umask creates the directory at 700.
  - If `~/.config/ccg` already exists with looser permissions, nothing tightens it. That's a nit
    with no effect today, because the directory is 700.
- **The password reaches the processes cleanly.** It goes to `redis-server` through the
  bind-mounted conf, never through env or argv. It goes to `redis-cli` through `REDISCLI_AUTH`,
  which is readable only in that process's `/proc/<pid>/environ`, by the same uid.
- **The conf file holds the password in plaintext,** at the same 600 trust level as the password
  file itself. Anyone in the Docker group can read it inside the container, but that group is
  equivalent to root anyway.
- **The README leaks nothing.** 0 hits in the scan. It names only paths, and its connect example
  uses an env assignment, not `-a`.

### What's solid

- The password design is right. It never goes through argv or env, the conf is mounted
  read-only, and the image's `REDIS_ARGS` hook is used so that `FALKORDB_ARGS` is untouched.
  The module args are the same as `falkordb-dev`'s.
- `reset-p0` puts its literal guards before `preflight`, and its binding check is exact.
- `state()` handles `docker inspect`'s empty line on a missing name correctly.
- The script binds `127.0.0.1` only. It gives a clear warning not to copy `falkordb-dev`'s `-p`
  form. `BROWSER=0` saves RAM. `--pull=never` keeps the image pinned.
- The README is short and accurate apart from the M1 implication. It also keeps to the §5.2
  single-writer rule.

### Open questions

1. *For `teco`, outside S01's scope.* `gdba-s07-probe` (`falkordb/falkordb:v4.18.11`) is running
   on `0.0.0.0:6394`, not `127.0.0.1`, and outside the 6391–6393 range D7 cites. D7 says the host
   cannot hold two instances plus a JVM. Is it meant to still be running, and is its binding
   intended?
2. *For `teco`.* The done-condition items "`down` idempotent" and "`down` → `up` restores" rest
   only on the delivery report. Should `devops` re-run them after the M1 fix, which changes
   `down`, as part of the fix's done-condition?

## S04 (P0-03b)

**Reviewed (2026-10-29):** `docs/plans/connected-code-graph-egress-spec.md` v1.0 (untracked),
against the S04 row of `docs/plans/connected-code-graph-p0.md` §3, plus its §2 D6, §4.2, §5.3 and §7;
proposal `docs/plans/connected-code-graph.md` §3.12 and §5.3; FR-9 and AC-8 in
`docs/requirements/connected-code-graph.md`; and review `docs/reviews/connected-code-graph-p0.md`
Pass 1 M1 and m4.

**Verdict: needs changes.** There is no blocker. There are five majors:
- M1 and M2 make S08 tests 1 and 7 impossible to pass as written, so they must be fixed before S08
  is dispatched.
- M3, M4 and M5 are holes in the P1–P2 broker contract. They don't affect P0, which builds offline
  only and implements no broker, but they must be fixed before P1 builds against §2–§6.

The minors can ride along in the same revision. (Paused once at teco's request and resumed; this
is the complete gate.)

CPG: considered, not relevant — the artifact is a contract spec for code that does not exist yet;
it was checked against Joern's launcher scripts and jars, S02's Dockerfile, and host/container probes.

**What I ran (read-only).** I re-ran §7.3.2's regex over the author's `--help` outputs, which
reproduced the 13 flags. I read `frontends/kotlin2cpg/bin/kotlin2cpg` (the native-packager
template) and `cpg/infra/toolchain/Dockerfile`, and ran `joern-parse --list-languages` inside
`unshare -rn`. I ran four throwaway `docker run --rm --pull=never --network none` containers of the
present `python:3.12-slim` image: a socket probe, an env probe, a mount/`/init` probe and a memfd
exec probe. I created no networks or volumes, and stopped nothing.

### Findings

#### M1 (major) — A6(a) cannot pass with the container mechanism the spec names, and `envKeys` records intent rather than the process environment
(verified) §7.2 A6 says the container closes the environment axis with an "explicit `-e` pass-list
only", and A6(a) requires the probe's env keys to **equal** the pass-list. Docker cannot remove an
image's `ENV` entries, and it always adds `HOSTNAME`. The probe used `-e PATH=… -e HOME=…` and still
saw `GPG_KEY, HOSTNAME, LANG, PYTHON_SHA256, PYTHON_VERSION`. S02's image sets `JOERN_HOME`,
`GOROOT`, `GOTOOLCHAIN`, `MAVEN_HOME`, `GRADLE_HOME`, `DOTNET_*` and `PATH`
(`cpg/infra/toolchain/Dockerfile:136`), and the Temurin base adds more. §5.3 `envKeys` is "every
variable passed in", so 8.6's "`envKeys` contains no forbidden name" would pass even if a later S24
layer added `ENV JAVA_TOOL_OPTIONS=…`.
**Fix:** contract the container launch as `--entrypoint /usr/bin/env` with `-i K=V… <tool>`, or as an
equivalent wrapper. Define `envKeys` as the keys of the environment the tool process actually
receives. Decide explicitly which safety-positive image keys join the pass-list: `GOTOOLCHAIN=local`
and `DOTNET_CLI_TELEMETRY_OPTOUT=1` matter in controlled mode. §7.3.4's `PATH`
(`/opt/joern:/opt/java/bin`) doesn't match S02's layout (`/opt/joern/joern-cli`,
`/opt/java/openjdk/bin`), so take those values from S02's image.

#### M2 (major) — A1(d)'s UDP clause gives a false red on loopback
(verified) A1(d) requires UDP `sendto` to fail with an errno "for each of … `127.0.0.1` on every
`local` allow-list port". In `--network none`, `lo` is up, and `sendto(("127.0.0.1",1234))` returned
1, a success. A connected UDP socket's `recv` gives `ECONNREFUSED` instead. An S08 implementer
can't satisfy the clause as written, and would end up weakening it ad hoc.
**Fix:** for loopback targets, specify the UDP probe as connect + send + `recv` with a 2 s timeout,
passing on `ECONNREFUSED`. Keep plain `sendto` for non-loopback targets, where it gave
`ENETUNREACH`.

#### M3 (major) — Self-declared `codebases` lets a structured call escape the sticky rule, and AC-8 doesn't see it
(traced in §3.3/§8) R0 only requires `request.codebases ⊆` the run's codebases. The empty set
passes, and with it `M = hosted`. In a local-only refresh or answer run, a structured request with
`codebases: []` is therefore decided by R7/R8/R9 (R5 still catches a model call). R3 is also not
bound to `runKind = vulndb`: any run's token can send to the OSV endpoint with `codebases: []` and a
self-declared `payloadClass: vuln-feed-request`. 8.1, 8.5 and 8.8 filter on `codebases ∩ L ≠ ∅`, so
those decisions never enter their checks, and the `conn` join matches them to an allow. Only 8.4
catches the R3 case, after the fact.
**Fix:** in R0, an empty `codebases` is invalid except in a `vulndb` run. R3's condition gains "the
run's `runKind = vulndb` and run codebases = []", and a `vulndb` grant carries no repository
codebases. For `answer` runs, compute `M` over the run's codebases, since FR-9 is sticky per answer.
Add an 8.x check: "no allow in a run with `L ≠ ∅` has `codebases = []` unless `runKind = vulndb`".

#### M4 (major) — The controlled path's "no route except the broker" is not true of `--internal`, and has no executable property (I5)
(documented, not probed: probing needs a Docker network, which the brief excludes) Docker's documentation, quoted in
the CVE-2024-29018 advisories, says containers on an `--internal` network **can reach the gateway
IP, and thus host services**. Host listeners on `0.0.0.0` include `falkordb-dev`'s 6379 and the
`0.0.0.0:6394` probe noted in S01. DNS forwarding from internal networks is fixed since Moby 26, so
29.7.2 is fine there. §6 step 4 ("a tool that ignores the proxy … has no route at all") is
therefore unproven. §7.2 runs only on offline backends, so no property covers controlled mode. Also:
- §6 step 1 says the broker attaches "to it and to the host network". A container can't join
  `host` plus another network, so say which.
- The token sits in the build env (`HTTPS_PROXY`) alongside repo-driven build code, for example
  Gradle scripts under `--download-dependencies`.
- §7.2's launcher clause and A6(c) assume `--network none` and a heap-only `JAVA_OPTS`, which
  contradicts §6 step 2 and 8.12.

**Fix (P2 contract, recorded now):** define a controlled-mode probe set, "A1c":
- only `broker:3128` is reachable;
- the gateway IP refuses on every listening host port;
- `getent ahosts example.com` fails.

Also: a host-side `DOCKER-USER` drop for the gateway, or put the broker in a container with no
published ports. Make the launcher clause and A6(c) mode-dependent. Issue a proxy-only sub-token.

#### m1 (minor) — `joern-parse --language` mapping is unspecified
(verified) Languages are case-insensitive and aliased: `JAVA`/`JAVASRC`, `PYTHON`/`PYTHONSRC`,
`JAVASCRIPT`/`JSSRC`, `CSHARP`/`CSHARPSRC`, `C`/`NEWC`, `LLVM` (console jar `cpgcreation/package$`).
`JavaCpgGenerator` references `java2cpg.sh` and `jimple2cpg`. With no `--language`, joern-parse
guesses, and may pick `rubysrc2cpg`. Generators spawn the frontend script (`runShellCommand`), so
`--frontend-args` reach its launcher. **Fix:** an explicit alias→frontend table, exactly one
`--language` (either form), and refusal when it is absent or duplicated.

#### m2 (minor) — N1 detection: missing fixtures and outcomes
(verified host format) The `magic 4d5a` check plus the `WSLInterop*` glob misses an
**extension-type** entry (`extension .exe`) under any other name. On a microsoft kernel, an
**empty** `binfmt_misc` dir (fs not mounted in the runner's mount namespace while entries still
exist kernel-wide on 5.15) must count as "cannot be read". Readability should key on `status`
existing. §10 test 8 lists fixtures but not each one's expected result. Make `microsoft`
case-insensitive (WSL1 is `Microsoft`). Also refuse on an `interpreter /init` entry. In practice
N2 (`/dev/vsock`) already refuses on every WSL2 host.
**Least-disruptive P0-plan fix:** no in-place edit. The P0 plan is being executed against, so
collision rule 5 would demand a successor. Instead, S08's brief and the ledger state that S04 §7.1
N1 controls over D6's literal path (P0 §5.3 already routes S04→S08).

#### m3 (minor) — Escape axes: three tightenings
- A4(a): noexec masks interpreter absence. A memfd exec of an `MZ` stub gave `ENOENT` (verified),
  so add that variant to exercise mechanism 1 independently.
- C2 checks only that seccomp is present. Require `profile=builtin` (or a runner-supplied profile),
  because a daemon `seccomp-profile` change would go unseen per invocation.
- §5.3 `preconditions.vsock` has no stated derivation for the container backend.

#### M5 (major) — R2 opens a host-wide tunnel, so "only its own git remote, for fetching" is unenforced and unasserted
(traced in §2.3, §3.3 and §8) §2.3 decides proxied streams on `scheme://host:port` only, with no
TLS interception. R2 therefore allows an opaque TLS tunnel to, for example, `github.com:443`. Any
holder of the run token can use that tunnel for any request to that host, such as a push to another
repo, a gist, or an issue body carrying source. This is a local-only exfiltration path within the
rules as written. 8.2 checks only `endpoint = canonical(remote(c))`, so it would pass.
Nothing in the spec says which process speaks `git-fetch`, or with what argv and credentials.
**Fix:**
- Contract the git-fetch path: a fixed argv `git fetch <remote>`, run by the runner (not by an
  arbitrary component), with no push or write credentials present.
- Have 8.2 also bound `close.bytesOut` per R2 connection (fetch negotiation is small) and flag any
  outlier.
- State the residual risk explicitly: the host is shared, and there is no path visibility. Where the
  stakeholder rejects that, require TLS interception for R2 only.

#### m4 (minor) — Remaining broker-contract gaps (§2–§4)
- **Multi-codebase proxy profile.** §2.3 derives proxied `codebases` "from the token's run". In a
  refresh run containing a local-only codebase, `M = local-only` for every proxied request, so a
  hosted-default codebase in the same refresh can never download (R8 unreachable). Bind proxy
  sub-tokens to `(runId, repoId)`; the runner mints one per sandbox invocation.
- **§4.3 resolution.**
  - Say that the broker connects to the address it validated and never re-resolves (no rebinding
    window).
  - Extend the forbidden set beyond loopback and link-local: RFC 1918/ULA, the host's own interface
    addresses, and Docker bridge gateways. Otherwise a hosted entry that resolves to `172.17.0.1`
    reaches local services.
- **CONNECT → `scheme`.** Under §4.3, endpoints carry a scheme, but a CONNECT target has none.
  Define the derivation per proxy profile. Also define how `ssh://` remotes reach the broker at
  all, or forbid them in `remote`. ssh ignores `HTTPS_PROXY`.
- Token lifetime (24 h, run-bound, memory-only), R4's closed body shape, and `R0` for unknown enums
  are sound as written.

#### m5 (minor) — AC-8 rows 8.12 and 8.13 are not computable as written
- **8.12:** "every `sandbox` record that downloads". No §5.3 field says a record downloaded. Restate
  it as two checks:
  - every record with `egressMode = controlled` has `network = broker` and
    `--network ccg-egress-<runId>`;
  - every record whose `toolArgv` holds a `forbiddenIn = offline` flag has
    `egressMode = controlled` and a hosted-effective codebase.
- **8.13:** this needs `--jar4import-url` to show up in `allowedHits`. But §7.3.3 lists it as
  forbidden `all`, and §7.3.6 fails a flag that is in both files. So it can never be in
  `allowed-flags.txt`, and §7.4 has no lifting step. Specify the lift: a §7.4 step that matches the
  value against `importNameServices`, and records the lift in its own field, for example
  `flagCheck.liftedHits`.
- 8.2–8.4, 8.6, 8.7 and 8.9–8.11 are computable from the named fields, apart from the
  dependencies on M1 (`envKeys`), M3 (empty `codebases`) and M5 (R2 granularity). 8.14 is sound as
  a revision trigger.

#### m6 (minor) — §10 tests 2, 5 and 7, and A1(a) on the namespace backend
- **Test 2's "`docker ps -a` gains no entry"** cannot see a `--rm` container that was created and
  removed between the two snapshots. Use `docker events --filter type=container --filter
  event=create` over the test window instead.
- **Test 7's "pass-list ∩ forbidden = ∅"** is a text check (I5). It must also be mode-aware,
  because `HTTPS_PROXY` is on the controlled pass-list but forbidden offline. Its runtime half is
  A6, which M1 fixes. Say so, and drop the claim that test 7 alone proves it.
- **A1(a) reads `/sys/class/net`, but the namespace bind set (§7.1) mounts no `/sys`.** Bind-mounting
  the host `/sys` would show the host's interfaces. For that backend, specify `/proc/net/dev` (per
  netns) instead, which matters for R5's direct-bwrap exercise.
- **Naming.** Today's executables include `c2cpg.sh` and `jssrc2cpg.sh` (the author's sweep reports
  `c2cpg.sh`), while §7.3.1 and the list files key on `c2cpg`. Fix one name, and specify the
  mapping from frontend to executable.
- Tests 3, 4 and 6 are fine. My re-run of the §7.3.6 splitting rule over all nine help files
  reproduces exactly the 13 hits. Continuation lines in every file (including `--delombok-mode`'s
  option list and `c2cpg`'s multi-line description) start with whitespace and a non-`-` character,
  so the rule holds on today's output.

#### n1 (nit) — anchored exceptions
The `-J-Xmx`/`-J-Xss` exceptions should be full-match regexes (`^-J-Xmx[0-9]+[mg]$`), not prefixes.

### Answers to the brief's questions
- **Q1:** the join proves only the broker's own upstream conduct. It cannot see the following:
  - UDP/DNS, since `conn` has no protocol and a CONNECT proxy is TCP-only;
  - traffic to `local` (R1) endpoints, which never reaches the logging proxy;
  - connections before the proxy starts or outside a run (add a check that the proxy started before
    the first `policy` record, plus a closing flush marker);
  - any non-broker component that has a route. The topology for non-sandboxed components is not
    contracted: §8's "only the broker is attached to it" is ambiguous, so add an A1-style probe from
    each component container;
  - under-declared structured calls (M3).

  Also, `ssh://host:22` remotes (§4.3) have no defined path through an HTTP-CONNECT broker. ssh
  ignores `HTTPS_PROXY`, and CONNECT→`scheme` derivation is undefined.
- **Q2:** A7 is verified on Docker 29.7.2. The default seccomp gives `EPERM` for `AF_VSOCK`, and
  `seccomp=unconfined` creates the socket, so the attribution is to seccomp. H10's assumption now
  holds. For `--network none` I found no axis missing on this host: netlink is allowed but
  per-netns, `AF_PACKET` gives `EPERM`, `/dev/shm` is `noexec`, `/init` is absent, and binfmt_misc
  is not mounted inside. A1/A6 are M2/M1.
- **Q3:** see m2.
- **Q4:** I read the launcher template. `-D*|-agentlib*|-agentpath*|-javaagent*|-XX*` and `-J*` are
  bash `case` patterns, which are case-sensitive, so a case-sensitive prefix list mirrors them
  exactly.
  - `--`: the launcher stops parsing there, and the spec over-checks, which is fail-closed.
  - `=` forms: `-jvm-debug=…` and `-java-home=…` fall to residual args, which the launcher ignores.
  - `-J@file` (a JVM argfile from the repo) is caught by `-J`.
  - `JAVA_OPTS` is expanded unquoted, but it is runner-set.
  - `JAVA_TOOL_OPTIONS`/`_JAVA_OPTIONS`/`JDK_JAVA_OPTIONS` are closed by the pass-list only once M1
    is fixed.

  The three Kotlin flags are correctly forbidden under the stated rule. Both Gradle flags' `--help`
  says "used to download dependencies", so that is evidence, not its absence. Lombok is bundled
  (`org.projectlombok.lombok-1.18.42.jar`), which supports allowing `--delombok-mode`. Omitted and
  harmless launcher options: `-mem` (which consumes the next argument), `-d`, `-v` and
  `-no-version-check`.
- **Q5:**
  - Refuse-and-continue is consistent with FR-9 ("never be overridden"), the proposal ("refused and
    logged") and AC-8 ("even when a run requests hosted"), and it is within S04's policy remit.
  - Refusing controlled mode in P0 is within the plan (P0 §0).
  - The `--internal` design is within remit, but see M4.
- **Q6:** S03 has already chosen `sandboxAuditIds` (an array, BF15, `bundle-spec.md:93`). The things
  S04 must change, or S03 must encode:
  - §5.3's singular "a bundle whose `auditId` has no record" → "every id in `sandboxAuditIds`";
  - the "output file name stem" convention applies to tool outputs, not to the bundle file;
  - S03 requires `egressMode` to be equal on every referenced record.

- **R3 (`/tmp/native`)** is acceptable as specified. An exec-capable tmpfs drops only the noexec
  layer. Interpreter absence still closes A4 there: my memfd exec of an `MZ` stub, which no noexec
  mount can stop, failed with `ENOENT`. Socket absence holds too. The spec already requires A4(a)
  to pass with a PE staged there.
- **The namespace bind set (§7.1)** is sound for what it can claim. `/usr/lib/wsl/{lib,drivers}`
  are separate Windows-backed mounts (H4), and the `--tmpfs /usr/lib/wsl` over a recursive `/usr`
  bind masks both. The `/sys` gap is m6. On this host the backend refuses under N2 in any case.

### What's solid
- **The invariants.** I1–I5, and above all I5 (a property that can only be checked by reading text
  is a spec defect), are the right frame. Most findings above are I5 applied to the spec's own text.
- **The probe record (§9) is honest.** It separates verified from inferred (H5, H7, H10). Finding 1
  (`WSLInterop-late`) is real, and it would have produced exactly the M1 escape.
- **AF_VSOCK (A7) as an added axis is correct.** I confirmed the seccomp attribution on Docker
  29.7.2, so H10's assumption now holds. N2 is the right refusal.
- **The launcher-option analysis (§7.3.5, H9) matches the template exactly.** Keeping those
  options out of the substring-matched lists is the right call.
- The sticky-rule table (§3.2), R5 placed before R7, R4's closed body shape, decide-before-send
  with no unrecorded allow (§2.5), the append-only `flock` audit, and the classification rule's
  fail-closed default.

### Open questions
1. *For `teco`.* Where should the N1 correction live (m2)? I recommend S08's brief plus a ledger
   note, not an in-place edit of the P0 plan.
2. *For the stakeholder, via `tico`, at P1.* Should a host-granular git tunnel (M5) be accepted as a
   residual risk with the mitigations, or does local-only require TLS interception for R2?
3. *For `teco`, S03/S04 seam.* §5.3's singular `auditId` loader check should be reworded to
   S03's `sandboxAuditIds` array (Q6) in the same S04 revision.

### Pass 2

**Re-reviewed (2026-10-29):** egress spec v1.1, including its §12 dispositions. **Verdict: needs
changes, for one new major only (P2-M1).** All other Pass 1 findings are fixed or properly
deferred. The new minors can ride along in the same small revision.

**What I ran (read-only):** throwaway `docker run --rm --pull=never --network none` containers of
`ccg-toolchain:2c5e60084f67` and `python:3.12-slim`, using v1.1's launch shape (`--hostname
localhost --read-only --cap-drop ALL --user 1000:1000 --entrypoint /usr/bin/env -i <pass-list>`).
`/src` was a read-only bind of a scratchpad fixture, and `/out`, `/tmp` and `/home/ccg` were
tmpfs. I ran `joern-parse` for nine `--language` values, with three `/tmp` shapes. I created no
networks or volumes.

**Dispositions.**
- **M1: fixed, verified.**
  - Through `env -i`, the tool's `/proc/self/environ` keys are exactly the 13 offline pass-list
    keys, and `HOSTNAME` is absent.
  - `--hostname localhost` doesn't conflict with A6(d): the uts name is `localhost`, while the
    `HOSTNAME` variable stays absent.
  - R9 holds: `joern-parse` parses without `JOERN_HOME`, once `/tmp` is fixed (P2-M1).
  - Two precision points remain, in P2-n1.
- **M2: fixed, verified.** Loopback UDP connect, send and `recv` gives `ECONNREFUSED` for
  `127.0.0.1:1234` and for `::1`, in both images.
- **M3: fixed.** The empty-set and `vulndb` holes are closed. §2.4a and 8.15 are consistent with
  8.1, 8.5 and 8.8, and 8.15 is a sound detector of broker bugs. A residual remains for refresh
  runs and 8.8's filter: P2-m2.
- **M4: fixed (documented, not probed).** A1c is a good ship criterion. On the departure (binding
  by network attachment instead of a sub-token): I accept it, because it is stronger, with no
  secret to leak. Its conditions are in P2-m3. I can't confirm the JDK `disabledSchemes` claim from
  here.
- **M5: mitigations written, decision correctly left open** to the stakeholder (§11 Q1). The fixed
  argv, no hooks, bare mirror, and `bytesOut` bound are what I asked for.
- **m1: fixed, but the table is wrong in one row, and the `java` rationale is wrong.** See P2-m1.
- **m2: fixed.** The `interop_state` function and the F1–F12 fixtures with outcomes are complete.
- **m3: fixed.**
- **m4: fixed.** The private-address departure (a per-entry `privateAddress: true` instead of a
  blanket ban) is accepted: the reasoning holds, and host, Docker and `local` addresses stay
  banned unconditionally. Nit: put the flag on the `decision` record too, so AC-8 can list those
  allows.
- **m5: fixed.** The lift step and `liftedHits` remove the both-lists contradiction.
- **m6: fixed.** The renames agree with the alias table and with what `joern-parse` actually runs:
  `c`/`newc` → `c2cpg.sh`, and `jssrc`/`javascript` → `jssrc2cpg.sh` (verified).
- **n1: fixed.**
- **Q6 seam: fixed.**

#### P2-M1 (major) — A `noexec` `/tmp` breaks every frontend, and R3's fallback can't fix it
(verified) With v1.1's `--tmpfs /tmp:rw,noexec,nosuid`, both `joern-parse --language javasrc` and
`--language pythonsrc` exit 1:
- `UnsatisfiedLinkError: /tmp/libzstd-jni-…so: failed to map segment`;
- ZstdWrapper warns about the `noexec` `/tmp`.

R3's permitted fallback also fails: a toolArgv `-J-Djava.io.tmpdir=/tmp/native`, with an
exec-capable `/tmp/native` tmpfs, still gives exit 1 with the same error for `javasrc` and `jssrc`.
`joern-parse` spawns the frontend as `/opt/joern/joern-cli/<fe> -J-Xmx2048m …` with
`extraEnv: Map()`, so the frontend's JVM never sees the parent's `-J` option.

Passing the property through **`JAVA_OPTS`** works, because every child launcher reads that
inherited variable. With `JAVA_OPTS="-Xmx2g -Djava.io.tmpdir=/tmp/native"` and `/tmp` still
`noexec`, all of these exit 0 and write `cpg.bin`, in 2–5 s:
- `pythonsrc`, `javasrc`, `javascript`→`jssrc2cpg.sh`, `csharp`→`csharpsrc2cpg`,
  `java`→`jimple2cpg`, `golang`→`gosrc2cpg`, `c` and `newc`.

I did not run `kotlin2cpg`. This agrees with the S02 gate's M1 on the failure. One difference:
jssrc succeeded for me with `TMPDIR=/tmp` still `noexec`, so the astgen extraction appears to follow
`java.io.tmpdir`, not `$TMPDIR`.

**What it costs the escape axes:** one writable location becomes exec-capable, so A4 loses its
`noexec` mechanism on that path only. Two mechanisms still close A4 there:
- interpreter absence (`/init` absent): the memfd exec, which no mount flag can stop, already gives
  `ENOENT`;
- interop-socket absence.

An ELF run from there is still inside `--network none`, the seccomp profile and `--cap-drop ALL`,
so A1, A3 and A7 are unaffected.

**The `/tmp` shape I'd accept:**
- `/tmp` and `/home/ccg` stay `noexec`.
- Add one `--tmpfs /tmp/native:rw,exec,nosuid,nodev,uid=<uid>`. State `exec` explicitly, because
  Docker's `--tmpfs` defaults to `noexec`.
- `JAVA_OPTS` carries `-Djava.io.tmpdir=/tmp/native`, so A6(c)'s offline and controlled patterns
  gain that literal.
- Remove the `-J-Djava.io.tmpdir` exception from §7.3.5 and from R3, because it doesn't work.
- Make R3 unconditional, not "only if test 3 fails".
- Add `/tmp/native` to A5's and test 5's allowed mount set, and to A4(a)'s staging locations.
- A4's mechanism column says `noexec` covers every writable mount except `/tmp/native`.

An all-exec `/tmp` would cost the same mechanism on a larger surface. I'd accept it too, but the
narrow shape is the better default.

#### P2-m1 (minor) — Alias table: `python` is mapped wrongly, and the `java` refusal rests on a misreading
(verified)
- **`python`:** `--language python` reaches `PyCpgGenerator`, which fails with `CPG generator does
  not exist at: /opt/joern/joern-cli/py2cpg.sh`. It does not reach `pysrc2cpg`. Mapped to `pysrc2cpg`,
  it would be flag-checked as `pysrc2cpg`, recorded as `pysrc2cpg`, and then fail. That fails
  closed, but the record is wrong. **Refuse `python`**, keeping `pythonsrc`.
- **`java`:** my Pass 1 note listed the strings `java2cpg.sh` and `jimple2cpg`, but I didn't say
  the target is ambiguous. The generators carry `commercialAvailable`/`ossAvailable`. With the
  commercial `java2cpg.sh` absent, `java` deterministically runs `jimple2cpg` (verified, exit 0).
  `csharp` and `golang` follow the same commercial-else-OSS pattern (`csharp2cpg.sh`/`go2cpg.sh`),
  and the table accepts them. Refusing `java` is harmless, and FR-1 stays covered: Java source goes
  through `javasrc`, and bytecode through `jimple2cpg` directly. But the stated reason is wrong, and
  it is inconsistent with `csharp`/`golang`.
- **Fix:** either map `java` → `jimple2cpg`, or keep refusing it with the honest reason
  ("redundant alias"). In both cases, add a test asserting that `java2cpg.sh`, `csharp2cpg.sh`,
  `go2cpg.sh`, `py2cpg.sh` and `js2cpg.sh` are absent from the image. Their presence would silently
  change the frontend that `csharp`, `golang`, `java` and `javascript` reach.

#### P2-m2 (minor) — Under-declaration remains for refresh-run structured calls, and 8.8 still filters on declared codebases
(traced in §3.3) In a `refresh` run, the scope set is `request.codebases`, and a structured call
declares it. R8 and R9 don't require `transport = proxy`. So a structured `dependency-download`
that declares only a hosted codebase in a mixed refresh is allowed, whatever its body holds.
**Fix:** R8 and R9 require `transport = proxy`, which means a binding-fixed `repoId`. 8.8 should
then select by the answer's `runId` alone: every decision in a run whose codebases ∩ L ≠ ∅.
Today it filters on `decision.codebases ∩ L`. The policy (run-scoped `M`) already closes the
answer case; 8.8 just doesn't verify it.

#### P2-m3 (minor) — Conditions for accepting network-attachment binding (judged statically)
- The broker must identify a binding by **both** its listening address on that network and the
  peer IP being that invocation's container IP. Use one listener, or one accept-time check, per
  network.
- A1c should add two checks, so the broker's multi-homed weak-host routing can't leak one binding
  into another:
  - the build container has no default route;
  - it cannot connect to the broker's addresses on other `ccg-egress-*` networks.

  With `--cap-drop ALL`, neither a route nor spoofing should be possible, but that is the
  property to probe.
- A runner killed mid-invocation leaves `ccg-egress-<auditId>` behind, and Docker's default
  address pools are small (roughly 30 networks). P2 needs a reaper keyed on the name prefix, and a
  refusal when the pool is exhausted, rather than reuse.

#### P2-n1 (nit) — the A6 boundary and `JRE_CACERTS_PATH`
- **A6's "exactly the pass-list" holds at the `env -i` exec boundary only.** The bash launchers
  add `PWD`, `SHLVL` and `_` to their children (verified: a bash child's environ shows them). So the
  JVM sees those three extra keys. Specify that the A6 probe reads its own `/proc/self/environ`,
  not a shell child's, and that `{PWD, SHLVL, _}` in descendants of a launcher is expected.
- **`JRE_CACERTS_PATH`** is not in the image's `Config.Env`. Temurin's `/__cacert_entrypoint.sh`
  sets it at runtime, and `--entrypoint` bypasses that script. Drop it from A6(d)'s examples.

### Pass 3

**Re-reviewed (2026-10-29):** egress spec v1.3, including §12.2. **Verdict: approve with
suggestions.** Every Pass 2 finding is fixed, and I confirmed each one by running it. Two new
items, one minor and one nit, can land in S08's brief or a v1.4 errata line; neither needs
re-gating.

**What I ran:** throwaway `--rm --pull=never --network none` containers of
`ccg-toolchain:d3184effb566` in v1.3's shape: `--entrypoint /usr/bin/env -i` plus the pass-list,
`--hostname localhost`, `--read-only`, `--cap-drop ALL` and `--user 1000:1000`. I ran
`joern-parse`, `joern --script` (19 small scripts) and `--help` for all 10 launchable tools. I
created no networks or volumes.

**Dispositions.**
- **P2-M1: fixed, and the departure is accepted.** The exec `/tmp` replaces my narrow shape, and
  both of the architect's reasons reproduce.
  - **The narrow shape fails for scripts.** With `/tmp` `noexec`, an exec `/tmp/native`,
    `JAVA_OPTS=-Xmx2g -Djava.io.tmpdir=/tmp/native` and `TMPDIR=/tmp`, `joern --script` exits 1
    with `UnsatisfiedLinkError … failed to map segment`.
  - **My Pass 2 "jssrc succeeded" was a false green.** I had checked only for "Successfully wrote".
    In that shape, `joern-parse --language jssrc` exits 0 with a 9,195 B CPG. `jssrc2cpg.sh` run
    directly logs `running astgen failed! … /tmp/gztmp…/astgen-linux: Permission denied`.
  - **v1.3's shape works.** `/tmp` is `rw,exec,nosuid,nodev` (mountinfo confirms it has no
    `noexec`). There, jssrc writes 76,296 B, and `joern --script` exits 0 printing
    `METHODS=f,:program,…`.
  - **Cost to A4:** the `noexec` mechanism is lost on `/tmp`. Two mechanisms remain, and both
    hold:
    - interpreter absence: an `MZ` stub in the exec `/tmp` fails with "required file not found",
      the same `ENOENT` as the Pass 1 memfd exec;
    - interop-socket absence: `/init` and `/run/WSL` are absent.

    A shell script in `/tmp` runs, as expected. An ELF from `/tmp` is still confined by A1, A3 and
    A7.
- **P2-m1: fixed.** Test 9 (absence of the commercial scripts) closes my concern.
- **P2-m2: fixed, and the added `bind_proxy(…, liftedEntryIds)` is consistent.**
  - `import-name-lookup` is derived only on a binding, so R9's `transport = proxy` can now match.
  - 8.15(ii) follows directly from that.
  - 8.15(iii) is sound: `L ≠ ∅` implies `systemLocalOnly`, so R5 pre-empts R7 for every structured
    model call in such a refresh run. R3 is `vulndb`-only, and R2/R8/R9 are proxy-only. That
    leaves only R1 and R4.
- **P2-m3: fixed.** The two-condition binding identity and A1c(f) are as asked. The reaper's
  10-minute age bound is sound (judged statically): a long invocation keeps its container
  attached, so it is never reaped, and the age bound covers the create→attach window. Pool
  exhaustion refuses rather than reusing a network.
- **P2-n1: fixed, and A6(e) verified.** The `env -i` target's own environ equals the 13 offline
  keys. Its `bash -c` and `sh -c` children add exactly `PWD`, `SHLVL` and `_`.
- **Flag count and sweep: correct.** My re-run in the new image (all 10 tools exit 0) gives 16
  hits: the same 13 frontend flags, plus `joern`'s `-d/--dep`, `-r/--repo` and `--verbose`. That
  is 13 forbidden and 3 allowed, as the corrected §7.3.3 says. The short forms `-d` and `-r` are
  refused by R10's closed grammar, not by the substring lists, and that is fine.
- **R10:**
  - The closed `--script /scripts/<name>.sc [--param k=v]…` grammar, offline-only, staging from a
    runner-owned copy with its hash, and the `//>` refusal are the right design.
  - Without `--workdir /tmp`, `joern` fails on the read-only `/work`. That matches H15, and
    `--workdir` adds no mount, so A5 is unaffected.
  - The block-comment case is covered: a `//>` line inside `/* … */` still makes replpp try the
    download, and the line-start rule catches it. P3-m1 has the remaining gap.

#### P3-m1 (minor) — the directive rule rests on a false "indented is inert" claim, and its whitespace set must be named
(verified) The test was a directive on line 1. The results:
- **Download attempted** (`UnknownHost`/`getOrDownload`): unindented `//> using dep …`; the same
  line indented by two spaces; and the same line led by `\r`, `\f` or `\v`.
- **Inert:** led by NBSP, by a BOM (a compile error), `///>`, a trailing mid-line `//>`, and
  Ammonite-style `import $ivy`/`import $dep` (a compile error).

The refusal (`//>` after leading whitespace is stripped) already covers every case that triggers.
But two things in the spec are off:
- §7.3.1's sentence "An indented directive is inert today (H15)" is false, at least on line 1.
- An implementation of "whitespace" as `[ \t]*` would miss `\r`, `\f` and `\v`, which replpp
  honours.

**Fix:**
- Define the stripped set as ASCII whitespace `[ \t\r\n\f\v]`, or simply refuse a line in which
  `//>` follows only those characters.
- Correct the sentence and H15.
- Add `\r`-, `\f`- and `\v`-led `//> using dep` fixtures to test 10(b).

#### P3-n1 (nit) — the `/scripts` mount point
Test 5's allowed set names the mount point `/scripts/<name>.sc`, a **file** bind, and §7.3.1
wants it `noexec`. Docker binds have no `noexec` option (§11 R2), and a `local`-volume bind, the
route §7.1.2 proposes, binds a **directory**. Bind the runner-owned temp directory at `/scripts`,
and list `/scripts` in test 5. Or drop `noexec` for this read-only, non-executed file.

## S05 (P0-04)

**Reviewed (2026-10-29):** `docs/plans/connected-code-graph-p0-ml.md`, the untracked v1 note,
reviewed against:
- the S05 row of `docs/plans/connected-code-graph-p0.md` §3, plus §4.3, §4.6 and §5.3;
- the S24–S28 rows;
- proposal (`docs/plans/connected-code-graph.md`) §3.2, §3.7 and §4 P0(a);
- ML note (`docs/plans/connected-code-graph-ml.md`) §3.6, §4.5 and §8.1.

I re-derived every figure the note's decisions rest on with a scratch script (exact binomial,
Wilson z = 1.959964, exact McNemar, cluster-linearized ratio variance); the full tables are in
the Appendix. I web-checked the maintenance status of three of the seven reference tools.

**Verdict: needs changes.** There are no blockers, and the statistical core is correct: every
figure in the note reproduces. But several gaps must close before the steps that depend on them
can run:
- S26 cannot yet be briefed from §6 (M1).
- One rule change exceeds the note's authority and needs `architect` (M2).
- Three pieces of S24/S25 work are circular or have no owner (M3).
- The `flowAvailable` rule is internally inconsistent (M4).
- Golden flow sets and adjudication have no step (M5).

S24 and S25 can start once M3's ordering is settled. S26 waits for M1, S27's rule encoding
for M2, and S28 for M4 and M5.

CPG: not applicable — this is a statistical method note with no code-level component; `cpg/`
spike code does not exist yet.

### Findings

#### M1 (major) — §6 cannot brief a blind labeller on its own, and per-language conventions are missing

*Traced.* §6.1 gives the labeller "the label schema (§6.3) and the instructions in this section,
verbatim. Nothing else of this note." Several things the labeller needs are not in that text:
- **The truth-set definition.** §6.3 says "every target per §2's `T(s)`", but §2 (virtual
  dispatch "compatible with the receiver's static type", callbacks, `dynamic`) is not given to
  the labeller.
- **Line convention.** `defLine` has no stated rule. §5.3 uses the name's line with decorators
  excluded, but only the harness knows that.
- **What counts as in-corpus.** Are tests, generated files and a committed `vendor/` in the
  corpus? The labeller is not told.
- **Per kind:** what the target is for a constructor (the class or the constructor), a C++
  function-like macro, a JSX component and a Python decorator call. Go conversions as
  `notACall` appear only in §4.1, not in §6.
- **Per language:** the `qname` form (the example is Python-style) and the `external:<package>`
  form (Maven `groupId:artifactId`? Go module path? `<vector>` → `stdlib`?).
- "This section" can be read as §6.1 or as all of §6. All of §6 includes §6.4, the flow
  task, which does not belong in the call-site brief.

Without these, the seven dispatches get seven coordinator-written supplements, and the labels
diverge by language for reasons that are not about the frontend.

**Suggested improvement:** add a self-contained `§6.5 Labeller brief`, the only text pasted
into a brief. It should contain:
- `T(s)` verbatim;
- the in-corpus boundary and the `defLine` rule;
- a 7-row per-language table: qname form, external-package form, constructor, macro, JSX and
  decorator conventions, and what counts as `notACall`;
- the stop and resume rules.

S26's checker (`ds`) can then gate on "brief = §6.5 + `sample/<lang>.jsonl`".

#### M2 (major) — "indexer adopted on precision only if the Wilson upper bound is < 0.95" overrides a rule the P0 plan fixes, and part of its reasoning is wrong

*Traced across the docs.* Three sources fix the rule:
- **P0 plan §4.3** ("Decision rules, stated once") puts "resolved-call precision below 0.95"
  under *indexer adopted (correctness)*.
- **Proposal §3.2** says the same: "adopted … where P0 measures … resolved-call precision below
  0.95".
- **ML §8.1 m1** says "gate it with AC-5's confirmed-precision bar (≥ 0.95 observed)".

Only proposal §3.7 says "candidate". The plan delegates to S05 the sample, the truth source and
the `flowAvailable` criterion. It does not delegate this rule. **This needs `architect`.**

The reasoning is half right:
- **Right:** since P3-1, a same-name true callee is never dropped. And a wrong resolution whose
  true callee has a *different* name is already a recall miss under §2's `T ⊆ Res ∪ Nm`.
- **Understated:** a wrong resolution is not "only a trust cost". It creates a false
  *confirmed* edge, and confirmed precision ≥ 0.95 is a **gated** AC-5 bar at P1 (ML §4.2 (i)).
  With the upper-bound rule, a language whose point estimate is 0.90–0.94 enters P1 with no
  indexer. At n_P = 100 that is up to 9 errors (Appendix A4), on edges that feed that gate.

**Suggested improvement:** keep the plan's point-estimate rule, `P_c` < 0.95 → adopted, and
add the note's interval as reported evidence. Or have `architect` ratify the trust category
explicitly, together with the AC-5 consequence. In either case the note should state the AC-5
link. The n_P ≥ 60 floor with *insufficient* → listed is a sound addition within S05's remit.

#### M3 (major) — S24 and S25 are circular as written, and the fallbacks are software nobody owns

*Traced.*
- **The circle.** §5.2 says "S24 pins each to a version and **proves it on its corpus**". But
  the corpora are chosen in S25, and S25 depends on S24 (plan §3).
- **A contradiction.** C5 ("the reference indexes it") is a *must* criterion for choosing a
  corpus (§3.2), so a tool failure should exclude the corpus. §5.2 says it replaces the tool.
  And §3.2 "never revisits the pick after its first Joern run", so the reference must run
  before Joern on every candidate. No step's done-condition says so.
- **The fallbacks are programs.** Several are code to write, not tools to install: Pyright plus
  "a small go-to-definition driver" (an LSP client), a `tsc` language-service driver, a Roslyn
  driver, and a clangd-index reader. No step owns writing them. S24 (`devops`) installs tools;
  it does not write analysers.
- **Kotlin, web-checked.** `sourcegraph/scip-kotlin` was archived on 2026-07-02 and merged into
  `scip-java` (`scip-kotlinc`). Kotlin 2.x/K2 support is not stated anywhere, yet C7 admits
  Kotlin ≥ 1.9.
- **C/C++, web-checked.** The README asks for about 2 GB RAM per core. On the 6 GB host that
  needs an explicit `-j`.

**Suggested improvement:**
- S24 proves each primary tool on a smoke project. C5 moves into S25, run before any Joern run
  on a candidate, and a C5 failure picks the next candidate rather than switching tools.
- Fallbacks are allowed only where they are installed tools (`scip-go`, `javac -Xplugin:semanticdb`).
  Otherwise the fallback is "hand label only, flagged single-source", as for Kotlin.
- C7 pins Kotlin to the plugin's supported range.
- §5.2 records scip-clang's `-j` on this host.

#### M4 (major) — `flowAvailable` criteria 2 and 3 disagree; "≥ 24 of 30 fully recovered" is the real bar, and it sits well above 0.95

*Computed (Appendix A6).* The note frames the rule as "0.95 pair recall". The binding criterion
depends on how misses spread across methods. With independent misses and 5–8 golden pairs per
method, a language at exactly 0.95 pair recall passes criterion 3 only **8–47%** of the time.
It needs about 0.97–0.98 pair recall to pass about 85% of the time.

If instead the misses cluster in a few methods, criterion 3 is easy to meet, and criterion 2's
cluster-t lower bound does the work. At 0.95 pooled recall with 9 misses in 2 methods the
lower bound is 0.875; spread over 9 methods it is 0.921. The 0.85 floor seldom binds.

So the stated threshold and the effective one differ. Which one holds depends on an unknown
miss structure. S28 cannot report "why it failed" coherently.

**Suggested improvement:** choose one primary criterion. If "fully recovered" is kept, derive
its count from the pair-level target at the observed pairs per method, or relax it to about
≥ 20 of 30. Then state the operating curve as §4.3 does for call-site recall.

#### M5 (major) — four pieces of work have no step: the golden flow sets (Q1), drawing the flow methods, the reference truth run, and pre-Joern adjudication (Q2)

*Traced against the S24–S28 rows and §4.6.*
- S25's done-condition covers corpora, the two Joern configurations and the call-site sample.
  It does **not** cover drawing the 30 flow methods (§6.4), the tree-sitter frame (implicit),
  the per-frontend matcher table (§7.1, "the S25 author writes"), the share predictor (§4.3),
  or **running the references into `truth/`** (§5.1). Nothing else does either.
- The flow labels are **single-source**. The note's own F4 rejects a single labeller at a
  2-point tolerance, and the flow rule has a 5-point tolerance.

My answers to Q1, Q2 and placement are under Q4 below.

#### m1 (minor) — the 0.375 cap trigger sits at the coin-flip point, and its predictor is biased upward

*Computed (Appendix A3).*
- At an internal share of exactly 0.375, a 400 cap reaches 150 internal sites only **52%** of
  the time. A 90% chance needs a share of **0.406**.
- A 500 cap covers shares ≥ **0.30** at 50% and ≥ **0.326** at 90%.
- At a share of 0.25, even the floor fails: 48% of runs end below 100 internal sites at 400.
- The predictor, "same-name in-corpus definition", counts every name collision (`get`,
  `append`) as internal, so it overstates the share. The trigger fires too rarely.

**Suggested improvement:** drop the predictor and make the cap 500 everywhere. Or let the
labeller extend past 400 to 500. Either change keeps the draw independent of Joern, because the
extension depends only on the labels' internal flag, the same information the stop rule uses.
Then state the session-length cost.

#### m2 (minor) — the fold probabilities are per configuration, but the rule needs both configurations

*Computed (Appendix A4).*
- Per configuration, at true 0.97 the note's "~90%" is right (0.89–0.93 for n_P = 60–120). At
  true 0.93, "30–40%" holds only at n_P ≈ 60–80; at 100–120 it is 0.26–0.29.
- §8.3 needs **both** configurations to pass. With the two treated as independent, that gives
  **0.80–0.86** at 0.97 and **0.07–0.15** at 0.93. The true values lie between the
  per-configuration and the independent figures, because both configurations use the same sites.
- The "about ±0.05" interval is ±0.04–0.05 at n_P ≥ 100. At n_P = 60 it is [0.863, 0.983],
  a width of 0.12.

**Suggested improvement:** state the range for both configurations.

#### m3 (minor) — McNemar: "≥ 6 net discordant sites" is wrong

*Computed (Appendix A5).* 6–0 is the minimum, and the net count needed grows with the
discordant sites going the other way:

| Discordant (one way – other way) | Net | Exact p |
|---|---|---|
| 6–0 | 6 | 0.031 |
| 7–1 | 6 | 0.070 |
| 8–1 | 7 | < 0.05 |
| 10–2 | 8 | < 0.05 |

**Suggested improvement:** say "the exact test decides; the minimum is 6 discordant sites, all
in one direction".

#### m4 (minor) — the held-back exclusion targets the wrong genre, and the reverse direction is not recorded

*Traced.* The criterion for the third codebase is in ML §4.5 condition 3: "a different shop's
style, with a messaging stack and an in-house-looking wrapper". That describes real product
code, which demo-genre rules 2–3 do not catch, and which §3.2's own selection rule ("prefers
ordinary product code") pushes the pick toward.

Two smaller gaps:
- A corpus that is a subdirectory of an excluded repository is not explicitly excluded.
- Nothing makes the P0(a) corpora ineligible as the later held-back pick.

**Suggested improvement:**
- Keep an explicit held-back candidate register, checked by name in S25.
- Require a corpus to be a whole repository root.
- Record the five P0(a) corpora as ineligible wherever the held-back codebase is chosen.

#### m5 (minor) — tree-sitter needs to reach S09's brief

`py-tree-sitter` and the ten grammars belong in S09's `spike` extra (§5.2). But S09 depends only
on S03, and `cpg/pyproject.toml` does not exist yet. The dependency reaches S09 only if `teco`
puts it in S09's brief.

#### n1 (nit) — wording and unstated details

- §4.3 says "file order", while §6.3 says "ordinal order". Use ordinal.
- §4.3's "nothing below uses the mix": `P_c` pools the labelled internal and external sites. The
  inverse-sampling bias this introduces is about 1/150 and negligible, but the sentence is false.
- §7.3 does not write out the linearized ratio-variance formula, and "a `DATA` path (any access
  path, depth ≤ 3)" leaves unclear which depth is meant. S28 would have to make a method
  choice of its own.
- §6.2's "§6.5's method" is the headless audit. The S26 dispatches are `qa-engineer` agents,
  not headless sessions, so say which transcript gets grepped.

### Answers to the coordinator's questions

**Q1 — fit and completeness.**

| Consumer | Gets what it needs? | Gap |
|---|---|---|
| S24 | yes | the tool list (§9) is complete, but proving tools on corpora is circular (M3) |
| S25 | partly | its done-condition lacks the reference run, the flow draw, the matcher table and the predictor (M5) |
| S26 (×7) | no | the brief cannot be assembled from §6 alone (M1) |
| S27 | yes, apart from the M2 rule question | metrics, intervals, cause classes and the §8.5 table are mechanical |
| S28 | no | no producer of golden sets (M5), an inconsistent rule (M4), and an unwritten variance formula (n1) |

**Q2 — the numbers.** Every figure the note states reproduces (Appendix A1–A2). Your four spot
checks hold. The figures that move under a different reasonable definition:
- the 0.375 trigger (52% vs 90% reach: 0.375 vs 0.406);
- the fold probabilities (per configuration vs both);
- the `flowAvailable` bar (pair-level 0.95 vs an effective 0.97–0.98);
- McNemar's "net 6".

**Q3 — rule changes on its own authority.**

| Change | Within the note's authority? | Is the reasoning right? |
|---|---|---|
| 0.98 recall as a screen | **Yes.** It matches ML §3.6 ("below ~0.98 on the sample, lower bound reported") and the plan's "observed". The n ≥ 100 floor, *insufficient* = fail, and the worse of the two configurations are additions within S05's remit. | **Correct**, verified. `architect` should carry the "screen, not certificate" wording into the proposal at S50, since proposal §4 calls it a gate. |
| Fold/list, and adoption on the precision upper bound | **No.** It contradicts P0 §4.3 and proposal §3.2 (M2). | **Half right** (M2). |
| `flowAvailable` raised above 0.9 | **Yes.** The plan delegates the criterion and only "suggests" 0.9 over ≥ 20 methods. | The path-loss arithmetic is right (0.9³ = 0.729, 0.95³ = 0.857). The criteria are inconsistent (M4). |

**Q4 — the note's open questions.** My recommendations:
- **Golden flow sets: a new step, using free id S29 if `architect` agrees.**
  - Seven fresh `qa-engineer` dispatches, blind to S11 and `extract.sc`.
  - Inputs: a `sample/<lang>-flow.jsonl` that S25 draws (add it to S25's row), plus S03's point
    names.
  - They need S25 and S03, *not* S26. They can run alongside S26 as separate sessions; the
    note's "after S26" is only about session length.
  - S28 depends on the new step. Every projection miss gets the same cited-evidence adjudication
    as call-site misses, to give the flow truth a second check (M5).
- **Adjudication: a new pre-Joern step for disagreements and `unsure` sites.**
  - Owner `data-scientist`, checker `analyst`.
  - Inputs: `labels/` plus `truth/`, with no Joern columns. It runs after S26 and S25's
    reference run, and S27 depends on it.
  - Today the "before Joern's results are opened" ordering is enforced only by
    `data-scientist`'s own discipline inside S27, which `graph-dba` runs with Joern output to
    hand. A step boundary makes it checkable.
  - Miss adjudication has to see Joern, so it stays in S27, as an adjudication log that
    `analyst` checks independently. That keeps the checker from being the only check on its
    own adjudications.
  - A fresh `qa-engineer` adjudicator would also work, but the judgments are method calls, so
    `data-scientist` is the better owner provided `analyst` checks the log.

**Q5 — truth sources.**
- **Maintenance:**
  - scip-kotlin is archived and merged into scip-java, with K2 support unconfirmed.
  - scip-clang and scip-java show active repos.
  - I did not verify scip-dotnet's activity, and none of the three READMEs says whether it
    emits `is_implementation`.
- **Can S24 fail cleanly with the stated fallback?** Not as written (M3). The fallbacks are
  unowned code, and C5 conflicts with "replace the tool".
- **Does the polyglot exclusion keep the held-back pool clean?** It guards the demo genre only
  (m4).

### What's solid

- **F1, the frame independent of Joern.** Without it, recall would be overstated exactly where
  the frontend is weakest.
- **F2, a simple random sample instead of strata.** Stratifying would be biased across the two
  configurations.
- **F3's inverse sampling, with a stop that depends only on the labels.** This is valid, and
  plain Wilson intervals apply.
- **The draw is a hash sort**, so any tool reproduces it.
- **The strict, site-level recall unit** respects repeated measures.
- **Misses are classified by cause**, and the cause routes the remedy.
- **The paired McNemar comparison** of the two configurations.
- **§9 lists real image gaps**: Node.js and the dependency-cache separation.
- **The note is honest** that the 0.98 figure cannot be certified at P0.

### Open questions

1. *For `architect`.* The M2 rule: point estimate or upper bound for adopting an indexer on
   precision, with AC-5's confirmed-precision gate in view.
2. *For `architect` and `teco`.* Whether to add the two new steps (flow labelling; pre-Joern
   adjudication), and how to amend S25's done-condition (M5). These belong in the queued v1.4
   amendment.

### Appendix (S05 computations)

Wilson z = 1.959964, exact binomial. Script:
`/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/13504717-4242-422a-b7d8-a910a6cc9348/scratchpad/s05.py`
(and `flow.py` for A6).

**A1. Wilson table at n = 150.** All five rows match §4.3:

| Misses | Interval |
|---|---|
| 0 | [0.975, 1.000] |
| 1 | [0.963, 0.999] |
| 2 | [0.953, 0.996] |
| 3 | [0.943, 0.993] |
| 4 | [0.933, 0.990] |

**A2. Pass probability, ≤ 3 misses of 150.** All match the note:

| True recall | 0.995 | 0.99 | 0.985 | 0.98 | 0.97 | 0.96 | 0.95 | 0.93 |
|---|---|---|---|---|---|---|---|---|
| P(pass) | 0.993 | 0.935 | 0.811 | 0.647 | 0.338 | 0.146 | 0.055 | 0.006 |

Other figures, all matching:
- Smallest zero-miss n with a Wilson lower bound ≥ 0.98: **189**, and 0.99^189 = 0.150.
- The 200-site "≤ 4 misses" rule: passes 0.948 at true 0.99 and 0.026 at 0.95; the lower bound
  at 196/200 is 0.950.
- At n = 100: zero misses gives a lower bound of 0.963. The "≤ 2 misses" rule passes 0.921 at
  0.99 and 0.118 at 0.95.
- ⌊0.02·n⌋ is 2 for n = 100–149 and 3 at 150. At n = 149, P(pass | 0.99) = 0.81.

**A3. The cap.** P(reach 150 internal sites):

| Internal share | 400 cap | 500 cap |
|---|---|---|
| 0.30 | 0.001 | 0.517 |
| 0.33 | 0.032 | 0.931 |
| 0.35 | 0.16 | 0.992 |
| 0.375 | 0.519 | 1.0 |
| 0.40 | 0.858 | |
| 0.42 | 0.97 | |

Share needed to reach 150:

| Target chance | 400 cap | 500 cap |
|---|---|---|
| 50% | 0.374 | 0.300 |
| 90% | 0.406 | 0.326 |
| 95% | 0.415 | 0.334 |

P(< 100 internal sites at 400) is 0.481 at a share of 0.25 and 0.012 at 0.30.

**A4. Precision.**
- Wilson at 57/60: [0.863, 0.983]. At 95/100: [0.888, 0.978]. At 114/120: [0.895, 0.977].
- P(fold), per configuration:

| n_P | 60 | 80 | 100 | 120 |
|---|---|---|---|---|
| at true 0.97 | 0.894 | 0.907 | 0.919 | 0.930 |
| at true 0.93 | 0.387 | 0.333 | 0.291 | 0.257 |

- With both configurations treated as independent: 0.80–0.86 at 0.97 and 0.07–0.15 at 0.93.
- Adopting on the trust rule (upper bound < 0.95) needs at least 7 errors at n_P = 60 (point
  0.883), 10 at n_P = 100 (point 0.900) and 11 at n_P = 120 (point 0.908).

**A5. McNemar, exact two-sided.**

| Discordant | 5–0 | 6–0 | 7–1 | 8–2 |
|---|---|---|---|---|
| p | 0.0625 | 0.031 | 0.070 | 0.109 |

The minimum one-way count is 6 when the other way is 0, 8 when it is 1, 10 when it is 2, and
12 when it is 3.

**A6. Flow.** 0.9³ = 0.729 and 0.95³ = 0.857.

P(≥ 24 of 30 methods fully recovered), with misses independent across pairs:

| Pairs per method | at 0.95 | at 0.97 | at 0.98 |
|---|---|---|---|
| 5 | 0.466 | 0.879 | 0.979 |
| 6 | 0.282 | 0.775 | 0.952 |
| 8 | 0.078 | 0.519 | 0.850 |

Cluster-t lower bound at 0.95 pooled recall (30 methods × 6 pairs, t₂₉ = 2.045):

| Misses spread over | Lower bound | Methods fully recovered |
|---|---|---|
| 9 methods | 0.921 | 21 (fails criterion 3) |
| 3 methods (pooled 0.956) | 0.904 | 27 |
| 2 methods | 0.875 | 28 |

### Pass 2

**Reviewed (2026-10-29):** `docs/plans/connected-code-graph-p0-ml.md` v1.1, specifically §2,
§3.2–§3.3, §4.3, §5.2, §6, §7.3–§7.4, §8.3–§8.4, §11, §12 and the Appendix. I checked the
figures with my own scripts, not the author's:
- `scratchpad/s05p2.py`: exact binomial for the independent case, and a beta-binomial Monte
  Carlo with 40k replications and seed 20261029, which is a different method from the author's
  convolution.
- `scratchpad/s05p2b.py`.

I also confirmed that the note's Appendix script is byte-identical to the scratch copy at
`scratchpad/ds/oc_clean.py` (`diff`, no output).

**Verdict: approve with suggestions.** Every Pass 1 finding is fixed or specified, and every
figure I re-ran reproduces. Two items need attention before the steps that use them: P2-1 must
be fixed before S27 encodes §8.3 (N5), and P2-2 should be fixed before S25 fills the briefs
(N2). The N-steps still need `architect` to write them into plan v1.4. That is outside the
note.

CPG: not applicable — the artifact is a statistical method note, and no code is involved.

**Dispositions of the Pass 1 findings.**
- **M1:** fixed. §6.5 is a brief S25 can fill and paste, with slots, a 7-row table, `T(s)`, the
  `defLine` rule, the corpus boundary, and stop and resume rules. Two small gaps remain (P2-2).
- **M2:** fixed. §8.3 now makes the point-estimate rule operative: observed `P_c` < 0.95 in
  either configuration, with n_P ≥ 60, adopts the indexer. The upper-bound rule is labelled
  "not in force". Adoption ("either configuration fails") is the exact complement of folding
  ("both pass"). Re-run: P(adopt) is 0.07–0.11 at true 0.97 and 0.61–0.74 at true 0.93,
  matching §8.3.
- **M3:** fixed. S24 now proves tools on smoke projects. C5 now carries a ≥ 90% pilot and runs
  before any Joern run, and a failure moves to the next candidate. Fallbacks are installed tools
  only. C7 now holds Kotlin to the plugin's range. scip-clang runs with `-j 2`.
- **M4:** fixed. There is one primary criterion. The operating curve reproduces with my
  instrument (exact for the independent case, Monte Carlo for ρ > 0); every cell of §8.4's table
  is within ±0.01. Examples:
  - m = 6, independent: 0.996 / 0.954 / 0.814 / 0.587 / 0.184 / 0.012 at true recall
    0.98 / 0.97 / 0.96 / 0.95 / 0.93 / 0.90.
  - m = 6, ρ = 0.3: 0.955 / 0.867 / 0.741 / 0.586 / 0.304 / 0.079 at the same recalls.
  - "≥ 0.98 passes ≥ 94%" and "≤ 0.90 fails ≥ 92%" both hold.
- **M5:** specified as N0–N7, and concrete enough to write step rows from. Minor gaps are under
  P2-4.
- **m1:** fixed. Re-run at a 500 cap:
  - P(reach 150) is 0.517 at a share of 0.30, 0.902 at 0.326 and 0.992 at 0.35;
  - P(< 100) is 0.482 at 0.20, 0.0036 at 0.25 and about 0 at 0.30.
- **m2:** fixed.
  - Wilson intervals reproduce: [0.863, 0.983] at n_P = 60, [0.888, 0.978] at 100 and
    [0.895, 0.977] at 120.
  - Both-configuration ranges reproduce: 0.80–0.93 at true 0.97 and 0.07–0.39 at true 0.93. The
    low end of each range is the independent product at the largest n, and the high end is
    the per-configuration figure at the smallest n.
- **m3:** fixed. The exact minima reproduce: 6–0 (p = 0.031), 8–1 (0.039), 10–2 (0.039), 12–3
  (0.035).
- **m4:** fixed. §3.3 now has the broker-client exclusion, a named register, whole repository
  roots only, and the reverse rule.
- **m5:** specified as N6.
- **n1:** fixed. Ordinal order is used throughout. The bias sentence is corrected. The variance
  formula and the `DATA`-path rule are written out (§7.3). The check now names the subagent
  transcript.

#### P2-1 (major, new) — the declared set `D(s)` counts resolutions that create no edge as correct, which pushes precision toward 1 in exactly the interface-heavy languages

*Traced: §2 lines 71–74 and 94; bundle spec §4.1 rule 8 and §5.6.*
- **The contradiction.** §2 brings in `D(s)` "matching the bundle spec's rule that a bodiless
  declaration is not a symbol". It then defines *correctly resolved* as `Res_c(s) ⊆ T(s) ∪ D(s)`.
  But under that rule, a resolution to an interface or abstract method has no `symbolId` to
  target. So the bundle's `targets` cannot hold it, and the site becomes `possible`/`name-match`
  with **no confirmed edge**.
- **The note's own rule points the other way.** It excludes "resolved away" sites from n_P
  because they create no confirmed edge. A D-only resolution is the same case, yet it is
  counted as a correct confirmed resolution.
- **Why it matters.** C6 *requires* interface dispatch in Java (≥ 10 interfaces), Go (≥ 10) and
  C++ (≥ 10 virtuals), so D-only resolutions will be common there. Counting them as correct
  raises `P_c`. That makes folding more likely and **suppresses the M2 adoption rule** on the
  very edges that feed AC-5's confirmed-precision gate.
- **Recall is not affected.** `D` is not in `T`, and implementations are reached by name match.

**Suggested improvement:** treat a resolution whose targets all lie in `D(s)` as *resolved to
declaration*:
- exclude it from n_P, as "resolved away" is;
- report its count per configuration;
- count a mixed `Res` by its symbol targets only.

If `architect` instead wants the assembler to map an interface-method target to its
implementations, that is a bundle-spec change and needs ratifying there first.

#### P2-2 (minor, new) — §6.5 leaves out two things a pasted brief needs

*Traced in §6.5.* After S25 fills the slots, the brief pastes as it is: it contains no Joern
detail, no threshold and no forbidden path beyond its own list. Two gaps remain:
- **The sample record is never described.** The brief doesn't say that `line` and `column`
  point at the **callee expression's first character** and that `callText` is that expression
  (§2). On a line such as `a.b().c()` with two sampled sites, the labeller has to guess which
  call is meant.
- **The inheritance rule has no stated end.** "The method that type has or inherits from its
  nearest in-corpus ancestor" doesn't say what to write when that method is inherited from an
  external base class and not overridden in the corpus. (It should be `external`, with any
  in-corpus overrides in subtypes listed as `internal`.)

**Suggested improvement:** add one sentence for each.

#### P2-3 (minor, new) — the *truth-incomplete* flag fires on any small spurious set or on a single omission

*Computed (`s05p2b.py`).* §8.4 flags a language when the Wilson upper bound of the omission rate
is above 0.10, over "up to 50" sampled spurious pairs:
- With 0 omissions, the upper bound is above 0.10 whenever n < 35: 0.114 at 30 and 0.161 at 20.
  A good projection with few spurious pairs is therefore always flagged.
- At n = 50, a **single** omission gives 0.105, which also trips the flag.

**Suggested improvement:** flag on the point estimate, for example > 0.05 with k/n and the
Wilson interval shown. Or require n ≥ 35 before the flag can apply, and report "insufficient"
otherwise.

#### P2-4 (minor, new) — small gaps in N0–N7, and one frame/bundle mismatch

*Traced in §11 and §4.1, and in bundle spec §5.6.*
- **N4:** the harness that generates `adjudication/<lang>.input.jsonl` has no owner. It must be
  run by someone who never opens `results/`.
- **N5:** it doesn't say that the miss log's overturns are applied to `truth/<lang>.final.jsonl`
  **before** the metrics are computed. Read literally, the metrics come from the N4 truth.
- **N3:** no checker is named. S26 has `ds`.
- **N1:** the C5 pilot's seed prefix is not stated (§4.4 lists `sym`, `flow` and `spur`, not a
  pilot prefix).
- **Vendored code:** the brief makes calls into committed vendored code `external`. But nothing
  stops Joern from parsing those directories, and a Joern `METHOD` there would make `Nm(s)` and
  `Res_c(s)` "in-corpus". The harness should exclude vendored paths from Joern's input, or
  treat their `METHOD`s as external.
- **Decorators:** the Python and TS frames, and the brief's table, label decorator calls. But
  bundle spec §5.6 says decorator applications are **not call sites**; they are `decorator`
  records. Report the decorator kind outside `R_c`, or justify keeping it in.

**No other contradictions found** among the fixes:
- §2's `declared` set against recall's `T ⊆ Res ∪ Nm`: consistent. P2-1 is a precision-side
  problem only.
- The ≥ 120-pair floor against the curve's m = 5–8 (150–240 pairs): consistent.
- The 500 cap in §4.3, §6.5's `{CAP}` and R5/R7: consistent.
- N2's internal order inside S25 (frame → C5 → draws and briefs → Joern) against C5's row:
  consistent.

### Pass 3

**Reviewed (2026-10-29):** `docs/plans/connected-code-graph-p0-ml.md` v1.2 — §2, §4.1, §6.5,
§7.1, §8.3, §8.4 and §12 Pass 2 — against bundle spec v1.1 (§4.1 rules 8 and 10, §5.6, §8). The
P2-3 numbers were re-run with my own script, `scratchpad/s05p3.py`.

**Verdict: approve.** All of P2-1 to P2-4 are resolved. Two questions for `architect` are left;
neither changes the protocol.

CPG: not applicable — the artifact is a statistical method note, and no code is involved.

**Dispositions**
- **P2-1: superseded and fixed.** My finding rested on bundle spec v1.0 rule 8 ("only
  definitions with a body are symbols").
  - v1.1 rule 8 makes every member declaration a call can resolve to a symbol: Protocol and
    abstract methods, Java/Kotlin/C#/TS interface and abstract methods, C++ pure virtuals, and
    the Kotlin primary constructor. A resolution to one is therefore a real `CALLS` edge to a
    real `Symbol`, so the note is right to count it as correct.
  - The `impl`/`decl` roles keep recall decided by implementations. The share of sites
    *resolved to declaration* is reported, not hidden.
  - Where v1.1 still yields no symbol (overload stubs, `.d.ts`, prototypes, bodiless accessors,
    and Go interface method specs), the new `N(s)` plus *resolved to non-symbol* excluded from
    n_P is exactly my Pass 2 remedy. It is applied where it now belongs.
  - I checked the claim that "`decl` is always name-matched" against rule 10. It holds even for
    constructors, because a `<init>` symbol's `name` is the type's name, which equals the
    `calleeName` `Foo`.
- **P2-2: fixed.**
  - The brief now says which call a sample line means: the start of the callee expression plus
    `callText`, with the `a.b().c()` example. It also covers the out-of-corpus base rule.
  - The matcher now joins on `nameLine` and `calleeName`. Both match §5.6's definitions: `line`
    is the line of the callee's last identifier, and `calleeName` is that identifier as written,
    with `new Foo` giving `Foo`.
  - Any residual Joern line convention is still caught by §7.1's rule that no metric is computed
    until zero matcher errors remain.
- **P2-3: fixed, and the numbers reproduce.** The rule flags when the observed omission rate is
  above 0.05 with n ≥ 20, and otherwise reports *insufficient*.

  | n | Omissions needed to flag | P(flag) at true rate 0.02 | 0.05 | 0.10 |
  |---|---|---|---|---|
  | 50 | 3 | 0.078 | 0.459 | 0.888 |
  | 20 | 2 | 0.060 | 0.264 | 0.608 |

  Both rows match §8.4. The flag is a report qualifier, so this sensitivity is enough.
- **P2-4: fixed.**
  - The N3/N4/N5/N7 gaps are closed, and the C5 seed prefix `ccg-p0a-v1-c5|` is stated.
  - **Vendored code.** Joern keeps parsing vendored code, but its `METHOD`s there count as
    external, are left out of `Nm(s)`, and resolutions into them count as *resolved away*. This
    is consistent with the labels, and the reason for rejecting the alternative is sound: dropping
    vendored code would distort the offline configuration, given Go's vendored-only policy.
  - **Decorators.** They are removed from the frame and the brief, which is consistent with
    §5.6, where decorators are `decorator` records. They are measured in the census instead.
    Calls inside decorator arguments are also dropped. Those are real call sites under §5.6, so
    this narrows scope a little, but it is stated as a blind spot. That is acceptable.

**Open questions for `architect` (not defects in the note):**
1. **Go interface method specs.** Rule 8's general clause ("every member declaration a call can
   resolve to") arguably covers them, but its list of examples leaves Go out. The note treats them
   as non-symbols "until the spec says otherwise". The bundle spec should settle this, because it
   moves Go's interface-call sites between n_P and *resolved to non-symbol*.
2. **Vendored code in production.** The bundle spec doesn't say whether committed vendored code
   becomes in-repo `Symbol`s. If it does, P1's name-match will include vendored candidates that
   the P0 fan-out and precision figures leave out. Those figures would then need re-reading.

## S03 (P0-03a)

**Reviewed (2026-10-29):** `docs/plans/connected-code-graph-bundle-spec.md` v1.0 (v1 candidate,
untracked), against the S03 row of `docs/plans/connected-code-graph-p0.md` §3, its §2 D5, D8,
D10, §4.2, §4.4 (S14/S15) and §5.3; proposal `docs/plans/connected-code-graph.md` §3.3 and §3.8;
graph note `docs/plans/connected-code-graph-graph.md` §3.2, §3.4 and §6; egress spec
`docs/plans/connected-code-graph-egress-spec.md` §5.1, §5.3 and R6. The graph-model mapping (§9)
is `graph-dba`'s half and is touched here only where it bears on ids or authority.

**Verdict: needs changes.** There is one blocker:
- **F1:** the `fieldAccess` id collides on code that is in the slice.

Four majors must be fixed before S09/S10 encode the spec, because each one is a tuple or
grammar fact, and after S15 such a fact can only change through a major bump (§8.6):
- **F2:** P-slice `qname` gaps;
- **F3:** ordinal locators;
- **F6:** `bindValue` exposes URL-path secrets;
- **F7:** an S09 golden-vector gap.

The rest of the spec is sound: the hash, the `repoId`, the `http` `keyNorm`, the freeze
condition and the determinism rules. The fixes are local edits to §4.1, §4.2, §5.13, §5.17 and
§3.6. Since nothing has been executed against v1.0 yet, they are an in-place revision, not a
successor.

CPG: considered, not relevant — the artifact is a spec document. Its source-side claims were
checked by reading the files and with a read-only Python `ast` probe, not through a graph.

**What I verified:**
- **All 14 golden vectors reproduce** from §3.1's rule, run as a scratch script under
  Python 3.12. That includes GV-14's embedded `getHealth` id `8ae57f34…` and GV-10's 5,032
  canonical bytes.
- With the type tag, `["callsite",S,"param",0]` ≠ `["point",S,"param",0]`. BF1's fix holds.
- **Cited lines.** Every Python and TS example line in §4.2 exists: `storefront_api.py:133`,
  `:321`, `:857`, `:879` and `:946`; `app.py:498`; `guards.py:147–149` (the first three
  module-level lambdas); `client.ts:73`, `:91`, `:95`, `:99–100` and `:107`; and `endpoints.ts:8`,
  `:10` and `:16`.
- **Hand derivation.** Each `qname` in §4.2's tables re-derives by hand to the single value the
  spec gives. I also derived symbols the spec doesn't list, and each had one answer:
  `StorefrontHTTPError.__init__`, `build_storefront_router.get_participant`,
  `timedSignal.<anon>#1`, and an arrow inside `useCallback(…)`, which is `<anon>`.
- **Duplicate-name probe.** An AST pass over all of the repo's Python found **no** same-scope
  duplicate names, so rule 7 (`#k`) is not exercised by the slice's Python.
- **Not checked (no `node` on this host):** a TS AST pass for duplicates, accessors and
  string-keyed function properties. I grepped `salesperson/src` and found none of the three, but a
  grep is a bound, not a parse.

### Findings

#### F1 (blocker) — the `fieldAccess` id omits `path`, so one call reading a field twice is a duplicate id, and §8.3 makes the writer fail the unit

*Confirmed on slice code (AST probe; Appendix S03-A1).* The §5.13 tuple is
`["fieldAccess", pointId, mode, receiverType, field]`. Take
`falkor-chat/server/falkorchat/tools.py:1480`,
`self._loop.call_soon_threadsafe(self._loop.stop)`:
- it reads `self._loop` at `recv` and again at `a0`;
- the point is the same (the call site), the mode is `read`, and `receiverType` resolves to the
  enclosing class (§5.13: `self` resolves);
- the field is `_loop` both times.

So the two records have one id. `services.py:966` and `:1000` (`self._repo` in two keyword
arguments) collide the same way. §8.3: "the writer must fail rather than drop one". So
`falkor-chat/server` cannot be bundled once fields are emitted. Its §10 mark is "P-slice if the
slice emits fields". If the slice doesn't emit them, P1 hits this on day one, after the freeze.
That makes it a major bump.

**Fix:** use `["fieldAccess", pointId, path, mode, receiverType, field]` (`path` is already a
required field). Add a golden vector for two records that differ only in `path`. S10 should add a
test that writes two such records and gets two ids.

#### F2 (major) — P-slice `qname` gaps that a source-only key writer, and S11's re-derivation check, will hit

*Probed (AST pass; Appendix S03-A1).*
- **(a) Python `Protocol` and abstract stubs.** §4.1 rule 8 says "abstract or interface method
  signatures … are not symbols", but every Python `def` has a body. The slice unit has five
  `Protocol` classes with `...`-bodied methods: `falkorchat/embedding.py:37`, `tools.py:121`,
  `llm.py:81`, `executor.py:181` and `modelconfig.py:236`. The repo has 13. If the spec is silent
  here, the normalizer decides, and S11's "re-derive from source" check is circular.
  - The choice is also a correctness one. Joern resolves `embedder.embed(…)` to the stub. If the
    stub is not a symbol, `targets[]` names an id that has no record.
  - **Fix:** "A Python `def` is a symbol unless it is `@overload`-decorated or in a `.pyi` file.
    `Protocol` and `@abstractmethod` methods are symbols."
- **(b) Decorator and default-value lambdas.** Rule 2's "contains" is read textually, and
  `lineStart` includes decorators. Python evaluates both positions in the enclosing scope.
  - The case occurs in the slice unit: `@pytest.mark.parametrize(…, ids=lambda c: …)` at
    `falkor-chat/server/tests/test_fusion.py:241`. There are 7 repo-wide.
  - The two readings give `test_x.<anon>#1` and a module-level `<anon>#k`, and the second shifts
    every later module-level ordinal.
  - **Fix:** one sentence: "decorator expressions, parameter defaults and annotations belong to
    the enclosing scope" (or the reverse; either works if stated).
- **(c) TS string-literal object keys.** In §4.2, "a string `KEY` gives `KEY`", so
  `{"a.b": () => 1}` or `{"x#2": fn}` inject `.` or `#N` into the chain. They can collide with
  real segments, and then §8.3 fails the unit on arbitrary third-party code. **Fix:** only a key
  that is a valid identifier names the value; any other key gives `<anon>#N`. (Absent from
  `salesperson`; TS not parsed, see above.)
- **(d) TS named function expressions** (`const f = function g() {}`,
  `arr.map(function mapper() {})`). The spec yields `f` and `<anon>#N`, but its appeal to "ES2015
  inference" points a reader to `g` and `mapper`, because inference applies only to *anonymous*
  functions. **Fix:** state the rule explicitly.

#### F3 (major) — ordinal tokens are stable at `EVAL_COMMIT`, but hard to write by hand, and they re-bind on insertion

- **For ground truth, no defect.** Keys and P1 builds both pin `EVAL_COMMIT` (D8), so `#N` means
  the same thing on both sides.
- **For key writers, error-prone.** A writer must count anonymous scopes by (line, column) under
  one parent chain. In `salesperson`, every test is a nested `describe`/`it` arrow:
  `src/api/dispatch.test.ts` has 44 `=>`, and a test is `path::<anon>#i.<anon>#j`. A
  miscount gives a wrong-but-valid id, and the S39 "qname found in file" check cannot catch it,
  because every `<anon>#N` up to the count exists.
- **Outside evaluation, it re-binds.** Inserting an earlier sibling turns the old `#1` into a
  *different* element. FR-29's fingerprint check then reports "possible, earlier verdict noted",
  and that note is about another function. Proposal §3.8's orphan causes (rename, move,
  signature) do not include this.

**Fix:**
- §3.5: let a locator for an `<anon>` scope (and a `#k` duplicate) be its nearest named
  ancestor's locator plus a required `line`. The id converter derives the ordinal mechanically
  from source at the pinned commit.
- S06 decides whether anonymous scopes are eligible as anchors.
- Add the re-binding to BF2's list, for `architect` to carry into proposal §3.8.

#### F6 (major) — `bindValue` rule (a) lets a URL-path secret through in clear

*Traced from §5.17's text.* `https://hooks.slack.com/services/T0/B0/XXXX` has no userinfo and no
query, so it qualifies as a binding URL. A `keyPath` such as `notify.webhook` matches none of the
secret-name words. Its secret is in the path, so it would be written to the bundle in clear.
"A path" and "a host name" are open classes too, and the name list misses `dsn`, `pwd`, `conn`
and `webhook`.

The bundle is a file on disk (§2.1), not only graph input, so it does not matter that §9.1 keeps
`bindValue` out of the graph. The spec already marks the field [P-later] and asks for review
(BF8).

**Fix, and an answer to Q4:**
- **Yes, commission the optional `security-expert` advisory before P1.** Give it this case.
- Meanwhile, narrow (a): a URL qualifies only with a path of at most one segment or no
  high-entropy segment. Or drop the URL class until the advisory, since S12's slice binds by
  constant, not by config value.

#### F7 (major) — no golden vector pins the 5-element signature rule, and none checks a padded integer element

- **Signature placement.** GV-7/GV-8 test a Java signature, but no vector shows that a Python,
  TS or Go symbol gets **no** 5th element. Test 6 of S09's own row ("a signature is accepted only
  for Java, Kotlin, C# and C++") has nothing to reproduce. That is the one rule whose violation
  silently re-ids every symbol.
- **Strings vs integers.** No vector tests that normalization strips strings and leaves integers
  alone: `["point", S, "param", -1]`, the receiver, is absent.

**Fix:** add three vectors:
- a C symbol in a `cpp` unit (4 elements, BF11);
- a Kotlin extension with signature `String.()`;
- the receiver point `-1`.

With graph-dba's suggestion, the `nameKey` vector makes four.

#### F4 (minor, P-later) — internal contradictions in grammars the slice can't test

- **Bodyless symbols in examples.** Rule 8 says only definitions with a body are symbols, yet the
  C# example lists an auto-property `Box`1.Value.<get>` and the Kotlin example lists a primary
  constructor `Orders.<init>`. Neither has a body. Rule 4 also sends Kotlin `init {}` code to the
  type, not to `<init>`.
- **C# extension `this`.** It is neither removed nor kept-with-space, so whitespace removal gives
  `thisstring`.
- **C# explicit interface implementations** (`void IDisposable.Dispose()`) have no rule.
- **Explicit receivers.** Go and Kotlin-extension receivers have no `param` index (§5.4 defines
  `-1` for Python only).
- **`.h` language.** It depends on whether the unit has any C++ file, so adding one `.cpp` re-ids
  every header symbol, which orphans their verdicts.
- **Extensionless shell.** §3.5 says "`lang` comes from `path`'s extension", but an extensionless
  shell script takes its language from the shebang.

**Fix:** a short §4.2 errata pass by `architect`. It needs no slice evidence.

#### F5 (minor) — two encodings for one `data` fact

`usePath` allows both `""` and `"ret"` for a value landing on a `Ret` point, and the AC-1 row uses
`""`. Two normalizers can differ, which shows up as an id diff in S11 or SL-6. **Fix:** "`ret`
heads appear only on … (or nowhere); a `Ret` use point takes `""`".

#### F8 (minor) — §8 determinism: three gaps in an otherwise mechanical rule

**Q6:** §8 is checkable as S14 needs it. Drop the `builtAt` and `sandboxAuditIds` keys from line
1, then `cmp` the files (§8.5). Every array is either sorted on a total key or in a stated order,
with three exceptions:
- **(a) `import.names` is sorted "by `name`" only.** `import {a as b, a as c}` ties. **Fix:** sort
  by (`name`, `alias`), with an absent alias first.
- **(b) The `fingerprint` line rule has no line splitting or whitespace set.** Python's
  `splitlines()` also splits on U+2028, `\x0c` and `\x1c`. **Fix:** "split on `\n`; strip per
  §3.1's `isspace` set".
- **(c) `header.history` depends on the source clone.** A shallow clone gives `unavailable`, and
  that input is missing from §8.4. **Fix:** "a shallow or missing history is a build error in
  P0" or list it as an input.

### Q4 — authority of the spec's own changes

| Finding | Within a spec's authority? | Right? | Who must agree |
|---|---|---|---|
| **BF3** `use`/`def` | Yes. P0 §4.2 is a "must contain" list written by the same contract owner, and the edge property names are unchanged. | Yes. `graph-dba` agrees above. | `teco`: one ledger line that P0 §4.2's `{from, to}` is superseded by the spec. No plan edit. |
| **BF4** `recv`, `kw.` | Bundle side yes. It extends graph-note §6.3's value set. | Yes. SL-2's `prefix=` needs it. | `graph-dba` (accepted above, with `arg(head, path)` binding by `Param.name`). S17 records it. Take `graph-dba`'s `["…"]` quoting for dotted keys now. |
| **BF5** three records | Yes. S03's row says "every record type", and S12 cannot resolve `{ref}` without `constant`. | `constant` and `schemaEntry` yes. `fieldAccess` yes, after F1. | none |
| **BF8** HMAC `valueHash` + `bindValue` | The HMAC is yes: the graph note's own FR-15 argument, and it is not id-bearing (`entryId` excludes it). `bindValue` is yes as a field, but its classifier is a security policy. | HMAC yes. `bindValue` not yet (F6). | `security-expert` advisory before P1 (recommended), and `graph-dba` for the property semantics. No stakeholder decision is needed, since "values never stored" already decides the intent. |
| **BF14** hash `nameKey` | It is a recommendation to S17, not a change. | Yes. It is the only unhashed `UNIQUE` value. | `graph-dba` ruled to adopt it now. Add the tuple row and a vector (F7). |
| **BF15** `sandboxAuditIds[]` | It deviates from P0 §4.2's singular "the sandbox audit id". The egress spec explicitly delegates the field name to S03 (§5.3, R6). | Yes. S11 runs two sandbox invocations per bundle. | Egress author (`architect`): §5.3's loader check must read "every id in `sandboxAuditIds`", and its "output file name stem" sentence must apply to the tool's outputs. `teco` reconciles under R6 at this gate. |

### Q5 — freeze condition

- **Consistent with D5 and S15.**
  - §1 freezes v1 only when all of SL-1–SL-7 pass, and stays provisional to P1 exit on any failure.
  - It writes a successor (S16) for any SL-7 delta, which matches P0 §4.4's "freeze applies to the
    amended schema, which S16's successor records".
  - It forbids a partial freeze (SQ-P0-2).
- **[P-later] freezing untested is not a contradiction of D5.** D5 freezes "schema v1" as a whole
  on a passing slice and rejects partial freezes. Excluding the P-later items from the freeze
  would *be* the partial freeze D5 rules out.
- **The cost is real and should be visible.** F4 shows that the P-later grammar already has
  defects. Each one found after the freeze that touches a tuple or a `qname` is a major bump and a
  successor (§8.6), and it orphans verdicts.
- **Suggestion:** S15's gate record should name the P-later items as "frozen, untested" explicitly.
  F4 should be fixed before S09 so the untested part is at least internally consistent. No
  stakeholder question is needed.

### What's solid

- **The hash (§3.1).** It is exact to the byte: a type-tagged tuple, NFC then a named `strip` set,
  and a Python one-liner. All 14 vectors reproduce.
- **Path-anchored `qname` (§4.1 rule 1).** It is the right call. It is writable from source, it
  avoids `unitId` (F2), and its rejected-alternative reasoning (two `tests/conftest.py`) is sound.
- **§3.5's prefix rule** makes S39's check mechanical.
- **The `repoId` rule and literal** are complete.
- **The `http` `keyNorm` (§6)** is thorough and checkable: mount composition before
  normalization, the trailing `/` kept, and `{?}` for unresolved segments.
- **The freeze section** is faithful to D5.
- **The findings table** records every deviation openly, as the authority clause promises.
- **§8** is mechanical apart from F8's three gaps.

### Open questions

1. *For `architect`.* F2(a): should `Protocol` and abstract stubs be symbols? I recommend yes,
   because Joern resolves calls to them. Either way, the answer is a fact S11's re-derivation
   check needs before it runs.
2. *For `teco` and `data-scientist` (S06).* F3: are `<anon>` scopes eligible as key and gold
   anchors? If they are, adopt the ancestor-plus-line locator.
3. *For `teco`.* Should the `security-expert` advisory on §5.17 (F6) run before S09? I recommend
   it runs before P1, with F6's interim narrowing applied now.

### Appendix S03-A1 — probes

- **Golden vectors:** recomputed with
  `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/13504717-4242-422a-b7d8-a910a6cc9348/scratchpad/gv.py`.
  Fourteen rows `True`.
- **Python AST probe (whole repo):**
  `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/13504717-4242-422a-b7d8-a910a6cc9348/scratchpad/pyq.py`
  found 0 same-scope duplicates, 393 lambdas, 7 decorator or default-value lambdas, and 13
  `Protocol` classes (5 in `falkor-chat/server/falkorchat`).
- **F1 probe:** calls in `falkor-chat/server` where the same `self.<attr>` is read in two distinct
  positions (receiver and arguments). Four hits:
  - `tests/test_ensure_demo_models_script.py:136`
  - `falkorchat/tools.py:1480`
  - `falkorchat/services.py:966`
  - `falkorchat/services.py:1000`

### Pass 2

**Re-reviewed (2026-10-29):** `docs/plans/connected-code-graph-bundle-spec.md` v1.1, including
its §13 dispositions. The confirmations below were executed with my own scripts (Appendix
S03-A2), not the author's.

**Verdict: approve with suggestions.**
- F1–F8 are all fixed; F6 is an interim fix, by design.
- graph-dba's rulings are applied as graph-dba stated them.
- What remains is five new minors (N1–N5) and two nits. None blocks S09/S10. N2 and N4 are
  one-line edits worth making before S09 encodes the spec.

CPG: considered, not relevant — the artifact is a spec document; its claims were checked by
executing hash and AST scripts and reading source.

#### Dispositions

- **F1 — fixed.** *Ran.*
  - The enclosing symbol of `tools.py:1480` is `McpToolClient.close` (lines 1470–1484).
  - The `call_soon_threadsafe` sites are at (1475, 39) and (1480, 43), so ordinal 1 is right.
  - GV-20 (`recv`) = `468fc6cb…` and GV-21 (`a0`) = `17e9de61…` both reproduce, and they differ.
  - Under v1.0's tuple, the same two records give one id: re-run, collision `True`.
- **F2(a) — fixed, and widened to every language.**
  - My derivation (`pyq2.py`) gives `embedding.py::Embedder.embed` (line 40) as a symbol.
  - The widening is sound in intent, since a call can resolve to a stub in every typed language.
  - It introduces N1.
- **F2(b) — fixed.**
  - *Ran:* under the evaluation-scope rule, `tests/test_fusion.py:241`'s decorator lambda is
    `test_fusion.py::<anon>#1`, the only module-level anonymous scope in that file. The textual
    reading would have given `test_find_fuzzy_candidates_….<anon>#1`, so the rule decides a real
    case.
  - `guards.py:147–149` still gives `<anon>#1`–`#3`.
  - Splitting Python and JS/TS defaults matches each runtime (Python evaluates defaults at
    definition time, JS at call time), so I accept it.
- **F2(c), F2(d) — fixed as recommended.** The TS side is **not checked by execution**: there is
  no `node` on this host, so there was no TS AST pass.
- **F3 — fixed.** The ancestor-plus-line locator is added in §3.5, and BF16 records the
  re-binding. Cross-checked against F2(b): consistent, with nit P2-n1 below.
- **F4 — fixed, every item.** Rule 8 against the accessor and constructor errata is consistent:
  - the C# example now has bodied accessors;
  - auto-accessors and Kotlin default accessors are excluded by name;
  - the Kotlin primary constructor is included by name.

  The new `.h` text test introduces N3, and the Kotlin constructor introduces N4.
- **F5 — fixed.** The AC-1 row is now `toPath: "ret"`, which agrees with graph note §6.3's
  `ret[.f…]`. Two cells are left stale (N2).
- **F6 — interim narrowing applied.** *Ran* (`bind.py`, implementing §5.17 (a)–(c)):
  - `http://localhost:8000/mcp` → binds;
  - the repo's real `.mcp.json` entry `mcpServers.falkor-chat-agent-team.url` =
    `http://localhost:8200/mcp` → binds;
  - `FALKORDB_PORT` `6379` and `FALKORDB_HOST` `host.docker.internal` → bind;
  - the Slack webhook → rejected, under either key name.

  Two residuals go to the `security-expert` advisory, not to a fix now:
  - the author's own 32-character lowercase token segment is still admitted (confirmed:
    `/abcdef0123456789xyzqwertyuiopas` binds);
  - a trailing-slash `/mcp/` and multi-segment base URLs (`/api/v1`) do not bind, which limits
    P1's URL binding.

  `DESIGN_URL` is redacted by the `sig` substring, as intended ("over-redaction is intended").
- **F7 — fixed.** *Ran:* all **21** golden vectors reproduce with my own `gv.py`, including
  GV-17's and GV-20's embedded ids (`3177d8d5…`, `e308e9b4…`, `79f8675f…`).
- **F8(a), (b), (c) — fixed as recommended** (§5.14/§8.1, §5.3, §8.4).
- **graph-dba's rulings — applied as stated.**
  - BF3: accepted, with a `teco` ledger line.
  - BF4: `arg(head, path)` with the `index`, `-1` and `Param.name` bindings, plus `["…"]` quoting.
  - BF6: one `Link` per consumer symbol, with `LIFTED_VIA {ord, siteId}` and best-over-sites
    status.
  - BF7: `role` narrowed to `{primary, example, doc}`.
  - BF13: S17 decides, and the bundle carries both fields.
  - BF14: `h(["name", lang, calleeName])`, with a §3.2 row and GV-19.
  - The §9.2 additions are all present: unbuilt `Unit`s, `componentId`, `flowAvailable`,
    `verdict-confirmed` `CALLS`, external `FROM_DEP` symbols, `paramMap` and `LIFTED_VIA.siteId`.
  - The `<dynamic>` and unbuilt-`targets[]` rules are in §9.1.
  - One stale cell remains (N2).

#### New findings

##### N1 (minor, P-later) — rule 8's C++ exception needs another file, against §4.0's source-only contract

Rule 8 says a C++ in-class member declaration "whose definition is out of line" is not a symbol,
"the definition is". Whether `void set(int) const;` in `motor.h` is a symbol then depends on
whether some `.cpp` defines it. A key writer reading `motor.h` alone cannot tell, which §4.0
forbids. The same question is open for `= default` and `= delete` members.

**Suggested improvement:** make the rule decidable from one file. A pure-virtual or
`= default`/`= delete` in-class declaration is a symbol, and any other bodyless in-class
declaration is not. An out-of-line definition is always its own symbol at its own path, so a
pure-virtual function with an out-of-line body gives two ids, at two paths.

##### N2 (minor) — three cells left stale by v1.1's own fixes

- **§5.10's `fromPath` cell** still lists `""`, but §5.1 now says "No `defPath` is `""`". S10's
  schema would encode one or the other.
- **§9.1's `data` row** says "between `Param`/`Ret`/`CallSite` points", but §5.1 now makes a
  `cond` a `data` use point (`toPath ""`). The DATA edge-load anchors need `Cond` among the
  endpoint labels. That is `graph-dba`'s S17 side, but the bundle spec should say it.
- **§9.1's `file` row** still says "`role` per graph note §6.4" (five values), against §5.2's
  narrowed three.

**Suggested improvement:** edit the three cells. They are text only, with no id effect.

##### N3 (minor, P-later) — the `.h` text test misclassifies in both directions

*By reading; not executed, since there is no C/C++ corpus here.*
- **C read as C++.** A C23 header using a prefixed attribute (`[[gnu::always_inline]]`) contains
  `::` and becomes `cpp`. Its C functions then carry signatures, which is exactly what BF11 exists
  to prevent.
- **C++ read as C.** A C++ header with only free inline functions, references or overloads, and
  none of the listed tokens, becomes `c`. Its overloads then collapse into the `#k` ordinals
  (`clamp`, `clamp#2`).
- **The `#if` exclusion.** "`#if`/`#ifdef __cplusplus` regions" leaves `#if defined(__cplusplus)`
  and the `#else` of `#ifndef __cplusplus` undefined.

The ids stay deterministic either way, so this is a P2 measurement item, and §10 already lists
the test as [P-later].

**Suggested improvement:** do one or both:
- exclude `[[…::…]]` attribute brackets from the `::` test;
- add `&`-reference parameters and `operator` to the C++ markers.

Define the excluded region as "any `#if…` group whose condition mentions `__cplusplus`, and that
group's `#else` arm".

##### N4 (minor, P-later) — the Kotlin primary constructor's parameters cannot reach its initializer code

Rule 8 makes the primary constructor a symbol whose span is the header's parameter list. Rule 4
sends `init {}` blocks and property initializers to the **type** symbol. §5.10 requires both
points of a `data` record to be in one symbol. So `class Orders(r: Repo) { init { r.load() } }`
has its `r` param on `Orders.<init>` and its use on `Orders`, and the flow between them is not
expressible. That is a silent miss in value-mode impact.

**Suggested improvement:** for Kotlin (and C# 12 primary constructors), assign `init {}` blocks
and property initializers to the primary constructor's symbol, not to the type. That reverses
F4's rule-4 errata for this one case. Use the class body as the span.

##### N5 (minor, P-later) — rule 8 leaves out Go interface method specs, though its own criterion covers them; a gap, not a choice

*Read across the docs (added at `teco`'s request).*
- **The criterion.** Rule 8 makes a symbol of "every member declaration a call can resolve to,
  with or without a body". A Go interface method spec (`type Store interface { Get(k string)
  string }`) is exactly the static target of a call through an interface value.
- **The list.** It names Python, Java, Kotlin, C#, TS and C++ cases, and not Go. Nothing excludes
  Go either: the "Not symbols" list does not mention it.
- **The rationale.** §13's own reason for widening F2(a) is "every typed language" ("a
  Python-only fix would leave the same dangling `targets[]` in Java, Kotlin, C#, TS and C++").
  That covers Go, which simply wasn't enumerated.
- **The qname.** Go's §4.2 bullets have no form for one: "Methods: `RecvType.NAME`" is about
  receivers, and an interface method spec has none.
- **The damage has started.** `docs/plans/connected-code-graph-p0-ml.md` (§2, `N(s)`) has
  already read the omission as an exclusion. So Go interface calls go to the non-symbol set,
  while Java/C#/TS interface calls go to `T(s)` as `decl`. The S26/S27 metrics would then treat
  one fact differently by language for a spec-text reason.
- **Not verified:** that `gosrc2cpg` resolves an interface call to the spec's `METHOD` node; no Go
  corpus is here. The gap holds either way: whether a declaration is a symbol is a source-side
  rule (§4.0), not a Joern-side one.

**Suggested improvement** (`architect`, before S09):
- Add "Go interface method specs" to rule 8's symbol list, and make the list explicitly
  exhaustive ("these, and nothing else").
- Add a Go §4.2 bullet: an interface method spec is `IfaceType.NAME`, under the interface type's
  symbol. An embedded interface adds no segment, and its methods stay under the embedded type's
  own `qname`.
- Have `data-scientist` re-align the ML note's `N(s)` example once the spec answers.

If `architect` instead rules Go specs out on purpose, rule 8 must say so in its "Not symbols"
list. Then rule 8's criterion needs a stated Go exception, because as written it includes them.

#### Nits

- **P2-n1 (§3.5).** Say that "nearest named non-duplicate ancestor" is taken by §4.1 rule 2's
  evaluation scope. `test_fusion.py:241`'s token line lies inside the decorated function's span,
  so a textual converter would choose `within` = the test function and find zero candidates.
  Also say that `nth` ranks by the **token's** column.
- **P2-n2 (§5.17).** Say whether a trailing `/` after the one permitted segment qualifies
  (`/mcp/`). It is a common form for MCP and HTTP base URLs.

#### Appendix S03-A2 — Pass 2 probes

All scripts are in
`/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/13504717-4242-422a-b7d8-a910a6cc9348/scratchpad/`:
- `gv.py`: GV-1–GV-21, with embedded ids recomputed from their own tuples. 21 rows `True`. The
  v1.0 `fieldAccess` tuple collides on `recv`/`a0` (`True`).
- `pyq2.py`: Python `qname` derivation under v1.1's §4.1 rules 2, 5, 7 and 8.
  - `tests/test_fusion.py:241` → `<anon>#1`, at module level;
  - `guards.py:147–149` → `<anon>#1`–`#3`;
  - `embedding.py:40` → `Embedder.embed`.
- `bind.py`: §5.17 (a)–(c) over 11 sample entries, with the results quoted under F6.
- An AST check of `tools.py`'s enclosing scopes and the `call_soon_threadsafe` positions, for F1.

### Pass 3

**Re-reviewed (2026-10-29):** `docs/plans/connected-code-graph-bundle-spec.md` v1.2, with its
§13.2. *Ran:*
- all **22** golden vectors with my own `gv.py`: 22 `True`, GV-22 = `afa8ae88…`;
- `git ls-files` for the default `vendoredDirs` list, which matches **0** files, confirming
  §5.0's "no path matching";
- a grep of `salesperson/src` for TS method signatures. It found none, but a grep is a bound,
  not a parse: there is no `node` here.

**Verdict: approve with suggestions.**
- N1–N5, P2-n1 and P2-n2 are fixed. N3 is partly fixed, with a stated, accepted limit.
- All three departures from my wording are improvements or justified.
- There are four new minors (P3-1 to P3-4), all [P-later], and two nits. None blocks S09/S10.
- The cheapest moment for P3-1 is now: it adds a field, and adding one after the freeze needs a
  minor version bump.

CPG: considered, not relevant — a spec document; checked by executing the hash script, the path
test and reading.

#### Dispositions

- **N1 — fixed.** The C/C++ row is decidable from the declaring file. A pure virtual with an
  out-of-line body gives two ids at two paths.
  - **Departure accepted:** `= delete` is not a symbol. No well-formed call resolves to it, so the
    symbol would be a node no edge reaches.
- **N2 — fixed.** In §5.10, `fromPath` no longer offers `""`. §9.1's `data` row names `Cond`, and
  its `file` row cites §5.2's three `role` values.
- **N3 — partly fixed; the limit is accepted.** `[[…]]` attribute contents are excluded, and every
  arm of any `__cplusplus` conditional is excluded.
  - **Departure accepted:** the `&` marker is not adopted. `&` is also C's address-of and
    bitwise-and operator, so it is not a token test.
  - The residual (a marker-free C++ header read as `c`) is stated, and §10 lists it for P2.
- **N4 — fixed, departure accepted.** The owned-regions design is better than my "class body as
  span", which would have put every method edit into the constructor's fingerprint. Its
  interactions are P3-1 and P3-2.
- **N5 — fixed.** `IfaceType.NAME` and GV-22 reproduce. There is no collision with
  `RecvType.NAME`, because Go forbids interface receivers.
  - Embedded interfaces add no segment, and a call through the embedding interface targets the
    embedded spec's own id. That is consistent with GV-22's note.
  - One undefined corner: nit P3-n1.
- **P2-n1 — fixed.** `within` is taken by evaluation scope, using the `test_fusion.py:241`
  example. The token and column are defined for every language.
- **P2-n2 — fixed.** `/mcp/` is admitted.

#### The architect's additions to rule 8 and rule 4 — checked

- **`native`, `extern`, `external` and Go bodyless functions as symbols: sound.** Each is the
  entity's only declaration, and each is decidable from one file.
- **C# primary constructors and Java record headers: sound** as constructors. Their gap is P3-2
  and P3-3.
- **TS `type`-alias method signatures (whole right-hand side only): sound and consistent** with
  the `interface` rule. `A & {m(): void}` and nested object types are not symbols, and the
  exhaustive table makes that the explicit answer.
- **The exhaustive table: an improvement.** It closes the class of gap N5 was.
- **Substitute-or-drop: sound and deterministic per unit.** A definition in another unit is
  dropped, not substituted, and that is stated. Nit P3-n2.
- **Vendored paths: sound, and consistent with the ML note.** No records for vendored code,
  `external` resolution, no `resolvedPath`, and no `targets[]` entry agree with §5.6 and §5.14 as
  amended. `CALLS_NAME` can no longer reach vendored code, because it has no `Symbol`. One
  ambiguity: P3-4.

#### New findings

##### P3-1 (minor, P-later) — owned regions exist only in the fingerprint, so the graph's file+line anchor and `lastChanged*` attribute initializer lines to the wrong symbol

*Traced.*
- **The bundle can't say which lines are owned.** §5.3 defines owned regions and counts them in
  `fingerprint` and `complexity`, but no field carries them. §9.1 carries only
  `lineStart`/`lineEnd`.
- **The store then picks the wrong symbol.** Graph note §3.4's file+line anchor
  (`s.lineStart <= $l AND s.lineEnd >= $l`) maps a line inside `init { repo.warm() }` to the
  **type** `Orders`, whose span covers the class body. But §3.5 and §4.1 rule 4 say the line
  belongs to `Orders.<init>`, whose span is only its parameter list. FR-12/RCA anchoring from a
  stack frame lands on the type, not on the constructor that ran.
- **`lastChanged*` is computed "over the span"**, so an edit to an `init {}` block changes
  `<init>`'s fingerprint but not its `lastChangedAt`.

**Fix:**
- Add `ownedRegions`, an array of `[lineStart, lineEnd]`, sorted, present exactly when the symbol
  has them.
- Compute `lastChanged*` over the span plus the owned regions.
- Hand S17 the anchor rule: the innermost symbol whose span **or** owned region contains the line.
  This is an additive field, so it is cheap now and a minor bump after S15.

##### P3-2 (minor, P-later) — C# static initializers would be owned by the instance primary constructor

Rule 4's exception assigns "the property and field initializers" of a C# type with a primary
constructor to `<init>`. A `static` field or property initializer runs in the **static**
constructor, not the primary one. It would get the wrong fingerprint, and flows in the wrong
symbol. Kotlin is unaffected, because companion-object initializers live in the companion's own
type scope.

**Fix:** "instance field and property initializers. Static ones belong to the static constructor
`<init>` with signature `static()` when one is declared, else to the type".

##### P3-3 (minor, P-later) — a call to a Java record's implicit accessor reaches nothing

`r.x()` resolves to the record component's compiler-synthesized accessor. That accessor is not a
symbol, and §4.1 rule 8's substitute-or-drop rule drops it. Syntactically it is a call, so no
`fieldAccess` is emitted either. Name-match then finds no symbol named `x`. So renaming or
retyping component `x` reaches **no** caller, which is a silent miss against FR-8's never-miss
rule. Kotlin and C# property access does not have this problem, because it is member access and
`fieldAccess` covers it.

**Fix:** pick one:
- substitute a call to an implicit record accessor by a `fieldAccess` read of
  `(record type, component)`;
- or make each record component an accessor symbol, `R.x.<get>` with signature `()`, spanning
  its header entry.

##### P3-4 (minor) — the vendored test doesn't say which path it reads, or how per-repo and per-unit relate

- §5.0 says "a directory segment" of "a path", and paths are repo-relative (§8.2).
- `vendoredDirs` replacement is per **repo** (manifest), but `header.vendoredDirs` is "applied to
  this unit".
- The result: a first-party unit rooted at `third_party/ourlib/` yields an empty bundle unless the
  whole repo's list is replaced.

**Fix:**
- Say the test reads the repo-relative path's segments **below the unit's `rootPath`**, so a
  unit's own root never makes it vendored.
- Or say the manifest list may be given per unit.

The proving repo has no such path, so there is no P0 impact.

#### Nits

- **P3-n1 (§4.2 Go).** Go 1.14+ allows two embedded interfaces with the same method
  (`interface{ A; B }`, both declaring `Close()`). Say whether a call through the embedder
  targets every embedded spec of that name (recommended, as the never-miss reading) or the first
  in source order.
- **P3-n2 (§4.1 rule 8).** After a drop empties `targets`, say what `resolution` becomes. I
  recommend it stays `resolved`, so S25/S27's resolved-precision denominators don't silently move.

## S03 (P0-03a) — graph-dba check

**Complete (2026-10-29; paused once mid-check at the coordinator's request, then resumed).** Reviewer
`graph-dba`, against `docs/plans/connected-code-graph-bundle-spec.md` v1.0, §3.1–§3.2, §5, §9 and
§12, and the graph note `docs/plans/connected-code-graph-graph.md` §3.2, §3.4 and §6. It is static,
plus `claude/graph-dba/falkordb-quirks.md` (4,096-byte and edge-whitespace `UNIQUE` crashes,
re-checked). **Verdict: approve with suggestions.** No finding blocks S09/S10 on the graph side. The
analyst's F1 is real and needs a tuple change before the freeze, and it has a graph-side half
(below). One
ruling (BF6) must be taken **now**, not at S17, because S17 runs after S15 freezes the tuples.

**Timing rule I applied.** S17 (graph model v1) runs **after** S15 freezes bundle v1. So "waits
for S17" is only acceptable where the decision is graph-side only, meaning the bundle already
carries the fact. A change to a tuple or to an assembly id that gold or verdicts cite has to be
made here; after S15 it is a major bump (§8.6).

### Rulings on the findings that touch the graph model

- **BF3 (`use`/`def`, not `from`/`to`): accept now.** The spec is right. In note §6.3 the edge is
  stored use → def. `fromPath` sits on the target (def) and `toPath` on the source (use). They
  name the value-flow direction, so `from`/`to` node fields would read backwards. The edge
  property names are unchanged, so there is no graph delta.
- **BF4 (`recv`, `kw.<name>`, `prefix=`): accept now. S17 records the graph side.** The edge
  properties are unconstrained strings, so there is no crash or index risk. The graph-side
  consequence for S17 and the BFS is that §6.1's `arg(i, path)` state becomes `arg(head, path)`,
  bound to the callee's `Param` like this:
  - `a<i>` by `index`, after the receiver;
  - `recv` ↔ `index -1`;
  - `kw.<n>` by `Param.name`, including `**kwargs`.

  So `Param.name` becomes a match key. It is already carried. Suggestion (S03): `FIELD` is "as
  written", so a dict key containing `.` (or an empty key) makes a path ambiguous. Reuse §5.17's
  `["…"]` quoting for such segments. Bundle grammar is cheap to fix now and a major bump later.
- **BF6 (`linkId` anchors): rule now, no wait for S17.**
  - (a) An anchor is the enclosing `symbolId`, or the `entryId` for a `ConfigEntry`. **Agreed.**
  - (b) **One `Link` per (kind, keyNorm, provider anchor, consumer anchor), i.e. per consumer
    symbol — the spec's recommendation, adopted.** Verdict stability outweighs per-site
    resolution: a person judges "symbol X calls endpoint Y".
    - This is safe structurally. Several consumer `CallSite`s `-[:VIA]->` one `Link`, and
      per-site status/reason already live on each site's `CONSUMES {status, reason}`.
    - `LIFTED_VIA` must then be keyed per site: `(:Link)-[:LIFTED_VIA {ord, siteId}]->(…)`, where
      `siteId` is the consumer `callSiteId`. Evidence rendering filters by the site on the path.
      Otherwise two sites' chains interleave under one `ord`. S17 writes this property.
    - `Link.status` becomes the best over its sites. S17 states that.
- **BF7 (`File.kind` vs `File.role`): carry both. S17 decides the graph property.** The overlap
  comes from my note (§6.4 gave `role` the values `config`/`script`). Read them as orthogonal:
  - `kind` is the extractor family;
  - `role` is the purpose, and only `example` and `doc` are load-bearing (`example-config`, FR-5).

  Suggestion (S03, before the freeze): narrow `role` to `{primary, example, doc}` so the two
  fields can't disagree. If that isn't done, it can't be undone later without a major bump, but
  it is harmless either way. Graph side: both go on `File`, unindexed.
- **BF13 (bundle facts with no graph property): waits for S17. That is correct, because all of
  them stay in the bundle.** My leanings for S17, all O(nodes) and negligible against V11:
  - carry `Symbol.complexity`, `Symbol.signature`, `Symbol.lang`, `File.lang` and
    `File.contentHash`;
  - put `scope` on the `DEPENDS_ON` edge (it is per declaration, and dev vs runtime matters for
    severity);
  - keep the launch-fact binding inputs and the header provenance out, as assembly inputs.

  **`DATA.status`/`reason`: measured; my lean is to store them only when they differ from the
  default.** Probe on `ccg-falkordb` (v4.18.11, module `41811`; scratch keys
  `ccg_s03_probe_c`/`_d`): 100k nodes and 300k `DATA` edges with `{fromPath, toPath}` gave
  `amortized_edge_attributes_by_type_sz_mb` `DATA: 11` and `total_graph_sz_mb` 38. Adding
  `status:'confirmed', reason:'intra-flow'` gave `DATA: 27` and total 54: about **+56 bytes per
  edge, +42 % on this graph**. In the v2 estimate (note §6.4) `DATA` is roughly 1.4k edges per
  file, so this would be about +0.7–0.8 GB per 10k-file snapshot, ×4 at peak. That is a real cost
  for a constant. My lean for S17: an absent `status`/`reason` on `DATA` means
  `confirmed`/`intra-flow`, and only `possible` edges (`field-truncated`, `shell-dynamic`) carry
  them. The operator reads `coalesce(e.status, 'confirmed')`, which is not an index anchor, so it
  costs nothing. Note §3.3's rule then holds by default value rather than by storage. The bundle
  is unaffected, since its records always carry both fields.
- **BF14 (`Name.nameKey`): hash it. Rule now; S17 records it.**
  - Under §9.1, `nameKey = "<lang>:<calleeName>"` is the only **unhashed** value under a `UNIQUE`
    constraint in the whole model. `calleeName` is a shell command word or a prompt's tool name
    "as written", which is unbounded. A value over 4,096 bytes SIGSEGVs the instance at the
    post-load constraint check, and the whole box goes with it, not just the build.
  - Edge whitespace is covered, because identity fields are stripped (§8.1). Length is not.
  - **Ruling:** `nameKey = h(["name", lang, calleeName])`. Keep `lang` and `name` readable and
    unconstrained.
  - Suggestion (S03): add the `nameKey` row to §3.2's tuple table as an assembly id, with a golden
    vector in §3.6, so S09 encodes it. It costs one row now. Otherwise it lands as an S17 delta to
    `ids.py`.

### Id design against FalkorDB's constraints (question 3)

- **32-hex truncated SHA-256 (128 bits):** fine. No collision concern at any plausible scale. It
  has no whitespace and is 32 bytes, far below 4,096.
- **Constrained properties in note §3.4 and §6.4:** `symbolId`, `callSiteId`, `cpId`, `linkId`,
  `unitId`, `fileId`, `Param`/`Ret`/`Cond.pointId`, `findingId` and `sightingId` are all hashes
  in this spec. **The one exception is `nameKey`** (BF14, above). No relationship-scoped
  constraint is in the model, which is good, because the whitespace crash is unchecked there
  (quirks file).
- **Range-indexed readable properties** (`File.path`, `Symbol.qname`, `Symbol.nameNorm`,
  `ContractPoint.keyNorm`): there is no crash risk without a constraint (the quirks file has
  verified RANGE-only to 1 MB). The silent-miss risk on unstripped values is covered: §8.1 strips
  identity fields and §6 strips `keyNorm`. Lookup code must strip the same way (C `isspace` ⊂
  Python `str.isspace`).

### §9.1 / §9.2 mapping (question 1) — checked so far

§9.1 matches note §3.2 as amended by §6.3/§6.4 for every record type. These gaps in §9.2's "not
from a bundle" list should be added, so the mapping is checkably complete:
- `Unit` nodes for never-built or failed units (`buildStatus` `stale-kept`/`not-mapped`,
  `failedAt`, `failReason`): from the manifest;
- `Unit.componentId` (FR-11 grouping);
- `CALLS {reason: 'verdict-confirmed'}`, which verdicts create alongside `DISMISSED`;
- the external `Symbol` nodes that `FROM_DEP` hangs from (P5);
- `Link.paramMap` (§6.1 item 5).

New against the note, for S17 to adopt: `Unit.flowAvailable`.

Suggestions:
- Say what the loader does with a `callsite.targets[]` `symbolId` whose unit has no node (stale
  or not mapped): skip and count it, or route it as `TO_UNIT`.
- `calleeName = <dynamic>` would make one `Name` node every dynamic site points at, with nothing
  to resolve. Suggest emitting no `CALLS_NAME` for it.

### Cross-check from S07, for S17 (not a spec defect)

The note's §3.5 step 2 says "writes ignore `TIMEOUT`". S07 re-verified on 2026-10-29 that a
**nonzero `TIMEOUT_MAX` also kills writes**, with nothing committed (quirks file, TIMEOUT
entry). FR-25 sets exactly that cap, so P0/P1 loads must pass an explicit `timeout` up to the cap,
or the cap must be sized above the longest batch.

### F1 (analyst) — the `fieldAccess` id omits `path`: my view

**Agree that it is real; major.** It is a slice blocker only if the slice emits `fieldAccess`
(§5.13 is `[P-slice]` conditionally). There is **no FalkorDB uniqueness risk**: `fieldAccess`
becomes `READS`/`WRITES` edges, the model has no relationship-scoped constraint, and the
`linkId` relationship index is non-unique. The defect is in the bundle (§8.3's fail-on-duplicate),
plus one graph-side gap:
- **Graph side.** §9.1 carries `{status, reason, linkId}` on the field edge but **not `path`**.
  The BFS needs the path. A field read reached from a `WRITES` in `value` state continues as
  `arg(head, path)` into the call (note §6.1 item 3), and `recv` vs `a0` decides which callee
  `Param` it enters. Without `path` the walk must assume every argument, which is a false-impact
  widening.
- **Ruling:** adopt the analyst's tuple
  `["fieldAccess", pointId, path, mode, receiverType, field]` (S03, before the freeze), **and**
  carry `path` on the `READS`/`WRITES` field edge (§9.1, recorded by S17).
  - Two reads of one field at different paths on one point then become two **parallel** edges
    between the same pair. That is correct, and FalkorDB stores them as distinct edges.
  - Operators must count them by binding the relationship: `count(*)` under-counts parallel
    edges (quirks file).
  - Both edges carry the same `linkId`, so one verdict covers both. That is the intended
    semantics.

### Remaining checks (completed on resume)

- **Edge-load anchors for `DATA`/`CONTROLLED_BY` endpoints: OK.** `use`/`def`/`point` are
  in-bundle references (§5.0), so the loader derives each endpoint's label from the referenced
  record's type. It then groups edges by (source label, target label) and `MATCH`es on the
  indexed property: `CallSite.callSiteId`, or `pointId` for `Param`/`Ret`/`Cond`. That is up to
  4×4 label pairs, each an index seek, and never a label-less match (which would be a full scan).
  The type tag (BF1) keeps the two id spaces disjoint.
- **`dependency` → `Dependency` (§5.15): OK, with one suggestion.** `depId` is repository-neutral,
  so many units share one node, and the loader must `MERGE` it. That is safe on the single loader
  connection (note §3.5). Suggestion for S17: add `UNIQUE NODE Dependency PROPERTIES 1 depId` to
  the post-load gate. It is a hash, so it is crash-safe, and it turns a dedup bug into a `FAILED`
  constraint instead of a silent double count.
- **`launchFact`/`configEntry` sharing `:ConfigEntry` (§5.17–§5.18): OK.** The distinct tags
  separate the id spaces. A `UNIQUE` `entryId` gate is optional, on the same reasoning as
  `depId`.
- **§9.1 property lists against §5's field tables:** consistent, apart from the gaps already
  listed in the §9.1/§9.2 subsection and the `fieldAccess` `path` above.

### Incident during this check

**`ccg-falkordb` stopped answering `PING` for several minutes during my first RAM-probe attempt
(scratch key `ccg_s03_probe_a`, a 100k-edge single-query write).**
- The container showed ~105 % CPU, the client hit a socket timeout, and the instance recovered
  on its own.
- **The cause is not established.** My guess that the index was missed was wrong:
  `GRAPH.EXPLAIN` of that write shows `Node By Index Scan` on both endpoints, and a 1,000-row
  batch of the same shape runs in 19 ms.
- I saw the same stall pattern once in S07's throwaway container. An oversized write (or the
  read before it) left the server unresponsive for minutes, even to `HELLO`.
- Worth a targeted probe in S23. Until then, P0 loads should stay in batches of ≤ 20k rows (the
  re-run used 20k-row batches, each well under 5 s).

I deleted the scratch keys `ccg_s03_probe_a`, `_c` and `_d` with `GRAPH.DELETE`; `ccg-falkordb`
is back to `DBSIZE 0`. Observation for S23 (one sighting, not yet a verified quirk): a `--scan`
right after `GRAPH.DELETE` still listed `telemetry{ccg_s03_probe_c}` and `telemetry{…_d}`.
Moments later, `TYPE` returned `none` and `DEL` returned `0`. So the telemetry stream seems to go
away asynchronously after the graph is deleted, and a retention or redaction check that runs
straight after `GRAPH.DELETE` may still see query text and parameter values.

## S02 (P0-02)

**Reviewed (2026-10-29):** `cpg/infra/toolchain/Dockerfile`, `build.sh` and `README.md` (all
untracked), plus the built image `ccg-toolchain:2c5e60084f67` (`sha256:fb66581bc1e3…`) and
devops' build/parse logs. The baseline is `docs/plans/connected-code-graph-p0.md`: the §3 S02 row,
§4.1 (S02, S08 tests, S24), and §7. For fit I read `docs/plans/connected-code-graph-egress-spec.md`
as it stands mid-revision to v1.1: §7 interface, §7.1, §7.1.2, §7.3.4, A4/A6, R3. `S02-STATUS.md` is
a hand-off note and was not reviewed as a deliverable. Every claim below was **run** unless it says
*inferred*. Only `docker run --rm --network none …` was used, plus registry and checksum lookups
from the host. Nothing was built, pulled or removed.

**Verdict: needs changes.** All three of S02's done-conditions hold, and five of the six
download pins check out against upstream. The apt snapshot pin, though, is not real (B1): the
image's packages came partly from the live archive, so the content-hash tag is not the identity
the Dockerfile, README and `ccg.apt.snapshot` label say it is. The fix is a few lines. Separately,
there is a cross-step defect (M1): the S04 launch shape, taken literally, breaks every frontend.
It is not S02's to fix, but this gate is the first place it was run.

CPG: considered, not relevant — the deliverable is a Dockerfile and a Bash script, and no loaded `cpg_*` graph covers `cpg/infra/`.

**Done-conditions (§3 S02 row):**
- **Image builds at `ccg-toolchain:<content-hash>`: met.** `build.sh --print-tag` gives
  `2c5e60084f67`, which equals `sha256(Dockerfile)[:12]`. Appending one byte to a copied
  Dockerfile moves the tag (`9114f0430766`).
- **Every §7 image-column tool prints its version: met.** I re-ran `ccg-versions` under
  `--network none`: 0 "not found" or error lines, and the output is byte-identical to devops'
  `s02-versions.txt`. That covers Joern and all 8 FR-1 frontend jars at 4.0.579, Temurin 21.0.12,
  Maven 3.9.16, Gradle 9.8.2, Go 1.27.3, .NET SDK 10.0.403, kotlinc 2.4.21, CMake 3.28.3,
  bear 3.1.3, gcc/g++ 13.3.0, make 4.3 and git 2.43.0. Note that exit 0 by itself proves nothing
  (M2).
- **One-file Python `joern-parse` under `--network none`: met.** Exit 0, 90 KB `cpg.bin`, about
  2 s. Without a hostname, the log4j `UnknownHostException` appears twice. With
  `--hostname localhost` it appears 0 times. Devops' claim reproduces exactly.

### Findings

**B1 — blocker. The apt snapshot pin does not hold: packages resolve from the live archive too.**
- **Evidence.** In `s02-build-attempt1.log`, the same `apt-get update --snapshot` step fetches
  `InRelease` and `Packages` from both `http://archive.ubuntu.com` / `security.ubuntu.com` and
  `https://snapshot.ubuntu.com/…/20261020T000000Z`. Packages are then downloaded from
  `archive.ubuntu.com`: `git 1:2.43.0-1ubuntu7.4`, `cmake`, `bear`, `curl`, and more.
- **Version check.** Of the 70 packages that `dpkg.log` shows the toolchain layer installing, 4
  are at versions that are absent from the snapshot's `noble{,-updates,-security}/{main,universe}`
  indexes, and newer than the snapshot's own: `libcares2 1.27.0-1.0ubuntu1.1`,
  `libprotobuf32t64`/`libprotoc32t64 3.21.12-8.2ubuntu0.4`, `xz-utils 5.6.1+really5.4.5-1ubuntu0.4`
  (Appendix A).
- **Cause.** apt consults the union of both sources and takes the highest version. That much is
  confirmed by the above. *Inferred:* `--snapshot` adds the snapshot sources here rather than
  replacing `ubuntu.sources`' URIs.
- **Why it matters.** A cold rebuild of the same Dockerfile (`build.sh --no-cache`, or another
  host) produces different packages under the same tag. S08's audit records `image`/`imageId` as
  the toolchain identity, and the README, the Dockerfile's PINNING header and the
  `ccg.apt.snapshot` label all assert a pin that is not there.
- **Fix (devops).** Before each `apt-get update` in **every** stage (`fetch`, `toolchain`, and any
  S24 apt layer), point the sources at the snapshot and drop `--snapshot`:
  `sed -Ei "s#^URIs: http://(archive|security)\.ubuntu\.com/ubuntu/?#URIs: https://snapshot.ubuntu.com/ubuntu/${SNAPSHOT}/#" /etc/apt/sources.list.d/ubuntu.sources`.
  I ran that sed in the image, and both `URIs:` lines come out correct. The sed is not
  build-verified. Then, in the same `RUN`, fail closed if any list in `/var/lib/apt/lists` is not
  from `snapshot.ubuntu.com_`.
- **Acceptance.** Use the Appendix A probe, which is independent of the fix and **red on the
  current image** (4 misses). It must report 0 packages missing from the snapshot for every
  `dpkg.log` install dated at build time. The build log must also show no `Get:` from any host
  other than snapshot.ubuntu.com.

**M1 — major (cross-step; owner `architect` for S04 v1.1, with `devops` for S08). S04's
`/tmp` mount (`--tmpfs …:rw,noexec,nosuid`, §7.1) makes every frontend fail. R3's fallback does
not repair it.**
- **The literal launch fails.** I ran the full v1.1 launch (Appendix B): `env -i` entrypoint,
  `--read-only`, `--cap-drop ALL`, the §7.3.4 pass-list. **All 7 FR-1 languages exit 1.**
  zstd-jni extracts its `.so` into `java.io.tmpdir` and gets `failed to map segment`. jssrc's
  `astgen-linux` is a gzexe wrapper that unpacks into `$TMPDIR` and gets `Permission denied`.
- **R3's fallback also fails.** I added a separate exec `tmpfs` at `/tmp/native` and passed
  `-J-Djava.io.tmpdir=/tmp/native` to `joern-parse`, as R3 prescribes. Python, Go and C# still
  fail on zstd: the flag configures only the `joern-parse` JVM, not the frontend subprocesses it
  spawns. jssrc also still fails, because astgen uses `$TMPDIR`.
- **What works.** Either of these passes:
  - `/tmp` mounted `exec`: all 7 exit 0, 0 `UnknownHost`, 2–8 s each.
  - `TMPDIR=/tmp/native` plus `JAVA_OPTS="-Xmx2g -Djava.io.tmpdir=/tmp/native"`: Python, JS and
    Java exit 0. This one breaks A6(c)'s heap-only pattern.
- **Trap.** Docker's `--tmpfs` defaults to `noexec`. Deleting the word from the spec is not
  enough; the mount must say `exec` explicitly. Verified: `--tmpfs /tmp:rw,nosuid` mounts
  `noexec`.
- **Fix.** Revise §7.1, R3 and A4(a)'s `/tmp` leg to one of the working shapes. That choice is
  the architect's, because it trades A4's noexec mechanism against A6(c)'s pattern.
  - *Image-side alternative, unverified:* pre-gunzip the astgen wrappers at build time and
    pre-extract the zstd-jni library, so nothing executes from a tmpfs.
  - S02's README "Run" section should state the requirement once it is decided.

**M2 — major. `ccg-versions` exits 0 even when every tool is missing, so it cannot gate S24.**
- **Evidence.** I ran it with `PATH=/usr/bin:/bin`. It prints `java: command not found`,
  `mvn: command not found`, `go: command not found`, … and exits **0**. `set -uo pipefail` has no
  `-e`, and `v()`'s status is discarded. The `jar_ver` lines print an empty version silently.
- **Why it matters.**
  - S24's done-condition is "every truth tool S05 names runs in the image", and S24 extends this
    script. A check that cannot fail would pass a broken layer.
  - The README's "exits 0 and prints the versions above" reads as evidence, and it is not.
- **Fix.** Have `v()` and the jar loop `return 1` on a non-zero tool exit or an empty version.
  Accumulate a `fail` counter and `exit "$fail"` at the end.
- **Acceptance.** The `PATH=/usr/bin:/bin` run above must exit non-zero. It is red today. The
  default run must still exit 0.

**m1 — minor. `GOTOOLCHAIN=local` exists only as image `ENV`, so `env -i` reverts it to `auto`.**
- **Evidence.** Under `env -i`, `go env GOTOOLCHAIN` prints `auto`, because `$GOROOT/go.env`
  ships `GOTOOLCHAIN=auto`. With the image env, it prints `local`.
- **Does it matter today?** Only through the spec. v1.1's §7.3.4 re-adds `GOTOOLCHAIN=local`
  (and `GOROOT`, `DOTNET_*`), so the S08 path is covered. The risk is a future pass-list edit or a
  direct `docker run --entrypoint`. Offline, a toolchain fetch would simply fail. In controlled
  mode it would go out through the broker and look like a dependency download.
- **Fix.** Defense in depth, one line in the fetch stage:
  `sed -i 's/^GOTOOLCHAIN=auto$/GOTOOLCHAIN=local/' /opt/go/go.env`. Keep the `ENV`.
- **What else `env -i` drops.** `JOERN_HOME`, `MAVEN_HOME`/`GRADLE_HOME` (offline),
  `JAVA_VERSION`, `LANGUAGE`. All 7 parses succeeded without them (M1's working shape), which
  confirms the spec's inference that the launchers do not need `JOERN_HOME`.

**m2 — minor. The base's `ENTRYPOINT ["/__cacert_entrypoint.sh"]` is inherited.**
- **Evidence.** `docker image inspect` shows it. The README's run examples go through it, and S08
  does not (`--entrypoint /usr/bin/env`).
- **Why it matters.** The script is inert unless `USE_SYSTEM_CA_CERTS` is set. If it is set, the
  script rewrites the JVM truststore from any `/certificates/*.crt` and exports
  `JAVA_TOOL_OPTIONS`, which is a §7.3.4 forbidden name.
- **Fix.** Add `ENTRYPOINT []` in the tail, so the image's default behaviour equals what S08
  runs. Mention it in the README.

**m3 — minor. S24 cannot extend `ccg-versions` without editing an S02 layer.** The script is an
S02 heredoc, and README "Extending" says "S24 also appends its tools to `ccg-versions`". That
contradicts the Dockerfile header ("S24 extends it without touching the S02 layers").
- **Fix.** Have `ccg-versions` also run each `/usr/local/lib/ccg-versions.d/*.sh`, so S24 adds a
  file in its own `[S24-LAYER]`. This pairs with M2: the drop-ins inherit the fail counter.

**m4 — minor. README accuracy.**
- **(a) Pinning.** "apt packages via Ubuntu snapshot" is false until B1 is fixed.
- **(b) Size.** "About 4.5 GiB": 4,718,374,834 B is **4.39 GiB** (4.72 GB). The 2.6 GiB Joern and
  1.2 GiB non-FR-1 figures are correct (`du`).
- **(c) Network at run time.** "A build needs network; a run never does": fetch-on runs (S25,
  §7) do need it, through controlled egress.
- **(d) Network-related ENV.** "The only network-related `ENV` is `GOTOOLCHAIN=local`":
  `DOTNET_CLI_TELEMETRY_OPTOUT` is egress-relevant too, as §7.3.4 itself says.
- **(e) Output size.** "99 KB `cpg.bin`": I measured 90,247 B.

**nit — inherited label.** `org.opencontainers.image.version=24.04` comes from the Ubuntu base
and reads as this image's version. Override it, for example with the tag hash or `4.0.579`.

### What's solid

- **Every download pin is real.** I checked each one against upstream:
  - Joern: GitHub asset digest `42ef562e…`, which also equals `sha256sum ~/joern/joern-cli.zip`.
    The host zip's sha512 matches Joern's own `joern-cli.zip.sha512`.
  - Go: `go.dev/dl` JSON.
  - Maven: Maven Central `.sha512`, confirmed by the downloads.apache.org copy.
  - Gradle: the `.sha256` sidecar.
  - Kotlin: the GitHub asset digest and the `.sha256` sidecar.
  - .NET: the 10.0 `releases.json` hash for 10.0.403, which is `latest-sdk`, LTS.
- **The base digest is real.** `4d271cd5…` is the current index digest of
  `eclipse-temurin:21.0.12_8-jdk-noble` (`imagetools inspect`).
- **The build design is sound.**
  - The build context is empty, so the Dockerfile is provably the only input.
  - The multi-stage layout keeps the 2.1 GB zip out of every layer.
  - `fetch-verify` fails closed on a checksum mismatch.
- **Hygiene: clean.**
  - The final user is `ccg` (1000:1000, no supplementary groups).
  - No `.m2`, `.gradle`, `.nuget`, `.dotnet`, `.cache` or `pkg/mod` anywhere (`find / -xdev`).
  - `/home/ccg` holds only the skeleton dotfiles, `/work` is empty, and the apt lists are removed.
  - No credentials in `ENV`, labels or history, and a key/token grep over
    `/etc /home /work /usr/local` finds nothing.
- **`build.sh` works as described.**
  - It is idempotent: an existing tag gives a 0.27 s no-op, tag on stdout and size on stderr.
  - An unknown option exits 2, and `--help` exits 0.
  - `--print-tag` is cwd-independent and honours `CCG_TOOLCHAIN_REPO`.
  - `bash -n` is clean.
- **The S08 fit is good, apart from M1.** With `/tmp` exec, the full v1.1 shape parses all 7 FR-1
  languages in 2–8 s each: `env -i` pass-list, `--read-only`, `--cap-drop ALL`,
  `no-new-privileges`, `--user 1000:1000`, a read-only `/work` cwd, and `--hostname localhost`.
  That is well inside test 3's 120 s. The image's `PATH` is usable verbatim, as §7.3.4 now
  specifies.
- **The S24 hand-off points are placed correctly.** `[S24-STAGE]` sits before the toolchain
  stage, and `[S24-LAYER]` sits before the non-root tail.

### Open questions

1. **M1's shape is a spec decision.** Which shape should §7.1 take: an exec `/tmp`, with A4(a)
   on `/tmp` resting on interpreter absence only; a dedicated exec tmpdir through `TMPDIR` plus
   `JAVA_OPTS -Djava.io.tmpdir`, which widens A6(c)'s pattern; or image-side pre-extraction, which
   is unverified? I recommend the exec `/tmp`, the smallest change. This is for the architect,
   during the v1.1 revision now in progress.
2. **Should the non-FR-1 frontends be pruned?** They take about 1.2 GiB: ghidra, abap, swift,
   rust, php and ruby. §7.3.1 refuses them anyway. This is a size and attack-surface call, not a
   defect, and it changes the tag.

### Appendix (S02)

**A. Snapshot-membership probe (B1).** All files are in the session scratchpad, `s02rev/`.
`docker run --rm --network none ccg-toolchain:2c5e60084f67 bash -c 'grep -h " install " /var/log/dpkg.log'`.
Keep the rows dated at build time, as `<pkg> <version>`. Fetch
`https://snapshot.ubuntu.com/ubuntu/20261020T000000Z/dists/{noble,noble-updates,noble-security}/{main,universe}/binary-amd64/Packages.gz`.
Report every (pkg, version) that is absent from the union. Today: `checked 70 not-in-snapshot 4`
(`libcares2`, `libprotobuf32t64`, `libprotoc32t64`, `xz-utils`; each is newer than the snapshot's
highest). Image-wide, 17 of 286 differ, but most of those come from the base image's own
September apt runs. Only S02's layers are in scope here.

**B. S04 v1.1 launch used for M1.**
`docker run --rm --network none --hostname localhost --read-only --cap-drop ALL --security-opt no-new-privileges --user 1000:1000 --pids-limit 4096 --memory 3g --tmpfs /tmp:rw,noexec,nosuid --tmpfs /home/ccg:rw,noexec,nosuid -v <src>:/src:ro -v <out>:/out:rw --entrypoint /usr/bin/env ccg-toolchain:2c5e60084f67 -i PATH=<image PATH> JAVA_HOME=/opt/java/openjdk HOME=/home/ccg TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8 JAVA_OPTS=-Xmx2g GOTOOLCHAIN=local GOROOT=/opt/go DOTNET_ROOT=/opt/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 joern-parse /src --language <pythonsrc|javasrc|jssrc|golang|csharpsrc|newc|kotlin> -o /out/cpg.bin`.
The 7 samples' outcomes:
- As written: all 7 exit 1.
- `/tmp:rw,exec,nosuid`: all 7 exit 0.
- R3 fallback (extra `--tmpfs /tmp/native:rw,exec,nosuid`, plus `joern-parse -J-Djava.io.tmpdir=/tmp/native`): python, jssrc, csharpsrc and golang still exit 1.
- `TMPDIR=/tmp/native` and `JAVA_OPTS="-Xmx2g -Djava.io.tmpdir=/tmp/native"`: python, jssrc and javasrc exit 0.

The logs are `s02rev/s04-*.log`, `s04e-*.log`, `r3-*.log` and `r3b-*.log`.

### Pass 2

**Re-checked (2026-10-29):** `ccg-toolchain:d3184effb566`, the revised `Dockerfile` and
`README.md`. Everything was run, using `docker run --rm --network none` plus two throwaway mutant
builds with no tag (neither produced an image).

**Verdict: approve with suggestions.** B1 and M2 are fixed, with evidence, and so are m1–m4 and
the nit. M1 is still open, but it is the S04 spec's to settle, not S02's. One new minor finding:
p2-m1.

- **B1: fixed.**
  - Probe: the Appendix A probe on the new image reports `checked 70 not-in-snapshot 0`. The four
    former misses now resolve to the snapshot's versions (`libcares2 1.27.0-1.0ubuntu1`,
    `libprotobuf32t64 …ubuntu0.3`, `xz-utils …ubuntu0.3`).
  - Sources: both `URIs:` lines in the image point at the snapshot, and `/var/lib/apt/lists` is
    empty.
  - Build log: all 116 `Get:` lines in `s02-fix-build.log` come from `snapshot.ubuntu.com`.
  - The guards fail closed, checked by two mutant builds:
    - **Mutant A** adds a live `.sources` file and then runs `apt-snapshot-install hello`. The
      build fails with exit 1 and `apt lists from a non-snapshot host: archive.ubuntu.com_…`.
    - **Mutant B** writes a live `deb` line to `/etc/apt/sources.list` before `apt-base`. The
      build fails with exit 1 and `/etc/apt/sources.list has live entries`.
  - Limit: the guard covers only stages built `FROM apt-base` that install through
    `apt-snapshot-install`. A stage `FROM ${BASE}`, or a bare `apt-get install`, would bypass it.
    The README and the `[S24-*]` comments state the rule, so S24's review must check it.
- **M2: fixed.** Measured exit codes:
  - Default run: 0, ending `all checks passed`.
  - `PATH=/usr/bin:/bin`: 1, with `6 check(s) FAILED`.
  - A failing drop-in (a missing tool plus an absent jar): 1, with 3 failed.
  - A drop-in with a syntax error: 1.
  - A passing drop-in: 0, and its version is printed.
- **m1: fixed.** Under `env -i` with only `PATH`/`HOME` set, `go env GOTOOLCHAIN` prints `local`.
- **m2: fixed.** `Config.Entrypoint` is `null` and `CMD` is `["ccg-versions"]`.
- **m3: fixed.** `/usr/local/lib/ccg-versions.d/` exists and is empty. The drop-ins share the
  fail count, and the Dockerfile and README describe S24's route without any edit to an S02 layer.
- **m4: fixed.** The README's pin text, size (4,718,150,014 B, which matches `inspect`),
  network-at-run wording, egress `ENV` list and `cpg.bin` size are now accurate. It also records
  the `/tmp` exec requirement as pending M1.
- **Nit: fixed.** `org.opencontainers.image.version` is now `joern-4.0.579`.
- **M1: still open, and still owned by S04.** That is unchanged. The new image parses the one-file
  Python sample with exit 0 and no `UnknownHostException` (90,241 B).

**p2-m1 — minor (new). A drop-in that calls `exit` ends `ccg-versions` with the drop-in's status
and skips the summary.**
- **Evidence.** A drop-in with two failing checks, followed by one containing `exit 0`, exits **0**
  without printing `ccg-versions: … FAILED`.
- **Cause.** The drop-in is sourced (`. "$d" || true`), so its `exit` ends the parent script.
- **Fix.** Run each drop-in in a subshell that reports its own count, for example
  `( fail=0; . "$d"; exit "$fail" ); fail=$((fail + $?))`, which also isolates `set -e` and
  `cd`. Alternatively, reject `^\s*exit\b` next to the existing `bash -n` gate.
- **Acceptance.** The `20-bad` + `40-exit` pair above must exit non-zero.

## S09 (P0-08)

**Reviewed (2026-10-29):** `cpg/pyproject.toml`, `cpg/README.md`, the `.ccg-work/` block in
`cpg/.gitignore`, `cpg/ccg/__init__.py`, `cpg/ccg/ids.py` and `cpg/tests/test_ids.py`, all
uncommitted. Baseline: `docs/plans/connected-code-graph-p0.md` (§3 S09 row, §2 D2/D10, §5.2,
§9.1); `docs/plans/connected-code-graph-bundle-spec.md` v1.3 (§2.2, §3.1–§3.3, §3.6, §4.3, every
§5 id tuple, §8.3); and `docs/plans/connected-code-graph-p0-ml.md` §5.2 and §11 N6. Every claim
below was **run** unless it says *inferred*. All mutation work was done on a copy in the session
scratchpad.

**Verdict: approve with suggestions.** Both done-conditions hold. I ran the plan's command
verbatim (`.venv/bin/python -m pytest -m "not live"`) and got 90 passed. The test file's 24
32-hex ids match the spec's §3.6 table exactly (`diff` of the sorted sets is empty). Every
constructor's tag and element order matches the spec. No finding is a blocker. Two majors should
be fixed before S10 starts calling these constructors:
- **M1:** an integer position silently accepts a string and returns a wrong id.
- **M2:** the §5.15 `dependency` *record* id has no constructor.

`pyproject.toml` is frozen to S09 (§5.2), so m2 and m4 can only be fixed in this window.

CPG: considered, not relevant — `GRAPHS` lists only `cpg_falkorchat`; no CPG covers `cpg/`, and the module is 164 lines, read in full.

**What I ran.**
- **Suite:** 90 passed, both with the default `addopts` and with `-m "not live"` given
  explicitly. `-m live` deselects all 90, so the command-line `-m` does override `addopts`.
- **Constructor mutants (scripted):** every pairwise element swap, plus a changed tag, across
  every `tuple_id([...])` literal in `ids.py`. That is 81 mutants, and **81 were killed**. This
  includes the `launchFact`/`secretSighting`/`controlledBy`/`testMarker` cases teco checked.
- **Targeted mutants:**
  - Killed: drop NFC (6 failures); drop the bool guard (2); check the raw rather than the
    normalized `lang` (1); admit `dict` in `_normalize` (1); drop the point-kind check (1);
    truncate from the tail (43).
  - Survived: `sort_keys=False`, NFC applied after strip, and `if signature:` in place of
    `is not None`. The first two are equivalent (see below). The third is m5.
- **Spec check by hand:** every constructor's tuple against the §3.2 table and its §5 entry.
  Table in the Appendix.
- **Grammars (N6's done-condition):** all ten load and parse under `tree-sitter` 0.25.2. ABI
  versions: python, javascript, c_sharp, go and c are 15; typescript, tsx, java, kotlin and cpp
  are 14.
- **Wheel builds:** done on a copy with system `setuptools` 68.1.2, with stub
  `ccg/schemas/*.json` and `ccg/store/*.cypher` files (m2).

**The equivalent-mutant argument is sound.**
- **`sort_keys`:** `json.dumps` only ever receives the output of `_normalize`, which is lists,
  `str` and `int`. `_normalize` raises `TypeError` on a `dict` at any depth. A top-level `dict`
  is flattened to its keys by `list(tup)` before that, so a `dict` never reaches `dumps` either
  way. That flattening is its own defect (m3). `sort_keys` only reorders object keys, so it
  cannot be observed. The equivalence rests on the `dict` guard, and the guard is pinned:
  `test_objects_never_appear_in_a_tuple` kills the admit-`dict` mutant.
- **NFC applied after strip:** also equivalent, though less obviously. *Inferred, from a
  probe.* I compared both orders over every code point in nine edge contexts (`c+"x"`,
  `"x"+c`, `" "+c`, `c+" "`, and with U+00A0/U+2000 neighbours) and found **0** differences.
  Only U+2000 and U+2001 change under NFC, and both map to whitespace (U+2002/U+2003). No action
  is needed: the code follows the spec's stated order.

**Signature rule (§4.3): correct as far as a constructor can judge.** `SIGNATURE_LANGS` is
exactly `{java, kotlin, csharp, cpp}`, using §3.3's tokens. `c` is rejected, which matches GV-15
and the §3.3 "`.c` in a `cpp` unit" rule. The check runs on the *normalized* `lang`, so
`" java "` behaves the same as in the hash. The function-like half of §4.3 cannot be checked
without a symbol kind, so it is the caller's job. The README overstates this (m5).

### Findings

**M1 — major. An integer position silently accepts a string and returns a different, valid-looking
id.**
- **Evidence (run):**
  - `callsite_id(apiFetch, "fetch", "0")` returns `e82cf9b5…`, not GV-12's `41d8c78a…`.
  - `point_id(<GV-17 symbol>, "param", " -1 ")` returns GV-18's `0eb4a24f…`. The spec gives that
    vector precisely to show that this is a *different* id.
  - The reverse also passes: `unit_id(1, "p", "python")` returns a hash.
- **Where the code falls short:** `ids.py:83-164` carries `int` type hints, but nothing checks
  them at runtime.
- **Why it matters:** an id mismatch never raises. It shows up later as a join that finds
  nothing: a call site with no `DATA` edge, or a key that does not match gold. Ordinals and line
  numbers come from parsed Joern exports, YAML manifests and key files (S06/S39/S40), where a
  stray `"0"` is ordinary.
- **Affected positions:** callsite `ordinal`, point `n`, and the `ordinal` of `launchFact`,
  `decorator` and `import`. Also §5.15's `line`, once M2 adds that constructor.
- **Fix:**
  - Check each constructor's arguments by position. Use `type(v) is int` for the integer
    positions; that also rejects `bool`. Use `isinstance(v, str)` for the string positions.
  - Do **not** enforce this inside `tuple_id`: GV-18 has to stay hashable as a raw tuple.
- **Test (a coverage probe, not a list of cases):**
  - Write a table that maps each constructor to its tag and the type of each position, copied
    from the spec (Appendix table).
  - One parametrized test passes the wrong type at **every** position of **every** row and
    expects `TypeError`.
  - A second test asserts that the table's keys equal the set of public `*_id`/`name_key`
    functions in `ccg.ids`. A constructor added later without a row then fails the suite.

**M2 — major. The §5.15 `dependency` record id has no constructor.**
- **Evidence (read):** §5.15 fixes the record id as
  `["dependency", ecosystem, package, manifestPath, line]`, with `line` an integer. It is
  separate from the 3-element `depId`, so that one manifest can declare a package twice. §8.3
  sorts the `dependency` block by this id, and duplicate ids are a bundle error.
  `ids.py:109` offers only `dep_id`, the 3-element node id. I checked every other §5 record:
  - `param`/`ret`/`cond` use `point_id`;
  - `header` carries `unitId`;
  - every other record has a constructor.

  This is the only gap.
- **Why it matters:** S10 (sorting and duplicate checks) and S11 (the emitter) would each build
  this tuple by hand. The README tells callers not to do that, and D10 makes this module "the one
  place". A record id built from `dep_id` instead would merge runtime and dev declarations into a
  false duplicate-id error.
- **Fix:** add `dependency_record_id(ecosystem, package, manifest_path, line)`. Its test should
  compare it to `h([...])` written out from §5.15 with distinct values. Add it to M1's table, so
  that table's completeness check is what catches the next missing constructor. **Build the
  table from the spec's record list (§3.2 rows plus §8.3's types), not from `dir(ids)`.**

**m1 — minor. `test_marker_id` becomes a pytest test in any test module that imports it by name.**
- **Evidence (run):** a test file containing `from ccg.ids import test_marker_id` reports
  `ERROR tests/…::test_marker_id`, because pytest collects module-level `test*` functions. S10's
  `test_bundle.py` is a likely importer.
- **Fix:** add `test_marker_id.__test__ = False` after the definition. I verified it: with that
  line, the same probe gives `91 passed`. It keeps the name aligned with the record type.

**m2 — minor. `pyproject.toml` does not ship `ccg/schemas/*.json` or `ccg/store/*.cypher` in a
built package, and S09 is the last step that may edit the file.**
- **Evidence (run):** a wheel built from a copy that contains stub `schemas/bundle-v1.schema.json`
  and `store/ddl_v1.cypher` holds only `ccg/__init__.py` and `ccg/ids.py`. The reason: neither
  directory is a package, and `packages.find` sets no package-data.
- **Impact:** an editable install reads them from the source tree, so P0 works *(inferred)*.
  Anything that installs normally, such as P1 or a CI job, loses the schemas S10/S31/S34/S39 add.
- **Fix (verified, the wheel then contains both files):**

  ```toml
  [tool.setuptools.package-data]
  ccg = ["schemas/*.json", "store/*.cypher"]
  ```

- **Test:** none in the suite constrains this. The check is the wheel build above.

**m3 — minor. `tuple_id` accepts any `Sequence` as the tuple, so a mistaken argument is hashed
instead of rejected.**
- **Evidence (run):**
  - `canonical_bytes("unit")` returns `["u","n","i","t"]`.
  - `canonical_bytes({"b":1,"a":2})` returns `["b","a"]`.
  - Cause: `list(tup)` at `ids.py:43`.
- **Fix:**
  - Require `isinstance(tup, (list, tuple))`, a length of at least 1, and a `str` first element
    (the §3.1 tag).
  - Add one test per rejected shape: `str`, `dict` and `[]`.

**m4 — minor. `redis` is a direct dependency of the store spikes but is not declared; it is
resolved transitively (8.1.0) and unpinned.**
- **Evidence:** `falkordb` 1.6.2 `Requires: python-dateutil, redis`. In the venv it resolved to
  `redis` 8.1.0.
- **Why it matters:** S21 (`MULTI`/`RENAME` rollback) and S22 (ACL matrix, lock replay) probe
  Redis-level semantics. Their scripts will plausibly `import redis`, or depend on its
  connection behaviour through `falkordb-py`, which is V17's own mechanism. Both cases fall under
  §5.2's "every P0 dependency up front".
- **Fix:** add `"redis==8.1.0"` to `live`.
- **Other transitive dependencies** (`referencing`, `rpds-py`, `python-dateutil`) can stay
  unpinned for P0. A lock file is a P1 concern.

**m5 — minor. The signature check has two gaps, and the README claims more than the code
enforces.**
- **Empty signature (run):** a signature that is empty or all whitespace is accepted.
  `symbol_id("java", "r", "A::f", signature="   ")` hashes a 5-element tuple ending in `""`. The
  §4.3 grammar never yields `""`; its minimum is `()`. The `if signature:` mutant survives for
  this reason.
  - **Fix:** after `_token(signature)`, reject `""`, with one test.
- **README overclaim:** the README's last paragraph says the constructors "enforce the signature
  rule". They enforce only the language half. Reword to: "reject a signature outside Java,
  Kotlin, C# and C++. Passing one only for function-like symbols, and always for those, is the
  caller's job (§4.3)."

**Nits.**
- **pyproject comment (`pyproject.toml:6`):** "No base dependencies: `ccg.ids` is stdlib-only" will
  be the wrong reason once S10's `ccg.bundle` imports `jsonschema`. S10 cannot edit the file.
  Reword now, for example: "Runtime imports of `ccg.bundle` and the pack loaders (`jsonschema`,
  `pyyaml`) live in `dev` per §5.2's extras rule. Install `.[dev]`."
- **`data_id` docstring (`ids.py:140`):** the record fields are `toPath`/`fromPath`, while the
  tuple names them `usePath`/`defPath`. BF3 shows this is easy to read backwards. Add one line:
  `use_path` is the record's `toPath` and `def_path` is its `fromPath`.
- **`TUPLE_SHAPES` sample values:** `dep_id("pypi", …)` and `launch_fact_id(…, "exec", …)` use
  tokens outside the spec. The spec's tokens are `PyPI` and `command`. This is harmless for an
  order test, but the values get copied as examples.

**On `jsonschema`/`pyyaml` in `dev`: keep them there.** The S09 row and §5.2 say "in extras: `dev`,
`live`, `spike`". There is one venv, and every step installs `.[dev]`, so S10 works. The cost is
that a bare `pip install -e .` followed by `import ccg.bundle` fails. That only matters once
something installs `ccg` without extras, which is P1's porting decision. The nit above keeps the
file honest in the meantime.

### What's solid

- **Tests are independent of the code.** Expected ids are copied from §3.6, not computed. All
  22 vectors (plus the inner symbol and call-site ids) pass through the raw hash, and 17 of them
  also pass through the constructors. `TUPLE_SHAPES` writes each remaining tuple in spec order
  with a distinct value per position. That is why all 81 swap mutants die.
- **The Unicode cases are real.** GV-6 and the NFD tests use actual U+0301/U+030A/U+0308
  sequences; I checked each line with `unicodedata.is_normalized`. The strip test uses
  U+2003 and U+001F as its comment says. GV-10's 5,032 canonical bytes are pinned.
- **`bool` is refused before the `int` branch,** and float and `null` are rejected, so §3.1's
  "never in a tuple" rule holds.
- **Pins match the plan.**
  - `falkordb==1.6.2` satisfies V17's 1.6.x, and it is the version the quirk was re-verified on.
  - All four §5.2 names are present.
  - N6's ten grammars load. Using separate grammar wheels rather than `tree-sitter-language-pack`
    is fine: the ML note says "e.g.".
- **The README is accurate** apart from m5's sentence.
  - Venv: `.venv` is ignored by root `.gitignore:140`, which I confirmed with
    `git check-ignore`. `.ccg-work/` and `*.egg-info` are ignored too.
  - `infra/` and `toolchain/` exist as described.
  - The test commands work, and the D2 rows not yet built are labelled as arriving with their
    own step.

### Open questions

1. *For `teco`.* m2 and m4 edit `pyproject.toml`, which §5.2 freezes after S09. Should they go
   into this S09 fix pass (my recommendation), or into a later serialized `teco` edit?

### Appendix — S09 constructor ↔ spec table

Every row was checked against the spec text (§3.2, then the §5 entry), and by the swap mutants.
`i` marks an integer position. Every row matches.

| Constructor | Tuple in code | Spec source |
|---|---|---|
| `unit_id` | `unit, repoId, rootPath, lang` | §3.2, §2.2, GV-1 |
| `file_id` | `file, repoId, path` | §3.2, §5.2, GV-2 |
| `symbol_id` | `symbol, lang, repoId, qname[, signature]` | §3.2, §4.3, GV-3/7/15/16/22 |
| `callsite_id` | `callsite, symbolId, calleeName, ordinal(i)` | §3.2, §5.6, GV-12/14 |
| `point_id` | `point, symbolId, kind∈{param,ret,cond}, n(i)` | §3.2, §5.4–§5.5, GV-13/17 |
| `constant_id` | `constant, lang, repoId, qname` | §3.2, §5.9 |
| `config_entry_id` | `configEntry, repoId, path, keyPath` | §3.2, §5.17 |
| `launch_fact_id` | `launchFact, repoId, path, kind, ordinal(i)` | §3.2, §5.18 |
| `schema_entry_id` | `schemaEntry, repoId, path, objectKey` | §3.2, §5.18 |
| `dep_id` | `dependency, ecosystem, package` | §3.2, §5.15 `depId` |
| **(missing)** | `dependency, ecosystem, package, manifestPath, line(i)` | §5.15 record id (**M2**) |
| `name_key` | `name, lang, calleeName` | §3.2, GV-19 |
| `contract_point_id` | `contractPoint, kind, keyNorm` | §3.2, §6, GV-9/10 |
| `link_id` | `link, kind, keyNorm, providerAnchorId, consumerAnchorId` | §3.2 |
| `sighting_id` | `secretSighting, repoId, path, commit, fingerprint` | §3.2, §5.19 |
| `value_set_id` | `valueSet, callSiteId, argPath` | §5.8 |
| `data_id` | `data, use, def, usePath, defPath` | §5.10 |
| `controlled_by_id` | `controlledBy, point, cond` | §5.11 |
| `decorator_id` | `decorator, targetId, ordinal(i)` | §5.12 |
| `field_access_id` | `fieldAccess, pointId, path, mode, receiverType, field` | §5.13, GV-20/21 |
| `import_id` | `import, fileId, ordinal(i)` | §5.14 |
| `test_marker_id` | `testMarker, symbolId, framework` | §5.16 |

### Pass 2

**Re-checked (2026-10-29):** the fix round in `cpg/ccg/ids.py`, `cpg/tests/test_ids.py`,
`cpg/pyproject.toml` and `cpg/README.md`, all uncommitted. Everything below was **run** against
the tree. Mutants ran on scratch copies with `python -B` and `__pycache__` cleared before each
run, and each copy was confirmed restored with `cmp`.

**Verdict: approve with suggestions.** Every Pass 1 finding is fixed. There are two new minors,
both test gaps in the new guards, and neither is a defect in the ids. The suite gives
**185 passed** with `-m "not live"`.

**Mutants:**
- **Swap and tag:** 88 of 88 killed (the Pass 1 sweep, now also covering `dependency_record_id`).
- **Guard mutants:** 91 of 94 killed. These drop each `_strs`/`_ints` call, drop each argument
  from a `_strs` call, and disable the m3, m5 and m1 guards one at a time. The 3 survivors:
  - `_ints` accepting `bool`: equivalent, because `_normalize` still raises on `bool`.
  - Dropping `kind` from `point_id`'s `_strs`: equivalent, because `_token(kind)` raises first.
  - Dropping `lang` from `symbol_id`'s `_strs`: **not** equivalent (P2-2).

**Dispositions:**
- **M1: fixed.** Every position is type-checked. The three Pass 1 cases now raise `TypeError`.
  GV-18 still hashes as a raw tuple.
- **M2: fixed.** `dependency_record_id` hashes the §5.15 5-tuple, with `line` as an integer.
  Runtime and dev declarations of one package give two different ids.
- **m1: fixed.** With the `from ccg.ids import test_marker_id` probe added, the suite gives
  186 passed and no collection error.
- **m2: fixed.** A wheel built from a copy with stub files contains
  `ccg/schemas/bundle-v1.schema.json` and `ccg/store/ddl_v1.cypher`.
- **m3: fixed.** A `str`, a `dict`, an empty list and a non-`str` tag are each rejected. A Python
  `tuple` is still accepted (it gives GV-19).
- **m4: fixed.** `redis==8.1.0` is in `live`.
- **m5: fixed.** Empty, whitespace-only and non-`str` signatures are rejected. The README now
  says that passing a signature only for function-like symbols is the caller's job.
- **Nits: all fixed.** The pyproject comment, the `data_id` `toPath`/`fromPath` docstring, and
  the spec tokens `PyPI`/`command` in the table.

**teco's three questions:**
1. **Is the spec side spec-derived?** Partly.
   - `SPEC_TAGS` is a literal transcription. I extracted every backticked `["tag", …]` from
     spec §3.2 and §5 by script, and its tag set equals `SPEC_TAGS` exactly.
   - `test_table_covers_every_public_constructor` reads the module, which is the check the brief
     asked for.
   - The spec-side check is keyed by **tag only**, though. See P2-1.
2. **Does the new strictness break a shape the spec allows?** No. I ran 12 shapes, and all
   hashed:
   - an empty `cond` `usePath`;
   - receiver `-1`;
   - `rootPath` `"."`;
   - the signatures `String.()`, `static()` and `(out Order)`;
   - `<dynamic>`;
   - `receiverType` `"?"`;
   - a quoted `keyPath`;
   - a `line` of 10⁹;
   - a raw `tuple`;
   - GV-18 as a raw tuple.

   `_strs` allows `""`, and `_ints` allows negative integers.
3. **Do the M1 and m3 checks interact?** No. Every constructor builds a `list` with a literal
   `str` tag, so the m3 checks never fire on a constructor path. The M1 checks run before
   `tuple_id`, and `tuple_id` itself stays untyped per position, which keeps GV-18.

**P2-1 — minor (new). The spec-completeness check cannot detect the M2 defect class: a second
tuple shape under an existing tag.**
- **Evidence (run):** on a copy, I removed `dependency_record_id`, its table row and its two
  tests, recreating Pass 1's state. Result: **177 passed.** `"dependency"` is still covered by
  `dep_id`, so `test_table_covers_every_spec_record_type` stays green.
- **Fix:** key the spec side by `(tag, length including the tag)`, transcribed from §3.2 and §5.
  It should include both `("dependency", 3)` and `("dependency", 5)`, and both `("symbol", 4)`
  and `("symbol", 5)`. Compare it to `{(tag, 1 + len(positions))}` from the table, plus
  `("symbol", 4)` for the optional signature. My script-extracted set has exactly these shapes,
  so the check would have gone red on the Pass 1 tree.

**P2-2 — minor (new). `symbol_id`'s 4-element path is never type-probed.**
- **Evidence (run):** with `lang` removed from `_strs`, the suite stays at 185 passed, yet
  `symbol_id(7, "r", "q")` returns `d71a2f1b…`. The table's `symbol_id` row always passes a
  signature, so a wrong-typed `lang` is caught by `_token(lang)` in the signature branch instead.
- **Current state:** the shipped code is correct.
- **Fix:** add a second table row (or one test) for `symbol_id` without a signature, so the
  every-position probe covers the 4-element path too.
