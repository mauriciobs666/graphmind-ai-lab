# Containerizing the `cpg` MCP server — design note

> Design for running the `cpg` MCP server (`cpg/mcp/`) as a **container** instead of a host venv.
> Stakeholder request, verbatim: *"i want the mcp to be containerized"* — no further constraints,
> so every default below is chosen and justified here.
> Component: [`cpg/mcp/README.md`](../../cpg/mcp/README.md) ·
> prior design: [`cpg-query-access.md`](./cpg-query-access.md) ·
> requirements: [`../requirements/cpg-query-access.md`](../requirements/cpg-query-access.md).
> Author: `devops`. v1 2026-07-25 · **v2 2026-07-25 — amended in response to
> [`../reviews/cpg-mcp-containerization.md`](../reviews/cpg-mcp-containerization.md)** Part I
> (`analyst`, verdict *needs changes*) · **v3 2026-07-26 — the settle-before-implementation edits
> from that review's Part II** (verdict *approve with suggestions*, implementation authorised).
> See **§11 Review response** (Part I) and **§12 Review response — Part II** for the
> finding-by-finding maps.
> Status: **implemented.** v1/v2 were design-only; v3 is the version the shipped artifacts were
> built from, with the measured results folded back into §6. The v2 amendment built throwaway
> images to measure (§2, probe residue) — measuring was in bounds for a design, shipping was not.

**Reading order for an implementer:** §2 (what was verified live) → §3 (the decisions) →
§4 (artifacts) → §5 (sequence) → §6 (verification). §7–§9 are risk, rollback and doc impact.
§10 is the stakeholder nod, §11 the review response.

**What changed in v2, in one paragraph.** The review upheld every load-bearing decision (D-net,
D-src, D-image, D-ctx, D-test, D-host, the C-310 / `start_falkordb.sh` scope fence, rollback) and
concentrated its objections on the launch path. Three things are now materially different: the
launch wrapper **no longer builds on every start** — it computes a content-hash image tag and
gates on `docker image inspect`, because a warm `docker build` turns out to make a **Docker Hub
round trip every single time** (§2 E-18, the decisive measurement); the **startup budget is now a
measured number** (`MCP_TIMEOUT`, 30 s default, and MCP startup is *non-blocking* by default —
§2 E-17) rather than an invented ~2 s threshold; and **container lifecycle is designed** (§3.10),
after measurement confirmed that PID-1 `python` ignores `SIGTERM` (§2 E-23).

**Path convention.** `<repo-root>` stands for the absolute path of this repository. No absolute
machine path appears in this file or in any artifact it specifies —
`claude/scripts/audit-team.sh` check 7 greps every **tracked** file for the maintainer's home
path and username and fails the repo on a hit (§3.5, R-9 of the prior plan). Check 7 is
`git grep`-based and therefore **blind to untracked files**, so every new file below must be
grepped directly before commit (backlog C-309b). This constraint reaches into §4.3's hash
function too: it digests file *contents* and *relative* paths only, never an absolute path.

---

## 1. Goal & scope

Run the `cpg` MCP server as a Docker container so that a clone needs **Docker**, not a correctly
built local Python 3.12 venv, to answer CPG queries. The tool contract does not change: one tool,
`mcp__cpg__query(graph, cypher)`, two parameters, read-only, same output format.

**In scope:** a Dockerfile, a `.dockerignore`, a build script, a tag-derivation helper, a launch
wrapper, the new `.mcp.json` entry, the test story, and the documentation that describes them.

**Out of scope (explicitly):**

- **C-310** (OpenCode + Kiro MCP wiring). This change *touches* C-310's surface — the launch
  command changes — so §3.9 records the effect, but no OpenCode/Kiro config is written here.
- Taking over FalkorDB's lifecycle. `falkordb-dev` stays exactly as `falkor-chat/scripts/
  start_falkordb.sh` starts it (§3.1 rejects the alternative that would have changed it).
- Any change to `server.py`, the tool contract, or the truncation/directive semantics. **One
  exception is named but not taken:** the `os.getpid()`-derived scratch-graph name in
  `tests/test_server.py` degenerates inside a container (§2 E-26, m-6). That is a test-code fix
  owned by `tdd-engineer`/`coder`; this plan files it as backlog **C-321** and works around it in
  V-4 instead.
- Publishing the image to a registry (§3.6 names the upgrade path; nothing is published).
- The open defects C-314/C-315/C-316/C-318 — unrelated, unaffected.

---

## 2. Environment — verified live, 2026-07-25/26

Everything in this table was probed on this box. Rows **E-1…E-16** are from the v1 design run
(the reviewer independently re-verified E-1, E-3, E-4, E-12, E-14 and found no false claim).
Rows **E-17…E-28** are the v2 amendment's measurements — they exist because the review correctly
refused to accept E-16 and the ~2 s threshold on the plan's word.

| # | Fact | How verified |
|---|---|---|
| E-1 | Docker **29.6.1**, server OS `Ubuntu 24.04.4 LTS`, kernel `5.15.167.4-microsoft-standard-WSL2`, driver `overlay2`, root `/var/lib/docker`. This is a **native Linux engine inside WSL2**, not the Docker Desktop proxy. `buildx v0.35.0` is present, so **BuildKit is the default builder**. | `docker info`, `docker version` |
| E-2 | Two contexts exist: `default` (active, `unix:///var/run/docker.sock`) and `desktop-linux` (`npipe://…dockerDesktopLinuxEngine`). Docker Desktop is installed on the Windows side. | `docker context ls` |
| E-3 | `falkordb-dev` is up, `falkordb/falkordb:v4.18.11`, publishing `0.0.0.0:6379->6379` and `0.0.0.0:3000->3000`, attached to the **default `bridge` network only** (IP `172.17.0.2`, gateway `172.17.0.1`). | `docker ps`, `docker inspect falkordb-dev` |
| E-4 | **No user-defined network exists.** Only `bridge`, `host`, `none`. | `docker network ls` |
| E-5 | **Path A works**: bridge + `--add-host=host.docker.internal:host-gateway` resolves to `172.17.0.1` and `PING`→`+PONG` against FalkorDB. | `docker run --rm --add-host=… alpine:3.20 sh -c 'printf "PING\r\n" \| nc -w 3 host.docker.internal 6379'` |
| E-6 | **Path B works**: `--network host` + `127.0.0.1:6379` → `+PONG`. | same probe, `--network host` |
| E-7 | **Path C works**: bridge + the container IP `172.17.0.2:6379` → `+PONG`. | same probe, literal IP |
| E-8 | Without `--add-host`, `host.docker.internal` does **not** resolve on the default bridge. The flag is mandatory for Path A. | `getent hosts host.docker.internal` → no result |
| E-9 | **`docker run` is stdout-clean.** The `Unable to find image … Pulling …` progress and all status output go to **stderr**; with `2>/dev/null` stdout carried only the container's own bytes. This is load-bearing: anything on stdout corrupts the MCP stream. | `docker run --rm alpine:3.19 echo HELLO 2>/dev/null` → exactly `HELLO` |
| E-10 | `docker run -i --rm` (no TTY) round-trips stdin→stdout byte-exactly; a 1-byte input produced a 1-byte stdout and nothing else. | `printf 'x\n' \| docker run -i --rm alpine:3.20 cat 2>/dev/null` |
| E-11 | Container start overhead ≈ **0.6 s** (`alpine true`, three runs: 0.67 / 0.61 / 0.59 s). Superseded for planning purposes by the end-to-end numbers in E-20. | `/usr/bin/time docker run --rm alpine:3.20 true` |
| E-12 | Test baseline **green**: `53 passed, 7 deselected` offline; `7 passed, 53 deselected` with `-m live`. | `cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q` (and `-q -m live`) |
| E-13 | The `live` tests **write**: the `live_graph` fixture `CREATE`s a scratch graph `_cpg_mcp_selftest_<pid>` and `graph.delete()`s it in teardown. So the in-container live run needs a *writable* connection, not just the read path. | `cpg/mcp/tests/test_server.py:464-481` |
| E-14 | **Amended in v2.** `python:3.12-slim` was absent from the local **image store**, so the first build pulls it (~119 MB on disk). *The v1 verification method was subtly wrong in a way that matters:* `docker images python` reads the **image store**, and a successful BuildKit build populates only the **build cache**, not the store — after v1's cold build, `docker images python` was *still* empty. Store-vs-cache is the whole of E-18. | `docker images python` before and after a cold build |
| E-15 | **Assumed, not tested:** `docker network connect <net> falkordb-dev` attaches a *running* container without a restart. Deliberately not probed — it would mutate the shared FalkorDB container. Documented Docker behavior; §3.1 does not depend on it being true. | — |
| E-16 | **v1: "assumed, not measured" — now measured. Superseded by E-18/E-19/E-20/E-21.** The v1 guess ("well under 2 s") was right about the warm build in isolation and wrong about what it implied, because it never asked what the build was *doing* in that time. | see below |
| **E-17** | **Claude Code's real MCP startup budget.** `MCP_TIMEOUT` — *"Timeout in milliseconds for MCP server startup (**default: 30000**, or 30 seconds)"*. Distinct from `MCP_CONNECT_TIMEOUT_MS` (*"How long **blocking** MCP startup waits … (default: 5000)"*), which applies **only** when `MCP_CONNECTION_NONBLOCKING=0` or the server sets `alwaysLoad: true` — `cpg` does neither, so the reviewer's "standard 5-second connect timeout" does **not** bind here. And since **v2.1.142 MCP startup is non-blocking by default**: *"servers connect in the background and their tools become available as they finish."* The per-server `"timeout": 60000` in `.mcp.json` is confirmed to be the **per-tool-call** wall, not startup headroom. This box runs Claude Code **2.1.220**, so all of the above is live. | official docs: `code.claude.com/docs/en/env-vars` (`MCP_TIMEOUT`, `MCP_CONNECT_TIMEOUT_MS`, `MCP_CONNECTION_NONBLOCKING`, `MCP_TOOL_TIMEOUT`) and `…/docs/en/mcp`; `claude --version` |
| **E-18** | **A warm, fully-cached `docker build` makes a Docker Hub round trip every time — and it is essentially the entire cost.** With the base absent from the image store, `--progress=plain` shows every layer `CACHED` (`DONE 0.0s`) and exactly one non-cached step: `[internal] load metadata for docker.io/library/python:3.12-slim` → **`DONE 0.5s`**, against a **0.54–0.65 s** total build. After `docker pull python:3.12-slim` puts the base in the image store, that same step drops to **`DONE 0.0s`** and the total falls to **0.31–0.34 s**. **This kills R-8's v1 claim that "warm builds are offline"** and is the decisive input to §3.3. | `docker build --progress=plain` before/after `docker pull python:3.12-slim`; 3 timed runs each |
| **E-19** | **Cold build (no cache, base not pulled): 14.15 s wall**, of which `pip install` ≈ 5.8 s; the remaining ≈ 8.4 s is dominated by the base-image pull. **No transfer rate is claimed** — *(corrected in v3, m-16: v1/v2 wrote "~119 MB at ~50 MB/s", which does not reconcile with its own total and mixed two different sizes. 119 MB is the **on-disk** size `docker images` reports; the wire transfer for `python:3.12-slim` is ~45 MB compressed. The observed rate was therefore nearer 5 MB/s than 50.)* That total is **under** the 30 s budget on this link, so the reviewer's "tens of seconds, far beyond any plausible startup budget" does not hold *here* — but it is a property of this link, not of the design: on a link an order of magnitude slower, the cold build exceeds the 30 s budget and `build.sh` becomes required rather than optional. Treated as a gate, not a guarantee (§3.3). **Not reproducible from the current machine state** — the base is now in the image store and the build cache is warm, so re-measuring would require a destructive reset (see the residue note below). | `/usr/bin/time docker build --target runtime …` on an empty cache |
| **E-20** | **End-to-end connect cost, spawn → `initialize` + `tools/list`** (8 runs each, medians): host venv **0.37 s** · container, `docker run` only **1.42 s** · container **+ warm build** in the path **1.61 s** (base in store; **~1.9–2.1 s** if not) · container **+ `docker image inspect`** in the path **1.36 s**. *(v3, m-16/m-15: the container numbers are **not separable at this n**. The chosen variant is a strict superset of the `docker run`-only variant — it adds a `docker info` preflight, the hash and an `image inspect` — so its 1.36 s cannot really be *faster* than 1.42 s; that ordering puts run-to-run noise at ≥ 0.06 s. **Read all three container variants as ≈ 1.4–1.6 s, indistinguishable at n=8**, and read no latency ranking into them.)* Against the 30 s budget of E-17 that is **1.2 %** for the venv and **≈ 5 %** for every container variant — all fit with ≥ 18× margin, so **latency does not decide §3.3**; offline behaviour and concurrency do. | `/usr/bin/time` around a real JSON-RPC handshake piped into each variant |
| **E-21** | `docker image inspect <tag>` costs **0.05–0.07 s** (5 runs) and is a pure local daemon call — **no registry contact of any kind**. Computing the content hash over the build inputs costs **< 0.01 s**. | `/usr/bin/time docker image inspect`; `/usr/bin/time` around `sha256sum` |
| **E-22** | **`docker build` with a *path* context does not consume stdin.** Two sentinel lines piped into a wrapper that ran a full build were both still readable afterwards. So M-1 is a **latent** hazard, not a live bug — which is exactly why it must be guarded rather than relied upon. | `printf 'A\nB\n' \| bash -c 'docker build … ; cat'` → both lines survived |
| **E-23** | **Container lifecycle, measured.** (a) Without `--init`, PID-1 `python` **ignores `SIGTERM`**: the container was still `running` 3 s after `docker kill --signal=TERM`, and still `running` a minute later. (b) With `--init`, the same `SIGTERM` → `exited`, `ExitCode 143`. (c) Happy path: stdin EOF → the real server exits **0** in **1.46 s** and `--rm` reaps it. (d) `SIGKILL` to the `docker run` CLI while PID 1 is a process that *ignores stdin* leaves the container **`running` indefinitely** — a true orphan. (e) But with the **real** server (which blocks reading stdin), the same `SIGKILL` breaks the attach stream → the server sees EOF → exits → auto-removed **within 2 s**; `docker ps -a --filter label=cpg-mcp=1` was empty. So the *only* orphan path is a **wedged** server that is not reading stdin, and `--init` closes it at the `SIGTERM` step. | `docker kill --signal=TERM` with/without `--init`; `kill -9` of the CLI with and without a stdin-reading PID 1 |
| **E-24** | **m-1 confirmed exactly.** Test stage as `USER appuser` over a root-owned `/app`: `53 passed, 7 deselected, 1 warning` plus a `PytestCacheWarning: could not create cache path /app/.pytest_cache/… Permission denied`. Adding `RUN chown appuser:appuser /app` to the test stage yields `53 passed, 7 deselected` with **no warning** — genuinely byte-identical to E-12. | two builds of the test stage, differing only in the `chown`, each run |
| **E-25** | **m-3 confirmed decisively.** Same context, same Dockerfile, only `.dockerignore` differing: bare `__pycache__`/`*.pyc`/`.pytest_cache` → `find /app -name __pycache__ \| wc -l` = **1** (`tests/__pycache__` shipped); `**/__pycache__`/`**/*.pyc`/`**/.pytest_cache` → **0**. | two builds in an isolated copy of the context |
| **E-26** | **m-6 confirmed.** `docker run --rm <test-image> python -c "print(os.getpid())"` → **`1`**. So `tests/test_server.py:472`'s `_cpg_mcp_selftest_{os.getpid()}` collapses to the constant `_cpg_mcp_selftest_1` for every containerized live run. No such residue on the shared instance today (`GRAPH.LIST` → `cpg_falkorchat`, `ws:test`, `ws:acme`, `cpg_salesperson`, `reference`). | `docker run … python -c`; `redis-cli GRAPH.LIST` |
| **E-27** | **m-5 confirmed.** Missing image **without** `--pull=never`: `Unable to find image … locally` then `pull access denied for cpg-mcp…, repository does not exist or may require 'docker login'` — actively misleading for a locally-built image. **With** `--pull=never`: `Error response from daemon: No such image: cpg-mcp:<tag>`. | `docker run` with and without `--pull=never` against an absent tag |
| **E-28** | **An interrupted build makes durable progress.** A `--no-cache` build whose *client* was `SIGKILL`ed at t=4 s left layers 6–8 `CACHED` on the very next build; only the in-flight `pip install` re-ran. So the fresh-clone auto-build **converges monotonically** across session restarts rather than restarting from zero — a materially better story than the review assumed, though still not a first-session guarantee. | `kill -9` of the build client mid-`pip install`, then `docker build --progress=plain` |
| **E-29** | A failed MCP connect is **no longer silent to the model**: *"When a configured server fails to connect, Claude Code tells Claude which server failed and its connection error, including in `ToolSearch` results that find no matching tool"* (as of v2.1.205; this box is 2.1.220). Stdio servers are still **not auto-reconnected** — recovery is `/mcp` → reconnect, or a session restart. | official docs `…/docs/en/mcp` §Automatic reconnection; `cpg/mcp/README.md:232-236` |

**Probe residue — read this before implementing, it changes your starting conditions.**

- `alpine:3.20`, `alpine:3.19` (~15 MB) from the v1 network probes. Nothing references them.
- **`python:3.12-slim` is now in the local image store** (119 MB), pulled deliberately during the
  E-18 experiment. **Leave it.** It is what makes a warm build offline-safe (E-18) and §4.3 makes
  pulling it an explicit step of `build.sh` anyway. Consequence for the implementer: **your first
  build will not be cold** — to reproduce E-19 you would have to remove it first.
- The BuildKit cache holds the layers of the measurement builds (~114 MB, 43 MB reclaimable).
- All `cpg-mcp-measure:*` probe images were removed; `docker images` shows no `cpg-mcp*` tag, so
  the implementer starts from a genuinely unbuilt state for the *component's own* images.
- **Disclosure:** the E-28 experiment ran `docker builder prune -f --filter unused-for=0s` to
  create an empty cache. That discards **regenerable build cache only** — no image, container, or
  volume was touched, and `falkordb-dev` and `falkordb-data` were never involved. The visible
  cost is that the next `falkorchat:dev` / `falkor-chat-server` rebuild re-runs its layers.
- `falkordb-dev` was up throughout and is untouched; the shared graph list is unchanged (E-26).

---

## 3. Decisions

### 3.1 D-net — how the container reaches FalkorDB

**Decision: default bridge network + `--add-host=host.docker.internal:host-gateway`, with the
image defaulting `FALKORDB_HOST=host.docker.internal`.** *(Upheld by review — not reopened.)*

This is the central decision, because `FALKORDB_HOST=127.0.0.1` — today's `.mcp.json` value — means
*the container itself* once containerized, and would fail with the server's own
`FalkorDB unreachable at 127.0.0.1:6379` message.

All three candidates were **verified working live** (E-5/E-6/E-7). The choice is therefore about
privilege, portability and blast radius, not about feasibility.

| Option | Verified | Why not chosen |
|---|---|---|
| **A. `--add-host=host.docker.internal:host-gateway`** | ✅ E-5 | **Chosen.** |
| **B. `--network host`** | ✅ E-6 | Hands the container the host's **entire** network namespace to serve one outbound TCP connection to one port — maximal privilege for minimal need, against the least-privilege line the rest of this repo holds (non-root containers, scoped tools). It is also **Linux-engine-specific**: on Docker Desktop `--network host` does not reach the host's loopback the way it does here, and E-2 shows a Desktop context on this very box, so a contributor one `docker context use` away would silently lose the database. Option A is Docker Desktop's *native* mechanism and, with `host-gateway`, works identically on Linux engines. |
| **C. Literal container IP `172.17.0.2`** | ✅ E-7 | The IP is assigned by Docker at container start. `start_falkordb.sh` runs with `--rm`, so every restart can reassign it. Hardcoding it in a tracked file is a time bomb, and discovering it at launch means an `inspect` call and a parse in the stdio hot path. Rejected outright. |
| **D. Shared user-defined network** | not probed | Costed below. |

**Why D is rejected — the cost, stated explicitly.** DNS-based service discovery
(`FALKORDB_HOST=falkordb-dev`) only exists on a **user-defined** network. E-4 shows none exists,
and E-3 shows `falkordb-dev` sits on the default bridge. Getting there needs one of two changes,
both of which reach into shared infrastructure:

- **D-a — change `falkor-chat/scripts/start_falkordb.sh`** to create and join a network. That
  script is the canonical FalkorDB launcher **shared by `falkor-chat` and `salesperson`** (which
  delegates to it), and it runs with `--rm`, so applying the change means **stopping and
  re-creating the running shared container** — an approval-gated, cross-component disruption, plus
  re-testing two other components and reconciling `falkor-chat/compose.yaml`, which defines its own
  implicit network for the same service. That is a large blast radius to serve one consumer.
- **D-b — `docker network connect cpg-net falkordb-dev`** on the running container: additive and
  non-disruptive (E-15), but **not persistent**. `--rm` means the next `start_falkordb.sh` drops
  back to bridge-only and the MCP server breaks with no signal until someone runs a query. A
  hidden, manual, non-reproducible setup step is precisely the failure mode this repo's
  "one documented command" convention exists to prevent.

And D buys nothing A does not already give: FalkorDB **already publishes 6379 on the host**, which
is exactly what `falkor-chat`, `salesperson`, `redis-cli` and today's venv-based MCP server all
depend on. Option A rides that same published port, so it **adds no new coupling** — it inherits an
existing one. Revisit D only if the repo ever adopts a single Compose stack for all components;
that is a different, larger decision.

**Consequences to carry:**

- `.mcp.json`'s `FALKORDB_HOST` changes `127.0.0.1` → `host.docker.internal`.
- The **image** also sets `ENV FALKORDB_HOST=host.docker.internal`, so an ad-hoc `docker run`
  without `-e` is correct by default. `.mcp.json` still states it explicitly — the redundancy is
  deliberate, so a reader of the config sees the truth and has an override point.
- `FALKORDB_PORT` stays `6379` and continues to mean **the host's published port**. If someone
  starts FalkorDB with `FALKORDB_PORT=6380`, `.mcp.json` is wrong — the same exposure as today, no
  regression (R-4).
- **The sharper dependency is the bind address, not just the port** (m-9): this works because
  `start_falkordb.sh:52-58` publishes with `-p "${FALKORDB_PORT}:6379"`, i.e. `HostIp:""` →
  `0.0.0.0`. See R-6.

### 3.2 D-src — source baked into the image, not bind-mounted

**Decision: `COPY server.py` into the image. No bind mount.** *(Upheld by review — not reopened.)*

Rejected alternative — bind-mount `-v "$CLAUDE_PROJECT_DIR/cpg/mcp:/app:ro"` so edits take effect
without a rebuild. It would not leak a home path (the value comes from the environment at runtime,
§3.5), and the fast edit loop is genuinely attractive. It loses because **an image whose code lives
outside it is a venv with extra steps**: it cannot be handed to another machine, cannot be
published, and gives up the single strongest reason to containerize — a self-contained, verifiable
artifact. It also splits the unit of reproducibility: dependencies inside, source outside, free to
skew.

The real cost of baking is **staleness** — edit `server.py`, forget to rebuild, and the tool keeps
answering *from old code*. A wrong answer, not an error, which is the worst failure shape. §3.3
solves that deterministically rather than with a README warning.

### 3.3 D-fresh — the launch wrapper gates on a **content-hash image tag**, and builds only on a miss

> **This section is a v2 rewrite.** v1 decided "run a cached `docker build` before every launch"
> and budgeted it against a self-invented ~2 s threshold. The review (M-2) was right to refuse
> both halves. What follows is a decision made against measurements (E-17…E-21, E-28), not
> against a guess.

**Decision: the launch tag is `cpg-mcp:<hash12>`, where `<hash12>` is a SHA-256 content hash of
every build input. `docker-run.sh` computes the tag, and:**

```
docker image inspect "cpg-mcp:<hash12>"  →  HIT  : exec docker run … "cpg-mcp:<hash12>"   (0.05 s)
                                         →  MISS : build it, then exec docker run …
```

**A miss is the only thing that triggers a build.** Because the tag *is* the content, a hit is a
proof — not a heuristic — that the image was built from exactly **the repo bytes** now on disk.

> **What "content" does and does not cover** (v3, m-11). The hash digests the **tracked build
> inputs**: the Dockerfile, `.dockerignore`, both `requirements*.txt`, `server.py`, `pytest.ini` and
> everything under `tests/`. It deliberately does **not** cover the two inputs that live outside the
> repo: the **base image** (`python:3.12-slim` is a moving tag, not digest-pinned — §3.7) and **pip's
> dependency resolution** (`requirements.txt` pins ranges: `mcp>=1.28,<1.29`, `falkordb>=1.6,<1.7`).
> Two consequences, both accepted for a dev-only local tool:
> - Two builds of the same tree a month apart can produce **different images under the same tag**,
>   and the later `build.sh` silently re-points it. So the tag is immutable **with respect to the
>   tracked build inputs**, which is all R-10's concurrency argument needs — different trees still
>   cannot collide — but it is not a full image identity.
> - More likely in practice: once an image exists, **nothing ever refreshes its base**. A hash hit
>   will keep serving an image built on a base with a since-patched CVE, indefinitely. The refresh is
>   therefore an explicit, documented housekeeping act, not something the gate does:
>   `docker pull python:3.12-slim && cpg/mcp/build.sh --no-cache` (§3.6, and a README line).
>
> If reproducibility ever matters more than convenience here, digest-pinning the base is now cheap —
> the launch path no longer builds. §3.7 records that the reasoning has moved even though the answer
> has not.

#### The budget, measured (M-2, problem 1)

The number that governs this is **`MCP_TIMEOUT`, default 30000 ms** (E-17) — not the ~2 s v1
invented, and not the `"timeout": 60000` in `.mcp.json`, which is the per-tool-call wall. Two
further facts from E-17 change the shape of the risk:

- **MCP startup is non-blocking by default** since v2.1.142. Servers connect in the background and
  their tools appear as they finish; a slow connect **delays tool availability**, it does not stall
  the session. `MCP_CONNECT_TIMEOUT_MS` (5 s) binds only under `MCP_CONNECTION_NONBLOCKING=0` or
  `alwaysLoad: true`, and `cpg` sets neither.
- A connect that fails is **reported to the model** (E-29), so the v1 framing of "silent absence"
  is now too pessimistic. It is still not auto-reconnected — `/mcp` reconnect or a restart.

Against 30 s, every variant measured in E-20 is comfortable:

| Launch variant | Connect (median of 8) | % of the 30 s budget |
|---|---|---|
| Host venv (today's path) | 0.37 s | 1.2 % |
| Container, `docker run` only | 1.42 s | ≈ 5 % |
| **Container + `docker image inspect` — chosen** | **1.36 s** | **≈ 5 %** |
| Container + warm `docker build` (base in image store) | 1.61 s | ≈ 5 % |
| Container + warm `docker build` (base *not* in store) | ~1.9–2.1 s | ~7 % |
| Cold: full build + connect (E-19, this link) | ~15.6 s | ~52 % |

> **Do not rank the three container rows against each other** (v3, m-15). They span 1.36–1.61 s and
> the chosen variant is a strict superset of the `docker run`-only variant, so the fact that it
> measured *lower* bounds the run-to-run noise at ≥ 0.06 s rather than showing a saving. Treat them
> as **≈ 1.4–1.6 s, indistinguishable at n=8**. The venv-vs-container gap (0.37 s vs ≈ 1.4 s) is the
> only difference here that exceeds the noise, and it is R-7.

**So latency does not decide this.** Both mechanisms fit with ≥ 18× margin. v1's ~2 s threshold was
not merely un-derived, it was off by an order of magnitude in the *conservative* direction — it
would have triggered a fallback that was never needed.

#### The comparison that does decide it

| Axis | Build on every launch | **Content-hash tag + `docker image inspect`** |
|---|---|---|
| **Startup latency** | 1.61 s warm (0.31 s of build), ~1.9–2.1 s if the base is not in the image store. | 1.36 s; the gate is 0.05 s and the hash < 0.01 s (E-21). **A wash, not a win** — see the noise caveat above; the two are indistinguishable at n=8 and this axis decides nothing. |
| **Offline behaviour** | ❌ **Decisive against.** A warm, *fully cached* build still performs a Docker Hub `load metadata` round trip — 0.5 s, and essentially the entire build cost (E-18). It is offline-safe **only** if the base image sits in the local *image store*, which a BuildKit build does **not** put it there (E-14 amended). So without an explicit `docker pull`, every session start depends on Docker Hub reachability, DNS, and anonymous rate limits. On a plane, or during a Hub incident, the MCP server fails to start — a straight regression against the venv path, which needs no network at all. | ✅ `docker image inspect` is a pure local daemon call; **no registry contact in any circumstance** (E-21). Once the image exists, launching is fully offline. Building on a miss still needs the network, but that is the honest, once-per-change case rather than the every-launch case. |
| **Concurrency** | ❌ Two sessions (this repo actively encourages a root session plus component sessions) both build the **same mutable `:dev` tag**. BuildKit serialises the overlapping work, so session B's connect waits on session A's entire build; and if the two see different trees — one mid-edit — `:dev` ends up pointing at whichever finished last, non-deterministically (m-8). | ✅ Hash tags are **immutable and content-addressed**. Same tree → same tag → at most one build, and a concurrent duplicate build is idempotent. Different trees → **different tags**, so neither session can clobber the other's image. The race is removed rather than tolerated. |
| **Staleness correctness** | ✅ Correct. Docker's layer cache invalidates `COPY server.py` and rebuilds. | ✅ Correct, and by a stronger argument: the tag is a *function of the bytes*, so a changed input cannot resolve to an existing image. Also immune to a pruned build cache (which would silently make a "cached" build expensive). **Cost:** one new coupling — the hash input list must track what the Dockerfile `COPY`s. §4.3 pins that in one place and adds a `--verify-inputs` guard, and R-11 tracks it. |
| **Fresh clone** | Builds, and per E-28 *converges monotonically* across restarts. | Identical — a miss builds. No difference on this axis. |

**Decision: adopt the content-hash gate.** Offline behaviour and concurrency are decisive;
**latency is a wash** (the variants are within noise of each other — v3, m-15); staleness is a tie
bought with one named, guarded coupling.

Note the two mechanisms are **not** mutually exclusive and the chosen design keeps both halves:
the gate is what runs on every launch, and `docker build` is what a *miss* falls through to. So
this is strictly a superset of v1's self-healing property, minus the per-launch registry call.

#### The fresh clone, stated honestly (M-2, problem 2)

v1 claimed the auto-build meant "the server does not silently vanish" on a fresh clone. That
overclaimed. The measured truth:

- A cold build is **14.15 s** on this link (E-19) — under the 30 s budget, so on *this* machine the
  first session probably does converge on its own. But that is a property of **this link**, not of
  the design: the base pull is ~45 MB compressed, and on a link an order of magnitude slower than
  this one the cold build exceeds the 30 s budget and the connect fails. *(v3, m-16: v2 stated this
  as "at 5 MB/s the pull alone exceeds 20 s", derived from a ~50 MB/s rate its own E-19 total
  contradicts. No rate is claimed now.)* **`build.sh` is then required rather than optional.**
- If it does fail, **an interrupted build is not wasted** (E-28): completed layers persist in the
  BuildKit cache and the next attempt resumes. Convergence is monotonic across restarts, not
  Sisyphean.
- With non-blocking startup (E-17) a slow first connect degrades to *"the `cpg` tools appear late,
  or at the next query"*, and an outright failure is reported to the model (E-29).

**So: `cpg/mcp/build.sh` is the supported fresh-clone path** — the exact analogue of today's
`setup.sh`, and the README says so in the quick start. The build-on-miss is a **self-healing
safety net whose success on a slow link is not guaranteed**, not the documented path. That is the
honest claim, and it is weaker than v1's.

#### Rejected and reserve alternatives

- **`.mcp.json` calls `docker run <image>` directly.** Puts infra policy (network flags, env
  forwarding, image tag, `--init`) into a config file, gives no self-healing at all, and — see
  §3.9 — a one-line script ports to OpenCode/Kiro where a JSON `args` array does not. Rejected.
- **Build unconditionally on every launch (v1's choice).** Rejected on the offline and concurrency
  axes above. Retained only as the *miss* branch.
- **mtime comparison** of the inputs against the image's creation timestamp. Now rejected outright
  rather than held in reserve: the content hash dominates it on every axis at the same cost, so
  there is no scenario left where an mtime heuristic is the right answer. *(v1 rejected it partly
  for a bad reason — the review is right that `git checkout` sets mtimes to* now*, never into the
  past, so the heuristic errs toward a rebuild, the safe direction. It loses on merit, not on that
  argument.)*
- **Build only when the image is missing, with a mutable tag.** This is the trap the content hash
  avoids: with `:dev`, "missing" and "stale" are different questions and only the first is asked.
  With a content-hash tag they are the **same question**, which is the whole trick.
- **Digest-pinning the base image** would also make warm builds offline-safe, but it diverges from
  the repo's precedent (§3.7) and only helps the branch we no longer take on every launch.
  Instead, §4.3's `build.sh` does an explicit `docker pull` of the base so that a *miss*-triggered
  build is as offline-tolerant as it can be.

**Escape hatches**, both documented in the README: `CPG_MCP_NO_AUTOBUILD=1` (never build in the
launch path — fail with a curated "run `cpg/mcp/build.sh`" message instead) and `MCP_TIMEOUT`
(raise the startup budget, e.g. `MCP_TIMEOUT=60000 claude`, for a slow first build).

### 3.4 D-test — host venv stays primary; a second command proves the image

**Decision: keep `cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q` (and `-q -m live`) exactly as they
are, and add an in-container run as a separate, documented gate.** *(Upheld by review.)*

This component's *only* regression signal must not get slower or more fragile. The host venv run is
0.5 s (E-12) and needs no Docker; routing it through a container would make every unit-test run
depend on a build. But a host-only test story lets the **image** drift from the tested code — a
Python bump or a dependency resolution difference inside the image would be invisible.

So, two paths with different jobs:

| Path | Command | Job | When |
|---|---|---|---|
| **Host venv** (unchanged, primary) | `cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q` · `-q -m live` | The regression signal. Fast inner loop. | Every code change. |
| **In-container** (new gate) | `docker run --rm --add-host=host.docker.internal:host-gateway cpg-mcp:test-<hash> python -m pytest tests -q` (and `-q -m live`) | Proves *the image* — its interpreter, its resolved deps, and its network path — actually works. | Dockerfile, `requirements*.txt` or base-image change; before declaring the container path good. **Precondition (m-7): run `cpg/mcp/build.sh` — both targets — immediately first.** |

**The m-7 precondition is not optional.** The launch wrapper only ever materialises the *runtime*
tag; nothing else refreshes the test tag. Running the gate against a stale test image would
reintroduce, on the test side, precisely the staleness failure §3.2/§3.3 exist to prevent. Making
the test tag **share the runtime tag's content hash** (§3.6) turns this from a discipline problem
into a checkable one: if `cpg-mcp:test-<hash>` does not exist for the current hash, the gate
cannot be run at all by accident — the tag simply is not there.

Expected results are identical to E-12: `53 passed, 7 deselected` / `7 passed, 53 deselected`.
The live run is also the **networking proof** (§6, V-4): those tests connect to a real FalkorDB from
inside the container, and per E-13 they write and delete their own scratch graph, so a green live
run proves a fully working bidirectional connection, not just a TCP handshake.

**Two container-specific caveats on the suite, both measured:**

- **The scratch-graph name degenerates (m-6, E-26).** `tests/test_server.py:472` derives it from
  `os.getpid()`, which is **`1`** in a PID namespace, so every containerized `-m live` run uses the
  same key `_cpg_mcp_selftest_1` on the **shared** FalkorDB. E-13's "self-contained" argument still
  holds against `cpg_*`/`ws:*`/`reference`, but the uniqueness that made it safe *against itself* is
  gone: two concurrent container live runs corrupt each other, and an interrupted one leaves
  residue on a shared instance. **This plan does not change test code** (§1). Mitigations here:
  V-4 is documented as **not to be run concurrently**, and its done-condition gains a
  `GRAPH.LIST` residue check. The proper fix — `uuid4().hex[:8]` instead of `getpid()` — is filed
  as backlog **C-321** for `tdd-engineer`/`coder`.
- **The test stage must own `/app` (m-1, E-24).** As `USER appuser` over a root-owned `/app`,
  pytest cannot write `.pytest_cache` and emits `PytestCacheWarning: … Permission denied`, giving
  `53 passed, 7 deselected, 1 warning`. §4.1 therefore adds `RUN chown appuser:appuser /app` to
  the test stage, which restores a genuinely clean, byte-identical `53 passed, 7 deselected`
  (measured both ways). The alternative, `-p no:cacheprovider` on the command line, was rejected:
  it changes the invocation rather than the environment, so the host and container gates would no
  longer be running the *same* command.

**Multi-stage, so the runtime image carries no test surface.** A `test` stage installs
`requirements-dev.txt` and copies `pytest.ini` + `tests/`; the `runtime` stage installs only
`requirements.txt` and copies only `server.py`. Rejected alternative — one image with pytest baked
in, run as `docker run cpg-mcp:dev python -m pytest`: simpler by one tag, but ships a test runner
and the test suite into the artifact that would eventually be published, for no gain. Multi-stage
is cheap here because both stages share the same base layers.

Note `tests/conftest.py` derives `sys.path` from its own location (`parents[1]`), so with
`/app/server.py` + `/app/tests/` and `WORKDIR /app` it resolves correctly with no `PYTHONPATH`.
`pytest.ini` must be copied to `/app` for `testpaths` and the `-m "not live"` default to apply.

### 3.5 D-host — `setup.sh`, `run.sh` and the venv are **retained**

**Decision: retained, unchanged, and re-documented as (a) the test path and (b) the fallback.
Containerization is additive.** *(Upheld by review.)*

Four reasons, any one sufficient:

1. **They are the test story** (§3.4). Deleting the venv deletes the fast regression loop.
2. **They are the fallback read path.** If the image is broken — a bad build, no Docker, a wrong
   context (R-5) — the alternative is `redis-cli GRAPH.QUERY`, i.e. back to hand-assembled shell
   command lines with all the quoting hazards this component was built to eliminate. Keeping
   `run.sh` means a broken image costs one line in `.mcp.json`, not a capability.
3. **They make rollback free** (§8). Revert two lines, restart the session, done.
4. **`run.sh` is what ports to a Docker-less host** — relevant to C-310 (§3.9). *(v1 also cited the
   README's `claude mcp add --scope local` recipe here. Per m-10 that citation was backwards: that
   recipe exists for `$CLAUDE_PROJECT_DIR` expansion failure, which is orthogonal to
   venv-vs-container, so after this change it must be updated to point at `docker-run.sh` — see
   §9. `run.sh`'s portability argument stands on its own without it.)*

Rejected alternative — container-only, delete `setup.sh`/`run.sh`/`.venv`. Smaller surface, one
way to do things. It loses on all four points above, and it trades a 3-file surface for a
single-point-of-failure at session start on a component whose recovery needs a session restart.

The consequence to document honestly: **two paths now exist and can drift.** §3.4's in-container
gate is the control for that, and both paths install from the same `requirements*.txt`.

### 3.6 D-name — content-hash tags, with human aliases, no registry

**Decision (amended in v2 to serve §3.3):**

| Tag | Mutability | Who uses it |
|---|---|---|
| `cpg-mcp:<hash12>` | **immutable w.r.t. the tracked build inputs** — content-addressed over the repo bytes only (m-11) | The launch tag. The only tag `docker-run.sh` ever names. |
| `cpg-mcp:test-<hash12>` | same, same `<hash12>` | The §3.4 gate. Sharing the runtime hash is what makes m-7 checkable. |
| `cpg-mcp:dev`, `cpg-mcp:test` | moving aliases | Humans and ad-hoc `docker run`. Re-pointed by `build.sh` to the tags it just built. Never referenced by the launch path, so they cannot cause the m-8 race. |

`<hash12>` is the first 12 hex chars of a SHA-256 over all build inputs (§4.3). `:dev` over
`:latest` for the alias because `latest` carries implicit-pull semantics that are wrong for a
locally-built image. No registry prefix — nothing is published. Overridable via `CPG_MCP_IMAGE`
(which, if set, **bypasses the hash gate entirely** and is treated as "the caller knows what they
are running"; documented as such). Upgrade path if it is ever published:
`ghcr.io/<org>/cpg-mcp:<version>`, tagged from a git tag; out of scope here.

Precedent check: the in-repo hand-built precedent is `falkorchat:dev` (the
`falkor-chat-server:latest` tag is Compose-generated, not hand-chosen), so the `:dev` alias keeps
the repo's naming shape while the hash tag carries the correctness property.

**Accumulation, and why cleanup is *not* automated.** Every distinct input state leaves an image
(~150 MB, mostly shared base layers, so the marginal cost of each is small). Old hash tags
accumulate. The Dockerfile therefore sets `LABEL cpg-mcp=1`, making them enumerable:

```bash
docker image ls --filter label=cpg-mcp=1        # inspect what has accumulated
```

Pruning is a **destructive, approval-gated** operation and is deliberately left to a human with
that command's output in front of them. Nothing in the launch path ever removes an image — a
wrapper on the MCP startup path must never be able to destroy state. Documented in the README as
periodic housekeeping.

**The other housekeeping act is a base refresh** (m-11): because the base image sits outside the
hash, a hash hit will serve an image built on an old base forever. There is no automatic refresh and
there deliberately is not one — it would put a network pull back on the launch path. The manual form,
documented in the README beside the pruning line:

```bash
docker pull python:3.12-slim && cpg/mcp/build.sh --no-cache   # rebuild on a fresh base
```

### 3.7 D-image — `python:3.12-slim`, non-root, **no** `HEALTHCHECK`, **no** `EXPOSE`

*(Upheld by review.)* Base image `python:3.12-slim` and a non-root `appuser` follow
`falkor-chat/Dockerfile`; `setup.sh` already requires Python ≥ 3.12 and the `mcp`/`falkordb` pins
are only verified there. Rejected: an Alpine base (the two deps are pure Python so it would work,
but it diverges from the repo's precedent to save ~40 MB on a dev-only tool). Not pinning a digest
also follows the precedent — `python:3.12-slim` is a moving tag, and consistency within the repo
beats a marginal reproducibility gain here; revisit if the image is ever published. (§3.3 notes
that digest-pinning would *also* solve E-18's registry round trip; it is unnecessary now that the
launch path does not build.)

**The reasoning behind that answer has moved, even though the answer has not** (v3, reviewer's open
question 3). While the launch path built on every start, a digest pin was a way to make that build
offline-safe — a correctness argument. It no longer builds, so the only thing a pin would buy now is
closing m-11's "same tag, different base" wobble and making `<hash12>` a true image identity, at the
cost of a manual bump and a divergence from `falkor-chat/Dockerfile`. That is a reproducibility
preference, not a defect, so it stays deferred — but it is now a **cheap** change if the stakeholder
ever wants it, and the README's base-refresh line (§3.6) is the interim answer.

**Deliberate divergence from that precedent, with reason:** `falkor-chat/Dockerfile` has
`EXPOSE 8000` and a `HEALTHCHECK`. **Both are meaningless here and are omitted.** This container
is not a service: it is a one-shot `docker run -i --rm` process that owns a stdio pipe for the
lifetime of one session, listens on **no** port, and has no orchestrator polling it. A healthcheck
would be cargo-culted ceremony that reports on nothing. For the same reason there is **no Compose
service**: Compose models long-lived services, `docker compose run --rm -T` would add a layer for
no benefit, and `falkor-chat/compose.yaml` already defines a `falkordb` service that would try to
bind a **second** engine on `:6379` over the same volume — actively harmful (its own header warns
about exactly this).

Hardening deliberately **not** in the first cut: `--read-only --tmpfs /tmp`. The server genuinely
never writes to disk (`PYTHONDONTWRITEBYTECODE=1`), so it should work — but if it does not, it
fails at *session start*, the worst place to discover it. V-8 tries it; adopt only if green.
**n-4:** those flags live on the wrapper's `docker run`, so adopting them **changes the launch
path** — re-run **V-6 and V-7 as well as V-8**, and treat a green V-8 alone as insufficient.

### 3.8 D-ctx — build context is `cpg/mcp/`, not `cpg/` or the repo root

*(Upheld by review.)* Everything the image needs lives in `cpg/mcp/`. A narrow context means the
daemon never receives `cpg/.cpg-artifacts/` — the durable CPG reload artifacts, including a ~37 MB
`load.cypher` and a `cpg.bin` — in the build tar. **Choosing the narrow context makes a
`cpg/.cpg-artifacts` ignore entry unnecessary** (it is outside the context entirely), which is the
stronger fix. `falkor-chat` used its component root only because its app resolves `web/` as a
sibling of `server/` — a constraint that does not apply here.

`.dockerignore` is resolved relative to the build context, so it goes at `cpg/mcp/.dockerignore`
with patterns relative to `cpg/mcp/`. **Per m-3/E-25 those patterns need a globstar prefix**
(`**/`), because Docker's ignore patterns are `filepath.Match`-based and a bare `__pycache__`
matches only at the context root. Measured: bare patterns shipped `tests/__pycache__` into the
image; globstar-prefixed patterns did not.

### 3.9 D-c310 — effect on C-310 (recorded, not scoped in)

*(Upheld by review.)* Containerizing is **net positive to slightly positive** for C-310:

- **Helps:** the launch surface stays *a single command* — `cpg/mcp/docker-run.sh` replacing
  `cpg/mcp/run.sh` — so the property C-310 depends on (the *command* ports even though the config
  file does not) is preserved exactly. It also removes a per-host variable: "is there a working
  Python 3.12 venv at that path" becomes "is there a Docker daemon", which is easier to state and
  to check. This is a second reason §3.3 chose a wrapper script over inline `docker run` args: a
  script is a portable command; a JSON `args` array is not.
- **Hinders:** it adds Docker as a prerequisite for any harness host, and each harness's config
  must carry the same two env vars. Note also that `MCP_TIMEOUT` (E-17) is a *Claude-Code* knob;
  OpenCode and Kiro will have their own startup budgets, which C-310 must establish separately.

No OpenCode or Kiro config is written by this work. Add one cross-reference sentence to C-310;
**do not renumber it.**

### 3.10 D-cycle — container lifecycle: `--init`, a label, no `--name`, `--rm` retained

> **New in v2** (M-3). v1 specified `docker run -i --rm --add-host=… "$IMAGE"` and said nothing
> about signals, naming, or orphans. Every claim below is measured (E-23).

**Decision: `docker run -i --rm --init --label cpg-mcp=1 --pull=never …`, with no `--name`.**

**Why `--init` is required, not decorative.** PID 1 has no default signal dispositions, and CPython
installs a handler only for `SIGINT`. Measured (E-23a): without `--init`, `SIGTERM` to the
container is **ignored** — still `running` 3 s later, and still `running` a minute later. With
`--init` (E-23b), tini is PID 1, forwards the signal, and the container `exited` with
`ExitCode 143`. `--init` also reaps zombies, though this workload forks nothing.

**The full shutdown analysis.** Claude Code's stdio shutdown sequence is: close the child's stdin →
wait → `SIGTERM` → `SIGKILL`.

| Path | Measured behaviour | Verdict |
|---|---|---|
| **Happy** — server blocked on a stdin read, stdin closes | EOF → clean `exit 0` in **1.46 s**; `--rm` reaps the container (E-23c). | Fine, no change needed. |
| **Escalation to `SIGTERM`** | Without `--init`: ignored, container survives (E-23a). With `--init`: `exited 143` (E-23b). | **`--init` is the fix.** |
| **`SIGKILL` of the CLI, server reading stdin** | The attach stream breaks → the server sees EOF → exits → auto-removed **within 2 s**; the label filter showed nothing left (E-23e). | Fine. |
| **`SIGKILL` of the CLI, server *wedged* and not reading stdin** | Container stays **`running` indefinitely** — a true orphan (E-23d). | The one real orphan path. `--init` closes it one step earlier, at `SIGTERM`. |

So the residual exposure is narrow — a server wedged inside a FalkorDB call *and* a `SIGKILL` that
skips `SIGTERM` — but it is real, it accumulates on an engine that also runs `falkordb-dev`, and
v1 gave no way to even *find* such a container. Hence:

- **`--label cpg-mcp=1`** on the `docker run` (and `LABEL cpg-mcp=1` in the image, §3.6) so
  orphans are enumerable: `docker ps -a --filter label=cpg-mcp=1`. This is the handle that turns
  an unknown into a checked property — see **V-11**.
- **No `--name`.** A deterministic name would collide the moment two sessions run concurrently
  (which this repo encourages), turning a benign duplicate into a hard `docker run` failure at
  session start. Docker's generated name plus the label is the right combination: unique per run,
  discoverable in aggregate.
- **`--rm` retained.** Measured to reap correctly on both normal exit and CLI death (E-23c/e).
  It is also what keeps the *non*-orphan cases from accumulating anything at all.
- **Cleanup of a found orphan is a human, approval-gated act** (`docker stop` then `docker rm`).
  Neither `docker-run.sh` nor `build.sh` ever stops or removes a container — a script on the MCP
  startup path must not be able to kill a process it did not start, and a stale-container heuristic
  that guesses wrong would take out a *live* session's server.

---

## 4. Artifacts

Five files: three new scripts (one of them sourced, not executed), the Dockerfile, the
`.dockerignore`, plus a two-line `.mcp.json` edit.

### 4.1 `cpg/mcp/Dockerfile` — new

```dockerfile
# The `cpg` MCP server as a container: ONE stdio process per session, launched by
# cpg/mcp/docker-run.sh (see the repo-root .mcp.json) — NOT a long-lived service.
# No EXPOSE and no HEALTHCHECK on purpose: it listens on nothing and no orchestrator
# polls it. See docs/plans/cpg-mcp-containerization.md §3.7.
#
# Build context is cpg/mcp/ (narrow, deliberately — keeps cpg/.cpg-artifacts/ out of
# the build tar). Build with cpg/mcp/build.sh, never by hand: the image TAG is a
# content hash of the build inputs and build.sh is what computes it (§3.3, §3.6).
#
# IMPORTANT: every path COPYed below must appear in image-tag.sh's input list, or the
# hash gate will not notice a change to it. `build.sh --verify-inputs` enforces this.

FROM python:3.12-slim AS base

LABEL cpg-mcp=1

# FALKORDB_HOST default: 127.0.0.1 would mean *this container*. The host-gateway
# alias is supplied by docker-run.sh's --add-host. See §3.1.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FALKORDB_HOST=host.docker.internal \
    FALKORDB_PORT=6379

WORKDIR /app

# Before the COPYs, so a source edit does not re-run it (n-1).
RUN useradd --create-home --shell /usr/sbin/nologin appuser

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py server.py

# --- test stage: dev deps + the suite. Never shipped as the runtime image. ----
FROM base AS test
COPY requirements-dev.txt requirements-dev.txt
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY pytest.ini pytest.ini
COPY tests tests
# pytest writes .pytest_cache into its rootdir; without this, appuser cannot, and the
# suite emits a PytestCacheWarning that makes the output differ from the host baseline
# for no reason (m-1, measured — see §3.4).
RUN chown appuser:appuser /app
USER appuser
CMD ["python", "-m", "pytest", "tests", "-q"]

# --- runtime stage: server + runtime deps only. No pytest, no tests. ---------
FROM base AS runtime
USER appuser
CMD ["python", "server.py"]
```

Note the `ENV` block carries no interleaved comment lines — a `#` inside a line-continued
instruction is not portable across builders; the explanation sits above it.

### 4.2 `cpg/mcp/.dockerignore` — new

```
# Context is cpg/mcp/. Patterns are relative to it, and are `filepath.Match`-based:
# a bare `__pycache__` matches ONLY at the context root, so the `**/` prefix is load-
# bearing for anything nested. Measured: bare patterns shipped tests/__pycache__ into
# the image; these do not. (§3.8, m-3.)
**/__pycache__
**/*.pyc
**/.pytest_cache
.venv
README.md
Dockerfile
.dockerignore
build.sh
docker-run.sh
image-tag.sh
run.sh
setup.sh
```

`cpg/.cpg-artifacts/` needs no entry — it is outside the context (§3.8). The host-path scripts are
excluded so they cannot be confused with the container path from inside the image.

> **Consistency obligation:** `.dockerignore` changes what ships, so it is itself a build input
> and appears in `image-tag.sh`'s hash list (§4.3).

### 4.3 `cpg/mcp/image-tag.sh` — new, **sourced** (not executable, no shebang behaviour relied on)

The single definition of the content-hash tag, shared by `build.sh` and `docker-run.sh` so the two
can never disagree about which image is current. Sourced rather than executed on purpose: it means
no fork on the launch path and, more importantly, **no function that writes to stdout** anywhere
near the MCP stream (§3.3, R-2).

```bash
# image-tag.sh — sourced by build.sh and docker-run.sh. Defines the content-hash image
# tag. NOT executable, prints nothing, reads nothing from stdin.
#
# cpg_mcp_input_files  -> the exact set of build inputs, NUL-separated, relative paths
# cpg_mcp_image_tag    -> sets CPG_MCP_TAG=<hash12>  (assigns a variable; no output)
#
# The hash covers file CONTENTS and RELATIVE paths only — never an absolute path, so
# the value is identical on every machine and no home path can reach a tracked file
# (see the plan's "Path convention"). `sha256sum < "$f"` is deliberate: it digests the
# bytes without the filename, and the relative path is contributed explicitly.
#
# INVARIANT: this list must cover every path the Dockerfile COPYs, plus the Dockerfile
# and .dockerignore themselves. `build.sh --verify-inputs` checks it (R-11).
```

**The enumeration is directory-driven, not glob-driven** *(v3, M-4 — settled before S2, because it
changes one function here and one check in §4.4)*. v2 specified the fixed file list **plus every
`*.py` under `tests/`**, while the Dockerfile does `COPY tests tests` — a **directory** operand. That
mismatch broke the guard in two ways at once:

- `--verify-inputs` was specified as "assert each `COPY` source is covered by the input list", and the
  list never contains the string `tests` — only `tests/conftest.py` and `tests/test_server.py`. The
  natural implementation (set membership) therefore **fails on the Dockerfile this plan itself
  specifies**, making S3's done-condition unreachable without inventing an unstated rule.
- The rule that *would* make it pass (prefix matching) leaves the real hole open: a **non-`.py`** file
  added under `tests/` — a JSON fixture, a `.cypher` sample — is `COPY`ed into the test image but not
  hashed, so `cpg-mcp:test-<hash>` silently stays stale and the gate runs against old test inputs.
  That is exactly the m-7 staleness §3.4 claims is now structurally impossible. No live defect today
  (`tests/` holds only `conftest.py`, `test_server.py` and an ignored `__pycache__`), which is
  precisely when it is cheap to close.

So the input set is defined by **the same two categories the Dockerfile uses** — file operands and
directory operands — and directory operands are *walked*, with the `.dockerignore` exclusions applied:

| Operand kind | Enumerated as | Covered by `--verify-inputs` iff |
|---|---|---|
| file (`server.py`, `pytest.ini`, `requirements*.txt`) | itself | the exact relative path is in the set |
| directory (`tests`) | `find <dir> -type f ! -path '*/__pycache__/*' -print0 \| LC_ALL=C sort -z` | **the enumeration walks that directory with the same exclusions** — i.e. the directory name appears in the *walked-directory* list, not in the file set |

Body:

- `CPG_MCP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`.
- `cpg_mcp_input_dirs()` emits the directory operands that are **walked**: `tests`. This is the list
  `--verify-inputs` checks a directory `COPY` against, and it is why the two can never disagree.
- `cpg_mcp_input_files()` emits, NUL-separated and in a deterministic order: the fixed file list
  `Dockerfile .dockerignore requirements.txt requirements-dev.txt server.py pytest.ini`, then, for
  **every** directory in `cpg_mcp_input_dirs`, every file under it that `.dockerignore` would not
  exclude — `find <dir> -type f ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z`.
  `LC_ALL=C` matters — a locale-dependent sort would make the hash machine-dependent. **Note the
  exclusion set here must stay in step with §4.2's `**/__pycache__` / `**/*.pyc` / `**/.pytest_cache`:
  hashing a file the build tar does not receive would make the tag change without the image
  changing.** A `.pyc` under `tests/` cannot occur outside `__pycache__/` in practice, so one
  `! -path` is sufficient and is deliberately not generalised into a second `.dockerignore` parser.
- `cpg_mcp_image_tag()` pipes, for each input, `printf '%s\0' "<relpath>"` followed by
  `sha256sum < "<abspath>"`, into a final `sha256sum`, and assigns
  `CPG_MCP_TAG="$(… | cut -c1-12)"`. A missing input is a hard error to stderr and a non-zero
  return — silently hashing an absent file would make two different trees collide.

Because the walk is the definition, **adding a fixture under `tests/` changes the hash
automatically** — §4.2's ignore patterns and §4.3's input set now agree by construction rather than
by an editor remembering.

Measured cost: **< 0.01 s** (E-21).

### 4.4 `cpg/mcp/build.sh` — new, executable

The Docker analogue of `setup.sh`: idempotent, once per change, `--help`, env overrides.

```bash
#!/usr/bin/env bash
set -euo pipefail
exec </dev/null                 # M-1: this script reads NOTHING from stdin, ever —
                                # when docker-run.sh calls it, stdin is the MCP pipe.
# build.sh — build the `cpg` MCP server images at the current content-hash tag.
# Idempotent, and specifically: if the target tags ALREADY EXIST it does nothing at
# all — no docker build, and no docker pull either. --no-cache is the way to rebuild
# an existing tag. (v3, m-12: v2 said "nothing to do" in the header and then described
# an unconditional pull + build in the body. The early exit is the rule.)
#
#   cpg/mcp/build.sh                  # runtime + test at <hash>, plus :dev/:test aliases
#   cpg/mcp/build.sh --runtime-only   # just cpg-mcp:<hash>  — what docker-run.sh calls
#   cpg/mcp/build.sh --no-cache       # force a clean rebuild
#   cpg/mcp/build.sh --verify-inputs  # check image-tag.sh covers every Dockerfile COPY
#   cpg/mcp/build.sh --help
#
# ALL output goes to stderr, and stdin is closed above: docker-run.sh calls this on the
# MCP stdio path, where a stray byte on stdout corrupts the protocol and a byte READ
# from stdin is a byte the server never sees. (§3.3, R-2, M-1.)
#
# Env overrides: CPG_MCP_IMAGE_REPO (default cpg-mcp), CPG_MCP_NO_PULL=1
```

Body:

1. Resolve `HERE` from `${BASH_SOURCE[0]}` (same idiom as `run.sh`/`setup.sh`, so the working
   directory never matters); `source "$HERE/image-tag.sh"`; `cpg_mcp_image_tag`.
2. **Preflight:** `docker` on `PATH` and `docker info >/dev/null 2>&1`, with a curated message on
   failure naming the host-venv fallback.
3. **`--verify-inputs`** (also run implicitly before every build): extract the source operand of
   every `COPY` line in the Dockerfile and assert each is covered. **Two rules, one per operand kind
   (v3, M-4):** a **file** operand is covered iff its exact relative path is in
   `cpg_mcp_input_files`; a **directory** operand is covered iff it appears in
   `cpg_mcp_input_dirs`, i.e. iff the enumeration *walks* it with the `.dockerignore` exclusions.
   Fail loudly on a gap, naming the missing operand and telling the editor which list to add it to.
   *This is the deterministic enforcement of R-11's coupling — a comment asking the next editor to
   remember would not hold.*
4. **The early exit (m-12).** Unless `--no-cache` was given, if **every** target tag for this hash
   already exists (`cpg-mcp:<hash>`, and `cpg-mcp:test-<hash>` too unless `--runtime-only`), print
   the hit to stderr and **exit 0 without pulling or building**. This is what keeps the *per
   invocation* registry dependency out of `build.sh` as well as out of the launch path: a warm
   `build.sh` is then a pure local `docker image inspect`, not a `docker pull` plus a ~0.3 s no-op
   build. `--no-cache` is the documented way to force a rebuild of an existing tag (and is what the
   base-refresh recipe in §3.6 uses).
5. **Pull the base into the image store** unless `CPG_MCP_NO_PULL=1` — reached only when step 4 did
   *not* early-exit, i.e. only when something will actually be built:
   `docker pull -q python:3.12-slim >&2`. Not cosmetic — per E-18 this is what makes the subsequent
   build resolve `FROM` metadata **locally instead of over the network**. Failure here is a
   *warning*, not an error: if the cache already suffices the build will still work offline.
6. `docker build --target runtime -t "$REPO:$CPG_MCP_TAG" -t "$REPO:dev" "$HERE" >&2`, and unless
   `--runtime-only`, `docker build --target test -t "$REPO:test-$CPG_MCP_TAG" -t "$REPO:test"
   "$HERE" >&2`. Both targets carry the **same** `<hash>` (§3.6) — deliberate slight
   over-invalidation (a `tests/` edit rebuilds the runtime tag too, ~0.3 s) bought in exchange for
   making the m-7 staleness question un-askable.
7. Print, **to stderr**, the resulting tags and the next commands.

### 4.5 `cpg/mcp/docker-run.sh` — new, executable

The launch surface named by `.mcp.json`. The container-path twin of `run.sh`.

```bash
#!/usr/bin/env bash
set -euo pipefail
# docker-run.sh — launch the `cpg` MCP server in a container, speaking MCP over stdio.
#
# This is the path the repo-root .mcp.json names. run.sh is the Docker-less variant
# (host venv) and stays the fallback + the test loop. Like run.sh it resolves
# everything from its own location, so the harness's working directory does not
# matter. (v3, m-14: "the only path that appears in a harness config" is run.sh:6's
# sentence and it is what this change makes false — neither file may claim it now.)
#
# TWO INVARIANTS, of equal weight:
#  1. NOTHING here may WRITE to stdout — the stdio transport owns it. Every diagnostic
#     goes to stderr explicitly. (docker itself is stdout-clean: its progress and status
#     go to stderr; verified 2026-07-25, E-9.)
#  2. NOTHING before the final `exec` may READ from stdin — from the moment this process
#     is spawned, stdin IS the MCP pipe and Claude Code has already written `initialize`
#     into it. A byte consumed here is a byte the server never sees, and the failure is
#     a hung or malformed handshake with no useful diagnostic. `docker build` with a path
#     context does not read stdin today (E-22), so this is a latent hazard, not a live
#     bug — which is exactly why it is guarded structurally rather than trusted.
#     Every helper that FORKS is invoked with `</dev/null`, and build.sh closes stdin
#     itself too. The one call without the redirect is `cpg_mcp_image_tag`, a sourced
#     shell function that forks nothing and reads no stdin — named here so the
#     invariant reads as absolute rather than as an unexplained exception (n-5).
#
# Env (all optional): FALKORDB_HOST/_PORT, CPG_MCP_MAX_ROWS/_MAX_CELL/_MAX_CHARS/
# _TIMEOUT_MS — forwarded into the container if set, else the image defaults apply.
#   CPG_MCP_IMAGE=<ref>       run this exact image, bypassing the hash gate entirely
#   CPG_MCP_NO_AUTOBUILD=1    never build here; fail with "run cpg/mcp/build.sh" (§3.3)
# See also MCP_TIMEOUT (Claude Code, default 30000 ms) if a first cold build needs more
# startup headroom than the default budget allows.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- preflight, HERE and not only in build.sh: it must also cover the no-autobuild
# --- path, where build.sh never runs at all (m-5).
if ! command -v docker >/dev/null 2>&1; then
  echo "cpg/mcp/docker-run.sh: docker not on PATH. Fall back to cpg/mcp/run.sh (host venv)." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1 </dev/null; then
  echo "cpg/mcp/docker-run.sh: Docker daemon not reachable (check 'docker context ls'). Fall back to cpg/mcp/run.sh." >&2
  exit 1
fi

# --- resolve the image: content hash unless the caller pinned one (§3.3, §3.6) ---
if [ -n "${CPG_MCP_IMAGE:-}" ]; then
  IMAGE="$CPG_MCP_IMAGE"
else
  source "$HERE/image-tag.sh"
  cpg_mcp_image_tag                      # sets CPG_MCP_TAG; writes nothing to stdout
  IMAGE="${CPG_MCP_IMAGE_REPO:-cpg-mcp}:$CPG_MCP_TAG"
fi

# --- the staleness gate. The tag IS the content of the build inputs, so an existing
# --- image is a proof, not a heuristic; a miss means something changed (or nothing was
# --- ever built). ~0.05 s, purely local — no registry contact. See §3.3.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1 </dev/null; then
  if [ "${CPG_MCP_NO_AUTOBUILD:-0}" = "1" ]; then
    echo "cpg/mcp/docker-run.sh: image $IMAGE not built and CPG_MCP_NO_AUTOBUILD=1. Run: cpg/mcp/build.sh" >&2
    exit 1
  fi
  echo "cpg/mcp/docker-run.sh: $IMAGE not present — building (first run after a change may take ~15 s)." >&2
  # m-13: a failed build is the ONE launch failure v2 left with no curated line, and
  # per R-8 it is the most likely one on a fresh clone or an offline machine. Without
  # this branch, `set -e` exits on raw BuildKit output with no hint that run.sh exists.
  if ! "$HERE/build.sh" --runtime-only >&2 </dev/null; then
    echo "cpg/mcp/docker-run.sh: build of $IMAGE FAILED (BuildKit output above)." >&2
    echo "  Offline? A build needs the network unless python:3.12-slim is already in the local image store." >&2
    echo "  Fall back to the host venv: cpg/mcp/run.sh (see cpg/mcp/README.md), or retry with more" >&2
    echo "  startup budget: MCP_TIMEOUT=60000 claude" >&2
    exit 1
  fi
fi

# --add-host is what makes host.docker.internal resolve on a Linux engine (§3.1).
# --init: PID-1 python IGNORES SIGTERM; tini forwards it. Measured — see §3.10.
# --label: the only handle for finding a leaked container (§3.10, V-11).
# --pull=never: this image is local-only, so a missing tag must say "No such image"
#   rather than docker's misleading "pull access denied" (m-5, E-27).
# The bare `-e VAR` form forwards a variable only if it is set in this environment;
# otherwise the image's own ENV default applies.
exec docker run -i --rm --init \
  --label cpg-mcp=1 \
  --pull=never \
  --add-host=host.docker.internal:host-gateway \
  -e FALKORDB_HOST -e FALKORDB_PORT \
  -e CPG_MCP_MAX_ROWS -e CPG_MCP_MAX_CELL -e CPG_MCP_MAX_CHARS -e CPG_MCP_TIMEOUT_MS \
  "$IMAGE"
```

#### §4.5 addendum — three corrections found during implementation (v3)

The shipped `cpg/mcp/docker-run.sh` differs from the block above in three ways. All three are
defects the spec would have shipped, found by running the verification plan rather than by reading:

1. **Env forwarding must be conditional — the bare `-e VAR` form is wrong here.** The block above
   ends with `-e FALKORDB_HOST -e FALKORDB_PORT …` and the comment *"forwards a variable only if it
   is set in this environment; otherwise the image's own ENV default applies."* **The second half is
   false.** Measured on Docker 29.6.1 against the shipped image:

   | Invocation | `FALKORDB_HOST` inside the container |
   |---|---|
   | no `-e` flag | `host.docker.internal` (image `ENV`) |
   | `-e FALKORDB_HOST`, **unset** in the caller env | **`None` — the image `ENV` is deleted** |
   | `-e FALKORDB_HOST`, set | the caller's value |

   Docker's own wording is literal: *"the variable is unset in the container"* — unset, not
   "defaulted". So the unconditional pass-through **defeated the very image default §3.1 added**, and
   anyone running `docker-run.sh` by hand without exporting `FALKORDB_HOST` got `server.py`'s
   `127.0.0.1` fallback, i.e. the container talking to itself — the exact failure §3.1 exists to
   prevent. *(Note this also corrects the review's §10 row, which marked the claim ✅ while quoting
   the sentence that contradicts it.)* The script now builds the argument list conditionally
   (`[ -n "${!v+set}" ]`), so a set-but-empty value is still honoured as the caller's choice.
   Verified all three ways, including that an explicitly wrong value is *not* silently overridden.

2. **`CPG_MCP_IMAGE` must skip the autobuild branch.** §3.6 says a pinned image *"bypasses the hash
   gate entirely"*, but the block above only bypasses the tag *computation* — a pinned-but-absent
   image still fell through to `build.sh`, which built the **content-hash** tag (never the pinned
   one) and then failed on `docker run` with docker's bare `No such image` and no curated line.
   Measured before the fix: `rc=125`, plus a confusing "already built — nothing to build" from
   `build.sh` about a *different* tag. The script now short-circuits on a pinned miss with a curated
   message saying that nothing will be built for it and naming the three ways out.

3. **`--read-only --tmpfs /tmp` is adopted** (§3.7's V-8 probe came back green). Per **n-4** these
   flags live on the launch path, so **V-6 and V-7 were re-run** after adopting them, and again after
   correction 2. Adoption was gated on more than V-8's happy path: every tool-body branch was
   exercised under the flags — a real query, `EXPLAIN`, unknown-graph, the `PROFILE` refusal and
   invalid Cypher — with no filesystem error, because §3.7's own argument is that a write path
   discovered later fails at *session start*, the worst place. The README names them as the first
   thing to drop if that ever happens.

### 4.6 `.mcp.json` — edited (two lines)

```json
{
  "mcpServers": {
    "cpg": {
      "command": "bash",
      "args": ["-c", "exec \"$CLAUDE_PROJECT_DIR/cpg/mcp/docker-run.sh\""],
      "env": {
        "FALKORDB_HOST": "host.docker.internal",
        "FALKORDB_PORT": "6379"
      },
      "timeout": 60000
    }
  }
}
```

Only `run.sh` → `docker-run.sh` and `127.0.0.1` → `host.docker.internal` change. **The
`bash -c` + unbraced `$CLAUDE_PROJECT_DIR` shape is preserved verbatim** — Claude Code expands only
`${VAR}` and `${VAR:-default}`, so the unbraced form passes through for bash to expand, which is
what keeps an absolute home path out of a tracked file and `audit-team.sh` check 7 clean. Writing
`${CLAUDE_PROJECT_DIR}` here would break it. `"timeout": 60000` and `enabledMcpjsonServers` in
`.claude/settings.json` are unchanged — and note (E-17) that this `timeout` is the **per-tool-call**
wall; the startup budget is `MCP_TIMEOUT`, which is an environment variable and is **not** set
here, so the 30 s default applies.

No `docker run` flag appears in this file — all of it lives in the wrapper (§3.3).

---

## 5. Implementation sequence

| # | Step | Done when |
|---|---|---|
| **S1** | Write `cpg/mcp/Dockerfile` + `cpg/mcp/.dockerignore` (§4.1, §4.2). | Files exist; `docker build --target runtime -t cpg-mcp:scratch cpg/mcp` succeeds by hand. |
| **S2** | Write `cpg/mcp/image-tag.sh` (§4.3). | Sourcing it and calling `cpg_mcp_image_tag` sets a stable 12-char `CPG_MCP_TAG`, prints **nothing** on stdout, and returns non-zero if an input is missing. Value is identical from three different working directories. |
| **S3** | Write `cpg/mcp/build.sh` (§4.4), `chmod +x`. | V-1, V-1b, V-2b pass; `--help` works; `--verify-inputs` passes and *fails* when a `COPY` is added without updating the input list (test it deliberately). Exits 0 from three different working directories. |
| **S4** | Write `cpg/mcp/docker-run.sh` (§4.5), `chmod +x`. | V-3…V-7 pass. **Do not touch `.mcp.json` yet** — the wrapper must be proven out-of-band first. |
| **S5** | Verification pass V-1…V-9 plus **V-11's out-of-band pre-check** (§6). Record the V-2 connect measurement. | All green; the §3.3 gate is confirmed, not assumed. |
| **S6** | Edit `.mcp.json` (§4.6). Grep the new **untracked** files with the full check-7 identifier set (V-9) — check 7 is blind to them (C-309b). | Diff is exactly two lines; grep clean. |
| **S7** | Restart the Claude Code session. In-session proof: **V-10**, and the one-command startup-budget check (§9's residual / reviewer open question 1). Then **V-11a**, and **V-11b once every session is closed** — both from a plain shell, never from inside a session (M-5). | `/mcp` shows `cpg` connected with **1** tool; a real `mcp__cpg__query` call returns rows; `Up` count equals the open-session count with nothing `Exited`, and the filter is empty once all sessions are closed. |
| **S8** | Documentation: `cpg/mcp/README.md`, root `AGENTS.md`, `docs/HISTORY.md`, `docs/BACKLOG.md` (§9). | Every row in §9 addressed in the same change. |

S1–S5 are non-destructive and need no approval. S6–S7 change the live tool wiring for the
stakeholder's own sessions — land them together, and never leave S6 committed without S7 verified.

---

## 6. Verification plan

**The constraint that shapes this plan:** a `.mcp.json` edit takes effect **only after a Claude
Code session restart**. So every step except the last two is deliberately **out-of-band** — it
drives the container directly rather than through the harness.

**V-1 — build.** `cpg/mcp/build.sh` → exit 0; `docker image ls --filter label=cpg-mcp=1` lists
`cpg-mcp:<hash>`, `cpg-mcp:test-<hash>`, `cpg-mcp:dev`, `cpg-mcp:test`, with the hash tags and
their aliases sharing an image ID.

**V-1b — the gate is idempotent and the hash is stable.** Re-run `cpg/mcp/build.sh`: same
`<hash>`, no new image ID. Then `touch cpg/mcp/server.py` (mtime only, contents unchanged) and
re-run: **the hash must not change** — that is the property an mtime heuristic would have failed
(§3.3). Then append a comment line to `server.py` and re-run: the hash **must** change and a new
image must appear.

**V-2 — end-to-end connect cost against the real startup budget (replaces v1's invented ~2 s).**
Time the full wrapper through a real handshake, not just the build:

```bash
time ( printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":4,"method":"ping"}' \
  | cpg/mcp/docker-run.sh 2>/dev/null >/dev/null )
```

**Done-condition:** median of ≥ 5 runs is **under 25 % of the effective startup budget**, i.e.
**< 7.5 s** at `MCP_TIMEOUT`'s 30 s default (E-17). Reference measurement on this box: **1.36 s
(4.5 %)**, an ~18× margin. Record the number in the README.

**If it exceeds 25 %:** do *not* reach for a different staleness mechanism — the gate is already
the cheapest option measured (E-21), so a blown budget means the *container start itself* is slow
on that host, and the remedies are, in order: (1) confirm the base image is in the image store
(`docker images python`) and re-run `build.sh`; (2) raise the budget — `MCP_TIMEOUT=60000 claude`;
(3) fall back to the host venv path (§3.5), which measured 0.37 s. Record which applied.

**V-2b — offline behaviour (closes m-2, and the reason §3.3 changed).**
Two parts, both required:

1. **The launch path must be fully offline.** With the image already built, disconnect the network
   (or run in a network-namespaced shell) and run V-2's pipeline. **Expect success** — the gate is
   `docker image inspect`, a local daemon call (E-21). *This is the property v1 did not have.*
2. **OPTIONAL — record what the build path does offline. Touches shared state; do not run without
   approval.** *(v3, m-17 + m-12.)* The command is `cpg/mcp/build.sh --runtime-only --no-cache`
   with the network down, after `docker rmi python:3.12-slim` — expect failure at
   `[internal] load metadata`; then `docker pull python:3.12-slim`, reconnect, repeat offline:
   expect success. **Two corrections to v2's wording:** the flag is `--no-cache` (`--no-cache=false`
   is not a flag this script defines — it would simply error out mid-verification); and
   `docker rmi python:3.12-slim` is a **shared-state mutation** of exactly the kind §2 discloses for
   the `docker builder prune` — `falkor-chat/Dockerfile:10` is `FROM python:3.12-slim`, so removing
   it makes the next `falkorchat:dev` build re-pull ~45 MB and **fail outright on an offline
   machine**. If it is run anyway, `docker pull python:3.12-slim` is a **mandatory closing step**,
   not an incidental one.
   **This part is optional and was deliberately not run:** E-18 already established the behaviour
   under controlled before/after conditions, and part 1 — the *launch path* works offline — is the
   property that actually needs proving. Skipping it costs nothing but a re-confirmation.

**V-3 — the image's own suite (offline).** *Precondition (m-7): run `cpg/mcp/build.sh` — both
targets — immediately before.*
```bash
docker run --rm cpg-mcp:test-<hash> python -m pytest tests -q
```
Expect the **same counts** as the host baseline E-12: `53 passed, 7 deselected`. With §4.1's
`chown` this is also byte-identical, with **no `PytestCacheWarning`** — if that warning appears,
the `chown` is missing (m-1, E-24), not a defect in the suite.

**V-4 — the image's own suite (live) — this is the networking proof.** *Same m-7 precondition.*
```bash
docker run --rm --add-host=host.docker.internal:host-gateway cpg-mcp:test-<hash> \
  python -m pytest tests -q -m live
redis-cli -h 127.0.0.1 -p 6379 GRAPH.LIST      # done-condition, see below
```
Expect `7 passed, 53 deselected`. Per E-13 these tests create *and delete* a scratch graph, so a
green run proves a fully working read+write connection from inside the container — not merely a TCP
handshake. Requires `falkordb-dev` up.

> **Do not run V-4 concurrently with another V-4, and check for residue.** Inside a container
> `os.getpid()` is **1** (E-26), so the scratch graph name collapses to the constant
> `_cpg_mcp_selftest_1` on the **shared** FalkorDB. Two simultaneous runs would corrupt each
> other. **Done-condition:** `GRAPH.LIST` afterwards shows **no `_cpg_mcp_selftest_*`** graph —
> today's list is `cpg_falkorchat`, `ws:test`, `ws:acme`, `cpg_salesperson`, `reference`. If one
> is left behind, an earlier run was interrupted; removing it is a **write to the shared
> database** and therefore needs explicit approval. Backlog C-321 fixes the root cause.

**V-5 — a real CPG query through the tool body, from inside the container.**
```bash
docker run --rm --add-host=host.docker.internal:host-gateway cpg-mcp:<hash> \
  python -c "import server; print(server.run_query('cpg_falkorchat','MATCH (m:METHOD) RETURN count(m) AS n'))"
```
Expect the stats line (`graph=cpg_falkorchat · rows=1 · N.Nms`), the header `n`, and a non-zero
count. This is the container twin of the README's existing debug recipe; it overrides the image's
`CMD`, so no MCP plumbing is involved.

**V-6 — the full MCP handshake over `docker run -i` (what V-5 does *not* prove).**
```bash
printf '%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
 '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
 '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"query","arguments":{"graph":"cpg_falkorchat","cypher":"MATCH (m:METHOD) RETURN count(m) AS n"}}}' \
 '{"jsonrpc":"2.0","id":4,"method":"ping"}' \
 | FALKORDB_HOST=host.docker.internal cpg/mcp/docker-run.sh 2>/tmp/cpg-mcp.err
```
**Expect responses for ids 1, 2 and 3 on stdout** — and **id 4's reply is expected to be missing**
(see the box). `tools/list` must report exactly **one** tool named `query`; id 3 must carry the count
row.

> **The trailing `ping` is not decoration — and its own reply is the sacrifice.** A known,
> already-captured gotcha (`claude/coder/kaizen/inbox.md`): on `mcp 1.28.x`, **EOF on stdin tears the
> anyio session down before the last response flushes**, so the reply to *the last request* never
> appears — reproduced twice during the original build. The trailing `{"method":"ping"}` (id 4) is
> what makes the reply to id 3 — the one that carries real information — survive.
>
> **So the guaranteed stdout is ids 1, 2, 3 — three responses, not four** *(v3, M-6: v2 asked for
> four in three separate places, which would have failed on a perfectly healthy run for the reason
> stated right here, in the one step added to close M-1. An implementer would then either chase a
> phantom stdin-theft bug or relax the assertion and lose the guard.)* If id 4's reply *does* appear,
> that is fine too — a fourth line is not a failure, and its presence would simply mean this `mcp`
> build flushes before teardown. **Nobody should investigate its absence.** All three places that
> assert on this (V-6 here, V-6b, V-7 done-condition 2) now say exactly this, in the same words.

**V-6b — the stdin invariant (M-1).** Run V-6 **immediately after an input edit**, so the gate
misses and a real build happens *inside* the launch path with the MCP pipe on stdin:
```bash
printf '\n# touch\n' >> cpg/mcp/server.py     # force a hash miss
… V-6's pipeline …
```
**Done-condition (M-6-corrected):** the replies to **ids 1, 2 and 3** still arrive — **the
`initialize` reply (id 1) above all** — and `/tmp/cpg-mcp.err` shows the build actually ran. **Id 4's
reply is expected to be absent** (the EOF gotcha in V-6's box); its absence is not a finding. This is
the only step that exercises build-in-the-launch-path with a live protocol stream; a lost or
truncated **id-1** response is the signal that something consumed stdin (E-22 says `docker build`
does not today — this is the regression guard). Revert the touch afterwards and re-run `build.sh`.

**V-7 — stdout purity, strengthened (the sharp test for R-2).**
```bash
… V-6's pipeline … 2>/dev/null | tee /tmp/cpg-mcp.out \
  | while read -r l; do printf '%s' "$l" | python3 -m json.tool >/dev/null || echo "NON-JSON: $l"; done
grep -c '"id":1,' /tmp/cpg-mcp.out    # must be 1 — n-7: the trailing comma keeps this
                                      # from also matching "id":10+ if the probe grows
```
Two done-conditions, not one:
1. Every stdout line parses as JSON. Any non-JSON line means something (the wrapper, the build, or
   docker) leaked to stdout and **must** be redirected to stderr. Inspect `/tmp/cpg-mcp.err` to
   confirm the build chatter landed there.
2. **The `initialize` response (id 1) is present, and ids 1, 2 and 3 all are** — with **id 4's reply
   expected to be absent** for the documented EOF reason (V-6's box; v3, M-6). v1's V-7 asserted only
   that every line was JSON, which would have passed on a stream that silently *lost* the first
   request — the exact M-1 failure. Purity without completeness is not enough; completeness that
   contradicts a known gotcha is a false alarm.

**V-8 — optional hardening probe (§3.7). Result: green → ADOPTED.** Re-run V-6 with `--read-only
--tmpfs /tmp` added to the wrapper's `docker run`. If green, keep the flags; if not, drop them and
note why. **n-4: these flags live on the launch path, so a green V-8 alone is not sufficient —
re-run V-6 and V-7 too before adopting them.**

> **Widened before adopting, and why.** §3.7's own argument against these flags is that a write path
> not exercised here would fail at *session start*. A green V-6 only covers `initialize`,
> `tools/list` and one successful query — so the probe was extended to **every branch of the tool
> body**: a real query, `EXPLAIN`, an unknown graph, the `PROFILE` refusal, and invalid Cypher. All
> returned their normal curated output with no filesystem error. Then V-6 and V-7 were re-run through
> the wrapper per n-4. Adopted on that basis; the README lists the flags as the first thing to drop
> if a read-only/permission error ever appears at session start.

**V-9 — host fallback and audit, before the switch.**
```bash
cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q          # still 53 passed, 7 deselected
cpg/mcp/run.sh </dev/null                          # still fails/starts as before — unchanged
claude/scripts/audit-team.sh                       # compare to the known-red baseline (C-309a)
```
plus the **full** personal-identifier grep, mirroring all five identifiers of `audit-team.sh`
check 7 (`claude/scripts/audit-team.sh:116-137`) rather than only `$HOME` (m-4). Check 7 is
`git grep`-based and cannot see these files until they are committed, so this manual grep *is* the
gate — and covering 1 of 5 identifiers would restore only 1/5 of it:

```bash
files="cpg/mcp/Dockerfile cpg/mcp/.dockerignore cpg/mcp/image-tag.sh cpg/mcp/build.sh cpg/mcp/docker-run.sh"
grep -I -n -i -F "$HOME"                     $files
grep -I -n -i -F -w "$(id -un)"              $files
grep -I -n -i -F -w "$(hostname)"            $files
grep -I -n -i -F "$(git config user.name)"   $files
grep -I -n -i -F "$(git config user.email)"  $files
```
All five must produce **no hits** (`-i`/`-F`, and `-w` on the short bare tokens, exactly as check 7
does). `audit-team.sh` is **already** `RESULT: FAIL` from pre-existing leaks (C-309a), so its
done-condition is **no *new* failures** — a before/after diff, the same convention M3 used.

**V-10 — in-session, after S6 + a session restart.** `/mcp` lists `cpg` as connected with **1**
tool; then a genuine `mcp__cpg__query(graph="cpg_falkorchat", cypher="MATCH (m:METHOD) RETURN
count(m) AS n")` returns the same count as V-5. This is the only end-to-end proof, and it cannot be
obtained in the session that made the edit.

**V-11 — no orphaned containers (closes M-3). Re-specified in v3 (M-5) — v2's version could not
pass, and following it literally would have killed the live session's own MCP server.**

> **Why v2's version was unexecutable.** The containerized server is a **session-lifetime**
> process: one container per open session, holding that session's stdio pipe (§3.7). §5 sequences
> V-11 *after* V-10, which can only be run **inside a live session** — so the label filter would show
> that session's own container, `Up`. v2's done-condition was "empty", which is therefore
> **false by construction on a healthy system**, and the step then said to capture and clean up
> "the survivor" — i.e. `docker stop` the very server V-10 had just proven. Stdio servers are not
> auto-reconnected (E-29), so the implementer would then need a session restart to undo it.
> E-23e's *"the label filter showed nothing left"* was measured with **no session open**, which is
> why the gap did not surface at design time.

The command is unchanged; the interpretation is what was wrong:

```bash
docker ps -a --filter label=cpg-mcp=1 --format '{{.Names}}\t{{.Status}}'
```

**Run it from a plain shell — not from inside a Claude Code session** — and read it against the
number of open sessions, in two passes:

| Pass | Precondition | Done-condition |
|---|---|---|
| **V-11a** | **N** Claude Code sessions open (N ≥ 1), after at least three session restarts | Exactly **N** entries in `Up`, and **zero** in `Exited`/`Created`. The `Up` entries are the live servers, not orphans. |
| **V-11b** | **Every** Claude Code session closed | **Empty.** This is the real orphan check. |

**Identifying an actual orphan:** an entry in `Exited`/`Created` (the `--rm` reap failed), or an `Up`
count that **exceeds** the number of open sessions. Only then capture
`docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' <name>` and treat cleanup (`docker stop`
then `docker rm`) as a **manual, approval-gated** act — never automated into the launch path (§3.10).

> **Never `docker stop` a labelled container while any session is open.** There is no way to tell
> from the outside which session owns which container, so stopping the wrong one silently removes a
> live agent's only CPG read path until that session is restarted.

**Out-of-band pre-check, runnable now (before `.mcp.json` is switched):** with the wiring still on
`run.sh`, no session owns a container, so V-11b's condition can be checked immediately after the
V-3…V-8 probes — every one of them uses `--rm`, so the filter must already be empty. That closes the
*reaping* half of the question without waiting on a restart; V-11a/V-11b then close the
*session-lifetime* half.

---

## 7. Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| **R-1** | Image missing or stale at session start → the `cpg` tools do not appear. | **Medium** *(was High; E-17/E-29 reduce it)* | §3.3's content-hash gate makes "stale" impossible to miss and builds on a miss. Startup is **non-blocking** (E-17), so a slow connect delays tool availability rather than stalling the session, and a failed connect is **reported to the model** (E-29) rather than silent. Residual: a *cold* build on a slow link may exceed the 30 s `MCP_TIMEOUT`; then `build.sh` is the documented fresh-clone step, an interrupted build resumes (E-28), and `MCP_TIMEOUT=60000 claude` raises the budget. Stdio servers are not auto-reconnected — recover via `/mcp` reconnect or a restart. |
| **R-2** | Anything written to **stdout** by the wrapper, `build.sh`, or docker corrupts the MCP stream. | High | E-9/E-10 verified docker itself is stdout-clean. The wrapper and `build.sh` redirect all output to stderr explicitly; `image-tag.sh` assigns a variable rather than printing. V-7 tests both purity *and* completeness. |
| **R-2b** | **Anything that *reads* stdin before `exec` steals bytes from the MCP handshake** — the exact twin of R-2, and the failure is a hung or malformed handshake with no useful diagnostic. | High | M-1. `docker build` with a path context does not read stdin today (E-22), so this is latent — and guarded structurally rather than trusted: `build.sh` does `exec </dev/null` as its second line, and every helper invocation in `docker-run.sh` carries `</dev/null`. V-6b exercises a real build inside the launch path with a live protocol stream and asserts the `initialize` reply survives. |
| **R-3** | Wrong Docker context (E-2): a session under `desktop-linux` would reach a different engine where `falkordb-dev` does not exist. | Medium | `--add-host host-gateway` still works there, so the failure surfaces as the server's own curated `FalkorDB unreachable at host.docker.internal:6379` message, not a crash. **`docker-run.sh` itself** preflights `docker info` (m-5 — v1 wrongly attributed this to the wrapper while only `build.sh` had it, leaving the `CPG_MCP_NO_AUTOBUILD=1` path uncovered). Document the symptom in the README. |
| **R-4** | FalkorDB started on a non-default port (`FALKORDB_PORT=6380`) leaves `.mcp.json`'s `6379` wrong. | Low | Pre-existing exposure, unchanged by this work. Documented, not fixed. |
| **R-5** | The container path and the host venv path drift (two ways to run the same code). | Medium | Both install from the same `requirements*.txt`; §3.4's in-container gate (V-3/V-4) is the control, with the m-7 precondition making a stale gate image structurally impossible (shared content hash). |
| **R-6** | The container path depends on FalkorDB publishing 6379 **on `0.0.0.0`, not just "to the host"**. | Low–Medium | *(Sharpened per m-9.)* `start_falkordb.sh:52-58` uses `-p "${FALKORDB_PORT}:6379"` → `HostIp:""` → all interfaces (confirmed via `docker inspect`, E-3). The venv path works either way; **the container path breaks the moment anyone hardens that to `-p 127.0.0.1:6379:6379`** — an obvious future security tidy-up. Second-order: a host firewall that **DROP**s traffic from `docker0` (rather than rejecting it) turns a fast `ECONNREFUSED` into a hang that only the 60 s tool timeout ends, so the curated "FalkorDB unreachable" message never appears. README troubleshooting gains: *if FalkorDB is ever published to loopback only, the container path needs `--network host` or a user-defined network (§3.1 options B/D).* If FalkorDB ever moves to a Compose-internal-only network, revisit §3.1 option D. |
| **R-7** | ~1 s of extra connect latency per session vs the venv path (E-20: 1.36 s vs 0.37 s). | Low | Once per session, not per query, and 4.5 % of the 30 s budget (E-17). Accepted; V-2 bounds it against the real budget. |
| **R-8** | **A build needs network access** for the base-image metadata resolution and `pip`. | Low–Medium | *(Corrected — v1 claimed "warm builds are offline", which E-18 disproves.)* The **launch path** is now fully offline (the gate is `docker image inspect`, E-21) and V-2b part 1 proves it. A **build** needs the network unless the base image is in the local *image store* — which is why `build.sh` explicitly `docker pull`s it (§4.4 step 4), since a BuildKit build alone does not (E-14 amended). Once pulled, a rebuild resolves `FROM` locally (E-18, measured 0.5 s → 0.0 s). V-2b part 2 records the offline behaviour of both states. |
| **R-9** | **A leaked (orphaned) container accumulates** on an engine that also runs `falkordb-dev`. | Low | *(New — M-3.)* Measured (E-23): the happy path and CLI-`SIGKILL` path both reap cleanly; the only orphan path is a **wedged** server that is not reading stdin plus a `SIGKILL` that skips `SIGTERM`. `--init` closes that at the `SIGTERM` step (without it, PID-1 python ignores `SIGTERM` outright). `--label cpg-mcp=1` makes any survivor findable, and **V-11** turns "no orphans" from an assumption into a checked property. Cleanup stays manual and approval-gated. |
| **R-10** | **Concurrent sessions racing on the image.** Two Claude Code sessions (repo root plus a component subdirectory — which this repo encourages) start at once. | Low | *(New — m-8.)* The content-hash tag **removes** this race rather than tolerating it: same tree → same immutable tag, so at most one builds and a duplicate is idempotent; different trees → different tags, so neither can clobber the other. Under v1's mutable `:dev` tag, session B's connect would have waited on session A's build and the tag would have pointed at whichever finished last. The moving `:dev`/`:test` aliases can still flap, which is why the launch path never names them (§3.6). |
| **R-11** | **The hash input list drifts from what the Dockerfile `COPY`s** — a new `COPY`ed file not in the list would be invisible to the gate, silently reintroducing the staleness bug §3.3 exists to prevent. | Medium | *(New — the one cost of the content-hash choice, named rather than hoped away.)* The list lives in exactly one place (`image-tag.sh`), and `build.sh --verify-inputs` extracts every `COPY` source from the Dockerfile and fails on a gap — run implicitly before every build, and S3's done-condition requires *demonstrating* that it fails when a `COPY` is added without the list. A missing input file is also a hard error, never a silently-skipped hash component. Deterministic enforcement, not a comment. |

---

## 8. Rollback

**Two lines, no cleanup.** *(n-2: v1 said "one line" and then described two edits; §4.6 was right.)*
Revert `.mcp.json`'s `args` to `["-c", "exec \"$CLAUDE_PROJECT_DIR/cpg/mcp/run.sh\""]` and
`FALKORDB_HOST` to `127.0.0.1`, then restart the session. The host venv, `setup.sh` and `run.sh` are
untouched by this change (§3.5) — that is the point of retaining them, and it makes rollback cost
nothing but a restart.

Optional tidy-up afterwards: `docker image ls --filter label=cpg-mcp=1` to see what was built, then
remove those tags. Not required; the images are inert. **Do not** touch `falkordb-dev` or the
`falkordb-data` volume at any point in this work — it is shared with `falkor-chat` and
`salesperson`, and nothing in this design has any reason to.

---

## 9. Documentation impact

Every row lands in the same change as the code.

| Doc | Change |
|---|---|
| **`cpg/mcp/README.md`** | Substantial. Quick start gains `cpg/mcp/build.sh` **as the supported fresh-clone step** (§3.3 — not as an optional nicety). New **"Running in a container"** section: the images and the **content-hash tag scheme** (§3.6), the wrapper, `--add-host`/`host.docker.internal` and *why* (§3.1), the `docker image inspect` gate and what a miss does, `CPG_MCP_NO_AUTOBUILD`, `CPG_MCP_IMAGE`, and **`MCP_TIMEOUT` (default 30000 ms) as the startup-budget escape hatch** — explicitly distinguished from the per-tool-call `"timeout": 60000` this file already documents at line 211. Record the **measured** V-2 connect time and the V-2b offline results. Update the `.mcp.json` block quoted at lines 189–200 and the env-var table's `FALKORDB_HOST` default/meaning. **"Running and debugging"** must now describe **two** launch surfaces and say which is the default; add the container debug recipe (V-5) beside the existing venv one. **"When the tool is unavailable"** gains the host-venv fallback *above* `redis-cli`. **Troubleshooting** gains the R-6 loopback/firewall note (m-9) and the R-3 wrong-context symptom. **"Wiring it elsewhere"** gains the §3.9 note **and — m-10 — line 269's `claude mcp add --scope local cpg -- <repo-root>/cpg/mcp/run.sh` must become `docker-run.sh`**, with a one-line note that `run.sh` is the Docker-less variant: that recipe exists for `$CLAUDE_PROJECT_DIR` expansion failure, which is orthogonal to venv-vs-container, so leaving it would quietly wire the local scope to a *different* launch path than the project scope. Add a housekeeping line for `docker image ls --filter label=cpg-mcp=1` (§3.6), **and the base-refresh line `docker pull python:3.12-slim && cpg/mcp/build.sh --no-cache` with the one-sentence reason (m-11): the base image sits outside the content hash, so a hash hit will serve an image built on an old base indefinitely.** |
| **`cpg/mcp/run.sh`** | *(v3, m-14 — header comment only; nothing executable changes, so §3.5's "retained, unchanged" stays true in substance.)* Line 6 says *"This is the only path that appears in a harness config (.mcp.json)"*, which **this change makes false** — and `run.sh` is the file a reader lands on when rollback (§8) points them at it. Reword to: the Docker-less variant; `.mcp.json` names `docker-run.sh`. Same class of gap as m-10. |
| **root `AGENTS.md`** | Two places. The `cpg/` structure bullet (**lines 20–24** — n-6: v2 cited "line ~30") — say the MCP server runs containerized, host venv retained for tests and fallback. The **"Key commands"** `cpg/mcp` block (lines 124–127) — its preamble says *"after `cpg/mcp/setup.sh`"*; add `cpg/mcp/build.sh` and the in-container test command. The two existing pytest lines stay **exactly as they are** (§3.4). |
| **`docs/HISTORY.md`** | Append one dated entry (most-recent-first, per the file's convention): what changed, the networking decision and its rejected alternatives in one line, **the content-hash launch gate and the E-18 registry-round-trip measurement that motivated it**, the retained host path, and the measured V-2/V-3/V-4 results. |
| **`docs/BACKLOG.md`** | Add **`C-320` — containerize the `cpg` MCP server** under *Follow-ups (post-M3)*, and **`C-321` — make the live suite's scratch-graph name unique inside a container** (`uuid4().hex[:8]` instead of `os.getpid()` at `cpg/mcp/tests/test_server.py:472`; owner `tdd-engineer`/`coder`; rationale E-26/m-6). `C-319` is the current maximum; **do not renumber anything**. Mark C-320 ✅ when delivered. Add one cross-reference sentence to **C-310** noting the launch command is now `docker-run.sh`, that the port-the-command property is preserved (§3.9), and that each harness's own startup budget must be established there — do not renumber or rescope C-310. |
| **`skills/cpg-analysis/SKILL.md`** | **No change required — re-verified by the reviewer independently.** `grep -n "run.sh\|setup.sh\|venv\|docker"` finds nothing; it names only the tool `mcp__cpg__query` and the `redis-cli` fallback, and neither changes. Re-check if the tool name or output format is ever touched. |
| **`claude/devops/kaizen/inbox.md`** | Entries filed by the v2 design run: the E-18 warm-build registry round trip and the image-store-vs-build-cache distinction; the E-23 PID-1 `SIGTERM`/`--init` measurements; and (post-implementation) the V-6 EOF/trailing-`ping` gotcha **if** it reproduces through the container layer (already captured in `claude/coder/kaizen/inbox.md`, so only a container-specific difference is worth filing). |
| **`docs/plans/cpg-query-access.md`** | No edit. It is the M3 design at approved status; this note supersedes only its §4.2 *"Runtime & launch"* choice, and says so here rather than rewriting a closed plan. |
| **`docs/reviews/cpg-mcp-containerization.md`** | No edit — a review is a record. §11 below is this plan's response to it. |

---

## 10. Open questions for the stakeholder

None blocking. Two of v1's three are now closed by measurement; what remains is one genuine
preference and one policy call the reviewer raised.

1. ~~**Does the build-on-every-launch cost fit the startup budget?**~~ **Closed.** The budget is
   30 s (`MCP_TIMEOUT`, E-17) and the launch path no longer builds at all; measured connect is
   1.36 s, 4.5 % of budget (E-20). §3.3.
2. ~~**Is a warm build offline-safe?**~~ **Closed, and the answer was no** (E-18) — which is why
   §3.3 changed mechanism. The launch path is now offline-safe by construction.
3. **§3.5 keeps two launch paths.** Deliberate (test loop + fallback + free rollback), but it is
   two things to keep in sync, and that drift cost is permanent. The alternative is container-only,
   which trades it for a single-point-of-failure at session start on a component whose recovery
   needs a session restart. **Recommendation: keep both**, and revisit only if the container path
   runs for a few weeks without a fallback ever being needed.
4. **Is a leftover container a defect or acceptable residue?** (Reviewer's open question 3.) This
   plan treats it as a **defect**, but cheaply: `--init` is justified on its own merits (PID-1
   `python` ignoring `SIGTERM` is simply wrong, E-23a) and the only added cost of the position is
   **V-11**, one `docker ps` command. **Recommendation: keep V-11** — it is the difference between
   knowing and assuming, and E-23 already showed one path where the assumption fails.

---

## 11. Review response

Map from `docs/reviews/cpg-mcp-containerization.md` to this v2. **Nothing is silently dropped**;
where a suggestion was not taken as written, the reason is stated.

| ID | Disposition | What changed |
|---|---|---|
| **M-1** — stdin unguarded on the pre-launch build | **Accepted in full, and widened** | The stdin invariant is now stated with the same weight as stdout, in the same places stdout is policed: §4.5's header carries **both** invariants as a numbered pair; §4.4's `build.sh` closes stdin structurally with `exec </dev/null` as its second line (stronger than the suggested per-call redirect, which a future caller could forget); every helper call in `docker-run.sh` also carries `</dev/null`. New risk row **R-2b** is the explicit twin of R-2. Verification strengthened two ways: **V-7** now asserts **completeness** (`initialize`/id-1 present, all four ids) not merely that each line is JSON — exactly the hole the review identified; and new **V-6b** forces a hash miss so a *real build runs inside the launch path with a live protocol stream*, which is the only step that would actually catch a stdin-consuming build. Also **measured**: `docker build` with a path context does **not** read stdin today (**E-22**), so the plan now says "latent hazard, guarded structurally" instead of implying a live bug. |
| **M-2** — invented budget; fresh-clone benefit undeliverable; better option missing | **Accepted; §3.3 rewritten as a decision, and the mechanism changed** | **(1) The budget is now measured, not invented.** Per official docs (**E-17**): `MCP_TIMEOUT` default **30000 ms** is the startup wall; `MCP_CONNECT_TIMEOUT_MS` (5000) binds only under `MCP_CONNECTION_NONBLOCKING=0` or `alwaysLoad: true`, **neither of which `cpg` uses** — so the reviewer's "may be as low as 5 s" does not apply here; and since v2.1.142 **MCP startup is non-blocking by default**, so a slow connect delays tool availability rather than stalling the session. The review was right that `"timeout": 60000` is the per-tool-call wall. **V-2 is rewritten** to time the **end-to-end handshake** (as suggested) with the trigger set at **25 % of the measured budget (< 7.5 s)**; measured **1.36 s = 4.5 %**, an ~18× margin (**E-20**). **(2) The mechanism changed to the reviewer's M-2(d).** The decisive input is new measurement **E-18**: a warm, fully-cached `docker build` makes a **Docker Hub round trip every launch** (`load metadata`, 0.5 s = essentially the whole build) unless the base is in the local *image store*, which a BuildKit build does not populate (**E-14 amended**). So build-on-every-launch was not offline-safe at all. §3.3 now compares the two on the four axes asked for — latency, offline, concurrency, staleness — and adopts **content-hash tag + `docker image inspect`** (0.05 s, purely local, immutable per-content tags). Latency was *not* the deciding axis; offline behaviour and concurrency were. The build survives as the **miss branch**, so the design is a superset of v1's self-healing minus the per-launch network call. **(3) The fresh-clone claim is restated honestly**, as asked: cold build measured **14.15 s** on this link (**E-19**) — under budget *here*, but link-dependent; and **E-28** adds a fact the review assumed against: an interrupted build's completed layers **persist**, so convergence across restarts is monotonic, not a restart from zero. `build.sh` is named as *the supported fresh-clone path*. **(4)** `MCP_TIMEOUT` is now named in R-1 and in the §9 README row alongside `CPG_MCP_NO_AUTOBUILD=1`. **(5)** The mtime fallback is **dropped entirely** rather than held in reserve, since the content hash dominates it at equal cost — and the plan now concedes the review's point that v1's *reason* for rejecting mtime (`git checkout` fooling it "in both directions") was wrong. |
| **M-3** — lifecycle undesigned | **Accepted in full, and measured rather than reasoned** | New **§3.10 D-cycle**. Every claim probed (**E-23**): without `--init`, PID-1 `python` **ignores `SIGTERM`** (still `running` a minute later); with `--init`, `exited 143`. So `--init` is adopted as *required*, not defensive. The shutdown sequence is tabulated across four paths, which located the *actual* orphan path precisely — a **wedged** server not reading stdin plus a `SIGKILL` that skips `SIGTERM`; the happy path (stdin EOF → `exit 0` in 1.46 s) and the CLI-`SIGKILL`-with-real-server path (EOF → exit → auto-removed in < 2 s) both reap cleanly, which is *narrower* than the review assumed. `--label cpg-mcp=1` added to both the image and the run; **`--name` deliberately not added**, with reason — a fixed name would collide across the concurrent sessions this repo encourages, converting benign duplication into a hard failure at session start. New risk **R-9**, new verification **V-11** ("after 3 restarts, `docker ps -a --filter label=cpg-mcp=1` is empty"). Cleanup is explicitly kept manual and approval-gated: a script on the MCP startup path must not be able to kill a container it did not start. |
| **m-1** — V-3 "byte-identical" likely wrong | **Accepted; fixed at the environment, not the assertion** | **Measured both ways (E-24):** without the fix, `53 passed, 7 deselected, 1 warning` + `PytestCacheWarning … Permission denied`. §4.1's test stage now has `RUN chown appuser:appuser /app`, which restores a **genuinely byte-identical** clean run — so V-3 keeps the strong claim rather than softening it, and gains a diagnostic ("if that warning appears, the `chown` is missing"). The suggested `-p no:cacheprovider` was **rejected**: it changes the *invocation*, so the host and container gates would stop running the same command, defeating §3.4's purpose. §3.4 records both the finding and this reasoning. |
| **m-2** — "warm builds are offline" unverified; BuildKit is default | **Accepted — and the claim turned out to be false** | This is the finding that changed the design. **E-18** measured it: BuildKit resolves `FROM` metadata over the network on every warm build (0.5 s) unless the base is in the image store, and **E-14 amended** records that a BuildKit build does not put it there — so v1's `docker images python` check was reading the wrong store. **R-8 is rewritten** from "warm builds are offline" to a precise, correct statement. The reviewer's own prediction — "if it fails offline, that is a decisive argument for the image-inspect/content-hash variant" — is exactly what happened. **V-2b** adds the two-part offline probe (launch path must work offline; build path's behaviour recorded in both store states), and `build.sh` now explicitly `docker pull`s the base (§4.4 step 4) so a miss-triggered build is as offline-tolerant as possible. |
| **m-3** — `.dockerignore` patterns not recursive | **Accepted, verified decisively** | **E-25:** identical context and Dockerfile, only the ignore file differing — bare patterns shipped `tests/__pycache__` (count 1), `**/`-prefixed patterns did not (count 0). §4.2 now uses `**/__pycache__`, `**/*.pyc`, `**/.pytest_cache`, with the reason inline; §3.8 states the `filepath.Match` mechanism. `.dockerignore` is additionally now a **hash input** (§4.3), since it changes what ships. |
| **m-4** — V-9's PII grep narrower than check 7 | **Accepted in full** | V-9 now runs all **five** identifiers from `claude/scripts/audit-team.sh:116-137` — home path, username, hostname, git `user.name`, git `user.email` — matching check 7's flags too (`-I -n -i -F`, with `-w` on the short bare tokens). The reason is stated where the commands are: this manual grep *is* the gate, because check 7 is `git grep`-based and blind to untracked files, so 1-of-5 coverage restores 1/5 of it. S6's wording updated from "`$HOME`/username" to the full set. The new `image-tag.sh` is included in the file list. |
| **m-5** — R-3's `docker info` preflight not where the plan says | **Accepted in full** | The daemon preflight **moved into `docker-run.sh`**, before the image-resolution and autobuild branches, so it also covers the `CPG_MCP_NO_AUTOBUILD=1` path that v1 left uncovered. R-3's wording corrected to match. `--pull=never` added to the launch `docker run`; **E-27** measured the difference: `No such image: cpg-mcp:<tag>` instead of the misleading `pull access denied for cpg-mcp, repository does not exist or may require 'docker login'`. |
| **m-6** — live suite's scratch graph collapses to a constant in a container | **Accepted; noted, worked around, and filed — not fixed here** | **E-26** confirms `os.getpid()` → **`1`** in the container, so the name is the constant `_cpg_mcp_selftest_1` on the *shared* FalkorDB. §3.4 now carries this as an explicit caveat qualifying E-13's self-containment argument. **V-4** gains a "do not run concurrently" instruction and the reviewer's suggested done-condition — `GRAPH.LIST` shows no `_cpg_mcp_selftest_*` residue — plus a note that removing residue is a **write to a shared database** and therefore approval-gated. The one-line proper fix (`uuid4().hex[:8]`) is **filed as backlog C-321** for `tdd-engineer`/`coder`, and §1's out-of-scope list names the exception explicitly, agreeing with the review that it is out of this plan's remit. |
| **m-7** — auto-build only refreshes the runtime tag, so the gate image can lag | **Accepted, and made structural rather than procedural** | Stated as an explicit precondition in both places asked for: §3.4's table "When" column and V-3/V-4's headers ("run `cpg/mcp/build.sh` — both targets — immediately before"). Beyond that, §3.6 makes the **test tag share the runtime tag's content hash** (`cpg-mcp:test-<hash>`), so a stale gate image cannot be *reached* by accident — if the current hash's test tag does not exist, the documented command simply does not resolve. Deterministic enforcement beats a remembered precondition. |
| **m-8** — concurrent sessions / image-tag races unanalysed | **Accepted, and the race is removed rather than documented** | The review asked for a paragraph explaining why it is acceptable, and noted the content-hash tag would remove it entirely; since §3.3 adopts that tag, the plan takes the stronger option. **Concurrency is now one of the four decision axes in §3.3's comparison table**, and new risk **R-10** states the mechanism: same tree → same immutable tag (at most one build, duplicates idempotent); different trees → different tags (neither can clobber the other). §3.6 notes the moving `:dev`/`:test` aliases can still flap, which is precisely why the launch path never names them. |
| **m-9** — R-6 tracks the port but not the bind address | **Accepted in full** | **R-6 rewritten** to name `0.0.0.0` as the actual dependency, citing `start_falkordb.sh:52-58` (`-p "${FALKORDB_PORT}:6379"` → `HostIp:""`), and raised to Low–Medium since hardening to `-p 127.0.0.1:6379:6379` is a plausible future tidy-up that would break only the container path. The second-order DROP-vs-REJECT firewall observation is included (a hang the curated "unreachable" message never reaches). §3.1's consequences list gains the bind-address line, and the §9 README row gains the suggested troubleshooting sentence verbatim in substance. |
| **m-10** — §9 misses README line 269's `claude mcp add --scope local` recipe | **Accepted in full** | §9's README row now names line 269 explicitly as a required sub-change (`run.sh` → `docker-run.sh`, with a one-line note that `run.sh` is the Docker-less variant), with the reason: that recipe addresses `$CLAUDE_PROJECT_DIR` expansion failure, which is orthogonal to venv-vs-container, so leaving it would wire the local scope to a different launch path than the project scope. **§3.5 point 4 also corrected** — the review was right that citing that recipe as a reason to keep `run.sh` reinforced the confusion; the portability argument now stands on its own. |
| **n-1** — `useradd` after `COPY` | **Accepted** | Moved above the `COPY` lines in §4.1's `base` stage, with a one-line comment saying why. |
| **n-2** — §8 says "one line", describes two | **Accepted** | §8 now opens "**Two lines**, no cleanup", aligned with §4.6. |
| **n-3** — no `--pull=never`, no `--memory`/`--cpus` | **Accepted in part** | `--pull=never` **adopted** for the error-message reason (m-5, measured in E-27). `--memory`/`--cpus` **rejected**: the review agrees they are not needed for correctness, and this is a short-lived single-process dev tool whose resource ceiling is the interpreter plus one FalkorDB result set — a wrong limit would surface as an opaque OOM kill at session start, trading a non-problem for a bad failure mode. Revisit only if the image is ever published or run untrusted. |
| **n-4** — V-8's `--read-only` flags belong to the wrapper | **Accepted** | Stated in both places: §3.7 and V-8 now say adopting those flags changes the launch path, so **V-6 and V-7 must be re-run** and a green V-8 alone is insufficient. |
| Review **open question 1** (what is the startup budget?) | **Answered** | 30 s (`MCP_TIMEOUT` default), from official docs, with the 5 s figure correctly attributed to `MCP_CONNECT_TIMEOUT_MS`/blocking startup, which does not apply here. **E-17**, and it is in V-2's done-condition as asked. |
| Review **open question 2** (container-only eventually?) | **Left open, with a recommendation** | Genuinely a stakeholder preference, not something measurement settles. §10 item 3: keep both, revisit after the container path has run without the fallback being needed. |
| Review **open question 3** (orphan: defect or residue?) | **Answered with a recommendation** | Treated as a defect, but the position costs only V-11's single `docker ps`, and `--init` is justified independently by E-23a. §10 item 4 recommends keeping V-11 and says why. |

---

## 12. Review response — Part II (v3)

Map from [`../reviews/cpg-mcp-containerization.md`](../reviews/cpg-mcp-containerization.md) **Part II**
(verdict *approve with suggestions*, implementation authorised) to this v3. The three majors were
done-conditions that could not be executed as written; all were settled **before** the step they
gate, as the review asked.

| ID | Disposition | What changed |
|---|---|---|
| **M-4** — hash list & `--verify-inputs` under-specified for the one directory `COPY` | **Accepted in full, settled before S2** | §4.3 is now **directory-driven, not glob-driven**: a new `cpg_mcp_input_dirs` names the walked directories (`tests`), `cpg_mcp_input_files` walks each with `.dockerignore`'s exclusions, and `--verify-inputs` has **one rule per operand kind** (file → exact membership; directory → "the enumeration walks it"). §4.1's Dockerfile header and §4.2's `.dockerignore` both state the coupling. **Demonstrated, not asserted:** a `.json` fixture dropped under `tests/` moves the hash (`ba910c48571d` → `f33f9f5273c6`) and moves it back on removal — v2's `*.py` glob would not have noticed it; `__pycache__` does *not* affect the hash. `--verify-inputs` was also shown to **fail** on both operand kinds (an uncovered `COPY <file>` and an uncovered `COPY <dir>`), which is S3's done-condition. |
| **M-5** — V-11 cannot pass where §5 sequences it, and its cleanup would kill the live server | **Accepted in full, V-11 re-specified before S7** | §6's V-11 now splits by status and liveness: **V-11a** (N sessions open → exactly N `Up`, **zero** `Exited`/`Created`) and **V-11b** (all sessions closed → empty), both **run from a plain shell, never from inside a session**. An orphan is now *defined* (an `Exited`/`Created` entry, or an `Up` count exceeding the open-session count) instead of being "the survivor", and a boxed warning forbids `docker stop` on a labelled container while any session is open. A new **out-of-band pre-check** closes the *reaping* half immediately: after the V-3…V-8 probes (all `--rm`), the filter must already be empty — **it was**. §5's S7 row was rewritten to match. |
| **M-6** — V-6/V-6b/V-7 require four responses; the documented EOF gotcha guarantees three | **Accepted, and then confirmed by measurement** | All three places now say the same thing in the same words: **expect ids 1, 2, 3; id 4's reply is expected to be absent**, and its absence is explicitly *not* to be investigated. Option (a) was taken over the reviewer's preferred (b) because **the measured behaviour is worse than either assumed**: it is a *race*, not a fixed "last reply" rule. V-6 through the container dropped only id 4 — but a 6-message probe lost **two** trailing replies (ids 5 and 9), and the same 4-message probe on the **host venv** path lost the trailing reply in 2 of 5 runs versus 4 of 5 through the container. So padding with a second throwaway would not have made ids 1–4 *guaranteed*; the honest rule is **assert only on the substantive ids, never on the padding**, and that is what V-6/V-6b/V-7 and the README now say. |
| **m-11** — "immutable" covers repo bytes only | **Accepted (one of the two the review called worth taking)** | §3.3 gains a boxed statement of exactly what the hash does and does not cover (base image and pip resolution are outside it), with both consequences: same tag → possibly different images a month apart, and — the likelier one — **nothing ever refreshes an existing image's base**. §3.6's table reads "immutable **w.r.t. the tracked build inputs**", and §3.6 adds the manual refresh (`docker pull python:3.12-slim && cpg/mcp/build.sh --no-cache`), which is also in the README's housekeeping section and in `image-tag.sh`'s own header. |
| **m-13** — the miss-branch build failure is the only launch path with no curated message | **Accepted (the second one worth taking)** | §4.5 and the shipped `docker-run.sh` wrap the build in `if ! …; then` with a four-line curated message naming the offline cause, the `cpg/mcp/run.sh` fallback and `MCP_TIMEOUT=60000`. **Verified by deliberately breaking the build** (a `RUN false` appended to the Dockerfile, then reverted): the curated lines appear after the BuildKit error, exit 1. |
| **m-14** — `run.sh:6` and `docker-run.sh` would both claim to be the only harness path | **Accepted in full** | `cpg/mcp/run.sh`'s header now says it is the **Docker-less variant** and that `.mcp.json` names `docker-run.sh`; `docker-run.sh`'s header drops "the only". Nothing executable in `run.sh` changed, so §3.5's "retained, unchanged" stays true in substance. §9 gained the missing row. |
| **m-12** — `build.sh` idempotence specified two ways; V-2b invokes an undefined flag | **Accepted, with one refinement the review could not have seen** | The rule is now stated once and implemented: **if the target tags exist, do nothing — no build and no `docker pull`** — with `--no-cache` as the way to rebuild. V-2b's `--no-cache=false` is corrected to `--no-cache`. **Refinement found in testing:** a bare early exit also skips re-pointing the moving `:dev`/`:test` **aliases**, so moving between two input states could leave `cpg-mcp:dev` on an older hash and an ad-hoc `docker run cpg-mcp:dev` running stale code. The early-exit branch now issues `docker tag` (an instant local metadata op — no network, no build, so m-12's intent is intact). |
| **m-15** — E-20's table is internally inconsistent | **Accepted** | E-20, §3.3's budget table and the comparison table now report the container variants as **≈ 1.4–1.6 s, indistinguishable at n=8**, with the superset-measured-faster ordering used to *bound the noise* (≥ 0.06 s) rather than to claim a saving. "Marginal win" is gone; the decision sentence says latency is **a wash**. V-2's own re-measurement supports this: median **1.47 s**, range 1.40–1.58, straddling both v2 figures. |
| **m-16** — E-19's arithmetic does not reconcile | **Accepted** | E-19 no longer claims a transfer rate: it reports the 14.15 s total and the ~5.8 s `pip install`, notes that **119 MB is on-disk while the wire transfer is ~45 MB compressed** (so the observed rate was nearer 5 MB/s than 50), and is explicitly labelled **not reproducible from the current machine state**. §3.3's slow-link extrapolation is restated in the honest form the review suggested. |
| **m-17** — V-2b part 2 removes a base image shared with `falkor-chat` | **Accepted, and part 2 was deliberately not run** | V-2b part 2 is now marked **OPTIONAL and shared-state**, citing `falkor-chat/Dockerfile:10`, with `docker pull python:3.12-slim` as a *mandatory* closing step if anyone runs it. It was skipped: E-18 already established the behaviour under controlled conditions, and part 1 is the property that needed proving. **Part 1 passed** in a network namespace with no connectivity — the full handshake returned real rows, and no registry contact appeared on stderr. |
| **n-5** — one helper call lacks `</dev/null` | **Accepted** | §4.5's header and the shipped script name the exception explicitly: `cpg_mcp_image_tag` is a sourced function that forks nothing, so the invariant reads as absolute rather than as an unexplained gap. |
| **n-6** — wrong line reference for the `AGENTS.md` bullet | **Accepted** | §9 now cites lines 20–24. |
| **n-7** — `grep -c '"id":1'` would also match `"id":10+` | **Accepted** | V-7 uses `'"id":1,'`. |
| **n-8** — E-16 is now only a pointer | **Not taken.** Cosmetic, and E-16's v1 claim *is* still stated inline ("the v1 guess … was right about the warm build in isolation and wrong about what it implied"). Restructuring it further would add diff noise to a row nothing depends on. |
| Review **open question 1** — is `MCP_TIMEOUT` really the startup knob? | **Settled live, in one command each, as suggested** | On Claude Code **2.1.220**: `claude mcp list` → `✔ Connected`; **`MCP_TIMEOUT=1 claude mcp list` → `✘ Failed to connect — MCP server "cpg" connection timed out after 1ms`**; `MCP_CONNECT_TIMEOUT_MS=1 claude mcp list` → still `✔ Connected`. So **`MCP_TIMEOUT` is the startup/connection budget** and E-17's prose reading was correct — the env-vars *reference table* is the misleading artifact, not the prose. The README documents the knob, the distinction from the per-tool-call `"timeout": 60000`, and this exact check. |
| Review **open question 2** — container-only eventually? | **Still open — a stakeholder call.** §10 item 3's recommendation (keep both) is unchanged and is what shipped. |
| Review **open question 3** — digest-pin the base now? | **Answered, answer unchanged, reasoning updated** | §3.7 now records that the *argument* moved: a pin used to be a correctness fix for the every-launch build; with no build on the launch path it would only close m-11's wobble. Deferred, and explicitly noted as now-cheap if the stakeholder wants reproducibility over convenience. |

### Measured results from the implementation run (2026-07-26)

| Step | Result |
|---|---|
| **S1** | `docker build --target runtime` succeeds by hand. |
| **S2** | `CPG_MCP_TAG=ba910c48571d`, **identical from three different working directories**; stdout **empty** (verified with `od -c`); a missing input returns **3** with a curated stderr message. |
| **S3** | `--help` works; `--verify-inputs` passes, and **fails as intended** on an uncovered file operand *and* an uncovered directory operand. |
| **V-1** | `cpg-mcp:ba910c48571d` + `cpg-mcp:dev` share image `3a1820f5b235`; `cpg-mcp:test-ba910c48571d` + `cpg-mcp:test` share `527191a571e4`. |
| **V-1b** | Re-run → early exit, no new image. `touch server.py` → **hash unchanged** (the property an mtime heuristic fails). Append a comment → hash moves to `2d877554c05e` and a new image appears; restoring the file restores the hash. |
| **V-2** | Median **1.47 s** of 7 runs (1.40–1.58) = **4.9 %** of the 30 s budget, far under the 25 %/7.5 s trigger. |
| **V-2b/1** | **Pass.** Full handshake inside a no-connectivity network namespace returned `rows=1 · n=1968`; no registry contact on stderr. |
| **V-2b/2** | **Not run** — optional, shared-state (m-17). |
| **V-3** | `53 passed, 7 deselected` — **no `PytestCacheWarning`**, so the `chown` works; byte-identical to the host baseline. |
| **V-4** | `7 passed, 53 deselected`; `GRAPH.LIST` afterwards = `cpg_falkorchat, ws:test, ws:acme, cpg_salesperson, reference` — **no `_cpg_mcp_selftest_*` residue**. |
| **V-5** | `graph=cpg_falkorchat · rows=1 · 1.6ms` / `n` / **1968**. |
| **V-6** | Replies for **ids 1, 2, 3**; `tools/list` reports exactly **one** tool named `query`; id 3 carries the count. Id 4 absent, as M-6 now specifies. |
| **V-6b** | **Pass.** A real build ran inside the launch path with a live protocol stream (`not present — building` then `build.sh: building …` on stderr) and the **id-1 `initialize` reply survived** — nothing consumed stdin. |
| **V-7** | Every stdout line parses as JSON; `grep -c '"id":1,'` = **1**; ids 1–3 present; all build chatter on stderr. |
| **V-8** | **Green, and adopted.** `--read-only --tmpfs /tmp` survived initialize, `tools/list`, a real query, `EXPLAIN`, unknown-graph, `PROFILE`-refusal and invalid-Cypher with no filesystem error. Per n-4, **V-6 and V-7 were re-run** after adopting them (and again after each later launch-path edit). |
| **V-9** | Host venv: `53 passed, 7 deselected` and `7 passed, 53 deselected`, unchanged. `run.sh` still answers. Audit: **no new failures** vs the C-309a baseline. All five personal-identifier greps clean on the new files. |
| **V-10** | **Partial in-session.** `claude mcp list` — a *fresh* process reading the edited `.mcp.json` — reports `cpg: bash -c exec "$CLAUDE_PROJECT_DIR/cpg/mcp/docker-run.sh" - ✔ Connected`, and left no container behind. The remaining half (`/mcp` + a real `mcp__cpg__query` **inside** a restarted session) cannot be obtained in the session that made the edit. |
| **V-11 pre-check** | `docker ps -a --filter label=cpg-mcp=1` → **empty** after every probe. V-11a/V-11b await the stakeholder's restart. |
| **Shared state** | `falkordb-dev` `StartedAt 2026-07-25T14:13:51Z`, `RestartCount 0`, unchanged throughout; `falkordb-data` present and untouched; `falkor-chat/scripts/start_falkordb.sh` not modified. |
