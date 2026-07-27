# Review — `docs/plans/cpg-mcp-containerization.md` (containerizing the `cpg` MCP server)

> Reviewer: `analyst`. 2026-07-25. Static pre-implementation review of the **plan only** —
> nothing is implemented yet, so nothing was reviewed as code.
> Artifact: [`../plans/cpg-mcp-containerization.md`](../plans/cpg-mcp-containerization.md) (632 lines, untracked, author `devops`).
> Baseline: working tree at `583e132` + the untracked plan.
> Findings route back to `devops`. This document changes nothing.
>
> **This file has three parts.** **Part I** (§1–§5) is the v1 review — verdict *needs changes* —
> kept verbatim as the audit trail. **[Part II](#part-ii--v2-re-review)** (§6–§14, 2026-07-26) is
> the re-review of the amended v2 plan — verdict **approve with suggestions**.
> **[Part III](#part-iii--delivered-change-review-c38f7dc)** (§15–§22, 2026-07-26) is the review of
> the **delivered code** at commit `c38f7dc` — verdict **approve with suggestions**. Read Part III
> first if you only need the current position; Parts I and II are what its findings are numbered
> against and are kept verbatim.

---

## 1. Scope & verdict

**Reviewed against:** `cpg/mcp/{README.md,run.sh,setup.sh,server.py,pytest.ini,requirements*.txt,tests/}`,
repo-root `.mcp.json`, `claude/scripts/audit-team.sh`, `docs/BACKLOG.md`, `docs/HISTORY.md`,
`docs/requirements/cpg-query-access.md`, `falkor-chat/{Dockerfile,compose.yaml,scripts/start_falkordb.sh}`,
root `AGENTS.md`, `skills/cpg-analysis/SKILL.md`, plus live read-only probes of the local Docker
engine and one run of the component's own test suite.

**Verdict: needs changes** — targeted amendments to the plan, not a redesign.

The design's load-bearing choices are right and should **not** be reopened: D-net (bridge +
`--add-host`), D-src (bake the source), D-image, D-ctx (narrow build context), D-test (host venv
stays primary), D-host (retain `setup.sh`/`run.sh`), the scope fence around C-310 and
`start_falkordb.sh`, and the rollback story. I re-verified the environment table where I could and
found no false claim in it.

What must change before implementation is concentrated in one area — **the launch path**: the plan
budgets the pre-launch build against an invented threshold rather than the harness's real
startup budget, omits the stdin invariant that is the exact twin of the stdout invariant it
polices so carefully, and does not consider container lifecycle (signals, orphans) at all. Three
majors, all cheap to close in the document.

**Required changes (the "needs changes" list):**

1. **M-1** — guard stdin on the pre-launch build (`</dev/null`) and state the stdin invariant.
2. **M-2** — anchor the §3.3/V-2 fallback trigger to the harness's measured startup budget
   (`MCP_TIMEOUT`), and state honestly what the cold-clone path actually does.
3. **M-3** — add container lifecycle to the design: `--init`, a label, and a "no orphaned
   containers after N session restarts" verification step.
4. **m-1 … m-4** below are wrong or under-specified done-conditions/doc rows that an implementer
   would otherwise have to invent (V-3's "byte-identical", the `.dockerignore` patterns, the
   `claude mcp add --scope local` recipe, V-9's grep).

---

## 2. What I verified independently

| Plan claim | Result |
|---|---|
| E-3 `falkordb-dev` on the default bridge only, publishing 6379/3000 | ✅ `docker ps`, `docker inspect falkordb-dev` → `bridge` only, IP `172.17.0.2`, gw `172.17.0.1`, `PortBindings` `HostIp:""` (i.e. `0.0.0.0`) |
| E-4 no user-defined network exists | ✅ `docker network ls` → `bridge`, `host`, `none` only |
| E-14 `python:3.12-slim` not cached | ✅ `docker images` → no `python` repo (present: `falkorchat:dev`, `falkor-chat-server:latest`, `falkordb/*`, `alpine:3.19/3.20` — the plan's own probe residue) |
| E-12 test baseline green | ✅ `cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q` → `53 passed, 7 deselected in 0.34s` (0.60 s wall) |
| E-1 Docker 29.6.1 | ✅ `docker version` → client/server 29.6.1; buildx v0.35.0 present, so **BuildKit is the default builder** (relevant to m-2) |
| §9 `skills/cpg-analysis/SKILL.md` needs no change | ✅ `grep -n "run.sh\|setup.sh\|venv\|docker"` → no hits |
| §9 `C-319` is the current maximum, `C-320` free | ✅ `docs/BACKLOG.md` runs C-301 … C-319 contiguously; no `C-320` anywhere in the repo |
| §4.5 unbraced `$CLAUDE_PROJECT_DIR` shape preserved | ✅ byte-compared against `.mcp.json:5`; only `run.sh`→`docker-run.sh` differs on that line |
| `docs/requirements/cpg-query-access.md` constrains the launch mechanism | ✅ it does **not** — FR-1…FR-6 and AC-1…AC-4 are silent on venv/Docker, so no requirement is violated or needs reconciliation |

**Not corroborated (accepted on the plan's word, correctly labelled by it):** E-5…E-11 (the network
and stdio probes) and E-16 (warm-build cost). Re-running them means creating containers and images,
i.e. mutating shared Docker state, which is out of a reviewer's remit. E-9/E-10 match documented
`docker run` behaviour and I have no reason to doubt them; **E-16 remains the plan's single most
load-bearing unmeasured assumption** — see M-2.

Baseline for the latency arithmetic below, measured here: host `import server` costs
**0.35–0.38 s** (3 runs, `cpg/mcp/.venv/bin/python -c "import server"`), on top of the plan's
measured ~0.6 s container start (E-11).

---

## 3. Findings

### Major

#### M-1 — `build.sh` inherits the MCP stdin pipe; the plan states the stdout invariant but not its twin

**Evidence:** §4.4, the line `"$HERE/build.sh" --runtime-only >&2` — stdout is redirected, stdin is
not. At that moment `docker-run.sh`'s stdin **is** the MCP protocol pipe: Claude Code writes
`initialize` into it as soon as the process is spawned, and nothing has consumed it yet.

**Why it matters:** the plan spends four separate passages (§4.3 header, §4.4 header, R-2, V-7) on
"nothing may write to stdout". The symmetric invariant — *nothing before `exec` may read stdin* — is
never stated and never guarded. Any byte read by `build.sh`, by a future `docker info` preflight, or
by a builder that decides to touch `/dev/stdin` is a byte the MCP server never sees: the handshake
then fails or, worse, half-consumes and the session gets a malformed stream. `docker build` with a
path context does not read stdin today, so this is a latent hazard rather than a live bug — but the
cost of closing it is ten characters and the cost of hitting it is an undiagnosable startup failure.

**Suggested improvement:** in §4.4 write `"$HERE/build.sh" --runtime-only >&2 </dev/null`, add
`exec </dev/null`-style discipline to `build.sh`'s own header contract ("reads nothing from stdin"),
and add the invariant to R-2's wording. Extend V-7's done-condition to assert that the
`initialize` response is present (not merely that every stdout line parses as JSON) — as written,
V-7 would pass on a stream that silently lost the first request.

#### M-2 — the "build before every launch" mechanism is budgeted against an invented threshold, and its headline benefit (fresh clone) is the one case it cannot deliver

**Evidence:** §3.3 and V-2 ("Expect ≪ 2 s… If it exceeds ~2 s, apply the fallback"), R-1, E-16
("assumed, not measured"), E-14 (`python:3.12-slim` not cached — corroborated).

Three separate problems:

1. **The ~2 s trigger is not derived from anything.** The number that matters is Claude Code's
   **server-startup timeout**, configured by the `MCP_TIMEOUT` environment variable (documented on
   the MCP page: *"Configure MCP server startup timeout using the `MCP_TIMEOUT` environment
   variable (for example, `MCP_TIMEOUT=10000 claude` sets a 10-second timeout)"*). The same page
   refers to *"the standard 5-second connect timeout"* in the `alwaysLoad` section. The default is
   not stated in the env-vars reference, so it must be **measured, not assumed** — and it may be as
   low as 5 s. The plan's `"timeout": 60000` is explicitly the **per-tool-call** wall
   (`cpg/mcp/README.md:211`, confirmed by the docs) and does **not** buy startup headroom.
   Against a 5 s budget, warm build (unknown, BuildKit's fixed overhead alone is typically several
   hundred ms) + 0.6 s container start + ~0.4 s interpreter/import leaves little margin, and the
   failure mode is the one the component was designed to avoid: a stdio server that fails to start
   is **absent for the whole session and is never auto-reconnected** (`README.md:233`).
2. **The fresh-clone benefit is largely illusory.** A cold build pulls ~150 MB and runs `pip
   install` — tens of seconds, far beyond any plausible startup budget. So on the exact scenario
   §3.3 cites first ("Image never built (fresh clone) → the wrapper builds it. The server does not
   silently vanish"), the server *does* vanish from that session; worse, the harness killing
   `docker-run.sh` mid-build cancels the build, so convergence may take several session restarts.
   The real fresh-clone fix is the documented `build.sh` step (which the plan already has). The
   auto-build's genuine value is **staleness**, which is real and worth keeping — but the plan
   should say so plainly rather than claiming both.
3. **The rejection of the mtime fallback is overstated, and a strictly better option is missing.**
   §3.3 says mtime "a `git checkout` … can fool in both directions". Checkout sets mtimes to *now*,
   never into the past, so the heuristic errs toward a (cheap, cached) rebuild — the safe
   direction. More to the point, a **deterministic, non-heuristic** variant exists that the plan
   never considers: tag the image by a content hash of the build inputs
   (`cpg-mcp:$(cat Dockerfile requirements*.txt server.py | sha256sum | cut -c1-12)`), then
   `docker image inspect "$IMAGE" >/dev/null 2>&1 || build`. That is an ~50 ms daemon call in the
   hot path instead of a full build, is exactly as staleness-proof as `docker build`, and needs no
   mtime reasoning at all.

**Suggested improvement:** (a) rewrite V-2 to measure the **end-to-end connect time** — `time` a
full `docker-run.sh` handshake (the V-6 pipeline), not just the build — and set the fallback
trigger from the measured `MCP_TIMEOUT` budget with a stated margin (e.g. "fallback if end-to-end
connect exceeds 40 % of the startup budget"); (b) name `MCP_TIMEOUT` in R-1's mitigation and in the
README as the documented escape hatch alongside `CPG_MCP_NO_AUTOBUILD=1`; (c) restate §3.3's
fresh-clone bullet honestly ("the auto-build converges over restarts; `build.sh` is the supported
fresh-clone path"); (d) add the content-hash-tag variant to the rejected/fallback alternatives with
a reason, since it dominates the mtime fallback the plan currently holds in reserve.

#### M-3 — container lifecycle is not designed: no signal story, no orphan check, no `--init`

**Evidence:** §4.4's `exec docker run -i --rm --add-host=… "$IMAGE"` — no `--init`, no `--name`, no
`--label`; §6 has no step that inspects `docker ps` after a session ends; §7's risk table has no row
for a leaked container.

**Why it matters:** the container's PID 1 is `python server.py`. PID 1 has no default signal
dispositions, and CPython installs a handler only for `SIGINT` — so **`SIGTERM` is ignored** inside
this container. The MCP stdio shutdown sequence is: close the child's stdin → wait → `SIGTERM` →
`SIGKILL`. The happy path is fine (stdin EOF propagates through `docker run -i` and the server
exits). The unhappy paths are not analysed: a `SIGTERM` to the `docker run` CLI is sig-proxied to a
process that ignores it; a `SIGKILL` to the CLI leaves the *container* alive, owned by the daemon,
until the attach stream teardown reaches it. Across many session restarts on a shared engine that
also runs `falkordb-dev`, silently accumulating `cpg-mcp:dev` containers is a real (if low-grade)
operational failure — and with no `--name`/`--label` there is no way to find or clean them.

**Suggested improvement:** add `--init` (tini as PID 1: forwards signals properly and reaps) and
`--label cpg-mcp=1` to §4.4's `docker run`; add a §7 risk row; and add a verification step —
"restart the session 3×, then `docker ps -a --filter label=cpg-mcp=1` is empty" — to §6. Cheap, and
it converts an unknown into a checked property.

### Minor

#### m-1 — V-3's "byte-identical to the host baseline" is likely wrong as a done-condition

**Evidence:** §4.1's test stage runs `USER appuser` while `/app` is root-owned (`WORKDIR /app`,
everything `COPY`ed as root); §6 V-3 says *"Expect `53 passed, 7 deselected` — byte-identical to the
host baseline E-12."*

pytest writes `.pytest_cache` into its rootdir. As a non-root user in a root-owned `/app` that write
fails and pytest emits a `could not create cache path …` warning. The counts will match; the output
will not be byte-identical, and an implementer chasing the stated done-condition will burn time on a
non-defect.

**Suggested improvement:** soften the done-condition to "the same counts (`53 passed, 7
deselected`)", and either add `-p no:cacheprovider` to the test stage's `CMD`/V-3 command or
`RUN chown appuser:appuser /app` in the test stage.

#### m-2 — R-8's "warm builds are offline" is unverified, and BuildKit is the default builder here

**Evidence:** R-8 (*"Warm builds are offline"*) has no probe behind it; `docker buildx version` →
v0.35.0 with Docker 29.6.1, so `docker build` goes through BuildKit, which resolves the `FROM`
tag's manifest (`[internal] load metadata for docker.io/library/python:3.12-slim`) before consulting
the layer cache. Whether that resolution is satisfied purely from cache when the network is gone is
version-dependent and is exactly the kind of claim this plan otherwise insists on probing.

**Why it matters:** the assertion is load-bearing for the design — an offline machine (or a Docker
Hub outage) turning every session start into a failed MCP server would be a serious regression over
the venv path.

**Suggested improvement:** add a V-2b probe — run `cpg/mcp/build.sh --runtime-only` with the network
down (or `docker build --network=none` plus a disconnected daemon check) and record the result;
if it fails offline, that is a decisive argument for the image-inspect/content-hash variant in M-2(d)
rather than a build call on every launch.

#### m-3 — `.dockerignore` patterns are not recursive; `tests/__pycache__` will ship into the test image

**Evidence:** §4.2 lists bare `__pycache__` and `*.pyc`. Docker's ignore patterns are
`filepath.Match`-based and match **relative to the context root** unless prefixed with `**/`
(Docker's own docs use `**/*.go` for the recursive case). `cpg/mcp/tests/__pycache__` exists on this
box (`ls -d cpg/mcp/tests/__pycache__` → present), so it lands in the build tar and in the `test`
stage's `COPY tests tests`.

**Suggested improvement:** use `**/__pycache__`, `**/*.pyc`, `**/.pytest_cache`. Harmless in effect
(bytecode is invalidated by mtime/size), but it defeats the stated purpose of the file and adds host
artefacts to an image the plan wants to be self-contained.

#### m-4 — V-9's PII grep is narrower than the check it stands in for

**Evidence:** V-9 runs `grep -rn -F "$HOME" …`; `claude/scripts/audit-team.sh:116-137` (check 7)
greps for **five** identifiers — home path, username (word-bounded), git `user.name`, git
`user.email`, hostname. S5 even says "for `$HOME`/username", but the command only covers `$HOME`.

**Why it matters:** the whole reason the plan grep is manual is that check 7 is `git grep`-based and
blind to untracked files (C-309b); a grep that covers 1 of 5 identifiers restores only 1/5 of the
gate. New shell files are a plausible place for a `# <user>@…` note to slip in.

**Suggested improvement:** in V-9, mirror check 7's set explicitly, e.g.
`grep -rnE -e "$HOME" -e "\b$(id -un)\b" -e "$(git config user.email)" -e "$(git config user.name)" -e "$(hostname)" cpg/mcp/Dockerfile cpg/mcp/.dockerignore cpg/mcp/build.sh cpg/mcp/docker-run.sh`.

#### m-5 — R-3's "the wrapper preflights `docker info`" is not what §4.4 specifies

**Evidence:** R-3 states the wrapper preflights `docker info`; §4.4's script only checks
`command -v docker`. The `docker info` preflight lives in `build.sh` (§4.3) — which is skipped
entirely under `CPG_MCP_NO_AUTOBUILD=1`. With autobuild off and a dead/wrong-context daemon, the
user gets docker's raw error, not the curated one the risk row promises. Related: with autobuild off
and no image, `docker run cpg-mcp:dev` will attempt a **registry pull** and fail with `pull access
denied` — an actively misleading message for a locally-built image.

**Suggested improvement:** move the daemon preflight into `docker-run.sh` (before the autobuild
branch) and add `--pull=never` to the `docker run`, so the missing-image case says "image not built —
run `cpg/mcp/build.sh`" instead of a registry error.

#### m-6 — the live suite's scratch-graph name collapses to a constant inside a container

**Evidence:** `cpg/mcp/tests/test_server.py:472` — `name = f"_cpg_mcp_selftest_{os.getpid()}"`. In a
PID namespace the test process is PID 1, so every containerized `-m live` run uses the *same* graph
key `_cpg_mcp_selftest_1` on the **shared** FalkorDB.

**Why it matters:** the plan leans on E-13 to argue the live run is self-contained. It still is
against `cpg_*`/`ws:*`/`reference`, but the uniqueness property that made it safe against *itself*
is lost: two concurrent container live runs corrupt each other, and an interrupted run leaves a
residual graph on the shared instance.

**Suggested improvement:** note it in §3.4 and make V-4 pass a disambiguator
(`-e PYTEST_XDIST_WORKER=…`-style is overkill; simplest is `docker run --pid=host`-free approach:
have V-4's done-condition include `redis-cli GRAPH.LIST` showing no `_cpg_mcp_selftest_*` residue).
A one-line test change (`uuid4().hex[:8]` instead of `getpid()`) would fix it properly — that is a
`tdd-engineer`/`coder` item, out of this plan's scope, and worth a backlog line.

#### m-7 — the auto-build only refreshes `cpg-mcp:dev`, so the gate image can silently lag

**Evidence:** §4.4 calls `build.sh --runtime-only`; V-3/V-4 run `cpg-mcp:test`. Nothing in §5/§6
requires a `build.sh` (both targets) immediately before the in-container gate, so the "proves the
image" gate can be run against a stale test image after an edit — the precise failure D-src/§3.3
exist to prevent, reintroduced on the test side.

**Suggested improvement:** make "run `cpg/mcp/build.sh` (both targets) first" an explicit
precondition of V-3/V-4 and of the §3.4 table's "When" column.

#### m-8 — concurrent sessions and image-tag races are not analysed

**Evidence:** §3.3, §7 — no mention. Two Claude Code sessions (repo root plus a component
subdirectory, which this repo actively encourages) starting at once each run `docker build` on the
same tag. BuildKit serialises overlapping work, so the second session's startup can wait on the
first's build — precisely the budget M-2 is about — and if the two sessions see different working
trees (e.g. one mid-edit), the `:dev` tag ends up pointing at whichever finished last.

**Suggested improvement:** one paragraph in §3.3 stating the behaviour and why it is acceptable
(both builds converge, results are idempotent for an unchanged tree), plus a note that the
content-hash tag from M-2(d) removes the race entirely.

#### m-9 — R-6 tracks the *port* but not the *bind address*, which is the new dependency

**Evidence:** R-6 says the design "depends on FalkorDB publishing 6379 to the host". The sharper
dependency is that it publishes on **`0.0.0.0`**, not `127.0.0.1`: `start_falkordb.sh:52-58` uses
`-p "${FALKORDB_PORT}:6379"` (`HostIp:""` — I confirmed via `docker inspect`). The venv path works
either way; the container path breaks the moment anyone hardens that to `-p 127.0.0.1:6379:6379`,
which is an obvious future security tidy-up. Second-order: a host firewall that DROPs (rather than
rejects) traffic from `docker0` turns a fast `ECONNREFUSED` into a hang that only the 60 s tool
timeout ends — the curated "FalkorDB unreachable" message would not appear.

**Suggested improvement:** reword R-6 to name the bind address, and add a line to the README's
troubleshooting section: "if FalkorDB is ever published to loopback only, the container path needs
`--network host` or a user-defined network (§3.1 options B/D)."

#### m-10 — §9's documentation list has one concrete gap

**Evidence:** `cpg/mcp/README.md:269` documents
`claude mcp add --scope local cpg -- <repo-root>/cpg/mcp/run.sh` as the **fallback wiring for when
`$CLAUDE_PROJECT_DIR` expansion fails**. That failure mode has nothing to do with venv-vs-container,
so after this change the local-scope recipe would quietly wire a *different* launch path than the
project-scope one. §9's `cpg/mcp/README.md` row lists five sub-changes and this is not one of them;
§3.5(4) actually reinforces the confusion by citing that recipe as a reason to keep `run.sh`.

Everything else in §9 checks out: I re-grepped the tracked tree and the only other `run.sh`/venv
references are in `docs/HISTORY.md:24`, `docs/BACKLOG.md:147` (both historical records of what
shipped — correctly left alone), the frozen `docs/plans|test-plans|test-reports/cpg-query-access*`
set, and agent `kaizen/inbox.md` files. `skills/cpg-analysis/SKILL.md` genuinely needs no change,
`C-320` genuinely does not collide, and `docs/requirements/cpg-query-access.md` imposes no
constraint this work violates.

**Suggested improvement:** add a §9 sub-bullet: update line 269 to `docker-run.sh` (keeping a
one-line note that `run.sh` is the Docker-less variant).

### Nits

- **n-1** — §4.1: `RUN useradd` sits *after* `COPY server.py` in `base`, so every source edit
  re-runs `useradd`. Move it above the `COPY` lines (falkor-chat's Dockerfile has the same ordering,
  so "follows the precedent" is intact either way).
- **n-2** — §8 opens with "**One line**, no cleanup" and then describes two edits (`args` and
  `FALKORDB_HOST`); §4.5 correctly says "edited (two lines)". Align the wording.
- **n-3** — no `--memory`/`--cpus` and no `--pull=never` on the launch `docker run`. Neither is
  needed for correctness; `--pull=never` is worth it for the error-message reason in m-5.
- **n-4** — §3.7's `--read-only --tmpfs /tmp` probe (V-8) is sequenced *after* V-6/V-7 but the
  flags belong to the wrapper; state explicitly that adopting them requires re-running V-6/V-7,
  not just V-8.

---

## 4. What's solid

- **D-net (§3.1) is correct and well argued.** Bridge + `--add-host=host.docker.internal:host-gateway`
  is the right call, and the rejection of `--network host` (privilege + Docker-Desktop portability,
  with E-2 showing a Desktop context one `docker context use` away) and of the literal container IP
  (reassigned on every `--rm` restart) is sound. **The rejection of option D is the best passage in
  the plan**: it correctly identifies that a user-defined network requires either mutating the
  shared `start_falkordb.sh` (stopping a container `falkor-chat` and `salesperson` also depend on)
  or a non-persistent, invisible manual `network connect` — and that A adds *no new coupling*
  because the published port is already the contract every other consumer uses. I confirmed the
  premises: `falkordb-dev` is bridge-only, no user-defined network exists, and the port is published
  on all interfaces.
- **Blast radius on the shared FalkorDB is respected throughout** — §1's out-of-scope fence, §3.1's
  D-a/D-b analysis, §8's explicit "do not touch `falkordb-dev` or `falkordb-data`". The compose
  double-engine hazard (`falkor-chat/compose.yaml`'s own header warning) is correctly cited as the
  reason not to add a Compose service.
- **D-test (§3.4)** keeps the component's only regression signal fast, host-side and Docker-free
  while adding a gate that tests the artefact that actually ships. That is the right split, and the
  multi-stage rationale (no pytest in the runtime image) is proportionate rather than ceremonial.
- **`.mcp.json` (§4.5) is exactly right** — the unbraced `$CLAUDE_PROJECT_DIR` under `bash -c` is
  preserved verbatim, with the reason restated, so check 7 stays clean and no home path enters a
  tracked file. Sub-directory sessions are unaffected: the shape, and therefore the pre-existing
  approval-scoping behaviour documented at `cpg/mcp/README.md:219-225` (and backlog C-319), is
  unchanged by this work — neither improved nor regressed.
- **Scope discipline is clean.** C-310 is analysed (§3.9) and explicitly not absorbed — no
  OpenCode/Kiro config is written and the "do not renumber" instruction is repeated in §9. FalkorDB's
  lifecycle is untouched. `server.py` and the tool contract are untouched.
- **§3.7's omissions are justified rather than defaulted** — no `EXPOSE`, no `HEALTHCHECK`, no
  Compose service, because this is a one-shot stdio process. That is the correct read of the
  workload and the plan says *why*.
- **V-6's trailing-`ping` note** (carrying `claude/coder/kaizen/inbox.md:34`'s EOF gotcha into the
  container verification) is exactly the kind of cross-run knowledge transfer that saves an
  implementer a wasted afternoon.
- **Rollback (§8) genuinely is cheap** because §3.5 retained the venv path — the two decisions
  reinforce each other, and the plan is honest about the drift cost that buys.

---

## 5. Open questions

1. **What is this machine's actual MCP server-startup budget?** `MCP_TIMEOUT` is documented as the
   knob; the default is not stated in the public env-vars reference, and the MCP page's only
   numeric hint is "the standard 5-second connect timeout" in the `alwaysLoad` section. This is
   measurable in one session restart and it determines whether §3.3's mechanism ships as designed or
   falls back — I could not measure it without changing `.mcp.json`. It belongs in V-2's
   done-condition either way (M-2).
2. **Does the stakeholder want the container path to be the *only* path eventually?** §10's own
   second question. The plan's answer (keep both) is well defended for now, but the two-path drift
   cost is permanent and the review can only note it, not settle it.
3. **Is a leftover `cpg-mcp` container after a session an acceptable residue, or a defect?** M-3
   assumes defect. If the stakeholder is relaxed about it, `--init` is still worth adding but the
   verification step can be dropped.

---
---

# Part II — v2 re-review

> Reviewer: `analyst`. **2026-07-26.** Re-review of **v2** of
> [`../plans/cpg-mcp-containerization.md`](../plans/cpg-mcp-containerization.md) (1104 lines, still
> untracked, still **design only — nothing implemented**), amended by `devops` in response to
> Part I. Baseline: working tree at `583e132` + the untracked plan + `claude/devops/kaizen/inbox.md`
> (modified, 4 new entries).
> Part I above is unchanged on purpose — it is the audit trail. Finding IDs continue Part I's
> namespace (`M-4`, `m-11`, `n-5`, …) so the two halves read as one ledger.

## 6. Scope & verdict (v2)

**Reviewed:** §11's disposition map against the plan body, finding by finding; then the substance of
the three things v2 changed materially (§3.3's mechanism reversal, the startup-budget correction,
§3.10's new lifecycle section) and the new artifacts and verification steps they pull in (§4.3
`image-tag.sh`, §4.4/§4.5's amended scripts, R-2b/R-8/R-9/R-10/R-11, V-1b/V-2/V-2b/V-6b/V-11,
backlog C-321). Re-verified the environment facts I could reach read-only, and checked the
load-bearing external claims (Claude Code's MCP startup budget, `docker run -e VAR`, `--pull=never`)
against current official docs.

**Verdict: approve with suggestions.** No blockers. **Implementation may start** — every design
decision in the plan is now either independently corroborated or soundly argued, and none of my
remaining findings changes a decision, an artifact's shape, or the sequence. Three of them (**M-4,
M-5, M-6**) are *majors* purely because they are done-conditions or specs that cannot be executed as
written: one would leave the plan's own new guard with a hole, one cannot pass where §5 sequences it
(and, followed literally, would kill the live session's own server), and one contradicts a gotcha the
same section documents. All three are line-level edits inside §4.3/§6.

**The M-2 reversal is correct and I withdraw the reserve position I held in Part I.** E-18 is the
right measurement, it was designed the right way (before/after `docker pull`, all layers `CACHED`,
`--progress=plain`), its mechanism matches documented BuildKit behaviour, and the conclusion it
drives — that `docker build` must not sit on a hot path that has to work offline — is the correct
one. Note that v2 reaches this by *disproving its own v1 claim*; that is the behaviour a review
process is for.

**Bar for this verdict, stated plainly** (since Part I was *needs changes* on findings that were
also "cheap to close in the document"): Part I's majors were a wrong mechanism, a missing design
section and an unguarded invariant in a shipped artifact. v2's are three wrong sentences in the
verification plan. That is a different order of defect and I am not going to re-gate a design over
it.

### What an implementer may proceed with

- **S1–S4 as written** — Dockerfile, `.dockerignore`, `image-tag.sh`, `build.sh`, `docker-run.sh`
  exactly as §4.1–§4.5 specify, **with M-4 settled first** (it changes one function in §4.3 and one
  check in §4.4 — a spec decision, not a redesign).
- **S5's verification pass**, with **M-6**'s correction applied to V-6/V-6b/V-7 before running them
  and **m-12**'s flag typo fixed in V-2b.
- **S6–S7** as written, with **M-5**'s correction applied to V-11 before running it. Do not run
  V-11's cleanup instruction inside a live session.
- **S8** documentation, adding **m-14**'s missing row.

Everything else in the "must change" column of Part I is closed. The rest of my v2 findings
(m-11 … m-17, n-5 … n-8) are non-blocking accuracy and hygiene items — take them or leave them, but
m-11 and m-13 are the two worth taking.

## 7. Disposition audit — was every Part I finding actually addressed?

I checked each §11 row against the plan body rather than taking the row's word for it. **No finding
was silently dropped, and no disposition misrepresents what the body does.** That is unusual and
worth saying.

| Part I ID | v2 disposition | My verdict on the disposition |
|---|---|---|
| **M-1** stdin | Accepted, widened | **Closed, and improved on the suggestion.** `exec </dev/null` as `build.sh`'s second line is structurally stronger than the per-call redirect I proposed (a future caller cannot forget it), *and* the per-call redirects are there too. R-2b is a real twin of R-2. E-22 measured the hazard as latent rather than live — correct calibration. V-6b is a better test than anything I suggested: it forces a hash miss so a *real* build runs inside the launch path with a live protocol stream. **But its done-condition is wrong — see M-6.** |
| **M-2** budget/mechanism | Accepted, §3.3 rewritten, mechanism changed | **Closed.** All five sub-points land. Independently scrutinised in §8 below. |
| **M-3** lifecycle | Accepted, measured | **Closed on design.** §3.10 is measured rather than reasoned, and E-23's four-path table is *narrower* than my speculation — the orphan window is genuinely small. The `--name` rejection is right and I had not thought of the collision. **V-11's done-condition is wrong — see M-5.** |
| **m-1** byte-identical | Accepted, fixed at the environment | **Closed, better than suggested.** `RUN chown appuser:appuser /app` keeps V-3's strong claim instead of weakening it, and the rejection of `-p no:cacheprovider` (it would break host/container command parity) is a better argument than my suggestion. |
| **m-2** offline builds | Accepted; claim was false | **Closed.** This is the finding that changed the design. R-8's rewrite is precise and no longer overclaims. |
| **m-3** `.dockerignore` globs | Accepted, measured (E-25) | **Closed.** Also correctly promoted `.dockerignore` to a hash input. |
| **m-4** PII grep | Accepted in full | **Closed — verified against the source.** V-9's five commands mirror `claude/scripts/audit-team.sh:116-137` exactly, including `-I -n -i -F` and `-w` on username/hostname only. `image-tag.sh` is in the file list. |
| **m-5** preflight location | Accepted in full | **Closed.** The `docker info` preflight is now in `docker-run.sh` *before* the autobuild branch, so it covers `CPG_MCP_NO_AUTOBUILD=1`. `--pull=never` adopted; E-27 measured both messages. |
| **m-6** scratch-graph name | Accepted; noted, worked around, filed | **Closed correctly.** Out-of-scope call is right, C-321 is the right home, and V-4's residue check is the right stopgap. Note `--init` does *not* fix this (PID would become 2, still constant) and the plan does not claim it does. |
| **m-7** stale gate image | Accepted, made structural | **Closed, better than suggested.** Sharing `<hash12>` between `cpg-mcp:<hash>` and `cpg-mcp:test-<hash>` turns a remembered precondition into an unreachable command. **Caveat:** the structural guarantee has one hole — see M-4. |
| **m-8** concurrency race | Accepted, race removed | **Closed.** One over-claim: "at most one build" (§3.3 table, R-10). Two concurrent misses on the same tag still *launch* two builds; BuildKit dedupes the work, so session B still waits on session A — the same wait v1 had, now only on a miss instead of every launch. The clobbering half of the race is genuinely removed. Not worth its own finding; just do not read "at most one build" as "no waiting". |
| **m-9** bind address | Accepted in full | **Closed.** R-6 now names `0.0.0.0`, cites the launcher, carries the DROP-vs-REJECT second-order note, and is raised to Low–Medium. |
| **m-10** README line 269 | Accepted in full | **Closed — line reference verified.** `cpg/mcp/README.md:269` is indeed the `claude mcp add --scope local … run.sh` recipe; `:211` is indeed the per-tool-call `timeout` bullet. §3.5 point 4's self-correction is the right call. |
| **n-1 … n-4** | Accepted (n-3 in part) | **All closed.** The `--memory`/`--cpus` rejection is well argued (a wrong ceiling would surface as an opaque OOM kill at session start). |
| **Open Q1** budget | Answered | **Answered, and I verified it — see §9.** |
| **Open Q2** two paths | Left open with a recommendation | Correct handling; still a stakeholder call. |
| **Open Q3** orphans | Answered with a recommendation | Correct handling; the "defect, but it only costs one `docker ps`" framing is right. |

## 8. Independent scrutiny of the M-2 reversal

**The measurements support the conclusion.** Three claims carry it, and each stands on its own:

1. **A fully-cached BuildKit build still resolves `FROM` over the network.** E-18's design is the
   right one: `--progress=plain`, every layer `CACHED`/`DONE 0.0s`, exactly one live step
   (`[internal] load metadata for docker.io/library/python:3.12-slim` → `DONE 0.5s`) against a
   0.54–0.65 s total; then `docker pull` and the same step → `DONE 0.0s`, total 0.31–0.34 s. That is
   a controlled before/after with the effect isolated to one step, repeated 3× per arm. It also
   matches the documented mechanism: with the default `docker` driver, the Dockerfile frontend's
   image-config resolution prefers the daemon's **image store** and falls back to the registry, and
   the *build cache* is not the image store. I did not re-run it (a build mutates shared state), so
   this is *corroborated by mechanism and by the honesty of the experiment design*, not re-measured.
2. **A BuildKit build does not populate the image store** (E-14 amended). Consistent with the
   above, and it is the correction of a genuine methodological error in v1 — `docker images python`
   was reading the wrong store. I confirmed the *current* state: `python:3.12-slim` is now in the
   image store (119 MB disk usage), which is exactly the residue the plan discloses.
3. **`docker image inspect` never contacts a registry** (E-21, 0.05–0.07 s). True by construction —
   it is a daemon-local metadata read of the image store — and consistent with `--pull=never`'s
   documented behaviour (*"Do not pull the image, even if it's missing, and produce an error if the
   image does not exist in the image cache"*, Docker CLI reference, verified).

**So the offline argument is sound, and it is the right axis to decide on**: the venv path needs no
network at all, so an every-launch registry round trip would have been a straight regression. v1's
R-8 was the load-bearing false claim in the whole design, and v2 found it by probing the thing I
told it to probe.

### The new mechanism's own failure modes — how well are they designed?

| Failure mode | Verdict |
|---|---|
| **Hash cost** | Non-issue. `< 0.01 s` (E-21) for 6 files + 2 test files, computed with `sha256sum` per file into a final `sha256sum`. Verified the tree: `cpg/mcp/tests/` holds only `conftest.py`, `test_server.py` and `__pycache__`, so the enumeration is tiny. |
| **Hash correctness — does it cover everything that affects the image?** | **Partly. Two gaps, one of which matters.** (a) The `tests/` enumeration is a `*.py` glob while the Dockerfile does `COPY tests tests` — a directory. That is **M-4**, and it also makes `--verify-inputs` unimplementable as specified. (b) The base image and pip resolution are outside the hash — **m-11**. `Dockerfile`, `.dockerignore`, `requirements*.txt`, `server.py`, `pytest.ini` are all covered, the relative-path-and-contents-only rule correctly keeps absolute paths out (check 7), and `LC_ALL=C sort -z` correctly removes locale dependence. A missing input being a hard error rather than a silent skip is the right call and closes the obvious collision. |
| **Concurrent sessions racing a miss** | **Adequately designed.** Content-addressed tags remove the clobbering failure entirely (different trees → different tags), and same-tree duplicates are idempotent. The residual — session B waiting on session A's build — is now confined to a miss, and at 30 s of budget for a ~1 s build it is immaterial. §3.3/R-10's "at most one build" is loose; the substance is right. |
| **Partial / failed build leaving the tag absent** | **Correct by construction, but the diagnostic is missing.** `docker build -t` applies tags only on success, so a failed build leaves no tag and the next launch retries — no half-built image can ever be run. `set -euo pipefail` means `docker-run.sh` exits before reaching `docker run`, so `--pull=never` never even fires. What is missing is the *message*: this is the only launch failure path with no curated line, and it is the most likely one on a fresh clone. **m-13.** |
| **`--pull=never`** | **Right flag, verified semantics, and E-27's before/after message capture is exactly the evidence I asked for.** It also means a hash miss with `CPG_MCP_NO_AUTOBUILD=1` fails fast and legibly instead of attempting a registry pull of a local-only name. |
| **Accumulation of hash tags** | Handled honestly: `LABEL cpg-mcp=1` makes them enumerable, pruning stays a human act, and the launch path is explicitly forbidden from removing anything. The marginal disk cost claim is right — new images share every layer but the `COPY server.py` one. |

## 9. The startup budget (Part I's M-2 problem 1 / open question 1) — verified

I checked v2's E-17 against current official docs. **The substance is confirmed on every point that
matters**, including the version number:

- *"MCP server connections time out after 30 seconds by default. If your server takes longer to
  start, the connection fails. Raise the limit with the `MCP_TIMEOUT` environment variable"* —
  `code.claude.com/docs/en/agent-sdk/mcp`, §Troubleshooting → Connection timeouts. **This is the
  30 s budget and it names `MCP_TIMEOUT` as the knob.** The Claude Code MCP page agrees:
  *"Configure MCP server startup timeout using the `MCP_TIMEOUT` environment variable (for example,
  `MCP_TIMEOUT=10000 claude` sets a 10-second timeout)"*.
- *"Connection is non-blocking by default: the first turn begins without waiting, and each server's
  tools become available once its connection completes. `{/* min-version: 2.1.142 */}` Before
  Claude Code v2.1.142, startup blocked on the connection batch for up to 5 seconds."* — **v2's
  "non-blocking by default since v2.1.142" is exactly right**, and this box is 2.1.220 (verified).
- *"To restore a bounded startup wait for every server, set `MCP_CONNECTION_NONBLOCKING` … to `0`.
  The wait is capped at 5 seconds by `MCP_CONNECT_TIMEOUT_MS`"*, and for `alwaysLoad`: *"blocks
  startup until the server connects, capped at the standard 5-second connect timeout. This applies
  even though MCP startup is otherwise non-blocking by default."* — **v2's attribution of the 5 s
  figure to the blocking/`alwaysLoad` paths only is correct**, and `cpg` uses neither. My Part I
  worry that the budget "may be as low as 5 s" does not apply. Even if it did, 1.36 s fits.
- *"Stdio servers are local processes and are not reconnected automatically"* — E-29's second half
  confirmed; `cpg/mcp/README.md:232-236` already says this.
- `.mcp.json`'s `"timeout": 60000` as the per-tool-call wall — confirmed, and `README.md:211`
  already documents it that way.

**One residual ambiguity, non-blocking.** The env-vars *reference table* is inconsistent with that
prose about which variable owns which default: two fetches of the same page disagreed with each
other, and the one that returned rows described `MCP_TIMEOUT` as a **tool-call** timeout
(default 600000) and `MCP_CONNECT_TIMEOUT_MS` as the **connection** timeout (default 30000) —
i.e. the mirror image of E-17's quoted strings. `MCP_TOOL_TIMEOUT`'s own description
(*"Overrides `MCP_TIMEOUT` for tool execution while keeping the connection timeout separate"*)
suggests `MCP_TIMEOUT` is a combined knob that governs both, which reconciles the two readings in
v2's favour. So: **treat E-17's quoted strings as paraphrase rather than verbatim**, and see open
question 1 below for the one-command live check. Nothing in the decision moves — every reading of
these docs leaves the measured 1.36 s connect with ≥ 3.7× margin, and the 30 s figure for
*connections* is stated identically on both prose pages.

## 10. Sanity-check of the measured claims (and the prune disclosure)

**Re-verified read-only, all consistent with v2's text:**

| v2 claim | Result |
|---|---|
| Probe residue: `python:3.12-slim` in the image store, `alpine:3.19/3.20` present, **no `cpg-mcp*` image** | ✅ `docker images` → exactly those 8 images, no `cpg-mcp` tag. Nothing was shipped. |
| Build cache "~114 MB, 43 MB reclaimable" | ✅ `docker system df` → Build Cache 113.8 MB, 43.4 MB reclaimable; `docker buildx du` agrees. |
| `falkordb-dev` untouched throughout | ✅ still `running`, `StartedAt 2026-07-25T14:13:51Z`, same image — never restarted by the measurement work. Only 1 container exists on the engine. |
| E-26's `GRAPH.LIST` (shared graph list unchanged, no `_cpg_mcp_selftest_*`) | ✅ `cpg_falkorchat`, `ws:test`, `ws:acme`, `cpg_salesperson`, `reference` — byte-identical to the plan's list. |
| E-12 host baseline green | ✅ `53 passed, 7 deselected in 0.47s`. |
| E-1 Docker 29.6.1, Claude Code 2.1.220 | ✅ both. |
| `-e VAR` bare form "forwards only if set, else the image ENV default applies" (§4.5) | ✅ Docker CLI reference: *"If no `=` is provided and that variable isn't exported in your local environment, the variable is unset in the container."* And `server.py:50-80`'s `_env_int` tolerates junk anyway. |
| `test_server.py:472`, `start_falkordb.sh` `-p "${FALKORDB_PORT}:6379"`, `README.md:211/269`, `AGENTS.md` 124–127, `audit-team.sh:116-137` | ✅ every cited anchor exists as described (one imprecise line reference — n-6). |
| devops inbox entries "filed by the v2 design run" (§9) | ✅ 4 new dated entries (E-18 store-vs-cache, E-23 PID-1 `SIGTERM`, E-28 interrupted build, plus the v1 stdout-clean one), all well-formed with evidence/context/home. |
| C-320 and C-321 free, C-319 the max, C-310 present | ✅ `docs/BACKLOG.md` runs C-301…C-319 contiguously; no `C-320`/`C-321` anywhere tracked. **C-321 does not collide.** |
| `falkor-chat/scripts/start_falkordb.sh` untouched | ✅ working tree clean except the two untracked docs and the devops inbox. |

**Not corroborable, and correctly labelled or flagged:** E-18/E-19/E-20/E-21/E-22/E-23/E-24/E-25/
E-27/E-28 all require building images or creating containers, which is outside a reviewer's remit.
Two of them do not survive an internal-consistency check and should be restated:

- **E-20's table is not internally coherent** — the chosen variant (1.36 s) is a strict superset of
  the "`docker run` only" variant (1.42 s) yet is reported as faster. **m-15.**
- **E-19's cold-build arithmetic does not reconcile** with its own "~119 MB at ~50 MB/s"
  attribution. **m-16.**

**Does `docker builder prune -f` invalidate anything?** No — with one caveat.

- It discarded **regenerable build cache only**; no image, container or volume was touched, which I
  can corroborate negatively (all 8 images and the single container are accounted for, the FalkorDB
  container was never restarted, the graph list is unchanged). The disclosed side effect — the next
  `falkorchat:dev` / `falkor-chat-server` build re-runs its layers — is real, bounded and honestly
  stated. This is the right way to disclose a shared-state mutation.
- **The caveat is about reproducibility, not validity.** The current machine state (base in the
  image store + warm cache) is the *opposite* of E-19's stated preconditions, so E-19 is the one
  measurement that can no longer be re-verified by anyone without a destructive reset. The plan
  discloses this ("your first build will not be cold"), which is why I am recording it as an
  accuracy finding (m-16) rather than a challenge to the number.
- **Leaving `python:3.12-slim` in the store is the right call** and the plan's reasoning is correct:
  it is what makes a *miss*-triggered build offline-tolerant, and `build.sh` step 4 makes it explicit
  policy rather than a lucky residue. The consequence to keep in view is that this machine will
  never exercise the fresh-clone cold path again, so R-1's residual risk stays inferred.

## 11. Findings on v2's new material

### Major

#### M-4 — the hash input list and `--verify-inputs` are under-specified for the one directory `COPY` that exists, leaving a hole in the guard R-11 relies on

**Evidence:** §4.1 `COPY tests tests`; §4.3's enumeration = a fixed list of 6 files **plus every
`*.py` under `tests/`**; §4.4 step 3 — *"extract the source operand of every `COPY` line in the
Dockerfile and assert each is covered by `cpg_mcp_input_files`"*; R-11 calls this *"deterministic
enforcement … not a comment"*, and §3.4/m-7 calls the shared hash a structural guarantee.

**Why it matters, two ways.** `tests` is a **directory** operand; the input list never contains the
string `tests`, only `tests/conftest.py` and `tests/test_server.py`. So (a) the natural
implementation of "assert each `COPY` source is covered" — set membership — **fails on the Dockerfile
the plan itself specifies**, making S3's done-condition ("`--verify-inputs` passes") unreachable
without inventing an unstated rule; and (b) the rule that makes it pass (prefix match) leaves the
real hole open: a non-`.py` file added under `tests/` — a JSON fixture, a `.cypher` sample, a
`pytest.ini` override — is `COPY`ed into the test image but **not hashed**, so `cpg-mcp:test-<hash>`
silently stays stale and the gate runs against old test inputs. That is exactly the m-7 failure §3.4
claims is now structurally impossible. Verified there is **no live defect today**: `cpg/mcp/tests/`
contains only `conftest.py`, `test_server.py` and `__pycache__` (excluded by `.dockerignore`). The
hole is in the guard, not in the current tree — which is precisely when it is cheap to close.

**Suggested improvement:** make the enumeration **directory-driven instead of glob-driven** — for a
directory operand, enumerate every file under it that `.dockerignore` would not exclude
(`find tests -type f ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z`), and state
`--verify-inputs`'s rule for directory operands explicitly: *"a directory operand is covered iff the
enumeration walks that directory with the same exclusions"*. Then §4.2's `**/__pycache__` and §4.3's
input set agree by construction, and adding a fixture under `tests/` changes the hash automatically.
Settle this before writing `image-tag.sh` (S2); it is one function and one check.

#### M-5 — V-11's done-condition cannot pass where §5 sequences it, and following its cleanup instruction literally would kill the live session's own MCP server

**Evidence:** V-11 — `docker ps -a --filter label=cpg-mcp=1 …`, *"Done-condition: empty"*, *"Run it
after three session restarts"*, *"capture its `docker inspect …` before cleaning it up"*; S7
sequences V-10 (in-session `/mcp` + a real query) **then** V-11; §3.10 puts `--label cpg-mcp=1` on
the launch `docker run`; §3.7 describes the container as owning *"a stdio pipe for the lifetime of
one session"*.

**Why it matters:** the containerized server is a **session-lifetime process**. Inside a live
session — the only place V-10 can be run, and therefore where V-11 lands — the current session's own
`cpg-mcp` container is `Up` and carries the label. So the filter is **never empty on a healthy
system**, the done-condition fails by construction, and the step then instructs the implementer to
capture and clean up "the survivor" — which for the running entry is the live session's MCP server.
An agent implementer following the text will `docker stop` the very tool it just proved in V-10, and
then (stdio servers are not auto-reconnected) will need a session restart to get it back. E-23e's
*"the label filter showed nothing left"* was measured with no session open, which is why the gap did
not surface.

**Suggested improvement:** split the condition by status and by liveness:
*"With N sessions open, expect exactly N containers in `Up` and **zero** in `Exited`/`Created`. Then
close every Claude Code session and re-run: expect empty."* Run it from a plain shell, not from
inside a session, and put a caveat next to the cleanup instruction — **never stop a labelled
container while any session is open**; identify an orphan by `Exited` status or by an
`Up` count that exceeds the number of open sessions. That keeps M-3's "checked property, not an
assumption" intent while making it satisfiable.

#### M-6 — V-6/V-6b/V-7 require four responses, but the EOF gotcha the same section documents guarantees only three

**Evidence:** V-6 *"Expect four JSON-RPC responses on stdout"*; V-6b *"all four responses still
arrive, **including the `initialize` reply (id 1)**"*; V-7 done-condition 2 *"The `initialize`
response (id 1) is present, **and all four ids are**"*. And V-6's own note, whose source I read:
`claude/coder/kaizen/inbox.md` — *"A FastMCP stdio server drops the response to **the LAST
request** when the client closes stdin immediately … Adding a trailing throwaway message
(`{"method":"ping"}`) made the previously-lost response appear. EOF on stdin tears the anyio session
down before the last write flushes."*

**Why it matters:** with the trailing `ping` (id 4) as the last request, **the reply that gets
dropped is the ping's own**. The guaranteed stdout is ids 1, 2, 3 — *three* responses. So the
completeness check that v2 added specifically to close M-1 will fail on a perfectly healthy run, for
the documented reason, in the one step (V-6b) that exercises a build inside the launch path. The
implementer's two exits are both bad: chase a phantom stdin-theft bug, or relax the assertion and
lose the M-1 guard I asked for.

**Suggested improvement:** make the sacrificial message explicit rather than accidental. Either
(a) require **ids 1, 2, 3** and state that id 4's reply is expected to be eaten by EOF — naming it,
so nobody investigates it; or better (b) append a **second** throwaway (`ping` id 5) so ids 1–4 are
guaranteed and the dropped reply is the extra one. Then say the same thing in all three places
(V-6's "expect", V-6b's done-condition, V-7's done-condition 2) — they currently phrase it three
slightly different ways.

### Minor

#### m-11 — "immutable, content-addressed" covers the repo bytes only; the base image and pip resolution sit outside the hash

**Evidence:** §3.6's table (`cpg-mcp:<hash12>` — *"**immutable** — content-addressed"*); §3.3 —
*"a hit is a **proof** … that the image was built from exactly the bytes now on disk"*; §3.7 (base
tag deliberately not digest-pinned); §4.4 step 4 (`docker pull` the base on **every** build);
`requirements.txt` pins are ranges (`mcp>=1.28,<1.29`, `falkordb>=1.6,<1.7` — verified).

**Why it matters:** two builds of the same tree a month apart can produce **different images under
the same tag** (a moved `python:3.12-slim`, a new `mcp` patch release), and the later `build.sh` will
silently re-point the "immutable" tag. Conversely — and more likely — once an image exists, *nothing
ever refreshes its base*: a hash hit will happily serve an image built on a base with a
since-patched CVE, forever. Neither is a defect for a dev-only local tool, but "immutable" is the
word R-10's concurrency argument leans on, and the base-refresh path is not documented anywhere.

**Suggested improvement:** say "immutable **with respect to the tracked build inputs**; the base
image and dependency resolution are deliberately outside the hash (§3.7)", and add one README
housekeeping line: *"to refresh the base image: `docker pull python:3.12-slim && cpg/mcp/build.sh
--no-cache`"*. If the stakeholder ever wants reproducibility rather than convenience, digest-pinning
the base is now cheap — the launch path no longer builds (see open question 3).

#### m-12 — `build.sh`'s idempotence is specified two different ways, and V-2b invokes a flag §4.4 does not define

**Evidence:** §4.4's header — *"Idempotent: **if the tag already exists there is nothing to do**"* —
versus its body, whose steps 4–5 describe an unconditional `docker pull` plus one or two
`docker build` invocations with **no tag-exists early exit**. And V-2b part 2's command:
`cpg/mcp/build.sh --runtime-only --no-cache=false`, where `--no-cache=false` is not among §4.4's
documented flags (`--runtime-only`, `--no-cache`, `--verify-inputs`, `--help`).

**Why it matters:** the choice is visible in behaviour, so an implementer inventing it will get it
wrong half the time. **With** an early exit, `--no-cache` becomes the only way to rebuild an existing
tag (and V-1b's "re-run: no new image ID" is trivially satisfied, which weakens that check). **Without**
it, every `build.sh` re-runs a ~0.3 s no-op build *and* a network `docker pull` — the very
per-invocation registry dependency §3.3 exists to remove, just moved from the launch path to
`build.sh`. The undefined flag will simply error out mid-verification.

**Suggested improvement:** state the rule — recommended: *skip both builds when the target tags
already exist, unless `--no-cache` is given; skip the `docker pull` too when the tag is a hit
(`CPG_MCP_NO_PULL=1` remains the explicit opt-out)* — and fix V-2b part 2's command to
`cpg/mcp/build.sh --runtime-only --no-cache`.

#### m-13 — the miss-branch build failure is the only launch failure with no curated message

**Evidence:** §4.5 — `"$HERE/build.sh" --runtime-only >&2 </dev/null` under `set -euo pipefail`.
Every other failure path gets a curated line naming a fallback: docker absent, daemon unreachable,
`CPG_MCP_NO_AUTOBUILD=1` with no image, and (via `--pull=never`) an absent pinned image.

**Why it matters:** a build that fails because the network is down is, per R-8, the *most likely*
launch failure on a fresh clone or an offline machine — and it is the one that exits with raw
BuildKit output and no hint that `cpg/mcp/run.sh` exists. The plan's own standard for this class of
error is higher.

**Suggested improvement:**

```bash
if ! "$HERE/build.sh" --runtime-only >&2 </dev/null; then
  echo "cpg/mcp/docker-run.sh: build of $IMAGE failed (see output above). Offline? A build needs the network unless python:3.12-slim is in the local image store. Fall back to cpg/mcp/run.sh (host venv), or retry with more startup budget: MCP_TIMEOUT=60000 claude." >&2
  exit 1
fi
```

#### m-14 — §9's documentation list misses `cpg/mcp/run.sh`'s own header, which this change makes false

**Evidence:** `cpg/mcp/run.sh:6` (verified) — *"This is the only path that appears in a harness
config (.mcp.json)"*. §4.5 gives `docker-run.sh` **the same sentence**. §3.5 keeps `run.sh`
*"retained, unchanged"* and §9 has no row for it.

**Why it matters:** after S6, two tracked files will each claim to be the only harness launch path,
and the false one is the file a reader lands on when rollback (§8) points them at `run.sh`. This is
the same class of gap as m-10, which v2 accepted — a stale cross-reference in a file nobody thought
of as documentation.

**Suggested improvement:** add a §9 row: `cpg/mcp/run.sh` — header edit only, *"the Docker-less
variant; `.mcp.json` names `docker-run.sh` (see the container section of the README)"* — and drop
"the only" from `docker-run.sh`'s header. Two lines, no logic change, keeps §3.5's "unchanged"
promise true in substance (nothing executable changes).

#### m-15 — E-20's latency table is not internally consistent, so §3.3 should not claim a latency win

**Evidence:** E-20 / §3.3's table — *"container, `docker run` only **1.42 s**"* versus
*"container + `docker image inspect` — chosen **1.36 s**"*, with the latter described as a
*"Marginal win"* over the build variant.

**Why it matters:** the chosen variant is a strict **superset** of the `docker run`-only variant — it
adds a `docker info` preflight, the hash, and `docker image inspect` — so it cannot be faster except
by noise. That puts run-to-run noise at ≥ 0.06 s and makes 1.36 vs 1.61 s a difference of a couple of
noise widths. The decision is unaffected (§3.3 says outright that latency is not the deciding axis,
which is the correct read), but a claimed latency win that the data cannot support undermines a
section whose whole strength is that it measures instead of guesses.

**Suggested improvement:** report the variants as "≈ 1.4–1.6 s, indistinguishable at n=8", drop
"marginal win" from the comparison table, and state what the measured pipeline included — in
particular whether the `docker info` preflight (a daemon round trip, and a slow one on a Desktop or
remote context) is inside the 1.36 s. V-2 measures the real wrapper, so it will settle this anyway.

#### m-16 — E-19's cold-build arithmetic does not reconcile, and the extrapolation built on it uses the wrong size figure

**Evidence:** E-19 — *"14.15 s wall, of which `pip install` ≈ 5.8 s and the rest is the ~119 MB base
pull at ~50 MB/s"*; §3.3 extrapolates *"at 5 MB/s the pull alone exceeds 20 s"*.

**Why it matters:** 14.15 − 5.8 leaves **8.35 s** for the pull, which at "~50 MB/s" would have been
~2.4 s. And **119 MB is the on-disk size** — I confirmed `docker images python` reports 119 MB *disk
usage* — while the wire transfer for `python:3.12-slim` is ~45 MB compressed, so the observed rate
was nearer 5 MB/s than 50. The headline number (14.15 s, under budget) is plausible and the risk
direction is conservative, so nothing decision-bearing moves; but §3.3's slow-link extrapolation is
derived from a rate the same row contradicts, and R-1's residual risk is stated in terms of it.

**Suggested improvement:** drop the MB/s attribution and state only what was measured (total, and
the `pip install` component). Replace the extrapolation with the honest form: *"the base pull is
~45 MB compressed; on a link an order of magnitude slower than this one the cold build exceeds the
30 s budget, and `build.sh` is then required rather than optional."* Also label E-19 explicitly
*"not reproducible from the current machine state"* (§2's residue note implies it; the row should
say it).

#### m-17 — V-2b part 2 prescribes removing a base image shared with `falkor-chat`

**Evidence:** V-2b part 2 — *"`cpg/mcp/build.sh --runtime-only …` after `docker rmi
python:3.12-slim`"*. And `falkor-chat/Dockerfile:10` (verified) — `FROM python:3.12-slim`.

**Why it matters:** that `docker rmi` is a shared-state mutation of exactly the kind the plan
otherwise discloses and approval-gates: the next `falkorchat:dev` build re-pulls ~45 MB, and on an
offline machine it fails outright. It is regenerable, so this is a sequencing-and-disclosure point,
not a veto — but the plan holds itself to disclosing the `docker builder prune` for less.

**Suggested improvement:** mark V-2b part 2 as touching shared state (same convention as §2's prune
disclosure), make `docker pull python:3.12-slim` a mandatory closing step of the probe rather than an
incidental one, and note that part 2 is **optional** — E-18 already establishes the behaviour, and
part 1 (the launch path must work offline) is the property that actually needs proving.

### Nits

- **n-5** — §4.5's header promises *"Every helper is invoked with `</dev/null`"*, but
  `cpg_mcp_image_tag` (line ~747) is the one call without it. It is a sourced shell function that
  forks nothing and reads no stdin, so this is cosmetic — either add the redirect or name the
  exception, so the invariant reads as absolute.
- **n-6** — §9's root-`AGENTS.md` row cites *"the `cpg/` structure bullet (line ~30)"*; it is at
  **lines 20–24**. The `"Key commands"` citation (124–127) is exact, as are all the
  `cpg/mcp/README.md` ones.
- **n-7** — V-7's `grep -c '"id":1'` counts *lines containing* the substring, so it would also match
  `"id":10`+ if the probe pipeline ever grows past nine messages. `'"id":1,'` is more robust.
- **n-8** — §2's E-16 row is now only a pointer ("superseded by E-18/E-19/E-20/E-21"). E-14 handles
  the same situation better by keeping the original claim inline and marking it *"amended in v2"*.
  Doing the same for E-16 would keep the audit trail readable without diffing against v1.

## 12. Scope and documentation impact — re-checked

- **C-310 is still not absorbed.** §3.9 records the effect and adds the useful new observation that
  `MCP_TIMEOUT` is a Claude-Code knob, so OpenCode/Kiro budgets are C-310's problem. "Do not
  renumber" is repeated. No OpenCode/Kiro config is written. ✅
- **`falkor-chat/scripts/start_falkordb.sh` is untouched**, and §3.1's D-a rejection is the reason
  it stays that way. The working tree confirms it. ✅
- **`server.py` and the tool contract untouched**, with the one exception (the `getpid()` scratch
  name) named in §1 and filed rather than fixed. ✅
- **C-321 does not collide** — C-301…C-319 are contiguous and neither C-320 nor C-321 appears
  anywhere tracked. Its content (`uuid4().hex[:8]` at `tests/test_server.py:472`, owner
  `tdd-engineer`/`coder`) is correctly scoped out of this plan and correctly worked around in V-4. ✅
- **Doc-impact list:** complete except **m-14** (`cpg/mcp/run.sh`'s header). The two additions v2
  made — C-321 and README line 269 — are both correct and both verified against the source. The
  `skills/cpg-analysis/SKILL.md` "no change" still holds. The `docs/plans/cpg-query-access.md` "no
  edit, this note supersedes only its §4.2" call is the right one.
- **New in v2 and correctly handled:** the plan measured with throwaway images and then removed
  them, disclosed its residue in enough detail that I could verify every item of it, and states
  plainly that measuring was in bounds while shipping was not. That is the right line and it held.

## 13. What's solid (v2)

- **§11 is a real review response, not a compliance table.** All 17 Part I items are dispositioned,
  I checked each against the plan body, and in every case the body does what the row claims. Three
  suggestions were improved on rather than merely accepted (`exec </dev/null` over per-call
  redirects; `chown` over `-p no:cacheprovider`; the shared test/runtime hash over a remembered
  precondition), and two were rejected with better arguments than mine (`-p no:cacheprovider`,
  `--memory/--cpus`).
- **The plan disproved its own load-bearing claim and changed mechanism on the evidence.** R-8 went
  from *"warm builds are offline"* to a precise, correct statement, and §3.3 was rebuilt as a
  four-axis comparison that names the deciding axes (offline, concurrency) and explicitly declines
  to decide on the one that looked decisive (latency). It also concedes where v1's *reasoning* was
  wrong (the mtime/`git checkout` argument) even while keeping the same conclusion.
- **§3.10 is measured, not reasoned.** E-23's four-path shutdown table located the orphan window
  precisely and made it *narrower* than my Part I speculation — and it still adopts `--init`,
  because PID-1 `python` ignoring `SIGTERM` is wrong on its own merits. The `--name` rejection
  (a fixed name turns benign concurrent duplication into a hard failure at session start) is a point
  I missed.
- **`--verify-inputs` is the right instinct** — the content-hash choice buys one new coupling, the
  plan names it (R-11), and then enforces it mechanically instead of with a comment. M-4 is about
  the rule being incomplete, not about the idea being wrong.
- **Curated failure messages, `--pull=never`, and E-27** together turn three separate opaque startup
  failures into legible ones. That is disproportionate value for the number of lines it costs.
- **Honesty about what is *not* guaranteed:** the fresh-clone restatement ("a self-healing safety net
  whose success on a slow link is not guaranteed, not the documented path") is exactly what I asked
  for, and E-28 supplies a fact I had assumed against.
- Everything Part I §4 listed as solid — D-net, the blast-radius discipline, D-test's split, the
  `.mcp.json` shape, D-host, the justified omissions, rollback — survives v2 unchanged.

## 14. Open questions

1. **Is `MCP_TIMEOUT` really the startup knob, or is it `MCP_CONNECT_TIMEOUT_MS`?** The prose on
   two official pages says `MCP_TIMEOUT` and 30 s (§9 above); the env-vars reference table appears
   to say the opposite. This changes **no decision** — the margin is ≥ 3.7× under every reading —
   but it changes one README line and one troubleshooting recipe, and a user who raises the wrong
   variable for a slow cold build will see no effect. It is settleable in one command during S7:
   `MCP_TIMEOUT=1 claude` and check whether `/mcp` reports `cpg` as failed; if not, try
   `MCP_CONNECT_TIMEOUT_MS=1`. Document whichever bites. **Verify during implementation, not before.**
2. **Container-only eventually?** (Part I Q2, §10 item 3.) Unchanged: a stakeholder preference the
   plan handles correctly by recommending "keep both, revisit later".
3. **Should the base image be digest-pinned now?** §3.7 defers to repo precedent, which was the
   right call while the launch path built on every start. It no longer does, so the trade-off has
   moved: a digest pin would close m-11's "same tag, different base" wobble and make the hash a true
   image identity, at the cost of a manual bump and a divergence from `falkor-chat/Dockerfile`. Not
   blocking; worth one sentence in §3.7 acknowledging that the reasoning changed even if the answer
   does not.

---

# Part III — delivered-change review (`c38f7dc`)

> Reviewer: `analyst`. 2026-07-26. Review of the **implemented and committed** change, not the plan.
> Artifact: commit `c38f7dc` ("feat(cpg): containerize the MCP server — content-hash image, host
> venv retained"), 15 files. Author `devops`, who verified its own work; this is the independent pass.
> Findings route back to `devops`. This document changes nothing.

## 15. Scope & verdict (Part III)

**Reviewed, as code:** `cpg/mcp/Dockerfile`, `cpg/mcp/.dockerignore`, `cpg/mcp/image-tag.sh`,
`cpg/mcp/build.sh`, `cpg/mcp/docker-run.sh`, `.mcp.json`, `cpg/mcp/run.sh` (header).
**Reviewed, as documentation, against the code rather than on its word:** `cpg/mcp/README.md`
(the whole container section), root `AGENTS.md` (the `cpg/` bullet + "Key commands"),
`docs/HISTORY.md`, `docs/BACKLOG.md` (C-310 amendment, C-320, C-321), and
`docs/plans/cpg-mcp-containerization.md` §12 against what was actually committed.
**Also swept for staleness:** every tracked file naming `cpg/mcp/run.sh` or the MCP wiring.

**Explicitly out of scope, because `teco` executed it directly after the commit and I treat it as
established:** the MCP handshake through the wired command, `tools/list` = 1 tool with
`readOnlyHint`, a live `tools/call`, orphan-container check, `claude mcp list`, the `falkordb-dev`
untouched-shared-state check, the host suite (53/7), the live suite (7), the in-image suite (53/7),
`audit-team.sh`, and the personal-identifier grep. I did not re-run any of these.

**What I ran myself** (all read-only; the repo working tree is unchanged — `git status` clean at
finish). Adversarial probes of `image-tag.sh` and `build.sh --verify-inputs` against a **copy** of
`cpg/mcp/` in the session scratchpad, never the tracked tree; live confirmation of the three
implementation-time defect fixes; `build.sh` stdout/stdin purity; `docker image ls`.

**Verdict: approve with suggestions.** No blocker. The change does what it says, the three defects
`devops` reports fixing are genuinely fixed in the committed code (I reproduced all three), and the
documentation is unusually accurate — I went looking for prose the code does not back and found
three small instances in ~600 lines of new docs. The two majors below are both about the *guard*
around the load-bearing safety property rather than the property itself: **M-7** is a hole in
`--verify-inputs` that a normal Dockerfile edit walks straight through, and **M-8** is an unbounded
network call that the miss branch puts inside the MCP startup budget. Neither breaks anything as
shipped.

Numbering continues from Part I/II (majors `M-`, minors `m-`, nits `n-`).

## 16. Verification of the three implementation-time defect fixes

All three were reproduced end-to-end. Each is real, and each fix is correct.

| Defect | Premise re-measured | Fix in the committed code | Verdict |
|---|---|---|---|
| **(a)** bare `-e VAR` unsets the image `ENV` | `env -u FALKORDB_HOST docker run --rm -e FALKORDB_HOST cpg-mcp:ba910c48571d python -c "…os.environ.get('FALKORDB_HOST')"` → **`None`**. Same run with **no** `-e` flag → **`'host.docker.internal'`**. Docker server 29.6.1. | `docker-run.sh:104-110` — `ENV_ARGS` built only for variables that pass `[ -n "${!_v+set}" ]`, then expanded with the `${arr[@]+…}` empty-array guard at `:131`. | **Correct.** The `+set` test (rather than `-n "${!_v}"`) also preserves a deliberately-empty value, which the inline comment claims and which the semantics do deliver. |
| **(b)** `CPG_MCP_IMAGE` fell into the autobuild branch | `CPG_MCP_IMAGE=cpg-mcp:definitely-not-here ./cpg/mcp/docker-run.sh </dev/null` → the four curated lines, **exit 1**, nothing built. | `docker-run.sh:48-51` sets `PINNED=1`; `:66-75` short-circuits before both the `CPG_MCP_NO_AUTOBUILD` test and the build. | **Correct**, and the ordering is right: pinned beats no-autobuild, which is the only sensible precedence. Also confirmed `CPG_MCP_IMAGE_REPO=cpg-mcp-nope CPG_MCP_NO_AUTOBUILD=1 …` → the *other* curated message, exit 1. |
| **(c)** early exit skipped the `:dev`/`:test` re-tag | `bash cpg/mcp/build.sh </dev/null` on a warm tree → *"already built … nothing to build"* + *"Aliases re-pointed: cpg-mcp:dev -> ba910c48571d, cpg-mcp:test -> test-ba910c48571d"*, **0 bytes on stdout**, exit 0. | `build.sh:167-181`. The `RUNTIME_ONLY` guard on the `:test` re-tag is correct — `have_image "$TEST_TAG"` is a precondition of reaching that line when `RUNTIME_ONLY=0`. | **Correct**, and the reported message honestly says only `:dev` was re-pointed under `--runtime-only`. |

The `pipefail`+early-`return` bug `devops` reports finding in its own first `build.sh` is fixed and,
more usefully, **documented at the site** (`build.sh:83-93`) with the reason process substitution is
used instead of a pipeline. I audited every other early-returning loop for the same shape:
`cpg_mcp_image_tag`'s missing-input loop (`image-tag.sh:54-61`) is also fed by process substitution
and is safe; there is no second instance of the bug.

I also confirmed **invariant 2** (nothing before `exec` reads stdin) holds by inspection of every
forking call: `docker info … </dev/null` (`:42`), `docker image inspect … </dev/null` (`:65`),
`build.sh … </dev/null` (`:84`) *plus* `build.sh`'s own `exec </dev/null` (`:3`), and inside
`cpg_mcp_image_tag` every loop is `< <(…)` and every digest is `sha256sum < "$f"`. The one call
without a redirect is named in the header comment and genuinely forks nothing. **Invariant 1**
(stdout purity) I checked at the two places a plan reader would not expect it to hold:
`build.sh` warm early-exit → **0 stdout bytes**; `build.sh --help` → **0 stdout bytes**.

## 17. Findings — Major

### M-7 — `--verify-inputs` is blind to line-continued `COPY`, and answers "OK" — must fix

**Evidence.** `build.sh:127-132` parses the Dockerfile with

```awk
toupper($1) == "COPY" { … for (i = 2; i <= NF; i++) if ($i !~ /^--/) { n++; f[n] = $i }
                        for (i = 1; i < n; i++) print f[i] }
```

This is line-oriented, so a `\`-continued `COPY` is invisible to it. Demonstrated on a scratchpad
copy of `cpg/mcp/`, appending to the Dockerfile and running the shipped `build.sh --verify-inputs`:

| Appended to the Dockerfile | `--verify-inputs` says |
|---|---|
| `COPY requirements.txt \`⏎`     setup.sh /app/` | `--verify-inputs OK` (`setup.sh` **not** covered by the hash) |
| `COPY \`⏎`     setup.sh \`⏎`     run.sh /app/` | `--verify-inputs OK` (**neither** covered) |
| `COPY setup.sh setup.sh` (single line — the control) | correctly **fails**: *"COPYs the FILE 'setup.sh', which is not in image-tag.sh's input list"* |

**Why it matters.** `--verify-inputs` is the *only* mechanism enforcing the invariant that three
separate files state as absolute — `Dockerfile:10-13`, `image-tag.sh:13-16`, `README.md:319-320`
("fails the build if the Dockerfile ever `COPY`s something the hash does not cover, so the two
cannot drift"). A false **pass** is the one direction that costs correctness: the file lands in the
image, the hash does not move, `docker image inspect` hits, and the launch path serves an image that
does not contain the change — precisely the failure the content hash exists to make unrepresentable.
Multi-line `COPY` is an ordinary Dockerfile idiom, so this fires on the first Dockerfile edit that
uses it, silently, with an "OK" line in the log. Nothing is wrong *today* (the committed Dockerfile
has no continued `COPY`, and I confirmed `--verify-inputs` passes on it) — this is a trap laid for
the next editor.

**Suggested improvement.** Join continuations before parsing. A three-line `awk` prelude is enough
and keeps the current structure:

```awk
{ line = line $0 }
/\\[[:space:]]*$/ { sub(/\\[[:space:]]*$/, " ", line); next }
{ $0 = line; line = "" }        # then the existing toupper($1)=="COPY" rule
```

Add a regression case to whatever S3 becomes: a `\`-continued `COPY` of an uncovered file **must**
make `--verify-inputs` exit non-zero. Given how much of the design rests on this one check, that
case is worth having as a checked-in probe rather than a remembered manual step.

### M-8 — the miss branch puts an unbounded `docker pull` inside the 30 s startup budget, and is triggered by inputs that cannot change the runtime image — worth doing

**Evidence, two halves that compound.**

1. `docker-run.sh:84` calls `"$HERE/build.sh" --runtime-only` on a hash miss. With the runtime tag
   absent, `build.sh` falls past the early exit and reaches `:191-196`:
   `if [ "${CPG_MCP_NO_PULL:-0}" != "1" ]; then docker pull -q "$BASE_IMAGE" …`. `docker-run.sh`
   does not set `CPG_MCP_NO_PULL`, so **every autobuild performs a Docker Hub pull with no
   timeout**, followed by a `docker build` that also resolves `FROM` metadata over the network if
   the base is not in the local store.
2. The hash covers `requirements-dev.txt`, `pytest.ini` and every file under `tests/`
   (`image-tag.sh:30`, `:38`; confirmed by running `cpg_mcp_input_files` — 8 operands), but the
   **runtime** stage COPYs none of them (`Dockerfile:51-53` = `FROM base` + `USER` + `CMD`; `base`
   carries only `requirements.txt` and `server.py`). So editing a single line of
   `tests/test_server.py` invalidates the launch image and forces the miss branch, to produce a
   byte-identical runtime image.

That is not hypothetical: **C-321, the very next queued item, edits `tests/test_server.py:472`.** The
first session start after it lands will pull and build.

**Why it matters.** The design's headline property — and the reason M-2 was reversed in Part II — is
that the launch path is offline and local. It is, on a **hit**. On a **miss** it is neither, and the
miss is reachable by a change that provably cannot affect the artifact being launched. The blast
radius is bounded (non-blocking MCP startup since 2.1.142, a curated fallback message at
`docker-run.sh:85-89`, `MCP_TIMEOUT=60000`), so this is a degradation, not a break — but it is a
degradation on the exact axis the design was chosen for, and `README.md:376-381` is the only place a
reader learns a build needs the network.

**Suggested improvement**, cheapest first:

- **(a)** Have `docker-run.sh:84` invoke the build as
  `CPG_MCP_NO_PULL=1 "$HERE/build.sh" --runtime-only …`. In the steady state the base *is* in the
  local store (`build.sh` put it there on the last interactive build), so this makes the common
  miss fully local and leaves the explicit `cpg/mcp/build.sh` as the thing that refreshes the store.
  Pair it with one added line in the existing curated failure message: "…or run
  `cpg/mcp/build.sh` once with a network connection."
- **(b)** Say the consequence out loud where it bites. `README.md:16-17` currently reads "once per
  clone (or after any code change)"; make it explicit that a **test** edit also invalidates the
  launch image, and that `cpg/mcp/build.sh` before restarting the session is how you avoid paying
  for it at session start.
- **(c)** The deeper fix, only if (a)+(b) prove insufficient: split the digest — a runtime-input
  hash (`Dockerfile`, `.dockerignore`, `requirements.txt`, `server.py`) for `cpg-mcp:<hash>`, the
  full hash for `cpg-mcp:test-<hash>`. This costs the elegant "same hash on both tags" property that
  closed Part I's m-7, so I would not reach for it first.

## 18. Findings — Minor

The hash is the load-bearing safety property, so I probed it adversarially. Five gaps; all of them
are narrow, and I record each with the exact probe so `devops` can judge rather than take my word.
`m-18`, `m-19` and `m-22` are **staleness** direction (a changed tree can hit an old tag); `m-20` and
`m-21` are the safe direction.

### m-18 — a file-mode change does not move the hash, but does change the image

**Evidence.** Scratchpad copy: `cpg_mcp_image_tag` → `ba910c48571d`; `chmod +x server.py`;
`cpg_mcp_image_tag` → **`ba910c48571d`**, unchanged. `image-tag.sh:63-68` digests `sha256sum < "$f"`
plus the relative path, and nothing else. Docker `COPY` **preserves** the executable bit, so the two
trees produce different images under the same tag.

**Why it matters.** It falsifies `README.md:314-315` ("a hit *proves* the image was built from
exactly the code on disk. Staleness is not merely unlikely, it is unrepresentable") as an absolute.
Git tracks the mode bit, so a committed `chmod +x` is a change a colleague can pull. The practical
blast radius here is nil — nothing in the image is invoked by its mode (`CMD` is
`python server.py`) — which is why this is minor and not major.

**Suggested improvement.** One line in the digest stream, e.g. contribute `x` or `-` per file:
`if [ -x "$CPG_MCP_DIR/$f" ]; then printf 'x\0'; else printf -- '-\0'; fi` next to the existing
`printf '%s\0' "$f"`. Avoid `stat -c %a` — it is GNU-only and would break the machine-independence
the header explicitly claims. Alternatively, weaken the README sentence to "the *contents* of the
code on disk".

### m-19 — a symlink under a walked directory is invisible to the hash

**Evidence.** Scratchpad: `ln -s <outside-target> tests/linked.py` → `cpg_mcp_image_tag` returns
**`ba910c48571d`**, identical to the tree without it. `image-tag.sh:44` walks with `find … -type f`,
which excludes `-type l`. Docker's build context tars symlinks *as symlinks*, so `COPY tests tests`
does put an entry in the image that the hash never saw.

**Why it matters.** Same class as m-18: a tracked change that alters the image without moving the
tag. No symlinks exist under `cpg/mcp/tests/` today, so likelihood is low.

**Suggested improvement.** `find "$d" \( -type f -o -type l \) …` and contribute the link target for
a symlink (`printf '%s\0' "$(readlink "$f")"` in place of the digest), or — if symlinks under a
walked directory are simply not wanted — make the walk **fail** on encountering one, matching the
"hard error rather than a silent skip" stance the file takes for missing files.

### m-20 — a missing walked directory is silently skipped, where a missing file is a hard error; two different trees collide

**Evidence.** `image-tag.sh:42` — `[ -d "$CPG_MCP_DIR/$d" ] || continue`, versus `:55-60` which
`return 3`s with a curated message for an absent file. Demonstrated collision on a scratchpad copy:

```
tests/ moved away entirely  -> cpg_mcp_image_tag rc=0, tag=f06da2a82b7f
tests/ present but empty    -> cpg_mcp_image_tag rc=0, tag=f06da2a82b7f   # same tag, different tree
```

**Why it matters.** `image-tag.sh:49-50` states the rule this violates in its own words: *"Returns
non-zero … rather than hashing an absent input: silently skipping one would let two different trees
collide on a tag."* The directory operand does exactly the silent skip the comment forbids. In
practice the harm is small — `--verify-inputs` correctly refuses to *build* the missing-`tests/`
state (verified: *"COPYs 'tests', which does not exist in the build context"*, exit 1), and the two
colliding states produce identical **runtime** images anyway. It is the stated invariant that is
broken, not the system.

**Suggested improvement.** Replace the `continue` with the same treatment the file gives a missing
file — an error naming the directory and pointing at `cpg_mcp_input_dirs` and the Dockerfile's
`COPY` list.

### m-21 — the walk's exclusions do not match `.dockerignore`, contradicting three places that say they do

**Evidence.** `.dockerignore:9-11` excludes `**/__pycache__`, `**/*.pyc` **and `**/.pytest_cache`**;
`image-tag.sh:44` excludes only `! -path '*/__pycache__/*' ! -name '*.pyc'`. Demonstrated: creating
`tests/.pytest_cache/CACHEDIR.TAG` on a scratchpad copy moves the tag `ba910c48571d` →
`2bdb7b923951`, although that file can never reach the image. The claim that they agree is asserted
in `.dockerignore:5-8` ("image-tag.sh's directory walk applies the same exclusions … Keep the two in
step"), in `image-tag.sh:15-16`, and in plan §12's M-4 row ("walks each with `.dockerignore`'s
exclusions").

**Why it matters.** Failure direction is safe (a spurious rebuild, never a stale image), and
`.pytest_cache` normally lands at the rootdir `cpg/mcp/` rather than under `tests/` — so the live
likelihood is low. What is not low-value is the comment: a future maintainer adding a pattern to
`.dockerignore` will read three files telling them the walk already mirrors it, and will not add it
to `image-tag.sh`. The *next* pattern may not be safe-direction.

**Suggested improvement.** Add `! -path '*/.pytest_cache/*'` to `image-tag.sh:44` to make the claim
true today, and put the two exclusion sets adjacent in one comment so drift is visible in a single
diff hunk.

### m-22 — a failed `find` is unobserved, so a partially-readable tree hashes as a smaller one

**Evidence.** `image-tag.sh:43-45` runs `( cd … && find … | sort -z )` inside a function consumed
via `< <(…)`; a process-substitution exit status is never examined, and the outer digest pipeline's
`pipefail` does not see it either. Scratchpad probe: `mkdir tests/sub; echo x > tests/sub/f.py;
chmod 000 tests/sub` → `find` prints *"Permission denied"* to stderr, `cpg_mcp_image_tag` returns
**0** with tag **`ba910c48571d`** — identical to the tree that never had `tests/sub` at all.

**Why it matters.** Same "two trees, one tag" class as m-20. Bounded by the fact that a *build* of
that tree would fail when Docker tars the context, so the collision is only reachable as a **hit** on
an image built earlier; and the `find` error is on stderr, which the harness logs. Low likelihood on
a single-developer box.

**Suggested improvement.** Capture the enumeration into a variable and check the status, e.g.
`files="$(cpg_mcp_input_files)" || return 3` before the digest loop, so a truncated enumeration is a
hard error like an absent file.

### m-23 — `--no-cache` can make `cpg-mcp:test-<hash>` and `cpg-mcp:<hash>` carry different resolved dependencies

**Evidence.** `build.sh:202-209` issues **two independent `docker build` invocations**, each
inheriting `CACHE_FLAG`. `requirements.txt` pins **ranges** (`mcp>=1.28,<1.29`, `falkordb>=1.6,<1.7`)
and `requirements-dev.txt` starts with `-r requirements.txt`, so under `--no-cache` the base stage's
`pip install` runs twice and a patch release landing between the two resolves differently.

**Why it matters.** `AGENTS.md:137` calls the in-image run "the gate that keeps the two paths from
drifting" and `README.md:405` says the container gate proves "its resolved dependencies". Under
`--no-cache` — which is exactly the command documented for a base-image refresh
(`README.md:387`) — the gate can be testing a dependency set the launch image does not have. Narrow
window, real mechanism.

**Suggested improvement.** Under `--no-cache`, build the `test` target **first** with `--no-cache`
and then build `runtime` **without** it, so the runtime image reuses the base layers the test image
was just validated on. Two swapped lines, and it makes the "same base" property true by
construction rather than by timing.

### m-24 — no `--cap-drop=ALL --security-opt no-new-privileges`, and it was never considered

**Evidence.** `docker-run.sh:126-132` — the flag set is `-i --rm --init --label --pull=never
--read-only --tmpfs /tmp --add-host`. Grepping the plan and Parts I/II for `cap-drop`,
`no-new-privileges`, `security-opt` and `capabilit` returns nothing on the design axis, so this is an
omission rather than a rejected option (unlike `--memory`/`--cpus`, which Part I's n-1…n-4 argued
down well).

**Why it matters.** The container is non-root under `--read-only` and makes exactly one outbound TCP
connection; every default capability is dead weight. `--cap-drop=ALL --security-opt
no-new-privileges` costs nothing measurable and cannot break this workload (no `bind`, no `chown`,
no `setuid`). It is the cheapest remaining hardening and fits the "least privilege" framing the
launch-flags table already uses.

**Suggested improvement.** Add both flags, and add a row to `README.md:347-354`'s table — including
that they are the *second* thing to drop, after `--read-only --tmpfs /tmp`, if the server ever fails
at session start.

### m-25 — `docs/BACKLOG.md` C-320 claims the launch path makes no registry contact, unqualified

**Evidence.** `docs/BACKLOG.md:304` — *"…gated by `docker image inspect`, so the launch path makes
**no registry contact** and a stale image is unrepresentable; a miss builds."* The miss branch pulls
(`build.sh:191-196`) and builds, both of which contact a registry — see M-8. `README.md:312` scopes
the same claim correctly (it is attached to the `docker image inspect` call and is qualified at
`:376-381`), and `docs/HISTORY.md` is likewise careful. BACKLOG is the one place the claim is stated
without its scope.

**Why it matters.** C-320 is a closed backlog entry — the thing a reader consults a year from now to
learn what the property actually is. Stating it unqualified next to "a miss builds" makes the two
clauses contradict each other for anyone who thinks about it, and misleads anyone who does not.

**Suggested improvement.** *"…so the launch path makes no registry contact **on a hit**, and a stale
image is unrepresentable; a miss builds, which does need the network."*

### m-26 — `docs/test-plans/cpg-query-access.md` still records the wiring as `run.sh`

**Evidence.** `docs/test-plans/cpg-query-access.md:78` — the environment table row: *"MCP wiring |
repo-root `.mcp.json`, project scope, `bash -c 'exec "$CLAUDE_PROJECT_DIR/cpg/mcp/run.sh"'`…"*. The
plan's §9 documentation-impact table enumerates every other doc and does not mention `docs/test-plans/`.

**Why it matters.** Per the repo's module-doc convention (root `AGENTS.md`), `docs/test-plans/` holds
**active** documents; `docs/archive/` does not exist yet. So a QA engineer re-running this plan would
set up an environment that no longer exists. The companion `docs/test-reports/cpg-query-access-report.md:49`
records the same string, but a *report* is a dated record of a past run and is correct as written —
only the plan is the problem. `docs/HISTORY.md:83` is likewise a dated entry and correctly left alone.

**Suggested improvement.** Either add a one-line "superseded by C-320: the wiring is now
`docker-run.sh`" note to that row, or (cleaner, and the convention's own answer) archive the M3
test-plan/test-report pair now that M3 has closed, fixing inbound links in the same change. The
second is arguably `teco`'s call rather than `devops`'s.

### m-27 — `--verify-inputs` mis-parses `COPY --from=` and the JSON-array form (safe direction, bad diagnostics)

**Evidence.** Same scratchpad harness as M-7, appending to the Dockerfile:

| Appended | `--verify-inputs` says |
|---|---|
| `COPY --from=base /app/server.py /app/server2.py` | *"COPYs '/app/server.py', which does not exist in the build context"* → refuses to build |
| `COPY ["setup.sh", "/app/setup.sh"]` | *"COPYs '\[\"setup.sh\",', which does not exist in the build context"* → refuses to build |

**Why it matters.** Both fail closed, so no correctness cost — the opposite of M-7. But this is a
**multi-stage** Dockerfile, where `COPY --from=<stage>` is the natural next edit (e.g. copying a
built artifact from `base` into `runtime`), and it will be rejected with a message that sends the
reader hunting for a missing file in the build context. A wrong diagnostic on a legitimate pattern
costs debugging time and invites someone to weaken the check.

**Suggested improvement.** Skip `COPY` instructions carrying a `--from=` flag entirely (their sources
are stage-internal, not build-context paths) and say so in the awk comment; and either handle the
JSON-array form or make the error text name the form it could not parse instead of claiming the file
is missing.

## 19. Findings — Nits

- **n-9** — `build.sh:42`'s usage text says `--verify-inputs` is *"Run implicitly before every
  build"*. True (`:184`), but the warm early exit at `:167-181` returns before it, which reads as an
  exception to an absolute. One clause — "before every build; the early exit performs none" — would
  settle it. Zero behavioural consequence, since a Dockerfile edit necessarily moves the hash and so
  cannot reach the early exit.
- **n-10** — `docker-run.sh:64`'s comment, *"~0.05 s, purely local — no registry contact in any
  circumstance"*, sits three lines above the branch that pulls and builds. It is scoped to the
  `inspect`, and a careful reader will see that; "no registry contact **on a hit**" would remove the
  need for care. Same wording as m-25's fix.
- **n-11** *(environment note, not a finding against the change)* — on this box `find` is shadowed
  by a shell function dispatching to `bfs`, which traverses breadth-first. `image-tag.sh` is immune
  because `LC_ALL=C sort -z` normalises order afterwards — that line is doing more work than its
  comment claims, and is worth keeping for that reason too. Filed to the analyst learnings inbox.

## 20. Conformance — does plan §12 match what was committed?

I checked every §12 row against the code rather than against its own prose. **All settle-items and
all taken minors are genuinely in the committed artefacts.** One row overstates.

| §12 row | In the code? |
|---|---|
| **M-4** — directory-driven enumeration + one `--verify-inputs` rule per operand kind | **Yes**, `image-tag.sh:29-47` and `build.sh:95-141`. I re-ran the S3 done-condition myself: passes on the real tree; fails on an uncovered file operand *and* on an uncovered directory operand. **One overstatement:** the row says the walk applies *"`.dockerignore`'s exclusions"*; it applies two of the three — **m-21**. And the check itself has the M-7 hole. |
| **M-5** — V-11 split by status and liveness, orphan defined, `docker stop` warned against | **Yes**, `README.md:395-400`, including the boxed prohibition. |
| **M-6** — ids 1–3 asserted, id 4 expected absent, in all three places | **Yes**, `README.md:436-451` says it twice (prose + the comment under the probe). The choice of option (a) over my preferred (b) is justified by a measurement I could not have had — the loss is a race, not a fixed rule — and option (a) is the correct response to that. |
| **m-11** — base image + pip resolution outside the hash | **Yes, and well placed.** `image-tag.sh:18-23` (the file a maintainer edits), the boxed note at `README.md:322-327` (the section a reader lands on), and the housekeeping command at `README.md:387`. This answers the "is it stated where a reader will actually hit it?" question affirmatively for both audiences. It is *not* in `build.sh --help`, which I consider fine. |
| **m-13** — curated build-failure message | **Yes**, `docker-run.sh:84-90`, four lines naming the offline cause, `run.sh`, and `MCP_TIMEOUT=60000`. |
| **m-14** — `run.sh` no longer claims to be the only harness path | **Yes**, `cpg/mcp/run.sh:4-11`, and `docker-run.sh:5-7` states the complementary fact. Nothing executable in `run.sh` changed (`git show c38f7dc -- cpg/mcp/run.sh` is header-only), so §3.5's "retained, unchanged" holds. |
| **m-12** — one idempotence rule, plus the alias re-tag refinement | **Yes**, `build.sh:167-181`, verified live (§16 defect (c)). |
| **m-17** — V-2b part 2 not run | **Consistent.** `python:3.12-slim` is intact and `falkordb-dev` untouched per `teco`'s checks. |
| **n-5 / n-6 / n-7** | **Yes** — `docker-run.sh:20-22` names the `cpg_mcp_image_tag` exception; §9 cites `AGENTS.md` lines 20–24; V-7 uses `'"id":1,'`. |
| **n-8 declined** | **Reasonable, and I accept it.** The stated reason — E-16's v1 claim is still present inline, so restructuring buys nothing but diff noise on a row nothing depends on — is the correct trade for a cosmetic nit. Declining a nit with a reason is the behaviour I want; a reviewer who expects 100 % uptake is not reviewing. |

**Did anything the plan promised quietly not ship?** No. I walked §9's documentation-impact table
row by row: the README quick start leads with `build.sh` as *the* supported step (`:16-17`), the
container section exists with the tag scheme, the gate, both escape hatches and `MCP_TIMEOUT`
(`:282-424`), the `.mcp.json` block quoted at `:226-237` matches the committed file byte for byte,
the env-var table carries the two-correct-values note (`:180-190`), the `claude mcp add --scope
local` recipe names `docker-run.sh` (`:521`) with the m-10 caveat, housekeeping and base-refresh are
present (`:385-388`), `AGENTS.md`'s two locations both landed with the two pytest lines untouched,
HISTORY has its dated entry, BACKLOG has C-320/C-321 with C-310 amended and nothing renumbered, and
`skills/cpg-analysis/SKILL.md` correctly needed no change (its only `cpg/mcp` reference is
`:80`, pointing at the README for env defaults). The only §9 gap is a doc §9 never listed — **m-26**.

## 21. What's solid

Short, because the list is long and the point is only to keep it from being churned.

- **The three defect fixes are real fixes to real defects**, each found by `devops` during
  implementation and each reproducible. Defect (a) in particular — bare `-e VAR` *deleting* an
  image `ENV` — is a genuinely surprising Docker behaviour that the plan would have shipped, and the
  fix is the right shape (`+set` rather than `-n`, so an explicit empty value survives).
- **Quoting and word-splitting discipline throughout.** NUL separation end to end (`printf '%s\0'`,
  `find -print0`, `read -r -d ''`, `sort -z`), the `${arr[@]+"${arr[@]}"}` empty-array guard in both
  scripts, every expansion quoted, `HERE`/`CPG_MCP_DIR` resolved from `BASH_SOURCE` so cwd never
  matters. I probed filenames with spaces — the tag computes cleanly. Under `set -euo pipefail` I
  found no `set -e`-swallowing construct: every function that can fail is either called under `if !`
  or intended to abort.
- **The digest is injective over what it does cover.** Contributing the NUL-terminated relative path
  *and* the content digest means two files swapping contents move the tag (probed: `860df103f445`
  → `178564481863`) — a real hazard that path-free or content-free schemes both miss.
- **`--pull=never`, `--init`, no `--name`, the label, `--read-only --tmpfs /tmp`** — each measured
  rather than assumed, each with a one-line reason at the call site, and the read-only pair
  correctly flagged as the first thing to drop.
- **The Dockerfile's layer ordering is right**: `useradd` before the COPYs, `requirements.txt` +
  `pip install` before `server.py`, dev deps and the suite only in the `test` stage. The
  `RUN chown appuser:appuser /app` (`:46`) is the correct fix for Part I's m-1 — it preserves V-3's
  strong byte-identical claim instead of weakening the assertion.
- **The documentation is the best part of this change.** I read it *against* the code specifically
  looking for prose the implementation does not back, and found three small instances (m-21, m-25,
  m-26) in roughly 600 lines of new README/AGENTS/HISTORY/BACKLOG text. The `FALKORDB_HOST`
  "two different correct values" box, the `MCP_TIMEOUT` vs per-tool `"timeout"` distinction (settled
  by live measurement, with the commands shown), the troubleshooting table's DROP-vs-REJECT and
  `0.0.0.0`-bind rows, and the `docker stop` prohibition are all things a reader will need and
  would not have derived.
- **§12 is an honest implementation record**, including the two things that reflect badly on the
  first attempt (the `pipefail` bug, the alias re-tag miss) and the one item declined with a reason.

## 22. Open questions

1. **Does C-321 want to grow a second clause?** It currently fixes `os.getpid()` → `uuid4()`. Because
   the hash covers `tests/`, landing it will also force the M-8 rebuild-on-session-start path. If
   M-8(a)+(b) are taken first, C-321 lands quietly; if not, whoever takes C-321 should expect the
   first session start afterwards to build. Sequencing call for `teco`.
2. **Should the M3 `cpg-query-access` test-plan/test-report pair be archived now?** m-26 is a symptom
   of a milestone-close step the repo's own convention prescribes but that has not happened (there is
   no `docs/archive/` yet). Fixing the one line is `devops`-sized; deciding to archive the pair is
   not.
3. **Part II's open question 3 (digest-pin the base) is still open and is now cheaper to answer.**
   m-18/m-19 show the hash is not quite a full image identity even over the repo bytes; a digest pin
   would close the base half of m-11. Still a stakeholder preference, not a defect — recorded here
   only so it does not get lost now that the plan is closed.
