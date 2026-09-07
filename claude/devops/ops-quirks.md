# Ops quirks — this lab's verified facts

> **Live-verified knowledge base for `devops`.** Facts confirmed by hands-on testing, not just
> docs. Treat each as **verified for the version it names** (Docker 29.6.1/BuildKit unless
> noted) — re-check on a toolchain upgrade.
>
> **This is a cache, not the source of truth.** Origin: distilled 2026-08-11 from the `devops`
> agent's learnings inbox via `agent-maintenance` skill §5. `devops.md` (the always-on prompt)
> points here and stays lean.

## `.mcp.json`'s launch shape can be verified end-to-end from a plain shell, without restarting the harness

Spawning `command`+`args` verbatim (with `CLAUDE_PROJECT_DIR` injected, matching where Claude Code
sets it) and speaking raw JSON-RPC (`initialize` → `notifications/initialized` → real calls) over
the pipes reproduces everything the in-session `/mcp` view would show — server name, tool
count/schema, annotations, a real query result. `claude mcp list` is a second, complementary
out-of-band check: it spawns a **fresh** process that re-reads `.mcp.json` and health-checks each
server, so an edit to the launch command can be proven **without restarting the current session**
(which can't see the change — stdio servers aren't reconnected mid-session). It also settles a
timeout-knob ambiguity where the docs' reference table and prose disagree: `MCP_TIMEOUT` is the
connection/startup budget (30s default); `MCP_CONNECT_TIMEOUT_MS` does not bind unless
`MCP_CONNECTION_NONBLOCKING=0` or the server sets `alwaysLoad: true`.

## `docker run` is stdout-clean: all CLI status/pull/progress output goes to stderr

`docker run --rm <image> echo HELLO 2>/dev/null` prints exactly `HELLO` on stdout even when the
image is absent and the run emits a full layer-pull progress log — all on stderr. Makes `docker run
-i --rm` a safe transport for a stdio protocol (e.g. wrapping an MCP server). Corollary: any wrapper
script around `docker run`/`docker build` on that path must redirect *its own* output to stderr
explicitly — Docker isn't the leak risk, a naive wrapper is.

## A fully-cached `docker build` still makes a registry round trip for `FROM` metadata unless the base image is in the local image store — and a BuildKit build never populates that store

Even when every layer reports `CACHED`, `[internal] load metadata for docker.io/library/<base>` is
a live step unless the base was explicitly `docker pull`ed (or pinned by digest) — after a *cold*
BuildKit build of an image, `docker images <base>` can still be **empty**, because BuildKit
populates only the build cache, not the image store. Consequence: `docker build` is not an
offline-safe operation to put on a hot path (session start, CI step, launch wrapper) unless the
base image is guaranteed present in the image store first. `docker image inspect` (not `docker
build`) is the cheap, always-local staleness probe.

## In a container whose PID 1 is a bare interpreter (`python -c ...`), `SIGTERM` is silently ignored — `--init` is required, not defensive

PID 1 gets no default signal dispositions, and CPython installs a handler only for `SIGINT`; a
`docker kill --signal=TERM` against such a container leaves it `running` indefinitely, while the
same run with `--init` exits promptly (`ExitCode 143`). The only true-orphan path is a process
*not* reading stdin when the `docker run -i` client is killed; a process still reading stdin sees
EOF on client death and exits normally, letting `--rm` reap it. `--label k=v` (not `--name`, which
collides under concurrent launches) is how to enumerate survivors afterward.

## An interrupted `docker build` keeps its completed layers — a retried build resumes, it doesn't restart

A `--no-cache` build killed mid-`pip install` reports the earlier layers `CACHED` on the very next
build attempt, re-running only the interrupted step. Repeated attempts converge monotonically, which
makes "build inside a timed/bounded startup window" less catastrophic than it looks — a timeout-killed
build isn't wasted work.

## Docker's bare `-e VAR` (no `=value`) does not fall through to the image's `ENV` when `VAR` is unset in the caller's shell — it *deletes* the variable in the container

The CLI reference's "unset in the container" is literal, not "the image default applies." An image
built with `ENV FOO=bar`, run with `env -u FOO docker run -e FOO ...`, reports `FOO` as unset
inside the container — not `bar`. The common `-e VAR1 -e VAR2 ...` pass-through idiom silently
defeats every image `ENV` default it names whenever the caller's shell doesn't happen to have that
var exported. Safe form: build the `-e` argument list conditionally, only including a var that's
actually set in the caller's environment (`[ -n "${!v+set}" ] && ARGS+=(-e "$v=${!v}")`).

## A `docker build` whose changed inputs touch no COPYed layer produces the *same image ID* under a new tag — `docker image ls`'s `CreatedSince` then reads as "my build did nothing"

If a content-hash image tag moves (inputs changed) but the *target* being built never COPYs the
changed file, the resulting image is byte-identical to the previous one — same `sha256:...` under
both tags — and `CreatedSince` reports the *original* creation time, which reads as staler than the
build that just ran. Always verify a rebuild by image **ID** (`docker image inspect --format
'{{.Id}}'`), never by `CreatedSince`. Corollary: a content hash covering more inputs than a given
build target actually COPYs will keep re-deriving the same image under new tags — a real argument
for scoping the hash per target when multiple targets exist in one Dockerfile.

## `set -euo pipefail` turns a legitimate early-exiting consumer's SIGPIPE into either a silent script-kill or a false "producer failed"

Two distinct, both reproduced: (1) a bare `var="$(producer | early_exit_consumer)"` assignment is
**not** `errexit`-exempt the way an `if`/`while`/`&&`/`||` condition is — a downstream consumer that
legitimately returns early (matched, stopped reading) triggers SIGPIPE upstream, and under
`pipefail` the whole script exits immediately with no diagnostic. (2) Wrapping the same pattern in
`if ! ( producer | consumer ); then ...` avoids the abort but then mis-attributes the SIGPIPE'd
pipeline's nonzero exit to "the producer failed" — indistinguishable by exit code alone from a real
producer error (confirmed: a healthy tree reported a fabricated "'find' failed" with zero actual
`find:` stderr, purely from an early-matching consumer closing its end of the pipe).

**Fix that resolves both:** judge producer success by whether it wrote to a **captured stderr
file** (`2>"$errfile"`; `[ -s "$errfile" ]`), never by exit status, when a pipe may legitimately be
read only partially by design; and keep any expression that could carry a SIGPIPE-tainted status
inside an `if !`/`if ! var=$(...)` so `errexit` can never fire on it directly.

## On WSL2, a `command -v npm` toolchain probe can pass on a box with no Linux Node at all — the `npm` it finds is the Windows shim inherited from the Windows `PATH`

Re-verified 2026-09-07: `node` is **not found** on this box, while `npm` resolves to `/mnt/c/Program
Files/nodejs/npm` (11.11.0), because WSL2 appends the Windows `PATH`. The failure is late and
misdirecting — that `npm` installs Windows-native binaries (`esbuild`, `rollup`, `lightningcss`)
that a Linux bundler cannot load, and the error surfaces at bundle time looking like a broken
dependency rather than a wrong toolchain. **Probe for `node`, never for `npm`**, and reject any
interpreter resolving under `/mnt/`. Two corollaries for provisioning: a per-user tarball install
needs no sudo (`sudo -n true` fails here), and exporting its bin dir in one shell does **not**
persist — nothing rewrites the login `PATH`, so a build script must resolve the interpreter itself
rather than trust `PATH`. Worked instance: `salesperson/scripts/install_node.sh` +
`salesperson/build.sh`'s resolution order (`salesperson/README.md` § Prerequisites).

## npm 11.19.0 can crash internally — `TypeError: Cannot read properties of null (reading 'children')` in `@npmcli/arborist` — where it should have reported an `ERESOLVE` peer conflict

Trigger: **one** `npm install` line mixing a pinned major with an unpinned sibling that resolves to
a *different* major — `vitest@^3` alongside a bare `@vitest/ui` that resolved to 4.x. The crash
names no package, so it reads as a broken npm rather than a version conflict; only the debug log
shows the cause (`placeDep ROOT @vitest/ui@4.1.11` immediately before the throw, while the manifest
fetched for the `vitest` spec was 3.2.7). **Workaround:** split the install into small groups until
the crashing pair is isolated, then pin both siblings to the same major. Verified 2026-09-02 on npm
11.19.0 / Node v24.20.0; the npm bug will be fixed upstream, but the *diagnostic* pattern — an
internal `TypeError` standing in for a peer conflict — outlives the fix.
