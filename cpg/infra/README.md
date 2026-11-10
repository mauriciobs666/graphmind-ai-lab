# cpg/infra — dedicated FalkorDB instance for C-1001

`ccg-falkordb.sh` runs **`ccg-falkordb`**, the dedicated FalkorDB instance of C-1001
(`docs/plans/connected-code-graph-p0.md` §2 D7, §5.1; P1 flags and throwaway mode: step P1-48 in
`docs/plans/connected-code-graph-p1-extraction.md`), and disposable **throwaway** instances for
live tests (below). It is separate from the shared `falkordb-dev` (6379), which nothing here
touches. The sandbox images (`toolchain/`, incl. the P1 `indexer` image) are described in
`cpg/infra/toolchain/README.md` and in "Indexer image" below.

| Setting | Value |
|---|---|
| Container | `ccg-falkordb` (label `ccg-falkordb=1`) |
| Image | `falkordb/falkordb:v4.18.11`, pinned, same as `falkordb-dev` |
| Binding | `127.0.0.1:6390` only (`falkordb-dev` publishes `0.0.0.0`, so don't copy its `-p` form) |
| Memory | `maxmemory 1536mb`, `maxmemory-policy noeviction` |
| Persistence | none: `save ""`, `appendonly no`, no volume, so the first query after a restart is a rebuild |
| Module config (P1) | `GRAPH.CONFIG SET TIMEOUT_MAX 10000`, `MAX_INFO_QUERIES 0`, applied by `up` (`-p0-graph.md` §6 rows 5, 12) |
| Container memory | `--memory 2g --memory-swap 2g` (cap, no swap; see below) |
| Web console | off (`BROWSER=0`), which saves RAM on the 6 GB WSL host |
| Admin password | `~/.config/ccg/falkordb-admin.pass` (mode 600, outside the repo, created on first `up`) |

## Commands

```bash
cpg/infra/ccg-falkordb.sh up        # start if not running (no-op if it is); reports TIMEOUT_MAX / MAX_INFO_QUERIES
cpg/infra/ccg-falkordb.sh status    # state, binding, D7 config, P1 hygiene, module version, key count
cpg/infra/ccg-falkordb.sh down      # stop and remove; refuses a non-empty keyspace
cpg/infra/ccg-falkordb.sh down --force
cpg/infra/ccg-falkordb.sh reset-p0 --yes   # FLUSHALL the keyspace, approval-gated

cpg/infra/ccg-falkordb.sh --throwaway up                       # disposable instance; prints NAME/PORT
CCG_FALKORDB_PORT=<port> cpg/infra/ccg-falkordb.sh --throwaway status
CCG_FALKORDB_PORT=<port> cpg/infra/ccg-falkordb.sh --throwaway down
cpg/infra/ccg-falkordb.sh --throwaway list
```

- **Ownership.** No verb stops, removes, or writes to a container that lacks the
  `ccg-falkordb=1` label, which only this script's `up` sets. A `CCG_FALKORDB_NAME` override that
  points at `falkordb-dev` or any other container is refused. Every Redis call, including
  `down`'s key count, goes to the port the target container itself publishes, never to
  `CCG_FALKORDB_PORT`.
- `status` exits non-zero if the instance is down or anything has drifted:
  - the label, the pinned image, or graph module version `41811`;
  - the binding is not exactly one `127.0.0.1:6390` line;
  - the memory cap;
  - auth is off (an unauthenticated `PING` must get `NOAUTH`);
  - any of the four D7 values;
  - the P1 steady state: `GRAPH.CONFIG TIMEOUT_MAX` must be nonzero (`up` sets 10000; P1-19
    re-derives it), and `MAX_INFO_QUERIES` must be `0`;
  - the ACL users: every user must be `default` or `ccg_<role>_<system>`, with role `build`,
    `verdict`, `engine` or `raw`. `<system>` must match the system-name whitelist of
    `docs/plans/connected-code-graph-p1-graph.md` §7. Any other user is drift, including a bare
    `ccg_build`.

  A `TIMEOUT_MAX` of `0` usually means the container was restarted behind `up`'s back, which also
  emptied it. In that case `up` on the running instance fails too, and names the remedy
  (`down --force`, then `up`). `teco` can run `status` when a reservation is released.
- `down` refuses while any db holds keys, because they may belong to another step's
  reservation (§5.1). A stopped container has already lost its data, because nothing persists.
  `up` recreates our own stopped container from scratch.
- **Full clean start:** `down --force`, then `up`. This resets the keyspace, ACL users and
  runtime config (`GRAPH.CONFIG`, `CONFIG SET`), because nothing persists.
- `reset-p0` (named for its P0 origin; it is the general keyspace reset) is the only destructive
  verb on `ccg-falkordb`. It needs `--yes` and it refuses unless the
  container is our labelled `ccg-falkordb` published on exactly `127.0.0.1:6390`. It reports done
  only once the authenticated key count is `0` after the flush. It empties the **keyspace only**:
  ACL users and `GRAPH.CONFIG` changes survive, so use the clean start above if you need those
  reset too. Run it only with approval.
- Env overrides: `CCG_FALKORDB_NAME`, `CCG_FALKORDB_PORT`, `CCG_FALKORDB_IMAGE`,
  `CCG_CONFIG_DIR`. `reset-p0` refuses any name or port override. `status` checks the image
  against the pinned default, not against `CCG_FALKORDB_IMAGE`.

## Throwaway instances

Live tests that mutate ACL users or `GRAPH.CONFIG`, share fixed key names, or run in parallel use
their own throwaway instead of `ccg-falkordb` (`docs/plans/connected-code-graph-p1.md` §8).

- `--throwaway up` starts `ccg-falkordb-tw-<port>` with exactly the D7 and P1 configuration above,
  on `CCG_FALKORDB_PORT` or the first free port in **16390–16489**. It labels the container
  `ccg-falkordb=1` and `ccg-falkordb.throwaway=1`. On success, stdout carries exactly two lines,
  `CCG_FALKORDB_NAME=…` and `CCG_FALKORDB_PORT=…`; logs go to stderr.
- It **refuses ports 6379 and 6390**, in any spelling (`06390` is read as 6390), before any Docker
  call. It also refuses any port already
  published by a container (`docker ps`) or listening on the host (`ss`). If a port is lost to a
  race between the check and `docker run`, the half-created container is removed and the next
  port is tried. With an explicit port, the run fails. A throwaway that starts but never answers
  `PING`, or refuses the module config, is stopped and removed, and no name is printed.
- The name is always derived from the port. A `CCG_FALKORDB_NAME` that is not `ccg-falkordb-tw-<port>`
  is refused, so a throwaway verb can never reach `ccg-falkordb` or `falkordb-dev`. `status` and
  `down` also need the `ccg-falkordb.throwaway=1` label. `reset-p0` is refused in this mode.
- `--throwaway down` needs no `--force`: the keyspace belongs to the caller that started it. It
  also removes the throwaway's generated config file. Throwaways share the admin password file.
- Each throwaway can use up to 2 GiB (the memory cap), so on the 6 GB host keep to one or two at
  a time, and always `down` what you started. `--throwaway list` shows any leftovers.

## Indexer image

`cpg/infra/toolchain/build.sh --indexer` builds **`ccg-toolchain-indexer:<hash>`** (target
`indexer`, `FROM toolchain`, same content hash as the other two images). It adds Node 24.22.2,
`scip-python` 0.6.8 and `scip-typescript` 0.6.0 (the `truth-npm/` lockfiles, copied from the
`npm-fetch` stage that `truth` also uses), plus `python3`, under `/opt/indexer`. The default
`toolchain` image that the sandbox runs still carries no Node and no indexer, so only an indexer
run widens the surface. That run goes through `cpg/sandbox/offline-run.sh`, which reads only
`CCG_TOOLCHAIN_IMAGE`, so set it per call:
`CCG_TOOLCHAIN_IMAGE="$(cpg/infra/toolchain/build.sh --indexer --print-tag)"`. Contents, size and
tests: `cpg/infra/toolchain/README.md`, "The `indexer` image".

**The runner has two indexer entries** (`-egress-spec3.md` §7.4, P1-48b): `--frontend scip-typescript`
and `--frontend scip-python`, `--mode offline`, container backend only, and only on an image
labelled `ccg-toolchain-indexer=1`. On any other image (the default `toolchain` included) the
runner refuses with `backend-unavailable`, exit 69, and records it. `toolArgv` must be `index` plus
`--cwd /src[/…]`; `scip-python` also needs `--environment /src/…` and `--project-version <v>`, and may
take `--project-name`. Each `--cwd`/`--environment` path is walked on the host and refused if it follows an
absolute symlink, leaves the staged tree or does not exist; the image is launched by its `sha256:` ID.
Any other element is `forbidden-flag`, a missing mandatory one is `usage`, and the runner appends
`--output /out/<auditId>.scip` itself. An indexer is never launched through `--frontend joern`.
`PYTHONSAFEPATH=1` is on the indexer pass-list, so `scip-python`'s `python3 -c` probe does not put
the staged `/src` on `sys.path`. Provisioning dependencies into the staged directory (`node_modules`,
a venv, `env.json`) is an operator step outside the system; the runner and `cpg/` modules run no
`npm`, `pip` or `uv`. Indexer runs are for the lab's trusted, pinned sources, and a run has no
memory, CPU or time limit. Tests: `cpg/sandbox/README.md`, "Indexer entries".

## Memory cap

`maxmemory 1536mb` limits only the memory Redis accounts for, and it is checked when each
command starts. It doesn't cover allocator fragmentation, client buffers, or a query that
overshoots after it was admitted.

- **The cap.** `--memory 2g` is `maxmemory` plus 512 MiB, about 1.33x. A runaway then stays
  inside this container instead of reaching the VM's OOM killer, which could take `falkordb-dev`
  or a Joern JVM a step is timing.
- **No swap.** `--memory-swap 2g`, equal to the memory cap, means the container never swaps.
  An instance that would have swapped, and quietly distorted S20's timings, gets a loud OOM kill
  instead. That costs nothing, because nothing persists. `docker inspect ccg-falkordb` shows
  `OOMKilled: true` if it happens.

## Regression test

```bash
cpg/infra/tests/test-ccg-falkordb.sh     # stubbed docker/redis-cli/ss/openssl; never real Docker
```

The test covers the ownership guard across every combination of:
- verb: `up`, `down`, `down --force`, `reset-p0 --yes`;
- override: name, port and config dir, alone and combined;
- container state: running, exited, absent;
- label: present, absent.

It also covers positive controls, the case where `down` must count the keys of the container it
removes, the `status` drift checks (including the P1 steady state), `reset-p0`'s check after the
flush, `up`'s module config, and every `--throwaway` refusal, port pick and race described above.

## Connecting

```bash
REDISCLI_AUTH="$(cat ~/.config/ccg/falkordb-admin.pass)" \
  redis-cli -h 127.0.0.1 -p 6390 --no-auth-warning GRAPH.QUERY ccg_smoke "RETURN 1"
```

From Python, read the password from the same file and pass it as `password=`. Every script
takes host and port explicitly, with no 6379 default (§5.1).

## Password handling

The password never appears on a command line, in a container env var (which `docker inspect`
would show), or in output. On each `up`, the script writes
`~/.config/ccg/ccg-falkordb.redis.conf` (mode 600: `requirepass` plus the D7 settings). It
mounts that file read-only at `/etc/ccg/redis.conf`, and the image's `run.sh` passes it to
`redis-server` as the config file through `REDIS_ARGS`. `FALKORDB_ARGS` is left unset, so the
module defaults match `falkordb-dev`. The P1 module config is applied with `GRAPH.CONFIG SET`
after the start, for the same reason. This script creates no ACL users; P1-19's `create_users`
does.

## Reservations

A step that uses `ccg-falkordb` itself, rather than a throwaway, holds a §5.1 reservation, which
`teco` grants. Each exclusive step restores any instance-wide config it changed before it releases
the reservation.

## Bundle-at-rest

Step P1-02 (F-1, `docs/plans/connected-code-graph-p1-extraction.md`). This decision **replaces**
the interim default in `docs/plans/connected-code-graph-bundle-spec3.md` §13 F-1 ("P0 bundles
live only in the gitignored work area"). `cpg/ccg/atrest.py` enforces it, and
`cpg/tests/test_atrest.py` tests it. A bundle can hold source literals in clear
(`valueSet`/`constant`), and FR-15 covers the graph and the answers, not the local bundle files
(stakeholder decision 2026-10-29).

| Item | Location | Mode | Rule |
|---|---|---|---|
| Bundles | `cpg/.ccg-work/bundles/<system>/<unit>.<build>.bundle.jsonl` (gitignored) | files `0600`, dirs `0700` | written with `atrest.write_bundle`, read with `atrest.read_bundle`. `ccg.bundle.write_bundle` alone takes the umask's mode |
| Retention | the same directory | — | the last good bundle per unit plus the one being built (a `.tmp` name). After a good build, `atrest.prune_unit(system, unit, keep_build)` deletes the unit's older bundles |
| HMAC key (`valueHash`, `bindParts`, §1.7.5) | `~/.config/ccg/keys/<system>.hmac`, **outside the repo** | `0600`, dir `0700` | 32 bytes from `secrets.token_bytes` (the OS CSPRNG), generated once by `atrest.hmac_key` with an atomic first write. Never logged, never in a bundle, an answer or the broker audit. `atrest.key_path(system)` is the one accessor of its location |
| ACL passwords (P1-19 `create_users`) | `$CCG_CREDENTIALS_DIR/<system>.acl`, default `~/.config/ccg/keys/` | `0600` | one line per user. Written with `atrest.write_acl`, read with `atrest.read_acl`; both follow `CCG_CREDENTIALS_DIR`. The `.mcp.json` entry carries no credential |

- **Every access is checked, in exactly one area.** Bundle functions use only the work area; the
  key and ACL functions use only the key area or `CCG_CREDENTIALS_DIR`. The generic
  `ensure_private`/`read_private`/`write_private` take the area explicitly. They refuse:
  - a path outside its area, or one with a lexical `..`;
  - **any symlink** below the area root, checked with `lstat` on the unresolved path. That covers
    a link into the other area, a link between two files in the same area, and a dangling link;
  - anything not owned by the user;
  - any file or directory with group or other bits set.

  A read then walks the chain itself. It opens each directory with `O_DIRECTORY|O_NOFOLLOW`
  relative to its parent's descriptor, and the leaf with `O_NOFOLLOW`, re-checking owner and mode
  on each descriptor. So a file or a parent directory swapped for a link after the name check is
  never followed. Writes do not walk this way: they check the chain, then create the file. An exposed existing file raises; it is **never silently
  re-moded**, because it may already have been read. The modes this module sets do not depend on
  the umask.
- **Why the keys are outside the repo** (security advisory S5-6,
  `docs/reviews/connected-code-graph-bundle-spec5-security.md`). The sandbox containers run as
  uid 1000, the host user, so `0600` gives no separation from a container that has a parent
  directory mounted. A mount-refusal check could police only `cpg/sandbox/offline-run.sh`'s
  mounts. It would miss a hand-run `docker run -v`, a repo-scoped `Read` grant, an archive and a
  backup. Outside the tree, none of these reaches the key by construction. This also matches
  `-bundle-spec3.md` §5.17 and P0's `~/.config/ccg/`. A key area or `CCG_CREDENTIALS_DIR` inside
  the repo is refused, and `*.hmac`/`*.acl` are gitignored in `cpg/` as a second line of defence.
  Once the key is known, every word-only external host or path falls to a dictionary attack. That
  is why the key, not the bundle, is the asset this custody protects.
- **The published test keys never load** (S5-7). `-bundle-spec5.md` §9 publishes two 32-byte
  keys and their `bindKeyCheck` values: `00…1f` (EH-30) and `20…3f` (§9.4's `K2`). `hmac_key` is
  the single place that refuses them. It refuses either one if it was just generated (it is then
  never written), and again on every load. There is no test-harness bypass: conformance tests use
  the test keys directly and never through `hmac_key`.
- **Rotation is a full rebuild**, because the key changes every `valueHash`. Delete the `.hmac`
  file by hand, then rebuild every unit of the system.
- **No LLM-backed tool reads a bundle.** The plain-language front and every agent read answers and
  `raw_query` only. `cpg/tests/test_atrest.py` checks this with `pattern_escapes` from
  `cpg/eval/tools/audit.py`, rooted at the headless kit directory (`cpg/eval/headless/`). It
  judges these sources:
  - the headless settings, and any extra `--settings` file;
  - the agent definitions;
  - every launcher flag that grants tools or paths, in each spelling: `--allowedTools`,
    `--allowed-tools`, `--tools`, as a separate argument or with `=`, plus `--add-dir`. A
    `--settings` that cannot be read statically, and any `--mcp-config`, fail closed;
  - the launch argv itself (the launcher's single `ARGV=( … )` block), against a reviewed flag
    allow-list. An unlisted flag fails closed, such as `--dangerously-skip-permissions` or
    `--plugin-dir`. So does a wrong value for a fixed flag: `--permission-mode` must be `dontAsk`,
    `--permission-prompts` must be `none`, and `--agents` must be the kit's own `agents.json`. So
    does a missing `--strict-mcp-config`, `--permission-mode` or full `--disallowedTools` set;
  - every string value of the `.mcp.json` `ccg` entry. P1-27's exact launch string goes in the
    test's reviewed allow-list;
  - `cpg/clients/mcp/server.py`, once it exists. Any MCP tool parameter outside a fixed safe-name
    set fails closed, as does any reference to the bundle store in that module.

  A pattern that can name anything outside the root fails. Planted `Read(cpg/**)` and
  `Read(cpg/.ccg-work/**)` are the positive controls. A bare `Read` grant passes only with the
  kit's fail-closed `CCG_GUARD` hook. **This is a static bound, not proof.** It does not follow a
  tool body that reaches the bundle store through an import chain, and it does not know a flag
  spelling the CLI adds later.
- **Residual, stated rather than closed.** An interactive agent with unrestricted `Read`, such as
  a Claude Code session in this repo, runs as the same OS user and can read
  `cpg/.ccg-work/bundles/`. File modes cannot separate it. Closing that would need a project-level
  deny rule (`Read(cpg/.ccg-work/**)`), which is `teco`'s or `cobb`'s wiring and not done here. The
  keys are not exposed this way unless such a session reads `~/.config/ccg/` on purpose.
