# cpg/infra — dedicated FalkorDB instance for C-1001 P0

`ccg-falkordb.sh` runs **`ccg-falkordb`**, the one FalkorDB instance that P0's store spikes and
vertical slice use (`docs/plans/connected-code-graph-p0.md` §2 D7, §4.1 S01, §5.1). It is
separate from the shared `falkordb-dev` (6379), which nothing here touches.

| Setting | Value |
|---|---|
| Container | `ccg-falkordb` (label `ccg-falkordb=1`) |
| Image | `falkordb/falkordb:v4.18.11`, pinned, same as `falkordb-dev` |
| Binding | `127.0.0.1:6390` only (`falkordb-dev` publishes `0.0.0.0`, so don't copy its `-p` form) |
| Memory | `maxmemory 1536mb`, `maxmemory-policy noeviction` |
| Persistence | none: `save ""`, `appendonly no`, no volume |
| Container memory | `--memory 2g --memory-swap 2g` (cap, no swap; see below) |
| Web console | off (`BROWSER=0`), which saves RAM on the 6 GB WSL host |
| Admin password | `~/.config/ccg/falkordb-admin.pass` (mode 600, outside the repo, created on first `up`) |

## Commands

```bash
cpg/infra/ccg-falkordb.sh up        # start if not running; no-op if it already is
cpg/infra/ccg-falkordb.sh status    # state, binding, D7 config, module version, key count
cpg/infra/ccg-falkordb.sh down      # stop and remove; refuses a non-empty keyspace
cpg/infra/ccg-falkordb.sh down --force
cpg/infra/ccg-falkordb.sh reset-p0 --yes   # FLUSHALL at P0 exit, approval-gated
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
  - §5.1 exit hygiene: `GRAPH.CONFIG TIMEOUT_MAX` must be `0`, and `default` must be the only
    ACL user. `teco` can run `status` when a reservation is released.
- `down` refuses while any db holds keys, because they may belong to another step's
  reservation (§5.1). A stopped container has already lost its data, because nothing persists.
  `up` recreates our own stopped container from scratch.
- **Full clean start:** `down --force`, then `up`. This resets the keyspace, ACL users and
  runtime config (`GRAPH.CONFIG`, `CONFIG SET`), because nothing persists.
- `reset-p0` is the only destructive verb. It needs `--yes` and it refuses unless the
  container is our labelled `ccg-falkordb` published on exactly `127.0.0.1:6390`. It reports done
  only once the authenticated key count is `0` after the flush. It empties the **keyspace only**:
  ACL users and `GRAPH.CONFIG` changes survive, so use the clean start above if you need those
  reset too. Run it only with approval.
- Env overrides: `CCG_FALKORDB_NAME`, `CCG_FALKORDB_PORT`, `CCG_FALKORDB_IMAGE`,
  `CCG_CONFIG_DIR`. `reset-p0` refuses any name or port override. `status` checks the image
  against the pinned default, not against `CCG_FALKORDB_IMAGE`.

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
removes, the `status` drift checks, and `reset-p0`'s check after the flush.

## Connecting

```bash
REDISCLI_AUTH="$(cat ~/.config/ccg/falkordb-admin.pass)" \
  redis-cli -h 127.0.0.1 -p 6390 --no-auth-warning GRAPH.QUERY ccg_smoke "RETURN 1"
```

From Python, read the password from the same file and pass it as `password=`. Every P0 script
takes host and port explicitly, with no 6379 default (§5.1).

## Password handling

The password never appears on a command line, in a container env var (which `docker inspect`
would show), or in output. On each `up`, the script writes
`~/.config/ccg/ccg-falkordb.redis.conf` (mode 600: `requirepass` plus the D7 settings). It
mounts that file read-only at `/etc/ccg/redis.conf`, and the image's `run.sh` passes it to
`redis-server` as the config file through `REDIS_ARGS`. `FALKORDB_ARGS` is left unset, so the
module defaults match `falkordb-dev`. This instance creates no ACL users. S22 probes them, and
the P1 loader creates them.

## Reservations

Each step that uses the instance holds a §5.1 reservation, which `teco` grants: R-B (S13), the
exclusive R-C1/R-C2/R-C3 (S20–S22), and R-D (S17). Each exclusive step restores any instance-wide
config it changed before it releases the reservation.
