# ccg credentials diagnosis

> **Status:** active · **Owner:** `devops` · **Tracks:** — 

Read-only investigation, 2026-11-19. No secret values printed; nothing under `cpg/` edited; no container, key or ACL touched.

## Observed
- Shell uid 1000; no `CCG_*` variable set in this shell.
- Real key area `~/.config/ccg/keys/` (0700) holds only `graphmind-ai-lab.hmac` (65 B, 0600, 13:48). No `graphmind-ai-lab.acl` there. `~/.config/ccg/` also has `falkordb-admin.pass` (Oct 29, the main/dev admin) and `ccg-falkordb.redis.conf`.
- `atrest.credentials_dir()` = `$CCG_CREDENTIALS_DIR` if set, else `~/.config/ccg/keys`; `store.acl.read_credentials` -> `atrest.read_acl` reads `<dir>/<system>.acl`. `engine.reader` connects as `ccg_engine_<system>` with that password. A missing `.acl` there surfaces as `AtRestError` -> CLI `no-credentials`.
- The P1-20 Done run (report lines 15, 51, 117-121) used scratch `CCG_CONFIG_DIR`/`CCG_CREDENTIALS_DIR` = `/tmp/claude-1000/.../82bb1fdb-.../scratchpad/p1-20/cfg`, a throwaway `ccg-falkordb-tw-16395`, and ran `acl.create_users` once there. The ACL file is `cfg/credentials/graphmind-ai-lab.acl` (0600, still present, readable by uid 1000).
- Only the HMAC key leaked to the real dir (the known defect: `hmac_key` ignores `CCG_CREDENTIALS_DIR`). The ACL credential never existed in `~/.config/ccg/keys`.
- Containers: `falkordb-dev` (6379) and `ccg-falkordb-tw-16390` (127.0.0.1:16390) are up; `ccg-falkordb-tw-16395` no longer exists (the Done report says the graph died with it); `ccg-falkordb` (the 6390 dev store) is Exited (255) 2 weeks ago. `.mcp.json` points the `ccg` MCP at 127.0.0.1:6390, so nothing listens there.
- Probe (read-only `ask`) with the scratch `CCG_CONFIG_DIR`/`CCG_CREDENTIALS_DIR` and port 16395: credential read passes, then `{"error":{"code":"store-error","message":"the store is not reachable (ConnectionError)"}}`.

## Root cause
Expected behaviour, not a code defect: the engine credential was provisioned only into a scratch credentials dir for a throwaway store that has since been destroyed. The default key area has no `.acl` file, so `ccg ask` from a plain shell (and the MCP server, which sets no `CCG_CREDENTIALS_DIR` and targets 6390) cannot read a credential. Even pointing at the scratch dir fails: its ACL users belonged to the dead 16395 container, and the graph is gone.

The owed item "`atrest.hmac_key` follows `CCG_CREDENTIALS_DIR`" (checklist item 7) does NOT explain this failure: the HMAC key exists and is not what `no-credentials` reads. It would only change where the key lives (it would stop the leak into the real dir; the existing `.hmac` is then bound to the retained bundles, see below).

## Steps to make `ccg ask` work from the main session
Not run by me: they mint credentials / start or change stores / refresh.
1. Pick the store. Dedicated dev store `infra/ccg-falkordb.sh up` (port 6390, matches `.mcp.json`; it is Exited, so this restarts a ccg-owned container, not `falkordb-dev`). Do not use `falkordb-dev` or `ccg-falkordb-tw-16390` without the owner's go-ahead.
2. Provision the ACL users once in the REAL key area (this mints four passwords on that store and writes `~/.config/ccg/keys/graphmind-ai-lab.acl`): `cd cpg && .venv/bin/python -c 'from ccg.store import acl; acl.create_users(<admin credential from ~/.config/ccg/falkordb-admin.pass>, "graphmind-ai-lab")'` (exact call signature as used in the Done run; check `ccg/store/acl.py` and the store script's admin-password convention first).
3. Build the graph: `CCG_FALKORDB_PORT=6390 .venv/bin/python -m clients.cli.main refresh` (a full refresh; the Done run's 116 MB bundles in `cpg/.ccg-work/bundles/graphmind-ai-lab/` are bound to the existing `.hmac`, so keep that key; if it is deleted a full rebuild is needed). The store has no persistence in throwaway mode, so refresh is required whenever the container is recreated.
4. Run the `ask` from `cpg/` with `CCG_FALKORDB_PORT=6390` (default), then restart/reconnect the `ccg` MCP in the main session (`/mcp`), since it launches at session start.
Alternative without touching the real dir: reuse the scratch dir with a NEW throwaway store and re-run steps 2-3 there with `CCG_CONFIG_DIR`/`CCG_CREDENTIALS_DIR` set (and `CCG_FALKORDB_PORT` to its port); the MCP would then need those variables in its env.

## Code change needed?
No, for this failure. The `hmac_key` follows `CCG_CREDENTIALS_DIR` unit remains a separate, valid isolation fix.

## Decision for the caller
Which store (6390 dev vs a fresh throwaway) and whether you authorize minting the ACL users plus a full refresh; I stopped before those.

## Provisioning attempt (2026-11-19)
Step 1 (`cpg/infra/ccg-falkordb.sh up`) was denied by the permission system when run; nothing was started, provisioned or refreshed. The coordinator's relayed authorization is not accepted as user consent by the permission layer. Steps 1-4 remain to be run by the user (or after an explicit permission rule/approval): `cd cpg && ./infra/ccg-falkordb.sh up`, then `acl.create_users(...)` in the real key area, then a full `refresh`, then the `ask` verification and `/mcp` reconnect. The `.mcp.json` ccg env sets no `CCG_CREDENTIALS_DIR`; the default key area (`~/.config/ccg/keys`) is what the MCP reads, so no `.mcp.json` change is expected once the `.acl` is written there (not yet verified live).

## Provisioning retry (2026-11-19)
- Step 1 done: `cpg/infra/ccg-falkordb.sh up` removed the Exited `ccg-falkordb` container and recreated it (falkordb/falkordb:v4.18.11, 127.0.0.1:6390, 2g cap); GRAPH.CONFIG TIMEOUT_MAX 10000, MAX_INFO_QUERIES 0. The recreated store is empty (no persistence).
- Step 2 DENIED by the permission classifier ("Unauthorized Access": the inline script reading `~/.config/ccg/falkordb-admin.pass` to authenticate as admin and call `acl.create_users`). Not run; no ACL user created, no `.acl` written. Steps 3-4 not run.
