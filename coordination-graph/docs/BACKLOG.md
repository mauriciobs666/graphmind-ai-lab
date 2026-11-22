# coordination-graph — Backlog

Living backlog, per the module documentation convention (root `AGENTS.md`) — forward-looking only:
delivered work leaves this file and is recorded in `HISTORY.md`. Items carry scheme IDs (`GAL-F00NN`) minted by `wi`.

## Open items

### GAL-F0035 — `retry-failed` exits 0 on a guard refusal

- **Status:** 🔵 proposed — defaulted; owner confirms status

`retry-failed` always exits 0, including when the guard refuses the endpoint (it prints the refusal,
paces `retry-due` 60 s ahead and leaves the failure state), so a caller cannot tell a refused retry
from a settled one. What remains: decide the exit status for a refusal (`hook-sync` exits 2 for the
same case) and pin it in `coordination-graph/docs/DESIGN.md` §8.4.

### GAL-F0036 — CLI surface for `identity.resolve_need` and `needs_status`

- **Status:** 🔵 proposed — defaulted; owner confirms status

Need resolution (`met` / `unmet` / `unresolvable` for a unit's `Needs` cell) exists only as library code
in `coordgraph/identity.py`; no subcommand or view calls it. What remains: a command or view that
reports a unit's needs status, as `claude/docs/plans/coordination-graph3.md` §3.4 and cross-initiative
AC-3/AC-4 (`docs/requirements/cross-initiative-coordination.md`) assume.

### GAL-F0037 — Engine cleanup for KB sections of files dropped from `kb-files.txt`

- **Status:** 🔵 proposed — defaulted; owner confirms status

A file removed from the registry before its headings are emptied leaves `:KbSection` rows and
falkor-chat documents behind; `kb-status` lists them as `unregistered` and no command clears them.
What remains: a `kb-sync --prune` or `retire <path>` that deletes the documents and mapping of a
section whose file is no longer registered.

### GAL-F0038 — KB failed-document retry minors

- **Status:** 🔵 proposed — defaulted; owner confirms status

Four edges in the retry path (`coordgraph/kbsync.py`): a `refused` section has already burned a retry
count before the refusal check; the `embedded < chunks` snapshot can race an embed still in flight and
read a document as unsearchable; an already-superseded document that is failed but embedded reads
`degraded`; once the hook is registered, a section held by the retry bound reports "not synced" on every
edit of its file. What remains: decide and fix each, or record it as accepted.

### GAL-F0039 — `kb-status` does not show retry counts

- **Status:** 🔵 proposed — defaulted; owner confirms status

`retryCount` and `lastRetryAt` are stored on `:KbSection` but `kb-status` prints neither, so the
remaining retry budget of a `failed-doc` section is visible only in the `kb-sync` outcome detail.
What remains: print them on `failed-doc` lines.

### GAL-F0040 — Discovery classes a role-suffixed topic slug as a ledger

- **Status:** 🔵 proposed — defaulted; owner confirms status

A plan whose topic slug ends in a role token (`docs/plans/cross-initiative-coordination.md`) matches
the `*-coordination.md` ledger glob and is mirrored as a ledger. What remains: an owner-approved rule
plus a rename, or a `not-ledgers` registry beside `kb-files.txt`; `lint` is not to change.
Owners: `architect` / `cobb`.

### GAL-F0041 — Symlinks: CLI discovery follows them, the hook does not

- **Status:** 🔵 proposed — defaulted; owner confirms status

`discover` and `sync <path>` follow a symlink to a file outside the repo and mirror it, while the hook's
`find -type f` never sees it. Exploiting it needs write access to the repo. What remains: reject
symlinks (or paths whose realpath leaves the root) in discovery and `sync`, matching the hook.

### GAL-F0042 — BOM-unsafe direct text reads

- **Status:** 🔵 proposed — defaulted; owner confirms status

The parsers decode through `parse.source.decode` (BOM removed), but `cli.py`, `migrate.py`, `lint.py`
and `history.py` read files with `read_text` and keep a leading BOM. What remains: route those reads
through the one decoder.

### GAL-F0043 — Hook go-live and the unmeasured real-backend KB round trip

- **Status:** 🔵 proposed — defaulted; owner confirms status

The hook's firing contexts were probed 2026-11-21 (`coordination-graph/docs/hook-settings.md` §6);
still unverified: permission modes other than `auto` and teammate sessions. Live constraint: a failed
Edit fires no hook event. The KB round trip against the real falkor-chat and embedder has not been
measured. These are operator steps, not code.

### GAL-F0044 — Fast-path budget holds only on an idle machine

- **Status:** 🔵 proposed — defaulted; owner confirms status

The hook's no-change path is about 20 ms on an idle machine and about 105 ms median under eight
busy CPU loops, against a 50 ms target stated without a load condition. What remains: state the
condition in `README.md`, or restrict the `find` to the source directories.

### GAL-F0045 — Test doubles forward `ro_query` to `query` and accept writes

- **Status:** 🔵 proposed — defaulted; owner confirms status

`MemGraph.ro_query` (and the `ProposeGraph` and `ReadOnly` wrappers) forward to `query`, so a view that
issued a write through `ro_query` would pass the offline tests; production is safe because the server
rejects writes on `RO_QUERY`. What remains: make `MemGraph.ro_query` reject write statements, with a test.
Also: `propose`, `drift` and `query` on an absent mirror print the server's raw error rather than the
`no synced mirror` hint that `inflight`, `unit` and `planned` give.
