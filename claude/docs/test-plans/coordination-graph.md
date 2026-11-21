# Coordination graph — acceptance test plan (S10)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** GAL-F0002 (coordination-graph S10) · **Extends:** `claude/docs/plans/coordination-graph.md` · **Item:** `GAL-F0002`

## 1. Scope and objective

Black-box acceptance of the `coordination-graph` component (`coordination-graph/`, CLI `coordgraph`, hook
`coordination-graph/hooks/post-tool-sync.sh`, KB mirror) against every acceptance criterion of
`claude/docs/requirements/coordination-graph.md` (AC-1..AC-15) and `docs/requirements/cross-initiative-coordination.md`
(AC-1..AC-6; a separate feature, `GAL-F0003`). Spec: base plan §4 row S10 and §5, `claude/docs/plans/coordination-graph3.md`
(§6 tests), `claude/docs/plans/coordination-graph-graph.md`, `docs/plans/cross-initiative-coordination.md` §5 (S10 addition).
Not a re-run of the unit layer: the 1155-test offline suite and the 88-test `live` suite are the baseline; this plan
extends past them (independently re-derived figures, real-corpus runs, adversarial inputs) and does not trust the
delivery record's numbers.

## 2. Execution rules (binding for this run)

| Mode | Meaning | Allowed commands |
|---|---|---|
| **R** read-only on the shared mirror | `--confirm coordination`, or `mcp__cypher__query` reads | `drift`, `kb-status`, `inflight`, `unit`, `planned`, `propose`, `lint`, `history-coverage`, `facts`, `rebuild --dry-run`, `query`; never `bootstrap`/`sync`/`rebuild`/`hook-sync`/`retry-failed`/`kb-sync`/`kb-reconcile`/`kb-finish` |
| **E** ephemeral | `falkordb/falkordb:v4.18.11` on a free loopback port (not 6379/6390/3000), `COORDGRAPH_EPHEMERAL_PORT` set, a **copy tree** under the scratchpad as `COORDGRAPH_REPO_ROOT`, a scratch `COORDGRAPH_STATE_DIR`, a scratch copy of `.work-items` via `COORDGRAPH_WI_ROOT`; KB legs on the scratch falkor-chat + fake embedder of `tests/fixtures/live.py`. Containers started by this run are removed afterwards. | everything, incl. writes |
| **H** hook script | `hooks/post-tool-sync.sh` fed `PostToolUse`/`PostToolUseFailure`-shaped JSON on stdin with `COORDGRAPH_HOOK_CLI` pointing at a shim that drops the literal `--confirm coordination` pair (the script hard-codes it; the guard refuses it on an ephemeral port by design). `CLAUDE_PROJECT_DIR` and the state dir are always scratch, so the real `.state/` and the shared graph are unreachable. Nothing is registered in `.claude/settings*.json`. | always with the shim |
| **S** static | read files, grep | read |
| **U** unverifiable now | needs the hook registered in a real Claude Code session and the S0 probe (`claude/cobb/scripts/hook-probe.md`), neither of which has happened | reported as *unverifiable until registration + S0*, never as passed |

No commits, no edits to `.claude/settings*.json`, tail-Edit-only on shared logs. Writes to this run's two deliverables only.

## 3. Risk assessment

| Risk | Why it matters | Where covered |
|---|---|---|
| Hook registered nowhere, yet read headers say `Hook: active` | A reader may trust same-moment freshness (FR-4) that does not exist | TP-040, TP-041 |
| Mirror silently wrong (count/parity) | Every view and proposal reads it | TP-010..TP-013 (counts re-derived from files by a different method) |
| Fast-path latency on every `Bash` call | Fails R2/50 ms budget; the plan measured it one way | TP-032 (several definitions, with and without payload, cold and warm) |
| Identity/legacy-ID join (`K-065` minted in two scopes) | The S1-close M4 finding; a wrong join merges two items | TP-021 |
| Malformed/hostile input crashing sync or leaving partial state | FR-3, plan §3.8 | TP-050..TP-055 |
| Concurrent hook/sync races | Two sessions edit at once | TP-056, TP-057 |
| KB swap reliability under a real embedder | falkor-chat K-070 closed but its live swap reduction is not measured | TP-060..TP-066, section 6 |
| Proposal never actually exercised on real data (no item records `Depends on`/`Touches`) | AC-11/13 pass on a fixture only | TP-023, TP-024 |
| Cross-initiative ACs (GAL-F0003) not built | S10's addition lists them | TP-070..TP-075 |

Deliberately **not** tested: search-quality/score floor (out of scope FR-10); LM Studio extraction throughput;
Windows/macOS shells (bash 5 on WSL2 only); the Mermaid rendering of the manual (reviewed by S9's QA pass); any shared-instance write.

## 4. AC-to-check map (every AC)

Legend for "Mode": R / E / H / S / U as in section 2. "Cannot verify now" is stated per row.

### 4.1 `claude/docs/requirements/coordination-graph.md`

| AC | Check (TP) | Method | Mode | Cannot be verified now |
|---|---|---|---|---|
| AC-1 doc consistency | TP-001 | Read root `AGENTS.md` for both rules; run the plan's n4 grep over non-archived docs and classify every hit | S | none |
| AC-2 crash safety | TP-010, TP-011 | Copy tree; `sync --all`; export every node/edge property; wipe the ephemeral graph's nodes (`MATCH (n) DETACH DELETE n`, not `GRAPH.DELETE`); `rebuild`; export again and diff; rebuild into a second empty key and diff; mutate one property and see which key reports it | E | Loss/restart of the **shared** `falkordb-dev` (never done) |
| AC-3 same-moment freshness | TP-030, TP-031 | Edit a ledger row in the copy tree; run the hook script with a `PostToolUse` and a `PostToolUseFailure` payload; run `unit` immediately; compare with the file | H+E | Same-moment freshness **in a real session** (hook not registered; S0 not run): U |
| AC-4 visible drift | TP-012 | Shared: `drift` as-is (16 sources drifted now, a natural experiment) and a view row carrying `⚠ drift`; E: hand-edit a file, then `drift`/`unit`/`planned` | R+E | none |
| AC-5 the view | TP-013, TP-014 | Shared `inflight` lists open coordinations (cross-checked against a grep of non-archived `*-coordination.md`), verdicts, requirements; E fixture adds a paused unit across two components | R+E | The real corpus has **no** paused unit (0 waiting), so that kind is proven on the fixture only; "asks an agent in a real session": U |
| AC-6 agent lookup | TP-015 | `unit <slug> <unit>` on the E fixture with a paused unit (status, agent, deliverable, question, date, no ledger open); `unit` by label, `~n`, ID on the shared mirror | R+E | none |
| AC-7 leaving the view | TP-016 | E: all-accepted `active` ledger stays; flip to `archived`, sync, gone; shared: the `inflight` ledger list equals the non-archived ledgers on disk | E+R | none |
| AC-8 reference freshness | TP-060..TP-066 | Scratch falkor-chat + fake embedder: add/delete a one-chunk `##` heading via the hook shim, `search_documents` immediately; hook bypassed -> `kb-status` flags; multi-chunk pending path; embedder down; kill mid-swap; duplicates; needs-resplit; degraded. Shared: read-only `kb-status` and `search_documents` on mapped claims | E+H+R | **Live swap reduction of falkor-chat K-070 is not measured**; no real `ws:agent-team` round-trip (writes forbidden this run); real LM Studio extraction/timeouts; hook not registered: U |
| AC-9 what's planned | TP-017 | Shared: `planned falkor-chat`, `planned cpg` (root `docs/BACKLOG.md`), `planned cobb` (agent plan); lists ID, marker, source file, sync header; no item appears in `inflight`; counts vs `### ` heading counts | R | none |
| AC-10 backlog mirror freshness | TP-018 | E: add/rewrite/remove an item, run hook then `planned`; hand edit without hook -> flagged | E+H | Real-session freshness: U |
| AC-11 proposal criteria | TP-023 | E fixture exactly as the AC (B deps A; C/D file; G/H graph key; I/J fact; E high, F low) -> `propose`; the criterion per placement | E | The real backlogs record **no** `Depends on`/`Touches` (verified in TP-024), so AC-11 has no real-data evidence |
| AC-12 not plannable | TP-024 | Shared `propose cobb/falkor-chat/cpg`: every item without facts listed with the missing facts; fixture drops each fact in turn | R+E | none |
| AC-13 cross-component blocker | TP-025 | E fixture: item depending on an open item in another backlog -> held with blocker and backlog; delivered per history heading -> satisfied | E | no real cross-backlog dependency exists to run on |
| AC-14 approval gate | TP-026 | `propose` on shared and E: `git status --porcelain`, sha256 of every backlog/plan file, graph node/edge counts and `Source.syncedAt` unchanged | R+E | The stakeholder-approval half (a recorded approval changes the backlog) has **no mechanism** in `coordgraph` (propose never writes); verify by reading the code and report as design, not behaviour |
| AC-15 identity | TP-020..TP-022 | `unit GAL-F0002`, `unit claude/...slug`, alias, legacy `falkor-chat:K-065`-style on the shared mirror (R) and on a scratch copy of the registry with a `wi move` (E); the M4 case (two scopes minting `K-065`); no `K-NNN` minted by `migrate --dry-run` | R+E | `migrate --apply` is never run (permanent allocation) |

### 4.2 `docs/requirements/cross-initiative-coordination.md` (GAL-F0003)

Its plan's steps X1..X6/S7c (`ready`, `overlap.py`, `touches.py`, `sections.py`, the Coordinations block) are not in the
code tree at the time of this run (checked in TP-070). Each AC gets the check that *can* run now plus the statement of what cannot.

| AC | Check (TP) | Method | Mode | Cannot be verified now |
|---|---|---|---|---|
| AC-1 description + path in the view | TP-070 | Read the shared `inflight` output for a one-line description per coordination and the path | R | Description line: feature not built |
| AC-2 overlap finders surface and raise | TP-071 | Static: do `tico`/`teco`/`architect` prompts name the overlap check; the three-transcript scenario of the plan is not runnable (needs live agents and a built view) | S | the scenario itself |
| AC-3 malformed `needs` | TP-072 | `coordgraph ready` absent from `--help`; no `needs` column parsing in `parse/ledger.py` beyond plan3's raw capture | R | feature not built |
| AC-4 ready-now variants A-E | TP-073 | same | R | feature not built |
| AC-5 scan list/closing report | TP-074 | same (`section` command absent) | R | feature not built |
| AC-6 first adopter | TP-075 | Read `docs/plans/connected-code-graph-coordination.md` for `Needs`/`Touches` columns | S | feature not built |

## 5. Test items

Priorities: P1 = release-blocking, P2 = important, P3 = nice-to-have. Types: F functional, C contract, I integration, X exploratory/adversarial, N non-functional.

| ID | Title | Pre | Steps (condensed) | Expected | Pri | Type |
|---|---|---|---|---|---|---|
| TP-000 | Baseline | venv present | from `coordination-graph/`: `pytest -q`; `pytest -q -m live` (ephemeral only) | 1155 passed / 88 passed; no leaked containers from this run | P1 | F |
| TP-001 | AC-1 doc check | — | read root `AGENTS.md` (Module documentation convention, backlog bullet); n4 grep over non-archived docs | both rules stated; no claim that history is headed for the graph | P1 | F |
| TP-010 | Counts re-derived | shared RO | `git ls-files`-based counts of ledgers/reviews/requirements/backlog files and `### ` item headings vs `mcp__cypher__query` `Source`/`Item` counts, `lint --count`, `rebuild --dry-run` | equal, differences explained | P1 | F |
| TP-011 | AC-2 wipe and rebuild | E copy tree | see map | full parity; one-property mutation reported by exactly that key | P1 | I |
| TP-012 | AC-4 drift | R + E | see map | `Synced`/`Hook`/`Registry`/`Drift` header, flagged rows, file wins | P1 | F |
| TP-013 | AC-5 view (real corpus) | R | `inflight`; compare to disk | all open ledgers; verdict list; requirements; component+path+header | P1 | F |
| TP-014 | AC-5 view (all four kinds) | E fixture | two components, root-tree ledger, paused unit, needs-changes review, Interviewing requirement | all four kinds, root labelled `root`, no backlog item in `inflight` | P1 | I |
| TP-015 | AC-6 lookup | E + R | see map | status/agent/deliverable/question+date | P1 | F |
| TP-016 | AC-7 leaving | E + R | see map | present while `active`, absent when `archived` | P1 | F |
| TP-017 | AC-9 planned | R | see map | ID/marker/file/sync time; none in `inflight` | P1 | F |
| TP-018 | AC-10 backlog freshness | E + H | see map | updated immediately via the hook; hand edit flagged | P1 | F |
| TP-020 | AC-15 lookup by ID/alias/slug | R | `unit GAL-F0002`, `unit <slug>`, `unit <alias>`; compare with the file header | one result, same ledger | P1 | F |
| TP-021 | M4 ruling | E scratch registry | mint (scratch) `falkor-chat:K-065` and `cobb:K-065`; call `identity.candidates` for each scope and bare; a planted `Depends on` join | disjoint candidate sets; no cross-join | P1 | F |
| TP-022 | Rename after move | E scratch registry | `wi move` a minted item; old citation still finds it | same single result | P2 | F |
| TP-023 | AC-11 proposal on the AC fixture | E | see map | waves/overlaps/criterion as the AC | P1 | F |
| TP-024 | AC-12 on the real backlogs and fixture | R + E | see map | every item without facts is "not plannable yet", missing facts named | P1 | F |
| TP-025 | AC-13 external blocker | E | see map | held back, blocker and backlog named | P1 | F |
| TP-026 | AC-14 no writes | R + E | see map | files, counts, `syncedAt` unchanged | P1 | F |
| TP-030 | AC-3 PostToolUse | H + E | see map | exit 0, graph row == file | P1 | F |
| TP-031 | AC-3 PostToolUseFailure | H + E | see map | same | P1 | F |
| TP-032 | Fast-path latency, several definitions | H | no-change call (a) to process exit, (b) to pipe EOF through `subprocess.run(capture_output=True)`, (c) with a 300 KB payload, (d) `Bash`-shaped payload, (e) warm vs cold page cache, (f) on the **real repo tree** with a scratch state dir and a stamp, (g) with more than one definition of "no change" (no stamp present / stamp newer); 30 runs each, p50/p95/max, plus a bare `bash -c :` baseline; Python not started (process accounting) | well under 50 ms, no Python | P1 | N |
| TP-033 | Real-repo cost of the pruned `find` | H | as (f) with `find` timings, file counts | recorded | P2 | N |
| TP-040 | Hook registration state | S | grep `.claude/settings*.json` for `post-tool-sync` | not registered (expected); recorded | P1 | F |
| TP-041 | Header honesty | R | read `Hook:` in a shared header against TP-040 | state the finding either way | P1 | X |
| TP-050 | Malformed ledger | E | ledger with no table; broken separator; duplicate unit ids; 1 MB cell; NUL bytes; invalid UTF-8; BOM; CRLF; missing header | no crash, other sources unaffected, warning or named failure, no partial state | P1 | X |
| TP-051 | Colliding backlog IDs | E | two `### K-001` in one file; the same scheme ID in two backlog files | error naming **both** paths, previous subgraph kept, other sources synced | P1 | X |
| TP-052 | NUL/oversized values and key hygiene | E | > 4096-byte `Unit`/`Item` keys, leading/trailing whitespace keys, `\t`/`\n` in a key | stripped or refused, never a crashed server; FalkorDB stays up | P1 | X |
| TP-053 | Unknown status token | E | `lint` and sync with an unknown `Status` | `lint` flags it in non-archived ledgers | P2 | F |
| TP-054 | Path traversal / odd file names in the hook payload | H | `file_path` with `..`, absolute outside the root, a symlink | not synced, no read outside the root | P2 | X |
| TP-055 | Kill switch mid-run | H | disable via `.state/disabled` and via env while a slow sync is in progress; then more edits | in-flight run completes; later calls exit 0 with a pending edit; header `Hook: disabled`; re-enable resyncs | P1 | X |
| TP-056 | Concurrent hooks | H | two hook invocations at once on one changed file; two on different files | both exit 0, graph equals file, no duplicate nodes | P1 | X |
| TP-057 | Concurrent syncs and edits | E | `sync --all` x2 concurrently; edit while syncing | no orphans/duplicates; `rebuild` parity ok afterwards | P2 | X |
| TP-058 | Outage handling | H + E | stop the container: exit 2 <= 5 s naming the file; 20 quiet calls; restart | as base plan §5 "Hook" | P1 | F |
| TP-060 | AC-8 add/delete claim | H + E scratch chat | see map | found immediately; deleted not returned | P1 | F |
| TP-061 | Unsynced KB edit flagged | E | edit with hook bypassed; `kb-status` | stale flagged | P1 | F |
| TP-062 | needs-resplit / duplicate claims | E | claim text duplicated under two headings; a heading crossing the chunk budget | `kb-sync` refuses duplicates naming both headings, first stays current; `needs-resplit` reported | P1 | F |
| TP-063 | degraded / failed-doc | E | a `failed` document that is embedded vs not | `degraded` informational; `failed-doc` re-posted and flagged | P1 | F |
| TP-064 | Pending path + kill mid-swap | E | slow embedder; kill the fake mid-swap | old claim still served and flagged, never lost | P1 | F |
| TP-065 | Shared KB status (read) | R | `kb-status --confirm coordination`; `search_documents` for a mapped claim | reflects the S8a state | P2 | F |
| TP-066 | KB writes refused toward :8200 in ephemeral mode | E | `COORDGRAPH_KB_URL=http://localhost:8200` with an ephemeral graph | refused before any call | P1 | X |
| TP-070 | GAL-F0003 build state | R | `--help`; `ls coordgraph`; grep for `ready`/`section`/`overlap` | recorded (absent) | P1 | F |
| TP-071..TP-075 | GAL-F0003 ACs | R/S | see 4.2 | not built -> not verifiable | P1 | F |
| TP-080 | Guard checks | E | refuse non-loopback, shared ports and :8200 without `--confirm`; `--confirm` mismatch refusal | refusals, no connection | P1 | X |
| TP-081 | Leak check | S | `docker ps` before/after; `git status`; scratch artefacts | no leaked containers of this run, no tree changes outside the two deliverables | P1 | F |

## 6. How AC-8 is verified (and what is not proven)

`falkor-chat:K-070` is closed (`falkor-chat/docs/HISTORY.md` 2026-11-20: embeds scheduled before extracts, embed retry),
but **its live swap reduction is not measured**, and this run may not write to the real `ws:agent-team`. AC-8 is therefore
verified by (1) the scratch falkor-chat + fake embedder legs (functional behaviour: immediate retrievability, deletion,
flagging, fail-fast, pending/finish, append-before-delete) and (2) read-only shared evidence (`kb-status`, mapped claims
searchable). **Not proven:** (a) AC-8 against the real backend under LM Studio load, including extraction time-outs that
leave documents `failed` (the S8a incident); (b) a one-chunk heading round-trip on the real `ws:agent-team`; (c) that K-070
shortens the real swap window. These need a human-approved real round-trip with a throwaway heading and are listed in the
report as unverified.

## 7. Environment and data

Container `falkordb/falkordb:v4.18.11` (ephemeral), `falkor-chat/server/.venv` + `redis-cli` for the scratch chat, the
component venv (`coordination-graph/.venv`). Copy tree: the tracked covered files of this repo (ledgers, reviews,
requirements, `BACKLOG.md`, `kaizen/plan.md`, `kb-files.txt`, the KB files) plus a copy of `.work-items/`, under the
scratchpad. Named fixtures are built in the scratchpad, never under a `docs/plans/`-shaped checked-in path.

## 8. Entry and exit criteria

Entry: offline suite green (1155) and live suite green (88) -> otherwise stop and report. Exit: every TP has a recorded
outcome with evidence; every AC has a verdict (pass / fail / partial / unverifiable / not built); defects are reproducible;
ephemeral containers started by the run are removed; `git status` shows only this run's two deliverables new.

## 9. Out of scope

Anything listed under "Deliberately not tested" in section 3; registering the hook or editing settings; running the S0
probe; archiving or triaging the 14 open verdicts (R3: a `teco`/stakeholder triage); `migrate --apply`; the GAL-F0003
feature itself (only its absence is tested).
