# Coordination graph — acceptance test report (S10)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** GAL-F0002 (coordination-graph S10) · **Extends:** `claude/docs/test-plans/coordination-graph.md` · **Item:** `GAL-F0002`

## 1. Summary

- **Tested:** `coordination-graph/` at repo HEAD `a81a597e` (working tree clean for that component), 2026-11-21, against both requirements documents. Plan: `claude/docs/test-plans/coordination-graph.md` (TP ids below).
- **Verdict:** **the delivered feature (GAL-F0002) passes acceptance on everything that can run today, with one major defect (D-1) and five minor ones. It is not fully accepted:** every criterion that needs the hook registered in a real session is *unverifiable until registration + S0*, AC-8 is verified on a scratch backend only, and the cross-initiative feature (GAL-F0003) is not built, so its six ACs are *not built*, not passed.
- **Method split:** shared mirror read-only (`--confirm coordination`, `mcp__cypher__query` reads); everything that writes ran on ephemeral FalkorDB containers (`falkordb/falkordb:v4.18.11`, loopback, ports 47311/47312 and the live fixture's own) with copy/fixture trees, a scratch state dir and a scratch copy of the work-item registry; the hook script was driven only through a shim that drops the literal `--confirm coordination` pair (the script hard-codes it). Nothing was registered in `.claude/settings*.json`; no commits; no write to the shared mirror, `ws:agent-team` or `:8200`.
- **CPG:** considered, not relevant — black-box acceptance of a CLI/hook over files and a graph; `GRAPHS` lists only `cpg_falkorchat` (no CPG of `coordination-graph/`), and no production-code-no-test-reaches question was asked.

## 2. Baseline (TP-000)

| Suite | Command (from `coordination-graph/`) | Result |
|---|---|---|
| offline | `.venv/bin/python -m pytest -q` | **1155 passed, 88 deselected** in 19.8 s |
| live (ephemeral only) | `.venv/bin/python -m pytest -q -m live` | **88 passed, 1155 deselected** in 73.2 s |

Both match the delivery record. No container from the live run survived. A container `coordgraph-test-3820761-747651` (up 11 h, started by a pytest session with pid 3820761 that is not this run) was already there and was left untouched: a leak of an earlier session, see D-6.

## 3. Per-AC verdicts

Legend: **PASS** observed here · **PASS (fixture)** observed on a fixture, no real-data evidence · **UNVERIFIABLE** needs a registered hook in a real session (S0 not run) · **NOT BUILT** · **PARTIAL**.

### 3.1 `claude/docs/requirements/coordination-graph.md`

| AC | Verdict | Evidence (TP) |
|---|---|---|
| AC-1 | **PASS** (TP-001) | Root `AGENTS.md` l.182-183: "Backlogs are mirrored in the coordination graph (...; the file stays the source) and headed there; history stays a file (`HISTORY.md`)". The plan's n4 grep over non-archived docs: 14 hits, all false positives ("already git history" in `kaizen/history.md`, the requirements/plan text that states the AC itself); no active document says history is headed for the graph. |
| AC-2 | **PASS** (TP-011) | 567-file copy tree, `bootstrap` + `sync --all` (3.0 s) + `rebuild`: `549 expected, 549 in graph, parity: ok`; property-hash snapshot (Source 549, Unit 1302, Item 246, HAS_UNIT 1302, HAS_ITEM 246). `MATCH (n) DETACH DELETE n` (0 nodes left; `drift` then `Drift: 549 source(s)` all `unsynced`), `rebuild`: snapshot byte-identical to the pre-wipe one (`IDENTICAL-AFTER-WIPE`). Not done: losing the shared instance. |
| AC-3 | **PASS at script level; real session UNVERIFIABLE** (TP-030/031) | Hook script, fixture tree, row edit then hook then `unit work U2`: `PostToolUse` -> `gated`, `PostToolUseFailure` -> `delivered`, each exit 0, equal to the file, `Drift: none`. First run with no stamp creates the stamp via `hook-sync --first-run`. Not run: a real Claude Code session, a Task-dispatched subagent, contexts where hooks fire (hook not registered, S0 probe not run). |
| AC-4 | **PASS** (TP-012) | Fixture: hand edit without hook -> header `Drift: 1 source(s)`, `changed <path> -> file wins`, `⚠ drift alpha:K-002 changed` on exactly the changed row in `planned`; a deleted file reported `file is gone`. Shared mirror right now (a natural experiment): `Drift: 16 source(s)`, 14 changed + 1 unsynced review, with `⚠ drift` on `falkor-chat:K-093` and one `live-spec-defects` unit; the oldest/newest sync time is printed (`11:18:50Z – 11:18:51Z (548 sources)`). |
| AC-5 | **PASS** (TP-013/014); asking an agent in a session **UNVERIFIABLE** | Shared `inflight`: 8 open coordinations with component + path each (`claude`, `root`, `falkor-chat`), 14 open "needs changes" verdicts, 9 requirements not picked up (5 `Ready for design`, 4 `Interviewing`), sync header. Cross-check: grep of `*-coordination.md` outside archive/staging finds 8 non-archived ledgers, identical to the view; `docs/plans/cross-initiative-coordination.md` (a plan with a `-coordination` name) is correctly not a ledger. The real corpus has **0 paused units** (`Waiting on the stakeholder (0)`, all 5 paused units are in archived ledgers), so the fourth kind was proven on the fixture: two components + a root-tree ledger (labelled `root`), a paused unit, a needs-changes review, an `Interviewing` requirement all listed; no backlog item in `inflight`. |
| AC-6 | **PASS** (TP-015) | Fixture `unit work U1` (paused): status, agent `a1`, deliverable, `question: which port? (asked 2026-11-10)`, no ledger opened. Shared `unit GAL-F0002` and `unit coordination-graph` both return `claude/docs/plans/coordination-graph-coordination.md`. Observation: a paused row written in a non-`teco.md` shape (question in the Status cell) shows `asked unknown` and the Deliverable as the question; honest, not silent. |
| AC-7 | **PASS** (TP-016) | Fixture: an all-`accepted` ledger with `Status: active` listed; after the flip to `archived` and sync, `inflight` no longer lists it, `unit rootwork` still answers with `(archived ledger: only key, label and status are kept)`; an archived ledger's paused unit never appears in `inflight`. |
| AC-8 | **PARTIAL: PASS on a scratch backend; real backend UNPROVEN** (TP-060..066, section 5) | See 3.3. |
| AC-9 | **PASS** (TP-017) | Shared `planned falkor-chat` (48 items), `planned cpg` (root `docs/BACKLOG.md`: 10 items = 3 grouping headings excluded, 8 planned + 2 deferred), `planned cobb` (30): each row `ref, marker, priority, title, source file`, sync header first, deferred/rejected in a trailing group. Item counts re-derived independently from `### ` headings in the 19 files: 249 headings, minus 3 grouping headings in `docs/BACKLOG.md` = **246 = `lint --count` = graph `:Item`**. Note: the root `docs/BACKLOG.md` is scope **`cpg`** (`planned root` refuses and lists the known scopes), so "root" in AC-9 reads as `cpg`. No backlog item appeared in `inflight` (fixture and shared). |
| AC-10 | **PASS at script level; real session UNVERIFIABLE** (TP-018) | Fixture: add / rewrite / remove an item, each via the hook script, then `planned` showed 15 -> rewritten title -> 14 items immediately; hand edit with no hook -> `⚠ drift` row. |
| AC-11 | **PASS (fixture)** (TP-023) | Fixture exactly as the AC: A in wave 1, B in `proposed-2` wave 2 `(a) after alpha:K-001`; C/D, G/H, I/J each in different waves with the overlap named (`(b) overlap with alpha:K-003 on file alpha/shared.py (changes)`, `... on graph ws:test (changes)`, `... on fact alpha/doc.md#name`); E (high) in the first milestone, F (low) later; every placement carries `(a)`/`(b)`/`(c)`. **No real-data evidence:** no real item records `Depends on` or `Touches` (TP-024). |
| AC-12 | **PASS** (TP-024) | Real: `propose cobb`, `falkor-chat`, `cpg` list every non-fixed open item under `Not plannable yet` with `missing Depends on, Touches` (and `Priority` for `cobb:K-021`); fixture: `K-011` -> `missing Touches`, `K-014` -> `missing Priority`. |
| AC-13 | **PASS (fixture)** (TP-025) | `alpha:K-012` depends on open `beta:K-001`: `Held: external blocker beta:K-001 in beta/docs/BACKLOG.md`; `alpha:K-013` depends on `beta:K-002`, delivered by a `## ... — K-002 delivered` history heading: placed. No real cross-backlog dependency exists. |
| AC-14 | **PASS for "no effect without approval"; approval half is design, not behaviour** (TP-026) | Shared: before/after `git status --porcelain` and sha256 of all 19 backlog/plan files identical across `propose cobb/falkor-chat/cpg`. Fixture: sha256 of every file, node count (31), edge count (21), `Source.syncedAt` unchanged across 3 proposals. `propose.py`/`commands_propose.py` contain no write path; "after the stakeholder's explicit approval they do change" has no mechanism in `coordgraph` (the stakeholder/`teco` edits the `Milestone` field), so it cannot be exercised here. |
| AC-15 | **PASS with a real-data caveat** (TP-020..022, M4) | Shared: `unit GAL-F0002` and `unit coordination-graph` resolve to one ledger. The real registry has **0 minted legacy items** (`mint-legacy` count 0), so the legacy-citation leg ran on a scratch registry copy: minted `falkor-chat:K-065` -> `GAL-F0035`, `cobb:K-065` -> `GAL-I02-K0092`. **M4 ruling holds:** `candidates("K-065","falkor-chat")` = `{GAL-F0035, falkor-chat:K-065}`, `candidates("K-065","cobb")` = `{GAL-I02-K0092, cobb:K-065}`: disjoint, neither contains bare `K-065`; bare with no scope stays `{K-065}` and `resolve` reports `ambiguous` listing every home. After `wi move GAL-F0035 GAL-I03` (-> `GAL-I03-F0035`): the old ID, the new ID and the legacy citation all give `{GAL-F0035, GAL-I03-F0035, falkor-chat:K-065}`. `migrate --apply` was never run. |

### 3.2 `docs/requirements/cross-initiative-coordination.md` (GAL-F0003)

| AC | Verdict | Evidence (TP-070..075) |
|---|---|---|
| AC-1 | **NOT BUILT** | `inflight` shows the ledger path but no one-line description (`parse/touches.py`, `overlap.py`, `ready.py`, `sections.py` do not exist; plan steps X1-X6, S7c open). |
| AC-2 | **NOT BUILT** | `teco.md`, `tico.md`, `architect.md` contain no overlap-check duty (0 hits for "overlap"); the three-transcript scenario cannot run. |
| AC-3 | **NOT BUILT** | No `needs` parsing beyond plan3's raw capture; no ready-now view to list the unit in. |
| AC-4 | **NOT BUILT** | `coordgraph --help` has no `ready`. |
| AC-5 | **NOT BUILT** | No `section` command. |
| AC-6 | **NOT BUILT** | `docs/plans/connected-code-graph-coordination.md` carries no `Needs`/`Touches` columns or description line. |

### 3.3 AC-8 and the KB legs (what is proven, what is not)

Scratch stack (own ephemeral FalkorDB, scratch falkor-chat, fake embedder, ws `coordtest`), hook script driving a registered scratch KB file:

| Step | Observed |
|---|---|
| add 2 one-chunk `##` headings, hook | exit 0 in 1.2 s; both claims returned by `/documents/search` immediately; 2 current documents |
| edit one heading via hook | exit 0 in 0.7 s; edited text found, the old text not current, the other claim untouched |
| delete a heading via hook | exit 0; the deleted claim no longer returned; 1 current document |
| edit with the hook bypassed | `kb-status` exit 2 `stale kb/a.md § Alpha`; the new text not served; the hook later catches it up (`kb-status` clean) |
| duplicate claim under two headings | hook exit 2 naming both (`Copy: refused (claim text duplicates 'Alpha' ...)`), the first heading's document stays current; `kb-status` lists `unmirrored` + `duplicate` |
| split heading with `<!-- kb-split: bullets -->` (3 docs), marker removed | re-ingested whole (2 docs), `kb-status` `coarse ... resplit at the next pass` (this is the plan's `needs-resplit`) |
| a `failed` document that is embedded | `kb-status` `degraded ... extraction failed, still searchable`; still returned by search; `kb-sync` did **not** re-post it (same id, still current) |
| embedder down, KB edit | hook exit 2 in 0.19 s naming the endpoint; document count unchanged (2 -> 2); `kb-status` `unmirrored`; after recovery the next hook call syncs it (exit 0) |
| `COORDGRAPH_KB_URL` = `:8200`, unset, or non-loopback, ephemeral mode | each refused (rc 1) before any call |
| shared, read-only | `kb-status --confirm coordination` rc 0, last sync 12:16:25Z, 17 `degraded` lines only (0 stale/unmirrored/duplicate/failed-doc); `search_documents` with the convention's prefix returns the mapped `qa-testing-techniques.md` claim as the top hit (score 0.307). |

**Not proven:** (a) AC-8 against the real backend under LM Studio load; the S8a incident (17 claims left `failed` by extraction timeouts) is the failure mode the scratch stack cannot reproduce; (b) one real round-trip on `ws:agent-team` with a throwaway heading (writes there were out of this run's rules); (c) that `falkor-chat:K-070` reduced the swap window: its own HISTORY says "not measured", and `GAL-I02-K0002` is the open item for exactly that live check; (d) hook-driven freshness in a real session. The 17 `degraded` claims on the shared instance are by design searchable; they are the residue of that incident.

## 4. Other plan items

| TP | Result | Evidence |
|---|---|---|
| TP-010 counts re-derived | **PASS** | By `git ls-files` (archive excluded): ledgers 98, requirements 54, backlogs 19 (6 `BACKLOG.md` + 13 `plan.md`), reviews 378 = graph `Source` 98/54/19/**377** + the one `unsynced` review (`docs/reviews/connected-code-graph-p1-57-impl.md`, which `drift` reports). Items 246 (see AC-9). Units 1302; the `statusToken` tally of a fresh `rebuild --dry-run` (abandoned 5, accepted 663, delivered 413, gated 53, in-flight 39, paused 5, queued 18, unknown 106 = 1302) equals the shared graph's except accepted 662 / in-flight 40: that is the drifted `coordination-graph-coordination.md` (S9 close edits since the load). Zero `unknown` in a non-archived ledger on the shared graph (Cypher). `lint`: 0 errors on the real repo; `history-coverage` 16% matched (66 of 423), consistent with the S1 probe. |
| TP-032 fast-path latency | **PASS; one deviation** | 40 runs each, hook script, scratch state dir with a fresh stamp, shim logging every call (always 0 Python starts, rc 0): bare `bash -c :` baseline p50 0.9 ms. **Real repo (4,517 files, pruned `find`):** to process exit p50 21.0 / p95 32.0 / max 34.8 ms; to pipe EOF p50 20.0 / p95 21.7 / max 22.8 ms; with a 300 KB `Write` payload p50 24.5 / p95 27.2 / max 27.7 ms; 567-file copy tree p50 6.4 ms; tiny tree p50 3.5 ms. Outage: 20 consecutive no-edit calls rc 0, empty stderr, max 8 ms. **Under 8 busy-loop CPU hogs on the 8 cores: p50 104.9 / p95 136.1 / max 147.9 ms** (D-5). Cold page cache not measured (no `sudo` for `drop_caches`). |
| TP-040 registration | **as expected: NOT REGISTERED** | No `post-tool-sync`/`coordination-graph` in `.claude/settings*.json` or `~/.claude/settings*.json`; `claude/scripts/audit-team.sh` check 15 independently prints `FAIL coordination-graph hook NOT REGISTERED correctly (PostToolUse missing ...; PostToolUseFailure missing ...)`. |
| TP-041 header honesty | **FAIL** (D-1) | Every shared view prints `Hook: active` while no hook is registered. |
| TP-050 malformed input | **PASS with notes** | 10 hostile ledgers synced with `0 failed`, FalkorDB stayed up: no table (0 units), broken separator (0 units, no warning, D-4), duplicate unit id (`U1`, `U1~2` + warning), 1.2 MB cell (stored whole, D-6), NULs (`5 NUL byte(s) replaced by U+FFFD` warning), invalid UTF-8 (replaced, no warning), CRLF (parsed), BOM (D-2), 5,000-char unit key (stored as `h:<sha256>`, key hygiene works), tab-padded key trimmed, empty file, unknown `Status:` (docStatus kept as written). |
| TP-051 colliding IDs | **PASS** | Two `### K-001` in one file: `lint` and `sync` fail naming both lines (`duplicate item K-001 (lines 7 and 131)`), rc 1, previous subgraph kept, `drift` says `does not parse: ...`. Same scheme ID `GAL-F0900` in two backlogs: second file fails `item GAL-F0900 headed in both beta/... and alpha/...: unique constraint violation ...; a wi move may be in progress`, first file synced, `lint` `duplicate-ref` names both paths, `⚠ drift ... added in file` on the failed one. |
| TP-052 key hygiene | **PASS** | See TP-050 (long key hashed; NUL replaced; no server crash). |
| TP-053 unknown status | **PASS** | `lint` exits 1 with `unknown-status` per unit in a non-archived ledger; the same cells in an archived ledger are not errors. |
| TP-054 payload/path traversal | **PASS with D-3** | Hook payloads with `../../../../etc/passwd` and `/etc/passwd` start no sync; `sync ../x /etc/passwd` refused (`not in the subpath`). Symlink case is D-3. |
| TP-055 kill switch mid-run | **PASS** | With a 3 s sync in flight, `.state/disabled` created at +1 s: the running sync completed (graph updated, rc 0). The next edit: hook exit 0, 0 Python calls, header `Hook: disabled`, `Drift: 1 source(s)` flagged; env `COORDGRAPH_HOOK_DISABLED=yes` also exit 0; after removal the next call synced the pending edit (stamp had not advanced). |
| TP-056 concurrent hooks | **PASS** | 6 simultaneous hook runs on one changed file: all rc 0, graph row equals file, exactly 1 `U9` node, 0 orphan units. |
| TP-057 concurrent syncs | **PASS** | 8 concurrent `sync --all`: all rc 0; `rebuild` afterwards `parity: ok`. |
| TP-058 outage | **PASS** | Container stopped: edit -> exit 2 in 0.10 s naming the file; second edit of the same file -> exit 2 again; 20 quiet calls rc 0, max 8 ms, 0 `retry-failed` forks inside the backoff; container restarted, next call after the 60 s window synced it and `failed.json` was `{}`. |
| TP-066 / TP-080 guards | **PASS** | Read commands refused: wrong confirm key, `COORDGRAPH_EPHEMERAL_PORT` together with `--confirm`, ephemeral var naming a shared port, no confirm on 6379. Not run on purpose: `bootstrap`/`rebuild` with only the env confirmation (a write command aimed at the shared instance; covered by the offline `tests/test_shared.py`). |

## 5. Defects

| ID | Severity | Title | Reproduce | Expected vs actual |
|---|---|---|---|---|
| D-1 | **Major** | Sync header says `Hook: active` while no hook is registered | From the repo root: `coordination-graph/coordgraph.sh drift --confirm coordination` (any view), then `grep -l post-tool-sync .claude/settings*.json` (no match), `bash claude/scripts/audit-team.sh` (check 15 FAIL) | The header is FR-5's "when to trust the file" signal; a reader sees `active` and may assume same-moment freshness (FR-4) that does not exist. `active` means only "kill switch not thrown" (`drift.py` l.92; README "Reading the sync header" documents it). Suggested: print `Hook: not registered` / `unverified` unless a registration check passes, or rename to `Hook kill switch: off`. Does not change the facts (drift is still listed, 16 sources now). |
| D-2 | Minor | A UTF-8 BOM before the `# ` title makes the header unparsed: an `archived` ledger shows as open | Fixture ledger beginning `EF BB BF # t`, `Status: archived`; `sync`, then `inflight` | Expected archived and absent; actual `docStatus` absent (null), no warning, the ledger is listed in `inflight`. CRLF alone is fine. No BOM exists in the real corpus today. |
| D-3 | Minor (hardening) | CLI discovery and `sync <path>` follow symlinks; the hook's `find -type f` does not | In a fixture root: `ln -s /etc/hostname alpha/docs/reviews/evil.md`; `coordgraph sync alpha/docs/reviews/evil.md` and `sync --all` | Both mirror it (`Source` row, rowCount 0) so the CLI reads a file outside the repo; the hook never notices the file. Needs write access to the repo to exploit. Suggested: reject symlinks (or paths whose realpath leaves the root) in `discover.py` and `sync`, matching the hook. |
| D-4 | Minor | A ledger whose table cannot be parsed yields 0 units with no warning | Fixture ledger: header `\| Unit \| ... \|` then a data row without a separator row, or prose only; `sync`, `lint`, `inflight` | Expected a warning like the plan's "pre-convention shapes -> warnings"; actual `warnings []`, `lint` clean, the coordination appears in `inflight` with no units. |
| D-5 | Minor (deviation) | Fast-path budget of < 50 ms holds only when the machine is idle | `lat.py` runs; 8 CPU-bound processes on 8 cores | Idle p50 20 ms (max 35); saturated p50 105 ms, p95 136 ms. The plan states the budget without a load condition and R2's mitigation (restrict the `find`) would apply. Not a defect while the typical load is idle; record the condition in the README. |
| D-6 | Minor / hygiene | Oversized cell stored whole; a leaked test container exists | (i) a 1.2 MB Deliverable cell synced and stored in full (`size = 1,200,000`); (ii) `docker ps` shows `coordgraph-test-3820761-747651` up 11 h | (i) RAM per cell is unbounded (the graph lives in RAM, plan §8); a cap with a warning is a cheap guard. (ii) the live fixture's `--rm` stop did not run for an earlier session (killed pytest); needs a human `docker stop`, I did not remove a container I did not start. |

Documented deviations (not defects): `planned` for root `docs/BACKLOG.md` is scope `cpg` (AC-9 wording says root); `unit` shows `asked unknown` when a paused row lacks `relayed <date>`; real backlogs record no `Depends on`/`Touches`, so every real proposal is "not plannable yet" (by design: no backfill, FR-14); history-coverage is 16%, so an external dependency on an item delivered before the heading convention reads as unknown.

## 6. Coverage, gaps, residual risk

- **Unverifiable until registration + S0:** same-moment freshness in a real session (AC-3, AC-10, AC-8's freshness leg); whether the hook fires for Task-dispatched subagents, `Bash` calls and in failed-tool contexts (R1); "ask an agent" answers (AC-5/6/9/11-14 as phrased in plan §5 "Acceptance"). Until registered, the mirror is stale by construction (16 drifted sources at the time of the run), and D-1 hides that in the header.
- **Unproven:** AC-8 on the real backend and K-070's swap reduction (3.3); an unattended hook against the shared instance (the real hook uses the literal confirmation key; only the shim path was run); cold-cache latency; Windows/macOS shells.
- **Not built:** GAL-F0003 (X1-X6, S7c); its ACs were checked only for absence.
- **No real-data evidence** for AC-11/13 (no recorded facts) and the legacy-ID leg of AC-15 (no minted legacy items): fixture/scratch registry only.
- **R3 triage is owed:** `inflight` lists 14 "needs changes" verdicts on day one (e.g. `falkor-chat/docs/reviews/workflows.md` pass 2, `docs/reviews/connected-code-graph-p1-18.md` pass 1); I did not judge which are stale; `teco` and the stakeholder triage them as plan R3 says.
- **Incidental, outside this component:** `search_documents` with `top_k=5` returned 20 chunks on `ws:agent-team`; not investigated.

## 7. Recommendations

1. Fix D-1 before the stakeholder registers the hook, or at the same time, so the header can say `active` truthfully; add a header test that fails when the registration check fails.
2. Registering the hook and running the S0 probe unlocks every UNVERIFIABLE verdict above; re-run TP-030/031/055/058 and plan §5 "Acceptance" in a real session afterwards (the harness is the scratchpad `hook.sh`/`lat.py` pattern: shim, scratch state, fixture root).
3. Schedule the real `ws:agent-team` round-trip with a throwaway one-chunk heading, and `GAL-I02-K0002`, with explicit approval.
4. Decide GAL-F0003: until X1-X6 land, its requirements stay "Ready for design" with no verification path.
5. D-2/D-3/D-4/D-6(i) are small parser/discovery guards; route to `coder`/`tdd-engineer` with the fixtures above as red tests.

## 8. Artifacts and cleanup

Created: `claude/docs/test-plans/coordination-graph.md`, this report. Scratch (not part of the repo): the copy tree, fixture roots, harness scripts (`lat.py`, `kbqa.py`, `hook.sh`, `shim.sh`) under the session scratchpad `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/7b73d98d-642c-4be0-8382-04d4981855aa/scratchpad/`. Containers `qa-s10-ephem` and `qa-s10-outage` and the scratch falkor-chat/fake embedder processes were removed; `git status` shows only the two deliverables new (plus other sessions' `cpg/tests/*` and `docs/test-*/connected-code-graph-p1*` changes, untouched). The real `coordination-graph/.state/` was not modified.
