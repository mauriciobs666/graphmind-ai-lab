# In-flight work, backlogs and reference lookup in the graph — implementation plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Version:** 2

Revised 2026-10-21 (v2): analyst Pass 1 (`claude/docs/reviews/coordination-graph.md`) and the
2026-10-21 decision-log answers folded in; Pass 2 findings and the 2026-10-22 decisions (insertion
ahead of approved milestones, milestone-table conversion) folded in on 2026-10-22, then Pass 3 findings (lease fencing, durable milestone start,
`failed.pending` locking, no-stamp ordering, map lint) — per-finding
disposition in Appendix A.

Implements `claude/docs/requirements/coordination-graph.md` (Ready for design, 2026-10-20).
Companion notes: `claude/docs/plans/coordination-graph-graph.md` (`graph-dba` — data model, write
path, indexes, rebuild, durability, first load; **canonical for every Cypher and schema detail**)
and the `cobb` consult folded into §3.5, §3.6 and §4 (common backlog-item format, KB split rule,
team-wide changes — cobb owns those steps).

## 1. Goal & scope

**Goal.** A FalkorDB mirror of the team's live work — non-archived coordination ledgers, open
review verdicts, not-yet-picked-up requirements, and every open backlog item — kept in step with
its Markdown files at the moment of each edit, plus a freshness guarantee for the existing
`ws:agent-team` reference-knowledge mirror. On top of it, one CLI answers four questions, each
answer stamped with sync time and drift flags: *what's in flight / waiting on me* (FR-6),
*this unit's state* (FR-8), *what's planned for X* (FR-12), *propose the next milestone(s) for X*
(FR-13).

**In scope:** FR-1–FR-15 / AC-1–AC-14 of the requirements doc; the common backlog-item format
(FR-14) and the one-time layout reformat of every backlog into it (decision log 2026-10-21, Q1/Q1b);
the team-wide prompt/skill/convention changes it entails; the ongoing freshness mechanism for the
KB mirror (FR-10).

**Out of scope** (requirements §Out of scope, restated only as boundaries for the implementer):
history files in the graph; graph-as-authoritative backlog home; analytics over past work;
unit↔knowledge links; search-quality changes; terminal-command / falkor-chat page / falkor-chat
chat channels (later deliveries — the CLI built here is *usable* from a terminal, but it is not
designed or supported as the FR-7 terminal channel); **the one-time repair of the KB drift already
on record** (cobb K-058 — consumed as a prerequisite, step S2, not redone); any falkor-chat
**server** code change — the one falkor-chat file this plan touches is a roster row in
`falkor-chat/scripts/seed_agent_team.sh` (S6) (falkor-chat:K-070 is scheduled alongside as falkor-chat's own item, §6 Q7, not part of this plan).

## 2. Context & findings

CPG: considered, not relevant — the code here is a new standalone component plus Markdown/hook
edits; the only loaded CPG (`cpg_falkorchat`) covers falkor-chat, which this plan calls only
through its existing REST surface and a read-only graph read; its server code is not modified
(the one falkor-chat edit is a roster row in its seed script, S6).

### 2.1 The sources (measured 2026-10-20)

| Source | Where | Shape today | Live count |
|---|---|---|---|
| Coordination ledgers | `<component>/docs/plans/<slug>-coordination.md` (root `docs/plans/` too) | Header line `> **Status:** … · **Owner:** … · **Tracks:** …`; one or more Markdown tables. Header row variants: `Unit\|Owner\|Agent id\|Status\|Deliverable\|Gate → verdict\|Cost` (46), same without `Cost` (14), `Agent` for `Owner` (3), `Agent (scope)` (1), and 2 pre-convention shapes with no `Status` column. A `paused` row puts the question + relay date in `Deliverable` (`claude/teco/teco.md` step 2). Vocabulary is teco's 7 values `queued·in-flight·delivered·gated·accepted·abandoned·paused`, but the **cells are free text: 362 distinct values** over 74 ledgers (`**in-flight** — dispatched at …`, `**accepted — APPROVE WITH SUGGESTIONS**`, `✅ delivered 2026-07-28 (…)`, `**relayed**`, `—`; review Appendix B). One file holds 30 tables. | 74 ledgers, **1 non-archived** (`falkor-chat/docs/plans/salesperson-language-salience-coordination.md`) |
| Review docs | `*/docs/reviews/*.md` | Header `Status:`; verdict lines vary (`**Verdict:** needs changes`, `Verdict (Pass 1): needs changes`, `NEEDS CHANGES`); re-reviews add `## Pass N` sections. | 168 total, **85 still `active`**; ~10 end on "needs changes" though most were resolved by later work (R3). |
| Requirements | `*/docs/requirements/*.md` | Header `Status:` — `Interviewing` / `Ready for design` / `superseded` / `archived` / `active`. | 12 at `Interviewing`/`Ready for design`; 3 of those already have a same-slug coordination. |
| Component backlogs | `docs/BACKLOG.md` (cpg, `C-NNN`), `falkor-chat/docs/BACKLOG.md` (milestone table + `### K-0NN — … (🔵 …)` sections), `model-bench/`, `opencode/`, `mcp-monitor/` (plain bold-lead bullets, **no ids**). | Five different shapes. | 5 files |
| Agent plans | `claude/<agent>/kaizen/plan.md` × 13 | Summary table `\| ID \| Added \| Priority \| Status \| Summary \|` + `### K-0NN — …` detail sections; 12 of 13 also carry a `## Parking lot / ideas` section of bold-lead bullets (proposals mixed with settled-judgment records such as "Judged and kept, do not re-litigate"). Priority recorded; deps/touched areas nowhere. | 13 files; all backlogs ≈ 0.3 MB |
| KB files (FR-10) | `claude/<agent>/<topic>.md` on-demand knowledge bases | Mirrored into `ws:agent-team` one document per `##` heading, some headings hand-split into several claim documents. Mapping in `claude/cobb/scripts/kb-claim-manifest.json`: 8 files under `files`, **5 more** only under `_`-prefixed keys (teco coordination-techniques, tdd-engineer guard-testing, data-scientist lm-studio, graph-dba falkordb-quirks, analyst review-techniques — K-058 (c)). **110 of 196 `##` sections exceed one 1,000-char chunk** (`falkor-chat/server/falkorchat/chunking.py:split_into_chunks`, size 1000/overlap 150). `claude/graph-dba/falkordb-quirks.md` carries an uncommitted edit today, so it is already drifted. | 366 documents |

Existing FalkorDB graphs (`GRAPHS`, 2026-10-20): `cpg_falkorchat`, `reference`, `ws:acme`,
`ws:agent-team`, `ws:demo`, `ws:eval`, `ws:k069probe`, `ws:nlq-eval`, `ws:test`.

### 2.2 Write paths that already exist — verified unfit

- **`mcp__cypher__query`** — `cypher-mcp/server.py:authorize_write` (line 564) accepts exactly six
  `KaizenEntry`/`Agent` write shapes; mirroring through it would mean adding bulk
  replace-a-subgraph shapes to a security-shaped authorizer. **Unfit for writes.** Its *read* path
  (`GRAPH.RO_QUERY`) is unrestricted; `coordgraph query` (§3.4) wraps the same kind of read.
- **falkor-chat `ingest_document`** — `services.ingest_document` writes a `Document` at
  `status:'processing'` and schedules per-chunk embedding **and** LLM extraction plus update
  detection; the only update primitive is exact-normalized-hash auto-supersede
  (`repository.create_document_with_auto_supersede`, which reports `autoSuperseded` /
  `supersededDocumentId` in the receipt). **Unfit for ledgers/backlogs; the right path for KB
  claims**, which must stay embedded and searchable.
- **KB search readiness** — `repository.search_chunks` filters `Chunk`s on `documentCurrent = true`
  and needs `Chunk.embedding`; it does not wait for `Document.status = 'ready'` (LLM extraction).
  So a claim is retrievable once **all its chunks** carry `embedding`.
- **No REST view of chunk embedding state** — `GET /documents/{id}` returns only `status`
  (`repository.get_document`); chunk state is "not part of the public MCP/REST surface"
  (`services.list_document_chunks` docstring). The only way to observe embed readiness is a
  **read-only graph read** of `ws:agent-team`: `(d:Document)-[:HAS_CHUNK]->(c:Chunk)`, property
  `c.embedding` (`repository.py:1085`, `:1261`), `d.currentVersion`, `d.textNormalizedHash`.
- **Embed order inside one document** — `background.py:_schedule_chunk_processing` schedules, per
  chunk, embed then extract on the request's Starlette `BackgroundTasks`, which run one after
  another: **chunk k's embed waits behind chunk k−1's LLM extraction.** A one-chunk claim is
  searchable in ~an embed call; a multi-chunk claim (most of today's sections, §2.1) only after
  every preceding extraction. falkor-chat:K-070 ("Schedule a document's embeds before its
  extracts", 🔵 proposed) would remove this.
- **falkor-chat REST** (`falkor-chat/server/falkorchat/api.py`, `ws:agent-team` process on port 8200
  via `falkor-chat/scripts/start_agent_team.sh`): `GET /health` (process + FalkorDB),
  `POST /documents` (accepts `producedBy`), `DELETE /documents/{id}`, `GET /documents/{id}`. The
  embedder's own base URL is `FALKORCHAT_EMBEDDING_BASE_URL` (`config.py:58`).

### 2.3 The trigger seam — Claude Code hooks (`code.claude.com/docs/en/hooks`, re-checked 2026-10-21)

- Hooks in a **settings file** fire for subagents too ("tool events such as `PreToolUse` and
  `PostToolUse` fire the same configured hooks as in the main conversation").
- `PostToolUse` is "After a tool call succeeds"; **`PostToolUseFailure`** is "After a tool call
  fails". Exit 2 on either: "Shows stderr to Claude". **Unverified:** whether a `Bash` call that
  exits non-zero counts as a failure — S0 probes it; this plan registers both events, so either
  answer is covered.
- A command hook blocks unless `async: true`; `timeout` is per handler.
- **Worktrees:** `${CLAUDE_PROJECT_DIR}` "still points at the project root where the session
  started"; the payload's `cwd` follows Claude into a worktree.
- Local caveat: `skills/agent-standards/claude-code.md` (K-019) records that `PreToolUse` **ask**
  decisions did not pause under Auto Mode — a permission-decision finding, not evidence that
  `PostToolUse` command hooks fail to run. Still unverified here → S0.
- Agents edit files through `Write`/`Edit` **and** `Bash` (`sed -i`, heredocs, scripted ledger
  replaces — `claude/teco/coordination-techniques.md`; `git checkout`/`merge`). The matcher must
  include `Bash`.

### 2.4 Persistence

The local FalkorDB (`falkordb/falkordb:v4.18.11`) is RDB-only — `appendonly no`, `save 3600 1 300
100 60 10000` — verified live by `graph-dba` (note §6.1); realistic loss window ≤ 60 minutes. One
RDB snapshot covers every graph on the instance, so the mirror and `ws:agent-team` roll back to the
same instant. Kept as is (decision log 2026-10-21, Q6).

### 2.5 Adjacent work

- `docs/requirements/cross-initiative-coordination.md` (`Interviewing`) — `tico` re-scopes it
  against this feature once this plan is approved (decision log 2026-10-21, Q4). Not a dependency.
- `claude/cobb/kaizen/plan.md` K-058 — the KB drift on record; this plan takes the
  ongoing-freshness half and leaves the one-time repair with cobb (S2).
- falkor-chat:K-070 — embeds before extracts; shortens KB retrievability for multi-chunk claims
  (§3.5); scheduled alongside S6 as a prerequisite of S10's AC-8 acceptance (§6 Q7).
- `.claude/settings.json` already puts `Edit(**/docs/BACKLOG.md)` on `ask` (AC-14, §3.7).
- **Write guards** (verified 2026-10-21): `claude/cobb/hooks/guard-cobb-topic-writes.sh` allows
  `claude/*/*.md`, `claude/*/kaizen/{plan,history}.md`, `claude/{README,AGENTS,CLAUDE}.md`,
  `skills/{agent-maintenance,agent-standards,python-web-quirks}/*`, `skills/README.md`,
  `cypher-mcp/README.md` — **not** `skills/coordination-graph/*`, `skills/agent-kb-retrieval/*`,
  `claude/scripts/audit-team.sh`, `claude/cobb/scripts/*`, or anything under `coordination-graph/`;
  the guard script itself is outside it too. Resolved by step S1c (§4).

## 3. Design & rationale

### 3.1 Shape of the solution

One new top-level component, **`coordination-graph/`**, a Python 3.12 package with one runtime
dependency (`falkordb>=1.6,<1.7`, the pin `cypher-mcp/` already verified) in its own venv
(`setup.sh`, the `cypher-mcp/setup.sh` pattern). One entry point, `coordination-graph/coordgraph
<subcommand>`:

| Subcommand | Does | Serves |
|---|---|---|
| `bootstrap` | Idempotent index/constraint creation (note §4.3) | — |
| `sync [PATH…]` / `sync --changed` / `sync --all` | Parse the named / changed / all mirrored files and replace each file's subgraph | FR-4, FR-11 |
| `rebuild` | Unconditional per-file replace, prune, parity check (no `GRAPH.DELETE` — note §5.1) | FR-3, AC-2 |
| `drift [PATH…]` | Every covered file whose content no longer matches what was synced, row by row | FR-5, AC-4/AC-10 |
| `inflight [--waiting]` | Units, paused questions, open "needs changes" verdicts, unpicked requirements | FR-6, FR-9, AC-5, AC-7 |
| `unit <ledger-slug> <unit-id>` | One unit's status, agent id, deliverable, question + date; every repeat of a repeated label, and every ledger sharing the slug, with paths (note §7.3) | FR-8, AC-6 |
| `planned <component\|agent> [--all]` | Open backlog items of one backlog: `🔵`/`🟡` items, then `⚪` deferred/rejected as a separate trailing group (count + ids; full entries with `--all`) so parking-lot records never read as planned work (P2-m1) | FR-12, AC-9 |
| `propose <component\|agent> [--per-milestone N]` | Read-only milestone proposal | FR-13–FR-15, AC-11–AC-14 |
| `query '<cypher>' [--covers PATH…]` | Ad-hoc `RO_QUERY` on the mirror key, printed under the sync header | FR-5 for ad-hoc reads |
| `kb-sync [PATH…]` / `kb-status [--ids ID…]` / `kb-reconcile` / `kb-finish` | KB mirror freshness | FR-10, AC-8 |
| `hook-sync PATH…` / `retry-failed` | The hook's foreground sync and its detached backoff retry (§3.2) | FR-4 |
| `lint [--count]` / `migrate --dry-run` / `facts [--like X]` | Backlog format check, one-time reformat diff, fact-token listing (§3.6) | FR-14 |

Every read subcommand prints a **sync header** first (§3.3). A Claude Code `PostToolUse` /
`PostToolUseFailure` hook syncs after every tool call that changed a mirrored file (§3.2). Agents
reach the views through a new skill, `skills/coordination-graph/SKILL.md` (§3.4).

**Configuration** (m5) — one module, `coordgraph/config.py`, every value env-overridable so tests
never touch shared state: `COORDGRAPH_REPO_ROOT` (default `$CLAUDE_PROJECT_DIR`, else the
script's repo), `COORDGRAPH_FALKORDB_HOST`/`_PORT` (default `localhost:6379`), `COORDGRAPH_KEY`
(default `coordination`), `COORDGRAPH_STATE_DIR` (default `coordination-graph/.state/`,
gitignored), `COORDGRAPH_KB_URL` (default `http://localhost:8200`), `COORDGRAPH_KB_GRAPH` (default
`ws:agent-team`), `COORDGRAPH_EMBED_URL` (default: `FALKORCHAT_EMBEDDING_BASE_URL` if set, else the
value `start_agent_team.sh` uses), `COORDGRAPH_KB_BUDGET_S` (default 10), `COORDGRAPH_KB_HTTP_TIMEOUT_S` (default 10; must stay < lease TTL/3, §3.5), `COORDGRAPH_HOOK_DISABLED`.

**Why a new component and not `claude/scripts/` or falkor-chat.** `claude/scripts/` is cobb's
hook/audit machinery; a coder-owned package with venv, tests and docs there would sit inside
another specialist's tree (and its `docs/` would land under `claude/docs/`, whose `*/docs/plans/*`
paths both broad implementers' write guards deny-list — `claude/coder/hooks/guard-coder-broad-write.sh`).
Building the mirror into the falkor-chat server would make *ledger* sync depend on that process
and couple a team-process tool to a chat product's release cycle. The `cypher-mcp/`,
`mcp-monitor/` precedent has neither cost.

### 3.2 Same-moment freshness (FR-4) — the post-edit hook

**Registration** — one project-level block in `.claude/settings.json`, the same handler on both
events (M1):

```json
"PostToolUse":        [{ "matcher": "Write|Edit|Bash", "hooks": [{ "type": "command",
  "command": "\"$CLAUDE_PROJECT_DIR\"/coordination-graph/hooks/post-tool-sync.sh", "timeout": 60 }] }],
"PostToolUseFailure": [{ "matcher": "Write|Edit|Bash", "hooks": [{ "type": "command",
  "command": "\"$CLAUDE_PROJECT_DIR\"/coordination-graph/hooks/post-tool-sync.sh", "timeout": 60 }] }]
```

A `Bash` call that writes a ledger and then exits non-zero is therefore covered whichever event
Claude Code classifies it under; the script is idempotent, so a double firing costs a no-op.

**State** (`$COORDGRAPH_STATE_DIR`): `stamp` (mtime marker), `failed.json` (per path:
`sha256, error, attempts, nextRetryAt`), `failed.pending` (one path per line — the shell's
no-Python append, made **under `flock "$STATE/state.lock"`** (util-linux), and merged then truncated
by Python under the same lock, so no append can land between the read and the truncate — P3-m2), `retry-due` (epoch seconds of
the earliest due retry, testable without Python), `state.lock` (guards `failed.json`
read-modify-write, held for milliseconds), `retry.lock`, `kb-finish.lock` (single-instance guard
for `kb-finish` only — never waited on by anyone else), `hook.log`, `disabled`. **Every path in
`failed.json`/`failed.pending` is repo-relative**; an absolute payload `file_path` is converted by
stripping `$ROOT/` (a path outside `$ROOT` is ignored).

**Detached children never hold the hook's pipes** (P2-M1). Every background launch is written as
`setsid <cmd> >>"$STATE/hook.log" 2>&1 </dev/null &` in the shell, and as
`subprocess.Popen(cmd, start_new_session=True, stdin=DEVNULL, stdout=log, stderr=log)` in Python —
otherwise the hook's stdout/stderr stay open until the child exits, and if Claude Code waits for
pipe EOF (inferred, not verified) a 15-minute `kb-finish` would hold the call to the 60 s timeout.

**`post-tool-sync.sh`** (bash; no Python on the fast path):

1. **Kill switch** (m6): `COORDGRAPH_HOOK_DISABLED=1` or `.state/disabled` exists → `exit 0`. The
   sync header then reads `Hook: disabled` (§3.3), so a disabled hook is never invisible.
2. **Root** is `$CLAUDE_PROJECT_DIR` — never the payload's `cwd` (m4). The mirror is the **main
   checkout's** state; an edit inside a worktree is not mirrored until it is merged into the main
   checkout (by a `Bash` `git merge`, which the hook sees, or outside a session, which §3.3 flags).
3. `start=$(date +%s.%N)`. **No stamp** (fresh clone, `.state/` cleaned — P2-M2a) → C = every
   discovered file and `hook-sync` runs with `--first-run`: it **compares hashes first** (files
   against `Source.sha256`, KB sections against their synced hashes) and keeps in C only files that
   differ, so step 5.1's fail-fast health checks apply only to those — a call that edited nothing
   never warns about untouched KB files (P3-m3). If FalkorDB itself is unreachable it exits 0
   silently, leaves the stamp absent and sets `retry-due` to now + 60 s; the shell runs the first
   run again only once `retry-due` has passed, never on every call. The stamp is created at step 6
   of the first run that reaches FalkorDB. Otherwise `find "$ROOT"` over the discovery set of §3.3, pruning
   `.git`, `node_modules`, `.venv`, `deprecated/` and `archive/` before descending,
   `-newer .state/stamp` → candidate set **C**. For a `Write`/`Edit` call, also extract
   `tool_input.file_path` from the payload (bash regex, no `jq`) and add it to C when it is a
   mirrored path present in `failed.json` or `failed.pending` (closes the cross-session race, R12).
4. **C empty** → if `retry-due` ≤ now, **first rewrite `retry-due` to now + 60 s** (P2-M2c — a
   crashed retry then costs one fork per minute, not one per call), then launch
   `coordgraph retry-failed` detached (as above; it takes `retry.lock` with `flock -n` and exits if
   held); `exit 0`. Nothing runs in the foreground.
5. **C non-empty** → `timeout 50 coordgraph hook-sync <C>` in the foreground, one Python process
   (the shell-side `timeout` keeps the shell alive to handle step 5.6 before Claude Code's 60 s):
   1. **Fail fast** (B1): FalkorDB connect + `PING`, 1 s timeout. If C holds a KB file:
      `GET $COORDGRAPH_KB_URL/health` and `GET $COORDGRAPH_EMBED_URL/models`, 2 s each. A dependency
      that is down marks every file it serves *failed* without attempting a sync.
   2. Coordination files: hash-compare, then per-file replace (§3.8).
   3. KB files: §3.5 `kb-sync`, section by section under the per-section lease of §3.5 (no
      file-wide lock), bounded by `COORDGRAPH_KB_BUDGET_S`: a section whose lease is held, or that
      is not retrievable within the budget, is *pending*, completed by a detached `kb-finish`.
   4. Under `state.lock`, merge `failed.pending`, then update `failed.json`: success removes the
      entry; failure/pending stores it with backoff 1, 2, 4, 8, then 15 min (capped); rewrite
      `retry-due`.
   5. **Exit 2 if any file of C failed or is pending**, stderr naming each file, the reason, and
      "this update is not complete — say so; it is retried automatically, or run
      `coordgraph sync <file>` once <dependency> is back". Else exit 0.
   6. **`hook-sync` itself fails** (P2-M2b — venv missing, Python exception, `timeout` kill: any
      exit other than 0 or 2) → the shell appends every path of C to `failed.pending`, sets
      `retry-due` to now + 60 s, and exits 2 naming those files ("coordgraph failed: <exit code>;
      see `hook.log`"). The hook's own fault is never read as "nothing to sync".
6. **Advance the stamp to `max(stamp, start)` always** (B1; created here when missing) — never to
   "now" (a write landing during the sync keeps a newer mtime and is caught next call), and never
   held back by a failure: failure state lives in `failed.json`/`failed.pending`, so a failing file
   does not re-enter C on unrelated calls. It comes back only through the detached backoff retry or
   a new edit of it.

**`retry-failed`** (detached): merges `failed.pending`; **drops an entry whose file is gone** (§3.3
reports it *gone*; `sync --all` prunes it) **or whose current hash already equals the graph's
`Source.sha256`** / the section's synced hash (P2-m4); skips KB sections still inside their
`kb-finish` cap (`pendingSince` + 15 min — `kb-finish` owns them, P2-M3); syncs the rest exactly as
`hook-sync` does; rewrites `retry-due`.

**Failure contract** (decision log 2026-10-21, failure behaviour): **every edit to a mirrored file
made while its sync cannot run gets exit 2** — no throttle (v1's 10-minute throttle is gone; this
also closes M8). A call that edits no mirrored file never warns and never waits; retries are
detached and backed off. Foreground cost of an edit during an outage is the fail-fast checks
(**≤ 5 s** worst case: 1 + 2 + 2 s), never the KB embed wait; foreground cost of a KB edit with
everything up is at most `COORDGRAPH_KB_BUDGET_S`. The handler `timeout: 60` is a backstop only.

`hook-sync`/`sync --changed` replace a source only when its sha256 differs from the stored one, so
`-newer` false positives (`touch`, a `git checkout` of identical bytes) cost a hash. Sync is
idempotent and converges on the file: two racing syncs of one file end in the state of the file.

**Why this meets FR-4 / AC-3.** A command hook blocks the tool result until it exits, so by the
time an agent can say "ledger updated", the graph already holds the new row — or the agent has
been told, in the same turn, that it does not. The only window in which file and graph differ
unflagged is the few hundred milliseconds between the file write and the hook's write; a reader
landing in it gets the row flagged by §3.3.

**Not covered by the hook, by construction:** an edit outside Claude Code (AC-4's case); a file
**deletion** (`find -newer` cannot see one — §3.3 reports it *gone*, `sync --all` prunes it); a
worktree edit before merge (step 2); any context S0 shows the hook does not fire in (decision log
Q2: accepted, FR-5 flagging there). All are visible through §3.3.

**Rejected alternatives.** *A watcher daemon*: one more always-on process under WSL2, no "done"
signal for the editing agent; ruled out by the stakeholder even as an S0 fallback (decision log
Q2). *A git pre-commit hook*: docs-only chains commit only at their terminal state
(`claude/AGENTS.md`), so the graph would lag by a whole chain. *Prompt-level "run sync after you
edit"*: the mechanism that drifted for the KB (K-058); ruled out (Q2). *Self-healing on read*:
rejected (decision log, FR-5). *`async: true` / `asyncRewake` for the KB leg*: would remove the
foreground budget but give the editing agent no in-turn answer; the detached `kb-finish` plus an
exit-2 "pending" message keeps the agent told without blocking.

### 3.3 Visible staleness (FR-5) — the read-time check

Every read subcommand, before printing, runs the same check over the sources its answer covers:

1. **Discovery** — exactly the note's §3.1 glob set (n5), one form everywhere, Python
   `glob(..., recursive=True)` semantics: `**/docs/plans/*-coordination.md`, `**/docs/reviews/*.md`,
   `**/docs/requirements/*.md`, `**/docs/BACKLOG.md`, `claude/*/kaizen/plan.md`; plus the files in
   the KB registry (§3.5). Excluded: `deprecated/**` and any path with an `archive/` segment.
   `discover.py` additionally prunes `.git`, `node_modules` and `.venv` while walking — a pure
   speed measure (none can hold a source), shared with the hook's `find` (§3.2 step 3). Globs, not
   a list, so a new component's backlog or ledger is discovered without a code change.
2. **Compare** each covered file's sha256 with its `Source.sha256`: *changed*, *unsynced* (no
   `Source`), *gone* (`Source`, no file).
3. For every *changed* file, re-parse with the sync's parser and diff by row key (note: unit key,
   item key, each with a row hash) → rows *changed* / *added in file* / *removed from file*.
4. Print the **sync header**: `Synced: <oldest> – <newest syncedAt> (N sources) · Hook: active|disabled`,
   then `Drift: none` or the flagged sources and rows with `→ file wins; run coordgraph sync`;
   flagged rows in the body carry an inline `⚠ drift`.

The check labels, never repairs. Covered text is < 1 MB; hashing is milliseconds.

### 3.4 How agents and the stakeholder reach the views (FR-7, FR-8, FR-12)

- **`skills/coordination-graph/SKILL.md`** (cobb, S5), `description` routing on "what's in
  flight", "what's waiting on me", "what's blocked", "what's planned for X", "propose a milestone
  for X", "status of unit U3 in <ledger>". Body: the subcommand table, "relay the answer **with its
  sync header**", "a drift flag means quote the file for that row".
- `teco`'s and `tico`'s resume step uses `coordgraph unit` / `inflight` instead of reading the
  whole ledger (US-2); they still *edit* the file and the hook mirrors it.
- **Ad-hoc questions** (m3) go through **`coordgraph query '<cypher>'`**, which runs
  `GRAPH.RO_QUERY` on the mirror key and prints the full sync header (all covered sources, or only
  `--covers` ones). The skill points at it, not at raw `mcp__cypher__query`, so FR-5 holds for
  ad-hoc answers mechanically rather than by memory.

### 3.4a What the in-flight view counts (FR-6, FR-9)

All status tests use the **normalized unit status** (`statusToken`, one of teco's 7 values or
`unknown`; raw cell kept as `statusText`; mapping canonical in note §2.5 — notably `relayed` →
`paused`, which is how all 5 real paused rows are written today), never the raw cell.
A row whose cell maps to `unknown` is still listed, with its raw text and a parse warning — it never
silently drops out.

- **(a) units** — every `:Unit` of a ledger whose `Status:` is not `archived` (any unit state:
  FR-9 / AC-7).
- **(b) waiting on the stakeholder** — those units with `statusToken = paused`, with question and
  relay date parsed from `Deliverable`; an unparseable date shows as unknown with a warning.
- **(c) open "needs changes" verdicts** — a review not `archived` whose **latest** verdict (last
  `## Pass N`, else its single verdict) normalizes to *needs changes*, **unless** it is resolved:
  a ledger row citing that review path has `statusToken` `accepted` or `abandoned`, **or** a ledger
  citing it is `archived`. Implemented over two per-ledger lists the parser emits for **every**
  ledger, archived or not — `citedReviews` (review paths cited anywhere in its rows) and
  `resolvedReviews` (those cited by an `accepted`/`abandoned` row) — because an archived ledger keeps
  no `:Unit` children. Model and query: `Ledger.citedReviews`/`resolvedReviews`, note §2.2 and
  §7.2(c). Residue: R3.
- **(d) requirements not yet picked up** — docs at `Interviewing`/`Ready for design` with no
  ledger of the same topic slug anywhere, any status (note §7.2(d)).

Each kind is listed with **component**, source path and the sync header (AC-5). Component is the
first path segment, except the repo-root `docs/` tree: `docs/BACKLOG.md` → `cpg` (root
`AGENTS.md`), root `docs/{plans,reviews,requirements}/*` → `root` (m1 — that tree also holds
model-bench, salesperson and kaizen-team work, which must not be filed under `cpg`).

### 3.5 Reference-knowledge freshness (FR-10, AC-8)

The KB mirror keeps its home (`ws:agent-team`), document shape (one document per `##` heading, or
per split sibling) and search path (`search_documents`, the `agent-kb-retrieval` convention).
What changes is *when* it is synced and *how a search learns it is stale*.

- **Trigger:** the hook (§3.2). A KB file is any file in the registry,
  **`coordination-graph/kb-files.txt`** (repo-relative paths; cobb-maintained — inside cobb's
  guard after S1c; seeded by S2).
- **Diff by heading:** split at `##`, hash each section body, compare with the hash at last sync
  → *changed*, *added*, *removed*. Untouched sections cost nothing.
- **Write path = REST; observation path = read-only graph read** (M4). `kbsync.py` ingests and
  deletes through falkor-chat REST (stdlib `urllib`), and observes embed readiness, current-version
  state and normalized hashes by `RO_QUERY` on `$COORDGRAPH_KB_GRAPH` through the same `falkordb`
  client the mirror uses — the only route that exists (§2.2). The read statement is note §6.4's
  (`embedded = chunks > 0` is the embed-landed poll). This couples the tool to falkor-chat's schema (`HAS_CHUNK`,
  `Chunk.embedding`, `Document.currentVersion`, `textNormalizedHash`) — R10, pinned by an S6
  contract test.
- **Append before delete** (never lose a claim): for a changed/added section, `POST /documents`
  (`producedBy: kb-mirror`, title per `skills/agent-maintenance/SKILL.md` §5 step 5) → record the
  section `pending` with old and new ids (note §2.4 two-phase marker) → poll until every new chunk
  has `embedding`, within `COORDGRAPH_KB_BUDGET_S` → then `DELETE` the old id(s) and mark `synced`.
  Removed section: `DELETE`, drop the record.
- **Budget exceeded** (the common case for multi-chunk sections until falkor-chat:K-070, §2.2):
  the section stays `pending` — **old claim still served, flagged stale by `kb-status`, new one
  submitted** — and a detached `coordgraph kb-finish` (15-minute cap per section) completes the
  swap when the embeds land. The hook exits 2 naming the section ("submitted, not yet searchable;
  old version served and flagged"). If the new document fails or the cap passes, `kb-finish`
  deletes the *new* document and leaves the section flagged; it is retried by the backoff of §3.2.
- **Concurrency unit = the section** (P2-M3). Three actors can touch one section — `kb-sync` from
  an edit, `retry-failed`, `kb-finish` — so there is **no file-wide KB lock**. Every step that
  changes a `:KbSection`'s mapping first takes a short **lease** on it by an atomic compare-and-set
  in the mirror graph (set `leaseToken`/`leaseUntil` = now + 60 s only if the lease is free or
  expired; one statement, so FalkorDB's serialized writes make it a CAS) and releases it after;
  a section whose lease is held is reported *pending* with a 30 s retry, never waited on.
  `kb-finish` holds no long lock: it loops over `pending` sections, taking each one's lease only for
  one poll-and-swap step (`kb-finish.lock` only stops a second finisher from starting).
  `retry-failed` skips sections inside their `kb-finish` cap. **A new edit to a section that is
  already `pending` cancels the pending submission first**: under the lease, `DELETE` every
  `pendingDocumentIds` document, clear the pending fields, then submit the new text — so no
  superseded submission is left current and searchable with no mapping. Lease properties and the
  CAS statement: note §2.4/§6.4.
- **Token fencing** (P3-M1; rules and statements canonical in note §6.4). Every mapping write
  (`pending*` fields, `documentIds`, `state`, release) is **fenced on the holder's lease token**
  (`WHERE ks.leaseToken = $token`); **0 rows = the lease was lost → stop, make no further
  falkor-chat call for that section, report it *pending***. The note's three rules apply verbatim:
  - **L1 — renew before every falkor-chat call**: a fenced renew runs before the `POST`, before a
    swap's old-id deletes, and before the give-up and cancel deletes; 0 rows → stop before calling,
    report pending. The renew matches on the token alone, so an expired lease nobody has taken over
    is still renewable.
  - **L2 — clean up your own `POST`**: a fenced write after a `POST` that returns 0 rows → `DELETE`
    that `POST`'s own ids (404 counts as done), then report pending. If the process dies in between,
    `kb-reconcile` finds the orphan through the `kb-mirror` identity.
  - **L3 — swap order**: renew, then delete the old ids, then the fenced swap `SET` with the
    release folded in — so an expired holder never deletes what a new holder serves.
  **Every falkor-chat HTTP call made under a renewed lease has a timeout well under the TTL —
  10 s against the 60 s TTL** (`COORDGRAPH_KB_HTTP_TIMEOUT_S`, asserted `< TTL/3` at start-up), so a
  renewed lease cannot expire while its call is still in flight.
- **Duplicate claim text** (m8, note §6.4). falkor-chat auto-supersedes a current document whose
  normalized text equals a new one's, so two headings with the same claim text would knock each
  other out of search. `kb-sync` therefore **refuses** to ingest a section whose `claimHash`
  (falkor-chat's normalizer, note §2.4) already belongs to another section: the section is
  recorded failed, the hook exits 2 naming both headings, and `kb-status` lists the duplicate for
  cobb to dedupe in the file. As a backstop the receipt's `supersededDocumentId` is checked: equal
  to the same section's old id (a normalization-only edit) → proceed; anything else → report.
- **Split headings** (cobb consult): split mode is recorded **in the file**. Default `whole`; a
  heading carrying `<!-- kb-split: bullets -->` splits one document per top-level bullet, keyed by
  bullet hash. An existing judgment split whose heading changes is re-ingested whole, its split
  documents deleted, the heading marked `needs-resplit` (`kb-status`: *coarse*, not stale) for
  cobb's next distillation pass. Authoring rule: one claim per heading, or a declared bullets
  heading.
- **Search answers carry sync time and drift (FR-5 for KB):** the `agent-kb-retrieval` skill gains
  one step — after `search_documents`, run `coordgraph kb-status --ids <documentIds>`, which prints
  the KB mirror's last sync time and flags (a) a returned id whose section's file text no longer
  matches its synced hash, or is `pending` ("stale — read `<file>` § `<heading>`"), (b) any
  unmirrored section, (c) duplicate claims.
- **Mapping lives in the mirror graph** (`:KbFile`/`:KbSection`, note §2.4), on the same instance as
  `ws:agent-team`, so one RDB snapshot keeps both consistent after a crash; with split modes in the
  files the mapping is derived state. `claude/cobb/scripts/kb-claim-manifest.json` is **frozen**
  (marked historical in S9); `check_content_loss.py` keeps reading it.
- **`kb-reconcile` (FR-3 for the KB):** every mapped id is checked against `ws:agent-team` —
  missing or hash-mismatched (`textNormalizedHash`) → re-ingest; **`currentVersion = false` →
  reported, not re-ingested** (m8, note §6.4 — re-ingesting would supersede the twin in turn);
  unmapped documents whose hash equals a section's are adopted without re-embedding; `kb-mirror`
  documents mapped by no section are reported as orphans. Runs inside `sync --all` and for every
  `pending` section older than the `kb-finish` cap.
- **Producer identity `kb-mirror`** — today's 333 KB claims are `INGESTED_BY` `cobb`,
  indistinguishable from raw capture (note §6.4). New claims use `kb-mirror`, a roster agent seeded
  by `falkor-chat/scripts/seed_agent_team.sh` (row added in S6, seeded on the shared instance in
  S8a). Existing claims keep `cobb`.

### 3.6 The common backlog-item format (FR-14) — cobb's answer, adopted

Owned and published by `cobb` in the new skill (S5); the parser implements it as a contract.

**Each item is its own `###` heading with a fixed field block under it**, then free prose
(existing `Rationale`/`Proposed change`/`Notes` bullets, falkor-chat's `> **Why it exists.**`
blockquotes stay prose):

```markdown
### K-999 — Example item title (illustrative id, not a real item)

- **Status:** 🔵 proposed — filed 2026-10-15
- **Priority:** medium
- **Depends on:** K-998, cpg:C-310
- **Touches:** file:falkor-chat/server/falkorchat/ingest/*.py; graph:ws:test; fact:falkor-chat/server/falkorchat/repository.py#list_documents (relies); fact:falkor-chat/docs/QUERIES.md#ingest-job-order (changes)
- **Milestone:** M7 — order 2 — approved 2026-10-21
```

- **Item heading** `^### (?P<id>[A-Z]+-\d{3,}) — (?P<title>.+)$` anywhere in a mirrored backlog
  file; nothing else is an item (mcp-monitor's "Cross-references" is not). **Agent plans'
  "Parking lot / ideas" entries are items** (decision log 2026-10-21, Q1b) — after the reformat
  they are `###` items like any other. Duplicate id in one file → parse error. Status prose inside
  falkor-chat headings moves into the `Status` field.
- **Field block** — consecutive `- **Label:** value` lines right after the heading (one blank line
  allowed); continuation lines indented two spaces; unknown labels allowed, not parsed. Canonical
  token first, free text after ` — `.

| Field | Required to | Value |
|---|---|---|
| `Status` | parse | `🔵 proposed` · `🟡 in-progress` · `⚪ deferred` · `⚪ rejected` · `✅ delivered` (keyed on the word; emoji must agree; `✅` only transient — lint flags it). No `blocked`: derived from `Depends on`. |
| `Priority` | plan | `high` · `medium` · `low` [` — free text`] |
| `Depends on` | plan | comma-separated refs or literal `none`; bare `K-070` = same backlog; qualified `<namespace>:<ID>` |
| `Touches` | plan | `; `-separated entries or `none`: `file:<repo-root path or glob>` · `graph:<key>` · `fact:<token> (changes\|relies)` (mode mandatory) |
| `Milestone` | optional | `<M-token>` [` — order <n>`] [` — approved YYYY-MM-DD`] [` — started YYYY-MM-DD`]. Written **only** when the stakeholder approves a proposal (FR-15) or by the reformat's table conversion; every recorded value is therefore fixed (§3.7). `started` is appended to **every** open item of that milestone the first time one of them turns `🟡` (convention, S5/S9 — P3-m1); `lint` errors if items of one milestone disagree on `started`, or a `🟡` item's milestone lacks it. |

**Absent label = not recorded** (item not plannable, fact named); **`none` = recorded empty.**
**Namespaces come from the path:** `<comp>/docs/BACKLOG.md` → `<comp>`; root `docs/BACKLOG.md`
→ `cpg`; `claude/<agent>/kaizen/plan.md` → `<agent>`; a component/agent name clash is a parse
error (none today). **Fact tokens are canonical**: `<path>#<symbol-or-anchor>` or
`mcp:<tool>.<field>`, compared lower-cased; `coordgraph facts [--like X]` lists existing tokens.
**Item summary tables are removed** (cobb's plan has 27 rows vs 19 headings) — an optional human
index may be generated between `<!-- coordgraph:index -->` markers and is never parsed. **A
milestone map is not an item summary table**: falkor-chat's milestone table stays as the
human-readable map (root `AGENTS.md`: a milestone-map row says what the milestone is and when it
landed; decision log 2026-10-22), and membership authority is each item's `Milestone` field. So the
two copies cannot drift silently (P3-m4), the reformat wraps the table in
`<!-- coordgraph:milestone-map -->` … `<!-- /coordgraph:milestone-map -->` markers and `lint`
checks it: every **open** item id a map row mentions must carry that row's milestone token, and
every open item with a `Milestone` must be mentioned in that milestone's row when the map has one
(ids no longer open are ignored — delivered rows are history). The map is otherwise never parsed.

**One-time layout reformat** (accepted, decision log 2026-10-21, Q1 — FR-14's no-backfill rule
bans filling facts, not reformatting). `coordgraph migrate --dry-run` emits a per-file diff that:

- gives every id-less item an id continuing that backlog's series **past the highest id found in
  the backlog and its history file** (so a new id never equals a delivered one — the history
  lookup of §3.7 depends on it); id-less backlogs with no series start at `K-001`;
- converts every top-level bullet of a "Parking lot / ideas" section into an item under that
  section; default status `🔵 proposed`, each listed in the dry-run report as "parking-lot origin
  — owner confirms status". A bullet that records a settled judgment rather than work ("Judged and
  kept, do not re-litigate", "Resolved …") may be set by the applying owner to
  `⚪ rejected — record, not work`; **no bullet's text is dropped**;
- **converts an open milestone table's rows into `Milestone` fields** on their items (decision log
  2026-10-22) — for falkor-chat, `M2.5` onto K-016, K-017 and K-018 — with no `order` and no
  `approved` date (none is recorded); `started` only if a row's items include a `🟡` one (listed in
  the dry-run for the owner to confirm); delivered rows are not converted; the table gets the
  milestone-map markers. The planner then treats
  those items as already placed (§3.7 step 0). This is a recorded fact moved into the item, not a
  backfilled one;
- carries over only what is recorded (status; priority where present; milestone membership as
  above; no status marker → `🔵 proposed`, listed for confirmation), **never** fills
  `Depends on`/`Touches`.

"Touching" an item (the fact duty) means a substantive content edit; the reformat is not one.
Applied one commit per file: the five `BACKLOG.md` files by `teco` (each edit already prompts the
human), the 13 `plan.md` files by `cobb` (S4).

### 3.7 The milestone proposal (FR-13–FR-15)

A **pure, deterministic function** (`coordgraph/propose.py:propose(items, resolver, per_milestone)
-> Proposal`), fed from the graph (note §7.5) — no LLM: FR-14 says recorded facts only, and
determinism makes AC-11 testable.

**Input set** — the open items of one backlog (FR-13). Dependents, effective priority and unblock
counts are computed **within that backlog only** (m7); other backlogs matter only as external
blockers (Step 2).

**Step 0 — fixed items** (decision log 2026-10-21, approved milestones). Every item with a recorded
`Milestone` keeps it and its `order`; it is never re-placed. Fixed milestones are ordered by token
(`M<n>[.<m>]` numerically; `ahead-of-<Mx>` — an inserted milestone approved under that name —
immediately before `Mx` (P3-n1); any other token by first appearance, with a warning). **Unassigned items
are never added to a fixed milestone** — an approved milestone was approved as a set — and fixed
milestones never move. Unassigned items go into new milestones after the last fixed one, named
`proposed-1`, `proposed-2`, … (the stakeholder names them at approval), **except insertions**
(decision log 2026-10-22): a plannable item whose **effective priority is `high`** may be proposed
into a new milestone `ahead-of-<Mx>` in the gap **immediately before a not-yet-started fixed
milestone Mx** — not started means no item of Mx is `🟡` **and** none carries `started` in its
`Milestone` field; the durable `started` marker survives delivery, so a half-delivered milestone
with no `🟡` item left still counts as started (P3-m1) — provided each of its internal deps is delivered, sits in a
fixed milestone before Mx, or is itself inserted into the same gap in an earlier wave. The earliest
eligible gap wins; within it, Steps 4–5 apply unchanged (waves, overlap, capacity N); a high item
that does not fit falls back to normal placement. Every item in an inserted milestone carries the
flag **`⤴ inserted ahead of approved <Mx>`** beside its one criterion, and is approved or rejected
like any other placement. Fixed items still feed Steps 2–3 as dependency targets and dependents.
A fixed item whose placement is now **impossible** is kept where it is and reported under
*approved placements now impossible*, with the cause, as a proposed revision for the stakeholder —
never changed by the tool: it depends on an unassigned item not inserted ahead of it, on an item
in a later fixed milestone or later `order` in its own, on a held / not-plannable / deferred /
externally-open item, or sits on a cycle; or it overlaps another item of its own fixed milestone
with no distinct `order` between them.

**Step 1 — classify** the unassigned items.
- `⚪` deferred/rejected → *deferred — not proposed*; an item depending on one is held.
- `🟡` in progress → *carried — in progress*, not re-placed; it still occupies its touched areas, so
  an overlapping item cannot sit in wave 1.
- `🔵` with `Priority`, `Depends on`, `Touches` all recorded (`none` counts) → **plannable**;
  otherwise **not plannable yet**, missing facts named (FR-14, AC-12).

**Step 2 — resolve dependencies** (refs qualified `<backlog>:<id>`). For each plannable item's dep:
- same backlog, plannable or fixed → internal edge;
- same backlog, `🟡` in progress (M5a) → **satisfied for planning, not before wave 2**: the
  dependent's earliest wave is 2 (wave 1 runs alongside the carried item); reason
  `(a) after <dep> (in progress)`;
- same backlog, not plannable / deferred / held → **held**, "waits on <ref> (<why>)";
- another backlog, still open in the mirror → **held**, "external blocker <ref> in `<path>`"
  (FR-13, AC-13);
- open nowhere → look it up in the owning backlog's history file (`<component>/docs/HISTORY.md`,
  root `docs/HISTORY.md` for `cpg`, `claude/<agent>/kaizen/history.md`; read from disk — history is
  never mirrored, FR-1). **Delivered only on a delivery-shaped match** (M5c): a Markdown heading
  line (`^#{2,4} `) in which the id — optionally followed by `/`-joined sibling ids — is
  immediately followed by `closed`, `delivered`, `fixed`, `shipped` or `done`, or a heading line
  containing both `✅` and the id. Matches today: `## 2026-10-21 — K-065 closed: …`,
  `## … — K-069 delivered: …`, `## … — K-075 fixed: …`; non-matches: `K-067: chunk … (mechanism
  delivered; live proof … outstanding)`, `K-054 revision`, and the body line "The fixes are filed as
  K-077". No match → **not plannable yet**, printed as "unknown dependency <ref> — if it was
  delivered, drop the ref or add a `<ID> delivered` history heading; otherwise fix the id" — a miss
  errs towards a visible, actionable flag, never a false pass.
  **Coverage** (P2-M4): today's history files match few deliveries (review Pass 2 table), so
  correctness rests on a **convention, not more regex shapes**: (i) a delivery's history heading
  carries `<ID> delivered` or `<ID> closed`; several ids are **slash-joined**,
  `K-070/K-071 delivered` — a comma list does not match (P3-n2) (S5 skill; teco's and cobb's close-out steps, S9);
  (ii) `coordgraph lint` reports every `Depends on` ref that is open nowhere and has no
  delivery-shaped heading, so a bad ref is caught when written, and `audit-team.sh` runs it (S9);
  (iii) S1 runs a **coverage probe** (`coordgraph history-coverage`) over every history file — for
  each id mentioned there and open in no backlog, matched/unmatched — and records the rates in
  `coordination-graph/docs/HISTORY.md`. Existing deliveries are not backfilled; an old one is
  fixed when something first depends on it (i.e. when lint flags it).
- A cycle → every item on it **not plannable yet**, the cycle named.
Holding propagates to dependents.

**Step 3 — rank.** `effective_priority(i) = max(priority(i), priority of every plannable or fixed
item in this backlog that transitively depends on i)`; ties by `unblocks(i)` (count of transitive
dependents, same backlog), then own priority, then file order. An item pushed to a later wave keeps
its rank there (m7).

**Step 4 — waves** (list scheduling). Wave *k* takes, in rank order, items whose internal deps all
sit in earlier waves or in fixed milestones (and, for an in-progress dep, *k* ≥ 2); an item is
added unless it **overlaps** an item already in wave *k* (or, for wave 1, a carried `🟡` item), in
which case it moves on. Two items **overlap** when their `touches` share (FR-13(b)):
- `file:` — equal paths, or a directory/glob covering the other path; **glob vs glob is
  conservative** (m7): overlap whenever the literal prefix (up to the first glob metacharacter) of
  one is a prefix of the other's — a possible intersection counts, because FR-13(b) is strict;
- `graph:` — the same key;
- `fact:` — the same lower-cased token, **unless both sides are `(relies)`**.
Overlapping items are always in different waves — sequenced, never parallel.

**Step 5 — milestones.** Waves are concatenated and cut into `proposed-n` milestones of at most
`--per-milestone N` items (**default 6, adjustable per proposal** — decision log 2026-10-21, Q3);
a wave that does not fit spills its lowest-ranked items into the next milestone.

**Step 6 — explain every placement** with exactly one criterion, chosen by this precedence (m7);
waves and "dependency-earliest wave" are counted within the item's own milestone sequence (the
inserted `ahead-of-<Mx>` sequence or the `proposed-n` one), and an inserted item additionally
carries the `⤴ inserted ahead of approved <Mx>` flag, which is not a criterion:
1. `(c) milestone capacity` — step 5 moved it to a later milestone than its wave alone implies;
2. `(b) overlap with <item> on <file|graph|fact> <value>` — its wave is later than its
   dependency-earliest wave;
3. `(a) after <dep>` / `(a) after <dep> (in progress)` — its dependency-earliest wave is > 1 and it
   sits there;
4. `(a) unblocks <items>` — wave 1, effective priority above own priority;
5. `(c) priority <p>` — otherwise.

**Output** — sync header (§3.3), then fixed milestones (as recorded, with *approved placements now
impossible* flagged inline and summarized), then each inserted `ahead-of-<Mx>` milestone in its place before Mx, flagged, then each
`proposed-n` milestone's ordered waves with items and reasons, then *carried*, *held* (blocker + its backlog path), *deferred*, *not plannable
yet* (missing facts). Every item carries its ID and source file.

**Approval (FR-15, AC-14).** `propose` writes nothing. A proposal takes effect only when the
stakeholder explicitly approves it in conversation; the agent then records each approved item's
`Milestone` (name chosen by the stakeholder, `order`, `approved <date>`) through a normal file
edit, which the hook mirrors. Component `BACKLOG.md` edits already require a human click; for agent
`plan.md` files the rule is prompt-level (S5/S9). `teco` may *request* a proposal but must relay it
for approval, never apply it.

### 3.8 Data model, write path, rebuild, durability

Canonical in `claude/docs/plans/coordination-graph-graph.md` — the implementer takes every label,
property, statement, index and query **verbatim from there**. Decisions this plan relies on:

- **Graph key `coordination`** (overridable), not inside `ws:agent-team` (note §1).
- **One `:Source` per covered file of any status**, kind label `:Ledger`/`:Backlog`/`:Review`/
  `:Requirement`; children `:Unit`/`:Item` only while the file is live; an archived ledger is its
  `:Source` with no units **but with its `citedReviews`/`resolvedReviews` lists** (note §2.1–2.2,
  §3.1).
- **Normalized unit status**: raw cell kept as text, `statusToken` (7-value vocabulary or
  `unknown`) indexed and queried; raw cell as `statusText` (note §2.5, §4.2).
- **No relationship crosses a source boundary** (note §2.3).
- **Absent = not recorded, `[]` = recorded none** for FR-14 facts (note §2.2).
- **One atomic statement per file** replaces its subgraph; after writing, the CLI re-hashes the
  file and re-writes if it moved (note §3.2–3.3). No lock for coordination sources; the KB lock
  stays.
- **Key and indexed-value hygiene is an availability requirement**: a UNIQUE-indexed value over
  4,096 bytes or with leading/trailing whitespace segfaults the shared FalkorDB; an indexed
  equality lookup silently misses a whitespace-padded value. `coordgraph/keys.py` provides
  `safe_key()` for UNIQUE values and `norm_value()` for every other indexed value, both stripping and
  asserting that neither end `isspace()` (the crashing set is exactly C `isspace()`; NBSP and other
  Unicode spaces are safe, and Python's `isspace()` is a superset), applied at write **and** at
  lookup (note §4.1).
- **Idempotent read-then-create bootstrap** (note §4.3).
- **Rebuild = unconditional per-file replace + prune + orphan sweep + parity check** — never a drop
  (note §5.1–5.2).
- **First load into the shared instance** follows note §5.5, executed in step S8a.
- **Durability**: RDB-only, kept (Q6); every read's set comparison detects a rollback (note
  §6.1–6.2); the KB mapping rolls back together with `ws:agent-team` and is reconciled by hash
  (note §2.4, §6.4).
- **Reads** cost under a millisecond per operator; footprint ~3–5 MB (note §7–8).

## 4. Step-by-step implementation

Each step leaves the tree buildable and is reviewable alone. `teco` coordinates. S0 runs first; its
result no longer gates the build (decision log Q2) but decides what the README and skill say about
where FR-4 holds.

| Step | Owner | Depends on | Deliverable / done when | Gate |
|---|---|---|---|---|
| **S0** Hook-firing probe | `cobb` authors; **stakeholder applies** the settings edit (decision log Q5; `.claude/` is protected — never routed around via Bash, `docs/reviews/mcp-json-edit-bypass-incident.md`) | S1c (P2-m3) | A temporary logging hook script at `claude/cobb/scripts/hook-probe.sh` (inside cobb's allowlist after S1c), logging to `/tmp/coordgraph-hook-probe.log`, registered on **`PostToolUse` and `PostToolUseFailure`** (matcher `Write\|Edit\|Bash`, `${CLAUDE_PROJECT_DIR}` path) logs event name, `agent_id`, `cwd`, `CLAUDE_PROJECT_DIR`. Legs: main session; foreground and **background** Task subagent; each doing Write, Edit, Bash; a **Bash that writes a file then `exit 1`** (which event fires — M1); an agent in a **worktree** (m4: confirms `CLAUDE_PROJECT_DIR` = main checkout); exit-2 stderr seen by the agent. Record the permission mode. Probe removed; result in `skills/agent-standards/claude-code.md` (Hooks, `Verified:` stamp). | A non-firing context does **not** stop the work (Q2): it is recorded as "FR-5 flagging only" for S9/S11. |
| **S1c** cobb guard widening (M7) | `cobb` drafts and makes the edit; the write escalates once (the guard script is outside its own allowlist) and **the stakeholder approves it** — the intended human review of a guard widening. If cobb runs delegated and the prompt cannot surface, `teco` applies cobb's drafted diff. | — | `claude/cobb/hooks/guard-cobb-topic-writes.sh` allowlist gains, each doubled bare + `*/`-prefixed: `skills/coordination-graph/*`, `skills/agent-kb-retrieval/*`, `claude/scripts/audit-team.sh`, `claude/cobb/scripts/*`, `coordination-graph/kb-files.txt`; header comment says why; entry in `claude/cobb/kaizen/history.md`. Done when a cobb `Write` to each path is allowed without a prompt. | `analyst` (light) |
| **S1b** Ephemeral test fixtures | `devops` | — | (1) A script/pytest fixture starting `falkordb/falkordb:v4.18.11` on a free non-default port, no volume, removed afterwards (note §5.4). (2) **Scratch falkor-chat** (m5): a falkor-chat server process on its own port and workspace against that container, wired to (3) a **fake embed endpoint** — stdlib HTTP server, OpenAI-compatible `/models` + `/embeddings`, deterministic vectors of the server's configured dimension, switchable to *down* / *slow N s*; extraction disabled or answered by the same fake (devops confirms which falkor-chat config does it). Used by every `live` and KB test. | — |
| **S1** Component skeleton + parsers | `coder` | — | `coordination-graph/` with `setup.sh`, `coordgraph` wrapper, `pyproject.toml`, `.gitignore` (`.state/`), `coordgraph/config.py` (§3.1), `coordgraph/parse/{ledger,review,requirements,backlog,kb}.py` (ledger: `statusToken` per note §2.5, `citedReviews`/`resolvedReviews` for every ledger incl. archived; component rule of §3.4a), `coordgraph/discover.py` (§3.3), `coordgraph/keys.py` (`safe_key`, `norm_value`), `coordgraph/history.py` (delivery-shaped match, §3.7), subcommands `lint [--count]` (incl. the unresolvable-`Depends on` check, §3.7), `migrate --dry-run`, `facts`, `history-coverage`. Offline parser tests (§5) green; **coverage probe run over every real history file and its matched/unmatched rates recorded** in `coordination-graph/docs/HISTORY.md` (P2-M4). No graph code yet. | `analyst` |
| **S2** KB prerequisite (narrowed K-058) | `cobb` | S1c | Heading→documentIds mapping recovered for the 5 unkeyed files (§2.1) by title-match + byte-exact `get_document`; the 5 permanently-failed ids reconciled; stale claims re-synced (current §5 procedure) — the teco claim, the missing devops claim, and `falkordb-quirks.md` once its pending edit is committed (n2); `coordination-graph/kb-files.txt` written; `<!-- kb-split: bullets -->` on every heading already split bullet-per-claim. K-058 rewritten to this scope. | `analyst` |
| **S3** Store, sync, rebuild, drift | `coder` | S1, S1b | `coordgraph/store.py` (bootstrap + per-source replace, Cypher verbatim from the note), `sync.py`, `drift.py`; subcommands `bootstrap`, `sync`, `rebuild` (parity check), **`rebuild --dry-run`** (no DB connection; note §5.5 step 1 assertions — keys, cross-load UNIQUE duplicates, no text on indexed properties, parameter size, `statusToken` tally, `(path, sha256)` manifest — P2-m2), `drift`, `query`. Offline tests for `--dry-run` with a crashing key and a cross-file duplicate; live tests incl. AC-2, AC-4, AC-6, AC-7 — on the ephemeral container only. | `graph-dba` (Cypher/atomicity) then `analyst` |
| **S4** Backlog layout reformat | `teco` applies the 5 `BACKLOG.md` files, `cobb` the 13 `plan.md` files (incl. parking lots), from S1's `migrate --dry-run`, one commit per file | S1, S5 | `coordgraph lint` clean over every backlog; `lint --count` per file equals the dry-run's count (parking-lot bullets included); every parking-lot status confirmed by its applier. | stakeholder approves each `BACKLOG.md` edit (existing `ask` rule) |
| **S5** Skill + convention text | `cobb` | S1, S1c | `skills/coordination-graph/SKILL.md`: §3.6 grammar incl. parking-lot items and fixed `Milestone`, the **delivery-heading convention** (`<ID> delivered`/`closed` in the history heading, multi-id form `K-070/K-071 delivered`; or drop the ref from dependents — §3.7, P2-M4), the **milestone `started` convention** (P3-m1), fact-token rules, CLI recipes for every view incl. `query`, "relay the sync header; a drift flag means quote the file", "a hook exit 2 means the edit is not done — say so", "a proposal never takes effect without explicit stakeholder approval". `skills/agent-maintenance/SKILL.md` §1 template switched to the field block. Catalog row in `skills/README.md`. | `analyst` |
| **S6** KB sync | `coder` | S2, S3 | `coordgraph/kbsync.py`: REST writes (stdlib `urllib`), `RO_QUERY` observation on `$COORDGRAPH_KB_GRAPH` (statements verbatim from note §6.4); `kb-sync`, `kb-status`, `kb-reconcile`, `kb-finish`; append-before-delete with budget and `pending`; per-section lease and cancel-on-re-edit (§3.5, P2-M3); duplicate-claim refusal and receipt check; split modes; `needs-resplit`. **Contract test** against scratch falkor-chat asserting the schema it reads (R10). Adds the `kb-mirror` row to `falkor-chat/scripts/seed_agent_team.sh` (reassigned from S2 — M7). No write to the shared instance. | `analyst` |
| **S7a** Views | `coder` | S3, S4 | `coordgraph/views.py`: `inflight [--waiting]`, `unit`, `planned`; sync header on all. AC-5/6/7/9 live tests. | `analyst` |
| **S7b** Proposal | `tdd-engineer` (clear behaviour contract, §3.7) | S3, S4 | `coordgraph/propose.py` pure function + `propose` subcommand; §5 proposal tests incl. AC-11–AC-14, fixed milestones, mutation check. | `analyst` |
| **S8a** First load into the shared instance (M6) | `graph-dba` | S3, S4, S6 | **Note §5.5 steps 1–5, verbatim** (offline key validation incl. KB keys → throwaway-container rehearsal, twice, with parity → freshness re-check → `BGSAVE` baseline → live `bootstrap` + `rebuild` + parity), each gate passing before the next. Plan additions, after §5.5 step 5 and before its step 6: `lint` reports no `unknown` `statusToken` in a non-archived ledger (in step 1); run `seed_agent_team.sh` for `kb-mirror`, then `kb-reconcile` to seed the KB mapping (leftovers reported). Parity verdict and counts in the ledger row (§5.5) and in `coordination-graph/docs/HISTORY.md`. Requires a `rebuild --dry-run` (§5.5 step 1) — built in S3. | `analyst` |
| **S8** Hook | `coder` writes `coordination-graph/hooks/post-tool-sync.sh` and `hook-sync`/`retry-failed`; `cobb` authors the settings block (both events); **stakeholder applies** it (Q5) | S8a | §3.2 behaviour; fast path measured < 50 ms on a no-change call (recorded in the README); every §5 hook test green. **Go-live** (P2-M2): immediately after the stakeholder registers the hook, run `coordgraph sync --all` (re-syncs files edited since the S8a load) and initialise `.state/stamp`; record both in `coordination-graph/docs/HISTORY.md`. | `analyst` |
| **S9** Team rollout | `cobb` (prompts, skills, audit); `teco` applies root `AGENTS.md` (cobb drafts); `tico` the manual | S5, S8 | `teco.md` (resume via `unit`/`inflight`; a ledger edit is done only when the hook reports no failure; request proposals, never apply unapproved; new items with all three facts; a delivery's history heading carries `<ID> delivered`/`closed` — same duty in `agent-maintenance` for agent plans); `tico.md` (stakeholder answers via the skill); `agent-kb-retrieval` gains the `kb-status` step; `agent-maintenance` §5 step 5 replaced (KB edits sync automatically; distiller confirms the hook report, resolves `needs-resplit`, duplicates and drift); manifest marked historical; `claude/scripts/audit-team.sh` gains `coordgraph lint` (incl. unresolvable `Depends on` refs), registry-vs-mirror, and hook-registered (both events) checks; `claude/AGENTS.md` Hook machinery gains the project-settings hook class with an offsetting trim; root `AGENTS.md` (**AC-1**, net-zero words): backlogs are mirrored in the graph and headed there, history stays a file; `claude/docs/manuals/agent-team.md` tracking section (`tico`, per K-057). Contexts where S0 found the hook silent are named in the skill. | `analyst` |
| **S10** Acceptance | `qa-engineer` | S7a, S7b, S8, S9; **falkor-chat:K-070 delivered** (AC-8 leg only, Q7) | Test plan + report (`claude/docs/test-plans/coordination-graph.md`, `claude/docs/test-reports/coordination-graph-report.md`) covering every AC, incl. the KB and acceptance legs of §5. | — |
| **S11** Component docs | `coder` (README/AGENTS.md/CLAUDE.md stub/`docs/HISTORY.md` in `coordination-graph/`), `teco` (root `AGENTS.md` Structure bullet + component table row, **inside R8's net-zero budget** — n5) | rolling with S1–S8 | Run instructions, the hook's contract and kill switch, failure/backoff behaviour, where FR-4 holds (S0), how to rebuild, how to read the sync header. | with each step |

Parallelism: S1 ∥ S1b ∥ S1c; S0 and S2 after S1c; S5 ∥ S3; S7a ∥ S7b ∥ S6 ∥ falkor-chat:K-070 (falkor-chat's own item, its own plan; `teco` schedules it — Q7). S4, S8a and S8 touch shared
state (every backlog; the shared FalkorDB; `.claude/settings.json`) and run alone.

## 5. Test strategy

Run from `coordination-graph/` as working directory. Tests needing FalkorDB run against S1b's
**ephemeral container** and are marked `live` — **never against the shared `falkordb-dev`** (the
graph-dba consult crashed it once, note §9.1). KB tests use S1b's **scratch falkor-chat + fake
embed endpoint**, never the real `ws:agent-team` or LM Studio (m5). Every test sets
`COORDGRAPH_FALKORDB_*`, `COORDGRAPH_STATE_DIR`, `COORDGRAPH_REPO_ROOT`, `COORDGRAPH_KB_*` to its
own fixture values. Fixture Markdown trees are **built in `tmp_path`**, never checked in under a
`docs/plans/`-shaped path (write-guard deny-lists; a careless glob would treat them as sources).

**Unit (offline)**
- *Ledger parser*: every header variant of §2.1 (incl. `Agent` for `Owner`, no `Cost`, multiple
  tables, duplicate unit id → `~n` + warning, a 600+-char cell, the two pre-convention shapes →
  warnings, not crashes); **`statusToken` from real cells** (M2) — every raw variant quoted in
  note §2.5, at least: `accepted`, `**in-flight** — dispatched at …`,
  `**accepted — APPROVE WITH SUGGESTIONS**` → `accepted`, `✅ delivered 2026-07-28 (…)`,
  `✅ done 2026-07-29 — …` → `delivered`, `**paused** — …` and `**relayed**` → `paused`,
  `superseded` → `abandoned`, `blocked (…)`, `teco-verified`, a verdict-shaped cell, `—` and an
  empty cell → `unknown` + warning; plus a **corpus test** over a frozen copy of today's 911 status
  cells (captured once into the test data) asserting no exception and the note §2.5 tally
  (`paused` 5, `unknown` 41, all `unknown` in archived ledgers);
  `paused` with and without a relay date; `citedReviews`/`resolvedReviews` for a live and an
  archived ledger.
- *Review parser*: each verdict spelling of §2.1, last `## Pass N` wins.
- *Requirements* header statuses; *component* rule (`docs/BACKLOG.md` → `cpg`, root
  `docs/plans/x.md` → `root`, `falkor-chat/…` → `falkor-chat`).
- *Backlog parser*: common format incl. "not recorded" vs `none`; `Milestone` with/without order and
  date; duplicate id → error.
- *Migration*: parking-lot bullets become items (count preserved, text preserved); new ids skip ids
  present in the history file; an open milestone-table row becomes `Milestone: M2.5` on each of its
  items (falkor-chat fixture: K-016/K-017/K-018), delivered rows are not converted, the table text
  is left in place inside milestone-map markers; no `Depends on`/`Touches` ever emitted.
- *Milestone lint* (P3-m1, P3-m4): items of one milestone disagreeing on `started` → error; a `🟡`
  item whose milestone lacks `started` → error; a map row naming an open id whose field says
  another milestone, or an open `Milestone: M2.5` item absent from the M2.5 row → error; a delivered
  id in a row → ignored.
- *History convention* (P3-n2): `K-070/K-071 delivered` matches both ids; `K-070, K-071 delivered`
  matches neither.
- *History match* (M5c): delivered on `## … — K-065 closed:`, `K-069 delivered`, `K-075 fixed`,
  `K-065/DEF-6 closed`, a `✅` heading; **not** on `K-067: … (mechanism delivered; …)`,
  `K-054 revision`, or a body line "filed as K-077". `lint` flags a `Depends on` ref open nowhere
  with no delivery heading; `history-coverage` prints matched/unmatched per file.
- *Keys*: `safe_key()`/`norm_value()` with note §4.1's crashing inputs (> 4,096 bytes; leading or
  trailing space, `\t`, `\n`, `\r`, `\v`, `\f`) → stripped or raised, never passed through; NBSP
  and em space as **safe** controls (m2).
- *KB split*: `##` sections, `bullets` mode, duplicate-claim detection.
- *Drift diff*: changed / added / removed rows, unsynced file, gone file.
- *Proposal* (`propose.py`, pure): **AC-11** fixture — B deps A; C/D share a `file:`; G/H a
  `graph:`; I/J a `fact:` (changes/relies); E high, F low, else unconstrained — A's wave < B's,
  each pair in different waves with the overlap named, wave(E) ≤ wave(F), exactly one criterion each.
  **AC-12**: each missing fact in turn → not plannable, fact named. **AC-13**: external dep open →
  held with ref + path; found delivered in history → satisfied; history mention without delivery
  shape → not plannable "unknown dependency". Plus: cycle; holding propagation; effective-priority
  pull-forward; `🟡` carried item blocking an overlapping wave-1 item; **dependency on a `🟡` item
  → wave ≥ 2, reason "(after X (in progress))"** (M5a); **fixed items** (M5b) — never moved, new
  non-high items only in `proposed-n` after the last fixed milestone; **insertion** (decision log
  2026-10-22): a high-priority, dependency-free item lands in `ahead-of-<Mx>` before a not-started
  fixed Mx, flagged `⤴ inserted ahead of approved Mx`, never before a started one (with a `🟡` item, **or** with `started` recorded and its first item
  already delivered and removed, the rest `🔵` — P3-m1), an approved `ahead-of-M7` sorting just
  before `M7` (P3-n1),
  not when one of its deps sits in Mx or later, a `medium` item never inserted, overflow beyond N
  falls back to `proposed-n`; a fixed item depending on an
  unassigned item / a later fixed milestone / an externally-open item → listed as *impossible*
  with cause and still in place, two overlapping fixed items without distinct order → flagged;
  directory-prefix and **glob-vs-glob** overlap; the step-6 precedence (an item moved by both a dep
  and an overlap reports the overlap; a spilled item reports capacity); moved items keep rank;
  capacity spill with `--per-milestone 6` and another N; determinism (same input twice → identical
  output).
- *Mutation check on the proposal* (team practice): drop the overlap check; ignore deps; sort by
  own priority instead of effective; treat fixed items as movable; allow insertion of a
  non-high item or before a started milestone; accept a non-delivery history mention.

**Integration (`live`, ephemeral container)**
- `sync` of each kind → graph rows equal the parse; re-sync of an unchanged file → no write;
  archived flip → units removed, stub kept with its review lists (**AC-7**).
- **AC-2**: `sync --all`; `rebuild --graph <empty key>`; parity (note §5.2) passes for both keys; a
  one-property mutation in the scratch key is reported by exactly that key.
- **AC-6** (M3a): `unit <slug> U3` returns the row through note §7.3's verbatim query; a
  repeated label returns both `U32` and `U32~2`; a `**relayed**` row shows status `paused`.
- **FR-6(c)** (M3b): a review cited by an `accepted` row, and one cited only by an archived ledger,
  are both absent from `inflight`; one cited by an `in-flight` row is present.
- `planned` (P2-m1): a `⚪ rejected — record, not work` item appears only in the trailing
  deferred/rejected group (in full with `--all`), never among planned items.
- **AC-5 / AC-6 / AC-9**: fixture tree with open ledgers in two components (one under root
  `docs/plans/`), a `**paused** — …` unit, an unresolved "needs changes" review, an `Interviewing`
  requirements doc, open items in a component backlog, `docs/BACKLOG.md`, and an agent `plan.md`
  (incl. a former parking-lot item) → each view lists exactly its kinds with component + path +
  sync header; no backlog item in `inflight`; the root-tree ledger labelled `root`.
- **AC-4 / AC-10**: hand-edit a file without the hook → views show sync time and flag exactly the
  changed rows; a deleted file is *gone*.
- `query`: an ad-hoc read prints the sync header and flags a drifted covered file.
- **AC-14**: `propose` → `git status --porcelain` and every mirrored file's hash unchanged; node/edge
  counts and `Source.syncedAt` unchanged.

**Hook (`live`; hook script fed synthetic `PostToolUse` / `PostToolUseFailure` JSON on stdin, run
through `subprocess.run(capture_output=True)` so every timing below is measured to **pipe EOF**, not
just process exit — P2-M1)**
- **AC-3**: edit a ledger row, run the hook, then `unit` → the new value, equal to the file; same
  with a `PostToolUseFailure` payload (M1).
- No change → exit 0, no Python started (process accounting), < 50 ms.
- **Outage, FalkorDB** (B1): stop the container → an edit exits 2 within 5 s naming the file;
  20 subsequent no-edit calls each exit 0, < 50 ms, no foreground Python; `retry-failed` runs
  detached at most once per backoff window (`hook.log` count); the stamp advanced; **a second edit
  of the same failing file exits 2 again** (decision log: every edit warns — M8); restart → the
  next due retry syncs it and clears `failed.json`.
- **Outage, embed endpoint** (B1): fake endpoint *down* → a KB edit exits 2 within the fail-fast
  bound and **no `Document` is created** in the scratch workspace.
- **Slow embed**: fake endpoint *slow 30 s* → the hook returns within `COORDGRAPH_KB_BUDGET_S` + 2 s,
  exit 2 "pending"; old claim still searchable and flagged by `kb-status`; `kb-finish` completes the
  swap afterwards; old id deleted only then.
- **Detached children** (P2-M1): with `kb-finish` (slow embed) and `retry-failed` running, the
  hook's pipes reach EOF within the bounds above.
- **Own faults** (P2-M2): (a) no `.state/stamp` → a mirrored file edited before the first run is
  synced and the stamp is created; (b) venv removed / `hook-sync` forced to raise / forced past
  `timeout 50` → exit 2 naming every file of C, paths in `failed.pending`, merged into
  `failed.json` by the next Python run and retried; (c) `retry-failed` killed before rewriting
  `retry-due` → at most one fork per 60 s across 20 calls; (d) shell appends to `failed.pending`
  racing a Python merge (both under `state.lock`) → no path lost (P3-m2); (e) no stamp, nothing
  edited, embed endpoint and KB server down → exit 0, no warning; FalkorDB down → exit 0, stamp
  still absent, first run retried only after `retry-due` (P3-m3).
- **Lease loss** (P3-M1; the same test note §6.4 specifies): lease TTL set to 1 s (HTTP timeout
  scaled below it), embed delayed past it, a second actor takes the lease → the stale holder's fenced write returns 0, it deletes the document its own `POST`
  returned and makes no further falkor-chat call; a stale swap never deletes old ids (its pre-delete
  renewal returns 0); once settled, exactly one `kb-mirror` document per section and **zero
  unmapped `kb-mirror` documents**.
- **Section concurrency** (P2-M3): two concurrent syncs of one section → the second reports
  *pending* at once (lease held), a sync of another section in the same file proceeds; edit, slow
  embed, edit the same heading again, release the embedder → once settled, exactly one `kb-mirror`
  document per section and no unmapped `kb-mirror` document; a running `kb-finish` never makes an
  edit to another file wait.
- **Stale entries** (P2-m4): a failed file deleted, or reverted to its synced hash → its
  `failed.json` entry is dropped by the next `retry-failed`; an absolute payload `file_path` is
  stored repo-relative.
- Cross-session race (R12): file in `failed.json`, stamp already past its mtime, a `Write` payload
  naming it → exit 2.
- Kill switch: `.state/disabled` → exit 0 with a pending edit; sync header shows `Hook: disabled`.
- Worktree payload (`cwd` elsewhere) → the hook reads only `$CLAUDE_PROJECT_DIR`.

**KB (`live`, scratch falkor-chat)**
- **AC-8**: add a one-chunk `##` heading to a registered scratch KB file via the hook →
  `search_documents` returns it before the hook returns; delete the heading → no longer returned;
  edit with the hook bypassed → `kb-status` flags it. Multi-chunk heading → *pending* path above.
- Append-before-delete: kill the fake embedder mid-sync → old claim still served and flagged, never
  lost.
- m8: a second heading with the same claim text → `kb-sync` refuses it (exit 2, both headings
  named), the first heading's document stays current; `kb-reconcile` reports, and does not
  re-ingest, a mapped id with `currentVersion = false`.
- Contract test (R10): after a `POST /documents`, the RO reads of note §6.4 find `HAS_CHUNK`,
  `Chunk.embedding`, `Document.currentVersion`, `textNormalizedHash`.

**Acceptance (`qa-engineer`, black-box, step S10)** — in a real Claude Code session, ask an agent
"what's in flight?", "what's waiting on me?", "what's planned for falkor-chat?", "status of U1 in
salesperson-language-salience", "propose a milestone for cobb", and check each answer against
AC-5/6/9/11–14 and the sync header; a Task-dispatched subagent editing a ledger proves the hook
fires for subagents; one real AC-8 round-trip on `ws:agent-team` with a throwaway one-chunk
heading, cleaned up afterwards (no backend is stopped). **AC-1** is a doc check: root `AGENTS.md`
states both rules, and `grep -rnE 'history.*(headed|into|moves? to) (for )?the graph'` over
non-archived docs finds no claim that history is headed for the graph (n4).

## 6. Risks & decisions

### Risks

- **R1 — The hook does not fire somewhere** (a subagent kind, a future Claude Code release, a
  failed-tool classification). FR-4 degrades to FR-5 there, by decision (Q2). Mitigation: S0 legs
  on both events; read-time flags; `audit-team.sh` checks both registrations; re-probe after Claude
  Code upgrades (cobb K-001 cadence).
- **R2 — Per-call latency.** The `Bash` leg runs on every `Bash` call. Fast path is a pruned
  `find -newer` over the repo (§3.2 step 3; a fixed-directory variant measured 0.00 s, review
  Appendix B — the pruned walk is re-measured in S8); Python starts only when a mirrored file
  changed, and never for retries. If S8 misses < 50 ms, first restrict the hook's `find` to the
  directories that hold sources today (the read-time check still discovers anything new), then, if
  needed, narrow the `Bash` leg with a command-text filter (`sed`, `>`, `git`, `mv`, `cp`,
  `python`) — recorded fallbacks, not defaults.
- **R3 — FR-6(c) will surface stale verdicts on day one** even after the resolution rule. At S10,
  `teco` lists them; the stakeholder/`analyst` archives each or adds a closing pass — a one-time
  triage under the existing by-kind owner table.
- **R4 — "Picked up" is inferred by slug** (note §7.2(d)); an ordinal-only family
  (`x2-coordination.md` with no `x-coordination.md`) would not match — the parser adds the
  ordinal-stripped slug as a second join key if S10 finds a live case.
- **R5 — KB sync depends on two more processes** (`ws:agent-team` server, embedder), normally up
  only in `teco` sessions (K-055). A KB edit elsewhere fails fast (≤ 5 s) with exit 2 and stays
  flagged until a backoff retry or a later edit succeeds — honest, never a stall.
- **R6 — Append-before-delete briefly serves two versions of one claim**, longer (until
  `kb-finish`) for multi-chunk claims; the old one is flagged. Verify in S6 that no
  `list_pending_document_updates` residue accumulates.
- **R7 — Fact tokens are only as good as their spelling.** Canonical rules + `coordgraph facts`;
  residual accepted.
- **R8 — Always-loaded context budgets.** Root `AGENTS.md` (~2,960 words) and `claude/AGENTS.md`
  (~3,400) are already over their smell; S9 and S11 stay net-zero / offsetting.
- **R9 — A malformed key or indexed value takes down or silently misses on the shared FalkorDB**
  (note §4.1, §9.1 — crashed once on 2026-10-20). Mandatory: `safe_key`/`norm_value` everywhere,
  crashing inputs as fixtures, tests only on ephemeral containers, and S8a's offline pre-flight and
  rehearsal before the first real load. Worth reporting upstream and recording in
  `claude/graph-dba/falkordb-quirks.md` (graph-dba's).
- **R10 — Coupling to falkor-chat's graph schema** (`HAS_CHUNK`, `Chunk.embedding`,
  `Document.currentVersion`, `textNormalizedHash`) through the KB observation reads (M4). A
  falkor-chat schema change breaks `kb-sync`'s readiness check. Mitigation: S6 contract test, run
  also by falkor-chat's maintainer when touching document storage (noted in the component README).
- **R11 — Multi-chunk KB edits will usually report *pending*** (110 of 196 sections exceed one
  chunk; embeds queue behind extraction, §2.2) until falkor-chat:K-070 lands. Honest under FR-4's
  "told" standard, but noisy — resolved by scheduling K-070 alongside (Q7).
- **R12 — Cross-session race.** If another session's hook runs between an agent's `Bash` edit and
  its own hook, that other session receives the warning (stamp already advanced). Closed for
  `Write`/`Edit` by the payload path check (§3.2 step 3); residual for `Bash` edits in a
  sub-second window, where the file is still recorded failed and flagged on every read.
- **Rollback:** the mirror is derived state — delete its graph key, remove the settings hook (or
  set `.state/disabled` at once); files are untouched. The backlog reformat stays — plain Markdown,
  readable without the tool.

### Decisions (resolved — decision log `claude/docs/requirements/coordination-graph.md`, 2026-10-21)

- **Q1 — Reformat:** a one-time *layout* reformat of every backlog is accepted; no planning facts
  are backfilled (§3.6).
- **Q1b — Parking lots are items:** they get ids and the item format in the same reformat (§3.6),
  reversing v1's assumption.
- **Q2 — If S0 shows a silent context:** accept instant sync where the hook fires and FR-5 flagging
  elsewhere; no watcher, no prompt-level sync (§3.2, S0).
- **Q3 — Milestone size:** cap 6 per milestone by default, adjustable per proposal (§3.7 step 5).
- **Q4 — `docs/requirements/cross-initiative-coordination.md`:** stays open; `tico` re-scopes it
  against this feature once this plan is approved.
- **Q5 — `.claude/settings.json`:** the stakeholder applies the S0 and S8 blocks by hand.
- **Q6 — Persistence:** RDB-only kept; the post-crash window is flagged stale and repaired by
  `sync --all`.
- **Failure behaviour:** warn on every edit made while sync is failing; retries backed off and
  never make a command hang (§3.2).
- **Q7 — falkor-chat:K-070 scheduled alongside:** it runs in parallel with S6 and is a
  prerequisite of S10's AC-8 acceptance; it stays falkor-chat's own item, not folded into this plan
  (R11).
- **Approved milestones are fixed:** later proposals place only unassigned items, around approved
  ones, and flag approved placements a new dependency makes impossible (§3.7 step 0).

## Appendix A — Pass 1 disposition

| Finding | Disposition | Where |
|---|---|---|
| B1 | fixed — per-file failure state, stamp always advances, detached backed-off retries, fail-fast health checks, `flock -n`, KB foreground budget + `kb-finish`; outage tests | §3.2, §3.5, §5 Hook |
| M1 | fixed — `PostToolUseFailure` registered; S0 leg | §2.3, §3.2, S0, §5 |
| M2 | fixed plan-side (parser + note §2.5 real-cell fixtures + corpus test, views on `statusToken`); mapping is note §2.5 | §3.4a, S1, §5 |
| M3 | fixed plan-side — parser emits `citedReviews`/`resolvedReviews` for every ledger; AC-6 and FR-6(c) tests; model/queries note §2.2, §7.2(c), §7.3 | §3.4a, §3.8, §5 |
| M4 | fixed — REST writes, RO graph reads for readiness; budget/pending for multi-chunk; R10, R11, Q7 (resolved) | §2.2, §3.5, S6 |
| M5 | fixed — (a) in-progress dep → wave ≥ 2; (b) fixed milestones per decision log; (c) delivery-shaped history match | §3.7, §5 |
| M6 | fixed — S8a runs note §5.5 (graph-dba), plus KB seeding; S6 no longer writes to the shared instance | §4 S8a |
| M7 | fixed — S1c widens cobb's allowlist (stakeholder-approved); seed script reassigned to `coder` (S6) | §2.5, S1c, S2, S6 |
| M8 | fixed, stricter — no throttle; every edit during a failure warns (decision log) | §3.2 |
| m1 | fixed — `root` component for root docs other than `docs/BACKLOG.md` | §3.4a |
| m2 | fixed — `norm_value()` (note §4.1), `isspace()` assertion, NBSP as safe control | §3.8, §5 |
| m3 | fixed — `coordgraph query` | §3.1, §3.4 |
| m4 | fixed — root is `$CLAUDE_PROJECT_DIR`; worktree S0 leg and test | §3.2, S0 |
| m5 | fixed — env-overridable config; scratch falkor-chat + fake embedder | §3.1, S1b, §5 |
| m6 | fixed — kill switch, visible in the sync header | §3.2, §3.3 |
| m7 | fixed — step-6 precedence, conservative glob overlap, same-backlog dependents, rank kept | §3.7 |
| m8 | fixed — `kb-sync` refuses duplicate claim text; reconcile reports `currentVersion = false` (note §6.4) | §3.5 |
| n1 | fixed — KB/acceptance tests under S10 | §5 |
| n2 | fixed — 5 unkeyed files; `falkordb-quirks.md` drift handled in S2 | §2.1, S2 |
| n3 | fixed — `K-999` | §3.6 |
| n4 | fixed — targeted grep | §5 |
| n5 | fixed — note §3.1's `**` glob set adopted verbatim; S11 under R8 | §3.3, S11 |
| P2-M1 | fixed — detached children redirect all three streams (shell and `Popen`); hook tests time pipe EOF | §3.2, §5 Hook |
| P2-M2 | fixed — missing stamp → full hash-compare sync; `hook-sync` failure → `failed.pending` + exit 2 (shell-side `timeout 50`); `retry-due` bumped before fork; S8 go-live `sync --all` + stamp | §3.2, S8, §5 Hook |
| P2-M3 | fixed — per-section lease (graph CAS), no file-wide KB lock, cancel-on-re-edit, `retry-failed` skips sections inside the `kb-finish` cap | §3.5, §3.2, S6, §5 Hook |
| P2-M4 | fixed — delivery-heading convention (S5/S9), `lint` + audit check on unresolvable refs, S1 coverage probe with recorded rates, actionable message | §3.7 step 2, S1, S5, S9 |
| P2-m1 | fixed — `planned` shows `⚪` items in a separate trailing group | §3.1, §5 |
| P2-m2 | fixed — `rebuild --dry-run` in S3's done-condition and tests | S3 |
| P2-m3 | fixed — probe at `claude/cobb/scripts/hook-probe.sh`; S0 depends on S1c | S0, parallelism |
| P2-m4 | fixed — `retry-failed` drops gone/already-synced entries; paths repo-relative | §3.2, §5 Hook |
| P2-n1 | fixed — ≤ 5 s | §3.2, R5, §5 |
| P2-n2 | fixed — scope narrowed to "no falkor-chat server code"; seed-script row named | §1, §2 |
| P3-M1 | fixed — token fencing on every mapping write; note §6.4 rules L1–L3 cited; HTTP timeout ≪ TTL (10 s vs 60 s); 1 s-TTL test matching the note's | §3.5, §5 Hook |
| P3-m1 | fixed — durable `— started <date>` on every open item of a milestone at its first `🟡`, lint-enforced; "not started" checks it | §3.6, §3.7 step 0, §5 |
| P3-m2 | fixed — shell appends `failed.pending` under `flock state.lock` | §3.2 |
| P3-m3 | fixed — no-stamp first run hash-compares before health checks; FalkorDB down → silent, retried via `retry-due` | §3.2 step 3, §5 |
| P3-m4 | fixed — map kept (stakeholder), wrapped in markers, `lint` checks it against the fields | §3.6, §5 |
| P3-n1 | fixed — `ahead-of-<Mx>` sorts immediately before `Mx` | §3.7 step 0 |
| P3-n2 | fixed — slash-joined multi-id form named; comma form tested as non-matching | §3.7 step 2, S5, §5 |
| P3-n3 | fixed — Appendix B cleared; "No open questions" now true | Appendix B |
| 2026-10-22 decisions | insertion ahead of a not-started approved milestone (flagged); milestone-table rows converted to `Milestone` fields, table kept as the map | §3.6, §3.7 steps 0/6, §5 |

## Appendix B — graph-note dependencies (resolved)

None open. Every item the plan needs from `claude/docs/plans/coordination-graph-graph.md` is in
the note: `statusMarker` and open-set semantics, same-backlog dependents and the glob-overlap
deferral (§7.5); `pending*` fields, the per-section lease, its CAS, the fenced statements and the
cancel order (§2.4, §6.4); component rule (§2.2); §9.3 Q1/Q2 answered.

## Ready to implement (after re-review)

Plan: `claude/docs/plans/coordination-graph.md` (v2). A standalone `coordination-graph/` CLI mirrors
ledgers, reviews, requirements and backlogs into FalkorDB key `coordination`; a `PostToolUse` +
`PostToolUseFailure` hook syncs each edit in the same turn, warns on every edit it cannot sync, and
never blocks (fail-fast, detached backed-off retries); every read carries a sync header with drift
flags; the KB mirror syncs append-before-delete with a bounded wait; `propose` is a pure function
that keeps approved milestones fixed (KB mapping writes fenced on a per-section lease; high-priority items may be inserted ahead of a not-started
one, flagged). Sequence: S1/S1b/S1c → S0/S2/S3/S5 → S4 → S6/S7a/S7b → S8a
(graph-dba first load) → S8 hook → S9 rollout → S10 acceptance. falkor-chat:K-070 runs alongside S6 and gates S10's AC-8 leg (Q7). No open questions.
