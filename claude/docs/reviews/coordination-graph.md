# Review — coordination-graph implementation plan and graph note

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — · **Item:** `GAL-F0002`

## Pass 1

**Date:** 2026-10-21 · **Reviewed:** `claude/docs/plans/coordination-graph.md` (architect,
Version 1) and `claude/docs/plans/coordination-graph-graph.md` (graph-dba), checked against
`claude/docs/requirements/coordination-graph.md` (Ready for design). Also read:
`claude/graph-dba/falkordb-quirks.md` (incl. the uncommitted whitespace-crash diff), root
`AGENTS.md`, `claude/AGENTS.md`, `claude/teco/teco.md` (ledger convention), `.claude/settings.json`,
the cobb/coder write guards, the falkor-chat REST/ingest code, and the Claude Code hooks reference.
Nothing has been implemented yet, so this is a static review backed by measurements on the live
repo. **Not covered:** I did not re-run graph-dba's `[live]` FalkorDB probes. I accepted them as
evidence and did not write to any FalkorDB instance.

**Verdict: needs changes.** 1 blocker · 7 major · 8 minor · 5 nits.

The design is sound. File-authoritative mirror, one-statement-per-file replace, string joins
instead of cross-source edges, a set-comparison drift check and a pure proposal function are all
good choices. The blocker is how the post-edit hook behaves when a sync fails, because the hook
runs on every tool call in every session. Most of the majors are spots where the two documents
disagree, or where the plan names a mechanism that has no concrete route in the code it relies on.

CPG: considered, not relevant — the code under review doesn't exist yet. The only loaded CPG
(`cpg_falkorchat`) covers falkor-chat, and I read the few falkor-chat REST/ingest functions the
plan depends on directly.

### Blocker

**B1. One failed sync makes every later tool call, in every session, retry the whole sync, and
the KB retry blocks for minutes.** (plan §3.2 steps 3–5, l.163–175; §3.5 l.275–283; R5 l.599)
On any failure the hook leaves the single global stamp where it is (step 5). From then on, every
`Write|Edit|Bash` call in every session sees the same files as newer than the stamp and runs the
full sync again: Python starts (about 0.1 s measured with `cypher-mcp/.venv`), then `kb-sync`
runs again with its 120 s embed wait per changed heading, under a 180 s hook timeout. The `flock`
on `.state/kb.lock` is a blocking lock as specified, so other sessions queue behind it. The
throttle in step 5 limits only how often the warning is shown, not how often the sync is retried.
R5 already says LM Studio is normally up only in `teco` sessions. So one KB edit while LM Studio
is down blocks every Bash call on the machine for up to 2–3 minutes. Each retry also creates and
deletes a `Document` in `ws:agent-team` and queues another LLM extraction. A ledger that fails
for good (for example a parser bug that hits the UNIQUE constraint, note §3.3) adds the Python
cost to every call for as long as it stays broken.
**Fix:**
- Keep failure state per file: `.state/failed` stores `(path, sha256, lastAttempt, error)`.
- Retry a failed file only when its hash changes or a backoff has passed (for example 5 min).
- Still advance the stamp when every *other* file synced.
- Use `flock -n` for KB: if another process holds the lock, report "KB sync in progress" and
  move on.
- Check KB health before syncing (fail fast if `GET /health` fails or the embed endpoint is
  unreachable), and put one overall time limit on `kb-sync` that stays under the hook timeout.
- In S8, test that "FalkorDB down" and "LM Studio down" each cost at most one slow call per
  backoff window, not one per tool call.

### Major

**M1. The hook is registered only on `PostToolUse`, which fires only when a tool call succeeds.**
(plan §3.2 l.151–156) The hooks reference describes `PostToolUse` as "After a tool call
succeeds" and `PostToolUseFailure` as "After a tool call fails", and exit 2 is documented for
both. A `Bash` call that edits a ledger and then exits non-zero (`sed -i … && grep -q X`, or a
script that writes and then raises) would get no hook. The agent then reports the update done
while the graph is behind, which is exactly the FR-4 break the plan says cannot happen (§3.2
"Why this meets FR-4"). The reference doesn't say whether a non-zero Bash exit counts as a
failure, so this is **verified in the docs, unverified live**.
**Fix:** register the same script under `PostToolUseFailure`. Add one leg to S0: a Bash call
that writes a file and then `exit 1`.

**M2. Ledger `Status` cells are stored raw but queried by exact token, so the paused-questions
view misses real rows.** (note §2.2 `Unit` row l.98, §7.2(b) l.533; plan §3.4a(b))
Measured over all 74 ledgers: **362 distinct** Status cell values, for example
`**in-flight** — dispatched at …`, `**accepted — APPROVE WITH SUGGESTIONS**`,
`✅ delivered 2026-07-28 (…)`, `**relayed**`, `—`. `MATCH (u:Unit {status:'paused'})` only
matches a bare `paused`. A real `**paused** — …` cell drops out of FR-6(b)/AC-5/AC-6 without any
warning. §5's parser tests use clean fixture tokens, so they would pass anyway.
**Fix:**
- In the note, add a `statusToken` property: emphasis and emoji stripped, first word, mapped to
  teco's 7-value vocabulary, and indexed in place of `status`. Keep the raw cell as `statusText`.
- An unmapped value becomes `unknown` plus a source warning.
- Add parser fixtures built from the real variants listed above.

**M3. Two errors in the note's Cypher: the single-unit query names a property that doesn't exist,
and the needs-changes query lacks the plan's resolution rule.**
(a) §7.3 l.572 matches `(u:Unit {unitKey: $unit})`, but the `:Unit` property table (l.92–101) has
no `unitKey`. The value only appears inside `key`. Since the plan says the Cypher is taken
"verbatim" (§3.8 l.449), AC-6 returns 0 rows. **Fix:** either store `unitKey`, or anchor on the
UNIQUE `key` as `safe_key(ledgerPath + '#' + unit)`.
(b) Plan §3.4a(c) (l.252–256) treats a review as resolved if a ledger row citing it is
`accepted`/`abandoned` **or that ledger is archived**. The note's §7.2(c) query (l.541–544) has
no such rule. Worse, its model can't support the "archived" branch at all: an archived ledger
keeps no `:Unit` children (note §3.1), so its `cites` are gone. **Fix:** add a `citedReviews`
list (and `resolvedReviews` for accepted/abandoned rows) on every `:Ledger` `:Source`, archived
or not. It is still a function of the file. Then update §7.2(c) in the note so it matches the
plan.

**M4. The KB embed wait has no REST endpoint to poll.** (plan §3.5 l.277–278, S6 l.496
"stdlib `urllib`")
`GET /documents/{id}` returns only `status`, which turns `ready` after LLM extraction
(`repository.get_document`, `falkor-chat/server/falkorchat/repository.py:1101`). Chunk-level
state is "not part of the public MCP/REST surface" (`services.list_document_chunks` docstring).
So "poll until every new chunk carries an embedding" can't be built over REST. Polling
`status = ready` instead waits for the LLM and breaks the plan's own AC-8 timing argument (§2.2).
**Fix:** say plainly that `kbsync.py` reads `ws:agent-team` directly through `falkordb-py`
(`MATCH (d:Document {documentId:$id})<-…(c:Chunk) RETURN count(c), count(c.embedding)`, with the
exact edge taken from falkor-chat's schema). Record that coupling to falkor-chat's schema
(`Chunk`, `textNormalizedHash`, `currentVersion`) as a risk with an S6 contract test. A
multi-chunk heading also waits behind extraction, because Starlette runs a request's background
tasks one after another, embed then extract, chunk by chunk (`background.py:158`).

**M5. The milestone planner (§3.7) leaves three input cases undefined, and the behaviour contract
for S7b has to cover them.**
(a) An item that depends on a 🟡 in-progress item in the same backlog fits none of Step 2's
bullets (l.397–398). Choose either "held, waits on carried X" or "placed no earlier than wave 2",
and add a test.
(b) Step 1 never reads `Milestone`. After the first approval, the next proposal reshuffles items
that were already approved (`Milestone: M7 — order 2 — approved …`). This is the normal case, not
an edge case. Treat approved-milestone items as fixed, and report any change against them as a
proposed revision.
(c) The history lookup (l.401–404) says "found → delivered" without defining what counts as found.
A plain id match is wrong today: `falkor-chat/docs/HISTORY.md:35` says "filed as K-077" for an
item that is still open, so an item removed from its backlog without delivery would read as
delivered. That is exactly the false pass §3.7 says it guards against. Require a delivery-shaped
match (an entry heading or a `✅`/"delivered" clause containing the id) and add a negative test.

**M6. No step does the first load into the shared instance.** (plan §4 l.489–502)
Every test runs on ephemeral containers (§5). S8 needs the "mirror complete before the hook goes
live" but no step runs `bootstrap` + `rebuild` against `falkordb-dev`. That first load is when R9
is most dangerous: the first time real repo text reaches UNIQUE constraints on the shared
instance, which is how the 2026-10-20 crash happened (note §9.1).
**Fix:** add a step "S8a: first load", owned by `graph-dba`.
- Pre-flight: run the real parser over the whole repo **offline** and assert every key that
  would be written passes `safe_key`'s assertions.
- Rehearse the full `rebuild` and parity check on an ephemeral container loaded with the real
  repo.
- Only then bootstrap and rebuild on the shared instance, and record the parity verdict.

**M7. cobb's write guard doesn't cover the paths cobb is assigned to write.** Glob check against
`claude/cobb/hooks/*.sh` (allowlist: `claude/*/*.md`, `claude/*/kaizen/{plan,history}.md`,
`skills/{agent-maintenance,agent-standards,python-web-quirks}/*`, `skills/README.md`,
`cypher-mcp/README.md`, …). These cobb-owned paths fall outside it and would escalate on every
write:
- S5: `skills/coordination-graph/SKILL.md`
- S9: `skills/agent-kb-retrieval/SKILL.md`, `claude/scripts/audit-team.sh`,
  `claude/cobb/scripts/kb-claim-manifest.json`
- S2: `coordination-graph/kb-files.txt`, `falkor-chat/scripts/seed_agent_team.sh`
The 13 `plan.md` edits in S4 are covered. **Fix:** put an allowlist update (a cobb hook script,
which is itself outside cobb's guard, so `teco` or the stakeholder applies it) ahead of S2/S5.
Alternatively, give `kb-files.txt` and the seed script to `coder`/`teco`.

**M8. The warning throttle can hide a fresh FR-4 failure from the agent that made it.** (plan
§3.2 step 5 l.173–175) "The same failing file set is re-reported at most once per 10 minutes."
If an agent edits a ledger that is already failing, the set is unchanged, so the hook stays
silent, and per the plan's own rule (§3.2) the agent then reports the update as done. **Fix:**
key the throttle on `(path, sha256)` instead of the file set. A file whose content changed since
the last warning always gets exit 2. Only unchanged failures are throttled. This fits naturally
with B1's per-file failure record.

### Minor

- **m1. The `cpg` component label is wrong for most of the root `docs/` tree.** Note §2.2 (l.71)
  sets component `cpg` for root `docs/` files. That is right for `docs/BACKLOG.md`, but root
  `docs/plans/` also holds `small-model-benchmarking-*`, `salesperson-ui*` and `kaizen-team-*`.
  AC-5 requires each item's component, so the stakeholder view would file model-bench and
  salesperson work under `cpg`. Use `—`/`root` for root `docs/{plans,reviews,requirements}`, or
  derive the component from the slug family.
- **m2. Only UNIQUE values are normalised, but other indexed values have the same silent-miss
  problem.** The quirks diff says an *indexed* equality lookup silently misses a value with
  leading/trailing whitespace, constraint or not ("normalise at write *and* at lookup"). `safe_key`
  covers only UNIQUE properties (note §4.1 l.267). `Unit.status`/`statusToken`, `Ledger.slug` and
  `Backlog.scope` are indexed too: apply the same strip to their values and to the lookup
  parameters. Separately, Python's `str.strip()` already strips NBSP and other Unicode spaces, but
  write the assertion over `str.isspace()` at both ends so the quirk's open NBSP question doesn't
  matter.
- **m3. FR-5 is only a prompt rule for ad-hoc reads.** Under §3.4 (l.240–243), an ad-hoc
  `mcp__cypher__query` answer carries a sync header only if the agent remembers to run `drift`.
  Add `coordgraph query '<cypher>'` (RO_QUERY plus the drift header) and point the skill at it
  rather than the raw MCP tool.
- **m4. Behaviour in worktrees is unspecified.** The hooks reference says `${CLAUDE_PROJECT_DIR}`
  stays at the main checkout and `cwd` follows a worktree. A worktree agent's ledger edit is never
  seen, and the hook exits 0. Define the mirror as the main checkout's state, never read `cwd`,
  and add a worktree leg to S0.
- **m5. Test isolation and fault injection.** The hook script and the CLI need env-overridable
  FalkorDB host/port and `.state/` directory. Otherwise the AC-3 hook test (§5 l.546) writes to
  the shared instance. The KB tests run against the real `ws:agent-team` and "kill the embedding
  backend", which takes LM Studio away from every session. Use a scratch falkor-chat process (its
  own port and workspace, on the ephemeral container) with a fake embed endpoint instead.
- **m6. No way to switch the hook off quickly.** Removing the hook needs the stakeholder to edit
  `.claude/settings.json` (Q5). Add an env/file switch (for example `.state/disabled`) that the
  script checks first, and document it in the README for when it misbehaves.
- **m7. Planner details.** Step 6 needs a precedence rule for when both a dependency and an overlap
  set the wave (the "exactly one criterion" assertion depends on it). Glob-vs-glob `file:` overlap
  has no stated rule: make it conservative and treat a possible intersection as an overlap, since
  FR-13(b) is strict. Step 3 counts "plannable items" only, while note §7.5 gives a cross-backlog
  dependents query: pick one. Items moved to a later wave keep their rank there.
- **m8. Same-text claims can silently remove each other from search.** If a re-ingested heading's
  normalised text equals another current document (the same claim under two headings), falkor-chat
  auto-supersedes that other document (`create_document_with_auto_supersede`), so it drops out of
  search. `kb-reconcile` checks only id and hash, so it passes. Have it also check
  `currentVersion = true`.

### Nits

- n1. §5 assigns the KB and acceptance tests to "step S9", but `qa-engineer` owns S10.
- n2. The count of unkeyed KB files disagrees: §2.1 says "4 more", S2 lists 5 (manifest: 8 `files`
  plus 3 `_`-keyed, missing `falkordb-quirks.md` and `review-techniques.md`). Also,
  `falkordb-quirks.md` has an uncommitted edit right now, so it is already a drifted KB file going
  into S2.
- n3. The §3.6 example uses `K-077` with a made-up title, but `falkor-chat:K-077` is a real, open,
  unrelated item. Use an obviously fake id (for example `K-999`).
- n4. The AC-1 check `grep -rn "history"` is too broad to show anything. Grep for the specific
  claim (`history.*(headed|into) the graph`) over active docs.
- n5. Plan §3.3 uses `*/docs/...` globs and note §3.1 uses `**/docs/...`. They match the same 74
  ledgers today, but `discover.py` should use one form, plus the plan's `deprecated/**` and
  `archive/**` exclusions, in both documents. S11's new root `AGENTS.md` Structure bullet also
  falls under R8's net-zero budget.

### Requirements coverage

Every FR and AC maps to a step and a test (Appendix A). The plan **explicitly flags** where FR-4
bends: the crash window (Q6), edits made outside Claude Code, deletions (§3.2), and the S0
contingency (Q2). It also flags the FR-14 one-time mechanical migration as an interpretation for
the stakeholder (Q1). I agree with that reading: the backfill ban is about planning facts, and
the decision log accepted a common item format. It is flagged, not silently reinterpreted.
**Not flagged:** the further ways FR-4 can bend found in M1, M8 and m4, and the KB-infrastructure
dependency that B1 turns into a cross-session stall. Nothing is silently dropped. FR-6(c) is
covered in intent, but its resolution rule can't be implemented as modelled (M3b).

### What's solid

- The file is the source of truth, each file's subgraph is rebuilt in one atomic statement (note
  §3.2–3.3, live-verified including rollback on constraint violation), and the client re-hashes
  after writing to close the stale-parse race. That is simple and correct.
- No relationship crosses a source boundary (note §2.3). Replacing one file can never change
  another file's facts, and this is what makes the parity check meaningful.
- Every covered file gets a `:Source`, so detecting drift after a crash is a three-way set
  comparison with no special cases (note §6.2).
- The rebuild is done by unconditional replace, never a `GRAPH.DELETE` (note §5.1). This keeps the
  KB state and avoids the destructive-ops prompt.
- §3.7 is deterministic and pure, and its tests are specified per AC with a mutation check.
- The plan is honest about its limits (Q2/Q6, R3/R4) and proportionate: one small component, with
  no daemon and no changes to falkor-chat.
- Ownership: `tico` directs no implementer, the stakeholder applies `.claude/settings.json`
  changes (S0/S8), `graph-dba` gates the Cypher in S3, and `analyst` gates S1–S9.

### Open questions (for the caller or stakeholder)

1. B1/M8 change what an agent sees while infrastructure is down. Is "one warning per file version,
   with a backoff retry" acceptable as FR-4's standard, or does the stakeholder want stricter
   behaviour?
2. M5(b): should an approved `Milestone` be fixed in later proposals (my recommendation), or
   should each proposal be free to re-plan and show a diff?

## Appendix A — FR/AC → plan coverage

| Req | Plan location | Test | Note |
|---|---|---|---|
| FR-1 / AC-1 | §1, S9 root `AGENTS.md` | §5 doc grep | grep too broad (n4) |
| FR-2, FR-3 / AC-2 | §3.8, note §5 | §5 AC-2 parity + mutation | ok |
| FR-4 / AC-3 | §3.2, S0, S8 | §5 AC-3 via hook | bends flagged; M1/M8/m4 not flagged; B1 |
| FR-5 / AC-4, AC-10 | §3.3 | §5 AC-4/10 | ad-hoc path prompt-only (m3) |
| FR-6 / AC-5 | §3.4a, note §7.2 | §5 AC-5 | (b) M2; (c) M3b; component label m1 |
| FR-7, FR-12 / AC-9 | §3.4, skill S5 | §5 AC-9, S10 | ok |
| FR-8 / AC-6 | `unit`, note §7.3 | §5 AC-6 | M3a |
| FR-9 / AC-7 | §3.4a(a)(d) | §5 AC-7 | ok |
| FR-10 / AC-8 | §3.5, S2, S6 | §5 KB | M4, m5, m8 |
| FR-11 | §3.6, §3.8 | AC-10 | ok |
| FR-13 / AC-11, AC-13 | §3.7 | §5 proposal | M5, m7 |
| FR-14 / AC-12 | §3.6, S4 migration | §5 AC-12 | migration flagged Q1 — acceptable |
| FR-15 / AC-14 | §3.7 approval | §5 AC-14 | plan.md edits prompt-only (stated) |

## Appendix B — evidence commands

- Status-cell tally: a Python scan of every `*-coordination.md` table's `Status` column found 362
  distinct values (top: `accepted` 294, `delivered` 180, `**delivered — accepted, …` 24,
  `gated` 17, `—` 15, `**accepted — APPROVE WITH SUGGESTIONS**` 11, `**in-flight** — dispatched…` 10).
- Hook fast path: `find <mirrored dirs> -newer <stamp> -print -quit` took 0.00 s ×3. The
  `python -c 'import falkordb'` cold start took 0.09–0.13 s.
- Dynamic property access `n[f] IS NULL` (note §7.5 `missing`): confirmed working with a read-only
  query against `ws:test`.
- REST surface: `falkor-chat/server/falkorchat/api.py:164–286`. `POST /documents` accepts
  `producedBy` (l.172). There is no chunk/embedding endpoint.
- Hooks reference (`code.claude.com/docs/en/hooks`, fetched 2026-10-21): subagent firing, exit-2
  semantics and the `${CLAUDE_PROJECT_DIR}`/worktree behaviour as quoted above.

## Pass 2

**Date:** 2026-10-21 · **Reviewed:** plan Version 2 (`claude/docs/plans/coordination-graph.md`,
dispositions in its Appendix A) and graph note Version 2 (`claude/docs/plans/coordination-graph-graph.md`),
read in full, against the requirements decision log's 2026-10-21 entries (warn on every failing
edit; approved milestones stay fixed; parking-lot entries are items; falkor-chat:K-070 runs
alongside S6). I focused on what the revisions introduced: S1c, the detached `retry-failed`/`kb-finish`
processes, `failed.json`, the `PostToolUseFailure` registration, and plan↔note consistency.

**Verdict: needs changes.** 0 blockers · 4 major · 4 minor · 2 nits. All 22 Pass 1 findings are
fixed, and the plan and note now agree on every schema, query, and step reference I checked. The
four new majors are local edits to §3.2, §3.5 and §3.7 and need no redesign. A short Pass 3 limited
to those sections is enough.

CPG: considered, not relevant — no code exists yet. The falkor-chat seams were already read
directly in Pass 1 and are unchanged.

### New findings

**P2-M1. The detached processes inherit the hook's stdout/stderr, so the hook can't return until
they exit.** (plan §3.2 step 4 l.205 `setsid coordgraph retry-failed &`; step 5.3 l.215 and §3.5
l.344, `kb-finish`.) `setsid … &` without redirection keeps the hook's output pipe open.
- Measured in the scratchpad: a script that runs `setsid sleep 3 &` and exits 0 returned its exit
  status at 0.00 s, but its pipes reached EOF only at 3.00 s. With
  `>/dev/null 2>&1 </dev/null`, both took 0.00 s.
- That Claude Code waits for pipe EOF, as a Node `close` handler does, is **inferred**.
- If it does, `kb-finish` (15-minute cap) holds every KB-editing call until the 60 s hook timeout,
  which brings back the stall B1 removed.

**Fix:** spell out `setsid … >>"$STATE/hook.log" 2>&1 </dev/null &` in the shell, and
`subprocess.Popen(..., start_new_session=True, stdin=DEVNULL, stdout=log, stderr=log)` in
`hook-sync`. Make the §5 hook tests run the script through `subprocess.run(capture_output=True)`
so they time pipe EOF, not just exit.

**P2-M2. When the hook's own machinery fails, the shell reads that as "nothing to sync".**
(§3.2 steps 3, 5, 6.)
(a) **Missing stamp.** `.state/` is gitignored (S1), so it is absent in a fresh clone or after
cleanup, and no step creates the stamp. `find -newer <missing>` prints an error and **no paths**
(checked: exit 1, empty output). C is therefore always empty and nothing ever syncs, with no
warning.
(b) **`hook-sync` dies.** If the venv is missing, Python raises, or the 60 s timeout kills it,
step 6 still advances the stamp "always". But nothing was written to `failed.json`, and the exit
code isn't 2, so the agent isn't told. The file is then neither synced nor queued for retry, and
only the read-time check will notice.
(c) **`retry-failed` crashes.** If it crashes before rewriting `retry-due`, the time stays in the
past and every later tool call forks a new Python process.

**Fix:**
- Missing stamp → treat every discovered file as a candidate. The first run then performs a
  hash-compare `sync --all` and creates the stamp.
- Any non-0 exit from `hook-sync` → the shell itself appends C to `failed.json` and exits 2
  naming those files. The shell needs a no-Python way to append; a line-per-path
  `failed.pending` file merged by the next Python run is enough.
- The shell bumps `retry-due` (for example now + 60 s) *before* forking `retry-failed`.
- S8's done-condition gains "go-live: `sync --all` + stamp initialised". Today nothing re-syncs
  the files edited between the S8a load and the hook going live.
- Add tests for all three cases.

**P2-M3. What happens when two KB operations touch the same section is not defined.** (plan §3.2
step 5.3, §3.5 l.337–347; note §2.4, §6.4 two-phase marker.)
Three actors can act on one section: `kb-sync` from a new edit, `retry-failed`, which re-runs
files recorded *pending* every 30 s or on backoff, and `kb-finish`. Two problems follow.
1. If `kb-sync` re-ingests a section that is already `pending`, it overwrites
   `pendingDocumentIds`. The earlier pending document is still a *current* falkor-chat document,
   and it becomes searchable as soon as its embedding lands. That leaves a stale claim in search
   with no mapping. `kb-reconcile` will only list it as a `kb-mirror` orphan; it does not stop it
   from being served.
2. If `kb-finish` holds `kb.lock` for its 15-minute cap (the plan doesn't say which lock it
   takes), every other KB edit in that window, even in another file, gets "pending" and exit 2.
   With 110 of 196 sections larger than one chunk, that is the normal case until K-070 lands.

**Fix:** make the section the unit of concurrency.
- `kb-finish` takes no long-held file lock. It works through `pending` sections, using an
  atomic compare-and-set on `state`/`pendingSince` in the graph.
- A new edit to a `pending` section first **cancels** the pending submission: delete
  `pendingDocumentIds`, then submit the new text.
- `retry-failed` skips sections that are within their `kb-finish` cap.
- Add a test: edit, slow embed, edit the same heading again, then assert exactly one
  `kb-mirror` document per section once things settle.

**P2-M4. The delivery-shaped history match (M5c) misses most real deliveries, so in steady state
the planner marks dependents of delivered items "not plannable".** (plan §3.7 step 2 l.481–490;
§5 *History match*.)
Delivered items leave their backlog, so any `Depends on: X` still in place after X is delivered
can only resolve through history. I measured how often the rule matches today, counting history
headings that match against distinct ids each file mentions:

| History file | Headings matched | Ids mentioned |
|---|---|---|
| `falkor-chat/docs/HISTORY.md` | 9 by the verb rule + 2 by `✅` | 73 |
| root `docs/HISTORY.md` | 0 by the verb rule, 5 by `✅` | 77 |
| `claude/cobb/kaizen/history.md` | 4 | 63 |
| `claude/teco/kaizen/history.md` | 9 (all `✅`) | 31 |
| `claude/analyst/kaizen/history.md` | 0 | 18 |

The plan's choice to err towards a visible flag is right. The problem is that the flag becomes
the normal outcome, so AC-11/AC-13 hold only on fixtures. The §5 test is a list of example
shapes, which can't establish coverage. **Fix:**
- Require a **coverage probe** in S1 instead: run `history.py` over every history file and, for
  each id mentioned there and open in no backlog, report matched / unmatched. Record the rates.
- Close the remaining gap by convention rather than by more regex shapes. The S5 skill requires a
  delivery's history heading to carry `<ID> delivered|closed`, and `audit-team.sh`/`lint` checks
  it. Alternatively, a delivery must also drop the delivered ref from its dependents'
  `Depends on`.
- `propose` prints "unknown dependency <ref> — add a delivery heading or drop the ref", so the
  flag tells the reader what to do.

### Minor

- **P2-m1. "What's planned" will list parking-lot records that aren't planned work.** Note §2.2
  (l.131) makes `open = true` for `⚪ rejected`, and §7.4 lists every open item. The Q1b
  reformat will turn settled-judgment parking-lot bullets into `⚪ rejected — record, not work`
  items. Show deferred and rejected items in a separate group (or behind a flag) in `planned`.
- **P2-m2. S3 doesn't build the `rebuild --dry-run` that S8a needs.** S8a (l.587) says it is
  "built in S3", but S3's done-condition (l.581) lists only `bootstrap`, `sync`, `rebuild`,
  `drift`, `query`. Add `rebuild --dry-run`, with the note §5.5 step 1 assertions, to S3's
  done-condition and tests.
- **P2-m3. The S0 probe script has no path inside cobb's allowlist.** S0 runs in parallel with
  S1c (l.593), and the plan names no path for the temporary logging hook script. Put it under a
  path S1c allows (for example `claude/cobb/scripts/`) and make S0 depend on S1c, or use the
  session scratchpad.
- **P2-m4. `failed.json` entries are never removed for a deleted or renamed file.** Such entries
  get retried on every backoff indefinitely. `retry-failed` should drop entries whose file is gone
  (§3.3 already reports it as *gone*) or whose hash now matches the graph's stored value. The plan
  should also say whether entries store repo-relative or absolute paths (the payload's `file_path`
  can be absolute).

### Nits

- P2-n1. The fail-fast bound is 1 s + 2 s + 2 s = 5 s in the worst case (§3.2 step 5.1), but §3.2
  l.230 and R5 say "≤ ~3 s".
- P2-n2. Adding the `kb-mirror` row to `falkor-chat/scripts/seed_agent_team.sh` (S6) is a
  falkor-chat change, but §1 puts "any falkor-chat code change" out of scope and §2 says the plan
  "does not modify" falkor-chat. Narrow those statements to "no falkor-chat server code".

### Pass 1 dispositions (rechecked in v2 text)

- B1 — fixed: per-file `failed.json`, stamp always advances, fail-fast checks, `flock -n`, KB budget + `kb-finish` (§3.2 l.188–231). New hazards in P2-M1/M2/M3.
- M1 — fixed: both events registered (§3.2 l.179–182); S0 has the Bash-then-`exit 1` leg.
- M2 — fixed: note §2.5 mapping (`relayed`→`paused`; the 5 real paused rows are all `**relayed**`), `statusToken` indexed (§4.2), §7.2(b) queries it, corpus test in §5.
- M3 — fixed: `Unit.unitKey` stored (note l.112), §7.3 filters on it; `citedReviews`/`resolvedReviews` on every ledger (l.89–99); §7.2(c) implements the plan rule.
- M4 — fixed: REST for writes, read-only graph read for readiness (note §6.4 query); R10 + S6 contract test.
- M5 — fixed: (a) wave ≥ 2; (b) Step 0 fixed milestones; (c) delivery-shaped match. Its coverage → P2-M4.
- M6 — fixed: S8a runs note §5.5 (offline validation, rehearsal ×2, `BGSAVE` baseline, live load + parity).
- M7 — fixed: S1c widens `guard-cobb-topic-writes.sh` (file confirmed); seed script moved to `coder` (S6). S0 path gap → P2-m3.
- M8 — fixed, stricter than proposed: no throttle; every failing edit gets exit 2 (decision log).
- m1 — fixed (`root`, note l.78) · m2 — fixed (`norm_value`, C `isspace()` set) · m3 — fixed (`coordgraph query`) · m4 — fixed (`$CLAUDE_PROJECT_DIR`, S0 worktree leg) · m5 — fixed (config env, S1b scratch falkor-chat + fake embedder) · m6 — fixed (kill switch, shown in the header) · m7 — fixed (precedence, conservative globs, same-backlog dependents) · m8 — fixed (refuse duplicate claim, reconcile reports `current = false`).
- n1–n5 — fixed (S10; 5 files; `K-999`; targeted AC-1 grep; one `**` glob form in plan §3.3 = note §3.1).

### Open questions

1. Decision log: approved milestones fixed, new items placed "around" them. Plan §3.7 Step 0 reads
   that as "only after the last fixed milestone", so an urgent, independent item can never be
   proposed ahead of an approved later milestone. Also, falkor-chat's milestone table is not
   migrated into `Milestone` fields (§3.6), so its current plan is *not* fixed for the planner.
   Does the stakeholder want either point changed?

**Evidence (Pass 2):**
- The detached-child pipe test and the missing-stamp `find` test ran in the session scratchpad.
- The pruned full-repo `find` (1,620 files) took 0.00–0.02 s, so R2's < 50 ms target looks
  reachable.
- History-match counts came from `grep -E` with the plan's verb rule and `✅` rule over every
  `HISTORY.md` and agent `kaizen/history.md`.

## Pass 3

**Date:** 2026-10-22 · **Scope:** only what changed since Pass 2: the 2026-10-22 revision of plan
v2 (Appendix A, P2 rows), note v2's §2.4/§6.4 additions (lease, fencing, cancel path), and the two
2026-10-22 decisions in the requirements log. I checked how S8 go-live, `failed.pending`,
`ahead-of-<Mx>`, the history-heading convention and the lease design interact with each other and
with the parts that didn't change.

**Verdict: needs changes, narrow.** 0 blockers · 1 major · 4 minor · 3 nits. All 10 Pass 2
findings are fixed. The one major is a few lines in note §6.4 and plan §3.5 plus one test; once
it is fixed, the minors can be taken up at the S1/S6/S7b/S8 analyst gates without another full
pass.

CPG: considered, not relevant — still no code.

### New findings

**P3-M1. If the lease is lost after the `POST`, the just-submitted document is left searchable and
unmapped, the same leak P2-M3 set out to close. The plan doesn't mention fencing at all.**
- Note §6.4, *Fencing*: a zero-row fenced write means "lease lost: stop, **delete nothing more in
  falkor-chat**, report the section pending".
- The new document's id only exists once `POST /documents` returns, and only the *next* (fenced)
  statement records it. If that write returns 0 rows, the document is current in `ws:agent-team`,
  becomes searchable once embedded, and no section maps it. `kb-reconcile` will only report it as
  an orphan.
- The swap has the same gap: it deletes the old ids in falkor-chat and only *then* runs the fenced
  `SET`. A holder whose lease expired can therefore delete old documents that the new holder is
  still serving or about to swap.
- The plan says only "takes a short lease … releases it after" (§3.5) and points to an Appendix B
  that still reads "**Open after Pass 2** … belong in note §2.4/§6.4". That is now stale, and so is
  the closing "No open questions". §5 has no lease-expiry test.

**Fix:**
- Renew the lease with one fenced statement (`MATCH … WHERE ks.leaseToken=$token SET
  ks.leaseUntil=timestamp()+$ttlMs RETURN count(*)`) **immediately before every falkor-chat side
  effect**: the `POST`, deleting old ids, and the cancel deletes.
- If a fenced write *after* a `POST` returns 0 rows, `DELETE` the ids that `POST` returned. They
  are the process's own unrecorded documents, so nobody else can reference them.
- Bring the fencing rule and these two rules into plan §3.5, and clear Appendix B.
- Add a §5 test: a 1 s TTL with the embed delayed past it; the stale holder's write returns 0, its
  submitted document is deleted, and in the end there is exactly one `kb-mirror` document per
  section.

### Minor

- **P3-m1. A milestone that is partly delivered looks "not started", so `ahead-of-<Mx>` can be
  inserted in front of it.** (§3.7 Step 0.) "Not yet started" is defined as "no `🟡` item", but
  delivered items *leave* the backlog (forward-only rule). A fixed milestone with half its items
  delivered and none currently in progress therefore qualifies for insertion, which goes against
  the decision log's "an approved one that hasn't started". `order` doesn't help either: converted
  table rows have none (§3.6). **Fix:** record the start durably. For example, at an item's first
  `🟡` transition, the convention appends ` — started <date>` to every remaining item's
  `Milestone` field, and `lint` keeps the milestone consistent. Add a §5 case: the milestone's
  first item delivered and removed, the rest `🔵` → no insertion.
- **P3-m2. A failure path written by the shell can be lost when Python merges `failed.pending`.**
  (§3.2 State, steps 5.4/5.6.) The shell appends with no lock, while Python reads, merges and
  presumably truncates under `state.lock`. An append that lands between that read and the truncate
  is lost, and that path then becomes a silent sync miss. **Fix:** either have the shell append
  under `flock "$STATE/state.lock"` (util-linux `flock`, no Python needed), or have Python
  `rename` `failed.pending` to a private name before reading it, so later appends go to a new file.
- **P3-m3. The first run with no stamp makes an unrelated call warn about files nobody edited.**
  (§3.2 step 3 with step 5.1.) With no stamp, C is every discovered file, KB files included. Step
  5.1 runs the KB health checks *before* any hash comparison, so with LM Studio or the
  `ws:agent-team` server down, a call that edited nothing exits 2 naming every KB file. That breaks
  §3.2's "a call that edits no mirrored file never warns". **Fix:** in the no-stamp case, compare
  hashes first and keep in C only files whose hash differs from the graph; fail-fast then applies
  only to those.
- **P3-m4. The milestone map and the `Milestone` fields are two copies that will drift.** (§3.6
  l.466, decision log 2026-10-22.) Item summary tables are removed precisely because they had
  drifted from the detail sections. A hand-kept milestone map next to authoritative `Milestone`
  fields has the same problem. **Fix:** generate the map between `<!-- coordgraph:index -->`-style
  markers (the mechanism §3.6 already provides), or have `lint` check that every id and token in
  the map agrees with the fields.

### Nits

- P3-n1. An approved milestone the stakeholder leaves named `ahead-of-M7` is a non-`M<n>` token,
  so Step 0 sorts it "by first appearance, with a warning". Parse `ahead-of-<Mx>` as sorting
  immediately before `Mx`.
- P3-n2. The history-heading convention (S5) should name the one accepted multi-id form,
  `K-070/K-071 delivered` (the regex accepts `/`-joined ids). A comma list
  (`K-070, K-071 delivered`) does not match.
- P3-n3. Plan Appendix B "Open after Pass 2" and the "No open questions" line are stale: the note
  has the lease. This is folded into P3-M1's fix.

### Pass 2 dispositions (rechecked)

- P2-M1 — fixed: shell `setsid … >>hook.log 2>&1 </dev/null &` and `Popen(start_new_session, DEVNULL)` (§3.2 State); hook tests time pipe EOF (§5 Hook header).
- P2-M2 — fixed: missing stamp → full sync (step 3); non-0/2 `hook-sync` exit → `failed.pending` + exit 2 under shell `timeout 50` (5.6); `retry-due` bumped before the fork (step 4); S8 go-live `sync --all` + stamp. Side effects in P3-m2/m3.
- P2-M3 — fixed in design: per-section lease CAS (note §6.4, live-verified 16-thread race), no file-wide lock, cancel deletes before clearing. Lease-loss leak → P3-M1.
- P2-M4 — fixed: delivery-heading convention (S5/S9), `lint` + audit on unresolvable refs, S1 `history-coverage` probe with rates recorded, actionable message (§3.7 step 2).
- P2-m1 — fixed: `⚪` items in a trailing group, with a test (§5).
- P2-m2 — fixed: `rebuild --dry-run` in S3's done-condition, with offline tests.
- P2-m3 — fixed: `claude/cobb/scripts/hook-probe.sh` (inside the S1c allowlist); S0 depends on S1c.
- P2-m4 — fixed: `retry-failed` drops gone or already-synced entries; paths stored repo-relative.
- P2-n1 — fixed (≤ 5 s throughout) · P2-n2 — fixed (scope narrowed to "no falkor-chat server code").

### Plan↔note consistency (lease)

- They agree on the 60 s TTL, CAS acquire on a free or expired lease, "held → pending, 30 s
  retry", `kb-finish` taking the lease per poll-and-swap step, and the cancel order (delete pending
  ids, then clear).
- The note's server-clock `timestamp()` is the right choice: client clocks never compare.
- The one gap is the fencing and lease-loss rule, which exists only in the note and is incomplete
  there (P3-M1).

## Pass 4

**Date:** 2026-10-22 · **Scope:** confirming the Pass 3 fixes only, not a fresh review. I read note
§6.4 (rules L1–L3) and §9.2, plan §3.1 config, §3.2, §3.5, §3.6, §3.7 Step 0, S5, §5 and
Appendices A/B. I looked closely at two things: a renewed-but-expired lease racing a new
acquirer, and how the `started` marker interacts with `ahead-of`.

**Verdict: approve with suggestions. The plan is closed for implementation.** 0 blockers · 0 major
· 1 new minor, which can be handled at the S6 gate.

CPG: considered, not relevant — still no code.

### Pass 3 dispositions

- P3-M1 — fixed: L1 renews by token before every falkor-chat call and stops when `renewed = 0`. L2 deletes the POST's own ids when a later fenced write returns 0 rows. L3 orders the swap renew → delete old → fenced swap with the release folded in (note §6.4). The plan cites all three (§3.5), sets HTTP timeout 10 s < TTL/3 with a start-up assertion, and has a 1 s-TTL test (§5 *Lease loss*). The renew is live-verified (note §9.2).
- P3-m1 — fixed: a durable `— started <date>` marker (§3.6), which "not started" now checks (§3.7 Step 0), lint-enforced both ways (§5 *Milestone lint*).
- P3-m2 — fixed: the shell appends under `flock "$STATE/state.lock"` (§3.2 State).
- P3-m3 — fixed: on the first run the plan compares hashes before running fail-fast (§3.2 step 3).
- P3-m4 — fixed: milestone-map markers plus a two-way lint check (§3.6, §5).
- P3-n1..n3 — fixed: `ahead-of-<Mx>` sorts immediately before Mx, `K-070/K-071 delivered` is the multi-id form (S5), and Appendix B reads "None open".

### The two cases you asked about

- **A renewed-but-expired lease racing a new acquirer: correct.** The renew (match on token, set
  `leaseUntil`) and the acquire (MERGE with a `leaseUntil IS NULL OR < timestamp()` guard) are each
  one statement, and FalkorDB serializes writes per graph, so one of them always commits first.
  - If the acquirer commits first, the token is replaced and the renew returns 0, so the old holder
    stops (L1).
  - If the renew commits first, `leaseUntil` moves into the future and the acquire's guard returns
    0, so the acquirer reports pending.
  - In the remaining case, a renew that revives an expired lease nobody took over, nobody acted in
    the gap, so extending it is safe.

  This agrees with note §9.2's live probe ("a renewed lease blocks a competing acquire; returns 0
  after a takeover").
- **`started` versus `ahead-of`: consistent.**
  - "Not started" requires no `🟡` item **and** no `started` marker. The marker is written to every
    open item at the first `🟡` transition, so it survives delivery.
  - Lint rejects a milestone whose items disagree, and a `🟡` item whose milestone lacks the marker.
  - An approved `ahead-of-<Mx>` milestone is just another fixed milestone, and unassigned items are
    never added to fixed milestones, so the rule never puts an item into a started one.
  - One residual: a converted table row only gets `started` if one of its items is `🟡` today. A
    pre-existing milestone that was partly delivered could therefore read "not started". The
    dry-run lists every converted row for the owner to confirm, which is enough. The only live case
    today, falkor-chat `M2.5`, is a whole deferred milestone with nothing delivered.

### New minor (can ride on the S6 gate)

- **P4-m1. A POST that times out has an unknown outcome, and the duplicate check can misread the
  retry.** `COORDGRAPH_KB_HTTP_TIMEOUT_S` stops the *client* from outliving the lease, but a POST
  that times out on the client can still commit on the server (the same problem as the quirks
  entry "a socket timeout abandons the answer, not the write"). That leaves a current, unmapped
  `kb-mirror` document whose ids L2 never received.
  - The next retry with the same text auto-supersedes it, so the search duplicate heals itself.
  - But §3.5's duplicate check ("`supersededDocumentId` equal to the same section's old id →
    proceed; anything else → report") would report that as a duplicate claim.

  **Fix (S6):** treat a POST timeout as "outcome unknown". On the retry, accept a
  `supersededDocumentId` that is an unmapped `kb-mirror` document with the same `claimHash`, as
  the abandoned POST. Add a test: the fake falkor-chat delays its response past the client
  timeout but still commits.

### Riding on implementation gates

P4-m1 belongs to S6's `analyst` gate. Nothing from Passes 1–3 is left open. Implementation can
start at S0/S1/S1b/S1c in the order of plan §4.

## Pass 5 (2026-11-20) — consistency audit against the work-item standard

**Scope.** Both requirements documents, plan v2, `coordination-graph2.md`, graph note v2, `docs/plans/cross-initiative-coordination-ids.md` and the ledger, against `docs/requirements/work-item-numbering.md`, `docs/plans/work-item-numbering.md` §3.6/§3.9, `skills/work-items/docs/SPEC.md` §1/§4/§5/§6/§9, the glossary and root `AGENTS.md`. Static read plus live probes: `wi resolve`/`wi check` against the repo registry, `wilib.rules.parse`, plan2's heading regex run on 26 candidate headings, a scratch-dir `wi check`. Out of scope: the Pass 1–4 design (unchanged; no dispositions owed).
**CPG:** considered, not relevant — documents and a not-yet-built component; no code under review.
**Growth-stops check:** passed. Since Pass 4 the base plan changed only its header (`Extended by:`); plan2 and the ids note are new scope; the FR-4 edit is a folded-in answer with its own log entry (but see P5-m4). **Verdict: needs changes** (1 blocker, 5 major, 6 minor, 2 nit). Nothing is built, so every fix is a document fix.

### Your five suspected gaps

1. **Confirmed (P5-M2).** Plan §2.1 L52 and note §2.2 L82-83 keep only `docStatus`, `owner`, `tracks`. No header regex is specified anywhere, so `Item:` cannot "break" it; it is simply dropped, and S1's header test (plan §5 L713-726) has no extra-field case. `wi`'s own header reader (`rebuild.header_fields`, used by `wi check` on the same line) tolerates the field, so a shared grammar exists to cite.
2. **Confirmed, and worse than stated (P5-B1).**
3. **Confirmed (P5-M5).**
4. **Confirmed (P5-m4).** Run: `wi resolve coordination-graph` and `wi resolve cross-initiative-coordination` both exit 4; `wi check` exits 0 only because both files are in `.work-items/baseline.txt` (lines 5, 18).
5. **Confirmed (P5-m3).**

### Blocker

**P5-B1. Item identity: the key and the dependency join assume a legacy-only world.** Evidence: note §2.2 L131 (`Item.ref = <scope>:<itemId>` UNIQUE), L135 (parser "qualifies a bare `K-065` with the item's own scope"), §2.3 L146 (join on `Item.ref`); plan §3.6 L478 (bare = same backlog); plan2 L18 (refs "legacy or scheme IDs, bare or `<namespace>:<ID>`") and L19-20 (history id matched "as an opaque token"). Verified against SPEC §4/§6: scheme IDs are project-unique and survive `move`/`promote` as aliases; `falkor-chat:K-065` is literally an alias string once minted. Three concrete failures follow, all traced in the text, none built yet:
- (a) a bare `GAL-F0042` in the `cpg` backlog meaning an item in `falkor-chat`'s is qualified `cpg:GAL-F0042` and never matches `falkor-chat:GAL-F0042` — the cross-backlog case (AC-13) reports "unknown dependency" or a wrong external blocker;
- (b) one item written `falkor-chat:K-065` by a dependent and `GAL-F0042` in its own heading (or in the history heading `GAL-F0042 delivered`) is two refs; the dependency dangles and the history match misses;
- (c) after `wi move`/`promote` any stored scheme ref is stale, while cross-initiative FR-4 requires "compared by resolved identity".
**Options (decision: `architect` for the plan, `graph-dba` for the key; no stakeholder input needed).** A — keep stored strings raw (preserves note §2.3 purity: a source's subgraph stays a function of its own file) and resolve **at read time** in `coordgraph` through `wilib` (load the registry once, ~500 items; never one subprocess per ref). A scheme-shaped ref is never scope-qualified; `Item.ref` for a scheme heading is the ID itself, legacy keeps `<scope>:<id>`; both sides go through `uid` before the join, an unminted legacy name (exit 5) compares as written. B — resolve at sync time and store the canonical ID: breaks purity (a `wi mint` changes an item whose file did not change; parity/drift cannot see it). C — mirror `.work-items/registry.jsonl` as a covered `:Source` kind so its age shows in the sync header. **Recommend A plus C's drift flag only.** Read through `wi`/`wilib`, not the raw tracked file: the journal can hold unflushed IDs (SPEC §8). Test with a planted rename: a `Depends on` written with the old ID must still join after `wi move` (red on a raw-string implementation).

### Major

**P5-M1. A `needs` on a task or feature ID has nowhere to land in the graph, and archived units are absent.** Evidence: ids note (a)-(c); numbering plan §3.9 ("units are not in the scheme"); note §2.2 L99 (archived ledger = no `:Unit` children); ids note (b) says status "comes from the ledger mirror (FR-5)". There is no `Unit.itemId`, no unit-cell syntax to say "this unit is `GAL-F0042-T03`", and a feature has no status in the registry (SPEC §4: identity only). So `GAL-F0042-T03` cannot be matched to a unit, "accepted" is undefined for a feature, and a need on an archived ledger's unit (AC-4, "counts as met") cannot be read from the mirror, contradicting the ids note and cross-initiative review N2. **Fix (owner `architect`, future `docs/plans/cross-initiative-coordination.md`; `graph-dba` for the note):** (1) a unit may carry an optional `Item` fact whose value is its task ID, stored as `Unit.itemId`; (2) a feature need is met when every ledger whose header `Item:` is that feature is `archived`, or every unit is `accepted` — **stakeholder decision D1**; (3) keep minimal `:Unit` nodes (`key`, `unitKey`, `statusToken`) for archived ledgers, excluded from the in-flight view by the existing `docStatus` filter (note §2.2 L99 and §7.2(a) change; ~900 small rows against the note's 2.9 MB design scope, graph-dba to size). Test: AC-4 variants D' (need cited by alias after a move) and a need on an archived ledger's task.

**P5-M2. The graph stores no `Item:`/`Aliases:`, so a document cannot be joined to its feature.** Evidence as in gap 1. Consequences: no lookup by feature ID (`unit`, `inflight` take only a slug: plan §3.1 L150), "picked up" stays a slug join (§3.4 R4; ordinal-only ledgers exist today: `docs/plans/team-learnings-distillation3`..`6`, `kaizen-inbox-distillation2`, though none has an open requirements twin, so R4 is not yet live) and P5-M1(2) has no anchor. **Fix (`architect` plan3, `graph-dba` note):** `:Source.itemId` (string, as written) and `aliases` (list) for every kind, written when present; plain RANGE index on `itemId` (non-unique: a family shares one); `unit`/`inflight` accept a feature ID or alias through the resolver of P5-B1; keep the slug join as fallback; header grammar = `wi`'s (cite `wilib` rather than restating); parser test with a header carrying every optional field (`Version`, `Extends`, `Reviews`, `Last updated`, `Item`, `Aliases`).

**P5-M3. The one-time reformat mints new legacy IDs, which the standard forbids and `wi check` rejects.** Evidence: plan §3.6 L503 ("id-less backlogs with no series start at `K-001`"); plan2 L21-22 limits the carve-out to backlogs "already using scheme IDs", so `model-bench`, `opencode`, `mcp-monitor` (all id-less today) still get `K-001…`. Run in a scratch dir with the repo's `project.json`: a `### K-001 — x` in `mcp-monitor/docs/BACKLOG.md` gives `namespace: … defines K- items but home 'mcp-monitor' is not in legacy_namespaces`, exit 1; `wi resolve mcp-monitor:K-001` exit 4. Numbering plan §3.10/L152: new items take scheme IDs. Parking-lot bullets and the agent-plan "ids continuing the series" (L503) are the same class. **Fix (`architect`):** `migrate` allocates through `wi new` for every item that has no ID — agent-plan items `K` (home from the path), component-backlog items `F`, with `--slug` unset — and never emits `K-NNN`/`C-NNN`; the "past the highest id" rule is deleted, not narrowed. **Decision D2 (stakeholder, recommended as stated):** type per backlog as above. Test: `lint`/`wi check` clean after a dry-run over each of the six `BACKLOG.md` files (`decision-models/docs/BACKLOG.md` also exists; the plan counts five).

**P5-M4. The `needs` token grammar in the ids note is internally inconsistent.** (i) Its example `connected-code-graph-coordination:U7` uses the file basename, but the graph's `slug` strips the role suffix (note §2.2 L81) and `unit <ledger-slug> <unit-id>` takes `connected-code-graph`. (ii) (b) says never shape-validate yet "for a scheme token call `wi resolve`": routing needs a shape test, and `a:b` tokens look alike (`falkor-chat:K-065` legacy, `connected-code-graph:U7` unit). Probed: `wi resolve falkor-chat:K-065` exits 5 (unminted), bare `K-065` exits 3; (b) maps only exit 4 to "unresolvable", leaving 3 and 5 undefined. (iii) `<slug>:<unit>` is not unique: unit labels repeat in 8 ledgers (`~n`, note §2.2) and a slug can exist in two components (note §7.3). **Fix (`architect`):** state the slug as the graph's role-stripped slug; route by a single documented shape test (opaque for storage, shape-tested only to pick the resolver); map exit 3/5/4 and a multi-match to "unresolvable, reason named"; and say a multi-match repeats the CLI's list rather than guessing.

**P5-M5. "Initiative" is used for the wrong level.** Glossary: initiative = `I`-level theme (`I01` Defects, `I02` Kaizen, `I03`+). The cross-initiative requirements use it for one coordination (the job a ledger tracks), which the standard calls a feature (numbering plan §3.9: "a feature's coordination ledger"). Occurrences, all in `docs/requirements/cross-initiative-coordination.md`, all the coordination sense: title and slug (L1), L8, L15-16 ("shows initiatives and units"), L19, L30 (FR-1), L61, L70 (AC-1), L72-74 (AC-2); the dated decision-log lines (L93-119, L144) are history and stay. Plan v2, note, `coordination-graph.md` requirements and plan2 never use the word in either sense (grep; the only hits are the path `cross-initiative-coordination.md`); the ids note only in its title. **Fix (`tico`):** keep the slug (collision rule 3), replace the prose with "coordination" — the term `coordination-graph.md` already uses — adding once "a coordination is the job one ledger tracks; it maps to a feature when one is registered"; do the same in the future plan's view labels. **D3 (stakeholder): yes.**

### Minor

- **P5-m1. Plan2's tests cannot fail where it matters.** The opacity test uses a *valid* scheme token, so a shape-validating implementation passes. Add a planted `GAL-F42` / free text expected stored verbatim and shown unresolved. The heading table: run, `GAL-F042`, `GAL-F42`, `GAL-F00042` all match (the regex is `\d{2,}`) while `wilib.rules.parse` rejects each as malformed — the stance "widths are `wi`'s call" is fine, but then `GAL-F4` is a negative only by accident; either tighten to `[IST]\d{2,}`/`[FDK]\d{4,}` or add a positive documenting that wrong widths pass shape and are caught by `lint` via `wilib`. Add `GAL-I01-D0001` (no defect positive today) and `GAL-F0042-FR2a` (negative; verified it does not match) and `GAL-F0042-S01` (positive). `K-0001` is correct as a legacy positive (one-letter prefixes are never keys, numbering plan §3.9). Owner `architect`.
- **P5-m2. Namespace-from-path is a second copy of `.work-items/project.json` `legacy_paths`.** Plan §3.6 L483-485 restates the same three mappings. They agree today (verified), but nothing keeps them so. Cite `legacy_paths` as the source, or add a test deriving the expected scope from it. `architect`.
- **P5-m3. The ledger omits the 2026-11-10 `coordination-graph2.md` extension, and the plan the ids note defers to does not exist.** `grep` of the ledger finds no row or note for it (commit `a834ea23`); `docs/plans/cross-initiative-coordination.md` is absent (`ls`), the ids note is its stand-in (and carries no ledger at all), and addendum row A3 schedules only `coordination-graph3.md`. Add a 2026-11-10 row, and either schedule the cross-initiative plan or state that A3 folds it in. Also the ids note's filename: `-ids` is outside the closed role set, so root `AGENTS.md` collision rule 2 makes it a new topic slug, and SPEC §5's family pattern (`<slug>(\d*)(-<role>)?`) will never give it an `Item:`; rename to the plan slug when that plan is written. `tico`.
- **P5-m4. Registration and citation form.** Neither feature is registered; once they are, cross-citations should read `<feature ID>-FR<n>` (SPEC §4; `wi check` matches the label `FR-14` in text, so existing definitions stay valid). `docs/requirements/cross-initiative-coordination.md` L2 still says `Version: 4`, `Last updated: 2026-11-03` while FR-4 was edited 2026-11-10 (L39, L175): bump `Last updated`. FR-4's examples `GAL-F0042-T03`/`GAL-F0042` are not in the registry; label them illustrative (the base plan did this for `K-999`). Plan2 L8 cites "FR-4(b)", which is the ids note's decision (b), not a requirement. **D4 (stakeholder):** register both now, under the project (no `I03`+ parent), `home` `claude` and `repo`, linked `relates`; then `wi headers sync`. `tico`/`teco`.
- **P5-m5. Two homes for milestone and touched facts.** The registry has `milestone`/`touched`/`home` attributes (SPEC §3 `set`, FR-16/FR-19); plan §3.6 declares the backlog `Milestone` field authoritative and `Touches` carries files/graph keys/facts (not components). Say once, in plan3, that the planner never reads the registry attributes and `Touches` is a different fact from `touched`, so `wi set milestone` on a backlog item is not a way to place it. `architect`.
- **P5-m6. Stale statements.** Plan L120 and L921 still call the cross-initiative requirements `Interviewing` (now Ready for design, v4); plan §2.1/S4 count five `BACKLOG.md` files (six exist). `architect` at next revision.

### Nits

- **P5-n1.** Defects, stories and tasks are representable by the plan2 heading regex (verified: `GAL-I01-D0001`, `…-S01`, `…-T01` match), but `Item` has no type property; add a derived `itemType` (last segment letter, parsed offline) so `planned` can list defects (`I01`) or kaizen (`I02`) without string tests.
- **P5-n2.** Plan2's "`FR`/`AC` path segments never match" is true but untested for `FR2a` (see P5-m1).

### What's solid

Plan2's header pairing (`Extends:`/`Extended by:`, status unchanged), its restriction to the change, and the shape-only heading regex (all documented positives and negatives behave as listed when run); the ids note's finding that the old heading regex misses `GAL-F0042` is correct (verified). The standard's `-coordination`/`-graph`/ordinal filenames are used correctly; header blocks of all reviewed documents conform; Bash-made edits (`wi headers sync`) are covered by the sync hook matcher `Write|Edit|Bash` (plan §3.2).

### Open decisions for the stakeholder (recommended)

- **D1** — a need on a feature ID is met when every ledger with that `Item:` is `archived` or all its units are `accepted` (recommended; `abandoned` does not count).
- **D2** — new IDs for id-less backlog items via `wi new`: `K` in agent plans, `F` in component backlogs; no new `K-NNN`/`C-NNN`.
- **D3** — retire "initiative" for a coordination in prose; keep the slug.
- **D4** — register both features now under the project with `relates`, run `wi headers sync`.
(P5-B1's key policy, P5-M1(3) archived unit nodes and P5-M2 are architect/graph-dba design calls with the recommendation given.)

## Pass 6 (2026-11-20) — re-review of the work-item alignment chain

**Scope.** Pass 5 dispositions against both requirements documents (read in full where touched), `coordination-graph3.md` (plan3), graph note v3 (§2.2/§2.3/§5.5/§7.6/§9.2), the registry diff, `Item:` headers and the ledger addendum; the four decisions D1–D4 and four open points. Run: `wi check` (exit 0), `wi resolve` of both slugs and both IDs (`GAL-F0002`/`GAL-F0003`, exit 0, slugs match), plan3 §3.6 heading pattern over all 22 table rows (all behave as listed), plan3 §3.1 route (c) on sample tokens, `wi new K --home claude` / `F --home model-bench` in a scratch copy (mint `GAL-I02-K…`/`GAL-F…`, valid), a baseline-less `wi check` on a two-file scratch. Not re-run: the note's PROFILE probes (read only; throwaway container).
**CPG:** considered, not relevant — documents and a not-yet-built component; no code under review.
**Growth-stops check:** passed with one residue (P6-m3: an answered question still open in plan3). Requirements edits are new scope (FR-16/AC-15, FR-5/AC-4 feature need) or folded-in answers each with a one-question log entry; the terminology edits are rework in place; plan3 is a new document. **Verdict: needs changes** (0 blocker, 2 major, 5 minor, 1 nit). Nothing is built; every fix is a document fix.

### Pass 5 dispositions

- **P5-B1** fixed — plan3 §3.1 (raw storage, read-time `wilib` resolution, scheme never scope-qualified), §3.3, note §2.3/§4 G4/G5; planted-rename test 6 and cross-backlog test 7 present.
- **P5-M1** fixed — `Unit.itemId`, feature need per D1, minimal archived units (note §2.2: 900 units, 0.48 MB, plans unchanged, synthetic fixture); two wording gaps below (P6-m1, P6-m2).
- **P5-M2** fixed in design (`Source.itemId`/`aliases`, index, `unit <slug|ID>`); see P6-M2 for the importer contradiction it introduced.
- **P5-M3** fixed — §3.5 allocates only via `wi new`; verified `K` (parent defaults to `I02`) and `F --home <component>` mint legal IDs; six backlogs named; test 10.
- **P5-M4** fixed — role-stripped slug, one shape test (a)–(e), exits 3/4/5 mapped, multi-match listed not guessed.
- **P5-M5** fixed — grep of `docs/requirements/cross-initiative-coordination.md` finds "initiative" only in the title/slug, the terminology note and dated log lines (history).
- **P5-m1** fixed — heading table verified by running (all 22 rows as stated; `GAL-F00042` passes shape, `lint` catches it); opacity test 2 uses `GAL-F42`.
- **P5-m2** fixed — §3.3 cites `legacy_paths`; test 11.
- **P5-m3** partly — plan3 §1/§7 now say the cross-initiative plan is still to be written; the ledger still has no 2026-11-10 `coordination-graph2.md` row (grep `graph2|plan2|2026-11-10` of the ledger: no hit); `-ids` filename: see open point 3.
- **P5-m4** partly — registered and `Item:` headers present (`wi check`/`resolve` verified); `Last updated` bumped in both requirements; plan2's "FR-4(b)" corrected by plan3 §3.6 (plan2's own text untouched, by design). **Not fixed:** FR-4's `GAL-F0042-T03`/`GAL-F0042` are still unlabelled and unregistered (P6-m4).
- **P5-m5** fixed — plan3 §3.3 last bullet. **P5-m6** fixed by replacement — table rows name base L120 and §6 Q4 ("Interviewing"), and six backlog files; the stale text stays in the base by design.
- **P5-n1** fixed (`itemType`). **P5-n2** fixed (`GAL-F0042-FR2a` in the table).

### Major

**P6-M1. "Statements replaced" is incomplete, and one row cites the wrong section.** Verified by reading the base plan against plan3 §3.7. Base statements that contradict plan3/G3 and are not listed: (i) §3.4a(c) L346 "an archived ledger keeps no `:Unit` children"; (ii) §3.8 L648-649 "children `:Unit`/`:Item` only while the file is live; an archived ledger is its `:Source` with no units"; (iii) §5 L776 integration test "archived flip → units removed, stub kept (AC-7)" — an implementer following it asserts the opposite of G3 (minimal units kept, hidden by the `docStatus` filter); (iv) §3.7 step 2 L565 "refs qualified `<backlog>:<id>`", against "a scheme reference is never scope-qualified"; (v) §3.6 L521 "the five `BACKLOG.md` files" (S4 is listed, this sentence is not); (vi) base §2.1 L52 ledger header grammar (`Status·Owner·Tracks`) against §3.2's `wilib` grammar. The row "Base §2.2 `Item.ref`/`dependsOn`, `:Unit` of archived ledgers" points at base §2.2 ("Write paths that already exist"); those statements are base §3.6/§3.8 and note §2.2. **Fix (`architect`):** add rows (i)-(vi), correct the citation, and change the AC-7 integration row so it asserts "archived units absent from `inflight`, minimal units present" (plan3 test 12 already says half of it). Per root `REVIEW.md` this is a "no X remains" check: sweep the base and plan2 unfiltered for `no :Unit`, `units removed`, `qualified`, `five`, `Interviewing` and map every hit.

**P6-M2. Plan3 contradicts itself on who imports `wilib`, and the sync path gains an unplanned dependency.** §3.1: `identity.py` is "the only importer of `wilib`" and "the `sync`/hook path never reads the registry … stays offline and fast". §3.2 and §5 S1: every `:Source` header parser uses `wilib.rebuild.header_fields`; §3.6 has `lint` use `wilib.rules`. The hook-time parser therefore imports `wilib` (path from `COORDGRAPH_WI_ROOT`); if that path is wrong the hook's sync fails, losing FR-4 freshness, and §7's degraded mode covers only the registry read, not the import. Also `header_fields(path)` reads the file itself (`rebuild.py:70`) while the parser computes `sha256` from its own read: two reads of one file, which a concurrent edit can split. **Fix:** either (a) state the importer set (`identity.py`, header parsing, `lint`) and add a degraded rule for an unimportable `wilib` on the sync path (fail the identity features, not the sync: parse the header with a stdlib copy of the grammar, or store `itemId` absent plus a warning), or (b) keep `wilib` out of the sync path and parse the header with a short in-repo reader checked against `wilib` by a parity test. Add a test that a hook-time sync still mirrors a file when `COORDGRAPH_WI_ROOT` is bogus (red on the naive implementation). Whichever: have the parser pass its already-read text, not a path. Decision: `architect`.

### Minor

- **P6-m1. AC-4's feature sentence is ambiguous for an archived ledger.** FR-5/AC-4 (as edited) say a feature need is met when every ledger is "archived or has all units `accepted`", then "not met when one ledger has an `abandoned` or unfinished unit". An archived ledger with an `abandoned` unit satisfies both. D1 and plan3 §3.4 mean archived wins. Reword AC-4: "…not met when a ledger that is not archived has an `abandoned` or unfinished unit", and make plan3 test 9's "`abandoned` only → unmet" say "in a non-archived ledger". `tico` / `architect`.
- **P6-m2. Plan3 maps an `unknown` status to `unmet`; FR-5 says `unresolvable`.** FR-5: "a need that cannot be found, or whose status is unknown, is listed as unresolvable and the unit is not ready". Plan3 §3.4 (task ID → "met iff accepted") leaves a found unit with `statusToken = unknown` in `unmet`; the result (not ready) is the same, the listing differs (AC-4 E vs a blocked unit). See open point 1.
- **P6-m3. Answered question still open.** Plan3 §7 Q2 and the end of §3.4 ("see open question Q2") remain though AC-4 was aligned (requirements log 2026-11-20, last entry). Remove Q2 and the pointer; keep Q1.
- **P6-m4. FR-4 examples and citation form.** `GAL-F0042-T03`/`GAL-F0042` are not in the registry and are unlabelled; add "illustrative". Optional: cite `GAL-F0003-FR4`-style instead of `FR-4` in new text (SPEC §4).
- **P6-m5. Ledger addendum is stale.** A1 reads "D1–D4 await stakeholder" (settled); no 2026-11-10 `coordination-graph2.md` row (P5-m3); A3's name omits that plan3 also carries the `needs` contract. Update A1, add a plan2 row (commit `a834ea23`), flip A2–A4 to gated after A5. `tico`.

### Nit

- `.work-items/baseline.txt` lines 5 and 18 still grandfather the two requirements files, which now carry `Item:`. A baseline-less `wi check` on a two-file scratch copy exits 0, so the lines look removable; confirm against the full tree before pruning (not verified repo-wide).

### Open points: recommendations

1. **`unknown` status in an archived ledger (41 today).** Consistent in outcome, not in classification. FR-5 literally sends an unknown status to `unresolvable`; plan3 says `unmet`. Recommend: a need that resolves to exactly one unit whose `statusToken` is `unknown` → `unresolvable`, reason "status unknown (<ledger>, <raw cell>)"; never `met`. Amend plan3 §3.4 only; requirements stand. The feature rule is unaffected (an archived ledger is met whatever its units read). Test: a task need on a unit whose archived cell is `blocked (…)` lists it unresolvable.
2. **Stale-holder retry (note §5.5).** Beyond plan3 (G4 asks only that the error name both paths), untested (`[inferred]`, note §9.2), and it nests a sync of a second file inside a failing sync. The base plan already has the mechanism for the transient case, `failed.pending` plus a backed-off retry (base §3.2): after a `wi move`, the target's sync fails atomically, the holder's edit syncs, the retry succeeds. **Recommend: drop to a plain error naming both paths and the "move in progress" hint**; nothing is lost, and the plan and note then agree. If graph-dba wants it, plan3 must adopt it with a test (planted move, target synced first).
3. **`docs/plans/cross-initiative-coordination-ids.md`.** `Status: active`, executed against (FR-4 log 2026-11-10 cites it; plan3 §3.4 cites it), so by collision rule 5 it stays intact: **do not rename** (a rename breaks those citations and the family rule gives it no basis; `-ids` is outside the closed role set, but a filename is metadata and the rule is for new documents). Its decisions (a)-(c) are replaced by plan3 §3.4 and (d) is applied in FR-4, so it should end `superseded`, but plan3 cannot carry `Supersedes:` (it supersedes part of it, and belongs to `GAL-F0002`; the note is `GAL-F0003`). Minimal handling: now, no edit except one line in plan3 §3.4 (already present: "replaces … (a)–(c)"); when `docs/plans/cross-initiative-coordination.md` (Item `GAL-F0003`) is written it carries `Supersedes:` the note, which flips to `superseded` with `Superseded by:`. Interim use is safe because plan3 §3.4's statement controls. Architect/`tico` call; no stakeholder input.
4. **Rejected "record, not work" bullets and D2.** Excluding them contradicts FR-16 ("a backlog item that has no ID gets a scheme ID"), FR-14's common item shape, and the Q1b entry ("they get ids … chosen over the recommended exclusion"); an id-less item also keeps the unstable `@ord` ref the reformat exists to remove. They stay items with status `⚪ rejected` (trailing group in `planned`, never planned work). The cost is registry inflation (plan3 §7), mitigated by the dry-run listing. **Recommend: no exclusion.** If the stakeholder wants one, it is a partial reversal of Q1b and must be logged as such; the lower-cost alternative inside Q1b is letting the applier move a settled judgment to a prose "Settled judgments" section (not a parking-lot item), which Q1b did not forbid because the text is kept.

### What's solid

Identity design (raw storage, candidate-set joins, planted-rename test) answers P5-B1 and the note's `IN $candidates` plans are consistently stated with live evidence and an explicit "registry is a second input" caveat; the heading table behaves as listed; D1 is implemented consistently across plan3 §3.4, note §7.6(b) and test 9; ID registration is consistent (`wi check`, `wi resolve`, headers). The requirements edits stay small, each with a log entry.

### Needs a stakeholder decision

None blocking. Only if open point 4's alternative is wanted (partial reversal of Q1b); otherwise P6-M1, P6-M2, the three minors and the three recommendations are `architect`/`tico`/`graph-dba` calls. P6-M2's choice (a) vs (b) is `architect`'s.

## Pass 7 (2026-11-20) — scoped confirmation of the Pass 6 fixes

**Scope.** Plan3 (§3.1/3.2/3.4/3.7, G4, §6), graph note v3 (§2.2 `itemId`, §5.5 step 1, §7.6, §9.2), `docs/requirements/cross-initiative-coordination.md` (FR-4, FR-5, AC-4, log), ledger addendum. Rechecked: `wi check` (exit 0); an unfiltered sweep of the base plan, plan2 and the note for `no :Unit`, `units removed`, `qualified`, `five`, `no units`, `without units`, `retries once`, `stale-holder` (every hit is either listed in plan3 §3.7 or is the note's own v3 text); re-read of the changed plan3/note sections. Not re-run: the note's PROFILE probes.
**CPG:** considered, not relevant — documents and a not-yet-built component; no code under review.
**Growth-stops check:** passed. Plan3 states the current design only; it cites findings by ID at most (`P5-*`), has no finding→resolution table, and its only open question is Q1 (the answered Q2 and its pointer are gone). The §3.7 table is a replace-statement list required by plan3's own extension semantics, not a disposition list. The note carries one dated revision line and no history; the retry text is replaced, not annotated. Requirements growth is new scope (FR-4 illustration label, AC-4 archived-wins) plus one question→answer log entry. **Verdict: approve with suggestions** (0 blocker, 0 major, 2 minor).

### Pass 6 dispositions

- **P6-M1** fixed — §3.7 now lists base L346, §3.8 L648-649, §5 L776 (with the replacement assertion, test 13), step-2 L565, L521/§2.1 L55/S4 "five", §2.1 L52 header grammar; the wrong-section row now cites base §3.6 and note §2.2. The sweep above leaves no unmapped hit.
- **P6-M2** fixed — §3.1 names the `wilib` importers (`identity.py`, `lint` via it, read paths only); §3.2 uses a stdlib `parse/header.py` on already-read text with a parity test against `wilib.rebuild.header_fields`/`tokens`; test 12 (bogus `COORDGRAPH_WI_ROOT` still syncs) can go red on an implementation that imports `wilib` in the sync path; the note's `itemId` row says the same.
- **P6-m1** fixed in plan3, note §7.6(b) and AC-4 ("archived wins even with an `abandoned` unit"; log entry added); see P7-m1 for the FR-5 sentence.
- **P6-m2** fixed — plan3 §3.4 and note §7.6(c): `unknown` → `unresolvable`, reason "status unknown (<ledger>, <raw cell>)", archived unit's cell read from the ledger file by `Unit.key` with a fallback; test 13 plants `blocked (…)`.
- **P6-m3** fixed — plan3 §7 keeps only Q1; the "see open question Q2" pointer is removed.
- **P6-m4** fixed — FR-4 labels `GAL-F0042-T03`/`GAL-F0042` illustrative (optional citation form not adopted; no longer owed).
- **P6-m5** partly — A1 updated, the 2026-11-10 `coordination-graph2.md` note added, A5–A8 recorded; see P7-m2.
- **Nit** (baseline lines 5 and 18) not fixed; unchanged nit, `teco`'s to prune after confirming repo-wide.
- **Open point 2** applied: the retry is dropped in note §5.5 and plan3 G4 (plain error, both paths, `a wi move may be in progress`, base `failed.pending` retry named); the note still marks the error wording `[inferred]`, owned by the S3 suite.

### Minor

- **P7-m1. FR-5's feature sentence still reads two ways.** "met when every ledger carrying that `Item:` is archived or all its units are `accepted` (an `abandoned` unit does not count)": the parenthetical can be read as applying to archived ledgers. AC-4 and the log entry settle it (archived wins), so no design risk; reword FR-5 to "(in a ledger that is not archived, an `abandoned` unit does not count)". `tico`.
- **P7-m2. Ledger rows are one step behind.** A6 reads `dispatched` and A7 `queued`; both are delivered (the plan3 and note edits are in the tree) and A8 is this pass. `tico` flips A6/A7 to delivered and A8 to gated once Pass 7 is recorded.

### Both requirements documents may stay `Ready for design`

Yes. Both were re-read in full for their edits in Passes 5–7 (terminology, FR-4/FR-5/AC-4, FR-16/AC-15, decision-log entries) under the growth-stops check, and the latest verdict is not "needs changes". The cross-initiative document's own review file (`docs/reviews/cross-initiative-coordination.md`) still ends at Pass 2; the later edits are covered here, so a one-line pointer from that file to this review would close the loop (optional, `analyst`).

### Needs a stakeholder decision

None.

## Pass 8 (2026-11-20) — scoped re-gate of the S1a spec-revision round

**Scope.** Uncommitted edits against HEAD: base §3.1 (wrapper), §3.4a(d) (slug), S1 row; plan3 §3.1 token forms, §3.4 `Unit label`, G3, §4 S1 row; note v4 §2.2 (`slug`, `warnings`, `unitKey`/`unitLabel`, `statusText`, minimal units), §2.3, §2.5, §3.2 write rules, §5.2/§5.5 step 1, §7.3, §7.6. Checked against `coordination-graph/coordgraph/{discover.py,parse/ledger.py}` at the working tree. **Verdict: approve with suggestions** (0 blocker, 0 major, 5 minor, 2 nit).
**CPG:** considered, not relevant — documents plus a small new Python component; no graph for it.
**Verification.** Ran `discover.discover` + `parse_ledger` over every real ledger (98; 88 archived, 10 not) and `wilib.rebuild.stem_slug`; replayed the frozen `status_cells_2026-10-21.json` through `status_token`; `git ls-tree 9b129b18` for the ledger count. The §7.3 Cypher was read, not run (one `OR` predicate; both columns are in `RETURN`), so the note's live-plan claims are inherited, not rechecked.

### Recomputed figures (two definitions each)

| Claim | Stated | Units / by-unit | Distinct (ledger, key) | Verdict |
|---|---|---|---|---|
| units | 1,296 | 1,296 | 1,263 | holds (repeats collapse 33) |
| spaced keys | 400 | 400 (`' '` = `\s` = label≠key) | 398 | holds |
| no bare sibling | 371 | 371 | 369 | holds |
| labels with >1 key / ledgers / exact / label-only | 40 / 10 / 25 / 15 | 40 / 10 / 25 / 15 (also case-insensitive) | — | holds |
| labels repeating in a ledger | 68 | 68 (12 ledgers) | — | holds |
| no-digit labels; empty; punctuation-ended | 105; 0; 0 | 105; 0; 0 | — | holds |
| tally | 986 / unknown 104 (43 empty + 61 other), 72 ledgers | frozen file 986/104/43/61 exact; 72 ledgers at `9b129b18` | — | holds for the frozen definition |
| tally, current tree | not stated | all 1,296: unknown 108; archived only 1,154: 106; **non-archived 142 units: unknown 2** | — | spread 104 to 108 |
| `slug_of` vs `stem_slug` | 8 files | 8 (6 requirement/review of the three named topics + `connected-code-graph-p1-graph.md`, `salesperson-ui-graph.md`) | — | holds; rule text equals `discover.slug_of` |

### Findings (all minor unless marked)

- **P8-m1. §2.5's "live ledger" claims are already false, and are an as-of narrative.** Note §2.5 says `coordination-graph-coordination.md` "now parses with no `unknown`" and that H1 is "the only `unknown` in a non-archived ledger". Parsed now: two, H1 and row `S1e` of `coordination-graph-coordination.md` (`gated-pending (…)`, ledger line 79). Verified by running the parser. A dated census of mutable files decays within the round that wrote it. Replace the "Applied to the two tokens … as of 2026-11-20" paragraph by the rule alone (`dispatched`/`executed` stay `unknown`, the cell is fixed); the S8a lint is the live count. Consistent with S8a's text ("no `unknown` in a non-archived ledger") as far as it goes, see P8-m2.
- **P8-m2. Nobody owns the lint rule the S8a precondition and §2.5 lean on.** Note §2.5 says "the S8a lint reports every such row"; base S8a (graph-dba) lists it as a plan addition. `coordgraph/lint.py` has no `unknown`-status check (grep: only the `Depends on` message), and graph-dba does not write parser/lint code. Add it to the S1/S3 `coder` row's `lint` scope (or an S1-fix item) and cite it from S8a. Verified by reading `lint.py`.
- **P8-m3. Reachability gap: `U32~2` cannot come out of the single query.** Plan3 §3.4 now says `U32~2` "addresses one by `key`"; the note's §7.3 query filters `unitKey = $u OR unitLabel = $u`, and neither property carries `~n`, so `$u='U32~2'` returns zero rows. No document says the client strips a trailing `~\d+$`, queries the rest and keeps the row whose `key` ends `#U32~2` (the pattern must anchor at the end: a real key contains `~80 tool calls)`). Pre-existing wording, newly restated; state the strip-and-select step in the note §7.3 precedence paragraph. Verified statically.
- **P8-m4. The precedence has two implementers and one prose rule.** Plan3 puts the need resolver in `identity.py` (S1d, "pure over fetched rows"); the note makes "the CLI" (`unit`, S3/S7) apply it. Without one named pure helper (e.g. `identity.pick_unit_rows(rows, u)`) the two will drift. Plan3 §6 test 9 lists "repeated label" and `U32~2` but no red-first case for the precedence itself: a mutant dropping the exact-key preference passes every listed case. Add `S6`/`S6 gate` (exact wins; `S6 gate` still reachable by its full key), `U1` vs `U1 K-091 …` (falls through), a label-only ambiguity (`Pass`), and a local label. Name the helper in both documents.
- **P8-m5. Plan3's local-token routing contradicts the note.** Plan3 §3.4 resolves `<slug>:<unit>` **/ local** through `Ledger.slug`; note §7.6 anchors a local label on the referring ledger's `Source.path`. A local token must not go through the slug (two ledgers sharing a slug in two components would turn every local `U3` ambiguous). Fix plan3's bullet; give the path-anchored form of the §7.3 query once (the note shows only the slug-anchored one).
- **P8-n1. The label rule's edge cases are stated as invariants and are false.** Note §2.2 ("never empty for a non-empty `unitKey`; equals `unitKey` when that has no space") and the §5.5 step-1 invariants repeat them; plan3 states only the rule. `unit_label_of('(x) y')` is `''`; `'U3:'`, `'U1(a)'`, `'U1;'` give `U3`, `U1`, `U1` (no space, label≠key). Zero hits in the corpus, so today the assertions pass; the first such cell reddens a correct parser and a rebuild. Drop "equals when no space", and either reject an empty label at parse (warning + fall back to the whole key) or weaken the invariant to "non-empty unless the key starts with `(`". Verified by running the function.
- **P8-n2. "Live" now means two things.** Plan3 and the note write "1,296 live units", "98 live ledgers", while elsewhere "live ledger" means non-archived (`Unit.needs` "live ledgers only"). 88 of the 98 are archived. Say "all 98 ledgers (88 archived)". Also note §2.2 `key` row still says "8 ledgers repeat one [live]"; parsed: 4 ledgers repeat an identical `unitKey`, 12 repeat a label. And §7.2(c) still says "~74 ledgers" (98 now; sizing conclusion unaffected).

### Cross-owner consistency (plan3 §3.4 vs note §2.2/§2.3/§7.3/§7.6)

Agree: the label rule's text (up to first whitespace or `(`, trim `:,;`), case-sensitivity, exact-key precedence decided client-side from one query's rows, label-only ambiguity stays ambiguous, the archived minimal set carries `unitLabel` (G3 and note §2.2 both), `~n` not a label feature, "plan3 controls". Cross-references checked and existing: base §3.4a(d), plan3 §3.4 `Unit label`, note §2.2, §2.3, §3.2, §5.2, §5.5, §7.3, §7.6(d), §7.2(d). The §7.3 query returns `unitKey` and `unitLabel` with every row, so a single query suffices for the precedence; the only missing link is P8-m3/m4. Examples in the text exist in the corpus (`S6`/`S6 gate`, `Pass` ×11 in `salesperson-ui-coordination.md`; `U10`/`U10 (a695…)`; `U10b Note:`, `A7 Note:`, `U2.3 between…`, `C-graph-dba (data half)`).

### Contradictions left unedited

- Base §5 (parser tests) still says "today's 911 status cells … `unknown` 41, all in archived ledgers", which the note now retires ("74 / 911 / 41 … do not reproduce"). Point it at the note's §2.5 tally (986 / 104) or drop the figures; the corpus test then asserts the frozen file's tally. **Minor, P8-m6.** Verified by grep.
- The `slug`, `unitKey`, wrapper and `§7.3` mentions elsewhere agree. Bare `coordgraph <sub>` in the base §3.2 hook steps and note §4.3/§5 is covered by the base abbreviation sentence, but that sentence binds a skill that does not exist yet and S5's row does not carry "define the abbreviation once"; add it to S5. (Folded into P8-m6.) `coordgraph.sh` exists at HEAD.
- §2.5's "`dispatched`/`executed` stay `unknown`" agrees with S8a's precondition text; it is the S8a lint ownership (P8-m2) that is open.

### Growth-stops check

Word counts HEAD to tree: base 12,997 to 13,205 (+208); plan3 3,409 to 3,828 (+419); note 12,905 to 14,136 (+1,231). New scope: `unitLabel` rule and precedence, wrapper, per-kind slug, store write rules, §7.3 plan. Rework / restated, to cut: (a) the note's `unitLabel` row restates the rule and plan3's figures (400/1,296/371/40/10) while plan3 says "the rule is here, once": keep the property and a pointer; (b) note §2.5 "Applied to the two tokens … at the S1a gate", "re-stamped", "the earlier 74 / 911 / 41 figures … are retired", §7.3 "Earlier probe … v2": changelog prose, belongs in `HISTORY.md`/the ledger, not the note; (c) the `slug` row's "Needed: …" rationale and the dangling closing sentence ("The join key for 'picked up'"). Base §3.4a(d) and plan3 §3.4 carry no changelog; plan3's 419 words are mostly one subsection of new scope. Growth-stops: **passed for the base plan and plan3; note has the (b)/(a) items above (minor)**.

### Open questions

None blocking. The architect/graph-dba decision on P8-n1 (reject empty label vs weaken invariant) is theirs; either is sound.
