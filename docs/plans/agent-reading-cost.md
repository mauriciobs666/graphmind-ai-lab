# Agent reading cost — implementation plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — (`docs/requirements/agent-reading-cost.md`) · **Version:** 1.4

2026-11-01 — v1.4: review Pass 5 (`docs/reviews/agent-reading-cost.md`) folded in.

## 0. Goal & scope

Agents get the part of a team document they need, and go from a plan step to the code it changed
and back, without reading either whole. Every claim drawn from a lookup cites its source. Tokens
per run are counted one way. A before figure is taken on the next 10–15 connected-code-graph
(C-1001) units, and quality and the reduction target are then checked against it. The
requirements are fixed in `docs/requirements/agent-reading-cost.md`. This plan is the HOW.

**In scope:** FR-1–FR-8, FR-13, FR-14 and AC-1–AC-7, AC-11–AC-14 and AC-16.

**Out of scope** (on top of the requirements' own list):

- changing C-1001's build. C-1001 is a read-only source here: git history, commit subjects, step
  tables, and later its graphs;
- a new MCP server;
- any change to how `ws:agent-team`'s raw capture or distilled claims are written or searched;
- bringing existing plan families to one current design;
- symbol lookups through the connected code graph (filed as a backlog item, §6).

## 1. Context & findings

CPG: considered, not relevant — the only change to an existing component's code is one
config-gated branch at `IngestionPipeline`'s single production construction site
(`falkor-chat/server/falkorchat/app.py:619`, `_build_default_app`), read directly.
`cpg_falkorchat` is stale, and everything else is a new component.

- **The falkor-chat document store** is used as `falkor-chat/docs/SERVER.md` §1.4 (REST), §1.12
  (out-of-band policy) and §2.2 (MCP) specify it. Three properties shape this design:
  - search hits carry no title;
  - `list_documents` has no cursor;
  - every ingested chunk is both embedded and sent to LLM extraction whenever
    `FALKORCHAT_ENABLE_AGENT` is on, and no flag separates the two.
- **`ws:agent-team`** has 14 `failed` documents (K-102, `falkor-chat/docs/BACKLOG.md`). Of these:
  - 11 have an unembedded chunk, so the embed itself failed;
  - 3 are fully embedded, so a step after the embed failed them.

  `:8200`'s extraction shares the one-model GPU with every embed (K-070).
- **Corpus.** The non-archived requirements and plans without ledgers are 155 files, about 431k
  words (about 3,000 chunks). The live specs come on top.
- **Plan steps and code are linked by convention already.**
  - C-1001 commit subjects name the step (`feat(cpg): C-1001 P0 S09 -- …`, `6f4e34f9`; likewise
    S01 `902b9498`, S02 `30bb4c5b`, S08 `c3ae9b1e`, S10 `6589e129`).
  - Step tables put the id in the first cell and the unit alias in the second
    (`docs/plans/connected-code-graph-p0-2.md` §3), with a "Deliverable (files)" column.
  - Code cites plans and steps in docstrings (`cpg/ccg/ids.py:3-4`, which cites the superseded
    `-p0.md`).
  - Some step rows cite FR/AC ids directly. Others cite only same-document `§` refs (S01 →
    `§2 D7`).
- **U20–U32.**
  - Code units: U20/S01, U21/S02, U27/S09, U28/S08, U30/S10. The rest are document-only.
  - Gate findings are recorded per step and per pass in
    `docs/reviews/connected-code-graph-p0-impl.md`.
  - The ledger has no gate id for U26, U28 or U30.
  - One gate agent served several units. `a72f2b8992f574958` gated S01 Pass 1, then S05 Passes
    1–3 as SendMessage legs.
  - The original briefs name the main tree's absolute root.
  - Docs-only deliverables were still uncommitted when gated, and some files were written through
    Bash.
- **Transcripts.**
  - How gates capture files, and the spawn-time harness entry, are covered by
    `docs/reviews/agent-reading-cost.md` Pass 4 (N4-1, N4-2).
  - A subagent run lives in `~/.claude/projects/<slug>/<session>/subagents/agent-<id>.jsonl`, with
    a `.meta.json` beside it (`agentType`, `toolUseId`, `spawnDepth`, and `parentAgentId` on
    nested runs).
  - A SendMessage resume appends a new string-content user entry to the same file.
  - The last `usage` per `message.id` is final (`docs/plans/typed-decision-models4.md` §2, §3.7).
  - U24 (`a24f61274678bd8b3`) sums to 14,153,036 tokens, 13.2M of them cache reads.
  - The oldest transcript on disk is 18 days old. No `cleanupPeriodDays` is set.
  - A headless session started elsewhere lands under another project slug.
- **Agent deployment.** `~/.claude/agents/<name>` are symlinks into this working tree.
  Project-level `.claude/agents/` are found by walking up from the working directory
  (`typed-decision-models4.md` §2). The host has `python3` 3.12, `git` and the `redis-cli` binary,
  but no Python `redis` module.
- **Collision rule 5** is stated only in root `AGENTS.md`. Prompts, manuals, kaizen files and code
  docstrings rely on it.
- **The docs-only commit-once rule** lives in five files:
  - `claude/AGENTS.md`: the "docs-only coordination chain commits once" paragraph;
  - `claude/teco/teco.md`: the "delegate's edits can vanish" bullet's batching exception,
    "Commit what you verified", and "The grant";
  - `claude/teco/coordination-techniques.md`: "A delegate's edits can vanish mid-run";
  - `claude/tico/tico.md`: the consult rule, "Coordinating a docs-only chain", and "Commit at
    document boundaries";
  - `claude/docs/manuals/agent-team.md`: the two "docs-only chain" answers.

  Its source, `docs/requirements/commit-granularity.md`, is `archived`.

## 2. Design & rationale

### 2.1 Shape: one zero-dependency CLI over files and git, with the falkor-chat index as a finder

A new component, **`agent-reading/`** (Python 3.12, stdlib only, pytest as the dev dependency),
ships one CLI, `agent-reading/bin/ar`:

| Command | Does | Satisfies |
|---|---|---|
| `ar outline <path>` | Lists headings, item ids, line ranges and word counts | FR-1 (navigation) |
| `ar get '<address>'` | Prints one section, row or item, under its citation line | FR-1, FR-3, FR-6, FR-7 |
| `ar find <regex> [paths]` | Section addresses containing an exact match | FR-1 |
| `ar search '<question>'` | Semantic top-K section addresses (§2.3) | FR-1, AC-12 |
| `ar trace step '<plan> #<id>'` / `ar trace code <path>[:a-b]` | Plan step ↔ code ↔ requirement | FR-2, AC-13 |
| `ar versions <path>` | Maps each `vN` to its commit, for a family document | FR-7 |
| `ar sync` | Brings the semantic index up to date with the files | FR-6 |
| `ar tokens …` | The FR-5 count | FR-5, AC-14 |
| `ar replay prepare …` | Rebuilds one past gate leg's inputs | AC-2, AC-5 |

**Why this shape.**

- **Text is always served from the file or from git, never from the index.** So FR-6 holds by
  construction for `get`, `find` and `trace`, and "as of a commit" (AC-2, AC-13) is one `--at
  <rev>` that reads `git show <rev>:<path>`.
- **The index only finds addresses.** Its staleness can cost recall only, and that is reported
  (§2.4).
- **A CLI adds no always-loaded context.** Every relevant agent has Bash, and the CLI does the
  parsing, git and HTTP work outside the agent's context.

**Rejected alternatives.**

- *Serving text from the index.* It would need a freshness check falkor-chat does not have, and
  one index per past commit.
- *A new MCP server.* It adds tool schemas and a process for no gain over Bash.
- *Paging with `Read` offsets.* There is no citation, no stale notice and no past-commit view.
  `ar outline` still prints line ranges, so `Read` stays available as a fallback.

### 2.2 Addresses and citations (FR-1, FR-3)

**Address grammar** (printed by every command, accepted by `ar get`):

```
<repo-path> [@<rev> | v<N>] [§<num> | §"<heading text>" | #<item-id>] [~p<k>]
```

- **`§<num>`** is the numeric prefix of a numbered heading (`### 4.2 Contracts` → `§4.2`).
- **`§"…"`** is a unique case-insensitive substring of an unnumbered heading. Several matches list
  the candidates; `ar` never guesses.
- **`#<item-id>`** is a list item or table row whose first token is an id. The id shapes include
  multi-segment and letter-after-dash ids (`FR-6`, `AC-12`, `S09`, `P0-08`, `K-102`, `U3e.2`,
  `N2-1`, `P2-M1`), and sentence-final punctuation is not part of an id. The exact regex is the
  implementer's choice, and the B1 tests pin it (§4).
  - A table row prints with its header row.
  - Cell ranges (`S09–S17`) are expanded only when *searching* for an id.
- **`@<rev>`** is any git rev. **`v<N>`** resolves through `ar versions` (§2.8).
- **`~p<k>`** names part *k* of a long leaf (search output only).
- **Extent.** A section runs to the next heading of the same or a higher level. Headings inside
  fenced code are ignored.

**Citation line.** Every `get` and `trace` block opens with one line:

```
[cite: `docs/plans/connected-code-graph-p0-2.md` §4.2 "Contracts (S03, S04, S30, S33)" @3f2a1c9 L521-L590]
```

- `@<sha>` is HEAD when the file is clean, and `@wt` when the working tree differs.
- Code is cited `` `cpg/ccg/ids.py:L60-L88` @6f4e34f9 ``, with `(symbol <qname>)` added when a
  fresh graph supplies one (§2.5).

**FR-3 obligation** (stated in the skill, C1): a claim drawn from `ar` output carries that block's
citation. A search snippet is never cited as content; the agent `ar get`s it first.

**Superseded and archived files** are still served, with one line naming the `Superseded by:`
target, so old citations such as `cpg/ccg/ids.py` → `-p0.md` resolve and point forward.

### 2.3 Semantic index (FR-1, AC-12)

**Corpus.** `agent-reading/corpus.toml` (read with `tomllib`) fixes what is indexed.

- **Include:**
  - `**/docs/requirements/*.md`;
  - `**/docs/plans/*.md`, except `*-coordination.md`;
  - the live specs each `AGENTS.md` names;
  - every FR-4 home (§2.7).
- **Exclude:**
  - `deprecated/**`, `**/archive/**`, `reviews/`, `HISTORY.md`, `BACKLOG.md`, `**/.env*`;
  - any file whose `Status:` is `archived` or `superseded`.

  Excluded files stay fetchable through `get`, `find` and `trace`.

**Index unit.**

- A *leaf section* runs from one heading to the next heading of any level (or is the preamble).
  Empty leaves are skipped.
- A leaf over ~600 words splits into parts `~p1…` at paragraph boundaries. A table splits at row
  boundaries, and each part repeats the header row.
- Each unit is one `Document`:
  - `title` = `doc:<component>:<address>`, hashed down past 200 characters;
  - `text` = a header line, `Source: <address> · sha256:<12 hex of body> · rev:<sha>`, then the
    body.

  The header line keeps every text unique, so the auto-supersede tier never hides one section
  behind an identical one.
- **Component** = the first path segment, or `root` for the repo-root `docs/` tree and repo-root
  files.
- Leaf size, the query prefix, K, the floor and the over-fetch are calibrated in B5 (A3(e)).

**Where.** One workspace, `ws:team-docs`, on the same falkor-chat machinery, embedder and model
overlay as `ws:agent-team`. It is served by its own process on port 8201, separate from
`ws:agent-team`'s corpus, review queue and calibrated floor.

**Search path** (`ar search '<q>' [--component X | --all]`):

- `--component X` searches X plus `root`.
- `--all`, or no flag, searches everything.

1. Incremental sync, without waiting (§2.4).
2. Query `GET /documents/search` with the doc prefix and an over-fetched `limit`. The server caps
   `limit` at 200, which is the ceiling for scoped over-fetch; B5 states it.
3. Map each `documentId` to its address through the local manifest, falling back to the
   document's `title`.
4. Keep only `doc:` hits in scope. Keep the best hit per section, then apply the floor. A scoped
   search that fills fewer than K says so and suggests `--all`.
5. Print K lines: `address · score · freshness · first line`. `--expand 1` prints the top section
   through `get`.

**Server change: one flag.** `FALKORCHAT_ENABLE_EXTRACTION` defaults to on, so nothing changes
unless it is set. When it is off, `_build_default_app` passes `ingestion_pipeline=None`, and
embedding stays wired. `create_app`'s injected parameter is untouched. `ws:team-docs` runs with
extraction off.

### 2.4 Never silently stale (FR-6, AC-6), and K-102

**Text.** `get`, `find` and `trace` read the file, or git at `--at`, at call time. Each stale
case prints a `STALE:` line that names the file to read:

- an unresolvable address, followed by the file's outline;
- a vanished file: its rename target from `git log --follow --diff-filter=R`, or "deleted at
  `<sha>`";
- an unresolvable `v<N>` or `@<rev>`.

**Index.**

- **`ar sync`:**
  - parses the corpus;
  - diffs each unit's body hash against `agent-reading/.state/<ws>.json`, which is gitignored and
    held under an `fcntl` lock;
  - for each changed unit, `DELETE`s the old document and `POST`s the new one (the `cobb`
    pattern, `claude/docs/plans/agent-knowledge-base-strategy.md` §4.5).
- **Waiting.** Only an explicit `ar sync` waits, polling each new document until it is `ready`
  (bounded). A `failed` document, or one not ready in time, is retried once and then recorded as
  **uncovered**.
- **`ar search`** posts changed units and does not wait. It labels each hit by comparing the hit's
  indexed hash with the current section:
  - match → `fresh`;
  - mismatch → `index older than file — ar get is current`;
  - unresolvable → `STALE:`.

  Changed units not yet embedded, and any uncovered units, produce one `COVERAGE:` line ("N
  sections not searchable: <files>; use `ar find`").
- **`ar sync --reconcile`:**
  - lists all documents and unembedded chunks with read-only `redis-cli GRAPH.RO_QUERY
    ws:team-docs …` (the K-102 query);
  - deletes orphans;
  - re-ingests failed or unembedded units.
- **`ar sync --rebuild`** drops and re-seeds the graph (`GRAPH.DELETE ws:team-docs`, then
  `seed_team_docs.sh team-docs`, B2), rather than deleting documents one by one. Each REST delete
  leaves an audit node (`repository.delete_document`). The guard cannot see the `GRAPH.DELETE`
  inside `ar`, so C1 adds `ar sync --rebuild` to the guard catalog.

**K-102 is a dependency, not a side task.**

- B3's RCA must cover both failure classes in §1, and must decide whether `:8200` also runs with
  extraction off, so that `ws:agent-team`'s extracts stop competing for the GPU. That decision
  belongs to K-102's owner, because it changes `:8200`'s behaviour.
- K-070 (embeds before extracts) and the update-detection signals sync will trigger (K-064,
  K-091) are cited, not changed.
- B4 depends on B3's RCA, and B4's acceptance is "zero uncovered after `ar sync --reconcile`",
  with the first-pass uncovered count recorded.

**Code locations.** Line ranges always come from git at the requested rev. Graph symbol names are
used only when `git diff --quiet <sourceCommit> <rev> -- <file>` holds and `sourceDirty` is
false. Otherwise the output reads `STALE: symbol data for <file> predates <rev>; read
<file>:<range>`.

### 2.5 Plan step ↔ code ↔ requirement (FR-2, AC-13)

The link is **derived on demand from git, the plan documents and in-file citations**. It is not
stored, so it needs no sync and cannot go stale. Symbols come through one `SymbolSource`
interface. `CpgSource` ships now. A connected-code-graph source is a backlog item (§6).

**Family token.** A plan's family token is its header `Tracks:` id (`C-1001`), else its topic slug.
Its requirements document is the one its header or ledger header names, else
`<component>/docs/requirements/<family>.md`.

**`ar trace step '<plan> #<id>' [--at <rev>]`:**

1. **Plan side.**
   - The step row.
   - Its aliases: the id-shaped tokens in its first two cells, such as `S09` and `P0-08`.
   - Every section or row that mentions the id, ranges included (the `§4.0` row `(b) … S09–S17`).
2. **Code side.**
   - Commits reachable from `<rev>` whose message holds the family token **and** an alias, both as
     whole tokens. Each commit is listed with its files and post-image hunk ranges
     (`git show -U0`).
   - Documentation paths (`docs/`, `HISTORY.md`, `BACKLOG.md`, `AGENTS.md`) are listed
     separately.
   - The row's "Deliverable (files)" are cross-checked both ways.
3. **Requirement side.**
   - The FR/AC ids reached by a **citation closure** to depth 3: the row, its sections,
     same-document `§` refs and path-qualified refs.
   - Each id is resolved to `#FR-n` in the family requirements document.
   - Alias refs the closure cannot resolve ("proposal §4") are printed as `unresolved ref`.
   - The family requirements document is always printed as the root.
4. **Symbols.** `CpgSource` reads `cpg_<component>` through `redis-cli`: the `CpgBuildInfo` source
   commit, and the `METHOD` nodes by `FILENAME` whose line span overlaps a hunk. It is used only
   under §2.4's freshness rule.

**`ar trace code <path>[:a-b] [--at <rev>]`:**

- Commits come from `git log -L a,b:<path>`, or `--follow` when no range is given.
- Family tokens and step ids are parsed from the commit messages.
- Each (family, id) resolves to its row in the family's plans as of `<rev>`. A non-superseded
  holder comes first, and any superseded holder is also named.
- The file's own citations are added: `docs/…md` paths, `§` refs and step ids in comments and
  docstrings.
- The requirement side runs as in step 3 above.

**For families started after delivery** (C4 writes this into the conventions):

- commit subjects name the family token and the step ids;
- step tables carry a `Reqs` column.

**Rejected alternative:** a persisted link graph. Git already holds the data exactly, and a store
would need its own sync.

### 2.6 One token-counting rule (FR-5, AC-14)

**A leg** is one transcript segment of one agent. It starts at a string-content user entry (the
spawn prompt, or a SendMessage resume) and runs to the next one. An entry that starts with
`<system-reminder>` never opens a leg. Its nested subagents come with
it: an agent whose `.meta.json` `parentAgentId` is this agent, or whose `toolUseId` is an `Agent`
`tool_use` inside the leg, followed recursively.

**Tokens of a leg** = for each `message.id` in the leg's `assistant` entries (`<synthetic>`
included), the last entry's

> `input_tokens + cache_creation_input_tokens + cache_read_input_tokens + output_tokens`,

summed. This is the per-message rule of `docs/plans/typed-decision-models4.md` §3.7,
re-implemented here because components stay independent.

**Attribution.**

- A **run** is the set of legs attributed to one (unit, role). Gates stay separate runs from
  builders and are never summed with them.
- `ar tokens legs --agent ID` lists the legs, each with its index, timestamp and the first 120
  characters of its message.
- `ar tokens --agent ID --legs 1-2 --unit "C-1001 U20" --role gate` records one run.
  - A continuation leg (a usage-stop resume, "continue") is given to the same unit as the leg it
    continues.
  - A leg already attributed in `runs.jsonl` cannot be attributed again. A leg left unattributed
    is reported.
  - A cold restart under a new id joins the run by naming both ids.

So `teco` keeps resuming gate agents across units, as it does today, and each unit still gets its
own legs.

**Record.** Each record is appended to `agent-reading/data/runs.jsonl`, which is committed and holds
numbers only, never transcript text. It carries:

- unit, role (`agentType`), `message.model`;
- agent id(s), leg indices and nested ids;
- the four components and the total;
- `finalContext`, `peakContext` and the message count;
- `ruleVersion: 1`, the counter's commit sha, `retrieval: true|false`;
- `sha256` per transcript.

The coordinator's own session is outside FR-5.

**Headline:** the processed total, with `finalContext` as a secondary figure. A3 confirms the
headline or replaces it before any figure is reported. Every component is stored, so a change
needs no recount.

**Rule home:** `agent-reading/README.md` §"Token counting", which `teco.md` and the ledgers cite.
Ledger Cost cells read `b 14.2M · g 9.1M (FR-5)`.

### 2.7 `AGENTS.md` core (FR-4, AC-11)

**The core** keeps the context-file convention's question, *what is true now, and what will bite*,
for this scope: commands and working directory, hazards, invariants. It adds one "where to look"
line per moved block, naming the home and the `ar` call.

**Everything else moves by kind:**

| Material | Home | Retrievable via |
|---|---|---|
| History, dated changes, settled rationale | the `HISTORY.md` of the component it concerns (`salesperson/` → `falkor-chat/docs/HISTORY.md`; `claude/` → a new `claude/docs/HISTORY.md`). History of a repo-wide convention is cited rather than moved **only** when an existing record holds it; the manifest gives that record's `ar get` address. Any other convention history is carried verbatim in C2's manifest entry in `agent-reading/docs/HISTORY.md` | `ar get`/`find` only (FR-1, Out of scope) |
| External contracts, design | the component's live spec | all of `ar` |
| Operator how-to, script tables, setup | the component's `README.md` | all of `ar` |
| Repo-wide conventions: documentation convention, lifecycle, owner table, collision rules, citation and filename grammar | new repo-root **`CONVENTIONS.md`** (a sibling of `REVIEW.md`, named in root `AGENTS.md`) | all of `ar` |
| `claude/` team machinery: hook cores, commit-authority detail | `claude/README.md` | all of `ar` |

**Manifest.** Each move lists every item with its home (`<path> §x`) in `agent-reading/docs/
HISTORY.md`, under the entry for that change. That list is AC-11's list. The `cpg/` ownership of
root `docs/HISTORY.md` is left as it is.

**Every file moves in two commits.**

1. A **pure relocation** commit with no rewording, so the no-loss check is a diff.
2. A separate core-rewording commit.

Root `AGENTS.md` goes first (C2). Active citations of "root `AGENTS.md` … bullet" in prompts,
skills, `REVIEW.md` and `claude/README.md` are repointed in the same unit. Archived documents keep
resolving through the pointer line. `opencode/AGENTS.md` (already core) and `deprecated/` are not
touched.

### 2.8 One current design per plan family (FR-7, FR-8, AC-7)

**The new rule** goes into `CONVENTIONS.md` and covers plan families started after delivery (a
plan with its `-ml` and `-graph` notes):

1. **A family document is always revised in place.**
   - No ordinal successor, `Extends:` or `Supersedes:`.
   - A later document never amends an earlier one by override. The owner amends the earlier one
     in place.
   - New scope with its own topic (a phase plan, `<family>-p1.md`) is a new topic slug.
2. **Versions are pinned by git.**
   - Every version that is approved, gated or executed against is committed (path-limited) when
     its gate or dispatch happens. For these documents, this is the stakeholder-approved exception
     to the docs-only commit-once rule.
   - Every later edit bumps `Version:`.
   - `ar versions` maps each `vN` to the newest commit carrying it. It warns when a cited
     version's content changes without a bump.
3. **Gated text is cited with its version** (`` `<path>` v<N> §x ``) in reviews, ledgers, briefs
   and code. Sections and ids are never renumbered.
4. **Families started before delivery keep today's rule 5.**

**Rejected alternatives.**

- *Always-`Supersedes:` successors.* Each revision restates the whole design, and the file count
  still grows.
- *A generated current view.* FR-8 requires the rule itself to change.

### 2.9 Replaying past gates (AC-2, AC-5)

**The replay unit is one gate leg** (§2.6): an (agent id, leg) pair holding one recorded pass.

`ar replay enumerate` builds the replay set, one row per pass with ≥1 blocker or major in
`docs/reviews/connected-code-graph-p0-impl.md` for U20–U32.

- **Locating each pass's leg:**
  - first through the ledger's gate ids;
  - for U26, U28, U30, or any pass without one, through the parent session's `Agent` `tool_use`
    (matched on its `description`/`prompt` naming the step) and the `.meta.json` `toolUseId`;
  - for a resumed agent, through the leg whose message names the step and pass.
- A pass with no locatable leg is listed as **not locatable**.

`ar replay prepare --agent ID --leg K`:

1. **Brief.** The brief is the leg's message text.
   - The main repo's absolute root is rewritten to the worktree root, and every rewrite is
     recorded.
   - One paragraph pointing to the retrieval skill is appended.
   - A resumed leg that relies on earlier context is replayed as a cold spawn of its own text.
     A3(f) rules on whether that is comparable.
2. **Base.** The base is the newest commit before the leg's timestamp, checked out in a
   `git worktree`.
3. **Tree as the gate saw it.**
   - **Candidate.** The base commit, with the builder's `Write`/`Edit` inputs up to the leg's
     start replayed onto it.
   - **Captures.** Every file content the gate leg itself captured:
     - `Read` results;
     - Bash read output. The parser handles `cd`, `;`/`&&` sequences, `var=path` assignments
       (expanded), absolute tool paths (`/usr/bin/grep`), and reads by `cat`, `cat -n`,
       `sed -n 'a,bp'`, `head -n`, `grep -n` and `awk` with `NR` ranges. `| head` and `| cut -c`
       count as truncating filters, so their output is a partial capture.
     - Chained reads' concatenated output is split only where the command shapes delimit it (line
       numbers, fixed ranges). Otherwise the chain is unparsed.

     A file the gate writes contributes only captures made before its first write to that file.
     Unparsed commands are listed. Files they mention, and files the gate executed without
     printing (`bash $S`, pytest targets, scripts copied to scratch), are classed **never
     captured**.
   - **Check per file.**
     - A whole capture (a full `Read` with `numLines == totalLines`, or an un-truncated plain
       `cat`) **replaces** the candidate. This recovers Bash-written and uncommitted files.
     - A partial capture is checked line by line against the candidate.
     - Each file the gate looked at is classed as **verified** (whole), **consistent** (every
       captured line matches), **contradicted**, or **never captured**.
   - Any contradicted file makes the pass **not reconstructable**. Never-captured files are
     served from the candidate and listed.
4. **Agent definition.** The gate's `claude/<name>/<name>.md` as of the base commit becomes the
   worktree's `.claude/agents/<name>.md`, overriding the user-level symlink.
5. **Index.** `ws:team-docs-replay` (port 8202) is synced to the worktree. Passes run in
   chronological order, so each sync is incremental.

A replay runs as a headless session in the worktree that spawns the gate as a subagent. The
recorded findings are never in the brief. Its cost is counted with `--projects-dir` set to the
worktree's project slug.

**AC-2 denominator.** Every pass in the replay set counts, the not-locatable and
not-reconstructable passes included. A pass counts as replayed only when every one of its
deliverable files is verified or consistent, unless the stakeholder's B7 ruling says otherwise.
D2 reports AC-2 per class. A pass that cannot be replayed cannot show that its findings
were caught, so D2 reports AC-2 as met only when every pass was replayed and passed. Otherwise
D2 names the shortfall for the stakeholder. A3(f) records the selection bias.

## 3. Build units and sequencing

**Gates:** `an` `analyst` · `ds` `data-scientist` · `gd` `graph-dba` · `tc` `teco` check · `sh`
stakeholder.

**Order.**

- Phase A runs now.
- Phase B runs in parallel with A4.
- Phase C, the first change agents read, waits until A5 closes the baseline window.
- Phase D verifies.

| Unit | Owner | Deliverable | Depends | Gate |
|---|---|---|---|---|
| A2 | `tdd-engineer` | `agent-reading/` skeleton + `ar tokens` | precondition | an |
| A3 | `data-scientist` | `docs/plans/agent-reading-cost-ml.md` | — | an |
| A4 | `teco` (rolling) | Baseline records | A2; A3 before any figure is reported | tc |
| A5 | `qa-engineer` | `docs/test-reports/agent-reading-cost-baseline-report.md` | A3, A4 | ds + an |
| B1 | `tdd-engineer` | `ar` document core | A2 | an |
| B2 | `coder` | Extraction flag + `start_team_docs.sh` | — | an |
| B3 | `analyst` → `coder` | K-102 RCA and fix | — | an |
| B4 | `tdd-engineer` | `ar sync` / `ar search` | B1, B2, B3 (RCA) | an + gd |
| B5 | `qa-engineer` (A3's method) | Calibration + AC-12 | B4, A3 | ds |
| B6 | `tdd-engineer` | `ar trace` + `CpgSource` | B1 | an + gd |
| B7 | `tdd-engineer` | `ar replay enumerate/prepare` | A2, B4 | an; sh rules on D2 |
| C2 | `cobb` (`teco` commits) | Root `AGENTS.md` → `CONVENTIONS.md` + core | A5 | an |
| C4 | `cobb` drafts | Rule-5 rewrite, commit-rule exception, sweeps | C2 | an + sh |
| C3 | `cobb` | The other six `AGENTS.md` | C4 | an per file |
| C1 | `cobb` | Skill, prompt pointers, `teco` rules, hooks | C3, B5, B6 | an |
| D1 | `qa-engineer` | AC-6, AC-13, AC-7 fixture family | C1 | an |
| D2 | `teco` runs, `analyst` reads | AC-2 replays + AC-5 | B7 + its `sh` ruling, C1 | ds |
| D3 | `teco` (rolling), `qa-engineer` reports | AC-3, AC-4, AC-16 | C1 | ds + an |

C2 → C4 → C3 → C1 are serial because they share `CONVENTIONS.md`, `claude/AGENTS.md` and the
prompts.

### Phase A — baseline, starting now

**Precondition — transcript retention.** The stakeholder sets `cleanupPeriodDays` in their own
Claude Code settings. When A2 is dispatched, `teco` checks two things:

- that the setting is present;
- that every U20–U32 agent id has its transcript.

`teco` records the result in the ledger. If the setting is absent, `teco` copies at once:
U20–U32's builder and gate transcripts, their `.meta.json` and parent session files go to
`~/.local/share/agent-reading/transcripts/`, outside the repo, with a `sha256` manifest.
`ar tokens --projects-dir` reads that copy.

**A2 — skeleton and `ar tokens`.**

- **Files:**
  - `agent-reading/{README.md, AGENTS.md (short), pyproject.toml, bin/ar}`;
  - `agent-reading/agentreading/{__init__,cli,tokens}.py`;
  - `agent-reading/tests/{test_tokens.py, fixtures/transcripts/…}`;
  - `agent-reading/data/runs.jsonl`;
  - `agent-reading/docs/{HISTORY,BACKLOG}.md`;
  - `.gitignore` for `.state/`.
- **Interface:** §2.6 (`legs`, `--legs`, `--unit`, `--role`, repeatable `--projects-dir`,
  `--retrieval`, `--append`).
- **README** §"Token counting" states §2.6's rule.
- **Done when:**
  - `pytest` is green from `agent-reading/`;
  - U24 reproduces 14,153,036;
  - `a72f2b8992f574958` splits into its four legs.

**A3 — `-ml` note** (`docs/plans/agent-reading-cost-ml.md`). It decides:

- (a) the FR-5 headline, given that cache reads dominate it;
- (b) the AC-1 target rule per role, including roles with fewer than 3 runs, and how a shift in
  the mix of docs-only and code units between the windows is handled;
- (c) AC-3's escape definition and recording rule, the before window, the upper-95% bound, and
  how two families with different review regimes are mixed;
- (d) AC-16's comparison;
- (e) the AC-12 golden set and B5's calibration (prefix, K, floor, leaf size, scoped over-fetch
  up to the 200 ceiling);
- (f) AC-2 comparability: definition drift, resumed legs replayed cold, and selection bias.

It is gated before A4 reports any figure.

**A4 — Baseline collection.**

- C-1001 runs as briefed today, including cross-unit gate resumes.
- For each leg, `teco` runs `ar tokens … --unit "C-1001 U<n>" --append` once the leg is final,
  and writes the Cost cell.
- Nothing agents read changes until A5.
- `teco` logs any other change in what agents read during the window, for A5.
- The window closes at ≥10 counted units (up to 15).

**A5 — Baseline report.** It holds:

- the before median per role and the target (AC-1);
- the AC-3 before share and bound;
- the AC-4 before practice (the coordinator's mutant rule as practised);
- the start word count of each `AGENTS.md`;
- A4's log of uncontrolled changes.

The report is the target's one home.

### Phase B — build, in parallel with A4

**B1 — Document core.**

- **Files:** `agentreading/{mdparse,address,docs,gitio,versions}.py`, with tests.
- **Behaviour:** `outline`, `get`, `find` and `versions`; `--at`; the `STALE:` cases; superseded
  pointers; repo root from the working directory or `--repo`.
- **Done when** every address form resolves on fixtures and on
  `docs/requirements/agent-reading-cost.md`, `docs/plans/connected-code-graph-p0-2.md` and
  `falkor-chat/docs/SERVER.md`.

**B2 — falkor-chat.**

- `config.ENABLE_EXTRACTION`, with its branch at `app.py:619`.
- `falkor-chat/scripts/seed_team_docs.sh <wsId>`:
  - runs `create_workspace.sh`;
  - ensures the ingest actor `User ar-sync` with `QUERIES.md` §2's guarded ensure (as
    `seed_demo.sh` step 1 does).
- `falkor-chat/scripts/start_team_docs.sh` is a few-line wrapper. It:
  - exports `FALKORCHAT_WS_ID=team-docs`, `FALKORCHAT_USER_ID=ar-sync`, `AGENT_TEAM_PORT=8201`
    and `FALKORCHAT_ENABLE_EXTRACTION=0`;
  - brings FalkorDB up and waits for `PING`, the same way `start_agent_team.sh` step 1 does;
  - runs `seed_team_docs.sh`;
  - then execs `start_agent_team.sh`.

  Overrides give the 8202 replay instance. `ar sync --rebuild` re-runs `seed_team_docs.sh` after
  its `GRAPH.DELETE`.
- **Tests:**
  - flag off: no extract job is scheduled, and a document reaches `ready` on embed alone;
  - flag on: behaviour is unchanged;
  - an ingest on a freshly seeded workspace returns 201;
  - the same ingest without the seed fails with `UnknownActorError`;
  - a cold bring-up with FalkorDB down succeeds, both directly and through
    `ensure-environment.sh`.
- The query suite stays green.

**B3 — K-102** (owner per `falkor-chat/docs/BACKLOG.md`). Covers:

- an RCA of both failure classes (§1);
- the `:8200` extraction decision (§2.4);
- the fix;
- a repair path for `ws:agent-team`.

**B4 — Index.**

- **Files:** `agentreading/{corpus,index,falkorclient}.py` (urllib REST and `redis-cli`
  subprocess), `corpus.toml`, and tests with a fake HTTP server.
- **Behaviour:** §2.3–§2.4.
- **Live acceptance:** a full `--rebuild` then `--reconcile` ends with zero uncovered units; the
  first-pass uncovered count and the timings are recorded.
- `graph-dba` checks the read queries.

**B5 — Calibration** per A3(e), scoped and `--all`. Writes
`docs/test-reports/agent-reading-cost-retrieval-report.md` (the AC-12 evidence). The values go
into `corpus.toml` and the skill.

**B6 — Trace.** `agentreading/{trace,closure,symbols}.py`, with fixture repositories built in
`tmp_path`. `graph-dba` checks `CpgSource`'s two queries.

**B7 — Replay.** `agentreading/replay.py`, tests, and a README section (§2.9).

- **Probes first:**
  - the project-level agent override;
  - reconstruction on U27/S09 (code);
  - reconstruction on a docs-only pass (S04 or S05 Pass 1);
  - `a72f…`'s S05 Pass 1 leg (resumed, multi-unit).
- **Each probe reports** its file counts per class (verified, consistent, contradicted, never
  captured) and its unparsed captures.
- **First deliverable:** the enumerated replay set, with those rates for every pass and the
  not-locatable and not-reconstructable passes listed. It gives D2's real count and cost.
- **Stakeholder ruling.** The stakeholder rules on D2's feasibility from those rates before D2 is
  dispatched.

### Phase C — cut-over, after A5 closes the window

**C2 — Root `AGENTS.md`.**

- Relocation into `CONVENTIONS.md`, then the core (§2.7).
- The manifest goes in `agent-reading/docs/HISTORY.md`. Each cited-not-moved history item is
  listed with the `ar get` address that resolves it (§2.7).
- Citations are repointed.
- Root `AGENTS.md` gains the `agent-reading/` Structure bullet and Component-docs row.

**C4 — Rule 5 and the commit rule.**

- §2.8 goes into `CONVENTIONS.md`.
- The exception goes into all five homes of the docs-only commit-once rule (§1).
- `claude/architect/plan-authoring-techniques.md` gains the `Reqs` column, the version-qualified
  citation form and the commit-subject convention.
- **Precondition:** `tico` has logged the stakeholder's commit-at-gate decision in
  `docs/requirements/agent-reading-cost.md`'s decision log.
- **Two unfiltered sweeps** from the repo root. The results go in the C4 review.
  - Rule 5: `git grep -n -i -E 'collision rule|rule 5|ordinal on the slug|successor|Extend(s|ed by)|Supersed|selector'`.
    That is about 2,500 hits in about 380 files, so it is ruled **per file**: each archived file
    is one no-change block with its hit list, and each live file gets an owner plus a fix or
    no-change ruling.
  - Commit rule: `git grep -n -i -E 'docs-only[- ](chain|batch)|terminal (state|commit)|commit(s|ted)? once'`.
    That is about 120 hits in 46 files, ruled per hit. It must reach all five files in §1.

**C3 — The other six files** (`claude/`, `falkor-chat/`, `model-bench/`, `salesperson/`,
`mcp-monitor/`, `decision-models/`). One dispatch per file, run serially, each with two commits
(§2.7). Manifests go in `agent-reading/docs/HISTORY.md`.

**C1 — Skill, prompts and hooks.**

- `skills/team-doc-retrieval/SKILL.md`:
  - the commands, with the grammar cited from the README;
  - the FR-3 obligation and "never cite a snippet";
  - the B5 values;
  - `STALE:`/`COVERAGE:` handling.
- One pointer line in each agent prompt that reads team documents.
- `teco.md`:
  - a brief names the sections a unit needs and its `--component`, and says `ar get`, never
    "read X whole" unless the unit revises X;
  - leg attribution and Cost cells per §2.6;
  - escape recording per A3(c).
- `claude/teco/hooks/ensure-environment.sh` brings up `start_team_docs.sh`.
- The shared core `claude/scripts/guard-destructive-ops.sh` adds `ar sync --rebuild` to its
  catalog. That one edit covers:
  - the thin `devops`, `graph-dba` and `qa-engineer` wrappers, which stay unchanged;
  - `coder` and `tdd-engineer`, through `guard-broad-bash.sh`.
- Update the catalogs (`skills/README.md`, `claude/README.md`).

### Phase D — verification

- **D1:** AC-6, AC-13 and the AC-7 fixture family (§4).
- **D2:** AC-2 and AC-5 over B7's replay set (§2.9, §4).
- **D3:** the first 20 units with retrieval (AC-3, AC-4), and their first ≥10 per role (AC-16).
  Results go to `docs/test-reports/agent-reading-cost-report.md`.

**Close:**

- The live specs are updated once (§6).
- `analyst` writes `docs/reviews/agent-reading-cost-impl.md`.
- History entries are written.

## 4. Test strategy (by AC)

| AC | Altitude | How it passes, and the control that would turn it red |
|---|---|---|
| AC-1 | process | A5: ≥10 units, every leg of every builder and gate attributed in `runs.jsonl` with `retrieval:false`, and a target per role per A3(b). *Red:* a ledger agent id with an unattributed leg; A5's checker recounts against the ledger |
| AC-2 | replay | D2, after the stakeholder's ruling on B7's rates: every pass in B7's replay set (§2.9), with its denominator rule. Each recorded blocker or major must be reported, as matched by an `analyst` reader who is not the replay gate. *Control:* a planted contradicting capture (from `Read` or Bash) halts reconstruction |
| AC-3 | process | D3 per A3(c) |
| AC-4 | process | D3: for each retrieval unit that ships tests, the coordinator's mutants follow A5's recorded before practice, and every mutant goes red |
| AC-5 | replay | D2 reader: every claim resting on an `ar` result carries its citation. The reader maps claims to the `ar` calls in the replay transcript |
| AC-6 | unit + live | B1/B4/B6 tests. D1 live: edit a section after a sync, rename a heading, and move code past a CPG's source commit; `search`/`get`/`trace` then give current text or a `STALE:` that names the file |
| AC-7 | review + fixture | C4's two sweeps, every live hit owned. D1: a fixture family taken through gate → revision → dispatch gives one current document; `ar get '<p> v1 §2'` returns the gated text, and `ar versions` warns on a planted post-gate edit with no bump |
| AC-11 | review | C2/C3 gates: each word count < A5's start count. The relocation commit's removed lines map to manifest rows whose homes contain them (`ar find`); `ar get` resolves each home. *Control:* a planted removed paragraph with no manifest row turns the check red |
| AC-12 | eval | B5: each golden question asked scoped and `--all`; the top-1 `ar search --expand 1` returns the labelled section with its citation and nothing more. The bar comes from A3(e). *Control:* a section in another component is not returned under the wrong `--component`, and is returned by `--all` and by `ar get` |
| AC-13 | live | D1: `ar trace step … --at <unit commit>` for U20, U21, U27, U28 and U30 reaches the committed code. `ar trace code` on a changed hunk returns the step and its FR/AC or family requirement. *Controls:* doc-only U22/U23 give "no code commits"; a planted `S99` commit matches nothing |
| AC-14 | unit + review | A2 tests. At close, every Cost cell after A2 carries `(FR-5)` and matching `runs.jsonl` records |
| AC-16 | process | D3 per A3(d) |

**Unit-level behaviours to drive red→green** (`tdd-engineer`, in this order):

1. **`tokens`:**
   - last entry per `message.id` (outputs 3, 900, 2087 count 2087, written by hand; a first-entry
     mutant goes red);
   - legs split at string user entries, and a `<system-reminder>` entry after a spawn opens no
     leg;
   - one agent serving two units gives two records, with hand-written expected figures;
   - re-attributing a leg is refused, and an unattributed leg is reported;
   - nested agents are included through `parentAgentId` or `toolUseId`, and an unrelated agent is
     excluded;
   - `<synthetic>` is included;
   - a missing id fails loudly.
2. **`mdparse` / `address`:**
   - numbered and unnumbered headings; a fenced `#` is ignored;
   - an **id-shape coverage probe**: `P0-08`, `N2-1`, `P2-M1`, `U3e.2`, `S09.` at sentence end,
     and an id in a list item and in a table row;
   - ranges are expanded for search only;
   - duplicate headings give an ambiguity list;
   - every grammar form parses, and malformed input is refused.
3. **`get`:**
   - current text, `@rev` and `v<N>` (fixture repo with three versions);
   - rename gives `STALE:` with the new path, and deletion gives `STALE:` with the sha;
   - the superseded pointer line.
4. **`index`:**
   - a changed hash gives exactly one delete and one post; an unchanged hash gives no call;
   - the path-derived component and `root`, and the scope filter;
   - search does not wait; a pending unit gives `COVERAGE:`;
   - `failed` → one retry → uncovered;
   - lock contention waits;
   - an index-older-than-file hit is flagged.
5. **`trace`:**
   - the commit filter needs both tokens as whole tokens (an `S1`/`S10` mutant goes red);
   - hunk ranges;
   - the deliverable cross-check works both ways;
   - the closure stops at depth 3;
   - an unresolved alias is reported;
   - CPG symbols are printed only when the source commit matches.
6. **`replay`:**
   - leg location through the ledger, then through `tool_use`;
   - the root path is rewritten;
   - a whole `Read` and a plain `cat` each replace the candidate;
   - `sed -n`/`grep -n` captures are checked line by line, including the
     `cd; f=…; sed -n …$f; grep … $f | cut -c…` chain;
   - an executed-but-unprinted file is classed never captured;
   - a capture made after the gate's own write to that file is ignored;
   - the class counts per file are right;
   - a planted contradicting capture halts;
   - the brief is the leg text plus exactly one paragraph.

## 5. Risks

- **C-1001 cadence.** If C-1001 pauses, A5 does not close and Phase C waits.
- **Transcript loss** voids AC-2 and any recount. The precondition check and its immediate copy
  are the mitigation.
- **Attribution of the bundle.** The `AGENTS.md` shrink and retrieval ship together, so AC-16
  measures the bundle.
- **AC-2 is lenient by design.** It detects only a large drop. Today's model runs the as-of
  definitions, cold-replayed legs lack their earlier context, and only reconstructable passes
  run. A3(f) records all three.
- **The closure may not reach an FR/AC for some old steps.** AC-13 then rests on the family root
  and the printed chain. The `Reqs` column is the remedy for new families.
- **Update detection runs on every ingest** (K-064, K-091). If B4's timed rebuild shows it
  dominating, a second flag is a follow-up.
- **`cobb`'s write guard** asks on `AGENTS.md`, `CONVENTIONS.md` and READMEs. Each ask is approved,
  or `cobb` widens its allowlist for C2–C4.
- **Rule-5 discipline.** A post-gate edit without a bump is caught only by `ar versions` and by
  reviewers. C4 proposes a `REVIEW.md` line.
- **Security.** `ar` reads files, git and localhost REST and Redis. The corpus excludes
  `**/.env*`.

## 6. Documentation impact and close

- **New live spec:** `agent-reading/README.md`, the whole-component spec, checked before the first
  close.
- **Changed live specs** (updated once, at close):
  - `falkor-chat/docs/SERVER.md` §1.3 (the flag) and §1.12 (extraction optional);
  - `falkor-chat/README.md` (`start_team_docs.sh`);
  - `CONVENTIONS.md` and root `AGENTS.md`.
- **Relied-on open feature:** C-1001, through `docs/plans/connected-code-graph-p0-2.md` §3.
- **Backlog:** `agent-reading/docs/BACKLOG.md` gets the connected-code-graph `SymbolSource`
  (`docs/plans/connected-code-graph.md` §3.5–§3.6), triggered when C-1001 P1 ships anchor
  resolution.
- **History:**
  - `agent-reading/docs/HISTORY.md` (including the C2/C3 manifests);
  - `falkor-chat/docs/HISTORY.md` (B2, B3).

## 7. Ready to implement

Implement the plan at `docs/plans/agent-reading-cost.md`. Phases A–D in §3; A2 and A3 start once
the retention precondition is checked.
