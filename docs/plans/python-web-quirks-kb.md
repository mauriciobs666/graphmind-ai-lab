# python-web-quirks on the knowledge base — implementation plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Extends:** `docs/plans/python-web-quirks-kb-ml.md` (method, Set A, targets) · **Reviews:** `docs/reviews/python-web-quirks-kb-plan.md`

Requirements: `docs/requirements/python-web-quirks-kb.md` (FR-n / AC-n). Method note:
`docs/plans/python-web-quirks-kb-ml.md` (ML §n). Ledger: `docs/plans/python-web-quirks-kb-coordination.md`.
Review findings are cited as F-n. This plan states the current design only.

## 1. Goal and scope

Replace the `skills/python-web-quirks/SKILL.md` skill (76,291 bytes, loaded whole) with (a) a git-tracked copy
of record, (b) a rebuildable index of it in `ws:agent-team`, retrieved through the existing
`agent-kb-retrieval` convention, (c) tool grants and a one-line pointer in eight consuming agents, and (d) a
deleted skill with no dangling citation. Out of scope: the requirements' Out-of-scope list; the retrieval
convention's floor, top-K and prefix literal (changed only if S12 proves it necessary, for all users, FR-7).

**Hard constraints on every step brief.** No step of this plan writes to `ws:agent-team` except S13's ingest
(agents' routine Learning-capture writes continue under their own prompts, outside the plan; §D11 handles their
effect on scoring). **No tool built here can delete or replace a document in `ws:agent-team`**; the stakeholder
does that with `curl -X DELETE` (`falkor-chat/docs/SERVER.md` documents routes). Destructive or repeated work
runs only in `ws:pwq-test`. No step runs a tree-mutating git command; `teco` commits. The shared `:8200` server
is never stopped. CPG: considered, not relevant — docs, config, a stdlib script and agent-definition edits.

## 2. Findings that shape the design

- The pin is `3fcb171c` (last commit touching the skill): YAML front matter (`description` 973 chars), a
  preamble (H1, scope paragraph, version caveat, a "This is a cache, not the source of truth" blockquote with
  lineage), then 42 `## ` sections; 16 contain an ISO date, 32 a version-like token; no `<!-- body` collision;
  longest title 182 chars (document `title` limit 200); heading lines total 5,767 bytes.
- **Cross-references** (independent sweep over `above|below|earlier|previous|preceding|next|elsewhere|half|
  sibling|section|entry`, each hit read; F1): inter-section — 587 (§22→§21), 624 and 633 (§23→§21,§22), 670
  (§25→§24), 891 (§34 "The bounded-pool entry above"→§2, 32 sections away), 951 (§39→§29), 974 (§40→§30);
  intra-section — 660, 744, 754, 916; pointing outside the file — 439, 861, 987, 999.
- `search_documents` returns only `chunkId, text, documentId, seq, score` (`falkor-chat/server/falkorchat/
  repository.py` `search_chunks`): **no title**. `get_document` returns title and full text.
- Re-ingesting byte-identical text **auto-supersedes** the older current document
  (`create_document_with_auto_supersede`, SERVER.md §14.9 row); an edited text lands as a *pending* suggestion
  (`claude/docs/plans/agent-knowledge-base-strategy.md` §4.5).
- `list_documents` is oldest-first, limit ≤ 200, no filter: it cannot list the gotcha titles in a 476+ document workspace.
- The write guard `claude/cobb/hooks/guard-cobb-topic-writes.sh` allows `claude/*/*.md` (a bare `*` crosses
  `/`), so any `.md` under `claude/<dir>/` is writable by `cobb`; the one stale entry is `skills/python-web-quirks/*`
  (header comment lines 82-93, glob list line 105). `claude/scripts/audit-team.sh` check 1 enumerates only
  directories holding `<name>/<name>.md` (`claude/docs` and `claude/scripts` already exist as non-agent dirs).
  Implementers' deny-lists (`claude/*/*.md`, `*/docs/plans/*`) do not match `.py`/`.json` under
  `claude/cobb/scripts/`, but will prompt on `.md` edits there and on `docs/plans/` files.
- `claude/cobb/scripts/check_content_loss.py` locates each stored claim's whitespace-normalised text as a
  substring of one source section body, reads sections through `flag_split_candidates.parse_sections`
  (**ignores everything before the first `## `**, excludes the heading line), checks only manifest-listed
  headings, and hard-codes `ws:agent-team` in `LIVE_FETCH_SNIPPET` (line 177).
- `ws:agent-team` is served by `:8200` (`falkor-chat/scripts/start_agent_team.sh`; env `FALKORCHAT_WS_ID`,
  `AGENT_TEAM_PORT`; the script itself runs `create_workspace.sh` and `seed_agent_team.sh`, then `exec uvicorn` in the foreground and
  `pip install -e` into the shared server venv). A server's workspace is fixed per process and `GET /health`
  does not report it. REST `POST /documents` takes `producedBy`; `GET /documents/search` is the same
  `search_chunks` as the MCP tool. FalkorDB docs: `GRAPH.COPY` clones "nodes, relationships, properties,
  indexes, and constraints" and errors if the destination exists; vector indexes and embeddings are not named.
- Chunking is the default (`falkorchat/chunking.py`, 1000/150). Over the §D2 units with a 250-400 char header
  I computed **129 chunks** (not 108); 12 units span ≥ 4 chunks (ML §2).
- Consumer agent files: `coder`, `tdd-engineer`, `qa-engineer`, `frontend-engineer` have no `tools:` line;
  `architect`, `analyst` list the retrieval tools; `tico`, `security-expert` enumerate tools without them. The
  retrieval quote line exists in `tdd-engineer:48`, `architect:52`, `analyst:55`, `qa-engineer:30`,
  `frontend-engineer:46`; absent in `coder`, `tico`, `security-expert`.
- `audit-team.sh` check 11 greps the fenced prefix literal in `skills/agent-kb-retrieval/SKILL.md` (`grep -qF`);
  `ROOT` derives from the script's own path.

## 3. Design decisions

### D1. The copy of record: one file, `claude/knowledge/python-web-quirks.md`

One Markdown file in a non-agent directory; every source section keeps its `## ` heading and verbatim body;
units are fenced by marker lines (D3). Rejected: a directory of ~36 files (the listing and by-section read then
need an index that is a second thing to keep in step; a git diff of one file is already per-section); `skills/`
(auto-loaded, FR-8); `claude/cobb/` (it would hold a KB read by eight consumers, unlike the precedents, which are
read by their own folder's agent). The directory is `claude/knowledge/`, beside the existing non-agent
`claude/docs` and `claude/scripts`. Checked this run: the real guard `guard-cobb-topic-writes.sh` returns `allow`
for `claude/knowledge/python-web-quirks.md` in relative, `/abs/x/claude/…` and the repo-absolute forms (glob
`claude/*/*.md`), so AC-12 needs no new allow entry: the change is retiring the `skills/python-web-quirks/*` entry
in the same change that creates the copy (S4). The only `audit-team.sh` loop over `claude/*/` is line 107, which
keeps only directories holding `<name>/<name>.md`, so a `knowledge/` directory is not enumerated as an agent. The
layout claim "one folder per agent" appears in `claude/AGENTS.md:5` and root `AGENTS.md:44`; S16b (`cobb`) adds a
one-line exception to each (`claude/knowledge/` holds shared KB copies of record; not an agent), gated at G3
(`claude/README.md` carries no such claim).
AC-12 test: feed the guard synthetic `Write` JSON: `claude/knowledge/python-web-quirks.md` allow;
`skills/python-web-quirks/SKILL.md` ask.

### D2. Units (FR-3) — derived by a coverage probe, ruled per hit

`pwq_kb.py units` prints **every line of the pinned source** matching the union vocabulary: noun (`section|entry|
gotcha|note|item|paragraph|rule|table|probe|half|sibling|companion`) × direction (`above|below|earlier|previous|
preceding|next|following|elsewhere`) within 40 characters, with its enclosing section. A committed ruling table
(data in the script, keyed by (enclosing section heading, text snippet), not by line number: "section above"
occurs in four sections) assigns each printed hit
`referent:<heading prefix>` | `intra` | `external`. **The build fails when any printed hit has no ruling**
(a new reference cannot pass unseen). Each referring section and its referent form one unit. The vocabulary is
a bound, not a proof: `analyst` repeats an independent broader sweep at G1 (F1's lesson).

| Unit | Members | Hits |
|---|---|---|
| u-bt | §2 bounded `BackgroundTasks` · §34 sequential per request | 891 |
| u-tc | §21 `TestClient` teardown · §22 re-raises · §23 version rot | 587, 624, 633 |
| u-st | §24 `app.state` · §25 flip the flag on the live client | 670 |
| u-pyc | §29 same-second `.pyc` · §39 `PYTHONPYCACHEPREFIX` | 951 |
| u-in | §30 `input()` `OSError` · §40 fd 0 is `/dev/null` | 974 |

Result: 42 sections → **36 gotcha units** (5 merged of 11 sections, 31 singletons) = **36 documents**. In the copy
a merged unit's members are adjacent (the copy's `##` order is not the source's).

**Preamble disposition (FR-2 "explicitly dispositioned"): copy only, not ingested.** The copy opens with a
`<!-- preamble -->` block holding the verbatim H1, scope paragraph and version caveat, then a
`<!-- preamble-note -->` block with the corrected statement (the copy is the source of record, the KB a
rebuildable index, new gotchas enter by raw capture then `cobb`; the analyst-inbox lineage kept). The original
"This is a cache, not the source of truth" blockquote is a **waived span**, pinned verbatim in the manifest, so
the waiver fails if the source blockquote ever held more than was waived. The YAML front matter is retired with
the skill (routing metadata; its trigger vocabulary is replaced by the FR-14 pointer). The preamble is read, not
retrieved: it is the first screen of the copy (lines up to the first `<!-- unit` fence), and the D6 listing
includes both preamble fences, so an agent or `cobb` reading the scope or provenance does `Read` on the copy's head. Testable
under AC-1 by the checker's preamble mode (D4 change 3): the verbatim part of the preamble block is located in the
source preamble, and the rest of the source preamble must equal the waived span exactly.

### D3. Stored-document format (FR-2, FR-3, FR-5)

A unit in the copy, and byte-identically its stored `text`, is:

```
<!-- unit u21 -->                       <- fence line, NOT stored
Gotcha: <title of the first member>      <- stored text begins here; = document title
Versions: <lib ver; lib ver | none stated>
Verified: <ISO dates stated as verification, ascending | unverified, migrated 2026-11-09>
Other-dates: <remaining ISO dates as written>        <- only when present
<!-- body -->                           <- exact delimiter line
## <first member heading, verbatim>

<verbatim body>

## <second member heading>   (merged units only)
```

- Merged units add `Members:` (one title per line, copy order) after `Gotcha:` and use indexed
  `Versions[n]:`, `Verified[n]:`, `Other-dates[n]:` (n = 1-based member). 
- **No value is invented (FR-5).** `Verified` = ISO dates in the member's body that sit within 100 characters
  after one of `verified|measured|confirmed|reproduced|re-run|checked|tested`; every other date is listed under
  `Other-dates` (provenance such as "observed 2026-08-26" is not a verification, F12). A member with no
  phrase-qualified date reads `unverified, migrated <D>` (`<D>` = the date recorded once in the copy's
  migration line, rendered at S4). `cobb` rules any disputed date at S4. `Versions` entries are `<lib> <token>`
  pairs; each token is a substring of the member's body with the lib name within 40 characters before it. A
  missed pin is a `cobb` review judgement sampled by `analyst` (weak clause, §7).
- The verbatim comparison starts after the first line equal to `<!-- body -->`; headers are outside it.
- **Preamble grammar (not stored).** The copy opens with the fence line `<!-- preamble -->`, then the verbatim H1,
  scope paragraph and version caveat (no `## ` line; it ends at the next fence); then the fence line
  `<!-- preamble-note -->`, then the corrected statement (free text, no `## ` line, no fence-shaped line); the
  first `<!-- unit ` fence ends the preamble. The waived blockquote appears in neither block.

### D4. `check_content_loss.py` needs six changes; `tdd-engineer` makes them, `cobb` gates

Needed because the stored text is no longer a contiguous substring of one section (header; merged units span
non-adjacent sections), unlisted sections are never looked at, and the workspace is hard-wired. All behind new
flags, existing behaviour byte-unchanged (the existing selftest stays green):

1. `--stored-format pwq`: strip the header at the delimiter (error if absent).
2. Split the stripped body at `^## ` lines into parts; each part is a claim located in the source section whose
   heading equals the part's heading (heading line excluded, as `parse_sections` does).
3. A **preamble pseudo-section** (source text between front matter and first `## `), checked against the copy's
   `<!-- preamble -->` block (D3 grammar; the mode takes `--copy <path>`) rather than the KB (the preamble is not ingested): the block's verbatim text must be a
   contiguous substring of the source preamble, and every remaining non-whitespace span of the source preamble must
   equal a `waivedSpans` entry exactly (a longer or altered source blockquote → reported gap).
4. **Completeness**: the manifest's heading set must equal the source's heading set plus the preamble pseudo-section (which must be checked, not merely listed); an extra
   source heading fails, naming it (AC-1 "an added extra section would fail").
5. `--source-format copy`: build the source side from the copy (strip each unit's header the same way), for AC-5.
6. `--ws <name>` (default `agent-team`, byte-unchanged for old calls) replacing the hard-coded graph key (F3).

Manifest: `claude/cobb/scripts/pwq-kb-manifest.json` — `{pin, migratedOn, units: [{id, title, headings,
documentId, sha256}], waivedSpans}`; the shared `kb-claim-manifest.json` is another migration's ledger and is
untouched. `documentId`s change at every rebuild, so `ingest` rewrites the manifest (AC-5 regenerates it).

### D5. Chunking and how a hit names its gotcha (FR-3, FR-4, FR-5)

Default chunker, no falkor-chat change (out of scope; ML §7). Chunk 0 starts with the header, so the head carries
identity. A **tail chunk carries no title**; the fix is procedural: the convention's content-class section (D6)
tells every consumer to take the distinct `documentId`s of the five chunks and `get_document` the best, which
returns title, versions, date and the whole unit. "A hit names its title" is therefore two calls, and AC-2 is
scored that way (`get_document(documentId).title` equals the unit's `Gotcha:`). Rejected: a per-chunk title line
(alters the verbatim body or the chunker). ML §7 risk 1 is measured by Set B's tail probes; a tail recall below
head recall by more than 0.15 goes back to the stakeholder, it is not patched ad hoc.

### D6. Retrieval procedure and degraded mode (FR-9, FR-11, FR-7)

`skills/agent-kb-retrieval/SKILL.md` gains one section, **"Python/web/test gotchas"**, after "Top-K", leaving the
step list and the fenced prefix literal byte-untouched (AC-14: the section names the class; `audit-team.sh`
passes; `git diff` shows no change inside the fence). Its front-matter consumer list is refreshed to include
`coder`, `tico`, `security-expert` (F13). The section is the only home of:

1. what the class is; the unit is a whole gotcha;
2. normal mode: prefixed query as for any search; take the distinct `documentId`s among the five chunks under
   the floor; `get_document` the best; when none is under the floor, say "no gotcha in the KB matched" and run the
   listing before concluding there is none (an empty result is never the only signal);
3. **the title listing (FR-11), one call, identical in both modes, no KB involved:**
   `grep -n -E '^(<!-- (unit |preamble)|## )' claude/knowledge/python-web-quirks.md` — 42 heading lines + one fence per unit (36) + the two preamble fences, about 6.5 KB; **bound
   ≤ 10,000 bytes** (`| wc -c`), roughly 25 more titles of headroom. The listing's source is the copy itself, so it
   stays in step by construction, and `diverge` (D8) also compares the KB's titles with it;
4. degraded mode: if `search_documents`/`get_document` errors, times out or is unavailable, **say so on the reply's
   first line** ("knowledge base unreachable: <error>"), find the unit with the listing, and read it with `Read`
   (`offset` = its fence line, `limit` = next fence line minus that line); never the whole file (~80 KB).

The guidance is file-backed and is one read from the pointer (the agent's retrieval line names this file).

### D7. Discovery pointer and tool grants (FR-13, FR-14, AC-13)

Fixed literal (`grep -F`): **`Python/web/test gotchas`**. Sentence, appended to the existing retrieval quote line
or added as one new line:

> The knowledge base also holds Python/web/test gotchas (asyncio, FastAPI/Starlette, pytest, urllib): query it by
> situation before debugging a surprising behaviour, via the convention in `skills/agent-kb-retrieval/SKILL.md`.

Eight consumers: `coder` (new line), `tdd-engineer`, `architect`, `analyst`, `qa-engineer`, `frontend-engineer`
(extend the existing line), `tico`, `security-expert` (new line **and** `mcp__falkor-chat-agent-team__search_documents`,
`mcp__falkor-chat-agent-team__get_document` added to `tools:`). Edits are `cobb`'s (S15b1). Check 12 in `audit-team.sh`
(S15b2) greps the literal in the eight `claude/<a>/<a>.md` and the two tool names on the two enumerated `tools:`
lines; its mutant runs in a scratch copy of `claude/` + `skills/agent-kb-retrieval/` (the copy runs its own script
copy, so `ROOT` follows) with the literal removed from one agent and must turn red.

### D8. Tool target guard; no agent-reachable production delete; add / edit / retire path (FR-10, AC-7)

Every network-touching subcommand takes `--base-url` and `--ws`, and the pair must match a **committed target
table** `{agent-team: http://localhost:8200 (production), pwq-test: http://localhost:8210 (test)}`; any other
pair, including `--ws pwq-test --base-url :8200`, refuses (the REST client cannot ask the server which workspace
it serves, so the check keys on the URL actually written to, F2). Production `ingest` additionally needs
`--confirm-production`. **`replace` and `retire` exist only for the test target and refuse production with no
flag**; for production they print the exact `curl -X DELETE <base>/documents/<id>` line for the stakeholder.
Consequence, accepted: in steady state an edit or retire in production is edit-the-copy → `diverge` names the
stale/orphan unit → stakeholder runs the printed `curl` → `cobb` runs `ingest` for the new text. Add = add a
unit with a fresh, never-reused id, then `ingest`. `diverge` (read-only, `falkordb` client via `cypher-mcp/.venv`)
renders each unit's stored text from the copy, hashes it, and compares with the current documents (registry:
the manifest); it reports, naming unit id and title: *missing*, *stale* (hash differs), *duplicate* (two current
documents with the unit's title), *orphan* (manifest document still current, unit gone from the copy), *title
drift*. Exit 1 on any. `ingest` idempotency: unit current with equal hash → skip; no manifest entry but an
identical current document exists whose status is not `failed` → adopt its id (never re-ingest: auto-supersede);
a unit whose manifest entry or title already has a current document with *different* text → refuse, printing the
`curl -X DELETE` line (the stakeholder deletes first); else ingest. AC-7's
add/edit/retire exercise runs only in `ws:pwq-test`.

### D9. The script: `claude/cobb/scripts/pwq_kb.py` (stdlib; `falkordb` from `cypher-mcp/.venv` for reads)

Subcommands: `units` (D2; also prints header sizes and the chunk table), `render-copy`, `verify-copy` (**recomputes
every header** — `Gotcha`, `Versions`, `Verified`, `Other-dates`, member order — and compares with the copy's, and compares
bodies with the pinned source), `ingest`, `gate`, `diverge`, `replace`/`retire` (test only), `eval`. `eval`
modes: `--set A [--before]`, `--set B --probe` (prints rank-1 source and score only; computes no recall; writes
no sentinel), `--set B --sealed-once` (the scoring), `--ac2`, `--sweep`, `--baseline bm25`; the floor is an explicit
`--floor` argument recorded in every report. Set B's path comes only from `--set-b-path` or `PWQ_SET_B`, never from
the source. The prefix is read from the retrieval skill's fenced literal by string extraction (not `eval`).

**Embedding hard gate (K-102).** After ingest, `gate` polls per unit `MATCH (d:Document {documentId:$id})
-[:HAS_CHUNK]->(c:Chunk) RETURN count(c), count(c.embedding), d.status` (names verified to run on the live graph,
`repository.py:1080,1150-1186`) until all chunks are embedded or a timeout (default 10 min); prints a per-unit table
(unit, chunks, embedded: the FR-4 report), exits non-zero on any unembedded chunk or `failed` status. The modes that score the feature
documents (`--ac2`, `--sweep`, `--set A` without `--before`, `--set B --sealed-once`, `--baseline bm25`) re-run
the gate query first and refuse when it fails; `--set B --probe` and `--set A --before` run before any ingest
(S8, S9) and are exempt. In `ws:pwq-test` a failed unit is replaced by the script;
in production the run stops, the stakeholder deletes the failed document by `curl`, and the idempotent `ingest`
completes it. `--sealed-once` takes a mandatory `--expect-sha256`, set by the S14 brief from the sha256 that S8 recorded in the ledger. `gate` and the scoring modes take the expected unit ids from `--copy` and exit 2 without a readable copy; `diverge` reads the manifest's `sha256`, `pin` and `migratedOn`.

### D10. The AC-9 sweep: scope and per-hit ruling

Unfiltered: `grep -rIl python-web-quirks` over the repo excluding only `.git`, `.venv`, `node_modules` (65 files
at plan time, including this feature's own documents). `analyst` re-runs it at G3 and fails on any hit missing
from the table (derived, not transcribed).

| Hit | Ruling | Step |
|---|---|---|
| `AGENTS.md:69` | re-point: drop the skill from the `skills/` list; one clause: Python/web gotchas are in the `ws:agent-team` KB, copy `claude/knowledge/python-web-quirks.md` | S16a |
| `falkor-chat/docs/SERVER.md:638` | re-point to "the `python-web-quirks` KB entry (copy `claude/knowledge/python-web-quirks.md`)" | S16a |
| `falkor-chat/server/falkorchat/app.py:220,452`, `storefront_api.py:246`, `server/tests/test_storefront_api.py:1344` | comment/docstring-only re-point to the copy; `py_compile` + offline suite subset stays green | S16a |
| `falkor-chat/server/tests/test_fusion.py:167` | **no change**: an entity-name fuzzy-match fixture, not a citation | S16a records |
| `falkor-chat/docs/plans/workflow-timers.md:207` | re-point (cites the file path); lines 640, 739, 1081 name a finding, not the file: no change | S16a |
| `claude/README.md` (rows at lines 9, 10, 13, 15, 16) | re-point each to the D7 pointer wording | S16b |
| `skills/README.md:24` | catalog row removed (AC-4) | S16b |
| `skills/agent-standards/claude-code.md:165` | cites the file as an example cobb-remit write: re-point to `claude/knowledge/python-web-quirks.md` | S16b |
| `claude/cobb/kaizen/plan.md` K-026 (lines 25, 298-314) | the item's own trigger ("a third `TestClient` section needs a parent") already fired (`grep -c '^## .*TestClient'` returns 4 on the pin); close it with a pointer to this migration (unit `u-tc`) instead of rewriting. K-029 (`:324-336`, open item about the skill's listing budget): rewrite to say the skill no longer exists, or close | S16b |
| `claude/cobb/hooks/guard-cobb-topic-writes.sh` | entry retired (D1) | S4 |
| `cpg/eval/kit-manifest.json` | **no change** (FR-15). Check: `export_kit.py --repo . --eval-commit cf7b2042cce7bad5d0ab7876940bcea00343d914 --out-dir <scratch> --manifest <scratch>/kit-manifest.json` (never the committed path: the tool writes it); the scratch manifest's `files` array must equal the committed one (`git archive` reads the commit, not the tree). Not yet run by anyone; blocked ≠ waived | S16c |
| `docs/requirements/cpg-agent-adoption.md:37` (archived), `docs/plans/cpg-agent-adoption.md`, `docs/plans/connected-code-graph-p0*.md`, `cpg/spikes/frontend-fitness/results/*.jsonl`, `claude/cobb/scripts/kb-claim-manifest.json` | no change: historical text or another migration's ledger; one line recorded | G3 |
| every `kaizen/history.md`, `docs/reviews/*`, `*-coordination.md`, `claude/docs/{plans,reviews}/*`, `falkor-chat/docs/reviews/*`, `opencode/docs/plans/*`, `docs/plans/document-size-discipline.md`, this feature's own documents | frozen or own: no change; one line recorded | G3 |

### D11. Evaluation design (FR-12, AC-3)

**Sizes and scoring over the units.** Both sets are sized per *section*, so they do not change: Set A 42+14, Set B
84+30 (15 off-domain, 15 near-domain); gate figures as approved (ML §1, not restated). A positive query for section s
hits iff the document of the unit containing s is among the five chunks under the floor (a hit on a merged unit
counts for each member). Coverage: each of the 42 sections has a hit among its two queries. A negative is a false
accept iff any of the 36 unit documents is under the floor. Replaced ML figures:
documents 42 → 36; chunks 108 → 129; ≥ 4-chunk units 7 → 12 (listed in ML §5 rule 4: Set B's tail-probe list; `units --chunks` prints the counts); confusable sets:
{§21-§25} becomes u-tc/u-st, {§2,§34}, {§29,§39}, {§30,§40} become intra-unit; the rest unchanged. `data-scientist`
rewrites the affected ML §2/§3/§7 numbers in place (S12).

**Authors and storage.** Set A: `data-scientist` (ML §3), committed as `claude/cobb/scripts/fixtures/
pwq-golden-set-a.json` (S5). Set B: `qa-engineer` (S8) from the pinned source and `units` output; stored at
`${XDG_DATA_HOME:-$HOME/.local/share}/graphmind-ai-lab/sealed/pwq-set-b.json`, outside the repo; only its sha256
is written to the ledger. **The seal is procedural, not mechanical**: nothing stops an agent with Bash from
reading that path. Protection is the timeline (no implementer runs between B's freeze and its scoring except
`data-scientist` S12 and `cobb` S13/S15, whose briefs never name the path), the path living outside the repo and
the tool source, and the `--sealed-once` sentinel against a second scoring. Only S8 and S14 briefs name the path.

**Method rulings before B exists (S7b, `data-scientist`).** (i) The probe may rewrite a B query only when its rank-1
score is < 0.25 (a leak sign); one that already surfaces from another document is **kept and recorded** as a
duplicate-coverage finding (ML §5). (ii) Captures that restate a gotcha are removed from the scored state by
deletion at H1 and by `--exclude` for later ones; the headline is the excluded run, the unfiltered numbers print
beside it, and S12 sweeps with the S7 list excluded (ML §4.2). S12 and S14 pass the S7 capture ids (from
`docs/plans/python-web-quirks-kb-captures.md`) as `--exclude`.

**Test workspace.** `ws:pwq-test` on a second `start_agent_team.sh` process (`FALKORCHAT_WS_ID=pwq-test
AGENT_TEAM_PORT=8210`) holding a copy of the current corpus plus the units. **Order (S6a, S6b):** a tiny throwaway
probe first decides whether `GRAPH.COPY` carries the `Chunk.embedding` vector index and the embeddings; if yes,
`GRAPH.COPY ws:agent-team ws:pwq-test` **before** starting the server (the script creates the workspace, and `COPY`
errors on an existing destination; its bootstrap/seed steps are idempotent), else re-ingest the corpus's documents
by REST (the primary path in that case). Verify with `CALL db.indexes()` and a known-answer search against `:8210`.
The copy inherits production's unembedded chunks (16 at plan time, K-102): S6b records them and `eval` reports
recall with and without them. Lifecycle: `graph-dba` starts the process detached with its log under
`~/.local/state/graphmind-ai-lab/`, `teco` keeps it alive (restart if down) from S6b to S17 (`teco`'s
`ensure-environment.sh` covers only `:8200`); S17's `GRAPH.DELETE` will ask, expected.

**Order so Set B is scored once and last.** (1) B authored (S8), leak-checked, `--probe`-run before ingestion,
frozen (sha256 recorded). (2) All tuning on A in `ws:pwq-test`: before-ingestion baseline (S9), ingest and AC
checks (S10, S11), floor sweep (S12). (3) Production ingest (S13), then the retrieval-skill edit incl. any floor
change (S15a): **a floor change lands in S15a before S14**, after S12's Stage 8 45-pair regression. (4) Capture
re-derivation (S13b) and H1 (stakeholder deletion). (5) B scored once in production (S14) with `--floor` explicit.
A failed B means B2 by a different author, never a re-run (ML §5).

**Raw captures and duplicates.** S7 (`cobb`; reads the workspace, writes the kaizen item) lists every `ws:agent-team` document whose text mentions
`python-web-quirks` or restates a gotcha (ids, titles, coverage by unit) and gives capture `38cd49cd…`'s gap
statement a promotion ruling and records the promotion text in that kaizen item before H1. Because every agent writes captures at the end of every run, the list goes stale:
**S13b re-derives it immediately before H1 and again at S14's start**. H1: the stakeholder deletes the listed
captures by `curl` (including `38cd49cd…`: S7 first records its promotion text, as a new item in `claude/cobb/kaizen/plan.md`, the team's backlog for `cobb`-owned follow-ups; no agent deletes it). **Scoring rule at S14:** any capture that appeared after the last re-derivation (S13b) is
excluded by id: the harness requests a fixed window of 50 chunks (one excluded document can hold several of the top
chunks; the API allows 200), drops chunks of excluded documents, truncates to five and scores them; the report
states that the convention's `limit=5` is deliberately exceeded for this measurement, lists the excluded ids and
also gives the unfiltered numbers. Before-ingestion baseline
(ML §4.3): A against `ws:pwq-test` before the units go in; lexical BM25 over the feature documents on A (S12) and on
B (S14). Reported: raw top-5 and floor-applied recall, MRR, mean distinct documents among five chunks,
tail-probe vs head-probe recall.

## 4. Steps

"Shared key" = writes to `ws:agent-team`; only S13. Tests run with the named component as working directory
(root `AGENTS.md`); no step runs a tree-mutating git command; `teco` commits after its own verification.
S1-S3b are test-first (`tdd-engineer`).

| Step | Owner | What | Done when | Shared key |
|---|---|---|---|---|
| G0 | `analyst` | re-review this revision (`docs/reviews/python-web-quirks-kb-plan.md` next Pass) | not "needs changes" | no |
| S1 | `tdd-engineer` | `claude/cobb/scripts/pwq_kb.py`: `parse_source`, `find_hits` + ruling table, `build_units`, `render_header`, `render-copy --pin`, `verify-copy --pin`, `units [--chunks]`; `pwq_kb_selftest.py` | T1-T9 | no |
| S2 | `tdd-engineer` | `check_content_loss.py` changes 1-6 (D4), new fixtures, selftest rows | T10-T15 + old selftest green | no |
| S3a | `tdd-engineer` | `pwq_kb.py`: target table + guard, `ingest`, `gate`, `diverge`, `replace`/`retire` (test only), manifest I/O; fake REST server built from `schemas.IngestDocumentIn` (`producedBy`, `sourceFormat`) and fake Cypher fetcher | T16-T23 | no |
| S3b | `tdd-engineer` | `pwq_kb.py eval`: all modes of D9, recall/coverage/MRR/distinct-docs/BM25, exclusion rule, sentinel | T24-T30 | no |
| G1 | `analyst` | review S1-S3b + mutation pass + independent broader cross-reference sweep (`docs/reviews/python-web-quirks-kb-impl.md` Pass 1) | not "needs changes" | no |
| S4 | `cobb` | `render-copy`, review every header, write `claude/knowledge/python-web-quirks.md`; `verify-copy`; retire the guard entry; AC-12 test. | one change; guard allows `claude/knowledge/python-web-quirks.md`, asks for the old skill path | no |
| S5 | `coder` | `pwq-golden-set-a.json` from ML §3 (section → unit via `units`); `pwq_leakcheck.py` (no shared 4-gram between a query and its section/title) | 42+14 rows; planted 4-gram caught | no |
| S6a | `graph-dba` | throwaway probe: does `GRAPH.COPY` carry vector index + embeddings; delete the probe key | verdict recorded | scratch key |
| S6b | `graph-dba` | create `ws:pwq-test` (D11 order), start `:8210` detached, record unembedded chunks | known-answer search works; counts recorded | reads only |
| S7 | `cobb` | derive the raw-capture list + coverage + gap ruling → ledger; write the promotion item for `38cd49cd…` into `claude/cobb/kaizen/plan.md` | list with ids; ruling | reads only |
| S7b | `data-scientist` | rulings (i), (ii) of D11 | in ML note | no |
| S8 | `qa-engineer` | author B (sealed path), `pwq_leakcheck.py`, `--probe` in `ws:pwq-test`, freeze, sha256 to ledger; `docs/test-plans/python-web-quirks-kb.md` | 84+30 rows, a `tail`-flagged positive for each of the 12 units with ≥ 4 chunks (S8's brief carries it; `validate_set_b` does not read the flags) | no |
| G2 | `analyst` | review S4-S8: copy vs pin sample, `Versions`/`Verified` sample, guard run, B outside repo | Pass 2 | no |
| S9 | `data-scientist` | before-ingestion baseline of A in `ws:pwq-test` | recorded in ML note | test ws |
| S10 | `cobb` | `ingest` + `gate` into `ws:pwq-test`. | gate table clean | test ws |
| S11 | `qa-engineer` | AC-2, AC-5, AC-7, AC-8, AC-12 in `ws:pwq-test` (T31-T38); report `docs/test-reports/python-web-quirks-kb-report.md` | all pass; planted divergence and 43rd title shown | test ws |
| S12 | `data-scientist` | floor sweep 0.38-0.50 with S7 ids `--exclude`d + BM25 on A (ML §6); rewrite ML counts in place; floor change ⇒ Stage 8 regression first | decision in ML note | test ws |
| S13 | `cobb` | **production ingest**: `ingest --ws agent-team --base-url http://localhost:8200 --confirm-production`, `gate`, AC-1 check at the pin, manifest | gate clean; AC-1 clean; extra-section mutant red | **yes** |
| S15a | `cobb` | D6 section, front-matter consumer list, floor value if S12 changed it; `audit-team.sh` | AC-14 | no |
| S13b | `cobb` | re-derive the capture list; hand to the stakeholder | list in ledger | reads only |
| H1 | stakeholder | `curl -X DELETE` the listed captures  | ids in ledger | stakeholder |
| S14 | `qa-engineer` | re-derive exclusions (S7 ids as `--exclude`), score B once (`--sealed-once`), BM25 on B, tail-vs-head; report | AC-3 verdict | reads only |
| S15b1 | `cobb` | pointer + tool grants in the eight agents (D7) | `grep -F` finds the literal in 8 | no |
| S15b2 | `cobb` | `audit-team.sh` check 12 + scratch-copy mutant | green; mutant red | no |
| S15c | `qa-engineer` | AC-6 (T39), AC-10, AC-13 black-box | pass | no |
| S16a | `coder` | D10 rows marked S16a | `py_compile` + offline suite subset green | no |
| S16b | `cobb` | D10 rows marked S16b; one-line `claude/knowledge/` exception to the layout claim in `claude/AGENTS.md:5` and root `AGENTS.md:44` (root file via `coder` in S16a if `cobb`'s guard asks) | rows done | no |
| S16c | `coder` | kit re-export check (D10) | files array equal, or reported blocked | no |
| S16d | `cobb` | first add a `--pin-from-git` source mode to the checker (via `coder`), so AC-1 re-runs once the source is gone; then remove `skills/python-web-quirks/` with filesystem `rm -r` | AC-4 | no |
| G3 | `analyst` | impl review Pass 2: unfiltered AC-9 sweep vs D10, AC-1/5 evidence, live-spec check (no component live-spec contract changes; SERVER.md:638 is a citation) | not "needs changes" | no |
| S17 | `graph-dba` | stop `:8210`, `GRAPH.DELETE ws:pwq-test` (asks; expected) | key gone | test ws |

Ordering facts: S1 → S3a → S3b serial (one file); S2 is independent and may run beside S1; S4 precedes S10 (units
come from the copy); S5/S6 are independent of S4; S12 precedes S13 so a floor change is known before production;
S15a precedes S14 so the convention scored is the convention shipped; the skill stays loadable until S16d.
`teco` records the `HISTORY.md` home at close (`claude/docs/` has none).

## 5. Test strategy

Altitudes: unit → contract (checker fixtures, fake REST) → live in `ws:pwq-test` → read-only acceptance on
production. Every row names a mutant; G1 rejects a row with none. Expected values come from the pin or the
fixtures, never from the code under test.

| T | Check | Mutant (must turn red) |
|---|---|---|
| T1 | pin parses: 42 headings (`grep -c '^## '` on the pin), preamble | drop one heading → count mismatch |
| T2 | every printed hit has a ruling | a scratch section "the bounded-pool entry above" with no ruling fails the build; the real pin passes |
| T3 | units equal D2's table (5 units, 11 members) | remove the ruling for 891 → u-bt disappears and T2 fails |
| T4 | header grammar: single and merged; preamble block + note fences end at the next fence | reorder members → `verify-copy` fails |
| T5 | `Verified`/`Other-dates` rule: phrase-qualified dates only; `verify-copy` recomputes headers | put §16's provenance date under `Verified` → fails |
| T6 | each `Versions` token is a body substring with lib within 40 chars | invent a token → fails |
| T7 | delimiter once per unit, never inside a body | a body containing the delimiter → fails |
| T8 | `verify-copy` bodies equal the pin | change one body character → fails naming the unit |
| T9 | listing bound ≤ 10,000 bytes (`grep` over the copy) | scratch copy with 25 extra 200-char titles, listed through the D6 regex → fails |
| T10 | header stripped, body located | header without delimiter → error |
| T11 | merged unit's non-adjacent members both located | swap a member heading → not found |
| T12 | AC-1 preamble: the copy's `<!-- preamble -->` text is a substring of the source preamble and the rest equals the waived span; unchanged source passes | widen the source blockquote, or alter one preamble word in the copy → reported |
| T13 | extra source heading fails by name | plant `## Extra` in a scratch source |
| T14 | `--source-format copy` clean on the copy | corrupt one copy body char → fails |
| T15 | `--ws` non-default reads that workspace; old invocations unchanged (existing selftest) | ignore `--ws` → fake returns missing |
| T16 | `ingest` sends `producedBy=cobb`, text equals stored text | send `produced_by` → fake rejects |
| T17 | re-run skips | remove skip → second POST seen |
| T18 | identical current document adopted, no POST; a `failed` identical document is not adopted; a current document with different text under the unit's title → `ingest` refuses and prints the `curl` | adopt path re-ingests, adopts `failed`, or posts over a different-text document → red |
| T19 | target table: `--ws pwq-test --base-url :8200` refuses; production without `--confirm-production` refuses | drop the URL check → red |
| T20 | `replace`/`retire` refuse production with and without `--confirm-production`, and print the `curl` | allow with the flag → red |
| T21 | `gate` fails on one null embedding and on `failed` status; `eval --set B --probe` and `--set A --before` run with no gate passed, `--ac2` refuses | gate check applied to `--probe` → red |
| T22 | `diverge` names missing, stale, duplicate, orphan, title drift (one fake each) | disable one rule → that case green→red |
| T23 | a unit whose heading form (`###`, no fence) hides from the listing is caught by `diverge` | `diverge` compares only fenced units → red |
| T24 | `eval` recall, floor-applied recall, MRR, distinct docs | wrong floor comparison (`<` vs `>`) → red |
| T25 | merged-unit coverage counts each member | counts first member only → red |
| T26 | false accept counts only feature documents | counts any document → red |
| T27 | prefix read from the fenced literal | alter the literal → harness aborts |
| T28 | `--set B --probe` writes no sentinel and prints no recall; `--sealed-once` writes it; a second run refuses | probe writes sentinel → red |
| T29 | exclusion rule: excluded ids dropped from a 50-chunk window, first five of the rest scored; an excluded document holding 3 of the top chunks still leaves five scored | `limit = 5 + |excluded|` or exclusion after truncation → fewer than five scored → red |
| T30 | BM25 deterministic | unseeded tie-break → red |
| T31 | AC-2 per unit: title, Set-A situation and last-paragraph queries each return the unit; `get_document` title = `Gotcha:`; zero unembedded | tail probe on a unit whose last chunk is deleted in the test graph → red |
| T32 | AC-8: `Verified: <date>` / `unverified, migrated <D>` per the derived counts; versions shown | a copy header with `Verified` swapped to `unverified` on a dated unit → count mismatch |
| T33 | listing (the D6 regex) shows every pin heading, one fence per unit and both preamble fences, ≤ 10,000 bytes (expected: pin count, not plan count) | the regex without `preamble` → the two preamble fences missing from the count → red |
| T34 | 43rd unit: shown in the listing, and `diverge` fails until ingested | as T23 |
| T35 | AC-5: remove the units from the test workspace, rebuild from the copy, regenerate the manifest, checker `--source-format copy --ws pwq-test` clean, T31 again | corrupt the copy → red |
| T36 | AC-7: add one, edit one, retire one (test ws; retire also removes the unit from the copy, else the next `gate` fails `ABSENT`); T31 for added/edited; retired not returned; `diverge` clean | skip the ingest of the edit → `diverge` stale |
| T37 | planted divergence → `diverge` exit 1 naming the unit | none planted → exit 0 (control) |
| T38 | AC-12 guard: `claude/knowledge/python-web-quirks.md` allow; `skills/python-web-quirks/SKILL.md`, `docs/BACKLOG.md` ask | re-add the entry → old path allowed |
| T39 | AC-6 (after S15b1): a tool-denied session of a consuming agent, and a throwaway MCP config pointing at a dead port (never stop `:8200`): reply starts "knowledge base unreachable", reads one unit with `Read` offset/limit, no whole-file read | remove D6 item 4 → the session reads the whole file or does not report |

### AC / FR map

| Requirement | Steps | Evidence |
|---|---|---|
| FR-1, FR-12, AC-3 | S5, S7b, S8, S9, S12, S14 | S14 report |
| FR-2, AC-1 | S1, S2, S4, S13 | T1-T15 (T12 = the preamble disposition); S13 checker at the pin |
| FR-3, AC-2 | S1, S10, S11 | T4, T31; D5 |
| FR-4 | S3a, S10, S13 | T21; per-unit gate table |
| FR-5, AC-8 | S1, S11 | T5, T6, T32-T34 |
| FR-6, AC-4 | S16b, S16d | directory gone; catalog row gone |
| FR-7, AC-14 | S15a | fence `git diff`; audit check 11 |
| FR-8, AC-5 | S4, S11 | T8, T35 |
| FR-9, AC-6 | S15a (guidance), S15c (test) | T39 |
| FR-10, AC-7, AC-11, AC-12 | S3a, S4, S7, S11, S13b, H1 | T16-T23, T36-T38; ledger |
| FR-11 | D6, S11 | T33, T34 |
| FR-13, AC-10 | S15b1, S15c | `tools:` lines |
| FR-14, AC-13 | S15b1, S15b2, S15c | check 12 + mutant |
| FR-15, AC-9 | S4, S16a-c, G3 | D10 table |

Deliverables other steps cite and who owns them: `skills/agent-kb-retrieval/SKILL.md` (S15a); the copy path in
`AGENTS.md`/`SERVER.md`/comments (S16a), `claude/README.md`/`skills/README.md`/`claude-code.md` (S16b); the pointer
literal and check 12 (S15b1/b2); `docs/test-plans/python-web-quirks-kb.md` and the report (S8, S11, S14);
`pwq-kb-manifest.json` (S13).

## 6. Risks

**Risks.**
1. `Versions` completeness and the `Verified` phrase rule are the weak FR-5 clauses; `cobb` rules, `analyst` samples (§D3).
2. Tail chunks carry no title; handled procedurally (D5); if Set B tail recall trails head by more than 0.15, a stakeholder call.
3. The 0.43 floor is unproven for this content (ML §6); a change is for all users and needs the Stage 8 regression first.
4. K-102 in production: the gate stops the run; recovery needs the stakeholder's `curl` (D9). The same constraint makes every
   production edit/retire stakeholder-gated (D8), an accepted narrowing of the strategy plan's cobb delete-then-ingest.
5. Auto-supersede on re-ingest silently demotes a document; mitigated by adopt-not-reingest and `diverge`'s duplicate check.
6. `GRAPH.COPY` carrying vector indexes is unverified (S6a decides; re-ingest fallback is slow but mechanical).
7. Write guards: `cobb` gets `ask` for `skills/agent-kb-retrieval/SKILL.md`, `claude/scripts/audit-team.sh`, the guard script and the `.json` manifest;
   `coder` editing `falkor-chat/docs/plans/workflow-timers.md` gets `ask`; `teco` expects these.
8. The kit re-export (S16c) has not been run; blocked is reported, not waived.
9. Learning-capture writes by agents continue throughout and can create new restating documents between S13b and S14 (rule: re-derive at S14's start).

## Ready to implement

Plan: `docs/plans/python-web-quirks-kb.md`. Sequence G0 → S1-S3b → G1 → S4 …; only S13 writes `ws:agent-team`.
