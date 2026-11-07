# ws:agent-team u1-attributed documents — attribution RCA

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-063

CPG: not applicable — a data-attribution investigation with no code-level component.

## Symptom
In `ws:agent-team`, the Documents with an `INGESTED_BY` edge to `(:User {userId:'u1'})` rather than to an `:Agent` were ingested without `produced_by`.

## Evidence (all observed)
- Graph `ws:agent-team`: 476 Documents. 472 have `INGESTED_BY` to an `:Agent` (cobb 343, analyst 37, teco 23, architect 22, devops 13, qa-engineer 9, data-scientist 8, tdd-engineer 6, coder 6, tico 4, graph-dba 1). 4 have `INGESTED_BY` to `User u1`.
- Those 4 (the only ones with `sourceKind:'document'`; the other 472 are `'agent'`, which `falkor-chat/server/falkorchat/repository.py:1859-1892` derives from the ingestor label):

| id | createdAt (UTC) | title |
|---|---|---|
| 12eb7b6c4d964b3b90c8f1aa12b29d5d | 2026-09-20 13:15:27 | falkor-chat: full pytest suite is nondeterministically flaky under live concurrent sessions |
| e34f31430df0489ab5709e8213255bb9 | 2026-09-20 19:50:11 | Piping a Bash command through `\| tail` ... discards its own exit code from `$?` |
| b7154a94b30848c8992c7914ce45fddd | 2026-10-17 15:21:28 | cypher MCP tool's write scope is kaizen-graph-shaped only, not generic per-graph writes |
| 6ce4a2321bcb44989d382e2c762edd0f | 2026-10-17 15:21:51 | Shadowing a setuptools-editable-installed package for isolated mutation testing needs PYTHONPATH + a non-package cwd |

  All four bodies are in the agent `Fact:/Evidence:` shape, so they are agent raw-capture writes that landed under the default actor.
- Transcripts under `~/.claude/projects/-home-mauricio-prg-graphmind-ai-lab/` (67 entries, oldest file dated 2026-10-14): a parse of every `tool_use` named `*ingest_document` found 250 calls, and **all 250 carry `produced_by`**; none omit it. None of the 4 titles appears as an `ingest_document` input in any transcript. Their text appears only in `Bash` tool_use inputs (3-5 each), which I read as cobb's distillation reads and did not examine further.
- The two 2026-09-20 documents predate the oldest retained transcript, so they cannot be correlated at all.
- The two 2026-10-17 documents fall inside the retained window (about 15:21 UTC, 23 s apart), yet no transcript has a matching ingest call. The only agent-team ingest I found near then (15:25:46, session cc651677) was a different title.
- Prompts (`claude/*/*.md`): the 13 role prompts that mention `ingest_document` (devops, qa-engineer, coder, frontend-engineer, analyst, security-expert, architect, tico, cobb, data-scientist, graph-dba, teco, tdd-engineer) each contain at least one `produced_by` match. Two files mention `ingest_document` with zero `produced_by`: `claude/devops/ops-quirks.md` (probe-script discussion, not a capture instruction) and `claude/teco/coordination-techniques.md:224` (a pointer to the capture rule). Neither is a capture example that omits it. I did not read every prompt's example line (only coder, graph-dba, frontend-engineer, in full).

## Cause
**Confirmed:** the 4 documents were ingested without `produced_by`, so the server fell back to the configured actor `u1`.

**Not determined:** which client did it. The evidence rules out the current agent prompts and any ingest call visible in the retained transcripts, and leaves two candidates: (a) a path that bypasses MCP `produced_by` — a script, a REST/other-client call, or the earlier KB migration — and (b) agent transcripts that are no longer retained (certain for September). I cannot distinguish them from the repo and graph alone.

## Missing to settle it
- `ws:agent-team` server access log around 2026-10-17 15:21 UTC (`~/.local/state/graphmind-ai-lab/agent-team-8200.log` exists but retention for that date is unverified; I did not read it).
- Whether a pre-`produced_by` era of the tool, or a migration script, wrote the September documents.

## Suggested fix and owner
- Re-attribute the 4 documents, owner `cobb` (graph write, curator rights), only after the user confirms who wrote them. Do not guess an author.
- Guard, owner `cobb` with `architect`/`coder` for any server change: either have capture fail loudly when `produced_by` is missing for a workspace that expects agent attribution, or have cobb's distillation sweep report any `sourceKind:'document'` entry (query: `MATCH (d:Document)-[:INGESTED_BY]->(:User) RETURN d`). The sweep check is the cheap one.
- No prompt edit is indicated by this evidence.
