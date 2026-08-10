# Kaizen — Learnings Inbox: teco

> Append-only capture of durable, non-obvious environment facts the `teco` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

## 2026-08-10 — A `grep --include='*.ext'` blast-radius scan structurally cannot find extensionless dotfiles like `.env.example`, and both `teco` and `architect` missed the same site because of it

- **Evidence:** decomposing falkor-chat K-042 (env-var replacement, FR-20), I swept for affected
  sites with `grep -rln 'FALKORCHAT_LLM_\|FALKORCHAT_EMBEDDING_' --include='*.sh' --include='*.md'
  --include='*.py' --include='*.yml' --include='*.json'` and recorded the result as the
  authoritative blast radius in the coordination doc. `architect` independently reached the same
  list and wrote "no such site exists" in its plan §2.9. The `analyst` gate then found
  `falkor-chat/server/.env.example:19-21,30-31` setting all four variables — confirmed by me with
  `grep -n 'FALKORCHAT_' server/.env.example`. `.env.example` matches none of the five `--include`
  globs, so neither sweep could ever have found it. Consequence had it shipped: the repo's own
  example env file becomes a startup brick against the new fail-loud behaviour (AC-13).
- **Context:** the documentation-impact / blast-radius scan at decomposition for a feature whose
  core requirement is "these env vars are replaced, not deprecated".
- **Suggested home:** prompt — the documentation-impact scan step. Candidate rule: for a
  *rename/removal* blast radius, run the sweep **unfiltered** (`grep -rn <token> .` minus
  `.git`/`.venv`/`node_modules`) rather than extension-filtered; config lives in dotfiles
  (`.env*`, `.envrc`, `Dockerfile`, `Makefile`, CI YAML with no extension) that extension globs
  silently exclude. Generalisation worth verifying: a scan whose *purpose* is proving a negative
  ("no other site sets this") must never be filtered by filename shape.

## 2026-08-10 — Correcting an in-flight subagent's premise via `SendMessage` beat discarding its work; two agents independently reached the same finding

- **Evidence:** `architect` (U1) discovered mid-run that `TraceEvent`s are debug-only
  (`falkor-chat/server/falkorchat/executor.py:390` — `tracer = self._tracer if run["trace"] else
  _NULL_TRACER`), which invalidated the premise of `graph-dba`'s (U2) brief, dispatched in the same
  turn and still running. I sent U2 the finding with my own verification rather than letting it
  deliver and re-briefing after. U2 reported it had reached the same conclusion independently
  *before* the message arrived and had already rejected `TraceEvent`s — so the correction cost
  nothing and confirmed the finding by two independent paths.
- **Context:** parallel `architect` + `graph-dba` dispatch on one feature, where the second agent's
  brief encoded an assumption the first agent disproved mid-flight.
- **Suggested home:** prompt — the "Track what's in flight" step. Two candidate rules: (1) when a
  delivered unit invalidates a premise in a *still-running* sibling's brief, `SendMessage` the
  correction immediately rather than waiting to re-brief on delivery; (2) parallel dispatch is
  cheap insurance for a genuinely uncertain premise, because independent agreement is stronger
  evidence than either agent alone.

## 2026-08-10 — A specialist's own knowledge base can be stale in a build-version-specific way, and the agent re-probing instead of trusting it prevented a wrong design

- **Evidence:** `claude/graph-dba/falkordb-quirks.md` asserted `db.indexes()` does not expose a
  vector index's dimension, recorded against the edge build (module 999999). `graph-dba` re-probed
  on the pinned v4.18.11 (module 41811) and found it false; I verified independently with
  `GRAPH.RO_QUERY ws:acme "CALL db.indexes() YIELD label, types, options ..."`, which returns
  `Chunk.embedding → {dimension: 1024, similarityFunction: cosine, M: 16, ...}`. Had the stale
  entry been trusted, K-042's FR-19 dimension guard would have been designed onto parsing an error
  message from a deliberately mismatched query vector.
- **Context:** graph design note for an embedding-dimension guard, where the knowledge base
  directly contradicted the live instance.
- **Suggested home:** knowledge base / prompt — worth a general note that entries in a pinned-build
  knowledge base carry an implicit build version, and a *negative* claim ("X is not exposed")
  deserves re-probing before a design is built on it. Possibly a convention of stamping each
  quirks entry with the module version it was verified against; graph-dba's file now does this
  for the corrected entry.
