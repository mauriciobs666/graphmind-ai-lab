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

## 2026-08-10 — A single "Landing 1" implementation unit spanning 6 sequenced plan steps and ~10 files ran past 370k subagent tokens and was still writing files

- **Evidence:** dispatched falkor-chat K-042's Landing 1 (plan `docs/plans/llm-provider-config.md`
  §6, steps L1-1..L1-6: new `transport.py`, new `modelconfig.py`, generalizing two client classes,
  rewiring five consumer bindings across `executor.py`/`guards.py`/`responder.py`/`embedding.py`/
  `tools.py`/`app.py`, an env-var cutover across 6 more files, plus docs) to one `coder` agent as
  a single unit, reasoning that the plan's own steps were tightly sequenced/file-coupled and
  splitting would add handoff friction. `/context` showed the background agent past 370k tokens
  and still mid-flight (before its own completion report was even seen) — for comparison, the
  architect/graph-dba design-doc units in the same coordination each ran 100-300k tokens for a
  *single document*, not six code modules plus rewiring plus docs.
- **Context:** decomposition judgment call for an implementation unit sized directly off an
  architect's plan's own step table, on the assumption that "the plan already sequenced it, so
  one coder should execute the sequence" (per the routing table's coder guidance). The step table
  being internally well-sequenced is not the same signal as the unit being small enough for one
  agent run.
- **Suggested home:** prompt — the "Delegate with complete briefs" / dispatch step. A plan's own
  step table (L1-1..L1-6, U1-U9, etc.) sequencing several files per step across 6+ steps is a
  signal to **split the dispatch to match the plan's own step boundaries** (one coder unit per
  step or small step-cluster, sequenced via chained briefs or SendMessage continuations) rather
  than handing the entire table to one agent as "a single coherent diff" — even when every step
  touches files touched by an adjacent step, which reads like an argument for one agent but in
  practice just produces one very long, hard-to-checkpoint run. Candidate rule: if a plan's step
  table has more than ~3 steps or spans more than ~5 files, treat that as the decomposition unit
  boundary, not "the whole landing," and sequence the resulting units as dependent (same-file)
  dispatches rather than one mega-brief. Not yet verified whether the eventual result was actually
  deficient because of the size (unconfirmed at the time of this entry — the run had not yet
  completed) — the fact worth capturing is the token/duration cost signal itself, independent of
  outcome quality.

## 2026-08-11 — Update: the oversized-landing entry above is now stakeholder-confirmed, and the outcome cost is in

- **Evidence:** the same K-042 Landing 1 unit finished at **458k subagent tokens / 222 tool calls
  / ~45 min** for the initial dispatch (on top of the ~370k already burned mid-run at the time of
  the entry above). Its diff-scoped `analyst` gate then found 2 majors — both test-coverage gaps
  against the plan's own named done-conditions (`test_executor_agent.py`, `test_responder.py`,
  `test_tools.py` left completely untouched despite being named in the plan's §5 file list; the
  AC-13 tripwire shipped with zero test coverage) — requiring a **second** dispatch to the same
  `coder` agent just to close two named gaps that were part of the original unit's own scope.
  Stakeholder's own words, unprompted, on seeing the cost: *"please never again create a landing
  so big."* This upgrades the prior entry from a `teco`-side cost observation to an explicit,
  standing user directive.
- **Context:** same coordination as the entry above; the missed-test-coverage finding is itself
  suggestive evidence for the mechanism — a single ~2.7M-ms, 6-step, ~10-file run plausibly lost
  track of 3 of the plan's ~15 named files under its own scope, in a way a smaller, checkpointed
  unit (one agent per plan step, verified before the next starts) would have caught immediately
  rather than needing a whole extra review-and-fix round trip. This specific gap — untouched
  files silently dropped from a large unit's own stated scope — argues that oversizing costs
  correctness, not just tokens.
- **Suggested home:** prompt — same "Delegate with complete briefs" / dispatch step as the entry
  above, now with a concrete correctness cost attached, not just a token-cost signal. Concrete
  rule to carry forward on this same feature's Landing 2 (already recorded in
  `falkor-chat/docs/plans/llm-provider-config-coordination.md`'s "Diff-scoped gate and
  fix-forward" section): split by the plan's own step boundaries (L2-1+L2-2 / L2-3 / L2-4 /
  L2-5+L2-6 / L2-7), one dispatch per step or small adjacent-step cluster, sequenced as dependent
  units rather than one landing-wide brief.
