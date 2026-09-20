# Embedding model migration & index rebuild — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** — (M6+)

Coordinates delivery of `docs/requirements/embedding-migration.md` (Status: Ready for design,
2026-09-19): a reusable, on-demand capability to re-embed a workspace's `Message`/`Chunk` data and
rebuild its vector index when the embedding model changes, plus a workspace-level model-pinning
safety net (FR-1/FR-2/FR-5) so a global default swap never silently affects existing workspaces.
Triggered by an urgent LM-Studio memory-pressure swap off the current model
(Qwen3-Embedding-0.6B); destination model choice and which production workspace(s) to migrate are
deliberately out of scope for this chain (requirements doc, "Open questions").

CPG note: `cpg_falkorchat` exists, built from commit `07c252d`; `HEAD` has since moved by exactly
one unrelated commit (`999141f`, a salesperson `systemPrompt` language-salience fix, no relation to
embedding code) — usable for structural navigation, not rebuilt for this chain.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `a2ce9950345d8a80a` | delivered | `docs/plans/embedding-migration.md` | `analyst` → — | 204k tok / 64 tool uses |
| U2 | `analyst` | `aea05bc9eb8f79ce7` | delivered | `docs/reviews/embedding-migration.md` | `analyst` → needs changes (2 blockers) | — |
| U3 | `graph-dba` | `a49343cae70f94e54` | delivered | `docs/plans/embedding-migration-graph.md` (+ `claude/graph-dba/falkordb-quirks.md` update) | teco (spot-check) → verified | 185k tok / 42 tool uses |
| U4 | `architect` (resume U1) | `a2ce9950345d8a80a` | delivered | `docs/plans/embedding-migration.md` rev. (in place) | `analyst` (re-review, U5) → **approve** | 343k tok / 125 tool uses (this turn; 547k/189 cumulative across U1+U4) |
| U5 | `analyst` (re-review of U4) | `a84297e88678205db` | delivered | `docs/reviews/embedding-migration.md` Pass 2 (in place) | teco (spot-check) → verified | 116k tok / 16 tool uses |
| U6 | `coder` | — | queued | `scripts/embedding_migration.py` (`pin`) + `scripts/pin_workspace_embedding_model.sh` (§5 step 1) | `analyst` → — | — |
| U7 | `coder` (after U6, same file) | — | queued | `scripts/embedding_migration.py` (`migrate`) + `scripts/migrate_embeddings.sh` + interrupt/resume tests (§5 steps 4-5) | `analyst` → — | — |
| U8 | `coder` (after U6; parallel-safe with U7, disjoint files) | — | queued, blocked on stakeholder's Option A/B pick (§3.2) | FR-2 enforcement mechanism (§5 step 2) | `analyst` → — | — |

_U4 correction, 2026-09-19: this row was logged `in-flight` before the revision brief was actually
sent — the agent sat idle since U1's handback until a status-check message (not a revision request)
reached it. Real revision brief (both blockers + 3 minors from `docs/reviews/embedding-migration.md`)
sent 2026-09-19 ~19:55 — genuinely in-flight from that point, delivered ~20:07._

**U4 spot-check (teco, before commit):** independently verified three of the revision's load-bearing
citations directly against source — `scripts/start_agent_team.sh:~202-205` and
`scripts/start_demo.sh:~164-166` both do carry a `bootstrap_schema.sh` call site as claimed (Option
B's named call sites), `scripts/test_queries.sh:~103` is the unchanged `ws:test` bootstrap call as
claimed, and `server/falkorchat/config.py:279`'s `get_context()` docstring does say "M1 resolves
every call to one hardcoded tenant" (backs §2.6's one-process-one-workspace claim). Section
numbering re-checked whole (`grep -n '^## \|^### '`): monotonic 1→7, 2.1→2.9, 3.1→3.5, no
duplicates/gaps — the renumbering claim holds. **Not yet independently re-derived:** whether Option
B's claimed regression (the `FALKORCHAT_ENABLE_AGENT=0` no-config mode) is real, or whether the new
§3.4 step 0 traffic-stop precondition and the `DROP VECTOR INDEX` resume guard are correctly wired
into §5's steps — left for U5's re-review rather than re-verified twice.

**U5 spot-check (teco, before committing/dispatching further):** independently re-verified all six
of Pass 2's newly-cited source facts against live source, not the review's paraphrase —
`start_server.sh:148`, `start_demo.sh:166`, `start_agent_team.sh:205` each do call
`bootstrap_schema.sh` as claimed; `start_server.sh:16-17`'s header text ("Set 0 to serve the UI/REST
without the AI loop") matches verbatim; the six `seed_*.sh` scripts each hold only a precondition
*comment*, never an actual `bootstrap_schema.sh` call (`grep -n bootstrap_schema.sh scripts/seed_*.sh`,
confirmed); and `server/falkorchat/app.py`'s executable `ModelGateway.from_env()` call (line 607) is
reached only after the `if not config.ENABLE_AGENT: return ...` early-return at line 588 — the one
other hit at line 289 is inside a docstring's illustrative code example, not live code, so it
doesn't undercut the claim. All confirmed exactly as Pass 2 states. Verdict **approve** stands.

## Pause (2026-09-19, user-requested) — resumed 2026-09-20

Paused at the user's request for exclusive `model-bench` machine access; resumed in a new session
per the plan (ledger + `git log`/`git show` against `efeb4a88` reconciled first, per this doc's own
resume instructions — U1-U4's committed artifacts matched exactly what this ledger described).
**U5 is now delivered and independently spot-checked — the design-review gate on
`embedding-migration.md` is closed.** Nothing about the plan itself is still blocked on review.

Two things remain open, tracked as U6-U8 above:

- **Implementation, unblocked now:** `graph-dba`'s companion note (U3) is final (item 6 verified
  safe), so both `pin` (§5 step 1, U6) and `migrate` (§5 steps 4-5, U7) can be built regardless of
  the item below. U7 is sequenced after U6 (both touch `scripts/embedding_migration.py`).
- **A stakeholder decision, independent of U6/U7:** Option A vs. Option B for FR-2's enforcement
  mechanism (§3.2, plan recommends B) — blocks only U8 (§5 step 2 specifically); steps 1/3/4/5 are
  unaffected either way. Surfaced to the user alongside this resume.
