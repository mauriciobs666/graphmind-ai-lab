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
| U4 | `architect` (resume U1) | `a2ce9950345d8a80a` | delivered | `docs/plans/embedding-migration.md` rev. (in place) | `analyst` (re-review) → **pending, next unit** | 343k tok / 125 tool uses (this turn; 547k/189 cumulative across U1+U4) |
| U5 | `analyst` (re-review of U4) | — | queued | `docs/reviews/embedding-migration.md` Pass 2 | — | — |

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

## Pause (2026-09-19, user-requested)

Paused here at the user's explicit request — they need exclusive machine access for a `model-bench`
run and will resume this chain in a **new session**. **U5 (analyst re-review of the U4 revision) is
the next unit — not yet dispatched.** Nothing else is blocked: `graph-dba`'s companion note (U3) is
final (item 6 verified safe), so once U5 clears, implementation (`coder`/`tdd-engineer` on `pin` +
the non-Cypher two-thirds of `migrate` immediately; the Cypher-dependent third once U5's verdict is
in) can proceed without further design-gate dependencies. The plan's own remaining open item —
Option A vs. Option B for FR-2's enforcement mechanism (§3.2) — is a stakeholder decision, not
blocked on U5; surface it to the user whenever convenient before implementing that specific step
(§5 step 2 only; steps 1/3 and `migrate` are unaffected either way).

A resuming session: read this ledger, confirm U1-U4's artifacts still match what's described here
(`git log`/`git show` against the commit below), then dispatch U5.
