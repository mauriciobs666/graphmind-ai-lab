# LLM Provider & Model Configuration — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-042 (M4)

Coordination record for delivering `docs/requirements/llm-provider-config.md`.
This document, not any agent's context window, is the state of record.

## Stakeholder decisions (2026-08-10)

1. **Requirements are settled.** The `Interviewing` → `Ready for design` flip is a
   mechanical unit routed to `tico` (U0); design proceeds in parallel, not behind it.
2. **Two landings**, not one push:
   - **Landing 1** — FR-1..FR-6, FR-11..FR-15, FR-20: the two config files, the single
     resolver seam, per-kind defaults, per-model settings, env-var cutover. Shippable and
     demonstrable on its own.
   - **Landing 2** — FR-7..FR-10, FR-16..FR-19: roles, ordered fallback chains, workspace
     override + precedence, resolved-model trace recording, embedding-dimension guard.
3. **No cloud API key available.** AC-2 (`{env:...}` secret substitution against a real
   hosted provider) and AC-3 (three provider kinds end-to-end) are **deferred / model-gated**
   for the acceptance pass — the design must still support them, and QA records them
   structurally verified rather than end-to-end, as K-025 handled its gated ACs.
4. **Milestone M4**, backlog item **K-042**.

## Environment facts established at decomposition

- The stakeholder's real `~/.config/opencode/opencode.json` declares **one** provider:
  `lmstudio`, `baseURL: http://192.168.0.69:1234` (LAN IP, **no `/v1` suffix**), one model,
  no `{env:}` substitution currently in use, no cloud provider. This is the only shared-file
  sample that exists; a second OpenAI-compatible LAN host does not exist today.
- FR-20 blast radius (sites setting the replaced env vars):
  `server/falkorchat/config.py`, `server/falkorchat/app.py`, `scripts/start_server.sh`,
  `README.md`, `server/tests/test_workflow_live.py`, plus docs.
  **Correction (2026-08-10, `architect`'s v2 plan §2.9/§11.2):** `compose.yaml` does **not**
  set any of the four — this document's original scan was wrong about that site (grep noise, not
  an extension-filter miss like `.env.example`). The plan's own unfiltered re-scan
  (`grep -rn -e FALKORCHAT_LLM_BASE_URL -e FALKORCHAT_LLM_MODEL -e FALKORCHAT_EMBEDDING_BASE_URL
  -e FALKORCHAT_EMBEDDING_MODEL .`, excluding `.git/`/`.venv/`/`__pycache__/`/`docs/archive/`/the
  K-042 documents themselves) is the list to trust; `compose.yaml` still needs Landing-1 changes
  (the two config-file paths + a read-only bind mount) but not because it sets a legacy var.
  **`server/.env.example` sets all four** and was missed by this document's original
  extension-filtered scan — folded into U1's plan as finding M-1 and into U6's scope below.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U0 | `tico` | `ae8daf5e6fb321743` | accepted | requirements `Status:` → `Ready for design` | none (mechanical) |
| U1 | `architect` | `a6607eea81e284553` → `a03bf509bc62cd995` | delivered | `docs/plans/llm-provider-config.md` v2 | Pass 1 → needs changes; v2 complete, awaiting Pass 2 |
| U2 | `graph-dba` | `a9469ba9c6b47c56b` → `a59fa97de2ef0a511` | delivered | `docs/plans/llm-provider-config-graph.md` v2 | Pass 1 → m-4/m-5/m-6; v2 complete |
| U3 | `analyst` | `a87afc398f73067b8` | accepted | `docs/reviews/llm-provider-config.md` (Pass 1) | needs changes |
| U3b | `analyst` | `a87afc398f73067b8` | accepted | `docs/reviews/llm-provider-config.md` (Pass 2) | needs changes — 1 blocker, 1 minor |
| U1v3 | `architect` | `a03bf509bc62cd995` | accepted | plan v3 — adopt `modelFallback` | Pass 3 → approve with suggestions |
| U2v3 | `graph-dba` | `a59fa97de2ef0a511` | accepted | graph note v3 — fix stale §6.5 language | Pass 3 → approve with suggestions |
| U3c | `analyst` | `a87afc398f73067b8` | accepted | `docs/reviews/llm-provider-config.md` (Pass 3) | approve with suggestions — design phase closed |
| U4 | `coder` | `ab38a5f2c9766f810` | delivered (uncommitted) | Landing 1 implementation (L1-1..L1-6, plan §6) — 778 passed offline; mutation-tests on §4.9 ladder + `/v1` rule all caught; `teco` independently re-verified | `analyst` diff-scoped re-gate — **dispatched 2026-08-10** |
| U4-gate | `analyst` | `a3b8fcad7a0088cfd` | accepted | `docs/reviews/llm-provider-config.md` `## Landing 1 code review` (`Version: 4`) | approve with suggestions — 2 majors (coverage gaps), 2 minors, no blocker |
| U4-fix | `coder` (resumed) | `ab38a5f2c9766f810` | accepted | closed Major 1 (7 new AC-13 tests) + Major 2 (6 new consumer-binding tests) — 13 new, 791 total | `teco` re-checked directly: diff scoped to exactly the 3 named test files (+207 lines, zero production code), 791 passed re-run myself |
| **committed** | `teco` | — | **`a2b8aa9`** | Landing 1 full diff (38 files, +3347/-193) | — |
| U5-prereq | `architect` | `a1893af3fc6cffdbd` | accepted | plan `Version: 4` — 3× `` `None`/`False` `` → `` `None` `` (§5, §7 L2-1, §12.1) + dated revision note | none (trivial wording fix, scope-verified by `teco` diff read) — committed `d7136ec` |
| U8 | `coder` | `aa36e66470469ff6d` | accepted | L2-1 + L2-2 (roles + ordered fallback chains; record resolved model/source/fallback on `StepRun`) — folds in the QUERIES.md/test_queries.sh gap. Committed `17c20dc` | `analyst` (`a5469d493547b45ca`) → **approve with suggestions**, no blocker |
| U9 | `tdd-engineer` | `a6012b2f9de191b86` | in-flight | L2-3 (workspace override + precedence — closes B-1) — carries the `modelSource` reshape + Minor 3 forward from the U8 gate | `analyst` diff-scoped gate |
| U10 | `coder` | — | queued | L2-4 (publish-time rejection) | `analyst` diff-scoped gate |
| U11 | `coder` | — | queued | L2-5 + L2-6 (loud use-time failure + embedding-dim guard) | `analyst` diff-scoped gate |
| U12 | `coder` | — | queued | L2-7 (docs + close) | — |
| U6 | `qa-engineer` | `a55e67da7ed500591` | accepted | Landing 1 acceptance pass — `docs/test-plans/llm-provider-config.md` + `-report.md`, committed `20d0262` | **PASS**, 1 minor defect (D-1) — `teco` independently re-verified (791 passed re-run, D-1 reproduced by direct read of `config/opencode.example.json`) |
| U7 | `qa-engineer` | — | queued | Landing 2 acceptance pass — remaining ACs | — |

**U6/U7 renumbered from the original devops placeholder:** L1-5's env-var cutover (config.py,
`.env.example`, `compose.yaml`, `start_server.sh`, README, AGENTS.md) is core resolver-coupled
code (`assert_no_legacy_model_env()` is called from `ModelGateway.from_env()`), not an environment
blocker — it stays inside `coder`'s U4, per the plan's own L1-5 file list, rather than forking to
`devops` and creating a same-file split. `devops` remains the fallback if U4 hits a genuine
environment blocker (deps, containers, secrets) mid-run.

## Documentation impact (scanned at decomposition)

Every item below is a done-condition on the unit that invalidates it, not a cleanup pass:

| Document | Why it changes | Rides with |
|---|---|---|
| `falkor-chat/docs/DESIGN.md` §1.3 (M2 stack), §14 (config seam) | the single-model stack and the env-var config seam are replaced | U1 (design), implementers (as built) |
| `falkor-chat/AGENTS.md` | key-scripts env table + M1-server run instructions reference the old vars | U6 |
| `falkor-chat/README.md` | documents the replaced env vars | U6 |
| `falkor-chat/scripts/start_server.sh` (header comment + body) | sets the replaced env vars | U6 |
| `falkor-chat/server/.env.example` | sets all four replaced env vars (missed by the original scan — see correction above) | U6 |
| `falkor-chat/compose.yaml` | does **not** set the vars (correction above), but still needs the two new config-file paths + read-only bind mount | U6 |
| `falkor-chat/docs/BACKLOG.md` | K-042 + M4 milestone row | U1 |
| `falkor-chat/docs/HISTORY.md` | one entry per delivered landing | each landing's closing unit |
| `falkor-chat/docs/manuals/` | admin-facing "how to configure models" is a plausible new manual — decide at Landing 1 close | flagged for `tico`, gated `qa-engineer` + `analyst` |
| `falkor-chat/docs/plans/local-model-ram-budget-ml.md` (`Status: active`, owner `data-scientist`) | 8 references incl. a literal `FALKORCHAT_LLM_MODEL=` env block, per `architect`'s plan §2.9/§9.3 item 5 — a live document this K-042 work invalidates but does not own | flagged for **`data-scientist`**: a dated amendment noting the env-var mechanism was replaced by K-042, applied by that document's own owner, at or before Landing 1 close |
| `falkor-chat/docs/QUERIES.md` §12.2 (`record_step_and_advance`) and §12.8 (`read_step_runs`) | plan §10 "Suite discipline": Landing 2 touches `record_step_and_advance` and adds two reads, so QUERIES.md must rise with the `-graph.md`-designed Cypher (§1.4/§1.7 `[proposed]` blocks) turned `[verified]` | U8 (StepRun `resolvedModel`/`modelSource`/`modelFallback`) |
| `falkor-chat/scripts/bootstrap_schema.sh` | `-graph.md` §4 specifies new DDL (a `WorkspaceConfig.workspaceConfigId` index + a backing `UNIQUE` constraint) for the workspace-override singleton — **named in `-graph.md` but absent from `architect`'s plan §7 L2-3 file list**, a gap this coordination's scan catches now rather than at implementation time | U9 (workspace override) |
| `falkor-chat/docs/QUERIES.md` (new entry) | `-graph.md` §2.4/§2.5 (`WorkspaceConfig` MERGE/read) and §3.2 (`db.indexes()` dimension introspection) are new query shapes with no QUERIES.md entry yet | U9 (§2.4/§2.5) and U11 (§3.2, embedding-dim guard) |
| `falkor-chat/scripts/test_queries.sh` | any new/changed Cypher above must be exercised by the live query suite (AGENTS.md rule 5) before that unit's diff is gated | whichever of U8/U9/U11 touches the Cypher in question |

## Milestone close

At M4 close, every document this coordination froze flips to `Status: archived`, routed by
kind per root `AGENTS.md` (plans → `architect`, `-graph` → `graph-dba`, reviews → `analyst`,
requirements/manuals → `tico`, test-plans/reports → `qa-engineer`, this coordination → `teco`).

## Findings re-verified by `teco` at integration

The plan rests on four live claims. Each was independently re-checked, not accepted on report:

| Claim | Verdict |
|---|---|
| A missing `/v1` prefix fails **silently** | **Confirmed live.** `POST localhost:1234/chat/completions` → HTTP **200** with body `{"error":"Unexpected endpoint or method. (POST /chat/completions)"}`. Surfaces as `KeyError: 'choices'`. |
| `TraceEvent`s are debug-only, so FR-8 cannot live on them | **Confirmed.** `executor.py:390` — `tracer = self._tracer if run["trace"] else _NULL_TRACER`. (Plan cites `:388`; 2-line drift, mechanism exact.) |
| `_urllib_transport` is duplicated with no timeout | **Confirmed.** Two copies: `llm.py:77`, `embedding.py:37`. |
| The stakeholder's declared endpoint is unreachable | **Confirmed.** `192.168.0.69:1234/v1/models` → connection failure (HTTP 000); `localhost:1234/v1` answers. |
| "37 `llm=` / 23 `guard_judge=` injection sites" | **Off by one each** — my count is 38 / 24. Rationale detail, not load-bearing; no action. |

## Stakeholder decisions taken mid-flight

- **2026-08-10 — FR-10's "suspends" ⇒ `failed`-with-cause. SETTLED, stakeholder's words: "failed is
  fine, go with that."** `architect`'s reasoning stands: `waiting` means "a human can unblock this",
  which an unresolvable model is not. The requirements doc's FR-10 wording ("the run suspends") is
  now narrower than the implemented behaviour — **`tico` should reconcile FR-10's text** at the next
  requirements touch so the document does not read as contradicting the build.

## Plan gate — Pass 1 (`analyst`, 2026-08-10): **needs changes**

2 blockers, 6 majors, 9 minors. Both blockers independently re-verified by `teco`:

- **B-1** — Landing 2 is not buildable for the `guard` kind; the plan's §6.1 "every resolution
  point carries the workspace" is false, and the naive fix lands inside the SHA-locked
  `_drive_loop`. Routed to **both** owners: plan-side to `architect`, and the `-graph.md` §2.6
  read-placement half to `graph-dba`.
- **B-2** — the FR-10 taxonomy has two holes. Verified in a live interpreter:
  `urllib.error.HTTPError` **is** a subclass of `URLError` (so the plan's catch order makes the
  HTTP-status branch dead code), and a `urlopen` read timeout raises a bare `TimeoutError` whose
  MRO is `(TimeoutError, OSError, …)` — **not** a `URLError`, so it escapes classification and
  FR-18's fallback would never fire on a hung endpoint. Routed to `architect`.

### A miss in this coordination's own scan

The gate found `server/.env.example:19-21,30-31` sets all four env vars FR-20 replaces — a site
**this document's blast-radius scan also missed**. Root cause: the original sweep was
`grep --include='*.sh' --include='*.md' --include='*.py' --include='*.yml' --include='*.json'`,
which structurally cannot match an extensionless dotfile. Any future env-var cutover scan must
not be extension-filtered. `architect` was told to re-derive §2.9 with a method that catches
dotfiles.

### Referred items, ruled

| Item | Ruling |
|---|---|
| `StepRun.model` vs `resolvedModel` | **`resolvedModel`** (+ `modelSource`) — `graph-dba` carried it; the plan had delegated the shape |
| `{kind\|"*"}` wildcard / FR-16 | **Per-kind is faithful**, not a narrowing — "everything" is a scope quantifier, not an arity claim. **Does not go to the stakeholder.** `graph-dba`'s "contradicts FR-19 by construction" reasoning was overstated and is being corrected |
| `/v1` normalization | Right in principle, insufficient as specified (3 gaps); also affects `/embeddings`, where `error` is a *string* not an *object* |
| FR-4 vs surviving `llm=` injection | **Faithful** — DI reads no config; close the hole with an enforcement test |
| §6.1 landing boundary | Right on 5 of 6 seams; the sixth is B-1, and the *inbound* override carrier is unsolved by both documents |

## Open questions still outstanding

*Both former open questions were closed by the Pass 1 gate — see the rulings table above. Neither
needs a stakeholder decision.*

- **Still unsolved by both documents:** the *inbound* workspace-override carrier — how a
  workspace-level setting physically reaches the resolver at each of the four consumers. Flagged by
  the gate under §6.1; expected to be closed by the two v2 revisions in flight. If v2 does not close
  it, it becomes a design unit of its own before Landing 2.

## Transient failure and recovery (2026-08-10)

Both v2-revision agents (`a6607eea81e284553` on the plan, `a9469ba9c6b47c56b` on the graph note)
were terminated mid-run by a platform API session-limit error — not a deficient result. Both had
substantial, coherent work already on disk:

- **Plan** — `Version: 2` header already stamped, ~894 lines of diff, B-1/B-2/A-1..A-5 and majors
  m-1/m-2/m-3/m-7/m-8/m-9 all appeared addressed with an index in-document.
- **Graph note** — B-1's read-site relocation addressed with a dated revision note, a new §8
  live-verification log (19 entries) appended, but its own last message
  (*"Now the three minors the review assigns to me. m-4 first..."*) showed m-4/m-5/m-6 not yet
  started, and no `Version:` field had been added to the header.

Re-dispatched as **fresh agents with state-recovery briefs** (inspect `git diff` of their own
file and continue from actual state, not restart) per this agent's own standing guardrail for
transient platform failures — distinct from the close-the-loop-on-the-same-delegate path, which
is for deficient results, not platform failures.

## Design phase closed (2026-08-10)

Pass 3 (`docs/reviews/llm-provider-config.md`, committed `0719e8a`): **approve with suggestions**,
no blocker survives. Both Pass 2 items independently re-verified by `teco` before acceptance:

- **P2-B adoption** — the `ChatResult.fallback` → `StepResult.modelFallback` →
  `record_step_and_advance(model_fallback=...)` carrier chain matches `-graph.md` §1.3/§6.2
  field-for-field (property name, nullability, computation formula, orthogonality reasoning, m-6
  last-wins extension to all three fields).
- **Stale §6.5 language** — confirmed via grep that only historical/quoted occurrences of the
  withdrawn phrase remain; §6.5's own bullet is rewritten in its own words.

**One residual minor, Landing-2-only, tracked for pre-L2 fix:** plan §5 (`llm.py` row), §7 L2-1's
done-condition, and §12 say the non-fallback `ChatResult.fallback` value is `` `None`/`False` ``;
`-graph.md` §1.3/§6.2 (lines 246, 257) is unambiguous the non-fallback state is **absent (`None`)
only**, matching what §7 L2-2's done-condition already states correctly two rows later. Confirmed
by `teco` via direct read of both documents. Does not touch Landing 1 — routed to `architect`
ahead of Landing 2 kickoff, not dispatched now.

Both design documents (`llm-provider-config.md` v3, `llm-provider-config-graph.md` v3) and the
review (`Version: 3`) are the documents of record. **Landing 1 implementation (U4) dispatched to
`coder`** (`ab38a5f2c9766f810`): L1-1..L1-6 per plan §6, Landing-2 scope explicitly fenced off,
mutation-testing of the §4.9 ladder and the `/v1` rule required, no commit — diff left for the
coordinator to gate and commit.

## Paused (2026-08-10) — out of credits

`coder` (U4) delivered: **778 passed** offline (verified green both normally and with `HOME`
pointed at an empty directory, the M-2 done-condition), new `test_transport.py` (13) +
`test_modelconfig.py` (51), the FR-4 AST-enforcement test in place, unfiltered legacy-env-var
grep clean, all 4 mutation-tests caught and reverted. Diff left **uncommitted** in the working
tree as instructed — none of it has been independently re-verified or committed yet.

**Stakeholder-flagged budget constraint: pause here, do not dispatch the `analyst` diff-scoped
re-gate or anything further.** U4 alone cost 458k subagent tokens / 222 tool calls / ~45 min —
on top of the ~370k it had already burned mid-flight per an earlier `/context` check. This is
roughly the same order of cost as the entire three-pass design-review cycle (three `analyst`
gates + two revision rounds each on `architect`/`graph-dba`) combined, for one implementation
unit. See `claude/teco/kaizen/inbox.md`'s 2026-08-10 entry on this unit's size.

**Deviations `coder` flagged in its own report, not yet adjudicated by anyone:**
- `Dockerfile` gained `COPY config config` — not in the plan's L1-5 file list, but necessary for
  the container's default `MODEL_CONFIG_PATH` to resolve.
- Two edits outside the plan's stated L1-5 sites: 2 lines in `docs/BACKLOG.md` and an addition to
  `falkor-chat/AGENTS.md` itself, both to keep the unfiltered legacy-env-var grep clean.
- `compose.yaml`/`Dockerfile` changes are **unverified against real `docker build`/`docker
  compose`** — no Docker available in `coder`'s environment; best-effort per the plan's spec.
- Flagged for the eventual review: `EmbeddingWorker`/`GraphragRetrieveTool` make two gateway
  calls per embed (one for the client, one for `dim`) when no `expected_dim` override is given —
  cheap offline, but worth a second look against the plan's "resolve is a cheap per-call lookup"
  assumption.

**Resume point, when credits allow:** independently verify `coder`'s claims (re-run the offline
suite myself, spot-check at least one mutation-test claim, read the flagged deviations against
the plan), then dispatch the `analyst` diff-scoped re-gate, then commit + `qa-engineer`
Landing-1 acceptance pass — exactly the sequence already recorded in the ledger above, just not
yet started.

## `teco`'s own re-verification (2026-08-10) — done, one step ahead, then paused again

Ran myself (no subagent dispatched — cheap enough to do directly):

- **Diff matches the reported file list** — `git status --short` lines up exactly with `coder`'s
  own "Files touched" list.
- **`778 passed, 1 deselected`** — reproduced exactly, twice: normal `HOME` and `HOME` pointed at
  an empty directory (the M-2 done-condition).
- **One mutation-test claim reproduced independently**, not just re-read: swapped the
  `HTTPError`/`URLError` catch order in `transport.py` myself, re-ran `-k transport` →
  `test_http_error_branch_is_reached_first_and_preserves_the_body` failed exactly as `coder`
  reported (asserted body string missing from the now-generic `URLError` message), then reverted
  cleanly (`git status` confirms no diff from the revert).
- **FR-4 AST-enforcement test** — exists (`test_fr4_only_modelconfig_constructs_openai_compatible_clients_directly`,
  `test_modelconfig.py:660`), passes standalone.
- **Unfiltered legacy-env-var grep, re-run independently** — clean: only `config.py`'s own
  tripwire list (`LEGACY_MODEL_ENV_VARS`) and `docs/plans/local-model-ram-budget-ml.md` (already
  flagged, out of this plan's scope, owner `data-scientist`).
- **`Dockerfile` deviation read against the plan** — the plan's L1-5 file list did not name
  `Dockerfile`, but `config.py`'s `MODEL_CONFIG_PATH` default does resolve via the same
  sibling-of-`server/` convention `app.py` already uses for `web/` (documented inline in
  `coder`'s `Dockerfile` comment addition) — the addition is consistent with the plan's own
  stated mechanism, not a scope add.

**Not yet done** (this was the "one step" — stopping here per stakeholder instruction, more
credits available but not unlimited): the `analyst` diff-scoped re-gate itself, the commit, and
`qa-engineer`'s acceptance pass. `compose.yaml`/`Dockerfile` remain unverified against a real
`docker build`/`docker compose` (no Docker in this environment either) — still an open item for
whichever step next has Docker access.

## Diff-scoped gate (`U4-gate`) and fix-forward (`U4-fix`) — 2026-08-10/11

`analyst` (`a3b8fcad7a0088cfd`) gated the actual Landing 1 diff (not just the design docs):
**approve with suggestions, no blocker.** Mutation-tested the §4.9 ladder and the §4.3
strip-then-normalize rule itself (copy-aside, never `git checkout`), both reproduced the exact
regressions the design-phase review had already found; suite reproduced at 778 passed, twice
(normal + empty-`HOME`); unfiltered legacy-env-var grep re-confirmed clean; both real config
fixtures confirmed byte-identical to the actual files on disk. `teco` independently re-verified
both majors before acting on them (`git diff --stat` on the three named test files → zero
changes; `grep` for `assert_no_legacy_model_env`/`LEGACY_MODEL_ENV_VARS` under `server/tests/` →
no hits). **2 majors** (both test-coverage gaps against the plan's own named done-conditions, not
behavior defects — reviewer hand-verified correctness for both): AC-13 tripwire untested; three
of five rewired consumer bindings (`test_executor_agent.py`, `test_responder.py`,
`test_tools.py`) untouched despite being named in plan §5's file list, one of them (the
`GraphragRetrieveTool`/M-3 binding) an explicit L1-4 done-condition. 2 minors (latent double-
resolve in `EmbeddingWorker`; an `.env.example` portability note), no Landing-2 scope leakage.

Stakeholder decision: fix now, same delegate. Sent via `SendMessage` to `coder`
(`ab38a5f2c9766f810`, resumed from its own transcript) — scope narrowed to exactly the two
majors' missing tests, explicitly no new scope, diff stays uncommitted. In flight.

**Stakeholder directive, standing for the rest of this feature: "please never again create a
landing so big."** Landing 1 (one `coder` dispatch covering plan §6's L1-1..L1-6, ~10 files, six
sequenced steps) cost ~830k subagent tokens across its two dispatches (458k initial + the
in-flight fix-forward), separate from the ~370k it had already burned mid-run before that count
was even taken. **Binding for Landing 2's dispatch:** do not repeat a single-mega-unit dispatch
against plan §7's L2-1..L2-7 table. Split by the plan's own step boundaries — one `coder`/
`tdd-engineer` unit per step or small adjacent-step cluster (e.g. L2-1+L2-2 together since
`modelFallback` spans both; L2-3 alone; L2-4 alone; L2-5+L2-6 together; L2-7 docs-only) — sequenced
as dependent dispatches (chained `SendMessage` continuations or fresh `Agent` calls handed the
prior step's diff state), not one brief covering the whole landing. See
`claude/teco/kaizen/inbox.md`'s 2026-08-10 entry, now confirmed by the stakeholder rather than
just a `teco`-side observation.

## Landing 1 QA acceptance pass (U6) — 2026-08-11, closes Landing 1

`qa-engineer` (`a55e67da7ed500591`) ran the first execution-based (black-box) pass on this
feature — driven against the real running server, real FalkorDB (`falkordb-dev`, untouched), and
real LM Studio at `localhost:1234`, using a throwaway `ws:qa-k042` graph deleted at teardown. All
nine in-scope ACs (AC-1, AC-4 partial, AC-5, AC-12, AC-13 end-to-end; AC-2/AC-3 structural, per
stakeholder decision 3) passed. Full offline suite re-reproduced independently: **791 passed, 1
deselected**.

**One defect, D-1 (Minor):** `config/opencode.example.json`'s `openai` provider entry has no
`options.baseURL`, so the shipped cloud-provider example can't resolve as shipped — a one-line
fixture gap, not a resolver defect (isolated: adding the key makes it resolve identically to the
other two providers). Not fixed in this pass, left for a documentation touch-up. `teco`
independently re-verified both the suite count and D-1 (direct read of the file — `apiKey` present,
`baseURL` absent) before accepting.

Deliverables committed as `20d0262`: `docs/test-plans/llm-provider-config.md` (TP-001..TP-010),
`docs/test-reports/llm-provider-config-report.md`, a `HISTORY.md` entry, and `qa-engineer`'s own
kaizen-inbox learnings (the no-implicit-`baseURL`-default fact, and a Bash-tool backgrounding
gotcha).

**Landing 1 is now fully closed**: implemented (`a2b8aa9`), diff-gated and fixed
(`U4-gate`/`U4-fix`), and QA-accepted (`20d0262`). Residual, not blocking: D-1 (cheap doc/fixture
fix), the pre-existing Landing-2-only `"None`/`False`"`→`None` phrasing fix routed to `architect`,
and `compose.yaml`/`Dockerfile` still unverified against a real Docker build (no Docker anywhere in
this pipeline).

**Landing 2 (U5+) has not been dispatched.** Per the stakeholder's standing directive, it must be
split along plan §7's L2-1..L2-7 step boundaries into multiple smaller, sequenced units — never
one landing-wide brief again — and is being picked up in a fresh session.

## Log

- **2026-08-10** — Opened. Requirements read at `f78b824`; working tree clean against it
  (the reported modification was a stale mtime, `git diff` empty). U0/U1/U2 dispatched.
- **2026-08-10** — U0 accepted, committed `57ac4ec`. Diff verified: exactly the four intended
  edits, no requirement prose touched.
- **2026-08-10** — U1 delivered and committed `4be53e6` (plan + M4 row + K-042 item; BACKLOG diff
  is 71 insertions / 0 deletions, so nothing existing was disturbed). Claims re-verified as above.
  Awaiting the `analyst` gate, which is held until U2 lands so both documents are reviewed together.
- **2026-08-10** — U1's debug-only-trace finding was sent to the **in-flight** U2 agent, because
  U2 was dispatched before the finding existed and its brief had pointed it at the execution trace
  generically. Also pointed U2 at the plan's §8 interface items. Correcting a running agent's
  premise beat discarding its work.

## Resumed 2026-08-11 — Landing 2 kickoff

Fresh session, per the prior session's pause point. Read this document, the plan (v3 at read
time), `-graph.md` v3, and the review's Pass 1/2/3 + Landing-1-code-review sections in full before
acting — no unit dispatched on a stale read.

**U5-prereq dispatched and closed.** `architect` (`a1893af3fc6cffdbd`) fixed the three
`` `None`/`False` ``→`` `None` `` occurrences, bumped the plan to `Version: 4`, added a dated
revision note. `teco` independently verified the diff (`git diff docs/plans/llm-provider-config.md`)
touches exactly those three spots plus the header/revision-note addition — nothing else. Committed
`d7136ec`.

**A gap in the plan's own Landing-2 file lists, caught before dispatch.** `-graph.md` §4 specifies
DDL (`WorkspaceConfig` index + `UNIQUE` constraint) that `architect`'s plan §7 L2-3 never lists a
file for (`bootstrap_schema.sh` is absent from L2-3's file column), and plan §10's own "Suite
discipline" line ("QUERIES.md + the query suite must rise... owned by `-graph.md`") has no
corresponding row in any L2-N unit either. Folded into U9's (workspace override) and U8/U11's
(StepRun trace fields, embedding-dim guard) briefs as explicit done-conditions rather than spun out
as separate units — the underlying Cypher is already fully designed and `[proposed]`/live-verified
in `-graph.md` §1.4/§1.7/§2.4/§2.5/§3.2/§4, so implementing it alongside the Python that drives it
is the same unit of work, not a new one. Recorded in the Documentation impact table above.

**Landing 2 split and sequencing, per the standing stakeholder directive** ("please never again
create a landing so big"): five implementation units along the plan's own step boundaries
(U8 = L2-1+L2-2, U9 = L2-3, U10 = L2-4, U11 = L2-5+L2-6, U12 = L2-7 docs-only), each with its own
`analyst` diff-scoped gate (U12 is docs-only, no code gate). **Sequenced, not parallel** — U8, U9
and U10 all touch `services.py`/`executor.py`/`repository.py`/`modelconfig.py` in overlapping
ways, so each unit is briefed with the current `git diff` state and dispatched only after the prior
unit's diff is gated and (fix-forward if needed) committed. Committing per-unit rather than
Landing-1's single end-of-landing commit — five sequenced units each landing a working, gated,
committed slice is a cleaner recovery boundary than one 830k-token mega-diff, and it is what the
directive is asking for in spirit, not just in dispatch-count.

## U8 delivered — 2026-08-11

`coder` (`aa36e66470469ff6d`) delivered L2-1 (roles + ordered fallback chains, `FallbackClient`
with structurally-enforced no-mutable-state via `__slots__`) + L2-2 (`StepResult`/
`record_step_and_advance`/`read_step_runs` gain `resolvedModel`/`modelSource`/`modelFallback`,
matching `-graph.md` §1.4/§1.7 Cypher exactly) + the QUERIES.md/`test_queries.sh` gap. `teco`
independently verified: `git diff --stat` matches the reported file list exactly (12 files, +958/
-55, no leakage into `services.py`/`schemas.py`/`api.py`/`guards.py`/`responder.py`/`embedding.py`/
`tools.py` — confirmed by their absence from `git status --short`); offline suite re-run from
scratch, **822 passed, 1 deselected**, exact match to `coder`'s report; read the full diffs of
`llm.py`, `modelconfig.py`, `executor.py`, `repository.py` by eye against `-graph.md` §1.4/§1.5/
§1.6/§1.7 and the plan's §7 L2-1/L2-2 rows — all match; confirmed `_drive_loop`'s locked body is
untouched by reading the diff region.

**Deviation flagged by `coder`, independently confirmed real by `teco`, carried forward to U9
rather than adjudicated here (routed to the `analyst` gate and then to U9's brief):** `modelSource`
is derived locally in `_run_agent_node` from `config.get("model")` truthiness (`'step'` vs
`'default'`), not carried on `Resolution`/returned by `ModelGateway.resolve()` — because
`test_executor_agent.py`'s `RecordingGateway` pins `gateway.calls` to exactly one call per node
execution (`teco` confirmed this test double and its call-count assertion exist as described,
`server/tests/test_executor_agent.py:622,657,672,691`). Correct for L2-1/L2-2's own reachable
outcomes ({step, default} only — no workspace rung exists yet), but a real forward-compatibility
question for **U9 (L2-3)**: a workspace override silently overruling an explicit step choice is
invisible to a `config.get("model")`-only check — only `resolve()` itself knows which rung won.
Sent to the `analyst` gate as an explicit item to judge (accept as correctly-scoped for U8 with the
risk flagged forward, or require a reshape now); U9's brief will carry whatever the gate concludes.

**Observation, not a finding:** `coder` reports the plain offline `pytest` suite (not just
`test_queries.sh`) also touches the shared `reference`/`ws:test` graph state via `conftest.py`'s
`wf_repo` fixture (`DETACH DELETE` on every `wf_repo`-based test) when live FalkorDB is reachable —
pre-existing test-infra behavior, not introduced by this diff. `falkor-chat/AGENTS.md`'s key-scripts
table currently only documents this hazard for `test_queries.sh`. Worth a doc note at some point;
not gating this unit.

`analyst` diff-scoped gate dispatched (`a5469d493547b45ca`), covering both the standard code-review
checks and an explicit request to judge the `modelSource` deviation above.

## U8 gated and committed — 2026-08-11

`analyst` (`a5469d493547b45ca`) verdict: **approve with suggestions, no blocker**
(`docs/reviews/llm-provider-config.md` `Version: 5`, `## Landing 2 — U8 (L2-1/L2-2) code review`).
Independently reproduced the offline suite (822/1 deselected), recomputed the `_drive_loop` SHA
lock (`71055f756280`, matches, no locked code touched), ran the live query suite (295/295,
reseeded), and ran two of its own mutation tests (advance-on-failure, last-wins overwrite) —
both caught the injected regression as predicted. Judged the `modelSource` deviation **acceptable
as shipped for U8's own scope**, but ruled it a genuine forward-compatibility risk for U9 and gave
a concrete, actionable requirement (below). One new minor found independently (Minor 3:
`ModelGateway.embedder()` silently drops fallback-chain elements beyond the primary for kind
`embedding` — no AC requires it, not gating, deferred to whichever unit next touches `embedder()`).

**Committed `17c20dc`** (U8's implementation + the review doc's new section, bundled — same
pattern Landing 1's `a2b8aa9` used).

**Carried into U9's brief, verbatim from the gate's recommendation:** replace `_run_agent_node`'s
local `config.get('model')`-truthiness `modelSource` derivation with a resolver-sourced value that
can also report `'workspace'` — stated as an explicit done-condition, not left as an implicit
consequence of adding the override read. The gate's own worked example of the wrong path (a third
`elif` bolted onto the local truthiness check) is included in U9's brief verbatim, since it names
the exact bug class (a workspace override targeting a role that itself falls back) a naive fix
would silently reintroduce. Minor 3 (`embedder()` fallback-chain silently truncated) is also folded
into U9's scope as a small additional fix, since U9 is the next unit likely to touch
`modelconfig.py`.

**Routing decision (`teco`, not resolved by the coordination doc):** U8/U10/U11/U12 → `coder` (a
fully detailed plan exists for each, `coder`'s stated fit). **U9 → `tdd-engineer`**, deliberately
different from the rest: this is the unit that closes B-1 (the guard-kind workspace-carrier gap
that cost a whole review cycle in the design phase), its behaviour contract is unusually crisp and
enumerable (three precedence rungs × four consumer kinds, hard-cap direction, `run["ws"]`/
`run["modelOverrides"]` already staged by Landing 1 with zero lock reopen required), and
test-first is the more efficient path for a unit whose main risk is a silently-wrong precedence
direction rather than an implementation-shape decision. This is the "give it real attention" the
task brief asked for, made concrete as a routing choice, not just a review-depth note.
