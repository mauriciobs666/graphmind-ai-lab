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
| U4 | `coder` | — | queued | Landing 1 implementation (L1-1..L1-6, plan §6) | `analyst` diff-scoped re-gate |
| U5+ | TBD | — | queued | Landing 2 implementation (L2-1..L2-7, plan §7) — **prereq:** architect's one-line "None/False"→"None" fix on §5/§7-L2-1/§12, tracked below | `analyst` re-gate |
| U6 | `qa-engineer` | — | queued | Landing 1 acceptance pass — `docs/test-plans/llm-provider-config.md` + `-report.md` (AC-1, AC-4 partial, AC-5, AC-12, AC-13; AC-2/AC-3 structural per stakeholder decision 3) | — |
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
`coder`.**

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
