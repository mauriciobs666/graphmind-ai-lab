# Review: Anthropic API adoption across modules (requirements)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/requirements/anthropic-api-adoption.md`

## Pass 1 — 2026-11-03

**Scope.** `docs/requirements/anthropic-api-adoption.md` (untracked, status Interviewing), read whole, against root `REVIEW.md` (no component `REVIEW.md` bears on a requirements doc), the repo's current state, and the prior decisions the brief named. Static review; nothing run.
**Verdict: needs changes** (two blockers, both small text fixes). Fix B1/B2 and the majors, then it is ready to flip.
CPG: considered, not relevant — a requirements document; the claims were checked by reading and grepping docs and config.

### Current-state claims (verified by reading/grep)
- **Confirmed.**
  - `modelconfig.py:73` has `SUPPORTED_PROTOCOLS = frozenset({"openai"})` and refuses other protocols at `:537`.
  - `decision-models/AGENTS.md:38` states "No hosted API client".
  - `tank` is `lmstudio/mistralai/ministral-3-3b` in `opencode/agents/tank/opencode.json`.
  - SERVER.md §1.8 matches the doc on per-workspace override (kinds `agent`/`step`/`embedding`/`guard`), roles, ordered fallback chains, and a hard-cap workspace rung.
  - The answering model and fallback flag are already recorded per step (SERVER.md §1.8 `resolvedModel`/`modelFallback`; DESIGN.md:578). FR-3's record requirement and FR-10 are therefore additive, not new plumbing, for falkor-chat.
  - `llm-provider-config.md` FR-13 (line 100) does name Anthropic.
- **Imprecise.** "model-bench targets LM Studio only (by design)": `docs/requirements/small-model-benchmarking.md` (archived) lines 210 and 336 say hosted models were offered and **declined, "the door stays open"**. This is not a prohibition, so no contradiction, but "by design" overstates it. Cite that line.
- **Not stated.** `extraction` is outside `KINDS` (SERVER.md §1.8, K-080), so a workspace switched to Anthropic still ingests and extracts locally. Say so under FR-2 or Out of scope.

### Findings

**B1 — blocker — FR-3 and FR-11 conflict on a bad or missing key.** Verified by reading. FR-3 falls back to local on any "failed Anthropic call". FR-11 says "a missing key fails closed (no silent switch to another provider, except FR-7's explicit subagent path)". A revoked or absent key in an Anthropic workspace is both. The falkor-chat `FallbackClient` chain would silently fall back. Required: state which wins. Suggested: FR-3 covers transient or capacity failures (credits, outage, rate limit); an authentication failure or missing key fails the step loudly, and FR-11's exception list names FR-3's fallback. AC-3 should then cover both cases.

**B2 — blocker (small) — FR-8 has no acceptance criterion.** Verified by mapping. FR-4, FR-8 and FR-9 have no AC. FR-4 and FR-9 are negative or trivial, so a one-line "unchanged" check is enough. FR-8 (tank switchable, default stays local) is real scope, and AC-1 only covers the default. Add an AC: "given tank pointed at an Anthropic model, a read-only check run completes and the transcript holds no key".

**M1 — major — FR-14 vs "no spend cap" (FR-10, Out of scope).** A bound "set before it starts" (AC-9) is a cap by another name, and the doc never says what is bounded (calls, tokens, wall-time, items). A designer can read it either way. Pin the unit: "a bound on number of API calls (or tokens) per campaign or sweep, supplied up front; this is a run limit, not a spend limit". Also say whether an interactive falkor-chat workspace or a workflow loop is meant to be bounded. As written, a looping workflow in a switched workspace is unbounded, and FR-14 mentions only campaigns and sweeps.

**M2 — major — FR-11 vs tank and subagents is not reconciled.** FR-8 says FR-11 "applies in full". `tank` has `bash` and `read` enabled (`opencode.json` `tools`), so any process env or config holding the key is reachable by the LLM-driven agent, which breaks "never into the context of any LLM-driven agent or tool". The same applies to FR-7's subagent path where children inherit the environment. AC-7's "subagent and child-process environments" check also collides with "held only by a closed list of components": the holders' own environments contain it. Reword AC-7 to "other than the closed list", and say whether FR-8 is deliverable under FR-11 or can be scoped back, since this is a real feasibility risk.

**M3 — major — reversing decision-log entries do not name what they overturn.** Root `AGENTS.md` requires a reversing entry to name the one it overturns.
- The entry "Keep the opus-generates / sonnet-labels split? → leave to design" loosens the stakeholder-set `docs/requirements/typed-decision-models.md` entry of 2026-10-31 "Generator opus, labeller sonnet". It names neither that entry nor the FRs it touches.
- The entry "Should decision-models ... use the Anthropic API?" cites the 2026-10-31 "Through Claude Code subagents" entry (good) but not the 2026-10-30 entry whose condition ("start local") is unrelated, so that is fine.
- The `typed-decision-models.md` log (Ready for design, plans 4–6 executed against it) carries no pointer to this partial supersession. Append one dated pointer line there (tico's own doc).

**M4 — major — stale text left from earlier interview turns.** Entry 3 says "flagged in Open questions", entry 4 says "model-bench's scope is now in question", the consult entry says "spend ceiling raised as Open question 4", yet `## Open questions` says "None". Entries are append-only, so leave them. Their pointers now dangle, so add a one-line note, or fold the facts into the entries that answered them (the 2026-11-03 "Will a spending limit..." entry answers the ceiling question). Also, entry "Do embeddings stay local" follows entry 6's "to be confirmed" out of order; harmless.

**m1 — minor — solution leakage.** FR-2 prescribes "the existing per-workspace override" and FR-10 "a hosted flag". A pin to the existing mechanism is defensible, since it is the stakeholder's stated intent ("configured through the default-and-fallback mechanism falkor-chat already has"). Keep it, but state the observable outcome ("the workspace's agent, steps and guard use Anthropic and nothing else changes"). The decision log's `x-api-key` / Bearer note correctly stays out of the FRs.

**m2 — minor — hosted model identity drifts.** FR-12 marks results hosted but does not require a pinned, full model id. An alias such as "latest" makes a stored run non-comparable over time, the same reproducibility concern FR-12 raises. Add: the stored model identifier is the exact version string the API reports.

**m3 — minor — role vocabulary collision.** "roles" in the Problem section means falkor-chat's fallback chains, and "roles" in FR-5 means model-bench packs. FR-5 lists `guard-judge` as a role while FR-6 lists "guard" as a decision point. In `model-bench/modelbench/roles.py` the guard-judge, join-in-judge and message-type-classifier are all roles. State whether FR-5's guard-judge and FR-6's guard are the same run, and use "fallback chain" for falkor-chat. As written, AC-4 and AC-5 could double count or omit the guard.

**m4 — minor — AC-7 positive control reaches only the error path.** A canary key cannot authenticate, so the run exercises the failure path (401 text, FR-11's error echo) but never a successful call's logging or usage recording (FR-10). Say AC-7 runs both a canary (error path) and one real-key success, or states the success path is covered by review.

**m5 — minor — data egress is unaddressed.** Switching a workspace (storefront shopper conversations), tank transcripts, or labelling inputs sends that content to a third party. FR-10 and FR-12 mark it "hosted" but nothing says which content may go. If the stakeholder accepts all of it, record that as a decision (one line). It costs nothing and closes a designer's question.

**m6 — minor — FR-13 and AC-8 baseline.** "Completes, as it does on the local model" does not say which conversation or how many turns. Reference the existing storefront acceptance scenario or give a minimum turn and tool-call count.

**Growth-stops check.** First pass over an untracked, first-version doc, so there is no prior text to compare. FR/AC and decision-log shape are compliant (the log is append-only; the 2026-11-03 "Reconsidered" entry names the entry it overturns). The only growth issues are the dangling pointers in M4. No restated facts: SERVER.md §1.8 content is cited, not copied.

### What's solid
- Intent, user stories, FRs and ACs map cleanly (except B2); FR-1/AC-1 is a clean regression guard.
- FR-4 and the model-bench embedder exclusion are consistent with the verified code and with Anthropic's lack of an embeddings model.
- AC-7 pairs a deliberate-plant control with the search, which is the right shape (it can both pass and fail). The enumerated leak surfaces (untracked files, ledgers, transcripts) are concrete. I confirmed `decision-models/docs/ledger.jsonl`, `model-bench/results` and `decision-models/campaigns` are not gitignored, so the security consult's leak paths are real.
- The out-of-scope list is tight, and the partial supersession of the 2026-10-31 decision is explicit and traceable.
- No contradiction with `docs/requirements/standard-local-model.md` (qwen stays the local default; fallback load on the one-slot qwen copy is a design note, not a conflict) or with `llm-provider-config.md` FR-13/16/17 (workspace hard cap is the mechanism FR-2 relies on).

### Open questions for the stakeholder
1. B1: on an invalid or missing key in a switched workspace, fail loudly or fall back to local?
2. M2: if FR-11 cannot hold for tank (it has a shell), is tank scoped back or accepted with a documented residual risk?
3. m5: any content that must not go to Anthropic?

## Pass 2 — 2026-11-03

**Scope.** The revised `docs/requirements/anthropic-api-adoption.md` (still untracked, Interviewing), read whole against Pass 1. Static; `K-056` and `K-080` references were grepped.
**Verdict: approve with suggestions.** No blocker remains. N1 and N2 are small text fixes I recommend before the Ready-for-design flip.
CPG: considered, not relevant — a requirements document.

### Pass 1 dispositions
- **B1 (FR-3 vs FR-11):** fixed in FR-3 ("missing, invalid or revoked key does not fall back"), consistent with FR-11. No AC covers the loud-failure case, though (N1).
- **B2 (no AC for tank):** superseded — tank dropped (FR-8, Out of scope). FR-8 and FR-9 now cite AC-1. FR-4 does the same, which is wrong (N1).
- **M1 (FR-14 bound):** fixed — FR-14 now says tokens, per run, not a budget across runs. A looping interactive workflow is still not addressed (N4).
- **M2 (FR-11 vs tank/subagents):** fixed by dropping tank. AC-7's "subagent and child-process environments" is still not reworded "other than the closed list" (N3).
- **M3 (unnamed split change):** fixed in substance — the entry "Consult: analyst" names the 2026-10-31 "Generator opus, labeller sonnet" entry. `typed-decision-models.md` has no Anthropic mention (grep), a deliberate "gains no pointer"; accepted.
- **M4 (dangling "Open questions" pointers):** fixed — the same entry says earlier references mean the questions answered by this log.
- **m1:** fixed (FR-2 observable switch, "expected carrier"). **m2:** fixed (FR-12 pinned version). **m3:** fixed in AC-4 (guard-judge and guard are separate runs); the "roles" vocabulary in Problem line 9 is unchanged, harmless. **m5:** fixed by FR-15. **m6:** fixed in form (AC-8), but see N5. **m4:** changed, but now unsatisfiable (N2).
- Current-state items from Pass 1 (the "by design" wording and the K-080 extraction note) are fixed (lines 12–13, FR-2).

### Stakeholder answers folded in (question → answer)
- Missing, invalid or revoked key → **fail loudly; only temporary failures fall back** (FR-3). Folded; the entry names the one it narrows.
- tank given its shell and file tools → **drop it** (FR-8, Out of scope). Folded; the entry names the one it overturns. Stale remnants: the tank user story is gone, and entry 81 still says "tightened FR-8", which is historical and fine.
- Unattended bound counted in → **tokens, set before each run** (FR-14, AC-9). Folded.
- Content barred → **real or private data barred** (FR-15, AC-10). Folded, but see N2'.

### New findings

**N1 — major — two coverage gaps.** (a) FR-4 (embeddings stay local) says "Satisfied by AC-1", but AC-1 is the no-configuration case. The risky case is an Anthropic-switched workspace, where the embedding kind must still resolve locally (the `embedding` kind is overridable; SERVER.md §1.8 `KINDS`). Add to AC-2 or a new AC: in a switched workspace, embedding calls reach the local model. (b) FR-3's new loud-failure branch has no AC. Extend AC-3: "given a missing, invalid or revoked key, the step fails with a clear message and no local model answers". Verified by mapping FR→AC.

**N2 — major — AC-7's success-path canary cannot pass as written.** The new sentence "the canary also runs on a successful call" cannot hold: a fake canary key never authenticates against Anthropic, so no call succeeds. Name how the success path is exercised, e.g. a local stub endpoint that accepts the canary and returns a valid response. That also gives the "can fail and pass" control root `REVIEW.md` requires. Inferred.

**N2' — major — FR-15/AC-10 leave the marker undefined.** "Workspace marked as holding real or private data": who marks it, and what is the default for an unmarked workspace? If the default is "not private", the guard fails open and the existing `ws` workspaces switch freely. State the default (suggest: a workspace is barred unless explicitly marked synthetic/demo, with the storefront demo and benchmark content pre-marked). Also say who enforces it at the switch (the override set path) and what happens to a switched workspace that later receives real data. FR-15's first sentence ("only synthetic... content is sent") also covers model-bench and decision-models inputs, but AC-10 tests only workspaces; add or scope it.

**N3 — minor — AC-7 environment clause.** Reword "subagent and child-process environments" to "other than the closed list's own". The API-path holders legitimately contain the key.

**N4 — minor — FR-14 scope.** It lists campaigns and sweeps only. Say that interactive or workflow-driven use in a switched workspace is unbounded by choice (no cap, per Out of scope), so a designer does not infer otherwise.

**N5 — minor — AC-8's script reference is unverified.** "The one K-056 validated" — K-056 is a defect/model-swap item (`falkor-chat/docs/HISTORY.md:2945`), not a named script. Name the actual script or scenario (candidates under `falkor-chat/scripts/`: `verify_salesperson.sh`; I did not confirm it is the multi-turn tool-use one).

**N6 — nit.** Line 19 has a stray `- - As the **lab owner**` bullet; the K-080 extraction fact is stated twice (line 13 and FR-2), so cite one.

### Growth-stops check
Growth is mostly new scope (FR-15, AC-10, four decision-log entries) and in-place rework (FR-3, FR-8, FR-12, FR-14, AC-3/4/7/8 replaced, not stacked). No prior text is retained alongside its replacement; the log is append-only and each reversing entry names its target. Findings are cited by ID only in the one consult entry (acceptable as a log line). The only restated fact is N6's K-080 duplicate. No growth finding beyond that.
