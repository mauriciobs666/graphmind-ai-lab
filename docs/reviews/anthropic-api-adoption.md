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

## Pass 3 — 2026-11-19 — the plan `docs/plans/anthropic-api-adoption.md`

**Scope.** The architect plan (active, untracked, first version), read whole against `docs/requirements/anthropic-api-adoption.md` (Ready for design), root `REVIEW.md`, `falkor-chat/REVIEW.md` and `model-bench/REVIEW.md` (`decision-models/` has none; no REVIEW.md line conflicted with this prompt). Static, plus reads of `falkor-chat/server/falkorchat/{transport,llm,modelconfig,retry}.py`, `repository.py` (override read/write), `model-bench/modelbench/{runner,fingerprint,lmstudio}.py`, `decision-models/{agents,decisionmodels/{audit,campaign,install,runner}.py,AGENTS.md}`, and the Anthropic models-overview page (fetched live). No suite run: nothing was implemented.
**Verdict: needs changes** — 1 blocker, 5 major, 5 minor, 1 nit.
CPG: considered, not relevant — a plan review; claims were checked by reading the cited files.

### Findings

**B1 — blocker — the decision-models `api-run` design contradicts the shipped agent definitions and pins (§3.6.1, §3.6.4, §3.6.5).** Verified in `decision-models/agents/dm-generator.md`, `dm-labeler.md`, `install.py:99-126,264`. (a) The plan sends the agent-definition *body* as the API system prompt and "the written request" as the user turn. The body is a tool-use script: "Your message names one request file. Read it… Use only the Read tool… Write tool on that answer path… reply `done`." A tool-less API call cannot do any of it, and would answer `done` with no items. The plan must say what the model is actually given (request text inline, the answer taken from the response text and written by `apirun.py` in the exchange format) and what happens to `promptSha256`, which hashes these files: a different prompt for the API path is a different dataset provenance, not "unchanged". (b) The pins are read from agent frontmatter at `install` (`model`, `effort`): today `dm-generator` pins `claude-opus-4-8`, effort `medium`, `dm-labeler` `claude-sonnet-5-5`, `high`. §3.6.5 chooses `claude-opus-5-5` for the generator, so §3.6.4's "refuses a model that differs from the recorded pin" refuses on every install made from today's definitions. State whether the API path has its own pin set/definition files, whether the pin or the definition changes, and the consequence for a campaign mixing both paths (one dataset, two generator models). Fix in §3.6; this is what an implementer would otherwise have to re-derive, and AC-6 depends on it.

**M1 — major — FR-10 is only partly met on the step record (§3.4.5, R6).** Tokens and `modelHosted` count only agent-loop `chat()`. FR-2 switches the guard judge and the responder `complete()`, which send to Anthropic with nothing recorded, so a step whose only hosted call is the guard shows `hosted` false and zero tokens; FR-10 says each Anthropic result records them, and "roughly" covers an undercount, not a missing hosted flag. Either count `complete()`/judge calls (they go through the same client; `ChatResult` simply isn't returned, so return usage out-of-band per call) or at least set `modelHosted` from the resolved chain, not from observed calls. Verified by reading `llm.py` (`FallbackClient.complete` returns `str`).

**M2 — major — existing retry/error plumbing is unaddressed (§3.4.1, §3.4.3).** `ProviderCallError` (`transport.py:66`) already carries `kind`, `status`, `detail` (the provider's error text) and `causes`; `retry.is_transient` switches on a closed kind set (`"http"`, `"connection"`, …) and treats unknown kinds as non-transient. The plan adds kinds `auth/billing/rate/overloaded/request` without saying (a) how they map into `retry.is_transient` (a 429/529 today would never be retried in place, and `fallback_exhausted` over such causes is judged non-transient), nor (b) that the wire adapter must leave `detail` `None`: `detail` is the provider text FR-10 forbids persisting. Add both to §3.4.1/3, with a test per (b) (`detail is None` and `str(exc)` free of a planted body text).

**M3 — major — `post()` can forward the key to a redirect target (§3.3 endpoint rule, §3.1).** The rule checks only the initial host. With `urllib`, a 301/302 on a POST is followed as a GET with all non-content headers copied (`HTTPRedirectHandler.redirect_request`, read in the stdlib source), so `x-api-key` goes to the redirect's host. Require an opener with redirects disabled (any 3xx → `request`), and name the test: stub returns 302 to a second loopback listener; the listener must see no key. Inferred from source, not run against a live redirect.

**M4 — major — the AC-7 leak sweep is last (U10) and the key code is reviewed late (§4, §3.3, §7).** (a) The plant-and-find harness is the check that can fail; building it after every writer (U2-U8) exists means leaks surface at the end. Move the harness skeleton (plant in each location, prove it finds the plant) into U1, and put "each new writer is swept" into U2/U3/U6/U8's done-conditions. Do not fold the harness into U10's own unit, where its declared set would be written to match what exists. (b) The security review gate is "before U2 merges", but the key loader, the endpoint rule and the live probe script are U1; the gate belongs before U1 merges (U1 is also where the probe handles a real key). (c) U10 puts `leak_sweep.py` in `falkor-chat/scripts/` though it sweeps three components; say why it is not at the repo root or in a neutral home.

**M5 — major — U2 done-conditions cannot be met as sequenced (§4).** §3.4 item 4 puts the private-flag refusal in `resolve()` under U2, but the flag and its read are item 6, U4; AC-10 and the "flag after switch" case live in U4. Either move the resolve-time private check to U4 or let U2 stub the port. Also U2's AC-11 test (7) covers the refusal only; AC-11's positive case (a switched workspace's embeddings resolve and run locally) has no named test.

**m1 — minor — bench hosted arm details (§3.5).** (a) Opus 5.5 thinking is always on and counts toward `max_tokens`; the packs' per-call `max_tokens` were sized for 4B local models, so U1's probe should include one pack-sized call and U7 must not start before the ML note fixes `maxTokens` per pack. (b) A run stopped at the bound writes no record (exit 6). A re-run with a larger bound is then an unrecorded selection against `model-bench/REVIEW.md`; record the aborted attempt (ledger line or results note), not a run record. (c) The new `hosted:chat` profile changes what a stored record means: per `model-bench/REVIEW.md`, add canonical vectors with expected values written independently of the profile table (one valid hosted record, one local-looking forgery each forbidden field), not only "forbidden-field tests".

**m2 — minor — spec handling.** The plan cites `SERVER.md` §1.8 and `QUERIES.md` §13 but names `DESIGN.md` only as "StepRun/WorkspaceConfig properties": cite the section(s). §2's first bullet restates live-spec content (resolution precedence, `KINDS`, step-record fields) beyond what the change needs; keep what the change depends on, cite the section for the rest. The plan leaves out whether `SERVER.md`'s own list of supported protocols (`only 'openai'` is also in the refusal text, `modelconfig.py:~537`) and the K-043 pointer in that message are updated.

**m3 — minor — stale code ranges (§3.5.3).** "read `runner.py` 457-830 and 1117-1200": the drivers are `_drive_single_call_items` 453-548, `_drive_decision_items` 584-686, `_drive_handoff_items` 861-993 and `_drive_conversations` 1152-1238; 1117-1200 begins inside `_turn_timings` (1112). Cite by function name. The substance (drivers take `lmstudio` and `baseline_residency`) is correct.

**m4 — minor — private-flag read failure is unspecified (§3.4.6).** If the graph read for `private` errors, `resolve()` must fail closed for hosted chains; say so and test it (a read that returns "not private" on error is the FR-15 guard failing open).

**m5 — minor — byte-equality tests that skip (§3.1).** The copies are guarded by tests that skip when model-bench is absent (matching `decision-models/AGENTS.md`). That is acceptable as precedent, but say which of the two copies runs it in CI-less practice: a drift between the falkor-chat copy and canonical is only caught from a checkout containing both. Add a planted-drift positive control (mutate one byte, expect red), per root `REVIEW.md`.

**n1 — nit.** `ChatResult` fallback on `complete()` is not recorded either (returns `str`); worth one sentence in §3.4.5 so R6 reads complete.

### Requirements conformance
- **Covered:** FR-1/AC-1 (§5 test 1), FR-2/AC-2 (§3.4.1-2,7; U3), FR-3/AC-3/AC-12 (§3.4.3; fatal vs fallback), FR-4/AC-11 (§3.4.4; positive case, M5), FR-5/AC-4 and FR-6/AC-5 (§3.5; guard-judge and guard decision point kept separate), FR-8/9 (excluded), FR-10 (partly, M1/M2), FR-11/AC-7 (§3.3, U10; M3, M4), FR-12 (§3.5.4-5, `require_pinned`), FR-13/AC-8 (U5 live plus stub), FR-14/AC-9 (§3.5.3, §3.6.3), FR-15/AC-10 (§3.4.6-7, enforced at resolution too: a good addition).
- **FR-7/AC-6:** designed, but as written not executable (B1).
- **No contradiction found** with a requirements decision; the two design-left items (models, split) are decided, the split provisionally pending the ML note. The one deviation worth stating: FR-7 says "when a key is available", the plan makes the choice explicit (`api-run` vs `run`); acceptable, and AC-6's "no key still runs via subagents" holds because `run` is untouched.

### Claims checked against code
Verified true: `SUPPORTED_PROTOCOLS = {"openai"}` (`modelconfig.py:73`), refusal at `:537`; `KINDS` is `{agent, step, embedding, guard}` (`:87`, no `extraction`); `_KIND_TO_OVERRIDE_KEY`; `resolve()` precedence and hard-cap behavior; `_build_llm` builds per call; `write_model_overrides` sets `None` as a clear (so read-modify-write is needed, as the plan says) and has only the four values (the `private` addition needs a separate write); `FallbackClient` catches every `ProviderCallError` (`llm.py:218`) and `replace()`s `fallback`; `ChatResult.model/fallback` exist; model-bench `RunConfig` has the `modelKey`/`candidateId` exactly-one rule; `fingerprint.py` documents additive profiles on schema 1; `test_every_stored_run_record_still_validates_unchanged` exists (`test_results.py:1535`); `CHILD_ENV_ALLOWLIST` (`runner.py:72`); "No hosted API client" (`decision-models/AGENTS.md:47`); `results/runs/` is tracked; no `*.key` rule exists in `.gitignore`; the storefront check names (`add_to_cart`, `view_cart`, `place_order`, `query_graph_data`) appear in `proof_defs.py`. Not verified: `anthropic-version` value, error-body shapes, the `thinking:{"type":"disabled"}` acceptance on Haiku 5.5 (U1's job). The Anthropic overview page confirms ids, prices, the pinned-snapshot note and Opus 5.5 "Adaptive (always on)"; it also says Sonnet 5.5 **rejects** `thinking: disabled` (Models API `thinking.types.disabled.supported` false), which matters to the ML note's labeller setting (§3.6.5), not to the plan as written.

### Growth-stops check
First version of an untracked document, so no revision growth to classify. It states the design, not a finding→resolution table; the only restated spec content is m2. Requirements are cited by number. No finding.

### Sequencing, risks, testability
U1's live-probe gate is sound in intent (it can fail: a contradicting probe edits §3.1-§3.2), but it needs a human with a key and gates six units; keep the wire/key/vector work separable from the probe so U6 and U8's offline work, which depend only on the wire module, are not blocked by the probe (state which of U2-U8 truly need the spike results: U2/U3/U6/U8 stub-based work does not). The stub-endpoint strategy gives most ACs a can-fail/can-pass control. The §5 test list names inputs and outcomes; add mutants only for the ones that guard config rather than code (test 1: spy on `socket.create_connection` plus a positive control that opens it when the env var is set). R2-R4 are real and honestly stated.

### Open questions
1. B1: does the API path get its own generator/labeller pins and prompt text, or is the subagent definition adapted? This is a design call for the architect (plus `data-scientist` for provenance), not the stakeholder.
2. Should the guard judge and responder count toward step tokens (M1), or is a stated floor plus a correct hosted flag enough?

**Recommended next step.** Architect revises §3.6 (B1), §3.4.1/3/5 (M1, M2), §3.3 (M3), §4 (M4, M5) in place; a short re-gate follows. The deep security pass on §3.3 stays a separate request.

## Pass 4 — 2026-11-19 — revised plan (4,581 words) with the security report and the ML note folded in

**Scope.** Plan re-read whole; the security report and ML note read for what the plan took and did not; B1's new design checked against `decision-models/decisionmodels/{install,finalize,audit}.py`, `agents/`; `retry.py`, `DESIGN.md` ~578, `services._check_models_resolvable` and `packs.py` re-read. Static.
**Verdict: approve with suggestions.** The Pass 3 blocker is closed in design; 1 new major (N1, a closure gap of the B1 fix itself), 2 minor, 1 nit. N1 should be folded before U8 is dispatched.
CPG: considered, not relevant — a plan review; claims checked by reading the cited files.

### Pass 3 dispositions
- **B1:** fixed in design (own `agents/api/` definitions, `install --surface api`, API-only pins, `apiDefinition` hash, new dataset version); the code-level closure has gaps, see N1.
- **M1:** partly fixed — `modelHosted` now comes from the resolved chain (§3.4.6), so the flag no longer depends on which calls returned a `ChatResult`; counting `complete()` tokens is returned to the stakeholder (§6), tokens stay a stated floor. Acceptable.
- **M2:** fixed — hosted errors reuse `kind="http"` with real `status`, `detail` always `None`, `fatal` added (§3.4.2). Rechecked `retry.py:45,75-96`: `TRANSIENT_HTTP_STATUSES = {408, 429}` plus 5xx, so 429/529 stay transient and 400/401/402/403 do not; an `http` kind with `detail=None` hits no marker path.
- **M3:** fixed — dedicated opener, any 3xx to `request`, `ProxyHandler({})`, tests with a second loopback listener (§3.3).
- **M4:** fixed — harness skeleton in U1a with the location set declared first; gate before U1a; location rationale given.
- **M5:** fixed — private check and `ws=None` moved to U4, AC-11 positive case in U2's done-list.
- **m1:** fixed (§3.2 truncation rule, `aborted.jsonl`, canonical vectors). **m2:** fixed (DESIGN ~578 matches `DESIGN.md:578`; SERVER §1.8 protocol statement named). **m3:** fixed (function names). **m4:** fixed (fail closed). **m5:** fixed (own SHA pins, planted drift). **n1:** fixed (§3.4.6 states `complete()` carries nothing out).

### New findings

**N1 — major — the B1 design is not closed against `install.py`/`finalize.py` (§3.6.1-2, U8).** Verified in the code:
- `finalize._prompt_hashes` (`finalize.py:245-256`) refuses unless `promptSha256` carries `agentDefinition` and `spawnTemplate` for `generator`/`labeler`/`driver` (`_PROMPT_KEYS`, `:80`). The plan says an API install records `apiDefinition` and "the subagent hashes are not recorded", so `finalize` would refuse every API campaign: "run `driver install` again". `finalize.py` is not in U8's file list.
- `generation.json` takes `model` from ingested runs' `run["model"]` (`finalize.py:229`) and writes the constants `SURFACE = "claude-code-subagent"` and `TIERS = {generator: opus, labeler: sonnet}` (`:72-73`, `:327`). An API campaign has no transcript to read the model from, a different surface, and a `fable` generator. Name the source of `run["model"]` (the `api-call` event or the answered model at ingest) and make surface/tier follow the recorded install.
- `install.definition_error` demands exactly each role's keys and pinned values; the API definitions carry `maxTokens` and none of `tools`/`maxTurns`, so the check needs a surface-specific key set. `_check_installed` also refuses stray files in `exchange/.claude/agents`; say what an API install copies (probably none, as there is no child).
- The refusal is one-directional: `api-run` on a subagent install is refused, but `driver run` on an API root, and `install` re-run with the other surface, are not stated.
- "Existing installs and the subagent `promptSha256` stay intact" holds by construction (`--surface` defaults to `subagent`, `fold` lets only the first event set pins), but nothing in U8's done-list can fail if it is broken: add a test that an install without the flag produces an `expectations` event byte-identical to today's (golden recorded from the current code before the change), the mutant being any new key leaking into it.
Fix: add `finalize.py` (and `labels`/`generation.json` surface and tier) to U8 with those done-conditions.

**m6 — minor — FR-11 is knowingly not met in full (§3.3, R4).** FR-11 says the key never enters the context of any LLM-driven agent or tool; the plan, honestly, only builds a tripwire. The "no hard control built" call is legitimate for the architect, but the deviation is the stakeholder's to accept: say in R4 that FR-11 holds only up to that residual, and that AC-7 tests the listed locations, not agent reach. §6 already lists the hard-control question; add the FR-11 wording to it.

**m7 — minor — `hosted-allowed-packs.json` (§3.5.1).** The external list is the right call (a `pack.json` marker would change the pack content hash that `packs.py` computes over the directory). It is an attestation file, so name its failure cases in U6's done-list: a pack edited after attestation (hash mismatch) must refuse, as well as an unlisted pack.

**n2 — nit.** The risk list starts at R2 (R1 is gone), and `F3`/`F11` finding IDs appear in plan text (citing by ID is allowed; just ensure the report file, which carries them, stays).

### Non-takes
- **No hard key control:** acceptable as a decision, stated plainly, with one human-run live check; see m6 for the FR-11 wording.
- **External pack attestation:** sound, see m7.
- **`modelHosted` from the chain:** sound; it is what Pass 3 M1 asked for as the minimum. One consequence to keep: a hosted-primary step that fell back to local shows `modelHosted` true and `modelFallback` true, so the two must be read together; SERVER §1.8 should say so at close.

### U1a/U1b split and gate
Sound: U1a is offline and carries every §3.3 test and the harness; U1b is the only unit needing the key and a human; the security gate precedes U1a. U2/U3/U4/U6/U8 depend only on U1a plus stubs, which the plan says. One ordering note: U8 needs U1b before its generator pin (Fable) and `maxTokens` are final, as the plan states; U8's offline work should not hard-code them.

### Growth-stops check
The plan grew from about 3,400 to 4,581 words (ML note excluded). Most growth is **new scope required by the folded-in reports** (loader/endpoint rule, replicates, `aborted.jsonl`, three-way stop map, API surface) or **rework in place** (§3.4.2, §3.6.1-2 replaced, not stacked). No finding→resolution table, no changelog, no superseded text left beside its replacement. Reported: (1) **restated facts**: pilot 0.6M / full 1.0M bounds, Opus `maxTokens` and thinking settings and the Sonnet-rejects-`disabled` fact are stated in §2, §3.2, §3.6.3 and the ML note, which "wins on method": keep each in one home and cite (§3.6.6's bound figures are the clearest copy); (2) §6's "Stakeholder questions are returned with this revision" is revision narrative: an open question stays until answered, but needs no "with this revision" framing. No other growth finding.

### Open questions
The five stakeholder questions in plan §6 stand (complete() token counting, workspace-override-only hosting, two packs' synthetic attestation, bounds and spend, hard key control); none is mine to settle.

## Pass 5 — 2026-11-19 — plan revised for Pass 4 (4,888 words)

**Scope.** Plan §3.6, U8, §6 re-read against `decision-models/decisionmodels/{install,finalize,campaign,audit}.py`; requirements doc (edited 2026-11-19) re-read against the plan. Static.
**Verdict: approve with suggestions.** 0 blocker, 1 major (N2, residual of the N1 fix), 1 minor, 2 nits. N2 should be folded before U8 is dispatched.
CPG: considered, not relevant — a plan review; claims checked by reading the cited files.

### Pass 4 dispositions
- **N1:** mostly fixed. `finalize` is now in U8 and surface-aware (`_prompt_hashes` per surface, model from the answered model, `SURFACE`/tier following the install; rechecked `finalize.py:72-73,80-82,229,245-256,327`: no other consumer of `tier`/`surface` values in `decision-models` or `model-bench`, grep); `api_definition_error` replaces the shared check; `install-api` leaves `install.install` untouched; four refusals; golden `expectations` event with a named mutant. One residual: N2.
- **m6:** fixed — R3 states FR-11 holds up to the residual and AC-7 tests listed locations; the requirements log (2026-11-19, tripwire only) matches.
- **m7:** fixed — U6 done-list refuses an unlisted pack and a content-hash mismatch.
- **n2:** fixed — risks renumbered R1-R7; finding IDs no longer cited in plan text (rechecked by grep).
- **Growth item 1 (restated facts):** fixed — §3.6.3 and U7/U9 now cite ML note §3.1/§3.3/§3.4 as the one home; the bound figures are gone from §3.6.6.
- **Growth item 2 (revision narrative):** fixed — "returned with this revision" replaced by a plain open-questions list.

### New findings

**N2 — major — an API root still passes through code that assumes the subagent expectations (§3.6.2, U8).** Verified:
- `Campaign._expected` (`campaign.py:2089-2107`) reads `spec["agentType"]`, `spec["spawnTemplate"]` and `expectations["guardCommand"]`. The API `expectations` event has none of them (roles carry model/effort/maxTokens only, no guard, no spawn template). `issue()` and the serve path call `_expected`; on an API root `next()`/`issue()` raises `KeyError` unless `_expected` is surface-aware or `api-run` is stated to bypass it. The plan lists `campaign.py` only for "events".
- `generator_template_hash` (`campaign.py:780-788`) hashes `agentDefinition`, `spawnTemplate` and `requestTemplate` of the generator; `finalize.hash_refusal` (`:167`) relies on it to demand a `driver resolve` between two hashes. On an API root the first two are absent, so an edit to an API definition mid-campaign changes no hash and raises no template halt; only `requestTemplate` would. Include `apiDefinition` for an API root.
Fix: add both to U8 with done-conditions that can fail: an API-root `issue()` against a fixture; an `apiDefinition` edit between two installs producing a second hash and a template halt (mutant: the hash ignoring `apiDefinition`).

**m8 — minor — FR-3's new wording versus the plan (requirements, not plan).** FR-3 (edited 2026-11-19) says any 4xx other than rate limit fails loudly, yet lists "credits exhausted" as falling back, and the plan (§3.4.2) classifies 402 and credit-message 400s as `billing`, which falls back. The plan is consistent with the intent; the sentence is not. Ask `tico` to say "a 4xx other than rate limit or a billing/credit failure". No change to the plan.

**n3 — nit.** R3/R5 narrate stakeholder decisions ("the stakeholder decided/confirmed"); state the design and cite the requirements log entry instead.

### Requirements consistency
FR-2 (override is the only carrier) matches §3.4.4; FR-10 (floor plus correct flag) matches §3.4.6/R5; FR-11 (tripwire only) matches §3.3/R3; FR-15 (two packs confirmed, attested) matches §3.5.1 and U6; Opus effort medium matches §3.2. No contradiction found other than m8's wording. The three open questions in §6 are not counted.

### Growth-stops check
4,581 to 4,888 words (+307). The growth is new scope (`install-api`, surface-aware finalize, golden event, surface refusals, `hosted-allowed-packs` failure cases) and in-place replacement of the earlier `--surface` text; no changelog, no resolution table, no stacked superseded text. §3.6.2 is now one very dense item; it is one home for the surface rules, but split it if it is edited again. Only n3 as a narrative remnant.
