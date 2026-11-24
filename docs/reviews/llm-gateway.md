# Common LLM gateway: requirements review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/requirements/llm-gateway.md` · **Last updated:** 2026-11-22 · **Item:** `GAL-F0046`

## Pass 1

### Scope & verdict

Reviewed `docs/requirements/llm-gateway.md` (GAL-F0046, Interviewing, untracked, 2,415 words) against the superseded `docs/requirements/anthropic-api-adoption.md` and its plan and security review, and against the code and docs it describes: `falkor-chat/docs/SERVER.md` §1.8, `model-bench/modelbench/lmstudio.py` and `model-bench/docs/DESIGN.md`, `decision-models/AGENTS.md`, `cpg/ccg/front/interpret.py`, `opencode/agents/tank/README.md`, and `docs/requirements/connected-code-graph.md`. Root `REVIEW.md` and the root `AGENTS.md` document conventions applied. Not run: nothing executable is under review.

**Growth-stops check:** a first, untracked draft with no earlier revision, so there is no rework growth to report. All 2,415 words are new scope. The check applies from the next revision. One housekeeping item: six decision-log entries still say "(FR/AC: TBD)" (m4).

CPG: not applicable — requirements document review; the code checks were by reading, and no CPG was queried.

**Verdict: needs changes.** Blocker 2 · major 7 · minor 7 · nit 2. Every finding was verified by reading the cited file, except where it says "inferred".

### Findings

#### Blockers

**B1. The key-handling requirement the stakeholder carried was weakened, and it now contradicts FR-20 and FR-11.** Verified by reading.
- Adoption FR-11 required: no key on a command line, in a prompt, or in an LLM agent's context; **held only by a closed list of components**; error text never echoes credentials; fixtures use obviously fake keys; a missing key fails closed.
- Gateway FR-8 keeps "never in repo/config/results/logs/graph/fixtures", "not passed to an LLM-driven agent", rotation, and fail-clearly. It drops the closed holder list, the command-line and error-text rules, and the fake-key fixtures.
- Adoption AC-7 searched named places (tree including untracked files, git history, results, ledgers, transcripts, logs, graph records, tracebacks, fixtures, child-process environments). It required a planted-canary positive control and a canary through a successful call against a local stand-in. Gateway AC-5 is one sentence with no places, no positive control, and no success-path run. By `REVIEW.md` ("a check that cannot fail"), AC-5 as written cannot go red.
- FR-20 ("one source of truth for URLs and keys, used by every component") plus FR-11's in-process library mode means every adopting process holds the key. `ccg` ships an agent-facing MCP server, and `decision-models` runs LLM children with `HOME` in their environment. The security review `docs/reviews/anthropic-api-adoption-security.md` F1 (agent-readable key; broker or `denyRead` as the hard control) and F2 (key text leaking via `str(exc)`; reproduced) are not carried.
- **Fix:** restore the closed-holder-list rule, the command-line/prompt/error-text rules and the fake-key fixtures. State that in library mode the holder list is the adopting processes, and say what keeps an agent-reachable process off it. Rewrite AC-5 as AC-7 was written: named places, planted positive control, successful call via a stand-in endpoint. Carry the F1 and F2 outcomes as requirements, or record a decision to defer them (a residual risk was already accepted on 2026-11-19, which `llm-gateway.md` does not cite).

**B2. FR-3, FR-6, FR-12 and AC-4 do not carry ccg's confidentiality contract and contradict how ccg reaches its model.** Verified in `cpg/ccg/front/interpret.py:3-11, 116-135` and `docs/requirements/connected-code-graph.md` FR-9/AC-8.
- ccg FR-9: "the machine" is an explicit allow-list of local endpoints, and anything not on it counts as hosted. Local-only is **sticky**: in a system containing any local-only repository, **every** model call is local. Under local-only, **nothing derived from repositories** leaves the machine, including error text.
- Gateway FR-15 adds a generic OpenAI-compatible endpoint, which can be local or remote. "Hosted", "off the machine" and "local-only" are never defined. FR-6/AC-4 ("nothing leaves the machine") needs the allow-list definition, or a misconfigured remote OpenAI-compatible URL is "local".
- FR-12 says the interpreter reaches its model "only through the gateway". `interpret.py` states "through the egress broker and nothing else" (`ccg.broker.client`, with per-call `payloadClass`/`codebases` and a refuse decision). The requirement does not say whether the gateway sits behind the broker, replaces it, or both enforce policy, nor which is authoritative. Two enforcement points with different answers is exactly the configuration slip that FR-3 exists to prevent.
- Stickiness is not carried. AC-2 tests only a single local-only codebase, not a mixed system where a hosted-default codebase's call must still go local.
- **Fix:** define "hosted" as any endpoint not on a per-deployment local allow-list, applying to every provider kind. Add a requirement and AC that, for ccg, the egress broker remains the egress enforcement point and a gateway call is a brokered call, or state the opposite as a stakeholder decision. Add stickiness to FR-12 and AC-2 (mixed-system case).

#### Major

**M1. FR-4, FR-18 and AC-1 conflict, so AC-1 is untestable.** Verified in `interpret.py:45-46, 204-208`.
- ccg sends `temperature, top_p, top_k, repeat_penalty, frequency_penalty, presence_penalty, max_tokens, seed` and a grammar-enforced `response_format` json_schema. FR-18 requires `seed`, schema and grammar passed unchanged.
- AC-1 requires switching ccg to Anthropic by configuration with "no ccg code change". From my knowledge of the Messages API (inferred, not re-checked against the docs) it has no `seed`, `repeat_penalty` or grammar mode, so those cannot pass through. FR-4 says an unsupported setting is "reported, not silently dropped", but "reported" is undefined: does the call proceed, warn, or fail?
- **Fix:** specify the outcome per setting class (a decoding-determinism parameter the provider lacks: fail, or proceed with a record flag). State which wins for ccg's sampler and whether the hosted arm is marked not-reproducible (FR-9). Name a call-site capability declaration. Re-word AC-1 to say what the caller sees.

**M2. FR-3 omits the fail-closed and carrier rules the security review found missing in the adoption plan.** Verified in `docs/reviews/anthropic-api-adoption-security.md` F3 and F4.
- Hosted use must be allowed only when set by the owner-controlled scope, never by a request-supplied or workflow-supplied model. Adoption FR-2 said "a workflow definition cannot name a hosted model directly"; it is lost (see M4). FR-2's "a run" override scope has no carrier rule.
- A private or local-only flag that is absent, ill-typed or unreadable must be treated as refuse, and must be read at call time, not from a snapshot taken at the start of a run. Neither is stated.
- The default for an unflagged scope is unstated. The adoption said unflagged is switchable. The gateway says "set local-only or flagged private" and leaves the default open.
- "No agent or running process can lift one" cannot be met literally, because the component's own settings editor is a running process. AC-2 tests only a narrower-scope override, not the lift path.
- **Fix:** add the three rules above and an AC per rule. Say who "the owner" is operationally, for example a process the LLM-driven agent cannot write to.

**M3. FR-16 conflicts with guarantees in model-bench and tank, and is not needed by the first delivery.**
- model-bench: `lmstudio.py` docstring ("No `unload`/`ps`... the harness cannot force a cold state, by design"), `DESIGN.md` §3.2 and ~920 ("the runner loads the arm's absent models itself, so no scored item is a JIT load"), and `fingerprint` including `loadedContextLength`. A decision-point pack performs an **explicit** load. FR-16's two modes (silent on-demand load, or strict already-loaded-or-fail) have no third "caller-directed explicit load, recorded" mode, which is the one model-bench actually uses.
- tank: `opencode/agents/tank/README.md` says the preflight only reads, "never load it while `lms ps` already lists it", and refuses k copies or a context or parallel mismatch; the two setups "cannot share one copy". A gateway that loads on demand "per configuration" (context length, parallel, GPU, and the 6 GB card limit) can create the duplicate copy tank refuses. tank is out of scope, but it shares the one LM Studio instance, and the shared state is not mentioned.
- ccg takes "whatever the endpoint reports" (`bound_model`) and loads nothing, so FR-16 is a follow-on need dressed as first delivery (see M6).
- The decision-log entry at line 93 says the tension "is raised in Open questions", but the Open questions section does not contain it (m4).
- **Fix:** state the load configuration FR-16 uses, who unloads or evicts, and the coexistence rule with other LM Studio clients. Add the explicit-load mode. Fix AC-13 so strict mode also asserts the recorded loaded state. Mark FR-16 follow-on, or justify it for ccg.

**M4. Adoption requirements are lost, and the decision log does not record the loss.** Verified by diffing the two FR/AC lists. The log's 2026-11-22 "Which adoption rules become gateway-wide?" entry names four groups only, and nothing records the rest as dropped.

| Adoption item | Gateway | Recorded? |
|---|---|---|
| FR-13 / AC-8 storefront multi-turn on Anthropic ("required, from the first release" 2026-11-03) | gone | no |
| FR-2: hosted only via the workspace override, workflow definitions cannot name it; ingestion extraction stays local (K-080) | gone | no |
| FR-4 / AC-11: embeddings stay local (Anthropic has none) | silent; FR-1/FR-2 allow routing any kind to any provider | no |
| FR-10: "nothing else about the call", incl. no Anthropic organization or workspace id | FR-5 omits the org id, and FR-26 now permits opt-in text | FR-26 decided; org id not |
| FR-15: synthetic-only rule, owner attestation of packs | gone | no |
| FR-5/6, AC-4/5: hosted model-bench arms, the original aim, "score Anthropic next to local" | deferred into FR-14 with no sequence | no |
| FR-7 / AC-6: decision-models generates and labels via the API | now "decided at its follow-on" | yes, but it does not name the 2026-11-03 entry it overturns (growth-stops rule: a reversing entry names its target) |

The stakeholder's stated aim was experiment and compare. Under the gateway plan the immediate experiment (model-bench hosted arms) is delayed until a follow-on phase, which may be a deliberate trade but must be a recorded decision. **Fix:** add a logged decision per row (carry it, defer it with a trigger, or drop it), or a short "not carried from adoption" list in the Out of scope section.

**M5. FR-13 and AC-9 rest on a stale premise.** Verified in `docs/requirements/connected-code-graph.md` decision log, around line 517: "2026-10-25 — FR-9 reversed: previously local-only with a local LLM allowed (2026-10-22). Now **local or hosted, configurable**". Per-codebase default plus per-run override is already the contract, and AC-8 already speaks of "a hosted-default codebase". The 10-22 local-only decision no longer stands, so there is nothing to "reconcile". What may genuinely need an edit is AC-24's sentence that the target applies "to the model that actually interprets" (already hosted-aware) and the interpreter's local-only prose. Also, ccg is `Ready for design` and its plan is at P1 gate rehearsal. FR-13 reopens that document, so it should say which ccg sections change and who owns them. **Fix:** drop FR-13/AC-9, or rewrite it to name the concrete ccg text that blocks hosted use (if any) after reading FR-9/AC-8/AC-24 in full.

**M6. Phasing is undefined and the statistics tooling should be split out.**
- Only FR-12/FR-13 are tagged "first delivery" and FR-14 "follow-on". FR-15 to FR-26 have no phase, though the first delivery is "ccg only". FR-16, FR-17 (LM Studio per-run stats and the decisions endpoint, which are model-bench needs) and FR-21 to FR-26 read as first-delivery scope by default.
- FR-21 to FR-26 are about 40% of the functional requirements, plus AC-15/16/17. They are a separate capability (a usage-analytics product) with their own unresolved decisions: where records live and for how long, retention, who can read them, alert delivery. None is stated. FR-5's record (model, fallback, hosted, tokens) lacks what FR-21 filters on: component, scope, call site, latency, outcome and failure class, price version. FR-26 adds times and outcome, so FR-5 and FR-26 should be one record definition.
- **Fix:** keep FR-5 and FR-26 (the record contract) in the gateway document as the one record definition, and move FR-21 to FR-25 to a new requirements document under its own slug (for example `llm-gateway-usage-stats`), with the family rule applied. Tag every remaining FR first-delivery or follow-on.

**M7. FR-7 and FR-19 are incomplete and contradict the decision log.**
- FR-7's temporary failures (credits, rate limit, outage) are hosted-provider concepts. Local and OpenAI-compatible failures (LM Studio down, model not loaded, timeout, 5xx, context overflow, HTTP 400) are unclassified. ccg already has a rule: one retry on a transport failure or 5xx (`interpret.py:RETRIES`, `_send`).
- FR-19 ("a caller that opts out...") makes retry and fallback default-on. The log entry at line 98 says "retry/fallback opt-in per caller". The two cannot both hold. FR-2 also configures chains per call site with no caller action, and AC-3 shows fallback without any caller opt-in.
- Retry (same provider) is never defined apart from fallback (next provider).
- **Fix:** classify failures for every provider kind. Define the default and where opt-out is set, and reconcile the log entry. Say how ccg's one-retry rule maps onto it.

#### Minor

- **m1. FR-10 is under-specified relative to adoption FR-14.** "Run", "unattended", the token basis (input plus output), who sets the bound, and overshoot are undefined. With the library in several processes and the service in one, the run identity needs a source. Carry adoption's "bound limits one run, interactive use deliberately unbounded" wording.
- **m2. FR-24 and FR-23.** Alert delivery (where, by whom) and evaluation (when a short-lived in-process library has no daemon) are unspecified, so AC-16 ("produces an alert") has no observable. FR-24 is consistent with the out-of-scope cross-run budget (a notify is not an enforce), but a spend threshold is a cross-run aggregate, so the stated distinction is the only thing separating them and deserves one sentence. FR-23 "on the same component's real traffic" compares different workloads unless the same inputs are replayed. State the comparison basis.
- **m3. AC coverage.** No AC for FR-4, FR-9 (a moving alias is refused), FR-17, FR-21's filters, FR-22 (the "defined interface"), or FR-26's positive case (text opt-in works). AC-10 ("a recorded follow-on unit") tests a process artifact, not behavior. FR-14's "no loss of today's behavior" has no checkable must-keep list. AC-13 omits strict mode's recorded loaded state.
- **m4. Housekeeping.** Open questions contains one open item and then "None open", contradicting itself. Log lines 93 and 96 cite Open questions that no longer exist. Six entries end "(FR/AC: TBD)". FR numbers are out of order (FR-25, 26, 22, 23, 24, 14), and AC-10 follows AC-17. The Supersedes pointer sits on the requirement only. The plan, ML note, coordination ledger and two reviews of the adoption stay `active` with no stated disposition, and the log should say so.
- **m5. FR-11 is solution language.** Library versus service is a decided stakeholder question (logged), but the requirement should state the outcome: usable in-process and from another process, with the same configuration and record (AC-7 already does). "Common core" and "standalone service" are design.
- **m6. FR-15, "generic OpenAI-compatible endpoint".** ccg's `LOCAL_ENDPOINT` assumes LM Studio's `/api/v0/models` fingerprint read, and FR-17 covers provider-specific detail. Say whether provenance fingerprints (ccg's cache key) are part of the call record, or the ccg cache and `samplingHash` break when the provider changes.
- **m7. Sequencing risk.** ccg's gate rehearsal is in flight (`git log`: P1-32b, "dress rehearsal dispatched"), and the interpreter configuration is frozen at the gate. Moving it onto the gateway mid-gate changes the measured system. State the order: the gateway adoption lands after the gate, or the gate is rerun.

#### Nit
- FR-9 and AC-1: the pinned-model rule has no AC naming a refused alias.
- FR-25 and AC-15: "tokens times price" ignores separate input and output prices; state the formula.

### What's solid
- The four chosen adoption groups are present and traceable by number: local default (FR-6), fallback only on temporary failures with key and request errors loud (FR-7, AC-3), pinned and hosted-marked (FR-9), token bound and private block (FR-10, FR-3). The gaps are in the detail around them (B1, M2, m1), not in their presence.
- Existing-component claims check out: `SERVER.md` §1.8 describes the `ModelGateway` resolution seam; `decision-models/AGENTS.md` line 56 has "No hosted API client"; `lmstudio.py` is LM Studio only; ccg's interpreter calls a local model through the broker. The Intent, scoping to chat completions first with embeddings deferred, and the CLI-agent exclusion are well reasoned.
- FR-18 (no injected decoding parameters), FR-20 and the metadata-only records (FR-26, AC-17) are testable and fit the stakeholder's confidentiality aim.

### Open questions (for the stakeholder)
1. Is the egress broker kept as ccg's egress enforcement point, with the gateway behind it, or does the gateway replace it (B2)?
2. For an unsupported decoding setting such as `seed` on Anthropic, does the call fail or proceed marked not-reproducible (M1)?
3. Is deferring the hosted model-bench experiment to the follow-on phase intended (M4)? Is storefront-on-Anthropic dropped or deferred?
4. Should the statistics tooling be a separate requirements document and phase (M6)?
5. Is FR-16's on-demand loading wanted for ccg's first delivery, given that ccg does not load today and tank's preflight refuses duplicate copies (M3)?

### Proposed REVIEW.md lines
None. No recurring class beyond what root `REVIEW.md` already requires.

## Pass 2

Scope: reworked `docs/requirements/llm-gateway.md` (3,251 words) and the new `docs/requirements/llm-gateway-usage-stats.md` (692 words, GAL-F0047). Both read in full. CPG: not applicable — requirements review; the one code check (ccg broker, below) was by reading.

**Verdict: needs changes** (no blocker; four majors, two of which need a stakeholder answer before design). Pass 2 findings: blocker 0 · major 4 · minor 9 · nit 3.

### Pass 1 dispositions
- **B1** fixed with a residual (N1): FR-8 restores the closed holder list and the command-line, prompt, error-text and fake-key rules; AC-5 has named places, a positive control and a stand-in endpoint. Not carried: the specific stdlib-exception leak form (security F2), only the generic "error text contains no key".
- **B2** fixed with a residual (N4): "hosted" defined by allow-list (definitions), sticky rule in FR-3/AC-2, broker kept (FR-12, AC-21), AC-4 covers a remote OpenAI-compatible URL.
- **M1** fixed: FR-4 and AC-1 now fail the call naming the setting. Residual m3 ("required" undefined).
- **M2** fixed in text (FR-2 owner-only carrier, FR-3 call-time read, AC-2). A new internal contradiction (N2).
- **M3** fixed: three modes, no second copy, AC-13. Residual m4.
- **M4** fixed: FR-14 lists the deferred adoption items, log 105 records the defer and names the decision-models entry it overturns, org id now in FR-5. Residual m6.
- **M5** fixed: FR-13/AC-9 dropped; the logged basis is the ccg 2026-10-25 entry (rechecked). Log 108 records it as a decision before the stakeholder confirmed it (m5).
- **M6** fixed: stats split, phase tags on every FR, FR-5 fields now cover the stats filters. Residuals N3 and m7.
- **M7** fixed: FR-7 classes for all providers, FR-19 retry vs fallback, ccg's one retry kept. Residual: log 97 still says "opt-in per caller" (m6).
- **m1** fixed (FR-10, AC-6; the "thinking included" part has no AC). **m2** fixed in the stats doc but still open there (m7). **m3** AC-18 to AC-21 added; AC-10 still checks a process artifact. **m4** Open questions fixed; still dangling or open: six "(FR/AC: TBD)" entries (lines 82-87), refs to "Open questions" in entries 92, 95, 96. **m5** fixed (FR-11 outcome wording). **m6** not fixed (fingerprint/cache-key provenance still undecided). **m7** fixed (FR-12, log 110). **Nits**: AC-18 and stats FR-5's input/output prices fixed.

### Stakeholder answers folded (question → answer, checked present)
Broker → gateway sits behind it (FR-12, AC-21, log 103). Unsupported setting → call fails (FR-4, AC-1). Defer model-bench hosted arms, storefront, decision-models generation → FR-14. Split stats → new document. FR-16 in first delivery, explicit mode added → FR-16, AC-13. Drop FR-13/AC-9 → pointer lines, but unconfirmed (m5). ccg hosted sampling profile, not directly comparable → FR-4, FR-12, AC-1. ccg adopts after the P1 gate → FR-12. Unload and second-copy rule, unflagged scope → log 111; FR-3 and FR-16 carry the second-copy and unflagged parts, the unload rule is in the log only (m4).

### Growth-stops check
2,415 → 3,251 words (+836), while the statistics text (about 260 words of FR, plus AC and log) moved out. Judgment: **mostly new scope** (stakeholder-confirmed rules: sticky local-only, three loading modes, failure classes, broker behind, record fields, AC-17 to AC-21). **Rework growth to remove:** (1) log entries 101 and 102 are a finding-to-resolution disposition list and a verdict-count line; the rule forbids both, and the review's Pass 2 is their home. (2) FR-14 carries history ("which would overturn the adoption requirements' 2026-11-03 decision-models entry, the 'no hosted client' rule being decided then"); the rationale belongs in log 105, which already holds it. (3) FR-14 restates adoption requirements in prose; the adoption document remains readable, so cite its FR/AC numbers (FR-2, FR-4, FR-13, FR-15, AC-8, AC-11) instead of paraphrasing, which also gives AC-10 something checkable. The log is 1,006 words (31% of the document). It is an append-only log, which is allowed, but entry 107 does not name entry 96 ("two rules") and FR-19 does not name entry 97 as overturned (m6). The deferred-requirements list itself is within the rules as current scope.

### Stable numbering
Acceptable, and necessary: log entries 98 to 100 and the stats document cite the old numbers, so pointer lines keep them resolvable. Improve the pointers: FR-21 to FR-25 and AC-15/16 should map old to new (FR-21→stats FR-1, FR-22→2, FR-23→3, FR-24→4, FR-25→5; AC-15→stats AC-1, AC-16→stats AC-5). Today they say only "moved". Place the AC-15/16 pointer in order. The stats document's "former FR-21 to FR-25" wording in its log is fine.

### New findings

**N1 (major). FR-8/AC-5 demand more than the repo can deliver and conflict with FR-11 in-process mode.** FR-8: key held only by processes "none of them reachable by an LLM-driven agent"; AC-5: "an LLM-driven agent process cannot read the key". The adoption decision of 2026-11-19 accepted a user-scope tripwire with residual risk "stated, not closed", and security F1 showed `.claude/settings.json` allows bare `Bash` and `Edit(**)`, so no tracked control makes the key unreadable. FR-11's in-process mode puts the key in the caller's process, and ccg's process is reached by agents through its MCP server. So the AC is either unsatisfiable or it silently mandates a hard control (a broker or separate OS user) as a requirement. Verified by reading both documents and the F1 text. Fix: state which it is. Either require the hard control (the broker is then the only key holder, see N4) or restate FR-8/AC-5 at the accepted level (never in an agent's context, tripwire control) and cite the 2026-11-19 decision.

**N2 (major). FR-3 contradicts itself on a missing flag.** "A flag that is absent, unreadable or ill-typed counts as refuse" versus "A scope with no flag may use hosted only where owner configuration allows it" (the confirmed answer, log 111). Both describe a scope with no flag. AC-2 says "unreadable or absent from a restricted scope", and "restricted scope" is undefined. Fix: separate "never set" (allowed where owner configuration allows) from "set but unreadable or ill-typed, or the read failed" (refuse), and define the term the AC uses. The lift path ("only owner configuration reaches") has no AC.

**N3 (major). One record per call loses per-attempt data that the reliability reports need.** FR-5: "every call yields one record... whether a fallback or retry occurred and why", with the answering model. Stats FR-1 asks for "failure rates of hosted versus local calls" and "how often and why a fallback fired". A call that fails on the hosted provider and then succeeds locally records only the local answer, so the hosted failure's latency and failure class are lost, as are the tokens billed on a failed hosted attempt (cost under-count). Fix: one record per attempt, linked by a call id, or an attempt list inside the record; and cost counts every billed attempt. Add an AC.

**N4 (major). FR-12/AC-21 leave ccg's policy sources and topology unresolved.** Verified in `cpg/ccg/broker/policy.py` (rules R1, R5, R6, R7, `P2-OVERRIDE-REFUSED`) and `server.py`: the broker already decides local-only, sticky and hosted per codebase from the system manifest, and "the exchange the broker makes after an allow" is the broker's own (`structured.py`). So (a) FR-3's "enforced by the gateway" and the broker's own policy are two engines with two configurations for the same flag, against FR-20's single source; the requirement should say which one holds the local-only flag for ccg, and that the gateway reads it rather than re-declaring it. (b) AC-21 says a refused call reaches "the broker's policy... with the gateway not reached", which contradicts "behind the broker". If the gateway builds the request and the broker sends it, the gateway is always reached first. (c) A hosted call needs the key header on a broker exchange; whether the key crosses the broker socket, is audited, or is injected by the broker is open, and it is the same question as N1. Fix: state the call path (ccg code, gateway, broker, endpoint) and the owner of each decision; rewrite AC-21 to match.

**Minor**
- **m1.** FR-12 contains a schedule sentence ("adopts the gateway after its P1 gate closes"). It is untestable; move it to the phasing line and the log.
- **m2.** FR-14 and the decision log state history inside FR text (see growth check).
- **m3.** FR-4: "a setting the call requires" is undefined. State whether declared-in-profile means required, or give an explicit flag per setting.
- **m4.** FR-16: the confirmed "never unloads a model it was not asked to" rule appears only in log 111, not in the FR or an AC. Explicit mode versus "never creates a second copy": when a model is loaded at a different context, the outcome (refuse, reuse) is unstated. Loading goes through the broker as a new target path (`/api/v1/models/load`), which the broker's path rules must permit; say so.
- **m5.** Log 108 records a drop as a decision on "no objection yet". An unanswered stakeholder question belongs in Open questions until answered, and the Ready-for-design flip needs it answered.
- **m6.** Log 105 says "Reverses nothing" and then names an overturned entry; 96→107 and 97→FR-19 are unnamed reversals.
- **m7.** Stats document: (a) no phase tag, though M6 asked for one and log 42 says "own phase"; (b) FR-3's "same recorded inputs replayed" basis needs prompt text, but the gateway records metadata only and text only on opt-in, never for private or local-only scopes; the dependency is unstated; (c) AC-5 refers to "the stated delivery point", which is still an open question, and AC-4 only checks that a basis is named; (d) "a defined interface" (FR-2) is not defined; (e) the price table is read at report time, so historic cost changes when prices change; say which. Solution language: none found.
- **m8.** AC-12 and AC-3 do not test retry (FR-19's second half) or ccg's one retry; AC-6 does not test "thinking included".
- **m9.** FR-1/FR-15: embeddings stay local until the follow-on, but nothing refuses an Anthropic embeddings call in the meantime; state that embeddings are not available at first delivery.

**Nit**
- Pointer lines AC-15/16 sit after AC-21 (see Stable numbering).
- Log 101 and 102 duplicate each other and the review (remove, see growth check).
- The Problem section names who flips the adoption plan documents ("architect, `teco`, `analyst`"); say so once in the log instead.

### Open questions for the stakeholder
1. N1: is the key requirement the hard control (the broker is the only key holder) or the accepted tripwire level?
2. N4: for ccg, which holds the local-only flag, the system manifest read by the broker or gateway configuration?
3. N3: one record per attempt, or per call with an attempt list?
4. m5: confirm that FR-13/AC-9 are dropped.

### Proposed REVIEW.md lines
None.

## Pass 3

Scope: `docs/requirements/llm-gateway.md` (3,619 words) and `docs/requirements/llm-gateway-usage-stats.md` (764 words), both read in full; the broker claims were checked in `cpg/ccg/broker/{__init__,server,structured,policy}.py` and `docs/plans/connected-code-graph-egress-spec3.md`. CPG: not applicable — requirements review; code read directly.

**Verdict: needs changes**, narrowly: no blocker, one major (N5, which needs a stakeholder answer), eight minors, two nits. **The gateway document may not flip to Ready for design yet.** After N5 is answered and the AC-2, AC-5, log-housekeeping fixes are in, a Pass 4 would be a spot-check. The stats document stays Interviewing; it is coherent and testable except where its own open questions say otherwise (m8).

### Pass 2 dispositions
- **N1** partly fixed. FR-8 (one dedicated key holder, no caller or library holds a key), FR-11 (own process reaches local providers only) and AC-5 (hosted attempt from a caller's process refused) are internally consistent. But the ccg path does not deliver the hard control: N5.
- **N2** fixed in FR-3 (restricted scope defined, unmarked scope unrestricted, unreadable or ill-typed refuses, call-time read). AC-2 still says "unreadable or absent from a restricted scope", which the new definition makes meaningless (m1). The lift path still has no AC (m6).
- **N3** fixed: FR-5 per-attempt records with a call id, AC-24 checks two records. The stats document's rates do not say whether they count calls or attempts (m8).
- **N4** mostly fixed. One data source (the manifest, FR-3, FR-20, AC-21) and a coherent path (gateway builds the request, the broker sends or refuses). Residual: m5 (two evaluators).
- **m1** fixed (schedule sentence in the Intent). **m2** fixed (FR-14 cites adoption FR-2, 4, 5, 6, 7, 13, 15 and no history; the overturn lives in the log). **m3** fixed (required unless marked optional). **m4** partly (unload and mismatch rules now in FR-16; residual m4 below). **m5** fixed (log: "confirmed by the stakeholder"). **m6** partly: the retry entry now says "refines" the "opt-in per caller" entry and the key entry names its target, but the loading entry does not name entry "Two rules", and the older entries' "Open questions above" pointers and six TBD refs are untouched (m7). **m7 stats** fixed except what its own open questions list. **m8** fixed (AC-22, AC-6 thinking tokens). **m9** fixed (FR-1, AC-23).
- **Pass 1 residue**: carried only as m7 below.

### Stakeholder answers folded (checked present)
Hard key control with a single key holder outside agent reach, service mandatory for hosted use → FR-8, FR-11, AC-5, log entry names the "Shared how" entry it overturns and the 2026-11-19 tripwire it narrows. ccg local-only flag is the system manifest read by the gateway → FR-3, AC-21. Record per attempt with a call id → FR-5, AC-24. FR-13/AC-9 drop confirmed → log. Loaded with other settings fails naming the mismatch → FR-16.

### Growth-stops check
2,415 → 3,251 → 3,619 words (+368 this pass), while the disposition entries (old 101/102) were removed and FR-14 shrank by citing. The growth is new scope (hard key control, FR-5 per attempt, AC-22 to AC-24, FR-16 rules); there is no rework growth, except that AC-2 was not replaced when FR-3 changed (m1). The log is 1,110 words (31%), append-only, permitted. Within the rules: the deferred list in FR-14 now cites the adoption document by FR number. Not within: the log entry "ccg adopts after the P1 gate closes" is a statement, not question → answer, and the consult entry says "findings above" with nothing above (m5).

### New findings

**N5 (major). FR-8's hard control does not hold on the ccg path as the broker is built.** Verified by reading.
- The broker is "one host process per user" (`broker/__init__.py`), and `server.py:289-290` closes any connection whose peer uid differs from its own. The egress spec §What this boundary does not cover says a same-uid process is not defended against and "root and same-uid compromise are out of scope". Agents in this repo run as the same user. A key the broker can read is therefore one an agent process of that user can read, so FR-8's "outside the reach of any LLM-driven agent" and AC-5's "cannot read the key" are not met by "the key holder sits on the broker's side of the boundary".
- Making the key holder a different OS identity breaks the broker's own rule: a key-holding process of another uid cannot be the broker's peer, and the broker's request builder sets "the only headers sent" and the caller "supplies no header" (egress spec §2.3; `structured.py build_request`). The spec says only that "a later phase's credentials are broker-held", so a hosted call needs a spec change (a credential header, a hosted endpoint rule R7, a key holder reachable under the broker's peer rule).
- "Exactly one" key-holding process versus "for ccg the key holder sits on the broker's side" (if the service of FR-11 and the broker both hold it, there are two). Open questions leave this to design, but the requirement must be satisfiable.
- Fix: the requirement should state the testable property (a process running with the same login as an agent cannot read the key; the key holder can) and which existing ccg boundary (broker spec 3.x) it amends, with the owner of that amendment. Ask the stakeholder whether the key holder runs as a separate identity, and whether the broker or the service makes the upstream hosted exchange. Until then AC-5's agent clause has no feasible check.

**Minor**
- **m1.** AC-2: "unreadable or absent from a restricted scope" contradicts FR-3, where a scope with no marking is unrestricted. Replace with the two cases: unreadable or ill-typed marker refuses; no marker is allowed only where owner configuration allows hosted.
- **m2.** AC-5: define the agent probe (the same read attempted from the identity an agent runs under, expected to fail; the same read from the key holder, expected to succeed as positive control), otherwise the clause cannot go red or green.
- **m3.** AC-7 ("the same call site works from the caller's own process and from another process") cannot hold for a hosted call site under FR-11; limit it to a local call site and add the hosted counterpart (own process refused, service accepted).
- **m4.** FR-16: (a) ccg today takes whatever model is loaded (`interpret.py bound_model`, fingerprint with `loaded_context_length`); a mismatch-fails rule changes that, so say what "configured settings" are for ccg and that adoption keeps the gated behaviour. (b) Does the mismatch rule apply to strict-mode callers such as model-bench, which record the loaded context rather than enforce one? (c) AC-13 has no case for mismatch or for "never unloads".
- **m5.** ccg's broker already evaluates local-only and sticky (`policy.py` R1, R5, R6, R7, P2-OVERRIDE-REFUSED). FR-3 has the gateway read the manifest too, so two evaluators share one source. State that for ccg the broker's decision is authoritative and the gateway's check may only refuse more, and that the manifest counts as owner configuration (confirm an agent cannot write it).
- **m6.** No AC for FR-3's call-time read (marker flipped between two calls) or for the lift path.
- **m7.** Log housekeeping: six "(FR/AC: TBD)" entries; entries 92, 95 and 96 point to "Open questions" that no longer hold their items; the loading entry does not name entry "Two rules"; the "ccg adopts" and consult entries are not question → answer.
- **m8.** Stats document (judged for coherence and testability only): (a) rates and "why a fallback fired" need a stated basis, calls or attempts; (b) the alert threshold's home (owner configuration) is not stated; (c) AC-4 only checks a basis is named, not that the figures are right; (d) AC-5 depends on an open question, which is acceptable while Interviewing but must close before Ready. No solution language found; FR-1 to FR-5 each trace to an AC, and the gateway's old-to-new pointers (FR-21..25 → 1..5, AC-15/16 → 1/5) resolve correctly.

**Nit**
- The consult log entry ("findings above") should cite `docs/reviews/llm-gateway.md` Pass numbers only, or go.
- The Pass 1 residual TBD cleanup is cheap and belongs in the same edit as m7.

### Counts and flip
Pass 3: blocker 0 · major 1 · minor 8 · nit 2. Verdict **needs changes**; the gateway document may not flip to Ready for design until N5 is resolved by the stakeholder and m1, m2 and m7 are fixed. The flip itself also needs explicit stakeholder confirmation.

### Proposed REVIEW.md lines
None.

## Pass 4

Scope: narrow spot-check of `docs/requirements/llm-gateway.md` (3,902 words) and `docs/requirements/llm-gateway-usage-stats.md` (796 words) on the four points asked. FR-3, FR-8, FR-11, FR-12, FR-16, AC-2, AC-5, AC-7, AC-13, AC-21, the Open questions and the log tail were read in full; `cpg/ccg/front/interpret.py` `pick_model`/`bound_model` was re-read for point 2. CPG: not applicable — requirements review.

**Verdict: approve with suggestions.** Blocker 0 · major 0 · minor 5 · nit 2. **The gateway document may flip to Ready for design once the stakeholder confirms**, subject to the two edits named under "Before the flip" (both mechanical). Nothing in the document blocks it.

### Stakeholder answer folded
N5 → separate account or container key holder that agents cannot reach and that alone makes hosted exchanges; ccg's broker forwards allowed hosted calls to it and the egress spec is amended: present in FR-8, AC-5, and in a log entry that names the "Hard control" entry it refines. Checked.

### Pass 3 dispositions
- **N5** resolved at the requirement level: FR-8 now names the property (separate account or container, only that process exchanges, broker forwards). Residual m1 below. The egress spec amendment is stated as scope, which is right: spec3 §2.3 currently has the broker set the only headers and "P1 sends no credential header".
- **m1** fixed: AC-2 now has unreadable or ill-typed marker, change between calls, unmarked scope, and lift attempt. **m2** fixed: AC-5 is an agent probe with a positive control (caveat in point 1). **m3** fixed: AC-7 is local only, hosted through the holder. **m4** fixed for (b) and (c) (rule applies in every mode, AC-13 case added); (a) opens m2 below. **m5** fixed: FR-3 and AC-21 make the broker authoritative and the manifest owner-only. **m6** fixed in AC-2. **m7** partly: superseding entries added (key control; loading rule), but six "(FR/AC: TBD)" entries remain and entries 92/95/96 still point to Open questions that no longer exist (m4 below). **m8** fixed: rates per call or attempt (stats FR-1), replay refusal (stats AC-4), threshold home asked (stats Open questions); AC-5 there still waits on the delivery-point question, which is fine while it is Interviewing.

### (1) FR-8, AC-5, FR-11, FR-12 hold together
The path is consistent: ccg → gateway → broker (policy, audit) → key holder (the only hosted exchange) → provider. FR-11 keeps hosted out of a caller's own process, FR-12 keeps the broker as enforcement point, AC-7 and AC-5's last sentence test the own-process refusal. The AC-5 probe is checkable: a probe as the agents run, searching the holder's account, files, process listing and environment for the planted fake key, with a planted marker in an agent-reachable place as positive control. Two gaps:
- **m1.** Nothing says who may call the key holder. The control stops an agent reading the key, not an agent driving the holder directly, which would bypass the broker's local-only decision (FR-3) and the per-run token bound (FR-10) for any hosted content. For the stated story (a configuration slip) that is acceptable; for the hard-control framing it should be stated: the holder accepts calls only from the broker and the gateway service, or the residual (use without reading) is accepted in one line, as the security review's F1 item 5 asked for in the adoption plan.
- **nit.** The probe's positive-control marker should sit in a place of the same exposure class as the key holder's (a file or process-listing entry) so that a probe blind to containers cannot pass vacuously.

### (2) FR-16 versus ccg's accept-whatever-is-loaded
No contradiction where load settings are configured, and FR-16 now says a call site with none "accepts whatever is loaded and records it (as ccg does today)". But two ccg behaviours are not covered (**m2**), verified in `interpret.py` `pick_model`: ccg names no model, it takes the loaded chat model, preferring `RECOMMENDED_MODEL`, else the first by id; and with none loaded it fails (`ModelError`), it never loads. FR-2 puts the model per call site and FR-16's on-demand mode "loads what a call needs", so for a ccg call site with no model named the requirement does not say (a) that "bind to the loaded model" is a legal call-site setting, (b) that nothing loaded fails rather than loads, or (c) that ccg's call site uses neither on-demand nor strict. AC-13 covers neither. Add one sentence to FR-16 and one case to AC-13; the "gated configuration unchanged" promise depends on it.

### (3) Growth-stops
3,619 → 3,902 (+283). New scope: the key holder (FR-8), AC-2 cases, AC-13 case, AC-5 probe, three log entries. No restated history in FR text. To cut: (a) the first Open questions bullet is now answered by FR-8 (the broker forwards) and its "optional service of FR-11" wording contradicts FR-8's mandatory holder; remove it, keep the zero-dependency design bullet (**m3**). (b) The log is about 1,190 words (30%): append-only is permitted, and the older entries were left untouched as the rule requires; no cut. (c) Two requirement changes have no log entry of their own: FR-16's "only where load settings are configured" (it narrows the logged "call fails naming the mismatch" entry) and FR-3's "broker authoritative, manifest owner-only". They read as the author's own decisions, not stakeholder answers; log them, or confirm them at the readback (**m4**).

### Minor and nit summary
- **m1** key-holder callers (above). **m2** FR-16 for a call site naming no model (above). **m3** stale Open questions bullet. **m4** housekeeping: six "(FR/AC: TBD)" refs in 2026-11-22 entries (add the FR/AC numbers; an addition, not a rewrite), the dangling "Open questions" pointers in entries 92/95/96, and the two unlogged changes above.
- **m5** (stats) FR-4 alerts stay without a threshold home or delivery point until those open questions are answered; not a gateway-document matter.
- **Nit** probe positive control exposure class (above). **Nit** the "ccg adopts after the P1 gate" and consult entries remain statements, not question → answer; leave or reword.

### Before the flip
(1) Fill the six TBD refs and drop the stale Open questions bullet (m3, m4). (2) Add the FR-16 unnamed-model sentence and AC-13 case, or log them as design items for the architect (m2). The key-holder caller rule (m1) and the log entries for the two unlogged changes may go to the readback. The flip needs explicit stakeholder confirmation; the latest verdict is not "needs changes", which is the other precondition in root `AGENTS.md`. The stats document is not flipped by this review.

### Proposed REVIEW.md lines
None.
