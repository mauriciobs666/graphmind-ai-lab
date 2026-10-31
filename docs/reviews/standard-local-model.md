# One standard local chat model (qwen) — Requirements Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## Pass 1

**Scope.** `docs/requirements/standard-local-model.md` (untracked, `Status: Interviewing`, first
version), read in full, ahead of its Ready-for-design flip. Checked: FR/AC testability and freedom
from solution language; internal consistency (FR ↔ AC ↔ out of scope ↔ decision log, and the
conditional `tank` fallback FR-3/AC-1/AC-5); current-state claims against the repo; supersession
coverage; growth-stops, header and decision-log conventions. Not looked at: anything in LM Studio
itself. I did not run anything against the live model server.

**Verdict: needs changes.** There is one blocker: on the trial-fails branch the document explicitly
allows, the acceptance criteria cannot be met. Two majors need stakeholder answers before an
architect can design the trial. All three are small wording or decision fixes. The rest of the
document is sound.

CPG: not applicable — a requirements-document review with no code-level change under review.

**Growth-stops check.** This is the first version (untracked, no earlier pass), so there is no
growth to classify and no restated history. The header block matches root `AGENTS.md`: H1, then
`Status:` token first, `Owner:`, `Tracks: —`, optional `Last updated:`. The decision log has the
right shape: one question → one answer per entry, each citing the FR/AC it affects. The one
reversal (entry 3) names what it supersedes. No findings.

### Findings

**B1 (blocker) — AC-2, AC-3 and FR-1 are unconditional, but FR-3/AC-5 allow `tank` to fail the trial.**
FR-3 and AC-5 make "`tank` fails the qwen trial and stays on ministral" an accepted outcome. Three
other criteria still require the pass branch:
- AC-2 is written as a pass assertion ("then every finding … traces …; one made-up result
  anywhere fails AC-2").
- AC-3 requires the README and `opencode.json` to "name qwen".
- FR-1 lists `tank` as a qwen consumer with no condition.

On the fail branch, AC-2 and AC-3 fail, so the change can never be accepted, even though the
stakeholder said it is done either way. AC-1 handles this correctly with "once `tank` has passed
the trial". *Fix:* (a) rewrite AC-2 so it checks that the trial **was run and its outcome
recorded**. The pass/fail bar stays in FR-4, and the result selects the AC-3 or AC-5 branch.
(b) Start AC-3 with "Given `tank` passed the trial". (c) Make FR-1's `tank` clause point to FR-3
("and `tank`, subject to FR-3"). *Verified by reading.*

**M1 (major) — The FR-4 trial cannot be run as written: the scenario set and the ministral baseline are not defined.**
- *"Each of `tank`'s existing test scenarios":* the existing set is TP-001–TP-011
  (`opencode/docs/test-plans/devops-opencode-headless.md` §5). Only TP-001/002/003 were ever run as
  real model completions. TP-004–011 were verified by a "debug-agent-bypass technique" at the
  script/permission layer, which does not involve the model
  (`opencode/docs/test-reports/devops-opencode-headless-report.md` lines 6–12, 62–71). Running
  those 3× "on qwen" either tests nothing about qwen or needs a new live method. Bring-up
  scenarios (TP-004/005) also need the shared `falkordb-dev` stop/restart bracket each time
  (`opencode/docs/HISTORY.md` 2026-09-13).
- *FR-4b "no worse than on ministral":* the known ministral baseline is DEF-3, partially resolved.
  Ministral is clean on the two plan-literal refusal prompts but ends on a question on two
  out-of-plan generalization probes (report lines 56–62). The document does not say whether the
  comparison is (i) the recorded baseline on TP-002/003 only, (ii) TP-002/003 plus those probes, or
  (iii) a fresh 3× ministral run of the same set. It also does not say what "no worse" counts
  (runs with any violation? violations per run?).

*Fix:* the stakeholder decides, and the decision log records: (1) the scenario set, most naturally
the model-driven ones, TP-001/002/003, plus the DEF-3 generalization probes; (2) the ministral
baseline source (recorded or re-run 3×); (3) the comparison unit. Fold the answers into FR-4 and
AC-2. *Verified by reading the test plan and report.*

**M2 (major) — "One chat model" is not tied to one loaded instance, and nothing says `tank` must work on the qwen that `:8200` already loaded.**
`tank` needs a loaded context of at least 16,384 tokens. Below that, every run returns HTTP 400
before the first turn (DEF-1, fixed by reloading ministral at 21,504 —
`opencode/agents/tank/README.md` Prerequisites; test report line 6). `:8200` JIT-loads qwen on
demand. `falkor-chat/scripts/start_agent_team.sh` sets no context length, so qwen gets LM Studio's
default load configuration. *(Inferred; I did not run LM Studio.)* That leaves two ways to break
the intent while passing the criteria as written:
- `tank` runs against a qwen loaded below its prompt size and fails.
- The operator loads a second qwen instance at a larger context. AC-1 ("the only chat model loaded
  is `qwen/qwen3-4b-2507`") still passes with two instances of the same model, which defeats the
  point of having one model.

AC-6 measures whether qwen plus the embedder fit in GPU memory, but it does not fix the context
length, and memory use grows with context. *Fix:* (a) AC-1 → "exactly one loaded instance of
`qwen/qwen3-4b-2507` and no other chat model". (b) Add an FR: `tank` works against the same qwen
instance the everyday tools use, loaded at whatever context both need, and FR-4's trial runs at
that context. (c) AC-6 measures fit at that context. The mechanism (pre-load, JIT settings) stays
design; Out of scope already leaves LM Studio JIT/TTL settings to the operator. That does not
conflict with stating the requirement.

**m1 (minor) — Current-state facts are wrong in two places.**
- "`tank` uses ministral after a single live test that its own plan calls 'validated shallowly'":
  §7's wording is from the plan, written before implementation. After that, ministral went through
  a three-round live-testing arc (DEF-1/2/3; `opencode/docs/HISTORY.md` 2026-09-13; the test
  report). It was not validated shallowly. It was validated with a known residual (DEF-3), and that
  residual is the baseline M1 needs. Rewrite the sentence along those lines.
- "auto-loads qwen and the embedder on every `ingest_document` or `search_documents` call":
  `search_documents` embeds only the query (`falkor-chat/server/falkorchat/services.py:1366–1405`,
  `self._models.embedder(...)`). Only ingestion (extraction) loads qwen. Rewrite: "`ingest_document`
  loads qwen and the embedder; `search_documents` loads the embedder."

*Verified by reading.*

**m2 (minor) — "Clearly weaker … on the safety check (guard)" mischaracterizes the guard result.**
In `model-bench/results/runs/guard-judge-understanding-*`, ministral has **0/40** false advances
against qwen's 4/40. On the safety-critical error it is *better*. It is much worse on usability:
advance recall 10/30 vs 26/30, false suspends 20/30 vs 4/30. The other claims check out:
- Grounded answers: grounding 9/30 vs 23/30, confirmed clearly weaker.
- NLQ: Layer-1 exact match 32 vs 34 items, plus 3 malformed specs and 1 parse failure against
  0/0, confirmed slightly weaker.

The qwen decision still holds. *Fix:* "much lower guard recall (it suspends far more often)"
instead of "weaker on the safety check". *Verified from the run JSONs (2026-09-19/20 runs).*

**m3 (minor) — FR-5/FR-6 doc coverage is narrower than the docs that actually contradict them.**
AC-3 names only `opencode/agents/tank/README.md` and `opencode.json`. Several other places state
or assume the current model:
- README Prerequisites (line 16) names ministral **or nemotron** as the model to load.
- README "Choosing a model" (lines 137–142) calls nemotron "a later, separate evaluation". That
  contradicts FR-6 and Out of scope.
- The README `--title` rationale (line 45) and the `opencode/agents/tank/scripts/health-check.sh`
  comment (line 21) are written around ministral's strict-alternation template.
- `opencode.json` `provider.lmstudio.models` still provisions `nvidia/nemotron-3-nano-4b`.

*Fix:* make AC-3 cover what FR-5 says ("every document stating `tank`'s model or alternates").
Either name these places or have AC-7 also check that no doc presents nemotron as pending work. The
stakeholder decides whether the provisioned nemotron entry stays (a one-line decision-log entry).
The test report's "try nemotron" recommendations are a by-lookup record and can stay.

**n1 (nit)** — AC-1 "everyday use" needs a definition, e.g. "`:8200` serving agents plus `tank`
runs; no storefront demo, no `model-bench`/storefront-script loads". **n2 (nit)** — The 2026-10-30
devops LM Studio log check has no citable artifact. Cite one, or say "observed by the operator".
**n3 (nit)** — AC-6 "measured and reported": say where it is reported (e.g. the delivery
`HISTORY.md` entry).

### What's solid

- Every cited current-state fact other than m1/m2 checks out:
  - `falkor-chat/config/models.json`: all four chat kinds are qwen; the embedder is separate.
  - The salesperson ministral pin: `proof_defs.py:423`; K-056 in `falkor-chat/docs/HISTORY.md`
    2026-08-30 (0/176 skip-and-fabricate on ministral).
  - `tank`'s model: `opencode.json` and README line 128.
  - Plan §7's "validated shallowly" wording, and the open nemotron item at
    `opencode/docs/BACKLOG.md:14`.
  - `ensure_demo_models.sh` refusing while a foreign model such as qwen is loaded:
    `falkor-chat/AGENTS.md` lines 87, 107–111.
- The FRs avoid solution language. The mechanism (pre-load vs JIT, config edits) is correctly left
  to design, and Out of scope fences off LM Studio settings, the embedder and `model-bench`
  cleanly.
- The conditional fallback is the right shape: decided up front, recorded in the decision log, and
  AC-1 already worded conditionally. B1 only asks the other criteria to match.
- Supersession of plan §7's ministral default is named (decision-log entry 3), and FR-6 retires
  the nemotron backlog item explicitly instead of leaving it to rot.

### Open questions (for the stakeholder, via `tico`)

1. Trial scenario set, ministral baseline source, and comparison unit (M1).
2. Should the provisioned nemotron entry in `tank`'s `opencode.json` stay or go (m3)?

## Pass 2

**Scope.** Re-read the revised `docs/requirements/standard-local-model.md` in full, after Pass 1
and the stakeholder's answers. I rechecked the Pass 1 evidence where the text changed.

**Verdict: approve with suggestions.** No blockers or majors remain. One new minor is worth fixing
before the Ready-for-design flip, because it changes what the trial measures (P2-m1). The others
can be taken or left.

CPG: not applicable — a requirements-document review with no code-level change under review.

**Growth-stops check.** The document grew. Most of the growth is **new scope** decided by the
stakeholder:
- FR-7, the single ≥16,384-context qwen copy;
- FR-4's ministral baseline and its scenario set;
- FR-6/AC-7's removal of nemotron from `opencode.json`.

The rest is **rework done in place**, with no stacking: FR-1, AC-1 to AC-3, AC-6, and the Problem
bullets were rewritten, not appended to. The new decision-log entries (119–122) each take the
required one question → one answer + FR/AC form. One exception, entry 118, is a finding (P2-m2).

### Pass 1 dispositions

- **B1** — fixed. FR-1's `tank` clause now says "subject to FR-3". AC-2 records the trial outcome
  instead of asserting a pass. AC-3 starts with "Given `tank` passed the trial".
- **M1** — fixed through the stakeholder's answers: TP-001–003, a fresh 3-run ministral baseline
  in the same session, and the per-scenario clean-run comparison (FR-4, AC-2).
- **M2** — fixed. FR-7 requires one qwen copy at ≥16,384 tokens however it is loaded. AC-1 now
  asks for "exactly one … one copy", covers the JIT-loaded case, and requires the `tank` run to
  succeed. AC-6 measures at FR-7's context.
- **m1** — fixed. The Problem bullets now name the later live-testing arc and DEF-3, and say
  `search_documents` loads only the embedder.
- **m2** — fixed. The guard result now cites 0/40 vs 4/40 false advances and 10/30 vs 26/30
  recall.
- **m3** — fixed. AC-3 lists the README sections, the `health-check.sh` comment, and
  `opencode.json`. AC-7 covers the README and the nemotron entry in `opencode.json`.
- **n1** — fixed: AC-1 now defines everyday use.
- **n2** — fixed: "a read-only devops check during this interview".
- **n3** — fixed: AC-6 says "recorded in the change's delivery record".

### Stakeholder answers folded in

- Which scenarios does the trial use? → TP-001–003, 3 runs each (FR-4).
- Where does the ministral baseline come from? → A fresh 3-run pass in the same session; qwen's
  clean-run count must be ≥ ministral's for each scenario (FR-4b, AC-2).
- Does nemotron stay in `tank`'s `opencode.json`? → No, it is removed (FR-6, AC-7).
- What context must the one qwen copy have? → ≥16,384 tokens, however it is loaded, JIT included
  (FR-7, AC-1).

### New findings

**P2-m1 (minor) — Now that AC-2 covers both models, "made-up result" is scoped to the wrong runs, and "clean run" is undefined for TP-001.**
- AC-2 runs the scenarios "on qwen and on ministral", then says "every finding in **all of those
  reports**" must trace to a command, and "one made-up result **anywhere** is a fail". Read
  literally, a hallucination in a ministral *baseline* run fails *qwen's* trial. FR-4a correctly
  says "any qwen run". *Fix:* AC-2 should say "every finding in the qwen runs' reports" and "one
  made-up result in any qwen run".
- FR-4b defines a clean run by refusal behaviour (refuse once, no retry, no closing question).
  TP-001 is the health check, where nothing should be refused, so for TP-001 that definition is
  either always met or meaningless. *Fix:* define a clean TP-001 run, for example "completes the
  check with no refusal, no retry, and no closing question", or limit FR-4b to TP-002/003.

*Verified by reading FR-4 and AC-2 against each other, and against the test plan's TP-001
definition (`opencode/docs/test-plans/devops-opencode-headless.md` §5).*

**P2-m2 (minor) — Decision-log entry 118 is a review-disposition record, not a decision.**
The entry reads: "Consult: `analyst` … → needs changes, Pass 1; folded B1, M2, m1–m3 and nits (…);
M1 and nemotron-in-config raised as open questions 1–3". That is a finding→resolution list. Root
`AGENTS.md`'s growth-stops bullet keeps those in the review's `## Pass N`, and allows a decision log
only question → answer + FR/AC entries. The answers it refers to already have their own entries
(119–122). *Fix:* delete entry 118. The `Reviews:` header field already points here.

**P2-n1 (nit) — FR-7 and Out of scope overlap.** Out of scope excludes LM Studio "JIT/TTL
behaviour". But FR-7/AC-1 require that a JIT-loaded qwen arrives at ≥16,384 tokens, and in practice
that is an LM Studio per-model load setting. *Fix:* add one clause to the Out-of-scope bullet, e.g.
"other than the qwen context length FR-7 requires", so the architect doesn't read the two as
conflicting.

**P2-n2 (nit) — FR-4 doesn't say which copy the qwen trial runs against.** FR-4 could say the
qwen runs use the FR-7 copy (≥16,384), so that a pass carries over to everyday use. The ≥16,384
figure was set against ministral's tokenizer (`opencode/agents/tank/README.md`); running the trial
at that context is what confirms it is enough for qwen too.
