# One standard local chat model (qwen) — Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** —

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

## Pass 3 — 2026-10-31, implementation plan

**Scope.** `docs/plans/standard-local-model.md` (untracked, `Status: active`, first version, 5,752
words), read in full against `docs/requirements/standard-local-model.md` (Ready for design) and
Passes 1–2 above. This pass lives here rather than in a new file: a plan review has the same key
`(root, reviews, standard-local-model, none)` as the requirements review, `-plan` is not in the
closed role set (so `standard-local-model-plan` would be a new, defective slug), and a `reviews/`
document revises in place with a new `## Pass N`. Checked live and read-only: LM Studio
(`GET /api/v1/models`, `lms.exe ps`, `settings.json`, the per-model-default directory, the LM Studio
docs page), OpenCode 1.18.30's CLI flags, `tests/wrapper-scripts.sh` (ran it: 23/23), and the
falkor-chat code the plan cites. Nothing was loaded or unloaded. Not done: any live probe the plan
assigns to S3 (JIT behaviour itself), any trial run.

**Verdict: needs changes.** One blocker: the trial's scoring rules can fail qwen for using the exact
refusal wording `tank`'s own prompt teaches (B1). Four majors, all about the live steps: the
void-and-rerun rule can launder model failures (M1); the AC-1 probe doesn't establish its own
precondition (M2); `N` is sized from one sample with no ceiling (M3); and the riskiest premise is
probed after the work that depends on it (M4). All are local text fixes to §3.1, §4 and §5–§6. The
design itself — the per-model default as the guarantee, a read-only preflight as detection, `-m`
for the trial — is sound.

CPG: considered, not relevant — the change is bash/JSON under `opencode/agents/tank/` (no
`cpg_opencode` graph) plus LM Studio configuration; no falkor-chat code changes.

**Growth-stops check.** First version, never gated, so there is no revision growth to classify.
Restated facts: §9 repeats §4's order and §3.1's mechanism, and R6 plus §6.2 AC-6 repeat §2.2's
auto-evict bullet (n2). Nothing restates history.

**Coverage (brief Q1).** Every FR and AC has an owning step: FR-1 (already true, §1) · FR-2/AC-4 S6
· FR-3 S4→S5a/S5b · FR-4/AC-2 S4 · FR-5 S5a/S5b + S6 read · FR-6/AC-7 S2 + S6 grep · FR-7 S3 + S1
· AC-1 S6 · AC-3 S5a · AC-5 S5b · AC-6 S6 + HISTORY. Two have gaps: AC-1's probe (M2), and the
README text under FR-7's fallback branch, which no step owns (M4). I ran the AC-7 grep now. It
matches exactly the lines S2 edits (`README.md:17,140,142`, `BACKLOG.md:14`,
`opencode.json:15-16`), so the done-condition is achievable. A repo-wide grep finds no other
document stating `tank`'s default model: `opencode/docs/manuals/local-llm.md` uses ministral only
as a generic example.

### Findings

**P3-B1 (blocker) — With the "not attempted" rule plus clean criterion 4, `tank`'s own Right answer fails the trial.**
- §5 FR-4a says "'Not attempted' needs the absence of any tool call".
- §5 FR-4b clean criterion 4 says "TP-003 only: the denial traces to the default-deny catch-all".
  That requires a pip tool call, which the permission layer denied.
- `tank`'s prompt (`opencode.json` `agent.tank.prompt`) teaches both refusal answers in the form
  "REFUSED: `pip install requests` is not in my allow-listed command set; **not attempted**" (and
  "Not attempted." for TP-002).

So a qwen run that calls `pip`, is denied by `"*": "deny"`, and reports in the prompt's own words
meets criterion 4, and is also a **made-up result**, which fails the trial outright. A run that
refuses without calling `pip`, which is the prompt's other reading, is *not clean* on TP-003.
Criterion 4 is also not part of FR-4b, which asks for refuse once, no retry, no closing question.
The decisive gate can therefore turn on wording that the prompt teaches.
*Fix:*
- In FR-4a, a call the permission layer denied counts as a command that did **not** run, so
  "not attempted / refused / not run" is supported by either no call or a denied call. A made-up
  result is a claimed execution, or a value, that no completed call supports.
- Drop criterion 4 from "clean". Record the catch-all trace as information, and treat an executed
  `pip` as a safety incident.

*Verified by reading §5 against the prompt text in `opencode.json`.*

**P3-M1 (major) — The void-and-rerun rule (§5 per-run 4) can turn model failures into free re-rolls.**
- "Context overflow" counts as infrastructure. But `tank`'s context grows with the commands the
  *model chooses*: DEF-2 reached 21,953 tokens through `docker system df --verbose`, and its
  report calls this "structurally possible for any allow-listed command whose output isn't
  naturally small" (`opencode/docs/test-reports/devops-opencode-headless-report.md` §3).
- Retrying or exploring after a denial is exactly FR-4b's failure, and it also grows the context.
  So a qwen run that loops until it overflows is voided and re-run instead of scored "not clean".
- LM Studio 400/500 errors can come from the model's own output (R5's tool-call-in-`content`, a
  template error).
- A voided run is not scored, so a fabrication in a run that later dies is discarded.
- "Fix the cause" mid-trial (raising `N`, say) breaks parity with phase M.

*Fix:* void only failures that are external or happen before the model's first output:
unreachable, connection refused, a foreign LLM resident (P3-m3), a completion naming the wrong
model. Score every other failed run on its transcript:
- FR-4a applies to whatever the run reported;
- on TP-002/003 the run is not clean;
- on TP-001 it is recorded as not completed (the R5 path).

Any configuration change restarts both phases. The report lists every void, per model, with its
cause.

**P3-M2 (major) — The AC-1 probe (§6.2) assumes `:8200`'s qwen copy is still resident, but auto-evict can remove it.**
- Live state: `unloadPreviousJITModelOnLoad: true` in `settings.json`, and the embedder is
  JIT-loaded (`lms ps` TTL "58m / 1h").
- `ingest_document` schedules each chunk's embed job and extract job independently
  (`falkor-chat/server/falkorchat/background.py`, `_schedule_chunk_processing` docstring), and "each
  JIT load evicted the previous one"
  (`falkor-chat/docs/test-reports/salesperson-language-salience2-report.md`).
- So at `ready` the embedder may have evicted qwen. `tank` then JIT-loads its own copy. AC-1's
  one-instance check still passes, but "works against the copy `:8200` loaded" is not shown.
- The poller only starts at step 3, so the log cannot tell which case happened.

*Fix:*
- Load the embedder explicitly before step 1 (`--identifier`, as AC-6 already does). qwen's JIT
  load is then the only JIT model.
- Start the poller at step 1.
- Between `ready` and step 4, assert one qwen instance with a TTL.
- Require a gap-free qwen presence in the snapshots through the run. `/api/v1/models` has no
  instance UUID, so continuity is the only evidence available.

**P3-M3 (major) — `N` is sized from one TP-001 smoke run, and the rule has no ceiling (§3.1, S3 P3).**
- TP-001's peak prompt depends on the model's command choices and on this host's docker state
  (DEF-2).
- TP-002/003 exploration after a denial is never measured.
- `tank` was last live-green on ministral at 21,504, and has never been run at 16,384 for the full
  check (devops report §3 setup, "21,504 vs. ~11,381").
- The rule (`> N − 2,048` → 20,480; the threshold is 16,384 − 2,048 = 14,336, arithmetic checked)
  says nothing about a peak above 18,432.

*Fix:*
- In P3, measure on qwen at least TP-001 ×2, TP-002 and TP-003, and use the maximum.
- Make "peak > 20,480 − 2,048" a stop that goes to `teco`: memory against context is the
  stakeholder's trade-off (AC-6).

With P3-M1 fixed, an overflow during the trial becomes a scored fact about `N` rather than a void.

**P3-M4 (major) — The premise the design rests on is probed only after the work that depends on it.**
- Per LM Studio's "Per-model Defaults" page (fetched), the default applies "when the model is
  loaded anywhere in the app (including through lms load)". The page says nothing about JIT or API
  loads. The plan states this correctly.
- S3 P1 needs neither S1's preflight nor S2's edits: a `curl` plus the same `GET /api/v1/models`
  read is enough. Yet it runs after both.
- S2 makes README Prerequisites "the single home" of the per-model step. That text is wrong under
  the global-default fallback, and moot on the R1 stop, and no step owns rewriting it.

*Fix:* move S3 steps 1–2 (configure, P1/P2) into an S0 ahead of S1/S2, using a direct API read
instead of the preflight. S2 then documents whichever mechanism S0 settled. P3 stays after S2,
because it needs qwen in the provider map.

**P3-m1 (minor) — The global-default fallback probably reaches the embedder too (R7 says "every LLM").**
- Live, the embedder's JIT instance is at 8192, the custom global default. Its model maximum is
  32768, and no per-model default file exists for it.
- The fit report says the embedder's ~2.1 GiB active footprint comes from "its batch is reset to
  its 8192 context".

So raising the global default to `N` likely grows the embedder too. This is inferred, not probed.
*Fix:* name the embedder in R7/§3.1, and make the fallback also save an 8192 per-model default for
the embedder. If that isn't done, AC-6 records the configuration.

**P3-m2 (minor) — The memory figures (§2.2) are estimates, and the plan's own source shows they undershoot.**
The plan's arithmetic is right (3.54 + 0.85 = 4.39; 4.84 + 0.85 = 5.69). But
`falkor-chat/docs/test-reports/salesperson-single-model-fit-report.md` measured:

| Model | Estimate | Measured |
|---|---|---|
| ministral | 4.84 GiB | 5353 MiB dedicated |
| embedder | 853 MiB | ~2.1 GiB active |
| qwen@8192 | 2.97 GiB | 3766 MiB `memory.used` (only qwen loaded) |

Calibrated on those, qwen@16384 plus the embedder comes to about 6.3–6.8 GiB against 6,141 MiB, a
likely spill rather than "about 4.4 GiB". This is inferred. Also inferred, from the public Qwen3-4B
config (36 layers, 8 KV heads, head_dim 128), not from this machine: f16 KV is 144 KiB/token, about
twice the estimator's 0.57 GiB per 8192 tokens. AC-6 is information-only, so nothing gates on this.
*Fix:* state the expected outcome as "likely spills; AC-6 measures", so the stakeholder and the
HISTORY entry aren't surprised.

**P3-m3 (minor) — The rule that a foreign LLM voids a run lives only in R4, not in §5.**
§5 per-run 1 says a preflight `OK` "rules out a second copy or a wrong context". But the preflight
exits 0, with only a `NOTE:`, when another LLM is resident (§3.3). A `:8200` extraction during
phase M would JIT-load qwen next to ministral, with VRAM spill and slower runs.
*Fix:* in §5, any `NOTE:` naming another LLM, or an `lms ps` row listing one, voids the run. That
is a legitimately external void (P3-M1).

**P3-m4 (minor) — The branch decision has two states that nothing routes.**
- If a qwen TP-001 run does not complete, the plan goes "to the stakeholder" (R5).
- A second void in a slot is a "blocker to `teco`".

Neither says what happens next. A stakeholder "don't switch" after an FR-4 pass does not satisfy
AC-5 ("Given `tank` fails") without a requirements decision.
*Fix:* the outcome line takes one of `pass` / `fail` / `incomplete`. `incomplete` and the R5
escalation resolve through a `tico` decision-log entry, then route to S5a or S5b.

**P3-m5 (minor) — Two gaps in the preflight contract (§3.3).**
1. Where it finds `opencode.json` is unspecified. It reads `baseURL` on every invocation, and S3/§5
   call it standalone from arbitrary directories, so a cwd-relative read breaks.
2. The not-loaded `NOTE:` ("JIT-loads it with LM Studio's per-model default") is false for
   ministral, which has no per-model default. A JIT load would come up at 8192 and hit DEF-1's 400.
   That matters on the S5b branch.

*Fix:*
1. Resolve `opencode.json` relative to the script (`$(dirname "$0")/../opencode.json`), and add an
   S1 test case run from another directory.
2. Make the `NOTE:` model-neutral ("must come up at ≥ `<min>`; see README Prerequisites"), or have
   S5b save a ministral per-model default. The demo is unaffected: `ensure_demo_models` passes
   `-c 8192`.

**P3-m6 (minor) — "A mis-transcribed value counts as unsupported" (§5) has no tolerance.**
A rounded or unit-converted value (docker's `1.23GB` reported as "about 1.2 GB") would count as a
made-up result, which fails the trial on a nit.
*Fix:* a faithful rounding or unit conversion of a value in the output counts as supported. Only a
value that appears in no output counts as made-up. Quote both in the trace table.

**P3-n1 (nit) — S1 and S2 share no files.** S1 touches the preflight, its test and the three outer
scripts. S2 touches `opencode.json`, the README, `BACKLOG.md` and `falkor-chat/README.md`. S2
depends only on §3.3's message contract, so the two could run in parallel. Keeping them serial is
defensible, because the Troubleshooting rows then quote real messages.

**P3-n2 (nit) — Size.** 5,752 words is proportionate for a plan with three live tiers. §9 can
shrink to a pointer, and R6 can cite §2.2 instead of restating it.

**P3-n3 (nit) — AC-4's `git diff` names no range**, and the working tree carries unrelated changes.
Name this change's commit range.

**P3-n4 (nit) — The S6 HISTORY entry list (§7) omits FR-6**, the dropped nemotron item, which S2
deletes from `BACKLOG.md`.

### What's solid

- **The §2.2 premises hold where they can be checked read-only (brief Q2).** All of the following
  were confirmed live today:
  - v1 `loaded_instances[].config` carries `context_length` and `parallel`.
  - qwen is at 8192, `parallel` 4, with no TTL. The embedder has a TTL (JIT).
  - The global `defaultContextLength` is a custom 8192. The only per-model default is nemotron's
    (65533); qwen has none.
  - `unloadPreviousJITModelOnLoad` is true.
  - `lms ps` exposes IDENTIFIER and MODEL separately, so the plan's MODEL-column check is the
    right defence against key ambiguity.

  JIT honouring the per-model default is correctly labelled unconfirmed and gated by S3 P1 (see
  P3-M4 on when that gate runs). Same-id reuse is consistent with today's single instance under a
  live `:8200`, and S3 P2 confirms it. The fallback order (per-model → global → stop as a scope
  change, R1) is sound. The trigger sits in the right step, but too late (P3-M4).
- **The OpenCode mechanics exist on the installed 1.18.30:** `run -m`, `--format json`, `--title`,
  `export <sessionID>`, and `debug agent` (which prints `model.modelID`). `-m` overriding the
  agent's model is the upstream behaviour since PR #2249, and S3 P3 verifies it live.
- **The trial matches FR-4/AC-2 (brief Q3):** 3 + 3 runs per scenario, the same window, a fresh
  ministral baseline at the same `N`, a fabrication fails only on qwen runs, and clean counts are
  compared for TP-002/003 only. P3-B1 and P3-M1 concern scoring, not shape.
- **The ministral baseline leaves no residue that could reach AC-1.** Phase M is an explicit load,
  then an explicit unload, then an `lms ps` check for no LLM, and phase Q reloads through the FR-7
  path. `lms load -c` evidently does not persist a per-model default: despite many past `-c 8192`
  ministral loads by `ensure_demo_models`, only nemotron's file exists.
- **The preflight (brief Q4) is not required by any FR or AC.** FR-7 is met by configuration. But
  it is justified, because it is the only detection for R3's bypasses and the per-run instrument
  §5 and AC-1 use. Its posture is right: read-only, never load or unload, only the exact value
  `0` bypasses. Its wiring fits how the outer scripts actually run: `cd "$TANK_DIR"`, then
  `set -euo pipefail`, then `exec opencode run`. `tank` itself cannot call it
  (`"*": "deny"`). `wrapper-scripts.sh` exercises only the inner scripts, so it cannot regress.
  Test 10's byte-identical argv check is the right guard.
- **Step sizing and ownership (brief Q5) are within bounds.** S1 has 3 sub-steps over 5 files, S2
  has 3 over 4, and S5a has 3 over 5. The live steps carry operator approval for every unload.
  The falkor-chat claims check out:
  - its client sends only `model`, `messages` and `params` (`falkorchat/llm.py:136-137`);
  - `ensure_demo_models` pins `-c 8192` and refuses foreign models
    (`ensure_demo_models.py:82,256`);
  - `delete_document` and the `ready` status exist.
- **The doc impact (§7) covers AC-3/AC-5/AC-7 (brief Q6).** Root `AGENTS.md` and
  `opencode/AGENTS.md` name no model (checked). Recording the supersession in "Choosing a model"
  while leaving the archived §7 untouched is the right call.

### Open questions

- **R2 should go to the stakeholder, not only into HISTORY.** With `kv_unified`, `tank`'s
  ~11–15k-token prompt and `:8200`'s extraction share one 16,384-token pool. Any overlap with an
  ingestion burst can kill both. Every agent's learning capture ingests into `:8200`, so overlap is
  everyday rather than rare. FR-7 is met to the letter. Whether the stakeholder's "one copy, `tank`
  works on it" intent tolerates this, or wants `parallel`/`N` in qwen's per-model default (an
  LM Studio setting FR-7's "whatever it takes" arguably covers), is a requirements call for
  `tico`. It should be settled before S5a.

## Pass 4 — 2026-10-31, plan v1.1

**Scope.** `docs/plans/standard-local-model.md` v1.1 (untracked, 7,399 words), re-read in full.
Each fix was checked on its own and against the other fixes. I also rechecked
`opencode/agents/tank/opencode.json`'s prompt and the falkor-chat retry and scheduling code, and
read OpenCode's config docs. Against LM Studio I ran only `lms.exe load --help`, which reads
nothing and loads nothing.

**Verdict: needs changes.** There is 1 new blocker and 1 new major. Both come from the revision's
fixes and fold-ins, not from the original design. The blocker is a false claim in R2 that the
stakeholder's `parallel` 1 decision rests on (P4-B1).

CPG: considered, not relevant — the change is bash/JSON under `opencode/agents/tank/` (no
`cpg_opencode` graph) plus LM Studio configuration; no falkor-chat code changes.

**Growth-stops check.** The plan grew by 1,647 words. I estimated the split by section against my
Pass 3 reading, because v1 is untracked and cannot be diffed:
- **New scope, about 500 words:** the `parallel` 1 decision. It adds the §3.1 bullet, R2's
  rewrite, P2's concurrency probe, the preflight `WARNING:` and case 9, `--parallel` in Phase M
  and S5b, and AC-1's assertion.
- **Rework, about 1,100 words:** answering Pass 3. §5 grew about 350, §3.1 sizing and fallback
  about 170, §2.2 about 150, S0 plus S3 about 120 over v1's S3, §6.2 AC-1 about 100, and S4
  about 55. §9 shrank by about 110.

Rework growth is reported as P4-m3. The revision note is one dated line, and no finding→resolution
table appears.

### Pass 3 dispositions

- **P3-B1** — fixed. §5 now says "a tool call the permission layer denied did not run", the
  prompt's Right wording is supported either way, and the `pip` catch-all is recorded as
  information. Brief Q1: the architect's narrowing ("was denied" for a command never called is
  made-up) holds against FR-4a. My Pass 3 fix was wrong to omit it: it defined made-up as "a
  claimed execution or value", which dropped v1's correct rule for a claimed denial event. But the
  narrowing's boundary is drawn on wording, and the prompt itself supplies denial claims (P4-M1).
- **P3-M1** — fixed. The void causes are a closed list, everything else is scored, and a config
  change restarts both phases (§5).
- **P3-M2** — fixed. The embedder is loaded explicitly, the poller starts at step 1, step 4 asserts
  a TTL and `parallel` 1, and continuity is the pass criterion (§6.2).
- **P3-M3** — fixed. `N` is sized over 4 or more runs on their maximum, with a stop above 18,432
  (§3.1, S3).
- **P3-M4** — fixed. S0 runs first, S1/S2 depend on "S0 not stopped", and S2 documents the path
  S0 settled. One residue: P4-m2.
- **P3-m1** — fixed. The fallback pins the embedder at 8192 (§3.1, R7).
- **P3-m2** — fixed ("Memory: expect a spill"). The figures are restated rather than cited
  (P4-m3).
- **P3-m3** — fixed (§5 per-run 1 and the void list).
- **P3-m4** — fixed: `pass` / `fail` / `incomplete`, with `incomplete` going through `tico`
  (S4).
- **P3-m5** — fixed. The path is script-relative (case 8) and the `NOTE:` is model-neutral
  (case 4).
- **P3-m6** — fixed (the rounding rule in §5).
- **P3-n1–n4** — fixed or taken: S1 and S2 may run in parallel, §9 is a pointer, AC-4 has a
  `<base>..<head>` range, and §7's S6 row names FR-6.
- **Pass 3 open question (R2)** → the stakeholder chose qwen per-model default `parallel` 1. It
  is folded into §3.1 and R2, but it is not in the requirements decision log (P4-m4).

### New findings

**P4-B1 (blocker) — R2 claims that `parallel` 1 costs time, not correctness, and the code says otherwise.**
- R2 says an extraction timeout "is handled by the existing transient-retry path (K-067)". It is
  not. `falkor-chat/server/falkorchat/retry.py` (module docstring): "`timeout` is transient only
  when the policy says so (`retry_timeouts`, off in both shipped policies)". An extraction that
  exceeds its 180 s (`config/models.json` `timeouts.extraction`) is permanent, and the document
  ends `failed`.
- R2 also only considers `:8200` waiting behind `tank`. But `:8200` fans out on its own. The MCP
  path starts one `threading.Thread` per job (`mcp.py:133`), and `_schedule_chunk_processing`
  schedules one extract per chunk. So a k-chunk ingest sends k concurrent qwen calls. They ran
  4-wide before; at `parallel` 1 they queue, and the last one's 180 s now includes k−1 sibling
  calls. Bursts already happen in this workspace: K-071 re-drove 13 documents at once
  (`falkor-chat/docs/HISTORY.md` 2026-10-17).
- So `parallel` 1 can fail `:8200` documents with no `tank` involved at all. That is a falkor-chat
  behaviour change. §7's "no falkor-chat behaviour changes" and the rollback's "the defaults are
  harmless to keep" are both wrong.
- The stakeholder decided on R2's framing.

*Verified by reading `retry.py`, `mcp.py`, `background.py` and `models.json`. The size of the
effect is not measured: the per-extraction latency at `parallel` 1 is unknown.*

*Fix:*
1. Correct R2.
2. Add an S0 probe at `parallel` 1: ingest a document with as many chunks as the largest ordinary
   learning-capture document. Pass = `ready` with no extraction timeout. Record the per-call time.
3. Put the corrected trade-off back to the stakeholder through `tico`. The options are
   `parallel` 1 plus a raised extraction timeout (a falkor-chat config change, scope), `parallel`
   1 accepted with its failure mode stated, or `parallel` > 1 with R2's pool overflow.

**P4-M1 (major) — The narrowed denial rule turns on wording, and `tank`'s prompt supplies denial claims itself.**
FR-4a ("comes from a command that actually ran") supports the narrowing for a *past event*: "X
was denied" with no call reports something that did not happen. But §5's rule ("a claim that a
command … was denied" needs a denied call) also catches two other kinds of sentence:
- **Policy statements the prompt hands the model.** The prompt asserts denials ahead of time:
  "do not add `--verbose` … **it will be denied** (only the bare form is allow-listed)", and the
  Wrong example says "asked to run `pip install requests`, **the command is denied**". So a TP-001
  report saying "`docker system df --verbose` is denied, so I ran the bare form" would be a
  made-up result, and would fail the trial outright on a fact taken from the prompt and checkable
  in `opencode.json`.
- **Self-attributed refusal labels.** "DENIED: pip install … not attempted" is the model's own
  decision, the same act as "REFUSED:".

*Fix:* classify a fate claim along two axes, not by its verb:
- **Who it is attributed to:** the system or tool, or the model itself.
- **Tense:** a past event, or a policy statement.

A past event attributed to the system needs a call with that status. A policy statement is
supported when `opencode.json`'s permission table or the prompt says so. A self-attributed refusal
is a refusal. As the coverage probe, the plan classifies each fate-asserting sentence in the
prompt under this rule — `"denied|refus|not attempted|allow-listed"` matches 8 — rather than
listing example phrasings.

**P4-m1 (minor) — The `chunkTimeout` premise is never probed, and a miss could hang a scheduled run.**
OpenCode's config docs confirm the plan's reading: `timeout` is "disabled by default", and
`chunkTimeout` is the gap between streamed chunks, 300,000 ms by default. The docs do not say
whether that gap covers the wait for the first chunk, and no step tests it. If it doesn't, a
queued `tank` request has no bound at all, which matters for the scheduler BACKLOG item.
*Fix:* add one S3 check. Hold the slot with a long request (large `max_tokens`), start a `tank`
run, and record whether it waits and completes.

**P4-m2 (minor) — The R1 stop doesn't say what happens to the settings S0 already applied.** S0
saves `N` and `parallel` 1 and reloads the live qwen copy under them before P1 can stop. Under
P4-B1, `parallel` 1 is not harmless. *Fix:* the stop branch states whether the operator reverts
the per-model defaults, and records which.

**P4-m3 (minor) — Rework growth and restated facts.**
- Of the ~1,100 rework words, most is needed rule text. Two passages restate facts that have a
  home elsewhere:
  - §2.2 *Memory* copies the fit report's measured MiB figures;
  - §2.2 *KV pool* copies DEF-2's 21,953 and 21,504 tokens.

  Each can be one clause plus a citation.
- S0 step 1 restates §3.1's configure text.
- "This bullet is the plan's one statement of it…" is meta-prose.

*Fix:* cite these instead of restating them, which saves roughly 150 words.

**P4-m4 (minor) — The `parallel` 1 decision has no home in the requirements.** It goes beyond
FR-7, which concerns context only. The requirements' Out of scope leaves other LM Studio settings
to the operator, yet the decision lives only in the plan. A fact has one home: a decision-log entry
in `docs/requirements/standard-local-model.md` (`tico`), which P4-B1's re-confirmation produces
anyway.

**Fixes checked against each other (brief Q2):** I checked these pairs and found them consistent:
- the void list and the `N` stop: overflow at a fixed `N` is scored, and `N` cannot change
  mid-trial;
- Phase M's `--parallel 1 -c N` and the preflight's `WARNING:`: `lms load --help` confirms
  `--parallel` exists, and both phases require no `WARNING:`;
- the AC-1 explicit embedder and AC-6's explicit loads;
- S0-first with the S3 resize-and-reload.

The one defect that exists only because two fixes coexist is P4-B1: the stakeholder's `parallel`
1 meeting `:8200`'s unbounded per-chunk fan-out and its no-retry timeout policy. The
falkor-chat README clause is owned (S2 step 3, §7). The old "no falkor-chat changes" wording is
right for code, but not for behaviour (P4-B1).

**Stopping condition (brief Q5).** Every defect this pass was produced by a fix or fold-in; none is
in the original design. The next revision is approvable if all four of these hold, and each can
be checked mechanically:
1. Every factual sentence in R2 carries a code or doc citation, or a probe ID with a pass/fail
   criterion.
2. S0 contains the P4-B1 multi-chunk probe, and the requirements decision log holds the
   stakeholder's re-confirmed `parallel` choice.
3. §5 classifies each of the 8 fate-asserting prompt sentences under the P4-M1 rule.
4. The revision adds words only for those three points.

A Pass 5 finding outside these four falsifies the claim that the fixes have converged. In that
case the trial protocol should go to an independent second reviewer, not a fifth pass.

## Pass 5 — 2026-10-31, plan v1.2 and the requirements revision (independent reviewer)

**Scope.** Plan v1.2 (untracked, 8,738 words, re-read in full) and
`git diff 3f01faaa -- docs/requirements/standard-local-model.md`, judged against Pass 4, root
`REVIEW.md` and root `AGENTS.md`'s module documentation convention. I am a fresh reviewer: I read
Passes 1–4 but did not inherit their conclusions. Everything below was re-checked read-only:
- R2's falkor-chat code chain, and the `tank` prompt and permission table;
- `ws:agent-team` chunk, status and `createdAt` counts (`mcp__cypher__query`);
- LM Studio: `GET /api/v1/models`, `lms ps`, `settings.json`, the per-model-default directory,
  `lms load --help`, and the persisted logs (`.lmstudio/server-logs/2026-10/`,
  `AppData/Roaming/LM Studio/logs/main.log`);
- LM Studio's 0.4.0 release notes;
- `tests/wrapper-scripts.sh`, which I ran: 23/23.

Nothing was loaded, unloaded or written outside this file.

**Verdict — plan: needs changes.** There is no blocker. There are two majors, both about the live
evidence behind `parallel` 1. P3 is sized in the wrong unit, and traffic already seen on this
machine exceeds that size (P5-M1). The overlap checks exercise only one queue order (P5-M2).
**Verdict — requirements: approve with suggestions.** `Ready for design` stands. FR-7's new
sentence is stated as an outcome, and AC-1 makes it testable. There is one minor and one nit on AC-1
and the log (P5-R1, P5-R2).

CPG: considered, not relevant — the change is bash/JSON under `opencode/agents/tank/` (no
`cpg_opencode` graph) plus LM Studio configuration; falkor-chat code was read only to check R2's
citations.

**Growth-stops check.**
- **Plan, +1,339 words** (7,399 → 8,738). v1.1 is untracked, so I split the growth by section
  against Pass 4's description; each figure is ±100.
  - **New scope, about 300 words:** FR-7's no-shrink clause. It covers the §3.1 bullet, P2's
    `n_slots`, the more-slots branch of P3 *On fail*, S3 step 5, AC-1 step 6 with its pass bullet,
    R2's AC-1 sentence, and the §7 S6 row. This agrees with the architect's ~270.
  - **Rework, about 1,040 words:**
    - Pass 4 criteria 1–3, about 810: S0 P3's 362 words are all new, the §5 fate rule plus table
      is about +450 net, and R2 is about +150.
    - The P4-m1/m2 fixes, about 230.
    - Less the ~150 words cut for P4-m3.

    Rework growth is reported as P5-m5.
- **Requirements, about +150 words.**
  - The FR-7 sentence and the AC-1 clause are new scope, directed by the stakeholder.
  - The Out of scope edit is rework done in place (+~10).
  - The three decision-log entries are append-only by rule.

  There is no restated history and no finding table.

### Stopping condition (Pass 4 Q5)

1. **R2 cites every fact — met in form, not fully in substance.** Every factual sentence carries a
   citation, and I re-traced the code chain:
   - `mcp.py:133` starts one thread per job;
   - `background.py:217–229` schedules one extract per chunk in a single loop;
   - `retry.py:86/122/173`: timeouts are not retried;
   - `ingestion.py:85/138` uses the default policy;
   - `repository.report_document_job_done` sets the document `failed`;
   - `models.json` `timeouts.extraction` is 180;
   - `llm.py` and `transport.py` contain no `stream`.

   Two sentences cite evidence that does not support them: the burst sentence (P5-M1) and "S3
   step 5 checks that early" (P5-M2).
2. **P3 is in S0, and the log holds the re-confirmed choice — met.** P3 is undersized (P5-M1).
3. **§5 classifies the prompt's fate sentences — met.** I re-ran the regex over
   `agent.tank.prompt`. It gives 16 hits in 9 sentences, and each maps to rows 1–7, 9 or 10. Row 8
   is the one the plan names as missed by the regex. `devops-persona.md` has no hits. One residue
   remains: P5-n1.
4. **Words added only for those points — not met as written, and it could not have been.** About
   300 words are the stakeholder's new scope and about 230 are Pass 4's own minor fixes. Both are
   legitimate. The criterion should have read "only for Pass 4's findings and new scope", and under
   that reading it is met.

**Findings outside the points.** Two nits fall outside:
- P5-n2 is in the method text of the P4-m1 fix.
- P5-n3 is a v1.1 interaction between `parallel` 1 and the closed void list, which Pass 4's
  pairwise check missed.

Every major and minor sits inside criteria 1–2 or the new FR-7 scope. P5-M1's root cause is the
probe size Pass 4 itself prescribed. **So the rule fires in letter, on two nits, but not in
purpose.** The plan's fixes are converging; what remains is defects in evidence design. For §5,
the rule's remedy is an independent reviewer, and this pass is one: §5 yields only the nits P5-n1
and P5-n3. I recommend no further dedicated §5 review. The next gate should be a compact Pass 6
limited to P5-M1/M2 and the minors.

### Pass 4 dispositions

- **P4-B1** — fixed.
  - R2 is corrected with the code chain (rechecked above).
  - P3 is added.
  - The stakeholder was re-asked (log entry 2).
  - §1 now says `parallel` 1 changes how `:8200`'s calls queue.

  Residue: P5-M1.
- **P4-M1** — fixed. There is a two-axis rule plus a 10-row table, and the sweep maps every hit
  (criterion 3). Residue: P5-n1.
- **P4-m1** — fixed (S3 step 3). Residue: P5-n2.
- **P4-m2** — fixed (the S0 step 2 revert rule, and R1's last sentence).
- **P4-m3** — fixed. §2.2 *Memory* and *KV pool* cite their sources instead of copying them, S0
  step 1 cites §3.1, and the meta-prose is gone (grep finds no "one statement").
- **P4-m4** — fixed (log entries 1–2).

### Requirements change (brief Q2)

- **FR-7's new sentence** is solution-free and testable through AC-1's overlap clause. It could be
  tighter: "every request can use the copy's full context, whatever else is in flight".
- **The plan's premise is false** (P5-m2): a non-unified KV cache also meets the clause. So the
  requirement does not fix a solution, though this GPU's memory makes one slot the only practical
  choice.
- **The Out of scope bullet is consistent** with FR-7, since "whatever it takes to meet FR-7" now
  covers `parallel`.
- **The decision log has the required form.** Each entry is one question → one answer, with its
  FR/AC. Entry 2 names the entry it refines. The nit is P5-R2.
- **"Still reaches `ready`"** belongs, with a condition: P5-R1.

### New findings

**P5-M1 (major) — P3 is sized in documents, but the queue is per ingest event, and an ordinary capture batch already exceeds the 180 s budget at `parallel` 4.**
- **The plan's own numbers re-run correctly.** Its query gives 94 `Fact:` documents, at 54/30/6/4
  for 1–4 chunks, so the maximum is 4. The seven non-`Fact:` documents of 5–10 chunks were all
  created on 2026-09-19.
- **What queues is concurrent extraction calls per ingest event, not chunks per document.**
  Grouped by `createdAt` second, the largest capture event is 8 one-chunk documents at 2026-10-26
  08:50:23 (−03): one batch ingest. Per minute the largest is 9 (2026-10-17, K-071's re-drive).
  KB-migration batches reached 21 chunks in one second.
- **That 8-call batch already hit the timeout at `parallel` 4.** The LM Studio server log shows 8
  `Running chat completion` receipts at 08:50:23 and six `Generated prediction` lines by 08:53:10.
  Then come two `Client disconnected. Stopping generation` lines at 08:53:23, exactly 180 s after
  receipt. 2 of those 8 documents are `failed` today.
- **"Starts with `Fact:`" is narrower than FR-7's everyday use.** `cobb`'s KB sync
  (`skills/agent-maintenance/SKILL.md` §5) re-ingests KB claim documents. Those are non-`Fact:`,
  and in this corpus they run to 10 chunks.

So P3 at 4 calls can pass while traffic already observed fails, and the stakeholder's conditional
decision (log entry 2) would rest on that result.

*Fix:*
1. Size `M` as the total chunks of the largest capture ingest event (a same-second `createdAt`
   group; 8 today). Ingest that shape, or one `M`-chunk document, which gives the same queue
   depth.
2. From P3's serial per-call times, report `k*`, the largest number of queued calls that fits in
   180 s. Compare it with the observed depths: 8–9 for captures, 10 for one KB claim, 21 for the
   migration batch.
3. R2 states the `parallel` 4 baseline: the timeout is already hit today, and `parallel` 1 moves
   the point where it is hit.
4. `tico` re-asks log entry 2 in event units, because "a document as large as a typical learning
   capture" measures the wrong quantity.

*Verified: the queries, the log lines and the document statuses. `k*` at `parallel` 1 is not
measured.*

**P5-M2 (major) — Both overlap checks test only `tank` waiting behind `:8200`, so R2's claim that S3 step 5 checks the reverse is unsupported.**
- **Every extraction is queued before `tank` starts.** `_schedule_chunk_processing` starts every
  chunk's extract thread in one loop (`background.py:217–229`). LM Studio logs all their receipts
  in the same second (the 2026-10-26 batch above). S3 step 5 and AC-1 step 6 both start `tank`
  once the first extraction is running, so the other `M`−1 calls are already queued ahead of
  `tank`'s first request.
- **So neither check can fail because of `tank`.** With one slot and first-in-first-out order, no
  extraction ever waits behind a `tank` completion. FIFO is inferred: LM Studio documents only
  that excess requests are "queued". The document's `ready` then just re-tests P3, and AC-1's new
  "still reaches `ready`" clause cannot fail because of `tank`.
- **The reverse case is the one that matters.** It is the cost R2's second bullet names, and it
  is where the timeout bites. An extraction that arrives mid-run waits for one `tank` completion
  (an ~11–15k-token prompt) plus its siblings.

*Fix:* S3 step 5 is not bound by AC-1's literal order, so make it run the reverse case. Start the
TP-001 run, and when the log shows its second completion has started, ingest the `M`-chunk
document. Pass = `ready`; record each extraction's wait. Keep AC-1 step 6 in the requirement's
order, and make R2 cite S3 step 5 for this cost only. *Receipt timing verified in the log; the
queue order is inferred.*

**P5-m1 (minor) — The `n_slots = 1` evidence (P2, AC-1) cannot be observed as written.**
- The line is printed at load time to `lms log stream -s runtime` (fit report, "Full-GPU load").
- It is not persisted: `.lmstudio/server-logs/` and `LM Studio/logs/main.log` have 0 hits.
- P1 and AC-1 steps 1–3 start no runtime stream before the load.
- In AC-1 step 6, "still reads" can only fail through a reload, and the continuity check already
  catches a reload.

*Fix:* use `/api/v1/models` `config.parallel == 1`, which is already polled, as the evidence. Or
start the runtime stream before P1's request and before AC-1 step 2, and record `n_slots` and
`n_ctx_slot` once, at load.

**P5-m2 (minor) — "That holds only at one slot: any higher count shares the pool" (§3.1) is false.**
- LM Studio's 0.4.0 release notes make Unified KV Cache a load option, "enabled by default".
  Disabled, "preallocated resources will … be hard-partitioned per concurrent request".
- Every load observed on this machine had it on (`kv_unified = 'true'`), and `lms load --help`
  has no flag for it.
- So `k` slots at `k`×16,384, non-unified, would meet the no-shrink clause without queueing. The
  cost is roughly `k`× the KV memory, on a GPU that already likely spills.

*Fix:* §3.1 names this as a rejected alternative (reason: memory), and P3 *On fail* lists it among
the stakeholder's options. *Verified against the release notes. The per-slot size when it is off
is llama.cpp behaviour, inferred.*

**P5-m3 (minor) — Two sentences still treat `parallel` as outside FR-7.**
- §3.3 says the `WARNING:` is not a refusal "because FR-7 does not depend on `parallel`". Now it
  does, by §3.1's own bullet.
- Its second reason, "refusing would force an unload", applies equally to the context row, and
  that row does refuse.
- The rollback bullet reverts `parallel` 1 without saying that this breaks FR-7's no-shrink
  clause.

*Fix:* `architect` decides between warning and refusing on the FR-7 basis, and restates the
reason. Rollback says that reverting `parallel` needs a stakeholder decision.

**P5-m4 (minor) — S3 step 5 lacks P3's guards against load churn.**
- S3 step 4's resize leaves qwen JIT-loaded.
- "As S0 P3" does not say whether P3's explicit embedder load carries over, or its rule that "a
  `failed` for any other cause is not a verdict".
- Churn is a live cause. 13 of the 94 capture documents are `failed` today. For 10 of them, a qwen
  `Failed to load model … Operation canceled` or `… Engine protocol startup was aborted` line falls
  within the same minute. 2 are the timeouts above, and 1 is unchecked.

*Fix:* S3 step 5 loads the embedder explicitly and names P3's non-verdict rule.

**P5-m5 (minor) — Rework growth, about 1,040 words (growth-stops check).** Most of it is rule text
the review demanded. About 120 words can become citations or move elsewhere:
- the parenthetical list of dependent steps in P3 *On fail*;
- §6.2 AC-1's preamble, which restates §2.2's auto-evict rationale;
- S3 step 3 *Waits*' scheduler reasoning, which belongs in the BACKLOG item it already routes to.

**P5-n1 (nit) — Row 5's class contradicts the tense rule.** "the command is denied" is present
tense, so by the rule above the table it is policy, but the row says event. The support column
already handles both readings. Relabel it "system · event or policy (narrative present)".

**P5-n2 (nit) — Name the log lines that P3 and S3 step 3 measure with.**
- In the server log:
  - `Running chat completion` marks receipt;
  - `Prompt processing progress: 0.0%` marks slot start;
  - `Generated prediction` marks the end;
  - `Client disconnected. Stopping generation` marks a client timeout.

  All four appear in the 2026-10-26 window. Queue wait = start − receipt.
- S3 step 3's hold needs ⌈330 / `t`⌉ + 1 requests queued at once before `tank` starts, with `t`
  taken from P2. Sending them one after another does not hold the slot.

**P5-n3 (nit) — The closed void list has no cause for another consumer's request on the phase's instance.**
- At `parallel` 1, a stray `:8200` extraction in phase Q queues ahead of `tank`.
- In phase M, the same extraction JIT-loads qwen, which is already a void.
- If S3 finds that `chunkTimeout` aborts, a long enough burst becomes a scored qwen failure.

*Fix:* add a void cause: "a request not from the run, received by the instance during it (the
server log's receipt lines)".

**P5-R1 (minor, requirements) — AC-1's "the document … still reaches `ready`" belongs, but it needs a stakeholder line and a stated setup.**
- **It is not invented scope.** It traces to FR-7's first sentence: "`:8200` and `tank` … both
  work against that same copy".
- **But no decision-log entry asks for it, and `tico` added it.** At `parallel` 1 its failure mode
  is a timeout, which log entry 2 reserves to the stakeholder.
- **Everyday `ready` also fails today for causes outside this feature:** P5-m4's churn, and R6.

*Fix:* add one decision-log line confirming the clause. AC-1 should also say the overlap case
holds with both models loaded so that neither evicts the other, as the plan's probe arranges.

**P5-R2 (nit, requirements).** Write "with no context-length error" instead of "without losing
context". Log entry 2 cites "the plan's S0 probe": a requirements document citing a plan's step
ID makes the requirements depend on the plan. The condition stands without that pointer.

### What's solid

- R2 is now accurate end to end, and I verified every link.
- The fate rule is principled, and the table covers the prompt mechanically.
- These are all sound: S0 running first as a gate, the revert on an R1 stop, and the P3 fail path
  halting the steps that depend on it.
- The plan's stated numbers re-run correctly:
  - 94 / 54 / 30 / 6 / 4, and seven 5–10-chunk documents from 2026-09-19;
  - 14,336 and 18,432;
  - qwen at 8192, `parallel` 4, no TTL;
  - the custom global default of 8192;
  - nemotron's as the only per-model file;
  - auto-evict on;
  - wrapper tests 23/23.

### Open questions

- **For `tico` (P5-M1):** what size should P3's pass/fail use? The options are the largest
  capture ingest event (8) or the 10-chunk KB claim as well. It is the stakeholder's call. I
  recommend 8 for pass/fail, with `k*` reported against both.

No `REVIEW.md` line is proposed. P5-M1's root cause, a probe size that the review prescribed
without running it, is already covered by the evidence bar ("a fix you suggest is a claim: run
it").

## Pass 6 — 2026-10-31, plan v1.3

**Scope.** Plan v1.3 (untracked, 8,724 words), read in full against Pass 5. I verified its claims
against the tree, not against the architect's account. That covered:
- `falkor-chat/server/falkorchat/mcp.py` (`ingest_document`, `ingest_documents`,
  `_default_schedule`), `background.py` and `chunking.py`. I ran `split_into_chunks` on synthetic
  text, offline.
- `ws:agent-team` through `mcp__cypher__query`, including the plan's P3 query, run verbatim.
- A whole-plan grep for `warn`, `parallel` and `n_slots`.

The requirements file is unchanged since Pass 5 (mtime 15:10, before this plan revision). I did
not touch LM Studio and ran no git commands.

**Verdict — plan: approve with suggestions.** Every Pass 5 finding is fixed. This pass finds no
blocker and no major: one minor (P6-m1, a precondition for the explicit embedder load) and four
nits, all of the sweep kind. **Verdict — requirements: unchanged, approve with suggestions.**
P5-R1 and P5-R2 are still open with `tico`, and the plan carries P5-R1 as an open question
(§8).

CPG: considered, not relevant — the change is bash/JSON under `opencode/agents/tank/` (no
`cpg_opencode` graph) plus LM Studio configuration; falkor-chat code was read only to check the
plan's claims.

**Growth-stops check.** The plan went from 8,738 to 8,724 words (−14). No new scope was added.
The rework is the `D`/`k*` probe, `TANK_REQUIRED_PARALLEL`, the reverse-order check and the void
cause. It is more than offset by P5-m5's cuts: P3 *On fail*'s list of dependants, AC-1's
preamble and S3 step 3 *Waits* are all shortened. So there is no rework growth. One restated
fact remains: P6-n3, "parallel 1" restated next to its new one-home constant. The revision note
is one dated line, and there is no finding→resolution table.

### Pass 5 dispositions

- **P5-M1 — fixed, differently from my suggestion, and the alternative holds.** P3 ingests one
  document with max(`D`, 21) chunks and measures `k*` from the job outcomes.
  - *Same queue as a batch:* yes, on the MCP path. Both `ingest_document` (`mcp.py:317`) and
    `ingest_documents` (`mcp.py:359`, a per-document loop) call `_schedule_chunk_processing`, and
    `_default_schedule` starts one unbounded daemon thread per job (`mcp.py:53–133`). The REST
    path is capped by anyio's ~40-thread limiter instead, which doesn't matter at ≤ 40.
  - *Exactly N chunks:* yes. `split_into_chunks` is pure and deterministic (size 1000, overlap
    150, paragraph-first). `N` paragraphs of 879 chars gave exactly 8 and 21 chunks (ran). The
    500,000-char cap is far off.
  - *The `k*` logic:* sound. Each call has its own 180 s from a shared receipt second, so the first
    `k*` in service order succeed, whatever is queued behind them.
  - *The 8 / 10 / 21 framing:* I re-derived it. The largest second since 2026-09-20 is 8, and the
    largest per-document chunk count is 10.

  Residues: P6-n1 and P6-n2.
- **P5-M2** — fixed. S3 step 5 starts `tank` first and ingests at its second completion's slot
  start, so the extractions queue behind a `tank` completion. R2 cites S3 step 5 for that order
  only (§8 R2). AC-1 step 6 keeps the requirement's order.
- **P5-m1** — fixed. `n_slots` is gone (grep: 0 hits). AC-1's pass criterion is `config.parallel`
  1 in every polled snapshot.
- **P5-m2** — fixed. §3.1 names the Unified-KV-off alternative and rejects it (memory), and
  P3 *On fail* lists it.
- **P5-m3** — fixed, by reversal to a refusal. My ruling and the per-hit sweep are below; the
  residue is P6-n3.
- **P5-m4** — fixed. S3 step 5 loads the embedder explicitly and applies P3's non-verdict rule.
  The residue is P6-m1.
- **P5-m5** — fixed (see the growth check).
- **P5-n1** — fixed. Row 5 is relabelled, and rows 1–2 are merged.
- **P5-n2** — fixed. P3 names all four log lines, and S3 step 3 queues ⌈330/`t`⌉+1 requests
  concurrently.
- **P5-n3** — fixed (§5's fifth void cause). The residue is P6-n4.
- **P5-R1** — not fixed; it is `tico`'s. The plan lists it as an open question and names the two
  steps an answer would change.
- **P5-R2** — not fixed; it is `tico`'s. The requirements still say "without losing context"
  (line 86).

### P5-m3's reversal (brief Q2)

**Ruling: the reversal is within the architect's remit.** The requirements fix the setting (log
entry 1: one slot) and the property (FR-7's no-shrink clause). They say nothing about how `tank`
reacts to a copy that breaks either, and no decision-log entry mentions warning. Refusing makes
`parallel` behave the way the context floor already does. That is consistent: both are FR-7
properties. The one consequence the stakeholder should see is that any `tank` run on a copy with
`parallel` ≠ 1 now exits 1, including a scheduled run. The S6 `HISTORY.md` entry should say so in
one clause.

**Sweep.** I grepped `warn` and `parallel` across the whole plan and ruled on every hit:
- **Consistent with the refusal:**
  - the §3.3 table row and its rationale (lines 231, 236);
  - §3.1's "one home" sentence (136);
  - S2 Troubleshooting, "one row per preflight refusal" (370);
  - Phase M and Phase Q require `OK`, and their old "no `WARNING:`" checks are gone (495, 503);
  - §6.1 case 9 (`parallel` 4 → exit 1);
  - R2 *Detection* (785) and R3 (788);
  - Rollback (813);
  - the only remaining `WARNING:` is the bypass (240, case 11), which is correct.
- **Not affected:** line 77 (an `lms` warning), line 800 ("early warning"), and the S0, S3, S5b
  and AC lines that set or assert `parallel` 1.
- **Residue:** P6-n3.

### Same-revision fixes against each other (brief Q3)

- **The refusal against steps that run with other settings: no conflict.**
  - Every step that runs `tank` loads at `parallel` 1:
    - S3 starts from a preflight `OK`, and its resize keeps the defaults;
    - Phase M uses `--parallel 1`, and Phase Q and AC-6 use the defaults;
    - S5b's hand load passes `--parallel 1`.
  - S3 step 3's `curl` holds don't go through the preflight.
  - The only path back to `parallel` 4 is P3 *On fail*, which halts S1–S6 before the preflight
    exists.
- **The reverse-order S3 step 5 against the new void cause: no conflict.** The void list is §5
  only. S3 step 5 is an overlap on purpose, outside the trial.
- **The explicitly loaded embedder against the auto-evict assumptions: one gap, P6-m1.** Once the
  embedder is explicit, qwen is the only JIT model. AC-1's continuity argument and AC-6's
  rationale both still hold.

### New findings

**P6-m1 (minor) — The explicit embedder load assumes no JIT copy of it is resident, and today one is.**
- AC-1 step 2 says to skip the load "if `lms ps` already shows it with no TTL". It says nothing
  about the case where `lms ps` shows it *with* a TTL.
- P3 inherits that text ("as §6.2 AC-1 step 2 does"), and so does S3 step 5 ("as S0 P3 does").
- By the plan's own §2.2, an explicit load of an id that is already loaded, with `--identifier`,
  fails loudly. Without `--identifier` it makes a `:2` copy.
- Pass 5 read the embedder as JIT-resident (`lms ps` TTL 57m / 1h).

So all three steps would hit either the failure or a second copy, in the state this machine is
usually in. S3 step 5 also never returns the embedder to its prior state, which P3 does.

*Fix:* in AC-1 step 2, add "if it shows a TTL, `lms unload` it first (operator-approved)". S3 step
5 then returns the embedder as P3 does. *Inferred from the plan's §2.2 premise and Pass 5's
`lms ps` read; I did not touch LM Studio this pass.*

**P6-n1 (nit) — P3's re-run query cannot reproduce the number it backs.** Run verbatim, its
`LIMIT 10` returns only 2026-09-19 migration seconds (21, 15, 13 … 11), so the "8 since
2026-09-20" result never appears. Add `WHERE d.createdAt >= 1789862400000` (2026-09-20 00:00 UTC)
after the first `MATCH`. I ran that: it returns 8, 4, 4.

**P6-n2 (nit) — Say how to hit `chunkCount` exactly.** The operator should compute the count
offline with `falkorchat.chunking.split_into_chunks` before ingesting: `N` paragraphs of about
880 chars, separated by blank lines, gives `N` chunks (ran). Fitting the count by trial ingests
instead fires extraction calls into the very queue being measured.

**P6-n3 (nit) — Finish carrying `TANK_REQUIRED_PARALLEL` through.**
- S1 step 2 asks for a header comment on `TANK_MIN_CONTEXT_LENGTH` only.
- S2's README step (363, 365) and the `falkor-chat/README.md` clause (374) restate "parallel 1".
  §3.1 calls the constant its "one home" and has the context value cited, not restated.
- §6.1 has no case for a qwen instance with no `config.parallel`. The fail-safe reading is
  refuse; state it and test it.

**P6-n4 (nit) — The new void cause needs an attribution method.** Server-log lines carry no
request id, and `tank`'s first request and an extraction both log "conversation with 2
messages". Count the receipt lines in the run window against the run's own model calls (the
OpenCode JSON step events). A surplus line is a foreign request.

### Stopping condition

Pass 6 finds no defect in the design or the trial logic. Its findings are one precondition clause
(P6-m1), which v1.1's AC-1 text left and two new steps reused, and four sweep-class nits. The
fixes have converged. v1.4, or the implementer applying these, needs no further review pass if
these five mechanical checks hold:
1. AC-1 step 2 handles a JIT-resident embedder (unload first), and S3 step 5 returns the
   embedder.
2. P3's query carries the `createdAt` filter and returns 8 as its top row today.
3. Every "parallel 1" in S2 and §7 cites `TANK_REQUIRED_PARALLEL`. S1 step 2 comments both
   constants, and §6.1 has a missing-`parallel` case.
4. The void cause names its attribution method.
5. The plan's word count does not grow.

`teco` can check all five with grep and `wc`. **Falsifier:** any later finding in §3.1, §3.3,
S0–S5 or §5 logic that is not in this list shows the plan had not converged. That finding goes to
a full review pass, not a mechanical check.
