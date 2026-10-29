# Review — connected-code-graph P0 implementation plan

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (M10)

## Pass 1

### 1. Scope & verdict

**Reviewed:** `docs/plans/connected-code-graph-p0.md` (U15, 2026-10-29, read in full). I checked it
against proposal v3.3 (§3.12–§3.13, §4 P0 row, §5, §6.2), graph note v2.1, ML note v2, the
requirements (Ready for design), my own review of the proposal (Passes 1–3), the ledger, and root
`AGENTS.md`. Static review, plus read-only probes: the host sandbox, ports and toolchains, the
transcript format, guard hooks, the source behind SL-1/SL-2, and gpg in a throwaway `GNUPGHOME`.
Nothing was written to FalkorDB and no container was started.

**Verdict: needs changes.** There is one blocker, and it sits in the ground-truth chain (§6,
S40–S47). The isolation protocol can't hold as written. The capture–recapture design behind it is
sound and matches ML §8.2. The rest of the plan is grounded and well-structured, and it covers the
proposal's deliverables (a)–(h) and every §6.2 item. Its majors are fixable in place with a v1.1
revision. They do not need a redesign.

**What can be dispatched now:** S01, S02, S05 and S07, plus S03 and S04 once they are split into
separate dispatches (m2). None of them depends on anything this review asks to change. Hold S06
until M2's locator decision is placed. Hold S40–S47 until B1 is fixed.

CPG: considered, not relevant — the plan's subject is new code in `cpg/` and docs that no loaded graph covers. The one code-level grounding check (SL-1/SL-2 against `salesperson/src/api/` and `falkor-chat/server/falkorchat/`) was done by reading source, because `cpg_falkorchat` is 21 server commits stale and has no TS half.

### 2. Findings

#### Blocker

**B1 — The §6 isolation can't hold while the key writers run as ordinary subagents of the build
session, and the audit can't detect the breaches that result.** (verified)
Each key writer (KA, KB, QW, AN1, AN2) is planned as an `Agent` dispatch in the same session as the
builders. In that setting:
1. **The working directory is the repo.** Bash `cwd` resets to the repo root on every call, and
   `Grep`/`Glob` default to cwd. So "work only in `kit-src/`" is a request the tools work against.
2. **The scratchpad is shared.** All subagents of one session share one scratchpad, and every
   doc-write guard auto-allows `/tmp/*`. KA's scratch notes can therefore be read by KB, which
   breaks the independence that capture–recapture assumes. I am reading the architect's probe
   files from my own scratchpad right now.
3. **The agents' own definitions pull the other way.** Both `qa-engineer.md` and `analyst.md` tell
   the agent to use `cpg_<component>` and to write learnings to `ws:agent-team` through
   `ingest_document`, the same knowledge base the builders search. §6.3 forbids `search`/`get`
   but not `ingest_document`, and the §6.5 grep list leaves it out. So a key writer's learnings can
   flow straight to the P1 pack authors.
4. **The guards fight the writes.** The analyst guard escalates every write outside `docs/reviews/`
   to the human (`claude/analyst/hooks/guard-review-doc-writes.sh`), so every KB and AN1 key write
   stops for approval.
5. **The audit is wrong in both directions.** Every transcript line carries `"cwd":"/home/mauricio/prg/graphmind-ai-lab"`
   (169 of 247 lines in U15's transcript; see the appendix). A grep for the repo prefix therefore
   hits every agent. A relative-path read of the repo (`grep -r X .`) contains no prefix at all and
   is missed.

**Fix (route to `architect`, with `cobb` for the agent side).** Run the five writers *outside* this
session. Make it a headless session started with its cwd in a per-role root
(`~/.local/share/ccg-eval/<E>/<role>/`, holding its own `kit-src/` copy and its output only). Because
`.mcp.json` is project-level, that session gets neither `cypher` nor `ws:agent-team`, and the
repo's `CLAUDE.md` is not loaded. Disallow Web and Agent with CLI flags, and use a dedicated
key-writer agent definition with no learning-capture step (a `cobb` unit). Rewrite §6.5 as a parse
of `tool_use` inputs, with the paths resolved, from that session's own project directory. Add a
new S40 sub-step **before** S41: run one throwaway writer and show that a planted read of the repo,
of a sibling role's directory, of `~/.claude/` and of the KB is each refused or flagged by the
audit.

#### Major

**M1 — The namespace fallback sandbox is not offline on WSL.** (verified) Inside `unshare -rn`,
`/mnt/c/Windows/System32/cmd.exe` launches, because WSL interop is kernel-wide binfmt, and
`curl.exe` is present. A Windows process uses Windows' network stack. `docker.sock` is also visible.
S08 test 1 probes only TCP, DNS and the LM Studio port from bash, so it would pass while the
property S04 contracts ("no route out") is false. I did not send an outbound request.
**Fix:**
- S04 states the property per backend.
- The namespace backend runs with a minimal bind set: no `/mnt/*`, no `/var/run/docker.sock`,
  no `/run/WSL`. It refuses to start when `/proc/sys/fs/binfmt_misc/WSLInterop` is enabled.
- S08 replaces test 1's three probes with a **probe over every escape axis**, run on both backends:
  IP, DNS, host Unix sockets, WSL interop exec, and mounted Windows paths.

**M2 — The locator and id form that the keys encode is decided by S03, but S06 and S40 don't
depend on S03. Two anchor kinds the gold needs also have no tuple.** (verified from the text)
- §5.3 says S03 decides "the locator form" that S06 encodes. In the table, S06 has `Deps —` and runs
  in wave 0 beside S03, and S40 depends only on S06.
- D8 writes the gold anchor as `(lang, repoPath, qname[, signature])`, but S03's tuple is
  `(lang, repoId, qname…)`. `repoId` for the monorepo is a manifest value that nothing in P0 fixes.
- S03's `qname` is not specified in a form a source-only reader can produce. The writer must
  produce it without Joern (§6.3). For example, the health handler is nested inside
  `build_storefront_router()` at `storefront_api.py:937`.
- S03's tuple table has no `ContractPoint` (route or tool key) row and no `keyNorm` rule. AC-24's
  questions anchor on those.
- F5's "id-equivalent without a snapshot" holds only once these are fixed.

**Fix:** S03 adds a source-derivable qname grammar per language, a fixed `repoId` for the proving
repo, and `ContractPoint`/`keyNorm` tuples. Then add S03 → S06 and S03 → S40 edges.

**M3 — The P1 HTTP stratum is being developed against before it is held out.** (inferred) The
slice (SL-1, SL-2, SL-4) and S30's FastAPI and fetch walkthroughs are built and tested on the real
`salesperson` → `falkor-chat` routes, which are exactly the main population of the HTTP stratum.
S47 splits by anchor at random. A held-out item whose anchor the slice hard-coded was "used for
development" (AC-5's text and ML §4.1's leakage rule), even though the key was written blind.
**Fix:** S06 adds a rule that every anchor touched by the slice, the S30 walkthroughs or fixture
tests is forced to dev. S47 reports how many held-out anchors are left per stratum, and that
number feeds SQ-P0-1. The rule belongs to `data-scientist` in S06.

**M4 — Several "P0 measures this" deferrals in the notes have no owning step and no named
handoff.** (verified against the notes) Answering question 1, these fall between documents:
- graph note Q3: "P0 must re-measure on real bundles from each FR-1 frontend, especially Java/C#
  call-site density, `Link` counts under runtime-built URLs, and `LogTemplate` volume". S28
  records only projection density.
- graph note §6.4's "remaining risks, all P0-measurable": (2) BFS state space under field paths,
  and (3) `RAN` volume.
- ML §3.6: "revisit [0.98] after P0 measures typical chain lengths". No step measures chain length.

**Fix:** give each one an owner, or list it in §14 as a handoff with its reason. For example:
- S28 records symbol and call-site density per language;
- S14 records the `Link` count and the BFS frontier state count;
- S47 reports the hop-length distribution of the key items;
- `LogTemplate`, `RAN` and the 18 GB peak are handed off to P3/P5 explicitly.

**M5 — The data-scientist steps S40, S45 and S47 conflict with that agent's definition and its
guard.** (verified) `guard-ds-doc-writes.sh` allows `docs/plans/` and `docs/reviews/` only, and the
definition says "advisory-only … never source, tests, config, or data". S40, S45 and S47 have it
write:
- `cpg/eval/**`;
- the kit-export, gold-strip and sealing tooling;
- `~/.local/share/ccg-eval/…/work/*.json`;
- a `docs/test-reports/` file.

Each write escalates to the human, and writing through Bash instead would be a guardrail breach.
**Fix:** split each step. `data-scientist` specifies the procedure and checks the output, and a
non-builder implementer runs the tooling. That implementer can be a fresh `qa-engineer`, which is
already on §6.5's decrypt list. Name the tooling files: S40's deliverable column lists none,
though §6.4 relies on "S40's tooling" to strip the gold. Alternatively, `cobb` widens the guard.
Either way, decide before S40.

**M6 — S26 is one dispatch of about 1,050 blind labels across 7 languages.** (inferred) At ≥150
sites per language (§4.3), one `qa-engineer` reads unfamiliar Java, Kotlin, C#, Go and C++ code
under the "fresh instance, never resumed" rule. A run killed by a usage limit, which has happened
twice in this ledger (U2, U11), would lose all of it. It is the largest dispatch in the plan and
the reason is not stated (question 4). The same applies, more mildly, to S41/S42, a whole census
in one run each.
**Fix:**
- Make S26 one dispatch per language. That also lets them run in parallel.
- Allow S41/S42 one fresh instance per (role, stratum).
- Change the rule to "never reused for another role", so a killed writer can be **resumed** in
  its own role. Each writer appends to its output file as it goes.

#### Minor

- **m1 — Step ids collide with stakeholder-decision ids (question 2): yes, the ambiguity is real.**
  - Seven sites cite decisions with a bare `S<n>` that is also a step id:
    - l.296, S46's done condition "(S36; ML §8.3)", where step S36 is the fx-svc fixture;
    - the §6 heading, l.764, l.766 and l.779 ("S36");
    - F4 (l.1011, "S36");
    - F12 (l.1019, "S35, S37").
  - The proposal, which implementers read alongside, also cites S14, S19 and S34 as decisions.
  - **Fix (least disruptive):** write decisions as "decision S36" at those 7 sites, and add one
    legend line under §3: "`S01`–`S51` are steps; stakeholder decisions are written *decision
    S<n>*". Renaming about 700 step references costs more than it buys. Ledger units should cite
    the `P0-nn` unit id.
- **m2 — Bundling S03 and S04 lengthens the critical path.** S08 needs S04, which unit P0-03 queues
  behind S03. Split them into two parallel `architect` dispatches.
- **m3 — Missing edges and shared files** (question 3; full list in Appendix B):
  - S16 should gate S30, S33 and S34 when it runs.
  - S39 lists S10 as an input but not as a dependency.
  - `cpg/infra/README.md` is written by S01 and S02, which run in parallel in wave 0.
  - `pyproject.toml` writers are missing S10, S34 and S20–S22.
  - S25 can write `.ccg-work/` before S09's `.gitignore` exists.
  - S08's JVM test is missing from S20's no-overlap list.
  - §5.3's S27 → S35 row contradicts §4.7's "plant in both", so drop the row.
- **m4 — The forbidden-flag list is a hand-made list** (F11 found one flag that was missing). The
  same `--help` output also shows `kotlin2cpg --enable-dependency-resolver-v2` (or its environment
  variable), `--generate-nodes-for-dependencies` and Gradle-download options, which nobody has
  classified. **Fix:** S08 adds a test, *before* S25 uses the list. It sweeps every frontend's
  `--help` for `fetch|download|url|dependenc|resolver` and fails on any flag that is neither in
  `forbidden-flags.txt` nor on a reviewed allow-list. Test 2 also covers `--flag=value` and the
  pass-through after `--`.
- **m5 — Text checks stand in for behaviour** (question 8). Each needs a behavioural check:
  - SL-2's "a test greps for `/shop/api`" doesn't catch `'/shop' + '/api'`. Add a counterfactual:
    stage a copy with `API_PREFIX = "/x/api"` and assert the key becomes `GET /x/api/health`.
  - S01's "127.0.0.1 only" should be checked by `docker port`. `falkordb-dev` itself publishes
    `0.0.0.0:6379`.
  - S47's "no held-out content" needs a mechanical check: grep the report for every held-out
    locator and expect zero hits.
  - The kit exclusion needs a post-export grep for `connected-code-graph|C-1001|ccg` (watch
    `docs/HISTORY.md` and the quirks file at `EVAL_COMMIT`), not a path list.
- **m6 — SQ-P0-1's recommended option (a) contradicts F7.** A sealed fixture repo is still not "in
  the proving repo", so (a) amends AC-5. Say so in the question, and add the options that stay
  inside AC-5: a wiring stratum (P1 builds `.mcp.json`/Compose extractors), or runtime-oracle
  misses. The "55 files" estimate is also stale. 129 files now name `mcp__cypher__query`, 120 of
  them Markdown, and documentation snippets don't count (decision S21).
- **m7 — Capture–recapture gaps for S06 to close** (question 5). A third census pass makes
  Lincoln–Petersen inapplicable, so S06 must name the multi-list estimator. The gold-strip step
  before S44 has no owner (M5).
- **m8 — `TIMEOUT_MAX` sizing.** It is derived from 1,000-file latencies but is instance-wide and
  applies to every credential (V18). Its effect on load writes and on `GRAPH.CONSTRAINT` is
  unprobed. S22 should probe both and label the value as a P0-scale default to re-derive at P1.

#### Nit

- n1 — §10 is missing the `Extended by:` pointers the convention requires: the proposal gets one
  for each of the four specs, the graph note for p0-graph (this also gets F1's correction seen),
  and the ML note for the two `-ml` notes.
- n2 — F12 says "100 English questions". ML §4.4 is in fact bilingual-optional. The disposition
  still stands.
- n3 — D11 puts the passphrase in a file, while §12 makes custody by the stakeholder the default.
  Pick one. Encrypting and decrypting with loopback and a passphrase file did not cache the
  passphrase in gpg-agent (probed). Interactive pinentry may cache it, so use `--no-symkey-cache`.

### 3. The plan's own findings and questions (question 6)

| Item | Diagnosis | Disposition |
|---|---|---|
| F1 | Correct (graph note §6.1 item 6 verified) | OK; add the graph-note pointer (n1) |
| F2 | Correct (§3.4 vs §6.4 verified) | OK |
| F3 | Correct (the proposal says "v2"; the graph note is v2.1) | OK |
| F4 | A reasonable reading: decision S36 sets the kind of writer, not how many | OK; M5 and B1 change *who* runs it |
| F5 | Right in principle | Incomplete (M2) |
| F6 | Correct and important | OK |
| F7 | Correct (AC-5 text verified) | OK; SQ-P0-1(a) contradicts it (m6) |
| F8, F9, F13 | Correct | OK |
| F10 | Claims about §3.13 and the ledger verified | OK |
| F11 | Correct | Incomplete (m4) |
| F12 | Slightly overstated (n2) | OK |
| SQ-P0-2, K-P0-1 | Correct | OK |

### 4. Environment claims (question 7)

I re-ran the probes. Every §7 claim checks out:
- Joern `io.joern.joern-cli-4.0.579.jar`, with every frontend directory present;
- 5,930 MB RAM, 4 GB swap, 8 vCPUs, and the `.wslconfig` comment about the cap;
- under `unshare -rn`, loopback is DOWN, DNS fails, and TCP to `1.1.1.1:443` and to
  `127.0.0.1:1234` is unreachable, while LM Studio *is* reachable from the host;
- port 6390 is free;
- `mvn`, `gradle`, `go`, `dotnet`, `kotlinc`, `cmake`, `bear`, `node` and `uv` are all absent.

The one claim that fails is the one §7 doesn't make explicitly: that the namespace means "no route
out" (M1). That changes S04 and S08, not the choice of the container as the primary backend.

### 5. What's solid

- **SL-1/SL-2 are grounded.** I read the source. `getHealth` passes `` `${API_PREFIX}/health` `` to
  `apiFetch`, then `buildUrl`, then `fetch(url, {method})`. The `/shop/api` prefix comes from the
  `include_router` call at `app.py:521`. The dev router's `/health` sits behind `if dev_surface`,
  and the bare re-registration is at `app.py:497`.
- **The freeze rule.** D5's binary freeze rule, with a successor document for a schema change, is
  faithful to proposal §4.
- **Splitting (a).** The a1/a2 split (F9) is right.
- **Shared-state discipline.** The §5.1 reservations, the "no 6379 default" rule and the
  serialization tables are the right shape.
- **Ground-truth design.**
  - Pinning `EVAL_COMMIT` (F8) is right.
  - So is the shared flow-anchor sample (F6).
  - The isolation section is honest about what the seal cannot do.
  - The design on paper is consistent with ML §8.2: two strategies, two models, and N̂ as a
    lower bound on incompleteness.

### 6. Open questions

- Is a separate headless session for the key writers (B1) acceptable to the stakeholder?
  It spends usage outside the coordinating session, and `teco` launches it through Bash.
- If not, the fallback is a dedicated tool-restricted agent definition, with `PreToolUse` path
  hooks on `Read|Grep|Glob|Bash` that deny anything without an explicit path under the role root.
  That is `cobb`'s to design.

## Appendix

**A. Probes run for this pass (2026-10-29).**
- Transcript format.
  - `~/.claude/projects/-home-mauricio-prg-graphmind-ai-lab/13504717-…/subagents/agent-a5c76dc11e6c767df.jsonl`
    has 247 lines. 201 contain the repo path, and 169 of them through `"cwd":"/home/mauricio/prg/graphmind-ai-lab"`.
  - `.meta.json` shows `agentType: architect`.
- Interop inside the namespace.
  - `unshare -rn bash -c '/mnt/c/Windows/System32/cmd.exe /c ver'` printed the CMD banner, with a
    `\\wsl.localhost\…\graphmind-ai-lab` cwd.
  - `/mnt/c/Windows/System32/curl.exe` exists.
  - `test -S /var/run/docker.sock` succeeds inside the namespace.
- MCP scope. Both `cypher` and `falkor-chat-agent-team` are defined only in the repo's `.mcp.json`,
  and `~/.claude.json` has no user-scope servers.
- gpg 2.4.4 in a throwaway `GNUPGHOME`. After encrypting and decrypting with
  `--pinentry-mode loopback --passphrase-file`, a later decrypt with `--pinentry-mode error`
  failed, so nothing was cached.
- `git grep -l mcp__cypher__query` gives 129 files: 120 `.md` and 7 code files outside `cpg/`
  and `deprecated/`.

**B. Sequencing deltas (m3).**
- **Add edges:**
  - S03 → S06 and S03 → S40 (M2);
  - S16 → S30, S33 and S34, conditional on S16 running;
  - S10 → S39;
  - S09 → S20–S22 and S25 (the venv and `.gitignore`).
- **§5.2: add rows:**
  - `cpg/infra/README.md`: S01 → S02 → S24;
  - `pyproject.toml`: add S10, S20–S22 and S34 to the writers.
- **§5.1:** add S08, plus the S02/S24 image builds, to S20's no-overlap set.
- **§5.3:** drop the S27 → S35 row.

## Pass 2

**Scope:** plan v1.2 (2026-10-29), U18 re-gate. I read the new §6 in full, together with §3, D6, D11, §4.1, §5.2, §12 and §15. I checked it against `claude --help` (2.1.312), `~/.claude/`, the role-root ancestry, and `env -i` binary resolution. No session was started, no container was started, and no FalkorDB write was made.

**Verdict: approve with suggestions.** B1 is fixed in structure: a headless cwd outside the repo, no MCP, `--restricted`, a path guard, and a resolved-path audit over two sources. Every Pass 1 finding is fixed or acceptably overruled.

The six new findings all come from **the fixes, not from the original plan**: the v1.2 launch construction (P2-1 to P2-4), the n3 change (P2-5), and the SQ-P0-1 rewrite (P2-6). None is silent. Each would surface as a failure in S53 or S54, or as a voided census. They are cheap to fix now and expensive to find there.

**Dispatch:** everything except S52, S53 and S54 may proceed now. Hold those three until the edits below are made.

**Stopping condition for Pass 3.** No analyst pass is needed if v1.3 meets all of these. `teco` can check each one mechanically:
1. `env -i <S53's exact env> timeout 10 <S53's binary> --version` exits 0.
2. `run-writer.sh start` and `resume` build argv from one function. Their dry-run outputs differ only in `--session-id <u>` versus `--resume <u>`, the prompt, and the stream-log name.
3. Every start and every resume writes a distinct stream-log path. S48 has a fixture of a session with two attempts, and the audit consumes both attempts without a false violation.
4. The launch command carries `--tools Read,Grep,Glob,Write,Edit` and `--disable-slash-commands`. The init-check tool set equals the `--tools` set.
5. S54 includes a kill-and-resume case.
6. D11 states the deletion trade-off correctly (P2-5).

After that, §6's remaining unknowns get settled by running S54, not by further review.

### Pass 1 dispositions

- **B1: fixed (superseded by the new §6).** I checked these by reading the help text and probing:
  - the role root has no `CLAUDE.md`, `AGENTS.md` or `CLAUDE.local.md` in any ancestor;
  - `~/.claude/CLAUDE.md` is 0 bytes;
  - `--restricted` and `--strict-mcp-config` do what the plan quotes.

  The residual issues are new findings P2-1 to P2-4.
- **M1: fixed.** D6 states per-axis properties, the namespace backend refuses to start under `WSLInterop`, the container mounts are `noexec`, and S08 test 1 now covers every escape axis. M1 does not interact with the writers: they have no Bash and need no sandbox.
- **M2: fixed.** The S03 row now covers qname, `repoId` and `ContractPoint`. S06 and S40 depend on S03.
- **M3: fixed.** S06 owns the rule. S14 and S30 list their anchors. S47 applies the rule before the split.
- **M4: fixed, and the overrule is confirmed.** A blind writer's stated hop count does not measure the system's evidence-chain length, which is what ML §3.6 needs, so a prior plus a P1 handoff is right. One cheap addition, a nit: S14 can also record the hop count of every `Link` chain the slice finds. That is the system's own chain length on real data, at P0.
- **M5: fixed.** The GC is a fresh `qa-engineer`, whose guard is `pass`. `data-scientist` writes only under `docs/plans/`.
- **M6: fixed.** S26 is now seven dispatches and S41/S42 are per stratum. Resumption interacts with the audit (P2-2).
- **m1: fixed.** No bare decision ids remain. Nit: the legend says "`S01`–`S51` are steps", but the steps now include S43a and S52–S54.
- **m2: fixed.**
- **m3: fixed, and the `pyproject.toml` overrule is confirmed.** One creator declares every dependency up front, and a missing dependency is a stop-and-ask that `teco` serializes. That removes the race rather than ordering it.
- **m4: fixed** (tests 2, 6 and 7).
- **m5: fixed.**
- **m6: fixed.** The new option (e) adds an issue (P2-6).
- **m7: fixed** (§4.3, the multi-list estimator).
- **m8: fixed** (S22 row).
- **n1: fixed** (§10 row).
- **n2: fixed.**
- **n3: the "fix" is refuted in part** (P2-5).

### New findings

#### Major

**P2-1 — The launch command can't start, and its tool flags don't produce the tool set the audit
checks for.** (The start failure is verified; the tool-set mismatch is inferred from the help text.)
- **The binary isn't on the launch PATH.** `claude` is at `~/.local/bin/claude`.
  `env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 timeout 10 claude --version` gives
  `timeout: failed to run command 'claude'` with exit 127. Every start fails. S53's dry-run mode
  never executes the command, so it can pass without catching this.
- **`--allowedTools` sets permission rules; it does not limit which tools exist.** `--tools <tools...>`
  ("the list of available tools from the built-in set") is the flag that limits availability.
  Without it, the session likely keeps the other built-ins, for example `TodoWrite` and `Skill`.
  §6.5 check 1 (tools ⊆ {Read, Grep, Glob, Write, Edit}) would then fail on every run.
- **Skills are not disabled, and the plan's inventory of them is wrong.** `~/.claude/skills/` holds
  **seven** entries, not two, and six are symlinks into the repo, including `cpg-analysis` and
  `joern-cpg`. A `Skill` invocation reads repo content without going through the file tools that
  `--restricted` confines. `--disable-slash-commands` ("Disable all skills and commands") closes
  this directly, which turns S54's open item 1 into a construction.

**Fix:**
- invoke the binary by absolute path, pinned to `~/.local/share/claude/versions/<v>`;
- add `--tools Read,Grep,Glob,Write,Edit --disable-slash-commands`;
- correct the skill inventory;
- S53's done-condition runs the binary-resolution smoke test from the stopping condition.

**P2-2 — Resuming a writer (the M6 fix) breaks the isolation construction and the audit (B1).**
(Verified from the plan text and shell semantics.) Three problems:
- **The flags may be dropped.** §6.1 writes a resume as `claude -p --resume <session-id>` plus a
  fixed prompt. The `--restricted`, `--tools`, `--settings`, `--agents` and `--strict-mcp-config`
  flags are launch-time, and nothing says they are passed again. A literal resume would therefore
  run with Bash and user settings.
- **The first stream log gets overwritten.** `--resume` keeps the same session id unless
  `--fork-session` is given (help). §6.3's `> runs/<uuid>.stream.jsonl` would truncate the first
  attempt's stream log, which is the audit's primary source.
- **The cross-check then fires falsely.** The transcript still holds the first attempt's
  `tool_use` ids, which are now missing from the stream log. §6.5's cross-check counts that as a
  violation, so every resumed census would be voided.

**Fix:**
- one argv builder for both `start` and `resume`;
- per-attempt stream logs, named `<uuid>.<n>.stream.jsonl`, all of which the audit reads;
- **audit the partial stream before resuming**, and never resume a session with a voiding event:
  redo it fresh;
- S54 adds a kill-and-resume case;
- the output tooling tolerates a truncated last line left by `timeout`.

**P2-5 — n3: the passphrase file at rest removes the only control the seal had against an
in-session builder.** (A security trade-off; the claim it refutes is checked against D11's own
text.)
- **Who can decrypt.** Every P1 builder (`coder`, `tdd-engineer`, …) runs with Bash under the same
  uid, so `gpg -d --passphrase-file ~/.config/ccg-eval/heldout.pass` is one line for any of them.
  The realistic leak is not an adversary. It is a builder debugging an AC-5 miss, the path that
  ML §4.1 calls "the most likely way this evaluation fails".
- **The plan's justification.** D11 says interactive custody "would block every gate", because
  agents can't answer pinentry. That is true. But D11 then says that deleting the file between
  gates "changes nothing about what the seal protects". **That is false.** A passphrase present
  only during a gate window can't be used by a builder outside that window. Time-boxed custody is
  fully non-interactive for GC: the stakeholder writes the file at the gate and removes it
  afterwards. Loopback with a passphrase file works without a prompt (probed in Pass 1).
- **Recommendation.** Make delete-between-gates the default. The stakeholder keeps the passphrase
  off the host, which also backs it up: losing it loses both the held-out set and `keys.tar.gpg`.
  If file-at-rest stays, D11 must say that the seal then stops only greps.

  My own Pass 1 n3 ("pick one") was under-specified, and that is how this got lost.

#### Minor

- **P2-3 — K-P0-12's version pin will trip mid-chain.** The versions directory holds 2.1.309
  (10-22), 2.1.310 (10-25) and 2.1.312 (10-28), so three builds landed in six days despite
  `"autoUpdates": false` (`autoUpdatesProtectedForNative: true`). "Refuse to start on another
  version, then re-run S54" would stall the census once or more. The absolute versioned path in
  P2-1 fixes this. Also record the SHA-256 of `agents.json`, `settings.json` and `guard-paths.sh`
  in the run index. They live in the repo, which builders edit.
- **P2-4 — Some "verify in S54" items are relied on before S54.**
  - **Item 5 (OAuth under `--restricted`).** It underpins the stakeholder's §6.6 cost acceptance
    and the reason `--bare` was rejected.
  - **The stream-json and transcript shapes.** They are assumed by S48's synthetic audit fixtures,
    which are built before any real stream exists.
  - **Fix:** have the stakeholder or `teco` capture one trivial real `-p --output-format stream-json`
    run (no repo, no tools) as a fixture before S48. That also settles items 5 and `--verbose`.
  - **Answer to your question about `~/.claude/agents/`:** neither `CLAUDE.md` nor `README.md` can
    reach the session as memory. They have no frontmatter, so they don't load as agents either.
    The symlinked agents would show up at most as `Agent` catalog descriptions, and those are gone
    once `--tools` excludes `Agent`.
  - **Your question about the help text:** each flag's quoted behaviour matches the 2.1.312 help.
    The plan omits one more reason to reject `--safe-mode`: it also disables hooks, which would
    disable S52's path guard.
- **P2-6 — SQ-P0-1 option (e) conflicts with ML §4.1.** It pools runtime-observed edges into the
  held-out set, but §4.1 says oracle edges are "reported next to seeded recall, **never pooled**",
  because they over-represent exercised paths. No P0 step collects them either. S06 may adopt (e)
  only by explicitly revising that rule and naming a collection step.
