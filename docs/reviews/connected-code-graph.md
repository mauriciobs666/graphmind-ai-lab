# Connected code graph — independent review (requirements, plan v2, graph note, ML note)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (`docs/requirements/connected-code-graph.md`)

## 1. Scope & verdict

**Reviewed:** gate U5 of `docs/plans/connected-code-graph-coordination.md`. Four documents, each
read in full:
- `docs/requirements/connected-code-graph.md` (Interviewing);
- `docs/plans/connected-code-graph.md` (v2, "the plan");
- `docs/plans/connected-code-graph-graph.md` ("graph note", G1–G13, Q-R1–Q-R6);
- `docs/plans/connected-code-graph-ml.md` ("ML note", M1–M5, m1–m8, RQ1–RQ11).

**Baseline:** the repo at `f04cbfd5` plus the untracked plan and notes. Docs only: no code exists.
I spot-checked the factual claims against the source, the installed Joern and `cpg_falkorchat`.

**Verdict: approve with changes.** The architecture is sound and well grounded. I independently
confirmed its load-bearing store facts, including G1, the `cypher-mcp` write escalation. It is fit
to go to the stakeholder now. No finding blocks the AC review. **Six majors must be folded in
before P0 freezes any contract**:
- a graph-direction flaw that drops writer→reader impact;
- a flow-summary schema too weak for AC-1's own wrapper;
- wrong or incomplete grounding of AC-1/AC-2;
- egress paths the broker does not cover;
- phase-ordering contradictions;
- a verdict/snapshot-swap race.

The stakeholder question set is complete only after five additions (§5). At 38 decisions it
needs batching to be decidable (finding m7).

CPG: used cpg_falkorchat — confirmed the plan's 82 inline-`MATCH` `query` call sites and its build marker (2026-10-15, `5c82170f`); the AC-1/AC-2 grounding defects (B3) came from reading source, which the CPG of `falkor-chat/server` alone cannot show (the consumers live in `salesperson/`, `kiro/`, `mcp-monitor/`).

## 2. Findings

Severity: **blocker** (wrong/unsafe, must fix) · **major** (works on paper but will hurt; fix before
P0 freezes contracts) · **minor** · **nit**. "Verified" = I ran or read it; "inferred" = from
reading the documents only.

### Major

**B1 — Data edges break the "one direction per question" rule: a writer change never reaches
its readers, and RCA never reaches the writer.** *Verified by reading both documents.*
- Graph note §3.1 rule 1 makes every dependency edge point dependent → dependency, so impact is
  always incoming reach. But §3.2 models both `(:CallSite)-[:READS]->(:ContractPoint)` and
  `(:CallSite)-[:WRITES]->(:ContractPoint)` in the same direction.
- Incoming reach from a writer W therefore never reaches the entity or its readers. That breaks
  plan §3.7 Impact ("a writer change reaches the readers").
- RCA's outgoing reach from a reader stops at the entity, so "the writer of a bad value in another
  component" (§3.7 RCA) is never a candidate.
- The two operator edge lists also disagree. Graph note Q7 traverses
  `CALLS|RESOLVES_TO|CALLS_NAME|HAS_CALL|TO|VIA|READS|WRITES`. Plan §3.7 adds `BOUND_BY`/`LAUNCHES`
  and omits `DEFINED_BY`.
- Shared data entities are also cross-boundary, but the graph note reifies `Link` "for
  cross-boundary links only" and keeps `READS`/`WRITES` as plain edges.

*Change (owner: `graph-dba` with `architect`):* make writes point entity → writer (provider-like),
or reify writer→reader data `Link`s. Publish one normative edge-type × direction table per
operator. *Assertion that catches it:* a fixture "change writer W → reader R in another unit is
listed", and "RCA from R's symptom lists W". Both fail on the model as written. I have not run
either remedy, so the choice stays with the owner.

**B2 — The flow-summary schema is positional and field-insensitive, but AC-1's own wrapper needs
field-sensitive, two-level role lifting. P0 freezes that schema having proven lifting only "on
paper".** *Verified in source.*
- `salesperson/src/api/client.ts:107-118`: `apiFetch(opts)` passes `opts.path` to
  `buildUrl(path, query)`, which returns `url`, which goes to `fetch(url, {method: opts.method})`.
- The key is a **field** of parameter 0, it is lifted through **two** functions, and the HTTP verb
  is another field.
- Plan §3.3 and graph note §3.2 carry only `ARG_FLOWS {param:int, arg:int}` and
  `paramToReturn:[int]`, which cannot express "param 0 `.path` → arg 0".
- Non-Joern sources have no summaries at all. AC-3's FalkorDB DDL is applied through a shell
  wrapper (`falkor-chat/scripts/bootstrap_schema.sh:26` `gquery` → `redis-cli GRAPH.QUERY`,
  50 uses).
- `falkor-chat/server/falkorchat/querygen.py:488` builds labels at runtime.

*Change:* the summary schema needs access paths (param + field path → arg + field path, and into
return). State how shell functions get lifting facts. Then add a **P0 vertical slice before the
freeze**: real bundles for `falkor-chat/server` + `salesperson/`, value sets, lifting, and the AC-1
link computed end to end. Failing that, mark bundle schema v1 provisional until P1 exit. This is
also the plan's largest unmeasured cost: build-time `reachableBy` summaries for every method. P0(a)
times the parse only.

**B3 — AC-1 and AC-2 grounding (§1) points at the wrong code, and misses the binding shapes those
ACs actually need.** *Verified in source; trace in Appendix A.*
- §1 cites `@router.get("/health")` in `falkorchat/api.py`. That is the legacy dev router's
  **bare** `/health` (mounted only when `dev_surface`, `app.py:483-490`).
- The storefront's `/shop/api/health` is a different decorator, at `storefront_api.py:937`. It
  gets its prefix from `app.include_router(..., prefix=storefront_api.API_PREFIX)` in **another
  file** (`app.py:521-524`), not from a module constant.
- The repo has three `/health` providers. Provider keys therefore need a **mount-prefix
  composition** recognizer (decorator path + `include_router` prefix, possibly conditional). That
  is not in §3.3's pack design, nor in P0(d)'s paper test.
- AC-2's falkor-chat MCP consumers bind by **URL**: `kiro/.kiro/agents/falkor-chat-demo.json` and
  `mcp-monitor/config.example.toml` use `http://localhost:8000/mcp`, and `.mcp.json`'s
  `falkor-chat-agent-team` uses `http://localhost:8200/mcp`. The port is set by
  `falkor-chat/scripts/start_agent_team.sh:137`.
- §3.4's binding model covers only Compose env → service → build context. §1's `.mcp.json`
  analysis covers the `cypher` entry, which is AC-4's subject, not AC-2's.

*Change:* re-ground §1 for AC-1 and AC-2. Add "router-mount prefix" and "URL host:port → launch
script/Compose port → unit" bindings to P0(d) and P1's pack list.

**B4 — Local-only (FR-9) has egress paths outside the egress broker, and §3.11 contradicts
§3.10/R17.** *Verified with `--help` on Joern 4.0.579.*
- The frontends §3.2 depends on can download from the network:
  - `javasrc2cpg --fetch-dependencies`;
  - `kotlin2cpg --download-dependencies` and `--jar4import-url` ("URL of service which fetches
    … jars for import names found in the project", which sends import names off the machine);
  - `csharpsrc2cpg --download-dependencies`;
  - `gosrc2cpg --fetch-dependencies`.
- §3.2's table says Java is "best with dependency jars on the classpath". The precise-indexer
  fallbacks generally run package-manager builds.
- Other unbrokered outbound paths: the staleness check needs remote heads (`git fetch`), plus the
  PR adapter (R17) and the hosted access provider (§3.10).
- Under R8's own definition of local-only ("no repository-derived content … including names …
  manifests"), every one of these is egress. Yet §3.11 routes only "LLM, embeddings, any hosted
  service" through the broker, and claims "no other component holds network credentials".

*Change:* the broker's scope becomes *every* outbound connection, build time included. Local-only
builds run offline against explicitly provisioned caches. P0(a) measures frontend fitness **with
dependency fetching off**, which is the local-only configuration, as well as on. AC-8's sandbox
test runs a full refresh, not only a query. Stakeholder question S18.

**B5 — The phase plan contradicts its own gates and omits the work that P1's gates need.**
*Verified against plan §4 and ML note §4.5/§4.7.*
1. **AC-7 at P2 vs SQL at P3.** ML §4.5 ground truth includes an SQL/cache stratum, and AC-7's text
   requires SQL. But embedded SQL, schema extractors and NoSQL/cache packs land in P3. With
   pre-registration (M1/RQ7), the P2 first contact either has no SQL support or is not the first
   contact for that stratum.
2. **AC-22 at P2** requires "all five use cases then answer across it". At P2 only impact exists
   (review/test map are P3, RCA/security P5).
3. **AC-10.** The analyst-review seam (§3.12) needs overlays (P3), but only P1 (impact) and P5
   (RCA/QA/security) list `teco` integration.
4. **Fixture system.** The 3-repo fixture (§5.1) and the census/planted strata are in no phase's
   deliverables, yet P1 claims AC-5 (planted shapes) and AC-6, whose ground truth is the planted
   fixture (ML §4.6).

*Change:* gate AC-7 after P3, or pre-register per-stratum first contacts explicitly. Re-assert
AC-22 at P5 exit. Add review-stage `teco` integration to P3. Put the fixture system and the P1
ground-truth strata (with their D4 producer) into P0/P1 deliverables.

**B6 — A verdict recorded during a build can vanish until the next refresh, and the log and live
patch are not atomic.** *Inferred from plan §3.8 + graph note Q6/§3.5. Not executed; no code
exists.* The sequence:
1. Assembly reads the JSONL log at build start.
2. A `verdict` issued mid-build appends to the log and patches the **old** live key.
3. `RENAME` then replaces that key with a snapshot assembled without it.

The dismissed link reappears for every client until the next full refresh. That breaks adopted
Q-R2 ("immediately") and makes R18's verdict-state marker inconsistent with the facts. Separately,
"appends to the file **and** patches the snapshot in one write" is two stores. A crash between
them diverges silently until the next assembly.

*Change:* stamp `:Snapshot.verdictLogOffset` at assembly. After loading, and under a lock shared
with the `verdict` command, replay log entries past that offset onto the new key before `RENAME`.
Derive the R18 marker from that offset. *Test:* issue a verdict while a build runs; after the swap
the verdict is present.

### Minor

**m1 — The `Name`-indirection hop is not a plain BFS node.** A `DISMISSED` pair is
(CallSite, Symbol), but the path `Symbol←RESOLVES_TO−Name←CALLS_NAME−CallSite` spans two BFS
levels. Graph note Q7's per-level query cannot anti-join it, and a visited `Name` loses which
Symbol it came from. *Change:* expand `Name` as one compound hop carrying the anti-join, and keep
`Name` out of the visited set. Also specify whether **resolved** call sites also get `CALLS_NAME`.
If they do not, a wrong Joern resolution is a silent miss plus a false *confirmed*. P0 should
measure resolved-call precision next to m4's fallback recall.

**m2 — The identity hash will orphan verdicts on ordinary edits, not just renames.**
`symbolId = h(lang, unitId, qname, signature)` (graph note §3.4). Adding a parameter, or moving a
Python file (its qname changes), orphans the verdict. FR-29's "noting the earlier dismissal" is
then lost on exactly the contract changes a reviewer cares about. Q-R3, as put to the stakeholder,
says only "renamed or moved". *Change:* drop `signature` from the id where a language has no
overloading, or re-match orphans by qname at assembly as *possible*. Say "signature change" in
Q-R3's wording (S24).

**m3 — The "no score cut-off" rule and θ are not reconciled.** Graph note Q9 condition 1 forbids a
score cut-off that decides membership. ML Q5 tier 3 (θ) and Q1(b) ("score floor") are exactly
that, and plan §3.6 step 1 adopts both silently. *Change:* state that θ and the floor are a
deliberate, frozen, recall-calibrated exception. They are applied client-side over the complete
full-text result set, never as an index `LIMIT`.

**m4 — The graph model has no node for most of AC-4's callers.** 52 of the 55 tracked files under
`claude/`+`skills/` naming `mcp__cypher__query` are Markdown (verified: `git ls-files … | xargs
grep -l`). But `CONSUMES` sources are `CallSite|ConfigEntry` only. Shell statements have no
enclosing `Symbol` for `HAS_CALL`. FR-20 needs file + line at each step. *Change:* define the
convention, e.g. a file-level synthetic `Symbol` with `CallSite`s for shell and prompt mentions.
`opencode/agents/tank/scripts/*.sh` is a further in-scope shell caller the inventory omits.

**m5 — Stale or inconsistent text in plan v2 (revised in place, not re-read).**
- §3.1 says "Five rules" and lists seven.
- §3.3 keeps the v1 edge list (`LINKS`, `FLOWS`, `BINDS`, `TESTS`/`COVERS`), and its recognizer
  "emit" list still names `BINDS`. Neither exists in the graph note's model. Replace them rather
  than relying on "supersedes where they differ".
- §3.5 calls ~0.115 MB/file "measured", but it is synthetic (graph note V11).
- §2.7's row "(R5, R6 on AC-6/AC-15) | RQ5, RQ6" cross-references unrelated items. The real
  partner of RQ6 is R10.
- R13 does not exist (numbering gap).
- §1 says "~210" Python files; `git ls-files` gives 197 non-deprecated plus 1 in `skills/`.
- §3.12 says `teco`'s freshness recipe is "replaced", yet keeps optional `cpg_*` graphs that still
  need it.

**m6 — The raw-credential details are unverified.** Graph note Q10's ACL adds `+graph.list`, but
V5 tested only `ro_query`/`explain`. Inferred: `GRAPH.LIST` takes no key, so the key pattern will
not filter it, and it reveals every key name on the instance, other systems included. A raw user's
runaway query also shares the instance with the engine. *Change:* verify both in P0(c). Set a read
`TIMEOUT` cap for the raw user.

**m7 — The question set is complete only with additions, and is too big to decide in one sitting.**
- §2.7 dedupes R↔Q-R↔RQ well.
- The ML note's open questions 1–2 (Portuguese questions, workplace statistics) are stakeholder
  questions, but they appear only in plan §6.3, not in D1–D5.
- R14 and D5 are awareness items, not decisions.
- Five questions are missing (S18, S19, S21, S26, S27 below).
- Total: 38 decisions. *Change (`tico`):* batch by theme, lead each with the recommended answer,
  and accept-by-default the wording-only items.

### Nit

- **n1** — FR-17's "test name" starting point appears in neither §3.6 step 1 nor AC-17's phasing.
  A lookup on `Symbol:Test` is trivial, but list it.
- **n2** — Requirements hygiene for `tico`:
  - decision log 2026-10-22 "logged as Open question 1" points at an open question since
    repurposed;
  - "non-component code", listed as an open edge case on 2026-10-22, was never closed. Only
    `.mcp.json` is tracked at the root, so it is close to moot; still, close it.

## 3. What's solid

- **The core shape, now confirmed.** Build-time facts, query-time traversal only, one
  summary-level graph per system, and Joern off the answer path. The graph note's live probes
  (V1–V14) make this the best-grounded part of the chain.
- **G1 checked independently.** In `cypher-mcp/server.py:1024-1037`, an `RO_QUERY` write rejection
  goes to `authorize_write()`, then `target.query()` runs against the caller's graph with no
  graph-name check. Enforcing FR-25 through the ACL credential is the right fix.
- **The determinism boundary.** The ML note's "an LLM may interpret and phrase, never assert", the
  interpretation cache, template-rendered facts, and the deterministic RCA ranker are the right
  calls for a small-local-model stack. Its evidence holds: I verified the cited
  notes/commit/quirks entries and its Wilson figures.
- **The evaluation design.** Dev/held-out splits, first-run reporting, the hit rule over expanded
  groups, and asymmetric reachability soundness. It turns AC-5 from gameable into measurable.
- **Other facts checked.** The proving-repo facts I checked all hold:
  - 82 inline-MATCH `query` sites;
  - 55 files / 3 hook scripts naming `mcp__cypher__query`;
  - `API_PREFIX` at `storefront_api.py:133`;
  - the template literal in `endpoints.ts`;
  - all seven FR-1 frontends and the bundled astgen binaries in `~/joern/joern-cli/frontends`;
  - the destructive-ops guard matching only literal `GRAPH.DELETE`/`FLUSH*`, not `RENAME`/`EXPIRE`;
  - six `-rca` docs in `falkor-chat/docs/reviews/`.
- **Honest deviations.** Each one is stated with its reason: non-Joern extractors, JSONL verdict
  log, no CodeQL.

## 4. Open questions (for the caller)

- None blocks the stakeholder AC review. B1–B6 go back to `architect` (B1 jointly with
  `graph-dba`) as a v3 revision **before P0 is planned**, not before the stakeholder sees §5.
- The CodeQL licence claim (§3.13) matches my understanding of GitHub's CodeQL terms. I did not
  re-fetch them.

## 5. Requirement-level items for the stakeholder

Deduplicated across R1–R18, Q-R1–Q-R6, RQ1–RQ11, D1–D5, the ML note's open questions and this
review (marked **new**). Each item is one decision, followed by the recommended answer. **38
decisions**, grouped for batching.

**A. Same facts for everyone (FR-19, FR-16, FR-25)**
- **S1** — Should "a raw query gives the same facts" mean "re-running the answer's published query
  set and algorithm version gives the same facts"? *Recommend yes.* (R1, Q-R4)
- **S2** — Should "same question, same facts" apply to the question *as the system understood it*,
  shown with every answer? *Recommend yes.* (R2, RQ8)
- **S3** — Now that the RCA ranking can be fully repeatable, should FR-19 require the same order
  too? *Recommend yes.* Keep the allowance only if a model re-ranker is ever adopted. (RQ3)
- **S4** — Should "same snapshot" also mean "same set of recorded verdicts", shown on every answer?
  *Recommend yes.* (R18)
- **S5** — Should raw queries be offered only to people who can read every repository in the
  system? *Recommend yes.* (R3, graph Q10)
- **S6** — Does "no code shown" also hide the *existence and shape* of code in repositories the
  asker cannot read? *Recommend content only.* Either way, raw access stays as in S5 for the first
  release. (Q-R1)
- **S7** — Should "raw queries are read-only" be guaranteed by the database credential itself?
  *Recommend yes.* (Q-R6)

**B. How "never miss" and quality are judged (AC-5, AC-6, AC-7, AC-12, AC-15, new ACs)**
- **S8** — Should AC-5 use a frozen, blind list in which each impact must be listed individually
  with its evidence? It would also require ≥95% of "confirmed" items to be correct and set a
  false-alarm limit; how many false alarms per real impact is acceptable? *Recommend yes, with a
  provisional limit of four to one, ratified after P1.* (R4, RQ1, RQ10)
- **S9** — Should AC-12 add a measured set of ≥40 seeded bugs, with the true cause in the top 5 for
  ≥70% (≥50% from a plain description), beating "shortest path first"? *Recommend yes, numbers
  provisional.* (R12, RQ2)
- **S10** — Should "ranked by likelihood" read "ranked by a stated score with its evidence"?
  *Recommend yes.* True probabilities need hundreds of labelled incidents. (RQ4)
- **S11** — Should AC-6 test planted unprovable links of named kinds, with none shown as
  confirmed? *Recommend yes.* (RQ5)
- **S12** — Should AC-15 add ≥10 real and ≥10 look-alike-safe cases per finding kind, and require
  that a reachable vulnerable dependency is never called unreachable? *Recommend yes.* (RQ6)
- **S13** — Should vulnerable-dependency reachability be reported at package level everywhere, and
  at function level only where the advisory names the vulnerable functions? *Recommend yes.* (R10)
- **S14** — Should AC-7 use the OpenTelemetry demo, with the configuration frozen before first
  contact, the first result reported, and a third held-back codebase later? *Recommend yes.*
  (RQ7, plan §2.6, ML m8)
- **S15** — Should a new AC require ≥92% of ~100 plain-language test questions to be understood
  correctly, with ambiguous ones answered with options rather than guesses? *Recommend yes,
  provisional.* (RQ9)

**C. Local vs hosted (FR-9)**
- **S16** — Should FR-9 say that a weaker local model may need more rephrasing but never changes
  the facts? *Recommend yes.* (RQ11)
- **S17** — Does "no code leaves the machine" mean nothing derived from the repositories leaves,
  including names, error messages and dependency lists? And which endpoints count as "the
  machine" (e.g. LM Studio on the Windows host under WSL)? *Recommend yes, with an explicit
  allow-list of local endpoints.* (R8)
- **S18 (new)** — In local-only mode, may the build download dependency libraries to understand
  the code better? Doing so reveals package and import names. *Recommend no.* Use pre-provisioned
  caches, and accept less precise Java/Kotlin/C#/Go analysis without them. (B4)
- **S19 (new)** — If a system mixes local-only and hosted repositories, should any answer touching
  a local-only repository be handled entirely locally? *Recommend yes.* Local-only should be
  "sticky" per answer. (B4, FR-9 × FR-22)

**D. What counts as code and callers (FR-1, FR-4, FR-5)**
- **S20** — Should FR-1 add shell scripts and structured config (JSON/YAML/TOML/Compose), and
  should an agent prompt that names a tool count as a *possible* caller? *Recommend yes to both.*
  (R5)
- **S21 (new)** — Do example/template configs (e.g. `mcp-monitor/config.example.toml`, the only
  `mcp-monitor` config in the repo, which AC-2 names) and code snippets in documentation count as
  callers? *Recommend yes, as possible with reason "example config"; documentation snippets no.*
  (B3)

**E. Verdicts on possible links (FR-28, FR-29)**
- **S22** — Should confirmations also expire when the code at either end changes, with "change"
  meaning the enclosing function (or config entry)? *Recommend yes.* (R6)
- **S23** — Should a verdict take effect immediately and survive refreshes? *Recommend yes.*
  (Q-R2)
- **S24** — When code is renamed, moved, **or has its signature changed**, may its link come back
  as possible without the "earlier dismissed" note, while the old verdict stays listable?
  *Recommend yes.* (Q-R3, m2)
- **S25** — Should the designated-person list come from a CODEOWNERS-style file plus the OS user
  locally (honest users only), and from the identity provider when hosted? *Recommend yes.* (R7)
- **S26 (new)** — When an agent or a CI job asks a question, whose repository access applies?
  *Recommend:* the person who launched it, or a configured service identity with its own grants.
  (FR-24 × FR-16)
- **S27 (new)** — For a link between two repositories, may an owner of *either* repository dismiss
  it, or must both agree? *Recommend either, with the dismissal shown to both sides.* (FR-28,
  plan §3.8)

**F. Refresh, coverage, scope (FR-7a, FR-14, FR-0/22, FR-17, FR-23, FR-26)**
- **S28** — A part that has never built gets reported as "not mapped", and anything crossing it as
  possible. Accept? *Recommend yes.* (R9)
- **S29** — Should runtime coverage require only totals, using per-test data when available?
  *Recommend yes.* (R11)
- **S30** — Does a list of which repositories form a system (plus exclusions) count as acceptable
  input, i.e. not "hand-describing connections"? *Recommend yes.* (R15)
- **S31** — Should "PR" be an optional add-on, with the core accepting a diff, commit range or
  branch? *Recommend yes.* (R17)
- **S32** — Should "never changes the core" mean "within the shared vocabulary", so that a
  genuinely new concept is a versioned core change? *Recommend yes.* (R16)
- **S33** — Should concurrent change previews be capped (default about 4), oldest idle retired
  first? *Recommend yes.* (Q-R5)

**G. Delivery and process**
- **S34** — Should this be a new top-level component (working name `codegraph/`) rather than
  growing `cpg/`? *Recommend new.* (D2)
- **S35** — Should the plain-language client move up into the first delivery phase? *Recommend no
  unless early people-facing value matters more than an early LLM dependency.* (D3)
- **S36** — Who writes the blind answer keys? *Recommend a read-only agent with no view of the
  system, spot-checked by you.* (D4, ML open question 3)
- **S37** — Will questions also be asked in Portuguese? *Recommend confirming. If yes, the test
  question set includes both languages.* (ML open question 1)
- **S38** — Can you supply counts-only statistics (no code) from one workplace service, to check
  that the "could be any endpoint" fallback stays readable at workplace scale? *Recommend yes.*
  (ML open question 2, K2)

**For awareness, no decision:**
- Change-impact answers inside the `teco` flow may take minutes (R14).
- The workplace deployment wants a dedicated FalkorDB instance of ~10 GB RAM (D5, G8).
- D1 is the meta-item this list replaces.

## Appendix A — AC-1/AC-2 grounding trace (B3)

| Claim in plan §1 | What the code says |
|---|---|
| `@router.get("/health")` in `falkorchat/api.py` is the storefront endpoint | `api.py:56` is inside `build_router()`, mounted with **no prefix** at `app.py:484-490` and only `if dev_surface`; otherwise `app.py:497` re-registers a bare `/health` |
| Prefix comes from module constant `API_PREFIX` | The storefront `/health` is `storefront_api.py:937` inside `build_storefront_router()` (`:857`); the prefix is applied by `app.include_router(..., prefix=storefront_api.API_PREFIX)` at `app.py:521-524` |
| Client: `apiFetch({ path: \`${API_PREFIX}/health\` })` | Correct (`endpoints.ts:8,17-19`), but `apiFetch` (`client.ts:107`) → `buildUrl(opts.path, opts.query)` (`:73,108`) → `fetch(url, {method: opts.method…})` (`:118`): field-sensitive, two-level (B2) |
| AC-2 callers: Kiro JSON, `mcp-monitor` TOML, `.mcp.json` → `docker-run.sh` | Kiro and `mcp-monitor` bind falkor-chat by URL `http://localhost:8000/mcp`; `.mcp.json`'s falkor-chat entry is `falkor-chat-agent-team` → `http://localhost:8200/mcp` (port set by `falkor-chat/scripts/start_agent_team.sh:137`); the `docker-run.sh` chain is the `cypher` entry (AC-4) |

## Pass 2

*2026-10-26.* **Reviewed:**
- `docs/requirements/connected-code-graph.md`, working tree (S1–S38, use-case re-check, SQ1–SQ8,
  Q-R7–Q-R11, V3-Q1–V3-Q5);
- plan v3;
- graph note v2 §6;
- ML note v2 §8.

I re-ran the source spot-checks for v3's new citations.

**Verdict: approve with changes.** v3 resolves every Pass 1 major in design, and the phase table
now closes all 28 ACs. Every S/SQ/Q-R id and AC id is referenced in v3 (scripted grep). Two new
majors remain:
- AC-8 is closed before most of the outbound paths it governs exist;
- four ACs no longer match their FRs, or the plan, after today's edits.

There are 7 new stakeholder questions and 1 wording fix for `tico` (below).

CPG: considered, not relevant — Pass 2 rechecks documents and shell/config line citations outside `cpg_falkorchat`'s scope; Pass 1's CPG checks stand.

### Pass 1 dispositions

- **B1 — fixed.** Graph §6.1–§6.2 adds a per-state traversal table. Its `READS`/`WRITES` rows give
  writer→reader, RCA→writer, and both sides for a contract walk, without W→W2. Plan rule 8, §3.7
  and §5.1 carry the four fixture assertions. Rejecting my mirror-edge remedy is justified
  (graph §6.1).
- **B2 — fixed (design).** `Param`/`Ret`/`Cond` points, `DATA {fromPath,toPath}`, `LIFTED_VIA` and
  the P0(b) vertical slice gate the schema freeze. The §6.3 AC-1 chain matches `client.ts:73,107-118`.
- **B3 — fixed.** Re-checked `storefront_api.py:937`, `app.py:483-497,521-524`, `kiro …json:7`,
  `config.example.toml:34`, `start_agent_team.sh:137,157` and `Dockerfile:42`. One citation slip,
  P2-n1.
- **B4 — fixed in design; residual P2-1.** Broker scope is now every connection, with an offline
  sandbox and fetch flags off (§3.2 table, §3.12).
- **B5 — fixed.**
  - AC-7: per-stratum first contact, ✓ at P3.
  - AC-22: ✓ at P5.
  - `teco` review/QA stages: P3.
  - Fixture system and P1 strata: P0(g).
- **B6 — fixed in design; residual P2-4.** Offset stamp, shared lock, replay before `RENAME`,
  self-healing replay, and a §5.1 test.
- **m1 — fixed; residual P2-5.** Compound `Name` hop with the anti-join. Resolved sites also carry
  `CALLS_NAME`.
- **m2 — fixed, but it now conflicts with decided FR-29/AC-21a** (P2-2c).
- **m3 — fixed** (plan §3.6 step 1, ML §8.1, graph §6.4).
- **m4 — fixed** (synthetic `script`/`document` Symbols, graph §6.3).
- **m5 — fixed.** The rule count is gone; there are no v1 edge names (grep finds no `LINKS`,
  `FLOWS`, `BINDS`, `COVERS` or `ARG_FLOWS`). Python is 198 = 197 + 1, matching my recount. The
  freshness recipe is kept.
- **m6 — fixed.** No `GRAPH.LIST`, `+info` added, `TIMEOUT_MAX`, live-verified V16–V18.
- **m7 — superseded.** `tico` batched S1–S38.
- **n1 — fixed** (§3.6 step 1 test-name anchor).
- **n2 — fixed in the requirements** (2026-10-26 housekeeping entry). v3 §2.5's hygiene note is now
  stale (P2-6).

**Tally:** 14 fixed (4 with a residual below), 1 superseded, 0 not fixed.

### New findings

**P2-1 (major) — AC-8 is marked ✓ at P1, but most of the egress surfaces it governs arrive
later.** *Verified against v3 §4 table rows P1–P5.*
- At P1 only the Python/TS frontends exist, and they need no fetch flags. The frontends B4 was
  about (Java/Kotlin/C#/Go fetch flags) land in P2, along with the hosted-default "controlled
  path" for dependency downloads.
- The PR adapter lands in P3. The IdP/access-provider adapter lands in P4. The git-history scan
  and the separate OSV-refresh job land in P5.
- §5.3's assertion that "every frontend is invoked with its fetch flags off" therefore covers
  none of the four flagged frontends when AC-8 closes.

*Change:* gate AC-8 per part, like AC-5 and AC-13. Re-run the sandboxed full-refresh test at the
exit of every phase that adds an outbound path (P2, P3, P4, P5), and close it at P5. The assertion
that catches a regression is the existing logging-proxy check: zero connections outside the
allow-list, run against the phase's full feature set.

**P2-2 (major) — four ACs no longer match their FR, or the plan, after today's edits.** *Verified
by reading the requirements against v3.* Each makes a correct system fail, or lets a wrong one
pass.
- **(a) AC-8 vs FR-9.**
  - FR-9 now says that in a system containing any local-only repository, *every* model call is
    local, interpretation included. AC-8 only checks "an answer touching it alongside a hosted
    repository". A question that touches only hosted repositories in a mixed system, interpreted
    by a hosted model, passes AC-8 and violates FR-9.
  - AC-8 also tests none of these: no dependency downloads under local-only, the
    two-permitted-fetches rule, or the hosted-default controlled path with no import-name
    service. → Stakeholder question SQ-A.
- **(b) AC-24 vs FR-9.** AC-24 binds "each codebase's configured default model". Under FR-9's
  system-wide rule, a hosted-default codebase in a mixed system is interpreted by the *local*
  model, so the gate binds a model that is never used there. → SQ-B.
- **(c) FR-29/AC-21a vs plan §3.3.** The requirement says a *signature change* returns the link
  without the earlier-verdict note. v3 adopted my m2:
  `symbolId = h(lang, repoId, qname[, signature only for overloading languages])`. So in Python,
  TS, Go and C, a signature change keeps the id, and the link returns *with* the note. That is
  better behaviour, but it fails AC-21a's signature leg as written. "Moved" is also
  language-dependent: a Java class moved within its package keeps its qname. → SQ-C.
- **(d) FR-14 vs AC-14** (wording only; SQ6 already decided this). FR-14 says risk is "how many
  callers reach it, security-relevant, changed recently". AC-14 and ML §8.4 say *entry points*
  and add *complexity*. → `tico` aligns FR-14 to AC-14. No decision needed.

**P2-3 (minor) — the B6 lock is not grantable as specified, and its expiry can reopen B6.**
*Inferred; nothing was run.*
- `SET ccg_<system>:verdict_lock NX PX` is a plain Redis key command. `ccg_verdict` is specified
  as "`GRAPH.QUERY` on the live key only" (plan §3.5), so it cannot take the lock.
- `ccg_build`'s pattern `~ccg_<system>*` matches the key but lists no `set`/`del`.
- If PX expires during replay plus `RENAME`, a verdict can patch the old live key after the replay
  read the tail. The verdict is then lost until the next verdict or swap self-heals it.

*Change:*
- add `+set +del` (or `+eval` for a compare-and-delete release) on that one key to both users;
- size PX well above the worst replay-plus-swap time;
- after `RENAME`, under the lock, compare the live offset with the log length and replay any
  remainder.

Extend §5.1's B6 test with a lock that expires mid-swap.

**P2-4 (minor) — two ACs are closed before the features they exercise exist.**
- AC-18 ("every item in every answer" carries a chain) is ✓ at P1, but review, test-map, RCA and
  security answers arrive at P3/P5.
- AC-9's decision-state stamp is ✓ at P1, but verdicts arrive at P4, so the stamp is trivially
  "empty".

*Change:* re-assert AC-18 at P5 exit and AC-9 at P4 exit, as v3 already does for AC-22.

**P2-5 (minor) — the m1 residual: one measurement leads to two different consequences.**
- Graph §6.1 item 6 says what the *operator* does with name candidates at a resolved site:
  suppress the resolved target and list the others as *possible* `resolution-unverified`, unless
  P0's resolved-call precision clears a threshold.
- v3 §3.2 and §2.2 use the same 0.95 measurement only as an *indexer-adoption* trigger. §3.6 and
  §3.7 never state the operator rule.

*Change:* state both consequences in §3.7, with one threshold.

**P2-6 (minor) — v3 is not updated for the late decisions.**
- V3-Q1–V3-Q5 are still described as open in §2.3 (S19 row), §2.5, §3.7, §3.9, §3.11, §3.12,
  §6.2 and §7.
- §2.2's B4 row cites "V3-Q4/Q5" for git and the vulnerability DB; Q5 is hosted dependencies.
- §2.5's hygiene note is already closed in the requirements.

The *content* matches the final wording of FR-9, FR-13 and FR-15 for all five, with one gap: the
missing-authorization kind (SQ-D). *Change:* rewrite the five references as decided, citing FR-9,
FR-13 and FR-15.

**P2-7 (minor) — `findingId`'s canonical tuple is undefined.** Graph §6.4 says only "hashes", and
plan §3.8 says "hashes of canonical tuples". If the tuple includes intermediate `STEP` hops, an
edit anywhere along a taint chain orphans the finding's verdict. The link would then return
without the note, contrary to FR-15's rule that a verdict expires only when "the code at the
finding" changes. *Change:* define `findingId = h(kind, end tuple per FR-15's list)` and keep the
chain out of the identity.

**P2-n1 (nit) — one citation is off.** v3 §1 cites `falkor-chat/scripts/start_server.sh:228-233`
as the uvicorn launch. Those lines are `echo` banners. The launch is `:251`
(`exec … uvicorn falkorchat.app:app $UVICORN_ARGS`, default `--reload`, no `--port`), so the
"uvicorn default 8000" conclusion holds.

### Stakeholder questions (Pass 2)

Each is one decision, with the recommended answer.
- **SQ-A (AC-8 ↔ FR-9).** Should AC-8 also check four things?
  1. In a mixed system, a question touching only hosted repositories is still interpreted locally.
  2. A local-only build makes no dependency download.
  3. Only the repository's own git remote and the separate vulnerability-database job connect
     out.
  4. A hosted-default build downloads only through the controlled path, with no import-name
     service.

  *Recommend yes.*
- **SQ-B (AC-24 ↔ FR-9).** Should AC-24's 92% bind the model that actually interprets that
  codebase's questions, which is local in any system containing a local-only repository, rather
  than the codebase's configured default? *Recommend yes.*
- **SQ-C (FR-29/AC-21a).** Should "renamed, moved or signature changed → returns without the
  note" apply only where the system cannot prove it is the same code? In languages without
  overloading (Python, TS, Go, C), a signature change would then keep the identity and return the
  link *with* the earlier-verdict note. *Recommend yes.* It keeps more history and matches the
  design.
- **SQ-D (FR-15 × FR-29).** FR-15's expiry list covers injection, risky configuration, secrets and
  dependencies, but not **missing authorization**. For that kind, which code's change expires the
  verdict? *Recommend:* the entry point's function and the sensitive operation's function,
  mirroring injection-type.
- **SQ-E (FR-9 vs FR-24/FR-28/FR-17).** Under local-only, FR-9 permits only two outbound fetches.
  May the company identity provider, the hosted access check and the optional PR add-on also be
  called? Each receives repository names and the asker's identity. *Recommend:* allow each only
  if it is a company-internal service named explicitly on the allow-list, receiving nothing beyond
  repository name and identity. Otherwise local-only uses the local designation file and access
  file, and no PR add-on.
- **SQ-F (FR-21 vs raw queries).** Protecting the shared instance needs an instance-wide time cap
  (`TIMEOUT_MAX`, graph V18), which is a fixed time budget for power users' raw queries. Accept a
  configurable cap for raw queries only, with built-in answers still unbounded (they are run in
  pieces)? *Recommend yes.*
- **SQ-G (FR-15 severity).** A vulnerability advisory with no score still needs one of the four
  levels. *Recommend:* use the advisory's own stated severity if it gives one; otherwise "high",
  with the reason "advisory unscored". Inferred, not checked against the OSV schema: many
  advisories carry no CVSS score.

**Wording fix for `tico`, no decision:** align FR-14's risk factors with AC-14 (entry points,
plus complexity), per SQ6 (P2-2d).

## Pass 3

*2026-10-28.* **Reviewed:** plan v3.1 (re-read in full for the Pass 2 targets); the requirements
working tree (FR-9, FR-14, FR-15, FR-21, FR-29, AC-8, AC-21a and AC-24, plus the decision-log
entries for SQ-A–SQ-G and the 2026-10-28 "moved" entry). Graph note v2 and ML note v2 are
unchanged since Pass 2 and were rechecked only where v3.1 cites them.

**Verdict: approve.** Every Pass 2 finding is fixed, and v3.1 matches the final requirement
wording. One new minor and one nit remain; neither needs a stakeholder decision. The plan is fit to
go forward once requirements reach `Ready for design`.

CPG: considered, not relevant — Pass 3 rechecks document consistency only; no new code-level claim.

### Pass 2 dispositions

- **P2-1 — fixed.** The §4 table gates AC-8 per part: part 1 at P1, re-runs at P2 (fetch-capable
  frontends, controlled path), P3 (PR add-on) and P4 (IdP and access check), closed ✓ at P5. §5.3
  re-runs the same scenario at every phase exit.
- **P2-2 — fixed.**
  - (a) AC-8 and §5.3 now assert every FR-9 rule: system-wide local interpretation, no
    downloads, the permitted fetches, the controlled path.
  - (b) AC-24 and plan §3.12 bind the model actually used.
  - (c) FR-29/AC-21a now keep the note where sameness is provable, matching §3.8's per-language
    rule. The "moved = qualified name changed" reading is in FR-29 (2026-10-28) and is consistent
    with `symbolId = h(lang, repoId, qname[, signature])`.
  - (d) FR-14 now names entry points and complexity, matching AC-14 and §3.7.
- **P2-3 — fixed.** §3.5 grants `+set +del +eval` on the single lock key to both users. §3.8
  covers the tokened lock, compare-and-delete release, a ttl sized from P0(c), and a post-`RENAME`
  remainder replay. §5.1 has a mid-swap-expiry test.
- **P2-4 — fixed.** AC-9 is ✓ at P4 and AC-18 ✓ at P5, with partial gates earlier. §4's ordering
  notes say why.
- **P2-5 — fixed.** §3.7 states both consequences of the single 0.95 threshold. One residual
  appears as P3-1.
- **P2-6 — fixed.** Every V3-Q reference reads as decided (§2.3, §2.5, §3.7, §3.12, §6.2 "none").
  The hygiene note is gone, and the B4 row cites git/OSV correctly.
- **P2-7 — fixed.** `findingId = h(kind, end tuple)` excludes the chain. The per-kind end table
  (§3.8) matches FR-15, including missing authorization (SQ-D).
- **P2-n1 — fixed** (`start_server.sh:251`).

### New findings

**P3-1 (minor) — at precision ≥ 0.95, suppressing a resolved call's other candidates turns every
wrong resolution into a silent miss.** *Inferred from plan §3.7 "Resolved calls".*
- A language that passes the gate may still mis-resolve up to ~5% of its resolved sites. Take a
  site resolved to Y whose true callee is X. With the other candidates suppressed, impact from a
  change to X never lists that caller.
- FR-8 calls that the worse error.
- AC-5's held-out set would catch it only if one of its items happens to sit on such a site.

*Change (owner: `architect`, with `data-scientist` for the bar):* keep the other candidates as
*possible* `resolution-unverified` in every language, collapsed in rendering so they don't add
burden. Use the threshold only to choose between "collapsed" and "listed", and to trigger the
indexer. Alternatively, state the accepted miss rate explicitly and add an AC-5 stratum of
resolved-call sites. Design choice for the owner; I have not modelled the noise cost.

**P3-n1 (nit) — graph note v2 still sizes for four previews.** Its §0 digest ("~30 GB peak with
four previews"), the §6.4 table header and risk (4), and §5 Q-R5's "default about 4" all predate
the stakeholder's choice of a default of 2. The note says §6 wins where it disagrees, so a reader
taking its sizing as authoritative gets ~25–30 GB instead of the plan's ~18 GB. A one-line
correction for `graph-dba`.

**Stakeholder questions:** none.

## Pass 4

*2026-11-01.* **Reviewed:** plan v3.4's P2-deferral edit (working-tree diff against HEAD, 55+/19−): §4 table
and ordering notes, new §4.1, and the cross-references in §2.2 (B5, P2-1), §2.3 (Arduino row), §2.4
(FR-26/AC-22), §4 Arduino row, §5.3, K4, K13 and the §6 digest. I read the requirements ACs the
moves touch (AC-3, 5–8, 12–15, 17, 19, 22, 23), ran unfiltered `P2` / `P3` / `after P` / `before P` /
`generality` sweeps over the plan, and swept the five extending notes and the `-coordination`
ledger for old-order claims.

**Verdict: needs changes.** The edit's own mechanics are clean, and the core argument holds: AC-7
cannot close before P2, and P3–P5 need none of P2's extractors for their own ACs, with the two
exceptions below. But the move exposes two gates that P2 used to precede and no longer does (one
major, one smaller), and the "P2 never runs" outcome is stated for AC-8 only.

CPG: considered, not relevant — document-consistency review against the requirements text; no code-level claim.

### Findings

**P4-1 (major) — AC-13 is gated at P3 and closed at P5 for contract kinds only P2 builds.**
*Verified: AC-13 text (`docs/requirements/connected-code-graph.md`, AC-13), plan §4 P3/P5/P2 rows, §3.4 line ~547, grep of "library_api".*
- AC-13 requires one planted change of *each* kind: endpoint, **message topic**, table/column or
  label, MCP tool, **shared-library signature**. The topic pack (Kafka/RabbitMQ/in-house) and the
  shared-library + version-skew pack are P2 deliverables (§4 P2 row; §3.4).
- The P3 row gates AC-13 "except its security element" and P5 closes it. Neither phase can plant or
  answer the topic and library-signature kinds. Under the old order P2 preceded P3, so this was
  sound; now it is a sequencing defect. §4.1 lists six ACs and omits AC-13. The §4 sentence "Every
  AC is gated in the phase that builds everything it needs (B5)" is false for it.
- *Change (`architect`):* gate AC-13 per kind: endpoint, table/label, MCP tool at P3 (+ security at
  P5); topic and library signature at the P2 exit, added to the P2 ACs column and to §4.1. State in
  §4.1 that AC-13 stays open without P2. Then re-sweep every other AC against the P2-only set (topic,
  rpc, library_api, multi-repo groupings, five non-Python/TS languages) with a table: AC × the
  link-kind or language it plants. I checked AC-3, 12, 14, 15, 17, 19; the sweep should be the
  architect's, over all 28, not mine.

**P4-2 (major) — AC-19 (P3 ✓) and AC-9b (P1 ✓) need a multi-repo manifest the P2 row claims.**
*Verified: P3/P1/P2 rows, §3.2 "System manifest", AC-19 text; the intent of the P2 phrase is inferred.*
- AC-19 needs a system of ≥3 repositories with an un-merged change in one; AC-9b needs one
  repository failing while others update. The P2 row delivers "multi-repo manifest with groupings".
  §3.2 makes the manifest (repositories, exclusions, groupings) the system's only input from P1.
- Ambiguous: either the manifest reader is P1 and only *groupings* are P2 (then say so in both
  rows), or a multi-repo manifest really is P2 and AC-19 at P3 is another reordering defect.
  The P0(g) fixture system has three repos, so the gate could run, but only if P1/P3 own the
  repo-list part.
- *Change (`architect`):* split the phrase: P1 owns "repo list + exclusions + per-codebase policy";
  P2 owns "component groupings" only. Check Q-R10 / `Unit.componentId` (§3.2) for what P3 needs.

**P4-3 (major) — "P2 never runs" is resolved for AC-8 only.**
*Verified: §4.1 and the ordering notes; the requirements doc has no counterpart (grep "P2" finds nothing).*
- If P2 is skipped, AC-6, AC-7, AC-17, AC-22, AC-23 and AC-8 are never closed. FR-1's required
  languages (Java, Kotlin, C#, Go, C/C++) and the FR-27 conventions requirement go undelivered. The
  plan says P2 is "optional", but the requirements call these required. That is a scope decision
  recorded only in a plan and the `-coordination` ledger (the ledger itself still says "(c)
  requirements doc note" is open).
- AC-8 is honest: the requirement says "closed only when all of it is built", and the plan keeps it
  open for the four fetch-capable frontends. The same statement is missing for the rest.
- *Change:* (`architect`) add one sentence to §4.1 stating that with P2 omitted the listed ACs are
  descoped, not met, and which FR/AC they sit under. (`tico`) record the stakeholder decision in the
  requirements decision log, naming the FRs whose delivery is now conditional. Stakeholder to confirm
  that "P2 optional" means those ACs may be formally dropped.

**P4-4 (minor) — §4.1 claims AC-6's P1 kinds are "re-run at P5"; the P5 row has no AC-6.**
*Verified: P5 row ACs list.* Either add "AC-6 (P1 kinds re-run)" to P5 or drop the claim. Also
name the AC-6 kinds for P3's data and config values (a runtime-built query string), since P3 adds
new link kinds.

**P4-5 (minor) — K4 now surfaces last; no early executable check.** *Inferred.*
Rule-pack format v1 freezes at P0 on a *paper* test (P0(d)). P3's and P5's many packs (test, data,
cache, security, risky-config, infrastructure signature) are written against it, and the first
real "someone other than the author" test (AC-23) is after all of them. A format flaw then forces
a migration of every pack. The plan accepts this ("What is lost"), but a cheap hedge exists: run
the in-house-bus pack of the P0(d) paper test as a *fixture* test by a second author at P3 exit,
non-closing, on the fixture system's topic link kind. Owner: `architect`, with the stakeholder
deciding whether to pay for it.

**P4-6 (minor) — AC-22 fixture-unit gate: adequate as a partial gate, with two conditions.**
*Verified: §5.1 "AC-22 is the 'adds one directory, touches nothing else' test", §4 P5 row.*
- Sound: the requirement says "language *or protocol*", so a pack satisfies the diff-shape half, and
  the plan correctly keeps the real-language half at P2. A fixture pack cannot show a test-run
  importer or trace parser, so "all five use cases answer across it" is only partly exercised.
- Conditions: (a) name the planted failing input that makes the check red, e.g. a fixture unit
  whose diff also edits one existing file; it must fail the contract test. Without that the P5
  gate cannot be shown to fail. (b) Inside P2, the unit that witnesses AC-22 must be a language added
  after the P2 core changes (controlled download path, groupings), because the core changes touch
  existing units and would taint the "touches nothing else" diff.

**P4-7 (minor) — P3 and P5 rows do not say "Python/TS only".** *Verified.*
Only the §4 intro (line ~1005) says P3–P5 stack on Python/TS. Rows still read "per-language traces",
"per-test runtime importers", "security packs". Say "Python/TS (+shell/config)" in those rows, and
list JaCoCo and Go-cover importers (§3.3) as P2 work, so P3 is not read as owing them.

### Item 5 — is the per-language use-case scope new?

A consequence, not new scope. Under the old order P3/P5 built importers, trace parsers and security
packs after P2's languages existed. Now they exist only for Python/TS, so the other languages'
share must land with their extractors. Two cautions: it makes P2 materially larger (five languages ×
test importer, trace parser, security packs, risky-config), and it is sized "at P2 planning" with no
estimate. State that it is a consequence in the P2 row, and that P2 may itself split per language
(P4-3's optionality then applies per language).

### Other checks

- **Sweep for old-order references (unfiltered):** `P2` (all hits mapped), `after P2`, `before P3`,
  `P3 exit`, `P3 needs`, `generality`. Remaining old-order text: none in the plan other than the
  items above. Edited correctly: B5, P2-1, Arduino rows (§2.3 and §4), FR-26/AC-22, §5.3, K4, K13.
  The §6 digest still reads "Digest (v3.1):" while carrying v3.4 content; relabel (nit).
  Extending notes: ML note 642 and P0 plan 814/817/1523/1533 use "P2" as a phase name only, and the
  bundle-spec and egress-spec likewise; none states a position, so none needs an edit. The
  `-coordination` ledger line 261 still says "Not yet applied" and "(c) requirements doc note"
  open; `teco`'s to update.
- **P3–P5 deliverables depending on P2 extractors/packs:** none beyond P4-1 and P4-2 (checked P3,
  P4, P5 rows and the §3.6–§3.8 operators; the review operator lists "topic" and "library"
  contract kinds, which is P4-1 restated).
- **AC-6/AC-7 not closing before P2 (§4.1):** AC-7 — agreed. It needs gRPC, messaging and a SQL
  database; first contact is single-shot, so an early partial contact would waste it. The claim that
  the demo's SQL services are not Python/TS is *inferred* and unchecked; the plan itself defers it to
  P0 pre-registration. AC-6 — sound but conservative: the requirement's "e.g." list makes
  "topic read from config" an example, so the plan's reading (the kind must close) is a choice, the
  safe one.
- **AC-8 at P2 exit:** consistent with the requirement ("closed only when all of it is built");
  the controlled download path is the last outbound path added. If P2 never runs the plan keeps it
  open, correctly (P4-3). The §5.3 assertions on the four fetch-capable frontends are vacuous
  until P2; the plan already says so.
- **Growth-stops:** the edit rewrote B5, P2-1, K4, K13, the digest and the ordering notes in place,
  and adds no finding→resolution prose. Growth is new scope (§4.1, ~30 lines). Rework items:
  (a) the v3.4 revision note stacks a fifth line under v3.1–v3.3 instead of replacing the previous
  line, as root `AGENTS.md` requires; the stack is pre-existing, but v3.4 continues it. (b) §4.1's
  heading ("Effect of deferring P2") and its bold question paragraph narrate the change and the
  coordination question; the plan should state the current design ("AC closure with P2 last")
  without the Q&A. (c) The AC-closure fact now sits in the P2 row, the P5 row, §4.1, B5, P2-1 and the
  digest; make §4.1 the one home and have the others cite it (nit).

### Proposed REVIEW.md lines

None.

**Stakeholder questions:** one — does "P2 optional" allow formally descoping AC-6/7/17/22/23 and the
non-Python/TS FR-1 languages (P4-3)?

## Pass 5

*2026-11-01.* **Reviewed:** plan v3.4 as revised against Pass 4 (working tree, 58+/25− against HEAD): §4 rows
and notes, §4.1, §5.1, K4/K13, the digest, the requirements decision log (lines 740–753), and the P0 plan's
fixture section (lines 808–845). I re-ran the unfiltered sweeps (`optional`, `descop`, `P2`, `Digest`).

**Verdict: approve with suggestions.** All seven Pass 4 findings and the growth-stops points are fixed. One new
major remains, from my own re-sweep: AC-19 at P3 needs a third repository the fixture can only build with the
Java extractor. It is a small fix, not a design change.

CPG: considered, not relevant — document-consistency review; no code-level claim.

### Pass 4 dispositions

- **P4-1 — fixed.** AC-13 is split by kind (P3: endpoint, table/label, MCP tool; P5: security element for those;
  P2: topic and library signature, all elements). P3, P5, P2 rows and §4.1 agree, and the P3 review-operator
  text names the P2 dependency.
- **P4-2 — fixed.** P1 owns repo list, exclusions and policy; P2 owns groupings only (P2 row, §4.1 intro). But see P5-1.
- **P4-3 — fixed, consistent.** Sweep for `optional|descop` finds only the Arduino slot and unrelated uses. The
  revision note, ordering note, K13, digest and §4.1 close all say "scheduled last, not descoped". The
  requirements decision log (lines 740–753) carries the answer. The "AC-8 stays open" sentence is gone.
- **P4-4 — fixed.** The "re-run at P5" claim is dropped from §4.1. P3 gains an AC-6 kind (see P5-3).
- **P4-5 — fixed in intent, premise wrong.** See P5-2.
- **P4-6 — fixed.** The planted failing input is in the P5 row and §5.1 (a new unit plus a one-line edit to an
  existing file must fail); the P2 witness-language ordering is stated in §4.1.
- **P4-7 — fixed.** P3 and P5 rows say Python/TS; JaCoCo and Go cover moved to P2.
- **Growth-stops — fixed.** The revision note is now a single v3.4 line, though the v3–v3.3 notes still stack
  above it (pre-existing, outside this edit). §4.1 is retitled and has no Q&A narrative. The digest label is
  "Digest:". The closure fact is in §4.1; the B5 row still restates the AC-7/AC-22 points (nit, acceptable as a cross-reference row).

### My re-sweep of the 28 ACs against what only P2 builds

I grouped them by what they plant on. AC-1–4, 9/9a/9b, 10, 11, 12, 14–16, 18, 20, 21/21a, 24, 25: Python/TS/shell/config
cases in the proving repo or fixtures; no P2 dependency (AC-12's config, schema-mismatch, wiring and
infrastructure causes are not broker or gRPC). **Agree** with the architect's eight (AC-5, 6, 7, 8, 13, 17, 22, 23).
Two additions: **AC-26** (needs P2's C/C++ extractor; already the Arduino slot, but §4.1's claim that "every other
AC closes at its row in §4" is then slightly off: name it), and **AC-19** (P5-1).

### New findings

**P5-1 (major) — AC-19 is gated at P3, but the fixture's third repo is Java.**
*Verified: P0 plan §4.7 repos (lines 827–832), P0 table row "AC-19 multi-repo overlay | P3" (line 823), §4.1 intro.*
- §4.1 says AC-19 "runs on the three-repo fixture system without P2". The fixture is `fx-svc` (Python), `fx-web`
  (TS) and `fx-lib` (Java: shared library, in-house bus, gRPC `.proto`). Without the Java extractor, `fx-lib`
  is never built, so the fixture has two mappable repos, and AC-19 requires at least three. The proving repo is
  one monorepo.
- *Change:* (`architect`, with the P0 plan owner) either add a third Python/TS/config repo to the P0(g)
  fixture for the P3 overlay gate and AC-9b, or gate AC-19 per part (two repos at P3, three at P2 exit) and add
  it to §4.1. Then correct the §4.1 sentence.

**P5-2 (minor) — the K4 hedge's premise is false: there is no Python in-house bus.**
*Verified: P0 plan lines 827–832; P0 table line 814.*
- P3's hedge says the second author writes "the P0(d) in-house-bus pack for the fixture system's Python bus".
  The fixture's in-house bus is in `fx-lib` (Java); the Python/TS pieces are a Kafka producer (`fx-svc`) and
  consumer (`fx-web`). The topic link kind is also tabled "gateable in P2".
- *Change:* retarget the hedge to the Kafka producer/consumer pair across `fx-svc` and `fx-web`, written as
  a pack by a second author, keeping it non-closing; or add a Python bus to the fixture. State that the
  topic link needs the engine's shared-state pattern (§3.4), which P3 already builds for config values.
  Also have K4's mitigation cell cite the P3 hedge.

**P5-3 (minor) — the new P3 AC-6 kind has no planted fixture.**
*Verified: P0 plan table lines 811–817.*
- The kind itself is sound: P3 builds embedded SQL/Cypher parsing, the plan records "runtime-built" labels
  at `querygen.py:488`, and the reason code exists (§3.4/§3.6). But the P0(g) fixture's AC-6 rows list only
  the four P1 kinds and the P2 topic kind. AC-6 needs at least 3 planted instances per kind in different shapes
  (§5.2).
- *Change:* add an "AC-6: query or label built at runtime, P3" row to the P0 fixture table (≥3 shapes).
  Owner: the P0 plan's owner.

### Settled?

Not yet. P5-1 should be fixed before the P1 plan is written, since the fixture is a P0 deliverable; P5-2
and P5-3 are edits of the same kind. None needs a stakeholder decision. After that, no further pass is needed:
a `teco` check that §4.1 and the P0 fixture table agree is enough.

### Proposed REVIEW.md lines

None.
