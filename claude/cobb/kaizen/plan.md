# Kaizen — Improvement Plan: cobb

> Forward-looking backlog for the `cobb` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-09-08

## Active

| ID | Added | Priority | Status | Summary |
|-------|------------|----------|--------|---------|
| K-001 | 2026-05-31 | high | 🔵 | Periodically re-verify the documented standards against live official docs (these ecosystems change fast). |
| K-002 | 2026-05-31 | medium | 🔵 | Add a worked "port an agent across tools" reference example (Claude subagent ↔ OpenCode agent ↔ Kiro steering) — now skill material, candidate for the `agent-maintenance` bundle. |
| K-003 | 2026-05-31 | low | 🔵 | Track additional agentic tools as they mature (e.g. Codex CLI, Cursor, Gemini CLI) where they share the open AGENTS.md / Agent Skills standards. |
| K-005 | 2026-06-07 | high | 🔵 | Automate doc-drift detection: a scheduled routine that re-fetches the canonical docs, diffs vs. stored snapshots, and files a kaizen item on change. |
| K-008 | 2026-06-07 | low | 🔵 | Dog-food the frontmatter cobb teaches: evaluate adding `memory: project` for a persistent cross-session drift/verified-date store (distinct from kaizen). |
| K-014 | 2026-07-25 | medium | 🔵 | Skills outside cobb's two have **no kaizen home**: `skills/` carries no `kaizen/` dirs and `skills/README.md` routes only `agent-maintenance`/`agent-standards` changes to cobb's history — so edits to `cpg-analysis`, `joern-cpg`, etc. land with no per-artifact log. Three skills now cover three different "owner" shapes — cobb's own machinery (`agent-maintenance`/`agent-standards`), `graph-dba`-driven (`cpg-analysis`, `joern-cpg`), and nobody's (`python-web-quirks`) — and **all three landed in an owning agent's history** by by-example precedent. That is converging evidence that "owner-agent's history" already *is* the convention; the work is writing it into `skills/README.md` + `claude/AGENTS.md` where a new skill author would find it, not deciding it. |
| K-009 | 2026-06-20 | medium | 🔵 | Add a CI/script guard that every component `AGENTS.md` has a sibling `CLAUDE.md` = `@AGENTS.md` stub (so Claude Code never silently misses context — it reads `CLAUDE.md`, not `AGENTS.md`). Fold into the K-005 drift job. *(Sibling shipped 2026-07-09: `claude/scripts/audit-team.sh` covers the agent-collection invariants — the `@AGENTS.md`-stub check could join it.)* |
| K-015 | 2026-07-31 | medium | 🔵 | `analyst/kaizen/inbox.md` has a substantial backlog of already-verified, "suggested home: prompt" entries never distilled (stub-package HEAD-vs-working-tree import, review-safe pytest subset, isolatable snapshot side, byte-identity AST hash, line-number-invariance re-gate, exclude_unset nested-model gotcha, scratch-copy-reverse-patch). Run a full §5 pass: verify each still holds, promote the prompt-worthy ones into `analyst.md` (or a knowledge base for the FastAPI/FalkorDB/MCP-version-sensitive ones), log in `analyst/kaizen/history.md`, clear the inbox. |
| K-017 | 2026-08-20 | low | 🔵 | Item 4 of the "Broader team-verbosity reduction" diagnosis (surfaced 2026-08-19; items 1-3 delivered, see `history.md`): prune hedge-stacking once a rule has structural backup (a hook, a routing table) instead of three defensive clauses. No specific instances identified yet — start with a scan across the agent prompts for hedge-stacked clauses backed by real harness enforcement (a `PreToolUse` hook, a routing table) and trim each to one clean statement. |
| K-020 | 2026-09-06 | medium | 🔵 | `cypher-mcp/server.py:881`'s FalkorDB-unreachable message advises `docker start falkordb-dev`, which cannot work — no launch path in this repo leaves a stopped container by that name. Out of cobb's write remit (component code); route to an implementer via `teco`. |
| K-021 | 2026-09-07 | medium | 🔵 | Validate `entryId` **shape** in the `cypher-mcp` producer-write authorizer — a malformed or colliding id is load-bearing for the curator-clear path. Content validation deliberately **not** proposed. |
| K-022 | 2026-09-08 | high | 🔵 | The **wrong-rather-than-absent** defect class now has **eight** instances — three values, **four prose**, and one **executable** (a test oracle that certified the wiring while structurally unable to see a wiring defect). The 2026-09-08 tombstone sequence isolates the mechanism: three certifications written in the same act as the fix they certify, **all three since corrected** — a retraction launders credibility onto whatever sits next to it. Decide at the next certification pass whether this is one rule or three; the candidates are now specific — an `agent-maintenance` §7 lint check (*verify the reason, not just the rule*; *a tombstone certifies nothing*) and a mutation standard (*the mutant worth running is the design that was rejected*). |
| K-024 | 2026-09-08 | medium | 🔵 | Two documents outside cobb's remit still describe the `:CpgBuildInfo` marker as it was before K-023. **The debt changed kind, not size** — do NOT add the five hand-authored keys to any schema table: the stamp never writes them, and after the map form they are not even named in the code. `docs/plans/cpg-agent-adoption-graph.md` §1.1 owes (a) the **eight** properties the stamp writes, (b) full 40-char OIDs rather than "short SHA", (c) the correction that its code block documents the superseded `SET b.X = …` write semantics, and (d) the schema-level fact this arc established — **the marker's property set is closed by construction**, so `:CpgBuildInfo` cannot be extended by any writer other than the stamp (**`architect`**; per the doc convention an executed-against plan takes a successor or a header pointer, not an edit). `docs/manuals/graph-ontology.md` needs the same shape update plus gate finding **P4-3** — its FAQ classifies on the `PROVENANCE` literal alone and never on `MARKER_ORIGIN`, so a hand-authored marker reads as a pipeline stamp to a manual-only reader (**`tico`**). Both route via `teco`; neither is cobb's to write. |
| K-019 | 2026-08-21 | **high — filed upstream** | 🔵 | **Systemic, now confirmed matcher-agnostic too. `PreToolUse` "ask" hooks do not reliably pause execution in this session under Auto Mode, on either `Bash` or `Write`/`Edit`, regardless of hook source or execution context.** Four independent, isolated live tests, 2026-08-21, Claude Code 2.1.238, all under Auto Mode: (1) `graph-dba`'s own frontmatter `Bash` hook, Task-dispatched with `subagent_type` explicitly correct — didn't fire. (2) The identical guard mirrored as a session-wide `.claude/settings.local.json` `Bash` hook, run from `cobb`'s own **main session** — didn't fire. (3) Same test repeated after the user explicitly reloaded hook config via `/hooks` (visibly listed as registered, `[Local] Bash — 1 hook`) — still didn't fire. (4) **`cobb`'s own frontmatter `Write`/`Edit` hook** (`guard-cobb-topic-writes.sh`) — a `Write` to a path plainly outside cobb's allowlist (`docs/_hook_test_k019_scratch.md`) went through with zero interruption; re-fed the exact real payload to the script directly afterward and confirmed it correctly returns `ask` for that path. **Every test used a real, disposable payload (scratch graph or scratch file, immediately cleaned up) and independently pipe-test-confirmed correct hook logic** — ruling out `subagent_type` omission, stale config, hook-not-registered, and matcher-specific quirks as explanations. **Working hypothesis:** Auto Mode's classifier layer silently resolves/overrides a correctly-emitted `ask` decision before a human ever sees it, across both tool matchers tested. **Filed upstream 2026-08-21** via `/feedback` (user-submitted, confirmed "Feedback / bug report submitted") with the 3-test Bash repro; the 4th (Write/Edit) test landed after filing, not yet included in a follow-up report. **Practical consequence, effective immediately: every "harness-enforced" Guardrails claim across every guarded agent in this team — all three destructive-ops guards, all eight doc-write allow-list guards, the one broad-write deny-list guard — is currently unverified, and actively disconfirmed on the two mechanisms tested, under Auto Mode, in every execution context tried.** Not yet tested: the Write/Edit + Task-dispatched-subagent combination specifically (all 4 tests covered 3 of the 4 matcher×context cells) — very likely shares the gap given the pattern, not confirmed. **Next steps:** (1) monitor for an Anthropic response to the filed report; (2) treat this as the standing state of the team's enforcement model — Auto Mode being off is the only known workaround, untested/not decided; (3) fill the last untested cell (Write/Edit, subagent-dispatched) if a clean answer is ever needed before Anthropic responds. |

### K-001 — Re-verify standards against live docs
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** Frontmatter fields, directory paths, and inclusion modes for Claude Code, Kiro, and OpenCode shift between releases. Stale specifics would make Cobb produce broken artifacts.
- **Proposed change:** On a cadence (or whenever a user reports a mismatch), fetch kiro.dev/docs, opencode.ai/docs, code.claude.com/docs, platform.claude.com/docs and reconcile the "Standards you know cold" section. Log diffs in history.md.
- **Notes:** Baseline verified 2026-05-31 at creation. Subagent context-loading + frontmatter re-verified 2026-06-07 against code.claude.com/docs/en/sub-agents. **Kiro + OpenCode re-verified 2026-06-07** (kiro.dev/docs/steering, opencode.ai/docs/agents+rules) during the K-007 skill extraction — caught real OpenCode drift (`mode: all` default, new `disable`/`color`/`top_p`/`steps` fields, granular permission keys, AGENTS.md precedence). All current specifics now live in the `agent-standards` skill with per-file `Verified:` stamps. **2026-06-20:** Claude Code **subagents** re-verified against code.claude.com/docs/en/sub-agents (tool inheritance + withheld-tools list, expanded frontmatter, discovery/scopes, agent teams + background agents) — claude-code.md stamp bumped to 2026-06-20. Claude Code **Skills/Memory/Hooks/MCP/SDK** still on the 2026-05-31 baseline — next refresh target. **Kiro** docs re-read 2026-06-20 (agents/subagents, steering, specs, hooks). NB: the old "does `inclusion: always` steering reach a subagent" dispute is **not** resolved by a doc re-read — docs affirm it, but it's field-disputed and needs a runtime test on the install. Specs/Hooks confirmed *not* reaching subagents. **OpenCode** re-verified 2026-06-20 (agents/permissions/rules — caught the `tools`→`permission` deprecation; subagent nesting documented; flagged the parent-context divergence vs Claude). **All three tools' agent/subagent surfaces now current as of 2026-06-20.** **2026-07-25:** Claude Code **MCP** fully re-verified against `code.claude.com/docs/en/mcp` and rewritten from a 3-line stub into a full section (scopes/approval + untrusted-workspace gate, `.mcp.json` shape, `${VAR}`-only expansion + the `CLAUDE_PROJECT_DIR`-in-the-server's-env trap, tool naming, the four timeout layers, **tool search on by default** + `alwaysLoad` + server `instructions`, **output limits and the persist-to-disk/file-reference behaviour**, stdio non-reconnection, and how MCP meets subagent `tools:` allowlists); **OpenCode** MCP + Skills sections added/re-verified against `opencode.ai/docs/{mcp-servers,skills}`. Remaining stale: Claude Code Skills/Memory/Hooks/SDK (2026-05-31); Kiro MCP not re-checked since 2026-06-20.

### K-002 — Worked cross-tool porting example
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** Porting between tools is a common request; a canonical example would make answers faster and more consistent.
- **Proposed change:** Add a reference walkthrough mapping a Claude Code subagent's frontmatter/body to an OpenCode markdown agent and to Kiro steering, noting what each tool drops or renames.
- **Notes:** Could live as a skill rather than bloat the agent prompt.

### K-003 — Broaden tool coverage
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** The open AGENTS.md and Agent Skills standards are adopted by more tools than the core three.
- **Proposed change:** Add concise coverage of Codex CLI, Cursor, Gemini CLI, Copilot where they intersect the open standards, clearly flagged as secondary.
- **Notes:** Keep the big-three depth primary; don't dilute.

### K-005 — Automate doc-drift detection
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** K-001 (manual re-verify) relies on someone remembering. A frozen prompt silently rots between checks. Determinism beats hope: a harness-run job that diffs the official docs and surfaces changes is the real safeguard.
- **Proposed change:** A scheduled agent (Claude Code `/schedule` cron routine, or local cron) that, per tool, fetches the canonical pages (code.claude.com/docs, opencode.ai/docs, kiro.dev/docs, platform.claude.com/docs), diffs the relevant sections against a stored `sources/` snapshot (last-verified date + section excerpt/hash), and on change appends an item to this plan + pings the user. Keeps perishable specifics out of the always-on prompt and re-checked on a cadence.
- **Notes:** Surfaced 2026-06-07 — user asked "how do we ensure the info won't drift?" Pairs with the new "Drift-resistance" principle (timestamp + verify volatile facts). Offered to build it; awaiting go-ahead.

### K-008 — Dog-food the frontmatter cobb teaches
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** Cobb runs on `name`/`description`/`model` only, yet teaches a rich field set (`memory`, `disallowedTools`, `permissionMode`, `skills`, `isolation`, `effort`). A `memory: project` store (auto-injected `MEMORY.md`) would give cobb persistent cross-session knowledge of drift findings / verified-dates / gotchas — distinct from kaizen (human change-log, not auto-injected into the prompt).
- **Proposed change:** Evaluate adding `memory: project`. Leave the `agent-maintenance` skill on-demand (do NOT pin via `skills:` — pinning defeats leanness; the on-demand choice is deliberate).
- **Notes:** Surfaced 2026-06-07 self-review.

### K-020 — Correct the cypher MCP server's FalkorDB-unreachable advice
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** `cypher-mcp/server.py:878-882` tells the agent that hits a connection failure to
  `Start it (falkor-chat/scripts/start_falkordb.sh, or docker start falkordb-dev) and retry.` The
  second half is unworkable. `falkor-chat/scripts/start_falkordb.sh:48-63` passes `--rm` in both
  its detached and foreground branches (live-confirmed: `docker inspect falkordb-dev --format
  '{{.HostConfig.AutoRemove}}'` → `true`), so the container is removed on exit and there is
  nothing to `docker start`. The only other launch path, `falkor-chat/compose.yaml`, declares no
  `container_name:`, so its engine is `falkor-chat-falkordb-1` — also never `falkordb-dev`. An
  agent that follows the message verbatim gets `Error: No such container` and may conclude the
  environment is broken rather than simply re-running the script.
- **Proposed change:** drop the `, or docker start falkordb-dev` clause from the error string,
  leaving `Start it (falkor-chat/scripts/start_falkordb.sh -d) and retry.` One line, one file.
- **Notes:** Surfaced by kaizen entry `c1f9a4d2-7b3e-4a86-9c05-2f8d61e0b774` (distilled 2026-09-06,
  see `history.md`). **Not cobb's to fix** — `cypher-mcp/server.py` is component code, outside
  the topic remit; dispatch an implementer through `teco`. Two adjacent sites deliberately left
  alone: `docs/plans/cpg-query-access.md:626` specifies the same wording but is `Status: archived`
  (archived documents are not amended), and `cypher-mcp/README.md` already points at the script,
  never at `docker start`.

### K-021 — the producer-write authorizer validates Cypher shape but not `entryId` shape
- **Status:** 🔵 proposed · **Priority:** medium
- **Trigger:** the 2026-09-07 chunk-B distillation of `teco` (U17) found
  `9f2b6c07-3e51-4a88-b174-c6d90e melhor` in `kaizen_team` — `fact`, `evidence` and `context` all
  the literal string `PLACEHOLDER`, and an `entryId` that is a truncated uuid4 with the Portuguese
  word `melhor` appended after a **literal space** (37 chars). It had sat there since 2026-09-02.
  The `cypher-mcp` write authorizer validates the *Cypher shape* of a producer-write (the
  `MERGE (:Agent)` + `CREATE (…)-[:PRODUCED]->(…:KaizenEntry)` form, with a matching `agentId`) and
  nothing about the values written.
- **The call (cobb's, made at U17): guard the id, not the content.** Two different risks, only one
  worth machinery.
  - **`entryId` — yes.** Every curator operation in `agent-maintenance` §5 is keyed on `entryId`:
    the count-and-decide read, the edge-resolve, and the full-node `DETACH DELETE`. A malformed id
    is survivable (this one was), but a **colliding** id is not — a clear would silently take the
    wrong node, and the failure raises no error. This graph already demonstrates the hazard: its
    ids are hand-shaped rather than `uuid4()` output, and U17's own twelve-entry scope contained
    `b7e41c92-3f8a-…` alongside a chunk-C entry `b7e41c92-3d5a-…` sharing the full first **eight**
    characters. Seven units of the 2026-09 pass hit false-positive dedup matches on 8-char
    prefixes. **U22 (2026-09-08) then found a third collision that ends the "eight characters is
    probably enough" reading outright:** `b7f3a1c2-5d84-4e19-9a6f-2c8e71d40b93` (`analyst`,
    2026-09-03, mutation-kill counts) and `b7f3a1c2-5d84-4e19-9a06-3c2e8f14d7b0` (`architect`,
    2026-09-07, grep-based done-conditions) share their **first 21 characters** —
    `b7f3a1c2-5d84-4e19-9a` — diverging only at index 21 (`6f` vs `06`). Both were live in the
    graph at the same time, produced by different agents on different dates about different
    subjects, and U22 had to clear one while leaving the other intact. No prefix length short of
    the whole id is a key in this dataset, which is the argument for shaping the id at write time
    rather than lengthening the prefix at read time. A
    `^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$` check at
    the authorizer is a few lines and closes it at the one chokepoint every agent passes through.
    Note what it does **not** close: both of these ids *pass* that regex. Shape validation stops
    malformed ids and raises the cost of an accidental near-duplicate; only real `uuid4()` entropy
    at the producer makes a 21-character collision implausible, so the check is a floor, not a
    fix. **U23 (2026-09-08) supplies the density data point behind all three collisions:** the
    thirteen `analyst` entries dated 2026-09-07 contain **eight** ids clustering into three
    near-prefix families — `b1f0e6c2`/`b1f0c7a4`, `b7f1c3a2`/`b3f1c7a2`/`b1e7c2a4`/`b18d5c47`,
    `3f6c2a91`/`3f6c1e28` — sharing 2 to 4 leading characters, produced by one agent on one day.
    Hand-shaped ids are not uniformly distributed over the hex space; they cluster on whatever the
    writer had recently typed, which is why collisions keep landing in *adjacent chunks of the same
    agent's inbox* rather than at random. That is an argument about the generator, and only real
    `uuid4()` entropy addresses it.
  - **Content — no.** Rejecting `PLACEHOLDER`/empty `fact` automates a judgment the curator
    already performs, and its whole cost is one discard line in a history entry. A sentinel
    blocklist would also be trivially routed around by any agent writing a differently-useless
    string, so it buys the appearance of validation rather than validation.
- **Where it lives:** the write authorizer in `cypher-mcp/server.py`, alongside the existing
  producer-write shape check — **not** a prompt rule. Every agent's Learning-capture section already
  says `entryId: '<uuid4>'`; the corrupt node is the evidence that prompt-level instruction is not
  enforcement, which is exactly the case for a harness-side check.
- **Out of cobb's write remit** (component code, same shape as K-020) — route to an implementer via
  `teco`. Filed, not implemented; U17 was scoped to distillation only.

### K-022 — The "wrong rather than absent" defect class may have outgrown per-agent guardrails
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** The same defect shape has now surfaced **eight** times across different agents and
  artifact kinds. **Three are values:** a CPG marker naming a commit whose tree was never parsed; a
  stamp that could fail and still announce success; `git rev-parse` echoing its argument back on
  stdout so a marker took the literal string `HEAD:./src` as a tree OID. **Four are prose:**
  `architect`'s `kaizen_team` entry `7f3c1a92` asserting a CPython deadlock mechanism that does not
  exist; a `cobb` sentence in `freshness.md` (`81b43cd`) giving a false mechanism for a correct
  rule; and — 2026-09-08, gate Pass 6 — two more in the *same passage as that repair*: *"there is no
  list at any layer"* (false one clause wide: `CPG_STAMPED_KEYS` is a list, at the assertion layer)
  and *"both earlier ones covered the primitive and not the call path"* (untrue of mechanism one,
  whose credential was a re-reading one). **The eighth is executable**, which is what makes the
  class bigger than prose: `test-stamp-wiring.sh`'s first oracle certified the *wiring* while being
  structurally unable to see a wiring defect — deleting `replay_stamp` left all six cases green at
  rc 127. The invariant across all eight: **a slot that should have been empty or loud instead held
  a plausible, checkable-looking value that was wrong**, and every downstream reader accepted it
  because it had the shape of a verified one. A green suite is that slot too.
  <br>**What the 2026-09-08 instances isolate, and it is sharper than the earlier diagnosis.** The
  three `freshness.md` tombstones are a controlled experiment: each was written in the same sitting
  as the fix it certifies, by whoever had just made it, and **each has since had its own certifying
  sentence corrected on review** — a 3-for-3 failure rate on one document. The earlier reading was
  *a rule's stated justification is the least-verified prose in a document, because agreement with
  the conclusion suppresses scrutiny of the premise*. That still holds, and this adds the stronger
  half: **a retraction launders credibility onto whatever sits next to it.** The retraction half is
  trustworthy — it reports a failure that already happened, against evidence. The certification half
  attached to it is a fresh, unreviewed claim, and it is read at the confidence of its neighbour.
  "The first two were wrong and here is why this one is different" is the highest-risk sentence
  shape found so far, and nobody re-opens the script behind a rule they already think is right.
- **Proposed change:** at the next certification pass (§4), decide whether this is one rule or
  several. The value cases and the prose cases do **not** share a fix — the value fix is "make the
  failure loud", and no amount of loudness helps prose, which has no schema to validate against,
  only a reader — so they are at least two rules, and the executable instance argues for a third.
  Each half now has a concrete candidate home:
  - **Prose** — a named check in the `agent-maintenance` §7 lint: *verify the reason, not just the
    rule*, with two corollaries now earned rather than proposed. (1) The repair for a correct rule
    with a false mechanism is a **tombstone on the mechanism, not deletion of the rule** (applied at
    v1.27). (2) **A tombstone certifies nothing.** A claim about the current mechanism gets no
    credit from the retraction it is attached to; state the mechanism and the level its evidence
    covers, and stop. Do not write a further tombstone certifying the last one — record a revision
    as one dated line. Both corollaries are now written into `freshness.md`'s third tombstone as a
    worked instance the lint can point at.
  - **Executable** — the mutation standard this arc converged on, which is stricter than "the test
    fails when the code is deleted": **the mutant worth running is the design that was rejected.**
    Deleting a mechanism only proves the test reaches the code; substituting the alternative the
    plan turned down proves the decision was load-bearing. Candidate home is the same §7 lint or
    `analyst`'s `review-techniques.md`; it is a review technique, not an agent rule.
  - **Values** — unchanged, in ascending cost: a shared on-demand knowledge base; one bullet in each
    affected agent's Guardrails (`architect` already has its half as of 2026-09-08 — *"a mechanism
    claim is only as verified as its least-verified clause"*); or a root-`AGENTS.md`-level
    statement, still probably wrong for a class this abstract.
- **Notes:** four of the eight instances have now been in evidence in a `cobb` run (the fourth via
  the retraction pass, the fifth self-inflicted and self-found, the sixth and seventh self-inflicted
  and caught by the gate), which retires the earlier reason for not acting — that writing a
  team-wide rule from a single instance is the same error the rule would be about. Three of the four
  `cobb` instances were **self-inflicted inside a repair**, which is the argument for a lint check
  rather than a Guardrails bullet: the author was already being careful. The remaining instances
  live in `analyst`'s and `graph-dba`'s files. Full trail: `docs/reviews/cpg-provenance-stamp.md`
  (Passes 5-6), `docs/reviews/salesperson-ui-impl.md` Passes 17-18, and `kaizen_team` entry
  `3b351bb5-a8a1-4006-8aa6-bdb3ef1c6448`.

### K-024 — Two out-of-remit documents still describe the pre-K-023 marker
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** K-023 landed in three mechanisms over one day (see `history.md`, U45→U47), ending
  at a **map assignment**: `SET b = {…the eight pipeline fields…}` replaces the marker's whole
  property set, so everything the stamp did not write is gone after any rebuild. Two documents cobb
  does not own still describe an earlier shape.
  **Read this before executing it — the obvious edit is the wrong one.** An earlier version of this
  ticket said §1.1 "never listed the five hand-authored keys and now understates what a stamp
  writes", which would send an implementer to add `MARKER_ORIGIN`, `MARKER_WRITTEN_AT`, `NOTE`,
  `STATUS` and `RENAMED_FROM` to a schema table. **The stamp has never written those five.** They
  were `= NULL` clearing assignments for exactly one commit and are now deleted; a stamp writes
  **eight** properties and no others. Adding them to the schema of the document this arc cited as
  the ownership argument for *erasing* them would be a net loss.
  What §1.1 actually owes is (a) those eight, (b) full 40-char OIDs rather than "short SHA", (c) the
  fact that its code block shows the superseded `SET b.X = …` concatenation, and its "omitted, not
  set to null/empty-string" rationale now describes an omission achieved by a `NULL` entry inside a
  replacing map, and (d) the one genuinely new schema-level fact: **the property set is closed by
  construction**, so `:CpgBuildInfo` cannot be extended by any writer other than the stamp — a
  hand-authored key survives only until the next rebuild, silently.
  `docs/manuals/graph-ontology.md` carries the same field list for readers — plus, per the gate, a
  `§1` cell telling a reader a marker "can carry more than eight" and never saying a rebuild takes
  them — and separately
  the gate's **P4-3**: its FAQ decision list keys entirely on the `PROVENANCE` literal and never on
  `MARKER_ORIGIN`, so a hand-authored marker classifies as a pipeline stamp for a manual-only
  reader — the exact case `freshness.md`'s bullet 1 was re-gated to catch.
- **Proposed change:** route both via `teco` — the plan doc to `architect` (as a header pointer or
  successor, since it has been executed against and the convention forbids an in-place amendment),
  the manual to `tico` (field list, corrected to eight and to "closed by construction — a rebuild
  silently erases anything else, including a human's note"; + one sentence at the head of the FAQ
  absent-cases list:
  *whatever `PROVENANCE` says, a marker with a `MARKER_ORIGIN` property was written or repaired by
  a human — read the whole node before trusting any other field*).
- **Notes:** flagged rather than written because `docs/plans/` is `architect`'s and `docs/manuals/`
  is `tico`'s under the by-kind owner table; cobb's write remit is agent/skill/MCP artifacts. The
  pairwise-consistency constraint that has held through this chain applies: `freshness.md` and the
  manual must end up describing the same set of marker shapes.

## Parking lot / ideas
- **`bypassPermissions` revert landed (U3, 2026-09-01)** — `.claude/settings.json`'s
  `defaultMode: "bypassPermissions"` pin removed (left unset, resolving to the global `auto`
  default), the three `allow`/`ask` rules kept unchanged, the KB entry actually written into
  `skills/agent-standards/claude-code.md`'s `## Hooks` section (the header stamp had been
  forward-referencing it since the Gen 4 design session, but the entry body didn't exist until
  now), and the strengthened upstream-feedback text drafted (not submitted) to
  `claude/cobb/kaizen/upstream-feedback-draft-bypass-permissions-subagent-gap.md`. **U4
  (`analyst`'s implementation review) is still queued** per the coordination ledger — that's
  the remaining open step in this family, not anything of cobb's own. Left as an untracked
  reminder rather than a formal K-item since U4 belongs to `analyst`/`teco`, not to cobb's own
  backlog.
- **Extend the independent-review-gate practice to `cobb.md` self-edits specifically (surfaced
  2026-08-20, `Q2`'s D-1 finding + its own fix).** `Q2`'s closing acceptance pass
  (`docs/test-reports/generic-cypher-mcp2-report.md`) found `cobb.md` had shipped a self-edit
  (the M7 `C-cobb` "Learning capture" retarget) that only touched one of two affected sections,
  leaving lines 65/71 stale and self-contradictory — undetected because no independent reviewer
  ever read that diff (the self-edit carve-out, §3.7 of `docs/plans/generic-cypher-mcp2.md`, makes
  `cobb` both author and sole editor of that one file). The fix for D-1 (this same 2026-08-20,
  `history.md` above) is **itself** another unreviewed `cobb.md` self-edit — the exact shape most
  likely to repeat the miss. Cheap mitigation the report proposes: whenever a self-edit unit
  closes, route a one-line "did every section I was supposed to touch actually change?" grep-diff
  check to a second agent. Not actionable unilaterally (no reviewer in a direct, non-`teco`
  dispatch) — raise with `teco` next time a `cobb.md` touch is coordinated, or self-apply the
  grep-diff check as a matter of discipline even without a formal second reviewer.
- **`kaizen/inbox.md` headers are enforced-frozen in practice, not just by written convention
  (surfaced 2026-08-20, M7 `C-<agent>` header-retarget attempt).** `docs/plans/generic-cypher-mcp2.md`
  §4.2/P3-M3 reasoned that a header note's *prescriptive* clause (the copy-pasteable
  `mcp__cypher__query(graph='kaizen_<agent>', ...)` pointer) was safely editable because each
  header's own immutability promise is scoped to "Content below," not the header itself — a
  textually sound argument, gated through 3 plan-review passes. Live execution disagreed: the
  permission system denied 3 of 4 attempted edits outright ("this is frozen"), and the stakeholder,
  relayed through `teco`, then directed dropping the header-retarget half entirely and reverting the
  one edit that had already landed (`teco`'s). **Don't plan future work that treats the "Content
  below" scoping argument as actionable without re-confirming live first** — a textual carve-out in
  a doc is not the same thing as a carve-out the actual permission gate (or the stakeholder) will
  honor at execution time. If a future delivery genuinely needs a frozen `inbox.md` header touched,
  raise it as its own small, explicitly-flagged ask rather than folding it into a larger unit's
  done-condition.
- **Redirected from `teco`'s 2026-08-12 inbox entry (distilled 2026-08-15):** a directly-invoked
  (non-`teco`-coordinated) large sweep I ran — the 39-file full-team kaizen-inbox distillation,
  gated "needs changes" by `analyst` — had no coordination ledger, and the session that ran it hit
  a mid-run credit exhaustion before the fix pass was dispatched. Recovery only worked because the
  review (`docs/reviews/kaizen-inbox-distillation2.md`) was self-sufficient: explicit baseline
  commit + explicit file scope, `analyst`'s standing review-header practice. **Not filed as a
  prompt change** — one data point, no repeat, and the safety net that saved it is already
  standard practice, not a gap. Parking here as a reminder: if I (`cobb`) ever run another large,
  review-gated, directly-invoked (not `teco`-routed) sweep, keep that same discipline (explicit
  baseline + scope in whatever review/report anchors the work) rather than assuming it'll survive
  on vibes — and if this pattern repeats, it graduates from parking lot to an actual guardrail.
- From the 2026-08-09 self-review during the C-308/C-312/C-319 skill-review pass
  (`docs/reviews/cpg-followups-skills-impl.md`): the C-319 promotion compressed two independently
  true, parallel "cwd-independent" facts (`.mcp.json` discovery walk-up; `${CLAUDE_PROJECT_DIR}`
  expansion) into one causal clause ("stays uniform via ...") that the source evidence never
  established. General lesson for future inbox distillations (§5): when promoting a fact that
  echoes or sits next to another fact in the doc, keep them stated as separate claims unless the
  *mechanism link* between them was itself verified — don't add connective "via"/"because" prose
  as free editorial polish. Consider adding a line to §5's procedure about this specific failure
  mode (causal-compression during promotion) if it recurs.
- From the 2026-08-09 independent review of `analyst`'s inbox-distillation pass
  (`docs/reviews/analyst-inbox-distillation.md`): consider a sub-list format for `analyst.md`'s
  "Evidence over vibes" Guardrails bullet (now 5 sub-rules in one run-on sentence after four clause
  extensions) to restore scannability without adding to the bullet count. Also: `claude-code.md`'s
  top-of-file `Verified:` stamp block could gain a one-line pointer to the new "Bash tool
  environment" section (observed-not-doc-sourced, so it doesn't fit the existing dated-doc-citation
  pattern, but a reader skimming only the header wouldn't know the section exists).

- **All 13 agents' `permissionMode: acceptEdits` frontmatter is confirmed dead configuration for
  controlling their own top-level starting mode (surfaced 2026-08-24,
  `claude/docs/plans/permission-default-mode.md`).** The documented session-start decision order has
  no frontmatter step; `~/.claude/settings.json`'s explicit `"defaultMode": "auto"` is what actually
  decides it, every time. Decide: remove the misleading line team-wide (13-file edit, its own
  blast-radius question — a stakeholder call, not unilateral), or leave it as declared intent for a
  future dispatch-time inheritance case (parent in `default`/`plan`/`dontAsk`, currently never
  arising since every parent starts in `auto`). Not urgent — the field being inert doesn't currently
  cause a behavior regression, just misleading configuration.
- **The graph-shaped Learning-capture template is the floor now (noted 2026-08-20).** The
  2026-08-19 verbosity pass had cut the near-identical "Learning capture" section (~1,500 w across
  13 files) by pointing each agent's paragraph at its own `kaizen/inbox.md` header. That option no
  longer exists: capture is graph-shaped, and the inline `mcp__cypher__query` template is verbosier
  than either prior state — a deliberate trade of prompt leanness for mechanism consistency.
  Recorded so the next verbosity pass over this section doesn't rediscover the file-pointer fix and
  find it unavailable. The one unexecuted slice of that pass is **K-017** (Active table).
- From the 2026-08-12 corrective pass fixing `analyst`'s gate on the 2026-08-11 distillation
  (`docs/reviews/kaizen-distillation-2026-08.md`): a raw inbox entry can leak the maintainer's
  home path/username into a tracked file the moment an agent *appends* it — before any
  distillation ever runs, since `kaizen/inbox.md` is itself a tracked file and an entry's
  `**Evidence:**` line often quotes a live shell command's literal output (`ls -la ~/.claude/
  agents` prints the real symlink target). Caught here only because `audit-team.sh` check 7 was
  re-run incidentally, not because anything in the distillation-review workflow prompts for it.
  Worth deciding whether check 7 (or a lighter version of it) should run as part of *any* agent's
  closing protocol when it appends an inbox entry that quotes command output, not just at
  distillation time — candidate for `agent-maintenance` §5 or the "Learning capture" boilerplate
  itself.
- **A `fork` subagent can drift into narrating the parent's own already-completed work instead of
  its assigned directive (observed 2026-08-21, team certification's §7 lint fold-in).** One of
  three forks, each given an explicit, narrow directive (§7 lint on 4 named files), came back
  reporting a status summary of *my own* preceding work instead of any finding about its assigned
  files — plausible mechanism: a fork inherits the full parent transcript, and this one launched
  right after a transcript stretch dense with the parent's own narrated fixes, which may have
  pulled its generation toward continuing that narration. Handled by treating the result as
  unverified and doing the bounded piece directly rather than re-forking blind. One data point —
  not promoted to a `kaizen_team` entry yet (see `claude/cobb/kaizen/history.md`, 2026-08-21
  certificate entry, for the full reasoning). If this recurs, it's worth: (a) a `kaizen_team`
  entry, and (b) a prompting mitigation — e.g. explicitly telling a fork mid-directive "ignore
  what the parent session already did; your only output is findings on these N files."
- Maintain a small catalog of agents/skills Cobb has authored, cross-linking their kaizen files.
- The §7 prompt-lint is judgment-only by design; if a *deterministic* pre-check for a single artifact ever proves cheap (frontmatter valid, description non-empty, no personal identifiers), consider a small script assist — but keep the seven semantic dimensions in the skill, not a grep. *(Noted 2026-07-16 during the §7 build; the composition load-set enumerator the design floated was skipped as not-cheap-enough.)*
- **Root `AGENTS.md` is 2,729w, over the ~2,500w budget its own new Context-file convention sets
  (2026-09-02).** The cause is not the new rule (257w) but the **Module documentation
  convention** bullet: **1,539w, 56% of the whole file**. Compacting it is a real judgment call
  — it encodes the doc-family grammar, the closed `Status:` set and the by-kind owner table that
  several agents depend on — so it needs its own scoped pass with the user, likely splitting the
  reference tables (owner routing, status tokens, filename grammar) into a progressively-loaded
  doc that the context file cites, keeping only the rules an agent must hold resident.
- **Enforcement for the context-file convention was explicitly declined (2026-09-02).** User
  chose rule-only over the offered `claude/scripts/audit-context-files.sh` + `PostToolUse` hook.
  The rule ships with a one-line `awk` bar instead. This is the class of rule that already
  failed seven consecutive times unenforced — if the bar is breached again, re-offer the script
  (cheap: line-length + word budget + append-tell patterns over `git ls-files '*AGENTS.md'`).
- **A duplicated narrative drifts in the copy nobody re-reads (2026-09-02).** `falkor-chat/
  AGENTS.md`'s salesperson row and `proof_defs.py:170-250` held the same version history; the
  *code* copy is the authoritative one and was the one that went stale (`v1→v4`, "FOUR sibling
  capabilities", unaware of v5/v6/v7). The context-file sweep fixed the AGENTS.md side by
  deleting it; **the `proof_defs.py` comment block is still stale and is `coder`/`teco`
  territory, not mine** — flag it rather than edit it.
