# Kaizen — Improvement Plan: cobb

> Forward-looking backlog for the `cobb` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-09-10

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
| K-022 | 2026-09-08 | high | 🔵 | The **wrong-rather-than-absent** defect class now has **eight** instances — three values, **four prose**, and one **executable** (a test oracle that certified the wiring while structurally unable to see a wiring defect). The 2026-09-08 tombstone sequence isolates the mechanism: three certifications written in the same act as the fix they certify, **all three since corrected** — a retraction launders credibility onto whatever sits next to it. Two candidates were named for where the rule should live. **The first is now settled and shipped elsewhere:** U38 (2026-09-09) promoted *verify the reason, not just the rule* into `claude/analyst/review-techniques.md` as a **review-methodology** section, not an `agent-maintenance` §7 lint check — §7's scope is one *prompt/skill/steering artifact* over seven prompt-authoring dimensions, and the instance that produced the rule is a `references/` document read by a reviewer. **The mutation standard is now settled too:** U43 (2026-09-10) promoted *the mutant to specify is the design the plan rejected* into `teco.md`'s briefing bullet — the rule already existed in `tdd-engineer.md:41` and reached `coder` (which carries no mutation rule at all) through nothing but the brief. **What remains is only the still-open question of whether *a tombstone certifies nothing* is a third rule or a restatement of the first. Decide at the next certification pass. |
| K-024 | 2026-09-08 | medium | 🔵 | Two documents outside cobb's remit still describe the `:CpgBuildInfo` marker as it was before K-023. **The debt changed kind, not size** — do NOT add the five hand-authored keys to any schema table: the stamp never writes them, and after the map form they are not even named in the code. `docs/plans/cpg-agent-adoption-graph.md` §1.1 owes (a) the **eight** properties the stamp writes, (b) full 40-char OIDs rather than "short SHA", (c) the correction that its code block documents the superseded `SET b.X = …` write semantics, and (d) the schema-level fact this arc established — **the marker's property set is closed by construction**, so `:CpgBuildInfo` cannot be extended by any writer other than the stamp (**`architect`**; per the doc convention an executed-against plan takes a successor or a header pointer, not an edit). `docs/manuals/graph-ontology.md` needs the same shape update plus gate finding **P4-3** — its FAQ classifies on the `PROVENANCE` literal alone and never on `MARKER_ORIGIN`, so a hand-authored marker reads as a pipeline stamp to a manual-only reader (**`tico`**). Both route via `teco`; neither is cobb's to write. |
| K-025 | 2026-09-09 | high | 🔵 | A `MENTIONS` tag writes an obligation into a queue with no consumer: a distillation unit is scoped `MATCH (a:Agent {agentId})-[:PRODUCED]->(e)`, which by construction cannot see a `MENTIONS`-only node. 11 such nodes accumulated across 7 agents and 2 passes; **six** of those agents were recorded "closed out, 0/0" while still holding an unrouted edge. Two candidate fixes — scope a unit by *either* edge kind, or stop letting the tag imply a promotion. |
| K-026 | 2026-09-09 | medium | 🔵 | `python-web-quirks` carries **two** `TestClient` **behaviour** sections (teardown task-cancellation; the default `raise_server_exceptions=True` re-raising into the caller), deliberately unmerged because the mechanisms, consequences and fixes differ. They are bound by a shared opening line naming the class. **On the third such entry, promote that line to a parent heading with sub-sections — do not add a third flat sibling.** The count is **still two**: U38 (2026-09-09) added a third section touching `TestClient` — the `httpx`→`httpx2` transport deprecation, which warns on *both* pinned starlette versions — and it is a **version-rot rider on the pair**, carrying no `TestClient` mechanism of its own, so the trigger did not fire. It was **retitled** to lead with the migration rather than the class precisely so a header scan cannot read it as a third sibling and defeat this count. |
| K-029 | 2026-09-09 | high | 🔵 | **Nothing audits a skill's frontmatter — that it parses, fits the 1,536-char listing budget, and names the right audience. One missing check, not three** (consolidates K-027 + K-028 + U36's parse finding; no content dropped). The three are the same gap seen three ways, and the budget is what makes the other two moot: `python-web-quirks`' description is **4,584 chars**, so U35's audience fix sits ~3,000 chars past the cut and reaches no router. Carries a remediation (that description) and a check (`audit-team.sh`), plus the instrument caveat that strict YAML is the **wrong** parser to audit with. |
| K-030 | 2026-09-09 | **high** | 🔵 | The always-loaded-prompt compaction backlog is **four agents**, not four separate follow-ups: `tdd-engineer.md:40`, `architect.md:51`, `data-scientist.md` (7 lines >700), and `teco.md`. The common mechanism is having **no knowledge base** — `teco`, `architect` and `data-scientist` have none, so every promotion can only fold onto the resident prompt. **`teco.md` after U44: 9,063 words, 31 lines >700, 17 >1,000, longest 1,716** — +583 (U42), +377 (U43), **+785 (U44)**, i.e. **+1,745 words / +24% in three units**, and U44 pushed **two** lines over 700 and three over 1,000. **U44 is the first unit to supply the counterfactual rather than the complaint:** 9 of its 12 entries routed to `teco.md` *because there was nowhere else*, while the two with an alternative home cost `teco.md` almost nothing — `511f0797` took one 400-char clause plus a pointer, with its measurement in `skills/agent-standards/claude-code.md`, and `d7a91e35` took **zero** `teco.md` words, landing whole in `claude/analyst/review-techniques.md`. Roughly half of U44's +785 is **method** content (how to build an instrument, which argument to mutate, how the accumulated lesson list is phrased) that a coordinator needs only while actually verifying — the definition of on-demand. Raised to **high**: the mechanism is now measured, not inferred. Decide the general question (knowledge base vs. restructure) before compacting any one file; `teco`'s answer is a stakeholder call, reserved as such in U42's brief. |
| K-031 | 2026-09-10 | medium | 🔵 | **Shared-working-tree commit knowledge is fragmented by *audience*, not by topic — and the split leaks.** Five statements now cover it: `claude/AGENTS.md`'s atomicity paragraph (index race, path-limited remedy, and — U44 — the unconditional index-ignored rule), its universal interactive-mode grant, `teco.md`'s grant bullet (path-limited form + the U43 disjointness condition), `teco.md`'s new holding-cost bullet (U44), and `teco.md`'s grant-scoping bullet. **Within a `teco` session the set is coherent** — the two `teco.md` bullets are adjacent and cite `claude/AGENTS.md` twice by section title. **For every other agent it is not:** `claude/AGENTS.md` grants all twelve agents path-limited committing into this shared tree and gives them the index race, but the two facts that bound the grant — *path-limiting protects nobody where the paths are not disjoint*, and *holding a shared file out has a rising cost with no natural end* — exist **only** in `teco`'s prompt, and nothing in `claude/AGENTS.md` points at them. Recommended home: the `claude/AGENTS.md` atomicity paragraph, because the grant that creates the hazard is already stated there for everyone. **Cost, and why it is not free either way:** moving prose *into* the always-loaded context file makes eleven other agents pay tokens for it every session (~+100 w on a file already 53 past its smell, forcing the K-032 trim first), while leaving it in `teco.md` keeps the cost narrow and leaves the other agents granted-but-uninformed. The deciding question is empirical and unmeasured: **how often does a non-`teco` agent actually commit into this tree under the interactive-mode grant?** Measure that before moving anything. Not performed in U44 by instruction. |
| K-032 | 2026-09-10 | low | 🔵 | **`claude/AGENTS.md`'s roster enumerates each agent's knowledge-base *topics*, duplicating `claude/README.md`, and the two have now measurably diverged.** `AGENTS.md:37-41` lists `guard-testing-techniques.md` as *coverage probe vs. mutation test, the two axes of a hand-written resolver, the docstring-states-more-than-the-body defect*; `README.md`'s parallel entry carries those **plus** the probe/oracle split U43 added. **The divergence is not the defect — the enumeration is.** A context file is always loaded in full, `README.md` is the catalog of record, and no agent routes to another agent's knowledge base (routing runs on the injected `description`), so the topic lists buy nothing and create a standing two-place update duty on the copy that rots unnoticed. **Recommended fix: do not add the missing topic — delete the enumerations**, leaving the roster to name each KB and point at `README.md` (~35 words back on a file 53 past its smell, and it retires the duty). Same judgement applies to the six other roster entries carrying topic lists. **Deliberately not fixed silently in U44**; U44 likewise added no `README.md` entry because it created **zero** new sections anywhere. Carries the K-031 dependency: this trim is what makes room for that consolidation. |
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

### K-025 — A `MENTIONS` tag creates an obligation no pass is shaped to drain
- **Status:** 🔵 proposed
- **Priority:** high
- **Origin:** U31 (`claude/docs/plans/kaizen-distillation2-coordination.md`), 2026-09-09 — see
  `history.md`. The unit's whole population, 11 nodes, was produced by this gap.
- **The defect, precisely.** When `cobb` tags an entry `MENTIONS → <other agent>` and resolves its
  `PRODUCED` edge, the node survives **by design**, carrying the half of the fact that belongs to
  the second agent. `teco`'s U18b states the intent: *"each survives with its `MENTIONS` edge …
  and surfaces in the tagged agent's own distillation pass."* **It does not.** A distillation unit
  is scoped `MATCH (a:Agent {agentId})-[:PRODUCED]->(e)`, which by construction cannot see a node
  whose only edge is `MENTIONS`. So the tag writes an obligation into a queue with no consumer.
- **Evidence it is structural, not incidental.** 11 nodes accumulated between 2026-08-30 and
  2026-09-07, spanning 7 agents and 2 coordination passes — the oldest population in the graph.
  **Six** of those agents were recorded "closed out, 0/0" by their own units while still holding an
  unrouted edge. The coordination doc noticed **two** instances as one-off deferrals owed at pass
  close and did not generalise; there were eleven.
- **Proposed change, two candidates — the choice is the work:**
  1. **Give the queue a consumer.** Add to `skills/agent-maintenance/SKILL.md` §5 that an agent's
     unit is scoped by **both** edge kinds — `PRODUCED` *or* `MENTIONS` — and that "closed out"
     means both read zero. Cheapest, and it makes every future unit self-draining.
  2. **Stop the tag implying a promotion.** Make `MENTIONS` purely an attribution marker and
     require the tagging unit to either promote the second agent's half itself, or open a plan item
     on the **mentioned** agent's backlog. This trades a queue for a synchronous cost at tag time.
- **Note either way:** the count-and-decide arithmetic already handles the multi-edge case
  correctly (`otherRemaining = producedEdges + mentionEdges − 1`); U31 exercised it on a two-edge
  node. The gap is in **scoping**, not in the clear.

### K-026 — `python-web-quirks` now has two `TestClient` sections; at a third, they need a parent, not a fourth sibling
- **Status:** 🔵 proposed
- **Priority:** medium
- **Origin:** U35 (`claude/docs/plans/kaizen-distillation2-coordination.md`), 2026-09-09 — see
  `history.md`.
- **The risk.** The skill now carries two sections on `starlette.testclient.TestClient` — teardown
  cancelling every still-running task, and the default `raise_server_exceptions=True` re-raising
  into the caller. They were deliberately **not** merged: different mechanisms, different
  consequences, different fixes, so one section would make a reader hunt inside it for the branch
  that applies. They are bound instead by a shared opening line naming the class — *`TestClient` is
  not a real client, and each convenience it adds is separately opt-outable.*
- **The trigger, stated so it is not a judgment call later.** On the **third** `TestClient` entry,
  promote that shared line to a parent heading (`## TestClient is not a real HTTP client`) with the
  individual surprises as sub-sections — do not add a third flat sibling. A skill that accretes one
  flat section per surprise in a single class stops being scannable, and this class has an obvious
  generator (every `TestClient` convenience is a divergence from a real client).
- **Cheap check:** `grep -c '^## .*TestClient' skills/python-web-quirks/SKILL.md` — currently 2.

### K-029 — Nothing audits a skill's frontmatter: that it parses, fits the listing budget, and names the right audience
- **Status:** 🔵 proposed
- **Priority:** high
- **Origin:** U35 (the audience list, the budget) and U36 (the parse, the consolidation),
  2026-09-09. **Consolidates and closes K-027 and K-028** — no content was dropped; those two IDs
  are retired and never reused (`history.md`, 2026-09-09).
- **Why one item and not three.** All three are the same missing check on the same six lines of
  YAML, and they are not independent: the budget is what makes the other two moot. U35 corrected
  `python-web-quirks`' audience clause in all four places it is stated — and that clause sits at
  the very **end** of a 4,584-character description, roughly 3,000 characters past the cut, so no
  router has ever read it. Auditing the audience list without auditing the budget certifies a
  string nothing reads.

**(a) The budget — the instance, and it is real.** Verified 2026-09-09 against
`code.claude.com/docs/en/skills`: *"the combined `description` and `when_to_use` text is truncated
at 1,536 characters in the skill listing to reduce context usage."* No error is raised, the skill
still loads, and the **body** is unaffected — only the listing, which is the text a router reads to
decide whether to invoke at all. Measured with `yaml.safe_load`, not raw text (see (d)):
`python-web-quirks` **4,584**; siblings all inside budget — `cpg-analysis` 1,003,
`agent-maintenance` 947, `joern-cpg` 555, `agent-standards` 475. This session's own listing cuts
`python-web-quirks` mid-clause at **1,534**. Everything past ~1,536 is invisible to routing: the
audience clause, and the fact clauses added by U7b, U8, U9, U12, U22 and U35. Those units' **body**
promotions are unaffected; their description extensions are not, and U35's audience fix survives
only in the three prose catalogs (`skills/README.md`, root `AGENTS.md`, `claude/README.md`), which
are documentation and are not truncated.

- **Not fixed as a rider, deliberately.** Compressing 4,584 → 1,536 means deleting roughly two
  thirds of an enumerated fact list six accepted distillation units deliberately built, in an
  artifact five agents load. Three candidates, increasing cost:
  1. **Reorder, delete nothing** — move the audience clause and the highest-value triggers into the
     first 1,536 chars. Cheapest, non-destructive, matches the doc's own *"put the key use case
     first"*; concedes the rest is decoration.
  2. **Compress to a routing description** — the description's job is *when to invoke*, not *what is
     inside*. Replace the fact enumeration with a trigger list plus categories; the body already
     carries every fact in full. Most correct, largest single edit.
  3. **Split the skill** — web/async framework quirks vs. pytest/test-harness traps, two packages
     each inside budget. Also fixes the grows-forever property that produced this, and gives
     K-026's `TestClient` family a natural home.

**(b) The audience list — a cross-artifact claim with no cross-artifact check.** `6b0a401 chore(claude):
trim agent frontmatter descriptions` removed the reciprocal `python-web-quirks` routing clause from
all four consumer prompts (`grep -rn 'python-web-quirks' claude/*/*.md` now returns nothing). The
trim is defensible — skills are discovered by their own description — but it leaves that
description the **sole** routing signal, and it had gone stale, naming `coder, tdd-engineer,
architect, analyst` while `qa-engineer` was producing kaizen entries on the skill's own subject.
Nothing checks this: §3 audits catalogs against disk, §4 audits agent-to-agent interfaces; a
skill's audience list is neither. The list is also stated in `SKILL.md`, `skills/README.md` and
root `AGENTS.md` — a greppable triple that drifted here.

**(c) The parse — found by U36, and its answer is the opposite of what it looked like.**
`skills/agent-standards/SKILL.md`'s frontmatter did not parse under `yaml.safe_load` (*mapping
values are not allowed here*, line 2 col 404) because a plain YAML scalar may not contain `: `, and
the description read *"Perishable: every fact is dated"*. **Claude Code loads it anyway** — see (d).
Fixed in U36 by converting the description to a folded block scalar (`>-`), the idiom
`cpg-analysis` and `python-web-quirks` already use, with the value byte-identical to `HEAD`. Done
for cross-tool portability, not because anything was broken: `skills/` is the cross-tool home and
another harness's parser may well be strict.

**(d) The instrument caveat — load-bearing, and the reason this must not be written with
`yaml.safe_load`.** Strict YAML is the **wrong** parser for auditing Claude Code frontmatter. A
repo-wide scan of all 24 frontmatter files, 2026-09-09, found **four** strict-parse failures, all
the same `: `-in-a-plain-scalar cause: `skills/agent-standards/SKILL.md` plus the agent definitions
`claude/tdd-engineer/tdd-engineer.md`, `claude/frontend-engineer/frontend-engineer.md` and
`claude/security-expert/security-expert.md`. **All four load correctly** — the three agents appear
in this session's agent-type listing and the skill in its skill listing, each with the complete
description *including the text after the offending colon*, and invoking `agent-standards` rendered
its body end to end. So the harness runs a real YAML parser (it folds `>-` block scalars correctly,
observed live) that is **more permissive than PyYAML on plain scalars containing `: `**. An audit
built on `yaml.safe_load` would fail three live agent definitions on day one — this pass's own
recurring lesson (*test the instrument before filing the finding*) aimed at the check itself. The
three agent files are therefore **left alone**: the defect is inert in the only harness that loads
them, and rewriting a routing `description` has a behavioural surface for no benefit.

- **Proposed change — one check, three assertions, in `claude/scripts/audit-team.sh`** (not §4's
  judgment checklist, and not §5's routing step): all three are mechanical and drift silently,
  which is exactly what checks 7 and 8 already are. For each `skills/*/SKILL.md` and each
  `claude/*/*.md`: (1) frontmatter yields `name` + `description` under a **lenient** read — accept
  a plain scalar containing `: `, and if a strict parser is used, fall back rather than fail;
  (2) `len(description) <= 1536`, reported as a number so a near-miss is visible before it becomes
  a cut; (3) for a skill, the audience list is stated identically in `SKILL.md`,
  `skills/README.md` and root `AGENTS.md`. Sub-check (3)'s harder half — *is any agent outside the
  list working in this skill's subject area* — is judgment and belongs in §4, referencing this
  check's output. Add to `skills/agent-maintenance/SKILL.md` §5's routing step the standing rule a
  script cannot enforce: **a unit promoting into a skill checks the description against the budget
  instead of appending to it by reflex.**

### K-030 — The always-loaded-prompt compaction backlog is now four agents, and distillation is what feeds it

- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** Four separate follow-ups in `claude/docs/plans/kaizen-distillation2-coordination.md`
  name the same defect in four prompts — `tdd-engineer.md:40` (1,502 chars), `architect.md:51`
  (1,617), `data-scientist.md` (seven lines over 700, longest 1,247), and `teco.md`. This is one
  item about a **mechanism**, not four about four files. **`teco.md`, measured across the pass:**
  7,901 w at `117df76` (U41) → 8,278 w at U43 → **9,063 w at `8973269`** → **10,445 w after U45**,
  i.e. **+2,544 words in four distillation units**, of which **+1,382 landed in U45 alone**. Lines
  over 700 chars went 26 → 32; over 1,000, 13 → **20**; and the longest line went 1,677 → **3,105**
  — `teco.md:89` (step 3, *Brief contents*), which absorbed two U45 promotions and is now nearly
  twice the baseline maximum. `teco`'s own `kaizen/plan.md` **K-016** (blocking) proposes the same
  split from the other side.
- **The mechanism, and U45 is the cleanest evidence for it yet:** an agent with **no knowledge
  base** has exactly one landing site for every promotion, so distillation can only ever fold onto
  its always-loaded prompt. `analyst`, `graph-dba`, `tdd-engineer`, `qa-engineer`, `data-scientist`
  and `devops` each have one; `teco` and `architect` do not. **Eight of U45's thirteen dispositions
  landed in `teco.md`, and the routing evidence says most of them did not belong there** — they are
  on-demand *gate and brief techniques* (how to read an empty result, how to probe a mutation
  table, how to size a dispatch), consulted when a specific situation arises, not routing rules
  needed in most sessions. That is §5's knowledge-base criterion stated exactly. The contrast is
  decisive rather than suggestive: in the same unit, two entries that met the same criterion
  **left `teco` entirely** — `f0f56a09` to `claude/analyst/review-techniques.md` and `5fc1dfea` to
  `skills/agent-maintenance/SKILL.md` — because a suitable on-demand home existed. The others had
  nowhere to go. **Argue this from routing, not from the word count**: the word count is the
  symptom, the absent landing site is the cause.
- **Proposed change:** decide the general question before compacting anything — does an agent whose
  prompt has crossed the density threshold get a knowledge base, or does its prompt get
  restructured? Then one compaction unit per prompt, none of them a side effect of a distillation
  (K-026's rule). For `teco` specifically the candidate is `claude/teco/coordination-techniques.md`,
  mirroring `analyst/review-techniques.md`. **`teco`'s answer is a stakeholder call and was
  explicitly reserved as such in U42's brief** — do not create one unilaterally, and note that
  creating it would restructure `teco.md` as a distillation side-effect, which is exactly what
  K-026 exists to prevent.
- **Notes:** Opened 2026-09-09 from U42; rewritten 2026-09-10 after U45 with the four-unit growth
  series, the eight-of-thirteen routing figure, and the two contrasting out-of-`teco` promotions.
  Raised to **high** — the table row already said high while this body still said medium; the body
  was the stale half. Entry ids are not the trigger here; no `entryId` dedup applies.

### K-031 — Shared-tree commit rules are split by audience, and the half that bounds the grant is teco-only

- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** U44 was asked to judge whether four (now five) statements about committing in a
  shared working tree are fragmentation or correct separation. Read as they now stand, they are
  coherent **for `teco`** and incomplete **for everyone else**. `claude/AGENTS.md` is loaded by every
  session in `claude/`; `teco.md` by one agent. The context file grants all twelve agents
  path-limited committing (*Universal interactive-mode grant*) and explains the index race — but the
  two conditions that bound the grant live only in `teco.md`: **disjointness** (a pathspec commit
  protects a concurrent session only where the paths do not overlap, U43 `0e2a0bf5`) and the
  **holding cost** (leaving a shared file uncommitted has a rising cost and no natural end, U44
  `9ba6b4f4`). Nothing in `claude/AGENTS.md` points at either. An `analyst` or `coder` committing
  interactively is therefore told it may commit, told how the index can bite it, and not told the
  two things that decide whether its commit is safe.
- **Proposed change:** move (not copy) the disjointness condition and the holding-cost bound into
  `claude/AGENTS.md`'s atomicity paragraph, and reduce `teco.md`'s two bullets to the integrator
  specifics plus a citation — the pattern `teco.md` already uses twice for that paragraph.
- **Notes:** Not free in either direction, which is why it is filed rather than done. Consolidating
  into the always-loaded file makes eleven agents pay for it every session and needs K-032's trim
  first (that file is already 53 words past its smell). Leaving it makes the split permanent. **The
  deciding fact is unmeasured: how often a non-`teco` agent actually commits into this tree under
  the interactive grant.** `git log` attribution cannot answer it directly — every commit carries
  the same author — so the measurement is a survey of agent kaizen histories, or an explicit
  decision that the universal grant is rare enough to ignore. Depends on: K-032. Related: K-030
  (the same "no knowledge base" pressure, from the other side).

### K-032 — The `claude/AGENTS.md` roster enumerates knowledge-base topics that `README.md` already owns

- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** The roster's per-agent parentheticals list each knowledge base's *topics*. So does
  `claude/README.md`, at greater length. U43 added the probe/oracle split to
  `guard-testing-techniques.md` and to `README.md`'s entry, and not to `AGENTS.md:37-41` — a
  divergence that is now demonstrable rather than hypothetical. The enumeration cannot be kept in
  sync by duty alone: it is a second copy of a catalog, in the one file that is always loaded in
  full and therefore the copy whose rot is invisible.
- **Proposed change:** delete the topic enumerations from the roster; keep the KB **names** and the
  existing pointer to `README.md`. Do **not** close the divergence by adding the missing topic —
  that ratifies the duplication. Roughly 35 words back, and the two-place update duty retires with
  it.
- **Notes:** Raised by U44 and deliberately left unfixed there, since the brief reserved the call.
  Seven roster entries carry such lists (`tdd-engineer`, `qa-engineer`, `analyst`,
  `data-scientist`, `graph-dba`, `devops`, and `tico`'s consult-roster clause); the same reasoning
  covers all of them, and it is one edit, not seven follow-ups. K-031 depends on the headroom this
  frees.

