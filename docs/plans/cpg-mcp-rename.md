# CPG MCP server/tool rename — Plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — (proposed **M6**, `docs/BACKLOG.md`) ·
> **Version:** 1.1 — revised after `analyst`'s plan-gate (`docs/reviews/cpg-mcp-rename.md`,
> verdict: needs changes); see §7 for the dated revision note.

Design for `docs/requirements/cpg-mcp-rename.md` (Status: Ready for design, FR-1…FR-7 /
AC-1…AC-6), per `docs/plans/cpg-mcp-rename-coordination.md` unit U1. Renames the MCP
server/tool from `cpg`/`mcp__cpg__query` to `cypher`/`mcp__cypher__query`, relocates
`cpg/mcp/` to a new top-level home, and sweeps every currently-active reference to the old
name repo-wide, while leaving genuinely CPG-specific naming (`cpg-analysis`, `joern-cpg`,
`cpg_<component>` graph names, the top-level `cpg/` directory, `graph-dba`'s Joern pipeline)
untouched.

**CPG:** considered, not relevant — this is a text/config/build-script rename across the
monorepo's own docs, agent prompts, and MCP plumbing, not a question about application code
semantics. Probed `GRAPH.LIST` live (via `mcp__cpg__query`) to confirm which graphs are
actually loaded: `ws:test, cpg_falkorchat, reference, ws:qa-tico-workflows-manual, ws:acme,
cpg_salesperson, ws:eval, kaizen_graph_dba` — none of these represents *this repo's own*
docs/scripts/prompts (the CPGs that exist are for `falkor-chat`/`salesperson` application
code, and `kaizen_graph_dba` is `graph-dba`'s working memory, unrelated to this sweep). A
call-graph/data-flow tool has nothing to offer a rename that lives entirely in markdown, YAML,
and shell scripts — the investigation was direct file reads plus `git grep`, matching the
precedent set by `docs/plans/generic-cypher-mcp.md`'s own CPG line for this exact component.

---

## 1. Goal & scope

Rename the MCP server/tool's entire visible identity — tool name, server key, directory,
Docker image/label namespace, environment-variable prefix — from `cpg` to `cypher`, and update
every currently **active** (non-archived) document, agent prompt, and skill that references it
by the old name, as **one atomic change** (FR-6: no dual-name period). Genuinely CPG-specific
naming is explicitly out of scope (FR-7) and must come out of a diff untouched.

In scope:
- Relocating `cpg/mcp/` to a new top-level directory (§3.1) and renaming every internal
  identity string inside it (server name, tool name, Docker image repo/label, environment
  variable prefix, internal shell-function names, log-line prefix).
- `.mcp.json` and `.claude/settings.json` harness wiring.
- Every agent's `tools:`/`allowed-tools:` allowlist entry naming the tool.
- The repo-wide documentation/prompt/skill sweep (§3.3), governed by a reusable discovery
  mechanism, not a fixed file list (the M5 precedent's B1 finding — see §2).
- A proposed `docs/BACKLOG.md` **M6** milestone entry and the numbering conflict it creates
  with `generic-cypher-mcp2.md` (§3.5, flagged as a decision, not silently resolved).
- Test/verification strategy for FR-6/FR-7 without re-reading 60+ files by hand (§5).

Out of scope (mirrors the requirements doc's own Out of scope, plus one call made here):
- Editing archived documents, or anything under `docs/archive/` (FR-4; §3.2).
- Any behavior change to the tool itself — read/write mechanics, author/curator enforcement,
  truncation, are byte-for-byte unchanged (requirements doc's own Out of scope).
- Moving this component's own documentation out of root `docs/` into a per-module
  `cypher-mcp/docs/` tree (the convention root `AGENTS.md` allows but does not mandate — "other
  modules adopt the structure when they first need it"; this delivery is a rename, not a
  docs-convention migration, and doing both at once would make the diff much harder to verify).
  This delivery's own artifacts stay at root `docs/{requirements,plans,reviews,test-plans,
  test-reports}/cpg-mcp-rename*.md`, matching where `docs/requirements/cpg-mcp-rename.md`
  already sits.
- A compatibility alias for the old name (FR-6, explicitly rejected by the stakeholder).

---

## 2. Context & findings

- **`cpg/mcp/`'s current contents** (read in full):
  `server.py` (35 KB, one FastMCP tool), `tests/{conftest.py, test_server.py,
  test_build_inputs.py}`, `Dockerfile`, `.dockerignore`, `build.sh`, `docker-run.sh`, `run.sh`,
  `setup.sh`, `image-tag.sh`, `pytest.ini`, `requirements.txt`, `requirements-dev.txt`,
  `README.md` (38 KB). Gitignored, untracked: `.venv/`, `.pytest_cache/`, `__pycache__/` — these
  regenerate at the new path via `setup.sh`/a fresh `pytest` run; nothing to "move."
- **`cpg/mcp/server.py`'s own identity strings** (line numbers from the file as read):
  - `mcp = FastMCP(name="cpg", instructions=SERVER_INSTRUCTIONS)` (line 680) — the server's own
    name, what produces the `mcp__cpg__query` callable.
  - `[cpg-mcp]` stderr log prefix (line 85).
  - `CPG_MCP_MAX_ROWS` / `CPG_MCP_MAX_CELL` / `CPG_MCP_MAX_CHARS` / `CPG_MCP_TIMEOUT_MS` /
    `CPG_MCP_CURATOR_AGENTS` env vars (lines 91–101).
  - Example calls embedded in curated error/instruction strings:
    `mcp__cpg__query(graph, cypher, agent='<your-agent-slug>')` (line 353) and others.
  - **Genuinely CPG-specific text that must NOT change** (FR-7), sitting in the *same*
    docstrings as the identity strings above: `"not limited to cpg_* graphs"`,
    `"cpg_<component>"`, `"skills/joern-cpg/references/cpg-model.md"`, `"CPG property keys are
    UPPER_CASE"`, `"the joern-cpg pipeline (graph-dba)"` (lines 117–154, 607–629). This is the
    clearest illustration in the whole sweep of the per-hit judgment call every step needs to
    make: same file, same docstring block, two categories of "cpg" that must be treated
    oppositely.
- **The `CPG_MCP_*` env-var prefix, `cpg-mcp` Docker image repo/label, and `cpg_mcp_*` shell
  function names are not named by FR-1…FR-3 individually**, but the requirements doc's own
  framing (*"Full identity, everywhere"*, decision log 2026-08-19) and FR-5 (*"any embedded
  references"*) cover them — they are "cpg" spelled into the tool's own implementation, not
  CPG-domain vocabulary. Decision: rename these too (§3.1). Flagged for the plan-gate reviewer
  as a call made without a named FR line, same pattern `docs/plans/generic-cypher-mcp.md` used
  for its own out-of-FR calls (its own §9 explicitly invited analyst pushback on one).
- **`cpg/mcp/image-tag.sh`'s content-hash mechanism is location-independent.** `CPG_MCP_DIR` is
  computed from `$(dirname "${BASH_SOURCE[0]}")` at run time, and every hashed path
  (`server.py`, `tests/...`, etc.) is relative to *that*, never to the repo root or an absolute
  path. Moving the directory changes nothing about how the hash is computed — only the renamed
  *bytes* inside those files (the identity strings above) change the hash's *value*, which is
  exactly the desired effect: one clean rebuild after the rename, not a broken gate. Confirmed
  by reading `image-tag.sh` in full (`cpg_mcp_input_files`/`cpg_mcp_input_dirs`/
  `cpg_mcp_image_tag`).
- **The M5 precedent failure this design must not repeat (B1):** `docs/plans/
  generic-cypher-mcp-coordination.md` U3 found that a *fixed file list* for a doc-sync step
  silently omitted `claude/graph-dba/graph-dba.md` and `claude/cobb/cobb.md` — both agents' own
  *operative* prompts, not catalog prose. The fix (`U2-fix`) replaced the fixed list with a
  `grep`-based before/after sweep as the actual done-condition. This plan adopts the same shape
  from the start (§3.3), not as a post-hoc fix.
- **Repo-wide discovery, already run** (`git grep -ilE 'mcp__cpg__query|cpg/mcp|"cpg"'`, minus
  `.git`): **97 files**. Cross-referenced every hit's own `Status:` header
  (`grep -m1 -oE 'Status:\*\* [A-Za-z ]*|Status: [a-zA-Z ]*'`). Representative findings by
  category (full categorization is §3.2, not restated per-file here):
  - **Archived, skip:** `docs/plans/{cpg-agent-adoption*, cpg-followups-coordination,
    cpg-mcp-containerization, cpg-query-access*, generic-cypher-mcp*,
    kaizen-inbox-distillation2-coordination}.md`, `docs/requirements/{cpg-agent-adoption,
    generic-cypher-mcp}.md`, `docs/reviews/{cpg-agent-adoption, cpg-followups-impl,
    cpg-mcp-containerization, cpg-query-access, generic-cypher-mcp}.md`,
    `docs/test-plans/{cpg-agent-adoption, generic-cypher-mcp}.md`,
    `docs/test-reports/{cpg-agent-adoption-report, generic-cypher-mcp-report}.md`,
    `falkor-chat/docs/plans/llm-provider-config.md`, `mcp-monitor/docs/plans/
    {mcp-monitor, mcp-monitor-coordination}.md`, `mcp-monitor/docs/reviews/mcp-monitor.md`.
  - **`docs/archive/` legacy tree, skip regardless of header** (pre-dates the `Status:` header
    convention; root `AGENTS.md`: *"nothing is ever moved into them again, and nothing is
    un-archived"*): `docs/archive/test-plans/cpg-query-access.md`,
    `docs/archive/test-reports/cpg-query-access-report.md`.
  - **Active, update:** `docs/manuals/cpg-getting-started.md`, `docs/requirements/
    {joern-cpg-pipeline, generic-cypher-mcp2}.md`, `docs/reviews/{analyst-inbox-distillation,
    cpg-getting-started, cpg-mcp-joern-agent-string-fix, graph-dba-kaizen-distillation,
    kaizen-inbox-distillation2, safety-net-guard-fixes}.md`, `docs/test-plans/
    {cpg-agent-adoption2, cpg-getting-started}.md`, `docs/test-reports/
    cpg-getting-started-report.md`, plus every `cpg/mcp/*` file, `claude/AGENTS.md`,
    `claude/README.md`, `claude/{analyst,architect}/*.md`, `skills/{cpg-analysis,
    agent-maintenance, agent-standards, joern-cpg}/**`, `mcp-monitor/{AGENTS,README}.md`,
    `mcp-monitor/docs/{BACKLOG,HISTORY}.md`, `falkor-chat/compose.yaml`, `.mcp.json`,
    `.claude/settings.json`, root `AGENTS.md`.
  - **Living logs, no `Status:` header, always in scope, surgical edit only (§3.2):**
    `docs/BACKLOG.md`, `docs/HISTORY.md`, every `claude/*/kaizen/{history,inbox}.md`.
  - **Self-referential to this very delivery, exempt from substitution (§3.2):**
    `docs/requirements/cpg-mcp-rename.md`, `docs/plans/cpg-mcp-rename.md` (this document),
    `docs/plans/cpg-mcp-rename-coordination.md`.
- **`.claude/settings.json`** carries `"enabledMcpjsonServers": ["cpg"]` — a config file with no
  `Status:` header (always in scope), and the literal server-name string that gates whether
  Claude Code loads the renamed server without a fresh interactive trust prompt surprise.
- **Two agents' `tools:` allowlists carry the literal callable name** —
  `claude/analyst/analyst.md:4` and `claude/architect/architect.md:4` both have
  `tools: Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch, Agent, mcp__cpg__query`.
  Per the M3 precedent (C-304), without this exact string the renamed tool is **invisible** to
  these two agents regardless of what the server itself is named — this is the functional
  equivalent of B1 for this delivery: a doc-prose sweep that missed these two lines would ship
  a rename that silently breaks two agents' CPG access.
- **`skills/cpg-analysis/SKILL.md:17`** carries `allowed-tools: mcp__cpg__query, Bash, Read` —
  same load-bearing category as the two `tools:` lines above, plus three body mentions (lines
  5, 35, 44, 185) of `mcp__cpg__query` as the tool the skill's recipes call. The skill's own
  *name* (`cpg-analysis`) and its CPG-domain description text stay untouched (FR-7) — only the
  tool-identity strings inside it move.
- **`docs/requirements/generic-cypher-mcp2.md`** (M6, Status: Ready for design — the next
  queued delivery) references `mcp__cpg__query` 4 times and its own header already claims
  `**Tracks:** — (M6)`. `docs/plans/cpg-mcp-rename-coordination.md` (teco, 2026-08-19) has
  already decided this rename lands **before** M6 goes into design specifically so M6 is
  designed against the final `cypher` name — which means this delivery's own natural milestone
  slot is also **M6**, creating a direct numbering collision addressed in §3.5.
- **No new dependency, no test-count change.** The rename touches string literals and file
  locations only; `requirements*.txt` are untouched, and the offline/live suite counts stay
  what they are today (last recorded: 84 passed/7 deselected offline) modulo the module import
  path (`import server` still resolves the same way from the new directory — nothing here
  changes Python import structure, since `cpg/mcp` was never a package, just a script directory
  run by path per `cpg/mcp/README.md`'s own "Dependencies" note).

---

## 3. Design & rationale

### 3.1 New location: top-level `cypher-mcp/`

**Decision: promote `cpg/mcp/` to a new top-level directory, `cypher-mcp/`, with the exact same
flat internal layout it has today** (`server.py`, `tests/`, `Dockerfile`, `.dockerignore`,
`build.sh`, `docker-run.sh`, `run.sh`, `setup.sh`, `image-tag.sh`, `pytest.ini`,
`requirements*.txt`, `README.md` — no new nesting level).

**Why a new top-level directory, not somewhere inside an existing one.** The requirements doc's
own "Context for the architect" note leaves this open but is explicit that `cpg/`'s role as
*"the CPG component's home (including `.cpg-artifacts/`)"* is unaffected — only the
generic-tool subdirectory moves *out*. There is no existing top-level directory this tool is a
natural sub-part of: it is not `falkor-chat`'s, not `salesperson`'s, and — per M5's own
already-recorded naming tension (`docs/plans/generic-cypher-mcp.md` §9: *"Revisit if this
pattern extends past `graph-dba` to a second agent's working memory"* — a trigger M6 is about
to pull) — it has outgrown being a CPG-component sub-tool. This repo's own precedent for "a
small, self-contained MCP-adjacent tool that is not part of any existing component" is
`mcp-monitor/`: a flat top-level directory, its own `AGENTS.md`/`README.md`, its own
`docs/BACKLOG.md`/`HISTORY.md`. `cypher-mcp/` follows that shape exactly.

**Why `cypher-mcp/`, not bare `cypher/`.** The server's own new name is literally `cypher`
(`FastMCP(name="cypher")`), so a bare top-level `cypher/` directory would read as "the Cypher
query language, generically" — plausibly confusable with a future non-MCP Cypher-related
component — whereas `cypher-mcp/` states precisely what is inside: *this MCP server*, mirroring
`cpg/mcp/`'s own "component-name + mcp" pattern and `mcp-monitor/`'s naming style.

**Why no `docs/` migration alongside the move.** Out of scope per §1 — bundling a docs-tree
migration into an already-60+-file rename would make the diff much harder for the `analyst`
plan-gate and the `qa-engineer` acceptance pass to verify cleanly. This component's docs stay
at root `docs/`, exactly where `cpg/mcp/README.md` already points readers via
`../../docs/plans/cpg-query-access.md`-style relative links — which become `../docs/plans/...`
once the directory is one level shallower (`cypher-mcp/` vs. `cpg/mcp/`); every such relative
link inside the moved `README.md` needs its leading `../../` shortened to `../` as a direct
consequence of losing one nesting level. Called out explicitly here because it is easy to miss
in a mechanical rename pass (the string doesn't say "cpg" so a `cpg`-focused grep wouldn't catch
it) — the implementer must inventory every relative `../../` link in the moved files as part of
step 1, not rely on the identity-string sweep to find it.

### 3.2 Discovery/sweep mechanism — status-driven, not a fixed file list

**The sweep is a live `git grep`, re-run at both the start and the end of implementation, never
a list transcribed once into this plan.** This is the direct fix for the B1 failure mode (§2):
a plan-authored file list is exactly the artifact that went stale last time.

```bash
git grep -zlE 'mcp__cpg__query|cpg/mcp|"cpg"|CPG_MCP_|cpg-mcp|cpg_mcp_|\bcpg\b' -- . ':!.git' \
  | tr '\0' '\n' | sort
```

Deliberately **unfiltered by extension** (per root `AGENTS.md`'s standing rename convention) —
this is why `.mcp.json`, `.claude/settings.json`, and `falkor-chat/compose.yaml` are in scope
even though none of them is markdown. The pattern set covers, beyond the bare tool name: the
directory path (`cpg/mcp`), the exact `.mcp.json`/`FastMCP` server-name literal (`"cpg"`), the
env-var prefix (`CPG_MCP_`), the Docker image/label namespace (`cpg-mcp`), the internal
shell-function-name prefix (`cpg_mcp_`), and — **added in this revision (§7, B1 fix)** — a
bare, case-sensitive `\bcpg\b` word-boundary alternative. The first five axes alone silently
missed every "the `` `cpg` `` server"/"`` `cpg` `` MCP tool"-shaped bare mention (no adjacent
`/`, `_`, `-`, or quote for the earlier alternatives to latch onto), including the running
server's own `SERVER_INSTRUCTIONS` self-description (§2) — the review's B1 finding, reproduced
live against 6 concrete lines before this fix and confirmed closed after it.

**Why the added alternative is deliberately case-*sensitive*, diverging from the review's own
literal suggested spelling (`` |\bcpg\b `` case-insensitive).** Tested both live against the
current tree before choosing: a case-*insensitive* `\bcpg\b` widens the file-hit count from 94
to 141 and pulls in two large, purely domain-vocabulary categories that rule 5 would then have
to triage for no forward benefit — the `` `CPG:` `` deliverable evidence-trail convention line
(`docs/plans/cpg-agent-adoption.md` §3, present in nearly every agent prompt and kaizen-history
entry) and "Code Property Graph (CPG)" acronym-definition prose — both spelled uppercase in
every occurrence found repo-wide. A case-*sensitive* `\bcpg\b` widens to 135 files (not 141),
still catches all 6 of the review's confirmed misses (verified: `cpg/mcp/server.py:2`,
`cpg/mcp/server.py:131`, `docs/BACKLOG.md:240`, `docs/BACKLOG.md:376`, `docs/HISTORY.md:465`,
`claude/graph-dba/falkordb-quirks.md:277` all match), and produces **zero** hits against either
noise category in the same live test — every real tool-identity mention found in this repo uses
lowercase `` `cpg` ``, never uppercase. The remaining ~40-file delta (135 vs. the pre-fix 94) is
overwhelmingly the `cpg-analysis` skill name and other pre-existing `cpg-`prefixed topic
slugs/filenames (§3.2 rule 5, expanded below) — real triage volume, not misclassification; §4's
step-sizing note is updated to state this honestly rather than understate it.

**Per-hit classification, in order:**

1. **Is the hit inside `docs/archive/`?** Skip unconditionally — this tree pre-dates the
   `Status:` header convention and root `AGENTS.md` states it is never touched again.
2. **Is the hit inside a document of this delivery's own family — any file under
   `docs/{requirements,plans,reviews,test-plans,test-reports}/` whose basename *starts with*
   `cpg-mcp-rename`** (`docs/requirements/cpg-mcp-rename.md`, `docs/plans/cpg-mcp-rename.md`,
   `docs/plans/cpg-mcp-rename-coordination.md`, `docs/reviews/cpg-mcp-rename.md`, and — once
   they exist — `docs/test-plans/cpg-mcp-rename.md` / `docs/test-reports/
   cpg-mcp-rename-report.md`)? **Widened in this revision (§7, M1 fix)** from a fixed 3-item
   list to this basename rule, because the review found a real 4th member the fixed list missed:
   `docs/reviews/cpg-mcp-rename.md` itself — produced *before* the docs sweep runs (per
   `docs/plans/cpg-mcp-rename-coordination.md`'s own unit ledger), `Status: active`, and by
   design full of literal `mcp__cpg__query`/`"cpg"`/`cpg/mcp` quotations naming its own findings'
   subject matter. Every document in this family describes the rename itself —
   *"was named `mcp__cpg__query`, becomes `mcp__cypher__query`"* is their subject matter, not a
   stale reference. Leave every old-name occurrence that is *naming what is being renamed from*
   exactly as written; only update a family member for its own lifecycle mechanics (a `Status:`
   flip at milestone close, a `Version:` bump, etc.), never for the substitution rule itself. A
   basename-prefix rule, not a fixed list, so a document this pass didn't think to name (a
   `-report` suffix, a future revision) is covered automatically — the same lesson B1 (§7) drew
   for the discovery pattern itself, applied here to the exemption list.
3. **Does the hit's own document carry a `Status:` header (`docs/{requirements,plans,
   reviews,test-plans,test-reports}/*.md`, including a `plans/*-coordination.md` sibling — there
   is no separate `docs/plans-coordination/` directory; coordination docs live inside
   `docs/plans/` itself, e.g. `docs/plans/cpg-mcp-rename-coordination.md`, corrected in this
   revision per §7's m1 fix)? Read the token
   (`grep -m1 -oE 'Status:\*\* [A-Za-z ]*'`). `archived` → skip, no edit, not even a header
   pointer (FR-4 names no supersession need here, unlike the AC-8 precedent in
   `generic-cypher-mcp.md` §5 — this is a pure naming change, not a scope widening). Anything
   else (`active`, `Ready for design`, `superseded`) → in scope, go to step 5.
4. **Is the hit inside a living log with no `Status:` header** — `docs/BACKLOG.md`,
   `docs/HISTORY.md`, `mcp-monitor/docs/{BACKLOG,HISTORY}.md`, or any
   `claude/*/kaizen/{history,inbox}.md`? These never carry the archived-freeze mechanism (root
   `AGENTS.md`: *"Modules do not use `kaizen/` dirs [for the `Status:` convention] — that
   convention exists only for agent folders,"* and `BACKLOG.md`/`HISTORY.md` are explicitly
   the two kinds *not* in the closed `Status:`-bearing set). They are always in scope under
   AC-1's literal wording (*"zero hits outside archived documents"* — these documents are never
   archived, so a hit here is always a live one). **Apply surgical, token-only substitution**:
   rewrite only the identity strings themselves (`mcp__cpg__query` → `mcp__cypher__query`,
   `cpg/mcp` → `cypher-mcp`, etc.) inside a dated entry's prose, and change nothing else about
   that entry — no rewording, no re-dating, no touching adjacent figures/decisions. This
   preserves the historical record's substance (a 2026-07-25 entry still says what happened on
   2026-07-25) while satisfying AC-1's letter. **This is an explicit interpretive call, not
   stated in the requirements doc's decision log — flagged in §6 for the plan-gate reviewer to
   confirm or overrule** (the plausible alternative is "never touch a dated log entry, since it
   is historical narration" — rejected here because AC-1 has no carve-out for undated-vs-dated
   documents, only for the `Status:`-driven archived/active split, and a living BACKLOG.md that
   still names a tool that no longer exists is exactly the confusion FR-6 exists to prevent for
   a reader using it as a current reference).
5. **Within an in-scope hit, is the specific occurrence tool-identity or CPG-domain
   vocabulary (FR-7)?** Tool-identity: `mcp__cpg__query`, the `.mcp.json`/`FastMCP` `"cpg"`
   server key, `cpg/mcp` as a path, `CPG_MCP_*` env vars, `cpg-mcp` Docker image/label,
   `cpg_mcp_*` shell functions, the `[cpg-mcp]` log prefix, **and — added in this revision (§7,
   B1 fix) — a bare, standalone `` `cpg` ``/`cpg` token used as a noun or adjective naming the
   server or tool itself and *not* part of a hyphenated/underscored compound**: "the `` `cpg` ``
   server," "`` `cpg` `` MCP tool," "restart `` `cpg` ``," "`` `cpg` `` is connected," "`` `cpg` ``
   uses/sets/lists..." → rename per §3.4. CPG-domain (leave untouched): "Code Property Graph,"
   `cpg_<component>` graph-name literals (`cpg_falkorchat`, `cpg_salesperson`, or the generic
   pattern description), `cpg-analysis`/`joern-cpg` skill names, `graph-dba`'s Joern pipeline,
   the top-level `cpg/` directory's own identity (including `.cpg-artifacts/`, `cpg/.gitignore`'s
   `cpg.bin` mention, and pipeline scripts like `cpg-to-falkordb.py`/`build-cpg.sh`/
   `export-cpg.sh`) — **and, added in this revision, two categories the widened net in §3.2
   newly surfaces at volume:** (i) the **`` `CPG:` `` deliverable evidence-trail convention
   line** (`` `CPG: used <graph> — <clause>` `` / `` `CPG: considered, not relevant — <clause>` ``
   / `` `CPG: not applicable — <clause>` ``, per `docs/plans/cpg-agent-adoption.md` §3, present
   in nearly every agent prompt's deliverable skeleton and kaizen-history entry) — always spelled
   uppercase with a trailing colon, never referring to the renamed server by name, so trivially
   distinguishable from a tool-identity hit; (ii) **pre-existing `cpg-`prefixed topic
   slugs/filenames naming a *different* document, milestone, or script** — not the MCP tool's own
   identity — e.g. `cpg-agent-adoption`, `cpg-followups`, `cpg-query-access`,
   `cpg-getting-started`, `cpg-mcp-containerization`, `m2-cpg-analysis`, `cpg-model.md`. §2's
   `server.py` docstring example (same lines, both categories present) is the canonical worked
   case; `docs/manuals/cpg-getting-started.md` is the second-clearest one — its title, its "what
   is a CPG" explanation, and its `cpg_<name>` graph-naming walkthrough all stay; its "the `cpg`
   MCP tool is connected... run `claude mcp list`... look for `cpg`" instructions (lines 65–67)
   become `cypher`.

**Proof nothing was missed or wrongly touched (the actual FR-6/FR-7 gate, replacing a fixed
list — §5 restates this as the acceptance strategy):** re-run the same `git grep` after every
step lands. Every surviving hit must resolve to one of: (a) inside `docs/archive/`, (b) inside a
document whose own `Status:` header reads `archived`, (c) inside a `cpg-mcp-rename*` family
document (rule 2), naming the old identity as the explicit subject of a "renamed from X"
sentence. Any other surviving hit is a defect. Separately, a **second** grep for the FR-7
preservation list (`cpg_falkorchat|cpg_salesperson|cpg-analysis|joern-cpg|Code Property Graph|
CPG:`) must return an **identical count and identical line content** before and after — any
delta there is a defect in the opposite direction (an over-eager rename).

### 3.3 Exact string mappings

| Axis | Old | New | Where |
|---|---|---|---|
| Tool callable | `mcp__cpg__query` | `mcp__cypher__query` | Everywhere (server code, `tools:`/`allowed-tools:` frontmatter, every doc/prompt reference) |
| `.mcp.json` server key / `FastMCP(name=...)` | `"cpg"` | `"cypher"` | `.mcp.json`, `cypher-mcp/server.py` |
| `.claude/settings.json` | `enabledMcpjsonServers: ["cpg"]` | `["cypher"]` | `.claude/settings.json` |
| Directory | `cpg/mcp/` | `cypher-mcp/` | Every path reference, repo-wide |
| Docker image repo | `cpg-mcp` (`CPG_MCP_IMAGE_REPO` default) | `cypher-mcp` | `build.sh`, `docker-run.sh`, `image-tag.sh`, `README.md` |
| Docker label | `cpg-mcp=1` | `cypher-mcp=1` | `Dockerfile` `LABEL`, `docker-run.sh --label`, `README.md`'s `docker ps`/`docker image ls --filter` recipes |
| Env-var prefix | `CPG_MCP_*` (`MAX_ROWS`, `MAX_CELL`, `MAX_CHARS`, `TIMEOUT_MS`, `CURATOR_AGENTS`, `IMAGE_REPO`, `NO_AUTOBUILD`, `IMAGE`, `NO_PULL`) | `CYPHER_MCP_*` | `server.py`, `docker-run.sh`, `build.sh`, `image-tag.sh`, `README.md`'s env-var table |
| Internal shell functions/vars | `cpg_mcp_input_dirs`, `cpg_mcp_input_files`, `cpg_mcp_image_tag`, `CPG_MCP_DIR`, `CPG_MCP_TAG` | `cypher_mcp_input_dirs`, `cypher_mcp_input_files`, `cypher_mcp_image_tag`, `CYPHER_MCP_DIR`, `CYPHER_MCP_TAG` | `image-tag.sh`, callers in `build.sh`/`docker-run.sh` |
| Log prefix | `[cpg-mcp]` | `[cypher-mcp]` | `server.py`'s stderr logger |
| Docker image tags (informational, produced by the hash — no manual edit) | `cpg-mcp:<hash>`, `cpg-mcp:test-<hash>`, `cpg-mcp:dev`, `cpg-mcp:test` | `cypher-mcp:<hash>`, `cypher-mcp:test-<hash>`, `cypher-mcp:dev`, `cypher-mcp:test` | Follows automatically from the `IMAGE_REPO` rename above; no separate step |

**Unchanged, deliberately (FR-7):** `cpg_<component>` graph-name literals and the naming
convention prose describing them, the `cpg-analysis` and `joern-cpg` skill names and their own
descriptions, `graph-dba`'s Joern build/load pipeline, the top-level `cpg/` directory and
`.cpg-artifacts/`, "Code Property Graph"/"CPG" as the domain term.

### 3.4 `.mcp.json` and `.claude/settings.json`

```json
{
  "mcpServers": {
    "cypher": {
      "command": "bash",
      "args": ["-c", "exec \"$CLAUDE_PROJECT_DIR/cypher-mcp/docker-run.sh\""],
      "env": { "FALKORDB_HOST": "host.docker.internal", "FALKORDB_PORT": "6379" },
      "timeout": 60000
    }
  }
}
```

Only the server key and the `args` path change; `command`, `env`, and `timeout` are unchanged
(nothing about launch mechanics moves). `.claude/settings.json`'s
`"enabledMcpjsonServers": ["cpg"]` becomes `["cypher"]` — without this, the renamed server would
sit `⏸ Pending approval` even though `.mcp.json` itself is correct (this is the exact
distinction `cpg/mcp/README.md`'s own "Running and debugging" section already documents: the
server config and the approval list are two different files that must agree). A session
restart is required for either edit to take effect, same as today.

### 3.5 Docker build/run implications — reproducibility is preserved, not just "probably fine"

- **The content-hash scheme itself needs zero structural change** (§2's finding on
  `image-tag.sh`'s location-independence). `build.sh`, `docker-run.sh`, and `image-tag.sh` move
  with the directory and keep computing paths relative to their own location.
- **The hash *value* changes, exactly once, as an intended side effect.** Every renamed file
  (`server.py`, `Dockerfile`, `requirements*.txt` untouched but still walked, `tests/**`) has
  different bytes after the string substitution, so `cpg_mcp_image_tag`/`cypher_mcp_image_tag`
  produces a new 12-char hash. `docker-run.sh`'s own staleness gate (`docker image inspect`)
  correctly reports a miss on first launch post-rename and builds — this is the designed
  behavior, not a regression; there is no old `cpg-mcp:<hash>` image to accidentally keep
  serving because the repo name itself also changes (`cpg-mcp` → `cypher-mcp`), so the two
  namespaces can never collide even transiently.
- **`build.sh --verify-inputs` must be re-run and pass** after the rename — it is the existing
  regression check that the `Dockerfile`'s `COPY` list and `image-tag.sh`'s hashed-input list
  still agree; renaming file *contents* without touching *which files exist* means this should
  pass unchanged, and running it is how the implementer proves that rather than assumes it.
- **No orphaned `cpg-mcp:*` images need deleting as part of this change** — per
  `cpg/mcp/README.md`'s own "Housekeeping" section, image pruning is already a manual,
  human-driven act, never something the launch path does automatically; the old-named images
  simply become inert history, exactly like any other superseded hash tag today.
- **`.dockerignore`'s header comment** (`# Context is cpg/mcp/. Patterns are relative to it...`)
  is a pure comment with no functional effect on which patterns match, but must still be
  updated (`git grep` catches it; leaving it stale would misdescribe the file to the next
  reader).

### 3.6 Proposed `docs/BACKLOG.md` milestone — and the M6 numbering collision

**Decision: this delivery becomes M6.** `docs/plans/cpg-mcp-rename-coordination.md` (teco,
2026-08-19) has already decided the rename lands *before* `generic-cypher-mcp2` goes into
design, specifically so M6-proper is designed against the final name. Milestone numbers in this
backlog are assigned by delivery order (M1…M5 already are), so the delivery that ships first
should hold the M6 slot.

**The collision this creates, stated plainly:** `docs/requirements/generic-cypher-mcp2.md`'s
own header already reads `**Tracks:** — (M6)` (committed 2026-08-19, before this plan). If this
rename also becomes M6, `docs/BACKLOG.md` would end up with two different `## M6 — …` sections
unless one of them is renumbered.

**Proposed resolution, flagged for the plan-gate reviewer to confirm rather than silently
applied:** bump `docs/requirements/generic-cypher-mcp2.md`'s header from `(M6)` to `(M7)` as a
one-line edit, folded into step 3b (§4) alongside the rest of that document's `mcp__cpg__query`
substitutions (it is already in scope for this delivery per §2). This is a call this plan makes
on a sibling, not-yet-designed delivery's behalf — legitimate because teco's own coordination
note already establishes the ordering, and the alternative (leaving both deliveries claiming
M6) is a guaranteed `docs/BACKLOG.md` defect — but it is exactly the kind of cross-delivery
decision that belongs in front of a reviewer before it ships, not assumed correct because it
was convenient to write here.

**Milestone-map row** (mirrors M3/M4/M5's exact format):

```markdown
| **M6 — MCP tool rename** | The MCP server/tool is renamed `cpg`/`mcp__cpg__query` →
`cypher`/`mcp__cypher__query`, relocated `cpg/mcp/` → `cypher-mcp/`; every active reference
repo-wide updated, genuinely CPG-specific naming (`cpg-analysis`, `joern-cpg`,
`cpg_<component>` graphs, top-level `cpg/`) untouched; AC-1…AC-6 acceptance-tested. | **C-601 → C-605** |
```

**Section body** (`## M6 — MCP tool rename`), items numbered to match §4's step table —
`C-601` (relocate + rebuild `cypher-mcp/`, step 1), `C-602` (harness + agent-tool-surface
wiring, step 2), `C-603` (`claude/`+`skills/` sweep, step 3a), `C-604` (`docs/`+
`mcp-monitor/`+`falkor-chat/` sweep, step 3b, including the `generic-cypher-mcp2.md` M6→M7
bump), `C-605` (acceptance pass, step 4) — each one line, status `🔵 proposed` until closed,
mirroring the M3/M4/M5 item style exactly. `docs/HISTORY.md`'s actual dated close-out entry is
written at milestone close (post-acceptance), same as M3/M4/M5's own pattern — not part of this
step table.

---

## 4. Implementation step table

Four steps (one is a same-owner adjacent pair, mirroring M5's 4a/4b), sized against this repo's
own ≤3-step/≤5-file dispatch guideline with two explicit, justified exceptions (steps 1, 3a,
3b each exceed 5 files):

| # | Owner | Files | Depends on | Done-condition |
|---|---|---|---|---|
| **1** (C-601) | `coder` | The 13 files of the relocated `cypher-mcp/` directory (`server.py`, `tests/{conftest.py,test_server.py,test_build_inputs.py}`, `Dockerfile`, `.dockerignore`, `build.sh`, `docker-run.sh`, `run.sh`, `setup.sh`, `image-tag.sh`, `pytest.ini`, `requirements.txt`, `requirements-dev.txt`, `README.md`) | — | `git mv cpg/mcp cypher-mcp`; every §3.3 mapping applied inside the moved files; every `../../docs/...` relative link in `README.md` shortened to `../docs/...` (§3.1); `cypher-mcp/setup.sh` runs clean; offline suite green at the same count as today (84 passed/7 deselected) from the new path; `cypher-mcp/build.sh` run once by hand, `--verify-inputs` passes; in-container offline+live gates green (`cypher-mcp/README.md`'s own "in-container test gate" recipe, run from the new path) |
| **2** (C-602) | `coder` | `.mcp.json`, `.claude/settings.json`, `claude/analyst/analyst.md`, `claude/architect/architect.md`, `skills/cpg-analysis/SKILL.md` (frontmatter `allowed-tools:` line + the 3 body mentions of `mcp__cpg__query`, nothing else in that file) | 1 | `.mcp.json`/`.claude/settings.json` match §3.4; both `tools:` lines and the skill's `allowed-tools:` line read `mcp__cypher__query`; session restart + `claude mcp list` shows `cypher — ✔ Connected`, no `cpg` entry at all |
| **3a** (C-603) | `cobb` | `claude/AGENTS.md`, `claude/README.md`, every `claude/*/*.md` operative prompt with a live hit (per §3.2's fresh sweep at dispatch time — not the file list in §2, which is a snapshot), every `claude/*/kaizen/{history,inbox}.md` with a hit (surgical edit per §3.2 rule 4), `skills/{cpg-analysis,agent-maintenance,agent-standards,joern-cpg}/**` body prose (frontmatter already done in step 2), root `AGENTS.md` | 1 | Re-run §3.2's widened `git grep`; every remaining hit under `claude/` or `skills/` resolves to one of rule 2's family exemption, an `archived` `Status:`, or (per §7's B1 fix) confirmed CPG-domain vocabulary under rule 5 — most of this subtree's wider hit volume is exactly this last category (`cpg-analysis` mentioned in nearly every agent's own routing description) and resolves to "no edit," which is expected, not a triage failure; FR-7 preservation grep (§3.2, second gate) unchanged in count and content for this subtree |
| **3b** (C-604) | `cobb` | `docs/{plans,requirements,reviews,test-plans,test-reports}/*.md` with a live hit (per the fresh sweep), `docs/manuals/cpg-getting-started.md`, `docs/BACKLOG.md` (surgical edits to existing prose **plus** the new §3.6 M6 section), `docs/HISTORY.md` (surgical edits only — no new entry yet), `mcp-monitor/{AGENTS,README}.md`, `mcp-monitor/docs/{BACKLOG,HISTORY}.md` (surgical), `falkor-chat/compose.yaml`, and the `docs/requirements/generic-cypher-mcp2.md` `(M6)`→`(M7)` bump (§3.6) | 1 (parallel with 3a — same owner, no interdependency, both gated only on step 1's new path existing) | Same grep-based done-condition as 3a, scoped to this file set; `docs/BACKLOG.md`'s M6 section present and matches §3.6; `generic-cypher-mcp2.md` reads `(M7)` |
| **4** (C-605) | `qa-engineer` | — (execution only; produces `docs/test-plans/cpg-mcp-rename.md` + `docs/test-reports/cpg-mcp-rename-report.md`) | 1, 2, 3a, 3b | AC-1…AC-6 each exercised live per §5 |

**Why steps 1/3a/3b exceed the ≤5-file guideline, explicitly.** Each is one *homogeneous*
operation governed by a single rule set (§3.2's classification + §3.3's mapping table) with a
mechanical, grep-based done-condition — not N independent judgment calls needing N separate
reviews. This mirrors the repo's own precedent: `docs/plans/generic-cypher-mcp.md`'s step 4a/4b
each covered ~9 files under the same reasoning, and its own close-out explicitly measured "on
the order of 30 hits" as the real triage volume for a step sized at "5 named files" — i.e., this
repo has already accepted that a mechanical sweep's *file count* and its *dispatch-sizing unit*
are different numbers. Splitting 3a/3b further by directory would multiply review overhead
(more diffs to gate) without reducing risk (the same single rule set governs all of it).
**Stated honestly (§7 revision), not understated:** §3.2's B1 fix widens the live discovery
`git grep` from 94 to 135 matching files repo-wide — most of the ~40-file delta resolves to "no
edit" under rule 5's expanded CPG-domain list (predominantly the `cpg-analysis` skill name,
already mentioned in nearly every agent's own routing description since M4). 3a/3b's real work
is therefore closer to "triage ~135 files, edit a smaller subset" than "edit 40-ish files" — the
same shape `generic-cypher-mcp.md`'s own close-out already measured and accepted (30 real hits
against a step named for "5 files"), not a new risk this revision introduces.

**Sequencing:**

```
1 ─┬─▶ 2  ─┐
   ├─▶ 3a ─┼─▶ 4 ⇒ M6 done
   └─▶ 3b ─┘
```

2, 3a, and 3b all depend only on step 1 (the new path/identity must exist before anything can
correctly reference it) and have no dependency on each other — 3a/3b dispatch as an adjacent
cluster (same owner, same single dependency), 2 can dispatch alongside them.

---

## 5. Test strategy

No new unit-test *behavior* to pin — this is a naming change, and §1's Out of scope already
states the tool's mechanics are unchanged. The verification burden is entirely "did every
active reference move, and did nothing CPG-specific get touched," which is what AC-1…AC-6 ask
and what §3.2's before/after grep gate already targets. `qa-engineer`'s step 4 exercises each
criterion live rather than re-reading 60+ files by hand:

| AC | Check | Altitude |
|---|---|---|
| AC-1 | `git grep -c 'mcp__cpg__query' -- . ':!.git'` for the literal callable, **plus** §3.2's full widened discovery pattern (`` |\bcpg\b `` included, per §7's B1 fix) re-run as the real proof — every surviving hit resolves to §3.2's allowed categories (archived doc, `docs/archive/`, a `cpg-mcp-rename*` family document per rule 2, or confirmed CPG-domain vocabulary under rule 5); zero unexplained hits | Static, two commands + manual triage of the survivor list (§4's revised sizing note: expect this list to be real triage volume, not a short one) |
| AC-2 | `python3 -c "import json; assert 'cypher' in json.load(open('.mcp.json'))['mcpServers'] and 'cpg' not in json.load(open('.mcp.json'))['mcpServers']"` or a plain read | Static |
| AC-3 | `cypher-mcp/build.sh` from a clean state; `docker-run.sh` launched; a live `mcp__cypher__query` call against a known graph (`cpg_falkorchat`, e.g. `MATCH (m:METHOD) RETURN count(m)`) compared against the pre-rename baseline figure already recorded in `docs/BACKLOG.md`'s C-306 entry | Live, one call, one comparison |
| AC-4 | Spot-check `claude/AGENTS.md`, `skills/cpg-analysis/SKILL.md`, and each of the six CPG-consuming agents' own prompts (`analyst`, `architect`, `qa-engineer`, `coder`, `tdd-engineer`, `frontend-engineer`) for `mcp__cypher__query`, per the requirements doc's own AC-4 wording | Static read, 8 files |
| AC-5 | `git diff` restricted to `skills/{cpg-analysis,joern-cpg}/**`, `claude/graph-dba/**`, and any `cpg_<component>` literal repo-wide — must be **empty** except for the two frontmatter/body lines step 2 explicitly touches in `cpg-analysis/SKILL.md` (which *are* tool-identity, not domain vocabulary, per §3.2 rule 5) | Static diff review |
| AC-6 | Live: `claude mcp list` / `/mcp` shows exactly one server, `cypher`, with tool count 1; a raw JSON-RPC `tools/call` probe (same recipe as `cpg/mcp/README.md`'s "Container debug recipe") naming server `cpg` or tool `mcp__cpg__query` fails to resolve — there is no `cpg` entry left in the protocol surface to call, not merely a discouraged one | Live, protocol-level probe |

**Regression floor (not a new AC, but must hold):** the offline suite count from the new
location matches the pre-rename count exactly (84 passed/7 deselected, per §2's baseline) — any
delta means the rename accidentally changed behavior, not just names.

---

## 6. Risks & open questions

- **The M6 numbering resolution (§3.6) is a call this plan makes on `generic-cypher-mcp2.md`'s
  behalf.** Confirmed sound by the plan-gate review (`docs/reviews/cpg-mcp-rename.md`): teco's
  coordination doc already establishes the ordering, no existing `C-6xx` entry collides (current
  ceiling `C-507`), and the bump is the only resolution avoiding a guaranteed `docs/BACKLOG.md`
  defect. One suggestion adopted from the review: step 3b's done-condition treats the
  `generic-cypher-mcp2.md` edit as *informational to `tico`* (a header-metadata bump, not a
  `Status:` flip, so it doesn't require `tico`'s hand under the letter of the convention, but a
  heads-up avoids surprise) — no plan text change needed beyond noting it here.
- **The "surgical substitution inside living logs" rule (§3.2, step 4) is an interpretive call,
  not stated in the requirements doc's decision log**, which lists "Open questions: (none)."
  The plausible alternative — never touch a dated `BACKLOG.md`/`HISTORY.md`/kaizen-history entry,
  treating it as historical narration exempt the same way an archived document is — was
  considered and rejected here because AC-1's wording ties the exemption to `Status: archived`
  specifically, and these documents never carry that header. **Confirmed sound by the plan-gate
  review**, which agreed with the rejection of the alternative for the same reason stated here.
  If the stakeholder later prefers the alternative anyway, the fix is narrow: exclude living
  logs from step 3a/3b's scope and note the exception in a future revision.
- **The `CPG_MCP_*`→`CYPHER_MCP_*` env-var rename (and the `cpg-mcp`→`cypher-mcp`
  Docker-image/label rename) is not individually named by any FR line** — justified in §2 from
  FR-5's "any embedded references" and the decision log's "Full identity, everywhere."
  **Confirmed sound by the plan-gate review**: these are "cpg" spelled into the tool's own
  implementation surface (`docker ps`, `docker image ls`, an operator's `.env` override), not
  CPG-domain vocabulary — leaving them stale would produce exactly the half-renamed state FR-6
  exists to prevent, just in a place FR-1…FR-3's literal text doesn't individually name. If a
  future reviewer still overrules it, `cpg/mcp/server.py`'s `_env_int()` calls and
  `docker-run.sh`'s `ENV_ARGS` loop simply keep the old env-var names — a small, isolated
  reversal that does not touch §3.1's directory-move decision or §3.4's `.mcp.json` change.
- **Closed in this revision (§7, m2 fix) — `docs/reviews/cpg-mcp-joern-agent-string-fix.md`.**
  Previously flagged here as an open risk on the theory that rewriting it would change what it
  claims a specific historical diff said. The plan-gate review read the file in full and
  confirmed all four of its pattern-matching hits (lines 1, 7, 8, 60) are plain path citations
  (`` `cpg/mcp/server.py` ``, `` `cpg/mcp/tests/test_server.py` ``) naming *where the fixed code
  lives*, not quoted identity-string literals under review — its actual quoted evidence concerns
  an unrelated `"joern agent"` → `"graph-dba agent"` phrasing fix. Updating a path citation to
  the file's new location is the same, unambiguous class of edit as any other doc's path
  reference in this sweep. **Resolved, not open:** approve the substitution in step 3b, no
  reviewer override needed.
- **No rollback path is designed beyond "this is one atomic change" (FR-6).** If step 1's
  in-container gate or a later acceptance check fails after `.mcp.json` already points at
  `cypher-mcp/docker-run.sh`, the fallback is the same one `cpg/mcp/README.md` already documents
  for any broken image/daemon: flip `.mcp.json` to the host-venv `run.sh` path. A full revert
  (git revert the whole rename) is always available but not specifically designed for here,
  since FR-6 explicitly rejects a transition period that would make a partial rollback safe to
  leave in place.
- **Relative-link breakage inside the moved `README.md` (§3.1) is easy to miss** because the
  affected strings (`../../docs/...`) don't contain "cpg" and so are invisible to the §3.2 grep
  pattern set. Called out as its own line item in step 1's done-condition rather than trusted to
  the general sweep.

---

## 7. Revision note (Pass 1 — addresses `analyst`'s plan-gate review)

2026-08-19 — `docs/reviews/cpg-mcp-rename.md` (verdict: needs changes) found one blocker, one
major, and two minors. All four addressed in place in this revision (`Version: 1.1`).

- **B1 (blocker) — fixed.** §3.2's discovery/proof `git grep` pattern gains a bare
  `` |\bcpg\b `` alternative (case-**sensitive**, deliberately diverging from the review's own
  literal case-insensitive suggestion — tested both live, documented in §3.2, chosen because
  case-sensitivity keeps the `` `CPG:` `` deliverable evidence-trail convention and "Code
  Property Graph (CPG)" acronym-definition prose out of the net for free, without missing any of
  the review's 6 confirmed misses). Rule 5 (§3.2) gains an explicit tool-identity sub-bullet for
  a bare standalone `` `cpg` `` token, plus two new CPG-domain (leave-alone) sub-categories the
  wider net surfaces at volume: the `` `CPG:` `` convention line and other pre-existing
  `cpg-`prefixed topic slugs/filenames unrelated to the MCP tool. §4's step-sizing note and §5's
  AC-1 check are updated to state the resulting wider triage volume (94→135 matching files)
  honestly rather than understate it, mirroring `generic-cypher-mcp.md`'s own precedent for the
  same shape of gap.
- **M1 (major) — fixed.** §3.2 step 2's exemption widens from a fixed 3-document list to a
  basename-prefix rule — any `docs/{requirements,plans,reviews,test-plans,test-reports}/` file
  whose basename starts with `cpg-mcp-rename` — closing the real 4th member the review found
  (`docs/reviews/cpg-mcp-rename.md` itself, produced before the docs sweep runs and, by design,
  full of old-name quotations naming its own findings). The proof-gate paragraph and §5's AC-1
  row are updated to reference the family rule instead of "the three self-referential documents."
- **m1 (minor) — fixed.** §3.2 step 3's classification no longer cites a nonexistent
  `docs/plans-coordination/` directory; corrected to state coordination docs live inside
  `docs/plans/` itself.
- **m2 (minor) — fixed.** §6's open item on `docs/reviews/cpg-mcp-joern-agent-string-fix.md` is
  closed outright (not carried forward) per the review's own finding that its four hits are path
  citations, not quoted identity-string content.
- The review's explicit views on the other two §6 open items (M6→M7 renumbering, the
  `CPG_MCP_*`/`cpg-mcp` env-var and Docker-namespace rename) — both confirmed sound, no plan
  change required — are folded into §6 as brief confirmations rather than left silently
  unaddressed.
