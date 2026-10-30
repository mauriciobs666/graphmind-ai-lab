# Connected code graph — evaluation protocol (P0(f))

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (M10) · **Extends:** `docs/plans/connected-code-graph-ml.md` · **Version:** 1.2 · **Reviews:** `docs/reviews/connected-code-graph-p0-impl.md` (§S06 (P0-05))

*Revision 2026-10-29 (v1.1): S06 gate (needs changes: B1, M1–M6, m1–m6, nits) addressed, with the stakeholder's two decisions of 2026-10-29 (AC-5 re-gate fresh-item floor; flow counts toward the 75 only across a unit boundary); dispositions in §16.*

*Revision 2026-10-29 (v1.2): S06 Pass 2 residuals P2-1 (served-anchor rule), P2-2 (H1 span test; §5.4 under `E`), P2-3 (trailing-punctuation boundary) and P2-4 nits addressed, each with S49 oracle cases; dispositions in §16, Pass 2.*

This is step S06 (unit P0-05) of the P0 plan. v1.0 cited `docs/plans/connected-code-graph-p0.md`.
That plan is now superseded by `docs/plans/connected-code-graph-p0-2.md`, whose §6 and §12 keep
the same numbering, and "plan §…" here means the successor.

## 0. Question, decision at stake, and the decisions in brief

**Question.** What is the frozen evaluation protocol that P1–P5 cite: how ground truth for AC-5's
P1 strata and AC-24's impact cells is written, matched, estimated for completeness, split and
frozen, and how a gate scores against it? Every procedure has to be precise enough for GC
(`qa-engineer`) to implement and run with no method choices of its own (review M5), and for me to
check its output.

**What depends on it.** The S06 row of plan §5.3 lists the consumers. S35 takes the planted minimums.
S40 and S48–S49 take the procedures. S52 takes the item schema, which it quotes to the writers.
S41–S47 take the counts, the matching rule, the estimators, the trigger and the
development-contact rule. S33 takes the hit rule. If any of these is wrong, the writers' output,
GC's tooling or the answer schema has to be redone.

**Binding constraints.**
- The held-out set is plaintext outside the repo, with a committed hash manifest (D11). The
  stakeholder rejected encryption, and this note adds none.
- Locators use the bundle spec's v1.3 forms (`docs/plans/connected-code-graph-bundle-spec.md`
  §3.2, §3.5, §4, §6) at `EVAL_COMMIT` (D8).
- The isolation protocol is the plan's §6. This note sets the method inside it and does not
  change the protocol.
- My write guard covers only `docs/`. The tools are GC's (S48, S49).

**Decisions (each is argued in its own section):**

| # | Decision | § |
|---|---|---|
| E1 | **Anonymous scopes (`<anon>#N`), `#k` duplicates and macro-generated names are never key anchors or gold anchors.** An impacted element inside one is recorded as its **nearest named ancestor**, with duplicate suffixes removed, plus the line(s). The hit rule credits a listed symbol whose `E` is that ancestor **and** whose span contains a recorded line (v1.2: the span test is required for every symbol, P2-2). Writers never use the ancestor-plus-line form, so the hand-counted-ordinal risk does not arise. | §2.3, §5 |
| E2 | **Five strata**, not four: `http`, `mcp`, `scripts`, `wiring`, `flow`. Wiring is SQ-P0-1's option (d), adopted up front, as the plan allows. An item's stratum is **derived** from its fields by a fixed rule, never taken from the writing session. | §3 |
| E3 | The flow stratum is a **shared seeded sample of 16 anchors** (8 Python, 8 TS/JS). Items are capped at **2 inter-function hops**, and an item whose chain only goes up through callers does not count as flow. | §3.6 |
| E4 | **Two-list estimator: Chapman.** With three or more lists, the lists are **collapsed by strategy** into two (the union of the provider-first sessions, and the consumer-first session) and Chapman is applied, alongside **Chao's heterogeneity lower bound** over the t lists. The report uses the larger of the two estimates. The interval is Chao's log-normal. | §6.4 |
| E5 | **Completeness threshold 0.95** per stratum, applied to the point estimate. Below it, one extra provider-first pass runs for that stratum, and at most one. | §6.5 |
| E6 | **Development-contact rule: forced to dev, per anchor.** Any anchor that the slice, S30's walkthroughs or a P0 golden test computed a link or an impact for goes to dev with all its items. Wrapper and chain hops are not contact. The rule takes precedence over every held-out minimum. | §8 |
| E7 | **Split by anchor cluster** (an anchor, plus a unit anchor's probe contract point), with seeded hash order and a 0.70 held-out share per stratum. **A unit is held out only while every contract anchor it serves is held out** (v1.2, the served-anchor rule, P2-1). If the **counting** held-out set (§3.7) is below 75 or a stratum is below 8, dev clusters are moved to held-out. If it is still short after that, SQ-P0-1 is triggered. Up to 35 counting items beyond gate 1's minimums go to a never-exposed **reserve** for re-gates. | §9 |
| E8 | **AC-24 at P1:** the impact use case × the two starting-point groups P1 can build (code location, contract point) × 2 languages × 9 questions. The split puts 6 per cell in held-out and the rest in dev, giving **24 held-out**. There are also 12 ambiguity candidates. The change and failure-evidence cells are written later, by the phase that builds them. | §10 |
| E9 | The hit rule is defined over the **canonical answer at the shipped default depth bound**, with every folded group expanded and an empty verdict log. It needs the element (E1's credit set, span test included), a status of confirmed or possible, and the element's **own** evidence chain from the anchor. The writer's chain is not compared (M4). | §5 |
| E10 | The **spot-check sample has 40 union items** (per stratum: 4 found by one writer, 4 found by both), plus 8 Portuguese gold items. | §7.3 |
| E11 | **A flow item counts toward AC-5's 75 only if its anchor and impacted element lie in different units or repos** (stakeholder, 2026-10-29). Every flow item is still tested and reported, and every held-out count is reported both with and without flow. | §3.7, §9 |
| E12 | **A re-gate pass counts only with a fresh-item floor** (stakeholder, 2026-10-29): at least **35** counting items on never-exposed anchors, all hits. Their Wilson lower bound is 0.901. Every re-gate also reports fresh-only recall and a cumulative first-exposure record. Kept items are a regression check, not recall evidence. | §12.1 |

**Constraints this places on P1, for `architect`.**
- A flow item can be 2 hops away, so the shipped default depth bound has to be at least 2. If it
  is lower, distance-2 flow items fail by construction. That is the correct measurement (ML m5),
  but P1 should know about it before choosing the bound (§15, Q1).
- AC-24 scores interpreted symbol anchors after the named-ancestor mapping (§10.4), so an
  interpreter that resolves a line to its lambda is not penalized (§15, Q1).

## 1. Inputs and findings

**Read for this note:**
- the plan (`docs/plans/connected-code-graph-p0.md` v1.3): §2 D8, D11; §3, rows S06, S14, S30,
  S33, S35, S40–S55; §4.3; §4.4, SL-1–SL-7; §5.3; §6; §12 SQ-P0-1;
- the ML note (`docs/plans/connected-code-graph-ml.md` v2), §4 and §8;
- the requirements (`docs/requirements/connected-code-graph.md`): AC-1–AC-7, AC-24, AC-25,
  FR-17, and the decision log;
- the bundle spec v1.3: §3 (hash, tuples, `repoId`, the §3.5 locators with the
  ancestor-plus-line form), §4 (`qname` grammar), §5.17–§5.18, §6, §11, §12 BF16, and §13;
- the proposal (`docs/plans/connected-code-graph.md`), §3.4 (binding, prompt mentions, "documentation
  snippets are not callers"), §3.6–§3.7 (BFS, distance, folding), §4 P0(f) and §5.2;
- the sibling S05 note (`docs/plans/connected-code-graph-p0-ml.md` v1.3), for its seeding and
  interval conventions, which this note reuses.

**Findings that shape the method.**

| # | Finding | Consequence here |
|---|---|---|
| G1 | The plan names the strata (HTTP, MCP, scripts, flow-shaped) but defines none of them. Under the obvious reading ("`cypher-mcp` ↔ hooks and scripts"), the scripts stratum has **one** anchor, the `query` tool. Split by anchor, it then lands wholly in held-out or wholly in dev. If S30 walks the `.mcp.json` → `cypher` launch chain, which is the proposal's own §3.4 example, the contact rule sends all of it to dev. | Scripts is defined by script entry points as well as the `cypher-mcp` tools (§3.4), which gives it many anchors. |
| G2 | One impact can be written as an HTTP item, an MCP item or a wiring item, depending on which session wrote it. If the stratum were part of the match key, KA's MCP session and KB's wiring session could find the same item and never match. | The stratum is derived from the item (§3.1) and is not part of the item id (§4.4). |
| G3 | Bundle spec §3.5 gives locators for code elements and contract points only. AC-2's wiring items are configuration entries. §3.2 has their tuple, `["configEntry", repoId, path, keyPath]`, and it can be written from source. | A third locator type, `config`, converts by the §3.2 tuple (§2.2). §3.5 should list it (§15, Q2). |
| G4 | The census needs a check, independent of the system, that a writer's `qname` exists at `EVAL_COMMIT`. Otherwise a writer's grammar slip shows up at the P1 gate as a "miss". | Validation runs from source at P0 (§7.1), and a rule covers ids still unresolved at the gate (§5.4). |
| G5 | 129 tracked files name `mcp__cypher__query`, 120 of them Markdown (plan §12). Most are documentation, which is not a caller (proposal §3.4). The bundle's `File.role = 'doc'` means "Markdown that is not a prompt-bearing file", but nothing lists the prompt-bearing files. | The census fixes the prompt-bearing set by glob in `eval-manifest.yaml` (§3.3). P1's extractor has to use the same set, because census scope must equal evaluated scope (plan §6.2). |
| G6 | Two LLM writers share blind spots. In an illustrative mixture (§6.4), Chapman reports completeness 0.975 when the true value is 0.860. A capture–recapture estimate is therefore optimistic exactly when it matters. | The threshold is set high (0.95). The estimate is labelled a lower bound on incompleteness. The runtime oracle is still the independent check (ML §4.1). |
| G7 | P1 builds only code-location and contract-point anchors (proposal §4, the P1 row). Change anchors arrive in P3 and failure-evidence anchors in P5. Of AC-24's four starting-point groups, only two can be scored at P1. | AC-24's P1 set covers those two groups (§10). |

## 2. Units, locators and anchor eligibility

### 2.1 Units

- **Item (AC-5).** An unordered fact `(anchor, impacted element)`: changing the anchor affects the
  element in at least one deployable configuration (ML §4.2's rubric; for indirect items, ML §8.1's,
  restated in §11.4). An item is **one element**, not one site. Two call sites of a route in one
  function are one item, and their lines go in `lines[]`.
- **Anchor.** What a person asks about: a contract point (a route or an MCP tool), a named symbol,
  or a script entry point (§3.4). For a server-level binding only, it is the provider unit (§3.5).
- **Impacted element.** A named symbol or a configuration entry (§2.2).
- **Split unit.** The **anchor**. Items that share an anchor are correlated, because one missing
  recognizer misses all of them (ML §4.1). They also leak together, because fixing one fixes all.
  So they never straddle the split (plan §6.4).
- **Analysis units at a gate.** Recall is reported over items, and again over **distinct anchors**,
  an anchor counting as a hit only if every one of its items is a hit. The anchor figure is the
  conservative one (ML §4.1).
- **Distance** counts inter-function hops only: call, link, shared state, config. Steps inside a
  function count 0 (proposal §3.6, decision Q-R7). A writer's chain lists the inter-function hops
  only, so `hops = len(chain)`.

### 2.2 Locator types (bundle spec §3.2, §3.5, §6)

Every locator is at `EVAL_COMMIT` and has `repoId = "graphmind-ai-lab"` where a `repoId` applies.

| `type` | Fields | Converts to | Source |
|---|---|---|---|
| `symbol` | `repoId, path, qname, signature?, line?` | `symbolId = h(["symbol", lang(path), repoId, qname(, signature)])` | §3.5, first row |
| `contract` | `kind, keyNorm`, with `kind` ∈ {`http`, `mcp_tool`} at P1 | `cpId = h(["contractPoint", kind, keyNorm])` | §3.5, second row, and §6 |
| `config` | `repoId, path, keyPath` | `entryId = h(["configEntry", repoId, path, keyPath])` | §3.2's `entryId` (config) tuple, and §5.17's `keyPath` grammar (G3) |
| `unit` | `repoId, rootPath, lang` | `unitId = h(["unit", repoId, rootPath, lang])` | §3.2's `unitId`. This type is used **only** as the anchor of a server-level wiring item (§3.5). |

- `h` is §3.1's hash, implemented once by `cpg/ccg/ids.py` (S09). GC's tools import it and never
  reimplement it.
- **A `config` locator is at the grain fixed per format by §2.4**, never at a writer-chosen
  level. GC's `normalize` truncates every `keyPath` to that grain mechanically (§13.4), and the hit
  rule accepts any entry under it (§5.2 H1). A Dockerfile line has no `keyPath` grammar in v1
  (§5.17 does not give one). Dockerfile items are therefore **out of census scope at P1** and are
  counted as excluded scope (§3.3).
- **A `unit` locator must be one of the units listed in `eval-manifest.yaml`** (§13.1, m3). It is
  never derived by a writer. S40 lists one unit per directory in census scope that holds a
  `pyproject.toml`, `package.json` or `requirements.txt`, with that manifest's language, plus the
  unit's served contract points where its registrations are grep-visible. For example,
  `cypher-mcp/`, which has only `requirements.txt`, is the unit `{rootPath: "cypher-mcp", lang:
  "python"}`. The brief gives writers this list.
- **`line` and `lines[]` are informative**, as in bundle spec §3.5. They are not identity. They do
  matter in one place: E1's credit rule (§5.2 H1).

### 2.3 The decision handed to S06: anonymous scopes as key or gold anchors

**Decision (E1).** No `<anon>#N` scope, `#k` duplicate or macro-generated name is ever:
- a key **anchor**;
- an AC-24 **gold anchor**;
- a member of the flow anchor frame (§3.6).

An **impacted element** inside one is written as its **eligible ancestor** `E(q)`, which is defined
mechanically:

```
E(q) = path "::" (the chain of q with every "#k" duplicate suffix removed, then truncated before its first "<anon>#N" segment)
       if that leaves an empty chain: path "::" <module> | <script> | <document>   (the file symbol, bundle §4.1 rule 4)
```

The writer records `E(q)` plus the line(s) of the site or sites. **v1.1:** a `#k` suffix is
*removed* rather than truncated at. That matches what a writer following Appendix B's "never
write `#2`" rule produces (`f` for a redefined `f#2`), so writers and the normalizer agree. The
span-contains-line test (§5.2 H1) then decides which definition gets credit. V2's `ast` set is
compared with the suffixes removed (§7.1). If a writer submits an
ineligible `qname` anyway, GC's normalizer rewrites it to `E(q)` and logs the rewrite (§13.4). If
the ancestor-plus-line form arrives, the tool converts it with bundle spec §3.5's converter and
then applies `E`. **Writers are never asked to count ordinals.**

**Why not make them eligible with the ancestor-plus-line form.**
1. **What is being measured.** AC-5 asks whether a real impact appears. A developer who is told
   "`ProductList`, line 42" is pointed at the same place as one told
   "`ProductList.<anon>#2`". The ordinal adds no evaluation information, only a way to be wrong.
2. **Writer error rate.** Hundreds of key items written by LLM sessions, each needing the
   introducing-token line and a column rank (`nth`), would produce mismatches between KA and KB
   that come from locator noise, not from disagreement about the impact. Every such mismatch
   either deflates the overlap `m`, which inflates N̂ and adds spurious extra passes, or becomes
   a near-miss pair to adjudicate (§6.3). The named-ancestor form is one that two independent
   readers write identically.
3. **Anchors are asked about by name.** Nobody asks "what if I change the second lambda in
   `timedSignal`?". FR-17's code-location start is a symbol name, or a file and line, and a file
   and line resolve to the enclosing named element. An anonymous anchor would test the
   interpreter's ordinal arithmetic, not understanding (AC-24).
4. **Stability.** Bundle spec BF16: inserting a sibling re-binds `<anon>#N` and `#k`. `EVAL_COMMIT`
   pins it for P0 (D8), but held-out items move forward to later gates after a fail (§12.1). There,
   a re-bound ordinal would silently point at a different element, while a named ancestor stays
   put.
5. **Macro-generated names** cannot be written from source at all (bundle spec §4.2, C/C++), so
   they are excluded outright, as S03 asked.

**What this costs, and how the cost is bounded.**
- **Granularity.** Two lambdas in one function, one of them impacted and one not, become a single
  item.
- **Spurious credit.** The system could list the *other* lambda for an unrelated reason and get the
  hit. E1's credit rule closes most of that: an anonymous descendant earns the hit only if its own
  span (owned regions included, bundle spec §4.1 rule 4) contains one of the item's recorded
  lines (§5.2 H1). What remains is a descendant that is listed for a different reason but whose
  span happens to contain the impacted line. That case is rare. It is also not a false hit in the
  user's terms, because the listed code is the code that breaks.
- **None on the key writers' protocol.** E1 makes their output simpler. S52's prompts quote one
  rule, "name the nearest enclosing named function, or the file if there is none, and give the
  line", instead of §3.5's converter conditions.

### 2.4 `config` grain, per format (M3, m3)

`normalize` matches each `keyPath` against its file's format row, top to bottom. The first
pattern that is a segment-prefix of the `keyPath` wins, and the `keyPath` is truncated to that
pattern's length. The **class** decides wiring's two shapes (§3.5): a `server` entry must have a
`unit` anchor, and a `contract` entry must have a contract anchor.

| Format (file test) | Pattern → grain | Class |
|---|---|---|
| `.mcp.json` | `mcpServers.<name>` | server |
| Kiro agent JSON (`**/.kiro/agents/*.json`) | `mcpServers.<name>` | server |
| | `tools[i]`, `allowedTools[i]`, `toolAliases.<key>`, `toolsSettings.<key>` | contract |
| | `includeMcpJson` (only when `true`: it binds the agent to every `.mcp.json` server, so one item per such server) | server |
| | any other key (v1.2, P2-4) | falls through to the "any other JSON" row |
| Claude settings (`**/.claude/settings*.json`) | `enabledMcpjsonServers[i]` | server |
| | `permissions.allow[i]`, `permissions.ask[i]`, `permissions.deny[i]` | contract |
| | `hooks.<Event>[i]` (the whole matcher group) | contract, or invokes a script (§3.4) |
| `mcp-monitor` TOML (`mcp-monitor/*.toml`) | `server.<name>` | server |
| | `watch[i]` | contract |
| Compose (`compose*.y*ml`, `docker-compose*.y*ml`) | `services.<svc>` | server |
| `.env`, `.env.*` | `<VAR>` (the whole variable) | server if its value is a URL whose port (explicit, or the scheme's default) is in some listed unit's `ports`, or an integer in some listed unit's `ports` (v1.2, P2-4); otherwise contract |
| any other JSON, YAML or TOML | the first two segments of the `keyPath`, an array index counting as a segment | contract |

- Each `<name>`, `<svc>` or `<key>` is one segment, and `[i]` is one segment.
- **Unit ports (P2-4).** Each listed unit in `eval-manifest.yaml` carries `ports`: the TCP ports it
  listens on, recorded by S40 from the unit's own source (a Compose service that builds it: the
  container side of `ports`; a Dockerfile `EXPOSE`; a literal `--port`/`port=` in its start
  script), possibly empty. It is frozen data, checked by the method owner at S40, never a GC
  judgment. At the working tree, `falkor-chat/server/.env.example`'s only port-like values are
  FalkorDB's, which is not a listed unit, so no `.env` entry is server class today.
- **A class mismatch the re-anchoring cannot fix** (a contract-class entry under a `unit` anchor)
  is invalid (§7.2).
- A writer's server-class entry under a contract anchor is **re-anchored** mechanically. It moves to
  the listed unit whose `rootPath` is the longest prefix of the writer's `provider.path`, with the
  original contract point added to `serves[]`. If `provider` is absent or no unit matches, the item
  is invalid (§7.2).
- Rows were checked against the working tree on 2026-10-29: `.mcp.json`;
  `kiro/.kiro/agents/falkor-chat-demo.json` (`mcpServers`, `tools`, `allowedTools`,
  `toolAliases`; `toolsSettings` is `{}` and `includeMcpJson` is `false`, so neither yields an item); `.claude/settings.json` (`enabledMcpjsonServers`, `permissions`);
  `mcp-monitor/config.example.toml` (`[server.*]`, `[[watch]]`); `falkor-chat/compose.yaml`;
  `falkor-chat/server/.env.example`. S40 re-checks them at `EVAL_COMMIT` and adds a row for any
  in-scope format the table does not cover. Adding a row is a protocol amendment, recorded in this
  note.

## 3. Strata and census scope

### 3.1 The derived-stratum rule (E2, G2)

GC computes an item's stratum from the item's own fields. The **first** rule that matches wins:

| Order | Rule | Stratum |
|---|---|---|
| 1 | The anchor is in the flow sample (§3.6) | `flow` |
| 2 | The impacted element is a `config` locator | `wiring` |
| 3 | The anchor is a script entry point (§3.4), or a `mcp_tool` contract point in `eval-manifest.yaml`'s `cypherMcpTools` list, or the impacted symbol's `lang` is `shell` | `scripts` |
| 4 | The anchor is a `contract` locator with `kind = http` | `http` |
| 5 | The anchor is a `contract` locator with `kind = mcp_tool` | `mcp` |
| — | None of the above (for example, a symbol anchor outside the flow sample) | **unclassified**: dropped, and counted in the report |

Each writing session is briefed on one stratum and searches it. Items a session writes that derive
to another stratum are kept, and they count for that stratum under the writer's strategy (§6.4).
A writer's list for stratum *s* is every item from any of its sessions that derives to *s*. The
writer's `stratum` field is recorded, and disagreement with the derived value is reported as a
count. It is never used for matching.

**Every input to the rule is a function of `itemId` alone** (m1):
- the anchor and impacted locator types;
- the flow-sample list;
- the `cypherMcpTools` list, which S40 takes from source (§13.1), never from a writer's `provider`;
- the impacted file's `lang`.

So all records of one `itemId` derive the same stratum. Flow **scope** (V4) is decided per item
after matching (§7.1).

### 3.2 `http`: storefront and other HTTP consumers (AC-1)

- **Anchor:** an `http` contract point, with `keyNorm` by bundle spec §6, mount prefix included.
  The writer also records `provider`: the handler's `symbol` locator.
- **Impacted element:** the element that **fixes the route**: the eligible element (§2.3) whose
  code supplies the path literal or constant reference from which the request's key is built, and
  which sends the request directly or through a wrapper. For SL-1 this is
  `salesperson/src/api/endpoints.ts::getHealth`. It is not `apiFetch`, `buildUrl` or the `fetch`
  call, which are chain hops (`LIFTED_VIA`), not items. A wrapper that receives the path as a
  parameter is an item only for a route whose path it builds itself.
- **In scope:** consumers in any language, apart from shell, which falls to `scripts` by rule 3.

### 3.3 Census scope, common to every stratum

Census scope is evaluated scope (plan §6.2). Every rule here goes into
`cpg/eval/eval-manifest.yaml` (S40) as data, and P1's gate builds with the same file.

- **Excluded paths** are plan §6.2's: `cpg/`, `deprecated/`, `docs/**/connected-code-graph*`,
  `docs/BACKLOG.md`, and every file the content check hits.
- **Test code is excluded as an anchor and as an impacted element** at P1: `**/tests/**`,
  `**/test_*.py`, `**/*_test.py`, `**/conftest.py`, `**/*.test.{ts,tsx,js,jsx}` and
  `**/*.spec.{ts,tsx,js,jsx}`. Server tests call nearly every route through a `TestClient`, so
  test items would swamp `http` with a single consumer pattern. Test impact also has its own gate
  (AC-14, P3), with its own ground truth. A test file that a non-test script *invokes*
  (for example `claude/teco/hooks/test-ensure-environment.sh`, invoked by a hook) is not test code
  for this rule, because it is a script entry point.
- **Prompt-bearing Markdown** (G5) is the following set. Every other `.md` file is documentation
  and never a consumer.
  - `**/AGENTS.md` and `**/CLAUDE.md`;
  - `claude/**/*.md`, except `claude/**/kaizen/**` and `claude/README.md`;
  - `skills/**/*.md`, except `skills/README.md`;
  - `opencode/agents/**/*.md`, `opencode/skills/**/*.md` and `.kiro/steering/*.md`;
  - `.claude/commands/**/*.md` and `.claude/agents/**/*.md`.

  P1's prompt extractor must classify `File.role = 'doc'` by this same list (§15, Q3).
- **Out of census scope at P1, and counted as excluded:**
  - Dockerfile lines (§2.2);
  - vendored paths (bundle spec §5.0);
  - impacts whose only connection is runtime-built, with no source-visible key. These are AC-6
    material, and they are fixture-planted (§12.2);
  - flow chains longer than 2 hops (§3.6).

### 3.4 `scripts`: script and hook invocation (AC-4)

- **Anchors** are of two kinds:
  - **script entry points:** the `<script>` symbol of a `shell` file, or the `<module>` of a
    Python file that has an `if __name__ == "__main__":` guard or a shebang;
  - the **`cypher-mcp` tools:** the `mcp_tool` contract points in `eval-manifest.yaml`'s
    `cypherMcpTools` list. S40 takes the list from the tool registrations in `cypher-mcp/server.py`
    (§13.1). At the time of writing that is the one tool `query`.
- **Impacted elements:** every element that runs the script by path (a shell command, a Python
  `subprocess` call with a literal argv, a prompt-bearing document that tells the agent to run it),
  and every shell symbol that names a `cypher-mcp` tool. Under rule 3 a shell consumer of *any*
  tool or route falls here.
- **Why the stratum is this broad (G1).** "Hooks and scripts that call `query`" gives one anchor.
  Script invocation gives many, so a split by anchor can work. `git ls-files` counts 32 `.sh`/`.py`
  files under `claude/`, 7 under `skills/joern-cpg/scripts/`, and 47 further `.sh` files outside
  `claude/`, `skills/`, `cpg/` and `deprecated/` (counted 2026-10-29; S40 recounts at
  `EVAL_COMMIT`). Not every one of these is invoked. AC-4's own case, a `cypher-mcp` tool and
  its script and hook callers, is a subset. The config entries that invoke scripts (`settings.json` hooks) are `wiring`, by rule 2.

### 3.5 `wiring`: configuration that names, binds or launches (AC-2; SQ-P0-1 option (d))

There are two shapes, distinguished by the anchor:
- **Per-contract config.** The anchor is a contract point. The impacted element is a
  contract-class `config` entry (§2.4) that names **that** tool or route specifically, such as an
  allow-list element (`allowedTools[i]`), an `mcp-monitor` watch on the tool (`watch[i]`), a
  permission rule (`permissions.allow[i]`), or a proxy entry for the route.
- **Server-level binding.** The anchor is the provider **unit**: one of `eval-manifest.yaml`'s
  listed units (§2.2), giving `unitId = h(["unit", repoId, rootPath, lang])` (bundle spec §3.2).
  The impacted element is a server-class `config` entry (§2.4) that launches the server or points
  a client at it: `mcpServers.<name>` in `.mcp.json` or a Kiro agent, `server.<name>` in
  `mcp-monitor`, `services.<svc>` in Compose, or a `.env` URL or port variable. The writer also
  records `serves[]`: at least one contract point the server provides. The probe is chosen from
  the unit's served set (§5.1), and the probe is **clustered with the unit for the split** (§9.1, B1).
  - **The served-anchor rule (v1.2, P2-1).** AC-2 has every tool's answer list the wiring that
    connects it, so the answer for **any** contract point the server provides shows the server's
    binding entries. A binding entry is therefore exposed by every served anchor, not only the
    probe. A unit is held out only while **every** anchor in its served set S(u) is held out
    (§9.1 steps 3b and 4; §12.1 steps 3 and 4). S(u) is the contract anchors in U that the unit
    serves: any member record's `provider.path` has u as its unit (the longest-prefix rule of
    §3.7), or the anchor is in u's `serves[]`. All contract kinds count, because whether P1 lists
    server bindings under a route's answer is P1's choice, not yet made.
  - The rule is **one-way**. A unit's items are binding entries, which are not items of any tool,
    so a unit in dev exposes no held-out tool item. Its probe, the one tool asked on its behalf,
    shares its side through the cluster.
  - **A binding entry is written once per server, never once per tool.** If it were written per
    tool, one entry would sit under some twenty `falkor-chat` tool anchors. It would straddle the
    split, and fixing it on a dev tool would make every held-out tool pass. That is a leak through
    a shared impacted element that a split by anchor alone does not see.
  - If a writer writes the per-tool form anyway, GC re-anchors it to the server's unit with that
    tool in `serves[]`, deduplicates, and logs the rewrite (§13.4).

### 3.6 `flow`: shared seeded anchors (E3, finding F6)

- **Frame.** Every eligible (§2.3) named function or method **with a body** whose `lang` is
  `python` or `javascript`, in census scope (§3.3), outside test code, and with a body of at least
  3 source lines (`lineEnd − lineStart ≥ 2`). The frame is enumerated from the kit at
  `EVAL_COMMIT` by a source parser: `ast` for Python, and py-tree-sitter for TS/JS. py-tree-sitter
  is the S05 frame's parser and is installed from `cpg/pyproject.toml`'s `spike` extra. It is
  **not** in the S02 toolchain image, whose Dockerfile has no tree-sitter (m5). Joern and the
  system are never used for the frame.
- **Draw.** 8 Python and 8 TS/JS anchors, by S05's hash-sort convention:
  `seed = "ccg-p0f-flow-v1|" + lang + "|" + EVAL_COMMIT`, ordered by `SHA-256(seed + "|" + qname)`
  ascending, taking the first 8 per language. Contacted anchors (§8) are **not** removed from the
  frame. If one is drawn, it yields dev items. Removing them would make the draw depend on S14
  and S30 finishing before S40.
- **Items.** For each sample anchor *f*: every eligible element *g ≠ f*, in census scope, that a
  change in *f* affects through a chain of **at most 2** inter-function hops, where the chain
  contains at least one **flow hop**: `return-use`, `callee-arg`, `control` or `shared-state`
  (§4.1). A chain made only of `caller` hops is call reach, the AC-1 shape, and is not a flow item.
- **Why a cap of 2.** AC-25's own example is "a changed return value used two calls away". A
  distance-3 census around a utility function runs to hundreds of elements, which no pair of
  writers enumerates the same way, so completeness would be unmeasurable. The cap is a census
  scope rule. It is not a claim that nothing exists at distance 3.
- **Why 16.** Each anchor contributes a correlated cluster. Sixteen anchors, split 0.70, leave
  about 11 held-out anchor clusters (§9), which comfortably meets the ≥ 8 items per stratum, and
  the KA/KB sessions stay census-length (plan §6.6). Each language has 8 anchors, so AC-25's shape
  is present in both P1 languages.
- **Flow and AC-5's 75:** see §3.7. Most flow items are expected to stay inside one unit, so they
  will not count toward the 75.

### 3.7 What counts as cross-boundary, toward AC-5's 75 (M6; stakeholder decision 2026-10-29)

AC-5 counts "known cross-boundary impacts". An item is **counting** when it crosses a boundary by
the rule below. Every item, counting or not, is still tested, scored and reported.

| Stratum | Counting when | Why |
|---|---|---|
| `http`, `mcp` | always | the item crosses a contract boundary (the `link` hop), whatever units hold its two ends |
| `scripts` | always | the item crosses a process boundary: one program invokes another, or names a tool another process serves |
| `wiring` | always | the item crosses a config-to-code boundary |
| `flow` | **only if** the anchor's unit and the impacted element's unit differ, or their `repoId`s differ | the stakeholder's decision. Units are `eval-manifest.yaml`'s list (§2.2), and a path's unit is the listed unit whose `rootPath` is its longest prefix. |

- The flow test uses the item's two endpoints, not the writer's chain. That makes it a function of
  `itemId` alone, and it is mechanical. A 2-hop chain that leaves a unit and comes back to it does
  not count. With the 2-hop cap that case is negligible, and it errs toward not counting.
- **Reporting.** Every held-out count, in S47's report and at every gate, is given twice: **all
  items** (`H_all`) and **counting items** (`H`). The 75 is tested on `H`. The per-stratum ≥ 8 is
  tested on all items, so `flow` keeps its 8 even when none of them count. Flow still has to be
  present and gated per stratum (ML §4.1).

## 4. Item schemas

This section is the method's reference for GC and reviewers. **Writers never see it.** S40's
`BRIEF.md`s and S52's agent definitions paste **Appendix B verbatim, and nothing else from this
note** (M3). Appendix B restates §2.2–§2.4, §3 and §4 in writer-facing form, with no JSON example,
no SL-1 element, no rationale, and no mention of contact, splits, sampling or held-out. The JSON
examples below are for this note's readers only.
Each output file is JSONL with one object per line, appended as the writer goes (plan §6.1). A
reader drops an unparseable last line (plan §6.1, step 3). Field names are camelCase to match the
bundle spec.

### 4.1 Key item (`KA.jsonl`, `KB.jsonl`; the extra pass writes `KA.jsonl` in its own role root)

```json
{
  "schema": "ccg-key-v1",
  "seq": 17,
  "stratum": "http",
  "anchor":   {"type": "contract", "kind": "http", "keyNorm": "GET /shop/api/health"},
  "provider": {"type": "symbol", "repoId": "graphmind-ai-lab",
               "path": "falkor-chat/server/falkorchat/storefront_api.py",
               "qname": "falkor-chat/server/falkorchat/storefront_api.py::build_storefront_router.health", "line": 946},
  "serves":   [],
  "impacted": {"type": "symbol", "repoId": "graphmind-ai-lab",
               "path": "salesperson/src/api/endpoints.ts",
               "qname": "salesperson/src/api/endpoints.ts::getHealth"},
  "lines": [17],
  "chain": [
    {"kind": "link", "from": "falkor-chat/server/falkorchat/storefront_api.py:946", "to": "salesperson/src/api/endpoints.ts:17"}
  ],
  "hops": 1,
  "why": "getHealth calls apiFetch with path `${API_PREFIX}/health`, method GET; the server mounts this router at /shop/api",
  "deployment": "default"
}
```

The example is SL-1's real item, as of the working tree on 2026-10-29: `getHealth` at
`endpoints.ts:16–19`, and the `apiFetch` call at line 17. It is a contacted anchor (§8), so it
lands in dev in any case.

| Field | Req | Rule |
|---|---|---|
| `schema` | yes | the literal `ccg-key-v1` |
| `seq` | yes | the writer's running number, from 1. It is unique per output file. |
| `stratum` | yes | the session's stratum. Recorded but not trusted (§3.1). |
| `anchor` | yes | a `contract`, `symbol` or `unit` locator (§2.2), eligible (§2.3). A `unit` anchor is allowed only with a `config` `impacted` (§3.5). |
| `provider` | when `anchor` is a contract | the handler's or registration's `symbol` locator. Used for near-miss detection (§6.3), never for identity. |
| `serves` | when `anchor` is a `unit` | ≥ 1 `contract` locator served by that unit (§3.5); otherwise `[]` |
| `impacted` | yes | a `symbol` locator (eligible, §2.3) or a `config` locator (§2.2) |
| `lines` | yes | ≥ 1 one-based line in `impacted.path`: the site(s), mention(s) or entry. It must be inside the element's span. |
| `chain` | yes | the writer's ordered **inter-function** hops from anchor to impacted (§2.1). Each hop is `{kind, from, to}` with `from` and `to` as `path:line`. `kind` is one of `caller`, `return-use`, `callee-arg`, `control`, `shared-state`, `link`, `config`. **A prior only** (M4): it is reported as a distribution and never compared with the system's chain. |
| `hops` | yes | `len(chain)`. For `flow`, at most 2 (§3.6). |
| `why` | yes | at most 300 characters, in plain words |
| `deployment` | yes | `"default"`, or the name of the deployable configuration in which the dependency is real (ML §4.2's rubric): a Compose file, an env profile, a `.mcp.json` variant |

**Hop kinds** (they define flow-shapedness, §3.6):
- `caller`: the next element calls the current one.
- `return-use`: the next element calls the current one and uses its return value or an
  out-parameter.
- `callee-arg`: the current element calls the next one with an argument whose value depends on the
  change.
- `control`: a condition that depends on the change guards whether the next element's code runs.
- `shared-state`: one side writes a field, a global, a store key or a file that the other side
  reads.
- `link` and `config`: a cross-boundary contract or binding hop.

### 4.2 Question item (`QW.jsonl`, AC-24)

```json
{
  "schema": "ccg-question-v1",
  "seq": 3,
  "lang": "pt",
  "group": "contract-point",
  "intent": "unambiguous",
  "ambiguityKind": null,
  "text": "o que quebra no storefront se eu mudar o /shop/api/products?",
  "gold": {"useCase": "impact", "direction": "impact",
           "anchors": [{"type": "contract", "kind": "http", "keyNorm": "GET /shop/api/products"}],
           "scope": null},
  "goldOptions": null
}
```

The example is illustrative. `GET /shop/api/products` is not asserted to exist.

| Field | Req | Rule |
|---|---|---|
| `lang` | yes | `en` or `pt`. Written natively, never translated (ML §8.3). |
| `group` | yes | FR-17's starting-point group: `code-location` or `contract-point` at P1 (§10) |
| `intent` | yes | `unambiguous` or `ambiguous`: the writer's intent. The annotators decide the final label (§10.3). |
| `ambiguityKind` | iff `intent = ambiguous` | `several-symbols` (a name that resolves to several eligible symbols), `unknown-name`, `two-use-cases` (impact vs depends-on), or `no-anchor` |
| `text` | yes | the question, as a developer types it: partial names, wrong case, pasted fragments and code-switching are expected |
| `gold` | iff unambiguous | `useCase` ∈ {`impact`}; `direction` ∈ {`impact`, `depends-on`}; `anchors` is a non-empty list of eligible `symbol`, `contract` or `config` locators (E1); `scope` is `null` (whole system) or `{"repoIds": […]}` / `{"paths": […]}` when the question restricts it |
| `goldOptions` | iff ambiguous | ≥ 2 objects of `gold`'s form, one per legitimate reading. For `unknown-name` and `no-anchor`, this is `[]`, and the correct behaviour is to ask. |

### 4.3 Annotation item (`AN1.jsonl`, `AN2.jsonl`)

`{"schema": "ccg-annotation-v1", "aid": "<12 hex>", "label": "ambiguous" | "unambiguous", "reason": "<≤ 200 chars>"}`

Here `aid` is the opaque id from `QW-nogold.jsonl` (§13.2). Annotators never see `qid`, `seq`,
`intent`, `group` or any gold field.

### 4.4 GC's normalized record (`work/`, never shown to writers)

For each validated key item, GC computes:
- `anchorId`: the `cpId`, `symbolId` or `unitId` of the normalized anchor;
- `impactedId`: the `symbolId` of `E(impacted)` (§2.3), or the `entryId`;
- **`itemId = h(["evalItem", anchorId, impactedId])`**. The stratum is not in it (G2), and
  neither are the lines;
- `derivedStratum` (§3.1), `writer` (`KA`, `KB`, `KA2`), `strategy` (`provider-first` or
  `consumer-first`), and the original object.

Within one writer, two records with the same `itemId` merge into one, with the union of their
`lines` and the first `chain` by `seq`. Across writers, `itemId` equality is the mechanical match
(§6.2).

## 5. The hit rule (AC-5)

### 5.1 The question asked per held-out anchor

- **One impact question per held-out anchor** answers all of that anchor's items. It is a
  structured request, sent through the CLI or the MCP `ask` tool, **never** through plain
  language. AC-5 measures mapping, and interpretation is AC-24's job.
- **Mode:**
  - a `contract` anchor is asked in contract mode, from its contract point;
  - a `symbol` anchor (flow, script entry) is asked in symbol mode;
  - a `unit` anchor is asked from its **probe**: of the contract points in the unit's served set
    S(u) or its `serves[]` (§3.5), after the union over writers, the one with the **fewest items in
    U** (zero for a point with no items), ties by smallest `cpId` (hex order). The probe is fixed
    at S47 **before** the split, and it is clustered with the unit, so both always sit on the same
    side (§9.1, B1). **v1.2:** "fewest items" replaces "smallest `cpId`" so that, when the
    served-anchor rule sends a unit to dev, its cluster takes the fewest tool items with it (P2-1).
- **Direction:** `impact`. **Depth bound:** the shipped default, recorded in the gate's
  pre-registration (§12.3) before the run (ML m5).
- **Snapshot:** built from a worktree at `EVAL_COMMIT` with `eval-manifest.yaml`'s exclusions
  (D8), and an **empty verdict log**. Any dismissal on the gate snapshot would be a leak, and GC
  checks that `stamp.verdictLogOffset = 0`.
- **Principal:** one that can read every repository, so no item is redacted (proposal §3.10).

### 5.2 When an item is a hit

The answer is evaluated as the **canonical answer JSON** (S33), never as the rendered text, with
**every folded group expanded**. S33 guarantees that each group enumerates every member with its
own chain. Held-out item *i* = (anchor *a*, impacted *e*, `lines` *L*) is a **hit** if and only if
the answer contains an item or group member *x* for which all four conditions hold:

| | Condition | Mechanical test (GC's gate harness at P1) |
|---|---|---|
| **H1** | *x* is *e*, under E1's credit set | **symbol** *e*: *x* is a symbol with `E(qname(x)) = qname(e)` (§2.3) **and** *x*'s span, or one of its owned regions, contains some ℓ ∈ *L*. Both are required; there is no id-only branch (v1.2, P2-2). **config** *e*: *x* is a `ConfigEntry` in `e.path` whose `keyPath` equals `e.keyPath`, or begins with `e.keyPath + "."` or `e.keyPath + "["` |
| **H2** | *x* is listed as `confirmed` or `possible` | the `status` field. The empty verdict log means nothing is hidden. |
| **H3** | *x* carries **its own** evidence chain from the asked anchor | *x*'s `chain` is non-empty, its first hop starts at the asked anchor (the contract point, or the symbol), and its last hop ends at *x* |
| **H4** | *x* is enumerated individually | *x* is an item or a group **member**. A group's header or count is never *x*. |

**What does not matter:**
- the distance the system reports;
- direct vs indirect;
- whether the system's chain equals the writer's (M4);
- which of several qualifying *x* is found;
- whether *x* is folded in rendering. Folding is display only (proposal §3.7, P3-1).

**What does not rescue a miss:**
- a "reach stopped at distance *d* with *k* unexplored" note (ML m5). It is reported next to the
  miss, as the likely cause, and the item still counts as missed;
- an item for the right file but the wrong element;
- *e*'s unit being `not-mapped` or `stale-kept` without *e*.

**Why H1 has no id-only branch (P2-2).** Under `E`, a `#k` duplicate maps onto its first
definition: `E(C.x#2) = C.x`, the getter's own id when `C.x#2` is the setter. An id-only branch
would credit the getter for an impact on the setter. With the span test always required, *x* = *e*
itself still qualifies whenever *L* lies in *e*'s span, which §4.1 requires and V2 checks for
Python. A recorded line outside every qualifying span (possible in TS/JS, which get V1 only) is
caught by §5.4's span case, not scored as a silent miss. Exposure today is nil: an `ast` scan of
census-scope non-test Python (86 files, 1,756 scopes, definitions reached through control-flow
blocks included) finds no same-scope duplicate name, and `git grep` finds no `@….setter` and no
`@overload`. The rule is pinned for when one appears.

**Why this shape.** H1 is the anchor-eligibility decision (E1), made testable. H3 and H4 are
AC-5's "individually, with its evidence". Requiring the chain to agree with the writer's would
score the writer's path-finding, not the system's: two correct chains often differ (proposal
§3.6: the best predecessor is chosen by fewest *possible* hops). It would also make S41–S47's key
writers produce chains to a precision they cannot reach. **The hit rule does not change the
key-writer protocol.** Writers produce what plan §6 and §4.1 already ask for.

### 5.3 How the gate reports

All results are first run, never a post-fix rerun (ML §4.1):
- **Item recall**, per stratum and pooled, as `k/n` with a Wilson 95% interval.
- **Anchor recall**, per stratum and pooled, over distinct anchors. An anchor counts only if all of
  its items are hits. This is the conservative figure.
- **The pass condition is AC-5's:** every held-out item is a hit, over at least 75 **counting**
  items (§3.7) pooled and at least 8 items per stratum (§9). The report gives `H` and `H_all`.
  Computed: 75/75 has a Wilson lower bound of **0.951**, and 8/8 has **0.676** per stratum.
  **The 0.951 is claimed only for a first gate.** A re-gate reports under §12.1's rules. Per-stratum claims at this size are smoke tests (ML §4.1). The P(pass)
  table in ML §8.2 still applies: a true recall of 0.99 passes 75 first-run items with
  probability **0.471** (computed, 0.99⁷⁵), which is why §12.1's fail procedure exists.
- **Each miss** is listed for the triage (§12.1). Per miss, the report gives the depth-bound flag
  and whether *e* is absent from the snapshot's symbol table altogether (an extraction miss). It
  gives no content in any document that builders read (§12.1).
- **Descriptive only:** the writers' `hops` distribution next to the system's reported distance on
  the hits (M4).

### 5.4 A key id that does not resolve in the gate snapshot

Everything here runs under `E`, exactly as H1 does (v1.2, P2-2).

**Trigger.** No snapshot symbol *s* has `E(qname(s)) = qname(e)` with a span or owned region
containing some ℓ ∈ *L*. Then either the system failed to extract *e* (a miss) or the writer's
`qname` or lines are wrong (a key error). The trigger never fires merely because the snapshot
holds `C.x#2.check` where the key holds its `E`, `C.x.check`: that symbol qualifies.

**Source comparison.** The two cases are told apart by **source alone**: the method owner derives,
from the file at `EVAL_COMMIT` by bundle spec §4, the grammar `qname` *q* of the element enclosing
each ℓ ∈ *L*. Neither the system's symbol table nor its answer is consulted.
- If `E(q) = qname(e)` for some ℓ, the key agrees with source, and the item is a **miss**.
- Otherwise the key is corrected from source: to `E(q)` if the lines are right and the name is
  wrong, or to lines inside *e*'s source span if the name is right and the lines are wrong. A
  correction is always an eligible locator, never a `#k` or `<anon>` one. It is logged as "locator
  corrected at gate, from source", and the item is scored under the correction **in the same first
  run**.

This keeps P0's validation (§7.1) cheap where a parser is unavailable, without letting the system
grade its own ground truth. The trigger (no qualifying symbol) does depend on the system. The
correction does not, so it adds no bias toward what the system finds.

## 6. Matching, capture–recapture and the completeness trigger

### 6.1 Inputs, in order

1. Only sessions whose audit is clean, or whose only findings are refused attempts (plan §6.5,
   check 3), contribute items. A voided session's items are discarded **before** matching, and
   its replacement session's items are used instead.
2. Items are normalized (§13.4) and validated (§7.1). An invalid item is held out of matching and
   estimation until the method owner fixes it from source or drops it (§7.2).
3. Items are grouped per writer into records by `itemId` (§4.4), and their stratum is derived
   (§3.1).
4. After matching (§6.2–§6.3), flow scope (V4) is decided **per item**, never per record. That
   way a single writer's chain cannot split one `itemId` into in-scope and out-of-scope (m1).

### 6.2 Mechanical matching

Two records from different writers **match** when their `itemId`s are equal. Nothing else matches
mechanically. In particular, a different line never prevents a match, because lines are not
identity.

### 6.3 Near-misses and pair-id adjudication

**Candidate pairs.** Within each derived stratum, after mechanical matching, GC forms a pair
(*x*, *y*) from each provider-first record *x* and each consumer-first record *y* that are both
still unmatched, when either rule below holds.

| Rule | Condition | Typical cause |
|---|---|---|
| **N1** | same `anchorId`; `impacted` in the same `path`; different `impactedId` | different named ancestor chosen (a nested function vs its parent), or a `qname` slip; or, for `config`, two `keyPath`s of which one is a segment-prefix of the other |
| **N2** | same `impactedId`; different `anchorId`; and either both anchors are `contract` locators of one `kind` with equal `provider` ids, or their `keyNorm`s are equal after casefolding, dropping a trailing `/`, and replacing every `{…}` segment with `{}` | `keyNorm` normalization slips (a mount prefix, a parameter syntax) |
| **N3** | same `impactedId`; both anchors are `unit` locators; different `anchorId` | two writers credit one server binding to different listed units, for example a parent and a child directory (m3) |

- **Pair id:** `pairId = SHA-256("ccg-p0f-pair-v1|" + itemId(x) + "|" + itemId(y))`, first 12 hex.
- **The file:** GC writes `work/nearmiss.jsonl`, one line per pair:
  `{pairId, rule, stratum, x: <normalized record>, y: <normalized record>}`, sorted by `pairId`.
- **Adjudication:** the method owner reads the file in place and returns, to GC only, a list of
  `{pairId, verdict: "same" | "different", keep: "x" | "y"}`. `keep` names the locator that
  survives in a `same` pair, chosen as the one that is correct by source. No item content leaves
  `work/` or passes through `teco` (plan §6.4).
- **Consistency:** GC rejects a verdict list in which one record is `same` in two pairs, and sends
  it back. A `same` pair becomes one matched record. GC logs every application in
  `work/adjudication.jsonl` as `{pairId, verdict, keep}`.
- **Unadjudicated pairs** count as `different`. That is conservative, because it lowers `m` and
  raises N̂.
- **Only pairs across strategies are adjudicated.** A near-miss between KA and KA2, both
  provider-first, stays two records. That can only raise `f₁` and lower `f₂` in Chao's bound
  (§6.4), so it can only raise N̂. The error runs in the conservative direction.

### 6.4 Estimators (E4)

Per derived stratum, the writers are grouped into **lists by strategy** (plan §6.1, ML §8.2): list
P = the union of provider-first writers' records (KA, plus the extra pass KA2 if one ran), and
list C = the consumer-first writer's (KB). Then:
- `n₁ = |P|`, `n₂ = |C|`, `m = |P ∩ C|` (after §6.2–§6.3), `r = |P ∪ C| = n₁ + n₂ − m`;
- **Chapman:** `N̂c = (n₁+1)(n₂+1)/(m+1) − 1`, with
  `Var = (n₁+1)(n₂+1)(n₁−m)(n₂−m) / ((m+1)²(m+2))`;
- **Chao's heterogeneity lower bound** over the *t* individual writing sessions (t = 2, or 3 with
  KA2): from the capture histories, `f₁` and `f₂` are the counts of records found by exactly one
  and exactly two sessions, and `N̂h = r + ((t−1)/t) · f₁(f₁−1) / (2(f₂+1))`;
- **`N̂* = max(N̂c, N̂h)`**, and **completeness = r / N̂***;
- **The Chapman interval** (Chao's log-normal, on `f₀ = N̂c − r`): `K = exp(1.96·√ln(1 + Var/f₀²))`,
  giving N in `[r + f₀/K, r + f₀·K]`. If `f₀ ≤ 0` or `Var = 0` (one list contains the other), the
  interval is `[r, r]`, flagged "degenerate", and is not evidence of completeness.
  - **It is labelled "Chapman interval"** and reported next to N̂c (m2). When Chao binds
    (`N̂h > N̂c`), the report says so, and no interval is given around N̂*. Chao's bias-corrected
    variance needs a delta-method derivation, which GC would have to choose and I have not
    specified, so no interval is better than a mislabelled one.
  - **Decisions use the point N̂* only** (§6.5). No interval enters a rule.
- **Pooled:** `Σr / ΣN̂*` over strata, stratified. **No pooled interval is reported.** Chapman is
  not applied to pooled lists, because capture probabilities differ by stratum, and pooling
  heterogeneous strata biases the estimate low.

**Why Chapman, and not Lincoln–Petersen, for two lists.** Lincoln–Petersen (`n₁n₂/m`) is
undefined at `m = 0` and biased upward at small `m`. Chapman is nearly unbiased when
`n₁ + n₂ ≥ N`, which holds for any census worth the name, and it is defined everywhere. Both
assume independent lists with homogeneous capture probabilities.

**Why the multi-list estimator is this one (m7).** A third session breaks Lincoln–Petersen's
two-list form, and that session is provider-first again. So it is positively dependent on KA: the
same strategy finds the same things. A naïve three-list independence model (Darroch's Mt) would
read that dependence as completeness and under-estimate N.
- **Collapsing the lists by strategy** keeps the one independence argument the design rests on,
  provider-first against consumer-first (ML §8.2), and folds the dependent pair into one list.
- **Chao's bound** uses the t-session frequency counts and remains a valid lower bound when items
  differ in how hard they are to find. That heterogeneity is the shared-blind-spot problem (G6).
- **Taking the maximum** reports the more pessimistic completeness, which is the conservative
  side for a never-miss claim.
- A log-linear model with interaction terms was rejected. It needs a model-selection step, which
  is a method choice GC would have to make, and three lists over 20–60 items cannot support one.

**Computed examples** (Appendix A):

| n₁ | n₂ | m | r | Chapman N̂c [log-normal 95%] | Chao N̂h | completeness r/N̂* |
|---|---|---|---|---|---|---|
| 40 | 35 | 28 | 47 | 49.9 [47.8, 58.0] | 49.9 | 0.941 |
| 30 | 25 | 15 | 40 | 49.4 [43.4, 66.2] | 49.4 | 0.810 |
| 12 | 10 | 8 | 14 | 14.9 [14.1, 20.5] | 14.8 | 0.940 |
| 10 | 9 | 3 | 16 | 26.5 [18.9, 53.5] | 25.8 | 0.604 |

The three-list case is also computed in Appendix A. With histories (KA, KA2, KB) counted as
111:20, 110:6, 101:4, 011:3, 100:5, 010:3, 001:4, we have r = 45. The collapsed lists give
n₁ = 41, n₂ = 31, m = 27, and Chapman 47.0. With f₁ = 12, f₂ = 13 and t = 3, Chao gives **48.1**.
So N̂* = 48.1 and completeness is **0.935**: Chao is the binding estimate here.

**What the estimate cannot see (G6).** In an illustrative mixture computed at expected counts
(Appendix A; not a sample), 48 easy items are found with probabilities (0.90, 0.85) and 12 hard
ones with (0.20, 0.20). The true completeness is **0.860**, but Chapman and Chao both report
**0.975**, because both writers miss the same hard items. So the report labels completeness as an
**upper bound on key completeness**, equivalently a lower bound on incompleteness (ML §8.2), and
keeps the runtime oracle as the independent check (ML §4.1).

### 6.5 The completeness trigger (E5)

- **Rule:** if a stratum's completeness `r / N̂*` is below **0.95**, `teco` launches **one** extra
  provider-first session for that stratum (KA2, a fresh session, plan §6.4), and S45 re-runs
  §6.1–§6.4 with t = 3. **At most one extra pass per stratum.** If completeness is still below
  0.95 after it, the census proceeds. The stratum is flagged "completeness below threshold" in the
  report, and its AC-5 results carry that caveat at every gate.
- **Why 0.95 and not 0.90.** The estimate is optimistic exactly in the shared-blind-spot case (§6.4),
  so the threshold has to sit high to mean anything. Its operating characteristics, computed
  exactly over every capture outcome under independent, homogeneous capture (Appendix A):

  | True N | Writer capture probabilities | True completeness | P(trigger) at 0.95 | P(trigger) at 0.90 |
  |---|---|---|---|---|
  | 20 | 0.95, 0.90 | 0.995 | 0.004 | 0.000 |
  | 20 | 0.90, 0.80 | 0.980 | 0.125 | 0.010 |
  | 20 | 0.80, 0.70 | 0.940 | 0.552 | 0.173 |
  | 20 | 0.70, 0.60 | 0.880 | 0.864 | 0.532 |
  | 40 | 0.90, 0.80 | 0.980 | 0.058 | 0.001 |
  | 40 | 0.80, 0.70 | 0.940 | 0.620 | 0.133 |
  | 40 | 0.70, 0.60 | 0.880 | 0.954 | 0.623 |

  - At 0.90, a pair of writers missing 12% of a 40-item stratum escapes the extra pass 38% of the
    time. At 0.95 it escapes 5% of the time.
  - The cost of 0.95 is a likely extra pass for writers at about 94% true completeness. That stays
    within plan §6.6's budget of up to one extra pass per stratum. With five strata that is up to
    5 sessions rather than 4 (§14).
- **Tiny strata.** The rule applies regardless of size. When `r < 8`, the estimate is reported but
  is not informative. The stratum gets its extra pass whenever completeness falls below 0.95,
  which at that size happens often.

### 6.6 What S45 records, per stratum and pooled

- `n₁`, `n₂`, `m` and `r`; the same counts per session;
- `f₁`, `f₂` and `f₃`;
- N̂c with its interval, N̂h and N̂*, and completeness;
- whether the trigger fired;
- near-miss pairs formed, and the `same`/`different` counts;
- invalid items, and how each was fixed or dropped;
- the unclassified count;
- the count of disagreements between the writer's stratum and the derived one;
- the writers' `hops` distribution.

All of it is counts, with no locators.

## 7. Locator validation, spot-check and corrections

### 7.1 Validation (S45, before matching; G4)

Every check runs against `kit-src/` at `EVAL_COMMIT`, never against the system.

| Level | Applies to | Check |
|---|---|---|
| **V1**, lexical, required | every locator | `repoId = graphmind-ai-lab`; `path` is a file in the kit and in census scope (§3.3); a `symbol`'s `qname` starts with `path + "::"`, its chain matches bundle spec §4.1's grammar, and after `E` (§2.3) every NAME segment occurs in the file as a whole-word token; every `lines` element and `line` lies in `1 … lineCount` |
| **V2**, parsed, required where a stdlib parser exists | Python `symbol`; every `config` | **Python:** the `qname` is in the set `{E(q)}` over the `qname`s derived by `ast` per bundle spec §4.1–§4.2 (Python), and each line lies in the span of some symbol whose `E` is that `qname`, decorators included. **`config`:** the file parses (`json`, `tomllib`, a YAML parser, or `KEY=` lines for `.env`), and `keyPath` resolves to a node. |
| **V3**, contract | every `contract` | `kind` ∈ {`http`, `mcp_tool`}; `keyNorm` is unchanged by GC's mechanical normalization (§13.4). A changed one is replaced by the normalized form and logged, not invalidated. |
| **V4**, stratum rules, **per item after matching** (m1) | `flow` items | the anchor is in the drawn sample, and **some** member record's chain has `hops ≤ 2` and at least one flow hop (§3.6). If none does, the item is **out of scope** (counted), not invalid. |

**When member records of one item disagree** in their display fields, which never affect identity:
- `lines` is the union;
- `chain` is the first qualifying chain, in writer order KA, KB, KA2 and then by `seq`;
- `why` and `deployment` come from the record whose chain was taken;
- `provider` and `serves[]` are unions.

- TS/JS, shell and Markdown `qname`s get V1 only at P0. A residual error there is caught at the
  gate by §5.4's source-only rule.
- The YAML parser is `pyyaml==6.0.3`, from `cpg/pyproject.toml`'s `dev` extra (S09). That answers
  v1.0's Q4.

### 7.2 Invalid items

GC lists invalid items in `work/invalid.jsonl` (item id, check failed). The method owner reads them
in place and fixes each one from source, by `qname` grammar or file content, or drops it, and
returns the decisions by item id only. Fixed items re-enter matching. Drops are counted per writer
and per check.

### 7.3 The spot-check sample (S46; E10)

- **Union sample: 40 items.** Per derived stratum: **4 singletons** (found by one strategy only)
  and **4 doubletons** (found by both). Within each (stratum, class) cell they are drawn by hash
  sort with `seed = "ccg-p0f-spot-v1|" + EVAL_COMMIT` over `itemId`.
  - **Shortfall:** if a cell has fewer than 4 items, all of them are taken, and the deficit goes,
    in stratum order `http, mcp, scripts, wiring, flow`, **wrapping from `flow` back to `http`**,
    to the same class of the next stratum that has a surplus, then to the other class.
- **Why this split.** Singletons are where a false item hides: an item only one writer believes
  in. Doubletons give the comparison. The count also bounds the error rate. Computed Wilson upper
  bounds on the incorrect rate are **0.088** at 0/40, **0.129** at 1/40 and **0.165** at 2/40.
  Per stratum, 0/8 still allows up to **0.324**, so per-stratum error is only screened, not
  estimated.
- **Portuguese gold sample: 8** of QW's Portuguese items, drawn the same way with
  `seed = "ccg-p0f-ptgold-v1|" + EVAL_COMMIT` over `qid`: 6 unambiguous and 2 ambiguous, if
  available. The stakeholder checks whether the gold is the right reading and whether the question
  is natural Portuguese (ML §8.3).
- **English gold check (my method check, S47):** 8 English items, drawn the same way with
  `"ccg-p0f-engold-v1|"`.
- **The stakeholder marks** each sampled item correct or incorrect. Each incorrect mark carries a
  note, and, where the item has a fixable locator, the correct one.

### 7.4 Corrections and escalation (S47)

- **An incorrect item** is corrected, if the note gives a locator that passes §7.1, or removed
  otherwise. Each action is logged.
- **Escalation:** if a stratum has **≥ 2 of its 8** incorrect, or the pooled sample has
  **≥ 4 of 40**, then before the split the method owner reviews by source **every singleton** in
  that stratum, or in every stratum for the pooled trigger. The outcome is logged per item id.
  - Why this threshold: 2/8 has a Wilson interval of [0.071, 0.591] (computed), so it cannot
    certify a high error rate. But two false items in eight sampled is too many to leave in a set
    where each false item becomes a spurious miss at the gate.
  - Singletons are reviewed because a false item is far more likely to be a singleton.
- **No re-estimation.** S45's N̂* stands, because a correction applied to a sample would bias it.
  The report gives the post-correction union size beside it.
- **Reported:** incorrect counts by stratum and class, with Wilson intervals, and the counts for the
  PT and EN gold checks.

## 8. Development-contact rule (M3)

### 8.1 The rule (E6)

**An anchor is contacted** if a P0 artifact computed, asserted or hand-walked a link or an impact
**from that anchor** in the proving repo. **Every item of a contacted anchor goes to dev**, before
the random split. No held-out minimum overrides this. If the minimums then fail, the shortfall is
SQ-P0-1's trigger (plan §12), never a reason to un-contact an anchor.

### 8.2 The contact set C: its sources, and what is and is not contact

| Source | Contact | Not contact |
|---|---|---|
| **S14** (slice report) | every anchor of a link or impact the slice computed or asserted: SL-1's contract point `GET /shop/api/health`; SL-2's three `/health` providers' contract points; SL-4's two anchors (the provider contract point, and `buildUrl` for value mode); the SL-2 counterfactual's anchor | the wrapper and chain hops (`apiFetch`, `buildUrl` *as a hop*, the `fetch` call), and the mount call in `app.py`. They are mechanism, and every storefront item flows through them. Counting them would empty `http` and measure nothing better. |
| **S30** (rule-pack walkthroughs) | every proving-repo anchor whose link a walkthrough derives: each route, each MCP tool, and each **server unit** whose URL or launch binding is walked (for example the `.mcp.json` → `cypher` chain). A walked unit contacts its server-level wiring items (§3.5) | pack vocabulary examples that name no proving-repo anchor; the invented in-house bus; Kafka |
| **S28's golden flow methods** (S05 §6.4; produced by **S56** in `p0-2.md`), drawn from `falkor-chat/server` and `salesperson/` | each such method, **as a flow anchor**. Its intraprocedural flow was hand-labelled to judge the projection that the flow stratum measures, so P1 will debug against it | the same method appearing as an impacted element or chain hop in another stratum |
| **`cpg/tests/golden/*`** and any P0 test fixture holding a proving-repo locator (S34's `answer-ac1.json`) | every anchor in it | — |
| **S35** (fixture test plan) | nothing in the proving repo. Its locators have `repoId` ∈ {`fx-svc`, `fx-web`, `fx-lib`}. GC asserts that no proving-repo locator appears in it, and any that does is treated as contact. | — |
| S25/S26 call-site labels (frontend fitness) | — | **not contact.** They are a uniform random sample that judges call resolution in general, with no recognizer or link tuned to a specific anchor. A held-out item running through a labelled call site tests the same frontend under the same measurement. |
| Worked examples in P0 contract documents (bundle spec §4.2, §6) | — | **not contact.** They illustrate the `qname` and `keyNorm` grammars, and no recognizer was built against them. (An anchor that is *also* in S14's or S30's list is contact through that list.) |

- **The lists.** C is the union of these lists, each converted to `anchorId` by §2.2. S14 and S30
  list their anchors as locators (plan §3). GC records C, with its source per entry, in
  `work/contact.jsonl`, and the ground-truth report lists the contacted anchors by locator.
  They are dev, so they are not secret.
- **Timing.** C is applied at S47. The flow draw (S40) does not wait for C (§3.6).
- **Probe clusters (B1).** Contacting a unit or its probe contact point contacts both, because the
  split treats them as one cluster (§9.1 step 1).
- **Served anchors (v1.2, P2-1).** Contacting any anchor in a unit's served set S(u) sends the unit
  to dev, by the served-anchor rule (§3.5, §9.1 step 3b). It does not contact the unit's other
  served anchors: the rule is one-way.

### 8.3 Why forced-to-dev, per anchor

- **Kept eligible (rejected).** The slice's recognizers were hard-coded against
  `GET /shop/api/health` (plan §4.4, S12), and S30's packs are proven on paper against specific
  routes and tools that P1's pack authors will port. A held-out item on such an anchor would
  measure whether the port preserved a worked example. That is the ML §4.1 failure mode,
  "measures the debugging effort", in its purest form.
- **Reported as a separate "contacted" sub-stratum inside held-out (rejected).** It keeps the
  items, but AC-5's text says the list is "never used for development", and these items already
  were. A sub-stratum would be a held-out set that is not held out.
- **Per anchor, not per item.** The split unit is the anchor (§2.1). A contacted anchor's other
  items would otherwise sit in held-out beside a debugged sibling, which is the correlated leak the
  anchor split exists to stop.
- **Wrapper hops are not contact,** and that is the line that keeps the rule affordable. The
  held-out `http` items then test exactly what AC-5 should test: whether the mechanism the slice
  proved on one route generalizes to routes nobody developed against.
- **The cost**, an estimate until S14 and S30 list their anchors: 4 to 10 anchors leave held-out,
  roughly SL-1's and SL-2's three `/health` contract points, `buildUrl` if it is drawn, and S30's
  examples. Each loses its cluster of items. §9's top-up absorbs this unless the census is small,
  in which case S47's counts raise SQ-P0-1 as planned.

### 8.4 From P1 on

Contact after P0 is governed by the standing rule: pack authors work on dev only (ML §4.1). A
held-out item that any builder debugs against moves to dev (§12.1). C is closed at S47 and never
re-opened to bring an anchor back.

## 9. Split, held-out manifest and leak check

### 9.1 The AC-5 split (S47; E7)

**Input:**
- U, the corrected union of matched records, each with `itemId`, `anchorId`, `derivedStratum` and
  `counting` (§3.7);
- C, the contact set (§8);
- each unit anchor's probe (§5.1) and served set S(u) (§3.5);
- `seed = "ccg-p0f-split-v1|" + EVAL_COMMIT`;
- each anchor's `rank = SHA-256(seed + "|" + anchorId)` as lowercase hex.

**Procedure.** All integer arithmetic is exact, with `ceil(a·7/10)` computed as `-(-a*7 // 10)`.
For each stratum *s*:
- `H_s` is the held-out **gate-1** items with derived stratum *s*, counting or not;
- `H` is the held-out gate-1 **counting** items, pooled.

1. **Clusters (B1).** Join each unit anchor with its probe contract point by union-find. Do this
   even when the probe has no items of its own. Every other anchor is its own cluster.
   - A cluster's `rank` is the smallest `rank` among its anchors.
   - A cluster's **primary stratum** is the stratum holding most of its items. A tie goes to the
     earlier stratum in the order `http, mcp, scripts, wiring, flow`.
   - Every later step moves whole clusters.
2. **Contact.** A cluster with any anchor in C goes to dev.
3. **Share.** For each stratum *s*:
   - let A_s be the clusters whose primary stratum is *s* and that are not in dev, sorted by
     `rank` ascending;
   - let `T_s = ceil(0.7 × (items of A_s))`;
   - walk A_s in order, assigning each cluster to held-out until the held-out items of A_s reach
     `T_s`;
   - assign the rest to dev.

   3b. **Served anchors (v1.2, P2-1).** Repeat until nothing changes: a held-out unit cluster with
   any anchor of S(u) in another cluster that is in dev moves to dev. Each such move is logged
   as `served-anchor`. Only moves to dev happen, so this terminates.
4. **Minimums (M2: it terminates).** Start with an empty set M of marked strata. A dev cluster is
   **eligible** when no anchor of it is in C and, for a unit cluster, no anchor of S(u) outside it
   is in dev (v1.2). Repeat:
   - If some stratum *s* ∉ M has `H_s < 8`: take the one with the smallest `H_s`, ties by stratum
     order. If an eligible dev cluster has ≥ 1 item in *s*, move the lowest-`rank` one to
     held-out. Otherwise add *s* to M.
   - Else, if `H < 75`: if an eligible dev cluster has ≥ 1 **counting** item, move the
     lowest-`rank` one to held-out. Otherwise stop.
   - Else stop.

   Recompute after each move. Each pass either moves a cluster, of which there are finitely many,
   or marks a stratum, of which there are at most 5. Marked strata are excluded from both the test
   and the selection, so the loop terminates. Moving a tool to held-out can make a unit eligible
   later in the loop, never the reverse, so step 3b's invariant still holds when the loop ends.
5. **Trigger.** If `H < 75`, or M is non-empty, S47 reports `H`, `H_all`, every `H_s`, held-out
   clusters per stratum, dev clusters remaining (contacted and not), and the count of
   `served-anchor` moves (v1.2). That report is SQ-P0-1's
   trigger (plan §12). The split is still written as computed. Step 6 is skipped.
6. **Reserve (E12).** Walk the held-out clusters in **descending** `rank`, considering only those
   with ≥ 1 counting item. Move a cluster to the **reserve** when two conditions hold: the reserve
   holds fewer than 35 counting items before the move, and after the move gate 1 still has
   `H ≥ 75` and every `H_s ≥ 8`. Otherwise skip that cluster. Stop when the reserve holds ≥ 35
   counting items, or when every cluster has been considered.
   - Reserve items are held-out: written, frozen and hashed with the rest.
   - They are **never exposed at gate 1**. They are the first source of fresh items for a re-gate
     (§12.1), or for the next phase's AC-5 run.

**Why a 0.70 share.** The binding constraint is AC-5's 75, and anchors are lumpy, so held-out
should hold most of the census. A 30% dev share keeps real proving-repo dev items for pack
authors, in addition to the fixture and the contacted anchors. That avoids P1 debugging only on
fixtures, whose link idioms are planted rather than organic. When the census is small, step 4
reduces dev toward zero automatically, which is exactly what plan §12 describes. Computed targets:
8 items give 6 held-out, 20 give 14, 30 give 21, 40 give 28.

**Why the served-anchor rule, and not one cluster per server (P2-1).**
- **The leak it closes.** Under v1.1 only the probe was clustered with its unit. A held-out unit
  whose sibling tool T2 sat in dev was scored on binding entries that T2's dev answer shows,
  because AC-2 makes every tool's answer list them.
- **One cluster per whole server (rejected).** It would merge falkor-chat's 25 registered tools,
  most of `mcp`, into one cluster. That cluster lands wholly on one side, so `mcp` would have no
  proving-repo dev items, or no held-out items. If S30 walks any one falkor-chat tool, contact then
  sends all of them to dev and `H_mcp` collapses.
- **Always sending server-binding items to dev (rejected).** That is safe, but the served-anchor
  rule is equally safe and keeps a single-tool server's binding held out whenever its tool is.

**What the rule costs.** Counted at the working tree on 2026-10-29; S40 recounts at
`EVAL_COMMIT`, and writers may find more.
- **The wiring inventory: 14 entries.**
  - Server class, 7: falkor-chat/server has 4 (`.mcp.json` `mcpServers.falkor-chat-agent-team`,
    the Kiro agent's `mcpServers.falkor-chat`, `mcp-monitor/config.example.toml`
    `server.falkor-chat`, `falkor-chat/compose.yaml` `services.server`); cypher-mcp has 2
    (`mcpServers.cypher`, `.claude/settings.json` `enabledMcpjsonServers[0]`); mcp-monitor has 1
    (`server.fake-test`).
  - Contract class, 7: the Kiro agent's `tools[0..1]` and `allowedTools[0..1]`, `watch[0..1]`, and
    `permissions.allow`'s `mcp__cypher__query`.
- **Where it fires.** Only for a unit serving more than one anchor. cypher-mcp serves one tool
  (`query`) and mcp-monitor's fake server one (`get_status`), and each is its unit's probe. So
  only falkor-chat/server is affected, and with 25 tools and its routes its cluster ends in dev
  in effect always.
- **On `H`.** The rule removes that cluster's counting items: its 4 binding entries plus its
  probe's, which "fewest items" (§5.1) should keep at 0–1 (an estimate). That threatens `H ≥ 75` only when the census
  is within about 5 of 75, and step 4's top-up then moves other dev clusters in.
- **On `H_wiring`: this is where it binds.** v1.1 reached 8 only by counting the 4 leaked entries.
  Under v1.2, the eligible wiring supply is 14 − 4 = **10**. It is **7** if S30 contacts
  cypher-mcp, because the probe cluster takes its 2 entries and `query`'s permission entry with
  it, and §8.2's example walkthrough does exactly that. At 7, `wiring` cannot reach 8.
- **Simulated.** This uses a repo-shaped synthetic union over 2,000 split seeds (scratch
  `split12.py`). It is not part of Appendix A; the durable check is §13.6's P2-1 oracle cases.
  - v1.1 leaks in every seed, and v1.2 in none.
  - The mean change in `H` is −3.0 to −4.2.
  - With cypher-mcp contacted, `H_wiring` is at most 7 and `wiring` is marked in every seed.
    Without that contact, `H_wiring` averages 8.7 (at most 10) and is never marked.
- **How a shortfall is handled: by the existing path.** Step 4 cannot move the falkor-chat unit
  cluster, which is ineligible. So it marks `wiring`, and step 5 reports it. That is SQ-P0-1's
  trigger (plan §12), whose options then apply.
- **The cheap avoidance is upstream.** S30 should take its server-binding walkthrough on
  falkor-chat/server, which v1.2 sends to dev anyway, so the contact costs nothing, or on S35's
  fixture. It should not take it on cypher-mcp or mcp-monitor (§14; §15 Q10).

### 9.2 Serialization

Every file is canonical JSONL:
- each line is the canonical JSON of one record (bundle spec §3.1: UTF-8 with no BOM, keys sorted,
  separators `,` and `:`, non-ASCII written literally);
- lines are sorted by `itemId` (`qid` for questions);
- LF line endings, with a final newline.

| File | Content |
|---|---|
| `~/.local/share/ccg-eval/<E>/heldout/ac5-<stratum>.jsonl` (five files) | gate 1's list: the normalized record (§4.4) with `counting`, plus the original writer objects of every member writer |
| `~/.local/share/ccg-eval/<E>/heldout/ac5-reserve.jsonl` | the reserve (§9.1 step 6), in the same form, each record carrying its stratum |
| `~/.local/share/ccg-eval/<E>/heldout/exposure.jsonl` | per held-out `itemId`: `{itemId, firstGate, hit}`, appended at every gate (§12.1). Empty at S47. |
| `~/.local/share/ccg-eval/<E>/heldout/ac24.jsonl` | the held-out question items with gold and the final ambiguity label (§10) |
| `cpg/eval/dev/ac5-<stratum>.jsonl`, `cpg/eval/dev/ac24.jsonl` | the same, for dev |

### 9.3 The held-out manifest (`cpg/eval/heldout-manifest.json`, committed; D11)

```json
{
  "schema": "ccg-heldout-manifest-v1",
  "evalCommit": "<40 hex>",
  "protocol": "docs/plans/connected-code-graph-eval-protocol-ml.md@1.2",
  "splitSeed": "ccg-p0f-split-v1|<40 hex>",
  "files": [{"name": "ac5-http.jsonl", "sha256": "<64 hex>", "bytes": 0, "items": 0, "countingItems": 0, "anchors": 0}],
  "contactAnchors": 0,
  "rawArtifacts": [{"path": "work/match.json", "sha256": "<64 hex>"}]
}
```

- `files` lists the five gate-1 files, `ac5-reserve.jsonl` and `ac24.jsonl`. `exposure.jsonl`
  grows at every gate, so its hash is re-recorded in the gate's own report rather than here.
- `rawArtifacts` hashes the raw keys, `work/`, the role roots' `out/` files, the `runs/` logs and
  the session transcripts, which stay beside `heldout/` (plan §6.4).
- **The manifest holds no locator, keyNorm, qname, line or question text.** Counts and hashes only.
- **`--verify`** re-hashes every listed file and re-counts its lines. Any mismatch in `sha256`,
  `bytes` or `items`, or a missing file, exits non-zero, and the gate stops (plan §6.5). It detects
  modification, not reading (D11, K-P0-7).

### 9.4 The report leak check (`split_manifest.py --check-report`)

- **Probe set.** For every held-out item:
  - its `itemId`, `anchorId` and `impactedId`, both at 32 hex and at their first 12 hex;
  - every `qname`;
  - every `path:line` and `path#keyPath`;
  - every `keyNorm`;
  - for questions, every 6-word shingle of the `text`, casefolded and whitespace-collapsed.

  Then **remove** every probe that is **exactly equal** to the value of the same kind of field in
  some dev item or in C (m4). Equality is on the whole string. Containment does not count: a
  held-out `a.py::f` stays a probe when dev holds only `a.py::f.g`. Removed probes are public
  (§8.2), and keeping them would fail the check on a contacted anchor named legitimately. Also
  drop probes shorter than 8 characters, such as a bare tool name like `query`.
- **Test.** The report is casefolded and whitespace-collapsed, and must contain **no** probe:
  - a `qname`, `keyNorm` or `path:…` probe is matched **as a whole token**, bounded on both sides
    by the start or end of the text, or by a character outside `[A-Za-z0-9_.:/#<>-]`. So
    `a.py::f` does not hit inside `a.py::f.g`;
  - **v1.2 (P2-3):** the right boundary may be preceded by a run of `.` and `:`, so a probe at the
    end of a sentence or before a colon still hits. The left side may carry a `./` prefix. With
    `T = A-Za-z0-9_.:/#<>-` and the probe escaped as `P`, the pattern is
    `(?:^|(?<=[^T]))(?:\./)?P(?=[.:]*(?:$|[^T]))`, applied to the casefolded, whitespace-collapsed
    report with casefolded probes. It was run on fifteen cases (scratch `leak12.py`), and every one came out as wanted. Among them:
    - it hits `held-out item a.py::f.`, `a.py::f: missed`, `a.py::f..`, `./a.py::f`,
      `pkg/a.py:12.` and `get /api/items/{}: missed`. v1.1 hit none of them;
    - it does not hit `a.py::f.g`, `x/a.py::f`, `a.py::fg`, `a.py::f::g`, `a.py::f.:g`,
      `pkg/a.py:123` or `get /api/items/{}/x`;
  - hex and shingle probes are matched as substrings. Any hit
  fails, and the hits are printed **by probe hash only**, so the failure output does not itself
  leak.
- **The report carries counts, hashes and the contacted anchors**, never a held-out or dev item's
  content (plan §6.4).

## 10. AC-24 at P1: counts, ambiguity and scoring

### 10.1 What QW writes (S43; E8, G7)

| Cell | Count per language | Total |
|---|---|---|
| impact × `code-location` (a symbol name, or a file and line) | 9 | 18 |
| impact × `contract-point` (a route or a tool) | 9 | 18 |
| ambiguity candidates, 6 per language, each of the four `ambiguityKind`s at least once per language | 6 | 12 |
| **Total** | 24 | **48** |

- Both `direction`s appear in each code-location and contract-point cell: at least 2 of the 9 are
  `depends-on`, because FR-11's reverse question is part of the impact use case.
- **Why only two starting-point groups.** P1's engine accepts code-location and contract-point
  anchors only. Change anchors arrive in P3 and failure evidence in P5 (proposal §4). Questions
  written now for those groups could not be scored for two or more phases, and their gold would
  need a pinned diff or a pinned stack trace that does not exist at `EVAL_COMMIT`. The phases that
  build them write those cells, under the same §6 protocol, at their own pinned commit. The full
  AC-24 gate (P5) gets its 20-cell set the same way (ML §8.3).
- **Why 9 per cell.** The split puts 6 per (group, language) in held-out (§10.2), and the rest go
  to dev. With no annotator drops that is 3 per cell. Up to 3 drops per cell, which §10.3 expects
  at a low rate, still leave the 6 held-out.

### 10.2 Split (S47)

- `seed = "ccg-p0f-ac24-v1|" + EVAL_COMMIT`, with hash sort over `qid`. GC assigns a `qid` of
  `QW-<seq>` when it copies the file.
- **Unambiguous set.** In each (group, language) cell, among the items that are final-unambiguous
  (§10.3), the first `min(6, n)` by rank go to held-out and the rest to dev. The target is
  **24 held-out**, 12 per language, matching ML §8.3's "12–13 per language". With no drops, dev
  gets 3 per cell, 12 in total.
- **Ambiguity set.** Per language, among the final-ambiguous items, the first by rank goes to dev
  and the rest to held-out. That gives at most 5 held-out per language, so at most 10 in total.
- **Contact does not apply to AC-24.** P0 develops no interpreter. A question whose gold anchor is
  contacted under §8 tests interpretation, which nobody tuned. From P1 on, held-out questions
  never appear as few-shot examples or in prompt work, and only dev questions may (ML §8.3).

### 10.3 Ambiguity labels (S44, S47)

AN1 and AN2 label every `aid`. GC joins their labels to QW's `intent`:

| AN1 | AN2 | QW `intent` | Outcome |
|---|---|---|---|
| ambiguous | ambiguous | ambiguous | **ambiguity set**, gold = `goldOptions` |
| unambiguous | unambiguous | unambiguous | **unambiguous set**, gold = `gold` |
| ambiguous | ambiguous | unambiguous | dropped, because there is no option-set gold. Counted as "writer-unambiguous, annotators-ambiguous". |
| unambiguous | unambiguous | ambiguous | dropped. Counted as "writer-ambiguous, annotators-unambiguous". |
| they disagree | | any | dropped and reported (ML §8.3) |

Reported: the four outcome counts per language, the raw agreement rate, and Cohen's κ, as
diagnostics only.

### 10.4 Scoring (at a gate; ML §8.3's table, made mechanical)

- **The interpretation equals gold** when `useCase` and `direction` are equal, the anchor sets are
  equal as **sets of ids**, and `scope` is equal. `null` means the whole system.
  - **E1 applies first (M4).** Before the sets are compared, every interpreted `symbol` anchor is
    mapped to `E(qname)` (§2.3), exactly as H1 does for AC-5. Then each locator is converted to its
    id (§2.2).
  - So an interpreter that resolves "line 42 of `ProductList.tsx`" to the lambda enclosing that
    line scores the same as one that resolves it to `ProductList`. Gold is always the eligible
    ancestor.
  - This needs the interpreter's resolved anchors to carry `qname`, which S33's interpretation
    block does (§15, Q1).
  - **Known limit (P2-2).** Two `#k` duplicates share one `E`, so AC-24 cannot tell an
    interpreter that picks the wrong one from one that picks the right one. Exposure is nil today
    (§5.2's scan). No span test is added here, because a question's gold has no lines.
- **An unambiguous item** is understood **only** when the answer gives a single interpretation equal
  to gold. Options, even with gold among them, are over-asking and score as not understood.
- **An ambiguous item** is understood when the answer is an option set, or an explicit request for
  clarification, with no silent single interpretation, and:
  - for `several-symbols` and `two-use-cases`, the options contain **every** `goldOptions` entry.
    A list that omits a legitimate reading leaves the asker unable to choose it;
  - for `unknown-name` and `no-anchor` (`goldOptions = []`), the answer resolves no anchor
    silently.
- **Reported:** understood rate pooled and per language; the over-asking rate on unambiguous items;
  ambiguity handling; and the flip rate over 5 uncached repeats per item (ML §4.4). Results are per
  configured interpreter model, and the gate binds the codebase's default model (SQ7, decided).

### 10.5 The P1 smoke gate (provisional; ML §8.3)

- **Pooled:** ≥ 0.92 understood over the held-out unambiguous items. At 24 items that is ≥ 23
  (computed: `ceil(0.92 × 24) = 23`). 23/24 has a Wilson interval of [0.798, 0.993], and 24/24
  [0.862, 1.000].
- **Per language:** ≥ 0.85, so ≥ 11 of 12 (computed). 11/12 is [0.646, 0.985].
- **Ambiguity:** every held-out ambiguous item is understood.
- **Reading:** at n = 24, this is a smoke gate. A pass shows the interpreter is not broken on
  the impact cells. It does not certify 0.92.
- **Deferred to the full set:** the paired EN/PT probe (ML §8.3). With 4 pairs it carries no
  signal, since McNemar needs 6 net discordant pairs to reach α = 0.05 at all.

## 11. Noise bound by distance band (P1 onward)

Noise is measured at every AC-5 gate, over the answers to the held-out anchors (§5.1). Labelling
system output is permitted here, unlike in §6: precision is a judgment *about* emitted items
(ML §4.2).

### 11.1 Frame and strata

- **Answers:** one per held-out anchor. **Items:** every listed item and every expanded group
  member.
- **Confirmed items:** all of them are labelled. This is a census.
- **Possible items:** sampled, stratified by *h* = (`reason` code, distance band), with the bands
  being **1**, **2** and **≥ 3** (inter-function hops, §2.1). `N_h` is the count of possible items
  in *h* over all answers.

### 11.2 The possible-item sample

- **Size:** at least 100.
- **Allocation:** first, every stratum gets `min(N_h, 5)`. Then the remainder of 100 is allocated
  in proportion to `N_h` by largest remainder, in exact integer arithmetic, capped at `N_h`.
  Leftover capacity goes to the stratum with the largest unsampled count. If the minima alone
  exceed 100, the sample is their sum.
- **Draw:** within *h*, by hash sort with
  `seed = "ccg-noise-v1|" + gateId + "|" + snapshotId`, over `anchorId + "|" + itemId`.
- **The labeller** is not a pack author, sees the item with its chain, and applies §11.4. Each
  label records `real` or `false`, and the distance the system reported.

### 11.3 Estimates

With `n_h` labelled in stratum *h*, `f_h` of them false and `t_h` real:
- **Confirmed precision** = real confirmed ÷ confirmed, with a Wilson interval. **Gate: ≥ 0.95
  observed. Every false confirmed item is a defect** (ML §4.2).
- **The AC-5 ratio** (SQ2(a), decided): `Q = Ŵ / (RC + R̂)`, where
  - `Ŵ = Σ_h N_h · f_h / n_h` is the estimated wrong possible items;
  - `R̂ = Σ_h N_h · t_h / n_h` is the estimated real possible items;
  - `RC` is the real confirmed items (from the census).

  This is a ratio of sums pooled over answers. **Gate: Q ≤ 4** (provisional; AC-5).
- **By band:** `Q_b` uses the same formula restricted to band *b*, both numerator and the
  real-possible term, with `RC_b` the real confirmed items at distance band *b*. Reported, not
  gated (AC-5: "also reported by distance").
- **Possible precision** is reported per reason code and per band, with Wilson intervals. These
  are descriptive where `n_h < 20`.
- **Burden:** expanded possible items per answer, median and p90. Ranks are computed exactly, the
  p90 index being `-(-9 * n // 10)`.
- **Reduction against the naive over-lister** (every name-match and every provider of the kind) at
  equal recall, as in ML §4.2.
- **Interval for Q:** bootstrap over **answers**, never items, because items within an answer are
  correlated (ML §8.2).
  - Answers are sorted by `anchorId` first, which fixes the row order.
  - `random.Random("ccg-noise-boot-v1|" + gateId)` draws `B = 10000` resamples of answers with
    replacement, by `randrange`.
  - Each labelled item travels with its answer. A stratum with `N_h > 0` but no labelled item in a
    resample uses the full-sample rate for *h*.
  - The percentile interval takes indices 250 and 9749 of the sorted resampled Q (0-based, exact).
  - A fixed seed alone does not make a bootstrap reproducible, because it still moves with the
    input order. The sort is what makes this one reproducible.
- **Starting thresholds** (ML §4.2): possible precision ≥ 0.20 pooled, p90 burden ≤ 100 and
  reduction ≥ 5× are reported. They are ratified after P1's first measurement and are not gated
  before then.

### 11.4 Labelling rubric (ML §4.2, §8.1)

- **Direct items (distance 1):** real if a dependency exists in at least one deployable
  configuration.
- **Indirect items (distance ≥ 2):** real if **some execution exists** in which a change at the
  anchor changes the item's inputs, or changes whether the item runs.
- The labeller records the distance and judges the item, not the chain. A real item reached by a
  wrong chain is `real`, and the chain error goes in a note.
- **Spot-check:** 20 labels per gate, drawn by hash sort from the labelled set, re-labelled by the
  stakeholder or by a second labeller who is not a pack author. The agreement is reported.

## 12. AC-5 fail procedure, planted minimums (AC-6, AC-25) and AC-7 pre-registration

### 12.1 AC-5 fail procedure (decided SQ1; ML §8.5)

1. **The first run is the result.** It goes into the gate's test report as run, including every
   miss.
2. **Triage of each miss,** by the gate runner and the method owner, into exactly one category:
   - `key-error`: the item is false. The stakeholder must confirm this;
   - `extraction`: the element is absent from the snapshot (§5.4);
   - `recognizer` (a pack gap);
   - `linker-engine`;
   - `depth-bound`: the frontier was cut before the item;
   - `dynamic`: it should have been possible, and was not listed.

   The categories are reported as counts. Item content goes only to builder-invisible places
   (`work/`, and the method owner's notes outside the repo).
   - A confirmed `key-error` is removed from held-out. The report then gives recall **as run** and
     recall **excluding confirmed key errors**, side by side. The first figure is the gate result.
3. **Fix on dev.** Every item of a missed item's **cluster** (§9.1 step 1) moves to dev, because a
   fix touches the whole cluster (§2.1). Clusters with no miss stay held-out, and they stay blind:
   builders never see the gate's answers or the triage notes.
   - **Then the served-anchor rule (v1.2, P2-1).** §9.1 step 3b is re-applied: every kept or
     reserve unit cluster with an anchor of S(u) now in dev moves to dev too. It is logged as
     `served-anchor`, reported as a count, and is not a miss.
4. **Re-gate on a new list, never the same one** (stakeholder decision, 2026-10-29). The new list
   is the **kept** held-out clusters (exposed, no miss), plus **fresh** items.
   - **Routing of fresh items, by anchor and not only by `itemId` (B1).** Each fresh item is
     clustered as in §9.1 step 1. Then, in this order:
     - an item whose `itemId` is already in dev, held-out or the reserve is discarded;
     - an item whose cluster contains an anchor in dev or in C **goes to dev**. That anchor's
       answer has been debugged, or developed against. The same holds for an item on a unit whose
       served set S(u) has an anchor in dev or in C (v1.2, P2-1);
     - an item whose cluster is a kept cluster joins it. It is scored, but it does **not** count
       as fresh-anchor evidence;
     - every other item is on a **fresh cluster**: one never exposed at any gate.
   - **Composition.** The list must reach `H ≥ 75` counting and every `H_s ≥ 8` (§3.7). For each
     stratum that had a miss, it must include at least as many fresh-cluster items of that stratum
     as were moved out of it, where the supply allows. Any shortfall is reported.
   - **At a new `EVAL_COMMIT`,** the kept and reserve items are re-validated from source (§7.1) by
     GC and the method owner. An item whose element no longer exists is dropped and counted.
5. **The fresh-item floor (E12; stakeholder decision, 2026-10-29).** A re-gate counts as an AC-5
   pass only if all three conditions hold:
   - every item on the list is a hit, kept and fresh alike;
   - the list meets the composition rule in step 4;
   - **the list holds at least F = 35 counting items on fresh clusters.** Every item on the list
     is a hit, so all of these are too.

   A re-gate that meets the first two conditions but not the third is reported as "all hits,
   floor not met". It is **not** an AC-5 pass. The next re-gate needs more fresh-cluster items.
6. **What every re-gate reports (M1)**, in addition to §5.3:
   - **fresh-cluster recall:** `k/n` with its Wilson interval, as its own figure;
   - **the cumulative first-exposure record:** each held-out item is counted once, at the first
     gate that exposed it, summed over gates, per stratum and pooled, with a Wilson interval. The
     record is kept in `heldout/exposure.jsonl`. For example, gate 1 at 73/75 followed by a
     floor-meeting gate 2 at 35/35 fresh gives 108/110, Wilson [0.936, 0.995] (computed);
   - **kept items, named as a regression check:** `k/n`. They are **not recall evidence**,
     because they were selected on having passed. A kept miss is a miss, and steps 2–3 apply.
   - **The claim on a re-gate pass** is the fresh-cluster figure: 35/35 has a Wilson lower bound of
     0.901 (computed). The cumulative record is given beside it. §5.3's 0.951 is never claimed
     for a re-gate.
7. **Never re-run the gate on the list that failed**, whether after a fix, a config change or a
   default-bound change.

**Sizing the floor.** The model is the gate's (`regate.py`):
- gate 1 has 76 items in 2-item clusters;
- kept items pass again;
- each re-gate's fresh clusters number `max(missed, ⌈F/2⌉)`;
- true recall p stays constant, which is the case where fixes do not generalize.

The result is computed exactly by a DP over the missed-cluster count (Appendix A). At F = 0 it
reproduces the gate's Monte Carlo figures (0.985 and 0.696).

| Floor F | Wilson lower bound at F/F | P(pass by gate 2 / 3) at p = 0.99 | at p = 0.97 | at p = 0.95 |
|---|---|---|---|---|
| 0 (v1.0) | — | 0.985 / 1.000 | 0.876 / 0.992 | 0.696 / 0.965 |
| 20 | 0.839 | 0.903 / 0.982 | 0.589 / 0.812 | 0.371 / 0.597 |
| **35** | **0.901** | **0.838 / 0.951** | **0.400 / 0.600** | **0.175 / 0.305** |
| 50 | 0.929 | 0.789 / 0.917 | 0.295 / 0.449 | 0.096 / 0.165 |
| 75 | 0.951 | 0.715 / 0.848 | 0.188 / 0.268 | 0.040 / 0.060 |

P(pass at gate 1) is 0.466, 0.099 and 0.020 respectively, for every F.

- **Why 35.** It is the smallest floor whose all-hit evidence alone bounds recall at ≥ 0.90
  (`n ≥ 9z² = 34.57`, computed), which is ML §4.1's 35-row. A non-generalizing system at p = 0.95
  now passes within two re-gates 30.5% of the time instead of 96.5%. When it does pass, the claim
  made, fresh-cluster 35/35 with a lower bound of 0.901 plus the cumulative record, is true of it.
- **Why not 50 or 75.** The proving repo cannot supply that many never-seen items per re-gate. A
  floor of 75 would amount to the all-fresh reading the stakeholder rejected.

**Where fresh items come from, and what they cost**, in order:
1. **The reserve** (§9.1 step 6): up to 35 counting items set aside at S47, never exposed. This
   costs no census. It is empty whenever S47's census only just reaches gate 1's minimums.
2. **A same-commit census top-up**: one KA and one KB session per stratum that needs items, so up
   to 10 headless sessions (plan §6.6 rates).
   - Fresh items here are only what the first census missed, about `r·(1/c − 1)` for completeness
     c. That is about 8 items for r = 150 at c = 0.95 (computed), and fewer when the writers share
     blind spots (G6).
   - **It will rarely reach 35 on its own.** If the reserve plus this top-up cannot meet the floor,
     go to 3.
3. **A newly pinned `EVAL_COMMIT`**: a full census, which is the P0 ground-truth chain without the
   rehearsal. That is about 10–15 headless sessions plus S46's spot-check, the same as §6.6's
   range. Fresh items are the new commit's items whose `itemId` is in no existing set. The yield
   depends on how much the repo has grown, and it cannot be estimated now.
4. **If 1–3 still fall short of 35**, the re-gate may run, as a regression check and as fresh
   evidence, but it cannot count as a pass. `teco` raises the shortfall with the stakeholder,
   with SQ-P0-1's option list (plan §12). Option (a), out-of-repo planted items reported as their
   own stratum, would amend AC-5 and is the stakeholder's call.

### 12.2 Planted minimums for S35 (the fixture system)

S35 sets the counts. These are the floors:

| What | Minimum | Source |
|---|---|---|
| Each AC-6 unprovable kind at P1 (runtime-built URL, dynamic import, example config, prompt mention); topic-from-config added at P2 | **≥ 3 instances per kind, in 3 distinct syntactic shapes** | ML §8.2; proposal §5.2 |
| AC-25 data-flow effects | **≥ 3**, at least one at distance 1 and one at distance 2 | ML §8.2 |
| AC-25 control-dependency effects | **≥ 3**, at least one guarding a call in another function | ML §8.2 |
| AC-25 reverse ("depends on") | **≥ 3** planted functions with a known dependency list | AC-25 |
| Each ML §4.1 difficulty shape (exact literal, constant across files, template literal, client wrapper, config-bound base URL, runtime-built key, dynamic import, DI or reflection, prompt mention, cross-repository call with version skew) | **≥ 3 each** across the fixture | ML §4.1 |
| Each link kind × each expected status present in the fixture | **≥ 1**; every `possible` one carries its expected reason code | proposal §5.1 |

- Fixture items are in the repo, so they are **dev by construction**. They are reported as their
  own stratum and **never pooled** with proving-repo AC-5 items (ML §4.1, finding F7).
- AC-6 passes per kind when every instance is `possible` with the expected reason **and none is
  confirmed**. The second clause is the real test (ML §4.6).
- **AC-25's presentation properties belong to S35's assertion column (m6).** For every planted
  AC-25 item, S35 lists four assertions against S33's answer fields, and S39's tests check them:
  - `direct`/`indirect` with the expected distance;
  - grouping by repository, then component, then distance;
  - confirmed before possible within a group;
  - the reverse question listing the planted dependencies the same way.

  This note owns only the counts above.

### 12.3 Pre-registration records (AC-7, and every proving-repo AC-5 gate)

The file is `cpg/eval/preregistration.jsonl`: committed, append-only, one canonical JSON line per
record.

```json
{
  "schema": "ccg-prereg-v1",
  "codebase": {"name": "opentelemetry-demo", "repo": "<url>", "tag": "<tag>", "commit": "<40 hex>"},
  "stratum": "grpc",
  "packSetHash": "<64 hex>",
  "coreCommit": "<40 hex>",
  "coreTreeHash": "<64 hex>",
  "algorithm": {"id": "<S20 choice>", "version": "<semver>"},
  "defaultDepthBound": 2,
  "firstContactDate": "2027-01-15",
  "recordedBy": "teco"
}
```

The example values are illustrative.

- **`packSetHash`** = SHA-256 over the canonical JSON (bundle spec §3.1) of the array, sorted by
  path, of `[relpath, sha256(file bytes)]` for **every** file under `cpg/packs/`. It covers all
  packs, not only the stratum's, because any pack can produce a link in any stratum.
- **`coreCommit`** is the commit of `cpg/` used for the build.
- **`coreTreeHash`** = the same construction as `packSetHash`, over every git-tracked file under
  `cpg/` **except** `cpg/eval/**`, `cpg/packs/**`, `cpg/tests/**` and `**/*.md`. The exclusions
  are: ground truth, which is not configuration; the packs, which `packSetHash` covers; and
  material that cannot change behaviour.
- **The rule (M5: the core is frozen too).** The record's commit must precede, in `git log`, the
  first build that includes support for that stratum (the per-stratum first contact, ML §8.1
  B5.1). At that build the harness recomputes all four of `packSetHash`, `coreTreeHash`,
  `algorithm` and `defaultDepthBound` from the build's own working tree, which must have no
  uncommitted changes under `cpg/`, and compares them with the record.
  - **All four equal:** the run is the first contact, and **its result is the AC-7 result.**
  - **Any one differs:** the harness refuses to label the run first contact. It may still run, but
    only as "after adaptation" (ML §4.5).

  AC-7's "configuration frozen before first contact" covers the core as well as the packs:
  stratum-specific handling added to the core after registration would otherwise still count.
- **Proving-repo AC-5 gates** use the same record, with `codebase` set to the proving repo at
  `EVAL_COMMIT` and `stratum` set to `ac5-gate-<n>`. That fixes the depth bound and the pack set
  before the run (§5.1).

## 13. Tooling procedures for GC (M5)

Common to every tool:
- Python 3.12. Ids come from `cpg/ccg/ids.py` (S09), and hash sorts and seeds follow S05 §4.4's
  convention: SHA-256 hex, ascending, over `seed + "|" + key`.
- Every output is canonical JSONL (§9.2).
- Every tool is deterministic on its inputs, and each tool's test (S48/S49) includes a
  byte-identical rerun.
- **No tool prints item content to stdout.** They print counts, ids and file paths only, so that
  `teco`'s transcript stays clean (plan §6.4).

### 13.1 `export_kit.py` (S48; run in S40)

1. `git archive EVAL_COMMIT | tar -x` into `kit-src/`.
2. Delete plan §6.2's excluded paths.
3. **Content check:** grep case-insensitively for
   `connected-code-graph|connected code graph|C-1001|\bccg[_ -]`. Every file with a hit is deleted,
   added to `kit-manifest.json` with reason `content-check`, and added to `eval-manifest.yaml`'s
   exclusions. Re-grep must give zero hits. An excluded file that would otherwise be in census
   scope counts as "excluded scope" for the report: a file with a code or config `lang` under
   bundle spec §3.3, or a prompt-bearing Markdown file, outside the test globs. The count is by
   path class; content is never inspected.
4. `kit-manifest.json` records, for every file sorted by path, its path and SHA-256; the SHA-256 of
   every `BRIEF.md`; and the exclusion list. Then `chmod -R a-w kit-src`, and no `.git` is left.
5. **`--draw-flow-anchors`:** builds §3.6's frame from `kit-src/` and writes
   `work/flow-anchors.jsonl`: `{lang, rank, qname, path, lineStart, lineEnd}`, 8 per language.
   It needs Python `ast` and py-tree-sitter from `cpg/pyproject.toml`'s `spike` extra (m5). S40
   copies the list verbatim into both flow briefs.
6. **`--list-units`:** lists every directory in census scope holding a `pyproject.toml`,
   `package.json` or `requirements.txt`, as `{rootPath, lang, ports}`, with `lang` taken from the
   manifest's ecosystem (§2.2). `ports` is emitted empty, and S40 fills it from the unit's source
   by §2.4's unit-ports rule (v1.2, P2-4).
7. **`--list-cypher-tools`:** reads `cypher-mcp/server.py` with `ast` and lists the tool names its
   `@mcp.tool` decorators register: the `name=` keyword if present, otherwise the function name
   (§3.4). At the time of writing, line 957 registers `name="query"`. If none is found, the
   step fails rather than producing an empty list.
8. **`eval-manifest.yaml`** (written by S40 with this tool) holds:
   - the exclusions;
   - §3.3's test-code globs and prompt-bearing globs;
   - the strata order;
   - `units` (step 6, with `ports`) and `cypherMcpTools` (step 7);
   - §2.4's grain table, as patterns;
   - the flow-anchor list's SHA-256;
   - this note's path and version.

### 13.2 `strip_gold.py` (S48; run in S43a)

- **Input:** `QW.jsonl`. **Output:** `QW-nogold.jsonl`, by a **whitelist**: each line is exactly
  `{aid, lang, text}`, where `aid = SHA-256("ccg-p0f-an-v1|" + EVAL_COMMIT + "|" + qid)`, first
  12 hex. Lines are sorted by `aid`, so neither QW's order nor its numbering reaches the
  annotators (QW may write its ambiguous items last).
- The `aid → qid` map goes to `work/aid-map.jsonl` only.
- **`--verify`:** every `qid` maps to exactly one output line; every output object has exactly the
  keys {`aid`, `lang`, `text`}; and no output line contains a `gold`, `goldOptions`, `intent`,
  `group` or `ambiguityKind` substring.

### 13.3 `audit.py` (S48; plan §6.5 is the specification)

Plan §6.5 specifies it fully. This note adds only what the report receives:
- per session: attempts, refused-attempt count by type, voiding-violation count by type, and the
  void/clean verdict;
- per run: the totals.

There are no paths and no tool inputs.

### 13.4 `match_keys.py` (S49; run in S45)

Verbs, in order:
- **`normalize`.** For each writer's file:
  - drop an unparseable final line (plan §6.1);
  - NFC and strip every string (bundle spec §3.1);
  - make `path` repo-relative POSIX, with no leading `./`;
  - set `repoId` to `graphmind-ai-lab` where it is missing; if a different value is given, the item
    fails V1;
  - **`E(qname)`** (§2.3). An ancestor-plus-line locator is first converted by bundle spec §3.5's
    converter. Each rewrite is logged;
  - **http `keyNorm`:** apply bundle spec §6, steps 1, 4, 5, 6 and 7 to the writer's string: an
    uppercase verb; a leading `/`; runs of `/` collapsed; RFC 3986 percent-decoding; every
    parameter syntax rewritten to `{}`/`{**}`. Step 3, mount composition, is the writer's job
    and cannot be done mechanically;
  - **mcp_tool `keyNorm`:** a namespaced form `mcp__<server>__<tool>` becomes `<tool>`;
  - **`config` grain:** truncate every `keyPath` to §2.4's grain for its format;
  - **re-anchoring:** a server-class entry under a contract anchor becomes unit-anchored by
    §2.4's rule, and an item whose `provider` matches no listed unit is invalid;
  - compute the §4.4 record;
  - merge duplicates within the writer.
- **`validate`.** Run §7.1's V1–V4 and write `work/invalid.jsonl` and `work/out-of-scope.jsonl`.
  Apply the method owner's fix/drop decisions by item id.
- **`match`.** Match on `itemId` (§6.2), and write `work/match.json` with each record's capture
  history.
- **`nearmiss`.** Form the N1/N2/N3 pairs (§6.3) and write `work/nearmiss.jsonl`.
- **`scope`.** After adjudication, run V4 per item (§7.1), and set each item's `counting` flag
  (§3.7).
- **`adjudicate --verdicts <file>`.** Check the consistency rule (§6.3), apply the verdicts, and
  write `work/adjudication.jsonl`, then re-run `match`.
- **Library functions (v1.2, P2-2).** `match_keys.py` exports `E(q)` (the one used by
  `normalize`), `h1_symbol_credit(x, e, lines)` (§5.2 H1 for a symbol: `E(qname(x)) = qname(e)`
  and some line in *x*'s span or owned regions) and `source_check_needed(e, lines, symbols)`
  (§5.4's trigger). They are pure functions over plain dicts. P1's gate harness imports them and
  never reimplements them, as with `ids.py`.
- **`sample`.** Draw §7.3's spot-check samples and write `work/spot.jsonl` (union) and
  `work/ptgold.jsonl`, plus `work/engold.jsonl` for the method check. These are for the
  stakeholder and the method owner, read in place.
- **Test oracle (S49; v1.2, P2-2)** for the library functions. Input: class `C` in `a.py` with a
  getter `C.x` (lines 10–12) and a same-scope redefinition `C.x#2`, the setter (lines 14–16),
  which holds a nested `C.x#2.check` (line 15).
  - **(P2-2a) duplicate getter/setter.** For key *e* = `a.py::C.x` with lines [15], listing only
    the getter `C.x` gives **no credit**. Listing the setter `C.x#2` **credits** it.
  - **(P2-2b) a plain element still hits.** For key `a.py::C.x` with lines [11], listing the
    getter `C.x` credits it, so dropping the id-only branch loses nothing.
  - **(P2-2c) no correction back to an ineligible id.** For key `a.py::C.x.check` with lines [15]
    and a snapshot holding `C.x#2.check`, `source_check_needed` is **false**. With a snapshot
    holding no symbol whose `E` is `a.py::C.x.check`, it is **true**.
  - **(P2-2d) a line outside every qualifying span.** For key `a.py::C.x` with lines [30], with
    `C.x` and `C.x#2` both in the snapshot, `source_check_needed` is **true**: it is §5.4's span
    case, never a silent miss.

### 13.5 `capture.py` (S49; run in S45)

- For each derived stratum, build lists P and C and the per-session histories from
  `work/match.json`, then compute §6.4 exactly as written, in floating point with no rounding
  before output.
- **Output:** `work/capture.json`, with the §6.6 counts plus N̂c, Var, the Chapman interval, N̂h,
  N̂*, which estimate binds, completeness and `trigger: true | false` for each stratum, and the
  pooled figures with no interval (§6.4).
- **Test oracle (S49):** reproduce this note's Appendix A outputs for the same inputs to
  `1e-9`. The four-row table in §6.4, the three-list example, and the degenerate case
  `n₂ = m` (interval `[r, r]`) are the fixtures.

### 13.6 `split_manifest.py` (S49; run in S47)

- **`apply-corrections`:** apply S46's marks and §7.4's escalation outcomes, and log each.
- **`ambiguity`:** run §10.3's join, writing the final labels and the dropped counts.
- **`split`:** run §9.1 for AC-5, with clusters, the terminating minimum loop and the reserve,
  and §10.2 for AC-24. Write the held-out, reserve and dev files (§9.2).
- **`regate-list --fresh <file>`** (from P1): route fresh items by cluster (§12.1 step 4), check
  the composition rule and the floor, and write the re-gate list with a per-item flag
  (`kept`, `kept-cluster-fresh` or `fresh-cluster`).
- **`exposure --gate <n> --results <file>`** (from P1): append the first exposures to
  `heldout/exposure.jsonl`, and compute §12.1 step 6's three figures.
- **`manifest`:** write §9.3. **`--verify`** re-checks it, and fails on a one-byte change.
- **`--check-report <path>`:** run §9.4.
- **Test oracle (S49):** on a synthetic union of about 30 anchors across all five strata, with a
  contact set and a deficit that forces step 4, the split equals a hand-computed assignment. The
  fixtures add:
  - **(B1a)** a unit anchor whose probe contract point has its own items: both land on the same
    side, and contacting the probe sends both to dev;
  - **(B1b)** a fresh item on a dev anchor, and one on a C anchor: both go to dev. A fresh item on a
    kept cluster is flagged `kept-cluster-fresh`, and does not count toward the floor;
  - **(M2)** `flow` stuck below 8 with no dev cluster holding flow items, and another stratum short
    with a dev cluster available: the loop marks `flow`, then moves the other cluster, and
    terminates;
  - **(M6)** a union whose all-items count is ≥ 75 but whose counting count is < 75: it triggers;
  - **the reserve:** it never takes gate 1 below its minimums;
  - **the leak check:** a planted held-out `qname` in a report fails the check; a contacted
    anchor's `qname` in the report does not; and a dev `a.py::f.g` neither removes the held-out
    probe `a.py::f` nor hits on it (m4).
  - **(P2-1a) a unit with a dev non-probe tool ends in dev.** Unit *u* serves probe T1 and T2,
    whose `provider` lies under *u*'s `rootPath`, and T2 is in dev, by contact or by rank. *u*'s
    cluster (with T1) ends in dev, logged `served-anchor`. With T2 held out instead, *u* follows
    its own rank.
  - **(P2-1b) a single-tool server is unaffected.** Unit *v* whose only served anchor is its
    probe: the rule never fires.
  - **(P2-1c) step 4 respects the rule.** `wiring` is short, and *u*'s dev cluster is the only one
    holding wiring items, while T2 is in dev. Step 4 does not move *u*: it marks `wiring`, and
    step 5 triggers.
  - **(P2-1d) re-gates respect it.** After a miss moves T2's cluster to dev, a kept *u* and a
    reserve *u′* (both serving T2) move to dev, and neither counts as a miss. A fresh item on *u*
    is routed to dev.
  - **(P2-1e) probe choice.** Of *u*'s served points with 0, 3 and 3 items, the 0-item one is the
    probe. With 3 and 3 items, the smaller `cpId` is.
  - **(P2-3) trailing punctuation.** A report containing `held-out item a.py::f.` fails the check,
    and so does one containing `a.py::f: missed`, for held-out probe `a.py::f`. One containing
    only `a.py::f.g` passes. The other cases listed in §9.4 are fixtures too.

## 14. What each downstream step takes from this note

The "changes" column is measured against `docs/plans/connected-code-graph-p0-2.md` v1.0. Rows
marked **v1.1** or **v1.2** are new since v1.0, and its §14 needs to absorb them.

| Step | Takes | Changes against the plan as written |
|---|---|---|
| **S33** (answer spec) | §5.2's hit rule: groups enumerate members with their own chains; each item carries a span-bearing symbol id and its `chain`; `stamp.verdictLogOffset`. **v1.1:** interpreted symbol anchors carry `qname`, so §10.4 can apply `E` | none beyond the `qname` field |
| **S35** (fixtures) | §12.2's minimums. **v1.1:** the AC-25 presentation assertions are S35's (m6) | **v1.1:** an assertion column for AC-25's four shape properties |
| **S40** (kit, briefs) | §13.1; **Appendix B pasted verbatim** into each `BRIEF.md`, and nothing else from this note; the flow-anchor list; `eval-manifest.yaml`'s fields | five strata. **v1.1:** `eval-manifest.yaml` gains `units`, `cypherMcpTools` and the grain table. The briefs paste Appendix B instead of §3/§4 (M3). **v1.2:** each unit carries `ports`, filled by S40 from source (§2.4); the grain table gains the Kiro `toolsSettings`/`includeMcpJson`/fallback rows (P2-4). |
| **S41/S42** (KA, KB) | Appendix B | 10 sessions instead of 8 (plan §6.6: base total 16–18, expected about 18–25 with up to 5 extra passes) |
| **S43/S43a/S44** (QW, AN) | §10.1's 48 questions; §13.2; Appendix B's QW and AN parts | none |
| **S45** | §6, §7.1–§7.3, §13.4–§13.5 | **v1.1:** a `scope` verb; N3 pairs |
| **S46** (stakeholder) | §7.3's 40 + 8 sample | none |
| **S47** | §7.4, §8, §9, §10.2–§10.3, §13.6 | **v1.1:** probe clusters; the counting count `H` beside `H_all`; the reserve file; an empty `exposure.jsonl`. **v1.2:** the probe is the served point with the fewest items (§5.1); step 3b's served-anchor rule and step 4's eligibility guard (§9.1); S47's report gives the `served-anchor` moves as a count (P2-1) |
| **S48** | §13.1–§13.3. `export_kit.py --draw-flow-anchors` needs `ast` and py-tree-sitter from the **`spike` extra**, not S02's image (m5) | **v1.1:** `--list-units` and `--list-cypher-tools`. Correct S48's "or S02's image" wording. |
| **S49** | §13.4–§13.6, with Appendix A as the oracle for `capture.py` | **v1.1:** the B1a, B1b, M2, M6, reserve and m4 oracle cases; `regate-list` and `exposure` verbs, needed from P1 (they may be built in P1). **v1.2:** the P2-1a–e, P2-2a–d and P2-3 oracle cases (§13.4, §13.6); `match_keys.py` exports `E`, `h1_symbol_credit` and `source_check_needed` (§13.4); `split` implements step 3b and the step 4 guard; `regate-list` re-applies step 3b and routes by S(u) (§12.1); `--check-report` uses §9.4's v1.2 pattern |
| **S52** (agent definitions) | **Appendix B verbatim**, and bundle spec §4 and §6 | **v1.1:** quote Appendix B, **not** "§4.1/§4.2/§4.3 verbatim", which would carry JSON examples (M3) |
| **S14 / S30** | §8.2's definition of "touched" | the lists must give **anchors** as locators, with the hops excluded. **v1.2 (S30):** take the server-binding walkthrough on `falkor-chat/server`, which the served-anchor rule sends to dev anyway, or on S35's fixture, and **not** on `cypher-mcp` or `mcp-monitor`. Walking `.mcp.json` → `cypher` costs `wiring` 3 of its at most 10 eligible items, and leaves it short of 8 (§9.1; §15 Q10) |
| **S56** (golden flow methods) | §8.2: its methods are contact for flow anchors | **v1.1:** S56's output is an input to S47's contact set |
| **P1 gate harness** | §5, §11, §12.1 (floor, routing, exposure record), §12.3 (`coreTreeHash`) | **v1.1:** the floor, the exposure record and the four-field pre-registration comparison. **v1.2:** H1 always requires the span test, and §5.4 runs under `E`; both are imported from `match_keys.py` (P2-2) |

## 15. Risks and open questions

### 15.1 Risks

| # | Risk | Effect | Mitigation |
|---|---|---|---|
| R1 | The writers share blind spots (G6) | Completeness is overstated, so AC-5 recall is overstated | 0.95 threshold; labelled as an upper bound; runtime oracle at P1; each oracle miss feeds the next held-out set (ML §4.1) |
| R2 | The census is small after contact (§8) | SQ-P0-1 fires | `wiring` adopted up front (E2); §9.1 step 4 top-up; then plan §12's ordered options |
| R3 | KA and KB differ in stratum reading | `m` falls and spurious extra passes run | The stratum is derived (§3.1); per-tool bindings are re-anchored (§3.5); N1/N2 near-misses |
| R4 | TS/JS `qname` slips survive V1 | A spurious "miss" at the gate | §5.4's source-only correction rule |
| R5 | E1's spurious credit (another lambda of the same element whose span holds the line) | A slightly optimistic hit | The span-contains-line test (§5.2 H1); the residue is reported as a known limit |
| R6 | A held-out item is read by a builder (D11, K-P0-7) | A silent leak | Accepted by the stakeholder decision. Builder briefs never name the path. This note adds no mechanism. |
| R7 | Fresh items are scarce in the proving repo (§12.1) | A re-gate passes on every item but cannot count as an AC-5 pass | The reserve; a top-up census; a new commit; then Q7 |
| R8 | The reserve shrinks gate 1's list | Gate 1 has less first-gate evidence | The reserve takes only items beyond gate 1's minimums, and at most 35 |
| R9 | The served-anchor rule (P2-1) leaves `wiring` short. At the working tree its eligible supply is at most 10, or 7 if S30 contacts `cypher-mcp` | `wiring` < 8, so SQ-P0-1 fires at S47 | S30 walks the server binding on `falkor-chat/server` or on the fixture (§14, Q10); otherwise the existing step 4 → step 5 → SQ-P0-1 path |

### 15.2 Open questions

- **Q1 (`architect`, P1 plan):**
  - P1's shipped default depth bound must be **≥ 2**, or `flow`'s distance-2 items fail by
    construction. That is a correct measurement (ML m5), but it is avoidable. Please confirm or
    record the bound.
  - **v1.1:** AC-24 scoring maps interpreted symbol anchors through `E` (§10.4), so S33's
    interpretation block must carry each resolved symbol's `qname`.
- **Q2 (`architect`, bundle spec):** §3.5 lists two locator types. This note uses four: the two in
  §3.5, plus `config` (§3.2's `entryId` tuple) and `unit` (§3.2's `unitId`), both converted by
  §3.2. A one-row addition to §3.5 would make the bundle spec the single source again. It is not
  blocking, because the conversion is already defined.
- **Q3 (`architect`, P1):** P1's prompt extractor must classify prompt-bearing Markdown by §3.3's
  glob list, or census scope ≠ evaluated scope. One option is to read that list from
  `eval-manifest.yaml`.
- **Q4: closed in v1.1.** `pyyaml==6.0.3` is in `cpg/pyproject.toml`'s `dev` extra (§7.1).
- **Q5: closed by the stakeholder (2026-10-29).** §12.1's reading stands, with the fresh-item
  floor added (E12).
- **Q7 (`teco`, stakeholder, only if it occurs):** if the reserve, a same-commit top-up and a newly
  pinned commit together cannot supply 35 fresh-cluster items for a re-gate, which option applies
  (§12.1, source 4)?
- **Q8 (`architect`):** `p0-2.md` §12–§15 are placeholders, so E7, §9.1 step 5 and §12.1's
  source 4 point at SQ-P0-1 text that only the superseded plan holds. Restore §12 in the
  successor.
- **Q9 (`teco`, stakeholder, only if it occurs):** E11 means flow will mostly not count toward the
  75. If S47's counting `H` falls below 75 while `H_all` reaches it, SQ-P0-1 fires as usual. That
  is the decided rule working, not a new question.
- **Q10 (`architect` for S30, via `teco`; v1.2):** take S30's server-binding walkthrough on
  `falkor-chat/server`, or on S35's fixture, rather than on `.mcp.json` → `cypher`, the example
  §8.2 and G1 give. Under the served-anchor rule (§9.1) the falkor-chat binding is dev anyway, so
  walking it costs nothing. Walking cypher-mcp costs `wiring` 3 items, and on the working-tree
  inventory that leaves it at 7, short of 8. If S30 has already been written on cypher, nothing
  breaks: SQ-P0-1 fires at S47, as R9 says.
- **Q6 (`teco`):** E2 adds one KA and one KB session, and E5 allows up to 5 extra passes rather
  than 4. That is §6.6's usage budget, already stakeholder-accepted in kind. The count changes by
  about 2 to 3 sessions.

## 16. Review dispositions (`docs/reviews/connected-code-graph-p0-impl.md`, §S06 (P0-05))

### Pass 1 (needs changes) → v1.1

| Finding | Disposition | Where |
|---|---|---|
| **B1(a)** the unit probe straddles the split | **Fixed as recommended.** A unit and its probe are one union-find cluster, the probe is fixed before the split, and contacting either contacts both. | §5.1, §8.2, §9.1 step 1 |
| **B1(b)** fresh items on a dev anchor | **Fixed, and extended.** Fresh items are routed by cluster: dev or C goes to dev. Also, a fresh item on a **kept** cluster joins it but does not count as fresh evidence, because its anchor passed, which mildly conditions it. The review did not raise that third case. | §12.1 step 4 |
| **M1** a re-gate overstates the evidence | **Fixed.** The reporting fix is adopted in full: fresh-cluster recall, the cumulative first-exposure record, kept items as a regression check, and 0.951 claimed only at gate 1. The **floor** follows the stakeholder decision: F = 35, sized and costed. The same-commit shortfall is specified (the four sources, then Q7). Added a **reserve** so the first re-gate usually needs no census. The exact DP reproduces `regate.py`. | §5.3, §9.1 step 6, §12.1 |
| **M2** the loop does not terminate | **Fixed as recommended.** Marked strata are excluded from the test and the selection, and the termination argument is stated. The `H < 75` branch now selects only clusters with counting items. | §9.1 step 4 |
| **M3** writer-facing text, `config` grain | **Fixed as recommended.** Appendix B is the only text S40 and S52 paste. §2.4 gives a per-format grain table with server/contract classes, checked against the working tree, and `normalize` truncates to it. | §2.4, §4, §13.4, Appendix B |
| **M4** E1 penalizes a precise interpreter on AC-24 | **Fixed with the first option.** `E` is applied before the set comparison. The second option, constraining QW, was rejected: it would remove realistic file-and-line questions. | §10.4, §15 Q1 |
| **M5** the core is not frozen | **Fixed as recommended, plus one exclusion choice.** `coreTreeHash` excludes `cpg/eval/`, `cpg/packs/` (covered separately), `cpg/tests/` and `*.md`. The four-field comparison decides "first contact" vs "after adaptation". | §12.3 |
| **M6** flow and the 75 | **Fixed per the stakeholder decision.** Flow counts only across units or repos, by endpoints. Cross-boundary is defined per stratum. `H` and `H_all` are reported throughout, and the per-stratum 8 uses all items. | §3.7, §9.1, §5.3 |
| **m1** stratum and scope depend on the writer | **Fixed.** `cypherMcpTools` comes from source, V4 runs per item after matching, and a field-precedence rule is stated. | §3.1, §6.1, §7.1, §13.1 |
| **m2** the interval vs N̂* | **Fixed by labelling.** It is the "Chapman interval". When Chao binds, no interval is given, because a delta-method Chao variance would be a method choice for GC. No pooled interval. Decisions use the point N̂* only. | §6.4 |
| **m3** unit anchors under-specified | **Fixed.** Units are listed by S40 (`--list-units`, which includes `requirements.txt`), N3 is added, and re-anchoring is mechanical through §2.4's classes and `provider`'s path. | §2.2, §2.4, §6.3, §13.1 |
| **m4** probe removal is ambiguous | **Fixed as recommended.** Probes are removed on exact equality, and matched as whole tokens. | §9.4 |
| **m5** grounding | **Fixed.** tree-sitter comes from the `spike` extra, not S02. `pyyaml` is in `dev`, which closes Q4. The golden flow methods are produced by S56. | §3.6, §7.1, §8.2, §13.1, §14 |
| **m6** AC-25's shape has no owner | **Fixed.** S35's assertion column owns it, and S39 tests it. | §12.2 |
| nit: `p_trigger` docstring | Fixed. | Appendix A |
| nit: §10.2's dev count | Fixed: 3 per cell, 12 with no drops. §10.1 and E8 are aligned. | §0, §10.1, §10.2 |
| nit: §7.3 wrap | Fixed: the order wraps from `flow` to `http`. | §7.3 |
| Gap: `p0-2.md` §12–§15 placeholders | Routed to `architect`. | §15 Q8 |
| Found while writing Appendix B: v1.0's `E` truncated at `#k`, while a writer told never to write `#2` writes `f` | Fixed: `E` removes `#k` suffixes instead, and V2 compares under `E`. | §2.3, §7.1 |

### Pass 2 (approve with suggestions) → v1.2

| Finding | Disposition | Where |
|---|---|---|
| **P2-1** binding entries leak through a unit's non-probe tools | **Fixed with the reviewer's direction, refined.** The served-anchor rule: a unit is held out only while every anchor of S(u) is (S(u) = provider under the unit by longest prefix, or in `serves[]`; all contract kinds). It is applied as a fixpoint after the share, as a guard in step 4, and again in re-gate steps 3–4. The probe is now the served point with the fewest items. Whole-server clusters were rejected: they merge about 25 `mcp` anchors and collapse `mcp` on any contact. Run: v1.1 leaks in 100% of 2,000 seeds, v1.2 in 0%. `H` changes by −3.0 to −4.2. `H_wiring`'s eligible supply is 10, or 7 if S30 contacts cypher-mcp, which SQ-P0-1 then handles; R9, Q10 and the S30 row cover it. | §3.5, §5.1, §8.2, §9.1, §12.1, §13.6, §14, §15 |
| **P2-2** `#k` removal lets the id branch skip the span test | **Fixed as recommended.** H1 requires `E(qname(x)) = qname(e)` **and** the span test, with no id-only branch. §5.4's trigger and source comparison run under `E`, a correction is always eligible, and §5.4 gains a span case for wrong lines. AC-24's indistinguishable duplicates are recorded as a known limit. The exposure scan finds 0 duplicates in 1,756 Python scopes. | §0, §5.2, §5.4, §10.4, §13.4 |
| **P2-3** trailing `.`/`:` defeat the whole-token match | **Fixed as recommended, plus a `./` left prefix.** The lookahead allows `[.:]*`. `a.py::f.` and `a.py::f:` hit, and `a.py::f.g` does not. 15/15 cases came out as wanted. | §9.4, §13.6 |
| **P2-4** nit: `.env` classification needs unit ports | **Fixed.** `units[].ports`, recorded by S40 from source and checked by the method owner. No working-tree `.env` value matches a unit port. | §2.4, §13.1, Appendix B |
| **P2-4** nit: Kiro keys outside the row | **Fixed.** `toolsSettings.<key>` is contract class. `includeMcpJson` is server class, only when it is `true`. Any other key falls to "any other JSON". | §2.4, Appendix B |
| M5 nit (`coreTreeHash` excludes `**/*.md`) | **Not changed (outside this round).** It is recorded here as a P1 condition: narrow the exclusion before the first pre-registration if P1 keeps interpreter prompts as `.md` under `cpg/`. | — |

## Appendix A. Figures script

Every figure this note labels as computed comes from this script: the Wilson bounds, the P(pass)
values, the exact `ceil` counts, the split targets, the Chapman and Chao examples with their
intervals, the three-list example, the heterogeneity illustration, the exact P(trigger) tables,
and (v1.1) the re-gate floor DP. It uses the Python 3.12 standard library only. Run it with
`python3 -I ep.py`; it takes under a second.

My guard allows writes only under `docs/`, so the script is kept here. If `teco` wants it as a
file, `cpg/eval/scripts/ep.py` is the suggested home, with this appendix as its source.
`capture.py`'s tests (S49) use its outputs as the oracle (§13.5).

The script prints exactly the following for the figures cited:
- §5.3: `75/75: [0.951, 1.000]`, `8/8: [0.676, 1.000]`, `0.99 0.471`;
- §6.4: the four Chapman/Chao rows; `r=45 … Chapman 47.0 … ChaoMh(t=3) 48.1 … completeness 0.935`;
  and `true completeness 0.860, estimated 0.975`;
- §6.5: the P(trigger) rows at 0.90 and 0.95;
- §7.3: `0/40 … upper 0.088`, `1/40 … 0.129`, `2/40 … 0.165`, `0/8 … 0.324`;
- §7.4: `2/8 … [0.071, 0.591]`;
- §9.1: `8 -> 6`, `20 -> 14`, `30 -> 21`, `40 -> 28`;
- §10.5: `pooled 24 @0.92 -> 23`, `per-language 12 @0.85 -> 11`, `23/24: [0.798, 0.993]`,
  `24/24: [0.862, 1.000]` and `11/12: [0.646, 0.985]`.
- §12.1 (v1.1): `9*z^2 = 34.57`; `F=20/35/50/75: Wilson LB … 0.839/0.901/0.929/0.951`; the five
  `F=…: P(pass by gate 1/2/3)` rows; `108/110: [0.936, 0.995]`; `yield … 7.9`. The `regate` DP
  at F = 0 reproduces the gate's `regate.py` Monte Carlo figures (0.985 and 0.696 against
  0.985 and 0.694).

```python
"""S06 eval-protocol figures. Python 3.12 stdlib only. Run: python3 -I ep.py
Every figure in docs/plans/connected-code-graph-eval-protocol-ml.md marked 'computed' comes from here."""
from math import comb, sqrt, log, exp, factorial

Z = 1.959963984540054          # two-sided 95%

def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h

def ceil_frac(n, num, den):    # exact ceil(n*num/den)
    return -(-n * num // den)

def chapman(n1, n2, m):
    N = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    var = (n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m) / ((m + 1) ** 2 * (m + 2))
    return N, var

def chao_mh(r, f1, f2, t):     # bias-corrected Chao lower-bound estimator, t lists
    return r + ((t - 1) / t) * f1 * (f1 - 1) / (2 * (f2 + 1))

def lognormal_ci(r, N, var):   # Chao (1987) log-normal interval on f0 = N - r
    f0 = N - r
    if f0 <= 0 or var <= 0:
        return r, r
    C = exp(Z * sqrt(log(1 + var / f0 ** 2)))
    return r + f0 / C, r + f0 * C

def nstar(n1, n2, m):          # two lists: max(Chapman, Chao Mh t=2)
    r = n1 + n2 - m
    Nc, _ = chapman(n1, n2, m)
    Nh = chao_mh(r, n1 + n2 - 2 * m, m, 2)
    return r, max(Nc, Nh)

def p_trigger(N, p1, p2, thr_num=9, thr_den=10):
    """Exact P(completeness r/N* < thr_num/thr_den) when N items are captured independently with p1, p2."""
    pa, pb, pc, pd = p1 * p2, p1 * (1 - p2), (1 - p1) * p2, (1 - p1) * (1 - p2)
    tot = 0.0
    for a in range(N + 1):
        for b in range(N - a + 1):
            for c in range(N - a - b + 1):
                d = N - a - b - c
                pr = (factorial(N) / (factorial(a) * factorial(b) * factorial(c) * factorial(d))
                      * pa ** a * pb ** b * pc ** c * pd ** d)
                n1, n2, m = a + b, a + c, a
                r = a + b + c
                if r == 0:
                    tot += pr; continue
                _, Ns = nstar(n1, n2, m)
                if r * thr_den < thr_num * Ns:
                    tot += pr
    return tot

def regate(p, F, c=2, n=76, gates=3):
    """v1.1 fresh-item floor (§12.1). Exact DP over the missed-cluster count. Model (the gate's regate.py):
    gate 1 = n items in clusters of c; kept items pass again; each re-gate's fresh clusters = max(missed, ceil(F/c));
    true recall p constant (fixes do not generalize). Returns cumulative P(pass) by gate 1..gates."""
    q = p ** c; fa = -(-F // c)
    dist = {n // c: 1.0}; acc = 0.0; out = []
    for _ in range(gates):
        nd = {}
        for f, w in dist.items():
            acc += w * q ** f
            for M in range(1, f + 1):
                nf = max(M, fa)
                nd[nf] = nd.get(nf, 0.0) + w * comb(f, M) * (1 - q) ** M * q ** (f - M)
        out.append(round(acc, 3)); dist = nd
    return out

if __name__ == "__main__":
    print("== Wilson, zero-miss / small-n bounds ==")
    for k, n in [(8, 8), (75, 75), (74, 75), (24, 24), (23, 24), (12, 12), (11, 12), (8, 8)]:
        lo, hi = wilson(k, n); print(f"{k}/{n}: [{lo:.3f}, {hi:.3f}]")
    print("== Spot-check error-rate upper bound (Wilson) ==")
    for k, n in [(0, 40), (1, 40), (2, 40), (3, 40), (0, 8), (1, 8), (2, 8), (0, 16)]:
        lo, hi = wilson(k, n); print(f"{k}/{n} incorrect: upper {hi:.3f}  (lower {lo:.3f})")
    print("== P(first-run zero-miss pass), n=75 ==")
    for r in (0.98, 0.99, 0.995, 0.999):
        print(r, round(r ** 75, 3))
    print("== AC-24 P1 smoke pass counts (exact ceil) ==")
    print("pooled 24 @0.92 ->", ceil_frac(24, 92, 100), " per-language 12 @0.85 ->", ceil_frac(12, 85, 100))
    print("== Split targets, share 0.70 (exact ceil) ==")
    for n in (8, 10, 12, 20, 30, 40):
        print(n, "->", ceil_frac(n, 7, 10))
    print("== Chapman / Chao examples ==")
    for n1, n2, m in [(40, 35, 28), (30, 25, 15), (24, 20, 12), (12, 10, 8), (10, 9, 3), (20, 20, 0)]:
        r = n1 + n2 - m
        Nc, var = chapman(n1, n2, m)
        Nh = chao_mh(r, n1 + n2 - 2 * m, m, 2)
        lo, hi = lognormal_ci(r, Nc, var)
        Ns = max(Nc, Nh)
        print(f"n1={n1} n2={n2} m={m} r={r}: Chapman {Nc:.1f} [{lo:.1f}, {hi:.1f}]  ChaoMh {Nh:.1f}  completeness r/N* {r/Ns:.3f}")
    print("== Heterogeneity illustration at expected counts (not a sample) ==")
    # 80% easy items (p1=.9, p2=.85), 20% hard items (p1=p2=.2), N=60
    N = 60; Ne, Nh_ = 48, 12
    e1, e2 = 0.9, 0.85; h1, h2 = 0.2, 0.2
    n1 = Ne * e1 + Nh_ * h1; n2 = Ne * e2 + Nh_ * h2; m = Ne * e1 * e2 + Nh_ * h1 * h2
    r = n1 + n2 - m
    Nc, _ = chapman(n1, n2, m)
    Nchao = chao_mh(r, n1 + n2 - 2 * m, m, 2)
    print(f"true N=60: E[n1]={n1:.1f} E[n2]={n2:.1f} E[m]={m:.2f} E[r]={r:.1f}; Chapman {Nc:.1f}, ChaoMh {Nchao:.1f};"
          f" true completeness {r/N:.3f}, estimated {r/max(Nc, Nchao):.3f}")
    print("== P(trigger: completeness < 0.90), homogeneous independent capture, exact ==")
    for N in (20, 40):
        for p1, p2 in [(0.95, 0.9), (0.9, 0.8), (0.8, 0.7), (0.7, 0.6), (0.6, 0.5)]:
            true_c = 1 - (1 - p1) * (1 - p2)
            print(f"N={N} p1={p1} p2={p2} true E[completeness]={true_c:.3f}  P(trigger)={p_trigger(N, p1, p2):.3f}")
    print("== P(trigger) at threshold 0.95 ==")
    for N in (20, 40):
        for p1, p2 in [(0.95, 0.9), (0.9, 0.8), (0.8, 0.7), (0.7, 0.6)]:
            print(f"N={N} p1={p1} p2={p2}  P(trigger@0.95)={p_trigger(N, p1, p2, 95, 100):.3f}")
    print("== Multi-list example (t=3: KA, KA2 provider-first; KB consumer-first) ==")
    # capture histories as counts: (KA, KA2, KB) -> count
    hist = {(1,1,1): 20, (1,1,0): 6, (1,0,1): 4, (0,1,1): 3, (1,0,0): 5, (0,1,0): 3, (0,0,1): 4}
    r = sum(hist.values())
    P = sum(v for k, v in hist.items() if k[0] or k[1]); C = sum(v for k, v in hist.items() if k[2])
    m = sum(v for k, v in hist.items() if (k[0] or k[1]) and k[2])
    Nc, var = chapman(P, C, m)
    f = {1: 0, 2: 0, 3: 0}
    for k, v in hist.items(): f[sum(k)] += v
    Nh = chao_mh(r, f[1], f[2], 3)
    print(f"r={r} collapsed n1={P} n2={C} m={m}: Chapman {Nc:.1f}; f1={f[1]} f2={f[2]} f3={f[3]}: ChaoMh(t=3) {Nh:.1f};"
          f" N*={max(Nc, Nh):.1f} completeness {r/max(Nc, Nh):.3f}")
    print("== v1.1: re-gate fresh-item floor (§12.1) ==")
    print("min F with all-hit Wilson LB >= 0.90: 9*z^2 =", round(9 * Z * Z, 2))
    for F in (20, 35, 50, 75):
        print(f"F={F}: Wilson LB at F/F = {wilson(F, F)[0]:.3f}")
    for F in (0, 20, 35, 50, 75):
        print(f"F={F}: P(pass by gate 1/2/3)", [(p, regate(p, F)) for p in (0.99, 0.97, 0.95)])
    lo, hi = wilson(108, 110); print(f"cumulative example 73/75 + 35/35 = 108/110: [{lo:.3f}, {hi:.3f}]")
    print("same-commit top-up yield r*(1/c-1), r=150, c=0.95:", round(150 * (1 / 0.95 - 1), 1))
```

## Appendix B. Writer-facing text (S40 and S52 paste this verbatim; M3)

**Rules for the pasters.**
- S40 pastes B.0 plus the role's part (B.1, B.2 or B.3) into each `BRIEF.md`, and S52 pastes the
  same parts into the matching agent definition.
- Each paster fills only the `{…}` slots, from `eval-manifest.yaml` and S40's draw.
- **The `{grammarExtract}` slot** takes the bundle spec's §4.0, §4.1 rules 1–4 and 6–10, and the
  rule bullets of §4.2 for Python, TS/JS, shell and Markdown, plus §6's `http` and `mcp_tool`
  rows. It **omits** §4.1 rule 5, every worked-example table, and §6's example column. Those name
  real proving-repo elements, the slice's among them.
- Nothing else from this note is pasted. The text below names no proving-repo element and gives no
  rationale. Its examples are synthetic paths that do not exist in the kit.

---

### B.0 Common to every role

**Where you work.** `kit-src/` is a read-only copy of one repository. Everything you record refers
to the files exactly as they are in `kit-src/`. The repository's id is `graphmind-ai-lab`. Write
your output only to `{outputPath}`, one JSON object per line, appending as you go. If you are
restarted, read your output file first and continue from where it ends.

**What is out of scope.** Do not record anything in:
- test code: `**/tests/**`, `**/test_*.py`, `**/*_test.py`, `**/conftest.py`,
  `**/*.test.{ts,tsx,js,jsx}` and `**/*.spec.{ts,tsx,js,jsx}`. A test script that a non-test
  script or hook runs is not test code;
- Dockerfiles;
- vendored code;
- Markdown files other than those listed under "prompt files" below.

**Prompt files.** These are Markdown files read by agents as instructions:
- `**/AGENTS.md` and `**/CLAUDE.md`;
- `claude/**/*.md`, except `claude/**/kaizen/**` and `claude/README.md`;
- `skills/**/*.md`, except `skills/README.md`;
- `opencode/agents/**/*.md`, `opencode/skills/**/*.md` and `.kiro/steering/*.md`;
- `.claude/commands/**/*.md` and `.claude/agents/**/*.md`.

Every other Markdown file is documentation. A mention in documentation is never a dependency.

**How to name things.** Use exactly one of these forms.

| Form | Fields | Use it for |
|---|---|---|
| `symbol` | `{"type":"symbol","repoId":"graphmind-ai-lab","path":…,"qname":…,"signature"?:…}` | a function, method, class, or a file's top-level code |
| `contract` | `{"type":"contract","kind":"http"\|"mcp_tool","keyNorm":…}` | an HTTP route or an MCP tool |
| `config` | `{"type":"config","repoId":"graphmind-ai-lab","path":…,"keyPath":…}` | an entry in a configuration file |
| `unit` | `{"type":"unit","repoId":"graphmind-ai-lab","rootPath":…,"lang":…}` | a whole server, chosen **only** from this list, which gives each server's ports: `{units}` |

- **`path`** is relative to the repository root, with `/` separators.
- **`qname`** is `path + "::" + the chain of enclosing named scopes`, joined by `.`. Examples:
  `pkg/app.py::Router.list_items` and `web/src/api.ts::loadItems`. The full grammar for each
  language is quoted below (bundle spec §4).
  - A file's top-level code is `path::<module>`. For a shell script it is `path::<script>`, and
    for a prompt file `path::<document>`.
- **Naming rule for code inside a lambda, callback, arrow function or anonymous class:** name the
  nearest enclosing **named** function, method or class. If there is none, name the file's
  top-level symbol. **Never write `<anon>` or a `#2`-style suffix, and never count lambdas.**
  Always give the line numbers.
- **`keyNorm` for `http`:** `<VERB> <path>`. The verb is uppercase. The path is the full path the
  server serves, mount prefixes included, with every path parameter written `{}`. Example:
  `GET /api/items/{}`. **For `mcp_tool`:** the bare tool name, without any `mcp__<server>__`
  prefix.
- **`keyPath` for `config`:** stop at the level this table gives for the file. Do not go deeper.

  | File | Write the `keyPath` at |
  |---|---|
  | `.mcp.json` | `mcpServers.<name>` |
  | Kiro agent JSON (`.kiro/agents/*.json`) | `mcpServers.<name>`; `includeMcpJson`, only when it is `true`; or `tools[i]`, `allowedTools[i]`, `toolAliases.<key>`, `toolsSettings.<key>`; any other key: the first two keys |
  | `.claude/settings*.json` | `enabledMcpjsonServers[i]`; `permissions.allow[i]`, `permissions.ask[i]`, `permissions.deny[i]`; `hooks.<Event>[i]` |
  | `mcp-monitor` TOML | `server.<name>`; `watch[i]` |
  | Compose files | `services.<svc>` |
  | `.env` files | the variable name |
  | any other JSON, YAML or TOML | the first two keys, where an `[i]` index counts as one key |

  `[i]` is the 0-based array position.
- **A config entry that launches a server, or points a client at one, is recorded once per
  server**, with a `unit` anchor. This covers `mcpServers.<name>`, `server.<name>`,
  `services.<svc>`, `enabledMcpjsonServers[i]`, a Kiro `includeMcpJson` set to `true` (once per
  server it brings in), and a `.env` variable whose value is a URL or port of a listed unit (each
  unit's `ports` are in the list above). A config entry that names one specific tool or route is
  recorded with that tool's or route's `contract` anchor.

**The `qname` and `keyNorm` grammar.** B.0's naming rule takes precedence over anything below.

{grammarExtract}

---

### B.1 Key writers (KA, KB)

**Your stratum for this session: `{stratum}`.** Search it as completely as you can. If you come
across a dependency of another kind, record it as well.

**What one item is.** An item says: changing the **anchor** affects the **impacted element** in at
least one way the repository can actually be deployed or run. Record each (anchor, impacted
element) pair once. Several call sites inside one function are one item: put all their lines in
`lines`.

**The five kinds:**
- **`http`.** The anchor is an HTTP route (`contract`, `kind: http`). The impacted element is the
  function that names the route's path, by literal or by constant, and sends the request, directly
  or through a helper. A helper that only passes a path it was given is not the impacted element.
  Also record `provider`: the route's handler, as a `symbol`.
- **`mcp`.** The anchor is an MCP tool (`contract`, `kind: mcp_tool`). The impacted element is
  code that calls the tool, or a prompt file that tells an agent to use it. Also record
  `provider`: the function that registers the tool.
- **`scripts`.** The anchor is either a script that something runs (its `<script>` or `<module>`
  symbol) or one of these tools: `{cypherMcpTools}`. The impacted element is whatever runs the
  script by its path (a shell command, a Python subprocess call, a prompt file telling an agent to
  run it), or a shell function that names one of those tools.
- **`wiring`.** The impacted element is a `config` entry. The anchor is a `unit`, for an entry that
  launches or points at a server; record `serves` with at least one route or tool that server
  provides. Or the anchor is a `contract`, for an entry naming that one tool or route.
- **`flow`.** The anchors are exactly these functions: `{flowAnchors}`. For each one, record every
  named element whose inputs, or whether it runs at all, change when the anchor's behaviour
  changes, through **at most 2** steps between functions. **At least one** step must be a
  `return-use`, `callee-arg`, `control` or `shared-state` step (below). Callers that merely call
  the anchor do not count.

**Fields of one line:**

| Field | Rule |
|---|---|
| `schema` | `"ccg-key-v1"` |
| `seq` | your running number, from 1 |
| `stratum` | `"{stratum}"`, or the other kind if the item is of another kind |
| `anchor` | a `contract`, `symbol` or `unit`, named by B.0 |
| `provider` | for a `contract` anchor: the handler or registering function, as a `symbol` |
| `serves` | for a `unit` anchor: a list of ≥ 1 `contract`; otherwise `[]` |
| `impacted` | a `symbol` or `config`, named by B.0 |
| `lines` | ≥ 1 line number in `impacted.path` where the dependency is (the call, the mention, the entry) |
| `chain` | the steps **between functions** from anchor to impacted, in order, each `{"kind":…,"from":"path:line","to":"path:line"}` |
| `hops` | the number of steps in `chain` |
| `why` | at most 300 characters, in plain words |
| `deployment` | `"default"`, or the name of the configuration in which the dependency is real (a Compose file, an env profile, an alternative `.mcp.json`) |

**Step kinds:**
- `caller`: the next element calls the current one.
- `return-use`: the next element calls the current one and uses what it returns.
- `callee-arg`: the current element calls the next one, passing a value that depends on the change.
- `control`: a condition that depends on the change decides whether the next element's code runs.
- `shared-state`: one side writes a field, global, store key or file that the other side reads.
- `link`: a route or tool call between programs.
- `config`: a configuration entry binding two sides.

---

### B.2 Question writer (QW)

Write questions as a developer would type them to a tool that maps the impact of a change: partial
names, wrong case, pasted fragments. In Portuguese, keep the code identifiers in English as
developers do. **Write each language natively. Never translate.**

**Counts:**
- for each language (`en`, `pt`), write **9** questions that start from a code location (a function
  or class name, or a file and line), and **9** that start from a route or a tool;
- in each group of 9, at least 2 ask the reverse question ("what does this depend on?");
- for each language, also write **6** ambiguous questions, using each of the four kinds below at
  least once.

**Fields of one line:**

| Field | Rule |
|---|---|
| `schema` | `"ccg-question-v1"` |
| `seq` | your running number, from 1 |
| `lang` | `"en"` or `"pt"` |
| `group` | `"code-location"` or `"contract-point"` |
| `intent` | `"unambiguous"` or `"ambiguous"` |
| `ambiguityKind` | for ambiguous questions: `several-symbols` (the name fits several functions or classes), `unknown-name` (nothing in the repository has that name), `two-use-cases` (it could mean "what does this affect" or "what does this depend on"), or `no-anchor` (it names nothing to start from); otherwise `null` |
| `text` | the question |
| `gold` | for unambiguous questions: `{"useCase":"impact","direction":"impact"\|"depends-on","anchors":[…],"scope":null}`. `anchors` holds B.0 forms (`symbol`, `contract` or `config`), following B.0's naming rule. `scope` is `null`, or `{"repoIds":[…]}` / `{"paths":[…]}` if the question limits where to look. |
| `goldOptions` | for ambiguous questions: a list of objects in `gold`'s form, one per reasonable reading, with at least 2; `[]` for `unknown-name` and `no-anchor`. Otherwise `null`. |

---

### B.3 Annotators (AN1, AN2)

For each line of `QW-nogold.jsonl`, decide whether a careful developer who knows this repository
would need to ask a clarifying question before answering. If so, the question is **ambiguous**.
Ambiguous means one of these: the name fits several functions or classes; nothing has that name;
it could mean either "what does this affect" or "what does this depend on"; or it names nothing to
start from. Judge each question on its own.

Write one line per question:
`{"schema":"ccg-annotation-v1","aid":…,"label":"ambiguous"|"unambiguous","reason":…}`, where
`aid` is copied from the input line and `reason` has at most 200 characters.
