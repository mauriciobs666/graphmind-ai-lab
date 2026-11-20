# Work-item identity in the coordination graph — plan extension

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Extends:** `claude/docs/plans/coordination-graph.md`, `claude/docs/plans/coordination-graph2.md` · **Item:** `GAL-F0002`

Extends the base plan (`claude/docs/plans/coordination-graph.md`, "§") and its first extension
(`claude/docs/plans/coordination-graph2.md`, "plan2"); both stay authoritative except where this
document names a section it **replaces** (stated once, §3.7). States only the change. Input:
`claude/docs/reviews/coordination-graph.md` Pass 5 (P5-*); standard: `skills/work-items/docs/SPEC.md`
§1/§4/§5/§6, `docs/plans/work-item-numbering.md` §3.9. Stakeholder decisions D1–D3 (2026-11-20) are
folded into §3. The graph side of every change is specified once, in "Requirements for graph-dba" (§4);
`claude/docs/plans/coordination-graph-graph.md` is edited by that owner, not here.

## 1. Goal & scope

Make the mirror and its views work in a project where backlog items, features and ledger units carry
work-item scheme IDs that survive `wi move`/`promote`: a stable identity for backlog items
(dependencies, history lookup), a join from a document to its feature (`Item:`/`Aliases:`), and a place
for a ledger unit's `needs` to land and be resolved (`docs/requirements/cross-initiative-coordination.md`
FR-4, FR-5, AC-3, AC-4).

Out of scope: the ready-now view, `Touches` on units and FR-1/2/3/6 of the cross-initiative
requirements (their own plan, `docs/plans/cross-initiative-coordination.md`, still to be written);
minting legacy names; changing `wi`.

## 2. Context & findings

- **CPG:** considered, not relevant — documents and a component (`coordination-graph/`) not yet built; no code under review.
- Verified 2026-11-20 against the repo registry: `wi resolve coordination-graph` → `GAL-F0002`, `cross-initiative-coordination` → `GAL-F0003` (both registered, headers carry `Item:`); `falkor-chat:K-065` exits 5 (declared, unminted); bare `K-065` exits 3 (several declared homes); bare `C-310` exits 5; `GAL-F0042` and `bogus:K-1` exit 4 (SPEC §3 exit codes).
- `wilib` (stdlib only, `skills/work-items/scripts/wilib/`): `store.Store(root, key).read_state()` + `resolve.resolve(state, name)` raise `resolve.ResolveError` with `.code` 3/4 (5 via the legacy fallback hook, `legacy.py`); `rebuild.header_fields(path)` and `rebuild.tokens(value)` are `wi`'s own header reader (used by `wi check`) and tolerate every optional field. `read_state` may append missing tracked events to the journal (SPEC §9) — a `wi` behavior, accepted.
- Backlogs today: six `BACKLOG.md` files, not five — `docs/BACKLOG.md` (cpg, `C-NNN` bullets under `###` group headings), `falkor-chat/docs/BACKLOG.md` (already mixes `K-0NN` and scheme headings such as `GAL-I02-K0002`), and four with **no IDs**: `model-bench` (bold-lead bullets), `opencode`, `mcp-monitor`, `decision-models` (four `###` headings, no ID). Twelve of thirteen agent plans carry a parking lot of id-less bullets.
- `.work-items/project.json` `legacy_paths` already maps `claude/*/kaizen/plan.md`→`$1`, `*/docs/BACKLOG.md`→`$1`, `docs/BACKLOG.md`→`cpg`.

## 3. Design & rationale

### 3.1 Identity: stored raw, resolved at read time through `wilib` (P5-B1)

**Rule.** Every ID string in a source file is stored **as written**; identity is computed at **read time**
by one module, `coordgraph/identity.py`. **`wilib` importers (path from `COORDGRAPH_WI_ROOT`, default
`<repo>/skills/work-items/scripts`): `identity.py` and, through it, `lint` — read paths only.** The
`sync`/hook path imports no `wilib` and never reads the registry, so a source's subgraph stays a pure
function of its own file (graph note §2.3), the hook stays offline and fast, and a bad
`COORDGRAPH_WI_ROOT` cannot break FR-4 freshness. Registry access is via `wilib`, never the tracked
`.work-items/registry.jsonl` (the journal can hold unflushed IDs, SPEC §8).

```
identity.resolve(token, scope) -> Resolution(state, uid, candidates, reason)
  state: resolved | unminted | ambiguous | unknown | retired | unavailable
  candidates: the strings under which this identity may be stored (the join set)
```

1. **Route by one shape test** (opaque for storage, shape-tested only to pick the resolver; ASCII):
   (a) `^<legacy-letter>-\d+$` (one uppercase letter) **bare** → legacy, qualified with the referrer's `scope`
   (base §3.6 "bare = same backlog" is kept; `wi` alone would answer exit 3);
   (b) `^[a-z0-9][a-z0-9-]*:<legacy-letter>-\d+$` → qualified legacy, as written;
   (c) `^[A-Z]{2,6}(-[A-Z]\d+|-FR\d+[a-z]?|-AC\d+[a-z]?)+$`-shaped (key, then segments) → scheme ID or alias;
   (d) in a `needs` cell only, `^[a-z0-9][a-z0-9-]*:\S+$` not matching (b) → `<ledger-slug>:<unit>` (§3.4), and a bare token equal to a unit label of the referring ledger → local unit;
   (e) anything else → `unknown`, "unrecognised token".
   A scheme **reference is never scope-qualified**; legacy keeps `<scope>:<id>`.
2. **Resolve** (a)–(c) by one `wilib` call per *distinct* token per process (registry state loaded once, ~500
   items; never one `wi` subprocess per ref). Map `ResolveError.code`: 3 → `ambiguous`, reason = the CLI's
   candidate list verbatim (never guess); 5 → `unminted`; 4 → `unknown`; `retired` item → `retired`;
   registry unreadable or no `.work-items/` → `unavailable`.
3. **`candidates`** = `{id, uid}` ∪ `aliases` ∪ the item's legacy `home:name` (when minted) ∪ the token as
   written. An `unminted` legacy name has only the qualified token; `unknown`/`unavailable` only the token as
   written. Two references denote one item iff their candidate sets intersect.
4. **Joins are `IN $candidates` on indexed raw strings** (graph note §4.2 indexes; §4 G5): a reference matches
   an item whose stored string — `Item.ref`, `Source.itemId`, `Unit.itemId` — is any candidate. A rename after
   `wi move` therefore joins without any stored string changing (alias ∈ candidates). No mirrored copy of the
   registry exists; **sync header** gains `Registry: <n> items · last event <date> · <m> unflushed
   journal events`, or `Registry: unavailable — identities compared as written`. A reader never sees
   a stale identity because none is stored.

Alternatives rejected: resolve at sync time and store the canonical ID (a `wi mint`/`move` would change
an item whose file did not change — parity and drift cannot see it); mirror the registry as a `:Source`
kind (a second source of truth for a file whose read path already exists).

### 3.2 Header fields (P5-M2)

The header parser for every `:Source` kind (`coordgraph/parse/header.py`, stdlib, one function taking the
**already-read text** the parser hashes — never a path) implements `wi`'s one-line header grammar (SPEC §5:
the `> **Status:**` line under the H1, blank allowed; ` · `-separated `**Label:** value` fields; backticked
tokens) and stores `Source.itemId` (first `Item:` token, as written) and `Source.aliases`
(`Aliases:` tokens), tolerating every other field. A **parity test** (§6) pins it to `wilib.rebuild.header_fields`/`tokens` over every family document in the repo. `docStatus`, `owner`, `tracks` as before. `aliases` is diagnostic; **no join uses it**
(§3.1 candidates come from the registry). `unit`/`inflight` accept a feature ID, alias, task ID or slug:
`unit <ledger-slug|ID> [<unit-id>]`; an ID is expanded to its candidates and matched on `Source.itemId`
(`Unit.itemId` for a task ID); the slug join stays as fallback (base §3.4 R4, "picked up"), and a requirement
is "picked up" when a ledger shares its slug **or** its resolved `itemId`.

### 3.3 Backlog items, dependencies, delivery (P5-B1, P5-m2, P5-n1)

- `Item.ref`: a scheme-shaped heading → the ID as written; a legacy heading → `<scope>:<id>`; `scope` from
  `project.json` `legacy_paths` (read by `identity.py`, not restated; replaces base §3.6 "Namespaces come from
  the path"). `Item.itemType` (optional): the last segment's letter (`I S T F D K`) parsed offline; absent for legacy.
- `Depends on`: raw tokens, split on commas, trimmed, `none` = `[]`, absent = not recorded (base §3.6);
  **not qualified at parse** (replaces base note §2.2 `dependsOn`). Resolution per §3.1 with the referrer's scope;
  an unminted legacy ref compares as written, an `unknown`/`ambiguous` ref is "unknown dependency, reason named".
- **Delivery matching** (base §3.7 step 2): the dependency is resolved first. A history heading is
  delivery-shaped as before; its ID tokens are **found** by shape (§3.1 (a)–(c), history-file scope for bare
  legacy) and each resolved — the ref is delivered when any token's candidates intersect the dependency's.
  Scheme refs search every history file; legacy refs the owning backlog's. Example `## … — GAL-F0042 delivered`,
  and a scheme ID among slash-joined siblings.
- **Registry attributes are never read by the planner.** `milestone`/`touched`/`home` in the registry (SPEC §3
  `set`) are a different fact from the backlog `Milestone` field (sole authority, base §3.6) and from
  `Touches` (files/graph keys/facts, not components): `wi set milestone` does not place an item.

### 3.4 Units and `needs` (P5-M1, P5-M4; D1)

- **Ledger columns, mapped by header name** (graph note §2.2 `:Unit`): optional `Item` → `Unit.itemId`, the
  unit's task ID (first token, as written); optional `Needs` → `Unit.needs`, comma-separated tokens or `none`.
  Absent column/cell = not recorded; `none` = `[]`. A malformed cell (empty token between commas) → `needs`
  absent, `Unit.needsError` set, source warning; AC-3 lists the unit with `needs not recorded`/the error.
  Existing ledgers without the columns are unchanged (FR-7). `Touches` stays with the cross-initiative plan.
- **Token forms** (FR-4, replaces `docs/plans/cross-initiative-coordination-ids.md` decisions (a)–(c)):
  a task or feature ID or alias (§3.1 (c)); `<ledger-slug>:<unit>`, where `<ledger-slug>` is the **graph's
  role-stripped slug** (`connected-code-graph`, never the basename `connected-code-graph-coordination`), the
  same key `unit` takes; within one ledger the local unit label (`U3`).
- **`identity.resolve_need(token, ledger) -> NeedResult(state: met|unmet|unresolvable, reason, matches)`**,
  pure over fetched rows:
  - task ID → `Unit.itemId IN candidates`; exactly one unit: `accepted` → met, `unknown` → `unresolvable`
    ("status unknown (<ledger>, <raw cell>)", never met or unmet — FR-5; for an archived unit, which keeps no cell text (§4 G3), the CLI reads the cell from the ledger file by `Unit.key`, and falls back to the reason without the cell if the file or row is gone), any other token → unmet (an archived
    ledger's unit is read from its minimal `:Unit`, §4 G3);
  - feature ID (D1) → ledgers with `Source.itemId IN candidates`; **met iff every such ledger is `archived`
    (whatever its units read) or, when not archived, has ≥1 unit and all units `accepted`** (`abandoned` and
    `unknown` units in a non-archived ledger do not count); no ledger carries it
    → `unresolvable`, "no ledger carries `<ID>`";
  - `<slug>:<unit>` / local → `Ledger.slug` then `unitKey` (`U32` matches every repeat; `U32~2` addresses
    one). **More than one match** (a repeated label, or one slug in two components) → `unresolvable`,
    "ambiguous: <paths and keys>" — the list is repeated, never a guess; cite by task ID instead;
  - `ambiguous`/`unminted`/`unknown`/`retired`/`unavailable` (§3.1 step 2) → `unresolvable`, reason named;
    an unminted legacy name carries the CLI's `wi mint-legacy …` text.
  A need never copies another ledger's status.

### 3.5 One-time reformat mints scheme IDs only (P5-M3; D2)

`coordgraph migrate` never emits `K-NNN`/`C-NNN` and has no "past the highest id" rule (replaces base
§3.6 reformat bullet 1 and plan2's reformat bullet). Every item without an ID — parking-lot bullets and
id-less component items included — gets one from `wi new`: agent plans (`claude/<agent>/kaizen/plan.md`)
`wi new K --home claude --title <t>`, component backlogs `wi new F --home <component> --title <t>`, `--slug`
unset, file order. Existing IDs (legacy or scheme) are kept; minting a legacy ID is not part of the reformat.
`migrate --dry-run` allocates nothing (placeholder per new item + the exact `wi new` command); **`migrate
--apply PATH`** performs the allocation and rewrite, idempotently, one file per run; each applied file's
commit also stages `.work-items/registry.jsonl` after `wi flush` (root `claude/AGENTS.md` flush rule).

### 3.6 Heading pattern and plan2 corrections (P5-m1, P5-n2)

Replaces plan2's pattern: `^### (?P<id>[A-Z]-\d{3,}|[A-Z]{2,6}(?:-(?:[IST]\d{2,}|[FDK]\d{4,}))+) — (?P<title>.+)$`
(verified against `wilib.rules.parse` over the table below). Shape plus the minimum widths; canonical
form (no leading zeros, parent matrix) is `lint`'s call through `wilib.rules`/`wi check`, not the parser's.
`FR`/`AC` segments never match.

| Heading id | Matches | Why |
|---|---|---|
| `K-004`, `C-310`, `K-0001`, `GAL-F0042`, `GAL-I03-F0012-S01-T01`, `GAL-I02-K0001`, `GAL-I01-D0001`, `GAL-F0042-S01` | yes | positives |
| `GAL-F00042` | yes | passes shape; `lint` reports it malformed via `wilib.rules.parse` (documents the split) |
| `GAL-`, `gal-F0042`, `GAL-F4`, `GAL-F42`, `GAL-F042`, `K-04`, `GAL-X0001`, `GAL-0042`, `GAL-F0042-T3` | no | negatives |
| `GAL-F0042-FR2`, `GAL-F0042-FR2a` | no | `FR`/`AC` are citations, not items |

Plan2's opacity rule becomes: `Depends on`, `Needs`, history and citation tokens are stored verbatim and
never regex-validated at parse; §3.1's shape test only chooses a resolver. Test: planted `GAL-F42` and free
text are stored verbatim and shown unresolved (a shape-validating parser fails it). Plan2 L8's "FR-4(b)"
cites a decision of the ids note, not a requirement; cite §3.1 instead.

### 3.7 Statements replaced (rework in place semantics, stated once)

| Where | Replaced by |
|---|---|
| Base §3.6 heading regex; plan2 pattern and test table | §3.6 above |
| Base §3.6 "bare = same backlog", `Depends on` row | kept, resolved per §3.1 (a); refs stored raw |
| Base §3.6 "Namespaces come from the path" | `legacy_paths` read, §3.3 |
| Base §3.6 reformat bullet 1 and plan2 reformat bullet | §3.5 |
| Base §3.7 step 2 delivery match, plan2 "opaque token" | §3.3, §3.6 |
| Base §2.1 backlog row and S4: "five `BACKLOG.md` files" | six (§2); `teco` applies six |
| Base §2.5 (L120) and §6 Q4: requirements "`Interviewing`" | `docs/requirements/cross-initiative-coordination.md` is `Ready for design` (v4) and its re-scope is done; a dependency of nothing here |
| Base §3.6 field table `dependsOn` and graph note §2.2 L131/L135 (`Item.ref`, qualified `dependsOn`), §2.3 | §3.3, §4 G4 |
| Base §3.4a(c) L346 "an archived ledger keeps no `:Unit` children"; §3.8 L648-649 "children only while the file is live; … `:Source` with no units"; graph note §2.2 L99 | an archived ledger keeps minimal `:Unit` nodes (§4 G3), hidden from the in-flight views by the `docStatus` filter; its `citedReviews`/`resolvedReviews` stay |
| Base §5 L776 integration "archived flip → units removed, stub kept (AC-7)" | asserts: archived units absent from `inflight`, minimal units present, review lists kept (§6 test 12) |
| Base §3.7 step 2 L565 "refs qualified `<backlog>:<id>`" | refs as written, resolved per §3.1; a scheme ref is never qualified |
| Base §3.6 L521 "the five `BACKLOG.md` files" (as §2.1 L55 and S4) | six |
| Base §2.1 L52 ledger header "`Status · Owner · Tracks`" | the header grammar of §3.2 (every optional field tolerated; `Item`/`Aliases` captured) |
| Prose "initiative" for the in-flight level (D3) | "coordination" — the job one ledger tracks; it maps to a feature when one is registered. The cross-initiative slug is unchanged; view labels in its plan say "coordination" |

## 4. Requirements for graph-dba

Hand to `graph-dba` for `claude/docs/plans/coordination-graph-graph.md` (next version). Each item states the
requirement; Cypher, sizing and verification are yours.

- **G1 `:Source` (all kinds):** `itemId` (string, as written; absent when the header has no `Item:`), `aliases`
  (list<string>, as written; `[]` when a header exists without `Aliases:`, absent when no header block).
  RANGE index on `Source.itemId`, **non-unique** (a family shares one). Apply §4.1 key hygiene to the
  indexed value.
- **G2 `:Unit`:** optional `itemId` (as written), `needs` (list<string> raw, absent-vs-`[]` rule of
  `dependsOn`), `needsError` (string, present only when malformed). RANGE index on `Unit.itemId` if sizing
  agrees (§3.4 looks up by candidates).
- **G3 Archived ledgers keep minimal `:Unit` nodes** — `key`, `unitKey`, `statusToken`, `itemId`, plus the
  denormalized `ledgerPath`, `slug`, `component` — and no cell text/`needs`; you decide `rowHash`/`tableIdx`/
  `rowIdx` and the parity (`rowCount`) and drift consequences. Change §2.2 L99 ("no `:Unit` children") and §7.2(a)
  and every `:Unit` read (paused view anchored on `statusToken`, §7.3, §7.5) so archived units stay out of the
  in-flight views (`docStatus <> 'archived'`); size the ~900 extra rows against §8; the `unknown`-in-archived
  tally rule of §2.5 stays.
- **G4 `:Item`:** `ref` = the heading ID as written for a scheme heading, `<scope>:<id>` for a legacy one
  (UNIQUE stays; a scheme ref in two backlog files is a plain error: the §5.5 step-1 assertion and the atomic
  failure message name **both paths** and hint that a `wi move` may be in progress. No stale-holder retry:
  the base `failed.pending` backed-off retry (base §3.2) covers the transient case); `scope` from `legacy_paths`; `dependsOn` holds **raw,
  unqualified** tokens (replaces L135); optional `itemType` (`I S T F D K`, absent for legacy); id-less
  positional `@ord` refs remain only until the reformat is applied.
- **G5 Reads (§7):** (a) `Item.ref IN $candidates` resolution of a dependency; (b) ledgers by
  `Source.itemId IN $candidates` with their units' `statusToken` (archived included); (c) units by
  `Unit.itemId IN $candidates`; (d) `unit` lookup by ID (§7.3). **Verify (PROFILE on a throwaway
  container) that `IN` with a list parameter seeks the index**, else give an `UNWIND`+equality form. Extend
  the §2.3 table: `Unit.needs`/`Source.itemId`/`Item.dependsOn` are raw strings joined at read time against
  candidates computed outside the graph; state that the registry is a second input not covered by parity.
- **G6 Versioning:** bump `parserVersion` (forces `sync --all`); add the new properties to the parity and
  `rebuild --dry-run` assertions (§5.2, §5.5 step 1).

## 5. Step-by-step implementation

Against base §4 (owners and gates as there); sequencing note: the graph note's next version (§4) precedes S3.

| Step | Change |
|---|---|
| **S1** parsers (`coordgraph/parse/*`, `history.py`) | header via the stdlib `parse/header.py` (§3.2, text passed in); backlog heading §3.6, raw `Depends on`, `Item.ref` policy; ledger `Item`/`Needs` columns; `migrate --dry-run/--apply` §3.5; history shape-scan §3.3 |
| **S1d** `coordgraph/identity.py` (new; `tdd-engineer`, clear contract; after S1; gate `analyst`) | §3.1 `resolve`, §3.4 `resolve_need`, registry line for the sync header; fake `wilib` state in tests |
| **S3** store | implements note §4 (G1–G6); no registry read on the sync path |
| **S4** reformat | six files; `migrate --apply` per file; `wi check` clean after each |
| **S5** skill | `Item`/`Needs` columns, `unit <slug\|ID>`, delivery-heading convention with scheme IDs, "cite legacy as `<home>:K-NNN`, never bare `K-`", registry line in the header |
| **S7a/S7b** views, proposal | `unit`/`inflight` by ID; `planned`/`propose` join through `identity`; the ready-now view stays with the cross-initiative plan |
| **S9** rollout | `teco.md` ledger template gains optional `Item`/`Needs` columns; `audit-team.sh` runs `lint` incl. unresolved refs |

## 6. Test strategy

Offline unless noted; each is red first on the naive implementation named.

1. Heading table §3.6 (all rows); `lint` reports `GAL-F00042` malformed through `wilib.rules`.
2. Opacity: planted `GAL-F42`/free text in `Depends on` and `Needs` stored verbatim, shown unresolved.
3. Header: `parse/header.py` parity with `wilib.rebuild.header_fields`/`tokens` over every repo family document; a header carrying `Version`, `Extends`, `Reviews`, `Last updated`, `Item`, `Aliases` parses with `itemId`/`aliases` set and no other field lost.
4. Routing (§3.1): each of (a)–(e) incl. `falkor-chat:K-065` (legacy) vs `connected-code-graph:U7` (unit).
5. Exit mapping: 3 (bare `K-065`, repeats the list), 4, 5 (unminted: compares as written, not unresolvable for a backlog ref), retired, registry unavailable.
6. **Planted rename:** a `Depends on`/`Needs`/header `Item:` written with the old ID joins after `wi move` (red on raw-string equality); a dependency `falkor-chat:K-065` joins an item headed `GAL-I02-K0042` once minted and the reverse.
7. Cross-backlog bare scheme ID in a `cpg` backlog for an item in `falkor-chat` (AC-13) joins; never `cpg:`-qualified.
8. Delivery: `## … — GAL-F0042 delivered` satisfies a dependency written as `falkor-chat:K-065`; slash-joined siblings.
9. Needs: task ID met/unmet; feature ID (D1) — every ledger archived; all units `accepted`; one ledger unfinished; `abandoned` unit in a non-archived ledger → unmet, in an archived one → met; none carries it → unresolvable; need on an archived ledger's task (live: minimal units); `<slug>:<unit>` unique, repeated label, same slug in two components → unresolvable with the list; `U32~2`; malformed cell → AC-3 listing; AC-4 variants D (other ledger) and D′ (cited by alias after a move).
10. Reformat: dry-run allocates nothing (registry/journal unchanged); `--apply` over a scratch copy of each of the six `BACKLOG.md` files and the agent plans yields only scheme IDs, `lint` and `wi check` clean, idempotent on a second run; id-less counts equal the dry-run's.
11. Scope derivation test reads `project.json` `legacy_paths` and asserts the parser's scope for each backlog path.
12. Hook-time sync with `COORDGRAPH_WI_ROOT` bogus still mirrors the file (red if the sync path imports `wilib`); a read view then reports `Registry: unavailable`.
13. Live (ephemeral container): archived units absent from `inflight`; `IN` seek plan; `unit GAL-F0002` and `unit <old-id>` return the ledger(s); archived flip → minimal units present, absent from `inflight` (AC-7); a task need on a unit whose archived cell reads `blocked (…)` is unresolvable, "status unknown".

## 7. Risks & open questions

- **ID inflation:** id-less parking-lot bullets recording settled judgments (`⚪ rejected — record, not work`) also get IDs under D2. Mitigation: the dry-run lists them; the applier may still drop none (text is never dropped).
- **Registry coupling:** the views now depend on `wilib` importability and a readable registry; degraded mode (§3.1) is visible in the header, not silent. `read_state`'s journal append is a `wi` behavior; read views run under the same user, outside the sync hook.
- **Precedence:** plan2's and the base plan's replaced text stays in place (this plan edits headers only); the implementer applies §3.7. Fold on their next revision.
- **Q1 (architect call, recommended):** this plan fixes the `Item`/`Needs` ledger-column syntax and the need-resolution contract, which the cross-initiative requirements left unspecified; the ready-now view and `Touches` remain for `docs/plans/cross-initiative-coordination.md`. Recommend that plan reuse §3.4 and the ids note (`docs/plans/cross-initiative-coordination-ids.md`) be superseded by it (its slug `-ids` is outside the closed role set).

## Ready to implement

Implement `claude/docs/plans/coordination-graph3.md` after `graph-dba` revises the graph note per §4 (G1–G6). Identity is stored raw and resolved at read time by `coordgraph/identity.py` over `wilib`; joins are `IN` candidate sets; archived ledgers keep minimal units; the reformat allocates scheme IDs via `wi new` only.
