# Live specs — Implementation Plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Version:** 1.2

*Revision note — 2026-10-31 (v1.1):* reworked per `docs/reviews/live-specs.md` Pass 4 and the stakeholder's answers of the same day; v1.2 (2026-10-31) folds in Pass 5.

## 1. Goal & scope

Deliver `docs/requirements/live-specs.md`: the repo conventions and agent prompts that make each
component's live spec the source of truth (FR-1–FR-9, FR-11–FR-13), three `REVIEW.md` files
(FR-5, FR-9), and the one-time catch-up of falkor-chat's spec set and model-bench's first spec
(FR-10). Governing rule for every gate below: **no agent gates a change to its own permissions,
its own prompt, or text it wrote or applied.** Integrity checks (e.g. §3.4 step 4's line
check) are not gates, and non-narrowing `REVIEW.md` additions are ungated by design (FR-8).

Out of scope: the requirements' list, plus editing archived documents or re-homing
model-bench's root-level `docs/plans/small-model-benchmarking.md` family (FR-1).

## 2. Context & findings

- **Write guards** (`claude/AGENTS.md`, "Hook machinery"): `analyst` writes only `docs/reviews/`;
  `cobb`'s allowlist excludes root and component `AGENTS.md`/`README.md` (one human approval
  per file); `coder`'s deny-list (`scripts/guard-broad-write.sh`) and `graph-dba` (no write
  guard) allow the spec paths; `teco` is repository-bounded.
- **falkor-chat drift baselines are per document** (`git log -1`): `DESIGN.md` `7a270329`
  (2026-09-20), `SERVER.md` `a5d8e807` (2026-10-25), `QUERIES.md` `9ba0d788` (2026-10-28). Code
  commits since each (`server/`, `scripts/`, `web/`, `config/`, `compose.yaml`, `Dockerfile`,
  `salesperson/src`): 23, 3 (K-079 F2, K-086) and 0. A last touch does not prove completeness,
  hence the inventory check (§3.5).
- **History in the specs** (AC-3): `K-` references 16/6/90 and ISO dates 6/3/16
  (DESIGN/SERVER/QUERIES), plus `DESIGN.md`'s revision note, `Version:`, H1 "Design Document &
  Blueprint", §12 Roadmap, §13 Open questions, the §14 "moved to `SERVER.md`" redirect table,
  `SERVER.md` §1.6 "TDD build order" and §1.6's same-change doc rule, and headings like "(v2,
  DEF-1 fix)".
- **Inbound section citations are live.** `DESIGN §14.x`/`§15` is cited from about 15 source
  docstrings (`api.py`, `config.py`, `repository.py`, `services.py`, `db.py`, `tools.py`,
  `web/app.js`), `DESIGN §13` from `guards.py`, `bootstrap_schema.sh` and `test_guards.py`, and
  §12/§13/§15 from `falkor-chat/docs/BACKLOG.md` and `skills/agent-standards/kiro.md`.
- **AC-9 hits** are mapped to units in §5.
- **typed-decision-models is a feature under FR-2** (requirements document, no backlog item)
  spanning model-bench (U1–U7) and falkor-chat (U9–U12, incl. a `SERVER.md` §1.8 edit). It closes
  after this plan's switch-over and is model-bench's first close; §3.6 states its handling.
- **No repo-level `HISTORY.md` exists** (root `docs/HISTORY.md` is `cpg/`'s); the
  document-size-discipline precedent recorded delivery in each touched agent's kaizen history.

CPG: considered, not relevant — this plan's investigation is of documents, prompts and git
history; `cpg_falkorchat` (being rebuilt, ledger L4) is assigned to the AC-1 checks (§3.5).

## 3. Design

### 3.1 Where each rule lives

Each rule has one home, and the other places cite it:

| Rule | Home | Cited from |
|---|---|---|
| What a live spec is (FR-1, FR-3, FR-4), feature close (FR-2, FR-7, FR-8, FR-11), `REVIEW.md` upkeep, delta plans (FR-12, FR-13) | root `AGENTS.md` | `teco.md`, `architect.md`, component `AGENTS.md` |
| How a reviewer checks delta plans and a spec update, and how it proposes a rule (FR-8, FR-12) | root `REVIEW.md` | reviewer prompts (one pointer each) |
| How `teco` runs the close (delegation, line check) | `claude/teco/teco.md` | `claude/README.md` row |
| Which documents make up each component's spec (FR-4, AC-2) | component `AGENTS.md` | — |
| The switch-over and the typed-decision-models arrangement (transitional) | `docs/plans/live-specs-coordination.md`, then the delivery records | — |

Reason: root `AGENTS.md` is bounded (~2,500-word smell; 3,198 today) and states only what stays
true, so reviewer-only rules go to `REVIEW.md`, read only when a review runs, and transitional
text never enters an always-loaded file.

### 3.2 Repo-convention edits

**Root `AGENTS.md`, "Module documentation convention":**

- *Live spec* (new bullet): what a live spec is; the component's `AGENTS.md` names its set; it
  states current externally observable contracts with no change history; spec documents never
  become `archived`, carry no `Version:` or revision notes, and keep their section numbering (an
  emptied section keeps its heading and a one-line pointer, so inbound citations resolve).
- *Feature close* (new bullet): FR-2's definition of a feature, the §3.4 sequence, the FR-11
  whole-component spec, the delta-plan rule (FR-12, FR-13), and `REVIEW.md` upkeep: its lines
  come only from review proposals and are applied by `teco` (§3.4 step 4).
- The `BACKLOG.md` bullet's "the component's `DESIGN.md` or `README.md`" becomes "the
  component's live spec".
- Elsewhere: the "Component docs" table marks falkor-chat's three spec documents as its live
  spec and adds model-bench's (§3.6); the growth-stops bullet's review-only sentences ("Every
  review gate checks a revision against this bullet …") move to root `REVIEW.md`, leaving a
  citation. Net growth ≤ +150 words (`wc -w` before and after).

**`claude/AGENTS.md`:** no change. **`falkor-chat/AGENTS.md`:** the opening decision-register
sentence and the "Key documents" rows for the three spec documents name them as the live spec
set (AC-2, AC-9). **`falkor-chat/README.md`:** l.23's "ops blueprint" pointer and l.237's layout
comment ("full blueprint … §14 M1 app architecture") describe `DESIGN.md` as part of the live
spec. **`model-bench/AGENTS.md`:** the spec index joins "Documentation map", and "Load-bearing
invariants" entries that state a spec contract shrink to citations (L6).

### 3.3 `REVIEW.md` files

**Location and form.** `REVIEW.md` at the repo root, `falkor-chat/REVIEW.md`,
`model-bench/REVIEW.md`, beside each `AGENTS.md`: instruction files like `AGENTS.md`, so an H1 and
no header block. Headings are the four FR-5 kinds (*What counts as serious*, *Always check*, *Do
not report*, *Evidence bar*); a component file carries only those it fills.

**Content: `analyst`, applied by `teco`, gated by `cobb`.** `analyst` owns review practice and its
review corpus shows which finding classes recur, but it cannot write outside `docs/reviews/` and
its guard is not widened. It drafts the seed as `### Proposed REVIEW.md lines` blocks (target file
named, each line's source stated) in a new `## Pass N` of `docs/reviews/live-specs.md`, the same
shape every later proposal takes. `teco` applies the blocks verbatim. `cobb` gates the files: it
neither wrote nor applied them, and the coherence of standing team instructions with prompts and
`AGENTS.md` is its remit.

**Seed sources** — a line is seeded only if it changes what a reviewer does, and only from:
1. Review-only clauses now in root `AGENTS.md`, moved, not copied (§3.2).
2. Duties this feature creates: restated spec content in a plan is a finding (FR-12), as is a
   missing citation of an open feature's plan (FR-13); a spec-update check is made against shipped
   code, not the plan (FR-7); the proposal format, including that a prune cites its enforcer.
3. Finding classes recurring in at least two reviews from the last 60 days, including lines
   already proposed in a review (e.g. `docs/reviews/live-specs.md` Pass 4). Component files draw
   on their own corpus and cite `AGENTS.md` hard rules ("check X holds, per `AGENTS.md` § Y")
   rather than restating them.

A seed line that duplicates a prompt line goes on a duplicate list; `cobb` turns that prompt line
into the pointer in L3a–L3c.

**Size.** FR-9 smells are ~1,000 words root, ~800 per component; seeds aim for ≤ 600 and ≤ 400,
leaving room for FR-8 growth.

### 3.4 Feature close

The sequence, in root `AGENTS.md`, run by `teco`:

1. Every build unit is gated.
2. For each touched component with a live spec, `teco` delegates the spec update, by default to
   the feature's implementer type (`coder`, `tdd-engineer`, `frontend-engineer`), which knows the
   diff; canonical-Cypher changes go to `graph-dba`. The writer edits the sections the plan cited
   (FR-12) plus any other contract the diff changed.
3. `analyst` checks the update against the diff from the feature's baseline sha, as a `## Pass N`
   in the feature's `reviews/<slug>-impl.md`. A slug-less change (FR-2's `HISTORY.md` clause) uses
   its existing review or RCA document if one exists, else `reviews/<topic>-impl.md` with a topic
   slug naming the change. The feature does not close on "needs changes".
4. `teco` collects every `### Proposed REVIEW.md lines` block from the feature's reviews. Each
   line is an addition or a prune; a prune names the test, CI job or hook that now enforces the
   rule. `teco` applies them verbatim (curation, not authorship), then runs an integrity check,
   not a gate: the `REVIEW.md` diff matches the proposals both ways and each cited enforcer
   exists. Additions need no gate. A narrowing change (a prune or a *Do not report* addition) is gated by `cobb`, or by
   `analyst` when `cobb` proposed it.
5. The `HISTORY.md` entry is written last, naming the spec sections updated and the check's review
   path (AC-4). Per-unit `HISTORY.md` entries do not close a feature.

Archiving stays at milestone close.

**Switch-over** (ledger text, not `AGENTS.md`): the conventions bind from the commit of L3c.
A unit dispatched earlier finishes under its brief; one dispatched later makes no live-spec edit,
and a plan's in-change spec instruction (e.g. `docs/plans/typed-decision-models3.md` §3.7, U9's
`SERVER.md` §1.8 line) becomes a close-step item.

### 3.5 falkor-chat catch-up (FR-10, AC-1, AC-3)

**One unit per document**, chosen by domain and code reading:

| Document | Writer | Why |
|---|---|---|
| `DESIGN.md` | `graph-dba` | Graph topology, data model, indexes, retrieval and capacity are its domain |
| `SERVER.md` | `coder` | REST/MCP surface, auth seam and layout are read off `server/` |
| `QUERIES.md` | `graph-dba` | Canonical Cypher, re-verifiable read-only against `falkordb-dev` |

`Owner:` headers stay; they name the producer.

**Home documents** — each contract surface has one home; the other documents cite it:

| Surface (code) | Home |
|---|---|
| Graph topology, schema, indexes, DDL (`bootstrap_schema.sh`); chat and workflow-engine graph model and executor semantics (`executor.py`); ingestion write paths; capacity | `DESIGN.md` |
| REST routes (`api.py`, `storefront_api.py`), MCP tools (`mcp.py`), auth/tenancy seam and config keys (`config.py`, `config/`), `/shop` and `web/` mounts (`app.py`), responder trigger, model resolution (`modelconfig.py`; `DESIGN.md` §1.3 keeps only the model-stack decision), agent tool surface incl. `query_graph_data`'s row contract (`querygen.py`) | `SERVER.md` |
| Every query's text and its guards (`repository.py`, `querygen.py`) | `QUERIES.md` |

**Writer's method** (git-based): `git log <baseline>..<pinned sha>` over §2's code paths and
baselines, plus `falkor-chat/docs/HISTORY.md` entries after each baseline. Plans they cite only
locate a contract; the spec states what the code does.

**History removal** (AC-3), in the same pass: revision notes, `Version:`, `vN`/`DEF-` qualifiers
and change dates go; a `K-` reference stays only if `BACKLOG.md` holds it open. Section numbers
never change. L5a: the H1 becomes "falkor-chat — Design" (no "Blueprint"); §12 becomes a
`BACKLOG.md` pointer line; §13 keeps its resolved decisions in present tense and points at
`BACKLOG.md` for open ones (the writer lists open ones with no backlog item, `teco` files them);
§14 is restated as a present-tense section map to `SERVER.md`, not deleted, because it resolves
the inbound `§14.x`/`§15` citations in §2. L5b: §1.6, including its same-change doc rule, becomes
a one-line pointer. Testing hazards (§1.7) and design rationale stay as current truths.

**Checker's method** (inventory-based, independent): enumerate each surface from the code, using
`cpg_falkorchat` once L4 has rebuilt it, else source reads. Each L5 check covers its document's
home surfaces plus any contradiction anywhere in it; L5s then checks the three documents together
against the whole inventory (the AC-1 verdict); any contract-bearing surface outside the
home table (e.g. `guards.py`, retrieval, `extraction.py`/`fusion.py`/`chunking.py`, `llm.py`,
`pricing.py`) gets a home assigned by L5s in its Pass; that document's L5 writer absorbs it before L5d. model-bench's inventory: CLI (`cli.py`), run
lifecycle (`runner.py`), pack format per role (`packs.py`, `packs/`), result and fingerprint
schema with versioning (`results.py`, `fingerprint.py`, `hostinfo.py`), scoring (`scoring/`,
`decisions.py`), verdict statistics (`stats.py`), report output (`report.py`), LM Studio
interaction (`lmstudio.py`).

All units and checks use one pinned code sha, recorded in each review, and serialize with any
in-flight unit editing the same document (`teco` checks open ledgers).

### 3.6 model-bench spec (FR-4, FR-10, AC-2)

**Documents:** `model-bench/docs/DESIGN.md`, read whole (CLI, run lifecycle, LM Studio contract,
scoring and decision semantics, verdict statistics, report), and `model-bench/docs/FORMATS.md`,
read by lookup (pack format per role, result and fingerprint schema, schema versioning),
mirroring falkor-chat's `DESIGN.md`/`QUERIES.md` split and keeping `DESIGN.md` one meaning
repo-wide (AC-9). `README.md`'s three contract sections shrink to pointers. **Writer:** `coder`,
from the code with the §3.5 inventory; a `data-scientist` consult covers `stats.py` wording only
if the check flags it.

**Timing (stakeholder, 2026-10-31):** L6 runs inside typed-decision-models' close, after its
build units are gated and before its closing `HISTORY.md` entry. The whole-component spec
includes that feature's behavior, and L6's `analyst` check is both the AC-1 check for model-bench
and that feature's FR-7 check. `teco` records this arrangement in the ledger. If another
model-bench feature reaches its close first, FR-11 applies to it and L6 moves ahead of it.

## 4. Units & sequence

Unit ids extend the ledger's (`docs/plans/live-specs-coordination.md`). A unit editing an agent
file also writes that agent's `kaizen/history.md` entry, the convention's delivery record.

| Unit | Owner | Inputs | Files | Done when | Gate |
|---|---|---|---|---|---|
| L2a | `analyst` | §3.3; root `AGENTS.md`; review corpus | `docs/reviews/live-specs.md` (new Pass) | Seed blocks for all three files within §3.3's targets, sources stated; duplicate list | — (drafted only) |
| L2b | `teco` | L2a | `REVIEW.md`, `falkor-chat/REVIEW.md`, `model-bench/REVIEW.md` | L2a blocks applied verbatim; brief states this is curation, ahead of L3c's `teco.md` carve-out | `cobb` |
| L2c | `cobb` | §3.2; L2b | `AGENTS.md`, `falkor-chat/AGENTS.md`, `falkor-chat/README.md` | §3.2 edits; root net ≤ +150 words | `analyst` |
| L3a | `cobb` | §3.1, §3.3; duplicate list | `analyst.md`, `qa-engineer.md`, `security-expert.md`, `data-scientist.md` (+ kaizen) | One pointer each: read and apply the root and touched-component `REVIEW.md` before any review or gate verdict; `data-scientist` also points at the delta-plan rule; duplicates become pointers | `analyst`; `security-expert` for `analyst.md` |
| L3b | `cobb` | as L3a | `graph-dba.md`, `cobb.md`, `tico.md` (+ kaizen) | As L3a, the pointer scoped to gate verdicts and line checks for `tico`; `graph-dba` also gets the delta-plan pointer | `analyst` |
| L3c | `cobb` | §3.4; L2c | `teco.md`, `claude/README.md`, `architect.md` (+ kaizen) | `teco.md`: close step citing root `AGENTS.md`; l.3, l.92, l.138, l.155 carve out live-spec content; "Documentation curation" ("you don't write the docs yourself") permits verbatim application of proposed `REVIEW.md` lines; the documentation scan flags spec sections for the close; `REVIEW.md` pointer for line checks. README `teco` row matches. `architect.md`: delta-plan pointer. Commit = switch-over | `analyst`, re-running §5's AC-9 sweep |
| L7 | `tico` | L3c | `claude/docs/manuals/agent-team.md` | l.118 and l.190 match the close sequence | `teco` doc check |
| L5a | `graph-dba` | §3.5; pinned sha | `falkor-chat/docs/DESIGN.md` | Home surfaces absorbed, §3.5 removals, open §13 orphans listed | `analyst`, scoped check |
| L5b | `coder` | §3.5; pinned sha | `falkor-chat/docs/SERVER.md` | Same | `analyst`, scoped check |
| L5c | `graph-dba` | §3.5; pinned sha | `falkor-chat/docs/QUERIES.md` | Same | `analyst`, scoped check |
| L5s | `analyst` | L5a–c gated | — (review Pass) | Set-level AC-1 verdict not "needs changes" | — (is the gate) |
| L5d | `teco` | L5s | `falkor-chat/docs/HISTORY.md`, `BACKLOG.md` | Catch-up entry; §13 orphans filed | — (mechanical) |
| L6 | `coder` | §3.5, §3.6 | `model-bench/docs/DESIGN.md`, `FORMATS.md`, `README.md`, `AGENTS.md`, `docs/HISTORY.md` | Whole-component spec, indexed; contract sections in `README.md`/`AGENTS.md` reduced to pointers | `analyst` AC-1 check |

Every gate obeys §1's rule: `cobb` gates text `teco` applied from `analyst`'s draft, `analyst`
gates `cobb`'s edits except to `analyst.md`, and no unit is gated by its writer.

**Order:**
1. Convention chain: L2a → L2b → L2c → {L3a ∥ L3b} → L3c → L7. L2c follows L2b because the
   sentences leaving root `AGENTS.md` must already be in `REVIEW.md`; L3a/L3b touch disjoint
   files; L3c's commit is the switch-over.
2. L5a ∥ L5b ∥ L5c, parallel with the convention chain (disjoint files); checks wait for L4 or
   fall back to source reads; then L5s → L5d.
3. L6 per §3.6.

**Shared files.** Root `AGENTS.md`, `claude/README.md`, `teco.md` and `claude/*/kaizen/` may be
touched by concurrent sessions; `teco` checks open ledgers before dispatching L2c and L3a–L3c,
serializes any overlap, and the writer re-reads the file immediately before editing.

## 5. Verification

- **AC-1** — L5s for falkor-chat, L6's check for model-bench, both with the §3.5 inventory.
- **AC-2** — `grep -n 'live spec' falkor-chat/AGENTS.md model-bench/AGENTS.md` names each set.
- **AC-3** — per spec document, `grep -nE 'Revision note|Version:|Blueprint|moved (out )?to|\(v[0-9]|DEF-[0-9]|20[0-9]{2}-[0-9]{2}-[0-9]{2}'`
  hits are zero or each ruled not-history in the review; every `K-[0-9]+` hit is open in that
  component's `BACKLOG.md`.
- **AC-4, AC-8** — behavioural: `teco` confirms the §3.4 records on the first close after L3c;
  the first plan's review applies root `REVIEW.md`'s FR-12/FR-13 rule.
- **AC-5** — `wc -w` per `REVIEW.md` within FR-9, checked at L2b's gate.
- **AC-6** — `for a in analyst qa-engineer security-expert data-scientist graph-dba cobb teco tico; do grep -L 'REVIEW.md' claude/$a/$a.md; done`
  prints nothing (prompt files only; knowledge files never carry the pointer).
- **AC-7** — root `REVIEW.md` defines the proposal block; root `AGENTS.md`'s close bullet names
  applying and pruning.
- **AC-9** — this sweep (glob pathspecs, so `*` does not cross `/`; kaizen and process-document
  kinds fall outside it), run before L3c's gate and by it:

  ```
  git grep -n -i -E 'same change|alongside|part of done|doc updates|blueprint|DESIGN\.md' -- \
    ':(glob)**/AGENTS.md' ':(glob)**/CLAUDE.md' ':(glob)**/README.md' ':(glob)claude/*/*.md' \
    ':(glob)skills/**/*.md' ':(glob)**/docs/manuals/*.md' falkor-chat/docs/DESIGN.md \
    falkor-chat/docs/SERVER.md falkor-chat/docs/QUERIES.md ':(exclude,glob)deprecated/**'
  ```

  Run 2026-10-31: 107 hits, including every one Pass 4 lists. Hits needing an edit, by unit:

  | Hit | Unit |
  |---|---|
  | `claude/teco/teco.md` l.3, l.92, l.138, l.155; `claude/README.md` l.7 | L3c |
  | root `AGENTS.md` l.90, l.190; `falkor-chat/AGENTS.md` l.12, l.199; `falkor-chat/README.md` l.23, l.237 | L2c |
  | `claude/docs/manuals/agent-team.md` l.118, l.190 | L7 |
  | `falkor-chat/docs/DESIGN.md` l.1 and its other hits | L5a |
  | `falkor-chat/docs/SERVER.md` l.480 and its other hits; `QUERIES.md` hits | L5b; L5c |

  Every other hit (agent-catalog "same change" rules in `skills/agent-maintenance/SKILL.md` and
  `cobb.md`, "alongside" prose, citations of `DESIGN.md` sections or `kiro/DESIGN.md`) is ruled
  no-change at the gate.
- **AC-10** — the root `AGENTS.md` text (L2c); L6 is its first exercise.

## 6. Risks & open questions

- **Concurrent spec edits.** typed-decision-models U9–U12 touch `SERVER.md` and possibly
  `DESIGN.md`; before L3c they edit in-change. Serialize them with L5b and L5a.
- **`QUERIES.md` volume** (29k words, 90 `K-` references): `teco` may split L5c into two
  serialized dispatches by section range under one gate.
- **Human approvals.** L2c writes three files outside `cobb`'s allowlist; one approval per file is
  expected and is the intended checkpoint for always-loaded text.

No open questions.

## Ready to implement

Implement `docs/plans/live-specs.md`. Conventions go in root `AGENTS.md`; `REVIEW.md` content is
drafted by `analyst` in its review, applied by `teco`, gated by `cobb`; prompt pointers come from
`cobb` (L2a → L2b → L2c → L3a ∥ L3b → L3c → L7; L3c's commit is the switch-over). The falkor-chat
catch-up (L5a–c, `graph-dba`/`coder`/`graph-dba`, per-document homes, then the set-level L5s)
runs in parallel. L6, the model-bench spec, runs inside typed-decision-models' close.
