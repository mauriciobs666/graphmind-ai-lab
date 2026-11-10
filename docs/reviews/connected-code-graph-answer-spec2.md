# Review: canonical answer spec, P1 changes (`-answer-spec2.md`, step P1-00)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Pass 1 (2026-11-09)

### Scope and verdict

**Reviewed:** `docs/plans/connected-code-graph-answer-spec2.md` (uncommitted, architect), plus its one-line
`Extended by:` header edit to `docs/plans/connected-code-graph-answer-spec.md` (`git diff`: the header line only).
**Baseline:** the P1-00 scope list in `docs/plans/connected-code-graph-p1-answers.md`, and the steps that consume
the spec (P1-45, P1-22, P1-23, P1-24, P1-25, P1-26). Also index `docs/plans/connected-code-graph-p1.md` (D2, D3,
§4, §5, §9, §10), v1 of the spec, EP (`-eval-protocol-ml.md` §3.5, §5.2.1, §10.4, §12.3), the requirements
(FR-11, AC-1/2/5/6), and the code: `cpg/ccg/answer.py`, `cpg/ccg/schemas/answer-v1.schema.json`,
`cpg/tests/test_answer.py`, `cpg/tests/golden/answer-ac1.json`. I read two ungated sibling notes that already cite
spec2, `-p1-graph.md` and `-p1-ml.md` §6.5, as evidence only; I did not review them. Root `REVIEW.md` applies.
There is no `cpg/REVIEW.md`.

**Verdict: needs changes.** §1, §2, §4 and most of §7 can be implemented, and I checked them against the code.
Four things need fixing first:
- §5's labelling rule chooses a weaker chain over a stronger one (F1).
- §5 breaks P1-22's pinned expectations once P1-23 lands (F2).
- §8's done-condition is false: two existing tests go red under a faithful implementation (F3).
- §3.2 says `bound[]` evidence answers AC-2/AC-4, but EP scores wiring only as items (F4).

P1-45 can be dispatched once F3 is fixed. §5 must be revised before P1-23 is dispatched.

CPG: considered, not relevant — no loaded `cpg_*` graph indexes `cpg/`; every code claim below was checked by reading the files and by running the answer suite on a scratch copy.

### Findings

**F1 (blocker for P1-23) — §5 counts the flow relabel in the cost order, so the walk can pick a name-match chain over a resolved call.**
§5 ("Labelling") marks hop *k* `possible` "as each hop is traversed, so the cost order `(possible hops, distance,
length)` counts it". The exception is a hop *k* that is already `possible` on its own edge: it keeps that reason
and gets no extra mark.
- Take a caller `T` that calls the anchor through two sites: X is a resolved `CALLS ←` and Y is a `CALLS_NAME ←`
  (`name-match`). Both sites are in a flag-false unit.
- Under §5, X costs 1 possible hop (the flow relabel) and Y costs 1 (its own). The tie goes to stable id.
- Under v1/the slice, X cost 0 and always won. Now the item can show `reason: name-match` and the speculative chain,
  even though a resolved call exists. The answer gives the wrong reason (FR-8). It also skews EP §11.1's noise
  strata by reason.
- The per-state "already crossed" bit makes the cost depend on path history, which the visited key does not hold.
  When flags are mixed, two equal-cost arrivals with different bits can therefore keep the wrong one. "Re-expand
  only on strictly smaller cost" does not repair that. This is dormant in P1, where every flag is false.

Suggested change: count only edge-evidence `possible` hops in the cost, and apply the flow relabel as a pass over
each chosen chain afterwards (*k* = the first hop with a flag-false unit). The per-state bit then goes away, and V18
(a)–(d) still hold. The P1-23 test that catches this: two sites into one caller, one resolved and one name-matched,
in a flag-false unit. Expect the resolved chain with `flow-unavailable:<lang>`. This is a design change, so it is
`architect`'s decision. Verified by reading §5 against v1 §5.1 and P1-22 behaviour 7. Inferred, because no engine
exists yet.

**F2 (major) — Fail-closed flags combined with §5 make P1-22's pins false once P1-23 lands.**
P1-22 behaviour 4 pins "`apiFetch` stays `confirmed`" and the per-walk item sets. Behaviour 1 is equivalence with the
slice's item sets.
- `apiFetch` is in `salesperson/` (javascript). Under §5, its hop 1 is in a flag-false unit, so it becomes
  `possible`.
- §5 also says "a missing flag counts as false". P1-22's fixture graphs come before the flow fallback (P1-23 needs
  22), so unless they set `Unit.flowAvailable: true`, they turn red when `flow.py` is wired in.

Suggested change: §5 (or §8's hand-off) should require P1-22's equivalence and row fixtures to set
`flowAvailable: true` explicitly, so they test rows and not labelling. P1-23 then re-pins the same walks with
flags false (every item `possible`). Under the Q-1 alternative only the call-level continuation hops flip, so the
§5 choice decides which pins survive. Verified by reading `-p1-answers.md` P1-22/P1-23 and §5.

**F3 (major) — §8's "existing tests and bases stay green unchanged" is false, and the hop-level `keyPath` relaxation causes half of it.**
I applied §7.1 verbatim to a scratch copy and prototyped §7.2 (V3, V12, V17–V19). The full results are in the
Appendix. 591/592 tests pass with the schema edit alone, and 590/592 with the validator rules added:
- `test_the_required_table_covers_every_required_list_of_the_schema` fails: 105 ≠ 104. `Bound.required` loses
  `keyPath` (−1), and the `anyOf`'s two `required` lists count (+2). That walker is a coverage test, so item 10's
  "coverage tests pass with no change to their expected sets" is also false.
- `test_port_bounds_and_the_allowed_extras_of_a_launch_hop` (`test_answer.py:791`) expects `V3.*keyPath` when hop 0
  of 1.0 `b_launch` loses `keyPath`. With V3 relaxed, `launch` is present, so V17 fires instead. Every faithful
  implementation does this, because each rule "reports its own rule id".
- The hop relaxation has no producer. §3.2 says a `launchFact`-origin entry is "never an anchor and never an item.
  Its only place is `bound[]`", and `LAUNCHES` is prohibited. So no 1.1 chain can hold a `keyPath`-less
  `ConfigEntry` hop, and §8 case 4 "a 1.1 `ConfigEntry` hop with `launch` and no `keyPath`" pins a shape that is
  never emitted.

Suggested change: drop the `ConfigEntry`-hop relaxation and keep it on `Bound` only. Then V3 and line 791 stay as
they are, and V17's 1.0 clause shrinks to the `Bound` part. State in §8 that `REQUIRED["Bound"]` drops `keyPath`,
and that the count walker either skips `anyOf` or counts its two lists. Verified: run.

**F4 (major) — §3.2 makes the wiring chain `bound[]` evidence, but AC-2 and EP's `wiring` stratum score wiring only as items.**
§3.2 says "AC-2 and AC-4 need the chain listed… The answer carries it on the `Link` hop's `bound[]`."
- EP §3.5's served-anchor rule has the answer for any served contract point "show the server's binding entries" as
  `config`-subject items. That is H1's config branch, and H4 says "an element of `items` or `groups[].members`".
- All 11 dev items in `cpg/eval/dev/ac5-wiring.jsonl` have `impacted.type: config` with a `keyPath`. Example:
  `services.server` in `falkor-chat/compose.yaml`.
- EP §3.5 says listing server bindings under a route's answer is "P1's choice, not yet made". No P1 plan makes that
  choice (grep of `-p1*.md`).
- §3.2 adds no row that yields a `configEntry`-origin `ConfigSubject` item in impact/contract mode. Graph note §6.2
  gives `BOUND_BY` an impact row only `←` from a config anchor. As written, every wiring gold item misses by
  construction.

Suggested change: spec2 either states the row (for example, contract mode: at a `Link` hop, each
`configEntry`-origin `BOUND_BY` target becomes a `ConfigSubject` item, with distance and status stated), or names
the owner and the step that will decide. This is `architect`'s call, with `data-scientist` for the EP side.
Verified by reading EP §3.5, v1 §5.3 and the dev file.

**F5 (major) — With the §5 default, more gated checks become unable to fail than R-1 lists, and the method owner has already answered Q-1 the other way.**
I checked the consequence the architect flags against the sources:
- Index §5 ("Through a unit whose flag is false… every item is `possible`") and the proposal ("every item is shown
  as *possible*") support §5's literal reading.
- The golden test is consistent with it: the 1.0 golden stays confirmed, and §5's AC-1 form validates (Appendix).
- AC-1 itself is unaffected, because it only requires the code to be "listed".
- Requirements FR-11 is narrower: "falls back to who-calls-whom reach, **every such item** shown as possible".
  AC-1's item is not fallback reach, as §9 Q-1 itself concedes.

R-1 lists only the AC-5 confirmed-precision bar. Two more go the same way:
- P1-42's AC-6 assertion "none `confirmed`" (`-p1-gates.md` P1-42) can no longer fail.
- AC-6's "shown as possible **with its reason**" loses its specific reason at item level. A runtime-built consumer
  reached from a contract anchor gets item reason `flow-unavailable:python` from hop 1 (`PROVIDES ←`), not
  `runtime-built`.

Root `REVIEW.md` counts a check that cannot fail as a finding. `-p1-ml.md` §6.5 (the method owner, ungated)
recommends Q-1's alternative, with a measurement table, as stakeholder question S-2.

Q-1's arguments against need correcting:
- "Keeps the `function` state": no. §5 already defines the continuation as a row (a `Symbol` reached by `HAS_* ←`
  continuing by `CALLS ←`/`CALLS_NAME ←`/`TO ←` in state `value`), so the alternative can label that row.
- "Invalidates the golden test": that test is a 1.0 V6 registry test. A label-scoped V18 does not touch it.

Suggested change: add AC-6 to R-1, correct Q-1's arguments, cite `-p1-ml.md` §6.5, and note that V18 (d) is the one
P1-45 rule that depends on Q-1. P1-45 may land it, but it is removed if S-2 reverses §5.

**F6 (minor) — V19 (and V18, if it applies to 1.0) tightens 1.0, which contradicts §1.1's "every 1.0 document stays valid".**
§7.2 V19 "applies to both labels". A v1-valid 1.0 document with a `LAUNCHES` hop is now rejected. I confirmed this
on the golden (Appendix). §3 keeps the enum names precisely because removing a value is major, and V19 removes them
for 1.0 anyway. V18's label scope is not stated, though §8's remark about the golden test implies it runs on 1.0.
Suggested change: scope V18 and V19 to `"1.1"`, which no 1.0 producer needs, or say plainly that 1.0 is tightened
and why that is safe (no 1.0 producer ever emitted those types). Verified: run.

**F7 (minor) — The `ALGORITHM_VERSIONS` fingerprint covers less than §1.2 claims, and its `1.0` entry cannot be emitted.**
- "A mislabelled or unversioned answer cannot leave it" holds only for parts of the definition that are data, and
  only if the walk interprets that same record. The cost order, the fallback rule and the `unexplored` rule are
  code. A behaviour change there that leaves the record alone passes. §1.2 should require the walk's row table to
  be the record the walk executes, and narrow the claim to it.
- `1.0` "is in the table only if the P1 engine can still run the slice's definition". But V17 rejects any 1.1
  answer with a `function` state or without `bound`, and the slice's definition has both. So no valid answer can
  carry `"1.0"`. Drop the conditional: the equivalence harness compares item sets, not answers.
- Define "published" in "a published entry is never edited". It means: once an answer has been emitted on `main`
  under the entry. That is what lets P1-23 change `1.1`'s literal after P1-22 pins it.

**F8 (minor) — §3.2's "both may be present" disagrees with the graph contract that consumes it.**
`-p1-graph.md` G2: "the assembler writes exactly one of `keyPath` and `launchKind`", and readers tell the two
origins apart by which one is present. §8's `B-chain` entry 0 carries both, which is a shape the graph never
produces. With F3's change, `Bound` can be `oneOf` in 1.1, which makes the origin rule checkable. One of the two
documents must yield. Raise it with `graph-dba`.

**F9 (minor) — `B-flow` and §5 disagree on state.**
§5 says no hop in a flag-false unit has state `path`. The `site_hop` helper defaults to `state="path"` with
`accessPath: "r"`, so §8 should say `state="value"` for `B-flow`'s sites. Otherwise the base contradicts the rule
it fixtures. (My Appendix probe used `value`.)

**F10 (nit)** — §3 says EP needs no amendment. True as worded. But EP §5.2.1 planted input (h) (a `LAUNCHES` hop
must be a harness error) can now only be built as a V19-invalid document. P1-32's harness test has to build it
without `validate`. Worth one clause in §3.

### Edits owed in other documents (each checked against the target text)

| Target | Edit | Real? |
|---|---|---|
| `-p1-answers.md` P1-45 | Behaviours list V17–V19, `Bound.launch`, `cacheBypass` `const: true` + `cacheHit: false`; `reasons.yaml` and golden untouched | Yes: the step lists behaviours (1)–(5) only, and "(bool)" in (1) |
| `-p1-answers.md` P1-22 | (4) "`apiFetch` stays `confirmed`" vs §5 (F2); (8) to §1.2's mechanism; (5) "when the request names none" is now always true | Yes |
| `-p1-answers.md` P1-23 | (1) cites "answer spec §4.1" for the first-hop rule; replace with spec2 §5. Add §3.2's depends-on `BOUND_BY →` no-item rule | Yes: line 99–100; the rule is absent |
| `-p1-answers.md` P1-25 | MCP `ask` argument and `--cache-bypass` CLI flag (§2) | Yes: absent from P1-25's behaviours. Not in the architect's list as relayed in the ledger |
| `-p1-answers.md` P1-26 | (4) "forces a fresh call", plus "neither read nor written" | Minor wording |
| index §5 "What the engine does" | Cites `-answer-spec.md` §4.1. v1's sentence is §5.2, and spec2 §5 now replaces it | Yes: index line 243–244 |
| `-p1-gates.md` P1-37 | "confirmed precision ≥ 0.95" is reported vacuous at P1 (R-1) | Yes: P1-37 step 2 lists it as a bar |
| `-p1-gates.md` P1-42 | AC-6 "none `confirmed`" is vacuous under the §5 default (F5) | Yes; not listed by spec2 |
| EP §12.3 template | `"<semver>"` → `MAJOR.MINOR` (R-6) | Yes: EP line 1568. `-p1-ml.md` line 557 already records `"1.1"` |
| `-p1-graph.md` (R-2) | `snapshotId` hashes every `Unit.flowAvailable` | Already answered there (line 103). R-2 can cite it once that note is gated |
| `-p1-linking.md` P1-17 (6) | `ord` in chain order, 0 nearest the consumer | Absent from P1-17. Stated in `-p1-graph.md` §2, which P1-17 should cite |

### What's solid

- §1.1/§7.1: the enum widening, `Provenance.cacheBypass` with `const: true` plus the `if`/`then` on `cacheHit`, and
  `Bound.launch` as a `$ref`. They are minimal and classification-neutral. `unclassified_strings`,
  `derived_data_fields == DATA_FIELDS` and the array probe all stay green (run).
- §1.2's claims check out:
  - the schema pattern already admits `"1.1"` and rejects `"1.1.0"` (run);
  - `answer.py` pins no version value (read);
  - keeping the version table out of the validator is the right coupling call.
- §2: `cacheBypass` cites EP §10.4 instead of restating it. It stays outside `facts_bytes` (run: equal facts bytes
  with and without the field) and is not a `cacheKey` input.
- §3.1's per-type "what lifts it" table, and the claim that the `launchFact` record has no `keyPath`
  (`-bundle-spec3.md` §1.6, read).
- §4's `unexplored = |E_cut|` is order-independent and deterministic. It counts edges, not paths, and excludes the
  increment-0 rows and the raising guards. It matches P1-22 (5).
- Every §8 case I prototyped behaves as stated (Appendix). V18's three named mutants are killable:
  - "(b) on the item's unit": the existing golden acceptance kills it, because its hop unit differs from its item
    unit;
  - "(c) dropped": §8 7(c) kills it;
  - "L per item": 7(d) kills it.

### Growth-stops check

This is the first version of a new successor document, not a revision, so there is no rework growth to measure. It
states only changes against v1: §6 is a replacement map, not restated v1 content. It has no finding→resolution table
or disposition list. The reasons are stated as design. Two cross-document facts are cited, not re-homed (the HA-1
non-replay, D3's value). One restated claim is inaccurate: "index R-1 already carries the consequence" (F5). The
index's R-1 says "call-level and `possible`", not that the AC-5 bar is vacuous. Separately, the `Extends:` pointer
carries a document that replaces named v1 clauses (§6). This is index §4.2's open repository-rule question, so no
finding here, but v1's readers get no signal from `Extended by:` that §5.2's flow sentence is overridden.

### Open questions

1. **Q-1 / S-2 (stakeholder, after `-p1-ml.md` §6.5):** keep §5's every-hop-*k* labelling, or label only the
   flow-dependent continuation row? It decides F2's pins, F5's vacuous checks and V18 (d). It does not change the
   schema. It should be settled before P1-23, not before P1-45.
2. **Wiring items (F4):** who decides whether `configEntry`-origin binding entries are listed as items under a
   served contract point (EP §3.5 "P1's choice"), and in which document?

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`. Source: F3. The plan claimed "existing tests stay green unchanged", and
only a scratch run showed two red. It changes what a reviewer does:
- A plan's claim that existing tests or bases stay green under its schema or validator change is checked by
  applying the change to a scratch copy and running the suite, never by reading. A coverage walker that counts
  schema keywords (`required`, `type`) is part of "existing tests".

## Pass 2 (2026-11-09)

**Reviewed:** spec2 revised in place (5,534 words, uncommitted). Also its edits to `-p1-answers.md` (P1-45, P1-22,
P1-23, P1-25, P1-26), index §5/R-1, and `-p1-gates.md` (P1-37, P1-42).

**Verdict: approve with suggestions.** There is no blocker. P1-45 can be dispatched now with N2 folded into §8.
**N1 must be decided before P1-45 is dispatched**, because the recommended design adds to the schema. P1-23 waits
for N3.

CPG: considered, not relevant — no `cpg_*` graph indexes `cpg/`. The evidence is a toy walker and the answer suite
on a scratch copy (Appendix, Pass 2).

### Pass 1 findings

- **F1 — fixed, though not by the fix I suggested; the rejection holds.** I built a toy walker in the scratchpad:
  the `function` state is in the visited key, labels attach to rows, and the cost is
  `(possible, distance, length, own-possible, stable key)`. An exhaustive-path oracle and an engine-style
  relaxation walker agree on all six cases.
  - In the two-sites case, the resolved chain wins at distance 1 and at distance 2, including when the name-matched
    site sorts first by stable key (C2b, `flow-unavailable`).
  - Without the own-possible tie-break, C2b picks `name-match`, so the tie-break is what decides the case.
  - Labels depend only on the row and the state, so there is no path-history bit.
  - A residual, by design: own-possible ranks after distance and length. A nearer name-match chain beats a farther
    all-resolved one (C3), and a chain with fewer possible hops wins even when its one hop is own-possible (C4).
    Both items are `possible` either way, so this is not a finding.
- **F2 — fixed.** P1-22 (4) now sets `Unit.flowAvailable` explicitly in every fixture. `apiFetch` stays
  `confirmed` under §5: a `CALLS ←` leaving an anchor in `value` is not a labelled row (walker C1).
- **F3 — fixed.** The `ConfigEntry`-hop relaxation is gone. Rerun results:
  - 592 passed at baseline;
  - 591 passed and 1 failed with the §7.1 schema alone (105 ≠ 104);
  - **593 passed** with §8's `REQUIRED`/`INSTANCES` edit and the §7.2 prototype;
  - line 791 stays green.
- **F4 — fixed.** The §3.3 wiring row and V5's second form are in. `B-wire` validates, and AC-2 in P1-42 now says
  `config` items. The EP §3.5 citation is still owed by `data-scientist`.
- **F5 — superseded** by the stakeholder decision in §5. Q-1 is removed. The AC-5 confirmed bar and AC-6's "none
  confirmed" can now fail: contract consumers and wiring items stay `confirmed`, and AC-6 kinds keep edge-level
  reasons (P1-42 edited).
- **F6 — fixed.** V18 and V19 apply to `"1.1"` only. The 1.0 golden with each prohibited type is accepted (run).
- **F7 — fixed.** §1.2 now scopes the fingerprint to the data record plus code identifiers, defines "published",
  and drops `1.0` from the table.
- **F8 — fixed.** `Bound` is `oneOf` `keyPath`/`launch`, and `B-chain`'s entries carry one each. `keyPath` added
  to a launch entry is rejected (run).
- **F9 — fixed.** `B-flow` sites use `state="value"`.
- **F10 — fixed.** §3 has the planted-input (h) clause.

### The stakeholder decision in §5: implemented faithfully

- **V18 (a)–(d) encode the closed row list** (prototype: `FUNCTION_ENTRY`/`FLOW_ROWS` keyed by direction).
  - `B-1.1` (AC-1) validates `confirmed`.
  - `flow-unavailable` on its `PROVIDES ←` is rejected (c).
  - Every §8 7 rejection reproduces.
  - 1.1 `b_possible` is accepted.
- **No check goes vacuous.**
  - P1-37's own-status split is readable from the answer, which V18 (c) guarantees.
  - AC-25's data-flow item (two calls away) is still `possible`, as index §5 states.
- **Two gaps, filed as N2 and N4:** V18 is weaker than the closed list, and one named mutant is not killed.

### New findings

**N1 (major) — O-1's default leaves an AC that P1 claims to close partly open, and P1-38 still claims it.**
§3.4/O-1: with no `TO_UNIT` row, a depends-on answer through a link into a never-built unit lists nothing. The
never-built unit is also absent from `staleness.units`.
- AC-9b says "anything crossing into it is shown as possible with that reason". FR-7a says the same, and "shown"
  means shown in an answer.
- The proposal's P1 row lists AC-9b (proposal line 1086, "AC-9b (P1)"). P1-38 still lists "AC-9b closed"
  (`-p1-gates.md` line 235).
- Only a contract-mode ask that carries the `possible`/`target-not-mapped` `Link` shows the crossing. That is
  inferred: no P1 text says P1-41's never-built fixture asserts it.

The recommended design (a `Unit` hop kind, `TO_UNIT`, a `unit` subject) adds to the schema, which is P1-45's
scope. Deciding it after P1-45 means a second validator round or schema 1.2. Suggested change: `architect` (with
`data-scientist`) decides O-1 now. If the default stands, deferring part of a P1 AC needs the stakeholder, recorded
by `tico`; P1-38 and index §8 then report AC-9b's reverse crossing as not closed. Verified by reading.

**N2 (minor) — The V18 (d) mutant §8 names cannot be killed by §8's cases.**
The mutant is "(d) reading the hop's own unit instead of hop *i*−1's". Every `B-flow` hop is in `U1`, and a
labelled `CALLS ←` site shares its caller's unit, so the mutant run shows **604 passed**, all green. A labelled
`TO ←` hop (provider in `function` → `Link`, no unit → consumer site) kills it: the mutant rejects the valid
document (run). Suggested change: add that case to §8 7. Root `REVIEW.md`: a check must be able to fail.

**N3 (minor) — P1-23's planted two-sites test passes for a walk without the tie-break unless the ids are ordered.**
The test is "one caller reached through a resolved and a name-matched site; expect the resolved chain". If the
resolved site's id sorts first, the stable key alone picks it, and the test passes without the own-possible
tie-break (walker C2 vs C2b). Suggested change: in P1-23's text and spec2 §5, require the name-matched site to sort
first by stable key, and name the mutant (the tie-break dropped).

**N4 (minor) — V18 does not enforce "these and no others" on the way out of `function`.**
I took `B-flow` item B, changed hop 3 (leaving `fb` in `function`) to `PROVIDES ->`, `confirmed`, and made the
item `confirmed`. The prototype accepts it (run). So an engine that continues out of `function` by an unlisted row
emits a `confirmed` item the stakeholder rule forbids, and the validator does not see it. Suggested change: add
(e): a hop whose predecessor is in state `function` arrives by a row in `FLOW_ROWS[direction]`. The plant above is
its red case.

**N5 (minor) — The stakeholder decision has no home outside the ledger, and it changes requirement wording.**
§5 cites a "2026-11-09 decision", but the record is only the ledger (`-coordination.md` line 662). The decision
narrows the proposal's "every item is shown as *possible*" and FR-11's reading. Suggested change: `tico` records it
in the requirements decision log as one question and one answer citing FR-11, and spec2 §5 cites that entry. The
proposal sentence then needs a pointer or a note in the index's open list.

**N6 (minor, growth-stops) — §7.2's "Scratch verification (2026-11-09)" paragraph is a dated process record inside
a plan.** It says what was run and when, which is history. The run's evidence belongs in this review or the ledger.
Its "13 prototype tests" are not identified in §8: I count 11–12 tests for the same cases. Suggested change: delete
the paragraph and keep §8's statement of the one expected test edit.

**N7 (nit)** — Two small gaps:
- Per-site `siteId` is checked against a following `CallSite` only. A wiring `Link` hop's single entry may carry a
  `siteId` no rule constrains, and "listed on every site's copy" is not checkable.
- P1-37's parenthetical "(measured on the items no labelled row … makes `possible`)" restates what "confirmed"
  already means, and can be misread as excluding own-possible items. Take it or leave it.

### New scope judged

- **`Bound` port evidence, `siteId` and `oneOf`.** The schema behaves as §8 4–5 states: `port` without
  `portOrigin`, `packId` with `compose`, and `pack-default` without `packId` are all rejected; `pack-default` with
  `fastapi` is accepted; a wrong `siteId` is a V5 failure (run).
- **Wiring items.** These are sound. Distance is consistent with the slice: `TO ←` adds 0 and `VIA ←` adds 1
  (`impact.py`). With `BOUND_BY ->` adding 1, a wiring item lands at distance 1, as `B-wire` says.
- **`builtAt`/`failedAt` outside 1.1 `facts_bytes`.** I accept the exception.
  - `answerSchemaVersion` is itself inside `facts_bytes`, so no 1.0-vs-1.1 facts equality exists to break. The
    property v1 §9 protects (stored 1.0 comparisons) holds.
  - `-p1-graph.md` §3 hashes `buildStatus`, `bundleHash` (which covers `builtCommit`) and `failReason` into
    `manifestHash`, but not the build times. Without the projection, FR-19 fails on every rebuild.
  - Verified: mutating the projection away makes the `builtAt` case red (run). A pointer from §6's table to v1 §9
    would make the exception findable.
- **`TO_UNIT`.** See N1.

### Cross-document edits

All edits named in Pass 1 are made. The unlisted P1-22 (7) and Files-line edits match §1.2 and §5. Not yet made:
- P1-17 (6) still lacks the `ord` rule. `-p1-graph.md` §2 states it, and spec2 cites that.
- EP §3.5 and §12.3 are `data-scientist`'s.

**For `teco`:** the uncommitted diffs to the index and `-p1-gates.md` also carry U36's `-egress-spec3.md` edits
(P1-28/29/31, index §3/§6/§7 rows), and P1-25 (7)'s import-boundary rewrite is egress-spec3's. A commit of P1-00
should not sweep them in unexamined.

### Growth-stops

Pass 1's document was 412 lines; this one is 544 lines and 5,534 words.
- **New scope, legitimate:** §3.2 port evidence and `siteId` (following the graph contract), §3.3 wiring (F4's
  answer, a new contract row), §3.4, §4.3 and O-1.
- **Rework, done properly:** §5 was replaced in place, with the stakeholder answer folded in and Q-1 removed. §9
  shrank, and the review cross-references are gone. §1.2's mechanism grew by a few lines while correcting an
  overclaim (F7). That is a replacement, not stacking.
- **Rework growth to remove:** §7.2's dated scratch-verification paragraph (N6).
- **Restated facts:** none found beyond what is cited.

## Pass 3 (2026-11-10)

**Reviewed:** spec2 rev 3 (6,109 words, uncommitted), plus its edits to `-p1-answers.md` (P1-45, P1-23),
`-p1-linking.md` (P1-17 (6)) and `-p1-gates.md` (P1-37).

**Verdict: approve with suggestions.** There is no blocker. P1-45 can be dispatched once P3-1 is folded into §8.
P3-2 must be settled with `graph-dba` before P1-23.

CPG: considered, not relevant — no `cpg_*` graph indexes `cpg/`. The evidence is the scratch prototype, extended to
rev 3 (Appendix, Pass 3).

### Pass 2 findings

- **N1 — fixed** by the new §3.5 (`Unit` hop, `TO_UNIT`, `UnitSubject`, V9's not-mapped check, V17 keeping it out
  of 1.0). P1-38's "AC-9b closed" is consistent again.
  - `B-unit` validates.
  - Making `U9` `fresh` is a V9 failure. A `TO` edge, `unit ≠ id`, or the direction flipped is a V5 failure.
    Relabelled 1.0, it is a V17 failure (run).
  - The not-mapped unit's `reason` is available through `-p1-graph.md`'s `units` read.
- **N2 — fixed.** Item C (labelled `TO ←`, no unit on the hop) kills the V18 (d) own-unit mutant (run).
- **N3 — fixed.** §5 and P1-23 now require the name-matched site to sort first by stable key, and name the mutant.
- **N4 — fixed.** With V18 (e), a `PROVIDES ->` hop leaving `function` is rejected, and dropping (e) is caught
  (run). V18 (e) does not reject 1.1 `b_possible` or `B-flow` (run).
- **N5 — not fixed (out of the architect's scope).** The stakeholder decision is still only in the ledger. The
  requirements decision-log entry is owed by `tico` (grep finds no 2026-11-09 entry in
  `docs/requirements/connected-code-graph.md`).
- **N6 — fixed.** The dated scratch paragraph is removed.
- **N7 — fixed in part.** The P1-37 parenthetical is removed. On `siteId`, see P3-3.

### New design: direction-aware V5

- **The claim is confirmed.** v1's V5 rejects every depends-on chain through a link. On the unmodified tree code, a
  well-formed `SA → HAS_CALL -> site → VIA -> Link → TO -> provider` chain fails with "V5: … a Link hop is followed
  by its CallSite hop (VIA <-)" (run).
- **The relaxation does not admit malformed 1.0 documents.** It applies to every label, so a 1.0 depends-on chain
  through a link becomes valid; that is a relaxation, and minor by v1 §9. I prototyped the §7.2 V5 text. On a 1.0
  depends-on base it rejects:
  - an impact-shaped exit (`VIA <-` after the `Link`);
  - a `Link` entered by `TO`;
  - `lifted` naming another site;
  - the direction flipped to impact;
  - a `Link` not entered from a `CallSite` (all run).

### New findings

**P3-1 (minor) — §8 has no case for the relaxation V5 adds, so its depends-on clauses survive the mutation check §8 claims.**
§8's only depends-on base is `B-unit` (`TO_UNIT`). It has no depends-on chain that leaves a `Link` by `TO ->` to a
`Symbol`, which is the shape the relaxation exists for. With §8's listed cases only, both of these mutants pass,
**609/609 green**:
- the depends-on entry check dropped (M7);
- the exit check dropped (M10).

My five malformed-1.0 probes kill both (run). Suggested change: add `B-dep` (the well-formed 1.0 chain above),
accepted on both labels, plus the entry and exit rejections, to §8 8.

**P3-2 (major) — The wiring row (§3.3) has no defined `BOUND_BY` edge and no read, so its `Bound`'s `ord`/`siteId` is unspecified.**
`-p1-graph.md` G7 makes `BOUND_BY` per consumer site: every edge carries `siteId`, and `ord` is per (link, site).
`bound_out` reads per (link, site).
- §3.3's row starts at a `Link` "reached by `TO ←`", which enters no site. One entry bound for several sites has
  several edges with different `ord`/`siteId`.
- §3.3 does not say which edge supplies the item's single `Bound`, and no reader text returns entries per link.
  The wiring item's `facts_bytes` therefore depend on a choice P1-23 must invent, and FR-19 needs it fixed.
- Suggested change: state the rule (for example, the entry once, with the lowest `(siteId, ord)` edge's `ord`, and
  no `siteId`). `graph-dba` adds a per-link `bound_of_link` read. Inferred from reading §3.3 against G7 and
  `bound_out`.

**P3-3 (minor) — The rule for `siteId` disagrees with the graph.**
- §3.2 makes `siteId` optional, and says "an entry without a `siteId` binds the whole link and is listed on every
  site's copy".
- G7 writes a `siteId` on every `BOUND_BY`, and `-p1-graph.md` (after `bound_out`) says "the read is per site, so
  `Bound` needs no `siteId`".

Either way the contract holds, but the case §3.2 describes is never produced. Pick one statement and cite it.

**P3-4 (minor, owed by `graph-dba`) — `-p1-graph.md` still states pre-rev-3 answer-side rulings.**
- `to_unit_out` is marked "read only if the answer side adopts a `TO_UNIT` row, §9 A3". The answer side has now
  adopted it.
- After `bound_out`, the text says `b.port`/`portOrigin`/`packId` "have no `Bound` field yet (§9 A1) … the engine
  reads them and does not emit them". §3.2 now emits them.

Those lines need flipping when that note is revised, together with P3-2's read.

### Same-revision consistency and growth-stops

- **Same-revision fixes checked against each other.**
  - V18 (e) with `b_possible` (its exits are `CALLS_NAME ←`/`CALLS ←`), the wiring row and `TO_UNIT`: no conflict.
  - `UnitSubject` ordering (`rootPath`, 0) and V7 (`unitId` = the last hop's `unit`) hold in `B-unit`.
  - The classification probe, `DATA_FIELDS` and the array probe are unchanged.
  - The full suite plus prototype: **595 passed** in the existing suite with §8's table edits, and **610 passed**
    with the prototype module.
- **Growth-stops.** The document grew by 575 words.
  - **New scope:** §3.5 and the direction-aware V5. The second is a v1 defect fix with a contract change, so it is
    new scope.
  - **Rework, done in place:** N2–N4 in §5/§8, and N6's paragraph deleted.
  - No restated history or facts.

## Pass 4 (2026-11-10)

**Reviewed (narrow pass):** spec2 rev 4 (§3.2, §3.3, §7, §8), together with `-p1-graph.md` v3 (§5 reads, G7, §8
check (f)).

**Verdict: approve.** No blocker or major remains. **P1-45 can be dispatched.** P1-23 is unblocked as far as this
document goes.

CPG: considered, not relevant — no `cpg_*` graph indexes `cpg/`. The evidence is the scratch prototype, updated to
rev 4.

### Pass 3 findings

- **P3-1 — fixed.** §8 5 adds `B-dep` (accepted on both labels) and four V5 rejections. With my Pass-3 malformed
  probes deselected, the two mutants that survived are now killed by §8's own cases (run):
  - the entry check dropped is caught by "entered by `TO`";
  - the exit check dropped is caught by "impact-shaped `VIA <-` exit".
- **P3-2 — fixed.**
  - §3.3 now fixes the wiring `Bound` to the edge with the lowest `(siteId, ord)`.
  - `-p1-graph.md` §5 adds `bound_of_link`, ordered `ORDER BY i, e.entryId, b.siteId, b.ord`. The engine keeps
    the first row with a `keyPath` per `(i, entryId)`.
  - That pick is deterministic. G7 allows one edge per (link, site, entry), so the sort key is total. `keyPath` is
    an entry property, so the filter keeps or drops all of an entry's rows together. The first row is therefore
    exactly the lowest `(siteId, ord)`, which satisfies FR-19.
  - The read returns every `Bound` field and every subject field (`entryId`, `repoId`, `unitId`, `path`, `line`,
    `keyPath`, port evidence). The graph note's check (f) tests the two-site case.
  - Verified by reading. I did not run it against a database.
- **P3-3 — fixed.** `Bound.siteId` is removed from §3.2, the §7.1 schema row, V5, V17 and §8. The per-site rule now
  holds by construction of `bound_out`'s `(link, site)` read, which matches G7 and the graph note.
  - In the prototype with `siteId` gone from the schema and V5, a `Bound` carrying `siteId` is rejected by the
    schema.
  - The full run: **611 passed**.
- **P3-4 — fixed.**
  - `to_unit_out` now cites spec2 §3.5 instead of the "read only if adopted" condition.
  - The "no `Bound` field yet" line is gone: "each `bound_out` row is a `Bound` … port fields included".
  - `mounted_by_out` is removed, with "`LAUNCHES` and `MOUNTED_BY` have no reader text in 1.1". No P1 document
    consumes it: grep finds only spec2 §3.4, and the linker's `MOUNTED_BY` use is graph-side.

### Nits (take or leave)

- §3.4 still names "`-p1-graph.md` §5's `mounted_by_out`", which no longer exists. Say "no reader text (§5)"
  instead.
- §3.3 still says `bound_of_link` is "to be added", and that it returns "configEntry-origin targets". The read
  returns every target, and the engine filters on `keyPath`. Present tense, and "the engine keeps …", would match
  the note.
- §8 5's "entered by `TO`" case must also drop `candidateCount`. Otherwise V3 ("candidateCount … exactly when
  edgeType is VIA") fires before V5 (run), and a test expecting `V5` goes red on its own mutation.
- §8 5 files "a `lifted` entry naming another site" under the exit clause. It exercises the `lifted` rule, not the
  exit.

**Same-revision interaction.**
- Wiring `Bound` from `bound_of_link`, per-site `bound_out`, no `siteId`, V5's wiring clause ("exactly that entry,
  no `lifted`") and V17's 1.0 list: no conflict.
- N5 (the requirements decision-log entry, `tico`) is still open, outside the architect's scope.

**Growth-stops:** the diff is a replacement in place (siteId removed, wiring-edge rule added, §8 5 cases). It
records no history.

## Appendix — scratch runs (`cpg/` sources copied to the session scratchpad, `PYTHONPATH` pointed at the copy; the working tree untouched)

**Pass 4.**
- **Prototype updated to rev 4:** `Bound.siteId` removed from the schema and from the V5/V17 checks. §8 5's
  `B-dep` and its four V5 rejections added, with strict `^V5` patterns.
- **Results:** 611 passed.
- **Mutants**, with my Pass-3 malformed-1.0 probes deselected so that only §8's cases run:

  | Mutant | Result | Killed by |
  |---|---|---|
  | V5 depends-on entry check dropped | **killed**, 1 failed / 609 passed | "entered by `TO`" |
  | V5 depends-on exit check dropped | **killed**, 1 failed / 609 passed | "impact-shaped exit" |

- **Rule order:** "entered by `TO`" with `candidateCount` kept fails at V3 first.

**Pass 3.**
- **Prototype extended to rev 3:**
  - schema: the `Unit` kind, `TO_UNIT`, `UnitSubject`;
  - V3, V4 and the order key for `Unit`;
  - direction-aware V5;
  - V9's not-mapped check;
  - V17 without `Unit` in 1.0;
  - V18 (e).

  The test tables were updated per §8: `REQUIRED`, `INSTANCES`, `ENUMS` and the `$defs` set.
- **Results:** `test_answer.py` gives **595 passed**. With the prototype module, **610 passed**.
- **Mutants** (run against both modules):

  | Mutant | Result |
  |---|---|
  | M1, V18 (d) reads the hop's own unit | killed (item C, and the Pass 2 `TO ←` case) |
  | M5, V18 (e) dropped | killed |
  | M6, V9 not-mapped check dropped | killed |
  | M8, V5 direction fixed to impact | killed |
  | M9, V5 `Unit` clause dropped | killed |
  | M7, V5 depends-on entry check dropped | killed only by my malformed-1.0 probes; **609 passed** without them |
  | M10, V5 depends-on exit check dropped | killed only by my malformed-1.0 probes; **609 passed** without them |

- **v1 V5 on the unmodified tree:** a well-formed depends-on chain through a link is rejected.

**Pass 2.**
- **Toy walker** (`walk.py` in the session scratchpad):

  | Case | Chosen chain | Item |
  |---|---|---|
  | C1, distance-1 two sites | the resolved site | `confirmed` |
  | C2/C2b, distance-2 two sites | the resolved site | `possible`/`flow-unavailable` |
  | C2b, mutant without the own-possible tie-break | the name-match site | `possible`/`name-match` |
  | C3, near name-match vs far resolved | the name-match chain, cost `(1,1,3,1)` | `possible` |
  | C4, own-possible vs two flow labels | the `resolution-unverified` chain | `possible` |
  | C5, flag-true intermediate caller | no labelled row | `confirmed` |

  The exhaustive oracle and the relaxation walker agree in every case.
- **Answer suite** (prototype of §7.1 and §7.2: V5, V12, the 1.1 `facts_bytes` projection, V17, V18 (a)–(d), V19):
  - schema only: 591 passed, 1 failed (105 ≠ 104);
  - with the `REQUIRED`/`INSTANCES` edit: **593 passed**;
  - the §8 prototype module: 11 tests pass, 12 with N2's labelled-`TO ←` case.
- **Mutants** (run against both modules):

  | Mutant | Result |
  |---|---|
  | M1, V18 (d) reads the hop's own unit | **604 passed (survives)**; killed only by N2's case |
  | M2, V5 wiring without exactly-one | killed |
  | M3, V18 (b) dropped | killed |
  | M4, 1.1 `facts_bytes` projection dropped | killed |

- **N4 probe:** a `PROVIDES ->` hop leaving `function`, with a `confirmed` item, is accepted.

**Pass 1.**

- Baseline: `cd cpg && .venv/bin/python -m pytest tests/test_answer.py -q`: **592 passed**.
- §7.1 applied verbatim (enum, `cacheBypass` + `if`/`then`, `Bound.launch` + `anyOf`, `keyPath` out of `required`):
  **1 failed, 591 passed**. `test_the_required_table_covers_every_required_list_of_the_schema`:
  `assert 105 == 104`.
- Plus a §7.2 prototype (V3 `keyPath`-or-`launch` on `ConfigEntry`, V12 conditional with `Bound.launch` screened,
  V17, V18 (a)–(d) on both labels, V19 on both labels): **2 failed, 590 passed**. The second failure is
  `test_port_bounds_and_the_allowed_extras_of_a_launch_hop`: `Expected regex: 'V3.*keyPath'`,
  `Actual message: 'V17: items[0]: 1.0 ConfigEntry needs keyPath'`.
- Probes:

  | Probe | Result |
  |---|---|
  | `B-1.1`, `cacheBypass: true`, `B-chain`, `B-flow`, `algorithm.version "1.1"` | ACCEPT |
  | `"1.2"`, `"1.1.0"`, `cacheBypass: false`, `cacheBypass` + `cacheHit: true`, `Bound` with neither field, `Bound.launch.args` | REJECT (schema) |
  | `cacheBypass` on 1.0; `B-chain` as 1.0; 1.1 without `bound` / `bound: 1`; `b_possible` as 1.1 | REJECT (V17) |
  | `Bound.launch.service` unmasked 25-letter | REJECT (V12) |
  | each of the four prohibited types on hop 1, 1.1 **and 1.0 golden** | REJECT (V19) |
  | `BOUND_BY`, `RAN` on hop 1 | ACCEPT |
  | `B-flow` second reason / `javascript` reason / item B confirmed | REJECT V18 (a) / (b) / (d) |
  | golden as 1.1 with hop 1 `flow-unavailable:python` (AC-1 as §5 emits it) | ACCEPT |
  | `facts_bytes` with vs without `cacheBypass` | equal |
