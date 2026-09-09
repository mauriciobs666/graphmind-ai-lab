# Change History — model-bench

> Dated log of actual changes to the `model-bench` component. Most recent first.

## 2026-09-09 — S2 U73: the row-count identity's own exemption widened to match its stated reach

**What:** `_row_count_identity_problems` (`modelbench/packs.py`) skipped silently — returned `[]`
— for *any* pack missing `data.conversations`, including a `scripts`-declaring (conversation-
shaped) pack that simply omitted the key; only an item-level pack (no `scripts` declared at all)
was the shape its own docstring named as the intended exemption. Verified by execution before
changing anything: a pack copied from `tests/fixtures/packs/valid` with `data.conversations`
deleted from the manifest but `scripts`/`replicatesPerScript`/`analysisUnit` intact passed
`validate_pack` with no problems. Closed per option (a) — widened the mechanism rather than the
docstring — because §3.3's own manifest literal and the "conversation pack" run-shape rule
(`docs/plans/small-model-benchmarking.md` §3.9 point 2) both tie `scripts` to conversation shape,
so a `scripts`-declaring pack without a rows file is a real problem, not "nothing to check."

**Change:** `_row_count_identity_problems` now skips only when `scripts` itself is absent or not a
plain int (the item-level exemption, unchanged). A pack that declares `scripts` but has no
`data.conversations` now gets exactly one reported problem naming the gap; `replicatesPerScript`
and `analysisUnit` validity are unchanged and still skip silently when malformed (left out of this
fix's scope). One existing fixture, `tests/fixtures/packs/replicates_per_script_violation`, had
declared `scripts: 6, replicatesPerScript: 2` with no `data` block at all — a pre-existing instance
of the same gap, invisible only because the route was exempt — so it gained a `data.conversations`
key and a matching 12-row `conversations.jsonl` (6 distinct `scriptId` values × 2) to keep its
Rule‑6‑only test isolated to the one violation it's meant to pin.

**Test-first, confirmed failing for the stated reason.** New fixture
`tests/fixtures/packs/missing_data_conversations/` (declares `scripts`/`replicatesPerScript`/
`analysisUnit`, no `data` block) and
`test_validate_pack_rejects_a_scripts_declaring_pack_missing_data_conversations` in
`tests/test_packs.py`, run alone against the pre-fix code: `assert [] == ['fixture-mis...']` —
red for the right reason before any production change.

**Mutations, both caught, each `cp`-aside / mutate / run / `cp`-back, `diff -q` byte-identical
after each restore:** (1) the whole file reverted to the pre-fix version — the new test alone goes
red with the exact same `[] == [...]` failure observed pre-fix, the sibling
`test_validate_pack_rejects_replicates_per_script_greater_than_one` still passes; (2) with the fix
in place, `declares_scripts` forced to the constant `True` — 5 of 29 `test_packs.py` tests redden
(`test_validate_pack_rejects_analysis_unit_outside_pairing_key_structurally`, both call-surface
rejection tests, the bad-import test, and `test_validate_pack_accepts_a_module_importing_
modelbench_tooling`), proving the item-level fixtures' green results depend on the exemption guard
rather than passing by accident.

**Observed, this run:** `model-bench/` as working directory. Baseline before any change:
`708 passed, 1 deselected`. After the fix, full suite: `709 passed, 1 deselected` (the one new
test; no other count moved). `.venv/bin/ruff check .`: `All checks passed!`.

**Files:** `modelbench/packs.py`, `tests/test_packs.py`,
`tests/fixtures/packs/missing_data_conversations/pack.json` (new),
`tests/fixtures/packs/replicates_per_script_violation/pack.json` and `conversations.jsonl` (new
data file). Left uncommitted for review; a separate concurrent unit appends its own entry to this
same `HISTORY.md`.

## 2026-09-09 — S2 U72: the LM Studio adapter, offline against stubbed HTTP

**What:** `docs/plans/small-model-benchmarking.md` §3.4.4a/§3.6's LM Studio adapter, landing beside
a concurrent unit building `modelbench/packs.py` (not touched here). `modelbench/lmstudio.py`
(new), `tests/test_lmstudio.py` (new), `tests/fixtures/lmstudio/` (new). `tests/test_lmstudio.py`
contributes **32 selected tests plus 1 deselected** (`-m live`) to the suite — this unit's own
attributable delta; the suite-wide total is reported separately, in the concurrent pack-loader
unit's entry above. `.venv/bin/ruff check .` clean on this unit's files. **Twelve mutations, all
caught**, each `cp`-aside / mutate / run / `cp`-back restore, diffed byte-identical against the
pre-mutation file after every single one: dropping the seconds→ms conversion; wrongly converting
`tokensPerSecond` (the one figure that must stay a raw rate); `_seconds_to_ms` returning `0.0`
instead of `None` on a missing key; removing the `stats or {}` guard so construction raises on a
missing `stats`; inverting `toolCallForm`'s native/prose branch; widening the eligibility gate's
`type` predicate to admit `"embeddings"`; changing the gate's scope constant from `"tool-caller"`
to `"embedder"`; flipping `residency()`'s filter operator; inverting `warm_up`'s residency
membership test; moving `warm_up`'s residency probe to *after* the chat call (an ordering defect,
not a branch); computing `EmbedResult.dimension` from `len(vectors)` instead of the first vector;
and swapping which raw `stats` key feeds `ttftMs` versus `generationMs`.

**One test caught not pinning what it claimed — by mutation, before review, not after.** The first
version of `test_warm_up_checks_residency_before_issuing_the_call_not_after` stubbed the residency
catalog to change on the *second* call to `/api/v0/models`, but `warm_up` only ever calls that
endpoint once regardless of where the call sits relative to the chat request — so the ordering
mutation above (residency probed after the timed call instead of before) left the test green. Its
name and docstring claimed to pin the ordering; its assertions did not. Rewritten to key the stub's
answer on whether the chat call has actually fired yet, re-confirmed green against the correct
implementation, then re-confirmed it fails under the same mutation that previously slipped past it.
Recorded because this is the first time on this coordination the *test-whose-name-outruns-its-
assertions* class was caught by the implementer during mutation testing rather than at review.

**Delivered.** `LMStudio(base_url, *, opener=urllib.request.urlopen)` — `base_url` a constructor
parameter (the not-yet-built `hostinfo` unit supplies it from `host.json`); `opener` injectable the
way `falkorchat/transport.py`'s HTTP transport is, so every test but the one `-m live` test stays
offline. `catalog()`, `residency()`, `probe()` (`GET /api/v0/models`, filtered, and the two-step
reachability probe — `"api-v0"` / `"v1-only"` / `"unreachable"`, all three tested against stubbed
HTTP); `chat()`, `embed()`, `warm_up()` with `timeout_s` required and **no default** on all three
(§3.6's two budgets belong to the runner, a later unit). Deliberately **no `load`/`unload`/`ps`**
— the CLI is gone and nothing on either HTTP surface can unload a model.

`ChatResult` normalises LM Studio's seconds-valued `stats.time_to_first_token`/`generation_time`
into `ttftMs`/`generationMs` on construction (§3.6's unit boundary, plan-gate P4-1); `tokensPerSecond`
is the one figure left unconverted. Each of the three is `None` — never `0` — when its source key
is absent, and construction never raises on a missing or partial `stats` object (plan-gate P5-8),
mutation-tested three separate ways above. `ChatResult.toolCallForm` (`"native"` / `"prose"`,
FR-8(b)) is decided at the transport boundary, on the one fact only this layer can observe
directly — whether the response used LM Studio's native `tool_calls` mechanism — rather than
deferred to a later prose-heuristic scorer. Transport failures raise one of two distinguishable
exceptions, `LMStudioCallTimeout` versus `LMStudioCallFailed`, matching §3.6's "timeout" versus
"no_response" dispositions.

`tool_calling_eligible(model_info)` / `check_tool_calling_eligibility(role, model_info)` implement
§3.6's eligibility gate. `role` is a plain `str`, **never a `Pack` object** — `packs.py` was a
concurrent unit this wave and its shape was not final; the wiring unit is expected to call
`check_tool_calling_eligibility(pack.role, model_info)`. Tested against the three real catalog
entries that break the naive `"tool_use" in capabilities` rule (an `embeddings` model advertising
`tool_use` → refused; an entry with no `capabilities` key → admitted; an `llm` with `tool_use` →
admitted), plus the v1.11/plan-gate-P5-1 negative case: the same `embeddings`-advertising-`tool_use`
entry is admitted, un-gated, on an `embedder` pack.

**Fixture note.** No literal captured `GET /api/v0/models` 19-model payload exists anywhere in this
repo's docs (checked: plan §2.5, review Pass 1 Appendix A.2, review Pass 4 Appendix D.3 — all
narrative descriptions of a live probe, never a saved response). `tests/fixtures/lmstudio/catalog.json`
holds the 7 entries the docs record a field for, each cited in a `_provenance` block, rather than a
fixture padded to 19 with invented models. The "llm with `tool_use`" entry (`qwen/qwen3-4b-2507`)
reuses `tests/conftest.py`'s own established S1 fixture precedent for that model (`modelType: "llm"`,
`modelCapabilities: ["tool_use"]`) rather than an independent capture — worth closing properly in a
future pass over the plan's fixture framing, not fixed here.

**Not this unit's:** `run`'s capture-order sequence, the two timing budgets in anger,
`coldLoadSeconds`, the withholding dispositions, `LatencyBlock`, and everything in `hostinfo.py` are
the runner/host-info units'. One `-m live` test is written
(`test_live_catalog_and_chat_stats_against_a_real_lm_studio`) and deselected by default — not run;
no model may be loaded by an agent.

## 2026-09-09 — S2 U71: the real pack loader (`load_pack`, `content_hash`, `validate_pack`)

**What:** `docs/plans/small-model-benchmarking.md` §4 S2's pack-loader portion — the first S2
unit, landing beside a concurrent unit building `modelbench/lmstudio.py` (not touched here).
`modelbench/packs.py`, `tests/test_packs.py` (new), `tests/conftest.py` (`pack_fixture()` +
`PACKS_DIR`), nine fixture packs under `tests/fixtures/packs/`. **648 → 708 tests** (this unit
added 28, all in `test_packs.py`; the remainder of the combined 708 is the concurrent `lmstudio`
unit's, filed separately), `.venv/bin/ruff check .` clean, **13 mutations, all caught**, each
`cp`-aside / mutate / run / `cp`-back restore, diffed byte-identical against the pre-mutation file
after every single one.

**Delivered.** `Pack` (frozen dataclass: `packId`, `packVersion`, `role`, `contentHash: str`
total, `manifest`, `root`) with `data_path`, `load_tool_module` (`importlib.util.spec_from_file_
location`, never `sys.path`) and `ref()`; `load_pack(root)`; `content_hash(root)` (SHA-256 over
sorted, NUL-delimited relative paths and bytes, excluding `PROVENANCE.md` **and** any
`__pycache__` a prior `load_tool_module` call left behind — not named by the plan, added because a
bytecode cache is a loader side effect, never pack content, and would otherwise make identity
depend on whether some earlier process happened to import the pack); `validate_pack(pack) ->
list[str]` (`[]` means valid, `Fingerprint.validate()`'s own shape), covering three independent
axes: the `sampling` contract — structural, by calling `Pack.ref()` (which runs the existing
`check_sampling_contract` rather than re-implementing it, per impl review Pass 1 §4 item 6), the
row-count identity (reading `data.conversations` — the plan's own manifest key, see the correction
below), and `-ml` §3.4 Rule 6's `replicatesPerScript > 1` rejection; `callSurface` derivation from
`environment.requires` (factored into a standalone `derive_call_surface` for direct unit testing),
rejecting a pack declaring neither or both of `lmstudio-chat` / `lmstudio-embeddings`; and an AST
import allowlist (stdlib, via `sys.stdlib_module_names`, plus `modelbench.tooling` by name only —
that module does not exist yet, and the check is a syntactic `ast.parse`/`ast.walk`, never an
import, so it needs no dependency on it). `pack_ref_from_manifest` and `Pack.ref()` now share one
manifest-parsing routine (`_ref_from_manifest_fields`) so the two routes cannot silently diverge on
what a valid `sampling` block is; `pack_ref_from_manifest`'s externally observed messages are
unchanged. The §3.3 totality boundary is asserted directly: `load_pack(...).ref().contentHash` is
not `None` and equals `content_hash(root)`, while `pack_ref_from_manifest(...).contentHash` is
`None` — both halves mutation-tested independently.

**Correction (coordinator finding, same day, before acceptance): the row-count identity was dead
on every plan-conformant manifest.** The first pass keyed the check off `sampling.dataFile`, a key
this module invented — no manifest the plan specifies carries it (the check's own docstring claim
that "the structural route already covers a pack that omits it" was false for that specific key,
since nothing but this module's own code knew it existed), so the route silently returned `[]` on
every real pack shape, including the `row_count_violation` fixture built to exercise it. The
plan's tool-caller manifest literal (`docs/plans/small-model-benchmarking.md` line 435) already
names the key: `"data": {"conversations": "conversations.jsonl", ...}`, beside the matching
`sampling` block. Fixed to read `data.conversations` instead; `sampling.dataFile` is gone from
every fixture. Added `tests/fixtures/packs/undeclared_replication/` and a test for §3.3's own
worked example of why this route exists — `replicatesPerScript: 1` declared, four conversations
per script shipped, "the case that slips past Rule 6's declaration check and past Rule 1 at once"
— which Rule 6 does not catch (declared value is 1, not `> 1`) and the structural route does not
catch (`analysisUnit == pairingKey[0]` holds); only the fixed row-count route does. Mutation:
disabling the row-count call site entirely (`_sampling_problems` returning before
`_row_count_identity_problems`) now fails two tests built on plan-conformant manifests (no invented
key) — `test_validate_pack_rejects_the_row_count_identity_specifically` and
`test_validate_pack_rejects_undeclared_replication_row_count_only` — where under the pre-fix code
that same mutation was survivable, which was the whole finding.

**Not this unit's:** `run` cross-checking a pack's derived `callSurface` against a model's catalog
`type`, and `run` calling `validate_pack`'s AST check and failing closed, are the runner/CLI unit's
(§3.3, §3.4.4a) — `packs.py` only builds the check and makes it callable.
`modelbench/lmstudio.py` / `tests/test_lmstudio.py` (a concurrent S2 unit) and
`modelbench/tooling.py` (not yet built by anyone) were neither read nor depended on.

**Verification:** `.venv/bin/python -m pytest -q` from `model-bench/` → **708 passed, 1
deselected** (the concurrent unit's one `-m live` test), exit 0. `.venv/bin/ruff check .` → `All
checks passed!`.

## 2026-09-09 — the support clamp moves off the envelope's arms, onto the printed interval

**What:** `docs/plans/small-model-benchmarking.md` §4 S1e Table H, implementing `-ml` v1.19 §3.4
Rule 4a — S1's last table. `modelbench/stats.py`, `modelbench/report.py`, `tests/test_stats.py`,
`tests/test_report.py`. **635 → 648 tests** (13 added: 12 in `test_stats.py`, 1 in
`test_report.py`), `.venv/bin/ruff check modelbench tests` clean, **9 mutations, all caught**,
each `cp`-aside / mutate / run / `cp`-back / `diff -q` byte-identical restore.

**Delivered.** A support is a property of the *estimand*, so it is applied once, to the printed
interval, never to a composition's input. `envelope_arms` now widens both arms with `clamp=None`
and returns them unclamped; the private composer `_compose` clamps its own composed result to the
new `SUPPORT_DIFF_PROPORTIONS` constant and returns `(interval, bound_by)`; `bound_by` is now the
three-token `BoundBy` alias (`"MOVER-D"`, `"exact paired bootstrap"`, `"support bound"`), computed
from the composed *unclamped* value against the support on a strict comparison — expressed as
`lo != u_lo` against the already-clamped bound rather than re-spelling the support subscript a
second time, which is what keeps the plan's residuals 2 and 3 at their stated count of one each.
`conservative_envelope` returns `_compose(...)`'s first element; `verdict()` takes both halves of
`_compose`'s return rather than recomputing the attribution inline, closing impl-gate P8-5's
finding as collateral. `report.py`'s `- decided by:` renderer gives a `support bound` token the
support's own boundary value and no `p=` clause: `support bound (-1)`, pinned verbatim by the
note's assertion 10. Confirmed bit-identical against the note's ten assertions and its two
exhaustive sweeps (commutation and verdict-invariance, both at n=12) before writing any test.

**The table's own line pins were stale** (`stats.py` +5, `report.py` +~155, against the table's
`e162ba9` baseline) from three units landed since — verified by content-match, not by line number,
per this coordination's standing discipline. All six residuals reached their stated target
(2→0, 0→1, 0→1, 1→0, 1→0, 2→0); `test_neither_printed_bound_is_ever_tighter_than_either_arm` was
the one shipped test the edit falsifies, and its comparison moved to the support-clamped arms per
the table's own row. `report.py`'s edit landed in the same `_decided_by_line` renderer region as
`c926308`'s continuous-verdict branch but does not touch it — confirmed by reading `c926308`'s
diff, which never reaches `_decided_by_line`.

**Mutation table (9, all caught):** `envelope_arms` reinstating `clamp=(-1.0, 1.0)` on both arms
(3 tests); `_compose`'s strict comparison weakened to `<=`/`>=` (2 tests); the tie-break's own
`<=`/`>=` narrowed to `<`/`>` (1 test, Pass 8's mutation 6 restored); `_compose` dropping its own
clamp on return (2 tests); `conservative_envelope` returning `_compose(...)`'s second element
instead of the first (8 tests); `verdict()` passing `_compose(exact_arm, mover_arm)` — arguments
swapped — instead of `(mover_arm, exact_arm)` (3 tests, the data-plumbing shape the coordination
flagged); `_decided_by_line` dropping its `support bound` branch so it falls into the `p=` clause
(1 test); `_decided_by_line`'s `zip` over `SUPPORT_DIFF_PROPORTIONS` reversed (1 test); `_compose`'s
two support subscripts transposed (12 tests). No equivalent mutants.

**Not delivered, and not owed.** No number in any §3.8 pack moves — the note's headline is that
arms-versus-composed is immaterial to the statistics, verified over 173,472 combinations, and
nothing in this change disturbs that; `bound_by` is computed at report time and was never stored,
so no `migrate` step is owed (`grep -rFn bound_by modelbench/results.py` stayed at 0 throughout).

## 2026-09-09 — the report-side seam: `compare_report` routes a continuous verdict metric

**What:** `docs/plans/small-model-benchmarking.md` §4 S1's `compare_report` block and §3.3 (iv),
against `cf54f5b`. `modelbench/report.py`, `tests/test_report.py`. **628 → 634 tests** (7 added,
1 rewritten test removed — the placeholder it replaced pinned a raise that is no longer reached),
`.venv/bin/ruff check modelbench tests` clean, **10 mutations, all caught**, each `cp`-aside /
mutate / run / `cp`-back / `diff -q` byte-identical restore.

**Delivered — Table F's `report.py:623-789` row, its stated exception now closed.** Pass 1
resolves each pre-registered verdict metric's kind from its own arm aggregate (`_metric_kind`,
falling back to `b`'s aggregate when `a` declares none, and to `"binary"` when neither does — DC-10's
existing cross-check has already reconciled an arm's own aggregate against its own items by this
point, so the aggregate type is a type fact rather than a guess). Three branches follow: a
**homogeneous binary** family is the unchanged two-pass Holm flow; an **all-continuous** family
takes one difference per analysis unit (`_paired_diffs`, the continuous sibling of `_paired_rows`
— joins at the unit id rather than at item `pairingKey`, folding a unit's items into one value by
averaging, so a unit spanning more than one item gets the mean §3.2d asks for and a unit ≡ item
gets the identity) and hands `diffs` and the metric's own `ContinuousMetric`/`DistributionSummary`
`.support` to `stats.continuous_verdict()` — no `holm_steps`, no `mcnemar_exact`, no
`resolving_power` on this path; a **mixed** family is refused *whole* per §3.3 (iv): no member
verdicted, nothing excluded, each member's own block names its resolved kind, the Exploratory
section widens to include the whole family, the headline (if one of the refused members) prints
the same exploratory label instead of the false `_NO_PAIRED_DATA` fallback, and the family-wise
section states the refusal instead of a Holm claim that never happened. The same replacement
mechanism serves the all-continuous `k > 1` case, which states its correction was taken in the
interval instead. `scored_outcome`'s `MetricKindError` raise (Table F's placeholder) is untouched
as a contract but is no longer reached by a well-formed continuous member — the family loop now
resolves kind before choosing an extractor rather than always calling `scored_outcome`.

**Two guards the continuous branch owns that the binary branch's `rp is None` parallel doesn't
quite cover.** Zero paired units renders the existing `_NO_PAIRED_DATA` message (parity with
binary). Exactly **one** paired unit is a case `_NO_PAIRED_DATA` cannot state truthfully — one
unit *is* paired — and `continuous_verdict` refuses a one-unit interval outright (`-ml` §3.4 Rule
8, refusal 4), so a new message (`_ONE_PAIRED_UNIT`) names it rather than letting the `ValueError`
escape uncaught.

**One design call made and not escalated, per the coordination's own precedent for routine
ambiguity:** which arm's aggregate to prefer when both declare one for the same metric (`a`, with
`b` as fallback) and what a metric with no aggregate on either arm resolves to (`"binary"`,
matching the pre-Table-F assumption). Neither is pinned by the plan or the note; both are cheap to
reverse and untested by any fixture that would distinguish them from the alternative.

**Mutation table (10, all caught):** `_metric_kind` forced to always return `"binary"` (7 tests
caught — everything continuous- or mixed-family-shaped); `mixed_kinds` forced `False` (2 mixed
tests — falls through to the binary branch, which raises `MetricKindError` on the continuous
member); `continuous_family` forced `False` (5 continuous tests, same failure mode); `_paired_diffs`
mutated to take a unit's first item instead of the mean (the averaging test only — required
redesigning the test's fixture values first, since the original values happened to make first-item
and mean coincide); the zero-diffs and one-diff guards each disabled in turn (each caught by its
own test, surfacing as an uncaught `ValueError` from `stats.py` instead of a rendered message); the
Exploratory filter's `or mixed_kinds` removed (`IndexError`, section absent); the headline's
`mixed_kinds` branch removed (falls back to the old `_NO_PAIRED_DATA`-vs-`.text` logic and raises
`StopIteration` looking up a verdict that was never computed); both family-wise replacement
branches removed (all three of the mixed/continuous/mixed-headline tests — one via a direct
assertion, two via an `AttributeError` reading `.mcnemar_p` off a `ContinuousVerdict`); the mixed
per-member label forced to always read `"continuous"` (caught by the binary member's assertion);
`_BOOTSTRAP_B` changed from `10_000` (caught once a `B=10000` provenance assertion was added to
the routing test — the constant had no witness before that).

**Whole-diff cross-check:** re-read as one change. Checked and found no issue: `computed`'s
3-tuple shape holds a `None` `HolmStep` for every mixed/continuous entry, but the only loop that
reads the third field is the Holm table, itself gated to the branch that never appends such an
entry; the widened Exploratory filter (`m.name not in family or mixed_kinds`) only ever *adds*
family members for a genuinely refused family, never suppresses an already-true case; the
`tables`/`p_values`/`steps` locals are now scoped inside the binary `else:` branch with no use
outside it. Nothing else found.

**Line-pin drift, reported and not fixed** (routes to `architect`). The insertions before
`compare_report` (`_BOOTSTRAP_B`, `_metric_aggregate`/`_metric_kind`, `PairedDiffs`/
`_paired_diffs`, four message constants — about 155 lines) push every citation below them down by
that much, and the family loop itself grew from 167 to about 280 lines. Table F's own site table
(`docs/plans/small-model-benchmarking.md:3845-3847`) cites three now-stale locations against the
pre-this-unit tree: `report.py:211` (DC-10's kind-cross-check selector) is now `:342`;
`report.py:581-601` (the Arms table's `else`-split) is now approximately `:746-786`; and
`report.py:623-789` (`compare_report`'s Table F row itself — "the family loop and the two
renderers downstream of it") is now approximately `:810-1088`, `compare_report`'s def itself now
at `:664`. §3.3 (iv)'s own citations (`:719`, `:738`, `:751`, `:763-777`, etc.) are pinned to a
named historical commit (`5878014`) rather than the live tree, per that section's own v1.23
discipline, and do not drift.

**Amendment (U67) — a mutation-testing gap found during integration review, not by the suite.**
The delivered `stats.continuous_verdict(family=family, ...)` call at `report.py:901` survives
being mutated to `family=[metric]`: all 634 tests above still pass. That argument is not
cosmetic — `continuous_verdict()` derives its quantile levels from `k = len(family)`
(`_family_ci_levels`), so collapsing it to the metric's own singleton family sets `k = 1` and
silently skips the `k > 1` Bonferroni-in-the-interval correction §3.3 (iv) commits to: a `k = 3`
family would render at `k = 1` levels, publishing an interval too narrow and a verdict too
confident, with no visible symptom.
`tests/test_report.py::test_an_all_continuous_family_takes_its_correction_in_the_interval_not_a_ladder`
names that correction in its docstring, but its four assertions only check the explanatory
sentence's presence, the absence of a Holm claim, that both members print the bootstrap-decided
text, and that no arm is excluded — none of which observes interval width, so all four hold
unchanged under the mutation. Its docstring is corrected to say only what it actually proves (no
Holm ladder, the explanatory section, both members verdicted, no exclusion) and to point at the
new test for the correction itself. Closed by
`tests/test_report.py::test_continuous_verdict_receives_the_whole_family_not_just_the_metric`:
renders identical `mrr` data and seed once as a `k = 1` family and once as a `k = 2` family
alongside a second all-continuous metric, and asserts the `k = 2` interval comes out strictly
wider — same seed and identical per-unit differences mean the two runs bootstrap-resample
identically, so only the quantile levels can account for a width difference; `family=[metric]`
renders both at the `k = 1` levels and the assertion fails with a message naming the two
intervals and widths. Mutation-confirmed: `report.py:901`'s `family=family,` changed to
`family=[metric],` now fails exactly the new test (`1 failed, 634 passed`); `cp`-aside / mutate /
run / `cp`-back / `diff -q` byte-identical restore. **634 → 635 tests**,
`.venv/bin/ruff check modelbench tests` clean. No production code changed by this amendment —
`modelbench/report.py` and `modelbench/stats.py` are exactly as this entry originally left them.

## 2026-09-09 — `-ml` §3.4 Rule 8: `continuous_verdict()`, the continuous producer

**What:** `docs/plans/small-model-benchmarking-ml.md` §3.4 Rule 8 (v1.16-v1.19), against `2d23482`.
`modelbench/stats.py`, `tests/test_stats.py`. **600 → 628 tests**, `.venv/bin/ruff check modelbench
tests` clean, **14 mutations** (2 caught real gaps, both closed in this entry — see below), each
`cp`-aside / mutate / run / `cp`-back / `diff -q` byte-identical.

**Delivered:** `ContinuousVerdict` (13 fields — a sibling type to `Verdict`, never one with six
fields left `None`) and `continuous_verdict()`, the entry point for every continuous metric
(MRR, score separation). Two new module-private helpers it composes: `_family_ci_levels(alpha,
k)`, the exact-rational quantile pair `alpha/(2k)`, `1 - alpha/(2k)` recovered as
`Fraction(str(alpha_family))` (never `Fraction(alpha_family)` — the double's own binary value);
and `_support_clamp(support)`, the clamp of the *difference's* support `(lo - hi, hi - lo)`,
`None` for an unbounded metric. `continuous_verdict()` takes no `resolving`, `alpha_step`,
percentile levels or `clamp` — all four are derived internally or unrepresentable at the call
site — and refuses on five conditions: an empty or non-finite `diffs` and `design_effect < 1.0`
are inherited unchanged from `paired_bootstrap`/`paired_cluster_bootstrap` rather than
re-checked; a metric not in its own `family` and a degenerate `support` (`lo >= hi`) are this
function's own. It calls `paired_cluster_bootstrap` — Rule 8's stated entry point — with the
derived levels and clamp, so the pack's declared `design_effect` and the metric's `support` both
reach the one interval that is printed. `diff`, both CI bounds and the half-width print at three
decimal places (never "pp" — this path has no percentage-point convention to inherit from
`verdict()`), and `alpha_used` is `alpha_family / k`, the two-sided alpha the printed interval
was actually taken at.

**State recovered, not started over.** `tests/test_stats.py`'s 348-line red phase (22 test
definitions, 26 test instances with parametrization) predated this unit, landed by an agent
killed twice by platform failures; this unit added the green phase plus one test the mutation
pass found the red phase had missed (below), and one unrelated one-line fix (`import re`, used
by two of the inherited tests but never added — a `NameError` waiting under the first `re.search`
call, not an assertion).

**A gap the red phase's own mutation table did not close, found and closed in this unit.**
`test_continuous_verdict_mrr_worked_case_from_the_note` (`diffs = [1.0] * 10`) survived a mutation
that dropped the derived clamp entirely (`clamp=None` unconditionally): with every difference
identical, the bootstrap interval is a zero-width point at 1.0, and widening a zero half-width by
any `sqrt(DEFF)` is still zero — so the clamp never has anything to clamp in that construction,
and `v.ci == (1.0, 1.0)` holds with or without it. The test's own docstring claim ("the unclamped
widened interval would run off `[-1, 1]`") was false of the construction it describes. Not a wrong
assertion — it passes, correctly — just not evidence the clamp ran. Added
`test_continuous_verdict_clamp_actually_binds_when_diffs_have_variance` (diffs with real spread,
`design_effect=9.0`, where the unclamped widened upper bound is measured at `1.08` against a
clamped `1.0`) as a second, load-bearing witness; confirmed it kills the mutation the worked case
did not. **`teco` reviewed and accepted this finding, then asked for the false claim itself to be
corrected** — a green test with a false docstring is how the next reader concludes the clamp is
covered when it is not. `test_continuous_verdict_mrr_worked_case_from_the_note`'s docstring is
rewritten to say only what the construction proves (the worked case runs and returns the note's
published `(1.0, 1.0)`) and points at the clamp-binding test above for actual coverage; its
assertion is untouched.

**Mutations, and what each targeted:** the metric-not-in-family refusal · the degenerate-support
refusal (both `_support_clamp` and `continuous_verdict`'s own path to it) · the single-analysis-
unit refusal, plus a variant merging it with the empty-`diffs` check (`len(diffs) < 2` instead of
`== 1`) to confirm the two refusals' messages would otherwise collide — the empty case must reach
`paired_bootstrap`'s own "at least one difference" text, not the single-unit refusal's — the
inherited empty-`diffs`, non-finite-`diffs` and `design_effect < 1.0` refusals (each disabled at
its source in `paired_bootstrap`/`paired_cluster_bootstrap`) · `_family_ci_levels`'s
`Fraction(str(...))` recovery and its `2 * k` factor · `_support_clamp`'s difference conversion ·
`alpha_used`'s `/ k` (against `/ (2 * k)`) · the family size `k` itself (off-by-one) · the clamp
call-through (the gap above) · `distinguishable`'s strict `>`/`<` (against `>=`/`<=`).

**The strict-comparison mutation initially survived** — both existing worked tests place the CI
bounds strictly away from zero, so the mutation went unwitnessed at the boundary — **and this
entry first reported it as a deferred, low-priority gap on the (wrong) assumption that an exact
zero bound needed Monte-Carlo luck to construct.** `teco` verified it against the committed code
and pushed back: an **all-zero** `diffs` sample makes every bootstrap percentile exactly `0.0`
deterministically (the resample distribution is a single atom at zero, for any `B`/`seed`), which
is neither Monte-Carlo-dependent nor blocked on unbuilt work — so the deferral was not earned.
Added `test_continuous_verdict_is_not_distinguishable_at_an_exact_zero_boundary`
(`diffs=[0.0]*8`, `ci == (0.0, 0.0)`, `distinguishable is False`); confirmed it now kills the
`>=`/`<=` mutation and that the mutation survives the rest of the 627-test suite (628th being the
new test itself) — i.e. this test is the sole witness. `verdict()`, the pre-existing binary-path
producer, carries the identical strict `ci[0] > 0 or ci[1] < 0` convention with the identical
absence of a boundary test; that predates this unit and is out of its scope, and is recorded as a
follow-up for whichever unit next touches `verdict()` rather than fixed here.

**Whole-diff cross-check, done before declaring done: nothing found beyond the one gap above.**
Walked `stats.py` and `test_stats.py` together as one change — the ordering of
`continuous_verdict`'s own checks (family membership, then `_support_clamp`, then the
single-unit refusal, then the call into `paired_cluster_bootstrap`) against every refusal test's
combination of valid/invalid arguments (no two refusal conditions are ever true in the same test,
so ordering never changes which message a test sees); the module docstring's rule count against
the new `continuous_verdict()` addition (updated "seven" to "eight", added item 8); `_plural`,
`Basis` and `Literal` reuse against their existing single definitions (no duplicate homes).

**Line-pin drift — reported, not fixed (routes to `architect`).** The module docstring's 5-line
Rule 8 addition sits above every function in the file, so every `stats.py:<line>` citation in
`docs/plans/small-model-benchmarking-ml.md` shifts by **+5**: `paired_bootstrap` `149` → `154`,
`paired_cluster_bootstrap` `202` → `207`, `_widen` `246` → `251`, and §11.2.2's `stats.py:159` →
`164`. New code was appended at the file's end, so nothing else moves.

## 2026-09-08 — §4 S1e Table F: the continuous carrier lands, scoped to its own proof surface

**What:** `docs/plans/small-model-benchmarking.md` §4 S1e **Table F** (v1.23, plan-gate P6-1), the
last of the eight S1e tables, against `e162ba9`. `modelbench/results.py`, `modelbench/report.py`,
`tests/test_results.py`, `tests/test_report.py`. **577 → 600 tests**, `.venv/bin/ruff check .`
clean, **6 mutations**, each `cp`-aside / mutate / run / `cp`-back / `diff -q` byte-identical.

**Delivered — the carrier and its two report-side readers, all 11 site rows** except one row's
continuous-verdict branch (below): `ItemResult.measures: Mapping[str, float]` beside `counts`, with
`__post_init__` refusing a metric name present in both maps (`MetricKindError`) and a non-finite
measure (`NonFiniteMeasure`); `scored_value(metric) -> float | None`, `scored_outcome`'s sibling
over `measures` with the same three states; `scored_outcome` itself now **raising**
`MetricKindError` on a `measures`-resident metric instead of booleanising it; `ContinuousMetric`
gains `support: tuple[float, float] | None`, required with no default; the new
`DistributionSummary(name, median, p10, n, unit, support)`, frozen; `RetrievalAggregates`'s
`separationRaw`/`separationZ` retype `float | None` → `DistributionSummary | None` and
`named_metrics()` returns both — `sep_z` reaches a table at all for the first time; the metric
(de)serialisers gain a **module-level tag→decoder mapping** (`_METRIC_DECODERS`) that
`_metric_from_dict` dispatches on and `_decode` gates on, so an unrecognised `"type"` **raises**
where it used to fall through as a raw `dict`; `_index_row`'s metrics cell renders a
`DistributionSummary` as `{name}=p50 {median:.4f}` rather than reading `.mean` (`AttributeError`
before this table). On the `report.py` side: DC-10's selector (`_aggregate_item_mismatches`)
widens past `isinstance(metric, BinaryMetric)` to a third arithmetic over `scored_value` for a
continuous member, with **no unit filter** (neither `ContinuousMetric` nor `DistributionSummary`
carries a denominator noun) — the same check is where a kind disagreement between an arm's
aggregate and its own items surfaces, in either direction, and both `IncompleteItemRecord` and
`MetricKindError` are caught there so neither escapes as a traceback; and the Arms table's bare
`else` splits so a `DistributionSummary` renders its median and p10, never `.mean`.

**Deliberately not delivered, per an explicit scope decision (Option A) put to the stakeholder and
confirmed before implementation.** Table F's own two-file scope statement ("the continuous carrier:
`modelbench/results.py` and `modelbench/report.py`") and its 8 enumerating commands / 3 residuals
never touch `modelbench/stats.py` — but the `report.py:623-789` row's own text ("pass 1 resolves
each member's kind and branches, per §4 S1's `compare_report` block") cites a spec that requires
calling `-ml` §3.4 Rule 8's `continuous_verdict()` and rendering its `ContinuousVerdict` sibling
type, neither of which exists anywhere in the tree (confirmed: only five docstring/comment
citations in `stats.py`/`test_stats.py`, no implementation, no test). Building it means authoring a
new statistical producer from a separate ~3,600-line note this unit was not scoped to implement, and
DC-13(e) / §3.3(iv)'s mixed-kind-family refusal depends on it too. **That gap is not silently
absorbed:** the still-binary family loop's `_paired_rows` calls `scored_outcome` on every family
member unconditionally, and the moment a pack declares a continuous `verdictMetrics` member,
`scored_outcome`'s new raise fires immediately and uncaught — converting "silently wrong when a
pack finally arrives" into "refuses loudly right now", which is the property this table exists to
guarantee ahead of that pack existing. `tests/test_report.py::test_a_continuous_verdict_member_refuses_loudly_rather_than_booleanising`
pins exactly this and doubles as the seam description for the follow-up unit: replace that raise
with a resolved-kind branch that calls `continuous_verdict()` instead of `_paired_rows` for a
continuous member. **A separate, properly-sized unit builds `continuous_verdict()`/`ContinuousVerdict`
from `-ml` §3.4 Rule 8, the family-loop continuous branch, and §3.3(iv)'s mixed-kind refusal, landing
before S1 closes.**

**Residuals — all three, before → after, re-run at `e162ba9` and again after the edit:**

| # | Command | Before | After |
|---|---|---|---|
| 1 | `grep -nF 'separationRaw: float \| None' modelbench/results.py` | 1 | 0 |
| 2 | `grep -nF 'separationZ: float \| None' modelbench/results.py` | 1 | 0 |
| 3 | `grep -rFn '{"binary", "continuous"}' modelbench --include='*.py'` | 1 | 0 |

**Whole-diff cross-check, done before declaring done: nothing found.** Walked every pair of edits
across both files for interaction — `_decode`'s widened `"type" in value` gate against every
`Aggregates` field's possible encoded shape (no field anywhere holds a bare dict with an unrelated
`"type"` key; `TurnPositionRate`'s `{"turnIndex", "metric"}` shape carries no `"type"` at its own
level, so the two branches stay mutually exclusive); `__post_init__`'s new refusal against all 14
existing `ItemResult(` fixture sites (none pass `measures`, so the overlap check is vacuously
satisfied everywhere unchanged — confirmed by the unchanged 577 continuing to pass unmodified); DC-10's
widened `continuous` selector against `_arm_label`/pooled-count rendering (untouched, since neither
reads the new flag); and `scored_outcome`'s ordering (`measures` check before the `counts` check) is
safe only because `__post_init__` already guarantees the two maps are disjoint — checked explicitly
rather than assumed. No self-created defect found.

**Line-pin drift — reported, not fixed; re-pinning is the coordinator's to route.** My insertions
in `results.py` and `report.py` are additive-only (no deletions), so every pin below moved by a
constant positive offset within its file, confirmed by locating each pinned line's exact text
rather than by arithmetic:

| Owner (unaffected by this unit) | Pin (old, `e162ba9`) | New | What's there |
|---|---|---|---|
| Appendix A | `results.py:466` | `results.py:628` | `FieldProblem(field=f"items[{item.itemId}].counts.{metric}", …)` |
| Table F's own prose (self-citation, now inside a landed table's body) | `results.py:327` | `results.py:427` | "A `KeyError` here surfaces as `unparseable`…" comment |
| Table F's own prose (self-citation) | `results.py:365-366` | `results.py:487-488` | the `BinaryMetric.unit` no-`.get`-fallback comment, cited by analogy for `support` |
| Table B (**landed**, `8fc2341`) | `test_results.py:62`, `:71`, `:239` | `:69`, `:78`, `:246` | the `Fingerprint(…)`/`_run(…)` construction sites the row names |
| Table B (**landed**, `8fc2341`) | `test_report.py:471`, `:486` | `:479`, `:494` | the two `arm_kind="deterministic",` fixture sites the row names |

No pin inside a table still describing an **instruction** (an un-landed row) was found stale by
this edit — Table H's `report.py:338`/`:927`/`:1199` sites are in `stats.py`/`report.py`'s
`_decided_by` region, which this table's `report.py` edits (`:211`, Arms table) do not overlap, per
Table F's own "meets Table H on `report.py`, neither order constrained" note.

**Mutation table** — each mutation `cp`-aside first, then a targeted disable (`if False and …` or a
silent-fallthrough rewrite), the pinned test(s) re-run alone, then restored by `cp` from the
untouched copy and `diff -q` confirmed byte-identical before the next mutation:

| # | Mutation | Result |
|---|---|---|
| M1 | `scored_outcome`'s `measures` check disabled | killed both `test_scored_outcome_raises_on_a_measures_resident_metric` and `test_a_continuous_verdict_member_refuses_loudly_rather_than_booleanising` |
| M2 | `__post_init__`'s both-maps overlap check disabled | killed `test_a_metric_name_present_in_both_maps_is_refused_at_construction` |
| M3 | `__post_init__`'s `math.isfinite` check disabled | killed `test_a_non_finite_measure_is_refused_at_construction` |
| M4 | `_metric_from_dict`'s unrecognised-tag raise replaced with a silent `return d` | killed `test_an_unrecognised_metric_type_tag_raises_rather_than_returning_a_raw_dict` and `test_a_record_with_an_unrecognised_metric_type_is_quarantined_as_unparseable` |
| M5 | DC-10's `continuous` selector forced to `False` | killed `test_a_continuous_member_declaring_a_measure_it_does_not_carry_is_a_mismatch` and `test_a_kind_disagreement_continuous_aggregate_binary_items_is_the_same_mismatch_class`; correctly left `test_a_continuous_verdict_member_refuses_loudly_rather_than_booleanising` passing (a different site) |
| M6 | Arms table's `DistributionSummary` branch disabled | killed `test_arms_table_renders_a_distribution_summary_without_reading_mean` (reproduces the exact `AttributeError` the site row describes) |

**Verification:** `.venv/bin/pytest -q` from `model-bench/` → **600 passed** (577 baseline + 23 new:
18 in `test_results.py`, 5 in `test_report.py`), 0 failed / 0 skipped / 0 deselected.
`.venv/bin/ruff check modelbench tests` → `All checks passed!`.

**What:** the last pass of `docs/reviews/small-model-benchmarking-impl.md` (gated as `## Pass 11`,
being renumbered to `## Pass 10`) against `93b0e42` — **N4** (major) and **N5** (minor), both in
scope and both closed. `modelbench/stats.py`, `tests/test_stats.py`. **560 → 577 tests**,
`.venv/bin/ruff check .` clean, **6 mutations plus one forward-compatibility probe**, run one at a
time. P8-1 remains held, blocked on `-ml` §3.4 Rule 4a's own unit.

**N4 (major) — `+inf`, or a non-finite difference, passed every guard and printed as a real
interval.** Rule 4's four precondition-4 guards close the *below* side of `>= 1.0` by construction.
Nothing closed the *above* side or the data. `not inf >= 1.0` is `False`, so `+inf` satisfies the
rule the note states; `sqrt(inf)` is `inf`; `_widen` returns `(-inf, +inf)`; and the clamp returns
the support bounds. All reproduced against the untouched tree before any fix:
`envelope_arms((4,5,3,0), design_effect=inf)` → `((-1.0, 1.0), (-1.0, 1.0))`;
`verdict(...)` at `deff=inf` → **`ci = (-1.0, 1.0)`, `bound_by = ('MOVER-D', 'MOVER-D')`**, a
full-support interval attributed to a named instrument; and one `nan` among `diffs` →
`paired_cluster_bootstrap(...)` → `(-1.0, 1.0)`, on a surface with **no guard on its data at all**.

**The mechanism is the clamp, not the predicate.** Every comparison with a NaN is `False`, so
`max(-1.0, nan)` is `-1.0` and `min(1.0, nan)` is `1.0`: the clamp converts *no number* into *the
widest honest number*, silently and in the direction that prints. Unclamped, the same input returns
`(nan, nan)` — visibly wrong. That contrast is what identifies the launderer, and both clamp
settings are swept in the test for exactly that reason.

**Fixed with a result guard in `_widen`, and this was the round's design call.** The alternative was
an input guard: extend the four precondition-4 predicates to reject `+inf`. Rejected, for two
reasons. **(1) Completeness.** The harm is one transformation — a non-number becoming a plausible
bound — and it happens at one place. A guard there closes every upstream cause, including ones
nobody has enumerated; an input guard closes the causes we happened to think of, and the whole
finding is that the previous round enumerated three sites and missed the fourth path entirely.
The module's stated shape is *"written so the anti-conservative version does not typecheck"*, and
the bad **state** is a laundered bound, not a particular bad input. **(2) Cost.** Extending four
predicates means also rewriting four messages — each says "must be >= 1.0", which is false for
`inf` — at four sites, for strictly less coverage: it would not catch a non-finite *difference* at
all. The guard is on the **result** rather than on `_widen`'s inputs for the same completeness
reason; it also catches overflow, which no input check does.

**But one guard was not enough, and this is where the fix goes past the reviewed candidate.**
`paired_bootstrap` does **not** go through `_widen` — it is public, is `-ml` §3.2d's own quantile
surface, and returns before any widening — so the result guard cannot protect it. Measured, it was
the worst-looking of them: `paired_bootstrap([nan, 1.0, 0.0, -1.0], ...)` returned **`(-0.6, nan)`**,
not a pair of `nan`s but a plausible lower bound beside a `nan` upper, which is the shape most
likely to be read as a rendering glitch over a real interval. The data precondition therefore sits
on the function that reads the data, where it names itself. Two guards, each owning the precondition
it actually has.

**It survives Rule 4a — verified, not argued.** Under Rule 4a `envelope_arms` widens with
`clamp=None` and the clamp moves into the composer. A guard placed *on the clamp* would move with it
or vanish; this one is on `_widen`'s **result, before the clamp branch**, so it fires on the
unclamped path too. Simulated the edit — both `clamp=(-1.0, 1.0)` in `envelope_arms` set to
`clamp=None` — and re-ran the N4 tests: **17 passed**, guard still firing, source restored
byte-identical. Guarding before the branch is also why the unclamped `sep_z` path is covered, which
it was not before.

**N5 (minor) — the rejection domain was written out three times.** `[0.5, 0.25, 0.999999, 0.0,
-1.0, nan]` appeared at three `parametrize` marks; a seventh failure class meant three edits and
missing one is silent — the sweep still passes, one surface just stops being swept. Now one
module constant, `_SUB_ONE_DESIGN_EFFECTS`, whose docstring records *why* each of the six is there
(they partition the rejection domain by predicate-failure mode) and why `-inf` is deliberately
absent (`not -inf >= 1.0` is `True`, so it is behaviourally the `-1.0` row). Plan §3.9's rule on the
test side, third application this arc after `_percentile` and `_compose`. Verified by collecting:
the three sweeps now produce identical id lists.

**The accepting side is under test now.** Both existing sweeps looked only at the rejection side,
which is precisely where N4 hid — no sweep of *refused* values could ever have reached a value that
is **accepted**. `test_no_design_effect_ever_yields_a_bound_that_is_not_a_number` asserts the
invariant across both sides over a mixed domain: for any design effect, a surface either refuses it
or returns bounds that are numbers. Nothing in between — and "in between" is exactly where the
laundered `(-1.0, 1.0)` sat, accepted and not a number.

**Line pins — these moved, and there was no line-count-neutral option.** A new refusal is new lines.
Both guards sit **before** every pinned line, so all ten shift by the same **+31** (31 insertions,
0 deletions; `stats.py` 1418 → 1449 lines):

| Table | Pin (old) | New | Table | Pin (old) | New |
|---|---|---|---|---|---|
| E | `:261` | `:292` | — | `:896` | `:927` |
| E | `:382` | `:413` | H | `:1168` | `:1199` |
| E | `:387` | `:418` | H | `:1184-1187` | `:1215-1218` |
| G | `:413` | `:444` | | | |

Verified by locating each pinned line's exact text in the new file, not by adding 31 by hand; the
`:1184-1187` block was confirmed intact as a unit. **Re-pinning is the coordinator's to route.**

**Mutation table** — each `cp` aside, mutated, run alone, `cp` back, `diff -q` byte-identical before
the next. Both guards were verified test-first: 13 of the 17 new ids fail on the shipped source
before `stats.py` is touched.

| # | Mutation | Result |
|---|---|---|
| N4-M1 | `_widen`'s finiteness guard deleted | killed |
| N4-M2 | that guard moved **after** the clamp | killed — placement is load-bearing, not just presence |
| N4-M3 | `paired_bootstrap`'s data guard deleted | killed — the path `_widen` cannot see |
| N4-M4 | `isfinite` weakened to a NaN-only test (`b == b`) | killed — the `inf` half survives a half-fix |
| N5-M1 | one row dropped from `_SUB_ONE_DESIGN_EFFECTS` | survives by design — collection falls 28 → 24, i.e. **one edit reached all three sweeps**, which is the property N5 asked for and which no assertion can carry |
| N-M6 | **control** — P8-1's tie-break `<=,>=` → `<,>` | **SURVIVES, as intended** (577 passed) |
| probe | Rule 4a simulated (`clamp=None` in `envelope_arms`) | 17 N4 ids still pass — the guard survives the in-flight edit |

N4-M2 is the one worth keeping: it proves the guard's **placement** carries the fix. Moved one line
later, after the clamp, it inspects the laundered value and is blind — green on the defect it exists
to catch, which is the exact failure mode this arc has been finding all week.

## 2026-09-08 — Implementation-review Pass 9 fix round: the same respelling at the two remaining sites

**What:** the `## Pass 9` findings of `docs/reviews/small-model-benchmarking-impl.md` against
`7f865e2`, all three in scope — **N1**, **N2** (majors) and **N3** (nit). `modelbench/stats.py`,
`tests/test_stats.py`, `AGENTS.md`. **550 → 560 tests**, `.venv/bin/ruff check .` clean,
**5 mutations run one at a time**, 4 killed and 1 surviving **by design** (P8-1's control again).
P8-1 remains held, blocked on `-ml` v1.19 Rule 4a's separate unit.

**One sentence for both majors: the Pass 8 round diagnosed the predicate correctly and respelled it
at one of the three sites that carried it.** `resolving_power` was already NaN-safe before that
round — which is why the "two spellings now coexist" worry it was briefed against turned out
backwards, the gate having checked `cc28d48` and found the safe spelling already there, comment and
all. `envelope_arms` joined it. `verdict()` and `paired_cluster_bootstrap` did not, and both were
live NaN holes.

**N1 (major) — `verdict()`'s precondition 4 was NaN-blind, so the layer-ordering property the
previous round restored was false at exactly one value.** `stats.py:1126` spelled
`resolving.design_effect < 1.0`, which is `False` for a NaN, so the value fell through and
`envelope_arms` raised **its** message one layer down. Reproduced before fixing: at `deff=0.5`
`verdict()` raises `verdict() precondition 4: …`, at `deff=nan` it raises
`design_effect must be >= 1.0 (-ml §3.4 Rule 4, precondition 4)` — the inner layer's sentence. That
is precisely the trap review P3-11 installed this check to prevent, alive again at one input.
`test_the_two_envelope_refusals_name_which_layer_raised` could not see it because it shipped with
`deff: float = 0.5` as a **default argument** rather than a parametrization — twelve lines below a
sibling already swept over `nan` for this exact reason. Now parametrized over the same six values;
the `nan` row fails on the old predicate and passes on the new one.

**N2 (major) — `paired_cluster_bootstrap` returned the full support as an interval for a NaN design
effect.** `stats.py:228`, same `< 1.0`. Reproduced verbatim before fixing:
`paired_cluster_bootstrap([1.0, 0.0, -1.0, 1.0], design_effect=nan, B=50, seed=1,
clamp=(-1.0, 1.0), levels=(LEVEL_CI95_LO, LEVEL_CI95_HI))` returned **`(-1.0, 1.0)`** — `sqrt(nan)`
widens both bounds to `nan`, and the clamp's `max(-1.0, nan)`/`min(1.0, nan)` return the clamp's own
endpoints, so a number nobody supplied prints as a maximally wide real interval. This is the same
symptom that justified the previous round's deviation, and it needs **no `dataclasses.replace`
bypass to reach**: `design_effect` is a bare float parameter with no `resolving_power` in the path,
on `-ml` §3.2d's **continuous** entry point that Rule 8's `continuous_verdict()` is specified to
call with a value arriving from a pack manifest. `test_paired_cluster_bootstrap_refuses_a_design_
effect_below_one` gains the `parametrize` its envelope sibling twenty lines away already carried,
plus a `match="precondition 4"` it lacked.

**N3 (nit) — `AGENTS.md` described the NaN-safe guard using the NaN-unsafe spelling.** In the
always-loaded file, in the round whose whole finding is that `< 1.0` is the wrong predicate. Fixed
wider than the three words asked for, because N1 and N2 make the wider version the live constraint:
the bullet now names **all four sites** that spell it `not … >= 1.0`, says why (a NaN widens to
`nan`, which the clamp turns into full support), and says plainly *do not simplify any of them*.

**Line pins — none moved.** Plan §4 S1e Tables E, G and H pin `stats.py:261`, `:382`, `:387`,
`:413`, `:896`, `:1168` and `:1184-1187`. Both source edits were made **in place**, one line for
one line (`git diff --numstat` reads `2 2`), the file is 1418 lines before and after, and all ten
pinned lines are byte-identical — verified by diffing the extracted lines, not by inspection. **No
re-pinning is required.** The NaN rationale went into an end-of-line comment at each site rather
than a comment block precisely to keep the edits line-count-neutral; both lines are 93 and 95
characters against the project's `line-length = 100`.

**Attributed delta, +10**, entirely from turning two single-value tests into sweeps:
`test_the_two_envelope_refusals_name_which_layer_raised` 1 → 6 ids and
`test_paired_cluster_bootstrap_refuses_a_design_effect_below_one` 1 → 6 ids. No test was retired.

**Mutation table** — each `cp` aside, mutated, run alone, `cp` back, `diff -q` byte-identical before
the next. Both majors were verified test-first: the new test fails on the shipped predicate before
`stats.py` is touched.

| # | Mutation | Result |
|---|---|---|
| N1-M1 | `verdict()`'s guard reverted to the NaN-blind `< 1.0` | killed — `…which_layer_raised[nan]` |
| N1-M2 | `verdict()`'s message stops naming itself (layers indistinguishable) | killed — `…which_layer_raised[0.5]` |
| N2-M1 | `paired_cluster_bootstrap`'s guard reverted to `< 1.0` | killed — `…below_one[nan]` |
| N2-M2 | that guard deleted outright | killed — `…below_one[0.5]` |
| N-M5 | **control** — P8-1's tie-break `<=,>=` → `<,>` | **SURVIVES, as intended** (560 passed) |

N1-M1 and N2-M1 are the review's own acceptance test — break the guard back and confirm the new
assertion fires — and each is caught by the `nan` row alone, which is the evidence that the sweep
rather than a second example was the necessary shape. N1-M2 pins the property that makes the
ordering checkable at all: the two layers' messages must stay distinguishable.

## 2026-09-08 — Implementation-review Pass 8 fix round: two restored refusals, one composition, five new assertions

**What:** the `## Pass 8` findings of `docs/reviews/small-model-benchmarking-impl.md` against
`cc28d48`, scope **P8-2 … P8-7** — **P8-1 is excluded and untouched**, being adjudicated by a
`data-scientist` as a methodology question (whether the `(-1, 1)` support clamp belongs on the
envelope's *arms* or on the *composed* interval — ruled mid-round by `-ml` v1.19's new §3.4
Rule 4a, `a707d09`, and implemented in a separate unit). `modelbench/stats.py`,
`tests/{test_stats,test_report}.py`, `AGENTS.md`; `report.py` is unchanged — P8-2's loss was an
absent assertion, not a wrong renderer. **519 → 550 tests**, `.venv/bin/ruff check .` clean,
**14 mutations run one at a time**, 13 killed and 1 surviving **by design** (the control below).

**P8-2 (major) — the `mcnemar-exact` bullet was asserted nowhere.** `_decided_by_line` has exactly
three renderings over its domain and only two were pinned; mutating the `bound_by is None` branch
to a constant string left the whole suite green. The positive assertion had lived in
`test_the_seed_is_not_printed_where_no_bootstrap_decided_anything` and moved to neither of the two
tests that replaced it — the one genuine coverage loss in that round's seven-test churn, and on
the branch **every** `by-construction` comparison at DEFF 1.00 takes. Closed by the third
rendering's own named test rather than by a line appended to an unrelated one, plus the invariant
nothing held: `(bound_by is None) == (decided_by == "mcnemar-exact")`, asserted on both constructed
verdicts. `bound_by` is a discriminated union over `decided_by`, not a field that is `None` until
it is not, and `_decided_by_line` transcribes that discriminator a second time by branching on the
wrong one of the two — so they could only ever disagree silently.

**P8-3 (major) — `envelope_arms`/`conservative_envelope` had stopped refusing `design_effect < 1.0`.**
The refusal shipped at `c19f875` inside `paired_cluster_bootstrap`, the call v1.11's closed form
deleted, so it **retired by accident**: Table D retires only the `n != len(diffs)` guard. What got
through was not an exception but a narrower interval — `(34, 6, 0, 0)` at DEFF 0.5 returned
`[6.6, 25.0] pp` against `[3.2, 29.1]` at 1.00, anti-conservative in the direction that prints.
`verdict()` still guarded its own entry, so nothing that ran was wrong; both functions are public
and `envelope_arms` is new, so this was a defence-in-depth loss on a surface the plan exposes.
Restored with `paired_cluster_bootstrap`'s exact message and the **NaN-safe** predicate
(`not design_effect >= 1.0`) that `resolving_power` already documents: `< 1.0` is False for a NaN,
and `sqrt(nan)` then clamps both arms to full `(-1, 1)` support — a maximally wide interval
conjured out of a missing number. **The ordering half of
`test_verdict_refuses_a_design_effect_below_one` comes back with it**: a second raise on the same
precondition exists again, so that test's `match` is re-anchored on `verdict()`'s own prefix rather
than on the bare `precondition 4` both layers now share — mutation M13 confirms deleting
`verdict()`'s check is caught rather than masked by the layer below.

**P8-4 (minor) — the atom-boundary fixture.** The `>=` → `>` mutant on the exact quantile's
selector was reported as equivalent; it is not, only unreachable at `LEVEL_CI95_LO`/`LEVEL_CI95_HI`
(no tie exists there — swept). `(0, 1, 1, 0)` has CDF exactly `1/4, 3/4, 1` over `s/n ∈ {-1, 0, 1}`,
so a level can land **on** an atom boundary — the one place `inf{ v : F(v) >= p }` and
`inf{ v : F(v) > p }` differ, and the property the docstring's "the two agree by construction"
claim rests on. Parametrized on the boundary, below it, above it, and at `Fraction(1)`, which under
`>` raises `IndexError` instead of returning the largest atom.

**P8-5 (minor) — one composition, not two.** `conservative_envelope` and `verdict()` each spelled
`min(exact[0], mover[0]), max(exact[1], mover[1])` out for itself, while `envelope_arms`' docstring
claimed "neither recomputes the other's arithmetic". Both copies were independently pinned, so this
was never a live bug — it was plan §3.9's rule, the one that retired the two private percentiles,
violated at four lines instead of forty. Extracted to `_compose(mover, exact)`, called from both;
the docstring sentence is now true. The arithmetic is **unchanged** and the arms are still composed
bound by bound at the same place.

`-ml` v1.19's Rule 4a (`a707d09`) landed mid-round and closes P8-5 as collateral by the same
reasoning, so `_compose` is the seam its clamp pass builds on rather than a second composer to
reconcile. What that pass still has to add to it, and what this round deliberately did **not** do
(P8-1 being out of scope): clamp `_compose`'s **result** to `(-1.0, 1.0)` once `envelope_arms`
widens with `clamp=None`, widen its return to `(interval, bound_by)`, and move `verdict()`'s inline
`bound_by` computation inside it as Rule 4a's three-token closed set. `conservative_envelope` then
returns the first element. Rule 4a measures compose-and-clamp as exactly commuting over 173 472
combinations, so nothing built here on the current arithmetic moves: the printed numbers are
bit-identical under either placement.

**P8-6 (minor) — `exact_paired_quantiles`' refusals were a weaker second copy.** Its transposed-pair
guard was `paired_bootstrap`'s twin and untested — deleting the six lines was green. It was also
*weaker* than `percentile` on two refusals it did not carry: a `float` level reached `.numerator`
and died with `AttributeError` where `-ml` §11.2.2 publishes a `TypeError`, and a level above 1
fell off the end of the atom loop and died with `IndexError` on `bounds[1]` where the note publishes
a `ValueError`. Both are reachable from the signature, which advertises `levels` as the caller's.
Closed with a shared `_check_level(level, *, caller)` — the two are the same estimator over a sample
and over a known distribution (`-ml` §11.2 reason 2), so they owe a caller the same errors — and the
three refusal tests now **parametrize over the estimators** rather than naming one, so a third has
to opt out rather than be forgotten. The `(0, 1]` bound is also what makes the atom loop total:
at `level <= 1` the final cumulative always satisfies the selector, so `bounds[1]` always exists.

**P8-7 (nit) — `AGENTS.md`.** The F2 clause's disposition has since been **ruled on** by plan v1.17
(`b6f578c`), which closes F2 by restating the residual's target as two survivors named and says
explicitly that renaming is not the answer, the residual's stated virtue being that it survives a
rename. So the clause is ratified rather than stale, and what it gains is the live constraint an
editor would otherwise revert: *do not rename `exact_paired_quantiles` to make the grep read 1*.
The 131-character line 81 is rewrapped and now carries the two-layer design-effect refusal.

**Also closed, from Pass 8's unnumbered nit:** the `n <= 0` refusal was written in `envelope_arms`
and `exact_paired_quantiles` and `grep -rn 'describes no rows' tests/` returned nothing — no test
reached either. It is the surviving half of the intent behind the retired
`…refuses_a_table_that_does_not_describe_its_rows`, whose own guard was correctly retired as
unrepresentable; this one is still representable, the table being the caller's. Now pinned on all
three functions that take one.

**A visible figure moved at `cc28d48` and was not recorded there.** Table C's estimator swap changes
`index.csv`, not only the code: over `1..100`, `latencyMsP50` goes **51 → 50**, because the retired
`int(round(pct/100 · (X−1)))` and Hyndman-Fan type 1 disagree by one rank at even sample sizes.
Anyone comparing runs across that commit sees the shift; `latencyMsP95` is unmoved at that size.

**Mutation table** — each `cp` aside, mutated, run alone, `cp` back, `diff -q` byte-identical
before the next.

| # | Mutation | Result |
|---|---|---|
| M1 | `_decided_by_line`'s `bound_by is None` branch → a constant string | killed (**was surviving**) |
| M2 | `envelope_arms`' restored precondition-4 refusal deleted | killed |
| M3 | that refusal weakened to the NaN-admitting `< 1.0` spelling | killed (only the `nan` case) |
| M4 | atom selector `>=` → `>` | killed (**was surviving**) |
| M5 | `_compose`'s `min`/`max` swapped | killed |
| M6 | `verdict` re-spells the composition and it drifts | killed |
| M7 | `exact_paired_quantiles`' transposed-pair guard deleted | killed (**was surviving**) |
| M8 | its `_check_level` calls deleted | killed |
| M9 | `_check_level`'s `0 < level` opened to `0 <= level` | killed |
| M10 | the `bound_by` iff broken — mcnemar path names MOVER-D twice | killed |
| M11 | M10 again, scoped to the new stats invariant assertion | killed |
| M12 | `envelope_arms`' `n <= 0` refusal deleted | killed |
| M13 | `verdict()`'s **own** precondition-4 check deleted | killed — *not* masked by M2's layer |
| M14 | **control** — P8-1's tie-break `<=,>=` → `<,>` | **SURVIVES, as intended** |

M14 is the control, not a gap: P8-1 is out of this round's scope and its mutant surviving is the
evidence that the tie-break and the clamp placement were left for the `data-scientist`'s ruling.

## 2026-09-08 — S1e Tables C, D, E and G: one percentile, the closed-form paired interval, and two required parameters

**What:** `docs/plans/small-model-benchmarking.md` **§4 S1e Tables C, D, E and G** (plan v1.16),
against `-ml` v1.18. `modelbench/{stats,results,report,packs}.py` and
`tests/{test_stats,test_results,test_report}.py`. **487 → 519 tests**, `.venv/bin/ruff check .`
clean, **13 mutations run, 11 killed, 2 surviving and both reported**. Tables C and G were applied
in the mandated order — C first, because Table G's `Fraction` level handed to the shipped
`_percentile(ordered, pct: float)` reads `Fraction(1, 40)` as `0.025` where that signature means
`2.5`, a unit error one substitution away from a plausible number.

**Table C — one percentile.** `stats.percentile(values, *, level: Fraction)` replaces both shipped
`int(round(pct/100 · (X−1)))` copies: Hyndman-Fan type 1, the rank taken as one integer expression
over the level's numerator and denominator, sorting a copy of its input, refusing an empty sample,
a `float` level and a level outside `(0, 1]`. Four `LEVEL_*` constants are the whole literal level
space. `results.py` imports *that object* — asserted as identity, not as equal behaviour, which is
what kills a second copy that happens to agree. `_index_row` keeps its own emptiness test, because
whether a latency figure exists at all is a decision about the run; **both index cells still bypass
`-ml` §11's two floors and closing that is S2's**, when the runner builds the `LatencyBlock` §3.5
requires every latency cell to be copied from.

**Table G — the bootstrap's levels.** `levels: tuple[Fraction, Fraction]`, keyword-only and
required, on `paired_bootstrap` and `paired_cluster_bootstrap`, with a transposed pair raising —
the one error that otherwise returns a plausible *inverted* interval. An all-continuous family with
`k > 1` takes its Bonferroni correction in the interval and had nowhere to put it.

**Table E — the clamp.** `_widen` and `paired_cluster_bootstrap` take `clamp`, required with no
default, `None` meaning do not clamp. `[-1, 1]` is correct for a difference of proportions and
false for `sep_z`; left as literals it printed an upper bound of 1.0 beside a point estimate of
1.48, the point estimate outside its own interval.

**Table D — the seed retires from the paired binary path.** `conservative_envelope(table, *,
design_effect)`: no `diffs`, no `B`, no seed, and the `n != len(diffs)` guard retires by being made
unrepresentable. Its bootstrap arm is now `exact_paired_quantiles`, the exact multinomial resample
quantile in integer arithmetic — no float tie-break, no Monte-Carlo estimate of an atomic quantile.
Both published anchors reproduce: `(0, 6, 0, 34)` at DEFF 1.00 renders `[3.2, 29.1] pp` and
`(4, 5, 3, 0)` renders `[−27.1, 58.3] pp`. `DecidedBy`'s `cluster-bootstrap` becomes
`conservative-envelope` and every string that named one arm of a two-arm interval is swept; the
`- decided by:` bullet loses its seed parenthetical and gains the audit that replaces it — which
arm bound each bound, carried on `Verdict.bound_by`. `PackRef.seed` stays, its consumer moved to
`-ml` §3.2d's continuous bootstrap.

**Three residual findings, all handed to `architect` and none fixed here.** (1) Table E's two
residuals go **blind** after a faithful edit: they match the shipped text `max(-1.0, point …)` /
`min(1.0, point …)`, which the edit necessarily rewrites, so a `clamp[1]` left as the literal `1.0`
is caught by neither — measured, that half-application passed the whole suite. A test asserting
both clamp components against an arbitrary `(0.9, 1.5)` stands in its place. (2) Table C's third
residual is at **2, not its stated 1**, because Table D's mandated closed form is a second `def
…quantiles`; renaming it to reach 1 would defeat the very property that residual exists to have,
so the name stands. (3) Table D's first residual is at **1, not 0**: the surviving line is
`def test_cluster_bootstrap_seed_is_keyword_only_with_no_default`, a substring collision on a test
for `cluster_bootstrap` — a function Table D explicitly *keeps*.

**Two surviving mutations, both reported rather than papered over.** `>=` → `>` in the atom
selector is an equivalent mutant on the reachable domain (an exact tie needs `40·cum == n**n`).
Not widening the exact arm by `sqrt(DEFF)` survived the suite as delivered, because the envelope's
own widening test sits on `(34, 6, 0, 0)` where MOVER-D binds both bounds; a new test asserts each
arm's widening separately on the table where the exact arm binds, and kills it.

## 2026-09-07 — P6-1: the stored `callSurface` shape, made total over invalid records

**What:** `docs/reviews/small-model-benchmarking-impl.md` **`## Pass 6`** finding **P6-1** (minor,
the pass's only one — Pass 5's four majors are dispositioned *fixed* and P5-7 *withdrawn*).
`modelbench/fingerprint.py` and `tests/test_fingerprint.py` only. **475 → 477 tests**,
`.venv/bin/ruff check .` clean, **3 mutations run, 3 killed**.

**The defect.** `to_dict` keyed its omission on the *value* — `{} if self.callSurface is None` —
so it fired on a **model** record whose surface was `None` too. Such a record validates as `null`
("something had the value and lost it"), serialises with no key, and reads back as `absent`
("never written"): the information loss the previous round closed in `validate()`, reintroduced one
method over. Unreachable through this package's writers, because `store()` validates before
`RunResult.to_dict()` — and reachable through **`model-bench migrate`** (§3.4.3), which reads with
`from_dict` and writes with `to_dict`, and is the one writer that serialises records `store()`
never validated. *(Corrected at Pass 7: an earlier phrasing here said a migration "by definition
walks records that did not validate". False, and it originated in the Pass 6 finding — §3.4.3
validates a record against **its own** schema entry, so most migrated records are valid. What is
unguarded is the write side, not the read, and that is enough to make the defect reachable.)*

**The fix is the conjunction, not either half.** The gate suggested keying on the arm instead
(`{} if self.armKind == "deterministic"`), which is a strictly worse trade on the same path:
executed over all eight `(armKind, callSurface)` shapes, it drops a **forbidden** surface off a
deterministic record, which then reads back **valid** — laundering an invalid reference arm into a
clean one and deleting the evidence of the claim §3.4.1 exists to refuse. So the key is omitted for
**exactly one** record, `armKind == "deterministic" and callSurface is None`, and written for every
other, whatever it holds. The round trip is then total over **all twelve** shapes —
`armKind` over `model`, `deterministic`, the `""` sentinel and an unrecognised value, times
`callSurface` over `"chat"`, `""` and `null` — where the value-keyed condition is asymmetric on
three of them and the arm-keyed one on two. *(The count is the suite's, not a transcript's, since
Pass 7: `ROUND_TRIP_SHAPES` in `tests/test_fingerprint.py` is that product, and this paragraph
states what it covers. The round originally claimed eight, enumerated by hand, and an independent
enumeration found nine — neither was in the suite, which was the finding.)*

**The mutations**, each applied to a file copied aside and restored by copy immediately, the tree
verified byte-identical after every one:

| # | The wrong implementation | Result |
|---|---|---|
| 1 | omission keyed on the value alone — the shipped `c523a35` condition | 1 failed — killed (`model-lost-its-surface`) |
| 2 | omission keyed on the arm alone — **the Pass 6 gate's suggested one-liner** | 1 failed — killed (`reference-arm-claiming-a-surface`) |
| 3 | the key always written — Pass 5's M2 / Pass 6's N1, re-run against the fix | 1 failed — killed, so P5-2's pin still holds |
| 4 | `callSurface is None and armKind != "model"` — omits on an unrecognised arm kind, the same laundering one arm-kind over | 2 failed — killed |
| 5 | `armKind == "deterministic" and not callSurface` — omits on `("deterministic", "")`, so a `forbidden` record migrates in clean | 1 failed — killed |
| 6 | the key never written | 40 failed — killed |

Mutations 4 and 5 are **Pass 7's** (P7-1): they passed the two-case version of the round-trip test,
which pinned the two narrowings its author had tried rather than the rule itself. The test is now a
`parametrize` over the product, so a wrong `omit` fails at the cell that names the record it
mishandles.

**One thing recorded rather than fixed, as a trigger.** §3.4.1's "`None` **iff** deterministic" is
now transcribed in two places — `validate()`'s deterministic branch and `to_dict`'s `omit` — with
nothing tying them together, and *which* value a profile pins is a **schema** fact that `to_dict`
hard-codes. At schema 1 there is exactly one such pin and the duplication is cheap, so it stays.
**A third transcription, or a schema-2 profile pinning another discriminator, is the moment to lift
the rule into `REQUIRED_BY_SCHEMA` rather than the moment to write it again.**

## 2026-09-07 — S1e Tables A and B, fix round: three states for the second discriminator, and two decisions that were held by comments

**What:** `docs/reviews/small-model-benchmarking-impl.md` **`## Pass 5`** findings **P5-1**, **P5-2**
and **P5-4**, plus **P5-3**'s code half once plan **v1.16** (`1ed8599`) made it writable.
`modelbench/fingerprint.py` and `tests/test_fingerprint.py` only — no other file in the package
moved, and neither remaining S1e unit opens either. **472 → 475 tests**, `.venv/bin/ruff check .`
clean, **7 mutations run, 7 killed**.

**P5-1 — `callSurface` reported *absent*, *empty* and *null* as one failure, under a test comment
claiming it did not.** The collapse was in `from_dict`: `d.get("callSurface")` maps a missing key
and a stored `null` to the same `None`, so a record written by something that *had* the surface and
lost it was reported to the operator as one that never carried it. The fix is `armKind`'s own
mechanism, applied one field over: the missing-key sentinel is `""`, and `validate()` answers `null`
before it answers `absent`. **Two reasons for three stored shapes, and the suite now says which
two** — `empty` still collapses into `absent`, because `""` *is* the sentinel and the two states are
indistinguishable on this field by construction, where `null` is not. That is the honest half of the
old comment, which claimed a three-state discipline directly above an assertion pinning the
collapse.

**The one subtlety, and it is why the sentinel is chosen per record rather than globally.** `None`
is not an absence marker on a **deterministic** arm — it is that arm's *value* (§3.4.1, "`None` iff
deterministic"), and it is what `to_dict` omits. So `from_dict` reconstructs a missing key as `None`
on a deterministic record and as `""` on a model one; a sentinel applied to both would report a
correct reference-arm record as carrying a forbidden surface, and fails the round trip.

**P5-2 — `to_dict` omits `callSurface` on a deterministic arm rather than writing `null`, and
nothing held it there.** The decision is right — that arm calls no surface, which is a different
fact from "we did not capture this", the one thing `null` means in this record (§3.4.2) — but the
round-trip test cannot see it, because `from_dict` reads a stored `null` back as the same `None` the
omission restores. Writing `"callSurface": self.callSurface` unconditionally passed all 472 tests.
It is now pinned by an assertion on the stored shape, with its positive twin on both model profiles
so "omit it" cannot be over-applied into "never write it". This matters beyond tidiness: S2's runner
and S3's `load_history` read that shape off disk.

**P5-3 — the retired residency element is now asserted by value, both keys named.** Plan v1.16
restates S1 done-condition 1 over the element's **key set** and re-scopes §4 S1e Table A's second
residual to `modelbench` plus `tests/conftest.py`, which makes `tests/test_fingerprint.py` the one
place the retired `lms ps --json` literal may live — and therefore makes the assertion writable at
all. Before it, an extra-key loop carrying a one-name tolerance for either retired key passed the
entire suite: the rule was right in the code and held by nothing. The new test constructs the
retired element on **both** snapshots and names both keys' `forbidden` problems, plus the two keys
it lacks. The residual stays at its stated target: `grep -rFn sizeBytes modelbench tests/conftest.py
--include='*.py'` → **0**.

**P5-4 — the half-swap test did not detect the half-application it is named for.** It asserted
`!= []`, and each of its two cases is *also* missing one of `{id, state}`, which produces an
`absent` problem on its own — so the assertion was satisfied whether or not the extra-key rule
existed. Deleting that rule entirely left both of its cases green while failing six other
parameters. It now asserts the surviving retired key's own `forbidden` problem, and its size case
uses the real retired spelling rather than the stand-in v1.15 forced on it.

**The mutations.** Each was applied to a file copied aside and restored by copy immediately, never
by `git restore`; the tree was verified byte-identical after every one.

| # | The wrong implementation | Result |
|---|---|---|
| 1 | `validate()` collapses `null` back into `absent` | 2 failed — killed |
| 2 | `from_dict` drops the missing-key sentinel (`d.get("callSurface")`) | 2 failed — killed |
| 3 | `to_dict` writes `"callSurface"` unconditionally — the review's M2, which had survived | 1 failed — killed |
| 4 | extra-key loop tolerates the retired size key — the review's M1, which had survived | 3 failed — killed |
| 5 | extra-key loop tolerates the retired identity key | 5 failed — killed |
| 6 | the extra-key `forbidden` loop deleted — the review's M5 | **10** failed — killed, where before this round it was 6 and **both** half-swap cases passed |
| 7 | the missing-key sentinel reaches a deterministic record too | 2 failed — killed |

**Also swept:** `ProblemReason`'s comment named two of `unknown`'s three families and omitted the
non-list snapshot; plan v1.16's Appendix A now writes all three out, so the module docstring says
what the plan says.

**Not in this round.** P5-5 and P5-6 are plan-side and closed at v1.16. Nothing from Pass 5 is
carried.

## 2026-09-07 — S1e Tables A and B: the residency source, the third arm profile, and the element shape

**What:** Plan **v1.15** §4 S1e **Table A** (`lmsCliCommit` → `residencySource`) and **Table B**
(`armKind` → `armProfile`, and `ARM_KINDS` decoupled), the first of the S1 fix round's three
implementation units. `modelbench/fingerprint.py` plus its three test surfaces; nothing else in
the package moved. **389 → 472 tests**, `.venv/bin/ruff check .` clean, all six of the two tables'
residuals at their stated target of zero, and **11 mutations run, 11 killed**.

**Table A — one field swapped, one element shape enforced.** `lmsCliCommit` recorded which `lms`
build produced the residency snapshot; after plan §3.4.4a nothing in the harness runs that CLI, so
the field had no source and could only have been kept by defaulting it to `""` — the
silently-defaulted fingerprint field FR-7 exists to refuse. `residencySource` replaces it
one-for-one on both model profiles: a `nonempty` token naming the surface the residency and catalog
data actually came from. The count is unchanged at 26 + 4 = 30.

**The fourth site carried no token, and it is the one that mattered.** `tests/conftest.py`'s
`residentModelsAtEnd` fixture declared the retired `lms ps --json` element where §3.4.4a's shape is
`{id, state}`. That field's tier is `present`, which checks presence and **never element shape**, so
the stale element validated, shipped green, and would have travelled into S2 — where `residency()`
emits `{id, state}` and the two disagree with nothing to catch them. The structural fix is
therefore not the fixture edit but the missing assertion (plan S1 done-condition 1):
`validate()` now checks each residency element's **whole key set** — exactly `{id, state}`, both
non-empty strings — so any key outside the pair is `forbidden`, a missing one `absent`, a `None`
`null`, and a non-string `unknown`. Problems name the element and key they came from
(`residentModelsAtEnd[0].modelKey`), which is what lets AC-2's block print them. Reverting the
fixture to the retired element now fails **63** tests where it previously failed none.

**Why that check has no residual over `modelKey`.** The retired element carried two keys and
Table A states a residual over only one of them, because `modelKey` keeps its meaning as a required
field of the same record — 90 lines in the tree — so a residual over it would fail on a *faithful*
edit, which is the trap §7 rule 5(b) forbids. The element-shape assertion is rule 5(b)'s named
alternative and covers that half: a half-application that swaps one retired key and keeps the other
fails there rather than in a count.

**Table B — the mapping key becomes a profile, the arm kind does not.** `REQUIRED_BY_SCHEMA[1]` is
re-keyed to the three profiles `model:chat` (30 fields), `model:embeddings` (26 — the chat set minus
`runtimeName`, `runtimeVersion`, `temperature`, `maxTokens`) and `deterministic` (11).
`FORBIDDEN_BY_ARM_KIND` becomes `FORBIDDEN_BY_ARM_PROFILE`, still the union-minus-mine **set
operation** and never a list, now over three profiles instead of two kinds. It resolves to exactly
what §3.4.1's table says, checked against independently written literals in the suite:
`{armId, armParametersHash}` on `model:chat`, those two plus the four chat-only fields on
`model:embeddings`, and 21 fields on `deterministic`. That last row is the derivation earning its
keep — nobody wrote the four embeddings names down, and forbidding them is exactly right, because
an embeddings call has no `runtime` object to observe and no sampling parameters to obey, so a
record carrying either is claiming something it cannot have measured.

**`ARM_KINDS` had to stop being derived from the forbidden mapping in the same edit.** It was
`frozenset(FORBIDDEN_BY_ARM_KIND)`; re-keying that mapping and leaving the derivation makes its
members the three *profiles*, so `armKind == "model"` fails the membership test in `validate()` and
**every model record returns `FieldProblem("armKind", "unknown")` and refuses on write** — a green
mapping and a dead harness. It is now derived from the profile keys' prefixes
(`p.split(":", 1)[0]`), which is decoupled from the forbidden mapping while staying a derivation
rather than a second hand-maintained list, and pinned by value in the suite. `CALL_SURFACES` comes
from the same split. Every `armKind == "model"` / `== "deterministic"` filter in `results.py` and
`report.py` is unchanged by design and was re-read to confirm each still means the two-valued
discriminator; `models --tested` is asserted to still return an embeddings arm.

**`callSurface` is a second discriminator, required with no default.** `Fingerprint` takes
`armKind` and `callSurface` (`None` **iff** deterministic) and derives `armProfile`; both are
members of no required set and are checked **before any mapping is consulted**, because without a
surface there is no profile and so no contract to report the fields against — answering a
surface-less model record with thirty `absent` problems would bury the one that is true. A
deterministic record carrying a surface is `forbidden`; a surface this build has never seen is
`unknown` rather than resolved to a profile key no mapping carries. `from_dict` strips both
discriminators (one left in `fields` lands in every profile's forbidden set) and `to_dict` omits
`callSurface` rather than writing `null` on a deterministic arm — that arm calls no surface, which
is a different fact from "we did not capture this". Identity (`__eq__`/`__hash__`) includes it.
Being required with no default is what made the type system enumerate the ten real `Fingerprint(`
construction sites for us, which is §7 rule 5's *adds rather than retires* half.

**Suite.** Both model profiles now get the per-required-field treatment — 30 + 26 cases for the
absent loop and the same for the null loop, where only `model` had them before — and the M-4
hand-transcribed literal gains a `model:embeddings` sibling, transcribed independently and
deliberately **not** derived from the chat one. Write-side acceptance covers all three profiles: a
clean embeddings arm stores, and one carrying `runtimeName` is refused on write.

**A plan friction worth recording.** S1 done-condition 1 names both retired residency keys, which
invites a test that spells the second one — and Table A's second residual requires that token to
reach **zero** across `modelbench` and `tests`, comments included. Written literally, the two
cannot both hold. Resolved by making the *implementation* rule key-set-exact, so it refuses either
retired key by construction, and asserting it through the rule and through `modelKey` (which has no
residual) rather than by naming the retired token. The tree therefore carries the token nowhere and
the behaviour is still pinned.

## 2026-09-03 — S1 fourth gate round: the nets that catch the first scorer's first mistake

**What:** Closed every major and minor from the fourth gate round on S1 —
`docs/reviews/small-model-benchmarking-impl.md` `## Pass 4` (**P4-1**…**P4-10**, plus nits
**P4-11**…**P4-13**) and `docs/reviews/small-model-benchmarking-ml.md` `## Pass 4` (**M-ML-8**,
**m-ML-9**…**m-ML-12**, nit **n-ML-8**) — against method note **v1.8** and plan **v1.8**.
Test-first throughout; **353 → 389 tests**, and **49 mutations run against the fixes, 49 killed,
no survivors**. `.venv/bin/ruff check .` clean.

**The round's shape.** Six of the fifteen findings are *"a mutation survives the suite"* — code
that was already right with nothing pinning it — so for those the test **is** the deliverable and
the implementation did not move. The rest divide into one real statistical correction (M-ML-8),
three note-published strings that were false on the path every comparison currently takes, and two
report-level nets that had been deferred to S2 and were pulled back to S1 because a net added after
the thing it protects has shipped is how all four of these rounds began.

**M-ML-8 (major) — the fail-safe path quantified with the narrower of its two instruments.**
B-ML-2's veto fixed what the non-`by-construction` path *decides* with two rounds ago and left what
it *quantifies* with on the bare percentile bootstrap. Measured exactly, that interval covers 0.939
at n=40 against MOVER-D's 0.976 while printing narrower bounds on **100%** of the probability mass,
and it degenerates at the sparse discordant counts `-ml` §3.2b says this lab will see: at n=30 with
`b=4, c=0` it returned `[3.3, 26.7] pp`, **excluding zero**, against an exact p of 0.125, because
four non-zero rows make `P(no +1 drawn) = (26/30)³⁰ = 1.4% < 2.5%` and a 2.5th percentile of zero
unreachable. Note v1.8 §3.4 Rule 4 replaces it with the **conservative envelope**
(`stats.conservative_envelope`): the wider of the `√DEFF`-widened bootstrap and the `√DEFF`-widened
MOVER-D, half-widths scaled about the same point estimate.

*Read bound by bound, not by picking the wider interval whole* — Rule 4's own stated property is
"uniformly at least as conservative as either alone", which choosing one interval does not deliver:
at `(4, 5, 3, 0)`, the tool-caller pack's own n, the bootstrap is the wider interval while MOVER-D's
lower bound is the more conservative one, so picking it whole would print a bound tighter than an
instrument the rule says it dominates. Confirmation that the reading is right came from the
rendered output: at DEFF 1.00 the guard-judge shape now prints
`+15.0 pp (95% CI [3.2, 29.1] pp)`, which is `-ml` §3.2e verdict 1's published string exactly —
Rule 4's "reduces to MOVER-D exactly at DEFF 1.00", observed rather than asserted.

**Not done, and deliberately: v1.8's second half, the closed-form percentile.** The note also says
the percentile "should" be computed in closed form rather than resampled, which removes the
seed/row-order sensitivity of an atomic quantile. The envelope hides that wherever MOVER-D is the
binding arm — measured, the overwhelming majority of tables — but not where the resample escapes it:
at n=12 the rendered lower bound still moves between `-27.1` and `-33.3 pp` across seeds (11 of 19
against 8 of 19, measured). Landing the closed form retires `-ml` §3.2d's seed from this decision
and with it review **P3-5**'s delivered contract (`PackRef.seed`, `verdict`'s `bootstrap_seed`, and
the report's `decided by: … (seed N)` line), which is a scope decision rather than a formula.
Carried to the coordinator, not taken here.

**Three note-published strings, all false on the default path** (m-ML-9, m-ML-10, m-ML-11). The
equality boundary now reads *"is at or above that"* — one comparative true across the whole branch,
and equality is reachable at this component's own n (`n_units=85, k=2, DEFF=1.9` gives
`mdd80 = 20.0 pp`, and `|b−c| = 17` over 85 rows is exactly that). The cluster-path label is
published verbatim in two variants keyed on the **design effect**, because the shipped sentence
asserted *"under clustering McNemar rejects too readily"* on a comparison that declares no
clustering — P3-3's defect surviving in the half P3-3 did not touch. The floor sentence gains an
effective-unit qualifier wherever `design_effect > 1.0`, because the floor is `b_min/n_eff` while
the McNemar p three clauses away is over the raw rows, and without it the line reads as a flat
contradiction of the number beside it (30.0 pp of floor beside `p=0.008`).

**P4-4 (major) → S1 done-condition 10 — the `aggregates`-versus-`items` cross-check.** An arm
declaring `BinaryMetric(m, successes=0, n=10)` for a metric no item declares scoreable printed
`0/10 = 0.000` — a claim that ten items were scored — in the same document as *"No verdict: no
paired data"*. On mismatch the arm is **excluded and named** in the `INVALID RESULTS EXCLUDED`
block: raising reproduces P4-5's shape, and suppressing one metric's row leaves a partly-trusted
arm in the comparison. **Two defects in the done-condition's own text were caught by the plan gate
mid-implementation and are recorded here because the code deviates from the written spec:**
**G3-6** — DC-10's selector names two disjoint vocabularies (`BinaryMetric.unit` is a denominator
noun, `PackRef.analysisUnit` a `pairingKey` component name), so the literal predicate is never true
and would have checked nothing; the code uses `metric.unit == roles.unit_kind(pack.role)`.
**G3-7** — DC-10 counts with `scored_outcome`, which *raises* for the sibling malformation; the
check treats that as a mismatch and names the offending item instead.

**P4-5 (major) — one bad item took the whole comparison down.** `IncompleteItemRecord` escaped
`compare_report` as a traceback at **exit 1**, outside §3.6a's closed `{0,2,3,4,5}`, with no report
written and the valid arms lost with it. `load_history` now quarantines such a record on read as an
ordinary `field` failure naming the item and the metric, which is AC-2's actual mechanism.

**The rest.** **P4-1** — a `--negative-control` run with nothing to duplicate wrote a durable report
claiming *"both arms are the same stored record … cannot fail"* ten lines above *"fewer than two
arms were selected"*; the banner is now decided after the arms are known and replaced, not merely
suppressed, so the artifact still says the mode was requested and did not run. **P4-2**, **P4-3**,
**P4-7**, **P4-8**, **P4-9**, **m-ML-12** — six surviving mutations pinned, of which m-ML-12's and
P4-2's each **flip a printed verdict** (the veto tested at `alpha_family` instead of the Holm step;
`holm_tested` hardcoded `True`, which reprints Pass 1's blocker verbatim — a significance claim
beside its own *"not tested (Holm stops here)"* row). **P4-6** — a zero-denominator aggregate is
rendered as `0/0 — no observations` rather than silently dropped. **P4-10** — two arms of one model,
which is plan §5 test 19a, are told apart by session (falling back to `runId` where the session does
not distinguish, and adding nothing where the model key already does). **n-ML-8** — Rule 7's raise
is no longer gated on `holm_tested`, which is not one of the theorem's premises.

**One defect found by rendering the output rather than by an assertion, as in every round so far.**
The P4-4 exclusion path left two individually-true sentences contradicting each other: the block at
the top said the arms were excluded, and the verdict line below still said *"fewer than two arms
were **selected** … Check `--models` and `--session`"* — the wrong remedy, sending a scorer author
to their command line when the defect is in their record. Split into its own reason.

**Verification:** from `model-bench/`, `.venv/bin/python -m pytest -q -m "" -rsx` → `389 passed`,
nothing skipped, xfailed or deselected; `.venv/bin/python -m pytest --collect-only -q -m ""` →
`389 tests collected`, so collected equals run; `.venv/bin/ruff check .` → `All checks passed!`.
Mutation testing was run by copying each source file aside and restoring from the copy, never
through git.

## 2026-09-03 — S1 third gate round: absence is not an outcome, and five sentences that were false

**What:** Closed the third round of gate findings on S1 —
`docs/reviews/small-model-benchmarking-impl.md` `## Pass 3` (**P3-1**…**P3-15**) and
`docs/reviews/small-model-benchmarking-ml.md` `## Pass 3` (**M-ML-7**, **m-ML-7**, **m-ML-8**,
**n-ML-4**…**n-ML-6**) — against method note **v1.7** and plan **v1.7**. Test-first throughout;
314 → 353 tests, and 28 mutations run against the fixes with **27 killed and one equivalent by
construction**. This round is dominated by a single failure mode: **the blocker and four of the six
majors are about what the report *says* rather than what it computes** — two false clauses, one
disclosure the report omitted entirely, and one number it never printed — and every one was found
by rendering a report and reading it, not by an assertion. The remaining two majors are tests that
could not fail.

**P3-1 (blocker) — an arm holding no data for a metric was scored as failing every item.**
`report.py` carried two defaults of its own: a missing `scoreable` entry read as *scoreable*, and a
missing `counts` entry then scored the row a **loss**. Together they rendered
*"cand is better than incumbent: +100.0 pp (95% CI [60.8, 100.0] pp) … p=0.002"* against an arm
whose ten items carried `counts={}` and `scoreable={}` — while the §4.3 tally, whose entire job is
to make dropped rows visible, printed `0 unscoreable in both`. Absence was laundered into the
denominator's complement, the mirror image of the laundering `-ml` §4.3 forbids. Which state a row
is in is now **`ItemResult.scored_outcome`'s call and nothing else's**, with three declared answers:
absent or `False` in `scoreable` is *no outcome* and routes through the tally; `True` **must** carry
a count, and one that does not is refused (`IncompleteItemRecord`) rather than read as a zero — a
scorer that declares an item scored must supply its score. A metric whose paired intersection is
empty now renders an explicit refusal with the tally beneath it and gets **no** `ResolvingPower`
(`n_effective` of zero is not a small sample); its Holm row prints `—` and `no verdict — no paired
data` rather than `mcnemar_exact(0, 0)`'s misleading `1.000`. **This is a contract on S2's scorers**
and is recorded in `AGENTS.md`.

**M-ML-7 / P3-2 (found independently by both gates) — the "not distinguishable" verdict asserted
the observed difference was below the MDD without checking.** `"; the observed X pp is below that."`
was fixed prose, false whenever `|diff| >= mdd80` — which is §7.1's *normal case for a model swap*,
a candidate that wins more than it loses without strictly dominating. Measured by the statistics
gate: 268 of the 1 580 by-construction tables that print the clause printed it falsely. Note v1.7
§3.2e mandates a conditional clause and publishes the alternate wording verbatim, discordance
counts included; it is implemented as published. The comparison is **strict**, and that boundary is
reachable rather than theoretical — swept over `6 <= n <= 120`, `|diff| == mdd80` occurs at k=2 for
n = 90, 100 and 120, where "below that" is false a second way — so it has its own test.

**P3-3 — the fail-safe path claimed a widening that never happened.** The cluster-path label read
*"widened by sqrt(DEFF)=1.00 **for the declared clustering**"* at `design_effect == 1.0`: nothing
was widened and no clustering was declared, on the path **every** comparison carries until S2's
determinism probe lands. What actually displaces McNemar there is Rule 4's other half, the
`basis` — and the sentence never named it. The clause is now conditional: it names the design
effect where one was applied, and the basis where none was. **The sentence itself is not note-owned
prose** — `-ml` §3.4 Rule 4 requires "the design effect and its basis printed" on this path but
publishes no string for it, which is recorded as an open item for `data-scientist`.

**P3-4 — `--negative-control` wrote a durable report indistinguishable from a real comparison.**
The mode puts two copies of one record in both arms, so `b = c = 0` is arithmetic; the report said
nothing about that (`grep -ic negative` returned 0) and was filed beside real comparisons under a
filename differing only in its sequence number. A reader got a plausible validated null. The report
now opens with a banner naming it a wiring smoke check that **cannot fail**, and pointing at the
real negative control (two independent runs, an acceptance step). The code comment that claimed the
report already said this is corrected.

**P3-5 — the bootstrap seed was a literal in the renderer and was never printed.** `report.py`
passed `bootstrap_seed=20260902`, duplicating the manifest's `sampling.seed` in a type that had no
field for it, so the pack's own declaration could not reach the decision — and on the fail-safe path
the seeded bootstrap is what decides. `PackRef` now carries `seed` with **no default**, read from
the manifest (a manifest omitting it is refused by name), and the `decided by:` line prints it where
a resample actually ran. The test asserting the printed line was not enough on its own: it left the
literal alive, printing one seed over an interval resampled at another. It now asserts that two
packs differing only in `seed` render **different intervals** — possible only at a fixture coarse
enough for the percentile to move (n=12, b=5, c=3; at n=40 the rendered bounds are identical at
every seed tried). The fingerprint half stays S2's.

**P3-6, P3-7 — two tests that could not fail.** The suite's only k=2 α assertion was
`"alpha=0.025" in md`, satisfied by the family-wise paragraph rather than by the MDD sentence it was
placed to guard, so `provenance` naming the wrong α survived; it now asserts the whole provenance
parenthetical, and the floor's own α beside it. The exploratory-label test asserted two strings'
presence and not their pairing, so **inverting** the filter — labelling the pre-registered verdict
metrics "exploratory" and hiding the genuinely exploratory ones — was green; it now asserts the
rendered line whole plus the negative. Both inversions now fail.

**The minors and nits.** `compare --session` had no test at all (P3-8) and now has two, from both
directions. `index.csv`'s `valid` column (P3-9), the `armKind` absent-vs-null discriminator
(P3-10), `verdict()`'s and `resolving_power`'s design-effect guards (P3-11), `compare_report`'s
headline-membership guard (P3-12, whose failure mode was a bare `StopIteration`), `wilson_interval`'s
probability clamps and `pack_ref_from_manifest`'s `analysisUnit` check (P3-15) each gained the one
assertion that kills their surviving mutation. `holm_steps`' `alpha` default — the second literal
`0.05` in the module that declares there is only one (P3-13, n-ML-4) — is **removed** rather than
re-pointed at `ALPHA_FAMILY`, matching plan v1.7's signature block. `resolving_power` now refuses
`design_effect < 1.0` at construction instead of `<= 0` (n-ML-5): below 1 a design effect *inflates*
effective *n* and shrinks both printed bounds, and the refusal used to arrive a layer later. The
report filename is now the manifest's `packId`, not the pack directory name (P3-14) — the half
Pass 1's m-6 left behind. m-ML-7's `floor_clause` boundary, m-ML-8's duplicated MDD stem and
n-ML-6's hard-coded `"80% power"` were closed by the same round.

**One correction to a test, not to the code.** `verdict()`'s design-effect precondition raised the
*same sentence* as `paired_cluster_bootstrap`'s identical bound one layer down, so the obvious test
passed with the precondition deleted. What it asserts now is the **ordering** Rule 4 states — every
precondition checked before any instrument is selected — which is visible only when the two orders
produce different errors.

**Verification.** `.venv/bin/python -m pytest -q -m "" -rsx` → **353 passed**, nothing skipped,
deselected or xfailed; `--collect-only` collects 353, so the run count equals the collected count.
`.venv/bin/ruff check .` → clean. 28 distinct mutations run, **27 killed**. The survivor is
**equivalent by construction**: restoring `report.py`'s duplicate of the MDD stem with *identical*
text renders identically, so no test can distinguish it. The mutation that matters — the same stem
edited in `stats.py` while the duplicate stays stale, which is exactly the drift m-ML-8 predicted
M-ML-7's fix would cause — **is** killed, by the test asserting the report renders
`stats.mdd_clause`'s own string.

## 2026-09-03 — S1 second gate round: the floor's α, McNemar as a veto, Rule 7 by path

**What:** Closed the second round of gate findings on S1 —
`docs/reviews/small-model-benchmarking-ml.md` `## Pass 2` (**B-ML-2**, **M-ML-6**, **m-ML-6**) and
`docs/reviews/small-model-benchmarking-impl.md` `## Pass 2` (**P2-1**…**P2-5**) — against method
note **v1.6**, which landed this morning and changed Rules 3, 4, 6 and 7, §3.3, §7.1 and §7.3.
Test-first throughout; 296 → 314 tests, and 22 mutations run with **one survivor, equivalent by
construction** (see below).

**M-ML-6 — the observable floor moves to the unadjusted α, and `ResolvingPower` carries two αs.**
The floor claims *"below Y nothing can reach significance at any observed outcome"*, which is true
only at the **loosest** Holm step a member can face. Printed at α/k it is `7/n` and **false**: at
n=40 a rank-2 member with b=6, c=0 reaches p=0.031, clears its own 0.05 step, and its 15.0 pp sits
below the 17.5 pp the old floor printed. `resolving_power` now takes `alpha_family` **and**
`alpha_mdd`, both keyword-only with no default, and each bound is computed at its own — the floor
at `alpha_family`, the MDD unchanged at `α/k`. §7.1's mandatory sentence names both. The third α,
Holm's data-dependent `alpha_step`, stays `verdict()`'s parameter rather than becoming a field:
it is known only after the family is ranked, so a field would be `None` until it was not, and the
number would have two homes. The sweep went past `stats.py`: `PackMetrics.alpha` — a code-side
restatement of exactly this α — is now `alpha_family` / `alpha_mdd`, and `report.py` reads them
instead of recomputing `0.05 / len(family)` inline. `stats.ALPHA_FAMILY` is the single home of the
unadjusted 0.05.

*Consequence, taken deliberately:* the α/k floor was reducing Holm to Bonferroni for every
difference in `[6/n, 7/n)` — precisely the band §7.3 already prices as the cost of a second verdict
metric — so the build was charging that price twice. It no longer is, and the rendered family table
for the review's own case now reads `distinguishable` where it read `not distinguishable — below
the observable floor`.

**B-ML-2 — the substitute path is a conjunction, not an interval.** At `design_effect == 1.0` with
`basis == "assumed"` — the fail-safe **every** comparison carries until S2 lands the determinism
probe — the decision moves off McNemar and `sqrt(1.0)` widens nothing, so a bare percentile interval
was deciding. Reproduced at n=40: `(b=7, c=1)`, `(9, 2)` and `(11, 3)` all rendered
*distinguishable* at p = 0.057–0.070, where the exact test refuses, and Rule 7 does not catch them
because 15.0, 17.5 and 20.0 pp are all **at or above** the floor. The non-`by-construction` decision
is now *"the widened CI excludes zero **and** `mcnemar_exact <= alpha_step`"*. Note v1.6's Rule 4
permits this explicitly: the objection to McNemar under clustering is that it *rejects* too readily,
and a necessary condition only ever removes rejections, so the pair is uniformly at least as
conservative as either instrument alone. The verdict strings say which instrument played which
role — one sentence for both paths would have contradicted one of them.

**m-ML-6 — Rule 7 splits by path.** On `mcnemar-exact` the invariant is a **theorem** (re-verified
here by binary search over every `b + c <= 400` at both αs, zero violations), so a fire is a module
bug and now raises `Rule7Violation`; silently demoting discarded exactly the detector property the
rule exists for. On `cluster-bootstrap` it stays demote-and-name, because a widened interval and a
shrunken effective *n* legitimately disagree there. Sequenced **after** M-ML-6, as the review
required: at α/k the McNemar branch is reachable and the raise would have fired on correct data. A
fifth precondition makes the theorem's premise checkable rather than assumed — `alpha_step` must
lie in `[alpha_mdd, alpha_family]`, which Holm's own steps do by construction.

**The bin-edge truncation guard stays; its justification was corrected.** It was load-bearing on the
17.5 pp cell at n=40 — **the cell M-ML-6 deleted**. Swept to n ≤ 2000: with `b_min = 7` naive
truncation misfires at n = 5, 10, 20, 40; with `b_min = 6`, which the floor now always uses, never.
So the guard is **defensive**, kept because `b_min` is a function of α and any future α reopens the
hazard, and its test is no longer a regression pin on a published figure. Both the code comment and
the test say so. The `floor(x/precision)` expression is still pinned by the code's own form — a
test now also pins the `precision` parameter, since `floor(x*1000)` agrees at the default and
nowhere else, which is how the original sweep missed the hazard.

**Engineering findings.** **P2-1** — `"measured"` at DEFF 1.0 had no test at Rule 4's branch, and
widening `mcnemar_may_decide` to admit it survived all 296 tests; two mirror tests (unit and report)
now close it. **P2-2** — `validate()` accepted `benchSchemaVersion: true` (`True == 1`) while
`load_history` quarantined it, so `store()` wrote a record the reader refused; the bool guard now
lives at both enforcement points, and a quarantined bool no longer lands in a field typed
`int | None`. **P2-3** — `holm_steps` builds its list without a `None`-filter and `report.py` zips
`strict=True`, so a short ladder raises instead of silently dropping a pre-registered verdict
metric. **P2-4** — accepted in part, with a correction: `Path(".").name` is `""` so `"."` is
redundant, but **`Path("..").name` is `".."`**, so `".."` is *not* already caught and stays; one
third of the guard was unreachable, not two thirds. **P2-5** — `packs.py`'s `contentHash` docstring
now says `None`, matching the code.

**Verification.** `.venv/bin/python -m pytest -q` → `314 passed`; `.venv/bin/ruff check .` → clean.
22 mutations run against the fixes, **21 killed**. The survivor — restoring `holm_steps`'
`None`-filter — is **equivalent by construction**: every index is assigned, so the filter changes
no output on its own. Compounding it with a ladder that actually returns short is killed twice over
(the strict zip and the length invariant), which is the honest statement of what P2-3's fix buys.

## 2026-09-03 — S1 gate remediation: both blockers, all ten majors, and Rule 7

**What:** Fixed the findings of the two independent S1 gates —
`docs/reviews/small-model-benchmarking-impl.md` (`analyst`: 1 blocker, 6 majors, 7 minors, 4 nits)
and `docs/reviews/small-model-benchmarking-ml.md` (`data-scientist`: 1 blocker, 4 majors, 5 minors,
3 nits) — against plan v1.5 and method note v1.5. Test-first throughout; every fix was
mutation-tested and the reviewer's **ten surviving mutations are now all killed**.

**The two blockers.**

- **B-ML-1 — the clustered decision path did not cluster.** `verdict()`'s substitute for McNemar
  was `paired_bootstrap` over the *rows* of the paired table: an i.i.d. resample of observations
  the declared design effect says are correlated, so the interval was identical at DEFF 2, 4 and 7
  and *narrower* than the MOVER-D it replaced. It changed the instrument's name, not its interval.
  New primitive `paired_cluster_bootstrap` inflates the percentile half-widths about the point
  estimate by `sqrt(design_effect)` — the Kish variance ratio is exactly the quantity that converts
  (`-ml` §3.4 Rule 5). **This is the note's "smallest honest version", taken deliberately:** the
  structurally right fix resamples clusters of paired differences, and `PairedOutcomes` carries one
  row per analysis unit with no grouping, which could only come from a pack declaring
  `replicatesPerScript > 1` — something Rule 6 makes a validation error while only the one-level
  `cluster_bootstrap` exists. Building it now would have had no data to consume and no seam to
  reach it.
- **B-1 / M-ML-2 — Holm–Bonferroni was printed and never applied.** `report.py` called `verdict()`
  without `alpha_step`, so every metric was decided at plain Bonferroni α/k, and `holm_thresholds`
  had no step-down stop. `compare_report` now runs **two passes** — Holm is a property of the
  family, so no verdict can be decided until every p-value exists — and `holm_thresholds` is
  replaced by `holm_steps`, returning a `HolmStep` per member with its rank, threshold, `tested`
  and `rejected`. `verdict()` gained `holm_tested`, which is the stop.

**Rule 7 (`-ml` v1.5 §3.4), enforced in `verdict()` rather than left to a test.** No verdict path
returns `distinguishable` when `|diff|` is below `resolving.observable_floor`. Three decisions in
it, each with a reason:

- **It demotes and says so; it does not raise.** The note's contrast is code-versus-test, not
  raise-versus-demote, and a raise would be unreachable in practice: the √DEFF-widened bootstrap
  and McNemar's exact rejection region are different instruments that do not align by construction
  (measured — at DEFF 2 on the `(34, 6, 0, 0)` table the widened interval still excludes zero while
  15.0 pp sits below the 30.0 pp floor). The demotion renders the contradiction it resolved, which
  surfaces the defect more loudly than a traceback the report never prints.
- **It compares against the exact float, never `format_floor_pp`'s truncation** — otherwise the
  invariant inherits the presentation layer's rounding and can fire, or fail to fire, by 0.05 pp.
- **The converse is not asserted.** `|diff| >= floor` does not imply distinguishable; §3.2c's row 4
  `(20, 8, 2, 10)` is the counterexample already in the suite — 15.0 pp exactly on the α=0.05
  floor, p = 7/64, not distinguishable.

It never fires on the McNemar branch: `test_the_mcnemar_path_satisfies_rule_7_by_construction`
checks every `(b, c)` split at n ∈ {12, 20, 30, 38, 40, 48, 85} and α ∈ {0.05, 0.025}. That
asymmetry is what makes it a detector rather than a formality.

**The floor's rounding direction, per the adjudication: the floor truncates, the MDD ceilings.**
`stats.format_floor_pp` is the one place the direction lives, and it is where the report and the
verdict strings both print from — the tests assert **through the formatter**, because re-rounding
inside a test (`round(observable_floor(...) * 100, 1)`) asserts the presentation layer's arithmetic
against itself. `ResolvingPower.observable_floor` stays exact, so Rule 7's guard is not weakened.
Truncation is guarded (`math.floor(x / precision + 1e-12)`), mirroring the MDD's `- 1e-12`, and
**the guard is load-bearing rather than defensive**: `7/40 = 0.175` is `174.99999999999997` bins in
IEEE doubles, so naive truncation prints `17.4` for the α=0.025, n=38–40 row the note publishes as
**17.5**. Corrected cells: 15.8→15.7 (n=38), 7.1→7.0 (n=85), 46.7→46.6 (n=15, α=0.025); 58.3 and
23.3 were already truncations.

**The other majors.**

- `load_history` validated *after* the pack filter, so a record whose `packId` was blanked or
  deleted on disk landed in **neither** returned list — the comparison quietly lost an arm (M-1).
  The filter now drops only a record that *says* it belongs to another pack; it also applies to an
  unknown schema, whose `packId` is readable, and stays **off** `unparseable`, which cannot declare
  one (m-1).
- `RunResult.designEffect`/`basis` lost their dataclass defaults (M-2, m-ML-3, plan v1.5 §3.5). The
  legacy fallback stays in `from_dict`, where it is a reader's §3.4.3 compatibility rule.
- `BinaryMetric` gained a required `unit`, and the Arms table prints a Wilson interval only over
  the analysis unit (M-ML-3). §4.4's first mandatory consequence is verbatim *"Never print a Wilson
  interval over a turn-pooled count"*, and a turn-pooled 142/320 was printing ±5 pp where the
  honest bound is ~48.7 pp. The count is never suppressed; only the precision claim is.
- `_paired_rows` returns a `PairedRows` tally and every verdict prints it — the `asymmetry` count
  §4.3's paired corollary requires, plus rows present in one arm only and unscoreable in both
  (M-5, M-ML-4). It is printed even when nothing was dropped, because otherwise a reader cannot
  tell a shrunken `n` from a full one.
- `min_detectable_difference` raises `UnattainablePower` below `b_min(alpha)` units instead of
  converging on its bisection bracket and returning `1.0`; `ResolvingPower.mdd80` is then `None`
  and the line reads *"No difference is resolvable…"* (M-ML-1). The delivered build printed
  *"resolves differences of >=100.0 pp with 80% power"* where power is identically **zero**.
- A comparison with fewer than two arms has its own reason, and `--models` naming a key with no
  stored run exits **2** rather than silently rendering a one-arm report (M-6).
- The basis/design-effect propagation is now tested at report level (M-3), and prints the **weaker
  of the two actual bases** rather than collapsing to `assumed` — false provenance in the one
  sentence whose job is auditability (m-ML-4). The decision rule is unchanged.
- `REQUIRED_BY_SCHEMA` and `FORBIDDEN_BY_ARM_KIND` are pinned against **independently transcribed
  literals**, by name and by tier (M-4). Parametrizing over them meant deleting an entry deleted
  its test case rather than failing one.

**Minors and nits:** the unpaired label distinguishes a content-hash divergence from a version one
(m-2); `_unit_ids` is called by `_paired_rows` instead of being dead code the docstring names
(m-3); `--role` and the index's `latencyMsP95` gained tests (m-4); `PackRef.contentHash` is
`str | None` so "not yet computed" is expressible (m-5); `compare` filters by the manifest's
`packId`, not the directory name (m-6); `store()` refuses a `runId` that is not a bare filename
(m-7); the tautological assertion is gone (n-1); `Fingerprint` copies its mapping behind a
`MappingProxyType` and hashes its **values** (n-2); an absent `aggregates` block is reported as
`unparseable` rather than repaired into an empty one (n-3); the conditionality clause names the
pack's own sample noun (n-4, m-ML-5); the `-ml` §3.2c fixtures are republished at 10 dp and
asserted at the mandated **1e-9 on the proportion**, with the docstring's margin claim corrected
from four orders to three (m-ML-1); `test_z_95_matches_the_inverse_normal_cdf` records that the
pinned literal is one ULP from `NormalDist().inv_cdf(0.975)` and must not be tightened to `==`
(n-ML-3).

**One finding declined, with its reason.** n-ML-1 asked for the floor and the MDD to share a
denominator (`observable_floor` divides by the unfloored `n_effective`; `min_detectable_difference`
floors first). Unifying them would make one of the two anti-conservative: Rule 3's principle is to
round each printed bound in the direction that keeps its own claim true, and the two claims point
opposite ways — a **larger** MDD is the safe error, a **smaller** floor is. The asymmetry is now
documented at `observable_floor`, which is the one line the finding asked for.

**Two defects found by reading rendered output, not assertions** — the same discipline that caught
the CI-orientation bug at S1. The "Best case — assumes the candidate wins every…" caveat was still
printing where no MDD exists, qualifying a figure that is not on the page; and the clustered label
was appended to two of the five verdict strings rather than all of them, so a reader seeing only a
demoted verdict was never told which instrument produced it.

**Verification, from `model-bench/`:** `.venv/bin/python -m pytest -q` → **296 passed** in 2.12s,
exit 0 (0 failed, 0 skipped, 0 deselected — the `live` marker still deselects nothing because no
live test exists until S2). `.venv/bin/ruff check .` → `All checks passed!`. **34 source mutations
against a scratch copy — 34 killed, 0 survivors**, including all ten the `analyst` gate reported as
surviving and 24 new ones aimed at this change's own fixes. Two of the new ones initially survived,
both because a test asserted a passthrough field instead of the behaviour it gates; both tests were
rewritten onto cases where the mutation changes a verdict.

## 2026-09-03 — S1: fingerprint, results, stats, report, CLI (no model calls)

**What:** Built the harness core per stage S1 of `docs/plans/small-model-benchmarking.md` §4 —
everything that decides whether a number may be printed, and nothing that produces one. No model
calls, no network, no LM Studio, no pack loader: the whole S1 suite runs offline.

- `modelbench/fingerprint.py` — `Fingerprint` (frozen, `armKind`-discriminated), `FieldSpec`,
  `FieldProblem`, `REQUIRED_BY_SCHEMA` (`{schemaVersion: {armKind: {field: spec}}}`) and
  `FORBIDDEN_BY_ARM_KIND`. Fields are held in a **mapping, not dataclass attributes**, because a
  dataclass with `None` defaults collapses *absent* into *null* — the two states plan §3.4.2 exists
  to separate. `validate()` returns problems and never raises; the `deterministic` arm kind
  forbids every model field, so `{"modelKey": "bm25"}` fails loudly on write (plan §3.4.1, gate B-3).
- `modelbench/results.py` — `ItemResult`, `RunResult`, `InvalidRecord`, `BENCH_SCHEMA_VERSION = 1`,
  a **closed union** of five typed aggregate dataclasses, `store()` (raises, no bypass parameter),
  `load_history()` (returns `(valid, invalid)`, re-validating each record against **its own**
  `benchSchemaVersion`), `rebuild_index()` and `models_with_stored_results()`.
- `modelbench/stats.py` — implements `docs/plans/small-model-benchmarking-ml.md` §3.4's six binding
  rules and nothing else: `wilson_interval` (`z` keyword-only, defaulting to the pinned
  `_Z_95 = 1.959963984540054`), `mcnemar_exact`, `mover_d_interval`, `paired_bootstrap`,
  `cluster_bootstrap`, `PairedOutcomes` (duplicate-unit guard in `__post_init__`, so it holds on
  every construction route), `resolving_power`/`ResolvingPower`, `min_detectable_difference`
  (exact bisection over the McNemar rejection region, ceilinged to the printed precision, and
  taking `n_effective: float` so a raw `int` count raises `TypeError`), `observable_floor`,
  `design_effect`/`effective_n`/`width_inflation`, `verdict()` and `holm_thresholds`.
- `modelbench/report.py` — `compare_report()`: the excluded-invalid block (AC-2), the pack
  version/content-hash banners (AC-3), the `SCHEMA VERSIONS IN THIS COMPARISON` line (§3.4.3), the
  comparison-kind line (§3.7), per-arm Wilson intervals labelled *descriptive, not the comparison
  instrument*, the resolving-power line, the three verdict strings (AC-4), Holm–Bonferroni for a
  k>1 family, and the marginal-overlap diagnostic with its footnote.
- `modelbench/packs.py` — `PackRef`, `PackMetrics`, `metrics_from_manifest`,
  `check_sampling_contract`, `pack_ref_from_manifest`. **Not** S2's pack loader: no content hash,
  no AST import walk, no data-file row-count identity. `PackRef` extends Appendix A's five fields
  with `pairingKey` and `analysisUnit`, without which §3.3's analysis-unit resolution has no source.
- `modelbench/roles.py` — FR-21's five roles and `-ml` §3.3's unit-kind column.
- `modelbench/cli.py` + `modelbench/__main__.py` — `compare` (with `--negative-control`),
  `index rebuild`, `models --tested`; §3.6a's closed exit-code set. `attest`, `validate` and `run`
  are S2's and their absence is asserted by a test.
- `run.sh` — the S0 guard block deleted, as S0's own entry said S1 would.

**Two decisions taken here that the plan does not state, both additive and both flagged to
`architect`:**

- **`RunResult` gains `designEffect: float` and `basis`.** §5 test 12b requires `runner` to *set*
  `basis`, and `-ml` §3.4 Rule 4 decides which instrument may decide from it — but the plan's
  `RunResult` shape carries neither, and a report cannot recompute either after the fact. Without
  them S1 done-condition 5b is unsatisfiable. The degradation is fail-safe: any arm not
  `by-construction` drops the comparison to `assumed`, which moves the decision off McNemar.
- **`FieldProblem.reason` gains `"unknown"`** beside Appendix A's four, for a discriminator this
  build cannot interpret — an unrecognized `armKind`, or a `benchSchemaVersion` from the future.
  Forcing either into `absent`/`empty` would mislabel it.

**One defect found and fixed by reading the rendered output rather than the assertions:** when arm
B won, `verdict()` re-oriented the difference to the winner (`+66.7 pp`) but left the confidence
interval in A-minus-B orientation (`[-86.2, -29.9]`) — a positive effect printed beside a wholly
negative interval. Nothing raised; it is a plausible-looking, internally contradictory line, which
is the exact failure mode a measuring instrument must not have. The non-significant strings now
keep the signed A-minus-B difference for the same reason.

**Verification:** `.venv/bin/python -m pytest -q` from `model-bench/` → **233 passed**, exit 0
(0 failed / 0 skipped; the `live` marker deselects nothing at S1 because no live test exists yet).
`.venv/bin/ruff check .` → `All checks passed!`. `./run.sh --help` and `./run.sh models --tested`
both exit 0. Every done-condition test was mutation-tested; the load-bearing one is S1
done-condition 5(c), where pairing on the conversation id instead of the pack-declared
`analysisUnit` is caught by the captured-argument assertion independently of the raise.

## 2026-09-02 — S0: component skeleton

**What:** Created the `model-bench/` component per stage S0 of
`docs/plans/small-model-benchmarking.md` §4 — packaging, scripts, docs skeleton and an empty
package/suite. No harness code: S0's done-condition is deliberately an empty test suite, so that
S1–S8 land against a tree that already builds and lints.

- `pyproject.toml` — `requires-python = ">=3.12"`, **no runtime dependencies** (plan §3.2, a hard
  design constraint), dev extras `pytest>=9.1,<10` + `ruff>=0.14,<0.15`, ruff `select = ["E","F","W","I"]`
  / `line-length = 100` (mcp-monitor's shape), pytest `testpaths = ["tests"]` plus falkor-chat's
  live-test convention verbatim: `addopts = '-ra -m "not live"'` and a `live` marker.
- `setup.sh` — adapted from `mcp-monitor/setup.sh`: idempotent, `--recreate`, resolves paths from the
  script's own location, ends with an import smoke test.
- `run.sh` — the mcp-monitor shape (venv check, then `exec .venv/bin/python -m modelbench "$@"`) with
  an **S0 guard**: `modelbench/__main__.py` does not exist until S1, so the script reports that in
  words and exits 1 rather than `exec`-ing into a `No module named` traceback. S1 deletes the guard.
- `.gitignore` — `.venv/`, `host.json` (the operator-attested fingerprint fields, plan §3.4),
  `results/transcripts/` (raw model output: large, and not needed for any comparison, plan §3.5).
- `README.md` — what the tool is, and the three non-features stated up front: no CI/scheduler, no
  pass/fail gate, no leaderboard or cross-role aggregate.
- `AGENTS.md` — working context: current state, the hard rules (zero runtime deps, FR-23 standalone,
  no cross-role aggregate), the `live` marker, the attested fingerprint fields, and the note that
  an empty suite exits 5.
- `docs/{BACKLOG.md,HISTORY.md}` plus empty `requirements/ plans/ reviews/ test-plans/ test-reports/`
  held by `.gitkeep` files. `BACKLOG.md` is seeded with the two items plan §7 carries forward.
- `modelbench/__init__.py` (`__version__`) and `tests/test_package.py` — one install smoke test,
  asserting `modelbench.__version__` equals the installed distribution's metadata version. The plan
  called for an empty suite at S0, but pytest exits 5 (`EXIT_NOTESTSCOLLECTED`) when nothing is
  collected, so "runs and passes with zero tests collected" cannot return 0 (plan gate finding m1).
  Resolved with this one real test rather than by configuring the exit code away: a permanent
  "no tests ran is fine" setting would still be in place at S5 and would hide a collection
  breakage. The assertion is not filler — that version string is what stamps `benchVersion` into
  every run record (plan §3.4), so a skew between `pyproject.toml` and `__init__.py` fails here.
- Root `AGENTS.md` — a `model-bench/` bullet in **Structure** and a row in **Component docs**. The
  feature's requirements and plan stay at the repo root, where they were written (plan §4 S0).

**One defect found and fixed by reading the rendered output rather than the assertions:** when arm
B won, `verdict()` re-oriented the difference to the winner (`+66.7 pp`) but left the confidence
interval in A-minus-B orientation (`[-86.2, -29.9]`) — a positive effect printed beside a wholly
negative interval. Nothing raised; it is a plausible-looking, internally contradictory line, which
is the exact failure mode a measuring instrument must not have. The non-significant strings now
keep the signed A-minus-B difference for the same reason.

**Verification:** `model-bench/setup.sh` → venv created with Python 3.12.3, `model-bench[dev]`
installed (pytest 9.1.1, ruff 0.14.14), smoke import printed `model-bench 0.1.0`; re-run to confirm
idempotence. `.venv/bin/python -m pytest -q` from `model-bench/` → `1 passed in 0.01s`, exit 0
(0 failed / 0 skipped / 0 deselected). `.venv/bin/ruff check .` → `All checks passed!`.
`./run.sh --help` → the S0 guard's message, exit 1. Note that the test command must be run with
`model-bench/` as the working directory: the repo has no root pytest configuration, so from the repo
root pytest ignores this component's `testpaths` and walks the whole monorepo (measured: 9 collected,
8 collection errors, exit 2).
