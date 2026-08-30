# Natural-language query generation over structured graph data — Plan Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-055 (M6)

## Scope & verdict

Reviewed `docs/plans/workflow-nl-query-generation.md` (`architect`, `Version: 1.1`) together with
its companion `docs/plans/workflow-nl-query-generation-ml.md` (`data-scientist`), against
`docs/requirements/workflow-nl-query-generation.md` (FR-1..FR-5, AC-1..AC-5), as part of the
combined M6 four-document gate. Per the coordinator's brief, `docs/reviews/
workflow-nl-query-generation-security.md` (Pass 1: approve with suggestions + 2 MAJOR; Pass 2,
2026-08-27: approve, all four findings fixed, no new finding) is **not re-litigated** — I read it
for context on what the DSL's safety design guarantees and confirmed it is reflected consistently
in the plan (§3.1's `field_validator`s, `CompiledQuery` frozen dataclass, and `extra="forbid"` all
match what the security review's Pass 2 verified). My own scope is FR/AC coverage, internal
consistency, engineering soundness outside the security lens, and cross-document fit.

**Verdict: approve with suggestions.**

**CPG:** considered, not relevant — new-code design over the current tree; `cpg_falkorchat` is
stale (coordinator's brief), and the plan correctly reads `services.py`/`repository.py`/`tools.py`
and `claude/graph-dba/falkordb-quirks.md` directly. I additionally verified the `KNOWLEDGE_BASE_SCHEMA`
registry's `Entity`/`Document`/`Chunk` property claims directly against `docs/DESIGN.md` §5.1/§7.1
and `docs/plans/document-ingestion-graph.md` (both confirm `entityId/name/nameNormalized/type`,
`documentId/title/sourceFormat`, `chunkId/text/seq/documentId` exactly as the plan states) rather
than trusting the "shipped and stable" characterization on faith.

## Findings

### MINOR — `docs/BACKLOG.md`'s K-055 entry is stale: it describes the `security-expert` review as "in progress" when it is in fact complete and approved

**Evidence:** `docs/BACKLOG.md` lines 30-32 and 66-73: "a `security-expert` review of its
structural safety mechanism (in progress) before it is considered complete" / "a `security-expert`
review of the mechanism (FR-3/FR-3a adversarial test cases, in progress) before this item is
considered complete." But `docs/reviews/workflow-nl-query-generation-security.md`'s Pass 2 (dated
2026-08-27 — the same day as this review) reads: "**Verdict: approve.** All 4 findings from
Pass 1 are fixed as specified; no new finding... No blocker, no open MAJOR/MINOR." The review is
done, not in progress.

**Why it matters:** low-stakes but directly relevant to milestone-gate accuracy — the M6 milestone
table's own ✅-when condition ("golden-set/adversarial gates passed for K-055") reads as still
pending a security sign-off that has, in fact, already landed. A reader relying on `BACKLOG.md`
alone (its intended purpose, per its own header, is "status of an open milestone is authoritative
here") would underestimate how close K-055 is to done.

**Suggested improvement:** when `teco` next touches this entry, flip "in progress" to "delivered"
(or drop the parenthetical entirely, since the dependency is satisfied) for both occurrences. Not
blocking this plan gate — the plan itself correctly and currently reflects the security review's
outcome (its own revision note and §6 cite Pass 1's fixes and anticipate, correctly, "a
`security-expert` confirmation pass... before the `analyst` plan gate," which has now happened).

### NIT — the requirements doc's own "Related work" section is stale (already self-flagged, not re-raised as new)

The plan's §1 already identifies and correctly declines to block on this
(`docs/requirements/workflow-nl-query-generation.md`'s "active `teco` coordination in flight"
wording for `document-ingestion.md`, which closed 2026-08-25). Noting only that I independently
confirmed the plan's own claim is right (`docs/HISTORY.md` shows K-050/M5 closed 2026-08-25) —
this is not a new finding, just confirmation the plan's self-assessment holds.

## Cross-cutting checks (per the coordinator's brief)

- **AC-2 generality design vs. the ml note's corpus recommendation — no drift found, correctly
  operating at different altitudes.** The ml note (`workflow-nl-query-generation-ml.md` §4)
  recommends a **fresh, purpose-built 10-15-document ingestion pass** for the golden-set corpus
  that gates AC-4 — explicitly rejecting `ws:acme`'s thin QA fixture data as "too thin, too
  type-homogeneous." The architect plan's §3.5 AC-2 verification, by contrast, only needs "seed at
  least one document via the existing `ingest_document` MCP/REST path" — but AC-2 (per the
  requirements doc's own wording) is a **live functional proof** that the mechanism works against
  a second schema, a materially lighter bar than AC-4's golden-set accuracy gate. The plan
  correctly defers the golden-set harness entirely to the ml note (§4 step 5-6: "Wait for
  `data-scientist`'s note... this plan's steps 1-4 do not depend on it... step 6 does") rather than
  substituting its own lighter AC-2 seed for the ml note's fuller corpus requirement. I checked for
  the specific risk the brief flagged — the plan quietly reusing the thin fixture data for the
  golden set too — and did not find it: the plan simply doesn't touch golden-set corpus
  construction at all, leaving it entirely to the note that owns it. No inconsistency.
- **`KNOWLEDGE_BASE_SCHEMA`'s claimed shipped schema — verified accurate**, not just plausible (see
  CPG line above). `DATASET_REGISTRY`'s `Entity`/`Document`/`Chunk` property sets match the actual
  live schema exactly.
- **FR-8-style "exact computation" mechanism — correctly treated as orthogonal**, not reused or
  reinvented. §3.1's own text explicitly distinguishes FR-3's structural non-mutation property from
  `workflow-cart-and-totals.md`'s determinism-of-arithmetic property, and I agree these are
  genuinely different axes that do not call for the same mechanism — no drift.
- **BACKLOG dependency chain** (K-052 → K-055; ml note "delivered," security review "in progress")
  — the sequencing itself is correctly stated (K-055 depends on K-052 landing first for the shared
  scaffold), but the security-review status is stale per the MINOR finding above.

## What's solid

- The DSL-over-free-form-Cypher decision (§2.3) is the right engineering call for this codebase's
  actual constraints (no Cypher-grammar dependency, FalkorDB's OpenCypher subset not matching any
  off-the-shelf grammar) — and, per the security review, has now been independently verified to
  hold under live adversarial pressure, not just argued from principle.
- The two-layer safety argument (inexpressible-mutation DSL + engine-enforced `GRAPH.RO_QUERY`
  refusal) is well-motivated and, unusually for a plan-gate review, has already been through a
  full independent security pass with live reproduction on the pinned build — a materially higher
  bar of verification than most plans reach before implementation starts.
- The declarative `DatasetSchema` registry design (§3.3) correctly satisfies FR-2's actual
  generality bar ("registering a new dataset is data, not code") without overreaching into live
  schema introspection the codebase hasn't verified is reliable here.
- The plan is transparent about its own residual risk surface (`run_readonly_query`'s generality,
  §6) rather than hiding it, and the ml note is equally transparent about its own biggest risk
  (the Layer 1 harness seam not being guaranteed to exist yet) — both documents model the kind of
  self-disclosure a plan-gate review wants to see.
- v1's explicit non-goal of relationship-pattern traversal (§3.6) is a genuine, stated scope
  reduction that keeps the compiler's auditable surface small — correctly flagged as a v2
  extension point, not silently under-built.

## Open questions

None that block this plan gate. The two "open" items the security review itself carries forward
(Group A live-vs-stubbed model choice; the declined private-constructor ceremony) are
implementation-time calls, not outstanding review asks, and are correctly not re-litigated here.
