# Generic Cypher MCP — plan-gate review (U1 + U2)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M5 proposed)

## Scope & verdict

Combined plan-gate pass (coordination unit U3 of
[`docs/plans/generic-cypher-mcp-coordination.md`](../plans/generic-cypher-mcp-coordination.md))
over both design notes against
[`docs/requirements/generic-cypher-mcp.md`](../requirements/generic-cypher-mcp.md) (FR-1…FR-11,
AC-1…AC-8, Out of scope, Decision log — confirmed the requirements doc reflects "round 2," the
working-memory model, not the superseded "round 1" mirror/index model):

- [`docs/plans/generic-cypher-mcp-graph.md`](../plans/generic-cypher-mcp-graph.md) — `graph-dba`'s
  U1 data-model note.
- [`docs/plans/generic-cypher-mcp.md`](../plans/generic-cypher-mcp.md) — `architect`'s U2
  tool-mechanism note.

Grounding reads performed directly against the real artifacts, not taken on the plans' word:
`cpg/mcp/server.py` (full, 507 lines), `cpg/mcp/README.md` (full), `claude/graph-dba/kaizen/
inbox.md`, `docs/requirements/cpg-query-access.md`, `docs/requirements/joern-cpg-pipeline.md`
(header only, to check its `Status:`), `docs/BACKLOG.md` M3/M4 sections, `skills/agent-maintenance/
SKILL.md` §5, `claude/AGENTS.md`, `claude/graph-dba/graph-dba.md`, `claude/cobb/cobb.md`,
`claude/README.md`, and both agents' frontmatter (to check the "no `tools:` allowlist" claim).
This is a design-level review — no code exists yet to diff against.

**CPG:** not applicable — the review's subject is ~1,200 combined lines of markdown design docs
plus `cpg/mcp/server.py` (507 lines, read directly and in full); no call-graph/impact-analysis
question was in scope, and no Joern CPG covers the `cpg` component's own MCP-server code in any
case (confirmed live: `mcp__cpg__query` currently reports FalkorDB unreachable in this
environment, consistent with U2's own "considered, not relevant" framing for the same reason).

**Verdict: needs changes.** One blocker (a concrete, load-bearing gap in the implementation step
table that would leave the feature's core promise silently unmet) and three majors in the FR-8
enforcement design (§3.2/§3.1 of U2) — one of them a direct, evidence-backed answer to the
open question U2 §9 explicitly poses to this gate. Everything else — the schema (U1), the
one-tool decision, the ordering constraint, the container/build claims, the AC-8 header-only edit
— holds up under verification.

---

## Findings

### Blocker

**B1 — Step 4's file list omits both agents' own *operative* prompts, which currently instruct
exactly the behavior FR-2/FR-9 supersede.** U2 §7's step 4 (owner `cobb`) lists `claude/AGENTS.md`,
`skills/agent-maintenance/SKILL.md` (§5), and `docs/BACKLOG.md` as the files FR-11/AC-7 require
updating. Two files that describe — and actively *direct* — the standing convention are missing
from that list, and neither U1 nor U2 claims ownership of them:

- `claude/graph-dba/graph-dba.md:76` — graph-dba's own always-loaded system prompt: *"Any...
  durable, non-obvious environment fact a run surfaces... is appended as a dated entry... to your
  learnings inbox at `$HOME/.claude/agents/graph-dba/kaizen/inbox.md` before finishing."* This is
  not descriptive prose about the convention — it is the literal instruction graph-dba reads and
  acts on every session. Left unedited, graph-dba will keep appending new learnings to the now-
  frozen `inbox.md` after this feature "ships," directly contradicting FR-2 ("`graph-dba` writes a
  new raw learning **directly into the graph**") and failing AC-4 on the very first post-delivery
  run, regardless of how correctly the tool itself is built.
- `claude/cobb/cobb.md:71` — cobb's own maintenance-duties section: *"Every agent (you included)
  appends run-time environment discoveries to its `kaizen/inbox.md`"* — a blanket claim that
  becomes false specifically for `graph-dba` once this feature lands, and it is cobb's own
  operative description of *how* to run distillation (FR-9), which for `graph-dba` now needs a
  graph read (`mcp__cpg__query(graph='kaizen_graph_dba', ...)`) instead of an `inbox.md` read.
- `claude/README.md:22-32` also describes the inbox→distill→history convention generically (no
  graph-dba carve-out) and is not in step 4's file list either.

Both `claude/graph-dba/graph-dba.md` and `claude/cobb/cobb.md` are writable by `cobb` (no `tools:`
allowlist restricts either agent — confirmed by reading both frontmatter blocks directly, matching
U2 §2's own claim), so this is a one-line fix, not a scoping problem. But as written, the step
table would close AC-7 against a search that never looked at the two files actually driving the
agents' behavior. **Fix:** add `claude/graph-dba/graph-dba.md`, `claude/cobb/cobb.md`, and
`claude/README.md` to step 4's file list; better, replace the fixed enumeration with a search-based
done-condition (e.g. `grep -rln 'kaizen/inbox\.md\|append.*inbox' claude/ skills/agent-maintenance/
SKILL.md` and confirm every hit is either non-`graph-dba`-specific or updated) so a future doc this
review didn't think to name doesn't repeat the gap.

### Major

**M1 — `_AUTHOR_LITERAL_RE` scans the whole raw Cypher string, not just the map-literal/`SET`
clause, so text embedded in free-text fields (`fact`/`evidence`/`context`) can be misread as an
authorship claim and cause a legitimate write to be wrongly rejected.** This is a direct answer to
U2 §9's explicit question ("is a static regex check... an acceptable substitute for 'the
enforcement logic'?") — from a different direction than the one U2 itself flags (the *aliasing*
evasion, which under-*rejects*). This is an over-*rejection* risk in the opposite direction, and
it is untested and unflagged anywhere in §9. Concretely: `evidence`/`context` routinely quote code,
config, or command output verbatim (see the real entries in `claude/graph-dba/kaizen/inbox.md` —
every current entry embeds file paths, error strings, and command output). A write whose `evidence`
text happens to contain a substring shaped like `author: '<value>'` or `.author = '<value>'` (a
JSON-style key/value pair, a quoted Cypher example, a discussion of this very schema — plausible
precisely because this pilot is `graph-dba` dogfooding its own new `author` property) gets picked
up by `_AUTHOR_LITERAL_RE.findall(cypher)` as an extra "claim." If that embedded text doesn't equal
the declared `agent`, `authorize_write()` rejects the *entire* write with "this write attributes an
entry to author '`<embedded text>`'... but the call declared agent='<real agent>'" — a confusing,
incorrect rejection of a write whose real `author:` property was correct all along. **Fix:** anchor
the regex to the actual `CREATE (...:KaizenEntry {...})` map-literal or scope matching to a
narrower, structurally-delimited span (e.g. only inside the outermost `{...}` immediately following
`:KaizenEntry`) instead of the full query text; add a unit test constructing this exact case
(an `author:` literal correctly matching `agent`, plus a decoy `author`-shaped substring inside
`evidence`/`context` that doesn't).

**M2 — the "Author" write shape as implemented is broader than FR-8 describes: it authorizes
`SET <var>.author = '<agent>'` against *any* `MATCH`-selected node, not just entries the caller is
creating.** FR-8's own wording: *"an agent creates new entries attributed to itself only."* Nothing
in `authorize_write()` distinguishes a `CREATE` (a genuinely new entry) from a `SET` that
reassigns an *existing* node's `author` property. A call like `MATCH (e:KaizenEntry
{entryId:'<not-mine>'}) SET e.author = 'graph-dba'` finds the literal `.author = 'graph-dba'`,
compares it to `agent='graph-dba'`, matches, and is allowed — reassigning authorship of an entry
the caller did not create, with no check that the matched node was ever the caller's to begin with.
This is a materially larger capability than "create your own entries," and it sits outside the
curator carve-out's own narrow-by-design guardrail (§3.2 goes out of its way to restrict
curator-clear to one exact skeleton; the author shape gets no equivalent restriction against
`SET`). Single-author pilot scale makes this currently inert (there's no second author's entry to
hijack yet), but the design is what ships, and FR-10 explicitly anticipates a second (human)
author arriving with *no redesign* — meaning this gap activates the moment FR-10 is exercised, with
no additional review gate in between. **Fix:** restrict the "author" shape's regex to match only
inside a `CREATE (...:KaizenEntry {...})` clause (never `SET`), or, if `SET`-based self-correction
needs to stay reachable, require the query to also match on `WHERE e.author = '<agent>'` (or
equivalent) so a caller can only touch nodes it already owns.

**M3 — the "empty key" write-detection branch (U2 §3.1) cannot distinguish a genuine write from a
genuine read once `agent` is supplied, because "empty key" fires for *any* query — read or write —
against a nonexistent graph key.** The branch's own comment claims `agent` is "the caller's explicit
signal of write intent" and "a plain read against a mistyped graph name never sets it" — but
nothing in the tool's signature, description, or the frozen-inbox recipe (§3.3) actively prevents a
caller from supplying `agent` on a read out of habit (plausible for `graph-dba`, which "owns" this
graph and might reasonably self-identify on every call to it). If that happens against a
not-yet-created or mistyped graph name, a plain `MATCH (e:KaizenEntry) RETURN ...` gets routed into
`authorize_write()`, finds no `author:` literal and no curator-clear match, and returns "this write
is neither an author-write... nor the recognized curator-clear shape" — a confusing, actively wrong
message for a call that was never a write and whose real problem is a missing/mistyped graph name.
Unit test 9 (§8.1) only covers the no-`agent` case; there is no test for `agent` set + a
non-write query + "empty key." **Fix:** classify write-vs-read from the Cypher text itself (a
lightweight `CREATE|SET|MERGE|DELETE` keyword scan) before invoking `authorize_write()` in the
"empty key" branch, falling through to `graph_not_found_message()` when the text isn't
recognizably a write, regardless of whether `agent` was supplied.

### Minor

**m1 — the `Superseded by:` header-pointer field on `cpg-query-access.md` (U2 §5) is used
cross-topic-slug, a use root `AGENTS.md`'s doc-lifecycle rule doesn't explicitly cover.** The
`Supersedes:`/`Superseded by:` field pair is introduced in that rule block specifically for "a
second document of the *same* kind and topic" (ordinal successors within one slug family); here it
points from `cpg-query-access` (a different topic slug) to `generic-cypher-mcp`. The general
"a header pointer is metadata, not an amendment — the one edit permitted on an archived document"
clause plausibly licenses *some* header note, and I don't think this is wrong, but reusing the
reserved successor-pointer field name for a cross-topic reference is a slight stretch of a
mechanism built for same-slug evolution. Not a blocker (the underlying pointer is correct and the
relative link target — `./generic-cypher-mcp.md`, resolving to the *requirements* doc from
`docs/requirements/`, not the architect's plan of the same basename in `docs/plans/` — is verified
correct). Consider confirming this generic cross-topic use is intended, or using plain header prose
instead of the `Superseded by:` field to avoid conflating it with the same-family successor case.

**m2 — test list (§8.1) has no case for M1/M2's failure modes.** Once M1/M2 are addressed, add: a
write with a matching `author:` literal *and* a decoy `author`-shaped substring in `evidence`/
`context` (should succeed); a `SET .author = '<agent>'` against a node the caller doesn't own
(should be rejected once M2's fix lands).

---

## What's solid

- **U1's schema design** (`:KaizenEntry`, `author` as a plain property, `entryId` as the
  clear-by-id anchor, index-before-constraint ordering) is well-reasoned, correctly cites the
  repo's property-naming convention (`falkor-chat/docs/DESIGN.md` §3.1), and is honest about what's
  unverified (no `randomUUID()`-shaped Cypher function asserted; footprint is an estimate, labeled
  as such).
- **The append-before-delete ordering rationale (U1 §3, carried into U2 §3.5)** is sound: the
  fail-safe direction (crash → duplicate, never loss) is the only ordering that matches the
  stated top-priority constraint (no permanent loss), and "procedural, documented in `cobb`'s
  workflow" is an acceptable resolution given the tool genuinely cannot span two independent tool
  calls in one transaction.
- **The one-tool decision (U2 §3.1) is correctly grounded against the BACKLOG.md M3 precedent** —
  verified directly: BACKLOG.md's M3 section confirms "7 tools including `delete_graph`" and the
  exact reversal-trigger wording U2 quotes. The plan's read of why Shape A isn't a return to the
  rejected shape is accurate.
- **Container/build claims (U2 §4) verified against the real `cpg/mcp/README.md`**: the content-hash
  image tag, the self-healing rebuild-on-miss, the "no `.mcp.json` change needed" claim, and the
  "neither `graph-dba` nor `cobb` carries a restrictive `tools:` allowlist" claim (checked both
  frontmatter files directly) all hold.
- **The `explain_error()`/`_is_missing_graph()` substring reuse (U2 §3.1) is accurate**: `server.py`
  really does key off `"ro_query"` and `"empty key"` exactly as described (lines 391–393, 367–368),
  and `claude/graph-dba/kaizen/inbox.md`'s 2026-08-16 entry really does say what U2 claims it says
  about the live-verified distinction.
- **The AC-8 header-only edit (U2 §5) is correctly reasoned**: `cpg-query-access.md`'s header is
  genuinely `Status: archived` (verified), and `joern-cpg-pipeline.md` — the document FR-1's
  "mirror" instruction points at — is genuinely `Status: active` (verified), which is exactly why
  U2's divergence (header-only, not a body rewrite) is the correct, convention-compliant substitute
  rather than a shortcut.
- **Honesty about unverified claims is consistent**, not confined to the one spot it's labeled:
  the `falkordb-py` mutation-counter names, the Cypher UUID function, and the footprint estimate
  are all flagged as unverified in both U1 and U2 rather than asserted as fact.
- **The curator-clear carve-out cannot be abused to bypass author-attribution checks** — verified
  by tracing `_CURATOR_CLEAR_RE`: the recognized skeleton has no `author:`/`.author =` token at all,
  so it never enters the author-claim branch, and it only ever deletes (nothing to reassign).

## Open questions

- U2 §9 already asks whether the regex-based check is an acceptable substitute for "the enforcement
  logic" — this review's answer is: **not as currently scoped** (M1/M2/M3 above are the concrete
  gaps), but the *shape* (a static, pre-execution text check, no parser) remains right-sized for
  FR-8's stated trust bar once those three gaps are closed — a full Cypher parser is not warranted
  by anything in the requirements doc.
- Should `claude/graph-dba/graph-dba.md`'s (or a future second author's) system prompt also record
  the ordering constraint (§3.5) as an explicit reminder, or is documenting it solely in
  `agent-maintenance` §5 (cobb's side) sufficient given `graph-dba` never runs the delete step
  itself? Not resolved here — routing question for `teco`/`architect`, not a defect either way.

---

## Pass 2 — 2026-08-17

Re-review of `docs/plans/generic-cypher-mcp.md` **Version 1.1**, `architect`'s revision in
response to Pass 1 (`needs changes`). Re-read the patched plan in full (927 lines). U1
(`docs/plans/generic-cypher-mcp-graph.md`) is unchanged since Pass 1 (confirmed: the coordination
ledger and §10's own revision note both say so) and is not re-reviewed here.

**Verification method for this pass:** every Pass-1 finding was checked by re-reading the specific
section that claims to fix it, and — for the four findings with new Python logic (M1/M2/M3) —
additionally verified by **transcribing the plan's own `_string_literal_spans` /
`_kaizen_entry_create_map_spans` / `_author_claims` / `authorize_write` / `_looks_like_write`
functions verbatim into a scratch script and executing them** against the plan's own new test
cases (14, 15, 11, 8) plus adversarial inputs of my own construction — not just read for
plausibility. Script and full output available in this run's scratchpad
(`/tmp/claude-1000/.../scratchpad/verify_authorize.py`) if needed to reproduce.

### B1 (blocker) — **closed**, with one new minor

§7 step 4 is now split into 4a (`claude/AGENTS.md`, `claude/README.md`, `docs/BACKLOG.md`) and 4b
(`claude/graph-dba/graph-dba.md`, `claude/cobb/cobb.md`, `skills/agent-maintenance/SKILL.md`).
Re-read both operative-prompt files directly: `claude/graph-dba/graph-dba.md:74-76` is indeed a
section literally titled `## Learning capture` (confirmed by grepping headings), containing the
exact sentence the plan's done-condition targets — so 4b's file list and its done-condition
wording are grounded, not just plausible-sounding. `claude/cobb/cobb.md:71` is likewise the exact
line quoted. Both files are correctly identified as the two the original step 4 missed.

The close-out mechanism is now a `grep -rln 'kaizen/inbox\.md\|append.*inbox' claude/
skills/agent-maintenance/SKILL.md` before/after sweep rather than a fixed list — this is a genuine
search-based done-condition, not a relocated fixed-list risk: I ran the exact command myself and it
returns **35 files**, not the 5 named in the plan's own text. Every one of the five 4a/4b/step-3
target files is in that hit set (confirmed), so the mechanism does catch what it needs to catch.

**New minor (m3) — the plan's framing ("the five named files above are the currently-known hits")
understates the sweep's actual noise.** In practice an implementer running this close-out check
gets 35 hits, not ~5, and has to triage ~30 files that are legitimately not `graph-dba`-specific
(every other agent's own `kaizen/inbox.md` convention, hook scripts, other agents' `kaizen/
history.md` entries that happen to contain "append" near "inbox"). This isn't a correctness
problem — the mechanism is sound and *more* thorough than the fixed list it replaced, which is the
point — but the "currently-known hits" phrasing could lead an implementer to expect a short list
and be caught off guard by the actual triage cost. Recommend the step 4a/4b done-condition note
this explicitly ("expect on the order of 30 hits on the current tree; most are correctly
non-`graph-dba`-specific and need no edit") so the volume isn't a surprise mid-implementation.

### M1 (major, free-text false-positive) — **partially closed; one new major (M1-residual)**

The redesign (`_kaizen_entry_create_map_spans` + `_string_literal_spans` + `_author_claims`,
§3.2) correctly closes the *specific* reproduction from Pass 1 and the plan's own new test 14: a
**bare** `author:`-shaped substring sitting inside a free-text field (no surrounding `CREATE (...
:KaizenEntry {`) is now excluded, because it's found within a `_string_literal_spans` range of the
enclosing property's own string. Executed directly:

```
TEST14 claims: ['graph-dba']     # only the real literal — the decoy is correctly excluded
TEST14 result: None              # allowed
```

**But the fix has a gap the plan doesn't disclose: `_kaizen_entry_create_map_spans`'s *location*
step (`re.finditer(r"\bCREATE\b", cypher, re.IGNORECASE)`) scans the raw, full query text for
candidate `CREATE` keywords — it is not itself string-literal-aware.** Only the *body-extraction*
step (once a candidate is found) tracks string literals. So a free-text field whose content happens
to look like a **complete** `CREATE (<var>:KaizenEntry {...})` clause — not just a bare
`author:`-shaped fragment — gets misread as a second, independent top-level clause, because its
embedded `CREATE` keyword is a real regex match against the whole text, and the text immediately
following it inside the string genuinely matches `\s*\(\s*[a-zA-Z_]\w*\s*:\s*KaizenEntry\s*\{`.
Executed directly against a realistic instance of exactly the pattern the plan's own M1 rationale
invites ("a `graph-dba` kaizen entry documenting this very schema... quoting a full recipe
example"):

```python
cy = ("CREATE (real:KaizenEntry {fact: 'x', "
      "evidence: \"example: CREATE (k:KaizenEntry {author: 'evil'})\", "
      "context: 'c', suggestedHome: 'unsure', author: 'graph-dba', createdAt: 't'})")
_kaizen_entry_create_map_spans(cy)
# → ['fact: \'x\', evidence: "example: CREATE (k:KaizenEntry {author: \'evil\'})", ...author: \'graph-dba\'...',
#    "author: 'evil'"]                                          ^^^^^^^^^^^^^^^^^ spurious second span
authorize_write(cy, "graph-dba")
# → "REJECTED_MISMATCH claims=['graph-dba', 'evil'] mismatched=['evil']"
```

A legitimate write, whose real top-level `author: 'graph-dba'` is correct, is rejected — the exact
failure mode M1 was supposed to close, just requiring a slightly richer decoy (a full clause shape,
not a bare fragment) than the one Pass 1 originally reproduced and test 14 now pins. This is
plausible **by accident**, not contrivance: it's precisely the shape of text a `graph-dba` kaizen
entry about *this migration or this schema* would contain (§3.2's `evidence` examples throughout
this very document are full `CREATE (...:KaizenEntry {...})` snippets), and the real `inbox.md`
entries already quote command/config text verbatim as a matter of course.

**A second, more serious variant of the same root cause: under-*enforcement*, not just
over-rejection.** If the *real* top-level `CREATE` clause omits its own `author:` property
entirely (a malformed/careless write — no recipe in this plan does this, but nothing rejects it
structurally either) while a free-text field embeds a decoy `CREATE (...:KaizenEntry {author:
'<declared-agent>'})`-shaped substring, the decoy's claim satisfies `authorize_write()` and the
write is **allowed** — creating a real `KaizenEntry` node with no genuine `author` property at all,
while the enforcement layer believes it verified attribution:

```python
cy_bypass = ("CREATE (real:KaizenEntry {fact: 'x', "
             "evidence: \"example: CREATE (k:KaizenEntry {author: 'graph-dba'})\", "
             "context: 'c', suggestedHome: 'unsure', createdAt: 't'})")  # no author: at top level
authorize_write(cy_bypass, "graph-dba")
# → None  (AUTHORIZED — target.query() would run, materializing an un-attributed node)
```

This sub-case requires the caller to omit the real `author:` literal from the actual top-level map
— a deviation from every recipe this plan hands to `graph-dba`/`cobb` (§3.2, §3.4, §3.5 all embed
the literal directly) — so I judge it closer to the *already-accepted* aliasing-evasion risk (FR-8's
stated "not hardened against malicious caller" bar) than to an accidental trap. I'm not treating it
as blocking on its own, but it shares one root cause and one fix with the over-rejection case above,
so closing one closes both — no reason to leave it open once the fix is in.

**Verified fix, one small change:** compute `_string_literal_spans` once over the **whole** `cypher`
text before locating `CREATE` candidates, and skip any `\bCREATE\b` match whose start position falls
inside one of those spans:

```python
def _kaizen_entry_create_map_spans(cypher: str) -> list[str]:
    outer_spans = _string_literal_spans(cypher)          # NEW: whole-text pass
    spans = []
    for cm in re.finditer(r"\bCREATE\b", cypher, re.IGNORECASE):
        if any(s <= cm.start() < e for s, e in outer_spans):   # NEW: skip embedded decoys
            continue
        ...  # unchanged from here down
```

Re-ran both adversarial cases plus the plan's own tests 14/8 against this one-line-larger version:

```
NESTED (fixed) claims: ['graph-dba']   # decoy correctly excluded — no false rejection
BYPASS (fixed) claims: []              # decoy no longer authorizes — falls to "unrecognized", rejected
TEST14 (fixed) claims: ['graph-dba']   # unaffected — still correct
MIGRATION (fixed) claims: ['graph-dba']  # unaffected — still correct
```

No regression against any case the plan already relies on. **Severity: Major, not blocker** — the
over-rejection direction fails safe (no data written, just a confusing message and a retry), and
the under-enforcement direction requires an off-recipe write shape no prescribed workflow produces.
But it's a real, executable gap in a mechanism the plan explicitly claims is now "fixed," so I'm not
closing M1 outright. **Suggested action:** land the one-method patch above (verified, shown in
full) and add a 16th unit test reproducing the nested-decoy case (both the over-rejection and the
under-enforcement variant, mirroring the two snippets above).

### M2 (major, SET-based reassignment) — **closed, verified**

Executed the plan's own reasoning and its exact reproduction case:

```python
authorize_write("MATCH (e:KaizenEntry {entryId:'not-mine'}) SET e.author = 'graph-dba'", "graph-dba")
# _kaizen_entry_create_map_spans(...) → []   (SET produces no CREATE-anchored spans, structurally)
# → "REJECTED_UNRECOGNIZED"
```

Confirms the plan's claim precisely: `_kaizen_entry_create_map_spans` only ever anchors on
`CREATE`, so a `SET`-based reassignment — even with a value that matches `agent` exactly — can never
register a claim and is unconditionally rejected. Test 15 (§8.1) reproduces this exact case. The
plan's stated reason for not adding a `WHERE`-ownership guard instead (no requirement asks for
self-correcting `SET`, and removing the path is simpler/safer than adding and re-verifying a second
guard) is sound engineering judgment, not a shortcut.

### M3 (major, empty-key/agent-implies-write) — **closed, verified**

`_looks_like_write()` (`_WRITE_KEYWORD_RE = r"\b(CREATE|MERGE|SET|DELETE|REMOVE)\b"`) is a
word-boundary-correct keyword scan (checked: `SET` does not false-match inside `OFFSET`/`RESET`,
`DELETE` does not false-match inside `deleted`, both via the `\b` anchors) run before
`authorize_write()` on the "empty key" branch. A plain `MATCH (e:KaizenEntry) RETURN e.fact` with
`agent` set now correctly routes to `graph_not_found_message()`, not into enforcement — test 11
pins exactly this. §9's own honest caveat (a read whose free text quotes a write keyword, e.g.
`WHERE e.context CONTAINS 'DETACH DELETE'`, could still route into enforcement and get a less-
precise message) is judged **acceptable as stated**: it only ever degrades *message accuracy*
("not a recognized write shape" instead of "graph not found"), never authorization outcome — since
`authorize_write()` itself is the string-literal-/`CREATE`-span-aware check that would then
correctly reject it anyway — and it only matters on a path (missing/mistyped graph name) that stops
existing forever once the one-time migration lands. No further closing needed here; going further
(a string-literal-aware pre-classifier) would be effort spent on message wording, not correctness.

### Minors m1/m2 — **closed**

- **m1**: §5's edit now uses plain `**Note:**` prose, confirmed by re-reading the new header block
  — the `Supersedes:`/`Superseded by:` field pair is gone, replaced with an unreserved bolded line
  carrying the identical pointer and rationale for the change spelled out explicitly (§5, "Plain
  prose, not the field pair"). Closes the concern as stated.
- **m2**: test 14 covers the bare-substring case exactly as asked. Test 15 covers the `SET`
  reassignment case exactly as asked. (Test 14 does *not* cover the deeper nested-`CREATE`-decoy
  case — that gap is folded into M1-residual above, with its own suggested new test.)

### §3.4 migration Cypher shape — **verified correct**

The revised shape (one shared `author: 'graph-dba'` literal in the `CREATE` clause itself, not
per-row inside the `UNWIND` list) is a necessary and correct consequence of the M1/M2 fix — with
the *old* per-row shape, the per-row maps inside `UNWIND [...]` are plain list literals, not
`CREATE (...:KaizenEntry {...})` bodies, so `_kaizen_entry_create_map_spans` would never have found
any of the six per-row `author:` literals under the new scan (it only ever looks inside a real
`CREATE`'s own map body), rejecting the whole migration. The revised single-literal-at-`CREATE`
shape sidesteps this correctly. Executed test 8's shape directly: `_author_claims(...) →
['graph-dba']`, `authorize_write(...) → None` (allowed). Confirmed correct, no regression.

### Resolved open question (§3.5, ordering constraint not duplicated into `graph-dba.md`) — **accepted**

The conclusion (document the ordering solely in `agent-maintenance` §5, not in `graph-dba.md`) is
right: re-reading §3.5's four-step sequence, all four steps belong to `cobb`'s distillation
workflow — `graph-dba`'s own raw-capture write (FR-2) is a separate, earlier, unordered act that
merely supplies the *input* to this sequence, not a participant in the ordering constraint itself.
One wording nit, not an action item: the plan's phrasing ("`graph-dba`... only ever runs the append
half indirectly by way of `cobb`'s workflow") is a little confusing, since `graph-dba` doesn't run
any half of *this* sequence at all — its involvement is upstream, not "indirect participation" in
it. Doesn't change the correctness of the decision.

### General sanity pass over the new `authorize_write()` machinery (beyond the four original findings)

Checked, and found sound:

- **`KaizenEntry` label matching is deliberately case-sensitive** (`_kaizen_entry_create_map_spans`'s
  inner `re.match` carries no `re.IGNORECASE`) while the `CREATE` keyword match does — correct, not
  an inconsistency: Cypher keywords are case-insensitive in FalkorDB, labels are not.
- **Brace-depth matching correctly stops at the `KaizenEntry` map's own closing `}`**, verified it
  does not run on into a subsequent relationship/node pattern (`-[:REL]->(a:Other {...})`) even
  though U1's schema has no relationships today, so this isn't a live risk, just confirmed robust.
- **Multi-line Cypher (FR-3) is handled correctly** — `\s*` in the map-open regex matches newlines,
  consistent with the tool's verbatim-multi-line contract.
- **A multi-label `CREATE (k:KaizenEntry:Other {...})` clause would silently produce zero spans**
  (the regex requires `KaizenEntry` immediately followed by `\s*\{`) and so would be rejected
  outright — not a vulnerability (fails closed), and moot under U1's single-label schema; noted only
  in case a future extension adds a second label without revisiting this regex.
- **`_string_literal_spans`'s backslash-escape handling** (`i += 2` on an escape) was checked against
  an escape landing exactly at the end of the text (`i == n-1`) — loop condition `i < n` correctly
  stops without indexing past the end; no crash, just an unterminated span silently dropped, which
  is the same "let FalkorDB's own parser catch a malformed literal" posture the rest of the tool
  already takes.

No further issues found beyond M1-residual.

### Pass 2 verdict

**Needs changes** — one Major (M1-residual) remains genuinely open, verified by direct execution
against the plan's own proposed code, with a concrete, already-verified one-method fix supplied
above (plus a 16th test suggestion) so this should be closeable in a small, targeted Pass 3 rather
than a broader rework. B1, M2, M3, m1, and m2 are all confirmed closed by direct verification (not
just re-reading the plan's own claims) — code executed, files re-read, headings/lines re-grepped.
One new minor (m3, sweep-noise framing) is opportunistic and does not block on its own.

**Summary for the next pass:** land the whole-text `_string_literal_spans` pre-filter in
`_kaizen_entry_create_map_spans` (shown above, verified against all of the plan's existing test
cases plus two new adversarial ones with no regression), add the corresponding 16th unit test, and
optionally soften the "five named files" / "currently-known hits" phrasing in §7's close-out
condition to set the right expectation for the sweep's actual ~35-file volume. Everything else in
Version 1.1 holds.

---

## Pass 3 — 2026-08-17

Final, focused re-verification of `docs/plans/generic-cypher-mcp.md` **Version 1.2** — scoped to
the two items Pass 2 left open (M1-residual, m3) plus a drift check over everything else, per the
coordinator's brief. Full document re-read (906 lines) to confirm scope; deep verification
concentrated on §3.2, the new "M1-residual, explicitly" paragraph, test 16 (§8.1), and §7's
reworded close-out note.

### M1-residual — **closed, verified by direct execution against the exact patched code**

Transcribed `_string_literal_spans()`, `_kaizen_entry_create_map_spans()`, `_author_claims()`, and
`authorize_write()` **verbatim from the current file (lines 243-353)** into a fresh scratch script —
not diffed by eye, executed — and re-ran both of Pass 2's adversarial cases plus every prior
regression case (tests 3, 4, 5, 6, 7, 8, 11, 14, 15) against this exact patched version:

```
test16_over_rejection_claims:                          ['graph-dba']
test16_over_rejection_result(agent=graph-dba):          None            # allowed — correct
test16_under_enforcement_claims:                        []
test16_under_enforcement_result(agent=graph-dba):       REJECTED_UNRECOGNIZED   # correct, bypass closed
test14_result / test15_result / test8_result:           unchanged, still correct
test3_result / test4_result / test5/6/7_result:         unchanged, still correct
```

Both adversarial cases from Pass 2 now land on the correct outcome, and none of the eight prior
regression cases moved. The patched function (lines 262-302) is **structurally identical** to the
fix I supplied and execution-verified in Pass 2 — confirmed by re-running it, not just visual diff:
`outer_spans = _string_literal_spans(cypher)` computed once over the whole text before the
`re.finditer(r"\bCREATE\b", ...)` loop, with `if any(s <= cm.start() < e for s, e in outer_spans):
continue` skipping any `CREATE` keyword whose start falls inside a pre-existing string literal —
exactly the mechanism that closes both the over-rejection and the under-enforcement variants. The
new docstring (lines 268-274) accurately describes what the code does. **M1-residual is closed.**

### Test 16 (§8.1) — **confirmed to genuinely reproduce both adversarial cases and pass**

Both sub-cases (lines 745-759) are the exact Cypher strings from my Pass-2 finding, copied
verbatim, and the expected outcomes stated in the plan (`allowed` for the over-rejection case,
`rejected` for the under-enforcement case) match what the executed code actually returns — verified
directly above, not inferred from the plan's prose. This is a real regression pin, not a
restatement of the finding.

### m3 (sweep-volume framing) — **adequate**

§7's close-out note (lines 667-672) now reads *"Expect on the order of 30 hits on the current tree,
not a short list (Pass-2 review actually ran the sweep: 35 hits total) — most are legitimately
non-`graph-dba`-specific noise... and need no edit; only the five files this table already names
require a change. This is a real triage cost, not a rubber-stamp."* This is exactly the expectation-
setting I asked for, cites the real number I measured, and correctly separates "hits to triage" from
"files to actually edit." Closed, no further action.

### Drift check over the rest of the document

Read the full file end to end. Everything verified in Pass 2 (§1, §2, §3.1, §3.3–3.6, §4, §5, §6,
§7's step table proper, §8.2, §9's other bullets, §10) is textually unchanged except for the two
targeted edits above and bookkeeping (the version bump, §11's new revision note, the "tests 1–16"
count in §8.2). No new claims were introduced anywhere else that would need independent grounding —
confirmed by reading, not assumed from the coordinator's summary.

### Pass 3 verdict

**Approve.** All findings from Pass 1 (B1) and Pass 2 (M1-residual, m3) are closed and verified by
direct execution of the plan's own patched code, not by re-reading its claims. U1
(`docs/plans/generic-cypher-mcp-graph.md`) had zero findings across all three passes and remains
unchanged. This plan (Version 1.2) plus U1 are ready for implementation dispatch — no further
plan-gate review needed before `teco` sequences the step table (§7).

**CPG:** not applicable — unchanged from Pass 1/2 (no Joern CPG covers this component's own tooling;
this pass's verification was direct code execution, not call-graph analysis).

---

## Code re-gate (diff-scoped) — 2026-08-18

Second gate: the **implementation diff** against Version 1.2 of `docs/plans/generic-cypher-mcp.md`
(plan-gate verdict: approve, Pass 3 above), coordination unit U4 (steps 1+2, owner `coder`). Four
files, all uncommitted in the working tree: `cpg/mcp/server.py` (+333/-…), `cpg/mcp/tests/
test_server.py` (+298), `cpg/mcp/README.md` (+89), `docs/requirements/cpg-query-access.md` (+5,
header only). Scope is exactly steps 1+2 of the plan's §7 table — migration (step 3), doc sweep
(4a/4b), and the acceptance pass (step 5, `qa-engineer`) are separate, not-yet-dispatched units and
are out of scope here.

**Verification method.** Read `cpg/mcp/server.py` in full and diffed every new function
(`CURATOR_AGENTS`, `_WRITE_KEYWORD_RE`/`_looks_like_write`, `_AUTHOR_LITERAL_RE`,
`_CURATOR_CLEAR_RE`, `_string_literal_spans`, `_kaizen_entry_create_map_spans`, `_author_claims`,
`authorize_write`, `format_write_result`, the `run_query()` branch) line-by-line against the plan's
§3.1/§3.2 code blocks — transcribed essentially verbatim, comments included. Ran the offline suite
myself (`cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q`): **83 passed, 7 deselected**, matching the
self-report exactly. Confirmed the pinned `falkordb` 1.6.2 package is actually installed
(`.venv/lib/python3.12/site-packages/falkordb`, `pip show` confirms `Version: 1.6.2`) and grepped
`query_result.py` directly: all ten names in `_WRITE_STAT_ATTRS` are genuine `@property` members on
`QueryResult` — confirmed, not guessed. Measured `TOOL_DESCRIPTION`/`SERVER_INSTRUCTIONS` lengths
myself (835 / 1052 chars), matching the self-report. Built and ran the **actual in-container test
gate** myself, not just trusted the report: `docker images` already had `cpg-mcp:test-2dadf10c24b0`
built; independently recomputed the content-hash tag by sourcing `image-tag.sh` and calling
`cpg_mcp_image_tag` against the live uncommitted tree — it produced `2dadf10c24b0`, an exact match,
confirming the image genuinely reflects this diff, not a stale build. Ran both gates for real:
`docker run --rm cpg-mcp:test-2dadf10c24b0 python -m pytest tests -q` → **74 passed, 7 deselected**;
`docker run --rm --add-host=host.docker.internal:host-gateway cpg-mcp:test-2dadf10c24b0 python -m
pytest tests -q -m live` (against the real, running `falkordb-dev` container) → **7 passed, 74
deselected** — both exact matches to the self-report, and both confirm the offline/live split (90
host tests − 9 host-only `test_build_inputs.py` cases = 81 = 74+7, verified by
`pytest --collect-only`). `redis-cli GRAPH.LIST` post-run shows no `_cpg_mcp_selftest_*` residue and
confirms `kaizen_graph_dba` does not exist yet — correctly consistent with migration (step 3) not
having run.

Reproduced **both** required mutation-test cases myself (backed up `server.py` first, restored it
byte-identically after each — confirmed via `md5sum` before/after and a clean `git diff` at the end):

- **Mutation 2 (whole-text string-literal pre-filter removed from `_kaizen_entry_create_map_spans`,
  the M1-residual/Pass-3 fix)** — reproduced exactly as self-reported: only
  `test_nested_create_decoy_in_free_text_is_excluded` (test 16) fails, cleanly isolated, no
  collateral. Confirms the fix is real and load-bearing, and confirms the self-report's claim for
  this case.
- **Mutation 1 (`_author_claims()` reverted to `_AUTHOR_LITERAL_RE.findall(cypher)` on the raw
  whole text, dropping both CREATE-scoping and string-literal exclusion)** — reproduced tests 14 and
  16 failing, exactly as self-reported. **But test 15
  (`test_set_based_author_reassignment_is_always_rejected`) did NOT fail** — it passed under this
  mutation, contradicting the self-report's claim ("test 15 failed as expected, with collateral
  failures on 14/16"). Full run: `2 failed, 81 passed, 7 deselected` (14, 16 failed; 15 did not).
  Isolated re-run of test 15 alone against the mutated file confirms it independently: `1 passed`.
  See M-A below — this is a real, reproducible discrepancy, not a transcription slip in this
  review.

---

### Findings

#### Major

**M-A — Test 15 does not actually exercise M2's CREATE-vs-SET distinction, and the self-reported
mutation-test result for it does not reproduce.** `test_set_based_author_reassignment_is_always_
rejected` (`cpg/mcp/tests/test_server.py:750-760`) sends `MATCH (e:KaizenEntry {entryId:'not-mine'})
SET e.author = 'graph-dba'` and asserts it's rejected, with the docstring attributing this to
`_kaizen_entry_create_map_spans()` finding no `CREATE` clause. That's true, but **it isn't why this
specific input is rejected**: `_AUTHOR_LITERAL_RE` (`server.py:260`) is anchored on the map-key
colon spelling (`\bauthor\s*:\s*['"]...`) and never matches the dot-assignment spelling
(`.author = '...'`) at all — a fact the code's own comment states explicitly ("The map-KEY form
only — never `.author = ...` (SET)", `server.py:256-259`). So this test's input contains **no
substring the regex could ever match, CREATE-scoped or not** — it would pass identically whether
`_kaizen_entry_create_map_spans()` correctly restricts to `CREATE`, restricts to nothing at all, or
is deleted outright and replaced with `_AUTHOR_LITERAL_RE.findall(cypher)` on the raw text (verified
directly: under exactly that mutation, test 15 passes — see reproduction above). The test is real,
it asserts a true and desirable outcome, and FR-8's own "not hardened against a malicious caller"
bar means the shipped behavior is fine — but as a **regression pin for M2** specifically, it is
inert: a future change that widens `_kaizen_entry_create_map_spans()` to also recognize a `SET`
clause carrying a **colon-form** map-merge (`SET e += {author: 'graph-dba'}`, valid Cypher, and a
shape `_AUTHOR_LITERAL_RE` *would* match) would reintroduce exactly the capability M2 closed,
and nothing in the current suite would catch it — tests 14/16 guard the string-literal-exclusion
half of the mechanism (M1/M1-residual), not the CREATE-only-clause-type restriction (M2) at all.
Separately, and worth surfacing on its own: the self-report to `teco` states this exact mutation
("reverted `_author_claims()` to scan raw text instead of CREATE-map-scoped spans") caused "test 15
failed as expected" — I ran precisely that mutation and test 15 passed, both in the full-suite run
(`2 failed, 81 passed` — 14 and 16, not 15) and in an isolated single-test run. This doesn't point
to a shipped-code defect (the current, unmutated code is correct on every case checked, including
live in-container), but it does mean the "M2 is now regression-pinned" confidence carried forward
from three plan-gate passes and the coder's own mutation-testing exercise rests on a claim that
doesn't hold up under direct reproduction. **Suggested fix:** add a test using the colon/map-merge
SET shape (`MATCH (e:KaizenEntry {entryId:'not-mine'}) SET e += {author: 'graph-dba'}`) as the actual
mutation-killing regression pin for M2's CREATE-only restriction — it is rejected correctly by the
current code (verified: `_kaizen_entry_create_map_spans` never anchors off `SET`, so this produces
zero spans exactly like the existing test 15 case) but, unlike test 15, would fail if that
restriction were ever loosened. Keep test 15 as-is (it's a legitimate, faithful reproduction of the
review's original literal finding) — just don't rely on it as M2's mutation-kill test, and correct
the internal record of what the mutation-testing pass actually showed.

#### Minor

**m-A — An unrelated, pre-existing uncommitted change sits in the working tree alongside this
diff.** `claude/tico/kaizen/inbox.md` shows as modified (`git status`), but its content (a
2026-08-17 entry about introducing a new "security-expert" agent via a `claude/docs/requirements/`
interview) has nothing to do with `generic-cypher-mcp` — it's a different `tico` session's own
kaizen capture, evidently still sitting uncommitted from before this unit started. Not a defect of
this diff (none of the four files this unit owns caused it, and it isn't cited by or coupled to
anything reviewed here) — flagging only so it doesn't get swept into a future commit of this diff by
accident, or mistaken for scope creep by a later reader of `git status`.

### What's solid

- **The shipped code is a faithful, line-for-line implementation of Version 1.2's design** — every
  helper (`_string_literal_spans`, `_kaizen_entry_create_map_spans` with the whole-text pre-filter,
  `_author_claims`, `authorize_write`, `_looks_like_write`, `format_write_result`) matches the plan's
  code blocks, including the exact fixes for B1/M1/M2/M3/M1-residual verified across three plan-gate
  passes. No drift found between "approved design" and "shipped code."
- **The `falkordb-py` 1.6.2 attribute names are genuinely confirmed**, not guessed — verified myself
  by grepping the installed package's actual `query_result.py`: all ten `_WRITE_STAT_ATTRS` names are
  real `@property` members, closing the one explicitly-flagged unverified fact carried since the U2
  plan.
- **All quantitative self-report claims reproduce exactly**: offline suite (83/7), in-container
  offline (74/7) and live (7/74) gates — both actually run against the real image and real FalkorDB
  in this review, not re-derived from the report — `TOOL_DESCRIPTION`/`SERVER_INSTRUCTIONS` lengths,
  and the final image tag (`2dadf10c24b0`, independently recomputed from the live tree's content
  hash, not just read off `docker images`).
- **Mutation-test 2 (M1-residual guard) is real and correctly attributed** — reproduced independently
  with the exact isolated failure signature claimed.
- **Both test renames are legitimate, not weakenings.** `test_input_schema_has_two_required_params_
  and_one_optional_agent` correctly reflects the widened tool contract (required set unchanged,
  `agent` added as `string | null`, default `None` — verified against the actual MCP schema via the
  passing test). `test_live_write_without_agent_is_rejected_server_side`'s corrected message
  assertion is the necessary and correct consequence of the new write-detection design: the old
  behavior (any write → generic "read-only" message) no longer exists once `authorize_write()`
  intercepts every `ro_query`-rejected write and returns the more specific "no `agent` supplied"
  message — confirmed live, against real FalkorDB, in this review's own in-container run. Not
  papering over anything.
- **`cpg/mcp/README.md` and the `cpg-query-access.md` header edit are accurate against what the code
  actually does** — the parameter table, the two recipe examples in "Writing through this tool," the
  `CPG_MCP_CURATOR_AGENTS` env-var row, and the in-container gate's expected counts all match the
  live-verified behavior; the header note's relative link (`./generic-cypher-mcp.md`) resolves to the
  *requirements* doc from `docs/requirements/`, not the architect plan of the same basename in
  `docs/plans/` — confirmed by reading the rendered file.
- **Scope discipline holds**: `git status`/`git diff --stat` show exactly the four files this unit
  owns changed (plus the one unrelated pre-existing file noted in m-A); `docs/requirements/
  cpg-query-access.md`'s diff is header-only, body untouched, matching AC-8's constraint exactly;
  nothing from steps 3/4a/4b/5 (migration, doc sweep, acceptance pass) was touched.

### Open questions

- None requiring the user's input. M-A is actionable by `coder`/`tdd-engineer` directly (add the one
  suggested test) and doesn't block dispatch of the remaining steps.

### Verdict

**Approve with suggestions.** No blocker: the shipped `server.py` correctly implements FR-8/AC-6 in
every case checked, including live against real FalkorDB, and every quantitative self-report claim
(suite counts, attribute names, image tag, string lengths) reproduces exactly under independent
verification. One Major (M-A): a required-verification mutation-test claim does not reproduce as
reported, and the test it concerns (test 15) is not actually the regression pin for M2 it claims to
be — a real gap in test-suite trustworthiness, not a shipped-code defect, closeable with one small
added test. Recommend landing the suggested fix before or shortly after this unit closes, but it
does not block sequencing steps 3/4a/4b/5.

**CPG:** not applicable — no Joern CPG covers this component's own MCP-server tooling in this repo
(confirmed: `redis-cli GRAPH.LIST` lists `ws:test`, `cpg_falkorchat`, `reference`, `ws:qa-tico-
workflows-manual`, `ws:acme`, `cpg_salesperson`, `ws:eval` — no `cpg_cpg` or equivalent), consistent
with every prior pass on this feature. Verification here was direct code reading, offline/in-
container test execution, and two hand-run mutation-test reproductions — not call-graph analysis.

---

## Code re-gate (U6, diff-scoped) — 2026-08-18

Third gate: the **implementation diff** for coordination unit U6 (steps 4a+4b, owner `cobb`) —
repo-wide catalog/convention docs plus both agents' own operative prompts, retargeting the standing
kaizen-inbox convention for `graph-dba` from "append to `inbox.md`" to "write into `kaizen_graph_dba`
via `mcp__cpg__query`" per `docs/plans/generic-cypher-mcp.md` §7 (Version 1.2, plan-gate verdict:
approve, Pass 3). Nine files, all uncommitted in the working tree: `AGENTS.md` (root),
`claude/AGENTS.md`, `claude/README.md`, `docs/BACKLOG.md`, `claude/graph-dba/graph-dba.md`,
`claude/cobb/cobb.md`, `skills/agent-maintenance/SKILL.md`, `claude/graph-dba/kaizen/history.md`,
`claude/cobb/kaizen/history.md`. Scope is exactly steps 4a+4b of the plan's §7 table plus the small,
explicitly-authorized root-`AGENTS.md` follow-up (per the coordination doc's U6 row); U4 (steps 1+2,
already re-gated above), U5 (step 3, migration — teco-verified separately, not re-litigated here),
and U7 (step 5, acceptance pass) are out of scope.

**Verification method.** Read every changed file's full diff (`git diff` against `HEAD`) and the
surrounding unchanged context in each. Read `docs/plans/generic-cypher-mcp-graph.md` §3 and
`docs/plans/generic-cypher-mcp.md` §3.5/§7 in full to check the append-before-delete ordering
constraint's placement, and `claude/graph-dba/kaizen/inbox.md`'s frozen note (U5's actual output) to
ground the doc edits against real, already-migrated state. Independently re-ran the close-out grep
sweep (`grep -rln 'kaizen/inbox\.md\|append.*inbox' claude/ skills/agent-maintenance/SKILL.md`):
**36 files**, matching the self-report's count exactly. Rather than diff a stashed "before" tree
(disallowed by this review's constraints), reconstructed the "before" hit-set precisely via
`git show HEAD:<file>` for each of the 9 changed files that fall inside the sweep's path scope (7 of
the 9 — root `AGENTS.md` and `docs/BACKLOG.md` sit outside `claude/`/`skills/agent-maintenance/
SKILL.md` and were never in the sweep's grep path to begin with, matching the self-report's own
framing): all 7 matched the pattern in both the `HEAD` and working-tree versions, and `git diff
--stat` confirms no other file under `claude/` or `skills/agent-maintenance/SKILL.md` changed —
together this proves the before/after hit-*set* is identical without needing to touch the git tree.
Sampled 5 of the ~30 untouched hits spanning the claimed categories (`claude/analyst/analyst.md`,
`claude/architect/architect.md` — each agent's own generic inbox-convention sentence;
`claude/analyst/hooks/guard-review-doc-writes.sh` — a doc-scoped write guard's allowlist comment;
`claude/scripts/audit-team.sh:76-79` — the kaizen-triple existence check; `claude/architect/kaizen/
history.md`, `claude/devops/kaizen/history.md`, `claude/teco/teco.md` — narration/generic-convention
hits; `claude/docs/requirements/security-expert.md` — a hypothetical future agent's requirements
doc) and read each in context: every one is correctly non-`graph-dba`-specific, correctly left
untouched. Independently ran `bash claude/scripts/audit-team.sh`: **96 PASS, 2 FAIL**, both in
`falkor-chat/docs/test-reports/graphrag-eval-report.md` (username/home-path leaks) — confirmed via
`git status --porcelain` (clean) and `git log -1 --date=short` (`2026-08-16`) that this file is
untouched by and unrelated to this diff, matching the self-report exactly. Independently measured
the `agent-maintenance` `SKILL.md` frontmatter `description` length at both `HEAD` and in the working
tree via `re.search` over the raw text: **889 → 940 chars**, matching the self-report exactly, both
under the 1024-char cap. Cross-checked `docs/BACKLOG.md`'s new M5 milestone-map row and item text
against plan §6's specified text **character-for-character** (word-wrap differences aside) — exact
match — and against the M3/M4 precedent's actual git history (`git log --follow -- docs/BACKLOG.md`,
inspecting commit `50f9aaa`'s diff on `docs/BACKLOG.md` directly) to establish what "same style as
M3/M4" actually means in practice, not just by inspection of the current file.

---

### Findings

#### Major

**M-B — The M5 milestone-map row carries no status marker and every C-5xx item is left `🔵
proposed`, even though five of the six correspond to steps already delivered — contradicting this
repo's own established, git-history-verified convention for exactly this situation.** `docs/
BACKLOG.md:49` adds `| **M5 — Generic Cypher MCP** | ... | **C-501 → C-506** |` with **no emoji**
next to the milestone name, and `docs/BACKLOG.md`'s new "## M5" section body marks all six items
(`C-501`…`C-506`) `🔵`. But per the coordination doc
(`docs/plans/generic-cypher-mcp-coordination.md`), by the time this diff was authored: U4 (steps 1+2
→ C-501, C-502) is `delivered`/`accepted`; U5 (step 3 → C-503) is `delivered`; and this very unit,
U6 (steps 4a+4b → C-504, C-505), is what the diff itself delivers. Only U7 (step 5 → C-506,
`qa-engineer`'s acceptance pass) is genuinely still `queued`. This repo has already faced exactly
this situation once, for M4, and resolved it a specific way — verified directly from git history, not
inferred: commit `50f9aaa` (`impl(cpg-agent-adoption): U4b-1..5`) added M4's own section with **all
seven** `C-401`…`C-407` items marked `✅` in the *same commit* that landed their implementation, while
the **milestone-map row** carried `🟡` (not `✅`, not blank) with the row's own prose stating
"Implementation (C-401…C-407) complete; U5 (`analyst` re-gate) and U6 (`qa-engineer` acceptance pass)
still queued" — the legend at `docs/BACKLOG.md:9` defines exactly this: `🔵 proposed · 🟡 in-progress
· ✅ done · ⚪ deferred`. `docs/plans/generic-cypher-mcp.md` §6 itself instructs "status 🔵 proposed
**until each step closes**" (emphasis reflects the plan's own conditional) — and five of the six
steps have, by the coordination ledger's own account, already closed. Leaving every item at `🔵` is
therefore not a style choice within tolerance; it's the literal M4 precedent's opposite. The
practical consequence: a reader of `docs/BACKLOG.md` — which `teco` reads and which other docs cite
by K-ID, per root `AGENTS.md`'s own description of the file — sees "M5: nothing started" when in
fact five of six implementation steps are done and the code has been running against a live,
migrated graph since U5. Cobb's own `kaizen/history.md` entry for this unit shows the reasoning that
produced this: *"all status 🔵 proposed — this unit doesn't close the milestone, U7/step 5's
acceptance pass does"* — conflating **milestone**-level completion (correctly still open, correctly
un-`✅`) with **item**-level completion (each item's own step, which does close independently, per
the M4 precedent and per the plan's own "until each step closes" wording). **Suggested fix:** flip
`C-501`, `C-502`, `C-503`, `C-504`, `C-505` to `✅` (each with the date/unit that delivered it,
mirroring C-401–C-407's format exactly — e.g. "✅ 2026-08-18 (U6)"), set the M5 milestone-map row to
`🟡` with prose stating implementation complete / acceptance pass (U7/C-506) still queued — mirroring
M4's row verbatim in structure — and leave `C-506` alone at `🔵` since U7 hasn't dispatched.

#### Minor

**m-B — Neither `graph-dba`'s nor `cobb`'s own `kaizen/history.md` U6 entry mentions the root
`AGENTS.md` edit, even though it is a real, verified part of this diff and the coordination doc
explicitly names it as folded into this unit.** `git diff -- AGENTS.md` confirms the repo-root file's
`claude/` bullet was genuinely edited (the `graph-dba` carve-out sentence), and `docs/plans/
generic-cypher-mcp-coordination.md`'s U6 row explicitly calls this out: *"plus root `AGENTS.md`'s
`claude/` bullet (gap self-flagged by cobb, folded in as a small follow-up rather than a new unit)."*
But `claude/cobb/kaizen/history.md`'s new 2026-08-18 entry numbers six edited files/items (1.
`claude/AGENTS.md` … 6. `skills/agent-maintenance/SKILL.md`) and its closing "Docs touched:" list
repeats the same six — the repo-root `AGENTS.md` appears in neither. `claude/graph-dba/kaizen/
history.md`'s matching entry has the identical gap in its own "Docs touched (this unit, U6):" list.
The actual *content* of the root `AGENTS.md` edit is correct and consistent with the other two
convention docs (verified above) — this is purely an audit-trail completeness gap: a future reader
of either history file reconstructing "what changed in U6" from the record alone would miss one of
the nine files this diff actually touches. **Suggested fix:** add a seventh bullet (or a one-line
addendum) to cobb's `kaizen/history.md` entry naming the root `AGENTS.md` edit and its reason (mirrors
item 1's phrasing, one level up: the directory-level bullet in the repo-root catalog gained the same
carve-out), and add it to both entries' "Docs touched" lists.

### What's solid

- **The `graph-dba` carve-out is worded consistently and non-contradictorily across all three
  convention docs** (root `AGENTS.md`, `claude/AGENTS.md`, `claude/README.md`) — each states the same
  three facts (graph replaces `inbox.md` as the write target, `:KaizenEntry`/`mcp__cpg__query`
  attribution, `inbox.md` now frozen) at a register appropriate to its audience, and none leaves a
  dangling unconditional claim elsewhere in the same file (checked via full-file `grep` on `inbox`/
  `graph-dba` in `claude/README.md`; the two remaining generic mentions there — the kaizen-table row
  missing an inbox link, and the "Folder per agent" convention bullet — are both pre-existing/
  unchanged by this diff and the latter explicitly defers to the corrected Kaizen section).
- **`graph-dba.md`'s new Learning-capture instruction is technically correct against the shipped
  enforcement code**, not just plausible-looking prose: the example `CREATE (k:KaizenEntry {...,
  author: 'graph-dba', ...})` places a literal, top-level `author:` claim inside the `CREATE`'s own
  map body — exactly the shape `_author_claims()`/`authorize_write()` (verified in the U4 re-gate
  above) authorizes for `agent='graph-dba'` (mirrors unit test 3's shape, not the migration's
  per-batch `UNWIND` shape, correctly — this instruction covers ongoing single-entry writes, FR-2,
  not the one-time import). Field names/order match `docs/plans/generic-cypher-mcp-graph.md` §1's
  schema exactly. The `falkordb-quirks.md` direct-home carve-out is untouched.
- **`cobb.md`'s corrected distillation-duties bullet is accurate and not overcorrected** — it now
  states `graph-dba`'s raw capture is graph-based and *every other agent's remains file-based*,
  which is still true (verified: no other agent's prompt or `SKILL.md` §5 changed in this diff).
- **`skills/agent-maintenance/SKILL.md` §5's four-step distillation sequence for `graph-dba` matches
  plan §3.5 exactly**, step for step (read with `agent` omitted → verify → `Edit history.md`, confirm
  before proceeding → curator-clear `DETACH DELETE` with `agent='cobb'`), and the append-before-delete
  ordering constraint appears **only** in `SKILL.md`, confirmed absent from `graph-dba.md` by grep
  (`graph-dba.md`'s only "ordering" hit is the unrelated `falkordb-quirks.md` index-before-constraint
  reference) — exactly the design decision `docs/plans/generic-cypher-mcp.md` §3.5 calls for and
  `graph-dba`'s own U6 `kaizen/history.md` entry explicitly states was honored.
- **The grep-sweep close-out, `audit-team.sh` run, and frontmatter char-count self-reports all
  reproduce exactly** under independent re-execution — see Verification method above.
- **Scope discipline holds.** `git diff --stat` shows exactly the 16 modified files this feature's
  units collectively own (9 of them this unit's), plus 4 untracked docs from earlier units; nothing
  from `cpg/mcp/`, `docs/requirements/cpg-query-access.md`, or the three files the brief flags as
  pre-existing/unrelated was touched by U6.
- **`docs/BACKLOG.md`'s M5 item-to-step mapping is correct 1:1** (`C-501`→step 1 … `C-506`→step 5)
  and its section placement (after `## Follow-ups (post-M4)`, before the legacy `## Follow-ups
  (post-M2)` tail) matches cobb's own claim — only the *status markers* are wrong (M-B), not the
  content or mapping.

### Open questions

- None requiring the user's input. Both findings are directly actionable by `cobb` (or `teco`
  fixing them as a trivial follow-up) without further design input, and neither blocks dispatching
  U7 (step 5, `qa-engineer`'s acceptance pass) — the code and docs U7 needs to exercise are all
  correct; only `docs/BACKLOG.md`'s status bookkeeping and the two history entries' completeness lag.

### Verdict

**Approve with suggestions.** No blocker — every convention-doc edit is accurate, consistent, and
technically correct against the shipped enforcement code; the append-before-delete ordering
constraint lives in exactly one place, as designed; the grep-sweep, `audit-team.sh`, and
frontmatter-length self-reports all reproduce exactly under independent verification; scope
discipline holds. One Major (M-B): `docs/BACKLOG.md`'s M5 section understates real, already-delivered
progress against this repo's own git-history-verified convention (demonstrated directly via M4's
`50f9aaa` commit) — a real inaccuracy in a document `teco` and other docs rely on, but a two-minute
fix with no design implications. One Minor (m-B): both agents' `kaizen/history.md` U6 entries omit
the root `AGENTS.md` edit from their own "what changed" accounting, despite it being real and
explicitly authorized. Recommend `cobb` land both fixes before or shortly after U7 dispatches; neither
blocks sequencing.

**CPG:** considered, not relevant — this unit's diff is nine markdown/prompt documents (agent
prompts, skill procedure doc, catalog/convention docs, kaizen history entries); no call-graph,
data-flow, or impact-analysis question was in scope, and no Joern CPG covers this repo's own
`claude/`/`skills/` agent-definition tree in any case (same conclusion as every prior pass on this
feature, and as `docs/plans/generic-cypher-mcp.md`'s own CPG line for its §3 tool-mechanism design).
Verification here was direct diff reading, live re-execution of the grep sweep and `audit-team.sh`,
and direct git-history inspection (`git show`, `git log --follow`) — not graph queries.
