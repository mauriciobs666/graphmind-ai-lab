# Agent-team graph sandbox — Plan Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M<n> TBD)

## Scope & verdict

Reviewed `docs/plans/kaizen-team-sandbox.md` (Status: active, Owner: `architect`, not yet
committed) against its requirements source of truth, `docs/requirements/kaizen-team-sandbox.md`
(Status: Ready for design, committed at `e723166`, FR-1..FR-9/AC-1..AC-6). Checked: every
traceability-table claim against the actual FR/AC text; the DDL block against the two prior
`kaizen_team` schema-provisioning precedents (`docs/plans/kaizen-agent-ontology-graph.md` §6,
`docs/plans/generic-cypher-mcp2.md` S0); the isolation-mechanism rationale (§3.1) against
`cypher-mcp/docker-run.sh`, `falkor-chat/scripts/start_falkordb.sh`, and
`cypher-mcp/README.md`'s documented write-authorization/env-forwarding behavior; the
`guard-destructive-ops.sh` hook-wiring claim against `claude/devops/devops.md` /
`claude/graph-dba/graph-dba.md` / `claude/qa-engineer/qa-engineer.md`; and whether `.mcp.json`
(the file the plan's provisioning/teardown workflow edits) is itself a tracked, committed repo
file (`git ls-files .mcp.json` — it is, with real edit history). Did not independently verify the
Redis-logical-DB-vs-graph-key architectural claim in §3.1 against live FalkorDB behavior — it
matches this reviewer's own general Redis/FalkorDB knowledge and is not load-bearing enough to the
verdict to warrant a live probe.

`CPG: considered, not relevant — GRAPH.LIST (checked this session via mcp__cypher__query) shows no
cpg_cypher-mcp/cpg_claude graph; only cpg_falkorchat and cpg_salesperson are loaded, and this
review's subject (a plan touching cypher-mcp/, claude/, and provisioning scripts) has no source in
a component with a loaded CPG — same conclusion the plan's own §1 CPG line reaches.`

**Verdict: needs changes.** No blockers — the core design decision (separate FalkorDB instance per
sandbox, not a same-instance graph-key split) is correctly derived from FR-3 and well-justified,
the DDL block is byte-consistent with prior verified precedent, and the traceability table's FR/AC
mappings hold up against the requirements text. Three Major findings need an architect revision
before this is ready to route to implementers — none of them reopen the requirements doc; all are
design-level gaps the plan can close without going back to `tico`/the stakeholder.

## Findings

### Major — no guidance on `.mcp.json`'s git lifecycle for sandbox entries

`.mcp.json` is a committed, tracked repo file (`git ls-files .mcp.json` confirms it; `git log -p`
shows real prior edits, e.g. commit `59a03c4`). §3.2 says sandboxes are per-request and
"`.mcp.json` accumulates one entry per active sandbox," and §4 steps 1/2/3 have devops
add/print/remove entries — but nowhere does the plan say whether a sandbox's entry gets
**committed**. If it does, the shared tracked file accumulates ephemeral, host/port-specific
entries pointing at containers other contributors don't have running, and every provision/teardown
needs its own commit that nothing in steps 1-3 mentions. If it doesn't, every contributor with an
active sandbox is carrying a permanent uncommitted working-tree diff, at risk from any git
operation that touches the tree (`checkout`, `stash`, `reset`) — and AC-4's own verification method
("diff `.mcp.json`'s `cypher` entry before/after... to confirm it is byte-unmodified," §5) implies
git-tracked diffing, which only makes sense if sandbox entries *are* meant to be committed.
**Suggested fix:** add one sentence to step 3 stating whether the entry is committed (and by whom)
or deliberately kept as an uncommitted local edit, and if committed, add the corresponding
`.mcp.json`-entry removal to step 2's "done" condition instead of leaving it as a "reminder line."

### Major — §3.3's "no process to route through" for undocumented work leaves AC-5 unmet in that case

AC-5's text has no carve-out: "Given a piece of work touching the agent-team graph is proposed...
then there is a recorded decision (with the stakeholder) on whether it goes through the sandbox —
not an assumption either way." §3.3 reads FR-5's "no fixed checklist... case by case" as license to
skip recording entirely for "smaller, undocumented work": "there is deliberately no process to
route through." FR-5 is about not mechanizing *how* the risk call is made; it says nothing about
whether the call must be recorded, and AC-5 explicitly requires that it is, for any work that
"touches the agent-team graph" — not just work big enough to get its own `docs/plans/<slug>.md`.
As written, a small undocumented change has no mechanism at all to leave a recorded decision, which
is a literal AC-5 gap, not a proportionality call the plan is entitled to make unilaterally.
**Suggested fix:** for undocumented work, require a minimal recorded artifact proportionate to
FR-5's "no checklist" spirit — e.g. a one-line kaizen entry via the requester's own
producer-write shape (`fact: "kaizen_team schema/data touched directly, no sandbox — <one-line
risk call>"`), not a full plan-doc subsection. This is a design-level fix `architect` can make
without reopening the requirement.

### Major — AC-2's verification step (§5) leaves permanent cruft on production schema

The AC-2 row: "Add a throwaway index/constraint to the sandbox instance directly... Then
deliberately replay **the same DDL** against production (the promotion path, §3.4) and confirm it
now appears there." Taken literally, this applies the sandbox's *throwaway* test index/constraint
to real production `kaizen_team` as part of verifying the plan, with no step telling the executor
(`qa-engineer`/`graph-dba`, per §5's intro) to drop it afterward — so running this check as written
permanently pollutes production's schema with a throwaway artifact.
**Suggested fix:** either (a) add an explicit teardown line ("drop the throwaway
index/constraint from production immediately after confirming promotion works"), or (b) replace
the promotion-path half of this check with replaying the plan's own real, validated DDL block
(§2) — which needs to exist in production eventually anyway — instead of an arbitrary throwaway
one.

### Minor — image-drift risk (§6) could be closed structurally instead of just documented

§6 flags that the provisioning script's `--image` default is a point-in-time copy of
`falkor-chat/scripts/start_falkordb.sh`'s pinned image, which can silently drift if that script's
pin is bumped later without updating the sandbox script to match — and proposes only "a standing
reminder in `cypher-mcp/README.md`," i.e. a disciplinary fix. The plan's own stated preference
elsewhere (§3.1: "a structural, not disciplinary, isolation guarantee") argues for better here too:
have the provisioning script source the image ref from `start_falkordb.sh` directly (e.g. parse its
`FALKORDB_IMAGE:-` default) rather than hardcoding a copy, which removes the drift risk instead of
documenting it. Not a blocker — the documented reminder is a legitimate fallback — but worth adding
to step 1 as the preferred implementation.

### Minor — host-level resource limits left as a "consider," despite directly serving FR-3/AC-3

§6 correctly identifies that container-per-instance isolation doesn't eliminate host-level CPU/RAM
contention, and AC-3's own check explicitly measures "no latency/error attributable to the sandbox
event." Given `docker run --cpus`/`--memory` are one flag each and directly address the named
residual risk against the plan's own acceptance criterion, making them a default in step 1's
`docker run` invocation (rather than "a devops operational call... not specified as a hard
requirement") would close a real gap at near-zero cost. Suggest making a conservative default (e.g.
`--cpus=1 --memory=1g`) part of step 1's script, overridable by flag for a load-test scenario that
needs headroom.

## What's solid

- **§3.1's core call is correctly derived and well-argued.** The same-instance graph-key
  alternative is rejected specifically because it fails FR-3 (one engine process, one query
  executor, one crash blast radius) — verified against the requirement's exact wording and against
  the repo's own precedent (the crash-repro RCA's "never `falkordb-dev`" pattern), not asserted.
- **The DDL block (§2) is byte-consistent with verified precedent.** Cross-checked against
  `docs/plans/kaizen-agent-ontology-graph.md` §6 and `docs/plans/generic-cypher-mcp2.md` S0:
  index-before-constraint ordering, `NODE` (not `LABEL`) keyword, and async
  `PENDING`→`OPERATIONAL` polling all match exactly.
- **No new server code, proportionate to scope.** Correctly leans on `authorize_write()` already
  being instance-agnostic (verified against `cypher-mcp/README.md`'s write-shape documentation)
  rather than proposing a `cypher-mcp/server.py` change nothing in the requirements calls for.
- **Traceability table holds up.** Checked every FR-1..FR-9/AC-1..AC-6 row against the requirements
  doc's actual text; all map to a real, correctly-scoped section of the plan.
- **Honest risk surfacing.** §6's four risks (host contention, image drift, no auto-expiry,
  unverified cross-instance copy) are real and none are hidden or downplayed — this is what let
  this review find the two that need a harder fix (above) instead of new ones.

## Open questions

None of the findings above reopen the requirements doc — FR-5's "case by case, no checklist"
already anticipated informality in the risk call itself; AC-5 still requires the call be recorded,
which is a plan-level gap, not a requirements one. The one item that does need a decision outside
`architect`'s remit is already correctly flagged by the plan itself (§6, last bullet): whether the
"Sandbox & promotion" subsection convention should be promoted from this plan into root
`AGENTS.md`'s documentation-convention section. This review has nothing to add to that — it's
appropriately routed to the stakeholder/`teco` already, not a defect in this plan.

## Pass 2 — 2026-08-31

Re-read the on-disk plan in full (Version 2, revision note under the header). Rechecked each
architect-summarized change against the actual text, plus live-verified the two new FalkorDB
commands the AC-2 fix depends on (`WebFetch` against `docs.falkordb.com/commands/
graph.constraint-drop.html`, `WebSearch` for `DROP INDEX ON` syntax) and the `jq` dependency the
`.mcp.json`-lifecycle fix now relies on (`which jq` / `jq --version` on this box).

**Verdict: needs changes.** All three original Majors are substantively closed, and both Minors
adopted as suggested — but the mechanism chosen to close Major #1 (`.mcp.json` lifecycle) has a new
gap, itself Major, that needs one more pass before this is ready to commit.

**Original findings — disposition:**

1. **`.mcp.json` git lifecycle (Major)** — fixed in decision (§3.2: "deliberately never committed,"
   rationale, residual risk of a mid-session `git` op dropping the entry, mitigated by idempotent
   re-provisioning) and in mechanism (§4 steps 1–3: scripts manage the entry via `jq`, "done" now
   means the entry is actually present/absent, not printed). See new finding below — the `jq`
   mechanism itself isn't safe to ship as specified.
2. **AC-5 gap for undocumented work (Major)** — fixed. §3.3 now states plainly that FR-5 governs
   *how* the risk call is made, not *whether* it's recorded, and adds a required one-line kaizen
   entry for undocumented work, threaded through §4 step 6 and the §5 AC-5 row. See new Minor below
   on the entry template's field completeness.
3. **AC-2 production-schema pollution (Major)** — fixed, verified. §5's AC-2 row now drops the
   throwaway constraint then index as its own last step. Checked the exact syntax against FalkorDB's
   current docs: `GRAPH.CONSTRAINT DROP <graph> UNIQUE NODE <label> PROPERTIES <n> <prop>` confirmed
   live via `WebFetch` (`docs.falkordb.com/commands/graph.constraint-drop.html`, synchronous, returns
   `OK`); `DROP INDEX ON :Label(prop)` confirmed via `WebSearch` against FalkorDB's own docs/blog
   pages. Both match the plan's text exactly, and constraint-before-index drop order is correct
   (mirrors the index-before-constraint create order already verified in Pass 1).
4. **Image-version drift (Minor)** — fixed as suggested. §4 step 1 now parses
   `start_falkordb.sh`'s `FALKORDB_IMAGE:-` default live rather than hardcoding a copy (§6 updated
   to match).
5. **Host resource limits (Minor)** — fixed as suggested. §4 step 1's `docker run` now defaults
   `--cpus=1 --memory=1g`, overridable, cited against AC-3 in §5's row.

**New — Major: the `jq` dependency the `.mcp.json`-lifecycle fix relies on isn't declared, and isn't
present on this box.** §4 steps 1–3 now have the provisioning/teardown scripts manage `.mcp.json`
"via `jq`" with no fallback and no prerequisite note. Checked live on this machine: `which jq`
returns nothing, `jq --version` fails with "command not found" (exit 127) — `jq` is not installed
here. This is not a one-off: this exact repo already hit and documented this — `claude/devops/
kaizen/history.md`: *"`jq` is **not installed** on this WSL box"* — and every `PreToolUse` guard
hook in `claude/scripts/` (`guard-destructive-ops.sh`, `guard-doc-writes.sh`,
`guard-broad-write.sh`, `guard-agent-dispatch.sh`, four more) was written with a `jq`→`python3`
fallback specifically because of that finding (`claude/README.md`:158: *"scripts prefer `jq`, fall
back to `python3` — install one for..."*). The provisioning/teardown scripts as specified would fail
outright with no fallback on the very machine this plan's own findings (§2) were live-verified on —
which also means Major #1's fix is currently non-functional here, not just fragile.
**Suggested fix:** either give `provision-kaizen-sandbox.sh`/`teardown-kaizen-sandbox.sh` the same
`jq`→`python3` fallback already standard in `claude/scripts/*.sh` (cheapest, matches repo
convention), or declare `jq` a new hard prerequisite in `cypher-mcp/README.md`'s setup section and
have the script check `command -v jq` up front and fail with a clear message instead of a bare
"command not found."

**New — Minor: the undocumented-work kaizen-entry template drops two of `KaizenEntry`'s standard
seven fields.** §3.3's template (repeated in §4 step 6, §5's AC-5 row) creates a `:KaizenEntry` with
only `entryId`/`date`/`fact`/`context`/`createdAt` — missing `evidence` and `suggestedHome`. Every
other producer-write example in this repo (`cypher-mcp/README.md`'s "Writing through this tool",
this reviewer's own learning-capture instructions, `docs/plans/kaizen-agent-ontology-graph.md`'s
"five markdown-sourced fields" schema) carries all seven. `suggestedHome` specifically drives
`cobb`'s distillation routing (`claude/AGENTS.md`: "verify → route to prompt/knowledge base/project
docs"); an entry missing it is a poor fit for that workflow, not just inconsistent shape.
**Suggested fix:** add `evidence` (e.g. "ran DDL/write directly against production, confirmed via
<check>") and `suggestedHome` (a real routing guess, e.g. `'project docs'`) to the template.

**New — Minor: the `.mcp.json` residual-risk mitigation ("re-run step 1, idempotent add-if-missing")
isn't obviously true for an auto-selected port.** §6's mitigation for a git op dropping the sandbox
entry is "re-running step 1's script... regenerates exactly the same entry." But step 1's `--port`
default is "first free port at/above 16380" (§4 step 1) — on a bare re-run without an explicit
`--port`, the *original* port is still bound by the still-running sandbox container, so "first free"
would skip it and select a **different** port, producing a mismatched entry rather than the same
one. **Suggested fix:** have the script check for an already-running `falkordb-sandbox-<slug>`
container first (`docker inspect` its bound port) and reuse that port before falling back to
auto-selecting a new one — the one-line change that actually makes "idempotent" true.

## Pass 3 — 2026-08-31

Re-read the on-disk plan in full (Version 3, two revision notes stacked under the header, per
convention). Rechecked each architect-summarized change against the actual text; empirically tested
the JSON-mutation mechanism the `jq`→`python3` fix depends on (ran the described python3
load/modify/dump round-trip against a copy of the real `.mcp.json` in the scratch dir, diffed the
result); and traced the recovery-vs-new-provisioning split for the specific case asked about (a bare
`--slug` re-run against a torn-down, not still-running, sandbox).

**Verdict: needs changes.** All three Pass-2 items are genuinely closed as described. But the
concrete mechanism chosen to implement the `jq`→`python3` fix (Pass-2 item 1) has a new,
empirically-confirmed Major bug of its own, plus one related Minor gap in the recovery path's
"add-if-missing" logic — both are narrow, both have a clean fix, neither reopens the design.

**Pass-2 findings — disposition:**

1. **`jq` dependency, no fallback (Major)** — fixed as described: §4 steps 1–2 now specify a
   `jq`→`python3` fallback (`command -v jq`, else `command -v python3`), hard-failure (not fail-open)
   if neither is present, matching `claude/scripts/*.sh` convention and `claude/README.md`:158. The
   *declaration* gap from Pass 2 is closed. See the new Major below — the mechanism this fix
   specifies has its own bug, found by actually running it.
2. **Undocumented-work template missing `evidence`/`suggestedHome` (Minor)** — fixed, verified. §3.3's
   template (lines 203–215) now includes all seven fields (`entryId`, `date`, `fact`, `evidence`,
   `context`, `suggestedHome`, `createdAt`), text explicitly calls out this was "the abbreviated five
   this template shipped with in the prior revision."
3. **Port-reuse idempotency not actually idempotent (Minor)** — fixed, verified. §4 step 1 now has an
   explicit "recovery path" (bare `--slug`, container already running → reuse its `docker inspect`-ed
   port, skip `docker run`/schema bootstrap, go straight to the `.mcp.json` step) distinct from "new
   provisioning" (§6's residual-risk bullet updated to match, §5's edge-case list updated
   consistently). See related new Minor below on one gap in this same recovery logic.

**New — Major, empirically confirmed: the JSON-mutation mechanism reformats `.mcp.json`'s existing
`cypher` entry, breaking AC-4's own "byte-unmodified" check.** §4 step 1 describes adding the
sandbox entry via `jq`, falling back to `python3`. Tested the `python3` path directly — the one that
actually runs on this box, since `jq` is confirmed absent (Pass 2's live check) — against a copy of
the real `.mcp.json`: a plain `json.load` → dict-mutate → `json.dump(..., indent=2)` round trip
reformats the pre-existing `"cypher"` entry's `"args": ["-c", "exec ..."]` (currently one line) into
a four-line array, even though no value changed. `jq`'s default pretty-printer has the same behavior
(it always expands arrays across multiple lines unless invoked with `-c`) — not independently run
here (no `jq` binary on this box to test against), but this is documented, well-known `jq` behavior,
not a guess. Consequence: **every sandbox provision/teardown reformats the untouched `cypher` entry**,
which directly breaks §5's AC-4 verification method ("a plain before/after file diff of `.mcp.json`'s
`cypher` entry... confirms it is byte-unmodified") — that diff would show a change on the very first
sandbox provisioned, despite FR-4 holding in substance. It also means the "sandbox entries are the
only uncommitted diff" framing (§3.2, §4 step 3) is false in practice: `git diff .mcp.json` after
provisioning shows the `cypher` entry's formatting changed too, not just the new key.
**Suggested fix:** don't round-trip the whole file through a generic JSON parser/serializer for a
change this narrow. Both scripts only ever add or remove one fixed-shape block keyed by
`cypher-sandbox-<slug>` — a text-level edit (insert the literal snippet right after the
`"mcpServers": {` line; delete by matching from `"cypher-sandbox-<slug>": {` to its closing `},`)
achieves the same result without touching anything else in the file, and incidentally removes the
need for the `jq`/`python3` dependency entirely. If `architect` prefers keeping the JSON-library
approach, then relax AC-4's check to a semantic comparison (`python3 -c "import json;
assert json.load(open('a')) == json.load(open('b'))"` or `jq -S .` equality) instead of a raw byte
diff, and note in §3.2 that the whole file's formatting, not just the added key, changes on
provisioning.

**New — Minor: the recovery path's "add-if-missing" doesn't handle a *stale* entry, only a *missing*
one — leaves a narrow limbo state.** Directly checking the case asked about: a bare `--slug` re-run
against a **torn-down** (not still-running) sandbox correctly falls through recovery's "container
already running?" check (no) into the "new provisioning" path (§4 step 1) — that part is *not*
limbo, it just re-provisions fresh, which is correct. The gap is one step earlier: teardown (§4
step 2) runs `docker stop`/`rm`/`volume rm` **before** removing the `.mcp.json` entry, so if that
last step doesn't complete — the jq/python3 hard-failure case Pass 2's fix itself introduced, or
simply a contributor who tore the container down manually (`docker stop`/`rm` by hand, not through
the script — nothing in this plan prevents that) — the container is gone but a **stale** `.mcp.json`
entry (wrong/dead port) survives. A subsequent provisioning run's "add-if-missing" wording (§4 step 1)
reads as skip-if-key-present, which would leave that stale port in place rather than correcting it to
the newly (re-)provisioned container's actual port. **Suggested fix:** "add-if-missing" should really
be "add-or-correct": on provisioning, write/overwrite the entry unconditionally with the
now-authoritative port rather than skipping when the key already exists.

## Pass 4 — 2026-08-31

Re-read the on-disk plan in full (Version 4, three revision notes stacked under the header). Per the
request, built and ran a working prototype of the described text-anchored splice (`awk` brace-depth
delete + file-based insert-after-anchor) against a realistic multi-entry `.mcp.json` in the scratch
dir, rather than tracing it statically — this mechanism is exactly the kind of thing that reads as
correct on paper and had a real bug last time it wasn't run.

**Verdict: approve with suggestions.** Both Pass-3 items are genuinely fixed and this time verified
by execution, not just reading. One new Minor — a real but avoidable implementation footgun in *how*
the insert step is coded, not in the design itself. No blockers, no majors outstanding.

**Pass-3 findings — disposition:**

1. **JSON round-trip reformats the untouched `cypher` entry (Major)** — fixed, verified by running
   it. Built the described mechanism (brace-balanced `awk` delete keyed on
   `"cypher-sandbox-<slug>": {`, skipping past the nested `"env": {...}` sub-object without
   false-matching its close; insert via reading a literal block file after the `"mcpServers": {`
   anchor) against a 3-scenario fixture: insert one sandbox entry, stack a second, delete the
   first-inserted (middle position), delete the last remaining (first position). At every step:
   `python3 -m json.tool` confirmed valid JSON, and a `diff` against the original file showed
   **only** the added/removed lines — the `cypher` entry's `args` array stayed on one line,
   untouched, through the whole sequence. After deleting both sandbox entries, the file was
   byte-identical to the original. AC-4's byte-unmodified check now holds in practice, not just in
   the plan's prose.
2. **`add-if-missing` doesn't correct a stale entry (Minor)** — fixed, verified by reading. §4 step 1
   now states "add-or-correct... a stale entry (wrong/dead port)... is corrected the same way a
   missing one is added, by the same delete-then-insert"; §6 and §5's edge-case list updated to match.

**New — Minor: the insert step's literal-text requirement is easy to violate with the wrong shell
tool, and the plan doesn't warn about it.** The entry being spliced in contains escaped quotes
(`"args": ["-c", "exec \"$CLAUDE_PROJECT_DIR/...\""]`) — the same content the existing `cypher` entry
already carries. Tried the naive, equally-plausible way to pass that literal text into an insertion
(`awk -v block="$block_text"` then `print block`) first: `awk -v` **reinterprets backslash escapes in
its `-v` argument**, so the inserted block came out with `\"` collapsed to `"`, producing invalid
JSON (`python3 -m json.tool` failed: `Expecting ',' delimiter`). Re-did it via a heredoc'd block file
plus `sed '/"mcpServers": {/r blockfile'` (reads the file's raw bytes, no escape reprocessing) and it
was clean — this is the version verified in finding 1 above. The plan's text (§4 step 1) says "`bash`
+ `awk`/`sed`/`grep` is sufficient" without flagging that not every idiom in that toolset is safe for
this specific payload — `awk -v`/`echo -e`/unquoted interpolation will silently corrupt it, `sed
r <file>`/direct file-read-and-splice will not. **Suggested fix:** add one line to §4 step 1 (or the
`cypher-mcp/README.md` section, step 4) naming the safe idiom explicitly (write the block to a temp
file, `sed '/anchor/r tempfile'`, or equivalent — never pass the literal JSON text through a shell
variable expansion or `awk -v`) so whoever implements this doesn't rediscover the footgun by shipping
it.

No other new findings — the rest of the plan is unchanged since Pass 3 (confirmed by reading it in
full again) and Pass 1–3's other closed items remain closed.
