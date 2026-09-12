# Cypher MCP tool surface — implementation plan

> **Status:** active · **Owner:** `architect` · **Tracks:** C-901, C-902 (M9)
>
> Design for **direct, in-tool FalkorDB graph discovery** on the `cypher` MCP server's existing
> single tool, `mcp__cypher__query`. Requirements:
> [`../requirements/cypher-mcp-tool-surface.md`](../requirements/cypher-mcp-tool-surface.md), which
> revisits [`cpg-query-access.md`](./cpg-query-access.md) FR-2 per that plan's own recorded reversal
> trigger (§3.3, quoted below). Builds on, and does **not** re-litigate, that plan's §3 (build vs.
> buy) or [`generic-cypher-mcp.md`](./generic-cypher-mcp.md)'s "why one tool, not a second tool"
> reasoning (~its §3.1, the widen-vs-second-tool table) — both cited below, not re-derived. Author:
> `architect`. v1 2026-09-12.

**CPG:** considered, not relevant — this is a design task over `cypher-mcp/server.py` (1030 lines,
read in full) and two markdown docs (`cypher-mcp/README.md`, `docs/requirements/
cpg-query-access.md`); no Joern CPG is loaded for the `cypher-mcp` component itself (loaded CPGs in
this repo cover `falkor-chat`/`salesperson` application code, not this repo's own MCP tooling), and
the investigation was direct file reads of a small, already well-documented surface — a call-graph
tool would add nothing here.

**Reading order for an implementer:** §3 (the design) → §4 (the steps, with exact code) → §5 (the
test list). §2 is background you can skim if you already trust the citations above.

---

## 1. Goal & scope

Give any caller of `mcp__cypher__query` a **direct** way to list every FalkorDB graph currently
loaded on the instance — no deliberately-wrong graph name, no shelling out to `redis-cli GRAPH.LIST`
— **without** adding a second tool or a required third parameter (FR-1/FR-2 of the requirements
doc). The set of names revealed must have the **same blast radius** as what already leaks today
through the tool's "graph not found" error message: every graph on the instance, unfiltered — no
narrower, no broader (FR-3).

**In scope:** the discovery mechanism itself (`cypher-mcp/server.py`), its tests, `cypher-mcp/
README.md`'s "Graph discovery" section (AC-3), and the header-pointer annotation on `docs/
requirements/cpg-query-access.md`'s FR-2 (FR-4).

**Out of scope** (per the requirements doc's own Out of scope, unchanged here): schema discovery
(a `get_graph_schema`-equivalent); any change to what is exposed or the tool's blast radius beyond
this one capability; re-running the build-vs-buy comparison against `@falkordb/mcpserver`, or any
multi-tool redesign; `delete_graph` or any other official-server-only capability; the write path
(`authorize_write()`'s six shapes, the `agent` parameter, `kaizen_team`'s schema) — all unaffected.

---

## 2. Context & findings

- **The tool today:** `cypher-mcp/server.py` exposes one MCP tool, `query`, callable
  `mcp__cypher__query(graph: str, cypher: str, agent: str | None = None) -> str`. `graph`/`cypher`
  are required; `agent` is the one optional parameter `generic-cypher-mcp.md` added (server.py:931).
  `split_directive()` (server.py:211-230) already classifies `cypher` into `"query"` / `"explain"` /
  `"profile"` via a comment-blind leading-trivia scan (`_scan_leading_trivia()`,
  server.py:179-208) before anything is sent to FalkorDB — this is the established, tested pattern
  a new directive extends, not a new mechanism.
- **The "leak" FR-3 anchors on already exists and is already used internally.** `graph_not_found_message()`
  (server.py:842-848) and two call sites in `run_query()` (server.py:967, 1015) already call
  `list(client.list_graphs())` to build the "Loaded graphs: …" line on a missing-graph error.
  `_list_graphs()` (server.py:907-911) is a best-effort wrapper used only for error-message
  enrichment (swallows exceptions, returns `None` on failure) — the new discovery path must call
  `client.list_graphs()` directly, the same way the two existing call sites do, **not** through
  `_list_graphs()`, so a real connection failure still surfaces the curated "FalkorDB unreachable"
  message instead of a silent empty list.
- **`README.md`'s "### Graph discovery" section (lines 311-314) makes exactly the claim this
  delivery falsifies:** *"There is no `list_graphs` tool (FR-2: one tool). To discover graph names,
  either query a name that does not exist — the error lists every loaded graph — or use `redis-cli
  -p 6379 GRAPH.LIST`."* This is the one place in the file making that claim (grepped for
  `list_graphs`/`discover`/`GRAPH.LIST` across the whole document); it is what AC-3 requires
  rewriting.
- **`docs/requirements/cpg-query-access.md` is `Status: archived`.** Its header already carries one
  `**Note:**` line pointing at `generic-cypher-mcp.md` FR-1 for two Out-of-scope bullets it widened.
  Root `AGENTS.md`'s doc-lifecycle convention: *"A header pointer is metadata, not an amendment — it
  is the one edit permitted on an archived document."* `generic-cypher-mcp.md` §5 already worked out
  the correct, convention-compliant substitute for "mirror how FR-9 was superseded" when the target
  is archived (plain `**Note:**` prose, not the `Supersedes:`/`Superseded by:` field pair, which is
  reserved for same-slug ordinal succession) — this delivery reuses that same substitute, extending
  the existing Note rather than inventing a second mechanism.
- **The settled reasoning this plan does not reopen:** `cpg-query-access.md` §3.2's recorded
  reversal trigger — *"if a future need arises for multi-tool graph access (schema discovery, write
  paths, non-CPG graphs for other agents), revisit — at that point FR-2 no longer binds and the
  official server becomes the cheaper answer"* — and `generic-cypher-mcp.md`'s own "why one tool,
  not a second tool" table (its §3.1) are both **cited, not re-derived**, per the requirements
  doc's own Out of scope line ruling out a build-vs-buy redo. This plan's only job is the mechanism
  for one additive capability inside the existing one-tool shape.
- **No dependency, container, or `.mcp.json` change.** `re` is already imported (stdlib); the
  server name, tool name, and launch command are all unchanged. The container image tag is a
  content hash over `server.py` and everything under `tests/` (`cypher-mcp/image-tag.sh`), so
  editing both changes the hash automatically — the next `docker-run.sh` self-heals (miss → build),
  exactly as `generic-cypher-mcp.md` §4 already established for this component.

---

## 3. Design & rationale

### 3.1 Mechanism: a fourth directive, `GRAPHS`, recognized by `split_directive()`

**Decision:** extend the existing directive-prefix mechanism (`EXPLAIN`/`PROFILE`) with a fourth
keyword, `GRAPHS`. Sending `cypher="GRAPHS"` (case-insensitive, comment-blind, surrounding
whitespace tolerated — identical trivia handling to `EXPLAIN`/`PROFILE`) returns the full,
unfiltered list of graph names on the instance. **No new parameter of any kind** — `graph` and
`cypher` stay exactly the two required parameters they are today; `graph`'s value is simply ignored
for this one directive (convention: pass an empty string).

**Why a directive, not a new optional parameter (the requirements doc's other named option).**
FR-2 explicitly permits an optional parameter, following the `agent` precedent — but a directive is
strictly cheaper on every axis that precedent doesn't force:

| | Directive (`GRAPHS` in `cypher`) — chosen | Optional 4th parameter (e.g. `list_graphs: bool = False`) |
|---|---|---|
| Schema change | **None.** `inputSchema` is untouched; `test_input_schema_has_two_required_params_and_one_optional_agent` (the frozen-shape regression pin) needs no edit. | A new property in `inputSchema`; that test's own assertions (`set(schema["properties"]) == {"graph", "cypher", "agent"}`) would need editing — touching a test whose whole point is pinning the shape. |
| `graph`/`cypher` still required but unused for this call | Same wrinkle either way: `graph` must still be supplied (MCP requires it) but plays no role. A directive at least keeps the *reason* localized to the one parameter (`cypher`) whose job is already "tell the server what to do." | The unused-`graph` wrinkle is identical, plus now `cypher` is *also* unused/ignored when the new flag is set — two parameters idle instead of one. |
| Reuses proven, tested machinery | Yes — `_scan_leading_trivia()` and the comment-blindness `D5a` already established for `EXPLAIN`/`PROFILE` (`docs/plans/cpg-query-access.md` §4.4 D5a) apply unchanged. | No — a boolean flag needs its own precedence rule against `cypher`/`agent` (what does `list_graphs=True, cypher="MATCH..."` mean?), a new class of ambiguity `EXPLAIN`/`PROFILE` don't have. |
| Precedent in this codebase | `EXPLAIN`/`PROFILE` are exactly this shape: a keyword in `cypher` that changes what the tool does, not what it queries. | `agent` is the only precedent, and it is semantically a *caller identity*, not a *mode switch* — `kind`/`mode` parameters were explicitly rejected when `PROFILE` was designed (`cpg-query-access.md` §4.4 D5, "a `mode` parameter (violates FR-2)") for conflating the two ideas the directive keeps separate. |

**Why not a sentinel `graph` value instead** (e.g. `graph="__list__"`): rejected outright — it
either collides with a real graph someone names that literally, or it re-introduces exactly the
"hardcoded magic value" FR-4 of the *original* `cpg-query-access.md` contract (unaffected by this
delivery) was written to prevent ("the graph name stays caller-supplied... never defaulted, never
inferred"). A directive in `cypher` has no such collision surface: `cypher` is free-text Cypher, and
`GRAPHS` is not valid Cypher syntax at the start of any real statement (every real statement opens
with `MATCH`/`CREATE`/`MERGE`/`WITH`/`UNWIND`/`CALL`/`RETURN`, …), so there is no realistic query
this directive could misclassify.

**The exact detection rule — `GRAPHS` is a standalone directive, unlike `EXPLAIN`/`PROFILE`.**
`EXPLAIN`/`PROFILE` are *prefixes* in front of a further Cypher statement; `GRAPHS` has no
"statement" to prefix — there is nothing to send anywhere. So, unlike the other two, the keyword
must consume the **entire** trivia-stripped input (trailing whitespace aside), not merely lead it:

1. Compute `start = _scan_leading_trivia(cypher)` — unchanged, comment-blind, exactly as today.
2. Match `_GRAPHS_DIRECTIVE_RE = re.compile(r"GRAPHS\b\s*\Z", re.IGNORECASE)` anchored at `start`
   via `.match(cypher, start)`. `\b` excludes `GRAPHS_LIST`/`GRAPHSTATS`-shaped decoys (mirrors the
   existing `EXPLAIN_ME`/`PROFILER`/`PROFILEDATA` protection); `\s*\Z` requires nothing but trailing
   whitespace to the end of the string.
3. If it matches → kind `"graphs"`, nothing to send (`""`, same convention as `"profile"`).
4. If it does not match — including the case where real query text follows `GRAPHS`, e.g. a caller
   who actually meant a query starting with the word "graphs" as an identifier — **fall through to
   `"query"` classification unchanged.** FalkorDB gets to return its own syntax error, the same
   fail-safe default `_scan_leading_trivia()` already uses for an unterminated block comment. No
   new decoy protection is needed beyond `\b` because `GRAPHS` is never a bare identifier a real
   Cypher statement would open with (see above).
5. `graph` is never read on this path — `run_query()`'s `"graphs"` branch never calls
   `client.select_graph(graph)`.

**Blast radius (FR-3), by construction, not by a new filter.** The `"graphs"` branch calls
`client.list_graphs()` — the exact same call `graph_not_found_message()`'s two existing call sites
already make. There is no new access-control surface to design: whatever a caller could already
learn by deliberately mistyping a graph name, they can now learn directly, and nothing more.

**Rendering.** `format_graph_list(graphs)` — sorted (FalkorDB's own `GRAPH.LIST` order is not
documented as stable, so sorting buys determinism without narrowing or widening the *set*, the same
scoping precedent `cpg-query-access.md`'s AC-3 reconciliation (`D5`) already established for
non-scalar rendering: values/membership are what equivalence protects, not incidental formatting):

```
graphs=3
cpg_falkorchat
kaizen_team
ws:acme
```

Zero graphs loaded → `graphs=0\n(none loaded)`, the same "don't leave an agent guessing whether the
call worked" idiom `format_result()` already uses for `(no rows)`.

### 3.2 FR-4 — the `cpg-query-access.md` header-pointer edit

Per §2's finding, the edit is header-metadata only, extending the existing `**Note:**` line rather
than adding a second one (kept as one coherent pointer paragraph):

**Old (current) header note (lines 6–9):**
```markdown
> **Note:** the "Non-CPG graphs / general agent access to FalkorDB" and "Authentication, per-user
> grants, and read-only enforcement" lines below are widened by
> [`generic-cypher-mcp.md`](./generic-cypher-mcp.md) FR-1 — read that document for the current
> scope; this archived document's body is left exactly as originally written.
```

**New header note:**
```markdown
> **Note:** the "Non-CPG graphs / general agent access to FalkorDB" and "Authentication, per-user
> grants, and read-only enforcement" lines below are widened by
> [`generic-cypher-mcp.md`](./generic-cypher-mcp.md) FR-1, and FR-2's "a single tool taking exactly
> two parameters" is revisited — without adding a required parameter or a second tool — by
> [`cypher-mcp-tool-surface.md`](./cypher-mcp-tool-surface.md) — read whichever document matches
> your question for the current scope/mechanism; this archived document's body is left exactly as
> originally written.
```

Only this paragraph changes; nothing else in `cpg-query-access.md` is touched (`git diff` must show
only the header block). Owner: `coder`, bundled with the `server.py` change — mirrors the M3/M5
precedent (`generic-cypher-mcp.md` §5, "Owner: `coder`, bundled with the `server.py` change").

---

## 4. Step-by-step implementation

Small enough for one implementer, one pass. Two files carry the mechanism; two carry documentation.

### Step 1 — `cypher-mcp/server.py`

**1a. Module docstring** (after the "`EXPLAIN` is honoured, `PROFILE` is actively refused" bullet,
~line 37, before "Display-only truncation"): add one bullet —

```
* **A fourth directive, `GRAPHS`, lists every graph on the instance directly** — no `graph`/
  `cypher` round trip, no "mistype a name to trigger the error" workaround. `graph` plays no role
  for this one call. See `docs/plans/cypher-mcp-tool-surface.md`.
```

**1b. New regex**, immediately after `_DIRECTIVE_RE` (~line 176):

```python
#: `GRAPHS` is a *standalone* directive, not a prefix in front of a further
#: statement like EXPLAIN/PROFILE — there is no "rest of the query" to send
#: anywhere. It must be the *entire* trivia-stripped input (trailing
#: whitespace aside); anything else after it falls through to plain `"query"`
#: classification and gets FalkorDB's own syntax error, the same fail-safe
#: default `_scan_leading_trivia` already uses for a malformed block comment.
_GRAPHS_DIRECTIVE_RE = re.compile(r"GRAPHS\b\s*\Z", re.IGNORECASE)
```

**1c. `split_directive()`** — add the new branch first (it can never collide with the existing
`_DIRECTIVE_RE` match, but checking it first keeps the function reading top-to-bottom by kind) and
extend the docstring:

```python
def split_directive(cypher: str) -> tuple[str, str]:
    """Classify a statement as ``"query"``, ``"explain"``, ``"profile"`` or
    ``"graphs"``.

    Returns ``(kind, cypher_to_send)``:

    * ``"query"`` — the caller's string **byte-for-byte**. The normalisation
      above is used for classification only; FR-3 promises verbatim
      transmission, and stripping comments would corrupt e.g.
      ``CONTAINS '// x'``.
    * ``"explain"`` — the text after the keyword, left-stripped. The consumed
      leading trivia is dropped; it cannot change a plan.
    * ``"profile"`` — the empty string. Nothing is ever sent.
    * ``"graphs"`` — the empty string. Nothing is ever sent, and `graph` is
      never read either: this directive lists every graph on the instance,
      which has nothing to do with any one graph's name
      (`docs/plans/cypher-mcp-tool-surface.md`).
    """
    start = _scan_leading_trivia(cypher)
    if _GRAPHS_DIRECTIVE_RE.match(cypher, start):
        return "graphs", ""
    match = _DIRECTIVE_RE.match(cypher, start)
    if match is None:
        return "query", cypher
    if match.group(1).upper() == "PROFILE":
        return "profile", ""
    return "explain", cypher[match.end():].lstrip()
```

**1d. New formatter**, immediately after `format_plan()` (~line 792):

```python
def format_graph_list(graphs: list[str]) -> str:
    """Render the `GRAPHS` directive's result: every graph name currently
    loaded on the instance, sorted for a deterministic reading order
    (`GRAPH.LIST`'s own order is not documented as stable). Same blast radius
    as `graph_not_found_message()`'s "Loaded graphs" line — both read
    `client.list_graphs()` unfiltered (FR-3, `cypher-mcp-tool-surface.md`).
    """
    if not graphs:
        return "graphs=0\n(none loaded)"
    return "\n".join([f"graphs={len(graphs)}", *sorted(graphs)])
```

**1e. `run_query()`** — insert the new branch immediately after computing `kind, to_send`, before
the existing `if kind == "profile":` check:

```python
    try:
        kind, to_send = split_directive(cypher)

        if kind == "graphs":
            # `graph` plays no role here (split_directive's docstring) — list
            # every graph on the instance, unfiltered, the same list
            # `graph_not_found_message()` already leaks (FR-3: identical
            # blast radius, never narrower or broader). A connection failure
            # here falls straight to the outer `except`, which curates it via
            # the existing `explain_error()` unreachable-DB branch — no new
            # error handling needed.
            return format_graph_list(list(get_client().list_graphs()))

        # Refused before any server call: FalkorDB would silently ignore the
        # prefix and return results, which is a wrong answer, not an error.
        # PROFILE can never be used to dodge write authorization below.
        if kind == "profile":
            return PROFILE_REFUSAL
        ...  # rest of the function unchanged
```

**1f. `TOOL_DESCRIPTION`** — insert one sentence between the existing `PROFILE`/`EXPLAIN` sentence
and the `FalkorDB is OpenCypher` sentence:

```
Send exactly `GRAPHS` (graph parameter ignored) to list every loaded graph directly, instead of
triggering the not-found error.
```

**1g. `SERVER_INSTRUCTIONS`** — insert one clause after "…answers with the list of loaded graphs."
and before "Reads need no `agent`…":

```
Sending exactly `GRAPHS` (graph parameter ignored) lists them directly.
```

Re-run `len(SERVER_INSTRUCTIONS)` by hand (or just run the test) after this edit —
`test_server_instructions_are_present_and_bounded` asserts `<= 2000`; the string is 1254 chars
today, so there is ~750 chars of headroom and this one clause (~65 chars) does not come close.

**Done-condition for step 1:** `cypher-mcp/.venv/bin/pytest cypher-mcp/tests -q` green (§5's new
tests included), no existing test edited except the `FakeClient` extension in §5.

### Step 2 — `cypher-mcp/tests/test_server.py`

Extend `FakeClient` to let a test simulate `list_graphs()` failing (needed for one new error-path
test; every existing use of `FakeClient(graphs=...)` is unaffected since the new parameter defaults
to `None`):

```python
class FakeClient:
    def __init__(self, *, graphs=(), list_graphs_error=None, **graph_kwargs):
        self.calls: list[tuple] = []
        self._graphs = list(graphs)
        self._list_graphs_error = list_graphs_error
        self.graph = FakeGraph(self.calls, **graph_kwargs)

    def select_graph(self, name):
        self.calls.append(("select_graph", name))
        return self.graph

    def list_graphs(self):
        self.calls.append(("list_graphs",))
        if self._list_graphs_error is not None:
            raise self._list_graphs_error
        return list(self._graphs)
```

The rest of §5 lists every test to add. **Done-condition:** the full list in §5 passes; no existing
test's assertions change (`test_input_schema_has_two_required_params_and_one_optional_agent` is a
regression pin proving this delivery added **zero** schema surface — run it and confirm it still
passes unedited).

### Step 3 — `cypher-mcp/README.md`

Replace the "### Graph discovery" section (current lines 311–314) in full:

**Old:**
```markdown
### Graph discovery

There is no `list_graphs` tool (FR-2: one tool). To discover graph names, either query a name that
does not exist — the error lists every loaded graph — or use `redis-cli -p 6379 GRAPH.LIST`.
```

**New:**
```markdown
### Graph discovery

Send exactly `GRAPHS` as the `cypher` text (case-insensitive, comment-blind, surrounding whitespace
tolerated — the same leading-trivia scan `EXPLAIN`/`PROFILE` use) to list every graph currently
loaded on the instance, directly — no need to mistype a graph name first, no shell command. `graph`
plays no role for this call; pass an empty string by convention. Output:

```
graphs=3
cpg_falkorchat
kaizen_team
ws:acme
```

(names sorted for a deterministic reading order — `GRAPH.LIST`'s own order is not documented as
stable). Same blast radius as everything else this tool already exposes: the full, unfiltered
instance-wide list (`client.list_graphs()`), identical to what a "graph not found" error already
lists — not narrower, not broader
([`../docs/plans/cypher-mcp-tool-surface.md`](../docs/plans/cypher-mcp-tool-surface.md) FR-3).
Nothing loaded → `graphs=0` and `(none loaded)`.

The two older paths still work, unaffected: query a name that does not exist (the error still lists
every loaded graph), or `redis-cli -p 6379 GRAPH.LIST` directly.
```

No other section of `README.md` makes the "no discovery" claim (grepped: only these four lines).

**Housekeeping, same step:** the "in-container test gate" section (~line 621) states exact pass
counts (`74 passed, 7 deselected` offline / `7 passed, 74 deselected` live) that will shift once
§5's new tests land — recount and update both numbers as part of this step's done-condition; a
stale count in this file is exactly the kind of drift `AGENTS.md`'s "the bar" convention warns
against.

**Done-condition for step 3:** the section reads correctly against the actually-implemented
behavior (re-read it after step 1 lands, don't write it from the plan alone), and the test-gate
counts are current.

### Step 4 — `docs/requirements/cpg-query-access.md`

Apply the header-note edit from §3.2 verbatim. **Done-condition:** `git diff` shows **only** the
header block changed (mirrors the M3/M5 precedent's own done-condition for this exact kind of edit).

### Step 5 — `docs/HISTORY.md` closeout entry

This delivery is self-contained (one implementer, one pass) — unlike M3/M5/M8 it never sits
"in-progress" in `docs/BACKLOG.md`'s Open section, so per that file's own "forward-looking only"
rule (root `AGENTS.md`: *"a delivered item does not stay in it, not even as an index row"*) there is
**no BACKLOG.md edit** for this delivery. Append one dated entry directly to `docs/HISTORY.md`
(mirroring the smaller, non-milestone-table entries already there, e.g. "2026-07-27 — Documentation
reference & naming convention adopted (C-322, doc-only)"):

```markdown
## <delivery date> — M9: Cypher MCP tool surface — direct in-tool graph discovery (C-901, C-902) ✅

`mcp__cypher__query` gains a fourth directive, `GRAPHS` (alongside `EXPLAIN`/`PROFILE`), that lists
every FalkorDB graph on the instance directly — no deliberately-wrong graph name, no `redis-cli`.
No new parameter; `graph`/`cypher` stay the only two required ones. Same blast radius as what the
"graph not found" error already leaked (`client.list_graphs()`, unfiltered). `docs/requirements/
cpg-query-access.md` FR-2 gains a header-pointer annotation to `docs/plans/
cypher-mcp-tool-surface.md`, alongside its existing `generic-cypher-mcp.md` FR-1 pointer.
AC-1…AC-3 verified: [describe the concrete verification actually run, live and offline].
```

Fill the bracketed clause with what was actually run (this plan cannot pre-write it — see §5).
Owner: whoever closes this out (`coder`, or `cobb` if a separate docs pass is preferred — mirrors
the M3/M4 precedent where the implementer closing a small, self-contained delivery also writes its
own `HISTORY.md` entry).

---

## 5. Test strategy

### 5.1 Offline unit tests (`cypher-mcp/tests/test_server.py`) — the concrete list

Add to the existing `test_split_directive_classification` parametrize table (extends the file's
established pattern, does not replace any existing case):

```python
        # GRAPHS directive
        ("GRAPHS", "graphs", ""),
        ("graphs", "graphs", ""),
        ("  GrApHs  \n", "graphs", ""),
        ("/* c */ GRAPHS", "graphs", ""),
        ("// c\nGRAPHS", "graphs", ""),
        # standalone only — real text after it is NOT the directive
        ("GRAPHS MATCH (n) RETURN n", "query", "GRAPHS MATCH (n) RETURN n"),
        # \b boundary decoy, same protection class as EXPLAIN_ME/PROFILER
        ("GRAPHS_LIST", "query", "GRAPHS_LIST"),
        # singular is deliberately NOT the directive
        ("GRAPH", "query", "GRAPH"),
```

New, standalone tests (placed near the existing `run_query`-path tests, ~after
`test_profile_still_refused_regardless_of_agent`):

1. `test_graphs_directive_lists_loaded_graphs_sorted_and_ignores_graph_param(fake_client)` —
   `fake_client(graphs=["ws:acme", "cpg_falkorchat", "kaizen_team"])`; call
   `server.run_query("this-value-is-ignored", "GRAPHS")`; assert the exact output
   `"graphs=3\ncpg_falkorchat\nkaizen_team\nws:acme"` (proves sorting); assert
   `client.calls == [("list_graphs",)]` — i.e. `select_graph` was **never** called, proving `graph`
   truly plays no role (AC-1's "no deliberately-wrong graph name" guarantee, verified structurally).
2. `test_graphs_directive_reports_zero_when_none_loaded(fake_client)` — `fake_client(graphs=[])`;
   `server.run_query("", "GRAPHS") == "graphs=0\n(none loaded)"`.
3. `test_graphs_directive_is_case_insensitive_through_run_query(fake_client)` — one representative
   case through the full `run_query` path (e.g. `"  GrApHs  "`), not just `split_directive`, so the
   integration is pinned once end-to-end, not only at the classifier.
4. `test_graphs_directive_reports_unreachable_when_list_graphs_fails(fake_client)` — using the new
   `list_graphs_error` fixture kwarg (§4 step 2):
   ```python
   from redis.exceptions import ConnectionError as RedisConnectionError

   def test_graphs_directive_reports_unreachable_when_list_graphs_fails(fake_client):
       fake_client(list_graphs_error=RedisConnectionError("Error 111 connecting"))
       out = server.run_query("", "GRAPHS")
       assert out.startswith("FalkorDB unreachable at")
   ```
   Proves the `"graphs"` branch needs no bespoke error handling — a `list_graphs()` failure
   propagates to the same outer `except Exception` / `explain_error()` path every other failure
   mode already uses.
5. **Regression check, no new test needed but must be confirmed:** run
   `test_input_schema_has_two_required_params_and_one_optional_agent` and
   `test_exactly_one_tool_named_query` unedited — both must still pass verbatim. This is the
   concrete evidence FR-2 holds (zero new tool, zero new parameter).

### 5.2 Live test (`@pytest.mark.live`)

```python
@pytest.mark.live
def test_live_graphs_directive_matches_list_graphs(live_graph):
    client = server.get_client()
    expected = sorted(client.list_graphs())
    out = server.run_query("ignored", "GRAPHS")
    lines = out.splitlines()
    assert lines[0] == f"graphs={len(expected)}"
    assert lines[1:] == expected
```

Uses the existing `live_graph` fixture only to guarantee at least one graph exists at test time; the
assertion itself is against the real, shared instance's **full** list (`cpg_*`, `ws:*`, `reference`,
`kaizen_team`, any scratch graphs) — this is deliberately AC-2's exact scenario (the tool's answer
must match `redis-cli GRAPH.LIST`'s output at the same point in time). Reading the list mutates
nothing, so this is safe against the shared instance under the same rule that already lets
`test_live_missing_graph_does_not_materialise_a_key` read `client.list_graphs()` directly.

### 5.3 Container gate

Run the existing "in-container test gate" (`cypher-mcp/README.md`, quoted in §4 step 3) after step 1
lands — `cypher-mcp/build.sh`, then both `docker run` invocations, then the `redis-cli GRAPH.LIST`
residue check. No script changes needed; the image tag content-hashes `server.py` and `tests/`
automatically.

### 5.4 Acceptance mapping

- **AC-1** (one direct tool call, no deliberately-wrong name, no shell) — 5.1#1, 5.2.
- **AC-2** (matches `redis-cli GRAPH.LIST` exactly, same point in time) — 5.2 (compares directly
  against `client.list_graphs()`, the same call `redis-cli GRAPH.LIST` reports).
- **AC-3** (no reader finds `cpg-query-access.md`/`README.md` disagreeing) — §4 steps 3 and 4; verify
  by re-reading both files after the edits, plus the repo-wide grep already run in §2's findings
  (only one location in each file made the now-stale claim).

---

## 6. Risks & open questions

- **Naming collision risk is judged, not merely assumed.** No real Cypher statement opens with the
  bare word `GRAPHS` (every real statement opens with a clause keyword —
  `MATCH`/`CREATE`/`MERGE`/`WITH`/`UNWIND`/`CALL`/`RETURN`, …), so misclassifying an intended real
  query is not a realistic failure mode. This is the same argument that already protects `EXPLAIN`/
  `PROFILE`, applied to a new keyword — not re-verified live against FalkorDB specifically for
  `GRAPHS` (there is nothing to verify: the directive never reaches the server when recognized, and
  when *not* recognized it falls through to ordinary query handling, unchanged).
  A future need to also expose the plural form as a real Cypher identifier (unlikely — the tool's
  domain is FalkorDB graphs, and `GRAPHS` is not a name anything in this repo's schemas uses) would
  surface as a `"query"`-classified statement returning FalkorDB's own syntax/semantic error, not a
  silent misbehavior — fails safe either way.
- **The `graph` parameter is still required but semantically unused for this one call shape** — a
  minor ergonomic wrinkle (a caller must pass *some* string, conventionally `""`), accepted because
  removing it would touch FR-4 of the *original*, unrelated `cpg-query-access.md` contract (graph
  name always caller-supplied, never defaulted) which this delivery does not reopen. If this wrinkle
  proves genuinely annoying in practice, the next revisit is a separate, small design question — not
  a blocker to this delivery.
- **Sort order vs. AC-2's "matches exactly."** §3.1 sorts the rendered list for determinism, while
  `GRAPH.LIST`'s own order is not documented as stable. This plan treats AC-2's equivalence as
  membership (the *set* of names), consistent with the precedent `cpg-query-access.md`'s AC-3
  reconciliation (D5) already set for this exact class of question (rendering choices don't count
  against equivalence when the underlying values match) — flagging this explicitly in case a
  stricter reading of AC-2 (byte-identical ordering) was intended; nothing in the requirements doc's
  Decision Log suggests it was, but it was not asked either.
- **`docs/HISTORY.md`'s closeout entry (§4 step 5) has one bracketed placeholder** the plan
  deliberately cannot fill in advance — the exact verification transcript belongs to whoever runs
  it, not to the plan.
