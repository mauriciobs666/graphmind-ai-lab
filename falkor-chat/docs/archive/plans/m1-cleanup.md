# M1-final cleanup batch — implementation spec

> Status: 🔵 proposed · scope: ~half-day, four small items. Planning doc only — no code changed here.
> Source backlog: `docs/BACKLOG.md` parking lot (the four items map 1:1).
>
> **Bottom line up front:** every server change is **adapter-only** (`mcp.py`, `api.py`) plus tests.
> **No change touches `repository.py`, `docs/QUERIES.md`, or the 92-assertion `test_queries.sh` suite.**
> The two web items touch `web/app.js` only. Both forks below are resolved toward the cheaper,
> schema-honest option, so the locked §4/§9 queries stay untouched.

---

## 0. Decision record (the two forks)

### Fork 3(a) — dead `isMention` highlight in `web/app.js`

**Decision: (i) remove the dead highlight from the JS.** Keep the `.mention` CSS rule for a future
M2 real-time/since-read wiring, but stop the renderer from reading a field the endpoint never sends.

- **Why.** `GET /threads/{tid}/messages` → `services.read_thread` → `repository.read_thread`
  (`QUERIES.md` §4 "Read a full thread") is deliberately **reader-agnostic**: it takes no `me_id`
  and returns no `isMention`. `isMention` is a **since-read (§9)** concept — the only queries that
  compute it (`read_thread_since`/`read_ws_since`) take `me_id` and are reached only via the MCP
  `read_messages` tool, not the REST thread read the web UI uses. So `m.isMention` is always
  `undefined` in the browser; the highlight is dead (harmless-falsy, but dead).
- **Rejected: (ii) make §4 return a per-reader `isMention`.** This would (a) mutate the **locked**
  canonical §4 query in `QUERIES.md` (single source of truth) and its `repository.read_thread`
  1:1 mate; (b) add a per-reader `OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me) WHERE me.userId=$meId
  OR me.agentId=$meId` to the **hot thread-read path**, costing an extra traversal on every thread
  open (AGENTS.md rule 6 — RAM/cost on the hot path); (c) require threading `ctx.actor` into
  `read_thread` (currently a thin passthrough); and (d) force a matching assertion change in the
  92-suite. All of that to restore a cosmetic highlight on a request/response M1 UI. Not worth it.
  The mention-highlight belongs with the since-read path that already carries the flag — revisit
  in M2 when the UI polls `read_messages`-style since-reads.

### Fork 4 — `GET /threads/{thread_id}/messages/{msg_id}` ignores `thread_id`

**Decision: replace the nested route with a flat `GET /messages/{msg_id}`.** Drop the
thread-scoped spelling entirely.

- **Why.** `services.get_message(ctx, msg_id)` → `repository.get_message` is an inherently
  **workspace-global** lookup: `Message.msgId` is unique per workspace and `Message` has **no
  `threadId` property** (locked schema). The `thread_id` path segment is therefore a contract the
  data model cannot cheaply honor — today a message from thread A resolves under thread B's URL.
- **Rejected: validate thread membership.** The only way to check "is `msg_id` in `thread_id`" is a
  reverse/forward `HEAD`/`NEXT` traversal that is **O(thread length)** — a hot-path cost on a route
  the **web UI does not use at all** (grep of `web/app.js`: no nested single-message fetch), purely
  to keep a URL shape. The denormalised `Message.threadId` that would make it O(1) is a schema
  change already parked in `docs/BACKLOG.md` (RAM rule 6) and out of scope for this cleanup.
- **Rejected: leave the route as-is.** It ships a false contract (wrong-thread resolution) — a latent
  correctness trap for any future consumer.
- A flat `GET /messages/{msg_id}` states the truth: msgId is globally unique in the workspace.

---

## 1. Item 1 — expose `search` as a 4th MCP tool

**Files:** `server/falkorchat/mcp.py` (add tool) · `server/tests/test_mcp.py` (edit discovery + add roundtrip).
**Repository/QUERIES/92-suite:** untouched (service + repo method already exist).

### Interface (add to `mcp.py`, after `create_thread`)

```python
@mcp.tool()
def search_messages(query: str, limit: int = 50) -> list[dict[str, Any]]:
    """Full-text keyword search over this workspace's messages (QUERIES.md §5).

    Returns matches newest-relevance first (RediSearch score). Invalid query
    syntax (unbalanced quotes, stray operators) surfaces as a tool error.
    """
    ctx = _get_context()
    return _svc().search_messages(ctx, query=query, limit=limit)
```

- **Naming rationale.** Tool name `search_messages` mirrors the service method and the `read_messages`
  tool (verb_noun). The `re`/`frm` terse aliases on `send_message` exist only because "re:"/"from:"
  are messaging idioms; search has no such idiom, so plain `query`/`limit` is clearer for the agent
  reading the tool schema. `limit` default `50` matches the service default and the REST `Query`
  default.
- **Error behavior.** `InvalidSearchQueryError` propagates out of the tool exactly like
  `UnknownMemberError` does from `send_message` today (FastMCP surfaces it as a tool error). No
  new handling needed — matches the existing thin-adapter pattern.
- **Return wrapping.** Returns a `list[dict]`, so `call_tool` wraps it as `{"result": [...]}` — the
  existing `_unwrap` helper in `test_mcp.py` already handles this (same as `read_messages`).

### Tests (edit `test_mcp.py`)

1. Update the discovery test (see Item 2 — it becomes the combined 5-tool assertion).
2. Add a roundtrip mirroring the REST search test (`test_api.py::test_search_returns_matching_messages`):

```python
def test_search_messages_tool_finds_posted_text(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    svc = _configure(repo)
    ch = svc.create_channel(TEST_CTX, name="general")

    async def scenario():
        th = _unwrap(await mcp_mod.mcp.call_tool(
            "create_thread", {"channel_id": ch["channelId"], "title": "hi"}
        ))
        tid = th["threadId"]
        await mcp_mod.mcp.call_tool("send_message", {"body": "hello world", "re": tid})
        await mcp_mod.mcp.call_tool("send_message", {"body": "goodbye moon", "re": tid})
        return _unwrap(await mcp_mod.mcp.call_tool(
            "search_messages", {"query": "hello"}
        ))

    hits = asyncio.run(scenario())
    assert [h["text"] for h in hits] == ["hello world"]
```

**Done when:** the tool appears in `list_tools()` and the roundtrip returns the matching row.

---

## 2. Item 2 — expose `create_channel` as a 5th MCP tool (Q#4)

**Files:** `server/falkorchat/mcp.py` (add tool) · `server/tests/test_mcp.py` (discovery + roundtrip).
**Repository/QUERIES/92-suite:** untouched (`services.create_channel` already exists).

### Interface (add to `mcp.py`)

```python
@mcp.tool()
def create_channel(name: str) -> dict[str, Any]:
    """Create a channel so an agent is self-sufficient (can set up its own space)."""
    ctx = _get_context()
    return _svc().create_channel(ctx, name=name)
```

- Returns a dict → `call_tool` returns it as structured content directly (like `create_thread`);
  `_unwrap` handles it.
- No validation branch: `create_channel` is an idempotent MERGE with no precondition (unlike
  `create_thread`'s channel-existence check), so nothing can 4xx here.

### Tests (edit `test_mcp.py`)

1. Replace the discovery assertion (now covers Items 1 + 2):

```python
def test_tool_discovery_lists_all_tools(repo):
    _configure(repo)
    tools = asyncio.run(mcp_mod.mcp.list_tools())
    assert {t.name for t in tools} == {
        "send_message", "read_messages", "create_thread",
        "search_messages", "create_channel",
    }
```

2. Add a roundtrip that creates a channel over MCP, then uses it to create a thread + post — proving
   an agent is now self-sufficient without any REST seeding:

```python
def test_create_channel_tool_enables_full_agent_flow(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)

    async def scenario():
        ch = _unwrap(await mcp_mod.mcp.call_tool("create_channel", {"name": "general"}))
        th = _unwrap(await mcp_mod.mcp.call_tool(
            "create_thread", {"channel_id": ch["channelId"], "title": "hi"}
        ))
        await mcp_mod.mcp.call_tool("send_message", {"body": "hi", "re": th["threadId"]})
        return _unwrap(await mcp_mod.mcp.call_tool(
            "read_messages", {"re": th["threadId"], "since": 0, "advance": False}
        ))

    rows = asyncio.run(scenario())
    assert [r["text"] for r in rows] == ["hi"]
```

> Note the existing `test_create_thread_send_and_read_roundtrip` comments "channel is REST-only;
> seed it directly through the service" — after this item that limitation is gone. Leave that test
> as-is (still valid) or drop the stale comment; not required.

**Done when:** discovery lists 5 tools and an agent can create channel → thread → message end-to-end
over MCP alone.

---

## 3. Item 3(a) — remove the dead mention highlight (`web/app.js`)

**Files:** `web/app.js` only. No server, no docs beyond BACKLOG/HISTORY.

In `renderMessages` (line ~107), drop the `isMention` class toggle:

```js
// before
el.className = "msg" + (m.isMention ? " mention" : "");
// after
el.className = "msg";
// NOTE: reader @-mention highlighting is a since-read (§9 isMention flag) concern,
// not exposed by the §4 thread read this view uses. Wire it in M2 with real-time reads.
```

Leave the `.mention` CSS rule in `index.html`/stylesheet in place (harmless, ready for M2).

**Done when:** no `m.isMention` reference remains in `app.js`; thread rendering is visually unchanged
(the highlight never fired anyway).

---

## 4. Item 3(b) — `parseMentions` must not let one bad `@token` nuke the send

**Files:** `web/app.js` only.

**Problem.** `parseMentions` (regex `@([\w:-]+)`) turns **every** `@token` into a mention id and
puts them all in the `mentions[]` array. The server validates members and returns **400
`UnknownMemberError`** if any one doesn't resolve — so a single unknown handle (or a stray "@"
in prose) drops the entire message. There is **no members-list endpoint in M1**, so the client
cannot pre-resolve handles.

**Decision (minimal, honest): drop unresolvable-but-still-optimistic — send tokens, but degrade
gracefully on the known failure instead of blanket-failing.** Concretely, two small changes in the
composer submit handler; pick per taste — recommend both:

1. **Keep parsing `@tokens` as mention candidates** (M1 has no roster to validate against, so the
   server remains the authority), **but** on a `400 UnknownMemberError`, **retry once without
   mentions** and surface a non-blocking notice, so the message still posts:

```js
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("message-text").value.trim();
  if (!text || !state.threadId) return;
  guard(async () => {
    const mentions = parseMentions(text);
    try {
      await postMessage(text, mentions);
    } catch (err) {
      // The server rejects unknown members (no roster endpoint in M1 to pre-check).
      // Don't lose the message: resend as plain text and tell the user the @-handles
      // didn't resolve.
      if (mentions.length && /UnknownMemberError/.test(err.message)) {
        await postMessage(text, []);
        alert("Message sent, but these @-handles weren't recognised and were not " +
              "linked as mentions: " + mentions.join(", "));
      } else {
        throw err;
      }
    }
    $("message-text").value = "";
    await loadMessages();
  });
});

async function postMessage(text, mentions) {
  return api(`/threads/${state.threadId}/messages`, {
    method: "POST",
    body: JSON.stringify(mentions.length ? { text, mentions } : { text }),
  });
}
```

- **Why this shape.** The server already owns member validation (correctly — it has the graph); the
  client shouldn't guess. The one failure mode we care about — "message vanished because of a typo'd
  handle" — is caught and recovered by a single plain-text retry, and the user is told which handles
  didn't link. `error` field in the 400 body is `UnknownMemberError` (see `test_api.py::
  test_post_message_mention_parity`) and `api()` already threads it into `err.message` as
  `"400: ..."`, so the `/UnknownMemberError/` test on the message string is reliable. No new
  endpoint, no roster, ~15 lines.
- **Rejected: client-side allowlist / member fetch.** No `GET /members`-style endpoint exists in
  M1; adding one is scope creep (new REST route + service + repo + query). Rejected.
- **Rejected: strip all `@tokens` from mentions client-side.** That silently disables the working
  mention feature for valid members — worse than the retry, which keeps valid mentions on the happy
  path and only degrades on actual failure.

**Done when:** posting text with an unknown `@handle` still creates the message (as plain text) and
the user sees a notice; posting with only valid handles still links mentions as before.

> No JS test harness exists in the repo — verify these two web items manually against a running
> server (see the checklist). Keep the diff small and self-evident.

---

## 5. Item 4 — replace nested single-message route with flat `GET /messages/{msg_id}`

**Files:** `server/falkorchat/api.py` (swap route) · `server/tests/test_api.py` (update 2 tests).
**Repository/QUERIES/92-suite:** untouched (`get_message` query and repo method are unchanged).

### Change in `api.py`

Replace the nested route (lines 65–72) with a flat one:

```python
@router.get("/messages/{msg_id}")
def get_message(msg_id: str, ctx: CallContext = Depends(get_context)):
    msg = services.get_message(ctx, msg_id=msg_id)
    if msg is None:
        raise HTTPException(status_code=404, detail="message not found")
    return msg
```

- Route registration order is irrelevant to correctness here (both `/messages/{id}` and
  `/threads/{tid}/messages` are concrete prefixes), but keep it grouped with the other message
  routes. The static `/` catch-all still mounts last in `app.py` — unaffected.

### Tests (edit `test_api.py`)

- `test_post_and_read_messages`: change the single-message fetch from
  `client.get(f"/threads/{tid}/messages/{mid}")` to `client.get(f"/messages/{mid}")`.
- `test_get_missing_message_404`: change `client.get(f"/threads/{tid}/messages/nope")` to
  `client.get("/messages/nope")` (the channel/thread setup is now unnecessary — a bare
  `client.get("/messages/nope")` asserting 404 suffices).
- Optional but recommended new assertion: a message posted in thread A is fetchable at
  `/messages/{mid}` (proving the flat lookup), documenting that resolution is workspace-global by
  design.

**Done when:** `GET /messages/{mid}` returns the message, `GET /messages/nope` is 404, and the old
nested path is gone (no test references it).

---

## 6. Repository / QUERIES.md / 92-suite impact — explicit

| Item | `repository.py` | `docs/QUERIES.md` | `test_queries.sh` (92) |
|---|---|---|---|
| 1 search MCP tool | none | none | none |
| 2 create_channel MCP tool | none | none | none |
| 3(a) dead highlight | none | none | none |
| 3(b) parseMentions | none | none | none |
| 4 flat message route | none | none | none |

**No item touches Cypher, the query library, or the 92-assertion suite.** (This is only true because
fork 3(a) was resolved toward *removing* the dead highlight. Had 3(a)(ii) been chosen — adding a
per-reader `isMention` to the §4 read — it *would* change `repository.read_thread`, `QUERIES.md` §4,
and a `test_queries.sh` assertion, and add hot-path cost. It was rejected; see §0.)

---

## 7. Work split for routing to implementers

### Batch A — server-side, TDD (pytest) → `tdd-engineer`

Adapter-only Python + tests; run `cd server && .venv/bin/python -m pytest -q` (needs FalkorDB up).

- Item 1 — `search_messages` MCP tool + test.
- Item 2 — `create_channel` MCP tool + discovery/roundtrip tests.
- Item 4 — flat `GET /messages/{msg_id}` route + two edited api tests.

These three are independent and can land in one commit. Red-first is natural: edit the discovery
test to expect 5 tools (red) → add the two tools (green); edit the api tests to the flat path (red)
→ swap the route (green).

### Batch B — web JS (no test harness) → `coder` (manual verification)

- Item 3(a) — remove dead `isMention` class toggle.
- Item 3(b) — mention-failure graceful retry in the composer submit handler.

No automated JS tests in the repo; verify manually per the checklist. Keep the diff minimal.

> Batches A and B are fully independent (different files, no shared surface) and can proceed in
> parallel.

---

## 8. Documentation updates required (repo documentation rule)

Every behavior change updates docs in the **same change**:

**Batch A (MCP tools — Items 1 & 2):**
- `docs/DESIGN.md` §15.2 "Tools → service → query" table — add two rows:
  `search_messages(query, limit=50)` → `search_messages` → §5; and
  `create_channel(name)` → `create_channel` → §3 create a channel.
- `README.md` (lines ~210–211) — extend the MCP tools list from
  `send_message, read_messages, create_thread` to include `search_messages` and `create_channel`.
- `README.md` roadmap M1 row / test count (line 156) and `docs/DESIGN.md` §14.6 if the "68 tests"
  figure is bumped — update to the new passing count (see §9).

**Batch A (Item 4 — flat route):**
- `docs/DESIGN.md` §14.4 "REST surface → service → verified query" table — change the
  `GET /threads/{tid}/messages/{mid}` row to `GET /messages/{mid}` → `get_message` → §4 get a single
  message.

**Both batches:**
- `docs/HISTORY.md` — one dated entry (e.g. `2026-07-02 — K-005: M1-final cleanup`) summarising
  the four items, the two fork decisions, and the new test count.
- `docs/BACKLOG.md` — remove the four completed parking-lot items (search-over-MCP, create_channel-
  over-MCP, web mention polish, nested-message-route), and update "Last reviewed".

**No changes to** `docs/QUERIES.md`, `AGENTS.md` (no new scripts/facts/rules), or
`scripts/*` (no schema/query change).

---

## 9. Verification checklist

**Server (Batches A):**
```bash
cd falkor-chat/server                  # from the repo root
.venv/bin/python -m pytest -q          # expect green; 68 → ~71
```
Expected count movement (honest estimate — implementer confirms):
- Item 1: +1 (search roundtrip); discovery test edited (net 0).
- Item 2: +1 (create_channel flow); discovery test now asserts 5 tools.
- Item 4: 2 api tests edited (net 0), optional +1 flat-lookup test.
- So **68 → 70 or 71** depending on the two optional tests. Update the "68" figure in
  `README.md` and any doc that cites it to the observed number.

**Query suite (must stay green, unchanged by this batch — a regression guard):**
```bash
cd falkor-chat                         # from the repo root
./scripts/test_queries.sh              # expect 92/92 (untouched)
```

**MCP discovery (optional live smoke, server running):**
```bash
# tools list should now contain 5 names incl. search_messages, create_channel
```

**Web (Batch B — manual, server at http://localhost:8000/):**
1. Open a thread, post a normal message → renders (no visual change from before).
2. Post a message containing an unknown handle, e.g. `hi @nobody` → message **still appears**;
   an alert notes `@nobody` wasn't linked. (Before: 400, message lost.)
3. Post `hi @u1` (a real member) → message posts with the mention linked (no alert).
4. Confirm no console error referencing `isMention`.

**Definition of done for the batch:** pytest green at the new count; `test_queries.sh` 92/92;
5 MCP tools discoverable; web mention-failure no longer drops messages; all docs in §8 updated in
the same change.

---

## Ready to implement

- **Batch A (tdd-engineer):** add `search_messages` + `create_channel` MCP tools in
  `server/falkorchat/mcp.py`; swap the nested single-message REST route for flat
  `GET /messages/{msg_id}` in `server/falkorchat/api.py`; edit `server/tests/test_mcp.py`
  (5-tool discovery + 2 roundtrips) and `server/tests/test_api.py` (2 route edits). Then update
  `docs/DESIGN.md` §15.2 + §14.4 and `README.md` MCP tools list + test count.
- **Batch B (coder, manual verify):** in `web/app.js`, remove the dead `isMention` class toggle in
  `renderMessages`, and make the composer submit handler retry a mention-rejected send as plain text
  with a user notice.
- **Both:** log in `docs/HISTORY.md`, prune `docs/BACKLOG.md`. No `repository.py`/`QUERIES.md`/
  `test_queries.sh` changes — 92/92 must stay green as a regression guard.
</content>
</invoke>
