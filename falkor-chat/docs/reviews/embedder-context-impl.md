# Embedder context and MCP `send_message` cap — live-spec check (feature close)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-099

## Scope & verdict

This is close step 3 (root `AGENTS.md`, live-spec bullet) for K-099. I checked the uncommitted
`falkor-chat/docs/SERVER.md` diff, as far as it covers the `send_message` text cap, and the
writer's claim that SERVER/DESIGN carry no embedder-context statement. The shipped code checked
is `6d4f5c05` (the service cap), `0402d9c9` and `f3e0015e` (`scripts/ensure_demo_models.py`), plus
`7490f634` because it changed the MCP path the cap applies to. Everything was read at HEAD.
The bounds review as a whole is `falkor-chat/docs/reviews/mcp-input-bounds-impl.md`.

**Verdict: needs changes.** F1 is the blocker.

CPG: considered, not relevant — the check covers a handful of named symbols and one operator script, all read directly and executed.

## Findings

### F1 — blocker: the live spec says nothing about the embedder's 2048-token load context or its consequence

The writer's claim is **true**: there is no embedder-context statement in `SERVER.md`,
`DESIGN.md` or `QUERIES.md`. I verified it with a sweep for `2048|4096|8192|n_ctx|context length`;
the only hits are the unrelated "4,096 vectors" in DESIGN §11. The inference drawn from it, that
nothing needs updating, does not follow.
The shipped diff moved the embedder from 8192 to its own `EMBED_CONTEXT_LENGTH = 2048`
(`scripts/ensure_demo_models.py:102`; it was 8192 before `0402d9c9`, per that diff's removed
`lms load <id> -c 8192` line). At 2048, a dense message under `MAX_TEXT_LEN` (8000 chars) can
exceed the window, and LM Studio then silently embeds only its head. That behavior is recorded
in `docs/test-reports/embedder-context-report.md` and the `MessageTooLongError` docstring, and
K-100 is open to fix it. The result is an externally observable change to what GraphRAG retrieval
can find in a long message. It is also a live constraint, so root `AGENTS.md` places it on the
design surface that owns it: `DESIGN.md` §1 is this component's decision register
(`falkor-chat/AGENTS.md`), and §1.3 already carries the embedder row and the "embedder + one chat
LLM co-resident" VRAM budget. Today only `falkor-chat/AGENTS.md` (context, not spec), a docstring
and the backlog state it.
**Fix:** add one clause to DESIGN §1.3's *Embedding model* row, or a sibling row: loaded at a
2048-token context by `scripts/ensure_demo_models.py` (the chat model keeps 8192), for VRAM;
LM Studio silently embeds only the first 2048 tokens of a longer input, so a dense message within
`MAX_TEXT_LEN` can be embedded by its head only (chunking: K-100). Cite the test report for the
probe. Write no history: drop the 4096 A/B and the 8192 predecessor, which belong in `HISTORY.md`.

### F2 — nit: `MessageTooLongError` is listed in the REST error envelope, but REST cannot raise it

`SERVER.md` §1.4 *Error envelope* now names `MessageTooLongError` among the `400`s. The mapping
is true (`app.py` generic `ServiceError` handler). No HTTP route reaches it, though: dev REST
answers `422` first (`PostMessageIn.text` `max_length=MAX_TEXT_LEN`; verified, 8001 chars →
`422`), and the storefront caps at 2000 (`storefront_api.py` lists it in
`SERVICE_ERRORS_UNREACHABLE`). The MCP signature now pre-empts it as well (verified: 8001 chars
gives a pydantic `isError`, "String should have at most 8000 characters", and the repository is
untouched). Only in-process callers see it. This is harmless, and *Bounds* states the 422-first
order correctly. Leave it, or move it out of the "e.g." list.

## The `send_message` cap text: verified

- §1.4 *Bounds*: message text is non-blank and ≤ 8000 (`MAX_TEXT_LEN`). Over-long raises
  `MessageTooLongError` and whitespace-only raises `InputBoundError`, both checked before any
  repository call (`Services.post_message`). True, verified with a fake repository that raises on
  access.
- §2.2: `send_message` `body` is 1–8000 in the `inputSchema` (`list_tools()` shows
  `minLength 1, maxLength 8000`). A violation is refused before the tool body runs, so nothing
  is posted or scheduled, and the text names the argument and the limit. True, verified.
- `tests/test_services.py`, `tests/test_input_bound_parity.py` and `tests/test_mcp.py` all pass
  (480 tests). These do not wipe `reference`; the full suite was not run, as the brief required.
- `falkor-chat/AGENTS.md`'s two copies ("the embedder at 2048", "an embedder at another context
  … only warns") match the script (`:102`, header step 6).
- Live-spec form: no history or `K-` phrasing in the added text, and numbering is unchanged.

## What's solid

The spec now places the text cap correctly, in the service for every front door, and states the
REST `422`-first ordering accurately. That makes the K-099 item-3 contract fully described.

## Open questions

None. Out of scope, but noticed: `falkor-chat/docs/BACKLOG.md` K-099 still reads
"Remaining: (1)…(4)". Those four items are delivered, so the item belongs in `teco`'s close.

### Proposed REVIEW.md lines

Target: `falkor-chat/REVIEW.md`, *Always check*. Source: F1. The spec writer saw no code-side
spec contract, because the change lived in operator tooling.

- *Addition:* A change to how a model is loaded (context length, offload, quantization) in
  `scripts/` is checked for its effect on spec'd behavior (`DESIGN.md` §1.3, e.g. what an embed
  or a turn can cover), not only for the script's own contract.

There are no prunes.

## Pass 2

**Verdict: needs changes**, on the `send_message`-cap half only. The embedder half is fixed.

- **F1:** fixed. `DESIGN.md` §1.3's *Embedding model* row now gives the 2048 load context
  (`EMBED_CONTEXT_LENGTH` = 2048, `ensure_demo_models.py:102`; the chat model's
  `DEMO_CONTEXT_LENGTH` = 8192, `:101`), the reason (VRAM) and the silent head-only embed. That
  last claim matches `docs/test-reports/embedder-context-report.md` ("silent head truncation, not a
  rejection"); I did not re-probe, since LM Studio was out of bounds. The row carries no history:
  no 4096 and no predecessor. It does restate the "8000-char" value next to its `SERVER.md` §1.4
  citation, a nit; the citation alone would do.
- **F2:** superseded. `MessageTooLongError` left the envelope list for §1.4 *Bounds*, but its new
  wording ("only an in-process caller sees it, since both front doors refuse it first") is false.
  None of the shipped in-process message writers can reach it, and the agent answers they write
  are not capped at all. That is blocker N1 in `falkor-chat/docs/reviews/mcp-input-bounds-impl.md`
  Pass 2, and it is the K-099 cap's own scope.
- **Growth-stops:** the only growth is the §1.3 row, which is a new fact (new scope). The
  `SERVER.md` rework shrank, at 56 inserted lines against 59 in pass 1.

## Pass 3

**Verdict: approve.**

- **N1 (`MessageTooLongError` scope):** fixed. Verified by symbol in
  `falkor-chat/docs/reviews/mcp-input-bounds-impl.md` Pass 3: three `post_message` callers all
  refuse first, `post_agent_answer` is uncapped, and no shipped caller reaches the error. The
  spec now says so.
- **Nit, restated "8000-char" in §1.3:** fixed. It now reads "within the message-text cap
  (`SERVER.md` §1.4 Bounds)", so the value has one home. The rest of the row is unchanged from
  Pass 2.
- **Growth-stops:** passes. The row's edit replaced a value with a citation and added nothing.
