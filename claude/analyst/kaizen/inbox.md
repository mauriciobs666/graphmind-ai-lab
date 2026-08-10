# Kaizen — Learnings Inbox: analyst

> Append-only capture of durable, non-obvious environment facts the `analyst` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

## 2026-08-09 — A "held, do not clear until X lands" kaizen-inbox note can go stale silently when X lands via a *different* agent's kaizen files than the one holding the note

- **Evidence:** Reviewing the cross-agent kaizen-inbox-distillation batch (`docs/reviews/kaizen-inbox-distillation.md`), `claude/architect/kaizen/inbox.md` carried two entries explicitly gated "do not clear until [a consolidated `skills/agent-standards/kiro.md` follow-up] lands." The follow-up *had* landed (confirmed via `git diff -- skills/agent-standards/kiro.md`, part of the same uncommitted batch) — but only `claude/analyst/kaizen/history.md`'s own entry ("Held entry 28 promoted: consolidated Kiro-facts edit landed") documented the closure. Neither `claude/architect/kaizen/inbox.md`/`history.md` nor the owning agent's (`cobb`'s) `kaizen/history.md` were updated to match, even though the same physical edit closed out both sides. Reading only the architect-side files would have made this look like a still-pending, not-yet-due item (the brief's own framing anticipated exactly this ambiguity).
- **Context:** cross-checking a "held pending a shared follow-up" note during a multi-agent kaizen-bookkeeping review — the note lived in one agent's inbox but the actual close-out evidence lived in a *sibling* agent's history file for the same batched edit.
- **Suggested home:** prompt (Guardrails — "Evidence over vibes" bullet, or a dedicated multi-agent-bookkeeping clause): when a document under review claims "held/pending until a shared/coordinated follow-up lands," don't take the holding document's own state as authoritative — grep sibling agents'/owners' kaizen history for the same target file/date to check whether the follow-up already landed elsewhere and just wasn't mirrored back.

## 2026-08-10 — `urllib` failure taxonomy: `HTTPError ⊂ URLError`, but a **read** timeout raises a bare `TimeoutError` that is *not* a `URLError`

- **Evidence:** Executed with `falkor-chat/server/.venv/bin/python` (3.12): `issubclass(urllib.error.HTTPError, urllib.error.URLError)` → `True`; `socket.timeout is TimeoutError` → `True`; `issubclass(TimeoutError, urllib.error.URLError)` → **`False`**. A real `urllib.request.urlopen(req, timeout=0.5)` against a local `http.server` that sleeps 3 s raised `TimeoutError` with MRO `(TimeoutError, OSError, Exception, BaseException)` — no `URLError` anywhere. Two consequences for any stdlib-only HTTP client: (a) an `except URLError` clause placed before `except HTTPError` makes the HTTP-status branch dead code and discards the response body; (b) a client catching only `URLError`/`HTTPError` lets every read timeout escape unclassified. Also relevant: a schemeless URL (`"host:1234/x"`) makes `urlopen` raise `ValueError: unknown url type`, which is in neither branch.
- **Context:** Gate review of `falkor-chat` K-042's plan, whose new `transport.py` enumerates its four failure classes as "1. URLError, timeout; 2. HTTPError" — an order that, transcribed into `except` clauses, is a bug, and a list that omits the timeout type the plan itself newly introduces (FR-14 per-model timeouts).
- **Suggested home:** knowledge base (`skills/python-web-quirks` — it is stdlib-generic, not falkor-chat-specific)

## 2026-08-10 — LM Studio answers a **missing `/v1` prefix** with HTTP 200 + an error envelope, on chat *and* embeddings — and the `error` value's JSON shape differs between the two prefixes

- **Evidence:** `curl -s -w '%{http_code}' -X POST … -d '{}'` against the box's LM Studio on `localhost:1234`, 2026-08-10: `POST /chat/completions` → **200** `{"error":"Unexpected endpoint or method. (POST /chat/completions)"}`; `POST /embeddings` → **200** `{"error":"Unexpected endpoint or method. (POST /embeddings)"}`; `POST /v1/chat/completions` → 400 `{"error": {"message": "No models loaded…"}}`; `POST /v1/embeddings` → 400 `{"error":"No models loaded…"}`. So (a) the wrong-prefix failure is *not* an HTTP error on either endpoint — an OpenAI-shaped client sees `KeyError: 'choices'` / `KeyError: 'data'`; and (b) `error` is a **string** on the wrong-prefix path but an **object** on the chat `/v1` path, so a classifier written as `body["error"]["message"]` raises `TypeError` in exactly the case it was written to diagnose. Related trap found the same run: `urllib.parse.urlparse("192.168.0.69:1234").path == "192.168.0.69:1234"` (non-empty!), so any "if the URL path is empty, append `/v1`" heuristic silently accepts a schemeless base URL.
- **Context:** Adjudicating the `/v1` normalization rule in `falkor-chat/docs/plans/llm-provider-config.md` §4.3/§4.9 (K-042) against the running LM Studio.
- **Suggested home:** project docs (`falkor-chat/AGENTS.md` or `docs/DESIGN.md` §14 hazards) — the `urlparse` half is generic enough for `skills/python-web-quirks`
