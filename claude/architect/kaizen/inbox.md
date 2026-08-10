# Kaizen — Learnings Inbox: architect

> Append-only capture of durable, non-obvious environment facts the `architect` agent
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

## 2026-08-10 — LM Studio serves its OpenAI-compatible API only under `/v1`, and a missing prefix returns **HTTP 200** with an error envelope, not a 404

- **Evidence:** on the dev box (WSL2 → LM Studio at `localhost:1234`):
  `POST /v1/chat/completions` with `{}` → `400` + a proper OpenAI error object;
  `POST /chat/completions` with `{}` → **`200`** + `{"error":"Unexpected endpoint or method. (POST /chat/completions)"}`;
  same for `/v1/embeddings` (400) vs `/embeddings` (200 + the same envelope).
  `GET /models` and `GET /v1/models` **both** return `200` with the real model list, so a
  "probe `/models`" check cannot discriminate the prefix. Any OpenAI-shaped client that omits
  `/v1` therefore fails as a bare `KeyError: 'choices'` with no mention of the URL.
- **Context:** designing falkor-chat's LLM provider/model configuration (K-042), where two real
  `opencode.json` files disagree on whether `options.baseURL` carries the `/v1` suffix.
- **Suggested home:** project docs (`falkor-chat/docs/DESIGN.md` §14 config seam) — possibly also
  the `python-web-quirks` skill, since "200 + error envelope" is a general OpenAI-compatible-server
  failure shape worth knowing before writing any such client.

