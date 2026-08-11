"""OpenAI-compatible chat client for the K-013 AI responder (K-042 Landing 1: renamed
from `LMStudioLLM`, generalized to any OpenAI-compatible provider — LM Studio is one
of several, not the only one).

An injectable `Transport` seam makes the `/v1/chat/completions` request payload +
response parsing unit-testable offline; production injects `transport.make_http_transport`
(construction is `modelconfig.py`'s job — FR-4: every consumer resolves through
`ModelGateway`, never constructs this client directly). The `LLM` protocol is what the
`AgentResponder` depends on — unit tests inject a deterministic stub, so no test ever
hits a live model.

The responder calls `.complete(...)` **before** the guarded §4 write, so LLM
latency or failure short-circuits before anything is posted (failure isolation).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .transport import ProviderCallError, Transport, make_http_transport

# Used only when a caller constructs the client with no injected transport (direct
# construction is a tests-only / dev affordance — production always builds the
# transport via `modelconfig.py`, which knows the resolved model's real timeout).
_DIRECT_CONSTRUCTION_TIMEOUT = 180.0

# Chat messages are OpenAI-shaped `{"role": ..., "content": ...}` dicts.
ChatMessage = dict[str, str]


@dataclass(frozen=True)
class ToolCall:
    """One resolved tool call — OpenAI-shaped, with `arguments` parsed to a dict.

    `id` is the call id the model (or our fallback) assigned; U8 echoes it back
    on the `tool` result message. `arguments` is always a dict here (the native
    OpenAI shape delivers it as a JSON *string*, which `chat` decodes) — the
    caller never re-parses.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatResult:
    """A single assistant turn: free text and/or a list of tool calls.

    `text` is the message content (may be None/empty when the model only calls
    tools); `tool_calls` is empty on a plain-text turn. `is_tool_call` is the
    convenience the agent loop (U8) branches on.

    `model` (K-042 M-5) carries the `"<provider>/<model-id>"` ref that actually
    answered — the FR-8 traceability carrier, set by `OpenAICompatibleLLM.chat`.
    Additive and default-safe (`None` for any pre-K-042 construction). Landing 2's
    `FallbackClient` reads this off the return value instead of any mutable
    per-client "last used" state (M-5) — never on `self`.

    `fallback` (K-042 P2-B, `-graph.md` §1.3/§6.2) is `True` iff the answering chain
    element's index in the resolved fallback chain (FR-18) is `> 0` — set by
    `FallbackClient` at the same point it already resolves `model`. It is `None`
    (never `False`) on a length-1 (non-fallback) chain or when the primary element
    answers: the property's *presence* is the operator-visible signal, not its value.
    """

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str | None = None
    fallback: bool | None = None

    @property
    def is_tool_call(self) -> bool:
        return bool(self.tool_calls)


class LLM(Protocol):
    """The LLM seam: single-shot completion plus tool-calling chat.

    `complete` is what `AgentResponder` (M2) depends on; `chat` is the
    tool-calling extension the M3 executor's agent nodes (U8) drive.
    """

    def complete(self, messages: list[ChatMessage]) -> str: ...

    def chat(
        self, messages: list[ChatMessage], tools: list[dict[str, Any]]
    ) -> ChatResult: ...


def _default_transport() -> Transport:
    """The fallback transport for a directly-constructed client (tests/dev only —
    production always builds the transport in `modelconfig.py`, which knows the
    resolved model's real per-kind/per-model timeout)."""
    return make_http_transport(timeout=_DIRECT_CONSTRUCTION_TIMEOUT)


class OpenAICompatibleLLM:
    """OpenAI-compatible `/v1/chat/completions` client (K-042: generalized from the
    LM-Studio-only `LMStudioLLM`; any `protocol="openai"` provider works the same way).

    `base_url`/`model` are **required** — no config-derived defaults (K-042 FR-20:
    model choice is no longer an env var). `transport` is injectable for tests; it
    defaults to a bare urllib POST with no timeout binding beyond a generic fallback
    (real production timeouts come from `modelconfig.py`). `ref` is the resolved
    `"<provider>/<model-id>"` string `modelconfig.py` builds this client from — the
    FR-8 carrier `ChatResult.model` reports; defaults to `model` when omitted (a
    direct construction with no provider prefix still gets a sensible value).

    **FR-4:** only `modelconfig.py` and `tests/` may construct this client directly
    — every real consumer resolves through `ModelGateway` (enforced by
    `test_modelconfig.py`'s AST check).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        transport: Transport | None = None,
        params: dict[str, Any] | None = None,
        ref: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._transport = transport or _default_transport()
        self._params = dict(params) if params else {}
        self._ref = ref or model

    def complete(self, messages: list[ChatMessage]) -> str:
        resp = self._transport(
            f"{self._base_url}/chat/completions",
            {"model": self._model, "messages": list(messages), **self._params},
        )
        try:
            return resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderCallError(
                f"{self._ref} @ {self._base_url}: response missing choices: {exc}"
            ) from exc

    def chat(
        self, messages: list[ChatMessage], tools: list[dict[str, Any]]
    ) -> ChatResult:
        """One tool-calling turn: POST `messages` + granted `tools`, parse a `ChatResult`.

        Dual-shape parsing (Q3): prefer the structured `message.tool_calls` field;
        if it is absent, fall back to detecting a tool call emitted inside
        `message.content` — as JSON (the known LM Studio / Qwen3 failure mode) or as
        bare `name({...})` call syntax (K-027). When neither yields a tool call, the
        turn is plain text. Name-against-granted-set and
        arg-schema validation live in the agent loop (U8), not here — `chat` only
        parses the wire shape into `ChatResult`.

        The returned `ChatResult.model` carries `ref` — the resolved model that
        actually answered (M-5); Landing 2's `FallbackClient` reads it off the
        return value rather than any per-client mutable state.
        """
        resp = self._transport(
            f"{self._base_url}/chat/completions",
            {
                "model": self._model, "messages": list(messages), "tools": list(tools),
                **self._params,
            },
        )
        try:
            message = resp["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderCallError(
                f"{self._ref} @ {self._base_url}: response missing choices: {exc}"
            ) from exc
        return replace(_parse_chat_message(message), model=self._ref)


class FallbackClient:
    """An ordered fallback chain of `LLM`-shaped clients (K-042 Landing 2, FR-18):
    built by `ModelGateway.llm()` when a role (FR-7) resolves to more than one model.

    `.chat(...)`/`.complete(...)` try `clients[0]`, then `clients[1]`, … on a
    `ProviderCallError`, in order; the first success answers. When **every** element
    fails, raises `ProviderCallError` naming every model tried — not just the last —
    so an operator sees the whole chain that failed. A `TimeoutError` at the transport
    layer is already converted to `ProviderCallError` by `transport.make_http_transport`
    (the B-2 fix), so it falls through the chain exactly like any other provider error.

    **No mutable "last used" state (M-5).** `_clients`/`_refs` are set once at
    construction and never reassigned — `__slots__` makes that structural, not just
    documented. The answering model (and whether it came from a fallback) travels on
    the `ChatResult` return value (`.model`/`.fallback`), never on `self`.
    """

    __slots__ = ("_clients", "_refs")

    def __init__(self, clients: Any, refs: Any) -> None:
        clients = tuple(clients)
        refs = tuple(refs)
        if len(clients) != len(refs) or not clients:
            raise ValueError("FallbackClient needs at least one (client, ref) pair")
        self._clients = clients
        self._refs = refs

    def _errors_message(self, errors: list[str]) -> str:
        return "all models in fallback chain failed: " + "; ".join(errors)

    def chat(
        self, messages: list[ChatMessage], tools: list[dict[str, Any]]
    ) -> ChatResult:
        errors: list[str] = []
        for index, client in enumerate(self._clients):
            try:
                result = client.chat(messages, tools)
            except ProviderCallError as exc:
                errors.append(f"{self._refs[index]}: {exc}")
                continue
            return replace(result, fallback=True if index > 0 else None)
        raise ProviderCallError(self._errors_message(errors))

    def complete(self, messages: list[ChatMessage]) -> str:
        errors: list[str] = []
        for index, client in enumerate(self._clients):
            try:
                return client.complete(messages)
            except ProviderCallError as exc:
                errors.append(f"{self._refs[index]}: {exc}")
        raise ProviderCallError(self._errors_message(errors))


def _parse_chat_message(message: dict[str, Any]) -> ChatResult:
    """Turn one OpenAI `choices[0].message` dict into a `ChatResult`.

    Order matters (Q3): the structured `tool_calls` field is authoritative; only
    when it is absent/empty do we probe `content` for an embedded tool call (the
    Qwen3 fallback). Content that is not a tool call is returned as text.
    """
    content = message.get("content")

    native = _parse_native_tool_calls(message.get("tool_calls"))
    if native:
        return ChatResult(text=content, tool_calls=native)

    embedded = _parse_content_tool_calls(content)
    if embedded:
        return ChatResult(text=None, tool_calls=embedded)

    return ChatResult(text=content, tool_calls=[])


def _parse_native_tool_calls(raw: Any) -> list[ToolCall]:
    """Parse the structured OpenAI `message.tool_calls` list into `ToolCall`s.

    Each entry nests the call under `function` with `arguments` as a JSON *string*
    (OpenAI shape); we decode it. LM Studio occasionally hands back an already-parsed
    dict — tolerated. Unparseable/nameless entries are skipped rather than raised.
    """
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function", entry)
        name = fn.get("name") if isinstance(fn, dict) else None
        if not name:
            continue
        args = _coerce_arguments(fn.get("arguments") if isinstance(fn, dict) else None)
        call_id = entry.get("id") or f"call_{i}"
        calls.append(ToolCall(id=str(call_id), name=str(name), arguments=args))
    return calls


def _parse_content_tool_calls(content: Any) -> list[ToolCall]:
    """Fallback: detect a tool call emitted inside `content` (Q3, Qwen3).

    Two probes, in order. **JSON first** (probe order unchanged; the empty-envelope
    fall-through below is new — see HISTORY 2026-07-24): the OpenAI-ish
    `{"name": ..., "arguments": ...}` and structured-output `{"action": ..., "args": ...}`
    shapes, a `{"function": {...}}` wrapper, and a `{"tool_calls": [...]}` envelope.
    **Bare call syntax second** (K-027 slice A): `name({...})` written as an
    expression — see `_parse_bare_call_syntax`. Plain prose (no recognizable
    tool-call keys and no call expression) yields no calls, so it stays text.
    """
    # NOTE (K-035): the JSON probe running first means an *argument* object whose keys
    # include `name`/`action`/`tool` shadows the surrounding call name —
    # `create_user({"name": "bob"})` parses as a call named `bob`, and the bare-call
    # probe below never sees it. Harmless only while no granted tool declares such a
    # parameter (today: `text`/`mentions`/`query`/`reason`). **Read K-035 in
    # `docs/BACKLOG.md` before registering a tool with a `name`, `action` or `tool`
    # argument** — the failure is silent and manufactures a call named after a
    # user-supplied value.
    obj = extract_json_object(content)
    if obj is not None:
        if isinstance(obj.get("tool_calls"), list):
            calls = _parse_native_tool_calls(obj["tool_calls"])
            if calls:
                return calls
        call = _normalize_tool_call(obj)
        if call:
            return [call]
    return _parse_bare_call_syntax(content)


def _normalize_tool_call(obj: dict[str, Any], index: int = 0) -> ToolCall | None:
    """Map one loosely-shaped dict to a `ToolCall`, or None if it isn't a call."""
    fn = obj.get("function")
    if isinstance(fn, dict):
        obj = fn
    name = obj.get("name") or obj.get("action") or obj.get("tool")
    if not name:
        return None
    args = _coerce_arguments(
        obj.get("arguments") if "arguments" in obj else obj.get("args")
    )
    call_id = obj.get("id") or f"call_{index}"
    return ToolCall(id=str(call_id), name=str(name), arguments=args)


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    """Normalize a tool-call `arguments` value to a dict.

    OpenAI delivers arguments as a JSON string; some backends pre-parse to a dict.
    Anything else (or unparseable JSON) becomes an empty dict — U8's schema check
    is the real gate, so `chat` never raises on a malformed arg blob.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# A call expression opening a line: optional indentation, `name(`. MULTILINE `^` is the
# first false-positive guard — prose *about* a tool ("I will call post_message(...)")
# never starts its line with the identifier, so it is never mistaken for a call. Leading
# whitespace is deliberately allowed, so an **indented** call is still recovered, as is
# `name ({…})` with a space or tab *between* the identifier and the paren; markdown
# list markers (`- name(…)`, `1. name(…)`) are not, because the marker precedes the
# identifier.
_BARE_CALL_OPEN = re.compile(r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(", re.MULTILINE)


def _parse_bare_call_syntax(content: Any) -> list[ToolCall]:
    """Fallback: recover a tool call written as bare `name({...})` syntax (K-027).

    Observed live (DEF-K027-B): a 4B model emitted `post_message({"text": "…"})` as
    its step output. It is a call, but no JSON probe recovers it — the argument
    object has no name/action/tool key — so it used to be lost as prose while the
    run parked looking healthy.

    **Exactly what is enforced** (three rules, all needed — gate M-1 found the first
    two alone let a *quoted* call fire and dispatch a real write):

    1. Each call opens a line — optional indentation, then the identifier, then the
       paren (a space or tab before the paren is tolerated) — and nothing but
       whitespace follows its closing parenthesis on that line.
    2. Its argument is empty or a single JSON **object**. `name(text="hi")`,
       `name("hi")` and `name(...)` are deliberately *not* recognised: mapping a
       keyword/positional argument onto the tool schema would be a guess.
    3. From the first accepted call onward the message holds **nothing but calls and
       whitespace** — no prose *between* two calls, and nothing at all after the last
       one. Anything else (a closing ```` ``` ````, another sentence) means a call was
       being *quoted or illustrated*, so the whole recovery is discarded and the
       content stays text. Anchoring only the *last* call is not enough (gate N-1):
       the executor dispatches **every** returned call, so "I considered X … but
       instead I will Y" would write the quoted X to the thread as well.

    Rule 3 is what makes "the expression owns its lines" true rather than aspirational.
    **Residuals, by design** — position alone cannot tell these from an intended call:

    * a message whose final line happens to be an illustrative call still fires;
    * a *contiguous catalogue* of own-line calls, with no prose between them, is
      structurally identical to an intended multi-call, so it still fires.

    Both are bounded and sit on the safe side of the trade — the executor re-prompts on
    a missed call, while an invented `post_message` is an irreversible thread write.
    Closing them needs the granted tool names as a recognition filter, not a position
    rule (K-027 open question 4).

    Identical repeated calls collapse to one (small models self-repeat; dispatching the
    repeat would post the same message twice). Distinct calls are all returned.
    """
    if not isinstance(content, str):
        return []
    text = _strip_code_fence(content.strip())

    calls: list[ToolCall] = []
    seen: set[tuple[str, str]] = set()
    consumed = end_of_last_call = 0
    for match in _BARE_CALL_OPEN.finditer(text):
        if match.start() < consumed:
            continue  # inside an already-scanned expression
        scanned = _scan_balanced(text, match.end() - 1)
        if scanned is None:
            continue
        inner, after = scanned
        # Advance past the whole expression whether or not we accept it: a nested
        # identifier inside a *rejected* expression is not a call of its own.
        consumed = after
        if text[after:].split("\n", 1)[0].strip():
            continue  # trailing prose on the same line ⇒ narration, not a call
        arguments = _parse_call_arguments(inner)
        if arguments is None:
            continue
        if calls and text[end_of_last_call : match.start()].strip():
            # Prose between two accepted calls ⇒ the earlier one was being quoted or
            # illustrated ("I considered X … but instead I will Y"). Position cannot
            # tell which of the two is meant, so discard the whole recovery.
            return []
        end_of_last_call = after
        fingerprint = (match.group(1), json.dumps(arguments, sort_keys=True))
        if fingerprint in seen:
            continue  # identical repeat ⇒ one dispatch, not two
        seen.add(fingerprint)
        calls.append(
            ToolCall(id=f"call_{len(calls)}", name=match.group(1), arguments=arguments)
        )

    if calls and text[end_of_last_call:].strip():
        return []  # the call was quoted/illustrated, not made — keep the prose
    return calls


def _scan_balanced(text: str, open_idx: int) -> tuple[str, int] | None:
    """Return `(inner, index_after_close)` for the bracket at `open_idx`, or None.

    Handles `(`/`)` and `{`/`}`. Scans for the balanced closer, ignoring brackets
    inside JSON string literals (and their backslash escapes) — a message body
    routinely contains them. Unbalanced input yields None (the content stays text).
    """
    opener = text[open_idx]
    closer = {"(": ")", "{": "}"}[opener]
    depth = 0
    in_string = False
    escaped = False
    for i in range(open_idx, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i + 1
    return None


def _parse_call_arguments(inner: str) -> dict[str, Any] | None:
    """Map a bare call's argument text to a dict, or None when it isn't recognisable.

    Empty ⇒ `{}` (a no-argument call). Otherwise it must be a JSON object; anything
    else (keyword syntax, a positional scalar, malformed JSON) returns None so the
    caller leaves the content as text rather than dispatching guessed arguments.
    """
    inner = inner.strip()
    if not inner:
        return {}
    try:
        parsed = json.loads(inner)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _strip_code_fence(text: str) -> str:
    """Strip one leading/trailing markdown code fence (``` or ```json) if present."""
    if not text.startswith("```"):
        return text
    inner = text[3:]
    if inner[:4].lower() == "json":
        inner = inner[4:]
    return inner.rsplit("```", 1)[0].strip()


def extract_json_object(content: Any) -> dict[str, Any] | None:
    """Best-effort: pull a single JSON object out of a `content` string.

    Handles a bare JSON object, one fenced in a ```json code block, and a JSON
    object embedded in surrounding prose (first `{` … last `}`). Returns None when
    no object parses — the caller then treats the content as plain text.

    **Tolerance contract:** this is the *permissive* extractor. It is order-blind —
    it does not care where in the prose the object sits, nor whether the surrounding
    sentence endorses it — so an object the model is merely *quoting* is extracted
    just the same. That is acceptable on the tool-call path, where the agent loop
    re-validates the name against the granted set and the schema against the tool.
    A caller that must not act on a quoted object wants
    `extract_own_line_json_object` instead (K-027 gate B-1).
    """
    if not isinstance(content, str):
        return None
    text = _strip_code_fence(content.strip())
    if not text:
        return None

    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# A JSON object opening a line: optional indentation, then `{`. The own-line anchor is
# the same discriminator `_BARE_CALL_OPEN` uses — an object a sentence merely quotes
# ("I would answer {…} but…") never starts its line.
_OWN_LINE_OBJECT_OPEN = re.compile(r"^[ \t]*\{", re.MULTILINE)


def extract_own_line_json_object(
    content: Any, *, require_key: str | None = None
) -> dict[str, Any] | None:
    """Conservative sibling of `extract_json_object` — biased to find *nothing*.

    **Exactly what is accepted**, after stripping one surrounding code fence:

    * a reply that is **entirely** one JSON object (the bare and fenced shapes), or
    * **exactly one** JSON object that *owns its lines* — its `{` is the first
      non-whitespace character of a line, nothing but whitespace follows its matching
      `}` on that line — and which carries `require_key` when one is given.

    Everything else returns None: an object embedded mid-sentence, **two or more**
    qualifying objects (which one is meant is a guess), or nothing that parses.

    This exists because the guard judge must never *advance* on an object it merely
    read out of prose. `extract_json_object` would happily lift the quoted verdict out
    of *"I would answer {"decision": true, …} but they did not, so I answer false"* —
    a false advance, the one direction `guards._coerce_verdict` biases against and
    cannot catch (the quoted rationale is clean, so no contradiction cue fires).

    **Residual, by design:** a hypothetical verdict the model writes on its own line,
    with no second object to disambiguate it, is still accepted. Closing that needs
    the model's intent, not a parser.
    """
    if not isinstance(content, str):
        return None
    text = _strip_code_fence(content.strip())
    if not text:
        return None

    whole = _load_json_object(text)
    if whole is not None:
        return whole

    matches: list[dict[str, Any]] = []
    consumed = 0
    for match in _OWN_LINE_OBJECT_OPEN.finditer(text):
        if match.start() < consumed:
            continue  # nested inside an object we already scanned
        scanned = _scan_balanced(text, match.end() - 1)
        if scanned is None:
            continue
        _, after = scanned
        consumed = after
        if text[after:].split("\n", 1)[0].strip():
            continue  # trailing prose on the closing line ⇒ quoted, not asserted
        parsed = _load_json_object(text[match.end() - 1 : after])
        if parsed is None or (require_key is not None and require_key not in parsed):
            continue
        matches.append(parsed)

    return matches[0] if len(matches) == 1 else None


def _load_json_object(text: str) -> dict[str, Any] | None:
    """`json.loads(text)` when it yields a dict, else None."""
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None
