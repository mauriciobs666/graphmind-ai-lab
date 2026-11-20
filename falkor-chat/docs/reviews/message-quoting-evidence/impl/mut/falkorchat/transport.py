"""The one HTTP transport every OpenAI-compatible provider call goes through
(K-042 Landing 1, FR-14/FR-10, plan §4.9).

`llm._urllib_transport` and `embedding._urllib_transport` used to be byte-identical
copies with **no `timeout=`** — FR-14's per-model timeout had nowhere to land, and a
hung endpoint escaped `FallbackClient`'s failure detection entirely (plan §2.2, §4.9).
This module replaces both: one POST-JSON transport, timeout + extra headers bound at
construction (so the `Transport` alias `(url, payload) -> dict` stays unchanged — the
seam every offline test injects), and the ordered exception ladder below.

**The §4.9 ladder — order is load-bearing, verified twice by review (B-2):**

  1. `urllib.error.HTTPError` — first, because it is a subclass of `URLError`; an
     `URLError`-first ladder makes this branch dead code. Includes the (truncated)
     response body — on a cloud 401 that body is the only thing that says *why*.
  2. `urllib.error.URLError` — connection refused, DNS failure, unknown URL type.
  3a. `http.client.InvalidURL` — a URL `modelconfig` lets through but `urlopen`
     rejects (non-numeric port, control characters in the host): a permanent
     misconfiguration, reported like rung 4. It **must precede rung 3**, because
     `InvalidURL` is an `HTTPException`, not a `ValueError` (K-067, plan
     `falkor-chat/docs/plans/extract-transient-retry.md` §3.1, review N-1).
  3. `(TimeoutError, OSError, http.client.HTTPException)` — the bare read-phase
     timeout, any other socket error, and a malformed status line (`BadStatusLine`,
     `LineTooLong` are not `OSError`s). `TimeoutError` is **not** a `URLError` and
     must be named explicitly, or a read timeout escapes classification entirely (the
     single most likely reason to want FR-18's fallback chain to fire).
  4. `ValueError` — belt and braces for malformed-URL shapes.
  5. **Body-level:** a `200` whose decoded JSON object carries an `error` key. LM
     Studio returns a **string** `error` on the wrong-`/v1`-prefix path and an
     **object** `error` on the right one (plan §2.3) — detect on key *presence*, then
     render, or `body["error"]["message"]` raises `TypeError` in exactly the case this
     check exists to diagnose.
  6. **Body-shape:** a `200` body that decodes to something other than a JSON object.

Rung 3b sits between the connect ladder and the body checks: a failure inside
`resp.read()` (a reset or truncated body, a read timeout) is wrapped as a
`ProviderCallError` ("reading response failed") instead of escaping raw (K-067).

Every raised `ProviderCallError` names provider · model · the resolved URL · cause —
callers pass `provider=`/`model=` at construction so the message identifies which
resolved model actually failed. A well-formed body missing `choices`/`data` is the
client layer's seventh class (`llm.py`/`embedding.py`), not this module's.
"""

from __future__ import annotations

import http.client
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any, Callable

_log = logging.getLogger(__name__)

# A transport is `(url, json_payload) -> parsed_json_dict`. Injected so a client's
# request/response handling is unit-testable without a live provider.
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]

# How much of an HTTPError's body to fold into the raised message — enough to carry
# a cloud 401's "why", short enough to keep a log/trace line sane.
_ERROR_BODY_MAX = 500


class ProviderCallError(RuntimeError):
    """A provider HTTP call failed — transport/connection, HTTP, timeout, malformed
    URL, or a loud body-level error (the §4.9 ladder). Always names provider, model,
    the resolved URL, and the underlying cause, so a run/log/trace never carries a
    bare `KeyError` or an opaque `TypeError` in its place (FR-10).

    Beyond the human-facing message, each raiser records structured, keyword-only
    facts (K-067, `falkor-chat/docs/plans/extract-transient-retry.md` §3.1): `kind`
    (which ladder rung or client check raised — `"http"`, `"connection"`,
    `"network"`, `"timeout"`, `"read"`, `"malformed_request"`, `"invalid_json"`,
    `"body_error"`, `"body_shape"`, `"response_shape"`, `"fallback_exhausted"`),
    `status` (the HTTP status, rung 1 only), `detail` (the provider's own error text)
    and `causes` (an exhausted fallback chain's per-element errors). They are additive
    — `str(exc)` is unchanged — and `retry.is_transient` is their only consumer of
    record: this module reports facts, `retry.py` alone decides what is transient."""

    def __init__(
        self,
        message: str,
        *,
        kind: str | None = None,
        status: int | None = None,
        detail: str | None = None,
        causes: tuple[ProviderCallError, ...] = (),
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.detail = detail
        self.causes = causes


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read()
    except Exception:  # pragma: no cover — defensive; a body read must never mask the real error
        return ""
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover
        return ""
    return text if len(text) <= _ERROR_BODY_MAX else text[:_ERROR_BODY_MAX] + "…"


def _render_body_error(err: Any) -> str:
    """Render a body-level `error` value — a **string** on one LM Studio path, an
    **object** on the other, on the very same server (plan §2.3). Detect on key
    presence, then render; never assume the shape."""
    if isinstance(err, Mapping):
        return str(err.get("message", err))
    return str(err)


def make_http_transport(
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
    provider: str = "?",
    model: str = "?",
) -> Transport:
    """Build a `Transport`: timeout, extra headers, and the injectable `opener` are
    all bound at construction, so the transport function itself keeps the stable
    `(url, payload) -> dict` shape every offline test relies on.

    `opener` defaults to `urllib.request.urlopen`; tests inject a fake to drive each
    ladder branch without a real HTTP call. `provider`/`model` are folded into every
    raised `ProviderCallError` so a failure names which resolved model it came from.
    """
    request_headers = {"Content-Type": "application/json", **(headers or {})}

    def _transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers=request_headers, method="POST"
        )
        label = f"{provider}/{model} @ {url}"

        try:
            resp = opener(req, timeout=timeout)
        except urllib.error.HTTPError as exc:  # rung 1 — MUST precede URLError (B-2)
            body = _read_error_body(exc)
            raise ProviderCallError(
                f"{label}: HTTP {exc.code} {exc.reason}: {body}",
                kind="http",
                status=exc.code,
                detail=body,
            ) from exc
        except urllib.error.URLError as exc:  # rung 2
            raise ProviderCallError(
                f"{label}: connection failed: {exc.reason}",
                kind="timeout" if isinstance(exc.reason, TimeoutError) else "connection",
            ) from exc
        except http.client.InvalidURL as exc:  # rung 3a — MUST precede rung 3 (N-1)
            raise ProviderCallError(
                f"{label}: malformed request: {exc}", kind="malformed_request"
            ) from exc
        except (TimeoutError, OSError, http.client.HTTPException) as exc:  # rung 3
            raise ProviderCallError(
                f"{label}: {type(exc).__name__}: {exc}",
                kind="timeout" if isinstance(exc, TimeoutError) else "network",
            ) from exc
        except ValueError as exc:  # rung 4 — malformed-URL shapes
            raise ProviderCallError(
                f"{label}: malformed request: {exc}", kind="malformed_request"
            ) from exc

        try:
            raw = resp.read()
        except (TimeoutError, OSError, http.client.HTTPException) as exc:  # rung 3b
            raise ProviderCallError(
                f"{label}: reading response failed: {type(exc).__name__}: {exc}",
                kind="timeout" if isinstance(exc, TimeoutError) else "read",
            ) from exc
        finally:
            close = getattr(resp, "close", None)
            if callable(close):
                close()

        try:
            body = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except (ValueError, TypeError) as exc:
            raise ProviderCallError(
                f"{label}: response body is not valid JSON: {exc}", kind="invalid_json"
            ) from exc

        if isinstance(body, dict):
            if "error" in body:  # rung 5 — body-level error, string OR object (m-1)
                detail = _render_body_error(body["error"])
                raise ProviderCallError(
                    f"{label}: provider error: {detail}", kind="body_error", detail=detail
                )
            return body

        # rung 6 — a 200 body that decoded to something other than a JSON object
        raise ProviderCallError(
            f"{label}: response body is not a JSON object (got {type(body).__name__})",
            kind="body_shape",
        )

    return _transport
