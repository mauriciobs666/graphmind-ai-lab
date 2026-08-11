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
  3. `(TimeoutError, OSError)` — the bare read-phase timeout and any other socket
     error. `TimeoutError` is **not** a `URLError` and must be named explicitly, or a
     read timeout escapes classification entirely (the single most likely reason to
     want FR-18's fallback chain to fire).
  4. `ValueError` — belt and braces for malformed-URL shapes.
  5. **Body-level:** a `200` whose decoded JSON object carries an `error` key. LM
     Studio returns a **string** `error` on the wrong-`/v1`-prefix path and an
     **object** `error` on the right one (plan §2.3) — detect on key *presence*, then
     render, or `body["error"]["message"]` raises `TypeError` in exactly the case this
     check exists to diagnose.
  6. **Body-shape:** a `200` body that decodes to something other than a JSON object.

Every raised `ProviderCallError` names provider · model · the resolved URL · cause —
callers pass `provider=`/`model=` at construction so the message identifies which
resolved model actually failed. A well-formed body missing `choices`/`data` is the
client layer's seventh class (`llm.py`/`embedding.py`), not this module's.
"""

from __future__ import annotations

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
    bare `KeyError` or an opaque `TypeError` in its place (FR-10)."""


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
                f"{label}: HTTP {exc.code} {exc.reason}: {body}"
            ) from exc
        except urllib.error.URLError as exc:  # rung 2
            raise ProviderCallError(f"{label}: connection failed: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:  # rung 3 — TimeoutError is NOT a URLError
            raise ProviderCallError(f"{label}: {type(exc).__name__}: {exc}") from exc
        except ValueError as exc:  # rung 4 — malformed-URL shapes
            raise ProviderCallError(f"{label}: malformed request: {exc}") from exc

        try:
            raw = resp.read()
        finally:
            close = getattr(resp, "close", None)
            if callable(close):
                close()

        try:
            body = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except (ValueError, TypeError) as exc:
            raise ProviderCallError(
                f"{label}: response body is not valid JSON: {exc}"
            ) from exc

        if isinstance(body, dict):
            if "error" in body:  # rung 5 — body-level error, string OR object (m-1)
                raise ProviderCallError(
                    f"{label}: provider error: {_render_body_error(body['error'])}"
                )
            return body

        # rung 6 — a 200 body that decoded to something other than a JSON object
        raise ProviderCallError(
            f"{label}: response body is not a JSON object (got {type(body).__name__})"
        )

    return _transport
