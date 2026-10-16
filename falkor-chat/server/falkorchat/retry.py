"""Transient provider-error retry for background model calls (K-067).

Design: `falkor-chat/docs/plans/extract-transient-retry.md` (§3.1 classifier, §3.2
policy, §3.6 helper). This module is the **only** place that decides whether a
provider failure is worth retrying; `transport.py`/`llm.py`/`embedding.py` only
record facts on `ProviderCallError` (`kind`/`status`/`detail`/`causes`).

**Classification (§3.1)** — an allow-list, so an unknown failure fails fast (today's
behaviour) rather than looping:

- anything that is not a `ProviderCallError` (graph errors, `EmbeddingDimensionError`,
  `ModelResolutionError`, bugs) is permanent;
- connection-level kinds (`connection`, `network`, `read`) are transient — they are
  what an LM Studio restart or crash looks like from the client;
- `timeout` is transient only when the policy says so (`retry_timeouts`, off in both
  shipped policies — §3.2.1 pre-states when the embed policy flips);
- `http`: 408, 429 and every 5xx are transient; any other status (LM Studio uses 400
  for both transient and permanent failures) is transient only when the provider's
  error text carries a known marker (`TRANSIENT_DETAIL_MARKERS`); a `body_error` the
  same way;
- `fallback_exhausted` is transient only when it has causes and every one is.

**Policy (§3.2)** — 4 attempts, nominal delays 2 s / 8 s / 30 s with upward jitter
(`× 1 + 0.5·U[0,1)`): attempt 2 catches a "within seconds" crash recovery, attempt 3
lands after a typical cold model reload, attempt 4 (40–60 s cumulative) clears the
slowest measured cold load (~21 s) by about 2×. Nothing longer: a minutes-long
contention episode should end the document in a visible `failed`.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .transport import ProviderCallError

_log = logging.getLogger(__name__)

T = TypeVar("T")

TRANSIENT_HTTP_STATUSES = frozenset({408, 429})  # plus every 5xx

# Case-insensitive substrings of `ProviderCallError.detail` that mark an otherwise
# permanent-looking status (LM Studio's 400) as transient.
TRANSIENT_DETAIL_MARKERS = (
    "terminated",  # LM Studio engine crash mid-request (K-067)
    "failed to load model",  # JIT (re)load failed — eviction / cold load
    "error loading model",
    "engine protocol startup was aborted",
    "channel error",  # serving-layer error seen alongside "terminated" (inferred, §3.1)
    "model unloaded",  # LM Studio mid-request unload, observed on `lms unload`; same body assumed for Eject/API — D-2
)

_CONNECTION_KINDS = frozenset({"connection", "network", "read"})


def _has_marker(detail: str | None) -> bool:
    if not detail:
        return False
    lowered = detail.lower()
    return any(marker in lowered for marker in TRANSIENT_DETAIL_MARKERS)


def is_transient(exc: BaseException, *, retry_timeouts: bool = False) -> bool:
    """True when `exc` is a provider failure worth retrying in place (plan §3.1)."""
    if not isinstance(exc, ProviderCallError):
        return False
    if exc.kind == "fallback_exhausted":
        return bool(exc.causes) and all(
            is_transient(cause, retry_timeouts=retry_timeouts) for cause in exc.causes
        )
    if exc.kind in _CONNECTION_KINDS:
        return True
    if exc.kind == "timeout":
        return retry_timeouts
    if exc.kind == "http":
        return (
            exc.status in TRANSIENT_HTTP_STATUSES
            or (exc.status is not None and 500 <= exc.status <= 599)
            or _has_marker(exc.detail)
        )
    if exc.kind == "body_error":
        return _has_marker(exc.detail)
    # "malformed_request", "invalid_json", "body_shape", "response_shape", None
    return False


@dataclass(frozen=True)
class RetryPolicy:
    """How many attempts, how long to wait before each retry, and whether a
    `timeout` counts as transient (plan §3.2)."""

    max_attempts: int
    delays_s: tuple[float, ...]
    jitter: float = 0.5
    retry_timeouts: bool = False

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if len(self.delays_s) != self.max_attempts - 1:
            raise ValueError(
                f"delays_s needs max_attempts - 1 = {self.max_attempts - 1} entries, "
                f"got {len(self.delays_s)}"
            )
        if self.jitter < 0:
            raise ValueError(f"jitter must be >= 0, got {self.jitter}")


# Equal today; separate so plan §3.2.1 can flip the embed one alone.
DEFAULT_PROVIDER_RETRY = RetryPolicy(max_attempts=4, delays_s=(2.0, 8.0, 30.0), jitter=0.5)
EMBED_CHUNK_RETRY = RetryPolicy(max_attempts=4, delays_s=(2.0, 8.0, 30.0), jitter=0.5)


def _gate_open(
    keep_going: Callable[[], bool] | None, label: str, kind: object, status: object
) -> bool:
    """One read of the liveness gate. A gate that raises is treated as open (logged):
    a flaky gate read must not turn a recoverable provider hiccup into a failure."""
    if keep_going is None:
        return True
    try:
        open_ = keep_going()
    except Exception:
        _log.exception("%s: liveness gate read failed; continuing to retry", label)
        return True
    if not open_:
        _log.info(
            "%s: owning document no longer processing; abandoning retry "
            "(kind=%s, status=%s)",
            label, kind, status,
        )
    return bool(open_)


def call_with_retry(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy = DEFAULT_PROVIDER_RETRY,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
    keep_going: Callable[[], bool] | None = None,
    label: str = "",
) -> T:
    """Call `fn()`, retrying a transient failure per `policy` (plan §3.6).

    On an exception, in this order: a non-transient one is re-raised at once, with no
    log line; on the last attempt a `giving up after N attempts` WARNING is logged and
    it is re-raised; otherwise `keep_going` is read (pre-sleep), the `attempt i/N`
    WARNING logged, `sleep(delay)` called, and `keep_going` read again (post-sleep)
    before the next attempt. Either gate read returning `False` logs one INFO line
    naming `label` and re-raises. Every re-raise is the same exception object, so the
    traceback is preserved. `delay` is `delays_s[i] × (1 + jitter·rand())` — upward
    jitter only, so no attempt fires before its derived point. `kind`/`status` are
    logged via `getattr`, so a non-provider error is logged safely too. Only
    `Exception` is caught: `KeyboardInterrupt`/`SystemExit` propagate at once.
    """
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if not is_transient(exc, retry_timeouts=policy.retry_timeouts):
                raise
            kind, status = getattr(exc, "kind", None), getattr(exc, "status", None)
            if attempt == policy.max_attempts:
                _log.warning(
                    "giving up after %d attempts (%s, kind=%s, status=%s): %s",
                    attempt, label, kind, status, exc,
                )
                raise
            if not _gate_open(keep_going, label, kind, status):  # pre-sleep read
                raise
            delay = policy.delays_s[attempt - 1] * (1 + policy.jitter * rand())
            _log.warning(
                "transient provider error (%s, kind=%s, status=%s), attempt %d/%d, "
                "retrying in %.1fs: %s",
                label, kind, status, attempt, policy.max_attempts, delay, exc,
            )
            sleep(delay)
            # Post-sleep read (review M-1): the document may have failed while we
            # slept — a sibling job gave up, or it was deleted. Never call again then.
            if not _gate_open(keep_going, label, kind, status):
                raise
    raise AssertionError("unreachable: the loop returns or raises")  # pragma: no cover
