"""Errors the model-backed half of the plain-language front raises (C-1001 P1-26 dispatch 2). Stdlib only, so ``ccg.front`` can name them without loading the broker client."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ModelError(Exception):
    """A model call could not be made or was refused: no broker, a latched broker, a denial, a failed or unusable upstream exchange. There is no other path to the model
    (``-egress-spec3.md`` §2.5), so the operation fails with this reason; ``detail`` is JSON-able and value-free."""

    def __init__(self, message: str, *, detail: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.detail = dict(detail or {})
