"""Lazy provider registry (docs/plans/llm-gateway.md §3.2, S7; FR-15, AC-8).

Each kind maps to a module path and is imported on first use, so a provider module's import cost
and failures stay out of every other kind's path, and adding a kind touches no caller.

The seam a provider module implements: define a module-level `PROVIDER`, an instance of a
`providers.base.Provider` subclass whose `kind` equals the registered kind. Nothing else here
needs editing. A kind outside the built-in four registers itself with
`REGISTRY.register(kind, "<module path>")` (or an instance), with no calling-component change.
"""

from __future__ import annotations

import importlib
import threading

from llmgateway.providers.base import Provider
from llmgateway.types import CallFailed, FailureClass

BUILTIN_KINDS: dict[str, str] = {
    "lmstudio": "llmgateway.providers.lmstudio",
    "ollama": "llmgateway.providers.ollama",
    "openai_compat": "llmgateway.providers.openai_compat",
    "anthropic": "llmgateway.providers.anthropic",
}


class UnknownProviderKind(CallFailed):
    """A configured provider `kind` has no (importable) implementation."""

    def __init__(self, kind: str, why: str) -> None:
        super().__init__(FailureClass.CONFIG, f"provider kind {kind!r}: {why}")
        self.kind = kind


class ProviderRegistry:
    def __init__(self, kinds: dict[str, str] | None = None) -> None:
        self._targets: dict[str, str | Provider] = dict(BUILTIN_KINDS if kinds is None else kinds)
        self._loaded: dict[str, Provider] = {}
        self._lock = threading.Lock()

    def register(self, kind: str, target: str | Provider) -> None:
        """Add or replace a kind: a module path (imported lazily) or a ready `Provider`."""
        if not isinstance(kind, str) or not kind:
            raise ValueError("kind must be a non-empty string")
        with self._lock:
            self._targets[kind] = target
            self._loaded.pop(kind, None)

    def kinds(self) -> frozenset[str]:
        return frozenset(self._targets)

    def loaded(self) -> frozenset[str]:
        """Kinds already imported (a test seam for laziness)."""
        return frozenset(self._loaded)

    def get(self, kind: str) -> Provider:
        with self._lock:
            cached = self._loaded.get(kind)
            if cached is not None:
                return cached
            target = self._targets.get(kind)
            if target is None:
                raise UnknownProviderKind(kind, "not registered")
            provider = target if isinstance(target, Provider) else self._import(kind, target)
            if not isinstance(provider, Provider) or provider.kind != kind:
                raise UnknownProviderKind(kind, "its PROVIDER is not a Provider of that kind")
            self._loaded[kind] = provider
            return provider

    @staticmethod
    def _import(kind: str, module_path: str) -> object:
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            if exc.name != module_path:
                raise  # a broken import inside the provider module is a bug: fail loudly
            raise UnknownProviderKind(kind, f"module {module_path!r} is not installed") from exc
        provider = getattr(module, "PROVIDER", None)
        if provider is None:
            raise UnknownProviderKind(kind, f"module {module_path!r} defines no PROVIDER")
        return provider


REGISTRY = ProviderRegistry()
