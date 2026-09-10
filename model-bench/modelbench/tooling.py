"""The simulated-tool plugin seam (§3.3, §4 S2) — FR-10's ground truth.

`docs/plans/small-model-benchmarking.md` §3.3 lets a pack ship executable Python
(`tools/<module>.py`) implementing a simulated tool environment for FR-10's "system ground truth":
a **dispatched-call trace** and the environment's **resulting state**, never the model's reply text
(§3.8.4). `validate_pack`'s AST import allowlist (`modelbench/packs.py`) restricts every pack
module to stdlib plus **exactly one** non-stdlib import: `modelbench.tooling` — this module. That
makes this module's public surface a contract pack authors code against from S5/S6 onward (the
`tool-caller-shop-assistant` pack's `tools/sim.py`), so it is kept deliberately small: the shared
vocabulary a pack's tool module and the harness both need, and nothing else.

Two names, taken verbatim from §4 S2's code sketch and Appendix A:

* **`ToolEnvironment`** — the `Protocol` a pack's `tools.entrypoint` callable (`build_environment`
  in the `pack.json` example, §3.3) returns an instance of. `modelbench.convo.drive` calls
  `dispatch()` for every native tool call the model under test makes, then reads `trace()` and
  `state()` to build that turn's record (§3.8.4's "the environment records a dispatch trace ...
  and exposes its final state"). A pack module implements this structurally (`@runtime_checkable`
  makes `isinstance(obj, ToolEnvironment)` a real check on the four methods' presence) — there is
  no base class to subclass, deliberately: a pack's tool state (a cart, an order list, a product
  catalog) is the pack's own business, and this module has no opinion on it.
* **`DispatchRecord`** — one entry of `ToolEnvironment.trace()`, `(name, rawArguments,
  parsedArguments, returnValue, timestamp)` per Appendix A. `rawArguments` is what the model's tool
  call actually carried, JSON-parsed into a mapping but not otherwise interpreted — **and the claim
  is now true of the mechanism, which it was not** (plan gate P15-4(ii)): `drive` used to degrade
  an unreadable `arguments` value to `{}` and dispatch it anyway, so unparseable JSON, a JSON
  array, a JSON scalar and a genuine `{}` all arrived here as the same four bytes and the field
  whose name promises the model's own object carried the harness's fallback. Such a call is no
  longer dispatched at all (`convo._parse_tool_arguments` returns `None` for it and §4 S2's replay
  contract answers it with a `tool` message naming the failure), so every `rawArguments` mapping
  that reaches this type is one the model really sent.
  `parsedArguments` is what the environment's own dispatch logic made of them after its own
  unit/boundary handling — the distinction FR-8(d)'s boundary-rule scoring needs, and it is the
  environment implementation's job to keep the two apart, not this module's: a `dispatch()` that
  never coerces anything is free to set `parsedArguments == rawArguments`. `timestamp` is a UTC
  ISO-8601 string (`YYYY-MM-DDTHH:MM:SSZ`), matching this codebase's other stored timestamps
  (`modelbench/hostinfo.py`'s `_utc_stamp`, `Fingerprint.startedAt`/`endedAt`) rather than a raw
  epoch float, since a dispatch trace is exactly the kind of record that ends up serialized
  alongside those.

**Not this module's job**, named so a pack author is not left guessing: producing `DispatchRecord`s
is the environment implementation's own bookkeeping (this module supplies the shape, not a
recorder); parsing a model's raw tool-call `arguments` JSON string into a mapping before calling
`dispatch()` is `modelbench.convo.drive`'s job, since a pack's `dispatch(name, arguments)` already
receives a mapping; and interpreting `trace()`/`state()` into scored outcomes is
`scoring/toolcalls.py`'s (S5), not this module's or `convo`'s.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolEnvironment(Protocol):
    """A pack's simulated tool environment (§3.3, §3.8.4). Structural, not nominal: a pack's
    `tools.entrypoint` callable need not import this class at all to satisfy it —
    `@runtime_checkable` exists so the harness side (`modelbench.convo.drive`) can assert
    conformance with a plain `isinstance` check rather than trusting a pack's `build_environment()`
    return value blind.
    """

    def schemas(self) -> list[dict[str, Any]]:
        """The tool schemas this environment implements, as JSON Schema function definitions —
        the same shape a pack declares in `tools.schemas` (e.g. `tools/schemas.json`, §3.3) and
        the shape passed to the model as its native `tools` parameter."""
        ...

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Any:
        """Execute one tool call against this environment's (in-memory, per-conversation) state
        and return the call's result. `arguments` is already a mapping — parsed from the model's
        raw tool-call JSON by the caller (`modelbench.convo.drive`), never a JSON string here.
        Implementations must append **exactly one** `DispatchRecord` to their own internal trace
        on every call, since `trace()` below is this environment's own record, not something the
        harness reconstructs from the outside. `modelbench.convo.drive` **enforces** that count
        rather than trusting it: a turn's dispatch slice and the replayed `tool` message per
        `tool_calls` entry are both read positionally out of the new tail (plan gate P15-5)."""
        ...

    def trace(self) -> list["DispatchRecord"]:
        """Every dispatched call so far, in call order — FR-10's ground truth half. Growing across
        one whole conversation; `drive()` isolates one turn's share as the tail beyond a prefix it
        read before that turn's dispatches, so this must return calls in a stable order and never
        drop or reorder an earlier entry on a later call. `drive()` **checks** that prefix on every
        iteration and raises `convo.TraceContractViolated` when it has moved — a pack defect, and
        pack defects fail closed (plan §3.3, gate P15-5)."""
        ...

    def state(self) -> dict[str, Any]:
        """This environment's current state (e.g. cart contents, placed orders) — FR-10's ground
        truth other half, and the only state a scorer may treat as real; the model's own reply text
        is never ground truth (§3.8.4)."""
        ...


@dataclass(frozen=True)
class DispatchRecord:
    """One dispatched tool call, exactly as `ToolEnvironment.trace()` reports it (Appendix A):
    `name`, the raw and parsed arguments, the call's return value, and a UTC ISO-8601 timestamp.

    `returnValue` is left as `Any` deliberately — a tool's return shape is pack data, not something
    this module constrains, the same way `tools.schemas` is (§3.3)."""

    name: str
    rawArguments: Mapping[str, Any]
    parsedArguments: Mapping[str, Any]
    returnValue: Any
    timestamp: str
