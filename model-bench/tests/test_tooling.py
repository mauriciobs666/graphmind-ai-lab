"""`modelbench.tooling` — the pack-importable plugin seam (plan §3.3, §4 S2).

Offline, no LM Studio, no pack loader from `modelbench.packs` beyond reading one already-checked-in
fixture (`tests/fixtures/packs/tooling_import_allowed/`, owned by `test_packs.py`) to confirm a real
pack module can now actually import this module — that fixture's own docstring, written before this
module existed, says it "does not exist yet"; it does now, and this file is where that gets proven,
without modifying the fixture.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from modelbench.tooling import DispatchRecord, ToolEnvironment

# --------------------------------------------------------------------------------------------
# DispatchRecord
# --------------------------------------------------------------------------------------------

#: Appendix A's literal, transcribed verbatim from `docs/plans/small-model-benchmarking.md`:
#: "`DispatchRecord` | `tooling` | `(name, rawArguments, parsedArguments, returnValue, timestamp)`
#: — FR-10's ground truth". Field order and names pinned against the plan's own text, not against
#: whatever this module happened to ship (the class the brief describes: "writing tests against
#: your own implementation rather than against the plan").
_APPENDIX_A_DISPATCH_RECORD_FIELDS = (
    "name",
    "rawArguments",
    "parsedArguments",
    "returnValue",
    "timestamp",
)


def test_dispatch_record_fields_match_the_plans_appendix_a_literal_in_order() -> None:
    fields = tuple(f.name for f in dataclasses.fields(DispatchRecord))
    assert fields == _APPENDIX_A_DISPATCH_RECORD_FIELDS


def test_dispatch_record_round_trips_distinct_raw_and_parsed_arguments() -> None:
    record = DispatchRecord(
        name="lookup_product_fact",
        rawArguments={"maxPrice": "50"},
        parsedArguments={"maxPrice": 50},
        returnValue={"price": 24.99},
        timestamp="2026-09-09T12:00:00Z",
    )
    assert record.name == "lookup_product_fact"
    assert record.rawArguments == {"maxPrice": "50"}
    assert record.parsedArguments == {"maxPrice": 50}
    assert record.rawArguments != record.parsedArguments
    assert record.returnValue == {"price": 24.99}
    assert record.timestamp == "2026-09-09T12:00:00Z"


def test_dispatch_record_is_frozen() -> None:
    record = DispatchRecord(
        name="view_cart",
        rawArguments={},
        parsedArguments={},
        returnValue=[],
        timestamp="2026-09-09T12:00:00Z",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.name = "clear_cart"  # type: ignore[misc]


# --------------------------------------------------------------------------------------------
# ToolEnvironment — a runtime-checkable structural Protocol
# --------------------------------------------------------------------------------------------


class _FullEnvironment:
    """A minimal, from-scratch conforming implementation — deliberately *not* importing
    `ToolEnvironment` at all, since the Protocol is structural and a pack author is never required
    to."""

    def schemas(self) -> list[dict]:
        return [{"name": "noop"}]

    def dispatch(self, name: str, arguments):
        return {"ok": True}

    def trace(self) -> list[DispatchRecord]:
        return []

    def state(self) -> dict:
        return {}


def test_tool_environment_isinstance_true_for_a_fully_conforming_object() -> None:
    assert isinstance(_FullEnvironment(), ToolEnvironment)


#: The four methods a `ToolEnvironment` must expose (plan §4 S2's code sketch). Named as a
#: constant, per the brief's house pattern, so the "missing exactly one method" coverage below is
#: checked against a declared set rather than four hand-written, driftable test functions.
_TOOL_ENVIRONMENT_METHODS: frozenset[str] = frozenset({"schemas", "dispatch", "trace", "state"})


def _environment_missing(method: str):
    """A conforming object with exactly one of `_TOOL_ENVIRONMENT_METHODS` removed."""

    class _Partial:
        def schemas(self) -> list[dict]:
            return []

        def dispatch(self, name: str, arguments):
            return None

        def trace(self) -> list[DispatchRecord]:
            return []

        def state(self) -> dict:
            return {}

    delattr(_Partial, method)
    return _Partial()


@pytest.mark.parametrize("missing", sorted(_TOOL_ENVIRONMENT_METHODS))
def test_tool_environment_isinstance_false_when_exactly_one_method_is_missing(missing: str) -> None:
    """Each of the four methods is checked individually — not just "all four absent" — so a
    Protocol narrowed to check only a subset could not pass this file's suite silently (the
    `isinstance` narrowing failure mode the brief names explicitly)."""
    assert not isinstance(_environment_missing(missing), ToolEnvironment)


def test_tool_environment_missing_method_coverage_matches_the_declared_set() -> None:
    """The parametrize grid above is generated from `_TOOL_ENVIRONMENT_METHODS`; this asserts the
    constant itself is exactly the Protocol's own method set, so a fifth method added to
    `ToolEnvironment` without updating the constant is caught here rather than silently under-
    covered."""
    declared = {
        name
        for name in vars(ToolEnvironment)
        if not name.startswith("_") and callable(getattr(ToolEnvironment, name, None))
    }
    assert declared == _TOOL_ENVIRONMENT_METHODS


def test_tool_environment_isinstance_false_for_an_unrelated_object() -> None:
    assert not isinstance(object(), ToolEnvironment)
    assert not isinstance(42, ToolEnvironment)


# --------------------------------------------------------------------------------------------
# The AST allowlist's other half: a real pack module can now actually import this module
# --------------------------------------------------------------------------------------------

_TOOLING_IMPORT_ALLOWED_FIXTURE = (
    Path(__file__).parent / "fixtures" / "packs" / "tooling_import_allowed" / "tools" / "sim.py"
)


def test_a_real_pack_tool_module_importing_modelbench_tooling_now_loads() -> None:
    """`tests/fixtures/packs/tooling_import_allowed/tools/sim.py` (owned by `test_packs.py`, not
    modified here) was written when `modelbench.tooling` did not exist yet and its own docstring
    says so; `validate_pack`'s AST check only ever parsed it, never imported it. Now that this
    module exists, importing that same file for real — the same `importlib.util.spec_from_file_
    location` mechanism `Pack.load_tool_module` uses — must succeed, and the `ToolEnvironment` it
    imports must be this module's own class, not a copy."""
    spec = importlib.util.spec_from_file_location(
        "modelbench_test_tooling_import_allowed_fixture", _TOOLING_IMPORT_ALLOWED_FIXTURE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ToolEnvironment is ToolEnvironment
