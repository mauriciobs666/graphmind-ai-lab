"""CO-5 / P5-2 (``docs/reviews/connected-code-graph-p1-32b-impl.md`` Pass 5): the run-wide read over the whole chain applies only when the outer is a
``StoreError`` or another ``clients.common`` class (the client's own wrappers); a product's own outer type is read for itself, and the driver's fault names
count only for the redis module. ``exceptionClass`` is the outer's own class unless the outer is a ``StoreError``. The tier follows where the fault was
raised, never a name a product can choose. Offline."""
import pytest
import redis.exceptions as RE

import gate_helpers  # noqa: F401  (puts the tools directory on sys.path)
import gate_run as R
from clients.common import RawRefused, RefreshFailed, ResultTooLarge, StoreError
from test_gate_run_fault_grid import explicit, implicit

PRODUCT_NOPERM = type("NoPermissionError", (Exception,), {})
PRODUCT_AUTH = type("AuthenticationError", (Exception,), {})


def engine_entry(subject):
    return R.ask_failure_entry(explicit(R.EngineAskError(type(subject).__name__), subject))


@pytest.mark.parametrize("fault", [RE.AuthenticationError("a"), RE.NoPermissionError("p"), RE.ReadOnlyError("r")], ids=lambda f: type(f).__name__)
def test_a_product_outer_over_a_driver_run_wide_fault_is_an_engine_error_of_the_outer_class(fault):
    own = implicit(ValueError("the product's own"), fault)
    assert R.classify(own) == "deterministic"
    assert engine_entry(own) == {"outcome": "engine-error", "exceptionClass": "ValueError"}
    assert R.classify(explicit(ValueError("x"), fault)) == "deterministic"
    assert engine_entry(explicit(ValueError("x"), fault)) == {"outcome": "engine-error", "exceptionClass": "ValueError"}


@pytest.mark.parametrize("fault", [RE.AuthenticationError("a"), RE.NoPermissionError("p"), RE.ReadOnlyError("r")], ids=lambda f: type(f).__name__)
def test_a_store_error_over_a_driver_run_wide_fault_stays_run_wide(fault):
    exc = explicit(StoreError("x"), fault)
    assert R.classify(exc) == "deterministic" and engine_entry(exc) is None


@pytest.mark.parametrize("outer", [RefreshFailed, ResultTooLarge, RawRefused], ids=lambda c: c.__name__)
def test_a_client_common_outer_over_a_driver_run_wide_fault_stays_run_wide(outer):
    """Any ``clients.common`` class is the client's own wrapper, not only ``StoreError`` (none of these three is run-wide by itself)."""
    assert engine_entry(outer("c")) == {"outcome": "engine-error", "exceptionClass": outer.__name__}
    assert engine_entry(explicit(outer("c"), RE.AuthenticationError("a"))) is None


@pytest.mark.parametrize("cls", [PRODUCT_NOPERM, PRODUCT_AUTH], ids=lambda c: c.__name__)
def test_a_product_class_that_only_shares_a_run_wide_name_is_not_run_wide_under_any_outer(cls):
    assert engine_entry(cls("mine")) == {"outcome": "engine-error", "exceptionClass": cls.__name__}
    assert engine_entry(implicit(ValueError("v"), cls("mine"))) == {"outcome": "engine-error", "exceptionClass": "ValueError"}
    assert engine_entry(explicit(StoreError("x"), cls("mine"))) == {"outcome": "engine-error", "exceptionClass": cls.__name__}


def test_exception_class_is_the_innermost_only_under_a_store_error():
    assert engine_entry(explicit(StoreError("x"), RE.ResponseError("WRONGTYPE"))) == {"outcome": "engine-error", "exceptionClass": "ResponseError"}
    assert engine_entry(implicit(ValueError("v"), KeyError("k"))) == {"outcome": "engine-error", "exceptionClass": "ValueError"}


def test_a_product_outer_over_a_host_fault_is_still_the_products():
    assert engine_entry(implicit(ValueError("v"), OSError(28, "full"))) == {"outcome": "engine-error", "exceptionClass": "ValueError"}
