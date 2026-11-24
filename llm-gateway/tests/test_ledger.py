"""The holder ledger (plan §3.4 FR-10; §5.2 FR-10 (2),(4),(5),(6))."""

import os
import stat
import threading

import pytest

from llmgateway.failures import HOLDER_REFUSALS
from llmgateway.ledger import Ceilings, Ledger, LedgerLocked, Refusal, positive_int


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


CEIL = Ceilings(window_seconds=3600, window_max_tokens=10_000, window_max_released=3)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def ledger(tmp_path, clock):
    led = Ledger(tmp_path / "ledger.sqlite3", clock=clock)
    yield led
    led.close()


def test_run_bound_accumulates_and_refuses_the_call_that_would_exceed(ledger):
    a = ledger.reserve(amount=400, ceilings=CEIL, run_id="eval-1", run_bound=1000)
    ledger.settle(a, 400)
    b = ledger.reserve(amount=400, ceilings=CEIL, run_id="eval-1", run_bound=1000)
    ledger.settle(b, 400)
    with pytest.raises(Refusal) as err:
        ledger.reserve(amount=400, ceilings=CEIL, run_id="eval-1", run_bound=1000)
    assert err.value.code == "run_bound_reached"
    assert ledger.run_total("eval-1") == 800  # the refused call reserved nothing
    ledger.reserve(amount=200, ceilings=CEIL, run_id="eval-1", run_bound=1000)  # exactly fits


def test_every_refusal_code_is_a_known_holder_code(ledger):
    codes = set()
    ledger.reserve(amount=9_000, ceilings=CEIL, run_id="a", run_bound=10**9)
    for kwargs in (
        dict(amount=2_000, run_id="a", run_bound=10**9),  # window
        dict(amount=2_000, run_id="b", run_bound=100),  # run
    ):
        with pytest.raises(Refusal) as err:
            ledger.reserve(ceilings=CEIL, **kwargs)
        codes.add(err.value.code)
    assert codes == {"window_ceiling_reached", "run_bound_reached"}
    assert codes <= set(HOLDER_REFUSALS)


def test_concurrent_reservations_cannot_overshoot(tmp_path, clock):
    led = Ledger(tmp_path / "c.sqlite3", clock=clock)
    ok, refused = [], []
    barrier = threading.Barrier(8)

    def call():
        barrier.wait()
        try:
            led.reserve(amount=1100, ceilings=CEIL, run_id="eval-c", run_bound=3500)
            ok.append(1)
        except Refusal as exc:
            refused.append(exc.code)

    threads = [threading.Thread(target=call) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(ok) == 3 and refused == ["run_bound_reached"] * 5
    assert led.run_total("eval-c") <= 3500
    led.close()


def test_100_fresh_run_ids_stop_at_the_window_ceiling_and_roll_off(ledger, clock):
    granted = 0
    code = None
    for n in range(100):
        try:
            rid = ledger.reserve(amount=1000, ceilings=CEIL, run_id=f"eval-{n}", run_bound=5000)
            ledger.settle(rid, 1000)
            granted += 1
        except Refusal as exc:
            code = exc.code
    assert granted == 10 and code == "window_ceiling_reached"
    assert ledger.window_total(CEIL.window_seconds) == 10_000
    clock.now += CEIL.window_seconds + 1  # the window slides past every reservation
    ledger.reserve(amount=1000, ceilings=CEIL, run_id="eval-after", run_bound=5000)


def test_unmatched_cap_bucket_is_one_shared_bucket_per_window(ledger, clock):
    for n in range(3):
        rid = ledger.reserve(amount=1000, ceilings=CEIL, cap_bound=3000)  # fresh ids, one bucket
        ledger.settle(rid, 1000)
    with pytest.raises(Refusal) as err:
        ledger.reserve(amount=1000, ceilings=CEIL, cap_bound=3000)
    assert err.value.code == "run_bound_reached"
    clock.now += CEIL.window_seconds + 1
    ledger.reserve(amount=1000, ceilings=CEIL, cap_bound=3000)


def test_ledger_survives_restart(tmp_path, clock):
    led = Ledger(tmp_path / "l.sqlite3", clock=clock)
    rid = led.reserve(amount=900, ceilings=CEIL, run_id="eval-r", run_bound=1000)
    led.settle(rid, 900)
    led.close()
    again = Ledger(tmp_path / "l.sqlite3", clock=clock)
    with pytest.raises(Refusal):
        again.reserve(amount=900, ceilings=CEIL, run_id="eval-r", run_bound=1000)
    again.close()


def test_second_holder_on_one_ledger_is_refused(tmp_path, ledger):
    with pytest.raises(LedgerLocked):
        Ledger(tmp_path / "ledger.sqlite3")


def test_files_are_holder_only(tmp_path, ledger):
    ledger.reserve(amount=1, ceilings=CEIL, run_id="x", run_bound=10)
    for name in os.listdir(tmp_path):
        mode = stat.S_IMODE(os.stat(tmp_path / name).st_mode)
        assert mode & 0o077 == 0, (name, oct(mode))


def test_release_returns_the_reservation_until_the_release_cap(ledger):
    for _ in range(3):
        rid = ledger.reserve(amount=1000, ceilings=CEIL, run_id="eval-s", run_bound=5000)
        assert ledger.release(rid, CEIL) is True
    assert ledger.window_total(CEIL.window_seconds) == 0
    with pytest.raises(Refusal) as err:  # refused before any send
        ledger.reserve(amount=1000, ceilings=CEIL, run_id="eval-s", run_bound=5000)
    assert err.value.code == "release_cap_reached"


def test_release_beyond_the_cap_keeps_the_reservation(ledger):
    tight = Ceilings(3600, 10_000, 1)
    first = ledger.reserve(amount=500, ceilings=tight, run_id="r", run_bound=5000)
    second = ledger.reserve(amount=500, ceilings=tight, run_id="r", run_bound=5000)
    assert ledger.release(first, tight) is True
    assert ledger.release(second, tight) is False
    assert ledger.window_total(3600) == 500


def test_kept_reservation_holds_until_window_rolls_off(ledger, clock):
    rid = ledger.reserve(amount=9_500, ceilings=CEIL, run_id="eval-k", run_bound=10**6)
    ledger.keep(rid)
    with pytest.raises(Refusal):
        ledger.reserve(amount=1000, ceilings=CEIL, run_id="eval-k2", run_bound=10**6)
    clock.now += CEIL.window_seconds + 1
    ledger.reserve(amount=1000, ceilings=CEIL, run_id="eval-k2", run_bound=10**6)


def test_settling_above_the_reservation_is_recorded_and_the_next_call_refused(ledger):
    rid = ledger.reserve(amount=600, ceilings=CEIL, run_id="eval-o", run_bound=1000)
    ledger.settle(rid, 1500)  # provider-injected tokens above the reservation
    assert ledger.run_total("eval-o") == 1500
    with pytest.raises(Refusal) as err:
        ledger.reserve(amount=1, ceilings=CEIL, run_id="eval-o", run_bound=1000)
    assert err.value.code == "run_bound_reached"


@pytest.mark.parametrize("bad", [0, -1, 1.5, "10", True, None])
def test_ceilings_must_be_positive_integers(bad):
    with pytest.raises(ValueError):
        positive_int(bad, "x")
    with pytest.raises(ValueError):
        Ceilings.from_table(
            {"window_seconds": bad, "window_max_tokens": 1, "window_max_released": 1}
        )


def test_window_seconds_has_an_upper_limit():
    with pytest.raises(ValueError):
        Ceilings.from_table(
            {"window_seconds": 31 * 86400, "window_max_tokens": 1, "window_max_released": 1}
        )
    Ceilings.from_table(
        {"window_seconds": 30 * 86400, "window_max_tokens": 1, "window_max_released": 1}
    )


def test_exactly_one_bound_kind(ledger):
    with pytest.raises(ValueError):
        ledger.reserve(amount=1, ceilings=CEIL)
    with pytest.raises(ValueError):
        ledger.reserve(amount=1, ceilings=CEIL, run_bound=1, cap_bound=1)
