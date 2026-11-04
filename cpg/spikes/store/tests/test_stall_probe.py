"""S23 (P0-16): the pure parts of ``stall_probe.py`` (window slicing and statistics of the PING samples)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import stall_probe as SP  # noqa: E402


def test_summarize_counts_a_ping_issued_inside_the_window_that_returns_after_it():
    # a PING issued at t=10.0 stuck behind a 3 s write: latency 3.0 s, issued inside [9.9, 10.5]
    samples = [(9.0, 0.001), (9.5, 0.002), (10.0, 3.0), (13.1, 0.001)]
    s = SP.summarize_pings(samples, 9.9, 10.5)
    assert s["n"] == 1 and s["max_ms"] == 3000.0 and s["over1s"] == 1 and s["over100ms"] == 1


def test_summarize_excludes_samples_outside_the_window_and_handles_empty():
    samples = [(1.0, 0.5), (2.0, 0.001), (3.0, 0.002)]
    s = SP.summarize_pings(samples, 1.5, 3.5)
    assert s["n"] == 2 and s["max_ms"] == 2.0 and s["over100ms"] == 0
    assert SP.summarize_pings(samples, 10, 11) == {"n": 0}


def test_summarize_percentiles_are_over_the_sorted_latencies():
    samples = [(float(i), (i + 1) / 1000) for i in range(100)]           # 1..100 ms
    s = SP.summarize_pings(samples, 0, 99)
    assert s["n"] == 100 and s["p50_ms"] == 51.0 and s["p99_ms"] == 100.0 and s["max_ms"] == 100.0


def test_summarize_max_is_the_true_maximum_and_over1s_is_not_over100ms():
    samples = [(float(i), 0.001) for i in range(100)] + [(100.0, 0.5)]    # 100 x 1 ms and one 500 ms stall
    s = SP.summarize_pings(samples, 0, 100)
    assert s["n"] == 101 and s["max_ms"] == 500.0 and s["p99_ms"] == 1.0
    assert s["over100ms"] == 1 and s["over1s"] == 0
