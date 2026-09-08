"""RobotEnv.Rate must hold its mean period exactly, with jittery per-tick work."""

import time

import numpy as np

from gello.env import Rate


def test_rate_mean_period_is_exact_under_jittery_work():
    r = Rate(30.0)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    n = 150
    ticks = []
    for _ in range(n):
        time.sleep(float(rng.uniform(0.0, 0.020)))  # 0-20 ms of "work"
        r.sleep()
        ticks.append(time.perf_counter())
    mean_ms = (ticks[-1] - t0) / n * 1e3
    assert abs(mean_ms - 1000 / 30) < 0.15, f"mean tick {mean_ms:.2f} ms, want 33.33"
    d = np.diff(ticks) * 1e3
    assert np.percentile(d, 99) < 40, f"p99 tick {np.percentile(d, 99):.1f} ms"


def test_rate_rebases_after_a_stall_instead_of_bursting():
    r = Rate(30.0)
    r.sleep()
    time.sleep(0.2)  # a 6-period stall
    t = time.perf_counter()
    r.sleep()  # must not return immediately six times in a row
    r.sleep()
    assert time.perf_counter() - t > 0.025, "ticks after a stall should be spaced ~one period, not burst"
