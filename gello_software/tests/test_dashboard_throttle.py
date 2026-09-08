"""KBReset must not render the dashboard on every control tick."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import time  # noqa: E402

import numpy as np  # noqa: E402

from gello.data_utils.keyboard_interface import KBReset  # noqa: E402


def _data():
    cams = {n: np.random.randint(0, 255, (360, 640, 3), np.uint8) for n in ("left", "front", "right")}
    return {"phase": "collecting", "status_text": "x", "traj_idx": 1, "total_traj": 1, "step_idx": 1,
            "max_steps": 10, "obs_count": 1, "joint_positions": np.zeros(14),
            "joint_velocities": np.zeros(14), "cameras": cams}


def test_render_is_throttled_to_render_hz(monkeypatch):
    kb = KBReset()
    calls = []
    monkeypatch.setattr(kb, "_render_dashboard", lambda d: calls.append(time.time()))
    kb.render_hz = 10.0
    t0 = time.time()
    while time.time() - t0 < 0.5:
        kb.update(_data())
        time.sleep(1 / 60)
    assert 3 <= len(calls) <= 7, f"expected ~5 renders in 0.5 s at 10 Hz, got {len(calls)}"


def test_full_render_is_fast():
    kb = KBReset()
    d = _data()
    kb._render_dashboard(d)  # warm-up
    t = time.perf_counter()
    for _ in range(5):
        kb._render_dashboard(d)
    per = (time.perf_counter() - t) / 5 * 1e3
    assert per < 25, f"dashboard render {per:.1f} ms (was ~55 ms with surfarray+smoothscale)"
