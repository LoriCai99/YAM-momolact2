"""Real-time replay sends exactly one frame per tick and refuses corrupted jumps."""
import numpy as np
import pytest

from gello.data_utils.data_replay import DataReplayer


class _FakeEnv:
    def __init__(self, start):
        self.q = np.array(start, dtype=float); self.cmds = []
    def get_obs(self):
        return {"joint_positions": self.q.copy()}
    def step(self, j, reset=False):
        self.step_command_only(j, reset); return self.get_obs()
    def step_command_only(self, j, reset=False):
        self.q = np.array(j, dtype=float); self.cmds.append(self.q.copy())


def _replayer(traj):
    r = DataReplayer.__new__(DataReplayer)  # skip __init__ (matplotlib / camera keys)
    r.demo = [{"left_joint": list(f[:7]), "right_joint": list(f[7:]), "timestamp": i / 30} for i, f in enumerate(traj)]
    r.left_camera_key = r.right_camera_key = r.front_camera_key = None
    return r


def test_realtime_one_command_per_frame(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    traj = np.cumsum(np.full((50, 14), 0.02), axis=0)  # 0.02 rad/frame, above the slow-mode cap
    env = _FakeEnv(np.zeros(14))
    _replayer(traj).replay(env, robot_trajectory=True, realtime=True)
    frame_cmds = [c for c in env.cmds if any(np.allclose(c, f) for f in traj)]
    assert len(env.cmds) >= 50 and np.allclose(env.cmds[-1], traj[-1])
    # after the approach, the 50 frames are commanded once each, in order
    assert np.allclose(np.array(env.cmds[-50:]), traj)


def test_realtime_refuses_corrupted_jump(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    traj = np.zeros((10, 14)); traj[5, 2] = 1.0  # a 1 rad jump between two frames
    with pytest.raises(RuntimeError):
        _replayer(traj).replay(_FakeEnv(np.zeros(14)), robot_trajectory=True, realtime=True)
