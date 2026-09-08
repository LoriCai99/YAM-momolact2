"""BimanualAgent.act must split only per-arm vectors.

Regression for the 2026-09-08 collection crash: RobotEnv.get_obs() now carries
depth maps and per-camera float timestamps alongside RGB, and the splitter
called .shape[0] on every value -> AttributeError on the float, and would have
split images by rows.
"""

import numpy as np

from gello.agents.agent import Agent, BimanualAgent


class _Fake(Agent):
    def __init__(self, value: float):
        self.value = value
        self.seen = None

    def act(self, obs):
        self.seen = obs
        return np.full(7, self.value)


def test_split_vectors_pass_through_everything_else():
    left, right = _Fake(1.0), _Fake(2.0)
    obs = {
        "joint_positions": np.arange(14.0),
        "joint_velocities": np.zeros(14),
        "gripper_position": np.array([0.1, 0.9]),
        "front_camera_rgb": np.zeros((360, 640, 3), np.uint8),
        "front_camera_depth": np.zeros((360, 640, 1), np.uint16),
        "front_camera_timestamp": 1234.5,
    }
    action = BimanualAgent(left, right).act(obs)
    assert action.tolist() == [1.0] * 7 + [2.0] * 7
    assert left.seen["joint_positions"].tolist() == list(range(7))
    assert right.seen["joint_positions"].tolist() == list(range(7, 14))
    assert left.seen["gripper_position"].tolist() == [0.1] and right.seen["gripper_position"].tolist() == [0.9]
    assert left.seen["front_camera_rgb"].shape == (360, 640, 3)
    assert right.seen["front_camera_depth"].shape == (360, 640, 1)
    assert left.seen["front_camera_timestamp"] == 1234.5
