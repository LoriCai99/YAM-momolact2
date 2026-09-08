"""move_to_start_position must use robot-only reads/commands, never the cameras."""

import numpy as np

from gello.env import RobotEnv
from gello.utils.launch_utils import move_to_start_position


class _Robot:
    def __init__(self):
        self.q = np.full(14, 0.5)
        self.cmds = []

    def num_dofs(self):
        return 14

    def get_joint_state(self):
        return self.q.copy()

    def command_joint_state(self, j):
        self.cmds.append(np.array(j))
        self.q = np.array(j)

    def get_observations(self):
        return {"joint_positions": self.q, "joint_velocities": np.zeros(14), "ee_pos_quat": np.zeros(7), "gripper_position": np.zeros(1)}


class _DeadCameraClient:
    """A camera client in the state that killed homing on 2026-09-08."""

    def get_obs_full(self):
        raise RuntimeError("camera server stopped publishing")

    def get_meta(self):
        return {}


def test_homing_completes_with_dead_cameras():
    robot = _Robot()
    env = RobotEnv(robot, control_rate_hz=2000, camera_client=_DeadCameraClient())
    left = {"agent": {"start_joints": [0.0] * 6 + [1.0]}}
    right = {"agent": {"start_joints": [0.0] * 6 + [1.0]}}
    move_to_start_position(env, True, left, right)
    assert len(robot.cmds) > 1, "should interpolate over several steps"
    assert np.allclose(robot.cmds[-1], [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0])
    d = np.abs(np.diff(np.stack(robot.cmds), axis=0)).max()
    assert d < 0.05, f"homing should be gentle: max per-step joint delta {d:.3f} rad"
