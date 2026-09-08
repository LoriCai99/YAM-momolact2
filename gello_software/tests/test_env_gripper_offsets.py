"""Start/dynamic offsets must never touch the gripper channels."""

import numpy as np

from gello.env import RobotEnv


class _Robot:
    def __init__(self):
        self.q = np.zeros(14)
        self.q[6], self.q[13] = 1.0, 0.0  # left open, right closed (old right home pose)
        self.cmd = None

    def num_dofs(self):
        return 14

    def get_joint_state(self):
        return self.q.copy()

    def command_joint_state(self, j):
        self.cmd = np.array(j)

    def get_observations(self):
        return {"joint_positions": self.q, "joint_velocities": np.zeros(14), "ee_pos_quat": np.zeros(7), "gripper_position": np.zeros(1)}


def test_gripper_commands_are_absolute_even_if_trigger_moved_between_launch_and_start():
    env = RobotEnv(_Robot(), control_rate_hz=1000)
    launch = np.zeros(14); launch[6], launch[13] = 1.0, 1.0        # both triggers released at launch
    launch[0] = 0.3                                                # left arm joint 0 not at the follower pose
    env.set_original_offset(launch)
    at_s = launch.copy(); at_s[13] = 0.73                           # right trigger parked partly squeezed at 's'
    env.set_dynamic_offset(at_s)
    assert env._original_offset[6] == 0 and env._original_offset[13] == 0
    assert env._dynamic_offset[6] == 0 and env._dynamic_offset[13] == 0
    assert env._original_offset[0] == 0.3                          # arm offsets still work
    squeeze = at_s.copy(); squeeze[13] = 0.0                        # full squeeze on the right
    env.step_command_only(squeeze)
    assert env._robot.cmd[13] == 0.0                                # fully closed, not shifted by +0.27


def test_default_gripper_indices_follow_yam_layout():
    assert RobotEnv(_Robot(), control_rate_hz=1000)._gripper_indices == (6, 13)
