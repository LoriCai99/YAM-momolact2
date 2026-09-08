from typing import Dict

import numpy as np

from gello.robots.robot import Robot
from i2rt.robots.utils import GripperType


class YAMRobot(Robot):
    """A class representing a simulated YAM robot."""

    # i2rt's power-on handshake waits max_retry x 10 ms for each motor's reply
    # (default 5 = 50 ms). That is enough in a quiet process but not once the
    # launcher's camera capture threads and Dynamixel readers are contending for
    # the GIL: reproduced deterministically on this rig -- 50 ms fails on motor 4
    # or 5, 500 ms succeeds every time. Widen it before constructing the chain.
    HANDSHAKE_RETRIES = 50

    @staticmethod
    def _widen_handshake_window(max_retry: int) -> None:
        import i2rt.motor_drivers.can_interface as ci

        f = ci.CanInterface._send_message_get_response
        if f.__defaults__ and f.__defaults__[0] != max_retry:
            f.__defaults__ = (max_retry,) + tuple(f.__defaults__[1:])

    def __init__(self, channel="can0", connect_attempts: int = 3, retry_delay_s: float = 1.0):
        from i2rt.robots.get_robot import get_yam_robot

        self._channel = channel

        self._widen_handshake_window(self.HANDSHAKE_RETRIES)
        # Belt and braces: a genuinely dead motor still fails, just with a clearer
        # message that names what to check physically.
        last_exc = None
        for attempt in range(1, connect_attempts + 1):
            try:
                self.robot = get_yam_robot(channel=channel, gripper_type=GripperType.LINEAR_4310)
                break
            except (AssertionError, RuntimeError) as exc:
                if "fail to communicate" not in str(exc):
                    raise
                last_exc = exc
                print(f"[YAMRobot {channel}] attempt {attempt}/{connect_attempts}: {exc}")
                if attempt < connect_attempts:
                    import time

                    time.sleep(retry_delay_s)
        else:
            raise RuntimeError(
                f"YAMRobot({channel}): a motor did not answer the power-on handshake in "
                f"{connect_attempts} attempts. Check that arm's power/E-stop, then reseat the CAN "
                f"daisy-chain connector at the motor named above. Last error: {last_exc}"
            ) from last_exc

        # YAM has 7 joints (6 arm joints + 1 gripper)
        self._joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper",
        ]
        self._joint_state = self.get_joint_state()  # robot stays where it was when reboot
        # self._joint_state = np.zeros(7)  # robot goes immediately to reset position (avoid using)
        self._joint_velocities = np.zeros(7)  # 7 joints
        self._gripper_state = 0.0 # didn't use because joint_state includes gripper position

    def num_dofs(self) -> int:
        return 7  # YAM has 7 DOFs

    def is_alive(self) -> bool:
        """False once i2rt's control loop has stopped (motor watchdog / lost comms)."""
        chain = getattr(self.robot, "motor_chain", None)
        return bool(getattr(chain, "running", True))

    def _assert_alive(self) -> None:
        if not self.is_alive():
            raise RuntimeError(
                f"{self._channel}: the arm's control loop has stopped (motor watchdog tripped or "
                f"CAN comms lost). Joint state is frozen -- refusing to record it as real data. "
                f"Reset CAN and relaunch."
            )

    def get_joint_state(self) -> np.ndarray:
        # Get actual joint positions from I2RT robot (7 joints total)
        self._assert_alive()
        joint_pos = self.robot.get_joint_pos()
        # Ensure we have exactly 7 joints
        if len(joint_pos) > 7:
            joint_pos = joint_pos[:7]
        elif len(joint_pos) < 7:
            # Pad with zeros if we have fewer than 7 joints
            joint_pos = np.pad(joint_pos, (0, 7 - len(joint_pos)), "constant")

        self._joint_state = joint_pos
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert (
            len(joint_state) == self.num_dofs()
        ), f"Expected {self.num_dofs()} joint values, got {len(joint_state)}"

        dt = 0.01
        self._joint_velocities = (joint_state - self._joint_state) / dt
        self._joint_state = joint_state

        # Command the I2RT robot with all 7 joints (6 arm + 1 gripper)
        self.command_joint_pos(joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        ee_pos_quat = np.zeros(7)  # Placeholder for FK
        return {
            "joint_positions": self._joint_state,
            "joint_velocities": self._joint_velocities,
            "ee_pos_quat": ee_pos_quat,
            "gripper_position": np.array([self._gripper_state]),
        }

    def get_joint_pos(self):
        # Get 7 joints from I2RT robot (6 arm + 1 gripper)
        joint_pos = self.robot.get_joint_pos()
        # Ensure we return exactly 7 joints
        if len(joint_pos) > 7:
            joint_pos = joint_pos[:7]
        elif len(joint_pos) < 7:
            # Pad with zeros if we have fewer than 7 joints
            joint_pos = np.pad(joint_pos, (0, 7 - len(joint_pos)), "constant")
        return joint_pos

    def command_joint_pos(self, target_pos):
        # Ensure we send exactly 7 joints to the I2RT robot
        if len(target_pos) > 7:
            target_pos = target_pos[:7]
        elif len(target_pos) < 7:
            # Pad with zeros if we have fewer than 7 joints
            target_pos = np.pad(target_pos, (0, 7 - len(target_pos)), "constant")
        self.robot.command_joint_pos(np.array(target_pos))


def main():
    robot = YAMRobot()
    print(robot.get_observations())


if __name__ == "__main__":
    main()
