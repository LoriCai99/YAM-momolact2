import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from gello.cameras.camera import CameraDriver
from gello.robots.robot import Robot


class Rate:
    """Fixed-schedule rate limiter: the average period is exactly 1/rate.

    The previous version measured each period from the END of the previous
    sleep, so every tick carried the sleep-granularity overshoot (~0.5 ms) and
    the collection loop settled at 29.2 Hz instead of 30 (tick p50 34.0 ms).
    Ticks are now scheduled on an absolute timeline: a tick that finishes late
    is followed by a shorter sleep, so the mean holds. If the loop falls more
    than one full period behind (a real stall), the schedule is re-based
    instead of firing a burst of catch-up ticks.
    """

    def __init__(self, rate: float):
        self.rate = rate
        self.period = 1.0 / rate
        self.next = time.perf_counter() + self.period
        self.last = time.time()  # kept for callers that read it

    def sleep(self) -> None:
        now = time.perf_counter()
        remaining = self.next - now
        if remaining < -self.period:
            self.next = now + self.period  # stalled: re-base rather than burst
        else:
            if remaining > 0.002:
                time.sleep(remaining - 0.0015)  # coarse sleep, then a short accurate spin
            while time.perf_counter() < self.next:
                time.sleep(0.0001)
            self.next += self.period
        self.last = time.time()


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
        camera_client: Optional[Any] = None,
        gripper_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self._robot = robot
        # Joint indices that are grippers. The start/dynamic offsets below exist so
        # the ARM does not jump when the leader is not exactly at the follower's
        # pose at 's'; a gripper must map ABSOLUTELY from its calibrated trigger.
        # Offsetting it turned "trigger parked at 0.73 when s was pressed" into
        # "every command shifted by +0.27 -> gripper never fully closes" (2026-09-08).
        # Default: YAM layout (6 arm joints + gripper per arm).
        if gripper_indices is None:
            n = robot.num_dofs()
            gripper_indices = tuple(range(6, n, 7)) if n % 7 == 0 else ()
        self._gripper_indices = tuple(int(i) for i in gripper_indices)
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict
        # When set, get_obs() pulls images from the camera server over ZMQ
        # instead of opening RealSense devices in-process. camera_dict is
        # then ignored. See gello/cameras/camera_client.py.
        self._camera_client = camera_client

    # dynamic offset is used in data collection to make sure the same
    # starting position between each episode.
        self._dynamic_offset = np.zeros(self._robot.num_dofs())
        self._original_offset = np.zeros(self._robot.num_dofs())

    def _zero_grippers(self, offset: np.ndarray) -> np.ndarray:
        for i in self._gripper_indices:
            if i < len(offset):
                offset[i] = 0.0
        return offset

    def set_original_offset(self, gello_joints: np.ndarray) -> None:
        self._original_offset = self._zero_grippers(gello_joints - self._robot.get_joint_state())

    def set_dynamic_offset(self, gello_joints: np.ndarray) -> None:
        self._dynamic_offset = self._zero_grippers(
            gello_joints - self._robot.get_joint_state() - self._original_offset
        )

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray, reset: Optional[bool] = False) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        self.step_command_only(joints, reset=reset)
        return self.get_obs()

    def step_command_only(
        self, joints: np.ndarray, reset: Optional[bool] = False
    ) -> None:
        """Command the robot + sleep on the control rate. Does NOT read cameras.

        Use this inside tight inner loops (e.g. interpolated sub-steps of an
        action) so each tick doesn't pay for a full ``get_obs()``. Call
        ``get_obs()`` once after the loop when you actually need the obs.
        """
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)

        if reset:
            self._robot.command_joint_state(joints)
        else:
            self._robot.command_joint_state(joints - self._dynamic_offset)
        self._rate.sleep()

    def get_robot_state(self) -> Dict[str, Any]:
        """Robot-only observations (joint positions/velocities, EE pose, gripper).

        Same fields as ``get_obs()`` minus the ``*_rgb`` camera images. Use this
        when you only need joints (e.g. computing an interpolation target).
        """
        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        return {
            "joint_positions": robot_obs["joint_positions"],
            "joint_velocities": robot_obs["joint_velocities"],
            "ee_pos_quat": robot_obs["ee_pos_quat"],
            "gripper_position": robot_obs["gripper_position"],
        }

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations: Dict[str, Any] = {}
        if self._camera_client is not None:
            full = getattr(self._camera_client, "get_obs_full", None)
            if callable(full):
                resp = full()
                for name, image in resp["frames"].items():
                    observations[f"{name}_rgb"] = image
                for name, depth in (resp.get("depth") or {}).items():
                    observations[f"{name}_depth"] = depth if depth.ndim == 3 else depth[:, :, None]
                for name, ts in (resp.get("timestamps") or {}).items():
                    observations[f"{name}_timestamp"] = float(ts)
            else:  # older client: RGB only
                for name, image in self._camera_client.get_obs().items():
                    observations[f"{name}_rgb"] = image
        else:
            for name, camera in self._camera_dict.items():
                image, depth = camera.read()
                observations[f"{name}_rgb"] = image
                # Depth is already captured (and aligned to colour) by the driver;
                # keep it so data collection can record it. (H, W, 1) uint16, native
                # sensor units -- see get_camera_meta() for metres-per-unit.
                observations[f"{name}_depth"] = depth
                ts = getattr(camera, "last_frame_timestamp", None)
                if ts is not None:
                    observations[f"{name}_timestamp"] = ts

        observations.update(self.get_robot_state())
        return observations

    def get_camera_meta(self) -> Dict[str, Dict[str, Any]]:
        """Static per-camera metadata (intrinsics, depth scale, serial).

        Keyed like ``camera_dict`` (e.g. ``"left_camera"``). Drivers that do not
        expose a field simply omit it, so this works with dummy/saved cameras too.
        """
        meta: Dict[str, Dict[str, Any]] = {}
        if self._camera_client is not None:
            get_meta = getattr(self._camera_client, "get_meta", None)
            if callable(get_meta):
                try:
                    return {name: dict(entry or {}) for name, entry in get_meta().items()}
                except Exception as exc:  # pragma: no cover - server specific
                    return {"__error__": {"errors": [f"get_meta: {exc}"]}}
            return meta
        for name, camera in self._camera_dict.items():
            entry: Dict[str, Any] = {}
            for attr, key in (
                ("get_intrinsics", "intrinsics"),
                ("get_depth_scale", "depth_scale_m_per_unit"),
            ):
                fn = getattr(camera, attr, None)
                if callable(fn):
                    try:
                        entry[key] = fn()
                    except Exception as exc:  # pragma: no cover - driver specific
                        entry[key] = None
                        entry.setdefault("errors", []).append(f"{attr}: {exc}")
            device_id = getattr(camera, "device_id", None)
            if device_id is not None:
                entry["device_id"] = device_id
            meta[name] = entry
        return meta


def main() -> None:
    pass


if __name__ == "__main__":
    main()
