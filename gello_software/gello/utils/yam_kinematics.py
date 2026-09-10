"""YAM arm kinematics for end-effector-space replay / checks.

Wraps i2rt's MuJoCo model + mink IK for ONE arm (6 joints, gripper excluded), using the
same model and site the flex-pi dataset state is computed with (LINEAR_4310 gripper
model, site ``grasp_site``, per-arm base frame). See docs/FLEXPI_DATA_SPEC.md.
"""

from typing import Optional, Tuple

import numpy as np


class YamArmKinematics:
    def __init__(self, ee_model: str = "gripper", site: str = "grasp_site", limit_margin_rad: float = 0.1):
        """``limit_margin_rad`` widens the model's joint ranges for the IK limits: the
        physical arm rests ~0.05 rad past the model's joint-3 limit at its home pose
        (episode 000237 frames 1-35), and mink otherwise refuses those poses."""
        import mink
        import mujoco
        from i2rt.robots.kinematics import Kinematics
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

        gripper = GripperType.LINEAR_4310 if ee_model == "gripper" else GripperType.NO_GRIPPER
        xml = combine_arm_and_gripper_xml(ArmType.YAM, gripper)
        self._model = mujoco.MjModel.from_xml_path(xml)
        self.nq = int(self._model.nq)
        self.site = site
        self._k = Kinematics(xml, site)
        lim_model = mujoco.MjModel.from_xml_path(xml)
        lim_model.jnt_range[:6, 0] -= limit_margin_rad
        lim_model.jnt_range[:6, 1] += limit_margin_rad
        self._limits = [mink.ConfigurationLimit(lim_model)]
        self._q = np.zeros(self.nq)

    def fk(self, q6: np.ndarray) -> np.ndarray:
        """4x4 pose of the site in the arm base frame for 6 arm joints."""
        self._q[:] = 0.0
        self._q[:6] = q6
        return self._k.fk(self._q)

    def ik(self, T: np.ndarray, q_seed: np.ndarray, max_iters: int = 100,
           pos_threshold: float = 2e-4, ori_threshold: float = 2e-4) -> Tuple[bool, np.ndarray]:
        """Solve the 6 arm joints reaching pose ``T`` (4x4), seeded from ``q_seed`` (6)."""
        seed = np.zeros(self.nq)
        seed[:6] = q_seed
        ok, q = self._k.ik(T, self.site, init_q=seed, limits=self._limits, max_iters=max_iters,
                           pos_threshold=pos_threshold, ori_threshold=ori_threshold)
        return bool(ok), np.asarray(q[:6], dtype=float).copy()


def ee_roundtrip_trajectory(joints14: np.ndarray, kin: Optional[YamArmKinematics] = None):
    """For a (N,14) joint trajectory: FK each arm's 6 joints to its EE pose (what the
    dataset stores), then IK back, seeding each frame with the previous solution (the
    first frame with the recorded joints). Grippers pass through untouched.

    Returns (ik_joints14 (N,14), per-frame max |ik - recorded| over the 12 arm joints,
    per-frame IK solve seconds, number of frames where IK did not converge).
    """
    import time

    kin = kin or YamArmKinematics()
    N = joints14.shape[0]
    out = joints14.astype(float).copy()
    err = np.zeros(N)
    secs = np.zeros(N)
    fails = 0
    seeds = [joints14[0, :6].copy(), joints14[0, 7:13].copy()]
    for i in range(N):
        t0 = time.perf_counter()
        for a, (lo, hi) in enumerate(((0, 6), (7, 13))):
            q_rec = joints14[i, lo:hi]
            ok, q_ik = kin.ik(kin.fk(q_rec), seeds[a])
            fails += (not ok)
            out[i, lo:hi] = q_ik
            seeds[a] = q_ik
        secs[i] = time.perf_counter() - t0
        err[i] = np.abs(np.concatenate([out[i, :6] - joints14[i, :6], out[i, 7:13] - joints14[i, 7:13]])).max()
    return out, err, secs, fails
