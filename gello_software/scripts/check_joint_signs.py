"""Find wrong ``joint_signs`` by comparing leader and follower motion directions.

Motors stay DE-ENERGIZED (each follower read is a zero-torque motor_on/off ping,
as ping_motors.py does), so nothing can move on its own. Both arms are
back-drivable by hand.

Procedure, one joint at a time:
  1. Move leader joint k by ~30 degrees and note the sign of its delta.
  2. Move the FOLLOWER's joint k by hand in the SAME physical direction.
  3. If the two deltas have opposite signs, that joint's sign in the config is
     wrong: flip it. The script says so in the last column.
Return both to the start pose (or press r to re-zero) before the next joint.

Usage:
    python scripts/check_joint_signs.py --side left
    python scripts/check_joint_signs.py --side right --config configs/yam_right.yaml
"""

import argparse
import select
import sys
import time

import numpy as np
from omegaconf import OmegaConf

from gello.dynamixel.driver import DynamixelDriver


def read_follower(mci, ids, motor_type):
    q = []
    for mid in ids:
        info = mci.motor_on(mid, motor_type)
        q.append(info.position)
        mci.motor_off(mid)
    return np.array(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg_path = args.config or f"configs/yam_{args.side}.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    agent, dxl = cfg["agent"], cfg["agent"]["dynamixel_config"]
    ids, signs = list(dxl["joint_ids"]), np.array(dxl["joint_signs"], dtype=float)
    channel = cfg["robot"]["channel"]

    from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface, MotorType

    print(f"{args.side.upper()}: leader {agent['port'].split('Converter_')[-1][:8]} ids={ids} signs={signs.tolist()}"
          f" | follower {channel} (motors stay off)")
    leader = DynamixelDriver(ids, port=agent["port"], baudrate=57600, max_retries=1, use_fake_fallback=False)
    for _ in range(10):
        leader.get_joints()
    mci = DMSingleMotorCanInterface(channel=channel, bustype="socketcan", control_mode=ControlMode.MIT)
    fids = [1, 2, 3, 4, 5, 6]

    def zero():
        return leader.get_joints()[:6].copy(), read_follower(mci, fids, MotorType.DM4310)

    l0, f0 = zero()
    print("\nMove ONE joint on the leader, then the same joint on the follower the same way.")
    print("Columns: leader delta (with config sign applied) | follower delta | verdict.  r = re-zero, q = quit\n")
    try:
        while True:
            lq = leader.get_joints()[:6]
            fq = read_follower(mci, fids, MotorType.DM4310)
            ld = signs * (lq - l0)
            fd = fq - f0
            rows = []
            for k in range(6):
                moving = abs(ld[k]) > 0.15 and abs(fd[k]) > 0.15
                if not moving:
                    v = "      "
                elif np.sign(ld[k]) == np.sign(fd[k]):
                    v = "OK    "
                else:
                    v = "FLIP  "
                rows.append(f"j{k + 1}: L{ld[k]:+6.2f} F{fd[k]:+6.2f} {v}")
            print("\r" + "  ".join(rows) + "   ", end="", flush=True)
            if select.select([sys.stdin], [], [], 0.0)[0]:
                ch = sys.stdin.readline().strip().lower()
                if ch == "q":
                    break
                if ch == "r":
                    l0, f0 = zero()
                    print("\n(re-zeroed)")
            time.sleep(0.15)
    finally:
        print()
        leader.close()
        mci.close()
        flips = [k + 1 for k in range(6) if abs(ld[k]) > 0.15 and abs(fd[k]) > 0.15 and np.sign(ld[k]) != np.sign(fd[k])]
        if flips:
            new = signs.copy()
            for k in flips:
                new[k - 1] *= -1
            print(f"Last reading says flip joint(s) {flips}. Suggested {cfg_path} value:")
            print(f"    joint_signs: {[float(x) for x in new]}")


if __name__ == "__main__":
    main()
