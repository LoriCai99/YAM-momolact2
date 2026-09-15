#!/usr/bin/env python3
"""Capture the pose you have posed the arms into, and write it as ``agent.start_joints``.

READ-ONLY with respect to motion: it uses the same call ``ping_motors.py`` uses
(``motor_on`` -> read the feedback frame -> ``motor_off``). No position, velocity or
torque is ever commanded, so the arms do not move. Pose them by hand first -- with no
launcher running the 400 ms watchdog has already de-energised them, so they are limp.

    # look only: print what would be written
    python scripts/capture_start_pose.py --left_config configs/yam_left.yaml \
                                         --right_config configs/yam_right.yaml
    # commit it into both configs (a .bak copy is kept)
    python scripts/capture_start_pose.py ... --write

``start_joints`` is ``[j1..j6, gripper]``. Only the six arm joints are captured; the
gripper entry keeps whatever the config already has (1.0 = open) unless --gripper is given,
because the config value is a command in [0, 1] while the motor reports raw radians.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from typing import List, Optional

ARM_JOINTS = 6


def read_arm(channel: str) -> List[float]:
    """Six joint positions in radians. Enables each motor only to read its feedback
    frame, then disables it again -- nothing is commanded, so nothing moves."""
    from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface, MotorType

    ci = DMSingleMotorCanInterface(channel=channel, bustype="socketcan", control_mode=ControlMode.MIT)
    try:
        out = []
        for motor_id in range(1, ARM_JOINTS + 1):
            info = ci.motor_on(motor_id, MotorType.DM4310)
            ci.motor_off(motor_id)
            if getattr(info, "error_message", "") != "normal":
                raise RuntimeError(f"{channel} motor {motor_id}: {info.error_message}")
            out.append(float(info.position))
        return out
    finally:
        ci.close()


def channel_of(path: str) -> str:
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return str(cfg["robot"]["channel"])


def current_start_joints(path: str) -> Optional[List[float]]:
    m = re.search(r"^\s*start_joints:\s*\[([^\]]*)\]", open(path).read(), re.M)
    return [float(x) for x in m.group(1).split(",")] if m else None


def write_start_joints(path: str, joints: List[float]) -> None:
    src = open(path).read()
    m = re.search(r"^(\s*start_joints:\s*)\[[^\]]*\]", src, re.M)
    if not m:
        raise SystemExit(f"{path}: no start_joints line to replace")
    shutil.copyfile(path, path + ".bak")
    body = ", ".join(f"{v:.4f}" for v in joints)
    open(path, "w").write(src[: m.start()] + f"{m.group(1)}[{body}]" + src[m.end():])


def main() -> int:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--left_config", default=os.path.join(here, "configs/yam_left.yaml"))
    ap.add_argument("--right_config", default=os.path.join(here, "configs/yam_right.yaml"))
    ap.add_argument("--gripper", type=float, default=None,
                    help="gripper command to store (default: keep what the config has; 1.0 = open)")
    ap.add_argument("--write", action="store_true", help="actually edit the configs (keeps a .bak)")
    a = ap.parse_args()

    print("Pose both arms by hand NOW; nothing is commanded, so they will not move.\n")
    rows = []
    for side, path in (("LEFT ", a.left_config), ("RIGHT", a.right_config)):
        ch = channel_of(path)
        joints = read_arm(ch)
        old = current_start_joints(path) or [0.0] * (ARM_JOINTS + 1)
        grip = a.gripper if a.gripper is not None else (old[ARM_JOINTS] if len(old) > ARM_JOINTS else 1.0)
        new = [round(v, 4) for v in joints] + [grip]
        print(f"{side} ({ch}, {os.path.basename(path)})")
        print(f"  now:  [{', '.join(f'{v:+.4f}' for v in joints)}]  gripper {grip}")
        print(f"  was:  [{', '.join(f'{v:+.4f}' for v in old[:ARM_JOINTS])}]")
        print(f"  delta:[{', '.join(f'{j - o:+.4f}' for j, o in zip(joints, old[:ARM_JOINTS]))}]  (rad)")
        rows.append((path, new))
    big = [f"{os.path.basename(p)} j{i+1} {v:+.2f} rad" for p, n in rows for i, v in enumerate(n[:ARM_JOINTS]) if abs(v) > 1.0]
    if big:
        print("\nNOTE: large angles in the captured pose: " + "; ".join(big))
        print("Make sure this really is the pose you want the arms to home to.")
    if not a.write:
        print("\nDry run. Re-run with --write to store these as start_joints.")
        return 0
    for path, new in rows:
        write_start_joints(path, new)
        print(f"wrote {path}  (backup at {os.path.basename(path)}.bak)")
    print("\nNext launch homes to this pose. Verify with:\n"
          "  python -m experiments.reset_to_home --left_config_path=<left> --right_config_path=<right>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
