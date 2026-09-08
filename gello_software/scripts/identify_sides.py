"""Which GELLO leader and which CAN bus are physically on the operator's RIGHT?

Nothing is energized: leader reads are passive Dynamixel reads, follower reads are
zero-torque motor_on/off pings. You move each device by hand when prompted.

Then compares the result with configs/yam_left.yaml + yam_right.yaml and prints
whether teleop is crossed, whether the config FILES are named backwards (which
would silently swap left_joint/right_joint in every saved episode), and the fix.

Usage:  python scripts/identify_sides.py
"""

import sys
import time

import numpy as np
from omegaconf import OmegaConf

from gello.dynamixel.driver import DynamixelDriver

LEADERS = {"FTAO9WCV": [8, 9, 10, 11, 12, 13, 14], "FTAO9WPU": [1, 2, 3, 4, 5, 6, 7]}
PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_{}-if00-port0"
THRESH = 0.15  # rad; noise floor is ~0.001


def read_leaders():
    out = {}
    for sn, ids in LEADERS.items():
        d = DynamixelDriver(ids, port=PORT.format(sn), baudrate=57600, max_retries=1, use_fake_fallback=False)
        for _ in range(8):
            j = d.get_joints()
        d.close()
        out[sn] = np.array(j[:6])
    return out


def read_followers():
    from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface, MotorType

    out = {}
    for ch in ["can0", "can1"]:
        mci = DMSingleMotorCanInterface(channel=ch, bustype="socketcan", control_mode=ControlMode.MIT)
        q = []
        for mid in [1, 2, 3, 4, 5, 6]:
            q.append(mci.motor_on(mid, MotorType.DM4310).position)
            mci.motor_off(mid)
        mci.close()
        out[ch] = np.array(q)
    return out


def which_moved(before, after, label):
    deltas = {k: float(np.abs(after[k] - before[k]).max()) for k in before}
    for k, v in deltas.items():
        print(f"    {k}: max |delta| {v:.3f} rad ({np.degrees(v):.1f} deg)")
    moved = [k for k, v in deltas.items() if v > THRESH]
    if len(moved) != 1:
        print(f"!! expected exactly one {label} to move, got {moved}. Move ONLY the right-hand one, ~30 deg. Aborting.")
        sys.exit(1)
    return moved[0]


def main():
    print("Reading both leaders and both arms (motors stay off)...")
    l0, f0 = read_leaders(), read_followers()

    input("\n>>> Move the LEADER on your RIGHT-hand side by ~30 deg on a couple of joints, then press Enter: ")
    right_leader = which_moved(l0, read_leaders(), "leader")
    print(f"==> RIGHT leader = {right_leader}")

    input("\n>>> Now move the ROBOT ARM on your RIGHT-hand side by ~30 deg (it is back-drivable), then press Enter: ")
    right_bus = which_moved(f0, read_followers(), "arm")
    print(f"==> RIGHT arm = {right_bus}")
    left_leader = next(k for k in LEADERS if k != right_leader)
    left_bus = "can0" if right_bus == "can1" else "can1"

    L = OmegaConf.to_container(OmegaConf.load("configs/yam_left.yaml"), resolve=True)
    R = OmegaConf.to_container(OmegaConf.load("configs/yam_right.yaml"), resolve=True)
    cfg = {"left": (L["agent"]["port"].split("Converter_")[-1][:8], L["robot"]["channel"]),
           "right": (R["agent"]["port"].split("Converter_")[-1][:8], R["robot"]["channel"])}
    print(f"\nPhysical:  LEFT = leader {left_leader} + arm {left_bus}   |   RIGHT = leader {right_leader} + arm {right_bus}")
    print(f"Config:    yam_left.yaml = leader {cfg['left'][0]} + {cfg['left'][1]}   |   yam_right.yaml = leader {cfg['right'][0]} + {cfg['right'][1]}")

    ok_left = cfg["left"] == (left_leader, left_bus)
    ok_right = cfg["right"] == (right_leader, right_bus)
    paired_but_swapped = cfg["left"] == (right_leader, right_bus) and cfg["right"] == (left_leader, left_bus)
    print()
    if ok_left and ok_right:
        print("VERDICT: CORRECT. Each config names the physical side it drives. Teleop and data labels are right.")
    elif paired_but_swapped:
        print("VERDICT: FILES NAMED BACKWARDS. Teleop looks right (each leader drives its own side) but yam_left.yaml")
        print("         drives the physical RIGHT pair, so saved episodes would have left/right joints SWAPPED.")
        print(f"FIX: in yam_left.yaml set  port -> {left_leader}, joint_ids/signs/gripper of that leader, channel: {left_bus}")
        print(f"     in yam_right.yaml set port -> {right_leader}, its dynamixel_config,          channel: {right_bus}")
    else:
        print("VERDICT: CROSSED. A leader drives the opposite side's arm.")
        print(f"FIX: yam_left.yaml  -> leader {left_leader} (its ids/signs/gripper) + channel {left_bus}")
        print(f"     yam_right.yaml -> leader {right_leader} (its ids/signs/gripper) + channel {right_bus}")
    print("\nPaste this whole output back and the config will be updated accordingly.")


if __name__ == "__main__":
    main()
