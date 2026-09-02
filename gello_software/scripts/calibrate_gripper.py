"""Measure a GELLO trigger's true angular range and print a gripper_config line.

Both yam_left.yaml and yam_right.yaml shipped with IDENTICAL gripper_config
values copied from another workstation's GELLO build. If a trigger's real range
differs, the normalized gripper command saturates (or never reaches its limit)
and that arm's gripper appears stuck. Run this per arm and paste the result.

Usage:
    python scripts/calibrate_gripper.py --side right
    python scripts/calibrate_gripper.py --side left  --seconds 30
"""

import argparse
import time

import numpy as np

from gello.dynamixel.driver import DynamixelDriver

SIDES = {
    "left": ("FTAO9WPU", [1, 2, 3, 4, 5, 6, 7]),
    "right": ("FTAO9WCV", [8, 9, 10, 11, 12, 13, 14]),
}
PORT_FMT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_{}-if00-port0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    serial, ids = SIDES[args.side]
    port = PORT_FMT.format(serial)
    gripper_id = ids[-1]

    print(f"{args.side.upper()} GELLO ({serial}), gripper servo id {gripper_id}")
    d = DynamixelDriver(ids, port=port, baudrate=57600, max_retries=1,
                        use_fake_fallback=False)
    for _ in range(10):
        d.get_joints()

    print(f"\n>>> Work the trigger through its FULL range for {args.seconds:.0f}s:")
    print(">>> squeeze all the way closed, release all the way open, repeat.\n")

    lo, hi = float("inf"), float("-inf")
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        deg = d.get_joints()[-1] * 180.0 / np.pi
        lo, hi = min(lo, deg), max(hi, deg)
        bar = int(np.clip((deg - lo) / max(hi - lo, 1e-6), 0, 1) * 40)
        print(f"\r  now {deg:7.2f} deg   min {lo:7.2f}   max {hi:7.2f}   "
              f"[{'#' * bar}{'.' * (40 - bar)}]", end="", flush=True)
        time.sleep(0.02)
    d.close()

    span = hi - lo
    print(f"\n\nRange: {lo:.2f} .. {hi:.2f} deg  (span {span:.2f} deg)")
    if span < 10:
        print("!! Span under 10 deg -- the trigger barely moved. Either it is")
        print("!! mechanically stuck, or you did not work its full travel.")
        return
    # Trim 5% off each end so normal use never saturates past 0/1.
    pad = span * 0.05
    print("\nPaste into the matching config's dynamixel_config:\n")
    print(f"    gripper_config: [{gripper_id}, {lo + pad:.4f}, {hi - pad:.4f}]")
    print("\n(first value = servo id, then the OPEN-end and CLOSED-end angles;")
    print(" swap the two angles if the gripper ends up inverted in teleop)")


if __name__ == "__main__":
    main()
