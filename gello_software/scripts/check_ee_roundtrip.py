"""Offline check of the EE-space path: FK -> IK round trip over a recorded episode.

No robot needed. Reports the joint error the IK introduces (should be ~1e-3 rad or
less), the per-frame solve time (must be well under 33 ms for real-time use), and any
frames where IK failed to converge.

Usage: python scripts/check_ee_roundtrip.py --episode 237 [--data_dir /home/evan/yam_data/put_pen_in_bag]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gello.utils.yam_kinematics import YamArmKinematics, ee_roundtrip_trajectory  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--data_dir", default="/home/evan/yam_data/put_pen_in_bag")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = whole episode")
    a = ap.parse_args()
    ep = f"{a.episode:06d}"
    rows = json.load(open(os.path.join(a.data_dir, ep, f"{ep}.json")))
    if a.max_frames:
        rows = rows[: a.max_frames]
    def _vec(v):  # rows store joints as stringified lists
        return np.asarray(json.loads(v) if isinstance(v, str) else v, dtype=float)

    q = np.array([np.concatenate([_vec(r["left_joint"]), _vec(r["right_joint"])]) for r in rows], dtype=float)
    kin = YamArmKinematics()
    _, err, secs, fails = ee_roundtrip_trajectory(q, kin)
    ms = secs * 1e3
    print(f"episode {ep}: {len(rows)} frames")
    print(f"  IK joint error vs recorded: median {np.median(err):.2e} rad, p99 {np.percentile(err, 99):.2e}, max {err.max():.2e}")
    print(f"  IK solve time per frame (both arms): p50 {np.percentile(ms, 50):.1f} ms, p99 {np.percentile(ms, 99):.1f} ms, max {ms.max():.1f} ms")
    print(f"  frames where IK did not converge: {fails}")
    ok = fails == 0 and err.max() < 5e-3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
