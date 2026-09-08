"""Quality check of raw episodes before converting or scaling up collection.

For each episode dir under <data_dir>: frame count, duration, effective fps,
control-tick regularity (dropped ticks), joint motion per arm (frozen-state
detection), gripper usage, per-camera frame counts (RGB == depth == rows),
duplicated camera frames, depth validity, and the instruction string.
Prints WARN lines for anything that would make the episode poor training data.

Usage:
    python scripts/inspect_episodes.py                       # storage.base_dir/task_directory from configs/yam_left.yaml
    python scripts/inspect_episodes.py /home/evan/yam_data/put_pen_in_bag
    python scripts/inspect_episodes.py <dir> --episodes 3 7  # only these
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


def _cfg_path(rel: str) -> str:
    """Resolve a configs/… path from the script's own location, so these tools work
    from any cwd (they used to die with FileNotFoundError when run from the repo root)."""
    import os

    if os.path.exists(rel):
        return rel
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # gello_software/
    return os.path.join(here, rel)

CAMS = ("front", "left", "right")


def inspect(ep: str, fps_nominal: float = 30.0) -> bool:
    ok = True
    jp = [p for p in glob.glob(f"{ep}/[0-9]*.json")]
    mp = f"{ep}/meta.json"
    name = os.path.basename(ep)
    if not jp or not os.path.exists(mp):
        print(f"{name}: INCOMPLETE (no json/meta) -- not saved with 'a', or the run crashed mid-episode. Delete it.")
        return False
    rows = json.load(open(jp[0]))
    m = json.load(open(mp))
    T = len(rows)
    ts = np.array([r["timestamp"] for r in rows])
    dt = np.diff(ts) * 1e3
    dropped = int((dt > 2 * 1000 / fps_nominal).sum())
    print(f"\n{name}: {T} frames, {m.get('duration_s', 0):.1f}s, {m.get('effective_fps', 0):.2f} fps | \"{m.get('instruction', '')[:70]}\"")
    print(f"  ticks: p50 {np.percentile(dt, 50):.1f} ms p99 {np.percentile(dt, 99):.1f} max {dt.max():.1f} | dropped {dropped}")
    if T < 20 * fps_nominal:
        print(f"  WARN: only {T / fps_nominal:.0f}s -- flex-pi episodes are 25-140 s"); ok = False
    if dropped > 0.01 * T:
        print(f"  WARN: {dropped} dropped ticks (>1%) -- loop not holding {fps_nominal:.0f} Hz"); ok = False
    L = np.array([json.loads(r["left_joint"]) for r in rows]); R = np.array([json.loads(r["right_joint"]) for r in rows])
    for side, A in (("left", L), ("right", R)):
        rng = np.ptp(A[:, :6], axis=0)
        g = A[:, 6]
        grasps = int(np.sum((g[1:] < 0.3) & (g[:-1] >= 0.3)))
        print(f"  {side:5}: joint travel max {rng.max():.2f} rad | gripper [{g.min():.2f},{g.max():.2f}] open>0.9 {100*np.mean(g>0.9):.0f}% mid {100*np.mean((g>0.15)&(g<0.85)):.0f}% | grasps {grasps}")
        if rng.max() < 0.02:
            print(f"  WARN: {side} arm did not move -- frozen joint state?"); ok = False
        if g.max() < 0.9 or g.min() > 0.3:
            print(f"  WARN: {side} gripper never {'opened' if g.max() < 0.9 else 'closed'} fully -- trigger calibration/spring?"); ok = False
    for cam in CAMS:
        rgb = sorted(glob.glob(f"{ep}/{cam}_rgb/*")); dep = sorted(glob.glob(f"{ep}/{cam}_depth/*"))
        cts = np.array([r.get("camera_timestamps", {}).get(cam, 0.0) for r in rows]); dup = int((np.diff(cts) == 0).sum())
        cm = m.get("cameras", {}).get(cam, {})
        line = f"  {cam:5}: rgb {len(rgb)} depth {len(dep)} dup {100*dup/max(T-1,1):.1f}%"
        if len(rgb) != T or len(dep) != T:
            line += "  WARN: frame count != rows"; ok = False
        if dep:
            d = cv2.imread(dep[len(dep) // 2], cv2.IMREAD_UNCHANGED)
            valid = float((d > 0).mean()) if d is not None else 0.0
            scale = cm.get("depth_scale_m_per_unit") or 0.0
            med = float(np.median(d[d > 0]) * scale * 1000) if d is not None and valid > 0 else 0.0
            line += f" | depth valid {100*valid:.0f}% median {med:.0f} mm"
            if valid < 0.4:
                line += "  WARN: mostly invalid depth"; ok = False
        if not cm.get("intrinsics") or not cm.get("depth_scale_m_per_unit"):
            line += "  WARN: no intrinsics/depth scale in meta"; ok = False
        if dup > 0.15 * T:
            line += "  WARN: >15% duplicated frames"; ok = False
        print(line)
    stale_n = sum(1 for r in rows if r.get("camera_stale"))
    if stale_n:
        runs, cur = [], 0
        for r in rows:
            cur = cur + 1 if r.get("camera_stale") else 0
            runs.append(cur)
        print(f"  WARN: {stale_n} frames ({100*stale_n/T:.1f}%) had a stale camera; longest stall {max(runs)/fps_nominal:.1f}s"); ok = False
    if m.get("image_write_errors"):
        print(f"  WARN: image write errors: {m['image_write_errors'][:2]}"); ok = False
    print(f"  => {'OK' if ok else 'CHECK WARNINGS'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?")
    ap.add_argument("--episodes", type=int, nargs="*")
    a = ap.parse_args()
    d = a.data_dir
    if d is None:
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(OmegaConf.load(_cfg_path("configs/yam_left.yaml")), resolve=True)["storage"]
        d = os.path.join(cfg["base_dir"], cfg["task_directory"])
    eps = sorted(p for p in glob.glob(os.path.join(d, "[0-9]*")) if os.path.isdir(p))
    if a.episodes:
        eps = [p for p in eps if int(os.path.basename(p)) in set(a.episodes)]
    if not eps:
        print(f"no episodes under {d}"); sys.exit(1)
    results = [inspect(e) for e in eps]
    print(f"\n{sum(results)}/{len(results)} episodes clean under {d}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
