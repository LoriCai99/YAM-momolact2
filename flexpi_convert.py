#!/usr/bin/env python
"""Convert raw gello_software episodes into a LeRobot v2.1 dataset laid out
exactly like flex-pi's real-world YAM datasets (e.g. flex-pi/soft_bag_zipping).

Why a hand-rolled writer instead of ``LeRobotDataset.create``: the lerobot in
this repo's envs is 0.5.x, which only writes v3.0, and flex-pi's depth streams
are an *extension* to stock LeRobot (``dtype: depth_video``, FFV1 gray16le in
.mkv) that no LeRobotDataset version emits. The v2.1 layout is small and fully
specified by the reference dataset's meta/ (cached under docs/flexpi/), so it
is written directly with pyarrow + PyAV. No lerobot import: this runs in the
``yam`` env, which has i2rt + mujoco for forward kinematics.

Input: episode dirs NNNNNN/ produced by gello/data_utils/data_saver.py:
    NNNNNN.json, meta.json, <cam>_rgb/*.jpg|png, <cam>_depth/*.png (uint16)

Output (LeRobot v2.1):
    meta/info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl, camera_intrinsics.json
    data/chunk-000/episode_000000.parquet                            32-D state/action + indices
    videos/chunk-000/observation.images.<cam>/episode_000000.mp4      h264 yuv420p
    videos/chunk-000/observation.depth_ffv1.<cam>/episode_000000.mkv  ffv1 gray16le, millimetres

32-D state/action layout (grouped by field -- flex-pi convention):
    [0:3]  left EE position (m)      [3:9]   left rot6d
    [9:12] right EE position (m)     [12:18] right rot6d
    [18:20] left_gripper, right_gripper (i2rt command space, 0..1, 1 = open)
    [20:26] left joints 0-5 (rad)    [26:32] right joints 0-5 (rad)
rot6d = first two ROWS of the 3x3 rotation matrix, row-major (Zhou et al. 2019).
EE pose = FK of the i2rt YAM MuJoCo model at site ``grasp_site`` (fingertip
TCP) in each arm's own base frame. Verified against flex-pi frame 0: their home
pose reads [0.2477, 0.0001, 0.1708] m; this model at q=0 gives [0.245, 0, 0.174].

Usage:
    python flexpi_convert.py                     # reads storage:/flexpi: from gello_software/configs/yam_left.yaml
    python flexpi_convert.py --data_dir D --output_dir O --task "..." --overwrite
    python gello_software/scripts/validate_flexpi_dataset.py O     # afterwards
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CODEBASE_VERSION = "v2.1"
CHUNKS_SIZE = 1000
BUILDER = "gello_software/flexpi_convert.py"

STATE_NAMES: List[str] = (
    ["left_pos_x", "left_pos_y", "left_pos_z"]
    + [f"left_rot6d_{i}" for i in range(6)]
    + ["right_pos_x", "right_pos_y", "right_pos_z"]
    + [f"right_rot6d_{i}" for i in range(6)]
    + ["left_gripper", "right_gripper"]
    + [f"left_joint_{i}" for i in range(6)]
    + [f"right_joint_{i}" for i in range(6)]
)
assert len(STATE_NAMES) == 32

DEFAULT_CAMERA_MAP = {"front": "cam_high", "left": "cam_left_wrist", "right": "cam_right_wrist"}
CAM_ORDER = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
REFERENCE_HW = (360, 640)


# ----------------------------------------------------------------------------- kinematics


def rot6d_from_matrix(R: np.ndarray) -> np.ndarray:
    """First two rows of R, row-major -- the flex-pi / Zhou et al. convention."""
    return np.concatenate([R[0, :3], R[1, :3]]).astype(np.float64)


class YamFK:
    """Forward kinematics for one YAM arm via the i2rt MuJoCo model."""

    def __init__(self, ee_model: str = "gripper", site: str = "grasp_site"):
        import mujoco
        from i2rt.robots.kinematics import Kinematics
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

        gripper = GripperType.LINEAR_4310 if ee_model == "gripper" else GripperType.NO_GRIPPER
        xml = combine_arm_and_gripper_xml(ArmType.YAM, gripper)
        self.nq = int(mujoco.MjModel.from_xml_path(xml).nq)
        self.site = site
        self._k = Kinematics(xml, site)
        self._q = np.zeros(self.nq)

    def pose(self, q6: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._q[:] = 0.0
        self._q[:6] = q6
        T = self._k.fk(self._q)
        return T[:3, 3].copy(), rot6d_from_matrix(T[:3, :3])


def joints14_to_state32(q14: np.ndarray, fk: YamFK) -> np.ndarray:
    L, R = q14[:7], q14[7:14]
    lp, lr = fk.pose(L[:6])
    rp, rr = fk.pose(R[:6])
    # Grippers are i2rt command-space [0, 1] (1 = open). The follower reports a few
    # hundredths past its calibrated limits when pressed hard (e.g. -0.01 fully
    # closed); flex-pi's range is [0, 1], so clip -- it is overshoot, not signal.
    grip = np.clip([L[6], R[6]], 0.0, 1.0)
    return np.concatenate([lp, lr, rp, rr, grip, L[:6], R[:6]]).astype(np.float32)


# ----------------------------------------------------------------------------- raw episodes


def _numeric_sorted(paths: List[Path]) -> List[Path]:
    def key(p: Path):
        return (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)

    return sorted(paths, key=key)


def _parse_vec(v: Any) -> List[float]:
    if isinstance(v, str):
        v = json.loads(v)
    return [float(x) for x in v]


def discover_episode_dirs(data_dir: Path) -> List[Path]:
    dirs = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and any(d.glob("*.json")) and any(d.glob("*_rgb")):
            dirs.append(d)
    return dirs


def load_episode(ep_dir: Path) -> Dict[str, Any]:
    json_candidates = [p for p in ep_dir.glob("*.json") if p.name != "meta.json"]
    if not json_candidates:
        raise FileNotFoundError(f"{ep_dir}: no per-frame json")
    json_path = _numeric_sorted(json_candidates)[0]
    with open(json_path) as f:
        rows = json.load(f)
    if not rows:
        raise ValueError(f"{json_path}: empty episode")

    qpos = np.array([_parse_vec(r["left_joint"]) + _parse_vec(r["right_joint"]) for r in rows], dtype=np.float64)
    next_qpos = None
    if all("next_left_joint" in r and "next_right_joint" in r for r in rows):
        next_qpos = np.array(
            [_parse_vec(r["next_left_joint"]) + _parse_vec(r["next_right_joint"]) for r in rows], dtype=np.float64
        )
    if qpos.shape[1] != 14:
        raise ValueError(f"{json_path}: expected 14 joint values per frame, got {qpos.shape[1]}")

    instruction = None
    for key in ("language_instruction", "instruction", "task"):
        v = rows[0].get(key)
        if isinstance(v, str) and v.strip():
            instruction = v.strip()
            break

    meta: Dict[str, Any] = {}
    if (ep_dir / "meta.json").exists():
        with open(ep_dir / "meta.json") as f:
            meta = json.load(f)

    cams: Dict[str, Dict[str, Any]] = {}
    for d in sorted(ep_dir.iterdir()):
        if not (d.is_dir() and d.name.endswith("_rgb")):
            continue
        cam = d.name[: -len("_rgb")]
        rgb = _numeric_sorted([p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        depth_dir = ep_dir / f"{cam}_depth"
        depth = _numeric_sorted([p for p in depth_dir.iterdir() if p.suffix.lower() == ".png"]) if depth_dir.exists() else None
        if len(rgb) != len(rows):
            raise ValueError(f"{ep_dir}: {cam} has {len(rgb)} rgb frames but json has {len(rows)} rows")
        if depth is not None and len(depth) != len(rows):
            raise ValueError(f"{ep_dir}: {cam} has {len(depth)} depth frames but json has {len(rows)} rows")
        cams[cam] = {"rgb": rgb, "depth": depth, "meta": (meta.get("cameras") or {}).get(cam, {})}

    return {
        "dir": ep_dir,
        "qpos": qpos,
        "next_qpos": next_qpos,
        "instruction": instruction,
        "meta": meta,
        "cams": cams,
        "length": len(rows),
    }


def build_actions(ep: Dict[str, Any], action_mode: str) -> np.ndarray:
    q = ep["qpos"]
    if action_mode == "next_joint_fields":
        if ep["next_qpos"] is not None:
            return ep["next_qpos"]
        action_mode = "next_state"
    if action_mode == "next_state":
        a = np.empty_like(q)
        a[:-1] = q[1:]
        a[-1] = q[-1]
        return a
    if action_mode == "copy_state":
        return q.copy()
    raise ValueError(f"unknown action_mode {action_mode}")


# ----------------------------------------------------------------------------- video encoding


def encode_rgb(paths: List[str], out: str, fps: int, crf: int) -> Tuple[int, int]:
    import av
    import cv2

    first = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise IOError(f"cannot read {paths[0]}")
    h, w = first.shape[:2]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with av.open(out, "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.time_base = Fraction(1, fps)
        s.options = {"crf": str(crf), "preset": "medium"}
        for i, p in enumerate(paths):
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2] != (h, w):
                raise IOError(f"bad/missized frame {p}")
            frame = av.VideoFrame.from_ndarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), format="rgb24")
            frame.pts = i
            for pkt in s.encode(frame):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return h, w


def depth_to_mm(raw: np.ndarray, depth_scale_m_per_unit: float) -> np.ndarray:
    """Native uint16 depth -> uint16 millimetres, the unit flex-pi stores.

    ``mm = round_half_even(raw * scale_mm)`` with ``scale_mm`` rounded to 9 decimals so
    that D405 (1e-4 m/unit -> 0.1 mm/unit) and D435 (1e-3 -> 1.0) are exact. Values
    that would exceed uint16 are clipped; 0 ("no return") stays 0. This is the ONE
    place the conversion is defined -- tests and downstream readers should use it.
    """
    scale_mm = round(float(depth_scale_m_per_unit) * 1000.0, 9)
    return np.clip(np.rint(raw.astype(np.float64) * scale_mm), 0, 65535).astype(np.uint16)


def encode_depth(paths: List[str], out: str, fps: int, depth_scale_m_per_unit: float) -> Tuple[int, int]:
    """PNG16 in native sensor units -> FFV1 gray16le in MILLIMETRES (lossless)."""
    import av
    import cv2

    first = cv2.imread(paths[0], cv2.IMREAD_UNCHANGED)
    if first is None or first.dtype != np.uint16 or first.ndim != 2:
        raise IOError(f"{paths[0]}: expected 2-D uint16 PNG, got {None if first is None else (first.dtype, first.shape)}")
    h, w = first.shape
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with av.open(out, "w", format="matroska") as c:
        s = c.add_stream("ffv1", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "gray16le"
        s.time_base = Fraction(1, fps)
        for i, p in enumerate(paths):
            raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if raw is None or raw.dtype != np.uint16 or raw.shape != (h, w):
                raise IOError(f"bad/missized depth frame {p}")
            mm = depth_to_mm(raw, depth_scale_m_per_unit)
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(mm), format="gray16le")
            frame.pts = i
            for pkt in s.encode(frame):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return h, w


def _encode_job(job: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    if job["kind"] == "rgb":
        hw = encode_rgb(job["paths"], job["out"], job["fps"], job["crf"])
    else:
        hw = encode_depth(job["paths"], job["out"], job["fps"], job["depth_scale"])
    return {**{k: v for k, v in job.items() if k != "paths"}, "hw": hw, "seconds": time.time() - t0}


# ----------------------------------------------------------------------------- v2.1 writers


def write_parquet(path: Path, state: np.ndarray, action: np.ndarray, ep_idx: int, index0: int, fps: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    T = len(state)

    def fsl(a: np.ndarray):
        return pa.FixedSizeListArray.from_arrays(pa.array(a.astype(np.float32).reshape(-1), type=pa.float32()), 32)

    table = pa.table(
        {
            "observation.state": fsl(state),
            "action": fsl(action),
            "timestamp": pa.array((np.arange(T) / fps).astype(np.float32), type=pa.float32()),
            "frame_index": pa.array(np.arange(T, dtype=np.int64), type=pa.int64()),
            "episode_index": pa.array(np.full(T, ep_idx, dtype=np.int64), type=pa.int64()),
            "index": pa.array(np.arange(index0, index0 + T, dtype=np.int64), type=pa.int64()),
            "task_index": pa.array(np.zeros(T, dtype=np.int64), type=pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _stats(a: np.ndarray) -> Dict[str, Any]:
    return {
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "count": [int(len(a))],
    }


def build_info(
    n_episodes: int, total_frames: int, fps: int, robot_type: str, hw: Dict[str, Tuple[int, int]],
    task_name: str, source_dirs: List[str], cams: List[str],
) -> Dict[str, Any]:
    # flex-pi ships names as a NESTED list ([[...32 names...]]); mirror it exactly so
    # our info.json is byte-compatible with what their tooling was built against.
    vec = {"dtype": "float32", "shape": [32], "names": [list(STATE_NAMES)]}
    feats: Dict[str, Any] = {
        "observation.state": dict(vec),
        "action": dict(vec),
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for cam in cams:
        h, w = hw[cam]
        feats[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "rgb"],
            "info": {"video.height": h, "video.width": w, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "video.fps": fps, "video.channels": 3, "has_audio": False},
        }
    for cam in cams:
        h, w = hw[cam]
        feats[f"observation.depth_ffv1.{cam}"] = {
            "dtype": "depth_video", "shape": [h, w, 1], "names": ["height", "width", "depth"],
            "info": {"video.height": h, "video.width": w, "video.codec": "ffv1", "video.pix_fmt": "gray16le",
                     "video.is_depth_map": True, "video.fps": fps, "video.channels": 1, "has_audio": False,
                     "depth_unit": "mm", "depth_dtype": "uint16", "depth_encoder": "ffv1",
                     "depth_ext": ".mkv", "depth_packing": "gray16le"},
        }
    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": n_episodes * 2 * len(cams),
        "total_chunks": max(1, math.ceil(n_episodes / CHUNKS_SIZE)),
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feats,
        "task_name": task_name,
        "source_dirs": source_dirs,
        "builder": BUILDER,
    }


# ----------------------------------------------------------------------------- driver


def convert(
    data_dir: Path,
    output_dir: Path,
    task: Optional[str] = None,
    fps: int = 30,
    camera_map: Optional[Dict[str, str]] = None,
    robot_type: str = "yam",
    ee_model: str = "gripper",
    ee_site: str = "grasp_site",
    action_mode: str = "next_joint_fields",
    rgb_crf: int = 20,
    workers: int = 8,
    depth_scale_override: Optional[Dict[str, float]] = None,
    episodes: Optional[List[int]] = None,
    overwrite: bool = False,
    require_all_cameras: bool = True,
    quiet: bool = False,
) -> Dict[str, Any]:
    from tqdm import tqdm

    camera_map = dict(DEFAULT_CAMERA_MAP if camera_map is None else camera_map)
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    ep_dirs = discover_episode_dirs(data_dir)
    if episodes is not None:
        ep_dirs = [d for d in ep_dirs if int(d.name) in set(episodes)]
    if not ep_dirs:
        raise FileNotFoundError(f"no episodes under {data_dir}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists (use --overwrite)")
        shutil.rmtree(output_dir)
    (output_dir / "meta").mkdir(parents=True)

    fk = YamFK(ee_model, ee_site)
    log = (lambda *a, **k: None) if quiet else print
    log(f"FK model: ee_model={ee_model} site={ee_site} nq={fk.nq}; {len(ep_dirs)} episodes from {data_dir}")

    jobs: List[Dict[str, Any]] = []
    episodes_meta: List[Dict[str, Any]] = []
    stats_lines: List[Dict[str, Any]] = []
    intrinsics_acc: Dict[str, List[Dict[str, float]]] = {c: [] for c in CAM_ORDER}
    task_text: Optional[str] = task
    total_frames = 0
    cams_used: Optional[List[str]] = None
    warnings: List[str] = []

    for new_idx, ep_dir in enumerate(tqdm(ep_dirs, desc="episodes", disable=quiet)):
        ep = load_episode(ep_dir)
        if task_text is None:
            task_text = ep["instruction"]
        elif ep["instruction"] and ep["instruction"] != task_text:
            warnings.append(f"{ep_dir.name}: instruction differs from task ('{ep['instruction'][:40]}...')")

        # --- cameras -> flex-pi keys
        present = {camera_map[c]: c for c in ep["cams"] if c in camera_map}
        missing = [k for k in CAM_ORDER if k not in present]
        if missing and require_all_cameras:
            raise ValueError(f"{ep_dir}: missing cameras {missing} (have {list(ep['cams'])})")
        cams_this = [k for k in CAM_ORDER if k in present]
        cams_used = cams_this if cams_used is None else [c for c in cams_used if c in cams_this]

        for key in cams_this:
            cam = present[key]
            info = ep["cams"][cam]
            chunk = new_idx // CHUNKS_SIZE
            jobs.append({"kind": "rgb", "ep": new_idx, "key": key, "fps": fps, "crf": rgb_crf,
                         "paths": [str(p) for p in info["rgb"]],
                         "out": str(output_dir / f"videos/chunk-{chunk:03d}/observation.images.{key}/episode_{new_idx:06d}.mp4")})
            if info["depth"] is None:
                raise ValueError(f"{ep_dir}: {cam} has no depth frames; flex-pi requires depth. "
                                 f"Was storage.save_depth true during collection?")
            scale = (depth_scale_override or {}).get(cam) or (depth_scale_override or {}).get(key) \
                or info["meta"].get("depth_scale_m_per_unit")
            if scale is None:
                raise ValueError(f"{ep_dir}: no depth_scale_m_per_unit for {cam} in meta.json. D435 is 0.001 and "
                                 f"D405 is 0.0001 -- pass --depth_scale {cam}=<m/unit> rather than guessing.")
            jobs.append({"kind": "depth", "ep": new_idx, "key": key, "fps": fps, "depth_scale": float(scale),
                         "paths": [str(p) for p in info["depth"]],
                         "out": str(output_dir / f"videos/chunk-{chunk:03d}/observation.depth_ffv1.{key}/episode_{new_idx:06d}.mkv")})
            intr = info["meta"].get("intrinsics")
            if intr:
                intrinsics_acc[key].append(intr)

        # --- 32-D state / action
        qpos, act14 = ep["qpos"], build_actions(ep, action_mode)
        state = np.stack([joints14_to_state32(q, fk) for q in qpos])
        action = np.stack([joints14_to_state32(q, fk) for q in act14])
        chunk = new_idx // CHUNKS_SIZE
        write_parquet(output_dir / f"data/chunk-{chunk:03d}/episode_{new_idx:06d}.parquet",
                      state, action, new_idx, total_frames, fps)
        total_frames += ep["length"]
        episodes_meta.append({"episode_index": new_idx, "tasks": [task_text or ""], "length": ep["length"],
                              "source": ep_dir.name})
        stats_lines.append({"episode_index": new_idx,
                            "stats": {"observation.state": _stats(state), "action": _stats(action)}})

    # --- videos (parallel, one process per video)
    hw: Dict[str, Tuple[int, int]] = {}
    # spawn, not fork: the caller may already have threads (PyAV/cv2/data saver).
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=multiprocessing.get_context("spawn")) as pool:
        futs = [pool.submit(_encode_job, j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="videos", disable=quiet):
            r = f.result()
            hw.setdefault(r["key"], tuple(r["hw"]))
            if tuple(r["hw"]) != hw[r["key"]]:
                raise ValueError(f"inconsistent resolution for {r['key']}: {r['hw']} vs {hw[r['key']]}")
    for key, (h, w) in hw.items():
        if (h, w) != REFERENCE_HW:
            warnings.append(f"{key}: {w}x{h} differs from flex-pi reference 640x360")

    # --- meta/
    cams_used = cams_used or []
    task_text = task_text or "perform the task"
    info = build_info(len(ep_dirs), total_frames, fps, robot_type, hw, task_name=data_dir.name,
                      source_dirs=[d.name for d in ep_dirs], cams=cams_used)
    with open(output_dir / "meta/info.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(output_dir / "meta/tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_text}) + "\n")
    with open(output_dir / "meta/episodes.jsonl", "w") as f:
        for e in episodes_meta:
            f.write(json.dumps({k: e[k] for k in ("episode_index", "tasks", "length")}) + "\n")
    with open(output_dir / "meta/episodes_stats.jsonl", "w") as f:
        for s in stats_lines:
            f.write(json.dumps(s) + "\n")
    intr_out: Dict[str, Any] = {}
    for key in cams_used:
        lst = intrinsics_acc.get(key) or []
        if not lst:
            warnings.append(f"{key}: no intrinsics recorded in any episode meta.json")
            continue
        h, w = hw[key]
        intr_out[key] = {k: float(np.mean([i[k] for i in lst])) for k in ("fx", "fy", "cx", "cy")}
        intr_out[key].update({"width": w, "height": h})
    if intr_out:
        with open(output_dir / "meta/camera_intrinsics.json", "w") as f:
            json.dump(intr_out, f, indent=2)
    with open(output_dir / "meta/conversion.json", "w") as f:  # provenance, not part of v2.1
        json.dump({"builder": BUILDER, "ee_model": ee_model, "ee_site": ee_site, "action_mode": action_mode,
                   "rgb_crf": rgb_crf, "camera_map": camera_map, "source_data_dir": str(data_dir),
                   "converted_at": time.strftime("%Y-%m-%d %H:%M:%S"), "warnings": warnings}, f, indent=2)

    log(f"\nWrote {len(ep_dirs)} episodes / {total_frames} frames / {len(jobs)} videos -> {output_dir}")
    for wmsg in warnings:
        log(f"  WARNING: {wmsg}")
    return {"episodes": len(ep_dirs), "frames": total_frames, "videos": len(jobs), "hw": hw,
            "warnings": warnings, "output_dir": str(output_dir), "cams": cams_used}


def load_defaults_from_yaml(cfg_path: Path) -> Dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    st, fp = cfg.get("storage", {}), cfg.get("flexpi", {}) or {}
    base = Path(st.get("base_dir", ".")).expanduser()
    task_dir = st.get("task_directory", "task")
    return {
        "data_dir": base / task_dir,
        "output_dir": Path(fp.get("output_dir") or (base / f"{task_dir}_flexpi_v21")),
        "task": st.get("language_instruction"),
        "fps": int(cfg.get("hz", 30)),
        "camera_map": fp.get("camera_map") or DEFAULT_CAMERA_MAP,
        "robot_type": fp.get("robot_type", "yam"),
        "ee_model": fp.get("ee_model", "gripper"),
        "ee_site": fp.get("ee_site", "grasp_site"),
        "action_mode": fp.get("action_mode", "next_joint_fields"),
        "rgb_crf": int(fp.get("rgb_crf", 20)),
        "workers": int(fp.get("workers", 8)),
    }


def main(argv: Optional[List[str]] = None) -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(here / "gello_software/configs/yam_left.yaml"))
    ap.add_argument("--data_dir")
    ap.add_argument("--output_dir")
    ap.add_argument("--task", help="instruction written to tasks.jsonl (default: from config / episodes)")
    ap.add_argument("--fps", type=int)
    ap.add_argument("--robot_type")
    ap.add_argument("--ee_model", choices=["gripper", "arm"])
    ap.add_argument("--ee_site")
    ap.add_argument("--action_mode", choices=["next_joint_fields", "next_state", "copy_state"])
    ap.add_argument("--rgb_crf", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--episodes", type=int, nargs="*", help="only these raw episode indices")
    ap.add_argument("--depth_scale", action="append", default=[], metavar="CAM=M_PER_UNIT",
                    help="override/supply depth scale, e.g. front=0.001 (only for episodes without meta.json)")
    ap.add_argument("--allow_missing_cameras", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    d = load_defaults_from_yaml(Path(args.config)) if Path(args.config).exists() else {}
    pick = lambda k, default=None: getattr(args, k) if getattr(args, k) is not None else d.get(k, default)
    ds_override = {}
    for item in args.depth_scale:
        k, v = item.split("=")
        ds_override[k] = float(v)

    convert(
        data_dir=Path(pick("data_dir")), output_dir=Path(pick("output_dir")), task=pick("task"),
        fps=pick("fps", 30), camera_map=d.get("camera_map"), robot_type=pick("robot_type", "yam"),
        ee_model=pick("ee_model", "gripper"), ee_site=pick("ee_site", "grasp_site"),
        action_mode=pick("action_mode", "next_joint_fields"), rgb_crf=pick("rgb_crf", 20),
        workers=pick("workers", 8), depth_scale_override=ds_override or None, episodes=args.episodes,
        overwrite=args.overwrite, require_all_cameras=not args.allow_missing_cameras,
    )


if __name__ == "__main__":
    main()
