"""Validate a converted dataset against the flex-pi reference schema.

Compares meta/info.json feature-by-feature with docs/flexpi/soft_bag_zipping.info.json
(the schema flex-pi/soft_bag_zipping actually ships), then opens real files: every
episode's parquet (columns, dtypes, row count, global index continuity), and for
the first N episodes decodes the first frame of every mp4/mkv and checks the 32-D
state (rot6d rows orthonormal, grippers in [0, 1]).

Exit code 0 = PASS, 1 = FAIL. Use it before handing a dataset to the training team.

Usage:
    python scripts/validate_flexpi_dataset.py /path/to/dataset
    python scripts/validate_flexpi_dataset.py /path/to/dataset --decode_episodes 3
    python scripts/validate_flexpi_dataset.py /path/to/dataset --no_strict_shape   # tiny test datasets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = REPO / "docs/flexpi/soft_bag_zipping.info.json"


class Report:
    def __init__(self) -> None:
        self.rows: List[Tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        return bool(ok)

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.rows)

    def print(self) -> None:
        w = max(len(n) for n, _, _ in self.rows) + 2
        for name, ok, detail in self.rows:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<{w}} {detail}")
        print(f"\n{'PASS' if self.passed else 'FAIL'}: {sum(ok for _, ok, _ in self.rows)}/{len(self.rows)} checks")


def _rot6d_ok(v: np.ndarray, tol: float = 1e-3) -> bool:
    r0, r1 = v[:3], v[3:6]
    return abs(np.linalg.norm(r0) - 1) < tol and abs(np.linalg.norm(r1) - 1) < tol and abs(float(r0 @ r1)) < tol


def validate(dataset: Path, reference: Path, decode_episodes: int = 1, strict_shape: bool = True) -> Report:
    import pyarrow.parquet as pq

    rep = Report()
    info_p = dataset / "meta/info.json"
    if not rep.check("meta/info.json exists", info_p.exists(), str(info_p)):
        return rep
    info = json.load(open(info_p))
    ref = json.load(open(reference)) if reference.exists() else None
    rep.check("codebase_version == v2.1", info.get("codebase_version") == "v2.1", str(info.get("codebase_version")))
    rep.check("robot_type", info.get("robot_type") == (ref or {}).get("robot_type", "yam"), str(info.get("robot_type")))
    rep.check("fps == 30", info.get("fps") == 30, str(info.get("fps")))
    for key in ("data_path", "video_path", "chunks_size", "splits", "total_episodes", "total_frames"):
        rep.check(f"info.{key} present", key in info, str(info.get(key))[:60])

    feats = info.get("features", {})
    if ref:
        rf = ref["features"]
        rep.check("feature keys == reference", set(feats) == set(rf),
                  f"missing={sorted(set(rf) - set(feats))} extra={sorted(set(feats) - set(rf))}")
        for k in sorted(set(feats) & set(rf)):
            a, b = feats[k], rf[k]
            ok = a.get("dtype") == b.get("dtype") and a.get("names") == b.get("names")
            detail = f"dtype={a.get('dtype')}"
            if strict_shape:
                ok = ok and list(a.get("shape", [])) == list(b.get("shape", []))
                detail += f" shape={a.get('shape')} (ref {b.get('shape')})"
            if "info" in b:
                ai, bi = a.get("info", {}), b["info"]
                for ik in ("video.codec", "video.pix_fmt", "video.is_depth_map", "video.channels", "depth_unit",
                           "depth_dtype", "depth_encoder", "depth_ext", "depth_packing"):
                    if ik in bi:
                        ok = ok and ai.get(ik) == bi[ik]
                        if ai.get(ik) != bi[ik]:
                            detail += f" {ik}={ai.get(ik)}!={bi[ik]}"
            rep.check(f"feature {k}", ok, detail)

    for fn in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl", "camera_intrinsics.json"):
        rep.check(f"meta/{fn} exists", (dataset / "meta" / fn).exists())
    if (dataset / "meta/tasks.jsonl").exists():
        tasks = [json.loads(l) for l in open(dataset / "meta/tasks.jsonl") if l.strip()]
        rep.check("exactly one task", len(tasks) == 1, tasks[0]["task"][:70] if tasks else "")
    eps = [json.loads(l) for l in open(dataset / "meta/episodes.jsonl") if l.strip()] if (dataset / "meta/episodes.jsonl").exists() else []
    rep.check("episodes.jsonl count == total_episodes", len(eps) == info.get("total_episodes"), f"{len(eps)}")
    rep.check("sum(length) == total_frames", sum(e["length"] for e in eps) == info.get("total_frames"), str(info.get("total_frames")))

    vid_keys = [k for k, v in feats.items() if v.get("dtype") in ("video", "depth_video")]
    cams = sorted({k.split(".")[-1] for k in vid_keys})
    if (dataset / "meta/camera_intrinsics.json").exists():
        intr = json.load(open(dataset / "meta/camera_intrinsics.json"))
        rep.check("intrinsics cover every camera", set(intr) == set(cams), f"{sorted(intr)}")

    # ---- per-episode files
    expected_cols = {"observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"}
    next_index = 0
    all_files_ok = True
    for e in eps:
        i, chunk = e["episode_index"], e["episode_index"] // info.get("chunks_size", 1000)
        pqp = dataset / info["data_path"].format(episode_chunk=chunk, episode_index=i)
        if not pqp.exists():
            all_files_ok = False
            rep.check(f"episode {i} parquet exists", False, str(pqp))
            continue
        t = pq.read_table(pqp)
        ok = set(t.column_names) == expected_cols and t.num_rows == e["length"]
        idx = t.column("index").to_numpy()
        ok = ok and idx[0] == next_index and np.all(np.diff(idx) == 1)
        next_index = int(idx[-1]) + 1
        st = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False))
        ok = ok and st.shape[1] == 32 and str(t.schema.field("observation.state").type) == "fixed_size_list<element: float>[32]"
        if not ok:
            all_files_ok = False
            rep.check(f"episode {i} parquet", False, f"cols={sorted(t.column_names)} rows={t.num_rows}/{e['length']} idx0={idx[0]}/{next_index}")
        for k in vid_keys:
            ext = ".mkv" if feats[k]["dtype"] == "depth_video" else ".mp4"
            vp = dataset / info["video_path"].format(episode_chunk=chunk, video_key=k, episode_index=i)
            vp = vp.with_suffix(ext)
            if not vp.exists():
                all_files_ok = False
                rep.check(f"episode {i} {k} file", False, str(vp))
    rep.check("all episode parquet/video files present & consistent", all_files_ok, f"{len(eps)} episodes, global index ends at {next_index}")

    # ---- decode + numeric sanity on the first N episodes
    import av

    for e in eps[:decode_episodes]:
        i, chunk = e["episode_index"], e["episode_index"] // info.get("chunks_size", 1000)
        t = pq.read_table(dataset / info["data_path"].format(episode_chunk=chunk, episode_index=i))
        st = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False))
        ac = np.stack(t.column("action").to_numpy(zero_copy_only=False))
        r6 = all(_rot6d_ok(row[3:9]) and _rot6d_ok(row[12:18]) for row in st[:: max(1, len(st) // 50)])
        rep.check(f"ep {i} rot6d rows orthonormal", r6)
        rep.check(f"ep {i} grippers in [0,1]", bool(np.all((st[:, 18:20] >= -1e-6) & (st[:, 18:20] <= 1 + 1e-6))),
                  f"L[{st[:,18].min():.2f},{st[:,18].max():.2f}] R[{st[:,19].min():.2f},{st[:,19].max():.2f}]")
        rep.check(f"ep {i} finite state/action", bool(np.isfinite(st).all() and np.isfinite(ac).all()))
        rep.check(f"ep {i} EE positions plausible (|p|<1.5 m)", bool(np.all(np.abs(st[:, [0, 1, 2, 9, 10, 11]]) < 1.5)),
                  f"L mean {np.round(st[:, :3].mean(0), 3).tolist()} R mean {np.round(st[:, 9:12].mean(0), 3).tolist()}")
        for k in vid_keys:
            is_depth = feats[k]["dtype"] == "depth_video"
            vp = (dataset / info["video_path"].format(episode_chunk=chunk, video_key=k, episode_index=i)).with_suffix(".mkv" if is_depth else ".mp4")
            try:
                with av.open(str(vp)) as c:
                    s = c.streams.video[0]
                    n = 0
                    first = None
                    for fr in c.decode(s):
                        if first is None:
                            first = fr.to_ndarray(format="gray16le" if is_depth else "rgb24")
                        n += 1
                shape_ok = first is not None and list(first.shape[:2]) == list(feats[k]["shape"][:2])
                cnt_ok = n == e["length"]
                extra = f"{n}/{e['length']} frames, first {None if first is None else first.shape} {None if first is None else first.dtype}"
                if is_depth:
                    extra += f", nonzero {100 * float((first > 0).mean()):.0f}%, max {int(first.max())} mm, codec {s.codec_context.name}/{s.codec_context.pix_fmt}"
                    ok = shape_ok and cnt_ok and first.dtype == np.uint16 and s.codec_context.name == "ffv1"
                else:
                    ok = shape_ok and cnt_ok and first.dtype == np.uint8 and s.codec_context.name == "h264"
                rep.check(f"ep {i} {k} decodes", ok, extra)
            except Exception as ex:  # noqa: BLE001
                rep.check(f"ep {i} {k} decodes", False, f"{type(ex).__name__}: {ex}")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--decode_episodes", type=int, default=1)
    ap.add_argument("--no_strict_shape", action="store_true")
    a = ap.parse_args()
    rep = validate(Path(a.dataset), Path(a.reference), a.decode_episodes, strict_shape=not a.no_strict_shape)
    print(f"dataset:   {a.dataset}\nreference: {a.reference}\n")
    rep.print()
    sys.exit(0 if rep.passed else 1)


if __name__ == "__main__":
    main()
