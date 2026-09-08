"""Episode recorder for teleop data collection.

Layout written per episode, under ``<save_dir>/<task_directory>/NNNNNN/``::

    NNNNNN.json      per-frame records (legacy keys kept, see below)
    meta.json        episode-level metadata: instruction, fps, per-camera
                     intrinsics + depth scale + flex-pi role, timestamps
    left_rgb/  front_rgb/  right_rgb/     000000.jpg|png   RGB, 8-bit
    left_depth/ front_depth/ right_depth/ 000000.png       depth, 16-bit PNG in
                                          NATIVE sensor units -- multiply by
                                          meta.json cameras.<cam>.depth_scale_m_per_unit
                                          for metres (D435: 1e-3, D405: 1e-4)

Frames are written to disk *as they arrive* on a background pool, so an episode
never has to fit in RAM (RGB + depth for three 640x360 cameras is ~2 MB/frame,
i.e. 8+ GB for a two-minute episode at 30 Hz). The JSON/meta files are written
when the episode is saved; a discarded episode's directory is deleted.

Per-frame JSON keys are a superset of what the older MolmoAct-style saver wrote,
so ``molmoact_to_lerobot_v30.py`` and the eval tooling keep working unchanged:
``language_instruction``, ``left_joint``, ``right_joint``, ``next_left_joint``,
``next_right_joint`` (stringified lists), ``image_<cam>_rgb``.
New: ``frame_index``, ``timestamp`` (wall clock, s), ``camera_timestamps``,
``image_<cam>_depth``. ``flexpi_convert.py`` consumes this layout.

Control-loop contract (unchanged from the previous saver): ``add_observation``
per step, ``save_episode_json(buffer)`` from ``EpisodeSaverThread`` on 'a',
``reset_buffer()`` at the top of every episode (which is also the only signal
that an unsaved episode was discarded with 'b').
"""

import concurrent.futures
import json
import logging
import os
import shutil
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("data_saver")
logger.setLevel(logging.INFO)

# Our camera name -> flex-pi camera key. Overridable via ``camera_roles``.
DEFAULT_CAMERA_ROLES = {
    "front": "cam_high",
    "left": "cam_left_wrist",
    "right": "cam_right_wrist",
}
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
# Warn (once per episode) when this many image writes are queued: the disk is
# not keeping up with 30 Hz and RAM is starting to absorb the difference.
_BACKLOG_WARN = 180


def _to_list(x: Any) -> List[float]:
    return [float(v) for v in np.asarray(x).reshape(-1)]


class _Episode:
    """Bookkeeping for one episode directory while it is being written."""

    def __init__(self, index: int, root: str):
        self.index = index
        self.dir = os.path.join(root, f"{index:06d}")
        self.records: List[Dict[str, Any]] = []
        self.pending = 0  # image writes submitted but not finished
        self._cv = threading.Condition()
        self.started_at = time.time()
        self.save_pending = False
        self.saved = False
        self.discarded = False
        self.warned_backlog = False
        self.write_errors: List[str] = []
        os.makedirs(self.dir, exist_ok=True)

    def submitted(self) -> None:
        with self._cv:
            self.pending += 1

    def finished(self, error: Optional[str] = None) -> None:
        with self._cv:
            self.pending -= 1
            if error:
                self.write_errors.append(error)
            if self.pending == 0:
                self._cv.notify_all()

    def wait_writes(self, timeout: Optional[float] = None) -> bool:
        with self._cv:
            return self._cv.wait_for(lambda: self.pending == 0, timeout=timeout)


class DataSaver:
    def __init__(
        self,
        save_dir: str = "/home/sean/Desktop/YAM/gello_software/data",
        task_directory: str = "Testing_dir",
        language_instruction: str = "Test",
        saver_max_workers: Optional[int] = None,
        png_compress_level: int = 1,
        save_depth: bool = True,
        image_format: str = "jpg",
        jpeg_quality: int = 95,
        fps: float = 30.0,
        camera_roles: Optional[Dict[str, str]] = None,
        camera_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.save_dir = os.path.join(save_dir, task_directory)
        self.instruction = language_instruction
        self.save_depth = bool(save_depth)
        self.image_format = image_format.lower().lstrip(".")
        if self.image_format not in ("jpg", "jpeg", "png"):
            raise ValueError(f"image_format must be jpg or png, got {image_format!r}")
        if self.image_format == "jpeg":
            self.image_format = "jpg"
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.png_compress_level = int(np.clip(png_compress_level, 0, 9))
        self.fps = float(fps)
        self.camera_roles = dict(DEFAULT_CAMERA_ROLES)
        if camera_roles:
            self.camera_roles.update(camera_roles)
        self.camera_meta: Dict[str, Dict[str, Any]] = {}
        if camera_meta:
            self.set_camera_meta(camera_meta)

        # Writer pool. cv2.imencode releases the GIL, so threads parallelise.
        if saver_max_workers is None:
            self.max_workers = max(2, min(8, (os.cpu_count() or 2) // 2))
        else:
            self.max_workers = max(1, int(saver_max_workers))
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="frame_writer"
        )

        self.traj_count = 1  # index the NEXT episode will get
        self.buffer: List[Dict[str, Any]] = []  # records of the episode being recorded
        self._episodes: Dict[int, _Episode] = {}
        self._current: Optional[_Episode] = None
        self._lock = threading.Lock()

        if os.path.exists(self.save_dir):
            self.traj_count = self._resume_existing_dir()

        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------- resume --

    @staticmethod
    def scan_episode_dirs(save_dir: str):
        """Return (complete_indices, incomplete_dirs) for NNNNNN/ dirs under save_dir.

        Complete = has a per-frame json AND meta.json (i.e. was saved with 'a').
        Incomplete = frames from a run that crashed or was discarded mid-episode.
        """
        complete, incomplete = [], []
        for name in sorted(os.listdir(save_dir)):
            d = os.path.join(save_dir, name)
            if not (os.path.isdir(d) and name.isdigit()):
                continue
            has_json = any(f.endswith(".json") and f != "meta.json" for f in os.listdir(d))
            if has_json and os.path.exists(os.path.join(d, "meta.json")):
                complete.append(int(name))
            else:
                incomplete.append(d)
        return complete, incomplete

    def _resume_existing_dir(self) -> int:
        """Decide the next episode index for an existing directory.

        The old prompt asked "remove? (y/n)" (where 'y' deleted real episodes) and
        then for the next index by hand. The index is inferred instead: leftovers
        from crashed runs are removed (they hold no per-frame json, so they are
        not data), and Enter appends after the last complete episode. Deleting
        everything requires typing the word.
        """
        complete, incomplete = self.scan_episode_dirs(self.save_dir)
        for d in incomplete:
            shutil.rmtree(d, ignore_errors=True)
            logger.warning(f"Removed incomplete episode dir (no json/meta -- crashed or discarded run): {d}")
        next_idx = (max(complete) + 1) if complete else 1
        if not complete:
            logger.info(f"{self.save_dir} exists but holds no complete episodes; starting at {next_idx:06d}.")
            return next_idx
        print(
            f"\n{self.save_dir} already holds {len(complete)} complete episode(s), last = {max(complete):06d}."
        )
        while True:
            ans = input(
                f"  [Enter] append as {next_idx:06d}   |   type a number to start at that index   |   "
                f"type 'delete' to remove ALL {len(complete)} episodes: "
            ).strip()
            if ans == "":
                logger.info(f"Appending to {self.save_dir} starting at episode {next_idx:06d}.")
                return next_idx
            if ans.isdigit():
                idx = int(ans)
                if idx in complete:
                    print(f"  {idx:06d} already exists and would be overwritten. Pick another, or Enter to append.")
                    continue
                logger.info(f"Appending to {self.save_dir} starting at episode {idx:06d}.")
                return idx
            if ans.lower() == "delete":
                confirm = input(f"  Really delete {len(complete)} episodes in {self.save_dir}? type 'yes' to confirm: ").strip()
                if confirm.lower() == "yes":
                    shutil.rmtree(self.save_dir)
                    logger.info(f"Removed existing directory: {self.save_dir}.")
                    return 1
                print("  Not deleted.")
                continue
            print("  Enter, a number, or 'delete'.")

    # ------------------------------------------------------------------ meta --

    def set_camera_meta(self, meta: Dict[str, Dict[str, Any]]) -> None:
        """Accept ``RobotEnv.get_camera_meta()`` output, keyed ``<cam>_camera`` or ``<cam>``."""
        for name, entry in (meta or {}).items():
            cam = name[: -len("_camera")] if name.endswith("_camera") else name
            self.camera_meta[cam] = dict(entry or {})

    # --------------------------------------------------------------- episodes --

    def _new_episode(self) -> _Episode:
        ep = _Episode(self.traj_count, self.save_dir)
        self.traj_count += 1
        self._episodes[ep.index] = ep
        self._current = ep
        logger.info(f"Recording episode {ep.index} -> {ep.dir}")
        return ep

    def reset_buffer(self) -> None:
        """Start a fresh episode. An unsaved, un-queued current episode is discarded."""
        old_size = len(self.buffer)
        with self._lock:
            ep = self._current
            self._current = None
            self.buffer = []
        if ep is not None and not ep.save_pending and not ep.saved and not ep.discarded:
            self._discard(ep)
        logger.info(f"Reset buffer: {old_size} observations cleared.")

    def _discard(self, ep: _Episode) -> None:
        # Indices are NEVER reused. Cleanup is asynchronous (an 8 GB episode dir is
        # slow to rmtree and must not stall the loop between takes), so it can fire
        # after the next episode has already started recording. When the index was
        # reused, the next episode wrote to the SAME directory path and this stale
        # rmtree deleted a just-saved episode -- the silent data loss seen on
        # 2026-09-08. A monotonic index gives every episode its own directory, so a
        # delayed cleanup can only ever remove the one it was created for. It also
        # waits for that episode's own writes to drain first (they land in ep.dir),
        # so no late write can recreate the directory after it is removed.
        ep.discarded = True

        def _rm() -> None:
            ep.wait_writes()
            shutil.rmtree(ep.dir, ignore_errors=True)
            self._episodes.pop(ep.index, None)
            logger.info(f"Discarded episode {ep.index} ({len(ep.records)} frames); removed {ep.dir}")

        self._pool.submit(_rm)

    def mark_pending_save(self, buffer: List[Dict[str, Any]]) -> None:
        """Called synchronously by EpisodeSaverThread.save_episode before queueing,
        so the next reset_buffer() does not mistake the queued episode for a discard."""
        ep = self._episode_for(buffer)
        if ep is not None:
            ep.save_pending = True

    def _episode_for(self, buffer: List[Dict[str, Any]]) -> Optional[_Episode]:
        if buffer:
            return self._episodes.get(int(buffer[0].get("episode_index", -1)))
        return self._current

    # ------------------------------------------------------------------ frames --

    def add_observation(self, obs: Dict[str, Any]) -> None:
        with self._lock:
            ep = self._current
            if ep is None or ep.save_pending or ep.saved or ep.discarded:
                ep = self._new_episode()
        frame_idx = len(ep.records)

        record: Dict[str, Any] = {
            "episode_index": ep.index,
            "frame_index": frame_idx,
            "timestamp": time.time(),
            "instruction": self.instruction,
            "joint": _to_list(obs["joint_positions"]),
            "next_joint": _to_list(obs["next_joint"]),
            "camera_timestamps": {},
            "camera_stale": list(obs.get("camera_stale") or []),
            "images": {},
        }

        cams = [k[: -len("_camera_rgb")] for k in obs if k.endswith("_camera_rgb")]
        for cam in sorted(cams):
            rgb = obs[f"{cam}_camera_rgb"]
            rgb_path = os.path.join(ep.dir, f"{cam}_rgb", f"{frame_idx:06d}.{self.image_format}")
            record["images"][f"{cam}_rgb"] = rgb_path
            self._submit(ep, self._write_rgb, np.ascontiguousarray(rgb), rgb_path)

            depth = obs.get(f"{cam}_camera_depth") if self.save_depth else None
            if depth is not None:
                depth_path = os.path.join(ep.dir, f"{cam}_depth", f"{frame_idx:06d}.png")
                record["images"][f"{cam}_depth"] = depth_path
                self._submit(ep, self._write_depth, np.ascontiguousarray(depth), depth_path)

            ts = obs.get(f"{cam}_camera_timestamp")
            if ts is not None:
                record["camera_timestamps"][cam] = float(ts)

        ep.records.append(record)
        self.buffer.append(record)

        if ep.pending > _BACKLOG_WARN and not ep.warned_backlog:
            ep.warned_backlog = True
            logger.warning(
                f"Episode {ep.index}: {ep.pending} image writes queued -- disk is not keeping up "
                f"with {self.fps:.0f} Hz. Consider image_format=jpg or fewer/faster cameras."
            )

    def _submit(self, ep: _Episode, fn, *args) -> None:
        ep.submitted()
        fut = self._pool.submit(fn, *args)

        def _done(f, ep=ep):
            exc = f.exception()
            ep.finished(None if exc is None else repr(exc))

        fut.add_done_callback(_done)

    def _write_rgb(self, rgb: np.ndarray, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self.image_format == "png":
            ok = cv2.imwrite(path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, self.png_compress_level])
        else:
            ok = cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise IOError(f"cv2.imwrite failed: {path}")

    def _write_depth(self, depth: np.ndarray, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d = np.asarray(depth)
        if d.ndim == 3:
            d = d[:, :, 0]
        if d.dtype != np.uint16:
            d = np.clip(np.rint(d), 0, 65535).astype(np.uint16)
        ok = cv2.imwrite(path, d, [cv2.IMWRITE_PNG_COMPRESSION, self.png_compress_level])
        if not ok:
            raise IOError(f"cv2.imwrite failed: {path}")

    # kept for callers that used the old API
    def save_image(self, image: np.ndarray, path: str) -> None:
        self._write_rgb(np.asarray(image), path)

    # ---------------------------------------------------------------- finalise --

    def save_episode_json(self, buffer: List[Dict[str, Any]], pickle_only: bool = False) -> None:
        if not buffer:
            logger.warning("Empty buffer, no observations to save.")
            return
        ep = self._episode_for(buffer)
        if ep is None:
            logger.error("save_episode_json: no episode matches the buffer; nothing written")
            return
        if ep.saved:
            return
        ep.save_pending = True
        logger.info(f"Saving episode {ep.index} to {ep.dir} with {len(ep.records)} observations.")
        ep.wait_writes()
        if ep.write_errors:
            logger.error(
                f"Episode {ep.index}: {len(ep.write_errors)} image writes FAILED, e.g. {ep.write_errors[0]}"
            )

        records = ep.records
        json_data = []
        for r in records:
            j, nj = r["joint"], r["next_joint"]
            row = {
                "language_instruction": r["instruction"],
                "frame_index": r["frame_index"],
                "timestamp": r["timestamp"],
                "left_joint": str(j[:7]),
                "right_joint": str(j[7:]),
                "next_left_joint": str(nj[:7]),
                "next_right_joint": str(nj[7:]),
                "camera_timestamps": r["camera_timestamps"],
                "camera_stale": r.get("camera_stale", []),
            }
            for key, path in r["images"].items():
                row[f"image_{key}"] = path
            json_data.append(row)

        json_path = os.path.join(ep.dir, f"{ep.index:06d}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

        meta = self._episode_meta(ep, records)
        with open(os.path.join(ep.dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        ep.saved = True
        logger.info(
            f"Complete!!!! Saved episode {ep.index} to {ep.dir} with {len(records)} observations "
            f"({meta['duration_s']:.1f}s, {meta['effective_fps']:.1f} fps effective)."
        )

    def _episode_meta(self, ep: _Episode, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        cams = sorted({k[: -len("_rgb")] for r in records for k in r["images"] if k.endswith("_rgb")})
        t0, t1 = records[0]["timestamp"], records[-1]["timestamp"]
        duration = max(t1 - t0, 0.0)
        cameras: Dict[str, Any] = {}
        for cam in cams:
            entry = dict(self.camera_meta.get(cam, {}))
            entry["role"] = self.camera_roles.get(cam, cam)
            entry["rgb_dir"] = f"{cam}_rgb"
            entry["rgb_format"] = self.image_format
            if any(f"{cam}_depth" in r["images"] for r in records):
                entry["depth_dir"] = f"{cam}_depth"
                entry["depth_format"] = "png16_native_units"
            intr = entry.get("intrinsics") or {}
            entry.setdefault("width", intr.get("width"))
            entry.setdefault("height", intr.get("height"))
            cameras[cam] = entry
        return {
            "format": "gello_data_saver/2",
            "episode_index": ep.index,
            "instruction": self.instruction,
            "fps": self.fps,
            "num_frames": len(records),
            "started_at": ep.started_at,
            "saved_at": time.time(),
            "duration_s": duration,
            "effective_fps": (len(records) - 1) / duration if duration > 0 and len(records) > 1 else 0.0,
            "cameras": cameras,
            "joint_layout": {"left": JOINT_NAMES, "right": JOINT_NAMES},
            "joint_units": "rad (arm joints), gripper in i2rt command space [0, 1], 1 = open",
            "action_semantics": "next_joint = follower joint positions after applying the leader command at this step",
            "image_write_errors": ep.write_errors,
            "frames_with_stale_camera": sum(1 for r in records if r.get("camera_stale")),
        }

    def finalize_on_exit(self) -> None:
        """Called from the launcher's cleanup on crash / Ctrl-C.

        - An episode the operator already pressed SAVE on but the background saver
          has not finalised yet is written now (synchronously), so it is not lost.
        - The in-progress episode (recording, never saved) is discarded: its frames
          are removed rather than left as an incomplete directory.
        """
        for ep in list(self._episodes.values()):
            if ep.save_pending and not ep.saved and not ep.discarded and ep.records:
                try:
                    logger.warning(f"Exit with episode {ep.index} still queued for saving; finalising it now.")
                    self.save_episode_json(ep.records)
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"could not finalise episode {ep.index}: {exc}")
        with self._lock:
            ep = self._current
            self._current = None
            self.buffer = []
        if ep is not None and not ep.save_pending and not ep.saved and not ep.discarded:
            logger.warning(f"Exit with episode {ep.index} in progress ({len(ep.records)} frames, never saved); discarding it.")
            ep.discarded = True
            ep.wait_writes(timeout=10.0)
            shutil.rmtree(ep.dir, ignore_errors=True)
            self._episodes.pop(ep.index, None)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
