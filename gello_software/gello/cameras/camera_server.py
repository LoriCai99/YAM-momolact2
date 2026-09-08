"""ZMQ-based camera server.

Hosts the RealSense cameras in a long-lived process so eval clients can pull
the latest frames on demand without paying for pipeline startup, fighting the
policy loop for camera I/O, or coupling robot-control timing to camera I/O.

Sockets
-------
REP  ``tcp://127.0.0.1:5555``  (default)
    Pull semantics. Client sends a pickled request dict; server replies with a
    pickled response dict. Used by the policy for on-demand obs.

PUB  ``tcp://127.0.0.1:5556``  (default, optional)
    Push semantics. Server publishes the latest obs every ``pub_period_sec``.
    Intended for the cv2 live viewer so it can render at camera rate without
    burning policy-side requests.

Request protocol
----------------
    {"cmd": "obs"}   ->  {"ok": True, "frames": {cam_name: np.ndarray (H,W,3) uint8 RGB},
                          "timestamps": {cam_name: float}}
    {"cmd": "ping"}  ->  {"ok": True, "pong": True}

Errors come back as ``{"ok": False, "error": str}``. The server keeps running
across any single bad request.

CLI
---
    python -m gello.cameras.camera_server --config gello_software/configs/yam_left.yaml
"""
from __future__ import annotations

import argparse
import logging
import json
import pickle
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import zmq
from omegaconf import OmegaConf

from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids


logger = logging.getLogger("camera_server")


DEFAULT_REP_ENDPOINT = "tcp://127.0.0.1:5555"
DEFAULT_PUB_ENDPOINT = "tcp://127.0.0.1:5556"
DEFAULT_PUB_PERIOD_SEC = 1.0 / 30.0
DEFAULT_HEARTBEAT_SEC = 10.0


class CameraServer:
    """Owns RealSense cameras and serves their latest frames over ZMQ."""

    def __init__(
        self,
        cameras: Dict[str, RealSenseCamera],
        rep_endpoint: str = DEFAULT_REP_ENDPOINT,
        pub_endpoint: Optional[str] = None,
        pub_period_sec: float = DEFAULT_PUB_PERIOD_SEC,
        heartbeat_sec: float = DEFAULT_HEARTBEAT_SEC,
        pub_format: str = "pickle",
        pub_on_new_frame: bool = False,
        pub_fallback_sec: float = 0.05,
    ) -> None:
        # Event-driven publishing: send once every camera has captured a new frame
        # (phase-locked to the cameras, ~camera rate), or after pub_fallback_sec if
        # one camera stalls. Publishing on a faster fixed timer starves the capture
        # threads (rs.align holds the GIL) and INCREASES duplicated frames.
        self.pub_on_new_frame = pub_on_new_frame
        self.pub_fallback_sec = float(pub_fallback_sec)
        self.cameras = cameras
        # "pickle": legacy PUB payload (cv2 viewer / CameraSubscriber). "multipart":
        # zero-copy JSON header + raw buffers incl. depth (CameraStreamClient).
        self.pub_format = pub_format
        self.rep_endpoint = rep_endpoint
        self.pub_endpoint = pub_endpoint
        self.pub_period_sec = float(pub_period_sec)
        self.heartbeat_sec = float(heartbeat_sec)

        self._ctx = zmq.Context.instance()
        self._rep: Optional[zmq.Socket] = None
        self._pub: Optional[zmq.Socket] = None

        self._stop_event = threading.Event()
        self._pub_thread: Optional[threading.Thread] = None

        self._req_total = 0
        self._req_window = 0
        self._last_heartbeat = time.time()

    # ------------------------------------------------------------------
    # Frame sourcing
    # ------------------------------------------------------------------

    def _snapshot(self) -> Dict[str, Any]:
        """Snapshot the latest colour (RGB uint8) AND depth (uint16, native units) frames.

        Depth is aligned to colour by the driver (``rs.align`` runs in THIS process,
        which is the whole point: it holds the GIL ~10-20 ms per frame and must not
        share a process with the arms' 250 Hz control loops). ``frames`` keeps its
        original meaning for older clients; ``depth`` is additive.
        """
        frames: Dict[str, Any] = {}
        depth: Dict[str, Any] = {}
        timestamps: Dict[str, float] = {}
        for name, cam in self.cameras.items():
            image, d = cam.read()
            frames[name] = image
            if d is not None:
                depth[name] = d[:, :, 0] if getattr(d, "ndim", 2) == 3 else d
            # Surface the capture timestamp so the client can detect staleness.
            ts = getattr(cam, "last_frame_timestamp", None)
            if ts is None:
                ts = getattr(cam, "_latest_frame_timestamp", None)
            timestamps[name] = float(ts or 0.0)
        return {"ok": True, "frames": frames, "depth": depth, "timestamps": timestamps}

    def _snapshot_multipart(self) -> list:
        """``obs2`` payload: a JSON header frame followed by one raw buffer per array.

        Zero-copy on both ends (``send_multipart(copy=False)`` here,
        ``np.frombuffer`` on the client) -- pickle copied every 3.5 MB observation
        twice and cost the collection loop ~20 ms per tick.
        """
        snap = self._snapshot()
        header: Dict[str, Any] = {"ok": True, "v": 2, "cams": [], "timestamps": snap["timestamps"]}
        parts: list = []
        for name, rgb in snap["frames"].items():
            rgb = np.ascontiguousarray(rgb)
            entry: Dict[str, Any] = {"name": name, "rgb": {"shape": list(rgb.shape), "dtype": str(rgb.dtype)}}
            parts.append(rgb)
            d = snap["depth"].get(name)
            if d is not None:
                d = np.ascontiguousarray(d)
                entry["depth"] = {"shape": list(d.shape), "dtype": str(d.dtype)}
                parts.append(d)
            header["cams"].append(entry)
        return [json.dumps(header).encode("utf-8")] + parts

    def _meta(self) -> Dict[str, Any]:
        """Static per-camera metadata: serial, colour intrinsics, depth scale."""
        meta: Dict[str, Any] = {}
        for name, cam in self.cameras.items():
            entry: Dict[str, Any] = {"device_id": getattr(cam, "device_id", None)}
            for attr, key in (("get_intrinsics", "intrinsics"), ("get_depth_scale", "depth_scale_m_per_unit")):
                fn = getattr(cam, attr, None)
                try:
                    entry[key] = fn() if callable(fn) else None
                except Exception as exc:  # noqa: BLE001 - report, don't die
                    entry[key] = None
                    entry.setdefault("errors", []).append(f"{attr}: {exc}")
            meta[name] = entry
        return {"ok": True, "meta": meta}

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def _handle_request(self) -> None:
        assert self._rep is not None
        raw = self._rep.recv()
        try:
            req = pickle.loads(raw)
            cmd = (req or {}).get("cmd", "obs")
        except Exception as exc:  # noqa: BLE001 — surface to client, stay alive
            self._rep.send(pickle.dumps({"ok": False, "error": f"bad request: {exc!r}"}))
            return

        try:
            if cmd == "obs2":
                self._rep.send_multipart(self._snapshot_multipart(), copy=False)
                self._req_total += 1
                self._req_window += 1
                return
            if cmd == "obs":
                resp = self._snapshot()
            elif cmd == "meta":
                resp = self._meta()
            elif cmd == "ping":
                resp = {"ok": True, "pong": True}
            else:
                resp = {"ok": False, "error": f"unknown cmd: {cmd!r}"}
        except Exception as exc:  # noqa: BLE001 — keep server alive
            logger.exception("Request failed (cmd=%r)", cmd)
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        self._rep.send(pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL), copy=False)
        self._req_total += 1
        self._req_window += 1

    def _all_cameras_advanced(self, last_counts: Dict[str, int]) -> bool:
        for name, cam in self.cameras.items():
            if getattr(cam, "frame_count", None) is None:
                return True  # camera does not expose a counter: fall back to timer behaviour
            if cam.frame_count <= last_counts.get(name, -1):
                return False
        return True

    def _pub_loop(self) -> None:
        assert self._pub is not None
        next_tick = time.time()
        last_counts: Dict[str, int] = {}
        last_pub = 0.0
        while not self._stop_event.is_set():
            now = time.time()
            if self.pub_on_new_frame:
                if not (self._all_cameras_advanced(last_counts) or now - last_pub >= self.pub_fallback_sec):
                    time.sleep(0.0005)
                    continue
                last_counts = {n: getattr(c, "frame_count", 0) or 0 for n, c in self.cameras.items()}
                last_pub = now
            else:
                if now < next_tick:
                    # Tiny sleep granularity so shutdown is snappy.
                    time.sleep(min(0.01, next_tick - now))
                    continue
                next_tick = now + self.pub_period_sec
            try:
                if self.pub_format == "multipart":
                    self._pub.send_multipart(self._snapshot_multipart(), copy=False)
                else:
                    resp = self._snapshot()
                    self._pub.send(pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL), copy=False)
            except Exception as exc:  # noqa: BLE001 — pub is best-effort
                logger.warning("PUB tick failed: %s", exc)

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        elapsed = now - self._last_heartbeat
        if elapsed < self.heartbeat_sec:
            return
        hz = self._req_window / elapsed if elapsed > 0 else 0.0
        logger.info(
            "alive: total_requests=%d window=%d (%.1f req/s) cameras=%d",
            self._req_total, self._req_window, hz, len(self.cameras),
        )
        self._req_window = 0
        self._last_heartbeat = now

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._rep = self._ctx.socket(zmq.REP)
        self._rep.bind(self.rep_endpoint)
        logger.info("REP bound on %s", self.rep_endpoint)

        if self.pub_endpoint:
            self._pub = self._ctx.socket(zmq.PUB)
            self._pub.bind(self.pub_endpoint)
            logger.info(
                "PUB bound on %s (period=%.3fs)", self.pub_endpoint, self.pub_period_sec,
            )
            self._pub_thread = threading.Thread(
                target=self._pub_loop, name="camera_server_pub", daemon=True,
            )
            self._pub_thread.start()

        poller = zmq.Poller()
        poller.register(self._rep, zmq.POLLIN)
        try:
            while not self._stop_event.is_set():
                # 100 ms tick keeps heartbeats responsive and shutdown snappy.
                socks = dict(poller.poll(timeout=100))
                if self._rep in socks:
                    self._handle_request()
                self._maybe_heartbeat()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        if self._pub_thread is not None:
            self._pub_thread.join(timeout=2.0)
        for sock in (self._rep, self._pub):
            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    pass
        for cam in self.cameras.values():
            try:
                close = getattr(cam, "close", None)
                if callable(close):
                    close()
                else:
                    cam._stop_event.set()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
        logger.info("Camera server stopped.")


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def _build_cameras_from_config(cfg_path: Path) -> Dict[str, RealSenseCamera]:
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    camera_cfg = cfg["sensors"]["cameras"]
    logger.info("Discovering RealSense devices...")
    ids = get_device_ids()
    logger.info("Found %d RealSense devices: %s", len(ids), ids)
    cameras: Dict[str, RealSenseCamera] = {}
    for name, spec in camera_cfg.items():
        device_id = spec["device_id"]
        logger.info("Opening camera %s (device_id=%s)", name, device_id)
        cameras[name] = RealSenseCamera(device_id)
    return cameras


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="YAM camera server (ZMQ).")
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to a yam_*.yaml whose sensors.cameras block lists the devices.",
    )
    parser.add_argument("--rep-endpoint", default=DEFAULT_REP_ENDPOINT)
    parser.add_argument(
        "--pub-endpoint", default=DEFAULT_PUB_ENDPOINT,
        help="ZMQ PUB endpoint. Pass empty string to disable the PUB stream.",
    )
    parser.add_argument("--pub-period-sec", type=float, default=DEFAULT_PUB_PERIOD_SEC)
    parser.add_argument("--pub-on-new-frame", action="store_true",
                        help="Publish once per new frame set (phase-locked to the cameras) instead of on a timer.")
    parser.add_argument("--pub-format", choices=("pickle", "multipart"), default="pickle",
                        help="PUB payload: pickle (legacy viewer) or multipart zero-copy incl. depth.")
    parser.add_argument("--heartbeat-sec", type=float, default=DEFAULT_HEARTBEAT_SEC)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--exit-with-parent", action="store_true",
                        help="Shut down when the launching process exits (child-process mode).")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cameras = _build_cameras_from_config(args.config)
    server = CameraServer(
        cameras=cameras,
        rep_endpoint=args.rep_endpoint,
        pub_endpoint=(args.pub_endpoint or None),
        pub_period_sec=args.pub_period_sec,
        heartbeat_sec=args.heartbeat_sec,
        pub_format=args.pub_format,
        pub_on_new_frame=args.pub_on_new_frame,
    )
    if args.exit_with_parent:
        import os

        parent = os.getppid()

        def _watch_parent() -> None:
            while os.getppid() == parent:
                time.sleep(0.5)
            logger.info("Parent process %d exited; shutting down.", parent)
            server.shutdown()

        threading.Thread(target=_watch_parent, name="parent_watch", daemon=True).start()

    def _handle(signum, _frame):
        logger.info("Signal %d received; shutting down.", signum)
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
