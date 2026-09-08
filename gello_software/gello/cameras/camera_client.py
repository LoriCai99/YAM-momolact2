"""ZMQ client for the camera server.

Used by eval launchers to pull the latest 3-camera observation without
holding RealSense devices in-process. See ``camera_server.py`` for the wire
protocol.
"""
from __future__ import annotations

import logging
import json
import pickle
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import zmq


logger = logging.getLogger(__name__)


class CameraClientError(RuntimeError):
    """Raised when the camera server is unreachable, slow, or returns an error."""


class CameraClient:
    """REQ-side wrapper. ``get_obs()`` returns ``{cam_name: np.ndarray (H,W,3) uint8 RGB}``.

    On timeout the underlying REQ socket is closed and recreated — REQ sockets
    become unusable after a recv timeout without a matching reply.
    """

    def __init__(
        self,
        endpoint: str,
        request_timeout_ms: int = 500,
        max_frame_age_sec: Optional[float] = 0.5,
    ) -> None:
        self.endpoint = endpoint
        self.request_timeout_ms = int(request_timeout_ms)
        self.max_frame_age_sec = max_frame_age_sec
        self._ctx = zmq.Context.instance()
        self._sock: Optional[zmq.Socket] = None
        self._legacy_obs = False
        self._connect()

    def _connect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, self.request_timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self.request_timeout_ms)
        sock.connect(self.endpoint)
        self._sock = sock

    def _request(self, cmd: str) -> Dict[str, Any]:
        assert self._sock is not None
        try:
            self._sock.send(pickle.dumps({"cmd": cmd}, protocol=pickle.HIGHEST_PROTOCOL))
            raw = self._sock.recv()
        except zmq.Again as exc:
            # REQ socket is now in a bad state; reset before raising.
            self._connect()
            raise CameraClientError(
                f"Camera server timeout ({self.request_timeout_ms} ms) on cmd={cmd!r} "
                f"at {self.endpoint}. Is the server running?"
            ) from exc
        try:
            resp = pickle.loads(raw)
        except Exception as exc:  # noqa: BLE001 — unparseable reply
            self._connect()
            raise CameraClientError(f"Unparseable reply from camera server: {exc!r}") from exc
        if not resp.get("ok"):
            raise CameraClientError(f"Server error: {resp.get('error')}")
        return resp

    def ping(self) -> bool:
        return bool(self._request("ping").get("pong"))

    def _check_fresh(self, resp: Dict[str, Any]) -> None:
        if self.max_frame_age_sec is None:
            return
        now = time.time()
        for name, ts in (resp.get("timestamps") or {}).items():
            if ts and (now - ts) > self.max_frame_age_sec:
                raise CameraClientError(
                    f"Stale frame from {name}: {now - ts:.3f}s old "
                    f"(>{self.max_frame_age_sec:.3f}s)."
                )

    def get_obs(self) -> Dict[str, np.ndarray]:
        """Return ``{cam_name: np.ndarray (H,W,3) uint8 RGB}`` with the latest frames."""
        resp = self._request("obs")
        self._check_fresh(resp)
        return resp["frames"]

    def get_obs_full(self) -> Dict[str, Any]:
        """Return ``{"frames": {cam: RGB uint8 (H,W,3)}, "depth": {cam: uint16 (H,W)},
        "timestamps": {cam: float}}``. Depth is in native sensor units; see ``get_meta``.

        Uses the zero-copy ``obs2`` multipart protocol; arrays are read-only views
        over the received buffers (copy before writing into them). Falls back to
        the pickle ``obs`` reply for servers that predate ``obs2``.
        """
        if self._legacy_obs:
            return self._get_obs_full_pickle()
        assert self._sock is not None
        try:
            self._sock.send(pickle.dumps({"cmd": "obs2"}, protocol=pickle.HIGHEST_PROTOCOL))
            parts = self._sock.recv_multipart(copy=False)
        except zmq.Again as exc:
            self._connect()
            raise CameraClientError(
                f"Camera server timeout ({self.request_timeout_ms} ms) on cmd='obs2' at {self.endpoint}. Is the server running?"
            ) from exc
        first = parts[0].buffer
        try:
            header = json.loads(bytes(first))
        except Exception:  # not JSON -> a pickle reply (error, or an older server)
            try:
                resp = pickle.loads(bytes(first))
            except Exception as exc:  # noqa: BLE001
                self._connect()
                raise CameraClientError(f"Unparseable reply from camera server: {exc!r}") from exc
            if resp.get("ok"):
                raise CameraClientError("unexpected pickle reply to obs2")
            if "unknown cmd" in str(resp.get("error", "")):
                logger.warning("camera server predates obs2; falling back to pickle obs")
                self._legacy_obs = True
                return self._get_obs_full_pickle()
            raise CameraClientError(f"Server error: {resp.get('error')}")
        frames: Dict[str, np.ndarray] = {}
        depth: Dict[str, np.ndarray] = {}
        i = 1
        for cam in header["cams"]:
            r = cam["rgb"]
            frames[cam["name"]] = np.frombuffer(parts[i].buffer, dtype=r["dtype"]).reshape(r["shape"])
            i += 1
            if "depth" in cam:
                d = cam["depth"]
                depth[cam["name"]] = np.frombuffer(parts[i].buffer, dtype=d["dtype"]).reshape(d["shape"])
                i += 1
        self._check_fresh(header)
        return {"frames": frames, "depth": depth, "timestamps": header.get("timestamps") or {}}

    def _get_obs_full_pickle(self) -> Dict[str, Any]:
        resp = self._request("obs")
        self._check_fresh(resp)
        return {"frames": resp["frames"], "depth": resp.get("depth") or {}, "timestamps": resp.get("timestamps") or {}}

    def get_meta(self) -> Dict[str, Dict[str, Any]]:
        """Per-camera ``{device_id, intrinsics, depth_scale_m_per_unit}`` from the server."""
        return self._request("meta")["meta"]

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
            self._sock = None


def _parse_multipart(parts) -> Optional[Dict[str, Any]]:
    """Decode an ``obs2``/multipart payload into {frames, depth, timestamps}; None if not one."""
    try:
        header = json.loads(bytes(parts[0].buffer if hasattr(parts[0], "buffer") else parts[0]))
    except Exception:  # noqa: BLE001
        return None
    if not header.get("ok") or "cams" not in header:
        return None
    frames: Dict[str, np.ndarray] = {}
    depth: Dict[str, np.ndarray] = {}
    i = 1
    for cam in header["cams"]:
        r = cam["rgb"]
        buf = parts[i].buffer if hasattr(parts[i], "buffer") else parts[i]
        frames[cam["name"]] = np.frombuffer(buf, dtype=r["dtype"]).reshape(r["shape"])
        i += 1
        if "depth" in cam:
            d = cam["depth"]
            buf = parts[i].buffer if hasattr(parts[i], "buffer") else parts[i]
            depth[cam["name"]] = np.frombuffer(buf, dtype=d["dtype"]).reshape(d["shape"])
            i += 1
    return {"frames": frames, "depth": depth, "timestamps": header.get("timestamps") or {}}


class CameraStreamClient:
    """Latest-frame client for the control loop: never blocks on the server.

    A daemon thread subscribes to the server's multipart PUB stream
    (``--pub-format multipart``) and keeps the newest observation; ``get_obs_full``
    returns it in microseconds. The REQ socket is used only for ``ping``/``meta``
    (and as a fallback for ``get_obs_full`` if the PUB stream is not multipart).

    Why: with REQ/REP the loop waited on the server's GIL while its ``rs.align``
    ran -- 11 ms mean, 29 ms p95, 170 ms worst -- every one a hitch in the arms.
    """

    def __init__(self, rep_endpoint: str, pub_endpoint: str, request_timeout_ms: int = 500,
                 max_frame_age_sec: Optional[float] = 0.5, recv_timeout_ms: int = 200) -> None:
        self.req = CameraClient(rep_endpoint, request_timeout_ms=request_timeout_ms, max_frame_age_sec=max_frame_age_sec)
        self.max_frame_age_sec = max_frame_age_sec
        self.pub_endpoint = pub_endpoint
        self._ctx = zmq.Context.instance()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.LINGER, 0)
        self._sub.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))
        self._sub.setsockopt(zmq.RCVHWM, 4)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.connect(pub_endpoint)
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._latest_rx = 0.0
        self._pub_ok: Optional[bool] = None  # None = undecided, False = not multipart -> REQ fallback
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="camera_stream_rx", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                parts = self._sub.recv_multipart(copy=False)
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            # drain to the newest message so we never fall behind the publisher
            while True:
                try:
                    parts = self._sub.recv_multipart(copy=False, flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
            obs = _parse_multipart(parts)
            if obs is None:
                if self._pub_ok is None:
                    logger.warning("camera PUB stream is not multipart; CameraStreamClient falls back to REQ obs2")
                    self._pub_ok = False
                continue
            with self._lock:
                self._latest, self._latest_rx, self._pub_ok = obs, time.time(), True

    def ping(self) -> bool:
        return self.req.ping()

    def get_meta(self) -> Dict[str, Dict[str, Any]]:
        return self.req.get_meta()

    def get_obs_full(self, first_timeout_sec: float = 3.0) -> Dict[str, Any]:
        if self._pub_ok is False:
            return self.req.get_obs_full()
        deadline = time.time() + first_timeout_sec
        while True:
            with self._lock:
                obs, rx = self._latest, self._latest_rx
            if obs is not None:
                break
            if self._pub_ok is False:
                return self.req.get_obs_full()
            if time.time() > deadline:
                raise CameraClientError(f"no frames received on {self.pub_endpoint} within {first_timeout_sec:.0f}s")
            time.sleep(0.005)
        if self.max_frame_age_sec is not None:
            age = time.time() - rx
            if age > self.max_frame_age_sec:
                raise CameraClientError(f"camera stream stale: last frame set received {age:.3f}s ago (>{self.max_frame_age_sec}s)")
        return obs

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self._sub.close(linger=0)
        except Exception:  # noqa: BLE001
            pass
        self.req.close()


class CameraSubscriber:
    """Optional PUB/SUB consumer for the live viewer.

    The eval inner loop should use ``CameraClient`` (REQ/REP). This subscriber
    exists so a cv2 viewer can render at camera rate without competing for the
    REP socket with the policy.
    """

    def __init__(self, endpoint: str, recv_timeout_ms: int = 100) -> None:
        self.endpoint = endpoint
        self.recv_timeout_ms = int(recv_timeout_ms)
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.connect(endpoint)

    def try_recv(self) -> Optional[Dict[str, np.ndarray]]:
        """Return the most recent frame dict if one is available, else None."""
        latest: Optional[bytes] = None
        # Drain the queue so we hand the consumer the freshest frame.
        while True:
            try:
                latest = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if latest is None:
            return None
        try:
            resp = pickle.loads(latest)
        except Exception as exc:  # noqa: BLE001 — drop malformed publish
            logger.warning("Dropped malformed PUB payload: %s", exc)
            return None
        if not resp.get("ok"):
            return None
        return resp.get("frames")

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Live viewer for the YAM camera server. "
                    "Defaults to the PUB stream so it doesn't fight the policy for REP."
    )
    parser.add_argument(
        "--mode", choices=("sub", "req"), default="sub",
        help="sub: subscribe to PUB stream (default). req: poll via REQ/REP.",
    )
    parser.add_argument("--rep-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--pub-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--req-hz", type=float, default=30.0,
                        help="Polling rate when --mode=req.")
    parser.add_argument("--window", default="camera_client",
                        help="cv2 window title.")
    args = parser.parse_args()

    import cv2  # imported lazily so library users don't pay for it

    def _fetch_sub():
        return sub.try_recv()

    def _fetch_req():
        try:
            return req.get_obs()
        except CameraClientError as exc:
            logger.warning("REQ fetch failed: %s", exc)
            return None

    if args.mode == "sub":
        sub = CameraSubscriber(args.pub_endpoint)
        fetch = _fetch_sub
        period = 0.0  # PUB drives the rate; just spin with a short waitKey
    else:
        req = CameraClient(args.rep_endpoint, request_timeout_ms=1000, max_frame_age_sec=None)
        fetch = _fetch_req
        period = 1.0 / max(args.req_hz, 1e-3)

    last_fps_t = time.time()
    fps_frames = 0
    fps_disp = 0.0
    last_loop_t = 0.0

    print(f"[camera_client] mode={args.mode} — press 'q' in the window to quit.", flush=True)
    try:
        while True:
            now = time.time()
            if period and (now - last_loop_t) < period:
                time.sleep(max(0.0, period - (now - last_loop_t)))
            last_loop_t = time.time()

            frames = fetch()
            if frames:
                panes = []
                for name, img in frames.items():
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.putText(bgr, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    panes.append(bgr)
                # Match heights so hstack works even if cameras report different sizes.
                h = min(p.shape[0] for p in panes)
                panes = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panes]
                grid = np.hstack(panes)
                cv2.putText(grid, f"{fps_disp:5.1f} fps", (8, grid.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(args.window, grid)

                fps_frames += 1
                if (last_loop_t - last_fps_t) >= 1.0:
                    fps_disp = fps_frames / (last_loop_t - last_fps_t)
                    fps_frames = 0
                    last_fps_t = last_loop_t

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if args.mode == "sub":
            sub.close()
        else:
            req.close()
