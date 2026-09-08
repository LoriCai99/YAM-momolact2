import os
import threading
import time
from typing import List, Optional, Tuple
import logging

import numpy as np

from gello.cameras.camera import CameraDriver

logger = logging.getLogger(__name__)

# The one stream configuration every camera must deliver (flex-pi trains on 640x360 @30).
STREAM_WIDTH, STREAM_HEIGHT, STREAM_FPS = 640, 360, 30


def check_stream_support(serial: str, width: int = STREAM_WIDTH, height: int = STREAM_HEIGHT,
                         fps: int = STREAM_FPS) -> dict:
    """Can this device deliver colour bgr8 + depth z16 at (width, height, fps)?

    Returns {"present", "usb", "color", "depth", "ok", "reason"}. Cheap (no streaming),
    so call it before constructing RealSenseCamera: a device on a USB 2 link
    enumerates normally but omits several modes -- a D435 at USB 2.1 has no colour
    640x360 at all -- and ``pipeline.start`` then fails with the opaque
    "Couldn't resolve requests".
    """
    import pyrealsense2 as rs

    out = {"present": False, "usb": None, "color": False, "depth": False, "ok": False, "reason": "not connected"}
    for dev in rs.context().query_devices():
        if dev.get_info(rs.camera_info.serial_number) != serial:
            continue
        out["present"] = True
        out["usb"] = dev.get_info(rs.camera_info.usb_type_descriptor)
        for sensor in dev.query_sensors():
            for prof in sensor.get_stream_profiles():
                try:
                    v = prof.as_video_stream_profile()
                except Exception:
                    continue
                if (v.width(), v.height(), prof.fps()) != (width, height, fps):
                    continue
                if prof.stream_type() == rs.stream.color and prof.format() == rs.format.bgr8:
                    out["color"] = True
                if prof.stream_type() == rs.stream.depth and prof.format() == rs.format.z16:
                    out["depth"] = True
        out["ok"] = out["color"] and out["depth"]
        if out["ok"]:
            out["reason"] = "ok"
        elif str(out["usb"]).startswith("2"):
            out["reason"] = (f"USB {out['usb']} link: this mode is not offered at USB 2. "
                             f"Needs a USB 3 link (cable/port/camera).")
        else:
            out["reason"] = f"mode {width}x{height}@{fps} unsupported (colour={out['color']}, depth={out['depth']})"
        break
    return out


def list_device_ids() -> List[str]:
    """Serial numbers of connected RealSense devices. No hardware reset."""
    import pyrealsense2 as rs

    return [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]


def get_device_ids(reset: bool = True) -> List[str]:
    """List connected RealSense serials, optionally hardware-resetting each first.

    IMPORTANT: hardware_reset() drops every device off the USB bus for ~2 s while
    it re-enumerates. Doing it twice in quick succession (e.g. once in the launcher
    and again in the camera-server child) has raced and left a camera disconnected
    (2026-09-08). Reset AT MOST ONCE per launch; pass reset=False everywhere else.
    """
    import pyrealsense2 as rs

    ctx = rs.context()
    device_ids = []
    for dev in ctx.query_devices():
        if reset:
            dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    if reset:
        time.sleep(2)
    return device_ids


class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(self, device_id: Optional[str] = None, flip: bool = False):
        import pyrealsense2 as rs

        self._device_id = device_id
        self._flip = flip
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._warmup_frames = 15
        self._read_timeout_ms = 1200
        self._read_wait_timeout_sec = 1.5
        self._max_frame_age_sec = 0.30
        self._max_read_attempts = 5
        self._latest_color_image = None
        self._latest_depth_image = None
        self._latest_frame_timestamp = None
        self._frame_count = 0  # incremented per captured frame; lets a publisher phase-lock to the camera
        self._last_capture_error = None
        self._frame_ready = threading.Event()
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._intrinsics = None
        self._depth_scale = None

        self._rs = rs
        self._pipeline = None
        self._config = None
        self._align = rs.align(rs.stream.color)

        self._start_pipeline()
        self._start_capture_thread()

    def _start_capture_thread(self):
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"realsense_capture_{self._device_id or 'default'}",
            daemon=True,
        )
        self._capture_thread.start()

    def _capture_loop(self):
        consecutive_failures = 0
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    frames = self._pipeline.wait_for_frames(timeout_ms=self._read_timeout_ms)
                    frames = self._align.process(frames)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    raise RuntimeError("Invalid RealSense frame pair received.")

                color_image = np.asanyarray(color_frame.get_data()).copy()
                depth_image = np.asanyarray(depth_frame.get_data()).copy()
                timestamp = time.time()

                with self._frame_lock:
                    self._latest_color_image = color_image
                    self._latest_depth_image = depth_image
                    self._latest_frame_timestamp = timestamp
                    self._frame_count += 1
                    self._last_capture_error = None
                    self._frame_ready.set()

                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                with self._frame_lock:
                    self._last_capture_error = exc
                if consecutive_failures >= self._max_read_attempts:
                    self._frame_ready.set()
                time.sleep(0.05)
                # pipeline.start() on a device that is NOT enumerated blocks ~5 s
                # inside librealsense ("Failed to reconnect: No device connected5000")
                # while HOLDING THE GIL -- which froze the whole camera server,
                # healthy cameras included, for 10+ s on 2026-09-08. Only attempt a
                # restart once the device is back on the bus; poll cheaply meanwhile.
                if not self._device_present():
                    with self._frame_lock:
                        self._last_capture_error = RuntimeError(
                            f"camera {self._device_id} is not enumerated (disconnected?)"
                        )
                    if self._stop_event.wait(1.0):
                        break
                    continue
                try:
                    self._start_pipeline()
                except Exception as exc2:  # noqa: BLE001
                    with self._frame_lock:
                        self._last_capture_error = exc2
                    if self._stop_event.wait(2.0):
                        break

    def _device_present(self) -> bool:
        """Is this serial currently enumerated? Milliseconds, no long blocking call."""
        if self._device_id is None:
            return True
        try:
            for dev in self._rs.context().query_devices():
                if dev.get_info(self._rs.camera_info.serial_number) == self._device_id:
                    return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def _start_pipeline(self):
        rs = self._rs

        with self._lock:
            if self._pipeline:
                try:
                    self._pipeline.stop()
                except Exception:
                    pass

            self._pipeline = rs.pipeline()
            self._config = rs.config()

            if self._device_id is not None:
                self._config.enable_device(self._device_id)

            self._config.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT, rs.format.z16, STREAM_FPS)
            self._config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, STREAM_FPS)

            # A device may still be re-enumerating (e.g. just after a hardware reset):
            # "Device disconnected"/"No device connected" here is usually transient.
            # Retry a few times before giving up, so a brief USB blip does not kill
            # the whole camera server.
            last_exc = None
            for attempt in range(1, 6):
                try:
                    profile = self._pipeline.start(self._config)
                    self._cache_stream_meta(profile)
                    for _ in range(self._warmup_frames):
                        self._pipeline.wait_for_frames()
                    break
                except RuntimeError as exc:
                    msg = str(exc)
                    if "resolve requests" in msg:
                        raise  # a mode/USB-link problem: retrying will not help
                    # A failure inside warm-up leaves the pipeline STARTED; a retry
                    # then fails with "start() cannot be called before stop()". Tear
                    # it down and build a fresh pipeline before the next attempt.
                    try:
                        self._pipeline.stop()
                    except Exception:  # noqa: BLE001
                        pass
                    self._pipeline = rs.pipeline()
                    if not self._device_present():
                        raise RuntimeError(f"camera {self._device_id} disconnected during start: {msg.splitlines()[0][:80]}")
                    last_exc = exc
                    logger.warning(
                        "camera %s start attempt %d/5 failed (%s); retrying",
                        self._device_id, attempt, msg.splitlines()[0][:80],
                    )
                    time.sleep(1.5)
            else:
                raise RuntimeError(
                    f"camera {self._device_id}: pipeline did not start after 5 attempts: {last_exc}"
                )

    def _cache_stream_meta(self, profile) -> None:
        """Record intrinsics + depth scale once per pipeline start.

        Depth is aligned to the colour stream in ``_capture_loop`` (``rs.align``),
        so the colour intrinsics describe both images -- one K per camera, which is
        what a flex-pi ``camera_intrinsics.json`` expects. Cached here (inside the
        pipeline lock) so callers never touch the pipeline from another thread.
        """
        rs = self._rs
        try:
            vs = profile.get_stream(rs.stream.color).as_video_stream_profile()
            i = vs.get_intrinsics()
            self._intrinsics = {
                "fx": float(i.fx), "fy": float(i.fy), "cx": float(i.ppx), "cy": float(i.ppy),
                "width": int(i.width), "height": int(i.height),
                "model": str(i.model).split(".")[-1], "coeffs": [float(c) for c in i.coeffs],
            }
        except Exception as exc:  # keep capturing even if a driver lacks intrinsics
            logger.warning(f"{self}: could not read colour intrinsics: {exc}")
            self._intrinsics = None
        try:
            sensor = profile.get_device().first_depth_sensor()
            # metres per raw uint16 unit. D435: 0.001 (1 mm). D405: 0.0001 (0.1 mm)!
            self._depth_scale = float(sensor.get_depth_scale())
        except Exception as exc:
            logger.warning(f"{self}: could not read depth scale: {exc}")
            self._depth_scale = None

    def close(self) -> None:
        """Stop the capture thread and the pipeline.

        Without this, interpreter exit tears librealsense down underneath a
        still-running capture thread and aborts with
        "terminate called without an active exception" (core dump). Harmless for
        saved data, alarming in a terminal, so call it from cleanup paths.
        """
        self._stop_event.set()
        t = self._capture_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=3.0)
        with self._lock:
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception:
                    pass
                self._pipeline = None

    # ---- metadata accessors used by RobotEnv.get_camera_meta() ---------------

    @property
    def device_id(self) -> Optional[str]:
        return self._device_id

    @property
    def frame_count(self) -> int:
        """Number of frames captured so far (monotonic)."""
        return self._frame_count

    @property
    def last_frame_timestamp(self) -> Optional[float]:
        """Wall-clock time (s) at which the frame returned by the next ``read()`` was captured."""
        with self._frame_lock:
            return self._latest_frame_timestamp

    def get_depth_scale(self) -> Optional[float]:
        """Metres per depth unit for this device (see ``_cache_stream_meta``)."""
        return getattr(self, "_depth_scale", None)

    def get_intrinsics(self) -> Optional[dict]:
        """Pinhole intrinsics of the frames ``read()`` returns (accounts for ``flip``)."""
        intr = getattr(self, "_intrinsics", None)
        if intr is None:
            return None
        intr = dict(intr)
        if self._flip:  # 180-degree rotation mirrors the principal point
            intr["cx"] = (intr["width"] - 1) - intr["cx"]
            intr["cy"] = (intr["height"] - 1) - intr["cy"]
        return intr

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,  # farthest: float = 0.12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """
        import cv2

        if not self._frame_ready.wait(timeout=self._read_wait_timeout_sec):
            raise RuntimeError("Timed out waiting for RealSense capture thread to produce a frame.")

        with self._frame_lock:
            color_image = self._latest_color_image
            depth_image = self._latest_depth_image
            frame_timestamp = self._latest_frame_timestamp
            last_error = self._last_capture_error

        if color_image is None or depth_image is None or frame_timestamp is None:
            if last_error is not None:
                raise RuntimeError("RealSense capture thread failed to produce a frame.") from last_error
            raise RuntimeError("RealSense frame is unavailable.")

        frame_age = time.time() - frame_timestamp
        if frame_age > self._max_frame_age_sec:
            raise RuntimeError(
                f"RealSense frame is stale ({frame_age:.3f}s old); camera may be stalled."
            )

        if img_size is None:
            image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            depth = depth_image
        else:
            resized_color = cv2.resize(color_image, img_size)
            image = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)
            depth = cv2.resize(depth_image, img_size)

        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)

        depth = depth[:, :, None]

        return image, depth


def _debug_read(camera, save_datastream=False):
    import cv2

    cv2.namedWindow("image")
    cv2.namedWindow("depth")
    counter = 0
    if not os.path.exists("images"):
        os.makedirs("images")
    if save_datastream and not os.path.exists("stream"):
        os.makedirs("stream")
    while True:
        time.sleep(0.1)
        image, depth = camera.read()
        depth = np.concatenate([depth, depth, depth], axis=-1)
        key = cv2.waitKey(1)
        cv2.imshow("image", image[:, :, ::-1])
        cv2.imshow("depth", depth)
        if key == ord("s"):
            cv2.imwrite(f"images/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"images/depth_{counter}.png", depth)
        if save_datastream:
            cv2.imwrite(f"stream/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"stream/depth_{counter}.png", depth)
        counter += 1
        if key == 27:
            break


if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    rs = RealSenseCamera(flip=True, device_id=device_ids[0])
    im, depth = rs.read()
    _debug_read(rs, save_datastream=True)
