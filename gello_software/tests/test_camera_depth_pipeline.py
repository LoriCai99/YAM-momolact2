"""Depth + metadata through the camera server -> client -> RobotEnv path.

Why this path exists: rs.align holds the GIL 10-20 ms per frame, which in the
arms' process jitters the 250 Hz control loops. The collection launcher runs
the cameras in a child process and must still receive depth, per-camera
timestamps and intrinsics/depth-scale for the flex-pi recorder.
"""

import threading
import time

import numpy as np
import pytest
import zmq

from gello.cameras.camera_client import CameraClient
from gello.cameras.camera_server import CameraServer
from gello.env import RobotEnv

H, W = 48, 64


class FakeCam:
    def __init__(self, serial, scale):
        self.device_id = serial
        self._scale = scale
        self.last_frame_timestamp = time.time()
        self.closed = False

    def read(self):
        rgb = np.full((H, W, 3), 7, np.uint8)
        depth = (np.arange(H * W, dtype=np.uint16).reshape(H, W, 1) % 5000)
        self.last_frame_timestamp = time.time()
        return rgb, depth

    def get_intrinsics(self):
        return {"fx": 300.0, "fy": 301.0, "cx": W / 2, "cy": H / 2, "width": W, "height": H}

    def get_depth_scale(self):
        return self._scale

    def close(self):
        self.closed = True


def _cams():
    return {"left_camera": FakeCam("L1", 1e-4), "front_camera": FakeCam("F1", 1e-3)}


def test_snapshot_has_depth_and_timestamps():
    snap = CameraServer(_cams(), rep_endpoint="inproc://x")._snapshot()
    assert set(snap) == {"ok", "frames", "depth", "timestamps"}
    assert snap["depth"]["left_camera"].dtype == np.uint16 and snap["depth"]["left_camera"].shape == (H, W)
    assert snap["frames"]["front_camera"].shape == (H, W, 3)
    assert abs(time.time() - snap["timestamps"]["front_camera"]) < 1.0


def test_meta_reports_serial_intrinsics_scale():
    meta = CameraServer(_cams(), rep_endpoint="inproc://x")._meta()["meta"]
    assert meta["left_camera"]["device_id"] == "L1"
    assert meta["left_camera"]["depth_scale_m_per_unit"] == 1e-4
    assert meta["front_camera"]["depth_scale_m_per_unit"] == 1e-3
    assert meta["front_camera"]["intrinsics"]["width"] == W


@pytest.fixture
def running_server():
    port = zmq.Context.instance().socket(zmq.REP)
    p = port.bind_to_random_port("tcp://127.0.0.1")
    port.close(linger=0)
    endpoint = f"tcp://127.0.0.1:{p}"
    cams = _cams()
    server = CameraServer(cams, rep_endpoint=endpoint, pub_endpoint=None)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(0.2)
    yield endpoint, cams
    server.shutdown()
    t.join(timeout=2)


def test_client_full_obs_and_meta_round_trip(running_server):
    endpoint, cams = running_server
    c = CameraClient(endpoint, request_timeout_ms=2000, max_frame_age_sec=1.0)
    try:
        assert c.ping()
        full = c.get_obs_full()
        assert full["depth"]["left_camera"].dtype == np.uint16 and full["depth"]["left_camera"].shape == (H, W)
        assert full["frames"]["left_camera"].shape == (H, W, 3)
        assert set(full["timestamps"]) == {"left_camera", "front_camera"}
        assert c.get_obs()["front_camera"].shape == (H, W, 3)  # legacy API still works
        assert c.get_meta()["front_camera"]["depth_scale_m_per_unit"] == 1e-3
    finally:
        c.close()


class _Robot:
    def num_dofs(self):
        return 14

    def get_joint_state(self):
        return np.zeros(14)

    def get_observations(self):
        return {"joint_positions": np.zeros(14), "joint_velocities": np.zeros(14),
                "ee_pos_quat": np.zeros(7), "gripper_position": np.zeros(1)}


def test_robot_env_uses_client_depth_timestamps_and_meta(running_server):
    endpoint, cams = running_server
    c = CameraClient(endpoint, request_timeout_ms=2000, max_frame_age_sec=1.0)
    try:
        env = RobotEnv(_Robot(), control_rate_hz=30, camera_client=c)
        obs = env.get_obs()
        assert obs["left_camera_rgb"].shape == (H, W, 3)
        assert obs["left_camera_depth"].shape == (H, W, 1) and obs["left_camera_depth"].dtype == np.uint16
        assert isinstance(obs["front_camera_timestamp"], float)
        assert "joint_positions" in obs
        meta = env.get_camera_meta()
        assert meta["left_camera"]["device_id"] == "L1" and meta["left_camera"]["depth_scale_m_per_unit"] == 1e-4
    finally:
        c.close()


def test_server_shutdown_closes_cameras(running_server):
    pass  # covered by fixture teardown below


def test_shutdown_calls_close():
    cams = _cams()
    s = CameraServer(cams, rep_endpoint="inproc://y")
    s._stop_event.clear()
    s.shutdown()
    assert all(c.closed for c in cams.values())


@pytest.fixture
def running_pub_server():
    ctx = zmq.Context.instance()
    ports = []
    for _ in range(2):
        sock = ctx.socket(zmq.REP)
        ports.append(sock.bind_to_random_port("tcp://127.0.0.1"))
        sock.close(linger=0)
    rep, pub = (f"tcp://127.0.0.1:{p}" for p in ports)
    cams = _cams()
    server = CameraServer(cams, rep_endpoint=rep, pub_endpoint=pub, pub_period_sec=1 / 60, pub_format="multipart")
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(0.3)
    yield rep, pub, cams
    server.shutdown()
    t.join(timeout=2)


def test_obs2_multipart_round_trip(running_server):
    endpoint, _ = running_server
    c = CameraClient(endpoint, request_timeout_ms=2000, max_frame_age_sec=1.0)
    try:
        full = c.get_obs_full()  # uses obs2 (multipart, zero-copy)
        assert not c._legacy_obs
        assert full["depth"]["front_camera"].dtype == np.uint16 and full["depth"]["front_camera"].shape == (H, W)
        assert full["frames"]["front_camera"].shape == (H, W, 3) and full["frames"]["front_camera"][0, 0, 0] == 7
        assert full["depth"]["left_camera"][0, 1] == 1  # arange % 5000
    finally:
        c.close()


def test_stream_client_serves_latest_frames_without_blocking(running_pub_server):
    from gello.cameras.camera_client import CameraStreamClient

    rep, pub, _ = running_pub_server
    c = CameraStreamClient(rep, pub, request_timeout_ms=2000, max_frame_age_sec=1.0)
    try:
        assert c.ping()
        full = c.get_obs_full()
        assert c._pub_ok is True
        assert full["depth"]["left_camera"].shape == (H, W) and full["frames"]["front_camera"].shape == (H, W, 3)
        t = time.perf_counter()
        for _ in range(200):
            c.get_obs_full()
        per_call_ms = (time.perf_counter() - t) / 200 * 1e3
        assert per_call_ms < 1.0, f"get_obs_full took {per_call_ms:.2f} ms; must not touch the server"
        assert c.get_meta()["front_camera"]["depth_scale_m_per_unit"] == 1e-3
        env = RobotEnv(_Robot(), control_rate_hz=30, camera_client=c)
        obs = env.get_obs()
        assert obs["front_camera_depth"].shape == (H, W, 1) and isinstance(obs["left_camera_timestamp"], float)
    finally:
        c.close()


def test_stream_client_falls_back_to_req_when_pub_is_pickle():
    from gello.cameras.camera_client import CameraStreamClient

    ctx = zmq.Context.instance()
    ports = []
    for _ in range(2):
        sock = ctx.socket(zmq.REP)
        ports.append(sock.bind_to_random_port("tcp://127.0.0.1"))
        sock.close(linger=0)
    rep, pub = (f"tcp://127.0.0.1:{p}" for p in ports)
    server = CameraServer(_cams(), rep_endpoint=rep, pub_endpoint=pub, pub_period_sec=1 / 60, pub_format="pickle")
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(0.3)
    c = CameraStreamClient(rep, pub, request_timeout_ms=2000, max_frame_age_sec=1.0)
    try:
        time.sleep(0.3)  # let the receiver see a (pickle) message and decide
        full = c.get_obs_full()
        assert c._pub_ok is False
        assert full["depth"]["left_camera"].shape == (H, W)
    finally:
        c.close()
        server.shutdown()
        t.join(timeout=2)


def test_event_driven_pub_waits_for_all_cameras():
    """With pub_on_new_frame, the PUB loop publishes only when every camera advanced (or on fallback)."""
    cams = _cams()
    for c in cams.values():
        c.frame_count = 0
    s = CameraServer(cams, rep_endpoint="inproc://z", pub_on_new_frame=True, pub_fallback_sec=10.0)
    assert s._all_cameras_advanced({}) is True          # nothing published yet -> publish
    last = {n: 0 for n in cams}
    assert s._all_cameras_advanced(last) is False       # no camera advanced past 0
    cams["left_camera"].frame_count = 1
    assert s._all_cameras_advanced(last) is False       # only one camera advanced
    cams["front_camera"].frame_count = 1
    assert s._all_cameras_advanced(last) is True        # all advanced
