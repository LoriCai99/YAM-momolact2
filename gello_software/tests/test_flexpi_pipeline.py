"""End-to-end test of the flex-pi data pipeline with no hardware.

Fake observations -> DataSaver (streaming writer, depth, meta.json) ->
flexpi_convert.convert -> validate_flexpi_dataset.validate.

Frames are tiny (16x12) to keep it fast; shape checks against the 640x360
reference are therefore run with strict_shape=False. Everything else -- feature
keys/dtypes/names, codecs, depth unit conversion, FK/rot6d, gripper passthrough,
index continuity -- is checked exactly.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gello_software" / "scripts"))

from gello.data_utils.data_saver import DataSaver  # noqa: E402
from gello.data_utils.data_saver_thread import EpisodeSaverThread  # noqa: E402

H, W = 12, 16
CAMS = ["left", "front", "right"]
SCALES = {"left": 1e-4, "front": 1e-3, "right": 1e-4}  # D405, D435, D405


def _fake_obs(t: int, rng: np.random.Generator):
    obs = {}
    for cam in CAMS:
        obs[f"{cam}_camera_rgb"] = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        d = rng.integers(0, 6000, (H, W, 1), dtype=np.uint16)
        d[0, 0, 0] = 0  # "no return"
        obs[f"{cam}_camera_depth"] = d
        obs[f"{cam}_camera_timestamp"] = 1000.0 + t / 30
    q = rng.uniform(-0.5, 0.5, 14)
    q[6], q[13] = rng.uniform(0, 1), rng.uniform(0, 1)  # grippers in [0,1]
    obs["joint_positions"] = q
    obs["next_joint"] = q + 0.01
    return obs


def _camera_meta():
    return {
        f"{cam}_camera": {
            "device_id": f"SN{cam}",
            "intrinsics": {"fx": 300.0 + i, "fy": 301.0, "cx": W / 2, "cy": H / 2, "width": W, "height": H},
            "depth_scale_m_per_unit": SCALES[cam],
        }
        for i, cam in enumerate(CAMS)
    }


def _record(tmp_path: Path, n_eps=2, n_frames=6, image_format="png"):
    rng = np.random.default_rng(0)
    saver = DataSaver(save_dir=str(tmp_path), task_directory="raw", language_instruction="Zip it.",
                      fps=30, image_format=image_format, camera_meta=_camera_meta(), saver_max_workers=2)
    thread = EpisodeSaverThread(saver)
    thread.start()
    raw = []
    for _ in range(n_eps):
        saver.reset_buffer()
        frames = [_fake_obs(t, rng) for t in range(n_frames)]
        for o in frames:
            saver.add_observation(o)
        raw.append(frames)
        thread.save_episode(saver.buffer.copy())  # what the control loop does on 'a'
    thread.stop()
    thread.join()
    saver.close()
    return saver, raw


def test_data_saver_streams_depth_and_meta(tmp_path):
    import cv2

    saver, raw = _record(tmp_path, n_eps=1, n_frames=5)
    ep = tmp_path / "raw" / "000001"
    rows = json.load(open(ep / "000001.json"))
    assert len(rows) == 5
    r0 = rows[0]
    for k in ("language_instruction", "left_joint", "right_joint", "next_left_joint", "next_right_joint",
              "frame_index", "timestamp", "camera_timestamps"):
        assert k in r0, k
    assert json.loads(r0["left_joint"]) == pytest.approx(raw[0][0]["joint_positions"][:7].tolist())
    for cam in CAMS:
        assert (ep / f"{cam}_rgb").is_dir() and (ep / f"{cam}_depth").is_dir()
        assert len(list((ep / f"{cam}_rgb").glob("*.png"))) == 5
        back = cv2.imread(r0[f"image_{cam}_depth"], cv2.IMREAD_UNCHANGED)
        assert back.dtype == np.uint16 and np.array_equal(back, raw[0][0][f"{cam}_camera_depth"][:, :, 0])
        rgb_back = cv2.cvtColor(cv2.imread(r0[f"image_{cam}_rgb"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        assert np.array_equal(rgb_back, raw[0][0][f"{cam}_camera_rgb"])
    meta = json.load(open(ep / "meta.json"))
    assert meta["num_frames"] == 5 and meta["fps"] == 30 and meta["instruction"] == "Zip it."
    assert meta["cameras"]["left"]["role"] == "cam_left_wrist"
    assert meta["cameras"]["front"]["role"] == "cam_high"
    assert meta["cameras"]["left"]["depth_scale_m_per_unit"] == 1e-4
    assert meta["cameras"]["front"]["intrinsics"]["fx"] == 301.0
    assert meta["image_write_errors"] == []


def test_data_saver_discard_removes_dir_and_does_not_reuse_index(tmp_path):
    saver = DataSaver(save_dir=str(tmp_path), task_directory="raw", language_instruction="x", fps=30,
                      camera_meta=_camera_meta(), saver_max_workers=2)
    rng = np.random.default_rng(1)
    saver.reset_buffer()
    for t in range(3):
        saver.add_observation(_fake_obs(t, rng))
    assert saver.traj_count == 2 and (tmp_path / "raw" / "000001").exists()
    saver.reset_buffer()  # 'b': nothing was queued for saving -> discard
    saver.close()  # waits for the async rmtree
    assert not (tmp_path / "raw" / "000001").exists()
    # Index is NOT reused: the next episode is 000002, so a stale async cleanup of
    # 000001 can never delete a live directory (2026-09-08 data-loss fix).
    assert saver.traj_count == 2


def test_convert_and_validate(tmp_path):
    import av
    import cv2
    import flexpi_convert as fc
    from validate_flexpi_dataset import validate

    _, raw = _record(tmp_path, n_eps=2, n_frames=6, image_format="jpg")
    out = tmp_path / "flexpi"
    res = fc.convert(tmp_path / "raw", out, task="Zip it.", fps=30, workers=2, quiet=True)
    assert res["episodes"] == 2 and res["frames"] == 12 and res["videos"] == 12
    assert res["cams"] == ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    # schema == reference (keys/dtypes/names/codecs); shapes differ only because frames are tiny
    rep = validate(out, REPO / "docs/flexpi/soft_bag_zipping.info.json", decode_episodes=2, strict_shape=False)
    failed = [r for r in rep.rows if not r[1]]
    assert not failed, failed

    info = json.load(open(out / "meta/info.json"))
    ref = json.load(open(REPO / "docs/flexpi/soft_bag_zipping.info.json"))
    assert list(info["features"]) == list(ref["features"])  # same keys, same order
    assert info["features"]["observation.state"]["names"] == ref["features"]["observation.state"]["names"]
    assert info["features"]["observation.images.cam_high"]["shape"] == [H, W, 3]

    # numeric content: grippers + joints pass through, FK matches i2rt at the home pose
    import pyarrow.parquet as pq

    t = pq.read_table(out / "data/chunk-000/episode_000000.parquet")
    st = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False))
    q0 = raw[0][0]["joint_positions"]
    assert st[0, 18:20] == pytest.approx([q0[6], q0[13]], abs=1e-6)
    assert st[0, 20:26] == pytest.approx(q0[:6].tolist(), abs=1e-6)
    assert st[0, 26:32] == pytest.approx(q0[7:13].tolist(), abs=1e-6)
    fk = fc.YamFK("gripper", "grasp_site")
    p, r6 = fk.pose(np.zeros(6))
    assert p == pytest.approx([0.245, 0.0, 0.174], abs=2e-3)  # flex-pi frame-0 home pose is [0.2477, 0.0001, 0.1708]
    assert r6 == pytest.approx([0, 0, 1, 0, 1, 0], abs=1e-4)  # MuJoCo fp noise ~4e-6
    ac = np.stack(t.column("action").to_numpy(zero_copy_only=False))
    assert ac[0, 20:26] == pytest.approx((q0[:6] + 0.01).tolist(), abs=1e-6)  # action = next_joint

    # depth: native units * scale -> millimetres, bit-exact through FFV1
    for key, cam in (("cam_left_wrist", "left"), ("cam_high", "front")):
        with av.open(str(out / f"videos/chunk-000/observation.depth_ffv1.{key}/episode_000000.mkv")) as c:
            frames = [f.to_ndarray(format="gray16le") for f in c.decode(video=0)]
        assert len(frames) == 6
        expect = fc.depth_to_mm(raw[0][0][f"{cam}_camera_depth"][:, :, 0], SCALES[cam])
        assert np.array_equal(frames[0], expect)
        assert frames[0][0, 0] == 0  # no-return stays 0

    # rgb mp4 decodes to the right count and roughly the right pixels (lossy)
    with av.open(str(out / "videos/chunk-000/observation.images.cam_high/episode_000001.mp4")) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    assert len(frames) == 6 and frames[0].shape == (H, W, 3)

    intr = json.load(open(out / "meta/camera_intrinsics.json"))
    assert intr["cam_high"]["fx"] == pytest.approx(301.0) and intr["cam_high"]["width"] == W
    assert (out / "meta/episodes_stats.jsonl").read_text().count("\n") == 2


def test_convert_refuses_missing_depth_scale(tmp_path):
    import flexpi_convert as fc

    saver = DataSaver(save_dir=str(tmp_path), task_directory="raw", language_instruction="x", fps=30,
                      saver_max_workers=2)  # no camera_meta -> no depth scale
    rng = np.random.default_rng(2)
    saver.reset_buffer()
    for t in range(3):
        saver.add_observation(_fake_obs(t, rng))
    saver.mark_pending_save(saver.buffer)
    saver.save_episode_json(saver.buffer)
    saver.close()
    with pytest.raises(ValueError, match="depth_scale_m_per_unit"):
        fc.convert(tmp_path / "raw", tmp_path / "out", task="x", workers=1, quiet=True)
    # explicit override works
    res = fc.convert(tmp_path / "raw", tmp_path / "out", task="x", workers=1, quiet=True, overwrite=True,
                     depth_scale_override={"left": 1e-4, "front": 1e-3, "right": 1e-4})
    assert res["episodes"] == 1


def test_gripper_channels_are_clipped_to_unit_interval():
    """Follower grippers report a few hundredths past their limits when pressed hard
    (e.g. -0.01 fully closed); flex-pi's range is [0, 1]."""
    import flexpi_convert as fc

    fk = fc.YamFK("gripper", "grasp_site")
    q = np.zeros(14)
    q[6], q[13] = -0.01, 1.02
    st = fc.joints14_to_state32(q, fk)
    assert st[18] == 0.0 and st[19] == 1.0
    assert st[20:26].tolist() == [0.0] * 6  # joints untouched


def test_recorder_logs_stale_cameras_per_frame(tmp_path):
    rng = np.random.default_rng(3)
    saver = DataSaver(save_dir=str(tmp_path), task_directory="raw", language_instruction="x", fps=30,
                      camera_meta=_camera_meta(), saver_max_workers=2)
    saver.reset_buffer()
    for t in range(4):
        o = _fake_obs(t, rng)
        if t >= 2:
            o["camera_stale"] = ["left"]
        saver.add_observation(o)
    saver.mark_pending_save(saver.buffer)
    saver.save_episode_json(saver.buffer)
    saver.close()
    rows = json.load(open(tmp_path / "raw" / "000001" / "000001.json"))
    assert rows[0]["camera_stale"] == [] and rows[3]["camera_stale"] == ["left"]
    assert json.load(open(tmp_path / "raw" / "000001" / "meta.json"))["frames_with_stale_camera"] == 2



def _keyframe_ratio(path):
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        pk = [p for p in c.demux(s) if p.size > 0]
        return sum(bool(p.is_keyframe) for p in pk), len(pk)


def test_all_frames_are_keyframes(tmp_path):
    """flex-pi requires RGB (and depth) videos where every frame is a keyframe (GOP 1)."""
    import numpy as np
    import cv2
    from flexpi_convert import encode_rgb, encode_depth

    n = 12
    rgb_paths, depth_paths = [], []
    for i in range(n):
        img = (np.random.rand(360, 640, 3) * 255).astype(np.uint8)
        cv2.circle(img, (50 + 30 * i, 180), 20, (255, 255, 255), -1)
        p = tmp_path / f"rgb_{i:03d}.jpg"; cv2.imwrite(str(p), img); rgb_paths.append(str(p))
        d = (np.random.rand(360, 640) * 5000).astype(np.uint16)
        q = tmp_path / f"d_{i:03d}.png"; cv2.imwrite(str(q), d); depth_paths.append(str(q))
    encode_rgb(rgb_paths, str(tmp_path / "rgb.mp4"), 30, 20)
    encode_depth(depth_paths, str(tmp_path / "depth.mkv"), 30, 0.001)
    k, tot = _keyframe_ratio(tmp_path / "rgb.mp4")
    assert tot == n and k == n, f"rgb: {k}/{tot} keyframes"
    k, tot = _keyframe_ratio(tmp_path / "depth.mkv")
    assert tot == n and k == n, f"depth: {k}/{tot} keyframes"
