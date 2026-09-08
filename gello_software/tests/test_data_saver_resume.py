"""Next episode index is inferred from the directory; crashed leftovers are removed."""

import json
import os

import numpy as np
import pytest

from gello.data_utils.data_saver import DataSaver


def _complete(root, idx):
    d = os.path.join(root, f"{idx:06d}")
    os.makedirs(os.path.join(d, "front_rgb"), exist_ok=True)
    json.dump([{"frame_index": 0}], open(os.path.join(d, f"{idx:06d}.json"), "w"))
    json.dump({"num_frames": 1}, open(os.path.join(d, "meta.json"), "w"))


def _incomplete(root, idx):
    d = os.path.join(root, f"{idx:06d}", "front_rgb")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "000000.jpg"), "wb").close()


def test_enter_appends_after_last_complete_and_removes_leftovers(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    for i in (1, 2, 5):
        _complete(root, i)
    _incomplete(root, 6)  # a crashed run
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    saver = DataSaver(save_dir=str(tmp_path), task_directory="raw", language_instruction="x", fps=30, saver_max_workers=1)
    saver.close()
    assert saver.traj_count == 6
    assert not (root / "000006").exists()  # leftover removed, slot reused
    assert (root / "000005" / "meta.json").exists()  # real data untouched


def test_explicit_number_refuses_to_overwrite(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    _complete(root, 1)
    answers = iter(["1", "7"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    saver = DataSaver(save_dir=str(tmp_path), task_directory="raw", language_instruction="x", fps=30, saver_max_workers=1)
    saver.close()
    assert saver.traj_count == 7 and (root / "000001" / "meta.json").exists()


def test_delete_requires_the_word_and_confirmation(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    _complete(root, 1)
    answers = iter(["y", "delete", "no", "delete", "yes"])  # 'y' is no longer a delete
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    saver = DataSaver(save_dir=str(tmp_path), task_directory="raw", language_instruction="x", fps=30, saver_max_workers=1)
    saver.close()
    assert saver.traj_count == 1 and not (root / "000001").exists()


def test_no_complete_episodes_starts_at_one_without_prompt(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    _incomplete(root, 3)
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("should not prompt"))
    saver = DataSaver(save_dir=str(tmp_path), task_directory="raw", language_instruction="x", fps=30, saver_max_workers=1)
    saver.close()
    assert saver.traj_count == 1 and not (root / "000003").exists()


def test_discard_then_resave_does_not_delete_the_saved_episode(tmp_path):
    """Regression for 2026-09-08 data loss: a discarded episode's async cleanup
    must never remove the directory of the episode recorded after it."""
    import concurrent.futures

    import numpy as np

    from gello.data_utils.data_saver import DataSaver
    from gello.data_utils.data_saver_thread import EpisodeSaverThread

    def obs(t):
        o = {"joint_positions": np.zeros(14), "next_joint": np.zeros(14)}
        for c in ("left", "front", "right"):
            o[f"{c}_camera_rgb"] = np.zeros((8, 8, 3), np.uint8)
            o[f"{c}_camera_depth"] = np.zeros((8, 8, 1), np.uint16)
        return o

    s = DataSaver(save_dir=str(tmp_path), task_directory="raw", language_instruction="x", fps=30, saver_max_workers=2)
    held = []
    real = s._pool.submit

    def gated(fn, *a, **k):
        if getattr(fn, "__name__", "") == "_rm":   # hold the discard cleanup
            held.append(fn)
            return real(lambda: None)
        return real(fn, *a, **k)

    s._pool.submit = gated
    th = EpisodeSaverThread(s)
    th.start()
    s.reset_buffer()
    for t in range(3):
        s.add_observation(obs(t))
    s.reset_buffer()                       # discard A (cleanup captured, not yet run)
    for t in range(3):
        s.add_observation(obs(t))
    idx_b = s._current.index
    th.save_episode(s.buffer.copy())
    th.stop()
    th.join()
    assert (tmp_path / "raw" / f"{idx_b:06d}" / "meta.json").exists()
    for fn in held:                        # the delayed discard cleanup now fires
        fn()
    assert (tmp_path / "raw" / f"{idx_b:06d}" / "meta.json").exists(), "saved episode was deleted by stale cleanup"
    s._pool.submit = real
    s.close()


def test_finalize_on_exit_flushes_pending_save_and_discards_in_progress(tmp_path):
    """Ctrl-C / crash: a take already sent to save is finalised; the take in progress is removed."""
    import numpy as np

    from gello.data_utils.data_saver import DataSaver

    def obs(t):
        o = {"joint_positions": np.zeros(14), "next_joint": np.zeros(14)}
        for c in ("left", "front", "right"):
            o[f"{c}_camera_rgb"] = np.zeros((8, 8, 3), np.uint8)
            o[f"{c}_camera_depth"] = np.zeros((8, 8, 1), np.uint16)
        return o

    s = DataSaver(save_dir=str(tmp_path), task_directory="raw", language_instruction="x", fps=30, saver_max_workers=2)
    s.reset_buffer()
    for t in range(3):
        s.add_observation(obs(t))
    s.mark_pending_save(s.buffer)   # operator pressed save; background saver has NOT run yet
    s.reset_buffer()                # loop moved on
    for t in range(2):
        s.add_observation(obs(t))   # new take in progress, never saved
    s.finalize_on_exit()            # what cleanup() calls on crash / Ctrl-C
    s.close()
    assert (tmp_path / "raw" / "000001" / "meta.json").exists(), "pending save was not finalised"
    assert not (tmp_path / "raw" / "000002").exists(), "in-progress take was not discarded"
