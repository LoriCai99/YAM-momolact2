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
