# Data spec: finetuning flex-pi on this rig

Target: `flex-pi/soft_bag_zipping` on HuggingFace — 534 episodes, 1.05 M frames,
**the same task as ours** (`source_dirs`: `put_pen_into_green_bag_V2`, `…grey…`,
`…pink…`, …). Its `meta/` is cached in `docs/flexpi/` and is the ground truth for
everything below. The goal of collection here is to make finetuning a *small*
domain shift from that dataset: identical format, identical sensing, identical
protocol; vary only the scene.

## 1. What flex-pi's dataset looks like

| | Value |
|---|---|
| Format | LeRobot **v2.1** |
| fps | 30 |
| RGB | 3 cameras, **640×360**, h264 `.mp4` (`yuv420p`) |
| Depth | 3 cameras, 640×360, `uint16` **millimetres**, FFV1 `gray16le` `.mkv` (lossless), `0` = no return |
| Camera keys | `cam_high` (overhead), `cam_left_wrist`, `cam_right_wrist` |
| State / action | **32-D** each, same layout (below) |
| Instruction | *"Unzip the bag, pick up the pens from the table one at a time and place them inside, then zip the bag closed."* |
| Episode length | mean 65 s (25–144 s) |
| Extras | `meta/camera_intrinsics.json` (pinhole K per camera at 640×360) |

32-D vector, grouped **by field** (not by arm):

```
[0:3]   left_pos_{x,y,z}      left end-effector position, metres
[3:9]   left_rot6d_{0..5}     first two ROWS of R, row-major (Zhou et al. 2019)
[9:12]  right_pos_{x,y,z}
[12:18] right_rot6d_{0..5}
[18:20] left_gripper, right_gripper      0..1
[20:26] left_joint_{0..5}                radians
[26:32] right_joint_{0..5}
```

End-effector frame: **each arm's own base frame**, site = fingertip TCP. Verified
against their data: flex-pi frame 0 (home pose) reads `[0.2477, 0.0001, 0.1708]`;
the i2rt YAM MuJoCo model *with* the LINEAR_4310 gripper at `q = 0`, site
`grasp_site`, gives `[0.245, 0.000, 0.174]`. The `arm`-only model gives `0.111` —
wrong. So: gripper model, `grasp_site`.

Quirk to mirror: their `features.*.names` for state/action is a **nested** list
`[[…32 names…]]`. `flexpi_convert.py` reproduces it.

## 2. What our pipeline produces

### Collection (`experiments/launch_yaml_collect_data.py`)

Per episode under `storage.base_dir/storage.task_directory/NNNNNN/`:

```
NNNNNN.json          per-frame: joints, next_joint (= action), timestamps, image paths
meta.json            instruction, fps, per-camera intrinsics + depth_scale + flex-pi role
left_rgb/ front_rgb/ right_rgb/       000000.jpg    (q95; storage.image_format: png to change)
left_depth/ front_depth/ right_depth/ 000000.png    16-bit, NATIVE sensor units
```

Two things that were **lost** before this rewrite and are now captured:

* **Depth.** The old saver dropped it at `env.py` (`image, _depth = camera.read()`).
  It is now written per frame as 16-bit PNG in native units.
* **Depth scale + intrinsics.** These differ per camera model: **D405 = 0.1 mm/unit
  (`1e-4` m), D435 = 1 mm/unit (`1e-3` m)**. Without `meta.json` the wrist depth
  would be off by 10×. The converter refuses to guess.

The recorder streams frames to disk as they arrive (a two-minute episode is
~8 GB of RGB+depth and must not sit in RAM). Measured on this box: 0.3 ms per
`add_observation` at 30 Hz, zero backlog.

### Conversion (`flexpi_convert.py`, runs in the `yam` env — no lerobot needed)

```bash
python flexpi_convert.py                       # reads storage:/flexpi: from gello_software/configs/yam_left.yaml
python gello_software/scripts/validate_flexpi_dataset.py /home/evan/yam_data/put_pen_in_bag_flexpi_v21
```

Writes the v2.1 layout directly with pyarrow + PyAV (lerobot 0.5.x only writes
v3.0, and `depth_video` is a flex-pi extension no LeRobot version emits):

* FK via `i2rt.robots.kinematics.Kinematics` → 32-D state and action
  (`action` = FK of the commanded/next joints, i.e. `next_joint_fields`).
* Depth: `depth_to_mm()` = `round_half_even(raw × scale_mm)`, then FFV1 gray16le.
* RGB: libx264 `crf 20` `yuv420p` (reference is h264 yuv420p).
* `meta/`: `info.json` (feature-for-feature identical to the reference),
  `tasks.jsonl`, `episodes.jsonl`, `episodes_stats.jsonl` (state/action only, as
  the reference), `camera_intrinsics.json` (mean over episodes), plus
  `conversion.json` provenance (not part of v2.1).

The validator diffs `info.json` against `docs/flexpi/soft_bag_zipping.info.json`,
opens every parquet (columns, dtypes, row counts, global-index continuity),
decodes the first frame of every video (codec, pix_fmt, shape, dtype) and checks
the state numerically (rot6d rows orthonormal, grippers in [0,1]). **Run it before
handing data over.**

## 3. Open questions for the flex-pi team

1. **Camera FOV.** Our D405 wrist cams report fx≈326 at 640×360; theirs are fx≈366
   (~12 % narrower). Their `cam_high` fx≈262 implies ~101° HFOV — wider than a
   D435's 69° colour FOV, so it may be a different camera or stream mode. Ask
   what hardware/modes they used; matching it is cheaper than fighting it in
   training.
2. **Gripper convention.** Ours is i2rt command space, 0..1, **1 = open**. Theirs
   spans 0.04–0.996. Confirm 1 = open on their side.
3. **`action26` in their builder name** (`flexpi-eef32-action26-openpi-aligned`)
   while `action` is 32-D in the dataset — presumably the trainer slices. Ask for
   the builder/trainer config rather than guessing.

## 4. Collection protocol (what the operator does)

* Instruction is fixed to flex-pi's exact string (config `storage.language_instruction`).
* One pen at a time; bag starts zipped and ends zipped. Don't rush: 25–144 s episodes.
* Save only clean successes (`a`); discard fumbles or any episode with a dropped camera (`b`).
* Leaders in the home pose at launch (defines the calibration); consistent arm start pose.
* Vary bags (colour, position, orientation) and pen count/placement. **Never move the cameras.**
* Collect **5–10 episodes first**, convert, validate, and run the training team's
  loader on them before scaling to 100.

## 5. Hard prerequisites on this rig

* The **D435 must link at USB 3** — at USB 2.1 it offers no colour 640×360 and
  sustains only ~27 fps. `python gello_software/scripts/check_cameras.py` must say READY.
* Both grippers calibrated (`scripts/calibrate_gripper.py`) — a trigger past its
  configured closed bound records a constant gripper channel.
