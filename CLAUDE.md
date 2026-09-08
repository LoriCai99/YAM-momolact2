# CLAUDE.md

Guidance for future Claude sessions working in this repo.

## What this repo is

**Bimanual YAM** — a workspace stitching together four upstream-ish projects to run teleop, data collection, and policy evaluation on a bimanual YAM robot arm setup.

Top-level layout:

| Path | Purpose |
|---|---|
| `i2rt/` | Low-level motor / CAN driver code. Contains `motor_config_tool/` (timeout/zero/ping) and `scripts/reset_all_can.sh`. |
| `gello_software/` | Main runtime: configs, teleop launchers, data collection, eval. Most day-to-day work happens here under `experiments/` and `configs/`. |
| `lerobot/` | Local checkout of LeRobot used for dataset conversion / training-side compat. |
| `oculus_reader/` | Optional Oculus VR teleop input. |
| `robots_realtime/` | Newer realtime-control sandbox (untracked in git on this machine). |
| `molmoact_to_lerobot_v30.py` | Top-level converter: raw collected JSON → LeRobot v3.0 dataset, with optional HF upload + tag. |
| `docs/` | Lab-facing tutorials; `docs/grasp_lab_eval.md` is the GRASP rig walkthrough. |
| `videos/` | Gitignored output dir. |

## This machine's environment

> Re-surveyed 2026-09-02. The previous contents of this section described a
> *different* workstation (`ai2_yam` env, `can_leader_l`/`can_follower_r`,
> `/home/kostas-lab/...`) and did not match any hardware here.

- **Conda env:** `yam` (Python 3.12) at `/opt/conda/envs/yam`. There is no `ai2_yam` env on this box. `yam_convert` is a separate, lerobot-only env (kept apart because lerobot needs `huggingface-hub>=1.0` while the MolmoAct server path needs `<1.0`).
- **CAN interfaces:** plain **`can0`** and **`can1`** — there are no udev naming rules, so nothing produces `can_leader_l`/`can_follower_r`. Both are gs_usb (OpenMoko 1d50:606f) adapters; `can0` is USB path `1-7`, `can1` is `1-8`.
  - **`can0` = LEFT arm, `can1` = RIGHT arm** — verified 2026-09-08 with `scripts/identify_sides.py` (moving the right arm by hand moved `can1` only). The 2026-09-02 wiggle test that recorded the opposite was performed on the right arm; `/home/evan/carter/STATUS.md` (can0 = left) was correct. **After any re-cabling, run `identify_sides.py`** — a mislabelled pair feels perfect in teleop but swaps left/right in every saved episode.
- **RealSense cameras:** D435 `922612071156` (front/top role), D405 `335122270697` (left), D405 `218622275075` (right) — confirmed against the physical rig by the operator 2026-09-02. Order must stay `[front/top, left, right]`, the order MolmoAct2 was trained on. Inspect live with `python gello_software/scripts/view_cameras.py` (3-pane cv2 viewer; `q` quits, `s` snapshots — quit it before teleop/collection, cameras are exclusive-access).
- **Camera USB caveats.** The front D435 must link at USB 3 — at USB 2.1 it offers no colour 640x360 and collection refuses to start. It spent 2026-09-02→08 at USB 2.1 and the cause was the **USB-C cable** (no SuperSpeed wires; identical-looking), not the port or camera — swap the cable first if it recurs. Both D405s are USB 3.2 but were originally on `05:00.0`, an **ASMedia ASM2142/3142** controller that dropped both mid-session; recover without a reboot via `echo 1 | sudo tee /sys/bus/pci/devices/0000:05:00.0/remove && echo 1 | sudo tee /sys/bus/pci/rescan`. Keep them on separate controllers if possible.
- **`i2rt` is NOT the vendored copy.** The `yam` env has an editable install pointing at `/home/evan/i2rt` (v1.2.4), which is *newer* than this repo's `i2rt/` subdir. `get_yam_robot()` there is API-compatible with `gello/robots/yam.py`, but be aware `import i2rt` never reads the in-repo tree.
- **GELLO leader:** working as of 2026-09-02. Two FT232H (U2D2) adapters, and the `port:` values already in the configs are correct:
  - `FTAO9WPU` → **left** leader, answers at **57600** with Dynamixel IDs `[1..7]`
  - `FTAO9WCV` → **right** leader, IDs `[8..14]`
  - The shipped configs had the CAN channels crossed (left leader drove the right arm). An interim fix on 2026-09-08 swapped the *leader* blocks instead, which paired each side correctly but left the files named backwards; `identify_sides.py` caught it and the channels were corrected the same day.
  - Diagnose with `python gello_software/scripts/ping_gello.py` (broadcast-pings every FTDI port at every common baudrate; never blocks). `evan` has been added to `dialout` but **needs a re-login** — until then wrap commands in `sg dialout -c "..."`.
- **Motor watchdogs are already set.** All 14 motors read back `timeout=8000` (400 ms) from flash, so the `set_timeout.py` step in the startup sequence is a no-op here — skip it unless a motor is replaced. Read current values without writing via `get_special_message_response(ci, id, "timeout")`.
- **If the arms/GELLO go silent, it is almost always actuator power, not config.** Every adapter (both gs_usb CAN and both FT232H) is USB-bus powered and enumerates fine with nothing alive downstream. Two fast discriminators: `cansend can1 001#11` returning `write: No buffer space available` means no CAN node is ACKing (the TX queue wedges — you must `bash i2rt/scripts/reset_all_can.sh` afterwards to clear it), and a raw-serial broadcast ping returning *zero* bytes (rather than garbage) means the Dynamixel bus is unpowered rather than misconfigured.
- **`hardware_reset()` at most once per launch.** `get_device_ids()` resets every RealSense (drops it off USB for ~2 s). Two resets in quick succession — the launcher's, then a second inside the camera-server child — raced and left the left D405 disconnected (2026-09-08), recovered only by a controller rescan (`echo 1 | sudo tee /sys/bus/pci/devices/0000:00:14.0/remove && echo 1 | sudo tee /sys/bus/pci/rescan`). The server now enumerates with `list_device_ids()` (no reset); `get_device_ids(reset=False)` elsewhere. `_start_pipeline` retries a transient "Device disconnected" 5x before failing.
- **`pipeline.start()` on a device that is not enumerated blocks ~5 s holding the GIL** ("Failed to reconnect: No device connected5000"). In the camera server that froze publishing for ALL cameras (2026-09-08). `RealSenseCamera` only reconnects after `query_devices()` shows the serial again; a stalled camera is served from its last frame and flagged `stale` so the loop keeps running. `scripts/test_camera_drop.py` validates this live.
- **A stale camera server may be holding all 3 cameras.** PID 124458 (`/home/evan/i2rt/.venv/bin/python examples/yam/camera_server.py --config /home/evan/carter/configs/yam_left.yaml`) has been running since Aug 16 and binds ZMQ `:5555`/`:5556`. It blocks any in-process RealSense open (data collection, replay). Teleop is unaffected — `launch_yaml.py` never opens cameras.
- Other config files in `gello_software/configs/` (e.g. `yam_passive.yaml`, `yam_active.yaml`) still use legacy `can_left`/`can_right` names — they are not the active configs and may be stale.

## Standard startup sequence (every fresh boot / replug)

```bash
conda activate yam
bash i2rt/scripts/reset_all_can.sh   # bash, not sh: the script uses [[ ]]
python i2rt/i2rt/motor_config_tool/set_timeout.py --channel can1 --timeout   # left
python i2rt/i2rt/motor_config_tool/set_timeout.py --channel can0 --timeout   # right
# Only if using the linear gripper at full grip and the gripper drifted on power-cycle:
python i2rt/i2rt/motor_config_tool/set_zero.py --channel=can1 --motor_id=7   # left
python i2rt/i2rt/motor_config_tool/set_zero.py --channel=can0 --motor_id=7   # right
```

`set_timeout.py --timeout` sets the motor watchdog to **400ms** (writes `8000`; 8000 × 0.05ms = 400ms) for motors 1–7 and saves it to flash (persists across power cycles). This is the desired state: on `ctrl+C` the command stream stops and the motors auto-de-energize (damping) ~400ms later — the arm safely powers down (LED green→red) instead of holding torque and forcing a physical power-cut. Do **not** use the no-flag form (`set_timeout.py` without `--timeout`), which writes `0` = watchdog disabled — that leaves the arm energized after exit. 400ms does not cause mid-run collapse: the CAN command stream is driven by a dedicated 250Hz background thread (`i2rt/i2rt/robots/motor_chain_robot.py:start_server`), giving ~100× margin; the watchdog only fires when the stream genuinely stops (i.e. on shutdown). The old "default timeout is too short / causes collapse" note was wrong — 400ms is the factory default and is fine for teleop/eval. **One caveat:** that 250Hz thread is pure Python, so a multi-second GIL-holding operation on the main thread *while the motors are live* will starve it and trip the watchdog — symptom is both buses reporting `loss communication` at the same instant. The MolmoAct local model load is exactly such an operation, which is why `launch_yaml_eval_molmoact.py` loads the policy (`_build_policy`) **before** `_build_env` energizes the motors. Keep any heavy/blocking init ahead of robot construction.

## Key entry points (`gello_software/experiments/`)

- `launch_yaml.py` — bimanual teleop.
- `launch_yaml_collect_data.py` — teleop + data collection + auto-convert/upload pipeline.
- `launch_yaml_eval.py` — eval for `dp` (DiffusionPolicy) or `pi05` (PI05Policy); selected via `configs/yam_left.yaml: policy.type`.
- `launch_yaml_eval_molmoact.py` — eval against the MolmoAct-v2 policy. `eval.mode` in `yam_left.yaml` picks `local` (in-process `MolmoActLocal` loading the HF snapshot via transformers; needs ~10–14 GB VRAM at bf16) or `server` (HTTP POST to a remote FastAPI server at `eval.molmoact_server`; accepts a full URL or bare `host:port`). Session-based: `-n N` runs N rollouts, arm interpolates to `agent.start_joints` between rollouts before each instruction prompt (Enter reuses last). A live 3-pane cv2 view shows LEFT / FRONT / RIGHT; press `y`/`n`/`q` in that window to end + label, or let it time out for a stdin prompt. When the camera server is on (default) the viewer is driven by a daemon thread subscribed to the PUB stream, so it keeps repainting through `policy.inference()`. Saves PNG + `episode.h5` per rollout under `{base_dir}/data/{task_directory}/{eval,success,failure}/...` (DROID-style), then batch-converts labeled rollouts to a LeRobot v3.0 dataset under `eval_lerobot_v30/{session_ts}/` at end-of-session. Helpers live in `gello/utils/eval_utils.py`. End-user walkthrough: `docs/grasp_lab_eval.md`.
- `launch_yaml_replay.py`, `launch_yaml_open_loop.py`, `launch_yaml_molmoact_open_loop.py` — replay / open-loop testing from collected JSON episodes.
- `reset_to_home.py` — drive the arm(s) to `agent.start_joints` (home pose) and exit. Builds the robot straight from config (no cameras / GELLO leader / policy) and interpolates via `move_to_start_position`. Bimanual by default; `--left-only` for a single arm. Motors must already be live (run the startup sequence first), or the arm sags mid-move. Run as `python -m experiments.reset_to_home` from `gello_software/`.

All launchers take `--left_config_path` and `--right_config_path`; most config knobs (cameras, storage, lerobot conversion, policy) live in `configs/yam_left.yaml` — `yam_right.yaml` mainly carries the right-arm robot/agent block.

## Camera server (eval-only)

`gello_software/gello/cameras/camera_server.py` runs the three RealSense pipelines in a long-lived process and serves the latest frames over ZMQ (REP on `:5555`, PUB on `:5556`). `camera_client.py` exposes `CameraClient` (REQ-side wrapper the policy uses for on-demand obs) and `CameraSubscriber` (SUB-side, drained by the `LiveCameraView` render thread). `RobotEnv` accepts a `camera_client=` kwarg and a `step_command_only(joints)` so sub-step interpolation no longer reads cameras.

Why: in the old path `dynamic_smoothing` re-read all 3 cameras on every interpolation tick (up to 100× per outer step). With the server architecture cameras stay warm across sessions and are sampled only when the policy actually needs an obs; the cv2 viewer subscribes to the PUB stream from a daemon thread so it keeps painting through `policy.inference()`.

Default-on for the MolmoAct eval launcher via `eval.camera_server.enabled: true` in `yam_left.yaml`. Two-terminal usage:

```bash
# Terminal A: leaves cameras hot across the workstation session
bash gello_software/scripts/start_camera_server.sh    # script hardcodes --config configs/yam_left.yaml
# Terminal B: run eval as usual
```

Set `eval.camera_server.enabled: false` to fall back to the in-process camera path (slower; the viewer freezes during inference). Data collection / replay / open-loop launchers still use the in-process path — the flag is per-launcher.

## Diagnostic scripts added 2026-09-02

Written while bringing this workstation up; all live in `gello_software/scripts/`.

| Script | Use |
|---|---|
| `ping_gello.py` | Broadcast-pings every FTDI port at every common baudrate and prints which Dynamixel IDs answer. Never blocks (unlike `DynamixelDriver`). First thing to run when teleop hangs or a leader is unresponsive. |
| `view_cameras.py` | Live 3-pane cv2 view (FRONT/LEFT/RIGHT) for aiming and identifying cameras. `q` quits, `s` snapshots to `/tmp/`. Quit before teleop/collection — cameras are exclusive-access. |
| `calibrate_gripper.py` | Measures a GELLO trigger's true angular range and prints a ready-to-paste `gripper_config`. Needed because both configs shipped with identical values from another workstation's build. |

## Known-bad failure modes on this box

- **`launch_yaml.py` hangs silently with no output.** [`gello/dynamixel/driver.py:509`](gello_software/gello/dynamixel/driver.py#L509) spins on `while self._joint_angles is None: time.sleep(0.1)` with no timeout. `_initialize_hardware()` counts as success once the *port* opens, so a non-responding servo only prints `Failed to set torque mode for Dynamixel with ID 1` and then blocks forever. Run `scripts/ping_gello.py` to confirm before debugging anything else.
- **`DynamixelDriver` can deadlock on sudo.** `_fix_port_permissions()` shells `sudo chmod 666` with `capture_output=True`, so it blocks forever on the password prompt. Also `use_fake_fallback=True` is the default, so some failure paths silently substitute a fake leader stuck at zeros.
- **`online motors: []` usually means the CAN interface went down, not dead arms.** A USB re-enumeration leaves `can0`/`can1` DOWN; `cansend can1 001#11` then reports `Network is down`. Fix with `bash i2rt/scripts/reset_all_can.sh` — run it immediately before every launch. Distinguish from unpowered arms: unpowered gives `No buffer space available` (nothing ACKs, TX queue wedges) rather than `Network is down`.

## Construction order: arms before leaders and cameras (2026-09-08)

`get_yam_robot()` starts an arm's 250 Hz chain thread and *then* loads the MuJoCo model and
auto-calibrates the gripper on the main thread. If camera capture threads and Dynamixel
readers are already running, that GIL stall exceeds the 400 ms motor watchdog and the arm
goes limp during construction (`fail to communicate with the motor 1` → `loss
communication` → `motor chain is not running`). Reproduced deterministically; every isolated
load was fine, only the combination killed it. `launch_yaml_collect_data.py` therefore builds
the robots first, in a quiet process, then opens leaders and cameras — the same principle the
eval launcher uses for the policy load. Keep it that way in any new launcher. `YAMRobot`
also widens i2rt's 5×10 ms power-on handshake (misses replies under the same contention) and
raises if its chain has died, so frozen joints are never recorded as data.

## Cameras run out of process during collection (2026-09-08)

Measured on this box: the RealSense capture threads hold the GIL 10–20 ms per frame —
specifically `rs.align.process` (align off → GIL wait p95 0.15 ms; on → 9.5 ms). The D435
*must* be aligned (its depth is a different sensor: fx 320 vs 462, 15 mm offset), so the cost
cannot simply be dropped. In the arms' process it jitters i2rt's 250 Hz loops (10 ms CAN
poll timeouts) and, stacked with the pygame dashboard (was 55 ms/tick → 14 Hz loop), trips
the motors' 400 ms watchdog. `launch_yaml_collect_data.py` therefore spawns
`gello.cameras.camera_server` as a child (`collection.camera_mode: subprocess`) and reads
frames through `CameraStreamClient` — a receiver thread on a zero-copy multipart PUB stream
(`--pub-format multipart`), so the loop never waits on the server. REQ/REP (`obs2`) is the
fallback. Result: 29.4 Hz, arms' GIL wait p95 2.8 ms, both arms alive with the dashboard. `RobotEnv.Rate`
now schedules on an absolute timeline (exact 30.0 Hz; the old version drifted to 29.2), and the
server publishes event-driven (`--pub-on-new-frame`): 0.5 % duplicated frames. Do **not** publish
on a faster timer — 60 Hz starved the capture threads and raised duplicates to 23 % on the D435.
The dashboard renders at 10 Hz with cv2-resized tiles (~7 ms). Do not add GIL-heavy work
(image encode/resize in pure Python, model loads, MuJoCo) to the collection process while
the arms are live; put it in a child process or before the robots are constructed.

## flex-pi data pipeline (added 2026-09-08)

The team finetunes **flex-pi** (`flex-pi/soft_bag_zipping` is literally this task). Read
`docs/FLEXPI_DATA_SPEC.md` before touching collection or conversion. Summary:

- Target is LeRobot **v2.1** with a `depth_video` extension, 32-D EE state (FK, rot6d =
  first two rows), 640×360, 30 fps, keys `cam_high/cam_left_wrist/cam_right_wrist`.
- `gello/data_utils/data_saver.py` is a streaming recorder: frames go to disk as they
  arrive; **depth (16-bit PNG, native units) and per-camera intrinsics/depth-scale are
  recorded** (`meta.json`). The pre-rewrite saver silently dropped depth at `env.py`.
  D405 depth is 0.1 mm/unit, D435 1 mm/unit — never assume.
- `flexpi_convert.py` (repo root, `yam` env, no lerobot) writes the v2.1 layout with
  pyarrow + PyAV; `scripts/validate_flexpi_dataset.py` diffs it against the cached
  reference `docs/flexpi/soft_bag_zipping.info.json`. `tests/test_flexpi_pipeline.py`
  runs the whole chain on synthetic frames.
- EE convention verified against their frame 0: gripper model + `grasp_site`.

## Tests

`gello_software/tests/` has pytest coverage for the camera server (`_snapshot`, `_maybe_heartbeat`, end-to-end REQ/REP wire protocol, stale-frame detection) and the MolmoAct eval launcher (`dynamic_smoothing`, `_park_robot`, `_convert_if_any`, `run_one_rollout`). All tests use inline fakes — no RealSense / no CAN motors required.

```bash
cd gello_software && python -m pytest tests/ -q
```

`tests/conftest.py` puts `experiments/` on `sys.path` so `from molmoact import ...` resolves the same way the launcher resolves it at runtime.

## Conventions / gotchas

- **Don't blindly copy CAN channel names out of `README.md` or `docs/grasp_lab_eval.md`** — both still document the GRASP-lab wiring (`can_leader_l`/`can_follower_r`) and `configs/yam_passive.yaml`/`yam_active.yaml` use the older `can_left`/`can_right`. On *this* machine the only real interfaces are `can0`/`can1`; mirror what's actually in `yam_left.yaml`/`yam_right.yaml`.
- The data-collection keypad (`s` start / `a` save+end / `b` discard+end) requires keyboard focus on the color pad window. `ctrl+c` only does cleanup — it skips the convert/upload pipeline.
- For `launch_yaml_eval_molmoact.py`, `ctrl+c` IS handled gracefully: the in-progress rollout is flushed to `eval/{timestamp}/` with an `err.md` marker, and the LeRobot conversion still runs over rollouts already labeled in this session.
- Conversion (`molmoact_to_lerobot_v30.py`) defaults to reading `gello_software/configs/yam_left.yaml` for `data_dir`, `output_dir`, and upload settings; CLI flags override.
- Setup order recommended by upstream: install `i2rt` first, then `gello_software`, then `lerobot`. Each subdir has its own README/requirements.

## When asked to add/change a runtime command

Cross-check three places before suggesting it works:
1. The actual CAN interface names (`ip link show | grep can`).
2. The `channel:` field in `configs/yam_left.yaml` and `configs/yam_right.yaml`.
3. That `ai2_yam` is activated.
