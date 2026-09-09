# Bimanual YAM — Workstation Runbook

Everything needed to run **teleoperation** and **data collection** on the bimanual
YAM rig with GELLO leaders, on `prior-workstation-4`.

Hardware here was verified end-to-end on **2026-09-02**. If you are on a
different machine, the serials, CAN names and ports below will not match — treat
this file as machine-specific and re-derive them with the diagnostics in §6.

> **`README.md` describes a different workstation.** Its CAN names
> (`can_leader_l` / `can_follower_r`) and the `ai2_yam` conda env do not exist
> here. Follow *this* file. `CLAUDE.md` has the deeper engineering notes.

---

## 1. This workstation at a glance

| Thing | Value |
|---|---|
| Conda env | **`yam`** (Python 3.12) — *not* `ai2_yam` |
| Conversion env | `yam_convert` (has `lerobot`; `yam` does not — see §7) |
| Left arm | CAN **`can0`** |
| Right arm | CAN **`can1`** |
| Left GELLO | `FTAO9WPU` → Dynamixel IDs `1–7`, 57600 baud |
| Right GELLO | `FTAO9WCV` → Dynamixel IDs `8–14`, 57600 baud |
| Front/top camera | D435 `922612071156` |
| Left camera | D405 `353322270868` (replaced 2026-09-08) |
| Right camera | D405 `218622275075` |
| Data output | `/home/evan/yam_data/<task_directory>/` |

`i2rt` resolves to an **editable install at `/home/evan/i2rt` (v1.2.4)**, *not*
this repo's `i2rt/` subdirectory. `import i2rt` never reads the in-repo tree.

---

## 2. Known blockers — read before you start

**① The front D435 must link at USB 3 — check before every session.**
At USB 2.1 the D435 does not offer **color 640×360**, which is exactly what
`gello/cameras/realsense_camera.py` requests. Collection dies with:

```
RuntimeError: Couldn't resolve requests
```

Resolved 2026-09-08: after a week of diagnosis it was the **USB-C cable** (no
SuperSpeed wires — it fits and powers the camera, and looks identical to a USB 3
lead). Not the port, not the camera. If it recurs, swap the cable first. Avoid the
ASMedia controller at `05:00.0` — it dropped both D405s mid-session once. Confirm
every camera reports `3.2`:

```bash
python -c "
import pyrealsense2 as rs
for d in rs.context().query_devices():
    print(d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.usb_type_descriptor))
"
```

Teleop is unaffected — it never opens cameras.

**② RESOLVED 2026-09-08 — the left D405 was replaced.** The original unit
(`335122270697`) disconnected from USB 60+ times, enumerated without streaming
(`xioctl VIDIOC_S_FMT errno=5`), and the fault followed the camera through a
cable and port swap. Eighteen episodes (000008–000044, see `inspect_episodes.py
--summary`) have a frozen left-wrist view and must not be used. The replacement
D405 `353322270868` is in `configs/yam_left.yaml`; `check_cameras.py` reports
READY. If a camera is ever swapped again: put its serial under `left_camera`,
run `python scripts/check_cameras.py` (must be READY) and
`python scripts/test_camera_drop.py --reset-serial <serial>` (must PASS), and
check its aim with `python scripts/view_cameras.py`.

**③ The RIGHT GELLO trigger does not spring back.** Calibration is done (both
triggers measured 2026-09-08), but the right trigger stays wherever it is left, so
the follower gripper idles partly closed unless the operator pushes it open. Fix
the spring/elastic; software cannot.

---

## 3. One-time setup

Already done on this machine; listed for rebuilds.

```bash
conda activate yam
pip install -e gello_software --no-deps
pip install dynamixel-sdk "omegaconf==2.3.0" tyro pyzmq opencv-python pygame \
            h5py json-numpy termcolor Pillow numpy-quaternion pyquaternion
sudo usermod -aG dialout $USER     # then LOG OUT AND BACK IN
```

Until you have logged out and back in, wrap every command that touches the GELLO
leaders in `sg dialout -c "..."`. The examples below do this; drop it once your
shell reports `dialout` in `id -nG`.

Motor watchdogs are **already set** — all 14 motors read `timeout=8000` (400 ms)
from flash. Skip `set_timeout.py` unless you replace a motor.

---

## 4. Every session — startup

```bash
cd "/home/evan/Lori-momolact2 setup/YAM-momolact2"
conda activate yam
sudo bash i2rt/scripts/reset_all_can.sh      # bash, NOT sh (script uses [[ ]])
```

**Run the CAN reset before every launch.** A USB re-enumeration silently leaves
`can0`/`can1` DOWN, and the symptom (`online motors: []`) looks exactly like
unpowered arms. See §8.

Then confirm all four subsystems answer:

```bash
python i2rt/i2rt/motor_config_tool/ping_motors.py --channel can0   # left,  expect [1..7]
python i2rt/i2rt/motor_config_tool/ping_motors.py --channel can1   # right, expect [1..7]
python gello_software/scripts/ping_gello.py        # expect WPU=[1..7] (left), WCV=[8..14] (right)
```

---

## 5. Running

### 5.1 Teleoperation

```bash
cd gello_software
sg dialout -c "python experiments/launch_yaml.py \
    --left_config_path=configs/yam_left.yaml \
    --right_config_path=configs/yam_right.yaml"
```

> **Hold both GELLO leaders in the home pose as it starts** — arm extended
> forward, links straight and level, per `gello_software/imgs/yam_default.JPG`.
>
> This is not cosmetic. The launcher re-solves `joint_offsets` at every startup
> by searching ±8π for the offset mapping the leader's *current* reading onto
> `start_joints`. **Whatever pose you hold becomes the defined zero.** Launch
> with a slumped leader and the leader↔follower correspondence is silently wrong.

`ctrl+C` to exit: the arms first return **gently to the start pose** (interpolated,
robot-only — it works even when the cameras are what failed), then the process
exits and the 400 ms watchdog de-energizes both arms (LED green→red). The start
pose is arm-extended, so expect the arms to sag when torque drops. No physical
power-cut needed.

### 5.2 Data collection

Set the task first in `configs/yam_left.yaml`:

```yaml
storage:
  episodes: 100                     # max episode index; loop ends here
  base_dir: "/home/evan/yam_data"
  task_directory: "put_pen_in_bag"  # one per task
  language_instruction: "unzip the bag, pick up pens on the desk, put them into bag, zip the bag to close"
```

```bash
sudo bash ../i2rt/scripts/reset_all_can.sh
sg dialout -c "python experiments/launch_yaml_collect_data.py \
    --left_config_path=configs/yam_left.yaml \
    --right_config_path=configs/yam_right.yaml"
```

**What you will see at startup, in this order:** camera pre-flight table → `Starting
camera server: …` (a child process that owns the RealSense devices; its log path is
printed) → the two arms energise → the two leaders → `Camera client connected` →
three `camera … depth_scale=…` lines → move to home → `Start 🚀🚀🚀`. The colour pad
is grey until `Start`; that is normal.

Why the cameras run in a child process: `rs.align` holds Python's GIL 10–20 ms per
frame. In the same process as the arms' 250 Hz control loops that jittered every
torque update (jerky arms) and, with the dashboard, tripped the motors' 400 ms
watchdog. Measured after the change: loop 29.4 Hz, arms' GIL wait p95 2.8 ms. If a
camera server is already running on the configured endpoint (e.g. one you started
for eval) the launcher reuses it and does not hardware-reset the cameras.

**Debug first, save nothing:** add `--dry_run` to the command above. Everything runs
exactly as in a real session but episodes go to a temp directory that is deleted on
exit, the real data directory is never touched or prompted about, and no
post-collection pipeline runs. Use it after any hardware or code change.

Hold the home pose at launch, same as teleop. Then, **with keyboard focus on the
colour pad window** (not the terminal — this catches everyone out):

| Key | Action |
|---|---|
| **Enter** | start recording — the pad counts **3-2-1** (get both hands on the leaders), then shows RECORDING |
| `s` | end and **save** — pad shows a green `SAVING episode N` banner |
| `d` | end and **discard** — pad shows a red `DISCARDED episode N` banner |

**A dead camera blocks Enter.** If any camera is stale (no new frame for 0.5 s)
the waiting screen says `!! CAMERA DEAD: <cam> -- Enter disabled` and the
terminal logs `CAMERA DEAD/STALE` every 2 s; Enter is refused until it recovers.
If a camera dies *during* a take the pad flashes a red `CAMERA STALE ... press D`
banner twice a second and the terminal logs it — press `d`. (Added 2026-09-08
after episodes 000035–000044 were recorded with the left camera frozen for 100%
of every frame and only a small status line said so.)

**Headless (no pad):** add `--no_dashboard` and type the keys in the terminal —
**Enter** start, `s` save, `d` discard (single keypress, no Enter needed;
the terminal is switched to cbreak mode and restored on exit). Status and
stale warnings print to the terminal. Use it to rule the pad out; measurements show it is not
the cause of camera drops (those are USB-level disconnects), but it costs
nothing to run without it.

On Ctrl-C or a crash: an episode you already pressed `s` on is finalised before
exit; the take in progress (never saved) is discarded automatically.

Episodes land in `/home/evan/yam_data/put_pen_in_bag/NNNNNN/`: per-frame JSON,
`meta.json` (intrinsics, depth scale, roles), `*_rgb/` JPEG q95 and `*_depth/`
16-bit PNG for all three cameras. Depth is **required** by flex-pi — check that the
launcher printed a `depth_scale` for every camera at startup and that
`storage.save_depth` is true.

**Resuming into an existing task directory:** the launcher counts the complete
episodes, removes leftovers from crashed runs (dirs with frames but no JSON), and
asks once — **Enter appends as the next index** (inferred, e.g. `000004`); a number
starts at that index (existing ones are refused); typing `delete` then `yes` wipes
the directory. There is no longer a `y/n` where `y` deletes real data.

---

## 6. Diagnostics

All in `gello_software/scripts/`. Run from `gello_software/`.

### 6.1 `ping_gello.py` — are the leaders alive?
Broadcast-pings every FTDI port at every common baudrate; never blocks.
**Run this first whenever teleop hangs.**

```bash
sg dialout -c "python scripts/ping_gello.py"
```

### 6.2 `view_cameras.py` — live 3-pane view
For aiming cameras and confirming which is which. `q` quits, `s` snapshots to
`/tmp/`. **Quit it before teleop/collection** — cameras are exclusive-access.

```bash
python scripts/view_cameras.py
```

### 6.3 `ping_motors.py` — are the arms alive?
```bash
python ../i2rt/i2rt/motor_config_tool/ping_motors.py --channel can1
```

### 6.4 `identify_sides.py` — which leader / which bus is on which side
Moves nothing; you move the right leader and right arm by hand when prompted.
Prints CORRECT / CROSSED / FILES NAMED BACKWARDS and the fix. **Run this after any
re-cabling** — the "named backwards" case feels perfect in teleop but swaps
`left_joint`/`right_joint` in every saved episode.

```bash
python scripts/identify_sides.py
```

### 6.5 `calibrate_gripper.py` — fix a saturating gripper
Work the trigger through its full travel; it prints a `gripper_config` line to
paste into the matching config.

```bash
python scripts/calibrate_gripper.py --side right
```

---

## 7. Conversion to the flex-pi format

Raw episodes are converted **here**, in the `yam` env, with no lerobot dependency
(`docs/FLEXPI_DATA_SPEC.md` has the full spec and the reasoning):

```bash
cd "/home/evan/Lori-momolact2 setup/YAM-momolact2"
conda activate yam
python flexpi_convert.py                        # reads storage:/flexpi: from configs/yam_left.yaml
python gello_software/scripts/validate_flexpi_dataset.py /home/evan/yam_data/put_pen_in_bag_flexpi_v21
```

Output is LeRobot **v2.1** in exactly the layout of `flex-pi/soft_bag_zipping`:
640×360 h264 RGB + FFV1 uint16-mm depth for `cam_high`/`cam_left_wrist`/
`cam_right_wrist`, 32-D end-effector state/action (FK via i2rt), intrinsics.
The validator must print **PASS** before data is handed to the training team.

The older `molmoact_to_lerobot_v30.py` (v3.0, RGB only, 14-D joints) still works
on the same raw episodes but needs the `yam_convert` env; it is not what flex-pi
consumes. `lerobot.auto_convert` stays false.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Converter: `no depth_scale_m_per_unit for <cam>` | Episode has no `meta.json` (recorded before the depth rewrite) | `--depth_scale front=0.001 --depth_scale left=0.0001 --depth_scale right=0.0001` (D435 = 1 mm, D405 = 0.1 mm per unit) |
| Converter: `has no depth frames` | Collected with `storage.save_depth: false` | Recollect; flex-pi needs depth |
| Validator FAIL on `feature ...` | info.json drifted from the reference | Don't hand-edit info.json; rerun `flexpi_convert.py` |
| `online motors: []` on both buses | CAN interfaces went DOWN after a USB re-enumeration | `sudo bash i2rt/scripts/reset_all_can.sh`. Confirm with `cansend can1 001#11` → `Network is down` |
| `cansend` → `No buffer space available` | No node is ACKing: arms genuinely unpowered or unplugged. TX queue wedges | Power the arms, then reset CAN to clear the wedge |
| **Teleop hangs, no output** | `driver.py:509` spins on `while self._joint_angles is None` with no timeout. A non-responding servo only prints `Failed to set torque mode…` then blocks forever | `scripts/ping_gello.py`. Silence at every baudrate ⇒ the Dynamixel **power rail** is off (USB enumeration proves nothing — the U2D2 is bus-powered) |
| Collection prints `Camera pre-flight … FAIL: USB 2.1 link` and exits (code 2) | That camera cannot serve colour+depth 640×360@30 on a USB 2 link — a D435 at USB 2.1 does not offer the mode at all | §2① — get it onto USB 3 (or use any other D4xx). Not a config error; nothing was opened or energized |
| `Device or resource busy` on a camera | Something else holds it | `ps aux \| grep camera_server`; `sudo fuser -v /dev/video*` |
| A D405 vanishes from USB | ASMedia controller `05:00.0` dropping devices | `echo 1 \| sudo tee /sys/bus/pci/devices/0000:05:00.0/remove && echo 1 \| sudo tee /sys/bus/pci/rescan` |
| Permission denied on `/dev/ttyUSB*` | `dialout` not active in this shell | `sg dialout -c "..."`, or log out and back in |
| Both arms go limp right after launch; log shows `fail to communicate with the motor 1` / `loss communication` | Arms constructed while cameras + leaders already load the process (GIL stall > 400 ms watchdog) | Fixed in the launcher (arms built first). If it recurs, check nothing else heavy runs in-process before the robots |
| Session dies with `loss communication` | `enable_auto_recovery` defaults to False (fail-fast) | Restart. Already-saved episodes are intact |
| `camera server exited with code …` / `did not answer within 60s` | The child camera server failed to start (cameras busy, USB, bad config) | Read the log path printed at startup; `python scripts/check_cameras.py`; `pkill -f camera_server` if a stale one holds the devices |
| Pad flashes red `CAMERA STALE: <cam> -- press D` mid-episode (terminal logs it too) | That camera stopped delivering frames (USB drop). The loop keeps running; the other cameras keep flowing | Finish the take with `d`. A camera that comes back recovers on its own (~2 s). Frames with a stale camera are logged per row and `inspect_episodes.py` WARNs |
| `camera server stopped publishing: no frame set for 3.0s` | The whole camera server went silent (process died, or every camera gone) | Read `/tmp/yam_camera_server_<pid>.log` and the session log `/tmp/yam_collect_<pid>.log` |
| A camera enumerates but streams **0 frames**, or `dmesg` shows repeated `usb 2-N: USB disconnect` for it | Cable / port / camera fault (left D405 on 2026-09-08: 63 disconnects in a day, then 0 frames while enumerated) | Swap its cable with a known-good one, then another port; if it still does not stream, replace the camera. `python scripts/test_camera_drop.py --reset-serial <serial>` validates recovery |
| Arms jerky during collection | Loop below 30 Hz or GIL contention in the arms' process | Confirm `collection.camera_mode: subprocess` and that the dashboard is not modified to render every tick |
| Right gripper stuck closed | Gripper calibration saturating (leader `FTAO9WCV`) | §6.5 |

---

## 9. Tests

```bash
cd gello_software
python -m pytest tests/ -q --ignore=tests/test_launch_yaml_eval_molmoact.py
```

10 pass. The excluded module imports `lerobot`, which is absent from `yam` by
design (§7); it is not a broken test.
