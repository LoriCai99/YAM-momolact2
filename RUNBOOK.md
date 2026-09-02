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
| Left arm | CAN **`can1`** |
| Right arm | CAN **`can0`** |
| Left GELLO | `FTAO9WPU` → Dynamixel IDs `1–7`, 57600 baud |
| Right GELLO | `FTAO9WCV` → Dynamixel IDs `8–14`, 57600 baud |
| Front/top camera | D435 `922612071156` |
| Left camera | D405 `335122270697` |
| Right camera | D405 `218622275075` |
| Data output | `/home/evan/yam_data/<task_directory>/` |

`i2rt` resolves to an **editable install at `/home/evan/i2rt` (v1.2.4)**, *not*
this repo's `i2rt/` subdirectory. `import i2rt` never reads the in-repo tree.

---

## 2. Known blockers — read before you start

**① The D435 is on USB 2.1, which breaks data collection.**
At USB 2.1 the D435 does not offer **color 640×360**, which is exactly what
`gello/cameras/realsense_camera.py` requests. Collection dies with:

```
RuntimeError: Couldn't resolve requests
```

*Fix:* a genuine **USB 3 cable into a USB 3 port**. Many cables that fit are USB 2
only. Avoid the ASMedia controller at `05:00.0` — it dropped both D405s
mid-session once. Confirm it reports `3.2`:

```bash
python -c "
import pyrealsense2 as rs
for d in rs.context().query_devices():
    print(d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.usb_type_descriptor))
"
```

Teleop is unaffected — it never opens cameras.

**② The right GELLO gripper is mis-calibrated.**
It rests at 113.03°, past its configured closed bound of 108.37°, so the
normalized command saturates above 1.0 and the right gripper sits pinned closed.
Both configs shipped with *identical* `gripper_config` values copied from another
workstation's GELLO build. Fix with §6.4. Its spring return is also mechanically
weak — that part is not a software problem.

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
python i2rt/i2rt/motor_config_tool/ping_motors.py --channel can1   # expect [1..7]
python i2rt/i2rt/motor_config_tool/ping_motors.py --channel can0   # expect [1..7]
sg dialout -c "python gello_software/scripts/ping_gello.py"        # expect [1..7] and [8..14]
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

`ctrl+C` to exit — the command stream stops and the 400 ms watchdog
de-energizes both arms (LED green→red). No physical power-cut needed.

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

Hold the home pose at launch, same as teleop. Then, **with keyboard focus on the
colour pad window** (not the terminal — this catches everyone out):

| Key | Action |
|---|---|
| `s` | start recording an episode |
| `a` | end and **save** |
| `b` | end and **discard** |

Episodes land in `/home/evan/yam_data/put_pen_in_bag/` as JSON.

**The "directory already exists — remove it? (y/n)" prompt:** on a first run the
launcher creates the directory then flags its own handiwork; `y` is safe when it
is empty. **Once you have real episodes, answer `n`** — `y` deletes them.

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

### 6.4 `calibrate_gripper.py` — fix a saturating gripper
Work the trigger through its full travel; it prints a `gripper_config` line to
paste into the matching config.

```bash
sg dialout -c "python scripts/calibrate_gripper.py --side right"
```

---

## 7. Downstream: conversion is NOT done on this box

`auto_convert` and `auto_upload` are **false** by design. Raw JSON is the
deliverable; conversion and training happen elsewhere.

This is not merely a preference — it cannot run in the `yam` env:

| Env | `lerobot` | `huggingface_hub` |
|---|---|---|
| `yam` (runs collection) | ✗ absent | 0.36.2 (pinned <1.0 by `transformers` 4.57) |
| `yam_convert` | ✓ 0.5.2 | 1.27.0 (lerobot needs ≥1.0) |

`launch_yaml_collect_data.py:273` shells the converter with `sys.executable` —
the `yam` interpreter. Setting `auto_convert: true` would run a **full
collection session** and then fail at the very end with
`ModuleNotFoundError: No module named 'lerobot.datasets'`. Raw JSON survives, but
you would have wasted the session's tail.

To convert manually:

```bash
/opt/conda/envs/yam_convert/bin/python molmoact_to_lerobot_v30.py \
    --data_dir /home/evan/yam_data/put_pen_in_bag \
    --output_dir /home/evan/yam_data/put_pen_in_bag_lerobot_v30
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `online motors: []` on both buses | CAN interfaces went DOWN after a USB re-enumeration | `sudo bash i2rt/scripts/reset_all_can.sh`. Confirm with `cansend can1 001#11` → `Network is down` |
| `cansend` → `No buffer space available` | No node is ACKing: arms genuinely unpowered or unplugged. TX queue wedges | Power the arms, then reset CAN to clear the wedge |
| **Teleop hangs, no output** | `driver.py:509` spins on `while self._joint_angles is None` with no timeout. A non-responding servo only prints `Failed to set torque mode…` then blocks forever | `scripts/ping_gello.py`. Silence at every baudrate ⇒ the Dynamixel **power rail** is off (USB enumeration proves nothing — the U2D2 is bus-powered) |
| `RuntimeError: Couldn't resolve requests` | D435 on USB 2.1 lacks color 640×360 | §2① — USB 3 cable |
| `Device or resource busy` on a camera | Something else holds it | `ps aux \| grep camera_server`; `sudo fuser -v /dev/video*` |
| A D405 vanishes from USB | ASMedia controller `05:00.0` dropping devices | `echo 1 \| sudo tee /sys/bus/pci/devices/0000:05:00.0/remove && echo 1 \| sudo tee /sys/bus/pci/rescan` |
| Permission denied on `/dev/ttyUSB*` | `dialout` not active in this shell | `sg dialout -c "..."`, or log out and back in |
| Session dies with `loss communication` | `enable_auto_recovery` defaults to False (fail-fast) | Restart. Already-saved episodes are intact |
| Right gripper stuck closed | Gripper calibration saturating | §6.4 |

---

## 9. Tests

```bash
cd gello_software
python -m pytest tests/ -q --ignore=tests/test_launch_yaml_eval_molmoact.py
```

10 pass. The excluded module imports `lerobot`, which is absent from `yam` by
design (§7); it is not a broken test.
