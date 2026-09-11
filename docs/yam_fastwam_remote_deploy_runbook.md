# Runbook: running FastWAM on the GPU server and deploying on the YAM arms here

Written 2026-09-09 for the UW rig (`prior-workstation-4`, 172.16.120.60).
Model runs remotely on the GPU server at 10.19.141.185; the robot loop runs on this
workstation. Read [fastwam_deploy_flexi_pi_july.md](fastwam_deploy_flexi_pi_july.md) for the
original per-checkpoint sweep doc this is derived from.

## 1. The shape of it

Three processes, two machines:

| Where | What | Command |
|---|---|---|
| GPU server 10.19.141.185 | **Policy server.** Loads the checkpoint, prewarms, listens on port 8000. Takes an observation, returns a 32-step action chunk. | `bash ~/venvs/fastwam_serve/serve_pen.sh` (as `rseyam`) |
| Workstation, terminal 1 | **SSH tunnel.** Port 8000 on the server is firewalled from the VPN, so the bridge talks to `localhost:8000` and the tunnel forwards it. | `bash /home/evan/projects/yam_deploy/tunnel.sh` |
| Workstation, terminal 2 | **Raiden bridge.** Captures 3 RealSense colour+depth frames and 14 joint angles, posts them with the prompt, converts the returned end-effector poses to joints with IK, streams to the arms at 30 Hz over CAN. | `bash /home/evan/projects/yam_deploy/run_bridge.sh` |

Everything on the server side sits behind UW's Husky OnNet VPN, so the VPN comes first.

Latency budget: ping to the server is 80–150 ms. Action-only inference is well under the
1067 ms of runway that chunk 32 gives at 30 Hz. Full-joint mode is not: measure it before
using it (§6).

## 2. What is installed where (one-time setup, already done)

**Workstation**

| Path | Purpose |
|---|---|
| `/home/evan/vpn.sh` | Opens the Husky OnNet portal in Firefox (F5 client login). |
| `/opt/f5/vpn/` | F5 BIG-IP Edge Client for Linux (`f5vpn`), installed from the portal. OpenConnect is also installed but cannot pass UW's SAML login; do not use it. |
| `/home/evan/projects/YAM_robot/` | Raiden bridge workspace (clone of `geyan21/YAM_robot`) with a `.venv` (Python 3.11). Two local edits, see §8: a RealSense capture path (the upstream copy is ZED-only) and environment-configurable CAN interface names. |
| `/home/evan/projects/HVLA/world_model/FastWAM/` | Sparse checkout of `geyan21/HVLA` branch `fastwam-sagemaker-anydata`. The bridge class lives in `experiments/yam/fastwam_policy/`. Installed into the venv without its pinned deps (its numpy pin conflicts with i2rt). |
| `/home/evan/projects/yam_deploy/` | Rig-specific files: `camera_config.json` (three RealSense serials), `calibration_results.json` (identity left/right base transform, copied from upstream), `activate_bridge.sh`, `tunnel.sh`, `run_bridge.sh`, `check_cameras.py`. |
| `~/.ssh/id_ed25519` | Key installed on the server for `rseyam`. |

**GPU server (as `rseyam`)**

| Path | Purpose |
|---|---|
| `~/projects/HVLA_deploy/world_model/FastWAM/` | Copy of the same September branch (`BRANCH_SNAPSHOT.txt` records the commit). Separate from `~/projects/HVLA`, which has uncommitted edits and is left alone. |
| `~/venvs/fastwam_serve/` | Serving venv layered on the `world` conda env's torch 2.7.1+cu128. `serve_pen.sh` is the boot script. |
| `~/model_ckpts/0907/.../0904_mm2yam_uflex_32d_rel_pm_ds2kf_anydata_bs9_8node_1p5ep_ga4_scratch/` | The checkpoint: `config.yaml`, `dataset_stats.json`, `checkpoints/weights/step_047000.pt`. |
| `~/projects/HVLA/flex-pi/checkpoints/` | Wan2.2-TI2V-5B base weights (`DIFFSYNTH_MODEL_BASE_PATH`). |
| `~/.cache/huggingface/hub/` | DINOv3 backbone (`timm/vit_base_patch16_dinov3.lvd1689m`). Do not override `HF_HOME`. |
| `~/logs/serve_pen.log` | Server log. |
| `~/.s3cfg` | Kopah S3 credentials (bucket `s3://rselab`). |

Disk on the server is 99 % full (about 60 GB free). Each checkpoint is 12 GB. Delete old
steps before pulling new ones.

## 3. Every session: connect

**VPN**

```bash
bash /home/evan/vpn.sh
```

Firefox opens `https://huskyonnet.uw.edu`. Log in with NetID `lcai9` and Duo, click the
Husky OnNet entry, let Firefox open it with "F5 VPN". A small F5 window shows Connected.
Leave it open. Verify:

```bash
ping -c2 10.19.141.185
```

The VPN is split-tunnel: only UW ranges go through it, the robot LAN and CAN are untouched.

**SSH to the server** (key-based, no password):

```bash
ssh rseyam@10.19.141.185
```

## 4. Every session: boot the model

On the server:

```bash
ssh rseyam@10.19.141.185
bash ~/venvs/fastwam_serve/serve_pen.sh
```

Boot takes 2–4 minutes: weight load, `torch.compile warmup done in ~43 s`, then one prewarm
inference. Ready when the log shows:

```
INFO:websockets.server:server listening on 0.0.0.0:8000
```

Confirm from a second server shell:

```bash
curl http://localhost:8000/healthz     # -> OK
tail -f ~/logs/serve_pen.log
```

Two checks the sweep doc insists on, both visible in the log:

- The `YamFastWAMPolicy ready | ckpt=... stats=...` line must name the `dataset_stats.json`
  next to the checkpoint, not some other stats file. Wrong stats gives wrong action scale
  with no error.
- `horizon=32`. This checkpoint is the standard shape (33 frames, horizon 32).

One boot per session. Stop with Ctrl+C in the server terminal; wait until port 8000 is free
and VRAM drops before booting again:

```bash
ss -ltn | grep 8000; nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

**Changing checkpoint or prompt.** Edit the `CKPT=` and `PROMPT=` lines in
`~/venvs/fastwam_serve/serve_pen.sh`. The server reads the checkpoint's own `config.yaml`
for layout, frame count and DINO stride, so standard-shape checkpoints need no other change.
Only the `nf49` checkpoints (#19/#20 in the sweep doc) need `--action-horizon` dropped and
the bridge chunk set to 48.

**Pulling a new checkpoint from Kopah** (on the server):

```bash
RUN=<task>/<corpus_tag>/<run_name>            # path under checkpoints/<date>/
SRC=s3://rselab/robotwin/checkpoints/<date>/$RUN
DST=~/model_ckpts/<date>/$RUN
mkdir -p $DST/checkpoints/weights
s3cmd ls $SRC/checkpoints/weights/            # pick the step
s3cmd get $SRC/config.yaml $SRC/dataset_stats.json $DST/
s3cmd get $SRC/checkpoints/weights/step_XXXXXX.pt $DST/checkpoints/weights/
```

About 30 MB/s, so 12 GB takes 6–7 minutes.

### 4.1 Checkpoint 0910 `pen3_ft_mm2yam` step 022000 (added 2026-09-10)

Finetuned from the 0904 mm2yam checkpoint on flex-pi's 300 pen episodes **plus this rig's
own RealSense data** (`put_pen_in_bag_flexpi_v21_Ai2_20260908_allintra` and
`..._20260909_allintra`), so no camera domain gap. Standard shape (33 frames, horizon 32,
ds2kf, robotwin_uniform14): same server flags, same prompt, only the path changes.

- Kopah: `s3://rselab/robotwin/checkpoints/0910/yam_realworld_yam_unified_flex_3cam_32d_rel_1e-4/pen3_ft_mm2yam/0910_yam_pen3_v3ep300_uflex_32d_rel_pm_ds2kf_ft_mm2yam53838_bs9_2node_10ep_ga1/`
- Server: `~/model_ckpts/0910/<same run path>/` (`config.yaml`, `dataset_stats.json`, `checkpoints/weights/step_022000.pt`, 12 GB)
- Boot script: `bash ~/venvs/fastwam_serve/serve_pen_0910.sh` (copy of `serve_pen.sh` with the new `CKPT=`).
  Stop the previous server first (Ctrl+C in its terminal, or `pkill -f serve_yam_fastwam`), wait for
  port 8000 and VRAM to free, then boot. Confirm the `YamFastWAMPolicy ready | ... stats=` line names the
  0910 `dataset_stats.json`.
- Bridge side is unchanged (`run_bridge.sh`).

## 5. Every session: run on the arms

Same physical prep as a collection session: bag and pens on the table, both arms powered
(LEDs on), GELLO leaders not needed, nobody's hands in the workspace. Do not run the
collection launcher or `view_cameras.py` at the same time; the cameras are exclusive.

**Terminal 1, tunnel** (needs the VPN up):

```bash
bash /home/evan/projects/yam_deploy/tunnel.sh
```

It prints nothing and just sits there. Check from another shell:
`curl http://localhost:8000/healthz` should print `OK`.

**Terminal 2, cameras and CAN:**

```bash
cd "/home/evan/Lori-momolact2 setup/YAM-momolact2"
sudo bash i2rt/scripts/reset_all_can.sh
source /home/evan/projects/yam_deploy/activate_bridge.sh
python /home/evan/projects/yam_deploy/check_cameras.py
```

All three cameras must print `OK`. `MISSING` means the camera is not on USB (§7). Run this
check **before** launching: the bridge starts its own camera server and the cameras are
exclusive.

**Terminal 2, launch:**

```bash
bash /home/evan/projects/yam_deploy/run_bridge.sh
```

The script (1) starts the out-of-process camera server (`start_cameras.sh`, the repo's
`gello.cameras.camera_server` in the `yam` env, ZMQ :5555/:5556) unless one is already
running, (2) checks the tunnel, then (3) runs `rd infer` with the WebSocket bridge,
`host=localhost`, **`action_source=model_joint` + `--action-type joint`** (the model's joint
targets go straight to the motors, no IK), `use_depth=true`, chunk 32, asynchronous
replanning (`use_rtc=true`, replan every 8 steps), action-only regime, and the pen prompt.
Every run is logged to `logs/bridge_<timestamp>.log`. Knobs, as environment variables in
front of the command: `CHUNK` (32), `RTC` (true), `REPLAN` (8), `MERGE` (4), `ACTION`
(`joint`; `ee_pose` runs raiden's IK, see §7), `MAXD` (joint-delta abort limit, 0.3 rad).

Why these defaults (measured 2026-09-09): with `ee_pose` raiden's IK failed to converge on
most steps and the loop ran at 0.2-0.5 Hz; with `model_joint` and in-process RealSense capture
it ran at ~7 Hz because `rs.align` holds the GIL 10-20 ms per frame per camera; with the
camera server out of process the capture threads never touch the SDK.

Watch the first chunk with a hand on the power switch. Ctrl+C stops the bridge; the motors
de-energise about 400 ms after the command stream stops (the watchdog is set on all 14
motors). The server stays warm; relaunching the bridge takes a couple of seconds.

On the server log, each chunk prints `[YAM-deploy] model.infer_action wall=... ms`. Read that
value over the first few chunks; it plus the network round trip must stay well under 1067 ms.

## 6. Full-joint mode

The doc's second boot per checkpoint. On this branch the regime is chosen by the bridge, not
the server (the server script has no `--infer-joint-*` flags), so no server restart is
needed. `run_bridge.sh` takes a `JOINT` switch (added 2026-09-10):

```bash
JOINT=true bash /home/evan/projects/yam_deploy/run_bridge.sh     # full-joint
bash /home/evan/projects/yam_deploy/run_bridge.sh                # action-only (default)
```

It sets `joint_video/joint_dino/joint_pointmap` in the bridge kwargs together.

The prewarm on the 5090 took 786 ms in this regime. Add 80–150 ms of network and you are
inside the 1067 ms chunk-32 runway with little margin. If the `wall=` line sits near 900 ms,
cut `--num-inference-steps` in `serve_pen.sh` (to 6 or 4) rather than lengthening the chunk.
Never add `--glue-cache`; never point one checkpoint at another's TRT engine (both are
explained in the sweep doc).

## 6.1 Saving the predicted future video (added 2026-09-10)

The server can write the model's imagined future for every chunk: decoded RGB plus DINO-PCA
and pointmap-PCA rows, one MP4 per bridge connection. It is a **server** flag
(`--record-predictions-dir`), so it needs its own boot; it forces the full-joint regime
regardless of the bridge's `JOINT` setting (about 3x slower than action-only, the arms pause
between chunks).

```bash
# server (stop the running server first: Ctrl+C or pkill -f serve_yam_fastwam)
bash ~/venvs/fastwam_serve/serve_pen_0910_record.sh      # serve_pen_0910.sh + --record-predictions-dir ~/predictions --record-predictions-fps 8
```

Then run the bridge as usual. Ctrl+C on the bridge closes the connection and the server
finalises `~/predictions/<timestamp>_<remote>/prediction.mp4`. Pull the videos to the
workstation:

```bash
rsync -av rseyam@10.19.141.185:predictions/ /home/evan/yam_predictions/
```

Knobs (edit `serve_pen_0910_record.sh`): `--record-predictions-fps 32` for real-time-ish
playback (default 8 = slow motion); `--record-predictions-frames-per-chunk 3` to keep only
the frames that actually elapse before the next chunk (seamless timeline instead of the
per-chunk anchor pattern). Incompatible with `--use-pinv-rtc`.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `openconnect` says `Unknown form ID 'hidden_form'` | UW's portal uses SAML web login | Use the F5 client via `vpn.sh`, not OpenConnect. |
| `ping 10.19.141.185` fails | VPN not connected | Check the F5 window; reconnect via `vpn.sh`. |
| `ssh rseyam@...` asks for a password | Key not accepted | Re-run `ssh-copy-id rseyam@10.19.141.185` (needs the rseyam password). |
| `curl localhost:8000/healthz` fails on the workstation | Tunnel not running, or server not booted | Start `tunnel.sh`; check the server log. Port 8000 is never reachable directly. |
| `nc 10.19.141.185 8000` refused | Expected | Firewall. Use the tunnel. |
| `check_cameras.py` says `MISSING` | Camera off USB | Reseat the camera's cable at both ends. If it was on the ASMedia controller, rescan: `echo 1 \| sudo tee /sys/bus/pci/devices/0000:05:00.0/remove && echo 1 \| sudo tee /sys/bus/pci/rescan` (also `06:00.0`). The front D435 must link at USB 3; a bad USB-C cable drops it to 2.1. |
| `Missing or unavailable CAN interfaces: can0` | Interfaces down after replug | `sudo bash i2rt/scripts/reset_all_can.sh`. The names come from `RAIDEN_CAN_FOLLOWER_L/R` set in `activate_bridge.sh`. |
| `online motors: []` / `No buffer space available` | Arm power off | Power the arms; then reset CAN again to clear the wedged TX queue. |
| Both arms report `loss communication` at once | Watchdog tripped by a GIL stall | Keep heavy work out of the bridge process; boot order is robots first. |
| Server: `LocalEntryNotFoundError` for DINOv3 | HF cache misdirected | Do not set `HF_HOME` in `serve_pen.sh`; weights are in `~/.cache/huggingface/hub`. |
| Server: `unrecognized arguments: --torch-compile-scope` | Flags from the older branch | This branch takes `--torch-compile --torch-compile-mode reduce-overhead`; regime flags are bridge kwargs. |
| Bridge `ModuleNotFoundError` | Not in the venv | `source /home/evan/projects/yam_deploy/activate_bridge.sh` first. |
| Loop `hz=` far below 30 in the bridge log | IK (`ACTION=ee_pose`) or in-process cameras | Use the defaults: `ACTION=joint` and `"type": "stream"` cameras. |
| `[SAFETY] Dangerously large joint delta` then arms go home and the bridge exits | Guard tripped: commanded joint > `MAXD` from measured | Expected behaviour of the guard. If the loop is slow the lag grows; fix the rate rather than raising `MAXD`. |
| `camera server died` at launch | Cameras held by another process | Quit `check_cameras.py` / `view_cameras.py` / collection; see `logs/cameras_*.log`. |
| Arms move to plausibly wrong places | Frame convention, not the model | Check the end-effector frame the bridge uses against the checkpoint's training frame before blaming the policy. |
| Left/right swapped | CAN mapping | Run `gello_software/scripts/identify_sides.py`; update `RAIDEN_CAN_FOLLOWER_L/R` in `activate_bridge.sh`. |

## 8. Local code changes (not upstream)

Both are in `/home/evan/projects/YAM_robot/raiden/raiden/`, uncommitted on the clone:

1. **RealSense capture in the inference server, and a `"stream"` camera type.**
   `policy_server.py` also gained `_open_stream` / `_stream_capture_loop`: frames come from
   the repo's out-of-process camera server (matched by serial via its `meta`), RGB uint8 +
   depth converted with the server's per-camera depth scale. This is the default now. `policy_server.py` gained `_open_realsense`
   and `_realsense_capture_loop` (ported from the standalone `geyan21/raiden` `server.py`) and
   `_open_cameras` / `_camera_loop` dispatch on the camera `type`. Streams are 640x360 colour
   and depth at 30 fps, depth aligned to colour, converted to metres using the device's own
   depth scale (D405 0.1 mm/unit, D435 1 mm/unit). `camera_config.py` no longer imports the
   ZED SDK at module level and accepts `"type": "realsense"`. `cameras/realsense.py` is a
   copy of the standalone backend.
2. **CAN names.** `robot/controller.py` reads `RAIDEN_CAN_FOLLOWER_L/R` and
   `RAIDEN_CAN_LEADER_L/R` from the environment, defaulting to the upstream udev names.

## 9. Known state at the end of 2026-09-09

- **CAN sides re-verified 2026-09-09 (continuous hand test, 70 deg on can0, 0 on can1,
  operator behind the arms): RIGHT arm = can0, LEFT arm = can1.** This contradicts
  CLAUDE.md's 2026-09-08 note (can0 = left). The bridge (`activate_bridge.sh`) uses the
  verified mapping. `gello_software/configs/yam_left.yaml` (can0) / `yam_right.yaml` (can1)
  are therefore crossed for teleop and collection until someone re-runs
  `gello_software/scripts/identify_sides.py` and fixes them; episodes collected since the
  right-arm CAN adapter was replugged (it moved from USB 1-8 to 1-5) may have left/right
  swapped.
- Wrist camera sides verified 2026-09-09 (hand-over-lens test): config is correct.

- Server booted and healthy with step 47000 of the 0904 mm2yam run.
- All three cameras stream. The left D405 is serial **353322270868** (replaced 2026-09-08; the
  old 335122270697 in CLAUDE.md is dead). First bridge launch reached the arms: both arms
  initialised and homed, bridge connected to the server; it aborted only because the camera
  config still had the old left serial.
- This checkpoint was trained before any RealSense data existed from this rig, so the first
  rollout is a domain-gap probe.

## 10. Train/deploy contract audit (2026-09-09)

Done after the first rollouts moved slowly and did not attempt the task. Two code audits of
the September branch plus the first-observation dump the bridge writes to
`~/yam_fastwam_ws_obs_debug/` on every run. Verdicts:

| Item | Verdict | Evidence |
|---|---|---|
| Camera order, per-cam sizes (head 256x320, wrists 224x224), stretch resize | MATCH | server metadata = training `_PER_CAM_HW`; bridge `resize_stretch` |
| Colour order at model input | MATCH | RealSense loop converts BGR->RGB; dump PNGs show natural colours |
| Depth units (uint16 mm), nearest resize, endpoint K rescale, camera-frame pointmap, 2 m gate, no extrinsics | MATCH | bridge `_resize_depth_and_scale_intrinsics` mirrors the anydata loader |
| Depth aligned to colour, colour K used | MATCH | `rs.align(color)` in the ported capture loop |
| 33-frame window | MATCH | frames are the *future*; deploy needs only the present frame, no history buffer |
| 32-D state layout, left arm first, rot6d rows, `grasp_site`, per-arm base | MATCH | dump rest pose 0.247/0.000/0.174 = spec value |
| Grippers 1 = open, no `1 - cmd` flip | MATCH | flip is AgiBot-only; dump reads 0.90 open |
| Relative action anchor = obs-time state; rel->abs inversion | MATCH | `Yam32DRelativeAction.backward` with current state |
| Normalisation stats | MATCH | served file is the run's boot-time v1 merged fit: 41,800 episodes, no `action_source` key, no coverage tail |
| Prompt template, tokenizer length 128 | MATCH | live T5 encode; pen instruction verbatim from the dataset |
| Bridge L/R swap vs this Raiden | MATCH | Raiden still returns `[L,R]` and slices `[R,L]`; the swap is required |
| 30 Hz, horizon 32, RTC blend, smoother/speed adapter off | MATCH | nothing in the default path scales amplitude |
| Next-state action convention | INHERENT | documented 100-256 ms lag, 0.75-0.89x velocity vs demos |
| Head camera FOV | **MISMATCH** | training ZED 2i fx 262 (~101 deg); D435 fx 462 (69 deg) |
| Depth holes | MISMATCH | training ~23-26 % invalid (ZED); ours 7 % head, ~55 % wrists (D405) |
| Eval history of this checkpoint | NONE | no real-robot eval; offline metrics documented as unreliable |

Conclusion: the software contract is satisfied. What remains is sensing domain gap, led by the
head camera's field of view, plus the checkpoint's own untested status. Options, in order:
mount a ZED 2i at the head (exact match); move the D435 up/back to frame the same table area;
or fine-tune on data from this rig. `action_source=model_joint` (bridge kwarg) with
`--action-type joint` bypasses IK entirely and is a useful A/B if motion looks wrong.
