"""Live 3-pane view of the configured RealSense cameras.

Use this to aim/mount the cameras and to work out which physical camera is
which config role. Panes are labelled with the role from the YAML and the
device serial, in the order the model expects: FRONT (top) | LEFT | RIGHT.

The cameras are exclusive-access, so QUIT THIS before running teleop, data
collection, or the camera server -- otherwise they fail with
"Device or resource busy".

Keys:  q or ESC = quit      s = save a snapshot of the current panes

Usage:
    python scripts/view_cameras.py
    python scripts/view_cameras.py --config configs/yam_left.yaml --width 848 --height 480
"""

import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from omegaconf import OmegaConf

# Order the policy expects: front/top first, then left, then right.
ROLE_ORDER = ["front_camera", "left_camera", "right_camera"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yam_left.yaml")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    cam_cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    cam_cfg = cam_cfg["sensors"]["cameras"]
    present = {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}

    pipes = []
    for role in ROLE_ORDER:
        serial = cam_cfg.get(role, {}).get("device_id")
        if serial is None:
            print(f"{role}: not in config, skipping")
            continue
        if serial not in present:
            print(f"{role} ({serial}): NOT CONNECTED, skipping")
            continue
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        try:
            pipe.start(cfg)
        except Exception as e:
            print(f"{role} ({serial}): failed to start -- {e}")
            continue
        pipes.append((role, serial, pipe))
        print(f"{role} ({serial}): streaming")

    if not pipes:
        print("\nNo cameras could be opened. If they are 'busy', something else "
              "holds them:\n  ps aux | grep camera_server\n  sudo fuser -v /dev/video*")
        return

    print(f"\n{len(pipes)} camera(s) live. Focus the window; q/ESC quits, s saves a snapshot.")
    win = "RealSense: FRONT | LEFT | RIGHT  (q=quit, s=snapshot)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    fps_t, frames, fps = time.time(), 0, 0.0

    try:
        while True:
            panes = []
            for role, serial, pipe in pipes:
                ok, fs = pipe.try_wait_for_frames(1000)
                if not ok or not fs.get_color_frame():
                    img = blank.copy()
                    cv2.putText(img, "NO FRAME", (20, args.height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    img = np.asanyarray(fs.get_color_frame().get_data()).copy()
                label = role.replace("_camera", "").upper()
                cv2.rectangle(img, (0, 0), (args.width, 34), (0, 0, 0), -1)
                cv2.putText(img, f"{label}  {serial}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                panes.append(img)

            grid = np.hstack(panes)
            frames += 1
            if time.time() - fps_t >= 1.0:
                fps, frames, fps_t = frames / (time.time() - fps_t), 0, time.time()
            cv2.putText(grid, f"{fps:.1f} fps", (8, grid.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(win, grid)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                fn = f"/tmp/camera_view_{int(time.time())}.png"
                cv2.imwrite(fn, grid)
                print(f"saved {fn}")
    finally:
        for _, _, pipe in pipes:
            try:
                pipe.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("cameras released")


if __name__ == "__main__":
    main()
