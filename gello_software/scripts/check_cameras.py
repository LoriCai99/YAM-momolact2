"""One-command readiness check for the three RealSense cameras.

Answers the only question that matters before data collection: can every
configured camera deliver the exact streams gello/cameras/realsense_camera.py
asks for (color + depth, 640x360 @30)?

A camera on a USB 2 link enumerates fine and streams at other resolutions, but
the D435 does NOT offer color 640x360 at USB 2.1 -- collection then dies with
"RuntimeError: Couldn't resolve requests". This tells you that before you set
up a session instead of after.

Run it after every physical replug.

Usage:  python scripts/check_cameras.py
"""

import sys
import time

import pyrealsense2 as rs
from omegaconf import OmegaConf

REQ_W, REQ_H, REQ_FPS = 640, 360, 30


def main() -> None:
    cfg = OmegaConf.to_container(OmegaConf.load("configs/yam_left.yaml"), resolve=True)
    cams = cfg["sensors"]["cameras"]
    # Enumeration can fail transiently for a few seconds right after another
    # process released the cameras (librealsense re-enumerates them); retry
    # rather than crash without a verdict.
    devs = {}
    last_err = None
    for attempt in range(6):
        try:
            devs = {d.get_info(rs.camera_info.serial_number): d for d in rs.context().query_devices()}
            if len(devs) >= len(cams):
                break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(1.0)
    if last_err is not None and not devs:
        print(f"could not enumerate RealSense devices after retries: {last_err}")

    print(f"Required by realsense_camera.py: color+depth {REQ_W}x{REQ_H} @{REQ_FPS}\n")
    print(f"{'role':14}{'serial':16}{'USB':6}{'color':>8}{'depth':>8}   verdict")
    print("-" * 64)

    ready = True
    for role, entry in cams.items():
        sn = entry["device_id"]
        dev = devs.get(sn)
        if dev is None:
            print(f"{role:14}{sn:16}{'--':6}{'--':>8}{'--':>8}   NOT CONNECTED")
            ready = False
            continue

        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        col = dep = False
        for sensor in dev.query_sensors():
            for p in sensor.get_stream_profiles():
                try:
                    v = p.as_video_stream_profile()
                except Exception:
                    continue
                if (v.width(), v.height(), p.fps()) != (REQ_W, REQ_H, REQ_FPS):
                    continue
                if p.stream_type() == rs.stream.color and p.format() == rs.format.bgr8:
                    col = True
                if p.stream_type() == rs.stream.depth and p.format() == rs.format.z16:
                    dep = True

        ok = col and dep
        ready &= ok
        verdict = "ok" if ok else ("USB2 LINK -- move to a USB 3 port"
                                   if usb.startswith("2") else "mode unsupported")
        print(f"{role:14}{sn:16}{usb:6}{'YES' if col else 'NO':>8}{'YES' if dep else 'NO':>8}   {verdict}")

    print("-" * 64)
    if ready:
        print("READY -- data collection can start.")
    else:
        print("NOT READY -- collection will fail with 'Couldn't resolve requests'.")
        print("\nA camera showing USB 2.x must move to a USB 3 socket. Note that not")
        print("every port on a USB 3 controller is wired for USB 3: swap it into a")
        print("socket where another camera already reports 3.x, and re-run this.")
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
