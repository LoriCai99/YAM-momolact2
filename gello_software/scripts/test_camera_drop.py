"""Live check that the camera server survives one camera dropping mid-stream.

Starts the camera server (same flags the collection launcher uses), consumes the
PUB stream like the control loop does, hardware-resets ONE camera at t=5 s to
simulate a USB drop, and reports: the longest gap between frame sets (the loop
aborts at 3 s), when the camera was flagged stale, and when it was fresh again.

Usage:
    python scripts/test_camera_drop.py --reset-serial 218622275075
    python scripts/test_camera_drop.py --config /path/to/cams.yaml --reset-serial <serial>

Verified 2026-09-08 (right D405): max gap 70 ms, stale 6.0 s -> fresh 7.9 s.
Run this after any camera/cable/port change; a camera that never comes back
fresh, or that enumerates but streams 0 frames, is a hardware problem.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

import numpy as np

from gello.cameras.camera_client import CameraClient, CameraStreamClient


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs/yam_left.yaml"))
    ap.add_argument("--reset-serial", required=True)
    ap.add_argument("--rep", default="tcp://127.0.0.1:5555")
    ap.add_argument("--pub", default="tcp://127.0.0.1:5556")
    ap.add_argument("--seconds", type=float, default=32.0)
    a = ap.parse_args()

    cmd = [sys.executable, "-m", "gello.cameras.camera_server", "--config", os.path.abspath(a.config),
           "--rep-endpoint", a.rep, "--pub-endpoint", a.pub, "--pub-format", "multipart", "--pub-on-new-frame"]
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(120):
            if srv.poll() is not None:
                print(f"FAIL: camera server exited with {srv.returncode} before serving (a camera did not start?)")
                return 1
            try:
                CameraClient(a.rep, request_timeout_ms=500, max_frame_age_sec=None).ping()
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        else:
            print("FAIL: camera server never answered")
            return 1

        c = CameraStreamClient(a.rep, a.pub, request_timeout_ms=500, stream_timeout_sec=3.0)
        gaps, stale_seen = [], set()
        state = {"last_rx": None, "first_stale": None, "recovered": None, "err": None}
        t0 = time.time()

        def consume() -> None:
            while time.time() - t0 < a.seconds:
                try:
                    o = c.get_obs_full()
                except Exception as e:  # noqa: BLE001
                    state["err"] = f"{time.time() - t0:.1f}s: {e}"
                    return
                rx = c._latest_rx
                if state["last_rx"] is not None and rx != state["last_rx"]:
                    gaps.append(rx - state["last_rx"])
                state["last_rx"] = rx
                if o["stale"]:
                    stale_seen.update(o["stale"])
                    if state["first_stale"] is None:
                        state["first_stale"] = time.time() - t0
                elif state["first_stale"] is not None and state["recovered"] is None:
                    state["recovered"] = time.time() - t0
                time.sleep(1 / 30)

        th = threading.Thread(target=consume, daemon=True)
        th.start()
        time.sleep(5)
        print(f"t=5s: hardware_reset() of {a.reset_serial}")
        subprocess.Popen([sys.executable, "-c",
                          "import pyrealsense2 as rs\nfor d in rs.context().query_devices():\n"
                          f"    if d.get_info(rs.camera_info.serial_number)=='{a.reset_serial}': d.hardware_reset()"])
        th.join(timeout=a.seconds + 10)
        g = np.array(gaps) * 1e3 if gaps else np.array([0.0])
        print(f"frame-set gaps: p50 {np.percentile(g, 50):.0f} ms  p99 {np.percentile(g, 99):.0f}  MAX {g.max():.0f} ms  (loop aborts at 3000)")
        print(f"client error: {state['err']}")
        print(f"stale seen: {sorted(stale_seen)} | first stale {state['first_stale'] and round(state['first_stale'], 1)}s | all fresh again {state['recovered'] and round(state['recovered'], 1)}s")
        ok = state["err"] is None and state["recovered"] is not None and g.max() < 1000
        print("PASS" if ok else "FAIL")
        c.close()
        return 0 if ok else 1
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:  # noqa: BLE001
            srv.kill()


if __name__ == "__main__":
    sys.exit(main())
