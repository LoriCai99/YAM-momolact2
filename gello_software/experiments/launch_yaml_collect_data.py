import atexit
from math import inf
from multiprocessing import Process
import signal
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import tyro
import zmq.error
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict, move_to_start_position
from gello.dynamixel.driver import DynamixelDriver
import numpy as np

from gello.cameras.realsense_camera import (
    STREAM_FPS,
    STREAM_HEIGHT,
    STREAM_WIDTH,
    RealSenseCamera,
    check_stream_support,
    get_device_ids,
)
from gello.data_utils.data_saver import DataSaver
from gello.data_utils.keyboard_interface import KBReset
from gello.utils.control_utils import run_control_loop_prior
from gello.zmq_core.camera_node import ZMQClientCamera, ZMQServerCamera

# Global variables for cleanup
active_threads = []
active_servers = []
cleanup_in_progress = False

_env = None
_bimanual = False
_left_cfg = None
_right_cfg = None
_agent = None
_robot = None
_robot_client = None
_cameras = None
_data_saver = None
_kb_interface = None


def _call_cleanup_methods(resource, resource_name: str, methods: list[str]) -> None:
    """Best-effort cleanup helper for heterogeneous resources."""
    if resource is None:
        return
    for method_name in methods:
        if hasattr(resource, method_name):
            try:
                getattr(resource, method_name)()
            except Exception as e:
                print(f"Error calling {resource_name}.{method_name}(): {e}")
            return


CAMERA_ROLES = ["left_camera", "front_camera", "right_camera"]


def _preflight_cameras(camera_cfg: dict) -> None:
    """Exit with a clear table if any configured camera cannot serve the required streams.

    Runs BEFORE the output-directory prompt and the GELLO ports are opened. A camera
    on a USB 2 link enumerates fine but cannot serve colour 640x360, and
    pipeline.start() would later die with the opaque "Couldn't resolve requests".
    """
    roles = CAMERA_ROLES
    report = {r: check_stream_support(camera_cfg[r]["device_id"]) for r in roles}
    bad = {r: v for r, v in report.items() if not v["ok"]}
    print(f"Camera pre-flight (need colour+depth {STREAM_WIDTH}x{STREAM_HEIGHT}@{STREAM_FPS}):")
    for r in roles:
        v = report[r]
        print(f"  {r:13} {camera_cfg[r]['device_id']}  USB {v['usb'] or '--':4}  {'OK' if v['ok'] else 'FAIL: ' + v['reason']}")
    if bad:
        print("\nRefusing to start: the camera(s) above cannot deliver the required streams.")
        print("This is a USB-link / hardware condition, not a config error. Diagnose with:")
        print("    python scripts/check_cameras.py")
        sys.exit(2)


def _open_cameras(camera_cfg: dict) -> dict:
    """Open all cameras; if one fails, close the ones already streaming so
    librealsense does not abort at interpreter exit."""
    roles = CAMERA_ROLES
    opened: dict = {}
    try:
        for r in roles:
            opened[r] = RealSenseCamera(camera_cfg[r]["device_id"])
    except Exception as exc:
        for name, cam in opened.items():
            _close_realsense_camera(cam, name)
        raise RuntimeError(f"failed to open {r} ({camera_cfg[r]['device_id']}): {exc}") from exc
    return opened


def _close_realsense_camera(camera, camera_name: str) -> None:
    """Stop RealSense capture thread/pipeline if camera has no public close()."""
    if camera is None:
        return
    if hasattr(camera, "close"):
        _call_cleanup_methods(camera, camera_name, ["close"])
        return
    try:
        if hasattr(camera, "_stop_event"):
            camera._stop_event.set()
        if hasattr(camera, "_capture_thread") and camera._capture_thread is not None:
            camera._capture_thread.join(timeout=2)
        if hasattr(camera, "_pipeline") and camera._pipeline is not None:
            camera._pipeline.stop()
    except Exception as e:
        print(f"Error closing camera {camera_name}: {e}")


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    global _env, _agent, _robot, _robot_client, _cameras, _data_saver, _kb_interface
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("Cleaning up resources...")
    try:
        if _env is not None and _left_cfg is not None:
            if _bimanual:
                move_to_start_position(_env, _bimanual, _left_cfg, _right_cfg)
            else:
                move_to_start_position(_env, _bimanual, _left_cfg)
    except Exception as e:
        print(f"Warning: failed to move robot to start position during cleanup: {e}")

    # Stop server loops first so background threads can exit.
    for server in active_servers:
        try:
            if hasattr(server, "stop"):
                server.stop()
        except Exception as e:
            print(f"Error stopping server: {e}")

    for server in active_servers:
        try:
            if hasattr(server, "close"):
                server.close()
        except Exception as e:
            print(f"Error closing server: {e}")

    for thread in active_threads:
        if thread.is_alive():
            thread.join(timeout=5)

    _call_cleanup_methods(_robot_client, "robot_client", ["close", "stop", "shutdown"])
    _call_cleanup_methods(_robot, "robot", ["close", "stop", "shutdown"])
    _call_cleanup_methods(_agent, "agent", ["close", "stop", "shutdown"])
    _call_cleanup_methods(_env, "env", ["close", "stop", "shutdown"])
    _call_cleanup_methods(_data_saver, "data_saver", ["close", "stop", "shutdown"])

    if isinstance(_cameras, dict):
        for camera_name, camera in _cameras.items():
            _close_realsense_camera(camera, camera_name)

    if _kb_interface is not None:
        _call_cleanup_methods(_kb_interface, "kb_interface", ["close", "stop", "shutdown"])
        try:
            import pygame

            pygame.quit()
        except Exception as e:
            print(f"Error quitting pygame: {e}")

    active_servers.clear()
    active_threads.clear()
    _robot_client = None
    _robot = None
    _agent = None
    _env = None
    _cameras = None
    _data_saver = None
    _kb_interface = None

    print("Cleanup completed.")


def wait_for_server_ready(port, host="127.0.0.1", timeout_seconds=5):
    """Wait for ZMQ server to be ready with retry logic."""
    from gello.zmq_core.robot_node import ZMQClientRobot

    attempts = int(timeout_seconds * 10)  # 0.1s intervals
    for attempt in range(attempts):
        try:
            client = ZMQClientRobot(port=port, host=host)
            time.sleep(0.1)
            return True
        except (zmq.error.ZMQError, Exception):
            time.sleep(0.1)
        finally:
            if "client" in locals():
                client.close()
            time.sleep(0.1)
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"Server failed to start on {host}:{port} within {timeout_seconds} seconds"
                )
    return False


@dataclass
class Args:
    left_config_path: str
    """Path to the left arm configuration YAML file."""

    right_config_path: Optional[str] = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    # use_save_interface: bool = False
    # """Enable saving data with keyboard interface."""


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    cleanup()
    import os

    os._exit(0)

def get_joint_offsets(
    cfg: dict, port: str
):
    """Get joint offsets using the same logic as gello_get_offset.py."""
    joint_ids = list(cfg["agent"]["dynamixel_config"]["joint_ids"])
    driver = DynamixelDriver(joint_ids, port=port, baudrate=57600)

    def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
        joint_sign_i = cfg["agent"]["dynamixel_config"]["joint_signs"][index]
        joint_i = joint_sign_i * (joint_state[index] - offset)
        start_i = cfg["agent"]["start_joints"][index]
        return np.abs(joint_i - start_i)

    # Warmup
    for _ in range(10):
        driver.get_joints()

    best_offsets = []
    curr_joints = driver.get_joints()

    for i in range(len(joint_ids)):
        best_offset = 0
        best_error = float('inf')
        for offset in np.linspace(-8 * np.pi, 8 * np.pi, 500):
            error = get_error(offset, i, curr_joints)
            if error < best_error:
                best_error = error
                best_offset = offset
        best_offsets.append(best_offset)

    driver.close()
    return best_offsets

def update_offsets(cfg):
    joint_offsets = get_joint_offsets(cfg, cfg["agent"]["port"])
    cfg["agent"]["dynamixel_config"]["joint_offsets"] = joint_offsets
    return cfg


def run_post_collection_pipeline(cfg: dict) -> None:
    """Run optional post-collection conversion/upload/tag pipeline."""
    storage_cfg = cfg.get("storage", {})
    lerobot_cfg = cfg.get("lerobot", {})
    auto_convert = bool(lerobot_cfg.get("auto_convert", False))
    auto_upload = bool(lerobot_cfg.get("auto_upload", False))
    if not auto_convert and not auto_upload:
        return
    if auto_upload and not auto_convert:
        print(
            "Skipping post-collection upload because lerobot.auto_convert is false. "
            "Enable lerobot.auto_convert to run conversion+upload pipeline."
        )
        return

    base_dir = Path(storage_cfg["base_dir"]).expanduser()
    task_directory = storage_cfg["task_directory"]
    json_data_dir = base_dir / task_directory
    lerobot_dir = base_dir / f"{task_directory}_lerobot_v30"
    repo_id = lerobot_cfg.get("hf_repo_id", storage_cfg.get("hf_repo_id"))
    if auto_upload and not repo_id:
        raise ValueError(
            "lerobot.hf_repo_id is required when lerobot.auto_upload is true."
        )

    converter_script = Path(__file__).resolve().parents[2] / "molmoact_to_lerobot_v30.py"
    if not converter_script.exists():
        raise FileNotFoundError(f"Converter script not found: {converter_script}")
    if not json_data_dir.exists():
        raise FileNotFoundError(f"Collected json directory not found: {json_data_dir}")
    if lerobot_dir.exists():
        remove_dir = input(
            f"The LeRobot output directory {lerobot_dir} already exists. "
            "Do you want to remove it and continue? (y/n): "
        ).strip().lower()
        if remove_dir == "y":
            shutil.rmtree(lerobot_dir)
            lerobot_dir.mkdir(parents=True, exist_ok=True)
            print(f"Removed and recreated output directory: {lerobot_dir}")
        elif remove_dir == "n":
            print("Conversion canceled by user because output directory already exists.")
            return
        else:
            print("Invalid input. Conversion canceled.")
            return

    convert_cmd = [
        sys.executable,
        str(converter_script),
        "--data_dir",
        str(json_data_dir),
        "--output_dir",
        str(lerobot_dir),
        "--repo_id",
        str(repo_id or "molmoact_v30"),
        "--fps",
        str(lerobot_cfg.get("fps", storage_cfg.get("lerobot_fps", cfg.get("hz", 30)))),
        "--robot_type",
        str(
            lerobot_cfg.get(
                "robot_type", storage_cfg.get("lerobot_robot_type", "molmoact_dual_arm")
            )
        ),
        "--skip_initial_frames",
        str(lerobot_cfg.get("skip_initial_frames", storage_cfg.get("lerobot_skip_initial_frames", 0))),
        "--action_mode",
        str(
            lerobot_cfg.get(
                "action_mode", storage_cfg.get("lerobot_action_mode", "next_joint_fields")
            )
        ),
        "--task_instruction",
        str(storage_cfg.get("language_instruction", "perform the task")),
        "--sanitize_online_viz_meta",
        str(
            int(
                bool(
                    lerobot_cfg.get(
                        "sanitize_online_viz_meta",
                        storage_cfg.get("sanitize_online_viz_meta", True),
                    )
                )
            )
        ),
        "--vcodec",
        str(lerobot_cfg.get("vcodec", "h264")),
        "--image_writer_processes",
        str(int(lerobot_cfg.get("image_writer_processes", 8))),
        "--image_writer_threads",
        str(int(lerobot_cfg.get("image_writer_threads", 8))),
        "--parallel_encoding",
        str(int(bool(lerobot_cfg.get("parallel_encoding", True)))),
        "--upload_to_hf",
        str(int(auto_upload)),
        "--delete_local_after_upload",
        str(
            int(
                bool(
                    lerobot_cfg.get(
                        "delete_local_after_upload",
                        storage_cfg.get("delete_local_after_upload", True),
                    )
                )
            )
        ),
    ]
    print(f"Running post-collection pipeline: {' '.join(convert_cmd)}")
    subprocess.run(convert_cmd, check=True)

    print("Post-collection pipeline completed successfully.")

def main():
    global _env, _bimanual, _left_cfg, _right_cfg
    global _agent, _robot, _robot_client, _cameras, _data_saver, _kb_interface
    # Register cleanup handlers
    # If terminated without cleanup, can leave ZMQ sockets bound causing "address in use" errors or resource leaks

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = tyro.cli(Args)

    # left, right front camera (the device id order is based on the plugged in order on the adapter)
    ids = get_device_ids()
    print(f"Found {len(ids)} camera devices")
    print(ids)

    bimanual = args.right_config_path is not None

    # Load configs
    left_cfg = OmegaConf.to_container(
        OmegaConf.load(args.left_config_path), resolve=True
    )
    # Fail fast on cameras BEFORE touching the GELLO ports, prompting about the
    # output dir, or energizing anything.
    _preflight_cameras(left_cfg["sensors"]["cameras"])
    left_cfg = update_offsets(left_cfg)
    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )
        right_cfg = update_offsets(right_cfg)

    # Initialize data saver and keyboard interface
    storage_cfg = left_cfg["storage"]
    data_saver = DataSaver(
        save_dir=storage_cfg["base_dir"],
        task_directory=storage_cfg["task_directory"],
        language_instruction=storage_cfg["language_instruction"],
        saver_max_workers=storage_cfg.get("saver_max_workers"),
        png_compress_level=storage_cfg.get("png_compress_level", 1),
        # flex-pi needs synchronised depth; the old saver silently dropped it.
        save_depth=storage_cfg.get("save_depth", True),
        image_format=storage_cfg.get("image_format", "jpg"),
        jpeg_quality=storage_cfg.get("jpeg_quality", 95),
        fps=left_cfg.get("hz", 30),
        camera_roles=(left_cfg.get("flexpi") or {}).get("camera_map"),
    )
    kb_interface = KBReset()

    camera_cfg = left_cfg["sensors"]["cameras"]
    cameras = _open_cameras(camera_cfg)
    # Register for cleanup NOW: if robot construction below raises (e.g. a motor
    # not answering), the atexit handler must still stop the capture threads, or
    # librealsense aborts/segfaults at interpreter exit.
    _cameras = cameras

    # Create agent
    if bimanual:
        from gello.agents.agent import BimanualAgent

        agent = BimanualAgent(
            agent_left=instantiate_from_dict(left_cfg["agent"]),
            agent_right=instantiate_from_dict(right_cfg["agent"]),
        )
    else:
        agent = instantiate_from_dict(left_cfg["agent"])

    # Create robot(s)
    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )

    left_robot = instantiate_from_dict(left_robot_cfg)

    if bimanual:
        from gello.robots.robot import BimanualRobot

        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )

        right_robot = instantiate_from_dict(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)

        # For bimanual, use the left config for general settings (hz, etc.)
        cfg = left_cfg
    else:
        robot = left_robot
        cfg = left_cfg

    # Handle different robot types
    if hasattr(robot, "serve"):  # MujocoRobotServer or ZMQServerRobot
        print("Starting robot server...")
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot

        # Get server configuration
        server_port = cfg["robot"].get("port", 5556)
        server_host = cfg["robot"].get("host", "127.0.0.1")

        # Start server in background (non-daemon for proper cleanup)
        server_thread = threading.Thread(target=robot.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(robot)

        # Wait for server to be ready
        print(f"Waiting for server to start on {server_host}:{server_port}...")
        wait_for_server_ready(server_port, server_host)
        print("Server ready!")

        # Create client to communicate with server using port and host from config
        robot_client = ZMQClientRobot(port=server_port, host=server_host)
    else:  # Direct robot (hardware)
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot

        # Get server configuration (use a different default port for hardware)
        hardware_port = cfg.get("hardware_server_port", 6001)
        hardware_host = "127.0.0.1"

        # Create ZMQ server for the hardware robot
        server = ZMQServerRobot(robot, port=hardware_port, host=hardware_host)
        server_thread = threading.Thread(target=server.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(server)

        # Wait for server to be ready
        print(
            f"Waiting for hardware server to start on {hardware_host}:{hardware_port}..."
        )
        wait_for_server_ready(hardware_port, hardware_host)
        print("Hardware server ready!")

        # Create client to communicate with hardware
        robot_client = ZMQClientRobot(port=hardware_port, host=hardware_host)

    env = RobotEnv(robot_client, control_rate_hz=cfg.get("hz", 30), camera_dict=cameras)
    # Intrinsics + depth scale per camera go into every episode's meta.json.
    data_saver.set_camera_meta(env.get_camera_meta())
    for _cam, _m in env.get_camera_meta().items():
        _i = _m.get("intrinsics") or {}
        print(
            f"camera {_cam}: serial={_m.get('device_id')} "
            f"{_i.get('width')}x{_i.get('height')} fx={_i.get('fx', float('nan')):.1f} "
            f"depth_scale={_m.get('depth_scale_m_per_unit')} m/unit"
        )

    # Store global variables for cleanup
    _env = env
    _bimanual = bimanual
    _left_cfg = left_cfg
    _right_cfg = right_cfg if bimanual else None
    _agent = agent
    _robot = robot
    _robot_client = robot_client
    _cameras = cameras
    _data_saver = data_saver
    _kb_interface = kb_interface

    # Move robot to start_joints position if specified in config
    from gello.utils.launch_utils import move_to_start_position

    if bimanual:
        move_to_start_position(env, bimanual, left_cfg, right_cfg)
    else:
        move_to_start_position(env, bimanual, left_cfg)

    print(
        f"Launching robot: {robot.__class__.__name__}, agent: {agent.__class__.__name__}"
    )
    print(f"Control loop: {cfg.get('hz', 30)} Hz")

    # from gello.utils.control_utils import SaveInterface, run_control_loop

    # Initialize save interface if requested
    # save_interface = None
    # if args.use_save_interface:
    #     save_interface = SaveInterface(
    #         data_dir=Path(args.left_config_path).parents[1] / "data",
    #         agent_name=agent.__class__.__name__,
    #         expand_user=True,
    #     )

    # # Run main control loop
    # run_control_loop(env, agent, save_interface)

    # Run main control loop
    if bimanual:
        run_control_loop_prior(env, agent, left_cfg=left_cfg, right_cfg=right_cfg, data_saver=data_saver, kb_interface=kb_interface)
    else:
        run_control_loop_prior(env, agent, left_cfg=left_cfg, data_saver=data_saver, kb_interface=kb_interface)

    cleanup()
    run_post_collection_pipeline(left_cfg)
    print("All tasks completed.")


if __name__ == "__main__":
    main()
