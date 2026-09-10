import atexit
from math import inf
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pygame import QUIT
import tyro
import zmq.error
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict, move_to_start_position
from gello.dynamixel.driver import DynamixelDriver
import numpy as np

from gello.cameras.realsense_camera import RealSenseCamera
from gello.data_utils.data_replay import DataReplayer
import logging

# Global variables for cleanup
active_threads = []
active_servers = []
cleanup_in_progress = False

_env = None
_bimanual = False
_left_cfg = None
_right_cfg = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("Cleaning up resources...")
    if _bimanual:
        move_to_start_position(_env, _bimanual, _left_cfg, _right_cfg)
    else:
        move_to_start_position(_env, _bimanual, _left_cfg)
    for server in active_servers:
        try:
            if hasattr(server, "close"):
                server.close()
        except Exception as e:
            print(f"Error closing server: {e}")

    for thread in active_threads:
        if thread.is_alive():
            thread.join(timeout=2)

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

    realtime: bool = False
    """Send one recorded frame per control tick (exact teleop timing) instead of the
    velocity-limited default (~half speed). The approach to frame 0 is always interpolated."""

    # use_save_interface: bool = False
    # """Enable saving data with keyboard interface."""


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    cleanup()
    import os

    os._exit(0)

def main():
    # Register cleanup handlers
    # If terminated without cleanup, can leave ZMQ sockets bound causing "address in use" errors or resource leaks

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = tyro.cli(Args)

    bimanual = args.right_config_path is not None

    # Load configs
    left_cfg = OmegaConf.to_container(
        OmegaConf.load(args.left_config_path), resolve=True
    )
    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )

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

    env = RobotEnv(robot_client, control_rate_hz=cfg.get("hz", 30))

    # Store global variables for cleanup
    # Store global variables for cleanup
    global _env, _bimanual, _left_cfg, _right_cfg
    _env = env
    _bimanual = bimanual
    _left_cfg = left_cfg
    _right_cfg = right_cfg if bimanual else None

    # Move robot to start_joints position if specified in config
    from gello.utils.launch_utils import move_to_start_position

    if bimanual:
        move_to_start_position(env, bimanual, left_cfg, right_cfg)
    else:
        move_to_start_position(env, bimanual, left_cfg)

    print(
        f"Launching robot: {robot.__class__.__name__}"
    )

    logger.info("Start replay...")
    task = input("Enter task to replay: ")
    episode_number = int(input("Enter episode number to replay: "))
    robot_trajectory = input("Replay robot trajectory? (y/n): ")
    if robot_trajectory == 'y':
        robot_trajectory = True
    else:
        robot_trajectory = False
    camera_trajectory = input("Replay camera trajectory? (y/n): ")
    if camera_trajectory == 'y':
        camera_trajectory = True
    else:
        camera_trajectory = False

    data_replayer = DataReplayer(save_format=left_cfg['storage']['save_format'], old_format=left_cfg['storage']['old_format'])
    data_replayer.load_episode(left_cfg['storage']['base_dir'] + '/' + task, episode_number, load_images=camera_trajectory)
    data_replayer.replay(env, visual=camera_trajectory, robot_trajectory=robot_trajectory, realtime=args.realtime)

    cleanup()

if __name__ == "__main__":
    main()
