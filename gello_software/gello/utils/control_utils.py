"""Shared utilities for robot control loops."""

import datetime
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tqdm

from gello.agents.agent import Agent
from gello.env import RobotEnv
import logging

from gello.data_utils.data_saver import DataSaver
from gello.data_utils.keyboard_interface import KBReset
from gello.data_utils.data_saver_thread import EpisodeSaverThread
DEFAULT_MAX_JOINT_DELTA = 1.0
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def move_to_start_position(
    env: RobotEnv, agent: Agent, max_delta: float = 1.0, steps: int = 25
) -> bool:
    """Move robot to start position gradually.

    Args:
        env: Robot environment
        agent: Agent that provides target position
        max_delta: Maximum joint delta per step
        steps: Number of steps for gradual movement

    Returns:
        bool: True if successful, False if position too far
    """
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = DEFAULT_MAX_JOINT_DELTA
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return False

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    for _ in range(steps):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    return True


class SaveInterface:
    """Handles keyboard-based data saving interface."""

    def __init__(
        self,
        data_dir: str = "data",
        agent_name: str = "Agent",
        expand_user: bool = False,
    ):
        """Initialize save interface.

        Args:
            data_dir: Base directory for saving data
            agent_name: Name of agent (used for subdirectory)
            expand_user: Whether to expand ~ in data_dir path
        """
        from gello.data_utils.keyboard_interface import KBReset

        self.kb_interface = KBReset()
        self.data_dir = Path(data_dir).expanduser() if expand_user else Path(data_dir)
        self.agent_name = agent_name
        self.save_path: Optional[Path] = None

        print("Save interface enabled. Use keyboard controls:")
        print("  S: Start recording")
        print("  Q: Stop recording")

    def update(self, obs: Dict[str, Any], action: np.ndarray) -> Optional[str]:
        """Update save interface and handle saving.

        Args:
            obs: Current observations
            action: Current action

        Returns:
            Optional[str]: "quit" if user wants to exit, None otherwise
        """
        from gello.data_utils.format_obs import save_frame

        dt = datetime.datetime.now()
        state = self.kb_interface.update()

        if state == "start":
            dt_time = datetime.datetime.now()
            self.save_path = (
                self.data_dir / self.agent_name / dt_time.strftime("%m%d_%H%M%S")
            )
            self.save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving to {self.save_path}")
        elif state == "save":
            if self.save_path is not None:
                save_frame(self.save_path, dt, obs, action)
        elif state == "normal":
            self.save_path = None
        elif state == "quit":
            print("\nExiting.")
            return "quit"
        else:
            raise ValueError(f"Invalid state {state}")

        return None


# teleop
def run_control_loop(
    env: RobotEnv,
    agent: Agent,
    save_interface: Optional[SaveInterface] = None,
    print_timing: bool = True,
    use_colors: bool = False,
) -> None:
    """Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent for control
        save_interface: Optional save interface for data collection
        print_timing: Whether to print timing information
        use_colors: Whether to use colored terminal output
    """
    # Check if we can use colors
    colors_available = False
    if use_colors:
        try:
            from termcolor import colored

            colors_available = True
            start_msg = colored("\nStart 🚀🚀🚀", color="green", attrs=["bold"])
        except ImportError:
            start_msg = "\nStart 🚀🚀🚀"
    else:
        start_msg = "\nStart 🚀🚀🚀"

    print(start_msg)

    start_time = time.time()
    obs = env.get_obs()

    while True:
        if print_timing:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}          "

            if colors_available:
                print(
                    colored(message, color="white", attrs=["bold"]), end="", flush=True
                )
            else:
                print(message, end="", flush=True)

        action = agent.act(obs)

        # Handle save interface
        if save_interface is not None:
            result = save_interface.update(obs, action)
            if result == "quit":
                break

        obs = env.step(action)

# data collection
def run_control_loop_prior(
    env: RobotEnv,
    agent: Agent,
    left_cfg: dict = None,
    right_cfg: Optional[dict] = None,
    print_timing: bool = True,
    use_colors: bool = False,
    data_saver: DataSaver = None,
    kb_interface: KBReset = None,
) -> None:
    """Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent for control
        save_interface: Optional save interface for data collection
        print_timing: Whether to print timing information
        use_colors: Whether to use colored terminal output
    """
    # Check if we can use colors
    colors_available = False
    if use_colors:
        try:
            from termcolor import colored

            colors_available = True
            start_msg = colored("\nStart 🚀🚀🚀", color="green", attrs=["bold"])
        except ImportError:
            start_msg = "\nStart 🚀🚀🚀"
    else:
        start_msg = "\nStart 🚀🚀🚀"

    print(start_msg)

    # for data collection
    saver_thread = EpisodeSaverThread(data_saver)
    saver_thread.start()
    logger = logging.getLogger(__name__)
    num_traj = data_saver.traj_count if data_saver is not None else 1
    logger.info(f"Starting trajectory index at {num_traj}")

    start_time = time.time()
    last_save_time = time.time()
    last_action = ""

    # implemented in env.py to allow dynamic offset during data collection
    env.set_original_offset(agent.act(env.get_obs()))

    def build_dashboard_data(
        obs: Dict[str, Any],
        phase: str,
        status_text: str,
        traj_idx: int,
        total_traj: int,
        step_idx: int = 0,
        max_steps: int = 1,
    ) -> Dict[str, Any]:
        camera_frames = {
            key.replace("_rgb", ""): value
            for key, value in obs.items()
            if key.endswith("_rgb")
        }
        return {
            "phase": phase,
            "status_text": status_text,
            "traj_idx": traj_idx,
            "total_traj": total_traj,
            "step_idx": step_idx,
            "max_steps": max_steps,
            "obs_count": len(data_saver.buffer),
            "joint_positions": obs.get("joint_positions"),
            "joint_velocities": obs.get("joint_velocities"),
            "cameras": camera_frames,
        }

    while num_traj <= left_cfg['storage']['episodes']:
        obs = env.get_obs()
        data_saver.reset_buffer()

        if print_timing:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}     "
            print(message, end="", flush=True)

        logger.info(f"Press Enter to start collecting data: ")
        while True:
            dashboard_data = build_dashboard_data(
                obs=obs,
                phase="waiting_start",
                status_text=("Press Enter to start (3-2-1 countdown)" + (f"   |   last: {last_action}" if last_action else "")),
                traj_idx=num_traj,
                total_traj=left_cfg['storage']['episodes'],
                step_idx=0,
                max_steps=left_cfg['collection']['max_episode_length'],
            )
            result = kb_interface.update(dashboard_data)
            if result == "start":
                logger.info("Enter pressed; 3-2-1 countdown before recording")
                # Three seconds to get both hands on the leaders. The pad shows the
                # count; the terminal too. Arms are not commanded during the count.
                for n in (3, 2, 1):
                    print(f"\rStarting in {n}...   ", end="", flush=True)
                    kb_interface.banner(f"Starting in {n}", dashboard_data, duration_s=1.1, color=(150, 110, 0))
                    time.sleep(1.0)
                print("\rRECORDING            ", flush=True)
                obs = env.get_obs()
                env.set_dynamic_offset(agent.act(obs))
                kb_interface.banner("RECORDING", dashboard_data, duration_s=1.0, color=(0, 120, 60))
                break
        logger.info(f"Press 's' to save the episode, 'd' to discard it")

        result = "normal"
        max_episode_length = left_cfg['collection']['max_episode_length']
        for step_idx in tqdm.tqdm(
            range(max_episode_length),
            desc=f"Collecting data {num_traj}/{left_cfg['storage']['episodes']}",
        ):
            _stale = obs.get("camera_stale") or []
            dashboard_data = build_dashboard_data(
                obs=obs,
                phase="collecting",
                status_text=("!! CAMERA STALE: " + ", ".join(_stale) + " -- consider D (discard)") if _stale
                            else "Collecting... [S] save   [D] discard",
                traj_idx=num_traj,
                total_traj=left_cfg['storage']['episodes'],
                step_idx=step_idx + 1,
                max_steps=max_episode_length,
            )
            result = kb_interface.update(dashboard_data)
            if result == "save" or result == "discard":
                break
            else:
                action = agent.act(obs)
                next_obs = env.step(action)
                obs["next_joint"] = next_obs["joint_positions"]
                data_saver.add_observation(obs)
                obs = next_obs.copy()

        if result == "save":
            if data_saver.buffer == []:
                logger.info(f"No data collected, skipping save")
                continue
            saver_thread.save_episode(data_saver.buffer.copy())
            kb_interface.banner(f"SAVING episode {num_traj}", dashboard_data, duration_s=3.0, color=(0, 120, 60))
            last_action = f"episode {num_traj} SAVED"
            num_traj += 1
            logger.info(f"Successfully collected data")
        else:
            kb_interface.banner(f"DISCARDED episode {num_traj}", dashboard_data, duration_s=3.0, color=(150, 30, 30))
            last_action = f"episode {num_traj} DISCARDED"
            logger.info(f"Episode discarded")
        from gello.utils.launch_utils import move_to_start_position
        if right_cfg is not None:
            move_to_start_position(env, agent, left_cfg=left_cfg, right_cfg=right_cfg)
        else:
            move_to_start_position(env, agent, left_cfg=left_cfg)

    saver_thread.stop()
    saver_thread.join()
    logger.info(f"Finished collecting data")
    return