from typing import Any, Dict, Protocol

import numpy as np


class Agent(Protocol):
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Returns an action given an observation.

        Args:
            obs: observation from the environment.

        Returns:
            action: action to take on the environment.
        """
        raise NotImplementedError


class DummyAgent(Agent):
    def __init__(self, num_dofs: int):
        self.num_dofs = num_dofs

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.zeros(self.num_dofs)


class BimanualAgent(Agent):
    def __init__(self, agent_left: Agent, agent_right: Agent):
        self.agent_left = agent_left
        self.agent_right = agent_right

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Split the bimanual observation into per-arm halves and concatenate the actions.

        Only 1-D vectors are per-arm concatenations (joint_positions, joint_velocities,
        ee_pos_quat, gripper_position) and get split. Everything else -- camera images
        (H, W, C), depth maps, per-camera timestamps (floats) -- is shared context and is
        passed through unchanged. Splitting images by rows was never meaningful, and a
        scalar has no shape at all.
        """
        left_obs = {}
        right_obs = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray) and val.ndim == 1:
                L = val.shape[0]
                half_dim = L // 2
                assert L == half_dim * 2, f"{key} must be even, something is wrong"
                left_obs[key] = val[:half_dim]
                right_obs[key] = val[half_dim:]
            else:
                left_obs[key] = val
                right_obs[key] = val
        return np.concatenate(
            [self.agent_left.act(left_obs), self.agent_right.act(right_obs)]
        )
