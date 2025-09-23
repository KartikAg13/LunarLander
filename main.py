from typing import override
import numpy as np

import gymnasium as gym


class GridEnv(gym.Env):
    def __init__(self, size: int) -> None:
        super(GridEnv, self).__init__()
        # size of the grid
        self.size = size

        # spawn locations
        self.agent_location = np.array([-1, -1], dtype=np.int32)
        self.target_location = np.array([-1, -1], dtype=np.int32)

        # observation space
        self.surrounding_space = gym.spaces.Box(-1, 1, (4, 0), np.int32)
        self.distance_space = gym.spaces.Box(0, 2 * size, (2, 0), np.int32)
        self.observation_space = gym.spaces.Dict(
            {
                "surroundings": self.surrounding_space,
                "distances": self.distance_space,
            }
        )

        # action space
        self.action_space = gym.spaces.Discrete(4)
        self.direction = {
            0: np.array([-1, 0]),  # left
            1: np.array([0, 1]),  # up
            2: np.array([1, 0]),  # right
            3: np.array([0, -1]),  # down
        }

    def _get_obs(self):
        # returns the current state
        return {"agent": self.agent_location, "target": self.target_location}

    def _get_info(self):
        # returns the distance between snake and fruit
        return {
            "distance": np.linalg.norm(
                self.agent_location - self.target_location, ord=1
            )
        }

    @override
    def reset(self, *, seed: int | None = None):
        # make sure the env is reproducible
        super().reset(seed=seed)

        # spawn the agent
        self.agent_location = self.np_random.integers(
            0, self.size, size=2, dtype=np.int32
        )

        # spawn the target
        self.target_location = self.agent_location
        while np.array_equal(self.agent_location, self.target_location):
            self.target_location = self.np_random.integers(
                0, self.size, size=2, dtype=np.int32
            )

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: int):
        # take the action and update location
        direction = self.direction[action]
        self.agent_location = np.clip(
            a=self.agent_location + direction, a_min=0, a_max=self.size - 1
        )
        terminated = np.array_equal(self.agent_location, self.target_location)

        truncated = False
        reward = 1 if terminated else -0.1

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info


def main():
    # gym.register("gymnasium_env/GridEnv-v0", GridEnv, 500)
